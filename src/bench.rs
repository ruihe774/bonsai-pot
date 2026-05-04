//! Bench / microbench helpers, exposed only with the `bench-internals`
//! feature. Not part of the stable public API.
//!
//! Included as a child module of `forward` via `#[path]` so the helpers can
//! reach `forward`'s private items via `super::`.

use std::cmp::Ordering;
use std::time::Instant;

use wgpu::PollType;

use super::{
    ATTN_CHUNK_SIZE, AttnMergeParams, AttnSplitParams, Config, EmbedParams,
    MatvecFusedNormedParams, MatvecSiluParams, Model, Result, StepEncoder, TopKParams, WeightSet,
    kv_layer_offsets, matvec_bg, matvec_fused_normed_bg, prefill_matmul_topk,
    step_matvec_no_sample, step_matvec_topk,
};
use crate::error::PotError;
use crate::model::{KvWritebackFusedParams, MatvecParams, QNormRopeFusedParams, RmsNormParams};

pub fn bench(model: &Model, pp_n: u32, tg_n: u32, repeats: u32) -> Result<()> {
    let cfg = &model.cfg;
    eprintln!("--- bench: pp={pp_n}, tg={tg_n}, repeats={repeats} (after 1 warmup) ---");

    let prompt: Vec<u32> = (0..pp_n).map(|i| (i % (cfg.n_vocab - 1)) + 1).collect();

    // ----- warm up -----
    let _ = prefill_matmul_topk(model, &prompt[..pp_n.min(model.m_max) as usize], 0, 1)?;
    step_matvec_no_sample(model, 1u32, 0);
    if let Err(e) = model.device.poll(PollType::wait_indefinitely()) {
        model.check_device()?;
        return Err(PotError::Poll(e));
    }

    // ----- pp{pp_n} -----
    let mut pp_times = Vec::with_capacity(repeats as usize);
    for _ in 0..repeats {
        let t = Instant::now();
        let _ = prefill_matmul_topk(model, &prompt, 0, 1)?;
        if let Err(e) = model.device.poll(PollType::wait_indefinitely()) {
            model.check_device()?;
            return Err(PotError::Poll(e));
        }
        pp_times.push(t.elapsed().as_secs_f32());
    }
    let pp_mean = pp_times.iter().sum::<f32>() / pp_times.len() as f32;
    let pp_std = stddev(&pp_times, pp_mean);
    let pp_t_s_mean = pp_n as f32 / pp_mean;
    let pp_t_s_std = pp_t_s_mean * (pp_std / pp_mean);

    // ----- tg{tg_n} (from empty KV) -----
    let mut tg_times = Vec::with_capacity(repeats as usize);
    for _ in 0..repeats {
        let t = Instant::now();
        for pos in 0..tg_n {
            let (_, _) = step_matvec_topk(model, 1u32, pos, 1)?;
        }
        if let Err(e) = model.device.poll(PollType::wait_indefinitely()) {
            model.check_device()?;
            return Err(PotError::Poll(e));
        }
        tg_times.push(t.elapsed().as_secs_f32());
    }
    let tg_mean = tg_times.iter().sum::<f32>() / tg_times.len() as f32;
    let tg_std = stddev(&tg_times, tg_mean);
    let tg_t_s_mean = tg_n as f32 / tg_mean;
    let tg_t_s_std = tg_t_s_mean * (tg_std / tg_mean);

    println!();
    println!("| backend            |          test |               t/s |");
    println!("| ------------------ | ------------- | ----------------: |");
    println!("| bonsai-pot        |        pp{pp_n:<3} | {pp_t_s_mean:>9.2} ± {pp_t_s_std:>5.2} |");
    println!("| bonsai-pot        |        tg{tg_n:<3} | {tg_t_s_mean:>9.2} ± {tg_t_s_std:>5.2} |");
    Ok(())
}

fn stddev(xs: &[f32], mean: f32) -> f32 {
    if xs.len() < 2 {
        return 0.0;
    }
    let var: f32 = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / (xs.len() - 1) as f32;
    var.sqrt()
}

/// Shared helper for the two `matvec_q1_0_fused_normed` entries (QKV and
/// gate+up). Kept as a helper because both call sites pass a different range
/// list but otherwise identical setup.
fn bench_fused_normed(
    model: &Model,
    cfg: &Config,
    se: &mut StepEncoder,
    k: u32,
    input_offset: u32,
    w_norm_off: u32,
    weights: WeightSet,
    ranges: &[(u32, u32, u32, u32)],
) {
    const ROWS_PER_WG: u32 = 8;
    let r = |i: usize| ranges.get(i).copied().unwrap_or((0, 0, 0, 0));
    let (d0, qs0, n0, o0) = r(0);
    let (d1, qs1, n1, o1) = r(1);
    let (d2, qs2, n2, o2) = r(2);
    let n_total = n0 + n1 + n2;
    let n_wg = n_total.div_ceil(ROWS_PER_WG);
    let dispatch_x = n_wg.min(65535);
    let dispatch_y = n_wg.div_ceil(dispatch_x);
    let p = MatvecFusedNormedParams {
        k,
        n_total,
        input_offset,
        dispatch_x_dim: dispatch_x,
        w_norm_off,
        eps: cfg.rms_eps,
        d_offset_0: d0,
        qs_offset_0: qs0,
        n_0: n0,
        output_offset_0: o0,
        d_offset_1: d1,
        qs_offset_1: qs1,
        n_1: n1,
        output_offset_1: o1,
        d_offset_2: d2,
        qs_offset_2: qs2,
        n_2: n2,
        output_offset_2: o2,
    };
    let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("matvec_fused_normed"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&model.pipes.matvec_fused_normed);
    cp.set_bind_group(0, matvec_fused_normed_bg(model, weights), &[]);
    cp.set_immediates(0, bytemuck::bytes_of(&p));
    cp.dispatch_workgroups(dispatch_x, dispatch_y, 1);
}

pub fn microbench_tg(model: &Model, repeats: u32) -> Result<()> {
    type DispatchFn<'a> = Box<dyn Fn(&Model, &Config, &mut StepEncoder) + 'a>;

    let cfg = &model.cfg;
    let n_per_cb: u32 = 200;
    let attn_pos: u32 = 64;

    eprintln!("--- microbench tg (N={n_per_cb}/CB, repeats={repeats}) ---");

    let lt0 = &model.layer_tensors[0];
    let ot = &model.output_tensors;
    let n_layer = cfg.n_layer;

    // Each entry is (label, calls/step, dispatch_closure). The closure encodes
    // a single dispatch into `se`; the breakdown loop below times it in
    // isolation by submitting `n_per_cb` copies in one CB.
    let mut entries: Vec<(String, u32, DispatchFn)> = Vec::new();

    // Wo: plain matvec_q1_0.
    entries.push((
        format!("matvec Wo K={} N={}", cfg.q_dim, cfg.n_embd),
        n_layer,
        Box::new(|model, cfg, se| {
            const ROWS_PER_WG: u32 = 8;
            let lt = &model.layer_tensors[0];
            let (k, n) = (cfg.q_dim, cfg.n_embd);
            let n_wg = n.div_ceil(ROWS_PER_WG);
            let dispatch_x = n_wg.min(65535);
            let dispatch_y = n_wg.div_ceil(dispatch_x);
            let p = MatvecParams {
                k,
                n,
                d_offset: lt.wo.0,
                qs_offset: lt.wo.1,
                input_offset: model.act_layout.x_norm,
                output_offset: model.act_layout.x,
                accumulate: 0,
                dispatch_x_dim: dispatch_x,
            };
            let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matvec"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&model.pipes.matvec);
            cp.set_bind_group(0, matvec_bg(model, WeightSet::Attn), &[]);
            cp.set_immediates(0, bytemuck::bytes_of(&p));
            cp.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }),
    ));

    // LM head: plain matvec_q1_0 over the embed buffer.
    entries.push((
        format!("matvec LM_head K={} N={}", cfg.n_embd, cfg.n_vocab),
        1,
        Box::new(|model, cfg, se| {
            const ROWS_PER_WG: u32 = 8;
            let ot = &model.output_tensors;
            let (k, n) = (cfg.n_embd, cfg.n_vocab);
            let n_wg = n.div_ceil(ROWS_PER_WG);
            let dispatch_x = n_wg.min(65535);
            let dispatch_y = n_wg.div_ceil(dispatch_x);
            let p = MatvecParams {
                k,
                n,
                d_offset: ot.lm_head_d,
                qs_offset: ot.lm_head_qs,
                input_offset: model.act_layout.x_norm,
                output_offset: model.act_layout.x,
                accumulate: 0,
                dispatch_x_dim: dispatch_x,
            };
            let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matvec"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&model.pipes.matvec);
            cp.set_bind_group(0, matvec_bg(model, WeightSet::Embed), &[]);
            cp.set_immediates(0, bytemuck::bytes_of(&p));
            cp.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }),
    ));

    entries.push((
        format!(
            "matvec QKV fused+normed N={}",
            cfg.q_dim + cfg.kv_dim + cfg.kv_dim
        ),
        n_layer,
        Box::new(|model, cfg, se| {
            let lt = &model.layer_tensors[0];
            let ranges = [
                (lt.wq.0, lt.wq.1, cfg.q_dim, model.act_layout.q),
                (lt.wk.0, lt.wk.1, cfg.kv_dim, model.act_layout.k_cur),
                (lt.wv.0, lt.wv.1, cfg.kv_dim, model.act_layout.v_cur),
            ];
            bench_fused_normed(
                model,
                cfg,
                se,
                cfg.n_embd,
                model.act_layout.x,
                lt.attn_norm_off,
                WeightSet::Attn,
                &ranges,
            );
        }),
    ));
    entries.push((
        format!("matvec gate_up fused+normed N={}", 2 * cfg.n_ff),
        n_layer,
        Box::new(|model, cfg, se| {
            let lt = &model.layer_tensors[0];
            let ranges = [
                (lt.wg.0, lt.wg.1, cfg.n_ff, model.act_layout.gate),
                (lt.wu.0, lt.wu.1, cfg.n_ff, model.act_layout.up),
                (0, 0, 0, 0),
            ];
            bench_fused_normed(
                model,
                cfg,
                se,
                cfg.n_embd,
                model.act_layout.x,
                lt.ffn_norm_off,
                WeightSet::FfnGU,
                &ranges,
            );
        }),
    ));

    // The lone surviving rms_norm dispatch: output_norm in the LM-head suffix.
    // Per-layer attn_norm/ffn_norm are folded into matvec_q1_0_fused_normed;
    // K-side rms_norm + RoPE are folded into kv_writeback_fused; Q-side into
    // q_norm_rope_fused.
    let output_norm_off = ot.output_norm_off;
    entries.push((
        format!("rms_norm ng=1 gs={} (output_norm)", cfg.n_embd),
        1,
        Box::new(move |model, cfg, se| {
            let p = RmsNormParams {
                group_size: cfg.n_embd,
                n_groups: 1,
                input_offset: model.act_layout.x_norm,
                output_offset: model.act_layout.x_norm,
                weight_offset: output_norm_off,
                eps: cfg.rms_eps,
            };
            let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rms_norm"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&model.pipes.rms_norm);
            cp.set_bind_group(0, &model.cached.rms_norm, &[]);
            cp.set_immediates(0, bytemuck::bytes_of(&p));
            cp.dispatch_workgroups(1, 1, 1);
        }),
    ));

    entries.push((
        format!("matvec Wdown_silu K={} N={}", cfg.n_ff, cfg.n_embd),
        n_layer,
        Box::new(|model, cfg, se| {
            const ROWS_PER_WG: u32 = 8;
            let lt = &model.layer_tensors[0];
            let (k, n) = (cfg.n_ff, cfg.n_embd);
            let n_wg = n.div_ceil(ROWS_PER_WG);
            let dispatch_x = n_wg.min(65535);
            let dispatch_y = n_wg.div_ceil(dispatch_x);
            let p = MatvecSiluParams {
                k,
                n,
                d_offset: lt.wd.0,
                qs_offset: lt.wd.1,
                gate_offset: model.act_layout.gate,
                up_offset: model.act_layout.up,
                output_offset: model.act_layout.x,
                accumulate: 1,
                dispatch_x_dim: dispatch_x,
            };
            let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matvec_silu"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&model.pipes.matvec_silu);
            cp.set_bind_group(0, matvec_bg(model, WeightSet::FfnD), &[]);
            cp.set_immediates(0, bytemuck::bytes_of(&p));
            cp.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }),
    ));

    entries.push((
        format!("attn_split+merge pos={attn_pos}"),
        n_layer,
        Box::new(move |model, cfg, se| {
            let (d_word, qs_byte) = kv_layer_offsets(cfg, model.max_seq, 0);
            let cur_pos = attn_pos + 1;
            let n_chunks_active = cur_pos.div_ceil(ATTN_CHUNK_SIZE);
            let ps = AttnSplitParams {
                head_dim: cfg.head_dim,
                n_head: cfg.n_head,
                n_kv_head: cfg.n_kv_head,
                pos: cur_pos,
                kv_stride: cfg.kv_dim,
                q_offset: model.act_layout.q,
                k_d_word_offset: d_word,
                k_qs_byte_offset: qs_byte,
                v_d_word_offset: d_word,
                v_qs_byte_offset: qs_byte,
                n_chunks_active,
                scale: 1.0 / (cfg.head_dim as f32).sqrt(),
            };
            let pm = AttnMergeParams {
                head_dim: cfg.head_dim,
                n_head: cfg.n_head,
                out_offset: model.act_layout.attn_out,
                n_chunks_active,
            };
            let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("attn_split+merge"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&model.pipes.attention_split);
            cp.set_bind_group(0, &model.cached.attn_split, &[]);
            cp.set_immediates(0, bytemuck::bytes_of(&ps));
            cp.dispatch_workgroups(cfg.n_kv_head, n_chunks_active, 1);
            cp.set_pipeline(&model.pipes.attention_merge);
            cp.set_bind_group(0, &model.cached.attn_merge, &[]);
            cp.set_immediates(0, bytemuck::bytes_of(&pm));
            cp.dispatch_workgroups(cfg.n_head, 1, 1);
        }),
    ));

    let attn_k_norm_off = lt0.attn_k_norm_off;
    entries.push((
        "kv_writeback_fused (K rms+rope+Q8_0 + V Q8_0)".to_string(),
        n_layer,
        Box::new(move |model, cfg, se| {
            let nb_per_row = cfg.kv_dim / 32;
            let (dst_d_word_offset, dst_qs_byte_offset) = kv_layer_offsets(cfg, model.max_seq, 0);
            let p = KvWritebackFusedParams {
                k_cur_off: model.act_layout.k_cur,
                v_cur_off: model.act_layout.v_cur,
                w_k_norm_off: attn_k_norm_off,
                rope_offset: 0,
                dst_d_word_offset,
                dst_qs_byte_offset,
                pos_base: 0,
                kv_dim: cfg.kv_dim,
                nb_per_row,
                eps: cfg.rms_eps,
            };
            let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("kv_writeback_fused"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&model.pipes.kv_writeback_fused);
            cp.set_bind_group(0, &model.cached.kv_writeback_fused, &[]);
            cp.set_immediates(0, bytemuck::bytes_of(&p));
            cp.dispatch_workgroups(cfg.n_kv_head, 1, 1);
        }),
    ));

    let attn_q_norm_off = lt0.attn_q_norm_off;
    entries.push((
        "q_norm_rope_fused (Q rms+rope)".to_string(),
        n_layer,
        Box::new(move |model, cfg, se| {
            let p = QNormRopeFusedParams {
                q_off: model.act_layout.q,
                w_q_norm_off: attn_q_norm_off,
                rope_offset: 0,
                pos_base: 0,
                q_dim: cfg.q_dim,
                eps: cfg.rms_eps,
            };
            let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("q_norm_rope_fused"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&model.pipes.q_norm_rope_fused);
            cp.set_bind_group(0, &model.cached.q_norm_rope_fused, &[]);
            cp.set_immediates(0, bytemuck::bytes_of(&p));
            cp.dispatch_workgroups(cfg.n_head, 1, 1);
        }),
    ));

    entries.push((
        "embed                           ".to_string(),
        1,
        Box::new(|model, cfg, se| {
            let ot = &model.output_tensors;
            let p = EmbedParams {
                k: cfg.n_embd,
                d_offset: ot.token_embd_d,
                qs_offset: ot.token_embd_qs,
                output_offset: model.act_layout.x,
                sample_offset: 0,
            };
            let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("embed"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&model.pipes.embed);
            cp.set_bind_group(0, &model.cached.embed, &[]);
            cp.set_immediates(0, bytemuck::bytes_of(&p));
            cp.dispatch_workgroups(1, 1, 1);
        }),
    ));

    entries.push((
        "topk_reduce n=n_vocab           ".to_string(),
        1,
        Box::new(|model, _cfg, se| {
            let p = TopKParams {
                n: model.cfg.n_vocab,
                in_offset: model.act_layout.logits,
                out_offset: 0,
                k: 1,
            };
            let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("topk_reduce"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&model.pipes.topk_reduce);
            cp.set_bind_group(0, &model.cached.topk_reduce, &[]);
            cp.set_immediates(0, bytemuck::bytes_of(&p));
            cp.dispatch_workgroups(1, 1, 1);
        }),
    ));

    // Warmup: run every entry once.
    {
        let mut se = StepEncoder::new(model);
        for (_l, _c, f) in &entries {
            f(model, cfg, &mut se);
        }
        let cb = se.finish();
        model.queue.submit(Some(cb));
        if let Err(e) = model.device.poll(PollType::wait_indefinitely()) {
            model.check_device()?;
            return Err(PotError::Poll(e));
        }
    }

    let mut breakdown: Vec<(String, u32, f32)> = Vec::new();
    for (label, calls, build) in &entries {
        let mut samples = Vec::with_capacity(repeats as usize);
        for _ in 0..repeats {
            let t = Instant::now();
            let mut se = StepEncoder::new(model);
            for _ in 0..n_per_cb {
                build(model, cfg, &mut se);
            }
            let cb = se.finish();
            model.queue.submit(Some(cb));
            if let Err(e) = model.device.poll(PollType::wait_indefinitely()) {
                model.check_device()?;
                return Err(PotError::Poll(e));
            }
            samples.push(t.elapsed().as_secs_f32());
        }
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        let per_call_us = mean / n_per_cb as f32 * 1e6;
        breakdown.push((label.clone(), *calls, per_call_us));
    }

    // Calibrate by timing real tg steps now that the GPU is warm. Scale
    // per-kernel times so the corrected total matches measured throughput
    // (covers per-pass overhead + readback that isolated dispatches miss).
    let step_ms: f32 = {
        let n: u32 = 32;
        let mut pos = 0u32;
        step_matvec_no_sample(model, 1, pos);
        if let Err(e) = model.device.poll(PollType::wait_indefinitely()) {
            model.check_device()?;
            return Err(PotError::Poll(e));
        }
        pos += 1;
        let t = Instant::now();
        for _ in 0..n {
            step_matvec_topk(model, 1, pos, 1)?;
            pos += 1;
        }
        let per_step = t.elapsed().as_secs_f32() / n as f32;
        eprintln!(
            "calibration (incl. readback): {:.3} ms/step  ({:.1} t/s)",
            per_step * 1000.0,
            1.0 / per_step
        );
        per_step * 1000.0
    };

    let raw_total_ms: f32 = breakdown
        .iter()
        .map(|(_, c, us)| (*c as f32) * (*us) / 1000.0)
        .sum();
    let scale = step_ms / raw_total_ms;
    let mut sorted = breakdown.clone();
    sorted.sort_by(|a, b| {
        let ma = a.1 as f32 * a.2;
        let mb = b.1 as f32 * b.2;
        mb.partial_cmp(&ma).unwrap_or(Ordering::Equal)
    });
    let adj_total_ms: f32 = sorted
        .iter()
        .map(|(_, c, us)| (*c as f32) * (*us) * scale / 1000.0)
        .sum();

    println!();
    println!(
        "| kernel                                        | calls/step | raw us | adj us | raw ms | adj ms | %step |"
    );
    println!(
        "|-----------------------------------------------|-----------:|-------:|-------:|-------:|-------:|------:|"
    );
    for (label, calls, raw_us) in &sorted {
        let adj_us = raw_us * scale;
        let raw_ms = (*calls as f32) * (*raw_us) / 1000.0;
        let adj_ms = (*calls as f32) * adj_us / 1000.0;
        let pct = 100.0 * adj_ms / adj_total_ms;
        println!(
            "| {label:<45} | {calls:>10} | {raw_us:>6.2} | {adj_us:>6.2} | {raw_ms:>6.3} | {adj_ms:>6.3} | {pct:>5.1} |"
        );
    }
    println!(
        "|-----------------------------------------------|-----------:|-------:|-------:|-------:|-------:|------:|"
    );
    println!(
        "| TOTAL (isolated)                              |            |        |        | {raw_total_ms:>6.3} | {adj_total_ms:>6.3} |       |"
    );
    println!();
    println!("raw (isolated):           {:.1} t/s", 1000.0 / raw_total_ms);
    println!(
        "calibrated step:          {:.3} ms  ({:.1} t/s)",
        step_ms,
        1000.0 / step_ms
    );
    println!("scale factor:             ×{scale:.4}");
    println!("corrected (scaled):       {:.1} t/s", 1000.0 / adj_total_ms);
    Ok(())
}
