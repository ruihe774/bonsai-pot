//! Forward pass and generation loop for Bonsai-4B.
//!
//! Two execution modes:
//!   - "gen":    processes every token (prompt + generated) via the multiply-free
//!               Q1_0 matvec kernel (single-token path).
//!   - "prompt": processes the prompt as one batch using the dot4I8Packed matmul
//!               with a Q8_0 quantize-activation pre-pass, then generates with
//!               matvec.

use crate::model::*;
use bytemuck::Pod;

// ---------- per-step encoder + uniform pool ---------------------------------

struct UniformPool {
    cpu: Vec<u8>,
    next_slot: u64,
}

impl UniformPool {
    fn new() -> Self {
        Self { cpu: Vec::with_capacity(64 * 1024), next_slot: 0 }
    }
    fn alloc<T: Pod>(&mut self, params: &T) -> u32 {
        let slot = self.next_slot;
        self.next_slot += 1;
        let off = (slot * UNIFORM_SLOT_SIZE) as usize;
        if self.cpu.len() < off + UNIFORM_SLOT_SIZE as usize {
            self.cpu.resize(off + UNIFORM_SLOT_SIZE as usize, 0);
        }
        let bytes = bytemuck::bytes_of(params);
        self.cpu[off..off + bytes.len()].copy_from_slice(bytes);
        (slot * UNIFORM_SLOT_SIZE) as u32
    }
}

struct StepEncoder<'a> {
    model: &'a Model,
    encoder: wgpu::CommandEncoder,
    uniforms: UniformPool,
}

impl<'a> StepEncoder<'a> {
    fn new(model: &'a Model) -> Self {
        let encoder = model.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("step"),
        });
        Self { model, encoder, uniforms: UniformPool::new() }
    }

    /// Append `params` into the uniform CPU buffer and return its dynamic offset.
    fn alloc_uniform<T: Pod>(&mut self, params: &T) -> u32 {
        self.uniforms.alloc(params)
    }

    fn finish(self) -> wgpu::CommandBuffer {
        if !self.uniforms.cpu.is_empty() {
            self.model.queue.write_buffer(&self.model.buffers.uniform, 0, &self.uniforms.cpu);
        }
        self.encoder.finish()
    }
}

// ---------- bind-group helpers ----------------------------------------------

fn ubo_entry<'a>(buffer: &'a wgpu::Buffer, size_bytes: u64) -> wgpu::BindingResource<'a> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer, offset: 0, size: Some(std::num::NonZeroU64::new(size_bytes).unwrap()),
    })
}

fn make_bg(model: &Model, layout: &wgpu::BindGroupLayout, label: &str,
           uniform_size: u64, storages: &[&wgpu::Buffer]) -> wgpu::BindGroup {
    let mut entries = vec![wgpu::BindGroupEntry {
        binding: 0,
        resource: ubo_entry(&model.buffers.uniform, uniform_size),
    }];
    for (i, b) in storages.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: i as u32 + 1,
            resource: b.as_entire_binding(),
        });
    }
    model.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label), layout, entries: &entries,
    })
}

// ---------- in-pass kernel dispatch helpers ---------------------------------
// These variants take a `&mut wgpu::ComputePass` already opened by the caller,
// allowing many dispatches to share one pass and amortize the
// begin_compute_pass cost (~25us each on RADV).

fn dispatch_rms_norm(
    model: &Model, cfg: &Config, uniforms: &mut UniformPool,
    pass: &mut wgpu::ComputePass<'_>,
    n_groups: u32, group_size: u32, in_off: u32, out_off: u32, w_off: u32,
) {
    let p = RmsNormParams {
        group_size, n_groups,
        input_offset: in_off, output_offset: out_off, weight_offset: w_off,
        eps: cfg.rms_eps,
        ..Default::default()
    };
    let off = uniforms.alloc(&p);
    let bg = make_bg(model, &model.bgls.rms_norm, "rms_bg",
                     std::mem::size_of::<RmsNormParams>() as u64,
                     &[&model.buffers.act, &model.buffers.w_norms]);
    pass.set_pipeline(&model.pipes.rms_norm);
    pass.set_bind_group(0, &bg, &[off]);
    pass.dispatch_workgroups(n_groups, 1, 1);
}

fn dispatch_rope(
    model: &Model, cfg: &Config, uniforms: &mut UniformPool,
    pass: &mut wgpu::ComputePass<'_>,
    data_off: u32, n_tokens: u32, n_heads: u32, pos_base: u32,
) {
    let p = RopeParams {
        head_dim: cfg.head_dim, n_heads, n_tokens, pos_base,
        data_offset: data_off, rope_table_offset: 0,
        ..Default::default()
    };
    let off = uniforms.alloc(&p);
    let bg = make_bg(model, &model.bgls.rope, "rope_bg",
                     std::mem::size_of::<RopeParams>() as u64,
                     &[&model.buffers.rope_table, &model.buffers.act]);
    pass.set_pipeline(&model.pipes.rope_neox);
    pass.set_bind_group(0, &bg, &[off]);
    pass.dispatch_workgroups(n_tokens, n_heads, 1);
}

fn dispatch_matvec_q1_0(
    model: &Model, uniforms: &mut UniformPool, pass: &mut wgpu::ComputePass<'_>,
    k: u32, n: u32,
    weights: &wgpu::Buffer, w_d: u32, w_qs: u32,
    in_off: u32, out_off: u32, accumulate: bool,
) {
    const ROWS_PER_WG: u32 = 8;
    let n_wg = (n + ROWS_PER_WG - 1) / ROWS_PER_WG;
    let dispatch_x = n_wg.min(65535);
    let dispatch_y = (n_wg + dispatch_x - 1) / dispatch_x;
    let p = MatvecParams {
        k, n,
        d_offset: w_d, qs_offset: w_qs,
        input_offset: in_off, output_offset: out_off,
        accumulate: if accumulate { 1 } else { 0 },
        dispatch_x_dim: dispatch_x,
    };
    let off = uniforms.alloc(&p);
    let bg = make_bg(model, &model.bgls.matvec, "matvec_bg",
                     std::mem::size_of::<MatvecParams>() as u64,
                     &[weights, &model.buffers.act]);
    pass.set_pipeline(&model.pipes.matvec);
    pass.set_bind_group(0, &bg, &[off]);
    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
}

fn dispatch_matvec_q1_0_fused(
    model: &Model, uniforms: &mut UniformPool, pass: &mut wgpu::ComputePass<'_>,
    k: u32, input_offset: u32, weights: &wgpu::Buffer,
    ranges: &[(u32, u32, u32, u32)],
) {
    debug_assert!(ranges.len() == 2 || ranges.len() == 3);
    for (_, _, n, _) in ranges { debug_assert!(n % 8 == 0); }
    let r = |i: usize| ranges.get(i).copied().unwrap_or((0, 0, 0, 0));
    let (d0, qs0, n0, o0) = r(0);
    let (d1, qs1, n1, o1) = r(1);
    let (d2, qs2, n2, o2) = r(2);
    let n_total = n0 + n1 + n2;
    const ROWS_PER_WG: u32 = 8;
    let n_wg = (n_total + ROWS_PER_WG - 1) / ROWS_PER_WG;
    let dispatch_x = n_wg.min(65535);
    let dispatch_y = (n_wg + dispatch_x - 1) / dispatch_x;
    let p = MatvecFusedParams {
        k, n_total, input_offset, dispatch_x_dim: dispatch_x,
        d_offset_0: d0, qs_offset_0: qs0, n_0: n0, output_offset_0: o0,
        d_offset_1: d1, qs_offset_1: qs1, n_1: n1, output_offset_1: o1,
        d_offset_2: d2, qs_offset_2: qs2, n_2: n2, output_offset_2: o2,
    };
    let off = uniforms.alloc(&p);
    let bg = make_bg(model, &model.bgls.matvec, "matvec_fused_bg",
                     std::mem::size_of::<MatvecFusedParams>() as u64,
                     &[weights, &model.buffers.act]);
    pass.set_pipeline(&model.pipes.matvec_fused);
    pass.set_bind_group(0, &bg, &[off]);
    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
}

fn dispatch_silu_mul(
    model: &Model, uniforms: &mut UniformPool, pass: &mut wgpu::ComputePass<'_>,
    n: u32, m: u32, gate_off: u32, up_off: u32, out_off: u32,
) {
    let total = n * m;
    let groups = (total + 63) / 64;
    let dispatch_x = groups.min(65535);
    let dispatch_y = (groups + dispatch_x - 1) / dispatch_x;
    let p = SiluMulParams {
        n, m, gate_offset: gate_off, up_offset: up_off, out_offset: out_off,
        dispatch_x_count: dispatch_x * 64,
        ..Default::default()
    };
    let off = uniforms.alloc(&p);
    let bg = make_bg(model, &model.bgls.silu_mul, "silu_bg",
                     std::mem::size_of::<SiluMulParams>() as u64,
                     &[&model.buffers.act]);
    pass.set_pipeline(&model.pipes.silu_mul);
    pass.set_bind_group(0, &bg, &[off]);
    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
}

// ---------- benchmark entry point -------------------------------------------

/// Match `llama-bench` style: pp{n} measures batched-prefill throughput (tokens
/// pushed through projections via the dot4I8Packed matmul kernel); tg{n}
/// measures single-token generation throughput (multiply-free matvec on the
/// hot path, `n` forward steps from an empty KV cache).
pub async fn bench(model: &mut Model, pp_n: u32, tg_n: u32, repeats: u32) {
    let cfg = model.cfg.clone();
    eprintln!("--- bench: pp={pp_n}, tg={tg_n}, repeats={repeats} (after 1 warmup) ---");

    // synthesize a pp_n-token prompt (any valid token IDs work; we cycle 1..N).
    let prompt: Vec<u32> = (0..pp_n).map(|i| (i % (cfg.n_vocab - 1)) + 1).collect();

    // ----- warm up -----
    let _ = prefill_matmul(model, &cfg, &prompt[..pp_n.min(model.m_max) as usize]).await;
    let _ = step_matvec(model, &cfg, 1u32, 0, false).await;
    let _ = model.device.poll(wgpu::PollType::wait_indefinitely());

    // ----- pp{pp_n} -----
    let mut pp_times = Vec::with_capacity(repeats as usize);
    for _ in 0..repeats {
        let t = std::time::Instant::now();
        let _ = prefill_matmul(model, &cfg, &prompt).await;
        let _ = model.device.poll(wgpu::PollType::wait_indefinitely());
        pp_times.push(t.elapsed().as_secs_f32());
    }
    let pp_mean = pp_times.iter().sum::<f32>() / pp_times.len() as f32;
    let pp_std  = stddev(&pp_times, pp_mean);
    let pp_t_s_mean = pp_n as f32 / pp_mean;
    let pp_t_s_std  = pp_t_s_mean * (pp_std / pp_mean);

    // ----- tg{tg_n} (from empty KV) -----
    let mut tg_times = Vec::with_capacity(repeats as usize);
    for _ in 0..repeats {
        let t = std::time::Instant::now();
        let mut tok = 1u32;
        for pos in 0..tg_n {
            tok = step_matvec(model, &cfg, tok, pos, true).await;
        }
        let _ = model.device.poll(wgpu::PollType::wait_indefinitely());
        tg_times.push(t.elapsed().as_secs_f32());
        let _ = tok;
    }
    let tg_mean = tg_times.iter().sum::<f32>() / tg_times.len() as f32;
    let tg_std  = stddev(&tg_times, tg_mean);
    let tg_t_s_mean = tg_n as f32 / tg_mean;
    let tg_t_s_std  = tg_t_s_mean * (tg_std / tg_mean);

    // ----- tg{tg_n} pipelined (chunks of CHUNK steps per CB, no per-step readback) -----
    // Lower bound on tg cost: amortizes per-step CPU sync. Chunk size keeps
    // each CB well under uniform pool / GPU TDR limits.
    const CHUNK: u32 = 8;
    let mut tgp_times = Vec::with_capacity(repeats as usize);
    for _ in 0..repeats {
        let t = std::time::Instant::now();
        let mut pos = 0u32;
        while pos < tg_n {
            let end = (pos + CHUNK).min(tg_n);
            let mut se = StepEncoder::new(model);
            for (i, p) in (pos..end).enumerate() {
                // Chain via sample buffer: step i reads sample[i], writes sample[i+1].
                encode_step_matvec(&mut se, &cfg, i as u32, (i + 1) as u32, p);
            }
            let cb = se.finish();
            model.queue.submit(Some(cb));
            pos = end;
        }
        let _ = model.device.poll(wgpu::PollType::wait_indefinitely());
        tgp_times.push(t.elapsed().as_secs_f32());
    }
    let tgp_mean = tgp_times.iter().sum::<f32>() / tgp_times.len() as f32;
    let tgp_std  = stddev(&tgp_times, tgp_mean);
    let tgp_t_s_mean = tg_n as f32 / tgp_mean;
    let tgp_t_s_std  = tgp_t_s_mean * (tgp_std / tgp_mean);

    println!();
    println!("| backend            |          test |               t/s |");
    println!("| ------------------ | ------------- | ----------------: |");
    println!("| bonsai-wgpu        |        pp{pp_n:<3} | {pp_t_s_mean:>9.2} ± {pp_t_s_std:>5.2} |");
    println!("| bonsai-wgpu        |        tg{tg_n:<3} | {tg_t_s_mean:>9.2} ± {tg_t_s_std:>5.2} |");
    println!("| bonsai-wgpu        |    tg{tg_n:<3} pipe | {tgp_t_s_mean:>9.2} ± {tgp_t_s_std:>5.2} |");
}

fn stddev(xs: &[f32], mean: f32) -> f32 {
    if xs.len() < 2 { return 0.0; }
    let var: f32 = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / (xs.len() - 1) as f32;
    var.sqrt()
}

// ---------- microbench: per-kernel breakdown for tg ------------------------
//
// For each kernel kind that appears in tg, build a command buffer with N
// identical dispatches, submit it, wait, and divide by N. Then multiply by
// per-step call count to estimate the contribution to one tg step.
//
// All matvecs are issued against blk.0's weights; shape is what determines
// dispatch cost, and shapes are uniform across layers.
pub async fn microbench_tg(model: &mut Model, repeats: u32) {
    let cfg = model.cfg.clone();
    let n_per_cb: u32 = 200;
    let attn_pos: u32 = 64; // representative cache size for tg128

    eprintln!("--- microbench tg (N={n_per_cb}/CB, repeats={repeats}) ---");

    // (label, calls_per_step, build_one_dispatch)
    type DispatchFn<'a> = Box<dyn Fn(&Model, &Config, &mut StepEncoder) + 'a>;
    let mut entries: Vec<(String, u32, DispatchFn)> = Vec::new();

    // matvecs (single-range shapes used as-is in tg)
    let matvec_shapes: &[(&str, u32, u32, u32, &str, &str)] = &[
        ("matvec Wo         K=4096 N=2560 ", cfg.q_dim,  cfg.n_embd, 36, "blk.0.attn_output.weight", "w_attn"),
        ("matvec Wdown      K=9728 N=2560 ", cfg.n_ff,   cfg.n_embd, 36, "blk.0.ffn_down.weight",    "w_ffn_d"),
        ("matvec LM_head    K=2560 N=152K ", cfg.n_embd, cfg.n_vocab, 1, "token_embd.weight",        "w_embed"),
    ];
    for &(label, k, n, calls, ten_name, buf_name) in matvec_shapes {
        let ten_name_s = ten_name.to_string();
        let buf_name_s = buf_name.to_string();
        entries.push((label.to_string(), calls, Box::new(move |model, cfg, se| {
            let t = tensor(cfg, &ten_name_s).clone();
            let buf = match buf_name_s.as_str() {
                "w_attn"   => &model.buffers.w_attn,
                "w_ffn_gu" => &model.buffers.w_ffn_gu,
                "w_ffn_d"  => &model.buffers.w_ffn_d,
                "w_embed"  => &model.buffers.w_embed,
                _ => unreachable!(),
            };
            matvec_q1_0(model, cfg, se, k, n, buf, t.d_offset as u32, t.qs_offset as u32,
                        model.act_layout.x_norm, model.act_layout.x, false);
        })));
    }

    // Fused matvecs: QKV (3 ranges, N=4608 total) and gate-up (2 ranges, N=19456 total).
    entries.push(("matvec QKV  fused N=4608        ".to_string(), 36, Box::new(move |model, cfg, se| {
        let wq = tensor(cfg, "blk.0.attn_q.weight").clone();
        let wk = tensor(cfg, "blk.0.attn_k.weight").clone();
        let wv = tensor(cfg, "blk.0.attn_v.weight").clone();
        matvec_q1_0_fused(model, se, cfg.n_embd, model.act_layout.x_norm,
                          &model.buffers.w_attn, &[
            (wq.d_offset as u32, wq.qs_offset as u32, cfg.q_dim,  model.act_layout.q),
            (wk.d_offset as u32, wk.qs_offset as u32, cfg.kv_dim, model.act_layout.k_cur),
            (wv.d_offset as u32, wv.qs_offset as u32, cfg.kv_dim, model.act_layout.v_cur),
        ]);
    })));
    entries.push(("matvec gate_up fused N=19456    ".to_string(), 36, Box::new(move |model, cfg, se| {
        let wg = tensor(cfg, "blk.0.ffn_gate.weight").clone();
        let wu = tensor(cfg, "blk.0.ffn_up.weight").clone();
        matvec_q1_0_fused(model, se, cfg.n_embd, model.act_layout.x_norm,
                          &model.buffers.w_ffn_gu, &[
            (wg.d_offset as u32, wg.qs_offset as u32, cfg.n_ff, model.act_layout.gate),
            (wu.d_offset as u32, wu.qs_offset as u32, cfg.n_ff, model.act_layout.up),
        ]);
    })));

    // rms_norm shapes (whole-vec, q-norm, k-norm, output_norm shares whole-vec shape)
    let rms_shapes: &[(&str, u32, u32, u32, &str)] = &[
        ("rms_norm  ng=1  gs=2560        ", 1,  cfg.n_embd,    36 + 36 + 1, "blk.0.attn_norm.weight"),
        ("rms_norm  ng=32 gs=128         ", cfg.n_head,    cfg.head_dim, 36, "blk.0.attn_q_norm.weight"),
        ("rms_norm  ng=8  gs=128         ", cfg.n_kv_head, cfg.head_dim, 36, "blk.0.attn_k_norm.weight"),
    ];
    for &(label, ng, gs, calls, ten_name) in rms_shapes {
        let ten_name_s = ten_name.to_string();
        entries.push((label.to_string(), calls, Box::new(move |model, cfg, se| {
            let t = tensor(cfg, &ten_name_s).clone();
            rms_norm(model, cfg, se, ng, gs,
                     model.act_layout.x_norm, model.act_layout.x_norm, (t.offset / 4) as u32);
        })));
    }

    // rope (32 heads / 8 heads)
    entries.push(("rope     n_heads=32             ".to_string(), 36, Box::new(move |model, cfg, se| {
        rope(model, cfg, se, model.act_layout.q,     1, cfg.n_head,    0);
    })));
    entries.push(("rope     n_heads=8              ".to_string(), 36, Box::new(move |model, cfg, se| {
        rope(model, cfg, se, model.act_layout.k_cur, 1, cfg.n_kv_head, 0);
    })));

    // silu_mul (n=n_ff)
    entries.push(("silu_mul n=9728                 ".to_string(), 36, Box::new(move |model, cfg, se| {
        silu_mul(model, se, cfg.n_ff, 1, model.act_layout.gate, model.act_layout.up, model.act_layout.ffn_in);
    })));

    // attention (single token, pos=64)
    entries.push((format!("attention pos={attn_pos:<3}                "), 36, Box::new(move |model, cfg, se| {
        let layer_offset = 0u64;
        let p = AttnParams {
            head_dim: cfg.head_dim, n_head: cfg.n_head, n_kv_head: cfg.n_kv_head,
            pos: attn_pos,
            kv_stride: cfg.kv_dim,
            q_offset: model.act_layout.q,
            k_cache_offset: ((layer_offset / 4) as u32),
            v_cache_offset: ((layer_offset / 4) as u32),
            out_offset: model.act_layout.attn_out,
            scale: 1.0 / (cfg.head_dim as f32).sqrt(),
            m_tokens: 1, is_prefill: 0,
        };
        let off = se.alloc_uniform(&p);
        let bg = make_bg(model, &model.bgls.attn, "attn_bg", std::mem::size_of::<AttnParams>() as u64,
                         &[&model.buffers.act, &model.buffers.kv_k, &model.buffers.kv_v]);
        let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("attn"), timestamp_writes: None });
        cp.set_pipeline(&model.pipes.attention);
        cp.set_bind_group(0, &bg, &[off]);
        cp.dispatch_workgroups(cfg.n_head, 1, 1);
    })));

    // embed (1 token, K=n_embd)
    entries.push(("embed                           ".to_string(), 1, Box::new(move |model, cfg, se| {
        let t = tensor(cfg, "token_embd.weight").clone();
        let p = EmbedParams {
            k: cfg.n_embd,
            d_offset: t.d_offset as u32,
            qs_offset: t.qs_offset as u32,
            output_offset: model.act_layout.x,
            sample_offset: 0,
            m_token: 0,
            ..Default::default()
        };
        let off = se.alloc_uniform(&p);
        let bg = make_bg(model, &model.bgls.embed, "embed_bg", std::mem::size_of::<EmbedParams>() as u64,
                         &[&model.buffers.w_embed, &model.buffers.act, &model.buffers.sample]);
        let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("embed"), timestamp_writes: None });
        cp.set_pipeline(&model.pipes.embed);
        cp.set_bind_group(0, &bg, &[off]);
        cp.dispatch_workgroups(1, 1, 1);
    })));

    // argmax (n=n_vocab)
    entries.push(("argmax  n=n_vocab               ".to_string(), 1, Box::new(move |model, _cfg, se| {
        let p = ArgmaxParams { n: model.cfg.n_vocab, in_offset: model.act_layout.logits, out_offset: 0, _p0: 0 };
        let off = se.alloc_uniform(&p);
        let bg = make_bg(model, &model.bgls.argmax, "argmax_bg", std::mem::size_of::<ArgmaxParams>() as u64,
                         &[&model.buffers.act, &model.buffers.sample]);
        let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("argmax"), timestamp_writes: None });
        cp.set_pipeline(&model.pipes.argmax);
        cp.set_bind_group(0, &bg, &[off]);
        cp.dispatch_workgroups(1, 1, 1);
    })));

    // kv-cache copies: 36 K + 36 V per step. Issue copy_buffer_to_buffer in
    // a dedicated CB. Treat as one combined entry (count = 72).
    let kv_row_bytes = (cfg.kv_dim * 4) as u64;

    // Warmup: each closure once.
    {
        let mut se = StepEncoder::new(model);
        for (_l, _c, f) in entries.iter() { f(model, &cfg, &mut se); }
        se.encoder.copy_buffer_to_buffer(&model.buffers.act, 0, &model.buffers.kv_k, 0, kv_row_bytes);
        let cb = se.finish();
        model.queue.submit(Some(cb));
        let _ = model.device.poll(wgpu::PollType::wait_indefinitely());
    }

    // Per-entry timing.
    let mut breakdown: Vec<(String, u32, f32)> = Vec::new(); // (label, calls, per_call_us)
    for (label, calls, build) in entries.iter() {
        let mut samples = Vec::with_capacity(repeats as usize);
        for _ in 0..repeats {
            let t = std::time::Instant::now();
            let mut se = StepEncoder::new(model);
            for _ in 0..n_per_cb { build(model, &cfg, &mut se); }
            let cb = se.finish();
            model.queue.submit(Some(cb));
            let _ = model.device.poll(wgpu::PollType::wait_indefinitely());
            samples.push(t.elapsed().as_secs_f32());
        }
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        let per_call_us = mean / n_per_cb as f32 * 1e6;
        breakdown.push((label.clone(), *calls, per_call_us));
    }

    // KV-cache copy (paired K+V) — measured separately because it's not a compute pass
    {
        let mut samples = Vec::with_capacity(repeats as usize);
        for _ in 0..repeats {
            let t = std::time::Instant::now();
            let mut enc = model.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("kvcopy") });
            for _ in 0..n_per_cb {
                enc.copy_buffer_to_buffer(&model.buffers.act, 0, &model.buffers.kv_k, 0, kv_row_bytes);
                enc.copy_buffer_to_buffer(&model.buffers.act, 0, &model.buffers.kv_v, 0, kv_row_bytes);
            }
            model.queue.submit(Some(enc.finish()));
            let _ = model.device.poll(wgpu::PollType::wait_indefinitely());
            samples.push(t.elapsed().as_secs_f32());
        }
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        // each iter is K+V (2 copies), and there are 36 layers = 36 paired calls per step
        let per_pair_us = mean / n_per_cb as f32 * 1e6;
        breakdown.push(("kv-cache copy (K+V)             ".to_string(), 36, per_pair_us));
    }

    println!();
    println!("| kernel                              | calls/step | us/call | ms/step | %step |");
    println!("|-------------------------------------|-----------:|--------:|--------:|------:|");
    let total_ms: f32 = breakdown.iter().map(|(_, c, us)| (*c as f32) * (*us) / 1000.0).sum();
    let mut sorted = breakdown.clone();
    sorted.sort_by(|a, b| {
        let ma = a.1 as f32 * a.2;
        let mb = b.1 as f32 * b.2;
        mb.partial_cmp(&ma).unwrap()
    });
    for (label, calls, per_us) in &sorted {
        let ms_step = (*calls as f32) * (*per_us) / 1000.0;
        let pct = 100.0 * ms_step / total_ms;
        println!("| {label:<35} | {calls:>10} | {per_us:>7.2} | {ms_step:>7.3} | {pct:>5.1} |");
    }
    println!("|-------------------------------------|-----------:|--------:|--------:|------:|");
    println!("| TOTAL (sum of isolated)             |            |         | {total_ms:>7.3} |       |");
    println!("expected tg t/s if isolated sum was reality: {:.1}", 1000.0 / total_ms);
}

// ---------- generation entry point ------------------------------------------

pub async fn generate(model: &mut Model, n_gen: u32, mode: &str) {
    let cfg = model.cfg.clone();
    let prompt = model.prompt.clone();
    let prompt_n = prompt.len() as u32;
    eprintln!("mode = {} | prompt_n = {} | n_gen = {}", mode, prompt_n, n_gen);

    let max_pos = prompt_n + n_gen;
    if max_pos > model.max_seq {
        panic!("prompt+gen ({max_pos}) exceeds MAX_SEQ ({})", model.max_seq);
    }

    let mut generated: Vec<u32> = Vec::with_capacity(n_gen as usize);

    let t_start = std::time::Instant::now();
    let last_tok = match mode {
        "prompt" => {
            // Single batched prefill of all prompt tokens; logits for the last
            // token live at the end of the activations buffer.
            let last = prefill_matmul(model, &cfg, &prompt).await;
            eprintln!("prefill ({} tokens) took {:?}", prompt_n, t_start.elapsed());
            last
        }
        "gen" => {
            // Run matvec single-token path over every prompt token.
            let mut last = 0u32;
            for (t, &tok) in prompt.iter().enumerate() {
                let want_logits = t == prompt.len() - 1;
                last = step_matvec(model, &cfg, tok, t as u32, want_logits).await;
                if !want_logits {
                    // discard sampled value — we just needed the cache to fill
                }
            }
            eprintln!("matvec prefill ({} tokens) took {:?}", prompt_n, t_start.elapsed());
            last
        }
        _ => panic!("mode must be 'gen' or 'prompt'"),
    };

    generated.push(last_tok);
    let t_gen = std::time::Instant::now();
    let pos_start = prompt_n;
    let remaining = n_gen.saturating_sub(1);
    // Pipelined gen: encode CHUNK steps per CB, chained via sample buffer.
    // We submit all chunks back-to-back (sample[CHUNK]→sample[0] copy on the GPU
    // hands the next chunk its input token) and await ONLY ONCE at the very
    // end, so CPU encoding for chunk N+1 overlaps with GPU running chunk N.
    // EOS is detected post-hoc from the readback (we may overshoot by up to
    // CHUNK-1 tokens; that's fine for a non-streaming CLI).
    const CHUNK: u32 = 8;
    if remaining > 0 {
        // Seed sample[0] with the input token for the first chunk.
        model.queue.write_buffer(&model.buffers.sample, 0, bytemuck::bytes_of(&last_tok));
        // Each chunk uses contiguous sample slots: chunk k step j reads
        // sample[k*CHUNK + j] and writes sample[k*CHUNK + j + 1]. The first
        // step of chunk k+1 then naturally reads sample[(k+1)*CHUNK], which
        // chunk k's last argmax just wrote.
        let mut produced = 0u32;
        let mut pos_local = pos_start;
        let mut chunk_idx = 0u32;
        while produced < remaining {
            let chunk_n = (remaining - produced).min(CHUNK);
            let base = chunk_idx * CHUNK;
            let mut se = StepEncoder::new(model);
            for i in 0..chunk_n {
                encode_step_matvec(&mut se, &cfg, base + i, base + i + 1, pos_local + i);
            }
            let cb = se.finish();
            model.queue.submit(Some(cb));
            pos_local += chunk_n;
            produced += chunk_n;
            chunk_idx += 1;
        }
        // After all chunks: sample[1..=remaining] holds the generated tokens.
        // Copy to readback in a tiny CB and await once.
        {
            let bytes = (remaining as u64) * 4;
            let mut enc = model.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gen_readback") });
            enc.copy_buffer_to_buffer(&model.buffers.sample, 4, &model.buffers.readback, 0, bytes);
            model.queue.submit(Some(enc.finish()));
            let slice = model.buffers.readback.slice(0..bytes);
            let (s, r) = futures_intrusive::channel::shared::oneshot_channel::<()>();
            slice.map_async(wgpu::MapMode::Read, move |_| { let _ = s.send(()); });
            let _ = model.device.poll(wgpu::PollType::wait_indefinitely());
            r.receive().await;
            let data = slice.get_mapped_range();
            let all_tokens: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&data[..bytes as usize]).to_vec();
            drop(data);
            model.buffers.readback.unmap();
            for &t in &all_tokens {
                generated.push(t);
                if t == cfg.eos_token_id { break; }
            }
        }
    }
    let gen_secs = t_gen.elapsed().as_secs_f32();
    eprintln!("generated {} tokens in {:.2}s ({:.1} tok/s)",
              generated.len(), gen_secs, generated.len() as f32 / gen_secs.max(1e-3));

    // Decode & print
    print!("{}", cfg.prompt_text);
    for &id in &generated {
        let s = &model.vocab[id as usize];
        // GPT-2-ish byte-level vocab: 'Ġ' = space, 'Ċ' = newline, etc.
        // Convert each unicode codepoint back to its byte using the inverse map.
        print!("{}", decode_token_bytes(s));
    }
    println!();
}

/// Minimal inverse of GPT-2's bytes_to_unicode map: maps each codepoint in a
/// vocab token back to its raw byte, then interprets the bytes as UTF-8.
fn decode_token_bytes(s: &str) -> String {
    // Build the inverse map lazily with a thread-local cache.
    use std::sync::OnceLock;
    static INV: OnceLock<[u8; 0x180]> = OnceLock::new();
    let inv = INV.get_or_init(|| {
        let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32)
            .chain(0xa1..=0xac).chain(0xae..=0xff).collect();
        let mut cs = bs.clone();
        let mut n = 0u32;
        for b in 0..256u32 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }
        // map cs[i] -> bs[i] (cs[i] is the codepoint, bs[i] is the original byte)
        // We need a lookup: codepoint -> byte. cs values fit in [0x21, 0x180).
        let mut inv = [0u8; 0x180];
        for (b, c) in bs.iter().zip(cs.iter()) {
            if (*c as usize) < inv.len() {
                inv[*c as usize] = *b as u8;
            }
        }
        inv
    });
    let mut out_bytes = Vec::new();
    for ch in s.chars() {
        let cp = ch as usize;
        if cp < inv.len() && (inv[cp] != 0 || cp == 0) {
            out_bytes.push(inv[cp]);
        } else {
            // Fallback: keep as UTF-8 (special tokens like <|im_start|> etc.)
            let mut buf = [0u8; 4];
            out_bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
        }
    }
    String::from_utf8_lossy(&out_bytes).into_owned()
}

// ---------- single-token forward (matvec / multiply-free) -------------------

/// Encode one tg step into the given encoder. The input token is read from
/// `sample[sample_in]`; the argmax writes to `sample[sample_out]`. Caller is
/// responsible for ensuring `sample[sample_in]` is populated (either by CPU
/// write_buffer for the first step of a chain, or by a prior step's argmax).
fn encode_step_matvec(
    se: &mut StepEncoder, cfg: &Config,
    sample_in: u32, sample_out: u32, pos: u32,
) {
    let cache_row_bytes = (cfg.kv_dim * 4) as u64;
    let StepEncoder { model: m, encoder, uniforms } = se;
    // Pass 0: embed + layer 0 pre-kv
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("phase0"), timestamp_writes: None });
        let tok_embd = tensor(cfg, "token_embd.weight").clone();
        let p = EmbedParams {
            k: cfg.n_embd,
            d_offset: tok_embd.d_offset as u32,
            qs_offset: tok_embd.qs_offset as u32,
            output_offset: m.act_layout.x,
            sample_offset: sample_in,
            m_token: 0,
            ..Default::default()
        };
        let off = uniforms.alloc(&p);
        let bg = make_bg(m, &m.bgls.embed, "embed_bg", std::mem::size_of::<EmbedParams>() as u64,
                         &[&m.buffers.w_embed, &m.buffers.act, &m.buffers.sample]);
        pass.set_pipeline(&m.pipes.embed);
        pass.set_bind_group(0, &bg, &[off]);
        pass.dispatch_workgroups(1, 1, 1);
        layer_pre_kv_in_pass(m, cfg, uniforms, &mut pass, 0, pos);
    }
    for il in 0..cfg.n_layer {
        let layer_offset_bytes = (il as u64) * (m.max_seq as u64) * cache_row_bytes;
        let dst_offset_bytes = layer_offset_bytes + (pos as u64) * cache_row_bytes;
        let k_src_bytes = (m.act_layout.k_cur as u64) * 4;
        let v_src_bytes = (m.act_layout.v_cur as u64) * 4;
        encoder.copy_buffer_to_buffer(&m.buffers.act, k_src_bytes, &m.buffers.kv_k, dst_offset_bytes, cache_row_bytes);
        encoder.copy_buffer_to_buffer(&m.buffers.act, v_src_bytes, &m.buffers.kv_v, dst_offset_bytes, cache_row_bytes);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("phase"), timestamp_writes: None,
        });
        layer_post_kv_in_pass(m, cfg, uniforms, &mut pass, il, pos);
        if il + 1 < cfg.n_layer {
            layer_pre_kv_in_pass(m, cfg, uniforms, &mut pass, il + 1, pos);
        } else {
            let on = tensor(cfg, "output_norm.weight").clone();
            dispatch_rms_norm(m, cfg, uniforms, &mut pass,
                              1, cfg.n_embd, m.act_layout.x, m.act_layout.x_norm,
                              (on.offset / 4) as u32);
            let lm_w = tensor(cfg, "token_embd.weight").clone();
            dispatch_matvec_q1_0(m, uniforms, &mut pass,
                                 cfg.n_embd, cfg.n_vocab,
                                 &m.buffers.w_embed, lm_w.d_offset as u32, lm_w.qs_offset as u32,
                                 m.act_layout.x_norm, m.act_layout.logits, false);
            let p = ArgmaxParams { n: cfg.n_vocab, in_offset: m.act_layout.logits, out_offset: sample_out, _p0: 0 };
            let off = uniforms.alloc(&p);
            let bg = make_bg(m, &m.bgls.argmax, "argmax_bg", std::mem::size_of::<ArgmaxParams>() as u64,
                             &[&m.buffers.act, &m.buffers.sample]);
            pass.set_pipeline(&m.pipes.argmax);
            pass.set_bind_group(0, &bg, &[off]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }
}

async fn step_matvec(model: &Model, cfg: &Config, token_id: u32, pos: u32, want_logits: bool) -> u32 {
    // CPU writes the input token to sample[0] before submitting.
    model.queue.write_buffer(&model.buffers.sample, 0, bytemuck::bytes_of(&token_id));
    let mut se = StepEncoder::new(model);
    encode_step_matvec(&mut se, cfg, 0, 0, pos);
    if want_logits {
        // argmax wrote sample[0]; copy to readback for CPU access.
        se.encoder.copy_buffer_to_buffer(&model.buffers.sample, 0, &model.buffers.readback, 0, 4);
    }
    let cb = se.finish();
    model.queue.submit(Some(cb));

    if !want_logits {
        // Wait for previous work to drain so caches/uniforms are stable.
        let _ = model.device.poll(wgpu::PollType::wait_indefinitely());
        return 0;
    }
    // Map readback
    let slice = model.buffers.readback.slice(..);
    let (s, r) = futures_intrusive::channel::shared::oneshot_channel::<()>();
    slice.map_async(wgpu::MapMode::Read, move |_| { let _ = s.send(()); });
    let _ = model.device.poll(wgpu::PollType::wait_indefinitely());
    r.receive().await;
    let data = slice.get_mapped_range();
    let id = bytemuck::pod_read_unaligned::<u32>(&data[..4]);
    drop(data);
    model.buffers.readback.unmap();
    id
}

/// Pre-KV-copy block of one layer: rms_norm → QKV fused → q/k norms → rope.
/// All dispatches go into the caller-provided open compute pass.
fn layer_pre_kv_in_pass(
    model: &Model, cfg: &Config, uniforms: &mut UniformPool,
    pass: &mut wgpu::ComputePass<'_>, il: u32, pos: u32,
) {
    // 2a. Pre-attn RMSNorm
    let attn_norm = tensor(cfg, &format!("blk.{il}.attn_norm.weight")).clone();
    dispatch_rms_norm(model, cfg, uniforms, pass, 1, cfg.n_embd,
                      model.act_layout.x, model.act_layout.x_norm, (attn_norm.offset / 4) as u32);

    // 2b. Q,K,V projections — fused 3-range
    let wq = tensor(cfg, &format!("blk.{il}.attn_q.weight")).clone();
    let wk = tensor(cfg, &format!("blk.{il}.attn_k.weight")).clone();
    let wv = tensor(cfg, &format!("blk.{il}.attn_v.weight")).clone();
    dispatch_matvec_q1_0_fused(model, uniforms, pass, cfg.n_embd, model.act_layout.x_norm,
                               &model.buffers.w_attn, &[
        (wq.d_offset as u32, wq.qs_offset as u32, cfg.q_dim,  model.act_layout.q),
        (wk.d_offset as u32, wk.qs_offset as u32, cfg.kv_dim, model.act_layout.k_cur),
        (wv.d_offset as u32, wv.qs_offset as u32, cfg.kv_dim, model.act_layout.v_cur),
    ]);

    // 2c. Per-head Q-norm and K-norm (BEFORE rope)
    let qn = tensor(cfg, &format!("blk.{il}.attn_q_norm.weight")).clone();
    dispatch_rms_norm(model, cfg, uniforms, pass, cfg.n_head, cfg.head_dim,
                      model.act_layout.q, model.act_layout.q, (qn.offset / 4) as u32);
    let kn = tensor(cfg, &format!("blk.{il}.attn_k_norm.weight")).clone();
    dispatch_rms_norm(model, cfg, uniforms, pass, cfg.n_kv_head, cfg.head_dim,
                      model.act_layout.k_cur, model.act_layout.k_cur, (kn.offset / 4) as u32);

    // 2d. RoPE
    dispatch_rope(model, cfg, uniforms, pass, model.act_layout.q,     1, cfg.n_head,    pos);
    dispatch_rope(model, cfg, uniforms, pass, model.act_layout.k_cur, 1, cfg.n_kv_head, pos);
}

/// Post-KV-copy block of one layer: attention → Wo (resid) → ffn_norm
/// → gate-up fused → silu_mul → Wd (resid).
fn layer_post_kv_in_pass(
    model: &Model, cfg: &Config, uniforms: &mut UniformPool,
    pass: &mut wgpu::ComputePass<'_>, il: u32, pos: u32,
) {
    let cache_row_bytes = (cfg.kv_dim * 4) as u64;
    let layer_offset_bytes = (il as u64) * (model.max_seq as u64) * cache_row_bytes;

    // 2f. Attention (single-token gen path)
    {
        let p = AttnParams {
            head_dim: cfg.head_dim, n_head: cfg.n_head, n_kv_head: cfg.n_kv_head,
            pos: pos + 1,
            kv_stride: cfg.kv_dim,
            q_offset: model.act_layout.q,
            k_cache_offset: ((layer_offset_bytes / 4) as u32),
            v_cache_offset: ((layer_offset_bytes / 4) as u32),
            out_offset: model.act_layout.attn_out,
            scale: 1.0 / (cfg.head_dim as f32).sqrt(),
            m_tokens: 1, is_prefill: 0,
        };
        let off = uniforms.alloc(&p);
        let bg = make_bg(model, &model.bgls.attn, "attn_bg", std::mem::size_of::<AttnParams>() as u64,
                         &[&model.buffers.act, &model.buffers.kv_k, &model.buffers.kv_v]);
        pass.set_pipeline(&model.pipes.attention);
        pass.set_bind_group(0, &bg, &[off]);
        pass.dispatch_workgroups(cfg.n_head, 1, 1);
    }

    // 2g. Wo with residual add
    let wo = tensor(cfg, &format!("blk.{il}.attn_output.weight")).clone();
    dispatch_matvec_q1_0(model, uniforms, pass, cfg.q_dim, cfg.n_embd,
                         &model.buffers.w_attn, wo.d_offset as u32, wo.qs_offset as u32,
                         model.act_layout.attn_out, model.act_layout.x, true /*accumulate*/);

    // 2h. FFN: pre-RMSNorm
    let fn_n = tensor(cfg, &format!("blk.{il}.ffn_norm.weight")).clone();
    dispatch_rms_norm(model, cfg, uniforms, pass, 1, cfg.n_embd,
                      model.act_layout.x, model.act_layout.x_norm, (fn_n.offset / 4) as u32);

    // 2i. Wgate, Wup fused
    let wg = tensor(cfg, &format!("blk.{il}.ffn_gate.weight")).clone();
    let wu = tensor(cfg, &format!("blk.{il}.ffn_up.weight")).clone();
    dispatch_matvec_q1_0_fused(model, uniforms, pass, cfg.n_embd, model.act_layout.x_norm,
                               &model.buffers.w_ffn_gu, &[
        (wg.d_offset as u32, wg.qs_offset as u32, cfg.n_ff, model.act_layout.gate),
        (wu.d_offset as u32, wu.qs_offset as u32, cfg.n_ff, model.act_layout.up),
    ]);

    // 2j. SiLU(gate) * up
    dispatch_silu_mul(model, uniforms, pass, cfg.n_ff, 1,
                      model.act_layout.gate, model.act_layout.up, model.act_layout.ffn_in);

    // 2k. Wd with residual add
    let wd = tensor(cfg, &format!("blk.{il}.ffn_down.weight")).clone();
    dispatch_matvec_q1_0(model, uniforms, pass, cfg.n_ff, cfg.n_embd,
                         &model.buffers.w_ffn_d, wd.d_offset as u32, wd.qs_offset as u32,
                         model.act_layout.ffn_in, model.act_layout.x, true /*accumulate*/);
}

// ---------- shader wrappers --------------------------------------------------

fn rms_norm(model: &Model, cfg: &Config, se: &mut StepEncoder,
            n_groups: u32, group_size: u32, in_off: u32, out_off: u32, w_off: u32) {
    let p = RmsNormParams {
        group_size, n_groups,
        input_offset: in_off, output_offset: out_off, weight_offset: w_off,
        eps: cfg.rms_eps,
        ..Default::default()
    };
    let off = se.alloc_uniform(&p);
    let bg = make_bg(model, &model.bgls.rms_norm, "rms_bg", std::mem::size_of::<RmsNormParams>() as u64,
                     &[&model.buffers.act, &model.buffers.w_norms]);
    let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("rms_norm"), timestamp_writes: None });
    cp.set_pipeline(&model.pipes.rms_norm);
    cp.set_bind_group(0, &bg, &[off]);
    cp.dispatch_workgroups(n_groups, 1, 1);
}

fn rope(model: &Model, cfg: &Config, se: &mut StepEncoder,
        data_off: u32, n_tokens: u32, n_heads: u32, pos_base: u32) {
    let p = RopeParams {
        head_dim: cfg.head_dim, n_heads, n_tokens, pos_base,
        data_offset: data_off, rope_table_offset: 0,
        ..Default::default()
    };
    let off = se.alloc_uniform(&p);
    let bg = make_bg(model, &model.bgls.rope, "rope_bg", std::mem::size_of::<RopeParams>() as u64,
                     &[&model.buffers.rope_table, &model.buffers.act]);
    let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("rope"), timestamp_writes: None });
    cp.set_pipeline(&model.pipes.rope_neox);
    cp.set_bind_group(0, &bg, &[off]);
    cp.dispatch_workgroups(n_tokens, n_heads, 1);
}

/// Fused multi-range matvec: 2 or 3 weight ranges, one shared input. Each
/// range's `n_rows` must be a multiple of 8 (ROWS_PER_WG) so that no
/// workgroup straddles a range boundary. Pass `n_2 = 0` for the 2-range case.
fn matvec_q1_0_fused(
    model: &Model, se: &mut StepEncoder,
    k: u32, input_offset: u32, weights: &wgpu::Buffer,
    ranges: &[(u32, u32, u32, u32)], // (d_offset, qs_offset, n_rows, output_offset)
) {
    debug_assert!(ranges.len() == 2 || ranges.len() == 3);
    for (_, _, n, _) in ranges { debug_assert!(n % 8 == 0); }
    let r = |i: usize| ranges.get(i).copied().unwrap_or((0, 0, 0, 0));
    let (d0, qs0, n0, o0) = r(0);
    let (d1, qs1, n1, o1) = r(1);
    let (d2, qs2, n2, o2) = r(2);
    let n_total = n0 + n1 + n2;
    const ROWS_PER_WG: u32 = 8;
    let n_wg = (n_total + ROWS_PER_WG - 1) / ROWS_PER_WG;
    let dispatch_x = n_wg.min(65535);
    let dispatch_y = (n_wg + dispatch_x - 1) / dispatch_x;
    let p = MatvecFusedParams {
        k, n_total, input_offset, dispatch_x_dim: dispatch_x,
        d_offset_0: d0, qs_offset_0: qs0, n_0: n0, output_offset_0: o0,
        d_offset_1: d1, qs_offset_1: qs1, n_1: n1, output_offset_1: o1,
        d_offset_2: d2, qs_offset_2: qs2, n_2: n2, output_offset_2: o2,
    };
    let off = se.alloc_uniform(&p);
    let bg = make_bg(model, &model.bgls.matvec, "matvec_fused_bg",
                     std::mem::size_of::<MatvecFusedParams>() as u64,
                     &[weights, &model.buffers.act]);
    let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("matvec_fused"), timestamp_writes: None });
    cp.set_pipeline(&model.pipes.matvec_fused);
    cp.set_bind_group(0, &bg, &[off]);
    cp.dispatch_workgroups(dispatch_x, dispatch_y, 1);
}

fn matvec_q1_0(model: &Model, _cfg: &Config, se: &mut StepEncoder,
               k: u32, n: u32,
               weights: &wgpu::Buffer, w_d: u32, w_qs: u32,
               in_off: u32, out_off: u32, accumulate: bool) {
    // Multi-row matvec: ROWS_PER_WG=8 rows per workgroup. Wgpu caps dispatch
    // per dim at 65535; wrap into a 2D grid for the LM-head case.
    const ROWS_PER_WG: u32 = 8;
    let n_wg = (n + ROWS_PER_WG - 1) / ROWS_PER_WG;
    let dispatch_x = n_wg.min(65535);
    let dispatch_y = (n_wg + dispatch_x - 1) / dispatch_x;
    let p = MatvecParams {
        k, n,
        d_offset: w_d, qs_offset: w_qs,
        input_offset: in_off, output_offset: out_off,
        accumulate: if accumulate { 1 } else { 0 },
        dispatch_x_dim: dispatch_x,
    };
    let off = se.alloc_uniform(&p);
    let bg = make_bg(model, &model.bgls.matvec, "matvec_bg", std::mem::size_of::<MatvecParams>() as u64,
                     &[weights, &model.buffers.act]);
    let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("matvec"), timestamp_writes: None });
    cp.set_pipeline(&model.pipes.matvec);
    cp.set_bind_group(0, &bg, &[off]);
    cp.dispatch_workgroups(dispatch_x, dispatch_y, 1);
}

// ---------- matmul (prompt prefill) ----------------------------------------

async fn prefill_matmul(model: &mut Model, cfg: &Config, prompt: &[u32]) -> u32 {
    let m = prompt.len() as u32;
    assert!(m <= model.m_max, "prompt {m} exceeds m_max={}", model.m_max);

    // ---- 1. Embed all M tokens (one shader call per token; cheap) ----------
    // Write all prompt token ids into sample[0..M] so each embed can read its
    // own row from sample[mi].
    let prompt_u32: Vec<u32> = prompt.iter().copied().collect();
    model.queue.write_buffer(&model.buffers.sample, 0, bytemuck::cast_slice(&prompt_u32));
    {
        let mut se = StepEncoder::new(model);
        let tok_embd = tensor(cfg, "token_embd.weight").clone();
        let mut pass = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("prefill_embed"), timestamp_writes: None });
        for (mi, _tid) in prompt.iter().enumerate() {
            let p = EmbedParams {
                k: cfg.n_embd,
                d_offset: tok_embd.d_offset as u32,
                qs_offset: tok_embd.qs_offset as u32,
                output_offset: model.act_layout.x,
                sample_offset: mi as u32,
                m_token: mi as u32,
                ..Default::default()
            };
            let off = se.uniforms.alloc(&p);
            let bg = make_bg(model, &model.bgls.embed, "embed_bg", std::mem::size_of::<EmbedParams>() as u64,
                             &[&model.buffers.w_embed, &model.buffers.act, &model.buffers.sample]);
            pass.set_pipeline(&model.pipes.embed);
            pass.set_bind_group(0, &bg, &[off]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        drop(pass);
        let cb = se.finish();
        model.queue.submit(Some(cb));
    }

    // ---- 2. Per-layer transformer using matmul (one CB for all layers) ----
    {
        let mut se = StepEncoder::new(model);
        for il in 0..cfg.n_layer {
            layer_step_matmul(model, cfg, &mut se, il, m);
        }
        let cb = se.finish();
        model.queue.submit(Some(cb));
    }

    // ---- 3. Final RMSNorm on each token's x; LM head matvec for last only --
    let mut se = StepEncoder::new(model);
    let on = tensor(cfg, "output_norm.weight").clone();
    rms_norm(model, cfg, &mut se, m, cfg.n_embd,
             model.act_layout.x, model.act_layout.x_norm, (on.offset / 4) as u32);

    // LM head matvec on the LAST token's x_norm only.
    let last_input_offset = model.act_layout.x_norm + (m - 1) * cfg.n_embd;
    let lm_w = tensor(cfg, "token_embd.weight").clone();
    matvec_q1_0(model, cfg, &mut se,
                cfg.n_embd, cfg.n_vocab,
                &model.buffers.w_embed, lm_w.d_offset as u32, lm_w.qs_offset as u32,
                last_input_offset, model.act_layout.logits, false);
    {
        let p = ArgmaxParams { n: cfg.n_vocab, in_offset: model.act_layout.logits, out_offset: 0, _p0: 0 };
        let off = se.alloc_uniform(&p);
        let bg = make_bg(model, &model.bgls.argmax, "argmax_bg", std::mem::size_of::<ArgmaxParams>() as u64,
                         &[&model.buffers.act, &model.buffers.sample]);
        let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("argmax"), timestamp_writes: None });
        cp.set_pipeline(&model.pipes.argmax);
        cp.set_bind_group(0, &bg, &[off]);
        cp.dispatch_workgroups(1, 1, 1);
    }
    se.encoder.copy_buffer_to_buffer(&model.buffers.sample, 0, &model.buffers.readback, 0, 4);
    let cb = se.finish();
    model.queue.submit(Some(cb));

    // Map readback
    let slice = model.buffers.readback.slice(..);
    let (s, r) = futures_intrusive::channel::shared::oneshot_channel::<()>();
    slice.map_async(wgpu::MapMode::Read, move |_| { let _ = s.send(()); });
    let _ = model.device.poll(wgpu::PollType::wait_indefinitely());
    r.receive().await;
    let id = bytemuck::pod_read_unaligned::<u32>(&slice.get_mapped_range()[..4]);
    model.buffers.readback.unmap();
    id
}

fn layer_step_matmul(model: &Model, cfg: &Config, se: &mut StepEncoder, il: u32, m: u32) {
    let pos_base = 0u32;  // prefill always starts at pos 0

    // 1. RMSNorm(attn_norm) for each token
    let an = tensor(cfg, &format!("blk.{il}.attn_norm.weight")).clone();
    rms_norm(model, cfg, se, m, cfg.n_embd,
             model.act_layout.x, model.act_layout.x_norm, (an.offset / 4) as u32);

    // 2. Quantize x_norm to Q8_0
    let (a_d_off, a_qs_off) = quantize_act(model, cfg, se, cfg.n_embd, m, model.act_layout.x_norm);

    // 3. Wq, Wk, Wv via matmul
    let wq = tensor(cfg, &format!("blk.{il}.attn_q.weight")).clone();
    matmul_q1_0(model, cfg, se, cfg.n_embd, cfg.q_dim, m,
                &model.buffers.w_attn, wq.d_offset as u32, wq.qs_offset as u32,
                a_d_off, a_qs_off, model.act_layout.q, false);
    let wk = tensor(cfg, &format!("blk.{il}.attn_k.weight")).clone();
    matmul_q1_0(model, cfg, se, cfg.n_embd, cfg.kv_dim, m,
                &model.buffers.w_attn, wk.d_offset as u32, wk.qs_offset as u32,
                a_d_off, a_qs_off, model.act_layout.k_cur, false);
    let wv = tensor(cfg, &format!("blk.{il}.attn_v.weight")).clone();
    matmul_q1_0(model, cfg, se, cfg.n_embd, cfg.kv_dim, m,
                &model.buffers.w_attn, wv.d_offset as u32, wv.qs_offset as u32,
                a_d_off, a_qs_off, model.act_layout.v_cur, false);

    // 4. Per-head Q-norm, K-norm (groups = M * n_head or M * n_kv_head)
    let qn = tensor(cfg, &format!("blk.{il}.attn_q_norm.weight")).clone();
    rms_norm(model, cfg, se, m * cfg.n_head, cfg.head_dim,
             model.act_layout.q, model.act_layout.q, (qn.offset / 4) as u32);
    let kn = tensor(cfg, &format!("blk.{il}.attn_k_norm.weight")).clone();
    rms_norm(model, cfg, se, m * cfg.n_kv_head, cfg.head_dim,
             model.act_layout.k_cur, model.act_layout.k_cur, (kn.offset / 4) as u32);

    // 5. RoPE
    rope(model, cfg, se, model.act_layout.q,     m, cfg.n_head,    pos_base);
    rope(model, cfg, se, model.act_layout.k_cur, m, cfg.n_kv_head, pos_base);

    // 6. Copy K, V to cache (M tokens at offsets [0, M))
    let cache_row_bytes = (cfg.kv_dim * 4) as u64;
    let layer_offset_bytes = (il as u64) * (model.max_seq as u64) * cache_row_bytes;
    let total_bytes = (m as u64) * cache_row_bytes;
    se.encoder.copy_buffer_to_buffer(&model.buffers.act, (model.act_layout.k_cur * 4) as u64,
                                     &model.buffers.kv_k, layer_offset_bytes, total_bytes);
    se.encoder.copy_buffer_to_buffer(&model.buffers.act, (model.act_layout.v_cur * 4) as u64,
                                     &model.buffers.kv_v, layer_offset_bytes, total_bytes);

    // 7. Attention: single batched dispatch of (n_head, M) WGs.
    {
        let p = AttnParams {
            head_dim: cfg.head_dim, n_head: cfg.n_head, n_kv_head: cfg.n_kv_head,
            pos: 0,                  // unused when is_prefill=1
            kv_stride: cfg.kv_dim,
            q_offset: model.act_layout.q,
            k_cache_offset: ((layer_offset_bytes / 4) as u32),
            v_cache_offset: ((layer_offset_bytes / 4) as u32),
            out_offset: model.act_layout.attn_out,
            scale: 1.0 / (cfg.head_dim as f32).sqrt(),
            m_tokens: m, is_prefill: 1,
        };
        let off = se.alloc_uniform(&p);
        let bg = make_bg(model, &model.bgls.attn, "attn_bg", std::mem::size_of::<AttnParams>() as u64,
                         &[&model.buffers.act, &model.buffers.kv_k, &model.buffers.kv_v]);
        let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("attn"), timestamp_writes: None });
        cp.set_pipeline(&model.pipes.attention);
        cp.set_bind_group(0, &bg, &[off]);
        cp.dispatch_workgroups(cfg.n_head, m, 1);
    }

    // 8. Quantize attn_out then Wo with residual add
    let (a_d2, a_qs2) = quantize_act(model, cfg, se, cfg.q_dim, m, model.act_layout.attn_out);
    let wo = tensor(cfg, &format!("blk.{il}.attn_output.weight")).clone();
    matmul_q1_0(model, cfg, se, cfg.q_dim, cfg.n_embd, m,
                &model.buffers.w_attn, wo.d_offset as u32, wo.qs_offset as u32,
                a_d2, a_qs2, model.act_layout.x, true);

    // 9. ffn_norm
    let fn_n = tensor(cfg, &format!("blk.{il}.ffn_norm.weight")).clone();
    rms_norm(model, cfg, se, m, cfg.n_embd,
             model.act_layout.x, model.act_layout.x_norm, (fn_n.offset / 4) as u32);

    // 10. Quantize x_norm again, then Wgate, Wup
    let (a_d3, a_qs3) = quantize_act(model, cfg, se, cfg.n_embd, m, model.act_layout.x_norm);
    let wg = tensor(cfg, &format!("blk.{il}.ffn_gate.weight")).clone();
    matmul_q1_0(model, cfg, se, cfg.n_embd, cfg.n_ff, m,
                &model.buffers.w_ffn_gu, wg.d_offset as u32, wg.qs_offset as u32,
                a_d3, a_qs3, model.act_layout.gate, false);
    let wu = tensor(cfg, &format!("blk.{il}.ffn_up.weight")).clone();
    matmul_q1_0(model, cfg, se, cfg.n_embd, cfg.n_ff, m,
                &model.buffers.w_ffn_gu, wu.d_offset as u32, wu.qs_offset as u32,
                a_d3, a_qs3, model.act_layout.up, false);

    // 11. SiLU(gate)*up
    silu_mul(model, se, cfg.n_ff, m, model.act_layout.gate, model.act_layout.up, model.act_layout.ffn_in);

    // 12. Quantize ffn_in then Wd with residual
    let (a_d4, a_qs4) = quantize_act(model, cfg, se, cfg.n_ff, m, model.act_layout.ffn_in);
    let wd = tensor(cfg, &format!("blk.{il}.ffn_down.weight")).clone();
    matmul_q1_0(model, cfg, se, cfg.n_ff, cfg.n_embd, m,
                &model.buffers.w_ffn_d, wd.d_offset as u32, wd.qs_offset as u32,
                a_d4, a_qs4, model.act_layout.x, true);
}

/// Layout (in bytes) of the q8 buffer: M * (K/32) FP32 d's, then M * K i8 qs's.
/// Returns (d_offset, qs_offset) in bytes.
fn quantize_act(model: &Model, _cfg: &Config, se: &mut StepEncoder,
                k: u32, m: u32, in_off_f32: u32) -> (u32, u32) {
    let nb_q8 = k / 32;
    let d_off  = 0u32;
    let qs_off = m * nb_q8 * 4;
    let total = m * nb_q8;
    let dispatch_x = total.min(65535);
    let dispatch_y = (total + dispatch_x - 1) / dispatch_x;
    let p = QuantParams {
        k, m, input_offset: in_off_f32,
        d_offset: d_off, qs_offset: qs_off,
        dispatch_x_dim: dispatch_x,
        ..Default::default()
    };
    let off = se.alloc_uniform(&p);
    let bg = make_bg(model, &model.bgls.quantize, "quant_bg", std::mem::size_of::<QuantParams>() as u64,
                     &[&model.buffers.act, &model.buffers.act_q8]);
    let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("quantize"), timestamp_writes: None });
    cp.set_pipeline(&model.pipes.quantize);
    cp.set_bind_group(0, &bg, &[off]);
    cp.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    (d_off, qs_off)
}

fn silu_mul(model: &Model, se: &mut StepEncoder,
            n: u32, m: u32, gate_off: u32, up_off: u32, out_off: u32) {
    let total = n * m;
    let groups = (total + 63) / 64;
    let dispatch_x = groups.min(65535);
    let dispatch_y = (groups + dispatch_x - 1) / dispatch_x;
    let p = SiluMulParams {
        n, m, gate_offset: gate_off, up_offset: up_off, out_offset: out_off,
        dispatch_x_count: dispatch_x * 64,
        ..Default::default()
    };
    let off = se.alloc_uniform(&p);
    let bg = make_bg(model, &model.bgls.silu_mul, "silu_bg", std::mem::size_of::<SiluMulParams>() as u64,
                     &[&model.buffers.act]);
    let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("silu_mul"), timestamp_writes: None });
    cp.set_pipeline(&model.pipes.silu_mul);
    cp.set_bind_group(0, &bg, &[off]);
    cp.dispatch_workgroups(dispatch_x, dispatch_y, 1);
}

fn matmul_q1_0(model: &Model, _cfg: &Config, se: &mut StepEncoder,
               k: u32, n: u32, m: u32,
               weights: &wgpu::Buffer, w_d: u32, w_qs: u32,
               a_d: u32, a_qs: u32, out_off: u32, accumulate: bool) {
    let p = MatmulParams {
        k, n, m,
        w_d_offset: w_d, w_qs_offset: w_qs,
        a_d_offset: a_d, a_qs_offset: a_qs,
        out_offset: out_off,
        accumulate: if accumulate { 1 } else { 0 },
        ..Default::default()
    };
    let off = se.alloc_uniform(&p);
    let bg = make_bg(model, &model.bgls.matmul, "matmul_bg", std::mem::size_of::<MatmulParams>() as u64,
                     &[weights, &model.buffers.act_q8, &model.buffers.act]);
    let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("matmul"), timestamp_writes: None });
    cp.set_pipeline(&model.pipes.matmul);
    cp.set_bind_group(0, &bg, &[off]);
    cp.dispatch_workgroups((n + 63) / 64, (m + 63) / 64, 1);
}
