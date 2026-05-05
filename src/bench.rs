//! Bench / microbench helpers, exposed only with the `bench-internals`
//! feature. Not part of the stable public API.
//!
//! Included as a child module of `forward` via `#[path]` so the helpers can
//! reach `forward`'s private items via `super::`.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::time::Instant;

use wgpu::PollType;

use super::{
    BenchMarker, BenchTimings, Model, Result, StepEncoder, encode_step_matvec, prefill_matmul_topk,
    step_matvec_no_sample, wait_topk_readback,
};
use crate::error::PotError;

pub fn bench(model: &Model, pp_n: u32, tg_n: u32, repeats: u32) -> Result<()> {
    let cfg = &model.cfg;
    eprintln!("--- bench: pp={pp_n}, tg={tg_n}, repeats={repeats} (after 1 warmup) ---");

    let prompt: Vec<u32> = (0..pp_n.min(model.m_max))
        .map(|i| (i % (cfg.n_vocab - 1)) + 1)
        .collect();

    // warm up
    let mut warm_marker = BenchMarker::new(model);
    let _ = prefill_matmul_topk(model, &prompt, 0, 1, &mut warm_marker)?;
    let _ = warm_marker.resolve()?;
    step_matvec_no_sample(model, 1u32, 0);
    if let Err(e) = model.device.poll(PollType::wait_indefinitely()) {
        model.check_device()?;
        return Err(PotError::Poll(e));
    }

    // ----- pp{pp_n} -----
    let mut pp_wall_ts: Vec<f32> = Vec::with_capacity(repeats as usize);
    let mut pp_gpu_ts: Vec<f32> = Vec::with_capacity(repeats as usize);
    for _ in 0..repeats {
        let t = Instant::now();
        let mut marker = BenchMarker::new(model);
        let _ = prefill_matmul_topk(model, &prompt, 0, 1, &mut marker)?;
        let gpu_ns = marker.resolve()?.total_ns();
        pp_wall_ts.push(pp_n as f32 / t.elapsed().as_secs_f32());
        pp_gpu_ts.push(pp_n as f32 / (gpu_ns / 1e9));
    }
    let (pp_wall_mean, pp_wall_std) = mean_std(&pp_wall_ts);
    let (pp_gpu_mean, pp_gpu_std) = mean_std(&pp_gpu_ts);

    // ----- tg{tg_n} -----
    // Non-pipelined: each step submits, resolves timestamps, reads back.
    // Wall-clock and GPU time are measured from the same loop so they're
    // directly comparable (the gap reveals per-step readback + poll overhead).
    let mut tg_wall_ts: Vec<f32> = Vec::with_capacity(repeats as usize);
    let mut tg_gpu_ts: Vec<f32> = Vec::with_capacity(repeats as usize);
    let tok: u32 = 1;
    for _ in 0..repeats {
        let t = Instant::now();
        let mut total_gpu_ns = 0.0f32;
        for pos in 0..tg_n {
            model
                .queue
                .write_buffer(&model.buffers.sample, 0, bytemuck::bytes_of(&tok));
            let mut se = StepEncoder::new(model);
            let mut marker = BenchMarker::new(model);
            encode_step_matvec(&mut se, cfg, 0, Some((0, 1)), pos, &mut marker);
            se.copy_sample_to_readback(8);
            let slot = se.schedule_topk_map(8);
            model.queue.submit(Some(se.finish()));
            wait_topk_readback(model, 1, slot)?;
            total_gpu_ns += marker.resolve()?.total_ns();
        }
        let wall_secs = t.elapsed().as_secs_f32();
        tg_wall_ts.push(tg_n as f32 / wall_secs);
        tg_gpu_ts.push(tg_n as f32 / (total_gpu_ns / 1e9));
    }
    let (tg_wall_mean, tg_wall_std) = mean_std(&tg_wall_ts);
    let (tg_gpu_mean, tg_gpu_std) = mean_std(&tg_gpu_ts);

    println!();
    println!("| backend           |         test |          wall t/s |           gpu t/s |");
    println!("| ----------------- | ------------ | ----------------: | ----------------: |");
    println!(
        "| bonsai-pot        |        pp{pp_n:<3} | {pp_wall_mean:>9.2} ± {pp_wall_std:>5.2} | {pp_gpu_mean:>9.2} ± {pp_gpu_std:>5.2} |"
    );
    println!(
        "| bonsai-pot        |        tg{tg_n:<3} | {tg_wall_mean:>9.2} ± {tg_wall_std:>5.2} | {tg_gpu_mean:>9.2} ± {tg_gpu_std:>5.2} |"
    );
    Ok(())
}

/// Per-kernel breakdown of one tg step at sequence position `pos`.
///
/// `pos` controls the realism of the attention measurement: at `pos=0`,
/// attention scans a single KV entry, which is unrepresentative. The KV cache
/// is pre-filled with `pos` no-readback steps before measurement so attention
/// sees `pos+1` cached tokens on each measured step.
pub fn microbench_tg(model: &Model, pos: u32, repeats: u32) -> Result<()> {
    if pos >= model.max_seq {
        return Err(PotError::ContextOverflow {
            pos,
            n: 1,
            max: model.max_seq,
        });
    }
    eprintln!("--- microbench tg (pos={pos}, repeats={repeats}) ---");

    // Pre-fill KV cache so attention scans `pos+1` cached tokens on each
    // measured step. `step_matvec_no_sample` skips the topk readback, and we
    // poll just once at the end.
    if pos > 0 {
        for p in 0..pos {
            step_matvec_no_sample(model, 1, p);
        }
        if let Err(e) = model.device.poll(PollType::wait_indefinitely()) {
            model.check_device()?;
            return Err(PotError::Poll(e));
        }
    }

    // warm up: one instrumented step at the measurement pos
    let _ = run_instrumented_step(model, pos)?;

    // Per-label, per-repeat aggregate: sum of all occurrences in one step
    // (i.e. n_layer for per-layer labels, 1 for globals). Storing per-step
    // sums (not per-occurrence) lets us report variance across steps.
    let mut per_step_label_ns: HashMap<&'static str, Vec<f32>> = HashMap::new();
    let mut calls_per_step: HashMap<&'static str, u32> = HashMap::new();
    let mut step_totals_ns: Vec<f32> = Vec::with_capacity(repeats as usize);

    for _ in 0..repeats {
        let timings = run_instrumented_step(model, pos)?;
        let mut step_label_sum: HashMap<&'static str, (u32, f32)> = HashMap::new();
        for (label, ns) in timings.spans() {
            let e = step_label_sum.entry(label).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += ns;
        }
        step_totals_ns.push(timings.total_ns());
        for (label, (calls, ns_sum)) in step_label_sum {
            per_step_label_ns.entry(label).or_default().push(ns_sum);
            calls_per_step.entry(label).or_insert(calls);
        }
    }

    // Build rows: (label, calls/step, per-call us, per-step ms mean, per-step ms std).
    let mut rows: Vec<(&'static str, u32, f32, f32, f32)> = per_step_label_ns
        .iter()
        .map(|(label, per_step_ns)| {
            let calls = calls_per_step[label];
            let (mean_ns, std_ns) = mean_std(per_step_ns);
            let per_call_us = mean_ns / calls as f32 / 1000.0;
            (*label, calls, per_call_us, mean_ns / 1e6, std_ns / 1e6)
        })
        .collect();
    rows.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(Ordering::Equal));

    let total_per_step_ms: f32 = rows.iter().map(|(_, _, _, ms, _)| ms).sum();

    println!();
    println!(
        "| kernel                                        | calls/step | per-call us |   per-step ms ± std | %step |"
    );
    println!(
        "|-----------------------------------------------|-----------:|------------:|--------------------:|------:|"
    );
    for (label, calls, per_call_us, per_step_ms, per_step_std_ms) in &rows {
        let pct = 100.0 * per_step_ms / total_per_step_ms;
        println!(
            "| {label:<45} | {calls:>10} | {per_call_us:>11.2} | {per_step_ms:>11.3} ± {per_step_std_ms:>5.3} | {pct:>5.1} |"
        );
    }
    println!(
        "|-----------------------------------------------|-----------:|------------:|--------------------:|------:|"
    );
    println!(
        "| TOTAL (sum of means)                          |            |             | {total_per_step_ms:>19.3} |       |"
    );

    let (step_mean_ns, step_std_ns) = mean_std(&step_totals_ns);
    let step_min_ms = step_totals_ns.iter().copied().fold(f32::INFINITY, f32::min) / 1e6;
    let step_max_ms = step_totals_ns.iter().copied().fold(0.0_f32, f32::max) / 1e6;
    let step_mean_ms = step_mean_ns / 1e6;
    let step_std_ms = step_std_ns / 1e6;
    println!();
    println!(
        "step time: {step_mean_ms:.3} ± {step_std_ms:.3} ms  (min {step_min_ms:.3}, max {step_max_ms:.3})  →  {:.1} t/s",
        1000.0 / step_mean_ms
    );
    Ok(())
}

/// Run one instrumented matvec step at `pos`, returning the resolved GPU timings.
fn run_instrumented_step(model: &Model, pos: u32) -> Result<BenchTimings> {
    let tok: u32 = 1;
    model
        .queue
        .write_buffer(&model.buffers.sample, 0, bytemuck::bytes_of(&tok));
    let mut se = StepEncoder::new(model);
    let mut marker = BenchMarker::new(model);
    encode_step_matvec(&mut se, &model.cfg, 0, Some((0, 1)), pos, &mut marker);
    se.copy_sample_to_readback(8);
    let slot = se.schedule_topk_map(8);
    model.queue.submit(Some(se.finish()));
    wait_topk_readback(model, 1, slot)?;
    marker.resolve()
}

fn mean_std(xs: &[f32]) -> (f32, f32) {
    let mean = xs.iter().sum::<f32>() / xs.len() as f32;
    let std = if xs.len() < 2 {
        0.0
    } else {
        let var: f32 =
            xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / (xs.len() - 1) as f32;
        var.sqrt()
    };
    (mean, std)
}
