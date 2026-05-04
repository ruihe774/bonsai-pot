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
    BenchMarker, Model, Result, StepEncoder, encode_step_matvec, prefill_matmul_topk,
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
        let spans = marker.resolve()?;
        let gpu_ns: f32 = spans.iter().map(|(_, ns)| ns).sum();
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
            model.queue.submit(Some(se.finish()));
            wait_topk_readback(model, 1)?;
            let spans = marker.resolve()?;
            total_gpu_ns += spans.iter().map(|(_, ns)| ns).sum::<f32>();
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

pub fn microbench_tg(model: &Model, repeats: u32) -> Result<()> {
    let cfg = &model.cfg;
    eprintln!("--- microbench tg (repeats={repeats}) ---");

    // warm up: one instrumented step
    {
        let tok: u32 = 1;
        model
            .queue
            .write_buffer(&model.buffers.sample, 0, bytemuck::bytes_of(&tok));
        let mut se = StepEncoder::new(model);
        let mut marker = BenchMarker::new(model);
        encode_step_matvec(&mut se, cfg, 0, Some((0, 1)), 0, &mut marker);
        se.copy_sample_to_readback(8);
        model.queue.submit(Some(se.finish()));
        wait_topk_readback(model, 1)?;
        let _ = marker.resolve()?;
    }

    // Accumulate per-label duration in ns across repeats.
    // HashMap<label, Vec<duration_ns_for_one_occurrence>>
    // Each key appears n_layer times per step (for per-layer kernels) or 1× (global).
    let mut accum: HashMap<&'static str, Vec<f32>> = HashMap::new();
    let tok: u32 = 1;

    for _ in 0..repeats {
        model
            .queue
            .write_buffer(&model.buffers.sample, 0, bytemuck::bytes_of(&tok));
        let mut se = StepEncoder::new(model);
        let mut marker = BenchMarker::new(model);
        encode_step_matvec(&mut se, cfg, 0, Some((0, 1)), 0, &mut marker);
        se.copy_sample_to_readback(8);
        model.queue.submit(Some(se.finish()));
        wait_topk_readback(model, 1)?;
        let spans = marker.resolve()?;
        for (label, ns) in spans {
            accum.entry(label).or_default().push(ns);
        }
    }

    // Compute per-call stats. Each label appears (repeats * calls_per_step) times in the Vec.
    // Determine calls_per_step by counting unique occurrences in one step.
    // Since each step produces n_layer occurrences of per-layer labels and 1 of globals,
    // divide total count by repeats.
    let mut rows: Vec<(&'static str, u32, f32)> = accum
        .iter()
        .map(|(label, ns_vec)| {
            let calls_per_step = ns_vec.len() as u32 / repeats;
            let per_call_us = ns_vec.iter().sum::<f32>() / ns_vec.len() as f32 / 1000.0;
            (*label, calls_per_step, per_call_us)
        })
        .collect();

    // Sort by total per-step us descending.
    rows.sort_by(|a, b| {
        let ma = a.1 as f32 * a.2;
        let mb = b.1 as f32 * b.2;
        mb.partial_cmp(&ma).unwrap_or(Ordering::Equal)
    });

    let total_per_step_ms: f32 = rows
        .iter()
        .map(|(_, calls, us)| *calls as f32 * us / 1000.0)
        .sum();

    println!();
    println!(
        "| kernel                                        | calls/step | per-call us | per-step ms | %step |"
    );
    println!(
        "|-----------------------------------------------|-----------:|------------:|------------:|------:|"
    );
    for (label, calls, per_call_us) in &rows {
        let per_step_ms = *calls as f32 * per_call_us / 1000.0;
        let pct = 100.0 * per_step_ms / total_per_step_ms;
        println!(
            "| {label:<45} | {calls:>10} | {per_call_us:>11.2} | {per_step_ms:>11.3} | {pct:>5.1} |"
        );
    }
    println!(
        "|-----------------------------------------------|-----------:|------------:|------------:|------:|"
    );
    println!(
        "| TOTAL                                         |            |             | {total_per_step_ms:>11.3} |       |"
    );
    println!();
    println!(
        "gpu step time (sum of ts deltas): {total_per_step_ms:.3} ms  ({:.1} t/s)",
        1000.0 / total_per_step_ms
    );
    Ok(())
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
