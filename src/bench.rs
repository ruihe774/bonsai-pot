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
use crate::session::{GenerateOptions, Sampler};

// ChatML-wrapped: system="You are a helpful assistant.", user="Write 200 words
// about the Roman Empire." Encoded once; vocab is identical for 4B and 8B.
#[allow(clippy::unreadable_literal, reason = "encoded tokens")]
const E2E_PROMPT: &[u32] = &[
    151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 7985, 220,
    17, 15, 15, 4244, 911, 279, 12751, 20448, 13, 151645, 198, 151644, 77091, 198,
];

// ChatML-wrapped: system="You are a helpful assistant.", user=<Roman Empire
// article, ~330 tokens> + "Summarize the above in one word." Total: 366 tokens.
// Used by e2e_pp to bench Session::prefill end-to-end on a real-length prompt.
#[allow(clippy::unreadable_literal, reason = "encoded tokens")]
const E2E_PP_PROMPT: &[u32] = &[
    151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 785, 12751,
    20448, 572, 279, 1736, 12, 68455, 1584, 315, 13833, 21718, 323, 374, 8789, 15985, 311, 3076,
    279, 4168, 323, 17971, 21286, 553, 279, 48717, 2701, 4915, 59178, 594, 24335, 315, 13309, 5912,
    1212, 279, 44529, 349, 304, 220, 17, 22, 18040, 13, 1084, 5230, 264, 3460, 51382, 5537, 2163,
    279, 37685, 15029, 304, 4505, 11, 4787, 10174, 11, 323, 10867, 13622, 11, 323, 572, 21286, 553,
    976, 712, 1087, 13, 576, 4399, 315, 279, 10867, 12751, 20448, 304, 220, 19, 22, 21, 9630, 11,
    892, 12864, 279, 835, 315, 13833, 21718, 11, 374, 825, 315, 279, 1429, 5089, 4357, 304, 3738,
    3840, 382, 1655, 1181, 12196, 12818, 11, 279, 12751, 20448, 9761, 220, 20, 3526, 9334, 51857,
    323, 5644, 60129, 916, 458, 12943, 220, 22, 15, 3526, 1251, 11, 518, 429, 882, 220, 17, 16,
    817, 2889, 315, 279, 1879, 594, 4453, 7042, 13, 576, 12751, 20448, 572, 4221, 279, 1429, 7988,
    6955, 11, 12752, 11, 4948, 11, 323, 6277, 8437, 304, 279, 1879, 315, 1181, 882, 13, 1084, 572,
    825, 315, 279, 7772, 976, 18968, 304, 1879, 3840, 13, 576, 1156, 1378, 23631, 315, 279, 31347,
    594, 13885, 1033, 264, 4168, 315, 29969, 19753, 323, 43102, 3881, 438, 279, 70321, 11774, 3362,
    11, 892, 46918, 15901, 438, 330, 60980, 25803, 2217, 60980, 8232, 572, 6351, 11, 7299, 12,
    80418, 11, 323, 2745, 49823, 13, 19458, 572, 279, 24456, 4128, 315, 3033, 323, 2329, 11, 714,
    17860, 14616, 13570, 21355, 304, 279, 23149, 39921, 13, 576, 48717, 7881, 17646, 11, 2329, 11,
    14667, 11, 323, 34086, 5942, 429, 3060, 311, 10173, 10867, 8267, 7923, 311, 419, 1899, 13,
    12751, 19241, 11, 15355, 80500, 82, 11, 323, 13702, 7146, 15978, 63301, 315, 14667, 13, 576,
    5777, 16170, 20329, 1870, 1212, 279, 12751, 20448, 1212, 79, 20561, 6481, 5777, 5942, 3941,
    4505, 323, 7797, 382, 785, 31347, 9583, 6718, 1119, 279, 10867, 12751, 20448, 323, 279, 18028,
    12751, 20448, 11, 1083, 3881, 438, 279, 81660, 38357, 20448, 11, 892, 25882, 3080, 220, 16, 19,
    20, 18, 9630, 382, 9190, 5612, 551, 279, 3403, 304, 825, 3409, 13, 151645, 198, 151644, 77091,
    198,
];

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
            let mut se = StepEncoder::new(model);
            se.write_sample(0, bytemuck::bytes_of(&tok));
            let mut marker = BenchMarker::new(model);
            encode_step_matvec(&mut se, cfg, 0, Some((0, 1)), pos, &mut marker);
            se.copy_sample_to_readback(8);
            let slot = se.schedule_topk_map(8);
            let cb = se.finish();
            model.belt_finish();
            model.queue.submit(Some(cb));
            model.belt_recall();
            wait_topk_readback(model, 1, slot)?;
            total_gpu_ns += marker.resolve()?.total_ns();
        }
        let wall_secs = t.elapsed().as_secs_f32();
        tg_wall_ts.push(tg_n as f32 / wall_secs);
        tg_gpu_ts.push(tg_n as f32 / (total_gpu_ns / 1e9));
    }
    let (tg_wall_mean, tg_wall_std) = mean_std(&tg_wall_ts);
    let (tg_gpu_mean, tg_gpu_std) = mean_std(&tg_gpu_ts);

    // ----- e2e_tg -----
    // Wall-clock only: prefill is untimed; we time Session::generate exclusively.
    // e2e_tg_n counts actual tokens produced: 1 (from prefill) + generate output.
    let e2e_sampler = Sampler::default(); // temperature=1.0, stochastic
    let opts = GenerateOptions {
        max_new_tokens: tg_n.saturating_sub(1),
        sampler: e2e_sampler.clone(),
        ..Default::default()
    };
    let mut e2e_tg_wall_ts: Vec<f32> = Vec::with_capacity(repeats as usize);

    // warmup — also establishes e2e_tg_n from the actual output length
    let e2e_tg_n = {
        let mut sess = model.new_session();
        let first = sess.prefill_one_at_a_time(E2E_PROMPT, &e2e_sampler)?;
        let (generated, _) = sess.generate(first, &opts)?;
        1 + generated.len() as u32
    };

    for _ in 0..repeats {
        let mut sess = model.new_session();
        let first = sess.prefill_one_at_a_time(E2E_PROMPT, &e2e_sampler)?;
        let t = Instant::now();
        let (generated, _) = sess.generate(first, &opts)?;
        let actual_n = 1 + generated.len() as u32;
        e2e_tg_wall_ts.push(actual_n as f32 / t.elapsed().as_secs_f32());
    }
    let (e2e_tg_wall_mean, e2e_tg_wall_std) = mean_std(&e2e_tg_wall_ts);

    // ----- e2e_pp -----
    // Wall-clock only: times Session::prefill end-to-end on a real long-article
    // prompt (ChatML-wrapped, 366 tokens). Includes matmul prefill + topk
    // readback + CPU sample of the one-word answer token.
    let pp_e2e_n = E2E_PP_PROMPT.len() as u32;
    let mut e2e_pp_wall_ts: Vec<f32> = Vec::with_capacity(repeats as usize);

    // warmup
    {
        let mut sess = model.new_session();
        sess.prefill(E2E_PP_PROMPT, &e2e_sampler)?;
    }

    for _ in 0..repeats {
        let mut sess = model.new_session();
        let t = Instant::now();
        sess.prefill(E2E_PP_PROMPT, &e2e_sampler)?;
        e2e_pp_wall_ts.push(pp_e2e_n as f32 / t.elapsed().as_secs_f32());
    }
    let (e2e_pp_wall_mean, e2e_pp_wall_std) = mean_std(&e2e_pp_wall_ts);

    println!();
    println!("| backend           |         test |          wall t/s |           gpu t/s |");
    println!("| ----------------- | ------------ | ----------------: | ----------------: |");
    println!(
        "| bonsai-pot        |        pp{pp_n:<3} | {pp_wall_mean:>9.2} ± {pp_wall_std:>5.2} | {pp_gpu_mean:>9.2} ± {pp_gpu_std:>5.2} |"
    );
    println!(
        "| bonsai-pot        |        tg{tg_n:<3} | {tg_wall_mean:>9.2} ± {tg_wall_std:>5.2} | {tg_gpu_mean:>9.2} ± {tg_gpu_std:>5.2} |"
    );
    println!(
        "| bonsai-pot        |    e2e_pp{pp_e2e_n:<3} | {e2e_pp_wall_mean:>9.2} ± {e2e_pp_wall_std:>5.2} |                 — |"
    );
    println!(
        "| bonsai-pot        |    e2e_tg{e2e_tg_n:<3} | {e2e_tg_wall_mean:>9.2} ± {e2e_tg_wall_std:>5.2} |                 — |"
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
    let mut se = StepEncoder::new(model);
    se.write_sample(0, bytemuck::bytes_of(&tok));
    let mut marker = BenchMarker::new(model);
    encode_step_matvec(&mut se, &model.cfg, 0, Some((0, 1)), pos, &mut marker);
    se.copy_sample_to_readback(8);
    let slot = se.schedule_topk_map(8);
    let cb = se.finish();
    model.belt_finish();
    model.queue.submit(Some(cb));
    model.belt_recall();
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
