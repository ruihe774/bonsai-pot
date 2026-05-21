//! Bench / microbench helpers, exposed only with the `bench-internals`
//! feature. Not part of the stable public API.
//!
//! Included as a child module of `forward` via `#[path]` so the helpers can
//! reach `forward`'s private items via `super::`.

#[cfg(all(not(feature = "ci"), not(target_vendor = "apple")))]
use std::cmp::Ordering;
#[cfg(all(not(feature = "ci"), not(target_vendor = "apple")))]
use std::collections::HashMap;
use std::time::Instant;

#[cfg(all(not(feature = "ci"), not(target_vendor = "apple")))]
use wgpu::PollType;

#[cfg(not(feature = "ci"))]
use super::BenchMarker;
#[cfg(any(feature = "ci", not(target_vendor = "apple")))]
use super::NoMarker;
#[cfg(all(not(feature = "ci"), not(target_vendor = "apple")))]
use super::{MicroMarker, Session};
use super::{
    Result, StepEncoder, encode_step_matvec, prefill_matmul_topk, step_matvec_no_sample,
    upload_sample, wait_topk_readback,
};
#[cfg(all(not(feature = "ci"), not(target_vendor = "apple")))]
use crate::error::PotError;
use crate::model::Model;
use crate::session::{GenerateOptions, Sampler};

// ChatML-wrapped: system instructs the model to write very long responses,
// user="Write an extremely detailed essay of at least 10000 words about the
// history of the Roman Empire. ...". The system prompt + the explicit
// "do not stop" instruction keeps the model from emitting <|im_end|> early so
// e2e_tg actually produces the requested number of tokens.
#[allow(clippy::unreadable_literal, reason = "encoded tokens")]
const E2E_PROMPT: &[u32] = &[
    151644, 8948, 198, 2610, 525, 264, 10950, 17847, 429, 13914, 1602, 1293, 11, 11682, 14507, 13,
    14695, 2936, 4124, 13, 151645, 198, 151644, 872, 198, 7985, 458, 9016, 11682, 8895, 315, 518,
    3245, 220, 16, 15, 15, 15, 15, 4244, 911, 279, 3840, 315, 279, 12751, 20448, 13, 17757, 1449,
    9294, 304, 7990, 13, 15003, 4378, 2041, 22535, 3080, 498, 5545, 279, 2169, 3084, 13, 151645,
    198, 151644, 77091, 198,
];

// ChatML-wrapped: system="You are a helpful assistant.", user=<Roman Empire
// article, ~600 tokens> + "Please summarize the above passage in a single word."
// Total: 632 tokens. Used by e2e_pp to bench Session::prefill end-to-end on a
// real-length prompt that exceeds m_max (512), so the prefill chunks at least
// twice and exercises the cross-chunk pos_base advancement.
#[allow(clippy::unreadable_literal, reason = "encoded tokens")]
const E2E_PP_PROMPT: &[u32] = &[
    151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 785, 12751,
    20448, 572, 279, 1736, 12, 68455, 1584, 315, 13833, 21718, 323, 374, 8789, 15985, 311, 3076,
    279, 4168, 323, 17971, 21286, 553, 279, 48717, 2701, 4915, 59178, 594, 24335, 315, 13309, 5912,
    1212, 279, 44529, 349, 304, 220, 17, 22, 18040, 13, 1084, 5230, 264, 3460, 13348, 315, 4505,
    11, 5310, 5671, 2163, 279, 37685, 15029, 11, 2670, 5479, 315, 4787, 10174, 11, 10867, 13622,
    11, 323, 16926, 4505, 13, 2411, 1181, 12196, 51382, 12818, 11, 432, 9761, 1045, 220, 20, 3526,
    9334, 51857, 323, 5644, 60129, 916, 458, 12943, 220, 22, 15, 3526, 1251, 11, 518, 429, 882,
    1045, 220, 17, 16, 3266, 315, 279, 1879, 594, 4453, 7042, 13, 576, 12751, 20448, 7881, 504,
    279, 12751, 5429, 979, 69245, 53653, 323, 6156, 355, 53653, 23507, 432, 504, 264, 34444, 1119,
    264, 86049, 13, 21718, 8643, 1181, 12196, 51382, 12818, 1212, 17298, 22838, 320, 265, 1542,
    220, 24, 23, 12, 16, 16, 22, 9630, 701, 37532, 287, 458, 3082, 315, 220, 20, 3526, 9334, 51857,
    13, 576, 12751, 7042, 518, 429, 882, 702, 1012, 12943, 518, 220, 20, 15, 311, 220, 24, 15,
    3526, 39671, 11, 476, 17267, 220, 17, 15, 4, 315, 279, 1879, 594, 7042, 518, 279, 882, 13, 576,
    4399, 315, 279, 10867, 12751, 20448, 304, 220, 19, 22, 21, 9630, 11, 892, 12864, 279, 835, 315,
    60286, 487, 11, 374, 825, 315, 279, 1429, 5089, 4357, 304, 3738, 3840, 13, 576, 12751, 20448,
    572, 4221, 279, 1429, 7988, 6955, 11, 12752, 11, 4948, 11, 323, 6277, 8437, 304, 279, 1879,
    315, 1181, 882, 13, 1084, 572, 825, 315, 279, 7772, 976, 18968, 304, 1879, 3840, 13, 576, 1156,
    1378, 23631, 315, 279, 20448, 594, 13885, 1033, 264, 4168, 315, 29969, 19753, 323, 43102, 3881,
    438, 279, 70321, 11774, 3362, 11, 892, 62457, 46918, 438, 12751, 25803, 13, 12751, 8232, 572,
    6351, 11, 7299, 1786, 494, 3073, 11, 323, 1602, 73482, 13, 19458, 572, 279, 24456, 4128, 315,
    8567, 323, 2329, 11, 714, 17860, 14616, 13570, 21355, 304, 279, 23149, 39921, 13, 576, 48717,
    7881, 17646, 11, 2329, 11, 14667, 11, 323, 4948, 5942, 429, 3060, 311, 10173, 10867, 34917,
    311, 419, 1899, 13, 12751, 19241, 11, 77756, 14389, 11, 323, 15355, 80500, 82, 7146, 9434,
    65597, 315, 14667, 13, 576, 5777, 16170, 20329, 1870, 1212, 279, 12751, 20448, 1212, 13273,
    6481, 5777, 5942, 3941, 4505, 323, 7797, 13, 576, 20448, 9583, 6718, 1119, 279, 10867, 12751,
    20448, 323, 279, 18028, 12751, 20448, 11, 1083, 3881, 438, 279, 81660, 38357, 20448, 11, 892,
    35413, 3080, 220, 16, 19, 20, 18, 9630, 13, 12751, 13587, 11, 19128, 11, 323, 7674, 8865, 6814,
    279, 37685, 323, 3060, 311, 10173, 10867, 3381, 13, 576, 17704, 315, 279, 10867, 4279, 374,
    29606, 311, 1657, 9363, 2670, 6955, 34565, 11, 916, 265, 742, 681, 389, 20362, 9327, 11, 6277,
    17460, 28210, 11, 3033, 21252, 323, 4948, 55299, 11, 323, 279, 82426, 315, 279, 12751, 2472,
    908, 13, 576, 18028, 4279, 33583, 1753, 315, 12751, 2329, 323, 17860, 7674, 369, 2441, 16183,
    1635, 13, 91187, 1164, 11, 18047, 553, 19305, 482, 279, 8513, 304, 220, 18, 18, 15, 9630, 11,
    10223, 438, 279, 6722, 315, 279, 18028, 12751, 20448, 323, 572, 825, 315, 279, 7772, 323,
    92018, 9720, 304, 279, 1879, 2337, 1181, 16162, 13, 576, 4399, 315, 91187, 1164, 304, 220, 16,
    19, 20, 18, 12864, 279, 835, 315, 279, 18028, 12751, 20448, 323, 279, 7167, 315, 264, 501,
    11385, 304, 7513, 3840, 13, 576, 19588, 315, 279, 12751, 20448, 374, 12767, 323, 9539, 311,
    6083, 6481, 34917, 304, 27601, 5510, 11, 504, 4128, 323, 2329, 311, 17646, 323, 4948, 3381,
    382, 5501, 62079, 279, 3403, 21085, 304, 264, 3175, 3409, 13, 151645, 198, 151644, 77091, 198,
];

pub fn bench(model: &Model, pp_n: u32, tg_n: u32, repeats: u32) -> Result<()> {
    let cfg = &model.cfg;
    eprintln!("--- bench: pp={pp_n}, tg={tg_n}, repeats={repeats} (after 1 warmup) ---");

    let prompt: Vec<u32> = (0..pp_n).map(|i| (i % (cfg.n_vocab - 1)) + 1).collect();
    let m_max = model.m_max as usize;

    // Reuse one session for all pp/tg measurements. The e2e tests below open
    // their own (sequential) sessions because they explicitly time
    // Session::prefill / Session::generate from a fresh state.
    let bench_sess = model.new_session();

    // Run a full prefill of `prompt`, chunking into `m_max`-sized batches so any
    // `pp_n` is supported. Returns the total GPU ns summed across chunks.
    let run_full_prefill = |prompt: &[u32]| -> Result<f32> {
        #[allow(unused, reason = "conditional compilation")]
        let mut total_gpu_ns = 0.0f32;
        let mut pos_base = 0u32;
        for slice in prompt.chunks(m_max) {
            #[cfg(not(feature = "ci"))]
            let mut marker = BenchMarker::new(&bench_sess);
            #[cfg(feature = "ci")]
            let mut marker = NoMarker;
            let _ = prefill_matmul_topk(&bench_sess, slice, pos_base, 1, &mut marker)?;
            #[cfg(not(feature = "ci"))]
            #[allow(
                clippy::let_unit_value,
                clippy::ignored_unit_patterns,
                reason = "conditional compilation"
            )]
            let _ = total_gpu_ns += marker.resolve()?;
            pos_base += slice.len() as u32;
        }
        Ok(total_gpu_ns)
    };

    // warm up
    let _ = run_full_prefill(&prompt)?;
    step_matvec_no_sample(&bench_sess, 1u32, 0);

    // ----- pp{pp_n} -----
    let mut pp_wall_ts: Vec<f32> = Vec::with_capacity(repeats as usize);
    let mut pp_gpu_ts: Vec<f32> = Vec::with_capacity(repeats as usize);
    for _ in 0..repeats {
        let t = Instant::now();
        let gpu_ns = run_full_prefill(&prompt)?;
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
        #[allow(unused, reason = "conditional compilation")]
        let mut total_gpu_ns = 0.0f32;
        for pos in 0..tg_n {
            let mut se = StepEncoder::new(&bench_sess);
            upload_sample(&bench_sess, &mut se.encoder, 0, bytemuck::bytes_of(&tok));
            #[cfg(not(feature = "ci"))]
            let mut marker = BenchMarker::new(&bench_sess);
            #[cfg(feature = "ci")]
            let mut marker = NoMarker;
            encode_step_matvec(&mut se, cfg, 0, Some((0, 1)), pos, &mut marker);
            se.copy_sample_to_readback(8);
            se.schedule_topk_map(8);
            model.queue.submit(Some(se.finish()));
            wait_topk_readback(&bench_sess, 1)?;
            #[cfg(not(feature = "ci"))]
            #[allow(
                clippy::let_unit_value,
                clippy::ignored_unit_patterns,
                reason = "conditional compilation"
            )]
            let _ = total_gpu_ns += marker.resolve()?;
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
    // Override `stop_pred` to disable EOS termination — the prompt is engineered
    // to keep the model writing, but a stochastic sample of <|im_end|> would
    // still cut tg short and inflate the reported t/s.
    let e2e_sampler = Sampler::default(); // temperature=1.0, stochastic
    let never_stop: fn(u32) -> bool = |_| false;
    let opts = GenerateOptions {
        max_new_tokens: tg_n.saturating_sub(1),
        sampler: e2e_sampler.clone(),
        stop_pred: Some(never_stop),
    };
    let mut e2e_tg_wall_ts: Vec<f32> = Vec::with_capacity(repeats as usize);

    // warmup — also establishes e2e_tg_n from the actual output length
    let e2e_tg_n = {
        let mut sess = model.new_session();
        let first = sess.prefill(E2E_PROMPT, &e2e_sampler)?;
        let (generated, _) = sess.generate(first, &opts)?;
        1 + generated.len() as u32
    };

    for _ in 0..repeats {
        let mut sess = model.new_session();
        let first = sess.prefill(E2E_PROMPT, &e2e_sampler)?;
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
#[cfg(all(not(feature = "ci"), not(target_vendor = "apple")))]
pub fn microbench_tg(model: &Model, pos: u32, repeats: u32, no_marker: bool) -> Result<()> {
    if pos >= model.max_seq {
        return Err(PotError::ContextOverflow {
            pos,
            n: 1,
            max: model.max_seq,
        });
    }
    eprintln!(
        "--- microbench tg (pos={pos}, repeats={repeats}{}) ---",
        if no_marker { ", NO MARKER" } else { "" }
    );

    let session = model.new_session();

    // Pre-fill KV cache so attention scans `pos+1` cached tokens on each
    // measured step. `step_matvec_no_sample` skips the topk readback but polls
    // internally to drive the staging-buffer remap callback.
    for p in 0..pos {
        step_matvec_no_sample(&session, 1, p);
    }

    // No-marker mode: switch to NoMarker so per-step timing isn't recorded.
    // Each measurement is a single submit (no extra resolve submit), giving
    // deterministic submit indices for external profilers like Nsight Graphics
    // GPU Trace, which select work via --start-after-submits / --limit-to-submits.
    if no_marker {
        // warmup
        run_uninstrumented_step(&session, pos)?;
        eprintln!(">>> PROFILE BEGIN: tg measurement ({repeats} submit(s)) <<<");
        for _ in 0..repeats {
            run_uninstrumented_step(&session, pos)?;
        }
        eprintln!(">>> PROFILE END: tg measurement <<<");
        return Ok(());
    }

    // warm up: one instrumented step at the measurement pos
    let _ = run_instrumented_step(&session, pos)?;

    // Per-label, per-repeat aggregate: sum of all occurrences in one step
    // (i.e. n_layer for per-layer labels, 1 for globals). Storing per-step
    // sums (not per-occurrence) lets us report variance across steps.
    let mut per_step_label_ns: HashMap<&'static str, Vec<f32>> = HashMap::new();
    let mut calls_per_step: HashMap<&'static str, u32> = HashMap::new();
    let mut step_totals_ns: Vec<f32> = Vec::with_capacity(repeats as usize);

    for _ in 0..repeats {
        let spans = run_instrumented_step(&session, pos)?;
        let mut step_label_sum: HashMap<&'static str, (u32, f32)> = HashMap::new();
        for (label, ns) in &spans {
            let e = step_label_sum.entry(*label).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += ns;
        }
        step_totals_ns.push(spans.iter().map(|(_, ns)| ns).sum());
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

/// Run one instrumented matvec step at `pos`, returning per-span GPU durations.
#[cfg(all(not(feature = "ci"), not(target_vendor = "apple")))]
fn run_instrumented_step(session: &Session<'_>, pos: u32) -> Result<Vec<(&'static str, f32)>> {
    let tok: u32 = 1;
    let mut se = StepEncoder::new(session);
    upload_sample(session, &mut se.encoder, 0, bytemuck::bytes_of(&tok));
    let mut marker = MicroMarker::new(session);
    encode_step_matvec(
        &mut se,
        &session.model.cfg,
        0,
        Some((0, 1)),
        pos,
        &mut marker,
    );
    se.copy_sample_to_readback(8);
    se.schedule_topk_map(8);
    session.model.queue.submit(Some(se.finish()));
    wait_topk_readback(session, 1)?;
    marker.resolve()
}

/// Like [`run_instrumented_step`] but with no per-kernel timestamp marker —
/// used in `--no-marker` profiling mode so the measurement is exactly one submit
/// with no resolve submit afterward.
#[cfg(all(not(feature = "ci"), not(target_vendor = "apple")))]
fn run_uninstrumented_step(session: &Session<'_>, pos: u32) -> Result<()> {
    let tok: u32 = 1;
    let mut se = StepEncoder::new(session);
    upload_sample(session, &mut se.encoder, 0, bytemuck::bytes_of(&tok));
    encode_step_matvec(
        &mut se,
        &session.model.cfg,
        0,
        Some((0, 1)),
        pos,
        &mut NoMarker,
    );
    se.copy_sample_to_readback(8);
    se.schedule_topk_map(8);
    session.model.queue.submit(Some(se.finish()));
    wait_topk_readback(session, 1)?;
    Ok(())
}

/// Per-kernel breakdown of one matmul-prefill step at batch size `m`.
///
/// Mirrors [`microbench_tg`] but for the batched-prefill (matmul) path.
/// Reports per-call us, per-step ms, and %step for each labeled dispatch in
/// `prefill_matmul_topk`. The KV cache is pre-filled with `m` tokens before
/// measurement so the measured prefill runs at `pos_base = m` — the matmul
/// attention then scans `[0, m + m_tok]` per query, exercising the realistic
/// "prefill into an existing context" path rather than always starting at 0.
#[cfg(all(not(feature = "ci"), not(target_vendor = "apple")))]
pub fn microbench_pp(model: &Model, m: u32, repeats: u32, no_marker: bool) -> Result<()> {
    let m = m.min(model.m_max);
    if m == 0 {
        return Err(PotError::PrefillTooLarge {
            n: 0,
            max: model.m_max,
        });
    }
    if 2 * m > model.max_seq {
        return Err(PotError::ContextOverflow {
            pos: m,
            n: m,
            max: model.max_seq,
        });
    }
    let pos_base = m;
    eprintln!(
        "--- microbench pp (m={m}, pos_base={pos_base}, repeats={repeats}{}) ---",
        if no_marker { ", NO MARKER" } else { "" }
    );
    let cfg = &model.cfg;
    let prompt: Vec<u32> = (0..m).map(|i| (i % (cfg.n_vocab - 1)) + 1).collect();

    let session = model.new_session();

    // Pre-fill KV[0..m] so the measured prefills land at pos_base=m. One
    // un-instrumented matmul prefill (NoMarker) does this in a single dispatch.
    let _ = prefill_matmul_topk(&session, &prompt, 0, 1, &mut NoMarker)?;
    if let Err(e) = model.device.poll(PollType::wait_indefinitely()) {
        model.check_device()?;
        return Err(PotError::Poll(e));
    }

    // No-marker mode: switch to NoMarker so per-step timing isn't recorded.
    // Each measurement is a single submit (no extra resolve submit), giving
    // deterministic submit indices for external profilers like Nsight Graphics
    // GPU Trace, which select work via --start-after-submits / --limit-to-submits.
    if no_marker {
        // warmup
        let _ = prefill_matmul_topk(&session, &prompt, pos_base, 1, &mut NoMarker)?;
        if let Err(e) = model.device.poll(PollType::wait_indefinitely()) {
            model.check_device()?;
            return Err(PotError::Poll(e));
        }
        eprintln!(">>> PROFILE BEGIN: pp measurement ({repeats} submit(s)) <<<");
        for _ in 0..repeats {
            let _ = prefill_matmul_topk(&session, &prompt, pos_base, 1, &mut NoMarker)?;
            if let Err(e) = model.device.poll(PollType::wait_indefinitely()) {
                model.check_device()?;
                return Err(PotError::Poll(e));
            }
        }
        eprintln!(">>> PROFILE END: pp measurement <<<");
        return Ok(());
    }

    // warm up
    let _ = run_instrumented_prefill(&session, &prompt, pos_base)?;

    let mut per_step_label_ns: HashMap<&'static str, Vec<f32>> = HashMap::new();
    let mut calls_per_step: HashMap<&'static str, u32> = HashMap::new();
    let mut step_totals_ns: Vec<f32> = Vec::with_capacity(repeats as usize);

    for _ in 0..repeats {
        let spans = run_instrumented_prefill(&session, &prompt, pos_base)?;
        let mut step_label_sum: HashMap<&'static str, (u32, f32)> = HashMap::new();
        for (label, ns) in &spans {
            let e = step_label_sum.entry(*label).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += ns;
        }
        step_totals_ns.push(spans.iter().map(|(_, ns)| ns).sum());
        for (label, (calls, ns_sum)) in step_label_sum {
            per_step_label_ns.entry(label).or_default().push(ns_sum);
            calls_per_step.entry(label).or_insert(calls);
        }
    }

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
    let step_mean_ms = step_mean_ns / 1e6;
    let step_std_ms = step_std_ns / 1e6;
    println!();
    println!(
        "step time: {step_mean_ms:.3} ± {step_std_ms:.3} ms  →  {:.1} t/s",
        m as f32 * 1000.0 / step_mean_ms
    );
    Ok(())
}

#[cfg(all(not(feature = "ci"), not(target_vendor = "apple")))]
fn run_instrumented_prefill(
    session: &Session<'_>,
    prompt: &[u32],
    pos_base: u32,
) -> Result<Vec<(&'static str, f32)>> {
    let mut marker = MicroMarker::new(session);
    let _ = prefill_matmul_topk(session, prompt, pos_base, 1, &mut marker)?;
    if let Err(e) = session.model.device.poll(PollType::wait_indefinitely()) {
        session.model.check_device()?;
        return Err(PotError::Poll(e));
    }
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
