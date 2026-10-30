//! Bench / microbench helpers, exposed only with the `bench-internals`
//! feature. Not part of the stable public API.
//!
//! Included as a child module of `forward` via `#[path]` so the helpers can
//! reach `forward`'s private items via `super::`.

#[cfg(not(feature = "ci"))]
use std::cmp::Ordering;
#[cfg(not(feature = "ci"))]
use std::collections::HashMap;
use std::time::Instant;

#[cfg(not(feature = "ci"))]
use wgpu::PollType;

#[cfg(feature = "ci")]
use super::NoMarker;
#[cfg(not(feature = "ci"))]
use super::{BenchMarker, MicroMarker, Session};
use super::{
    DumpTensor, Result, StepEncoder, encode_step, prefill_topk, step_no_sample, upload_sample,
    wait_topk_readback,
};
#[cfg(not(feature = "ci"))]
use crate::error::PotError;
use crate::model::Model;
use crate::session::{GenerateOptions, Sampler};

/// `ChatML` prompt asking for a very long essay (thinking disabled), so the
/// e2e tg run produces the requested number of tokens.
#[allow(clippy::unreadable_literal, reason = "encoded tokens")]
const E2E_PROMPT: &[u32] = &[
    248045, 8678, 198, 2523, 513, 264, 10631, 17313, 421, 13514, 1546, 1248, 11, 11346, 14090, 13,
    14271, 2842, 3988, 13, 248046, 198, 248045, 846, 198, 7734, 449, 8744, 11346, 8627, 314, 506,
    3140, 220, 16, 15, 15, 15, 15, 4105, 883, 279, 3712, 314, 279, 12386, 19825, 13, 17227, 1396,
    4031, 303, 7739, 13, 14569, 4237, 1973, 21835, 2980, 488, 5372, 279, 2100, 2984, 13, 248046,
    198, 248045, 74455, 198, 248068, 271, 248069, 198,
];

/// Long prefill prompt: [`E2E_PROMPT`] repeated past `M_MAX` so prefill chunks.
fn e2e_pp_prompt() -> Vec<u32> {
    E2E_PROMPT
        .iter()
        .copied()
        .cycle()
        .take(9 * E2E_PROMPT.len())
        .collect()
}

/// Run `tokens` through the single-token path from position 0 and return
/// llama.cpp-named activations (see [`super::dump_tokens`]).
pub fn dump(model: &Model, tokens: &[u32]) -> Result<Vec<DumpTensor>> {
    let mut sess = model.new_session();
    super::dump_tokens(&mut sess, tokens)
}

/// Debug hook: residual after the first `n_layers` batched layers.
pub fn debug_prefill_x(model: &Model, tokens: &[u32], n_layers: usize) -> Result<Vec<f32>> {
    let mut sess = model.new_session();
    super::debug_prefill_x(&mut sess, tokens, n_layers)
}

pub fn bench(model: &Model, pp_n: u32, tg_n: u32, repeats: u32) -> Result<()> {
    let cfg = &model.cfg;
    eprintln!("--- bench: pp={pp_n}, tg={tg_n}, repeats={repeats} (after 1 warmup) ---");
    let prompt: Vec<u32> = (0..pp_n).map(|i| (i % (cfg.n_vocab - 1)) + 1).collect();
    let m_max = model.max_prefill_tokens() as usize;
    let bench_sess = model.new_session();

    let run_full_prefill = |prompt: &[u32]| -> Result<f32> {
        #[allow(unused, reason = "conditional compilation")]
        let mut total_gpu_ns = 0.0f32;
        let mut pos_base = 0u32;
        for slice in prompt.chunks(m_max) {
            #[cfg(not(feature = "ci"))]
            let mut marker = BenchMarker::new(&bench_sess);
            #[cfg(feature = "ci")]
            let mut marker = NoMarker;
            let _ = prefill_topk(&bench_sess, slice, pos_base, 1, &mut marker)?;
            #[cfg(not(feature = "ci"))]
            {
                total_gpu_ns += marker.resolve()?;
            }
            pos_base += slice.len() as u32;
        }
        Ok(total_gpu_ns)
    };

    let _ = run_full_prefill(&prompt)?;
    step_no_sample(&bench_sess, 1u32, 0);

    let mut pp_wall_ts = Vec::with_capacity(repeats as usize);
    let mut pp_gpu_ts = Vec::with_capacity(repeats as usize);
    for _ in 0..repeats {
        let t = Instant::now();
        let gpu_ns = run_full_prefill(&prompt)?;
        pp_wall_ts.push(pp_n as f32 / t.elapsed().as_secs_f32());
        pp_gpu_ts.push(pp_n as f32 / (gpu_ns / 1e9));
    }
    let (pp_wall_mean, pp_wall_std) = mean_std(&pp_wall_ts);
    let (pp_gpu_mean, pp_gpu_std) = mean_std(&pp_gpu_ts);

    let mut tg_wall_ts = Vec::with_capacity(repeats as usize);
    let mut tg_gpu_ts = Vec::with_capacity(repeats as usize);
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
            encode_step(&mut se, 0, pos, Some(1), &mut marker);
            se.copy_topk_to_readback(8);
            se.schedule_topk_map(8);
            model.queue.submit(Some(se.finish()));
            wait_topk_readback(&bench_sess, 1)?;
            #[cfg(not(feature = "ci"))]
            {
                total_gpu_ns += marker.resolve()?;
            }
        }
        tg_wall_ts.push(tg_n as f32 / t.elapsed().as_secs_f32());
        tg_gpu_ts.push(tg_n as f32 / (total_gpu_ns / 1e9));
    }
    let (tg_wall_mean, tg_wall_std) = mean_std(&tg_wall_ts);
    let (tg_gpu_mean, tg_gpu_std) = mean_std(&tg_gpu_ts);

    let e2e_sampler = Sampler::default();
    let never_stop: fn(u32) -> bool = |_| false;
    let opts = GenerateOptions {
        max_new_tokens: tg_n.saturating_sub(1),
        sampler: e2e_sampler.clone(),
        stop_pred: Some(never_stop),
    };
    let e2e_tg_n = {
        let mut sess = model.new_session();
        let first = sess.prefill(E2E_PROMPT, &e2e_sampler)?;
        let (generated, _) = sess.generate(first, &opts)?;
        1 + generated.len() as u32
    };
    let mut e2e_tg_wall_ts = Vec::with_capacity(repeats as usize);
    for _ in 0..repeats {
        let mut sess = model.new_session();
        let first = sess.prefill(E2E_PROMPT, &e2e_sampler)?;
        let t = Instant::now();
        let (generated, _) = sess.generate(first, &opts)?;
        e2e_tg_wall_ts.push((1 + generated.len() as u32) as f32 / t.elapsed().as_secs_f32());
    }
    let (e2e_tg_wall_mean, e2e_tg_wall_std) = mean_std(&e2e_tg_wall_ts);

    let pp_prompt = e2e_pp_prompt();
    let pp_e2e_n = pp_prompt.len() as u32;
    {
        let mut sess = model.new_session();
        sess.prefill(&pp_prompt, &e2e_sampler)?;
    }
    let mut e2e_pp_wall_ts = Vec::with_capacity(repeats as usize);
    for _ in 0..repeats {
        let mut sess = model.new_session();
        let t = Instant::now();
        sess.prefill(&pp_prompt, &e2e_sampler)?;
        e2e_pp_wall_ts.push(pp_e2e_n as f32 / t.elapsed().as_secs_f32());
    }
    let (e2e_pp_wall_mean, e2e_pp_wall_std) = mean_std(&e2e_pp_wall_ts);

    println!();
    println!("| backend           |         test |          wall t/s |           gpu t/s |");
    println!("| ----------------- | ------------ | ----------------: | ----------------: |");
    println!(
        "| bonsai2-pot       | {:>12} | {pp_wall_mean:>9.2} ± {pp_wall_std:>5.2} | {pp_gpu_mean:>9.2} ± {pp_gpu_std:>5.2} |",
        format!("pp{pp_n}")
    );
    println!(
        "| bonsai2-pot       | {:>12} | {tg_wall_mean:>9.2} ± {tg_wall_std:>5.2} | {tg_gpu_mean:>9.2} ± {tg_gpu_std:>5.2} |",
        format!("tg{tg_n}")
    );
    println!(
        "| bonsai2-pot       | {:>12} | {e2e_pp_wall_mean:>9.2} ± {e2e_pp_wall_std:>5.2} |                 — |",
        format!("e2e_pp{pp_e2e_n}")
    );
    println!(
        "| bonsai2-pot       | {:>12} | {e2e_tg_wall_mean:>9.2} ± {e2e_tg_wall_std:>5.2} |                 — |",
        format!("e2e_tg{e2e_tg_n}")
    );
    Ok(())
}

/// Per-kernel breakdown of one decoding step at position `pos` (the state is
/// pre-filled with `pos` steps so attention scans `pos + 1` positions).
#[cfg(not(feature = "ci"))]
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
    for p in 0..pos {
        step_no_sample(&session, 1, p);
    }
    if no_marker {
        run_step(&session, pos, None)?;
        eprintln!(">>> PROFILE BEGIN: tg measurement ({repeats} submit(s)) <<<");
        for _ in 0..repeats {
            run_step(&session, pos, None)?;
        }
        eprintln!(">>> PROFILE END: tg measurement <<<");
        return Ok(());
    }
    let mut spans_all = Vec::new();
    let () = run_step(&session, pos, Some(&mut spans_all))?;
    let mut steps = Vec::with_capacity(repeats as usize);
    for _ in 0..repeats {
        let mut spans = Vec::new();
        run_step(&session, pos, Some(&mut spans))?;
        steps.push(spans);
    }
    print_breakdown(&steps, 1);
    Ok(())
}

#[cfg(not(feature = "ci"))]
fn run_step(
    session: &Session<'_>,
    pos: u32,
    spans: Option<&mut Vec<(&'static str, f32)>>,
) -> Result<()> {
    let tok: u32 = 1;
    let mut se = StepEncoder::new(session);
    upload_sample(session, &mut se.encoder, 0, bytemuck::bytes_of(&tok));
    if let Some(out) = spans {
        let mut marker = MicroMarker::new(session);
        encode_step(&mut se, 0, pos, Some(1), &mut marker);
        se.copy_topk_to_readback(8);
        se.schedule_topk_map(8);
        session.model.queue.submit(Some(se.finish()));
        wait_topk_readback(session, 1)?;
        *out = marker.resolve()?;
    } else {
        encode_step(&mut se, 0, pos, Some(1), &mut super::NoMarker);
        se.copy_topk_to_readback(8);
        se.schedule_topk_map(8);
        session.model.queue.submit(Some(se.finish()));
        wait_topk_readback(session, 1)?;
    }
    Ok(())
}

/// Per-kernel breakdown of one prefill of `m` tokens at `pos_base = m`.
#[cfg(not(feature = "ci"))]
pub fn microbench_pp(model: &Model, m: u32, repeats: u32, no_marker: bool) -> Result<()> {
    let m = m.min(model.max_prefill_tokens()).max(1);
    if 2 * m > model.max_seq {
        return Err(PotError::ContextOverflow {
            pos: m,
            n: m,
            max: model.max_seq,
        });
    }
    eprintln!(
        "--- microbench pp (m={m}, pos_base={m}, repeats={repeats}{}) ---",
        if no_marker { ", NO MARKER" } else { "" }
    );
    let prompt: Vec<u32> = (0..m).map(|i| (i % (model.cfg.n_vocab - 1)) + 1).collect();
    let session = model.new_session();
    let _ = prefill_topk(&session, &prompt, 0, 1, &mut super::NoMarker)?;
    let wait = || -> Result<()> {
        if let Err(e) = model.device.poll(PollType::wait_indefinitely()) {
            model.check_device()?;
            return Err(PotError::Poll(e));
        }
        Ok(())
    };
    wait()?;
    if no_marker {
        eprintln!(">>> PROFILE BEGIN: pp measurement ({repeats} submit(s)) <<<");
        for _ in 0..repeats {
            let _ = prefill_topk(&session, &prompt, m, 1, &mut super::NoMarker)?;
            wait()?;
        }
        eprintln!(">>> PROFILE END: pp measurement <<<");
        return Ok(());
    }
    let mut steps = Vec::with_capacity(repeats as usize);
    for i in 0..=repeats {
        let mut marker = MicroMarker::new(&session);
        let _ = prefill_topk(&session, &prompt, m, 1, &mut marker)?;
        wait()?;
        let spans = marker.resolve()?;
        if i > 0 {
            steps.push(spans);
        }
    }
    print_breakdown(&steps, m);
    Ok(())
}

#[cfg(not(feature = "ci"))]
fn print_breakdown(steps: &[Vec<(&'static str, f32)>], tokens_per_step: u32) {
    let mut per_step_label_ns: HashMap<&'static str, Vec<f32>> = HashMap::new();
    let mut calls_per_step: HashMap<&'static str, u32> = HashMap::new();
    let mut step_totals_ns = Vec::with_capacity(steps.len());
    for spans in steps {
        let mut sum: HashMap<&'static str, (u32, f32)> = HashMap::new();
        for (label, ns) in spans {
            let e = sum.entry(*label).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += ns;
        }
        step_totals_ns.push(spans.iter().map(|(_, ns)| ns).sum::<f32>());
        for (label, (calls, ns)) in sum {
            per_step_label_ns.entry(label).or_default().push(ns);
            calls_per_step.entry(label).or_insert(calls);
        }
    }
    let mut rows: Vec<(&'static str, u32, f32, f32, f32)> = per_step_label_ns
        .iter()
        .map(|(label, ns)| {
            let calls = calls_per_step[label];
            let (mean, std) = mean_std(ns);
            (
                *label,
                calls,
                mean / calls as f32 / 1000.0,
                mean / 1e6,
                std / 1e6,
            )
        })
        .collect();
    rows.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(Ordering::Equal));
    let total: f32 = rows.iter().map(|r| r.3).sum();
    println!();
    println!(
        "| kernel                         | calls/step | per-call us |   per-step ms ± std | %step |"
    );
    println!(
        "|--------------------------------|-----------:|------------:|--------------------:|------:|"
    );
    for (label, calls, us, ms, std) in &rows {
        println!(
            "| {label:<30} | {calls:>10} | {us:>11.2} | {ms:>11.3} ± {std:>5.3} | {:>5.1} |",
            100.0 * ms / total
        );
    }
    println!(
        "| TOTAL (sum of means)           |            |             | {total:>19.3} |       |"
    );
    let (mean_ns, std_ns) = mean_std(&step_totals_ns);
    println!();
    println!(
        "step time: {:.3} ± {:.3} ms  →  {:.1} t/s",
        mean_ns / 1e6,
        std_ns / 1e6,
        tokens_per_step as f32 * 1e9 / mean_ns
    );
}

fn mean_std(xs: &[f32]) -> (f32, f32) {
    if xs.is_empty() {
        return (0.0, 0.0);
    }
    let mean = xs.iter().sum::<f32>() / xs.len() as f32;
    let std = if xs.len() < 2 {
        0.0
    } else {
        (xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / (xs.len() - 1) as f32).sqrt()
    };
    (mean, std)
}
