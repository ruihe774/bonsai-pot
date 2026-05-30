#![cfg(feature = "std")]
#![allow(clippy::significant_drop_tightening, reason = "tests")]

use std::env;
use std::path::{Path, PathBuf};

use bonsai_pot::{GenerateOptions, KvSnapshot, Model, PotError, Sampler, Session, StopReason};
use parking_lot::{MappedMutexGuard, Mutex, MutexGuard, const_mutex};
use wgpu::DeviceLostReason;

fn model_dir() -> PathBuf {
    env::var_os("BONSAI_POT_MODEL_DIR").map_or_else(|| PathBuf::from("./model"), PathBuf::from)
}

#[allow(clippy::panic, reason = "test helper")]
fn load_model() -> Model {
    let dir = model_dir();
    Model::load(&dir).unwrap_or_else(|e| panic!("failed to load {}: {e}", dir.display()))
}

static SHARED_SESSION: Mutex<Option<Session<'static>>> = const_mutex(None);

fn shared_session() -> MappedMutexGuard<'static, Session<'static>> {
    MutexGuard::map(SHARED_SESSION.lock(), |o| {
        o.get_or_insert_with(|| Box::leak(Box::new(load_model())).new_session())
    })
}

fn greedy_sampler() -> Sampler {
    Sampler {
        temperature: 0.0,
        ..Sampler::default()
    }
}

/// A short prompt using low-index token ids that are guaranteed to exist in
/// any model with `vocab_size` >= 10. The exact ids don't matter for correctness
/// tests; what matters is that all runs see the same input.
fn short_prompt() -> Vec<u32> {
    vec![1u32, 2, 3, 4, 5, 6, 7, 8]
}

/// `GenerateOptions` for determinism tests: greedy, never-stop, bounded length.
fn greedy_opts(max_new_tokens: u32) -> GenerateOptions {
    GenerateOptions {
        max_new_tokens,
        stop_pred: Some((|_| false) as fn(u32) -> bool),
        sampler: greedy_sampler(),
    }
}

// ---- model loading -----------------------------------------------------------

#[test]
fn model_load_bad_path_is_io_error() {
    let result = Model::load(Path::new("./does-not-exist"));
    assert!(matches!(result, Err(PotError::Io { .. })));
}

// ---- vocab -------------------------------------------------------------------

#[test]
fn vocab_round_trip_specials() {
    let session = shared_session();
    let model = session.model();
    for tok in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"] {
        let id = model
            .token_id(tok)
            .unwrap_or_else(|| panic!("token '{tok}' not in vocab"));
        assert_eq!(model.vocab_token(id), Some(tok));
    }
}

#[test]
fn decode_tokens_round_trip_specials() {
    let session = shared_session();
    let model = session.model();
    // Special tokens contain only printable ASCII chars, so decode_token_bytes
    // maps each char back to its own byte.
    for tok in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"] {
        let id = model.token_id(tok).unwrap();
        let decoded = model.decode_tokens(&[id]);
        assert_eq!(decoded, tok, "decode_tokens round-trip failed for '{tok}'");
    }
}

// ---- prefill error guards ----------------------------------------------------

#[test]
fn batched_prefill_pos_nonzero_matches_matvec_loop() {
    // Batched matmul prefill at pos > 0 must yield the same first-sampled token
    // as the matvec-loop variant under greedy sampling.
    let prompt = short_prompt();
    let greedy = greedy_sampler();

    let mut sess_matmul = shared_session();
    sess_matmul.reset();
    let _ = sess_matmul.prefill(&prompt, &greedy).unwrap();
    let first_matmul = sess_matmul.prefill(&prompt, &greedy).unwrap();

    let mut sess_matvec = sess_matmul.model().new_session();
    let _ = sess_matvec.prefill_one_at_a_time(&prompt, &greedy).unwrap();
    let first_matvec = sess_matvec.prefill_one_at_a_time(&prompt, &greedy).unwrap();

    assert_eq!(
        first_matmul, first_matvec,
        "batched prefill at pos>0 ({first_matmul}) != matvec-loop ({first_matvec})"
    );
    assert_eq!(sess_matmul.pos(), 2 * prompt.len() as u32);
}

#[test]
fn prefill_context_overflow_rejected() {
    let mut sess = shared_session();
    sess.reset();
    // Prompts longer than max_seq must be rejected up front (chunking can't
    // help since the KV cache itself isn't large enough).
    let too_many: Vec<u32> = vec![1u32; (sess.model().max_seq_len() + 1) as usize];
    let err = sess.prefill(&too_many, &greedy_sampler()).unwrap_err();
    assert!(
        matches!(err, PotError::ContextOverflow { .. }),
        "unexpected error: {err}"
    );
}

// ---- determinism & parity ----------------------------------------------------

#[test]
fn greedy_is_byte_deterministic() {
    let prompt = short_prompt();
    let opts = greedy_opts(16);

    let mut sess1 = shared_session();
    sess1.reset();
    let first1 = sess1.prefill(&prompt, &greedy_sampler()).unwrap();
    let (toks1, _) = sess1.generate(first1, &opts).unwrap();

    let mut sess2 = sess1.model().new_session();
    let first2 = sess2.prefill(&prompt, &greedy_sampler()).unwrap();
    let (toks2, _) = sess2.generate(first2, &opts).unwrap();

    assert_eq!(first1, first2, "prefill returned different first tokens");
    assert_eq!(toks1, toks2, "greedy generation is not byte-deterministic");
}

/// Coherence smoke test: greedy generation on a real natural-language prompt
/// must not collapse into a low-entropy attractor. The MSL buffer-binding
/// regression on Apple manifested as outputs like "OTOTOTOT…" — every layer
/// read garbage, the LM head landed on the same handful of tokens, and the
/// model oscillated between two ids forever. A soft entropy floor catches
/// any future regression with that shape (or worse: `!!!!!!!!`) without
/// pinning the exact baseline output, which is fragile across model and
/// kernel changes.
#[test]
fn greedy_does_not_collapse_into_low_entropy_loop() {
    // BPE encoding of "Once upon a time" against `./model`'s tokenizer.
    // Hard-coded so the test doesn't depend on `scripts/bpe.py` at runtime.
    let prompt: Vec<u32> = vec![12522, 5193, 264, 882];
    let opts = greedy_opts(24);

    let mut sess = shared_session();
    sess.reset();
    let first = sess.prefill(&prompt, &greedy_sampler()).unwrap();
    let (toks, _) = sess.generate(first, &opts).unwrap();

    let mut all = vec![first];
    all.extend_from_slice(&toks);
    let mut unique = all.clone();
    unique.sort_unstable();
    unique.dedup();
    assert!(
        unique.len() >= 8,
        "greedy generation collapsed: {} unique tokens in {} (sequence: {:?})",
        unique.len(),
        all.len(),
        all,
    );
}

#[test]
fn matvec_matmul_parity_first_token() {
    // Both prefill paths must sample the same first token under greedy sampling.
    let prompt = short_prompt();
    let greedy = greedy_sampler();

    let mut sess_matmul = shared_session();
    sess_matmul.reset();
    let first_matmul = sess_matmul.prefill(&prompt, &greedy).unwrap();

    let mut sess_matvec = sess_matmul.model().new_session();
    let first_matvec = sess_matvec.prefill_one_at_a_time(&prompt, &greedy).unwrap();

    assert_eq!(
        first_matmul, first_matvec,
        "matmul prefill ({first_matmul}) != matvec prefill ({first_matvec})"
    );
}

#[test]
fn seeded_sampler_reproducibility() {
    let prompt = short_prompt();
    let seeded = Sampler {
        temperature: 1.0,
        seed: 42,
        ..Sampler::default()
    };
    let run = |s: &Sampler| -> (u32, Vec<u32>) {
        let opts_local = GenerateOptions {
            max_new_tokens: 8,
            stop_pred: Some((|_| false) as fn(u32) -> bool),
            sampler: s.clone(),
        };
        let mut sess = shared_session().model().new_session();
        let first = sess.prefill(&prompt, s).unwrap();
        let (toks, _) = sess.generate(first, &opts_local).unwrap();
        (first, toks)
    };

    let (f1, t1) = run(&seeded);
    let (f2, t2) = run(&seeded);
    assert_eq!(f1, f2, "prefill sampled different tokens with same seed");
    assert_eq!(t1, t2, "generation not reproducible with same seed");

    let seeded2 = Sampler { seed: 43, ..seeded };
    let (f3, t3) = run(&seeded2);
    // Different seed should almost certainly produce a different sequence (may
    // collide on rare inputs, but won't for a real language model).
    assert!(
        f1 != f3 || t1 != t3,
        "different seeds produced identical output — likely a bug"
    );
}

// ---- generate options --------------------------------------------------------

#[test]
fn generate_max_tokens_zero_returns_immediately() {
    let mut sess = shared_session();
    sess.reset();
    let opts = GenerateOptions {
        max_new_tokens: 0,
        stop_pred: Some((|_| false) as fn(u32) -> bool),
        sampler: greedy_sampler(),
    };
    let mut fired = false;
    let stop = sess
        .generate_streaming(0, &opts, |_| {
            fired = true;
        })
        .unwrap();
    assert_eq!(stop, StopReason::MaxTokens);
    assert!(!fired, "on_token callback fired when max_new_tokens=0");
}

// ---- snapshot / restore ------------------------------------------------------

#[test]
fn snapshot_restore_round_trip_continues_identically() {
    let prompt = short_prompt();
    let greedy = greedy_sampler();

    // Original session: prefill, snapshot, continue.
    let mut sess = shared_session();
    sess.reset();
    let first = sess.prefill(&prompt, &greedy).unwrap();
    let snap = sess.snapshot().unwrap();
    let (toks_orig, _) = sess.generate(first, &greedy_opts(8)).unwrap();

    // Restored session: restore snapshot, continue from same point.
    let mut sess2 = sess.model().new_session();
    sess2.restore(&snap).unwrap();
    assert_eq!(sess2.pos(), snap.pos());
    let (toks_restored, _) = sess2.generate(first, &greedy_opts(8)).unwrap();

    assert_eq!(
        toks_orig, toks_restored,
        "continuation after snapshot/restore diverged from original"
    );
}

#[test]
fn snapshot_to_bytes_round_trip() {
    let prompt = short_prompt();
    let greedy = greedy_sampler();

    let mut sess = shared_session();
    sess.reset();
    let first = sess.prefill(&prompt, &greedy).unwrap();
    let snap = sess.snapshot().unwrap();
    let (toks_orig, _) = sess.generate(first, &greedy_opts(4)).unwrap();

    // Serialize → deserialize → restore.
    let bytes = snap.to_bytes();
    let snap2 = KvSnapshot::from_bytes(&bytes).unwrap();
    let mut sess2 = sess.model().new_session();
    sess2.restore(&snap2).unwrap();
    let (toks_via_disk, _) = sess2.generate(first, &greedy_opts(4)).unwrap();

    assert_eq!(
        toks_orig, toks_via_disk,
        "continuation via to_bytes/from_bytes round-trip diverged"
    );
}

#[test]
fn restore_pos_zero_snapshot_leaves_session_ready_for_prefill() {
    // Empty snapshot (pos=0) should restore to a clean state, allowing prefill.
    let mut sess = shared_session();
    sess.reset();
    let snap = sess.snapshot().unwrap();
    assert_eq!(snap.pos(), 0);

    let mut sess2 = sess.model().new_session();
    sess2.restore(&snap).unwrap();
    assert_eq!(sess2.pos(), 0);

    // Should be able to prefill as if fresh.
    let prompt = short_prompt();
    sess2.prefill(&prompt, &greedy_sampler()).unwrap();
    assert_eq!(sess2.pos(), prompt.len() as u32);
}

// ---- shared model / multiple sessions ---------------------------------------

/// Two sessions borrowing the same `Model` must be fully independent: interleaving
/// their prefill and generation calls must not corrupt either session's KV state
/// or activation buffers, and each must produce the same tokens it would produce
/// if run alone.
#[test]
fn two_sessions_interleaved_are_independent() {
    let model = shared_session().model();
    let prompt_a = short_prompt();
    // A distinct prompt for session B so the test exercises two different KV states.
    let prompt_b: Vec<u32> = vec![9u32, 8, 7, 6, 5, 4, 3, 2];
    let greedy = greedy_sampler();
    let opts = greedy_opts(8);

    // Establish baselines by running each prompt in isolation.
    let baseline_a = {
        let mut s = model.new_session();
        let first = s.prefill(&prompt_a, &greedy).unwrap();
        let (toks, _) = s.generate(first, &opts).unwrap();
        (first, toks)
    };
    let baseline_b = {
        let mut s = model.new_session();
        let first = s.prefill(&prompt_b, &greedy).unwrap();
        let (toks, _) = s.generate(first, &opts).unwrap();
        (first, toks)
    };

    // Now run the two sessions interleaved: prefill A, prefill B, generate A,
    // generate B.  If the shared activation buffers are not properly
    // re-initialised between sessions this will produce garbage for at least
    // one of them.
    let mut sess_a = shared_session();
    sess_a.reset();
    let mut sess_b = model.new_session();

    let first_a = sess_a.prefill(&prompt_a, &greedy).unwrap();
    let first_b = sess_b.prefill(&prompt_b, &greedy).unwrap();

    assert_eq!(
        first_a, baseline_a.0,
        "session A prefill diverged when interleaved with session B"
    );
    assert_eq!(
        first_b, baseline_b.0,
        "session B prefill diverged when interleaved with session A"
    );

    let (toks_a, _) = sess_a.generate(first_a, &opts).unwrap();
    let (toks_b, _) = sess_b.generate(first_b, &opts).unwrap();

    assert_eq!(
        toks_a, baseline_a.1,
        "session A generation diverged when interleaved with session B"
    );
    assert_eq!(
        toks_b, baseline_b.1,
        "session B generation diverged when interleaved with session A"
    );
}

/// After session A finishes, creating a third session from the same model and
/// running the same prompt must still yield the baseline output.
#[test]
fn session_reuse_after_prior_session_dropped() {
    let model = shared_session().model();
    let prompt = short_prompt();
    let greedy = greedy_sampler();
    let opts = greedy_opts(8);

    let baseline = {
        let mut s = model.new_session();
        let first = s.prefill(&prompt, &greedy).unwrap();
        let (toks, _) = s.generate(first, &opts).unwrap();
        (first, toks)
    };

    // Run a second session that leaves the GPU in an arbitrary state.
    {
        let mut s = model.new_session();
        let first = s.prefill(&prompt, &greedy).unwrap();
        let _ = s.generate(first, &opts).unwrap();
    }

    // Third session must still match baseline.
    let mut s3 = shared_session();
    s3.reset();
    let first3 = s3.prefill(&prompt, &greedy).unwrap();
    let (toks3, _) = s3.generate(first3, &opts).unwrap();

    assert_eq!(
        first3, baseline.0,
        "prefill diverged on third session after prior session dropped"
    );
    assert_eq!(
        toks3, baseline.1,
        "generation diverged on third session after prior session dropped"
    );
}

// ---- device-lost handling ---------------------------------------------------

#[test]
fn device_lost_is_surfaced_as_error_and_model_is_recoverable() {
    // Must use a private model — this test destroys the device, which would
    // corrupt the shared statics for all other tests.
    let model = load_model();

    // Confirm liveness before we destroy the device.
    let mut sess = model.new_session();
    let first = sess.prefill(&short_prompt(), &greedy_sampler()).unwrap();

    // Capture a snapshot while the device is still healthy, so we can verify
    // that restore() is also guarded after loss.
    let snap = sess.snapshot().unwrap();

    // Destroy the device; this triggers the device-lost callback synchronously.
    model.__destroy_device_for_test();

    // The flag must be latched immediately after destroy.
    assert!(
        model.is_device_lost(),
        "is_device_lost() must be true after destroy"
    );

    // step() must return DeviceLost, not panic.
    let err = sess
        .step(first, &greedy_sampler())
        .expect_err("step on a lost device must fail");
    assert!(
        matches!(
            err,
            PotError::DeviceLost {
                reason: DeviceLostReason::Destroyed,
                ..
            }
        ),
        "expected DeviceLost(Destroyed), got: {err}"
    );

    // snapshot() must return DeviceLost.
    let err = sess
        .snapshot()
        .expect_err("snapshot on a lost device must fail");
    assert!(
        matches!(err, PotError::DeviceLost { .. }),
        "expected DeviceLost, got: {err}"
    );

    // restore() (sync) must return DeviceLost.
    let err = sess
        .restore(&snap)
        .expect_err("restore on a lost device must fail");
    assert!(
        matches!(err, PotError::DeviceLost { .. }),
        "expected DeviceLost, got: {err}"
    );

    // A fresh load must succeed and the new model must be healthy.
    // (sess and model are dropped here by NLL when they fall out of use.)
    let model2 = load_model();
    assert!(
        !model2.is_device_lost(),
        "freshly loaded model should not be lost"
    );
    let mut sess2 = model2.new_session();
    let first2 = sess2
        .prefill(&short_prompt(), &greedy_sampler())
        .expect("prefill on recovered model must succeed");
    let _ = sess2
        .step(first2, &greedy_sampler())
        .expect("step on recovered model must succeed");
}
