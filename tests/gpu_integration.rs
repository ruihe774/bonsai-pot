use std::env;
use std::path::{Path, PathBuf};

use bonsai_pot::{GenerateOptions, KvSnapshot, Model, PotError, Sampler, StopReason};
use wgpu::DeviceLostReason;

fn model_dir() -> PathBuf {
    env::var_os("BONSAI_POT_MODEL_DIR").map_or_else(|| PathBuf::from("./model"), PathBuf::from)
}

#[allow(clippy::panic, reason = "test helper")]
fn load_model() -> Model {
    let dir = model_dir();
    pollster::block_on(Model::load(&dir))
        .unwrap_or_else(|e| panic!("failed to load {}: {e}", dir.display()))
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

/// `GenerateOptions` for determinism tests: greedy, no stop token, bounded length.
fn greedy_opts(max_new_tokens: u32) -> GenerateOptions {
    GenerateOptions {
        max_new_tokens,
        stop_token: Some(u32::MAX), // won't match anything
        sampler: greedy_sampler(),
    }
}

// ---- model loading -----------------------------------------------------------

#[test]
fn model_load_succeeds_and_config_is_bonsai_family() {
    let model = load_model();
    let cfg = model.config();
    // Invariants that hold across the Bonsai/Qwen3 dense family.
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.n_kv_head, 8);
    assert_eq!(cfg.n_head, 32);
    assert!(matches!(cfg.n_layer, 28 | 36 | 40));
    assert_ne!(cfg.eos_token_id, 0);
    assert!(cfg.n_vocab > 100_000);
    // max_seq_len / max_prefill_tokens are ModelOptions-driven, not model-specific.
    assert_eq!(model.max_seq_len(), 1024);
    assert_eq!(model.max_prefill_tokens(), 512);
}

#[test]
fn model_load_bad_path_is_io_error() {
    let result = pollster::block_on(Model::load(Path::new("./does-not-exist")));
    assert!(matches!(result, Err(PotError::Io { .. })));
}

// ---- vocab -------------------------------------------------------------------

#[test]
fn vocab_round_trip_specials() {
    let model = load_model();
    for tok in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"] {
        let id = model
            .token_id(tok)
            .unwrap_or_else(|| panic!("token '{tok}' not in vocab"));
        assert_eq!(model.vocab_token(id), Some(tok));
    }
}

#[test]
fn decode_tokens_round_trip_specials() {
    let model = load_model();
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
fn prefill_pos_nonzero_rejected() {
    let model = load_model();
    let mut sess = model.new_session();
    let prompt = short_prompt();
    let _ = sess.prefill(&prompt, &greedy_sampler()).unwrap();
    // pos is now 8; calling prefill again must fail.
    let err = sess.prefill(&prompt, &greedy_sampler()).unwrap_err();
    assert!(matches!(err, PotError::Config(_)));
}

#[test]
fn prefill_too_large_rejected() {
    let model = load_model();
    let mut sess = model.new_session();
    let too_many: Vec<u32> = vec![1u32; 513]; // > M_MAX=512
    let err = sess.prefill(&too_many, &greedy_sampler()).unwrap_err();
    assert!(
        matches!(
            err,
            PotError::PrefillTooLarge { .. } | PotError::ContextOverflow { .. }
        ),
        "unexpected error: {err}"
    );
}

// ---- determinism & parity ----------------------------------------------------

#[test]
fn greedy_is_byte_deterministic() {
    let model = load_model();
    let prompt = short_prompt();
    let opts = greedy_opts(16);

    let mut sess1 = model.new_session();
    let first1 = sess1.prefill(&prompt, &greedy_sampler()).unwrap();
    let (toks1, _) = sess1.generate(first1, &opts).unwrap();

    let mut sess2 = model.new_session();
    let first2 = sess2.prefill(&prompt, &greedy_sampler()).unwrap();
    let (toks2, _) = sess2.generate(first2, &opts).unwrap();

    assert_eq!(first1, first2, "prefill returned different first tokens");
    assert_eq!(toks1, toks2, "greedy generation is not byte-deterministic");
}

#[test]
fn matvec_matmul_parity_first_token() {
    // Both prefill paths must sample the same first token under greedy sampling.
    let model = load_model();
    let prompt = short_prompt();
    let greedy = greedy_sampler();

    let mut sess_matmul = model.new_session();
    let first_matmul = sess_matmul.prefill(&prompt, &greedy).unwrap();

    let mut sess_matvec = model.new_session();
    let first_matvec = sess_matvec.prefill_one_at_a_time(&prompt, &greedy).unwrap();

    assert_eq!(
        first_matmul, first_matvec,
        "matmul prefill ({first_matmul}) != matvec prefill ({first_matvec})"
    );
}

#[test]
fn seeded_sampler_reproducibility() {
    let model = load_model();
    let prompt = short_prompt();
    let seeded = Sampler {
        temperature: 1.0,
        seed: 42,
        ..Sampler::default()
    };
    let opts = GenerateOptions {
        max_new_tokens: 8,
        stop_token: Some(u32::MAX),
        sampler: seeded.clone(),
    };

    let run = |s: &Sampler| -> (u32, Vec<u32>) {
        let opts_local = GenerateOptions {
            sampler: s.clone(),
            ..opts.clone()
        };
        let mut sess = model.new_session();
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
    let model = load_model();
    let mut sess = model.new_session();
    let opts = GenerateOptions {
        max_new_tokens: 0,
        stop_token: Some(u32::MAX),
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
    let model = load_model();
    let prompt = short_prompt();
    let greedy = greedy_sampler();

    // Original session: prefill, snapshot, continue.
    let mut sess = model.new_session();
    let first = sess.prefill(&prompt, &greedy).unwrap();
    let snap = sess.snapshot().unwrap();
    let (toks_orig, _) = sess.generate(first, &greedy_opts(8)).unwrap();

    // Restored session: restore snapshot, continue from same point.
    let mut sess2 = model.new_session();
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
    let model = load_model();
    let prompt = short_prompt();
    let greedy = greedy_sampler();

    let mut sess = model.new_session();
    let first = sess.prefill(&prompt, &greedy).unwrap();
    let snap = sess.snapshot().unwrap();
    let (toks_orig, _) = sess.generate(first, &greedy_opts(4)).unwrap();

    // Serialize → deserialize → restore.
    let bytes = snap.to_bytes();
    let snap2 = KvSnapshot::from_bytes(&bytes).unwrap();
    let mut sess2 = model.new_session();
    sess2.restore(&snap2).unwrap();
    let (toks_via_disk, _) = sess2.generate(first, &greedy_opts(4)).unwrap();

    assert_eq!(
        toks_orig, toks_via_disk,
        "continuation via to_bytes/from_bytes round-trip diverged"
    );
}

#[test]
fn restore_pos_zero_snapshot_leaves_session_ready_for_prefill() {
    let model = load_model();
    // Empty snapshot (pos=0) should restore to a clean state, allowing prefill.
    let mut sess = model.new_session();
    let snap = sess.snapshot().unwrap();
    assert_eq!(snap.pos(), 0);

    let mut sess2 = model.new_session();
    sess2.restore(&snap).unwrap();
    assert_eq!(sess2.pos(), 0);

    // Should be able to prefill as if fresh.
    let prompt = short_prompt();
    sess2.prefill(&prompt, &greedy_sampler()).unwrap();
    assert_eq!(sess2.pos(), prompt.len() as u32);
}

// ---- device-lost handling ---------------------------------------------------

#[test]
fn device_lost_is_surfaced_as_error_and_model_is_recoverable() {
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
