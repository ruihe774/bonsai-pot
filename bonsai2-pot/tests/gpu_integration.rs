#![cfg(feature = "std")]
#![allow(clippy::significant_drop_tightening, reason = "tests")]

use std::env;
use std::path::{Path, PathBuf};

use bonsai2_pot::{
    GenerateOptions, KvSnapshot, LoadOptions, Model, PotError, Sampler, Session, StopReason,
};
use parking_lot::{
    MappedMutexGuard, Mutex, MutexGuard, RwLock, RwLockReadGuard, const_mutex, const_rwlock,
};
use wgpu::DeviceLostReason;

fn model_dir() -> PathBuf {
    env::var_os("BONSAI_POT_MODEL_DIR").map_or_else(|| PathBuf::from("./model"), PathBuf::from)
}

/// KV capacity used by the tests: small, so several concurrent sessions fit
/// next to the 6 GB of weights.
const TEST_MAX_SEQ: u32 = 2048;

#[allow(clippy::panic, reason = "test helper")]
fn load_model() -> Model {
    let dir = model_dir();
    Model::load_from_dir(
        &dir,
        LoadOptions {
            max_seq: TEST_MAX_SEQ,
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| panic!("failed to load {}: {e}", dir.display()))
}

static SHARED_SESSION: Mutex<Option<Session<'static>>> = const_mutex(None);

/// GPU tests take this shared; the device-lost test takes it exclusively
/// because it loads two extra 6 GB models and would starve concurrent tests
/// of VRAM.
static GPU: RwLock<()> = const_rwlock(());

fn gpu() -> RwLockReadGuard<'static, ()> {
    GPU.read()
}

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
    let _gpu = gpu();
    let session = shared_session();
    let model = session.model();
    for tok in [
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<think>",
        "</think>",
    ] {
        let id = model
            .token_id(tok)
            .unwrap_or_else(|| panic!("token '{tok}' not in vocab"));
        assert_eq!(model.vocab_token(id), Some(tok));
        assert!(
            model.added_tokens().contains(&id),
            "'{tok}' must be an added token"
        );
    }
    let im_end = model.token_id("<|im_end|>").unwrap();
    assert_eq!(model.config().eos_token_id, im_end);
    assert!(model.stop_tokens().contains(&im_end));
    assert!(
        model
            .stop_tokens()
            .contains(&model.token_id("<|endoftext|>").unwrap())
    );
}

#[test]
fn decode_tokens_round_trip_specials() {
    let _gpu = gpu();
    let session = shared_session();
    let model = session.model();
    // Special tokens contain only printable ASCII chars, so decode_token_bytes
    // maps each char back to its own byte.
    for tok in ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>"] {
        let id = model.token_id(tok).unwrap();
        let decoded = model.decode_tokens(&[id]);
        assert_eq!(decoded, tok, "decode_tokens round-trip failed for '{tok}'");
    }
}

// ---- prefill error guards ----------------------------------------------------

#[test]
fn batched_prefill_pos_nonzero_matches_matvec_loop() {
    let _gpu = gpu();
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
fn out_of_vocab_token_rejected() {
    let _gpu = gpu();
    let mut sess = shared_session();
    sess.reset();
    let bad = sess.model().config().n_vocab;
    assert!(matches!(
        sess.prefill(&[1, bad], &greedy_sampler()),
        Err(PotError::Config(_))
    ));
    assert_eq!(sess.pos(), 0, "rejected prefill must not advance pos");
}

/// `reset()` is O(1): kernels must treat the recurrent state as zero at
/// position 0 even though the previous conversation left it dirty.
#[test]
fn reset_discards_recurrent_state() {
    let _gpu = gpu();
    let prompt = short_prompt();
    let other: Vec<u32> = vec![100u32, 200, 300, 400, 500];
    let opts = greedy_opts(8);
    let mut sess = shared_session();
    sess.reset();
    let _ = sess.prefill(&other, &greedy_sampler()).unwrap();
    sess.reset();
    let first = sess.prefill(&prompt, &greedy_sampler()).unwrap();
    let (toks, _) = sess.generate(first, &opts).unwrap();

    let mut fresh = sess.model().new_session();
    let first2 = fresh.prefill(&prompt, &greedy_sampler()).unwrap();
    let (toks2, _) = fresh.generate(first2, &opts).unwrap();
    assert_eq!((first, toks), (first2, toks2));
}

#[test]
fn prefill_context_overflow_rejected() {
    let _gpu = gpu();
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
    let _gpu = gpu();
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
/// must not collapse into a low-entropy attractor (e.g. "OTOTOTOT…" when the
/// layers read garbage). A soft entropy floor catches that shape without
/// pinning the exact baseline output.
#[test]
fn greedy_does_not_collapse_into_low_entropy_loop() {
    let _gpu = gpu();
    // BPE encoding of "Once upon a time" against `./model`'s tokenizer.
    // Hard-coded so the test doesn't depend on `scripts/bpe.py` at runtime.
    let prompt: Vec<u32> = vec![12162, 5028, 264, 854];
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
    let _gpu = gpu();
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

/// 600 tokens of natural English text (a history-of-computing paragraph,
/// repeated): long enough that batched prefill splits into two chunks.
#[allow(clippy::unreadable_literal, reason = "encoded tokens")]
const LONG_PROMPT: &[u32] = &[
    760, 3712, 314, 23470, 369, 264, 3255, 314, 40584, 59993, 12282, 321, 40584, 7578, 48767, 13,
    733, 279, 220, 16, 24, 19, 15, 82, 11, 13934, 17943, 9896, 4307, 11683, 11, 25152, 21701,
    14287, 314, 2279, 11, 321, 998, 53218, 539, 21219, 75931, 5998, 836, 43517, 13, 561, 26257,
    314, 279, 94562, 303, 220, 16, 24, 19, 22, 5324, 4156, 25, 424, 12215, 42780, 27286, 31890,
    440, 2526, 11, 14294, 30147, 1801, 314, 83742, 3558, 13, 357, 12828, 2843, 11, 23438, 9218,
    310, 1925, 1599, 1308, 375, 1052, 383, 264, 3074, 6365, 314, 48889, 11, 6611, 279, 17708,
    15746, 13, 25043, 19606, 12784, 303, 220, 16, 24, 21, 20, 421, 279, 1324, 314, 1308, 375, 1052,
    383, 264, 15911, 557, 57020, 16757, 1396, 1007, 11, 264, 8876, 2843, 30687, 310, 1396, 1330,
    1578, 321, 48501, 3750, 430, 19606, 579, 2253, 13, 271, 1919, 56768, 6310, 1801, 4204, 17943,
    3102, 303, 279, 3200, 220, 16, 24, 22, 15, 82, 321, 3988, 220, 16, 24, 23, 15, 82, 13, 42452,
    1680, 430, 279, 7905, 7696, 11, 279, 75502, 451, 220, 21, 19, 11, 321, 279, 27016, 6464, 6895,
    23470, 1083, 10105, 11, 8417, 11, 321, 2526, 9387, 13, 4236, 13538, 15784, 11436, 25, 41139,
    19689, 11, 3299, 35378, 11, 321, 3739, 6316, 279, 6165, 494, 264, 23011, 579, 13736, 1083, 449,
    17248, 5224, 13, 561, 18085, 314, 51357, 1156, 23344, 11, 92328, 506, 1543, 94774, 25247, 34,
    321, 5243, 1452, 539, 279, 7166, 59515, 321, 2843, 539, 4938, 5342, 11, 1801, 17943, 15156,
    310, 1208, 852, 995, 2496, 5158, 264, 1500, 314, 1970, 13, 271, 760, 1727, 17454, 3571, 440,
    27123, 13, 561, 7361, 11, 864, 5820, 430, 264, 3302, 2313, 23159, 539, 279, 3516, 4042, 5705,
    314, 15892, 11, 8660, 310, 8094, 9163, 303, 279, 3988, 220, 16, 24, 24, 15, 82, 13, 9076,
    13762, 382, 7869, 2059, 579, 4196, 31660, 4736, 6348, 424, 264, 11321, 3460, 11, 321, 2785,
    264, 2342, 1578, 11391, 314, 1208, 998, 11293, 2469, 11, 30290, 3370, 6602, 11, 321, 11580,
    2767, 13, 760, 3712, 314, 23470, 369, 264, 3255, 314, 40584, 59993, 12282, 321, 40584, 7578,
    48767, 13, 733, 279, 220, 16, 24, 19, 15, 82, 11, 13934, 17943, 9896, 4307, 11683, 11, 25152,
    21701, 14287, 314, 2279, 11, 321, 998, 53218, 539, 21219, 75931, 5998, 836, 43517, 13, 561,
    26257, 314, 279, 94562, 303, 220, 16, 24, 19, 22, 5324, 4156, 25, 424, 12215, 42780, 27286,
    31890, 440, 2526, 11, 14294, 30147, 1801, 314, 83742, 3558, 13, 357, 12828, 2843, 11, 23438,
    9218, 310, 1925, 1599, 1308, 375, 1052, 383, 264, 3074, 6365, 314, 48889, 11, 6611, 279, 17708,
    15746, 13, 25043, 19606, 12784, 303, 220, 16, 24, 21, 20, 421, 279, 1324, 314, 1308, 375, 1052,
    383, 264, 15911, 557, 57020, 16757, 1396, 1007, 11, 264, 8876, 2843, 30687, 310, 1396, 1330,
    1578, 321, 48501, 3750, 430, 19606, 579, 2253, 13, 271, 1919, 56768, 6310, 1801, 4204, 17943,
    3102, 303, 279, 3200, 220, 16, 24, 22, 15, 82, 321, 3988, 220, 16, 24, 23, 15, 82, 13, 42452,
    1680, 430, 279, 7905, 7696, 11, 279, 75502, 451, 220, 21, 19, 11, 321, 279, 27016, 6464, 6895,
    23470, 1083, 10105, 11, 8417, 11, 321, 2526, 9387, 13, 4236, 13538, 15784, 11436, 25, 41139,
    19689, 11, 3299, 35378, 11, 321, 3739, 6316, 279, 6165, 494, 264, 23011, 579, 13736, 1083, 449,
    17248, 5224, 13, 561, 18085, 314, 51357, 1156, 23344, 11, 92328, 506, 1543, 94774, 25247, 34,
    321, 5243, 1452, 539, 279, 7166, 59515, 321, 2843, 539, 4938, 5342, 11, 1801, 17943, 15156,
    310, 1208, 852, 995, 2496, 5158, 264, 1500, 314, 1970, 13, 271,
];

/// Batched prefill (two chunks: 512 + 88 tokens, the second at `pos_base` 512)
/// must continue exactly like the token-at-a-time path.
#[test]
fn chunked_batched_prefill_matches_token_loop() {
    let _gpu = gpu();
    let greedy = greedy_sampler();
    let opts = greedy_opts(12);
    let mut sess = shared_session();
    sess.reset();
    let first = sess.prefill(LONG_PROMPT, &greedy).unwrap();
    assert_eq!(sess.pos(), 600);
    let (toks, _) = sess.generate(first, &opts).unwrap();

    let mut sess2 = sess.model().new_session();
    let first2 = sess2.prefill_one_at_a_time(LONG_PROMPT, &greedy).unwrap();
    let (toks2, _) = sess2.generate(first2, &opts).unwrap();
    assert_eq!((first, toks), (first2, toks2));
}

#[test]
fn seeded_sampler_reproducibility() {
    let _gpu = gpu();
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
    let _gpu = gpu();
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
    let _gpu = gpu();
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
    let _gpu = gpu();
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
    let _gpu = gpu();
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
    let _gpu = gpu();
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
    let _gpu = gpu();
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
    let _gpu = GPU.write();
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
    drop(sess);
    drop(model);
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

// ---- reference match ---------------------------------------------------------

/// Compares per-layer residuals and logits against a llama.cpp reference dump.
/// Set `BONSAI_POT_REF_DIR` to a directory holding `<name>.u32` token files and
/// matching `<name>/index.jsonl` dumps from `scripts/llama_dump` (e.g. the CPU
/// backend). Skipped when unset. Requires `--features bench-internals`.
#[cfg(feature = "bench-internals")]
#[test]
#[allow(clippy::panic, clippy::unwrap_used, reason = "test")]
fn matches_llama_cpp_reference_dump() {
    use std::collections::HashMap;
    use std::fs;

    use bonsai2_pot::__bench::dump;

    let _gpu = gpu();

    let Some(ref_dir) = env::var_os("BONSAI_POT_REF_DIR").map(PathBuf::from) else {
        eprintln!("BONSAI_POT_REF_DIR unset; skipping");
        return;
    };
    let name = env::var("BONSAI_POT_REF_NAME").unwrap_or_else(|_| "p1_cpu".into());
    let tokens_name = name.trim_end_matches("_cpu");
    let raw = fs::read(ref_dir.join(format!("{tokens_name}.u32"))).unwrap();
    let tokens: Vec<u32> = raw
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let dir = ref_dir.join(&name);
    let mut refs: HashMap<String, Vec<f32>> = HashMap::new();
    for line in fs::read_to_string(dir.join("index.jsonl")).unwrap().lines() {
        let field = |k: &str| {
            let start = line.find(&format!("\"{k}\":\"")).unwrap() + k.len() + 4;
            line[start..start + line[start..].find('"').unwrap()].to_owned()
        };
        let (n, file) = (field("name"), field("file"));
        if n.starts_with("l_out-") || n == "logits" {
            let bytes = fs::read(dir.join(file)).unwrap();
            let v = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            refs.entry(n).or_insert(v);
        }
    }
    let model = shared_session().model();
    let ours = dump(model, &tokens).unwrap();
    let mut compared = 0;
    for t in &ours {
        let Some(r) = refs.get(&t.name) else { continue };
        assert_eq!(r.len(), t.data.len(), "{}: length mismatch", t.name);
        let n = t.ne0 as usize;
        let mut top1 = 0usize;
        for (a, b) in r.chunks_exact(n).zip(t.data.chunks_exact(n)) {
            let dot: f64 = a
                .iter()
                .zip(b)
                .map(|(x, y)| f64::from(*x) * f64::from(*y))
                .sum();
            let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
            let nb: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
            let cos = dot / (na * nb);
            assert!(cos >= 0.999, "{}: cosine {cos} < 0.999", t.name);
            let argmax = |v: &[f32]| {
                v.iter()
                    .enumerate()
                    .max_by(|x, y| x.1.total_cmp(y.1))
                    .unwrap()
                    .0
            };
            top1 += usize::from(argmax(a) == argmax(b));
        }
        if t.name == "logits" {
            let rows = r.len() / n;
            assert!(
                top1 * 100 >= rows * 95,
                "logits top-1 agreement {top1}/{rows} < 95%"
            );
        }
        compared += 1;
    }
    assert!(compared > 1, "no reference tensors matched");
}
