//! Forward pass and per-step inference helpers.
//!
//! Two execution modes:
//!   - **matvec single-token** (`step_matvec_topk`, `prefill_matvec_loop_topk`):
//!     processes one token at a time via the multiply-free `Q1_0` matvec kernel.
//!     Used for token generation and for incremental prefill (when there's an
//!     existing KV cache prefix).
//!   - **matmul batched prefill** (`prefill_matmul_topk`): processes the prompt
//!     as one batch using `dot4I8Packed` matmul with a `Q8_0` quantize-activation
//!     pre-pass. Faster on long prompts; works at any `pos_base` (the matmul
//!     attention kernel scans `[0, pos_base + m_tok]` per query).
//!
//! Sampling lives outside this module: each entry point ends with a
//! topk multi-WG dispatch that writes the top-K logit values + indices to
//! the `sample` buffer. The caller reads them back and applies its own
//! temperature / top-p / multinomial logic on CPU.
//!
//! Per-token encoder hot path: per-layer tensor offsets are precomputed at
//! load time (`Model::layer_tensors`) so we don't re-do
//! `format!` + `HashMap` lookup + clone on every dispatch; bind groups are
//! built once per session (`SessionState::cached`) and shared across every
//! dispatch of a given (kind, weight buffer) pair, since the dynamic uniform
//! offset is the only per-dispatch variation.

use alloc::vec::Vec;

use wgpu::PollType;

use crate::error::{PotError, Result};
use crate::model::{
    ATTN_CHUNK_SIZE, AttnMergeParams, AttnPrefillTiledParams, AttnSplitParams, EmbedParams,
    KvWritebackFusedParams, MATVEC_FUSED_NORMED_ROWS_PER_WG, MATVEC_SILU_ROWS_PER_WG, MatmulParams,
    MatvecFusedNormedParams, MatvecParams, MatvecSiluParams, ModelConfig, QNormRopeFusedParams,
    RmsNormParams, RmsNormQ8Params, SiluMulQ8Params, TOPK_MAX, TOPK_NUM_PARTIAL_WG,
    TopKMergeParams, TopKPartialParams, WeightSet,
};
use crate::session::Session;

// ---------- Q8_0 KV cache layout helpers ------------------------------------
// Each kv_{k,v} buffer carries:
//   d-section  (FP32 scales): bytes [0, n_layer * max_seq * (kv_dim/32) * 4)
//   qs-section (i8 packed):   bytes [d_total, d_total + n_layer * max_seq * kv_dim)
// The block index for (layer il, position pos, block b in [0, kv_dim/32)) is
// `(il * max_seq + pos) * (kv_dim/32) + b` (single-row stride = kv_dim/32 in
// d-elements, kv_dim in qs-bytes).

const fn kv_qs_byte_base(cfg: &ModelConfig, max_seq: u32) -> u32 {
    cfg.n_layer * max_seq * (cfg.kv_dim / 32) * 4
}

/// `(d_word_offset, qs_byte_offset)` for the start of layer `il` inside each
/// kv buffer. Same layout for `kv_k` and `kv_v`, so callers reuse the pair.
const fn kv_layer_offsets(cfg: &ModelConfig, max_seq: u32, il: u32) -> (u32, u32) {
    let d_word = il * max_seq * (cfg.kv_dim / 32);
    let qs_byte = kv_qs_byte_base(cfg, max_seq) + il * max_seq * cfg.kv_dim;
    (d_word, qs_byte)
}

// ---------- step-encoder marker --------------------------------------------
// Generic hook for instrumenting `encode_step_matvec` / `prefill_matmul_topk`
// without forking them. Production passes `&mut NoMarker` (zero-cost: the
// trait methods are `#[inline(always)]` with empty bodies, so the compiler
// drops the calls entirely). Bench builds (`bench-internals`) provide two
// alternative impls — `BenchMarker` (whole-pass begin/end timestamps via
// `ComputePassDescriptor::timestamp_writes`, no per-dispatch overhead) and
// `MicroMarker` (per-dispatch `pass.write_timestamp` for kernel breakdowns);
// see further down.

pub trait StepMarker {
    /// Populate the pass descriptor before `begin_compute_pass`. `BenchMarker`
    /// uses this hook to install whole-pass `timestamp_writes` (cheap, no
    /// per-dispatch flushes); `NoMarker` and `MicroMarker` leave it untouched.
    fn setup_desc<'a>(&'a self, desc: &mut wgpu::ComputePassDescriptor<'a>);

    /// Write a per-dispatch timestamp inside an open pass. `MicroMarker` uses
    /// this; `NoMarker` and `BenchMarker` are no-ops.
    fn mark(&mut self, pass: &mut wgpu::ComputePass<'_>, label: &'static str);
}

pub struct NoMarker;
impl StepMarker for NoMarker {
    #[inline(always)]
    fn setup_desc<'a>(&'a self, _desc: &mut wgpu::ComputePassDescriptor<'a>) {}
    #[inline(always)]
    fn mark(&mut self, _pass: &mut wgpu::ComputePass<'_>, _label: &'static str) {}
}

// ---------- per-step encoder ------------------------------------------------

pub struct StepEncoder<'a> {
    session: &'a Session<'a>,
    pub(crate) encoder: wgpu::CommandEncoder,
}

impl<'a> StepEncoder<'a> {
    pub fn new(session: &'a Session<'a>) -> Self {
        let encoder =
            session
                .model
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("step"),
                });
        Self { session, encoder }
    }

    /// Append a `sample → readback` copy to this encoder so the readback
    /// transfer rides in the same command buffer as the step it follows.
    /// Avoids a separate submit purely for the copy.
    pub fn copy_sample_to_readback(&mut self, bytes: u64) {
        self.encoder.copy_buffer_to_buffer(
            &self.session.state.sample,
            0,
            &self.session.state.readback,
            0,
            bytes,
        );
    }

    /// Schedule a `MAP_READ` mapping of the readback buffer on submission.
    /// Must be called after [`copy_sample_to_readback`] and before [`finish`].
    pub fn schedule_topk_map(&self, bytes: u64) {
        self.encoder.map_buffer_on_submit(
            &self.session.state.readback,
            wgpu::MapMode::Read,
            0..bytes,
            |_| {},
        );
    }

    pub fn finish(self) -> wgpu::CommandBuffer {
        self.encoder.finish()
    }
}

/// Encode-side: record `copy_buffer_to_buffer(staging[0..n_bytes] → sample[dst_off..])`
/// into `encoder` and schedule a full-buffer remap of `staging` on submit.
///
/// This does not touch staging contents. Pair with [`commit_sample_upload`] before
/// `queue.submit`. Used in pipelined paths where the data isn't known at encode time.
pub fn encode_sample_upload(
    session: &Session<'_>,
    encoder: &mut wgpu::CommandEncoder,
    dst_off: u64,
    n_bytes: u64,
) {
    encoder.copy_buffer_to_buffer(
        &session.state.staging,
        0,
        &session.state.sample,
        dst_off,
        n_bytes,
    );
    encoder.map_buffer_on_submit(&session.state.staging, wgpu::MapMode::Write, 0.., |_| ());
}

/// Write-side: copy `data` into the currently-mapped `staging` buffer and unmap it.
///
/// Precondition: `buffers.staging` is currently mapped (guaranteed by `mapped_at_creation`
/// initially, and by [`encode_sample_upload`]'s `map_buffer_on_submit` after every preceding
/// submit + poll). Must be called before the corresponding `queue.submit` so the GPU sees
/// `staging` unmapped with the right data.
pub fn commit_sample_upload(session: &Session<'_>, data: &[u8]) {
    {
        let mut view = session
            .state
            .staging
            .slice(..data.len() as u64)
            .get_mapped_range_mut();
        view.copy_from_slice(data);
    }
    session.state.staging.unmap();
}

/// Convenience wrapper for non-pipelined paths: encode the copy + schedule the remap, then
/// commit `data` immediately. Equivalent to `encode_sample_upload` + `commit_sample_upload`.
pub fn upload_sample(
    session: &Session<'_>,
    encoder: &mut wgpu::CommandEncoder,
    dst_off: u64,
    data: &[u8],
) {
    encode_sample_upload(session, encoder, dst_off, data.len() as u64);
    commit_sample_upload(session, data);
}

// ---------- weight-set selection -------------------------------------------

const fn matvec_bg<'a>(session: &'a Session<'_>, ws: WeightSet) -> &'a wgpu::BindGroup {
    match ws {
        WeightSet::Attn => &session.state.cached.matvec_w_attn,
        WeightSet::FfnGU => &session.state.cached.matvec_w_ffn_gu,
        WeightSet::FfnD => &session.state.cached.matvec_w_ffn_d,
        WeightSet::Embed => &session.state.cached.matvec_w_embed,
    }
}

/// Bind group for `matvec_q1_0_fused_normed`. Only `Attn` (QKV) and `FfnGU`
/// (gate+up) are valid: those are the two sites in the matvec single-token
/// path that are preceded by an `rms_norm` and use the fused matvec.
#[allow(
    clippy::panic,
    reason = "internal invariant: only Attn / FfnGU are wired up"
)]
fn matvec_fused_normed_bg<'a>(session: &'a Session<'_>, ws: WeightSet) -> &'a wgpu::BindGroup {
    match ws {
        WeightSet::Attn => &session.state.cached.matvec_fused_normed_w_attn,
        WeightSet::FfnGU => &session.state.cached.matvec_fused_normed_w_ffn_gu,
        _ => panic!("matvec_fused_normed only supports WeightSet::Attn / FfnGU"),
    }
}

const fn matmul_bg<'a>(session: &'a Session<'_>, ws: WeightSet) -> &'a wgpu::BindGroup {
    match ws {
        WeightSet::Attn => &session.state.cached.matmul_w_attn,
        WeightSet::FfnGU => &session.state.cached.matmul_w_ffn_gu,
        WeightSet::FfnD => &session.state.cached.matmul_w_ffn_d,
        WeightSet::Embed => &session.state.cached.matmul_w_embed,
    }
}

// ---------- in-pass kernel dispatch helpers ---------------------------------
// These variants take a `&mut wgpu::ComputePass` already opened by the caller,
// allowing many dispatches to share one pass and amortize the
// begin_compute_pass cost (~25us each on RADV).

fn dispatch_rms_norm(
    session: &Session<'_>,
    cfg: &ModelConfig,
    pass: &mut wgpu::ComputePass<'_>,
    n_groups: u32,
    group_size: u32,
    in_off: u32,
    out_off: u32,
    w_off: u32,
) {
    let p = RmsNormParams {
        group_size,
        n_groups,
        input_offset: in_off,
        output_offset: out_off,
        weight_offset: w_off,
        eps: cfg.rms_eps,
    };
    pass.set_pipeline(&session.model.pipes.rms_norm);
    pass.set_bind_group(0, &session.state.cached.rms_norm, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&p));
    pass.dispatch_workgroups(n_groups, 1, 1);
}

fn dispatch_matvec_q1_0(
    session: &Session<'_>,
    pass: &mut wgpu::ComputePass<'_>,
    k: u32,
    n: u32,
    weights: WeightSet,
    w_d: u32,
    w_qs: u32,
    in_off: u32,
    out_off: u32,
    accumulate: bool,
) {
    const ROWS_PER_WG: u32 = 16;
    let n_wg = n.div_ceil(ROWS_PER_WG);
    let dispatch_x = n_wg.min(65535);
    let dispatch_y = n_wg.div_ceil(dispatch_x);
    let p = MatvecParams {
        k,
        n,
        d_offset: w_d,
        qs_offset: w_qs,
        input_offset: in_off,
        output_offset: out_off,
        accumulate: u32::from(accumulate),
        dispatch_x_dim: dispatch_x,
    };
    pass.set_pipeline(&session.model.pipes.matvec);
    pass.set_bind_group(0, matvec_bg(session, weights), &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&p));
    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
}

/// `matvec_q1_0` with the `silu(gate) * up` activation fold on the input side
/// (no separate `silu_mul` dispatch + `ffn_in` round-trip). Used for the
/// `Wd/ffn_down` dispatch on the matvec single-token path. Same multi-row WG
/// shape and bind-group layout as `dispatch_matvec_q1_0` — the only
/// difference is the kernel reads two activation regions (`gate`, `up`) and
/// fuses `silu(g)*u` per element.
fn dispatch_matvec_q1_0_silu(
    session: &Session<'_>,
    pass: &mut wgpu::ComputePass<'_>,
    k: u32,
    n: u32,
    weights: WeightSet,
    w_d: u32,
    w_qs: u32,
    gate_off: u32,
    up_off: u32,
    out_off: u32,
    accumulate: bool,
) {
    const ROWS_PER_WG: u32 = MATVEC_SILU_ROWS_PER_WG;
    let n_wg = n.div_ceil(ROWS_PER_WG);
    let dispatch_x = n_wg.min(65535);
    let dispatch_y = n_wg.div_ceil(dispatch_x);
    let p = MatvecSiluParams {
        k,
        n,
        d_offset: w_d,
        qs_offset: w_qs,
        gate_offset: gate_off,
        up_offset: up_off,
        output_offset: out_off,
        accumulate: u32::from(accumulate),
        dispatch_x_dim: dispatch_x,
    };
    pass.set_pipeline(&session.model.pipes.matvec_silu);
    pass.set_bind_group(0, matvec_bg(session, weights), &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&p));
    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
}

/// Fused: `rms_norm(x) * w_norm` → multi-range `Q1_0` matvec, in one dispatch.
/// Replaces `dispatch_rms_norm + dispatch_matvec_q1_0_fused` for the matvec
/// single-token path. See `shaders/matvec_q1_0_fused_normed.comp`.
fn dispatch_matvec_q1_0_fused_normed(
    session: &Session<'_>,
    cfg: &ModelConfig,
    pass: &mut wgpu::ComputePass<'_>,
    k: u32,
    input_offset: u32,
    w_norm_off: u32,
    weights: WeightSet,
    ranges: &[(u32, u32, u32, u32)],
) {
    const ROWS_PER_WG: u32 = MATVEC_FUSED_NORMED_ROWS_PER_WG;
    debug_assert!(ranges.len() == 2 || ranges.len() == 3);
    for (_, _, n, _) in ranges {
        debug_assert!(n % ROWS_PER_WG == 0);
    }
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
    pass.set_pipeline(&session.model.pipes.matvec_fused_normed);
    pass.set_bind_group(0, matvec_fused_normed_bg(session, weights), &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&p));
    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
}

/// Two-pass top-K over `n` f16 logits at `in_off` (f16 elements). Pass-1
/// (`topk_partial`) launches `TOPK_NUM_PARTIAL_WG` WGs, each producing the
/// top-K_MAX of its `n / TOPK_NUM_PARTIAL_WG` slice into a scratch region of
/// the `sample` buffer at u32 offset `2*TOPK_MAX`. Pass-2 (`topk_merge`)
/// runs a single WG that merges the partial slots and writes the final top-K
/// to `sample[out_off_u32 .. out_off_u32 + 2*k]`. The two dispatches share
/// the same compute pass — wgpu inserts the implicit storage barrier
/// between them.
fn dispatch_topk_reduce(
    session: &Session<'_>,
    pass: &mut wgpu::ComputePass<'_>,
    n: u32,
    k: u32,
    in_off: u32,
    out_off_u32: u32,
) {
    // n_per_wg: ceil(n / NUM_PARTIAL_WG), rounded up to a multiple of 2 so each
    // WG starts at an even f16 boundary (= u32 word boundary in `logits`).
    let n_per_wg = (n.div_ceil(TOPK_NUM_PARTIAL_WG) + 1) & !1;
    let partials_off = 2 * TOPK_MAX;

    let p1 = TopKPartialParams {
        n,
        in_offset: in_off,
        partials_off,
        n_per_wg,
    };
    pass.set_pipeline(&session.model.pipes.topk_partial);
    pass.set_bind_group(0, &session.state.cached.topk_partial, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&p1));
    pass.dispatch_workgroups(TOPK_NUM_PARTIAL_WG, 1, 1);

    let p2 = TopKMergeParams {
        partials_off,
        num_partials: TOPK_NUM_PARTIAL_WG,
        out_offset: out_off_u32,
        k,
    };
    pass.set_pipeline(&session.model.pipes.topk_merge);
    pass.set_bind_group(0, &session.state.cached.topk_merge, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&p2));
    pass.dispatch_workgroups(1, 1, 1);
}

/// Fused: `rms_norm(K` head) → \*`w_k_norm` → NEOX-RoPE → `Q8_0` quantize → write
/// `kv_k`. V runs in the same workgroup (just quantize + write `kv_v`). Replaces
/// `rms_norm(K) + rope(K) + kv_writeback` with one dispatch.
fn dispatch_kv_writeback_fused(
    session: &Session<'_>,
    cfg: &ModelConfig,
    pass: &mut wgpu::ComputePass<'_>,
    k_cur_off: u32,
    v_cur_off: u32,
    w_k_norm_off: u32,
    layer_il: u32,
    pos_base: u32,
    m_tokens: u32,
) {
    let nb_per_row = cfg.kv_dim / 32;
    let (dst_d_word_offset, dst_qs_byte_offset) =
        kv_layer_offsets(cfg, session.model.max_seq, layer_il);
    let p = KvWritebackFusedParams {
        k_cur_off,
        v_cur_off,
        w_k_norm_off,
        rope_offset: 0,
        dst_d_word_offset,
        dst_qs_byte_offset,
        pos_base,
        kv_dim: cfg.kv_dim,
        nb_per_row,
        eps: cfg.rms_eps,
    };
    pass.set_pipeline(&session.model.pipes.kv_writeback_fused);
    pass.set_bind_group(0, &session.state.cached.kv_writeback_fused, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&p));
    pass.dispatch_workgroups(cfg.n_kv_head, m_tokens, 1);
}

/// Fused: `rms_norm(Q` head) → \*`w_q_norm` → NEOX-RoPE, written back into
/// `act.q` in place. Replaces `rms_norm(Q) + rope(Q)` with one dispatch.
fn dispatch_q_norm_rope_fused(
    session: &Session<'_>,
    cfg: &ModelConfig,
    pass: &mut wgpu::ComputePass<'_>,
    q_off: u32,
    w_q_norm_off: u32,
    pos_base: u32,
    m_tokens: u32,
) {
    let p = QNormRopeFusedParams {
        q_off,
        w_q_norm_off,
        rope_offset: 0,
        pos_base,
        q_dim: cfg.q_dim,
        eps: cfg.rms_eps,
    };
    pass.set_pipeline(&session.model.pipes.q_norm_rope_fused);
    pass.set_bind_group(0, &session.state.cached.q_norm_rope_fused, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&p));
    pass.dispatch_workgroups(cfg.n_head, m_tokens, 1);
}

/// Fused: `rms_norm(x) * w` → `Q8_0` quantize → write to `act_q8`. One WG per
/// token. Replaces the prefill-path `dispatch_rms_norm + dispatch_quantize_act`
/// pair (eliminates the `act.x_norm` round-trip and one dispatch). Returns
/// `(d_offset, qs_offset)` of the freshly-written `Q8_0` region in `act_q8`.
fn dispatch_rms_norm_q8_0(
    session: &Session<'_>,
    cfg: &ModelConfig,
    pass: &mut wgpu::ComputePass<'_>,
    k: u32,
    m: u32,
    in_off_f16: u32,
    w_off_f16: u32,
) -> (u32, u32) {
    let nb_q8 = k / 32;
    let d_off = 0u32;
    let qs_off = m * nb_q8 * 4;
    let p = RmsNormQ8Params {
        k,
        input_offset: in_off_f16,
        weight_offset: w_off_f16,
        d_offset: d_off,
        qs_offset: qs_off,
        eps: cfg.rms_eps,
    };
    pass.set_pipeline(&session.model.pipes.rms_norm_q8_0);
    pass.set_bind_group(0, &session.state.cached.rms_norm_q8_0, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&p));
    pass.dispatch_workgroups(m, 1, 1);
    (d_off, qs_off)
}

/// Fused: `silu(gate) * up` → `Q8_0` quantize → write to `act_q8`. One WG per
/// token. Replaces `dispatch_silu_mul + dispatch_quantize_act` (eliminates
/// the `act.ffn_in` round-trip and one dispatch).
fn dispatch_silu_mul_q8_0(
    session: &Session<'_>,
    pass: &mut wgpu::ComputePass<'_>,
    k: u32,
    m: u32,
    gate_off: u32,
    up_off: u32,
) -> (u32, u32) {
    let nb_q8 = k / 32;
    let d_off = 0u32;
    let qs_off = m * nb_q8 * 4;
    let p = SiluMulQ8Params {
        k,
        gate_offset: gate_off,
        up_offset: up_off,
        d_offset: d_off,
        qs_offset: qs_off,
    };
    pass.set_pipeline(&session.model.pipes.silu_mul_q8_0);
    pass.set_bind_group(0, &session.state.cached.silu_mul_q8_0, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&p));
    pass.dispatch_workgroups(m, 1, 1);
    (d_off, qs_off)
}

fn dispatch_matmul_q1_0(
    session: &Session<'_>,
    pass: &mut wgpu::ComputePass<'_>,
    k: u32,
    n: u32,
    m: u32,
    weights: WeightSet,
    w_d: u32,
    w_qs: u32,
    a_d: u32,
    a_qs: u32,
    out_off: u32,
    accumulate: bool,
) {
    let p = MatmulParams {
        k,
        n,
        m,
        w_d_offset: w_d,
        w_qs_offset: w_qs,
        a_d_offset: a_d,
        a_qs_offset: a_qs,
        out_offset: out_off,
        accumulate: u32::from(accumulate),
    };
    pass.set_pipeline(&session.model.pipes.matmul);
    pass.set_bind_group(0, matmul_bg(session, weights), &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&p));
    pass.dispatch_workgroups(n.div_ceil(64), m.div_ceil(64), 1);
}

/// Q-tiled + GQA-batched FlashAttention-2 prefill kernel with fused `Q8_0`
/// output. One workgroup handles `Q_TILE=2` consecutive query tokens × 4 GQA
/// Q-heads sharing the same KV head. K/V are loaded once per cache position
/// and reused across all 2 * 4 queries — the bandwidth win that keeps
/// prefill from degrading quadratically. The attention output is quantized
/// to `Q8_0` in-place and written to `act_q8` (no f16 staging in
/// `act.attn_out`), so the Wo matmul reads it as-is. Returns the
/// `(d_offset, qs_offset)` byte offsets of the freshly written `Q8_0` region.
/// See `shaders/attention_prefill_tiled.wgsl`.
fn dispatch_attention_prefill_tiled(
    session: &Session<'_>,
    cfg: &ModelConfig,
    pass: &mut wgpu::ComputePass<'_>,
    layer_il: u32,
    max_seq: u32,
    m_tokens: u32,
    pos_base: u32,
) -> (u32, u32) {
    const Q_TILE: u32 = 2;
    let (d_word, qs_byte) = kv_layer_offsets(cfg, max_seq, layer_il);
    let nb_q8 = cfg.q_dim / 32;
    let out_d_offset = 0u32;
    let out_qs_offset = m_tokens * nb_q8 * 4;
    let p = AttnPrefillTiledParams {
        head_dim: cfg.head_dim,
        n_head: cfg.n_head,
        n_kv_head: cfg.n_kv_head,
        m_tokens,
        pos_base,
        kv_stride: cfg.kv_dim,
        q_offset: session.model.act_layout.q,
        k_d_word_offset: d_word,
        k_qs_byte_offset: qs_byte,
        v_d_word_offset: d_word,
        v_qs_byte_offset: qs_byte,
        out_d_offset,
        out_qs_offset,
        scale: 1.0 / (cfg.head_dim as f32).sqrt(),
    };
    pass.set_pipeline(&session.model.pipes.attention_prefill_tiled);
    pass.set_bind_group(0, &session.state.cached.attn_prefill_tiled, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&p));
    pass.dispatch_workgroups(cfg.n_kv_head, m_tokens.div_ceil(Q_TILE), 1);
    (out_d_offset, out_qs_offset)
}

// ---------- async readback helper -------------------------------------------

/// Wait for the readback mapping scheduled via [`StepEncoder::schedule_topk_map`]
/// to complete and return the K f32 logits + K u32 indices.
pub fn wait_topk_readback(session: &Session<'_>, k: u32) -> Result<(Vec<f32>, Vec<u32>)> {
    let bytes = u64::from(k) * 8; // K f32 + K u32
    let slice = session.state.readback.slice(0..bytes);
    if let Err(e) = session.model.device.poll(PollType::wait_indefinitely()) {
        session.model.check_device()?;
        return Err(PotError::Poll(e));
    }
    let data = slice.get_mapped_range();
    let words: &[u32] = bytemuck::cast_slice(&data[..bytes as usize]);
    let logits: Vec<f32> = words[..k as usize]
        .iter()
        .map(|w| f32::from_bits(*w))
        .collect();
    let indices: Vec<u32> = words[k as usize..2 * k as usize].to_vec();
    drop(data);
    session.state.readback.unmap();
    Ok((logits, indices))
}

// ---------- single-token forward (matvec) ----------------------------------

/// Encode one tg step into the given encoder. The input token is read from
/// `sample[sample_in]`. If `topk_out = Some((base, k))`, the suffix
/// (`output_norm` + LM head + `topk_reduce`) is appended and the top-K logits +
/// indices land at `sample[base..base + 2*k]`. If `topk_out = None`, the
/// suffix is skipped — useful for KV-fill-only steps (e.g. mid-prefill) where
/// the sampled token isn't read.
pub fn encode_step_matvec<M: StepMarker>(
    se: &mut StepEncoder,
    cfg: &ModelConfig,
    sample_in: u32,
    topk_out: Option<(u32, u32)>,
    pos: u32,
    marker: &mut M,
) {
    let session: &Session<'_> = se.session;
    let encoder = &mut se.encoder;
    let m = session.model;
    let ot = &m.output_tensors;
    let mut desc = wgpu::ComputePassDescriptor {
        label: Some("step_matvec"),
        timestamp_writes: None,
    };
    marker.setup_desc(&mut desc);
    let mut pass = encoder.begin_compute_pass(&desc);
    marker.mark(&mut pass, "start");

    // embed
    {
        let p = EmbedParams {
            k: cfg.n_embd,
            d_offset: ot.token_embd_d,
            qs_offset: ot.token_embd_qs,
            output_offset: m.act_layout.x,
            sample_offset: sample_in,
        };
        pass.set_pipeline(&m.pipes.embed);
        pass.set_bind_group(0, &session.state.cached.embed, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(1, 1, 1);
    }
    marker.mark(&mut pass, "embed");

    for il in 0..cfg.n_layer {
        layer_pre_kv_in_pass(session, cfg, &mut pass, il, pos, marker);
        // Fused: K (rms_norm + *w_k_norm + RoPE + Q8_0 quantize) and V (Q8_0
        // quantize) → write both into kv_{k,v}. Replaces the previous
        // rms_norm(K) + rope(K) + kv_writeback trio (3 dispatches → 1).
        let lt = &m.layer_tensors[il as usize];
        dispatch_kv_writeback_fused(
            session,
            cfg,
            &mut pass,
            m.act_layout.k_cur,
            m.act_layout.v_cur,
            lt.attn_k_norm_off,
            il,
            pos,
            1,
        );
        marker.mark(&mut pass, "kv_writeback");
        layer_post_kv_in_pass(session, cfg, &mut pass, il, pos, marker);
    }
    if let Some((topk_out_u32_base, k)) = topk_out {
        // output suffix: rms_norm in-place on x, then LM head reads
        // directly from x (saves one f16 vector round-trip vs. x_norm staging).
        dispatch_rms_norm(
            session,
            cfg,
            &mut pass,
            1,
            cfg.n_embd,
            m.act_layout.x,
            m.act_layout.x,
            ot.output_norm_off,
        );
        marker.mark(&mut pass, "output_norm");

        dispatch_matvec_q1_0(
            session,
            &mut pass,
            cfg.n_embd,
            cfg.n_vocab,
            WeightSet::Embed,
            ot.lm_head_d,
            ot.lm_head_qs,
            m.act_layout.x,
            m.act_layout.logits,
            false,
        );
        marker.mark(&mut pass, "lm_head");

        dispatch_topk_reduce(
            session,
            &mut pass,
            cfg.n_vocab,
            k,
            m.act_layout.logits,
            topk_out_u32_base,
        );
        marker.mark(&mut pass, "topk_reduce");
    }
    drop(pass);
}

/// Build (but do not submit) one full tg-step `CommandBuffer`: staging upload →
/// embed → all layers → `output_norm` → LM head → `topk_reduce` → sample→readback copy.
///
/// The CB records `copy_buffer_to_buffer(staging[0..4] → sample[0..4])` as the first op
/// and schedules a staging remap on submit. The caller must invoke
/// [`commit_sample_upload`] with the input token id between encoding this CB and submitting
/// it (in pipelined paths this happens after `wait_topk_readback` for the previous step,
/// which fires the prior remap callback so `staging` is mapped again). `topk_reduce` then
/// overwrites `sample[0..2k]` with the K f32 logits + K u32 indices for CPU readback; the
/// two roles never alias inside one CB (embed runs before `topk_reduce`).
pub fn build_step_matvec_topk_cb(session: &Session<'_>, pos: u32, k: u32) -> wgpu::CommandBuffer {
    let k = k.clamp(1, TOPK_MAX);
    let mut se = StepEncoder::new(session);
    encode_sample_upload(session, &mut se.encoder, 0, 4);
    encode_step_matvec(
        &mut se,
        &session.model.cfg,
        0,
        Some((0, k)),
        pos,
        &mut NoMarker,
    );
    let bytes = u64::from(k) * 8;
    se.copy_sample_to_readback(bytes);
    se.schedule_topk_map(bytes);
    se.finish()
}

/// Run one matvec step at `pos`, reading the current token from CPU and
/// returning the top-`k` logits + indices for the next token.
pub fn step_matvec_topk(
    session: &Session<'_>,
    token_id: u32,
    pos: u32,
    k: u32,
) -> Result<(Vec<f32>, Vec<u32>)> {
    let cb = build_step_matvec_topk_cb(session, pos, k);
    commit_sample_upload(session, bytemuck::bytes_of(&token_id));
    session.model.queue.submit(Some(cb));
    wait_topk_readback(session, k)
}

/// Same as [`step_matvec_topk`] but does not perform any sampling readback.
/// Used by perf benches to avoid coupling forward-pass cost to readback I/O —
/// callers `device.poll(wait_indefinitely)` themselves to time the work.
#[cfg(feature = "bench-internals")]
pub fn step_matvec_no_sample(session: &Session<'_>, token_id: u32, pos: u32) {
    let mut se = StepEncoder::new(session);
    upload_sample(session, &mut se.encoder, 0, bytemuck::bytes_of(&token_id));
    // We still encode the topk_reduce dispatch (with k=1, the single argmax case)
    // so the timing reflects real generation cost; we just skip the readback.
    encode_step_matvec(
        &mut se,
        &session.model.cfg,
        0,
        Some((0, 1)),
        pos,
        &mut NoMarker,
    );
    let cb = se.finish();
    session.model.queue.submit(Some(cb));
    // Drive the staging-buffer remap callback so `staging` is mapped for the next call.
    let _ = session.model.device.poll(PollType::wait_indefinitely());
}

/// Pre-KV-copy block of one layer: `rms_norm` → QKV fused → q/k norms → rope.
fn layer_pre_kv_in_pass<M: StepMarker>(
    session: &Session<'_>,
    cfg: &ModelConfig,
    pass: &mut wgpu::ComputePass<'_>,
    il: u32,
    pos: u32,
    marker: &mut M,
) {
    let lt = &session.model.layer_tensors[il as usize];
    // Fused: rms_norm(x) * w_attn_norm → matvec_q1_0_fused (QKV).
    // Replaces a 2-dispatch sequence (rms_norm, matvec_q1_0_fused).
    // x is read directly (NOT x_norm); the kernel stages x to LDS, normalizes
    // in place, and runs the matvec inner loop off the normed shmem.
    dispatch_matvec_q1_0_fused_normed(
        session,
        cfg,
        pass,
        cfg.n_embd,
        session.model.act_layout.x,
        lt.attn_norm_off,
        WeightSet::Attn,
        &[
            (lt.wq.0, lt.wq.1, cfg.q_dim, session.model.act_layout.q),
            (lt.wk.0, lt.wk.1, cfg.kv_dim, session.model.act_layout.k_cur),
            (lt.wv.0, lt.wv.1, cfg.kv_dim, session.model.act_layout.v_cur),
        ],
    );
    marker.mark(pass, "qkv_fused_normed");

    // Q's rms_norm + *w_q_norm + NEOX-RoPE, written back into act.q in place.
    // K's rms_norm + RoPE + Q8_0 quantize + writeback into kv_k, plus V's
    // quantize + writeback into kv_v, all happen inside dispatch_kv_writeback_fused
    // (called from encode_step_matvec).
    dispatch_q_norm_rope_fused(
        session,
        cfg,
        pass,
        session.model.act_layout.q,
        lt.attn_q_norm_off,
        pos,
        1,
    );
    marker.mark(pass, "q_norm_rope");
}

/// Post-KV-copy block of one layer: attention → Wo (resid) → `ffn_norm`
/// → gate-up fused → `silu_mul` → Wd (resid).
fn layer_post_kv_in_pass<M: StepMarker>(
    session: &Session<'_>,
    cfg: &ModelConfig,
    pass: &mut wgpu::ComputePass<'_>,
    il: u32,
    pos: u32,
    marker: &mut M,
) {
    let lt = &session.model.layer_tensors[il as usize];

    // Split-K + GQA-batched flash-attention for tg (m_tokens=1).
    {
        let cur_pos = pos + 1;
        let n_chunks_active = cur_pos.div_ceil(ATTN_CHUNK_SIZE);
        let (d_word, qs_byte) = kv_layer_offsets(cfg, session.model.max_seq, il);

        let ps = AttnSplitParams {
            head_dim: cfg.head_dim,
            n_head: cfg.n_head,
            n_kv_head: cfg.n_kv_head,
            pos: cur_pos,
            kv_stride: cfg.kv_dim,
            q_offset: session.model.act_layout.q,
            k_d_word_offset: d_word,
            k_qs_byte_offset: qs_byte,
            v_d_word_offset: d_word,
            v_qs_byte_offset: qs_byte,
            n_chunks_active,
            scale: 1.0 / (cfg.head_dim as f32).sqrt(),
        };
        pass.set_pipeline(&session.model.pipes.attention_split);
        pass.set_bind_group(0, &session.state.cached.attn_split, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&ps));
        pass.dispatch_workgroups(cfg.n_kv_head, n_chunks_active, 1);
        marker.mark(pass, "attn_split");

        let pm = AttnMergeParams {
            head_dim: cfg.head_dim,
            n_head: cfg.n_head,
            out_offset: session.model.act_layout.attn_out,
            n_chunks_active,
        };
        pass.set_pipeline(&session.model.pipes.attention_merge);
        pass.set_bind_group(0, &session.state.cached.attn_merge, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&pm));
        pass.dispatch_workgroups(cfg.n_head, 1, 1);
        marker.mark(pass, "attn_merge");
    }

    dispatch_matvec_q1_0(
        session,
        pass,
        cfg.q_dim,
        cfg.n_embd,
        WeightSet::Attn,
        lt.wo.0,
        lt.wo.1,
        session.model.act_layout.attn_out,
        session.model.act_layout.x,
        true, /*accumulate*/
    );
    marker.mark(pass, "wo");

    // Fused: rms_norm(x) * w_ffn_norm → matvec_q1_0_fused (gate+up).
    // Replaces a 2-dispatch sequence (rms_norm, matvec_q1_0_fused).
    dispatch_matvec_q1_0_fused_normed(
        session,
        cfg,
        pass,
        cfg.n_embd,
        session.model.act_layout.x,
        lt.ffn_norm_off,
        WeightSet::FfnGU,
        &[
            (lt.wg.0, lt.wg.1, cfg.n_ff, session.model.act_layout.gate),
            (lt.wu.0, lt.wu.1, cfg.n_ff, session.model.act_layout.up),
        ],
    );
    marker.mark(pass, "gate_up_fused_normed");

    // Fused: silu(gate) * up on the input side of Wd, in one dispatch (no
    // ffn_in round-trip, no standalone silu_mul). The standalone silu_mul
    // shader and the matmul-prefill path's `silu_mul -> matmul_q1_0_q8_0`
    // pair are unchanged — this fusion is matvec-path-only.
    dispatch_matvec_q1_0_silu(
        session,
        pass,
        cfg.n_ff,
        cfg.n_embd,
        WeightSet::FfnD,
        lt.wd.0,
        lt.wd.1,
        session.model.act_layout.gate,
        session.model.act_layout.up,
        session.model.act_layout.x,
        true, /*accumulate*/
    );
    marker.mark(pass, "wd_silu");
}

/// Run the matvec single-token path over every token in `prompt`, advancing
/// `pos` from `pos_base` to `pos_base + prompt.len()`, and return the top-K
/// candidates from the LAST token's logits. Suitable for incremental prefill
/// after an existing KV cache (any `pos_base`).
pub fn prefill_matvec_loop_topk(
    session: &Session<'_>,
    prompt: &[u32],
    pos_base: u32,
    k: u32,
) -> Result<(Vec<f32>, Vec<u32>)> {
    // Fill the KV cache for the (n - 1) non-last tokens by chunking them into
    // separate command buffers + submits. Two reasons we don't pack the whole
    // thing into one CB:
    //   1. The `sample` buffer holds only 1024 u32 input slots, so a single
    //      up-front write capped at `rest.len() == n - 1` would overflow on
    //      long re-prefills (e.g. Qwen3 thinking strip-and-rewind in
    //      examples/chat.rs, which can re-feed > 1k tokens at once).
    //   2. A single command buffer that encodes hundreds of full
    //      transformer-step passes can exceed the GPU's lockup-detection
    //      timeout on wider models (8B), causing a context loss.
    // CHUNK is chosen so each submit is well under both limits and the
    // per-submit overhead (~tens of µs) is negligible vs. the in-CB work.
    const CHUNK: usize = 256;
    let Some((&last, rest)) = prompt.split_last() else {
        return Err(PotError::PrefillTooLarge {
            n: 0,
            max: session.model.m_max,
        });
    };
    // We pass `topk_out=None` for these steps so the suffix
    // (output_norm + LM head + topk_reduce) is skipped — those logits are
    // thrown away anyway, and skipping the topk avoids it stomping on the
    // prompt region of `sample`.
    for chunk_start in (0..rest.len()).step_by(CHUNK) {
        let chunk_end = (chunk_start + CHUNK).min(rest.len());
        let chunk = &rest[chunk_start..chunk_end];
        // Upload this chunk's tokens to sample[0..chunk.len()]; the embed
        // shader for step `i` reads `sample[i]`.
        let mut se = StepEncoder::new(session);
        upload_sample(session, &mut se.encoder, 0, bytemuck::cast_slice(chunk));
        for (i, t) in (chunk_start..chunk_end).enumerate() {
            encode_step_matvec(
                &mut se,
                &session.model.cfg,
                /*sample_in=*/ i as u32,
                /*topk_out=*/ None,
                /*pos=*/ pos_base + t as u32,
                &mut NoMarker,
            );
        }
        let cb = se.finish();
        session.model.queue.submit(Some(cb));
        // Drive the staging buffer remap callback so staging is mapped for the next upload.
        let _ = session.model.device.poll(PollType::wait_indefinitely());
    }
    step_matvec_topk(session, last, pos_base + rest.len() as u32, k)
}

// ---------- matmul (batched prefill) ---------------------------------------

/// Batched matmul prefill of `prompt` starting from KV-cache position
/// `pos_base`. Advances pos from `pos_base` to `pos_base + prompt.len()`.
/// Returns top-K candidates from the last token's logits.
pub fn prefill_matmul_topk<M: StepMarker>(
    session: &Session<'_>,
    prompt: &[u32],
    pos_base: u32,
    k: u32,
    marker: &mut M,
) -> Result<(Vec<f32>, Vec<u32>)> {
    let m = prompt.len() as u32;
    if m == 0 || m > session.model.m_max {
        return Err(PotError::PrefillTooLarge {
            n: m,
            max: session.model.m_max,
        });
    }
    let cfg = &session.model.cfg;
    let ot = &session.model.output_tensors;
    let k = k.clamp(1, TOPK_MAX);

    // ---- All phases (embed → per-layer transformer → final norm/LM-head/topk
    //      → readback copy) into ONE command buffer / ONE submit, with all
    //      compute dispatches sharing ONE pass to amortize the
    //      begin_compute_pass cost (~25us each on RADV).
    let mut se = StepEncoder::new(session);
    upload_sample(session, &mut se.encoder, 0, bytemuck::cast_slice(prompt));

    {
        let mut desc = wgpu::ComputePassDescriptor {
            label: Some("prefill_matmul"),
            timestamp_writes: None,
        };
        marker.setup_desc(&mut desc);
        let mut pass = se.encoder.begin_compute_pass(&desc);
        marker.mark(&mut pass, "start");

        // Phase 1: embed all M tokens (one dispatch).
        let p = EmbedParams {
            k: cfg.n_embd,
            d_offset: ot.token_embd_d,
            qs_offset: ot.token_embd_qs,
            output_offset: session.model.act_layout.x,
            sample_offset: 0,
        };
        pass.set_pipeline(&session.model.pipes.embed);
        pass.set_bind_group(0, &session.state.cached.embed, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(m, 1, 1);

        marker.mark(&mut pass, "embed");

        // Phase 2: per-layer transformer.
        for il in 0..cfg.n_layer {
            layer_step_matmul_in_pass(session, cfg, &mut pass, il, m, pos_base, marker);
        }

        // Phase 3: output_norm (last token, in-place) + LM head + topk_reduce.
        let last_x = session.model.act_layout.x + (m - 1) * cfg.n_embd;
        dispatch_rms_norm(
            session,
            cfg,
            &mut pass,
            1,
            cfg.n_embd,
            last_x,
            last_x,
            ot.output_norm_off,
        );
        marker.mark(&mut pass, "output_norm");
        dispatch_matvec_q1_0(
            session,
            &mut pass,
            cfg.n_embd,
            cfg.n_vocab,
            WeightSet::Embed,
            ot.lm_head_d,
            ot.lm_head_qs,
            last_x,
            session.model.act_layout.logits,
            false,
        );
        marker.mark(&mut pass, "lm_head");
        dispatch_topk_reduce(
            session,
            &mut pass,
            cfg.n_vocab,
            k,
            session.model.act_layout.logits,
            0,
        );
        marker.mark(&mut pass, "topk_reduce");
    }

    // Phase 4: append readback copy + schedule map, all in the same command buffer.
    let bytes = u64::from(k) * 8;
    se.copy_sample_to_readback(bytes);
    se.schedule_topk_map(bytes);

    let cb = se.finish();
    session.model.queue.submit(Some(cb));

    wait_topk_readback(session, k)
}

fn layer_step_matmul_in_pass<M: StepMarker>(
    session: &Session<'_>,
    cfg: &ModelConfig,
    pass: &mut wgpu::ComputePass<'_>,
    il: u32,
    m: u32,
    pos_base: u32,
    marker: &mut M,
) {
    let lt = &session.model.layer_tensors[il as usize];

    // attn_norm fused with Q8_0 quantize (writes act_q8 directly).
    let (a_d, a_qs) = dispatch_rms_norm_q8_0(
        session,
        cfg,
        pass,
        cfg.n_embd,
        m,
        session.model.act_layout.x,
        lt.attn_norm_off,
    );
    marker.mark(pass, "rms_norm_q8");
    dispatch_matmul_q1_0(
        session,
        pass,
        cfg.n_embd,
        cfg.q_dim,
        m,
        WeightSet::Attn,
        lt.wq.0,
        lt.wq.1,
        a_d,
        a_qs,
        session.model.act_layout.q,
        false,
    );
    marker.mark(pass, "wq_matmul");
    dispatch_matmul_q1_0(
        session,
        pass,
        cfg.n_embd,
        cfg.kv_dim,
        m,
        WeightSet::Attn,
        lt.wk.0,
        lt.wk.1,
        a_d,
        a_qs,
        session.model.act_layout.k_cur,
        false,
    );
    marker.mark(pass, "wk_matmul");
    dispatch_matmul_q1_0(
        session,
        pass,
        cfg.n_embd,
        cfg.kv_dim,
        m,
        WeightSet::Attn,
        lt.wv.0,
        lt.wv.1,
        a_d,
        a_qs,
        session.model.act_layout.v_cur,
        false,
    );
    marker.mark(pass, "wv_matmul");

    // Q/K rms+rope (in-place) + KV writeback into kv_{k,v}.
    dispatch_q_norm_rope_fused(
        session,
        cfg,
        pass,
        session.model.act_layout.q,
        lt.attn_q_norm_off,
        pos_base,
        m,
    );
    marker.mark(pass, "q_norm_rope");
    dispatch_kv_writeback_fused(
        session,
        cfg,
        pass,
        session.model.act_layout.k_cur,
        session.model.act_layout.v_cur,
        lt.attn_k_norm_off,
        il,
        pos_base,
        m,
    );
    marker.mark(pass, "kv_writeback");

    // Attention (Q-tiled FA-2 prefill, Q8_0 output written directly to act_q8).
    let (a_d2, a_qs2) = dispatch_attention_prefill_tiled(
        session,
        cfg,
        pass,
        il,
        session.model.max_seq,
        m,
        pos_base,
    );
    marker.mark(pass, "attention");

    // Wo (residual) + ffn_norm + gate/up + silu_mul_q8 + Wd (residual).
    dispatch_matmul_q1_0(
        session,
        pass,
        cfg.q_dim,
        cfg.n_embd,
        m,
        WeightSet::Attn,
        lt.wo.0,
        lt.wo.1,
        a_d2,
        a_qs2,
        session.model.act_layout.x,
        true,
    );
    marker.mark(pass, "wo_matmul");
    let (a_d3, a_qs3) = dispatch_rms_norm_q8_0(
        session,
        cfg,
        pass,
        cfg.n_embd,
        m,
        session.model.act_layout.x,
        lt.ffn_norm_off,
    );
    marker.mark(pass, "rms_norm_q8");
    dispatch_matmul_q1_0(
        session,
        pass,
        cfg.n_embd,
        cfg.n_ff,
        m,
        WeightSet::FfnGU,
        lt.wg.0,
        lt.wg.1,
        a_d3,
        a_qs3,
        session.model.act_layout.gate,
        false,
    );
    marker.mark(pass, "wg_matmul");
    dispatch_matmul_q1_0(
        session,
        pass,
        cfg.n_embd,
        cfg.n_ff,
        m,
        WeightSet::FfnGU,
        lt.wu.0,
        lt.wu.1,
        a_d3,
        a_qs3,
        session.model.act_layout.up,
        false,
    );
    marker.mark(pass, "wu_matmul");
    let (a_d4, a_qs4) = dispatch_silu_mul_q8_0(
        session,
        pass,
        cfg.n_ff,
        m,
        session.model.act_layout.gate,
        session.model.act_layout.up,
    );
    marker.mark(pass, "silu_mul_q8");
    dispatch_matmul_q1_0(
        session,
        pass,
        cfg.n_ff,
        cfg.n_embd,
        m,
        WeightSet::FfnD,
        lt.wd.0,
        lt.wd.1,
        a_d4,
        a_qs4,
        session.model.act_layout.x,
        true,
    );
    marker.mark(pass, "wd_matmul");
}

// =========================================================================
// Bench / microbench (CLI utilities; gated behind the bench-internals feature
// so they don't pollute the public surface). Lives in a separate file but
// stays a child module of `forward` so the helpers can reach private items
// here via `super::`.
// =========================================================================

/// Per-dispatch GPU timestamp marker for microbench (per-kernel breakdown).
///
/// Call [`MicroMarker::mark`] between dispatches inside a compute pass to write
/// GPU-side timestamps. After the command buffer containing those dispatches has
/// been submitted and fully polled (GPU done), call [`MicroMarker::resolve`] to
/// read the per-span durations in nanoseconds.
///
/// Each `pass.write_timestamp` forces a flush at the dispatch boundary on most
/// drivers (RADV included), which slows the GPU work itself — so this marker is
/// only appropriate when you actually want a per-kernel breakdown. For clean
/// whole-pass timing of pp/tg, use [`BenchMarker`] instead, which installs the
/// timestamps via the pass descriptor (no per-dispatch flushes).
///
/// Each `MicroMarker` reuses `session.state.bench_query_set` starting from slot 0,
/// so only one `MicroMarker` may be live at a time (the previous one must be
/// resolved before a new one is marked into the same query set).
#[cfg(all(
    feature = "bench-internals",
    not(feature = "ci"),
    not(target_vendor = "apple")
))]
pub struct MicroMarker<'a> {
    session: &'a Session<'a>,
    next_idx: u32,
    labels: Vec<&'static str>,
}

#[cfg(all(
    feature = "bench-internals",
    not(feature = "ci"),
    not(target_vendor = "apple")
))]
impl<'a> MicroMarker<'a> {
    pub const fn new(session: &'a Session<'a>) -> Self {
        Self {
            session,
            next_idx: 0,
            labels: Vec::new(),
        }
    }

    /// After the step CB has been submitted and polled (GPU work done),
    /// resolve all timestamps and return `(label, duration_ns)` spans.
    /// Each span is the GPU time between the previous and current mark.
    /// The "start" sentinel is consumed but not returned as a span.
    pub fn resolve(self) -> Result<Vec<(&'static str, f32)>> {
        let n = self.next_idx;
        if n < 2 {
            return Ok(vec![]);
        }
        let ticks = bench_resolve_ticks(self.session, n)?;
        let period = self.session.model.bench_ts_period_ns;
        let mut spans: Vec<(&'static str, f32)> = Vec::with_capacity((n - 1) as usize);
        for i in 1..n as usize {
            let dt_ns = ticks[i].saturating_sub(ticks[i - 1]) as f32 * period;
            spans.push((self.labels[i], dt_ns));
        }
        Ok(spans)
    }
}

#[cfg(all(
    feature = "bench-internals",
    not(feature = "ci"),
    not(target_vendor = "apple")
))]
impl StepMarker for MicroMarker<'_> {
    #[inline(always)]
    fn setup_desc<'a>(&'a self, _desc: &mut wgpu::ComputePassDescriptor<'a>) {}

    /// Write a GPU timestamp at the current slot and associate `label` with it.
    /// The label conventionally names the kernel that just COMPLETED (the slot
    /// before the first label is the pass start sentinel named "start").
    fn mark(&mut self, pass: &mut wgpu::ComputePass<'_>, label: &'static str) {
        use crate::model::BENCH_QS_SLOTS;
        assert!(
            self.next_idx < BENCH_QS_SLOTS,
            "MicroMarker: exceeded BENCH_QS_SLOTS"
        );
        pass.write_timestamp(&self.session.state.bench_query_set, self.next_idx);
        self.labels.push(label);
        self.next_idx += 1;
    }
}

/// Whole-pass GPU timestamp marker for end-to-end pp/tg timing.
///
/// Installs `timestamp_writes` on the [`wgpu::ComputePassDescriptor`] (slots
/// 0=begin, 1=end of pass) via [`StepMarker::setup_desc`]; [`mark`] is a no-op.
/// Unlike [`MicroMarker`], no per-dispatch `pass.write_timestamp` calls are
/// inserted, so the GPU work itself isn't slowed by mid-pass flushes — what we
/// measure is the bare execution time of the pass.
///
/// Reuses `session.state.bench_query_set` slots 0/1, so only one `BenchMarker`
/// may be live at a time and the pass it instruments must be the only one to
/// write those slots between `new` and `resolve`.
///
/// [`mark`]: BenchMarker::mark
#[cfg(all(feature = "bench-internals", not(feature = "ci")))]
pub struct BenchMarker<'a> {
    session: &'a Session<'a>,
}

#[cfg(all(feature = "bench-internals", not(feature = "ci")))]
impl<'a> BenchMarker<'a> {
    pub const fn new(session: &'a Session<'a>) -> Self {
        Self { session }
    }

    /// After the instrumented pass has been submitted and polled (GPU done),
    /// resolve the begin/end timestamps and return total GPU duration in
    /// nanoseconds.
    pub fn resolve(self) -> Result<f32> {
        let ticks = bench_resolve_ticks(self.session, 2)?;
        Ok(ticks[1].saturating_sub(ticks[0]) as f32 * self.session.model.bench_ts_period_ns)
    }
}

#[cfg(all(feature = "bench-internals", not(feature = "ci")))]
impl StepMarker for BenchMarker<'_> {
    fn setup_desc<'a>(&'a self, desc: &mut wgpu::ComputePassDescriptor<'a>) {
        desc.timestamp_writes = Some(wgpu::ComputePassTimestampWrites {
            query_set: &self.session.state.bench_query_set,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        });
    }
    #[inline(always)]
    fn mark(&mut self, _pass: &mut wgpu::ComputePass<'_>, _label: &'static str) {}
}

/// Resolve `n` timestamps from `bench_query_set[0..n]` to host memory. Shared
/// by [`MicroMarker::resolve`] and [`BenchMarker::resolve`].
#[cfg(all(feature = "bench-internals", not(feature = "ci")))]
fn bench_resolve_ticks(session: &Session<'_>, n: u32) -> Result<Vec<u64>> {
    let bytes = u64::from(n) * 8;
    let mut enc = session
        .model
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bench_resolve"),
        });
    enc.resolve_query_set(
        &session.state.bench_query_set,
        0..n,
        &session.state.bench_resolve,
        0,
    );
    enc.copy_buffer_to_buffer(
        &session.state.bench_resolve,
        0,
        &session.state.bench_readback,
        0,
        bytes,
    );
    enc.map_buffer_on_submit(
        &session.state.bench_readback,
        wgpu::MapMode::Read,
        0..bytes,
        |_| {},
    );
    session.model.queue.submit(Some(enc.finish()));

    let slice = session.state.bench_readback.slice(0..bytes);
    if let Err(e) = session.model.device.poll(PollType::wait_indefinitely()) {
        session.model.check_device()?;
        return Err(PotError::Poll(e));
    }

    let data = slice.get_mapped_range();
    let ticks: Vec<u64> = bytemuck::cast_slice::<_, u64>(&data[..bytes as usize]).to_vec();
    drop(data);
    session.state.bench_readback.unmap();
    Ok(ticks)
}

#[cfg(feature = "bench-internals")]
#[path = "bench.rs"]
#[allow(
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    reason = "internal"
)]
pub mod bench_internals;
