//! Forward pass and per-step inference helpers.
//!
//! Two execution modes:
//!   - **matvec single-token** (`step_matvec_topk`, `prefill_matvec_loop_topk`):
//!     processes one token at a time via the multiply-free `Q1_0` matvec kernel.
//!     Used for token generation and for incremental prefill (when there's an
//!     existing KV cache prefix).
//!   - **matmul batched prefill** (`prefill_matmul_topk`): processes the prompt
//!     as one batch using `dot4I8Packed` matmul with a `Q8_0` quantize-activation
//!     pre-pass. Faster on long prompts, but assumes the KV cache is empty
//!     (`pos_base == 0`).
//!
//! Sampling lives outside this module: each entry point ends with a
//! `topk_reduce` GPU dispatch that writes the top-K logit values + indices to
//! the `sample` buffer. The caller reads them back and applies its own
//! temperature / top-p / multinomial logic on CPU.
//!
//! Per-token encoder hot path: per-layer tensor offsets are precomputed at
//! load time (`Model::layer_tensors`) so we don't re-do
//! `format!` + `HashMap` lookup + clone on every dispatch; bind groups are also
//! precomputed at load time (`Model::cached`) and shared across all dispatches
//! of a given (kind, weight buffer) pair, since the dynamic uniform offset is
//! the only per-dispatch variation.

use std::result::Result as StdResult;
use std::sync::{Arc, OnceLock};

use wgpu::PollType;

use crate::error::{PotError, Result};
use crate::model::{
    ATTN_CHUNK_SIZE, AttnMergeParams, AttnParams, AttnSplitParams, Config, EmbedParams,
    KvWritebackFusedParams, MatmulParams, MatvecFusedNormedParams, MatvecParams, MatvecSiluParams,
    Model, QNormRopeFusedParams, QuantParams, RmsNormParams, SiluMulParams, TOPK_MAX, TopKParams,
    WeightSet,
};

type MapSlot = Arc<OnceLock<StdResult<(), wgpu::BufferAsyncError>>>;

// ---------- Q8_0 KV cache layout helpers ------------------------------------
// Each kv_{k,v} buffer carries:
//   d-section  (FP32 scales): bytes [0, n_layer * max_seq * (kv_dim/32) * 4)
//   qs-section (i8 packed):   bytes [d_total, d_total + n_layer * max_seq * kv_dim)
// The block index for (layer il, position pos, block b in [0, kv_dim/32)) is
// `(il * max_seq + pos) * (kv_dim/32) + b` (single-row stride = kv_dim/32 in
// d-elements, kv_dim in qs-bytes).

const fn kv_qs_byte_base(cfg: &Config, max_seq: u32) -> u32 {
    cfg.n_layer * max_seq * (cfg.kv_dim / 32) * 4
}

/// `(d_word_offset, qs_byte_offset)` for the start of layer `il` inside each
/// kv buffer. Same layout for `kv_k` and `kv_v`, so callers reuse the pair.
const fn kv_layer_offsets(cfg: &Config, max_seq: u32, il: u32) -> (u32, u32) {
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
    model: &'a Model,
    pub(crate) encoder: wgpu::CommandEncoder,
}

impl<'a> StepEncoder<'a> {
    pub fn new(model: &'a Model) -> Self {
        let encoder = model
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("step"),
            });
        Self { model, encoder }
    }

    /// Append a `sample → readback` copy to this encoder so the readback
    /// transfer rides in the same command buffer as the step it follows.
    /// Avoids a separate submit purely for the copy.
    pub fn copy_sample_to_readback(&mut self, bytes: u64) {
        self.encoder.copy_buffer_to_buffer(
            &self.model.buffers.sample,
            0,
            &self.model.buffers.readback,
            0,
            bytes,
        );
    }

    /// Schedule a `MAP_READ` mapping of the readback buffer on submission.
    /// Must be called after [`copy_sample_to_readback`] and before [`finish`].
    /// Returns the slot that [`wait_topk_readback`] will drain.
    pub fn schedule_topk_map(&self, bytes: u64) -> MapSlot {
        let slot: MapSlot = Arc::new(OnceLock::new());
        let slot2 = slot.clone();
        self.encoder.map_buffer_on_submit(
            &self.model.buffers.readback,
            wgpu::MapMode::Read,
            0..bytes,
            move |res| {
                let _ = slot2.set(res);
            },
        );
        slot
    }

    /// Stage `data` into `model.buffers.sample[offset..]` via the belt.
    /// `data.len()` and `offset` must be multiples of 4.
    pub fn write_sample(&mut self, offset: u64, data: &[u8]) {
        self.model
            .belt_write(&mut self.encoder, &self.model.buffers.sample, offset, data);
    }

    pub fn finish(self) -> wgpu::CommandBuffer {
        self.encoder.finish()
    }
}

// ---------- weight-set selection -------------------------------------------

const fn matvec_bg(model: &Model, ws: WeightSet) -> &wgpu::BindGroup {
    match ws {
        WeightSet::Attn => &model.cached.matvec_w_attn,
        WeightSet::FfnGU => &model.cached.matvec_w_ffn_gu,
        WeightSet::FfnD => &model.cached.matvec_w_ffn_d,
        WeightSet::Embed => &model.cached.matvec_w_embed,
    }
}

/// Bind group for `matvec_q1_0_fused_normed`. Only `Attn` (QKV) and `FfnGU`
/// (gate+up) are valid: those are the two sites in the matvec single-token
/// path that are preceded by an `rms_norm` and use the fused matvec.
#[allow(
    clippy::panic,
    reason = "internal invariant: only Attn / FfnGU are wired up"
)]
const fn matvec_fused_normed_bg(model: &Model, ws: WeightSet) -> &wgpu::BindGroup {
    match ws {
        WeightSet::Attn => &model.cached.matvec_fused_normed_w_attn,
        WeightSet::FfnGU => &model.cached.matvec_fused_normed_w_ffn_gu,
        _ => panic!("matvec_fused_normed only supports WeightSet::Attn / FfnGU"),
    }
}

const fn matmul_bg(model: &Model, ws: WeightSet) -> &wgpu::BindGroup {
    match ws {
        WeightSet::Attn => &model.cached.matmul_w_attn,
        WeightSet::FfnGU => &model.cached.matmul_w_ffn_gu,
        WeightSet::FfnD => &model.cached.matmul_w_ffn_d,
        WeightSet::Embed => &model.cached.matmul_w_embed,
    }
}

// ---------- in-pass kernel dispatch helpers ---------------------------------
// These variants take a `&mut wgpu::ComputePass` already opened by the caller,
// allowing many dispatches to share one pass and amortize the
// begin_compute_pass cost (~25us each on RADV).

fn dispatch_rms_norm(
    model: &Model,
    cfg: &Config,
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
    pass.set_pipeline(&model.pipes.rms_norm);
    pass.set_bind_group(0, &model.cached.rms_norm, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&p));
    pass.dispatch_workgroups(n_groups, 1, 1);
}

fn dispatch_matvec_q1_0(
    model: &Model,
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
    const ROWS_PER_WG: u32 = 8;
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
    pass.set_pipeline(&model.pipes.matvec);
    pass.set_bind_group(0, matvec_bg(model, weights), &[]);
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
    model: &Model,
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
    const ROWS_PER_WG: u32 = 16;
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
    pass.set_pipeline(&model.pipes.matvec_silu);
    pass.set_bind_group(0, matvec_bg(model, weights), &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&p));
    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
}

/// Fused: `rms_norm(x) * w_norm` → multi-range `Q1_0` matvec, in one dispatch.
/// Replaces `dispatch_rms_norm + dispatch_matvec_q1_0_fused` for the matvec
/// single-token path. See `shaders/matvec_q1_0_fused_normed.wgsl`.
fn dispatch_matvec_q1_0_fused_normed(
    model: &Model,
    cfg: &Config,
    pass: &mut wgpu::ComputePass<'_>,
    k: u32,
    input_offset: u32,
    w_norm_off: u32,
    weights: WeightSet,
    ranges: &[(u32, u32, u32, u32)],
) {
    const ROWS_PER_WG: u32 = 16;
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
    pass.set_pipeline(&model.pipes.matvec_fused_normed);
    pass.set_bind_group(0, matvec_fused_normed_bg(model, weights), &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&p));
    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
}

fn dispatch_topk_reduce(
    model: &Model,
    pass: &mut wgpu::ComputePass<'_>,
    n: u32,
    k: u32,
    in_off: u32,
    out_off_u32: u32,
) {
    let p = TopKParams {
        n,
        in_offset: in_off,
        out_offset: out_off_u32,
        k,
    };
    pass.set_pipeline(&model.pipes.topk_reduce);
    pass.set_bind_group(0, &model.cached.topk_reduce, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&p));
    pass.dispatch_workgroups(1, 1, 1);
}

/// Fused: `rms_norm(K` head) → \*`w_k_norm` → NEOX-RoPE → `Q8_0` quantize → write
/// `kv_k`. V runs in the same workgroup (just quantize + write `kv_v`). Replaces
/// `rms_norm(K) + rope(K) + kv_writeback` with one dispatch.
fn dispatch_kv_writeback_fused(
    model: &Model,
    cfg: &Config,
    pass: &mut wgpu::ComputePass<'_>,
    k_cur_off: u32,
    v_cur_off: u32,
    w_k_norm_off: u32,
    layer_il: u32,
    pos_base: u32,
    m_tokens: u32,
) {
    let nb_per_row = cfg.kv_dim / 32;
    let (dst_d_word_offset, dst_qs_byte_offset) = kv_layer_offsets(cfg, model.max_seq, layer_il);
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
    pass.set_pipeline(&model.pipes.kv_writeback_fused);
    pass.set_bind_group(0, &model.cached.kv_writeback_fused, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&p));
    pass.dispatch_workgroups(cfg.n_kv_head, m_tokens, 1);
}

/// Fused: `rms_norm(Q` head) → \*`w_q_norm` → NEOX-RoPE, written back into
/// `act.q` in place. Replaces `rms_norm(Q) + rope(Q)` with one dispatch.
fn dispatch_q_norm_rope_fused(
    model: &Model,
    cfg: &Config,
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
    pass.set_pipeline(&model.pipes.q_norm_rope_fused);
    pass.set_bind_group(0, &model.cached.q_norm_rope_fused, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&p));
    pass.dispatch_workgroups(cfg.n_head, m_tokens, 1);
}

fn dispatch_quantize_act(
    model: &Model,
    pass: &mut wgpu::ComputePass<'_>,
    k: u32,
    m: u32,
    in_off_f32: u32,
) -> (u32, u32) {
    let nb_q8 = k / 32;
    let d_off = 0u32;
    let qs_off = m * nb_q8 * 4;
    let total = m * nb_q8;
    let dispatch_x = total.min(65535);
    let dispatch_y = total.div_ceil(dispatch_x);
    let p = QuantParams {
        k,
        m,
        input_offset: in_off_f32,
        d_offset: d_off,
        qs_offset: qs_off,
        dispatch_x_dim: dispatch_x,
    };
    pass.set_pipeline(&model.pipes.quantize);
    pass.set_bind_group(0, &model.cached.quantize, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&p));
    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    (d_off, qs_off)
}

fn dispatch_matmul_q1_0(
    model: &Model,
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
    pass.set_pipeline(&model.pipes.matmul);
    pass.set_bind_group(0, matmul_bg(model, weights), &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&p));
    pass.dispatch_workgroups(n.div_ceil(64), m.div_ceil(64), 1);
}

fn dispatch_silu_mul(
    model: &Model,
    pass: &mut wgpu::ComputePass<'_>,
    n: u32,
    m: u32,
    gate_off: u32,
    up_off: u32,
    out_off: u32,
) {
    let total = n * m;
    let groups = total.div_ceil(64);
    let dispatch_x = groups.min(65535);
    let dispatch_y = groups.div_ceil(dispatch_x);
    let p = SiluMulParams {
        n,
        m,
        gate_offset: gate_off,
        up_offset: up_off,
        out_offset: out_off,
        dispatch_x_count: dispatch_x * 64,
    };
    pass.set_pipeline(&model.pipes.silu_mul);
    pass.set_bind_group(0, &model.cached.silu_mul, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&p));
    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
}

fn dispatch_attention(
    model: &Model,
    cfg: &Config,
    pass: &mut wgpu::ComputePass<'_>,
    layer_il: u32,
    max_seq: u32,
    pos: u32,
    m_tokens: u32,
    is_prefill: bool,
) {
    let (d_word, qs_byte) = kv_layer_offsets(cfg, max_seq, layer_il);
    let p = AttnParams {
        head_dim: cfg.head_dim,
        n_head: cfg.n_head,
        n_kv_head: cfg.n_kv_head,
        pos,
        kv_stride: cfg.kv_dim,
        q_offset: model.act_layout.q,
        k_d_word_offset: d_word,
        k_qs_byte_offset: qs_byte,
        v_d_word_offset: d_word,
        v_qs_byte_offset: qs_byte,
        out_offset: model.act_layout.attn_out,
        scale: 1.0 / (cfg.head_dim as f32).sqrt(),
        m_tokens,
        is_prefill: u32::from(is_prefill),
    };
    pass.set_pipeline(&model.pipes.attention);
    pass.set_bind_group(0, &model.cached.attn, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&p));
    pass.dispatch_workgroups(cfg.n_head, m_tokens, 1);
}

// ---------- async readback helper -------------------------------------------

/// Wait for the readback mapping scheduled via [`StepEncoder::schedule_topk_map`]
/// to complete and return the K f32 logits + K u32 indices.
///
/// `slot` must have been obtained from `schedule_topk_map` on the encoder that
/// produced the submitted command buffer. The mapping fires at submit time;
/// this call just polls until GPU work finishes.
pub fn wait_topk_readback(model: &Model, k: u32, slot: MapSlot) -> Result<(Vec<f32>, Vec<u32>)> {
    let bytes = u64::from(k) * 8; // K f32 + K u32
    let slice = model.buffers.readback.slice(0..bytes);
    if let Err(e) = model.device.poll(PollType::wait_indefinitely()) {
        model.check_device()?;
        return Err(PotError::Poll(e));
    }
    match slot.get() {
        Some(Ok(())) => {}
        Some(Err(e)) => {
            model.check_device()?;
            return Err(PotError::BufferMap(e.clone()));
        }
        None => unreachable!("map_async callback did not fire before poll returned"),
    }
    drop(slot);
    let data = slice.get_mapped_range();
    let words: &[u32] = bytemuck::cast_slice(&data[..bytes as usize]);
    let logits: Vec<f32> = words[..k as usize]
        .iter()
        .map(|w| f32::from_bits(*w))
        .collect();
    let indices: Vec<u32> = words[k as usize..2 * k as usize].to_vec();
    drop(data);
    model.buffers.readback.unmap();
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
    cfg: &Config,
    sample_in: u32,
    topk_out: Option<(u32, u32)>,
    pos: u32,
    marker: &mut M,
) {
    let StepEncoder { model: m, encoder } = se;
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
        pass.set_bind_group(0, &m.cached.embed, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(1, 1, 1);
    }
    marker.mark(&mut pass, "embed");

    for il in 0..cfg.n_layer {
        layer_pre_kv_in_pass(m, cfg, &mut pass, il, pos, marker);
        // Fused: K (rms_norm + *w_k_norm + RoPE + Q8_0 quantize) and V (Q8_0
        // quantize) → write both into kv_{k,v}. Replaces the previous
        // rms_norm(K) + rope(K) + kv_writeback trio (3 dispatches → 1).
        let lt = &m.layer_tensors[il as usize];
        dispatch_kv_writeback_fused(
            m,
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
        layer_post_kv_in_pass(m, cfg, &mut pass, il, pos, marker);
    }
    if let Some((topk_out_u32_base, k)) = topk_out {
        // output suffix: rms_norm in-place on x, then LM head reads
        // directly from x (saves one f16 vector round-trip vs. x_norm staging).
        dispatch_rms_norm(
            m,
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
            m,
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
            m,
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

/// Build (but do not submit) one full tg-step `CommandBuffer`: embed → all
/// layers → `output_norm` → LM head → `topk_reduce` → sample→readback copy.
///
/// Stages `token_id` into `sample[0]` via the staging belt — the copy is
/// encoded at the start of the CB before the embed dispatch reads it, so
/// wgpu's implicit barrier ensures ordering. Callers must call
/// `model.belt_finish()` before submitting the returned `CommandBuffer`, and
/// `model.belt_recall()` after.
pub fn build_step_matvec_topk_cb(
    model: &Model,
    token_id: u32,
    pos: u32,
    k: u32,
) -> (wgpu::CommandBuffer, MapSlot) {
    let k = k.clamp(1, TOPK_MAX);
    let mut se = StepEncoder::new(model);
    se.write_sample(0, bytemuck::bytes_of(&token_id));
    encode_step_matvec(&mut se, &model.cfg, 0, Some((0, k)), pos, &mut NoMarker);
    let bytes = u64::from(k) * 8;
    se.copy_sample_to_readback(bytes);
    let slot = se.schedule_topk_map(bytes);
    (se.finish(), slot)
}

/// Like [`build_step_matvec_topk_cb`] but does not stage the token — the
/// returned `wgpu::BufferViewMut` is a host-mapped 4-byte slot in the staging
/// chunk. The caller must:
///   1. fill the view with the chosen token bytes (`view.copy_from_slice(...)`),
///   2. drop the view (unregisters the mapped range),
///   3. call `model.belt_finish()` (unmaps the chunk),
///   4. submit the `CommandBuffer`,
///   5. call `model.belt_recall()`.
pub fn build_step_matvec_topk_cb_deferred(
    model: &Model,
    pos: u32,
    k: u32,
) -> (wgpu::CommandBuffer, MapSlot, wgpu::BufferViewMut) {
    let k = k.clamp(1, TOPK_MAX);
    let mut se = StepEncoder::new(model);
    let view = {
        // BufferSize(4) is a non-zero constant; NonZeroU64::new(4) is always Some.
        const TOKEN_ID_SIZE: wgpu::BufferSize = wgpu::BufferSize::new(4).unwrap();
        #[allow(
            clippy::expect_used,
            reason = "mutex poison indicates a preceding panic"
        )]
        let mut belt = model.belt.lock().expect("belt mutex poisoned");
        belt.write_buffer(&mut se.encoder, &model.buffers.sample, 0, TOKEN_ID_SIZE)
        // guard drops here — BufferViewMut is owned, no borrow back to the belt.
    };
    encode_step_matvec(&mut se, &model.cfg, 0, Some((0, k)), pos, &mut NoMarker);
    let bytes = u64::from(k) * 8;
    se.copy_sample_to_readback(bytes);
    let slot = se.schedule_topk_map(bytes);
    (se.finish(), slot, view)
}

/// Run one matvec step at `pos`, reading the current token from CPU and
/// returning the top-`k` logits + indices for the next token.
pub fn step_matvec_topk(
    model: &Model,
    token_id: u32,
    pos: u32,
    k: u32,
) -> Result<(Vec<f32>, Vec<u32>)> {
    let (cb, slot) = build_step_matvec_topk_cb(model, token_id, pos, k);
    model.belt_finish();
    model.queue.submit(Some(cb));
    model.belt_recall();
    wait_topk_readback(model, k, slot)
}

/// Same as [`step_matvec_topk`] but does not perform any sampling readback.
/// Used by perf benches to avoid coupling forward-pass cost to readback I/O —
/// callers `device.poll(wait_indefinitely)` themselves to time the work.
#[cfg(feature = "bench-internals")]
pub fn step_matvec_no_sample(model: &Model, token_id: u32, pos: u32) {
    let mut se = StepEncoder::new(model);
    se.write_sample(0, bytemuck::bytes_of(&token_id));
    // We still encode the topk_reduce dispatch (with k=1, the single argmax case)
    // so the timing reflects real generation cost; we just skip the readback.
    encode_step_matvec(&mut se, &model.cfg, 0, Some((0, 1)), pos, &mut NoMarker);
    let cb = se.finish();
    model.belt_finish();
    model.queue.submit(Some(cb));
    model.belt_recall();
}

/// Pre-KV-copy block of one layer: `rms_norm` → QKV fused → q/k norms → rope.
fn layer_pre_kv_in_pass<M: StepMarker>(
    model: &Model,
    cfg: &Config,
    pass: &mut wgpu::ComputePass<'_>,
    il: u32,
    pos: u32,
    marker: &mut M,
) {
    let lt = &model.layer_tensors[il as usize];
    // Fused: rms_norm(x) * w_attn_norm → matvec_q1_0_fused (QKV).
    // Replaces a 2-dispatch sequence (rms_norm, matvec_q1_0_fused).
    // x is read directly (NOT x_norm); the kernel stages x to LDS, normalizes
    // in place, and runs the matvec inner loop off the normed shmem.
    dispatch_matvec_q1_0_fused_normed(
        model,
        cfg,
        pass,
        cfg.n_embd,
        model.act_layout.x,
        lt.attn_norm_off,
        WeightSet::Attn,
        &[
            (lt.wq.0, lt.wq.1, cfg.q_dim, model.act_layout.q),
            (lt.wk.0, lt.wk.1, cfg.kv_dim, model.act_layout.k_cur),
            (lt.wv.0, lt.wv.1, cfg.kv_dim, model.act_layout.v_cur),
        ],
    );
    marker.mark(pass, "qkv_fused_normed");

    // Q's rms_norm + *w_q_norm + NEOX-RoPE, written back into act.q in place.
    // K's rms_norm + RoPE + Q8_0 quantize + writeback into kv_k, plus V's
    // quantize + writeback into kv_v, all happen inside dispatch_kv_writeback_fused
    // (called from encode_step_matvec).
    dispatch_q_norm_rope_fused(
        model,
        cfg,
        pass,
        model.act_layout.q,
        lt.attn_q_norm_off,
        pos,
        1,
    );
    marker.mark(pass, "q_norm_rope");
}

/// Post-KV-copy block of one layer: attention → Wo (resid) → `ffn_norm`
/// → gate-up fused → `silu_mul` → Wd (resid).
fn layer_post_kv_in_pass<M: StepMarker>(
    model: &Model,
    cfg: &Config,
    pass: &mut wgpu::ComputePass<'_>,
    il: u32,
    pos: u32,
    marker: &mut M,
) {
    let lt = &model.layer_tensors[il as usize];

    // Split-K + GQA-batched flash-attention for tg (m_tokens=1).
    {
        let cur_pos = pos + 1;
        let n_chunks_active = cur_pos.div_ceil(ATTN_CHUNK_SIZE);
        let (d_word, qs_byte) = kv_layer_offsets(cfg, model.max_seq, il);

        let ps = AttnSplitParams {
            head_dim: cfg.head_dim,
            n_head: cfg.n_head,
            n_kv_head: cfg.n_kv_head,
            pos: cur_pos,
            kv_stride: cfg.kv_dim,
            q_offset: model.act_layout.q,
            k_d_word_offset: d_word,
            k_qs_byte_offset: qs_byte,
            v_d_word_offset: d_word,
            v_qs_byte_offset: qs_byte,
            n_chunks_active,
            scale: 1.0 / (cfg.head_dim as f32).sqrt(),
        };
        pass.set_pipeline(&model.pipes.attention_split);
        pass.set_bind_group(0, &model.cached.attn_split, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&ps));
        pass.dispatch_workgroups(cfg.n_kv_head, n_chunks_active, 1);
        marker.mark(pass, "attn_split");

        let pm = AttnMergeParams {
            head_dim: cfg.head_dim,
            n_head: cfg.n_head,
            out_offset: model.act_layout.attn_out,
            n_chunks_active,
        };
        pass.set_pipeline(&model.pipes.attention_merge);
        pass.set_bind_group(0, &model.cached.attn_merge, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&pm));
        pass.dispatch_workgroups(cfg.n_head, 1, 1);
        marker.mark(pass, "attn_merge");
    }

    dispatch_matvec_q1_0(
        model,
        pass,
        cfg.q_dim,
        cfg.n_embd,
        WeightSet::Attn,
        lt.wo.0,
        lt.wo.1,
        model.act_layout.attn_out,
        model.act_layout.x,
        true, /*accumulate*/
    );
    marker.mark(pass, "wo");

    // Fused: rms_norm(x) * w_ffn_norm → matvec_q1_0_fused (gate+up).
    // Replaces a 2-dispatch sequence (rms_norm, matvec_q1_0_fused).
    dispatch_matvec_q1_0_fused_normed(
        model,
        cfg,
        pass,
        cfg.n_embd,
        model.act_layout.x,
        lt.ffn_norm_off,
        WeightSet::FfnGU,
        &[
            (lt.wg.0, lt.wg.1, cfg.n_ff, model.act_layout.gate),
            (lt.wu.0, lt.wu.1, cfg.n_ff, model.act_layout.up),
        ],
    );
    marker.mark(pass, "gate_up_fused_normed");

    // Fused: silu(gate) * up on the input side of Wd, in one dispatch (no
    // ffn_in round-trip, no standalone silu_mul). The standalone silu_mul
    // shader and the matmul-prefill path's `silu_mul -> matmul_q1_0_q8_0`
    // pair are unchanged — this fusion is matvec-path-only.
    dispatch_matvec_q1_0_silu(
        model,
        pass,
        cfg.n_ff,
        cfg.n_embd,
        WeightSet::FfnD,
        lt.wd.0,
        lt.wd.1,
        model.act_layout.gate,
        model.act_layout.up,
        model.act_layout.x,
        true, /*accumulate*/
    );
    marker.mark(pass, "wd_silu");
}

/// Run the matvec single-token path over every token in `prompt`, advancing
/// `pos` from `pos_base` to `pos_base + prompt.len()`, and return the top-K
/// candidates from the LAST token's logits. Suitable for incremental prefill
/// after an existing KV cache (any `pos_base`).
pub fn prefill_matvec_loop_topk(
    model: &Model,
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
            max: model.m_max,
        });
    };
    // We pass `topk_out=None` for these steps so the suffix
    // (output_norm + LM head + topk_reduce) is skipped — those logits are
    // thrown away anyway, and skipping the topk avoids it stomping on the
    // prompt region of `sample`.
    for chunk_start in (0..rest.len()).step_by(CHUNK) {
        let chunk_end = (chunk_start + CHUNK).min(rest.len());
        let chunk = &rest[chunk_start..chunk_end];
        // Stage this chunk's tokens into sample[0..chunk.len()]; the embed
        // shader for step `i` reads `sample[i]`. The staged copy is encoded
        // before the first compute pass so wgpu's implicit barrier applies.
        let mut se = StepEncoder::new(model);
        se.write_sample(0, bytemuck::cast_slice(chunk));
        for (i, t) in (chunk_start..chunk_end).enumerate() {
            encode_step_matvec(
                &mut se,
                &model.cfg,
                /*sample_in=*/ i as u32,
                /*topk_out=*/ None,
                /*pos=*/ pos_base + t as u32,
                &mut NoMarker,
            );
        }
        let cb = se.finish();
        model.belt_finish();
        model.queue.submit(Some(cb));
        model.belt_recall();
    }
    step_matvec_topk(model, last, pos_base + rest.len() as u32, k)
}

// ---------- matmul (batched prefill) ---------------------------------------

/// Batched matmul prefill of `prompt` starting from KV-cache position 0.
/// Advances pos from 0 to `prompt.len()`. Returns top-K candidates from the
/// last token's logits. Requires `pos_base == 0` (the matmul attention shader
/// assumes a fresh cache).
pub fn prefill_matmul_topk<M: StepMarker>(
    model: &Model,
    prompt: &[u32],
    pos_base: u32,
    k: u32,
    marker: &mut M,
) -> Result<(Vec<f32>, Vec<u32>)> {
    if pos_base != 0 {
        // Caller must use prefill_matvec_loop_topk for incremental prefill.
        return Err(PotError::Config(
            "prefill_matmul_topk requires pos_base == 0; use prefill_one_at_a_time for incremental prefill",
        ));
    }
    let m = prompt.len() as u32;
    if m == 0 || m > model.m_max {
        return Err(PotError::PrefillTooLarge {
            n: m,
            max: model.m_max,
        });
    }
    let cfg = &model.cfg;
    let ot = &model.output_tensors;
    let k = k.clamp(1, TOPK_MAX);

    // ---- All phases (embed → per-layer transformer → final norm/LM-head/topk
    //      → readback copy) into ONE command buffer / ONE submit, with all
    //      compute dispatches sharing ONE pass to amortize the
    //      begin_compute_pass cost (~25us each on RADV).
    let mut se = StepEncoder::new(model);
    // Stage the prompt token ids into sample[0..m] before the embed pass.
    se.write_sample(0, bytemuck::cast_slice(prompt));

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
            output_offset: model.act_layout.x,
            sample_offset: 0,
        };
        pass.set_pipeline(&model.pipes.embed);
        pass.set_bind_group(0, &model.cached.embed, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(m, 1, 1);

        // Phase 2: per-layer transformer.
        for il in 0..cfg.n_layer {
            layer_step_matmul_in_pass(model, cfg, &mut pass, il, m);
        }

        // Phase 3: output_norm (last token, in-place) + LM head + topk_reduce.
        let last_x = model.act_layout.x + (m - 1) * cfg.n_embd;
        dispatch_rms_norm(
            model,
            cfg,
            &mut pass,
            1,
            cfg.n_embd,
            last_x,
            last_x,
            ot.output_norm_off,
        );
        dispatch_matvec_q1_0(
            model,
            &mut pass,
            cfg.n_embd,
            cfg.n_vocab,
            WeightSet::Embed,
            ot.lm_head_d,
            ot.lm_head_qs,
            last_x,
            model.act_layout.logits,
            false,
        );
        dispatch_topk_reduce(model, &mut pass, cfg.n_vocab, k, model.act_layout.logits, 0);
        marker.mark(&mut pass, "prefill_pass_end");
    }

    // Phase 4: append readback copy + schedule map, all in the same command buffer.
    let bytes = u64::from(k) * 8;
    se.copy_sample_to_readback(bytes);
    let slot = se.schedule_topk_map(bytes);

    let cb = se.finish();
    model.belt_finish();
    model.queue.submit(Some(cb));
    model.belt_recall();

    wait_topk_readback(model, k, slot)
}

fn layer_step_matmul_in_pass(
    model: &Model,
    cfg: &Config,
    pass: &mut wgpu::ComputePass<'_>,
    il: u32,
    m: u32,
) {
    let pos_base = 0u32;
    let lt = &model.layer_tensors[il as usize];

    // attn_norm + quantize + Q/K/V matmul.
    dispatch_rms_norm(
        model,
        cfg,
        pass,
        m,
        cfg.n_embd,
        model.act_layout.x,
        model.act_layout.x_norm,
        lt.attn_norm_off,
    );
    let (a_d, a_qs) = dispatch_quantize_act(model, pass, cfg.n_embd, m, model.act_layout.x_norm);
    dispatch_matmul_q1_0(
        model,
        pass,
        cfg.n_embd,
        cfg.q_dim,
        m,
        WeightSet::Attn,
        lt.wq.0,
        lt.wq.1,
        a_d,
        a_qs,
        model.act_layout.q,
        false,
    );
    dispatch_matmul_q1_0(
        model,
        pass,
        cfg.n_embd,
        cfg.kv_dim,
        m,
        WeightSet::Attn,
        lt.wk.0,
        lt.wk.1,
        a_d,
        a_qs,
        model.act_layout.k_cur,
        false,
    );
    dispatch_matmul_q1_0(
        model,
        pass,
        cfg.n_embd,
        cfg.kv_dim,
        m,
        WeightSet::Attn,
        lt.wv.0,
        lt.wv.1,
        a_d,
        a_qs,
        model.act_layout.v_cur,
        false,
    );

    // Q/K rms+rope (in-place) + KV writeback into kv_{k,v}.
    dispatch_q_norm_rope_fused(
        model,
        cfg,
        pass,
        model.act_layout.q,
        lt.attn_q_norm_off,
        pos_base,
        m,
    );
    dispatch_kv_writeback_fused(
        model,
        cfg,
        pass,
        model.act_layout.k_cur,
        model.act_layout.v_cur,
        lt.attn_k_norm_off,
        il,
        pos_base,
        m,
    );

    // Attention.
    dispatch_attention(
        model,
        cfg,
        pass,
        il,
        model.max_seq,
        /*pos=*/ 0,
        m,
        /*is_prefill=*/ true,
    );

    // Wo (residual) + ffn_norm + gate/up + silu_mul + Wd (residual).
    let (a_d2, a_qs2) = dispatch_quantize_act(model, pass, cfg.q_dim, m, model.act_layout.attn_out);
    dispatch_matmul_q1_0(
        model,
        pass,
        cfg.q_dim,
        cfg.n_embd,
        m,
        WeightSet::Attn,
        lt.wo.0,
        lt.wo.1,
        a_d2,
        a_qs2,
        model.act_layout.x,
        true,
    );
    dispatch_rms_norm(
        model,
        cfg,
        pass,
        m,
        cfg.n_embd,
        model.act_layout.x,
        model.act_layout.x_norm,
        lt.ffn_norm_off,
    );
    let (a_d3, a_qs3) = dispatch_quantize_act(model, pass, cfg.n_embd, m, model.act_layout.x_norm);
    dispatch_matmul_q1_0(
        model,
        pass,
        cfg.n_embd,
        cfg.n_ff,
        m,
        WeightSet::FfnGU,
        lt.wg.0,
        lt.wg.1,
        a_d3,
        a_qs3,
        model.act_layout.gate,
        false,
    );
    dispatch_matmul_q1_0(
        model,
        pass,
        cfg.n_embd,
        cfg.n_ff,
        m,
        WeightSet::FfnGU,
        lt.wu.0,
        lt.wu.1,
        a_d3,
        a_qs3,
        model.act_layout.up,
        false,
    );
    dispatch_silu_mul(
        model,
        pass,
        cfg.n_ff,
        m,
        model.act_layout.gate,
        model.act_layout.up,
        model.act_layout.ffn_in,
    );
    let (a_d4, a_qs4) = dispatch_quantize_act(model, pass, cfg.n_ff, m, model.act_layout.ffn_in);
    dispatch_matmul_q1_0(
        model,
        pass,
        cfg.n_ff,
        cfg.n_embd,
        m,
        WeightSet::FfnD,
        lt.wd.0,
        lt.wd.1,
        a_d4,
        a_qs4,
        model.act_layout.x,
        true,
    );
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
/// Each `MicroMarker` reuses `model.buffers.bench_query_set` starting from slot 0,
/// so only one `MicroMarker` may be live at a time (the previous one must be
/// resolved before a new one is marked into the same query set).
#[cfg(feature = "bench-internals")]
pub struct MicroMarker<'m> {
    model: &'m Model,
    next_idx: u32,
    labels: Vec<&'static str>,
}

#[cfg(feature = "bench-internals")]
impl<'m> MicroMarker<'m> {
    pub const fn new(model: &'m Model) -> Self {
        Self {
            model,
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
        let ticks = bench_resolve_ticks(self.model, n)?;
        let period = self.model.bench_ts_period_ns;
        let mut spans: Vec<(&'static str, f32)> = Vec::with_capacity((n - 1) as usize);
        for i in 1..n as usize {
            let dt_ns = ticks[i].saturating_sub(ticks[i - 1]) as f32 * period;
            spans.push((self.labels[i], dt_ns));
        }
        Ok(spans)
    }
}

#[cfg(feature = "bench-internals")]
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
        pass.write_timestamp(&self.model.buffers.bench_query_set, self.next_idx);
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
/// Reuses `model.buffers.bench_query_set` slots 0/1, so only one `BenchMarker`
/// may be live at a time and the pass it instruments must be the only one to
/// write those slots between `new` and `resolve`.
///
/// [`mark`]: BenchMarker::mark
#[cfg(feature = "bench-internals")]
pub struct BenchMarker<'m> {
    model: &'m Model,
}

#[cfg(feature = "bench-internals")]
impl<'m> BenchMarker<'m> {
    pub const fn new(model: &'m Model) -> Self {
        Self { model }
    }

    /// After the instrumented pass has been submitted and polled (GPU done),
    /// resolve the begin/end timestamps and return total GPU duration in
    /// nanoseconds.
    pub fn resolve(self) -> Result<f32> {
        let ticks = bench_resolve_ticks(self.model, 2)?;
        Ok(ticks[1].saturating_sub(ticks[0]) as f32 * self.model.bench_ts_period_ns)
    }
}

#[cfg(feature = "bench-internals")]
impl StepMarker for BenchMarker<'_> {
    fn setup_desc<'a>(&'a self, desc: &mut wgpu::ComputePassDescriptor<'a>) {
        desc.timestamp_writes = Some(wgpu::ComputePassTimestampWrites {
            query_set: &self.model.buffers.bench_query_set,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        });
    }
    #[inline(always)]
    fn mark(&mut self, _pass: &mut wgpu::ComputePass<'_>, _label: &'static str) {}
}

/// Resolve `n` timestamps from `bench_query_set[0..n]` to host memory. Shared
/// by [`MicroMarker::resolve`] and [`BenchMarker::resolve`].
#[cfg(feature = "bench-internals")]
fn bench_resolve_ticks(model: &Model, n: u32) -> Result<Vec<u64>> {
    let bytes = u64::from(n) * 8;
    let mut enc = model
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bench_resolve"),
        });
    enc.resolve_query_set(
        &model.buffers.bench_query_set,
        0..n,
        &model.buffers.bench_resolve,
        0,
    );
    enc.copy_buffer_to_buffer(
        &model.buffers.bench_resolve,
        0,
        &model.buffers.bench_readback,
        0,
        bytes,
    );
    let slot: MapSlot = Arc::new(OnceLock::new());
    let slot2 = slot.clone();
    enc.map_buffer_on_submit(
        &model.buffers.bench_readback,
        wgpu::MapMode::Read,
        0..bytes,
        move |res| {
            let _ = slot2.set(res);
        },
    );
    model.queue.submit(Some(enc.finish()));

    let slice = model.buffers.bench_readback.slice(0..bytes);
    if let Err(e) = model.device.poll(PollType::wait_indefinitely()) {
        model.check_device()?;
        return Err(PotError::Poll(e));
    }
    match slot.get() {
        Some(Ok(())) => {}
        Some(Err(e)) => {
            model.check_device()?;
            return Err(PotError::BufferMap(e.clone()));
        }
        None => unreachable!("bench map_async callback did not fire before poll returned"),
    }

    let data = slice.get_mapped_range();
    let ticks: Vec<u64> = bytemuck::cast_slice::<_, u64>(&data[..bytes as usize]).to_vec();
    drop(data);
    model.buffers.bench_readback.unmap();
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
