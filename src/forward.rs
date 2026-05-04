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

use bytemuck::Pod;

use crate::error::{PotError, Result};
use crate::model::{
    ATTN_CHUNK_SIZE, AttnMergeParams, AttnParams, AttnSplitParams, Config, EmbedParams,
    KvWritebackFusedParams, MatmulParams, MatvecFusedNormedParams, MatvecFusedParams, MatvecParams,
    MatvecSiluParams, Model, QNormRopeFusedParams, QuantParams, RmsNormParams, SiluMulParams,
    TOPK_MAX, TopKParams, UNIFORM_POOL_SLOTS, UNIFORM_SLOT_SIZE, WeightSet,
};

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

// ---------- per-step encoder + uniform pool ---------------------------------

pub struct UniformPool {
    cpu: Vec<u8>,
    next_slot: u64,
}

impl UniformPool {
    fn new() -> Self {
        // Pre-size the CPU staging Vec to the full pool budget so per-step
        // encoding doesn't reallocate. Tg uses ~120 KiB/step (508 slots),
        // matmul prefill ~165 KiB (652 slots) — both blow past the old 64
        // KiB initial capacity and triggered grow-and-copy on every step.
        let cap = (UNIFORM_POOL_SLOTS * UNIFORM_SLOT_SIZE) as usize;
        Self {
            cpu: Vec::with_capacity(cap),
            next_slot: 0,
        }
    }
    const fn remaining_slots(&self) -> u64 {
        UNIFORM_POOL_SLOTS.saturating_sub(self.next_slot)
    }
    fn alloc<T: Pod>(&mut self, params: &T) -> u32 {
        let slot = self.next_slot;
        assert!(
            slot < UNIFORM_POOL_SLOTS,
            "UniformPool exhausted: {slot} >= {UNIFORM_POOL_SLOTS}; raise UNIFORM_POOL_SLOTS",
        );
        self.next_slot += 1;
        let off = (slot * UNIFORM_SLOT_SIZE) as usize;
        if self.cpu.len() < off + UNIFORM_SLOT_SIZE as usize {
            self.cpu.resize(off + UNIFORM_SLOT_SIZE as usize, 0);
        }
        let bytes = bytemuck::bytes_of(params);
        self.cpu[off..off + bytes.len()].copy_from_slice(bytes);
        (slot * UNIFORM_SLOT_SIZE) as u32
    }
}

pub struct StepEncoder<'a> {
    model: &'a Model,
    pub(crate) encoder: wgpu::CommandEncoder,
    pub(crate) uniforms: UniformPool,
}

impl<'a> StepEncoder<'a> {
    pub fn new(model: &'a Model) -> Self {
        let encoder = model
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("step"),
            });
        Self {
            model,
            encoder,
            uniforms: UniformPool::new(),
        }
    }

    pub fn alloc_uniform<T: Pod>(&mut self, params: &T) -> u32 {
        self.uniforms.alloc(params)
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

    pub fn finish(self) -> wgpu::CommandBuffer {
        if !self.uniforms.cpu.is_empty() {
            self.model
                .queue
                .write_buffer(&self.model.buffers.uniform, 0, &self.uniforms.cpu);
        }
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
    uniforms: &mut UniformPool,
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
    let off = uniforms.alloc(&p);
    pass.set_pipeline(&model.pipes.rms_norm);
    pass.set_bind_group(0, &model.cached.rms_norm, &[off]);
    pass.dispatch_workgroups(n_groups, 1, 1);
}

fn dispatch_matvec_q1_0(
    model: &Model,
    uniforms: &mut UniformPool,
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
    let off = uniforms.alloc(&p);
    pass.set_pipeline(&model.pipes.matvec);
    pass.set_bind_group(0, matvec_bg(model, weights), &[off]);
    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
}

// Currently unused: both call sites (QKV, gate+up) on the matvec single-token
// path now go through `dispatch_matvec_q1_0_fused_normed` which folds in the
// preceding `rms_norm`. Kept as the canonical 2-/3-range fused matvec for any
// future caller that doesn't have an `rms_norm` immediately before it.
#[allow(dead_code, reason = "kept for future no-rms-norm callers")]
fn dispatch_matvec_q1_0_fused(
    model: &Model,
    uniforms: &mut UniformPool,
    pass: &mut wgpu::ComputePass<'_>,
    k: u32,
    input_offset: u32,
    weights: WeightSet,
    ranges: &[(u32, u32, u32, u32)],
) {
    const ROWS_PER_WG: u32 = 8;
    debug_assert!(ranges.len() == 2 || ranges.len() == 3);
    for (_, _, n, _) in ranges {
        debug_assert!(n % 8 == 0);
    }
    let r = |i: usize| ranges.get(i).copied().unwrap_or((0, 0, 0, 0));
    let (d0, qs0, n0, o0) = r(0);
    let (d1, qs1, n1, o1) = r(1);
    let (d2, qs2, n2, o2) = r(2);
    let n_total = n0 + n1 + n2;
    let n_wg = n_total.div_ceil(ROWS_PER_WG);
    let dispatch_x = n_wg.min(65535);
    let dispatch_y = n_wg.div_ceil(dispatch_x);
    let p = MatvecFusedParams {
        k,
        n_total,
        input_offset,
        dispatch_x_dim: dispatch_x,
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
    let off = uniforms.alloc(&p);
    pass.set_pipeline(&model.pipes.matvec_fused);
    pass.set_bind_group(0, matvec_bg(model, weights), &[off]);
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
    uniforms: &mut UniformPool,
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
    let off = uniforms.alloc(&p);
    pass.set_pipeline(&model.pipes.matvec_silu);
    pass.set_bind_group(0, matvec_bg(model, weights), &[off]);
    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
}

/// Fused: `rms_norm(x) * w_norm` → multi-range `Q1_0` matvec, in one dispatch.
/// Replaces `dispatch_rms_norm + dispatch_matvec_q1_0_fused` for the matvec
/// single-token path. See `shaders/matvec_q1_0_fused_normed.wgsl`.
fn dispatch_matvec_q1_0_fused_normed(
    model: &Model,
    cfg: &Config,
    uniforms: &mut UniformPool,
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
    let off = uniforms.alloc(&p);
    pass.set_pipeline(&model.pipes.matvec_fused_normed);
    pass.set_bind_group(0, matvec_fused_normed_bg(model, weights), &[off]);
    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
}

fn dispatch_topk_reduce(
    model: &Model,
    uniforms: &mut UniformPool,
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
    let off = uniforms.alloc(&p);
    pass.set_pipeline(&model.pipes.topk_reduce);
    pass.set_bind_group(0, &model.cached.topk_reduce, &[off]);
    pass.dispatch_workgroups(1, 1, 1);
}

/// Fused: `rms_norm(K` head) → \*`w_k_norm` → NEOX-RoPE → `Q8_0` quantize → write
/// `kv_k`. V runs in the same workgroup (just quantize + write `kv_v`). Replaces
/// `rms_norm(K) + rope(K) + kv_writeback` with one dispatch.
fn dispatch_kv_writeback_fused(
    model: &Model,
    cfg: &Config,
    uniforms: &mut UniformPool,
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
    let off = uniforms.alloc(&p);
    pass.set_pipeline(&model.pipes.kv_writeback_fused);
    pass.set_bind_group(0, &model.cached.kv_writeback_fused, &[off]);
    pass.dispatch_workgroups(cfg.n_kv_head, m_tokens, 1);
}

/// Fused: `rms_norm(Q` head) → \*`w_q_norm` → NEOX-RoPE, written back into
/// `act.q` in place. Replaces `rms_norm(Q) + rope(Q)` with one dispatch.
fn dispatch_q_norm_rope_fused(
    model: &Model,
    cfg: &Config,
    uniforms: &mut UniformPool,
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
    let off = uniforms.alloc(&p);
    pass.set_pipeline(&model.pipes.q_norm_rope_fused);
    pass.set_bind_group(0, &model.cached.q_norm_rope_fused, &[off]);
    pass.dispatch_workgroups(cfg.n_head, m_tokens, 1);
}

fn dispatch_quantize_act(
    model: &Model,
    uniforms: &mut UniformPool,
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
    let off = uniforms.alloc(&p);
    pass.set_pipeline(&model.pipes.quantize);
    pass.set_bind_group(0, &model.cached.quantize, &[off]);
    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    (d_off, qs_off)
}

fn dispatch_matmul_q1_0(
    model: &Model,
    uniforms: &mut UniformPool,
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
    let off = uniforms.alloc(&p);
    pass.set_pipeline(&model.pipes.matmul);
    pass.set_bind_group(0, matmul_bg(model, weights), &[off]);
    pass.dispatch_workgroups(n.div_ceil(64), m.div_ceil(64), 1);
}

fn dispatch_silu_mul(
    model: &Model,
    uniforms: &mut UniformPool,
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
    let off = uniforms.alloc(&p);
    pass.set_pipeline(&model.pipes.silu_mul);
    pass.set_bind_group(0, &model.cached.silu_mul, &[off]);
    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
}

fn dispatch_attention(
    model: &Model,
    cfg: &Config,
    uniforms: &mut UniformPool,
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
    let off = uniforms.alloc(&p);
    pass.set_pipeline(&model.pipes.attention);
    pass.set_bind_group(0, &model.cached.attn, &[off]);
    pass.dispatch_workgroups(cfg.n_head, m_tokens, 1);
}

// ---------- async readback helper -------------------------------------------

/// Wait the `sample → readback` copy that was already encoded into the
/// step's command buffer (via [`StepEncoder::copy_sample_to_readback`]) and
/// return the K f32 logits + K u32 indices the caller asked for.
///
/// This must be called AFTER the step's command buffer has been submitted —
/// i.e. the readback copy is in flight. There is no separate submit here.
fn wait_topk_readback(model: &Model, k: u32) -> Result<(Vec<f32>, Vec<u32>)> {
    use core::result::Result as StdResult;
    use std::sync::{Arc, OnceLock};

    use wgpu::{BufferAsyncError, PollType};
    type MapResult = StdResult<(), BufferAsyncError>;
    let bytes = u64::from(k) * 8; // K f32 + K u32
    let slice = model.buffers.readback.slice(0..bytes);
    let slot: Arc<OnceLock<MapResult>> = Arc::new(OnceLock::new());
    let slot2 = slot.clone();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = slot2.set(res);
    });
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

/// Number of uniform slots `encode_step_matvec` allocates for one step when
/// the LM-head/topk suffix is omitted (the `topk_out = None` path used by
/// `prefill_matvec_loop_topk`'s non-last steps).
fn encode_step_matvec_slots_no_suffix(cfg: &Config) -> u64 {
    // 1 embed + 8 per layer (qkv_fused_normed, q_norm_rope_fused,
    // kv_writeback_fused, attn_split, attn_merge, Wo,
    // gate_up_fused_normed, Wd_silu).
    1 + 8 * u64::from(cfg.n_layer)
}

/// Encode one tg step into the given encoder. The input token is read from
/// `sample[sample_in]`. If `topk_out = Some((base, k))`, the suffix
/// (`output_norm` + LM head + `topk_reduce`) is appended and the top-K logits +
/// indices land at `sample[base..base + 2*k]`. If `topk_out = None`, the
/// suffix is skipped — useful for KV-fill-only steps (e.g. mid-prefill) where
/// the sampled token isn't read.
pub fn encode_step_matvec(
    se: &mut StepEncoder,
    cfg: &Config,
    sample_in: u32,
    topk_out: Option<(u32, u32)>,
    pos: u32,
) {
    let StepEncoder {
        model: m,
        encoder,
        uniforms,
    } = se;
    let ot = &m.output_tensors;
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("step_matvec"),
        timestamp_writes: None,
    });
    // embed
    {
        let p = EmbedParams {
            k: cfg.n_embd,
            d_offset: ot.token_embd_d,
            qs_offset: ot.token_embd_qs,
            output_offset: m.act_layout.x,
            sample_offset: sample_in,
        };
        let off = uniforms.alloc(&p);
        pass.set_pipeline(&m.pipes.embed);
        pass.set_bind_group(0, &m.cached.embed, &[off]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    for il in 0..cfg.n_layer {
        layer_pre_kv_in_pass(m, cfg, uniforms, &mut pass, il, pos);
        // Fused: K (rms_norm + *w_k_norm + RoPE + Q8_0 quantize) and V (Q8_0
        // quantize) → write both into kv_{k,v}. Replaces the previous
        // rms_norm(K) + rope(K) + kv_writeback trio (3 dispatches → 1).
        let lt = &m.layer_tensors[il as usize];
        dispatch_kv_writeback_fused(
            m,
            cfg,
            uniforms,
            &mut pass,
            m.act_layout.k_cur,
            m.act_layout.v_cur,
            lt.attn_k_norm_off,
            il,
            pos,
            1,
        );
        layer_post_kv_in_pass(m, cfg, uniforms, &mut pass, il, pos);
    }
    if let Some((topk_out_u32_base, k)) = topk_out {
        // output suffix: rms_norm in-place on x, then LM head reads
        // directly from x (saves one f16 vector round-trip vs. x_norm staging).
        dispatch_rms_norm(
            m,
            cfg,
            uniforms,
            &mut pass,
            1,
            cfg.n_embd,
            m.act_layout.x,
            m.act_layout.x,
            ot.output_norm_off,
        );
        dispatch_matvec_q1_0(
            m,
            uniforms,
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
        dispatch_topk_reduce(
            m,
            uniforms,
            &mut pass,
            cfg.n_vocab,
            k,
            m.act_layout.logits,
            topk_out_u32_base,
        );
    }
    drop(pass);
}

/// Run one matvec step at `pos`, reading the current token from CPU and
/// returning the top-`k` logits + indices for the next token.
pub fn step_matvec_topk(
    model: &Model,
    token_id: u32,
    pos: u32,
    k: u32,
) -> Result<(Vec<f32>, Vec<u32>)> {
    let k = k.clamp(1, TOPK_MAX);
    model
        .queue
        .write_buffer(&model.buffers.sample, 0, bytemuck::bytes_of(&token_id));
    let mut se = StepEncoder::new(model);
    encode_step_matvec(&mut se, &model.cfg, 0, Some((0, k)), pos);
    // Fold the readback copy into the same command buffer so the step + the
    // sample→readback transfer are one submit, not two.
    se.copy_sample_to_readback(u64::from(k) * 8);
    let cb = se.finish();
    model.queue.submit(Some(cb));
    wait_topk_readback(model, k)
}

/// Same as [`step_matvec_topk`] but does not perform any sampling readback.
/// Used by perf benches to avoid coupling forward-pass cost to readback I/O —
/// callers `device.poll(wait_indefinitely)` themselves to time the work.
#[cfg(feature = "bench-internals")]
pub fn step_matvec_no_sample(model: &Model, token_id: u32, pos: u32) {
    model
        .queue
        .write_buffer(&model.buffers.sample, 0, bytemuck::bytes_of(&token_id));
    let mut se = StepEncoder::new(model);
    // We still encode the topk_reduce dispatch (with k=1, the single argmax case)
    // so the timing reflects real generation cost; we just skip the readback.
    encode_step_matvec(&mut se, &model.cfg, 0, Some((0, 1)), pos);
    let cb = se.finish();
    model.queue.submit(Some(cb));
}

/// Pre-KV-copy block of one layer: `rms_norm` → QKV fused → q/k norms → rope.
fn layer_pre_kv_in_pass(
    model: &Model,
    cfg: &Config,
    uniforms: &mut UniformPool,
    pass: &mut wgpu::ComputePass<'_>,
    il: u32,
    pos: u32,
) {
    let lt = &model.layer_tensors[il as usize];
    // Fused: rms_norm(x) * w_attn_norm → matvec_q1_0_fused (QKV).
    // Replaces a 2-dispatch sequence (rms_norm, matvec_q1_0_fused).
    // x is read directly (NOT x_norm); the kernel stages x to LDS, normalizes
    // in place, and runs the matvec inner loop off the normed shmem.
    dispatch_matvec_q1_0_fused_normed(
        model,
        cfg,
        uniforms,
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

    // Q's rms_norm + *w_q_norm + NEOX-RoPE, written back into act.q in place.
    // K's rms_norm + RoPE + Q8_0 quantize + writeback into kv_k, plus V's
    // quantize + writeback into kv_v, all happen inside dispatch_kv_writeback_fused
    // (called from encode_step_matvec).
    dispatch_q_norm_rope_fused(
        model,
        cfg,
        uniforms,
        pass,
        model.act_layout.q,
        lt.attn_q_norm_off,
        pos,
        1,
    );
}

/// Post-KV-copy block of one layer: attention → Wo (resid) → `ffn_norm`
/// → gate-up fused → `silu_mul` → Wd (resid).
fn layer_post_kv_in_pass(
    model: &Model,
    cfg: &Config,
    uniforms: &mut UniformPool,
    pass: &mut wgpu::ComputePass<'_>,
    il: u32,
    pos: u32,
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
        let off = uniforms.alloc(&ps);
        pass.set_pipeline(&model.pipes.attention_split);
        pass.set_bind_group(0, &model.cached.attn_split, &[off]);
        pass.dispatch_workgroups(cfg.n_kv_head, n_chunks_active, 1);

        let pm = AttnMergeParams {
            head_dim: cfg.head_dim,
            n_head: cfg.n_head,
            out_offset: model.act_layout.attn_out,
            n_chunks_active,
        };
        let off = uniforms.alloc(&pm);
        pass.set_pipeline(&model.pipes.attention_merge);
        pass.set_bind_group(0, &model.cached.attn_merge, &[off]);
        pass.dispatch_workgroups(cfg.n_head, 1, 1);
    }

    dispatch_matvec_q1_0(
        model,
        uniforms,
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

    // Fused: rms_norm(x) * w_ffn_norm → matvec_q1_0_fused (gate+up).
    // Replaces a 2-dispatch sequence (rms_norm, matvec_q1_0_fused).
    dispatch_matvec_q1_0_fused_normed(
        model,
        cfg,
        uniforms,
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

    // Fused: silu(gate) * up on the input side of Wd, in one dispatch (no
    // ffn_in round-trip, no standalone silu_mul). The standalone silu_mul
    // shader and the matmul-prefill path's `silu_mul -> matmul_q1_0_q8_0`
    // pair are unchanged — this fusion is matvec-path-only.
    dispatch_matvec_q1_0_silu(
        model,
        uniforms,
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
    let Some((&last, rest)) = prompt.split_last() else {
        return Err(PotError::PrefillTooLarge {
            n: 0,
            max: model.m_max,
        });
    };
    // For the (n - 1) non-last tokens we just need to fill the KV cache; we
    // can encode them all back-to-back into a SINGLE command buffer + submit,
    // then run the final token through `step_matvec_topk` (which does the
    // CPU readback for sampling).
    if !rest.is_empty() {
        // One up-front write covers every non-last token's input slot. The
        // embed shader for step `i` reads `sample[i]`. We pass `topk_out=None`
        // for these steps so the suffix (output_norm + LM head + topk_reduce)
        // is skipped — those logits are thrown away anyway, and skipping the
        // topk avoids it stomping on the prompt region of `sample` for prompts
        // longer than the previous TOPK_SCRATCH_BASE (768) workaround allowed.
        model
            .queue
            .write_buffer(&model.buffers.sample, 0, bytemuck::cast_slice(rest));
        // Each non-last step allocates a fixed number of uniform slots
        // (one for embed + 8 per transformer layer; no suffix). The 1 MiB
        // pool fits ~14 steps at n_layer=36, so for any prefill longer than
        // that we split the work into multiple submits — KV-cache writes from
        // a previous submit are visible to the next one's attention reads.
        let slots_per_step = encode_step_matvec_slots_no_suffix(&model.cfg);
        let mut se = StepEncoder::new(model);
        for t in 0..rest.len() {
            if se.uniforms.remaining_slots() < slots_per_step {
                let cb = se.finish();
                model.queue.submit(Some(cb));
                se = StepEncoder::new(model);
            }
            encode_step_matvec(
                &mut se,
                &model.cfg,
                /*sample_in=*/ t as u32,
                /*topk_out=*/ None,
                /*pos=*/ pos_base + t as u32,
            );
        }
        let cb = se.finish();
        model.queue.submit(Some(cb));
    }
    step_matvec_topk(model, last, pos_base + rest.len() as u32, k)
}

// ---------- matmul (batched prefill) ---------------------------------------

/// Batched matmul prefill of `prompt` starting from KV-cache position 0.
/// Advances pos from 0 to `prompt.len()`. Returns top-K candidates from the
/// last token's logits. Requires `pos_base == 0` (the matmul attention shader
/// assumes a fresh cache).
pub fn prefill_matmul_topk(
    model: &Model,
    prompt: &[u32],
    pos_base: u32,
    k: u32,
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
    model
        .queue
        .write_buffer(&model.buffers.sample, 0, bytemuck::cast_slice(prompt));
    let mut se = StepEncoder::new(model);
    let StepEncoder {
        encoder, uniforms, ..
    } = &mut se;

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("prefill_matmul"),
            timestamp_writes: None,
        });

        // Phase 1: embed all M tokens (one dispatch).
        let p = EmbedParams {
            k: cfg.n_embd,
            d_offset: ot.token_embd_d,
            qs_offset: ot.token_embd_qs,
            output_offset: model.act_layout.x,
            sample_offset: 0,
        };
        let off = uniforms.alloc(&p);
        pass.set_pipeline(&model.pipes.embed);
        pass.set_bind_group(0, &model.cached.embed, &[off]);
        pass.dispatch_workgroups(m, 1, 1);

        // Phase 2: per-layer transformer.
        for il in 0..cfg.n_layer {
            layer_step_matmul_in_pass(model, cfg, uniforms, &mut pass, il, m);
        }

        // Phase 3: output_norm (last token, in-place) + LM head + topk_reduce.
        let last_x = model.act_layout.x + (m - 1) * cfg.n_embd;
        dispatch_rms_norm(
            model,
            cfg,
            uniforms,
            &mut pass,
            1,
            cfg.n_embd,
            last_x,
            last_x,
            ot.output_norm_off,
        );
        dispatch_matvec_q1_0(
            model,
            uniforms,
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
        dispatch_topk_reduce(
            model,
            uniforms,
            &mut pass,
            cfg.n_vocab,
            k,
            model.act_layout.logits,
            0,
        );
    }

    // Phase 4: append readback copy so it ships with this same command buffer.
    se.copy_sample_to_readback(u64::from(k) * 8);

    let cb = se.finish();
    model.queue.submit(Some(cb));

    wait_topk_readback(model, k)
}

fn layer_step_matmul_in_pass(
    model: &Model,
    cfg: &Config,
    uniforms: &mut UniformPool,
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
        uniforms,
        pass,
        m,
        cfg.n_embd,
        model.act_layout.x,
        model.act_layout.x_norm,
        lt.attn_norm_off,
    );
    let (a_d, a_qs) = dispatch_quantize_act(
        model,
        uniforms,
        pass,
        cfg.n_embd,
        m,
        model.act_layout.x_norm,
    );
    dispatch_matmul_q1_0(
        model,
        uniforms,
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
        uniforms,
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
        uniforms,
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
        uniforms,
        pass,
        model.act_layout.q,
        lt.attn_q_norm_off,
        pos_base,
        m,
    );
    dispatch_kv_writeback_fused(
        model,
        cfg,
        uniforms,
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
        uniforms,
        pass,
        il,
        model.max_seq,
        /*pos=*/ 0,
        m,
        /*is_prefill=*/ true,
    );

    // Wo (residual) + ffn_norm + gate/up + silu_mul + Wd (residual).
    let (a_d2, a_qs2) = dispatch_quantize_act(
        model,
        uniforms,
        pass,
        cfg.q_dim,
        m,
        model.act_layout.attn_out,
    );
    dispatch_matmul_q1_0(
        model,
        uniforms,
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
        uniforms,
        pass,
        m,
        cfg.n_embd,
        model.act_layout.x,
        model.act_layout.x_norm,
        lt.ffn_norm_off,
    );
    let (a_d3, a_qs3) = dispatch_quantize_act(
        model,
        uniforms,
        pass,
        cfg.n_embd,
        m,
        model.act_layout.x_norm,
    );
    dispatch_matmul_q1_0(
        model,
        uniforms,
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
        uniforms,
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
        uniforms,
        pass,
        cfg.n_ff,
        m,
        model.act_layout.gate,
        model.act_layout.up,
        model.act_layout.ffn_in,
    );
    let (a_d4, a_qs4) =
        dispatch_quantize_act(model, uniforms, pass, cfg.n_ff, m, model.act_layout.ffn_in);
    dispatch_matmul_q1_0(
        model,
        uniforms,
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

// ---------- shader wrappers --------------------------------------------------

fn rms_norm(
    model: &Model,
    cfg: &Config,
    se: &mut StepEncoder,
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
    let off = se.alloc_uniform(&p);
    let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("rms_norm"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&model.pipes.rms_norm);
    cp.set_bind_group(0, &model.cached.rms_norm, &[off]);
    cp.dispatch_workgroups(n_groups, 1, 1);
}

fn matvec_q1_0(
    model: &Model,
    se: &mut StepEncoder,
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
    let off = se.alloc_uniform(&p);
    let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("matvec"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&model.pipes.matvec);
    cp.set_bind_group(0, matvec_bg(model, weights), &[off]);
    cp.dispatch_workgroups(dispatch_x, dispatch_y, 1);
}

fn kv_writeback_fused(
    model: &Model,
    cfg: &Config,
    se: &mut StepEncoder,
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
    let off = se.alloc_uniform(&p);
    let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("kv_writeback_fused"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&model.pipes.kv_writeback_fused);
    cp.set_bind_group(0, &model.cached.kv_writeback_fused, &[off]);
    cp.dispatch_workgroups(cfg.n_kv_head, m_tokens, 1);
}

fn q_norm_rope_fused(
    model: &Model,
    cfg: &Config,
    se: &mut StepEncoder,
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
    let off = se.alloc_uniform(&p);
    let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("q_norm_rope_fused"),
        timestamp_writes: None,
    });
    cp.set_pipeline(&model.pipes.q_norm_rope_fused);
    cp.set_bind_group(0, &model.cached.q_norm_rope_fused, &[off]);
    cp.dispatch_workgroups(cfg.n_head, m_tokens, 1);
}

// =========================================================================
// Bench / microbench (CLI utilities; gated behind the bench-internals feature
// so they don't pollute the public surface).
// =========================================================================

#[cfg(feature = "bench-internals")]
#[allow(
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    reason = "internal"
)]
pub mod bench_internals {
    use std::cmp::Ordering;
    use std::time::Instant;

    use wgpu::PollType;

    use super::{
        ATTN_CHUNK_SIZE, AttnMergeParams, AttnSplitParams, Config, EmbedParams,
        MatvecFusedNormedParams, MatvecSiluParams, Model, Result, StepEncoder, TopKParams,
        WeightSet, kv_layer_offsets, kv_writeback_fused, matvec_bg, matvec_fused_normed_bg,
        matvec_q1_0, prefill_matmul_topk, q_norm_rope_fused, rms_norm, step_matvec_no_sample,
        step_matvec_topk,
    };
    use crate::error::PotError;

    pub fn bench(model: &Model, pp_n: u32, tg_n: u32, repeats: u32) -> Result<()> {
        let cfg = &model.cfg;
        eprintln!("--- bench: pp={pp_n}, tg={tg_n}, repeats={repeats} (after 1 warmup) ---");

        let prompt: Vec<u32> = (0..pp_n).map(|i| (i % (cfg.n_vocab - 1)) + 1).collect();

        // ----- warm up -----
        let _ = prefill_matmul_topk(model, &prompt[..pp_n.min(model.m_max) as usize], 0, 1)?;
        step_matvec_no_sample(model, 1u32, 0);
        if let Err(e) = model.device.poll(PollType::wait_indefinitely()) {
            model.check_device()?;
            return Err(PotError::Poll(e));
        }

        // ----- pp{pp_n} -----
        let mut pp_times = Vec::with_capacity(repeats as usize);
        for _ in 0..repeats {
            let t = Instant::now();
            let _ = prefill_matmul_topk(model, &prompt, 0, 1)?;
            if let Err(e) = model.device.poll(PollType::wait_indefinitely()) {
                model.check_device()?;
                return Err(PotError::Poll(e));
            }
            pp_times.push(t.elapsed().as_secs_f32());
        }
        let pp_mean = pp_times.iter().sum::<f32>() / pp_times.len() as f32;
        let pp_std = stddev(&pp_times, pp_mean);
        let pp_t_s_mean = pp_n as f32 / pp_mean;
        let pp_t_s_std = pp_t_s_mean * (pp_std / pp_mean);

        // ----- tg{tg_n} (from empty KV) -----
        let mut tg_times = Vec::with_capacity(repeats as usize);
        for _ in 0..repeats {
            let t = Instant::now();
            for pos in 0..tg_n {
                let (_, _) = step_matvec_topk(model, 1u32, pos, 1)?;
            }
            if let Err(e) = model.device.poll(PollType::wait_indefinitely()) {
                model.check_device()?;
                return Err(PotError::Poll(e));
            }
            tg_times.push(t.elapsed().as_secs_f32());
        }
        let tg_mean = tg_times.iter().sum::<f32>() / tg_times.len() as f32;
        let tg_std = stddev(&tg_times, tg_mean);
        let tg_t_s_mean = tg_n as f32 / tg_mean;
        let tg_t_s_std = tg_t_s_mean * (tg_std / tg_mean);

        println!();
        println!("| backend            |          test |               t/s |");
        println!("| ------------------ | ------------- | ----------------: |");
        println!(
            "| bonsai-pot        |        pp{pp_n:<3} | {pp_t_s_mean:>9.2} ± {pp_t_s_std:>5.2} |"
        );
        println!(
            "| bonsai-pot        |        tg{tg_n:<3} | {tg_t_s_mean:>9.2} ± {tg_t_s_std:>5.2} |"
        );
        Ok(())
    }

    fn stddev(xs: &[f32], mean: f32) -> f32 {
        if xs.len() < 2 {
            return 0.0;
        }
        let var: f32 =
            xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / (xs.len() - 1) as f32;
        var.sqrt()
    }

    fn bench_fused_normed(
        model: &Model,
        cfg: &Config,
        se: &mut StepEncoder,
        k: u32,
        input_offset: u32,
        w_norm_off: u32,
        weights: WeightSet,
        ranges: &[(u32, u32, u32, u32)],
    ) {
        const ROWS_PER_WG: u32 = 8;
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
        let off = se.alloc_uniform(&p);
        let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matvec_fused_normed"),
            timestamp_writes: None,
        });
        cp.set_pipeline(&model.pipes.matvec_fused_normed);
        cp.set_bind_group(0, matvec_fused_normed_bg(model, weights), &[off]);
        cp.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    pub fn microbench_tg(model: &Model, repeats: u32) -> Result<()> {
        type DispatchFn<'a> = Box<dyn Fn(&Model, &Config, &mut StepEncoder) + 'a>;

        let cfg = &model.cfg;
        let n_per_cb: u32 = 200;
        let attn_pos: u32 = 64;

        eprintln!("--- microbench tg (N={n_per_cb}/CB, repeats={repeats}) ---");

        let mut entries: Vec<(String, u32, DispatchFn)> = Vec::new();

        // (label, k, n, calls/step, weight_set, w_d_offset, w_qs_offset)
        let lt0 = &model.layer_tensors[0];
        let ot = &model.output_tensors;
        let n_layer = cfg.n_layer;
        // Wdown is no longer here — it's fused with silu_mul into a single
        // `matvec_q1_0_silu` dispatch (entry below). Wo and LM_head still use
        // the plain `matvec_q1_0` kernel.
        let matvec_shapes: Vec<(String, u32, u32, u32, WeightSet, u32, u32)> = vec![
            (
                format!("matvec Wo K={} N={}", cfg.q_dim, cfg.n_embd),
                cfg.q_dim,
                cfg.n_embd,
                n_layer,
                WeightSet::Attn,
                lt0.wo.0,
                lt0.wo.1,
            ),
            (
                format!("matvec LM_head K={} N={}", cfg.n_embd, cfg.n_vocab),
                cfg.n_embd,
                cfg.n_vocab,
                1,
                WeightSet::Embed,
                ot.lm_head_d,
                ot.lm_head_qs,
            ),
        ];
        for (label, k, n, calls, ws, w_d, w_qs) in &matvec_shapes {
            let (label, k, n, calls, ws, w_d, w_qs) =
                (label.clone(), *k, *n, *calls, *ws, *w_d, *w_qs);
            entries.push((
                label,
                calls,
                Box::new(move |model, _cfg, se| {
                    matvec_q1_0(
                        model,
                        se,
                        k,
                        n,
                        ws,
                        w_d,
                        w_qs,
                        model.act_layout.x_norm,
                        model.act_layout.x,
                        false,
                    );
                }),
            ));
        }

        entries.push((
            format!(
                "matvec QKV fused+normed N={}",
                cfg.q_dim + cfg.kv_dim + cfg.kv_dim
            ),
            n_layer,
            Box::new(move |model, cfg, se| {
                let lt = &model.layer_tensors[0];
                let ranges = [
                    (lt.wq.0, lt.wq.1, cfg.q_dim, model.act_layout.q),
                    (lt.wk.0, lt.wk.1, cfg.kv_dim, model.act_layout.k_cur),
                    (lt.wv.0, lt.wv.1, cfg.kv_dim, model.act_layout.v_cur),
                ];
                bench_fused_normed(
                    model,
                    cfg,
                    se,
                    cfg.n_embd,
                    model.act_layout.x,
                    lt.attn_norm_off,
                    WeightSet::Attn,
                    &ranges,
                );
            }),
        ));
        entries.push((
            format!("matvec gate_up fused+normed N={}", 2 * cfg.n_ff),
            n_layer,
            Box::new(move |model, cfg, se| {
                let lt = &model.layer_tensors[0];
                let ranges = [
                    (lt.wg.0, lt.wg.1, cfg.n_ff, model.act_layout.gate),
                    (lt.wu.0, lt.wu.1, cfg.n_ff, model.act_layout.up),
                    (0, 0, 0, 0),
                ];
                bench_fused_normed(
                    model,
                    cfg,
                    se,
                    cfg.n_embd,
                    model.act_layout.x,
                    lt.ffn_norm_off,
                    WeightSet::FfnGU,
                    &ranges,
                );
            }),
        ));

        // (label, n_groups, group_size, calls/step, norm_off)
        // NOTE: per-layer rms_norm calls (attn_norm, ffn_norm) are folded into
        // matvec_q1_0_fused_normed; per-K-head rms_norm + K-side rope are
        // folded into kv_writeback_fused; per-Q-head rms_norm + Q-side rope
        // are folded into q_norm_rope_fused. The lone remaining rms_norm
        // dispatch is the output_norm in the LM-head suffix (1 call/step).
        let rms_shapes: Vec<(String, u32, u32, u32, u32)> = vec![(
            format!("rms_norm ng=1 gs={} (output_norm)", cfg.n_embd),
            1,
            cfg.n_embd,
            1,
            ot.output_norm_off,
        )];
        for (label, ng, gs, calls, norm_off) in &rms_shapes {
            let (label, ng, gs, calls, norm_off) = (label.clone(), *ng, *gs, *calls, *norm_off);
            entries.push((
                label,
                calls,
                Box::new(move |model, cfg, se| {
                    rms_norm(
                        model,
                        cfg,
                        se,
                        ng,
                        gs,
                        model.act_layout.x_norm,
                        model.act_layout.x_norm,
                        norm_off,
                    );
                }),
            ));
        }

        entries.push((
            format!("matvec Wdown_silu K={} N={}", cfg.n_ff, cfg.n_embd),
            n_layer,
            Box::new(move |model, cfg, se| {
                const ROWS_PER_WG: u32 = 8;
                let lt = &model.layer_tensors[0];
                let (k, n) = (cfg.n_ff, cfg.n_embd);
                let n_wg = n.div_ceil(ROWS_PER_WG);
                let dispatch_x = n_wg.min(65535);
                let dispatch_y = n_wg.div_ceil(dispatch_x);
                let p = MatvecSiluParams {
                    k,
                    n,
                    d_offset: lt.wd.0,
                    qs_offset: lt.wd.1,
                    gate_offset: model.act_layout.gate,
                    up_offset: model.act_layout.up,
                    output_offset: model.act_layout.x,
                    accumulate: 1,
                    dispatch_x_dim: dispatch_x,
                };
                let off = se.alloc_uniform(&p);
                let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("matvec_silu"),
                    timestamp_writes: None,
                });
                cp.set_pipeline(&model.pipes.matvec_silu);
                cp.set_bind_group(0, matvec_bg(model, WeightSet::FfnD), &[off]);
                cp.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }),
        ));

        entries.push((
            format!("attn_split+merge pos={attn_pos}"),
            n_layer,
            Box::new(move |model, cfg, se| {
                let (d_word, qs_byte) = kv_layer_offsets(cfg, model.max_seq, 0);
                let cur_pos = attn_pos + 1;
                let n_chunks_active = cur_pos.div_ceil(ATTN_CHUNK_SIZE);
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
                let pm = AttnMergeParams {
                    head_dim: cfg.head_dim,
                    n_head: cfg.n_head,
                    out_offset: model.act_layout.attn_out,
                    n_chunks_active,
                };
                let ps_off = se.alloc_uniform(&ps);
                let pm_off = se.alloc_uniform(&pm);
                let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("attn_split+merge"),
                    timestamp_writes: None,
                });
                cp.set_pipeline(&model.pipes.attention_split);
                cp.set_bind_group(0, &model.cached.attn_split, &[ps_off]);
                cp.dispatch_workgroups(cfg.n_kv_head, n_chunks_active, 1);
                cp.set_pipeline(&model.pipes.attention_merge);
                cp.set_bind_group(0, &model.cached.attn_merge, &[pm_off]);
                cp.dispatch_workgroups(cfg.n_head, 1, 1);
            }),
        ));

        entries.push((
            "embed                           ".to_string(),
            1,
            Box::new(move |model, cfg, se| {
                let ot = &model.output_tensors;
                let p = EmbedParams {
                    k: cfg.n_embd,
                    d_offset: ot.token_embd_d,
                    qs_offset: ot.token_embd_qs,
                    output_offset: model.act_layout.x,
                    sample_offset: 0,
                };
                let off = se.alloc_uniform(&p);
                let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("embed"),
                    timestamp_writes: None,
                });
                cp.set_pipeline(&model.pipes.embed);
                cp.set_bind_group(0, &model.cached.embed, &[off]);
                cp.dispatch_workgroups(1, 1, 1);
            }),
        ));

        entries.push((
            "topk_reduce n=n_vocab           ".to_string(),
            1,
            Box::new(move |model, _cfg, se| {
                let p = TopKParams {
                    n: model.cfg.n_vocab,
                    in_offset: model.act_layout.logits,
                    out_offset: 0,
                    k: 1,
                };
                let off = se.alloc_uniform(&p);
                let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("topk_reduce"),
                    timestamp_writes: None,
                });
                cp.set_pipeline(&model.pipes.topk_reduce);
                cp.set_bind_group(0, &model.cached.topk_reduce, &[off]);
                cp.dispatch_workgroups(1, 1, 1);
            }),
        ));

        // Warmup
        {
            let mut se = StepEncoder::new(model);
            for (_l, _c, f) in &entries {
                f(model, cfg, &mut se);
            }
            kv_writeback_fused(
                model,
                cfg,
                &mut se,
                model.act_layout.k_cur,
                model.act_layout.v_cur,
                lt0.attn_k_norm_off,
                /*layer_il=*/ 0,
                /*pos_base=*/ 0,
                /*m_tokens=*/ 1,
            );
            q_norm_rope_fused(
                model,
                cfg,
                &mut se,
                model.act_layout.q,
                lt0.attn_q_norm_off,
                /*pos_base=*/ 0,
                /*m_tokens=*/ 1,
            );
            let cb = se.finish();
            model.queue.submit(Some(cb));
            if let Err(e) = model.device.poll(PollType::wait_indefinitely()) {
                model.check_device()?;
                return Err(PotError::Poll(e));
            }
        }

        let mut breakdown: Vec<(String, u32, f32)> = Vec::new();
        for (label, calls, build) in &entries {
            let mut samples = Vec::with_capacity(repeats as usize);
            for _ in 0..repeats {
                let t = Instant::now();
                let mut se = StepEncoder::new(model);
                for _ in 0..n_per_cb {
                    build(model, cfg, &mut se);
                }
                let cb = se.finish();
                model.queue.submit(Some(cb));
                if let Err(e) = model.device.poll(PollType::wait_indefinitely()) {
                    model.check_device()?;
                    return Err(PotError::Poll(e));
                }
                samples.push(t.elapsed().as_secs_f32());
            }
            let mean = samples.iter().sum::<f32>() / samples.len() as f32;
            let per_call_us = mean / n_per_cb as f32 * 1e6;
            breakdown.push((label.clone(), *calls, per_call_us));
        }

        {
            let mut samples = Vec::with_capacity(repeats as usize);
            for _ in 0..repeats {
                let t = Instant::now();
                let mut se = StepEncoder::new(model);
                for _ in 0..n_per_cb {
                    kv_writeback_fused(
                        model,
                        cfg,
                        &mut se,
                        model.act_layout.k_cur,
                        model.act_layout.v_cur,
                        lt0.attn_k_norm_off,
                        /*layer_il=*/ 0,
                        /*pos_base=*/ 0,
                        /*m_tokens=*/ 1,
                    );
                }
                let cb = se.finish();
                model.queue.submit(Some(cb));
                if let Err(e) = model.device.poll(PollType::wait_indefinitely()) {
                    model.check_device()?;
                    return Err(PotError::Poll(e));
                }
                samples.push(t.elapsed().as_secs_f32());
            }
            let mean = samples.iter().sum::<f32>() / samples.len() as f32;
            let per_pair_us = mean / n_per_cb as f32 * 1e6;
            breakdown.push((
                "kv_writeback_fused (K rms+rope+Q8_0 + V Q8_0)".to_string(),
                n_layer,
                per_pair_us,
            ));
        }

        {
            let mut samples = Vec::with_capacity(repeats as usize);
            for _ in 0..repeats {
                let t = Instant::now();
                let mut se = StepEncoder::new(model);
                for _ in 0..n_per_cb {
                    q_norm_rope_fused(
                        model,
                        cfg,
                        &mut se,
                        model.act_layout.q,
                        lt0.attn_q_norm_off,
                        /*pos_base=*/ 0,
                        /*m_tokens=*/ 1,
                    );
                }
                let cb = se.finish();
                model.queue.submit(Some(cb));
                if let Err(e) = model.device.poll(PollType::wait_indefinitely()) {
                    model.check_device()?;
                    return Err(PotError::Poll(e));
                }
                samples.push(t.elapsed().as_secs_f32());
            }
            let mean = samples.iter().sum::<f32>() / samples.len() as f32;
            let per_call_us = mean / n_per_cb as f32 * 1e6;
            breakdown.push((
                "q_norm_rope_fused (Q rms+rope)".to_string(),
                n_layer,
                per_call_us,
            ));
        }

        // Calibrate by timing real tg steps now that the GPU is warm. Scale
        // per-kernel times so the corrected total matches measured throughput
        // (covers per-pass overhead + readback that isolated dispatches miss).
        let step_ms: f32 = {
            let n: u32 = 32;
            let mut pos = 0u32;
            step_matvec_no_sample(model, 1, pos);
            if let Err(e) = model.device.poll(PollType::wait_indefinitely()) {
                model.check_device()?;
                return Err(PotError::Poll(e));
            }
            pos += 1;
            let t = Instant::now();
            for _ in 0..n {
                step_matvec_topk(model, 1, pos, 1)?;
                pos += 1;
            }
            let per_step = t.elapsed().as_secs_f32() / n as f32;
            eprintln!(
                "calibration (incl. readback): {:.3} ms/step  ({:.1} t/s)",
                per_step * 1000.0,
                1.0 / per_step
            );
            per_step * 1000.0
        };

        let raw_total_ms: f32 = breakdown
            .iter()
            .map(|(_, c, us)| (*c as f32) * (*us) / 1000.0)
            .sum();
        let scale = step_ms / raw_total_ms;
        let mut sorted = breakdown.clone();
        sorted.sort_by(|a, b| {
            let ma = a.1 as f32 * a.2;
            let mb = b.1 as f32 * b.2;
            mb.partial_cmp(&ma).unwrap_or(Ordering::Equal)
        });
        let adj_total_ms: f32 = sorted
            .iter()
            .map(|(_, c, us)| (*c as f32) * (*us) * scale / 1000.0)
            .sum();

        println!();
        println!(
            "| kernel                                        | calls/step | raw us | adj us | raw ms | adj ms | %step |"
        );
        println!(
            "|-----------------------------------------------|-----------:|-------:|-------:|-------:|-------:|------:|"
        );
        for (label, calls, raw_us) in &sorted {
            let adj_us = raw_us * scale;
            let raw_ms = (*calls as f32) * (*raw_us) / 1000.0;
            let adj_ms = (*calls as f32) * adj_us / 1000.0;
            let pct = 100.0 * adj_ms / adj_total_ms;
            println!(
                "| {label:<45} | {calls:>10} | {raw_us:>6.2} | {adj_us:>6.2} | {raw_ms:>6.3} | {adj_ms:>6.3} | {pct:>5.1} |"
            );
        }
        println!(
            "|-----------------------------------------------|-----------:|-------:|-------:|-------:|-------:|------:|"
        );
        println!(
            "| TOTAL (isolated)                              |            |        |        | {raw_total_ms:>6.3} | {adj_total_ms:>6.3} |       |"
        );
        println!();
        println!("raw (isolated):           {:.1} t/s", 1000.0 / raw_total_ms);
        println!(
            "calibrated step:          {:.3} ms  ({:.1} t/s)",
            step_ms,
            1000.0 / step_ms
        );
        println!("scale factor:             ×{scale:.4}");
        println!("corrected (scaled):       {:.1} t/s", 1000.0 / adj_total_ms);
        Ok(())
    }
}
