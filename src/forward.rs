//! Forward pass and per-step inference helpers.
//!
//! Two execution modes:
//!   - **matvec single-token** (`step_matvec_topk`, `prefill_matvec_loop_topk`):
//!     processes one token at a time via the multiply-free Q1_0 matvec kernel.
//!     Used for token generation and for incremental prefill (when there's an
//!     existing KV cache prefix).
//!   - **matmul batched prefill** (`prefill_matmul_topk`): processes the prompt
//!     as one batch using dot4I8Packed matmul with a Q8_0 quantize-activation
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
//! `format!` + HashMap lookup + clone on every dispatch; bind groups are also
//! precomputed at load time (`Model::cached`) and shared across all dispatches
//! of a given (kind, weight buffer) pair, since the dynamic uniform offset is
//! the only per-dispatch variation.

use crate::error::{PotError, Result};
use crate::model::*;
use bytemuck::Pod;

/// Byte size of one activation / KV / norm-weight element.
const ACT_ELEM_BYTES: u64 = std::mem::size_of::<half::f16>() as u64;

// ---------- per-step encoder + uniform pool ---------------------------------

pub(crate) struct UniformPool {
    cpu: Vec<u8>,
    next_slot: u64,
}

impl UniformPool {
    fn new() -> Self {
        Self { cpu: Vec::with_capacity(64 * 1024), next_slot: 0 }
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

pub(crate) struct StepEncoder<'a> {
    model: &'a Model,
    pub encoder: wgpu::CommandEncoder,
    pub uniforms: UniformPool,
}

impl<'a> StepEncoder<'a> {
    pub fn new(model: &'a Model) -> Self {
        let encoder = model.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("step"),
        });
        Self { model, encoder, uniforms: UniformPool::new() }
    }

    pub fn alloc_uniform<T: Pod>(&mut self, params: &T) -> u32 {
        self.uniforms.alloc(params)
    }

    /// Append a `sample → readback` copy to this encoder so the readback
    /// transfer rides in the same command buffer as the step it follows.
    /// Avoids a separate submit purely for the copy.
    pub fn copy_sample_to_readback(&mut self, bytes: u64) {
        self.encoder.copy_buffer_to_buffer(
            &self.model.buffers.sample, 0,
            &self.model.buffers.readback, 0,
            bytes,
        );
    }

    pub fn finish(self) -> wgpu::CommandBuffer {
        if !self.uniforms.cpu.is_empty() {
            self.model.queue.write_buffer(&self.model.buffers.uniform, 0, &self.uniforms.cpu);
        }
        self.encoder.finish()
    }
}

// ---------- weight-set selection -------------------------------------------

fn matvec_bg(model: &Model, ws: WeightSet) -> &wgpu::BindGroup {
    match ws {
        WeightSet::Attn  => &model.cached.matvec_w_attn,
        WeightSet::FfnGU => &model.cached.matvec_w_ffn_gu,
        WeightSet::FfnD  => &model.cached.matvec_w_ffn_d,
        WeightSet::Embed => &model.cached.matvec_w_embed,
    }
}

fn matmul_bg(model: &Model, ws: WeightSet) -> &wgpu::BindGroup {
    match ws {
        WeightSet::Attn  => &model.cached.matmul_w_attn,
        WeightSet::FfnGU => &model.cached.matmul_w_ffn_gu,
        WeightSet::FfnD  => &model.cached.matmul_w_ffn_d,
        WeightSet::Embed => &model.cached.matmul_w_embed,
    }
}

// ---------- in-pass kernel dispatch helpers ---------------------------------
// These variants take a `&mut wgpu::ComputePass` already opened by the caller,
// allowing many dispatches to share one pass and amortize the
// begin_compute_pass cost (~25us each on RADV).

fn dispatch_rms_norm(
    model: &Model, cfg: &Config, uniforms: &mut UniformPool,
    pass: &mut wgpu::ComputePass<'_>,
    n_groups: u32, group_size: u32, in_off: u32, out_off: u32, w_off: u32,
) {
    let p = RmsNormParams {
        group_size, n_groups,
        input_offset: in_off, output_offset: out_off, weight_offset: w_off,
        eps: cfg.rms_eps,
        ..Default::default()
    };
    let off = uniforms.alloc(&p);
    pass.set_pipeline(&model.pipes.rms_norm);
    pass.set_bind_group(0, &model.cached.rms_norm, &[off]);
    pass.dispatch_workgroups(n_groups, 1, 1);
}

fn dispatch_rope(
    model: &Model, cfg: &Config, uniforms: &mut UniformPool,
    pass: &mut wgpu::ComputePass<'_>,
    data_off: u32, n_tokens: u32, n_heads: u32, pos_base: u32,
) {
    let p = RopeParams {
        head_dim: cfg.head_dim, n_heads, n_tokens, pos_base,
        data_offset: data_off,
        ..Default::default()
    };
    let off = uniforms.alloc(&p);
    pass.set_pipeline(&model.pipes.rope_neox);
    pass.set_bind_group(0, &model.cached.rope, &[off]);
    pass.dispatch_workgroups(n_tokens, n_heads, 1);
}

fn dispatch_matvec_q1_0(
    model: &Model, uniforms: &mut UniformPool, pass: &mut wgpu::ComputePass<'_>,
    k: u32, n: u32,
    weights: WeightSet, w_d: u32, w_qs: u32,
    in_off: u32, out_off: u32, accumulate: bool,
) {
    const ROWS_PER_WG: u32 = 8;
    let n_wg = (n + ROWS_PER_WG - 1) / ROWS_PER_WG;
    let dispatch_x = n_wg.min(65535);
    let dispatch_y = (n_wg + dispatch_x - 1) / dispatch_x;
    let p = MatvecParams {
        k, n,
        d_offset: w_d, qs_offset: w_qs,
        input_offset: in_off, output_offset: out_off,
        accumulate: if accumulate { 1 } else { 0 },
        dispatch_x_dim: dispatch_x,
    };
    let off = uniforms.alloc(&p);
    pass.set_pipeline(&model.pipes.matvec);
    pass.set_bind_group(0, matvec_bg(model, weights), &[off]);
    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
}

fn dispatch_matvec_q1_0_fused(
    model: &Model, uniforms: &mut UniformPool, pass: &mut wgpu::ComputePass<'_>,
    k: u32, input_offset: u32, weights: WeightSet,
    ranges: &[(u32, u32, u32, u32)],
) {
    debug_assert!(ranges.len() == 2 || ranges.len() == 3);
    for (_, _, n, _) in ranges { debug_assert!(n % 8 == 0); }
    let r = |i: usize| ranges.get(i).copied().unwrap_or((0, 0, 0, 0));
    let (d0, qs0, n0, o0) = r(0);
    let (d1, qs1, n1, o1) = r(1);
    let (d2, qs2, n2, o2) = r(2);
    let n_total = n0 + n1 + n2;
    const ROWS_PER_WG: u32 = 8;
    let n_wg = (n_total + ROWS_PER_WG - 1) / ROWS_PER_WG;
    let dispatch_x = n_wg.min(65535);
    let dispatch_y = (n_wg + dispatch_x - 1) / dispatch_x;
    let p = MatvecFusedParams {
        k, n_total, input_offset, dispatch_x_dim: dispatch_x,
        d_offset_0: d0, qs_offset_0: qs0, n_0: n0, output_offset_0: o0,
        d_offset_1: d1, qs_offset_1: qs1, n_1: n1, output_offset_1: o1,
        d_offset_2: d2, qs_offset_2: qs2, n_2: n2, output_offset_2: o2,
    };
    let off = uniforms.alloc(&p);
    pass.set_pipeline(&model.pipes.matvec_fused);
    pass.set_bind_group(0, matvec_bg(model, weights), &[off]);
    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
}

fn dispatch_silu_mul(
    model: &Model, uniforms: &mut UniformPool, pass: &mut wgpu::ComputePass<'_>,
    n: u32, m: u32, gate_off: u32, up_off: u32, out_off: u32,
) {
    let total = n * m;
    let groups = (total + 63) / 64;
    let dispatch_x = groups.min(65535);
    let dispatch_y = (groups + dispatch_x - 1) / dispatch_x;
    let p = SiluMulParams {
        n, m, gate_offset: gate_off, up_offset: up_off, out_offset: out_off,
        dispatch_x_count: dispatch_x * 64,
        ..Default::default()
    };
    let off = uniforms.alloc(&p);
    pass.set_pipeline(&model.pipes.silu_mul);
    pass.set_bind_group(0, &model.cached.silu_mul, &[off]);
    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
}

fn dispatch_topk_reduce(
    model: &Model, uniforms: &mut UniformPool, pass: &mut wgpu::ComputePass<'_>,
    n: u32, k: u32, in_off: u32, out_off_u32: u32,
) {
    let p = TopKParams { n, in_offset: in_off, out_offset: out_off_u32, k };
    let off = uniforms.alloc(&p);
    pass.set_pipeline(&model.pipes.topk_reduce);
    pass.set_bind_group(0, &model.cached.topk_reduce, &[off]);
    pass.dispatch_workgroups(1, 1, 1);
}

// ---------- async readback helper -------------------------------------------

/// Await the `sample → readback` copy that was already encoded into the
/// step's command buffer (via [`StepEncoder::copy_sample_to_readback`]) and
/// return the K f32 logits + K u32 indices the caller asked for.
///
/// This must be called AFTER the step's command buffer has been submitted —
/// i.e. the readback copy is in flight. There is no separate submit here.
async fn await_topk_readback(model: &Model, k: u32) -> Result<(Vec<f32>, Vec<u32>)> {
    let bytes = (k as u64) * 8; // K f32 + K u32
    let slice = model.buffers.readback.slice(0..bytes);
    let (s, r) = futures_intrusive::channel::shared::oneshot_channel::<
        std::result::Result<(), wgpu::BufferAsyncError>,
    >();
    slice.map_async(wgpu::MapMode::Read, move |res| { let _ = s.send(res); });
    model.device.poll(wgpu::PollType::wait_indefinitely()).expect("device.poll failed");
    match r.receive().await {
        Some(Ok(())) => {}
        Some(Err(e)) => return Err(PotError::BufferMap(e)),
        None => unreachable!("oneshot channel dropped without sending"),
    }
    let data = slice.get_mapped_range();
    let words: &[u32] = bytemuck::cast_slice(&data[..bytes as usize]);
    let logits: Vec<f32> = words[..k as usize].iter().map(|w| f32::from_bits(*w)).collect();
    let indices: Vec<u32> = words[k as usize..2 * k as usize].to_vec();
    drop(data);
    model.buffers.readback.unmap();
    Ok((logits, indices))
}

// ---------- single-token forward (matvec) ----------------------------------

/// Encode one tg step into the given encoder. The input token is read from
/// `sample[sample_in]`. The final `topk_reduce` writes K f32 logits at
/// `sample[topk_out_u32_base..+K]` and K u32 indices at
/// `sample[topk_out_u32_base + k..+2*k]`.
pub(crate) fn encode_step_matvec(
    se: &mut StepEncoder, cfg: &Config,
    sample_in: u32, topk_out_u32_base: u32, pos: u32, k: u32,
) {
    let cache_row_bytes = cfg.kv_dim as u64 * ACT_ELEM_BYTES;
    let StepEncoder { model: m, encoder, uniforms } = se;
    let ot = &m.output_tensors;
    // Pass 0: embed + layer 0 pre-kv
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("phase0"), timestamp_writes: None });
        let p = EmbedParams {
            k: cfg.n_embd,
            d_offset: ot.token_embd_d,
            qs_offset: ot.token_embd_qs,
            output_offset: m.act_layout.x,
            sample_offset: sample_in,
            ..Default::default()
        };
        let off = uniforms.alloc(&p);
        pass.set_pipeline(&m.pipes.embed);
        pass.set_bind_group(0, &m.cached.embed, &[off]);
        pass.dispatch_workgroups(1, 1, 1);
        layer_pre_kv_in_pass(m, cfg, uniforms, &mut pass, 0, pos);
    }
    for il in 0..cfg.n_layer {
        let layer_offset_bytes = (il as u64) * (m.max_seq as u64) * cache_row_bytes;
        let dst_offset_bytes = layer_offset_bytes + (pos as u64) * cache_row_bytes;
        let k_src_bytes = (m.act_layout.k_cur as u64) * ACT_ELEM_BYTES;
        let v_src_bytes = (m.act_layout.v_cur as u64) * ACT_ELEM_BYTES;
        encoder.copy_buffer_to_buffer(&m.buffers.act, k_src_bytes, &m.buffers.kv_k, dst_offset_bytes, cache_row_bytes);
        encoder.copy_buffer_to_buffer(&m.buffers.act, v_src_bytes, &m.buffers.kv_v, dst_offset_bytes, cache_row_bytes);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("phase"), timestamp_writes: None,
        });
        layer_post_kv_in_pass(m, cfg, uniforms, &mut pass, il, pos);
        if il + 1 < cfg.n_layer {
            layer_pre_kv_in_pass(m, cfg, uniforms, &mut pass, il + 1, pos);
        } else {
            dispatch_rms_norm(m, cfg, uniforms, &mut pass,
                              1, cfg.n_embd, m.act_layout.x, m.act_layout.x_norm,
                              ot.output_norm_off);
            dispatch_matvec_q1_0(m, uniforms, &mut pass,
                                 cfg.n_embd, cfg.n_vocab,
                                 WeightSet::Embed, ot.token_embd_d, ot.token_embd_qs,
                                 m.act_layout.x_norm, m.act_layout.logits, false);
            dispatch_topk_reduce(m, uniforms, &mut pass, cfg.n_vocab, k,
                                 m.act_layout.logits, topk_out_u32_base);
        }
    }
}

/// Run one matvec step at `pos`, reading the current token from CPU and
/// returning the top-`k` logits + indices for the next token.
pub(crate) async fn step_matvec_topk(
    model: &Model, token_id: u32, pos: u32, k: u32,
) -> Result<(Vec<f32>, Vec<u32>)> {
    let k = k.min(TOPK_MAX).max(1);
    model.queue.write_buffer(&model.buffers.sample, 0, bytemuck::bytes_of(&token_id));
    let mut se = StepEncoder::new(model);
    encode_step_matvec(&mut se, &model.cfg, 0, 0, pos, k);
    // Fold the readback copy into the same command buffer so the step + the
    // sample→readback transfer are one submit, not two.
    se.copy_sample_to_readback((k as u64) * 8);
    let cb = se.finish();
    model.queue.submit(Some(cb));
    await_topk_readback(model, k).await
}

/// Same as [`step_matvec_topk`] but does not perform any sampling readback.
/// Used by perf benches to avoid coupling forward-pass cost to readback I/O —
/// callers `device.poll(wait_indefinitely)` themselves to time the work.
#[cfg(feature = "bench-internals")]
pub(crate) fn step_matvec_no_sample(model: &Model, token_id: u32, pos: u32) {
    model.queue.write_buffer(&model.buffers.sample, 0, bytemuck::bytes_of(&token_id));
    let mut se = StepEncoder::new(model);
    // We still encode the topk_reduce dispatch (with k=1, the single argmax case)
    // so the timing reflects real generation cost; we just skip the readback.
    encode_step_matvec(&mut se, &model.cfg, 0, 0, pos, 1);
    let cb = se.finish();
    model.queue.submit(Some(cb));
}

/// Pre-KV-copy block of one layer: rms_norm → QKV fused → q/k norms → rope.
fn layer_pre_kv_in_pass(
    model: &Model, cfg: &Config, uniforms: &mut UniformPool,
    pass: &mut wgpu::ComputePass<'_>, il: u32, pos: u32,
) {
    let lt = &model.layer_tensors[il as usize];
    dispatch_rms_norm(model, cfg, uniforms, pass, 1, cfg.n_embd,
                      model.act_layout.x, model.act_layout.x_norm, lt.attn_norm_off);

    dispatch_matvec_q1_0_fused(model, uniforms, pass, cfg.n_embd, model.act_layout.x_norm,
                               WeightSet::Attn, &[
        (lt.wq.0, lt.wq.1, cfg.q_dim,  model.act_layout.q),
        (lt.wk.0, lt.wk.1, cfg.kv_dim, model.act_layout.k_cur),
        (lt.wv.0, lt.wv.1, cfg.kv_dim, model.act_layout.v_cur),
    ]);

    dispatch_rms_norm(model, cfg, uniforms, pass, cfg.n_head, cfg.head_dim,
                      model.act_layout.q, model.act_layout.q, lt.attn_q_norm_off);
    dispatch_rms_norm(model, cfg, uniforms, pass, cfg.n_kv_head, cfg.head_dim,
                      model.act_layout.k_cur, model.act_layout.k_cur, lt.attn_k_norm_off);

    dispatch_rope(model, cfg, uniforms, pass, model.act_layout.q,     1, cfg.n_head,    pos);
    dispatch_rope(model, cfg, uniforms, pass, model.act_layout.k_cur, 1, cfg.n_kv_head, pos);
}

/// Post-KV-copy block of one layer: attention → Wo (resid) → ffn_norm
/// → gate-up fused → silu_mul → Wd (resid).
fn layer_post_kv_in_pass(
    model: &Model, cfg: &Config, uniforms: &mut UniformPool,
    pass: &mut wgpu::ComputePass<'_>, il: u32, pos: u32,
) {
    let lt = &model.layer_tensors[il as usize];
    let cache_row_bytes = cfg.kv_dim as u64 * ACT_ELEM_BYTES;
    let layer_offset_bytes = (il as u64) * (model.max_seq as u64) * cache_row_bytes;

    {
        let p = AttnParams {
            head_dim: cfg.head_dim, n_head: cfg.n_head, n_kv_head: cfg.n_kv_head,
            pos: pos + 1,
            kv_stride: cfg.kv_dim,
            q_offset: model.act_layout.q,
            k_cache_offset: ((layer_offset_bytes / ACT_ELEM_BYTES) as u32),
            v_cache_offset: ((layer_offset_bytes / ACT_ELEM_BYTES) as u32),
            out_offset: model.act_layout.attn_out,
            scale: 1.0 / (cfg.head_dim as f32).sqrt(),
            m_tokens: 1, is_prefill: 0,
        };
        let off = uniforms.alloc(&p);
        pass.set_pipeline(&model.pipes.attention);
        pass.set_bind_group(0, &model.cached.attn, &[off]);
        pass.dispatch_workgroups(cfg.n_head, 1, 1);
    }

    dispatch_matvec_q1_0(model, uniforms, pass, cfg.q_dim, cfg.n_embd,
                         WeightSet::Attn, lt.wo.0, lt.wo.1,
                         model.act_layout.attn_out, model.act_layout.x, true /*accumulate*/);

    dispatch_rms_norm(model, cfg, uniforms, pass, 1, cfg.n_embd,
                      model.act_layout.x, model.act_layout.x_norm, lt.ffn_norm_off);

    dispatch_matvec_q1_0_fused(model, uniforms, pass, cfg.n_embd, model.act_layout.x_norm,
                               WeightSet::FfnGU, &[
        (lt.wg.0, lt.wg.1, cfg.n_ff, model.act_layout.gate),
        (lt.wu.0, lt.wu.1, cfg.n_ff, model.act_layout.up),
    ]);

    dispatch_silu_mul(model, uniforms, pass, cfg.n_ff, 1,
                      model.act_layout.gate, model.act_layout.up, model.act_layout.ffn_in);

    dispatch_matvec_q1_0(model, uniforms, pass, cfg.n_ff, cfg.n_embd,
                         WeightSet::FfnD, lt.wd.0, lt.wd.1,
                         model.act_layout.ffn_in, model.act_layout.x, true /*accumulate*/);
}

/// Run the matvec single-token path over every token in `prompt`, advancing
/// `pos` from `pos_base` to `pos_base + prompt.len()`, and return the top-K
/// candidates from the LAST token's logits. Suitable for incremental prefill
/// after an existing KV cache (any `pos_base`).
pub(crate) async fn prefill_matvec_loop_topk(
    model: &Model, prompt: &[u32], pos_base: u32, k: u32,
) -> Result<(Vec<f32>, Vec<u32>)> {
    let Some((&last, rest)) = prompt.split_last() else {
        return Err(PotError::PrefillTooLarge { n: 0, max: model.m_max });
    };
    // For the (n - 1) non-last tokens we just need to fill the KV cache; we
    // can encode them all back-to-back into a SINGLE command buffer + submit,
    // then run the final token through `step_matvec_topk` (which does the
    // CPU readback for sampling).
    if !rest.is_empty() {
        // One up-front write covers every non-last token's input slot. The
        // embed shader for step `i` reads `sample[i]`. The topk_reduce output
        // for steps 0..last-1 is routed to a scratch slot beyond the prompt
        // region (sample buffer is 1024 u32 slots / 4 KB; M_MAX = 512, so
        // `TOPK_SCRATCH_BASE = 768` is well clear of any prompt tokens).
        const TOPK_SCRATCH_BASE: u32 = 768;
        model.queue.write_buffer(&model.buffers.sample, 0, bytemuck::cast_slice(rest));
        let mut se = StepEncoder::new(model);
        for t in 0..rest.len() {
            encode_step_matvec(
                &mut se, &model.cfg,
                /*sample_in=*/ t as u32,
                /*topk_out_u32_base=*/ TOPK_SCRATCH_BASE,
                /*pos=*/ pos_base + t as u32,
                /*k=*/ 1,
            );
        }
        let cb = se.finish();
        model.queue.submit(Some(cb));
    }
    step_matvec_topk(model, last, pos_base + rest.len() as u32, k).await
}

// ---------- matmul (batched prefill) ---------------------------------------

/// Batched matmul prefill of `prompt` starting from KV-cache position 0.
/// Advances pos from 0 to `prompt.len()`. Returns top-K candidates from the
/// last token's logits. Requires `pos_base == 0` (the matmul attention shader
/// assumes a fresh cache).
pub(crate) async fn prefill_matmul_topk(
    model: &Model, prompt: &[u32], pos_base: u32, k: u32,
) -> Result<(Vec<f32>, Vec<u32>)> {
    if pos_base != 0 {
        // Caller must use prefill_matvec_loop_topk for incremental prefill.
        return Err(PotError::Config(
            "prefill_matmul_topk requires pos_base == 0; use prefill_one_at_a_time for incremental prefill",
        ));
    }
    let m = prompt.len() as u32;
    if m == 0 || m > model.m_max {
        return Err(PotError::PrefillTooLarge { n: m, max: model.m_max });
    }
    let cfg = &model.cfg;
    let ot = &model.output_tensors;
    let k = k.min(TOPK_MAX).max(1);

    // ---- All phases (embed → per-layer transformer → final norm/LM-head/topk
    //      → readback copy) into ONE command buffer / ONE submit. ------------
    model.queue.write_buffer(&model.buffers.sample, 0, bytemuck::cast_slice(prompt));
    let mut se = StepEncoder::new(model);

    // Phase 1: embed all M tokens in a SINGLE dispatch (m, 1, 1). The shader
    // reads its row from sample[sample_offset + wg.x] and writes to row wg.x
    // of the output activation buffer.
    {
        let p = EmbedParams {
            k: cfg.n_embd,
            d_offset: ot.token_embd_d,
            qs_offset: ot.token_embd_qs,
            output_offset: model.act_layout.x,
            sample_offset: 0,
            ..Default::default()
        };
        let off = se.uniforms.alloc(&p);
        let mut pass = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("prefill_embed"), timestamp_writes: None });
        pass.set_pipeline(&model.pipes.embed);
        pass.set_bind_group(0, &model.cached.embed, &[off]);
        pass.dispatch_workgroups(m, 1, 1);
    }

    // Phase 2: per-layer transformer. `layer_step_matmul` opens its own
    // compute passes for each kernel; that's fine — the KV-cache copies it
    // emits are transfer commands and are legal between compute passes inside
    // the same command encoder.
    for il in 0..cfg.n_layer {
        layer_step_matmul(model, cfg, &mut se, il, m);
    }

    // Phase 3: final RMSNorm + LM head + topk_reduce on the LAST token.
    rms_norm(model, cfg, &mut se, m, cfg.n_embd,
             model.act_layout.x, model.act_layout.x_norm, ot.output_norm_off);

    let last_input_offset = model.act_layout.x_norm + (m - 1) * cfg.n_embd;
    matvec_q1_0(model, cfg, &mut se,
                cfg.n_embd, cfg.n_vocab,
                WeightSet::Embed, ot.token_embd_d, ot.token_embd_qs,
                last_input_offset, model.act_layout.logits, false);
    {
        let p = TopKParams {
            n: cfg.n_vocab, in_offset: model.act_layout.logits,
            out_offset: 0, k,
        };
        let off = se.alloc_uniform(&p);
        let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("topk_reduce"), timestamp_writes: None });
        cp.set_pipeline(&model.pipes.topk_reduce);
        cp.set_bind_group(0, &model.cached.topk_reduce, &[off]);
        cp.dispatch_workgroups(1, 1, 1);
    }

    // Phase 4: append readback copy so it ships with this same command buffer.
    se.copy_sample_to_readback((k as u64) * 8);

    let cb = se.finish();
    model.queue.submit(Some(cb));

    await_topk_readback(model, k).await
}

fn layer_step_matmul(model: &Model, cfg: &Config, se: &mut StepEncoder, il: u32, m: u32) {
    let pos_base = 0u32;
    let lt = &model.layer_tensors[il as usize];

    rms_norm(model, cfg, se, m, cfg.n_embd,
             model.act_layout.x, model.act_layout.x_norm, lt.attn_norm_off);

    let (a_d_off, a_qs_off) = quantize_act(model, cfg, se, cfg.n_embd, m, model.act_layout.x_norm);

    matmul_q1_0(model, cfg, se, cfg.n_embd, cfg.q_dim, m,
                WeightSet::Attn, lt.wq.0, lt.wq.1,
                a_d_off, a_qs_off, model.act_layout.q, false);
    matmul_q1_0(model, cfg, se, cfg.n_embd, cfg.kv_dim, m,
                WeightSet::Attn, lt.wk.0, lt.wk.1,
                a_d_off, a_qs_off, model.act_layout.k_cur, false);
    matmul_q1_0(model, cfg, se, cfg.n_embd, cfg.kv_dim, m,
                WeightSet::Attn, lt.wv.0, lt.wv.1,
                a_d_off, a_qs_off, model.act_layout.v_cur, false);

    rms_norm(model, cfg, se, m * cfg.n_head, cfg.head_dim,
             model.act_layout.q, model.act_layout.q, lt.attn_q_norm_off);
    rms_norm(model, cfg, se, m * cfg.n_kv_head, cfg.head_dim,
             model.act_layout.k_cur, model.act_layout.k_cur, lt.attn_k_norm_off);

    rope(model, cfg, se, model.act_layout.q,     m, cfg.n_head,    pos_base);
    rope(model, cfg, se, model.act_layout.k_cur, m, cfg.n_kv_head, pos_base);

    let cache_row_bytes = cfg.kv_dim as u64 * ACT_ELEM_BYTES;
    let layer_offset_bytes = (il as u64) * (model.max_seq as u64) * cache_row_bytes;
    let total_bytes = (m as u64) * cache_row_bytes;
    se.encoder.copy_buffer_to_buffer(&model.buffers.act, (model.act_layout.k_cur as u64) * ACT_ELEM_BYTES,
                                     &model.buffers.kv_k, layer_offset_bytes, total_bytes);
    se.encoder.copy_buffer_to_buffer(&model.buffers.act, (model.act_layout.v_cur as u64) * ACT_ELEM_BYTES,
                                     &model.buffers.kv_v, layer_offset_bytes, total_bytes);

    {
        let p = AttnParams {
            head_dim: cfg.head_dim, n_head: cfg.n_head, n_kv_head: cfg.n_kv_head,
            pos: 0,
            kv_stride: cfg.kv_dim,
            q_offset: model.act_layout.q,
            k_cache_offset: ((layer_offset_bytes / ACT_ELEM_BYTES) as u32),
            v_cache_offset: ((layer_offset_bytes / ACT_ELEM_BYTES) as u32),
            out_offset: model.act_layout.attn_out,
            scale: 1.0 / (cfg.head_dim as f32).sqrt(),
            m_tokens: m, is_prefill: 1,
        };
        let off = se.alloc_uniform(&p);
        let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("attn"), timestamp_writes: None });
        cp.set_pipeline(&model.pipes.attention);
        cp.set_bind_group(0, &model.cached.attn, &[off]);
        cp.dispatch_workgroups(cfg.n_head, m, 1);
    }

    let (a_d2, a_qs2) = quantize_act(model, cfg, se, cfg.q_dim, m, model.act_layout.attn_out);
    matmul_q1_0(model, cfg, se, cfg.q_dim, cfg.n_embd, m,
                WeightSet::Attn, lt.wo.0, lt.wo.1,
                a_d2, a_qs2, model.act_layout.x, true);

    rms_norm(model, cfg, se, m, cfg.n_embd,
             model.act_layout.x, model.act_layout.x_norm, lt.ffn_norm_off);

    let (a_d3, a_qs3) = quantize_act(model, cfg, se, cfg.n_embd, m, model.act_layout.x_norm);
    matmul_q1_0(model, cfg, se, cfg.n_embd, cfg.n_ff, m,
                WeightSet::FfnGU, lt.wg.0, lt.wg.1,
                a_d3, a_qs3, model.act_layout.gate, false);
    matmul_q1_0(model, cfg, se, cfg.n_embd, cfg.n_ff, m,
                WeightSet::FfnGU, lt.wu.0, lt.wu.1,
                a_d3, a_qs3, model.act_layout.up, false);

    silu_mul(model, se, cfg.n_ff, m, model.act_layout.gate, model.act_layout.up, model.act_layout.ffn_in);

    let (a_d4, a_qs4) = quantize_act(model, cfg, se, cfg.n_ff, m, model.act_layout.ffn_in);
    matmul_q1_0(model, cfg, se, cfg.n_ff, cfg.n_embd, m,
                WeightSet::FfnD, lt.wd.0, lt.wd.1,
                a_d4, a_qs4, model.act_layout.x, true);
}

// ---------- shader wrappers --------------------------------------------------

fn rms_norm(model: &Model, cfg: &Config, se: &mut StepEncoder,
            n_groups: u32, group_size: u32, in_off: u32, out_off: u32, w_off: u32) {
    let p = RmsNormParams {
        group_size, n_groups,
        input_offset: in_off, output_offset: out_off, weight_offset: w_off,
        eps: cfg.rms_eps,
        ..Default::default()
    };
    let off = se.alloc_uniform(&p);
    let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("rms_norm"), timestamp_writes: None });
    cp.set_pipeline(&model.pipes.rms_norm);
    cp.set_bind_group(0, &model.cached.rms_norm, &[off]);
    cp.dispatch_workgroups(n_groups, 1, 1);
}

fn rope(model: &Model, cfg: &Config, se: &mut StepEncoder,
        data_off: u32, n_tokens: u32, n_heads: u32, pos_base: u32) {
    let p = RopeParams {
        head_dim: cfg.head_dim, n_heads, n_tokens, pos_base,
        data_offset: data_off,
        ..Default::default()
    };
    let off = se.alloc_uniform(&p);
    let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("rope"), timestamp_writes: None });
    cp.set_pipeline(&model.pipes.rope_neox);
    cp.set_bind_group(0, &model.cached.rope, &[off]);
    cp.dispatch_workgroups(n_tokens, n_heads, 1);
}

#[cfg(feature = "bench-internals")]
fn matvec_q1_0_fused_pass(
    model: &Model, se: &mut StepEncoder,
    k: u32, input_offset: u32, weights: WeightSet,
    ranges: &[(u32, u32, u32, u32)],
) {
    debug_assert!(ranges.len() == 2 || ranges.len() == 3);
    for (_, _, n, _) in ranges { debug_assert!(n % 8 == 0); }
    let r = |i: usize| ranges.get(i).copied().unwrap_or((0, 0, 0, 0));
    let (d0, qs0, n0, o0) = r(0);
    let (d1, qs1, n1, o1) = r(1);
    let (d2, qs2, n2, o2) = r(2);
    let n_total = n0 + n1 + n2;
    const ROWS_PER_WG: u32 = 8;
    let n_wg = (n_total + ROWS_PER_WG - 1) / ROWS_PER_WG;
    let dispatch_x = n_wg.min(65535);
    let dispatch_y = (n_wg + dispatch_x - 1) / dispatch_x;
    let p = MatvecFusedParams {
        k, n_total, input_offset, dispatch_x_dim: dispatch_x,
        d_offset_0: d0, qs_offset_0: qs0, n_0: n0, output_offset_0: o0,
        d_offset_1: d1, qs_offset_1: qs1, n_1: n1, output_offset_1: o1,
        d_offset_2: d2, qs_offset_2: qs2, n_2: n2, output_offset_2: o2,
    };
    let off = se.alloc_uniform(&p);
    let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("matvec_fused"), timestamp_writes: None });
    cp.set_pipeline(&model.pipes.matvec_fused);
    cp.set_bind_group(0, matvec_bg(model, weights), &[off]);
    cp.dispatch_workgroups(dispatch_x, dispatch_y, 1);
}

fn matvec_q1_0(model: &Model, _cfg: &Config, se: &mut StepEncoder,
               k: u32, n: u32,
               weights: WeightSet, w_d: u32, w_qs: u32,
               in_off: u32, out_off: u32, accumulate: bool) {
    const ROWS_PER_WG: u32 = 8;
    let n_wg = (n + ROWS_PER_WG - 1) / ROWS_PER_WG;
    let dispatch_x = n_wg.min(65535);
    let dispatch_y = (n_wg + dispatch_x - 1) / dispatch_x;
    let p = MatvecParams {
        k, n,
        d_offset: w_d, qs_offset: w_qs,
        input_offset: in_off, output_offset: out_off,
        accumulate: if accumulate { 1 } else { 0 },
        dispatch_x_dim: dispatch_x,
    };
    let off = se.alloc_uniform(&p);
    let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("matvec"), timestamp_writes: None });
    cp.set_pipeline(&model.pipes.matvec);
    cp.set_bind_group(0, matvec_bg(model, weights), &[off]);
    cp.dispatch_workgroups(dispatch_x, dispatch_y, 1);
}

fn quantize_act(model: &Model, _cfg: &Config, se: &mut StepEncoder,
                k: u32, m: u32, in_off_f32: u32) -> (u32, u32) {
    let nb_q8 = k / 32;
    let d_off  = 0u32;
    let qs_off = m * nb_q8 * 4;
    let total = m * nb_q8;
    let dispatch_x = total.min(65535);
    let dispatch_y = (total + dispatch_x - 1) / dispatch_x;
    let p = QuantParams {
        k, m, input_offset: in_off_f32,
        d_offset: d_off, qs_offset: qs_off,
        dispatch_x_dim: dispatch_x,
        ..Default::default()
    };
    let off = se.alloc_uniform(&p);
    let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("quantize"), timestamp_writes: None });
    cp.set_pipeline(&model.pipes.quantize);
    cp.set_bind_group(0, &model.cached.quantize, &[off]);
    cp.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    (d_off, qs_off)
}

fn silu_mul(model: &Model, se: &mut StepEncoder,
            n: u32, m: u32, gate_off: u32, up_off: u32, out_off: u32) {
    let total = n * m;
    let groups = (total + 63) / 64;
    let dispatch_x = groups.min(65535);
    let dispatch_y = (groups + dispatch_x - 1) / dispatch_x;
    let p = SiluMulParams {
        n, m, gate_offset: gate_off, up_offset: up_off, out_offset: out_off,
        dispatch_x_count: dispatch_x * 64,
        ..Default::default()
    };
    let off = se.alloc_uniform(&p);
    let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("silu_mul"), timestamp_writes: None });
    cp.set_pipeline(&model.pipes.silu_mul);
    cp.set_bind_group(0, &model.cached.silu_mul, &[off]);
    cp.dispatch_workgroups(dispatch_x, dispatch_y, 1);
}

fn matmul_q1_0(model: &Model, _cfg: &Config, se: &mut StepEncoder,
               k: u32, n: u32, m: u32,
               weights: WeightSet, w_d: u32, w_qs: u32,
               a_d: u32, a_qs: u32, out_off: u32, accumulate: bool) {
    let p = MatmulParams {
        k, n, m,
        w_d_offset: w_d, w_qs_offset: w_qs,
        a_d_offset: a_d, a_qs_offset: a_qs,
        out_offset: out_off,
        accumulate: if accumulate { 1 } else { 0 },
        ..Default::default()
    };
    let off = se.alloc_uniform(&p);
    let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("matmul"), timestamp_writes: None });
    cp.set_pipeline(&model.pipes.matmul);
    cp.set_bind_group(0, matmul_bg(model, weights), &[off]);
    cp.dispatch_workgroups((n + 63) / 64, (m + 63) / 64, 1);
}

// =========================================================================
// Bench / microbench (CLI utilities; gated behind the bench-internals feature
// so they don't pollute the public surface).
// =========================================================================

#[cfg(feature = "bench-internals")]
pub mod bench_internals {
    use super::*;

    /// Match `llama-bench` style: pp{n} measures batched-prefill throughput;
    /// tg{n} measures single-token generation throughput.
    pub async fn bench(model: &Model, pp_n: u32, tg_n: u32, repeats: u32) -> Result<()> {
        let cfg = &model.cfg;
        eprintln!("--- bench: pp={pp_n}, tg={tg_n}, repeats={repeats} (after 1 warmup) ---");

        let prompt: Vec<u32> = (0..pp_n).map(|i| (i % (cfg.n_vocab - 1)) + 1).collect();

        // ----- warm up -----
        let _ = prefill_matmul_topk(model, &prompt[..pp_n.min(model.m_max) as usize], 0, 1).await?;
        step_matvec_no_sample(model, 1u32, 0);
        model.device.poll(wgpu::PollType::wait_indefinitely()).expect("device.poll failed");

        // ----- pp{pp_n} -----
        let mut pp_times = Vec::with_capacity(repeats as usize);
        for _ in 0..repeats {
            let t = std::time::Instant::now();
            let _ = prefill_matmul_topk(model, &prompt, 0, 1).await?;
            model.device.poll(wgpu::PollType::wait_indefinitely()).expect("device.poll failed");
            pp_times.push(t.elapsed().as_secs_f32());
        }
        let pp_mean = pp_times.iter().sum::<f32>() / pp_times.len() as f32;
        let pp_std = stddev(&pp_times, pp_mean);
        let pp_t_s_mean = pp_n as f32 / pp_mean;
        let pp_t_s_std = pp_t_s_mean * (pp_std / pp_mean);

        // ----- tg{tg_n} (from empty KV) -----
        let mut tg_times = Vec::with_capacity(repeats as usize);
        for _ in 0..repeats {
            let t = std::time::Instant::now();
            for pos in 0..tg_n {
                let (_, _) = step_matvec_topk(model, 1u32, pos, 1).await?;
            }
            model.device.poll(wgpu::PollType::wait_indefinitely()).expect("device.poll failed");
            tg_times.push(t.elapsed().as_secs_f32());
        }
        let tg_mean = tg_times.iter().sum::<f32>() / tg_times.len() as f32;
        let tg_std = stddev(&tg_times, tg_mean);
        let tg_t_s_mean = tg_n as f32 / tg_mean;
        let tg_t_s_std = tg_t_s_mean * (tg_std / tg_mean);

        println!();
        println!("| backend            |          test |               t/s |");
        println!("| ------------------ | ------------- | ----------------: |");
        println!("| bonsai-pot        |        pp{pp_n:<3} | {pp_t_s_mean:>9.2} ± {pp_t_s_std:>5.2} |");
        println!("| bonsai-pot        |        tg{tg_n:<3} | {tg_t_s_mean:>9.2} ± {tg_t_s_std:>5.2} |");
        Ok(())
    }

    fn stddev(xs: &[f32], mean: f32) -> f32 {
        if xs.len() < 2 { return 0.0; }
        let var: f32 = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / (xs.len() - 1) as f32;
        var.sqrt()
    }

    pub async fn microbench_tg(model: &Model, repeats: u32) {
        let cfg = &model.cfg;
        let n_per_cb: u32 = 200;
        let attn_pos: u32 = 64;

        eprintln!("--- microbench tg (N={n_per_cb}/CB, repeats={repeats}) ---");

        type DispatchFn<'a> = Box<dyn Fn(&Model, &Config, &mut StepEncoder) + 'a>;
        let mut entries: Vec<(String, u32, DispatchFn)> = Vec::new();

        // (label, k, n, calls/step, weight_set, w_d_offset, w_qs_offset)
        let lt0 = &model.layer_tensors[0];
        let ot = &model.output_tensors;
        let matvec_shapes: &[(&str, u32, u32, u32, WeightSet, u32, u32)] = &[
            ("matvec Wo         K=4096 N=2560 ", cfg.q_dim,  cfg.n_embd, 36, WeightSet::Attn,  lt0.wo.0,        lt0.wo.1),
            ("matvec Wdown      K=9728 N=2560 ", cfg.n_ff,   cfg.n_embd, 36, WeightSet::FfnD,  lt0.wd.0,        lt0.wd.1),
            ("matvec LM_head    K=2560 N=152K ", cfg.n_embd, cfg.n_vocab, 1, WeightSet::Embed, ot.token_embd_d, ot.token_embd_qs),
        ];
        for &(label, k, n, calls, ws, w_d, w_qs) in matvec_shapes {
            entries.push((label.to_string(), calls, Box::new(move |model, cfg, se| {
                matvec_q1_0(model, cfg, se, k, n, ws, w_d, w_qs,
                            model.act_layout.x_norm, model.act_layout.x, false);
            })));
        }

        entries.push(("matvec QKV  fused N=4608        ".to_string(), 36, Box::new(move |model, cfg, se| {
            let lt = &model.layer_tensors[0];
            matvec_q1_0_fused_pass(model, se, cfg.n_embd, model.act_layout.x_norm,
                              WeightSet::Attn, &[
                (lt.wq.0, lt.wq.1, cfg.q_dim,  model.act_layout.q),
                (lt.wk.0, lt.wk.1, cfg.kv_dim, model.act_layout.k_cur),
                (lt.wv.0, lt.wv.1, cfg.kv_dim, model.act_layout.v_cur),
            ]);
        })));
        entries.push(("matvec gate_up fused N=19456    ".to_string(), 36, Box::new(move |model, cfg, se| {
            let lt = &model.layer_tensors[0];
            matvec_q1_0_fused_pass(model, se, cfg.n_embd, model.act_layout.x_norm,
                              WeightSet::FfnGU, &[
                (lt.wg.0, lt.wg.1, cfg.n_ff, model.act_layout.gate),
                (lt.wu.0, lt.wu.1, cfg.n_ff, model.act_layout.up),
            ]);
        })));

        // (label, n_groups, group_size, calls/step, norm_off)
        let rms_shapes: &[(&str, u32, u32, u32, u32)] = &[
            ("rms_norm  ng=1  gs=2560        ", 1,             cfg.n_embd,    36 + 36 + 1, lt0.attn_norm_off),
            ("rms_norm  ng=32 gs=128         ", cfg.n_head,    cfg.head_dim,  36,          lt0.attn_q_norm_off),
            ("rms_norm  ng=8  gs=128         ", cfg.n_kv_head, cfg.head_dim,  36,          lt0.attn_k_norm_off),
        ];
        for &(label, ng, gs, calls, norm_off) in rms_shapes {
            entries.push((label.to_string(), calls, Box::new(move |model, cfg, se| {
                rms_norm(model, cfg, se, ng, gs,
                         model.act_layout.x_norm, model.act_layout.x_norm, norm_off);
            })));
        }

        entries.push(("rope     n_heads=32             ".to_string(), 36, Box::new(move |model, cfg, se| {
            rope(model, cfg, se, model.act_layout.q,     1, cfg.n_head,    0);
        })));
        entries.push(("rope     n_heads=8              ".to_string(), 36, Box::new(move |model, cfg, se| {
            rope(model, cfg, se, model.act_layout.k_cur, 1, cfg.n_kv_head, 0);
        })));

        entries.push(("silu_mul n=9728                 ".to_string(), 36, Box::new(move |model, cfg, se| {
            silu_mul(model, se, cfg.n_ff, 1, model.act_layout.gate, model.act_layout.up, model.act_layout.ffn_in);
        })));

        entries.push((format!("attention pos={attn_pos:<3}                "), 36, Box::new(move |model, cfg, se| {
            let layer_offset = 0u64;
            let p = AttnParams {
                head_dim: cfg.head_dim, n_head: cfg.n_head, n_kv_head: cfg.n_kv_head,
                pos: attn_pos,
                kv_stride: cfg.kv_dim,
                q_offset: model.act_layout.q,
                k_cache_offset: ((layer_offset / ACT_ELEM_BYTES) as u32),
                v_cache_offset: ((layer_offset / ACT_ELEM_BYTES) as u32),
                out_offset: model.act_layout.attn_out,
                scale: 1.0 / (cfg.head_dim as f32).sqrt(),
                m_tokens: 1, is_prefill: 0,
            };
            let off = se.alloc_uniform(&p);
            let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("attn"), timestamp_writes: None });
            cp.set_pipeline(&model.pipes.attention);
            cp.set_bind_group(0, &model.cached.attn, &[off]);
            cp.dispatch_workgroups(cfg.n_head, 1, 1);
        })));

        entries.push(("embed                           ".to_string(), 1, Box::new(move |model, cfg, se| {
            let ot = &model.output_tensors;
            let p = EmbedParams {
                k: cfg.n_embd,
                d_offset: ot.token_embd_d,
                qs_offset: ot.token_embd_qs,
                output_offset: model.act_layout.x,
                sample_offset: 0,
                ..Default::default()
            };
            let off = se.alloc_uniform(&p);
            let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("embed"), timestamp_writes: None });
            cp.set_pipeline(&model.pipes.embed);
            cp.set_bind_group(0, &model.cached.embed, &[off]);
            cp.dispatch_workgroups(1, 1, 1);
        })));

        entries.push(("topk_reduce n=n_vocab           ".to_string(), 1, Box::new(move |model, _cfg, se| {
            let p = TopKParams { n: model.cfg.n_vocab, in_offset: model.act_layout.logits, out_offset: 0, k: 1 };
            let off = se.alloc_uniform(&p);
            let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("topk_reduce"), timestamp_writes: None });
            cp.set_pipeline(&model.pipes.topk_reduce);
            cp.set_bind_group(0, &model.cached.topk_reduce, &[off]);
            cp.dispatch_workgroups(1, 1, 1);
        })));

        let kv_row_bytes = cfg.kv_dim as u64 * ACT_ELEM_BYTES;

        // Warmup
        {
            let mut se = StepEncoder::new(model);
            for (_l, _c, f) in entries.iter() { f(model, cfg, &mut se); }
            se.encoder.copy_buffer_to_buffer(&model.buffers.act, 0, &model.buffers.kv_k, 0, kv_row_bytes);
            let cb = se.finish();
            model.queue.submit(Some(cb));
            model.device.poll(wgpu::PollType::wait_indefinitely()).expect("device.poll failed");
        }

        let mut breakdown: Vec<(String, u32, f32)> = Vec::new();
        for (label, calls, build) in entries.iter() {
            let mut samples = Vec::with_capacity(repeats as usize);
            for _ in 0..repeats {
                let t = std::time::Instant::now();
                let mut se = StepEncoder::new(model);
                for _ in 0..n_per_cb { build(model, cfg, &mut se); }
                let cb = se.finish();
                model.queue.submit(Some(cb));
                model.device.poll(wgpu::PollType::wait_indefinitely()).expect("device.poll failed");
                samples.push(t.elapsed().as_secs_f32());
            }
            let mean = samples.iter().sum::<f32>() / samples.len() as f32;
            let per_call_us = mean / n_per_cb as f32 * 1e6;
            breakdown.push((label.clone(), *calls, per_call_us));
        }

        {
            let mut samples = Vec::with_capacity(repeats as usize);
            for _ in 0..repeats {
                let t = std::time::Instant::now();
                let mut enc = model.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("kvcopy") });
                for _ in 0..n_per_cb {
                    enc.copy_buffer_to_buffer(&model.buffers.act, 0, &model.buffers.kv_k, 0, kv_row_bytes);
                    enc.copy_buffer_to_buffer(&model.buffers.act, 0, &model.buffers.kv_v, 0, kv_row_bytes);
                }
                model.queue.submit(Some(enc.finish()));
                model.device.poll(wgpu::PollType::wait_indefinitely()).expect("device.poll failed");
                samples.push(t.elapsed().as_secs_f32());
            }
            let mean = samples.iter().sum::<f32>() / samples.len() as f32;
            let per_pair_us = mean / n_per_cb as f32 * 1e6;
            breakdown.push(("kv-cache copy (K+V)             ".to_string(), 36, per_pair_us));
        }

        println!();
        println!("| kernel                              | calls/step | us/call | ms/step | %step |");
        println!("|-------------------------------------|-----------:|--------:|--------:|------:|");
        let total_ms: f32 = breakdown.iter().map(|(_, c, us)| (*c as f32) * (*us) / 1000.0).sum();
        let mut sorted = breakdown.clone();
        sorted.sort_by(|a, b| {
            let ma = a.1 as f32 * a.2;
            let mb = b.1 as f32 * b.2;
            mb.partial_cmp(&ma).unwrap()
        });
        for (label, calls, per_us) in &sorted {
            let ms_step = (*calls as f32) * (*per_us) / 1000.0;
            let pct = 100.0 * ms_step / total_ms;
            println!("| {label:<35} | {calls:>10} | {per_us:>7.2} | {ms_step:>7.3} | {pct:>5.1} |");
        }
        println!("|-------------------------------------|-----------:|--------:|--------:|------:|");
        println!("| TOTAL (sum of isolated)             |            |         | {total_ms:>7.3} |       |");
        println!("expected tg t/s if isolated sum was reality: {:.1}", 1000.0 / total_ms);
    }
}
