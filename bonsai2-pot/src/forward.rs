//! Forward pass. Every step is encoded into one compute pass; each kernel is a
//! dispatch over the universal bind group of the session (binding 0 selects
//! the weight buffer).
//!
//! Per layer (single token):
//! - Gated-DeltaNet: `norm_rot_q8` → matvec(qkv, z) → `gdn_gates` → `gdn_conv`
//!   → `gdn_recurrence` → `gdn_out_rot_q8` → `matvec(ssm_out`, += x)
//! - full attention: `norm_rot_q8` → matvec(q|gate, k, v) → `attn_qk_norm_rope_kv`
//!   → `attn_split` → `attn_out_rot_q8` → `matvec(attn_output`, += x)
//! - FFN: `norm_rot_q8` → matvec(gate, up) → `silu_mul_rot_q8` → matvec(down, += x)
//!
//! Sampling: `topk_partial` + `topk_merge` write the top-K logits and ids to
//! `io[0..2K]`; the host applies temperature / top-p / min-p.

#[cfg(feature = "bench-internals")]
use alloc::string::String;
use alloc::vec::Vec;
#[cfg(all(feature = "bench-internals", not(feature = "ci")))]
use core::cell::Cell;

use bytemuck::Pod;
use wgpu::PollType;

use crate::config::{AttnTensors, GdnTensors, LayerKind, LayerTensors};
use crate::error::{PotError, Result};
use crate::model::{ATTN_MAX_CHUNKS, ATTN_MIN_CHUNK, M_MAX, TOPK_MAX, TOPK_NUM_PARTIAL_WG};
use crate::params::{
    AttnOutParams, AttnPrefillParams, AttnQkParams, AttnSplitParams, EmbedParams, GdnConvParams,
    GdnGatesParams, GdnOutParams, GdnRecurrenceParams, MatmulParams, MatvecParams, NormRotParams,
    SiluMulParams, TopKMergeParams, TopKPartialParams,
};
use crate::session::Session;

// ---------- step markers -----------------------------------------------------

/// Instrumentation hook for the encoders. Production uses [`NoMarker`].
pub trait StepMarker {
    /// Populate the pass descriptor before `begin_compute_pass`.
    fn setup_desc<'a>(&'a self, desc: &mut wgpu::ComputePassDescriptor<'a>);
    /// Called after each dispatch with the kernel's label.
    fn mark(&mut self, pass: &mut wgpu::ComputePass<'_>, label: &'static str);
}

pub struct NoMarker;
impl StepMarker for NoMarker {
    #[inline(always)]
    fn setup_desc<'a>(&'a self, _desc: &mut wgpu::ComputePassDescriptor<'a>) {}
    #[inline(always)]
    fn mark(&mut self, _pass: &mut wgpu::ComputePass<'_>, _label: &'static str) {}
}

// ---------- encoder plumbing -------------------------------------------------

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

    /// Append the `io → readback` copy of the top-K results.
    pub fn copy_topk_to_readback(&mut self, bytes: u64) {
        self.encoder.copy_buffer_to_buffer(
            &self.session.state.io,
            0,
            &self.session.state.readback,
            0,
            bytes,
        );
    }

    /// Schedule a `MAP_READ` mapping of the readback buffer on submission.
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

/// Record `staging[0..n] → io[dst..]` and schedule the staging remap; pair
/// with [`commit_sample_upload`] before submitting.
pub fn encode_sample_upload(
    session: &Session<'_>,
    encoder: &mut wgpu::CommandEncoder,
    dst_off: u64,
    n_bytes: u64,
) {
    encoder.copy_buffer_to_buffer(
        &session.state.staging,
        0,
        &session.state.io,
        dst_off,
        n_bytes,
    );
    encoder.map_buffer_on_submit(&session.state.staging, wgpu::MapMode::Write, 0.., |_| ());
}

/// Write `data` into the mapped staging buffer and unmap it.
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

/// Non-pipelined upload: encode + commit at once.
pub fn upload_sample(
    session: &Session<'_>,
    encoder: &mut wgpu::CommandEncoder,
    dst_off: u64,
    data: &[u8],
) {
    encode_sample_upload(session, encoder, dst_off, data.len() as u64);
    commit_sample_upload(session, data);
}

/// One dispatch over the session bind group of weight window `win`.
fn dispatch<P: Pod>(
    s: &Session<'_>,
    pass: &mut wgpu::ComputePass<'_>,
    pipe: &wgpu::ComputePipeline,
    win: usize,
    params: &P,
    groups: (u32, u32, u32),
) {
    pass.set_pipeline(pipe);
    pass.set_bind_group(0, &s.state.bind_groups[win], &[]);
    pass.set_immediates(0, bytemuck::bytes_of(params));
    pass.dispatch_workgroups(groups.0, groups.1, groups.2);
}

// ---------- kernel helpers -----------------------------------------------------

/// Output rows per `matvec_ptq1_0` workgroup (`ROWS` in the shader).
const MATVEC_ROWS: u32 = 16;

/// Q8 activation section offsets for row width `k`.
const fn q8_offsets(s: &Session<'_>) -> (u32, u32, u32) {
    let q = s.model.q8;
    (q.qs, q.d, q.s)
}

fn norm_rot(
    s: &Session<'_>,
    pass: &mut wgpu::ComputePass<'_>,
    m: u32,
    in_off: u32,
    norm_off: u32,
    xn_off: Option<u32>,
) {
    let cfg = &s.model.cfg;
    let (qs, d, sm) = q8_offsets(s);
    let p = NormRotParams {
        k: cfg.n_embd,
        in_off,
        in_stride: cfg.n_embd,
        norm_off,
        signs_off: s.model.output.signs_embd,
        xn_off: xn_off.unwrap_or(0),
        q8_qs: qs,
        q8_d: d,
        q8_s: sm,
        eps: cfg.rms_eps,
        flags: u32::from(xn_off.is_some()),
    };
    dispatch(
        s,
        pass,
        &s.model.pipes.norm_rot_q8,
        s.model.win_attn,
        &p,
        (cfg.n_embd / 1024, m, 1),
    );
}

/// Matvec of the Q8 activation row 0 against up to three weight segments.
/// `segs`: (weight word offset, rows, act output offset).
fn matvec(
    s: &Session<'_>,
    pass: &mut wgpu::ComputePass<'_>,
    win: usize,
    k: u32,
    segs: &[(u32, u32, u32)],
    accumulate: bool,
) {
    debug_assert!(!segs.is_empty() && segs.len() <= 3);
    let (qs, d, sm) = q8_offsets(s);
    let get = |i: usize| segs.get(i).copied().unwrap_or((0, 0, 0));
    let (s0, s1, s2) = (get(0), get(1), get(2));
    let total = s0.1 + s1.1 + s2.1;
    debug_assert!(segs.iter().all(|sg| sg.1 % MATVEC_ROWS == 0));
    let p = MatvecParams {
        nb: k / 128,
        q8_qs: qs,
        q8_d: d,
        q8_s: sm,
        seg1: s0.1,
        seg2: s0.1 + s1.1,
        w0: s0.0,
        w1: s1.0,
        w2: s2.0,
        out0: s0.2,
        out1: s1.2,
        out2: s2.2,
        flags: u32::from(accumulate),
    };
    dispatch(
        s,
        pass,
        &s.model.pipes.matvec,
        win,
        &p,
        (total / MATVEC_ROWS, 1, 1),
    );
}

/// Projection of the `m` Q8 activation rows: matvec for one token, the tiled
/// matmul for a batch. Segment `s` writes rows of width `rows_s`.
fn proj(
    s: &Session<'_>,
    pass: &mut wgpu::ComputePass<'_>,
    win: usize,
    k: u32,
    segs: &[(u32, u32, u32)],
    accumulate: bool,
    m: u32,
) {
    if m == 1 {
        matvec(s, pass, win, k, segs, accumulate);
        return;
    }
    debug_assert!(segs.iter().all(|sg| sg.1 % 64 == 0));
    let (qs, d, sm) = q8_offsets(s);
    let get = |i: usize| segs.get(i).copied().unwrap_or((0, 0, 0));
    let (s0, s1, s2) = (get(0), get(1), get(2));
    let total = s0.1 + s1.1 + s2.1;
    let p = MatmulParams {
        nb: k / 128,
        m_tokens: m,
        q8_qs: qs,
        q8_d: d,
        q8_s: sm,
        seg1: s0.1,
        seg2: s0.1 + s1.1,
        w0: s0.0,
        w1: s1.0,
        w2: s2.0,
        out0: s0.2,
        out1: s1.2,
        out2: s2.2,
        total_rows: total,
        flags: u32::from(accumulate),
    };
    if let Some((cm, tm)) = s.model.pipes.matmul_cm.as_ref() {
        dispatch(s, pass, cm, win, &p, (total / 64, m.div_ceil(*tm), 1));
    } else {
        dispatch(
            s,
            pass,
            &s.model.pipes.matmul,
            win,
            &p,
            (total / 64, m.div_ceil(64), 1),
        );
    }
}

fn ffn_block<M: StepMarker>(
    s: &Session<'_>,
    pass: &mut wgpu::ComputePass<'_>,
    lt: &LayerTensors,
    m: u32,
    marker: &mut M,
) {
    let cfg = &s.model.cfg;
    let a = &s.model.act;
    norm_rot(s, pass, m, a.x, lt.post_norm, None);
    marker.mark(pass, "ffn_norm_rot");
    proj(
        s,
        pass,
        lt.gu_win,
        cfg.n_embd,
        &[(lt.ffn_gate, cfg.n_ff, a.gate), (lt.ffn_up, cfg.n_ff, a.up)],
        false,
        m,
    );
    marker.mark(pass, "ffn_gate_up");
    let (qs, d, sm) = q8_offsets(s);
    let p = SiluMulParams {
        gate_off: a.gate,
        up_off: a.up,
        k: cfg.n_ff,
        signs_off: s.model.output.signs_ff,
        q8_qs: qs,
        q8_d: d,
        q8_s: sm,
    };
    dispatch(
        s,
        pass,
        &s.model.pipes.silu_mul_rot_q8,
        s.model.win_attn,
        &p,
        (cfg.n_ff / 1024, m, 1),
    );
    marker.mark(pass, "silu_mul_rot");
    proj(
        s,
        pass,
        s.model.win_ffn_d,
        cfg.n_ff,
        &[(lt.ffn_down, cfg.n_embd, a.x)],
        true,
        m,
    );
    marker.mark(pass, "ffn_down");
}

fn gdn_block<M: StepMarker>(
    s: &Session<'_>,
    pass: &mut wgpu::ComputePass<'_>,
    lt: &LayerTensors,
    t: &GdnTensors,
    m: u32,
    pos: u32,
    marker: &mut M,
) {
    let cfg = &s.model.cfg;
    let a = &s.model.act;
    let conv_dim = cfg.ssm_conv_dim();
    let hv = cfg.ssm_n_v_heads;
    norm_rot(s, pass, m, a.x, lt.attn_norm, Some(a.xn));
    marker.mark(pass, "attn_norm_rot");
    proj(
        s,
        pass,
        s.model.win_attn,
        cfg.n_embd,
        &[(t.qkv, conv_dim, a.qkv), (t.gate, cfg.ssm_inner, a.z)],
        false,
        m,
    );
    marker.mark(pass, "gdn_qkvz");
    let pg = GdnGatesParams {
        xn_off: a.xn,
        alpha_off: t.alpha,
        beta_off: t.beta,
        dt_off: t.dt_bias,
        a_off: t.ssm_a,
        out_off: a.gates,
        n_embd: cfg.n_embd,
        n_heads: hv,
    };
    dispatch(
        s,
        pass,
        &s.model.pipes.gdn_gates,
        s.model.win_attn,
        &pg,
        (2 * hv, m, 1),
    );
    marker.mark(pass, "gdn_gates");
    let fresh = u32::from(pos == 0);
    let pc = GdnConvParams {
        m_tokens: m,
        in_off: a.qkv,
        in_stride: conv_dim,
        out_off: a.conv,
        w_off: t.conv1d,
        state_off: s.model.state.conv(t.idx),
        n_qk_heads: 2 * cfg.ssm_n_k_heads,
        channels: conv_dim,
        eps: cfg.rms_eps,
        fresh,
    };
    dispatch(
        s,
        pass,
        &s.model.pipes.gdn_conv,
        s.model.win_attn,
        &pc,
        (conv_dim / 128, 1, 1),
    );
    marker.mark(pass, "gdn_conv");
    let hk = cfg.ssm_n_k_heads * cfg.ssm_head_dim;
    let pr = GdnRecurrenceParams {
        m_tokens: m,
        qkv_off: a.conv,
        stride: conv_dim,
        k_base: hk,
        v_base: 2 * hk,
        gates_off: a.gates,
        o_off: a.o,
        state_off: s.model.state.ssm(t.idx),
        n_k_heads: cfg.ssm_n_k_heads,
        n_v_heads: hv,
        scale: 1.0 / (cfg.ssm_head_dim as f32).sqrt(),
        fresh,
    };
    dispatch(
        s,
        pass,
        &s.model.pipes.gdn_recurrence,
        s.model.win_attn,
        &pr,
        (hv, 4, 1),
    );
    marker.mark(pass, "gdn_recurrence");
    let (qs, d, sm) = q8_offsets(s);
    let po = GdnOutParams {
        o_off: a.o,
        z_off: a.z,
        norm_off: t.ssm_norm,
        signs_off: s.model.output.signs_inner,
        inner: cfg.ssm_inner,
        n_k_heads: cfg.ssm_n_k_heads,
        rep: hv / cfg.ssm_n_k_heads,
        q8_qs: qs,
        q8_d: d,
        q8_s: sm,
        eps: cfg.rms_eps,
    };
    dispatch(
        s,
        pass,
        &s.model.pipes.gdn_out_rot_q8,
        s.model.win_attn,
        &po,
        (cfg.ssm_inner / 1024, m, 1),
    );
    marker.mark(pass, "gdn_out_rot");
    proj(
        s,
        pass,
        s.model.win_attn,
        cfg.ssm_inner,
        &[(t.ssm_out, cfg.n_embd, a.x)],
        true,
        m,
    );
    marker.mark(pass, "gdn_ssm_out");
}

/// Split-K geometry for `n_kv` cache positions: (chunk length, chunk count).
/// Chunks are whole 32-key tiles and there are at most `ATTN_MAX_CHUNKS`.
pub fn attn_chunks(n_kv: u32) -> (u32, u32) {
    let chunk = n_kv
        .div_ceil(ATTN_MAX_CHUNKS)
        .max(ATTN_MIN_CHUNK)
        .div_ceil(32)
        * 32;
    (chunk, n_kv.div_ceil(chunk))
}

fn attn_block<M: StepMarker>(
    s: &Session<'_>,
    pass: &mut wgpu::ComputePass<'_>,
    lt: &LayerTensors,
    t: &AttnTensors,
    m: u32,
    pos: u32,
    marker: &mut M,
) {
    let cfg = &s.model.cfg;
    let a = &s.model.act;
    let kv = &s.model.kv;
    norm_rot(s, pass, m, a.x, lt.attn_norm, Some(a.xn));
    marker.mark(pass, "attn_norm_rot");
    proj(
        s,
        pass,
        s.model.win_attn,
        cfg.n_embd,
        &[
            (t.q, 2 * cfg.q_dim(), a.qkv),
            (t.k, cfg.kv_dim(), a.k),
            (t.v, cfg.kv_dim(), a.v),
        ],
        false,
        m,
    );
    marker.mark(pass, "attn_qkv");
    let pq = AttnQkParams {
        qg_off: a.qkv,
        k_off: a.k,
        v_off: a.v,
        qnorm_off: t.q_norm,
        knorm_off: t.k_norm,
        pos_base: pos,
        n_head: cfg.n_head,
        n_kv_head: cfg.n_kv_head,
        rope_half: cfg.rope_dim / 2,
        eps: cfg.rms_eps,
        kv_d_word: kv.d_word(t.idx),
        kv_qs_byte: kv.qs_byte(t.idx),
        kv_dim: cfg.kv_dim(),
    };
    dispatch(
        s,
        pass,
        &s.model.pipes.attn_qk,
        s.model.win_attn,
        &pq,
        (cfg.n_head + 2 * cfg.n_kv_head, m, 1),
    );
    marker.mark(pass, "attn_qk_rope_kv");
    let (qs, d, sm) = q8_offsets(s);
    if m > 1 {
        let pp = AttnPrefillParams {
            qg_off: a.qkv,
            o_off: a.o,
            n_head: cfg.n_head,
            n_kv_head: cfg.n_kv_head,
            m_tokens: m,
            pos_base: pos,
            kv_dim: cfg.kv_dim(),
            kv_d_word: kv.d_word(t.idx),
            kv_qs_byte: kv.qs_byte(t.idx),
            q_dim: cfg.q_dim(),
            scale: 1.0 / (cfg.head_dim as f32).sqrt(),
        };
        dispatch(
            s,
            pass,
            &s.model.pipes.attn_prefill,
            s.model.win_attn,
            &pp,
            (cfg.n_kv_head, m.div_ceil(2), 1),
        );
        marker.mark(pass, "attn_prefill");
        let po = AttnOutParams {
            mode: 1,
            part_off: a.partials,
            n_chunks: 0,
            o_off: a.o,
            qg_off: a.qkv,
            signs_off: s.model.output.signs_q,
            q_dim: cfg.q_dim(),
            q8_qs: qs,
            q8_d: d,
            q8_s: sm,
        };
        dispatch(
            s,
            pass,
            &s.model.pipes.attn_out_rot_q8,
            s.model.win_attn,
            &po,
            (cfg.q_dim() / 1024, m, 1),
        );
        marker.mark(pass, "attn_out_rot");
        proj(
            s,
            pass,
            s.model.win_attn,
            cfg.q_dim(),
            &[(t.o, cfg.n_embd, a.x)],
            true,
            m,
        );
        marker.mark(pass, "attn_output");
        return;
    }
    let n_kv = pos + 1;
    let (chunk_len, n_chunks) = attn_chunks(n_kv);
    let ps = AttnSplitParams {
        q_off: a.qkv,
        n_head: cfg.n_head,
        n_kv_head: cfg.n_kv_head,
        n_kv,
        chunk_len,
        n_chunks,
        kv_dim: cfg.kv_dim(),
        kv_d_word: kv.d_word(t.idx),
        kv_qs_byte: kv.qs_byte(t.idx),
        part_off: a.partials,
        scale: 1.0 / (cfg.head_dim as f32).sqrt(),
    };
    dispatch(
        s,
        pass,
        &s.model.pipes.attn_split,
        s.model.win_attn,
        &ps,
        (cfg.n_kv_head, n_chunks, 1),
    );
    marker.mark(pass, "attn_split");
    let po = AttnOutParams {
        mode: 0,
        part_off: a.partials,
        n_chunks,
        o_off: a.o,
        qg_off: a.qkv,
        signs_off: s.model.output.signs_q,
        q_dim: cfg.q_dim(),
        q8_qs: qs,
        q8_d: d,
        q8_s: sm,
    };
    dispatch(
        s,
        pass,
        &s.model.pipes.attn_out_rot_q8,
        s.model.win_attn,
        &po,
        (cfg.q_dim() / 1024, 1, 1),
    );
    marker.mark(pass, "attn_out_rot");
    matvec(
        s,
        pass,
        s.model.win_attn,
        cfg.q_dim(),
        &[(t.o, cfg.n_embd, a.x)],
        true,
    );
    marker.mark(pass, "attn_output");
}

fn embed(s: &Session<'_>, pass: &mut wgpu::ComputePass<'_>, sample_in: u32, m: u32) {
    let cfg = &s.model.cfg;
    let p = EmbedParams {
        w_word: s.model.output.token_embd,
        nb_row: cfg.n_embd / 128,
        x_off: s.model.act.x,
        sample_off: sample_in,
        signs_off: s.model.output.signs_embd,
        n_embd: cfg.n_embd,
    };
    dispatch(
        s,
        pass,
        &s.model.pipes.embed,
        s.model.win_embed,
        &p,
        (cfg.n_embd / 1024, m, 1),
    );
}

/// Final norm + LM head for the token row at `x_off` into `act.logits`.
fn lm_head(s: &Session<'_>, pass: &mut wgpu::ComputePass<'_>, x_off: u32) {
    let cfg = &s.model.cfg;
    norm_rot(s, pass, 1, x_off, s.model.output.output_norm, None);
    matvec(
        s,
        pass,
        s.model.win_embed,
        cfg.n_embd,
        &[(s.model.output.output, cfg.n_vocab, s.model.act.logits)],
        false,
    );
}

fn topk(s: &Session<'_>, pass: &mut wgpu::ComputePass<'_>, k: u32) {
    let n = s.model.cfg.n_vocab;
    let partials_off = 2 * TOPK_MAX;
    let p1 = TopKPartialParams {
        n,
        in_offset: s.model.act.logits,
        partials_off,
        n_per_wg: n.div_ceil(TOPK_NUM_PARTIAL_WG),
    };
    dispatch(
        s,
        pass,
        &s.model.pipes.topk_partial,
        s.model.win_attn,
        &p1,
        (TOPK_NUM_PARTIAL_WG, 1, 1),
    );
    let p2 = TopKMergeParams {
        partials_off,
        num_partials: TOPK_NUM_PARTIAL_WG,
        out_offset: 0,
        k,
    };
    dispatch(
        s,
        pass,
        &s.model.pipes.topk_merge,
        s.model.win_attn,
        &p2,
        (1, 1, 1),
    );
}

/// Encode `layers` for `m` consecutive tokens starting at `pos`.
fn encode_layers<M: StepMarker>(
    s: &Session<'_>,
    pass: &mut wgpu::ComputePass<'_>,
    layers: &[LayerTensors],
    m: u32,
    pos: u32,
    marker: &mut M,
) {
    for lt in layers {
        match &lt.kind {
            LayerKind::Gdn(t) => gdn_block(s, pass, lt, t, m, pos, marker),
            LayerKind::Attn(t) => attn_block(s, pass, lt, t, m, pos, marker),
        }
        ffn_block(s, pass, lt, m, marker);
    }
}

/// Encode all layers for one token at `pos` (the token id is read from
/// `io[sample_in]`), optionally followed by the LM head.
fn encode_token<M: StepMarker>(
    s: &Session<'_>,
    pass: &mut wgpu::ComputePass<'_>,
    sample_in: u32,
    pos: u32,
    with_logits: bool,
    marker: &mut M,
) {
    embed(s, pass, sample_in, 1);
    marker.mark(pass, "embed");
    encode_layers(s, pass, &s.model.layers, 1, pos, marker);
    if with_logits {
        lm_head(s, pass, s.model.act.x);
        marker.mark(pass, "lm_head");
    }
}

/// Encode one decoding step (token at `io[sample_in]`, position `pos`) in a
/// single compute pass; with `topk_k = Some(k)` the top-K lands in `io[0..2k]`.
pub fn encode_step<M: StepMarker>(
    se: &mut StepEncoder<'_>,
    sample_in: u32,
    pos: u32,
    topk_k: Option<u32>,
    marker: &mut M,
) {
    let s = se.session;
    let mut desc = wgpu::ComputePassDescriptor {
        label: Some("step"),
        timestamp_writes: None,
    };
    marker.setup_desc(&mut desc);
    let mut pass = se.encoder.begin_compute_pass(&desc);
    marker.mark(&mut pass, "start");
    encode_token(s, &mut pass, sample_in, pos, topk_k.is_some(), marker);
    if let Some(k) = topk_k {
        topk(s, &mut pass, k);
        marker.mark(&mut pass, "topk");
    }
}

/// Build (not submit) one decoding-step command buffer whose first op copies
/// the token id from staging; pair with [`commit_sample_upload`].
pub fn build_step_topk_cb(session: &Session<'_>, pos: u32, k: u32) -> wgpu::CommandBuffer {
    let k = k.clamp(1, TOPK_MAX);
    let mut se = StepEncoder::new(session);
    encode_sample_upload(session, &mut se.encoder, 0, 4);
    encode_step(&mut se, 0, pos, Some(k), &mut NoMarker);
    let bytes = u64::from(k) * 8;
    se.copy_topk_to_readback(bytes);
    se.schedule_topk_map(bytes);
    se.finish()
}

/// Wait for the readback mapped by [`StepEncoder::schedule_topk_map`].
pub fn wait_topk_readback(session: &Session<'_>, k: u32) -> Result<(Vec<f32>, Vec<u32>)> {
    let bytes = u64::from(k) * 8;
    let slice = session.state.readback.slice(0..bytes);
    if let Err(e) = session.model.device.poll(PollType::wait_indefinitely()) {
        session.model.check_device()?;
        return Err(PotError::Poll(e));
    }
    session.model.check_device()?;
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

/// One decoding step: returns the top-`k` candidates for the next token.
pub fn step_topk(
    session: &Session<'_>,
    token: u32,
    pos: u32,
    k: u32,
) -> Result<(Vec<f32>, Vec<u32>)> {
    let cb = build_step_topk_cb(session, pos, k);
    commit_sample_upload(session, bytemuck::bytes_of(&token));
    session.model.queue.submit(Some(cb));
    wait_topk_readback(session, k.clamp(1, TOPK_MAX))
}

/// Same as [`step_topk`] without reading the result back (benches).
#[cfg(feature = "bench-internals")]
pub fn step_no_sample(session: &Session<'_>, token: u32, pos: u32) {
    let mut se = StepEncoder::new(session);
    upload_sample(session, &mut se.encoder, 0, bytemuck::bytes_of(&token));
    encode_step(&mut se, 0, pos, Some(1), &mut NoMarker);
    session.model.queue.submit(Some(se.finish()));
    let _ = session.model.device.poll(PollType::wait_indefinitely());
}

/// Token-at-a-time prefill: all but the last token without the LM head,
/// chunked into command buffers of at most `CHUNK` tokens.
pub fn prefill_token_loop_topk(
    session: &Session<'_>,
    prompt: &[u32],
    pos_base: u32,
    k: u32,
) -> Result<(Vec<f32>, Vec<u32>)> {
    const CHUNK: usize = 64;
    let Some((&last, rest)) = prompt.split_last() else {
        return Err(PotError::Config("prefill requires non-empty input"));
    };
    for chunk_start in (0..rest.len()).step_by(CHUNK) {
        let chunk_end = (chunk_start + CHUNK).min(rest.len());
        let chunk = &rest[chunk_start..chunk_end];
        let mut se = StepEncoder::new(session);
        upload_sample(session, &mut se.encoder, 0, bytemuck::cast_slice(chunk));
        for (i, t) in (chunk_start..chunk_end).enumerate() {
            encode_step(&mut se, i as u32, pos_base + t as u32, None, &mut NoMarker);
        }
        session.model.queue.submit(Some(se.finish()));
        if let Err(e) = session.model.device.poll(PollType::wait_indefinitely()) {
            session.model.check_device()?;
            return Err(PotError::Poll(e));
        }
    }
    step_topk(session, last, pos_base + rest.len() as u32, k)
}

/// Prefill of up to `M_MAX` tokens starting at `pos_base`; returns the
/// top-`k` candidates after the last token.
pub fn prefill_topk<M: StepMarker>(
    session: &Session<'_>,
    prompt: &[u32],
    pos_base: u32,
    k: u32,
    marker: &mut M,
) -> Result<(Vec<f32>, Vec<u32>)> {
    let m = prompt.len() as u32;
    if m == 0 || m > M_MAX {
        return Err(PotError::PrefillTooLarge { n: m, max: M_MAX });
    }
    let s = session;
    let k = k.clamp(1, TOPK_MAX);
    let layers = &s.model.layers;
    let n_groups = layers.len().div_ceil(PREFILL_LAYERS_PER_SUBMIT);
    let mut se = StepEncoder::new(s);
    upload_sample(s, &mut se.encoder, 0, bytemuck::cast_slice(prompt));
    // One submission per group of layers keeps every GPU job well below the
    // driver's lockup timeout (about 2 s on amdgpu) even for M_MAX tokens.
    for (gi, group) in layers.chunks(PREFILL_LAYERS_PER_SUBMIT).enumerate() {
        let last = gi + 1 == n_groups;
        {
            let mut desc = wgpu::ComputePassDescriptor {
                label: Some("prefill"),
                timestamp_writes: None,
            };
            marker.setup_desc(&mut desc);
            let mut pass = se.encoder.begin_compute_pass(&desc);
            marker.mark(&mut pass, if gi == 0 { "start" } else { "(submit gap)" });
            if gi == 0 {
                embed(s, &mut pass, 0, m);
                marker.mark(&mut pass, "embed");
            }
            encode_layers(s, &mut pass, group, m, pos_base, marker);
            if last {
                lm_head(s, &mut pass, s.model.act.x + (m - 1) * s.model.cfg.n_embd);
                marker.mark(&mut pass, "lm_head");
                topk(s, &mut pass, k);
                marker.mark(&mut pass, "topk");
            }
        }
        if !last {
            s.model.queue.submit(Some(se.finish()));
            se = StepEncoder::new(s);
        }
    }
    let bytes = u64::from(k) * 8;
    se.copy_topk_to_readback(bytes);
    se.schedule_topk_map(bytes);
    s.model.queue.submit(Some(se.finish()));
    wait_topk_readback(s, k)
}

/// Layers per command buffer in batched prefill (see [`prefill_topk`]).
const PREFILL_LAYERS_PER_SUBMIT: usize = 16;

#[cfg(test)]
mod tests {
    use super::attn_chunks;
    use crate::model::ATTN_MAX_CHUNKS;

    #[test]
    fn attn_chunk_geometry() {
        for n in [
            1u32, 31, 32, 33, 129, 2047, 2048, 2049, 4096, 116_000, 233_016,
        ] {
            let (len, cnt) = attn_chunks(n);
            assert_eq!(len % 32, 0);
            assert!(cnt <= ATTN_MAX_CHUNKS);
            assert!(len * cnt >= n && len * (cnt - 1) < n, "n={n}");
        }
    }
}

// ---------- reference dumps ----------------------------------------------------

#[cfg(feature = "bench-internals")]
/// One dumped tensor: llama.cpp-style name, row width, and the rows of every
/// processed token in order (`data.len() == ne0 * n_tokens`).
#[derive(Debug, Clone)]
pub struct DumpTensor {
    pub name: String,
    pub ne0: u32,
    pub data: Vec<f32>,
}

/// Run `tokens` from position 0 through the single-token path and capture
/// per-layer activations named like the llama.cpp graph (`attn_norm-N`,
/// `attn_residual-N`, `l_out-N`, `linear_attn_qkv_mixed-N`, `z-N`, `gate-N`,
/// `beta_sigmoid-N`) plus the logits of every position. Resets the session.
///
/// # Errors
///
/// Returns an error on GPU failure.
#[cfg(feature = "bench-internals")]
#[allow(clippy::too_many_lines, reason = "flat capture plan")]
pub fn dump_tokens(session: &mut Session<'_>, tokens: &[u32]) -> Result<Vec<DumpTensor>> {
    // Capture plan: (name, act offset, len) per layer, recorded at two points.
    struct Cap {
        name: String,
        off: u32,
        len: u32,
    }
    session.reset();
    let s: &Session<'_> = session;
    let m = s.model;
    let cfg = &m.cfg;
    let a = m.act;
    let mut mid: Vec<Vec<Cap>> = Vec::new();
    let mut end: Vec<Vec<Cap>> = Vec::new();
    for (il, lt) in m.layers.iter().enumerate() {
        let mut v = vec![
            Cap {
                name: format!("attn_norm-{il}"),
                off: a.xn,
                len: cfg.n_embd,
            },
            Cap {
                name: format!("attn_residual-{il}"),
                off: a.x,
                len: cfg.n_embd,
            },
        ];
        if let LayerKind::Gdn(_) = lt.kind {
            let hv = cfg.ssm_n_v_heads;
            v.push(Cap {
                name: format!("linear_attn_qkv_mixed-{il}"),
                off: a.qkv,
                len: cfg.ssm_conv_dim(),
            });
            v.push(Cap {
                name: format!("z-{il}"),
                off: a.z,
                len: cfg.ssm_inner,
            });
            v.push(Cap {
                name: format!("gate-{il}"),
                off: a.gates,
                len: hv,
            });
            v.push(Cap {
                name: format!("beta_sigmoid-{il}"),
                off: a.gates + hv,
                len: hv,
            });
        }
        mid.push(v);
        end.push(vec![Cap {
            name: format!("l_out-{il}"),
            off: a.x,
            len: cfg.n_embd,
        }]);
    }
    let per_token: u64 = mid
        .iter()
        .chain(end.iter())
        .flatten()
        .map(|c| u64::from(c.len))
        .sum::<u64>()
        + u64::from(cfg.n_vocab);
    let dump_buf = m.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dump"),
        size: per_token * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut out: Vec<DumpTensor> = Vec::new();
    for caps in mid.iter().zip(end.iter()) {
        for c in caps.0.iter().chain(caps.1.iter()) {
            out.push(DumpTensor {
                name: c.name.clone(),
                ne0: c.len,
                data: Vec::new(),
            });
        }
    }
    out.push(DumpTensor {
        name: "logits".into(),
        ne0: cfg.n_vocab,
        data: Vec::new(),
    });

    for (pos, &tok) in tokens.iter().enumerate() {
        let pos = pos as u32;
        let mut enc = m
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dump"),
            });
        upload_sample(s, &mut enc, 0, bytemuck::bytes_of(&tok));
        let mut cursor = 0u64;
        let copy = |enc: &mut wgpu::CommandEncoder, cursor: &mut u64, off: u32, len: u32| {
            enc.copy_buffer_to_buffer(
                &s.state.act,
                u64::from(off) * 4,
                &dump_buf,
                *cursor * 4,
                u64::from(len) * 4,
            );
            *cursor += u64::from(len);
        };
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            embed(s, &mut pass, 0, 1);
        }
        for (il, lt) in m.layers.iter().enumerate() {
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                match &lt.kind {
                    LayerKind::Gdn(t) => gdn_block(s, &mut pass, lt, t, 1, pos, &mut NoMarker),
                    LayerKind::Attn(t) => attn_block(s, &mut pass, lt, t, 1, pos, &mut NoMarker),
                }
            }
            for c in &mid[il] {
                copy(&mut enc, &mut cursor, c.off, c.len);
            }
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                ffn_block(s, &mut pass, lt, 1, &mut NoMarker);
            }
            for c in &end[il] {
                copy(&mut enc, &mut cursor, c.off, c.len);
            }
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            lm_head(s, &mut pass, a.x);
        }
        copy(&mut enc, &mut cursor, a.logits, cfg.n_vocab);
        enc.map_buffer_on_submit(&dump_buf, wgpu::MapMode::Read, .., |_| {});
        m.queue.submit(Some(enc.finish()));
        if let Err(e) = m.device.poll(PollType::wait_indefinitely()) {
            m.check_device()?;
            return Err(PotError::Poll(e));
        }
        m.check_device()?;
        {
            let view = dump_buf.slice(..).get_mapped_range();
            let all: &[f32] = bytemuck::cast_slice(&view);
            let mut off = 0usize;
            for t in &mut out {
                let n = t.ne0 as usize;
                t.data.extend_from_slice(&all[off..off + n]);
                off += n;
            }
        }
        dump_buf.unmap();
    }
    session.pos = tokens.len() as u32;
    Ok(out)
}

/// Debug: run the batched path for the first `n_layers` layers over `tokens`
/// (from position 0) and return the residual rows `x` (`tokens.len() × n_embd`).
#[cfg(feature = "bench-internals")]
pub fn debug_prefill_x(
    session: &mut Session<'_>,
    tokens: &[u32],
    n_layers: usize,
) -> Result<Vec<f32>> {
    session.reset();
    let s: &Session<'_> = session;
    let m = tokens.len() as u32;
    let n = u64::from(m) * u64::from(s.model.cfg.n_embd) * 4;
    let buf = s.model.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("debug"),
        size: n,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = s
        .model
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    upload_sample(s, &mut enc, 0, bytemuck::cast_slice(tokens));
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        embed(s, &mut pass, 0, m);
    }
    for group in
        s.model.layers[..n_layers.min(s.model.layers.len())].chunks(PREFILL_LAYERS_PER_SUBMIT)
    {
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            encode_layers(s, &mut pass, group, m, 0, &mut NoMarker);
        }
        s.model.queue.submit(Some(enc.finish()));
        enc = s
            .model
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    }
    enc.copy_buffer_to_buffer(&s.state.act, u64::from(s.model.act.x) * 4, &buf, 0, n);
    enc.map_buffer_on_submit(&buf, wgpu::MapMode::Read, .., |_| {});
    s.model.queue.submit(Some(enc.finish()));
    if let Err(e) = s.model.device.poll(PollType::wait_indefinitely()) {
        s.model.check_device()?;
        return Err(PotError::Poll(e));
    }
    s.model.check_device()?;
    let v = bytemuck::cast_slice::<u8, f32>(&buf.slice(..).get_mapped_range()).to_vec();
    Ok(v)
}

// ---------- bench markers --------------------------------------------------------

/// Per-dispatch GPU timestamp marker (microbench).
#[cfg(all(feature = "bench-internals", not(feature = "ci")))]
pub struct MicroMarker<'a> {
    session: &'a Session<'a>,
    next_idx: u32,
    labels: Vec<&'static str>,
}

#[cfg(all(feature = "bench-internals", not(feature = "ci")))]
impl<'a> MicroMarker<'a> {
    pub const fn new(session: &'a Session<'a>) -> Self {
        Self {
            session,
            next_idx: 0,
            labels: Vec::new(),
        }
    }

    /// Resolve the timestamps into `(label, duration_ns)` spans.
    pub fn resolve(self) -> Result<Vec<(&'static str, f32)>> {
        let n = self.next_idx;
        if n < 2 {
            return Ok(vec![]);
        }
        let ticks = bench_resolve_ticks(self.session, n)?;
        let period = self.session.model.bench_ts_period_ns;
        Ok((1..n as usize)
            .map(|i| {
                (
                    self.labels[i],
                    ticks[i].saturating_sub(ticks[i - 1]) as f32 * period,
                )
            })
            .collect())
    }
}

#[cfg(all(feature = "bench-internals", not(feature = "ci")))]
impl StepMarker for MicroMarker<'_> {
    fn setup_desc<'a>(&'a self, _desc: &mut wgpu::ComputePassDescriptor<'a>) {}
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

/// Whole-pass GPU timestamp marker (end-to-end timing without per-dispatch flushes).
#[cfg(all(feature = "bench-internals", not(feature = "ci")))]
pub struct BenchMarker<'a> {
    session: &'a Session<'a>,
    passes: Cell<u32>,
}

#[cfg(all(feature = "bench-internals", not(feature = "ci")))]
impl<'a> BenchMarker<'a> {
    pub const fn new(session: &'a Session<'a>) -> Self {
        Self {
            session,
            passes: Cell::new(0),
        }
    }

    /// Summed GPU duration of all instrumented passes in nanoseconds.
    pub fn resolve(self) -> Result<f32> {
        let n = self.passes.get();
        let ticks = bench_resolve_ticks(self.session, 2 * n)?;
        let total: u64 = ticks
            .chunks_exact(2)
            .map(|c| c[1].saturating_sub(c[0]))
            .sum();
        Ok(total as f32 * self.session.model.bench_ts_period_ns)
    }
}

#[cfg(all(feature = "bench-internals", not(feature = "ci")))]
impl StepMarker for BenchMarker<'_> {
    fn setup_desc<'a>(&'a self, desc: &mut wgpu::ComputePassDescriptor<'a>) {
        let i = self.passes.get();
        self.passes.set(i + 1);
        desc.timestamp_writes = Some(wgpu::ComputePassTimestampWrites {
            query_set: &self.session.state.bench_query_set,
            beginning_of_pass_write_index: Some(2 * i),
            end_of_pass_write_index: Some(2 * i + 1),
        });
    }
    #[inline(always)]
    fn mark(&mut self, _pass: &mut wgpu::ComputePass<'_>, _label: &'static str) {}
}

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
