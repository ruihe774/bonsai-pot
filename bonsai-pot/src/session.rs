use alloc::vec::Vec;
use core::mem::size_of;

use crate::error::{PotError, Result};
use crate::forward::{self, build_step_matvec_topk_cb, commit_sample_upload, wait_topk_readback};
use crate::kv_snapshot::{self, KvSnapshot};
#[cfg(all(feature = "bench-internals", not(feature = "ci")))]
use crate::model::BENCH_QS_SLOTS;
use crate::model::{
    ATTN_CHUNK_SIZE, BindGroupLayouts, Buffers, M_MAX, Model, STAGING_CHUNK, TOPK_MAX,
    TOPK_NUM_PARTIAL_WG,
};

/// Per-conversation GPU resources, owned by [`Session`]. Holds the KV cache,
/// the activation scratch, the sample/staging/readback buffers, the split-K
/// attention partials, and the cached bind groups that bind any of the above.
/// One `SessionState` per session means concurrent `Session`s don't alias on
/// these buffers.
pub struct SessionState {
    pub(crate) kv_k: wgpu::Buffer,
    pub(crate) kv_v: wgpu::Buffer,
    // `act` / `act_q8` / `attn_partials` are accessed through `cached`
    // (the bind groups bound at session init); host code never reaches them
    // by name, but the fields keep ownership so the buffers outlive the
    // session.
    #[allow(dead_code, reason = "kept alive via cached bind groups")]
    pub(crate) act: wgpu::Buffer, // f16 activations
    #[allow(dead_code, reason = "kept alive via cached bind groups")]
    pub(crate) act_q8: wgpu::Buffer, // Q8_0 activations (raw u32 buffer)
    #[allow(dead_code, reason = "kept alive via cached bind groups")]
    pub(crate) attn_partials: wgpu::Buffer, // f32 partials for split-K attention
    pub(crate) sample: wgpu::Buffer, // u32 storage: input token id @ [0..M], topk output @ [0..2K]
    pub(crate) staging: wgpu::Buffer, // MAP_WRITE | COPY_SRC staging for sample uploads
    pub(crate) readback: wgpu::Buffer, // u32 readback (mappable)
    pub(crate) cached: SessionBindGroups,
    #[cfg(all(feature = "bench-internals", not(feature = "ci")))]
    pub(crate) bench_query_set: wgpu::QuerySet,
    #[cfg(all(feature = "bench-internals", not(feature = "ci")))]
    pub(crate) bench_resolve: wgpu::Buffer, // QUERY_RESOLVE | COPY_SRC
    #[cfg(all(feature = "bench-internals", not(feature = "ci")))]
    pub(crate) bench_readback: wgpu::Buffer, // COPY_DST | MAP_READ
}

/// Pre-built bind groups indexed by (BGL kind, weight buffer). One BG per
/// (kind, weight buffer) is reused across every dispatch of that kind in a
/// step. Every bind group here references at least one per-session buffer
/// (`act`, `act_q8`, `kv_k`, `kv_v`, `attn_partials`, or `sample`), so the
/// whole struct lives in [`SessionState`] and is rebuilt per session.
pub struct SessionBindGroups {
    pub(crate) embed: wgpu::BindGroup,         // (w_embed, act, sample)
    pub(crate) rms_norm: wgpu::BindGroup,      // (act, w_norms)
    pub(crate) matvec_w_attn: wgpu::BindGroup, // (w_attn,   act)
    pub(crate) matvec_w_ffn_gu: wgpu::BindGroup, // (w_ffn_gu, act)
    pub(crate) matvec_w_ffn_d: wgpu::BindGroup, // (w_ffn_d,  act)
    pub(crate) matvec_w_embed: wgpu::BindGroup, // (w_embed,  act) — LM head
    pub(crate) matvec_fused_normed_w_attn: wgpu::BindGroup, // (w_attn,   act, w_norms)
    pub(crate) matvec_fused_normed_w_ffn_gu: wgpu::BindGroup, // (w_ffn_gu, act, w_norms)
    pub(crate) matmul_w_attn: wgpu::BindGroup, // (w_attn,   act_q8, act)
    pub(crate) matmul_w_ffn_gu: wgpu::BindGroup,
    pub(crate) matmul_w_ffn_d: wgpu::BindGroup,
    pub(crate) matmul_w_embed: wgpu::BindGroup,
    pub(crate) attn_prefill_tiled: wgpu::BindGroup, // (act ro, kv_k, kv_v, act_q8 rw)
    pub(crate) attn_split: wgpu::BindGroup,         // (act, kv_k, kv_v, attn_partials)
    pub(crate) attn_merge: wgpu::BindGroup,         // (act, attn_partials)
    pub(crate) rms_norm_q8_0: wgpu::BindGroup,      // (act, w_norms, act_q8)
    pub(crate) silu_mul_q8_0: wgpu::BindGroup,      // (act, act_q8)
    pub(crate) topk_partial: wgpu::BindGroup,       // (act, sample)
    pub(crate) topk_merge: wgpu::BindGroup,         // (sample)
    pub(crate) kv_writeback_fused: wgpu::BindGroup, // (act, w_norms, rope_cs, kv_k, kv_v)
    pub(crate) q_norm_rope_fused: wgpu::BindGroup,  // (act, w_norms, rope_cs)
}

impl SessionState {
    /// Allocate this session's per-conversation GPU buffers and build the
    /// bind groups that reference them. Sized from `model.cfg` and
    /// `model.max_seq`. All resources are independent of any other session
    /// allocated against the same `Model`.
    fn new(model: &Model) -> Self {
        // Sample buffer roles (max footprint dictates size):
        //   - input (matmul prefill): M_MAX u32 token ids
        //   - output: 2*TOPK_MAX u32 (K f32 logits + K u32 indices) at offset 0
        //   - partials scratch: TOPK_NUM_PARTIAL_WG * 2*TOPK_MAX u32 at offset 2*TOPK_MAX
        const SAMPLE_OUT_AND_PARTIALS: u64 =
            ((1 + TOPK_NUM_PARTIAL_WG as u64) * 2 * TOPK_MAX as u64) * 4;
        const SAMPLE_INPUT: u64 = M_MAX as u64 * 4;
        const SAMPLE_RAW: u64 = if SAMPLE_INPUT > SAMPLE_OUT_AND_PARTIALS {
            SAMPLE_INPUT
        } else {
            SAMPLE_OUT_AND_PARTIALS
        };
        const SAMPLE_BYTES: u64 = SAMPLE_RAW.next_power_of_two();
        const READBACK_BYTES: u64 = 2 * TOPK_MAX as u64 * 4;

        let device = &model.device;
        let cfg = &model.cfg;
        let max_seq = model.max_seq;

        // KV cache (Q8_0): per-buffer layout is d-section followed by qs-section.
        let nb_per_row = u64::from(cfg.kv_dim / 32);
        let kv_d_total: u64 = u64::from(cfg.n_layer) * u64::from(max_seq) * nb_per_row * 4;
        let kv_qs_total: u64 = u64::from(cfg.n_layer) * u64::from(max_seq) * u64::from(cfg.kv_dim);
        let kv_total: u64 = kv_d_total + kv_qs_total;
        let kv_k = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kv_k"),
            size: kv_total,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let kv_v = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kv_v"),
            size: kv_total,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let act_size =
            (u64::from(model.act_layout.total_elems) * size_of::<half::f16>() as u64 + 3) & !3;
        let act = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("act"),
            size: act_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let max_k = cfg.n_ff;
        let q8_d_section_bytes = M_MAX * (max_k / 32) * 4;
        let q8_qs_section_bytes = M_MAX * max_k;
        let act_q8_size = q8_d_section_bytes + q8_qs_section_bytes;
        let act_q8_size = act_q8_size.div_ceil(16) * 16;
        let act_q8 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("act_q8"),
            size: u64::from(act_q8_size),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let n_chunks_max = max_seq.div_ceil(ATTN_CHUNK_SIZE);
        let attn_partials_size =
            u64::from(cfg.n_head) * u64::from(n_chunks_max) * (u64::from(cfg.head_dim) + 2) * 4;
        let attn_partials = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("attn_partials"),
            size: attn_partials_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let sample = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sample"),
            size: SAMPLE_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: STAGING_CHUNK,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: READBACK_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        #[cfg(all(feature = "bench-internals", not(feature = "ci")))]
        let bench_query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("bench_qs"),
            ty: wgpu::QueryType::Timestamp,
            count: BENCH_QS_SLOTS,
        });
        #[cfg(all(feature = "bench-internals", not(feature = "ci")))]
        let bench_resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bench_resolve"),
            size: u64::from(BENCH_QS_SLOTS) * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        #[cfg(all(feature = "bench-internals", not(feature = "ci")))]
        let bench_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bench_readback"),
            size: u64::from(BENCH_QS_SLOTS) * 8,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let cached = build_session_bind_groups(
            device,
            &model.bgls,
            &model.buffers,
            &kv_k,
            &kv_v,
            &act,
            &act_q8,
            &attn_partials,
            &sample,
        );

        Self {
            kv_k,
            kv_v,
            act,
            act_q8,
            attn_partials,
            sample,
            staging,
            readback,
            cached,
            #[cfg(all(feature = "bench-internals", not(feature = "ci")))]
            bench_query_set,
            #[cfg(all(feature = "bench-internals", not(feature = "ci")))]
            bench_resolve,
            #[cfg(all(feature = "bench-internals", not(feature = "ci")))]
            bench_readback,
        }
    }
}

/// Build the full set of cached bind groups for one session. Called once
/// per [`SessionState::new`].
fn build_session_bind_groups(
    device: &wgpu::Device,
    bgls: &BindGroupLayouts,
    weights: &Buffers,
    kv_k: &wgpu::Buffer,
    kv_v: &wgpu::Buffer,
    act: &wgpu::Buffer,
    act_q8: &wgpu::Buffer,
    attn_partials: &wgpu::Buffer,
    sample: &wgpu::Buffer,
) -> SessionBindGroups {
    let mk = |label: &str,
              layout: &wgpu::BindGroupLayout,
              storages: &[&wgpu::Buffer]|
     -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry<'_>> = storages
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &entries,
        })
    };
    SessionBindGroups {
        embed: mk(
            "session_embed",
            &bgls.embed,
            &[&weights.w_embed, act, sample],
        ),
        rms_norm: mk("session_rms_norm", &bgls.rms_norm, &[act, &weights.w_norms]),
        matvec_w_attn: mk("session_matvec_attn", &bgls.matvec, &[&weights.w_attn, act]),
        matvec_w_ffn_gu: mk(
            "session_matvec_ffngu",
            &bgls.matvec,
            &[&weights.w_ffn_gu, act],
        ),
        matvec_w_ffn_d: mk(
            "session_matvec_ffnd",
            &bgls.matvec,
            &[&weights.w_ffn_d, act],
        ),
        matvec_w_embed: mk(
            "session_matvec_embed",
            &bgls.matvec,
            &[&weights.w_embed, act],
        ),
        matvec_fused_normed_w_attn: mk(
            "session_matvec_fused_normed_attn",
            &bgls.matvec_fused_normed,
            &[&weights.w_attn, act, &weights.w_norms],
        ),
        matvec_fused_normed_w_ffn_gu: mk(
            "session_matvec_fused_normed_ffngu",
            &bgls.matvec_fused_normed,
            &[&weights.w_ffn_gu, act, &weights.w_norms],
        ),
        matmul_w_attn: mk(
            "session_matmul_attn",
            &bgls.matmul,
            &[&weights.w_attn, act_q8, act],
        ),
        matmul_w_ffn_gu: mk(
            "session_matmul_ffngu",
            &bgls.matmul,
            &[&weights.w_ffn_gu, act_q8, act],
        ),
        matmul_w_ffn_d: mk(
            "session_matmul_ffnd",
            &bgls.matmul,
            &[&weights.w_ffn_d, act_q8, act],
        ),
        matmul_w_embed: mk(
            "session_matmul_embed",
            &bgls.matmul,
            &[&weights.w_embed, act_q8, act],
        ),
        attn_prefill_tiled: mk(
            "session_attn_prefill_tiled",
            &bgls.attn_prefill_tiled,
            &[act, kv_k, kv_v, act_q8],
        ),
        attn_split: mk(
            "session_attn_split",
            &bgls.attn_split,
            &[act, kv_k, kv_v, attn_partials],
        ),
        attn_merge: mk(
            "session_attn_merge",
            &bgls.attn_merge,
            &[act, attn_partials],
        ),
        rms_norm_q8_0: mk(
            "session_rms_norm_q8_0",
            &bgls.rms_norm_q8_0,
            &[act, &weights.w_norms, act_q8],
        ),
        silu_mul_q8_0: mk("session_silu_mul_q8_0", &bgls.silu_mul_q8_0, &[act, act_q8]),
        topk_partial: mk("session_topk_partial", &bgls.topk_partial, &[act, sample]),
        topk_merge: mk("session_topk_merge", &bgls.topk_merge, &[sample]),
        kv_writeback_fused: mk(
            "session_kv_writeback_fused",
            &bgls.kv_writeback_fused,
            &[act, &weights.w_norms, &weights.rope_table, kv_k, kv_v],
        ),
        q_norm_rope_fused: mk(
            "session_q_norm_rope_fused",
            &bgls.q_norm_rope_fused,
            &[act, &weights.w_norms, &weights.rope_table],
        ),
    }
}

/// Token sampler.
///
/// There is no separate greedy mode — set `temperature = 0.0` (or
/// `top_k = Some(1)`) for argmax-like behavior. `top_k` is silently capped at 32.
#[derive(Debug, Clone)]
pub struct Sampler {
    /// Logit temperature. `0.0` ⇒ argmax over the K candidates. Must be ≥ 0.
    pub temperature: f32,
    /// Truncate to top-`k` candidates before sampling. `None` ⇒ keep all
    /// 32 candidates returned by the GPU.
    pub top_k: Option<u32>,
    /// Nucleus filter: keep the smallest set of candidates whose cumulative
    /// probability ≥ `p`. `None` ⇒ no nucleus filter.
    pub top_p: Option<f32>,
    /// PRNG seed for reproducible sampling. Combined with the current
    /// position so that two `reset()`-and-rerun sequences match.
    pub seed: u64,
}

impl Default for Sampler {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            seed: 0,
        }
    }
}

/// Reason a `generate*` call returned.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum StopReason {
    /// Produced a stop token (defaults to model EOS).
    Eos,
    /// Reached `max_new_tokens` without a stop token.
    MaxTokens,
}

/// Options for [`Session::generate`] / [`Session::generate_streaming`].
pub struct GenerateOptions<F: Fn(u32) -> bool = fn(u32) -> bool> {
    pub max_new_tokens: u32,
    /// Predicate run on each newly sampled token. Returning `true` ends
    /// generation with [`StopReason::Eos`] before the token is delivered to
    /// the streaming callback. `None` ⇒ stop on the model's default EOS token.
    pub stop_pred: Option<F>,
    pub sampler: Sampler,
}

impl Default for GenerateOptions<fn(u32) -> bool> {
    fn default() -> Self {
        Self {
            max_new_tokens: 32,
            stop_pred: None,
            sampler: Sampler::default(),
        }
    }
}

/// Per-conversation inference state.
///
/// Carries the current KV-cache cursor and owns this session's per-conversation
/// GPU resources (KV cache, activation scratch, sample/staging/readback
/// buffers, split-K attention partials, and the bind groups bound to those
/// buffers). Multiple `Session`s may live against one [`Model`] without
/// aliasing — each owns disjoint GPU state.
pub struct Session<'m> {
    pub(crate) model: &'m Model,
    pub(crate) state: SessionState,
    pub(crate) pos: u32,
}

impl<'m> Session<'m> {
    pub(crate) fn new(model: &'m Model) -> Self {
        Self {
            model,
            state: SessionState::new(model),
            pos: 0,
        }
    }

    /// Current position (number of tokens consumed so far).
    #[must_use]
    pub const fn pos(&self) -> u32 {
        self.pos
    }

    /// Borrow the [`Model`] this session was opened against.
    #[must_use]
    pub const fn model(&self) -> &'m Model {
        self.model
    }

    /// Reset to a fresh conversation. O(1) — the KV cache is overwritten in
    /// place by subsequent prefill / step calls, so no GPU work is needed.
    pub const fn reset(&mut self) {
        self.pos = 0;
    }

    /// Read back the live `[0..pos)` slice of the GPU KV cache to host memory.
    ///
    /// The resulting [`KvSnapshot`] is not tied to this `Session` and can be
    /// freely cloned, persisted to disk via [`KvSnapshot::to_bytes`], and
    /// restored into any `Session` created from the same [`crate::Model`].
    ///
    /// Cost: one `PCIe` round-trip plus a memcpy. At `pos=512` with `kv_dim=1024`,
    /// roughly 1–2 ms on `PCIe 4`.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU readback fails.
    pub fn snapshot(&mut self) -> Result<KvSnapshot> {
        self.model.check_device()?;
        kv_snapshot::capture(self)
    }

    /// Replace the GPU KV cache with `snap`'s contents and set
    /// `self.pos = snap.pos()`.
    ///
    /// Validates `snap` against the model (matching `n_layer` / `kv_dim`;
    /// `snap.pos() ≤ model.max_seq_len()`). After a successful restore,
    /// the next call is typically [`prefill`](Self::prefill) (works at any
    /// `pos`) or [`step`](Self::step) if `snap.pos() > 0`.
    ///
    /// Cost: one `Queue::write_buffer` upload per layer pair (cold path; uses the
    /// queue's built-in staging since the KV buffers are too large for the sample staging buffer).
    ///
    /// # Errors
    ///
    /// Returns an error if `snap` does not match the model's `n_layer` /
    /// `kv_dim`, or if `snap.pos()` exceeds `model.max_seq_len()`.
    pub fn restore(&mut self, snap: &KvSnapshot) -> Result<()> {
        self.model.check_device()?;
        kv_snapshot::apply(self, snap)
    }

    /// Batched matmul prefill. Advances `pos` by `tokens.len()` and returns
    /// the first sampled token (drawn from the last logits via `sampler`).
    ///
    /// Works at any `self.pos()` (including incremental prefill into an
    /// existing context). Prompts longer than `model.max_prefill_tokens()`
    /// are chunked transparently into batched dispatches; intermediate
    /// chunks' logits are discarded (only the final chunk's top-K is sampled).
    ///
    /// # Errors
    ///
    /// Returns an error if the prompt would overflow the KV cache, or if the
    /// underlying GPU dispatch fails.
    pub fn prefill(&mut self, tokens: &[u32], sampler: &Sampler) -> Result<u32> {
        self.model.check_device()?;
        let n = tokens.len() as u32;
        if n == 0 {
            return Err(PotError::Config(
                "Session::prefill requires non-empty input",
            ));
        }
        if self.pos + n > self.model.max_seq {
            return Err(PotError::ContextOverflow {
                pos: self.pos,
                n,
                max: self.model.max_seq,
            });
        }
        let k = effective_k(sampler);
        let chunk = self.model.max_prefill_tokens() as usize;
        let mut chosen = 0u32;
        for slice in tokens.chunks(chunk) {
            let pos_base = self.pos;
            let (logits, indices) =
                forward::prefill_matmul_topk(self, slice, pos_base, k, &mut forward::NoMarker)?;
            chosen = sample_from_topk(&logits, &indices, sampler, pos_base);
            self.pos += slice.len() as u32;
        }
        Ok(chosen)
    }

    /// Single-token-at-a-time matvec-loop prefill. Strictly dominated by
    /// [`Session::prefill`] for end-users (slower, same result); kept as an
    /// independent implementation for the matvec-vs-matmul parity tests and
    /// for the CLI's `--mode gen` (which exercises the matvec prefill path
    /// end-to-end).
    ///
    /// # Errors
    ///
    /// Returns an error if the prompt would overflow the KV cache, or if the
    /// underlying GPU dispatch fails.
    #[doc(hidden)]
    pub fn prefill_one_at_a_time(&mut self, tokens: &[u32], sampler: &Sampler) -> Result<u32> {
        self.model.check_device()?;
        let n = tokens.len() as u32;
        if self.pos + n > self.model.max_seq {
            return Err(PotError::ContextOverflow {
                pos: self.pos,
                n,
                max: self.model.max_seq,
            });
        }
        let k = effective_k(sampler);
        let pos_base = self.pos;
        let (logits, indices) = forward::prefill_matvec_loop_topk(self, tokens, pos_base, k)?;
        let chosen = sample_from_topk(&logits, &indices, sampler, pos_base);
        self.pos += n;
        Ok(chosen)
    }

    /// One matvec decoding step on `token`. Advances `pos` by 1 and returns
    /// the next sampled token.
    ///
    /// # Errors
    ///
    /// Returns an error if advancing `pos` would overflow the KV cache, or if
    /// the underlying GPU dispatch fails.
    pub fn step(&mut self, token: u32, sampler: &Sampler) -> Result<u32> {
        self.model.check_device()?;
        if self.pos + 1 > self.model.max_seq {
            return Err(PotError::ContextOverflow {
                pos: self.pos,
                n: 1,
                max: self.model.max_seq,
            });
        }
        let k = effective_k(sampler);
        let pos = self.pos;
        let (logits, indices) = forward::step_matvec_topk(self, token, pos, k)?;
        let chosen = sample_from_topk(&logits, &indices, sampler, pos);
        self.pos += 1;
        Ok(chosen)
    }

    /// Collect-mode generation. `first_token` is fed as the next input but
    /// is **not** included in the returned `Vec`. Returns
    /// `(generated_tokens, stop_reason)`.
    ///
    /// # Errors
    ///
    /// Returns an error if generation would overflow the KV cache, or if the
    /// underlying GPU dispatch fails.
    pub fn generate<S: Fn(u32) -> bool>(
        &mut self,
        first_token: u32,
        opts: &GenerateOptions<S>,
    ) -> Result<(Vec<u32>, StopReason)> {
        let mut out = Vec::with_capacity(opts.max_new_tokens as usize);
        let stop = self.generate_streaming(first_token, opts, |id| out.push(id))?;
        Ok((out, stop))
    }

    /// Streaming generation. `on_token` fires once per emitted token (NOT
    /// including `first_token`). The stop predicate, when it returns `true`,
    /// terminates generation **before** the token is delivered to the callback.
    ///
    /// Pipelined: the `CommandBuffer` for step i+1 is encoded on the CPU while
    /// the GPU drains step i, hiding the encode latency behind the GPU step time.
    /// The token-id staging write is deferred to after `wait_topk_readback`
    /// fires the previous step's staging remap callback.
    ///
    /// # Errors
    ///
    /// Returns an error if generation would overflow the KV cache, or if the
    /// underlying GPU dispatch fails.
    pub fn generate_streaming<S: Fn(u32) -> bool, On: FnMut(u32)>(
        &mut self,
        first_token: u32,
        opts: &GenerateOptions<S>,
        mut on_token: On,
    ) -> Result<StopReason> {
        self.model.check_device()?;
        let default_eos = self.model.cfg.eos_token_id;
        let is_stop = |t: u32| opts.stop_pred.as_ref().map_or(t == default_eos, |p| p(t));
        let max_new = opts.max_new_tokens;
        let max_seq = self.model.max_seq;
        let k = effective_k(&opts.sampler);

        if max_new == 0 {
            return Ok(StopReason::MaxTokens);
        }
        if self.pos + 1 > max_seq {
            return Err(PotError::ContextOverflow {
                pos: self.pos,
                n: 1,
                max: max_seq,
            });
        }

        // --- prime step 0 ---
        let prime_cb = build_step_matvec_topk_cb(self, self.pos, k);
        commit_sample_upload(self, bytemuck::bytes_of(&first_token));
        self.model.queue.submit(Some(prime_cb));

        for i in 0..max_new {
            // Encode next step's CB while the GPU drains the current one.
            // The next CB records `copy_buffer_to_buffer(staging → sample)` but doesn't
            // need staging contents at encode time — the host write happens after
            // `wait_topk_readback` (which fires the prior remap so staging is mapped).
            // Gated on pos+2 <= max_seq and a next iteration existing.
            let next_cb = if self.pos + 2 <= max_seq && i + 1 < max_new {
                Some(build_step_matvec_topk_cb(self, self.pos + 1, k))
            } else {
                None
            };

            let (logits, indices) = wait_topk_readback(self, k)?;
            let chosen = sample_from_topk(&logits, &indices, &opts.sampler, self.pos);
            self.pos += 1;

            if is_stop(chosen) {
                return Ok(StopReason::Eos);
            }
            on_token(chosen);

            if i + 1 == max_new {
                return Ok(StopReason::MaxTokens);
            }
            let Some(cb) = next_cb else {
                return Err(PotError::ContextOverflow {
                    pos: self.pos,
                    n: 1,
                    max: max_seq,
                });
            };
            commit_sample_upload(self, bytemuck::bytes_of(&chosen));
            self.model.queue.submit(Some(cb));
        }
        Ok(StopReason::MaxTokens)
    }
}

fn effective_k(s: &Sampler) -> u32 {
    let cap = TOPK_MAX;
    match s.top_k {
        Some(k) if k > 0 => k.min(cap),
        _ => cap,
    }
}

/// CPU side of the hybrid sampler. Inputs are the GPU-side top-K candidates
/// (logits descending, paired indices); output is the chosen vocab id.
fn sample_from_topk(logits: &[f32], indices: &[u32], s: &Sampler, pos: u32) -> u32 {
    debug_assert_eq!(logits.len(), indices.len());
    let n = logits.len();
    if n == 0 {
        return 0;
    }

    // Argmax fast path: logits[0] is already the max because the GPU returns
    // K candidates sorted descending.
    if s.temperature <= 0.0 || s.top_k == Some(1) {
        return indices[0];
    }

    // Apply user-supplied top-k cap (effective_k already capped to TOPK_MAX).
    let kk = match s.top_k {
        Some(k) if (k as usize) < n => k as usize,
        _ => n,
    };

    // Temperature-scaled softmax over top-kk.
    let inv_t = 1.0 / s.temperature;
    let max_l = logits[0] * inv_t;
    let mut probs: Vec<f32> = (0..kk)
        .map(|i| logits[i].mul_add(inv_t, -max_l).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return indices[0];
    }
    for p in &mut probs {
        *p /= sum;
    }

    // Top-p (nucleus) filter on the (already descending) probabilities.
    let cutoff = s.top_p.map_or(1.0, |p| p.clamp(0.0, 1.0));
    let mut cum = 0.0f32;
    let mut keep = kk;
    for (i, &p) in probs.iter().enumerate().take(kk) {
        cum += p;
        if cum >= cutoff {
            keep = i + 1;
            break;
        }
    }
    let kept = &probs[..keep];
    let kept_sum: f32 = kept.iter().sum();
    if kept_sum <= 0.0 || !kept_sum.is_finite() {
        return indices[0];
    }

    // Multinomial: xorshift64-seeded uniform in [0, kept_sum).
    let r = uniform_f32(s.seed.wrapping_add(u64::from(pos))) * kept_sum;
    let mut acc = 0.0f32;
    for i in 0..keep {
        acc += probs[i];
        if r < acc {
            return indices[i];
        }
    }
    indices[keep - 1]
}

/// xorshift64 → uniform f32 in [0, 1). Seed is mixed via `SplitMix64` first so
/// nearby seeds (e.g. seed+pos) yield well-distributed outputs.
fn uniform_f32(seed: u64) -> f32 {
    // SplitMix64 finalizer
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z = z ^ (z >> 31);
    // Take top 24 bits → f32 in [0, 1).
    let bits24 = (z >> 40) as u32; // 24 bits
    bits24 as f32 / (1u32 << 24) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn greedy() -> Sampler {
        Sampler {
            temperature: 0.0,
            ..Sampler::default()
        }
    }

    #[test]
    fn greedy_temperature_zero() {
        let logits = vec![3.0f32, 2.0, 1.0];
        let indices = vec![42u32, 7, 99];
        assert_eq!(sample_from_topk(&logits, &indices, &greedy(), 0), 42);
    }

    #[test]
    fn greedy_top_k_one() {
        let logits = vec![3.0f32, 2.0, 1.0];
        let indices = vec![42u32, 7, 99];
        let s = Sampler {
            top_k: Some(1),
            temperature: 1.0,
            ..Sampler::default()
        };
        assert_eq!(sample_from_topk(&logits, &indices, &s, 0), 42);
    }

    #[test]
    fn top_p_zero_picks_argmax() {
        let logits = vec![3.0f32, 2.0, 1.0];
        let indices = vec![42u32, 7, 99];
        let s = Sampler {
            top_p: Some(0.0),
            temperature: 1.0,
            ..Sampler::default()
        };
        // With top_p=0.0, cumulative >= 0.0 on first element, so keep=1 and
        // indices[0] is always returned regardless of the random draw.
        for seed in 0..20u64 {
            let s2 = Sampler { seed, ..s.clone() };
            assert_eq!(sample_from_topk(&logits, &indices, &s2, 0), 42);
        }
    }

    #[test]
    fn seed_determinism() {
        let logits = vec![1.0f32; 8];
        let indices: Vec<u32> = (0..8).collect();
        let s = Sampler {
            temperature: 1e9,
            seed: 42,
            ..Sampler::default()
        };
        let r1 = sample_from_topk(&logits, &indices, &s, 5);
        let r2 = sample_from_topk(&logits, &indices, &s, 5);
        assert_eq!(r1, r2);
        // Different pos gives a different seed input.
        let r3 = sample_from_topk(&logits, &indices, &s, 6);
        assert_ne!(r1, r3);
    }

    #[test]
    fn empty_input_returns_zero() {
        assert_eq!(sample_from_topk(&[], &[], &Sampler::default(), 0), 0);
    }

    #[test]
    fn non_finite_falls_back_to_argmax() {
        // INF logit → INF - INF = NaN in softmax → sum is NaN → fallback.
        let logits = vec![f32::INFINITY, 1.0, 0.0];
        let indices = vec![42u32, 7, 99];
        let s = Sampler {
            temperature: 1.0,
            ..Sampler::default()
        };
        assert_eq!(sample_from_topk(&logits, &indices, &s, 0), 42);
    }

    #[test]
    fn effective_k_caps_to_topk_max() {
        assert_eq!(
            effective_k(&Sampler {
                top_k: None,
                ..Sampler::default()
            }),
            TOPK_MAX
        );
        assert_eq!(
            effective_k(&Sampler {
                top_k: Some(0),
                ..Sampler::default()
            }),
            TOPK_MAX
        );
        assert_eq!(
            effective_k(&Sampler {
                top_k: Some(100),
                ..Sampler::default()
            }),
            TOPK_MAX
        );
        assert_eq!(
            effective_k(&Sampler {
                top_k: Some(5),
                ..Sampler::default()
            }),
            5
        );
    }

    #[test]
    fn uniform_f32_in_range() {
        for seed in 0u64..1000 {
            let v = uniform_f32(seed);
            assert!(
                (0.0..1.0).contains(&v),
                "uniform_f32({seed}) = {v} not in [0, 1)"
            );
        }
    }

    #[test]
    fn sampler_default_values() {
        let s = Sampler::default();
        assert_eq!(s.temperature, 1.0);
        assert_eq!(s.top_k, None);
        assert_eq!(s.top_p, None);
        assert_eq!(s.seed, 0);
    }

    #[test]
    fn generate_options_default() {
        let o = GenerateOptions::default();
        assert_eq!(o.max_new_tokens, 32);
        assert!(o.stop_pred.is_none());
        assert_eq!(o.sampler.temperature, 1.0);
    }
}
