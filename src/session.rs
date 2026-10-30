use alloc::vec::Vec;

use crate::error::{PotError, Result};
use crate::forward::{self, build_step_topk_cb, commit_sample_upload, wait_topk_readback};
use crate::kv_snapshot::{self, KvSnapshot};
#[cfg(all(feature = "bench-internals", not(feature = "ci")))]
use crate::model::BENCH_QS_SLOTS;
use crate::model::{M_MAX, Model, STAGING_CHUNK, TOPK_MAX, TOPK_NUM_PARTIAL_WG};

/// Per-conversation GPU resources owned by a [`Session`]: KV cache, recurrent
/// state, activation scratch, token/sample buffers, and the bind groups.
pub struct SessionState {
    pub(crate) kv_k: wgpu::Buffer,
    pub(crate) kv_v: wgpu::Buffer,
    pub(crate) state: wgpu::Buffer,
    #[cfg_attr(
        not(feature = "bench-internals"),
        allow(dead_code, reason = "kept alive via bind groups")
    )]
    pub(crate) act: wgpu::Buffer,
    #[allow(dead_code, reason = "kept alive via bind groups")]
    pub(crate) q8: wgpu::Buffer,
    /// u32 storage: token ids in at `[0..M)`, top-K results out at `[0..2K)`.
    pub(crate) io: wgpu::Buffer,
    pub(crate) staging: wgpu::Buffer,
    pub(crate) readback: wgpu::Buffer,
    /// One bind group per weight window (indexed like `Model::windows`).
    pub(crate) bind_groups: Vec<wgpu::BindGroup>,
    #[cfg(all(feature = "bench-internals", not(feature = "ci")))]
    pub(crate) bench_query_set: wgpu::QuerySet,
    #[cfg(all(feature = "bench-internals", not(feature = "ci")))]
    pub(crate) bench_resolve: wgpu::Buffer,
    #[cfg(all(feature = "bench-internals", not(feature = "ci")))]
    pub(crate) bench_readback: wgpu::Buffer,
}

impl SessionState {
    fn new(model: &Model) -> Self {
        const IO_OUT_AND_PARTIALS: u64 =
            ((1 + TOPK_NUM_PARTIAL_WG as u64) * 2 * TOPK_MAX as u64) * 4;
        const IO_INPUT: u64 = M_MAX as u64 * 4;
        const IO_BYTES: u64 = if IO_INPUT > IO_OUT_AND_PARTIALS {
            IO_INPUT
        } else {
            IO_OUT_AND_PARTIALS
        }
        .next_power_of_two();
        const READBACK_BYTES: u64 = 2 * TOPK_MAX as u64 * 4;

        let device = &model.device;
        let buf = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(4).div_ceil(4) * 4,
                usage,
                mapped_at_creation: false,
            })
        };
        let rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let kv_k = buf("kv_k", model.kv.total_bytes, rw);
        let kv_v = buf("kv_v", model.kv.total_bytes, rw);
        let state = buf("gdn_state", model.state.total * 4, rw);
        let act = buf("act", u64::from(model.act.total) * 4, rw);
        let q8 = buf("q8", u64::from(model.q8.total_words) * 4, rw);
        let io = buf("io", IO_BYTES, rw);
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: STAGING_CHUNK,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        let readback = buf(
            "readback",
            READBACK_BYTES,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        let bg = |win: usize| {
            let rest: [&wgpu::Buffer; 8] = [
                &model.buffers.w_small,
                &act,
                &q8,
                &kv_k,
                &kv_v,
                &state,
                &io,
                &model.buffers.rope,
            ];
            let mut entries = vec![wgpu::BindGroupEntry {
                binding: 0,
                resource: model.window_binding(win),
            }];
            entries.extend(rest.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32 + 1,
                resource: b.as_entire_binding(),
            }));
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("session_bg"),
                layout: &model.bgl,
                entries: &entries,
            })
        };
        let bind_groups: Vec<wgpu::BindGroup> = (0..model.windows.len()).map(bg).collect();

        #[cfg(all(feature = "bench-internals", not(feature = "ci")))]
        let bench_query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("bench_qs"),
            ty: wgpu::QueryType::Timestamp,
            count: BENCH_QS_SLOTS,
        });
        #[cfg(all(feature = "bench-internals", not(feature = "ci")))]
        let bench_resolve = buf(
            "bench_resolve",
            u64::from(BENCH_QS_SLOTS) * 8,
            wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        );
        #[cfg(all(feature = "bench-internals", not(feature = "ci")))]
        let bench_readback = buf(
            "bench_readback",
            u64::from(BENCH_QS_SLOTS) * 8,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );

        Self {
            kv_k,
            kv_v,
            state,
            act,
            q8,
            io,
            staging,
            readback,
            bind_groups,
            #[cfg(all(feature = "bench-internals", not(feature = "ci")))]
            bench_query_set,
            #[cfg(all(feature = "bench-internals", not(feature = "ci")))]
            bench_resolve,
            #[cfg(all(feature = "bench-internals", not(feature = "ci")))]
            bench_readback,
        }
    }
}

/// Token sampler.
///
/// Filtering follows llama.cpp's order on the GPU-provided top-K candidates:
/// `top_k` → softmax → `top_p` → `min_p` (both on the un-tempered
/// distribution) → temperature → multinomial draw. `temperature == 0.0` (or
/// `top_k == Some(1)`) is argmax. `top_k` is capped at [`TOPK_MAX`] (32).
#[derive(Debug, Clone)]
pub struct Sampler {
    /// Logit temperature. `0.0` ⇒ argmax. Must be ≥ 0.
    pub temperature: f32,
    /// Keep the `k` most likely candidates. `None` ⇒ all 32 GPU candidates.
    pub top_k: Option<u32>,
    /// Nucleus filter: smallest prefix with cumulative probability ≥ `p`.
    pub top_p: Option<f32>,
    /// Drop candidates with probability < `min_p` × the top probability.
    /// `0.0` disables it (Bonsai-2 recommends 0.05 for thinking mode).
    pub min_p: f32,
    /// PRNG seed. Combined with the position, so reruns are reproducible.
    pub seed: u64,
}

impl Default for Sampler {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            min_p: 0.0,
            seed: 0,
        }
    }
}

/// Reason a `generate*` call returned.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum StopReason {
    /// Produced a stop token.
    Eos,
    /// Reached `max_new_tokens` without a stop token.
    MaxTokens,
}

/// Options for [`Session::generate`] / [`Session::generate_streaming`].
pub struct GenerateOptions<F: Fn(u32) -> bool = fn(u32) -> bool> {
    pub max_new_tokens: u32,
    /// Predicate run on each sampled token; `true` ends generation with
    /// [`StopReason::Eos`] before the token is delivered. `None` ⇒ stop on
    /// [`Model::stop_tokens`] (`<|im_end|>` and `<|endoftext|>`).
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
/// Holds the position cursor plus this session's GPU resources (KV cache of
/// the full-attention layers, recurrent state of the Gated-DeltaNet layers,
/// activation scratch). Sessions on one [`Model`] never alias.
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

    /// Reset to a fresh conversation. O(1): kernels treat the recurrent state
    /// as zero when processing position 0, and the KV cache is overwritten
    /// in place.
    pub const fn reset(&mut self) {
        self.pos = 0;
    }

    /// Copy the live conversation state (KV prefix `[0..pos)` of the
    /// full-attention layers plus the complete recurrent state) to host
    /// memory.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is lost or the readback fails.
    pub fn snapshot(&mut self) -> Result<KvSnapshot> {
        self.model.check_device()?;
        kv_snapshot::capture(self)
    }

    /// Replace this session's state with `snap` and set `pos = snap.pos()`.
    /// Recurrent state cannot be truncated, so a snapshot restores exactly the
    /// position it was captured at.
    ///
    /// # Errors
    ///
    /// Returns an error if `snap` does not match the model geometry or
    /// exceeds `max_seq_len()`.
    pub fn restore(&mut self, snap: &KvSnapshot) -> Result<()> {
        self.model.check_device()?;
        kv_snapshot::apply(self, snap)
    }

    const fn check_room(&self, n: u32) -> Result<()> {
        if self.pos + n > self.model.max_seq {
            return Err(PotError::ContextOverflow {
                pos: self.pos,
                n,
                max: self.model.max_seq,
            });
        }
        Ok(())
    }

    fn check_tokens(&self, tokens: &[u32]) -> Result<()> {
        if tokens.iter().any(|&t| t >= self.model.cfg.n_vocab) {
            return Err(PotError::Config("token id out of vocabulary range"));
        }
        Ok(())
    }

    /// Batched prefill. Advances `pos` by `tokens.len()` and returns the
    /// first sampled token. Works at any `pos`; prompts longer than
    /// `max_prefill_tokens()` are chunked transparently.
    ///
    /// # Errors
    ///
    /// Returns an error for empty input, out-of-range token ids, KV-cache
    /// overflow, or GPU failure.
    pub fn prefill(&mut self, tokens: &[u32], sampler: &Sampler) -> Result<u32> {
        self.model.check_device()?;
        if tokens.is_empty() {
            return Err(PotError::Config(
                "Session::prefill requires non-empty input",
            ));
        }
        self.check_tokens(tokens)?;
        self.check_room(tokens.len() as u32)?;
        let k = effective_k(sampler);
        let mut chosen = 0u32;
        for slice in tokens.chunks(self.model.max_prefill_tokens() as usize) {
            let pos_base = self.pos;
            let (logits, indices) =
                forward::prefill_topk(self, slice, pos_base, k, &mut forward::NoMarker)?;
            chosen = sample_from_topk(&logits, &indices, sampler, pos_base);
            self.pos += slice.len() as u32;
        }
        Ok(chosen)
    }

    /// Token-at-a-time prefill through the single-token path. Same result as
    /// [`Session::prefill`]; kept for parity tests and the CLI `--mode gen`.
    ///
    /// # Errors
    ///
    /// Returns an error for empty input, out-of-range token ids, KV-cache
    /// overflow, or GPU failure.
    #[doc(hidden)]
    pub fn prefill_one_at_a_time(&mut self, tokens: &[u32], sampler: &Sampler) -> Result<u32> {
        self.model.check_device()?;
        if tokens.is_empty() {
            return Err(PotError::Config(
                "Session::prefill requires non-empty input",
            ));
        }
        self.check_tokens(tokens)?;
        self.check_room(tokens.len() as u32)?;
        let k = effective_k(sampler);
        let pos_base = self.pos;
        let (logits, indices) = forward::prefill_token_loop_topk(self, tokens, pos_base, k)?;
        let chosen = sample_from_topk(&logits, &indices, sampler, pos_base);
        self.pos += tokens.len() as u32;
        Ok(chosen)
    }

    /// One decoding step on `token`. Advances `pos` by 1 and returns the next
    /// sampled token.
    ///
    /// # Errors
    ///
    /// Returns an error for an out-of-range token id, KV-cache overflow, or
    /// GPU failure.
    pub fn step(&mut self, token: u32, sampler: &Sampler) -> Result<u32> {
        self.model.check_device()?;
        self.check_tokens(&[token])?;
        self.check_room(1)?;
        let k = effective_k(sampler);
        let pos = self.pos;
        let (logits, indices) = forward::step_topk(self, token, pos, k)?;
        let chosen = sample_from_topk(&logits, &indices, sampler, pos);
        self.pos += 1;
        Ok(chosen)
    }

    /// Collect-mode generation. `first_token` is fed as the next input but is
    /// not included in the returned tokens.
    ///
    /// # Errors
    ///
    /// Returns an error on KV-cache overflow or GPU failure.
    pub fn generate<S: Fn(u32) -> bool>(
        &mut self,
        first_token: u32,
        opts: &GenerateOptions<S>,
    ) -> Result<(Vec<u32>, StopReason)> {
        let mut out = Vec::with_capacity(opts.max_new_tokens as usize);
        let stop = self.generate_streaming(first_token, opts, |id| out.push(id))?;
        Ok((out, stop))
    }

    /// Streaming generation. `on_token` fires once per emitted token (not
    /// including `first_token`); a stop token ends generation before delivery.
    /// The command buffer of step i+1 is encoded while the GPU runs step i.
    ///
    /// # Errors
    ///
    /// Returns an error on KV-cache overflow or GPU failure.
    pub fn generate_streaming<S: Fn(u32) -> bool, On: FnMut(u32)>(
        &mut self,
        first_token: u32,
        opts: &GenerateOptions<S>,
        mut on_token: On,
    ) -> Result<StopReason> {
        self.model.check_device()?;
        self.check_tokens(&[first_token])?;
        let default_stops = self.model.stop_tokens();
        let is_stop = |t: u32| {
            opts.stop_pred
                .as_ref()
                .map_or_else(|| default_stops.contains(&t), |p| p(t))
        };
        let max_new = opts.max_new_tokens;
        let max_seq = self.model.max_seq;
        let k = effective_k(&opts.sampler);
        if max_new == 0 {
            return Ok(StopReason::MaxTokens);
        }
        self.check_room(1)?;

        let prime_cb = build_step_topk_cb(self, self.pos, k);
        commit_sample_upload(self, bytemuck::bytes_of(&first_token));
        self.model.queue.submit(Some(prime_cb));

        for i in 0..max_new {
            let next_cb = if self.pos + 2 <= max_seq && i + 1 < max_new {
                Some(build_step_topk_cb(self, self.pos + 1, k))
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
    match s.top_k {
        Some(k) if k > 0 => k.min(TOPK_MAX),
        _ => TOPK_MAX,
    }
}

/// CPU side of the hybrid sampler. Inputs are the GPU top-K candidates
/// (logits descending, paired indices).
fn sample_from_topk(logits: &[f32], indices: &[u32], s: &Sampler, pos: u32) -> u32 {
    debug_assert_eq!(logits.len(), indices.len());
    let n = logits.len();
    if n == 0 {
        return 0;
    }
    if s.temperature <= 0.0 || s.top_k == Some(1) {
        return indices[0];
    }
    let kk = match s.top_k {
        Some(k) if (k as usize) < n => k as usize,
        _ => n,
    };

    // Un-tempered softmax for the top-p / min-p filters.
    let max_l = logits[0];
    let mut probs: Vec<f32> = logits[..kk].iter().map(|&l| (l - max_l).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return indices[0];
    }
    for p in &mut probs {
        *p /= sum;
    }
    let mut keep = kk;
    if let Some(top_p) = s.top_p {
        let cutoff = top_p.clamp(0.0, 1.0);
        let mut cum = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if cum >= cutoff {
                keep = i + 1;
                break;
            }
        }
    }
    if s.min_p > 0.0 {
        let thresh = s.min_p * probs[0];
        keep = probs[..keep]
            .iter()
            .take_while(|&&p| p >= thresh)
            .count()
            .max(1);
    }

    // Temperature on the survivors.
    let inv_t = 1.0 / s.temperature;
    let w: Vec<f32> = logits[..keep]
        .iter()
        .map(|&l| ((l - max_l) * inv_t).exp())
        .collect();
    let w_sum: f32 = w.iter().sum();
    if w_sum <= 0.0 || !w_sum.is_finite() {
        return indices[0];
    }
    let r = uniform_f32(s.seed.wrapping_add(u64::from(pos))) * w_sum;
    let mut acc = 0.0f32;
    for (i, &wi) in w.iter().enumerate() {
        acc += wi;
        if r < acc {
            return indices[i];
        }
    }
    indices[keep - 1]
}

/// `SplitMix64` → uniform f32 in [0, 1).
fn uniform_f32(seed: u64) -> f32 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let bits24 = (z >> 40) as u32;
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
        assert_eq!(
            sample_from_topk(&[3.0, 2.0, 1.0], &[42, 7, 99], &greedy(), 0),
            42
        );
    }

    #[test]
    fn greedy_top_k_one() {
        let s = Sampler {
            top_k: Some(1),
            ..Sampler::default()
        };
        assert_eq!(sample_from_topk(&[3.0, 2.0, 1.0], &[42, 7, 99], &s, 0), 42);
    }

    #[test]
    fn top_p_zero_picks_argmax() {
        for seed in 0..20u64 {
            let s = Sampler {
                top_p: Some(0.0),
                seed,
                ..Sampler::default()
            };
            assert_eq!(sample_from_topk(&[3.0, 2.0, 1.0], &[42, 7, 99], &s, 0), 42);
        }
    }

    #[test]
    fn min_p_filters_unlikely_candidates() {
        // p ∝ e^0, e^-1, e^-10: min_p 0.1 keeps the first two only.
        let logits = [0.0f32, -1.0, -10.0];
        let idx = [1u32, 2, 3];
        for seed in 0..200u64 {
            let s = Sampler {
                min_p: 0.1,
                seed,
                ..Sampler::default()
            };
            assert_ne!(sample_from_topk(&logits, &idx, &s, 0), 3);
        }
        let s = Sampler {
            min_p: 0.9,
            ..Sampler::default()
        };
        assert_eq!(sample_from_topk(&logits, &idx, &s, 0), 1);
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
        assert_eq!(r1, sample_from_topk(&logits, &indices, &s, 5));
        assert_ne!(r1, sample_from_topk(&logits, &indices, &s, 6));
    }

    #[test]
    fn empty_input_returns_zero() {
        assert_eq!(sample_from_topk(&[], &[], &Sampler::default(), 0), 0);
    }

    #[test]
    fn non_finite_falls_back_to_argmax() {
        let s = Sampler::default();
        assert_eq!(
            sample_from_topk(&[f32::INFINITY, 1.0, 0.0], &[42, 7, 99], &s, 0),
            42
        );
    }

    #[test]
    fn effective_k_caps_to_topk_max() {
        let k = |top_k| {
            effective_k(&Sampler {
                top_k,
                ..Sampler::default()
            })
        };
        assert_eq!(k(None), TOPK_MAX);
        assert_eq!(k(Some(0)), TOPK_MAX);
        assert_eq!(k(Some(100)), TOPK_MAX);
        assert_eq!(k(Some(5)), 5);
    }

    #[test]
    fn uniform_f32_in_range() {
        for seed in 0u64..1000 {
            assert!((0.0..1.0).contains(&uniform_f32(seed)));
        }
    }

    #[test]
    fn defaults() {
        let s = Sampler::default();
        assert_eq!(s.temperature, 1.0);
        assert_eq!((s.top_k, s.top_p, s.min_p, s.seed), (None, None, 0.0, 0));
        let o = GenerateOptions::default();
        assert_eq!(o.max_new_tokens, 32);
        assert!(o.stop_pred.is_none());
    }
}
