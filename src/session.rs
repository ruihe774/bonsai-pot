use crate::error::{PotError, Result};
use crate::forward::{
    self, build_step_matvec_topk_cb, build_step_matvec_topk_cb_deferred, wait_topk_readback,
};
use crate::kv_snapshot::{self, KvSnapshot};
use crate::model::{Model, TOPK_MAX};

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

/// Per-conversation inference state. Carries the current KV-cache cursor.
///
/// # Safety contract: drive sequentially
///
/// A `Session` borrows the [`Model`] *immutably*, so the borrow checker will
/// happily let you mint two `Session`s against the same `Model` and use them
/// in alternation. **Don't.** All sessions on a given `Model` share the same
/// GPU buffers — the KV cache, the activation scratch, and the `sample` /
/// `readback` buffers are owned by the `Model`, not the `Session`. Interleaving
/// calls across sessions will silently corrupt each other's state (a `step` on
/// session B overwrites the activations and KV slots session A's next call
/// expects to find intact).
///
/// The shared (`&Model`) borrow is deliberate: it allows `Model` methods like
/// [`Model::is_device_lost`] / [`Model::decode_token`] to be called while a
/// session is alive, and supports the device-lost test pattern of holding a
/// session, destroying the device on the model, and observing the session's
/// methods returning `DeviceLost`. Type-enforcing exclusivity (`&mut Model` on
/// session creation) would forbid those patterns. The cost of that choice is
/// this contract: **only one session per `Model` may be in active use at a
/// time.** You may freely [`reset`](Self::reset) and reuse a session, or open
/// a fresh one once the previous is dropped.
pub struct Session<'m> {
    model: &'m Model,
    pos: u32,
}

impl<'m> Session<'m> {
    pub(crate) const fn new(model: &'m Model) -> Self {
        Self { model, pos: 0 }
    }

    /// Current position (number of tokens consumed so far).
    #[must_use]
    pub const fn pos(&self) -> u32 {
        self.pos
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
        kv_snapshot::capture(self.model, self.pos)
    }

    /// Replace the GPU KV cache with `snap`'s contents and set
    /// `self.pos = snap.pos()`.
    ///
    /// Validates `snap` against the model (matching `n_layer` / `kv_dim`;
    /// `snap.pos() ≤ model.max_seq_len()`). After a successful restore,
    /// the next call should be [`prefill_one_at_a_time`](Self::prefill_one_at_a_time)
    /// or [`step`](Self::step) if `snap.pos() > 0`, or [`prefill`](Self::prefill)
    /// if `snap.pos() == 0`.
    ///
    /// Cost: one staging-belt-backed upload (all layers in one encoder),
    /// serialized against subsequent dispatches by queue ordering.
    ///
    /// # Errors
    ///
    /// Returns an error if `snap` does not match the model's `n_layer` /
    /// `kv_dim`, or if `snap.pos()` exceeds `model.max_seq_len()`.
    pub fn restore(&mut self, snap: &KvSnapshot) -> Result<()> {
        self.model.check_device()?;
        kv_snapshot::apply(self.model, snap)?;
        self.pos = snap.pos();
        Ok(())
    }

    /// Batched matmul prefill. Advances `pos` by `tokens.len()` and returns
    /// the first sampled token (drawn from the last logits via `sampler`).
    ///
    /// Requires `self.pos() == 0` because the matmul attention shader assumes
    /// a fresh KV cache. For incremental prefill into an existing context, use
    /// [`prefill_one_at_a_time`](Self::prefill_one_at_a_time).
    ///
    /// # Errors
    ///
    /// Returns an error if `self.pos() != 0`, if the prompt would overflow the
    /// KV cache, or if the underlying GPU dispatch fails.
    pub fn prefill(&mut self, tokens: &[u32], sampler: &Sampler) -> Result<u32> {
        self.model.check_device()?;
        if self.pos != 0 {
            return Err(PotError::Config(
                "Session::prefill requires pos == 0; use prefill_one_at_a_time for incremental prefill",
            ));
        }
        let n = tokens.len() as u32;
        if self.pos + n > self.model.max_seq {
            return Err(PotError::ContextOverflow {
                pos: self.pos,
                n,
                max: self.model.max_seq,
            });
        }
        let k = effective_k(sampler);
        let (logits, indices) =
            forward::prefill_matmul_topk(self.model, tokens, self.pos, k, &mut forward::NoMarker)?;
        let chosen = sample_from_topk(&logits, &indices, sampler, self.pos);
        self.pos += n;
        Ok(chosen)
    }

    /// Single-token-at-a-time matvec prefill. Slower than
    /// [`Session::prefill`] for long prompts, but supports any `pos`
    /// (incremental prefill into an existing context).
    ///
    /// # Errors
    ///
    /// Returns an error if the prompt would overflow the KV cache, or if the
    /// underlying GPU dispatch fails.
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
        let (logits, indices) = forward::prefill_matvec_loop_topk(self.model, tokens, self.pos, k)?;
        let chosen = sample_from_topk(&logits, &indices, sampler, self.pos);
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
        let (logits, indices) = forward::step_matvec_topk(self.model, token, self.pos, k)?;
        let chosen = sample_from_topk(&logits, &indices, sampler, self.pos);
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

        // --- prime step 0: token write + step in one CB ---
        let model = self.model;
        let (prime_cb, mut cur_slot) = build_step_matvec_topk_cb(model, first_token, self.pos, k);
        model.belt_finish();
        model.queue.submit(Some(prime_cb));
        model.belt_recall();

        for i in 0..max_new {
            // Pre-encode the next step's CB while the GPU drains the current
            // one. The staged copy_buffer_to_buffer for `sample[0]` is encoded
            // now (allocated slot in the staging chunk), but the token bytes are
            // written in later once sampling produces the chosen value.
            let next = if self.pos + 2 <= max_seq && i + 1 < max_new {
                Some(build_step_matvec_topk_cb_deferred(model, self.pos + 1, k))
            } else {
                None
            };

            let (logits, indices) = wait_topk_readback(model, k, cur_slot)?;
            let chosen = sample_from_topk(&logits, &indices, &opts.sampler, self.pos);
            self.pos += 1;

            if is_stop(chosen) {
                // Discard the pre-built CB and clean up the staging chunk.
                if let Some((_cb, _slot, view)) = next {
                    drop(view);
                    model.belt_finish();
                    model.belt_recall();
                }
                return Ok(StopReason::Eos);
            }
            on_token(chosen);

            if i + 1 == max_new {
                if let Some((_cb, _slot, view)) = next {
                    drop(view);
                    model.belt_finish();
                    model.belt_recall();
                }
                return Ok(StopReason::MaxTokens);
            }
            let Some((cb, slot, mut view)) = next else {
                return Err(PotError::ContextOverflow {
                    pos: self.pos,
                    n: 1,
                    max: max_seq,
                });
            };
            // Fill the pre-allocated staging slot with the chosen token. Must
            // drop the view before belt_finish() (which calls buffer.unmap()).
            view.copy_from_slice(bytemuck::bytes_of(&chosen));
            drop(view);
            model.belt_finish();
            model.queue.submit(Some(cb));
            model.belt_recall();
            cur_slot = slot;
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
