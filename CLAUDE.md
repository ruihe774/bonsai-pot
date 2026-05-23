# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`bonsai-pot` is a from-scratch, dependency-light **Bonsai / Qwen3-architecture inference engine** built on **wgpu compute shaders**. It supports Bonsai 4B and 8B models in **Q1_0** (binary), **Q2_0** (ternary, Vulkan only), and **Q8_0** (Vulkan only) quantization. There is no `llama.cpp`, `ggml`, or PyTorch on the hot path — weights are loaded from a custom flat-file layout (produced by `scripts/extract.py` from a GGUF), every kernel ships as a single hand-rolled GLSL source under `src/shaders/`, and the host side is plain Rust + wgpu 29. Both backends feed wgpu via `create_shader_module_passthrough`, so naga is bypassed entirely on the hot path. Build prerequisites: `glslangValidator` (>= 14, for `GL_EXT_integer_dot_product`) and `spirv-opt` on `$PATH`; the Apple build additionally needs `spirv-cross` to translate the SPIR-V to MSL.

The crate exposes both:
- a **library** (`bonsai_pot::{Model, Session, Sampler, GenerateOptions, …}`) for embedding the engine in other Rust programs;
- a **demo CLI** (`src/bin/bonsai-pot.rs`) that reads pre-tokenized u32 prompts from stdin and prints decoded output. The CLI bundles bench/microbench utilities behind the `bench-internals` feature.

Tokenization is intentionally out of the Rust crate. Use `scripts/bpe.py` to BPE-encode prompts; pipe its u32 output into the bin.

## Build / run / bench

```
# Library only:
cargo build --lib

# Demo CLI (pulls in the bench/microbench helpers):
cargo build --features bench-internals

# End-to-end run: tokenize then generate. `uv run` resolves the script's
# inline dependencies on the fly — no separate `pip install` step.
uv run scripts/bpe.py ./model "Once upon a time" \
  | cargo run --features bench-internals -- ./model \
        --mode prompt --max-new-tokens 64

# Benches and microbench don't need stdin:
cargo run --features bench-internals -- ./model --mode bench --pp 512 --tg 128
cargo run --features bench-internals -- ./model --mode microbench
```

`<model_dir>` is the output of `scripts/extract.py` (default `./model`). It must contain `config.ini`, the five `weights_*.bin` files, `vocab.bin`, `vocab_offsets.bin`, and (for the tokenizer) `merges.txt`. The runtime no longer reads `prompt.bin`; prompts come in over stdin from `scripts/bpe.py`.

Tests live in `tests/gpu_integration.rs` (end-to-end on a real GPU against `./model`) plus unit tests in `src/session.rs` (CPU sampler) and `src/kv_snapshot.rs` (header round-trips). Run with `cargo test`. Beyond that, `--mode gen` / `--mode prompt` plus parity diffs against captured baselines (and `examples/chat.rs`) are the correctness harness.

- `--mode gen` (default): single-token matvec path for both prompt and generation (Q1_0 hot path: `dotPacked4x8EXT` against Q8_0 acts on Vulkan, f16 ±-accumulate on Metal).
- `--mode prompt`: batched matmul prefill (`dot4I8Packed` matmul on Vulkan; on Metal the hand-ported matmul kernel materializes Q1_0 weights to fp16 just-in-time into threadgroup memory and runs the inner matmul on `simdgroup_matrix<half,8,8>` MMAs with fp32 accumulators), then matvec for generation.
- `--mode bench`: prints an `llama-bench`-style table with pp/tg t/s. (There used to be a `tg{N} pipe` row from CHUNK=8 pipelining; that path was removed when sampling moved off the GPU.)
- `--mode microbench`: per-kernel GPU-timestamp breakdown for a matmul prefill step (`--pp` tokens) followed by a tg step (KV pre-filled to `--tg` positions); shows us/call × calls/step for both paths.

CLI sampling flags: `--temperature`, `--top-k`, `--top-p`, `--seed`. Default is greedy (`--temperature 0.0`). Greedy runs are byte-deterministic; stochastic runs are reproducible per seed.

## Rebuilding the model directory

```
uv run scripts/extract.py path/to/Bonsai-4B-Q1_0.gguf --out ./model
uv run scripts/extract.py path/to/Bonsai-8B-Q1_0.gguf --out ./model-8b
# Ternary Bonsai (Q2_0, Vulkan only):
uv run scripts/extract.py path/to/Ternary-Bonsai-4B-Q2_0.gguf --out ./model-ternary-4b
# Qwen3 Q8_0 (Vulkan only):
uv run scripts/extract.py path/to/Qwen3-4B-Q8_0.gguf --out ./model-q8
```

Both Python scripts use [PEP 723 inline metadata](https://peps.python.org/pep-0723/) — `uv run` reads the dependency block at the top of each script and runs it in an isolated env. No virtualenv setup or `pip install` needed; just have `uv` on `$PATH`.

`scripts/extract.py` writes weights / vocab / `merges.txt` / `config.ini`. It does **not** encode prompts. To encode a prompt:

```
uv run scripts/bpe.py ./model "Once upon a time" --out ./model/prompt.bin
# or stream directly:
uv run scripts/bpe.py ./model "Once upon a time" | cargo run ...
```

`scripts/bpe.py` depends only on the `regex` package (for `\p{L}` / `\p{N}` in the Qwen2 pretokenizer regex) — no `gguf`, no GPU, no compilation. It reads `vocab.bin` / `vocab_offsets.bin` / `merges.txt` from the model dir. Dependencies are declared inline in PEP 723 format and resolved automatically by `uv run`. By default it splits the input on `<|...|>` literals that exist in the vocab (e.g. `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`) and emits each as its atomic token id, so ChatML-rendered prompts round-trip correctly. Pass `--no-specials` to byte-level-encode them instead.

## Architecture

### Files

- `src/lib.rs` — public API surface, re-exports.
- `src/model.rs` — config + manifest loading, GPU device/buffer/pipeline/BGL setup, RoPE table precompute, activation-buffer layout. Owns the public `Model`, `ModelConfig`, and `LoadOptions` types.
- `src/session.rs` — public `Session<'m>` (per-conversation state), `Sampler`, `GenerateOptions`, `StopReason`, and the CPU-side sampler (temperature → top-p → multinomial via SplitMix64-seeded uniform).
- `src/kv_snapshot.rs` — `KvSnapshot`: host-resident, persistable copy of the GPU KV cache for a `Session` at some `pos`. Used by `Session::snapshot` / `Session::restore`.
- `src/forward.rs` — entire forward pass and per-step inference helpers. Two end-to-end paths (matvec / matmul) plus encoder plumbing (immediates / push constants — there is no uniform-pool), plus a `bench_internals` submodule gated on the `bench-internals` feature. Long file (~1.5k lines), organized top-down: helpers → step encoders → matmul prefill → bench/microbench. The bench/microbench logic is factored into `src/bench.rs`, included via `#[path = "bench.rs"]` so it can reach `forward`'s private items through `super::`.
- `src/error.rs` — `PotError` + `Result` (built on `thiserror`).
- `src/decode.rs` — inverse of GPT-2 byte-level vocab encoding (codepoint → raw byte), used by `Model::decode_token`.
- `src/bin/bonsai-pot.rs` — demo CLI on top of the public lib API. Argv parsing, stdin u32 reader, sampler construction, calls into `Session`. Routes `--mode bench`/`microbench` to `bonsai_pot::__bench` (only available when built with `--features bench-internals`). Exposes `--max-seq` to size the KV cache.
- `src/shaders/*.comp` — one shader per kernel kind. The matvec/matmul shaders are the perf-critical ones; `topk_partial.comp` + `topk_merge.comp` are the multi-WG sampler reduction; `kv_writeback_fused.comp` does the K-side rms_norm + `*w_k_norm` + NEOX-RoPE + Q8_0 quantize + write into `kv_k`, plus V Q8_0 quantize + write into `kv_v`, all in one workgroup per (kv_head, token); `q_norm_rope_fused.comp` does the same Q-side rms_norm + `*w_q_norm` + NEOX-RoPE in one workgroup per (head, token), writing back to `act.q` in place; `matvec_q1_0_fused_normed.comp` folds `rms_norm(x) * w_norm` into the multi-range fused QKV/gate-up matvec (eliminates the `act.x_norm` round-trip on the matvec path); `matvec_q1_0_silu.comp` folds `silu(gate) * up` into the ffn_down matvec (eliminates the `act.ffn_in` round-trip on the matvec path); `attention_prefill_tiled.comp` is the Q-tiled + GQA-batched FA-2 prefill attention (Q_TILE=2 queries × 4 GQA Q-heads per WG, K/V loaded once per cache pos, fused Q8_0 output into `act_q8`); `rms_norm_q8_0.comp` and `silu_mul_q8_0.comp` fold their op + Q8_0 quantize into a single dispatch on the prefill path.
- `examples/chat.rs` — interactive ChatML REPL built on the public library API. Renders the Qwen-style `<|im_start|>...<|im_end|>` chat template per turn and shells out to `scripts/bpe.py` for tokenization. Both the system prompt and each user turn go through `Session::prefill`, which uses batched matmul (the matmul attention kernel scans `[0, pos_base + m_tok]` per query, so it works at any `pos`) and chunks transparently into `m_max`-sized batches when needed. The system-prompt KV state is snapshotted via `Session::snapshot`. Generation then streams with `Session::step` until `<|im_end|>`. `/reset` calls `Session::restore` on the system snapshot (~1–2 ms over PCIe), avoiding a full re-prefill. KV-cache capacity is configurable via `--max-seq`.
- `tests/gpu_integration.rs` — end-to-end tests that load `./model` on a real GPU: model-config sanity, vocab/decode round-trips, prefill error guards, KV snapshot/restore, greedy determinism, and matmul-vs-matvec parity.
- `scripts/extract.py` — GGUF → flat-file converter. Writes weights + vocab + merges + config.
- `scripts/bpe.py` — standalone BPE encoder; reads model dir, writes u32 token IDs.

### Weight formats and the matvec/matmul inner loops

Three quantization formats are supported, all sharing a 128-weight super-block layout. `scripts/extract.py` splits each tensor into a contiguous **d-array** (FP16 scales) followed by a **qs-array** (raw weight codes); the manifest in `config.ini` records `d_offset`, `qs_offset`, and `nb` (super-blocks per row) per tensor. Both halves are u32-aligned — all reads are word loads. The model-wide format is autodetected from the GGUF and stored as `quant_format` in `config.ini`.

- **Q1_0** — 16 bytes of sign bits per 128-weight block (1 bit/weight, ±1); 18 B/block. Sign-bit convention: bit=1 → +1, bit=0 → −1 (verifiable via `expand_4_bits`'s 4-byte spread). Supported on Vulkan and Metal.
- **Q2_0** (Ternary Bonsai) — 32 bytes of 2-bit codes per 128-weight block (ternary: −1, 0, +1); 34 B/block. Expanded by `expand_8_bits`. Vulkan only.
- **Q8_0** — four native 32-element GGML blocks per super-block (4 × 34 B = 136 B); each native block is 32 i8 weights + 2-byte FP16 scale. The d-array has `4*nb` entries; qs-array has `128*nb` bytes. Vulkan only.

Two inner-loop formulations live in the codebase, picked by backend:

- **Vulkan / GLSL** (AMD, NVIDIA, Intel): the matvec/matmul kernels stage activations as Q8_0 in shmem and accumulate via `dotPacked4x8EXT` (one hardware DP4a instruction per 4-element dot on AMD Vega+, NVIDIA Pascal+, Intel Gen12+). Q1_0 weights are expanded to ±1 packed-byte form by `expand_4_bits`; Q2_0 ternary weights are expanded to −1/0/+1 packed-byte form by `expand_8_bits`; Q8_0 weights are i8 values fed directly to `dot4I8Packed`. In all cases the scale multiply (`block_sum * d`) closes each block — no weights are ever materialized as float.
- **Apple / MSL**: Apple GPUs lack a hardware integer dot product (no DP4a / `dotPacked4x8` / `OpSDot` equivalent — confirmed against `philipturner/metal-benchmarks` and the WebGPU DP4a proposal `gpuweb/gpuweb#2677`; `IMUL32` is 4 cyc/lane on Apple Silicon). Two divergent inner-loop strategies live in the MSL counterparts, both selected via `#ifdef METAL_BACKEND` branches in the GLSL source that spirv-cross translates verbatim:
  - **Matvec** (`matvec_q1_0*.comp`, single-token path): skip the Q8_0 round-trip in shmem entirely and run an f32 `select(±a, cond)` accumulate per Q1_0 sign bit, with one f16 weight-scale multiply per block. Inner-loop ops drop from ~19 cyc per 4-element dot (4 IMUL32 + 3 ADD when expanded from `dot4i8packed`) to ~7 cyc fp ops. Empirically on M2 Pro / Bonsai-4B this was a ~3× e2e_tg lift over the Q8_0 + dot4i8packed baseline.
  - **Matmul** (`matmul_q1_0_q8_0.metal`, hand-ported, Q1_0 prefill path on Metal): a 64×64 tile / 256-thread kernel that materializes Q1_0 weights to fp16 just-in-time into threadgroup memory (`w_sh`), dequantizes Q8_0 activations to fp16 once into `a_sh`, and runs the inner matmul on `simdgroup_matrix<half,8,8>` MMA instructions with fp32 accumulators (per-WG simdgroup grid 2 × 4, per-simdgroup tile TM_SG=4 × TN_SG=2). M2 Pro and newer Apple GPUs lower this to the same f16 FMA pipe as a hand-rolled inner loop, but the `simdgroup_matrix` formulation lets the compiler schedule register tiling and broadcast reuse — equivalent in throughput to the way the Vulkan path uses `dotPacked4x8EXT` to expose hardware-accelerated MMAs to the scheduler. This is why the kernel is hand-ported rather than translated from GLSL: there is no GLSL surface for `simdgroup_matrix`.

`matvec_q1_0_fused_normed` packs 2- or 3-range dispatches (QKV; gate+up) into one workgroup to amortize x-load cost, and additionally folds `rms_norm(x) * w_norm` over the activation so there's no `act.x_norm` round-trip. On the MSL path, `inv_rms` is folded once at row-write (after the per-row 8-lane reduction); on the GLSL path it's folded into the per-Q8 sub-block scale.

### Two execution paths

The model is run in one of two regimes, selected by the call-site (`forward.rs` has both):

1. **Single-token (matvec) path** — `step_matvec_topk` / `encode_step_matvec` / `layer_pre_kv_in_pass` / `layer_post_kv_in_pass`. Used for **all of `--mode gen`** and for the generation phase of `--mode prompt`. Operates on `m=1` token; uses `matvec_q1_0`, `matvec_q1_0_silu`, and `matvec_q1_0_fused_normed`. The whole forward step (embed → 36× transformer layers → output_norm → LM head → topk_reduce) is encoded into **one** compute pass per step. Per-layer fusions on this path: (a) `matvec_q1_0_fused_normed` folds `rms_norm(x) * w_norm` into the QKV and gate+up matvecs (no `act.x_norm` traffic, no separate `rms_norm` dispatch); (b) `q_norm_rope_fused` folds Q's rms_norm + `*w_q_norm` + NEOX-RoPE in place into `act.q`; (c) `kv_writeback_fused` folds K's rms_norm + `*w_k_norm` + NEOX-RoPE + K/V Q8_0 quantize + cache write; (d) `matvec_q1_0_silu` folds `silu(gate) * up` into the ffn_down matvec (no `act.ffn_in` traffic, no separate `silu_mul` dispatch). The post-attention Wo and the ffn_down matvecs both use the matvec `accumulate=true` mode to fuse the residual add. After the GPU step, the CPU reads back up to `TOPK_MAX = 32` candidates from `sample[0..2K]` and performs CPU-side sampling. Attention in this path uses the split-K kernel pair (see below).

2. **Batched-prefill (matmul) path** — `prefill_matmul_topk` / `layer_step_matmul`. Used by `Session::prefill` and the bin's `--mode prompt`. Heavily fused: `rms_norm_q8_0` writes Q8_0 directly into `act_q8` (no `act.x_norm` round-trip, no separate quantize), `silu_mul_q8_0` does the same for `silu(gate)*up`, and `attention_prefill_tiled` (the Q-tiled FA-2 prefill kernel, see below) writes its output as Q8_0 to `act_q8` so the Wo matmul reads it directly. The `dot4I8Packed`-based `matmul_q1_0_q8_0.comp` is the projection kernel. `q_norm_rope_fused` and `kv_writeback_fused` handle the Q/K rms_norm + RoPE (and K/V Q8_0 quantize into the cache) for all M tokens, both pos_base-parameterized. `prefill_matmul_topk` caps `M` at `m_max=512`; `Session::prefill` chunks longer prompts into `m_max`-sized batches transparently. `Session::prefill_one_at_a_time` (matvec-loop variant) remains for callers that need single-token-at-a-time prefill.

These two paths use different shaders (`matvec_q1_0*` vs `matmul_q1_0_q8_0`) and different bind-group layouts (`bgls.matvec` vs `bgls.matmul`).

### Attention: split-K + GQA-batched flash-attention (matvec / tg path)

For tg (`m_tokens=1`), the matvec path uses a two-kernel split-K flash-attention pipeline that decouples per-step latency from KV length. `attention_split.comp` is dispatched as `(n_kv_head, n_chunks_active, 1)`; each workgroup processes one `(kv_group, chunk)` pair, scanning `[chunk * ATTN_CHUNK_SIZE, min((chunk+1) * ATTN_CHUNK_SIZE, pos))`. The four Q heads sharing the KV group are processed together so K/V loads are reused 4×; the four Q·K dots are packed into a single `vec4<f32>` and reduced with `subgroupAdd` (and a cross-subgroup merge if the device's subgroup is smaller than the workgroup, see below). Per-chunk `(m, l, o)` partials are written to `attn_partials`. `attention_merge.comp` then runs `(n_head, 1, 1)`, doing a flash-attention log-sum-exp combination across the active chunks. Both dispatches share the same compute pass — wgpu inserts the storage-buffer barrier between them. `ATTN_CHUNK_SIZE=8`; `n_chunks_active = ceil(pos / 8)`. Requires `Features::SUBGROUP`.

### Attention: Q-tiled + GQA-batched FA-2 prefill (matmul path)

For prefill (`m_tokens=M > 1`), `attention_prefill_tiled.comp` dispatches `(n_kv_head, ceil(M/Q_TILE), 1)` workgroups (WG=32, ELEMS_PER_THREAD=4 — head_dim=128 = 32 × 4). Each WG handles `Q_TILE=2` consecutive query tokens × `Q_PER_GROUP=4` GQA Q-heads sharing the KV head — so a single K[t] load is reused across **8 queries** within the WG. This is the bandwidth fix that prevented prefill from degrading quadratically: the previous (now-removed) `attention.comp` dispatched one WG per `(Q-head, query-token)` and re-loaded every K/V byte `n_head * M = 4 * 8 = 32×` more than necessary. The per-`t` Q·K reduction goes through `wg_sum_v4`; on hardware with `subgroup_size >= WG=32` it early-returns to a single `subgroupAdd`, on smaller subgroups it falls back to a smem-staged combine. The earlier WG=64 / EPT=2 layout split head_dim across two subgroups on wave32 hardware, *forcing* the smem combine and a pair of workgroup barriers per cache position; line-level NSight on Bonsai-4B / GB10 attributed ~30% of pp512 wall time to that combine machinery. Q_TILE=2 is the sweet spot: per-query state (Q registers, output accumulators, m/l) is held in small `array<…, Q_TILE>` / `array<…, Q_PER_GROUP>` arrays whose every access is inside a `[[unroll]]` for-loop (`GL_EXT_control_flow_attributes`). The unroll attribute makes the SPIR-V loop unroller (run by `spirv-opt -O` in build.rs) rewrite each index as a constant before the driver compiler sees it, so the arrays stay in registers — equivalent in codegen to a fully named-scalar manual unroll, but vastly more readable. (Dropping the `[[unroll]]` and letting indices stay dynamic does still cause NVIDIA to spill to local memory and the kernel becomes ~5× slower, so the attribute is load-bearing.) Q_TILE=4 was tried and regresses ~15% from register pressure. Per-query causal mask: query at m_idx attends to cache positions [0, pos_base + m_idx], with out-of-range positions scoring to NEG_INF. Output is **Q8_0 quantized inline** and written to `act_q8` (no f16 staging in `act.attn_out`); per-32-elem max-abs uses a `subgroupShuffleXor` butterfly with mask ≤ 16, contained within the 32-lane subgroup on both wave32 and wave64. The Wo matmul reads `act_q8` directly. This eliminates the per-layer `quantize_act(attn_out)` dispatch.

### Subgroup ops and the runtime SUBGROUP_MIN_SIZE bound

`rms_norm`, `attention_prefill_tiled`, `attention_split`, `attention_merge`, `rms_norm_q8_0`, and `kv_writeback_fused` use `subgroupAdd` / `subgroupMax` (or `subgroupShuffleXor` butterfly) with a runtime cross-subgroup merge when `num_subgroups > 1`. `attention_prefill_tiled` runs at WG=32 so on `subgroup_size >= 32` hardware (wave32, wave64, Apple, Intel Gen12+) `wg_sum_v4` early-returns to a single `subgroupAdd` — the merge path is dead code; on smaller subgroups (Intel Gen11, some Mali) WG=32 spans 2–4 subgroups and the smem-staged combine fires. The dead-code path costs ~5% on the fast-path kernel time on GB10 (driver can't fully fold the spec-constant conditional + still-present `barrier()` paths), traded for portability. The cross-subgroup shared-memory arrays are sized post-specialization for the **worst case**: `SG_PARTIAL_MAX = ceil(WG / SUBGROUP_MIN_SIZE)` slots, where `SUBGROUP_MIN_SIZE` is a Vulkan **specialization constant** (SpecId 0). At `Model::load` we patch the `OpSpecConstant` default in the SPIR-V to `adapter.get_info().subgroup_min_size` (wgpu's passthrough API doesn't expose `VkSpecializationInfo`, so we rewrite the default operand directly via `spirv_set_spec_const_u32`). `MAX_CHUNKS` (SpecId 1, sizes `attention_merge`'s weights_sh) and `N_EMBD_V4` (SpecId 2, sizes `matvec_q1_0_fused_normed`'s x_sh) are baked the same way. The actual per-dispatch subgroup count is read from `gl_NumSubgroups` and individual subgroup indices from `gl_SubgroupID` — no division by a compile-time constant.

This design is correct under `ALLOW_VARYING_SUBGROUP_SIZE` (which wgpu-hal sets unconditionally for all SUBGROUP-enabled pipelines): the size can differ across pipelines or even across dispatches of the same pipeline; the shmem is always large enough and all branching uses the runtime builtins. The WebGPU subgroups proposal requires either this flag or `VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT` to lock the size; wgpu 29 does not expose the latter, so we rely on the former.

`matvec_q1_0` and `matvec_q1_0_fused` use a `subgroupShuffleXor` butterfly for the per-row 8-lane reduction. This assumes `subgroup_invocation_id` increases linearly with `local_invocation_index` (true on AMD/NVIDIA/Intel/Apple) and `subgroup_size >= 8` so XOR mask 4 stays within a subgroup. The hard requirement `subgroup_min_size >= 8` is validated at `Model::load` against the adapter; hardware below this bound is rejected.

To exercise the wave32 / multi-subgroup merge path on AMD RDNA, run with `RADV_PERFTEST=cswave32`. The WG=32 rewrite was a wave32-only fix in mechanism: on wave64 hardware (RX 9070 default, `subgroup_size=64`) the previous WG=64 layout already had `gl_NumSubgroups == 1`, took the early-return path of `wg_sum_v4`, and never executed a `barrier()` in the inner loop. Any wave64 perf delta from this commit is therefore secondary — possibly RADV codegen finding a slightly different VGPR layout from the EPT=4 state shape, or the elimination of the now-dead spec-constant conditional — and within run-to-run noise on most measurements.

Bonsai-4B / RX 9070 (RDNA4, default wave64), post-rewrite reference numbers:

| test         | wall t/s | gpu t/s |
|--------------|---------:|--------:|
| pp512        |   ~2510  |  ~2532  |
| tg128        |   ~323   |  ~404   |
| e2e_pp632    |   ~2284  |    —    |
| e2e_tg128    |   ~385   |    —    |

(For reference, the pre-rewrite pp512 was ≈ 2477 with std ±42; the means here differ by less than one combined standard deviation. Note also that `tg128` exercises `attention_split` / `attention_merge`, not `attention_prefill_tiled`, so any tg128 delta across this commit is unrelated to the kernel that was actually changed.)

Bonsai-4B / GB10 (NVIDIA Blackwell, wave32), microbench step time at m=512, pos_base=512:

| state                                  | attn (ms) | step (ms) | pp512 t/s |
|----------------------------------------|----------:|----------:|----------:|
| pre-FA-2-prefill rewrite (historical)  |        —  |        —  |     ~460  |
| FA-2 prefill (WG=64, EPT=2)            |    199.94 |    368.23 |    ~1390  |
| WG=32, EPT=4 (single-sg fast path)     |     86.66 |    247.00 |    ~2073  |
| WG=32 + portable wg_sum_v4 (current)   |     84.6  |    243.8  |    ~2100  |

The WG=32 rewrite landed −56.6% on the attention kernel at m=512/pos_base=512; reintroducing the cross-subgroup fallback (so the kernel stays correct for subgroup_size < 32 hardware) gave back about 5% of the kernel time as dead-code overhead on the SG ≥ 32 fast path.

Two Apple inner-loop rewrites have landed in series. The first replaced the `dot4i8packed`-emulated matvec/matmul path with an f32 `select(±a, cond)` accumulate (no Q8_0 shmem round-trip), recovering ~3× on e2e_tg over the dot4i8packed baseline. The second rewrote `matmul_q1_0_q8_0.metal` around `simdgroup_matrix<half,8,8>` MMAs (with Q1_0 weights materialized to fp16 just-in-time into `w_sh`), which is the current state of the prefill matmul; the matvec / generation path is unchanged across that second rewrite.

### Apple / Metal backend

The Apple build path is structurally distinct enough to call out explicitly:

- **Shaders.** Every kernel has exactly one source: `src/shaders/{name}.comp` (GLSL). On Apple `build.rs` runs `glslangValidator -DMETAL_BACKEND=1` → `spirv-opt -O` → `spirv-cross --msl --msl-version 30000 --msl-fixed-subgroup-size 32 --msl-decoration-binding --rename-entry-point main cs_main` to produce `OUT_DIR/{name}.comp.msl`. The `#ifdef METAL_BACKEND` branches in the GLSL select Apple-specific kernel variants (e.g. f16 ±-accumulate inner loops in the matvec family vs `dotPacked4x8EXT` on Vulkan); `spirv-opt -O` is required so `[[unroll]]` annotations actually unroll before spirv-cross sees the SPIR-V — otherwise Apple's driver compiler spills register-resident arrays to local memory. The lone exception is `matmul_q1_0_q8_0.metal`, hand-ported around `simdgroup_matrix<half,8,8>` MMAs which have no GLSL surface; `build.rs` copies it verbatim into `OUT_DIR/matmul_q1_0_q8_0.comp.msl`. The `HAND_PORTED_MSL` list in `build.rs` is the canonical list. Both translated and hand-ported MSL feed `create_shader_module_passthrough`.
- **MSL slot fixups.** spirv-cross emits `[[buffer(N)]]` for SSBOs starting at `N=0` (because of `--msl-decoration-binding` plus our binding-numbered SPIR-V), but wgpu's Metal HAL puts the push-constant block at `[[buffer(0)]]`. `model::msl_shift_ssbo_buffer_indices` rewrites the `kernel void cs_main(...)` argument list at load time to shift every SSBO slot by +1; hand-ported MSL passes through unchanged (its slots are already laid out for wgpu).
- **Specialization constants.** `SUBGROUP_SIZE` (SpecId 0) is forced to `32` (Apple Silicon's simdgroup width is fixed); spirv-cross usually constant-folds it but emits `[[function_constant(0)]]` in cases it can't. `MAX_CHUNKS` (SpecId 1) and `K_V4` / `N_EMBD_V4` (SpecId 2) are also patched. wgpu's passthrough API doesn't expose `MTLFunctionConstantValues`, so the work is split in two: at *build* time, `msl_strip` (called from `build.rs`) walks the spirv-cross MSL via tree-sitter and normalises both spec-constant emission forms — it deletes the `#ifndef SPIRV_CROSS_CONSTANT_ID_<n> / #define / #endif` guard blocks, and rewrites any `constant uint NAME_tmp [[function_constant(N)]];` + `constant uint NAME = is_function_constant_defined(NAME_tmp) ? … : <default>u;` pair into a single `constant uint NAME = SPIRV_CROSS_CONSTANT_ID_<N>;` line (extracting `NAME` from the `_tmp` declarator and `N` from the attribute, so no slot-to-name table is needed). At *load* time, `model::msl_set_function_const_u32` simply prepends `#define SPIRV_CROSS_CONSTANT_ID_<n> <value>u` to the source — the preprocessor resolves the references that strip left in place.
- **Inner-loop divergence.** Apple GPUs have no DP4a / hardware integer dot product (see `memory/apple_gpu_no_dp4a.md`). The matvec MSL skips the Q8_0 shmem round-trip and runs f32 `select(±a, cond)` accumulate; the matmul MSL is the hand-ported `simdgroup_matrix<half,8,8>` kernel — see the "Weight formats" section above. Future kernel changes that touch the matvec inner loop should preserve the `#ifdef METAL_BACKEND` ±-accumulate pattern in the GLSL; matmul changes have to be made in `matmul_q1_0_q8_0.metal` directly. **Q2_0 and Q8_0 are not supported on Metal** — `Model::load` returns an error if those formats are used with the Apple backend.
- **Microbench.** Apple does not support `TIMESTAMP_QUERY_INSIDE_PASSES`, so the Apple microbench path uses one compute pass per labeled dispatch, with begin/end timestamps installed via `ComputePassDescriptor::timestamp_writes`. The single-pass `PassCtx` and per-dispatch `PerDispatchCtx` in `forward.rs` cover both backends behind one `DispatchCtx` trait. End-to-end pp/tg bench (`--mode bench`) uses single-pass timestamps and works identically on both backends.
- **Limits.** `SUBGROUP_SIZE_CONTROL` is not requested on Apple (Metal lacks it). The validated subgroup floor is still 8, satisfied by Apple's fixed 32. `max_immediate_size` is requested at ≥128 B as on Vulkan.

### Sampling: hybrid GPU top-K → CPU finish

There is no GPU argmax shader. After the LM-head matvec, a two-pass multi-WG top-K reduction over the full logits array writes the top-`TOPK_MAX=32` candidates to the `sample` buffer. Pass 1 (`topk_partial.comp`, WG=128) launches `TOPK_NUM_PARTIAL_WG=32` workgroups, each handling a `n_vocab / 32` slice: per-thread top-K_MAX min-heap → per-thread heap-sort to descending → in-WG bitonic merge tree across the WG's 128 threads → top-K_MAX of that slice written to a scratch slot in `sample`. Pass 2 (`topk_merge.comp`, WG=64, single workgroup) loads the 32 already-sorted-descending partial slots one-per-thread (upper 32 threads pad with `-inf`) and runs the same parallel pairwise bitonic merge tree to produce the global top-K. Skipping the per-thread heap-build/heap-sort phases in pass 2 is sound because pass 1 emitted each partial in sorted order. Output: K `f32` logits (descending) followed by K `u32` indices at `sample[0..2K]`. The single-WG version cost ~0.45 ms/step on RX 9070 (1 of 56 CUs busy, 64 KiB LDS); the multi-WG version is ~0.15 ms (32 CUs busy in pass 1, ~−67%).

The CPU then reads `sample[0..2K]` back and finishes sampling: temperature scale → softmax → top-p nucleus filter → multinomial via xorshift/SplitMix64 PRNG seeded by `(sampler.seed + pos)`. Implementation in `session.rs::sample_from_topk`. With `temperature == 0.0` this short-circuits to argmax over the K candidates — which is exact, because the global max is always `sample[0]`.

This design intentionally trades the old CHUNK=8 pipelined-gen path (which required sampling on-GPU to chain via `sample[]`) for sampler flexibility. The perf cost is ~22% of tg t/s vs the pipelined version (see commit history).

### GPU memory layout

All weights live in **5 storage buffers** grouped by role: `w_attn` (per-layer Wq/Wk/Wv/Wo), `w_ffn_gu` (Wgate/Wup), `w_ffn_d` (Wdown), `w_norms` (FP16 norm vectors), `w_embed` (token_embd, used as both embed and tied LM head). Tensor offsets within each buffer come from `cfg.manifest` — always look weights up via `model::tensor(cfg, name)` rather than hard-coding offsets.

`Buffers::act` is one f16 buffer with named regions (`ActLayout` in `model.rs`: `x`, `q`, `k_cur`, `v_cur`, `attn_out`, `gate`, `up`, `logits`). Sized for `M_MAX=512` tokens (the prefill batch size cap). The historical `x_norm` and `ffn_in` staging regions were removed when `rms_norm_q8_0` and `silu_mul_q8_0` started writing Q8_0 directly into `act_q8`; `attn_out` survives because the matvec/tg path's `attention_merge` still writes f16 there.

KV cache is split into `kv_k` and `kv_v`, both stored as **Q8_0** (32-element blocks: FP32 scale + 32 i8 quants ⇒ ~2.25 bytes/element, ~12.5% smaller than f16 and producing a `dot4I8Packed`-friendly load shape for the attention kernels). Each buffer has a contiguous d-section followed by a qs-section (see `kv_layer_offsets` / `kv_qs_byte_base` in `forward.rs`); helpers there compute per-layer offsets. Per-step K/V is quantized straight into the cache by `kv_writeback_fused.comp` — there is no f16 staging copy. Total per-buffer size is `n_layer * max_seq * kv_dim * 2.25` bytes (≈170 MB combined K+V at `max_seq = 1024`).

`max_seq` is **not** a compile-time constant — it's an allocate-time tunable (`LoadOptions::max_seq`, default 1024, exposed by both the bin and `examples/chat.rs` as `--max-seq`). The `attn_partials` buffer (split-K attention scratch, sized as `n_head * ceil(max_seq / ATTN_CHUNK_SIZE) * (head_dim + 2) * 4` bytes) and the RoPE table (`max_seq * head_dim * 2` bytes) scale with `max_seq`. `attention_merge.comp`'s `MAX_CHUNKS` specialization constant (SpecId 1) is patched to `ceil(max_seq / ATTN_CHUNK_SIZE)` at `Model::load` time (like SUBGROUP_MIN_SIZE), so there is no hard sequence-length cap in the shader. VRAM is the practical limit; `Model::load` checks the KV buffer against the adapter's `max_buffer_size` and returns a clean error if `max_seq` is too large. Note: the engine has no YaRN/NTK RoPE scaling, so output quality degrades for `max_seq` significantly beyond `~2 × rope_orig_context` (≈16k for 4B, ≈32k for 8B).

`act_q8` is the Q8_0 activation scratch used only on the matmul path (FP32 d-section followed by i8 qs-section).

### Per-dispatch params via immediates

Every dispatch passes its `Params` struct as wgpu immediates (push constants): each dispatch helper in `forward.rs` calls `pass.set_immediates(0, bytemuck::bytes_of(&p))`. Each pipeline is created with an `immediate_size` field baked into its `PipelineLayoutDescriptor` (`mk_pipe` in `model.rs`). The device limit `max_immediate_size` is requested at ≥ 128 bytes. All `Params` structs are ≤ 64 bytes; this is enforced by the unit test `params_struct_sizes_fit_immediate_limit` in `model.rs`. BGLs contain only storage bindings (starting at binding 0) — there is no UBO.

### Bind-group layout discipline

Activation buffers always go through **one** `read_write` storage binding per bind group — never aliased as both `read` and `read_write` within a single dispatch. This is enforced by the bind-group construction in `model.rs` (the `rw_mask` argument to `make_bgl`); when adding a kernel, follow the same pattern.

### Encoder organization (perf-critical)

`begin_compute_pass` costs ~25us on RADV. The `_in_pass` family of helpers (`dispatch_rms_norm`, `dispatch_matvec_q1_0`, `dispatch_matvec_q1_0_fused_normed`, `dispatch_matvec_q1_0_silu`, `dispatch_kv_writeback_fused`, `dispatch_q_norm_rope_fused`, `dispatch_topk_reduce`) all accept a caller-provided `&mut wgpu::ComputePass<'_>` so many dispatches share one pass. The matvec generation step is encoded as **a single** big pass with **8 dispatches per layer** — see `encode_step_matvec`. (Historical note: an earlier version split it in two around a `copy_buffer_to_buffer` for the K/V cache write; that copy was replaced by the kv_writeback kernel, then expanded into `kv_writeback_fused` which folds in K's rms_norm + RoPE, then `q_norm_rope_fused` collapsed the Q-side rms_norm + RoPE, then `matvec_q1_0_fused_normed` absorbed the two per-layer rms_norm dispatches feeding fused QKV / gate+up, and finally `matvec_q1_0_silu` absorbed the `silu_mul` dispatch into ffn_down — eliminating the pass break and dropping per-layer dispatches from ~14 to 8.)

The matmul prefill path was historically less fused — it staged through `act.x_norm` and `act.ffn_in` so the activation could be Q8_0 quantized before the matmul. After the FA-2 prefill rewrite, all four `quantize_q8_0` dispatches per layer were eliminated: `rms_norm_q8_0` (writes Q8_0 to `act_q8` directly, replaces `rms_norm` + `quantize_q8_0` for both attn_norm and ffn_norm), `silu_mul_q8_0` (replaces `silu_mul` + `quantize_q8_0` for the ffn_down input), and the new `attention_prefill_tiled` (writes Q8_0 attn_out directly, eliminating the post-attention `quantize_q8_0`). Per-layer prefill dispatches dropped from 17 to 13. The legacy `attention.comp`, `silu_mul.comp`, and `quantize_q8_0.comp` were deleted entirely.

When adding a new dispatch in tg, prefer the `_in_pass` form and slot it into an existing pass. When adding to prefill, the per-pass wrapper is fine.

### Tied vs. untied embeddings

Bonsai 4B has **tied** embeddings: `token_embd.weight` is used both for the embedding lookup (in `embed.comp`) and for the LM head full matvec. Bonsai 8B has **untied** embeddings — it ships a separate `output.weight` tensor. Both layouts share the same `[n_embd, n_vocab]` Q1_0 row shape (gather a single row for embed, full matvec for LM head), so the kernels and the `w_embed` storage buffer are identical; only the byte offsets differ.

`scripts/extract.py` packs `token_embd.weight` (and, when not tied, `output.weight`) consecutively into `weights_embed_lmhead.bin` and sets `cfg.tied_embeddings`. `OutputTensors` in `model.rs` then exposes both `token_embd_*` (used by the embed kernel) and `lm_head_*` (used by the LM head matvec); when tied they coincide.

### Sample / readback layout

`buffers.sample` is a 1024-u32 storage buffer used in two roles within a single step:
1. **Input** during embed: CPU writes the input token ID to `sample[0]` (or all M prompt token IDs to `sample[0..M]` for matmul prefill).
2. **Output** from the topk merge: K f32 logits at `sample[0..K]` (bitcast to u32) followed by K u32 vocab indices at `sample[K..2K]`. A scratch region at `sample[2K..2K + TOPK_NUM_PARTIAL_WG * 2K]` holds the pass-1 partials in flight (overwritten by pass 2 before readback).

Since embed runs before topk_partial in any step, the two roles never alias. `buffers.readback` is the matching `MAP_READ` buffer; per-step generation does one `device.poll` + `map_async` (the CPU sampler can't run until the readback completes — there's no pipelining anymore).

### Adapter / limits

`Model::load` raises `max_storage_buffer_binding_size` and `max_buffer_size` to a minimum of 300 MB so the largest grouped weight buffer fits (~252 MB at 4B, ~510 MB at 8B). No upper cap is imposed — the adapter's natural limit is used, which on desktop GPUs is typically 2–4 GB (needed for 8B KV at 32k ctx ≈ 1.27 GB per buffer). `max_storage_buffers_per_shader_stage` is bumped to ≥ 8.

## When making changes

- **Adding a new kernel**: write `shaders/foo.comp` (a single GLSL source covers both backends — Vulkan compiles it to SPIR-V, Apple translates that SPIR-V to MSL via spirv-cross). Use `#ifdef METAL_BACKEND` for any Apple-specific inner-loop variants. If the kernel needs Apple-only intrinsics that have no GLSL surface (e.g. `simdgroup_matrix`), add `shaders/foo.metal` and add the basename to `HAND_PORTED_MSL` in `build.rs`. Add the basename to `build.rs`'s `SHADERS` array regardless. Then `load_shader!("foo", spec_consts, (wg_x, wg_y, wg_z))` in `model.rs` (pass `no_spec` if it doesn't reference SpecId 0/1/2; the `wg` tuple is required by the MSL passthrough path and ignored by the SPIR-V one). Add a `FooParams` struct (≤ 64 bytes, `Pod + Zeroable + repr(C)`) in `model.rs`, register a BGL in `BindGroupLayouts` (single-rw discipline above), build a pipeline in `Pipelines`. Then add a `dispatch_foo` helper in `forward.rs` and route it through `ctx.dispatch("foo_label", |pass| dispatch_foo(...))` at each call site.
- **Modifying weight layout**: changes to Q1_0/Q2_0/Q8_0 packing or to the grouping of tensors into the 5 buffers must be made in **both** `scripts/extract.py` (writer) and `model.rs` / shaders (reader), and the manifest format in `config.ini` will need to round-trip. Re-extract the model dir after any layout change.
- **Modifying the tokenizer / pretok regex**: changes to vocab encoding must be made in **both** `scripts/extract.py` (which writes `vocab.bin` + `merges.txt`) and `scripts/bpe.py` (which encodes prompts) and possibly `src/decode.rs` (which inverts the byte-level mapping). The Rust runtime never tokenizes, only decodes.
- **Public API changes**: re-exports live in `src/lib.rs`. Anything not re-exported there is internal and may change without notice; keep `Model`, `ModelConfig`, `LoadOptions`, `Session`, `Sampler`, `GenerateOptions`, `StopReason`, `KvSnapshot`, `PotError`, `Result`, `TOPK_MAX` stable.
- **Perf work**: use `--mode microbench` for per-kernel deltas; use `--mode bench` (specifically `tg{N}` and `pp{N}`) for end-to-end. Most tg time is in matvec dispatches (LM head, fused-normed QKV, fused-normed gate+up, attn_output Wo, ffn_down with silu fold). LM head is ~0.25 ms/step on 4B; the two-pass topk is ~0.12 ms. Attention, q_norm_rope_fused, kv_writeback_fused are minor. For prefill, attention is the dominant kernel and scales O(M²) per chunk (inherent); `attention_prefill_tiled`'s Q-tiling + GQA-batching + fused Q8_0 output is the main lever — keep Q_TILE=2 and the manual-unrolled state layout when touching it (Q_TILE=4 regresses on Blackwell GB10 from register pressure; Q_TILE=8 spills catastrophically). Recent commits document the wins from multi-row matvec, fused QKV/gate-up, rms_norm-into-matvec fusion, silu-into-ffn_down fusion, single-pass-per-phase, flash-attn online softmax, multi-WG topk, FA-2 prefill, prefill activation-quantize fusion, single-subgroup (WG=32) attention prefill, and (in earlier history) pipelined generation — keep these intact when refactoring.
