# CLAUDE.md

## What this is

`bonsai-pot` is a from-scratch **Bonsai / Qwen3-architecture inference engine** on **wgpu compute shaders**. Supports Bonsai 4B/8B in Q1_0 (binary), Q2_0 (ternary, Vulkan only), Q8_0 (Vulkan only). No llama.cpp/ggml/PyTorch — weights are loaded from a custom flat-file layout (produced by `scripts/extract.py`), every kernel is hand-rolled GLSL under `src/shaders/`, host side is Rust + wgpu 29. Both backends feed wgpu via `create_shader_module_passthrough` (naga bypassed). Build prereqs: `glslangValidator` ≥ 14 and `spirv-opt` on `$PATH`; Apple additionally needs `spirv-cross`.

Exposes a **library** (`bonsai_pot::{Model, Session, Sampler, GenerateOptions, …}`) and a **demo CLI** (`src/bin/bonsai-pot.rs`). Tokenization is out-of-crate — use `scripts/bpe.py` and pipe u32 output into the bin.

## Build / run / bench

```
cargo build --lib
cargo build --features bench-internals

uv run scripts/bpe.py ./model "Once upon a time" \
  | cargo run --features bench-internals -- ./model --mode prompt --max-new-tokens 64

cargo run --features bench-internals -- ./model --mode bench --pp 512 --tg 128
cargo run --features bench-internals -- ./model --mode microbench
```

Model dir must contain: `config.ini`, five `weights_*.bin` files, `vocab.bin`, `vocab_offsets.bin`, `merges.txt`. Prompts come via stdin from `scripts/bpe.py`; `prompt.bin` is no longer read.

Modes: `--mode gen` (default, matvec for prompt+gen), `--mode prompt` (matmul prefill + matvec gen), `--mode bench` (pp/tg t/s table), `--mode microbench` (per-kernel GPU-timestamp breakdown). Sampling flags: `--temperature`, `--top-k`, `--top-p`, `--seed`. Default greedy (`--temperature 0.0`).

Tests: `tests/gpu_integration.rs` (end-to-end GPU), unit tests in `src/session.rs` and `src/kv_snapshot.rs`. Run with `cargo test`.

## Rebuilding the model directory

```
uv run scripts/extract.py path/to/Bonsai-4B-Q1_0.gguf --out ./model
uv run scripts/extract.py path/to/Bonsai-8B-Q1_0.gguf --out ./model-8b
uv run scripts/extract.py path/to/Ternary-Bonsai-4B-Q2_0.gguf --out ./model-ternary-4b
uv run scripts/extract.py path/to/Qwen3-4B-Q8_0.gguf --out ./model-q8
```

PEP 723 inline deps — `uv run` resolves everything; no virtualenv needed. `scripts/bpe.py` splits on `<|...|>` special tokens by default; pass `--no-specials` to byte-level-encode them.

## Architecture

### Files

- `src/lib.rs` — public API re-exports.
- `src/model.rs` — config/manifest loading, GPU device/buffer/pipeline/BGL setup, RoPE table, activation-buffer layout. Owns `Model`, `ModelConfig`, `LoadOptions`.
- `src/session.rs` — `Session<'m>`, `Sampler`, `GenerateOptions`, `StopReason`, CPU-side sampler (temperature → top-p → multinomial, SplitMix64 PRNG).
- `src/kv_snapshot.rs` — `KvSnapshot`: host-resident KV cache copy. Used by `Session::snapshot` / `Session::restore`.
- `src/forward.rs` — entire forward pass (~1.5k lines). Both execution paths plus encoder plumbing. `bench_internals` submodule included via `#[path = "bench.rs"]`.
- `src/error.rs` — `PotError` + `Result`.
- `src/decode.rs` — GPT-2 byte-level vocab inverse (used by `Model::decode_token`).
- `src/bin/bonsai-pot.rs` — demo CLI. Argv parsing, stdin u32 reader, routes `--mode bench`/`microbench` to `bonsai_pot::__bench`.
- `src/shaders/*.comp` — one GLSL source per kernel. Perf-critical: `matvec_q1_0*.comp`, `matmul_q1_0_q8_0.comp`. Fused kernels: `matvec_q1_0_fused_normed.comp` (folds rms_norm into QKV/gate-up matvec), `matvec_q1_0_silu.comp` (folds silu into ffn_down), `kv_writeback_fused.comp` (K rms_norm + RoPE + Q8_0 quantize + cache write + V Q8_0), `q_norm_rope_fused.comp` (Q rms_norm + RoPE), `attention_prefill_tiled.comp` (FA-2 prefill, Q_TILE=2), `rms_norm_q8_0.comp`, `silu_mul_q8_0.comp`, `topk_partial.comp` + `topk_merge.comp`.
- `examples/chat.rs` — interactive ChatML REPL. Uses `Session::prefill` + `Session::step`; snapshots system prompt KV via `Session::snapshot`; `/reset` restores without re-prefill.
- `tests/gpu_integration.rs` — end-to-end GPU tests against `./model`.
- `scripts/extract.py` — GGUF → flat-file converter.
- `scripts/bpe.py` — standalone BPE encoder.

### Weight formats

All formats share 128-weight super-block layout. `scripts/extract.py` splits each tensor into a **d-array** (FP16 scales) and **qs-array** (weight codes); `config.ini` manifest records `d_offset`, `qs_offset`, `nb` per tensor.

- **Q1_0** — 16 bytes of sign bits per 128-weight block (±1, 18 B/block). bit=1 → +1, bit=0 → −1. Vulkan + Metal.
- **Q2_0** — 32 bytes of 2-bit codes (ternary −1/0/+1, 34 B/block). Vulkan only.
- **Q8_0** — 4 native 32-element GGML blocks per super-block (136 B/block; 32 i8 + 2-byte FP16 scale each). Vulkan only.

**Vulkan inner loop**: activations staged as Q8_0 in shmem, accumulated via `dotPacked4x8EXT`. Q1_0 expanded to ±1 packed bytes by `expand_4_bits`; Q2_0 by `expand_8_bits`; Q8_0 fed directly. Scale multiply closes each block.

**Metal inner loop**: Apple lacks DP4a. Matvec uses f32 `select(±a, cond)` accumulate per Q1_0 sign bit — no Q8_0 round-trip, ~3× faster than dot4i8packed emulation. Matmul uses hand-ported `matmul_q1_0_q8_0.metal` with `simdgroup_matrix<half,8,8>` MMAs (Q1_0 weights materialized to fp16 just-in-time into threadgroup memory). Selected via `#ifdef METAL_BACKEND` branches in GLSL.

### Two execution paths

1. **Single-token (matvec) path** — `step_matvec_topk` / `encode_step_matvec`. Used for all `--mode gen` and the generation phase of `--mode prompt`. Operates on m=1 token. Whole forward step encoded into **one** compute pass with **8 dispatches per layer**. Fusions: `matvec_q1_0_fused_normed` folds rms_norm into QKV/gate-up; `q_norm_rope_fused` handles Q rms_norm + RoPE; `kv_writeback_fused` handles K rms_norm + RoPE + K/V Q8_0 cache write; `matvec_q1_0_silu` folds silu into ffn_down. Wo and ffn_down use `accumulate=true` to fuse residual add. Attention uses split-K kernel pair.

2. **Batched-prefill (matmul) path** — `prefill_matmul_topk` / `layer_step_matmul`. Used by `Session::prefill` and `--mode prompt`. `rms_norm_q8_0` writes Q8_0 directly into `act_q8`; `silu_mul_q8_0` does the same for silu(gate)\*up; `attention_prefill_tiled` writes Q8_0 attn_out directly. Caps M at `m_max=512`; `Session::prefill` chunks transparently. Per-layer dispatches: 13.

Different shaders (`matvec_q1_0*` vs `matmul_q1_0_q8_0`) and different BGLs (`bgls.matvec` vs `bgls.matmul`).

### Attention kernels

**tg path (split-K)**: `attention_split.comp` dispatched `(n_kv_head, n_chunks_active, 1)`. Each WG scans one `(kv_group, chunk)` pair; 4 GQA Q-heads share K/V loads. Per-chunk (m, l, o) partials → `attn_partials`. `attention_merge.comp` combines across chunks with flash-attention LSE. `ATTN_CHUNK_SIZE=8`. Requires `Features::SUBGROUP`.

**prefill path (FA-2 tiled)**: `attention_prefill_tiled.comp` dispatched `(n_kv_head, ceil(M/Q_TILE), 1)`. WG=32, ELEMS_PER_THREAD=4. Each WG: Q_TILE=2 query tokens × Q_PER_GROUP=4 GQA Q-heads — one K[t] load reused across 8 queries. Q_TILE=2 is load-bearing: arrays stay in registers under `[[unroll]]` + `spirv-opt -O`. **Do not increase Q_TILE** (Q_TILE=4 regresses ~15% on GB10; Q_TILE=8 spills catastrophically). Output is Q8_0 quantized inline into `act_q8`.

### Subgroup ops and SUBGROUP_MIN_SIZE

Kernels using `subgroupAdd`/`subgroupMax`/`subgroupShuffleXor`: `rms_norm`, `attention_prefill_tiled`, `attention_split`, `attention_merge`, `rms_norm_q8_0`, `kv_writeback_fused`. All handle multi-subgroup merge via smem when `num_subgroups > 1` (triggered on Intel Gen11, some Mali at WG=32).

`SUBGROUP_MIN_SIZE` (SpecId 0), `MAX_CHUNKS` (SpecId 1), `N_EMBD_V4` (SpecId 2) are specialization constants patched at `Model::load` by rewriting `OpSpecConstant` defaults directly in SPIR-V via `spirv_set_spec_const_u32`. Hard requirement: `subgroup_min_size >= 8`, validated at load. To exercise wave32 path on AMD: `RADV_PERFTEST=cswave32`.

### Apple / Metal backend

- **Shader translation**: `build.rs` runs `glslangValidator -DMETAL_BACKEND=1` → `spirv-opt -O` → `spirv-cross --msl --msl-version 30000 --msl-fixed-subgroup-size 32 --msl-decoration-binding --rename-entry-point main cs_main`. `spirv-opt -O` is required so `[[unroll]]` unrolls before spirv-cross (otherwise driver spills to local memory). Exception: `matmul_q1_0_q8_0.metal` is hand-ported (no GLSL surface for `simdgroup_matrix`); listed in `HAND_PORTED_MSL` in `build.rs`.
- **MSL slot fixups**: wgpu Metal HAL puts push-constant at `[[buffer(0)]]`, but spirv-cross emits SSBOs starting at N=0. `msl_shift_ssbo_buffer_indices` shifts every SSBO slot +1 at load time. Hand-ported MSL slots are already correct.
- **Spec constants on Metal**: at build time, `msl_strip` (tree-sitter, called from `build.rs`) normalizes spirv-cross spec-constant emission to `constant uint NAME = SPIRV_CROSS_CONSTANT_ID_<N>;`. At load time, `msl_set_function_const_u32` prepends `#define SPIRV_CROSS_CONSTANT_ID_<n> <value>u`.
- **Apple constraints**: Q2_0 and Q8_0 not supported on Metal (`Model::load` returns error). No `TIMESTAMP_QUERY_INSIDE_PASSES` — microbench uses one pass per labeled dispatch with `ComputePassDescriptor::timestamp_writes`. `SUBGROUP_SIZE_CONTROL` not requested.
- **Matmul inner loop**: future matvec changes must preserve `#ifdef METAL_BACKEND` ±-accumulate pattern in GLSL. Matmul changes go directly in `matmul_q1_0_q8_0.metal`.

### Sampling

Two-pass GPU top-K reduction: `topk_partial.comp` (WG=128, 32 WGs, per-WG min-heap → bitonic merge) → `topk_merge.comp` (WG=64, single WG, merges 32 sorted partials). Output: K f32 logits + K u32 indices at `sample[0..2K]`. ~0.15 ms on RX 9070.

CPU finishes: temperature scale → softmax → top-p → multinomial (SplitMix64, seeded by `sampler.seed + pos`). `temperature == 0.0` short-circuits to argmax. Implementation: `session.rs::sample_from_topk`.

### GPU memory layout

**Weights**: 5 storage buffers — `w_attn` (Wq/Wk/Wv/Wo), `w_ffn_gu` (Wgate/Wup), `w_ffn_d` (Wdown), `w_norms` (FP16 norm vecs), `w_embed` (token_embd + LM head). Always look up tensors via `model::tensor(cfg, name)`, never hard-code offsets.

**Activations**: `Buffers::act` is one f16 buffer with regions defined in `ActLayout` (`model.rs`): `x`, `q`, `k_cur`, `v_cur`, `attn_out`, `gate`, `up`, `logits`. Sized for M_MAX=512. `act_q8` is separate Q8_0 scratch for the matmul path.

**KV cache**: `kv_k` + `kv_v`, both Q8_0 (~2.25 bytes/element). Contiguous d-section + qs-section per buffer; helpers in `forward.rs` compute per-layer offsets. Quantized directly by `kv_writeback_fused.comp`.

**`max_seq`**: runtime tunable (`LoadOptions::max_seq`, default 1024). `attn_partials`, RoPE table, and `MAX_CHUNKS` spec constant all scale with it. `Model::load` checks against `max_buffer_size`. No YaRN/NTK scaling — quality degrades beyond ~2× `rope_orig_context`.

**`buffers.sample`**: 1024-u32 buffer, dual role: embed input (`sample[0]` or `[0..M]`) then topk output (`[0..2K]`). No aliasing since embed runs before topk.

### Immediates and BGL discipline

Every dispatch passes a `Params` struct as wgpu push constants via `pass.set_immediates(0, bytemuck::bytes_of(&p))`. All Params structs ≤ 64 bytes (enforced by `params_struct_sizes_fit_immediate_limit` test in `model.rs`). BGLs contain only storage bindings starting at binding 0 — no UBO.

**Single-rw rule**: activation buffers get exactly one `read_write` storage binding per bind group — never aliased as both `read` and `read_write` in one dispatch. Enforced by `rw_mask` in `make_bgl`.

### Encoder organization

`begin_compute_pass` costs ~25 µs on RADV. tg step is encoded as one big pass with 8 dispatches per layer. All `dispatch_*` helpers accept a caller-provided `&mut wgpu::ComputePass<'_>` — use `_in_pass` form when adding tg dispatches and slot into an existing pass. For prefill, per-pass wrapper is fine.

### Tied vs. untied embeddings

4B: tied (`token_embd.weight` used for both embed and LM head). 8B: untied (separate `output.weight`). Both packed into `weights_embed_lmhead.bin`; `OutputTensors` in `model.rs` exposes `token_embd_*` and `lm_head_*` (coincide when tied).

### Adapter limits

`Model::load` computes the largest single buffer it will allocate (largest grouped weight buffer / RoPE table / per-buffer KV cache, scaled by `opts.max_seq`), rounds up to the next power of two, and requests that as `max_storage_buffer_binding_size` & `max_buffer_size`. Also requests `max_storage_buffers_per_shader_stage` ≥ 8.

## When making changes

- **Adding a new kernel**: write `shaders/foo.comp` (single GLSL for both backends; use `#ifdef METAL_BACKEND` for Apple variants). If it needs Apple-only intrinsics, also add `shaders/foo.metal` and add basename to `HAND_PORTED_MSL` in `build.rs`. Add basename to `SHADERS` in `build.rs`. Then `load_shader!("foo", spec_consts, (wg_x, wg_y, wg_z))` in `model.rs` (`no_spec` if no SpecId refs). Add `FooParams` (≤ 64 bytes, `Pod + Zeroable + repr(C)`), register BGL in `BindGroupLayouts` (single-rw rule), build pipeline in `Pipelines`. Add `dispatch_foo` in `forward.rs`.
- **Modifying weight layout**: update both `scripts/extract.py` (writer) and `model.rs`/shaders (reader). Re-extract model dir after changes.
- **Modifying tokenizer**: update both `scripts/extract.py` (writes `vocab.bin` + `merges.txt`) and `scripts/bpe.py` (encodes prompts), and possibly `src/decode.rs`. Rust runtime never tokenizes.
- **Public API**: re-exports in `src/lib.rs`. Stable: `Model`, `ModelConfig`, `LoadOptions`, `Session`, `Sampler`, `GenerateOptions`, `StopReason`, `KvSnapshot`, `PotError`, `Result`, `TOPK_MAX`.
- **Perf work**: `--mode microbench` for per-kernel deltas; `--mode bench` for end-to-end. tg bottleneck: matvec dispatches (LM head ~0.25 ms/step, topk ~0.12 ms). Prefill bottleneck: attention (O(M²)). Keep Q_TILE=2 and `[[unroll]]` in `attention_prefill_tiled`. Keep single-pass-per-step on tg. Keep `#ifdef METAL_BACKEND` ±-accumulate in matvec GLSL.
