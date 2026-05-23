# bonsai-pot

A from-scratch, dependency-light **Bonsai / Qwen3-architecture inference engine** running on **wgpu compute shaders**. Supports Bonsai 4B and 8B models in **Q1_0** (binary), **Q2_0** (ternary), and **Q8_0** quantization.

The defining property of this engine: **weights are never dequantized**. Quantized weight storage is consumed directly by the matvec/matmul kernels — there is no intermediate FP16 weight tensor, no unpack to float. For Q1_0 each weight is a single sign bit selecting `+x` or `-x`; for Q2_0 a 2-bit ternary code selects `−x`, `0`, or `+x`; for Q8_0 an i8 value is fed directly to `dot4I8Packed`. In all cases one FP16 scale multiply closes out each 128-weight block. The inner loop has **zero float multiplications** per weight.

No `llama.cpp`, no `ggml`, no PyTorch on the hot path. Weights are loaded from a custom flat-file layout (produced from a GGUF by `scripts/extract.py`), every kernel is hand-rolled GLSL (with one hand-ported `.metal` for the Apple `simdgroup_matrix` matmul), and the host side is plain Rust + wgpu 29.

## Performance

End-to-end throughput from `--mode bench --pp 512 --tg 128` (5 timed reps after 1 warmup):

| GPU                         | model     | e2e_pp632 t/s        | e2e_tg128 t/s     |
|-----------------------------|-----------|---------------------:|------------------:|
| AMD RX 9070 (RDNA4, wave64) | Bonsai-4B | **2283.52** ± 2.76   | **384.82** ± 1.09 |
| AMD RX 9070 (RDNA4, wave64) | Bonsai-8B |  1293.56 ± 1.70      |  256.18 ± 1.17    |
| NVIDIA GB10 (Blackwell, 32) | Bonsai-4B | 2406.83 ± 6.40       |  182.19 ± 2.09    |
| NVIDIA GB10 (Blackwell, 32) | Bonsai-8B |  976.51 ± 8.22       |  121.46 ± 1.10    |

`e2e_pp632` runs `Session::prefill` on a 632-token ChatML-wrapped prompt — long enough to exceed the `m_max=512` chunk cap and exercise the cross-chunk `pos_base` advancement. `e2e_tg128` runs `Session::generate` end-to-end with **stochastic top-K=32 sampling** (the GPU top-K reduction is the implicit top-K cap; the CPU does temperature → softmax → multinomial over those 32 candidates) and **encode pipelining** — the next step's command buffer is encoded on the CPU while the GPU drains the current step. So these are realistic chat-time numbers, not bare-forward microbenches. Both holds up at long context: on RX 9070, `e2e_tg1024` is ~332 t/s for 4B and ~232 t/s for 8B (~14% / ~9% drop vs `e2e_tg128`); bare-forward prefill (`--mode bench`, no sampler / no readback) goes pp512 ~2510, pp1024 ~2300, pp4096 ~1564 t/s, held together by the Q-tiled + GQA-batched FA-2 prefill kernel ([Attention](#attention)) — Q_TILE=2 × Q_PER_GROUP=4 means a single K[t] load is reused across 8 queries inside the workgroup, instead of being re-loaded once per Q-head per query.

## What's in here

- A **library** (`bonsai_pot::{Model, Session, Sampler, GenerateOptions, …}`) for embedding the engine in other Rust programs.
- A **demo CLI** (`bonsai-pot`) that reads pre-tokenized `u32` prompts from stdin and prints decoded output. Bench/microbench utilities live behind the `bench-internals` feature.
- An **interactive ChatML chatbot** (`examples/chat.rs`) that demonstrates KV-cache reuse across turns and out-of-process tokenization.
- Two Python helpers under `scripts/` (PEP 723 inline-deps; just need `uv` on `$PATH`):
  - `extract.py` — GGUF → flat-file model directory.
  - `bpe.py` — standalone BPE encoder for prompts.

Tokenization is intentionally outside the Rust crate.

## Prerequisites

- **Rust** ≥ 1.87, and C compiler
- **`glslangValidator`** ≥ 16 — ships in the `glslang-tools` package (`apt install glslang-tools` / `brew install glslang`)
- **`spirv-opt`** >= 2026 — ships in the `spirv-tools` package (`apt install spirv-tools` / `brew install spirv-tools`)
- **`uv`** — see <https://docs.astral.sh/uv/getting-started/installation/>
- **`spirv-cross`** — Apple builds only (`brew install spirv-cross` / `apt install spirv-cross`)

All tools must be on `$PATH` at build time.

## Building the model directory

```sh
# Bonsai Q1_0 (binary):
uv run scripts/extract.py path/to/Bonsai-4B-Q1_0.gguf --out ./model
uv run scripts/extract.py path/to/Bonsai-8B-Q1_0.gguf --out ./model-8b
# Ternary Bonsai Q2_0 (Vulkan only):
uv run scripts/extract.py path/to/Ternary-Bonsai-4B-Q2_0.gguf --out ./model-ternary-4b
# Qwen3 Q8_0 (Vulkan only):
uv run scripts/extract.py path/to/Qwen3-4B-Q8_0.gguf --out ./model-q8
```

All models are available on Hugging Face.
This writes `config.ini`, five `weights_*.bin` files, `vocab.bin`, `vocab_offsets.bin`, and `merges.txt`.

## Build and run

```sh
# Library only:
cargo build --release --lib

# Demo CLI (pulls in bench/microbench helpers):
cargo build --release --features bench-internals

# End-to-end: tokenize on the Python side, generate on the Rust side.
uv run scripts/bpe.py ./model "Once upon a time" \
  | cargo run --release --features bench-internals -- ./model \
        --mode prompt --max-new-tokens 64

# Bench / microbench (no stdin needed):
cargo run --release --features bench-internals -- ./model --mode bench --pp 512 --tg 128
cargo run --release --features bench-internals -- ./model --mode microbench

# Interactive ChatML chatbot:
cargo run --release --example chat -- ./model
```

### CLI modes

- `--mode gen` (default) — single-token matvec path for both prompt and generation; the dequant-free, multiply-free Q1_0 hot path.
- `--mode prompt` — batched `dot4I8Packed` matmul prefill (weights stay in Q1_0; activations arrive Q8_0-quantized inline by the upstream fused kernels), then matvec for generation.
- `--mode bench` — `llama-bench`-style table with pp/tg t/s.
- `--mode microbench` — per-kernel breakdown (us/call × calls/step).

### Sampling

`--temperature`, `--top-k`, `--top-p`, `--seed`. Default is greedy (`--temperature 0.0`). Greedy runs are byte-deterministic; stochastic runs are reproducible per seed.

Sampling is hybrid: a two-pass multi-WG reduction (`topk_partial.comp` + `topk_merge.comp`) reduces the full logits tensor to top-K candidates (`K = TOPK_MAX = 32`) on the GPU; the CPU then does temperature → softmax → top-p → multinomial.

## How it works

### Weight formats

Three quantization formats are supported, all sharing the 128-weight super-block layout. `extract.py` splits each tensor into a contiguous **d-array** of FP16 scales followed by a **qs-array** of raw weight codes, both `u32`-aligned. The manifest in `config.ini` records `d_offset`, `qs_offset`, and `nb` (super-blocks per row) per tensor. The format is autodetected from the GGUF and recorded as `quant_format` in `config.ini`.

- **Q1_0** — 16 bytes of sign bits per 128-weight block (1 bit/weight, ±1); 18 B/block total. Binary weights. Supported on Vulkan and Metal.
- **Q2_0** (Ternary Bonsai) — 32 bytes of 2-bit codes per 128-weight block (−1, 0, +1); 34 B/block total. Vulkan only.
- **Q8_0** — four native 32-element GGML blocks per super-block (4 × 34 B = 136 B); each native block is 32 i8 weights + 2-byte FP16 scale. Vulkan only.

The shaders consume these arrays **directly** with no float dequantization:

- **Matvec (`shaders/matvec_q1_0.comp`)** — the kernel expands each block's weight codes to ±1/0 packed-byte form and accumulates via `dotPacked4x8EXT` (one DP4a instruction per 4-element dot), with one FP16 scale multiply per block. No weight is ever materialized as a float; nothing is unpacked into shared memory beyond what DP4a needs. `matvec_q1_0_fused_normed.comp` packs 2- or 3-range dispatches (QKV; gate+up) into one workgroup, amortizes the activation load across them, and folds `rms_norm(x) * w_norm` over the activation so there's no `act.x_norm` round-trip. `matvec_q1_0_silu.comp` folds `silu(gate) * up` into the ffn_down matvec. On Metal (Q1_0 only) the Q8_0 activation shmem round-trip is skipped in favour of an f32 `select(±a, cond)` accumulate — no DP4a equivalent exists on Apple Silicon.
- **Matmul (`shaders/matmul_q1_0_q8_0.comp`)** — used in batched prefill. Activations arrive Q8_0-quantized inline from the upstream fused kernels (`rms_norm_q8_0`, `silu_mul_q8_0`, `attention_prefill_tiled`) — no separate quantize pass. Weight codes are expanded to packed-byte form and the dot product runs via `dot4I8Packed`, with one combined `d_w * d_x` scale multiply per block. On Apple (Q1_0 only) this kernel is hand-ported to MSL as `matmul_q1_0_q8_0.metal` around `simdgroup_matrix<half,8,8>` MMA instructions.

### Two execution paths

1. **Single-token (matvec) path** — used for all of `--mode gen` and the generation phase of `--mode prompt`. The whole step (embed → transformer layers → output_norm → LM head → topk) is encoded into a single compute pass.
2. **Batched-prefill (matmul) path** — used by `Session::prefill` and `--mode prompt`. Activations are Q8_0-quantized inline by fused kernels (`rms_norm_q8_0`, `silu_mul_q8_0`, and `attention_prefill_tiled` for the attn output); weights stay in their storage format.

Pass setup is expensive (~25 us/pass on RADV), so the matvec generation step batches every dispatch — embed, all layers, output norm, LM head, and the two-pass top-K — into a single compute pass. Per-layer K/V is rms-normed, RoPE'd, Q8_0-quantized, and written directly into the KV cache by `kv_writeback_fused`, replacing the `copy_buffer_to_buffer` that used to break the pass.

### Attention

Two attention kernels, both fused-softmax flash-attention variants:

- **Generation (tg, m=1)** — split-K + GQA-batched (`attention_split.comp` + `attention_merge.comp`). Each `(kv_head, chunk)` workgroup scans an 8-position slice of the cache; the four Q-heads sharing the KV group are processed together so each K/V load is reused 4×. Per-chunk `(m, l, o)` partials are written to scratch; `attention_merge` does a flash-attention log-sum-exp combination across the chunks. This decouples per-step latency from KV length.
- **Prefill (matmul, m>1)** — Q-tiled + GQA-batched FA-2 (`attention_prefill_tiled.comp`). Each workgroup handles `Q_TILE=2` consecutive query tokens × 4 GQA Q-heads sharing the KV head, so a single K[t] load is reused across 8 queries. The output is **inline Q8_0-quantized** and written directly to the activation Q8_0 buffer — the downstream Wo matmul reads it without a round-trip. Per-query state (Q registers, output accumulators, m/l) is manually unrolled rather than indexed dynamically out of an array (NVIDIA spills `array<…, Q_TILE>` with dynamic indexing to local memory; manual unrolling stays in registers).

### GPU memory layout

Weights live in **5 storage buffers** grouped by role: `w_attn`, `w_ffn_gu`, `w_ffn_d`, `w_norms`, `w_embed`. Activations are one f16 buffer with named regions (`ActLayout`). KV cache is split into `kv_k` / `kv_v` and stored in **Q8_0** (~2.25 bytes/element); per-step K/V is quantized straight into the cache, with no f16 staging copy. Capacity is set at load via `LoadOptions::max_seq` (default 1024; `--max-seq` on both the bin and the chat example). Bonsai 4B uses **tied** embeddings (`token_embd.weight` serves as both embed table and LM head); Bonsai 8B ships a separate `output.weight` tensor for the LM head.

There is no UBO. Every dispatch's params struct (≤ 64 B) is passed as wgpu immediates (push constants) via `pass.set_immediates`, and BGLs hold only storage bindings.

## Architecture map

| Path | What's in it |
| --- | --- |
| `src/lib.rs` | Public API surface, re-exports |
| `src/model.rs` | Config / manifest loading, GPU device & buffer & pipeline & BGL setup, RoPE precompute, `LoadOptions` |
| `src/session.rs` | `Session<'m>`, `Sampler`, `GenerateOptions`, `StopReason`, CPU sampler |
| `src/kv_snapshot.rs` | Host-resident `KvSnapshot` of the GPU KV cache (used by `Session::snapshot` / `Session::restore`) |
| `src/forward.rs` | Forward pass (both paths) and per-step encoder helpers |
| `src/error.rs` | `PotError` / `Result` |
| `src/decode.rs` | GPT-2 byte-level decode |
| `src/bin/bonsai-pot.rs` | Demo CLI |
| `src/shaders/*.comp` (+ `matmul_q1_0_q8_0.metal`) | One GLSL source per kernel; one hand-ported MSL exception for the Apple `simdgroup_matrix` matmul |
| `examples/chat.rs` | Interactive ChatML REPL on the public API |
| `tests/gpu_integration.rs` | End-to-end GPU tests (load `./model`, prefill/generate, snapshot round-trip) |
| `scripts/extract.py` | GGUF → flat-file converter |
| `scripts/bpe.py` | Standalone BPE encoder |

## Public API

`Model`, `ModelConfig`, `LoadOptions`, `Session`, `Sampler`, `GenerateOptions`, `StopReason`, `KvSnapshot`, `PotError`, `Result`, `TOPK_MAX`. Anything not re-exported from `src/lib.rs` is internal.
