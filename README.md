# bonsai-pot

A from-scratch, dependency-light **Bonsai (Qwen3-architecture) Q1_0 inference engine** running on **wgpu compute shaders**. Supports the Bonsai 4B and 8B model sizes.

The defining property of this engine: **weights are never dequantized**. Q1_0 storage is consumed directly by the matvec/matmul kernels — there is no intermediate FP16 weight tensor, no on-the-fly unpack into shared memory, nothing. Each weight contributes to the dot product as a single sign bit selecting `+x` or `-x`, with one FP16 multiply per 128-weight block to apply the block scale. The hot inner loop has **zero multiplications** — every accumulation is an add or a sign-flipped add.

No `llama.cpp`, no `ggml`, no PyTorch on the hot path. Weights are loaded from a custom flat-file layout (produced from a GGUF by `scripts/extract.py`), every kernel is hand-rolled WGSL, and the host side is plain Rust + wgpu 29 + pollster.

## What's in here

- A **library** (`bonsai_pot::{Model, Session, Sampler, GenerateOptions, …}`) for embedding the engine in other Rust programs.
- A **demo CLI** (`bonsai-pot`) that reads pre-tokenized `u32` prompts from stdin and prints decoded output. Bench/microbench utilities live behind the `bench-internals` feature.
- An **interactive ChatML chatbot** (`examples/chat.rs`) that demonstrates KV-cache reuse across turns and out-of-process tokenization.
- Two Python helpers under `scripts/` (PEP 723 inline-deps; just need `uv` on `$PATH`):
  - `extract.py` — GGUF → flat-file model directory.
  - `bpe.py` — standalone BPE encoder for prompts.

Tokenization is intentionally outside the Rust crate.

## Building the model directory

```sh
# 4B:
uv run scripts/extract.py path/to/Bonsai-4B-Q1_0.gguf --out ./model
# 8B:
uv run scripts/extract.py path/to/Bonsai-8B-Q1_0.gguf --out ./model-8b
```

Both model sizes are available on Hugging Face.
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
- `--mode prompt` — batched `dot4I8Packed` matmul prefill (with a Q8_0 activation quantize pre-pass; weights still consumed in Q1_0 with no dequant), then matvec for generation.
- `--mode bench` — `llama-bench`-style table with pp/tg t/s.
- `--mode microbench` — per-kernel breakdown (us/call × calls/step).

### Sampling

`--temperature`, `--top-k`, `--top-p`, `--seed`. Default is greedy (`--temperature 0.0`). Greedy runs are byte-deterministic; stochastic runs are reproducible per seed.

Sampling is hybrid: a single-workgroup `topk_reduce.wgsl` kernel reduces the full logits tensor to top-K candidates (`K = TOPK_MAX = 32`) on the GPU; the CPU then does temperature → softmax → top-p → multinomial.

## How it works

### Q1_0: dequant-free, multiply-free

Q1_0 stores 128 weights per block as 16 bytes of sign bits (±1 per weight) plus a 2-byte FP16 scale `d` — 18 bytes per block. `extract.py` splits each tensor into a contiguous **d-array** of FP16 scales followed by a **qs-array** of raw 16-byte sign blocks, both `u32`-aligned. The manifest in `config.ini` records `d_offset`, `qs_offset`, and `nb` (blocks per row) per tensor.

The shaders consume these two arrays **directly**:

- **Matvec (`shaders/matvec_q1_0.wgsl`)** — for each block, the kernel walks 128 sign bits and accumulates `±x` via `select(-xv, xv, bit_set)`. The block contributes one FP16 multiply at the end (`block_sum * d`). No weight is ever materialized as a real number; nothing is unpacked into shared memory; the inner loop has zero multiplications. `matvec_q1_0_fused.wgsl` packs 2- or 3-range dispatches (QKV; gate+up) into one workgroup to amortize the activation load.
- **Matmul (`shaders/matmul_q1_0_q8_0.wgsl`)** — used in batched prefill. Activations are pre-quantized to Q8_0 (`quantize_q8_0.wgsl`); the kernel then computes the dot product as `sum_of_signed_q8 * d_w * d_x` per block, using `dot4I8Packed` over the activation bytes with the weight sign bits selecting their sign. Again: no FP16 weight tensor, no dequantize step, weight bits are read straight from storage.

### Two execution paths

1. **Single-token (matvec) path** — used for all of `--mode gen` and the generation phase of `--mode prompt`. The whole step (embed → transformer layers → output_norm → LM head → topk) is encoded into a single compute pass.
2. **Batched-prefill (matmul) path** — used by `Session::prefill` and `--mode prompt`. Activations are quantized to Q8_0 once per layer; weights stay in Q1_0.

Pass setup is expensive (~25 us/pass on RADV), so the matvec generation step batches every dispatch — embed, all layers, output norm, LM head, and `topk_reduce` — into a single compute pass. Per-layer K/V is quantized to Q8_0 directly into the KV cache by a `kv_writeback` kernel, replacing the `copy_buffer_to_buffer` that used to break the pass.

### GPU memory layout

Weights live in **5 storage buffers** grouped by role: `w_attn`, `w_ffn_gu`, `w_ffn_d`, `w_norms`, `w_embed`. Activations are one f16 buffer with named regions (`ActLayout`). KV cache is split into `kv_k` / `kv_v` and stored in **Q8_0** (~2.25 bytes/element); per-step K/V is quantized straight into the cache, with no f16 staging copy. Capacity is set at load via `ModelOptions::max_seq` (default 1024; `--max-seq` on both the bin and the chat example). Bonsai 4B uses **tied** embeddings (`token_embd.weight` serves as both embed table and LM head); Bonsai 8B ships a separate `output.weight` tensor for the LM head.

There is exactly one `uniform` buffer; every dispatch's params struct is appended to a CPU-side pool with a dynamic offset, and the whole pool is uploaded in one `write_buffer` per step.

## Architecture map

| Path | What's in it |
| --- | --- |
| `src/lib.rs` | Public API surface, re-exports |
| `src/model.rs` | Config / manifest loading, GPU device & buffer & pipeline & BGL setup, RoPE precompute, `ModelOptions` |
| `src/session.rs` | `Session<'m>`, `Sampler`, `GenerateOptions`, `StopReason`, CPU sampler |
| `src/kv_snapshot.rs` | Host-resident `KvSnapshot` of the GPU KV cache (used by `Session::snapshot` / `Session::restore`) |
| `src/forward.rs` | Forward pass (both paths) and per-step encoder helpers |
| `src/error.rs` | `PotError` / `Result` |
| `src/decode.rs` | GPT-2 byte-level decode |
| `src/bin/bonsai-pot.rs` | Demo CLI |
| `src/shaders/*.wgsl` | One file per kernel |
| `examples/chat.rs` | Interactive ChatML REPL on the public API |
| `tests/gpu_integration.rs` | End-to-end GPU tests (load `./model`, prefill/generate, snapshot round-trip) |
| `scripts/extract.py` | GGUF → flat-file converter |
| `scripts/bpe.py` | Standalone BPE encoder |

## Public API

`Model`, `ModelConfig`, `ModelOptions`, `Session`, `Sampler`, `GenerateOptions`, `StopReason`, `KvSnapshot`, `PotError`, `Result`, `TOPK_MAX`. Anything not re-exported from `src/lib.rs` is internal.
