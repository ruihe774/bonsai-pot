# bonsai-pot

A from-scratch, dependency-light inference engine for **Ternary-Bonsai-2-27B** (PrismML's ternary build of the Qwen3.5 hybrid-attention architecture) running on **wgpu compute shaders** (Vulkan).

The defining property of this engine: **weights are never dequantized into a weight tensor**. The `PTQ1_0` packing (five base-3 trits per byte, one FP16 scale per 128 weights, 1.75 bits/weight) is consumed directly. The single-token matvec decodes trits on the fly with a shift-add residue trick and feeds them to `dotPacked4x8EXT` against Q8 activations. Batched prefill decodes each weight tile once into workgroup memory as packed trit bytes and multiplies them with int8 cooperative matrices (WMMA), or with DP4a tiles where those are unavailable.

No `llama.cpp`, no `ggml`, no PyTorch on the hot path. Weights are loaded from a flat-file layout (produced from the GGUF by `scripts/extract.py`), every kernel is hand-rolled GLSL compiled to SPIR-V at build time, and the host side is plain Rust + wgpu.

## Performance

AMD RX 9070 (RDNA4, 16 GB, RADV), `Ternary-Bonsai-2-27B-PTQ1_0`, `--mode bench --pp 512 --tg 128`:

| backend | pp512 t/s | tg128 t/s (GPU) | e2e_tg128 t/s |
|---|---:|---:|---:|
| **bonsai-pot** | **483** | **77.4** | **75.8** |
| PrismML llama.cpp fork, Vulkan (`llama-bench -p 512 -n 128`) | 485 | 54.6 | — |

`e2e_tg128` runs `Session::generate` end-to-end with stochastic sampling and encode pipelining (the next step's command buffer is encoded while the GPU runs the current one). The decode floor set by weight bandwidth is about 110 t/s (5.7 GB of weights per token at 640 GB/s); the matvecs run at about 540 GB/s.

## Correctness

Output is checked against the PrismML llama.cpp fork (the reference implementation for this model):

- Greedy generations are token-identical to the fork for 48 tokens on three test prompts.
- Per-layer residuals vs the fork's CPU backend: cosine ≥ 0.9998. Final-logit top-1 agreement: 100% on the prompts and 99% over 256 teacher-forced positions. That is as close to the CPU reference as the fork's own Vulkan backend gets.

See [Reference tooling](#reference-tooling).

## What's in here

- A **library** (`bonsai_pot::{Model, Session, Sampler, GenerateOptions, …}`).
- A **demo CLI** (`bonsai-pot`) that reads pre-tokenized `u32` prompts from stdin. Bench/microbench/dump utilities live behind the `bench-internals` feature.
- An **interactive chat REPL** (`examples/chat.rs`) using the Qwen3.5 chat template (reasoning effort, `<think>` blocks).
- Python helpers under `scripts/` (PEP 723 inline deps; just need `uv`):
  - `extract.py`: GGUF → flat-file model directory.
  - `bpe.py`: standalone BPE encoder (qwen35 pre-tokenizer, added tokens matched atomically).
  - `compare.py`: compares two tensor dumps (ours vs the reference).
  - `llama_dump/`: tiny C++ tool linked against the fork that dumps reference tensors.

Tokenization is intentionally outside the Rust crate.

## Prerequisites

- **Rust** ≥ 1.87
- **`glslangValidator`** ≥ 16 (`glslang-tools`) and **`spirv-opt`** (`spirv-tools`) on `$PATH` at build time
- **`uv`** for the scripts
- A Vulkan 1.3 GPU with subgroup ops, integer dot product, 16-bit storage and ≥ 32 KiB workgroup memory. Cooperative matrices (`VK_KHR_cooperative_matrix`, 16×16×16 u8×s8→s32 or f16×f16→f32) are used for prefill when available. Set `BONSAI_POT_COOPMAT=i8|f16|off` to override the choice.

## Building the model directory

```sh
uv run scripts/extract.py path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf --out ./model
```

`extract.py` validates the `prism.hadamard.*` metadata and writes:
- `config.ini`;
- the weight files `weights_attn.bin`, `weights_ffn_gate_up.bin`, `weights_ffn_down.bin`, `weights_embed_lmhead.bin`, `weights_small.bin`;
- the tokenizer files `vocab.bin`, `vocab_offsets.bin`, `added_tokens.txt`, `merges.txt`.

## Build and run

```sh
cargo build --release --features bench-internals

# Tokenize on the Python side, generate on the Rust side.
uv run scripts/bpe.py ./model "Once upon a time" \
  | cargo run --release --features bench-internals -- ./model --mode prompt --max-new-tokens 64

# Bench / per-kernel microbench
cargo run --release --features bench-internals -- ./model --mode bench --pp 512 --tg 128
cargo run --release --features bench-internals -- ./model --mode microbench --pp 512 --tg 128

# Interactive chat (thinking on by default; --reasoning-effort medium|low|none)
cargo run --release --example chat -- ./model
```

CLI modes:
- `--mode gen` (default): token-at-a-time prefill, then generation.
- `--mode prompt`: batched prefill, then generation.
- `--mode bench`: pp/tg table.
- `--mode microbench`: per-kernel GPU timings.
- `--mode dump --out DIR`: llama.cpp-named activations, for `compare.py`.

Sampling flags: `--temperature` (default 0 = greedy), `--top-k`, `--top-p`, `--min-p`, `--seed`. The model card recommends temperature 1.0, top-p 0.95, top-k 20, min-p 0.05 for thinking mode.

## How it works

### Model

Qwen3.5-27B hybrid architecture: 64 layers, hidden size 5120, vocabulary 248320.
- **Gated-DeltaNet linear attention (48 layers):** the projections feed a causal conv1d (kernel 4) + SiLU, then L2-normalized q/k. A gated delta-rule recurrence keeps 48 value heads, each with a 128×128 state. Its output goes through a gated RMSNorm.
- **Gated full attention (every 4th layer, 16 layers):** 24 query heads, 4 KV heads, head dim 256. Partial NEOX RoPE on 64 of the 256 dimensions, then a sigmoid output gate.
- **FFN:** SwiGLU in every layer.

### Hadamard rotation

The weights are stored in a blockwise-rotated input basis. Before every PTQ1_0 projection the activation is transformed: `x' = FWHT_1024(s ⊙ x)`, with Prism's explicit ±1 signs. The `ssm_out` input is first reordered from tiled to grouped head order. The embedding lookup applies the inverse, `s ⊙ FWHT(row)`.

Each kernel that produces a projection input fuses this transform: RMSNorm, SwiGLU, gated DeltaNet output norm, and gated attention output. That kernel then quantizes the result to Q8 with one scale per 128 elements. After the rotation removes outliers, a 128-element scale is sufficient.

### Execution paths

1. **Decode (single token):** the whole step is one compute pass of about 690 dispatches.
   - Projections use `matvec_ptq1_0`: 16 lanes per row, up to three weight segments per dispatch.
   - Full-attention layers use split-K tiled attention (`attn_split`), merged by log-sum-exp in `attn_out_rot_q8`.
   - Residual adds are fused into the output projections.
2. **Batched prefill (≤ 512 tokens per chunk):** the same kernels over M rows.
   - Projections use `matmul_ptq1_0_cmi8`: int8 cooperative matrices, with a per-128-block rescale through LDS into f32 accumulators. The fallbacks are `matmul_ptq1_0_cm` (f16 coopmat) and `matmul_ptq1_0` (DP4a).
   - `attn_prefill` is causal FA-2 with f16 LDS tiles.
   - The GDN conv and recurrence scan the chunk sequentially.
   - Prefill is submitted in groups of 16 layers so no GPU job approaches the driver's ~2 s timeout.

### Memory

Weights live in one storage buffer per window. The wgpu fork reports the driver's buffer and binding limit (4 GiB − 4 on Mesa/RADV), so the 2.5 GB gate/up file loads as a single window. On an adapter with a smaller limit (for example 2 GiB − 1), that file is split at layer boundaries into several windows.

Each session owns:
- f32 activation scratch;
- a Q8 activation buffer;
- the Gated-DeltaNet recurrent state (about 157 MB);
- a Q8_0 KV cache for the 16 full-attention layers, in separate `kv_k` and `kv_v` buffers at 36 KiB per position.

`LoadOptions::max_seq` defaults to 32768. It is capped by the model's context length and by the per-buffer limit: about 233K positions with the 4 GiB limit that the wgpu fork reports on Mesa/RADV, and about 116K on an adapter capped at 2 GiB − 1.

There is one universal bind-group layout of 9 storage buffers for every kernel. Per-dispatch parameters (≤ 64 B) are wgpu immediates.

`Session::reset` is O(1): kernels treat the recurrent state as zero at position 0. `KvSnapshot` captures the KV prefix plus the full recurrent state. Because recurrent state can't be truncated, a snapshot restores exactly the position it was captured at.

## Reference tooling

```sh
# Build the PrismML fork (shared libs) and the dumper
cmake -S scripts/llama_dump -B target/llama_dump -DLLAMA_CPP_DIR=/path/to/prismml-llama.cpp \
      -DLLAMA_BUILD_DIR=/path/to/prismml-llama.cpp/build-cpu && cmake --build target/llama_dump
uv run scripts/bpe.py ./model < prompt.txt > prompt.u32
target/llama_dump/llama_dump -m Ternary-Bonsai-2-27B-PTQ1_0.gguf -t prompt.u32 -o ref/ -ngl 0 \
      -f 'l_out-[0-9]+' --logits-all
bonsai-pot ./model --mode dump --out ours/ < prompt.u32
uv run scripts/compare.py ref/ ours/
```

`tests/gpu_integration.rs::matches_llama_cpp_reference_dump` runs the same comparison when `BONSAI_POT_REF_DIR` points at a directory holding `<name>.u32` files and `<name>_cpu/` dumps.

## Architecture map

| Path | What's in it |
| --- | --- |
| `src/lib.rs` | Public API surface |
| `src/config.rs` | `config.ini` parsing, validation, resolved tensor offsets |
| `src/model.rs` | Device creation (HAL path: async compute queue, global priority, cooperative matrix), weight windows, pipelines, buffer layouts, `LoadOptions` |
| `src/session.rs` | `Session`, `Sampler`, `GenerateOptions`, `StopReason`, CPU sampler |
| `src/forward.rs` | Forward pass (decode + batched prefill), reference dumps, bench markers |
| `src/kv_snapshot.rs` | `KvSnapshot` (KV prefix + recurrent state) |
| `src/params.rs` | Push-constant structs |
| `src/cpu_ref.rs` | CPU PTQ1_0 decode and FWHT references (tests) |
| `src/shaders/*.comp` | One GLSL source per kernel; shared helpers in `src/shaders/lib/` |
| `src/bin/bonsai-pot.rs` | Demo CLI |
| `examples/chat.rs` | Interactive chat REPL |
| `tests/gpu_integration.rs` | End-to-end GPU tests against `./model` |
| `scripts/` | `extract.py`, `bpe.py`, `compare.py`, `llama_dump/` |

## Public API

`Model`, `ModelConfig`, `ModelSnapshot`, `TensorEntry`, `LoadOptions`, `GlobalPriority`, `PowerPreference`, `Session`, `Sampler`, `GenerateOptions`, `StopReason`, `KvSnapshot`, `PotError`, `Result`, `TOPK_MAX`. Anything not re-exported from `src/lib.rs` is internal.
