# CLAUDE.md

## What this is

`bonsai-pot` is a from-scratch inference engine for **Ternary-Bonsai-2-27B** (`qwen35` hybrid attention, `PTQ1_0` ternary weights with a Prism Hadamard rotation) on **wgpu compute shaders, Vulkan only**. No llama.cpp/ggml/PyTorch: weights come from a flat-file layout produced by `scripts/extract.py`, every kernel is hand-rolled GLSL under `src/shaders/` compiled to SPIR-V by `build.rs` (`glslangValidator` ≥ 16 + `spirv-opt` on `$PATH`) and fed through `create_shader_module_passthrough` (naga bypassed). Host side is Rust + the user's wgpu fork.

Exposes a **library** (`bonsai_pot::{Model, Session, Sampler, GenerateOptions, …}`) and a **demo CLI** (`src/bin/bonsai-pot.rs`). Tokenization is out-of-crate — `scripts/bpe.py` (qwen35 regex, added tokens atomic) pipes u32 ids into the bin.

## Build / run / bench

```
cargo build --release --features bench-internals
uv run scripts/extract.py path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf --out ./model
uv run scripts/bpe.py ./model "Once upon a time" | target/release/bonsai-pot ./model --mode prompt --max-new-tokens 64
target/release/bonsai-pot ./model --mode bench --pp 512 --tg 128
target/release/bonsai-pot ./model --mode microbench --pp 512 --tg 128     # per-kernel GPU timings
target/release/bonsai-pot ./model --mode dump --out DIR < prompt.u32       # llama.cpp-named tensors
cargo run --release --example chat -- ./model
```

Tests: `cargo test --release --features bench-internals` (unit tests + `tests/gpu_integration.rs` against `./model`). Set `BONSAI_POT_REF_DIR` to a directory of `<name>.u32` + `<name>_cpu/` llama.cpp dumps to enable the reference-match test. GPU tests share one model; the device-lost test takes a write lock (VRAM).

`BONSAI_POT_COOPMAT=i8|f16|off` overrides the prefill matmul choice (default: int8 coopmat, then f16 coopmat, then DP4a).

## Correctness workflow (use it for every kernel change)

Reference = PrismML llama.cpp fork (`/mnt/workspace/prismml-llama.cpp`, builds `build-vk`, `build-cpu`, shared libs). `scripts/llama_dump/` (C++, `cb_eval` + name regex) writes raw f32 tensors + `index.jsonl`; `--mode dump` writes ours in the same format; `scripts/compare.py REF OURS` prints per-layer cosine and logits top-1. The **CPU** backend is the precise reference: the fork's Vulkan backend itself only reaches ~0.997 per-layer cosine vs CPU. Golden dumps live in `/mnt/workspace/bonsai2-ref` (`p1`–`p3`, `tf` = 256-token teacher-forced; `*_cpu` = CPU backend).

Acceptance: per-layer cosine ≥ 0.999 vs CPU dump, logits top-1 ≥ 95%.

## Model / math (what the shaders implement)

- 64 layers; layer `il` is full attention iff `(il+1) % 4 == 0` (16 layers), else Gated-DeltaNet (48).
- Every PTQ1_0 matmul input is `FWHT_1024blockwise(s_width ⊙ x)` (normalized Sylvester WHT, signs from `prism.hadamard.sign_values`, packed as bits in `weights_small.bin`). `ssm_out` input is first permuted tiled → grouped (`d + 128·(k + 16r)` → `d + 128·(r + 3k)`). Embedding: `x = s ⊙ FWHT(dequant(row))`.
- PTQ1_0 block = 7 u32: `qs[24]` (byte j<16 → elements 16t+j; byte 16+j → 80+8t+j), `qh[2]` (→ 120+2n+h), f16 d in bytes 26–27. Trit recurrence `t = (v*3)>>8, v = (v*3)&0xFF`; weight = (t−1)·d.
- Activations for projections: Q8 with one f32 scale per 128 + a precomputed `d·Σq` per 128 (the −1 offset term).
- GDN: qkv (10240) → conv4 + SiLU → l2norm q/k (eps = rms_eps, `x/max(‖x‖,eps)`); v head h uses q/k head `h % 16`; `g = softplus(α·xn + dt_bias)·ssm_a`, `β = σ(β·xn)` (BF16 weights, **un-rotated** `xn`); delta rule `S←S·e^g; δ=β(v−Sᵀk); S+=kδᵀ; o=Sᵀq/√128`; gated RMSNorm(o)·w ⊙ silu(z).
- Full attention: `attn_q` rows interleave per head `[q(256) | gate(256)]`; q/k RMSNorm; NEOX RoPE on dims 0..63 (pairs i, i+32, base 1e7, f64 host table); GQA 6; output ⊙ σ(gate).

## Files

- `src/config.rs` — `ModelConfig` (qwen35 fields), ini/manifest parsing, `validate` (kernel invariants), `resolve_layers/resolve_output` → word offsets (`LayerTensors`, `GdnTensors`, `AttnTensors`, `OutputTensors`).
- `src/model.rs` — `Model::load_from_snapshot`: adapter checks, limits, HAL device open (async-compute queue, global priority, maximal reconvergence, `VK_KHR_cooperative_matrix` if a u8×s8 or f16 16×16×16 config exists), **weight windows** (window size limit = adapter's buffer/binding limit; the wgpu fork's branch `relax-4gib-buffer-limits` reports 4 GiB−4 on Mesa/RADV, so the gate/up file is one window; on adapters with a smaller limit it is split at layer boundaries into separate buffers; `LayerTensors::gu_win`), pipelines, layouts `ActLayout` (f32 regions × `M_MAX`), `Q8Layout`, `StateLayout`, `KvLayout`.
- `src/session.rs` — per-session buffers (act, q8, gdn state, kv_k/kv_v, io, staging, readback), one bind group per weight window, `Sampler` (top_k → top_p → min_p on un-tempered probs → temperature), `Session` API.
- `src/forward.rs` — `proj` (matvec for m=1, matmul for m>1), `gdn_block` / `attn_block` / `ffn_block` (all take `m`, `pos`), `encode_step` (one pass), `prefill_topk` (submits every `PREFILL_LAYERS_PER_SUBMIT=16` layers — a single >2 s submit triggers an amdgpu context reset), `prefill_token_loop_topk`, `dump_tokens`, `debug_prefill_x`, bench markers (`BenchMarker` supports multiple passes).
- `src/kv_snapshot.rs` — format v2 (`BONSAI2S`): KV prefix of attention layers + whole recurrent state; chunked staging (≤256 MiB).
- `src/params.rs` — push-constant structs (≤ 64 B, test-enforced).
- `src/cpu_ref.rs` — CPU PTQ1_0 decode / FWHT references.
- `src/bench.rs` — bench, microbench (tg/pp), dump + debug hooks (`bench-internals`).

## Shaders

All kernels include `lib/bindings.glsl` — **one universal bind-group layout** (0 weights window ro, 1 `ws` small weights ro, 2 `act` f32, 3 `q8`, 4 `kv_k`, 5 `kv_v`, 6 `st` state, 7 `io` (u32 tokens / top-K; `sample` is a GLSL keyword), 8 `rope` ro). Override qualifiers with `#define ACT_QUAL readonly` etc. before including. One pipeline layout, `IMMEDIATE_BYTES = 64`.

- `embed`, `norm_rot_q8`, `silu_mul_rot_q8`, `gdn_out_rot_q8`, `attn_out_rot_q8` — produce rotated Q8 inputs via `lib/fwht_q8.glsl` (WG 256, 4 elements/invocation, FWHT stages 1–2 in registers, then subgroup shuffles, then LDS; Q8 store via 32-lane clustered reductions with LDS fallback).
- `matvec_ptq1_0` — 16 lanes/row, 16 rows/WG (8 blocks in flight divides K ∈ {40, 48, 136} blocks), ≤3 segments, optional `+=`. Tried and rejected (slower on RX 9070): 8 lanes/row, LDS-staged activations, software-pipelined loads, forced wave32.
- `matmul_ptq1_0_cmi8` (default prefill) — coopmat u8 trits × s8 Q8 → s32, tile 64 rows × 128 tokens, 8 wave32 subgroups (pipeline `SubgroupSize::Fixed(32)`), per-block rescale `d_w·(d_a·C − d_a·Σq)` via a per-subgroup 16×16 LDS stage into f32 registers, ~37 KiB LDS. Tried and rejected: 256-token tile (slower). `matmul_ptq1_0_cm` — f16 coopmat fallback (~60 KiB LDS). `matmul_ptq1_0` — DP4a fallback, 64×64.
- `gdn_gates` (one output per WG), `gdn_conv` (sequential over m), `gdn_recurrence` (4 lanes per state column, 32 columns/WG, sequential over m).
- `attn_qk_norm_rope_kv` (Q in place; K/V → Q8_0 cache), `attn_split` (decode, split-K over ≤ 64 chunks of whole 32-key tiles; K/V read directly from cache), `attn_prefill` (causal FA-2, f16 LDS tiles, QT=2 tokens × GQA heads).
- `topk_partial` / `topk_merge` — f32 logits from `act`, results in `io[0..2K]`.

## When making changes

- New kernel: `src/shaders/foo.comp` including `lib/bindings.glsl`; add to `SHADERS` in `build.rs`; `FooParams` in `params.rs` (+ size test); pipeline field in `Pipelines` + `pipeline!("foo")` in `model.rs`; dispatch via `forward::dispatch(s, pass, pipe, win, &params, groups)`.
- Model-format changes: update `scripts/extract.py` and `config.rs` together, re-extract.
- Always re-run `--mode dump` + `compare.py` against the CPU goldens and `cargo test --release --features bench-internals`; keep `cargo clippy --all-targets --features bench-internals` clean.
- Perf: `--mode microbench` for per-kernel deltas, `--mode bench` for end-to-end. RX 9070: ours pp512 ≈ 483, tg128 ≈ 77 t/s (GPU); fork baseline pp512 485, tg128 54.6 t/s.
