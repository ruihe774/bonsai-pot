# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`bonsai-wgpu` is a from-scratch, dependency-light **Bonsai-4B (Qwen3-architecture) Q1_0 inference engine** built on **wgpu compute shaders**. There is no `llama.cpp`, `ggml`, or PyTorch on the hot path — weights are loaded from a custom flat-file layout (produced by `extract.py` from a GGUF), all kernels are hand-rolled WGSL, and the host side is plain Rust + wgpu 29 + pollster.

## Build / run / bench

```
cargo build --release
cargo run  --release -- <model_dir> [--mode {gen,prompt,bench,microbench}] [--max-new-tokens N] [--pp N] [--tg N] [--repeats N]
```

`<model_dir>` is the output of `extract.py` (default `./model`); it must contain `config.json`, the five `weights_*.bin` files, `vocab.bin`, `vocab_offsets.bin`, and `prompt.bin`. There is no separate test suite — `--mode gen` and `--mode bench` are the correctness/perf harness.

- `--mode gen` (default): single-token matvec path for both prompt and generation (multiply-free Q1_0 hot path).
- `--mode prompt`: batched dot4I8Packed matmul prefill (with a Q8_0 activation quantize pre-pass), then matvec for generation.
- `--mode bench`: prints an `llama-bench`-style table with pp / tg / pipelined-tg t/s. The pipelined-tg row is the meaningful tg number — it amortizes the per-step CPU sync. Set `--pp 512 --tg 128` for comparable results with `llama.cpp`.
- `--mode microbench`: per-kernel breakdown (us/call × calls/step) so you can see where tg time is spent.

To rebuild the model directory from a `.gguf`:

```
python3 extract.py path/to/Bonsai-4B.gguf --out ./model --prompt "Once upon a time"
```

`extract.py` depends on `gguf` (currently sourced via `sys.path.insert(0, '/tmp/llama.cpp/gguf-py')` — adjust if your llama.cpp checkout lives elsewhere) and the `regex` package for the Qwen2 pretokenizer regex. It also BPE-encodes the prompt and writes it as `prompt.bin`, so changing the prompt requires re-running `extract.py`.

## Architecture

### Files

- `src/main.rs` — CLI arg parsing; dispatches to `forward::generate` / `bench` / `microbench_tg`.
- `src/model.rs` — config + manifest loading, GPU device/buffer/pipeline/BGL setup, RoPE table precompute, activation-buffer layout.
- `src/forward.rs` — the entire forward pass and generation loop. Two end-to-end paths (matvec / matmul) plus encoder + uniform-pool plumbing. Long file (~1.2k lines) but organized top-down: helpers → bench → generate → matvec step → matmul prefill.
- `src/shaders/*.wgsl` — one shader per kernel kind. The matvec/matmul shaders are the two perf-critical ones.
- `extract.py` — GGUF → flat-file converter and BPE prompt encoder.

### Q1_0 weight format and the multiply-free matvec

Q1_0 stores 128 weights per block as 16 bytes of sign bits (+1/-1 per weight) + a 2-byte FP16 scale `d` — 18 bytes per block. `extract.py` splits each Q1_0 tensor into a contiguous **d-array** (FP16 scales) followed by a **qs-array** (raw 16-byte sign blocks); the manifest in `config.json` records `d_offset`, `qs_offset`, and `nb` (blocks per row) for every tensor. Both halves are u32-aligned — all WGSL reads are word loads.

The hot-path kernel `shaders/matvec_q1_0.wgsl` therefore needs **no multiplications inside the inner loop** — it accumulates `±x` per weight via `select(-xv, xv, bit_set)` and only multiplies by `d` once per block. `matvec_q1_0_fused.wgsl` packs 2- or 3-range dispatches (QKV; gate+up) into one workgroup to amortize x-load cost.

### Two execution paths

The model is run in one of two regimes, chosen at the call-site (`forward.rs` has both):

1. **Single-token (matvec) path** — `step_matvec` / `encode_step_matvec` / `layer_pre_kv_in_pass` / `layer_post_kv_in_pass`. Used for **all of `--mode gen`** and for the generation phase of `--mode prompt`. Operates on `m=1` token; uses `matvec_q1_0` and the fused variant. The whole forward step (embed → 36× transformer layers → output_norm → LM head → argmax) is encoded into a few compute passes per step. Pipelined-gen mode chains CHUNK=8 steps per CB via the `sample` buffer (each step's argmax writes the next step's input-token slot — no CPU sync until the chunk finishes).

2. **Batched-prefill (matmul) path** — `prefill_matmul` / `layer_step_matmul`. Used by `--mode prompt`. Quantizes activations to Q8_0 (`quantize_q8_0.wgsl`), then uses `dot4I8Packed`-based `matmul_q1_0_q8_0.wgsl` for the projections. Has its own per-token KV-copy choreography: K/V for all M tokens are written to the cache after rope.

These two paths use different shaders (`matvec_q1_0*` vs `matmul_q1_0_q8_0`) and different bind-group layouts (`bgls.matvec` vs `bgls.matmul`).

### GPU memory layout

All weights live in **5 storage buffers** grouped by role: `w_attn` (per-layer Wq/Wk/Wv/Wo), `w_ffn_gu` (Wgate/Wup), `w_ffn_d` (Wdown), `w_norms` (FP32 norm vectors), `w_embed` (token_embd, used as both embed and tied LM head). Tensor offsets within each buffer come from `cfg.manifest` — always look weights up via `model::tensor(cfg, name)` rather than hard-coding offsets.

`Buffers::act` is one FP32 buffer with named regions (`ActLayout` in `model.rs`: `x`, `x_norm`, `q`, `k_cur`, `v_cur`, `attn_out`, `gate`, `up`, `ffn_in`, `logits`). Sized for `M_MAX=512` tokens (the prefill batch size cap).

KV cache is split into `kv_k` and `kv_v`, sized as `n_layer * MAX_SEQ * kv_dim * 4` bytes each. `MAX_SEQ=1024`.

`act_q8` is the Q8_0 activation scratch used only on the matmul path (FP32 d-section followed by i8 qs-section).

### Uniforms via dynamic offsets

There is **one** `uniform` buffer (`UNIFORM_POOL_SLOTS * UNIFORM_SLOT_SIZE = 65536 * 256` bytes). Every dispatch's params struct is appended into a `UniformPool` (CPU-side `Vec<u8>`), and the dynamic offset is recorded; the pool is flushed in one `write_buffer` at `StepEncoder::finish`. All `Params` structs in `model.rs` are ≤ 64 bytes and packed into 256-byte slots. **Every BGL has its UBO at binding 0 with `has_dynamic_offset: true`** — bind-group helpers in `forward.rs::make_bg` set this up.

### Bind-group layout discipline

Activation buffers always go through **one** `read_write` storage binding per bind group — never aliased as both `read` and `read_write` within a single dispatch. This is enforced by the bind-group construction in `model.rs` (the `rw_mask` argument to `make_bgl`); when adding a kernel, follow the same pattern.

### Encoder organization (perf-critical)

`begin_compute_pass` costs ~25us on RADV. The `_in_pass` family of helpers (`dispatch_rms_norm`, `dispatch_matvec_q1_0`, `dispatch_matvec_q1_0_fused`, `dispatch_silu_mul`, `dispatch_rope`) all accept a caller-provided `&mut wgpu::ComputePass<'_>` so many dispatches share one pass. The matvec generation step is encoded as **two** big passes (one before the per-layer KV-copy `copy_buffer_to_buffer`, one after) to amortize this cost — see `encode_step_matvec`. The plain wrappers (`rms_norm`, `silu_mul`, `matmul_q1_0`, `quantize_act`, etc.) open their own pass and are used in the batched-prefill path where pass-cost is amortized over batch size.

When adding a new dispatch in tg, prefer the `_in_pass` form and slot it into an existing pass. When adding to prefill, the per-pass wrapper is fine.

### Tied embeddings

Bonsai-4B has tied embeddings: `token_embd.weight` is used as both the embedding lookup (in `embed.wgsl`) and the LM head full matvec. The Q1_0 row layout `[n_embd, n_vocab]` happens to be the right shape for both ops (gather a single row for embed, full matvec for LM head). `cfg.tied_embeddings` is set by `extract.py`; current code paths assume tied embeddings.

### Sample / readback ring

`buffers.sample` is a 1024-u32 storage buffer used both as the embed input ("which token to look up") and the argmax output. Pipelined gen indexes successive slots `0..=N` in one CB so CPU never has to read back per step. `buffers.readback` is the matching `MAP_READ` buffer; the only CPU-side awaits in normal generation are the final `device.poll` + `map_async` after all chunks have been submitted.

### Adapter / limits

`Model::load` clamps `max_storage_buffer_binding_size` and `max_buffer_size` to `min(adapter_limit, 1 GB)` but raises the floor to 300 MB so the largest grouped weight buffer (~252 MB) fits. `max_storage_buffers_per_shader_stage` is bumped to ≥ 8.

## When making changes

- **Adding a new kernel**: write `shaders/foo.wgsl`, add a `FooParams` struct (≤ 64 bytes, `Pod + Zeroable + repr(C)`) in `model.rs`, register a BGL in `BindGroupLayouts` (single-rw discipline above), build a pipeline in `Pipelines`. Then add a `dispatch_foo` (in-pass) helper in `forward.rs` and slot it into the layer pipeline.
- **Modifying weight layout**: changes to Q1_0 packing or to the grouping of tensors into the 5 buffers must be made in **both** `extract.py` (writer) and `model.rs` / shaders (reader), and the manifest format in `config.json` will need to round-trip. Re-extract the model dir after any layout change.
- **Perf work**: use `--mode microbench` for per-kernel deltas; use `--mode bench` (specifically the `tg{N} pipe` row and `pp{N}`) for end-to-end. Most tg time is in matvec dispatches (LM head, ffn_down, attn_output, fused QKV, fused gate-up); rms_norm/silu/rope/attention are minor. Recent commits document the wins from multi-row matvec, fused QKV/gate-up, single-pass-per-phase, flash-attn online softmax, and pipelined generation — keep these intact when refactoring.
