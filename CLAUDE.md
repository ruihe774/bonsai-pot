# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`bonsai-pot` is a from-scratch, dependency-light **Bonsai (Qwen3-architecture) Q1_0 inference engine** built on **wgpu compute shaders**. It supports the Bonsai 4B and 8B model sizes. There is no `llama.cpp`, `ggml`, or PyTorch on the hot path — weights are loaded from a custom flat-file layout (produced by `scripts/extract.py` from a GGUF), all kernels are hand-rolled WGSL, and the host side is plain Rust + wgpu 29 + pollster.

The crate exposes both:
- a **library** (`bonsai_pot::{Model, Session, Sampler, GenerateOptions, …}`) for embedding the engine in other Rust programs;
- a **demo CLI** (`src/bin/bonsai-pot.rs`) that reads pre-tokenized u32 prompts from stdin and prints decoded output. The CLI bundles bench/microbench utilities behind the `bench-internals` feature.

Tokenization is intentionally out of the Rust crate. Use `scripts/bpe.py` to BPE-encode prompts; pipe its u32 output into the bin.

## Build / run / bench

```
# Library only:
cargo build --release --lib

# Demo CLI (pulls in the bench/microbench helpers):
cargo build --release --features bench-internals

# End-to-end run: tokenize then generate. `uv run` resolves the script's
# inline dependencies on the fly — no separate `pip install` step.
uv run scripts/bpe.py ./model "Once upon a time" \
  | cargo run --release --features bench-internals -- ./model \
        --mode prompt --max-new-tokens 64

# Benches and microbench don't need stdin:
cargo run --release --features bench-internals -- ./model --mode bench --pp 512 --tg 128
cargo run --release --features bench-internals -- ./model --mode microbench
```

`<model_dir>` is the output of `scripts/extract.py` (default `./model`). It must contain `config.ini`, the five `weights_*.bin` files, `vocab.bin`, `vocab_offsets.bin`, and (for the tokenizer) `merges.txt`. The runtime no longer reads `prompt.bin`; prompts come in over stdin from `scripts/bpe.py`.

Tests live in `tests/gpu_integration.rs` (end-to-end on a real GPU against `./model`) plus unit tests in `src/session.rs` (CPU sampler) and `src/kv_snapshot.rs` (header round-trips). Run with `cargo test --release`. Beyond that, `--mode gen` / `--mode prompt` plus parity diffs against captured baselines (and `examples/chat.rs`) are the correctness harness.

- `--mode gen` (default): single-token matvec path for both prompt and generation (multiply-free Q1_0 hot path).
- `--mode prompt`: batched dot4I8Packed matmul prefill (with a Q8_0 activation quantize pre-pass), then matvec for generation.
- `--mode bench`: prints an `llama-bench`-style table with pp/tg t/s. (There used to be a `tg{N} pipe` row from CHUNK=8 pipelining; that path was removed when sampling moved off the GPU.)
- `--mode microbench`: per-kernel breakdown (us/call × calls/step) so you can see where tg time is spent.

CLI sampling flags: `--temperature`, `--top-k`, `--top-p`, `--seed`. Default is greedy (`--temperature 0.0`). Greedy runs are byte-deterministic; stochastic runs are reproducible per seed.

## Rebuilding the model directory

```
uv run scripts/extract.py path/to/Bonsai-4B-Q1_0.gguf --out ./model
# or for 8B:
uv run scripts/extract.py path/to/Bonsai-8B-Q1_0.gguf --out ./model-8b
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
- `src/model.rs` — config + manifest loading, GPU device/buffer/pipeline/BGL setup, RoPE table precompute, activation-buffer layout. Owns the public `Model`, `ModelConfig`, and `ModelOptions` types.
- `src/session.rs` — public `Session<'m>` (per-conversation state), `Sampler`, `GenerateOptions`, `StopReason`, and the CPU-side sampler (temperature → top-p → multinomial via SplitMix64-seeded uniform).
- `src/kv_snapshot.rs` — `KvSnapshot`: host-resident, persistable copy of the GPU KV cache for a `Session` at some `pos`. Used by `Session::snapshot` / `Session::restore`.
- `src/forward.rs` — entire forward pass and per-step inference helpers. Two end-to-end paths (matvec / matmul) plus encoder + uniform-pool plumbing, plus a `bench_internals` submodule gated on the `bench-internals` feature. Long file (~1.9k lines), organized top-down: helpers → step encoders → matmul prefill → bench/microbench.
- `src/error.rs` — `PotError` + `Result` with hand-rolled `Display`/`Error` impls (no `thiserror` dep).
- `src/decode.rs` — inverse of GPT-2 byte-level vocab encoding (codepoint → raw byte), used by `Model::decode_token`.
- `src/bin/bonsai-pot.rs` — demo CLI on top of the public lib API. Argv parsing, stdin u32 reader, sampler construction, calls into `Session`. Routes `--mode bench`/`microbench` to `bonsai_pot::__bench` (only available when built with `--features bench-internals`). Exposes `--max-seq` to size the KV cache.
- `src/shaders/*.wgsl` — one shader per kernel kind. The matvec/matmul shaders are the perf-critical ones; `topk_reduce.wgsl` is the sampler helper; `kv_writeback_fused.wgsl` does the K-side rms_norm + `*w_k_norm` + NEOX-RoPE + Q8_0 quantize + write into `kv_k`, plus V Q8_0 quantize + write into `kv_v`, all in one workgroup per (kv_head, token); `q_norm_rope_fused.wgsl` does the same Q-side rms_norm + `*w_q_norm` + NEOX-RoPE in one workgroup per (head, token), writing back to `act.q` in place; `probe_subgroup.wgsl` is the runtime probe used to bake `SG_SIZE` into the other shaders.
- `examples/chat.rs` — interactive ChatML REPL built on the public library API. Renders the Qwen-style `<|im_start|>...<|im_end|>` chat template per turn and shells out to `scripts/bpe.py` for tokenization. The system prompt is batched-matmul prefilled once at `pos==0`, then `Session::snapshot` captures that KV state to host memory. Each user turn is appended via `prefill_one_at_a_time` (the matvec-loop path, since `pos > 0`), then generation streams with `Session::step` until `<|im_end|>`. `/reset` calls `Session::restore` on the system snapshot (~1–2 ms over PCIe), avoiding a full re-prefill. KV-cache capacity is configurable via `--max-seq`.
- `tests/gpu_integration.rs` — end-to-end tests that load `./model` on a real GPU: model-config sanity, vocab/decode round-trips, prefill error guards, KV snapshot/restore, greedy determinism, and matmul-vs-matvec parity.
- `scripts/extract.py` — GGUF → flat-file converter. Writes weights + vocab + merges + config.
- `scripts/bpe.py` — standalone BPE encoder; reads model dir, writes u32 token IDs.

### Q1_0 weight format and the multiply-free matvec

Q1_0 stores 128 weights per block as 16 bytes of sign bits (+1/-1 per weight) + a 2-byte FP16 scale `d` — 18 bytes per block. `scripts/extract.py` splits each Q1_0 tensor into a contiguous **d-array** (FP16 scales) followed by a **qs-array** (raw 16-byte sign blocks); the manifest in `config.ini` records `d_offset`, `qs_offset`, and `nb` (blocks per row) for every tensor. Both halves are u32-aligned — all WGSL reads are word loads.

The hot-path kernel `shaders/matvec_q1_0.wgsl` therefore needs **no multiplications inside the inner loop** — it accumulates `±x` per weight via `select(-xv, xv, bit_set)` and only multiplies by `d` once per block. `matvec_q1_0_fused.wgsl` packs 2- or 3-range dispatches (QKV; gate+up) into one workgroup to amortize x-load cost.

### Two execution paths

The model is run in one of two regimes, selected by the call-site (`forward.rs` has both):

1. **Single-token (matvec) path** — `step_matvec_topk` / `encode_step_matvec` / `layer_pre_kv_in_pass` / `layer_post_kv_in_pass`. Used for **all of `--mode gen`** and for the generation phase of `--mode prompt`. Operates on `m=1` token; uses `matvec_q1_0` and the fused variant. The whole forward step (embed → 36× transformer layers → output_norm → LM head → topk_reduce) is encoded into **one** compute pass per step — the per-layer `kv_writeback_fused` kernel folds in K's rms_norm + `*w_k_norm` + NEOX-RoPE before quantizing K/V to Q8_0 and writes both into the cache, and `q_norm_rope_fused` folds in Q's rms_norm + `*w_q_norm` + NEOX-RoPE writing back to `act.q` in place — neither breaks the pass. After the GPU step, the CPU reads back up to `TOPK_MAX = 32` candidates from `sample[0..2K]` and performs CPU-side sampling. Attention in this path uses the split-K kernel pair (see below).

2. **Batched-prefill (matmul) path** — `prefill_matmul_topk` / `layer_step_matmul`. Used by `Session::prefill` and the bin's `--mode prompt`. Quantizes activations to Q8_0 (`quantize_q8_0.wgsl`), then uses `dot4I8Packed`-based `matmul_q1_0_q8_0.wgsl` for the projections. `q_norm_rope_fused` and `kv_writeback_fused` handle the Q/K rms_norm + RoPE (and K/V Q8_0 quantize into the cache) for all M tokens. Requires `pos_base == 0` (the matmul attention kernel assumes a fresh KV cache); for incremental prefill into an existing context, use `Session::prefill_one_at_a_time` (matvec-loop variant). Attention in this path uses the original single-pass `attention.wgsl` (`m_tokens=M`, `is_prefill=1`).

These two paths use different shaders (`matvec_q1_0*` vs `matmul_q1_0_q8_0`) and different bind-group layouts (`bgls.matvec` vs `bgls.matmul`).

### Attention: split-K + GQA-batched flash-attention (matvec / tg path)

For tg (`m_tokens=1`), the matvec path uses a two-kernel split-K flash-attention pipeline that decouples per-step latency from KV length. `attention_split.wgsl` is dispatched as `(n_kv_head, n_chunks_active, 1)`; each workgroup processes one `(kv_group, chunk)` pair, scanning `[chunk * ATTN_CHUNK_SIZE, min((chunk+1) * ATTN_CHUNK_SIZE, pos))`. The four Q heads sharing the KV group are processed together so K/V loads are reused 4×; the four Q·K dots are packed into a single `vec4<f32>` and reduced with `subgroupAdd` (and a cross-subgroup merge if the device's subgroup is smaller than the workgroup, see below). Per-chunk `(m, l, o)` partials are written to `attn_partials`. `attention_merge.wgsl` then runs `(n_head, 1, 1)`, doing a flash-attention log-sum-exp combination across the active chunks. Both dispatches share the same compute pass — wgpu inserts the storage-buffer barrier between them. `ATTN_CHUNK_SIZE=32`; `n_chunks_active = ceil(pos / 32)`. Requires `Features::SUBGROUP`.

The original single-pass `attention.wgsl` is retained for the matmul prefill path, where the per-token causal lengths vary across the `m_tokens` dimension and the `is_prefill=1` branch does the `cur_pos = m_tok + 1` computation.

### Subgroup ops and the runtime SUBGROUP_MIN_SIZE bound

`rms_norm`, `attention`, `attention_split`, `attention_merge`, `quantize_q8_0`, and `kv_writeback_fused` use `subgroupAdd` / `subgroupMax` with a runtime cross-subgroup merge when `num_subgroups > 1`. The cross-subgroup shared-memory arrays are sized at compile time for the **worst case**: `SG_PARTIAL_MAX = ceil(WG / SUBGROUP_MIN_SIZE)` slots, where `SUBGROUP_MIN_SIZE` is baked from `adapter.get_info().subgroup_min_size` at `Model::load` time via `{{SUBGROUP_MIN_SIZE}}` text replacement. The actual per-dispatch subgroup count is read from `@builtin(num_subgroups)` and individual subgroup indices from `@builtin(subgroup_id)` — no division by a compile-time constant.

This design is correct under `ALLOW_VARYING_SUBGROUP_SIZE` (which wgpu-hal sets unconditionally for all SUBGROUP-enabled pipelines): the size can differ across pipelines or even across dispatches of the same pipeline; the shmem is always large enough and all branching uses the runtime builtins. The WebGPU subgroups proposal requires either this flag or `VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT` to lock the size; wgpu 29 does not expose the latter, so we rely on the former.

`matvec_q1_0` and `matvec_q1_0_fused` use a `subgroupShuffleXor` butterfly for the per-row 8-lane reduction. This assumes `subgroup_invocation_id` increases linearly with `local_invocation_index` (true on AMD/NVIDIA/Intel/Apple) and `subgroup_size >= 8` so XOR mask 4 stays within a subgroup. The hard requirement `subgroup_min_size >= 8` is validated at `Model::load` against the adapter; hardware below this bound is rejected.

To exercise the wave32 / multi-subgroup merge path on AMD RDNA, run with `RADV_PERFTEST=cswave32`. RX 9070 (RDNA4) at default wave64 (pp512/tg1024, after 1 warmup):

| model    | pp512 t/s | tg1024 t/s |
|----------|----------:|-----------:|
| Bonsai-4B | ~1377    | ~128       |
| Bonsai-8B | ~812     | ~97        |

Under `RADV_PERFTEST=cswave32` (4B), generation is unaffected but prefill carries a ~10% cost on this hardware.

### Sampling: hybrid GPU top-K → CPU finish

There is no GPU argmax shader. After the LM-head matvec, `topk_reduce.wgsl` (single workgroup, WG=64, K_MAX=32) reduces the full logits array to top-K candidates: each thread maintains a per-thread top-K_MAX min-heap in shmem, then a halving-tree two-pointer merge produces the global top-K. Output: K `f32` logits (descending) followed by K `u32` indices, written to the `sample` buffer.

The CPU then reads `sample[0..2K]` back and finishes sampling: temperature scale → softmax → top-p nucleus filter → multinomial via xorshift/SplitMix64 PRNG seeded by `(sampler.seed + pos)`. Implementation in `session.rs::sample_from_topk`. With `temperature == 0.0` this short-circuits to argmax over the K candidates — which is exact, because the global max is always `sample[0]`.

This design intentionally trades the old CHUNK=8 pipelined-gen path (which required sampling on-GPU to chain via `sample[]`) for sampler flexibility. The perf cost is ~22% of tg t/s vs the pipelined version (see commit history).

### GPU memory layout

All weights live in **5 storage buffers** grouped by role: `w_attn` (per-layer Wq/Wk/Wv/Wo), `w_ffn_gu` (Wgate/Wup), `w_ffn_d` (Wdown), `w_norms` (FP16 norm vectors), `w_embed` (token_embd, used as both embed and tied LM head). Tensor offsets within each buffer come from `cfg.manifest` — always look weights up via `model::tensor(cfg, name)` rather than hard-coding offsets.

`Buffers::act` is one f16 buffer with named regions (`ActLayout` in `model.rs`: `x`, `x_norm`, `q`, `k_cur`, `v_cur`, `attn_out`, `gate`, `up`, `ffn_in`, `logits`). Sized for `M_MAX=512` tokens (the prefill batch size cap).

KV cache is split into `kv_k` and `kv_v`, both stored as **Q8_0** (32-element blocks: FP32 scale + 32 i8 quants ⇒ ~2.25 bytes/element, ~12.5% smaller than f16 and producing a `dot4I8Packed`-friendly load shape for the attention kernels). Each buffer has a contiguous d-section followed by a qs-section (see `kv_layer_offsets` / `kv_qs_byte_base` in `forward.rs`); helpers there compute per-layer offsets. Per-step K/V is quantized straight into the cache by `kv_writeback_fused.wgsl` — there is no f16 staging copy. Total per-buffer size is `n_layer * max_seq * kv_dim * 2.25` bytes (≈170 MB combined K+V at `max_seq = 1024`).

`max_seq` is **not** a compile-time constant — it's an allocate-time tunable (`ModelOptions::max_seq`, default 1024, exposed by both the bin and `examples/chat.rs` as `--max-seq`). The `attn_partials` buffer (split-K attention scratch, sized as `n_head * ceil(max_seq / ATTN_CHUNK_SIZE) * (head_dim + 2) * 4` bytes) and the RoPE table (`max_seq * head_dim * 2` bytes) scale with `max_seq`. The `attention_merge.wgsl` constant `MAX_CHUNKS` is **runtime-baked** from `ceil(max_seq / ATTN_CHUNK_SIZE)` at `Model::load` time (like `SUBGROUP_MIN_SIZE`), so there is no hard sequence-length cap in the shader. VRAM is the practical limit; `Model::load` checks the KV buffer against the adapter's `max_buffer_size` and returns a clean error if `max_seq` is too large. Note: the engine has no YaRN/NTK RoPE scaling, so output quality degrades for `max_seq` significantly beyond `~2 × rope_orig_context` (≈16k for 4B, ≈32k for 8B).

`act_q8` is the Q8_0 activation scratch used only on the matmul path (FP32 d-section followed by i8 qs-section).

### Uniforms via dynamic offsets

There is **one** `uniform` buffer (`UNIFORM_POOL_SLOTS * UNIFORM_SLOT_SIZE = 4096 * 256 = 1,048,576 bytes = 1 MiB`). Every dispatch's params struct is appended into a `UniformPool` (CPU-side `Vec<u8>`), and the dynamic offset is recorded; the pool is flushed in one `write_buffer` at `StepEncoder::finish`. All `Params` structs in `model.rs` are ≤ 64 bytes and packed into 256-byte slots. **Every BGL has its UBO at binding 0 with `has_dynamic_offset: true`** — bind-group helpers in `forward.rs::make_bg` set this up.

### Bind-group layout discipline

Activation buffers always go through **one** `read_write` storage binding per bind group — never aliased as both `read` and `read_write` within a single dispatch. This is enforced by the bind-group construction in `model.rs` (the `rw_mask` argument to `make_bgl`); when adding a kernel, follow the same pattern.

### Encoder organization (perf-critical)

`begin_compute_pass` costs ~25us on RADV. The `_in_pass` family of helpers (`dispatch_rms_norm`, `dispatch_matvec_q1_0`, `dispatch_matvec_q1_0_fused`, `dispatch_silu_mul`, `dispatch_kv_writeback_fused`, `dispatch_q_norm_rope_fused`, `dispatch_topk_reduce`) all accept a caller-provided `&mut wgpu::ComputePass<'_>` so many dispatches share one pass. The matvec generation step is encoded as **a single** big pass — see `encode_step_matvec`. (Historical note: an earlier version split it in two around a `copy_buffer_to_buffer` for the K/V cache write; that copy was replaced by the kv_writeback kernel, then expanded into `kv_writeback_fused` which folds in K's rms_norm + RoPE, and finally `q_norm_rope_fused` collapsed the Q-side rms_norm + RoPE into one dispatch — eliminating the pass break and 4 dispatches/layer overall.) The plain wrappers (`rms_norm`, `silu_mul`, `matmul_q1_0`, `quantize_act`, etc.) open their own pass and are used in the batched-prefill path where pass-cost is amortized over batch size.

When adding a new dispatch in tg, prefer the `_in_pass` form and slot it into an existing pass. When adding to prefill, the per-pass wrapper is fine.

### Tied vs. untied embeddings

Bonsai 4B has **tied** embeddings: `token_embd.weight` is used both for the embedding lookup (in `embed.wgsl`) and for the LM head full matvec. Bonsai 8B has **untied** embeddings — it ships a separate `output.weight` tensor. Both layouts share the same `[n_embd, n_vocab]` Q1_0 row shape (gather a single row for embed, full matvec for LM head), so the kernels and the `w_embed` storage buffer are identical; only the byte offsets differ.

`scripts/extract.py` packs `token_embd.weight` (and, when not tied, `output.weight`) consecutively into `weights_embed_lmhead.bin` and sets `cfg.tied_embeddings`. `OutputTensors` in `model.rs` then exposes both `token_embd_*` (used by the embed kernel) and `lm_head_*` (used by the LM head matvec); when tied they coincide.

### Sample / readback layout

`buffers.sample` is a 1024-u32 storage buffer used in two roles within a single step:
1. **Input** during embed: CPU writes the input token ID to `sample[0]` (or all M prompt token IDs to `sample[0..M]` for matmul prefill).
2. **Output** from `topk_reduce`: K f32 logits at `sample[0..K]` (bitcast to u32) followed by K u32 vocab indices at `sample[K..2K]`.

Since embed runs before topk_reduce in any step, the two roles never alias. `buffers.readback` is the matching `MAP_READ` buffer; per-step generation does one `device.poll` + `map_async` (the CPU sampler can't run until the readback completes — there's no pipelining anymore).

### Adapter / limits

`Model::load` raises `max_storage_buffer_binding_size` and `max_buffer_size` to a minimum of 300 MB so the largest grouped weight buffer fits (~252 MB at 4B, ~510 MB at 8B). No upper cap is imposed — the adapter's natural limit is used, which on desktop GPUs is typically 2–4 GB (needed for 8B KV at 32k ctx ≈ 1.27 GB per buffer). `max_storage_buffers_per_shader_stage` is bumped to ≥ 8.

## When making changes

- **Adding a new kernel**: write `shaders/foo.wgsl`, add a `FooParams` struct (≤ 64 bytes, `Pod + Zeroable + repr(C)`) in `model.rs`, register a BGL in `BindGroupLayouts` (single-rw discipline above), build a pipeline in `Pipelines`. Then add a `dispatch_foo` (in-pass) helper in `forward.rs` and slot it into the layer pipeline.
- **Modifying weight layout**: changes to Q1_0 packing or to the grouping of tensors into the 5 buffers must be made in **both** `scripts/extract.py` (writer) and `model.rs` / shaders (reader), and the manifest format in `config.ini` will need to round-trip. Re-extract the model dir after any layout change.
- **Modifying the tokenizer / pretok regex**: changes to vocab encoding must be made in **both** `scripts/extract.py` (which writes `vocab.bin` + `merges.txt`) and `scripts/bpe.py` (which encodes prompts) and possibly `src/decode.rs` (which inverts the byte-level mapping). The Rust runtime never tokenizes, only decodes.
- **Public API changes**: re-exports live in `src/lib.rs`. Anything not re-exported there is internal and may change without notice; keep `Model`, `ModelConfig`, `ModelOptions`, `Session`, `Sampler`, `GenerateOptions`, `StopReason`, `KvSnapshot`, `PotError`, `Result`, `TOPK_MAX` stable.
- **Perf work**: use `--mode microbench` for per-kernel deltas; use `--mode bench` (specifically `tg{N}` and `pp{N}`) for end-to-end. Most tg time is in matvec dispatches (LM head, ffn_down, attn_output, fused QKV, fused gate-up) plus rms_norm (which adds up across 73 calls/step); `topk_reduce` and LM head are each ~0.4–0.6 ms/step on 4B. Attention, silu, rope are minor. Recent commits document the wins from multi-row matvec, fused QKV/gate-up, single-pass-per-phase, flash-attn online softmax, and (in earlier history) pipelined generation — keep these intact when refactoring.
