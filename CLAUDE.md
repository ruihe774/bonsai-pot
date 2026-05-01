# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`bonsai-pot` is a from-scratch, dependency-light **Bonsai-4B (Qwen3-architecture) Q1_0 inference engine** built on **wgpu compute shaders**. There is no `llama.cpp`, `ggml`, or PyTorch on the hot path — weights are loaded from a custom flat-file layout (produced by `scripts/extract.py` from a GGUF), all kernels are hand-rolled WGSL, and the host side is plain Rust + wgpu 29 + pollster.

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

`<model_dir>` is the output of `scripts/extract.py` (default `./model`). It must contain `config.json`, the five `weights_*.bin` files, `vocab.bin`, `vocab_offsets.bin`, and (for the tokenizer) `merges.txt`. The runtime no longer reads `prompt.bin`; prompts come in over stdin from `scripts/bpe.py`.

There is no separate test suite — `--mode gen` / `--mode prompt` plus parity diffs against captured baselines (and `examples/chat.rs`) are the correctness harness.

- `--mode gen` (default): single-token matvec path for both prompt and generation (multiply-free Q1_0 hot path).
- `--mode prompt`: batched dot4I8Packed matmul prefill (with a Q8_0 activation quantize pre-pass), then matvec for generation.
- `--mode bench`: prints an `llama-bench`-style table with pp/tg t/s. (There used to be a `tg{N} pipe` row from CHUNK=8 pipelining; that path was removed when sampling moved off the GPU.)
- `--mode microbench`: per-kernel breakdown (us/call × calls/step) so you can see where tg time is spent.

CLI sampling flags: `--temperature`, `--top-k`, `--top-p`, `--seed`. Default is greedy (`--temperature 0.0`). Greedy runs are byte-deterministic; stochastic runs are reproducible per seed.

## Rebuilding the model directory

```
uv run scripts/extract.py path/to/Bonsai-4B.gguf --out ./model
```

Both Python scripts use [PEP 723 inline metadata](https://peps.python.org/pep-0723/) — `uv run` reads the dependency block at the top of each script and runs it in an isolated env. No virtualenv setup or `pip install` needed; just have `uv` on `$PATH`.

`scripts/extract.py` writes weights / vocab / `merges.txt` / `config.json`. It does **not** encode prompts. To encode a prompt:

```
uv run scripts/bpe.py ./model "Once upon a time" --out ./model/prompt.bin
# or stream directly:
uv run scripts/bpe.py ./model "Once upon a time" | cargo run ...
```

`scripts/bpe.py` depends only on the `regex` package (for `\p{L}` / `\p{N}` in the Qwen2 pretokenizer regex) — no `gguf`, no GPU, no compilation. It reads `vocab.bin` / `vocab_offsets.bin` / `merges.txt` from the model dir. Dependencies are declared inline in PEP 723 format and resolved automatically by `uv run`. By default it splits the input on `<|...|>` literals that exist in the vocab (e.g. `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`) and emits each as its atomic token id, so ChatML-rendered prompts round-trip correctly. Pass `--no-specials` to byte-level-encode them instead.

## Architecture

### Files

- `src/lib.rs` — public API surface, re-exports.
- `src/model.rs` — config + manifest loading, GPU device/buffer/pipeline/BGL setup, RoPE table precompute, activation-buffer layout. Owns the public `Model` and `ModelConfig` types.
- `src/session.rs` — public `Session<'m>` (per-conversation state), `Sampler`, `GenerateOptions`, `StopReason`, and the CPU-side sampler (temperature → top-p → multinomial via SplitMix64-seeded uniform).
- `src/forward.rs` — entire forward pass and per-step inference helpers. Two end-to-end paths (matvec / matmul) plus encoder + uniform-pool plumbing, plus a `bench_internals` submodule gated on the `bench-internals` feature. Long file (~900 lines) but organized top-down: helpers → step encoders → matmul prefill → bench/microbench.
- `src/error.rs` — `PotError` + `Result` with hand-rolled `Display`/`Error` impls (no `thiserror` dep).
- `src/decode.rs` — inverse of GPT-2 byte-level vocab encoding (codepoint → raw byte), used by `Model::decode_token`.
- `src/bin/bonsai-pot.rs` — demo CLI on top of the public lib API. Argv parsing, stdin u32 reader, sampler construction, calls into `Session`. Routes `--mode bench`/`microbench` to `bonsai_pot::__bench` (only available when built with `--features bench-internals`).
- `src/shaders/*.wgsl` — one shader per kernel kind. The matvec/matmul shaders are the two perf-critical ones; `topk_reduce.wgsl` is the new sampler-helper kernel.
- `examples/chat.rs` — interactive ChatML REPL built on the public library API. Renders the Qwen-style `<|im_start|>...<|im_end|>` chat template per turn, shells out to `scripts/bpe.py` for tokenization, batched-matmul prefills the first turn (`pos==0`) and matvec-loop prefills subsequent turns, then streams generation with `Session::step` until `<|im_end|>`. Persists KV state across turns; `/reset` clears it. Demonstrates that out-of-process tokenization is fast enough for an interactive UX (subprocess startup is ~tens of ms vs. seconds of GPU work).
- `scripts/extract.py` — GGUF → flat-file converter. Writes weights + vocab + merges + config.
- `scripts/bpe.py` — standalone BPE encoder; reads model dir, writes u32 token IDs.

### Q1_0 weight format and the multiply-free matvec

Q1_0 stores 128 weights per block as 16 bytes of sign bits (+1/-1 per weight) + a 2-byte FP16 scale `d` — 18 bytes per block. `scripts/extract.py` splits each Q1_0 tensor into a contiguous **d-array** (FP16 scales) followed by a **qs-array** (raw 16-byte sign blocks); the manifest in `config.json` records `d_offset`, `qs_offset`, and `nb` (blocks per row) for every tensor. Both halves are u32-aligned — all WGSL reads are word loads.

The hot-path kernel `shaders/matvec_q1_0.wgsl` therefore needs **no multiplications inside the inner loop** — it accumulates `±x` per weight via `select(-xv, xv, bit_set)` and only multiplies by `d` once per block. `matvec_q1_0_fused.wgsl` packs 2- or 3-range dispatches (QKV; gate+up) into one workgroup to amortize x-load cost.

### Two execution paths

The model is run in one of two regimes, selected by the call-site (`forward.rs` has both):

1. **Single-token (matvec) path** — `step_matvec_topk` / `encode_step_matvec` / `layer_pre_kv_in_pass` / `layer_post_kv_in_pass`. Used for **all of `--mode gen`** and for the generation phase of `--mode prompt`. Operates on `m=1` token; uses `matvec_q1_0` and the fused variant. The whole forward step (embed → 36× transformer layers → output_norm → LM head → topk_reduce) is encoded into a few compute passes per step. After the GPU step, the CPU reads back K (default K_MAX=32) candidates from `sample[0..2K]` and performs CPU-side sampling. Attention in this path uses the split-K kernel pair (see below).

2. **Batched-prefill (matmul) path** — `prefill_matmul_topk` / `layer_step_matmul`. Used by `Session::prefill` and the bin's `--mode prompt`. Quantizes activations to Q8_0 (`quantize_q8_0.wgsl`), then uses `dot4I8Packed`-based `matmul_q1_0_q8_0.wgsl` for the projections. Has its own per-token KV-copy choreography: K/V for all M tokens are written to the cache after rope. Requires `pos_base == 0` (assumes a fresh KV cache); for incremental prefill into an existing context, use `Session::prefill_one_at_a_time` (matvec-loop variant). Attention in this path uses the original single-pass `attention.wgsl` (`m_tokens=M`, `is_prefill=1`).

These two paths use different shaders (`matvec_q1_0*` vs `matmul_q1_0_q8_0`) and different bind-group layouts (`bgls.matvec` vs `bgls.matmul`).

### Attention: split-K + GQA-batched flash-attention (matvec / tg path)

For tg (`m_tokens=1`), the matvec path uses a two-kernel split-K flash-attention pipeline that decouples per-step latency from KV length. `attention_split.wgsl` is dispatched as `(n_kv_head, n_chunks_active, 1)`; each workgroup processes one `(kv_group, chunk)` pair, scanning `[chunk * ATTN_CHUNK_SIZE, min((chunk+1) * ATTN_CHUNK_SIZE, pos))`. The four Q heads sharing the KV group are processed together so K/V loads are reused 4×; the four Q·K dots are packed into a single `vec4<f32>` and reduced with `subgroupAdd` (and a cross-subgroup merge if the device's subgroup is smaller than the workgroup, see below). Per-chunk `(m, l, o)` partials are written to `attn_partials`. `attention_merge.wgsl` then runs `(n_head, 1, 1)`, doing a flash-attention log-sum-exp combination across the active chunks. Both dispatches share the same compute pass — wgpu inserts the storage-buffer barrier between them. `ATTN_CHUNK_SIZE=32`; `n_chunks_active = ceil(pos / 32)`. Requires `Features::SUBGROUP`.

The original single-pass `attention.wgsl` is retained for the matmul prefill path, where the per-token causal lengths vary across the `m_tokens` dimension and the `is_prefill=1` branch does the `cur_pos = m_tok + 1` computation.

### Subgroup ops and the runtime-baked SG_SIZE

`rms_norm`, `attention`, and `attention_split` use `subgroupAdd` instead of a shmem barrier-tree, with a parametrized cross-subgroup merge for hardware where `subgroup_size < workgroup_size`. The size used at codegen time is determined by **dispatching a probe shader** (`probe_subgroup.wgsl`) that writes the runtime `subgroup_size` builtin to a readback buffer; the value is then substituted into shader source via `{{SG_SIZE}}` text replacement before `create_shader_module`. We probe at runtime rather than reading `adapter.get_info().subgroup_max_size` because the latter reports the device range (e.g. 32..=64 on AMD RDNA), not the size the driver actually chose for our compute pipelines — `RADV_PERFTEST=cswave32` flips RDNA from wave64 to wave32 without changing the adapter info, and `VK_EXT_subgroup_size_control` (not exposed by wgpu 29) would do the same.

The resulting `const N_SG = WG / SG_SIZE` makes the merge branch a constant, so on devices where `SG_SIZE == WG` (e.g. AMD wave64) the fast path const-folds and no cross-subgroup barriers run. `matvec_q1_0` and `matvec_q1_0_fused` use a `subgroupShuffleXor` butterfly for the per-row 8-lane reduction; this assumes `subgroup_invocation_id` increases linearly with `local_invocation_index` (true on AMD/NVIDIA/Intel/Apple) and `SG_SIZE >= 8` so XOR mask 4 stays within a subgroup. Hardware with runtime subgroup size < 8 or non-power-of-2 is rejected at load.

To exercise the wave32 / multi-subgroup merge path on AMD RDNA, run with `RADV_PERFTEST=cswave32`. RX 9070 (RDNA4) at default wave64 hits ~87 t/s tg1024; under wave32 it's ~87 t/s tg1024, ~1340 t/s pp512 (vs. ~1484 t/s wave64) — wave32 has a ~10% prefill cost on this hardware but generation is unaffected.

### Sampling: hybrid GPU top-K → CPU finish

There is no GPU argmax shader. After the LM-head matvec, `topk_reduce.wgsl` (single workgroup, WG=64, K_MAX=32) reduces the full logits array to top-K candidates: each thread maintains a per-thread top-K_MAX min-heap in shmem, then a halving-tree two-pointer merge produces the global top-K. Output: K `f32` logits (descending) followed by K `u32` indices, written to the `sample` buffer.

The CPU then reads `sample[0..2K]` back and finishes sampling: temperature scale → softmax → top-p nucleus filter → multinomial via xorshift/SplitMix64 PRNG seeded by `(sampler.seed + pos)`. Implementation in `session.rs::sample_from_topk`. With `temperature == 0.0` this short-circuits to argmax over the K candidates — which is exact, because the global max is always `sample[0]`.

This design intentionally trades the old CHUNK=8 pipelined-gen path (which required sampling on-GPU to chain via `sample[]`) for sampler flexibility. The perf cost is ~22% of tg t/s vs the pipelined version (see commit history).

### GPU memory layout

All weights live in **5 storage buffers** grouped by role: `w_attn` (per-layer Wq/Wk/Wv/Wo), `w_ffn_gu` (Wgate/Wup), `w_ffn_d` (Wdown), `w_norms` (FP16 norm vectors), `w_embed` (token_embd, used as both embed and tied LM head). Tensor offsets within each buffer come from `cfg.manifest` — always look weights up via `model::tensor(cfg, name)` rather than hard-coding offsets.

`Buffers::act` is one f16 buffer with named regions (`ActLayout` in `model.rs`: `x`, `x_norm`, `q`, `k_cur`, `v_cur`, `attn_out`, `gate`, `up`, `ffn_in`, `logits`). Sized for `M_MAX=512` tokens (the prefill batch size cap).

KV cache is split into `kv_k` and `kv_v`, sized as `n_layer * MAX_SEQ * kv_dim * 2` bytes each (f16). `MAX_SEQ=1024`.

`act_q8` is the Q8_0 activation scratch used only on the matmul path (FP32 d-section followed by i8 qs-section).

### Uniforms via dynamic offsets

There is **one** `uniform` buffer (`UNIFORM_POOL_SLOTS * UNIFORM_SLOT_SIZE = 4096 * 256 = 1,048,576 bytes = 1 MiB`). Every dispatch's params struct is appended into a `UniformPool` (CPU-side `Vec<u8>`), and the dynamic offset is recorded; the pool is flushed in one `write_buffer` at `StepEncoder::finish`. All `Params` structs in `model.rs` are ≤ 64 bytes and packed into 256-byte slots. **Every BGL has its UBO at binding 0 with `has_dynamic_offset: true`** — bind-group helpers in `forward.rs::make_bg` set this up.

### Bind-group layout discipline

Activation buffers always go through **one** `read_write` storage binding per bind group — never aliased as both `read` and `read_write` within a single dispatch. This is enforced by the bind-group construction in `model.rs` (the `rw_mask` argument to `make_bgl`); when adding a kernel, follow the same pattern.

### Encoder organization (perf-critical)

`begin_compute_pass` costs ~25us on RADV. The `_in_pass` family of helpers (`dispatch_rms_norm`, `dispatch_matvec_q1_0`, `dispatch_matvec_q1_0_fused`, `dispatch_silu_mul`, `dispatch_rope`, `dispatch_topk_reduce`) all accept a caller-provided `&mut wgpu::ComputePass<'_>` so many dispatches share one pass. The matvec generation step is encoded as **two** big passes (one before the per-layer KV-copy `copy_buffer_to_buffer`, one after) to amortize this cost — see `encode_step_matvec`. The plain wrappers (`rms_norm`, `silu_mul`, `matmul_q1_0`, `quantize_act`, etc.) open their own pass and are used in the batched-prefill path where pass-cost is amortized over batch size.

When adding a new dispatch in tg, prefer the `_in_pass` form and slot it into an existing pass. When adding to prefill, the per-pass wrapper is fine.

### Tied embeddings

Bonsai-4B has tied embeddings: `token_embd.weight` is used as both the embedding lookup (in `embed.wgsl`) and the LM head full matvec. The Q1_0 row layout `[n_embd, n_vocab]` happens to be the right shape for both ops (gather a single row for embed, full matvec for LM head). `cfg.tied_embeddings` is set by `scripts/extract.py`; current code paths assume tied embeddings.

### Sample / readback layout

`buffers.sample` is a 1024-u32 storage buffer used in two roles within a single step:
1. **Input** during embed: CPU writes the input token ID to `sample[0]` (or all M prompt token IDs to `sample[0..M]` for matmul prefill).
2. **Output** from `topk_reduce`: K f32 logits at `sample[0..K]` (bitcast to u32) followed by K u32 vocab indices at `sample[K..2K]`.

Since embed runs before topk_reduce in any step, the two roles never alias. `buffers.readback` is the matching `MAP_READ` buffer; per-step generation does one `device.poll` + `map_async` (the CPU sampler can't run until the readback completes — there's no pipelining anymore).

### Adapter / limits

`Model::load` clamps `max_storage_buffer_binding_size` and `max_buffer_size` to `min(adapter_limit, 1 GB)` but raises the floor to 300 MB so the largest grouped weight buffer (~252 MB) fits. `max_storage_buffers_per_shader_stage` is bumped to ≥ 8.

## When making changes

- **Adding a new kernel**: write `shaders/foo.wgsl`, add a `FooParams` struct (≤ 64 bytes, `Pod + Zeroable + repr(C)`) in `model.rs`, register a BGL in `BindGroupLayouts` (single-rw discipline above), build a pipeline in `Pipelines`. Then add a `dispatch_foo` (in-pass) helper in `forward.rs` and slot it into the layer pipeline.
- **Modifying weight layout**: changes to Q1_0 packing or to the grouping of tensors into the 5 buffers must be made in **both** `scripts/extract.py` (writer) and `model.rs` / shaders (reader), and the manifest format in `config.json` will need to round-trip. Re-extract the model dir after any layout change.
- **Modifying the tokenizer / pretok regex**: changes to vocab encoding must be made in **both** `scripts/extract.py` (which writes `vocab.bin` + `merges.txt`) and `scripts/bpe.py` (which encodes prompts) and possibly `src/decode.rs` (which inverts the byte-level mapping). The Rust runtime never tokenizes, only decodes.
- **Public API changes**: re-exports live in `src/lib.rs`. Anything not re-exported there is internal and may change without notice; keep `Model`, `ModelConfig`, `Session`, `Sampler`, `GenerateOptions`, `StopReason`, `PotError`, `Result`, `TOPK_MAX` stable.
- **Perf work**: use `--mode microbench` for per-kernel deltas; use `--mode bench` (specifically `tg{N}` and `pp{N}`) for end-to-end. Most tg time is in matvec dispatches (LM head, ffn_down, attn_output, fused QKV, fused gate-up) plus `topk_reduce` (currently ~1.1 ms/step, a notable fraction of total step time); rms_norm/silu/rope/attention are minor. Recent commits document the wins from multi-row matvec, fused QKV/gate-up, single-pass-per-phase, flash-attn online softmax, and (in earlier history) pipelined generation — keep these intact when refactoring.
