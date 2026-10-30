# bonsai-pot

From-scratch, dependency-light LLM inference engines for the **Bonsai** model family, running on **wgpu compute shaders**. This repository is a Cargo workspace with two independent crates, one per model generation:

| Crate | Models | Quantization | Backends | Docs |
|---|---|---|---|---|
| [`bonsai-pot`](bonsai-pot/) | Bonsai 4B / 8B (Qwen3 architecture) | Q1_0 (binary), Q2_0 (ternary), Q8_0 | Vulkan, Metal | [README](bonsai-pot/README.md) · [CLAUDE.md](bonsai-pot/CLAUDE.md) |
| [`bonsai2-pot`](bonsai2-pot/) | Ternary-Bonsai-2-27B (Qwen3.5 hybrid attention) | PTQ1_0 (1.75 bits/weight) | Vulkan | [README](bonsai2-pot/README.md) · [CLAUDE.md](bonsai2-pot/CLAUDE.md) |

Pick the crate that matches your model; the two do not depend on each other and use different weight layouts, so a model directory produced for one is not valid for the other.

## Highlights

- **Weights are never dequantized.** Packed weights are consumed directly by the matvec/matmul kernels, with no intermediate FP16 weight tensor.
  - `bonsai-pot`: Q1_0 is one sign bit selecting `+x` / `-x`, Q2_0 a 2-bit ternary code, and Q8_0 an i8 fed to `dot4I8Packed`. One FP16 scale multiply closes each 128-weight block, so the inner loop has zero float multiplications per weight.
  - `bonsai2-pot`: PTQ1_0 packs five base-3 trits per byte. Decode matvec unpacks trits on the fly and uses `dotPacked4x8EXT` against Q8 activations. Prefill decodes each tile once into workgroup memory and multiplies with int8 cooperative matrices (WMMA), falling back to DP4a.
- **No `llama.cpp`, `ggml` or PyTorch on the hot path.** Weights come from a flat-file layout produced from a GGUF by `scripts/extract.py`, every kernel is hand-rolled GLSL compiled to SPIR-V at build time, and the host side is plain Rust + wgpu.
- **Fast on consumer GPUs, with realistic numbers.** End-to-end throughput includes stochastic sampling and encode pipelining, not just bare-forward microbenchmarks.
  - `bonsai-pot` on an RX 9070: Bonsai-4B reaches ~2284 t/s prefill and ~385 t/s decode, and Bonsai-8B ~1294 / ~256 t/s.
  - `bonsai2-pot` on an RX 9070: the 27B model reaches ~483 t/s prefill and ~76 t/s end-to-end decode, vs 54.6 t/s for the PrismML llama.cpp fork on Vulkan.
- **Verified against a reference.** `bonsai2-pot` produces greedy generations token-identical to the PrismML llama.cpp fork, with per-layer residual cosine ≥ 0.9998.
- **Small, embeddable library.** Each crate exposes `Model`, `Session`, `Sampler` and `GenerateOptions`. `bonsai-pot` is imported as `bonsai_pot`, and `bonsai2-pot` as `bonsai2_pot`. Each also ships a demo CLI and an interactive chat example. `no_std + alloc` is supported without the default `std` feature.
- **Tokenization stays out of the Rust crates.** Use the `scripts/bpe.py` in each crate and pipe `u32` ids into the CLI.

See each crate's README for full benchmarks, architecture notes, prerequisites and usage.

## Repository layout

```
Cargo.toml          workspace manifest (shared [profile.*]; members below)
Cargo.lock          shared lockfile
LICENSE             Unlicense
.github/workflows/  CI (macOS/Metal job for bonsai-pot, Linux job for bonsai2-pot)
bonsai-pot/         Bonsai 4B/8B crate:   src/, src/shaders/, build.rs, examples/, tests/, scripts/
bonsai2-pot/        Ternary-Bonsai-2 crate: same layout, plus scripts/llama_dump/ reference tooling
```

Each crate is self-contained: its own `build.rs` (shader compilation), `scripts/` (`extract.py`, `bpe.py`), `examples/chat.rs` and `tests/`. Model directories (`model*/`) are gitignored.

## Quick start

Build tools (`glslangValidator`, `spirv-opt`, and `uv` for the helper scripts) must be on `$PATH`; see the crate READMEs for exact versions.

```sh
cargo build --release -p bonsai-pot  --features bench-internals   # binary: target/release/bonsai-pot
cargo build --release -p bonsai2-pot --features bench-internals   # binary: target/release/bonsai2-pot
```

Always pass `-p <crate>`: running `cargo test` or `cargo run` from the root without it affects both crates. Run the `scripts/` helpers and the CLI from inside the crate directory, since paths like `./model` are relative to it.

## License

[Unlicense](LICENSE).
