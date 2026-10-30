# CLAUDE.md

This is a **Cargo workspace of two independent crates**. Read the CLAUDE.md of the crate you are working on before changing code; it holds the architecture, commands and gotchas. This file only covers the repo level.

| Crate | Dir | Target model | Lib name | Binary | Agent docs | User docs |
|---|---|---|---|---|---|---|
| `bonsai-pot` | `bonsai-pot/` | Bonsai 4B/8B, Qwen3 arch; Q1_0 / Q2_0 / Q8_0; Vulkan + Metal | `bonsai_pot` | `bonsai-pot` | [bonsai-pot/CLAUDE.md](bonsai-pot/CLAUDE.md) | [bonsai-pot/README.md](bonsai-pot/README.md) |
| `bonsai2-pot` | `bonsai2-pot/` | Ternary-Bonsai-2-27B, Qwen3.5 hybrid attention; PTQ1_0; Vulkan only | `bonsai2_pot` | `bonsai2-pot` | [bonsai2-pot/CLAUDE.md](bonsai2-pot/CLAUDE.md) | [bonsai2-pot/README.md](bonsai2-pot/README.md) |

## What both share

Both are from-scratch wgpu compute-shader inference engines. Weights are never dequantized (packed weights are consumed directly by the kernels). There is no llama.cpp/ggml/PyTorch on the hot path. GLSL kernels live in `<crate>/src/shaders/` and are compiled to SPIR-V by `<crate>/build.rs`. Tokenization is out-of-crate (`<crate>/scripts/bpe.py`). The shared highlights are in the root [README.md](README.md); keep it in sync when performance or feature claims change.

## Workspace rules

- The crates **do not depend on each other** and share no code. They have independent model formats, shaders and APIs, so a fix in one does not automatically apply to the other. Check whether the other crate needs it.
- Always scope cargo with `-p bonsai-pot` or `-p bonsai2-pot`. The crates use different wgpu fork branches (`staging` vs `relax-4gib-buffer-limits`), so unscoped workspace builds compile wgpu twice.
- `[profile.*]` lives only in the root `Cargo.toml` (Cargo ignores it in members). Clippy lints are per-crate under `[lints.clippy]`.
- Run scripts, the CLI and tests from inside the crate directory. Paths such as `./model`, `scripts/...` and `src/shaders/` are crate-relative. `model*/` is gitignored.
- CI is `.github/workflows/CI.yml`: the macOS/Metal job runs `bonsai-pot` from its directory, and the Linux job runs clippy, unit tests and the no_std build for `bonsai2-pot`.
- Don't add `bonsai-pot` / `bonsai_pot` names to `bonsai2-pot`; its names are all `bonsai2-pot` / `bonsai2_pot` so the two binaries don't collide in `target/`.

## Common commands (from the repo root)

```
cargo build --release -p bonsai-pot  --features bench-internals
cargo build --release -p bonsai2-pot --features bench-internals
cargo clippy -p bonsai2-pot --all-targets --features ci -- -D warnings
cargo test   -p bonsai2-pot --features ci --lib      # CPU-only unit tests
```

For GPU tests, benches and model setup, see the per-crate CLAUDE.md.
