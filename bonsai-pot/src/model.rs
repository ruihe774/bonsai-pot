//! Bonsai (Qwen3 architecture) `Q1_0` inference: model loading & GPU resource setup.
//!
//! All large weights live in 5 storage buffers organized by role; the activation
//! workspace is a single f16 buffer with named regions plus a separate buffer for
//! `Q8_0` activations (used by the `dot4I8Packed` matmul path). Norm weights and the
//! `RoPE` cos/sin table are also f16; `Q8_0` scales remain f32.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString as _};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::mem::size_of;
use core::pin::pin;
use core::str::{FromStr, from_utf8};
use core::task::{Context, Poll, Waker};
#[cfg(feature = "std")]
use std::fs::{read, read_to_string};
#[cfg(feature = "std")]
use std::path::Path;

use bytemuck::{Pod, Zeroable, cast_slice};
use once_cell::sync::OnceCell;
use wgpu::util::DeviceExt as _;
use wgpu::{Backends, InstanceDescriptor};

use crate::decode;
use crate::error::{PotError, Result};

// ----- config & manifest ----------------------------------------------------

/// Metadata for a single tensor in the model manifest.
#[derive(Debug, Clone)]
pub struct TensorEntry {
    pub dtype: &'static str,
    pub shape: Vec<u32>,
    pub buffer: &'static str,
    pub offset: u32,
    pub length: u32,
    pub d_offset: u32,
    pub qs_offset: u32,
    pub nb: u32,
}

/// Parsed `config.ini`: model dimensions, hyper-parameters, and the tensor manifest.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub n_layer: u32,
    pub n_embd: u32,
    pub n_ff: u32,
    pub n_head: u32,
    pub n_kv_head: u32,
    pub head_dim: u32,
    pub rope_freq_base: f32,
    pub rms_eps: f32,
    pub n_vocab: u32,
    pub eos_token_id: u32,
    pub add_bos: bool,
    pub rope_orig_context: u32,
    pub n_kv_groups: u32,
    pub q_dim: u32,
    pub kv_dim: u32,
    pub tied_embeddings: bool,
    /// Weight quantization format: "`Q1_0`" (binary signs, 16 B qs/block) or
    /// "`Q2_0`" (ternary 2-bit codes, 32 B qs/block). Selected at extract time;
    /// every Q tensor in `manifest` shares this format.
    pub quant_format: &'static str,
    pub manifest: BTreeMap<String, TensorEntry>,
}

impl ModelConfig {
    /// Parse a `config.ini` text blob (as produced by `scripts/extract.py`).
    ///
    /// Useful in `no_std` environments where you need to construct a
    /// [`ModelSnapshot`] without a filesystem.
    ///
    /// # Errors
    ///
    /// Return `PotError::Config` for malformed ini
    pub fn from_ini(text: &str) -> Result<Self> {
        parse_config_ini(text)
    }
}

/// Pre-loaded model weights and metadata.
///
/// Pass to [`Model::load_from_snapshot`] to initialise the engine without any
/// filesystem access. Callers can obtain a `ModelSnapshot` by:
///
/// - Calling [`ModelSnapshot::from_dir`] (requires the `std` feature) to read
///   the files from disk, or
/// - Constructing the struct directly from bytes obtained by any other means
///   (e.g. `include_bytes!`, fetched from the network, embedded in flash).
#[derive(Debug, Clone)]
pub struct ModelSnapshot {
    /// Parsed `config.ini` contents.
    pub config: ModelConfig,
    /// Contents of `weights_attn.bin`.
    pub w_attn: Vec<u8>,
    /// Contents of `weights_ffn_gate_up.bin`.
    pub w_ffn_gate_up: Vec<u8>,
    /// Contents of `weights_ffn_down.bin`.
    pub w_ffn_down: Vec<u8>,
    /// Contents of `weights_norms.bin`.
    pub w_norms: Vec<u8>,
    /// Contents of `weights_embed_lmhead.bin`.
    pub w_embed_lmhead: Vec<u8>,
    /// Contents of `vocab.bin`.
    pub vocab_bytes: Vec<u8>,
    /// Contents of `vocab_offsets.bin` (little-endian `u32` stream).
    pub vocab_offsets: Vec<u8>,
}

impl ModelSnapshot {
    /// Read the seven model files from `model_dir` and parse `config.ini`.
    ///
    /// This is the std-backed equivalent of constructing `ModelSnapshot` by hand;
    /// it reads the same files that `Model::load` would read.
    ///
    /// # Errors
    ///
    /// Return `PotError::Io` for IO error, and `PotError::Config` for invalid config
    #[cfg(feature = "std")]
    pub fn from_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let cfg_path = model_dir.join("config.ini");
        let cfg_text = read_to_string(&cfg_path).map_err(|e| PotError::Io {
            path: cfg_path,
            source: e,
        })?;
        let config = parse_config_ini(&cfg_text)?;

        let load = |fname: &str| -> Result<Vec<u8>> {
            let p = model_dir.join(fname);
            read(&p).map_err(|e| PotError::Io { path: p, source: e })
        };

        Ok(Self {
            config,
            w_attn: load("weights_attn.bin")?,
            w_ffn_gate_up: load("weights_ffn_gate_up.bin")?,
            w_ffn_down: load("weights_ffn_down.bin")?,
            w_norms: load("weights_norms.bin")?,
            w_embed_lmhead: load("weights_embed_lmhead.bin")?,
            vocab_bytes: load("vocab.bin")?,
            vocab_offsets: load("vocab_offsets.bin")?,
        })
    }
}

// ----- uniform-param structs (WGSL-side struct layouts) ---------------------

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Default, Debug)]
pub struct EmbedParams {
    pub(crate) k: u32,
    pub(crate) d_offset: u32,
    pub(crate) qs_offset: u32,
    pub(crate) output_offset: u32,
    pub(crate) sample_offset: u32,
}
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Default, Debug)]
pub struct RmsNormParams {
    pub(crate) group_size: u32,
    pub(crate) n_groups: u32,
    pub(crate) input_offset: u32,
    pub(crate) output_offset: u32,
    pub(crate) weight_offset: u32,
    pub(crate) eps: f32,
}
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Default, Debug)]
pub struct MatvecParams {
    pub(crate) k: u32,
    pub(crate) n: u32,
    pub(crate) d_offset: u32,
    pub(crate) qs_offset: u32,
    pub(crate) input_offset: u32,
    pub(crate) output_offset: u32,
    pub(crate) accumulate: u32,
    pub(crate) dispatch_x_dim: u32,
}
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Default, Debug)]
pub struct MatvecSiluParams {
    pub(crate) k: u32,
    pub(crate) n: u32,
    pub(crate) d_offset: u32,
    pub(crate) qs_offset: u32,
    pub(crate) gate_offset: u32,
    pub(crate) up_offset: u32,
    pub(crate) output_offset: u32,
    pub(crate) accumulate: u32,
    pub(crate) dispatch_x_dim: u32,
}
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Default, Debug)]
pub struct MatvecFusedParams {
    pub(crate) k: u32,
    pub(crate) n_total: u32,
    pub(crate) input_offset: u32,
    pub(crate) dispatch_x_dim: u32,
    pub(crate) d_offset_0: u32,
    pub(crate) qs_offset_0: u32,
    pub(crate) n_0: u32,
    pub(crate) output_offset_0: u32,
    pub(crate) d_offset_1: u32,
    pub(crate) qs_offset_1: u32,
    pub(crate) n_1: u32,
    pub(crate) output_offset_1: u32,
    pub(crate) d_offset_2: u32,
    pub(crate) qs_offset_2: u32,
    pub(crate) n_2: u32,
    pub(crate) output_offset_2: u32,
}
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Default, Debug)]
pub struct MatvecFusedNormedParams {
    pub(crate) k: u32,
    pub(crate) n_total: u32,
    pub(crate) input_offset: u32,
    pub(crate) dispatch_x_dim: u32,
    pub(crate) w_norm_off: u32,
    pub(crate) eps: f32,
    pub(crate) d_offset_0: u32,
    pub(crate) qs_offset_0: u32,
    pub(crate) n_0: u32,
    pub(crate) output_offset_0: u32,
    pub(crate) d_offset_1: u32,
    pub(crate) qs_offset_1: u32,
    pub(crate) n_1: u32,
    pub(crate) output_offset_1: u32,
    pub(crate) d_offset_2: u32,
    pub(crate) qs_offset_2: u32,
    pub(crate) n_2: u32,
    pub(crate) output_offset_2: u32,
}
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Default, Debug)]
pub struct MatmulParams {
    pub(crate) k: u32,
    pub(crate) n: u32,
    pub(crate) m: u32,
    pub(crate) w_d_offset: u32,
    pub(crate) w_qs_offset: u32,
    pub(crate) a_d_offset: u32,
    pub(crate) a_qs_offset: u32,
    pub(crate) out_offset: u32,
    pub(crate) accumulate: u32,
}
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Default, Debug)]
pub struct AttnPrefillTiledParams {
    pub(crate) head_dim: u32,
    pub(crate) n_head: u32,
    pub(crate) n_kv_head: u32,
    pub(crate) m_tokens: u32,
    pub(crate) pos_base: u32,
    pub(crate) kv_stride: u32,
    pub(crate) q_offset: u32,
    pub(crate) k_d_word_offset: u32,
    pub(crate) k_qs_byte_offset: u32,
    pub(crate) v_d_word_offset: u32,
    pub(crate) v_qs_byte_offset: u32,
    pub(crate) out_d_offset: u32,
    pub(crate) out_qs_offset: u32,
    pub(crate) scale: f32,
}
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Default, Debug)]
pub struct AttnSplitParams {
    pub(crate) head_dim: u32,
    pub(crate) n_head: u32,
    pub(crate) n_kv_head: u32,
    pub(crate) pos: u32,
    pub(crate) kv_stride: u32,
    pub(crate) q_offset: u32,
    pub(crate) k_d_word_offset: u32,
    pub(crate) k_qs_byte_offset: u32,
    pub(crate) v_d_word_offset: u32,
    pub(crate) v_qs_byte_offset: u32,
    pub(crate) n_chunks_active: u32,
    pub(crate) scale: f32,
}
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Default, Debug)]
pub struct AttnMergeParams {
    pub(crate) head_dim: u32,
    pub(crate) n_head: u32,
    pub(crate) out_offset: u32,
    pub(crate) n_chunks_active: u32,
}
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Default, Debug)]
pub struct TopKPartialParams {
    pub(crate) n: u32,
    pub(crate) in_offset: u32,
    pub(crate) partials_off: u32,
    pub(crate) n_per_wg: u32,
}
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Default, Debug)]
pub struct TopKMergeParams {
    pub(crate) partials_off: u32,
    pub(crate) num_partials: u32,
    pub(crate) out_offset: u32,
    pub(crate) k: u32,
}
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Default, Debug)]
pub struct RmsNormQ8Params {
    pub(crate) k: u32,
    pub(crate) input_offset: u32,
    pub(crate) weight_offset: u32,
    pub(crate) d_offset: u32,
    pub(crate) qs_offset: u32,
    pub(crate) eps: f32,
}
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Default, Debug)]
pub struct SiluMulQ8Params {
    pub(crate) k: u32,
    pub(crate) gate_offset: u32,
    pub(crate) up_offset: u32,
    pub(crate) d_offset: u32,
    pub(crate) qs_offset: u32,
}
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Default, Debug)]
pub struct QNormRopeFusedParams {
    pub(crate) q_off: u32,
    pub(crate) w_q_norm_off: u32,
    pub(crate) rope_offset: u32,
    pub(crate) pos_base: u32,
    pub(crate) q_dim: u32,
    pub(crate) eps: f32,
}
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Default, Debug)]
pub struct KvWritebackFusedParams {
    pub(crate) k_cur_off: u32,
    pub(crate) v_cur_off: u32,
    pub(crate) w_k_norm_off: u32,
    pub(crate) rope_offset: u32,
    pub(crate) dst_d_word_offset: u32,
    pub(crate) dst_qs_byte_offset: u32,
    pub(crate) pos_base: u32,
    pub(crate) kv_dim: u32,
    pub(crate) nb_per_row: u32,
    pub(crate) eps: f32,
}

/// Maximum K supported by the topk shaders (matches `K_MAX` in the WGSL).
pub const TOPK_MAX: u32 = 32;

/// Number of pass-1 workgroups in the multi-WG top-K reduction. Each WG
/// produces its own top-K_MAX over a `n / TOPK_NUM_PARTIAL_WG` slice; pass-2
/// merges the resulting `TOPK_NUM_PARTIAL_WG * K_MAX` candidates into the
/// final top-K. Picked to spread enough CUs to saturate VRAM bandwidth on
/// the LM-head logits scan without bloating the merge step.
pub const TOPK_NUM_PARTIAL_WG: u32 = 32;

// ----- activation layout ----------------------------------------------------

/// Region offsets into the single f16 activation buffer. All values are
/// **element offsets** (count of f16 elements from the buffer's start), to
/// match the shader-side `array<f16>` indexing — never bytes.
#[derive(Copy, Clone, Debug)]
pub struct ActLayout {
    pub(crate) x: u32,
    pub(crate) q: u32,
    pub(crate) k_cur: u32,
    pub(crate) v_cur: u32,
    pub(crate) attn_out: u32,
    pub(crate) gate: u32,
    pub(crate) up: u32,
    pub(crate) logits: u32,
    pub(crate) total_elems: u32,
}

impl ActLayout {
    pub fn build(cfg: &ModelConfig, m_max: u32) -> Self {
        let mut o = 0u32;
        let alloc = |n: u32, o: &mut u32| -> u32 {
            let r = *o;
            *o += n;
            r
        };
        let x = alloc(m_max * cfg.n_embd, &mut o);
        let q = alloc(m_max * cfg.q_dim, &mut o);
        let k_cur = alloc(m_max * cfg.kv_dim, &mut o);
        let v_cur = alloc(m_max * cfg.kv_dim, &mut o);
        let attn_out = alloc(m_max * cfg.q_dim, &mut o);
        let gate = alloc(m_max * cfg.n_ff, &mut o);
        let up = alloc(m_max * cfg.n_ff, &mut o);
        let logits = alloc(cfg.n_vocab, &mut o);
        Self {
            x,
            q,
            k_cur,
            v_cur,
            attn_out,
            gate,
            up,
            logits,
            total_elems: o,
        }
    }
}

// ----- SPIR-V specialization-constant patcher --------------------------------

/// Specialization-constant ID convention used by every shader in this crate.
/// `SpecId` 0 is the effective per-pipeline subgroup size (the value the shader
/// uses to compute `NUM_SUBGROUPS = ceil(WG / SUBGROUP_SIZE)`); `SpecId` 1 is
/// `MAX_CHUNKS` (= `ceil(max_seq` / `ATTN_CHUNK_SIZE`)); `SpecId` 2 is
/// `N_EMBD_V4` (= `n_embd` / 4). Shaders that don't use a given slot simply
/// ignore it.
pub const SPEC_SUBGROUP_SIZE: u32 = 0;
pub const SPEC_MAX_CHUNKS: u32 = 1;
pub const SPEC_N_EMBD_V4: u32 = 2;
/// Weight quantization format selector for Q-tensor shaders: `0` = `Q1_0`
/// (16 qs bytes/block, binary signs), `1` = `Q2_0` (32 qs bytes/block, ternary
/// 2-bit codes). Driver compilers fold the spec constant so the inactive arm
/// of each `if (QUANT_FORMAT == ...)` branch is dead code.
pub const SPEC_QUANT_FORMAT: u32 = 3;

// ----- Per-kernel workgroup shapes ------------------------------------------
//
// Single source of truth for `threads_per_threadgroup` of every kernel.
// MSL passthrough consumes this directly via
// `ShaderModuleDescriptorPassthrough::num_workgroups`; SPIR-V kernels embed
// `local_size_*` in their own `OpExecutionMode` so the value passed here is
// ignored on Vulkan — but we keep the Vulkan column populated anyway so the
// table stays a faithful record of what each backend's shader actually uses.
//
// Numbers must match the literal `WG_X`/`WG_Y`/`WG` constants in the
// `.comp` source. A few kernels (matvec_q1_0_silu, matvec_q1_0_fused_normed,
// attention_split, attention_merge) deliberately diverge between backends —
// the GLSL guards both forms with `#ifdef METAL_BACKEND`; see the in-file
// "Apple sweep" comments for the perf rationale.
const fn wg_pick(apple: (u32, u32, u32), vulkan: (u32, u32, u32)) -> (u32, u32, u32) {
    if cfg!(target_vendor = "apple") {
        apple
    } else {
        vulkan
    }
}

/// Total lane count of a workgroup (product of its three dims).
const fn wg_lanes(wg: (u32, u32, u32)) -> u32 {
    wg.0 * wg.1 * wg.2
}

//                                                          apple          vulkan
const WG_EMBED: (u32, u32, u32) = wg_pick((32, 1, 1), (32, 1, 1));
const WG_RMS_NORM: (u32, u32, u32) = wg_pick((512, 1, 1), (512, 1, 1));
const WG_MATVEC: (u32, u32, u32) = wg_pick((8, 16, 1), (8, 16, 1));
const WG_MATVEC_SILU: (u32, u32, u32) = wg_pick((8, 32, 1), (8, 16, 1));
const WG_MATVEC_FUSED_NORMED: (u32, u32, u32) = wg_pick((256, 1, 1), (128, 1, 1));
const WG_MATMUL: (u32, u32, u32) = wg_pick((256, 1, 1), (256, 1, 1));
const WG_ATTN_PREFILL_TILED: (u32, u32, u32) = wg_pick((32, 1, 1), (32, 1, 1));
const WG_ATTN_SPLIT: (u32, u32, u32) = wg_pick((32, 1, 1), (64, 1, 1));
const WG_ATTN_MERGE: (u32, u32, u32) = wg_pick((32, 1, 1), (128, 1, 1));
const WG_RMS_NORM_Q8: (u32, u32, u32) = wg_pick((64, 1, 1), (64, 1, 1));
const WG_SILU_MUL_Q8: (u32, u32, u32) = wg_pick((64, 1, 1), (64, 1, 1));
const WG_TOPK_PARTIAL: (u32, u32, u32) = wg_pick((128, 1, 1), (128, 1, 1));
const WG_TOPK_MERGE: (u32, u32, u32) = wg_pick((64, 1, 1), (64, 1, 1));
const WG_KV_WRITEBACK_FUSED: (u32, u32, u32) = wg_pick((32, 1, 1), (32, 1, 1));
const WG_Q_NORM_ROPE_FUSED: (u32, u32, u32) = wg_pick((128, 1, 1), (128, 1, 1));

/// Rows of the output matrix processed per workgroup by `matvec_q1_0_silu`.
/// Equals `WG_Y` in the corresponding shader (Apple) and total threads / 8 in
/// the GLSL counterpart. Used by `forward::dispatch_matvec_q1_0_silu` to size
/// the dispatch grid.
pub const MATVEC_SILU_ROWS_PER_WG: u32 = WG_MATVEC_SILU.1;

/// Rows of the output matrix processed per workgroup by
/// `matvec_q1_0_fused_normed`. Both backends use a flat 1D threadgroup
/// (`WG_X * WG_Y` threads) with `WG_X = 8` lanes per row, so
/// `ROWS_PER_WG = total_threads / 8`.
pub const MATVEC_FUSED_NORMED_ROWS_PER_WG: u32 = WG_MATVEC_FUSED_NORMED.0 / 8;

/// Pick a `SubgroupSize` request and the matching `SUBGROUP_SIZE` spec-const
/// value for a pipeline whose workgroup is `wg` lanes wide.
///
/// The spec-const value is the *effective lane count* used by the shader's
/// `NUM_SUBGROUPS = ceil(WG / SUBGROUP_SIZE)`:
///
/// * `wg <= sg_min` → `Varying`, spec = `wg`. The hardware subgroup is
///   wider than the WG; only the first `wg` lanes are active. Subgroup
///   reductions still see only the active lanes.
/// * `wg < sg_max`  → `Fixed(wg)`, spec = `wg`. The WG fits exactly in one
///   subgroup of the requested size (`wg % wg == 0`).
/// * `wg >= sg_max` → `Fixed(sg_max)`, spec = `sg_max`. The WG spans
///   `wg / sg_max` full subgroups. `Fixed` pins the runtime to `sg_max`,
///   matching the spec-const (`wg % sg_max == 0` since both are POT and
///   `wg >= sg_max`). `Full` is intentionally avoided here: after the
///   upstream wgpu change it sets `ALLOW_VARYING_SUBGROUP_SIZE` alongside
///   `REQUIRE_FULL_SUBGROUPS`, so the runtime is free to pick any size in
///   `[sg_min, sg_max]` rather than `sg_max`, which would disagree with the
///   spec-const baked into the shader.
///
/// Order matters — `wg <= sg_min` must be tested first, since `Fixed(wg)` is
/// invalid when `wg < sg_min` (requested size must lie in `[sg_min, sg_max]`).
#[cfg(not(target_vendor = "apple"))]
pub const fn pick_subgroup_config(wg: u32, sg_min: u32, sg_max: u32) -> (wgpu::SubgroupSize, u32) {
    if wg <= sg_min {
        (wgpu::SubgroupSize::Varying, wg)
    } else if wg < sg_max {
        (wgpu::SubgroupSize::Fixed(wg), wg)
    } else {
        (wgpu::SubgroupSize::Fixed(sg_max), sg_max)
    }
}

/// Apple Silicon stub: `SUBGROUP_SIZE_CONTROL` is not available on Metal, so
/// every pipeline runs with `Varying`. The simdgroup width is always 32 on
/// Apple Silicon, but that constant is baked directly into the MSL as a
/// literal — the spec value returned here is unused (slot 0 is dropped at
/// MSL load time).
#[cfg(target_vendor = "apple")]
pub const fn pick_subgroup_config(
    _wg: u32,
    _sg_min: u32,
    _sg_max: u32,
) -> (wgpu::SubgroupSize, u32) {
    (wgpu::SubgroupSize::Varying, 32)
}

/// Override the default of a spirv-cross-emitted `SpecConstant` in an MSL
/// source string by prepending a `#define` for the slot's `SPIRV_CROSS_CONSTANT_ID_<N>`
/// macro.
///
/// The build pipeline has already normalised the spirv-cross spec-constant
/// scaffolding to a uniform shape (see `msl_strip` in build.rs): the
/// `#ifndef SPIRV_CROSS_CONSTANT_ID_<N> / #define … / #endif` guard blocks
/// and any `[[function_constant(N)]]` declarations have been stripped /
/// rewritten so that every kernel reaches the macro through a single
/// `constant uint NAME = SPIRV_CROSS_CONSTANT_ID_<N>;` declaration. All we
/// have to do at load time is prepend the value as a `#define` — the
/// preprocessor handles the rest.
///
/// Asserts the macro actually appears in the source so a typo or missing
/// `layout(constant_id = N)` is caught at load time rather than silently
/// shipping a stale value. (Except slot 3, which is QUANT_FORMAT that is unused)
#[cfg(target_vendor = "apple")]
#[allow(clippy::panic, reason = "shaders are written by us")]
fn msl_set_function_const_u32(src: &str, slot: u32, value: u32) -> String {
    let macro_name = format!("SPIRV_CROSS_CONSTANT_ID_{slot}");
    assert!(
        src.contains(&macro_name) || slot == 3,
        "spec constant slot {slot} ({macro_name}) not referenced in MSL source"
    );
    format!("#define {macro_name} {value}u\n{src}")
}

/// Shift every SSBO `[[buffer(N)]]` on the `cs_main` signature to
/// `[[buffer(N+1)]]` so the push constant at `[[buffer(0)]]` no longer
/// collides with the SSBO at SPIR-V binding 0.
///
/// build.rs passes `--msl-decoration-binding` to spirv-cross, which makes
/// MSL buffer slots equal SPIR-V binding numbers. That gives us the SSBO
/// order we want — but spirv-cross still emits the push constant at
/// `[[buffer(0)]]`, conflicting with the SSBO at binding 0. wgpu's Metal
/// HAL allocates the push constant at slot 0 and SSBOs at slots 1.. in
/// binding order (`wgpu-hal/src/metal/device.rs::create_pipeline_layout`),
/// so a uniform `+1` shift on every SSBO lines the kernel up with the
/// host bindings. Push constants (`constant T& ...`) are left at 0;
/// storage buffers (`device T& ...`) are shifted.
///
/// Detection: if no SSBO sits at `[[buffer(0)]]`, the source is already in
/// wgpu form (e.g. hand-ported `.metal`) and we return it unchanged.
#[cfg(target_vendor = "apple")]
#[allow(
    clippy::panic,
    clippy::expect_used,
    reason = "shaders are written by us"
)]
fn msl_shift_ssbo_buffer_indices(src: &str) -> String {
    use core::fmt::Write as _;

    let head = "kernel void cs_main(";
    let head_pos = src
        .find(head)
        .expect("'kernel void cs_main(' not found in MSL source");
    let args_start = head_pos + head.len();

    let bytes = src.as_bytes();
    let mut depth: u32 = 1;
    let mut args_end = args_start;
    let mut i = args_start;
    while i < bytes.len() {
        match bytes[i] {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    args_end = i;
                    break;
                }
            }
            _ => {}
        }
        i += 1;
    }
    assert!(depth == 0, "unterminated cs_main argument list");

    let sig = &src[args_start..args_end];
    let marker = "[[buffer(";

    // Pass 1: scan for an SSBO at slot 0 — the push-constant-collision
    // signature of spirv-cross output. Hand-ported MSL has no such
    // collision and round-trips unchanged.
    let mut needs_shift = false;
    let mut p = 0;
    while let Some(rel) = sig[p..].find(marker) {
        let abs = p + rel;
        let after = abs + marker.len();
        let close_rel = sig[after..]
            .find(")]]")
            .expect("malformed [[buffer(N)]] decoration");
        let close = after + close_rel;
        let slot: u32 = sig[after..close]
            .parse()
            .expect("non-numeric buffer index in [[buffer(N)]]");
        let arg_start = sig[..abs].rfind(',').map_or(0, |c| c + 1);
        let is_ssbo = !sig[arg_start..abs].trim_start().starts_with("constant ");
        if is_ssbo && slot == 0 {
            needs_shift = true;
            break;
        }
        p = close + ")]]".len();
    }
    if !needs_shift {
        return src.to_owned();
    }

    let mut out = String::with_capacity(src.len() + 16);
    out.push_str(&src[..args_start]);

    let mut last = 0;
    let mut p = 0;
    while let Some(rel) = sig[p..].find(marker) {
        let abs = p + rel;
        let after = abs + marker.len();
        let close_rel = sig[after..]
            .find(")]]")
            .expect("malformed [[buffer(N)]] decoration");
        let close = after + close_rel;
        let slot: u32 = sig[after..close]
            .parse()
            .expect("non-numeric buffer index in [[buffer(N)]]");
        let arg_start = sig[..abs].rfind(',').map_or(0, |c| c + 1);
        let is_ssbo = !sig[arg_start..abs].trim_start().starts_with("constant ");
        let new_slot = if is_ssbo { slot + 1 } else { slot };

        out.push_str(&sig[last..abs]);
        let _ = write!(&mut out, "[[buffer({new_slot})]]");
        last = close + ")]]".len();
        p = last;
    }

    out.push_str(&sig[last..]);
    out.push_str(&src[args_end..]);
    out
}

/// Patch the default value of an `OpSpecConstant` (32-bit type) in a SPIR-V
/// module identified by its `SpecId` decoration. wgpu's passthrough shader
/// path doesn't expose `VkSpecializationInfo` to the application, so we bake
/// the runtime value into the SPIR-V's default operand instead — the result
/// is observably equivalent because the spec constant is then used unchanged
/// by every pipeline created from this module.
///
/// Encoding reference: SPIR-V spec §3, instruction stream is `(length<<16 |
/// opcode)` words; `OpDecorate` is opcode 71 with the `SpecId` decoration
/// numbered 1; `OpSpecConstant` is opcode 50 with layout `<type-id>
/// <result-id> <literal…>`.
#[cfg(not(target_vendor = "apple"))]
#[allow(clippy::panic, reason = "shaders are compiled by us")]
fn spirv_set_spec_const_u32(words: &mut [u32], spec_id: u32, value: u32) {
    const MAGIC: u32 = 0x0723_0203;
    const OP_DECORATE: u32 = 71;
    const OP_SPEC_CONSTANT: u32 = 50;
    const DEC_SPEC_ID: u32 = 1;

    assert_eq!(words[0], MAGIC, "bad SPIR-V magic");
    assert!(words.len() >= 5, "truncated SPIR-V header");

    // Pass 1: locate the result-id decorated `SpecId <spec_id>`.
    let mut target_id: Option<u32> = None;
    let mut i = 5usize;
    while i < words.len() {
        let header = words[i];
        let len = (header >> 16) as usize;
        assert!(len > 0 && i + len <= words.len(), "malformed SPIR-V");
        if (header & 0xFFFF) == OP_DECORATE
            && len >= 4
            && words[i + 2] == DEC_SPEC_ID
            && words[i + 3] == spec_id
        {
            target_id = Some(words[i + 1]);
            break;
        }
        i += len;
    }
    let target_id = target_id.unwrap_or_else(|| panic!("SpecId {spec_id} not found in SPIR-V"));

    // Pass 2: find the matching OpSpecConstant and patch its default literal.
    let mut i = 5usize;
    while i < words.len() {
        let header = words[i];
        let len = (header >> 16) as usize;
        if (header & 0xFFFF) == OP_SPEC_CONSTANT && len == 4 && words[i + 2] == target_id {
            words[i + 3] = value;
            return;
        }
        i += len;
    }
    panic!("OpSpecConstant for SpecId {spec_id} not found");
}

// ----- the model ------------------------------------------------------------

pub struct Pipelines {
    pub(crate) embed: wgpu::ComputePipeline,
    pub(crate) rms_norm: wgpu::ComputePipeline,
    pub(crate) matvec: wgpu::ComputePipeline,
    // Kept built but unused on the active code path: the matvec single-token
    // path's two callers both fold in the preceding rms_norm via
    // `matvec_fused_normed`. Retained for future no-rms-norm fused callers.
    pub(crate) matvec_silu: wgpu::ComputePipeline,
    pub(crate) matvec_fused_normed: wgpu::ComputePipeline,
    pub(crate) matmul: wgpu::ComputePipeline,
    pub(crate) attention_prefill_tiled: wgpu::ComputePipeline,
    pub(crate) attention_split: wgpu::ComputePipeline,
    pub(crate) attention_merge: wgpu::ComputePipeline,
    pub(crate) rms_norm_q8_0: wgpu::ComputePipeline,
    pub(crate) silu_mul_q8_0: wgpu::ComputePipeline,
    pub(crate) topk_partial: wgpu::ComputePipeline,
    pub(crate) topk_merge: wgpu::ComputePipeline,
    pub(crate) kv_writeback_fused: wgpu::ComputePipeline,
    pub(crate) q_norm_rope_fused: wgpu::ComputePipeline,
}

/// Truly shared / read-only GPU buffers held by [`Model`]. All weights plus
/// the precomputed `RoPE` table live here; nothing in this struct is mutated
/// during inference, so it can be aliased across any number of live
/// [`crate::Session`]s without races.
pub struct Buffers {
    pub(crate) w_attn: wgpu::Buffer,
    pub(crate) w_ffn_gu: wgpu::Buffer,
    pub(crate) w_ffn_d: wgpu::Buffer,
    pub(crate) w_norms: wgpu::Buffer,
    pub(crate) w_embed: wgpu::Buffer,
    pub(crate) rope_table: wgpu::Buffer,
}

pub struct BindGroupLayouts {
    pub(crate) embed: wgpu::BindGroupLayout,
    pub(crate) rms_norm: wgpu::BindGroupLayout,
    pub(crate) matvec: wgpu::BindGroupLayout,
    pub(crate) matvec_fused_normed: wgpu::BindGroupLayout,
    pub(crate) matmul: wgpu::BindGroupLayout,
    pub(crate) attn_prefill_tiled: wgpu::BindGroupLayout,
    pub(crate) attn_split: wgpu::BindGroupLayout,
    pub(crate) attn_merge: wgpu::BindGroupLayout,
    pub(crate) rms_norm_q8_0: wgpu::BindGroupLayout,
    pub(crate) silu_mul_q8_0: wgpu::BindGroupLayout,
    pub(crate) topk_partial: wgpu::BindGroupLayout,
    pub(crate) topk_merge: wgpu::BindGroupLayout,
    pub(crate) kv_writeback_fused: wgpu::BindGroupLayout,
    pub(crate) q_norm_rope_fused: wgpu::BindGroupLayout,
}

/// Selects which weight buffer a matvec / matmul dispatch reads from. Maps
/// directly to one of the cached bind groups in
/// [`crate::session::SessionBindGroups`].
#[derive(Copy, Clone, Debug)]
pub enum WeightSet {
    Attn,
    FfnGU,
    FfnD,
    Embed,
}

/// Per-layer `Q1_0` weight + norm-vector offsets, precomputed at load time so
/// the per-step encoder doesn't go through `format!` + `HashMap` lookup +
/// `TensorEntry::clone` for every dispatch. The per-`Q1_0`-tensor pair is
/// `(d_offset, qs_offset)`; norm offsets are pre-divided by `ACT_ELEM_BYTES`
/// so they're directly usable as the element-offset that the shader expects.
#[derive(Clone, Debug)]
pub struct LayerTensors {
    // per-layer Q1_0 weights (d_offset, qs_offset) — values are byte offsets
    // into the corresponding weight buffer (w_attn / w_ffn_gu / w_ffn_d).
    pub(crate) wq: (u32, u32),
    pub(crate) wk: (u32, u32),
    pub(crate) wv: (u32, u32),
    pub(crate) wo: (u32, u32),
    pub(crate) wg: (u32, u32), // ffn_gate
    pub(crate) wu: (u32, u32), // ffn_up
    pub(crate) wd: (u32, u32), // ffn_down
    // per-layer F16 norm element offsets (already divided by ACT_ELEM_BYTES)
    pub(crate) attn_norm_off: u32,
    pub(crate) attn_q_norm_off: u32,
    pub(crate) attn_k_norm_off: u32,
    pub(crate) ffn_norm_off: u32,
}

/// Precomputed offsets for global / output-side tensors (LM head + `output_norm`).
///
/// `token_embd_*` is the embedding-lookup tensor; `lm_head_*` is the row-major
/// projection used by the final matvec. For tied-embedding models (e.g.
/// Bonsai-4B) the two pairs are identical; for untied models (e.g. Bonsai-8B,
/// which ships a separate `output.weight`) they point to distinct rows in
/// `weights_embed_lmhead.bin`.
#[derive(Clone, Debug)]
pub struct OutputTensors {
    pub(crate) token_embd_d: u32,
    pub(crate) token_embd_qs: u32,
    pub(crate) lm_head_d: u32,
    pub(crate) lm_head_qs: u32,
    pub(crate) output_norm_off: u32,
}

/// Latched state set by the wgpu device-lost or uncaptured-error callbacks.
/// Stored in an `OnceCell` so reads are lock-free and the first writer wins.
#[derive(Debug, Clone)]
pub struct DeviceLostInfo {
    pub reason: wgpu::DeviceLostReason,
    pub message: String,
}

/// GPU-bearing handle to a loaded Bonsai model.
///
/// Holds the immutable GPU resources: weights, pipelines, bind-group layouts,
/// the precomputed `RoPE` table, and host-side bookkeeping (vocab, manifest,
/// device-lost flag). All per-conversation state — KV cache, activation
/// scratch, sample/staging/readback buffers, split-K attention partials, and
/// the bind groups bound to those buffers — lives on [`crate::Session`]
/// (via the crate-private `SessionState`), so any number of `Session`s may
/// safely run concurrently against one `Model`.
pub struct Model {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) cfg: ModelConfig,
    pub(crate) act_layout: ActLayout,
    pub(crate) m_max: u32,
    pub(crate) max_seq: u32,
    pub(crate) buffers: Buffers,
    pub(crate) pipes: Pipelines,
    pub(crate) bgls: BindGroupLayouts,
    pub(crate) layer_tensors: Vec<LayerTensors>,
    pub(crate) output_tensors: OutputTensors,
    pub(crate) vocab: Vec<String>,
    pub(crate) lost: Arc<OnceCell<DeviceLostInfo>>,
    #[cfg(all(feature = "bench-internals", not(feature = "ci")))]
    pub(crate) bench_ts_period_ns: f32,
}

pub const M_MAX: u32 = 512;
const DEFAULT_MAX_SEQ: u32 = 1024;
// Largest single staging write is the matmul-prefill token chunk: M_MAX u32 token
// ids. Tg path stages 4 B per submission (with up to 2 in flight from
// `generate_streaming` pipelining).
pub const STAGING_CHUNK: u64 = (M_MAX as u64 * 4).next_power_of_two();
/// Number of timestamp query slots allocated for bench/microbench.
/// Must be >= max marks per step (`n_layer` * 8 + 5 for the matvec path).
#[cfg(all(feature = "bench-internals", not(feature = "ci")))]
pub const BENCH_QS_SLOTS: u32 = 2048;
/// Cache positions per workgroup in the split-K attention pass. Must match
/// `CHUNK_SIZE` in `attention_split.comp`.
pub const ATTN_CHUNK_SIZE: u32 = 8;
/// System-wide GPU queue scheduling priority, passed via `VK_KHR_global_priority`.
///
/// Controls how the OS/driver kernel schedules this process's GPU work relative
/// to *all* other GPU clients (compositors, games, other ML processes). Requires
/// driver support for `VK_KHR_global_priority`; if unsupported the requested
/// value is ignored and a warning is logged.
#[cfg(not(target_vendor = "apple"))]
pub use ash::vk::QueueGlobalPriorityKHR as GlobalPriority;

#[cfg(not(target_vendor = "apple"))]
fn global_priority_fallback_to_queue_prio(priority: GlobalPriority) -> f32 {
    const LOWEST: i32 = GlobalPriority::LOW.as_raw();
    const HIGHEST: i32 = GlobalPriority::HIGH.as_raw();
    (priority.as_raw().clamp(LOWEST, HIGHEST) - LOWEST) as f32 / (HIGHEST - LOWEST) as f32
}

/// Power Preference when choosing a physical adapter.
pub use wgpu::PowerPreference;

/// Allocate-time tunables for [`Model::load_from_dir`].
///
/// These affect GPU buffer sizing (KV cache, `RoPE` table) and so cannot be
/// changed per call — pick them once at load.
#[derive(Debug, Copy, Clone)]
pub struct LoadOptions {
    /// Maximum sequence length (positions in the KV cache). Default: 1024.
    ///
    /// VRAM cost is linear: KV cache (`Q8_0` K and V combined) uses roughly
    /// `n_layer * max_seq * kv_dim * 2.25 bytes` — 32 i8 qs + one FP32 scale
    /// per 32-element block, doubled for K and V. The `RoPE` table grows as
    /// `max_seq * head_dim * 2 bytes`. The shaders themselves don't bake in
    /// a sequence-length limit, so any value up to the model's
    /// `context_length` is supported (subject to VRAM).
    pub max_seq: u32,
    /// System-wide GPU scheduling priority via `VK_KHR_global_priority`.
    ///
    /// See [`GlobalPriority`] for the available levels. Default is
    /// [`GlobalPriority::LOW`] — yields to compositors and other GPU clients,
    /// which is appropriate for background inference. If the driver does not
    /// expose `VK_KHR_global_priority` this field is silently ignored.
    #[cfg(not(target_vendor = "apple"))]
    pub priority: GlobalPriority,
    /// Power Preference when choosing a physical adapter.
    ///
    /// See [`PowerPreference`] for for the available levels. Default is
    /// [`PowerPreference::HighPerformance`].
    pub power_perference: PowerPreference,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            max_seq: DEFAULT_MAX_SEQ,
            #[cfg(not(target_vendor = "apple"))]
            priority: GlobalPriority::LOW,
            power_perference: PowerPreference::HighPerformance,
        }
    }
}

fn parse_config_ini(text: &str) -> Result<ModelConfig> {
    fn get<'a>(map: &'a BTreeMap<&str, &str>, key: &'static str) -> Result<&'a str> {
        map.get(key)
            .copied()
            .ok_or(PotError::Config("config.ini missing required field"))
    }
    fn parse_field<T: FromStr>(map: &BTreeMap<&str, &str>, key: &'static str) -> Result<T> {
        get(map, key)?
            .parse()
            .map_err(|_| PotError::Config("config.ini field has invalid value"))
    }
    fn parse_opt<T: FromStr + Default>(map: &BTreeMap<&str, &str>, key: &'static str) -> Result<T> {
        map.get(key).map_or_else(
            || Ok(T::default()),
            |v| {
                v.parse()
                    .map_err(|_| PotError::Config("config.ini field has invalid value"))
            },
        )
    }
    fn intern_dtype(s: &str) -> Result<&'static str> {
        match s {
            "Q1_0" => Ok("Q1_0"),
            "Q2_0" => Ok("Q2_0"),
            "Q8_0" => Ok("Q8_0"),
            "F16" => Ok("F16"),
            _ => Err(PotError::Config("manifest: unknown dtype")),
        }
    }
    fn intern_quant_format(s: &str) -> Result<&'static str> {
        match s {
            "Q1_0" => Ok("Q1_0"),
            "Q2_0" => Ok("Q2_0"),
            "Q8_0" => Ok("Q8_0"),
            _ => Err(PotError::Config("config: unknown quant_format")),
        }
    }
    fn intern_buffer(s: &str) -> Result<&'static str> {
        match s {
            "weights_attn.bin" => Ok("weights_attn.bin"),
            "weights_ffn_gate_up.bin" => Ok("weights_ffn_gate_up.bin"),
            "weights_ffn_down.bin" => Ok("weights_ffn_down.bin"),
            "weights_norms.bin" => Ok("weights_norms.bin"),
            "weights_embed_lmhead.bin" => Ok("weights_embed_lmhead.bin"),
            _ => Err(PotError::Config("manifest: unknown buffer")),
        }
    }
    fn build_entry(g: &BTreeMap<&str, &str>) -> Result<TensorEntry> {
        let shape = get(g, "shape")?
            .split(',')
            .map(|s| {
                s.parse::<u32>()
                    .map_err(|_| PotError::Config("config.ini shape has invalid value"))
            })
            .collect::<Result<Vec<u32>>>()?;
        Ok(TensorEntry {
            dtype: intern_dtype(get(g, "dtype")?)?,
            shape,
            buffer: intern_buffer(get(g, "buffer")?)?,
            offset: parse_field(g, "offset")?,
            length: parse_field(g, "length")?,
            d_offset: parse_opt(g, "d_offset")?,
            qs_offset: parse_opt(g, "qs_offset")?,
            nb: parse_opt(g, "nb")?,
        })
    }

    let mut globals: BTreeMap<&str, &str> = BTreeMap::new();
    let mut manifest: BTreeMap<String, TensorEntry> = BTreeMap::new();
    let mut cur_section: Option<&str> = None;
    let mut cur_fields: BTreeMap<&str, &str> = BTreeMap::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            if let Some(sec) = cur_section {
                manifest.insert(sec.to_string(), build_entry(&cur_fields)?);
                cur_fields.clear();
            }
            cur_section = Some(name);
        } else if let Some((k, v)) = line.split_once(" = ") {
            if cur_section.is_some() {
                cur_fields.insert(k, v);
            } else {
                globals.insert(k, v);
            }
        }
    }
    if let Some(sec) = cur_section {
        manifest.insert(sec.to_string(), build_entry(&cur_fields)?);
    }

    let g = &globals;
    Ok(ModelConfig {
        n_layer: parse_field(g, "n_layer")?,
        n_embd: parse_field(g, "n_embd")?,
        n_ff: parse_field(g, "n_ff")?,
        n_head: parse_field(g, "n_head")?,
        n_kv_head: parse_field(g, "n_kv_head")?,
        head_dim: parse_field(g, "head_dim")?,
        rope_freq_base: parse_field(g, "rope_freq_base")?,
        rms_eps: parse_field(g, "rms_eps")?,
        n_vocab: parse_field(g, "n_vocab")?,
        eos_token_id: parse_field(g, "eos_token_id")?,
        add_bos: get(g, "add_bos")? == "true",
        rope_orig_context: parse_field(g, "rope_orig_context")?,
        n_kv_groups: parse_field(g, "n_kv_groups")?,
        q_dim: parse_field(g, "q_dim")?,
        kv_dim: parse_field(g, "kv_dim")?,
        tied_embeddings: get(g, "tied_embeddings")? == "true",
        quant_format: intern_quant_format(get(g, "quant_format")?)?,
        manifest,
    })
}

fn validate_cfg(cfg: &ModelConfig) -> Result<()> {
    const fn pad4(n: u32) -> u32 {
        (n + 3) & !3
    }

    // --- global hyperparameter invariants ---
    if cfg.add_bos {
        return Err(PotError::Config(
            "config: add_bos=true but the runtime never prepends BOS; re-extract with a supported model",
        ));
    }
    if cfg.rope_orig_context == 0 {
        return Err(PotError::Config("config: rope_orig_context must be > 0"));
    }
    if cfg.n_kv_head == 0 || !cfg.n_head.is_multiple_of(cfg.n_kv_head) {
        return Err(PotError::Config(
            "config: n_head must be divisible by n_kv_head",
        ));
    }
    if cfg.n_kv_groups != cfg.n_head / cfg.n_kv_head {
        return Err(PotError::Config(
            "config: n_kv_groups != n_head / n_kv_head",
        ));
    }
    if cfg.q_dim != cfg.n_head * cfg.head_dim {
        return Err(PotError::Config("config: q_dim != n_head * head_dim"));
    }
    if cfg.kv_dim != cfg.n_kv_head * cfg.head_dim {
        return Err(PotError::Config("config: kv_dim != n_kv_head * head_dim"));
    }
    // matmul_q1_0_q8_0 stores two f16 outputs at a time via packHalf2x16, so
    // every output column dim it sees (q_dim, kv_dim, n_embd, n_ff) must be
    // even. The act-layout region offsets are likewise aligned to even f16
    // indices by ActLayout construction.
    if (cfg.n_embd | cfg.q_dim | cfg.kv_dim | cfg.n_ff) & 1 != 0 {
        return Err(PotError::Config(
            "config: n_embd, q_dim, kv_dim, n_ff must all be even (matmul paired f16 store)",
        ));
    }

    // --- per-tensor checks ---
    // Each spec: (name, dtype, shape as [u64;2-or-1], buffer filename)
    // For 1-D F16 tensors encode shape as &[dim] (single element).

    let n_embd = cfg.n_embd;
    let n_ff = cfg.n_ff;
    let q_dim = cfg.q_dim;
    let kv_dim = cfg.kv_dim;
    let n_vocab = cfg.n_vocab;
    let head_dim = cfg.head_dim;
    let qf = cfg.quant_format; // "Q1_0", "Q2_0", or "Q8_0"

    // Per-layer tensors (iterated over all layers).
    let layer_specs: &[(&str, &str, &[u32], &str)] = &[
        ("attn_q.weight", qf, &[n_embd, q_dim], "weights_attn.bin"),
        ("attn_k.weight", qf, &[n_embd, kv_dim], "weights_attn.bin"),
        ("attn_v.weight", qf, &[n_embd, kv_dim], "weights_attn.bin"),
        (
            "attn_output.weight",
            qf,
            &[q_dim, n_embd],
            "weights_attn.bin",
        ),
        (
            "ffn_gate.weight",
            qf,
            &[n_embd, n_ff],
            "weights_ffn_gate_up.bin",
        ),
        (
            "ffn_up.weight",
            qf,
            &[n_embd, n_ff],
            "weights_ffn_gate_up.bin",
        ),
        (
            "ffn_down.weight",
            qf,
            &[n_ff, n_embd],
            "weights_ffn_down.bin",
        ),
        ("attn_norm.weight", "F16", &[n_embd], "weights_norms.bin"),
        (
            "attn_q_norm.weight",
            "F16",
            &[head_dim],
            "weights_norms.bin",
        ),
        (
            "attn_k_norm.weight",
            "F16",
            &[head_dim],
            "weights_norms.bin",
        ),
        ("ffn_norm.weight", "F16", &[n_embd], "weights_norms.bin"),
    ];
    // Non-layer tensors.
    let global_specs: &[(&str, &str, &[u32], &str)] = &[
        ("output_norm.weight", "F16", &[n_embd], "weights_norms.bin"),
        (
            "token_embd.weight",
            qf,
            &[n_embd, n_vocab],
            "weights_embed_lmhead.bin",
        ),
    ];

    let check = |name: &str, dtype: &str, shape: &[u32], buffer: &str| -> Result<()> {
        let e = cfg.manifest.get(name).ok_or(PotError::Config(
            "manifest: expected tensor missing (re-extract the model dir)",
        ))?;
        if e.dtype != dtype {
            return Err(PotError::Config("manifest: tensor has wrong dtype"));
        }
        if e.shape.as_slice() != shape {
            return Err(PotError::Config("manifest: tensor has wrong shape"));
        }
        if e.buffer != buffer {
            return Err(PotError::Config("manifest: tensor is in wrong buffer file"));
        }
        match dtype {
            "Q1_0" | "Q2_0" | "Q8_0" => {
                // Per 128-elem super-block: (d_fp16_per_block × 2 B) + qs_bytes_per_block.
                // Q8_0 fuses 4 native 32-elem blocks per super-block → 4 d, 128 qs.
                let (d_fp16_per_block, qs_bytes_per_block): (u32, u32) = match dtype {
                    "Q1_0" => (1, 16),
                    "Q2_0" => (1, 32),
                    "Q8_0" => (4, 128),
                    _ => unreachable!(),
                };
                let n_in = shape[0];
                let n_out = if shape.len() > 1 { shape[1] } else { 1 };
                if !n_in.is_multiple_of(128) {
                    return Err(PotError::Config(
                        "manifest: Q-tensor n_in not divisible by 128",
                    ));
                }
                let nb = n_in / 128;
                if e.nb != nb {
                    return Err(PotError::Config("manifest: Q-tensor nb != n_in/128"));
                }
                let expected_qs_offset = pad4(e.d_offset + n_out * nb * d_fp16_per_block * 2);
                if e.qs_offset != expected_qs_offset {
                    return Err(PotError::Config(
                        "manifest: Q-tensor qs_offset != pad4(d_offset + n_out*nb*d_fp16_per_block*2)",
                    ));
                }
                let expected_length =
                    (e.qs_offset - e.d_offset) + pad4(n_out * nb * qs_bytes_per_block);
                if e.length != expected_length {
                    return Err(PotError::Config("manifest: Q-tensor length mismatch"));
                }
            }
            "F16" => {
                let expected_length = pad4(shape.iter().product::<u32>() * 2);
                if e.length != expected_length {
                    return Err(PotError::Config("manifest: F16 tensor length mismatch"));
                }
            }
            _ => return Err(PotError::Config("manifest: unknown dtype")),
        }
        Ok(())
    };

    for il in 0..cfg.n_layer {
        for &(tag, dtype, shape, buf) in layer_specs {
            let name = format!("blk.{il}.{tag}");
            check(&name, dtype, shape, buf)?;
        }
    }
    for &(name, dtype, shape, buf) in global_specs {
        check(name, dtype, shape, buf)?;
    }
    if !cfg.tied_embeddings {
        check(
            "output.weight",
            qf,
            &[n_embd, n_vocab],
            "weights_embed_lmhead.bin",
        )?;
    }

    Ok(())
}

impl Model {
    /// Load weights with default options. Equivalent to
    /// Convenience wrapper: reads weights from `model_dir` and calls
    /// [`Model::load_with_options`] with `LoadOptions::default()`.
    ///
    /// # Errors
    ///
    /// See [`Model::load_with_options`].
    #[cfg(feature = "std")]
    #[doc(hidden)]
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        Self::load_from_dir(model_dir, LoadOptions::default())
    }

    /// Load weights from `model_dir`, build pipelines, allocate the KV cache.
    /// Reads `config.ini`, `weights_*.bin`, `vocab.bin`, and `vocab_offsets.bin`.
    ///
    /// # Errors
    ///
    /// Returns an error if `opts.max_seq == 0`, the model files cannot be read
    /// or parsed, no suitable wgpu adapter is available, the adapter does not
    /// support the required features (`SHADER_F16`, `SUBGROUP`), the runtime
    /// subgroup size is unsupported, or the vocab files are malformed.
    #[cfg(feature = "std")]
    pub fn load_from_dir(model_dir: impl AsRef<Path>, opts: LoadOptions) -> Result<Self> {
        Self::load_from_snapshot(ModelSnapshot::from_dir(model_dir)?, opts)
    }

    /// Build pipelines and allocate the KV cache from pre-loaded `snapshot`.
    ///
    /// This is the primary entry point in `no_std` environments. Callers
    /// supply the model bytes directly (e.g. via `include_bytes!`, mmap, or
    /// a network fetch) rather than letting the engine read from the filesystem.
    ///
    /// [`ModelSnapshot::from_dir`] is a convenience constructor that reads the
    /// files from disk (requires the `std` feature).
    ///
    /// # Errors
    ///
    /// Returns an error if `opts.max_seq == 0`, the snapshot or config are
    /// malformed, no suitable wgpu adapter is available, the adapter does not
    /// support the required features (`SHADER_F16`, `SUBGROUP`), the runtime
    /// subgroup size is unsupported, or the vocab files are malformed.
    pub fn load_from_snapshot(snapshot: ModelSnapshot, opts: LoadOptions) -> Result<Self> {
        // Per-session buffer sizing constants (sample / staging / readback /
        // bench query-set) live in `SessionState::new`. Only the BGL helper
        // is hoisted here so we don't trip items-after-statements lints.
        const fn ssbo(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        }

        if opts.max_seq == 0 {
            return Err(PotError::Config("max_seq must be > 0"));
        }
        let cfg = snapshot.config;
        validate_cfg(&cfg)?;

        // ---- wgpu init ------------------------------------------------------
        let mut instance_desc = InstanceDescriptor::new_without_display_handle();
        if cfg!(target_vendor = "apple") {
            instance_desc.backends = Backends::METAL;
        } else {
            instance_desc.backends = Backends::VULKAN;
        }
        let instance = wgpu::Instance::new(instance_desc);
        let adapter = pin!(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: opts.power_perference,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));
        let adapter = match adapter.poll(&mut Context::from_waker(Waker::noop())) {
            Poll::Pending => unreachable!("native wgpu always resolves immediately"),
            Poll::Ready(Ok(adapter)) => adapter,
            Poll::Ready(Err(_)) => return Err(PotError::NoAdapter),
        };
        log::info!("adapter: {:?}", adapter.get_info());

        let features = adapter.features();
        if !features.contains(wgpu::Features::SHADER_F16) {
            return Err(PotError::FeatureUnsupported("SHADER_F16"));
        }
        if !features.contains(wgpu::Features::SHADER_I16) {
            return Err(PotError::FeatureUnsupported("SHADER_I16"));
        }
        if !features.contains(wgpu::Features::SUBGROUP) {
            return Err(PotError::FeatureUnsupported("SUBGROUP"));
        }
        #[cfg(not(target_vendor = "apple"))]
        if !features.contains(wgpu::Features::SUBGROUP_SIZE_CONTROL) {
            return Err(PotError::FeatureUnsupported("SUBGROUP_SIZE_CONTROL"));
        }
        if !features.contains(wgpu::Features::IMMEDIATES) {
            return Err(PotError::FeatureUnsupported("IMMEDIATES"));
        }
        if !features.contains(wgpu::Features::PASSTHROUGH_SHADERS) {
            return Err(PotError::FeatureUnsupported("PASSTHROUGH_SHADERS"));
        }
        #[cfg(all(feature = "bench-internals", not(feature = "ci")))]
        if !features.contains(wgpu::Features::TIMESTAMP_QUERY) {
            return Err(PotError::FeatureUnsupported("TIMESTAMP_QUERY"));
        }
        #[cfg(all(
            not(feature = "ci"),
            feature = "bench-internals",
            not(target_vendor = "apple")
        ))]
        if !features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES) {
            return Err(PotError::FeatureUnsupported(
                "TIMESTAMP_QUERY_INSIDE_PASSES",
            ));
        }

        let info = adapter.get_info();
        if cfg!(target_vendor = "apple") {
            log::info!("adapter subgroup size: 32 (Apple Silicon)");
        } else {
            log::info!(
                "adapter subgroup range: min={}, max={}",
                info.subgroup_min_size,
                info.subgroup_max_size,
            );
        }

        // Compute the largest single GPU buffer we will allocate and request
        // exactly that as the storage-binding limit (rounded up to the next
        // power of two). `device.limits()` returns exactly what we request
        // here — the adapter's higher ceiling is not available unless asked
        // for — so any buffer we ever create must fit under these.
        //
        // Storage-bound candidates: each grouped weight buffer, the RoPE
        // table, and one per-buffer KV cache (kv_k / kv_v).
        //
        // `max_buffer_size` additionally needs to cover the kv_snapshot
        // staging buffer, which holds both K and V for the live prefix in
        // one allocation (see `kv_snapshot::snapshot` / `apply`) — i.e.
        // `2 * kv_per_buf_bytes`. It's MAP_READ/COPY only, never bound as
        // STORAGE, so it doesn't enter the binding-size limit.
        let kv_nb_per_row = u64::from(cfg.kv_dim / 32);
        let kv_d_bytes: u64 = u64::from(cfg.n_layer) * u64::from(opts.max_seq) * kv_nb_per_row * 4;
        let kv_qs_bytes: u64 =
            u64::from(cfg.n_layer) * u64::from(opts.max_seq) * u64::from(cfg.kv_dim);
        let kv_per_buf_bytes = kv_d_bytes + kv_qs_bytes;
        let rope_bytes: u64 = u64::from(opts.max_seq) * u64::from(cfg.head_dim) * 2;
        let Some(largest_storage_buf) = [
            snapshot.w_attn.len() as u64,
            snapshot.w_ffn_gate_up.len() as u64,
            snapshot.w_ffn_down.len() as u64,
            snapshot.w_norms.len() as u64,
            snapshot.w_embed_lmhead.len() as u64,
            rope_bytes,
            kv_per_buf_bytes,
        ]
        .into_iter()
        .max() else {
            unreachable!();
        };
        let required_binding = largest_storage_buf.next_power_of_two();
        let kv_snapshot_staging_bytes = 2 * kv_per_buf_bytes;
        let required_buffer_size = largest_storage_buf
            .max(kv_snapshot_staging_bytes)
            .next_power_of_two();

        let limits = wgpu::Limits {
            max_buffer_size: required_buffer_size,
            max_storage_buffer_binding_size: required_binding,
            max_storage_buffers_per_shader_stage: 8,
            max_immediate_size: 128,
            ..Default::default()
        };
        log::debug!("requesting limits: {limits:?}");

        let required_features = wgpu::Features::SHADER_F16
            | wgpu::Features::SHADER_I16
            | wgpu::Features::SUBGROUP
            | {
                if cfg!(target_vendor = "apple") {
                    wgpu::Features::empty()
                } else {
                    wgpu::Features::SUBGROUP_SIZE_CONTROL
                }
            }
            | wgpu::Features::IMMEDIATES
            | wgpu::Features::PASSTHROUGH_SHADERS
            | {
                if cfg!(all(feature = "bench-internals", not(feature = "ci"))) {
                    if cfg!(target_vendor = "apple") {
                        wgpu::Features::TIMESTAMP_QUERY
                    } else {
                        wgpu::Features::TIMESTAMP_QUERY
                            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES
                    }
                } else {
                    wgpu::Features::empty()
                }
            };
        let desc = wgpu::DeviceDescriptor {
            label: None,
            required_features,
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            experimental_features: wgpu::ExperimentalFeatures::default(),
            trace: wgpu::Trace::Off,
        };
        // Open the Vulkan device via the HAL so we can select the async compute
        // queue family (compute-only, no graphics bit) and request low priority.
        // Falls back to request_device if the adapter isn't a Vulkan adapter.
        //
        // We can't use `open_with_callback` here: it hardcodes
        // `family_index = 0` (see wgpu-hal-29.0.1/src/vulkan/adapter.rs:2821)
        // and passes that to `device_from_raw`, so even if the callback swaps
        // the family in the create-info the post-create `vkGetDeviceQueue`
        // still asks for queue (0,0) which we never requested — segfault.
        // Instead we replicate `open_with_callback` ourselves and pass the
        // chosen family index through to `device_from_raw`.
        #[cfg(not(target_vendor = "apple"))]
        let hal_open = unsafe {
            use ash::khr::{global_priority, shader_maximal_reconvergence};
            use ash::vk;
            use wgpu::hal::api::Vulkan as VulkanApi;

            let Some(hal_adapter) = adapter.as_hal::<VulkanApi>() else {
                unreachable!("Vulkan adapter expected");
            };
            let pd = hal_adapter.raw_physical_device();
            let instance = hal_adapter.shared_instance().raw_instance();
            let families = instance.get_physical_device_queue_family_properties(pd);
            // Prefer a compute-only family (no GRAPHICS bit) — the async
            // compute queue on AMD.  Fall back to family 0 if none found.
            let family_idx: u32 = families
                .iter()
                .position(|p| {
                    p.queue_flags.contains(vk::QueueFlags::COMPUTE)
                        && !p.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                })
                .map_or(0, |i| i as u32);
            log::info!(
                "vk queue family {} (async_compute={})",
                family_idx,
                family_idx != 0,
            );

            let enabled_extensions = hal_adapter.required_device_extensions(desc.required_features);
            let mut enabled_phd_features =
                hal_adapter.physical_device_features(&enabled_extensions, desc.required_features);

            let supported_extensions = instance
                .enumerate_device_extension_properties(pd)
                .unwrap_or_default();
            let mut gp_supported = false;
            let mut mr_supported = false;
            for e in supported_extensions {
                let name = e.extension_name_as_c_str();
                if name == Ok(global_priority::NAME) {
                    gp_supported = true;
                } else if name == Ok(shader_maximal_reconvergence::NAME) {
                    mr_supported = true;
                } else {
                    continue;
                }
                if gp_supported && mr_supported {
                    break;
                }
            }

            let priorities = [global_priority_fallback_to_queue_prio(opts.priority)];
            let mut gp_info = vk::DeviceQueueGlobalPriorityCreateInfoKHR::default()
                .global_priority(opts.priority);
            let mut qci = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family_idx)
                .queue_priorities(&priorities);
            if gp_supported {
                qci = qci.push_next(&mut gp_info);
            } else {
                log::warn!(
                    "VK_KHR_global_priority not supported; ignoring global_priority {:?}",
                    opts.priority
                );
            }
            let queue_infos = [qci];

            let mut str_pointers: Vec<_> = enabled_extensions.iter().map(|s| s.as_ptr()).collect();
            if gp_supported {
                str_pointers.push(global_priority::NAME.as_ptr());
            }
            if mr_supported {
                str_pointers.push(shader_maximal_reconvergence::NAME.as_ptr());
            }

            let pre_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_infos)
                .enabled_extension_names(&str_pointers);
            let mut info = enabled_phd_features.add_to_device_create(pre_info);

            let mut mr = vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR::default()
                .shader_maximal_reconvergence(true);
            if mr_supported {
                info = info.push_next(&mut mr);
            } else {
                log::warn!(
                    "VK_KHR_shader_maximal_reconvergence not supported; shader behaviors may be unexpected"
                );
            }

            let raw_device = match instance.create_device(pd, &info, None) {
                Ok(d) => d,
                Err(e) => {
                    log::warn!("vkCreateDevice on family {family_idx} failed: {e:?}");
                    return Err(PotError::NoDevice);
                }
            };

            match hal_adapter.device_from_raw(
                raw_device,
                None,
                &enabled_extensions,
                desc.required_features,
                &desc.required_limits,
                &desc.memory_hints,
                family_idx,
                0,
            ) {
                Ok(d) => d,
                Err(_) => {
                    return Err(PotError::NoDevice);
                }
            }
        };

        #[cfg(target_vendor = "apple")]
        let r = {
            let future = pin!(adapter.request_device(&desc));
            match future.poll(&mut Context::from_waker(Waker::noop())) {
                Poll::Pending => unreachable!("native wgpu always resolves immediately"),
                Poll::Ready(r @ Ok(_)) => r,
                Poll::Ready(e @ Err(_)) => e,
            }
        };
        #[cfg(not(target_vendor = "apple"))]
        let r = unsafe { adapter.create_device_from_hal(hal_open, &desc) };
        let Ok((device, queue)) = r else {
            return Err(PotError::NoDevice);
        };

        // ---- wire up device-lost and uncaptured-error callbacks ------------
        // Both callbacks write into `lost` via OnceCell; the first writer wins
        // (the device-lost reason is more specific, so that path fires first).
        let lost: Arc<OnceCell<DeviceLostInfo>> = Arc::new(OnceCell::new());
        {
            let lost = Arc::clone(&lost);
            device.set_device_lost_callback(move |reason, message| {
                let _ = lost.set(DeviceLostInfo { reason, message });
            });
        }
        {
            let lost = Arc::clone(&lost);
            device.on_uncaptured_error(Arc::new(move |err: wgpu::Error| {
                let _ = lost.set(DeviceLostInfo {
                    reason: wgpu::DeviceLostReason::Unknown,
                    message: err.to_string(),
                });
            }));
        }

        // ---- validate adapter subgroup range ----------------------------------
        // With SUBGROUP_SIZE_CONTROL we pin a concrete subgroup size per
        // pipeline via PipelineCompilationOptions::subgroup_size and patch
        // each shader's SPEC_SUBGROUP_SIZE so NUM_SUBGROUPS is constant after
        // specialization. The only hard requirement on the adapter is
        // sg_min >= 8 (matvec subgroupShuffleXor(_, 1|2|4) butterfly).
        // Metal HAL hardcodes subgroup_min_size=4 / max=64; the real Apple Silicon
        // simdgroup width is 32, which the MSL bakes in as a literal. So on Apple
        // we skip the sg_min >= 8 portability gate (it would trip on the bogus
        // hardcoded 4) and treat the subgroup size as a known constant.
        let sg_min = info.subgroup_min_size;
        let sg_max = info.subgroup_max_size;
        #[cfg(not(target_vendor = "apple"))]
        if sg_min < 8 || sg_min & (sg_min - 1) != 0 {
            return Err(PotError::Config(
                "adapter subgroup_min_size must be a power-of-2 >= 8 (required by Q1_0 matvec butterfly)",
            ));
        }
        log::info!(
            "adapter={} backend={:?} subgroup range={}..={}",
            info.name,
            info.backend,
            sg_min,
            sg_max,
        );

        // ---- upload weight buffers to GPU -----------------------------------
        // create_buffer_init uses mapped_at_creation, so COPY_DST is not needed
        // for the initial upload. The weight + rope buffers are read-only after load.
        let w_storage = wgpu::BufferUsages::STORAGE;
        let make_storage = |label: &str, bytes: &[u8]| -> wgpu::Buffer {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: w_storage,
            })
        };
        let w_attn = make_storage("w_attn", &snapshot.w_attn);
        let w_ffn_gu = make_storage("w_ffn_gu", &snapshot.w_ffn_gate_up);
        let w_ffn_d = make_storage("w_ffn_d", &snapshot.w_ffn_down);
        let w_norms = make_storage("w_norms", &snapshot.w_norms);
        let w_embed = make_storage("w_embed", &snapshot.w_embed_lmhead);

        // ---- build RoPE table (f32 host-side, then downcast to f16) --------
        let rope_table_f32 = build_rope_table(&cfg, opts.max_seq);
        let rope_table_f16: Vec<half::f16> = rope_table_f32
            .iter()
            .map(|&v| half::f16::from_f32(v))
            .collect();
        let rope_buf = make_storage("rope_table", cast_slice(&rope_table_f16));

        // ---- KV cache shape sanity (per-session buffers themselves are
        //      allocated by `SessionState::new`; we still validate the
        //      load-time invariants and the adapter buffer-size budget here so
        //      a too-large `max_seq` errors out before the first session is
        //      created).
        if !cfg.kv_dim.is_multiple_of(32) {
            return Err(PotError::Config(
                "kv_dim must be a multiple of 32 (Q8_0 block size)",
            ));
        }
        if cfg.head_dim != 128 {
            return Err(PotError::Config(
                "kv_writeback_fused.comp / q_norm_rope_fused.comp assume head_dim == 128",
            ));
        }
        let nb_per_row = u64::from(cfg.kv_dim / 32);
        let kv_d_total: u64 = u64::from(cfg.n_layer) * u64::from(opts.max_seq) * nb_per_row * 4;
        let kv_qs_total: u64 =
            u64::from(cfg.n_layer) * u64::from(opts.max_seq) * u64::from(cfg.kv_dim);
        let kv_total: u64 = kv_d_total + kv_qs_total;
        {
            let dl: wgpu::Limits = device.limits();
            let max_buf = dl.max_buffer_size;
            let max_bind = dl.max_storage_buffer_binding_size;
            if kv_total > max_buf || kv_total > max_bind {
                return Err(PotError::Config(
                    "KV cache exceeds adapter buffer/binding limit; \
                     reduce --max-seq or use a GPU with larger storage buffers",
                ));
            }
        }

        let act_layout = ActLayout::build(&cfg, M_MAX);

        let buffers = Buffers {
            w_attn,
            w_ffn_gu,
            w_ffn_d,
            w_norms,
            w_embed,
            rope_table: rope_buf,
        };

        // ---- shaders & pipelines -------------------------------------------
        // SUBGROUP_SIZE is the per-pipeline effective subgroup size (paired
        // with PipelineCompilationOptions::subgroup_size via
        // `pick_subgroup_config`); MAX_CHUNKS is the attention split-K
        // partial-buffer / weights_sh dimension; N_EMBD_V4 is the per-model
        // x_sh dimension in matvec_q1_0_fused_normed (sized exactly so the
        // LDS allocation matches one model's row, maximizing WG-occupancy on
        // AMD RDNA, which allocates LDS in 1 KiB granularity per WG).
        let max_chunks = opts.max_seq.div_ceil(ATTN_CHUNK_SIZE);

        // Per-shader subgroup choice + spec-const value, computed from each
        // shader's WG width via `pick_subgroup_config`.
        let (sg_choice_rms, sg_spec_rms) =
            pick_subgroup_config(wg_lanes(WG_RMS_NORM), sg_min, sg_max);
        let (sg_choice_attn_prefill_tiled, sg_spec_attn_prefill_tiled) =
            pick_subgroup_config(wg_lanes(WG_ATTN_PREFILL_TILED), sg_min, sg_max);
        let (sg_choice_attn_split, sg_spec_attn_split) =
            pick_subgroup_config(wg_lanes(WG_ATTN_SPLIT), sg_min, sg_max);
        let (sg_choice_attn_merge, sg_spec_attn_merge) =
            pick_subgroup_config(wg_lanes(WG_ATTN_MERGE), sg_min, sg_max);
        let (sg_choice_matvec_fused_normed, sg_spec_matvec_fused_normed) =
            pick_subgroup_config(wg_lanes(WG_MATVEC_FUSED_NORMED), sg_min, sg_max);
        let (sg_choice_kv_writeback_fused, sg_spec_kv_writeback_fused) =
            pick_subgroup_config(wg_lanes(WG_KV_WRITEBACK_FUSED), sg_min, sg_max);
        let (sg_choice_q_norm_rope_fused, sg_spec_q_norm_rope_fused) =
            pick_subgroup_config(wg_lanes(WG_Q_NORM_ROPE_FUSED), sg_min, sg_max);
        let (sg_choice_rms_q8, sg_spec_rms_q8) =
            pick_subgroup_config(wg_lanes(WG_RMS_NORM_Q8), sg_min, sg_max);
        let (sg_choice_silu_q8, _sg_spec_silu_q8) =
            pick_subgroup_config(wg_lanes(WG_SILU_MUL_Q8), sg_min, sg_max);

        // Pre-flight: check the attention_merge LDS budget before shader compile.
        // weights_sh needs MAX_CHUNKS f32 slots; sg_partial needs NUM_SUBGROUPS f32 slots.
        let merge_num_subgroups = wg_lanes(WG_ATTN_MERGE).div_ceil(sg_spec_attn_merge);
        let merge_lds_bytes = 4 * u64::from(max_chunks) + 4 * u64::from(merge_num_subgroups);
        if merge_lds_bytes > u64::from(device.limits().max_compute_workgroup_storage_size) {
            return Err(PotError::Config(
                "max_seq exceeds attention_merge LDS budget; reduce --max-seq",
            ));
        }

        // SAFETY: every kernel ships as one of two forms produced by
        // `build.rs` from a single GLSL source under `src/shaders/`:
        //
        //  * Vulkan: precompiled SPIR-V (`OUT_DIR/{name}.comp.spv`).
        //  * Apple: MSL produced by spirv-cross from the same GLSL compiled
        //    with `-DMETAL_BACKEND=1` (`OUT_DIR/{name}.comp.msl`). The lone
        //    `matmul_q1_0_q8_0` kernel is hand-ported (Apple's
        //    `simdgroup_matrix<half,8,8>` MMA has no GLSL surface) and the
        //    build copies its `.metal` sibling into the same OUT_DIR slot.
        //
        // The three values that the WGSL versions text-substituted
        // (`SUBGROUP_SIZE`, `MAX_CHUNKS`, `K_V4`/`N_EMBD_V4`) are Vulkan
        // specialization constants at SpecId 0, 1, 2 in the GLSL. On the
        // Vulkan side we patch their `OpSpecConstant` defaults in the
        // SPIR-V at load time. On the Apple side, build.rs's `msl_strip`
        // pass has already normalised each SpecId to a single
        // `constant uint NAME = SPIRV_CROSS_CONSTANT_ID_<n>;` reference —
        // see `msl_set_function_const_u32`, which prepends a `#define`
        // for the macro at load time. Either way the result is fed
        // through `create_shader_module_passthrough`, bypassing naga.
        #[cfg(not(target_vendor = "apple"))]
        macro_rules! load_shader {
            ($name:expr, $spec_consts:expr, $wg:expr) => {{
                const SPV: &[u8] =
                    include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".comp.spv"));
                let _ = $wg;
                let mut words: Vec<u32> = bytemuck::pod_collect_to_vec(SPV);
                for &(id, value) in $spec_consts as &[(u32, u32)] {
                    spirv_set_spec_const_u32(&mut words, id, value);
                }
                unsafe {
                    device.create_shader_module_passthrough(
                        wgpu::ShaderModuleDescriptorPassthrough {
                            label: Some($name),
                            spirv: Some(alloc::borrow::Cow::Owned(words)),
                            ..Default::default()
                        },
                    )
                }
            }};
        }
        #[cfg(target_vendor = "apple")]
        macro_rules! load_shader {
            ($name:expr, $spec_consts:expr, $wg:expr) => {{
                const SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/", $name, ".comp.msl"));
                let mut s: String = SRC.to_owned();
                for &(id, value) in $spec_consts as &[(u32, u32)] {
                    // Slot 0 (`SUBGROUP_SIZE`) is always 32 on Apple Silicon
                    // (`pick_subgroup_config` enforces this).
                    if id == SPEC_SUBGROUP_SIZE {
                        debug_assert_eq!(value, 32, "Apple SUBGROUP_SIZE must be 32");
                    }
                    s = msl_set_function_const_u32(&s, id, value);
                }
                s = msl_shift_ssbo_buffer_indices(&s);
                unsafe {
                    device.create_shader_module_passthrough(
                        wgpu::ShaderModuleDescriptorPassthrough {
                            label: Some($name),
                            msl: Some(alloc::borrow::Cow::Owned(s)),
                            num_workgroups: $wg,
                            ..Default::default()
                        },
                    )
                }
            }};
        }
        let no_spec: &[(u32, u32)] = &[];
        // Spec-constant values: Q1_0=0, Q2_0=1, Q8_0=3. The encoding lets the
        // existing `16u << QUANT_FORMAT` / `4u << QUANT_FORMAT` formulas in
        // lib/q1_0_load.glsl yield 128 qs bytes / 32 u32 words per 128-elem
        // super-block for Q8_0 without any per-format branch on the size.
        let quant_fmt_id: u32 = match cfg.quant_format {
            "Q1_0" => 0,
            "Q2_0" => {
                if cfg!(target_vendor = "apple") {
                    return Err(PotError::Config("Q2_0 is not supported in Metal backend"));
                }
                1
            }
            "Q8_0" => {
                if cfg!(target_vendor = "apple") {
                    return Err(PotError::Config("Q8_0 is not supported in Metal backend"));
                }
                3
            }
            _ => return Err(PotError::Config("unknown quant_format")),
        };
        let qf_spec: &[(u32, u32)] = &[(SPEC_QUANT_FORMAT, quant_fmt_id)];

        let sh_embed = load_shader!("embed", qf_spec, WG_EMBED);
        let sh_rms = load_shader!(
            "rms_norm",
            &[(SPEC_SUBGROUP_SIZE, sg_spec_rms)],
            WG_RMS_NORM
        );
        let sh_matvec = load_shader!("matvec_q1_0", qf_spec, WG_MATVEC);
        let sh_matvec_silu = load_shader!("matvec_q1_0_silu", qf_spec, WG_MATVEC_SILU);
        let sh_matvec_fused_normed = load_shader!(
            "matvec_q1_0_fused_normed",
            &[
                (SPEC_SUBGROUP_SIZE, sg_spec_matvec_fused_normed),
                (SPEC_N_EMBD_V4, cfg.n_embd / 4),
                (SPEC_QUANT_FORMAT, quant_fmt_id),
            ],
            WG_MATVEC_FUSED_NORMED
        );
        let sh_matmul = load_shader!("matmul_q1_0_q8_0", qf_spec, WG_MATMUL);
        let sh_attn_prefill_tiled = load_shader!(
            "attention_prefill_tiled",
            &[(SPEC_SUBGROUP_SIZE, sg_spec_attn_prefill_tiled)],
            WG_ATTN_PREFILL_TILED
        );
        let sh_attn_split = load_shader!(
            "attention_split",
            &[(SPEC_SUBGROUP_SIZE, sg_spec_attn_split)],
            WG_ATTN_SPLIT
        );
        let sh_attn_merge = load_shader!(
            "attention_merge",
            &[
                (SPEC_SUBGROUP_SIZE, sg_spec_attn_merge),
                (SPEC_MAX_CHUNKS, max_chunks),
            ],
            WG_ATTN_MERGE
        );
        let sh_rms_q8 = load_shader!(
            "rms_norm_q8_0",
            &[(SPEC_SUBGROUP_SIZE, sg_spec_rms_q8)],
            WG_RMS_NORM_Q8
        );
        let sh_silu_q8 = load_shader!("silu_mul_q8_0", no_spec, WG_SILU_MUL_Q8);
        let sh_topk_partial = load_shader!("topk_partial", no_spec, WG_TOPK_PARTIAL);
        let sh_topk_merge = load_shader!("topk_merge", no_spec, WG_TOPK_MERGE);
        let sh_kv_writeback_fused = load_shader!(
            "kv_writeback_fused",
            &[(SPEC_SUBGROUP_SIZE, sg_spec_kv_writeback_fused)],
            WG_KV_WRITEBACK_FUSED
        );
        let sh_q_norm_rope_fused = load_shader!(
            "q_norm_rope_fused",
            &[(SPEC_SUBGROUP_SIZE, sg_spec_q_norm_rope_fused)],
            WG_Q_NORM_ROPE_FUSED
        );

        let make_bgl =
            |label: &'static str, n_storage: u32, rw_mask: u32| -> wgpu::BindGroupLayout {
                let mut entries = Vec::new();
                for i in 0..n_storage {
                    let read_only = (rw_mask >> i) & 1 == 0;
                    entries.push(ssbo(i, read_only));
                }
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(label),
                    entries: &entries,
                })
            };
        let bgls = BindGroupLayouts {
            embed: make_bgl("embed_bgl", 3, 0b010), // weights ro, act rw, sample ro
            rms_norm: make_bgl("rms_norm_bgl", 2, 0b01), // act rw, w ro
            matvec: make_bgl("matvec_bgl", 2, 0b10), // weights ro, act rw
            matvec_fused_normed: make_bgl("matvec_fused_normed_bgl", 3, 0b010), // weights ro, act rw, w_norms ro
            matmul: make_bgl("matmul_bgl", 3, 0b100), // weights ro, acts ro, y rw
            attn_prefill_tiled: make_bgl("attn_prefill_tiled_bgl", 4, 0b1000), // act ro, k ro, v ro, act_q8 rw
            attn_split: make_bgl("attn_split_bgl", 4, 0b1000), // act ro, k ro, v ro, partials rw
            attn_merge: make_bgl("attn_merge_bgl", 2, 0b01),   // act rw, partials ro
            rms_norm_q8_0: make_bgl("rms_norm_q8_0_bgl", 3, 0b100), // act ro, w ro, outbuf rw
            silu_mul_q8_0: make_bgl("silu_mul_q8_0_bgl", 2, 0b10), // act ro, outbuf rw
            topk_partial: make_bgl("topk_partial_bgl", 2, 0b10), // logits ro, result rw
            topk_merge: make_bgl("topk_merge_bgl", 1, 0b1),    // result rw
            kv_writeback_fused: make_bgl("kv_writeback_fused_bgl", 5, 0b11000), // act ro, w_norms ro, rope_cs ro, kv_k rw, kv_v rw
            q_norm_rope_fused: make_bgl("q_norm_rope_fused_bgl", 3, 0b001), // act rw, w_norms ro, rope_cs ro
        };

        let mk_pipe = |layout: &wgpu::BindGroupLayout,
                       sh: &wgpu::ShaderModule,
                       label: &str,
                       imm_size: u32,
                       subgroup_size: wgpu::SubgroupSize|
         -> wgpu::ComputePipeline {
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{label}_pl")),
                bind_group_layouts: &[Some(layout)],
                immediate_size: imm_size,
            });
            // Every shader's workgroup memory is fully written before being
            // read (cooperative tile loads + barrier, or per-thread slot
            // writes), so we don't rely on the implicit zero-init that wgpu
            // would otherwise inject as a per-dispatch prelude.
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module: sh,
                #[cfg(not(target_vendor = "apple"))]
                entry_point: Some("main"),
                #[cfg(target_vendor = "apple")]
                entry_point: Some("cs_main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    zero_initialize_workgroup_memory: false,
                    subgroup_size,
                    ..Default::default()
                },
                cache: None,
            })
        };
        let pipes = Pipelines {
            embed: mk_pipe(
                &bgls.embed,
                &sh_embed,
                "embed",
                size_of::<EmbedParams>() as u32,
                wgpu::SubgroupSize::Varying,
            ),
            rms_norm: mk_pipe(
                &bgls.rms_norm,
                &sh_rms,
                "rms_norm",
                size_of::<RmsNormParams>() as u32,
                sg_choice_rms,
            ),
            matvec: mk_pipe(
                &bgls.matvec,
                &sh_matvec,
                "matvec",
                size_of::<MatvecParams>() as u32,
                wgpu::SubgroupSize::Varying,
            ),
            matvec_silu: mk_pipe(
                &bgls.matvec,
                &sh_matvec_silu,
                "matvec_silu",
                size_of::<MatvecSiluParams>() as u32,
                wgpu::SubgroupSize::Varying,
            ),
            matvec_fused_normed: mk_pipe(
                &bgls.matvec_fused_normed,
                &sh_matvec_fused_normed,
                "matvec_fused_normed",
                size_of::<MatvecFusedNormedParams>() as u32,
                sg_choice_matvec_fused_normed,
            ),
            matmul: mk_pipe(
                &bgls.matmul,
                &sh_matmul,
                "matmul",
                size_of::<MatmulParams>() as u32,
                wgpu::SubgroupSize::Varying,
            ),
            attention_prefill_tiled: mk_pipe(
                &bgls.attn_prefill_tiled,
                &sh_attn_prefill_tiled,
                "attention_prefill_tiled",
                size_of::<AttnPrefillTiledParams>() as u32,
                sg_choice_attn_prefill_tiled,
            ),
            attention_split: mk_pipe(
                &bgls.attn_split,
                &sh_attn_split,
                "attention_split",
                size_of::<AttnSplitParams>() as u32,
                sg_choice_attn_split,
            ),
            attention_merge: mk_pipe(
                &bgls.attn_merge,
                &sh_attn_merge,
                "attention_merge",
                size_of::<AttnMergeParams>() as u32,
                sg_choice_attn_merge,
            ),
            rms_norm_q8_0: mk_pipe(
                &bgls.rms_norm_q8_0,
                &sh_rms_q8,
                "rms_norm_q8_0",
                size_of::<RmsNormQ8Params>() as u32,
                sg_choice_rms_q8,
            ),
            silu_mul_q8_0: mk_pipe(
                &bgls.silu_mul_q8_0,
                &sh_silu_q8,
                "silu_mul_q8_0",
                size_of::<SiluMulQ8Params>() as u32,
                sg_choice_silu_q8,
            ),
            topk_partial: mk_pipe(
                &bgls.topk_partial,
                &sh_topk_partial,
                "topk_partial",
                size_of::<TopKPartialParams>() as u32,
                wgpu::SubgroupSize::Varying,
            ),
            topk_merge: mk_pipe(
                &bgls.topk_merge,
                &sh_topk_merge,
                "topk_merge",
                size_of::<TopKMergeParams>() as u32,
                wgpu::SubgroupSize::Varying,
            ),
            kv_writeback_fused: mk_pipe(
                &bgls.kv_writeback_fused,
                &sh_kv_writeback_fused,
                "kv_writeback_fused",
                size_of::<KvWritebackFusedParams>() as u32,
                sg_choice_kv_writeback_fused,
            ),
            q_norm_rope_fused: mk_pipe(
                &bgls.q_norm_rope_fused,
                &sh_q_norm_rope_fused,
                "q_norm_rope_fused",
                size_of::<QNormRopeFusedParams>() as u32,
                sg_choice_q_norm_rope_fused,
            ),
        };

        // ---- vocab ----------------------------------------------------------
        let offs: &[u32] = cast_slice(&snapshot.vocab_offsets);
        if offs.len() as u32 != cfg.n_vocab + 1 {
            return Err(PotError::Vocab("offsets length doesn't match n_vocab + 1"));
        }
        let mut vocab = Vec::with_capacity(cfg.n_vocab as usize);
        for i in 0..cfg.n_vocab as usize {
            let s = from_utf8(&snapshot.vocab_bytes[offs[i] as usize..offs[i + 1] as usize])
                .unwrap_or("?")
                .to_string();
            vocab.push(s);
        }

        // ---- precompute per-layer tensor offsets ---------------------------
        let layer_tensors: Vec<LayerTensors> = (0..cfg.n_layer)
            .map(|il| {
                let q = tensor(&cfg, &format!("blk.{il}.attn_q.weight"));
                let k = tensor(&cfg, &format!("blk.{il}.attn_k.weight"));
                let v = tensor(&cfg, &format!("blk.{il}.attn_v.weight"));
                let o = tensor(&cfg, &format!("blk.{il}.attn_output.weight"));
                let g = tensor(&cfg, &format!("blk.{il}.ffn_gate.weight"));
                let u = tensor(&cfg, &format!("blk.{il}.ffn_up.weight"));
                let d = tensor(&cfg, &format!("blk.{il}.ffn_down.weight"));
                let an = tensor(&cfg, &format!("blk.{il}.attn_norm.weight"));
                let qn = tensor(&cfg, &format!("blk.{il}.attn_q_norm.weight"));
                let kn = tensor(&cfg, &format!("blk.{il}.attn_k_norm.weight"));
                let fn_ = tensor(&cfg, &format!("blk.{il}.ffn_norm.weight"));
                let act_elem_bytes = size_of::<half::f16>() as u32;
                LayerTensors {
                    wq: (q.d_offset as u32, q.qs_offset as u32),
                    wk: (k.d_offset as u32, k.qs_offset as u32),
                    wv: (v.d_offset as u32, v.qs_offset as u32),
                    wo: (o.d_offset as u32, o.qs_offset as u32),
                    wg: (g.d_offset as u32, g.qs_offset as u32),
                    wu: (u.d_offset as u32, u.qs_offset as u32),
                    wd: (d.d_offset as u32, d.qs_offset as u32),
                    attn_norm_off: (an.offset / act_elem_bytes) as u32,
                    attn_q_norm_off: (qn.offset / act_elem_bytes) as u32,
                    attn_k_norm_off: (kn.offset / act_elem_bytes) as u32,
                    ffn_norm_off: (fn_.offset / act_elem_bytes) as u32,
                }
            })
            .collect();

        let output_tensors = {
            let te = tensor(&cfg, "token_embd.weight");
            let on = tensor(&cfg, "output_norm.weight");
            let lm = if cfg.tied_embeddings {
                te
            } else {
                tensor(&cfg, "output.weight")
            };
            let act_elem_bytes = size_of::<half::f16>() as u32;
            OutputTensors {
                token_embd_d: te.d_offset,
                token_embd_qs: te.qs_offset,
                lm_head_d: lm.d_offset,
                lm_head_qs: lm.qs_offset,
                output_norm_off: (on.offset / act_elem_bytes) as u32,
            }
        };

        // Bind groups now live on `SessionState` (rebuilt per session) — see
        // `SessionState::new`. `bgls` is stored on `Model` so each session can
        // build its own bind groups without rebuilding the layouts.

        #[cfg(all(feature = "bench-internals", not(feature = "ci")))]
        let bench_ts_period_ns = queue.get_timestamp_period();

        Ok(Self {
            device,
            queue,
            cfg,
            act_layout,
            m_max: M_MAX,
            max_seq: opts.max_seq,
            buffers,
            pipes,
            bgls,
            layer_tensors,
            output_tensors,
            vocab,
            lost,
            #[cfg(all(feature = "bench-internals", not(feature = "ci")))]
            bench_ts_period_ns,
        })
    }

    /// Maximum sequence length supported by the allocated KV cache.
    #[must_use]
    pub const fn max_seq_len(&self) -> u32 {
        self.max_seq
    }

    /// Maximum batch size supported by a single matmul prefill dispatch.
    #[must_use]
    pub const fn max_prefill_tokens(&self) -> u32 {
        self.m_max
    }

    /// Open a fresh inference session.
    ///
    /// Allocates this session's per-conversation GPU buffers (KV cache,
    /// activation scratch, sample/staging/readback, split-K attention
    /// partials) and builds the bind groups bound to them. Cost scales with
    /// `LoadOptions::max_seq`: at the default `max_seq=1024`, roughly 170 MB
    /// for the KV cache plus a few tens of MB for activation/sample buffers.
    /// Subsequent inference calls on this session do no allocation.
    ///
    /// Multiple sessions on one `Model` are independent — they own disjoint
    /// GPU resources and may safely run concurrently.
    #[must_use]
    pub fn new_session(&self) -> crate::Session<'_> {
        crate::Session::new(self)
    }

    /// Decode a single token id to its raw bytes (after inverting the GPT-2
    /// byte-level vocab encoding). Returns the UTF-8 encoding of the literal
    /// vocab string for special tokens like `<|im_start|>`.
    #[must_use]
    pub fn decode_token(&self, id: u32) -> Vec<u8> {
        let s = self.vocab.get(id as usize).map_or("", String::as_str);
        decode::decode_token_bytes(s)
    }

    /// Decode a sequence of token ids into a string (lossy UTF-8).
    #[must_use]
    pub fn decode_tokens(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend(self.decode_token(id));
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// The raw vocab string for a token id (still in GPT-2 byte-encoded form).
    #[must_use]
    pub fn vocab_token(&self, id: u32) -> Option<&str> {
        self.vocab.get(id as usize).map(String::as_str)
    }

    /// Reverse lookup: find the token id whose raw vocab string matches
    /// `token` exactly. Linear scan over the vocab — only intended for
    /// occasional startup-time lookups (e.g. resolving `<|im_end|>` in a chat
    /// REPL), not per-token decode work.
    #[must_use]
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.vocab.iter().position(|s| s == token).map(|i| i as u32)
    }

    /// Returns `true` if the underlying wgpu device has been lost.
    ///
    /// Once `true`, this `Model` (and any [`crate::Session`] borrowed from it)
    /// is permanently unusable. Drop both, then call [`Model::load`] again to
    /// recover. A [`crate::KvSnapshot`] captured before loss can be used to
    /// warm-restart the new session via [`crate::Session::restore`].
    #[must_use]
    pub fn is_device_lost(&self) -> bool {
        self.lost.get().is_some()
    }

    /// Returns `Err(PotError::DeviceLost)` if the device has been lost, `Ok`
    /// otherwise. Used as a fail-fast guard at the start of every GPU-touching
    /// Session method.
    pub(crate) fn check_device(&self) -> Result<()> {
        if let Some(d) = self.lost.get() {
            return Err(PotError::DeviceLost {
                reason: d.reason,
                message: d.message.clone(),
            });
        }
        Ok(())
    }

    /// Calls `Device::destroy()` to simulate device loss. Intended only for
    /// tests; not part of the stable public API.
    #[doc(hidden)]
    pub fn __destroy_device_for_test(&self) {
        self.device.destroy();
        // Poll to flush the device-lost callback into the OnceCell. The callback
        // is queued by destroy() but only fires during poll().
        let _ = self.device.poll(wgpu::PollType::Poll);
    }
}

/// Precompute cos/sin table for NEOX rope: per position p (`0..max_seq`),
/// per j (`0..head_dim/2`), interleaved (cos, sin) pairs => `head_dim` floats per pos.
fn build_rope_table(cfg: &ModelConfig, max_seq: u32) -> Vec<f32> {
    let half = (cfg.head_dim / 2) as usize;
    let mut out = vec![0f32; max_seq as usize * cfg.head_dim as usize];
    for p in 0..max_seq as usize {
        for j in 0..half {
            let theta =
                f64::from(cfg.rope_freq_base).powf(-2.0 * j as f64 / f64::from(cfg.head_dim));
            let angle = p as f64 * theta;
            out[p * cfg.head_dim as usize + 2 * j] = angle.cos() as f32;
            out[p * cfg.head_dim as usize + 2 * j + 1] = angle.sin() as f32;
        }
    }
    out
}

// ----- public(crate) helpers used by forward.rs -----------------------------

#[allow(
    clippy::unwrap_used,
    reason = "manifest is fully validated at load; missing tensor is a programmer error"
)]
pub fn tensor<'a>(cfg: &'a ModelConfig, name: &str) -> &'a TensorEntry {
    cfg.manifest.get(name).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bonsai4b_cfg() -> ModelConfig {
        ModelConfig {
            n_layer: 36,
            n_embd: 2560,
            n_ff: 9728,
            n_head: 32,
            n_kv_head: 8,
            head_dim: 128,
            rope_freq_base: 1_000_000.0,
            rms_eps: 1e-6,
            n_vocab: 151_936,
            eos_token_id: 151_645,
            add_bos: false,
            rope_orig_context: 4_096,
            n_kv_groups: 4,
            q_dim: 4_096,
            kv_dim: 1_024,
            tied_embeddings: true,
            quant_format: "Q1_0",
            manifest: BTreeMap::new(),
        }
    }

    fn rope_test_cfg() -> ModelConfig {
        ModelConfig {
            n_layer: 1,
            n_embd: 8,
            n_ff: 8,
            n_head: 1,
            n_kv_head: 1,
            head_dim: 8,
            rope_freq_base: 10_000.0,
            rms_eps: 1e-6,
            n_vocab: 10,
            eos_token_id: 1,
            add_bos: false,
            rope_orig_context: 4,
            n_kv_groups: 1,
            q_dim: 8,
            kv_dim: 8,
            tied_embeddings: false,
            quant_format: "Q1_0",
            manifest: BTreeMap::new(),
        }
    }

    #[test]
    fn topk_max_is_thirty_two() {
        assert_eq!(TOPK_MAX, 32);
    }

    #[cfg(not(target_vendor = "apple"))]
    #[test]
    fn pick_subgroup_config_branches() {
        use wgpu::SubgroupSize;
        // wave32 hardware (NVIDIA, Apple, Intel Gen12+, AMD wave32)
        assert_eq!(
            super::pick_subgroup_config(32, 32, 32),
            (SubgroupSize::Varying, 32)
        );
        assert_eq!(
            super::pick_subgroup_config(128, 32, 32),
            (SubgroupSize::Fixed(32), 32)
        );
        assert_eq!(
            super::pick_subgroup_config(512, 32, 32),
            (SubgroupSize::Fixed(32), 32)
        );
        // wave64 hardware (default RDNA): WG=32 → small-WG branch (Varying, spec=WG).
        assert_eq!(
            super::pick_subgroup_config(32, 64, 64),
            (SubgroupSize::Varying, 32)
        );
        assert_eq!(
            super::pick_subgroup_config(64, 64, 64),
            (SubgroupSize::Varying, 64)
        );
        assert_eq!(
            super::pick_subgroup_config(128, 64, 64),
            (SubgroupSize::Fixed(64), 64)
        );
        // mixed wave32-and-64 hardware (e.g. Intel Gen11): RDNA with cswave32.
        assert_eq!(
            super::pick_subgroup_config(32, 32, 64),
            (SubgroupSize::Varying, 32)
        );
        assert_eq!(
            super::pick_subgroup_config(64, 32, 64),
            (SubgroupSize::Fixed(64), 64)
        );
        assert_eq!(
            super::pick_subgroup_config(128, 32, 64),
            (SubgroupSize::Fixed(64), 64)
        );
    }

    #[test]
    fn act_layout_offsets_monotonic() {
        let cfg = bonsai4b_cfg();
        let m = ActLayout::build(&cfg, 512);
        // Each region starts where the previous one ended.
        assert_eq!(m.x, 0);
        assert_eq!(m.q, 512 * cfg.n_embd);
        assert_eq!(m.k_cur, m.q + 512 * cfg.q_dim);
        assert_eq!(m.v_cur, m.k_cur + 512 * cfg.kv_dim);
        assert_eq!(m.attn_out, m.v_cur + 512 * cfg.kv_dim);
        assert_eq!(m.gate, m.attn_out + 512 * cfg.q_dim);
        assert_eq!(m.up, m.gate + 512 * cfg.n_ff);
        assert_eq!(m.logits, m.up + 512 * cfg.n_ff);
        assert_eq!(m.total_elems, m.logits + cfg.n_vocab);
        // Regions are strictly ordered.
        assert!(
            m.x < m.q
                && m.q < m.k_cur
                && m.k_cur < m.v_cur
                && m.v_cur < m.attn_out
                && m.attn_out < m.gate
                && m.gate < m.up
                && m.up < m.logits
        );
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    fn msl_patcher_prepends_spirv_cross_constant_id_define() {
        // build.rs's msl_strip pass leaves the source in this normalised
        // shape; the load-time patcher just prepends a `#define` for each
        // slot.
        let src = "constant uint MAX_CHUNKS = SPIRV_CROSS_CONSTANT_ID_1;\n\
                   constant uint K_V4 = SPIRV_CROSS_CONSTANT_ID_2;\n\
                   kernel void cs_main() {}\n";
        let s = super::msl_set_function_const_u32(src, 1, 1234);
        let s = super::msl_set_function_const_u32(&s, 2, 7);
        assert!(s.starts_with("#define SPIRV_CROSS_CONSTANT_ID_2 7u\n"));
        assert!(s.contains("#define SPIRV_CROSS_CONSTANT_ID_1 1234u\n"));
        // The original references stay in place — preprocessor substitution
        // resolves them through the prepended `#define`s.
        assert!(s.contains("constant uint MAX_CHUNKS = SPIRV_CROSS_CONSTANT_ID_1;"));
        assert!(s.contains("constant uint K_V4 = SPIRV_CROSS_CONSTANT_ID_2;"));
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    #[should_panic(expected = "not referenced in MSL source")]
    fn msl_patcher_panics_on_missing_slot() {
        let _ = super::msl_set_function_const_u32("kernel void cs_main() {}\n", 0, 32);
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    fn msl_shift_lifts_single_ssbo_off_collision_with_push_constant() {
        // matvec_q1_0 shape under `--msl-decoration-binding`: push constant
        // and SSBO 0 both land on `[[buffer(0)]]`. The shift moves all SSBOs
        // by +1 so the push constant has slot 0 to itself.
        let src = "kernel void cs_main(constant Params& p [[buffer(0)]], device WBuf& _17 [[buffer(0)]], device ActBuf& _170 [[buffer(1)]], uint3 wid [[threadgroup_position_in_grid]]) { }\n";
        let out = super::msl_shift_ssbo_buffer_indices(src);
        assert!(out.contains("constant Params& p [[buffer(0)]]"));
        assert!(out.contains("device WBuf& _17 [[buffer(1)]]"));
        assert!(out.contains("device ActBuf& _170 [[buffer(2)]]"));
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    fn msl_shift_handles_attention_split_shape() {
        // attention_split shape: 4 SSBOs at bindings 0..3 plus the push
        // constant at slot 0. Target layout is push @ 0, SSBOs at 1..4 in
        // binding order — exactly what +1 produces.
        let src = "kernel void cs_main(constant Params& p [[buffer(0)]], device ActBuf& _353 [[buffer(0)]], device KCache& _107 [[buffer(1)]], device VCache& _167 [[buffer(2)]], device Partials& _566 [[buffer(3)]]) { }\n";
        let out = super::msl_shift_ssbo_buffer_indices(src);
        assert!(out.contains("constant Params& p [[buffer(0)]]"));
        assert!(out.contains("device ActBuf& _353 [[buffer(1)]]"));
        assert!(out.contains("device KCache& _107 [[buffer(2)]]"));
        assert!(out.contains("device VCache& _167 [[buffer(3)]]"));
        assert!(out.contains("device Partials& _566 [[buffer(4)]]"));
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    fn msl_shift_handles_matvec_fused_normed_shape() {
        // matvec_q1_0_fused_normed shape: WBuf, ActBuf, NormBuf at bindings
        // 0, 1, 2 plus the push constant at slot 0.
        let src = "kernel void cs_main(constant Params& p [[buffer(0)]], device WBuf& a [[buffer(0)]], device ActBuf& b [[buffer(1)]], device NormBuf& c [[buffer(2)]]) { }\n";
        let out = super::msl_shift_ssbo_buffer_indices(src);
        assert!(out.contains("constant Params& p [[buffer(0)]]"));
        assert!(out.contains("device WBuf& a [[buffer(1)]]"));
        assert!(out.contains("device ActBuf& b [[buffer(2)]]"));
        assert!(out.contains("device NormBuf& c [[buffer(3)]]"));
    }

    #[cfg(not(target_vendor = "apple"))]
    #[test]
    fn spirv_patcher_rewrites_rms_norm_subgroup_size_default() {
        // Sanity-check the runtime SPIR-V patcher against the precompiled
        // rms_norm SPIR-V: the build emits OpSpecConstant with the GLSL
        // default of 32 for SpecId 0; after patching to 64, the default
        // operand should read 64.
        const RMS_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rms_norm.comp.spv"));
        let mut words: Vec<u32> = bytemuck::pod_collect_to_vec(RMS_SPV);
        super::spirv_set_spec_const_u32(&mut words, 0, 64);

        // Locate OpDecorate(SpecId, 0) → result-id, then the matching OpSpecConstant.
        let mut target_id = None;
        let mut i = 5;
        while i < words.len() {
            let header = words[i];
            let len = (header >> 16) as usize;
            if (header & 0xFFFF) == 71 && len >= 4 && words[i + 2] == 1 && words[i + 3] == 0 {
                target_id = Some(words[i + 1]);
                break;
            }
            i += len;
        }
        let target_id = target_id.expect("SpecId 0 decoration");
        let mut i = 5;
        let mut patched = None;
        while i < words.len() {
            let header = words[i];
            let len = (header >> 16) as usize;
            if (header & 0xFFFF) == 50 && len == 4 && words[i + 2] == target_id {
                patched = Some(words[i + 3]);
                break;
            }
            i += len;
        }
        assert_eq!(patched, Some(64));
    }

    #[test]
    fn params_struct_sizes_fit_immediate_limit() {
        // All Params structs must fit within the 128-byte minimum Vulkan push-constant
        // (immediates) limit, which is what we request via max_immediate_size.
        const LIMIT: usize = 128;
        assert!(size_of::<EmbedParams>() <= LIMIT);
        assert!(size_of::<RmsNormParams>() <= LIMIT);
        assert!(size_of::<MatvecParams>() <= LIMIT);
        assert!(size_of::<MatvecSiluParams>() <= LIMIT);
        assert!(size_of::<MatvecFusedParams>() <= LIMIT);
        assert!(size_of::<MatvecFusedNormedParams>() <= LIMIT);
        assert!(size_of::<MatmulParams>() <= LIMIT);
        assert!(size_of::<AttnPrefillTiledParams>() <= LIMIT);
        assert!(size_of::<AttnSplitParams>() <= LIMIT);
        assert!(size_of::<AttnMergeParams>() <= LIMIT);
        assert!(size_of::<RmsNormQ8Params>() <= LIMIT);
        assert!(size_of::<SiluMulQ8Params>() <= LIMIT);
        assert!(size_of::<TopKPartialParams>() <= LIMIT);
        assert!(size_of::<TopKMergeParams>() <= LIMIT);
        assert!(size_of::<KvWritebackFusedParams>() <= LIMIT);
        assert!(size_of::<QNormRopeFusedParams>() <= LIMIT);
    }

    #[test]
    fn build_rope_table_shape_and_values() {
        let cfg = rope_test_cfg(); // head_dim=8, freq_base=10_000, max_seq not in cfg
        let max_seq = 4u32;
        let table = build_rope_table(&cfg, max_seq);
        assert_eq!(table.len(), (max_seq * cfg.head_dim) as usize);
        // pos=0: all cos=1.0, sin=0.0
        assert!((table[0] - 1.0f32).abs() < 1e-6); // cos(0)
        assert!((table[1] - 0.0f32).abs() < 1e-6); // sin(0)
        // pos=1, j=0: theta=10000^0=1.0, angle=1.0
        let cos1 = (1.0f64).cos() as f32;
        let sin1 = (1.0f64).sin() as f32;
        assert!(
            (table[8] - cos1).abs() < 1e-5,
            "cos(1)={} got {}",
            cos1,
            table[8]
        );
        assert!(
            (table[9] - sin1).abs() < 1e-5,
            "sin(1)={} got {}",
            sin1,
            table[9]
        );
    }
}
