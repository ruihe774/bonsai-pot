//! Bonsai (Qwen3 architecture) `Q1_0` inference: model loading & GPU resource setup.
//!
//! All large weights live in 5 storage buffers organized by role; the activation
//! workspace is a single f16 buffer with named regions plus a separate buffer for
//! `Q8_0` activations (used by the `dot4I8Packed` matmul path). Norm weights and the
//! `RoPE` cos/sin table are also f16; `Q8_0` scales remain f32.

use std::collections::HashMap;
use std::fs::{read, read_to_string, write};
use std::mem::size_of;
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::str::{FromStr, from_utf8};
use std::sync::{Arc, OnceLock};

use ash::vk;
use bytemuck::{Pod, Zeroable, cast_slice};
use wgpu::hal::DeviceError;
use wgpu::hal::api::Vulkan as VulkanApi;
use wgpu::util::DeviceExt as _;

use crate::decode;
use crate::error::{PotError, Result};

// ----- config & manifest ----------------------------------------------------

#[derive(Debug, Clone)]
pub struct TensorEntry {
    pub(crate) dtype: String,
    pub(crate) shape: Vec<u64>,
    pub(crate) buffer: String,
    pub(crate) offset: u64,
    pub(crate) length: u64,
    pub(crate) d_offset: u64,
    pub(crate) qs_offset: u64,
    pub(crate) nb: u64,
}

/// Internal config: the full struct deserialized from `config.ini`.
#[derive(Debug, Clone)]
pub struct ConfigRaw {
    pub(crate) n_layer: u32,
    pub(crate) n_embd: u32,
    pub(crate) n_ff: u32,
    pub(crate) n_head: u32,
    pub(crate) n_kv_head: u32,
    pub(crate) head_dim: u32,
    pub(crate) rope_freq_base: f32,
    pub(crate) rms_eps: f32,
    pub(crate) n_vocab: u32,
    pub(crate) eos_token_id: u32,
    pub(crate) padding_token_id: u32,
    pub(crate) add_bos: bool,
    pub(crate) context_length: u32,
    pub(crate) rope_orig_context: u32,
    pub(crate) n_kv_groups: u32,
    pub(crate) q_dim: u32,
    pub(crate) kv_dim: u32,
    pub(crate) tied_embeddings: bool,
    pub(crate) manifest: HashMap<String, TensorEntry>,
}

// Internal alias for the full config — most code uses this name and reaches the
// fields directly. Kept as `Config` so existing call sites are minimally
// disturbed.
pub type Config = ConfigRaw;

/// Public, read-only view of the model's hyperparameters. Stable for the
/// library API; does not expose the GGUF tensor manifest or any other internal
/// fields.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub n_layer: u32,
    pub n_embd: u32,
    pub n_head: u32,
    pub n_kv_head: u32,
    pub head_dim: u32,
    pub n_vocab: u32,
    pub eos_token_id: u32,
    pub padding_token_id: u32,
    pub context_length: u32,
    pub tied_embeddings: bool,
}

impl ModelConfig {
    const fn from_raw(c: &ConfigRaw) -> Self {
        Self {
            n_layer: c.n_layer,
            n_embd: c.n_embd,
            n_head: c.n_head,
            n_kv_head: c.n_kv_head,
            head_dim: c.head_dim,
            n_vocab: c.n_vocab,
            eos_token_id: c.eos_token_id,
            padding_token_id: c.padding_token_id,
            context_length: c.context_length,
            tied_embeddings: c.tied_embeddings,
        }
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
    pub(crate) _p0: u32,
    pub(crate) _p1: u32,
    pub(crate) _p2: u32,
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
    pub(crate) input_offset_1: u32,
    pub(crate) output_offset_1: u32,
    pub(crate) weight_offset_1: u32,
    pub(crate) n_groups_0: u32,
}
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Default, Debug)]
pub struct RopeParams {
    pub(crate) head_dim: u32,
    pub(crate) n_heads: u32,
    pub(crate) n_tokens: u32,
    pub(crate) pos_base: u32,
    pub(crate) data_offset: u32,
    pub(crate) data_offset_1: u32,
    pub(crate) n_heads_0: u32,
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
pub struct QuantParams {
    pub(crate) k: u32,
    pub(crate) m: u32,
    pub(crate) input_offset: u32,
    pub(crate) d_offset: u32,
    pub(crate) qs_offset: u32,
    pub(crate) dispatch_x_dim: u32,
    pub(crate) _p1: u32,
    pub(crate) _p2: u32,
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
    pub(crate) _p0: u32,
    pub(crate) _p1: u32,
    pub(crate) _p2: u32,
}
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Default, Debug)]
pub struct AttnParams {
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
    pub(crate) out_offset: u32,
    pub(crate) scale: f32,
    pub(crate) m_tokens: u32,
    pub(crate) is_prefill: u32,
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
    pub(crate) _p0: u32,
    pub(crate) _p1: u32,
    pub(crate) _p2: u32,
    pub(crate) _p3: u32,
}
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Default, Debug)]
pub struct SiluMulParams {
    pub(crate) n: u32,
    pub(crate) m: u32,
    pub(crate) gate_offset: u32,
    pub(crate) up_offset: u32,
    pub(crate) out_offset: u32,
    pub(crate) dispatch_x_count: u32,
    pub(crate) _p1: u32,
    pub(crate) _p2: u32,
}
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Default, Debug)]
pub struct TopKParams {
    pub(crate) n: u32,
    pub(crate) in_offset: u32,
    pub(crate) out_offset: u32,
    pub(crate) k: u32,
}
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Default, Debug)]
pub struct KvWritebackParams {
    pub(crate) k_cur_off: u32,          // f16 element offset in act
    pub(crate) v_cur_off: u32,          // f16 element offset in act
    pub(crate) dst_d_word_offset: u32, // u32-word offset into kv_{k,v} (FP32 d-section, layer base)
    pub(crate) dst_qs_byte_offset: u32, // byte offset into kv_{k,v} (qs-section, layer base)
    pub(crate) pos_base: u32,          // first absolute cache position to write into
    pub(crate) nb_per_row: u32,        // kv_dim / 32
    pub(crate) kv_dim: u32,
    pub(crate) dispatch_x_dim: u32,
}

// All of these <= 64 bytes; we pack each into a 256-byte uniform slot.
pub const UNIFORM_SLOT_SIZE: u64 = 256;

/// Maximum K supported by the `topk_reduce` shader (matches `K_MAX` in the WGSL).
pub const TOPK_MAX: u32 = 32;

// ----- activation layout ----------------------------------------------------

/// Region offsets into the single f16 activation buffer. All values are
/// **element offsets** (count of f16 elements from the buffer's start), to
/// match the shader-side `array<f16>` indexing — never bytes.
#[derive(Copy, Clone, Debug)]
pub struct ActLayout {
    pub(crate) x: u32,
    pub(crate) x_norm: u32,
    pub(crate) q: u32,
    pub(crate) k_cur: u32,
    pub(crate) v_cur: u32,
    pub(crate) attn_out: u32,
    pub(crate) gate: u32,
    pub(crate) up: u32,
    pub(crate) ffn_in: u32,
    pub(crate) logits: u32,
    pub(crate) total_elems: u32,
}

impl ActLayout {
    pub fn build(cfg: &Config, m_max: u32) -> Self {
        let mut o = 0u32;
        let alloc = |n: u32, o: &mut u32| -> u32 {
            let r = *o;
            *o += n;
            r
        };
        let x = alloc(m_max * cfg.n_embd, &mut o);
        let x_norm = alloc(m_max * cfg.n_embd, &mut o);
        let q = alloc(m_max * cfg.q_dim, &mut o);
        let k_cur = alloc(m_max * cfg.kv_dim, &mut o);
        let v_cur = alloc(m_max * cfg.kv_dim, &mut o);
        let attn_out = alloc(m_max * cfg.q_dim, &mut o);
        let gate = alloc(m_max * cfg.n_ff, &mut o);
        let up = alloc(m_max * cfg.n_ff, &mut o);
        let ffn_in = alloc(m_max * cfg.n_ff, &mut o);
        let logits = alloc(cfg.n_vocab, &mut o);
        Self {
            x,
            x_norm,
            q,
            k_cur,
            v_cur,
            attn_out,
            gate,
            up,
            ffn_in,
            logits,
            total_elems: o,
        }
    }
}

// ----- the model ------------------------------------------------------------

pub struct Pipelines {
    pub(crate) embed: wgpu::ComputePipeline,
    pub(crate) rms_norm: wgpu::ComputePipeline,
    pub(crate) rope_neox: wgpu::ComputePipeline,
    pub(crate) matvec: wgpu::ComputePipeline,
    pub(crate) matvec_fused: wgpu::ComputePipeline,
    pub(crate) quantize: wgpu::ComputePipeline,
    pub(crate) matmul: wgpu::ComputePipeline,
    pub(crate) attention: wgpu::ComputePipeline,
    pub(crate) attention_split: wgpu::ComputePipeline,
    pub(crate) attention_merge: wgpu::ComputePipeline,
    pub(crate) silu_mul: wgpu::ComputePipeline,
    pub(crate) topk_reduce: wgpu::ComputePipeline,
    pub(crate) kv_writeback: wgpu::ComputePipeline,
}

pub struct Buffers {
    pub(crate) w_attn: wgpu::Buffer,
    pub(crate) w_ffn_gu: wgpu::Buffer,
    pub(crate) w_ffn_d: wgpu::Buffer,
    pub(crate) w_norms: wgpu::Buffer,
    pub(crate) w_embed: wgpu::Buffer,
    pub(crate) kv_k: wgpu::Buffer,
    pub(crate) kv_v: wgpu::Buffer,
    pub(crate) act: wgpu::Buffer,           // f16 activations
    pub(crate) act_q8: wgpu::Buffer,        // Q8_0 activations (raw u32 buffer)
    pub(crate) attn_partials: wgpu::Buffer, // f32 partials for split-K attention
    pub(crate) rope_table: wgpu::Buffer,
    pub(crate) uniform: wgpu::Buffer,  // pool of 256-byte slots
    pub(crate) sample: wgpu::Buffer, // u32 storage: input token id @ [0..M], topk output @ [0..2K]
    pub(crate) readback: wgpu::Buffer, // u32 readback (mappable)
}

pub struct BindGroupLayouts {
    pub(crate) embed: wgpu::BindGroupLayout,
    pub(crate) rms_norm: wgpu::BindGroupLayout,
    pub(crate) rope: wgpu::BindGroupLayout,
    pub(crate) matvec: wgpu::BindGroupLayout,
    pub(crate) quantize: wgpu::BindGroupLayout,
    pub(crate) matmul: wgpu::BindGroupLayout,
    pub(crate) attn: wgpu::BindGroupLayout,
    pub(crate) attn_split: wgpu::BindGroupLayout,
    pub(crate) attn_merge: wgpu::BindGroupLayout,
    pub(crate) silu_mul: wgpu::BindGroupLayout,
    pub(crate) topk_reduce: wgpu::BindGroupLayout,
    pub(crate) kv_writeback: wgpu::BindGroupLayout,
}

/// Pre-built bind groups indexed by (BGL kind, weight buffer). The UBO binding
/// in every cached BG is sized to one full uniform slot (`UNIFORM_SLOT_SIZE`)
/// since we use dynamic offsets and the slot is large enough to hold any of
/// the per-dispatch params structs. One BG per (kind, weight buffer) is reused
/// across every dispatch of that kind in a step.
pub struct CachedBindGroups {
    pub(crate) embed: wgpu::BindGroup, // (uniform, w_embed, act, sample)
    pub(crate) rms_norm: wgpu::BindGroup, // (uniform, act, w_norms)
    pub(crate) rope: wgpu::BindGroup,  // (uniform, rope_table, act)
    pub(crate) matvec_w_attn: wgpu::BindGroup, // (uniform, w_attn,   act)
    pub(crate) matvec_w_ffn_gu: wgpu::BindGroup, // (uniform, w_ffn_gu, act)
    pub(crate) matvec_w_ffn_d: wgpu::BindGroup, // (uniform, w_ffn_d,  act)
    pub(crate) matvec_w_embed: wgpu::BindGroup, // (uniform, w_embed,  act) — LM head
    pub(crate) quantize: wgpu::BindGroup, // (uniform, act, act_q8)
    pub(crate) matmul_w_attn: wgpu::BindGroup, // (uniform, w_attn,   act_q8, act)
    pub(crate) matmul_w_ffn_gu: wgpu::BindGroup,
    pub(crate) matmul_w_ffn_d: wgpu::BindGroup,
    pub(crate) matmul_w_embed: wgpu::BindGroup,
    pub(crate) attn: wgpu::BindGroup, // (uniform, act, kv_k, kv_v)
    pub(crate) attn_split: wgpu::BindGroup, // (uniform, act, kv_k, kv_v, attn_partials)
    pub(crate) attn_merge: wgpu::BindGroup, // (uniform, act, attn_partials)
    pub(crate) silu_mul: wgpu::BindGroup, // (uniform, act)
    pub(crate) topk_reduce: wgpu::BindGroup, // (uniform, act, sample)
    pub(crate) kv_writeback: wgpu::BindGroup, // (uniform, act, kv_k, kv_v)
}

/// Selects which weight buffer a matvec / matmul dispatch reads from. Maps
/// directly to one of the cached bind groups in [`CachedBindGroups`].
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
/// Stored in an `OnceLock` so reads are lock-free and the first writer wins.
#[derive(Debug, Clone)]
pub struct DeviceLostInfo {
    pub reason: wgpu::DeviceLostReason,
    pub message: String,
}

/// GPU-bearing handle to a loaded Bonsai model.
///
/// Holds weights, pipelines, activations, and the KV cache. Cheap to share
/// across [`crate::Session`]s (they each carry only their own `pos` cursor);
/// but only one inference can run at a time because they share the
/// activation/KV/sample buffers.
pub struct Model {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) cfg: Config,
    pub(crate) public_cfg: ModelConfig,
    pub(crate) act_layout: ActLayout,
    pub(crate) m_max: u32,
    pub(crate) max_seq: u32,
    pub(crate) buffers: Buffers,
    pub(crate) pipes: Pipelines,
    pub(crate) cached: CachedBindGroups,
    pub(crate) layer_tensors: Vec<LayerTensors>,
    pub(crate) output_tensors: OutputTensors,
    pub(crate) vocab: Vec<String>,
    pub(crate) lost: Arc<OnceLock<DeviceLostInfo>>,
}

const M_MAX: u32 = 512;
const DEFAULT_MAX_SEQ: u32 = 1024;
/// Cache positions per workgroup in the split-K attention pass. Must match
/// `CHUNK_SIZE` in `attention_split.wgsl`.
pub const ATTN_CHUNK_SIZE: u32 = 32;
// 4096 * 256 B = 1 MiB. Each per-token step uses ~450 slots, matmul prefill ~470.
// 4096 leaves ~8x headroom and is 16x smaller than the historical 65536.
pub const UNIFORM_POOL_SLOTS: u64 = 4096;

/// Allocate-time tunables for [`Model::load_with_options`].
///
/// These affect GPU buffer sizing (KV cache, `RoPE` table) and so cannot be
/// changed per call — pick them once at load.
#[derive(Debug, Clone)]
pub struct ModelOptions {
    /// Maximum sequence length (positions in the KV cache). Default: 1024.
    ///
    /// VRAM cost is linear: KV cache (`Q8_0` K and V combined) uses roughly
    /// `n_layer * max_seq * kv_dim * 2.25 bytes` — 32 i8 qs + one FP32 scale
    /// per 32-element block, doubled for K and V. The `RoPE` table grows as
    /// `max_seq * head_dim * 2 bytes`. The shaders themselves don't bake in
    /// a sequence-length limit, so any value up to the model's
    /// `context_length` is supported (subject to VRAM).
    pub max_seq: u32,
    /// Path to persist the wgpu pipeline cache between runs. `None` disables
    /// caching. Best-effort: backends without `Features::PIPELINE_CACHE`
    /// silently skip, and read/write errors are logged but never fatal.
    pub pipeline_cache_path: Option<PathBuf>,
    /// Priority of the command queue, a normalized floating-point value between 0.0 and 1.0,
    /// which is then translated to a discrete priority level by the implementation.
    /// Higher values indicate a higher priority,
    /// with 0.0 being the lowest priority and 1.0 being the highest.
    /// The implementation makes no guarantees with regards to queues across different devices.
    pub queue_priority: f32,
}

impl Default for ModelOptions {
    fn default() -> Self {
        Self {
            max_seq: DEFAULT_MAX_SEQ,
            pipeline_cache_path: None,
            queue_priority: 0.0,
        }
    }
}

fn parse_config_ini(text: &str) -> Result<Config> {
    fn get<'a>(map: &'a HashMap<&str, &str>, key: &'static str) -> Result<&'a str> {
        map.get(key)
            .copied()
            .ok_or(PotError::Config("config.ini missing required field"))
    }
    fn parse_field<T: FromStr>(map: &HashMap<&str, &str>, key: &'static str) -> Result<T> {
        get(map, key)?
            .parse()
            .map_err(|_| PotError::Config("config.ini field has invalid value"))
    }
    fn parse_opt<T: FromStr + Default>(map: &HashMap<&str, &str>, key: &'static str) -> Result<T> {
        map.get(key).map_or_else(
            || Ok(T::default()),
            |v| {
                v.parse()
                    .map_err(|_| PotError::Config("config.ini field has invalid value"))
            },
        )
    }
    fn build_entry(g: &HashMap<&str, &str>) -> Result<TensorEntry> {
        let shape = get(g, "shape")?
            .split(',')
            .map(|s| {
                s.parse::<u64>()
                    .map_err(|_| PotError::Config("config.ini shape has invalid value"))
            })
            .collect::<Result<Vec<u64>>>()?;
        Ok(TensorEntry {
            dtype: get(g, "dtype")?.to_string(),
            shape,
            buffer: get(g, "buffer")?.to_string(),
            offset: parse_field(g, "offset")?,
            length: parse_field(g, "length")?,
            d_offset: parse_opt(g, "d_offset")?,
            qs_offset: parse_opt(g, "qs_offset")?,
            nb: parse_opt(g, "nb")?,
        })
    }

    let mut globals: HashMap<&str, &str> = HashMap::new();
    let mut manifest: HashMap<String, TensorEntry> = HashMap::new();
    let mut cur_section: Option<&str> = None;
    let mut cur_fields: HashMap<&str, &str> = HashMap::new();

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
    Ok(ConfigRaw {
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
        padding_token_id: parse_field(g, "padding_token_id")?,
        add_bos: get(g, "add_bos")? == "true",
        context_length: parse_field(g, "context_length")?,
        rope_orig_context: parse_field(g, "rope_orig_context")?,
        n_kv_groups: parse_field(g, "n_kv_groups")?,
        q_dim: parse_field(g, "q_dim")?,
        kv_dim: parse_field(g, "kv_dim")?,
        tied_embeddings: get(g, "tied_embeddings")? == "true",
        manifest,
    })
}

fn validate_cfg(cfg: &Config) -> Result<()> {
    const fn pad4(n: u64) -> u64 {
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

    // --- per-tensor checks ---
    // Each spec: (name, dtype, shape as [u64;2-or-1], buffer filename)
    // For 1-D F16 tensors encode shape as &[dim] (single element).

    let n_embd = u64::from(cfg.n_embd);
    let n_ff = u64::from(cfg.n_ff);
    let q_dim = u64::from(cfg.q_dim);
    let kv_dim = u64::from(cfg.kv_dim);
    let n_vocab = u64::from(cfg.n_vocab);
    let head_dim = u64::from(cfg.head_dim);

    // Per-layer tensors (iterated over all layers).
    let layer_specs: &[(&str, &str, &[u64], &str)] = &[
        (
            "attn_q.weight",
            "Q1_0",
            &[n_embd, q_dim],
            "weights_attn.bin",
        ),
        (
            "attn_k.weight",
            "Q1_0",
            &[n_embd, kv_dim],
            "weights_attn.bin",
        ),
        (
            "attn_v.weight",
            "Q1_0",
            &[n_embd, kv_dim],
            "weights_attn.bin",
        ),
        (
            "attn_output.weight",
            "Q1_0",
            &[q_dim, n_embd],
            "weights_attn.bin",
        ),
        (
            "ffn_gate.weight",
            "Q1_0",
            &[n_embd, n_ff],
            "weights_ffn_gate_up.bin",
        ),
        (
            "ffn_up.weight",
            "Q1_0",
            &[n_embd, n_ff],
            "weights_ffn_gate_up.bin",
        ),
        (
            "ffn_down.weight",
            "Q1_0",
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
    let global_specs: &[(&str, &str, &[u64], &str)] = &[
        ("output_norm.weight", "F16", &[n_embd], "weights_norms.bin"),
        (
            "token_embd.weight",
            "Q1_0",
            &[n_embd, n_vocab],
            "weights_embed_lmhead.bin",
        ),
    ];

    let check = |name: &str, dtype: &str, shape: &[u64], buffer: &str| -> Result<()> {
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
            "Q1_0" => {
                let n_in = shape[0];
                let n_out = if shape.len() > 1 { shape[1] } else { 1 };
                if !n_in.is_multiple_of(128) {
                    return Err(PotError::Config(
                        "manifest: Q1_0 tensor n_in not divisible by 128",
                    ));
                }
                let nb = n_in / 128;
                if e.nb != nb {
                    return Err(PotError::Config("manifest: Q1_0 tensor nb != n_in/128"));
                }
                let expected_qs_offset = pad4(e.d_offset + n_out * nb * 2);
                if e.qs_offset != expected_qs_offset {
                    return Err(PotError::Config(
                        "manifest: Q1_0 tensor qs_offset != pad4(d_offset + n_out*nb*2)",
                    ));
                }
                let expected_length = (e.qs_offset - e.d_offset) + pad4(n_out * nb * 16);
                if e.length != expected_length {
                    return Err(PotError::Config("manifest: Q1_0 tensor length mismatch"));
                }
            }
            "F16" => {
                let expected_length = pad4(shape.iter().product::<u64>() * 2);
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
            "Q1_0",
            &[n_embd, n_vocab],
            "weights_embed_lmhead.bin",
        )?;
    }

    Ok(())
}

impl Model {
    /// Load weights with default options. Equivalent to
    /// [`Model::load_with_options`] with `ModelOptions::default()`.
    ///
    /// # Errors
    ///
    /// See [`Model::load_with_options`].
    pub async fn load(model_dir: &Path) -> Result<Self> {
        Self::load_with_options(model_dir, ModelOptions::default()).await
    }

    /// Load weights, build pipelines, allocate the KV cache. Reads
    /// `config.json`, `weights_*.bin`, `vocab.bin`, and `vocab_offsets.bin` from
    /// `model_dir`. Does **not** read `prompt.bin` — callers supply
    /// pre-tokenized prompts via [`crate::Session`].
    ///
    /// # Errors
    ///
    /// Returns an error if `opts.max_seq == 0`, the model files cannot be read
    /// or parsed, no suitable wgpu adapter is available, the adapter does not
    /// support the required features (`SHADER_F16`, `SUBGROUP`), the runtime
    /// subgroup size is unsupported, or the vocab files are malformed.
    pub async fn load_with_options(model_dir: &Path, opts: ModelOptions) -> Result<Self> {
        // Hoisted constants/items so we don't trip items-after-statements lints.
        const SAMPLE_BYTES: u64 = 4 * 1024;
        const fn ubo_dyn(binding: u32) -> wgpu::BindGroupLayoutEntry {
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: None,
                },
                count: None,
            }
        }
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
        let cfg_path = model_dir.join("config.ini");
        let cfg_text = read_to_string(&cfg_path).map_err(|e| PotError::Io {
            path: cfg_path.clone(),
            source: e,
        })?;
        let cfg = parse_config_ini(&cfg_text)?;
        validate_cfg(&cfg)?;
        let public_cfg = ModelConfig::from_raw(&cfg);

        // ---- wgpu init ------------------------------------------------------
        // Use new_without_display_handle_from_env() so that WGPU_BACKEND and
        // WGPU_ADAPTER_NAME env vars are honoured at instance creation time.
        // Instance::default() uses all backends and ignores WGPU_BACKEND.
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|_| PotError::NoAdapter)?;
        log::info!("adapter: {:?}", adapter.get_info());

        if !adapter.features().contains(wgpu::Features::SHADER_F16) {
            return Err(PotError::FeatureUnsupported("SHADER_F16"));
        }
        if !adapter.features().contains(wgpu::Features::SUBGROUP) {
            return Err(PotError::FeatureUnsupported("SUBGROUP"));
        }
        let pipeline_cache_supported = adapter.features().contains(wgpu::Features::PIPELINE_CACHE);
        if opts.pipeline_cache_path.is_some() && !pipeline_cache_supported {
            log::info!(
                "pipeline_cache_path set but adapter does not support PIPELINE_CACHE; skipping"
            );
        }

        let info = adapter.get_info();
        log::info!(
            "adapter subgroup range: min={}, max={}",
            info.subgroup_min_size,
            info.subgroup_max_size,
        );

        let mut limits = adapter.limits();
        // Floor at 300 MB so the largest grouped weight buffer (~510 MB at 8B) fits
        // even on adapters whose wgpu defaults are below that. No upper cap — the
        // KV cache for 8B at practical 32k ctx needs ~1.27 GB per buffer, so we
        // rely on the adapter's natural max_buffer_size (typically 2–4 GB on desktop).
        limits.max_storage_buffer_binding_size = limits
            .max_storage_buffer_binding_size
            .max(300 * 1024 * 1024);
        limits.max_buffer_size = limits.max_buffer_size.max(300 * 1024 * 1024);
        limits.max_storage_buffers_per_shader_stage =
            limits.max_storage_buffers_per_shader_stage.max(8);

        let mut required_features = wgpu::Features::SHADER_F16 | wgpu::Features::SUBGROUP;
        if pipeline_cache_supported && opts.pipeline_cache_path.is_some() {
            required_features |= wgpu::Features::PIPELINE_CACHE;
        }
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
        let hal_open = unsafe {
            adapter.as_hal::<VulkanApi>().map(|hal_adapter| {
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

                let enabled_extensions =
                    hal_adapter.required_device_extensions(desc.required_features);
                let mut enabled_phd_features = hal_adapter
                    .physical_device_features(&enabled_extensions, desc.required_features);

                let priorities = [opts.queue_priority];
                let queue_infos = [vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(family_idx)
                    .queue_priorities(&priorities)];

                let str_pointers: Vec<_> = enabled_extensions.iter().map(|s| s.as_ptr()).collect();

                let pre_info = vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queue_infos)
                    .enabled_extension_names(&str_pointers);
                let info = enabled_phd_features.add_to_device_create(pre_info);

                let raw_device = match instance.create_device(pd, &info, None) {
                    Ok(d) => d,
                    Err(e) => {
                        // Mirror wgpu-hal's own `vkCreateDevice` error mapping
                        // (private helpers `hal_usage_error` /
                        // `map_host_device_oom_and_lost_err` aren't reachable
                        // from outside the crate, so we inline the cases).
                        log::warn!("vkCreateDevice on family {family_idx} failed: {e:?}");
                        let mapped = match e {
                            vk::Result::ERROR_OUT_OF_HOST_MEMORY
                            | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
                            | vk::Result::ERROR_TOO_MANY_OBJECTS => DeviceError::OutOfMemory,
                            vk::Result::ERROR_INITIALIZATION_FAILED
                            | vk::Result::ERROR_DEVICE_LOST => DeviceError::Lost,
                            _ => DeviceError::Unexpected,
                        };
                        return Err(mapped);
                    }
                };

                hal_adapter.device_from_raw(
                    raw_device,
                    None,
                    &enabled_extensions,
                    desc.required_features,
                    &desc.required_limits,
                    &desc.memory_hints,
                    family_idx,
                    0,
                )
            })
        };
        let (device, queue) = match hal_open {
            Some(Ok(open)) => unsafe { adapter.create_device_from_hal(open, &desc)? },
            Some(Err(e)) => {
                log::warn!("Vulkan HAL open failed ({e:?}), falling back to request_device");
                adapter.request_device(&desc).await?
            }
            None => adapter.request_device(&desc).await?,
        };

        // ---- pipeline cache (best-effort) -----------------------------------
        // Only created when the caller supplied a path AND the adapter supports
        // PIPELINE_CACHE. On a cold start `prior` is None and the cache is
        // created empty; the driver populates it as pipelines compile, and the
        // writeback step after pipeline construction persists it for future runs.
        let pipeline_cache = match (pipeline_cache_supported, opts.pipeline_cache_path.as_ref()) {
            (true, Some(path)) => {
                let prior = read(path).ok();
                log::info!(
                    "pipeline cache: loading {} prior bytes from {}",
                    prior.as_deref().map_or(0, <[u8]>::len),
                    path.display(),
                );
                // SAFETY: `fallback: true` means the driver silently discards
                // invalid data (e.g. corrupt file, driver version change) rather
                // than causing UB. Data comes from a prior `get_data()` call so
                // the driver family normally matches; fallback handles the rest.
                let cache = unsafe {
                    device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                        label: Some("pot_pipeline_cache"),
                        data: prior.as_deref(),
                        fallback: true,
                    })
                };
                Some((cache, path.clone(), prior))
            }
            _ => None,
        };

        // ---- wire up device-lost and uncaptured-error callbacks ------------
        // Both callbacks write into `lost` via OnceLock; the first writer wins
        // (the device-lost reason is more specific, so that path fires first).
        let lost: Arc<OnceLock<DeviceLostInfo>> = Arc::new(OnceLock::new());
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

        // ---- probe actual runtime subgroup size ----------------------------
        // The adapter exposes the device's subgroup_min..max range, not the
        // size the driver picks for any given pipeline. RADV in particular
        // can be flipped to wave32 via `RADV_PERFTEST=cswave32` or kept at
        // wave64 (default), and either choice is opaque to the API. We dispatch
        // a tiny `subgroup_size`-builtin readback to learn the real value, then
        // bake that into every other shader at compile time.
        let sg_size = probe_subgroup_size(&device, &queue, &lost)?;
        if !(8..=64).contains(&sg_size) || (sg_size & (sg_size - 1)) != 0 {
            return Err(PotError::Config(
                "unsupported runtime subgroup size (need power-of-2 in [8, 64])",
            ));
        }
        log::info!(
            "adapter={} backend={:?} subgroup_runtime={} (adapter range={}..={})",
            info.name,
            info.backend,
            sg_size,
            info.subgroup_min_size,
            info.subgroup_max_size,
        );

        // ---- load weight buffers from disk ---------------------------------
        let load = |fname: &str| -> Result<Vec<u8>> {
            let p = model_dir.join(fname);
            read(&p).map_err(|e| PotError::Io { path: p, source: e })
        };
        let w_storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let make_storage = |label: &str, bytes: &[u8]| -> wgpu::Buffer {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: w_storage,
            })
        };
        let w_attn = make_storage("w_attn", &load("weights_attn.bin")?);
        let w_ffn_gu = make_storage("w_ffn_gu", &load("weights_ffn_gate_up.bin")?);
        let w_ffn_d = make_storage("w_ffn_d", &load("weights_ffn_down.bin")?);
        let w_norms = make_storage("w_norms", &load("weights_norms.bin")?);
        let w_embed = make_storage("w_embed", &load("weights_embed_lmhead.bin")?);

        // ---- build RoPE table (f32 host-side, then downcast to f16) --------
        let rope_table_f32 = build_rope_table(&cfg, opts.max_seq);
        let rope_table_f16: Vec<half::f16> = rope_table_f32
            .iter()
            .map(|&v| half::f16::from_f32(v))
            .collect();
        let rope_buf = make_storage("rope_table", cast_slice(&rope_table_f16));

        // ---- KV cache (Q8_0), activations, scratch -------------------------
        // Layout per buffer (kv_k / kv_v):
        //   d-section  (FP32 scales): bytes [0, d_total_bytes)
        //   qs-section (i8 packed):   bytes [d_total_bytes, d_total_bytes + qs_total_bytes)
        // Block size = 32 elements; one FP32 scale per block. The two sections
        // are u32-aligned (4 | block-bytes; 4 | row-bytes given kv_dim % 32 == 0
        // and kv_dim >= 32), so reads/writes are clean word loads.
        if !cfg.kv_dim.is_multiple_of(32) {
            return Err(PotError::Config(
                "kv_dim must be a multiple of 32 (Q8_0 block size)",
            ));
        }
        let nb_per_row = u64::from(cfg.kv_dim / 32);
        let kv_d_total: u64 = u64::from(cfg.n_layer) * u64::from(opts.max_seq) * nb_per_row * 4;
        let kv_qs_total: u64 =
            u64::from(cfg.n_layer) * u64::from(opts.max_seq) * u64::from(cfg.kv_dim);
        let kv_total: u64 = kv_d_total + kv_qs_total;
        {
            let dl = device.limits();
            let max_buf = dl.max_buffer_size;
            let max_bind = dl.max_storage_buffer_binding_size;
            if kv_total > max_buf || kv_total > max_bind {
                return Err(PotError::Config(
                    "KV cache exceeds adapter buffer/binding limit; \
                     reduce --max-seq or use a GPU with larger storage buffers",
                ));
            }
        }
        let kv_k = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kv_k"),
            size: kv_total,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let kv_v = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kv_v"),
            size: kv_total,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let act_layout = ActLayout::build(&cfg, M_MAX);
        let act_size = (u64::from(act_layout.total_elems) * size_of::<half::f16>() as u64 + 3) & !3;
        let act_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("act"),
            size: act_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let max_k = cfg.n_ff;
        let q8_d_section_bytes = M_MAX * (max_k / 32) * 4;
        let q8_qs_section_bytes = M_MAX * max_k;
        let act_q8_size = q8_d_section_bytes + q8_qs_section_bytes;
        let act_q8_size = act_q8_size.div_ceil(16) * 16;
        let act_q8_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("act_q8"),
            size: u64::from(act_q8_size),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Split-K attention partials: per layer, n_head * n_chunks_max chunks of
        // (head_dim + 2) f32 each. Reused across layers within a step.
        let n_chunks_max = opts.max_seq.div_ceil(ATTN_CHUNK_SIZE);
        let attn_partials_size =
            u64::from(cfg.n_head) * u64::from(n_chunks_max) * (u64::from(cfg.head_dim) + 2) * 4;
        let attn_partials_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("attn_partials"),
            size: attn_partials_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniform"),
            size: UNIFORM_POOL_SLOTS * UNIFORM_SLOT_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Sample storage + readback. Sized for: M_MAX prompt token ids during
        // matmul prefill (M_MAX * 4 = 2048 bytes), or 2*TOPK_MAX u32 entries
        // for the topk_reduce output (K floats + K indices = 2*64*4 = 512 bytes).
        // 4 KB is comfortable.
        let sample = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sample"),
            size: SAMPLE_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: SAMPLE_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let buffers = Buffers {
            w_attn,
            w_ffn_gu,
            w_ffn_d,
            w_norms,
            w_embed,
            kv_k,
            kv_v,
            act: act_buf,
            act_q8: act_q8_buf,
            attn_partials: attn_partials_buf,
            rope_table: rope_buf,
            uniform: uniform_buf,
            sample,
            readback,
        };

        // ---- shaders & pipelines -------------------------------------------
        // `{{SG_SIZE}}` and `{{MAX_CHUNKS}}` are substituted with runtime values
        // at load time; shaders that don't reference them get pass-through `replace`.
        let sg_size_str = sg_size.to_string();
        let max_chunks = opts.max_seq.div_ceil(ATTN_CHUNK_SIZE);
        let max_chunks_str = max_chunks.to_string();

        // Pre-flight: check the attention_merge LDS budget before shader compile.
        // weights_sh needs MAX_CHUNKS f32 slots; sg_partial needs N_SG f32 slots.
        let merge_lds_bytes = 4 * u64::from(max_chunks) + 4 * (64 / u64::from(sg_size));
        if merge_lds_bytes > u64::from(device.limits().max_compute_workgroup_storage_size) {
            return Err(PotError::Config(
                "max_seq exceeds attention_merge LDS budget; reduce --max-seq",
            ));
        }

        macro_rules! load_shader {
            ($file:expr) => {{
                let src: &str = include_str!(concat!("shaders/", $file));
                let templated = src
                    .replace("{{SG_SIZE}}", &sg_size_str)
                    .replace("{{MAX_CHUNKS}}", &max_chunks_str);
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some($file),
                    source: wgpu::ShaderSource::Wgsl(templated.into()),
                })
            }};
        }
        let sh_embed = load_shader!("embed.wgsl");
        let sh_rms = load_shader!("rms_norm.wgsl");
        let sh_rope = load_shader!("rope_neox.wgsl");
        let sh_matvec = load_shader!("matvec_q1_0.wgsl");
        let sh_matvec_fused = load_shader!("matvec_q1_0_fused.wgsl");
        let sh_quant = load_shader!("quantize_q8_0.wgsl");
        let sh_matmul = load_shader!("matmul_q1_0_q8_0.wgsl");
        let sh_attn = load_shader!("attention.wgsl");
        let sh_attn_split = load_shader!("attention_split.wgsl");
        let sh_attn_merge = load_shader!("attention_merge.wgsl");
        let sh_silu = load_shader!("silu_mul.wgsl");
        let sh_topk = load_shader!("topk_reduce.wgsl");
        let sh_kv_writeback = load_shader!("kv_writeback.wgsl");

        let make_bgl =
            |label: &'static str, n_storage: u32, rw_mask: u32| -> wgpu::BindGroupLayout {
                let mut entries = vec![ubo_dyn(0)];
                for i in 0..n_storage {
                    let read_only = (rw_mask >> i) & 1 == 0;
                    entries.push(ssbo(i + 1, read_only));
                }
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(label),
                    entries: &entries,
                })
            };
        let bgls = BindGroupLayouts {
            embed: make_bgl("embed_bgl", 3, 0b010), // weights ro, act rw, sample ro
            rms_norm: make_bgl("rms_norm_bgl", 2, 0b01), // act rw, w ro
            rope: make_bgl("rope_bgl", 2, 0b10),    // rope_cs ro, act rw
            matvec: make_bgl("matvec_bgl", 2, 0b10), // weights ro, act rw
            quantize: make_bgl("quantize_bgl", 2, 0b10), // act ro, outbuf rw
            matmul: make_bgl("matmul_bgl", 3, 0b100), // weights ro, acts ro, y rw
            attn: make_bgl("attn_bgl", 3, 0b001),   // act rw, k ro, v ro
            attn_split: make_bgl("attn_split_bgl", 4, 0b1000), // act ro, k ro, v ro, partials rw
            attn_merge: make_bgl("attn_merge_bgl", 2, 0b01), // act rw, partials ro
            silu_mul: make_bgl("silu_mul_bgl", 1, 0b1), // act rw
            topk_reduce: make_bgl("topk_reduce_bgl", 2, 0b10), // logits ro, result rw
            kv_writeback: make_bgl("kv_writeback_bgl", 3, 0b110), // act ro, kv_k rw, kv_v rw
        };

        let mk_pipe = |layout: &wgpu::BindGroupLayout,
                       sh: &wgpu::ShaderModule,
                       label: &str|
         -> wgpu::ComputePipeline {
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{label}_pl")),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module: sh,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: pipeline_cache.as_ref().map(|(c, _, _)| c),
            })
        };
        let pipes = Pipelines {
            embed: mk_pipe(&bgls.embed, &sh_embed, "embed"),
            rms_norm: mk_pipe(&bgls.rms_norm, &sh_rms, "rms_norm"),
            rope_neox: mk_pipe(&bgls.rope, &sh_rope, "rope_neox"),
            matvec: mk_pipe(&bgls.matvec, &sh_matvec, "matvec"),
            matvec_fused: mk_pipe(&bgls.matvec, &sh_matvec_fused, "matvec_fused"),
            quantize: mk_pipe(&bgls.quantize, &sh_quant, "quantize"),
            matmul: mk_pipe(&bgls.matmul, &sh_matmul, "matmul"),
            attention: mk_pipe(&bgls.attn, &sh_attn, "attention"),
            attention_split: mk_pipe(&bgls.attn_split, &sh_attn_split, "attention_split"),
            attention_merge: mk_pipe(&bgls.attn_merge, &sh_attn_merge, "attention_merge"),
            silu_mul: mk_pipe(&bgls.silu_mul, &sh_silu, "silu_mul"),
            topk_reduce: mk_pipe(&bgls.topk_reduce, &sh_topk, "topk_reduce"),
            kv_writeback: mk_pipe(&bgls.kv_writeback, &sh_kv_writeback, "kv_writeback"),
        };

        // ---- persist pipeline cache -----------------------------------------
        if let Some((cache, path, prior)) = &pipeline_cache
            && let Some(data) = cache.get_data()
            && prior.as_deref() != Some(data.as_slice())
            && let Err(e) = write(path, &data)
        {
            log::warn!("failed to write pipeline cache to {}: {e}", path.display());
        }

        // ---- vocab ----------------------------------------------------------
        let vocab_path = model_dir.join("vocab.bin");
        let offs_path = model_dir.join("vocab_offsets.bin");
        let vocab_bytes = read(&vocab_path).map_err(|e| PotError::Io {
            path: vocab_path,
            source: e,
        })?;
        let offs_bytes = read(&offs_path).map_err(|e| PotError::Io {
            path: offs_path,
            source: e,
        })?;
        let offs: &[u32] = cast_slice(&offs_bytes);
        if offs.len() as u32 != cfg.n_vocab + 1 {
            return Err(PotError::Vocab("offsets length doesn't match n_vocab + 1"));
        }
        let mut vocab = Vec::with_capacity(cfg.n_vocab as usize);
        for i in 0..cfg.n_vocab as usize {
            let s = from_utf8(&vocab_bytes[offs[i] as usize..offs[i + 1] as usize])
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
                let act_elem_bytes = size_of::<half::f16>() as u64;
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
            let act_elem_bytes = size_of::<half::f16>() as u64;
            OutputTensors {
                token_embd_d: te.d_offset as u32,
                token_embd_qs: te.qs_offset as u32,
                lm_head_d: lm.d_offset as u32,
                lm_head_qs: lm.qs_offset as u32,
                output_norm_off: (on.offset / act_elem_bytes) as u32,
            }
        };

        // ---- build cached bind groups --------------------------------------
        // The UBO binding is sized to one full uniform slot (256 B). Every
        // params struct is <= 64 B and is packed into a 256-B-aligned slot, so
        // this size is correct for every dispatch that reuses the cached BG.
        let cached = build_cached_bind_groups(&device, &bgls, &buffers);

        Ok(Self {
            device,
            queue,
            cfg,
            public_cfg,
            act_layout,
            m_max: M_MAX,
            max_seq: opts.max_seq,
            buffers,
            pipes,
            cached,
            layer_tensors,
            output_tensors,
            vocab,
            lost,
        })
    }

    /// Read-only view of the model's hyperparameters.
    #[must_use]
    pub const fn config(&self) -> &ModelConfig {
        &self.public_cfg
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

    /// Open a fresh inference session. Cheap; does no GPU work.
    #[must_use]
    pub const fn new_session(&self) -> crate::Session<'_> {
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
        // Poll to flush the device-lost callback into the OnceLock. The callback
        // is queued by destroy() but only fires during poll().
        let _ = self.device.poll(wgpu::PollType::Poll);
    }
}

/// Precompute cos/sin table for NEOX rope: per position p (`0..max_seq`),
/// per j (`0..head_dim/2`), interleaved (cos, sin) pairs => `head_dim` floats per pos.
fn build_rope_table(cfg: &Config, max_seq: u32) -> Vec<f32> {
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

// Dispatch a one-thread-writes-the-builtin probe shader to find out what
// subgroup size the driver actually picks for our compute pipelines.
fn probe_subgroup_size(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    lost: &OnceLock<DeviceLostInfo>,
) -> Result<u32> {
    use core::result::Result as StdResult;
    type MapResult = StdResult<(), wgpu::BufferAsyncError>;
    let result = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe_result"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe_readback"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("probe_subgroup.wgsl"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/probe_subgroup.wgsl").into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("probe_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("probe_pl"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("probe_pipe"),
        layout: Some(&pl),
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("probe_bg"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: result.as_entire_binding(),
        }],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("probe_enc"),
    });
    {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("probe_cp"),
            timestamp_writes: None,
        });
        cp.set_pipeline(&pipe);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups(1, 1, 1);
    }
    enc.copy_buffer_to_buffer(&result, 0, &readback, 0, 4);
    queue.submit(Some(enc.finish()));

    let slice = readback.slice(..);
    let slot: Arc<OnceLock<MapResult>> = Arc::new(OnceLock::new());
    let slot2 = slot.clone();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = slot2.set(r);
    });
    if let Err(e) = device.poll(wgpu::PollType::wait_indefinitely()) {
        if let Some(d) = lost.get() {
            return Err(PotError::DeviceLost {
                reason: d.reason,
                message: d.message.clone(),
            });
        }
        return Err(PotError::Poll(e));
    }
    slot.get()
        .ok_or(PotError::Config("probe map_async failed"))?
        .as_ref()
        .map_err(|_| PotError::Config("probe map_async errored"))?;
    let sg = {
        let data = slice.get_mapped_range();
        #[allow(
            clippy::unwrap_used,
            reason = "data[..4] is always 4 bytes, so try_into to [u8; 4] is infallible"
        )]
        let bytes: [u8; 4] = data[..4].try_into().unwrap();
        u32::from_le_bytes(bytes)
    };
    readback.unmap();
    Ok(sg)
}

// ----- public(crate) helpers used by forward.rs -----------------------------

#[allow(
    clippy::panic,
    reason = "manifest is fully validated at load; missing tensor is a programmer error"
)]
pub fn tensor<'a>(cfg: &'a Config, name: &str) -> &'a TensorEntry {
    cfg.manifest
        .get(name)
        .unwrap_or_else(|| panic!("missing tensor in manifest: {name}"))
}

/// Build the full set of cached bind groups in one go. Called once at load.
fn build_cached_bind_groups(
    device: &wgpu::Device,
    bgls: &BindGroupLayouts,
    buffers: &Buffers,
) -> CachedBindGroups {
    // The UBO binding always uses UNIFORM_SLOT_SIZE — every params struct
    // fits in one slot, and the dynamic offset selects which slot.
    #[allow(
        clippy::unwrap_used,
        reason = "UNIFORM_SLOT_SIZE is a non-zero compile-time constant"
    )]
    let ubo_size = NonZeroU64::new(UNIFORM_SLOT_SIZE).unwrap();
    let ubo = || {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &buffers.uniform,
            offset: 0,
            size: Some(ubo_size),
        })
    };
    let mk = |label: &str,
              layout: &wgpu::BindGroupLayout,
              storages: &[&wgpu::Buffer]|
     -> wgpu::BindGroup {
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: ubo(),
        }];
        for (i, b) in storages.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: i as u32 + 1,
                resource: b.as_entire_binding(),
            });
        }
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &entries,
        })
    };
    CachedBindGroups {
        embed: mk(
            "cached_embed",
            &bgls.embed,
            &[&buffers.w_embed, &buffers.act, &buffers.sample],
        ),
        rms_norm: mk(
            "cached_rms_norm",
            &bgls.rms_norm,
            &[&buffers.act, &buffers.w_norms],
        ),
        rope: mk(
            "cached_rope",
            &bgls.rope,
            &[&buffers.rope_table, &buffers.act],
        ),
        matvec_w_attn: mk(
            "cached_matvec_attn",
            &bgls.matvec,
            &[&buffers.w_attn, &buffers.act],
        ),
        matvec_w_ffn_gu: mk(
            "cached_matvec_ffngu",
            &bgls.matvec,
            &[&buffers.w_ffn_gu, &buffers.act],
        ),
        matvec_w_ffn_d: mk(
            "cached_matvec_ffnd",
            &bgls.matvec,
            &[&buffers.w_ffn_d, &buffers.act],
        ),
        matvec_w_embed: mk(
            "cached_matvec_embed",
            &bgls.matvec,
            &[&buffers.w_embed, &buffers.act],
        ),
        quantize: mk(
            "cached_quantize",
            &bgls.quantize,
            &[&buffers.act, &buffers.act_q8],
        ),
        matmul_w_attn: mk(
            "cached_matmul_attn",
            &bgls.matmul,
            &[&buffers.w_attn, &buffers.act_q8, &buffers.act],
        ),
        matmul_w_ffn_gu: mk(
            "cached_matmul_ffngu",
            &bgls.matmul,
            &[&buffers.w_ffn_gu, &buffers.act_q8, &buffers.act],
        ),
        matmul_w_ffn_d: mk(
            "cached_matmul_ffnd",
            &bgls.matmul,
            &[&buffers.w_ffn_d, &buffers.act_q8, &buffers.act],
        ),
        matmul_w_embed: mk(
            "cached_matmul_embed",
            &bgls.matmul,
            &[&buffers.w_embed, &buffers.act_q8, &buffers.act],
        ),
        attn: mk(
            "cached_attn",
            &bgls.attn,
            &[&buffers.act, &buffers.kv_k, &buffers.kv_v],
        ),
        attn_split: mk(
            "cached_attn_split",
            &bgls.attn_split,
            &[
                &buffers.act,
                &buffers.kv_k,
                &buffers.kv_v,
                &buffers.attn_partials,
            ],
        ),
        attn_merge: mk(
            "cached_attn_merge",
            &bgls.attn_merge,
            &[&buffers.act, &buffers.attn_partials],
        ),
        silu_mul: mk("cached_silu_mul", &bgls.silu_mul, &[&buffers.act]),
        topk_reduce: mk(
            "cached_topk_reduce",
            &bgls.topk_reduce,
            &[&buffers.act, &buffers.sample],
        ),
        kv_writeback: mk(
            "cached_kv_writeback",
            &bgls.kv_writeback,
            &[&buffers.act, &buffers.kv_k, &buffers.kv_v],
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bonsai4b_cfg() -> ConfigRaw {
        ConfigRaw {
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
            padding_token_id: 151_654,
            add_bos: false,
            context_length: 32_768,
            rope_orig_context: 4_096,
            n_kv_groups: 4,
            q_dim: 4_096,
            kv_dim: 1_024,
            tied_embeddings: true,
            manifest: HashMap::new(),
        }
    }

    fn rope_test_cfg() -> ConfigRaw {
        ConfigRaw {
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
            padding_token_id: 0,
            add_bos: false,
            context_length: 4,
            rope_orig_context: 4,
            n_kv_groups: 1,
            q_dim: 8,
            kv_dim: 8,
            tied_embeddings: false,
            manifest: HashMap::new(),
        }
    }

    #[test]
    fn topk_max_is_thirty_two() {
        assert_eq!(TOPK_MAX, 32);
    }

    #[test]
    fn act_layout_offsets_monotonic() {
        let cfg = bonsai4b_cfg();
        let m = ActLayout::build(&cfg, 512);
        // Each region starts where the previous one ended.
        assert_eq!(m.x, 0);
        assert_eq!(m.x_norm, 512 * cfg.n_embd);
        assert_eq!(m.q, m.x_norm + 512 * cfg.n_embd);
        assert_eq!(m.k_cur, m.q + 512 * cfg.q_dim);
        assert_eq!(m.v_cur, m.k_cur + 512 * cfg.kv_dim);
        assert_eq!(m.attn_out, m.v_cur + 512 * cfg.kv_dim);
        assert_eq!(m.gate, m.attn_out + 512 * cfg.q_dim);
        assert_eq!(m.up, m.gate + 512 * cfg.n_ff);
        assert_eq!(m.ffn_in, m.up + 512 * cfg.n_ff);
        assert_eq!(m.logits, m.ffn_in + 512 * cfg.n_ff);
        assert_eq!(m.total_elems, m.logits + cfg.n_vocab);
        // Regions are strictly ordered.
        assert!(
            m.x < m.x_norm
                && m.x_norm < m.q
                && m.q < m.k_cur
                && m.k_cur < m.v_cur
                && m.v_cur < m.attn_out
                && m.attn_out < m.gate
                && m.gate < m.up
                && m.up < m.ffn_in
                && m.ffn_in < m.logits
        );
    }

    #[test]
    fn params_struct_sizes_fit_uniform_slot() {
        let limit = UNIFORM_SLOT_SIZE as usize;
        assert!(size_of::<EmbedParams>() <= limit);
        assert!(size_of::<RmsNormParams>() <= limit);
        assert!(size_of::<RopeParams>() <= limit);
        assert!(size_of::<MatvecParams>() <= limit);
        assert!(size_of::<MatvecFusedParams>() <= limit);
        assert!(size_of::<QuantParams>() <= limit);
        assert!(size_of::<MatmulParams>() <= limit);
        assert!(size_of::<AttnParams>() <= limit);
        assert!(size_of::<AttnSplitParams>() <= limit);
        assert!(size_of::<AttnMergeParams>() <= limit);
        assert!(size_of::<SiluMulParams>() <= limit);
        assert!(size_of::<TopKParams>() <= limit);
        assert!(size_of::<KvWritebackParams>() <= limit);
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
