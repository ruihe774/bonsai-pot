//! Model loading and GPU resource setup: device creation, weight upload,
//! pipelines, the universal bind-group layout, and the activation / Q8 /
//! state / KV-cache layouts shared by every session.

use alloc::string::{String, ToString as _};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::iter::once;
use core::pin::pin;
use core::str::from_utf8;
use core::task::{Context, Poll, Waker};
#[cfg(feature = "std")]
use std::env::var_os;
#[cfg(feature = "std")]
use std::fs::{read, read_to_string};
#[cfg(feature = "std")]
use std::path::Path;

use bytemuck::cast_slice;
use once_cell::sync::OnceCell;
use wgpu::{Backends, InstanceDescriptor};

use crate::config::{self, LayerTensors, ModelConfig, OutputTensors};
use crate::decode;
use crate::error::{PotError, Result};
use crate::params::IMMEDIATE_BYTES;

// ----- snapshot & options ---------------------------------------------------

/// Pre-loaded model weights and metadata.
///
/// Pass to [`Model::load_from_snapshot`] to initialise the engine without any
/// filesystem access, or obtain one with [`ModelSnapshot::from_dir`].
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
    /// Contents of `weights_embed_lmhead.bin`.
    pub w_embed_lmhead: Vec<u8>,
    /// Contents of `weights_small.bin`.
    pub w_small: Vec<u8>,
    /// Contents of `vocab.bin`.
    pub vocab_bytes: Vec<u8>,
    /// Contents of `vocab_offsets.bin` (little-endian `u32` stream).
    pub vocab_offsets: Vec<u8>,
    /// Ids from `added_tokens.txt` (CONTROL / `USER_DEFINED` tokens that
    /// tokenizers must match atomically).
    pub added_tokens: Vec<u32>,
}

impl ModelSnapshot {
    /// Read the model files from `model_dir` and parse `config.ini`.
    ///
    /// # Errors
    ///
    /// Returns `PotError::Io` for IO errors and `PotError::Config` for an
    /// invalid config or `added_tokens.txt`.
    #[cfg(feature = "std")]
    pub fn from_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let read_text = |fname: &str| -> Result<String> {
            let p = model_dir.join(fname);
            read_to_string(&p).map_err(|e| PotError::Io { path: p, source: e })
        };
        let load = |fname: &str| -> Result<Vec<u8>> {
            let p = model_dir.join(fname);
            read(&p).map_err(|e| PotError::Io { path: p, source: e })
        };
        let config = ModelConfig::from_ini(&read_text("config.ini")?)?;
        let added_tokens = read_text("added_tokens.txt")?
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                l.trim()
                    .parse::<u32>()
                    .map_err(|_| PotError::Config("added_tokens.txt: invalid id"))
            })
            .collect::<Result<Vec<u32>>>()?;
        Ok(Self {
            config,
            w_attn: load("weights_attn.bin")?,
            w_ffn_gate_up: load("weights_ffn_gate_up.bin")?,
            w_ffn_down: load("weights_ffn_down.bin")?,
            w_embed_lmhead: load("weights_embed_lmhead.bin")?,
            w_small: load("weights_small.bin")?,
            vocab_bytes: load("vocab.bin")?,
            vocab_offsets: load("vocab_offsets.bin")?,
            added_tokens,
        })
    }
}

/// System-wide GPU queue scheduling priority, passed via `VK_KHR_global_priority`.
pub use ash::vk::QueueGlobalPriorityKHR as GlobalPriority;
/// Power preference when choosing a physical adapter.
pub use wgpu::PowerPreference;

fn global_priority_fallback_to_queue_prio(priority: GlobalPriority) -> f32 {
    const LOWEST: i32 = GlobalPriority::LOW.as_raw();
    const HIGHEST: i32 = GlobalPriority::HIGH.as_raw();
    (priority.as_raw().clamp(LOWEST, HIGHEST) - LOWEST) as f32 / (HIGHEST - LOWEST) as f32
}

/// Allocate-time tunables for [`Model::load_from_dir`].
#[derive(Debug, Copy, Clone)]
pub struct LoadOptions {
    /// Maximum sequence length (KV-cache positions). Default: 32768.
    ///
    /// The `Q8_0` KV cache of the full-attention layers costs
    /// `n_attn_layers * kv_dim * 2.25` bytes per position (36 KiB for
    /// Bonsai-2-27B); K and V each live in one buffer, so `max_seq` is capped
    /// at the adapter's per-buffer limit and at the model's `context_length`.
    /// The wgpu fork reports the driver's limit (4 GiB − 4 on Mesa/RADV, i.e.
    /// about 233K positions); an adapter capped at 2 GiB − 1 would allow
    /// about 116K.
    pub max_seq: u32,
    /// System-wide GPU scheduling priority via `VK_KHR_global_priority`.
    /// Default [`GlobalPriority::LOW`]; ignored if unsupported.
    pub priority: GlobalPriority,
    /// Power preference when choosing a physical adapter. Default
    /// [`PowerPreference::HighPerformance`].
    pub power_perference: PowerPreference,
}

pub const DEFAULT_MAX_SEQ: u32 = 32768;

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            max_seq: DEFAULT_MAX_SEQ,
            priority: GlobalPriority::LOW,
            power_perference: PowerPreference::HighPerformance,
        }
    }
}

// ----- constants & layouts --------------------------------------------------

/// Largest number of tokens processed by one batched prefill chunk.
pub const M_MAX: u32 = 512;
/// Largest top-K the GPU sampler returns.
pub const TOPK_MAX: u32 = 32;
/// Pass-1 workgroups of the top-K reduction.
pub const TOPK_NUM_PARTIAL_WG: u32 = 32;
/// Upper bound on split-K attention chunks (sizes `attn_partials`).
pub const ATTN_MAX_CHUNKS: u32 = 64;
/// Smallest split-K attention chunk.
pub const ATTN_MIN_CHUNK: u32 = 32;
/// Staging bytes for token-id uploads (`M_MAX` u32).
pub const STAGING_CHUNK: u64 = (M_MAX as u64 * 4).next_power_of_two();
/// Timestamp query slots for bench/microbench.
#[cfg(all(feature = "bench-internals", not(feature = "ci")))]
pub const BENCH_QS_SLOTS: u32 = 2048;

/// f32 element offsets of the regions of the session activation buffer, each
/// sized for `M_MAX` token rows.
#[derive(Copy, Clone, Debug)]
pub struct ActLayout {
    pub x: u32,
    pub xn: u32,
    /// GDN qkv projection (stride `ssm_conv_dim`) / attention [q|gate]
    /// projection (stride `2 * q_dim`).
    pub qkv: u32,
    pub z: u32,
    pub k: u32,
    pub v: u32,
    pub conv: u32,
    pub gates: u32,
    pub o: u32,
    pub gate: u32,
    pub up: u32,
    pub logits: u32,
    pub partials: u32,
    pub total: u32,
}

impl ActLayout {
    pub fn build(cfg: &ModelConfig) -> Self {
        let mut o = 0u32;
        let mut alloc = |n: u32| -> u32 {
            let r = o;
            o += n.div_ceil(4) * 4;
            r
        };
        let m = M_MAX;
        let x = alloc(m * cfg.n_embd);
        let xn = alloc(m * cfg.n_embd);
        let qkv = alloc(m * cfg.ssm_conv_dim().max(2 * cfg.q_dim()));
        let z = alloc(m * cfg.ssm_inner);
        let k = alloc(m * cfg.kv_dim());
        let v = alloc(m * cfg.kv_dim());
        let conv = alloc(m * cfg.ssm_conv_dim());
        let gates = alloc(m * 2 * cfg.ssm_n_v_heads);
        let o_ = alloc(m * cfg.ssm_inner.max(cfg.q_dim()));
        let gate = alloc(m * cfg.n_ff);
        let up = alloc(m * cfg.n_ff);
        let logits = alloc(cfg.n_vocab);
        let partials = alloc(cfg.n_head * ATTN_MAX_CHUNKS * (cfg.head_dim + 2));
        Self {
            x,
            xn,
            qkv,
            z,
            k,
            v,
            conv,
            gates,
            o: o_,
            gate,
            up,
            logits,
            partials,
            total: o,
        }
    }
}

/// Word offsets of the three sections of the Q8 activation buffer. Each
/// section is sized for `M_MAX` rows of the widest projection input.
#[derive(Copy, Clone, Debug)]
pub struct Q8Layout {
    pub qs: u32,
    pub d: u32,
    pub s: u32,
    pub total_words: u32,
}

impl Q8Layout {
    pub fn build(cfg: &ModelConfig) -> Self {
        let k_max = cfg.n_embd.max(cfg.n_ff).max(cfg.q_dim()).max(cfg.ssm_inner);
        let qs_words = M_MAX * k_max / 4;
        let blocks = M_MAX * k_max / 128;
        Self {
            qs: 0,
            d: qs_words,
            s: qs_words + blocks,
            total_words: qs_words + 2 * blocks,
        }
    }
}

/// f32 offsets of the recurrent state of each Gated-DeltaNet layer.
#[derive(Copy, Clone, Debug)]
pub struct StateLayout {
    conv_per_layer: u32,
    ssm_base: u32,
    ssm_per_layer: u32,
    pub total: u64,
}

impl StateLayout {
    pub fn build(cfg: &ModelConfig) -> Self {
        let n = cfg.n_gdn_layers();
        let conv_per_layer = (cfg.ssm_conv_kernel - 1) * cfg.ssm_conv_dim();
        let ssm_per_layer = cfg.ssm_n_v_heads * cfg.ssm_head_dim * cfg.ssm_head_dim;
        Self {
            conv_per_layer,
            ssm_base: n * conv_per_layer,
            ssm_per_layer,
            total: u64::from(n) * u64::from(conv_per_layer + ssm_per_layer),
        }
    }
    pub const fn conv(&self, idx: u32) -> u32 {
        idx * self.conv_per_layer
    }
    pub const fn ssm(&self, idx: u32) -> u32 {
        self.ssm_base + idx * self.ssm_per_layer
    }
}

/// Byte layout of one `Q8_0` KV-cache buffer (`kv_k` or `kv_v`): an f32 scale
/// section (one per 32 elements) followed by an i8 qs section.
#[derive(Copy, Clone, Debug)]
pub struct KvLayout {
    kv_dim: u32,
    max_seq: u32,
    n_layers: u32,
    pub total_bytes: u64,
}

impl KvLayout {
    pub const fn bytes_per_position(cfg: &ModelConfig, n_attn: u32) -> u64 {
        // per buffer: kv_dim i8 + kv_dim/32 f32 scales
        n_attn as u64 * (cfg.kv_dim() as u64 + cfg.kv_dim() as u64 / 8)
    }
    pub fn build(cfg: &ModelConfig, max_seq: u32) -> Self {
        let n = cfg.n_attn_layers();
        Self {
            kv_dim: cfg.kv_dim(),
            max_seq,
            n_layers: n,
            total_bytes: Self::bytes_per_position(cfg, n) * u64::from(max_seq),
        }
    }
    /// Word offset of the scale section of attention layer `idx`.
    pub const fn d_word(&self, idx: u32) -> u32 {
        idx * self.max_seq * (self.kv_dim / 32)
    }
    /// Byte offset of the qs section of attention layer `idx`.
    pub const fn qs_byte(&self, idx: u32) -> u32 {
        self.n_layers * self.max_seq * (self.kv_dim / 32) * 4 + idx * self.max_seq * self.kv_dim
    }
    /// `(byte offset, byte length)` of the scale rows `[0, pos)` of layer `idx`.
    pub const fn d_range(&self, idx: u32, pos: u32) -> (u64, u64) {
        (
            self.d_word(idx) as u64 * 4,
            pos as u64 * (self.kv_dim as u64 / 8),
        )
    }
    /// `(byte offset, byte length)` of the qs rows `[0, pos)` of layer `idx`.
    pub const fn qs_range(&self, idx: u32, pos: u32) -> (u64, u64) {
        (self.qs_byte(idx) as u64, pos as u64 * self.kv_dim as u64)
    }
}

// ----- GPU objects ----------------------------------------------------------

pub struct Pipelines {
    pub(crate) embed: wgpu::ComputePipeline,
    pub(crate) norm_rot_q8: wgpu::ComputePipeline,
    pub(crate) matvec: wgpu::ComputePipeline,
    pub(crate) matmul: wgpu::ComputePipeline,
    /// Cooperative-matrix matmul (int8 preferred, else f16) when the device
    /// supports 16×16×16 WMMA with a fixed subgroup size of 32, plus its
    /// token-tile width.
    pub(crate) matmul_cm: Option<(wgpu::ComputePipeline, u32)>,
    pub(crate) gdn_gates: wgpu::ComputePipeline,
    pub(crate) gdn_conv: wgpu::ComputePipeline,
    pub(crate) gdn_recurrence: wgpu::ComputePipeline,
    pub(crate) gdn_out_rot_q8: wgpu::ComputePipeline,
    pub(crate) attn_qk: wgpu::ComputePipeline,
    pub(crate) attn_split: wgpu::ComputePipeline,
    pub(crate) attn_prefill: wgpu::ComputePipeline,
    /// Cooperative-matrix (f16) flash attention for batched prefill, when the
    /// device supports 16×16×16 f16 WMMA with a fixed subgroup size of 32.
    pub(crate) attn_prefill_cm: Option<wgpu::ComputePipeline>,
    pub(crate) attn_out_rot_q8: wgpu::ComputePipeline,
    pub(crate) silu_mul_rot_q8: wgpu::ComputePipeline,
    pub(crate) topk_partial: wgpu::ComputePipeline,
    pub(crate) topk_merge: wgpu::ComputePipeline,
}

/// Immutable GPU buffers shared by every session.
pub struct Buffers {
    /// One buffer per weight window (indexed like `Model::windows`).
    pub(crate) weights: Vec<wgpu::Buffer>,
    pub(crate) w_small: wgpu::Buffer,
    pub(crate) rope: wgpu::Buffer,
}

/// Large weight files (each uploaded as one GPU buffer).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WeightFile {
    Attn = 0,
    FfnGU = 1,
    FfnD = 2,
    Embed = 3,
}

/// A slice of a weight file uploaded as its own GPU buffer. wgpu caps buffers
/// and storage bindings (2 GiB − 1 on Mesa), so files larger than that are
/// split into several windows at layer boundaries.
#[derive(Copy, Clone, Debug)]
pub struct WeightWindow {
    pub file: WeightFile,
    pub offset: u64,
    pub size: u64,
}

/// Split a file of `len` bytes into windows of at most `limit` bytes, cutting
/// only at the (ascending) candidate offsets in `cuts`.
fn plan_windows(len: u64, cuts: &[u64], limit: u64, align: u64) -> Result<Vec<(u64, u64)>> {
    let mut out = Vec::new();
    let mut start = 0u64;
    let mut prev = 0u64;
    for &c in cuts.iter().chain(once(&len)) {
        if c - start > limit {
            if prev == start {
                return Err(PotError::Config(
                    "weight tensor larger than the storage binding limit",
                ));
            }
            out.push((start, prev - start));
            start = prev;
            if c - start > limit {
                return Err(PotError::Config(
                    "weight tensor larger than the storage binding limit",
                ));
            }
        }
        prev = c;
    }
    out.push((start, len - start));
    if out.iter().any(|w| w.0 % align != 0) {
        return Err(PotError::Config(
            "weight window offset violates storage offset alignment",
        ));
    }
    Ok(out)
}

/// Latched state set by the wgpu device-lost or uncaptured-error callbacks.
#[derive(Debug, Clone)]
pub struct DeviceLostInfo {
    pub reason: wgpu::DeviceLostReason,
    pub message: String,
}

/// GPU-bearing handle to a loaded Ternary-Bonsai-2 model.
///
/// Holds the immutable GPU resources (weights, pipelines, bind-group layout,
/// `RoPE` table) and host-side bookkeeping. Per-conversation state (KV cache,
/// recurrent state, activations) lives on [`crate::Session`], so any number of
/// sessions may run against one `Model`.
pub struct Model {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) cfg: ModelConfig,
    pub(crate) act: ActLayout,
    pub(crate) q8: Q8Layout,
    pub(crate) state: StateLayout,
    pub(crate) kv: KvLayout,
    pub(crate) max_seq: u32,
    pub(crate) buffers: Buffers,
    pub(crate) pipes: Pipelines,
    pub(crate) bgl: wgpu::BindGroupLayout,
    /// Bindable weight windows; each session builds one bind group per window.
    pub(crate) windows: Vec<WeightWindow>,
    pub(crate) win_attn: usize,
    pub(crate) win_ffn_d: usize,
    pub(crate) win_embed: usize,
    pub(crate) layers: Vec<LayerTensors>,
    pub(crate) output: OutputTensors,
    pub(crate) vocab: Vec<String>,
    pub(crate) added_tokens: Vec<u32>,
    pub(crate) lost: Arc<OnceCell<DeviceLostInfo>>,
    #[cfg(all(feature = "bench-internals", not(feature = "ci")))]
    pub(crate) bench_ts_period_ns: f32,
}

/// Number of storage bindings in the universal bind-group layout.
pub const N_BINDINGS: u32 = 9;
/// Bindings that are read-only for every kernel (weights, small weights, rope).
const READ_ONLY_BINDINGS: [u32; 3] = [0, 1, 8];

impl Model {
    /// Load with default options.
    ///
    /// # Errors
    ///
    /// See [`Model::load_from_snapshot`].
    #[cfg(feature = "std")]
    #[doc(hidden)]
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        Self::load_from_dir(model_dir, LoadOptions::default())
    }

    /// Read the model directory and load it.
    ///
    /// # Errors
    ///
    /// See [`Model::load_from_snapshot`]; additionally IO errors.
    #[cfg(feature = "std")]
    pub fn load_from_dir(model_dir: impl AsRef<Path>, opts: LoadOptions) -> Result<Self> {
        Self::load_from_snapshot(ModelSnapshot::from_dir(model_dir)?, opts)
    }

    /// Create the device, upload weights, and build pipelines from `snapshot`.
    ///
    /// # Errors
    ///
    /// Returns an error if the config is invalid, `opts.max_seq` is zero or
    /// exceeds the supported maximum, no suitable Vulkan adapter/device is
    /// available, a required feature is missing, or the vocab is malformed.
    #[allow(clippy::too_many_lines, reason = "linear setup sequence")]
    pub fn load_from_snapshot(snapshot: ModelSnapshot, opts: LoadOptions) -> Result<Self> {
        let cfg = snapshot.config;
        config::validate(&cfg)?;
        let mut layers = config::resolve_layers(&cfg)?;
        let output = config::resolve_output(&cfg)?;
        if opts.max_seq == 0 {
            return Err(PotError::Config("max_seq must be > 0"));
        }
        if opts.max_seq > cfg.context_length {
            return Err(PotError::Config(
                "max_seq exceeds the model's context_length",
            ));
        }

        // ---- adapter ------------------------------------------------------
        let mut instance_desc = InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = Backends::VULKAN;
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
        let info = adapter.get_info();
        log::info!("adapter: {info:?}");

        let features = adapter.features();
        let mut required_features = wgpu::Features::empty();
        for (f, name) in [
            (wgpu::Features::SUBGROUP, "SUBGROUP"),
            (wgpu::Features::IMMEDIATES, "IMMEDIATES"),
            (wgpu::Features::PASSTHROUGH_SHADERS, "PASSTHROUGH_SHADERS"),
            (wgpu::Features::SHADER_F16, "SHADER_F16"),
            (wgpu::Features::SHADER_I16, "SHADER_I16"),
        ] {
            if !features.contains(f) {
                return Err(PotError::FeatureUnsupported(name));
            }
            required_features |= f;
        }
        #[cfg(all(feature = "bench-internals", not(feature = "ci")))]
        for (f, name) in [
            (wgpu::Features::TIMESTAMP_QUERY, "TIMESTAMP_QUERY"),
            (
                wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
                "TIMESTAMP_QUERY_INSIDE_PASSES",
            ),
        ] {
            if !features.contains(f) {
                return Err(PotError::FeatureUnsupported(name));
            }
            required_features |= f;
        }
        let size_control = features.contains(wgpu::Features::SUBGROUP_SIZE_CONTROL)
            && info.subgroup_min_size <= 32
            && info.subgroup_max_size >= 32;
        if size_control {
            required_features |= wgpu::Features::SUBGROUP_SIZE_CONTROL;
        }
        if info.subgroup_min_size < 4 {
            return Err(PotError::Config("adapter subgroup size must be >= 4"));
        }

        // ---- limits -------------------------------------------------------
        let alimits = adapter.limits();
        // Largest buffer that is bound whole (KV cache, activations, state).
        let max_buf = alimits
            .max_buffer_size
            .min(alimits.max_storage_buffer_binding_size)
            .min(u64::from(u32::MAX));

        let kv = KvLayout::build(&cfg, opts.max_seq);
        if kv.total_bytes > max_buf {
            return Err(PotError::Config(
                "KV cache exceeds the adapter's per-buffer limit; reduce max_seq",
            ));
        }
        let act = ActLayout::build(&cfg);
        let q8 = Q8Layout::build(&cfg);
        let state = StateLayout::build(&cfg);
        let rope_half = cfg.rope_dim / 2;
        let rope_bytes = u64::from(opts.max_seq) * u64::from(rope_half) * 8;
        let largest = [
            snapshot.w_small.len() as u64,
            kv.total_bytes,
            u64::from(act.total) * 4,
            u64::from(q8.total_words) * 4,
            state.total * 4,
            rope_bytes,
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        if largest > max_buf {
            return Err(PotError::Config(
                "a model buffer exceeds the adapter's per-buffer limit",
            ));
        }
        if alimits.max_storage_buffers_per_shader_stage < N_BINDINGS {
            return Err(PotError::Config(
                "adapter supports too few storage buffers per stage",
            ));
        }
        if alimits.max_compute_workgroup_storage_size < 32768 {
            return Err(PotError::Config(
                "adapter workgroup storage must be >= 32 KiB",
            ));
        }
        let limits = wgpu::Limits {
            max_buffer_size: max_buf,
            max_storage_buffer_binding_size: max_buf,
            min_storage_buffer_offset_alignment: alimits.min_storage_buffer_offset_alignment,
            max_storage_buffers_per_shader_stage: N_BINDINGS,
            max_immediate_size: IMMEDIATE_BYTES,
            max_compute_workgroup_storage_size: alimits
                .max_compute_workgroup_storage_size
                .min(65536),
            ..Default::default()
        };
        log::debug!("requesting limits: {limits:?}");
        let desc = wgpu::DeviceDescriptor {
            label: None,
            required_features,
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            experimental_features: wgpu::ExperimentalFeatures::default(),
            trace: wgpu::Trace::Off,
        };
        let choice = coopmat_choice_from_env();
        let want_coopmat = size_control
            && alimits.max_compute_workgroup_storage_size >= 61 * 1024
            && choice != CoopChoice::Off;
        let (device, queue, coop) = open_device(&adapter, &desc, opts.priority, want_coopmat)?;
        let use_i8 = coop.i8 && choice != CoopChoice::F16;
        let use_f16 = !use_i8 && coop.f16 && choice != CoopChoice::I8;
        log::info!("cooperative-matrix matmul: i8={use_i8} f16={use_f16} (device: {coop:?})");

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

        // ---- weights ------------------------------------------------------
        let rope = build_rope_table(&cfg, opts.max_seq);
        // ---- weight windows ---------------------------------------------
        let bind_limit = max_buf;
        let align = 4u64;
        let file_lens = [
            snapshot.w_attn.len() as u64,
            snapshot.w_ffn_gate_up.len() as u64,
            snapshot.w_ffn_down.len() as u64,
            snapshot.w_embed_lmhead.len() as u64,
        ];
        let files = [
            WeightFile::Attn,
            WeightFile::FfnGU,
            WeightFile::FfnD,
            WeightFile::Embed,
        ];
        let mut windows = Vec::new();
        let mut first_win = [0usize; 4];
        for (fi, &file) in files.iter().enumerate() {
            // Cut candidates: layer starts of the gate/up file; none elsewhere.
            let cuts: Vec<u64> = if file == WeightFile::FfnGU {
                layers.iter().map(|l| u64::from(l.ffn_gate) * 4).collect()
            } else {
                Vec::new()
            };
            first_win[fi] = windows.len();
            for (offset, size) in plan_windows(file_lens[fi], &cuts, bind_limit, align)? {
                windows.push(WeightWindow { file, offset, size });
            }
        }
        for &file in &files {
            let n = windows
                .iter()
                .filter(|w: &&WeightWindow| w.file == file)
                .count();
            if file != WeightFile::FfnGU && n != 1 {
                return Err(PotError::Config(
                    "weight file exceeds the storage binding limit",
                ));
            }
        }
        for l in &mut layers {
            let byte = u64::from(l.ffn_gate) * 4;
            let (wi, w) = windows
                .iter()
                .enumerate()
                .find(|(_, w)| {
                    w.file == WeightFile::FfnGU && w.offset <= byte && byte < w.offset + w.size
                })
                .ok_or(PotError::Config("ffn_gate outside every weight window"))?;
            let up_end =
                u64::from(l.ffn_up) * 4 + u64::from(cfg.n_ff) * u64::from(cfg.n_embd / 128) * 28;
            if up_end > w.offset + w.size {
                return Err(PotError::Config("ffn_gate/ffn_up straddle a weight window"));
            }
            let base = (w.offset / 4) as u32;
            l.gu_win = wi;
            l.ffn_gate -= base;
            l.ffn_up -= base;
        }
        log::info!("weight windows: {windows:?}");

        let file_bytes: [&[u8]; 4] = [
            &snapshot.w_attn,
            &snapshot.w_ffn_gate_up,
            &snapshot.w_ffn_down,
            &snapshot.w_embed_lmhead,
        ];
        let rope_bytes: &[u8] = cast_slice(&rope);
        let max_upload = windows
            .iter()
            .map(|w| w.size)
            .chain([snapshot.w_small.len() as u64, rope_bytes.len() as u64])
            .max()
            .unwrap_or(0);
        let up = Uploader::new(&device, &queue, &lost, max_upload);
        let weights = windows
            .iter()
            .map(|w| {
                let bytes =
                    &file_bytes[w.file as usize][w.offset as usize..(w.offset + w.size) as usize];
                up.upload("weights", bytes)
            })
            .collect::<Result<Vec<_>>>()?;
        let buffers = Buffers {
            weights,
            w_small: up.upload("w_small", &snapshot.w_small)?,
            rope: up.upload("rope", rope_bytes)?,
        };
        drop(up);
        drop(snapshot.w_attn);
        drop(snapshot.w_ffn_gate_up);
        drop(snapshot.w_ffn_down);
        drop(snapshot.w_embed_lmhead);

        // ---- layout & pipelines --------------------------------------------
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..N_BINDINGS)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: READ_ONLY_BINDINGS.contains(&binding),
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("universal_bgl"),
            entries: &entries,
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("universal_pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: IMMEDIATE_BYTES,
        });
        macro_rules! pipeline {
            ($name:literal) => {
                pipeline!($name, wgpu::SubgroupSize::Varying)
            };
            ($name:literal, $sg:expr) => {{
                const SPV: &[u8] =
                    include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".comp.spv"));
                let words: Vec<u32> = bytemuck::pod_collect_to_vec(SPV);
                // SAFETY: the SPIR-V is produced by build.rs from our own GLSL
                // and declares only the universal bind-group layout bindings.
                let module = unsafe {
                    device.create_shader_module_passthrough(
                        wgpu::ShaderModuleDescriptorPassthrough {
                            label: Some($name),
                            spirv: Some(alloc::borrow::Cow::Owned(words)),
                            ..Default::default()
                        },
                    )
                };
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some($name),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        zero_initialize_workgroup_memory: false,
                        subgroup_size: $sg,
                        ..Default::default()
                    },
                    cache: None,
                })
            }};
        }
        let pipes = Pipelines {
            embed: pipeline!("embed"),
            norm_rot_q8: pipeline!("norm_rot_q8"),
            matvec: pipeline!("matvec_ptq1_0"),
            matmul: pipeline!("matmul_ptq1_0"),
            matmul_cm: if use_i8 {
                Some((
                    pipeline!("matmul_ptq1_0_cmi8", wgpu::SubgroupSize::Fixed(32)),
                    128,
                ))
            } else if use_f16 {
                Some((
                    pipeline!("matmul_ptq1_0_cm", wgpu::SubgroupSize::Fixed(32)),
                    128,
                ))
            } else {
                None
            },
            gdn_gates: pipeline!("gdn_gates"),
            gdn_conv: pipeline!("gdn_conv"),
            gdn_recurrence: pipeline!("gdn_recurrence"),
            gdn_out_rot_q8: pipeline!("gdn_out_rot_q8"),
            attn_qk: pipeline!("attn_qk_norm_rope_kv"),
            attn_split: pipeline!("attn_split"),
            attn_prefill: pipeline!("attn_prefill"),
            attn_prefill_cm: if coop.f16 && attn_coopmat_from_env() {
                Some(pipeline!("attn_prefill_cm", wgpu::SubgroupSize::Fixed(32)))
            } else {
                None
            },
            attn_out_rot_q8: pipeline!("attn_out_rot_q8"),
            silu_mul_rot_q8: pipeline!("silu_mul_rot_q8"),
            topk_partial: pipeline!("topk_partial"),
            topk_merge: pipeline!("topk_merge"),
        };

        // ---- vocab ----------------------------------------------------------
        let offs: Vec<u32> = snapshot
            .vocab_offsets
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if offs.len() as u32 != cfg.n_vocab + 1 {
            return Err(PotError::Vocab("offsets length doesn't match n_vocab + 1"));
        }
        let mut vocab = Vec::with_capacity(cfg.n_vocab as usize);
        for i in 0..cfg.n_vocab as usize {
            let (a, b) = (offs[i] as usize, offs[i + 1] as usize);
            if a > b || b > snapshot.vocab_bytes.len() {
                return Err(PotError::Vocab("offset out of range"));
            }
            vocab.push(
                from_utf8(&snapshot.vocab_bytes[a..b])
                    .unwrap_or("?")
                    .to_string(),
            );
        }
        if snapshot.added_tokens.iter().any(|&t| t >= cfg.n_vocab) {
            return Err(PotError::Config("added_tokens.txt: id out of range"));
        }

        #[cfg(all(feature = "bench-internals", not(feature = "ci")))]
        let bench_ts_period_ns = queue.get_timestamp_period();

        Ok(Self {
            device,
            queue,
            act,
            q8,
            state,
            kv,
            max_seq: opts.max_seq,
            buffers,
            pipes,
            bgl,
            windows,
            win_attn: first_win[WeightFile::Attn as usize],
            win_ffn_d: first_win[WeightFile::FfnD as usize],
            win_embed: first_win[WeightFile::Embed as usize],
            layers,
            output,
            vocab,
            added_tokens: snapshot.added_tokens,
            lost,
            cfg,
            #[cfg(all(feature = "bench-internals", not(feature = "ci")))]
            bench_ts_period_ns,
        })
    }

    /// Hyper-parameters and manifest of the loaded model.
    #[must_use]
    pub const fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    /// Maximum sequence length supported by the allocated KV cache.
    #[must_use]
    pub const fn max_seq_len(&self) -> u32 {
        self.max_seq
    }

    /// Maximum tokens per batched prefill dispatch (longer prompts are chunked).
    #[must_use]
    pub const fn max_prefill_tokens(&self) -> u32 {
        M_MAX
    }

    /// Open a fresh inference session (allocates its KV cache, recurrent
    /// state, and activation scratch).
    #[must_use]
    pub fn new_session(&self) -> crate::Session<'_> {
        crate::Session::new(self)
    }

    /// Decode a single token id to its raw bytes (inverting the GPT-2
    /// byte-level vocab encoding). Added tokens decode to their literal text.
    #[must_use]
    pub fn decode_token(&self, id: u32) -> Vec<u8> {
        let s = self.vocab.get(id as usize).map_or("", String::as_str);
        if self.added_tokens.binary_search(&id).is_ok() {
            return s.as_bytes().to_vec();
        }
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

    /// The raw vocab string for a token id (GPT-2 byte-encoded form).
    #[must_use]
    pub fn vocab_token(&self, id: u32) -> Option<&str> {
        self.vocab.get(id as usize).map(String::as_str)
    }

    /// Reverse lookup of a raw vocab string (linear scan; startup use only).
    #[must_use]
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.vocab.iter().position(|s| s == token).map(|i| i as u32)
    }

    /// Ids of the added (CONTROL / `USER_DEFINED`) tokens, ascending. Tokenizers
    /// must match their literal text atomically before BPE.
    #[must_use]
    pub fn added_tokens(&self) -> &[u32] {
        &self.added_tokens
    }

    /// Default stop tokens: the model EOS plus `<|endoftext|>` when present.
    #[must_use]
    pub fn stop_tokens(&self) -> Vec<u32> {
        let mut v = vec![self.cfg.eos_token_id];
        if let Some(t) = self.token_id("<|endoftext|>") {
            if t != self.cfg.eos_token_id {
                v.push(t);
            }
        }
        v
    }

    /// Returns `true` if the underlying wgpu device has been lost. A lost
    /// model is permanently unusable; drop it and load again.
    #[must_use]
    pub fn is_device_lost(&self) -> bool {
        self.lost.get().is_some()
    }

    pub(crate) fn check_device(&self) -> Result<()> {
        if let Some(d) = self.lost.get() {
            return Err(PotError::DeviceLost {
                reason: d.reason,
                message: d.message.clone(),
            });
        }
        Ok(())
    }

    /// Calls `Device::destroy()` to simulate device loss (tests only).
    #[doc(hidden)]
    pub fn __destroy_device_for_test(&self) {
        self.device.destroy();
        let _ = self.device.poll(wgpu::PollType::Poll);
    }

    /// Binding resource of weight window `win`.
    pub(crate) fn window_binding(&self, win: usize) -> wgpu::BindingResource<'_> {
        self.buffers.weights[win].as_entire_binding()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum CoopChoice {
    Auto,
    I8,
    F16,
    Off,
}

/// Largest staging chunk used for uploads (see [`Uploader`]).
const UPLOAD_CHUNK_BYTES: u64 = 64 << 20;

/// Uploads host data to the GPU through one reusable staging buffer of at
/// most [`UPLOAD_CHUNK_BYTES`], waiting for each chunk's copy before reusing
/// it. Used for the weights at load time and for snapshot restore.
///
/// `create_buffer_init` stages every buffer through its own equally large
/// host-visible allocation, which with resizable BAR lives in VRAM. Loading the
/// weights then briefly needed about twice their size, and a session created
/// right after found VRAM full: the kernel placed its KV cache in system
/// memory and never moved it back, halving attention speed. With one bounded
/// staging buffer the peak is the weights plus one chunk. `Queue::write_buffer`
/// has the same problem when called for gigabytes before a submit.
pub struct Uploader<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    lost: &'a OnceCell<DeviceLostInfo>,
    staging: wgpu::Buffer,
    chunk: u64,
}

impl<'a> Uploader<'a> {
    /// Staging sized for the largest upload, capped at [`UPLOAD_CHUNK_BYTES`].
    pub fn new(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        lost: &'a OnceCell<DeviceLostInfo>,
        max_len: u64,
    ) -> Self {
        let align = wgpu::COPY_BUFFER_ALIGNMENT;
        let chunk = max_len
            .next_multiple_of(align)
            .clamp(align, UPLOAD_CHUNK_BYTES);
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("upload_staging"),
            size: chunk,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        Self {
            device,
            queue,
            lost,
            staging,
            chunk,
        }
    }

    /// A new `STORAGE` buffer holding `bytes` (zero-padded to 4 bytes).
    fn upload(&self, label: &str, bytes: &[u8]) -> Result<wgpu::Buffer> {
        let align = wgpu::COPY_BUFFER_ALIGNMENT;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (bytes.len() as u64).next_multiple_of(align).max(align),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.copy_chunks(&buf, 0, bytes)?;
        Ok(buf)
    }

    /// Write `bytes` into `buf` (which needs `COPY_DST`) at `offset`. Both
    /// must be 4-byte aligned: unlike [`Self::upload`] there is no padding,
    /// which would overwrite the bytes after the range.
    pub fn write(&self, buf: &wgpu::Buffer, offset: u64, bytes: &[u8]) -> Result<()> {
        let align = wgpu::COPY_BUFFER_ALIGNMENT;
        if !offset.is_multiple_of(align) || !(bytes.len() as u64).is_multiple_of(align) {
            return Err(PotError::Config("upload range is not 4-byte aligned"));
        }
        self.copy_chunks(buf, offset, bytes)
    }

    /// Copy `bytes` to `buf[offset..]` chunk by chunk; a ragged tail is
    /// zero-padded to 4 bytes.
    fn copy_chunks(&self, buf: &wgpu::Buffer, mut offset: u64, bytes: &[u8]) -> Result<()> {
        let align = wgpu::COPY_BUFFER_ALIGNMENT;
        for part in bytes.chunks(self.chunk as usize) {
            let n = (part.len() as u64).next_multiple_of(align);
            {
                let mut view = self.staging.slice(..n).get_mapped_range_mut();
                // Mapped memory may be write-combining: write-only access.
                view.slice(..part.len()).copy_from_slice(part);
                view.slice(part.len()..).fill(0);
            }
            self.staging.unmap();
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            enc.copy_buffer_to_buffer(&self.staging, 0, buf, offset, n);
            enc.map_buffer_on_submit(&self.staging, wgpu::MapMode::Write, .., |_| ());
            self.queue.submit([enc.finish()]);
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(PotError::Poll)?;
            if let Some(d) = self.lost.get() {
                return Err(PotError::DeviceLost {
                    reason: d.reason,
                    message: d.message.clone(),
                });
            }
            offset += n;
        }
        Ok(())
    }
}

/// `BONSAI_POT_ATTN=scalar` forces the scalar prefill attention kernel even
/// when f16 cooperative matrices are available.
fn attn_coopmat_from_env() -> bool {
    #[cfg(feature = "std")]
    {
        var_os("BONSAI_POT_ATTN").is_none_or(|v| v != "scalar")
    }
    #[cfg(not(feature = "std"))]
    {
        true
    }
}

/// `BONSAI_POT_COOPMAT=i8|f16|off` overrides the prefill matmul selection
/// (A/B testing); default prefers int8, then f16, then the `DP4a` kernel.
fn coopmat_choice_from_env() -> CoopChoice {
    #[cfg(feature = "std")]
    {
        match var_os("BONSAI_POT_COOPMAT")
            .as_deref()
            .and_then(|v| v.to_str())
        {
            Some("i8") => CoopChoice::I8,
            Some("f16") => CoopChoice::F16,
            Some("off" | "0") => CoopChoice::Off,
            _ => CoopChoice::Auto,
        }
    }
    #[cfg(not(feature = "std"))]
    {
        CoopChoice::Auto
    }
}

/// Cooperative-matrix configurations (16×16×16, subgroup scope) the device offers.
#[derive(Copy, Clone, Debug, Default)]
struct CoopSupport {
    /// f16 × f16 → f32
    f16: bool,
    /// u8 × s8 → s32
    i8: bool,
}

/// Open the Vulkan device through the HAL so we can pick the async-compute
/// queue family, request a global priority, and enable
/// `VK_KHR_cooperative_matrix` when `want_coopmat` and the device offers a
/// usable 16×16×16 subgroup configuration (returned as [`CoopSupport`]).
fn open_device(
    adapter: &wgpu::Adapter,
    desc: &wgpu::DeviceDescriptor<'_>,
    priority: GlobalPriority,
    want_coopmat: bool,
) -> Result<(wgpu::Device, wgpu::Queue, CoopSupport)> {
    let mut coop = CoopSupport::default();
    // SAFETY: we replicate wgpu-hal's `open_with_callback`, passing the chosen
    // queue family through to `device_from_raw`, and hand the raw device back
    // to wgpu with the same features/limits it was created with.
    let hal_open = unsafe {
        use core::ffi::CStr;

        use ash::khr::{cooperative_matrix, global_priority, shader_maximal_reconvergence};
        use ash::vk;
        use wgpu::hal::api::Vulkan as VulkanApi;

        let Some(hal_adapter) = adapter.as_hal::<VulkanApi>() else {
            return Err(PotError::NoAdapter);
        };
        let pd = hal_adapter.raw_physical_device();
        let instance = hal_adapter.shared_instance().raw_instance();
        let families = instance.get_physical_device_queue_family_properties(pd);
        let family_idx: u32 = families
            .iter()
            .position(|p| {
                p.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    && !p.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            })
            .map_or(0, |i| i as u32);
        log::info!(
            "vk queue family {family_idx} (async_compute={})",
            family_idx != 0
        );

        let enabled_extensions = hal_adapter.required_device_extensions(desc.required_features);
        let mut enabled_phd_features =
            hal_adapter.physical_device_features(&enabled_extensions, desc.required_features);
        let supported = instance
            .enumerate_device_extension_properties(pd)
            .unwrap_or_default();
        let has = |n: &CStr| {
            supported
                .iter()
                .any(|e| e.extension_name_as_c_str() == Ok(n))
        };
        let gp_supported = has(global_priority::NAME);
        let mr_supported = has(shader_maximal_reconvergence::NAME);
        if want_coopmat && has(cooperative_matrix::NAME) {
            let shared = hal_adapter.shared_instance();
            let cm = cooperative_matrix::Instance::new(shared.entry(), instance);
            let props = cm
                .get_physical_device_cooperative_matrix_properties(pd)
                .unwrap_or_default();
            let has_cfg =
                |a: vk::ComponentTypeKHR, b: vk::ComponentTypeKHR, c: vk::ComponentTypeKHR| {
                    props.iter().any(|p| {
                        p.m_size == 16
                            && p.n_size == 16
                            && p.k_size == 16
                            && p.a_type == a
                            && p.b_type == b
                            && p.c_type == c
                            && p.result_type == c
                            && p.scope == vk::ScopeKHR::SUBGROUP
                    })
                };
            coop.f16 = has_cfg(
                vk::ComponentTypeKHR::FLOAT16,
                vk::ComponentTypeKHR::FLOAT16,
                vk::ComponentTypeKHR::FLOAT32,
            );
            coop.i8 = has_cfg(
                vk::ComponentTypeKHR::UINT8,
                vk::ComponentTypeKHR::SINT8,
                vk::ComponentTypeKHR::SINT32,
            );
        }

        let priorities = [global_priority_fallback_to_queue_prio(priority)];
        let mut gp_info =
            vk::DeviceQueueGlobalPriorityCreateInfoKHR::default().global_priority(priority);
        let mut qci = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family_idx)
            .queue_priorities(&priorities);
        if gp_supported {
            qci = qci.push_next(&mut gp_info);
        } else {
            log::warn!(
                "VK_KHR_global_priority not supported; ignoring global_priority {priority:?}"
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
        let cm_ext_needed =
            (coop.f16 || coop.i8) && !enabled_extensions.contains(&cooperative_matrix::NAME);
        if cm_ext_needed {
            str_pointers.push(cooperative_matrix::NAME.as_ptr());
        }
        let pre_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&str_pointers);
        let mut info = enabled_phd_features.add_to_device_create(pre_info);
        let mut mr = vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR::default()
            .shader_maximal_reconvergence(true);
        if mr_supported {
            info = info.push_next(&mut mr);
        }
        let mut cm_features =
            vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
        if cm_ext_needed {
            info = info.push_next(&mut cm_features);
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
            Err(_) => return Err(PotError::NoDevice),
        }
    };
    // SAFETY: `hal_open` was created from this adapter with `desc`.
    let (device, queue) = unsafe { adapter.create_device_from_hal(hal_open, desc) }
        .map_err(|_| PotError::NoDevice)?;
    Ok((device, queue, coop))
}

/// cos/sin table for NEOX `RoPE`: `rope[pos * rope_half + i] = (cos, sin)` of
/// `pos * base^(-2i / rope_dim)`, computed in f64 so long positions stay exact.
fn build_rope_table(cfg: &ModelConfig, max_seq: u32) -> Vec<f32> {
    let half = (cfg.rope_dim / 2) as usize;
    let mut out = vec![0f32; max_seq as usize * half * 2];
    for p in 0..max_seq as usize {
        for i in 0..half {
            let theta =
                f64::from(cfg.rope_freq_base).powf(-2.0 * i as f64 / f64::from(cfg.rope_dim));
            let angle = p as f64 * theta;
            out[(p * half + i) * 2] = angle.cos() as f32;
            out[(p * half + i) * 2 + 1] = angle.sin() as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::tests::bonsai2_cfg;

    #[test]
    fn kv_cap_matches_4gib_estimate() {
        let cfg = bonsai2_cfg();
        let per = KvLayout::bytes_per_position(&cfg, cfg.n_attn_layers());
        assert_eq!(per, 18432);
        assert_eq!(u64::from(u32::MAX) / per, 233_016);
        let kv = KvLayout::build(&cfg, 233_016);
        assert!(u32::try_from(kv.total_bytes).is_ok());
        // last qs byte of the last layer stays addressable as u32
        let (off, len) = kv.qs_range(cfg.n_attn_layers() - 1, 233_016);
        assert!(u32::try_from(off + len).is_ok());
    }

    #[test]
    fn layouts_are_disjoint_and_monotonic() {
        let cfg = bonsai2_cfg();
        let a = ActLayout::build(&cfg);
        let offs = [
            a.x, a.xn, a.qkv, a.z, a.k, a.v, a.conv, a.gates, a.o, a.gate, a.up, a.logits,
            a.partials, a.total,
        ];
        assert!(offs.windows(2).all(|w| w[0] < w[1]));
        let q = Q8Layout::build(&cfg);
        assert!(q.qs < q.d && q.d < q.s && q.s < q.total_words);
        let s = StateLayout::build(&cfg);
        assert_eq!(s.ssm(0), 48 * 3 * 10240);
        assert_eq!(s.total, 48 * (3 * 10240 + 48 * 128 * 128));
    }

    #[test]
    fn window_planning() {
        let gib = 1u64 << 30;
        // 2.5 GiB file with cuts every 0.25 GiB under a 2 GiB limit → two windows.
        let cuts: Vec<u64> = (0..10).map(|i| i * gib / 4).collect();
        let w = plan_windows(10 * gib / 4, &cuts, 2 * gib - 1, 16).unwrap();
        assert_eq!(w.len(), 2);
        assert_eq!(w[0], (0, 7 * gib / 4));
        assert_eq!(w[1], (7 * gib / 4, 3 * gib / 4));
        assert_eq!(plan_windows(100, &[], 200, 4).unwrap(), [(0, 100)]);
        assert!(plan_windows(300, &[], 200, 4).is_err());
    }

    #[test]
    fn rope_table_first_entries() {
        let cfg = bonsai2_cfg();
        let t = build_rope_table(&cfg, 4);
        assert_eq!(t.len(), 4 * 32 * 2);
        assert_eq!(t[0], 1.0);
        assert_eq!(t[1], 0.0);
        let angle = 3.0f64 * 1e7f64.powf(-2.0 / 64.0);
        assert!((t[(3 * 32 + 1) * 2] - angle.cos() as f32).abs() < 1e-6);
    }
}
