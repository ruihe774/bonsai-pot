//! Bonsai-4B (Qwen3 architecture) Q1_0 inference: model loading & GPU resource setup.
//!
//! All large weights live in 5 storage buffers organized by role; the activation
//! workspace is a single FP32 buffer with named regions plus a separate buffer for
//! Q8_0 activations (used by the dot4I8Packed matmul path).

use bytemuck::{Pod, Zeroable};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use wgpu::util::DeviceExt;

// ----- config & manifest ----------------------------------------------------

#[allow(dead_code)]
#[derive(Deserialize, Debug, Clone)]
pub struct TensorEntry {
    pub dtype: String,
    pub shape: Vec<u64>,
    pub buffer: String,
    pub offset: u64,
    pub length: u64,
    #[serde(default)] pub d_offset: u64,
    #[serde(default)] pub qs_offset: u64,
    #[serde(default)] pub nb: u64,
}

#[allow(dead_code)]
#[derive(Deserialize, Debug, Clone)]
pub struct Config {
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
    pub padding_token_id: u32,
    pub add_bos: bool,
    pub context_length: u32,
    pub rope_orig_context: u32,
    pub n_kv_groups: u32,
    pub q_dim: u32,
    pub kv_dim: u32,
    pub tied_embeddings: bool,
    pub manifest: HashMap<String, TensorEntry>,
    pub prompt_n: u32,
    pub prompt_text: String,
}

// ----- uniform-param structs (WGSL-side struct layouts) ---------------------

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Default, Debug)]
pub struct EmbedParams {
    pub k: u32, pub d_offset: u32, pub qs_offset: u32, pub output_offset: u32,
    pub token_id: u32, pub m_token: u32, pub _p0: u32, pub _p1: u32,
}
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Default, Debug)]
pub struct RmsNormParams {
    pub group_size: u32, pub n_groups: u32, pub input_offset: u32, pub output_offset: u32,
    pub weight_offset: u32, pub eps: f32, pub _p0: u32, pub _p1: u32,
}
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Default, Debug)]
pub struct RopeParams {
    pub head_dim: u32, pub n_heads: u32, pub n_tokens: u32, pub pos_base: u32,
    pub data_offset: u32, pub rope_table_offset: u32, pub _p0: u32, pub _p1: u32,
}
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Default, Debug)]
pub struct MatvecParams {
    pub k: u32, pub n: u32, pub d_offset: u32, pub qs_offset: u32,
    pub input_offset: u32, pub output_offset: u32, pub accumulate: u32, pub dispatch_x_dim: u32,
}
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Default, Debug)]
pub struct QuantParams {
    pub k: u32, pub m: u32, pub input_offset: u32, pub d_offset: u32,
    pub qs_offset: u32, pub dispatch_x_dim: u32, pub _p1: u32, pub _p2: u32,
}
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Default, Debug)]
pub struct MatmulParams {
    pub k: u32, pub n: u32, pub m: u32,
    pub w_d_offset: u32, pub w_qs_offset: u32,
    pub a_d_offset: u32, pub a_qs_offset: u32,
    pub out_offset: u32, pub accumulate: u32,
    pub _p0: u32, pub _p1: u32, pub _p2: u32,
}
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Default, Debug)]
pub struct AttnParams {
    pub head_dim: u32, pub n_head: u32, pub n_kv_head: u32, pub pos: u32,
    pub kv_stride: u32, pub q_offset: u32, pub k_cache_offset: u32, pub v_cache_offset: u32,
    pub out_offset: u32, pub scale: f32, pub _p0: u32, pub _p1: u32,
}
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Default, Debug)]
pub struct SiluMulParams {
    pub n: u32, pub m: u32, pub gate_offset: u32, pub up_offset: u32,
    pub out_offset: u32, pub dispatch_x_count: u32, pub _p1: u32, pub _p2: u32,
}
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Default, Debug)]
pub struct ArgmaxParams {
    pub n: u32, pub in_offset: u32, pub out_offset: u32, pub _p0: u32,
}

// All of these <= 64 bytes; we pack each into a 256-byte uniform slot.
pub const UNIFORM_SLOT_SIZE: u64 = 256;

// ----- activation layout ----------------------------------------------------

#[derive(Copy, Clone, Debug)]
pub struct ActLayout {
    pub x: u32, pub x_norm: u32,
    pub q: u32, pub k_cur: u32, pub v_cur: u32,
    pub attn_out: u32,
    pub gate: u32, pub up: u32, pub ffn_in: u32,
    pub logits: u32,
    pub total_f32: u32,
}

impl ActLayout {
    pub fn build(cfg: &Config, m_max: u32) -> Self {
        let mut o = 0u32;
        let alloc = |n: u32, o: &mut u32| -> u32 { let r = *o; *o += n; r };
        let x       = alloc(m_max * cfg.n_embd, &mut o);
        let x_norm  = alloc(m_max * cfg.n_embd, &mut o);
        let q       = alloc(m_max * cfg.q_dim,  &mut o);
        let k_cur   = alloc(m_max * cfg.kv_dim, &mut o);
        let v_cur   = alloc(m_max * cfg.kv_dim, &mut o);
        let attn_out= alloc(m_max * cfg.q_dim,  &mut o);
        let gate    = alloc(m_max * cfg.n_ff,   &mut o);
        let up      = alloc(m_max * cfg.n_ff,   &mut o);
        let ffn_in  = alloc(m_max * cfg.n_ff,   &mut o);
        let logits  = alloc(cfg.n_vocab,        &mut o);
        Self { x, x_norm, q, k_cur, v_cur, attn_out, gate, up, ffn_in, logits, total_f32: o }
    }
}

// ----- the model ------------------------------------------------------------

pub struct Pipelines {
    pub embed: wgpu::ComputePipeline,
    pub rms_norm: wgpu::ComputePipeline,
    pub rope_neox: wgpu::ComputePipeline,
    pub matvec: wgpu::ComputePipeline,
    pub quantize: wgpu::ComputePipeline,
    pub matmul: wgpu::ComputePipeline,
    pub attention: wgpu::ComputePipeline,
    pub silu_mul: wgpu::ComputePipeline,
    pub argmax: wgpu::ComputePipeline,
}

pub struct Buffers {
    pub w_attn: wgpu::Buffer,
    pub w_ffn_gu: wgpu::Buffer,
    pub w_ffn_d: wgpu::Buffer,
    pub w_norms: wgpu::Buffer,
    pub w_embed: wgpu::Buffer,
    pub kv_k: wgpu::Buffer,
    pub kv_v: wgpu::Buffer,
    pub act: wgpu::Buffer,        // FP32 activations
    pub act_q8: wgpu::Buffer,     // Q8_0 activations (raw u32 buffer)
    pub rope_table: wgpu::Buffer,
    pub uniform: wgpu::Buffer,    // pool of 256-byte slots
    pub sample: wgpu::Buffer,     // u32 storage for sampled token id
    pub readback: wgpu::Buffer,   // u32 readback (mappable)
}

pub struct BindGroupLayouts {
    pub embed:    wgpu::BindGroupLayout,
    pub rms_norm: wgpu::BindGroupLayout,
    pub rope:     wgpu::BindGroupLayout,
    pub matvec:   wgpu::BindGroupLayout,
    pub quantize: wgpu::BindGroupLayout,
    pub matmul:   wgpu::BindGroupLayout,
    pub attn:     wgpu::BindGroupLayout,
    pub silu_mul: wgpu::BindGroupLayout,
    pub argmax:   wgpu::BindGroupLayout,
}

pub struct Model {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub cfg: Config,
    pub act_layout: ActLayout,
    pub m_max: u32,
    pub max_seq: u32,
    pub buffers: Buffers,
    pub pipes: Pipelines,
    pub bgls: BindGroupLayouts,
    pub vocab: Vec<String>,
    pub prompt: Vec<u32>,
}

const M_MAX: u32 = 512;
const MAX_SEQ: u32 = 1024;
const UNIFORM_POOL_SLOTS: u64 = 65536;

// ----- helper to look up a tensor entry strictly --------------------------

fn entry<'a>(cfg: &'a Config, name: &str) -> &'a TensorEntry {
    cfg.manifest.get(name).unwrap_or_else(|| panic!("missing tensor {name}"))
}

impl Model {
    pub async fn load(model_dir: &Path) -> Self {
        let cfg_text = std::fs::read_to_string(model_dir.join("config.json")).expect("config.json");
        let cfg: Config = serde_json::from_str(&cfg_text).expect("parse config");

        // ---- wgpu init ------------------------------------------------------
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("no GPU adapter");
        eprintln!("adapter: {:?}", adapter.get_info());

        let mut limits = adapter.limits();
        // Cap the requested binding size to the smaller of (1 GB, what the
        // adapter advertises). Bonsai-4B's largest grouped weight buffer is
        // ~252 MB so we need at least that.
        limits.max_storage_buffer_binding_size = limits
            .max_storage_buffer_binding_size
            .min(1u64 << 30)
            .max(300 * 1024 * 1024);
        limits.max_buffer_size = limits.max_buffer_size.min(1u64 << 30).max(300 * 1024 * 1024);
        limits.max_storage_buffers_per_shader_stage = limits.max_storage_buffers_per_shader_stage.max(8);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                    experimental_features: wgpu::ExperimentalFeatures::default(),
                    trace: wgpu::Trace::Off,
                },
            )
            .await
            .expect("request_device");

        // ---- load weight buffers from disk ---------------------------------
        let load = |fname: &str| -> Vec<u8> {
            std::fs::read(model_dir.join(fname)).unwrap_or_else(|e| panic!("read {fname}: {e}"))
        };
        let w_storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let make_storage = |label: &str, bytes: &[u8]| -> wgpu::Buffer {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label), contents: bytes, usage: w_storage,
            })
        };
        let w_attn   = make_storage("w_attn",   &load("weights_attn.bin"));
        let w_ffn_gu = make_storage("w_ffn_gu", &load("weights_ffn_gate_up.bin"));
        let w_ffn_d  = make_storage("w_ffn_d",  &load("weights_ffn_down.bin"));
        let w_norms  = make_storage("w_norms",  &load("weights_norms.bin"));
        let w_embed  = make_storage("w_embed",  &load("weights_embed_lmhead.bin"));

        // ---- build RoPE table ----------------------------------------------
        let rope_table = build_rope_table(&cfg, MAX_SEQ);
        let rope_buf = make_storage("rope_table", bytemuck::cast_slice(&rope_table));

        // ---- KV cache, activations, scratch --------------------------------
        let kv_per_layer: u64 = MAX_SEQ as u64 * cfg.kv_dim as u64;
        let kv_total: u64 = kv_per_layer * cfg.n_layer as u64 * 4;
        let kv_k = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kv_k"), size: kv_total,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let kv_v = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kv_v"), size: kv_total,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let act_layout = ActLayout::build(&cfg, M_MAX);
        let act_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("act"),
            size: act_layout.total_f32 as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // x_q8: per-token bytes = 2*nb_q8 (FP32 d's actually, so 4*nb_q8) + K
        // Sized for max K = n_ff (9728) and M_MAX tokens
        let max_k = cfg.n_ff;
        let q8_d_section_bytes  = M_MAX * (max_k / 32) * 4;
        let q8_qs_section_bytes = M_MAX * max_k;
        let act_q8_size = q8_d_section_bytes + q8_qs_section_bytes;
        // round up to multiple of 16
        let act_q8_size = ((act_q8_size + 15) / 16) * 16;
        let act_q8_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("act_q8"), size: act_q8_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Uniform pool: enough slots for a forward step
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniform"),
            size: UNIFORM_POOL_SLOTS * UNIFORM_SLOT_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Sample storage + readback
        let sample = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sample"), size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"), size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let buffers = Buffers {
            w_attn, w_ffn_gu, w_ffn_d, w_norms, w_embed,
            kv_k, kv_v, act: act_buf, act_q8: act_q8_buf,
            rope_table: rope_buf, uniform: uniform_buf, sample, readback,
        };

        // ---- shaders & pipelines -------------------------------------------
        macro_rules! load_shader {
            ($file:expr) => { device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some($file), source: wgpu::ShaderSource::Wgsl(include_str!(concat!("shaders/", $file)).into()),
            }) };
        }
        let sh_embed = load_shader!("embed.wgsl");
        let sh_rms = load_shader!("rms_norm.wgsl");
        let sh_rope = load_shader!("rope_neox.wgsl");
        let sh_matvec = load_shader!("matvec_q1_0.wgsl");
        let sh_quant = load_shader!("quantize_q8_0.wgsl");
        let sh_matmul = load_shader!("matmul_q1_0_q8_0.wgsl");
        let sh_attn = load_shader!("attention.wgsl");
        let sh_silu = load_shader!("silu_mul.wgsl");
        let sh_arg = load_shader!("argmax.wgsl");

        // bind group layouts
        fn ubo_dyn(binding: u32) -> wgpu::BindGroupLayoutEntry {
            wgpu::BindGroupLayoutEntry {
                binding, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: None,
                },
                count: None,
            }
        }
        fn ssbo(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
            wgpu::BindGroupLayoutEntry {
                binding, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        }
        let make_bgl = |label: &'static str, n_storage: u32, rw_mask: u32| -> wgpu::BindGroupLayout {
            let mut entries = vec![ubo_dyn(0)];
            for i in 0..n_storage {
                let read_only = (rw_mask >> i) & 1 == 0;
                entries.push(ssbo(i + 1, read_only));
            }
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label), entries: &entries,
            })
        };
        // Bind group layouts. Activation buffers always go through ONE
        // read_write binding so we never have the same buffer aliased as both
        // read and read_write within a dispatch.
        let bgls = BindGroupLayouts {
            embed:    make_bgl("embed_bgl",     2, 0b10),    // weights ro, act rw
            rms_norm: make_bgl("rms_norm_bgl",  2, 0b01),    // act rw, w ro
            rope:     make_bgl("rope_bgl",      2, 0b10),    // rope_cs ro, act rw
            matvec:   make_bgl("matvec_bgl",    2, 0b10),    // weights ro, act rw
            quantize: make_bgl("quantize_bgl",  2, 0b10),    // act ro, outbuf rw
            matmul:   make_bgl("matmul_bgl",    3, 0b100),   // weights ro, acts ro, y rw
            attn:     make_bgl("attn_bgl",      3, 0b001),   // act rw, k ro, v ro
            silu_mul: make_bgl("silu_mul_bgl",  1, 0b1),     // act rw
            argmax:   make_bgl("argmax_bgl",    2, 0b10),    // logits ro, result rw
        };

        let mk_pipe = |layout: &wgpu::BindGroupLayout, sh: &wgpu::ShaderModule, label: &str| -> wgpu::ComputePipeline {
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
                cache: None,
            })
        };
        let pipes = Pipelines {
            embed:     mk_pipe(&bgls.embed,    &sh_embed,  "embed"),
            rms_norm:  mk_pipe(&bgls.rms_norm, &sh_rms,    "rms_norm"),
            rope_neox: mk_pipe(&bgls.rope,     &sh_rope,   "rope_neox"),
            matvec:    mk_pipe(&bgls.matvec,   &sh_matvec, "matvec"),
            quantize:  mk_pipe(&bgls.quantize, &sh_quant,  "quantize"),
            matmul:    mk_pipe(&bgls.matmul,   &sh_matmul, "matmul"),
            attention: mk_pipe(&bgls.attn,     &sh_attn,   "attention"),
            silu_mul:  mk_pipe(&bgls.silu_mul, &sh_silu,   "silu_mul"),
            argmax:    mk_pipe(&bgls.argmax,   &sh_arg,    "argmax"),
        };

        // ---- vocab + prompt -------------------------------------------------
        let vocab_bytes = std::fs::read(model_dir.join("vocab.bin")).unwrap();
        let offs_bytes = std::fs::read(model_dir.join("vocab_offsets.bin")).unwrap();
        let offs: &[u32] = bytemuck::cast_slice(&offs_bytes);
        assert_eq!(offs.len() as u32, cfg.n_vocab + 1);
        let mut vocab = Vec::with_capacity(cfg.n_vocab as usize);
        for i in 0..cfg.n_vocab as usize {
            let s = std::str::from_utf8(&vocab_bytes[offs[i] as usize..offs[i+1] as usize])
                .unwrap_or("?")
                .to_string();
            vocab.push(s);
        }

        let prompt_bytes = std::fs::read(model_dir.join("prompt.bin")).unwrap();
        let prompt: Vec<u32> = bytemuck::cast_slice(&prompt_bytes).to_vec();
        eprintln!("prompt ({}): {} tokens", cfg.prompt_text, prompt.len());

        let _ = entry; // silence unused warning if helper stays unreferenced
        Self {
            device, queue, cfg, act_layout, m_max: M_MAX, max_seq: MAX_SEQ,
            buffers, pipes, bgls, vocab, prompt,
        }
    }
}

/// Precompute cos/sin table for NEOX rope: per position p (0..max_seq),
/// per j (0..head_dim/2), interleaved (cos, sin) pairs => head_dim floats per pos.
fn build_rope_table(cfg: &Config, max_seq: u32) -> Vec<f32> {
    let half = (cfg.head_dim / 2) as usize;
    let mut out = vec![0f32; max_seq as usize * cfg.head_dim as usize];
    for p in 0..max_seq as usize {
        for j in 0..half {
            let theta = (cfg.rope_freq_base as f64).powf(-2.0 * j as f64 / cfg.head_dim as f64);
            let angle = p as f64 * theta;
            out[p * cfg.head_dim as usize + 2 * j + 0] = angle.cos() as f32;
            out[p * cfg.head_dim as usize + 2 * j + 1] = angle.sin() as f32;
        }
    }
    out
}

// ----- public helpers used by forward.rs --------------------------------------

pub fn tensor<'a>(cfg: &'a Config, name: &str) -> &'a TensorEntry {
    cfg.manifest.get(name).unwrap_or_else(|| panic!("missing tensor {name}"))
}
