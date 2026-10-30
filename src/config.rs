//! `config.ini` parsing (written by `scripts/extract.py`), hyper-parameter
//! validation, and load-time resolution of every tensor offset the forward
//! pass needs.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString as _};
use alloc::vec::Vec;
use core::str::FromStr;

use crate::error::{PotError, Result};

/// Metadata for a single tensor in the model manifest.
#[derive(Debug, Clone)]
pub struct TensorEntry {
    /// `"PTQ1_0"`, `"F32"`, `"BF16"`, or `"BITS"` (packed Hadamard signs).
    pub dtype: &'static str,
    /// GGUF shape (`ne0` first: input dimension, then output rows).
    pub shape: Vec<u32>,
    /// Weight file holding the tensor.
    pub buffer: &'static str,
    /// Byte offset inside `buffer` (4-byte aligned).
    pub offset: u32,
    /// Byte length.
    pub length: u32,
    /// `PTQ1_0` blocks per row (`shape[0] / 128`); 0 for other dtypes.
    pub nb: u32,
}

/// Parsed `config.ini`: model hyper-parameters and the tensor manifest.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub n_layer: u32,
    pub n_embd: u32,
    pub n_ff: u32,
    /// Full-attention query heads.
    pub n_head: u32,
    /// Full-attention key/value heads.
    pub n_kv_head: u32,
    /// Full-attention head dimension.
    pub head_dim: u32,
    /// Rotary dimensions per full-attention head (NEOX pairs `i`, `i + rope_dim/2`).
    pub rope_dim: u32,
    pub rope_freq_base: f32,
    pub rms_eps: f32,
    /// Every `full_attn_interval`-th layer is full attention; the rest are
    /// Gated-DeltaNet linear attention.
    pub full_attn_interval: u32,
    pub ssm_conv_kernel: u32,
    /// Gated-DeltaNet head dimension (both key and value).
    pub ssm_head_dim: u32,
    pub ssm_n_k_heads: u32,
    pub ssm_n_v_heads: u32,
    /// `ssm_n_v_heads * ssm_head_dim`.
    pub ssm_inner: u32,
    pub context_length: u32,
    pub n_vocab: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    pub add_bos: bool,
    /// Block size of the Prism Hadamard rotation folded into the weights.
    pub hadamard_block: u32,
    /// Recommended sampling defaults carried by the GGUF.
    pub sampling_top_k: u32,
    pub sampling_top_p: f32,
    pub sampling_temp: f32,
    pub manifest: BTreeMap<String, TensorEntry>,
}

impl ModelConfig {
    /// Parse a `config.ini` text blob (as produced by `scripts/extract.py`).
    ///
    /// # Errors
    ///
    /// Returns `PotError::Config` for a malformed ini.
    pub fn from_ini(text: &str) -> Result<Self> {
        parse_config_ini(text)
    }

    /// `true` if layer `il` is a full-attention layer.
    #[must_use]
    pub const fn is_attn_layer(&self, il: u32) -> bool {
        (il + 1).is_multiple_of(self.full_attn_interval)
    }

    /// Number of full-attention layers (those owning a KV cache).
    #[must_use]
    pub fn n_attn_layers(&self) -> u32 {
        (0..self.n_layer)
            .filter(|&il| self.is_attn_layer(il))
            .count() as u32
    }

    /// Number of Gated-DeltaNet layers (those owning recurrent state).
    #[must_use]
    pub fn n_gdn_layers(&self) -> u32 {
        self.n_layer - self.n_attn_layers()
    }

    /// Full-attention query width (`n_head * head_dim`).
    #[must_use]
    pub const fn q_dim(&self) -> u32 {
        self.n_head * self.head_dim
    }

    /// Full-attention key/value width (`n_kv_head * head_dim`).
    #[must_use]
    pub const fn kv_dim(&self) -> u32 {
        self.n_kv_head * self.head_dim
    }

    /// Gated-DeltaNet conv channels: q, k (`n_k_heads` heads each) and v.
    #[must_use]
    pub const fn ssm_conv_dim(&self) -> u32 {
        2 * self.ssm_n_k_heads * self.ssm_head_dim + self.ssm_inner
    }

    pub(crate) fn tensor(&self, name: &str) -> Result<&TensorEntry> {
        self.manifest
            .get(name)
            .ok_or(PotError::Config("manifest: missing tensor"))
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
    fn intern_dtype(s: &str) -> Result<&'static str> {
        match s {
            "PTQ1_0" => Ok("PTQ1_0"),
            "F32" => Ok("F32"),
            "BF16" => Ok("BF16"),
            "BITS" => Ok("BITS"),
            _ => Err(PotError::Config("manifest: unknown dtype")),
        }
    }
    fn intern_buffer(s: &str) -> Result<&'static str> {
        match s {
            "weights_attn.bin" => Ok("weights_attn.bin"),
            "weights_ffn_gate_up.bin" => Ok("weights_ffn_gate_up.bin"),
            "weights_ffn_down.bin" => Ok("weights_ffn_down.bin"),
            "weights_embed_lmhead.bin" => Ok("weights_embed_lmhead.bin"),
            "weights_small.bin" => Ok("weights_small.bin"),
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
            nb: g.get("nb").map_or(Ok(0), |v| {
                v.parse()
                    .map_err(|_| PotError::Config("config.ini field has invalid value"))
            })?,
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
    if get(g, "arch")? != "qwen35" {
        return Err(PotError::Config(
            "config.ini: unsupported arch (expected qwen35)",
        ));
    }
    Ok(ModelConfig {
        n_layer: parse_field(g, "n_layer")?,
        n_embd: parse_field(g, "n_embd")?,
        n_ff: parse_field(g, "n_ff")?,
        n_head: parse_field(g, "n_head")?,
        n_kv_head: parse_field(g, "n_kv_head")?,
        head_dim: parse_field(g, "head_dim")?,
        rope_dim: parse_field(g, "rope_dim")?,
        rope_freq_base: parse_field(g, "rope_freq_base")?,
        rms_eps: parse_field(g, "rms_eps")?,
        full_attn_interval: parse_field(g, "full_attn_interval")?,
        ssm_conv_kernel: parse_field(g, "ssm_conv_kernel")?,
        ssm_head_dim: parse_field(g, "ssm_head_dim")?,
        ssm_n_k_heads: parse_field(g, "ssm_n_k_heads")?,
        ssm_n_v_heads: parse_field(g, "ssm_n_v_heads")?,
        ssm_inner: parse_field(g, "ssm_inner")?,
        context_length: parse_field(g, "context_length")?,
        n_vocab: parse_field(g, "n_vocab")?,
        eos_token_id: parse_field(g, "eos_token_id")?,
        pad_token_id: parse_field(g, "pad_token_id")?,
        add_bos: get(g, "add_bos")? == "true",
        hadamard_block: parse_field(g, "hadamard_block")?,
        sampling_top_k: parse_field(g, "sampling_top_k")?,
        sampling_top_p: parse_field(g, "sampling_top_p")?,
        sampling_temp: parse_field(g, "sampling_temp")?,
        manifest,
    })
}

/// Check the hyper-parameters against the invariants the kernels hard-code.
pub fn validate(cfg: &ModelConfig) -> Result<()> {
    let checks: &[(bool, &'static str)] = &[
        (cfg.n_layer > 0, "n_layer must be > 0"),
        (cfg.full_attn_interval > 0, "full_attn_interval must be > 0"),
        (
            cfg.hadamard_block == 1024,
            "kernels require hadamard_block == 1024",
        ),
        (
            cfg.n_embd.is_multiple_of(1024),
            "n_embd must be a multiple of 1024",
        ),
        (
            cfg.n_ff.is_multiple_of(1024),
            "n_ff must be a multiple of 1024",
        ),
        (
            cfg.ssm_inner.is_multiple_of(1024),
            "ssm_inner must be a multiple of 1024",
        ),
        (
            cfg.head_dim == 256,
            "full-attention kernels require head_dim == 256",
        ),
        (
            cfg.q_dim().is_multiple_of(1024),
            "n_head * head_dim must be a multiple of 1024",
        ),
        (
            cfg.n_kv_head > 0
                && cfg.n_head.is_multiple_of(cfg.n_kv_head)
                && cfg.n_head / cfg.n_kv_head <= 8,
            "n_head must be a multiple of n_kv_head with at most 8 query heads per group",
        ),
        (
            cfg.rope_dim.is_multiple_of(2) && cfg.rope_dim <= cfg.head_dim,
            "rope_dim must be even and <= head_dim",
        ),
        (
            cfg.ssm_head_dim == 128,
            "Gated-DeltaNet kernels require ssm_head_dim == 128",
        ),
        (
            cfg.ssm_conv_kernel == 4,
            "Gated-DeltaNet kernels require ssm_conv_kernel == 4",
        ),
        (
            cfg.ssm_n_k_heads > 0 && cfg.ssm_n_v_heads.is_multiple_of(cfg.ssm_n_k_heads),
            "ssm_n_v_heads must be a multiple of ssm_n_k_heads",
        ),
        (
            cfg.ssm_inner == cfg.ssm_n_v_heads * cfg.ssm_head_dim,
            "ssm_inner must equal ssm_n_v_heads * ssm_head_dim",
        ),
        (
            cfg.ssm_n_v_heads.is_multiple_of(8),
            "ssm_n_v_heads must be a multiple of 8",
        ),
        (!cfg.add_bos, "add_bos models are not supported"),
        (cfg.eos_token_id < cfg.n_vocab, "eos_token_id out of range"),
        (
            cfg.n_vocab.is_multiple_of(8),
            "n_vocab must be a multiple of 8",
        ),
    ];
    for &(ok, msg) in checks {
        if !ok {
            return Err(PotError::Config(msg));
        }
    }
    Ok(())
}

// ----- resolved offsets -----------------------------------------------------

/// Per-layer tensor offsets. `PTQ1_0` offsets are **u32 word** offsets into
/// their weight buffer; small-tensor offsets are word offsets into
/// `weights_small.bin`.
#[derive(Clone, Debug)]
pub struct LayerTensors {
    /// Weight window (see `Model::windows`) holding this layer's `ffn_gate/ffn_up`.
    pub gu_win: usize,
    pub attn_norm: u32,
    pub post_norm: u32,
    pub ffn_gate: u32,
    pub ffn_up: u32,
    pub ffn_down: u32,
    pub kind: LayerKind,
}

#[derive(Clone, Debug)]
pub enum LayerKind {
    Gdn(GdnTensors),
    Attn(AttnTensors),
}

#[derive(Clone, Debug)]
pub struct GdnTensors {
    pub qkv: u32,
    pub gate: u32,
    pub ssm_out: u32,
    pub ssm_norm: u32,
    pub conv1d: u32,
    pub dt_bias: u32,
    pub ssm_a: u32,
    pub alpha: u32,
    pub beta: u32,
    /// Index among the Gated-DeltaNet layers (selects the recurrent state slot).
    pub idx: u32,
}

#[derive(Clone, Debug)]
pub struct AttnTensors {
    pub q: u32,
    pub k: u32,
    pub v: u32,
    pub o: u32,
    pub q_norm: u32,
    pub k_norm: u32,
    /// Index among the full-attention layers (selects the KV-cache slot).
    pub idx: u32,
}

/// Global tensor offsets (embedding, LM head, output norm, Hadamard signs).
#[derive(Clone, Debug)]
pub struct OutputTensors {
    pub token_embd: u32,
    pub output: u32,
    pub output_norm: u32,
    pub signs_embd: u32,
    pub signs_q: u32,
    pub signs_inner: u32,
    pub signs_ff: u32,
}

fn expect(
    cfg: &ModelConfig,
    name: &str,
    dtype: &'static str,
    buffer: &'static str,
    shape: &[u32],
) -> Result<u32> {
    let t = cfg.tensor(name)?;
    if t.dtype != dtype || t.buffer != buffer || t.shape != shape {
        return Err(PotError::Config(
            "manifest: tensor dtype/buffer/shape mismatch",
        ));
    }
    if dtype == "PTQ1_0" {
        let rows = u64::from(shape[1]);
        if t.nb * 128 != shape[0] || u64::from(t.length) != rows * u64::from(t.nb) * 28 {
            return Err(PotError::Config("manifest: bad PTQ1_0 tensor size"));
        }
    }
    if t.offset % 4 != 0 {
        return Err(PotError::Config(
            "manifest: tensor offset not 4-byte aligned",
        ));
    }
    Ok(t.offset / 4)
}

const W_ATTN: &str = "weights_attn.bin";
const W_GU: &str = "weights_ffn_gate_up.bin";
const W_D: &str = "weights_ffn_down.bin";
const W_EMB: &str = "weights_embed_lmhead.bin";
const W_SMALL: &str = "weights_small.bin";

pub fn resolve_layers(cfg: &ModelConfig) -> Result<Vec<LayerTensors>> {
    let (e, f, hd) = (cfg.n_embd, cfg.n_ff, cfg.head_dim);
    let mut out = Vec::with_capacity(cfg.n_layer as usize);
    let (mut n_gdn, mut n_attn) = (0u32, 0u32);
    for il in 0..cfg.n_layer {
        let n = |s: &str| format!("blk.{il}.{s}");
        let kind = if cfg.is_attn_layer(il) {
            let t = AttnTensors {
                q: expect(
                    cfg,
                    &n("attn_q.weight"),
                    "PTQ1_0",
                    W_ATTN,
                    &[e, 2 * cfg.q_dim()],
                )?,
                k: expect(
                    cfg,
                    &n("attn_k.weight"),
                    "PTQ1_0",
                    W_ATTN,
                    &[e, cfg.kv_dim()],
                )?,
                v: expect(
                    cfg,
                    &n("attn_v.weight"),
                    "PTQ1_0",
                    W_ATTN,
                    &[e, cfg.kv_dim()],
                )?,
                o: expect(
                    cfg,
                    &n("attn_output.weight"),
                    "PTQ1_0",
                    W_ATTN,
                    &[cfg.q_dim(), e],
                )?,
                q_norm: expect(cfg, &n("attn_q_norm.weight"), "F32", W_SMALL, &[hd])?,
                k_norm: expect(cfg, &n("attn_k_norm.weight"), "F32", W_SMALL, &[hd])?,
                idx: n_attn,
            };
            n_attn += 1;
            LayerKind::Attn(t)
        } else {
            let (hv, inner) = (cfg.ssm_n_v_heads, cfg.ssm_inner);
            let t = GdnTensors {
                qkv: expect(
                    cfg,
                    &n("attn_qkv.weight"),
                    "PTQ1_0",
                    W_ATTN,
                    &[e, cfg.ssm_conv_dim()],
                )?,
                gate: expect(cfg, &n("attn_gate.weight"), "PTQ1_0", W_ATTN, &[e, inner])?,
                ssm_out: expect(cfg, &n("ssm_out.weight"), "PTQ1_0", W_ATTN, &[inner, e])?,
                ssm_norm: expect(
                    cfg,
                    &n("ssm_norm.weight"),
                    "F32",
                    W_SMALL,
                    &[cfg.ssm_head_dim],
                )?,
                conv1d: expect(
                    cfg,
                    &n("ssm_conv1d.weight"),
                    "F32",
                    W_SMALL,
                    &[cfg.ssm_conv_kernel, cfg.ssm_conv_dim()],
                )?,
                dt_bias: expect(cfg, &n("ssm_dt.bias"), "F32", W_SMALL, &[hv])?,
                ssm_a: expect(cfg, &n("ssm_a"), "F32", W_SMALL, &[hv])?,
                alpha: expect(cfg, &n("ssm_alpha.weight"), "BF16", W_SMALL, &[e, hv])?,
                beta: expect(cfg, &n("ssm_beta.weight"), "BF16", W_SMALL, &[e, hv])?,
                idx: n_gdn,
            };
            n_gdn += 1;
            LayerKind::Gdn(t)
        };
        out.push(LayerTensors {
            gu_win: 0,
            attn_norm: expect(cfg, &n("attn_norm.weight"), "F32", W_SMALL, &[e])?,
            post_norm: expect(cfg, &n("post_attention_norm.weight"), "F32", W_SMALL, &[e])?,
            ffn_gate: expect(cfg, &n("ffn_gate.weight"), "PTQ1_0", W_GU, &[e, f])?,
            ffn_up: expect(cfg, &n("ffn_up.weight"), "PTQ1_0", W_GU, &[e, f])?,
            ffn_down: expect(cfg, &n("ffn_down.weight"), "PTQ1_0", W_D, &[f, e])?,
            kind,
        });
    }
    Ok(out)
}

pub fn resolve_output(cfg: &ModelConfig) -> Result<OutputTensors> {
    let signs = |w: u32| expect(cfg, &format!("hadamard.signs.{w}"), "BITS", W_SMALL, &[w]);
    Ok(OutputTensors {
        token_embd: expect(
            cfg,
            "token_embd.weight",
            "PTQ1_0",
            W_EMB,
            &[cfg.n_embd, cfg.n_vocab],
        )?,
        output: expect(
            cfg,
            "output.weight",
            "PTQ1_0",
            W_EMB,
            &[cfg.n_embd, cfg.n_vocab],
        )?,
        output_norm: expect(cfg, "output_norm.weight", "F32", W_SMALL, &[cfg.n_embd])?,
        signs_embd: signs(cfg.n_embd)?,
        signs_q: signs(cfg.q_dim())?,
        signs_inner: signs(cfg.ssm_inner)?,
        signs_ff: signs(cfg.n_ff)?,
    })
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// Hyper-parameters of Ternary-Bonsai-2-27B with an empty manifest.
    pub fn bonsai2_cfg() -> ModelConfig {
        ModelConfig {
            n_layer: 64,
            n_embd: 5120,
            n_ff: 17408,
            n_head: 24,
            n_kv_head: 4,
            head_dim: 256,
            rope_dim: 64,
            rope_freq_base: 1e7,
            rms_eps: 1e-6,
            full_attn_interval: 4,
            ssm_conv_kernel: 4,
            ssm_head_dim: 128,
            ssm_n_k_heads: 16,
            ssm_n_v_heads: 48,
            ssm_inner: 6144,
            context_length: 262_144,
            n_vocab: 248_320,
            eos_token_id: 248_046,
            pad_token_id: 248_044,
            add_bos: false,
            hadamard_block: 1024,
            sampling_top_k: 20,
            sampling_top_p: 0.95,
            sampling_temp: 1.0,
            manifest: BTreeMap::new(),
        }
    }

    #[test]
    fn layer_pattern_and_dims() {
        let cfg = bonsai2_cfg();
        validate(&cfg).unwrap();
        assert_eq!(cfg.n_attn_layers(), 16);
        assert_eq!(cfg.n_gdn_layers(), 48);
        assert!(!cfg.is_attn_layer(0));
        assert!(cfg.is_attn_layer(3));
        assert!(cfg.is_attn_layer(63));
        assert_eq!(cfg.q_dim(), 6144);
        assert_eq!(cfg.kv_dim(), 1024);
        assert_eq!(cfg.ssm_conv_dim(), 10240);
    }

    #[test]
    fn parse_minimal_ini() {
        let text = "arch = qwen35\nn_layer = 64\nn_embd = 5120\nn_ff = 17408\nn_head = 24\n\
            n_kv_head = 4\nhead_dim = 256\nrope_dim = 64\nrope_freq_base = 10000000.0\n\
            rms_eps = 1e-06\nfull_attn_interval = 4\nssm_conv_kernel = 4\nssm_head_dim = 128\n\
            ssm_n_k_heads = 16\nssm_n_v_heads = 48\nssm_inner = 6144\ncontext_length = 262144\n\
            n_vocab = 248320\neos_token_id = 248046\npad_token_id = 248044\nadd_bos = false\n\
            hadamard_block = 1024\nsampling_top_k = 20\nsampling_top_p = 0.95\nsampling_temp = 1.0\n\
            \n[output.weight]\ndtype = PTQ1_0\nshape = 5120,248320\nbuffer = weights_embed_lmhead.bin\n\
            offset = 0\nlength = 278118400\nnb = 40\n";
        let cfg = ModelConfig::from_ini(text).unwrap();
        validate(&cfg).unwrap();
        let t = cfg.tensor("output.weight").unwrap();
        assert_eq!(t.nb, 40);
        assert_eq!(t.shape, [5120, 248_320]);
        assert!(expect(&cfg, "output.weight", "PTQ1_0", W_EMB, &[5120, 248_320]).is_ok());
    }

    #[test]
    fn rejects_other_arch() {
        assert!(ModelConfig::from_ini("arch = qwen3\n").is_err());
    }
}
