//! Ternary-Bonsai-2 (`Qwen3.5` hybrid attention, `PTQ1_0` ternary weights)
//! inference on wgpu compute shaders (Vulkan).
//!
//! See the README for a tour of the engine. This crate exposes the engine as
//! a library; the `bonsai-pot` binary is a thin CLI on top of this API that
//! reads pre-tokenized prompts from stdin.
//!
//! ```no_run
//! use bonsai_pot::{Model, GenerateOptions, Sampler};
//!
//! let model = Model::load("./model").unwrap();
//! let mut sess = model.new_session();
//!
//! // Tokenization is out-of-crate — pass pre-tokenized ids (e.g. from scripts/bpe.py).
//! let prompt: &[u32] = &[/* token ids … */];
//! let sampler = Sampler { temperature: 0.7, ..Default::default() };
//! let first_tok = sess.prefill(prompt, &sampler).unwrap();
//!
//! let opts = GenerateOptions { max_new_tokens: 64, sampler: sampler.clone(), ..Default::default() };
//! print!("{}", String::from_utf8_lossy(&model.decode_token(first_tok)));
//! sess.generate_streaming(first_tok, &opts, |id| {
//!     print!("{}", String::from_utf8_lossy(&model.decode_token(id)));
//! }).unwrap();
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#[macro_use]
extern crate alloc;

mod config;
mod cpu_ref;
mod decode;
mod error;
pub(crate) mod forward;
mod kv_snapshot;
mod model;
mod params;
mod session;

pub use config::{ModelConfig, TensorEntry};
pub use error::{PotError, Result};
pub use kv_snapshot::KvSnapshot;
pub use model::{GlobalPriority, LoadOptions, Model, ModelSnapshot, PowerPreference, TOPK_MAX};
pub use session::{GenerateOptions, Sampler, Session, StopReason};

/// Bench / microbench / reference-dump helpers, exposed only with the
/// `bench-internals` feature. Not part of the stable public API.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub mod __bench {
    pub use crate::forward::DumpTensor;
    pub use crate::forward::bench_internals::{bench, debug_prefill_x, dump};
    #[cfg(not(feature = "ci"))]
    pub use crate::forward::bench_internals::{microbench_pp, microbench_tg};
}
