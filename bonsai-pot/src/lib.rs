//! Bonsai (`Qwen3` architecture) `Q1_0` inference on wgpu compute shaders.
//!
//! See the README for a one-page tour of the engine. This crate exposes the
//! engine as a library; the `bonsai-pot` binary is a thin CLI on top of this
//! API that reads pre-tokenized prompts from stdin.
//!
//! ```no_run
//! use bonsai_pot::{Model, GenerateOptions, Sampler};
//!
//! // `Model::load` is synchronous; wgpu device init is internal.
//! let model = Model::load("./model").unwrap();
//! let mut sess = model.new_session();
//!
//! // Tokenization is out-of-crate — pass pre-tokenized ids (e.g. from scripts/bpe.py).
//! let prompt: &[u32] = &[/* token ids … */];
//! let sampler = Sampler { temperature: 0.7, ..Default::default() };
//! let first_tok = sess.prefill(prompt, &sampler).unwrap();
//!
//! // `generate_streaming` feeds `first_tok` as input and fires the callback
//! // for every subsequent token; emit the prefill-sampled token manually first.
//! let opts = GenerateOptions { max_new_tokens: 64, sampler: sampler.clone(), ..Default::default() };
//! print!("{}", String::from_utf8_lossy(&model.decode_token(first_tok)));
//! sess.generate_streaming(first_tok, &opts, |id| {
//!     print!("{}", String::from_utf8_lossy(&model.decode_token(id)));
//! }).unwrap();
//! ```
//!
//! Tokenization is **not** included — pass pre-tokenized `&[u32]` to
//! [`Session::prefill`]. Decode token ids back to bytes/text via
//! [`Model::decode_token`] and [`Model::decode_tokens`].

#![cfg_attr(not(feature = "std"), no_std)]
#[macro_use]
extern crate alloc;

#[cfg(all(target_vendor = "apple", not(target_arch = "aarch64")))]
compile_error!("the Metal backend requires Apple Silicon (aarch64); Intel Macs are not supported");

mod decode;
mod error;
pub(crate) mod forward;
mod kv_snapshot;
mod model;
mod session;

pub use error::{PotError, Result};
pub use kv_snapshot::KvSnapshot;
#[cfg(not(target_vendor = "apple"))]
pub use model::GlobalPriority;
pub use model::{LoadOptions, Model, ModelConfig, ModelSnapshot, TensorEntry};
pub use session::{GenerateOptions, Sampler, Session, StopReason};

/// Bench / microbench helpers, exposed only with the `bench-internals` feature.
/// Not part of the stable public API.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub mod __bench {
    pub use crate::forward::bench_internals::bench;
    #[cfg(all(not(feature = "ci"), not(target_vendor = "apple")))]
    pub use crate::forward::bench_internals::{microbench_pp, microbench_tg};
}
