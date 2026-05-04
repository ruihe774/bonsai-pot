#![cfg_attr(feature = "nightly", feature(f16))]
//! Bonsai (`Qwen3` architecture) `Q1_0` inference on wgpu compute shaders.
//!
//! See the README for a one-page tour of the engine. This crate exposes the
//! engine as a library; the `bonsai-pot` binary is a thin CLI on top of this
//! API that reads pre-tokenized prompts from stdin.
//!
//! ```ignore
//! use bonsai_pot::{Model, GenerateOptions, Sampler};
//!
//! pollster::block_on(async {
//!     let model = Model::load(std::path::Path::new("./model")).await.unwrap();
//!     let mut sess = model.new_session();
//!     let prompt: &[u32] = &[/* token ids ... */];
//!     let first = sess.prefill(prompt, &Sampler::default()).unwrap();
//!     let opts = GenerateOptions { max_new_tokens: 64, ..Default::default() };
//!     let (toks, _stop) = sess.generate(first, &opts).unwrap();
//!     print!("{}", model.decode_tokens(&[&[first][..], &toks].concat()));
//! });
//! ```
//!
//! Tokenization is **not** included — pass pre-tokenized `&[u32]` to
//! [`Session::prefill`]. Decode token ids back to bytes/text via
//! [`Model::decode_token`] and [`Model::decode_tokens`].

mod decode;
mod error;
pub(crate) mod forward;
mod kv_snapshot;
mod model;
mod session;

pub use error::{PotError, Result};
pub use kv_snapshot::KvSnapshot;
pub use model::{GlobalPriority, Model, ModelConfig, ModelOptions};
pub use session::{GenerateOptions, Sampler, Session, StopReason};

/// Bench / microbench helpers, exposed only with the `bench-internals` feature.
/// Not part of the stable public API.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub mod __bench {
    pub use crate::forward::bench_internals::{bench, microbench_tg};
}
