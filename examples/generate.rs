//! Smallest end-to-end example using the public `bonsai_wgpu` API.
//!
//! Run with:
//!   cargo run --release --example generate -- ./model < ./model/prompt.bin
//!
//! Reads pre-tokenized u32 IDs from stdin, prefills the model, then greedily
//! generates 64 tokens and prints the decoded text.

use bonsai_wgpu::{GenerateOptions, Model, Sampler};
use std::io::{Read, Write};

fn main() {
    env_logger::builder().filter_level(log::LevelFilter::Warn).init();
    let dir = std::env::args().nth(1).expect("usage: generate <model_dir> < prompt.bin");
    let dir = std::path::PathBuf::from(dir);

    let mut buf = Vec::new();
    std::io::stdin().lock().read_to_end(&mut buf).expect("read stdin");
    let prompt: Vec<u32> = bytemuck::cast_slice(&buf).to_vec();

    pollster::block_on(async move {
        let model = Model::load(&dir).await.expect("load model");
        let sampler = Sampler { temperature: 0.0, ..Default::default() }; // greedy
        let mut sess = model.new_session();
        let first = sess.prefill(&prompt, &sampler).await.expect("prefill");

        let mut stdout = std::io::stdout().lock();
        stdout.write_all(&model.decode_token(first)).ok();
        let opts = GenerateOptions {
            max_new_tokens: 63,
            sampler,
            ..Default::default()
        };
        sess.generate_streaming(first, &opts, |id| {
            stdout.write_all(&model.decode_token(id)).ok();
            stdout.flush().ok();
        }).await.expect("generate");
        writeln!(stdout).ok();
    });
}
