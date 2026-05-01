//! Interactive chat REPL on top of the `bonsai_wgpu` library.
//!
//! Bonsai-4B is an instruction-tuned model, so we render each turn into the
//! Qwen-style ChatML template (`<|im_start|>...<|im_end|>`), shell out to
//! `scripts/bpe.py` for tokenization, prefill, and stream the assistant's
//! reply token-by-token. Multi-turn conversation is preserved across the
//! `Session`'s KV cache.
//!
//! Run:
//!   cargo run --release --example chat -- ./model
//!
//! Optional flags:
//!   --bpe scripts/bpe.py       path to the bpe.py tokenizer script
//!   --system "..."             override the system prompt
//!   --temperature 0.7
//!   --top-p 0.9
//!   --top-k 40
//!   --seed N                   default = wallclock-derived (non-reproducible)
//!   --max-new-tokens 512
//!
//! In-REPL commands: `/reset` clears the conversation, `/quit` exits.

use bonsai_wgpu::{BonsaiError, Model, ModelOptions, Sampler};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

// Qwen2/3 special-token IDs (see model/vocab.bin). The chat template uses
// IM_END as the assistant-turn terminator.
const IM_END: u32 = 151645;
const ENDOFTEXT: u32 = 151643;

struct Args {
    model_dir: PathBuf,
    bpe: String,
    system: String,
    temperature: f32,
    top_p: Option<f32>,
    top_k: Option<u32>,
    seed: u64,
    max_new_tokens: u32,
    max_seq: u32,
}

const HELP: &str = "\
Interactive ChatML REPL for the bonsai-wgpu engine.

USAGE:
    chat <model_dir> [OPTIONS]

ARGS:
    <model_dir>            Path to a directory produced by scripts/extract.py
                           (must contain config.json, weights_*.bin,
                           vocab.bin, vocab_offsets.bin, merges.txt).

OPTIONS:
    --bpe <path>           Path to the bpe.py tokenizer script.
                           [default: scripts/bpe.py]
    --system <text>        System prompt prepended on the first turn.
                           [default: \"You are a helpful assistant.\"]
    --temperature <f>      Sampling temperature; 0.0 ⇒ greedy/argmax.
                           [default: 0.7]
    --top-p <p>            Nucleus filter cutoff in (0, 1].
                           [default: 0.9]
    --top-k <k>            Truncate to the top-k logits before sampling
                           (capped at the engine's TOPK_MAX).
                           [default: 40]
    --seed <n>             PRNG seed for reproducible sampling.
                           [default: wallclock-derived]
    --max-new-tokens <n>   Hard cap on tokens emitted per assistant turn.
                           [default: 512]
    --max-seq <n>          KV-cache capacity (positions). Sets the upper
                           bound on prompt + generated tokens across the
                           whole conversation; raising it linearly grows
                           VRAM use. [default: 1024]
    -h, --help             Show this help and exit.

REPL COMMANDS:
    /reset                 Clear the conversation (drop KV cache).
    /quit, /exit           Exit the chat.

EXAMPLE:
    cargo run --release --example chat -- ./model
    cargo run --release --example chat -- ./model --temperature 0.0 \\
        --system \"You are a terse Rust expert.\"
";

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    // Handle -h / --help before requiring a positional model_dir.
    if argv.iter().skip(1).any(|a| a == "-h" || a == "--help") {
        print!("{HELP}");
        std::process::exit(0);
    }
    let model_dir = argv.get(1).map(PathBuf::from).unwrap_or_else(|| {
        eprintln!("error: missing <model_dir>\n\n{HELP}");
        std::process::exit(1);
    });
    let mut a = Args {
        model_dir,
        bpe: "scripts/bpe.py".into(),
        system: "You are a helpful assistant.".into(),
        temperature: 0.7,
        top_p: Some(0.9),
        top_k: Some(40),
        seed: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0),
        max_new_tokens: 512,
        max_seq: 1024,
    };
    let mut i = 2;
    while i < argv.len() {
        let val = || {
            argv.get(i + 1).cloned().unwrap_or_else(|| {
                eprintln!("error: missing value for {}\n\n{HELP}", argv[i]);
                std::process::exit(1);
            })
        };
        match argv[i].as_str() {
            "--bpe" => { a.bpe = val(); i += 2; }
            "--system" => { a.system = val(); i += 2; }
            "--temperature" => { a.temperature = val().parse().unwrap(); i += 2; }
            "--top-p" => { a.top_p = Some(val().parse().unwrap()); i += 2; }
            "--top-k" => { a.top_k = Some(val().parse().unwrap()); i += 2; }
            "--seed" => { a.seed = val().parse().unwrap(); i += 2; }
            "--max-new-tokens" => { a.max_new_tokens = val().parse().unwrap(); i += 2; }
            "--max-seq" => { a.max_seq = val().parse().unwrap(); i += 2; }
            _ => {
                eprintln!("error: unknown flag: {}\n\n{HELP}", argv[i]);
                std::process::exit(1);
            }
        }
    }
    a
}

/// Tokenize `text` by shelling out to `bpe.py`. The script reads vocab and
/// merges from `model_dir` and emits raw little-endian u32 token IDs on
/// stdout. Subprocess overhead (~tens of ms) is negligible compared to a
/// single GPU prefill, so we just spawn one per turn.
fn tokenize(bpe: &str, model_dir: &Path, text: &str) -> Vec<u32> {
    let out = Command::new("uv")
        .args(["run", "--quiet", bpe])
        .arg(model_dir)
        .arg(text)
        .stderr(Stdio::inherit())
        .output()
        .expect("failed to spawn bpe.py — is `uv` on PATH?");
    if !out.status.success() {
        eprintln!("bpe.py exited with status {}", out.status);
        std::process::exit(2);
    }
    bytemuck::cast_slice(&out.stdout).to_vec()
}

fn read_user_line() -> Option<String> {
    let mut s = String::new();
    match std::io::stdin().read_line(&mut s) {
        Ok(0) => None, // EOF
        Ok(_) => Some(s.trim_end_matches(['\n', '\r']).to_string()),
        Err(_) => None,
    }
}

fn main() {
    env_logger::builder().filter_level(log::LevelFilter::Warn).init();
    let args = parse_args();

    pollster::block_on(async move {
        eprintln!("loading model from {}…", args.model_dir.display());
        let model = Model::load_with_options(
            &args.model_dir,
            ModelOptions { max_seq: args.max_seq },
        ).await.expect("load model");
        let sampler = Sampler {
            temperature: args.temperature,
            top_k: args.top_k,
            top_p: args.top_p,
            seed: args.seed,
        };
        let mut sess = model.new_session();
        let mut stdout = std::io::stdout().lock();
        let mut turn: u32 = 0;

        eprintln!(
            "ready. type a message (Ctrl-D to quit, /reset to clear, /quit to exit).",
        );

        loop {
            write!(stdout, "\nYou: ").ok();
            stdout.flush().ok();
            let user = match read_user_line() {
                Some(l) => l,
                None => { writeln!(stdout).ok(); break; }
            };
            let user = user.trim();
            if user.is_empty() { continue; }
            if user == "/quit" || user == "/exit" { break; }
            if user == "/reset" {
                sess.reset();
                turn = 0;
                writeln!(stdout, "(conversation reset)").ok();
                continue;
            }

            // Build the ChatML segment to feed for THIS turn. On turn 0 we
            // include the system prompt. On later turns the previous turn's
            // closing `<|im_end|>` was sampled but not added to the KV cache
            // (generation stops *before* the EOS goes in), so we re-emit it
            // here to keep the conversation well-formed.
            let segment = if turn == 0 {
                format!(
                    "<|im_start|>system\n{}<|im_end|>\n\
                     <|im_start|>user\n{}<|im_end|>\n\
                     <|im_start|>assistant\n",
                    args.system, user,
                )
            } else {
                format!(
                    "<|im_end|>\n\
                     <|im_start|>user\n{}<|im_end|>\n\
                     <|im_start|>assistant\n",
                    user,
                )
            };
            let tokens = tokenize(&args.bpe, &args.model_dir, &segment);

            // Prefill: turn 0 uses the fast batched-matmul path (pos==0
            // required); later turns use the matvec-loop variant which works
            // at any pos. The first sampled token is returned but not yet in
            // KV — we feed it via `step` in the streaming loop below.
            let mut next = if turn == 0 {
                sess.prefill(&tokens, &sampler).await.expect("prefill")
            } else {
                sess.prefill_one_at_a_time(&tokens, &sampler).await
                    .expect("prefill_one_at_a_time")
            };

            write!(stdout, "Assistant: ").ok();
            stdout.flush().ok();
            let mut hit_overflow = false;
            for _ in 0..args.max_new_tokens {
                if next == IM_END || next == ENDOFTEXT { break; }
                stdout.write_all(&model.decode_token(next)).ok();
                stdout.flush().ok();
                match sess.step(next, &sampler).await {
                    Ok(t) => next = t,
                    Err(BonsaiError::ContextOverflow { .. }) => {
                        hit_overflow = true;
                        break;
                    }
                    Err(e) => panic!("step: {e}"),
                }
            }
            writeln!(stdout).ok();
            if hit_overflow {
                writeln!(
                    stdout,
                    "(context full at {} tokens — use /reset to start a new conversation)",
                    sess.pos(),
                ).ok();
            }
            turn += 1;
        }
    });
}
