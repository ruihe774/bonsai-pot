#![allow(
    clippy::panic,
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "demo"
)]

//! Interactive chat REPL on top of the `bonsai_pot` library.
//!
//! Bonsai is an instruction-tuned model family; we render each turn into the
//! Qwen-style `ChatML` template (`<|im_start|>...<|im_end|>`), tokenize with
//! the `tokenizers` crate (Qwen2 byte-level BPE, built from the model dir's
//! `vocab.bin` + `merges.txt`), prefill, and stream the assistant's reply
//! token-by-token. Multi-turn conversation is preserved across the `Session`'s
//! KV cache.
//!
//! Run:
//!   cargo run --release --example chat -- ./model
//!
//! Optional flags:
//!   --system "..."             override the system prompt
//!   --temperature 0.7
//!   --top-p 0.9
//!   --top-k 40
//!   --seed N                   default = wallclock-derived (non-reproducible)
//!   --max-new-tokens 512
//!
//! In-REPL commands: `/reset` clears the conversation, `/quit` exits.

use std::fmt::Display;
use std::io::{Write as _, stdin, stdout};
use std::path::{Path, PathBuf};
use std::process::exit;
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{env, fs};

use bonsai_pot::{KvSnapshot, Model, ModelOptions, PotError, Sampler};
use tokenizers::models::bpe::{BPE, Vocab};
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{AddedToken, SplitDelimiterBehavior, Tokenizer};

struct Args {
    model_dir: PathBuf,
    system: String,
    temperature: f32,
    top_p: Option<f32>,
    top_k: Option<u32>,
    seed: u64,
    max_new_tokens: u32,
    max_seq: u32,
}

const HELP: &str = "\
Interactive ChatML REPL for the bonsai-pot engine.

USAGE:
    chat <model_dir> [OPTIONS]

ARGS:
    <model_dir>            Path to a directory produced by scripts/extract.py
                           (must contain config.json, weights_*.bin,
                           vocab.bin, vocab_offsets.bin, merges.txt).

OPTIONS:
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
                           [default: 4096]
    --max-seq <n>          KV-cache capacity (positions). Sets the upper
                           bound on prompt + generated tokens across the
                           whole conversation; raising it linearly grows
                           VRAM use. [default: 16384]
    -h, --help             Show this help and exit.

REPL COMMANDS:
    /reset                 Restore the KV cache to just after the system prompt,
                           skipping a re-prefill (~1-2 ms vs hundreds of ms).
    /quit, /exit           Exit the chat.

EXAMPLE:
    cargo run --release --example chat -- ./model
    cargo run --release --example chat -- ./model --temperature 0.0 \\
        --system \"You are a terse Rust expert.\"
";

fn parse_or_die<T: FromStr>(flag: &str, raw: &str) -> T
where
    T::Err: Display,
{
    raw.parse::<T>().unwrap_or_else(|e| {
        eprintln!("error: invalid value for {flag}: {raw:?}: {e}\n\n{HELP}");
        exit(1);
    })
}

fn parse_args() -> Args {
    let argv: Vec<String> = env::args().collect();
    // Handle -h / --help before requiring a positional model_dir.
    if argv.iter().skip(1).any(|a| a == "-h" || a == "--help") {
        print!("{HELP}");
        exit(0);
    }
    let model_dir_arg = argv.get(1).map_or("model", AsRef::as_ref).to_owned();
    let model_dir = PathBuf::from(model_dir_arg);
    let mut a = Args {
        model_dir,
        system: "You are a helpful assistant.".into(),
        temperature: 0.7,
        top_p: Some(0.9),
        top_k: Some(40),
        seed: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64),
        max_new_tokens: 4096,
        max_seq: 16384,
    };
    let mut i = 2;
    while i < argv.len() {
        let val = || {
            argv.get(i + 1).cloned().unwrap_or_else(|| {
                eprintln!("error: missing value for {}\n\n{HELP}", argv[i]);
                exit(1);
            })
        };
        match argv[i].as_str() {
            "--system" => {
                a.system = val();
                i += 2;
            }
            "--temperature" => {
                a.temperature = parse_or_die("--temperature", &val());
                i += 2;
            }
            "--top-p" => {
                a.top_p = Some(parse_or_die("--top-p", &val()));
                i += 2;
            }
            "--top-k" => {
                a.top_k = Some(parse_or_die("--top-k", &val()));
                i += 2;
            }
            "--seed" => {
                a.seed = parse_or_die("--seed", &val());
                i += 2;
            }
            "--max-new-tokens" => {
                a.max_new_tokens = parse_or_die("--max-new-tokens", &val());
                i += 2;
            }
            "--max-seq" => {
                a.max_seq = parse_or_die("--max-seq", &val());
                i += 2;
            }
            _ => {
                eprintln!("error: unknown flag: {}\n\n{HELP}", argv[i]);
                exit(1);
            }
        }
    }
    a
}

// Qwen2 pretokenizer regex (from llama.cpp's qwen2 branch / scripts/bpe.py).
// Exhaustively matches every character so there are no unmatched gaps.
const QWEN2_RE: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|",
    r"[^\r\n\p{L}\p{N}]?\p{L}+|",
    r"\p{N}|",
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|",
    r"\s*[\r\n]+|",
    r"\s+(?!\S)|",
    r"\s+",
);

fn build_tokenizer(model: &bonsai_pot::Model, model_dir: &Path) -> Tokenizer {
    // Vocab from Model's already-loaded table (GPT-2 byte-encoded strings).
    let mut token_id: u32 = 0;
    let mut token_strs: Vec<String> = Vec::new();
    while let Some(s) = model.vocab_token(token_id) {
        token_strs.push(s.to_owned());
        token_id += 1;
    }

    let specials: Vec<AddedToken> = token_strs
        .iter()
        .filter(|s| s.starts_with("<|") && s.ends_with("|>"))
        .map(|s| AddedToken::from(s.as_str(), true))
        .collect();

    let vocab: Vocab = token_strs
        .into_iter()
        .enumerate()
        .map(|(i, s)| (s, i as u32))
        .collect();

    let merges: Vec<(String, String)> = fs::read_to_string(model_dir.join("merges.txt"))
        .expect("merges.txt not found")
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            let (a, b) = l.split_once(' ').expect("malformed merges.txt");
            (a.to_owned(), b.to_owned())
        })
        .collect();

    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .byte_fallback(false)
        .build()
        .expect("build BPE model");

    let mut tok = Tokenizer::new(bpe);

    // Qwen2 pretokenizer: regex matches tokens to keep (invert=true), then
    // ByteLevel maps each UTF-8 byte to its GPT-2 unicode codepoint.
    let split = Split::new(
        SplitPattern::Regex(QWEN2_RE.to_owned()),
        SplitDelimiterBehavior::Isolated,
        true,
    )
    .expect("build Split pretokenizer");
    tok.with_pre_tokenizer(Some(PreTokenizerWrapper::Sequence(Sequence::new(vec![
        PreTokenizerWrapper::Split(split),
        PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)),
    ]))));

    let _ = tok.add_special_tokens(specials);
    tok
}

fn read_user_line() -> Option<String> {
    let mut s = String::new();
    match stdin().read_line(&mut s) {
        Ok(n) if n > 0 => Some(s.trim_end_matches(['\n', '\r']).to_string()),
        _ => None, // EOF or error
    }
}

/// Remove every `<think>...</think>` span (inclusive) from `tokens`.
///
/// If a `<think>` is opened but never closed (e.g. hit `max_new_tokens`
/// mid-thought), every token from the unclosed `<think>` to the end is
/// dropped — there is no usable response in that tail.
fn strip_thinking(tokens: &[u32], open: Option<u32>, close: Option<u32>) -> Vec<u32> {
    let (Some(open), Some(close)) = (open, close) else {
        return tokens.to_vec();
    };
    let mut out = Vec::with_capacity(tokens.len());
    let mut depth = 0u32;
    for &tok in tokens {
        if tok == open {
            depth += 1;
        }
        if depth == 0 {
            out.push(tok);
        }
        if tok == close && depth > 0 {
            depth -= 1;
        }
    }
    // If depth > 0 an open tag was never closed — the tail was already
    // excluded by the depth==0 guard above, so no extra work needed.
    out
}

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Warn)
        .parse_default_env()
        .init();
    let args = parse_args();

    eprintln!("seed: {}", args.seed);

    pollster::block_on(async move {
        eprintln!("loading model from {}…", args.model_dir.display());
        let model = Model::load_with_options(
            &args.model_dir,
            ModelOptions {
                max_seq: args.max_seq,
                ..Default::default()
            },
        )
        .await
        .expect("load model");
        let im_end = model.token_id("<|im_end|>").unwrap_or_else(|| {
            eprintln!("error: vocab missing <|im_end|>");
            exit(2);
        });
        let endoftext = model.token_id("<|endoftext|>").unwrap_or_else(|| {
            eprintln!("error: vocab missing <|endoftext|>");
            exit(2);
        });
        // Qwen3 thinking tokens — absent on non-Qwen3 vocabs, in which case
        // stripping is silently disabled.
        let think_open = model.token_id("<think>");
        let think_close = model.token_id("</think>");
        let thinking_supported = think_open.is_some() && think_close.is_some();
        let sampler = Sampler {
            temperature: args.temperature,
            top_k: args.top_k,
            top_p: args.top_p,
            seed: args.seed,
        };
        let tok = build_tokenizer(&model, &args.model_dir);
        let mut sess = model.new_session();
        let mut stdout = stdout().lock();

        // Prefill the system prompt once with the fast batched-matmul path
        // (requires pos == 0), then snapshot the KV state. /reset restores
        // from this snapshot (~1-2 ms) instead of re-prefilling from scratch.
        let system_segment = format!("<|im_start|>system\n{}<|im_end|>\n", args.system);
        let system_tokens: Vec<u32> = tok
            .encode(system_segment.as_str(), false)
            .expect("tokenize system segment")
            .get_ids()
            .to_vec();
        eprintln!("prefilling system prompt ({} tokens)…", system_tokens.len());
        sess.prefill(&system_tokens, &sampler)
            .expect("system prefill");
        let system_snap: KvSnapshot = sess.snapshot().expect("system snapshot");

        let mut turn: u32 = 0;

        eprintln!("ready. type a message (Ctrl-D to quit, /reset to clear, /quit to exit).");

        loop {
            write!(stdout, "\nYou: ").ok();
            stdout.flush().ok();
            let Some(user) = read_user_line() else {
                writeln!(stdout).ok();
                break;
            };
            let user = user.trim();
            if user.is_empty() {
                continue;
            }
            if user == "/quit" || user == "/exit" {
                break;
            }
            if user == "/reset" {
                sess.restore(&system_snap).expect("restore system snapshot");
                turn = 0;
                writeln!(stdout, "(conversation reset)").ok();
                continue;
            }

            // Build the ChatML segment for this user turn. The system prompt
            // is already in the KV cache, so we only encode the user/assistant
            // wrapper. On turn 0 there is no prior assistant close; on later
            // turns we re-emit <|im_end|> to close the previous assistant turn
            // (generation stops before that token enters the KV cache).
            let segment = if turn == 0 {
                format!(
                    "<|im_start|>user\n{user}<|im_end|>\n\
                     <|im_start|>assistant\n"
                )
            } else {
                format!(
                    "<|im_end|>\n\
                     <|im_start|>user\n{user}<|im_end|>\n\
                     <|im_start|>assistant\n"
                )
            };
            let tokens: Vec<u32> = tok
                .encode(segment.as_str(), false)
                .expect("tokenize turn segment")
                .get_ids()
                .to_vec();

            // All user-turn prefills use the matvec-loop path (pos > 0 after
            // the system prompt). The first sampled token is not yet in KV —
            // it is fed back via `step` in the streaming loop below.
            let mut next = sess
                .prefill_one_at_a_time(&tokens, &sampler)
                .expect("prefill_one_at_a_time");

            // Snapshot here: the first sampled token hasn't been `step`-fed
            // yet, so this captures KV state just after `<|im_start|>assistant\n`.
            // We'll rewind to this point after generation to re-prefill with
            // thinking stripped, as required by the Qwen3 chat template.
            let pre_assistant_snap = if thinking_supported {
                Some(sess.snapshot().expect("pre-assistant snapshot"))
            } else {
                None
            };

            write!(stdout, "Assistant: ").ok();
            stdout.flush().ok();
            let mut hit_overflow = false;
            let mut out_tokens: Vec<u32> = Vec::new();
            let mut in_think = false;
            for _ in 0..args.max_new_tokens {
                if next == im_end || next == endoftext {
                    break;
                }
                // Track thinking-block boundaries for display and stripping.
                if thinking_supported {
                    if Some(next) == think_open {
                        in_think = true;
                        stdout.write_all(b"\x1b[2m").ok(); // dim
                    }
                    out_tokens.push(next);
                }
                stdout.write_all(&model.decode_token(next)).ok();
                stdout.flush().ok();
                if thinking_supported && Some(next) == think_close {
                    in_think = false;
                    stdout.write_all(b"\x1b[0m").ok(); // reset
                }
                match sess.step(next, &sampler) {
                    Ok(t) => next = t,
                    Err(PotError::ContextOverflow { .. }) => {
                        hit_overflow = true;
                        break;
                    }
                    Err(e) => panic!("step: {e}"),
                }
            }
            // Reset dim if we hit max_new_tokens or overflow mid-think.
            if in_think {
                stdout.write_all(b"\x1b[0m").ok();
            }
            writeln!(stdout).ok();
            if hit_overflow {
                writeln!(
                    stdout,
                    "(context full at {} tokens — use /reset to start a new conversation)",
                    sess.pos(),
                )
                .ok();
            }

            // Rewind KV cache to just after the assistant header, then
            // re-prefill with thinking blocks removed. This matches the Qwen3
            // chat template, which strips prior-turn thinking from history.
            if let Some(snap) = pre_assistant_snap {
                let stripped = strip_thinking(&out_tokens, think_open, think_close);
                if stripped.len() != out_tokens.len() {
                    // Something was stripped — rewind and rebuild KV.
                    sess.restore(&snap).expect("restore pre-assistant snapshot");
                    if stripped.is_empty() {
                        writeln!(stdout, "(no response after thinking)").ok();
                    } else {
                        sess.prefill_one_at_a_time(&stripped, &sampler)
                            .expect("re-prefill stripped response");
                    }
                }
            }
            turn += 1;
        }
    });
}
