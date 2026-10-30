#![allow(
    clippy::panic,
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "demo"
)]

//! Interactive chat REPL on top of the `bonsai_pot` library.
//!
//! Renders each turn with the Ternary-Bonsai-2 (Qwen3.5) chat template:
//! the system message carries the reasoning-effort instruction, and the
//! assistant turn opens with `<think>\n` (or an empty think block when
//! thinking is disabled). Following the template's default
//! `preserve_thinking`, previous reasoning stays in the conversation, so the
//! session state simply keeps growing turn by turn. Tokenization uses the
//! `tokenizers` crate (byte-level BPE with the qwen35 pre-tokenizer, built
//! from the model dir's vocab + `merges.txt`, with the added tokens matched
//! atomically).
//!
//! Run:
//!   cargo run --release --example chat -- ./model
//!
//! In-REPL commands: `/reset` clears the conversation, `/quit` exits.

use std::fmt::Display;
use std::io::{Write, stdin, stdout};
use std::path::{Path, PathBuf};
use std::process::exit;
use std::str::FromStr;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::{env, fs};

use bonsai_pot::{GenerateOptions, KvSnapshot, LoadOptions, Model, PotError, Sampler};
use tokenizers::models::bpe::{BPE, Vocab};
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{AddedToken, SplitDelimiterBehavior, Tokenizer};

struct Args {
    model_dir: PathBuf,
    system: String,
    effort: String,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    min_p: Option<f32>,
    seed: u64,
    max_new_tokens: u32,
    max_seq: u32,
}

const HELP: &str = "\
Interactive chat REPL for the bonsai-pot engine (Ternary-Bonsai-2).

USAGE:
    chat <model_dir> [OPTIONS]

OPTIONS:
    --system <text>        System prompt. [default: \"You are a helpful assistant\"]
    --reasoning-effort <e> xhigh | medium | low | none (none disables thinking).
                           [default: xhigh]
    --temperature <f>      [default: 1.0 thinking / 0.7 none]
    --top-p <p>            [default: 0.95 thinking / 0.8 none]
    --top-k <k>            [default: 20]
    --min-p <p>            [default: 0.05 thinking / 0.0 none]
    --seed <n>             PRNG seed. [default: wallclock-derived]
    --max-new-tokens <n>   Cap per assistant turn. [default: 16384]
    --max-seq <n>          Session capacity in tokens. [default: 65536]
    -h, --help             Show this help and exit.

REPL COMMANDS:
    /reset                 Restore the state captured after the system prompt.
    /quit, /exit           Exit.
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
    if argv.iter().skip(1).any(|a| a == "-h" || a == "--help") {
        print!("{HELP}");
        exit(0);
    }
    let mut a = Args {
        model_dir: PathBuf::from(argv.get(1).map_or("model", AsRef::as_ref)),
        system: "You are a helpful assistant".into(),
        effort: "xhigh".into(),
        temperature: None,
        top_p: None,
        top_k: None,
        min_p: None,
        seed: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64),
        max_new_tokens: 16384,
        max_seq: 65536,
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
            "--system" => a.system = val(),
            "--reasoning-effort" => a.effort = val(),
            "--temperature" => a.temperature = Some(parse_or_die("--temperature", &val())),
            "--top-p" => a.top_p = Some(parse_or_die("--top-p", &val())),
            "--top-k" => a.top_k = Some(parse_or_die("--top-k", &val())),
            "--min-p" => a.min_p = Some(parse_or_die("--min-p", &val())),
            "--seed" => a.seed = parse_or_die("--seed", &val()),
            "--max-new-tokens" => a.max_new_tokens = parse_or_die("--max-new-tokens", &val()),
            "--max-seq" => a.max_seq = parse_or_die("--max-seq", &val()),
            _ => {
                eprintln!("error: unknown flag: {}\n\n{HELP}", argv[i]);
                exit(1);
            }
        }
        i += 2;
    }
    if !["xhigh", "medium", "low", "none"].contains(&a.effort.as_str()) {
        eprintln!("error: --reasoning-effort must be xhigh, medium, low or none");
        exit(1);
    }
    a
}

/// qwen35 pre-tokenizer regex (llama.cpp `LLAMA_VOCAB_PRE_TYPE_QWEN35`).
const QWEN35_RE: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|",
    r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|",
    r"\p{N}|",
    r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|",
    r"\s*[\r\n]+|",
    r"\s+(?!\S)|",
    r"\s+",
);

fn build_tokenizer(model: &Model, model_dir: &Path) -> Tokenizer {
    let mut token_strs: Vec<String> = Vec::new();
    while let Some(s) = model.vocab_token(token_strs.len() as u32) {
        token_strs.push(s.to_owned());
    }
    let specials: Vec<AddedToken> = model
        .added_tokens()
        .iter()
        .map(|&id| AddedToken::from(token_strs[id as usize].clone(), true))
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
    let split = Split::new(
        SplitPattern::Regex(QWEN35_RE.to_owned()),
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
        _ => None,
    }
}

fn reasoning_instructions(effort: &str) -> &'static str {
    match effort {
        "xhigh" => {
            "Reasoning effort is set to xhigh. Please think carefully through the task, validate key \
             assumptions, consider plausible alternatives, and prioritize correctness, consistency, and \
             clarity in the final answer."
        }
        "low" => {
            "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the \
             conclusion without unnecessary elaboration."
        }
        _ => "",
    }
}

fn fmt_phase(label: &str, n_tok: usize, dt: Duration) -> String {
    let s = dt.as_secs_f64();
    let tps = if s > 0.0 {
        n_tok as f64 / s
    } else {
        f64::INFINITY
    };
    format!("{label} {n_tok} tok / {:.1} ms = {tps:.1} t/s", s * 1000.0)
}

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Warn)
        .parse_default_env()
        .init();
    let args = parse_args();
    eprintln!("seed: {}", args.seed);
    eprintln!("loading model from {}…", args.model_dir.display());
    let model = Model::load_from_dir(
        &args.model_dir,
        LoadOptions {
            max_seq: args.max_seq,
            ..Default::default()
        },
    )
    .expect("load model");
    let think_open = model.token_id("<think>");
    let think_close = model.token_id("</think>");
    let thinking = args.effort != "none";
    let sampler = Sampler {
        temperature: args.temperature.unwrap_or(if thinking { 1.0 } else { 0.7 }),
        top_k: Some(args.top_k.unwrap_or(20)),
        top_p: Some(args.top_p.unwrap_or(if thinking { 0.95 } else { 0.8 })),
        min_p: args.min_p.unwrap_or(if thinking { 0.05 } else { 0.0 }),
        seed: args.seed,
    };
    let stops = model.stop_tokens();
    let tok = build_tokenizer(&model, &args.model_dir);
    let encode = |text: &str| -> Vec<u32> {
        tok.encode(text, false)
            .expect("tokenize")
            .get_ids()
            .to_vec()
    };
    let mut sess = model.new_session();
    let mut stdout = stdout().lock();

    let instr = reasoning_instructions(&args.effort);
    let system_content = match (instr.is_empty(), args.system.trim().is_empty()) {
        (true, true) => String::new(),
        (true, false) => args.system.trim().to_owned(),
        (false, true) => instr.to_owned(),
        (false, false) => format!("{instr}\n\n{}", args.system.trim()),
    };
    let system_snap: Option<KvSnapshot> = if system_content.is_empty() {
        None
    } else {
        let system_tokens = encode(&format!("<|im_start|>system\n{system_content}<|im_end|>\n"));
        eprintln!("prefilling system prompt ({} tokens)…", system_tokens.len());
        sess.prefill(&system_tokens, &sampler)
            .expect("system prefill");
        Some(sess.snapshot().expect("system snapshot"))
    };
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
            match &system_snap {
                Some(snap) => sess.restore(snap).expect("restore system snapshot"),
                None => sess.reset(),
            }
            turn = 0;
            writeln!(stdout, "(conversation reset)").ok();
            continue;
        }

        // The previous assistant turn stopped before its <|im_end|> entered the
        // state, so close it first.
        let close = if turn == 0 { "" } else { "<|im_end|>\n" };
        let open_think = if thinking {
            "<think>\n"
        } else {
            "<think>\n\n</think>\n\n"
        };
        let tokens = encode(&format!(
            "{close}<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{open_think}"
        ));

        let pp_t0 = Instant::now();
        let next = sess.prefill(&tokens, &sampler).expect("prefill");
        let pp_dt = pp_t0.elapsed();

        write!(stdout, "Assistant: ").ok();
        if thinking {
            stdout.write_all(b"\x1b[2m").ok();
        }
        stdout.flush().ok();
        let mut in_think = thinking;
        let mut emit = |id: u32, out: &mut dyn Write| {
            if Some(id) == think_open {
                in_think = true;
                out.write_all(b"\x1b[2m").ok();
            }
            out.write_all(&model.decode_token(id)).ok();
            if Some(id) == think_close {
                in_think = false;
                out.write_all(b"\x1b[0m").ok();
            }
            out.flush().ok();
        };
        let stop = |t: u32| stops.contains(&t);
        let gen_opts = GenerateOptions {
            max_new_tokens: args.max_new_tokens,
            stop_pred: Some(stop),
            sampler: sampler.clone(),
        };
        let tg_t0 = Instant::now();
        let mut tg_count = 0usize;
        let mut overflow = false;
        if !stop(next) {
            emit(next, &mut stdout);
            tg_count += 1;
            match sess.generate_streaming(next, &gen_opts, |id| {
                emit(id, &mut stdout);
                tg_count += 1;
            }) {
                Ok(_) => {}
                Err(PotError::ContextOverflow { .. }) => overflow = true,
                Err(e) => panic!("generate_streaming: {e}"),
            }
        }
        let tg_dt = tg_t0.elapsed();
        if in_think {
            stdout.write_all(b"\x1b[0m").ok();
        }
        writeln!(stdout).ok();
        if overflow {
            writeln!(
                stdout,
                "(context full at {} tokens — use /reset to start a new conversation)",
                sess.pos()
            )
            .ok();
        }
        eprintln!(
            "[{}  |  {}]",
            fmt_phase("PP", tokens.len(), pp_dt),
            fmt_phase("TG", tg_count, tg_dt)
        );
        turn += 1;
    }
}
