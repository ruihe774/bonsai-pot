#![allow(clippy::panic)]
#![allow(clippy::expect_used)]
#![allow(clippy::unwrap_used)]

//! Demo CLI for `bonsai-pot`.
//!
//! Reads pre-tokenized prompts from stdin (little-endian u32 token IDs) and
//! generates text. Tokenization stays in your script of choice (e.g.
//! `extract.py`'s `bpe_encode`); pipe its u32 output here.
//!
//! Examples:
//!   cat ./model/prompt.bin | bonsai-pot ./model --max-new-tokens 64
//!   bonsai-pot ./model --mode bench --pp 512 --tg 128
//!   bonsai-pot ./model --temperature 0.8 --top-k 50 --top-p 0.95 --seed 42 \
//!       < ./model/prompt.bin

use std::env;
use std::fmt::Display;
use std::io::{stdin, stdout, Read, Write};
use std::path::PathBuf;
use std::process::exit;
use std::str::FromStr;

use bonsai_pot::{__bench, GenerateOptions, Model, ModelOptions, Sampler};

struct Args {
    model_dir: PathBuf,
    mode: String,
    n_gen: u32,
    pp_n: u32,
    tg_n: u32,
    max_seq: u32,
    repeats: u32,
    use_matmul_prefill: bool,
    temperature: f32,
    top_k: Option<u32>,
    top_p: Option<f32>,
    seed: u64,
    pipeline_cache_path: Option<PathBuf>,
}

const HELP: &str = "\
Demo CLI for the bonsai-pot engine.

USAGE:
    bonsai-pot <model_dir> [OPTIONS]

ARGS:
    <model_dir>            Path to a directory produced by scripts/extract.py.

MODES:
    --mode gen             (default) Single-token matvec for prefill + generation.
                           Reads u32 prompt tokens from stdin.
    --mode prompt          Batched matmul prefill, then matvec generation.
                           Reads u32 prompt tokens from stdin.
    --mode bench           Print an llama-bench-style pp/tg t/s table.
                           No stdin input required.
    --mode microbench      Per-kernel us/call breakdown for a tg step.
                           No stdin input required.

GEN/PROMPT OPTIONS:
    --max-new-tokens <n>   Tokens to generate (incl. the first sampled token).
                           [default: 32]
    --temperature <f>      Sampling temperature; 0.0 ⇒ greedy.
                           [default: 0.0]
    --top-k <k>            Truncate to top-k logits before sampling.
                           [default: unset]
    --top-p <p>            Nucleus filter cutoff in (0, 1].
                           [default: unset]
    --seed <n>             PRNG seed for reproducible sampling.
                           [default: 0]

BENCH OPTIONS:
    --pp <n>               Prefill batch size for the pp{N} bench row.
                           [default: 512]
    --tg <n>               Token-generation length for the tg{N} bench row.
                           [default: 128]
    --repeats <n>          Repeat count per bench row.
                           [default: 5]

LOAD OPTIONS:
    --max-seq <n>          KV-cache capacity (positions). Larger values cost
                           VRAM linearly. [default: 1024]
    --pipeline-cache <p>   Path to persist the compiled pipeline cache across
                           runs. Ignored on backends without PIPELINE_CACHE
                           support. [default: <model_dir>/pipeline_cache.bin]
    --no-pipeline-cache    Disable pipeline caching entirely.

OTHER:
    -h, --help             Show this help and exit.

EXAMPLES:
    uv run scripts/bpe.py ./model \"Once upon a time\" \\
        | bonsai-pot ./model --mode prompt --max-new-tokens 64
    bonsai-pot ./model --mode bench --pp 512 --tg 128
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
    let Some(model_dir_arg) = argv.get(1) else {
        eprintln!("error: missing <model_dir>\n\n{HELP}");
        exit(1);
    };
    let model_dir = PathBuf::from(model_dir_arg);
    let default_cache = model_dir.join("pipeline_cache.bin");
    let mut a = Args {
        model_dir,
        mode: "gen".to_string(),
        n_gen: 32,
        pp_n: 512,
        tg_n: 128,
        max_seq: 1024,
        repeats: 5,
        use_matmul_prefill: false,
        temperature: 0.0, // CLI default = greedy, for reproducible runs
        top_k: None,
        top_p: None,
        seed: 0,
        pipeline_cache_path: Some(default_cache),
    };
    let mut i = 2;
    while i < argv.len() {
        let next = || {
            argv.get(i + 1).cloned().unwrap_or_else(|| {
                eprintln!("error: missing value for {}\n\n{HELP}", argv[i]);
                exit(1)
            })
        };
        match argv[i].as_str() {
            "--mode" => {
                a.mode = next();
                i += 2;
            }
            "--max-new-tokens" => {
                a.n_gen = parse_or_die("--max-new-tokens", &next());
                i += 2;
            }
            "--pp" => {
                a.pp_n = parse_or_die("--pp", &next());
                i += 2;
            }
            "--tg" => {
                a.tg_n = parse_or_die("--tg", &next());
                i += 2;
            }
            "--repeats" => {
                a.repeats = parse_or_die("--repeats", &next());
                i += 2;
            }
            "--max-seq" => {
                a.max_seq = parse_or_die("--max-seq", &next());
                i += 2;
            }
            "--pipeline-cache" => {
                a.pipeline_cache_path = Some(PathBuf::from(next()));
                i += 2;
            }
            "--no-pipeline-cache" => {
                a.pipeline_cache_path = None;
                i += 1;
            }
            "--temperature" => {
                a.temperature = parse_or_die("--temperature", &next());
                i += 2;
            }
            "--top-k" => {
                a.top_k = Some(parse_or_die("--top-k", &next()));
                i += 2;
            }
            "--top-p" => {
                a.top_p = Some(parse_or_die("--top-p", &next()));
                i += 2;
            }
            "--seed" => {
                a.seed = parse_or_die("--seed", &next());
                i += 2;
            }
            _ => {
                eprintln!("error: unknown flag: {}\n\n{HELP}", argv[i]);
                exit(1);
            }
        }
    }
    // Mode shortcut: `--mode prompt` ⇒ batched matmul prefill;
    //                `--mode gen`    ⇒ matvec-loop prefill (parity with old CLI).
    a.use_matmul_prefill = a.mode == "prompt";
    a
}

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Warn)
        .parse_default_env()
        .init();
    let args = parse_args();

    pollster::block_on(async move {
        let model = Model::load_with_options(
            &args.model_dir,
            ModelOptions {
                max_seq: args.max_seq,
                pipeline_cache_path: args.pipeline_cache_path,
            },
        )
        .await
        .unwrap_or_else(|e| {
            eprintln!("load error: {e}");
            exit(2)
        });

        match args.mode.as_str() {
            "bench" => {
                __bench::bench(&model, args.pp_n, args.tg_n, args.repeats).unwrap_or_else(|e| {
                    eprintln!("bench error: {e}");
                    exit(3)
                });
            }
            "microbench" => {
                __bench::microbench_tg(&model, args.repeats).unwrap_or_else(|e| {
                    eprintln!("microbench error: {e}");
                    exit(3)
                });
            }
            "gen" | "prompt" => {
                let mut buf = Vec::new();
                stdin().lock().read_to_end(&mut buf).expect("read stdin");
                if buf.len() % 4 != 0 {
                    eprintln!("stdin must be a multiple of 4 bytes (u32 token IDs)");
                    exit(1);
                }
                let prompt: Vec<u32> = bytemuck::cast_slice(&buf).to_vec();
                if prompt.is_empty() {
                    eprintln!("empty prompt on stdin");
                    exit(1);
                }

                let sampler = Sampler {
                    temperature: args.temperature,
                    top_k: args.top_k,
                    top_p: args.top_p,
                    seed: args.seed,
                };
                let mut sess = model.new_session();
                let first = if args.use_matmul_prefill {
                    sess.prefill(&prompt, &sampler).expect("prefill")
                } else {
                    sess.prefill_one_at_a_time(&prompt, &sampler)
                        .expect("prefill")
                };

                let mut stdout = stdout().lock();
                // Echo the decoded prompt before the first sampled token,
                // matching the original CLI's behavior.
                for &id in &prompt {
                    stdout.write_all(&model.decode_token(id)).ok();
                }
                stdout.write_all(&model.decode_token(first)).ok();
                stdout.flush().ok();
                let opts = GenerateOptions {
                    max_new_tokens: args.n_gen.saturating_sub(1),
                    sampler,
                    ..Default::default()
                };
                sess.generate_streaming(first, &opts, |id| {
                    stdout.write_all(&model.decode_token(id)).ok();
                    stdout.flush().ok();
                })
                .expect("generate");
                writeln!(stdout).ok();
            }
            other => {
                eprintln!("unknown mode: {other}");
                exit(1);
            }
        }
    });
}
