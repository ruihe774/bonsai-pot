use std::path::PathBuf;

mod model;
mod forward;

fn main() {
    env_logger::builder().filter_level(log::LevelFilter::Warn).init();
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: bonsai-wgpu <model_dir> [--max-new-tokens N] [--mode {gen,prompt,bench}] [--pp N] [--tg N] [--repeats N]";
    let model_dir = args.get(1).map(PathBuf::from).expect(usage);
    let mut n_gen: u32 = 32;
    let mut mode = "gen".to_string();
    let mut pp_n: u32 = 512;
    let mut tg_n: u32 = 128;
    let mut repeats: u32 = 5;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--max-new-tokens" => { n_gen = args[i+1].parse().unwrap(); i += 2; }
            "--mode" => { mode = args[i+1].clone(); i += 2; }
            "--pp" => { pp_n = args[i+1].parse().unwrap(); i += 2; }
            "--tg" => { tg_n = args[i+1].parse().unwrap(); i += 2; }
            "--repeats" => { repeats = args[i+1].parse().unwrap(); i += 2; }
            _ => { eprintln!("{}", usage); std::process::exit(1); }
        }
    }

    pollster::block_on(async {
        let mut m = model::Model::load(&model_dir).await;
        if mode == "bench" {
            forward::bench(&mut m, pp_n, tg_n, repeats).await;
        } else if mode == "microbench" {
            forward::microbench_tg(&mut m, repeats).await;
        } else {
            forward::generate(&mut m, n_gen, &mode).await;
        }
    });
}
