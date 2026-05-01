//! Forward pass and generation loop for Bonsai-4B.
//!
//! Two execution modes:
//!   - "gen":    processes every token (prompt + generated) via the multiply-free
//!               Q1_0 matvec kernel (single-token path).
//!   - "prompt": processes the prompt as one batch using the dot4I8Packed matmul
//!               with a Q8_0 quantize-activation pre-pass, then generates with
//!               matvec.

use crate::model::*;
use bytemuck::Pod;

// ---------- per-step encoder + uniform pool ---------------------------------

struct StepEncoder<'a> {
    model: &'a Model,
    encoder: wgpu::CommandEncoder,
    uniform_cpu: Vec<u8>,        // staged uniform writes
    next_slot: u64,              // 0..UNIFORM_POOL_SLOTS
}

impl<'a> StepEncoder<'a> {
    fn new(model: &'a Model) -> Self {
        let encoder = model.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("step"),
        });
        Self {
            model, encoder,
            uniform_cpu: Vec::with_capacity(64 * 1024),
            next_slot: 0,
        }
    }

    /// Append `params` into the uniform CPU buffer and return its dynamic offset.
    fn alloc_uniform<T: Pod>(&mut self, params: &T) -> u32 {
        let slot = self.next_slot;
        self.next_slot += 1;
        let off = (slot * UNIFORM_SLOT_SIZE) as usize;
        if self.uniform_cpu.len() < off + UNIFORM_SLOT_SIZE as usize {
            self.uniform_cpu.resize(off + UNIFORM_SLOT_SIZE as usize, 0);
        }
        let bytes = bytemuck::bytes_of(params);
        self.uniform_cpu[off..off + bytes.len()].copy_from_slice(bytes);
        // pad slot to UNIFORM_SLOT_SIZE (zero-init from resize)
        (slot * UNIFORM_SLOT_SIZE) as u32
    }

    fn finish(self) -> wgpu::CommandBuffer {
        // Write all queued uniforms in one go (well, this is our handoff to
        // queue.write_buffer which schedules the upload before submit).
        if !self.uniform_cpu.is_empty() {
            self.model.queue.write_buffer(&self.model.buffers.uniform, 0, &self.uniform_cpu);
        }
        self.encoder.finish()
    }
}

// ---------- bind-group helpers ----------------------------------------------

fn ubo_entry<'a>(buffer: &'a wgpu::Buffer, size_bytes: u64) -> wgpu::BindingResource<'a> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer, offset: 0, size: Some(std::num::NonZeroU64::new(size_bytes).unwrap()),
    })
}

fn make_bg(model: &Model, layout: &wgpu::BindGroupLayout, label: &str,
           uniform_size: u64, storages: &[&wgpu::Buffer]) -> wgpu::BindGroup {
    let mut entries = vec![wgpu::BindGroupEntry {
        binding: 0,
        resource: ubo_entry(&model.buffers.uniform, uniform_size),
    }];
    for (i, b) in storages.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: i as u32 + 1,
            resource: b.as_entire_binding(),
        });
    }
    model.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label), layout, entries: &entries,
    })
}

// ---------- benchmark entry point -------------------------------------------

/// Match `llama-bench` style: pp{n} measures batched-prefill throughput (tokens
/// pushed through projections via the dot4I8Packed matmul kernel); tg{n}
/// measures single-token generation throughput (multiply-free matvec on the
/// hot path, `n` forward steps from an empty KV cache).
pub async fn bench(model: &mut Model, pp_n: u32, tg_n: u32, repeats: u32) {
    let cfg = model.cfg.clone();
    eprintln!("--- bench: pp={pp_n}, tg={tg_n}, repeats={repeats} (after 1 warmup) ---");

    // synthesize a pp_n-token prompt (any valid token IDs work; we cycle 1..N).
    let prompt: Vec<u32> = (0..pp_n).map(|i| (i % (cfg.n_vocab - 1)) + 1).collect();

    // ----- warm up -----
    let _ = prefill_matmul(model, &cfg, &prompt[..pp_n.min(model.m_max) as usize]).await;
    let _ = step_matvec(model, &cfg, 1u32, 0, false).await;
    let _ = model.device.poll(wgpu::PollType::wait_indefinitely());

    // ----- pp{pp_n} -----
    let mut pp_times = Vec::with_capacity(repeats as usize);
    for _ in 0..repeats {
        let t = std::time::Instant::now();
        let _ = prefill_matmul(model, &cfg, &prompt).await;
        let _ = model.device.poll(wgpu::PollType::wait_indefinitely());
        pp_times.push(t.elapsed().as_secs_f32());
    }
    let pp_mean = pp_times.iter().sum::<f32>() / pp_times.len() as f32;
    let pp_std  = stddev(&pp_times, pp_mean);
    let pp_t_s_mean = pp_n as f32 / pp_mean;
    let pp_t_s_std  = pp_t_s_mean * (pp_std / pp_mean);

    // ----- tg{tg_n} (from empty KV) -----
    let mut tg_times = Vec::with_capacity(repeats as usize);
    for _ in 0..repeats {
        let t = std::time::Instant::now();
        let mut tok = 1u32;
        for pos in 0..tg_n {
            tok = step_matvec(model, &cfg, tok, pos, true).await;
        }
        let _ = model.device.poll(wgpu::PollType::wait_indefinitely());
        tg_times.push(t.elapsed().as_secs_f32());
        let _ = tok;
    }
    let tg_mean = tg_times.iter().sum::<f32>() / tg_times.len() as f32;
    let tg_std  = stddev(&tg_times, tg_mean);
    let tg_t_s_mean = tg_n as f32 / tg_mean;
    let tg_t_s_std  = tg_t_s_mean * (tg_std / tg_mean);

    println!();
    println!("| backend            |          test |               t/s |");
    println!("| ------------------ | ------------- | ----------------: |");
    println!("| bonsai-wgpu        |        pp{pp_n:<3} | {pp_t_s_mean:>9.2} ± {pp_t_s_std:>5.2} |");
    println!("| bonsai-wgpu        |        tg{tg_n:<3} | {tg_t_s_mean:>9.2} ± {tg_t_s_std:>5.2} |");
}

fn stddev(xs: &[f32], mean: f32) -> f32 {
    if xs.len() < 2 { return 0.0; }
    let var: f32 = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / (xs.len() - 1) as f32;
    var.sqrt()
}

// ---------- generation entry point ------------------------------------------

pub async fn generate(model: &mut Model, n_gen: u32, mode: &str) {
    let cfg = model.cfg.clone();
    let prompt = model.prompt.clone();
    let prompt_n = prompt.len() as u32;
    eprintln!("mode = {} | prompt_n = {} | n_gen = {}", mode, prompt_n, n_gen);

    let max_pos = prompt_n + n_gen;
    if max_pos > model.max_seq {
        panic!("prompt+gen ({max_pos}) exceeds MAX_SEQ ({})", model.max_seq);
    }

    let mut generated: Vec<u32> = Vec::with_capacity(n_gen as usize);

    let t_start = std::time::Instant::now();
    let mut last_tok = match mode {
        "prompt" => {
            // Single batched prefill of all prompt tokens; logits for the last
            // token live at the end of the activations buffer.
            let last = prefill_matmul(model, &cfg, &prompt).await;
            eprintln!("prefill ({} tokens) took {:?}", prompt_n, t_start.elapsed());
            last
        }
        "gen" => {
            // Run matvec single-token path over every prompt token.
            let mut last = 0u32;
            for (t, &tok) in prompt.iter().enumerate() {
                let want_logits = t == prompt.len() - 1;
                last = step_matvec(model, &cfg, tok, t as u32, want_logits).await;
                if !want_logits {
                    // discard sampled value — we just needed the cache to fill
                }
            }
            eprintln!("matvec prefill ({} tokens) took {:?}", prompt_n, t_start.elapsed());
            last
        }
        _ => panic!("mode must be 'gen' or 'prompt'"),
    };

    generated.push(last_tok);
    let t_gen = std::time::Instant::now();
    let mut pos = prompt_n;
    for _ in 1..n_gen {
        last_tok = step_matvec(model, &cfg, last_tok, pos, true).await;
        generated.push(last_tok);
        pos += 1;
        if last_tok == cfg.eos_token_id { break; }
    }
    let gen_secs = t_gen.elapsed().as_secs_f32();
    eprintln!("generated {} tokens in {:.2}s ({:.1} tok/s)",
              generated.len(), gen_secs, generated.len() as f32 / gen_secs.max(1e-3));

    // Decode & print
    print!("{}", cfg.prompt_text);
    for &id in &generated {
        let s = &model.vocab[id as usize];
        // GPT-2-ish byte-level vocab: 'Ġ' = space, 'Ċ' = newline, etc.
        // Convert each unicode codepoint back to its byte using the inverse map.
        print!("{}", decode_token_bytes(s));
    }
    println!();
}

/// Minimal inverse of GPT-2's bytes_to_unicode map: maps each codepoint in a
/// vocab token back to its raw byte, then interprets the bytes as UTF-8.
fn decode_token_bytes(s: &str) -> String {
    // Build the inverse map lazily with a thread-local cache.
    use std::sync::OnceLock;
    static INV: OnceLock<[u8; 0x180]> = OnceLock::new();
    let inv = INV.get_or_init(|| {
        let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32)
            .chain(0xa1..=0xac).chain(0xae..=0xff).collect();
        let mut cs = bs.clone();
        let mut n = 0u32;
        for b in 0..256u32 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }
        // map cs[i] -> bs[i] (cs[i] is the codepoint, bs[i] is the original byte)
        // We need a lookup: codepoint -> byte. cs values fit in [0x21, 0x180).
        let mut inv = [0u8; 0x180];
        for (b, c) in bs.iter().zip(cs.iter()) {
            if (*c as usize) < inv.len() {
                inv[*c as usize] = *b as u8;
            }
        }
        inv
    });
    let mut out_bytes = Vec::new();
    for ch in s.chars() {
        let cp = ch as usize;
        if cp < inv.len() && (inv[cp] != 0 || cp == 0) {
            out_bytes.push(inv[cp]);
        } else {
            // Fallback: keep as UTF-8 (special tokens like <|im_start|> etc.)
            let mut buf = [0u8; 4];
            out_bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
        }
    }
    String::from_utf8_lossy(&out_bytes).into_owned()
}

// ---------- single-token forward (matvec / multiply-free) -------------------

async fn step_matvec(model: &Model, cfg: &Config, token_id: u32, pos: u32, want_logits: bool) -> u32 {
    let mut se = StepEncoder::new(model);

    // 1. Embed lookup
    let tok_embd = tensor(cfg, "token_embd.weight").clone();
    {
        let p = EmbedParams {
            k: cfg.n_embd,
            d_offset: tok_embd.d_offset as u32,
            qs_offset: tok_embd.qs_offset as u32,
            output_offset: model.act_layout.x,
            token_id,
            m_token: 0,
            ..Default::default()
        };
        let off = se.alloc_uniform(&p);
        let bg = make_bg(model, &model.bgls.embed, "embed_bg", std::mem::size_of::<EmbedParams>() as u64,
                         &[&model.buffers.w_embed, &model.buffers.act]);
        let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("embed"), timestamp_writes: None });
        cp.set_pipeline(&model.pipes.embed);
        cp.set_bind_group(0, &bg, &[off]);
        cp.dispatch_workgroups(1, 1, 1);
    }

    // 2. Per-layer transformer
    for il in 0..cfg.n_layer {
        layer_step_matvec(model, cfg, &mut se, il, pos);
    }

    // 3. Final RMSNorm
    {
        let on = tensor(cfg, "output_norm.weight").clone();
        rms_norm(model, cfg, &mut se,
                 1, cfg.n_embd, model.act_layout.x, model.act_layout.x_norm,
                 (on.offset / 4) as u32);
    }

    // 4. LM head matvec (tied embeddings: same weights as token_embd)
    if want_logits {
        let lm_w = tensor(cfg, "token_embd.weight").clone();
        matvec_q1_0(model, cfg, &mut se,
                    cfg.n_embd, cfg.n_vocab,
                    &model.buffers.w_embed, lm_w.d_offset as u32, lm_w.qs_offset as u32,
                    model.act_layout.x_norm, model.act_layout.logits, false);

        // 5. Argmax → sample
        let p = ArgmaxParams { n: cfg.n_vocab, in_offset: model.act_layout.logits, out_offset: 0, _p0: 0 };
        let off = se.alloc_uniform(&p);
        let bg = make_bg(model, &model.bgls.argmax, "argmax_bg", std::mem::size_of::<ArgmaxParams>() as u64,
                         &[&model.buffers.act, &model.buffers.sample]);
        {
            let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("argmax"), timestamp_writes: None });
            cp.set_pipeline(&model.pipes.argmax);
            cp.set_bind_group(0, &bg, &[off]);
            cp.dispatch_workgroups(1, 1, 1);
        }
        // copy sample to readback
        se.encoder.copy_buffer_to_buffer(&model.buffers.sample, 0, &model.buffers.readback, 0, 4);
    }

    let cb = se.finish();
    model.queue.submit(Some(cb));

    if !want_logits {
        // Wait for previous work to drain so caches/uniforms are stable.
        let _ = model.device.poll(wgpu::PollType::wait_indefinitely());
        return 0;
    }
    // Map readback
    let slice = model.buffers.readback.slice(..);
    let (s, r) = futures_intrusive::channel::shared::oneshot_channel::<()>();
    slice.map_async(wgpu::MapMode::Read, move |_| { let _ = s.send(()); });
    let _ = model.device.poll(wgpu::PollType::wait_indefinitely());
    r.receive().await;
    let data = slice.get_mapped_range();
    let id = bytemuck::pod_read_unaligned::<u32>(&data[..4]);
    drop(data);
    model.buffers.readback.unmap();
    id
}

fn layer_step_matvec(model: &Model, cfg: &Config, se: &mut StepEncoder, il: u32, pos: u32) {
    // 2a. Pre-attn RMSNorm
    let attn_norm = tensor(cfg, &format!("blk.{il}.attn_norm.weight")).clone();
    rms_norm(model, cfg, se, 1, cfg.n_embd,
             model.act_layout.x, model.act_layout.x_norm, (attn_norm.offset / 4) as u32);

    // 2b. Q,K,V projections (matvec)
    let wq = tensor(cfg, &format!("blk.{il}.attn_q.weight")).clone();
    matvec_q1_0(model, cfg, se, cfg.n_embd, cfg.q_dim,
                &model.buffers.w_attn, wq.d_offset as u32, wq.qs_offset as u32,
                model.act_layout.x_norm, model.act_layout.q, false);
    let wk = tensor(cfg, &format!("blk.{il}.attn_k.weight")).clone();
    matvec_q1_0(model, cfg, se, cfg.n_embd, cfg.kv_dim,
                &model.buffers.w_attn, wk.d_offset as u32, wk.qs_offset as u32,
                model.act_layout.x_norm, model.act_layout.k_cur, false);
    let wv = tensor(cfg, &format!("blk.{il}.attn_v.weight")).clone();
    matvec_q1_0(model, cfg, se, cfg.n_embd, cfg.kv_dim,
                &model.buffers.w_attn, wv.d_offset as u32, wv.qs_offset as u32,
                model.act_layout.x_norm, model.act_layout.v_cur, false);

    // 2c. Per-head Q-norm and K-norm (BEFORE rope)
    let qn = tensor(cfg, &format!("blk.{il}.attn_q_norm.weight")).clone();
    rms_norm(model, cfg, se, cfg.n_head, cfg.head_dim,
             model.act_layout.q, model.act_layout.q, (qn.offset / 4) as u32);
    let kn = tensor(cfg, &format!("blk.{il}.attn_k_norm.weight")).clone();
    rms_norm(model, cfg, se, cfg.n_kv_head, cfg.head_dim,
             model.act_layout.k_cur, model.act_layout.k_cur, (kn.offset / 4) as u32);

    // 2d. RoPE
    rope(model, cfg, se, model.act_layout.q,     1, cfg.n_head,    pos);
    rope(model, cfg, se, model.act_layout.k_cur, 1, cfg.n_kv_head, pos);

    // 2e. Append K, V to cache
    let cache_row_bytes = (cfg.kv_dim * 4) as u64;
    let layer_offset_bytes = (il as u64) * (model.max_seq as u64) * cache_row_bytes;
    let dst_offset_bytes = layer_offset_bytes + (pos as u64) * cache_row_bytes;
    let k_src_bytes = (model.act_layout.k_cur as u64) * 4;
    let v_src_bytes = (model.act_layout.v_cur as u64) * 4;
    se.encoder.copy_buffer_to_buffer(&model.buffers.act, k_src_bytes, &model.buffers.kv_k, dst_offset_bytes, cache_row_bytes);
    se.encoder.copy_buffer_to_buffer(&model.buffers.act, v_src_bytes, &model.buffers.kv_v, dst_offset_bytes, cache_row_bytes);

    // 2f. Attention (single-token gen path: m_tokens=1, is_prefill=0, pos=pos+1)
    {
        let p = AttnParams {
            head_dim: cfg.head_dim, n_head: cfg.n_head, n_kv_head: cfg.n_kv_head,
            pos: pos + 1,
            kv_stride: cfg.kv_dim,
            q_offset: model.act_layout.q,
            k_cache_offset: ((layer_offset_bytes / 4) as u32),
            v_cache_offset: ((layer_offset_bytes / 4) as u32),
            out_offset: model.act_layout.attn_out,
            scale: 1.0 / (cfg.head_dim as f32).sqrt(),
            m_tokens: 1, is_prefill: 0,
        };
        let off = se.alloc_uniform(&p);
        let bg = make_bg(model, &model.bgls.attn, "attn_bg", std::mem::size_of::<AttnParams>() as u64,
                         &[&model.buffers.act, &model.buffers.kv_k, &model.buffers.kv_v]);
        let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("attn"), timestamp_writes: None });
        cp.set_pipeline(&model.pipes.attention);
        cp.set_bind_group(0, &bg, &[off]);
        cp.dispatch_workgroups(cfg.n_head, 1, 1);
    }

    // 2g. Wo with residual add
    let wo = tensor(cfg, &format!("blk.{il}.attn_output.weight")).clone();
    matvec_q1_0(model, cfg, se, cfg.q_dim, cfg.n_embd,
                &model.buffers.w_attn, wo.d_offset as u32, wo.qs_offset as u32,
                model.act_layout.attn_out, model.act_layout.x, true /*accumulate*/);

    // 2h. FFN: pre-RMSNorm
    let fn_n = tensor(cfg, &format!("blk.{il}.ffn_norm.weight")).clone();
    rms_norm(model, cfg, se, 1, cfg.n_embd,
             model.act_layout.x, model.act_layout.x_norm, (fn_n.offset / 4) as u32);

    // 2i. Wgate, Wup
    let wg = tensor(cfg, &format!("blk.{il}.ffn_gate.weight")).clone();
    matvec_q1_0(model, cfg, se, cfg.n_embd, cfg.n_ff,
                &model.buffers.w_ffn_gu, wg.d_offset as u32, wg.qs_offset as u32,
                model.act_layout.x_norm, model.act_layout.gate, false);
    let wu = tensor(cfg, &format!("blk.{il}.ffn_up.weight")).clone();
    matvec_q1_0(model, cfg, se, cfg.n_embd, cfg.n_ff,
                &model.buffers.w_ffn_gu, wu.d_offset as u32, wu.qs_offset as u32,
                model.act_layout.x_norm, model.act_layout.up, false);

    // 2j. SiLU(gate) * up
    silu_mul(model, se, cfg.n_ff, 1, model.act_layout.gate, model.act_layout.up, model.act_layout.ffn_in);

    // 2k. Wd with residual add
    let wd = tensor(cfg, &format!("blk.{il}.ffn_down.weight")).clone();
    matvec_q1_0(model, cfg, se, cfg.n_ff, cfg.n_embd,
                &model.buffers.w_ffn_d, wd.d_offset as u32, wd.qs_offset as u32,
                model.act_layout.ffn_in, model.act_layout.x, true /*accumulate*/);
}

// ---------- shader wrappers --------------------------------------------------

fn rms_norm(model: &Model, cfg: &Config, se: &mut StepEncoder,
            n_groups: u32, group_size: u32, in_off: u32, out_off: u32, w_off: u32) {
    let p = RmsNormParams {
        group_size, n_groups,
        input_offset: in_off, output_offset: out_off, weight_offset: w_off,
        eps: cfg.rms_eps,
        ..Default::default()
    };
    let off = se.alloc_uniform(&p);
    let bg = make_bg(model, &model.bgls.rms_norm, "rms_bg", std::mem::size_of::<RmsNormParams>() as u64,
                     &[&model.buffers.act, &model.buffers.w_norms]);
    let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("rms_norm"), timestamp_writes: None });
    cp.set_pipeline(&model.pipes.rms_norm);
    cp.set_bind_group(0, &bg, &[off]);
    cp.dispatch_workgroups(n_groups, 1, 1);
}

fn rope(model: &Model, cfg: &Config, se: &mut StepEncoder,
        data_off: u32, n_tokens: u32, n_heads: u32, pos_base: u32) {
    let p = RopeParams {
        head_dim: cfg.head_dim, n_heads, n_tokens, pos_base,
        data_offset: data_off, rope_table_offset: 0,
        ..Default::default()
    };
    let off = se.alloc_uniform(&p);
    let bg = make_bg(model, &model.bgls.rope, "rope_bg", std::mem::size_of::<RopeParams>() as u64,
                     &[&model.buffers.rope_table, &model.buffers.act]);
    let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("rope"), timestamp_writes: None });
    cp.set_pipeline(&model.pipes.rope_neox);
    cp.set_bind_group(0, &bg, &[off]);
    cp.dispatch_workgroups(n_tokens, n_heads, 1);
}

fn matvec_q1_0(model: &Model, _cfg: &Config, se: &mut StepEncoder,
               k: u32, n: u32,
               weights: &wgpu::Buffer, w_d: u32, w_qs: u32,
               in_off: u32, out_off: u32, accumulate: bool) {
    // Wgpu caps dispatch size per dim at 65535; for the LM-head matvec
    // (n=151669) we wrap into a 2D grid.
    let dispatch_x = n.min(65535);
    let dispatch_y = (n + dispatch_x - 1) / dispatch_x;
    let p = MatvecParams {
        k, n,
        d_offset: w_d, qs_offset: w_qs,
        input_offset: in_off, output_offset: out_off,
        accumulate: if accumulate { 1 } else { 0 },
        dispatch_x_dim: dispatch_x,
    };
    let off = se.alloc_uniform(&p);
    let bg = make_bg(model, &model.bgls.matvec, "matvec_bg", std::mem::size_of::<MatvecParams>() as u64,
                     &[weights, &model.buffers.act]);
    let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("matvec"), timestamp_writes: None });
    cp.set_pipeline(&model.pipes.matvec);
    cp.set_bind_group(0, &bg, &[off]);
    cp.dispatch_workgroups(dispatch_x, dispatch_y, 1);
}

// ---------- matmul (prompt prefill) ----------------------------------------

async fn prefill_matmul(model: &mut Model, cfg: &Config, prompt: &[u32]) -> u32 {
    let m = prompt.len() as u32;
    assert!(m <= model.m_max, "prompt {m} exceeds m_max={}", model.m_max);

    // ---- 1. Embed all M tokens (one shader call per token; cheap) ----------
    {
        let mut se = StepEncoder::new(model);
        let tok_embd = tensor(cfg, "token_embd.weight").clone();
        for (mi, &tid) in prompt.iter().enumerate() {
            let p = EmbedParams {
                k: cfg.n_embd,
                d_offset: tok_embd.d_offset as u32,
                qs_offset: tok_embd.qs_offset as u32,
                output_offset: model.act_layout.x,
                token_id: tid,
                m_token: mi as u32,
                ..Default::default()
            };
            let off = se.alloc_uniform(&p);
            let bg = make_bg(model, &model.bgls.embed, "embed_bg", std::mem::size_of::<EmbedParams>() as u64,
                             &[&model.buffers.w_embed, &model.buffers.act]);
            let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("embed"), timestamp_writes: None });
            cp.set_pipeline(&model.pipes.embed);
            cp.set_bind_group(0, &bg, &[off]);
            cp.dispatch_workgroups(1, 1, 1);
        }
        let cb = se.finish();
        model.queue.submit(Some(cb));
    }

    // ---- 2. Per-layer transformer using matmul (one CB for all layers) ----
    {
        let mut se = StepEncoder::new(model);
        for il in 0..cfg.n_layer {
            layer_step_matmul(model, cfg, &mut se, il, m);
        }
        let cb = se.finish();
        model.queue.submit(Some(cb));
    }

    // ---- 3. Final RMSNorm on each token's x; LM head matvec for last only --
    let mut se = StepEncoder::new(model);
    let on = tensor(cfg, "output_norm.weight").clone();
    rms_norm(model, cfg, &mut se, m, cfg.n_embd,
             model.act_layout.x, model.act_layout.x_norm, (on.offset / 4) as u32);

    // LM head matvec on the LAST token's x_norm only.
    let last_input_offset = model.act_layout.x_norm + (m - 1) * cfg.n_embd;
    let lm_w = tensor(cfg, "token_embd.weight").clone();
    matvec_q1_0(model, cfg, &mut se,
                cfg.n_embd, cfg.n_vocab,
                &model.buffers.w_embed, lm_w.d_offset as u32, lm_w.qs_offset as u32,
                last_input_offset, model.act_layout.logits, false);
    {
        let p = ArgmaxParams { n: cfg.n_vocab, in_offset: model.act_layout.logits, out_offset: 0, _p0: 0 };
        let off = se.alloc_uniform(&p);
        let bg = make_bg(model, &model.bgls.argmax, "argmax_bg", std::mem::size_of::<ArgmaxParams>() as u64,
                         &[&model.buffers.act, &model.buffers.sample]);
        let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("argmax"), timestamp_writes: None });
        cp.set_pipeline(&model.pipes.argmax);
        cp.set_bind_group(0, &bg, &[off]);
        cp.dispatch_workgroups(1, 1, 1);
    }
    se.encoder.copy_buffer_to_buffer(&model.buffers.sample, 0, &model.buffers.readback, 0, 4);
    let cb = se.finish();
    model.queue.submit(Some(cb));

    // Map readback
    let slice = model.buffers.readback.slice(..);
    let (s, r) = futures_intrusive::channel::shared::oneshot_channel::<()>();
    slice.map_async(wgpu::MapMode::Read, move |_| { let _ = s.send(()); });
    let _ = model.device.poll(wgpu::PollType::wait_indefinitely());
    r.receive().await;
    let id = bytemuck::pod_read_unaligned::<u32>(&slice.get_mapped_range()[..4]);
    model.buffers.readback.unmap();
    id
}

fn layer_step_matmul(model: &Model, cfg: &Config, se: &mut StepEncoder, il: u32, m: u32) {
    let pos_base = 0u32;  // prefill always starts at pos 0

    // 1. RMSNorm(attn_norm) for each token
    let an = tensor(cfg, &format!("blk.{il}.attn_norm.weight")).clone();
    rms_norm(model, cfg, se, m, cfg.n_embd,
             model.act_layout.x, model.act_layout.x_norm, (an.offset / 4) as u32);

    // 2. Quantize x_norm to Q8_0
    let (a_d_off, a_qs_off) = quantize_act(model, cfg, se, cfg.n_embd, m, model.act_layout.x_norm);

    // 3. Wq, Wk, Wv via matmul
    let wq = tensor(cfg, &format!("blk.{il}.attn_q.weight")).clone();
    matmul_q1_0(model, cfg, se, cfg.n_embd, cfg.q_dim, m,
                &model.buffers.w_attn, wq.d_offset as u32, wq.qs_offset as u32,
                a_d_off, a_qs_off, model.act_layout.q, false);
    let wk = tensor(cfg, &format!("blk.{il}.attn_k.weight")).clone();
    matmul_q1_0(model, cfg, se, cfg.n_embd, cfg.kv_dim, m,
                &model.buffers.w_attn, wk.d_offset as u32, wk.qs_offset as u32,
                a_d_off, a_qs_off, model.act_layout.k_cur, false);
    let wv = tensor(cfg, &format!("blk.{il}.attn_v.weight")).clone();
    matmul_q1_0(model, cfg, se, cfg.n_embd, cfg.kv_dim, m,
                &model.buffers.w_attn, wv.d_offset as u32, wv.qs_offset as u32,
                a_d_off, a_qs_off, model.act_layout.v_cur, false);

    // 4. Per-head Q-norm, K-norm (groups = M * n_head or M * n_kv_head)
    let qn = tensor(cfg, &format!("blk.{il}.attn_q_norm.weight")).clone();
    rms_norm(model, cfg, se, m * cfg.n_head, cfg.head_dim,
             model.act_layout.q, model.act_layout.q, (qn.offset / 4) as u32);
    let kn = tensor(cfg, &format!("blk.{il}.attn_k_norm.weight")).clone();
    rms_norm(model, cfg, se, m * cfg.n_kv_head, cfg.head_dim,
             model.act_layout.k_cur, model.act_layout.k_cur, (kn.offset / 4) as u32);

    // 5. RoPE
    rope(model, cfg, se, model.act_layout.q,     m, cfg.n_head,    pos_base);
    rope(model, cfg, se, model.act_layout.k_cur, m, cfg.n_kv_head, pos_base);

    // 6. Copy K, V to cache (M tokens at offsets [0, M))
    let cache_row_bytes = (cfg.kv_dim * 4) as u64;
    let layer_offset_bytes = (il as u64) * (model.max_seq as u64) * cache_row_bytes;
    let total_bytes = (m as u64) * cache_row_bytes;
    se.encoder.copy_buffer_to_buffer(&model.buffers.act, (model.act_layout.k_cur * 4) as u64,
                                     &model.buffers.kv_k, layer_offset_bytes, total_bytes);
    se.encoder.copy_buffer_to_buffer(&model.buffers.act, (model.act_layout.v_cur * 4) as u64,
                                     &model.buffers.kv_v, layer_offset_bytes, total_bytes);

    // 7. Attention: single batched dispatch of (n_head, M) WGs.
    {
        let p = AttnParams {
            head_dim: cfg.head_dim, n_head: cfg.n_head, n_kv_head: cfg.n_kv_head,
            pos: 0,                  // unused when is_prefill=1
            kv_stride: cfg.kv_dim,
            q_offset: model.act_layout.q,
            k_cache_offset: ((layer_offset_bytes / 4) as u32),
            v_cache_offset: ((layer_offset_bytes / 4) as u32),
            out_offset: model.act_layout.attn_out,
            scale: 1.0 / (cfg.head_dim as f32).sqrt(),
            m_tokens: m, is_prefill: 1,
        };
        let off = se.alloc_uniform(&p);
        let bg = make_bg(model, &model.bgls.attn, "attn_bg", std::mem::size_of::<AttnParams>() as u64,
                         &[&model.buffers.act, &model.buffers.kv_k, &model.buffers.kv_v]);
        let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("attn"), timestamp_writes: None });
        cp.set_pipeline(&model.pipes.attention);
        cp.set_bind_group(0, &bg, &[off]);
        cp.dispatch_workgroups(cfg.n_head, m, 1);
    }

    // 8. Quantize attn_out then Wo with residual add
    let (a_d2, a_qs2) = quantize_act(model, cfg, se, cfg.q_dim, m, model.act_layout.attn_out);
    let wo = tensor(cfg, &format!("blk.{il}.attn_output.weight")).clone();
    matmul_q1_0(model, cfg, se, cfg.q_dim, cfg.n_embd, m,
                &model.buffers.w_attn, wo.d_offset as u32, wo.qs_offset as u32,
                a_d2, a_qs2, model.act_layout.x, true);

    // 9. ffn_norm
    let fn_n = tensor(cfg, &format!("blk.{il}.ffn_norm.weight")).clone();
    rms_norm(model, cfg, se, m, cfg.n_embd,
             model.act_layout.x, model.act_layout.x_norm, (fn_n.offset / 4) as u32);

    // 10. Quantize x_norm again, then Wgate, Wup
    let (a_d3, a_qs3) = quantize_act(model, cfg, se, cfg.n_embd, m, model.act_layout.x_norm);
    let wg = tensor(cfg, &format!("blk.{il}.ffn_gate.weight")).clone();
    matmul_q1_0(model, cfg, se, cfg.n_embd, cfg.n_ff, m,
                &model.buffers.w_ffn_gu, wg.d_offset as u32, wg.qs_offset as u32,
                a_d3, a_qs3, model.act_layout.gate, false);
    let wu = tensor(cfg, &format!("blk.{il}.ffn_up.weight")).clone();
    matmul_q1_0(model, cfg, se, cfg.n_embd, cfg.n_ff, m,
                &model.buffers.w_ffn_gu, wu.d_offset as u32, wu.qs_offset as u32,
                a_d3, a_qs3, model.act_layout.up, false);

    // 11. SiLU(gate)*up
    silu_mul(model, se, cfg.n_ff, m, model.act_layout.gate, model.act_layout.up, model.act_layout.ffn_in);

    // 12. Quantize ffn_in then Wd with residual
    let (a_d4, a_qs4) = quantize_act(model, cfg, se, cfg.n_ff, m, model.act_layout.ffn_in);
    let wd = tensor(cfg, &format!("blk.{il}.ffn_down.weight")).clone();
    matmul_q1_0(model, cfg, se, cfg.n_ff, cfg.n_embd, m,
                &model.buffers.w_ffn_d, wd.d_offset as u32, wd.qs_offset as u32,
                a_d4, a_qs4, model.act_layout.x, true);
}

/// Layout (in bytes) of the q8 buffer: M * (K/32) FP32 d's, then M * K i8 qs's.
/// Returns (d_offset, qs_offset) in bytes.
fn quantize_act(model: &Model, _cfg: &Config, se: &mut StepEncoder,
                k: u32, m: u32, in_off_f32: u32) -> (u32, u32) {
    let nb_q8 = k / 32;
    let d_off  = 0u32;
    let qs_off = m * nb_q8 * 4;
    let total = m * nb_q8;
    let dispatch_x = total.min(65535);
    let dispatch_y = (total + dispatch_x - 1) / dispatch_x;
    let p = QuantParams {
        k, m, input_offset: in_off_f32,
        d_offset: d_off, qs_offset: qs_off,
        dispatch_x_dim: dispatch_x,
        ..Default::default()
    };
    let off = se.alloc_uniform(&p);
    let bg = make_bg(model, &model.bgls.quantize, "quant_bg", std::mem::size_of::<QuantParams>() as u64,
                     &[&model.buffers.act, &model.buffers.act_q8]);
    let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("quantize"), timestamp_writes: None });
    cp.set_pipeline(&model.pipes.quantize);
    cp.set_bind_group(0, &bg, &[off]);
    cp.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    (d_off, qs_off)
}

fn silu_mul(model: &Model, se: &mut StepEncoder,
            n: u32, m: u32, gate_off: u32, up_off: u32, out_off: u32) {
    let total = n * m;
    let groups = (total + 63) / 64;
    let dispatch_x = groups.min(65535);
    let dispatch_y = (groups + dispatch_x - 1) / dispatch_x;
    let p = SiluMulParams {
        n, m, gate_offset: gate_off, up_offset: up_off, out_offset: out_off,
        dispatch_x_count: dispatch_x * 64,
        ..Default::default()
    };
    let off = se.alloc_uniform(&p);
    let bg = make_bg(model, &model.bgls.silu_mul, "silu_bg", std::mem::size_of::<SiluMulParams>() as u64,
                     &[&model.buffers.act]);
    let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("silu_mul"), timestamp_writes: None });
    cp.set_pipeline(&model.pipes.silu_mul);
    cp.set_bind_group(0, &bg, &[off]);
    cp.dispatch_workgroups(dispatch_x, dispatch_y, 1);
}

fn matmul_q1_0(model: &Model, _cfg: &Config, se: &mut StepEncoder,
               k: u32, n: u32, m: u32,
               weights: &wgpu::Buffer, w_d: u32, w_qs: u32,
               a_d: u32, a_qs: u32, out_off: u32, accumulate: bool) {
    let p = MatmulParams {
        k, n, m,
        w_d_offset: w_d, w_qs_offset: w_qs,
        a_d_offset: a_d, a_qs_offset: a_qs,
        out_offset: out_off,
        accumulate: if accumulate { 1 } else { 0 },
        ..Default::default()
    };
    let off = se.alloc_uniform(&p);
    let bg = make_bg(model, &model.bgls.matmul, "matmul_bg", std::mem::size_of::<MatmulParams>() as u64,
                     &[weights, &model.buffers.act_q8, &model.buffers.act]);
    let mut cp = se.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("matmul"), timestamp_writes: None });
    cp.set_pipeline(&model.pipes.matmul);
    cp.set_bind_group(0, &bg, &[off]);
    cp.dispatch_workgroups((n + 63) / 64, (m + 63) / 64, 1);
}
