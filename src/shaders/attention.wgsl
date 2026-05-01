// Causal self-attention with online softmax. One workgroup per (token, head).
// GQA: Q-head h maps to KV-head h / (n_head / n_kv_head).
// dispatch: (n_head, m_tokens, 1)
//
// Layout assumptions for batched prefill (m_tokens > 1):
//   q_offset / out_offset point to [m_tokens][n_head * head_dim] flat regions.
//   k_cache / v_cache: per-layer slab [max_seq][kv_dim]; this layer's k/v for
//   token t lives at offset (k_cache_offset + t * kv_stride).
//
// For the single-token path (m_tokens == 1), pos = absolute position + 1
// (i.e., causal scan over the whole cache up through `pos - 1`).
// For batched prefill, each token computes its own causal scan: token m sees
// keys [0, m] (i.e., pos = m + 1), assuming the cache was filled in order.

struct Params {
  head_dim: u32,
  n_head: u32,
  n_kv_head: u32,
  pos: u32,           // absolute end position for the token-0 of this dispatch
  kv_stride: u32,
  q_offset: u32,
  k_cache_offset: u32,
  v_cache_offset: u32,
  out_offset: u32,
  scale: f32,
  m_tokens: u32,      // number of tokens in this dispatch (1 for gen, M for prefill)
  is_prefill: u32,    // 1 -> per-token causal pos = m_tok + 1; 0 -> use p.pos
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> act: array<f32>;
@group(0) @binding(2) var<storage, read> k_cache: array<f32>;
@group(0) @binding(3) var<storage, read> v_cache: array<f32>;

const WG: u32 = 64u;
var<workgroup> partial: array<f32, WG>;

fn dot_q_k(q_base: u32, k_base: u32, hd: u32, tid: u32) -> f32 {
  var local_dot: f32 = 0.0;
  for (var d: u32 = tid; d < hd; d += WG) {
    local_dot = local_dot + act[q_base + d] * k_cache[k_base + d];
  }
  partial[tid] = local_dot;
  workgroupBarrier();
  var s: u32 = WG / 2u;
  while (s > 0u) {
    if (tid < s) { partial[tid] = partial[tid] + partial[tid + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  return partial[0];
}

@compute @workgroup_size(WG)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let h = wg.x;
  let m_tok = wg.y;
  if (h >= p.n_head || m_tok >= p.m_tokens) { return; }
  let g = h / (p.n_head / p.n_kv_head);
  let tid = lid.x;
  let hd = p.head_dim;
  let cur_pos = select(p.pos, m_tok + 1u, p.is_prefill != 0u);
  let q_stride = p.n_head * hd;
  let q_base = p.q_offset + m_tok * q_stride + h * hd;

  var m_run: f32 = -1e30;
  var l_run: f32 = 0.0;
  for (var t: u32 = 0u; t < cur_pos; t++) {
    let k_base = p.k_cache_offset + t * p.kv_stride + g * hd;
    let dot = dot_q_k(q_base, k_base, hd, tid);
    let score = dot * p.scale;
    let m_new = max(m_run, score);
    l_run = l_run * exp(m_run - m_new) + exp(score - m_new);
    m_run = m_new;
  }

  let out_base = p.out_offset + m_tok * q_stride + h * hd;
  for (var d: u32 = tid; d < hd; d += WG) {
    act[out_base + d] = 0.0;
  }
  workgroupBarrier();
  for (var t: u32 = 0u; t < cur_pos; t++) {
    let k_base = p.k_cache_offset + t * p.kv_stride + g * hd;
    let dot = dot_q_k(q_base, k_base, hd, tid);
    let score = dot * p.scale;
    let w_attn = exp(score - m_run) / l_run;
    let v_base = p.v_cache_offset + t * p.kv_stride + g * hd;
    for (var d: u32 = tid; d < hd; d += WG) {
      act[out_base + d] = act[out_base + d] + w_attn * v_cache[v_base + d];
    }
    workgroupBarrier();
  }
}
