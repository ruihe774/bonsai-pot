enable f16;

// Causal self-attention with one-pass online softmax (flash-attention-1 style).
// One workgroup per (token, head). GQA: Q-head h maps to KV-head h/(n_head/n_kv_head).
// dispatch: (n_head, m_tokens, 1).
//
// Single pass over K/V cache: for each cache position t, compute the Q·K dot,
// then update (m_run, l_run, o) all together. Avoids the second pass over K
// that the textbook two-pass online-softmax does.
//
// Per-thread output accumulator is `array<f32, ELEMS_PER_THREAD>` where each
// thread owns elements (tid, tid+WG, ...) of the head-dim-sized output vector.
// For Bonsai-4B (head_dim=128, WG=64), ELEMS_PER_THREAD = 2.

struct Params {
  head_dim: u32,
  n_head: u32,
  n_kv_head: u32,
  pos: u32,
  kv_stride: u32,
  q_offset: u32,
  k_cache_offset: u32,
  v_cache_offset: u32,
  out_offset: u32,
  scale: f32,
  m_tokens: u32,
  is_prefill: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> act: array<f16>;
@group(0) @binding(2) var<storage, read> k_cache: array<f16>;
@group(0) @binding(3) var<storage, read> v_cache: array<f16>;

const WG: u32 = 64u;
const ELEMS_PER_THREAD: u32 = 2u;  // head_dim (128) / WG (64)
var<workgroup> partial: array<f32, WG>;

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

  // Per-thread output slice: element indices tid, tid+WG, ...
  var o: array<f32, ELEMS_PER_THREAD>;
  for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) { o[i] = 0.0; }

  var m_run: f32 = -1e30;
  var l_run: f32 = 0.0;

  for (var t: u32 = 0u; t < cur_pos; t++) {
    // 1) Q · K[t] cooperative reduction → partial[0]
    let k_base = p.k_cache_offset + t * p.kv_stride + g * hd;
    var local_dot: f32 = 0.0;
    for (var d: u32 = tid; d < hd; d += WG) {
      local_dot = local_dot + f32(act[q_base + d]) * f32(k_cache[k_base + d]);
    }
    partial[tid] = local_dot;
    workgroupBarrier();
    var s: u32 = WG / 2u;
    while (s > 0u) {
      if (tid < s) { partial[tid] = partial[tid] + partial[tid + s]; }
      workgroupBarrier();
      s = s / 2u;
    }
    let dot = partial[0];
    // Required: without this, thread 0 could write partial[0] = local_dot in
    // the next iteration before another thread reads partial[0] in this one.
    workgroupBarrier();

    // 2) Online softmax update + V accumulation
    let score = dot * p.scale;
    let m_new = max(m_run, score);
    let correction = exp(m_run - m_new);
    let weight = exp(score - m_new);
    l_run = l_run * correction + weight;
    m_run = m_new;

    let v_base = p.v_cache_offset + t * p.kv_stride + g * hd;
    for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) {
      let d = tid + i * WG;
      if (d < hd) {
        o[i] = o[i] * correction + weight * f32(v_cache[v_base + d]);
      }
    }
  }

  // 3) Normalize and write out
  let out_base = p.out_offset + m_tok * q_stride + h * hd;
  let inv_l = 1.0 / l_run;
  for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) {
    let d = tid + i * WG;
    if (d < hd) {
      act[out_base + d] = f16(o[i] * inv_l);
    }
  }
}
