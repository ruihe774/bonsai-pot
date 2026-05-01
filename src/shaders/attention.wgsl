enable f16;

// Causal self-attention with one-pass online softmax (flash-attention-1 style).
// One workgroup per (token, head). GQA: Q-head h maps to KV-head h/(n_head/n_kv_head).
// dispatch: (n_head, m_tokens, 1).
//
// The outer scan over the K/V cache is unrolled by UNROLL=4: each iteration
// loads four cache positions worth of K, computes four Q·K dots packed into a
// single vec4<f32> reduction (shared barrier tree), and then runs four
// sequential online-softmax updates / V accumulations per reduction. This
// amortizes the workgroupBarrier cost (the dominant overhead at long context)
// and lets multiple K/V loads stay in flight per barrier wait.
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
const UNROLL: u32 = 4u;
var<workgroup> partial4: array<vec4<f32>, WG>;
var<workgroup> partial1: array<f32, WG>;

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
  let g_off = g * hd;

  // Per-thread output slice: element indices tid, tid+WG, ...
  var o: array<f32, ELEMS_PER_THREAD>;
  for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) { o[i] = 0.0; }

  var m_run: f32 = -1e30;
  var l_run: f32 = 0.0;

  let main_end = (cur_pos / UNROLL) * UNROLL;
  var t: u32 = 0u;

  // ---- Unrolled-by-4 main loop --------------------------------------------
  while (t < main_end) {
    // 1) Compute four Q·K dots (one per upcoming cache position) packed into
    //    a single vec4<f32> reduction.
    let k_base0 = p.k_cache_offset + (t + 0u) * p.kv_stride + g_off;
    let k_base1 = p.k_cache_offset + (t + 1u) * p.kv_stride + g_off;
    let k_base2 = p.k_cache_offset + (t + 2u) * p.kv_stride + g_off;
    let k_base3 = p.k_cache_offset + (t + 3u) * p.kv_stride + g_off;
    var local: vec4<f32> = vec4<f32>(0.0);
    for (var d: u32 = tid; d < hd; d += WG) {
      let q = f32(act[q_base + d]);
      let k0 = f32(k_cache[k_base0 + d]);
      let k1 = f32(k_cache[k_base1 + d]);
      let k2 = f32(k_cache[k_base2 + d]);
      let k3 = f32(k_cache[k_base3 + d]);
      local = local + q * vec4<f32>(k0, k1, k2, k3);
    }
    partial4[tid] = local;
    workgroupBarrier();
    var s: u32 = WG / 2u;
    while (s > 0u) {
      if (tid < s) { partial4[tid] = partial4[tid] + partial4[tid + s]; }
      workgroupBarrier();
      s = s / 2u;
    }
    let dots = partial4[0];
    // Required: prevent the next iteration from clobbering partial4[0] before
    // every thread has read it via this iteration's reduction.
    workgroupBarrier();

    // 2) Sequential online-softmax + V accumulation for the four positions.
    let scores = dots * p.scale;
    for (var i: u32 = 0u; i < UNROLL; i++) {
      let score = scores[i];
      let m_new = max(m_run, score);
      let correction = exp(m_run - m_new);
      let weight = exp(score - m_new);
      l_run = l_run * correction + weight;
      m_run = m_new;

      let v_base = p.v_cache_offset + (t + i) * p.kv_stride + g_off;
      for (var j: u32 = 0u; j < ELEMS_PER_THREAD; j++) {
        let d = tid + j * WG;
        if (d < hd) {
          o[j] = o[j] * correction + weight * f32(v_cache[v_base + d]);
        }
      }
    }

    t = t + UNROLL;
  }

  // ---- Tail loop (remaining < UNROLL positions) ---------------------------
  while (t < cur_pos) {
    let k_base = p.k_cache_offset + t * p.kv_stride + g_off;
    var local_dot: f32 = 0.0;
    for (var d: u32 = tid; d < hd; d += WG) {
      local_dot = local_dot + f32(act[q_base + d]) * f32(k_cache[k_base + d]);
    }
    partial1[tid] = local_dot;
    workgroupBarrier();
    var s: u32 = WG / 2u;
    while (s > 0u) {
      if (tid < s) { partial1[tid] = partial1[tid] + partial1[tid + s]; }
      workgroupBarrier();
      s = s / 2u;
    }
    let dot = partial1[0];
    workgroupBarrier();

    let score = dot * p.scale;
    let m_new = max(m_run, score);
    let correction = exp(m_run - m_new);
    let weight = exp(score - m_new);
    l_run = l_run * correction + weight;
    m_run = m_new;

    let v_base = p.v_cache_offset + t * p.kv_stride + g_off;
    for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) {
      let d = tid + i * WG;
      if (d < hd) {
        o[i] = o[i] * correction + weight * f32(v_cache[v_base + d]);
      }
    }
    t = t + 1u;
  }

  // ---- Normalize and write out --------------------------------------------
  let out_base = p.out_offset + m_tok * q_stride + h * hd;
  let inv_l = 1.0 / l_run;
  for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) {
    let d = tid + i * WG;
    if (d < hd) {
      act[out_base + d] = f16(o[i] * inv_l);
    }
  }
}
