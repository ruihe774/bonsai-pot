enable f16;

// Causal self-attention with one-pass online softmax (flash-attention-1 style).
// One workgroup per (token, head). GQA: Q-head h maps to KV-head h/(n_head/n_kv_head).
// dispatch: (n_head, m_tokens, 1).
//
// The outer scan over the K/V cache is unrolled by UNROLL=4: each iteration
// loads four cache positions worth of K, computes four Q·K dots packed into
// one vec4<f32> reduced via a single subgroupAdd (with a templated cross-
// subgroup merge for SG_SIZE < WG), and then runs four sequential online-
// softmax updates / V accumulations per reduction. This amortizes the
// per-iteration reduction cost and keeps multiple K/V loads in flight.
//
// K/V cache is Q8_0: per 32 contiguous elements of the kv-row we have one
// FP32 scale (in the d-section) and 32 packed i8 values (in the qs-section).
// Both sections live in the SAME u32 storage buffer; layer-base offsets are
// passed in (`k_d_word_offset` / `k_qs_byte_offset` / `v_*`).
//
// Per-thread output accumulator is `array<f32, ELEMS_PER_THREAD>` where each
// thread owns elements (tid, tid+WG, ...) of the head-dim-sized output vector.
// For the Bonsai family (head_dim=128, WG=64), ELEMS_PER_THREAD = 2.
//
// Invariant: head_dim == ELEMS_PER_THREAD * WG. The per-element loops below
// don't bounds-check `d < hd` because every (tid, i) pair indexes a real
// element under this invariant. Models with a different head_dim would need
// ELEMS_PER_THREAD retemplated to match.

struct Params {
  head_dim: u32,
  n_head: u32,
  n_kv_head: u32,
  pos: u32,
  kv_stride: u32,
  q_offset: u32,
  k_d_word_offset: u32,
  k_qs_byte_offset: u32,
  v_d_word_offset: u32,
  v_qs_byte_offset: u32,
  out_offset: u32,
  scale: f32,
  m_tokens: u32,
  is_prefill: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> act: array<f16>;
@group(0) @binding(2) var<storage, read> k_cache: array<u32>;
@group(0) @binding(3) var<storage, read> v_cache: array<u32>;

const SG_SIZE: u32 = {{SG_SIZE}}u;
const WG: u32 = 64u;
const N_SG: u32 = WG / SG_SIZE;
const ELEMS_PER_THREAD: u32 = 2u;  // head_dim (128) / WG (64)
const UNROLL: u32 = 4u;

// Cross-subgroup partial slots; unused (and dead-coded out) when N_SG == 1.
var<workgroup> sg_partial4: array<vec4<f32>, N_SG>;
var<workgroup> sg_partial1: array<f32, N_SG>;

fn wg_sum_v4(local: vec4<f32>, tid: u32, sg_inv_id: u32) -> vec4<f32> {
  let sg_sum = subgroupAdd(local);
  if (N_SG == 1u) {
    return sg_sum;
  }
  let sg_id = tid / SG_SIZE;
  if (sg_inv_id == 0u) { sg_partial4[sg_id] = sg_sum; }
  workgroupBarrier();
  if (sg_id == 0u) {
    var combined = vec4<f32>(0.0);
    if (sg_inv_id < N_SG) { combined = sg_partial4[sg_inv_id]; }
    let final_sum = subgroupAdd(combined);
    if (sg_inv_id == 0u) { sg_partial4[0] = final_sum; }
  }
  workgroupBarrier();
  return sg_partial4[0];
}

fn wg_sum_f32(local: f32, tid: u32, sg_inv_id: u32) -> f32 {
  let sg_sum = subgroupAdd(local);
  if (N_SG == 1u) {
    return sg_sum;
  }
  let sg_id = tid / SG_SIZE;
  if (sg_inv_id == 0u) { sg_partial1[sg_id] = sg_sum; }
  workgroupBarrier();
  if (sg_id == 0u) {
    var combined: f32;
    if (sg_inv_id < N_SG) { combined = sg_partial1[sg_inv_id]; }
    let final_sum = subgroupAdd(combined);
    if (sg_inv_id == 0u) { sg_partial1[0] = final_sum; }
  }
  workgroupBarrier();
  return sg_partial1[0];
}

// Dequant load helpers — see attention_split.wgsl for layout notes.
fn load_k(t: u32, e_local: u32) -> f32 {
  let elem_idx = t * p.kv_stride + e_local;
  let block_idx = elem_idx >> 5u;
  let scale = bitcast<f32>(k_cache[p.k_d_word_offset + block_idx]);
  let qs_byte_idx = p.k_qs_byte_offset + elem_idx;
  let qs_word = k_cache[qs_byte_idx >> 2u];
  let shift = (qs_byte_idx & 3u) << 3u;
  let qs_byte = (qs_word >> shift) & 0xFFu;
  let qs_signed = bitcast<i32>(qs_byte << 24u) >> 24u;
  return scale * f32(qs_signed);
}

fn load_v(t: u32, e_local: u32) -> f32 {
  let elem_idx = t * p.kv_stride + e_local;
  let block_idx = elem_idx >> 5u;
  let scale = bitcast<f32>(v_cache[p.v_d_word_offset + block_idx]);
  let qs_byte_idx = p.v_qs_byte_offset + elem_idx;
  let qs_word = v_cache[qs_byte_idx >> 2u];
  let shift = (qs_byte_idx & 3u) << 3u;
  let qs_byte = (qs_word >> shift) & 0xFFu;
  let qs_signed = bitcast<i32>(qs_byte << 24u) >> 24u;
  return scale * f32(qs_signed);
}

@compute @workgroup_size(WG)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_inv_id: u32,
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

  var m_run: f32 = -1e30;
  var l_run: f32;

  let main_end = (cur_pos / UNROLL) * UNROLL;
  var t: u32 = 0u;

  // ---- Unrolled-by-4 main loop --------------------------------------------
  while (t < main_end) {
    // 1) Compute four Q·K dots (one per upcoming cache position) packed into
    //    a single vec4<f32> reduction.
    var local: vec4<f32> = vec4<f32>(0.0);
    for (var d: u32 = tid; d < hd; d += WG) {
      let q = f32(act[q_base + d]);
      let e_local = g_off + d;
      let k0 = load_k(t + 0u, e_local);
      let k1 = load_k(t + 1u, e_local);
      let k2 = load_k(t + 2u, e_local);
      let k3 = load_k(t + 3u, e_local);
      local = local + q * vec4<f32>(k0, k1, k2, k3);
    }
    // Workgroup-wide reduction (one subgroupAdd if SG covers WG, else two-step).
    let dots = wg_sum_v4(local, tid, sg_inv_id);

    // 2) Sequential online-softmax + V accumulation for the four positions.
    let scores = dots * p.scale;
    for (var i: u32 = 0u; i < UNROLL; i++) {
      let score = scores[i];
      let m_new = max(m_run, score);
      let correction = exp(m_run - m_new);
      let weight = exp(score - m_new);
      l_run = l_run * correction + weight;
      m_run = m_new;

      for (var j: u32 = 0u; j < ELEMS_PER_THREAD; j++) {
        let d = tid + j * WG;
        let v = load_v(t + i, g_off + d);
        o[j] = o[j] * correction + weight * v;
      }
    }

    t = t + UNROLL;
  }

  // ---- Tail loop (remaining < UNROLL positions) ---------------------------
  while (t < cur_pos) {
    var local_dot: f32 = 0.0;
    for (var d: u32 = tid; d < hd; d += WG) {
      local_dot = local_dot + f32(act[q_base + d]) * load_k(t, g_off + d);
    }
    let dot = wg_sum_f32(local_dot, tid, sg_inv_id);

    let score = dot * p.scale;
    let m_new = max(m_run, score);
    let correction = exp(m_run - m_new);
    let weight = exp(score - m_new);
    l_run = l_run * correction + weight;
    m_run = m_new;

    for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) {
      let d = tid + i * WG;
      let v = load_v(t, g_off + d);
      o[i] = o[i] * correction + weight * v;
    }
    t = t + 1u;
  }

  // ---- Normalize and write out --------------------------------------------
  let out_base = p.out_offset + m_tok * q_stride + h * hd;
  let inv_l = 1.0 / l_run;
  for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) {
    let d = tid + i * WG;
    act[out_base + d] = f16(o[i] * inv_l);
  }
}
