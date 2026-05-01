enable f16;

// Split-K + GQA-batched attention chunk pass for the matvec (m_tokens=1) path.
//
// dispatch: (n_kv_head, n_chunks_active, 1).
//
// Each workgroup handles one (kv_group, chunk) pair, scanning the cache slice
// [chunk * CHUNK_SIZE, min((chunk+1) * CHUNK_SIZE, pos)) and producing partial
// (m, l, o) state for all Q_PER_GROUP=4 Q heads that share this KV head. K and
// V are loaded once per cache position and reused across the four Q heads.
// The four Q·K dots are computed in parallel via vec4<f32> packing and reduced
// in a single `subgroupAdd` (assumes subgroup_size == workgroup_size == 64,
// i.e. RDNA wave64). No workgroup barriers in the inner loop.
//
// K/V cache is Q8_0: per 32 contiguous elements of the kv-row we have one
// FP32 scale (in the d-section) and 32 packed i8 values (in the qs-section).
// The two sections live in the SAME u32 storage buffer; layer-base offsets
// are passed in as `k_d_word_offset` (u32-word offset, FP32 scales) and
// `k_qs_byte_offset` (byte offset, i8 qs).
//
// Output: partials[h_global, chunk] = (o[head_dim], m, l) — 130 f32s per
// (head, chunk). Stride in the chunk dim is `n_chunks_active`, set by host.

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
  n_chunks_active: u32,
  scale: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> act: array<f16>;
@group(0) @binding(2) var<storage, read> k_cache: array<u32>;
@group(0) @binding(3) var<storage, read> v_cache: array<u32>;
@group(0) @binding(4) var<storage, read_write> partials: array<f32>;

const SG_SIZE: u32 = {{SG_SIZE}}u;
const WG: u32 = 64u;
const N_SG: u32 = WG / SG_SIZE;
const Q_PER_GROUP: u32 = 4u;
const ELEMS_PER_THREAD: u32 = 2u;  // head_dim (128) / WG (64)
const CHUNK_SIZE: u32 = 32u;
const PARTIAL_STRIDE: u32 = 130u;  // head_dim + 2

// Cross-subgroup partial slots; unused (and dead-coded out) when N_SG == 1.
var<workgroup> sg_partial4: array<vec4<f32>, N_SG>;

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

// Load and dequantize one element from a Q8_0 cache buffer.
//   buf:           &k_cache or &v_cache (both array<u32> in this BG)
//   d_word_off:    u32-word offset of the layer's d-section start
//   qs_byte_off:   byte offset of the layer's qs-section start
//   t:             cache position (in elements within the layer's row-grid)
//   kv_stride:     elements per row (= kv_dim)
//   e_local:       element index within the row (= g_off + d)
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
  let g = wg.x;
  let chunk = wg.y;
  if (g >= p.n_kv_head || chunk >= p.n_chunks_active) { return; }
  let tid = lid.x;
  let hd = p.head_dim;
  let chunk_start = chunk * CHUNK_SIZE;
  let chunk_end = min(chunk_start + CHUNK_SIZE, p.pos);
  let g_off = g * hd;

  let q_base0 = p.q_offset + (g * Q_PER_GROUP + 0u) * hd;
  let q_base1 = p.q_offset + (g * Q_PER_GROUP + 1u) * hd;
  let q_base2 = p.q_offset + (g * Q_PER_GROUP + 2u) * hd;
  let q_base3 = p.q_offset + (g * Q_PER_GROUP + 3u) * hd;

  // Per-thread V accumulators for each Q head in the group.
  var o0: array<f32, ELEMS_PER_THREAD>;
  var o1: array<f32, ELEMS_PER_THREAD>;
  var o2: array<f32, ELEMS_PER_THREAD>;
  var o3: array<f32, ELEMS_PER_THREAD>;
  for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) {
    o0[i] = 0.0; o1[i] = 0.0; o2[i] = 0.0; o3[i] = 0.0;
  }
  var m: vec4<f32> = vec4<f32>(-1e30);
  var l: vec4<f32> = vec4<f32>(0.0);

  for (var t: u32 = chunk_start; t < chunk_end; t++) {
    // Load K[t] once, broadcast against all four Qs.
    var local: vec4<f32> = vec4<f32>(0.0);
    for (var d: u32 = tid; d < hd; d += WG) {
      let k = load_k(t, g_off + d);
      let q0 = f32(act[q_base0 + d]);
      let q1 = f32(act[q_base1 + d]);
      let q2 = f32(act[q_base2 + d]);
      let q3 = f32(act[q_base3 + d]);
      local = local + k * vec4<f32>(q0, q1, q2, q3);
    }
    // Workgroup-wide reduction (one subgroupAdd if SG covers WG, else two-step).
    let dots = wg_sum_v4(local, tid, sg_inv_id);
    let scores = dots * p.scale;

    let m_new = max(m, scores);
    let correction = exp(m - m_new);
    let weight = exp(scores - m_new);
    l = l * correction + weight;
    m = m_new;

    for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) {
      let d = tid + i * WG;
      if (d < hd) {
        let v = load_v(t, g_off + d);
        o0[i] = o0[i] * correction.x + weight.x * v;
        o1[i] = o1[i] * correction.y + weight.y * v;
        o2[i] = o2[i] * correction.z + weight.z * v;
        o3[i] = o3[i] * correction.w + weight.w * v;
      }
    }
  }

  // Write per-Q-head partials. Layout stride is n_chunks_active (per call).
  let h0 = g * Q_PER_GROUP + 0u;
  let h1 = g * Q_PER_GROUP + 1u;
  let h2 = g * Q_PER_GROUP + 2u;
  let h3 = g * Q_PER_GROUP + 3u;
  let stride_h = p.n_chunks_active * PARTIAL_STRIDE;
  let base0 = h0 * stride_h + chunk * PARTIAL_STRIDE;
  let base1 = h1 * stride_h + chunk * PARTIAL_STRIDE;
  let base2 = h2 * stride_h + chunk * PARTIAL_STRIDE;
  let base3 = h3 * stride_h + chunk * PARTIAL_STRIDE;
  for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) {
    let d = tid + i * WG;
    if (d < hd) {
      partials[base0 + d] = o0[i];
      partials[base1 + d] = o1[i];
      partials[base2 + d] = o2[i];
      partials[base3 + d] = o3[i];
    }
  }
  if (tid == 0u) {
    partials[base0 + hd] = m.x;
    partials[base0 + hd + 1u] = l.x;
    partials[base1 + hd] = m.y;
    partials[base1 + hd + 1u] = l.y;
    partials[base2 + hd] = m.z;
    partials[base2 + hd + 1u] = l.z;
    partials[base3 + hd] = m.w;
    partials[base3 + hd + 1u] = l.w;
  }
}
