enable f16;

// Fused per-layer K-side pre-KV pipeline:
//   rms_norm(K head) -> *w_k_norm -> NEOX-RoPE -> Q8_0 quantize -> write kv_k.
// V-side runs in the same workgroup: Q8_0 quantize + write kv_v (no rms, no rope).
// Replaces a 3-dispatch sequence (rms_norm K, rope K, kv_writeback) with 1.
//
// Workgroup = head_dim (128). One workgroup per (kv_head, token) writes one
// head's worth of K and V into the cache. Dispatch (n_kv_head, m_tokens, 1).
//
// Cache layout (per kv_k / kv_v buffer), unchanged:
//   d-section  (FP32 scales): bytes [0, n_layer * max_seq * (kv_dim/32) * 4)
//   qs-section (i8 packed):   bytes [d_total, d_total + n_layer * max_seq * kv_dim)

struct Params {
  k_cur_off: u32,            // f16 element offset of K_cur in act
  v_cur_off: u32,            // f16 element offset of V_cur in act
  w_k_norm_off: u32,         // f16 element offset of w_k_norm[0..head_dim] in w_norms
  rope_offset: u32,          // f16 element base offset into rope_cs (typically 0)
  dst_d_word_offset: u32,    // u32-word offset into kv_{k,v} d-section, layer base
  dst_qs_byte_offset: u32,   // byte offset into kv_{k,v} qs-section, layer base
  pos_base: u32,             // first absolute cache position to write into
  kv_dim: u32,               // = n_kv_head * head_dim
  nb_per_row: u32,           // = kv_dim / 32
  eps: f32,
  _p0: u32, _p1: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       act: array<f16>;
@group(0) @binding(2) var<storage, read>       w_norms: array<f16>;
@group(0) @binding(3) var<storage, read>       rope_cs: array<f16>;
@group(0) @binding(4) var<storage, read_write> kv_k: array<u32>;
@group(0) @binding(5) var<storage, read_write> kv_v: array<u32>;

const HEAD_DIM: u32 = 128u;
const HALF_DIM: u32 = 64u;
const NB_PER_HEAD: u32 = HEAD_DIM / 32u;
const SUBGROUP_MIN_SIZE: u32 = {{SUBGROUP_MIN_SIZE}}u;
const WG: u32 = 128u;
const SG_PARTIAL_MAX: u32 = (WG + SUBGROUP_MIN_SIZE - 1u) / SUBGROUP_MIN_SIZE;

// Two scratch arrays: f32 for normed/RoPE'd values + amax tree, u32 for byte
// packing. Reuse k_sh / v_sh across phases (rms read -> rope -> amax).
var<workgroup> k_sh: array<f32, 128>;
var<workgroup> v_sh: array<f32, 128>;
var<workgroup> qk_sh: array<u32, 128>;
var<workgroup> qv_sh: array<u32, 128>;
var<workgroup> sg_partial: array<f32, SG_PARTIAL_MAX>;

@compute @workgroup_size(WG)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_inv_id: u32,
  @builtin(subgroup_id) sg_id: u32,
  @builtin(num_subgroups) num_subgroups: u32,
) {
  let head = wg.x;
  let tok  = wg.y;
  let tid  = lid.x;

  let kv_token_off = tok * p.kv_dim + head * HEAD_DIM;
  let k_raw = f32(act[p.k_cur_off + kv_token_off + tid]);
  let v_raw = f32(act[p.v_cur_off + kv_token_off + tid]);

  // ---- RMS-norm reduction over the K head (sum of squares) ----------------
  let sg_sum = subgroupAdd(k_raw * k_raw);
  var total: f32;
  if (num_subgroups == 1u) {
    total = sg_sum;
  } else {
    if (sg_inv_id == 0u) { sg_partial[sg_id] = sg_sum; }
    workgroupBarrier();
    if (sg_id == 0u) {
      var c: f32;
      if (sg_inv_id < num_subgroups) { c = sg_partial[sg_inv_id]; }
      let f = subgroupAdd(c);
      if (sg_inv_id == 0u) { sg_partial[0] = f; }
    }
    workgroupBarrier();
    total = sg_partial[0];
  }
  let inv_h = inverseSqrt(total / f32(HEAD_DIM) + p.eps);

  // ---- norm * weight, stash to shmem so RoPE can swap pairs ---------------
  k_sh[tid] = k_raw * inv_h * f32(w_norms[p.w_k_norm_off + tid]);
  workgroupBarrier();

  // ---- NEOX RoPE: rotate (j, j + half_dim) pairs --------------------------
  let pos_abs = p.pos_base + tok;
  let cs_base = p.rope_offset + pos_abs * HEAD_DIM;
  if (tid < HALF_DIM) {
    let c = f32(rope_cs[cs_base + tid * 2u]);
    let s = f32(rope_cs[cs_base + tid * 2u + 1u]);
    let x0 = k_sh[tid];
    let x1 = k_sh[tid + HALF_DIM];
    k_sh[tid]            = x0 * c - x1 * s;
    k_sh[tid + HALF_DIM] = x0 * s + x1 * c;
  }
  workgroupBarrier();
  let k_post = k_sh[tid];

  // ---- per-32-block amax for K and V (shmem tree, 5 levels) ---------------
  // Reuse k_sh / v_sh: now holds abs(value) for amax reduction.
  k_sh[tid] = abs(k_post);
  v_sh[tid] = abs(v_raw);
  workgroupBarrier();

  let lane32 = tid & 31u;
  if (lane32 < 16u) {
    k_sh[tid] = max(k_sh[tid], k_sh[tid + 16u]);
    v_sh[tid] = max(v_sh[tid], v_sh[tid + 16u]);
  }
  workgroupBarrier();
  if (lane32 < 8u) {
    k_sh[tid] = max(k_sh[tid], k_sh[tid + 8u]);
    v_sh[tid] = max(v_sh[tid], v_sh[tid + 8u]);
  }
  workgroupBarrier();
  if (lane32 < 4u) {
    k_sh[tid] = max(k_sh[tid], k_sh[tid + 4u]);
    v_sh[tid] = max(v_sh[tid], v_sh[tid + 4u]);
  }
  workgroupBarrier();
  if (lane32 < 2u) {
    k_sh[tid] = max(k_sh[tid], k_sh[tid + 2u]);
    v_sh[tid] = max(v_sh[tid], v_sh[tid + 2u]);
  }
  workgroupBarrier();
  if (lane32 < 1u) {
    k_sh[tid] = max(k_sh[tid], k_sh[tid + 1u]);
    v_sh[tid] = max(v_sh[tid], v_sh[tid + 1u]);
  }
  workgroupBarrier();

  let block_in_head = tid >> 5u;          // 0..3
  let block_base    = block_in_head * 32u;
  let amax_k = k_sh[block_base];
  let amax_v = v_sh[block_base];
  let dk = amax_k / 127.0;
  let dv = amax_v / 127.0;
  let inv_dk = select(0.0, 1.0 / dk, dk > 0.0);
  let inv_dv = select(0.0, 1.0 / dv, dv > 0.0);

  let qk = u32(i32(clamp(round(k_post * inv_dk), -127.0, 127.0))) & 0xFFu;
  let qv = u32(i32(clamp(round(v_raw  * inv_dv), -127.0, 127.0))) & 0xFFu;

  // ---- pack 4 bytes into one u32 via shmem, write to cache ----------------
  qk_sh[tid] = qk;
  qv_sh[tid] = qv;
  workgroupBarrier();

  // qs base byte for THIS head's elements within the layer's qs-section:
  //   layer_base + pos_abs*kv_dim + head*HEAD_DIM (bytes; one byte per elem).
  // Lanes (tid % 4 == 0) emit one packed u32 each (32 emitters per head).
  if ((tid & 3u) == 0u) {
    let pk = qk_sh[tid + 0u]
           | (qk_sh[tid + 1u] <<  8u)
           | (qk_sh[tid + 2u] << 16u)
           | (qk_sh[tid + 3u] << 24u);
    let pv = qv_sh[tid + 0u]
           | (qv_sh[tid + 1u] <<  8u)
           | (qv_sh[tid + 2u] << 16u)
           | (qv_sh[tid + 3u] << 24u);
    let qs_byte_in_layer =
      p.dst_qs_byte_offset + pos_abs * p.kv_dim + head * HEAD_DIM + tid;
    kv_k[qs_byte_in_layer >> 2u] = pk;
    kv_v[qs_byte_in_layer >> 2u] = pv;
  }

  // d (FP32 scale) — one writer per block.
  if (lane32 == 0u) {
    let block_global =
      pos_abs * p.nb_per_row + head * NB_PER_HEAD + block_in_head;
    kv_k[p.dst_d_word_offset + block_global] = bitcast<u32>(dk);
    kv_v[p.dst_d_word_offset + block_global] = bitcast<u32>(dv);
  }
}
