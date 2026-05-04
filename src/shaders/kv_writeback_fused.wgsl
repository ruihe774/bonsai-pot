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

// Only shmem live across barriers. k_sh holds normed-and-weighted K so RoPE
// can fetch each thread's pair partner across the head-half boundary
// (partners always cross a subgroup boundary at WG=128). After RoPE the
// rotated value lives in a register; k_sh is dead and not reused.
var<workgroup> k_sh: array<f32, 128>;
// Used by the RMS cross-subgroup merge (slot 0..num_subgroups; only .x), and
// when SUBGROUP_MIN_SIZE < 32 also by the amax cross-cluster merge that
// finishes the per-32 reduction (writes both .x=K, .y=V).
var<workgroup> sg_partial: array<vec2<f32>, SG_PARTIAL_MAX>;

@compute @workgroup_size(WG)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_inv_id: u32,
  @builtin(subgroup_id) sg_id: u32,
  @builtin(num_subgroups) num_subgroups: u32,
  @builtin(subgroup_size) sg_size: u32,
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
    if (sg_inv_id == 0u) { sg_partial[sg_id].x = sg_sum; }
    workgroupBarrier();
    if (sg_id == 0u) {
      var c: f32 = 0.0;
      if (sg_inv_id < num_subgroups) { c = sg_partial[sg_inv_id].x; }
      let f = subgroupAdd(c);
      if (sg_inv_id == 0u) { sg_partial[0].x = f; }
    }
    workgroupBarrier();
    total = sg_partial[0].x;
  }
  let inv_h = inverseSqrt(total / f32(HEAD_DIM) + p.eps);

  // ---- norm * weight, stash to shmem so RoPE can swap pairs ---------------
  k_sh[tid] = k_raw * inv_h * f32(w_norms[p.w_k_norm_off + tid]);
  workgroupBarrier();

  // ---- NEOX RoPE: each lane computes its own output, kept in a register ---
  // Lower-half (tid in [0, 64)) emits the cosθ·x0 - sinθ·x1 component; the
  // upper-half emits sinθ·x0 + cosθ·x1 with the same (c, s) for j = tid - 64.
  // No second shmem round-trip / barrier; k_sh is dead from here on.
  let pos_abs = p.pos_base + tok;
  let cs_base = p.rope_offset + pos_abs * HEAD_DIM;
  var k_post: f32;
  if (tid < HALF_DIM) {
    let c = f32(rope_cs[cs_base + tid * 2u]);
    let s = f32(rope_cs[cs_base + tid * 2u + 1u]);
    k_post = k_sh[tid] * c - k_sh[tid + HALF_DIM] * s;
  } else {
    let j = tid - HALF_DIM;
    let c = f32(rope_cs[cs_base + j * 2u]);
    let s = f32(rope_cs[cs_base + j * 2u + 1u]);
    k_post = k_sh[j] * s + k_sh[tid] * c;
  }

  // ---- Per-32-block amax for K and V via subgroup butterfly ---------------
  // Replaces the 5-level shmem tree. Masks 1/2/4 are always safe
  // (SUBGROUP_MIN_SIZE >= 8 is enforced at load time); 8/16 are gated by the
  // larger of the baked min size and the runtime subgroup size, so AMD/NVIDIA
  // (min >= 32) const-fold to the full 32-wide butterfly while Intel-class
  // hardware (min=8) still picks up the wider butterfly when the actual size
  // happens to be 16 or 32. The stitch path is taken only when no cluster
  // covers a full 32-block; clusters there are sized by the runtime sg_size.
  var ak = abs(k_post);
  var av = abs(v_raw);

  ak = max(ak, subgroupShuffleXor(ak, 1u));
  av = max(av, subgroupShuffleXor(av, 1u));
  ak = max(ak, subgroupShuffleXor(ak, 2u));
  av = max(av, subgroupShuffleXor(av, 2u));
  ak = max(ak, subgroupShuffleXor(ak, 4u));
  av = max(av, subgroupShuffleXor(av, 4u));
  if (SUBGROUP_MIN_SIZE >= 16u || sg_size >= 16u) {
    ak = max(ak, subgroupShuffleXor(ak, 8u));
    av = max(av, subgroupShuffleXor(av, 8u));
  }
  if (SUBGROUP_MIN_SIZE >= 32u || sg_size >= 32u) {
    ak = max(ak, subgroupShuffleXor(ak, 16u));
    av = max(av, subgroupShuffleXor(av, 16u));
  }
  if (SUBGROUP_MIN_SIZE < 32u && sg_size < 32u) {
    // Stitch the in-subgroup partials into per-32 blocks. Each cluster of
    // sg_size lanes contributes one entry; a 32-block spans 32/sg_size
    // entries (2 for sg_size=16, 4 for sg_size=8).
    let sg_lane = tid % sg_size;
    let cluster_id = tid / sg_size;
    if (sg_lane == 0u) {
      sg_partial[cluster_id] = vec2<f32>(ak, av);
    }
    workgroupBarrier();
    let group_id = tid >> 5u;
    let per_group = 32u / sg_size;
    let g_base = group_id * per_group;
    var mk: f32 = 0.0;
    var mv: f32 = 0.0;
    for (var i: u32 = 0u; i < per_group; i = i + 1u) {
      let m = sg_partial[g_base + i];
      mk = max(mk, m.x);
      mv = max(mv, m.y);
    }
    ak = mk;
    av = mv;
  }

  let block_in_head = tid >> 5u;          // 0..3
  let dk = ak / 127.0;
  let dv = av / 127.0;
  let inv_dk = select(0.0, 1.0 / dk, dk > 0.0);
  let inv_dv = select(0.0, 1.0 / dv, dv > 0.0);

  let qk = u32(i32(clamp(round(k_post * inv_dk), -127.0, 127.0))) & 0xFFu;
  let qv = u32(i32(clamp(round(v_raw  * inv_dv), -127.0, 127.0))) & 0xFFu;

  // ---- Pack 4 bytes into one u32 via subgroup shuffle, write to cache -----
  // 4-lane clusters (tid % 4 == 0 lanes are the leaders) live entirely
  // within one subgroup since SUBGROUP_MIN_SIZE >= 8, so subgroupShuffle can
  // gather the partner bytes without going through shmem. Shuffles are
  // unconditional (uniform CF requirement); only the leader writes.
  let cluster_base = sg_inv_id & ~3u;
  let qk0 = subgroupShuffle(qk, cluster_base);
  let qk1 = subgroupShuffle(qk, cluster_base + 1u);
  let qk2 = subgroupShuffle(qk, cluster_base + 2u);
  let qk3 = subgroupShuffle(qk, cluster_base + 3u);
  let qv0 = subgroupShuffle(qv, cluster_base);
  let qv1 = subgroupShuffle(qv, cluster_base + 1u);
  let qv2 = subgroupShuffle(qv, cluster_base + 2u);
  let qv3 = subgroupShuffle(qv, cluster_base + 3u);

  // qs base byte for THIS head's elements within the layer's qs-section:
  //   layer_base + pos_abs*kv_dim + head*HEAD_DIM (bytes; one byte per elem).
  if ((tid & 3u) == 0u) {
    let pk = qk0 | (qk1 <<  8u) | (qk2 << 16u) | (qk3 << 24u);
    let pv = qv0 | (qv1 <<  8u) | (qv2 << 16u) | (qv3 << 24u);
    let qs_byte_in_layer =
      p.dst_qs_byte_offset + pos_abs * p.kv_dim + head * HEAD_DIM + tid;
    kv_k[qs_byte_in_layer >> 2u] = pk;
    kv_v[qs_byte_in_layer >> 2u] = pv;
  }

  // d (FP32 scale) — one writer per block.
  if ((tid & 31u) == 0u) {
    let block_global =
      pos_abs * p.nb_per_row + head * NB_PER_HEAD + block_in_head;
    kv_k[p.dst_d_word_offset + block_global] = bitcast<u32>(dk);
    kv_v[p.dst_d_word_offset + block_global] = bitcast<u32>(dv);
  }
}
