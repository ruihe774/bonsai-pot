enable f16;

// Q-tiled + GQA-batched FlashAttention-2 prefill kernel.
//
// dispatch: (n_kv_head, ceil(M / Q_TILE), 1)
//
// Each workgroup handles Q_TILE consecutive query tokens × Q_PER_GROUP=4 Q
// heads sharing the same KV head. K and V are loaded ONCE per cache position
// and reused across all Q_TILE * Q_PER_GROUP queries within the workgroup.
// This is the bandwidth win over the original `attention.wgsl`, which
// dispatched one WG per (Q-head, query-token) and re-loaded every K/V byte
// 4 * M times.
//
// Per-query causal mask: query at row m_idx = m_tile * Q_TILE + m_local
// attends to cache positions [0, pos_base + m_idx]. Out-of-range positions
// score to -inf so they contribute neither to m_run nor l_run nor o.
//
// Q_TILE is fixed at 2 and the per-query state (Q registers, output
// accumulators, online-softmax m/l) is **manually unrolled** as ml0/ml1
// sets of named variables rather than `array<…, Q_TILE>` with dynamic
// indexing. With dynamic indexing NVIDIA spills to local memory and the
// kernel becomes ~5× slower than the non-tiled baseline (measured: 40 ms
// vs ~9 ms per call on Bonsai-4B at m=512, pos_base=512).
//
// Why Q_TILE=2 specifically: Q_TILE=1 (GQA-batching only, no Q-tile) gives
// ~5.7 ms; Q_TILE=2 gives ~5.1 ms; Q_TILE=4 regresses to ~6.0 ms (likely a
// register-pressure cliff on the GB10 dev hardware — wave32 with ~96 VGPRs
// usable before occupancy drops). Q_TILE>2 also hurts L2 cache reuse for
// K/V across consecutive (kv_head, m_tile) WGs.
//
// **Output is Q8_0 written directly to `act_q8`** (no f16 staging in
// `act.attn_out`). The Wo matmul reads it as Q8_0 input. This drops the
// per-layer `quantize_act(attn_out)` dispatch. Per-block max-abs reduction
// uses subgroupShuffleXor butterfly within 32-lane groups (mask <= 16, so
// partners stay within the same 32-lane block on both wave32 and wave64).
// Each query-head's 128 output elements span 4 Q8_0 blocks; with WG=64 +
// EPT=2, lanes 0..31 own (block 0 element, block 2 element) and lanes
// 32..63 own (block 1 element, block 3 element).
//
// K/V cache layout: same as attention_split.wgsl (Q8_0, FP32 d-section
// followed by i8 qs-section in the same u32 storage buffer).
//
// Invariants (asserted by the host):
//   - head_dim == ELEMS_PER_THREAD * WG  (head_dim=128, WG=64, EPT=2)
//   - n_head == n_kv_head * Q_PER_GROUP   (Q_PER_GROUP=4 for Bonsai 4B/8B)
//   - SUBGROUP_MIN_SIZE >= 32 (validated at Model::load against the adapter)

struct Params {
  head_dim: u32,
  n_head: u32,
  n_kv_head: u32,
  m_tokens: u32,
  pos_base: u32,
  kv_stride: u32,
  q_offset: u32,         // f16 element offset of Q in `act`
  k_d_word_offset: u32,
  k_qs_byte_offset: u32,
  v_d_word_offset: u32,
  v_qs_byte_offset: u32,
  out_d_offset: u32,     // byte offset of Q8_0 d-section in `act_q8`
  out_qs_offset: u32,    // byte offset of Q8_0 qs-section in `act_q8`
  scale: f32,
};

var<immediate> p: Params;
@group(0) @binding(0) var<storage, read> act: array<f16>;
@group(0) @binding(1) var<storage, read> k_cache: array<u32>;
@group(0) @binding(2) var<storage, read> v_cache: array<u32>;
@group(0) @binding(3) var<storage, read_write> outbuf: array<u32>;

const SUBGROUP_MIN_SIZE: u32 = {{SUBGROUP_MIN_SIZE}}u;
const WG: u32 = 64u;
const SG_PARTIAL_MAX: u32 = (WG + SUBGROUP_MIN_SIZE - 1u) / SUBGROUP_MIN_SIZE;
const Q_PER_GROUP: u32 = 4u;
const ELEMS_PER_THREAD: u32 = 2u;  // head_dim (128) / WG (64)
const Q_TILE: u32 = 2u;            // queries per WG along m-axis
const NEG_INF: f32 = -1.0e30;

var<workgroup> sg_partial4: array<vec4<f32>, SG_PARTIAL_MAX>;

fn wg_sum_v4(local: vec4<f32>, sg_id: u32, sg_inv_id: u32, num_subgroups: u32) -> vec4<f32> {
  let sg_sum = subgroupAdd(local);
  if (SUBGROUP_MIN_SIZE >= WG || num_subgroups == 1u) {
    return sg_sum;
  }
  if (sg_inv_id == 0u) { sg_partial4[sg_id] = sg_sum; }
  workgroupBarrier();
  if (sg_id == 0u) {
    var combined: vec4<f32>;
    if (sg_inv_id < num_subgroups) { combined = sg_partial4[sg_inv_id]; }
    let final_sum = subgroupAdd(combined);
    if (sg_inv_id == 0u) { sg_partial4[0] = final_sum; }
  }
  workgroupBarrier();
  return sg_partial4[0];
}

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

// ---- Q8_0 quantize helpers (per-32-lane block) -------------------------
// Reduce abs(v) within the calling thread's 32-lane block via shuffle-XOR
// butterfly with masks <= 16 (partners stay within the same 32-lane half on
// both wave32 and wave64, since bit 5 of the lane index is preserved).
fn block_max_abs(v: f32) -> f32 {
  var m = abs(v);
  m = max(m, subgroupShuffleXor(m, 1u));
  m = max(m, subgroupShuffleXor(m, 2u));
  m = max(m, subgroupShuffleXor(m, 4u));
  m = max(m, subgroupShuffleXor(m, 8u));
  m = max(m, subgroupShuffleXor(m, 16u));
  return m;
}

// Pack 4 consecutive lanes' i8 quantized bytes into a u32 via subgroupShuffle.
// `sg_inv_id` is the calling thread's subgroup-invocation id; the target
// formula `(sg_inv_id & ~3u) + i` works for both wave32 and wave64 because
// it preserves bit 5 (the 32-lane block selector).
fn pack4(qv: u32, sg_inv_id: u32) -> u32 {
  let base = sg_inv_id & ~3u;
  return  subgroupShuffle(qv, base + 0u)
       | (subgroupShuffle(qv, base + 1u) <<  8u)
       | (subgroupShuffle(qv, base + 2u) << 16u)
       | (subgroupShuffle(qv, base + 3u) << 24u);
}

// Quantize one f32 value into Q8_0 within a 32-lane block. Writes the
// per-block d (FP32) and packed i8 bytes to `outbuf`. `block_global` is
// the global Q8_0 block index, `lane_in_block` is 0..31, `sg_inv_id` is
// the subgroup-invocation id of the calling thread.
fn write_block(v: f32, block_global: u32, lane_in_block: u32, sg_inv_id: u32) {
  let max_abs = block_max_abs(v);
  let d = max_abs / 127.0;
  let id_inv = select(0.0, 1.0 / d, d > 0.0);
  let qv = u32(i32(clamp(round(v * id_inv), -127.0, 127.0))) & 0xFFu;
  let packed = pack4(qv, sg_inv_id);
  if ((lane_in_block & 3u) == 0u) {
    let qs_byte = p.out_qs_offset + block_global * 32u + lane_in_block;
    outbuf[qs_byte >> 2u] = packed;
  }
  if (lane_in_block == 0u) {
    outbuf[(p.out_d_offset >> 2u) + block_global] = bitcast<u32>(d);
  }
}

@compute @workgroup_size(WG)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_inv_id: u32,
  @builtin(subgroup_id) sg_id: u32,
  @builtin(num_subgroups) num_subgroups: u32,
) {
  let g = wg.x;
  let m_tile = wg.y;
  if (g >= p.n_kv_head) { return; }
  let m_base = m_tile * Q_TILE;
  if (m_base >= p.m_tokens) { return; }
  let tid = lid.x;
  let hd = p.head_dim;
  let q_stride = p.n_head * hd;
  let g_off = g * hd;
  let m_idx0 = m_base + 0u;
  let m_idx1 = m_base + 1u;
  let valid1 = m_idx1 < p.m_tokens;

  // Outer t-bound: longest-running query in the tile.
  let t_max = p.pos_base + select(m_idx0 + 1u, m_idx1 + 1u, valid1);

  let q_base0 = p.q_offset + m_idx0 * q_stride + g * Q_PER_GROUP * hd;
  let q_base1 = p.q_offset + m_idx1 * q_stride + g * Q_PER_GROUP * hd;

  let d0 = tid;
  let d1 = tid + WG;

  // Per-thread Q registers — manually unrolled. Reads past p.m_tokens may
  // grab stale data, but those queries are masked to NEG_INF below.
  let q0_0_0 = f32(act[q_base0 + 0u * hd + d0]); let q0_0_1 = f32(act[q_base0 + 0u * hd + d1]);
  let q0_1_0 = f32(act[q_base0 + 1u * hd + d0]); let q0_1_1 = f32(act[q_base0 + 1u * hd + d1]);
  let q0_2_0 = f32(act[q_base0 + 2u * hd + d0]); let q0_2_1 = f32(act[q_base0 + 2u * hd + d1]);
  let q0_3_0 = f32(act[q_base0 + 3u * hd + d0]); let q0_3_1 = f32(act[q_base0 + 3u * hd + d1]);
  let q1_0_0 = f32(act[q_base1 + 0u * hd + d0]); let q1_0_1 = f32(act[q_base1 + 0u * hd + d1]);
  let q1_1_0 = f32(act[q_base1 + 1u * hd + d0]); let q1_1_1 = f32(act[q_base1 + 1u * hd + d1]);
  let q1_2_0 = f32(act[q_base1 + 2u * hd + d0]); let q1_2_1 = f32(act[q_base1 + 2u * hd + d1]);
  let q1_3_0 = f32(act[q_base1 + 3u * hd + d0]); let q1_3_1 = f32(act[q_base1 + 3u * hd + d1]);

  var m_run0: vec4<f32> = vec4<f32>(NEG_INF);
  var m_run1: vec4<f32> = vec4<f32>(NEG_INF);
  var l_run0: vec4<f32> = vec4<f32>(0.0);
  var l_run1: vec4<f32> = vec4<f32>(0.0);

  var o0_0_0: f32; var o0_0_1: f32; var o0_1_0: f32; var o0_1_1: f32;
  var o0_2_0: f32; var o0_2_1: f32; var o0_3_0: f32; var o0_3_1: f32;
  var o1_0_0: f32; var o1_0_1: f32; var o1_1_0: f32; var o1_1_1: f32;
  var o1_2_0: f32; var o1_2_1: f32; var o1_3_0: f32; var o1_3_1: f32;

  for (var t: u32 = 0u; t < t_max; t = t + 1u) {
    let k0 = load_k(t, g_off + d0);
    let k1 = load_k(t, g_off + d1);

    // --- ml=0 ---
    var local0: vec4<f32>;
    local0.x = q0_0_0 * k0 + q0_0_1 * k1;
    local0.y = q0_1_0 * k0 + q0_1_1 * k1;
    local0.z = q0_2_0 * k0 + q0_2_1 * k1;
    local0.w = q0_3_0 * k0 + q0_3_1 * k1;
    let dots0 = wg_sum_v4(local0, sg_id, sg_inv_id, num_subgroups);
    let scores0 = select(vec4<f32>(NEG_INF), dots0 * p.scale, t <= p.pos_base + m_idx0);
    let m_new0 = max(m_run0, scores0);
    let corr0 = exp(m_run0 - m_new0);
    let w0 = exp(scores0 - m_new0);
    l_run0 = l_run0 * corr0 + w0;
    m_run0 = m_new0;

    // --- ml=1 ---
    var local1: vec4<f32>;
    local1.x = q1_0_0 * k0 + q1_0_1 * k1;
    local1.y = q1_1_0 * k0 + q1_1_1 * k1;
    local1.z = q1_2_0 * k0 + q1_2_1 * k1;
    local1.w = q1_3_0 * k0 + q1_3_1 * k1;
    let dots1 = wg_sum_v4(local1, sg_id, sg_inv_id, num_subgroups);
    let scores1 = select(vec4<f32>(NEG_INF), dots1 * p.scale, valid1 && (t <= p.pos_base + m_idx1));
    let m_new1 = max(m_run1, scores1);
    let corr1 = exp(m_run1 - m_new1);
    let w1 = exp(scores1 - m_new1);
    l_run1 = l_run1 * corr1 + w1;
    m_run1 = m_new1;

    // Apply correction (V·weight added after V load).
    o0_0_0 = o0_0_0 * corr0.x; o0_0_1 = o0_0_1 * corr0.x;
    o0_1_0 = o0_1_0 * corr0.y; o0_1_1 = o0_1_1 * corr0.y;
    o0_2_0 = o0_2_0 * corr0.z; o0_2_1 = o0_2_1 * corr0.z;
    o0_3_0 = o0_3_0 * corr0.w; o0_3_1 = o0_3_1 * corr0.w;
    o1_0_0 = o1_0_0 * corr1.x; o1_0_1 = o1_0_1 * corr1.x;
    o1_1_0 = o1_1_0 * corr1.y; o1_1_1 = o1_1_1 * corr1.y;
    o1_2_0 = o1_2_0 * corr1.z; o1_2_1 = o1_2_1 * corr1.z;
    o1_3_0 = o1_3_0 * corr1.w; o1_3_1 = o1_3_1 * corr1.w;

    let v0 = load_v(t, g_off + d0);
    let v1 = load_v(t, g_off + d1);
    o0_0_0 = o0_0_0 + w0.x * v0; o0_0_1 = o0_0_1 + w0.x * v1;
    o0_1_0 = o0_1_0 + w0.y * v0; o0_1_1 = o0_1_1 + w0.y * v1;
    o0_2_0 = o0_2_0 + w0.z * v0; o0_2_1 = o0_2_1 + w0.z * v1;
    o0_3_0 = o0_3_0 + w0.w * v0; o0_3_1 = o0_3_1 + w0.w * v1;
    o1_0_0 = o1_0_0 + w1.x * v0; o1_0_1 = o1_0_1 + w1.x * v1;
    o1_1_0 = o1_1_0 + w1.y * v0; o1_1_1 = o1_1_1 + w1.y * v1;
    o1_2_0 = o1_2_0 + w1.z * v0; o1_2_1 = o1_2_1 + w1.z * v1;
    o1_3_0 = o1_3_0 + w1.w * v0; o1_3_1 = o1_3_1 + w1.w * v1;
  }

  // ---- Normalize and write Q8_0 directly to act_q8 -----------------------
  // Per (query, head): 128 elements span 4 Q8_0 blocks of 32 elements each.
  // Lanes 0..31 own (block 0 element @ d0, block 2 element @ d1).
  // Lanes 32..63 own (block 1 element @ d0, block 3 element @ d1).
  let lane_in_block = sg_inv_id & 31u;        // 0..31, partner-stable for shuffle butterfly
  let half = tid >> 5u;                       // 0 or 1 → block-pair selector for d0/d1
  let nb_per_token = p.n_head * hd / 32u;     // = q_dim / 32 = 128 for Bonsai 4B/8B

  let inv0 = vec4<f32>(1.0) / l_run0;
  let block_base0 = m_idx0 * nb_per_token + g * Q_PER_GROUP * 4u;  // h*4 added below
  // q=0
  write_block(o0_0_0 * inv0.x, block_base0 + 0u * 4u + half + 0u, lane_in_block, sg_inv_id);
  write_block(o0_0_1 * inv0.x, block_base0 + 0u * 4u + half + 2u, lane_in_block, sg_inv_id);
  // q=1
  write_block(o0_1_0 * inv0.y, block_base0 + 1u * 4u + half + 0u, lane_in_block, sg_inv_id);
  write_block(o0_1_1 * inv0.y, block_base0 + 1u * 4u + half + 2u, lane_in_block, sg_inv_id);
  // q=2
  write_block(o0_2_0 * inv0.z, block_base0 + 2u * 4u + half + 0u, lane_in_block, sg_inv_id);
  write_block(o0_2_1 * inv0.z, block_base0 + 2u * 4u + half + 2u, lane_in_block, sg_inv_id);
  // q=3
  write_block(o0_3_0 * inv0.w, block_base0 + 3u * 4u + half + 0u, lane_in_block, sg_inv_id);
  write_block(o0_3_1 * inv0.w, block_base0 + 3u * 4u + half + 2u, lane_in_block, sg_inv_id);

  if (valid1) {
    let inv1 = vec4<f32>(1.0) / l_run1;
    let block_base1 = m_idx1 * nb_per_token + g * Q_PER_GROUP * 4u;
    write_block(o1_0_0 * inv1.x, block_base1 + 0u * 4u + half + 0u, lane_in_block, sg_inv_id);
    write_block(o1_0_1 * inv1.x, block_base1 + 0u * 4u + half + 2u, lane_in_block, sg_inv_id);
    write_block(o1_1_0 * inv1.y, block_base1 + 1u * 4u + half + 0u, lane_in_block, sg_inv_id);
    write_block(o1_1_1 * inv1.y, block_base1 + 1u * 4u + half + 2u, lane_in_block, sg_inv_id);
    write_block(o1_2_0 * inv1.z, block_base1 + 2u * 4u + half + 0u, lane_in_block, sg_inv_id);
    write_block(o1_2_1 * inv1.z, block_base1 + 2u * 4u + half + 2u, lane_in_block, sg_inv_id);
    write_block(o1_3_0 * inv1.w, block_base1 + 3u * 4u + half + 0u, lane_in_block, sg_inv_id);
    write_block(o1_3_1 * inv1.w, block_base1 + 3u * 4u + half + 2u, lane_in_block, sg_inv_id);
  }
}
