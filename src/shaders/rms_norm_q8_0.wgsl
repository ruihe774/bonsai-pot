enable f16;

// Fused: y = rms_norm(x) * w  →  Q8_0 quantize  →  write to act_q8.
//
// Replaces the prefill-path pair `rms_norm` (writes act.x_norm) +
// `quantize_q8_0` (reads act.x_norm, writes act_q8). The fused version
// keeps the normalized values in registers and never touches act.x_norm.
//
// Output layout (matches `quantize_q8_0.wgsl`):
//   d (f32):  outbuf[d_offset/4 + token*nb_q8 + b]
//   qs (i8):  outbuf[qs_offset/4 + (token*k + b*32)/4 + sub] for sub in 0..8
//
// dispatch: (m, 1, 1) — one workgroup per token.
//
// Workgroup organization:
//   WG = 256 threads. The Q8_0 block size is 32, so each WG processes
//   BLOCKS_PER_ITER = WG/32 = 8 blocks per "stride iteration", with
//   `lane_in_block = sg_inv_id & 31` and `block_within_iter = sg_id-relative`.
//   Number of iterations = nb_q8 / 8 (clean for k ∈ {n_embd=2560,
//   q_dim=4096, n_ff=9728} on Bonsai 4B/8B; the host enforces this).

struct Params {
  k: u32,             // input dim, multiple of 256
  input_offset: u32,  // f16 elements
  weight_offset: u32, // f16 elements (k weights, reused across tokens)
  d_offset: u32,      // bytes (Q8_0 d-section, FP32 scales)
  qs_offset: u32,     // bytes (Q8_0 qs-section, packed i8)
  eps: f32,
};

var<immediate> p: Params;
@group(0) @binding(0) var<storage, read> act: array<f16>;
@group(0) @binding(1) var<storage, read> w: array<f16>;
@group(0) @binding(2) var<storage, read_write> outbuf: array<u32>;

const SUBGROUP_MIN_SIZE: u32 = {{SUBGROUP_MIN_SIZE}}u;
const WG: u32 = 256u;
const BLOCK_SIZE: u32 = 32u;
const BLOCKS_PER_ITER: u32 = WG / BLOCK_SIZE;  // 8
const SG_PARTIAL_MAX: u32 = (WG + SUBGROUP_MIN_SIZE - 1u) / SUBGROUP_MIN_SIZE;

// Cross-subgroup reduction storage (used iff num_subgroups > 1).
var<workgroup> sg_partial: array<f32, SG_PARTIAL_MAX>;
// Workgroup-broadcast inv_h after rms_norm reduction.
var<workgroup> inv_h_sh: f32;
// Per-block byte packing fallback (used iff subgroup_size < 32).
var<workgroup> qv_sh: array<u32, WG>;

fn wg_sum_f32(local: f32, sg_id: u32, sg_inv_id: u32, num_subgroups: u32) -> f32 {
  let sg_sum = subgroupAdd(local);
  if (num_subgroups == 1u) { return sg_sum; }
  if (sg_inv_id == 0u) { sg_partial[sg_id] = sg_sum; }
  workgroupBarrier();
  if (sg_id == 0u) {
    var combined: f32;
    if (sg_inv_id < num_subgroups) { combined = sg_partial[sg_inv_id]; }
    let final_sum = subgroupAdd(combined);
    if (sg_inv_id == 0u) { sg_partial[0] = final_sum; }
  }
  workgroupBarrier();
  return sg_partial[0];
}

@compute @workgroup_size(WG)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_inv_id: u32,
  @builtin(subgroup_id) sg_id: u32,
  @builtin(num_subgroups) num_subgroups: u32,
) {
  let token_idx = wg.x;
  let tid = lid.x;
  let nb_q8 = p.k / BLOCK_SIZE;
  let in_base = p.input_offset + token_idx * p.k;
  let qs_token_byte = p.qs_offset + token_idx * p.k;     // bytes
  let d_token_word = (p.d_offset >> 2u) + token_idx * nb_q8;

  // ---- Phase 1: rms_norm sum-of-squares ----------------------------------
  var s: f32;
  for (var i: u32 = tid; i < p.k; i = i + WG) {
    let v = f32(act[in_base + i]);
    s = s + v * v;
  }
  let total_sq = wg_sum_f32(s, sg_id, sg_inv_id, num_subgroups);
  if (tid == 0u) {
    inv_h_sh = inverseSqrt(total_sq / f32(p.k) + p.eps);
  }
  workgroupBarrier();
  let inv_h = inv_h_sh;

  // ---- Phase 2: per-block normalize + Q8_0 quantize ----------------------
  // Each WG handles BLOCKS_PER_ITER blocks per iteration. Within each iter,
  // 32 lanes (one block) cooperate on per-block max-abs reduction and byte
  // packing. The 32-lane reduction is done via a subgroupShuffleXor butterfly
  // limited to mask <= 16 — this stays within a 32-lane group regardless of
  // hardware subgroup size (wave32 = 1 block per subgroup; wave64 = 2
  // blocks per subgroup, each in its own contiguous half).
  let lane_in_block = tid & 31u;            // 0..31 — XOR partner stays in block
  let block_within_iter = tid >> 5u;        // 0..7
  // For shuffle-based byte packing we need the in-subgroup target lane. With
  // wave32, sg_inv_id == lane_in_block. With wave64, sg_inv_id is 0..63 and
  // sg_inv_id & ~31u marks the half that holds this thread's block.
  let sg_block_base = sg_inv_id & ~31u;

  let n_iters = nb_q8 / BLOCKS_PER_ITER;
  for (var it: u32 = 0u; it < n_iters; it = it + 1u) {
    let block_global = it * BLOCKS_PER_ITER + block_within_iter;
    let elem_idx = it * WG + tid;           // == block_global*32 + lane_in_block
    let v = f32(act[in_base + elem_idx]) * inv_h * f32(w[p.weight_offset + elem_idx]);

    // Per-32-lane max-abs reduction via shuffle-XOR butterfly (mask <= 16).
    var m = abs(v);
    m = max(m, subgroupShuffleXor(m, 1u));
    m = max(m, subgroupShuffleXor(m, 2u));
    m = max(m, subgroupShuffleXor(m, 4u));
    m = max(m, subgroupShuffleXor(m, 8u));
    m = max(m, subgroupShuffleXor(m, 16u));
    let d = m / 127.0;
    let id_inv = select(0.0, 1.0 / d, d > 0.0);
    let qv = u32(i32(clamp(round(v * id_inv), -127.0, 127.0))) & 0xFFu;

    // Pack 4 bytes per u32. Two paths:
    //   - subgroup_size >= 32 (universal on AMD/NVIDIA wave32+): subgroupShuffle
    //     within the 32-lane block (relative target = sg_block_base | (lane_in_block & ~3) + i).
    //   - else (Intel-class < 32): LDS staging.
    var packed: u32;
    if (SUBGROUP_MIN_SIZE >= 32u) {
      let group4_in_block = lane_in_block & ~3u;
      let tgt = sg_block_base | group4_in_block;
      packed = subgroupShuffle(qv, tgt + 0u)
             | (subgroupShuffle(qv, tgt + 1u) <<  8u)
             | (subgroupShuffle(qv, tgt + 2u) << 16u)
             | (subgroupShuffle(qv, tgt + 3u) << 24u);
    } else {
      qv_sh[tid] = qv;
      workgroupBarrier();
      let base = (block_within_iter * 32u) + (lane_in_block & ~3u);
      packed = qv_sh[base + 0u]
             | (qv_sh[base + 1u] <<  8u)
             | (qv_sh[base + 2u] << 16u)
             | (qv_sh[base + 3u] << 24u);
      workgroupBarrier();
    }
    if ((lane_in_block & 3u) == 0u) {
      let qs_byte = qs_token_byte + block_global * 32u + lane_in_block;
      outbuf[qs_byte >> 2u] = packed;
    }
    if (lane_in_block == 0u) {
      outbuf[d_token_word + block_global] = bitcast<u32>(d);
    }
  }
}
