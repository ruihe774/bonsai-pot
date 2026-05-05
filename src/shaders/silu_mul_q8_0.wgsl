enable f16;

// Fused: y = silu(gate) * up  →  Q8_0 quantize  →  write to act_q8.
//
// Replaces the prefill-path pair `silu_mul` (writes act.ffn_in) +
// `quantize_q8_0` (reads act.ffn_in, writes act_q8). The fused version
// keeps each silu*up product in registers, computes per-32-block max-abs,
// and writes Q8_0 directly.
//
// Output layout (matches `quantize_q8_0.wgsl`):
//   d (f32):  outbuf[d_offset/4 + token*nb_q8 + b]
//   qs (i8):  outbuf[qs_offset/4 + (token*k + b*32)/4 + sub] for sub in 0..8
//
// dispatch: (m, 1, 1) — one workgroup per token.
//
// WG/block organization is identical to `rms_norm_q8_0.wgsl` (WG=256,
// BLOCKS_PER_ITER=8). Caller must guarantee `(k/32) % 8 == 0`, which holds
// for k = n_ff = 9728 on Bonsai 4B/8B.

struct Params {
  k: u32,             // n_ff, multiple of 256
  gate_offset: u32,   // f16 elements
  up_offset: u32,     // f16 elements
  d_offset: u32,      // bytes (Q8_0 d-section, FP32 scales)
  qs_offset: u32,     // bytes (Q8_0 qs-section, packed i8)
};

var<immediate> p: Params;
@group(0) @binding(0) var<storage, read> act: array<f16>;
@group(0) @binding(1) var<storage, read_write> outbuf: array<u32>;

const SUBGROUP_MIN_SIZE: u32 = {{SUBGROUP_MIN_SIZE}}u;
const WG: u32 = 256u;
const BLOCK_SIZE: u32 = 32u;
const BLOCKS_PER_ITER: u32 = WG / BLOCK_SIZE;  // 8

var<workgroup> qv_sh: array<u32, WG>;

@compute @workgroup_size(WG)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_inv_id: u32,
) {
  let token_idx = wg.x;
  let tid = lid.x;
  let nb_q8 = p.k / BLOCK_SIZE;
  let g_base = p.gate_offset + token_idx * p.k;
  let u_base = p.up_offset + token_idx * p.k;
  let qs_token_byte = p.qs_offset + token_idx * p.k;
  let d_token_word = (p.d_offset >> 2u) + token_idx * nb_q8;

  let lane_in_block = tid & 31u;
  let block_within_iter = tid >> 5u;
  let sg_block_base = sg_inv_id & ~31u;

  let n_iters = nb_q8 / BLOCKS_PER_ITER;
  for (var it: u32 = 0u; it < n_iters; it = it + 1u) {
    let block_global = it * BLOCKS_PER_ITER + block_within_iter;
    let elem_idx = it * WG + tid;
    let g = f32(act[g_base + elem_idx]);
    let silu = g / (1.0 + exp(-g));
    let v = silu * f32(act[u_base + elem_idx]);

    var m = abs(v);
    m = max(m, subgroupShuffleXor(m, 1u));
    m = max(m, subgroupShuffleXor(m, 2u));
    m = max(m, subgroupShuffleXor(m, 4u));
    m = max(m, subgroupShuffleXor(m, 8u));
    m = max(m, subgroupShuffleXor(m, 16u));
    let d = m / 127.0;
    let id_inv = select(0.0, 1.0 / d, d > 0.0);
    let qv = u32(i32(clamp(round(v * id_inv), -127.0, 127.0))) & 0xFFu;

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
