enable f16;

// Quantize f16 activations to Q8_0 layout, with d's stored as FP32 in a
// separate region from qs's (so both regions are u32-aligned). Output buffer
// (`outbuf`) is treated as `array<u32>`; the matmul shader reads:
//   d (f32):  bitcast<f32>(outbuf[d_offset/4 + block_global])
//   qs (i8x4): outbuf[qs_offset/4 + (m*K + b*32)/4 + sub] for sub in 0..8
// dispatch: (m * nb_q8, 1, 1) workgroups, 32 threads each.

struct Params {
  k: u32,             // K, multiple of 32
  m: u32,             // number of tokens (rows)
  input_offset: u32,  // f16 elements
  d_offset: u32,      // bytes
  qs_offset: u32,     // bytes
  dispatch_x_dim: u32,
};

var<immediate> p: Params;
@group(0) @binding(0) var<storage, read> x: array<f16>;
@group(0) @binding(1) var<storage, read_write> outbuf: array<u32>;

const SUBGROUP_MIN_SIZE: u32 = {{SUBGROUP_MIN_SIZE}}u;
const WG: u32 = 32u;
// Ceiling division: if SUBGROUP_MIN_SIZE > WG (WG occupies only part of one
// subgroup), num_subgroups == 1 and the fast path is taken; the extra slot is
// never written. If SUBGROUP_MIN_SIZE <= WG, this gives the exact worst case.
const SG_PARTIAL_MAX: u32 = (WG + SUBGROUP_MIN_SIZE - 1u) / SUBGROUP_MIN_SIZE;

// Only used on the subgroup_size < 32 path; the fast path uses subgroupShuffle.
var<workgroup> q_sh: array<u32, 32>;
var<workgroup> sg_amax: array<f32, SG_PARTIAL_MAX>;

fn wg_max(local: f32, sg_id: u32, sg_inv_id: u32, num_subgroups: u32) -> f32 {
  let sg_m = subgroupMax(local);
  if (SUBGROUP_MIN_SIZE >= WG || num_subgroups == 1u) { return sg_m; }
  if (sg_inv_id == 0u) { sg_amax[sg_id] = sg_m; }
  workgroupBarrier();
  if (sg_id == 0u) {
    var combined: f32;
    if (sg_inv_id < num_subgroups) { combined = sg_amax[sg_inv_id]; }
    let final_m = subgroupMax(combined);
    if (sg_inv_id == 0u) { sg_amax[0] = final_m; }
  }
  workgroupBarrier();
  return sg_amax[0];
}

@compute @workgroup_size(WG)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_inv_id: u32,
  @builtin(subgroup_id) sg_id: u32,
  @builtin(num_subgroups) num_subgroups: u32,
) {
  let block_global = wg.y * p.dispatch_x_dim + wg.x;
  let nb_q8 = p.k / 32u;
  let m_idx = block_global / nb_q8;
  let b_idx = block_global % nb_q8;
  let tid = lid.x;
  if (m_idx >= p.m) { return; }

  let in_base = p.input_offset + m_idx * p.k + b_idx * 32u;
  let v = f32(x[in_base + tid]);

  let amax = wg_max(abs(v), sg_id, sg_inv_id, num_subgroups);
  let d = amax / 127.0;
  let id_inv = select(0.0, 1.0 / d, d > 0.0);
  let qv = u32(i32(clamp(round(v * id_inv), -127.0, 127.0))) & 0xFFu;

  // Pack each thread's byte into 8 output u32s (4 bytes per word). Two paths:
  //   - subgroup_size >= 32 (one subgroup covers WG): reach lane 31 via
  //     subgroupShuffle, no LDS round-trip needed. The compile-time arm
  //     (SUBGROUP_MIN_SIZE >= WG) lets the compiler drop the else branch and
  //     `q_sh` shmem on AMD/NVIDIA; the runtime arm catches Intel-class
  //     hardware where min < 32 but the actual size is >= 32.
  //   - else: LDS fallback.
  // Assumes subgroup_invocation_id == local_invocation_index (universal on
  // AMD/NVIDIA/Intel/Apple; same assumption already relied on in matvec_q1_0).
  var packed: u32;
  if (SUBGROUP_MIN_SIZE >= WG || num_subgroups == 1u) {
    let base = (tid & 7u) * 4u;
    packed = subgroupShuffle(qv, base + 0u)
           | (subgroupShuffle(qv, base + 1u) <<  8u)
           | (subgroupShuffle(qv, base + 2u) << 16u)
           | (subgroupShuffle(qv, base + 3u) << 24u);
  } else {
    q_sh[tid] = qv;
    workgroupBarrier();
    if (tid < 8u) {
      let base = tid * 4u;
      packed = q_sh[base + 0u]
             | (q_sh[base + 1u] <<  8u)
             | (q_sh[base + 2u] << 16u)
             | (q_sh[base + 3u] << 24u);
    }
  }
  if (tid < 8u) {
    let qs_byte = p.qs_offset + m_idx * p.k + b_idx * 32u;
    outbuf[(qs_byte >> 2u) + tid] = packed;
  }
  if (tid == 0u) {
    outbuf[(p.d_offset >> 2u) + block_global] = bitcast<u32>(d);
  }
}
