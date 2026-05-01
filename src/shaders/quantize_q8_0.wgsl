// Quantize FP32 activations to Q8_0 layout, with d's stored as FP32 in a
// separate region from qs's (so both regions are u32-aligned). Output buffer
// (`outbuf`) is treated as `array<u32>`; the matmul shader reads:
//   d (f32):  bitcast<f32>(outbuf[d_offset/4 + block_global])
//   qs (i8x4): outbuf[qs_offset/4 + (m*K + b*32)/4 + sub] for sub in 0..8
// dispatch: (m * nb_q8, 1, 1) workgroups, 32 threads each.

struct Params {
  k: u32,             // K, multiple of 32
  m: u32,             // number of tokens (rows)
  input_offset: u32,  // f32 elements
  d_offset: u32,      // bytes
  qs_offset: u32,     // bytes
  dispatch_x_dim: u32,
  _p0: u32, _p1: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> outbuf: array<u32>;

const WG: u32 = 32u;
var<workgroup> shared_x: array<f32, 32>;
var<workgroup> shared_amax: array<f32, 32>;
var<workgroup> packed: array<atomic<u32>, 8>;

@compute @workgroup_size(WG)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let block_global = wg.y * p.dispatch_x_dim + wg.x;
  let nb_q8 = p.k / 32u;
  let m_idx = block_global / nb_q8;
  let b_idx = block_global % nb_q8;
  let tid = lid.x;
  if (m_idx >= p.m) { return; }

  let in_base = p.input_offset + m_idx * p.k + b_idx * 32u;
  let v = x[in_base + tid];
  shared_x[tid] = v;
  shared_amax[tid] = abs(v);
  if (tid < 8u) { atomicStore(&packed[tid], 0u); }
  workgroupBarrier();

  // amax reduction
  var s: u32 = WG / 2u;
  while (s > 0u) {
    if (tid < s) { shared_amax[tid] = max(shared_amax[tid], shared_amax[tid + s]); }
    workgroupBarrier();
    s = s / 2u;
  }
  let amax = shared_amax[0];
  let d = amax / 127.0;
  let id_inv = select(0.0, 1.0 / d, d > 0.0);
  let qv_f = round(shared_x[tid] * id_inv);
  let qv = u32(i32(clamp(qv_f, -127.0, 127.0))) & 0xFFu;

  let pack_idx = tid / 4u;
  let byte_in_pack = tid % 4u;
  atomicOr(&packed[pack_idx], qv << (byte_in_pack * 8u));
  workgroupBarrier();

  if (tid < 8u) {
    let qs_byte = p.qs_offset + m_idx * p.k + b_idx * 32u;
    outbuf[(qs_byte >> 2u) + tid] = atomicLoad(&packed[tid]);
  }
  if (tid == 0u) {
    outbuf[(p.d_offset >> 2u) + block_global] = bitcast<u32>(d);
  }
}
