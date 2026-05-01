enable f16;

// Quantize K and V activations from `act` (f16) into the Q8_0 KV cache.
//
// Cache layout (per kv_k / kv_v buffer):
//   d-section  (FP32 scales): bytes [0, n_layer * max_seq * (kv_dim/32) * 4)
//   qs-section (i8 packed):   bytes [d_total, d_total + n_layer * max_seq * kv_dim)
//
// One workgroup of 32 threads quantizes one 32-element block. Each block has
// its own scale `d = amax / 127.0`. Layout offsets are passed in:
//   dst_d_word_offset:  u32-word offset of the LAYER's d-section start
//   dst_qs_byte_offset: byte offset of the LAYER's qs-section start
// `pos_base` is the absolute cache position of the first token (tg: pos;
// matmul prefill: 0). Dispatch grid = (m_tokens * nb_per_row, 1, 1), wrapped
// to 2D when the x-extent exceeds the 65535 cap (via dispatch_x_dim).

struct Params {
  k_cur_off: u32,           // f16 element offset in act
  v_cur_off: u32,           // f16 element offset in act
  dst_d_word_offset: u32,   // u32-word offset (layer base, d-section)
  dst_qs_byte_offset: u32,  // byte offset (layer base, qs-section)
  pos_base: u32,
  nb_per_row: u32,          // kv_dim / 32
  kv_dim: u32,
  dispatch_x_dim: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> act: array<f16>;
@group(0) @binding(2) var<storage, read_write> kv_k: array<u32>;
@group(0) @binding(3) var<storage, read_write> kv_v: array<u32>;

const WG: u32 = 32u;

var<workgroup> sh_k: array<f32, 32>;
var<workgroup> sh_v: array<f32, 32>;
var<workgroup> sh_amax_k: array<f32, 32>;
var<workgroup> sh_amax_v: array<f32, 32>;
var<workgroup> packed_k: array<atomic<u32>, 8>;
var<workgroup> packed_v: array<atomic<u32>, 8>;

@compute @workgroup_size(WG)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let block_global = wg.y * p.dispatch_x_dim + wg.x;
  let tok = block_global / p.nb_per_row;
  let b   = block_global % p.nb_per_row;
  let tid = lid.x;

  // Load K[tok, b*32 + tid] and V[tok, b*32 + tid] from act (f16 → f32).
  let src_off = tok * p.kv_dim + b * 32u + tid;
  let kv = f32(act[p.k_cur_off + src_off]);
  let vv = f32(act[p.v_cur_off + src_off]);
  sh_k[tid] = kv;
  sh_v[tid] = vv;
  sh_amax_k[tid] = abs(kv);
  sh_amax_v[tid] = abs(vv);
  if (tid < 8u) {
    atomicStore(&packed_k[tid], 0u);
    atomicStore(&packed_v[tid], 0u);
  }
  workgroupBarrier();

  // amax reduction (32 → 1).
  var s: u32 = WG / 2u;
  while (s > 0u) {
    if (tid < s) {
      sh_amax_k[tid] = max(sh_amax_k[tid], sh_amax_k[tid + s]);
      sh_amax_v[tid] = max(sh_amax_v[tid], sh_amax_v[tid + s]);
    }
    workgroupBarrier();
    s = s / 2u;
  }
  let amax_k = sh_amax_k[0];
  let amax_v = sh_amax_v[0];
  let dk = amax_k / 127.0;
  let dv = amax_v / 127.0;
  let id_inv_k = select(0.0, 1.0 / dk, dk > 0.0);
  let id_inv_v = select(0.0, 1.0 / dv, dv > 0.0);
  let qk = u32(i32(clamp(round(sh_k[tid] * id_inv_k), -127.0, 127.0))) & 0xFFu;
  let qv = u32(i32(clamp(round(sh_v[tid] * id_inv_v), -127.0, 127.0))) & 0xFFu;

  let pack_idx = tid / 4u;
  let byte_in_pack = tid % 4u;
  atomicOr(&packed_k[pack_idx], qk << (byte_in_pack * 8u));
  atomicOr(&packed_v[pack_idx], qv << (byte_in_pack * 8u));
  workgroupBarrier();

  // Write the block out.
  let pos = p.pos_base + tok;
  let block_pos = pos * p.nb_per_row + b;
  if (tid < 8u) {
    let qs_byte = p.dst_qs_byte_offset + pos * p.kv_dim + b * 32u;
    kv_k[(qs_byte >> 2u) + tid] = atomicLoad(&packed_k[tid]);
    kv_v[(qs_byte >> 2u) + tid] = atomicLoad(&packed_v[tid]);
  }
  if (tid == 0u) {
    kv_k[p.dst_d_word_offset + block_pos] = bitcast<u32>(dk);
    kv_v[p.dst_d_word_offset + block_pos] = bitcast<u32>(dv);
  }
}
