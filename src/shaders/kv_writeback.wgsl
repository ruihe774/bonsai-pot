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

const SUBGROUP_MIN_SIZE: u32 = {{SUBGROUP_MIN_SIZE}}u;
const WG: u32 = 32u;
// Ceiling division: if SUBGROUP_MIN_SIZE > WG (WG occupies only part of one
// subgroup), num_subgroups == 1 and the fast path is taken.
const SG_PARTIAL_MAX: u32 = (WG + SUBGROUP_MIN_SIZE - 1u) / SUBGROUP_MIN_SIZE;

// Per-thread quantized bytes (one for K, one for V), packed by tid 0..7.
var<workgroup> qk_sh: array<u32, 32>;
var<workgroup> qv_sh: array<u32, 32>;
// Cross-subgroup partials for the amax reduction; fast-path taken when
// num_subgroups == 1.
var<workgroup> sg_amax_k: array<f32, SG_PARTIAL_MAX>;
var<workgroup> sg_amax_v: array<f32, SG_PARTIAL_MAX>;

fn wg_max2(a: f32, b: f32, sg_id: u32, sg_inv_id: u32, num_subgroups: u32) -> vec2<f32> {
  let sa = subgroupMax(a);
  let sb = subgroupMax(b);
  if (num_subgroups == 1u) { return vec2<f32>(sa, sb); }
  if (sg_inv_id == 0u) {
    sg_amax_k[sg_id] = sa;
    sg_amax_v[sg_id] = sb;
  }
  workgroupBarrier();
  if (sg_id == 0u) {
    var ca: f32;
    var cb: f32;
    if (sg_inv_id < num_subgroups) {
      ca = sg_amax_k[sg_inv_id];
      cb = sg_amax_v[sg_inv_id];
    }
    let fa = subgroupMax(ca);
    let fb = subgroupMax(cb);
    if (sg_inv_id == 0u) {
      sg_amax_k[0] = fa;
      sg_amax_v[0] = fb;
    }
  }
  workgroupBarrier();
  return vec2<f32>(sg_amax_k[0], sg_amax_v[0]);
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
  let tok = block_global / p.nb_per_row;
  let b   = block_global % p.nb_per_row;
  let tid = lid.x;

  // Load K[tok, b*32 + tid] and V[tok, b*32 + tid] from act (f16 → f32).
  let src_off = tok * p.kv_dim + b * 32u + tid;
  let kv = f32(act[p.k_cur_off + src_off]);
  let vv = f32(act[p.v_cur_off + src_off]);

  // amax (32 → 1) via subgroup reduction (with cross-subgroup merge if needed).
  let amax = wg_max2(abs(kv), abs(vv), sg_id, sg_inv_id, num_subgroups);
  let dk = amax.x / 127.0;
  let dv = amax.y / 127.0;
  let id_inv_k = select(0.0, 1.0 / dk, dk > 0.0);
  let id_inv_v = select(0.0, 1.0 / dv, dv > 0.0);
  let qk = u32(i32(clamp(round(kv * id_inv_k), -127.0, 127.0))) & 0xFFu;
  let qv = u32(i32(clamp(round(vv * id_inv_v), -127.0, 127.0))) & 0xFFu;

  // Pack via shmem: each thread stores its byte; tid 0..7 then assembles
  // four bytes into one u32 and writes it out.
  qk_sh[tid] = qk;
  qv_sh[tid] = qv;
  workgroupBarrier();

  let pos = p.pos_base + tok;
  if (tid < 8u) {
    let base = tid * 4u;
    let pk = qk_sh[base + 0u]
           | (qk_sh[base + 1u] <<  8u)
           | (qk_sh[base + 2u] << 16u)
           | (qk_sh[base + 3u] << 24u);
    let pv = qv_sh[base + 0u]
           | (qv_sh[base + 1u] <<  8u)
           | (qv_sh[base + 2u] << 16u)
           | (qv_sh[base + 3u] << 24u);
    let qs_byte = p.dst_qs_byte_offset + pos * p.kv_dim + b * 32u;
    kv_k[(qs_byte >> 2u) + tid] = pk;
    kv_v[(qs_byte >> 2u) + tid] = pv;
  }
  if (tid == 0u) {
    let block_pos = pos * p.nb_per_row + b;
    kv_k[p.dst_d_word_offset + block_pos] = bitcast<u32>(dk);
    kv_v[p.dst_d_word_offset + block_pos] = bitcast<u32>(dv);
  }
}
