requires packed_4x8_integer_dot_product;

// dot4I8Packed matmul: Q1_0 weights × Q8_0 activations -> FP32 output.
// Per Q1_0 block (128 weights = 4 Q8_0 sub-blocks of 32 weights each):
//   for each sub-block s in 0..4:
//     8 packed-i8 dot products of u32 weight-pack vs u32 activation-pack
//     scaled by (d_w * d_a)
// Each output cell (m, n) handled by one thread. Workgroup tile (8,8).
// dispatch: (ceil(n/8), ceil(m/8), 1)

struct Params {
  k: u32,
  n: u32,                 // output dim (rows of W)
  m: u32,                 // number of tokens (rows of X)
  w_d_offset: u32,
  w_qs_offset: u32,
  a_d_offset: u32,        // bytes (FP32 d's, per-block)
  a_qs_offset: u32,       // bytes (i8 quants)
  out_offset: u32,
  accumulate: u32,
  _p0: u32, _p1: u32, _p2: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> acts: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

fn load_w_byte(b_offset: u32) -> u32 {
  let word = weights[b_offset >> 2u];
  return (word >> ((b_offset & 3u) * 8u)) & 0xFFu;
}

fn load_w_f16(b_offset: u32) -> f32 {
  let word = weights[b_offset >> 2u];
  let half = (word >> ((b_offset & 2u) * 8u)) & 0xFFFFu;
  return unpack2x16float(half).x;
}

// Expand 4 sign bits (LSB-first) to packed-int8 (+1 or -1) values.
fn expand_4_bits(bits: u32) -> u32 {
  let b0 = select(0xFFu, 0x01u, (bits & 1u) != 0u);
  let b1 = select(0xFFu, 0x01u, (bits & 2u) != 0u);
  let b2 = select(0xFFu, 0x01u, (bits & 4u) != 0u);
  let b3 = select(0xFFu, 0x01u, (bits & 8u) != 0u);
  return b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
}

const TILE_M: u32 = 8u;
const TILE_N: u32 = 8u;

@compute @workgroup_size(TILE_N, TILE_M, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n_idx = gid.x;
  let m_idx = gid.y;
  if (m_idx >= p.m || n_idx >= p.n) { return; }

  let nb_q1 = p.k / 128u;
  let nb_q8 = p.k / 32u;

  let w_d_row  = p.w_d_offset  + n_idx * nb_q1 * 2u;
  let w_qs_row = p.w_qs_offset + n_idx * nb_q1 * 16u;
  let a_d_row  = p.a_d_offset  + m_idx * nb_q8 * 4u;
  let a_qs_row = p.a_qs_offset + m_idx * p.k;

  var acc: f32 = 0.0;
  for (var b: u32 = 0u; b < nb_q1; b++) {
    let dw = load_w_f16(w_d_row + b * 2u);
    let w_qs_b = w_qs_row + b * 16u;
    for (var s: u32 = 0u; s < 4u; s++) {
      let a_b = b * 4u + s;
      let da = bitcast<f32>(acts[(a_d_row >> 2u) + a_b]);
      let a_qs_block = a_qs_row + a_b * 32u;
      let w_qs_sub = w_qs_b + s * 4u;  // 4 bytes = 32 sign bits
      var sumi: i32 = 0;
      for (var u: u32 = 0u; u < 8u; u++) {
        let bit_byte = load_w_byte(w_qs_sub + (u >> 1u));
        let bits4 = (bit_byte >> ((u & 1u) * 4u)) & 0xFu;
        let w_pack = expand_4_bits(bits4);
        let a_pack = acts[(a_qs_block >> 2u) + u];
        sumi = dot4I8Packed(w_pack, a_pack) + sumi;
      }
      acc = acc + dw * da * f32(sumi);
    }
  }

  let yi = p.out_offset + m_idx * p.n + n_idx;
  if (p.accumulate != 0u) {
    y[yi] = y[yi] + acc;
  } else {
    y[yi] = acc;
  }
}
