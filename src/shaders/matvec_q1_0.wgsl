// Multiply-free Q1_0 matvec — Metal-style identity acc = d · (2·Σ_{bit=1} y − Σy).
// Each workgroup computes ONE output row.
// Layout: weights_buf has [d-array][qs-array]; per row r:
//   d[r, 0..nb] : 2·nb bytes starting at d_offset + r*nb*2
//   qs[r, 0..nb, 0..16]: nb*16 bytes starting at qs_offset + r*nb*16
// dispatch: (n, 1, 1)
//
// Single read_write binding to activations to avoid same-buffer aliasing
// across input and output bindings.

struct Params {
  k: u32,
  n: u32,
  d_offset: u32,
  qs_offset: u32,
  input_offset: u32,
  output_offset: u32,
  accumulate: u32,
  dispatch_x_dim: u32,  // size of X dispatch grid (so 2D wraparound for n > 65535)
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read_write> act: array<f32>;

fn load_f16_at(b_offset: u32) -> f32 {
  let word = weights[b_offset >> 2u];
  let half = (word >> ((b_offset & 2u) * 8u)) & 0xFFFFu;
  return unpack2x16float(half).x;
}

fn load_byte_at(b_offset: u32) -> u32 {
  let word = weights[b_offset >> 2u];
  return (word >> ((b_offset & 3u) * 8u)) & 0xFFu;
}

const WG: u32 = 64u;
var<workgroup> partial: array<f32, WG>;

@compute @workgroup_size(WG)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = wg.y * p.dispatch_x_dim + wg.x;
  if (row >= p.n) { return; }
  let tid = lid.x;
  let nb = p.k / 128u;

  let row_d_byte  = p.d_offset  + row * nb * 2u;
  let row_qs_byte = p.qs_offset + row * nb * 16u;

  var acc: f32 = 0.0;
  for (var b: u32 = tid; b < nb; b += WG) {
    let d = load_f16_at(row_d_byte + b * 2u);
    let qs_b = row_qs_byte + b * 16u;
    let x_base = p.input_offset + b * 128u;
    var sum_pos: f32 = 0.0;
    var sumy: f32 = 0.0;
    for (var k: u32 = 0u; k < 16u; k++) {
      let qb = load_byte_at(qs_b + k);
      for (var bit: u32 = 0u; bit < 8u; bit++) {
        let xv = act[x_base + k * 8u + bit];
        sumy = sumy + xv;
        if (((qb >> bit) & 1u) != 0u) {
          sum_pos = sum_pos + xv;
        }
      }
    }
    acc = acc + d * (2.0 * sum_pos - sumy);
  }

  partial[tid] = acc;
  workgroupBarrier();
  var step: u32 = WG / 2u;
  while (step > 0u) {
    if (tid < step) { partial[tid] = partial[tid] + partial[tid + step]; }
    workgroupBarrier();
    step = step / 2u;
  }
  if (tid == 0u) {
    let yi = p.output_offset + row;
    if (p.accumulate != 0u) {
      act[yi] = act[yi] + partial[0];
    } else {
      act[yi] = partial[0];
    }
  }
}
