// Multiply-free Q1_0 matvec.
// Multi-row workgroup: WG_X threads cooperate per row, ROWS_PER_WG rows per WG.
// Each thread accumulates ±xv per weight via select(), then scales by d.
// Layout: weights_buf has [d-array][qs-array]; per row r:
//   d[r, 0..nb] : 2·nb bytes starting at d_offset + r*nb*2
//   qs[r, 0..nb, 0..16]: nb*16 bytes starting at qs_offset + r*nb*16
//   (qs region is u32-aligned by extract.py's pad4)

struct Params {
  k: u32,
  n: u32,
  d_offset: u32,
  qs_offset: u32,
  input_offset: u32,
  output_offset: u32,
  accumulate: u32,
  dispatch_x_dim: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read_write> act: array<f32>;

fn load_f16_at(b_offset: u32) -> f32 {
  let word = weights[b_offset >> 2u];
  let half = (word >> ((b_offset & 2u) * 8u)) & 0xFFFFu;
  return unpack2x16float(half).x;
}

const WG_X: u32 = 8u;
const WG_Y: u32 = 8u;
const ROWS_PER_WG: u32 = WG_Y;

var<workgroup> partial: array<f32, 64u>;

@compute @workgroup_size(WG_X, WG_Y)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let wg_idx = wg.y * p.dispatch_x_dim + wg.x;
  let row = wg_idx * ROWS_PER_WG + lid.y;
  let tx = lid.x;
  let ty = lid.y;
  let valid = row < p.n;
  let nb = p.k / 128u;

  var acc: f32 = 0.0;
  if (valid) {
    let row_d_byte  = p.d_offset  + row * nb * 2u;
    let row_qs_byte = p.qs_offset + row * nb * 16u;
    for (var b: u32 = tx; b < nb; b += WG_X) {
      let d = load_f16_at(row_d_byte + b * 2u);
      let qs_word_base = (row_qs_byte + b * 16u) >> 2u;  // 4 u32 words per block
      let x_base = p.input_offset + b * 128u;
      var block_acc: f32 = 0.0;
      for (var w: u32 = 0u; w < 4u; w++) {
        let qword = weights[qs_word_base + w];
        let x_word_off = x_base + w * 32u;
        for (var i: u32 = 0u; i < 32u; i++) {
          let xv = act[x_word_off + i];
          let bit_set = ((qword >> i) & 1u) != 0u;
          block_acc = block_acc + select(-xv, xv, bit_set);
        }
      }
      acc = acc + d * block_acc;
    }
  }

  let slot = ty * WG_X + tx;
  partial[slot] = acc;
  workgroupBarrier();
  var step: u32 = WG_X / 2u;
  while (step > 0u) {
    if (tx < step) {
      partial[slot] = partial[slot] + partial[slot + step];
    }
    workgroupBarrier();
    step = step / 2u;
  }
  if (tx == 0u && valid) {
    let yi = p.output_offset + row;
    if (p.accumulate != 0u) {
      act[yi] = act[yi] + partial[ty * WG_X];
    } else {
      act[yi] = partial[ty * WG_X];
    }
  }
}
