enable f16;

// Multi-range Q1_0 matvec: one dispatch produces output rows belonging to 2
// or 3 separate weight tensors that share the same K dimension and the same
// input activation. Used for fused Q/K/V projection (3 ranges) and fused
// gate/up projection (2 ranges).
//
// Per-WG layout matches matvec_q1_0.wgsl: WG_X=8 threads cooperate per row,
// ROWS_PER_WG=8 rows per workgroup. Range sizes (n_0, n_1, n_2) must all be
// multiples of ROWS_PER_WG so no workgroup straddles a range boundary.

struct Params {
  k: u32,
  n_total: u32,
  input_offset: u32,
  dispatch_x_dim: u32,
  // range 0
  d_offset_0: u32, qs_offset_0: u32, n_0: u32, output_offset_0: u32,
  // range 1
  d_offset_1: u32, qs_offset_1: u32, n_1: u32, output_offset_1: u32,
  // range 2 (set n_2=0 for the 2-range case)
  d_offset_2: u32, qs_offset_2: u32, n_2: u32, output_offset_2: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read_write> act: array<f16>;

fn load_f16_at(b_offset: u32) -> f32 {
  let word = weights[b_offset >> 2u];
  let half = (word >> ((b_offset & 2u) * 8u)) & 0xFFFFu;
  return unpack2x16float(half).x;
}

const WG_X: u32 = 8u;
const WG_Y: u32 = 8u;
const ROWS_PER_WG: u32 = WG_Y;

@compute @workgroup_size(WG_X, WG_Y)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let wg_idx = wg.y * p.dispatch_x_dim + wg.x;
  let global_row = wg_idx * ROWS_PER_WG + lid.y;
  let tx = lid.x;
  let ty = lid.y;
  let valid = global_row < p.n_total;
  let nb = p.k / 128u;

  // Resolve which range this row belongs to. Range sizes are guaranteed to be
  // multiples of ROWS_PER_WG, so the entire WG falls in a single range and the
  // branches below are uniform across the workgroup.
  var d_off: u32 = 0u;
  var qs_off: u32 = 0u;
  var out_off: u32 = 0u;
  var local_row: u32 = 0u;
  if (valid) {
    if (global_row < p.n_0) {
      d_off = p.d_offset_0; qs_off = p.qs_offset_0;
      out_off = p.output_offset_0; local_row = global_row;
    } else if (global_row < p.n_0 + p.n_1) {
      d_off = p.d_offset_1; qs_off = p.qs_offset_1;
      out_off = p.output_offset_1; local_row = global_row - p.n_0;
    } else {
      d_off = p.d_offset_2; qs_off = p.qs_offset_2;
      out_off = p.output_offset_2; local_row = global_row - p.n_0 - p.n_1;
    }
  }

  var acc: f32 = 0.0;
  if (valid) {
    let row_d_byte  = d_off  + local_row * nb * 2u;
    let row_qs_byte = qs_off + local_row * nb * 16u;
    for (var b: u32 = tx; b < nb; b += WG_X) {
      let d = load_f16_at(row_d_byte + b * 2u);
      let qs_word_base = (row_qs_byte + b * 16u) >> 2u;
      let x_base = p.input_offset + b * 128u;
      var block_acc: f32 = 0.0;
      for (var w: u32 = 0u; w < 4u; w++) {
        let qword = weights[qs_word_base + w];
        let x_word_off = x_base + w * 32u;
        for (var i: u32 = 0u; i < 32u; i++) {
          let xv = f32(act[x_word_off + i]);
          let bit_set = ((qword >> i) & 1u) != 0u;
          block_acc = block_acc + select(-xv, xv, bit_set);
        }
      }
      acc = acc + d * block_acc;
    }
  }

  // 8-lane row-wise reduction via subgroup-shuffle butterfly (see matvec_q1_0
  // for the SG_SIZE / lid-mapping assumptions).
  acc = acc + subgroupShuffleXor(acc, 1u);
  acc = acc + subgroupShuffleXor(acc, 2u);
  acc = acc + subgroupShuffleXor(acc, 4u);
  if (tx == 0u && valid) {
    let yi = out_off + local_row;
    act[yi] = f16(acc);  // fused matvecs don't accumulate; outputs are fresh
  }
}
