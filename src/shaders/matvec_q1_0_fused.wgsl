enable f16;

// Multi-range Q1_0 matvec: one dispatch produces output rows belonging to 2
// or 3 separate weight tensors that share the same K dimension and the same
// input activation. Used for fused Q/K/V projection (3 ranges) and fused
// gate/up projection (2 ranges).
//
// Per-WG layout matches matvec_q1_0.wgsl: WG_X=8 threads cooperate per row,
// ROWS_PER_WG=8 rows per workgroup. Range sizes (n_0, n_1, n_2) must all be
// multiples of ROWS_PER_WG so no workgroup straddles a range boundary.
// Activation `x` is staged into LDS in tiles to share loads across ty-rows
// and to enable vec4<f16> reads in the inner loop.

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
const WG: u32 = WG_X * WG_Y;
const ROWS_PER_WG: u32 = WG_Y;

// Fused matvec is only ever called with K = n_embd = 2560, but the loop and
// partial-tile handling are written generically (matches matvec_q1_0.wgsl).
const TILE_K: u32 = 2048u;
const TILE_VEC4: u32 = TILE_K / 4u;

var<workgroup> x_sh: array<vec4<f16>, TILE_VEC4>;

@compute @workgroup_size(WG_X, WG_Y)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let wg_idx = wg.y * p.dispatch_x_dim + wg.x;
  let global_row = wg_idx * ROWS_PER_WG + lid.y;
  let tx = lid.x;
  let ty = lid.y;
  let tid = ty * WG_X + tx;
  let valid = global_row < p.n_total;
  let nb = p.k / 128u;

  // Resolve which range this row belongs to. Range sizes are guaranteed to be
  // multiples of ROWS_PER_WG, so the entire WG falls in a single range and the
  // branches below are uniform across the workgroup.
  var d_off: u32;
  var qs_off: u32;
  var out_off: u32;
  var local_row: u32;
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

  let row_d_byte  = d_off  + local_row * nb * 2u;
  let row_qs_byte = qs_off + local_row * nb * 16u;

  var acc: f32;

  for (var tile_start: u32 = 0u; tile_start < p.k; tile_start += TILE_K) {
    let tile_size = min(TILE_K, p.k - tile_start);
    let tile_v4 = tile_size >> 2u;
    let nb_tile = tile_size / 128u;

    let in_v4_off = (p.input_offset + tile_start) >> 2u;
    for (var v: u32 = tid; v < tile_v4; v += WG) {
      let base = (in_v4_off + v) << 2u;
      x_sh[v] = vec4<f16>(
        act[base + 0u], act[base + 1u], act[base + 2u], act[base + 3u],
      );
    }
    workgroupBarrier();

    if (valid) {
      let b_base = tile_start / 128u;
      for (var b_local: u32 = tx; b_local < nb_tile; b_local += WG_X) {
        let b = b_base + b_local;
        let d = load_f16_at(row_d_byte + b * 2u);
        let qs_word_base = (row_qs_byte + b * 16u) >> 2u;
        let x_v4_base = b_local * 32u;
        // f16 accumulator across the 128-weight block — see matvec_q1_0.wgsl.
        var block_acc: vec4<f16> = vec4<f16>(0.0);
        for (var w: u32 = 0u; w < 4u; w++) {
          let qword = weights[qs_word_base + w];
          for (var i: u32 = 0u; i < 8u; i++) {
            let bits = (qword >> (i * 4u)) & 0xFu;
            let mask4 = vec4<bool>(
              (bits & 1u) != 0u,
              (bits & 2u) != 0u,
              (bits & 4u) != 0u,
              (bits & 8u) != 0u,
            );
            let xv4 = x_sh[x_v4_base + w * 8u + i];
            block_acc += select(-xv4, xv4, mask4);
          }
        }
        let lane_sum = f32(block_acc.x + block_acc.y + block_acc.z + block_acc.w);
        acc = acc + d * lane_sum;
      }
    }
    workgroupBarrier();
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
