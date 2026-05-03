enable f16;

// Multiply-free Q1_0 matvec.
// Multi-row workgroup: WG_X threads cooperate per row, ROWS_PER_WG rows per WG.
// Each thread accumulates ±xv per weight via select(), then scales by d.
// Layout: weights_buf has [d-array][qs-array]; per row r:
//   d[r, 0..nb] : 2·nb bytes starting at d_offset + r*nb*2
//   qs[r, 0..nb, 0..16]: nb*16 bytes starting at qs_offset + r*nb*16
//   (qs region is u32-aligned by extract.py's pad4)
//
// Activation `x` is staged into an LDS tile so the 8 ty-rows in each WG share
// loads instead of redundantly fetching from global per-row, and so the inner
// body can issue vec4<f16> reads to halve instruction count vs. scalar
// f16-per-bit.

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

// LDS tile size for the staged input activations. K values used by
// matvec_q1_0 dispatches (n_embd=2560, q_dim=4096, n_ff=9728) aren't all
// multiples of TILE_K, so the loop handles a partial trailing tile (K and
// all tile boundaries are multiples of 128 = the q1_0 block size).
// 2048 was empirically the sweet spot at 64 threads/WG on RDNA4 — 4 KiB of
// LDS keeps occupancy high while still amortizing redundant per-row reads.
const TILE_K: u32 = 2048u;
const TILE_VEC4: u32 = TILE_K / 4u;

var<workgroup> x_sh: array<vec4<f16>, TILE_VEC4>;

@compute @workgroup_size(WG_X, WG_Y)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let wg_idx = wg.y * p.dispatch_x_dim + wg.x;
  let row = wg_idx * ROWS_PER_WG + lid.y;
  let tx = lid.x;
  let ty = lid.y;
  let tid = ty * WG_X + tx;
  let valid = row < p.n;
  let nb = p.k / 128u;

  let row_d_byte  = p.d_offset  + row * nb * 2u;
  let row_qs_byte = p.qs_offset + row * nb * 16u;

  var acc: f32;

  for (var tile_start: u32 = 0u; tile_start < p.k; tile_start += TILE_K) {
    let tile_size = min(TILE_K, p.k - tile_start);
    let tile_v4 = tile_size >> 2u;
    let nb_tile = tile_size / 128u;

    // Cooperative vec4<f16> load of the activation tile into LDS.
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
        let x_v4_base = b_local * 32u;  // 128 elements / 4 = 32 vec4s per block
        // Accumulate in f16 across the block (max 128 ±xv values, well under
        // the f16 representable range for typical post-norm activations) so the
        // 32 inner adds stay packed and avoid the f16→f32 widening per step.
        var block_acc: vec4<f16> = vec4<f16>(0.0);
        for (var w: u32 = 0u; w < 4u; w++) {
          let qword = weights[qs_word_base + w];
          // Process 32 sign bits in 8 groups of 4, one vec4<f16> per group.
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

  // 8-lane row-wise reduction via subgroup-shuffle butterfly. Local invocation
  // index is `tx + ty*8`, so masks 1/2/4 stay within the current row's tx
  // bits. Requires SG_SIZE >= 8 and the row-major lid → subgroup_invocation_id
  // mapping (universal on AMD/NVIDIA/Intel/Apple); not specific to wave64.
  acc = acc + subgroupShuffleXor(acc, 1u);
  acc = acc + subgroupShuffleXor(acc, 2u);
  acc = acc + subgroupShuffleXor(acc, 4u);
  if (tx == 0u && valid) {
    let yi = p.output_offset + row;
    if (p.accumulate != 0u) {
      act[yi] = f16(f32(act[yi]) + acc);
    } else {
      act[yi] = f16(acc);
    }
  }
}
