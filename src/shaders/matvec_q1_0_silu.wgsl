enable f16;

// Multiply-free Q1_0 matvec, with the SiLU·gate × up activation fold on the
// input side: instead of reading `xv` from a single staged region, this kernel
// computes `xv = (g * sigmoid(g)) * u` per element, where `g` and `u` come
// from the `gate` and `up` regions of `act` respectively. Used for the
// matvec-path Wd/ffn_down dispatch, replacing the previous
// `silu_mul -> ffn_in -> matvec_q1_0(Wd)` pair.
//
// Sigmoid is computed in f32 to avoid the f16 overflow risk that would arise
// from `1 / (1 + exp(-g))` when `g` is large-negative — matches the standalone
// `silu_mul.wgsl` formula exactly.
//
// All other structure (LDS-staged x tile, multi-row WG, subgroupShuffleXor
// row reduction, accumulate-or-write final) mirrors `matvec_q1_0.wgsl`.

struct Params {
  k: u32,
  n: u32,
  d_offset: u32,
  qs_offset: u32,
  gate_offset: u32,
  up_offset: u32,
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
const WG_Y: u32 = 16u;
const WG: u32 = WG_X * WG_Y;
const ROWS_PER_WG: u32 = WG_Y;

// Same TILE_K as matvec_q1_0 — Wd's K=n_ff=9728 is a multiple of 128 (the
// q1_0 block size) but not of TILE_K=2048; the trailing-partial-tile handling
// mirrors the unfused kernel.
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
    // Each lane fuses silu(gate[i]) * up[i] across 4 elements.
    let g_v4_off = (p.gate_offset + tile_start) >> 2u;
    let u_v4_off = (p.up_offset + tile_start) >> 2u;
    for (var v: u32 = tid; v < tile_v4; v += WG) {
      let g_base = (g_v4_off + v) << 2u;
      let u_base = (u_v4_off + v) << 2u;
      let g = vec4<f32>(
        f32(act[g_base + 0u]),
        f32(act[g_base + 1u]),
        f32(act[g_base + 2u]),
        f32(act[g_base + 3u]),
      );
      let u = vec4<f16>(
        act[u_base + 0u], act[u_base + 1u], act[u_base + 2u], act[u_base + 3u],
      );
      // silu(g) = g * sigmoid(g); sigmoid in f32 to avoid f16 exp underflow.
      let silu = g / (vec4<f32>(1.0) + exp(-g));
      x_sh[v] = vec4<f16>(silu) * u;
    }
    workgroupBarrier();

    if (valid) {
      let b_base = tile_start / 128u;
      for (var b_local: u32 = tx; b_local < nb_tile; b_local += WG_X) {
        let b = b_base + b_local;
        let d = load_f16_at(row_d_byte + b * 2u);
        let qs_word_base = (row_qs_byte + b * 16u) >> 2u;
        let x_v4_base = b_local * 32u;  // 128 elements / 4 = 32 vec4s per block
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

  // 8-lane row-wise reduction via subgroup-shuffle butterfly.
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
