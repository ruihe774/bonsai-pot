enable f16;
requires packed_4x8_integer_dot_product;

// Q1_0 matvec with an in-LDS Q8_0 activation quantize, so the inner loop runs
// as `expand_4_bits(sign_nibble) · dot4I8Packed(...)` against the staged i8
// activation. Used by the matvec-path Wo and lm_head dispatches.
//
// Per tile: each thread takes ownership of one Q8_0 block (32 elements = 8
// vec4<f16>), loads them from global, finds max-abs in registers, quantizes
// and writes to `a_qs_sh` + `a_d_sh`. After a workgroup barrier, the matvec
// inner over the tile runs the dot4I8Packed loop. No f16 activation staging
// in LDS — the loaded values stay in per-thread registers long enough to
// compute max-abs and quantize, freeing LDS for a wider tile.
//
// TILE_K=4096 with WG=128 gives 1 Q8_0 block per thread on a full tile.
// Wo (K=4096) fits in one tile; lm_head (K=2560) likewise (with 80/128
// threads active in the quantize phase). The trailing partial tile is
// handled by the `tid < nb_q8_tile` guard.

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

var<immediate> p: Params;
@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read_write> act: array<f16>;

fn load_f16_at(b_offset: u32) -> f32 {
  let word = weights[b_offset >> 2u];
  let half = (word >> ((b_offset & 2u) * 8u)) & 0xFFFFu;
  return unpack2x16float(half).x;
}

fn expand_4_bits(bits: u32) -> u32 {
  let spread = (bits & 1u)
             | ((bits & 2u) <<  7u)
             | ((bits & 4u) << 14u)
             | ((bits & 8u) << 21u);
  return ~(spread * 0xFEu);
}

const WG_X: u32 = 8u;
const WG_Y: u32 = 16u;
const WG: u32 = WG_X * WG_Y;
const ROWS_PER_WG: u32 = WG_Y;

const TILE_K: u32 = 4096u;
const TILE_NB_Q8: u32 = TILE_K / 32u;   // = 128, matches WG
const TILE_QS_LEN: u32 = TILE_K >> 2u;  // u32 slots packing the i8 activation

var<workgroup> a_qs_sh: array<u32, TILE_QS_LEN>;
var<workgroup> a_d_sh: array<f32, TILE_NB_Q8>;

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
  let nb_q1 = p.k / 128u;

  let row_d_byte  = p.d_offset  + row * nb_q1 * 2u;
  let row_qs_byte = p.qs_offset + row * nb_q1 * 16u;

  var acc: f32 = 0.0;

  for (var tile_start: u32 = 0u; tile_start < p.k; tile_start += TILE_K) {
    let tile_size = min(TILE_K, p.k - tile_start);
    let nb_q8_tile = tile_size / 32u;
    let nb_q1_tile = tile_size / 128u;

    // ---- Stage 1+2: each thread handles one Q8_0 block of the tile ----
    // Load 32 elements; find max-abs in registers; quantize; store to
    // a_qs_sh + a_d_sh.
    if (tid < nb_q8_tile) {
      let elem_base = tile_start + tid * 32u;
      let in_v4_off = (p.input_offset + elem_base) >> 2u;

      var v0: vec4<f16>; var v1: vec4<f16>; var v2: vec4<f16>; var v3: vec4<f16>;
      var v4: vec4<f16>; var v5: vec4<f16>; var v6: vec4<f16>; var v7: vec4<f16>;
      var max_abs: f32 = 0.0;

      // Inline-unrolled to keep the f16 vec4 reads in named scalars (so the
      // WGSL→SPIR-V→AMD-LLVM pipeline keeps them in registers, not LDS).
      {
        let b = (in_v4_off + 0u) << 2u;
        v0 = vec4<f16>(act[b], act[b+1u], act[b+2u], act[b+3u]);
        let af = abs(vec4<f32>(v0));
        max_abs = max(max_abs, max(max(af.x, af.y), max(af.z, af.w)));
      }
      {
        let b = (in_v4_off + 1u) << 2u;
        v1 = vec4<f16>(act[b], act[b+1u], act[b+2u], act[b+3u]);
        let af = abs(vec4<f32>(v1));
        max_abs = max(max_abs, max(max(af.x, af.y), max(af.z, af.w)));
      }
      {
        let b = (in_v4_off + 2u) << 2u;
        v2 = vec4<f16>(act[b], act[b+1u], act[b+2u], act[b+3u]);
        let af = abs(vec4<f32>(v2));
        max_abs = max(max_abs, max(max(af.x, af.y), max(af.z, af.w)));
      }
      {
        let b = (in_v4_off + 3u) << 2u;
        v3 = vec4<f16>(act[b], act[b+1u], act[b+2u], act[b+3u]);
        let af = abs(vec4<f32>(v3));
        max_abs = max(max_abs, max(max(af.x, af.y), max(af.z, af.w)));
      }
      {
        let b = (in_v4_off + 4u) << 2u;
        v4 = vec4<f16>(act[b], act[b+1u], act[b+2u], act[b+3u]);
        let af = abs(vec4<f32>(v4));
        max_abs = max(max_abs, max(max(af.x, af.y), max(af.z, af.w)));
      }
      {
        let b = (in_v4_off + 5u) << 2u;
        v5 = vec4<f16>(act[b], act[b+1u], act[b+2u], act[b+3u]);
        let af = abs(vec4<f32>(v5));
        max_abs = max(max_abs, max(max(af.x, af.y), max(af.z, af.w)));
      }
      {
        let b = (in_v4_off + 6u) << 2u;
        v6 = vec4<f16>(act[b], act[b+1u], act[b+2u], act[b+3u]);
        let af = abs(vec4<f32>(v6));
        max_abs = max(max_abs, max(max(af.x, af.y), max(af.z, af.w)));
      }
      {
        let b = (in_v4_off + 7u) << 2u;
        v7 = vec4<f16>(act[b], act[b+1u], act[b+2u], act[b+3u]);
        let af = abs(vec4<f32>(v7));
        max_abs = max(max_abs, max(max(af.x, af.y), max(af.z, af.w)));
      }

      let d = max_abs * (1.0 / 127.0);
      a_d_sh[tid] = d;
      let inv_max = 127.0 / max(max_abs, 1.0e-30);
      let qs_base = tid * 8u;
      let q0v = clamp(round(vec4<f32>(v0) * inv_max), vec4<f32>(-128.0), vec4<f32>(127.0));
      a_qs_sh[qs_base + 0u] = (u32(i32(q0v.x)) & 0xFFu) | ((u32(i32(q0v.y)) & 0xFFu) << 8u) | ((u32(i32(q0v.z)) & 0xFFu) << 16u) | ((u32(i32(q0v.w)) & 0xFFu) << 24u);
      let q1v = clamp(round(vec4<f32>(v1) * inv_max), vec4<f32>(-128.0), vec4<f32>(127.0));
      a_qs_sh[qs_base + 1u] = (u32(i32(q1v.x)) & 0xFFu) | ((u32(i32(q1v.y)) & 0xFFu) << 8u) | ((u32(i32(q1v.z)) & 0xFFu) << 16u) | ((u32(i32(q1v.w)) & 0xFFu) << 24u);
      let q2v = clamp(round(vec4<f32>(v2) * inv_max), vec4<f32>(-128.0), vec4<f32>(127.0));
      a_qs_sh[qs_base + 2u] = (u32(i32(q2v.x)) & 0xFFu) | ((u32(i32(q2v.y)) & 0xFFu) << 8u) | ((u32(i32(q2v.z)) & 0xFFu) << 16u) | ((u32(i32(q2v.w)) & 0xFFu) << 24u);
      let q3v = clamp(round(vec4<f32>(v3) * inv_max), vec4<f32>(-128.0), vec4<f32>(127.0));
      a_qs_sh[qs_base + 3u] = (u32(i32(q3v.x)) & 0xFFu) | ((u32(i32(q3v.y)) & 0xFFu) << 8u) | ((u32(i32(q3v.z)) & 0xFFu) << 16u) | ((u32(i32(q3v.w)) & 0xFFu) << 24u);
      let q4v = clamp(round(vec4<f32>(v4) * inv_max), vec4<f32>(-128.0), vec4<f32>(127.0));
      a_qs_sh[qs_base + 4u] = (u32(i32(q4v.x)) & 0xFFu) | ((u32(i32(q4v.y)) & 0xFFu) << 8u) | ((u32(i32(q4v.z)) & 0xFFu) << 16u) | ((u32(i32(q4v.w)) & 0xFFu) << 24u);
      let q5v = clamp(round(vec4<f32>(v5) * inv_max), vec4<f32>(-128.0), vec4<f32>(127.0));
      a_qs_sh[qs_base + 5u] = (u32(i32(q5v.x)) & 0xFFu) | ((u32(i32(q5v.y)) & 0xFFu) << 8u) | ((u32(i32(q5v.z)) & 0xFFu) << 16u) | ((u32(i32(q5v.w)) & 0xFFu) << 24u);
      let q6v = clamp(round(vec4<f32>(v6) * inv_max), vec4<f32>(-128.0), vec4<f32>(127.0));
      a_qs_sh[qs_base + 6u] = (u32(i32(q6v.x)) & 0xFFu) | ((u32(i32(q6v.y)) & 0xFFu) << 8u) | ((u32(i32(q6v.z)) & 0xFFu) << 16u) | ((u32(i32(q6v.w)) & 0xFFu) << 24u);
      let q7v = clamp(round(vec4<f32>(v7) * inv_max), vec4<f32>(-128.0), vec4<f32>(127.0));
      a_qs_sh[qs_base + 7u] = (u32(i32(q7v.x)) & 0xFFu) | ((u32(i32(q7v.y)) & 0xFFu) << 8u) | ((u32(i32(q7v.z)) & 0xFFu) << 16u) | ((u32(i32(q7v.w)) & 0xFFu) << 24u);
    }
    workgroupBarrier();

    // ---- Stage 3: matvec inner over this tile, dot4I8Packed style ----
    if (valid) {
      let b_q1_base_global = tile_start / 128u;
      for (var b_local: u32 = tx; b_local < nb_q1_tile; b_local += WG_X) {
        let b = b_q1_base_global + b_local;
        let d_w = load_f16_at(row_d_byte + b * 2u);
        let qs_word_base = (row_qs_byte + b * 16u) >> 2u;
        let a_block_local_base = b_local * 4u;
        var sub_acc: f32 = 0.0;
        for (var s: u32 = 0u; s < 4u; s++) {
          let qword = weights[qs_word_base + s];
          let a_d = a_d_sh[a_block_local_base + s];
          let qs_l_base = (a_block_local_base + s) * 8u;
          var sumi: i32 = 0;
          for (var i: u32 = 0u; i < 8u; i++) {
            let bits = (qword >> (i * 4u)) & 0xFu;
            let w_packed = expand_4_bits(bits);
            let a_packed = a_qs_sh[qs_l_base + i];
            sumi = dot4I8Packed(w_packed, a_packed) + sumi;
          }
          sub_acc = sub_acc + a_d * f32(sumi);
        }
        acc = acc + d_w * sub_acc;
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
