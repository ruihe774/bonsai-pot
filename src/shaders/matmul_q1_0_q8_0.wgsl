enable f16;
requires packed_4x8_integer_dot_product;

// dot4I8Packed matmul: Q1_0 weights × Q8_0 activations -> f16 output.
//
// 64×64 tile, 256 threads, each thread = 4×4 = 16 cells.

struct Params {
  k: u32,
  n: u32,
  m: u32,
  w_d_offset: u32,
  w_qs_offset: u32,
  a_d_offset: u32,
  a_qs_offset: u32,
  out_offset: u32,
  accumulate: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> acts: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f16>;

const WG_N: u32 = 16u;
const WG_M: u32 = 16u;
const WG: u32 = 256u;
const TN: u32 = 4u;
const TM: u32 = 4u;
const TILE_N: u32 = 64u;
const TILE_M: u32 = 64u;

var<workgroup> w_d_lds:    array<f32, 64>;
var<workgroup> w_bits_lds: array<u32, 256>;
var<workgroup> a_d_lds:    array<f32, 256>;
// a_qs[m_local*32 + (s*8+u)] — stride-32 between m's (consecutive su = consecutive banks).
var<workgroup> a_qs_lds:   array<u32, 2048>;

fn load_w_f16(b_offset: u32) -> f32 {
  let word = weights[b_offset >> 2u];
  let half = (word >> ((b_offset & 2u) * 8u)) & 0xFFFFu;
  return unpack2x16float(half).x;
}

// Expand 4 sign bits to packed-i8 ±1.  Each byte: 0x01 if bit set, 0xFF if clear.
//
// Trick: spread the 4 bits to byte LSBs (0x01 per "set" byte), multiply by 0xFE
// to lift to 0xFE per set byte, then bit-invert to get 0x01 / 0xFF per byte.
fn expand_4_bits(bits: u32) -> u32 {
  let spread = (bits & 1u)
             | ((bits & 2u) <<  7u)
             | ((bits & 4u) << 14u)
             | ((bits & 8u) << 21u);
  return ~(spread * 0xFEu);
}

@compute @workgroup_size(WG, 1, 1)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_index) tid: u32,
) {
  let n_base = wg.x * TILE_N;
  let m_base = wg.y * TILE_M;
  let nb_q1 = p.k / 128u;
  let nb_q8 = p.k / 32u;

  let lx = tid % WG_N;
  let ly = tid / WG_N;

  // 4×4 register-tile accumulator. Stored as 16 named scalars rather than
  // `var acc: array<f32, 16>` because AMD's Windows shader-compilation stack
  // (both AMDVLK / Vulkan and DXIL / DX12) miscompiles function-scope
  // `array<f32, N>` whenever it's accessed by a dynamic index anywhere in
  // the function — the array gets lowered to spill/scratch memory and
  // produces garbage outputs. Linux RADV is unaffected. Replacing with
  // scalars keeps every value in registers across all backends.
  var acc00: f32; var acc01: f32; var acc02: f32; var acc03: f32;
  var acc10: f32; var acc11: f32; var acc12: f32; var acc13: f32;
  var acc20: f32; var acc21: f32; var acc22: f32; var acc23: f32;
  var acc30: f32; var acc31: f32; var acc32: f32; var acc33: f32;

  for (var b: u32 = 0u; b < nb_q1; b = b + 1u) {
    // ---- Cooperative loads ----
    if (tid < 64u) {
      let n_idx = n_base + tid;
      var v: f32 = 0.0;
      if (n_idx < p.n) {
        v = load_w_f16(p.w_d_offset + (n_idx * nb_q1 + b) * 2u);
      }
      w_d_lds[tid] = v;
    }
    {
      let n_local = tid / 4u;
      let s = tid % 4u;
      let n_idx = n_base + n_local;
      var v: u32 = 0u;
      if (n_idx < p.n) {
        let off = p.w_qs_offset + n_idx * (nb_q1 * 16u) + b * 16u + s * 4u;
        v = weights[off >> 2u];
      }
      w_bits_lds[s * 64u + n_local] = v;
    }
    {
      let s = tid / 64u;
      let m_local = tid % 64u;
      let m_idx = m_base + m_local;
      var v: f32 = 0.0;
      if (m_idx < p.m) {
        let a_block = b * 4u + s;
        let off = p.a_d_offset + (m_idx * nb_q8 + a_block) * 4u;
        v = bitcast<f32>(acts[off >> 2u]);
      }
      a_d_lds[s * 64u + m_local] = v;
    }
    for (var li: u32 = 0u; li < 8u; li = li + 1u) {
      let idx = li * WG + tid;
      let m_local = idx / 32u;
      let su = idx % 32u;
      let m_idx = m_base + m_local;
      var v: u32 = 0u;
      if (m_idx < p.m) {
        let off = p.a_qs_offset + m_idx * p.k + b * 128u + su * 4u;
        v = acts[off >> 2u];
      }
      a_qs_lds[m_local * 32u + su] = v;
    }
    workgroupBarrier();

    let wd0 = w_d_lds[lx * TN + 0u];
    let wd1 = w_d_lds[lx * TN + 1u];
    let wd2 = w_d_lds[lx * TN + 2u];
    let wd3 = w_d_lds[lx * TN + 3u];

    for (var s: u32 = 0u; s < 4u; s = s + 1u) {
      var w0_0: u32; var w0_1: u32; var w0_2: u32; var w0_3: u32;
      var w0_4: u32; var w0_5: u32; var w0_6: u32; var w0_7: u32;
      var w1_0: u32; var w1_1: u32; var w1_2: u32; var w1_3: u32;
      var w1_4: u32; var w1_5: u32; var w1_6: u32; var w1_7: u32;
      var w2_0: u32; var w2_1: u32; var w2_2: u32; var w2_3: u32;
      var w2_4: u32; var w2_5: u32; var w2_6: u32; var w2_7: u32;
      var w3_0: u32; var w3_1: u32; var w3_2: u32; var w3_3: u32;
      var w3_4: u32; var w3_5: u32; var w3_6: u32; var w3_7: u32;
      {
        let bw0 = w_bits_lds[s * 64u + lx * TN + 0u];
        w0_0 = expand_4_bits((bw0      ) & 0xFu);
        w0_1 = expand_4_bits((bw0 >>  4u) & 0xFu);
        w0_2 = expand_4_bits((bw0 >>  8u) & 0xFu);
        w0_3 = expand_4_bits((bw0 >> 12u) & 0xFu);
        w0_4 = expand_4_bits((bw0 >> 16u) & 0xFu);
        w0_5 = expand_4_bits((bw0 >> 20u) & 0xFu);
        w0_6 = expand_4_bits((bw0 >> 24u) & 0xFu);
        w0_7 = expand_4_bits((bw0 >> 28u) & 0xFu);
      }
      {
        let bw1 = w_bits_lds[s * 64u + lx * TN + 1u];
        w1_0 = expand_4_bits((bw1      ) & 0xFu);
        w1_1 = expand_4_bits((bw1 >>  4u) & 0xFu);
        w1_2 = expand_4_bits((bw1 >>  8u) & 0xFu);
        w1_3 = expand_4_bits((bw1 >> 12u) & 0xFu);
        w1_4 = expand_4_bits((bw1 >> 16u) & 0xFu);
        w1_5 = expand_4_bits((bw1 >> 20u) & 0xFu);
        w1_6 = expand_4_bits((bw1 >> 24u) & 0xFu);
        w1_7 = expand_4_bits((bw1 >> 28u) & 0xFu);
      }
      {
        let bw2 = w_bits_lds[s * 64u + lx * TN + 2u];
        w2_0 = expand_4_bits((bw2      ) & 0xFu);
        w2_1 = expand_4_bits((bw2 >>  4u) & 0xFu);
        w2_2 = expand_4_bits((bw2 >>  8u) & 0xFu);
        w2_3 = expand_4_bits((bw2 >> 12u) & 0xFu);
        w2_4 = expand_4_bits((bw2 >> 16u) & 0xFu);
        w2_5 = expand_4_bits((bw2 >> 20u) & 0xFu);
        w2_6 = expand_4_bits((bw2 >> 24u) & 0xFu);
        w2_7 = expand_4_bits((bw2 >> 28u) & 0xFu);
      }
      {
        let bw3 = w_bits_lds[s * 64u + lx * TN + 3u];
        w3_0 = expand_4_bits((bw3      ) & 0xFu);
        w3_1 = expand_4_bits((bw3 >>  4u) & 0xFu);
        w3_2 = expand_4_bits((bw3 >>  8u) & 0xFu);
        w3_3 = expand_4_bits((bw3 >> 12u) & 0xFu);
        w3_4 = expand_4_bits((bw3 >> 16u) & 0xFu);
        w3_5 = expand_4_bits((bw3 >> 20u) & 0xFu);
        w3_6 = expand_4_bits((bw3 >> 24u) & 0xFu);
        w3_7 = expand_4_bits((bw3 >> 28u) & 0xFu);
      }

      // Inner `tm` loop fully unrolled — see the acc declaration above for
      // why we don't use a function-scope `array<f32, 16>` with a dynamic
      // index into it.
      for (var tm: u32 = 0u; tm < TM; tm = tm + 1u) {
        let m_local = ly * TM + tm;
        let a_d = a_d_lds[s * 64u + m_local];
        let a_base = m_local * 32u + s * 8u;
        let a0 = a_qs_lds[a_base + 0u];
        let a1 = a_qs_lds[a_base + 1u];
        let a2 = a_qs_lds[a_base + 2u];
        let a3 = a_qs_lds[a_base + 3u];
        let a4 = a_qs_lds[a_base + 4u];
        let a5 = a_qs_lds[a_base + 5u];
        let a6 = a_qs_lds[a_base + 6u];
        let a7 = a_qs_lds[a_base + 7u];

        var sumi0: i32 = dot4I8Packed(w0_0, a0);
        var sumi1: i32 = dot4I8Packed(w1_0, a0);
        var sumi2: i32 = dot4I8Packed(w2_0, a0);
        var sumi3: i32 = dot4I8Packed(w3_0, a0);
        sumi0 = dot4I8Packed(w0_1, a1) + sumi0;
        sumi1 = dot4I8Packed(w1_1, a1) + sumi1;
        sumi2 = dot4I8Packed(w2_1, a1) + sumi2;
        sumi3 = dot4I8Packed(w3_1, a1) + sumi3;
        sumi0 = dot4I8Packed(w0_2, a2) + sumi0;
        sumi1 = dot4I8Packed(w1_2, a2) + sumi1;
        sumi2 = dot4I8Packed(w2_2, a2) + sumi2;
        sumi3 = dot4I8Packed(w3_2, a2) + sumi3;
        sumi0 = dot4I8Packed(w0_3, a3) + sumi0;
        sumi1 = dot4I8Packed(w1_3, a3) + sumi1;
        sumi2 = dot4I8Packed(w2_3, a3) + sumi2;
        sumi3 = dot4I8Packed(w3_3, a3) + sumi3;
        sumi0 = dot4I8Packed(w0_4, a4) + sumi0;
        sumi1 = dot4I8Packed(w1_4, a4) + sumi1;
        sumi2 = dot4I8Packed(w2_4, a4) + sumi2;
        sumi3 = dot4I8Packed(w3_4, a4) + sumi3;
        sumi0 = dot4I8Packed(w0_5, a5) + sumi0;
        sumi1 = dot4I8Packed(w1_5, a5) + sumi1;
        sumi2 = dot4I8Packed(w2_5, a5) + sumi2;
        sumi3 = dot4I8Packed(w3_5, a5) + sumi3;
        sumi0 = dot4I8Packed(w0_6, a6) + sumi0;
        sumi1 = dot4I8Packed(w1_6, a6) + sumi1;
        sumi2 = dot4I8Packed(w2_6, a6) + sumi2;
        sumi3 = dot4I8Packed(w3_6, a6) + sumi3;
        sumi0 = dot4I8Packed(w0_7, a7) + sumi0;
        sumi1 = dot4I8Packed(w1_7, a7) + sumi1;
        sumi2 = dot4I8Packed(w2_7, a7) + sumi2;
        sumi3 = dot4I8Packed(w3_7, a7) + sumi3;

        let v0 = a_d * wd0 * f32(sumi0);
        let v1 = a_d * wd1 * f32(sumi1);
        let v2 = a_d * wd2 * f32(sumi2);
        let v3 = a_d * wd3 * f32(sumi3);
        switch (tm) {
          case 0u: { acc00 = acc00 + v0; acc01 = acc01 + v1; acc02 = acc02 + v2; acc03 = acc03 + v3; }
          case 1u: { acc10 = acc10 + v0; acc11 = acc11 + v1; acc12 = acc12 + v2; acc13 = acc13 + v3; }
          case 2u: { acc20 = acc20 + v0; acc21 = acc21 + v1; acc22 = acc22 + v2; acc23 = acc23 + v3; }
          default: { acc30 = acc30 + v0; acc31 = acc31 + v1; acc32 = acc32 + v2; acc33 = acc33 + v3; }
        }
      }
    }
    workgroupBarrier();
  }

  for (var tm: u32 = 0u; tm < TM; tm = tm + 1u) {
    let m_idx = m_base + ly * TM + tm;
    if (m_idx >= p.m) { continue; }
    for (var tn: u32 = 0u; tn < TN; tn = tn + 1u) {
      let n_idx = n_base + lx * TN + tn;
      if (n_idx >= p.n) { continue; }
      let yi = p.out_offset + m_idx * p.n + n_idx;
      var val: f32 = 0.0;
      switch (tm) {
        case 0u: {
          switch (tn) {
            case 0u: { val = acc00; } case 1u: { val = acc01; }
            case 2u: { val = acc02; } default: { val = acc03; }
          }
        }
        case 1u: {
          switch (tn) {
            case 0u: { val = acc10; } case 1u: { val = acc11; }
            case 2u: { val = acc12; } default: { val = acc13; }
          }
        }
        case 2u: {
          switch (tn) {
            case 0u: { val = acc20; } case 1u: { val = acc21; }
            case 2u: { val = acc22; } default: { val = acc23; }
          }
        }
        default: {
          switch (tn) {
            case 0u: { val = acc30; } case 1u: { val = acc31; }
            case 2u: { val = acc32; } default: { val = acc33; }
          }
        }
      }
      if (p.accumulate != 0u) {
        y[yi] = f16(f32(y[yi]) + val);
      } else {
        y[yi] = f16(val);
      }
    }
  }
}
