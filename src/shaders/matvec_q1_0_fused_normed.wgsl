enable f16;
requires packed_4x8_integer_dot_product;

// Multi-range Q1_0 matvec **fused with the preceding RMS-norm + scale**:
// one dispatch produces output rows for 2 or 3 separate weight tensors that
// share the same K dimension and the same input activation. Used for the
// matvec single-token (tg) path's fused QKV (3 ranges) and gate+up (2 ranges)
// projections.
//
// Inner matmul uses dot4I8Packed: stage 1 stages `x*w_norm` to f16 LDS while
// accumulating ssq; after the cross-WG ssq reduce gives `inv_rms`, stage 2
// quantizes the staged values to Q8_0 (32-element blocks, FP32 d + 4 i8 per
// u32 packed); stage 3 runs the matmul as expand_4_bits(sign_nibble) ·
// dot4I8Packed against the staged Q8_0 activation, scaled by per-Q1_0-block
// d × per-Q8_0-block d. This replaces the previous `select(-xv, xv, mask) +
// vec4<f16> add` inner loop with one dot4 per 4 weights — same instruction
// count per nibble, but each nibble issues a single V_DOT4_I32_I8 instead of
// 4 cndmask + 2 v_pk_add, plus halves the per-block LDS-load bandwidth (i8
// activation reads as ds_read_b32 vs vec4<f16> as ds_read_b64).
//
// The Q1_0 d carries `inv_rms` baked in via `a_d = max_abs * inv_rms / 127`,
// so the matvec output equals `matvec(rms_norm(x) * w_norm)` exactly, modulo
// Q8_0 round-off (max relative error ~1/254 per element, well within the
// noise floor of the Q1_0 weights).
//
// Per-WG layout: WG_X=8 threads per row, ROWS_PER_WG=16 rows per WG (WG=128).
// Range sizes (n_0, n_1, n_2) must all be multiples of ROWS_PER_WG so no
// workgroup straddles a range boundary.

struct Params {
  k: u32,
  n_total: u32,
  input_offset: u32,
  dispatch_x_dim: u32,
  // norm
  w_norm_off: u32, eps: f32,
  // range 0
  d_offset_0: u32, qs_offset_0: u32, n_0: u32, output_offset_0: u32,
  // range 1
  d_offset_1: u32, qs_offset_1: u32, n_1: u32, output_offset_1: u32,
  // range 2 (set n_2=0 for the 2-range case)
  d_offset_2: u32, qs_offset_2: u32, n_2: u32, output_offset_2: u32,
};

var<immediate> p: Params;
@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read_write> act: array<f16>;
@group(0) @binding(2) var<storage, read>       w_norms: array<f16>;

fn load_f16_at(b_offset: u32) -> f32 {
  let word = weights[b_offset >> 2u];
  let half = (word >> ((b_offset & 2u) * 8u)) & 0xFFFFu;
  return unpack2x16float(half).x;
}

// Expand 4 sign bits to packed-i8 ±1 (0x01 if bit set, 0xFF if clear).
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

// `K_V4` = n_embd / 4, baked at load time per model so the LDS arrays fit
// the actual row exactly.
const K_V4: u32 = {{N_EMBD_V4}}u;
const K: u32 = K_V4 << 2u;
const NB_Q8: u32 = K / 32u;          // # Q8_0 blocks across K
const A_QS_LEN: u32 = K >> 2u;        // K bytes packed as u32 (= K_V4)

const SUBGROUP_MIN_SIZE: u32 = {{SUBGROUP_MIN_SIZE}}u;
const SG_PARTIAL_MAX: u32 = (WG + SUBGROUP_MIN_SIZE - 1u) / SUBGROUP_MIN_SIZE;

// f16 staging for `x * w_norm`. Used by stage 1 (load + ssq) and consumed by
// stage 2 (quantize). Dead after stage 2 — could in principle alias with
// a_qs_sh, but WGSL has no union/alias mechanism and the saved LDS isn't
// worth the indirection complexity here.
var<workgroup> x_sh: array<vec4<f16>, K_V4>;
// Q8_0 staged activation: K bytes of i8 packed 4 per u32. Layout is
// **transposed** vs. block-major (`[i_off * NB_Q8 + block_idx]` instead of
// `[block_idx * 8 + i_off]`) to dodge LDS bank conflicts: the matvec inner
// has WG_X=8 lanes-per-row reading the same `i_off` but different
// `block_idx` (= different `b_local`); block-major strides those 8 reads
// by 32 u32s → all 8 lanes map to one bank → 8-way serialization.
// Transposed, the 8 lanes' addresses differ by 4 → 8 unique banks.
var<workgroup> a_qs_sh: array<u32, A_QS_LEN>;
// Q8_0 per-block scales (FP32, with inv_rms folded in).
var<workgroup> a_d_sh: array<f32, NB_Q8>;
// Cross-subgroup merge slot for the RMS reduction (only used when num_subgroups > 1).
var<workgroup> sg_partial: array<f32, SG_PARTIAL_MAX>;

@compute @workgroup_size(WG)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_index) tid: u32,
  @builtin(subgroup_invocation_id) sg_inv_id: u32,
  @builtin(subgroup_id) sg_id: u32,
  @builtin(num_subgroups) num_subgroups: u32,
) {
  let wg_idx = wg.y * p.dispatch_x_dim + wg.x;
  let ty = tid / WG_X;
  let tx = tid % WG_X;
  let global_row = wg_idx * ROWS_PER_WG + ty;
  let valid = global_row < p.n_total;
  let nb_q1 = K / 128u;

  // ---- Stage 1: load x and w_norm; stage `x * w_norm` into x_sh; accumulate ssq.
  let in_v4_off = p.input_offset >> 2u;
  let w_v4_off = p.w_norm_off >> 2u;
  var ssq: f32 = 0.0;
  for (var v: u32 = tid; v < K_V4; v += WG) {
    let base = (in_v4_off + v) << 2u;
    let xv4 = vec4<f16>(
      act[base + 0u], act[base + 1u], act[base + 2u], act[base + 3u],
    );
    let xv4f = vec4<f32>(xv4);
    ssq = ssq + xv4f.x * xv4f.x + xv4f.y * xv4f.y + xv4f.z * xv4f.z + xv4f.w * xv4f.w;
    let w_base = (w_v4_off + v) << 2u;
    let wv4 = vec4<f16>(
      w_norms[w_base + 0u], w_norms[w_base + 1u],
      w_norms[w_base + 2u], w_norms[w_base + 3u],
    );
    x_sh[v] = xv4 * wv4;
  }

  // ---- Stage 1.5: workgroup-wide ssq sum (subgroupAdd + cross-subgroup merge) ----
  let sg_sum = subgroupAdd(ssq);
  var total: f32;
  if (num_subgroups == 1u) {
    total = sg_sum;
  } else {
    if (sg_inv_id == 0u) { sg_partial[sg_id] = sg_sum; }
    workgroupBarrier();
    if (sg_id == 0u) {
      var c: f32 = 0.0;
      if (sg_inv_id < num_subgroups) { c = sg_partial[sg_inv_id]; }
      let f = subgroupAdd(c);
      if (sg_inv_id == 0u) { sg_partial[0] = f; }
    }
    workgroupBarrier();
    total = sg_partial[0];
  }
  let inv_rms = inverseSqrt(total / f32(K) + p.eps);
  workgroupBarrier();

  // ---- Stage 2: quantize each Q8_0 block (32 elements = 8 vec4<f16>) of
  // `x * w_norm * inv_rms` to i8 (`a_qs_sh`) + FP32 d with inv_rms folded
  // (`a_d_sh`). Each thread owns one block; threads with tid >= NB_Q8 idle.
  // q[i] = round(127 * (x * w_norm) / max_abs_block) is invariant to inv_rms,
  // so we never multiply by inv_rms here — it's baked into a_d only.
  if (tid < NB_Q8) {
    let block_v4_base = tid * 8u;
    // amax over f16 staged values in f16 (V_PK_MAX_F16 on AMD); widen for the
    // f32 inverse / round chain, which still needs full precision near ±127.
    var max_abs_h: f16 = 0.0h;
    for (var i: u32 = 0u; i < 8u; i++) {
      let af = abs(x_sh[block_v4_base + i]);
      max_abs_h = max(max_abs_h, max(max(af.x, af.y), max(af.z, af.w)));
    }
    let max_abs = f32(max_abs_h);
    let d = max_abs * inv_rms * (1.0 / 127.0);
    a_d_sh[tid] = d;
    let inv_max = 127.0 / max(max_abs, 1.0e-30);
    for (var i: u32 = 0u; i < 8u; i++) {
      let v = vec4<f32>(x_sh[block_v4_base + i]);
      let q = clamp(round(v * inv_max), vec4<f32>(-128.0), vec4<f32>(127.0));
      let q0 = u32(i32(q.x)) & 0xFFu;
      let q1 = u32(i32(q.y)) & 0xFFu;
      let q2 = u32(i32(q.z)) & 0xFFu;
      let q3 = u32(i32(q.w)) & 0xFFu;
      // Transposed layout: i is the outer stride, block (= tid) the inner.
      a_qs_sh[i * NB_Q8 + tid] = q0 | (q1 << 8u) | (q2 << 16u) | (q3 << 24u);
    }
  }
  workgroupBarrier();

  // ---- Stage 3: matvec inner loop (full-row, no tiling — a_qs_sh covers k) ----
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

  let row_d_byte  = d_off  + local_row * nb_q1 * 2u;
  let row_qs_byte = qs_off + local_row * nb_q1 * 16u;

  var acc: f32 = 0.0;
  if (valid) {
    for (var b: u32 = tx; b < nb_q1; b += WG_X) {
      let d_w = load_f16_at(row_d_byte + b * 2u);
      let qs_word_base = (row_qs_byte + b * 16u) >> 2u;
      // Each Q1_0 block (128 weights) covers 4 Q8_0 sub-blocks.
      let a_block_base = b * 4u;
      var sub_acc: f32 = 0.0;
      for (var s: u32 = 0u; s < 4u; s++) {
        let qword = weights[qs_word_base + s];
        let a_d = a_d_sh[a_block_base + s];
        let block_l = a_block_base + s;
        var sumi: i32 = 0;
        // 8 dot4s × 4 weights = 32 weights = one Q8_0 sub-block.
        for (var i: u32 = 0u; i < 8u; i++) {
          let bits = (qword >> (i * 4u)) & 0xFu;
          let w_packed = expand_4_bits(bits);
          let a_packed = a_qs_sh[i * NB_Q8 + block_l];
          sumi = dot4I8Packed(w_packed, a_packed) + sumi;
        }
        sub_acc = sub_acc + a_d * f32(sumi);
      }
      acc = acc + d_w * sub_acc;
    }
  }

  // 8-lane row-wise reduction via subgroup-shuffle butterfly (see matvec_q1_0).
  acc = acc + subgroupShuffleXor(acc, 1u);
  acc = acc + subgroupShuffleXor(acc, 2u);
  acc = acc + subgroupShuffleXor(acc, 4u);
  if (tx == 0u && valid) {
    let yi = out_off + local_row;
    act[yi] = f16(acc);  // fused matvecs don't accumulate; outputs are fresh
  }
}
