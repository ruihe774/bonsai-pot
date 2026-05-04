enable f16;

// Multi-range Q1_0 matvec **fused with the preceding RMS-norm + scale**:
// one dispatch produces output rows for 2 or 3 separate weight tensors that
// share the same K dimension and the same input activation. Used for the
// matvec single-token (tg) path's fused QKV (3 ranges) and gate+up (2 ranges)
// projections, where the activation `x` is preceded by an `rms_norm(x) * w`
// step. This kernel folds that step in by:
//   1. loading `x[i] * w_norm[i]` into shmem and accumulating Σ x[i]² in f32;
//   2. reducing the sum and computing `inv_rms = 1 / sqrt(mean(x²) + eps)`;
//   3. running the matvec inner loop with per-block scale `d * inv_rms` —
//      so Σ sign[i] · (x[i]·w_norm[i]) · (d·inv_rms) computes the same
//      output as `matvec(rms_norm(x) * w_norm)` would, with one global LDS
//      pass over x_sh and a single FMA bake of inv_rms into d.
// The 2 dispatches per layer that this replaces (rms_norm + matvec_q1_0_fused)
// collapse into 1 — saves 72 dispatches/step at n_layer=36 plus the global
// x_norm round-trip.
//
// Per-WG layout matches matvec_q1_0_fused.wgsl: WG_X=8 threads cooperate per
// row, ROWS_PER_WG=8 rows per workgroup. Range sizes (n_0, n_1, n_2) must all
// be multiples of ROWS_PER_WG so no workgroup straddles a range boundary.
// Activation `x` is staged into LDS in full (no tiling) so x_sh covers the
// full row of normed-and-weighted x for the inner loop.

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

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read_write> act: array<f16>;
@group(0) @binding(3) var<storage, read>       w_norms: array<f16>;

fn load_f16_at(b_offset: u32) -> f32 {
  let word = weights[b_offset >> 2u];
  let half = (word >> ((b_offset & 2u) * 8u)) & 0xFFFFu;
  return unpack2x16float(half).x;
}

const WG_X: u32 = 8u;
const WG_Y: u32 = 8u;
const WG: u32 = WG_X * WG_Y;
const ROWS_PER_WG: u32 = WG_Y;

// `K_V4` = n_embd / 4, baked at load time per model so x_sh fits the actual
// row exactly (5 KiB at 4B; 8 KiB at 8B). AMD RDNA allocates LDS in 1 KiB
// granularity per WG, so right-sizing matters for occupancy.
const K_V4: u32 = {{N_EMBD_V4}}u;

// SUBGROUP_MIN_SIZE is baked from adapter.subgroup_min_size at load time.
// SG_PARTIAL_MAX is the worst-case number of subgroups per workgroup.
const SUBGROUP_MIN_SIZE: u32 = {{SUBGROUP_MIN_SIZE}}u;
const SG_PARTIAL_MAX: u32 = (WG + SUBGROUP_MIN_SIZE - 1u) / SUBGROUP_MIN_SIZE;

var<workgroup> x_sh: array<vec4<f16>, K_V4>;
// Cross-subgroup merge slot for the RMS reduction (only used when num_subgroups > 1).
var<workgroup> sg_partial: array<f32, SG_PARTIAL_MAX>;

// 1-D workgroup so we can use `@builtin(subgroup_id)` /
// `@builtin(subgroup_invocation_id)` — naga rejects subgroup builtins on
// multi-dimensional workgroups. The (tx, ty) of the matvec_q1_0_fused 2-D
// layout is reconstructed from `local_invocation_index` below: `ty = tid /
// WG_X`, `tx = tid % WG_X`. Since `subgroup_invocation_id` increases linearly
// with `local_invocation_index` (true on AMD/NVIDIA/Intel/Apple), the per-row
// 8-lane subgroupShuffleXor butterfly used at the end still holds.
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
  // K is fixed = n_embd (baked via K_V4); the matvec single-token path's
  // fused QKV / gate+up callers always pass `k = n_embd`. We assert the
  // uniform `p.k` matches on the host side.
  let nb = (K_V4 << 2u) / 128u;
  let k_v4 = K_V4;

  // ---- Stage 1: load x into shmem PRE-MULTIPLIED BY w_norm, accumulate ssq.
  // We bake w_norm into x_sh during the load, then bake inv_rms into the
  // matvec's per-block `d` scale once below — so the multiply-free inner
  // loop only sees `(x * w_norm)` in shmem and a single `d_eff = d * inv_rms`
  // per block. This avoids a second pass through x_sh (which would add
  // ~k_v4/WG worth of LDS round-trip per WG; with 2432 gate_up WGs that's
  // measurable). ssq is computed from the raw x (before w_norm scale).
  let in_v4_off = p.input_offset >> 2u;
  let w_v4_off = p.w_norm_off >> 2u;
  var ssq: f32 = 0.0;
  for (var v: u32 = tid; v < k_v4; v += WG) {
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

  // ---- Stage 2: workgroup-wide sum (subgroupAdd + cross-subgroup merge) ----
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
  let inv_rms = inverseSqrt(total / f32(K_V4 << 2u) + p.eps);
  workgroupBarrier();

  // ---- Stage 3: matvec inner loop (full-row, no tiling — x_sh covers k) ----
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
  if (valid) {
    for (var b: u32 = tx; b < nb; b += WG_X) {
      // d_eff = d * inv_rms — folds the RMS-norm scale into the per-block
      // weight scale. Combined with `x_sh` already containing `x * w_norm`,
      // the inner accumulator computes Σ sign[i] * (x*w_norm)[i] * (d*inv_rms)
      //   = Σ sign[i] * x[i] * inv_rms * w_norm[i] * d
      //   = ⟨normed_x, q1_0_row⟩, exactly the rms_norm + matvec composition.
      let d_eff = load_f16_at(row_d_byte + b * 2u) * inv_rms;
      let qs_word_base = (row_qs_byte + b * 16u) >> 2u;
      let x_v4_base = b * 32u;
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
      acc = acc + d_eff * lane_sum;
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
