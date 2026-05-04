enable f16;

// RMSNorm: y = x * rsqrt(mean(x^2) + eps) * w
// Single binding to the activations buffer; input and output regions live at
// different offsets within it. This avoids the wgpu rule that disallows the
// same buffer being bound as both read and read_write within one dispatch.
// For per-head Q-/K-norm: n_groups=n_head (or n_kv_head), group_size=head_dim,
// `w[group_size]` reused across all groups.
// dispatch: (n_groups, 1, 1)

struct Params {
  group_size: u32,
  n_groups: u32,
  input_offset: u32,
  output_offset: u32,
  weight_offset: u32,
  eps: f32,
  _p0: u32, _p1: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> act: array<f16>;
@group(0) @binding(2) var<storage, read> w: array<f16>;

// SUBGROUP_MIN_SIZE is baked from adapter.subgroup_min_size at load time.
// SG_PARTIAL_MAX is the worst-case number of subgroups per workgroup, used
// to size the cross-subgroup shmem array. The actual count (num_subgroups)
// is read from @builtin(num_subgroups) at runtime.
const SUBGROUP_MIN_SIZE: u32 = {{SUBGROUP_MIN_SIZE}}u;
// 512 threads / WG = 8 wave64. For the per-layer n_embd norm (single WG over
// 2560 elements, called 73× per token) this is the only knob that controls
// in-flight memory ops on the one CU running the WG; bumping from 64 → 512
// lets that CU keep 8× more loads outstanding, which is what closes the gap
// vs. llama.cpp's BLOCK_SIZE=512 rms_norm and is by itself worth ~+10% tg t/s.
const WG: u32 = 512u;
const SG_PARTIAL_MAX: u32 = (WG + SUBGROUP_MIN_SIZE - 1u) / SUBGROUP_MIN_SIZE;

// Sized for the cross-subgroup partial sum. With the actual subgroup_size == WG
// (if hardware supports it), num_subgroups == 1 and the merge branch is skipped.
var<workgroup> sg_partial: array<f32, SG_PARTIAL_MAX>;

@compute @workgroup_size(WG)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_inv_id: u32,
  @builtin(subgroup_id) sg_id: u32,
  @builtin(num_subgroups) num_subgroups: u32,
) {
  let gi = wg.x;
  if (gi >= p.n_groups) { return; }
  let tid = lid.x;
  let n = p.group_size;
  let in_base = p.input_offset + gi * n;
  let out_base = p.output_offset + gi * n;

  var s: f32;
  for (var i: u32 = tid; i < n; i += WG) {
    let v = f32(act[in_base + i]);
    s += v * v;
  }

  // Workgroup-wide sum: subgroupAdd within subgroup, then if num_subgroups > 1
  // merge the per-subgroup totals through shmem and a second subgroupAdd in sg 0.
  let sg_sum = subgroupAdd(s);
  var total: f32;
  if (num_subgroups == 1u) {
    total = sg_sum;
  } else {
    if (sg_inv_id == 0u) { sg_partial[sg_id] = sg_sum; }
    workgroupBarrier();
    if (sg_id == 0u) {
      var combined: f32;
      if (sg_inv_id < num_subgroups) { combined = sg_partial[sg_inv_id]; }
      let final_sum = subgroupAdd(combined);
      if (sg_inv_id == 0u) { sg_partial[0] = final_sum; }
    }
    workgroupBarrier();
    total = sg_partial[0];
  }

  let inv_h = f16(inverseSqrt(total / f32(n) + p.eps));
  for (var i: u32 = tid; i < n; i += WG) {
    act[out_base + i] = act[in_base + i] * inv_h * w[p.weight_offset + i];
  }
}
