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
@group(0) @binding(1) var<storage, read_write> act: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;

const WG: u32 = 64u;
var<workgroup> partial: array<f32, WG>;

@compute @workgroup_size(WG)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let gi = wg.x;
  if (gi >= p.n_groups) { return; }
  let tid = lid.x;
  let n = p.group_size;
  let in_base = p.input_offset + gi * n;
  let out_base = p.output_offset + gi * n;

  var s: f32 = 0.0;
  for (var i: u32 = tid; i < n; i += WG) {
    let v = act[in_base + i];
    s += v * v;
  }
  partial[tid] = s;
  workgroupBarrier();
  var step: u32 = WG / 2u;
  while (step > 0u) {
    if (tid < step) { partial[tid] += partial[tid + step]; }
    workgroupBarrier();
    step = step / 2u;
  }
  let inv = inverseSqrt(partial[0] / f32(n) + p.eps);
  for (var i: u32 = tid; i < n; i += WG) {
    act[out_base + i] = act[in_base + i] * inv * w[p.weight_offset + i];
  }
}
