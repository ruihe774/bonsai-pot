enable f16;

// y[i] = silu(gate[i]) * up[i] for i in 0..(n*m)
// Single binding to activations; gate/up/out regions live at different offsets.
// dispatch: (ceil(n*m / 64), 1, 1)

struct Params {
  n: u32,
  m: u32,
  gate_offset: u32,
  up_offset: u32,
  out_offset: u32,
  dispatch_x_count: u32,    // = dispatch_x_dim * 64 (workgroup_size)
};

var<immediate> p: Params;
@group(0) @binding(0) var<storage, read_write> act: array<f16>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * p.dispatch_x_count + gid.x;
  let total = p.n * p.m;
  if (i >= total) { return; }
  let g = f32(act[p.gate_offset + i]);
  let silu = g / (1.0 + exp(-g));
  act[p.out_offset + i] = f16(silu) * act[p.up_offset + i];
}
