// Single-workgroup argmax over `n` floats. Threads cooperate via shmem.
// dispatch: (1, 1, 1)

struct Params {
  n: u32,
  in_offset: u32,
  out_offset: u32,
  _p0: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<u32>;

const WG: u32 = 256u;
var<workgroup> sh_val: array<f32, WG>;
var<workgroup> sh_idx: array<u32, WG>;

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  var best_v: f32 = -1e30;
  var best_i: u32 = 0u;
  for (var i: u32 = tid; i < p.n; i += WG) {
    let v = logits[p.in_offset + i];
    if (v > best_v) { best_v = v; best_i = i; }
  }
  sh_val[tid] = best_v;
  sh_idx[tid] = best_i;
  workgroupBarrier();
  var s: u32 = WG / 2u;
  while (s > 0u) {
    if (tid < s) {
      if (sh_val[tid + s] > sh_val[tid]) {
        sh_val[tid] = sh_val[tid + s];
        sh_idx[tid] = sh_idx[tid + s];
      }
    }
    workgroupBarrier();
    s = s / 2u;
  }
  if (tid == 0u) { result[p.out_offset] = sh_idx[0]; }
}
