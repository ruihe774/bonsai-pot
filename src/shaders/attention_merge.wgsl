enable f16;

// Combines per-chunk (m, l, o) partials produced by attention_split.wgsl into
// the final attention output via a flash-attention-style log-sum-exp merge.
//
// dispatch: (n_head, 1, 1). One workgroup per Q head. Each workgroup makes
// two passes over the n_chunks_active partials: pass 1 finds the global max
// of the chunk-local m's; pass 2 reweights and accumulates l and o.
//
// Cheap relative to the chunk pass: ~n_chunks_active * (head_dim + 2) f32
// reads per workgroup.

struct Params {
  head_dim: u32,
  n_head: u32,
  out_offset: u32,
  n_chunks_active: u32,
  _p0: u32, _p1: u32, _p2: u32, _p3: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> act: array<f16>;
@group(0) @binding(2) var<storage, read> partials: array<f32>;

const WG: u32 = 64u;
const ELEMS_PER_THREAD: u32 = 2u;
const PARTIAL_STRIDE: u32 = 130u;  // head_dim + 2

@compute @workgroup_size(WG)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let h = wg.x;
  if (h >= p.n_head) { return; }
  let tid = lid.x;
  let hd = p.head_dim;
  let n = p.n_chunks_active;
  let head_base = h * n * PARTIAL_STRIDE;

  // Pass 1: global max of chunk-local m's.
  var m_global: f32 = -1e30;
  for (var c: u32 = 0u; c < n; c++) {
    let pb = head_base + c * PARTIAL_STRIDE;
    m_global = max(m_global, partials[pb + hd]);
  }

  // Pass 2: weighted accumulation.
  var o: array<f32, ELEMS_PER_THREAD>;
  for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) { o[i] = 0.0; }
  var l_global: f32 = 0.0;
  for (var c: u32 = 0u; c < n; c++) {
    let pb = head_base + c * PARTIAL_STRIDE;
    let m_c = partials[pb + hd];
    let l_c = partials[pb + hd + 1u];
    let weight = exp(m_c - m_global);
    l_global = l_global + l_c * weight;
    for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) {
      let d = tid + i * WG;
      if (d < hd) {
        o[i] = o[i] + partials[pb + d] * weight;
      }
    }
  }

  let inv_l = 1.0 / l_global;
  let out_base = p.out_offset + h * hd;
  for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) {
    let d = tid + i * WG;
    if (d < hd) {
      act[out_base + d] = f16(o[i] * inv_l);
    }
  }
}
