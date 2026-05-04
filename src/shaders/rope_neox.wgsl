enable f16;

// NEOX-style RoPE applied in place. Pairs (j, j+head_dim/2) rotate with
// (cos(p·θ_j), sin(p·θ_j)). The cos/sin table is precomputed on the host
// as `[max_seq, head_dim/2, 2]` f16 (cos, sin pairs).
// dispatch: (n_tokens, n_heads, 1)
//
// When n_heads_0 < n_heads (fused Q/K dispatch): heads 0..n_heads_0 use
// data_offset with stride n_heads_0; the rest use data_offset_1 with
// stride (n_heads - n_heads_0). Otherwise all heads use data_offset.

struct Params {
  head_dim: u32,
  n_heads: u32,
  n_tokens: u32,
  pos_base: u32,           // first token's position
  data_offset: u32,        // f16 element offset for heads 0..n_heads_0
  data_offset_1: u32,      // f16 element offset for heads n_heads_0..
  n_heads_0: u32,          // split point (== n_heads when not fused)
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> rope_cs: array<f16>;
@group(0) @binding(2) var<storage, read_write> data: array<f16>;

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let m = wg.x;
  let h = wg.y;
  if (m >= p.n_tokens || h >= p.n_heads) { return; }
  let half_dim = p.head_dim / 2u;
  let tid = lid.x;
  // Fused Q/K dispatch: heads 0..n_heads_0 come from data_offset (Q),
  // heads n_heads_0.. come from data_offset_1 (K), each with its own stride.
  var off: u32;
  var h_eff: u32;
  var n_h: u32;
  if (h < p.n_heads_0) {
    off = p.data_offset;
    h_eff = h;
    n_h = p.n_heads_0;
  } else {
    off = p.data_offset_1;
    h_eff = h - p.n_heads_0;
    n_h = p.n_heads - p.n_heads_0;
  }
  let base = off + (m * n_h + h_eff) * p.head_dim;
  let pos = p.pos_base + m;
  let cs_base = pos * p.head_dim;
  for (var j: u32 = tid; j < half_dim; j += 64u) {
    let c = rope_cs[cs_base + j * 2u];
    let s = rope_cs[cs_base + j * 2u + 1u];
    let x0 = data[base + j];
    let x1 = data[base + j + half_dim];
    data[base + j]            = x0 * c - x1 * s;
    data[base + j + half_dim] = x0 * s + x1 * c;
  }
}
