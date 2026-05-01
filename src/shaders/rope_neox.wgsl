enable f16;

// NEOX-style RoPE applied in place. Pairs (j, j+head_dim/2) rotate with
// (cos(p·θ_j), sin(p·θ_j)). The cos/sin table is precomputed on the host
// as `[max_seq, head_dim/2, 2]` f16 (cos, sin pairs).
// dispatch: (n_tokens, n_heads, 1)

struct Params {
  head_dim: u32,
  n_heads: u32,
  n_tokens: u32,
  pos_base: u32,           // first token's position
  data_offset: u32,        // f16 element offset of [n_tokens, n_heads, head_dim]
  rope_table_offset: u32,  // f16 element offset; per pos: head_dim/2 (cos,sin) pairs = head_dim halves
  _p0: u32, _p1: u32,
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
  let base = p.data_offset + (m * p.n_heads + h) * p.head_dim;
  let pos = p.pos_base + m;
  let cs_base = p.rope_table_offset + pos * p.head_dim;
  for (var j: u32 = tid; j < half_dim; j += 64u) {
    let c = rope_cs[cs_base + j * 2u];
    let s = rope_cs[cs_base + j * 2u + 1u];
    let x0 = data[base + j];
    let x1 = data[base + j + half_dim];
    data[base + j]            = x0 * c - x1 * s;
    data[base + j + half_dim] = x0 * s + x1 * c;
  }
}
