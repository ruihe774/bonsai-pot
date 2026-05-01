enable f16;

// Copy K and V activations into the KV cache, in-pass — replaces the
// copy_buffer_to_buffer that used to break the per-token compute pass.
//
// total_elems = m_tokens * kv_dim. For tg (single-token) total = kv_dim.
// dispatch: (ceil(total / 64), 1, 1)

struct Params {
  k_cur_off: u32,    // f16 element offset in act
  v_cur_off: u32,    // f16 element offset in act
  dst_off: u32,      // f16 element offset in kv_k / kv_v
  total_elems: u32,  // m_tokens * kv_dim
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> act: array<f16>;
@group(0) @binding(2) var<storage, read_write> kv_k: array<f16>;
@group(0) @binding(3) var<storage, read_write> kv_v: array<f16>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.total_elems) { return; }
  kv_k[p.dst_off + i] = act[p.k_cur_off + i];
  kv_v[p.dst_off + i] = act[p.v_cur_off + i];
}
