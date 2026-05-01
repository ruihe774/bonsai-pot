// Q1_0 row gather: dequant ONE row of the embedding matrix into FP32.
// dispatch: (1, 1, 1) workgroups; one workgroup, 64 threads.
// One thread handles one Q1_0 block (128 outputs). For n_embd=2560 we have
// nb=20 blocks per row, so 44 of the 64 threads idle — fine for this rare op.

struct Params {
  k: u32,             // n_embd
  d_offset: u32,      // bytes into weights buffer
  qs_offset: u32,     // bytes
  output_offset: u32, // f32 elements
  token_id: u32,
  m_token: u32,       // which output row in the M batch
  _p0: u32, _p1: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;

fn load_f16_at(b_offset: u32) -> f32 {
  let word = weights[b_offset >> 2u];
  let half = (word >> ((b_offset & 2u) * 8u)) & 0xFFFFu;
  return unpack2x16float(half).x;
}

fn load_byte_at(b_offset: u32) -> u32 {
  let word = weights[b_offset >> 2u];
  return (word >> ((b_offset & 3u) * 8u)) & 0xFFu;
}

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let nb = p.k / 128u;
  let row = p.token_id;
  let tid = lid.x;
  if (tid >= nb) { return; }
  let b = tid;
  let d = load_f16_at(p.d_offset + (row * nb + b) * 2u);
  let qs_b = p.qs_offset + (row * nb + b) * 16u;
  let out_base = p.output_offset + p.m_token * p.k + b * 128u;
  for (var k: u32 = 0u; k < 16u; k++) {
    let qb = load_byte_at(qs_b + k);
    for (var bit: u32 = 0u; bit < 8u; bit++) {
      x[out_base + k * 8u + bit] = select(-d, d, ((qb >> bit) & 1u) != 0u);
    }
  }
}
