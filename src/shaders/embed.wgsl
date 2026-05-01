enable f16;

// Q1_0 row gather: dequant ONE row of the embedding matrix into f16.
// Reads the row index (token_id) from `sample[sample_offset]`; this lets us
// chain steps on the GPU without a CPU round-trip — argmax of step N writes
// the input token for step N+1.
// dispatch: (1, 1, 1) workgroups; one workgroup, 64 threads.

struct Params {
  k: u32,             // n_embd
  d_offset: u32,
  qs_offset: u32,
  output_offset: u32, // f16 elements
  sample_offset: u32, // index into sample[] for the input token
  m_token: u32,       // which output row in the M batch
  _p0: u32, _p1: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read_write> x: array<f16>;
@group(0) @binding(3) var<storage, read> sample: array<u32>;

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
  let row = sample[p.sample_offset];
  let tid = lid.x;
  if (tid >= nb) { return; }
  let b = tid;
  let d = load_f16_at(p.d_offset + (row * nb + b) * 2u);
  let qs_b = p.qs_offset + (row * nb + b) * 16u;
  let out_base = p.output_offset + p.m_token * p.k + b * 128u;
  for (var k: u32 = 0u; k < 16u; k++) {
    let qb = load_byte_at(qs_b + k);
    for (var bit: u32 = 0u; bit < 8u; bit++) {
      x[out_base + k * 8u + bit] = f16(select(-d, d, ((qb >> bit) & 1u) != 0u));
    }
  }
}
