enable f16;

// Q1_0 row gather: dequant ONE row of the embedding matrix into f16.
// Reads the row index (token_id) from `sample[sample_offset + wg.x]`; this lets us
// chain steps on the GPU without a CPU round-trip — argmax of step N writes
// the input token for step N+1.
// dispatch: (m, 1, 1) workgroups; one workgroup per output row, 32 threads each.
// (Only nb = n_embd/128 threads do useful work; for n_embd=2560 that's 20.
// 32 keeps the workgroup small and SIMD-aligned.)

struct Params {
  k: u32,             // n_embd
  d_offset: u32,
  qs_offset: u32,
  output_offset: u32, // f16 elements
  sample_offset: u32, // base index into sample[] for the input tokens
};

var<immediate> p: Params;
@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read_write> x: array<f16>;
@group(0) @binding(2) var<storage, read> sample: array<u32>;

// Load the raw 16-bit pattern of the FP16 scale into the low half of a u32
// (high half zero). The dequant output is `±d`, and on a sign-magnitude FP16
// flipping the sign is a single XOR with 0x8000 — no FP unit involvement.
fn load_half_bits(b_offset: u32) -> u32 {
  let word = weights[b_offset >> 2u];
  return (word >> ((b_offset & 2u) * 8u)) & 0xFFFFu;
}

fn load_byte_at(b_offset: u32) -> u32 {
  let word = weights[b_offset >> 2u];
  return (word >> ((b_offset & 3u) * 8u)) & 0xFFu;
}

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wg: vec3<u32>) {
  let nb = p.k / 128u;
  let row = sample[p.sample_offset + wg.x];
  let tid = lid.x;
  if (tid >= nb) { return; }
  let b = tid;
  let d_bits = load_half_bits(p.d_offset + (row * nb + b) * 2u);
  let qs_b = p.qs_offset + (row * nb + b) * 16u;
  let out_base = p.output_offset + wg.x * p.k + b * 128u;
  for (var k: u32 = 0u; k < 16u; k++) {
    let qb = load_byte_at(qs_b + k);
    for (var bit: u32 = 0u; bit < 8u; bit++) {
      // bit==1 → +d; bit==0 → -d (XOR sign).
      let signed_bits = d_bits ^ select(0x8000u, 0u, ((qb >> bit) & 1u) != 0u);
      // unpack2x16float reinterprets the low 16 bits as an f16 (returned as f32);
      // the high 16 bits are zero so .x is exactly ±d. The f32→f16 cast is
      // bit-exact since f16→f32 widens losslessly.
      x[out_base + k * 8u + bit] = f16(unpack2x16float(signed_bits).x);
    }
  }
}
