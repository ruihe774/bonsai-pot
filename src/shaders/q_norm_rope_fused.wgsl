enable f16;

// Fused per-layer Q-side pre-attention pipeline:
//   rms_norm(Q head) -> *w_q_norm -> NEOX-RoPE -> write back to act.q (f16, in place).
// Replaces a 2-dispatch sequence (rms_norm Q, rope Q) with 1.
//
// Workgroup = head_dim (128). One workgroup per (head, token) processes one
// Q head. Dispatch (n_head, m_tokens, 1).

struct Params {
  q_off: u32,                // f16 element offset of Q in act (q region base)
  w_q_norm_off: u32,         // f16 element offset of w_q_norm[0..head_dim] in w_norms
  rope_offset: u32,          // f16 element base offset into rope_cs (typically 0)
  pos_base: u32,             // first absolute RoPE position
  q_dim: u32,                // = n_head * head_dim
  eps: f32,
};

var<immediate> p: Params;
@group(0) @binding(0) var<storage, read_write> act: array<f16>;
@group(0) @binding(1) var<storage, read>       w_norms: array<f16>;
@group(0) @binding(2) var<storage, read>       rope_cs: array<f16>;

const HEAD_DIM: u32 = 128u;
const HALF_DIM: u32 = 64u;
const SUBGROUP_MIN_SIZE: u32 = {{SUBGROUP_MIN_SIZE}}u;
const WG: u32 = 128u;
const SG_PARTIAL_MAX: u32 = (WG + SUBGROUP_MIN_SIZE - 1u) / SUBGROUP_MIN_SIZE;

// q_sh holds normed-and-weighted Q so RoPE can fetch each thread's pair partner
// across the head-half boundary (partners always cross a subgroup boundary at
// WG=128). After RoPE the rotated value lives in a register; q_sh is dead.
var<workgroup> q_sh: array<f32, 128>;
// Cross-subgroup merge slot for the RMS reduction (used only when num_subgroups > 1).
var<workgroup> sg_partial: array<f32, SG_PARTIAL_MAX>;

@compute @workgroup_size(WG)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_inv_id: u32,
  @builtin(subgroup_id) sg_id: u32,
  @builtin(num_subgroups) num_subgroups: u32,
) {
  let head = wg.x;
  let tok  = wg.y;
  let tid  = lid.x;

  let q_token_off = tok * p.q_dim + head * HEAD_DIM;
  let q_idx = p.q_off + q_token_off + tid;
  let q_raw = f32(act[q_idx]);

  // ---- RMS-norm reduction over the Q head (sum of squares) ----------------
  let sg_sum = subgroupAdd(q_raw * q_raw);
  var total: f32;
  if (num_subgroups == 1u) {
    total = sg_sum;
  } else {
    if (sg_inv_id == 0u) { sg_partial[sg_id] = sg_sum; }
    workgroupBarrier();
    if (sg_id == 0u) {
      var c: f32 = 0.0;
      if (sg_inv_id < num_subgroups) { c = sg_partial[sg_inv_id]; }
      let f = subgroupAdd(c);
      if (sg_inv_id == 0u) { sg_partial[0] = f; }
    }
    workgroupBarrier();
    total = sg_partial[0];
  }
  let inv_h = inverseSqrt(total / f32(HEAD_DIM) + p.eps);

  // ---- norm * weight, stash to shmem so RoPE can swap pairs ---------------
  q_sh[tid] = q_raw * inv_h * f32(w_norms[p.w_q_norm_off + tid]);
  workgroupBarrier();

  // ---- NEOX RoPE: each lane computes its own output ------------------------
  // Lower-half (tid in [0, 64)) emits cosθ·x0 - sinθ·x1; upper-half emits
  // sinθ·x0 + cosθ·x1 with the same (c, s) for j = tid - 64. The write below
  // aliases the same act slot we read at the top of this shader, but every
  // lane writes only its own slot and the read happened before the barrier.
  let pos_abs = p.pos_base + tok;
  let cs_base = p.rope_offset + pos_abs * HEAD_DIM;
  var q_post: f32;
  if (tid < HALF_DIM) {
    let c = f32(rope_cs[cs_base + tid * 2u]);
    let s = f32(rope_cs[cs_base + tid * 2u + 1u]);
    q_post = q_sh[tid] * c - q_sh[tid + HALF_DIM] * s;
  } else {
    let j = tid - HALF_DIM;
    let c = f32(rope_cs[cs_base + j * 2u]);
    let s = f32(rope_cs[cs_base + j * 2u + 1u]);
    q_post = q_sh[j] * s + q_sh[tid] * c;
  }

  act[q_idx] = f16(q_post);
}
