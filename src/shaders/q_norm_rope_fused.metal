#include <metal_stdlib>
using namespace metal;

// Fused per-layer Q-side pre-attention pipeline:
//   rms_norm(Q head) -> *w_q_norm -> NEOX-RoPE -> write back to act.q.
// Workgroup = head_dim (128). One workgroup per (head, token).

struct Params {
    uint q_off;
    uint w_q_norm_off;
    uint rope_offset;
    uint pos_base;
    uint q_dim;
    float eps;
};

constant uint HEAD_DIM = 128u;
constant uint HALF_DIM = 64u;
constant uint WG = 128u;
constant uint NUM_SUBGROUPS = WG / 32u; // = 4

kernel void cs_main(
    constant Params& p [[buffer(0)]],
    device half* act [[buffer(1)]],
    device const half* w_norms [[buffer(2)]],
    device const half* rope_cs [[buffer(3)]],
    uint3 wg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup half q_sh[128];
    threadgroup float sg_partial[NUM_SUBGROUPS];

    uint head = wg_id.x;
    uint tok = wg_id.y;

    uint q_token_off = tok * p.q_dim + head * HEAD_DIM;
    uint q_idx = p.q_off + q_token_off + tid;
    float q_raw = float(act[q_idx]);

    float sg_sum = simd_sum(q_raw * q_raw);
    if (sg_lane == 0u)
        sg_partial[sg_id] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0u) {
        float c = (sg_lane < NUM_SUBGROUPS) ? sg_partial[sg_lane] : 0.0f;
        float f = simd_sum(c);
        if (sg_lane == 0u)
            sg_partial[0] = f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = sg_partial[0];
    float inv_h = rsqrt(total / float(HEAD_DIM) + p.eps);

    q_sh[tid] = half(q_raw * inv_h) * w_norms[p.w_q_norm_off + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint pos_abs = p.pos_base + tok;
    uint cs_base = p.rope_offset + pos_abs * HEAD_DIM;
    half q_post;
    if (tid < HALF_DIM) {
        half c = rope_cs[cs_base + tid * 2u];
        half s = rope_cs[cs_base + tid * 2u + 1u];
        q_post = q_sh[tid] * c - q_sh[tid + HALF_DIM] * s;
    } else {
        uint j = tid - HALF_DIM;
        half c = rope_cs[cs_base + j * 2u];
        half s = rope_cs[cs_base + j * 2u + 1u];
        q_post = q_sh[j] * s + q_sh[tid] * c;
    }

    act[q_idx] = q_post;
}
