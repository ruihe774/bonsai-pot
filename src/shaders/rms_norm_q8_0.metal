#include <metal_stdlib>
using namespace metal;

// Fused: y = rms_norm(x) * w  →  Q8_0 quantize  →  write to act_q8.
// dispatch: (m, 1, 1) — one workgroup per token.

struct Params {
    uint k;
    uint input_offset;
    uint weight_offset;
    uint d_offset;
    uint qs_offset;
    float eps;
};

constant uint WG = 256u;
constant uint BLOCK_SIZE = 32u;
constant uint BLOCKS_PER_ITER = WG / BLOCK_SIZE; // 8
constant uint NUM_SUBGROUPS = WG / 32u; // = 8

kernel void cs_main(
    constant Params& p [[buffer(0)]],
    device const half* act [[buffer(1)]],
    device const half* w [[buffer(2)]],
    device uint* outbuf [[buffer(3)]],
    uint3 wg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float sg_partial[NUM_SUBGROUPS];
    threadgroup float inv_h_sh;

    uint token_idx = wg_id.x;
    uint nb_q8 = p.k / BLOCK_SIZE;
    uint in_base = p.input_offset + token_idx * p.k;
    uint qs_token_byte = p.qs_offset + token_idx * p.k;
    uint d_token_word = (p.d_offset >> 2) + token_idx * nb_q8;

    // ---- Phase 1: rms_norm sum-of-squares ----
    float s = 0.0f;
    for (uint i = tid; i < p.k; i += WG) {
        float v = float(act[in_base + i]);
        s += v * v;
    }
    float sg_sum = simd_sum(s);
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
    float total_sq = sg_partial[0];
    if (tid == 0u) {
        inv_h_sh = rsqrt(total_sq / float(p.k) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_h = inv_h_sh;

    // ---- Phase 2: per-block normalize + Q8_0 quantize ----
    // SG=32, so lane_in_block == sg_lane and one block = one subgroup.
    uint lane_in_block = tid & 31u;
    uint block_within_iter = tid >> 5;

    uint n_iters = nb_q8 / BLOCKS_PER_ITER;
    for (uint it = 0u; it < n_iters; ++it) {
        uint block_global = it * BLOCKS_PER_ITER + block_within_iter;
        uint elem_idx = it * WG + tid;
        float v = float(act[in_base + elem_idx]) * inv_h * float(w[p.weight_offset + elem_idx]);

        float m = fabs(v);
        m = simd_max(m);
        float d = m / 127.0f;
        float id_inv = (d > 0.0f) ? (1.0f / d) : 0.0f;
        uint qv = uint(int(clamp(rint(v * id_inv), -127.0f, 127.0f))) & 0xFFu;

        uint tgt = lane_in_block & ~3u;
        uint packed = simd_shuffle(qv, tgt + 0u) | (simd_shuffle(qv, tgt + 1u) << 8) |
                      (simd_shuffle(qv, tgt + 2u) << 16) | (simd_shuffle(qv, tgt + 3u) << 24);
        if ((lane_in_block & 3u) == 0u) {
            uint qs_byte = qs_token_byte + block_global * 32u + lane_in_block;
            outbuf[qs_byte >> 2] = packed;
        }
        if (lane_in_block == 0u) {
            outbuf[d_token_word + block_global] = as_type<uint>(d);
        }
    }
}
