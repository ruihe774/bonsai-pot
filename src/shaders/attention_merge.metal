#include <metal_stdlib>
using namespace metal;

// Combines per-chunk (m, l, o) partials produced by attention_split into the
// final attention output via a flash-attention-style log-sum-exp merge.
// dispatch: (n_head, 1, 1).

constant uint MAX_CHUNKS [[function_constant(1)]];

struct Params {
    uint head_dim;
    uint n_head;
    uint out_offset;
    uint n_chunks_active;
};

constant uint WG = 128u;
constant uint NUM_SUBGROUPS = WG / 32u; // = 4
constant uint ELEMS_PER_THREAD = 1u;
constant uint PARTIAL_STRIDE = 130u;

kernel void cs_main(
    constant Params& p [[buffer(0)]],
    device half* act [[buffer(1)]],
    device const float* partials [[buffer(2)]],
    uint3 wg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float sg_partial[NUM_SUBGROUPS];
    threadgroup float weights_sh[MAX_CHUNKS];

    uint h = wg_id.x;
    if (h >= p.n_head)
        return;
    uint hd = p.head_dim;
    uint n = p.n_chunks_active;
    uint head_base = h * n * PARTIAL_STRIDE;

    // Phase A: parallel max
    float m_local = -1e30f;
    for (uint c = tid; c < n; c += WG) {
        m_local = max(m_local, partials[head_base + c * PARTIAL_STRIDE + hd]);
    }
    float sg_m = simd_max(m_local);
    if (sg_lane == 0u)
        sg_partial[sg_id] = sg_m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0u) {
        float c = (sg_lane < NUM_SUBGROUPS) ? sg_partial[sg_lane] : -1e30f;
        float fm = simd_max(c);
        if (sg_lane == 0u)
            sg_partial[0] = fm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m_global = sg_partial[0];

    // Phase B: parallel weight precompute + parallel l_global reduction
    float l_local = 0.0f;
    for (uint c = tid; c < n; c += WG) {
        uint pb = head_base + c * PARTIAL_STRIDE;
        float m_c = partials[pb + hd];
        float l_c = partials[pb + hd + 1u];
        float w = exp(m_c - m_global);
        weights_sh[c] = w;
        l_local += l_c * w;
    }
    float sg_s = simd_sum(l_local);
    if (sg_lane == 0u)
        sg_partial[sg_id] = sg_s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0u) {
        float c = (sg_lane < NUM_SUBGROUPS) ? sg_partial[sg_lane] : 0.0f;
        float fs = simd_sum(c);
        if (sg_lane == 0u)
            sg_partial[0] = fs;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float l_global = sg_partial[0];

    // Phase C: per-thread o accumulation
    float o[ELEMS_PER_THREAD];
    #pragma unroll
    for (uint i = 0u; i < ELEMS_PER_THREAD; ++i)
        o[i] = 0.0f;
    for (uint c = 0u; c < n; ++c) {
        uint pb = head_base + c * PARTIAL_STRIDE;
        float weight = weights_sh[c];
        #pragma unroll
        for (uint i = 0u; i < ELEMS_PER_THREAD; ++i) {
            uint d = tid + i * WG;
            o[i] += partials[pb + d] * weight;
        }
    }

    float inv_l = 1.0f / l_global;
    uint out_base = p.out_offset + h * hd;
    #pragma unroll
    for (uint i = 0u; i < ELEMS_PER_THREAD; ++i) {
        uint d = tid + i * WG;
        act[out_base + d] = half(o[i] * inv_l);
    }
}
