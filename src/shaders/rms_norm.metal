#include <metal_stdlib>
using namespace metal;

// RMSNorm: y = x * rsqrt(mean(x^2) + eps) * w
// dispatch: (n_groups, 1, 1)

// SUBGROUP_SIZE is fixed to 32 on modern Apple GPUs.
constant uint WG = 512u;
constant uint NUM_SUBGROUPS = WG / 32u; // = 16

struct Params {
    uint group_size;
    uint n_groups;
    uint input_offset;
    uint output_offset;
    uint weight_offset;
    float eps;
};

kernel void cs_main(
    constant Params& p [[buffer(0)]],
    device half* act [[buffer(1)]],
    device const half* w [[buffer(2)]],
    uint3 wg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float sg_partial[NUM_SUBGROUPS];

    uint gi = wg_id.x;
    if (gi >= p.n_groups)
        return;
    uint n = p.group_size;
    uint in_base = p.input_offset + gi * n;
    uint out_base = p.output_offset + gi * n;

    float s = 0.0f;
    for (uint i = tid; i < n; i += WG) {
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
    float total = sg_partial[0];

    half inv_h = half(rsqrt(total / float(n) + p.eps));
    for (uint i = tid; i < n; i += WG) {
        act[out_base + i] = act[in_base + i] * inv_h * w[p.weight_offset + i];
    }
}
