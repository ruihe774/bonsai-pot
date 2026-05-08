#include <metal_stdlib>
using namespace metal;

// Q1_0 row gather: dequant ONE row of the embedding matrix into f16.
// dispatch: (m, 1, 1) — one workgroup per output row, 32 threads each.

struct Params {
    uint k;
    uint d_offset;
    uint qs_offset;
    uint output_offset;
    uint sample_offset;
};

static inline uint load_half_bits(device const uint* weights, uint b_offset) {
    uint word = weights[b_offset >> 2];
    return (word >> ((b_offset & 2u) * 8u)) & 0xFFFFu;
}

static inline uint load_byte_at(device const uint* weights, uint b_offset) {
    uint word = weights[b_offset >> 2];
    return (word >> ((b_offset & 3u) * 8u)) & 0xFFu;
}

kernel void cs_main(
    constant Params& p [[buffer(0)]],
    device const uint* weights [[buffer(1)]],
    device half* x [[buffer(2)]],
    device const uint* sample_in [[buffer(3)]],
    uint3 wg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint nb = p.k / 128u;
    uint row = sample_in[p.sample_offset + wg_id.x];
    if (tid >= nb)
        return;
    uint b = tid;
    uint d_bits = load_half_bits(weights, p.d_offset + (row * nb + b) * 2u);
    uint qs_b = p.qs_offset + (row * nb + b) * 16u;
    uint out_base = p.output_offset + wg_id.x * p.k + b * 128u;
    for (uint k = 0u; k < 16u; ++k) {
        uint qb = load_byte_at(weights, qs_b + k);
        for (uint bit = 0u; bit < 8u; ++bit) {
            uint signed_bits = d_bits ^ (((qb >> bit) & 1u) != 0u ? 0u : 0x8000u);
            half2 hp = as_type<half2>(signed_bits);
            x[out_base + k * 8u + bit] = hp.x;
        }
    }
}
