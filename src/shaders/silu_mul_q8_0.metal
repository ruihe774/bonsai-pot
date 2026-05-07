#include <metal_stdlib>
using namespace metal;

// Fused: y = silu(gate) * up  →  Q8_0 quantize  →  write to act_q8.
// dispatch: (m, 1, 1) — one workgroup per token.

struct Params {
    uint k;
    uint gate_offset;
    uint up_offset;
    uint d_offset;
    uint qs_offset;
};

constant uint WG = 256u;
constant uint BLOCK_SIZE = 32u;
constant uint BLOCKS_PER_ITER = WG / BLOCK_SIZE;

kernel void silu_mul_q8_0(
    constant Params& p [[buffer(0)]],
    device const half* act [[buffer(1)]],
    device uint* outbuf [[buffer(2)]],
    uint3 wg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint token_idx = wg_id.x;
    uint nb_q8 = p.k / BLOCK_SIZE;
    uint g_base = p.gate_offset + token_idx * p.k;
    uint u_base = p.up_offset + token_idx * p.k;
    uint qs_token_byte = p.qs_offset + token_idx * p.k;
    uint d_token_word = (p.d_offset >> 2) + token_idx * nb_q8;

    // SG=32, so lane_in_block == sg_lane and one block = one subgroup.
    uint lane_in_block = tid & 31u;
    uint block_within_iter = tid >> 5;

    uint n_iters = nb_q8 / BLOCKS_PER_ITER;
    for (uint it = 0u; it < n_iters; ++it) {
        uint block_global = it * BLOCKS_PER_ITER + block_within_iter;
        uint elem_idx = it * WG + tid;
        float g = float(act[g_base + elem_idx]);
        float silu = g / (1.0f + exp(-g));
        float v = silu * float(act[u_base + elem_idx]);

        float m = fabs(v);
        m = max(m, simd_shuffle_xor(m, 1u));
        m = max(m, simd_shuffle_xor(m, 2u));
        m = max(m, simd_shuffle_xor(m, 4u));
        m = max(m, simd_shuffle_xor(m, 8u));
        m = max(m, simd_shuffle_xor(m, 16u));
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
