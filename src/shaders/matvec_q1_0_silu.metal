#include <metal_stdlib>
using namespace metal;

// Q1_0 matvec with SiLU(gate)*up activation fold.
//
// Apple variant: weights are ±1, so the inner loop is a sign-flip-and-
// accumulate against the f16 staged activation. No int8 dot product (Apple
// GPUs lack DP4a / dotPacked4x8 — see memory/apple_gpu_no_dp4a.md), no
// Q8_0 quantize-then-dequantize round-trip in shmem.

struct Params {
    uint k;
    uint n;
    uint d_offset;
    uint qs_offset;
    uint gate_offset;
    uint up_offset;
    uint output_offset;
    uint accumulate;
    uint dispatch_x_dim;
};

static inline float load_f16_at(device const uint* weights, uint b_offset) {
    uint word = weights[b_offset >> 2];
    uint half_bits = (word >> ((b_offset & 2u) * 8u)) & 0xFFFFu;
    return float(as_type<half2>(half_bits).x);
}

constant uint WG_X = 8u;
constant uint WG_Y = 16u;
constant uint WG = WG_X * WG_Y;
constant uint ROWS_PER_WG = WG_Y;
constant uint TILE_K = 4096u;
constant uint TILE_K_V4 = TILE_K >> 2;

kernel void cs_main(
    constant Params& p [[buffer(0)]],
    device const uint* weights [[buffer(1)]],
    device half* act [[buffer(2)]],
    uint3 wg_id [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]])
{
    threadgroup half4 a_sh[TILE_K_V4];

    uint wg_idx = wg_id.y * p.dispatch_x_dim + wg_id.x;
    uint row = wg_idx * ROWS_PER_WG + lid.y;
    uint tx = lid.x;
    uint ty = lid.y;
    uint tid = ty * WG_X + tx;
    bool valid = row < p.n;
    uint nb_q1 = p.k / 128u;

    uint row_d_byte = p.d_offset + row * nb_q1 * 2u;
    uint row_qs_byte = p.qs_offset + row * nb_q1 * 16u;

    float acc = 0.0f;

    for (uint tile_start = 0u; tile_start < p.k; tile_start += TILE_K) {
        uint tile_size = min(TILE_K, p.k - tile_start);
        uint tile_v4 = tile_size >> 2;
        uint nb_q1_tile = tile_size / 128u;

        // Stage `silu(gate) * up` as fp16 directly (no Q8_0 round-trip).
        uint g_v4_off = (p.gate_offset + tile_start) >> 2;
        uint u_v4_off = (p.up_offset + tile_start) >> 2;
        for (uint v = tid; v < tile_v4; v += WG) {
            uint gb = (g_v4_off + v) << 2;
            uint ub = (u_v4_off + v) << 2;
            float4 g = float4(float(act[gb]), float(act[gb + 1u]), float(act[gb + 2u]), float(act[gb + 3u]));
            half4 u = half4(act[ub], act[ub + 1u], act[ub + 2u], act[ub + 3u]);
            float4 sf = g / (float4(1.0f) + exp(-g));
            a_sh[v] = half4(sf) * u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid) {
            uint b_q1_base_global = tile_start / 128u;
            for (uint b_local = tx; b_local < nb_q1_tile; b_local += WG_X) {
                uint b = b_q1_base_global + b_local;
                float d_w = load_f16_at(weights, row_d_byte + b * 2u);
                uint qs_word_base = (row_qs_byte + b * 16u) >> 2;
                uint x_block_base = b_local * 32u; // half4 stride per Q1_0 block in tile-local a_sh
                float sub_acc = 0.0f;
                #pragma unroll
                for (uint s = 0u; s < 4u; ++s) {
                    uint qword = weights[qs_word_base + s];
                    #pragma unroll
                    for (uint i = 0u; i < 8u; ++i) {
                        uint bits = (qword >> (i * 4u)) & 0xFu;
                        float4 a4 = float4(a_sh[x_block_base + s * 8u + i]);
                        // Q1_0 sign convention: bit=1 → weight=+1, bit=0 → weight=-1.
                        sub_acc += (bits & 1u) != 0u ? a4.x : -a4.x;
                        sub_acc += (bits & 2u) != 0u ? a4.y : -a4.y;
                        sub_acc += (bits & 4u) != 0u ? a4.z : -a4.z;
                        sub_acc += (bits & 8u) != 0u ? a4.w : -a4.w;
                    }
                }
                acc += d_w * sub_acc;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    acc += simd_shuffle_xor(acc, 1u);
    acc += simd_shuffle_xor(acc, 2u);
    acc += simd_shuffle_xor(acc, 4u);
    if (tx == 0u && valid) {
        uint yi = p.output_offset + row;
        if (p.accumulate != 0u) {
            act[yi] = half(float(act[yi]) + acc);
        } else {
            act[yi] = half(acc);
        }
    }
}
