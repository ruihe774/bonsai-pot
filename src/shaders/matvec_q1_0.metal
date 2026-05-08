#include <metal_stdlib>
using namespace metal;

// Q1_0 matvec.
//
// Apple variant: weights are ±1, so the inner loop is a sign-flip-and-
// accumulate against the f16 staged activation. No int8 dot product (Apple
// GPUs lack DP4a / dotPacked4x8 — see memory/apple_gpu_no_dp4a.md), no
// Q8_0 quantize-then-dequantize round-trip in shmem.
//
// dispatch: (dispatch_x_dim, ceil(n / ROWS_PER_WG / dispatch_x_dim), 1)
// local: (8, 16, 1)

struct Params {
    uint k;
    uint n;
    uint d_offset;
    uint qs_offset;
    uint input_offset;
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

        // Stage activations as fp16 (no Q8_0 round-trip).
        uint in_v4_off = (p.input_offset + tile_start) >> 2;
        for (uint v = tid; v < tile_v4; v += WG) {
            uint base = (in_v4_off + v) << 2;
            a_sh[v] = half4(act[base + 0u], act[base + 1u], act[base + 2u], act[base + 3u]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid) {
            uint b_q1_base_global = tile_start / 128u;
            for (uint b_local = tx; b_local < nb_q1_tile; b_local += WG_X) {
                uint b = b_q1_base_global + b_local;
                float d_w = load_f16_at(weights, row_d_byte + b * 2u);
                uint qs_word_base = (row_qs_byte + b * 16u) >> 2;
                uint x_block_base = b_local * 32u; // half4 stride per Q1_0 block in tile-local a_sh
                // Run inner reduction on packed-half2 f16 ALU pipe (2 MAC/cyc/lane
                // vs f32's 1). 128 ±-terms per Q1_0 block × O(1) activations fits
                // comfortably in fp16; the d_w * sub_acc accumulate stays f32.
                half4 acc4 = half4(0.0h);
                #pragma unroll
                for (uint s = 0u; s < 4u; ++s) {
                    uint qword = weights[qs_word_base + s];
                    #pragma unroll
                    for (uint i = 0u; i < 8u; ++i) {
                        uint bits = (qword >> (i * 4u)) & 0xFu;
                        half4 a4 = a_sh[x_block_base + s * 8u + i];
                        // Q1_0 sign convention: bit=1 → +a, bit=0 → -a. Build a
                        // half4 of ±1 signs and FMA into the half4 accumulator —
                        // the compiler maps this to 2 packed-half2 vmul/vadd.
                        half4 signs = half4(
                            ((bits & 1u) != 0u) ? 1.0h : -1.0h,
                            ((bits & 2u) != 0u) ? 1.0h : -1.0h,
                            ((bits & 4u) != 0u) ? 1.0h : -1.0h,
                            ((bits & 8u) != 0u) ? 1.0h : -1.0h);
                        acc4 += a4 * signs;
                    }
                }
                float sub_acc = float(acc4.x) + float(acc4.y) + float(acc4.z) + float(acc4.w);
                acc += d_w * sub_acc;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 8-lane row-wise reduction via SIMD shuffle butterfly.
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
