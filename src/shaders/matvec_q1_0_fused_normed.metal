#include <metal_stdlib>
using namespace metal;

// Multi-range Q1_0 matvec fused with the preceding RMS-norm + scale.
//
// Apple variant: weights are ±1, so the inner loop is a sign-flip-and-
// accumulate against the f16 staged activation. No int8 dot product (Apple
// GPUs lack DP4a / dotPacked4x8 — see memory/apple_gpu_no_dp4a.md), no
// Q8_0 quantize-then-dequantize round-trip in shmem. This trades 4 IMUL32
// (4 cyc each on Apple) + 3 ADD per 4-element dot for 4 fselect-with-neg +
// 3 FADD on the fast f32 ALU pipe.

constant uint K_V4 [[function_constant(2)]];

struct Params {
    uint k;
    uint n_total;
    uint input_offset;
    uint dispatch_x_dim;
    uint w_norm_off;
    float eps;
    uint d_offset_0;
    uint qs_offset_0;
    uint n_0;
    uint output_offset_0;
    uint d_offset_1;
    uint qs_offset_1;
    uint n_1;
    uint output_offset_1;
    uint d_offset_2;
    uint qs_offset_2;
    uint n_2;
    uint output_offset_2;
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
constant uint NUM_SUBGROUPS = WG / 32u; // = 4

kernel void cs_main(
    constant Params& p [[buffer(0)]],
    device const uint* weights [[buffer(1)]],
    device half* act [[buffer(2)]],
    device const half* w_norms [[buffer(3)]],
    uint3 wg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup half4 x_sh[K_V4];
    threadgroup float sg_partial[NUM_SUBGROUPS];

    uint K = K_V4 << 2;
    uint wg_idx = wg_id.y * p.dispatch_x_dim + wg_id.x;
    uint ty = tid / WG_X;
    uint tx = tid % WG_X;
    uint global_row = wg_idx * ROWS_PER_WG + ty;
    bool valid = global_row < p.n_total;
    uint nb_q1 = K / 128u;

    // ---- Stage 1: load x and w_norm; stage `x*w_norm` into x_sh; accumulate ssq.
    uint in_v4_off = p.input_offset >> 2;
    uint w_v4_off = p.w_norm_off >> 2;
    float ssq = 0.0f;
    for (uint v = tid; v < K_V4; v += WG) {
        uint base = (in_v4_off + v) << 2;
        half4 xv4 = half4(act[base + 0u], act[base + 1u], act[base + 2u], act[base + 3u]);
        float4 xv4f = float4(xv4);
        ssq += xv4f.x * xv4f.x + xv4f.y * xv4f.y + xv4f.z * xv4f.z + xv4f.w * xv4f.w;
        uint w_base = (w_v4_off + v) << 2;
        half4 wv4 = half4(w_norms[w_base + 0u], w_norms[w_base + 1u], w_norms[w_base + 2u], w_norms[w_base + 3u]);
        x_sh[v] = xv4 * wv4;
    }

    // ---- Stage 1.5: workgroup-wide ssq sum ----
    float sg_sum = simd_sum(ssq);
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
    float inv_rms = rsqrt(total / float(K) + p.eps);

    // ---- Stage 2: matvec inner. ±-accumulate against f16 x_sh; multiply
    //              by d_w per Q1_0 block; fold inv_rms once at row write.
    uint d_off = 0u;
    uint qs_off = 0u;
    uint out_off = 0u;
    uint local_row = 0u;
    if (valid) {
        if (global_row < p.n_0) {
            d_off = p.d_offset_0;
            qs_off = p.qs_offset_0;
            out_off = p.output_offset_0;
            local_row = global_row;
        } else if (global_row < p.n_0 + p.n_1) {
            d_off = p.d_offset_1;
            qs_off = p.qs_offset_1;
            out_off = p.output_offset_1;
            local_row = global_row - p.n_0;
        } else {
            d_off = p.d_offset_2;
            qs_off = p.qs_offset_2;
            out_off = p.output_offset_2;
            local_row = global_row - p.n_0 - p.n_1;
        }
    }

    uint row_d_byte = d_off + local_row * nb_q1 * 2u;
    uint row_qs_byte = qs_off + local_row * nb_q1 * 16u;

    float acc = 0.0f;
    if (valid) {
        for (uint b = tx; b < nb_q1; b += WG_X) {
            float d_w = load_f16_at(weights, row_d_byte + b * 2u);
            uint qs_word_base = (row_qs_byte + b * 16u) >> 2;
            uint x_block_base = b * 32u; // half4 stride per Q1_0 block (128 elems / 4)
            // Inner reduction on packed-half2 f16 ALU pipe (2 MAC/cyc/lane).
            half4 acc4 = half4(0.0h);
            #pragma unroll
            for (uint s = 0u; s < 4u; ++s) {
                uint qword = weights[qs_word_base + s];
                #pragma unroll
                for (uint i = 0u; i < 8u; ++i) {
                    uint bits = (qword >> (i * 4u)) & 0xFu;
                    half4 a4 = x_sh[x_block_base + s * 8u + i];
                    // Q1_0 sign convention: bit=1 → +a, bit=0 → -a.
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

    // 8-lane row reduction via SIMD shuffle butterfly (within one simdgroup).
    acc += simd_shuffle_xor(acc, 1u);
    acc += simd_shuffle_xor(acc, 2u);
    acc += simd_shuffle_xor(acc, 4u);
    if (tx == 0u && valid) {
        acc *= inv_rms;
        uint yi = out_off + local_row;
        act[yi] = half(acc);
    }
}
