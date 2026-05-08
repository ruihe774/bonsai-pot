#include <metal_stdlib>
using namespace metal;

// Multi-range Q1_0 matvec fused with the preceding RMS-norm + scale.
// Inner matmul uses dotPacked4x8-style accumulation against a Q8_0 staged
// activation in LDS. Used for fused QKV (3 ranges) and gate+up (2 ranges).

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

static inline uint expand_4_bits(uint bits) {
    uint spread = (bits * 0x00204081u) & 0x01010101u;
    return ~(spread * 0xFEu);
}

static inline int dot4i8packed(uint a, uint b) {
    packed_char4 vec1 = as_type<packed_char4>(a);
    packed_char4 vec2 = as_type<packed_char4>(b);
    return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2] + vec1[3] * vec2[3];
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
    threadgroup uint a_qs_sh[K_V4];
    threadgroup float a_d_sh[(K_V4 << 2) / 32];
    threadgroup float sg_partial[NUM_SUBGROUPS];

    uint K = K_V4 << 2;
    uint NB_Q8 = K >> 5;
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
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Stage 2: per-block Q8_0 quantize ----
    if (tid < NB_Q8) {
        uint block_v4_base = tid * 8u;
        half max_abs_h = half(0.0);
        #pragma unroll
        for (uint i = 0u; i < 8u; ++i) {
            half4 af = abs(x_sh[block_v4_base + i]);
            max_abs_h = max(max_abs_h, max(max(af.x, af.y), max(af.z, af.w)));
        }
        float max_abs = float(max_abs_h);
        float d = max_abs * inv_rms * (1.0f / 127.0f);
        a_d_sh[tid] = d;
        float inv_max = 127.0f / max(max_abs, 1.0e-30f);
        #pragma unroll
        for (uint i = 0u; i < 8u; ++i) {
            float4 v = float4(x_sh[block_v4_base + i]);
            float4 q = clamp(rint(v * inv_max), float4(-128.0f), float4(127.0f));
            uint q0 = uint(int(q.x)) & 0xFFu;
            uint q1 = uint(int(q.y)) & 0xFFu;
            uint q2 = uint(int(q.z)) & 0xFFu;
            uint q3 = uint(int(q.w)) & 0xFFu;
            a_qs_sh[i * NB_Q8 + tid] = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Stage 3: matvec inner ----
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
            uint a_block_base = b * 4u;
            float sub_acc = 0.0f;
            #pragma unroll
            for (uint s = 0u; s < 4u; ++s) {
                uint qword = weights[qs_word_base + s];
                float a_d = a_d_sh[a_block_base + s];
                uint block_l = a_block_base + s;
                int sumi = 0;
                #pragma unroll
                for (uint i = 0u; i < 8u; ++i) {
                    uint bits = (qword >> (i * 4u)) & 0xFu;
                    uint w_packed = expand_4_bits(bits);
                    uint a_packed = a_qs_sh[i * NB_Q8 + block_l];
                    sumi = dot4i8packed(w_packed, a_packed) + sumi;
                }
                sub_acc += a_d * float(sumi);
            }
            acc += d_w * sub_acc;
        }
    }

    acc += simd_shuffle_xor(acc, 1u);
    acc += simd_shuffle_xor(acc, 2u);
    acc += simd_shuffle_xor(acc, 4u);
    if (tx == 0u && valid) {
        uint yi = out_off + local_row;
        act[yi] = half(acc);
    }
}
