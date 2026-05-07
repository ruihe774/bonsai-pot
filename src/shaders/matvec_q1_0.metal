#include <metal_stdlib>
using namespace metal;

// Q1_0 matvec with an in-LDS Q8_0 activation quantize.
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

static inline uint expand_4_bits(uint bits) {
    uint spread = (bits * 0x00204081u) & 0x01010101u;
    return ~(spread * 0xFEu);
}

static inline int dot4i8packed(uint a, uint b) {
    return dot(as_type<char4>(a), as_type<char4>(b));
}

constant uint WG_X = 8u;
constant uint WG_Y = 16u;
constant uint WG = WG_X * WG_Y;
constant uint ROWS_PER_WG = WG_Y;
constant uint TILE_K = 4096u;
constant uint TILE_NB_Q8 = TILE_K / 32u;
constant uint TILE_QS_LEN = TILE_K >> 2;

kernel void matvec_q1_0(
    constant Params& p [[buffer(0)]],
    device const uint* weights [[buffer(1)]],
    device half* act [[buffer(2)]],
    uint3 wg_id [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]])
{
    threadgroup uint a_qs_sh[TILE_QS_LEN];
    threadgroup float a_d_sh[TILE_NB_Q8];

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
        uint nb_q8_tile = tile_size / 32u;
        uint nb_q1_tile = tile_size / 128u;

        if (tid < nb_q8_tile) {
            uint elem_base = tile_start + tid * 32u;
            uint in_v4_off = (p.input_offset + elem_base) >> 2;

            half4 v[8];
            half max_abs_h = half(0.0);
            #pragma unroll
            for (uint i = 0u; i < 8u; ++i) {
                uint b = (in_v4_off + i) << 2;
                v[i] = half4(act[b], act[b + 1u], act[b + 2u], act[b + 3u]);
                half4 af = abs(v[i]);
                max_abs_h = max(max_abs_h, max(max(af.x, af.y), max(af.z, af.w)));
            }

            float max_abs = float(max_abs_h);
            float d = max_abs * (1.0f / 127.0f);
            a_d_sh[tid] = d;
            float inv_max = 127.0f / max(max_abs, 1.0e-30f);

            #pragma unroll
            for (uint i = 0u; i < 8u; ++i) {
                float4 q = clamp(rint(float4(v[i]) * inv_max), float4(-128.0f), float4(127.0f));
                a_qs_sh[i * TILE_NB_Q8 + tid] = (uint(int(q.x)) & 0xFFu) | ((uint(int(q.y)) & 0xFFu) << 8) |
                                                ((uint(int(q.z)) & 0xFFu) << 16) | ((uint(int(q.w)) & 0xFFu) << 24);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid) {
            uint b_q1_base_global = tile_start / 128u;
            for (uint b_local = tx; b_local < nb_q1_tile; b_local += WG_X) {
                uint b = b_q1_base_global + b_local;
                float d_w = load_f16_at(weights, row_d_byte + b * 2u);
                uint qs_word_base = (row_qs_byte + b * 16u) >> 2;
                uint a_block_local_base = b_local * 4u;
                float sub_acc = 0.0f;
                #pragma unroll
                for (uint s = 0u; s < 4u; ++s) {
                    uint qword = weights[qs_word_base + s];
                    float a_d = a_d_sh[a_block_local_base + s];
                    uint block_l = a_block_local_base + s;
                    int sumi = 0;
                    #pragma unroll
                    for (uint i = 0u; i < 8u; ++i) {
                        uint bits = (qword >> (i * 4u)) & 0xFu;
                        uint w_packed = expand_4_bits(bits);
                        uint a_packed = a_qs_sh[i * TILE_NB_Q8 + block_l];
                        sumi = dot4i8packed(w_packed, a_packed) + sumi;
                    }
                    sub_acc += a_d * float(sumi);
                }
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
