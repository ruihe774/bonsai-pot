#include <metal_stdlib>
using namespace metal;

// Q1_0 weights × Q8_0 activations -> f16 output. 64×64 tile, 256 threads.

struct Params {
    uint k;
    uint n;
    uint m;
    uint w_d_offset;
    uint w_qs_offset;
    uint a_d_offset;
    uint a_qs_offset;
    uint out_offset;
    uint accumulate;
};

static inline float load_w_f16(device const uint* weights, uint b_offset) {
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

constant uint WG_N = 16u;
constant uint WG_M = 16u;
constant uint WG = 256u;
constant uint TN = 4u;
constant uint TM = 4u;
constant uint TILE_N = 64u;
constant uint TILE_M = 64u;

kernel void matmul_q1_0_q8_0(
    constant Params& p [[buffer(0)]],
    device const uint* weights [[buffer(1)]],
    device const uint* acts [[buffer(2)]],
    device half* y [[buffer(3)]],
    uint3 wg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float w_d_lds[64];
    threadgroup uint w_bits_lds[256];
    threadgroup float a_d_lds[256];
    threadgroup uint a_qs_lds[2048];

    uint n_base = wg_id.x * TILE_N;
    uint m_base = wg_id.y * TILE_M;
    uint nb_q1 = p.k / 128u;
    uint nb_q8 = p.k / 32u;

    uint lx = tid % WG_N;
    uint ly = tid / WG_N;

    float acc[16];
    #pragma unroll
    for (uint i = 0u; i < 16u; ++i)
        acc[i] = 0.0f;

    for (uint b = 0u; b < nb_q1; ++b) {
        // ---- Cooperative loads ----
        if (tid < 64u) {
            uint n_idx = n_base + tid;
            w_d_lds[tid] = load_w_f16(weights, p.w_d_offset + (n_idx * nb_q1 + b) * 2u);
        }
        {
            uint n_local = tid / 4u;
            uint s = tid % 4u;
            uint n_idx = n_base + n_local;
            uint off = p.w_qs_offset + n_idx * (nb_q1 * 16u) + b * 16u + s * 4u;
            w_bits_lds[s * 64u + n_local] = weights[off >> 2];
        }
        {
            uint s = tid / 64u;
            uint m_local = tid % 64u;
            uint m_idx = m_base + m_local;
            uint a_block = b * 4u + s;
            uint off = p.a_d_offset + (m_idx * nb_q8 + a_block) * 4u;
            a_d_lds[s * 64u + m_local] = as_type<float>(acts[off >> 2]);
        }
        #pragma unroll
        for (uint li = 0u; li < 8u; ++li) {
            uint idx = li * WG + tid;
            uint m_local = idx / 32u;
            uint su = idx % 32u;
            uint m_idx = m_base + m_local;
            uint off = p.a_qs_offset + m_idx * p.k + b * 128u + su * 4u;
            a_qs_lds[m_local * 32u + su] = acts[off >> 2];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float wd[TN];
        #pragma unroll
        for (uint tn = 0u; tn < TN; ++tn) { wd[tn] = w_d_lds[lx * TN + tn]; }

        #pragma unroll
        for (uint s = 0u; s < 4u; ++s) {
            uint w[TN][8];
            #pragma unroll
            for (uint tn = 0u; tn < TN; ++tn) {
                uint bw = w_bits_lds[s * 64u + lx * TN + tn];
                #pragma unroll
                for (uint i = 0u; i < 8u; ++i) { w[tn][i] = expand_4_bits((bw >> (i * 4u)) & 0xFu); }
            }

            #pragma unroll
            for (uint tm = 0u; tm < TM; ++tm) {
                uint m_local = ly * TM + tm;
                float a_d = a_d_lds[s * 64u + m_local];
                uint a_base = m_local * 32u + s * 8u;
                uint a[8];
                #pragma unroll
                for (uint i = 0u; i < 8u; ++i) { a[i] = a_qs_lds[a_base + i]; }

                int sumi[TN];
                #pragma unroll
                for (uint tn = 0u; tn < TN; ++tn)
                    sumi[tn] = 0;
                #pragma unroll
                for (uint i = 0u; i < 8u; ++i) {
                    #pragma unroll
                    for (uint tn = 0u; tn < TN; ++tn) {
                        sumi[tn] = dot4i8packed(w[tn][i], a[i]) + sumi[tn];
                    }
                }

                #pragma unroll
                for (uint tn = 0u; tn < TN; ++tn) { acc[tm * TN + tn] += a_d * wd[tn] * float(sumi[tn]); }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (uint tm = 0u; tm < TM; ++tm) {
        uint m_idx = m_base + ly * TM + tm;
        if (m_idx >= p.m)
            continue;
        #pragma unroll
        for (uint tn = 0u; tn < TN; ++tn) {
            uint n_idx = n_base + lx * TN + tn;
            if (n_idx >= p.n)
                continue;
            uint yi = p.out_offset + m_idx * p.n + n_idx;
            float val = acc[tm * TN + tn];
            if (p.accumulate != 0u) {
                y[yi] = half(float(y[yi]) + val);
            } else {
                y[yi] = half(val);
            }
        }
    }
}
