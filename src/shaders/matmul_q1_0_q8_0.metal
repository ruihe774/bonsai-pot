#include <metal_stdlib>
using namespace metal;

// Q1_0 weights × Q8_0 activations -> f16 output. 64×64 tile, 256 threads.
//
// Apple variant: Q1_0 weights are ±1, so the inner loop is a sign-flip-and-
// accumulate against f16 staged activations. No int8 dot product (Apple GPUs
// lack DP4a — see memory/apple_gpu_no_dp4a.md). Q8_0 activations are
// dequantized to f16 once during the cooperative load into shmem; the inner
// loop runs entirely on Apple's fast f32 ALU pipe.

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

constant uint WG_N = 16u;
constant uint WG = 256u;
constant uint TN = 4u;
constant uint TM = 4u;
constant uint TILE_N = 64u;
constant uint TILE_M = 64u;

kernel void cs_main(
    constant Params& p [[buffer(0)]],
    device const uint* weights [[buffer(1)]],
    device const uint* acts [[buffer(2)]],
    device half* y [[buffer(3)]],
    uint3 wg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float w_d_lds[64];
    threadgroup uint w_bits_lds[256];
    // 64 tokens × 32 half4 per token (= 128 elements) = 2048 half4 = 16 KB.
    threadgroup half4 a_sh[TILE_M * 32u];

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
        // ---- Cooperative loads: weight scales + sign bits ----
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

        // ---- Cooperative load: activations (dequant Q8_0 → f16 inline) ----
        // a_sh layout: a_sh[m_local * 32 + v4_local], where v4_local = s * 8 + i
        // covers element [m_local, b*128 + s*32 + i*4 .. i*4 + 3].
        // 2048 half4 across 256 threads → 8 half4 per thread.
        #pragma unroll
        for (uint li = 0u; li < 8u; ++li) {
            uint idx = li * WG + tid;       // 0..2047
            uint m_local = idx / 32u;        // 0..63 (token within tile)
            uint v4_local = idx % 32u;        // 0..31 (half4 index within 128 elems)
            uint sub = v4_local / 8u;         // 0..3 (Q8_0 sub-block)
            uint v4_in_sub = v4_local % 8u;   // 0..7 (half4 within sub-block)

            uint m_idx = m_base + m_local;
            uint a_block = b * 4u + sub;

            // Q8_0 d-scale (one per 32-element sub-block).
            uint d_off = p.a_d_offset + (m_idx * nb_q8 + a_block) * 4u;
            float a_d = as_type<float>(acts[d_off >> 2]);

            // Q8_0 quantized i8 quants — 4 i8 per uint.
            uint qs_off = p.a_qs_offset + m_idx * p.k + b * 128u + sub * 32u + v4_in_sub * 4u;
            uint q_packed = acts[qs_off >> 2];
            char4 q_chars = as_type<char4>(q_packed);

            float4 q_f = float4(float(q_chars[0]), float(q_chars[1]), float(q_chars[2]), float(q_chars[3])) * a_d;
            a_sh[idx] = half4(q_f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Inner: per-thread (4 tm × 4 tn) outputs ----
        float wd[TN];
        #pragma unroll
        for (uint tn = 0u; tn < TN; ++tn) {
            wd[tn] = w_d_lds[lx * TN + tn];
        }

        // Hoist all 4×TN sign-bit words into registers once per Q1_0 block so
        // the inner s-loop never re-reads from threadgroup memory.
        uint bw_all[4][TN];
        #pragma unroll
        for (uint s = 0u; s < 4u; ++s) {
            #pragma unroll
            for (uint tn = 0u; tn < TN; ++tn) {
                bw_all[s][tn] = w_bits_lds[s * 64u + lx * TN + tn];
            }
        }

        #pragma unroll
        for (uint s = 0u; s < 4u; ++s) {
            #pragma unroll
            for (uint tm = 0u; tm < TM; ++tm) {
                uint m_local = ly * TM + tm;
                // Inner reduction on packed-half2 f16 ALU pipe (2 MAC/cyc/lane).
                half4 acc4[TN];
                #pragma unroll
                for (uint tn = 0u; tn < TN; ++tn) {
                    acc4[tn] = half4(0.0h);
                }

                #pragma unroll
                for (uint i = 0u; i < 8u; ++i) {
                    half4 a4 = a_sh[m_local * 32u + s * 8u + i];
                    #pragma unroll
                    for (uint tn = 0u; tn < TN; ++tn) {
                        uint bits = (bw_all[s][tn] >> (i * 4u)) & 0xFu;
                        // Q1_0 sign convention: bit=1 → +a, bit=0 → -a.
                        half4 signs = half4(
                            ((bits & 1u) != 0u) ? 1.0h : -1.0h,
                            ((bits & 2u) != 0u) ? 1.0h : -1.0h,
                            ((bits & 4u) != 0u) ? 1.0h : -1.0h,
                            ((bits & 8u) != 0u) ? 1.0h : -1.0h);
                        acc4[tn] += a4 * signs;
                    }
                }

                #pragma unroll
                for (uint tn = 0u; tn < TN; ++tn) {
                    float sub_acc = float(acc4[tn].x) + float(acc4[tn].y)
                                  + float(acc4[tn].z) + float(acc4[tn].w);
                    acc[tm * TN + tn] += wd[tn] * sub_acc;
                }
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
