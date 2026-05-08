#include <metal_stdlib>
using namespace metal;

// Pass-2 of the multi-WG top-K reduction. dispatch: (1, 1, 1)

struct Params {
    uint partials_off;
    uint num_partials;
    uint out_offset;
    uint k;
};

constant uint WG = 64u;
constant uint K_MAX = 32u;
constant half NEG_INF_H = half(-65504.0h);

kernel void cs_main(
    constant Params& p [[buffer(0)]],
    device uint* result [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup half sh_val[2048];
    threadgroup uint sh_idx[2048];

    uint base = tid * K_MAX;

    if (tid < p.num_partials) {
        uint slot_base = p.partials_off + tid * (2u * K_MAX);
        for (uint i = 0u; i < K_MAX; ++i) {
            sh_val[base + i] = half(as_type<float>(result[slot_base + i]));
            sh_idx[base + i] = result[slot_base + K_MAX + i];
        }
    } else {
        for (uint i = 0u; i < K_MAX; ++i) {
            sh_val[base + i] = NEG_INF_H;
            sh_idx[base + i] = 0u;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint s = WG / 2u;
    while (s > 0u) {
        uint n_stage0 = s * K_MAX;
        for (uint k = tid; k < n_stage0; k += WG) {
            uint pair_id = k / K_MAX;
            uint i = k % K_MAX;
            uint a_base = pair_id * K_MAX;
            uint b_base = (pair_id + s) * K_MAX;
            uint bi = K_MAX - 1u - i;
            half av = sh_val[a_base + i];
            half bv = sh_val[b_base + bi];
            if (bv > av) {
                sh_val[a_base + i] = bv;
                sh_idx[a_base + i] = sh_idx[b_base + bi];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint n_perstage = s * (K_MAX / 2u);
        uint stride = K_MAX / 2u;
        while (stride > 0u) {
            for (uint k = tid; k < n_perstage; k += WG) {
                uint pair_id = k / (K_MAX / 2u);
                uint local_k = k % (K_MAX / 2u);
                uint lo = ((local_k & ~(stride - 1u)) << 1) | (local_k & (stride - 1u));
                uint hi = lo | stride;
                uint a_base = pair_id * K_MAX;
                half vlo = sh_val[a_base + lo];
                half vhi = sh_val[a_base + hi];
                if (vhi > vlo) {
                    uint ilo = sh_idx[a_base + lo];
                    sh_val[a_base + lo] = vhi;
                    sh_val[a_base + hi] = vlo;
                    sh_idx[a_base + lo] = sh_idx[a_base + hi];
                    sh_idx[a_base + hi] = ilo;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            stride = stride / 2u;
        }

        s = s / 2u;
    }

    if (tid == 0u) {
        uint kk = min(p.k, K_MAX);
        for (uint i = 0u; i < kk; ++i) {
            result[p.out_offset + i] = as_type<uint>(float(sh_val[i]));
            result[p.out_offset + kk + i] = sh_idx[i];
        }
    }
}
