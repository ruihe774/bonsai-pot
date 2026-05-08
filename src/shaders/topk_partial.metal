#include <metal_stdlib>
using namespace metal;

// Multi-WG pass-1 of top-K reduction.
// dispatch: (NUM_PARTIAL_WG, 1, 1)

struct Params {
    uint n;
    uint in_offset;
    uint partials_off;
    uint n_per_wg;
};

constant uint WG = 128u;
constant uint K_MAX = 32u;

constant half NEG_INF_H = half(-65504.0h);

static void sift_down(threadgroup half* sh_val, threadgroup uint* sh_idx, uint base) {
    uint i = 0u;
    while (true) {
        uint l = 2u * i + 1u;
        uint r = l + 1u;
        uint smallest = i;
        if (l < K_MAX && sh_val[base + l] < sh_val[base + smallest])
            smallest = l;
        if (r < K_MAX && sh_val[base + r] < sh_val[base + smallest])
            smallest = r;
        if (smallest == i)
            break;
        half tv = sh_val[base + i];
        sh_val[base + i] = sh_val[base + smallest];
        sh_val[base + smallest] = tv;
        uint ti = sh_idx[base + i];
        sh_idx[base + i] = sh_idx[base + smallest];
        sh_idx[base + smallest] = ti;
        i = smallest;
    }
}

kernel void cs_main(
    constant Params& p [[buffer(0)]],
    device const uint* logits [[buffer(1)]],
    device uint* result [[buffer(2)]],
    uint3 wg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup half sh_val[4096];
    threadgroup uint sh_idx[4096];

    uint base = tid * K_MAX;

    uint i_lo = wg_id.x * p.n_per_wg;
    uint i_hi = min(i_lo + p.n_per_wg, p.n);
    uint in_word_off = p.in_offset >> 1;
    uint w_lo = i_lo >> 1;
    uint w_hi = (i_hi + 1u) >> 1;

    // Phase 1: per-thread top-K min-heap.
    for (uint i = 0u; i < K_MAX; ++i) {
        sh_val[base + i] = NEG_INF_H;
        sh_idx[base + i] = 0u;
    }
    for (uint w = w_lo + tid; w < w_hi; w += WG) {
        half2 pair = as_type<half2>(logits[in_word_off + w]);
        uint i0 = w << 1;
        uint i1 = i0 + 1u;
        half v0 = pair.x;
        if (i0 < i_hi && v0 > sh_val[base]) {
            sh_val[base] = v0;
            sh_idx[base] = i0;
            sift_down(sh_val, sh_idx, base);
        }
        if (i1 < i_hi) {
            half v1 = pair.y;
            if (v1 > sh_val[base]) {
                sh_val[base] = v1;
                sh_idx[base] = i1;
                sift_down(sh_val, sh_idx, base);
            }
        }
    }

    // Phase 2a: heap-sort to descending.
    for (uint active_minus1 = K_MAX - 1u; active_minus1 > 0u; --active_minus1) {
        uint last = base + active_minus1;
        half tv = sh_val[base];
        sh_val[base] = sh_val[last];
        sh_val[last] = tv;
        uint ti = sh_idx[base];
        sh_idx[base] = sh_idx[last];
        sh_idx[last] = ti;
        uint i = 0u;
        while (true) {
            uint l = 2u * i + 1u;
            uint r = l + 1u;
            uint smallest = i;
            if (l < active_minus1 && sh_val[base + l] < sh_val[base + smallest])
                smallest = l;
            if (r < active_minus1 && sh_val[base + r] < sh_val[base + smallest])
                smallest = r;
            if (smallest == i)
                break;
            half tv2 = sh_val[base + i];
            sh_val[base + i] = sh_val[base + smallest];
            sh_val[base + smallest] = tv2;
            uint ti2 = sh_idx[base + i];
            sh_idx[base + i] = sh_idx[base + smallest];
            sh_idx[base + smallest] = ti2;
            i = smallest;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2b: parallel pairwise bitonic merge.
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

    // Phase 3: write top-K_MAX to scratch slot.
    uint slot_base = p.partials_off + wg_id.x * (2u * K_MAX);
    if (tid < K_MAX) {
        result[slot_base + tid] = as_type<uint>(float(sh_val[tid]));
        result[slot_base + K_MAX + tid] = sh_idx[tid];
    }
}
