#include <metal_stdlib>
using namespace metal;

// Split-K + GQA-batched attention chunk pass for the matvec (m_tokens=1) path.
// dispatch: (n_kv_head, n_chunks_active, 1).

struct Params {
    uint head_dim;
    uint n_head;
    uint n_kv_head;
    uint pos;
    uint kv_stride;
    uint q_offset;
    uint k_d_word_offset;
    uint k_qs_byte_offset;
    uint v_d_word_offset;
    uint v_qs_byte_offset;
    uint n_chunks_active;
    float scale;
};

// Apple sweep: WG=32 (1 simdgroup) wins ~5% on tg over the previous WG=64
// (2-SG) layout. The cross-SG `wg_sum_v4` becomes a no-op (NUM_SUBGROUPS=1
// → simd_sum already covers the whole WG). EPT=4 covers head_dim=128.
constant uint WG = 32u;
constant uint NUM_SUBGROUPS = WG / 32u; // = 1
constant uint Q_PER_GROUP = 4u;
constant uint ELEMS_PER_THREAD = 4u;
constant uint CHUNK_SIZE = 8u;
constant uint PARTIAL_STRIDE = 130u;

static float load_kv_q8(device const uint* cache, uint d_word_offset, uint qs_byte_offset, uint t, uint e_local, uint kv_stride) {
    uint elem_idx = t * kv_stride + e_local;
    uint block_idx = elem_idx >> 5;
    float scale = as_type<float>(cache[d_word_offset + block_idx]);
    uint qs_byte_idx = qs_byte_offset + elem_idx;
    uint qs_word = cache[qs_byte_idx >> 2];
    uint shift = (qs_byte_idx & 3u) << 3;
    uint qs_byte = (qs_word >> shift) & 0xFFu;
    int qs_signed = int(qs_byte << 24) >> 24;
    return scale * float(qs_signed);
}

kernel void cs_main(
    constant Params& p [[buffer(0)]],
    device const half* act [[buffer(1)]],
    device const uint* k_cache [[buffer(2)]],
    device const uint* v_cache [[buffer(3)]],
    device float* partials [[buffer(4)]],
    uint3 wg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float4 sg_partial4[NUM_SUBGROUPS];

    uint g = wg_id.x;
    uint chunk = wg_id.y;
    if (g >= p.n_kv_head || chunk >= p.n_chunks_active)
        return;
    uint hd = p.head_dim;
    uint chunk_start = chunk * CHUNK_SIZE;
    uint chunk_end = min(chunk_start + CHUNK_SIZE, p.pos);
    uint g_off = g * hd;

    uint q_base0 = p.q_offset + (g * Q_PER_GROUP + 0u) * hd;
    uint q_base1 = p.q_offset + (g * Q_PER_GROUP + 1u) * hd;
    uint q_base2 = p.q_offset + (g * Q_PER_GROUP + 2u) * hd;
    uint q_base3 = p.q_offset + (g * Q_PER_GROUP + 3u) * hd;

    float o0[ELEMS_PER_THREAD];
    float o1[ELEMS_PER_THREAD];
    float o2[ELEMS_PER_THREAD];
    float o3[ELEMS_PER_THREAD];
    #pragma unroll
    for (uint i = 0u; i < ELEMS_PER_THREAD; ++i) {
        o0[i] = 0.0f; o1[i] = 0.0f; o2[i] = 0.0f; o3[i] = 0.0f;
    }
    float4 m_run = float4(-1e30f);
    float4 l_run = float4(0.0f);

    for (uint t = chunk_start; t < chunk_end; ++t) {
        float4 local_v = float4(0.0f);
        for (uint d = tid; d < hd; d += WG) {
            float k = load_kv_q8(k_cache, p.k_d_word_offset, p.k_qs_byte_offset, t, g_off + d, p.kv_stride);
            float q0 = float(act[q_base0 + d]);
            float q1 = float(act[q_base1 + d]);
            float q2 = float(act[q_base2 + d]);
            float q3 = float(act[q_base3 + d]);
            local_v += k * float4(q0, q1, q2, q3);
        }

        // wg_sum_v4 inlined; NUM_SUBGROUPS=2 here.
        float4 sg_sum = simd_sum(local_v);
        if (sg_lane == 0u)
            sg_partial4[sg_id] = sg_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0u) {
            float4 combined = (sg_lane < NUM_SUBGROUPS) ? sg_partial4[sg_lane] : float4(0.0f);
            float4 final_sum = simd_sum(combined);
            if (sg_lane == 0u)
                sg_partial4[0] = final_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float4 dots = sg_partial4[0];
        float4 scores = dots * p.scale;

        float4 m_new = max(m_run, scores);
        float4 correction = exp(m_run - m_new);
        float4 weight = exp(scores - m_new);
        l_run = l_run * correction + weight;
        m_run = m_new;

        #pragma unroll
        for (uint i = 0u; i < ELEMS_PER_THREAD; ++i) {
            uint d = tid + i * WG;
            float v = load_kv_q8(v_cache, p.v_d_word_offset, p.v_qs_byte_offset, t, g_off + d, p.kv_stride);
            o0[i] = o0[i] * correction.x + weight.x * v;
            o1[i] = o1[i] * correction.y + weight.y * v;
            o2[i] = o2[i] * correction.z + weight.z * v;
            o3[i] = o3[i] * correction.w + weight.w * v;
        }
    }

    uint h0 = g * Q_PER_GROUP + 0u;
    uint h1 = g * Q_PER_GROUP + 1u;
    uint h2 = g * Q_PER_GROUP + 2u;
    uint h3 = g * Q_PER_GROUP + 3u;
    uint stride_h = p.n_chunks_active * PARTIAL_STRIDE;
    uint base0 = h0 * stride_h + chunk * PARTIAL_STRIDE;
    uint base1 = h1 * stride_h + chunk * PARTIAL_STRIDE;
    uint base2 = h2 * stride_h + chunk * PARTIAL_STRIDE;
    uint base3 = h3 * stride_h + chunk * PARTIAL_STRIDE;
    #pragma unroll
    for (uint i = 0u; i < ELEMS_PER_THREAD; ++i) {
        uint d = tid + i * WG;
        partials[base0 + d] = o0[i];
        partials[base1 + d] = o1[i];
        partials[base2 + d] = o2[i];
        partials[base3 + d] = o3[i];
    }
    if (tid == 0u) {
        partials[base0 + hd] = m_run.x;
        partials[base0 + hd + 1u] = l_run.x;
        partials[base1 + hd] = m_run.y;
        partials[base1 + hd + 1u] = l_run.y;
        partials[base2 + hd] = m_run.z;
        partials[base2 + hd + 1u] = l_run.z;
        partials[base3 + hd] = m_run.w;
        partials[base3 + hd + 1u] = l_run.w;
    }
}
