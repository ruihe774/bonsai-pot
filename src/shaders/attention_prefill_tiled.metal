#include <metal_stdlib>
using namespace metal;

// Q-tiled + GQA-batched FlashAttention-2 prefill kernel.
// dispatch: (n_kv_head, ceil(M / Q_TILE), 1)
// WG=32; subgroup_size>=32 required for the Q8_0 max-abs butterfly mask=16.

struct Params {
    uint head_dim;
    uint n_head;
    uint n_kv_head;
    uint m_tokens;
    uint pos_base;
    uint kv_stride;
    uint q_offset;
    uint k_d_word_offset;
    uint k_qs_byte_offset;
    uint v_d_word_offset;
    uint v_qs_byte_offset;
    uint out_d_offset;
    uint out_qs_offset;
    float scale;
};

constant uint WG = 32u; // exactly one subgroup on Apple (SG=32)
constant uint Q_PER_GROUP = 4u;
constant uint ELEMS_PER_THREAD = 4u;
constant uint Q_TILE = 2u;
constant float NEG_INF = -1.0e30f;

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

kernel void attention_prefill_tiled(
    constant Params& p [[buffer(0)]],
    device const half* act [[buffer(1)]],
    device const uint* k_cache [[buffer(2)]],
    device const uint* v_cache [[buffer(3)]],
    device uint* outbuf [[buffer(4)]],
    uint3 wg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]])
{
    uint g = wg_id.x;
    uint m_tile = wg_id.y;
    if (g >= p.n_kv_head)
        return;
    uint m_base = m_tile * Q_TILE;
    if (m_base >= p.m_tokens)
        return;
    uint hd = p.head_dim;
    uint q_stride = p.n_head * hd;
    uint g_off = g * hd;
    uint m_idx0 = m_base + 0u;
    uint m_idx1 = m_base + 1u;
    bool valid1 = m_idx1 < p.m_tokens;

    uint t_max = p.pos_base + (valid1 ? (m_idx1 + 1u) : (m_idx0 + 1u));

    uint q_base0 = p.q_offset + m_idx0 * q_stride + g * Q_PER_GROUP * hd;
    uint q_base1 = p.q_offset + m_idx1 * q_stride + g * Q_PER_GROUP * hd;

    uint d_lane[ELEMS_PER_THREAD];
    #pragma unroll
    for (uint e = 0u; e < ELEMS_PER_THREAD; ++e) { d_lane[e] = tid + e * WG; }

    uint q_base[Q_TILE];
    q_base[0] = q_base0;
    q_base[1] = q_base1;

    uint m_idx_arr[Q_TILE];
    m_idx_arr[0] = m_idx0;
    m_idx_arr[1] = m_idx1;

    bool ml_valid[Q_TILE];
    ml_valid[0] = true;
    ml_valid[1] = valid1;

    float q[Q_TILE][Q_PER_GROUP][ELEMS_PER_THREAD];
    #pragma unroll
    for (uint ml = 0u; ml < Q_TILE; ++ml) {
        #pragma unroll
        for (uint pg = 0u; pg < Q_PER_GROUP; ++pg) {
            #pragma unroll
            for (uint e = 0u; e < ELEMS_PER_THREAD; ++e) {
                q[ml][pg][e] = float(act[q_base[ml] + pg * hd + d_lane[e]]);
            }
        }
    }

    float4 m_run[Q_TILE];
    float4 l_run[Q_TILE];
    #pragma unroll
    for (uint ml = 0u; ml < Q_TILE; ++ml) {
        m_run[ml] = float4(NEG_INF);
        l_run[ml] = float4(0.0f);
    }

    float o[Q_TILE][Q_PER_GROUP][ELEMS_PER_THREAD];
    #pragma unroll
    for (uint ml = 0u; ml < Q_TILE; ++ml) {
        #pragma unroll
        for (uint pg = 0u; pg < Q_PER_GROUP; ++pg) {
            #pragma unroll
            for (uint e = 0u; e < ELEMS_PER_THREAD; ++e) { o[ml][pg][e] = 0.0f; }
        }
    }

    for (uint t = 0u; t < t_max; ++t) {
        float k_e[ELEMS_PER_THREAD];
        #pragma unroll
        for (uint e = 0u; e < ELEMS_PER_THREAD; ++e) {
            k_e[e] = load_kv_q8(k_cache, p.k_d_word_offset, p.k_qs_byte_offset, t, g_off + d_lane[e], p.kv_stride);
        }

        float4 corr[Q_TILE];
        float4 w_attn[Q_TILE];
        #pragma unroll
        for (uint ml = 0u; ml < Q_TILE; ++ml) {
            float4 local_dots;
            #pragma unroll
            for (uint pg = 0u; pg < Q_PER_GROUP; ++pg) {
                float d_acc = 0.0f;
                #pragma unroll
                for (uint e = 0u; e < ELEMS_PER_THREAD; ++e) { d_acc += q[ml][pg][e] * k_e[e]; }
                local_dots[pg] = d_acc;
            }
            // WG=32 = one subgroup on Apple, so a single simd_sum suffices.
            float4 dots = simd_sum(local_dots);
            bool causal_ok = ml_valid[ml] && (t <= p.pos_base + m_idx_arr[ml]);
            float4 scores = causal_ok ? (dots * p.scale) : float4(NEG_INF);
            float4 m_new = max(m_run[ml], scores);
            corr[ml] = exp(m_run[ml] - m_new);
            w_attn[ml] = exp(scores - m_new);
            l_run[ml] = l_run[ml] * corr[ml] + w_attn[ml];
            m_run[ml] = m_new;
        }

        float v_e[ELEMS_PER_THREAD];
        #pragma unroll
        for (uint e = 0u; e < ELEMS_PER_THREAD; ++e) {
            v_e[e] = load_kv_q8(v_cache, p.v_d_word_offset, p.v_qs_byte_offset, t, g_off + d_lane[e], p.kv_stride);
        }

        #pragma unroll
        for (uint ml = 0u; ml < Q_TILE; ++ml) {
            #pragma unroll
            for (uint pg = 0u; pg < Q_PER_GROUP; ++pg) {
                #pragma unroll
                for (uint e = 0u; e < ELEMS_PER_THREAD; ++e) {
                    o[ml][pg][e] = o[ml][pg][e] * corr[ml][pg] + w_attn[ml][pg] * v_e[e];
                }
            }
        }
    }

    // ---- Normalize and write Q8_0 directly to outbuf ----
    uint nb_per_token = p.n_head * hd / 32u;

    #pragma unroll
    for (uint ml = 0u; ml < Q_TILE; ++ml) {
        if (!ml_valid[ml])
            continue;
        float4 inv = float4(1.0f) / l_run[ml];
        uint block_base = m_idx_arr[ml] * nb_per_token + g * Q_PER_GROUP * 4u;
        #pragma unroll
        for (uint pg = 0u; pg < Q_PER_GROUP; ++pg) {
            #pragma unroll
            for (uint e = 0u; e < ELEMS_PER_THREAD; ++e) {
                float v = o[ml][pg][e] * inv[pg];
                uint block_global = block_base + pg * 4u + e;

                uint lane_in_block = sg_lane & 31u;
                float m_abs = fabs(v);
                m_abs = max(m_abs, simd_shuffle_xor(m_abs, 1u));
                m_abs = max(m_abs, simd_shuffle_xor(m_abs, 2u));
                m_abs = max(m_abs, simd_shuffle_xor(m_abs, 4u));
                m_abs = max(m_abs, simd_shuffle_xor(m_abs, 8u));
                m_abs = max(m_abs, simd_shuffle_xor(m_abs, 16u));
                float d = m_abs / 127.0f;
                float id_inv = (d > 0.0f) ? (1.0f / d) : 0.0f;
                uint qv = uint(int(clamp(rint(v * id_inv), -127.0f, 127.0f))) & 0xFFu;
                uint base = sg_lane & ~3u;
                uint packed = simd_shuffle(qv, base + 0u) | (simd_shuffle(qv, base + 1u) << 8) |
                              (simd_shuffle(qv, base + 2u) << 16) | (simd_shuffle(qv, base + 3u) << 24);
                if ((lane_in_block & 3u) == 0u) {
                    uint qs_byte = p.out_qs_offset + block_global * 32u + lane_in_block;
                    outbuf[qs_byte >> 2] = packed;
                }
                if (lane_in_block == 0u) {
                    outbuf[(p.out_d_offset >> 2) + block_global] = as_type<uint>(d);
                }
            }
        }
    }
}
