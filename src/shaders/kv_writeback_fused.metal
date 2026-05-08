#include <metal_stdlib>
using namespace metal;

// Fused per-layer K-side pre-KV pipeline:
//   rms_norm(K head) -> *w_k_norm -> NEOX-RoPE -> Q8_0 quantize -> write kv_k.
// V-side: Q8_0 quantize + write kv_v in the same workgroup.

struct Params {
    uint k_cur_off;
    uint v_cur_off;
    uint w_k_norm_off;
    uint rope_offset;
    uint dst_d_word_offset;
    uint dst_qs_byte_offset;
    uint pos_base;
    uint kv_dim;
    uint nb_per_row;
    float eps;
};

constant uint HEAD_DIM = 128u;
constant uint HALF_DIM = 64u;
constant uint NB_PER_HEAD = HEAD_DIM / 32u;
constant uint WG = 128u;
constant uint NUM_SUBGROUPS = WG / 32u; // = 4

kernel void cs_main(
    constant Params& p [[buffer(0)]],
    device const half* act [[buffer(1)]],
    device const half* w_norms [[buffer(2)]],
    device const half* rope_cs [[buffer(3)]],
    device uint* kv_k [[buffer(4)]],
    device uint* kv_v [[buffer(5)]],
    uint3 wg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup half k_sh[128];
    threadgroup float sg_partial[NUM_SUBGROUPS];

    uint head = wg_id.x;
    uint tok = wg_id.y;

    uint kv_token_off = tok * p.kv_dim + head * HEAD_DIM;
    float k_raw = float(act[p.k_cur_off + kv_token_off + tid]);
    float v_raw = float(act[p.v_cur_off + kv_token_off + tid]);

    float sg_sum = simd_sum(k_raw * k_raw);
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
    float inv_h = rsqrt(total / float(HEAD_DIM) + p.eps);

    k_sh[tid] = half(k_raw * inv_h) * w_norms[p.w_k_norm_off + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint pos_abs = p.pos_base + tok;
    uint cs_base = p.rope_offset + pos_abs * HEAD_DIM;
    half k_post_h;
    if (tid < HALF_DIM) {
        half c = rope_cs[cs_base + tid * 2u];
        half s = rope_cs[cs_base + tid * 2u + 1u];
        k_post_h = k_sh[tid] * c - k_sh[tid + HALF_DIM] * s;
    } else {
        uint j = tid - HALF_DIM;
        half c = rope_cs[cs_base + j * 2u];
        half s = rope_cs[cs_base + j * 2u + 1u];
        k_post_h = k_sh[j] * s + k_sh[tid] * c;
    }
    float k_post = float(k_post_h);

    float ak = fabs(k_post);
    float av = fabs(v_raw);

    ak = max(ak, simd_shuffle_xor(ak, 1u));
    av = max(av, simd_shuffle_xor(av, 1u));
    ak = max(ak, simd_shuffle_xor(ak, 2u));
    av = max(av, simd_shuffle_xor(av, 2u));
    ak = max(ak, simd_shuffle_xor(ak, 4u));
    av = max(av, simd_shuffle_xor(av, 4u));
    ak = max(ak, simd_shuffle_xor(ak, 8u));
    av = max(av, simd_shuffle_xor(av, 8u));
    ak = max(ak, simd_shuffle_xor(ak, 16u));
    av = max(av, simd_shuffle_xor(av, 16u));

    uint block_in_head = tid >> 5;
    float dk = ak / 127.0f;
    float dv = av / 127.0f;
    float inv_dk = (dk > 0.0f) ? (1.0f / dk) : 0.0f;
    float inv_dv = (dv > 0.0f) ? (1.0f / dv) : 0.0f;

    uint qk = uint(int(clamp(rint(k_post * inv_dk), -127.0f, 127.0f))) & 0xFFu;
    uint qv = uint(int(clamp(rint(v_raw * inv_dv), -127.0f, 127.0f))) & 0xFFu;

    uint cluster_base = sg_lane & ~3u;
    uint qk0 = simd_shuffle(qk, cluster_base);
    uint qk1 = simd_shuffle(qk, cluster_base + 1u);
    uint qk2 = simd_shuffle(qk, cluster_base + 2u);
    uint qk3 = simd_shuffle(qk, cluster_base + 3u);
    uint qv0 = simd_shuffle(qv, cluster_base);
    uint qv1 = simd_shuffle(qv, cluster_base + 1u);
    uint qv2 = simd_shuffle(qv, cluster_base + 2u);
    uint qv3 = simd_shuffle(qv, cluster_base + 3u);

    if ((tid & 3u) == 0u) {
        uint pk = qk0 | (qk1 << 8) | (qk2 << 16) | (qk3 << 24);
        uint pv = qv0 | (qv1 << 8) | (qv2 << 16) | (qv3 << 24);
        uint qs_byte_in_layer = p.dst_qs_byte_offset + pos_abs * p.kv_dim + head * HEAD_DIM + tid;
        kv_k[qs_byte_in_layer >> 2] = pk;
        kv_v[qs_byte_in_layer >> 2] = pv;
    }

    if ((tid & 31u) == 0u) {
        uint block_global = pos_abs * p.nb_per_row + head * NB_PER_HEAD + block_in_head;
        kv_k[p.dst_d_word_offset + block_global] = as_type<uint>(dk);
        kv_v[p.dst_d_word_offset + block_global] = as_type<uint>(dv);
    }
}
