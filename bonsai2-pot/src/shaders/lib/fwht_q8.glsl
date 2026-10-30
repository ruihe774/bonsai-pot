// 1024-point normalized Walsh-Hadamard transform and Q8/128 activation store
// for a 256-invocation workgroup where invocation `t` owns the four elements
// 4t..4t+3 of one 1024-element block.
//
// fwht1024(v): y = H x / 32 with H[r][c] = (-1)^popcount(r & c) (Sylvester
// order, natural indexing) — the transform declared by prism.hadamard v1.
// Stages h = 1, 2 run in registers, stages h = 4 .. 2*subgroup_size via
// subgroup shuffles (partner invocation t ^ (h/4)), the rest through LDS.
// Requires subgroups to be aligned runs of consecutive invocations.
//
// store_q8_1024(v, qs_word, d_idx, s_idx): quantize the block to eight Q8
// sub-blocks of 128 (one f32 scale each) and write
//   q8[qs_word + t]          4 packed i8 (elements 4t..4t+3)
//   q8[d_idx + b]            scale d_b = amax_b / 127               (b = t / 32)
//   q8[s_idx + b]            d_b * sum(q) over sub-block b (offset term for
//                            the PTQ1_0 "trit - 1" decode)
// A sub-block is 32 consecutive invocations: one 32-lane cluster when the
// subgroup has at least 32 lanes, otherwise reduced through LDS.

shared float fwht_sh[1024];
shared float q8_red_sh[256];
shared float q8_blk_sh[8];

vec4 fwht1024(vec4 v) {
    uint t = gl_LocalInvocationID.x;
    v = vec4(v.x + v.y, v.x - v.y, v.z + v.w, v.z - v.w);
    v = vec4(v.x + v.z, v.y + v.w, v.x - v.z, v.y - v.w);
    uint sgs = min(gl_SubgroupSize, 256u);
    uint lane = gl_SubgroupInvocationID;
    for (uint m = 1u; m < sgs; m <<= 1u) {
        vec4 o = subgroupShuffleXor(v, m);
        v = (lane & m) != 0u ? o - v : v + o;
    }
    uint h0 = 4u * sgs;
    if (h0 < 1024u) {
        fwht_sh[4u * t + 0u] = v.x;
        fwht_sh[4u * t + 1u] = v.y;
        fwht_sh[4u * t + 2u] = v.z;
        fwht_sh[4u * t + 3u] = v.w;
        barrier();
        for (uint h = h0; h < 1024u; h <<= 1u) {
            [[unroll]] for (uint r = 0u; r < 2u; r += 1u) {
                uint p = t + r * 256u;
                uint i = ((p & ~(h - 1u)) << 1u) | (p & (h - 1u));
                float a = fwht_sh[i];
                float b = fwht_sh[i + h];
                fwht_sh[i] = a + b;
                fwht_sh[i + h] = a - b;
            }
            barrier();
        }
        v = vec4(fwht_sh[4u * t], fwht_sh[4u * t + 1u], fwht_sh[4u * t + 2u], fwht_sh[4u * t + 3u]);
        barrier();
    }
    return v * (1.0 / 32.0);
}

void store_q8_1024(vec4 v, uint qs_word, uint d_idx, uint s_idx) {
    uint t = gl_LocalInvocationID.x;
    bool fast = gl_SubgroupSize >= 32u;
    vec4 a = abs(v);
    float m = max(max(a.x, a.y), max(a.z, a.w));
    float amax;
    if (fast) {
        amax = subgroupClusteredMax(m, 32u);
    } else {
        q8_red_sh[t] = m;
        barrier();
        if ((t & 31u) == 0u) {
            float mm = 0.0;
            for (uint i = 0u; i < 32u; i += 1u) {
                mm = max(mm, q8_red_sh[t + i]);
            }
            q8_blk_sh[t >> 5u] = mm;
        }
        barrier();
        amax = q8_blk_sh[t >> 5u];
    }
    float d = amax / 127.0;
    float id = d > 0.0 ? 1.0 / d : 0.0;
    ivec4 q = clamp(ivec4(round(v * id)), ivec4(-127), ivec4(127));
    q8[qs_word + t] = (uint(q.x) & 0xFFu) | ((uint(q.y) & 0xFFu) << 8u) | ((uint(q.z) & 0xFFu) << 16u) |
                      ((uint(q.w) & 0xFFu) << 24u);
    float qsum = float(q.x + q.y + q.z + q.w);
    float s;
    if (fast) {
        s = subgroupClusteredAdd(qsum, 32u);
    } else {
        barrier();
        q8_red_sh[t] = qsum;
        barrier();
        s = 0.0;
        if ((t & 31u) == 0u) {
            for (uint i = 0u; i < 32u; i += 1u) {
                s += q8_red_sh[t + i];
            }
        }
        barrier();
    }
    if ((t & 31u) == 0u) {
        q8[d_idx + (t >> 5u)] = floatBitsToUint(d);
        q8[s_idx + (t >> 5u)] = floatBitsToUint(d * s);
    }
}
