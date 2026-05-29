// Q1_0/Q2_0/Q8_0 per-super-block dot product shared across matvec kernels.
//
// Must be included after lib/q1_0_load.glsl (needs expand_*_bits / reduce_add
// and access to weights[]).
//
// Contract — caller must declare at file scope before this include:
//   Vulkan:
//     shared uint  q1_a_qs_sh[...];  // packed i8 activation tile
//     shared float q1_a_d_sh[...];   // Q8_0 per-block scales
//     uint (or const uint) Q1_NB_Q8; // stride of q1_a_qs_sh across 8 i-lanes
//   Metal:
//     shared f16vec4 q1_x_sh[...];   // f16 activation tile (4 elems per slot)
//
// q1_0_block_dot(qs_word_base, b_idx, d_byte_base) → float
//   qs_word_base : u32-word offset into weights[] of this super-block's qs
//   b_idx        : super-block index (tile-local for tiled kernels, absolute
//                  for whole-row kernels); determines the shmem base address
//   d_byte_base  : byte offset into weights[] of this super-block's FP16 d.
//                  Q1_0/Q2_0 read 1 scale; Q8_0 reads 4 (one per 32-elem
//                  sub-block at d_byte_base + s*2).
//
// Return value is fully scaled by the per-block weight d(s) — callers
// accumulate it directly without an outer d_w multiply.

float q1_0_block_dot(uint qs_word_base, uint b_idx, uint d_byte_base) {
#ifndef METAL_BACKEND
    float sub_acc = 0.0;
    if (QUANT_FORMAT == 0u) {
        // Q1_0: 4 u32 words per super-block, one per Q8_0 sub-block. Each word
        // packs 32 weights as 8 sign nibbles; each nibble feeds expand_4_bits
        // and contributes one dotPacked4x8 against 4 i8 activation lanes.
        float w_d = load_f16_at(d_byte_base);
        [[unroll]] for (uint s = 0u; s < 4u; s += 1u) {
            uint qword = weights[qs_word_base + s];
            uint block_l = b_idx * 4u + s;
            float a_d = q1_a_d_sh[block_l];
            int sumi = 0;
            [[unroll]] for (uint i = 0u; i < 8u; i += 1u) {
                uint bits = extract_bits(qword, i * 4u, 4u);
                uint w_packed = expand_4_bits(bits);
                uint a_packed = q1_a_qs_sh[i * Q1_NB_Q8 + block_l];
                sumi = dotPacked4x8EXT(int(w_packed), int(a_packed)) + sumi;
            }
            sub_acc += a_d * float(sumi);
        }
        sub_acc *= w_d;
    } else if (QUANT_FORMAT == 1u) {
        // Q2_0: 8 u32 words per super-block (2 per Q8_0 sub-block). Each word
        // packs 16 weights as 4 bytes of 4 × 2-bit codes; each byte feeds
        // expand_8_bits and contributes one dotPacked4x8 against 4 i8 lanes.
        float w_d = load_f16_at(d_byte_base);
        [[unroll]] for (uint s = 0u; s < 4u; s += 1u) {
            uint block_l = b_idx * 4u + s;
            float a_d = q1_a_d_sh[block_l];
            int sumi = 0;
            [[unroll]] for (uint w = 0u; w < 2u; w += 1u) {
                uint qword = weights[qs_word_base + s * 2u + w];
                [[unroll]] for (uint i = 0u; i < 4u; i += 1u) {
                    uint byte = extract_bits(qword, i * 8u, 8u);
                    uint w_packed = expand_8_bits(byte);
                    uint a_packed = q1_a_qs_sh[(w * 4u + i) * Q1_NB_Q8 + block_l];
                    sumi = dotPacked4x8EXT(int(w_packed), int(a_packed)) + sumi;
                }
            }
            sub_acc += a_d * float(sumi);
        }
        sub_acc *= w_d;
    } else {
        // Q8_0: 32 u32 words per super-block (8 per 32-elem native sub-block);
        // each word is 4 i8 weights consumed directly by dotPacked4x8 — no
        // expand step. One FP16 weight scale per sub-block; the 4 scales sit
        // in 8 contiguous bytes, fetched as two f16vec2 loads.
        vec4 wd = vec4(load_2xf16_at(d_byte_base), load_2xf16_at(d_byte_base + 4u));
        [[unroll]] for (uint s = 0u; s < 4u; s += 1u) {
            uint block_l = b_idx * 4u + s;
            float a_d = q1_a_d_sh[block_l];
            float w_d = wd[s];
            int sumi = 0;
            [[unroll]] for (uint i = 0u; i < 8u; i += 1u) {
                uint w_packed = weights[qs_word_base + s * 8u + i];
                uint a_packed = q1_a_qs_sh[i * Q1_NB_Q8 + block_l];
                sumi = dotPacked4x8EXT(int(w_packed), int(a_packed)) + sumi;
            }
            sub_acc += w_d * a_d * float(sumi);
        }
    }
    return sub_acc;
#else
    // Metal path: only Q1_0 is reachable (Q2_0/Q8_0 are rejected at load).
    float w_d = load_f16_at(d_byte_base);
    uint x_base = b_idx * 32u;
    f16vec4 acc4 = f16vec4(0.0);
    [[unroll]] for (uint s = 0u; s < 4u; s += 1u) {
        uint qword = weights[qs_word_base + s];
        [[unroll]] for (uint i = 0u; i < 8u; i += 1u) {
            uint bits = (qword >> (i * 4u)); // & 0xFu;
            f16vec4 a4 = q1_x_sh[x_base + s * 8u + i];
            // Q1_0 sign convention: bit=1 → +a, bit=0 → -a.
            bvec4 sb = bvec4((bits & 1u) != 0u, (bits & 2u) != 0u, (bits & 4u) != 0u, (bits & 8u) != 0u);
            acc4 += mix(-a4, a4, sb);
        }
    }
    return reduce_add(vec4(acc4)) * w_d;
#endif
}
