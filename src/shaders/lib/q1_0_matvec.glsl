// Q1_0 per-block dot product shared across matvec kernels.
//
// Must be included after lib/q1_0_load.glsl (needs expand_4_bits / reduce_add
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
// q1_0_block_dot(qs_word_base, b_idx) → float
//   qs_word_base : u32-word offset into weights[] of this row's 4 sign words
//   b_idx        : Q1_0 block index (tile-local for tiled kernels, absolute
//                  for whole-row kernels); determines the shmem base address

float q1_0_block_dot(uint qs_word_base, uint b_idx) {
#ifndef METAL_BACKEND
    float sub_acc = 0.0;
    if (QUANT_FORMAT == 0u) {
        // Q1_0: 4 u32 words per Q1_0 block, one per Q8_0 sub-block. Each word
        // packs 32 weights as 8 sign nibbles; each nibble feeds expand_4_bits
        // and contributes one dotPacked4x8 against 4 i8 activation lanes.
        [[unroll]] for (uint s = 0u; s < 4u; ++s) {
            uint qword = weights[qs_word_base + s];
            uint block_l = b_idx * 4u + s;
            float a_d = q1_a_d_sh[block_l];
            int sumi = 0;
            [[unroll]] for (uint i = 0u; i < 8u; ++i) {
                uint bits = (qword >> (i * 4u)) & 0xFu;
                uint w_packed = expand_4_bits(bits);
                uint a_packed = q1_a_qs_sh[i * Q1_NB_Q8 + block_l];
                sumi = dotPacked4x8EXT(int(w_packed), int(a_packed)) + sumi;
            }
            sub_acc += a_d * float(sumi);
        }
    } else {
        // Q2_0: 8 u32 words per Q1_0 block (2 per Q8_0 sub-block). Each word
        // packs 16 weights as 4 bytes of 4 × 2-bit codes; each byte feeds
        // expand_8_bits and contributes one dotPacked4x8 against 4 i8 lanes.
        [[unroll]] for (uint s = 0u; s < 4u; ++s) {
            uint block_l = b_idx * 4u + s;
            float a_d = q1_a_d_sh[block_l];
            int sumi = 0;
            [[unroll]] for (uint w = 0u; w < 2u; ++w) {
                uint qword = weights[qs_word_base + s * 2u + w];
                [[unroll]] for (uint i = 0u; i < 4u; ++i) {
                    uint byte = (qword >> (i * 8u)) & 0xFFu;
                    uint w_packed = expand_8_bits(byte);
                    uint a_packed = q1_a_qs_sh[(w * 4u + i) * Q1_NB_Q8 + block_l];
                    sumi = dotPacked4x8EXT(int(w_packed), int(a_packed)) + sumi;
                }
            }
            sub_acc += a_d * float(sumi);
        }
    }
    return sub_acc;
#else
    uint x_base = b_idx * 32u;
    f16vec4 acc4 = f16vec4(0.0);
    [[unroll]] for (uint s = 0u; s < 4u; ++s) {
        uint qword = weights[qs_word_base + s];
        [[unroll]] for (uint i = 0u; i < 8u; ++i) {
            uint bits = (qword >> (i * 4u)); // & 0xFu;
            f16vec4 a4 = q1_x_sh[x_base + s * 8u + i];
            // Q1_0 sign convention: bit=1 → +a, bit=0 → -a.
            bvec4 sb = bvec4((bits & 1u) != 0u, (bits & 2u) != 0u, (bits & 4u) != 0u, (bits & 8u) != 0u);
            acc4 += mix(-a4, a4, sb);
        }
    }
    return reduce_add(vec4(acc4));
#endif
}
