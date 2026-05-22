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
