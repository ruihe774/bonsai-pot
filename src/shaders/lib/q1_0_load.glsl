// Q1_0 / Q2_0 weight helpers.
// Contract: caller declares `WBuf { uint weights[]; }` at an appropriate binding.

// Pipeline-time format selector: 0 = Q1_0 (16 qs bytes/block, binary signs),
// 1 = Q2_0 (32 qs bytes/block, 2-bit codes). Patched at `Model::load` via
// `spirv_set_spec_const_u32`; the driver constant-folds the conditional below
// so only the active arm survives.
layout(constant_id = 3) const uint QUANT_FORMAT = 0u;

// Per-block qs byte stride. Q1_0 packs 128 weights as 16 bytes of sign bits;
// Q2_0 packs the same 128 weights as 32 bytes of 2-bit codes.
const uint QS_BYTES_PER_BLOCK = 16u << QUANT_FORMAT;

// Q1_0: 4 u32 qwords per 128-weight block (16 qs bytes). Q2_0: 8 qwords
// (32 qs bytes). Driver folds the conditional via SpecConstId 3.
const uint W_QWORDS_PER_BLOCK = 4u << QUANT_FORMAT;

float load_f16_at(uint b_offset) {
    uint word = weights[b_offset >> 2];
    uint half_bits = (word >> ((b_offset & 2u) * 8u));
    return unpackFloat2x16(half_bits).x;
}

#ifndef METAL_BACKEND
uint expand_4_bits(uint bits) {
    // Spread 4 input bits to 4 byte LSBs: bit i → byte i, value 0 or 1.
    // 0x00204081 has bits at positions {0,7,14,21}; the mul+mask lands each
    // nibble bit at the LSB of its byte. Negate to map {0,1}→{0xFF,0x01}
    // in the packed signed-byte form expected by dotPacked4x8EXT.
    uint spread = (bits * 0x00204081u) & 0x01010101u;
    return ~(spread * 0xFEu);
}

uint expand_8_bits(uint byte) {
    // Spread 4 input 2-bit codes (in `byte`'s low 8 bits) to 4 byte lanes
    // holding signed `(q - 1)` ∈ {-1, 0, +1} (= bytes {0xFF, 0x00, 0x01})
    // for dotPacked4x8EXT consumption. (0b11 is not valid for Q2_0)
    uint spread = ((byte >> 2u) * 0x104100u) | byte;
    uint mask = ~spread;
    uint trail = mask & 0x01010101u;
    return (((mask >> 1u) & trail) * 0xFFu) | trail;
}
#else
float reduce_add(vec4 v) { return v.x + v.y + v.z + v.w; }
#endif
