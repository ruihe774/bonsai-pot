// Q1_0 weight helpers.
// Contract: caller declares `WBuf { uint weights[]; }` at an appropriate binding.

float load_f16_at(uint b_offset) {
    uint word = weights[b_offset >> 2];
    uint half_bits = (word >> ((b_offset & 2u) * 8u)) & 0xFFFFu;
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
#else
float reduce_add(vec4 v) { return v.x + v.y + v.z + v.w; }
#endif
