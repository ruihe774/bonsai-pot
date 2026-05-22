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
    // holding signed `(q - 1)` ∈ {-1, 0, +1, +2} (= bytes {0xFF, 0x00, 0x01,
    // 0x02}) for dotPacked4x8EXT consumption.
    //
    // The mul-spread trick used by expand_4_bits doesn't work here: for any
    // byte where two set bits collide under the shifted sum (e.g. 0x43 →
    // bit 6 from byte<<0 plus bit 0 from byte<<6 both at u32 bit 6), the carry
    // propagates and corrupts the next code's lane. Extract the 4 codes
    // explicitly instead — still cheap (4 shifts + 4 ANDs + 3 ORs).
    uint packed = (byte & 0x03u) | (((byte >> 2u) & 0x03u) << 8u) | (((byte >> 4u) & 0x03u) << 16u) |
                  (((byte >> 6u) & 0x03u) << 24u);
    // Apply (q - 1) per byte without inter-byte borrow:
    //   - low 2 bits = (q + 3) & 3 (no carry: each byte's q+3 ≤ 6 < 256);
    //   - high 6 bits = 0xFC iff q == 0 else 0x00 (sign extension of −1).
    uint low2 = (packed + 0x03030303u) & 0x03030303u;
    uint nz = (packed | (packed >> 1u)) & 0x01010101u;
    uint sx = (nz ^ 0x01010101u) * 0xFCu;
    return low2 | sx;
}
#else
float reduce_add(vec4 v) { return v.x + v.y + v.z + v.w; }
#endif
