// Q1_0 / Q2_0 / Q8_0 weight helpers.
// Contract: caller declares `WBuf { uint weights[]; }` at an appropriate binding.

// Pipeline-time format selector. Encoded so the `<< QUANT_FORMAT` shifts
// below give the right qs stride per 128-elem super-block:
//   0 = Q1_0 (16 qs bytes, binary signs)
//   1 = Q2_0 (32 qs bytes, ternary 2-bit codes)
//   3 = Q8_0 (128 qs bytes = 4 × 32 native i8 blocks, 4 FP16 d/super-block)
// Patched at `Model::load` via `spirv_set_spec_const_u32`; the driver
// constant-folds the conditionals below so only the active arm survives.
layout(constant_id = 3) const uint QUANT_FORMAT = 0u;

// Per-super-block qs byte stride: 16 / 32 / 128 for Q1_0/Q2_0/Q8_0.
const uint QS_BYTES_PER_BLOCK = 16u << QUANT_FORMAT;

// Per-super-block u32 word count: 4 / 8 / 32 for Q1_0/Q2_0/Q8_0.
const uint W_QWORDS_PER_BLOCK = 4u << QUANT_FORMAT;

// FP16 d-scales per 128-elem super-block: 1 for Q1_0/Q2_0, 4 for Q8_0.
// Drives the per-row d-array stride at every caller.
const uint D_FP16_PER_BLOCK = (QUANT_FORMAT == 3u) ? 4u : 1u;

vec2 load_2xf16_at(uint b_offset) {
    // b_offset must be 4-byte aligned. All Bonsai d-arrays are u32-aligned
    // and we only ever pair adjacent (even-indexed-base) f16 scales.
    return unpackHalf2x16(weights[b_offset >> 2u]);
}

float load_f16_at(uint b_offset) { return load_2xf16_at(b_offset)[extract_bits(b_offset, 1u, 1u)]; }

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

float reduce_add(vec4 v) { return v.x + v.y + v.z + v.w; }
