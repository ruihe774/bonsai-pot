// PTQ1_0 block decode helpers. A block is 7 u32 words (28 bytes):
//   words 0..5  qs[24]: byte j < 16 carries elements 16t + j (t = 0..4);
//                       byte 16 + j (j < 8) carries elements 80 + 8t + j
//   word  6     qh[2] in bytes 0..1: byte h carries elements 120 + 2n + h
//               (n = 0..3); fp16 scale d in bytes 2..3
// Trit recurrence per byte v: trit = (v*3) >> 8, then v = (v*3) & 0xFF.
// Weight = (trit - 1) * d. Requires `w[]` (lib/bindings.glsl).

float ptq1_0_scale(uint wb) { return unpackHalf2x16(w[wb + 6u]).y; }

// Unscaled ternary value (trit - 1) of element e in [0, 128) of the block at
// word offset wb. Slow per-element path, used by the embedding lookup.
float ptq1_0_trit(uint wb, uint e) {
    uint byte_idx;
    uint n;
    if (e < 80u) {
        byte_idx = e & 15u;
        n = e >> 4u;
    } else if (e < 120u) {
        uint t = e - 80u;
        byte_idx = 16u + (t & 7u);
        n = t >> 3u;
    } else {
        uint t = e - 120u;
        byte_idx = 24u + (t & 1u);
        n = t >> 1u;
    }
    uint b = (w[wb + (byte_idx >> 2u)] >> ((byte_idx & 3u) * 8u)) & 0xFFu;
    for (uint i = 0u; i < n; i += 1u) {
        b = (b * 3u) & 0xFFu;
    }
    return float(int((b * 3u) >> 8u) - 1);
}

// v*3 on both 16-bit lanes of a word (a shift-add on GCN/RDNA).
#define MUL3(v) (((v) << 1u) + (v))
