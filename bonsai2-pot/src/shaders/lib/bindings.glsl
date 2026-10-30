// Universal bind-group layout shared by every kernel (one pipeline layout for
// the whole engine). Each kernel includes this after optionally overriding the
// memory qualifier of the buffers it only reads (e.g. `#define ACT_QUAL readonly`),
// which lets the driver use non-coherent loads. Unused bindings are dead-code
// eliminated by spirv-opt.
//
//   0  w       weights of the active weight set (PTQ1_0 blocks, 7 u32 each)  ro
//   1  ws      small weights: f32 norms / conv / gates, BF16 pairs, sign bits ro
//   2  act     f32 activation scratch (regions laid out by ActLayout)        rw
//   3  q8      Q8/128 activations: qs bytes | f32 scales | f32 scale*sum     rw
//   4  kv_k    Q8_0 K cache: f32 scales section | i8 qs section             rw
//   5  kv_v    Q8_0 V cache (same layout)                                    rw
//   6  st      f32 recurrent state: GDN conv states | GDN ssm states         rw
//   7  io      u32 token ids in / top-K results out                         rw
//   8  rope    vec2(cos, sin) per (position, rotary pair)                    ro

#ifndef ACT_QUAL
#define ACT_QUAL
#endif
#ifndef Q8_QUAL
#define Q8_QUAL
#endif
#ifndef KV_QUAL
#define KV_QUAL
#endif
#ifndef ST_QUAL
#define ST_QUAL
#endif
#ifndef SAMPLE_QUAL
#define SAMPLE_QUAL
#endif

layout(set = 0, binding = 0, std430) readonly buffer WBuf { uint w[]; };
layout(set = 0, binding = 1, std430) readonly buffer SmallBuf { uint ws[]; };
layout(set = 0, binding = 2, std430) ACT_QUAL buffer ActBuf { float act[]; };
layout(set = 0, binding = 3, std430) Q8_QUAL buffer Q8Buf { uint q8[]; };
layout(set = 0, binding = 4, std430) KV_QUAL buffer KBuf { uint kv_k[]; };
layout(set = 0, binding = 5, std430) KV_QUAL buffer VBuf { uint kv_v[]; };
layout(set = 0, binding = 6, std430) ST_QUAL buffer StateBuf { float st[]; };
layout(set = 0, binding = 7, std430) SAMPLE_QUAL buffer IoBuf { uint io[]; };
layout(set = 0, binding = 8, std430) readonly buffer RopeBuf { vec2 rope[]; };

float ws_f32(uint word) { return uintBitsToFloat(ws[word]); }

// Hadamard sign for element `i` of a width whose packed bits start at `off`
// (bit set <=> sign -1).
float hsign(uint off, uint i) { return ((ws[off + (i >> 5u)] >> (i & 31u)) & 1u) != 0u ? -1.0 : 1.0; }

vec4 hsign4(uint off, uint i0) {
    uint bits = ws[off + (i0 >> 5u)] >> (i0 & 31u);
    return vec4(((bits & 1u) != 0u) ? -1.0 : 1.0, ((bits & 2u) != 0u) ? -1.0 : 1.0,
                ((bits & 4u) != 0u) ? -1.0 : 1.0, ((bits & 8u) != 0u) ? -1.0 : 1.0);
}

float silu(float x) { return x / (1.0 + exp(-x)); }
vec4 silu4(vec4 x) { return x / (1.0 + exp(-x)); }
