// Q8_0 KV-cache load helpers shared by attention_split and attention_prefill_tiled.
// Requires the caller to have bound k_cache / v_cache and defined push-constant
// fields: kv_stride, k_d_word_offset, k_qs_byte_offset, v_d_word_offset, v_qs_byte_offset.

float load_k(uint t, uint e_local) {
    uint elem_idx = t * p.kv_stride + e_local;
    uint block_idx = elem_idx >> 5u;
    float scale = uintBitsToFloat(k_cache[p.k_d_word_offset + block_idx]);
    uint qs_byte_idx = p.k_qs_byte_offset + elem_idx;
    uint qs_word = k_cache[qs_byte_idx >> 2u];
    uint shift = (qs_byte_idx & 3u) << 3u;
    int qs_signed = extract_bits_signed(int(qs_word), shift, 8u);
    return scale * float(qs_signed);
}

float load_v(uint t, uint e_local) {
    uint elem_idx = t * p.kv_stride + e_local;
    uint block_idx = elem_idx >> 5u;
    float scale = uintBitsToFloat(v_cache[p.v_d_word_offset + block_idx]);
    uint qs_byte_idx = p.v_qs_byte_offset + elem_idx;
    uint qs_word = v_cache[qs_byte_idx >> 2u];
    uint shift = (qs_byte_idx & 3u) << 3u;
    int qs_signed = extract_bits_signed(int(qs_word), shift, 8u);
    return scale * float(qs_signed);
}
