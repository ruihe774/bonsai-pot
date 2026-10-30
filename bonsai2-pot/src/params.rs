//! Push-constant (immediate) structs, one per kernel. Field order and types
//! must match the `Params` block of the corresponding `src/shaders/*.comp`.
//! Offsets are element offsets into their buffer (f32 elements for `act`,
//! u32 words for the weight / q8 / kv buffers) unless stated otherwise.

use bytemuck::{Pod, Zeroable};

/// Size in bytes of the immediate range every pipeline layout declares.
pub const IMMEDIATE_BYTES: u32 = 64;

macro_rules! params {
    ($(#[$m:meta])* $name:ident { $($field:ident : $ty:ty),* $(,)? }) => {
        $(#[$m])*
        #[repr(C)]
        #[derive(Pod, Zeroable, Copy, Clone, Default, Debug)]
        pub struct $name {
            $(pub $field: $ty),*
        }
    };
}

params!(EmbedParams {
    w_word: u32,
    nb_row: u32,
    x_off: u32,
    sample_off: u32,
    signs_off: u32,
    n_embd: u32
});

params!(NormRotParams {
    k: u32,
    in_off: u32,
    in_stride: u32,
    norm_off: u32,
    signs_off: u32,
    xn_off: u32,
    q8_qs: u32,
    q8_d: u32,
    q8_s: u32,
    eps: f32,
    flags: u32,
});

params!(MatvecParams {
    nb: u32,
    q8_qs: u32,
    q8_d: u32,
    q8_s: u32,
    seg1: u32,
    seg2: u32,
    w0: u32,
    w1: u32,
    w2: u32,
    out0: u32,
    out1: u32,
    out2: u32,
    flags: u32,
});

params!(MatmulParams {
    nb: u32,
    m_tokens: u32,
    q8_qs: u32,
    q8_d: u32,
    q8_s: u32,
    seg1: u32,
    seg2: u32,
    w0: u32,
    w1: u32,
    w2: u32,
    out0: u32,
    out1: u32,
    out2: u32,
    total_rows: u32,
    flags: u32,
});

params!(AttnPrefillParams {
    qg_off: u32,
    o_off: u32,
    n_head: u32,
    n_kv_head: u32,
    m_tokens: u32,
    pos_base: u32,
    kv_dim: u32,
    kv_d_word: u32,
    kv_qs_byte: u32,
    q_dim: u32,
    scale: f32,
});

params!(GdnGatesParams {
    xn_off: u32,
    alpha_off: u32,
    beta_off: u32,
    dt_off: u32,
    a_off: u32,
    out_off: u32,
    n_embd: u32,
    n_heads: u32,
});

params!(GdnConvParams {
    m_tokens: u32,
    in_off: u32,
    in_stride: u32,
    out_off: u32,
    w_off: u32,
    state_off: u32,
    n_qk_heads: u32,
    channels: u32,
    eps: f32,
    fresh: u32,
});

params!(GdnRecurrenceParams {
    m_tokens: u32,
    qkv_off: u32,
    stride: u32,
    k_base: u32,
    v_base: u32,
    gates_off: u32,
    o_off: u32,
    state_off: u32,
    n_k_heads: u32,
    n_v_heads: u32,
    scale: f32,
    fresh: u32,
});

params!(GdnOutParams {
    o_off: u32,
    z_off: u32,
    norm_off: u32,
    signs_off: u32,
    inner: u32,
    n_k_heads: u32,
    rep: u32,
    q8_qs: u32,
    q8_d: u32,
    q8_s: u32,
    eps: f32,
});

params!(AttnQkParams {
    qg_off: u32,
    k_off: u32,
    v_off: u32,
    qnorm_off: u32,
    knorm_off: u32,
    pos_base: u32,
    n_head: u32,
    n_kv_head: u32,
    rope_half: u32,
    eps: f32,
    kv_d_word: u32,
    kv_qs_byte: u32,
    kv_dim: u32,
});

params!(AttnSplitParams {
    q_off: u32,
    n_head: u32,
    n_kv_head: u32,
    n_kv: u32,
    chunk_len: u32,
    n_chunks: u32,
    kv_dim: u32,
    kv_d_word: u32,
    kv_qs_byte: u32,
    part_off: u32,
    scale: f32,
});

params!(AttnOutParams {
    mode: u32,
    part_off: u32,
    n_chunks: u32,
    o_off: u32,
    qg_off: u32,
    signs_off: u32,
    q_dim: u32,
    q8_qs: u32,
    q8_d: u32,
    q8_s: u32,
});

params!(SiluMulParams {
    gate_off: u32,
    up_off: u32,
    k: u32,
    signs_off: u32,
    q8_qs: u32,
    q8_d: u32,
    q8_s: u32
});

params!(TopKPartialParams {
    n: u32,
    in_offset: u32,
    partials_off: u32,
    n_per_wg: u32
});

params!(TopKMergeParams {
    partials_off: u32,
    num_partials: u32,
    out_offset: u32,
    k: u32
});

#[cfg(test)]
mod tests {
    use core::mem::size_of;

    use super::*;

    #[test]
    fn params_fit_immediate_range() {
        let sizes = [
            size_of::<EmbedParams>(),
            size_of::<NormRotParams>(),
            size_of::<MatvecParams>(),
            size_of::<MatmulParams>(),
            size_of::<AttnPrefillParams>(),
            size_of::<GdnGatesParams>(),
            size_of::<GdnConvParams>(),
            size_of::<GdnRecurrenceParams>(),
            size_of::<GdnOutParams>(),
            size_of::<AttnQkParams>(),
            size_of::<AttnSplitParams>(),
            size_of::<AttnOutParams>(),
            size_of::<SiluMulParams>(),
            size_of::<TopKPartialParams>(),
            size_of::<TopKMergeParams>(),
        ];
        for s in sizes {
            assert!(
                s <= IMMEDIATE_BYTES as usize,
                "params struct of {s} bytes exceeds the immediate range"
            );
        }
    }
}
