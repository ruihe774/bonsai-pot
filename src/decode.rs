use std::sync::OnceLock;

/// Inverse of GPT-2's bytes_to_unicode map: each codepoint in a vocab token
/// maps back to its raw byte. Special tokens (e.g. `<|im_start|>`) fall through
/// as their UTF-8 encoding.
pub fn decode_token_bytes(s: &str) -> Vec<u8> {
    static INV: OnceLock<[u8; 0x180]> = OnceLock::new();
    let inv = INV.get_or_init(|| {
        let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32)
            .chain(0xa1..=0xac)
            .chain(0xae..=0xff)
            .collect();
        let mut cs = bs.clone();
        let mut n = 0u32;
        for b in 0..256u32 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }
        let mut inv = [0u8; 0x180];
        for (b, c) in bs.iter().zip(cs.iter()) {
            if (*c as usize) < inv.len() {
                inv[*c as usize] = *b as u8;
            }
        }
        inv
    });
    let mut out = Vec::with_capacity(s.len());
    for ch in s.chars() {
        let cp = ch as usize;
        if cp < inv.len() && (inv[cp] != 0 || cp == 0) {
            out.push(inv[cp]);
        } else {
            let mut buf = [0u8; 4];
            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
        }
    }
    out
}
