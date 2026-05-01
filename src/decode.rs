use std::sync::OnceLock;

/// Inverse of GPT-2's bytes_to_unicode map: each codepoint in a vocab token
/// maps back to its raw byte. Special tokens (e.g. `<|im_start|>`) fall through
/// as their UTF-8 encoding.
pub fn decode_token_bytes(s: &str) -> Vec<u8> {
    static INV: OnceLock<[Option<u8>; 0x180]> = OnceLock::new();
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
        let mut inv: [Option<u8>; 0x180] = [None; 0x180];
        for (b, c) in bs.iter().zip(cs.iter()) {
            if (*c as usize) < inv.len() {
                inv[*c as usize] = Some(*b as u8);
            }
        }
        inv
    });
    let mut out = Vec::with_capacity(s.len());
    for ch in s.chars() {
        let cp = ch as usize;
        match inv.get(cp).copied().flatten() {
            Some(b) => out.push(b),
            None => {
                let mut buf = [0u8; 4];
                out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_passthrough() {
        assert_eq!(decode_token_bytes("!"), b"!");
        assert_eq!(decode_token_bytes("Hello"), b"Hello");
        assert_eq!(decode_token_bytes("~"), b"~");
    }

    #[test]
    fn gpt2_space_mapping() {
        // Ġ (U+0120) is the GPT-2 encoding of space (0x20).
        assert_eq!(decode_token_bytes("\u{0120}"), b" ");
        assert_eq!(decode_token_bytes("\u{0120}world"), b" world");
    }

    #[test]
    fn gpt2_newline_mapping() {
        // Ċ (U+010A) is the GPT-2 encoding of newline (0x0A).
        assert_eq!(decode_token_bytes("\u{010A}"), b"\n");
    }

    #[test]
    fn chatml_specials_passthrough() {
        // Special-token strings contain only printable ASCII, which falls
        // through the table as-is.
        let s = "<|im_start|>";
        assert_eq!(decode_token_bytes(s), s.as_bytes());
        let s2 = "<|endoftext|>";
        assert_eq!(decode_token_bytes(s2), s2.as_bytes());
    }

    #[test]
    fn inverse_round_trip_byte_range() {
        // Mirror the GPT-2 bytes_to_unicode forward map and verify every byte
        // survives a round-trip through decode_token_bytes.
        fn encode_byte(b: u8) -> char {
            let printable_single: Vec<u8> = (b'!' ..= b'~')
                .chain(0xa1..=0xac)
                .chain(0xae..=0xff)
                .collect();
            if let Some(pos) = printable_single.iter().position(|&x| x == b) {
                printable_single[pos] as char
            } else {
                // Non-printable byte: mapped to 256 + offset in the complement set.
                let complement_singles: Vec<u8> = (0u8..=255)
                    .filter(|x| !printable_single.contains(x))
                    .collect();
                let offset = complement_singles.iter().position(|&x| x == b).unwrap();
                char::from_u32(256 + offset as u32).unwrap()
            }
        }
        for b in 0u8..=255 {
            let encoded = encode_byte(b).to_string();
            let decoded = decode_token_bytes(&encoded);
            assert_eq!(decoded, vec![b], "round-trip failed for byte 0x{b:02X}");
        }
    }
}
