//! CPU reference implementations of the `PTQ1_0` codec and the Prism Hadamard
//! transform. They mirror the GPU kernels and are used by unit tests and the
//! hidden kernel-check hooks.

#![allow(dead_code, reason = "used by tests and bench-internals hooks")]

use alloc::vec::Vec;

/// Bytes per `PTQ1_0` block (128 weights).
pub const PTQ1_0_BLOCK_BYTES: usize = 28;
/// Weights per `PTQ1_0` block.
pub const PTQ1_0_BLOCK_ELEMS: usize = 128;

/// Decode one `PTQ1_0` block into 128 f32 weights, following the fork's
/// `dequantize_row_ptq1_0` traversal (qs in 16- then 8-byte stages, then qh).
#[must_use]
pub fn ptq1_0_dequant_block(b: &[u8; PTQ1_0_BLOCK_BYTES]) -> [f32; PTQ1_0_BLOCK_ELEMS] {
    const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];
    let d = half::f16::from_le_bytes([b[26], b[27]]).to_f32();
    let mut out = [0f32; PTQ1_0_BLOCK_ELEMS];
    let mut y = 0usize;
    let trit = |q: u8| -> f32 { f32::from(((u16::from(q) * 3) >> 8) as u8) - 1.0 };
    let mut j = 0usize;
    for c in [32usize, 16, 8] {
        while j + c <= 24 {
            for p in POW3.iter().take(5) {
                for m in 0..c {
                    out[y] = trit(b[j + m].wrapping_mul(*p)) * d;
                    y += 1;
                }
            }
            j += c;
        }
    }
    for p in POW3.iter().take(4) {
        for h in 0..2 {
            out[y] = trit(b[24 + h].wrapping_mul(*p)) * d;
            y += 1;
        }
    }
    out
}

/// Unscaled ternary value of element `e` of a block via the per-element
/// mapping the shaders use (`lib/ptq1_0.glsl::ptq1_0_trit`).
#[must_use]
pub fn ptq1_0_trit(b: &[u8; PTQ1_0_BLOCK_BYTES], e: usize) -> i32 {
    let (byte, n) = if e < 80 {
        (e & 15, e >> 4)
    } else if e < 120 {
        (16 + ((e - 80) & 7), (e - 80) >> 3)
    } else {
        (24 + ((e - 120) & 1), (e - 120) >> 1)
    };
    let mut v = u32::from(b[byte]);
    for _ in 0..n {
        v = (v * 3) & 0xFF;
    }
    i32::from(((v * 3) >> 8) as u8) - 1
}

/// Dequantize a full `PTQ1_0` row of `nb` blocks.
#[must_use]
pub fn ptq1_0_dequant_row(row: &[u8]) -> Vec<f32> {
    row.chunks_exact(PTQ1_0_BLOCK_BYTES)
        .flat_map(|c| {
            let mut b = [0u8; PTQ1_0_BLOCK_BYTES];
            b.copy_from_slice(c);
            ptq1_0_dequant_block(&b)
        })
        .collect()
}

/// In-place normalized Walsh-Hadamard transform of each consecutive
/// `block`-element chunk of `x` (natural/Sylvester order, scale `1/√block`).
pub fn fwht_blockwise(x: &mut [f32], block: usize) {
    debug_assert!(block.is_power_of_two() && x.len().is_multiple_of(block));
    let scale = 1.0 / (block as f32).sqrt();
    for chunk in x.chunks_exact_mut(block) {
        let mut h = 1;
        while h < block {
            for i in (0..block).step_by(2 * h) {
                for j in i..i + h {
                    let (a, b) = (chunk[j], chunk[j + h]);
                    chunk[j] = a + b;
                    chunk[j + h] = a - b;
                }
            }
            h *= 2;
        }
        for v in chunk.iter_mut() {
            *v *= scale;
        }
    }
}

/// Unpack the Hadamard sign bits written by `extract.py` (bit set ⇔ −1).
#[must_use]
pub fn unpack_signs(words: &[u32], width: usize) -> Vec<f32> {
    (0..width)
        .map(|i| {
            if (words[i / 32] >> (i % 32)) & 1 != 0 {
                -1.0
            } else {
                1.0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Port of the fork's `quantize_row_ptq1_0_ref` for one block of ternary
    /// values in {-1, 0, 1} with scale `d`.
    fn quantize_block(x: &[i32; 128], d: f32) -> [u8; 28] {
        let mut out = [0u8; 28];
        let mut xi = 0usize;
        let mut j = 0usize;
        for c in [32usize, 16, 8] {
            while j + c <= 24 {
                for m in 0..c {
                    let mut q: u16 = 0;
                    for n in 0..5 {
                        q = q * 3 + u16::try_from(x[xi + m + n * c] + 1).unwrap();
                    }
                    out[j + m] = (q * 256).div_ceil(243) as u8;
                }
                xi += 5 * c;
                j += c;
            }
        }
        for h in 0..2 {
            let mut q: u16 = 0;
            for m in 0..4 {
                q = q * 3 + u16::try_from(x[xi + h + m * 2] + 1).unwrap();
            }
            q *= 3;
            out[24 + h] = (q * 256).div_ceil(243) as u8;
        }
        let db = half::f16::from_f32(d).to_le_bytes();
        out[26] = db[0];
        out[27] = db[1];
        out
    }

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *seed >> 33
    }

    #[test]
    fn dequant_roundtrips_quantizer_and_matches_element_map() {
        let mut seed = 7u64;
        for _ in 0..200 {
            let mut x = [0i32; 128];
            for v in &mut x {
                *v = (lcg(&mut seed) % 3) as i32 - 1;
            }
            let blk = quantize_block(&x, 0.5);
            let deq = ptq1_0_dequant_block(&blk);
            for e in 0..128 {
                assert_eq!(deq[e], x[e] as f32 * 0.5, "element {e}");
                assert_eq!(ptq1_0_trit(&blk, e), x[e], "element map {e}");
            }
        }
    }

    #[test]
    fn fwht_matches_parity_matrix_and_is_involutive() {
        let n = 64usize;
        let mut seed = 3u64;
        let x: Vec<f32> = (0..n)
            .map(|_| (lcg(&mut seed) % 1000) as f32 / 500.0 - 1.0)
            .collect();
        let mut y = x.clone();
        fwht_blockwise(&mut y, n);
        let s = 1.0 / (n as f32).sqrt();
        for (r, &yr) in y.iter().enumerate() {
            let mut acc = 0.0f32;
            for (c, &xc) in x.iter().enumerate() {
                let sign = if (r & c).count_ones() % 2 == 1 {
                    -1.0
                } else {
                    1.0
                };
                acc = (sign * s).mul_add(xc, acc);
            }
            assert!((acc - yr).abs() < 1e-4, "row {r}");
        }
        fwht_blockwise(&mut y, n);
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn sign_unpacking() {
        let words = [0b101u32, 0x8000_0000];
        let s = unpack_signs(&words, 64);
        assert_eq!(&s[..4], &[-1.0, 1.0, -1.0, 1.0]);
        assert_eq!(s[63], -1.0);
        assert_eq!(s[62], 1.0);
    }
}
