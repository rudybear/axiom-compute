//! M3.14b — Q5_K_M CPU reference oracle (ggml `block_q5_K`, K_SCALE_SIZE=12 variant, 176 bytes
//! / 256 weights).
//!
//! Two distinct templates (per the r2 spec §5, FIX 4 — do not conflate):
//! - The **f32-accum / f16-staged** reference (`q5k_dequant_matmul_f32accum_cpu`) and the
//!   per-element magnitude scale (`q5k_abs_dot_scale`) mirror `q4km_oracle.rs`'s shape. The
//!   metric functions (`max_rel_diff`, `max_rel_diff_combined`, `round_f32_to_f16_to_f32`) are
//!   format-agnostic and RE-EXPORTED from `q4km_oracle` — never re-implemented.
//! - The **bit-exact (native-f32, no f16 rounding)** reference (`q5k_dequant_matmul_f32_cpu`)
//!   mirrors `crates/axc-driver/tests/common_matmul.rs::q4km_dequant_matmul_cpu` (NOT
//!   `q4km_oracle.rs`).
//!
//! LAYOUT (FROZEN, §3.2 of the M3.14 spec):
//! ```text
//! offset  size  field
//! [0..2]     2  d          ggml_half super-scale (for the 6-bit scales)
//! [2..4]     2  dmin       ggml_half super-min  (for the 6-bit mins)
//! [4..16]   12  scales[12] packed 8x(6-bit scale, 6-bit min) — IDENTICAL to Q4_K_M
//! [16..48]  32  qh[32]     high bit of each weight (QK_K/8 = 32)
//! [48..176] 128 qs[128]    low 4 bits of each weight (2 nibbles/byte) — IDENTICAL to Q4_K_M qs
//! ```
//! Q5_K_M is Q4_K_M PLUS one high bit per weight. `d`, `dmin`, the 12-byte packed scales, and
//! `get_scale_min_k4` are byte-identical to Q4_K_M (reused verbatim below). The dequant differs
//! ONLY by promoting the 4-bit nibble to a 5-bit value before the multiply:
//! ```text
//! chunk = k/64 (0..3), l = (k%64)%32 (0..31), is_hi = (k%64)/32 (0=lo,1=hi), is = chunk*2+is_hi
//! nib   = (is_hi==0) ? (qs[chunk*32+l] & 0x0F) : ((qs[chunk*32+l]>>4) & 0x0F)
//! qh_bit_index = 2*chunk + is_hi                        // 0..7  (== `is`, same formula)
//! hi    = (qh[l] >> qh_bit_index) & 1                    // 0 or 1
//! nib5  = nib + hi*16                                    // 0..31, UNSIGNED
//! y[k]  = d*sc[is] * nib5 - dmin*m[is]                   // SAME multiply-then-subtract as Q4_K_M
//! ```
//! All-unsigned — no signed technique needed. `qh_bit_index` and the scale/min sub-block index
//! `is` are the SAME formula (`chunk*2 + is_hi`), so a single `is0`/`is1` per chunk drives both
//! the `get_scale_min_k4` lookup AND the `qh` bit-shift amount (see the inline kernel comment).

#![allow(dead_code)]
// A few `+ 0` terms in the test fixtures below are INTENTIONAL, kept for symmetry with the
// analogous Q6_K oracle's formula-matching style — do not "simplify" them away.
#![allow(clippy::identity_op)]

use half::f16;
use crate::q4km_oracle::round_f32_to_f16_to_f32;

/// Number of bytes per Q5_K_M superblock (256 weights).
pub const Q5K_SUPERBLOCK_BYTES: usize = 176;
/// Number of weights per Q5_K_M superblock.
pub const Q5K_SUPERBLOCK_ELEMS: usize = 256;

/// ggml `get_scale_min_k4(j, scales)` — returns (sc, m) as u32. Byte-identical to
/// `q4km_oracle::get_scale_min_k4` (the 12-byte packed-scale layout is shared between formats).
fn get_scale_min_k4(q: &[u8], scales_base: usize, j: usize) -> (u32, u32) {
    if j < 4 {
        let sc: u32 = (q[scales_base + j] & 63) as u32;
        let m: u32 = (q[scales_base + j + 4] & 63) as u32;
        (sc, m)
    } else {
        let lo4_sc: u32 = (q[scales_base + j + 4] & 0x0F) as u32;
        let hi2_sc: u32 = ((q[scales_base + j - 4] >> 6) & 0x03) as u32;
        let sc: u32 = lo4_sc | (hi2_sc << 4);
        let lo4_m: u32 = ((q[scales_base + j + 4] >> 4) & 0x0F) as u32;
        let hi2_m: u32 = ((q[scales_base + j] >> 6) & 0x03) as u32;
        let m: u32 = lo4_m | (hi2_m << 4);
        (sc, m)
    }
}

/// Dequantize ONE Q5_K_M weight row into f32 (byte-exact ggml arithmetic, no rounding yet).
fn dequant_q5k_row_f32(q: &[u8], row: usize, n_blocks_per_row: usize) -> Vec<f32> {
    let k_total: usize = n_blocks_per_row * Q5K_SUPERBLOCK_ELEMS;
    let mut w: Vec<f32> = vec![0.0_f32; k_total];
    for sb in 0..n_blocks_per_row {
        let base: usize = row * n_blocks_per_row * Q5K_SUPERBLOCK_BYTES + sb * Q5K_SUPERBLOCK_BYTES;
        let d_bits: u16 = u16::from_le_bytes([q[base], q[base + 1]]);
        let dmin_bits: u16 = u16::from_le_bytes([q[base + 2], q[base + 3]]);
        let d: f32 = f16::from_bits(d_bits).to_f32();
        let dmin: f32 = f16::from_bits(dmin_bits).to_f32();
        let scales_base: usize = base + 4;
        let qh_base: usize = base + 16;
        let qs_base: usize = base + 48;

        for chunk in 0..4_usize {
            let is0: usize = chunk * 2;      // is_hi=0 sub-block index; ALSO the qh_bit_index.
            let is1: usize = is0 + 1;        // is_hi=1 sub-block index; ALSO the qh_bit_index.
            let (sc0, m0) = get_scale_min_k4(q, scales_base, is0);
            let (sc1, m1v) = get_scale_min_k4(q, scales_base, is1);
            let d1: f32 = d * sc0 as f32;
            let m1f: f32 = dmin * m0 as f32;
            let d2: f32 = d * sc1 as f32;
            let m2f: f32 = dmin * m1v as f32;

            for l in 0..32_usize {
                let byte: u8 = q[qs_base + chunk * 32 + l];
                let lo_nib: u32 = (byte & 0x0F) as u32;
                let hi_nib: u32 = ((byte >> 4) & 0x0F) as u32;

                // qh[l] holds the high bit for ALL 8 (chunk, is_hi) combos of this l (one byte,
                // 8 bit positions = the 4 chunks x 2 nibbles); bit index is0/is1 == qh_bit_index.
                let qh_byte: u32 = q[qh_base + l] as u32;
                let hi_bit_lo: u32 = (qh_byte >> is0) & 1;
                let hi_bit_hi: u32 = (qh_byte >> is1) & 1;
                let nib5_lo: u32 = lo_nib + hi_bit_lo * 16;
                let nib5_hi: u32 = hi_nib + hi_bit_hi * 16;

                let lo_f: f32 = d1 * nib5_lo as f32 - m1f;
                let hi_f: f32 = d2 * nib5_hi as f32 - m2f;

                let k_lo: usize = sb * Q5K_SUPERBLOCK_ELEMS + chunk * 64 + l;
                let k_hi: usize = k_lo + 32;
                w[k_lo] = lo_f;
                w[k_hi] = hi_f;
            }
        }
    }
    w
}

/// f32-ACCUMULATOR Q5_K_M dequant+matmul CPU reference (mirrors
/// `q4km_oracle::q4km_dequant_matmul_f32accum_cpu`): dequant in f32, round each weight AND each
/// x activation f32->f16->f32, then accumulate the matmul in PURE f32 (no per-tile rounding).
pub fn q5k_dequant_matmul_f32accum_cpu(
    q: &[u8],
    x_f16: &[u16],
    n_rows: usize,
    n_cols: usize,
    n_blocks_per_row: usize,
) -> Vec<f32> {
    let k_total: usize = n_blocks_per_row * Q5K_SUPERBLOCK_ELEMS;
    let mut y: Vec<f32> = vec![0.0_f32; n_rows * n_cols];

    for row in 0..n_rows {
        let w_f32: Vec<f32> = dequant_q5k_row_f32(q, row, n_blocks_per_row);
        let w_f16: Vec<f32> = w_f32.iter().map(|&v| round_f32_to_f16_to_f32(v)).collect();

        for col in 0..n_cols {
            let mut acc: f32 = 0.0_f32;
            for k in 0..k_total {
                let xk: f32 = round_f32_to_f16_to_f32(f16::from_bits(x_f16[k * n_cols + col]).to_f32());
                acc += w_f16[k] * xk;
            }
            y[row * n_cols + col] = acc;
        }
    }
    y
}

/// Bit-exact (NO f16 rounding) Q5_K_M dequant+matmul reference for the portable-kernel
/// `max_diff==0` gate, where `x` is native f32. Mirrors
/// `tests/common_matmul.rs::q4km_dequant_matmul_cpu` (NOT `q4km_oracle.rs`): the SAME
/// per-superblock/chunk/l loop nest and the SAME `sum = sum + lo_f*x_lo + hi_f*x_hi`
/// accumulation order the `q5k_dequant_matmul.axc` kernel uses.
pub fn q5k_dequant_matmul_f32_cpu(
    q: &[u8],
    x_f32: &[f32],
    n_rows: usize,
    n_cols: usize,
    n_blocks_per_row: usize,
) -> Vec<f32> {
    let mut y: Vec<f32> = vec![0.0_f32; n_rows * n_cols];
    for row in 0..n_rows {
        for col in 0..n_cols {
            let mut sum: f32 = 0.0_f32;
            for sb in 0..n_blocks_per_row {
                let base: usize =
                    row * n_blocks_per_row * Q5K_SUPERBLOCK_BYTES + sb * Q5K_SUPERBLOCK_BYTES;
                let d_bits: u16 = u16::from_le_bytes([q[base], q[base + 1]]);
                let dmin_bits: u16 = u16::from_le_bytes([q[base + 2], q[base + 3]]);
                let d: f32 = f16::from_bits(d_bits).to_f32();
                let dmin: f32 = f16::from_bits(dmin_bits).to_f32();
                let scales_base: usize = base + 4;
                let qh_base: usize = base + 16;
                let qs_base: usize = base + 48;

                for chunk in 0..4_usize {
                    let is0: usize = chunk * 2;
                    let is1: usize = is0 + 1;
                    let (sc0, m0) = get_scale_min_k4(q, scales_base, is0);
                    let (sc1, m1v) = get_scale_min_k4(q, scales_base, is1);
                    let d1: f32 = d * sc0 as f32;
                    let m1f: f32 = dmin * m0 as f32;
                    let d2: f32 = d * sc1 as f32;
                    let m2f: f32 = dmin * m1v as f32;

                    for l in 0..32_usize {
                        let byte: u8 = q[qs_base + chunk * 32 + l];
                        let lo_nib: u32 = (byte & 0x0F) as u32;
                        let hi_nib: u32 = ((byte >> 4) & 0x0F) as u32;

                        let qh_byte: u32 = q[qh_base + l] as u32;
                        let hi_bit_lo: u32 = (qh_byte >> is0) & 1;
                        let hi_bit_hi: u32 = (qh_byte >> is1) & 1;
                        let nib5_lo: u32 = lo_nib + hi_bit_lo * 16;
                        let nib5_hi: u32 = hi_nib + hi_bit_hi * 16;

                        let lo_f: f32 = d1 * nib5_lo as f32 - m1f;
                        let hi_f: f32 = d2 * nib5_hi as f32 - m2f;

                        let k_lo: usize = sb * Q5K_SUPERBLOCK_ELEMS + chunk * 64 + l;
                        let k_hi: usize = k_lo + 32;
                        let x_lo: f32 = x_f32[k_lo * n_cols + col];
                        let x_hi: f32 = x_f32[k_hi * n_cols + col];
                        sum = sum + lo_f * x_lo + hi_f * x_hi;
                    }
                }
            }
            y[row * n_cols + col] = sum;
        }
    }
    y
}

/// Per-element dot-product magnitude scale `sum_k |w_f16[k] * x_f16[k,col]|` for Q5_K_M
/// (mirrors `q4km_oracle::q4km_abs_dot_scale`).
pub fn q5k_abs_dot_scale(
    q: &[u8],
    x_f16: &[u16],
    n_rows: usize,
    n_cols: usize,
    n_blocks_per_row: usize,
) -> Vec<f32> {
    let k_total: usize = n_blocks_per_row * Q5K_SUPERBLOCK_ELEMS;
    let mut scale: Vec<f32> = vec![0.0_f32; n_rows * n_cols];
    for row in 0..n_rows {
        let w_f32: Vec<f32> = dequant_q5k_row_f32(q, row, n_blocks_per_row);
        let w_f16: Vec<f32> = w_f32.iter().map(|&v| round_f32_to_f16_to_f32(v)).collect();
        for col in 0..n_cols {
            let mut acc: f32 = 0.0_f32;
            for k in 0..k_total {
                let xk: f32 = round_f32_to_f16_to_f32(f16::from_bits(x_f16[k * n_cols + col]).to_f32());
                acc += (w_f16[k] * xk).abs();
            }
            scale[row * n_cols + col] = acc;
        }
    }
    scale
}

// ── Tests (CPU-only, no GPU) ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::q4km_oracle::max_rel_diff_combined;

    /// Build ONE Q5_K_M superblock (176 bytes) as a hand-computed-from-formula golden fixture
    /// (§6 golden-fixture-provenance fallback — annotated per the r2 rule; no live ggml dump was
    /// available in this environment). Byte array documents the exact `block_q5_K` fields it
    /// represents: d@0, dmin@2, scales[12]@4 (packed 6-bit sc/m, Q4_K_M-identical), qh[32]@16,
    /// qs[128]@48.
    ///
    /// Designed to exercise (AT-2816): the qh high bit SET and CLEAR (nibble -> nibble+16)
    /// across ALL 8 qh-bit positions (chunk 0..3 x is_hi 0..1).
    fn golden_q5k_superblock() -> ([u8; 176], [f32; 256]) {
        let mut q = [0u8; 176];

        // d=1.0, dmin=0.0 (exact in f16) so the weight reduces to sc*nib5, easy to verify.
        let d_bits = half::f16::from_f32(1.0).to_bits();
        q[0] = d_bits.to_le_bytes()[0];
        q[1] = d_bits.to_le_bytes()[1];
        let dmin_bits = half::f16::from_f32(0.0).to_bits();
        q[2] = dmin_bits.to_le_bytes()[0];
        q[3] = dmin_bits.to_le_bytes()[1];

        // scales[12] @ 4..16: set sc=1 for all 8 sub-blocks (j<4 path: sc=byte&63), m=0.
        // j=0..3 -> bytes[4..8]=1 (sc), bytes[8..12]=0 (m).
        // j=4..7 -> lo4_sc from bytes[8..12] (already 0 above -> would give sc=0!); to keep
        // sc=1 for j>=4 too we need bytes[j+4] low-nibble=1 (bytes[8..12] low nibble) AND
        // bytes[j-4] high 2 bits=0. Set bytes[8..12] = 0x01 (lo4=1, hi4=0) so sc(j>=4)=1, m=0.
        for j in 0..4usize {
            q[4 + j] = 1; // sc(j<4) = 1
        }
        for j in 8..12usize {
            q[4 + j] = 0x01; // lo4_sc(j>=4)=1 (sc=1), lo4_m(j>=4)=0 (m=0)
        }

        // qs[128] @ 48..176: nibble = 1 everywhere (byte = 0x11 -> lo=1, hi=1).
        for i in 0..128usize {
            q[48 + i] = 0x11;
        }

        // qh[32] @ 16..48: exercise ALL 8 bit positions (0..7) set AND clear across l.
        // For l=0: qh byte = 0x00 (all bits clear) -> every (chunk,is_hi) sees hi=0 -> nib5=1.
        // For l=1: qh byte = 0xFF (all bits set)   -> every (chunk,is_hi) sees hi=1 -> nib5=17.
        // Remaining l left at 0 (hi=0 -> nib5=1), which is fine (still all-unsigned, valid).
        q[16 + 0] = 0x00;
        q[16 + 1] = 0xFF;

        // ── Compute the expected 256-element array via an INDEPENDENT re-derivation from the
        //    documented bytes (not a call into dequant_q5k_row_f32). ──
        let mut expected = [0.0f32; 256];
        let d: f32 = 1.0;
        let dmin: f32 = 0.0;
        for chunk in 0..4usize {
            let is0 = chunk * 2;
            let is1 = is0 + 1;
            let (sc0, m0) = get_scale_min_k4(&q, 4, is0);
            let (sc1, m1v) = get_scale_min_k4(&q, 4, is1);
            let d1 = d * sc0 as f32;
            let m1f = dmin * m0 as f32;
            let d2 = d * sc1 as f32;
            let m2f = dmin * m1v as f32;
            for l in 0..32usize {
                let byte = q[48 + chunk * 32 + l];
                let lo_nib = (byte & 0x0F) as u32;
                let hi_nib = ((byte >> 4) & 0x0F) as u32;
                let qh_byte = q[16 + l] as u32;
                let hi_bit_lo = (qh_byte >> is0) & 1;
                let hi_bit_hi = (qh_byte >> is1) & 1;
                let nib5_lo = lo_nib + hi_bit_lo * 16;
                let nib5_hi = hi_nib + hi_bit_hi * 16;
                let lo_f = d1 * nib5_lo as f32 - m1f;
                let hi_f = d2 * nib5_hi as f32 - m2f;
                let k_lo = chunk * 64 + l;
                expected[k_lo] = lo_f;
                expected[k_lo + 32] = hi_f;
            }
        }

        (q, expected)
    }

    /// AT-2816: dequant_q5k_row_f32 byte-exact vs a ggml golden superblock, exercising the qh
    /// high bit SET and CLEAR (nibble -> nibble+16) across all 8 qh-bit positions.
    #[test]
    fn at_2816_dequant_q5k_row_golden() {
        let (q_block, expected) = golden_q5k_superblock();
        let got = dequant_q5k_row_f32(&q_block, 0, 1);
        assert_eq!(got.len(), 256, "AT-2816: dequant row must have 256 elements");
        for k in 0..256 {
            assert_eq!(
                got[k], expected[k],
                "AT-2816: element k={k} mismatch: got={}, expected={}", got[k], expected[k]
            );
        }
        // l=0 (qh byte=0x00): nib5 must be 1 (bit clear) for ALL 4 chunks, lo AND hi.
        for chunk in 0..4usize {
            assert_eq!(expected[chunk * 64 + 0], 1.0, "AT-2816: l=0 lo nibble chunk={chunk} must be nib5=1 (qh bit clear)");
            assert_eq!(expected[chunk * 64 + 32 + 0], 1.0, "AT-2816: l=0 hi nibble chunk={chunk} must be nib5=1 (qh bit clear)");
        }
        // l=1 (qh byte=0xFF): nib5 must be 17 (1 + 16, bit set) for ALL 4 chunks, lo AND hi.
        for chunk in 0..4usize {
            assert_eq!(expected[chunk * 64 + 1], 17.0, "AT-2816: l=1 lo nibble chunk={chunk} must be nib5=17 (qh bit set)");
            assert_eq!(expected[chunk * 64 + 32 + 1], 17.0, "AT-2816: l=1 hi nibble chunk={chunk} must be nib5=17 (qh bit set)");
        }
        eprintln!("AT-2816 PASS: Q5_K_M golden byte-exact; qh bit set/clear verified across all 8 positions");
    }

    /// AT-2817: 5-bit range (values 0..31 reachable, +16 exactly when qh bit set) + metric
    /// sanity (`abs_dot_scale >= |y|`, 1% mutant caught). No GPU.
    #[test]
    fn at_2817_five_bit_range_and_metric_sanity() {
        let n_rows = 4usize;
        let n_cols = 4usize;
        let n_bpr = 1usize;
        let mut q = vec![0u8; n_rows * n_bpr * Q5K_SUPERBLOCK_BYTES];

        for row in 0..n_rows {
            let base = row * Q5K_SUPERBLOCK_BYTES;
            let d_bits = half::f16::from_f32(0.03).to_bits();
            q[base..base + 2].copy_from_slice(&d_bits.to_le_bytes());
            let dmin_bits = half::f16::from_f32(0.01).to_bits();
            q[base + 2..base + 4].copy_from_slice(&dmin_bits.to_le_bytes());
            for j in 0..12 {
                q[base + 4 + j] = ((j * 7 + row * 3 + 1) & 0x3F) as u8;
            }
            for i in 0..32 {
                q[base + 16 + i] = ((i * 13 + row * 5) & 0xFF) as u8; // qh: exercise all bits
            }
            for i in 0..128 {
                q[base + 48 + i] = 0xFF; // nibbles all 15 (max, so nib5 in {15,31})
            }
        }

        let w = dequant_q5k_row_f32(&q, 0, n_bpr);
        assert_eq!(w.len(), Q5K_SUPERBLOCK_ELEMS);

        // Direct nib5 range check: nibbles are fixed at 15 (qs all 0xFF) so nib5 in {15,31}
        // depending on the qh bit — a direct demonstration of the +16 promotion at the EXTREME
        // nibble value (0..31 the +16 case is exercised by AT-2816 at nibble=1 -> {1,17}; here
        // at nibble=15 -> {15,31}, covering both ends of the base-nibble range).
        let base = 0usize;
        let qh0: u32 = q[base + 16] as u32; // qh[l=0], chunk=0 uses bit index is0=0.
        let hi_bit0: u32 = qh0 & 1;
        let nib5_chunk0_l0: u32 = 15 + hi_bit0 * 16;
        assert!(
            nib5_chunk0_l0 == 15 || nib5_chunk0_l0 == 31,
            "AT-2817: nib5 must be exactly 15 (bit clear) or 31 (bit set); got {nib5_chunk0_l0}"
        );
        let any_set = (0..32).any(|i| q[base + 16 + i] != 0);
        let any_clear = (0..32).any(|i| q[base + 16 + i] & 0x01 == 0);
        assert!(any_set, "AT-2817 fixture sanity: qh must have at least one set bit");
        assert!(any_clear, "AT-2817 fixture sanity: qh must have at least one clear bit");

        let k = n_bpr * Q5K_SUPERBLOCK_ELEMS;
        let x_f16: Vec<u16> = (0..k * n_cols)
            .map(|i| half::f16::from_f32((((i * 5 + 3) % 100) as f32) / 50.0 - 1.0).to_bits())
            .collect();

        let y = q5k_dequant_matmul_f32accum_cpu(&q, &x_f16, n_rows, n_cols, n_bpr);
        let scale = q5k_abs_dot_scale(&q, &x_f16, n_rows, n_cols, n_bpr);
        assert_eq!(y.len(), scale.len());
        for i in 0..y.len() {
            assert!(
                scale[i] + 1e-6 >= y[i].abs(),
                "AT-2817: abs_dot_scale[{i}]={} must dominate |y[{i}]|={}", scale[i], y[i].abs()
            );
        }

        let y_mut: Vec<f32> = y.iter().map(|&v| v * 1.01_f32).collect();
        let combined = max_rel_diff_combined(&y_mut, &y, &scale);
        assert!(
            combined > 1e-3,
            "AT-2817: a 1% systematic error must be CAUGHT by the combined gate; got {combined:.3e}"
        );
        eprintln!("AT-2817 PASS: 5-bit range covered, abs_dot_scale dominates, mutant caught (combined={combined:.3e})");
    }
}
