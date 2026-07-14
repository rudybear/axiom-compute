//! M3.14a — Q6_K CPU reference oracle (ggml `block_q6_K`, 210 bytes / 256 weights).
//!
//! Two distinct templates (per the r2 spec §5, FIX 4 — do not conflate):
//! - The **f32-accum / f16-staged** reference (`q6k_dequant_matmul_f32accum_cpu`) and the
//!   per-element magnitude scale (`q6k_abs_dot_scale`) mirror
//!   `crates/axc-driver/src/q4km_oracle.rs`'s shape (dequant in f32, round weight+x
//!   f32->f16->f32, accumulate in PURE f32 — the tensor-core-comparable path). The metric
//!   functions themselves (`max_rel_diff`, `max_rel_diff_combined`, `round_f32_to_f16_to_f32`)
//!   are format-agnostic and are RE-EXPORTED from `q4km_oracle` — never re-implemented (single
//!   source of truth for the frozen 1e-3 gate).
//! - The **bit-exact (native-f32, no f16 rounding)** reference (`q6k_dequant_matmul_f32_cpu`)
//!   mirrors `crates/axc-driver/tests/common_matmul.rs::q4km_dequant_matmul_cpu` — the
//!   portable-kernel `max_diff==0` gate template — NOT `q4km_oracle.rs`.
//!
//! LAYOUT (FROZEN, §3.1 of the M3.14 spec — verified against canonical ggml `block_q6_K`):
//! ```text
//! offset  size  field
//! [0..128]  128  ql[128]    low 4 bits of each weight   (QK_K/2 = 128)
//! [128..192] 64  qh[64]     high 2 bits of each weight  (QK_K/4 = 64)
//! [192..208] 16  scales[16] int8 (SIGNED) per-16-element scales (QK_K/16 = 16)
//! [208..210]  2  d          ggml_half super-scale (IEEE-754 binary16 LE)
//! ```
//! NOTE the field ORDER differs from Q4_K_M: `d` is the LAST 2 bytes (offset 208), and there is
//! NO `dmin` (Q6_K is a symmetric, zero-offset quant — the -32 bias is on the QUANT, not a
//! per-block min).
//!
//! DEQUANT (ggml, byte-exact). The superblock is processed in 2 halves of 128 weights; within
//! each half `h in {0,1}`, for `l in 0..32`, `is = l/16`:
//! ```text
//! q1 = ((ql[64h+l]      & 0x0F) | (((qh[32h+l] >> 0) & 3) << 4)) - 32
//! q2 = ((ql[64h+l + 32] & 0x0F) | (((qh[32h+l] >> 2) & 3) << 4)) - 32
//! q3 = ((ql[64h+l]      >>  4)  | (((qh[32h+l] >> 4) & 3) << 4)) - 32
//! q4 = ((ql[64h+l + 32] >>  4)  | (((qh[32h+l] >> 6) & 3) << 4)) - 32
//! y[128h+l+ 0] = d * sc[8h+is+0] * q1
//! y[128h+l+32] = d * sc[8h+is+2] * q2
//! y[128h+l+64] = d * sc[8h+is+4] * q3
//! y[128h+l+96] = d * sc[8h+is+6] * q4
//! ```
//! `sc[.]` is the SIGNED int8 scale. `q*` in [-32,31]; `sc` in [-128,127].
//!
//! SIGN RECONSTRUCTION: `q[byte] as i8 as f32` is used throughout below. This is a bitwise
//! reinterpretation (Rust `u8 as i8` is NOT a lossy numeric cast — it is bit-preserving), and is
//! bit-identical to the pure-source kernel's technique
//! (`f32_from_u32(u); if u >= 128 { v -= 256.0; }`) for every `u in 0..=255`: for `u < 128` both
//! give `u as f32`; for `u >= 128` both give `(u as f32) - 256.0`. This is the pessimistic
//! review's bit-exact-safety proof (M3.14 spec §5) applied directly.
//!
//! BIT-EXACTNESS-BY-CONSTRUCTION (FROZEN): the bit-exact oracle below uses the SAME
//! `(d * sc_f) * q_f` grouping and the SAME q1/q2/q3/q4 accumulation order as the
//! `q6k_dequant_matmul.axc` kernel (see its inline `sc[8h+is+{0,2,4,6}]` comment), so
//! `max_diff==0` holds by construction on an exact-integer fixture (AT-2814).

#![allow(dead_code)]
// The `>> 0` / `+ 0` terms below are INTENTIONAL: they mirror the §3.1 formula's literal
// `qh[...] >> 0` / `sc[8h+is+0]` bit-pair/index positions verbatim so the code stays
// line-by-line auditable against the spec (the MANDATORY discipline for the most
// transcription-error-prone part of this milestone) — do not "simplify" them away.
#![allow(clippy::identity_op)]

use half::f16;
use crate::q4km_oracle::round_f32_to_f16_to_f32;

/// Number of bytes per Q6_K superblock (256 weights).
pub const Q6K_SUPERBLOCK_BYTES: usize = 210;
/// Number of weights per Q6_K superblock.
pub const Q6K_SUPERBLOCK_ELEMS: usize = 256;

/// Dequantize ONE Q6_K weight row into f32 (byte-exact ggml arithmetic, no rounding yet).
///
/// `row` indexes the weight matrix; the returned vector has `n_blocks_per_row * 256` elements,
/// where element `k` is the dequantized weight at contraction index `k`.
fn dequant_q6k_row_f32(q: &[u8], row: usize, n_blocks_per_row: usize) -> Vec<f32> {
    let k_total: usize = n_blocks_per_row * Q6K_SUPERBLOCK_ELEMS;
    let mut w: Vec<f32> = vec![0.0_f32; k_total];
    for sb in 0..n_blocks_per_row {
        let base: usize = row * n_blocks_per_row * Q6K_SUPERBLOCK_BYTES + sb * Q6K_SUPERBLOCK_BYTES;
        let ql_base: usize = base;         // [0..128)
        let qh_base: usize = base + 128;   // [128..192)
        let sc_base: usize = base + 192;   // [192..208)
        // d is the LAST 2 bytes (offset 208) — Q6_K field order differs from Q4_K_M. No dmin.
        let d_bits: u16 = u16::from_le_bytes([q[base + 208], q[base + 209]]);
        let d: f32 = f16::from_bits(d_bits).to_f32();

        for h in 0..2_usize {
            for l in 0..32_usize {
                // is = l/16 selects which HALF of the 2 scale-pairs this l falls into.
                let is: usize = l / 16;

                let ql1: u32 = q[ql_base + 64 * h + l] as u32;
                let ql2: u32 = q[ql_base + 64 * h + l + 32] as u32;
                let qh_byte: u32 = q[qh_base + 32 * h + l] as u32;

                // q1..q4: low 4 bits from ql + high 2 bits from the qh bit-pair, minus the -32 bias.
                let q1_u: u32 = (ql1 & 0x0F) | (((qh_byte >> 0) & 3) << 4);
                let q2_u: u32 = (ql2 & 0x0F) | (((qh_byte >> 2) & 3) << 4);
                let q3_u: u32 = (ql1 >> 4) | (((qh_byte >> 4) & 3) << 4);
                let q4_u: u32 = (ql2 >> 4) | (((qh_byte >> 6) & 3) << 4);

                let q1_f: f32 = q1_u as f32 - 32.0;
                let q2_f: f32 = q2_u as f32 - 32.0;
                let q3_f: f32 = q3_u as f32 - 32.0;
                let q4_f: f32 = q4_u as f32 - 32.0;

                // sc[8h+is+{0,2,4,6}] — the four scale entries this l selects (§3.1 formula).
                let sc1: f32 = q[sc_base + 8 * h + is + 0] as i8 as f32;
                let sc2: f32 = q[sc_base + 8 * h + is + 2] as i8 as f32;
                let sc3: f32 = q[sc_base + 8 * h + is + 4] as i8 as f32;
                let sc4: f32 = q[sc_base + 8 * h + is + 6] as i8 as f32;

                // Kernel-matching associativity: (d * sc_f) * q_f.
                let w1: f32 = (d * sc1) * q1_f;
                let w2: f32 = (d * sc2) * q2_f;
                let w3: f32 = (d * sc3) * q3_f;
                let w4: f32 = (d * sc4) * q4_f;

                let k_base: usize = sb * Q6K_SUPERBLOCK_ELEMS + 128 * h + l;
                w[k_base] = w1;
                w[k_base + 32] = w2;
                w[k_base + 64] = w3;
                w[k_base + 96] = w4;
            }
        }
    }
    w
}

/// f32-ACCUMULATOR Q6_K dequant+matmul CPU reference (mirrors
/// `q4km_oracle::q4km_dequant_matmul_f32accum_cpu`): dequant in f32, round each weight AND each
/// x activation f32->f16->f32, then accumulate the matmul in PURE f32 (no per-tile rounding).
///
/// `x_f16` holds the activation f16 bit patterns, row-major: x[k*n_cols + col].
pub fn q6k_dequant_matmul_f32accum_cpu(
    q: &[u8],
    x_f16: &[u16],
    n_rows: usize,
    n_cols: usize,
    n_blocks_per_row: usize,
) -> Vec<f32> {
    let k_total: usize = n_blocks_per_row * Q6K_SUPERBLOCK_ELEMS;
    let mut y: Vec<f32> = vec![0.0_f32; n_rows * n_cols];

    for row in 0..n_rows {
        let w_f32: Vec<f32> = dequant_q6k_row_f32(q, row, n_blocks_per_row);
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

/// Bit-exact (NO f16 rounding) Q6_K dequant+matmul reference for the portable-kernel
/// `max_diff==0` gate, where `x` is native f32. Mirrors
/// `tests/common_matmul.rs::q4km_dequant_matmul_cpu` (NOT `q4km_oracle.rs`): the SAME
/// per-superblock loop nest, `(d * sc_f) * q_f` weight associativity, and left-to-right
/// `sum = sum + w1*x1 + w2*x2 + w3*x3 + w4*x4` accumulation order the `q6k_dequant_matmul.axc`
/// kernel uses, so `max_diff==0` holds by construction on an exact-integer fixture (AT-2814).
pub fn q6k_dequant_matmul_f32_cpu(
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
                    row * n_blocks_per_row * Q6K_SUPERBLOCK_BYTES + sb * Q6K_SUPERBLOCK_BYTES;
                let ql_base: usize = base;
                let qh_base: usize = base + 128;
                let sc_base: usize = base + 192;
                let d_bits: u16 = u16::from_le_bytes([q[base + 208], q[base + 209]]);
                let d: f32 = f16::from_bits(d_bits).to_f32();

                for h in 0..2_usize {
                    for l in 0..32_usize {
                        let is: usize = l / 16;

                        let ql1: u32 = q[ql_base + 64 * h + l] as u32;
                        let ql2: u32 = q[ql_base + 64 * h + l + 32] as u32;
                        let qh_byte: u32 = q[qh_base + 32 * h + l] as u32;

                        let q1_u: u32 = (ql1 & 0x0F) | (((qh_byte >> 0) & 3) << 4);
                        let q2_u: u32 = (ql2 & 0x0F) | (((qh_byte >> 2) & 3) << 4);
                        let q3_u: u32 = (ql1 >> 4) | (((qh_byte >> 4) & 3) << 4);
                        let q4_u: u32 = (ql2 >> 4) | (((qh_byte >> 6) & 3) << 4);

                        let q1_f: f32 = q1_u as f32 - 32.0;
                        let q2_f: f32 = q2_u as f32 - 32.0;
                        let q3_f: f32 = q3_u as f32 - 32.0;
                        let q4_f: f32 = q4_u as f32 - 32.0;

                        // sc[8h+is+{0,2,4,6}] — see the dequant_q6k_row_f32 comment above.
                        let sc1: f32 = q[sc_base + 8 * h + is + 0] as i8 as f32;
                        let sc2: f32 = q[sc_base + 8 * h + is + 2] as i8 as f32;
                        let sc3: f32 = q[sc_base + 8 * h + is + 4] as i8 as f32;
                        let sc4: f32 = q[sc_base + 8 * h + is + 6] as i8 as f32;

                        let w1: f32 = (d * sc1) * q1_f;
                        let w2: f32 = (d * sc2) * q2_f;
                        let w3: f32 = (d * sc3) * q3_f;
                        let w4: f32 = (d * sc4) * q4_f;

                        let k1: usize = sb * Q6K_SUPERBLOCK_ELEMS + 128 * h + l;
                        let k2: usize = k1 + 32;
                        let k3: usize = k1 + 64;
                        let k4: usize = k1 + 96;

                        let x1: f32 = x_f32[k1 * n_cols + col];
                        let x2: f32 = x_f32[k2 * n_cols + col];
                        let x3: f32 = x_f32[k3 * n_cols + col];
                        let x4: f32 = x_f32[k4 * n_cols + col];

                        // Left-to-right accumulation, matching the kernel's `sum = sum + ...`.
                        sum = sum + w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4;
                    }
                }
            }
            y[row * n_cols + col] = sum;
        }
    }
    y
}

/// Per-element dot-product magnitude scale `sum_k |w_f16[k] * x_f16[k,col]|` for Q6_K
/// (mirrors `q4km_oracle::q4km_abs_dot_scale`), the natural error scale used by
/// `max_rel_diff_combined` for the condition-aware gate.
pub fn q6k_abs_dot_scale(
    q: &[u8],
    x_f16: &[u16],
    n_rows: usize,
    n_cols: usize,
    n_blocks_per_row: usize,
) -> Vec<f32> {
    let k_total: usize = n_blocks_per_row * Q6K_SUPERBLOCK_ELEMS;
    let mut scale: Vec<f32> = vec![0.0_f32; n_rows * n_cols];
    for row in 0..n_rows {
        let w_f32: Vec<f32> = dequant_q6k_row_f32(q, row, n_blocks_per_row);
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

    /// Build ONE Q6_K superblock (210 bytes) as a hand-computed-from-formula golden fixture
    /// (§6 golden-fixture-provenance fallback — annotated per the r2 rule; no live ggml dump
    /// was available in this environment). Byte array documents the exact `block_q6_K` fields
    /// it represents: ql[128]@0, qh[64]@128, scales[16]@192 (signed int8), d@208 (f16 LE).
    ///
    /// Designed to exercise (AT-2811, per the r2 FIX-1 full-coverage requirement):
    ///  (i)   all FOUR quant-groups q1/q2/q3/q4 (distinct ql/qh bit patterns per group),
    ///  (ii)  BOTH halves h=0 and h=1 (distinct qh bytes per half),
    ///  (iii) the is=l/16 boundary (l=15 selects sc[.+is=0], l=16 selects sc[.+is=1]),
    ///  (iv)  at least one NEGATIVE int8 scale (sc[3] = -5, two's complement 0xFB),
    ///  (v)   quant extremes: l=0 -> q1_u=0 (q1_f=-32); a qh/ql combo giving q_u=63 (q_f=+31).
    fn golden_q6k_superblock() -> ([u8; 210], [f32; 256]) {
        let mut q = [0u8; 210];

        // scales[16] @ 192..208 — signed int8. Mix of positive/negative/extreme values,
        // distinct per index so a wrong sc[.] index is detectable.
        // sc[0..8] used for h=0 (is=0 -> sc[0,2,4,6]; is=1 -> sc[1,3,5,7]).
        // sc[8..16] used for h=1 (is=0 -> sc[8,10,12,14]; is=1 -> sc[9,11,13,15]).
        // sc[2] = -5 is the NEGATIVE entry (used by q3 at h=0, is=0).
        let sc: [i8; 16] = [
            2, 3, -5, 4, 1, -1, 6, -3, // h=0
            5, -2, 7, -4, 3, 2, -6, 1, // h=1
        ];
        for (i, &v) in sc.iter().enumerate() {
            q[192 + i] = v as u8;
        }

        // d @ 208..210 (f16) = 0.5 (exact in f16, keeps products modest).
        let d: f32 = 0.5;
        let d_bits = half::f16::from_f32(d).to_bits();
        q[208] = d_bits.to_le_bytes()[0];
        q[209] = d_bits.to_le_bytes()[1];

        // ql[128] @ 0..128, qh[64] @ 128..192.
        // For each h in {0,1}, l in {0,15,16,31} (is-boundary + extremes) plus a spread of
        // other l values, choose ql/qh bits so q1..q4 take DISTINCT, checkable values.
        // Simple deterministic scheme covering both extremes and all 4 quant groups exactly:
        //   l=0:  ql1_lo=0x0, ql2_lo=0xF, ql1_hi=0x0, ql2_hi=0xF, qh bits=0b00 for all 4 pairs
        //         -> q1_u=0 (q1_f=-32, EXTREME LOW), q2_u=15, q3_u=0, q4_u=15
        //   l=15: ql1=0xF|0xF<<4=0xFF, ql2=0xF|0xF<<4=0xFF, qh bits=0b11 for all 4 pairs
        //         -> q1_u=0x0F|0x3<<4=0x3F=63 (q1_f=+31, EXTREME HIGH), q2_u=63,q3_u=63,q4_u=63
        //   l=16: (is flips to 1) ql1=0x21 (lo=1,hi=2), ql2=0x43 (lo=3,hi=4), qh bits vary
        //   l=31: ql1=0x65 (lo=5,hi=6), ql2=0x87 (lo=7,hi=8 -> nibble max 0xF so use 0x7),
        //         qh bits vary
        // All OTHER l in 1..14, 17..30 are left at 0 (q_u=0 -> q_f=-32, sc from that l's is).
        //
        // h loop: ql_base = 64h, ql2_base = 64h+32; qh_base = 128 + 32h (qh occupies [128..192)).
        for h in 0..2usize {
            let ql_base = 64 * h;
            let qh_base = 128 + 32 * h;

            // l=0: ql1=0x00, ql2=0xFF, qh=0b00000000 (all 2-bit fields = 0).
            q[ql_base + 0] = 0x00;
            q[ql_base + 32] = 0xFF;
            q[qh_base + 0] = 0x00;

            // l=15 (is=0 boundary top): ql1=0xFF, ql2=0xFF, qh=0b11111111 (all 2-bit fields = 3).
            q[ql_base + 15] = 0xFF;
            q[ql_base + 32 + 15] = 0xFF;
            q[qh_base + 15] = 0xFF;

            // l=16 (is=1 boundary bottom): ql1=0x21 (lo=1,hi=2), ql2=0x43 (lo=3,hi=4),
            // qh=0b01_00_11_10 (bit01=2,bit23=3,bit45=0,bit67=1) chosen distinctly.
            q[ql_base + 16] = 0x21;
            q[ql_base + 32 + 16] = 0x43;
            q[qh_base + 16] = 0b0100_1110;

            // l=31 (is=1 top): ql1=0x65 (lo=5,hi=6), ql2=0x87 & 0x7F pattern -> use 0x78 (lo=8&0xF=8,hi=7).
            q[ql_base + 31] = 0x65;
            q[ql_base + 32 + 31] = 0x78;
            q[qh_base + 31] = 0b1001_0110;
        }

        // ── Compute the expected FULL 256-element array from the SAME formula (independent
        //    re-derivation from the documented bytes, not a call into dequant_q6k_row_f32,
        //    so the test is a genuine byte-level cross-check). ──
        let mut expected = [0.0f32; 256];
        for h in 0..2usize {
            let ql_base = 64 * h;
            let qh_base = 128 + 32 * h;
            for l in 0..32usize {
                let is = l / 16;
                let ql1 = q[ql_base + l] as u32;
                let ql2 = q[ql_base + 32 + l] as u32;
                let qhb = q[qh_base + l] as u32;

                let q1_u = (ql1 & 0x0F) | (((qhb >> 0) & 3) << 4);
                let q2_u = (ql2 & 0x0F) | (((qhb >> 2) & 3) << 4);
                let q3_u = (ql1 >> 4) | (((qhb >> 4) & 3) << 4);
                let q4_u = (ql2 >> 4) | (((qhb >> 6) & 3) << 4);

                let q1_f = q1_u as f32 - 32.0;
                let q2_f = q2_u as f32 - 32.0;
                let q3_f = q3_u as f32 - 32.0;
                let q4_f = q4_u as f32 - 32.0;

                let sc1 = q[192 + 8 * h + is + 0] as i8 as f32;
                let sc2 = q[192 + 8 * h + is + 2] as i8 as f32;
                let sc3 = q[192 + 8 * h + is + 4] as i8 as f32;
                let sc4 = q[192 + 8 * h + is + 6] as i8 as f32;

                let base_k = 128 * h + l;
                expected[base_k] = (d * sc1) * q1_f;
                expected[base_k + 32] = (d * sc2) * q2_f;
                expected[base_k + 64] = (d * sc3) * q3_f;
                expected[base_k + 96] = (d * sc4) * q4_f;
            }
        }

        (q, expected)
    }

    /// AT-2811: dequant_q6k_row_f32 byte-exact vs the FULL 256-element golden array.
    /// Covers all four quant-groups, both halves, the is=l/16 boundary, a negative scale,
    /// and both quant extremes (per the r2 FIX-1 requirement). ANNOTATED golden-fixture
    /// PROVENANCE: hand-computed-from-formula fallback (§6) — no live ggml byte dump was
    /// available in this environment; QA should note this when a ggml cross-check lands.
    #[test]
    fn at_2811_dequant_q6k_row_full_256_golden() {
        let (q_block, expected) = golden_q6k_superblock();
        let got = dequant_q6k_row_f32(&q_block, 0, 1);
        assert_eq!(got.len(), 256, "AT-2811: dequant row must have 256 elements");
        for k in 0..256 {
            assert_eq!(
                got[k], expected[k],
                "AT-2811: element k={k} mismatch: got={}, expected={}", got[k], expected[k]
            );
        }
        // Sanity (r2 FIX-1): the fixture really does contain a negative scale and both quant
        // extremes, verified DIRECTLY from the raw bytes (not reverse-engineered from output).
        assert!(
            (192..208).any(|i| (q_block[i] as i8) < 0),
            "AT-2811 fixture sanity: must contain at least one negative int8 scale"
        );
        // l=0, h=0: ql1=0x00, ql2=0xFF, qh=0x00 -> q1_u=0 (EXTREME LOW, q1_f=-32).
        let q1_u_l0: u32 = (q_block[0] as u32 & 0x0F) | (((q_block[128] as u32) & 3) << 4);
        assert_eq!(q1_u_l0, 0, "AT-2811 fixture sanity: l=0,h=0 must give q1_u=0 (extreme low)");
        // l=15, h=0: ql1=0xFF, ql2=0xFF, qh=0xFF -> q1_u=0x0F|(3<<4)=63 (EXTREME HIGH, q1_f=+31).
        let q1_u_l15: u32 = (q_block[15] as u32 & 0x0F) | (((q_block[128 + 15] as u32) & 3) << 4);
        assert_eq!(q1_u_l15, 63, "AT-2811 fixture sanity: l=15,h=0 must give q1_u=63 (extreme high)");
        eprintln!("AT-2811 PASS: full 256-element Q6_K golden byte-exact");
    }

    /// AT-2812: signed-path + metric sanity. A superblock with sc = {-128, -1, 127} verifies
    /// the `sc_u>=128 -> -256` (here: `as i8`) reconstruction, `q6k_abs_dot_scale >= |output|`,
    /// and that `max_rel_diff_combined` catches a 1% systematic mutant (rubber-stamp guard,
    /// mirror AT-1790). No GPU.
    #[test]
    fn at_2812_signed_scale_and_metric_sanity() {
        let n_rows = 4usize;
        let n_cols = 4usize;
        let n_bpr = 1usize;
        let mut q = vec![0u8; n_rows * n_bpr * Q6K_SUPERBLOCK_BYTES];

        // Per-row distinct signed scales, cycling through -128/-1/127/0 patterns across the
        // 16-entry scales array, plus non-trivial ql/qh so weights are non-zero.
        for row in 0..n_rows {
            let base = row * Q6K_SUPERBLOCK_BYTES;
            let sc_pattern: [i8; 16] = [-128, -1, 127, 5, -128, -1, 127, 5, -128, -1, 127, 5, -128, -1, 127, 5];
            for i in 0..16 {
                q[base + 192 + i] = sc_pattern[i] as u8;
            }
            let d_bits = half::f16::from_f32(0.25).to_bits();
            q[base + 208] = d_bits.to_le_bytes()[0];
            q[base + 209] = d_bits.to_le_bytes()[1];
            // Non-trivial ql/qh: deterministic pseudo-random-ish fill.
            for i in 0..128 {
                q[base + i] = ((i * 37 + row * 11 + 13) & 0xFF) as u8;
            }
            for i in 0..64 {
                q[base + 128 + i] = ((i * 19 + row * 7 + 3) & 0xFF) as u8;
            }
        }

        // Verify the sign reconstruction directly: sc[0]=-128 (0x80), sc[1]=-1 (0xFF), sc[2]=127.
        assert_eq!(q[192] as i8, -128, "sc[0] must decode to -128");
        assert_eq!(q[193] as i8, -1, "sc[1] must decode to -1");
        assert_eq!(q[194] as i8, 127, "sc[2] must decode to 127");
        // The `sc_u>=128 -> -256` technique gives the same numeric value:
        assert_eq!((q[192] as u32) as f32 - 256.0, -128.0);
        assert_eq!((q[193] as u32) as f32 - 256.0, -1.0);
        assert_eq!(q[194] as u32 as f32, 127.0); // < 128, no subtraction needed.

        let k = n_bpr * Q6K_SUPERBLOCK_ELEMS;
        let x_f16: Vec<u16> = (0..k * n_cols)
            .map(|i| half::f16::from_f32((((i * 5 + 3) % 100) as f32) / 50.0 - 1.0).to_bits())
            .collect();

        let y = q6k_dequant_matmul_f32accum_cpu(&q, &x_f16, n_rows, n_cols, n_bpr);
        let scale = q6k_abs_dot_scale(&q, &x_f16, n_rows, n_cols, n_bpr);
        assert_eq!(y.len(), scale.len());
        for i in 0..y.len() {
            assert!(
                scale[i] + 1e-6 >= y[i].abs(),
                "AT-2812: abs_dot_scale[{i}]={} must dominate |y[{i}]|={}", scale[i], y[i].abs()
            );
        }

        // Mutation guard: a 1% systematic scale MUST be caught by the combined gate.
        let y_mut: Vec<f32> = y.iter().map(|&v| v * 1.01_f32).collect();
        let combined = max_rel_diff_combined(&y_mut, &y, &scale);
        assert!(
            combined > 1e-3,
            "AT-2812: a 1% systematic error must be CAUGHT by the combined gate; got {combined:.3e}"
        );
        eprintln!("AT-2812 PASS: signed-scale reconstruction correct; abs_dot_scale dominates; mutant caught (combined={combined:.3e})");
    }
}
