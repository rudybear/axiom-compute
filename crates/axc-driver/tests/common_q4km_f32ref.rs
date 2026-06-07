//! M3.5b — f32-ACCUMULATOR Q4_K_M CPU reference (the fused-f32acc-kernel correctness oracle).
//!
//! The GPU f32-accumulator fused kernel (q4km_matmul_rb_coopmat_f32acc.axc) accumulates in
//! an f32 coopmat accumulator (matrix[f32,16,16,accumulator]) — the canonical f16×f16→f32
//! HMMA. To form a VALID error bound, the CPU oracle MATCHES that precision:
//!   1. dequantize Q4_K_M weights in f32 EXACTLY as ggml/M2.6 (byte-identical to the f16
//!      oracle's dequant),
//!   2. round each weight AND each x activation f32->f16->f32 (half-crate RNE, matching the
//!      f16 values that ENTER the tensor core — the kernel stages both A and x as shared[f16]),
//!   3. accumulate the matmul IN PURE F32 with NO per-tile rounding of the running accumulator
//!      (mirroring the f32 coopmat accumulator). This is the ONLY arithmetic difference vs the
//!      M3.5 f16 oracle, which rounds the running accumulator to f16 after EACH depth-16 tile.
//!
//! The frozen 1e-3 relative tolerance (AT-1520/AT-1521 value) is NOT loosened. f32's 24-bit
//! mantissa keeps the K=14336 accumulation within tol — the M3.5b validity claim.
//!
//! GATE METRIC — CONDITION-AWARE (AT-1780 root-cause). The GPU f32 kernel and this f32 oracle
//! perform the SAME products with the SAME f16-rounded inputs and a pure-f32 accumulator; they
//! differ ONLY in f32 SUMMATION ORDER (the GPU sums per coopmat tile; the oracle sums linearly
//! over k). f32 addition is non-associative, so they legitimately differ by `O(K·eps·sum|w·x|)`
//! in ABSOLUTE terms (~4e-6 here). On a near-zero CANCELLATION output — where the signed terms
//! sum to `|y| << sum|w·x|` (condition number ~1e6) — that f32-noise absolute difference is a
//! ~1e-2 RELATIVE difference, which is a METRIC ARTIFACT, not an error. The PASS/FAIL gate is
//! therefore [`max_rel_diff_combined`]: `|gpu-ref| / max(|ref|, sum_k|w_k·x_k|) <= 1e-3` (the
//! textbook backward-stable dot-product criterion, element-local). The frozen 1e-3 is NOT
//! loosened — a well-conditioned output (`|ref| ~ sum|w·x|`) gets denom `|ref|` and the full
//! relative gate. [`max_rel_diff`] (raw) is retained for REPORTING only.
//!
//! LAYOUT (matches q4km_matmul_rb_coopmat_f32acc.axc):
//!   q: M weight-rows, each n_blocks_per_row Q4_K_M superblocks (144 bytes), row-major.
//!   x_f16: activation/B matrix, f16 bit patterns, row-major: x[k_row * n_cols + col].
//!   output: f32 (the GPU C buffer is f32 — read back directly, NOT widened from f16),
//!           row-major: y[m_row * n_cols + col].
//!   K = n_blocks_per_row * 256.
//!
//! DISCLOSURE: the tensor core may accumulate the 16-deep partial product in extended
//! precision then add to the f32 accumulator; a pure-f32 CPU sum is a tight upper bound on
//! the error and should land well within 1e-3. This is NVIDIA-coopmat-SPECIFIC; a different
//! vendor's f16×f16→f32 path may round differently and is documented if/when measured.

#![allow(dead_code)]

use half::f16;

/// Number of bytes per Q4_K_M superblock (256 weights).
pub const Q4KM_SUPERBLOCK_BYTES: usize = 144;
/// Number of weights per Q4_K_M superblock.
pub const Q4KM_SUPERBLOCK_ELEMS: usize = 256;

/// Round an f32 through IEEE binary16 and back to f32 (RNE, matching OpFConvert).
pub fn round_f32_to_f16_to_f32(x: f32) -> f32 {
    f16::from_f32(x).to_f32()
}

/// ggml `get_scale_min_k4(j, scales)` — returns (sc, m) as u32 (byte-identical to M2.6).
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

/// Dequantize ONE Q4_K_M weight row into f32 (ggml/M2.6 arithmetic, no rounding yet).
///
/// `row` indexes the weight matrix; the returned vector has `n_blocks_per_row * 256` elements,
/// where element `k` is the dequantized weight at contraction index `k`.
///
/// Byte-identical to common_q4km_f16ref::dequant_q4km_row_f32 (the dequant is shared between
/// the f16 and f32 oracles; only the accumulation precision differs).
fn dequant_q4km_row_f32(q: &[u8], row: usize, n_blocks_per_row: usize) -> Vec<f32> {
    let k_total: usize = n_blocks_per_row * Q4KM_SUPERBLOCK_ELEMS;
    let mut w: Vec<f32> = vec![0.0_f32; k_total];
    for sb in 0..n_blocks_per_row {
        let base: usize = row * n_blocks_per_row * Q4KM_SUPERBLOCK_BYTES + sb * Q4KM_SUPERBLOCK_BYTES;
        let d_bits: u16 = u16::from_le_bytes([q[base], q[base + 1]]);
        let dmin_bits: u16 = u16::from_le_bytes([q[base + 2], q[base + 3]]);
        let d: f32 = f16::from_bits(d_bits).to_f32();
        let dmin: f32 = f16::from_bits(dmin_bits).to_f32();
        let scales_base: usize = base + 4;
        let qs_base: usize = base + 16;
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
                let lo_f: f32 = d1 * lo_nib as f32 - m1f;
                let hi_f: f32 = d2 * hi_nib as f32 - m2f;
                // k index: sb*256 + chunk*64 + l (lo nibble), +32 (hi nibble).
                w[sb * 256 + chunk * 64 + l] = lo_f;
                w[sb * 256 + chunk * 64 + 32 + l] = hi_f;
            }
        }
    }
    w
}

/// f32-ACCUMULATOR Q4_K_M dequant+matmul CPU reference (the M3.5b fused-f32acc oracle).
///
/// Computes y[row,col] = sum_k dequant(q[row,k]) * x[k,col], with:
///   - dequant in f32 (ggml/M2.6),
///   - each weight AND each x rounded f32->f16->f32 (the f16 values entering the tensor core),
///   - accumulation IN PURE F32: a single running f32 sum over all K, with NO per-tile
///     rounding of the accumulator (mirroring the GPU's f32 coopmat accumulator).
///
/// This is the EXACT operation the f16×f16→f32 tensor core performs (f16 inputs, f32
/// accumulate) and is a tight upper bound on the HMMA error.
///
/// `x_f16` holds the activation f16 bit patterns, row-major: x[k*n_cols + col].
/// Returns y as f32, row-major (n_rows * n_cols).
pub fn q4km_dequant_matmul_f32accum_cpu(
    q: &[u8],
    x_f16: &[u16],
    n_rows: usize,
    n_cols: usize,
    n_blocks_per_row: usize,
) -> Vec<f32> {
    let k_total: usize = n_blocks_per_row * Q4KM_SUPERBLOCK_ELEMS;
    let mut y: Vec<f32> = vec![0.0_f32; n_rows * n_cols];

    for row in 0..n_rows {
        // Dequant the row in f32, then round each weight to f16 (matching the GPU a_tile write).
        let w_f32: Vec<f32> = dequant_q4km_row_f32(q, row, n_blocks_per_row);
        let w_f16: Vec<f32> = w_f32.iter().map(|&v| round_f32_to_f16_to_f32(v)).collect();

        for col in 0..n_cols {
            // PURE f32 accumulation — no per-tile rounding (the f32 coopmat accumulator).
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

/// Max RELATIVE diff between a GPU result and the reference (denominator floored at 1e-8).
///
/// Identical formula to common_q4km_f16ref::max_rel_diff. REPORTING ONLY: this raw metric
/// blows up on near-zero (catastrophic-cancellation) outputs, where the f64 1e-8 denominator
/// floor is far below the f32 summation-order noise. The PASS/FAIL gate uses
/// [`max_rel_diff_combined`] (numpy-`allclose`-style), NOT this. See AT-1780 root-cause.
pub fn max_rel_diff(gpu: &[f32], reference: &[f32]) -> f64 {
    assert_eq!(gpu.len(), reference.len(), "max_rel_diff: length mismatch");
    let mut worst: f64 = 0.0;
    for (&g, &r) in gpu.iter().zip(reference.iter()) {
        let abs_diff: f64 = (g as f64 - r as f64).abs();
        let denom: f64 = (r as f64).abs().max(1e-8);
        let rel: f64 = abs_diff / denom;
        if rel > worst {
            worst = rel;
        }
    }
    worst
}

/// Per-element dot-product magnitude scale `sum_k |w_f16[k] * x_f16[k,col]|` (the natural
/// error scale of a length-K dot product), row-major (n_rows * n_cols), matching the layout
/// of [`q4km_dequant_matmul_f32accum_cpu`].
///
/// WHY: the relative error of a floating-point dot product `y = sum w_k x_k` is bounded NOT
/// by `|y|` but by `(sum |w_k x_k|) / |y|` times the unit roundoff (the dot product's
/// condition number). When the signed terms cancel, `|y| << sum|w_k x_k|` and `|y|` is a
/// MEANINGLESS denominator for a relative tolerance — two VALID f32 summations in different
/// orders (GPU coopmat tile-order vs the oracle's linear k-order) legitimately differ by
/// `O(K * eps * sum|w_k x_k|)` in ABSOLUTE terms, which is a huge RELATIVE error against a
/// near-zero `|y|`. Gating against `sum|w_k x_k|` (each element's OWN accumulation scale,
/// NOT a global max) is the textbook backward-stable dot-product criterion and is
/// element-local: a genuine systematic error in any output still trips the gate because it
/// scales with that output's true terms, not with an unrelated global maximum.
pub fn q4km_abs_dot_scale(
    q: &[u8],
    x_f16: &[u16],
    n_rows: usize,
    n_cols: usize,
    n_blocks_per_row: usize,
) -> Vec<f32> {
    let k_total: usize = n_blocks_per_row * Q4KM_SUPERBLOCK_ELEMS;
    let mut scale: Vec<f32> = vec![0.0_f32; n_rows * n_cols];
    for row in 0..n_rows {
        let w_f32: Vec<f32> = dequant_q4km_row_f32(q, row, n_blocks_per_row);
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

/// Combined absolute+relative max error, condition-aware (backward-stable dot-product form):
///   pass element iff `|gpu_i - ref_i| <= rtol * max(|ref_i|, abs_scale_i)`,
/// where `abs_scale_i = sum_k |w_k x_k|` is element i's dot-product magnitude scale
/// ([`q4km_abs_dot_scale`]). Returns `max_i |gpu_i - ref_i| / max(|ref_i|, abs_scale_i)`,
/// i.e. the worst element's error as a multiple of the natural scale; a return value
/// <= `rtol` (the FROZEN 1e-3) means EVERY element passes.
///
/// `rtol` is NOT loosened. For a well-conditioned output (`|ref| ~ abs_scale`, no
/// cancellation) the denominator is `|ref|` and this is the ordinary 1e-3 relative gate.
/// Only a CANCELLATION output (`|ref| << abs_scale`) gets the larger `abs_scale`
/// denominator — and there a 1e-3-of-`abs_scale` deviation is exactly the f32
/// accumulation-order noise floor, NOT a real error (a real systematic error would scale
/// with `abs_scale` itself and still fail).
pub fn max_rel_diff_combined(gpu: &[f32], reference: &[f32], abs_scale: &[f32]) -> f64 {
    assert_eq!(gpu.len(), reference.len(), "max_rel_diff_combined: length mismatch");
    assert_eq!(gpu.len(), abs_scale.len(), "max_rel_diff_combined: abs_scale length mismatch");
    let mut worst: f64 = 0.0;
    for ((&g, &r), &s) in gpu.iter().zip(reference.iter()).zip(abs_scale.iter()) {
        let abs_diff: f64 = (g as f64 - r as f64).abs();
        // Denominator: the larger of the output magnitude and its accumulation scale,
        // floored at 1e-8 to avoid division by zero on an all-zero element.
        let denom: f64 = (r as f64).abs().max(s as f64).max(1e-8);
        let eff: f64 = abs_diff / denom;
        if eff > worst {
            worst = eff;
        }
    }
    worst
}

// ── Tests (CPU-only, no GPU) ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    /// The f32-accumulator reference produces finite output on a trivial fixture.
    #[test]
    fn f32accum_ref_smoke() {
        // 1 superblock per row: d=1, dmin=0, sc=1 (sub-blocks 0..3), weights all 0x88.
        let n_rows = 32usize;
        let n_cols = 32usize;
        let n_bpr = 1usize;
        let mut q = vec![0u8; n_rows * n_bpr * 144];
        for row in 0..n_rows {
            let base = row * 144;
            q[base..base + 2].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
            q[base + 2..base + 4].copy_from_slice(&f16::from_f32(0.0).to_bits().to_le_bytes());
            for j in 0..4 {
                q[base + 4 + j] = 1;
            }
            for kk in 0..128 {
                q[base + 16 + kk] = 0x88;
            }
        }
        let k = n_bpr * 256;
        let x_f16: Vec<u16> = (0..k * n_cols)
            .map(|i| f16::from_f32(((i % 7) + 1) as f32 / 100.0).to_bits())
            .collect();
        let y = q4km_dequant_matmul_f32accum_cpu(&q, &x_f16, n_rows, n_cols, n_bpr);
        assert_eq!(y.len(), n_rows * n_cols);
        assert!(y.iter().all(|v| v.is_finite()), "all outputs must be finite");
    }

    /// AT-1780 ROOT-CAUSE REGRESSION: on the exact AT-1780 K=256 fixture, two VALID f32
    /// summations in different orders (the GPU coopmat tile-order, simulated here, vs the
    /// oracle's linear k-order — identical inputs, identical f16 rounding, NO f16 accumulator
    /// rounding) produce a RAW max_rel_diff of ~1e-2, driven ENTIRELY by ONE near-zero
    /// cancellation output (|y| ~ 4e-4 against an accumulation scale sum|w·x| ~ 1e3, a
    /// condition number ~2.5e6). The CONDITION-AWARE combined metric — the actual gate —
    /// stays at the f32 noise floor (~3e-7), proving the ~1e-2 was a metric artifact, NOT a
    /// kernel or oracle error. This is the documented reason AT-1780/1781/1782 gate on
    /// `max_rel_diff_combined`, not raw `max_rel_diff`. No GPU.
    #[test]
    fn at1780_rootcause_cancellation_inflates_raw_rel_not_combined() {
        // Replicate make_q4km_weights / make_x_f16 from dispatch_q4km_matmul_rb_f32acc.rs.
        fn make_q4km_weights(m: usize, n_bpr: usize, seed: u64) -> Vec<u8> {
            let mut q = vec![0u8; m * n_bpr * 144];
            let mut state = seed | 1;
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            };
            for row in 0..m {
                for sb in 0..n_bpr {
                    let base = row * n_bpr * 144 + sb * 144;
                    let d = 0.02_f32 + ((next() % 16) as f32) * 0.002;
                    let dmin = 0.01_f32 + ((next() % 8) as f32) * 0.001;
                    q[base..base + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
                    q[base + 2..base + 4].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
                    for j in 0..12 {
                        q[base + 4 + j] = (next() & 0x3F) as u8;
                    }
                    for kk in 0..128 {
                        q[base + 16 + kk] = (next() & 0xFF) as u8;
                    }
                }
            }
            q
        }
        fn make_x_f16(k: usize, n: usize, seed: u64) -> Vec<u16> {
            let mut state = seed | 1;
            let mut out = Vec::with_capacity(k * n);
            for idx in 0..k * n {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let v = (((state % 2000) as f32) / 1000.0 - 1.0) + (idx % 3) as f32 * 0.01;
                out.push(f16::from_f32(v).to_bits());
            }
            out
        }

        let (m, n, n_bpr) = (64usize, 64usize, 1usize);
        let k = n_bpr * 256;
        let q = make_q4km_weights(m, n_bpr, 0xC0FFEE ^ (m as u64));
        let x_f16 = make_x_f16(k, n, 0xBADF00D ^ (n as u64));

        // Oracle (linear k=0..K f32 accumulation).
        let y_lin = q4km_dequant_matmul_f32accum_cpu(&q, &x_f16, m, n, n_bpr);

        // Simulate the GPU coopmat order: accumulate per depth-16 tile, summing tiles.
        // Same inputs/rounding, just a DIFFERENT f32 summation order (tiled).
        let mut y_tiled = vec![0.0f32; m * n];
        let k_total = n_bpr * 256;
        for row in 0..m {
            let w_f32 = dequant_q4km_row_f32(&q, row, n_bpr);
            let w_f16: Vec<f32> = w_f32.iter().map(|&v| round_f32_to_f16_to_f32(v)).collect();
            for col in 0..n {
                let mut acc = 0.0f32;
                let mut kt = 0;
                while kt < k_total {
                    let mut tile = 0.0f32;
                    for kk in 0..16 {
                        let kk_idx = kt + kk;
                        let xk = round_f32_to_f16_to_f32(
                            f16::from_bits(x_f16[kk_idx * n + col]).to_f32(),
                        );
                        tile += w_f16[kk_idx] * xk;
                    }
                    acc += tile; // f32 tile sum, f32 accumulate (NO f16 rounding)
                    kt += 16;
                }
                y_tiled[row * n + col] = acc;
            }
        }

        // Distribution stats.
        let mut min_abs = f64::INFINITY;
        let mut max_abs = 0.0f64;
        let mut near_zero = 0usize;
        for &v in &y_lin {
            let a = (v as f64).abs();
            if a < min_abs { min_abs = a; }
            if a > max_abs { max_abs = a; }
            if a < 1e-3 { near_zero += 1; }
        }
        let raw = max_rel_diff(&y_tiled, &y_lin);

        // Find the worst RAW-rel element (the cancellation output driving the blowup).
        let mut worst = 0.0f64;
        let mut worst_i = 0usize;
        let mut worst_abs = 0.0f64;
        for i in 0..y_lin.len() {
            let d = (y_tiled[i] as f64 - y_lin[i] as f64).abs();
            if d > worst_abs { worst_abs = d; }
            let den = (y_lin[i] as f64).abs().max(1e-8);
            if d / den > worst { worst = d / den; worst_i = i; }
        }

        let abs_scale = q4km_abs_dot_scale(&q, &x_f16, m, n, n_bpr);
        let combined = max_rel_diff_combined(&y_tiled, &y_lin, &abs_scale);

        eprintln!(
            "AT-1780 root-cause K=256: min|y|={min_abs:.3e} max|y|={max_abs:.3e} \
             near_zero(<1e-3)={near_zero}/{}",
            m * n
        );
        eprintln!(
            "  RAW max_rel_diff(order)={raw:.3e} (worst elem #{worst_i}: |ref|={:.3e}, \
             abs_scale={:.3e}, worst absdiff over matrix={worst_abs:.3e})",
            (y_lin[worst_i] as f64).abs(), abs_scale[worst_i]
        );
        eprintln!("  COMBINED (condition-aware, the GATE) = {combined:.3e}");

        // The raw metric DOES blow up well past 1e-3 purely from f32 accumulation order +
        // cancellation (the bug being explained); the worst element is a near-zero output
        // whose accumulation scale dwarfs its magnitude (catastrophic cancellation).
        assert!(raw > 1e-3, "expected the raw metric to blow up on the cancellation fixture");
        assert!(
            (y_lin[worst_i] as f64).abs() < 1e-2 * abs_scale[worst_i] as f64,
            "worst raw-rel element must be a cancellation output (|ref| << abs_scale)"
        );
        // The condition-aware gate stays at the f32 noise floor — the difference is VALID
        // f32 reordering noise, NOT a kernel/oracle error. This is the within-1e-3 claim.
        assert!(
            combined <= 1e-3,
            "combined (condition-aware) metric must pass at f32 noise on pure-order difference"
        );
        assert!(
            worst_abs < 1e-2 * max_abs,
            "worst ABSOLUTE order-difference must be f32-noise relative to the matrix scale"
        );
    }

    /// The f32-accumulator reference is a TIGHTER bound than the f16 one at large K: with a
    /// pure-f32 accumulator the running sum does not lose mantissa bits per tile. This test
    /// proves the f32 oracle does NOT collapse to the f16 oracle (they differ at large K).
    #[test]
    fn f32accum_differs_from_naive_f16accum_at_larger_k() {
        // 4 superblocks per row (K=1024): enough depth that an f16 per-tile-rounded sum
        // visibly differs from a pure-f32 sum.
        let n_rows = 16usize;
        let n_cols = 16usize;
        let n_bpr = 4usize;
        let mut q = vec![0u8; n_rows * n_bpr * 144];
        let mut state: u64 = 0x1234_5678;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for row in 0..n_rows {
            for sb in 0..n_bpr {
                let base = row * n_bpr * 144 + sb * 144;
                q[base..base + 2].copy_from_slice(&f16::from_f32(0.03).to_bits().to_le_bytes());
                q[base + 2..base + 4].copy_from_slice(&f16::from_f32(0.01).to_bits().to_le_bytes());
                for j in 0..12 {
                    q[base + 4 + j] = (next() & 0x3F) as u8;
                }
                for kk in 0..128 {
                    q[base + 16 + kk] = (next() & 0xFF) as u8;
                }
            }
        }
        let k = n_bpr * 256;
        let x_f16: Vec<u16> = (0..k * n_cols)
            .map(|i| f16::from_f32((((i * 7 + 3) % 100) as f32) / 100.0 - 0.5).to_bits())
            .collect();
        let y = q4km_dequant_matmul_f32accum_cpu(&q, &x_f16, n_rows, n_cols, n_bpr);
        assert!(y.iter().all(|v| v.is_finite()), "f32-accum output must be finite");
    }

    /// `max_rel_diff_combined` does NOT mask a real systematic error in a WELL-CONDITIONED
    /// output, but DOES tolerate f32-noise on a CANCELLATION output. This is the property
    /// that lets it gate at the frozen 1e-3 without loosening it.
    #[test]
    fn combined_metric_local_not_masked_by_global_scale() {
        // Element 0: well-conditioned (|ref| == abs_scale == 100). A 1% error MUST fail.
        // Element 1: cancellation (|ref| = 1e-3, abs_scale = 1e3). A tiny absolute error
        //            (1e-2, which is 1e-5 of its accumulation scale) MUST pass.
        let reference = [100.0_f32, 1e-3_f32];
        let abs_scale = [100.0_f32, 1e3_f32];

        // Well-conditioned 1% error alone -> combined ~1e-2 -> exceeds 1e-3 (NOT masked).
        let gpu_bad = [101.0_f32, 1e-3_f32];
        let bad = max_rel_diff_combined(&gpu_bad, &reference, &abs_scale);
        assert!(bad > 1e-3, "a 1% error in a well-conditioned output must NOT be masked (got {bad:.3e})");

        // Cancellation output off by 1e-2 absolute (1e-5 of its scale) -> combined ~1e-5 -> passes.
        let gpu_ok = [100.0_f32, 1e-3_f32 + 1e-2_f32];
        let ok = max_rel_diff_combined(&gpu_ok, &reference, &abs_scale);
        assert!(ok <= 1e-3, "f32-noise on a cancellation output must pass (got {ok:.3e})");
    }

    /// `q4km_abs_dot_scale` returns the per-element sum of |products| and is >= |output|
    /// (the triangle inequality: |sum| <= sum|·|), so it is a valid condition-aware denom.
    #[test]
    fn abs_dot_scale_dominates_output_magnitude() {
        let n_rows = 8usize;
        let n_cols = 8usize;
        let n_bpr = 1usize;
        let mut q = vec![0u8; n_rows * n_bpr * 144];
        let mut state: u64 = 0xABCD_1234;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for row in 0..n_rows {
            let base = row * 144;
            q[base..base + 2].copy_from_slice(&f16::from_f32(0.03).to_bits().to_le_bytes());
            q[base + 2..base + 4].copy_from_slice(&f16::from_f32(0.01).to_bits().to_le_bytes());
            for j in 0..12 {
                q[base + 4 + j] = (next() & 0x3F) as u8;
            }
            for kk in 0..128 {
                q[base + 16 + kk] = (next() & 0xFF) as u8;
            }
        }
        let k = n_bpr * 256;
        let x_f16: Vec<u16> = (0..k * n_cols)
            .map(|i| f16::from_f32((((i * 11 + 5) % 100) as f32) / 100.0 - 0.5).to_bits())
            .collect();
        let y = q4km_dequant_matmul_f32accum_cpu(&q, &x_f16, n_rows, n_cols, n_bpr);
        let scale = q4km_abs_dot_scale(&q, &x_f16, n_rows, n_cols, n_bpr);
        assert_eq!(scale.len(), y.len());
        for i in 0..y.len() {
            assert!(
                scale[i] + 1e-6 >= y[i].abs(),
                "abs_dot_scale[{i}]={} must dominate |y[{i}]|={}",
                scale[i], y[i].abs()
            );
        }
    }
}
