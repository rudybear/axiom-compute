//! Shared CPU reference + fixture generator for tiled matmul tests and benches — M3.1.
//!
//! Provides:
//! - Non-symmetric transpose-distinguishing 16x16 f16 fixtures (AT-1510/AT-1512).
//! - 16x16x16 C=A*B+C0 f16 CPU reference (accumulate in f32, round to f16).
//! - Four transpose-variant CPU references for AT-1512.
//! - General MxNxK row-major f32 matmul CPU reference.
//! - Q4_K_M multi-row dequant+matmul CPU reference (AT-1520).
//! - effective_tflops() + CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000 constant (AT-1532).
//! - f16_tile_tol() per-element tolerance.
//!
//! # Transpose-distinguishing fixture (AT-1510/AT-1512)
//!
//! A[i][j] = f16(1.0 + 0.5*i - 0.25*j)  (row-gradient)
//! B[i][j] = f16(-0.5 + 0.5*j + 0.125*i) (column-gradient)
//!
//! The pessimistic reviewer computed max|AB-AtB|=746, |AB-ABt|=330, |AB-BA|=664,
//! |AB-(AB)^T|=157 — all O(100x) the f16 tol (~5 at |C|~512).
//! A transposed/swapped/mis-strided coopmat_load CANNOT pass AT-1510.

#![allow(dead_code)]

/// Build the non-symmetric, transpose-distinguishing A matrix as f16 bit patterns.
///
/// A[i][j] = f16(1.0 + 0.5*i - 0.25*j), row-major 16x16.
/// Row-monotone: rows increase, columns decrease.
pub fn make_transpose_fixture_a() -> Vec<u16> {
    let mut out: Vec<u16> = Vec::with_capacity(256);
    for i in 0..16_usize {
        for j in 0..16_usize {
            let val: f32 = 1.0_f32 + 0.5_f32 * (i as f32) - 0.25_f32 * (j as f32);
            out.push(f32_to_f16_bits(val));
        }
    }
    out
}

/// Build the non-symmetric, transpose-distinguishing B matrix as f16 bit patterns.
///
/// B[i][j] = f16(-0.5 + 0.5*j + 0.125*i), row-major 16x16.
/// Column-monotone: columns increase, rows also increase (but at a different rate).
pub fn make_transpose_fixture_b() -> Vec<u16> {
    let mut out: Vec<u16> = Vec::with_capacity(256);
    for i in 0..16_usize {
        for j in 0..16_usize {
            let val: f32 = -0.5_f32 + 0.5_f32 * (j as f32) + 0.125_f32 * (i as f32);
            out.push(f32_to_f16_bits(val));
        }
    }
    out
}

/// CPU reference for 16x16x16 tiled matrix multiply-accumulate (f16).
///
/// C[i][j] = C0[i][j] + sum_{k=0..16} A[i][k] * B[k][j]
/// Accumulates in f32, rounds to f16 at the end.
///
/// Per HN-5: this is the tolerance-acceptable oracle — NOT bit-exact because tensor-core
/// intermediate rounding order is implementation-defined.
pub fn matmul_tile_16x16x16_f16_cpu(a: &[u16], b: &[u16], c0: &[u16]) -> Vec<u16> {
    assert_eq!(a.len(), 256);
    assert_eq!(b.len(), 256);
    assert_eq!(c0.len(), 256);
    let mut result: Vec<u16> = vec![0u16; 256];
    for i in 0..16_usize {
        for j in 0..16_usize {
            let mut acc: f32 = f16_bits_to_f32(c0[i * 16 + j]);
            for k in 0..16_usize {
                let a_val: f32 = f16_bits_to_f32(a[i * 16 + k]);
                let b_val: f32 = f16_bits_to_f32(b[k * 16 + j]);
                acc += a_val * b_val;
            }
            result[i * 16 + j] = f32_to_f16_bits(acc);
        }
    }
    result
}

/// Transpose a 16x16 f16 matrix (row-major → row-major transposed).
pub fn transpose_16x16_f16(m: &[u16]) -> Vec<u16> {
    assert_eq!(m.len(), 256);
    let mut out: Vec<u16> = vec![0u16; 256];
    for i in 0..16_usize {
        for j in 0..16_usize {
            out[j * 16 + i] = m[i * 16 + j];
        }
    }
    out
}

/// General MxNxK row-major f32 matmul CPU reference.
///
/// C[i][j] = sum_{k} A[i][k] * B[k][j]
pub fn matmul_f32_cpu(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let mut c: Vec<f32> = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc: f32 = 0.0_f32;
            for ki in 0..k {
                acc += a[i * k + ki] * b[ki * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// Q4_K_M multi-row dequant+matmul CPU reference (AT-1520).
///
/// Matches q4km_dequant_matmul.axc exactly (per HN-11 layout):
/// - q: n_rows consecutive weight-rows of n_blocks_per_row Q4_K_M superblocks (144 bytes each)
/// - x: column-of-output-major: x[k * n_cols + col]
/// - output: row-major: y[row * n_cols + col]
///
/// For 256x256: n_rows=256, n_cols=256, n_blocks_per_row=1.
pub fn q4km_dequant_matmul_cpu(
    q: &[u8],
    x: &[f32],
    n_rows: usize,
    n_cols: usize,
    n_blocks_per_row: usize,
) -> Vec<f32> {
    let mut y: Vec<f32> = vec![0.0_f32; n_rows * n_cols];
    for row in 0..n_rows {
        for col in 0..n_cols {
            let mut sum: f32 = 0.0_f32;
            for sb in 0..n_blocks_per_row {
                let base: usize = row * n_blocks_per_row * 144 + sb * 144;
                let d_bits: u16 = u16::from_le_bytes([q[base], q[base + 1]]);
                let dmin_bits: u16 = u16::from_le_bytes([q[base + 2], q[base + 3]]);
                let d: f32 = f16_bits_to_f32(d_bits as u16);
                let dmin: f32 = f16_bits_to_f32(dmin_bits as u16);
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
                        let k_lo: usize = sb * 256 + chunk * 64 + l;
                        let k_hi: usize = sb * 256 + chunk * 64 + 32 + l;
                        let x_lo: f32 = x[k_lo * n_cols + col];
                        let x_hi: f32 = x[k_hi * n_cols + col];
                        sum += lo_f * x_lo + hi_f * x_hi;
                    }
                }
            }
            y[row * n_cols + col] = sum;
        }
    }
    y
}

/// Replicate ggml's `get_scale_min_k4(j, scales)` — returns (sc, m) as u32.
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
        let hi2_m: u32 = ((q[scales_base + j] >> 6) & 0x03) as u32; // q[j], NOT q[j-4]
        let m: u32 = lo4_m | (hi2_m << 4);
        (sc, m)
    }
}

/// NVIDIA RTX PRO 6000 Blackwell f32 GEMM throughput estimate (external, not in-tree cuBLAS).
///
/// Source: NVIDIA RTX PRO 6000 Blackwell datasheet / published microbench results.
/// Peak f32 TFLOPS for the RTX PRO 6000 (Blackwell GB202) is approximately 125 TFLOPS
/// (single-precision non-tensor). This is an EXTERNAL ESTIMATE, labeled as such.
/// Only effective_tflops() > 0 && finite() is asserted (AT-1532).
pub const CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000: f64 = 125.0;

/// Compute effective TFLOPS for a matmul kernel.
///
/// effective_tflops = 2 * M * N * K / kernel_seconds
/// The factor of 2 accounts for both the multiplies and the adds.
pub fn effective_tflops(m: usize, n: usize, k: usize, kernel_seconds: f64) -> f64 {
    if kernel_seconds <= 0.0 { return 0.0; }
    let flops: f64 = 2.0 * (m as f64) * (n as f64) * (k as f64);
    let tflops: f64 = flops / kernel_seconds / 1e12;
    tflops
}

/// Per-element f16 tolerance band for the coopmat CPU oracle.
///
/// tol = max(2 f16 ULP at the element magnitude, 2^-9 * |element|).
///
/// For the transpose fixtures with |C| ~ a few hundred, this yields ~5 per element,
/// while transpose differences are O(100x) larger — decisive.
pub fn f16_tile_tol(element_magnitude: f32) -> f32 {
    let ulp_tol: f32 = f16_ulp_2(element_magnitude);
    let rel_tol: f32 = element_magnitude.abs() / 512.0_f32; // 2^-9
    ulp_tol.max(rel_tol).max(1e-4_f32) // floor to avoid zero tol on zero elements
}

/// 2 ULP in f16 at a given magnitude.
fn f16_ulp_2(magnitude: f32) -> f32 {
    if magnitude == 0.0 { return f16_bits_to_f32(1); }
    let bits: u16 = f32_to_f16_bits(magnitude.abs());
    // Add 2 to the raw f16 bits and convert back.
    let plus2: u16 = bits.saturating_add(2);
    (f16_bits_to_f32(plus2) - f16_bits_to_f32(bits)).abs()
}

// ── F16 bit-manipulation helpers ──────────────────────────────────────────────

/// Convert f32 to f16 bit pattern (round-to-nearest-even, no denormal flush).
pub fn f32_to_f16_bits(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}

/// Convert f16 bit pattern to f32 (exact).
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

/// Convert a slice of f16 bit patterns to bytes (little-endian per element).
pub fn f16_slice_to_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|&b| b.to_le_bytes()).collect()
}

/// Convert a byte slice to a Vec<u16> of f16 bit patterns (little-endian).
pub fn bytes_to_f16_vec(bytes: &[u8]) -> Vec<u16> {
    assert_eq!(bytes.len() % 2, 0);
    bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// AT-1512: transpose-distinguishing guard (CPU-only).
///
/// Asserts that the four products A·B, Aᵀ·B, A·Bᵀ, B·A differ by >10x the
/// per-element f16 tolerance at one or more elements.
/// Proves a transposed/swapped/mis-strided coopmat_load CANNOT pass AT-1510.
#[test]
fn at_1512_transpose_fixture_is_distinguishing() {
    let a: Vec<u16> = make_transpose_fixture_a();
    let b: Vec<u16> = make_transpose_fixture_b();
    let zeros: Vec<u16> = vec![0u16; 256]; // C0 = 0

    let ab:  Vec<u16> = matmul_tile_16x16x16_f16_cpu(&a, &b, &zeros);
    let at_b: Vec<u16> = matmul_tile_16x16x16_f16_cpu(&transpose_16x16_f16(&a), &b, &zeros);
    let a_bt: Vec<u16> = matmul_tile_16x16x16_f16_cpu(&a, &transpose_16x16_f16(&b), &zeros);
    let ba:   Vec<u16> = matmul_tile_16x16x16_f16_cpu(&b, &a, &zeros);
    // Also test (AB)^T: result should differ from AB.
    let ab_t: Vec<u16> = transpose_16x16_f16(&ab);

    // For each comparison, find the max abs diff.
    let max_diff = |x: &[u16], y: &[u16]| -> f32 {
        x.iter().zip(y.iter()).map(|(&xb, &yb)| {
            (f16_bits_to_f32(xb) - f16_bits_to_f32(yb)).abs()
        }).fold(0.0_f32, f32::max)
    };

    // Per reviewer: max|AB-AtB|=746.25, |AB-ABt|=330.0, |AB-BA|=663.75, |AB-(AB)^T|=157.5
    // Tolerance at |C|~512 is ~5. We require >10x tol = >50.
    let min_distinguishing_diff: f32 = 50.0_f32;

    let diff_atb: f32 = max_diff(&ab, &at_b);
    let diff_abt: f32 = max_diff(&ab, &a_bt);
    let diff_ba: f32 = max_diff(&ab, &ba);
    let diff_abt_t: f32 = max_diff(&ab, &ab_t);

    assert!(
        diff_atb >= min_distinguishing_diff,
        "AB vs AtB must differ by >{min_distinguishing_diff}; got {diff_atb}"
    );
    assert!(
        diff_abt >= min_distinguishing_diff,
        "AB vs ABt must differ by >{min_distinguishing_diff}; got {diff_abt}"
    );
    assert!(
        diff_ba >= min_distinguishing_diff,
        "AB vs BA must differ by >{min_distinguishing_diff}; got {diff_ba}"
    );
    assert!(
        diff_abt_t >= min_distinguishing_diff,
        "AB vs (AB)^T must differ by >{min_distinguishing_diff}; got {diff_abt_t}"
    );
}

/// Verify f16 round-trip for a simple value.
#[test]
fn f16_roundtrip_simple() {
    let v: f32 = 1.5_f32;
    let bits: u16 = f32_to_f16_bits(v);
    let rt: f32 = f16_bits_to_f32(bits);
    assert!((rt - v).abs() < 1e-3_f32, "f16 roundtrip must be within 1e-3 for 1.5");
}

/// Verify CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000 is positive and finite.
#[test]
fn cublas_constant_is_positive_finite() {
    assert!(CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000 > 0.0, "cuBLAS constant must be > 0");
    assert!(CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000.is_finite(), "cuBLAS constant must be finite");
}

/// Verify effective_tflops returns a sane value for a small matmul.
#[test]
fn effective_tflops_sanity() {
    // 64x64x64 in 1ms should be ~0.53 GFLOPS = 0.00053 TFLOPS.
    let tflops: f64 = effective_tflops(64, 64, 64, 0.001);
    assert!(tflops > 0.0 && tflops.is_finite(), "effective_tflops must be > 0 and finite");
    assert!(tflops < 1000.0, "effective_tflops must be < 1000 TFLOPS for a tiny matmul");
}
