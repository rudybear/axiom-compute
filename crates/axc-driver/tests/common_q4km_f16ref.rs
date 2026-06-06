//! M3.5 — f16-ACCUMULATOR Q4_K_M CPU reference (the fused-kernel correctness oracle).
//!
//! The GPU fused kernel (q4km_matmul_rb_coopmat.axc) accumulates in an f16 coopmat
//! accumulator (matrix[f16,16,16,accumulator], confirmed matmul_rb_coopmat.axc:122-125)
//! over depth-16 (tile_k) coopmat tiles. To form a VALID error bound, the CPU oracle
//! MATCHES that precision:
//!   1. dequantize Q4_K_M weights in f32 EXACTLY as ggml/M2.6,
//!   2. round each weight AND each x activation f32->f16->f32 (half-crate RNE, matching
//!      OpFConvert RNE),
//!   3. accumulate the matmul IN f16 — walk K in depth-tile_k chunks, sum each chunk,
//!      round the running accumulator to f16 after EACH chunk add (mirroring the f16
//!      coopmat accumulator's per-tile rounding).
//!
//! The frozen 1e-3 relative tolerance (AT-1520/AT-1521 value) is NOT loosened.
//!
//! LAYOUT (matches q4km_matmul_rb_coopmat.axc):
//!   q: M weight-rows, each n_blocks_per_row Q4_K_M superblocks (144 bytes), row-major.
//!   x_f16: activation/B matrix, f16 bit patterns, row-major: x[k_row * n_cols + col].
//!   output: f32 (the test widens the GPU f16 output to f32 for comparison),
//!           row-major: y[m_row * n_cols + col].
//!   K = n_blocks_per_row * 256.
//!
//! NOTE: this is NVIDIA-coopmat-accumulator-SPECIFIC. A different vendor's strict-f16
//! accumulator (or an NVIDIA tensor core accumulating internally above f16) may differ;
//! the within-tol result is documented as such, not a portable guarantee.

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

/// f16-ACCUMULATOR Q4_K_M dequant+matmul CPU reference (the M3.5 fused-kernel oracle).
///
/// Computes y[row,col] = sum_k dequant(q[row,k]) * x[k,col], with:
///   - dequant in f32 (ggml/M2.6),
///   - each weight AND each x rounded f32->f16->f32,
///   - accumulation IN f16: K walked in depth-`tile_k` chunks; each chunk's partial
///     dot is summed, then the RUNNING accumulator is rounded to f16 after the chunk
///     add (mirroring the GPU's f16 coopmat accumulator per-tile rounding).
///
/// `x_f16` holds the activation f16 bit patterns, row-major: x[k*n_cols + col].
/// Returns y as f32, row-major (n_rows * n_cols).
pub fn q4km_dequant_matmul_f16accum_cpu(
    q: &[u8],
    x_f16: &[u16],
    n_rows: usize,
    n_cols: usize,
    n_blocks_per_row: usize,
    tile_k: usize,
) -> Vec<f32> {
    assert!(tile_k > 0, "tile_k must be > 0");
    let k_total: usize = n_blocks_per_row * Q4KM_SUPERBLOCK_ELEMS;
    assert_eq!(
        k_total % tile_k, 0,
        "K ({k_total}) must be a multiple of tile_k ({tile_k}) for depth-tiled accumulation"
    );
    let n_k_tiles: usize = k_total / tile_k;

    let mut y: Vec<f32> = vec![0.0_f32; n_rows * n_cols];

    for row in 0..n_rows {
        // Dequant the row in f32, then round each weight to f16 (matching the GPU a_tile write).
        let w_f32: Vec<f32> = dequant_q4km_row_f32(q, row, n_blocks_per_row);
        let w_f16: Vec<f32> = w_f32.iter().map(|&v| round_f32_to_f16_to_f32(v)).collect();

        for col in 0..n_cols {
            // f16 accumulator, rounded to f16 after each depth-tile_k tile (matches GPU).
            let mut acc: f32 = 0.0_f32;
            for kt in 0..n_k_tiles {
                // Per-tile partial: sum tile_k products. The GPU coopmat mul_add forms the
                // 16-deep product+add as one fused op; we sum the tile in f32 then fold the
                // RUNNING accumulator (acc + tile_partial) to f16 — the per-tile rounding.
                let mut tile_partial: f32 = 0.0_f32;
                for kk in 0..tile_k {
                    let k: usize = kt * tile_k + kk;
                    let xk: f32 = round_f32_to_f16_to_f32(f16::from_bits(x_f16[k * n_cols + col]).to_f32());
                    tile_partial += w_f16[k] * xk;
                }
                acc = round_f32_to_f16_to_f32(acc + tile_partial);
            }
            y[row * n_cols + col] = acc;
        }
    }
    y
}

/// Re-derive the EXPECTED a_tile contents for a single 2×2 block / k_block, for the
/// staging-coverage assertion (AT-1620 class). Returns a 512-element Vec of f16 bit
/// patterns indexed by ei_a = a_iter*32 + lane (0..511), each written EXACTLY ONCE.
///
/// Mirrors the kernel's superblock-to-tile decode + f32_to_f16 narrowing.
pub fn expected_a_tile_f16(
    q: &[u8],
    block_row: u32,
    k_block: u32,
    n_blocks_per_row: u32,
    tile_k: u32,
) -> Vec<u16> {
    let a_block_elems: usize = 512; // RB_M*16 rows × tile_k cols = 32*16
    let mut a_tile: Vec<Option<u16>> = vec![None; a_block_elems];
    for ei_a in 0..a_block_elems as u32 {
        let a_row: u32 = ei_a / 16;
        let a_col: u32 = ei_a - a_row * 16;
        let g_row: u32 = block_row * 32 + a_row;
        let g_k: u32 = k_block * tile_k + a_col;

        let sb: u32 = g_k / 256;
        let wk: u32 = g_k - sb * 256;
        let chunk: u32 = wk / 64;
        let wc: u32 = wk - chunk * 64;
        let l: u32 = wc - (wc / 32) * 32;
        let is_hi: u32 = wc / 32;
        let is: u32 = chunk * 2 + is_hi;

        let base: usize = (g_row * n_blocks_per_row * 144 + sb * 144) as usize;
        let d: f32 = f16::from_bits(u16::from_le_bytes([q[base], q[base + 1]])).to_f32();
        let dmin: f32 = f16::from_bits(u16::from_le_bytes([q[base + 2], q[base + 3]])).to_f32();
        let scales_base: usize = base + 4;
        let qs_base: usize = base + 16;

        let (sc, mm) = get_scale_min_k4(q, scales_base, is as usize);
        let dsc: f32 = d * sc as f32;
        let dmm: f32 = dmin * mm as f32;

        let byte: u8 = q[qs_base + (chunk * 32 + l) as usize];
        let nib: u32 = if is_hi >= 1 {
            ((byte >> 4) & 0x0F) as u32
        } else {
            (byte & 0x0F) as u32
        };
        let w: f32 = dsc * nib as f32 - dmm;
        let bits: u16 = f16::from_f32(w).to_bits();
        assert!(a_tile[ei_a as usize].is_none(), "ei_a {ei_a} written twice");
        a_tile[ei_a as usize] = Some(bits);
    }
    a_tile
        .into_iter()
        .enumerate()
        .map(|(i, o)| o.unwrap_or_else(|| panic!("a_tile element {i} never written (staging gap)")))
        .collect()
}

/// Max RELATIVE diff between a GPU result and the reference (denominator floored at 1e-8).
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

// ── Tests (CPU-only, no GPU) ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// round_f32_to_f16_to_f32 is idempotent for exactly-representable f16 values.
    #[test]
    fn round_f16_idempotent_for_representable() {
        for &v in &[0.0_f32, 1.0, 1.5, -2.0, 0.25, 64.0] {
            let r1 = round_f32_to_f16_to_f32(v);
            let r2 = round_f32_to_f16_to_f32(r1);
            assert_eq!(r1.to_bits(), r2.to_bits(), "round not idempotent for {v}");
        }
    }

    /// f16-accumulator reference produces finite output on a trivial fixture and the
    /// staging-coverage helper writes every element exactly once.
    #[test]
    fn f16accum_ref_and_staging_coverage_smoke() {
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
        let y = q4km_dequant_matmul_f16accum_cpu(&q, &x_f16, n_rows, n_cols, n_bpr, 16);
        assert_eq!(y.len(), n_rows * n_cols);
        assert!(y.iter().all(|v| v.is_finite()), "all outputs must be finite");

        // Staging coverage: every a_tile element written exactly once (panics otherwise).
        let a_tile = expected_a_tile_f16(&q, 0, 0, n_bpr as u32, 16);
        assert_eq!(a_tile.len(), 512, "a_tile must have 512 elements");
    }
}
