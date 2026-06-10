//! M3.10a (Q4_K_M) — AT-2402/2403/2404: the BANK-PADDED Q4_K_M scale-cached f32-accumulator RB
//! coopmat matmul (examples/q4km_matmul_rb_coopmat_f32acc_cached_pad.axc) is correct + BIT-IDENTICAL
//! to the M3.6 leader (examples/q4km_matmul_rb_coopmat_f32acc_cached.axc).
//!
//! AT-2402: combined condition-aware metric `|gpu-ref|/max(|ref|, sum|w_k·x_k|)` <= FROZEN 1e-3
//!          at K=256/512/14336 vs the f32-accumulator oracle (axc_driver::q4km_oracle). 1e-3
//!          NEVER loosened.
//! AT-2403: cross-kernel BIT-IDENTITY vs the M3.6 leader (max|y_pad - y_m36| == 0) at the same K.
//!          Padding is a PURE shared-address-layout change, so the two GPU outputs MUST match
//!          bit-for-bit. The load-bearing detector of a mis-strided shared read or the
//!          GATE-NOTE #1 (B-offset) / GATE-NOTE #2 (cache-index-on-padded-stride) traps.
//! AT-2404: bit-identity vs the M3.6 leader on a wider (A/B-class) grid (mis-stride detector at
//!          a multi-superblock shape).
//!
//! GPU-gated (#[ignore] + AXC_ENABLE_GPU_TESTS=1), typed-skip on Lavapipe / subgroup != 32.

use axc_driver::q4km_oracle;

use std::collections::BTreeMap;
use axc_driver::compile_source_with_assignments;
use axc_hir::ParamBindingPlan;
use axc_runtime::{VulkanContext, DispatchError};

const M36_SRC: &str = include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached.axc");
const PAD_SRC: &str = include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached_pad.axc");

/// Frozen relative tolerance (AT-1520/AT-1521 value — NEVER loosened).
const FROZEN_REL_TOL: f64 = 1e-3;

fn gpu_tests_enabled() -> bool {
    std::env::var("AXC_ENABLE_GPU_TESTS").as_deref() == Ok("1")
}

/// M3.6 unpadded leader strategy (a_block_size PINNED at 512).
fn rb2x2_assignments() -> BTreeMap<String, i64> {
    let mut m = BTreeMap::new();
    m.insert("rb_m".to_owned(), 2_i64);
    m.insert("rb_n".to_owned(), 2_i64);
    m.insert("tile_k".to_owned(), 16_i64);
    m.insert("a_block_size".to_owned(), 512_i64);
    m.insert("b_block_size".to_owned(), 512_i64);
    m
}

/// M3.10a padded strategy (PAD_A=PAD_B=8; SAME pad tuple as plain-f32 + the a_block_size=512 PIN).
fn rb2x2_pad_assignments() -> BTreeMap<String, i64> {
    let mut m = rb2x2_assignments();
    m.insert("a_pad_stride".to_owned(), 24_i64);
    m.insert("a_pad_size".to_owned(), 768_i64);
    m.insert("a_pad_mat1off".to_owned(), 384_i64);
    m.insert("b_pad_stride".to_owned(), 40_i64);
    m.insert("b_pad_size".to_owned(), 640_i64);
    m
}

fn assemble_pc(plan: &ParamBindingPlan, m: u32, n: u32, k: u32, n_blocks_per_row: u32) -> Vec<u8> {
    let mut pc: Vec<u8> = vec![0u8; plan.push_constant_total_bytes as usize];
    for s in &plan.scalars {
        let val: u32 = match s.name.as_str() {
            "M" => m,
            "N" => n,
            "K" => k,
            "n_blocks_per_row" => n_blocks_per_row,
            other => panic!("unexpected scalar param {other}"),
        };
        let start = s.offset as usize;
        pc[start..start + 4].copy_from_slice(&val.to_le_bytes());
    }
    pc
}

/// IDENTICAL fixture scheme to dispatch_q4km_matmul_rb_f32acc_cached.rs (so the M3.6 leader and
/// the padded kernel see the SAME inputs).
fn make_q4km_weights(m: usize, n_bpr: usize, seed: u64) -> Vec<u8> {
    use half::f16;
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
    use half::f16;
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

/// Dispatch one Q4_K_M kernel source on the given fixture; return the f32 output. None on skip.
#[allow(clippy::too_many_arguments)]
fn dispatch_one(
    at: &str,
    src: &str,
    kernel_name: &str,
    assignments: &BTreeMap<String, i64>,
    q_bytes: &[u8],
    x_bytes: &[u8],
    m: usize,
    n: usize,
    k: usize,
    n_bpr: usize,
    ctx: &VulkanContext,
) -> Option<Vec<f32>> {
    let (bytes, meta) = compile_source_with_assignments(src, assignments)
        .unwrap_or_else(|e| panic!("{at}: {kernel_name} must compile: {e:?}"));
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let handle = match ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, meta.coopmat.as_ref(), kernel_name, meta.shared_memory_bytes,
    ) {
        Ok(h) => h,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("{at}: {kernel_name} CoopMatUnsupported (typed-skip): {reason}");
            return None;
        }
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
            eprintln!("{at}: {kernel_name} DeviceFeatureUnsupported {feature}/{kernel} (typed-skip)");
            return None;
        }
        Err(e) => panic!("{at}: {kernel_name} prepare_kernel_checked failed: {e:?}"),
    };

    let pc = assemble_pc(&meta.binding_plan, m as u32, n as u32, k as u32, n_bpr as u32);
    let c_size: usize = m * n * 4; // f32 output
    let workgroups = ((n / 32) as u32, (m / 32) as u32, 1u32);

    let outputs = match ctx.dispatch_handle(
        &handle, workgroups, &[q_bytes, x_bytes, &vec![0u8; c_size]], &[0, 0, c_size], &pc,
    ) {
        Ok(v) => v,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("{at}: {kernel_name} CoopMatUnsupported at dispatch (typed-skip): {reason}");
            return None;
        }
        Err(e) => panic!("{at}: {kernel_name} dispatch failed: {e:?}"),
    };
    let y: Vec<f32> = outputs[2].chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    Some(y)
}

/// Set up ctx + fixtures, dispatch BOTH the M3.6 leader and the padded kernel.
/// Returns (y_m36, y_pad, q_bytes, x_f16) or None on a typed-skip.
#[allow(clippy::type_complexity)]
fn dispatch_both(
    at: &str,
    m: usize,
    n: usize,
    n_bpr: usize,
) -> Option<(Vec<f32>, Vec<f32>, Vec<u8>, Vec<u16>)> {
    if !gpu_tests_enabled() {
        eprintln!("{at}: AXC_ENABLE_GPU_TESTS not set; skipping");
        return None;
    }
    let ctx = VulkanContext::new().expect("VulkanContext must init");
    eprintln!("{at}: device={}", ctx.physical_device_name());
    if !ctx.coopmat_support().feature_present {
        eprintln!("{at}: coopmat not supported; typed-skip");
        return None;
    }
    if ctx.subgroup_size() != 32 {
        eprintln!("{at}: subgroup_size()={} != 32; typed-skip", ctx.subgroup_size());
        return None;
    }

    let k: usize = n_bpr * 256;
    assert!(m.is_multiple_of(32) && n.is_multiple_of(32), "{at}: M and N must be multiples of 32");

    // SAME fixture fed to BOTH kernels (same seed scheme as the M3.6 dispatch test).
    let q_bytes = make_q4km_weights(m, n_bpr, 0xC0FFEE ^ (m as u64));
    let x_f16 = make_x_f16(k, n, 0xBADF00D ^ (n as u64));
    let x_bytes: Vec<u8> = x_f16.iter().flat_map(|&b| b.to_le_bytes()).collect();

    let y_m36 = dispatch_one(
        at, M36_SRC, "q4km_matmul_rb_coopmat_f32acc_cached", &rb2x2_assignments(),
        &q_bytes, &x_bytes, m, n, k, n_bpr, &ctx,
    )?;
    let y_pad = dispatch_one(
        at, PAD_SRC, "q4km_matmul_rb_coopmat_f32acc_cached_pad", &rb2x2_pad_assignments(),
        &q_bytes, &x_bytes, m, n, k, n_bpr, &ctx,
    )?;
    Some((y_m36, y_pad, q_bytes, x_f16))
}

/// AT-2402 + AT-2403: combined metric <= FROZEN 1e-3 AND bit-identical to the M3.6 leader.
fn run_combined_and_identity(at: &str, m: usize, n: usize, n_bpr: usize) {
    let Some((y_m36, y_pad, q_bytes, x_f16)) = dispatch_both(at, m, n, n_bpr) else { return; };
    let k = n_bpr * 256;

    // AT-2402: combined condition-aware metric vs the f32-accumulator oracle.
    let y_ref = q4km_oracle::q4km_dequant_matmul_f32accum_cpu(&q_bytes, &x_f16, m, n, n_bpr);
    let raw_rel = q4km_oracle::max_rel_diff(&y_pad, &y_ref);
    let abs_scale = q4km_oracle::q4km_abs_dot_scale(&q_bytes, &x_f16, m, n, n_bpr);
    let combined = q4km_oracle::max_rel_diff_combined(&y_pad, &y_ref, &abs_scale);
    eprintln!(
        "{at}: MEASURED raw max_rel_diff={raw_rel:.3e} | COMBINED (the GATE)={combined:.3e} \
         (M={m} N={n} K={k}, frozen rtol={FROZEN_REL_TOL:.0e})"
    );
    assert!(
        combined <= FROZEN_REL_TOL,
        "{at}: K={k} padded Q4_K_M combined (condition-aware) max_rel_diff={combined:.3e} exceeds \
         frozen rtol {FROZEN_REL_TOL:.0e}. Padding is a PURE address-layout change; a violation is \
         a padded-index/cache-index bug (GATE-NOTE #2) — investigate that BEFORE touching the \
         frozen 1e-3 (NEVER loosened). raw={raw_rel:.3e}"
    );

    // AT-2403: bit-identical to the M3.6 leader.
    assert_eq!(y_m36.len(), y_pad.len(), "{at}: output length mismatch");
    let mut n_diff = 0usize;
    let mut first_diff: Option<(usize, f32, f32)> = None;
    for (i, (&a, &b)) in y_m36.iter().zip(y_pad.iter()).enumerate() {
        if a.to_bits() != b.to_bits() {
            n_diff += 1;
            if first_diff.is_none() { first_diff = Some((i, a, b)); }
        }
    }
    assert_eq!(
        n_diff, 0,
        "{at}: padded Q4_K_M output NOT bit-identical to the M3.6 leader at K={k}: {n_diff} of {} \
         differ; first {:?}. Padding is pure address layout (same accumulation order, same dequant \
         order) -> MUST match bit-for-bit. A non-zero diff is a mis-strided shared read or the \
         B-offset (GATE-NOTE #1) / cache-index (GATE-NOTE #2) trap. HONESTY GATE: do NOT relax.",
        y_m36.len(), first_diff
    );
    eprintln!(
        "{at} PASS: padded Q4_K_M within frozen 1e-3 (combined={combined:.3e}) AND bit-identical \
         to the M3.6 leader (n_diff=0) at M={m} N={n} K={k}"
    );
}

/// AT-2402/2403 at K=256 (1 superblock), M=N=64.
#[test]
#[ignore]
fn at_2402_combined_metric_k256() {
    run_combined_and_identity("AT-2402_k256", 64, 64, 1);
}

/// AT-2402/2403 at K=512 (2 superblocks), M=64, N=128.
#[test]
#[ignore]
fn at_2402_combined_metric_k512() {
    run_combined_and_identity("AT-2402_k512", 64, 128, 2);
}

/// AT-2402/2403 at K=14336 (56 superblocks, inference K), M=N=64.
#[test]
#[ignore]
fn at_2402_combined_metric_k14336() {
    run_combined_and_identity("AT-2402_k14336", 64, 64, 56);
}

/// AT-2403 alias: bit-identity at K=14336 (the r1-style-reset detector under padding).
#[test]
#[ignore]
fn at_2403_bit_identical_m36_k14336() {
    run_combined_and_identity("AT-2403_k14336", 64, 64, 56);
}

/// AT-2404: bit-identity to the M3.6 leader on a wider (A/B-class) grid — a multi-workgroup,
/// multi-superblock mis-stride detector. M=N=128 (4x4 block grid), K=512 (2 superblocks).
#[test]
#[ignore]
fn at_2404_asymmetric_grid_bit_identical() {
    let at = "AT-2404";
    let Some((y_m36, y_pad, _q, _x)) = dispatch_both(at, 128, 128, 2) else { return; };
    assert_eq!(y_m36.len(), y_pad.len(), "{at}: output length mismatch");
    let mut n_diff = 0usize;
    let mut first_diff: Option<(usize, f32, f32)> = None;
    for (i, (&a, &b)) in y_m36.iter().zip(y_pad.iter()).enumerate() {
        if a.to_bits() != b.to_bits() {
            n_diff += 1;
            if first_diff.is_none() { first_diff = Some((i, a, b)); }
        }
    }
    eprintln!("{at}: 128x128 grid, K=512, n_diff={n_diff}/{} first={:?}", y_m36.len(), first_diff);
    assert_eq!(
        n_diff, 0,
        "{at}: padded Q4_K_M NOT bit-identical to the M3.6 leader on the 128x128/K=512 grid \
         ({n_diff} differ; first {first_diff:?}). A non-zero diff at a multi-workgroup shape is a \
         mis-strided shared read / offset trap. HONESTY GATE: do NOT relax."
    );
    eprintln!(
        "{at} PASS: padded Q4_K_M bit-identical to the M3.6 leader (n_diff=0) on the 128x128/K=512 grid"
    );
}
