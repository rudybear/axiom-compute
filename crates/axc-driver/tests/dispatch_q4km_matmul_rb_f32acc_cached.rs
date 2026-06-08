//! M3.6 — GPU bit-within-tol correctness tests for the DEQUANT-SCALE-CACHED f32-accumulator
//! fused Q4_K_M register-blocked coopmat matmul
//! (examples/q4km_matmul_rb_coopmat_f32acc_cached.axc).
//!
//! AT-1800: K=256 (1 superblock = 16 tile_k K-blocks), M=N=64. ASSERTED within frozen 1e-3.
//! AT-1801: K=512 (2 superblocks = 32 tile_k K-blocks), M=64, N=128. ASSERTED UNCONDITIONALLY
//!          within frozen 1e-3 — the M3.5 f16-accumulator failure case, must hold with f32 acc
//!          + caching. Also exercises the cross-superblock cache WAR (2 fills; the r3 HOISTED
//!          unconditional RAW barrier fires every k_block, so each fill is correctly ordered).
//! AT-1802: K=14336 (56 superblocks, the inference-K A/B shape), M=N=64. The validity claim
//!          survives caching. This is the test that would CATCH an r1-style accumulator reset
//!          (silent zeros -> combined ~= 1 >> 1e-3).
//!
//! The cached kernel is a PURE REASSOCIATION of M3.5b (same dsc/dmm products, same
//! multiply-then-subtract order, same single-level OpPhi accumulation order — dsc/dmm read
//! from shared instead of recomputed), so the combined condition-aware diff vs the
//! f32-accumulator oracle (common_q4km_f32ref::q4km_dequant_matmul_f32accum_cpu) MUST stay
//! within the FROZEN 1e-3 (NOT loosened). The cross-kernel BIT-IDENTITY anchor is AT-1803
//! (dispatch_q4km_f32acc_cached_equiv.rs).
//!
//! GATE METRIC: the CONDITION-AWARE diff `|gpu-ref| / max(|ref|, sum_k|w_k·x_k|)` (identical
//! to AT-1780/1782), NOT the raw `|gpu-ref|/|ref|` (which blows up on near-zero cancellation
//! outputs — a metric artifact). Both are REPORTED at every size. The 1e-3 NEVER loosens.
//!
//! NVIDIA #[ignore]-gated (AXC_ENABLE_GPU_TESTS=1). Typed-skip on Lavapipe
//! (CoopMatUnsupported / DeviceFeatureUnsupported) and on subgroup_size() != 32.

// M4.1p2: the oracle is now the pub lib module axc_driver::q4km_oracle (single source of truth).
use axc_driver::q4km_oracle as common_q4km_f32ref;

use std::collections::BTreeMap;
use axc_driver::compile_source_with_assignments;
use axc_hir::ParamBindingPlan;
use axc_runtime::{VulkanContext, DispatchError};

const CACHED_F32ACC_SRC: &str =
    include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached.axc");

/// Frozen relative tolerance (AT-1520/AT-1521 value — NOT loosened).
const FROZEN_REL_TOL: f64 = 1e-3;

fn gpu_tests_enabled() -> bool {
    std::env::var("AXC_ENABLE_GPU_TESTS").as_deref() == Ok("1")
}

/// RB 2×2 strategy assignments (matching the cached kernel; a_block_size PINNED at 512).
fn rb2x2_assignments() -> BTreeMap<String, i64> {
    let mut m = BTreeMap::new();
    m.insert("rb_m".to_owned(), 2_i64);
    m.insert("rb_n".to_owned(), 2_i64);
    m.insert("tile_k".to_owned(), 16_i64);
    m.insert("a_block_size".to_owned(), 512_i64);
    m.insert("b_block_size".to_owned(), 512_i64);
    m
}

/// Assemble push constants by scalar name (robust to layout).
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

/// Build a NON-symmetric Q4_K_M weight matrix fixture (M rows × n_bpr superblocks).
/// IDENTICAL to dispatch_q4km_matmul_rb_f32acc.rs (same seed scheme) so AT-1803 can compare
/// the two kernels on the SAME fixture.
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

/// Build a NON-symmetric f16 activation matrix x[K, N] (row-major, f16 bit patterns).
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

/// Measured error for one dispatch: RAW relative diff (reported) + COMBINED condition-aware
/// diff (the PASS/FAIL gate).
struct MeasuredErr {
    raw_rel: f64,
    combined: f64,
}

/// Core: dispatch the CACHED f32-accumulator fused kernel and compare vs the f32-acc oracle.
/// Returns the measured errors, or None on a typed-skip (no GPU/coopmat).
fn run_cached_f32acc(at: &str, m: usize, n: usize, n_bpr: usize) -> Option<MeasuredErr> {
    if !gpu_tests_enabled() {
        eprintln!("{at}: AXC_ENABLE_GPU_TESTS not set; skipping");
        return None;
    }
    let ctx = VulkanContext::new().expect("VulkanContext must init");
    eprintln!("{at}: device={}", ctx.physical_device_name());

    if !ctx.coopmat_support().feature_present {
        eprintln!("{at}: coopmat not supported on {}; typed-skip", ctx.physical_device_name());
        return None;
    }
    if ctx.subgroup_size() != 32 {
        eprintln!("{at}: subgroup_size()={} != 32; typed-skip (kernel requires wave32)", ctx.subgroup_size());
        return None;
    }

    let k: usize = n_bpr * 256;
    assert!(m.is_multiple_of(32) && n.is_multiple_of(32), "{at}: M and N must be multiples of 32");

    // Fixtures (non-symmetric) — SAME seed scheme as the M3.5b dispatch test.
    let q_bytes = make_q4km_weights(m, n_bpr, 0xC0FFEE ^ (m as u64));
    let x_f16 = make_x_f16(k, n, 0xBADF00D ^ (n as u64));
    let x_bytes: Vec<u8> = x_f16.iter().flat_map(|&b| b.to_le_bytes()).collect();

    // f32-accumulator CPU reference.
    let y_ref = common_q4km_f32ref::q4km_dequant_matmul_f32accum_cpu(&q_bytes, &x_f16, m, n, n_bpr);

    // Compile the CACHED kernel with RB assignments.
    let assignments = rb2x2_assignments();
    let (bytes, meta) = compile_source_with_assignments(CACHED_F32ACC_SRC, &assignments)
        .expect("q4km_matmul_rb_coopmat_f32acc_cached.axc must compile");
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let handle = match ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, meta.coopmat.as_ref(), "q4km_matmul_rb_coopmat_f32acc_cached",
        meta.shared_memory_bytes,
    ) {
        Ok(h) => h,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("{at}: CoopMatUnsupported (typed-skip): {reason}");
            return None;
        }
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
            eprintln!("{at}: DeviceFeatureUnsupported {feature}/{kernel} (typed-skip)");
            return None;
        }
        Err(e) => panic!("{at}: prepare_kernel_checked failed: {e:?}"),
    };

    let pc = assemble_pc(&meta.binding_plan, m as u32, n as u32, k as u32, n_bpr as u32);
    let c_size: usize = m * n * 4; // f32 output (4 bytes/elem — NOT f16).
    // Grid = (N/32, M/32, 1).
    let workgroups = ((n / 32) as u32, (m / 32) as u32, 1u32);

    let outputs = match ctx.dispatch_handle(
        &handle, workgroups,
        &[&q_bytes, &x_bytes, &vec![0u8; c_size]],
        &[0, 0, c_size],
        &pc,
    ) {
        Ok(v) => v,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("{at}: CoopMatUnsupported at dispatch (typed-skip): {reason}");
            return None;
        }
        Err(e) => panic!("{at}: dispatch failed: {e:?}"),
    };

    let c_bytes: &[u8] = &outputs[2];
    assert_eq!(c_bytes.len(), c_size, "{at}: C output size mismatch");
    // f32 readback — 4 bytes/elem, NO f16 widening.
    let y_gpu: Vec<f32> = c_bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let raw_rel = common_q4km_f32ref::max_rel_diff(&y_gpu, &y_ref);
    let abs_scale = common_q4km_f32ref::q4km_abs_dot_scale(&q_bytes, &x_f16, m, n, n_bpr);
    let combined = common_q4km_f32ref::max_rel_diff_combined(&y_gpu, &y_ref, &abs_scale);
    eprintln!(
        "{at}: MEASURED raw max_rel_diff={raw_rel:.3e} | COMBINED (condition-aware, the GATE) \
         ={combined:.3e} (M={m} N={n} K={k}, frozen rtol={FROZEN_REL_TOL:.0e}) on {}",
        ctx.physical_device_name()
    );
    Some(MeasuredErr { raw_rel, combined })
}

/// AT-1800: K=256 (1 superblock), M=N=64. Within-tol ASSERTED.
#[test]
#[ignore]
fn at_1800_q4km_rb_coopmat_f32acc_cached_within_tol_k256() {
    let Some(e) = run_cached_f32acc("at_1800", 64, 64, 1) else { return; };
    assert!(
        e.combined <= FROZEN_REL_TOL,
        "AT-1800: K=256 cached f32-accumulator combined (condition-aware) max_rel_diff={:.3e} \
         exceeds frozen rtol {FROZEN_REL_TOL:.0e} (caching is pure reassociation; the frozen \
         1e-3 is NOT loosened; raw={:.3e})",
        e.combined, e.raw_rel
    );
    eprintln!(
        "at_1800: PASS — cached K=256 within frozen 1e-3 (combined={:.3e}, raw={:.3e})",
        e.combined, e.raw_rel
    );
}

/// AT-1801: K=512 (2 superblocks), M=64, N=128. Within-tol ASSERTED UNCONDITIONALLY.
/// The M3.5 f16-accumulator failure case (3.6e-3); must hold with f32 acc + caching. Exercises
/// the cross-superblock cache WAR (2 fills, the r3 hoisted unconditional RAW barrier each k_block).
#[test]
#[ignore]
fn at_1801_q4km_rb_coopmat_f32acc_cached_within_tol_k512() {
    let Some(e) = run_cached_f32acc("at_1801", 64, 128, 2) else { return; };
    assert!(
        e.combined <= FROZEN_REL_TOL,
        "AT-1801: K=512 cached f32-accumulator combined (condition-aware) max_rel_diff={:.3e} \
         exceeds frozen rtol {FROZEN_REL_TOL:.0e} — the f32 accumulator + caching MUST hold the \
         M3.5 f16 failure case (3.6e-3). If it fires, investigate the cache WAR / RAW barrier \
         and cache-index decode BEFORE touching the frozen 1e-3. (raw={:.3e})",
        e.combined, e.raw_rel
    );
    eprintln!(
        "at_1801: PASS — cached K=512 within frozen 1e-3 (combined={:.3e}, raw={:.3e}); \
         cross-superblock cache WAR correct",
        e.combined, e.raw_rel
    );
}

/// AT-1802: K=14336 (56 superblocks, inference K), M=N=64. The validity claim survives caching.
/// This is the test that would CATCH an r1-style accumulator reset (silent zeros -> combined ~= 1).
#[test]
#[ignore]
fn at_1802_q4km_rb_coopmat_f32acc_cached_within_tol_k14336() {
    let Some(e) = run_cached_f32acc("at_1802", 64, 64, 56) else { return; };
    assert!(
        e.combined <= FROZEN_REL_TOL,
        "AT-1802: K=14336 cached f32-accumulator combined (condition-aware) max_rel_diff={:.3e} \
         exceeds frozen rtol {FROZEN_REL_TOL:.0e} — the inference-K validity claim MUST survive \
         caching. If this fires, the most likely cause is a barrier/cache-index/loop-structure \
         bug (e.g. an r1-style silent-zeros accumulator reset would show combined ~= 1); \
         investigate that BEFORE touching the frozen 1e-3. (raw={:.3e})",
        e.combined, e.raw_rel
    );
    eprintln!(
        "at_1802: PASS — cached K=14336 (inference K) within frozen 1e-3 \
         (combined={:.3e}, raw={:.3e}); the cached kernel is NUMERICALLY VALID at inference K",
        e.combined, e.raw_rel
    );
}
