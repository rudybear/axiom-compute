//! M3.7 — GPU bit-within-tol correctness tests for the DOUBLE-BUFFERED (software-pipelined)
//! dequant-scale-cached f32-accumulator fused Q4_K_M register-blocked coopmat matmul
//! (examples/q4km_matmul_rb_coopmat_f32acc_db.axc).
//!
//! AT-1900: K=256 (1 superblock = 16 tile_k K-blocks), M=N=64. ASSERTED within frozen 1e-3.
//! AT-1901: K=512 (2 superblocks = 32 tile_k K-blocks), M=64, N=128. ASSERTED UNCONDITIONALLY
//!          within frozen 1e-3 — the M3.5 f16-accumulator failure case, must hold with f32 acc
//!          + caching + double-buffering. Exercises the cross-superblock cache WAR (2 fills) and
//!          the ping-pong buffer WAR (the 2-barrier scheme).
//! AT-1902: K=14336 (56 superblocks, the inference-K A/B shape), M=N=64. The validity claim
//!          survives double-buffering. CATCHES an r1-style accumulator reset (silent zeros ->
//!          combined ~= 1 >> 1e-3) AND a missing-barrier race that perturbs accumulation.
//! AT-1906 (degenerate): K=16 (num_k_blocks==1), M=N=64. The degenerate pipeline — the prologue
//!          stages buffer 0, the single iteration computes it, the kn<num_k_blocks guard skips the
//!          prefetch entirely (no over-read past K).
//!
//! Double-buffering is PURE SCHEDULING of M3.6 (same arithmetic, same coopmat tile order, same
//! single-level OpPhi accumulation order), so the combined condition-aware diff vs the
//! f32-accumulator oracle (common_q4km_f32ref::q4km_dequant_matmul_f32accum_cpu) MUST stay within
//! the FROZEN 1e-3 (NOT loosened). The cross-kernel BIT-IDENTITY anchor is AT-1903
//! (dispatch_q4km_f32acc_db_equiv.rs) — the load-bearing missing-barrier-race detector.
//!
//! GATE METRIC: the CONDITION-AWARE diff `|gpu-ref| / max(|ref|, sum_k|w_k·x_k|)` (identical to
//! AT-1800/1802), NOT the raw `|gpu-ref|/|ref|`. Both are REPORTED. The 1e-3 NEVER loosens.
//!
//! NVIDIA #[ignore]-gated (AXC_ENABLE_GPU_TESTS=1). Typed-skip on Lavapipe
//! (CoopMatUnsupported / DeviceFeatureUnsupported) and on subgroup_size() != 32.

// M4.1p2: the oracle is now the pub lib module axc_driver::q4km_oracle (single source of truth).
use axc_driver::q4km_oracle as common_q4km_f32ref;

use std::collections::BTreeMap;
use axc_driver::compile_source_with_assignments;
use axc_hir::ParamBindingPlan;
use axc_runtime::{VulkanContext, DispatchError};

const DB_F32ACC_SRC: &str =
    include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_db.axc");

/// Frozen relative tolerance (AT-1520/AT-1521 value — NOT loosened).
const FROZEN_REL_TOL: f64 = 1e-3;

fn gpu_tests_enabled() -> bool {
    std::env::var("AXC_ENABLE_GPU_TESTS").as_deref() == Ok("1")
}

/// RB 2×2 strategy assignments (a_block_size_db/b_block_size_db PINNED at 1024 = 2*512).
fn rb2x2_db_assignments() -> BTreeMap<String, i64> {
    let mut m = BTreeMap::new();
    m.insert("rb_m".to_owned(), 2_i64);
    m.insert("rb_n".to_owned(), 2_i64);
    m.insert("tile_k".to_owned(), 16_i64);
    m.insert("a_block_size_db".to_owned(), 1024_i64);
    m.insert("b_block_size_db".to_owned(), 1024_i64);
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
/// IDENTICAL seed scheme to dispatch_q4km_matmul_rb_f32acc_cached.rs (so AT-1903 can compare
/// the two kernels on the SAME fixture).
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

/// Core: dispatch the DOUBLE-BUFFERED f32-accumulator fused kernel and compare vs the oracle.
/// Returns the measured errors, or None on a typed-skip (no GPU/coopmat).
fn run_db_f32acc(at: &str, m: usize, n: usize, n_bpr: usize) -> Option<MeasuredErr> {
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

    let q_bytes = make_q4km_weights(m, n_bpr, 0xC0FFEE ^ (m as u64));
    let x_f16 = make_x_f16(k, n, 0xBADF00D ^ (n as u64));
    let x_bytes: Vec<u8> = x_f16.iter().flat_map(|&b| b.to_le_bytes()).collect();

    let y_ref = common_q4km_f32ref::q4km_dequant_matmul_f32accum_cpu(&q_bytes, &x_f16, m, n, n_bpr);

    let assignments = rb2x2_db_assignments();
    let (bytes, meta) = compile_source_with_assignments(DB_F32ACC_SRC, &assignments)
        .expect("q4km_matmul_rb_coopmat_f32acc_db.axc must compile");
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let handle = match ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, meta.coopmat.as_ref(), "q4km_matmul_rb_coopmat_f32acc_db",
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
    let c_size: usize = m * n * 4; // f32 output (4 bytes/elem).
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

/// AT-1900: K=256 (1 superblock), M=N=64. Within-tol ASSERTED.
#[test]
#[ignore]
fn at_1900_q4km_rb_coopmat_f32acc_db_within_tol_k256() {
    let Some(e) = run_db_f32acc("at_1900", 64, 64, 1) else { return; };
    assert!(
        e.combined <= FROZEN_REL_TOL,
        "AT-1900: K=256 double-buffered f32-accumulator combined (condition-aware) max_rel_diff \
         ={:.3e} exceeds frozen rtol {FROZEN_REL_TOL:.0e} (double-buffering is pure scheduling; \
         the frozen 1e-3 is NOT loosened; raw={:.3e})",
        e.combined, e.raw_rel
    );
    eprintln!(
        "at_1900: PASS — double-buffered K=256 within frozen 1e-3 (combined={:.3e}, raw={:.3e})",
        e.combined, e.raw_rel
    );
}

/// AT-1901: K=512 (2 superblocks), M=64, N=128. Within-tol ASSERTED UNCONDITIONALLY.
/// Exercises the cross-superblock cache WAR (2 fills) + the ping-pong buffer WAR.
#[test]
#[ignore]
fn at_1901_q4km_rb_coopmat_f32acc_db_within_tol_k512() {
    let Some(e) = run_db_f32acc("at_1901", 64, 128, 2) else { return; };
    assert!(
        e.combined <= FROZEN_REL_TOL,
        "AT-1901: K=512 double-buffered f32-accumulator combined (condition-aware) max_rel_diff \
         ={:.3e} exceeds frozen rtol {FROZEN_REL_TOL:.0e} — the f32 accumulator + caching + \
         double-buffering MUST hold the M3.5 f16 failure case. If it fires, investigate the \
         ping-pong buffer WAR (B1) / the cross-superblock cache WAR (B2) BEFORE touching the \
         frozen 1e-3. (raw={:.3e})",
        e.combined, e.raw_rel
    );
    eprintln!(
        "at_1901: PASS — double-buffered K=512 within frozen 1e-3 (combined={:.3e}, raw={:.3e}); \
         ping-pong + cross-superblock cache WAR correct",
        e.combined, e.raw_rel
    );
}

/// AT-1902: K=14336 (56 superblocks, inference K), M=N=64. The validity claim survives
/// double-buffering. CATCHES an r1-style accumulator reset (silent zeros -> combined ~= 1).
#[test]
#[ignore]
fn at_1902_q4km_rb_coopmat_f32acc_db_within_tol_k14336() {
    let Some(e) = run_db_f32acc("at_1902", 64, 64, 56) else { return; };
    assert!(
        e.combined <= FROZEN_REL_TOL,
        "AT-1902: K=14336 double-buffered f32-accumulator combined (condition-aware) max_rel_diff \
         ={:.3e} exceeds frozen rtol {FROZEN_REL_TOL:.0e} — the inference-K validity claim MUST \
         survive double-buffering. A barrier/parity/loop-structure bug (e.g. an r1-style \
         silent-zeros accumulator reset -> combined ~= 1, or a ping-pong slip) is the likely \
         cause; investigate that BEFORE touching the frozen 1e-3. (raw={:.3e})",
        e.combined, e.raw_rel
    );
    eprintln!(
        "at_1902: PASS — double-buffered K=14336 (inference K) within frozen 1e-3 \
         (combined={:.3e}, raw={:.3e}); the double-buffered kernel is NUMERICALLY VALID at \
         inference K",
        e.combined, e.raw_rel
    );
}

/// Dispatch one kernel source on the given fixture; return the raw f32 output. None on typed-skip.
/// Used by the degenerate K=16 bit-identity check below (the oracle requires K=n_bpr*256, so the
/// genuine num_k_blocks==1 case is validated against the M3.6 cached kernel, not the oracle).
#[allow(clippy::too_many_arguments)]
fn dispatch_kernel(
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
    let c_size: usize = m * n * 4;
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
    let c_bytes: &[u8] = &outputs[2];
    Some(c_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// AT-1906 (degenerate pipeline, num_k_blocks==1): K=16, M=N=64. The prologue stages buffer 0,
/// the single loop iteration computes it, and the kn<num_k_blocks guard (kn=1 not < 1) skips the
/// prefetch entirely — no over-read past K, no under-compute.
///
/// The f32-accumulator oracle requires K = n_bpr*256, which K=16 does not satisfy. Both kernels,
/// however, decode superblocks directly from K and run identically at K=16 (num_k_blocks=1), so
/// the genuine degenerate pipeline is validated by BIT-IDENTITY vs the M3.6 cached kernel
/// (max|y_db - y_cached| == 0) — which is also the load-bearing race/over-read detector for the
/// degenerate path. The single-superblock FULL-pipeline correctness is covered by AT-1900 (K=256).
#[test]
#[ignore]
fn at_1906_q4km_rb_coopmat_f32acc_db_degenerate_k16() {
    if !gpu_tests_enabled() {
        eprintln!("at_1906_degen: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    let ctx = VulkanContext::new().expect("VulkanContext must init");
    eprintln!("at_1906_degen: device={}", ctx.physical_device_name());
    if !ctx.coopmat_support().feature_present {
        eprintln!("at_1906_degen: coopmat not supported; typed-skip");
        return;
    }
    if ctx.subgroup_size() != 32 {
        eprintln!("at_1906_degen: subgroup_size()={} != 32; typed-skip", ctx.subgroup_size());
        return;
    }

    let (m, n, k) = (64usize, 64usize, 16usize); // K=16 -> num_k_blocks = 16/16 = 1 (degenerate).
    // n_bpr=1 generates a full 144-byte superblock fixture; K=16 reads only sub-block 0's first
    // 16 nibbles (in-bounds of that superblock) — a valid Q4_K_M weight layout.
    let n_bpr = 1usize;
    let q_bytes = make_q4km_weights(m, n_bpr, 0xC0FFEE ^ (m as u64));
    let x_f16 = make_x_f16(k, n, 0xBADF00D ^ (n as u64));
    let x_bytes: Vec<u8> = x_f16.iter().flat_map(|&b| b.to_le_bytes()).collect();

    const CACHED_SRC: &str =
        include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached.axc");
    let mut cached_a = BTreeMap::new();
    cached_a.insert("rb_m".to_owned(), 2_i64);
    cached_a.insert("rb_n".to_owned(), 2_i64);
    cached_a.insert("tile_k".to_owned(), 16_i64);
    cached_a.insert("a_block_size".to_owned(), 512_i64);
    cached_a.insert("b_block_size".to_owned(), 512_i64);

    let Some(y_cached) = dispatch_kernel(
        "at_1906_degen", CACHED_SRC, "q4km_matmul_rb_coopmat_f32acc_cached", &cached_a,
        &q_bytes, &x_bytes, m, n, k, n_bpr, &ctx,
    ) else { return; };
    let Some(y_db) = dispatch_kernel(
        "at_1906_degen", DB_F32ACC_SRC, "q4km_matmul_rb_coopmat_f32acc_db", &rb2x2_db_assignments(),
        &q_bytes, &x_bytes, m, n, k, n_bpr, &ctx,
    ) else { return; };

    assert_eq!(y_cached.len(), y_db.len(), "at_1906_degen: output length mismatch");
    let n_diff = y_cached.iter().zip(y_db.iter())
        .filter(|(&c, &d)| c.to_bits() != d.to_bits()).count();
    assert!(
        n_diff == 0,
        "AT-1906 degenerate (K=16, num_k_blocks==1): db output NOT bit-identical to the M3.6 cached \
         kernel ({n_diff} of {} elements differ). The degenerate pipeline (prologue stages buffer \
         0; the single iteration computes it; the kn<num_k_blocks guard skips the prefetch) must \
         match cached exactly — a non-zero diff is an over-read past K or a degenerate-path slip.",
        y_cached.len()
    );
    eprintln!(
        "at_1906_degen: PASS — db == cached BIT-IDENTICAL at degenerate K=16 (num_k_blocks==1, \
         {} elements) on {}",
        y_cached.len(), ctx.physical_device_name()
    );
}
