//! M3.14b — GPU bit-within-tol correctness tests for the Q5_K_M dequant-scale-cached
//! f32-accumulator fused register-blocked coopmat matmul
//! (examples/q5km_matmul_rb_coopmat_f32acc_cached.axc) — the M3.6 leader extended verbatim
//! (§4.3 of the M3.14 spec) with the 176-byte stride + qh nibble->5-bit promotion.
//!
//! AT-2822: K=256 / 512 / 14336 (mirror AT-1800/1801/1802 — the cross-superblock cache WAR and
//!          the inference-K accumulator-reset catch), combined ≤ 1e-3 vs
//!          q5k_oracle::q5k_dequant_matmul_f32accum_cpu. NVIDIA-only, #[ignore]-gated.
//!          Typed-skip on Lavapipe (CoopMatUnsupported) and subgroup_size() != 32.
//! AT-2823: apples-to-apples cross-check of the coopmat Q5 kernel against
//!          q5k_dequant_matmul_f32accum_cpu (which rounds weights f32->f16->f32, matching the
//!          coopmat kernel's f32_to_f16 A-staging) within combined ≤ 1e-3 on a shared K=256
//!          fixture. Per the M3.14 spec §6/r2 FIX-2: this does NOT compare against the portable
//!          q5k_dequant_matmul.axc kernel (which dequants weights in FULL f32 — a cross-
//!          PRECISION comparison ~5e-4, not the ~1e-6 AT-1803 bit-identity class). Cross-decode
//!          consistency of the two Q5 paths is already proven transitively by AT-2819 (portable
//!          bit-exact vs oracle) + AT-2822 (coopmat combined vs the same oracle family). Both
//!          metrics reported. NVIDIA-only, #[ignore]-gated.

use axc_driver::q5k_oracle;
use std::collections::BTreeMap;
use axc_driver::compile_source_with_assignments;
use axc_hir::ParamBindingPlan;
use axc_runtime::{VulkanContext, DispatchError};
use half::f16;

const CACHED_Q5K_SRC: &str =
    include_str!("../../../examples/q5km_matmul_rb_coopmat_f32acc_cached.axc");

/// Frozen relative tolerance (NOT loosened).
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

/// Build a NON-symmetric Q5_K_M weight matrix fixture (M rows × n_bpr superblocks, 176 bytes
/// each). Same seed scheme style as the Q4_K_M M3.6 dispatch tests.
fn make_q5k_weights(m: usize, n_bpr: usize, seed: u64) -> Vec<u8> {
    let mut q = vec![0u8; m * n_bpr * q5k_oracle::Q5K_SUPERBLOCK_BYTES];
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for row in 0..m {
        for sb in 0..n_bpr {
            let base = (row * n_bpr + sb) * q5k_oracle::Q5K_SUPERBLOCK_BYTES;
            let d = 0.02_f32 + ((next() % 16) as f32) * 0.002;
            let dmin = 0.01_f32 + ((next() % 8) as f32) * 0.001;
            q[base..base + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
            q[base + 2..base + 4].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
            for j in 0..12 {
                q[base + 4 + j] = (next() & 0x3F) as u8;
            }
            for i in 0..32 {
                q[base + 16 + i] = (next() & 0xFF) as u8; // qh — NEW vs Q4_K_M.
            }
            for i in 0..128 {
                q[base + 48 + i] = (next() & 0xFF) as u8; // qs.
            }
        }
    }
    q
}

/// Build a NON-symmetric f16 activation matrix x[K, N] (row-major, f16 bit patterns).
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

struct MeasuredErr {
    raw_rel: f64,
    combined: f64,
}

/// Core: dispatch the Q5_K_M cached kernel and compare vs the f32-acc oracle.
fn run_cached_q5k(at: &str, m: usize, n: usize, n_bpr: usize) -> Option<MeasuredErr> {
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

    let k: usize = n_bpr * q5k_oracle::Q5K_SUPERBLOCK_ELEMS;
    assert!(m.is_multiple_of(32) && n.is_multiple_of(32), "{at}: M and N must be multiples of 32");

    let q_bytes = make_q5k_weights(m, n_bpr, 0xC0FFEE ^ (m as u64) ^ 0xA5A5);
    let x_f16 = make_x_f16(k, n, 0xBADF00D ^ (n as u64) ^ 0xA5A5);
    let x_bytes: Vec<u8> = x_f16.iter().flat_map(|&b| b.to_le_bytes()).collect();

    let y_ref = q5k_oracle::q5k_dequant_matmul_f32accum_cpu(&q_bytes, &x_f16, m, n, n_bpr);

    let assignments = rb2x2_assignments();
    let (bytes, meta) = compile_source_with_assignments(CACHED_Q5K_SRC, &assignments)
        .expect("q5km_matmul_rb_coopmat_f32acc_cached.axc must compile");
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let handle = match ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, meta.coopmat.as_ref(), "q5km_matmul_rb_coopmat_f32acc_cached",
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
    let c_size: usize = m * n * 4;
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

    let raw_rel = axc_driver::q4km_oracle::max_rel_diff(&y_gpu, &y_ref);
    let abs_scale = q5k_oracle::q5k_abs_dot_scale(&q_bytes, &x_f16, m, n, n_bpr);
    let combined = axc_driver::q4km_oracle::max_rel_diff_combined(&y_gpu, &y_ref, &abs_scale);
    eprintln!(
        "{at}: MEASURED raw max_rel_diff={raw_rel:.3e} | COMBINED (condition-aware, the GATE) \
         ={combined:.3e} (M={m} N={n} K={k}, frozen rtol={FROZEN_REL_TOL:.0e}) on {}",
        ctx.physical_device_name()
    );
    Some(MeasuredErr { raw_rel, combined })
}

/// AT-2822: K=256 (1 superblock), M=N=64.
#[test]
#[ignore]
fn at_2822_q5km_rb_coopmat_f32acc_cached_within_tol_k256() {
    let Some(e) = run_cached_q5k("at_2822_k256", 64, 64, 1) else { return; };
    assert!(
        e.combined <= FROZEN_REL_TOL,
        "AT-2822: K=256 Q5_K_M cached combined (condition-aware) max_rel_diff={:.3e} exceeds \
         frozen rtol {FROZEN_REL_TOL:.0e} (raw={:.3e})", e.combined, e.raw_rel
    );
    eprintln!("at_2822_k256: PASS — combined={:.3e}, raw={:.3e}", e.combined, e.raw_rel);
}

/// AT-2822: K=512 (2 superblocks), M=64, N=128. Exercises the cross-superblock cache WAR.
#[test]
#[ignore]
fn at_2822_q5km_rb_coopmat_f32acc_cached_within_tol_k512() {
    let Some(e) = run_cached_q5k("at_2822_k512", 64, 128, 2) else { return; };
    assert!(
        e.combined <= FROZEN_REL_TOL,
        "AT-2822: K=512 Q5_K_M cached combined (condition-aware) max_rel_diff={:.3e} exceeds \
         frozen rtol {FROZEN_REL_TOL:.0e} (raw={:.3e})", e.combined, e.raw_rel
    );
    eprintln!("at_2822_k512: PASS — combined={:.3e}, raw={:.3e}; cross-superblock cache WAR correct", e.combined, e.raw_rel);
}

/// AT-2822: K=14336 (56 superblocks, inference K), M=N=64. Catches an r1-style accumulator reset.
#[test]
#[ignore]
fn at_2822_q5km_rb_coopmat_f32acc_cached_within_tol_k14336() {
    let Some(e) = run_cached_q5k("at_2822_k14336", 64, 64, 56) else { return; };
    assert!(
        e.combined <= FROZEN_REL_TOL,
        "AT-2822: K=14336 Q5_K_M cached combined (condition-aware) max_rel_diff={:.3e} exceeds \
         frozen rtol {FROZEN_REL_TOL:.0e} (raw={:.3e})", e.combined, e.raw_rel
    );
    eprintln!("at_2822_k14336: PASS — combined={:.3e}, raw={:.3e}", e.combined, e.raw_rel);
}

/// AT-2823: apples-to-apples cross-check vs q5k_dequant_matmul_f32accum_cpu at K=256 (r2 FIX-2 —
/// NOT a comparison against the portable q5k_dequant_matmul.axc kernel; see the module header).
#[test]
#[ignore]
fn at_2823_q5km_rb_coopmat_f32acc_cached_apples_to_apples_k256() {
    let Some(e) = run_cached_q5k("at_2823_k256", 64, 64, 1) else { return; };
    assert!(
        e.combined <= FROZEN_REL_TOL,
        "AT-2823: apples-to-apples (f16-weight-rounded oracle) combined (condition-aware) \
         max_rel_diff={:.3e} exceeds frozen rtol {FROZEN_REL_TOL:.0e} at K=256 (raw={:.3e})",
        e.combined, e.raw_rel
    );
    eprintln!(
        "at_2823_k256: PASS — coopmat Q5 kernel matches the f16-weight-rounded oracle within \
         frozen 1e-3 (combined={:.3e}, raw={:.3e}); cross-decode consistency with the portable \
         kernel is proven TRANSITIVELY via AT-2819 + AT-2822 (see module header, r2 FIX-2)",
        e.combined, e.raw_rel
    );
}
