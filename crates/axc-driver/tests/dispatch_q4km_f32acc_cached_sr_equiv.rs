//! M3.11a — CROSS-KERNEL BIT-IDENTITY (AT-2603, the AUTHORITATIVE off-by-one falsifier) +
//! COMBINED condition-aware metric (AT-2605).
//!
//! AT-2603 (the AUTHORITATIVE, LOAD-BEARING correctness gate): the dequant-index STRENGTH-REDUCED
//! kernel (q4km_matmul_rb_coopmat_f32acc_cached_sr.axc) must be BIT-IDENTICAL
//! (max|y_sr - y_cached|.to_bits() == 0) to the M3.6 leader (cached.axc) on the ASYMMETRIC,
//! position-varying `make_q4km_weights` fixture (the seeded per-row/col-varying weights — IDENTICAL
//! to dispatch_q4km_f32acc_cached_equiv.rs:58/185) at a multi-superblock shape. The SR kernel reads
//! the IDENTICAL nibble via carried counters instead of recomputed div/mod, so the f32 dequant and
//! the coopmat accumulation order are unchanged => the outputs MUST match bit-for-bit. A wrong-nibble
//! carry (wrong delta, wrong wrap boundary, a destroyed/re-let carry that yields a wrong output)
//! DIVERGES from the M3.6 leader at the output, and the asymmetric fixture prevents any pass-by-
//! symmetry. THIS binds the EMITTED KERNEL (AT-2601 is the subordinate CPU design pre-filter).
//!
//! AT-2605: the COMBINED condition-aware metric |gpu-ref|/max(|ref|, Σ|wₖxₖ|) <= FROZEN 1e-3 at
//! K=256/512/14336 vs the `axc_driver::q4km_oracle` lib module (single source of truth). FROZEN 1e-3
//! NEVER loosens. Expected ~bit-identical (4e-7 class) since the formula/order are unchanged.
//!
//! GPU-gated (#[ignore] + AXC_ENABLE_GPU_TESTS=1), typed-skip on Lavapipe / subgroup != 32.
//! This is the ORCHESTRATOR's real-NVIDIA gate, NOT a CI gate (anti-pattern #9).

use std::collections::BTreeMap;
use axc_driver::compile_source_with_assignments;
use axc_driver::q4km_oracle;
use axc_hir::ParamBindingPlan;
use axc_runtime::{VulkanContext, DispatchError};

const CACHED_SRC: &str = include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached.axc");
const SR_SRC: &str = include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached_sr.axc");

/// Frozen relative tolerance (AT-2605 — NEVER loosened).
const FROZEN_REL_TOL: f64 = 1e-3;

fn gpu_tests_enabled() -> bool {
    std::env::var("AXC_ENABLE_GPU_TESTS").as_deref() == Ok("1")
}

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

/// ASYMMETRIC, position-varying Q4_K_M weights — IDENTICAL to dispatch_q4km_f32acc_cached_equiv.rs:58.
/// The per-row/col-varying seed makes a wrong-nibble carry diverge (no pass-by-symmetry).
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

/// Dispatch one kernel source on the given fixture; return the raw f32 output.
/// Returns None on a typed-skip (no GPU / coopmat / subgroup != 32).
#[allow(clippy::too_many_arguments)]
fn dispatch_one(
    at: &str,
    src: &str,
    kernel_name: &str,
    q_bytes: &[u8],
    x_bytes: &[u8],
    m: usize,
    n: usize,
    k: usize,
    n_bpr: usize,
    ctx: &VulkanContext,
) -> Option<Vec<f32>> {
    let assignments = rb2x2_assignments();
    let (bytes, meta) = compile_source_with_assignments(src, &assignments)
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
        &handle, workgroups,
        &[q_bytes, x_bytes, &vec![0u8; c_size]],
        &[0, 0, c_size],
        &pc,
    ) {
        Ok(v) => v,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("{at}: {kernel_name} CoopMatUnsupported at dispatch (typed-skip): {reason}");
            return None;
        }
        Err(e) => panic!("{at}: {kernel_name} dispatch failed: {e:?}"),
    };

    let c_bytes: &[u8] = &outputs[2];
    assert_eq!(c_bytes.len(), c_size, "{at}: {kernel_name} C output size mismatch");
    let y: Vec<f32> = c_bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Some(y)
}

/// AT-2603 core: dispatch BOTH the SR and the M3.6 leader on the SAME asymmetric fixture, assert
/// bit-identical output.
fn run_bit_identity(at: &str, m: usize, n: usize, n_bpr: usize) {
    if !gpu_tests_enabled() {
        eprintln!("{at}: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    let ctx = VulkanContext::new().expect("VulkanContext must init");
    eprintln!("{at}: device={}", ctx.physical_device_name());

    if !ctx.coopmat_support().feature_present {
        eprintln!("{at}: coopmat not supported on {}; typed-skip", ctx.physical_device_name());
        return;
    }
    if ctx.subgroup_size() != 32 {
        eprintln!("{at}: subgroup_size()={} != 32; typed-skip", ctx.subgroup_size());
        return;
    }

    let k: usize = n_bpr * 256;
    assert!(m.is_multiple_of(32) && n.is_multiple_of(32), "{at}: M and N must be multiples of 32");

    // SAME asymmetric fixture (same seed scheme) fed to BOTH kernels.
    let q_bytes = make_q4km_weights(m, n_bpr, 0xC0FFEE ^ (m as u64));
    let x_f16 = make_x_f16(k, n, 0xBADF00D ^ (n as u64));
    let x_bytes: Vec<u8> = x_f16.iter().flat_map(|&b| b.to_le_bytes()).collect();

    let Some(y_cached) = dispatch_one(
        at, CACHED_SRC, "q4km_matmul_rb_coopmat_f32acc_cached",
        &q_bytes, &x_bytes, m, n, k, n_bpr, &ctx,
    ) else { return; };
    let Some(y_sr) = dispatch_one(
        at, SR_SRC, "q4km_matmul_rb_coopmat_f32acc_cached_sr",
        &q_bytes, &x_bytes, m, n, k, n_bpr, &ctx,
    ) else { return; };

    assert_eq!(
        y_cached.len(), y_sr.len(),
        "{at}: cached/sr output length mismatch ({} vs {})", y_cached.len(), y_sr.len()
    );

    // BIT-IDENTITY: compare raw f32 bit patterns (NOT a tolerance). Find the first/worst diff.
    let mut n_diff = 0usize;
    let mut first_diff: Option<(usize, f32, f32)> = None;
    for (i, (&a, &b)) in y_cached.iter().zip(y_sr.iter()).enumerate() {
        if a.to_bits() != b.to_bits() {
            n_diff += 1;
            if first_diff.is_none() {
                first_diff = Some((i, a, b));
            }
        }
    }

    assert!(
        n_diff == 0,
        "AT-2603: the SR kernel output is NOT bit-identical to the M3.6 leader at \
         M={m} N={n} K={k}: {n_diff} of {} elements differ; first at idx {:?} \
         (cached={:?}, sr={:?}). The strength-reduction is a PURE integer-index reformulation that \
         reads the IDENTICAL nibble with the SAME f32 dequant + coopmat accumulation order, so the \
         outputs MUST match bit-for-bit. A non-zero diff means a WRONG-NIBBLE CARRY (wrong delta, \
         wrong wrap boundary, or a destroyed/re-let carry) — debug the counter init/advance. The \
         asymmetric fixture prevents any pass-by-symmetry. STOP and investigate.",
        y_cached.len(),
        first_diff.map(|(i, _, _)| i),
        first_diff.map(|(_, a, _)| a),
        first_diff.map(|(_, _, b)| b),
    );

    eprintln!(
        "at_2603: PASS — SR == M3.6 leader BIT-IDENTICAL at M={m} N={n} K={k} ({} elements) on {}",
        y_cached.len(), ctx.physical_device_name()
    );
}

/// AT-2605 core: dispatch the SR kernel and assert the COMBINED condition-aware metric vs the
/// f32-accumulator oracle is <= FROZEN 1e-3.
fn run_combined_metric(at: &str, m: usize, n: usize, n_bpr: usize) {
    if !gpu_tests_enabled() {
        eprintln!("{at}: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    let ctx = VulkanContext::new().expect("VulkanContext must init");
    eprintln!("{at}: device={}", ctx.physical_device_name());

    if !ctx.coopmat_support().feature_present {
        eprintln!("{at}: coopmat not supported on {}; typed-skip", ctx.physical_device_name());
        return;
    }
    if ctx.subgroup_size() != 32 {
        eprintln!("{at}: subgroup_size()={} != 32; typed-skip", ctx.subgroup_size());
        return;
    }

    let k: usize = n_bpr * 256;
    let q_bytes = make_q4km_weights(m, n_bpr, 0xC0FFEE ^ (m as u64));
    let x_f16 = make_x_f16(k, n, 0xBADF00D ^ (n as u64));
    let x_bytes: Vec<u8> = x_f16.iter().flat_map(|&b| b.to_le_bytes()).collect();

    let Some(y_sr) = dispatch_one(
        at, SR_SRC, "q4km_matmul_rb_coopmat_f32acc_cached_sr",
        &q_bytes, &x_bytes, m, n, k, n_bpr, &ctx,
    ) else { return; };

    // The f32-accumulator oracle (single source of truth: axc_driver::q4km_oracle).
    let y_ref = q4km_oracle::q4km_dequant_matmul_f32accum_cpu(&q_bytes, &x_f16, m, n, n_bpr);
    let abs_scale = q4km_oracle::q4km_abs_dot_scale(&q_bytes, &x_f16, m, n, n_bpr);
    let combined = q4km_oracle::max_rel_diff_combined(&y_sr, &y_ref, &abs_scale);
    let raw = q4km_oracle::max_rel_diff(&y_sr, &y_ref);

    assert!(
        combined <= FROZEN_REL_TOL,
        "AT-2605: SR combined condition-aware metric {combined:.3e} > FROZEN {FROZEN_REL_TOL:.0e} \
         at M={m} N={n} K={k} (raw={raw:.3e}). The dequant formula + accumulation order are \
         unchanged, so this must stay within the FROZEN tolerance (expected ~bit-identical, \
         4e-7 class). FROZEN 1e-3 NEVER loosens."
    );

    eprintln!(
        "at_2605: PASS — SR combined={combined:.3e} <= {FROZEN_REL_TOL:.0e} (raw={raw:.3e}) at \
         M={m} N={n} K={k} on {}",
        ctx.physical_device_name()
    );
}

// ── AT-2603: bit-identity vs the M3.6 leader on the asymmetric fixture (multi-superblock shapes) ──

/// AT-2603: bit-identity at K=512 (2 superblocks), M=64, N=128 — multiple superblocks exercise the
/// carry wrap. A wrong-wrap/off-by-one carry diverges here.
#[test]
#[ignore]
fn at_2603_sr_bit_identical_to_cached_asymmetric_k512() {
    run_bit_identity("at_2603_k512", 64, 128, 2);
}

/// AT-2603: bit-identity at K=14336 (56 superblocks, inference K), M=N=64 — the full-range carry
/// across 55 superblock wraps. The strongest off-by-one falsifier.
#[test]
#[ignore]
fn at_2603_sr_bit_identical_to_cached_asymmetric_k14336() {
    run_bit_identity("at_2603_k14336", 64, 64, 56);
}

/// AT-2603: bit-identity at K=256 (1 superblock), M=N=64 — the single-superblock anchor (no wrap).
#[test]
#[ignore]
fn at_2603_sr_bit_identical_to_cached_asymmetric_k256() {
    run_bit_identity("at_2603_k256", 64, 64, 1);
}

// ── AT-2605: combined condition-aware metric <= FROZEN 1e-3 at K=256/512/14336 ──

#[test]
#[ignore]
fn at_2605_sr_combined_metric_at_k256() {
    run_combined_metric("at_2605_k256", 64, 64, 1);
}

#[test]
#[ignore]
fn at_2605_sr_combined_metric_at_k512() {
    run_combined_metric("at_2605_k512", 64, 128, 2);
}

#[test]
#[ignore]
fn at_2605_sr_combined_metric_at_k14336() {
    run_combined_metric("at_2605_k14336", 64, 64, 56);
}
