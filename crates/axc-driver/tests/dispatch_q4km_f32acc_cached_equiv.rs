//! M3.6 — CROSS-KERNEL BIT-IDENTITY equivalence test (AT-1803).
//!
//! The load-bearing correctness anchor for "dequant-scale caching is PURE reassociation".
//! Compiles BOTH the M3.5b kernel (q4km_matmul_rb_coopmat_f32acc.axc) AND the M3.6 cached
//! kernel (q4km_matmul_rb_coopmat_f32acc_cached.axc), dispatches BOTH on the SAME fixture at
//! K=256/512/14336, and asserts the two GPU outputs are BIT-IDENTICAL
//! (max |y_cached - y_baseline| == 0, compared on the raw u32 bit patterns).
//!
//! WHY BIT-IDENTITY IS THE CORRECT ANCHOR: the cached kernel keeps the M3.5b accumulation
//! order EXACTLY (same single-level OpPhi carry, same coopmat tile order) and the cached
//! per-nibble dequant is the SAME f32 multiply-then-subtract expression (M3.5b line 181) with
//! dsc/dmm READ from shared instead of recomputed — identical f32 bits. So the two GPU outputs
//! must match bit-for-bit. This is STRONGER than oracle agreement and is the DIRECT detector
//! of the r1 silent-zeros class (a per-superblock accumulator reset would drop earlier-K
//! contributions -> a non-zero diff) and of any barrier/cache-index slip.
//!
//! GPU-gated (#[ignore] + AXC_ENABLE_GPU_TESTS=1), typed-skip on Lavapipe / subgroup != 32.
//! This is the ORCHESTRATOR's real-NVIDIA gate, NOT a CI gate.

use std::collections::BTreeMap;
use axc_driver::compile_source_with_assignments;
use axc_hir::ParamBindingPlan;
use axc_runtime::{VulkanContext, DispatchError};

const BASELINE_SRC: &str = include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc.axc");
const CACHED_SRC: &str = include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached.axc");

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

/// Dispatch one kernel source on the given fixture; return the raw f32 output bytes.
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

/// Core: dispatch BOTH kernels on the same fixture, assert bit-identical output.
fn run_equiv(at: &str, m: usize, n: usize, n_bpr: usize) {
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

    // SAME fixture (same seed scheme) fed to BOTH kernels.
    let q_bytes = make_q4km_weights(m, n_bpr, 0xC0FFEE ^ (m as u64));
    let x_f16 = make_x_f16(k, n, 0xBADF00D ^ (n as u64));
    let x_bytes: Vec<u8> = x_f16.iter().flat_map(|&b| b.to_le_bytes()).collect();

    let Some(y_base) = dispatch_one(
        at, BASELINE_SRC, "q4km_matmul_rb_coopmat_f32acc",
        &q_bytes, &x_bytes, m, n, k, n_bpr, &ctx,
    ) else { return; };
    let Some(y_cached) = dispatch_one(
        at, CACHED_SRC, "q4km_matmul_rb_coopmat_f32acc_cached",
        &q_bytes, &x_bytes, m, n, k, n_bpr, &ctx,
    ) else { return; };

    assert_eq!(
        y_base.len(), y_cached.len(),
        "{at}: baseline/cached output length mismatch ({} vs {})", y_base.len(), y_cached.len()
    );

    // BIT-IDENTITY: compare raw f32 bit patterns (NOT a tolerance). Find the first/worst diff.
    let mut n_diff = 0usize;
    let mut first_diff: Option<(usize, f32, f32)> = None;
    for (i, (&a, &b)) in y_base.iter().zip(y_cached.iter()).enumerate() {
        if a.to_bits() != b.to_bits() {
            n_diff += 1;
            if first_diff.is_none() {
                first_diff = Some((i, a, b));
            }
        }
    }

    assert!(
        n_diff == 0,
        "AT-1803: cached kernel output is NOT bit-identical to the M3.5b baseline at \
         M={m} N={n} K={k}: {n_diff} of {} elements differ; first at idx {:?} \
         (baseline={:?}, cached={:?}). Caching is PURE reassociation with the SAME single-level \
         accumulation order and the SAME multiply-then-subtract dequant order, so the outputs \
         MUST match bit-for-bit. A non-zero diff means a STRUCTURAL deviation (missing/misplaced \
         barrier, cache-index mismatch, or an accumulation-order change — e.g. an r1-style \
         silent-zeros accumulator reset), NOT a permissible f32 reorder. STOP and investigate.",
        y_base.len(),
        first_diff.map(|(i, _, _)| i),
        first_diff.map(|(_, a, _)| a),
        first_diff.map(|(_, _, b)| b),
    );

    eprintln!(
        "at_1803: PASS — cached == baseline BIT-IDENTICAL at M={m} N={n} K={k} ({} elements) on {}",
        y_base.len(), ctx.physical_device_name()
    );
}

/// AT-1803: bit-identity at K=256 (1 superblock), M=N=64.
#[test]
#[ignore]
fn at_1803_q4km_f32acc_cached_equiv_baseline_k256() {
    run_equiv("at_1803_k256", 64, 64, 1);
}

/// AT-1803: bit-identity at K=512 (2 superblocks), M=64, N=128.
#[test]
#[ignore]
fn at_1803_q4km_f32acc_cached_equiv_baseline_k512() {
    run_equiv("at_1803_k512", 64, 128, 2);
}

/// AT-1803: bit-identity at K=14336 (56 superblocks, inference K), M=N=64.
/// The direct detector of an r1-style per-superblock accumulator reset.
#[test]
#[ignore]
fn at_1803_q4km_f32acc_cached_equiv_baseline_k14336() {
    run_equiv("at_1803_k14336", 64, 64, 56);
}
