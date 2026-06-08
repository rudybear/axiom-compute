//! M3.8 — CROSS-KERNEL BIT-IDENTITY equivalence (AT-2003): each LARGER-REGISTER-TILE variant
//! (4x2 / 4x4) vs the M3.6 2x2 scale-cached leader on the SAME fixture at K=256/512/14336.
//!
//! WHY BIT-IDENTITY IS THE EXPECTED FIRST-LINE GATE: larger register blocking adds MORE output
//! tiles per workgroup but does NOT regroup the accumulation WITHIN any 16x16 output tile — each
//! output element accumulates the SAME K-depth dot in the SAME single-level flat-K-loop order with
//! the SAME per-tile coopmat decomposition and the SAME dequant expression (dsc/dmm read from the
//! RESIZED 512-entry cache) as M3.6. Therefore each output element's accumulation order is
//! byte-identical -> n_diff==0 is EXPECTED. A non-zero diff is a STRUCTURAL-deviation signal
//! (wrong cache-resize index, wrong A-row/B-col coopmat offset/stride, or accumulation-order
//! change) — INVESTIGATE.
//!
//! NR-1 FALLBACK (reported, not auto-failed here): if the driver spuriously reassociates HMMA
//! across the larger register tile and produces a byte-diff DESPITE the COMBINED condition-aware
//! metric being <= FROZEN 1e-3 (AT-2000/2001/2002), the combined metric is the AUTHORITATIVE
//! correctness floor and the bit-diff is a driver-reassociation artifact, NOT a kernel bug. This
//! test ASSERTS n_diff==0 (the expected gate) AND prints the combined metric vs M3.6 so the
//! orchestrator can apply the NR-1 fallback judgment. The frozen 1e-3 NEVER loosens.
//!
//! NR-2 PER-VARIANT guard: tightened from the inherited %32 to per-variant divisibility
//! (4x2: M%64==0 && N%32==0; 4x4: M%64==0 && N%64==0). The chosen M,N are multiples of 64 (both
//! grids tile cleanly AND >1 workgroup so the 64-row block-row remapping is exercised).
//!
//! NVIDIA #[ignore]-gated (AXC_ENABLE_GPU_TESTS=1). Typed-skip on Lavapipe / subgroup != 32.

// M4.1p2: the oracle is now the pub lib module axc_driver::q4km_oracle (single source of truth).
use axc_driver::q4km_oracle as common_q4km_f32ref;

use std::collections::BTreeMap;
use axc_driver::compile_source_with_assignments;
use axc_hir::ParamBindingPlan;
use axc_runtime::{VulkanContext, DispatchError};

const CACHED_2X2_SRC: &str =
    include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached.axc");
const SRC_4X2: &str =
    include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached_4x2.axc");
const SRC_4X4: &str =
    include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached_4x4.axc");

const FROZEN_REL_TOL: f64 = 1e-3;

fn gpu_tests_enabled() -> bool {
    std::env::var("AXC_ENABLE_GPU_TESTS").as_deref() == Ok("1")
}

/// One dispatchable kernel: source + name + @strategy assignments + grid tile dims.
struct Kernel {
    src: &'static str,
    name: &'static str,
    assignments: BTreeMap<String, i64>,
    tile_rows: usize,
    tile_cols: usize,
}

fn cached_2x2() -> Kernel {
    let mut a = BTreeMap::new();
    a.insert("rb_m".to_owned(), 2_i64);
    a.insert("rb_n".to_owned(), 2_i64);
    a.insert("tile_k".to_owned(), 16_i64);
    a.insert("a_block_size".to_owned(), 512_i64);
    a.insert("b_block_size".to_owned(), 512_i64);
    Kernel { src: CACHED_2X2_SRC, name: "q4km_matmul_rb_coopmat_f32acc_cached",
        assignments: a, tile_rows: 32, tile_cols: 32 }
}

fn bigrb_4x2() -> Kernel {
    let mut a = BTreeMap::new();
    a.insert("rb_m".to_owned(), 4_i64);
    a.insert("rb_n".to_owned(), 2_i64);
    a.insert("tile_k".to_owned(), 16_i64);
    a.insert("a_block_size".to_owned(), 1024_i64);
    a.insert("b_block_size".to_owned(), 512_i64);
    Kernel { src: SRC_4X2, name: "q4km_matmul_rb_coopmat_f32acc_cached_4x2",
        assignments: a, tile_rows: 64, tile_cols: 32 }
}

fn bigrb_4x4() -> Kernel {
    let mut a = BTreeMap::new();
    a.insert("rb_m".to_owned(), 4_i64);
    a.insert("rb_n".to_owned(), 4_i64);
    a.insert("tile_k".to_owned(), 16_i64);
    a.insert("a_block_size".to_owned(), 1024_i64);
    a.insert("b_block_size".to_owned(), 1024_i64);
    Kernel { src: SRC_4X4, name: "q4km_matmul_rb_coopmat_f32acc_cached_4x4",
        assignments: a, tile_rows: 64, tile_cols: 64 }
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

#[allow(clippy::too_many_arguments)]
fn dispatch_one(
    at: &str,
    kernel: &Kernel,
    q_bytes: &[u8],
    x_bytes: &[u8],
    m: usize,
    n: usize,
    k: usize,
    n_bpr: usize,
    ctx: &VulkanContext,
) -> Option<Vec<f32>> {
    let (bytes, meta) = compile_source_with_assignments(kernel.src, &kernel.assignments)
        .unwrap_or_else(|e| panic!("{at}: {} must compile: {e:?}", kernel.name));
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let handle = match ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, meta.coopmat.as_ref(), kernel.name, meta.shared_memory_bytes,
    ) {
        Ok(h) => h,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("{at}: {} CoopMatUnsupported (typed-skip): {reason}", kernel.name);
            return None;
        }
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel: kn }) => {
            eprintln!("{at}: {} DeviceFeatureUnsupported {feature}/{kn} (typed-skip)", kernel.name);
            return None;
        }
        Err(e) => panic!("{at}: {} prepare_kernel_checked failed: {e:?}", kernel.name),
    };

    let pc = assemble_pc(&meta.binding_plan, m as u32, n as u32, k as u32, n_bpr as u32);
    let c_size: usize = m * n * 4;
    let workgroups = ((n / kernel.tile_cols) as u32, (m / kernel.tile_rows) as u32, 1u32);

    let outputs = match ctx.dispatch_handle(
        &handle, workgroups,
        &[q_bytes, x_bytes, &vec![0u8; c_size]],
        &[0, 0, c_size],
        &pc,
    ) {
        Ok(v) => v,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("{at}: {} CoopMatUnsupported at dispatch (typed-skip): {reason}", kernel.name);
            return None;
        }
        Err(e) => panic!("{at}: {} dispatch failed: {e:?}", kernel.name),
    };

    let c_bytes: &[u8] = &outputs[2];
    assert_eq!(c_bytes.len(), c_size, "{at}: {} C output size mismatch", kernel.name);
    let y: Vec<f32> = c_bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Some(y)
}

/// Dispatch the M3.6 2x2 leader AND a big-RB variant on the SAME fixture; assert bit-identity
/// (AT-2003 expected gate) AND report the combined-vs-oracle metric (NR-1 floor authority).
fn run_equiv(at: &str, variant: &Kernel, m: usize, n: usize, n_bpr: usize) {
    if !gpu_tests_enabled() {
        eprintln!("{at}: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }

    // NR-2 PER-VARIANT divisibility (tightened from the inherited %32): both grids must tile
    // cleanly, so M,N must satisfy BOTH the variant's predicate AND the 2x2's (%32 -> implied by %64).
    if !m.is_multiple_of(variant.tile_rows) || !n.is_multiple_of(variant.tile_cols)
        || !m.is_multiple_of(64) || !n.is_multiple_of(64) {
        eprintln!(
            "{at} [{}]: SKIP-WITH-LOG — M={m} N={n} must be multiples of 64 AND of the {}x{} \
             output tile; not dispatched (avoids OOB unconditional loads)",
            variant.name, variant.tile_rows, variant.tile_cols
        );
        return;
    }

    let ctx = VulkanContext::new().expect("VulkanContext must init");
    eprintln!("{at} [{}]: device={}", variant.name, ctx.physical_device_name());

    if !ctx.coopmat_support().feature_present {
        eprintln!("{at}: coopmat not supported; typed-skip");
        return;
    }
    if ctx.subgroup_size() != 32 {
        eprintln!("{at}: subgroup_size()={} != 32; typed-skip", ctx.subgroup_size());
        return;
    }

    let k: usize = n_bpr * 256;

    // SAME fixture (same seed scheme as the cached dispatch test) fed to BOTH kernels.
    let q_bytes = make_q4km_weights(m, n_bpr, 0xC0FFEE ^ (m as u64));
    let x_f16 = make_x_f16(k, n, 0xBADF00D ^ (n as u64));
    let x_bytes: Vec<u8> = x_f16.iter().flat_map(|&b| b.to_le_bytes()).collect();

    let cached = cached_2x2();
    let Some(y_cached) = dispatch_one(at, &cached, &q_bytes, &x_bytes, m, n, k, n_bpr, &ctx)
        else { return; };
    let Some(y_variant) = dispatch_one(at, variant, &q_bytes, &x_bytes, m, n, k, n_bpr, &ctx)
        else { return; };

    assert_eq!(
        y_cached.len(), y_variant.len(),
        "{at}: M3.6/variant output length mismatch ({} vs {})", y_cached.len(), y_variant.len()
    );

    // Bit-identity over the full (overlapping) output region.
    let mut n_diff = 0usize;
    let mut first_diff: Option<(usize, f32, f32)> = None;
    for (i, (&c, &vv)) in y_cached.iter().zip(y_variant.iter()).enumerate() {
        if c.to_bits() != vv.to_bits() {
            n_diff += 1;
            if first_diff.is_none() {
                first_diff = Some((i, c, vv));
            }
        }
    }

    // NR-1: report the variant's combined-vs-oracle metric so the orchestrator can apply the
    // fallback judgment (a byte-diff with combined <= 1e-3 is a driver-reassociation artifact).
    let y_ref = common_q4km_f32ref::q4km_dequant_matmul_f32accum_cpu(&q_bytes, &x_f16, m, n, n_bpr);
    let abs_scale = common_q4km_f32ref::q4km_abs_dot_scale(&q_bytes, &x_f16, m, n, n_bpr);
    let combined = common_q4km_f32ref::max_rel_diff_combined(&y_variant, &y_ref, &abs_scale);
    let combined_within = combined <= FROZEN_REL_TOL;

    eprintln!(
        "{at} [{}]: n_diff={n_diff}/{} | variant combined-vs-oracle={combined:.3e} \
         (frozen rtol={FROZEN_REL_TOL:.0e}, within={combined_within}) on {}",
        variant.name, y_cached.len(), ctx.physical_device_name()
    );

    assert!(
        n_diff == 0,
        "AT-2003 [{}]: variant output is NOT bit-identical to the M3.6 2x2 leader at M={m} N={n} \
         K={k}: {n_diff} of {} elements differ; first at idx {:?} (M3.6={:?}, variant={:?}). \
         The per-16x16-tile accumulation order is unchanged, so bit-identity is EXPECTED. A \
         non-zero diff signals a STRUCTURAL deviation (wrong cache-resize index, wrong A-row/B-col \
         coopmat offset/stride, or accumulation-order change). \
         NR-1 FALLBACK: the variant's combined-vs-oracle metric here is {combined:.3e} \
         (within frozen 1e-3 = {combined_within}). IF combined <= 1e-3 the kernel is numerically \
         CORRECT and the byte-diff is a driver-HMMA-reassociation artifact (document, do NOT \
         loosen 1e-3); the orchestrator applies that fallback. STOP and investigate.",
        variant.name, y_cached.len(),
        first_diff.map(|(i, _, _)| i),
        first_diff.map(|(_, c, _)| c),
        first_diff.map(|(_, _, v)| v),
    );

    eprintln!(
        "{at} [{}]: PASS — variant == M3.6 2x2 BIT-IDENTICAL at M={m} N={n} K={k} ({} elements)",
        variant.name, y_cached.len()
    );
}

// ── 4x2 vs M3.6 ──────────────────────────────────────────────────────────────────────────
#[test]
#[ignore]
fn at_2003_bigrb_4x2_equiv_k256() { run_equiv("at_2003_4x2_k256", &bigrb_4x2(), 128, 128, 1); }
#[test]
#[ignore]
fn at_2003_bigrb_4x2_equiv_k512() { run_equiv("at_2003_4x2_k512", &bigrb_4x2(), 128, 128, 2); }
#[test]
#[ignore]
fn at_2003_bigrb_4x2_equiv_k14336() { run_equiv("at_2003_4x2_k14336", &bigrb_4x2(), 128, 128, 56); }

// ── 4x4 vs M3.6 ──────────────────────────────────────────────────────────────────────────
#[test]
#[ignore]
fn at_2003_bigrb_4x4_equiv_k256() { run_equiv("at_2003_4x4_k256", &bigrb_4x4(), 128, 128, 1); }
#[test]
#[ignore]
fn at_2003_bigrb_4x4_equiv_k512() { run_equiv("at_2003_4x4_k512", &bigrb_4x4(), 128, 128, 2); }
#[test]
#[ignore]
fn at_2003_bigrb_4x4_equiv_k14336() { run_equiv("at_2003_4x4_k14336", &bigrb_4x4(), 128, 128, 56); }
