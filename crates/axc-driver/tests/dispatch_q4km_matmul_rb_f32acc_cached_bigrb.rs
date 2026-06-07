//! M3.8 — GPU bit-within-tol correctness tests for the LARGER-REGISTER-TILE (4x2 / 4x4)
//! variants of the M3.6 scale-cached f32-accumulator fused Q4_K_M coopmat matmul
//! (examples/q4km_matmul_rb_coopmat_f32acc_cached_4x2.axc / _4x4.axc).
//!
//! AT-2000: K=256  (1 superblock),   M=128 N=128. Combined condition-aware diff <= FROZEN 1e-3.
//! AT-2001: K=512  (2 superblocks),  M=128 N=128. Exercises cross-superblock cache WAR with the
//!          RESIZED 512-entry caches.
//! AT-2002: K=14336 (56 superblocks, inference K), M=128 N=128. The validity claim at inference K;
//!          catches an r1-style accumulator-reset silent-zeros (combined ~= 1 >> 1e-3).
//!
//! Larger register blocking is the SAME arithmetic (more output tiles per workgroup, same
//! per-16x16-tile accumulation order, same single-level OpPhi carry, same dequant expression with
//! dsc/dmm read from the RESIZED 512-entry caches), so the combined condition-aware diff vs the
//! f32-accumulator oracle (common_q4km_f32ref::q4km_dequant_matmul_f32accum_cpu) MUST stay within
//! the FROZEN 1e-3 (NEVER loosened). The cross-kernel BIT-IDENTITY anchor is AT-2003.
//!
//! NR-2 PER-VARIANT divisibility guard: 4x2 needs M%64==0 && N%32==0 (output-tile 64x32);
//! 4x4 needs M%64==0 && N%64==0 (output-tile 64x64). A non-conforming size is SKIPPED WITH LOG
//! (NOT dispatched -> the unconditional loads never OOB).
//!
//! NVIDIA #[ignore]-gated (AXC_ENABLE_GPU_TESTS=1). Typed-skip on Lavapipe / subgroup != 32.

#[path = "common_q4km_f32ref.rs"]
mod common_q4km_f32ref;

use std::collections::BTreeMap;
use axc_driver::compile_source_with_assignments;
use axc_hir::ParamBindingPlan;
use axc_runtime::{VulkanContext, DispatchError};

const SRC_4X2: &str =
    include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached_4x2.axc");
const SRC_4X4: &str =
    include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached_4x4.axc");

/// Frozen relative tolerance (NEVER loosened).
const FROZEN_REL_TOL: f64 = 1e-3;

fn gpu_tests_enabled() -> bool {
    std::env::var("AXC_ENABLE_GPU_TESTS").as_deref() == Ok("1")
}

/// One big-RB variant's compile-time + grid parameters.
struct Variant {
    label: &'static str,
    src: &'static str,
    kernel_name: &'static str,
    /// @strategy assignments (rb_n + b_block_size differ between 4x2 and 4x4).
    assignments: BTreeMap<String, i64>,
    /// Output-tile dims for the per-variant divisibility guard + grid divisor (rows, cols).
    tile_rows: usize,
    tile_cols: usize,
}

fn variant_4x2() -> Variant {
    let mut a = BTreeMap::new();
    a.insert("rb_m".to_owned(), 4_i64);
    a.insert("rb_n".to_owned(), 2_i64);
    a.insert("tile_k".to_owned(), 16_i64);
    a.insert("a_block_size".to_owned(), 1024_i64);
    a.insert("b_block_size".to_owned(), 512_i64);
    Variant {
        label: "4x2",
        src: SRC_4X2,
        kernel_name: "q4km_matmul_rb_coopmat_f32acc_cached_4x2",
        assignments: a,
        tile_rows: 64,
        tile_cols: 32,
    }
}

fn variant_4x4() -> Variant {
    let mut a = BTreeMap::new();
    a.insert("rb_m".to_owned(), 4_i64);
    a.insert("rb_n".to_owned(), 4_i64);
    a.insert("tile_k".to_owned(), 16_i64);
    a.insert("a_block_size".to_owned(), 1024_i64);
    a.insert("b_block_size".to_owned(), 1024_i64);
    Variant {
        label: "4x4",
        src: SRC_4X4,
        kernel_name: "q4km_matmul_rb_coopmat_f32acc_cached_4x4",
        assignments: a,
        tile_rows: 64,
        tile_cols: 64,
    }
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

/// SAME seed scheme as dispatch_q4km_matmul_rb_f32acc_cached.rs so AT-2003 can reuse it.
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

struct MeasuredErr {
    raw_rel: f64,
    combined: f64,
}

/// Dispatch one big-RB variant and compare vs the f32-acc oracle. None on a typed-skip or a
/// per-variant divisibility skip (NR-2: M,N not multiples of THIS variant's output tile).
fn run_bigrb(at: &str, v: &Variant, m: usize, n: usize, n_bpr: usize) -> Option<MeasuredErr> {
    if !gpu_tests_enabled() {
        eprintln!("{at} [{}]: AXC_ENABLE_GPU_TESTS not set; skipping", v.label);
        return None;
    }

    // NR-2 PER-VARIANT divisibility guard (NOT the inherited %32-only guard). Skip-with-log so the
    // unconditional loads can never OOB on a non-conforming size.
    if !m.is_multiple_of(v.tile_rows) || !n.is_multiple_of(v.tile_cols) {
        eprintln!(
            "{at} [{}]: SKIP-WITH-LOG — M={m} N={n} not a multiple of the {}x{} output tile \
             (need M%{}==0 && N%{}==0); not dispatched (avoids OOB unconditional loads)",
            v.label, v.tile_rows, v.tile_cols, v.tile_rows, v.tile_cols
        );
        return None;
    }

    let ctx = VulkanContext::new().expect("VulkanContext must init");
    eprintln!("{at} [{}]: device={}", v.label, ctx.physical_device_name());

    if !ctx.coopmat_support().feature_present {
        eprintln!("{at} [{}]: coopmat not supported; typed-skip", v.label);
        return None;
    }
    if ctx.subgroup_size() != 32 {
        eprintln!("{at} [{}]: subgroup_size()={} != 32; typed-skip", v.label, ctx.subgroup_size());
        return None;
    }

    let k: usize = n_bpr * 256;

    let q_bytes = make_q4km_weights(m, n_bpr, 0xC0FFEE ^ (m as u64));
    let x_f16 = make_x_f16(k, n, 0xBADF00D ^ (n as u64));
    let x_bytes: Vec<u8> = x_f16.iter().flat_map(|&b| b.to_le_bytes()).collect();

    let y_ref = common_q4km_f32ref::q4km_dequant_matmul_f32accum_cpu(&q_bytes, &x_f16, m, n, n_bpr);

    let (bytes, meta) = compile_source_with_assignments(v.src, &v.assignments)
        .unwrap_or_else(|e| panic!("{}: must compile: {e:?}", v.kernel_name));
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let handle = match ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, meta.coopmat.as_ref(), v.kernel_name, meta.shared_memory_bytes,
    ) {
        Ok(h) => h,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("{at} [{}]: CoopMatUnsupported (typed-skip): {reason}", v.label);
            return None;
        }
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
            eprintln!("{at} [{}]: DeviceFeatureUnsupported {feature}/{kernel} (typed-skip)", v.label);
            return None;
        }
        Err(e) => panic!("{at} [{}]: prepare_kernel_checked failed: {e:?}", v.label),
    };

    let pc = assemble_pc(&meta.binding_plan, m as u32, n as u32, k as u32, n_bpr as u32);
    let c_size: usize = m * n * 4;
    // Per-variant grid: (N / tile_cols, M / tile_rows, 1). block_col = gid(0)/32 (32 threads),
    // block_row = gid(1). With local_size.x=32, gid(0) ranges over N/tile_cols * 32.
    let workgroups = ((n / v.tile_cols) as u32, (m / v.tile_rows) as u32, 1u32);

    let outputs = match ctx.dispatch_handle(
        &handle, workgroups,
        &[&q_bytes, &x_bytes, &vec![0u8; c_size]],
        &[0, 0, c_size],
        &pc,
    ) {
        Ok(v) => v,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("{at} [{}]: CoopMatUnsupported at dispatch (typed-skip): {reason}", v.label);
            return None;
        }
        Err(e) => panic!("{at} [{}]: dispatch failed: {e:?}", v.label),
    };

    let c_bytes: &[u8] = &outputs[2];
    assert_eq!(c_bytes.len(), c_size, "{at} [{}]: C output size mismatch", v.label);
    let y_gpu: Vec<f32> = c_bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let raw_rel = common_q4km_f32ref::max_rel_diff(&y_gpu, &y_ref);
    let abs_scale = common_q4km_f32ref::q4km_abs_dot_scale(&q_bytes, &x_f16, m, n, n_bpr);
    let combined = common_q4km_f32ref::max_rel_diff_combined(&y_gpu, &y_ref, &abs_scale);
    eprintln!(
        "{at} [{}]: MEASURED raw max_rel_diff={raw_rel:.3e} | COMBINED (the GATE)={combined:.3e} \
         (M={m} N={n} K={k}, frozen rtol={FROZEN_REL_TOL:.0e}) on {}",
        v.label, ctx.physical_device_name()
    );
    Some(MeasuredErr { raw_rel, combined })
}

/// Assert one variant is within frozen 1e-3 (the combined condition-aware floor authority).
fn assert_variant(at: &str, v: &Variant, m: usize, n: usize, n_bpr: usize) {
    let Some(e) = run_bigrb(at, v, m, n, n_bpr) else { return; };
    assert!(
        e.combined <= FROZEN_REL_TOL,
        "{at} [{}]: combined (condition-aware) max_rel_diff={:.3e} exceeds frozen rtol \
         {FROZEN_REL_TOL:.0e}. Larger register tiles are the SAME arithmetic as M3.6 (more output \
         tiles, same per-tile accumulation order, dsc/dmm from the resized 512-entry cache); a \
         miss points to a wrong cache-resize index / A-row offset / B-col stride / accumulator \
         reset. Investigate that BEFORE touching the frozen 1e-3. (raw={:.3e})",
        v.label, e.combined, e.raw_rel
    );
    eprintln!("{at} [{}]: PASS — combined={:.3e} within frozen 1e-3 (raw={:.3e})",
        v.label, e.combined, e.raw_rel);
}

/// AT-2000: K=256 (1 superblock), M=128 N=128 — both variants.
#[test]
#[ignore]
fn at_2000_bigrb_k256() {
    assert_variant("at_2000", &variant_4x2(), 128, 128, 1);
    assert_variant("at_2000", &variant_4x4(), 128, 128, 1);
}

/// AT-2001: K=512 (2 superblocks), M=128 N=128 — cross-superblock cache WAR with 512-entry caches.
#[test]
#[ignore]
fn at_2001_bigrb_k512() {
    assert_variant("at_2001", &variant_4x2(), 128, 128, 2);
    assert_variant("at_2001", &variant_4x4(), 128, 128, 2);
}

/// AT-2002: K=14336 (56 superblocks, inference K), M=128 N=128 — the TRUE correctness gate.
#[test]
#[ignore]
fn at_2002_bigrb_k14336() {
    assert_variant("at_2002", &variant_4x2(), 128, 128, 56);
    assert_variant("at_2002", &variant_4x4(), 128, 128, 56);
}
