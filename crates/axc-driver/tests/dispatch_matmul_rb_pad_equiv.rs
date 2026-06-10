//! M3.10a (plain-f32) — AT-2400/2401: the BANK-PADDED register-blocked coopmat matmul
//! (examples/matmul_rb_coopmat_pad.axc) is BIT-IDENTICAL to the M3.3c plain RB leader
//! (examples/matmul_rb_coopmat.axc).
//!
//! AT-2400 (the load-bearing mis-stride/mis-offset detector): cross-kernel BIT-IDENTITY on the
//! ASYMMETRIC make_transpose_fixture_a/b TILED across 768x768 (a multi-workgroup shape). Padding
//! is a PURE shared-address-layout change (same coopmat tile order, same accumulation order), so
//! the output MUST be BIT-IDENTICAL (max|y_pad - y_base|.to_bits() == 0). A mis-strided shared
//! read or a wrongly-bumped coopmat_load offset (e.g. the B-tile GATE-NOTE #1 trap) reads garbage
//! and CANNOT pass by symmetry — and spirv-val cannot catch a mis-strided shared read, so this
//! real-GPU gate is mandatory (anti-pattern #9).
//!
//! AT-2401: the padded kernel is bit-exact to the CPU integer-exact f16 matmul oracle at the
//! small fixture (M3.3c AT-1731 precedent, max_diff == 0).
//!
//! GPU-gated (#[ignore] + AXC_ENABLE_GPU_TESTS=1), typed-skip on Lavapipe / subgroup != 32.

#[path = "common_matmul.rs"]
mod common_matmul;

use std::collections::BTreeMap;
use axc_driver::compile_source_with_assignments;
use axc_hir::ParamBindingPlan;
use axc_runtime::{VulkanContext, DispatchError};

const BASE_SRC: &str = include_str!("../../../examples/matmul_rb_coopmat.axc");
const PAD_SRC: &str = include_str!("../../../examples/matmul_rb_coopmat_pad.axc");

fn gpu_tests_enabled() -> bool {
    std::env::var("AXC_ENABLE_GPU_TESTS").as_deref() == Ok("1")
}

/// M3.3c plain RB strategy (unpadded base).
fn rb2x2_assignments() -> BTreeMap<String, i64> {
    let mut m = BTreeMap::new();
    m.insert("rb_m".to_owned(), 2_i64);
    m.insert("rb_n".to_owned(), 2_i64);
    m.insert("tile_k".to_owned(), 16_i64);
    m.insert("a_block_size".to_owned(), 512_i64);
    m.insert("b_block_size".to_owned(), 512_i64);
    m
}

/// M3.10a padded RB strategy (PAD_A=PAD_B=8 canonical ship tuple). Adds the derived pad holes
/// on top of the base RB strategy.
fn rb2x2_pad_assignments() -> BTreeMap<String, i64> {
    let mut m = rb2x2_assignments();
    m.insert("a_pad_stride".to_owned(), 24_i64);   // 16 + PAD_A
    m.insert("a_pad_size".to_owned(), 768_i64);    // 32 * 24
    m.insert("a_pad_mat1off".to_owned(), 384_i64); // 16 * 24 (SCALES with pad)
    m.insert("b_pad_stride".to_owned(), 40_i64);   // 32 + PAD_B
    m.insert("b_pad_size".to_owned(), 640_i64);    // 16 * 40
    m
}

fn f32_slice_to_f16_le_bytes(vals: &[f32]) -> Vec<u8> {
    use half::f16;
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        out.extend_from_slice(&f16::from_f32(v).to_le_bytes());
    }
    out
}

fn f16_le_bytes_to_f32_slice(bytes: &[u8]) -> Vec<f32> {
    use half::f16;
    bytes.chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

/// Assemble M/N/K push constants by scalar name (robust to layout).
fn assemble_pc(plan: &ParamBindingPlan, m: u32, n: u32, k: u32) -> Vec<u8> {
    let mut pc: Vec<u8> = vec![0u8; plan.push_constant_total_bytes as usize];
    for s in &plan.scalars {
        let val: u32 = match s.name.as_str() {
            "M" => m,
            "N" => n,
            "K" => k,
            other => panic!("unexpected scalar param {other}"),
        };
        let start = s.offset as usize;
        pc[start..start + 4].copy_from_slice(&val.to_le_bytes());
    }
    pc
}

/// CPU f32 matmul reference (the AT-1731 integer-exact oracle).
fn cpu_f32_matmul(a_f32: &[f32], b_f32: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0_f32;
            for kk in 0..k {
                acc += a_f32[row * k + kk] * b_f32[kk * n + col];
            }
            c[row * n + col] = acc;
        }
    }
    c
}

/// Build an ASYMMETRIC MxK A fixture by tiling the 16x16 make_transpose_fixture_a primitive.
/// A swapped/transposed/mis-strided sub-tile cannot pass by symmetry.
fn tiled_fixture_a(m: usize, k: usize) -> Vec<f32> {
    let base = common_matmul::make_transpose_fixture_a();
    let mut out = vec![0.0_f32; m * k];
    for i in 0..m {
        for j in 0..k {
            out[i * k + j] = common_matmul::f16_bits_to_f32(base[(i % 16) * 16 + (j % 16)]);
        }
    }
    out
}

/// Build an ASYMMETRIC KxN B fixture by tiling make_transpose_fixture_b.
fn tiled_fixture_b(k: usize, n: usize) -> Vec<f32> {
    let base = common_matmul::make_transpose_fixture_b();
    let mut out = vec![0.0_f32; k * n];
    for i in 0..k {
        for j in 0..n {
            out[i * n + j] = common_matmul::f16_bits_to_f32(base[(i % 16) * 16 + (j % 16)]);
        }
    }
    out
}

/// Dispatch one kernel; return f32 output. None on typed-skip.
#[allow(clippy::too_many_arguments)]
fn dispatch_one(
    at: &str,
    src: &str,
    kernel_name: &str,
    assignments: &BTreeMap<String, i64>,
    a_bytes: &[u8],
    b_bytes: &[u8],
    m: usize,
    n: usize,
    k: usize,
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
        Err(DispatchError::DeviceFeatureUnsupported { feature, .. }) => {
            eprintln!("{at}: {kernel_name} DeviceFeatureUnsupported({feature}) — typed-skip");
            return None;
        }
        Err(e) => panic!("{at}: {kernel_name} prepare_kernel_checked failed: {e:?}"),
    };

    let pc = assemble_pc(&meta.binding_plan, m as u32, n as u32, k as u32);
    let c_size = m * n * 2; // f16 output
    let wg = ((n / 32) as u32, (m / 32) as u32, 1u32);

    let outputs = match ctx.dispatch_handle(
        &handle, wg, &[a_bytes, b_bytes, &vec![0u8; c_size]], &[0, 0, c_size], &pc,
    ) {
        Ok(v) => v,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("{at}: {kernel_name} CoopMatUnsupported at dispatch (typed-skip): {reason}");
            return None;
        }
        Err(e) => panic!("{at}: {kernel_name} dispatch failed: {e:?}"),
    };
    Some(f16_le_bytes_to_f32_slice(&outputs[2]))
}

/// AT-2400: padded RB BIT-IDENTICAL to the unpadded RB on the ASYMMETRIC fixture at 768^3.
#[test]
#[ignore]
fn at_2400_bit_identical_asymmetric() {
    let at = "AT-2400";
    if !gpu_tests_enabled() {
        eprintln!("{at}: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    const M: usize = 768;
    const N: usize = 768;
    const K: usize = 768;

    let ctx = VulkanContext::new().unwrap_or_else(|e| panic!("{at}: VulkanContext init: {e}"));
    eprintln!("{at}: device={}", ctx.physical_device_name());
    if !ctx.coopmat_support().feature_present {
        eprintln!("{at}: coopmat not supported; typed-skip");
        return;
    }
    if ctx.subgroup_size() != 32 {
        eprintln!("{at}: subgroup_size()={} != 32; typed-skip", ctx.subgroup_size());
        return;
    }

    // ASYMMETRIC per-row/col-varying fixture (a mis-stride cannot pass by symmetry).
    let a_f32 = tiled_fixture_a(M, K);
    let b_f32 = tiled_fixture_b(K, N);
    let a_bytes = f32_slice_to_f16_le_bytes(&a_f32);
    let b_bytes = f32_slice_to_f16_le_bytes(&b_f32);

    let Some(y_base) = dispatch_one(
        at, BASE_SRC, "matmul_rb_coopmat", &rb2x2_assignments(),
        &a_bytes, &b_bytes, M, N, K, &ctx,
    ) else { return; };
    let Some(y_pad) = dispatch_one(
        at, PAD_SRC, "matmul_rb_coopmat_pad", &rb2x2_pad_assignments(),
        &a_bytes, &b_bytes, M, N, K, &ctx,
    ) else { return; };

    assert_eq!(y_base.len(), y_pad.len(), "{at}: output length mismatch");
    let mut n_diff = 0usize;
    let mut first_diff: Option<(usize, f32, f32)> = None;
    for (idx, (&b, &p)) in y_base.iter().zip(y_pad.iter()).enumerate() {
        if b.to_bits() != p.to_bits() {
            n_diff += 1;
            if first_diff.is_none() { first_diff = Some((idx, b, p)); }
        }
    }
    eprintln!(
        "{at}: 768^3 asymmetric fixture, padded vs base, n_diff={n_diff}/{} first_diff={:?}",
        y_base.len(), first_diff
    );
    assert_eq!(
        n_diff, 0,
        "{at}: padded RB must be BIT-IDENTICAL to the unpadded RB (padding is pure shared-address \
         layout); n_diff={n_diff}. A non-zero diff is a mis-strided shared read or a wrongly-bumped \
         coopmat_load offset (e.g. the B-tile GATE-NOTE #1 offset trap). first_diff={first_diff:?}. \
         HONESTY GATE: do NOT relax — debug the padded write index + the coopmat_load strides/offsets."
    );
    eprintln!(
        "{at} PASS: padded RB BIT-IDENTICAL to unpadded RB (n_diff=0) on {} — 768^3 asymmetric fixture",
        ctx.physical_device_name()
    );
}

/// AT-2401: padded RB bit-exact to the CPU integer-exact oracle at M=N=64, K=32.
#[test]
#[ignore]
fn at_2401_bit_exact_cpu() {
    let at = "AT-2401";
    if !gpu_tests_enabled() {
        eprintln!("{at}: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    const M: usize = 64;
    const N: usize = 64;
    const K: usize = 32;

    let ctx = VulkanContext::new().unwrap_or_else(|e| panic!("{at}: VulkanContext init: {e}"));
    eprintln!("{at}: device={}", ctx.physical_device_name());
    if !ctx.coopmat_support().feature_present {
        eprintln!("{at}: coopmat not supported; typed-skip");
        return;
    }
    if ctx.subgroup_size() != 32 {
        eprintln!("{at}: subgroup_size()={} != 32; typed-skip", ctx.subgroup_size());
        return;
    }

    // AT-1731 integer-exact fixture: A in {1..4}, B in {1..3}; per-element max 32*4*3=384 <= 2048.
    let a_f32: Vec<f32> = (0..M * K).map(|idx| ((idx % 4) + 1) as f32).collect();
    let b_f32: Vec<f32> = (0..K * N).map(|idx| ((idx % 3) + 1) as f32).collect();
    let a_bytes = f32_slice_to_f16_le_bytes(&a_f32);
    let b_bytes = f32_slice_to_f16_le_bytes(&b_f32);

    let Some(y_pad) = dispatch_one(
        at, PAD_SRC, "matmul_rb_coopmat_pad", &rb2x2_pad_assignments(),
        &a_bytes, &b_bytes, M, N, K, &ctx,
    ) else { return; };

    let y_cpu = cpu_f32_matmul(&a_f32, &b_f32, M, N, K);
    let max_diff = y_pad.iter().zip(y_cpu.iter())
        .map(|(&g, &c)| (g - c).abs()).fold(0.0_f32, f32::max);
    assert!(
        max_diff == 0.0,
        "{at}: padded RB NOT bit-exact to the CPU reference — max_diff={max_diff} != 0. The integer \
         fixture is f16-exact; a non-zero diff is a padded-index addressing bug. First4 GPU={:?} CPU={:?}",
        &y_pad[..4.min(y_pad.len())], &y_cpu[..4.min(y_cpu.len())]
    );
    eprintln!(
        "{at} PASS: padded RB bit-exact to CPU (max_diff=0.0) on {} — M={M} N={N} K={K}",
        ctx.physical_device_name()
    );
}
