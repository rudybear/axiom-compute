//! M3.7 (OPTIONAL) — AT-1907: the PLAIN double-buffered register-blocked coopmat matmul
//! (examples/matmul_rb_coopmat_db.axc) is BIT-EXACT to the M3.3c plain RB coopmat matmul
//! (examples/matmul_rb_coopmat.axc) on the AT-1731 integer-exact fixture (max_diff == 0).
//!
//! This isolates the PURE matmul-core latency-overlap effect from the Q4_K_M dequant path:
//! the _db variant is a pure SCHEDULING rewrite of M3.3c (same coopmat tile order, same
//! single-level OpPhi accumulation order), so its output MUST match M3.3c bit-for-bit. The
//! integer-valued fixture (A∈{1..4}, B∈{1..3}, per-element max 32*4*3=384 <= 2048) is
//! f16-integer-exact, so the comparison is exact with NO tolerance.
//!
//! BIT-EXACT vs M3.3c is the load-bearing race/parity detector for the plain double-buffer
//! pipeline (a missing barrier in the ping-pong reorder produces a non-zero diff that spirv-val
//! cannot catch).
//!
//! GPU-gated (#[ignore] + AXC_ENABLE_GPU_TESTS=1), typed-skip on Lavapipe / subgroup != 32.

use std::collections::BTreeMap;
use axc_driver::compile_source_with_assignments;
use axc_hir::ParamBindingPlan;
use axc_runtime::{VulkanContext, DispatchError};

const M33C_SRC: &str = include_str!("../../../examples/matmul_rb_coopmat.axc");
const DB_SRC: &str = include_str!("../../../examples/matmul_rb_coopmat_db.axc");

fn gpu_tests_enabled() -> bool {
    std::env::var("AXC_ENABLE_GPU_TESTS").as_deref() == Ok("1")
}

/// M3.3c plain RB strategy (single-buffer a_block_size=512).
fn m33c_assignments() -> BTreeMap<String, i64> {
    let mut m = BTreeMap::new();
    m.insert("rb_m".to_owned(), 2_i64);
    m.insert("rb_n".to_owned(), 2_i64);
    m.insert("tile_k".to_owned(), 16_i64);
    m.insert("a_block_size".to_owned(), 512_i64);
    m.insert("b_block_size".to_owned(), 512_i64);
    m
}

/// M3.7 plain double-buffered strategy (a_block_size_db/b_block_size_db PINNED at 1024).
fn db_assignments() -> BTreeMap<String, i64> {
    let mut m = BTreeMap::new();
    m.insert("rb_m".to_owned(), 2_i64);
    m.insert("rb_n".to_owned(), 2_i64);
    m.insert("tile_k".to_owned(), 16_i64);
    m.insert("a_block_size_db".to_owned(), 1024_i64);
    m.insert("b_block_size_db".to_owned(), 1024_i64);
    m
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

/// AT-1907: plain _db variant is BIT-EXACT to M3.3c AND to the CPU reference at M=N=64, K=32.
#[test]
#[ignore]
fn at_1907_matmul_rb_db_bit_exact_to_m33c() {
    if !gpu_tests_enabled() {
        eprintln!("at_1907: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    let ctx = VulkanContext::new().expect("VulkanContext must init");
    eprintln!("at_1907: device={}", ctx.physical_device_name());
    if !ctx.coopmat_support().feature_present {
        eprintln!("at_1907: coopmat not supported; typed-skip");
        return;
    }
    if ctx.subgroup_size() != 32 {
        eprintln!("at_1907: subgroup_size()={} != 32; typed-skip", ctx.subgroup_size());
        return;
    }

    const M: usize = 64;
    const N: usize = 64;
    const K: usize = 32; // 2 K-blocks -> 1 prefetch + 1 final-iter compute.

    // AT-1731 integer-exact fixture: A∈{1..4}, B∈{1..3}, per-element max 32*4*3=384 <= 2048.
    let a_f32: Vec<f32> = (0..M * K).map(|idx| ((idx % 4) + 1) as f32).collect();
    let b_f32: Vec<f32> = (0..K * N).map(|idx| ((idx % 3) + 1) as f32).collect();
    let a_bytes = f32_slice_to_f16_le_bytes(&a_f32);
    let b_bytes = f32_slice_to_f16_le_bytes(&b_f32);

    let Some(y_m33c) = dispatch_one(
        "at_1907", M33C_SRC, "matmul_rb_coopmat", &m33c_assignments(),
        &a_bytes, &b_bytes, M, N, K, &ctx,
    ) else { return; };
    let Some(y_db) = dispatch_one(
        "at_1907", DB_SRC, "matmul_rb_coopmat_db", &db_assignments(),
        &a_bytes, &b_bytes, M, N, K, &ctx,
    ) else { return; };

    // Bit-exact vs the CPU integer-exact reference.
    let y_cpu = cpu_f32_matmul(&a_f32, &b_f32, M, N, K);
    let max_diff_cpu = y_db.iter().zip(y_cpu.iter())
        .map(|(&g, &c)| (g - c).abs()).fold(0.0_f32, f32::max);
    assert!(
        max_diff_cpu == 0.0,
        "AT-1907: plain _db variant NOT bit-exact to the CPU reference — max_diff={max_diff_cpu} \
         != 0. The integer fixture is f16-exact; a non-zero diff is a double-buffer pipeline bug \
         (missing barrier / parity slip / accumulator reset). First4 GPU={:?} CPU={:?}",
        &y_db[..4.min(y_db.len())], &y_cpu[..4.min(y_cpu.len())]
    );

    // Bit-IDENTICAL vs M3.3c (the direct race/parity detector — pure scheduling).
    assert_eq!(y_m33c.len(), y_db.len(), "at_1907: output length mismatch");
    let n_diff = y_m33c.iter().zip(y_db.iter())
        .filter(|(&a, &b)| a.to_bits() != b.to_bits()).count();
    assert!(
        n_diff == 0,
        "AT-1907: plain _db variant NOT bit-identical to M3.3c matmul_rb_coopmat \
         ({n_diff} of {} elements differ). Double-buffering is pure scheduling; a non-zero diff \
         is a missing-barrier race or a parity/offset slip in the ping-pong reorder.",
        y_m33c.len()
    );

    eprintln!(
        "at_1907: PASS — plain _db variant bit-exact to CPU AND bit-identical to M3.3c at \
         M={M} N={N} K={K} on {}",
        ctx.physical_device_name()
    );
}
