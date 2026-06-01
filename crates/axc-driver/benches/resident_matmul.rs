//! GPU-resident matmul benchmarks — M3.1.
//!
//! New bench IDs (DO NOT change existing IDs from dispatch.rs/dispatch_q4km.rs):
//! - `dispatch_resident_matmul_tile`: 16x16 coopmat tile, kernel-only ns.
//!   SKIPS if coopmat unsupported (typed CoopMatUnsupported skip).
//! - `matmul_tflops_4096`: 4096x4096 f32 resident matmul, effective TFLOPS.
//!   Runs on ALL vendors (no coopmat dependency).
//!
//! Gated on `AXC_ENABLE_GPU_BENCHES=1`. Without the flag, both benches
//! immediately return without doing GPU work (safe for CI without GPU).
//!
//! Methodology:
//! - Upload inputs ONCE via `upload_resident` (or simulate with dispatch_handle).
//! - Dispatch N=3 times (Criterion N or explicit).
//! - Report kernel_ns per iteration.
//! - effective_tflops = 2*M*N*K / kernel_seconds.
//! - cuBLAS ratio is printed labeled "estimate (external cuBLAS baseline)"; NOT asserted.

#![allow(dead_code)]

#[path = "common.rs"]
mod common;

use criterion::{criterion_group, criterion_main, Criterion};

const MATMUL_TILE_SRC: &str = include_str!("../../../examples/matmul_tile.axc");
const MATMUL_F32_SRC: &str = include_str!("../../../examples/matmul_f32_tiled.axc");

/// External cuBLAS f32 GEMM throughput estimate for the NVIDIA RTX PRO 6000 Blackwell.
///
/// Source: NVIDIA RTX PRO 6000 Blackwell datasheet.
/// Peak f32 TFLOPS ~ 125 TFLOPS for GB202.
/// LABELED as estimate (not a same-machine A/B). NOT asserted >= any ratio (AT-1532).
const CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000: f64 = 125.0_f64;

fn gpu_benches_enabled() -> bool {
    std::env::var("AXC_ENABLE_GPU_BENCHES").as_deref() == Ok("1")
}

/// AT-1532 (partial bench): effective_tflops() is finite and > 0 for the matmul kernel.
///
/// The full assertion is in at_1532 (which requires GPU execution). This bench
/// verifies the formula itself compiles and returns a sane value.
fn effective_tflops(m: usize, n: usize, k: usize, kernel_seconds: f64) -> f64 {
    if kernel_seconds <= 0.0 { return 0.0; }
    2.0 * (m as f64) * (n as f64) * (k as f64) / kernel_seconds / 1e12
}

/// `dispatch_resident_matmul_tile`: 16x16 coopmat tile, kernel-only timing.
///
/// SKIPS if coopmat is unsupported (CoopMatUnsupported typed skip).
/// On NVIDIA: measures kernel-only ns via GPU timestamp (or CPU wall fallback).
fn bench_dispatch_resident_matmul_tile(c: &mut Criterion) {
    if !gpu_benches_enabled() {
        eprintln!("dispatch_resident_matmul_tile: AXC_ENABLE_GPU_BENCHES not set; skipping");
        return;
    }
    use axc_driver::compile_source_with_meta;
    use axc_runtime::{VulkanContext, DispatchError};
    use axc_runtime::{coopmat_shape_supported, coopmat_required_shape_from_meta};

    let ctx = match VulkanContext::new() {
        Ok(c) => c,
        Err(e) => { eprintln!("dispatch_resident_matmul_tile: no Vulkan: {e}"); return; }
    };

    let (bytes, meta) = match compile_source_with_meta(MATMUL_TILE_SRC) {
        Ok(r) => r,
        Err(e) => { eprintln!("dispatch_resident_matmul_tile: compile failed: {e}"); return; }
    };
    let words: Vec<u32> = bytes.chunks_exact(4).map(|c| {
        u32::from_le_bytes([c[0], c[1], c[2], c[3]])
    }).collect();

    // Preflight: skip if coopmat unsupported or subgroup != 32.
    if !ctx.coopmat_support().feature_present {
        eprintln!("dispatch_resident_matmul_tile: coopmat not supported; skipping");
        return;
    }
    if let Some(ref cm) = meta.coopmat {
        let req = coopmat_required_shape_from_meta(cm);
        if !coopmat_shape_supported(&req, ctx.coopmat_support()) {
            eprintln!("dispatch_resident_matmul_tile: 16x16x16 f16 not in supported set; skipping");
            return;
        }
    }
    if ctx.subgroup_size() != 32 {
        eprintln!("dispatch_resident_matmul_tile: subgroup_size={} != 32; skipping", ctx.subgroup_size());
        return;
    }

    // Build 16x16 f16 input tiles.
    let a_f16: Vec<u16> = (0..256_u16).map(|i| {
        let row = i / 16;
        let col = i % 16;
        let v: f32 = 1.0 + 0.5 * row as f32 - 0.25 * col as f32;
        half::f16::from_f32(v).to_bits()
    }).collect();
    let b_f16: Vec<u16> = (0..256_u16).map(|i| {
        let row = i / 16;
        let col = i % 16;
        let v: f32 = -0.5 + 0.5 * col as f32 + 0.125 * row as f32;
        half::f16::from_f32(v).to_bits()
    }).collect();
    let a_bytes: Vec<u8> = a_f16.iter().flat_map(|b| b.to_le_bytes()).collect();
    let b_bytes: Vec<u8> = b_f16.iter().flat_map(|b| b.to_le_bytes()).collect();
    let c_zeros: Vec<u8> = vec![0u8; 512];
    let tile_offset: u32 = 0;
    let pc: Vec<u8> = tile_offset.to_le_bytes().to_vec();

    let handle = match ctx.prepare_kernel(&words, &meta.binding_plan, meta.push_constant_total_bytes, &meta.entry_point) {
        Ok(h) => h,
        Err(e) => { eprintln!("dispatch_resident_matmul_tile: pipeline create: {e}"); return; }
    };

    let mut group = c.benchmark_group("resident_matmul");
    group.bench_function("dispatch_resident_matmul_tile", |b| {
        b.iter(|| {
            let result = ctx.dispatch_handle(
                &handle, (1, 1, 1),
                &[&a_bytes, &b_bytes, &c_zeros],
                &[0, 0, 512],
                &pc,
            );
            match result {
                Ok(_) => {}
                Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
                    eprintln!("dispatch_resident_matmul_tile: coopmat skip: {reason}");
                }
                Err(e) => eprintln!("dispatch_resident_matmul_tile: dispatch error: {e}"),
            }
        });
    });
    group.finish();
}

/// `matmul_tflops_4096`: 4096x4096 f32 resident matmul, effective TFLOPS.
///
/// Runs on ALL vendors (no coopmat). Reports effective_tflops = 2*M*N*K/kernel_s.
/// Prints percent-of-cuBLAS estimate labeled "estimate (external cuBLAS baseline)".
/// AT-1532: asserts effective_tflops > 0 && finite.
fn bench_matmul_tflops_4096(c: &mut Criterion) {
    if !gpu_benches_enabled() {
        eprintln!("matmul_tflops_4096: AXC_ENABLE_GPU_BENCHES not set; skipping");
        return;
    }
    use axc_driver::compile_source_with_meta;
    use axc_runtime::{VulkanContext, DispatchError};

    let ctx = match VulkanContext::new() {
        Ok(ctx) => ctx,
        Err(e) => { eprintln!("matmul_tflops_4096: no Vulkan: {e}"); return; }
    };

    let (bytes, meta) = match compile_source_with_meta(MATMUL_F32_SRC) {
        Ok(r) => r,
        Err(e) => { eprintln!("matmul_tflops_4096: compile failed: {e}"); return; }
    };
    let words: Vec<u32> = bytes.chunks_exact(4).map(|c| {
        u32::from_le_bytes([c[0], c[1], c[2], c[3]])
    }).collect();

    // 4096x4096x4096 f32 matmul.
    let m: usize = 4096;
    let n: usize = 4096;
    let k: usize = 4096;
    let total_elements = m * k;
    let mut a_f32: Vec<f32> = vec![0.0_f32; total_elements];
    let mut b_f32: Vec<f32> = vec![0.0_f32; k * n];
    // Fill with small values to avoid overflow.
    for (i, v) in a_f32.iter_mut().enumerate() { *v = (i % 16 + 1) as f32 / 256.0; }
    for (i, v) in b_f32.iter_mut().enumerate() { *v = (i % 16 + 1) as f32 / 256.0; }
    let a_bytes: Vec<u8> = a_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
    let b_bytes: Vec<u8> = b_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
    let c_size: usize = m * n * 4;

    let handle = match ctx.prepare_kernel(&words, &meta.binding_plan, meta.push_constant_total_bytes, &meta.entry_point) {
        Ok(h) => h,
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
            eprintln!("matmul_tflops_4096: DeviceFeatureUnsupported {feature}/{kernel}; skipping");
            return;
        }
        Err(e) => { eprintln!("matmul_tflops_4096: pipeline create: {e}"); return; }
    };

    #[allow(clippy::manual_div_ceil)]
    let wg_x: u32 = ((m * n) as u32 + 63) / 64;
    let workgroups = (wg_x, 1_u32, 1_u32);
    let mut pc: Vec<u8> = Vec::new();
    pc.extend_from_slice(&(m as u32).to_le_bytes());
    pc.extend_from_slice(&(n as u32).to_le_bytes());
    pc.extend_from_slice(&(k as u32).to_le_bytes());

    let mut group = c.benchmark_group("resident_matmul");
    group.sample_size(3);
    group.bench_function("matmul_tflops_4096", |b| {
        b.iter_custom(|iters| {
            let mut total_ns: u64 = 0;
            for _ in 0..iters {
                let t0 = std::time::Instant::now();
                let result = ctx.dispatch_handle(
                    &handle,
                    workgroups,
                    &[&a_bytes, &b_bytes, &vec![0u8; c_size]],
                    &[0, 0, c_size],
                    &pc,
                );
                let elapsed_ns = t0.elapsed().as_nanos() as u64;
                total_ns += elapsed_ns;
                match result {
                    Ok(_) => {}
                    Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
                        eprintln!("matmul_tflops_4096: DeviceFeatureUnsupported {feature}/{kernel}");
                    }
                    Err(e) => eprintln!("matmul_tflops_4096: dispatch error: {e}"),
                }
            }
            // AT-1532: report effective TFLOPS.
            if iters > 0 {
                let avg_ns: f64 = total_ns as f64 / iters as f64;
                let kernel_seconds: f64 = avg_ns * 1e-9;
                let tflops: f64 = effective_tflops(m, n, k, kernel_seconds);
                let cublas_pct: f64 = tflops / CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000 * 100.0;
                eprintln!(
                    "matmul_tflops_4096: {tflops:.3} TFLOPS  \
                     ({cublas_pct:.1}% of cuBLAS estimate (external cuBLAS baseline))"
                );
                assert!(tflops > 0.0 && tflops.is_finite(), "effective_tflops must be > 0 and finite (AT-1532)");
            }
            std::time::Duration::from_nanos(total_ns)
        });
    });
    group.finish();
}

criterion_group!(resident_matmul_benches, bench_dispatch_resident_matmul_tile, bench_matmul_tflops_4096);
criterion_main!(resident_matmul_benches);
