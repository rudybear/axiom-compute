//! M3.3 competitive coopmat matmul resident TFLOPS bench (AT-1710).
//!
//! NEW bench file for the competitive coopmat matmul resident TFLOPS measurement.
//! Bench id: `dispatch_resident_matmul_shared_coopmat`.
//!
//! This bench is a SEPARATE file so it does NOT change existing bench IDs
//! (resident_timing_kernels, naive_gemm_harness_validation, dispatch_resident_matmul_tile)
//! in resident_matmul.rs.
//!
//! ## Honesty disclaimer (OQ-E, AT-1710)
//!
//! Reports the BARE measured effective_tflops + % of the labeled cuBLAS DATASHEET ESTIMATE.
//! NOT a same-machine A/B (cuBLAS is not run). The 125-TFLOPS figure is the NVIDIA RTX PRO
//! 6000 Blackwell f32 GEMM peak from the datasheet — a theoretical upper bound.
//!
//! A single-16x16-coopmat-tile-per-workgroup kernel (no register blocking, no multi-warp
//! tiles, no double buffering) is expected to land at a MODEST fraction of the datasheet
//! estimate. The bare measured number is reported as-is; no "competitive" framing unless
//! the measured % warrants it.
//!
//! ## 50 GB cap
//!
//! Fixture sizes are asserted < 50 GB total before allocation.

#![allow(dead_code)]

#[path = "common.rs"]
mod common;

use criterion::{criterion_group, criterion_main, Criterion};
use axc_driver::compile_source_with_assignments;
use axc_runtime::{
    VulkanContext, DispatchError,
    ResidentBenchConfig, ResidentTimingSource,
};
use std::collections::BTreeMap;

const MATMUL_SHARED_COOPMAT_SRC: &str =
    include_str!("../../../examples/matmul_shared_coopmat.axc");

/// cuBLAS f32 GEMM throughput DATASHEET ESTIMATE for NVIDIA RTX PRO 6000 Blackwell.
///
/// Source: NVIDIA RTX PRO 6000 Blackwell datasheet (GB202 peak f32 GEMM ~125 TFLOPS).
/// This is a LABELED estimate — NOT a same-machine A/B measurement. Do NOT assert >= ratio.
const CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000: f64 = 125.0_f64;

/// Number of warmup dispatches (MANDATORY — TIMING-1 / HANDOFF-12).
const N_WARMUP: usize = 2;
/// Number of measured dispatches for MIN-of-N (AT-1710).
const N_MEASURED: usize = 10;

fn gpu_benches_enabled() -> bool {
    std::env::var("AXC_ENABLE_GPU_BENCHES").as_deref() == Ok("1")
}

fn timing_source_label(ts: ResidentTimingSource) -> &'static str {
    match ts {
        ResidentTimingSource::GpuTimestamp => "GpuTimestamp",
        ResidentTimingSource::CpuFenceWall =>
            "CpuFenceWall — scheduling-inclusive, NOT a GPU kernel time",
    }
}

fn effective_tflops(m: usize, n: usize, k: usize, kernel_seconds: f64) -> f64 {
    if kernel_seconds <= 0.0 {
        return 0.0;
    }
    2.0 * (m as f64) * (n as f64) * (k as f64) / kernel_seconds / 1e12
}

fn f32_slice_to_f16_le_bytes(vals: &[f32]) -> Vec<u8> {
    use half::f16;
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        let h = f16::from_f32(v);
        out.extend_from_slice(&h.to_le_bytes());
    }
    out
}

fn tile_assignments(tile_m: i64, tile_n: i64, tile_k: i64) -> BTreeMap<String, i64> {
    let mut m = BTreeMap::new();
    m.insert("tile_m".to_owned(), tile_m);
    m.insert("tile_n".to_owned(), tile_n);
    m.insert("tile_k".to_owned(), tile_k);
    m.insert("tile_a_size".to_owned(), tile_m * tile_k);
    m.insert("tile_b_size".to_owned(), tile_k * tile_n);
    m
}

/// AT-1710: `dispatch_resident_matmul_shared_coopmat` — competitive coopmat TFLOPS bench.
///
/// Uploads once, dispatches N_WARMUP + N_MEASURED times, reports MIN-of-10 effective_tflops.
/// Typed-skip on CoopMatUnsupported (Lavapipe) or DeviceFeatureUnsupported.
///
/// Reports: bare X TFLOPS = Y% of cuBLAS f32 DATASHEET ESTIMATE (NOT a same-machine A/B).
/// Does NOT assert >= any TFLOPS ratio.
fn bench_dispatch_resident_matmul_shared_coopmat(c: &mut Criterion) {
    if !gpu_benches_enabled() {
        eprintln!("dispatch_resident_matmul_shared_coopmat: AXC_ENABLE_GPU_BENCHES not set; skipping");
        return;
    }

    // Fixture: multi-tile dispatch covering a >=256×256×256 problem so the K-loop
    // runs many blocks across many workgroups.
    // Each workgroup handles one 16×16 output tile. For a 256×256 output:
    //   workgroups = (256/16, 256/16, 1) = (16, 16, 1).
    // K = 256 with tile_k=16 → 16 K-blocks per workgroup (multi-tile accumulation).
    const M: usize = 256;
    const N: usize = 256;
    const K: usize = 256;
    const TILE_K: usize = 16;

    // 50 GB cap (anti-pattern #1 compliance).
    let total_bytes_f16 = (M * K + K * N + M * N) * 2; // f16 = 2 bytes
    assert!(
        total_bytes_f16 < 50 * 1024 * 1024 * 1024,
        "dispatch_resident_matmul_shared_coopmat: fixture exceeds 50 GB cap ({total_bytes_f16} bytes)"
    );

    let assignments = tile_assignments(16, 16, TILE_K as i64);
    let (bytes, meta) = match compile_source_with_assignments(MATMUL_SHARED_COOPMAT_SRC, &assignments) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("dispatch_resident_matmul_shared_coopmat: compile failed: {e}");
            return;
        }
    };
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let ctx = match VulkanContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("dispatch_resident_matmul_shared_coopmat: no Vulkan: {e}");
            return;
        }
    };

    let handle = match ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, None, "matmul_shared_coopmat",
        meta.shared_memory_bytes,
    ) {
        Ok(h) => h,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("dispatch_resident_matmul_shared_coopmat: CoopMatUnsupported ({reason}) — typed-skip (Lavapipe)");
            return;
        }
        Err(DispatchError::DeviceFeatureUnsupported { feature, .. }) => {
            eprintln!("dispatch_resident_matmul_shared_coopmat: DeviceFeatureUnsupported ({feature}) — typed-skip");
            return;
        }
        Err(e) => {
            eprintln!("dispatch_resident_matmul_shared_coopmat: prepare_kernel_checked failed: {e}");
            return;
        }
    };

    eprintln!("dispatch_resident_matmul_shared_coopmat: device={}", ctx.physical_device_name());
    eprintln!("  fixture: M={M} N={N} K={K} tile_k={TILE_K}");

    // Build input data (small values to keep f16 exact for correctness if measured).
    let a: Vec<f32> = (0..M*K).map(|i| 0.5_f32 + (i % 4) as f32 * 0.25_f32).collect();
    let b: Vec<f32> = (0..K*N).map(|i| 0.5_f32 + (i % 3) as f32 * 0.25_f32).collect();
    let a_bytes = f32_slice_to_f16_le_bytes(&a);
    let b_bytes = f32_slice_to_f16_le_bytes(&b);
    let c_size = M * N * 2; // f16 output

    // Push constants: M, N, K.
    let mut pc = Vec::new();
    pc.extend_from_slice(&(M as u32).to_le_bytes());
    pc.extend_from_slice(&(N as u32).to_le_bytes());
    pc.extend_from_slice(&(K as u32).to_le_bytes());

    // Dispatch: (N/16, M/16, 1) workgroups — one per 16×16 output tile.
    let wg_x = (N / 16) as u32;
    let wg_y = (M / 16) as u32;
    let workgroups = (wg_x, wg_y, 1_u32);
    let cfg = ResidentBenchConfig { workgroups, push_constants: pc };

    // Upload once.
    let resident = match ctx.upload_resident(
        &handle,
        &[&a_bytes, &b_bytes, &vec![0u8; c_size]],
        &[0, 0, c_size as u64],
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("dispatch_resident_matmul_shared_coopmat: upload_resident failed: {e}");
            return;
        }
    };

    let mut group = c.benchmark_group("resident_matmul_competitive");
    group.sample_size(10);

    group.bench_function("dispatch_resident_matmul_shared_coopmat", |b| {
        b.iter_custom(|iters| {
            let mut total_ns: u64 = 0;
            for _ in 0..iters {
                // Mandatory warmup.
                for _ in 0..N_WARMUP {
                    ctx.dispatch_resident(&handle, &resident, &cfg)
                        .expect("warmup dispatch must succeed");
                }
                // Measured MIN-of-N.
                let mut min_ns: u64 = u64::MAX;
                let mut last_source = ResidentTimingSource::CpuFenceWall;
                for _ in 0..N_MEASURED {
                    let t = ctx.dispatch_resident(&handle, &resident, &cfg)
                        .expect("measured dispatch must succeed");
                    if t.kernel_ns < min_ns {
                        min_ns = t.kernel_ns;
                    }
                    last_source = t.timing_source;
                }
                let kernel_seconds = min_ns as f64 / 1e9;
                let tflops = effective_tflops(M, N, K, kernel_seconds);
                let pct_of_datasheet = 100.0 * tflops / CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000;

                // HONEST reporting: bare TFLOPS + % of labeled datasheet estimate.
                // Do NOT assert >= any ratio. Do NOT use the word "competitive" here.
                eprintln!(
                    "dispatch_resident_matmul_shared_coopmat: {:.3} TFLOPS = {:.1}% of \
                     cuBLAS f32 GEMM DATASHEET ESTIMATE ({} TFLOPS, RTX PRO 6000 Blackwell) \
                     — NOT a same-machine A/B ({})",
                    tflops, pct_of_datasheet, CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000,
                    timing_source_label(last_source)
                );
                assert!(tflops > 0.0 && tflops.is_finite(),
                    "dispatch_resident_matmul_shared_coopmat: effective_tflops must be > 0 and finite; got {tflops}");

                total_ns += min_ns;
            }
            std::time::Duration::from_nanos(total_ns)
        });
    });

    group.finish();
}

criterion_group!(
    resident_matmul_competitive_benches,
    bench_dispatch_resident_matmul_shared_coopmat,
);
criterion_main!(resident_matmul_competitive_benches);
