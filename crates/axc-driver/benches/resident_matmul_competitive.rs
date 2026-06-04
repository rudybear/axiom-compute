//! AT-1710: Honest effective-TFLOPS bench for matmul_shared_coopmat.axc — M3.3b.
//!
//! Dispatches a LARGE multiple-of-16 matmul (default 256×256×256, full tile grid)
//! via the resident upload-once / dispatch-N path (N_WARMUP=2 discarded, MIN-of-10,
//! GpuTimestamp). Reports effective_tflops = 2·M·N·K / kernel_ns as the bare number
//! plus % of the 125-TFLOPS f32 datasheet ESTIMATE. NO ratio is asserted (only
//! tflops>0 && finite). Typed-skip on CoopMatUnsupported (Lavapipe).
//!
//! Label honesty rules (AT-1710):
//! - Default label: "matmul_shared_coopmat effective TFLOPS = {tflops:.3} ({pct:.1}% ...)"
//! - Word "competitive" is OMITTED unless pct >= COMPETITIVE_PCT_THRESHOLD (25.0).
//! - CpuFenceWall path: omits the % and appends scheduling-inclusive qualifier (AT-1582).
//!
//! 50 GB cap asserted in this bench before allocation (not in the library).
//! 2D grid pre-check: both max_compute_work_group_count()[0] AND [1] are checked
//! (bench dispatches (N/16, M/16, 1), NOT (wg_x, 1, 1)).

#![allow(dead_code)]

use criterion::{criterion_group, criterion_main, Criterion};
use axc_driver::compile_source_with_assignments;
use axc_runtime::{
    VulkanContext, DispatchError,
    ResidentBenchConfig, ResidentTimingSource,
};
use std::collections::BTreeMap;

const MATMUL_SHARED_COOPMAT_SRC: &str = include_str!("../../../examples/matmul_shared_coopmat.axc");

/// External cuBLAS f32 GEMM throughput estimate for the NVIDIA RTX PRO 6000 Blackwell.
/// Source: NVIDIA RTX PRO 6000 Blackwell datasheet. Peak f32 TFLOPS ~ 125 TFLOPS for GB202.
/// LABELED as estimate (not a same-machine A/B). NOT asserted >= any ratio (AT-1532/AT-1710).
const CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000: f64 = 125.0_f64;

/// Threshold for printing the word "competitive" in the bench label.
/// At a single 32-lane subgroup per 16x16 output tile, throughput is expected to be MODEST.
/// "competitive" is omitted unless the measured % warrants it.
const COMPETITIVE_PCT_THRESHOLD: f64 = 25.0_f64;

/// Number of warmup dispatches (MANDATORY — TIMING-1 / HANDOFF-12).
const N_WARMUP: usize = 2;
/// Number of measured dispatches for MIN-of-N.
const N_MEASURED: usize = 10;

fn gpu_benches_enabled() -> bool {
    std::env::var("AXC_ENABLE_GPU_BENCHES").as_deref() == Ok("1")
}

/// Format timing_source for printing (AT-1582: CpuFenceWall must carry qualifier).
fn timing_source_label(ts: ResidentTimingSource) -> &'static str {
    match ts {
        ResidentTimingSource::GpuTimestamp => "GpuTimestamp",
        ResidentTimingSource::CpuFenceWall =>
            "CpuFenceWall — scheduling-inclusive, NOT a GPU kernel time",
    }
}

/// Compute effective TFLOPS from matrix dimensions and kernel nanoseconds.
fn effective_tflops(m: usize, n: usize, k: usize, kernel_ns: u64) -> f64 {
    if kernel_ns == 0 {
        return 0.0;
    }
    let kernel_seconds: f64 = kernel_ns as f64 * 1e-9_f64;
    2.0_f64 * (m as f64) * (n as f64) * (k as f64) / kernel_seconds / 1e12_f64
}

/// Build tile strategy assignments for the coopmat matmul (tile_m=16, tile_n=16, tile_k=16).
fn tile_assignments_16() -> BTreeMap<String, i64> {
    let mut m = BTreeMap::new();
    m.insert("tile_m".to_owned(), 16_i64);
    m.insert("tile_n".to_owned(), 16_i64);
    m.insert("tile_k".to_owned(), 16_i64);
    m.insert("tile_a_size".to_owned(), 256_i64); // 16 * 16
    m.insert("tile_b_size".to_owned(), 256_i64); // 16 * 16
    m
}

/// Run resident_min_of_n: upload-once / N_WARMUP+N_MEASURED dispatches / MIN-of-N.
///
/// Returns (min_kernel_ns, timing_source). Panics on any GPU error (bench context).
fn resident_min_of_n(
    ctx: &VulkanContext,
    handle: &axc_runtime::KernelHandle,
    inputs: &[&[u8]],
    output_sizes: &[u64],
    workgroups: (u32, u32, u32),
    push_constants: Vec<u8>,
) -> (u64, ResidentTimingSource) {
    let resident = ctx.upload_resident(handle, inputs, output_sizes)
        .expect("upload_resident must succeed");

    let cfg = ResidentBenchConfig { workgroups, push_constants };

    // MANDATORY warmup: discard first N_WARMUP dispatches (TIMING-1 / HANDOFF-12).
    for _ in 0..N_WARMUP {
        ctx.dispatch_resident(handle, &resident, &cfg)
            .expect("warmup dispatch must succeed");
    }

    // Measured loop: collect N_MEASURED samples, report MIN.
    let mut min_ns: u64 = u64::MAX;
    let mut last_source = ResidentTimingSource::CpuFenceWall;
    for _ in 0..N_MEASURED {
        let t = ctx.dispatch_resident(handle, &resident, &cfg)
            .expect("measured dispatch must succeed");
        if t.kernel_ns < min_ns {
            min_ns = t.kernel_ns;
        }
        last_source = t.timing_source;
    }

    (min_ns, last_source)
}

/// AT-1710: large coopmat matmul effective-TFLOPS bench (honest bare number + %).
///
/// `dispatch_resident_matmul_shared_coopmat` — bench ID must not change (AT-1711).
///
/// Default M=N=K=256 (16x16 tile grid = 256 workgroups; 3 f16 matrices = 393KB, well under 50GB).
/// Shrinks by /2 if N/16 > max_compute_work_group_count()[0] OR M/16 > [1] or 50GB cap exceeded.
/// Typed-skip on CoopMatUnsupported (Lavapipe).
fn bench_resident_matmul_competitive(c: &mut Criterion) {
    if !gpu_benches_enabled() {
        eprintln!("resident_matmul_competitive: AXC_ENABLE_GPU_BENCHES not set; skipping");
        return;
    }

    let ctx = match VulkanContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("resident_matmul_competitive: no Vulkan: {e}");
            return;
        }
    };

    // Compile with tile_assignments(16,16,16).
    let assignments = tile_assignments_16();
    let (bytes, meta) = match compile_source_with_assignments(MATMUL_SHARED_COOPMAT_SRC, &assignments) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("resident_matmul_competitive: compile failed: {e:?}");
            return;
        }
    };
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    // Typed preflight via prepare_kernel_checked (coopmat).
    let handle = match ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, meta.coopmat.as_ref(), "matmul_shared_coopmat",
        meta.shared_memory_bytes,
    ) {
        Ok(h) => h,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("resident_matmul_competitive: CoopMatUnsupported (typed-skip): {reason}");
            return;
        }
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
            eprintln!("resident_matmul_competitive: DeviceFeatureUnsupported {feature}/{kernel} (typed-skip)");
            return;
        }
        Err(e) => {
            eprintln!("resident_matmul_competitive: prepare_kernel_checked failed: {e}");
            return;
        }
    };

    // 2D grid pre-check: check BOTH axis [0] (wg_x = N/16) AND axis [1] (wg_y = M/16).
    // Shrink M and N together by /2 until they fit, or until they become too small.
    let max_wg_count = ctx.max_compute_work_group_count();
    let max_wg_x: u64 = max_wg_count[0] as u64;
    let max_wg_y: u64 = max_wg_count[1] as u64;
    let mut m: usize = 256;
    let mut n: usize = 256;
    let k: usize = 256;

    loop {
        // Both M and N must be multiples of 16 (tile size) and produce grids within limits.
        let wg_x: u64 = (n / 16) as u64;
        let wg_y: u64 = (m / 16) as u64;
        // 50 GB cap: 3 f16 matrices = 3 * m * n * 2 bytes (a_buf is m*k, b_buf is k*n).
        // For square m=n=k we bound by 3*m*m*2 = 6*m*m bytes.
        let total_bytes: u64 = 2 * (m as u64) * (k as u64) + 2 * (k as u64) * (n as u64) + 2 * (m as u64) * (n as u64);
        const FIFTY_GB: u64 = 50 * 1024 * 1024 * 1024;
        if wg_x <= max_wg_x && wg_y <= max_wg_y && total_bytes < FIFTY_GB && m >= 16 && n >= 16 {
            break;
        }
        m /= 2;
        n /= 2;
        if m < 16 || n < 16 {
            eprintln!("resident_matmul_competitive: could not find valid matrix dims; skipping");
            return;
        }
    }

    eprintln!(
        "resident_matmul_competitive: ACTUAL dims M={m} N={n} K={k} \
         (wg_x={}, wg_y={})",
        n / 16, m / 16
    );

    // Build f16 inputs (timing bench; small integer values are fine).
    let a_f32: Vec<f32> = (0..m * k).map(|i| ((i % 4) + 1) as f32).collect();
    let b_f32: Vec<f32> = (0..k * n).map(|i| ((i % 3) + 1) as f32).collect();
    let a_bytes: Vec<u8> = {
        use half::f16;
        let mut out = Vec::with_capacity(a_f32.len() * 2);
        for &v in &a_f32 {
            let h = f16::from_f32(v);
            out.extend_from_slice(&h.to_le_bytes());
        }
        out
    };
    let b_bytes: Vec<u8> = {
        use half::f16;
        let mut out = Vec::with_capacity(b_f32.len() * 2);
        for &v in &b_f32 {
            let h = f16::from_f32(v);
            out.extend_from_slice(&h.to_le_bytes());
        }
        out
    };
    let c_size: usize = m * n * 2; // f16 output

    let mut pc = Vec::with_capacity(12);
    pc.extend_from_slice(&(m as u32).to_le_bytes());
    pc.extend_from_slice(&(n as u32).to_le_bytes());
    pc.extend_from_slice(&(k as u32).to_le_bytes());

    // Full tile grid dispatch: (N/16, M/16, 1).
    let wg_x: u32 = (n / 16) as u32;
    let wg_y: u32 = (m / 16) as u32;
    let workgroups = (wg_x, wg_y, 1);

    let mut group = c.benchmark_group("resident_matmul_competitive");
    group.sample_size(10);

    group.bench_function("dispatch_resident_matmul_shared_coopmat", |b| {
        b.iter_custom(|iters| {
            let mut total_ns: u64 = 0;
            for _ in 0..iters {
                let (min_ns, ts) = resident_min_of_n(
                    &ctx, &handle,
                    &[&a_bytes, &b_bytes, &vec![0u8; c_size]],
                    &[0, 0, c_size as u64],
                    workgroups,
                    pc.clone(),
                );

                let tflops: f64 = effective_tflops(m, n, k, min_ns);
                let pct: f64 = tflops / CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000 * 100.0_f64;

                let label = if ts == ResidentTimingSource::CpuFenceWall {
                    // CpuFenceWall: omit % (scheduling-inclusive, not a GPU kernel time — AT-1582).
                    format!(
                        "matmul_shared_coopmat effective TFLOPS (honest) = {tflops:.3} \
                         | {min_ns} ns ({}) | M={m} N={n} K={k} dispatch=({wg_x},{wg_y},1)",
                        timing_source_label(ts)
                    )
                } else if pct >= COMPETITIVE_PCT_THRESHOLD {
                    // GpuTimestamp, high %: include "competitive" label.
                    format!(
                        "matmul_shared_coopmat effective TFLOPS (competitive) = {tflops:.3} \
                         ({pct:.1}% of {CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000}-TFLOPS f32 datasheet \
                         ESTIMATE — single-subgroup-per-tile, NOT a same-machine A/B) \
                         | {min_ns} ns ({}) | M={m} N={n} K={k} dispatch=({wg_x},{wg_y},1)",
                        timing_source_label(ts)
                    )
                } else {
                    // GpuTimestamp, modest %: honest label without "competitive".
                    format!(
                        "matmul_shared_coopmat effective TFLOPS (honest) = {tflops:.3} \
                         ({pct:.1}% of {CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000}-TFLOPS f32 datasheet \
                         ESTIMATE — single-subgroup-per-tile, NOT a same-machine A/B) \
                         | {min_ns} ns ({}) | M={m} N={n} K={k} dispatch=({wg_x},{wg_y},1)",
                        timing_source_label(ts)
                    )
                };
                eprintln!("{label}");

                // Assert only tflops > 0 and finite (NO ratio assertion — AT-1710).
                assert!(
                    tflops > 0.0_f64 && tflops.is_finite(),
                    "effective_tflops must be > 0 and finite (AT-1710)"
                );

                total_ns += min_ns;
            }
            std::time::Duration::from_nanos(total_ns)
        });
    });

    group.finish();
}

criterion_group!(resident_matmul_competitive_benches, bench_resident_matmul_competitive);
criterion_main!(resident_matmul_competitive_benches);
