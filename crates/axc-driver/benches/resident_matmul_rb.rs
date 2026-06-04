//! AT-1730: Honest effective-TFLOPS bench for matmul_rb_coopmat.axc — M3.3c register blocking.
//!
//! Register-blocked (RB 2×2) coopmat matmul: ONE workgroup (32 threads) computes a 32×32
//! output block (4 output tiles of 16×16 each). Dispatch grid (N/32, M/32, 1) workgroups.
//! This bench measures the honest effective TFLOPS improvement over the M3.3b baseline.
//!
//! Methodology: resident upload-once / N_WARMUP=2 discarded / MIN-of-10 / GpuTimestamp.
//! Verbatim copy of the resident_matmul_competitive.rs methodology (AT-1710 template).
//!
//! OCCUPANCY NOTE (pessimistic reviewer warning):
//!   At M=N=K=256: grid = (8,8,1) = 64 workgroups < ~188 SMs on RTX PRO 6000.
//!   The smaller grid (vs M3.3b's 256 workgroups) may cause RB to be SLOWER at 256³
//!   despite higher arithmetic intensity per workgroup. The bench measures HONESTLY
//!   at 256³ AND at larger sizes (≥512³ and ≥768³ if under 50GB), reporting ALL results.
//!   If 256³ regresses vs M3.3b (5.04 TFLOPS) but a larger size improves, both are reported.
//!
//! Label honesty rules (AT-1730, mirrors AT-1710):
//!   - Reports bare TFLOPS + % of 125-TFLOPS datasheet ESTIMATE.
//!   - Word "competitive" ONLY if pct >= COMPETITIVE_PCT_THRESHOLD (25.0).
//!   - NO ratio asserted (only tflops>0 && finite).
//!   - QA records the ACTUAL M/N/K alongside each TFLOPS (gaming guard).
//!
//! 50 GB cap asserted before allocation. 2D grid pre-check on BOTH axes.
//! Typed-skip on CoopMatUnsupported (Lavapipe) / DeviceFeatureUnsupported.
//!
//! Bench ID: dispatch_resident_matmul_rb_coopmat (MUST NOT collide with AT-1711).

#![allow(dead_code)]

use criterion::{criterion_group, criterion_main, Criterion};
use axc_driver::compile_source_with_assignments;
use axc_runtime::{
    VulkanContext, DispatchError,
    ResidentBenchConfig, ResidentTimingSource,
};
use std::collections::BTreeMap;

const MATMUL_RB_COOPMAT_SRC: &str = include_str!("../../../examples/matmul_rb_coopmat.axc");

/// External cuBLAS f32 GEMM throughput estimate for the NVIDIA RTX PRO 6000 Blackwell.
/// Source: NVIDIA RTX PRO 6000 Blackwell datasheet. Peak f32 TFLOPS ~ 125 TFLOPS for GB202.
/// LABELED as estimate (not a same-machine A/B). NOT asserted >= any ratio (AT-1730).
const CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000: f64 = 125.0_f64;

/// Threshold for printing the word "competitive" in the bench label.
/// 25% of 125 TFLOPS = 31.25 TFLOPS — realistic ceiling for single-subgroup register blocking.
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

/// Build RB 2×2 strategy assignments (rb_m=2, rb_n=2, tile_k=16, a_block_size=512, b_block_size=512).
fn rb2x2_assignments() -> BTreeMap<String, i64> {
    let mut m = BTreeMap::new();
    m.insert("rb_m".to_owned(), 2_i64);
    m.insert("rb_n".to_owned(), 2_i64);
    m.insert("tile_k".to_owned(), 16_i64);
    m.insert("a_block_size".to_owned(), 512_i64);
    m.insert("b_block_size".to_owned(), 512_i64);
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

/// Measure effective TFLOPS at one (m, n, k) size, print honest label, assert tflops > 0 && finite.
///
/// Returns true if measurement was taken (false if typed-skip or grid check failed).
fn measure_rb_tflops(
    ctx: &VulkanContext,
    handle: &axc_runtime::KernelHandle,
    m: usize,
    n: usize,
    k: usize,
) -> bool {
    // 2D grid pre-check: grid = (N/32, M/32, 1) for RB 2×2.
    let wg_x: u64 = (n / 32) as u64;
    let wg_y: u64 = (m / 32) as u64;
    let max_wg_count = ctx.max_compute_work_group_count();
    let max_wg_x: u64 = max_wg_count[0] as u64;
    let max_wg_y: u64 = max_wg_count[1] as u64;
    if wg_x > max_wg_x || wg_y > max_wg_y {
        eprintln!(
            "resident_matmul_rb: grid ({wg_x},{wg_y},1) exceeds device limits \
             ({max_wg_x},{max_wg_y}) at M={m} N={n} K={k} — skipping this size"
        );
        return false;
    }

    // 50 GB cap: sum of 3 f16 matrices.
    let total_bytes: u64 = 2 * (m as u64) * (k as u64)
        + 2 * (k as u64) * (n as u64)
        + 2 * (m as u64) * (n as u64);
    const FIFTY_GB: u64 = 50 * 1024 * 1024 * 1024;
    if total_bytes >= FIFTY_GB {
        eprintln!(
            "resident_matmul_rb: 50 GB cap exceeded at M={m} N={n} K={k} \
             ({total_bytes} bytes) — skipping this size"
        );
        return false;
    }

    // M and N must be multiples of 32 (RB_M*16 = 2*16 = 32).
    if !m.is_multiple_of(32) || !n.is_multiple_of(32) {
        eprintln!("resident_matmul_rb: M={m} and/or N={n} not multiples of 32 — skipping");
        return false;
    }

    eprintln!(
        "resident_matmul_rb: ACTUAL dims M={m} N={n} K={k} \
         (wg_x={wg_x}, wg_y={wg_y})"
    );

    // Build f16 inputs.
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

    let workgroups = (wg_x as u32, wg_y as u32, 1);

    let (min_ns, ts) = resident_min_of_n(
        ctx, handle,
        &[&a_bytes, &b_bytes, &vec![0u8; c_size]],
        &[0, 0, c_size as u64],
        workgroups,
        pc,
    );

    let tflops: f64 = effective_tflops(m, n, k, min_ns);
    let pct: f64 = tflops / CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000 * 100.0_f64;

    let label = if ts == ResidentTimingSource::CpuFenceWall {
        // CpuFenceWall: omit % (scheduling-inclusive, not a GPU kernel time — AT-1582).
        format!(
            "matmul_rb_coopmat effective TFLOPS (honest) = {tflops:.3} \
             | {min_ns} ns ({}) | M={m} N={n} K={k} dispatch=({wg_x},{wg_y},1)",
            timing_source_label(ts)
        )
    } else if pct >= COMPETITIVE_PCT_THRESHOLD {
        // GpuTimestamp, high %: include "competitive" label.
        format!(
            "matmul_rb_coopmat effective TFLOPS (competitive) = {tflops:.3} \
             ({pct:.1}% of {CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000}-TFLOPS f32 datasheet \
             ESTIMATE — single-subgroup-per-block 2×2 RB, NOT a same-machine A/B) \
             | {min_ns} ns ({}) | M={m} N={n} K={k} dispatch=({wg_x},{wg_y},1)",
            timing_source_label(ts)
        )
    } else {
        // GpuTimestamp, modest %: honest label without "competitive".
        format!(
            "matmul_rb_coopmat effective TFLOPS (honest) = {tflops:.3} \
             ({pct:.1}% of {CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000}-TFLOPS f32 datasheet \
             ESTIMATE — single-subgroup-per-block 2×2 RB, NOT a same-machine A/B) \
             | {min_ns} ns ({}) | M={m} N={n} K={k} dispatch=({wg_x},{wg_y},1)",
            timing_source_label(ts)
        )
    };
    eprintln!("{label}");

    // Assert only tflops > 0 and finite (NO ratio assertion — AT-1730).
    assert!(
        tflops > 0.0_f64 && tflops.is_finite(),
        "effective_tflops must be > 0 and finite (AT-1730)"
    );

    true
}

/// AT-1730: register-blocked coopmat matmul honest effective-TFLOPS bench.
///
/// `dispatch_resident_matmul_rb_coopmat` — bench ID.
///
/// Measures at M=N=K=256 (occupancy-constrained: 64 workgroups < 188 SMs),
/// M=N=K=512 (larger size: 256 workgroups, better occupancy), and M=N=K=768 (if ≤ 50 GB).
/// All sizes within dispatch limits and 50 GB cap are measured and reported HONESTLY.
/// If 256³ regresses vs M3.3b (5.04 TFLOPS) but 512³ improves, both are reported.
/// Typed-skip on CoopMatUnsupported (Lavapipe).
fn bench_resident_matmul_rb(c: &mut Criterion) {
    if !gpu_benches_enabled() {
        eprintln!("resident_matmul_rb: AXC_ENABLE_GPU_BENCHES not set; skipping");
        return;
    }

    let ctx = match VulkanContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("resident_matmul_rb: no Vulkan: {e}");
            return;
        }
    };

    // Compile with RB 2×2 assignments.
    let assignments = rb2x2_assignments();
    let (bytes, meta) = match compile_source_with_assignments(MATMUL_RB_COOPMAT_SRC, &assignments) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("resident_matmul_rb: compile failed: {e:?}");
            return;
        }
    };
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    // Typed preflight via prepare_kernel_checked (coopmat).
    let handle = match ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, meta.coopmat.as_ref(), "matmul_rb_coopmat",
        meta.shared_memory_bytes,
    ) {
        Ok(h) => h,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("resident_matmul_rb: CoopMatUnsupported (typed-skip): {reason}");
            return;
        }
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
            eprintln!("resident_matmul_rb: DeviceFeatureUnsupported {feature}/{kernel} (typed-skip)");
            return;
        }
        Err(e) => {
            eprintln!("resident_matmul_rb: prepare_kernel_checked failed: {e}");
            return;
        }
    };

    let mut group = c.benchmark_group("resident_matmul_rb");
    group.sample_size(10);

    group.bench_function("dispatch_resident_matmul_rb_coopmat", |b| {
        b.iter_custom(|iters| {
            let mut total_ns: u64 = 0;
            for _ in 0..iters {
                // Measure at 256³ (baseline comparison point).
                // Occupancy note: 64 workgroups < ~188 SMs; may under-occupy vs M3.3b's 256 WGs.
                let _ = measure_rb_tflops(&ctx, &handle, 256, 256, 256);

                // Measure at 512³ (better occupancy: 256 workgroups at 512/32=16 per axis).
                let _ = measure_rb_tflops(&ctx, &handle, 512, 512, 512);

                // Measure at 768³ if within 50 GB and dispatch limits.
                // 768/32=24 workgroups per axis → (24,24,1)=576 workgroups.
                // Memory: 3 * 768*768 * 2 = 3.54 MB — well under 50 GB.
                let _ = measure_rb_tflops(&ctx, &handle, 768, 768, 768);

                // Use 256³ timing for criterion's duration accounting.
                // (Criterion only needs a non-zero duration; the honest TFLOPS are printed above.)
                let (ns_256, _) = resident_min_of_n(
                    &ctx, &handle,
                    &[
                        &(0..256_usize*256).flat_map(|i| {
                            use half::f16;
                            let h = f16::from_f32(((i % 4) + 1) as f32);
                            h.to_le_bytes()
                        }).collect::<Vec<u8>>(),
                        &(0..256_usize*256).flat_map(|i| {
                            use half::f16;
                            let h = f16::from_f32(((i % 3) + 1) as f32);
                            h.to_le_bytes()
                        }).collect::<Vec<u8>>(),
                        &vec![0u8; 256*256*2],
                    ],
                    &[0, 0, (256*256*2) as u64],
                    (8, 8, 1),
                    {
                        let mut pc = Vec::with_capacity(12);
                        pc.extend_from_slice(&(256u32).to_le_bytes());
                        pc.extend_from_slice(&(256u32).to_le_bytes());
                        pc.extend_from_slice(&(256u32).to_le_bytes());
                        pc
                    },
                );
                total_ns += ns_256;
            }
            std::time::Duration::from_nanos(total_ns)
        });
    });

    group.finish();
}

criterion_group!(resident_matmul_rb_benches, bench_resident_matmul_rb);
criterion_main!(resident_matmul_rb_benches);
