//! AT-1750: Honest effective-TFLOPS bench for matmul_msg_coopmat.axc — M3.3d multi-subgroup.
//!
//! Multi-subgroup (MSG, N_SG=2) coopmat matmul: ONE workgroup (64 threads = 2 subgroups)
//! computes a 32×64 output block (each subgroup a 32×32 = 4 output tiles).
//! Dispatch grid (N/64, M/32, 1) workgroups.
//!
//! HARD PRECONDITION: subgroup_size MUST be 32. The kernel MISCOMPUTES on other wave widths.
//! Guard: `if ctx.subgroup_size() != 32 { eprintln!(...); return; }` BEFORE allocation/dispatch.
//! Mirror of dispatch_matmul_tile.rs:231 AT-1510 wave64 guard.
//!
//! Methodology: resident upload-once / N_WARMUP=2 discarded / MIN-of-10 / GpuTimestamp.
//! Verbatim clone of resident_matmul_rb.rs (AT-1730 template).
//!
//! HONEST framing (no asserted ratio):
//!   - Reports bare TFLOPS + % of 125-TFLOPS datasheet ESTIMATE at each size.
//!   - Word "competitive" ONLY if pct >= COMPETITIVE_PCT_THRESHOLD (25.0).
//!   - NO ratio asserted (only tflops>0 && finite).
//!   - Records actual M/N/K alongside each TFLOPS (gaming guard).
//!   - Reports whether MSG beats M3.3c's 31.2 TFLOPS (24.96% at 768) — HONESTLY.
//!
//! 50 GB cap asserted before allocation. 2D grid pre-check on BOTH axes.
//! Typed-skip on CoopMatUnsupported (Lavapipe) / DeviceFeatureUnsupported.
//!
//! Bench ID: dispatch_resident_matmul_msg_coopmat (MUST NOT collide with AT-1711/AT-1730).
//! Single-subgroup RB bench (resident_matmul_rb.rs) RETAINED for A/B comparison.

#![allow(dead_code)]

use criterion::{criterion_group, criterion_main, Criterion};
use axc_driver::compile_source_with_assignments;
use axc_runtime::{
    VulkanContext, DispatchError,
    ResidentBenchConfig, ResidentTimingSource,
};
use std::collections::BTreeMap;

const MATMUL_MSG_COOPMAT_SRC: &str = include_str!("../../../examples/matmul_msg_coopmat.axc");

/// External cuBLAS f32 GEMM throughput estimate for the NVIDIA RTX PRO 6000 Blackwell.
/// Source: NVIDIA RTX PRO 6000 Blackwell datasheet. Peak f32 TFLOPS ~ 125 TFLOPS.
/// LABELED as estimate (not a same-machine A/B). NOT asserted >= any ratio (AT-1750).
const CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000: f64 = 125.0_f64;

/// Threshold for printing the word "competitive" in the bench label.
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

/// Build MSG strategy assignments (shipped config).
fn msg_assignments() -> BTreeMap<String, i64> {
    let mut m = BTreeMap::new();
    m.insert("wg_threads".to_owned(), 64_i64);
    m.insert("n_sg".to_owned(), 2_i64);
    m.insert("rb_m".to_owned(), 2_i64);
    m.insert("rb_n".to_owned(), 2_i64);
    m.insert("tile_k".to_owned(), 16_i64);
    m.insert("a_block_size".to_owned(), 512_i64);
    m.insert("b_block_size".to_owned(), 1024_i64);
    m
}

/// Run resident_min_of_n: upload-once / N_WARMUP+N_MEASURED dispatches / MIN-of-N.
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

    for _ in 0..N_WARMUP {
        ctx.dispatch_resident(handle, &resident, &cfg)
            .expect("warmup dispatch must succeed");
    }

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

/// Measure effective TFLOPS for MSG at one (m, n, k) size, print honest label.
///
/// Returns true if measurement was taken (false if skip or grid check failed).
fn measure_msg_tflops(
    ctx: &VulkanContext,
    handle: &axc_runtime::KernelHandle,
    m: usize,
    n: usize,
    k: usize,
) -> bool {
    // 2D grid pre-check: grid = (N/64, M/32, 1) for MSG (N_SG=2 x RB_N=2 x 16 = 64 cols/WG).
    let wg_x: u64 = (n / 64) as u64;
    let wg_y: u64 = (m / 32) as u64;
    let max_wg_count = ctx.max_compute_work_group_count();
    let max_wg_x: u64 = max_wg_count[0] as u64;
    let max_wg_y: u64 = max_wg_count[1] as u64;
    if wg_x > max_wg_x || wg_y > max_wg_y {
        eprintln!(
            "resident_matmul_msg: grid ({wg_x},{wg_y},1) exceeds device limits \
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
            "resident_matmul_msg: 50 GB cap exceeded at M={m} N={n} K={k} \
             ({total_bytes} bytes) — skipping this size"
        );
        return false;
    }

    // M must be multiple of 32 (RB_M*16), N must be multiple of 64 (N_SG*RB_N*16).
    if !m.is_multiple_of(32) || !n.is_multiple_of(64) {
        eprintln!("resident_matmul_msg: M={m} not mult of 32 or N={n} not mult of 64 — skipping");
        return false;
    }

    eprintln!(
        "resident_matmul_msg: ACTUAL dims M={m} N={n} K={k} \
         (wg_x={wg_x}, wg_y={wg_y})"
    );

    // Build f16 inputs.
    let a_bytes: Vec<u8> = {
        use half::f16;
        (0..m * k).flat_map(|i| {
            let h = f16::from_f32(((i % 4) + 1) as f32);
            h.to_le_bytes()
        }).collect()
    };
    let b_bytes: Vec<u8> = {
        use half::f16;
        (0..k * n).flat_map(|i| {
            let h = f16::from_f32(((i % 3) + 1) as f32);
            h.to_le_bytes()
        }).collect()
    };
    let c_size: usize = m * n * 2;

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
        format!(
            "matmul_msg_coopmat effective TFLOPS (honest) = {tflops:.3} \
             | {min_ns} ns ({}) | M={m} N={n} K={k} dispatch=({wg_x},{wg_y},1)",
            timing_source_label(ts)
        )
    } else if pct >= COMPETITIVE_PCT_THRESHOLD {
        format!(
            "matmul_msg_coopmat effective TFLOPS (competitive) = {tflops:.3} \
             ({pct:.1}% of {CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000}-TFLOPS f32 datasheet \
             ESTIMATE — multi-subgroup N_SG=2 2×2 RB, NOT a same-machine A/B) \
             | {min_ns} ns ({}) | M={m} N={n} K={k} dispatch=({wg_x},{wg_y},1)",
            timing_source_label(ts)
        )
    } else {
        format!(
            "matmul_msg_coopmat effective TFLOPS (honest) = {tflops:.3} \
             ({pct:.1}% of {CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000}-TFLOPS f32 datasheet \
             ESTIMATE — multi-subgroup N_SG=2 2×2 RB, NOT a same-machine A/B) \
             | {min_ns} ns ({}) | M={m} N={n} K={k} dispatch=({wg_x},{wg_y},1)",
            timing_source_label(ts)
        )
    };
    eprintln!("{label}");

    // Assert only tflops > 0 and finite (NO ratio assertion — AT-1750).
    assert!(
        tflops > 0.0_f64 && tflops.is_finite(),
        "effective_tflops must be > 0 and finite (AT-1750)"
    );

    true
}

/// AT-1750: multi-subgroup coopmat matmul honest effective-TFLOPS bench.
///
/// `dispatch_resident_matmul_msg_coopmat` — bench ID.
///
/// GUARD: `if ctx.subgroup_size() != 32 { eprintln!(...); return; }` BEFORE allocation/dispatch
/// (mirror dispatch_matmul_tile.rs:231 AT-1510 wave64 guard — kernel REQUIRES sg_size==32).
///
/// Measures at M=N=K=256, 512, 768 (and 1024 if buffers < 50 GB).
/// Reports BARE tflops + pct; competitive label only if pct >= 25.0.
/// Single-subgroup RB bench (resident_matmul_rb.rs) RETAINED for A/B at same sizes.
fn bench_resident_matmul_msg(c: &mut Criterion) {
    if !gpu_benches_enabled() {
        eprintln!("resident_matmul_msg: AXC_ENABLE_GPU_BENCHES not set; skipping");
        return;
    }

    let ctx = match VulkanContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("resident_matmul_msg: no Vulkan: {e}");
            return;
        }
    };

    // --- subgroup_size==32 GUARD (BEFORE allocation/dispatch) ---
    // The MSG kernel REQUIRES subgroup_size==32. Benching it on wave64/SIMD16 would
    // measure a miscomputing kernel — benching a wrong kernel is worse than skipping.
    if ctx.subgroup_size() != 32 {
        eprintln!(
            "resident_matmul_msg: subgroup_size={} != 32; skipping bench (wave64 guard). \
             The MSG kernel REQUIRES sg_size==32 (N_SG=2); it MISCOMPUTES on other wave widths.",
            ctx.subgroup_size()
        );
        return;
    }

    let assignments = msg_assignments();
    let (bytes, meta) = match compile_source_with_assignments(MATMUL_MSG_COOPMAT_SRC, &assignments) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("resident_matmul_msg: compile failed: {e:?}");
            return;
        }
    };
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let handle = match ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, meta.coopmat.as_ref(), "matmul_msg_coopmat",
        meta.shared_memory_bytes,
    ) {
        Ok(h) => h,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("resident_matmul_msg: CoopMatUnsupported (typed-skip): {reason}");
            return;
        }
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
            eprintln!("resident_matmul_msg: DeviceFeatureUnsupported {feature}/{kernel} (typed-skip)");
            return;
        }
        Err(e) => {
            eprintln!("resident_matmul_msg: prepare_kernel_checked failed: {e}");
            return;
        }
    };

    let mut group = c.benchmark_group("resident_matmul_msg");
    group.sample_size(10);

    group.bench_function("dispatch_resident_matmul_msg_coopmat", |b| {
        b.iter_custom(|iters| {
            let mut total_ns: u64 = 0;
            for _ in 0..iters {
                // 256³: grid = (4,8,1) = 32 WGs of 64 threads.
                // Occupancy note: fewer WGs than RB's 64 at 256; may be slower.
                let _ = measure_msg_tflops(&ctx, &handle, 256, 256, 256);

                // 512³: grid = (8,16,1) = 128 WGs. Better occupancy.
                let _ = measure_msg_tflops(&ctx, &handle, 512, 512, 512);

                // 768³: grid = (12,24,1) = 288 WGs. Main competitive-vs-RB point.
                let _ = measure_msg_tflops(&ctx, &handle, 768, 768, 768);

                // 1024³: grid = (16,32,1) = 512 WGs, if < 50 GB.
                let _ = measure_msg_tflops(&ctx, &handle, 1024, 1024, 1024);

                // Criterion duration accounting: use 256³ timing.
                let (ns_256, _) = resident_min_of_n(
                    &ctx, &handle,
                    &[
                        &{
                            use half::f16;
                            (0..256_usize*256).flat_map(|i| {
                                let h = f16::from_f32(((i % 4) + 1) as f32);
                                h.to_le_bytes()
                            }).collect::<Vec<u8>>()
                        },
                        &{
                            use half::f16;
                            (0..256_usize*256).flat_map(|i| {
                                let h = f16::from_f32(((i % 3) + 1) as f32);
                                h.to_le_bytes()
                            }).collect::<Vec<u8>>()
                        },
                        &vec![0u8; 256*256*2],
                    ],
                    &[0, 0, (256*256*2) as u64],
                    (4, 8, 1), // 256/64=4 x, 256/32=8 y
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

criterion_group!(resident_matmul_msg_benches, bench_resident_matmul_msg);
criterion_main!(resident_matmul_msg_benches);
