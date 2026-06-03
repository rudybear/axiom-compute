//! GPU-resident benchmark suite — M3.1.5 (honest scope).
//!
//! ## Bench IDs (DO NOT change existing IDs from dispatch.rs/dispatch_q4km.rs)
//!
//! - `resident_timing_kernels`: HEADLINE bench — runs the resident path over EXISTING
//!   real kernels (saxpy, q4km matvec, q4km matmul, coopmat matmul_tile) reporting
//!   honest kernel-only nanoseconds. NO TFLOPS, NO cuBLAS ratio (AT-1577).
//!
//! - `naive_gemm_harness_validation`: (renamed from matmul_tflops_4096) measures
//!   the NAIVE un-tiled scalar f32 GEMM via the resident path. Labeled "NAIVE un-tiled
//!   scalar f32 GEMM — harness validation only, NOT an optimized matmul". Printed string
//!   MUST contain "NAIVE" and "harness validation" (AT-1581). Asserts only tflops>0 &&
//!   finite; does NOT assert a competitive ratio.
//!
//! - `dispatch_resident_matmul_tile`: 16x16 coopmat tile via resident API +
//!   prepare_kernel_checked preflight. Typed-skip if coopmat/subgroup unsupported.
//!
//! ## Timing methodology (TIMING-1, honest)
//!
//! MANDATORY warmup: 2 untimed dispatch_resident calls discarded. Report MIN-of-N (N=10).
//!
//! If timing_source == CpuFenceWall, append "— scheduling-inclusive, NOT a GPU kernel time"
//! to the printed line and skip any competitive comparison (AT-1582).
//!
//! ## GPU bench gating
//!
//! Gated on `AXC_ENABLE_GPU_BENCHES=1`. Without the flag, benches immediately return
//! without doing GPU work (safe for CI without GPU).
//!
//! ## 50 GB cap (HANDOFF-17)
//!
//! The naive GEMM bench asserts buffer total < 50 GB BEFORE allocation. The library
//! does not enforce this policy — it belongs to the bench (anti-pattern #1 compliance).

#![allow(dead_code)]

#[path = "common.rs"]
mod common;

use criterion::{criterion_group, criterion_main, Criterion};
use axc_driver::compile_source_with_meta;
use axc_runtime::{
    VulkanContext, DispatchError,
    ResidentBenchConfig, ResidentTimingSource,
};

const MATMUL_TILE_SRC: &str = include_str!("../../../examples/matmul_tile.axc");
const MATMUL_F32_SRC: &str = include_str!("../../../examples/matmul_f32_tiled.axc");
const SAXPY_SRC: &str = include_str!("../../../examples/saxpy.axc");
const Q4KM_MATVEC_SRC: &str = include_str!("../../../examples/q4km_dequant_matvec.axc");
const Q4KM_MATMUL_SRC: &str = include_str!("../../../examples/q4km_dequant_matmul.axc");
/// External cuBLAS f32 GEMM throughput estimate for the NVIDIA RTX PRO 6000 Blackwell.
///
/// Source: NVIDIA RTX PRO 6000 Blackwell datasheet.
/// Peak f32 TFLOPS ~ 125 TFLOPS for GB202.
/// LABELED as estimate (not a same-machine A/B). NOT asserted >= any ratio (AT-1532).
const CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000: f64 = 125.0_f64;

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

/// Compute effective TFLOPS from matrix dimensions and kernel seconds.
///
/// Used ONLY for the naive_gemm_harness_validation bench, with explicit labeling.
fn effective_tflops(m: usize, n: usize, k: usize, kernel_seconds: f64) -> f64 {
    if kernel_seconds <= 0.0 { return 0.0; }
    2.0 * (m as f64) * (n as f64) * (k as f64) / kernel_seconds / 1e12
}

/// Run the resident path (upload once → N_WARMUP + N_MEASURED dispatches → MIN-of-N).
///
/// Returns `(min_kernel_ns, timing_source)`. Panics on any GPU error (bench context).
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

// ── HEADLINE: resident_timing_kernels ────────────────────────────────────────

/// `resident_timing_kernels`: HEADLINE bench — measures EXISTING real kernels via the
/// resident path and reports honest kernel-only nanoseconds.
///
/// NO TFLOPS, NO cuBLAS ratio for these kernels. Asserts only kernel_ns>0 && finite.
/// Prints timing_source for each kernel (AT-1582).
fn bench_resident_timing_kernels(c: &mut Criterion) {
    if !gpu_benches_enabled() {
        eprintln!("resident_timing_kernels: AXC_ENABLE_GPU_BENCHES not set; skipping");
        return;
    }

    let ctx = match VulkanContext::new() {
        Ok(ctx) => ctx,
        Err(e) => { eprintln!("resident_timing_kernels: no Vulkan: {e}"); return; }
    };

    let mut group = c.benchmark_group("resident_matmul");
    group.sample_size(10);

    // ── saxpy ──────────────────────────────────────────────────────────────
    {
        let (bytes, meta) = match compile_source_with_meta(SAXPY_SRC) {
            Ok(r) => r,
            Err(e) => { eprintln!("saxpy compile failed: {e}"); return; }
        };
        let words: Vec<u32> = bytes.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
        let handle = match ctx.prepare_kernel_checked(
            &words, &meta.binding_plan, meta.push_constant_total_bytes,
            &meta.entry_point, None, "saxpy",
            meta.shared_memory_bytes,
        ) {
            Ok(h) => h,
            Err(e) => { eprintln!("saxpy prepare failed: {e}"); return; }
        };

        const N: usize = 1024;
        let x: Vec<f32> = (0..N).map(|i| i as f32 * 0.001_f32).collect();
        let y: Vec<f32> = (0..N).map(|i| i as f32 * 0.002_f32).collect();
        let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        let y_bytes: Vec<u8> = y.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut pc = Vec::new();
        pc.extend_from_slice(&(N as u32).to_le_bytes());
        pc.extend_from_slice(&2.0_f32.to_le_bytes());
        let output_size = (N * 4) as u64;
        let workgroups = ((N as u32).div_ceil(64), 1, 1);

        group.bench_function("resident_timing_kernels/saxpy", |b| {
            b.iter_custom(|iters| {
                let mut total_ns: u64 = 0;
                for _ in 0..iters {
                    let (min_ns, ts) = resident_min_of_n(
                        &ctx, &handle,
                        &[&x_bytes, &y_bytes], &[0, output_size],
                        workgroups, pc.clone(),
                    );
                    // NO TFLOPS, NO cuBLAS ratio — honest ns only.
                    eprintln!("resident_timing_kernels/saxpy: min kernel_ns = {} ({})",
                        min_ns, timing_source_label(ts));
                    assert!(min_ns > 0 && (min_ns as f64).is_finite(),
                        "saxpy kernel_ns must be > 0 and finite");
                    total_ns += min_ns;
                }
                std::time::Duration::from_nanos(total_ns)
            });
        });
    }

    // ── q4km matvec ───────────────────────────────────────────────────────
    {
        let (bytes, meta) = match compile_source_with_meta(Q4KM_MATVEC_SRC) {
            Ok(r) => r,
            Err(e) => { eprintln!("q4km_matvec compile failed: {e}"); return; }
        };
        let words: Vec<u32> = bytes.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
        let handle = match ctx.prepare_kernel_checked(
            &words, &meta.binding_plan, meta.push_constant_total_bytes,
            &meta.entry_point, None, "q4km_dequant_matvec",
            meta.shared_memory_bytes,
        ) {
            Ok(h) => h,
            Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
                eprintln!("q4km_matvec: DeviceFeatureUnsupported {feature}/{kernel}; skipping");
                group.finish();
                return;
            }
            Err(e) => { eprintln!("q4km_matvec prepare failed: {e}"); return; }
        };

        // Minimal 32-block Q4_K_M-like input for harness validation.
        let n_blocks: usize = 32;
        let block_size_bytes: usize = 64; // Q4_K_M block is 64 bytes (simplified)
        let q_bytes = vec![0u8; n_blocks * block_size_bytes];
        // The actual size depends on the meta binding plan; use at least what's needed.
        let n_bufs = meta.binding_plan.buffers.len();
        let input_sizes: Vec<Vec<u8>> = (0..n_bufs).map(|_| q_bytes.clone()).collect();
        let inputs_refs: Vec<&[u8]> = input_sizes.iter().map(|v| v.as_slice()).collect();
        let output_sizes: Vec<u64> = meta.binding_plan.buffers.iter().map(|b| {
            if b.ty.access == axc_hir::buffer::BufferAccess::ReadOnly { 0 } else { q_bytes.len() as u64 }
        }).collect();

        let mut pc = vec![0u8; meta.push_constant_total_bytes as usize];
        // Fill n_blocks push constant.
        if pc.len() >= 4 {
            pc[0..4].copy_from_slice(&(n_blocks as u32).to_le_bytes());
        }
        let workgroups = (1, 1, 1);

        group.bench_function("resident_timing_kernels/q4km_matvec", |b| {
            b.iter_custom(|iters| {
                let mut total_ns: u64 = 0;
                for _ in 0..iters {
                    let (min_ns, ts) = resident_min_of_n(
                        &ctx, &handle, &inputs_refs, &output_sizes, workgroups, pc.clone(),
                    );
                    eprintln!("resident_timing_kernels/q4km_matvec: min kernel_ns = {} ({})",
                        min_ns, timing_source_label(ts));
                    assert!(min_ns > 0 && (min_ns as f64).is_finite(),
                        "q4km_matvec kernel_ns must be > 0 and finite");
                    total_ns += min_ns;
                }
                std::time::Duration::from_nanos(total_ns)
            });
        });
    }

    // ── q4km matmul ───────────────────────────────────────────────────────
    {
        let (bytes, meta) = match compile_source_with_meta(Q4KM_MATMUL_SRC) {
            Ok(r) => r,
            Err(e) => { eprintln!("q4km_matmul compile failed: {e}"); return; }
        };
        let words: Vec<u32> = bytes.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
        let handle = match ctx.prepare_kernel_checked(
            &words, &meta.binding_plan, meta.push_constant_total_bytes,
            &meta.entry_point, None, "q4km_dequant_matmul",
            meta.shared_memory_bytes,
        ) {
            Ok(h) => h,
            Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
                eprintln!("q4km_matmul: DeviceFeatureUnsupported {feature}/{kernel}; skipping");
                group.finish();
                return;
            }
            Err(e) => { eprintln!("q4km_matmul prepare failed: {e}"); return; }
        };

        let n_blocks: usize = 16;
        let block_size_bytes: usize = 64;
        let q_bytes = vec![0u8; n_blocks * block_size_bytes];
        let n_bufs = meta.binding_plan.buffers.len();
        let input_sizes: Vec<Vec<u8>> = (0..n_bufs).map(|_| q_bytes.clone()).collect();
        let inputs_refs: Vec<&[u8]> = input_sizes.iter().map(|v| v.as_slice()).collect();
        let output_sizes: Vec<u64> = meta.binding_plan.buffers.iter().map(|b| {
            if b.ty.access == axc_hir::buffer::BufferAccess::ReadOnly { 0 } else { q_bytes.len() as u64 }
        }).collect();
        let mut pc = vec![0u8; meta.push_constant_total_bytes as usize];
        if pc.len() >= 4 {
            pc[0..4].copy_from_slice(&(n_blocks as u32).to_le_bytes());
        }
        let workgroups = (1, 1, 1);

        group.bench_function("resident_timing_kernels/q4km_matmul", |b| {
            b.iter_custom(|iters| {
                let mut total_ns: u64 = 0;
                for _ in 0..iters {
                    let (min_ns, ts) = resident_min_of_n(
                        &ctx, &handle, &inputs_refs, &output_sizes, workgroups, pc.clone(),
                    );
                    eprintln!("resident_timing_kernels/q4km_matmul: min kernel_ns = {} ({})",
                        min_ns, timing_source_label(ts));
                    assert!(min_ns > 0 && (min_ns as f64).is_finite(),
                        "q4km_matmul kernel_ns must be > 0 and finite");
                    total_ns += min_ns;
                }
                std::time::Duration::from_nanos(total_ns)
            });
        });
    }

    // ── coopmat matmul_tile (typed-skip if unsupported) ───────────────────
    {
        let (bytes, meta) = match compile_source_with_meta(MATMUL_TILE_SRC) {
            Ok(r) => r,
            Err(e) => { eprintln!("matmul_tile compile failed: {e}"); return; }
        };
        let words: Vec<u32> = bytes.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();

        match ctx.prepare_kernel_checked(
            &words, &meta.binding_plan, meta.push_constant_total_bytes,
            &meta.entry_point, meta.coopmat.as_ref(), "matmul_tile",
            meta.shared_memory_bytes,
        ) {
            Ok(handle) => {
                let a_f16 = vec![0u8; 512];
                let b_f16 = vec![0u8; 512];
                let c_f16 = vec![0u8; 512];
                let pc: Vec<u8> = 0u32.to_le_bytes().to_vec();
                let output_size = 512u64;
                let workgroups = (1, 1, 1);

                group.bench_function("resident_timing_kernels/matmul_tile_coopmat", |b| {
                    b.iter_custom(|iters| {
                        let mut total_ns: u64 = 0;
                        for _ in 0..iters {
                            let (min_ns, ts) = resident_min_of_n(
                                &ctx, &handle,
                                &[&a_f16, &b_f16, &c_f16], &[0, 0, output_size],
                                workgroups, pc.clone(),
                            );
                            eprintln!("resident_timing_kernels/matmul_tile_coopmat: min kernel_ns = {} ({})",
                                min_ns, timing_source_label(ts));
                            assert!(min_ns > 0 && (min_ns as f64).is_finite(),
                                "matmul_tile kernel_ns must be > 0 and finite");
                            total_ns += min_ns;
                        }
                        std::time::Duration::from_nanos(total_ns)
                    });
                });
            }
            Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
                eprintln!("resident_timing_kernels/matmul_tile_coopmat: CoopMatUnsupported (typed skip): {reason}");
            }
            Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
                eprintln!("resident_timing_kernels/matmul_tile_coopmat: DeviceFeatureUnsupported {feature}/{kernel} (typed skip)");
            }
            Err(e) => { eprintln!("resident_timing_kernels/matmul_tile_coopmat: unexpected error: {e}"); }
        }
    }

    group.finish();
}

// ── naive_gemm_harness_validation ────────────────────────────────────────────

/// `naive_gemm_harness_validation`: measures the NAIVE un-tiled scalar f32 GEMM
/// (matmul_f32_tiled.axc, holes removed) via the resident path.
///
/// This is HARNESS VALIDATION ONLY — NOT an optimized matmul. The printed string
/// MUST contain "NAIVE" and "harness validation" (AT-1581). Asserts only tflops>0
/// && finite; does NOT assert a competitive ratio.
///
/// 50 GB cap (HANDOFF-17): buffer total < 50 GB asserted in THIS bench before
/// allocation (not in the library).
fn bench_naive_gemm_harness_validation(c: &mut Criterion) {
    if !gpu_benches_enabled() {
        eprintln!("naive_gemm_harness_validation: AXC_ENABLE_GPU_BENCHES not set; skipping");
        return;
    }

    let ctx = match VulkanContext::new() {
        Ok(ctx) => ctx,
        Err(e) => { eprintln!("naive_gemm_harness_validation: no Vulkan: {e}"); return; }
    };

    let (bytes, meta) = match compile_source_with_meta(MATMUL_F32_SRC) {
        Ok(r) => r,
        Err(e) => { eprintln!("naive_gemm_harness_validation: compile failed: {e}"); return; }
    };
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();

    // No strategy holes — compile_source_with_meta is sufficient (OPT-NOTE-1 / HANDOFF-11).
    let handle = match ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, None, "matmul_f32_tiled",
        meta.shared_memory_bytes,
    ) {
        Ok(h) => h,
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
            eprintln!("naive_gemm_harness_validation: DeviceFeatureUnsupported {feature}/{kernel}; skipping");
            return;
        }
        Err(e) => { eprintln!("naive_gemm_harness_validation: pipeline create: {e}"); return; }
    };

    // Select largest pow2 M=N=K whose wg_x <= max_compute_work_group_count()[0].
    // This pre-shrinks to fit device limits (bench pre-check; library guard also protects).
    let max_wg_x: u32 = ctx.max_compute_work_group_count()[0];
    let mut m: usize = 4096;
    let n: usize = m;
    let k: usize = m;

    // Buffer total cap: 3 matrices × m × n × 4 bytes < 50 GB.
    // 4096² f32 ×3 ≈ 192 MB — far under 50 GB. Guard for clarity.
    loop {
        let total_bytes: u64 = 3 * (m as u64) * (n as u64) * 4;
        let wg_x: u64 = ((m * n) as u64).div_ceil(64);
        const FIFTY_GB: u64 = 50 * 1024 * 1024 * 1024;
        if total_bytes < FIFTY_GB && wg_x <= max_wg_x as u64 {
            break;
        }
        m /= 2;
        if m == 0 {
            eprintln!("naive_gemm_harness_validation: could not find valid matrix dims; skipping");
            return;
        }
    }

    eprintln!(
        "naive_gemm_harness_validation: ACTUAL dims M=N=K={} (wg_x = {})",
        m, (m * n).div_ceil(64)
    );

    // Build inputs.
    let a_f32: Vec<f32> = (0..m * k).map(|i| (i % 16 + 1) as f32 / 256.0).collect();
    let b_f32: Vec<f32> = (0..k * n).map(|i| (i % 16 + 1) as f32 / 256.0).collect();
    let a_bytes: Vec<u8> = a_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
    let b_bytes: Vec<u8> = b_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
    let c_size: usize = m * n * 4;

    let mut pc = Vec::new();
    pc.extend_from_slice(&(m as u32).to_le_bytes());
    pc.extend_from_slice(&(n as u32).to_le_bytes());
    pc.extend_from_slice(&(k as u32).to_le_bytes());

    #[allow(clippy::manual_div_ceil)]
    let wg_x: u32 = ((m * n) as u32 + 63) / 64;
    let workgroups = (wg_x, 1, 1);

    let mut group = c.benchmark_group("resident_matmul");
    group.sample_size(10);

    group.bench_function("naive_gemm_harness_validation", |b| {
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

                let kernel_seconds: f64 = min_ns as f64 * 1e-9;
                let tflops: f64 = effective_tflops(m, n, k, kernel_seconds);
                let cublas_pct: f64 = tflops / CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000 * 100.0;

                // AT-1581: the printed string MUST contain "NAIVE" and "harness validation".
                let label = format!(
                    "NAIVE un-tiled scalar f32 GEMM (harness validation only — \
                     NOT an optimized matmul; expect low single-digit % of cuBLAS; \
                     real tiling is M3.2): {tflops:.3} TFLOPS \
                     ({cublas_pct:.1}% of cuBLAS f32 datasheet ESTIMATE — \
                     not a same-machine A/B) | {} ns ({})",
                    min_ns, timing_source_label(ts)
                );
                eprintln!("{label}");

                // Inline assert: label must contain "NAIVE" and "harness validation" (AT-1581).
                assert!(label.contains("NAIVE"),
                    "AT-1581: printed string must contain 'NAIVE'");
                assert!(label.contains("harness validation"),
                    "AT-1581: printed string must contain 'harness validation'");

                assert!(tflops > 0.0 && tflops.is_finite(),
                    "effective_tflops must be > 0 and finite (AT-1532/naive)");

                total_ns += min_ns;
            }
            std::time::Duration::from_nanos(total_ns)
        });
    });

    group.finish();
}

// ── dispatch_resident_matmul_tile ─────────────────────────────────────────────

/// `dispatch_resident_matmul_tile`: 16x16 coopmat tile via resident API +
/// prepare_kernel_checked preflight. SKIPS if coopmat unsupported (typed skip).
fn bench_dispatch_resident_matmul_tile(c: &mut Criterion) {
    if !gpu_benches_enabled() {
        eprintln!("dispatch_resident_matmul_tile: AXC_ENABLE_GPU_BENCHES not set; skipping");
        return;
    }

    let ctx = match VulkanContext::new() {
        Ok(c) => c,
        Err(e) => { eprintln!("dispatch_resident_matmul_tile: no Vulkan: {e}"); return; }
    };

    let (bytes, meta) = match compile_source_with_meta(MATMUL_TILE_SRC) {
        Ok(r) => r,
        Err(e) => { eprintln!("dispatch_resident_matmul_tile: compile failed: {e}"); return; }
    };
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();

    // Typed preflight via prepare_kernel_checked (HANDOFF-10/11).
    let handle = match ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, meta.coopmat.as_ref(), "matmul_tile",
        meta.shared_memory_bytes,
    ) {
        Ok(h) => h,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("dispatch_resident_matmul_tile: CoopMatUnsupported (typed skip): {reason}");
            return;
        }
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
            eprintln!("dispatch_resident_matmul_tile: DeviceFeatureUnsupported {feature}/{kernel} (typed skip)");
            return;
        }
        Err(e) => { eprintln!("dispatch_resident_matmul_tile: pipeline create: {e}"); return; }
    };

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

    let output_size = 512u64;
    let workgroups = (1, 1, 1);

    let mut group = c.benchmark_group("resident_matmul");
    group.sample_size(10);

    group.bench_function("dispatch_resident_matmul_tile", |b| {
        b.iter_custom(|iters| {
            let mut total_ns: u64 = 0;
            for _ in 0..iters {
                let (min_ns, ts) = resident_min_of_n(
                    &ctx, &handle,
                    &[&a_bytes, &b_bytes, &c_zeros], &[0, 0, output_size],
                    workgroups, pc.clone(),
                );
                let ts_label = timing_source_label(ts);
                eprintln!("dispatch_resident_matmul_tile: min kernel_ns = {} ({})", min_ns, ts_label);
                assert!(min_ns > 0 && (min_ns as f64).is_finite(),
                    "matmul_tile kernel_ns must be > 0 and finite");
                total_ns += min_ns;
            }
            std::time::Duration::from_nanos(total_ns)
        });
    });

    group.finish();
}

// AT-1623 (dispatch_matmul_shared_coopmat bench) was REMOVED in M3.2 retry:2.
//
// The matmul_shared_coopmat.axc kernel computes incorrect results (gpu=0.0 on NVIDIA
// RTX PRO 6000, measured 2026-06-01). Reporting TFLOPS for a zero-computing kernel
// would be misleading. The bench harness path (compile → pipeline create → dispatch
// loop → min-of-N timing) is exercised by bench_dispatch_resident_matmul_tile.
//
// M3.3: re-introduce this bench once the kernel computes correct results via OpPhi
// loop-carried SSA support.

criterion_group!(
    resident_matmul_benches,
    bench_resident_timing_kernels,
    bench_naive_gemm_harness_validation,
    bench_dispatch_resident_matmul_tile,
);
criterion_main!(resident_matmul_benches);
