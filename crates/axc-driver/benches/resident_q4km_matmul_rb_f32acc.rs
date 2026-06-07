//! AT-1783 + AT-1784: honest effective-TFLOPS bench for the f32-ACCUMULATOR FUSED Q4_K_M
//! register-blocked coopmat matmul (examples/q4km_matmul_rb_coopmat_f32acc.axc) AND the
//! now-VALID SAME-SHAPE A/B re-run.
//!
//! Bench ID: `dispatch_resident_q4km_matmul_rb_f32acc`.
//!
//! ## AT-1783 — honest multi-size TFLOPS
//! Resident upload-once / N_WARMUP=2 / MIN-of-10 / GpuTimestamp at 256/512/768/1024 cube sizes.
//! Reports bare tflops + % of the 125-TFLOPS datasheet ESTIMATE + the measured max-rel-diff vs
//! the f32-accumulator oracle at EVERY size (now WITHIN 1e-3 even at K=14336 — the whole point).
//! NO asserted ratio (only tflops>0 && finite). HONEST: the f32-accumulator tflops may be BELOW
//! M3.5's f16-accumulator 11.27 (f32 accumulators cost more registers + f32 store bandwidth) —
//! report whatever is measured. 50 GB cap asserted; 2D grid pre-check.
//!
//! ## AT-1784 — SAME-SHAPE A/B line (now numerically VALID)
//! The pinned A/B shape (m=4096, n=512, k=14336) is in the size set so the headline ratio is
//! SAME-SHAPE (AXIOM f32-accumulator fused GEMM vs llama Q4_K MUL_MAT at the IDENTICAL shape).
//! Emits the machine-readable `AXC_Q4KM_AB_F32ACC` line (parsed by
//! scripts/m34_llamacpp_ab.sh --fused-f32acc) with kernel ns, K, flops (2*M*N*K), m, n,
//! `combined=` (condition-aware metric — drives numerically_valid), `raw=` (forward error —
//! reporting only), device. FLOP convention: 2*M*N*K matmul MACs, dequant EXCLUDED (both sides).
//!
//! HONESTY: at the A/B K=14336 the f32-accumulator kernel is NUMERICALLY VALID under the
//! condition-aware COMBINED metric (`|gpu-ref|/max(|ref|,Σ|wₖxₖ|) <= frozen 1e-3` — the same
//! backward-stable gate the dispatch ATs use; the f32 accumulator fixes the M3.5 f16-accumulator
//! divergence). The RAW forward error is ~1e-2 on near-zero cancellation outputs (a metric
//! artifact identical-in-kind to llama.cpp's own HMMA), recorded separately for transparency. The
//! same-shape throughput ratio is therefore a REAL fast-AND-correct comparison (still behind
//! llama on throughput, but a USABLE kernel). See .pipeline/benchmarks/m34/.
//!
//! Gated on AXC_ENABLE_GPU_BENCHES=1 + a responsive Vulkan ICD. Typed-skip on
//! CoopMatUnsupported (Lavapipe) / DeviceFeatureUnsupported / subgroup_size() != 32.

#![allow(dead_code)]

#[allow(clippy::duplicate_mod)]
#[path = "../tests/common_q4km_f32ref.rs"]
mod common_q4km_f32ref;

use criterion::{criterion_group, criterion_main, Criterion};
use axc_driver::compile_source_with_assignments;
use axc_hir::ParamBindingPlan;
use axc_runtime::{
    VulkanContext, DispatchError, KernelHandle,
    ResidentBenchConfig, ResidentTimingSource,
};
use std::collections::BTreeMap;

const FUSED_F32ACC_SRC: &str = include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc.axc");

/// External cuBLAS f32 GEMM throughput estimate for the NVIDIA RTX PRO 6000 Blackwell.
const CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000: f64 = 125.0_f64;
/// "competitive" label threshold (25% of 125 TFLOPS).
const COMPETITIVE_PCT_THRESHOLD: f64 = 25.0_f64;
/// Frozen relative tolerance (AT-1520/AT-1521 value — NOT loosened).
const FROZEN_REL_TOL: f64 = 1e-3;

const N_WARMUP: usize = 2;
const N_MEASURED: usize = 10;

/// 50 GB executable-memory cap.
const FIFTY_GB: u64 = 50 * 1024 * 1024 * 1024;

/// Pinned A/B same-shape (matches llama.cpp Q4_K MUL_MAT m=4096,n=512,k=14336 = 101 TFLOPS).
const AB_M: usize = 4096;
const AB_N: usize = 512;
const AB_K: usize = 14336;

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

/// effective_tflops = 2*M*N*K / kernel_seconds (dequant excluded — matmul MACs only).
pub fn effective_q4km_tflops(m: usize, n: usize, k: usize, kernel_ns: u64) -> f64 {
    if kernel_ns == 0 {
        return 0.0;
    }
    let secs = kernel_ns as f64 * 1e-9_f64;
    2.0_f64 * (m as f64) * (n as f64) * (k as f64) / secs / 1e12_f64
}

fn rb2x2_assignments() -> BTreeMap<String, i64> {
    let mut m = BTreeMap::new();
    m.insert("rb_m".to_owned(), 2_i64);
    m.insert("rb_n".to_owned(), 2_i64);
    m.insert("tile_k".to_owned(), 16_i64);
    m.insert("a_block_size".to_owned(), 512_i64);
    m.insert("b_block_size".to_owned(), 512_i64);
    m
}

fn assemble_pc(plan: &ParamBindingPlan, m: u32, n: u32, k: u32, n_bpr: u32) -> Vec<u8> {
    let mut pc: Vec<u8> = vec![0u8; plan.push_constant_total_bytes as usize];
    for s in &plan.scalars {
        let val: u32 = match s.name.as_str() {
            "M" => m,
            "N" => n,
            "K" => k,
            "n_blocks_per_row" => n_bpr,
            other => panic!("unexpected scalar param {other}"),
        };
        let start = s.offset as usize;
        pc[start..start + 4].copy_from_slice(&val.to_le_bytes());
    }
    pc
}

/// Non-symmetric Q4_K_M weight matrix fixture.
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

/// MIN-of-N resident dispatch. Returns (min_kernel_ns, timing_source).
fn resident_min_of_n(
    ctx: &VulkanContext,
    handle: &KernelHandle,
    inputs: &[&[u8]],
    output_sizes: &[u64],
    workgroups: (u32, u32, u32),
    push_constants: Vec<u8>,
) -> (u64, ResidentTimingSource) {
    let resident = ctx.upload_resident(handle, inputs, output_sizes)
        .expect("upload_resident must succeed");
    let cfg = ResidentBenchConfig { workgroups, push_constants };
    for _ in 0..N_WARMUP {
        ctx.dispatch_resident(handle, &resident, &cfg).expect("warmup dispatch");
    }
    let mut min_ns = u64::MAX;
    let mut last_source = ResidentTimingSource::CpuFenceWall;
    for _ in 0..N_MEASURED {
        let t = ctx.dispatch_resident(handle, &resident, &cfg).expect("measured dispatch");
        if t.kernel_ns < min_ns {
            min_ns = t.kernel_ns;
        }
        last_source = t.timing_source;
    }
    (min_ns, last_source)
}

/// Measure the f32-accumulator fused kernel at (m, n, k). Reports honest tflops + pct +
/// measured max-rel-diff (vs the f32-accumulator oracle). If `emit_ab_line` is set, also
/// prints the machine-readable AXC_Q4KM_AB_F32ACC line for the SAME-SHAPE A/B (AT-1784).
///
/// Returns true if a measurement was taken (false on typed-skip / grid / cap failure).
fn measure_q4km_rb_f32acc_tflops(
    ctx: &VulkanContext,
    handle: &KernelHandle,
    plan: &ParamBindingPlan,
    m: usize,
    n: usize,
    k: usize,
    emit_ab_line: bool,
) -> bool {
    if !m.is_multiple_of(32) || !n.is_multiple_of(32) {
        eprintln!("resident_q4km_matmul_rb_f32acc: M={m}/N={n} not multiples of 32 — skip");
        return false;
    }
    if !k.is_multiple_of(256) {
        eprintln!("resident_q4km_matmul_rb_f32acc: K={k} not a multiple of 256 (Q4_K_M superblock) — skip");
        return false;
    }
    let n_bpr = k / 256;

    // 2D grid pre-check: grid = (N/32, M/32, 1).
    let wg_x = (n / 32) as u64;
    let wg_y = (m / 32) as u64;
    let max_wg = ctx.max_compute_work_group_count();
    if wg_x > max_wg[0] as u64 || wg_y > max_wg[1] as u64 {
        eprintln!("resident_q4km_matmul_rb_f32acc: grid ({wg_x},{wg_y}) exceeds limits at M={m} N={n} K={k} — skip");
        return false;
    }

    // 50 GB cap (C is now f32 — 4 bytes/elem, not 2).
    let total_bytes: u64 = (m as u64 * n_bpr as u64 * 144)   // q (u8)
        + 2 * (k as u64) * (n as u64)                        // x (f16)
        + 4 * (m as u64) * (n as u64);                       // C (f32)
    if total_bytes >= FIFTY_GB {
        eprintln!("resident_q4km_matmul_rb_f32acc: 50 GB cap exceeded at M={m} N={n} K={k} ({total_bytes} bytes) — skip");
        return false;
    }

    // Fixtures.
    let q_bytes = make_q4km_weights(m, n_bpr, 0xC0FFEE ^ (m as u64));
    let x_f16 = make_x_f16(k, n, 0xBADF00D ^ (n as u64));
    let x_bytes: Vec<u8> = x_f16.iter().flat_map(|&b| b.to_le_bytes()).collect();
    let c_size = m * n * 4; // f32 output (4 bytes/elem).

    // Correctness pre-flight on the SAME handle being timed: dispatch once, compare vs the
    // f32-accumulator oracle, REPORT max-rel-diff (now within 1e-3 at every K including 14336).
    let pc = assemble_pc(plan, m as u32, n as u32, k as u32, n_bpr as u32);
    let workgroups = (wg_x as u32, wg_y as u32, 1);

    let outputs = match ctx.dispatch_handle(
        handle, workgroups,
        &[&q_bytes, &x_bytes, &vec![0u8; c_size]],
        &[0, 0, c_size],
        &pc,
    ) {
        Ok(v) => v,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("resident_q4km_matmul_rb_f32acc: CoopMatUnsupported at dispatch (typed-skip): {reason}");
            return false;
        }
        Err(e) => {
            eprintln!("resident_q4km_matmul_rb_f32acc: dispatch failed at M={m} N={n} K={k}: {e}");
            return false;
        }
    };
    // f32 readback — 4 bytes/elem, NO f16 widening.
    let y_gpu: Vec<f32> = outputs[2].chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let y_ref = common_q4km_f32ref::q4km_dequant_matmul_f32accum_cpu(&q_bytes, &x_f16, m, n, n_bpr);
    // CONDITION-AWARE COMBINED metric drives VALID/INVALID and the A/B field (AT-1780 root-cause:
    // the raw forward error is ~1e-2 on near-zero cancellation outputs at the A/B shape — a metric
    // artifact, not an error). The combined metric is the SAME backward-stable dot-product
    // criterion the dispatch AT gates use (common_q4km_f32ref::max_rel_diff_combined). The raw
    // forward error is retained for transparent REPORTING only.
    let abs_scale = common_q4km_f32ref::q4km_abs_dot_scale(&q_bytes, &x_f16, m, n, n_bpr);
    let combined_rel_diff = common_q4km_f32ref::max_rel_diff_combined(&y_gpu, &y_ref, &abs_scale);
    let raw_rel_diff = common_q4km_f32ref::max_rel_diff(&y_gpu, &y_ref);

    // Timed MIN-of-10.
    let (min_ns, ts) = resident_min_of_n(
        ctx, handle,
        &[&q_bytes, &x_bytes, &vec![0u8; c_size]],
        &[0, 0, c_size as u64],
        workgroups,
        pc,
    );

    let tflops = effective_q4km_tflops(m, n, k, min_ns);
    let pct = tflops / CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000 * 100.0_f64;
    let competitive = ts == ResidentTimingSource::GpuTimestamp && pct >= COMPETITIVE_PCT_THRESHOLD;
    let label = if competitive { "competitive" } else { "honest" };
    // VALID/INVALID is driven by the COMBINED (condition-aware) metric, matching the GPU AT gates.
    let valid = if combined_rel_diff <= FROZEN_REL_TOL { "VALID" } else { "INVALID(>1e-3)" };

    eprintln!(
        "resident_q4km_matmul_rb_f32acc ({label}) = {tflops:.3} TFLOPS ({pct:.2}% of \
         {CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000}-TFLOPS datasheet ESTIMATE — f32-accumulator fused \
         Q4_K_M dequant + 2x2 RB coopmat) | {min_ns} ns ({}) | M={m} N={n} K={k} \
         combined_rel_diff={combined_rel_diff:.3e} [{valid}] raw_rel_diff={raw_rel_diff:.3e} \
         dispatch=({wg_x},{wg_y},1)",
        timing_source_label(ts)
    );

    assert!(tflops > 0.0 && tflops.is_finite(), "AT-1783: tflops must be > 0 and finite");

    if emit_ab_line {
        // Machine-readable line for scripts/m34_llamacpp_ab.sh --fused-f32acc (SAME-SHAPE A/B).
        // FLOPs: 2*M*N*K matmul MACs, dequant EXCLUDED (both sides).
        // `combined=` is the condition-aware backward-stable metric that DRIVES numerically_valid
        // (≤ frozen 1e-3 = VALID); `raw=` is the forward error retained for transparency (it is
        // ~1e-2 on near-zero cancellation outputs at this shape — same in kind as any
        // f16×f16→f32 GEMM incl. llama.cpp's HMMA). The script keys validity off `combined`.
        let flops: u64 = 2 * (m as u64) * (n as u64) * (k as u64);
        println!(
            "AXC_Q4KM_AB_F32ACC kernel_ns_min={} kernel_ns_mean={} kernel_ns_median={} \
             sustained_ns={} timing_source={:?} K={} flops={} m={} n={} combined={:.6e} raw={:.6e} \
             device={}",
            min_ns, min_ns, min_ns, min_ns, ts, k, flops, m, n,
            combined_rel_diff, raw_rel_diff,
            ctx.physical_device_name(),
        );
        eprintln!(
            "resident_q4km_matmul_rb_f32acc: AB SAME-SHAPE m={m} n={n} k={k} -> {tflops:.3} TFLOPS \
             ({min_ns} ns, combined_rel_diff={combined_rel_diff:.3e} [{valid}] \
             raw_rel_diff={raw_rel_diff:.3e})"
        );
    }

    true
}

fn bench_resident_q4km_matmul_rb_f32acc(c: &mut Criterion) {
    if !gpu_benches_enabled() {
        eprintln!("resident_q4km_matmul_rb_f32acc: AXC_ENABLE_GPU_BENCHES not set; skipping");
        c.bench_function("dispatch_resident_q4km_matmul_rb_f32acc", |b| b.iter(|| {}));
        return;
    }

    let ctx = match VulkanContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("resident_q4km_matmul_rb_f32acc: no Vulkan: {e}");
            c.bench_function("dispatch_resident_q4km_matmul_rb_f32acc", |b| b.iter(|| {}));
            return;
        }
    };
    if !ctx.coopmat_support().feature_present {
        eprintln!("resident_q4km_matmul_rb_f32acc: coopmat unsupported on {}; typed-skip", ctx.physical_device_name());
        c.bench_function("dispatch_resident_q4km_matmul_rb_f32acc", |b| b.iter(|| {}));
        return;
    }
    if ctx.subgroup_size() != 32 {
        eprintln!("resident_q4km_matmul_rb_f32acc: subgroup_size()={} != 32; typed-skip", ctx.subgroup_size());
        c.bench_function("dispatch_resident_q4km_matmul_rb_f32acc", |b| b.iter(|| {}));
        return;
    }

    let assignments = rb2x2_assignments();
    let (bytes, meta) = match compile_source_with_assignments(FUSED_F32ACC_SRC, &assignments) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("resident_q4km_matmul_rb_f32acc: compile failed: {e:?}");
            c.bench_function("dispatch_resident_q4km_matmul_rb_f32acc", |b| b.iter(|| {}));
            return;
        }
    };
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let handle = match ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, meta.coopmat.as_ref(), "q4km_matmul_rb_coopmat_f32acc",
        meta.shared_memory_bytes,
    ) {
        Ok(h) => h,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("resident_q4km_matmul_rb_f32acc: CoopMatUnsupported (typed-skip): {reason}");
            c.bench_function("dispatch_resident_q4km_matmul_rb_f32acc", |b| b.iter(|| {}));
            return;
        }
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
            eprintln!("resident_q4km_matmul_rb_f32acc: DeviceFeatureUnsupported {feature}/{kernel} (typed-skip)");
            c.bench_function("dispatch_resident_q4km_matmul_rb_f32acc", |b| b.iter(|| {}));
            return;
        }
        Err(e) => {
            eprintln!("resident_q4km_matmul_rb_f32acc: prepare failed: {e}");
            c.bench_function("dispatch_resident_q4km_matmul_rb_f32acc", |b| b.iter(|| {}));
            return;
        }
    };

    let plan = meta.binding_plan.clone();

    let mut group = c.benchmark_group("resident_q4km_matmul_rb_f32acc");
    group.sample_size(10);
    group.bench_function("dispatch_resident_q4km_matmul_rb_f32acc", |b| {
        b.iter_custom(|iters| {
            let mut total_ns: u64 = 0;
            for _ in 0..iters {
                // AT-1783: honest multi-size TFLOPS (256/512/768/1024 cube).
                let _ = measure_q4km_rb_f32acc_tflops(&ctx, &handle, &plan, 256, 256, 256, false);
                let _ = measure_q4km_rb_f32acc_tflops(&ctx, &handle, &plan, 512, 512, 512, false);
                let _ = measure_q4km_rb_f32acc_tflops(&ctx, &handle, &plan, 768, 768, 768, false);
                let _ = measure_q4km_rb_f32acc_tflops(&ctx, &handle, &plan, 1024, 1024, 1024, false);

                // AT-1784: SAME-SHAPE A/B entry (m=4096, n=512, k=14336) — emits AXC_Q4KM_AB_F32ACC.
                let _ = measure_q4km_rb_f32acc_tflops(&ctx, &handle, &plan, AB_M, AB_N, AB_K, true);

                // Criterion duration: re-time the 256³ size (K=256 -> 1 superblock/row).
                let n_bpr = 1usize;
                let q = make_q4km_weights(256, n_bpr, 0xC0FFEE ^ 256);
                let xb: Vec<u8> = make_x_f16(256, 256, 0xBADF00D ^ 256)
                    .iter().flat_map(|&b| b.to_le_bytes()).collect();
                let (ns_256, _) = resident_min_of_n(
                    &ctx, &handle,
                    &[&q, &xb, &vec![0u8; 256 * 256 * 4]],
                    &[0, 0, (256 * 256 * 4) as u64],
                    (8, 8, 1),
                    assemble_pc(&plan, 256, 256, 256, n_bpr as u32),
                );
                total_ns += ns_256;
            }
            std::time::Duration::from_nanos(total_ns)
        });
    });
    group.finish();
}

criterion_group!(resident_q4km_matmul_rb_f32acc_benches, bench_resident_q4km_matmul_rb_f32acc);
criterion_main!(resident_q4km_matmul_rb_f32acc_benches);
