//! AT-1772 + AT-1774: honest effective-TFLOPS bench for the FUSED Q4_K_M register-blocked
//! coopmat matmul (examples/q4km_matmul_rb_coopmat.axc) AND the SAME-SHAPE M3.4 A/B re-run.
//!
//! Bench ID: `dispatch_resident_q4km_matmul_rb`.
//!
//! ## AT-1772 — honest multi-size TFLOPS
//! Resident upload-once / N_WARMUP=2 / MIN-of-10 / GpuTimestamp at 256/512/768(/1024) cube
//! sizes. Reports bare tflops + % of the 125-TFLOPS datasheet ESTIMATE + the measured
//! max-rel-diff at EVERY size (including larger-K where the within-tol gate does NOT apply,
//! documenting the f16-accumulator precision frontier). NO asserted ratio (only tflops>0 &&
//! finite); "competitive" label only if pct >= 25.0. Expected BELOW M3.3c's 31.2 TFLOPS
//! (Q4_K_M dequant is ALU-heavy). 50 GB cap asserted; 2D grid pre-check.
//!
//! ## AT-1774 — SAME-SHAPE A/B line
//! The pinned A/B shape (m=4096, n=512, k=14336) is in the size set so the headline ratio
//! is SAME-SHAPE (AXIOM fused GEMM vs llama Q4_K MUL_MAT at the IDENTICAL shape = 101.00
//! TFLOPS). Emits the machine-readable `AXC_Q4KM_AB_FUSED` line (parsed by
//! scripts/m34_llamacpp_ab.sh --fused) with kernel ns, K, flops (2*M*N*K), m, n,
//! max_rel_diff, device. FLOP convention: 2*M*N*K matmul MACs, dequant EXCLUDED (both sides).
//!
//! Gated on AXC_ENABLE_GPU_BENCHES=1 + a responsive Vulkan ICD. Typed-skip on
//! CoopMatUnsupported (Lavapipe) / DeviceFeatureUnsupported / subgroup_size() != 32.

#![allow(dead_code)]

#[allow(clippy::duplicate_mod)]
#[path = "../tests/common_q4km_f16ref.rs"]
mod common_q4km_f16ref;

use criterion::{criterion_group, criterion_main, Criterion};
use axc_driver::compile_source_with_assignments;
use axc_hir::ParamBindingPlan;
use axc_runtime::{
    VulkanContext, DispatchError, KernelHandle,
    ResidentBenchConfig, ResidentTimingSource,
};
use std::collections::BTreeMap;

const FUSED_SRC: &str = include_str!("../../../examples/q4km_matmul_rb_coopmat.axc");

/// External cuBLAS f32 GEMM throughput estimate for the NVIDIA RTX PRO 6000 Blackwell.
const CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000: f64 = 125.0_f64;
/// "competitive" label threshold (25% of 125 TFLOPS).
const COMPETITIVE_PCT_THRESHOLD: f64 = 25.0_f64;

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

/// Measure the fused kernel at (m, n, k). Reports honest tflops + pct + measured
/// max-rel-diff (vs the f16-accumulator oracle). If `emit_ab_line` is set, also prints the
/// machine-readable AXC_Q4KM_AB_FUSED line for the SAME-SHAPE A/B (AT-1774).
///
/// Returns true if a measurement was taken (false on typed-skip / grid / cap failure).
fn measure_q4km_rb_tflops(
    ctx: &VulkanContext,
    handle: &KernelHandle,
    plan: &ParamBindingPlan,
    m: usize,
    n: usize,
    k: usize,
    emit_ab_line: bool,
) -> bool {
    if !m.is_multiple_of(32) || !n.is_multiple_of(32) {
        eprintln!("resident_q4km_matmul_rb: M={m}/N={n} not multiples of 32 — skip");
        return false;
    }
    if !k.is_multiple_of(256) {
        eprintln!("resident_q4km_matmul_rb: K={k} not a multiple of 256 (Q4_K_M superblock) — skip");
        return false;
    }
    let n_bpr = k / 256;

    // 2D grid pre-check: grid = (N/32, M/32, 1).
    let wg_x = (n / 32) as u64;
    let wg_y = (m / 32) as u64;
    let max_wg = ctx.max_compute_work_group_count();
    if wg_x > max_wg[0] as u64 || wg_y > max_wg[1] as u64 {
        eprintln!("resident_q4km_matmul_rb: grid ({wg_x},{wg_y}) exceeds limits at M={m} N={n} K={k} — skip");
        return false;
    }

    // 50 GB cap.
    let total_bytes: u64 = (m as u64 * n_bpr as u64 * 144)   // q (u8)
        + 2 * (k as u64) * (n as u64)                        // x (f16)
        + 2 * (m as u64) * (n as u64);                       // C (f16)
    if total_bytes >= FIFTY_GB {
        eprintln!("resident_q4km_matmul_rb: 50 GB cap exceeded at M={m} N={n} K={k} ({total_bytes} bytes) — skip");
        return false;
    }

    // Fixtures.
    let q_bytes = make_q4km_weights(m, n_bpr, 0xC0FFEE ^ (m as u64));
    let x_f16 = make_x_f16(k, n, 0xBADF00D ^ (n as u64));
    let x_bytes: Vec<u8> = x_f16.iter().flat_map(|&b| b.to_le_bytes()).collect();
    let c_size = m * n * 2;

    // Correctness pre-flight on the SAME handle being timed: dispatch once, compare vs the
    // f16-accumulator oracle, REPORT max-rel-diff (un-gated for larger K, gated reporting
    // in the dispatch tests). A wrong-but-fast kernel must not pollute the A/B.
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
            eprintln!("resident_q4km_matmul_rb: CoopMatUnsupported at dispatch (typed-skip): {reason}");
            return false;
        }
        Err(e) => {
            eprintln!("resident_q4km_matmul_rb: dispatch failed at M={m} N={n} K={k}: {e}");
            return false;
        }
    };
    let y_gpu: Vec<f32> = outputs[2].chunks_exact(2)
        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();
    let y_ref = common_q4km_f16ref::q4km_dequant_matmul_f16accum_cpu(&q_bytes, &x_f16, m, n, n_bpr, 16);
    let max_rel_diff = common_q4km_f16ref::max_rel_diff(&y_gpu, &y_ref);

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

    eprintln!(
        "resident_q4km_matmul_rb ({label}) = {tflops:.3} TFLOPS ({pct:.2}% of \
         {CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000}-TFLOPS datasheet ESTIMATE — fused Q4_K_M dequant \
         + 2x2 RB coopmat, NOT a same-machine A/B) | {min_ns} ns ({}) | M={m} N={n} K={k} \
         max_rel_diff={max_rel_diff:.3e} dispatch=({wg_x},{wg_y},1)",
        timing_source_label(ts)
    );

    assert!(tflops > 0.0 && tflops.is_finite(), "AT-1772: tflops must be > 0 and finite");

    if emit_ab_line {
        // Machine-readable line for scripts/m34_llamacpp_ab.sh --fused (SAME-SHAPE A/B).
        // FLOPs: 2*M*N*K matmul MACs, dequant EXCLUDED (both sides).
        let flops: u64 = 2 * (m as u64) * (n as u64) * (k as u64);
        println!(
            "AXC_Q4KM_AB_FUSED kernel_ns_min={} kernel_ns_mean={} kernel_ns_median={} \
             sustained_ns={} timing_source={:?} K={} flops={} m={} n={} max_rel_diff={:.6e} \
             device={}",
            min_ns, min_ns, min_ns, min_ns, ts, k, flops, m, n, max_rel_diff,
            ctx.physical_device_name(),
        );
        eprintln!(
            "resident_q4km_matmul_rb: AB SAME-SHAPE m={m} n={n} k={k} -> {tflops:.3} TFLOPS \
             ({min_ns} ns, max_rel_diff={max_rel_diff:.3e})"
        );
    }

    true
}

fn bench_resident_q4km_matmul_rb(c: &mut Criterion) {
    if !gpu_benches_enabled() {
        eprintln!("resident_q4km_matmul_rb: AXC_ENABLE_GPU_BENCHES not set; skipping");
        c.bench_function("dispatch_resident_q4km_matmul_rb", |b| b.iter(|| {}));
        return;
    }

    let ctx = match VulkanContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("resident_q4km_matmul_rb: no Vulkan: {e}");
            c.bench_function("dispatch_resident_q4km_matmul_rb", |b| b.iter(|| {}));
            return;
        }
    };
    if !ctx.coopmat_support().feature_present {
        eprintln!("resident_q4km_matmul_rb: coopmat unsupported on {}; typed-skip", ctx.physical_device_name());
        c.bench_function("dispatch_resident_q4km_matmul_rb", |b| b.iter(|| {}));
        return;
    }
    if ctx.subgroup_size() != 32 {
        eprintln!("resident_q4km_matmul_rb: subgroup_size()={} != 32; typed-skip", ctx.subgroup_size());
        c.bench_function("dispatch_resident_q4km_matmul_rb", |b| b.iter(|| {}));
        return;
    }

    let assignments = rb2x2_assignments();
    let (bytes, meta) = match compile_source_with_assignments(FUSED_SRC, &assignments) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("resident_q4km_matmul_rb: compile failed: {e:?}");
            c.bench_function("dispatch_resident_q4km_matmul_rb", |b| b.iter(|| {}));
            return;
        }
    };
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let handle = match ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, meta.coopmat.as_ref(), "q4km_matmul_rb_coopmat",
        meta.shared_memory_bytes,
    ) {
        Ok(h) => h,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("resident_q4km_matmul_rb: CoopMatUnsupported (typed-skip): {reason}");
            c.bench_function("dispatch_resident_q4km_matmul_rb", |b| b.iter(|| {}));
            return;
        }
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
            eprintln!("resident_q4km_matmul_rb: DeviceFeatureUnsupported {feature}/{kernel} (typed-skip)");
            c.bench_function("dispatch_resident_q4km_matmul_rb", |b| b.iter(|| {}));
            return;
        }
        Err(e) => {
            eprintln!("resident_q4km_matmul_rb: prepare failed: {e}");
            c.bench_function("dispatch_resident_q4km_matmul_rb", |b| b.iter(|| {}));
            return;
        }
    };

    let plan = meta.binding_plan.clone();

    let mut group = c.benchmark_group("resident_q4km_matmul_rb");
    group.sample_size(10);
    group.bench_function("dispatch_resident_q4km_matmul_rb", |b| {
        b.iter_custom(|iters| {
            let mut total_ns: u64 = 0;
            for _ in 0..iters {
                // AT-1772: honest multi-size TFLOPS (256/512/768; 1024 if <50GB).
                // K = cube size (multiple of 256): 256->1 sb, 512->2, 768->3, 1024->4.
                let _ = measure_q4km_rb_tflops(&ctx, &handle, &plan, 256, 256, 256, false);
                let _ = measure_q4km_rb_tflops(&ctx, &handle, &plan, 512, 512, 512, false);
                let _ = measure_q4km_rb_tflops(&ctx, &handle, &plan, 768, 768, 768, false);
                let _ = measure_q4km_rb_tflops(&ctx, &handle, &plan, 1024, 1024, 1024, false);

                // AT-1774: SAME-SHAPE A/B entry (m=4096, n=512, k=14336) — emits AXC_Q4KM_AB_FUSED.
                let _ = measure_q4km_rb_tflops(&ctx, &handle, &plan, AB_M, AB_N, AB_K, true);

                // Criterion duration: re-time the 256³ size (K=256 -> 1 superblock/row).
                let n_bpr = 1usize;
                let q = make_q4km_weights(256, n_bpr, 0xC0FFEE ^ 256);
                let xb: Vec<u8> = make_x_f16(256, 256, 0xBADF00D ^ 256)
                    .iter().flat_map(|&b| b.to_le_bytes()).collect();
                let (ns_256, _) = resident_min_of_n(
                    &ctx, &handle,
                    &[&q, &xb, &vec![0u8; 256 * 256 * 2]],
                    &[0, 0, (256 * 256 * 2) as u64],
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

criterion_group!(resident_q4km_matmul_rb_benches, bench_resident_q4km_matmul_rb);
criterion_main!(resident_q4km_matmul_rb_benches);
