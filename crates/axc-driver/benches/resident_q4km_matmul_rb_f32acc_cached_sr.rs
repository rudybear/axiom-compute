//! AT-2602 + AT-2604: GPU-resident effective-TFLOPS bench for the M3.11a dequant-index
//! STRENGTH-REDUCED scale-cached f32-accumulator fused Q4_K_M register-blocked coopmat matmul
//! (examples/q4km_matmul_rb_coopmat_f32acc_cached_sr.axc) vs the M3.6 leader (cached.axc).
//!
//! Bench ID: `dispatch_resident_q4km_matmul_rb_f32acc_cached_sr`.
//!
//! SIBLING of resident_q4km_matmul_rb_f32acc_cached.rs — clones it, points at the M3.11a SR kernel,
//! and ALSO times the M3.6 leader at the SAME shapes so the orchestrator reads the honest A/B ratio
//! directly. Emits the machine-readable `AXC_Q4KM_AB_F32ACC_CACHED_SR` line (parsed by
//! scripts/m34_llamacpp_ab.sh --sr). Same honesty contract: the COMBINED condition-aware metric
//! (`|gpu-ref|/max(|ref|,Σ|wₖxₖ|) <= frozen 1e-3`) DRIVES VALID/INVALID; the RAW forward error is
//! reported separately. MIN-of-10 / GpuTimestamp resident harness.
//!
//! ## AT-2602 — PRIMARY PERF GATE
//! cached_sr at the A/B shape (4096x512x14336) >= 1.15x the M3.6 leader (>= 1.15 * 42.86 = >= 49.3
//! TFLOPS) AND at 768^3. The orchestrator evaluates the gate from the printed SR-vs-M3.6 ratio.
//!
//! ## AT-2604 — ARMED HONEST-NEGATIVE (the LIKELY base case)
//! If the emitted ALU-count drops (AT-2600) but TFLOPS is FLAT (< 1.15x), that is the latency-hidden
//! finding: the dequant integer ALU issues on the int/SFU pipes CONCURRENTLY with the HMMA tensor-
//! core mul_add => NOT on the critical path => the 1.77x front-end tax is NOT the integer index-
//! decode. Flat is the EXPECTED, still-profound result; a >= 1.15x win is the surprising upside.
//! The gate is NEVER loosened; M3.6 stays leader.
//!
//! Gated on AXC_ENABLE_GPU_BENCHES=1 + a responsive Vulkan ICD. Typed-skip on CoopMatUnsupported
//! (Lavapipe) / DeviceFeatureUnsupported / subgroup_size() != 32.

#![allow(dead_code)]

// The oracle is the pub lib module axc_driver::q4km_oracle (single source of truth).
use axc_driver::q4km_oracle as common_q4km_f32ref;

use criterion::{criterion_group, criterion_main, Criterion};
use axc_driver::compile_source_with_assignments;
use axc_hir::ParamBindingPlan;
use axc_runtime::{
    VulkanContext, DispatchError, KernelHandle,
    ResidentBenchConfig, ResidentTimingSource,
};
use std::collections::BTreeMap;

const SR_SRC: &str =
    include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached_sr.axc");
const CACHED_SRC: &str =
    include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached.axc");

/// External cuBLAS f32 GEMM throughput estimate for the NVIDIA RTX PRO 6000 Blackwell.
const CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000: f64 = 125.0_f64;
const COMPETITIVE_PCT_THRESHOLD: f64 = 25.0_f64;
/// Frozen relative tolerance (NOT loosened).
const FROZEN_REL_TOL: f64 = 1e-3;
/// M3.6 leader A/B baseline TFLOPS (the AT-2602 1.15x PRIMARY GATE reference).
const M36_LEADER_AB_TFLOPS: f64 = 42.86_f64;
/// The AT-2602 primary-gate multiplier (>= 1.15x => real signal => green-light M3.11b).
const PERF_GATE_RATIO: f64 = 1.15_f64;

const N_WARMUP: usize = 2;
const N_MEASURED: usize = 10;

const FIFTY_GB: u64 = 50 * 1024 * 1024 * 1024;

/// Pinned A/B same-shape (matches llama.cpp Q4_K MUL_MAT m=4096,n=512,k=14336).
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

/// ASYMMETRIC Q4_K_M weight fixture (SAME seed scheme as the M3.6 leader bench).
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

/// One prepared kernel handle + its binding plan.
struct Prepared {
    handle: KernelHandle,
    plan: ParamBindingPlan,
}

/// Compile + prepare a kernel; returns None on typed-skip.
fn prepare(ctx: &VulkanContext, src: &str, kernel_name: &str) -> Option<Prepared> {
    let assignments = rb2x2_assignments();
    let (bytes, meta) = match compile_source_with_assignments(src, &assignments) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("resident_sr: {kernel_name} compile failed: {e:?}");
            return None;
        }
    };
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let handle = match ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, meta.coopmat.as_ref(), kernel_name, meta.shared_memory_bytes,
    ) {
        Ok(h) => h,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("resident_sr: {kernel_name} CoopMatUnsupported (typed-skip): {reason}");
            return None;
        }
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
            eprintln!("resident_sr: {kernel_name} DeviceFeatureUnsupported {feature}/{kernel} (typed-skip)");
            return None;
        }
        Err(e) => {
            eprintln!("resident_sr: {kernel_name} prepare failed: {e}");
            return None;
        }
    };
    Some(Prepared { handle, plan: meta.binding_plan })
}

/// Measure one kernel at (m,n,k). Returns (tflops, combined_rel_diff, min_ns, timing_source) or None.
fn measure(
    ctx: &VulkanContext,
    prep: &Prepared,
    label: &str,
    m: usize,
    n: usize,
    k: usize,
) -> Option<(f64, f64, u64, ResidentTimingSource)> {
    if !m.is_multiple_of(32) || !n.is_multiple_of(32) || !k.is_multiple_of(256) {
        eprintln!("resident_sr {label}: M={m}/N={n}/K={k} shape invalid — skip");
        return None;
    }
    let n_bpr = k / 256;
    let wg_x = (n / 32) as u64;
    let wg_y = (m / 32) as u64;
    let max_wg = ctx.max_compute_work_group_count();
    if wg_x > max_wg[0] as u64 || wg_y > max_wg[1] as u64 {
        eprintln!("resident_sr {label}: grid ({wg_x},{wg_y}) exceeds limits at M={m} N={n} K={k} — skip");
        return None;
    }
    let total_bytes: u64 = (m as u64 * n_bpr as u64 * 144)
        + 2 * (k as u64) * (n as u64)
        + 4 * (m as u64) * (n as u64);
    if total_bytes >= FIFTY_GB {
        eprintln!("resident_sr {label}: 50 GB cap exceeded at M={m} N={n} K={k} — skip");
        return None;
    }

    let q_bytes = make_q4km_weights(m, n_bpr, 0xC0FFEE ^ (m as u64));
    let x_f16 = make_x_f16(k, n, 0xBADF00D ^ (n as u64));
    let x_bytes: Vec<u8> = x_f16.iter().flat_map(|&b| b.to_le_bytes()).collect();
    let c_size = m * n * 4;

    let pc = assemble_pc(&prep.plan, m as u32, n as u32, k as u32, n_bpr as u32);
    let workgroups = (wg_x as u32, wg_y as u32, 1);

    let outputs = match ctx.dispatch_handle(
        &prep.handle, workgroups,
        &[&q_bytes, &x_bytes, &vec![0u8; c_size]],
        &[0, 0, c_size],
        &pc,
    ) {
        Ok(v) => v,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("resident_sr {label}: CoopMatUnsupported at dispatch (typed-skip): {reason}");
            return None;
        }
        Err(e) => {
            eprintln!("resident_sr {label}: dispatch failed at M={m} N={n} K={k}: {e}");
            return None;
        }
    };
    let y_gpu: Vec<f32> = outputs[2].chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let y_ref = common_q4km_f32ref::q4km_dequant_matmul_f32accum_cpu(&q_bytes, &x_f16, m, n, n_bpr);
    let abs_scale = common_q4km_f32ref::q4km_abs_dot_scale(&q_bytes, &x_f16, m, n, n_bpr);
    let combined = common_q4km_f32ref::max_rel_diff_combined(&y_gpu, &y_ref, &abs_scale);

    let (min_ns, ts) = resident_min_of_n(
        ctx, &prep.handle,
        &[&q_bytes, &x_bytes, &vec![0u8; c_size]],
        &[0, 0, c_size as u64],
        workgroups,
        pc,
    );
    let tflops = effective_q4km_tflops(m, n, k, min_ns);
    let valid = if combined <= FROZEN_REL_TOL { "VALID" } else { "INVALID(>1e-3)" };
    eprintln!(
        "resident_sr {label} = {tflops:.3} TFLOPS | {min_ns} ns ({}) | M={m} N={n} K={k} \
         combined={combined:.3e} [{valid}]",
        timing_source_label(ts)
    );
    Some((tflops, combined, min_ns, ts))
}

/// Run the SR-vs-M3.6 A/B + 768^3 + per-size comparison and print the honest ratio + the AT-2602/2604
/// verdict. `emit_ab_line` prints the machine-readable AXC_Q4KM_AB_F32ACC_CACHED_SR line.
fn run_sr_vs_cached(ctx: &VulkanContext, sr: &Prepared, cached: &Prepared) {
    // Per-size honest table (256/512/768/1024 cube).
    for &sz in &[256usize, 512, 768, 1024] {
        let s = measure(ctx, sr, "SR", sz, sz, sz);
        let c = measure(ctx, cached, "M3.6", sz, sz, sz);
        if let (Some((sr_t, _, _, sr_ts)), Some((c_t, _, _, _))) = (s, c) {
            let ratio = if c_t > 0.0 { sr_t / c_t } else { 0.0 };
            let gate = if sr_ts == ResidentTimingSource::GpuTimestamp && ratio >= PERF_GATE_RATIO {
                "GATE-MET(>=1.15x)"
            } else {
                "HONEST-NEGATIVE(<1.15x or non-GpuTimestamp)"
            };
            eprintln!(
                "resident_sr: {sz}^3 SR={sr_t:.3} vs M3.6={c_t:.3} -> {ratio:.3}x [{gate}]"
            );
        }
    }

    // AT-2602 / AT-2604: the A/B same-shape gate.
    let s = measure(ctx, sr, "SR-AB", AB_M, AB_N, AB_K);
    let c = measure(ctx, cached, "M3.6-AB", AB_M, AB_N, AB_K);
    if let (Some((sr_t, sr_combined, sr_ns, sr_ts)), Some((c_t, _, _, _))) = (s, c) {
        let ratio = if c_t > 0.0 { sr_t / c_t } else { 0.0 };
        let ratio_vs_const = sr_t / M36_LEADER_AB_TFLOPS;
        let gate = if sr_ts == ResidentTimingSource::GpuTimestamp && ratio >= PERF_GATE_RATIO {
            "AT-2602 GATE-MET(>=1.15x) — green-light M3.11b (the surprising upside)"
        } else {
            "AT-2604 HONEST-NEGATIVE(<1.15x) — latency-hidden under HMMA (the LIKELY base case); \
             the 1.77x tax is NOT the integer index-decode; M3.11b NO-GO; M3.6 stays leader"
        };
        eprintln!(
            "resident_sr: AT-2602 A/B SR={sr_t:.3} vs M3.6 leader={c_t:.3} (const {M36_LEADER_AB_TFLOPS}) \
             -> {ratio:.3}x (vs-const {ratio_vs_const:.3}x) [{gate}]"
        );
        // Machine-readable line for scripts/m34_llamacpp_ab.sh --sr.
        let flops: u64 = 2 * (AB_M as u64) * (AB_N as u64) * (AB_K as u64);
        let valid = if sr_combined <= FROZEN_REL_TOL { "VALID" } else { "INVALID(>1e-3)" };
        println!(
            "AXC_Q4KM_AB_F32ACC_CACHED_SR kernel_ns_min={} timing_source={:?} K={} flops={} m={} n={} \
             sr_tflops={:.4} cached_tflops={:.4} ratio={:.4} combined={:.6e} [{}] device={}",
            sr_ns, sr_ts, AB_K, flops, AB_M, AB_N, sr_t, c_t, ratio, sr_combined, valid,
            ctx.physical_device_name(),
        );
    }
}

fn bench_resident_q4km_matmul_rb_f32acc_cached_sr(c: &mut Criterion) {
    let bench_id = "dispatch_resident_q4km_matmul_rb_f32acc_cached_sr";
    if !gpu_benches_enabled() {
        eprintln!("resident_sr: AXC_ENABLE_GPU_BENCHES not set; skipping");
        c.bench_function(bench_id, |b| b.iter(|| {}));
        return;
    }
    let ctx = match VulkanContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("resident_sr: no Vulkan: {e}");
            c.bench_function(bench_id, |b| b.iter(|| {}));
            return;
        }
    };
    if !ctx.coopmat_support().feature_present {
        eprintln!("resident_sr: coopmat unsupported on {}; typed-skip", ctx.physical_device_name());
        c.bench_function(bench_id, |b| b.iter(|| {}));
        return;
    }
    if ctx.subgroup_size() != 32 {
        eprintln!("resident_sr: subgroup_size()={} != 32; typed-skip", ctx.subgroup_size());
        c.bench_function(bench_id, |b| b.iter(|| {}));
        return;
    }

    let Some(sr) = prepare(&ctx, SR_SRC, "q4km_matmul_rb_coopmat_f32acc_cached_sr") else {
        c.bench_function(bench_id, |b| b.iter(|| {}));
        return;
    };
    let Some(cached) = prepare(&ctx, CACHED_SRC, "q4km_matmul_rb_coopmat_f32acc_cached") else {
        c.bench_function(bench_id, |b| b.iter(|| {}));
        return;
    };

    let mut group = c.benchmark_group("resident_q4km_matmul_rb_f32acc_cached_sr");
    group.sample_size(10);
    group.bench_function(bench_id, |b| {
        b.iter_custom(|iters| {
            let mut total_ns: u64 = 0;
            for _ in 0..iters {
                run_sr_vs_cached(&ctx, &sr, &cached);
                // Criterion duration: re-time the 256³ size on the SR kernel.
                let q = make_q4km_weights(256, 1, 0xC0FFEE ^ 256);
                let xb: Vec<u8> = make_x_f16(256, 256, 0xBADF00D ^ 256)
                    .iter().flat_map(|&b| b.to_le_bytes()).collect();
                let (ns_256, _) = resident_min_of_n(
                    &ctx, &sr.handle,
                    &[&q, &xb, &vec![0u8; 256 * 256 * 4]],
                    &[0, 0, (256 * 256 * 4) as u64],
                    (8, 8, 1),
                    assemble_pc(&sr.plan, 256, 256, 256, 1),
                );
                total_ns += ns_256;
            }
            std::time::Duration::from_nanos(total_ns)
        });
    });
    group.finish();
}

criterion_group!(resident_q4km_matmul_rb_f32acc_cached_sr_benches, bench_resident_q4km_matmul_rb_f32acc_cached_sr);
criterion_main!(resident_q4km_matmul_rb_f32acc_cached_sr_benches);
