//! AT-2004 + AT-2005: honest effective-TFLOPS bench for BOTH M3.8 LARGER-REGISTER-TILE variants
//! (4x2 = 8 accumulators / 4x4 = 16 accumulators) of the M3.6 scale-cached f32-accumulator fused
//! Q4_K_M coopmat matmul.
//!
//! Bench ID: `dispatch_resident_q4km_matmul_rb_f32acc_cached_bigrb`.
//!
//! CLONES resident_q4km_matmul_rb_f32acc_cached.rs (the M3.6 bench): same MIN-of-10 / GpuTimestamp
//! resident harness, 50 GB cap, COMBINED condition-aware metric driving VALID/INVALID, the
//! f32-accumulator oracle. DIFFERENCES:
//!   (a) compiles BOTH the 4x2 and 4x4 kernels with their pinned @strategy assignments;
//!   (b) per-variant grid: 4x2 = (N/32, M/64, 1); 4x4 = (N/64, M/64, 1);
//!   (c) NR-2 PER-VARIANT grid pre-check: 4x2 needs M%64==0 && N%32==0; 4x4 needs M%64==0 &&
//!       N%64==0. SKIP-WITH-LOG (NOT OOB) any non-conforming size (do NOT inherit the cloned
//!       %32-only guard);
//!   (d) emits AXC_Q4KM_AB_F32ACC_BIGRB_4X2 and _4X4 at the A/B shape;
//!   (e) prints the WINNER among {M3.6=42.86, 4x2, 4x4} and the >=1.15x / >=49.3 TFLOPS gate.
//!
//! HONEST-NEGATIVE provision: more accumulators = more registers = a possible occupancy regression
//! (M3.7/M3.3d precedent) that nets SLOWER. The bench measures BOTH and reports per-variant TFLOPS
//! plus shared_memory_bytes; an all-negative outcome is reported, NOT papered over. The frozen
//! 1e-3 NEVER loosens. M3.6 (42.86) remains the leader unless a variant beats the gate.
//!
//! Gated on AXC_ENABLE_GPU_BENCHES=1. Typed-skip on CoopMatUnsupported / subgroup_size() != 32.

#![allow(dead_code)]

// M4.1p2: the oracle is now the pub lib module axc_driver::q4km_oracle (single source of truth).
use axc_driver::q4km_oracle as common_q4km_f32ref;

use criterion::{criterion_group, criterion_main, Criterion};
use axc_driver::compile_source_with_assignments;
use axc_hir::ParamBindingPlan;
use axc_runtime::{
    VulkanContext, DispatchError, KernelHandle,
    ResidentBenchConfig, ResidentTimingSource,
};
use std::collections::BTreeMap;

const SRC_4X2: &str =
    include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached_4x2.axc");
const SRC_4X4: &str =
    include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached_4x4.axc");

const CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000: f64 = 125.0_f64;
const COMPETITIVE_PCT_THRESHOLD: f64 = 25.0_f64;
const FROZEN_REL_TOL: f64 = 1e-3;

/// M3.6 cached A/B-shape leader TFLOPS (the >=1.15x exit-gate reference for AT-2004).
const M36_CACHED_AB_TFLOPS: f64 = 42.86_f64;
/// >=1.15x exit gate threshold.
const M38_GATE_TFLOPS: f64 = 49.3_f64;
/// llama.cpp Q4_K MUL_MAT A/B-shape throughput (reference, NOT a match promise).
const LLAMA_AB_TFLOPS: f64 = 102.49_f64;

const N_WARMUP: usize = 2;
const N_MEASURED: usize = 10;
const FIFTY_GB: u64 = 50 * 1024 * 1024 * 1024;

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

/// One big-RB variant: source, name, @strategy, AB-line tag, output-tile dims (rows, cols).
struct Variant {
    label: &'static str,
    src: &'static str,
    kernel_name: &'static str,
    ab_tag: &'static str,
    assignments: BTreeMap<String, i64>,
    tile_rows: usize,
    tile_cols: usize,
}

fn variant_4x2() -> Variant {
    let mut a = BTreeMap::new();
    a.insert("rb_m".to_owned(), 4_i64);
    a.insert("rb_n".to_owned(), 2_i64);
    a.insert("tile_k".to_owned(), 16_i64);
    a.insert("a_block_size".to_owned(), 1024_i64);
    a.insert("b_block_size".to_owned(), 512_i64);
    Variant {
        label: "4x2", src: SRC_4X2,
        kernel_name: "q4km_matmul_rb_coopmat_f32acc_cached_4x2",
        ab_tag: "AXC_Q4KM_AB_F32ACC_BIGRB_4X2",
        assignments: a, tile_rows: 64, tile_cols: 32,
    }
}

fn variant_4x4() -> Variant {
    let mut a = BTreeMap::new();
    a.insert("rb_m".to_owned(), 4_i64);
    a.insert("rb_n".to_owned(), 4_i64);
    a.insert("tile_k".to_owned(), 16_i64);
    a.insert("a_block_size".to_owned(), 1024_i64);
    a.insert("b_block_size".to_owned(), 1024_i64);
    Variant {
        label: "4x4", src: SRC_4X4,
        kernel_name: "q4km_matmul_rb_coopmat_f32acc_cached_4x4",
        ab_tag: "AXC_Q4KM_AB_F32ACC_BIGRB_4X4",
        assignments: a, tile_rows: 64, tile_cols: 64,
    }
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

/// Measure one big-RB variant at (m, n, k). Returns Some(tflops) on success, None on a typed-skip /
/// per-variant grid / cap / divisibility skip. If `emit_ab_line` is set, prints the variant's
/// machine-readable A/B line (AT-2005).
#[allow(clippy::too_many_arguments)]
fn measure_bigrb_tflops(
    ctx: &VulkanContext,
    handle: &KernelHandle,
    plan: &ParamBindingPlan,
    v: &Variant,
    m: usize,
    n: usize,
    k: usize,
    emit_ab_line: bool,
) -> Option<f64> {
    // NR-2 PER-VARIANT divisibility guard (NOT the inherited %32-only guard).
    if !m.is_multiple_of(v.tile_rows) || !n.is_multiple_of(v.tile_cols) {
        eprintln!(
            "bigrb [{}]: SKIP-WITH-LOG — M={m} N={n} not a multiple of the {}x{} output tile \
             (need M%{}==0 && N%{}==0); not dispatched (avoids OOB unconditional loads)",
            v.label, v.tile_rows, v.tile_cols, v.tile_rows, v.tile_cols
        );
        return None;
    }
    if !k.is_multiple_of(256) {
        eprintln!("bigrb [{}]: K={k} not a multiple of 256 (Q4_K_M superblock) — skip", v.label);
        return None;
    }
    let n_bpr = k / 256;

    // Per-variant grid: (N / tile_cols, M / tile_rows, 1).
    let wg_x = (n / v.tile_cols) as u64;
    let wg_y = (m / v.tile_rows) as u64;
    let max_wg = ctx.max_compute_work_group_count();
    if wg_x > max_wg[0] as u64 || wg_y > max_wg[1] as u64 {
        eprintln!("bigrb [{}]: grid ({wg_x},{wg_y}) exceeds limits at M={m} N={n} K={k} — skip", v.label);
        return None;
    }

    let total_bytes: u64 = (m as u64 * n_bpr as u64 * 144)
        + 2 * (k as u64) * (n as u64)
        + 4 * (m as u64) * (n as u64);
    if total_bytes >= FIFTY_GB {
        eprintln!("bigrb [{}]: 50 GB cap exceeded at M={m} N={n} K={k} ({total_bytes} bytes) — skip", v.label);
        return None;
    }

    let q_bytes = make_q4km_weights(m, n_bpr, 0xC0FFEE ^ (m as u64));
    let x_f16 = make_x_f16(k, n, 0xBADF00D ^ (n as u64));
    let x_bytes: Vec<u8> = x_f16.iter().flat_map(|&b| b.to_le_bytes()).collect();
    let c_size = m * n * 4;

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
            eprintln!("bigrb [{}]: CoopMatUnsupported at dispatch (typed-skip): {reason}", v.label);
            return None;
        }
        Err(e) => {
            eprintln!("bigrb [{}]: dispatch failed at M={m} N={n} K={k}: {e}", v.label);
            return None;
        }
    };
    let y_gpu: Vec<f32> = outputs[2].chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let y_ref = common_q4km_f32ref::q4km_dequant_matmul_f32accum_cpu(&q_bytes, &x_f16, m, n, n_bpr);
    let abs_scale = common_q4km_f32ref::q4km_abs_dot_scale(&q_bytes, &x_f16, m, n, n_bpr);
    let combined_rel_diff = common_q4km_f32ref::max_rel_diff_combined(&y_gpu, &y_ref, &abs_scale);
    let raw_rel_diff = common_q4km_f32ref::max_rel_diff(&y_gpu, &y_ref);

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
    let valid = if combined_rel_diff <= FROZEN_REL_TOL { "VALID" } else { "INVALID(>1e-3)" };

    eprintln!(
        "resident_q4km_matmul_rb_f32acc_cached_bigrb [{}] ({label}) = {tflops:.3} TFLOPS \
         ({pct:.2}% of {CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000}-TFLOPS ESTIMATE) | {min_ns} ns ({}) | \
         M={m} N={n} K={k} combined_rel_diff={combined_rel_diff:.3e} [{valid}] \
         raw_rel_diff={raw_rel_diff:.3e} dispatch=({wg_x},{wg_y},1)",
        v.label, timing_source_label(ts)
    );

    assert!(tflops > 0.0 && tflops.is_finite(), "AT-2004 [{}]: tflops must be > 0 and finite", v.label);

    if emit_ab_line {
        let flops: u64 = 2 * (m as u64) * (n as u64) * (k as u64);
        println!(
            "{} kernel_ns_min={} kernel_ns_mean={} kernel_ns_median={} sustained_ns={} \
             timing_source={:?} K={} flops={} m={} n={} combined={:.6e} raw={:.6e} device={}",
            v.ab_tag, min_ns, min_ns, min_ns, min_ns, ts, k, flops, m, n,
            combined_rel_diff, raw_rel_diff, ctx.physical_device_name(),
        );
        let ratio_m36 = tflops / M36_CACHED_AB_TFLOPS;
        let gate = if ts == ResidentTimingSource::GpuTimestamp && tflops >= M38_GATE_TFLOPS {
            "GATE-MET(>=1.15x / >=49.3 TFLOPS)"
        } else {
            "HONEST-NEGATIVE(<49.3 TFLOPS or non-GpuTimestamp)"
        };
        eprintln!(
            "resident_q4km_matmul_rb_f32acc_cached_bigrb [{}]: AB SAME-SHAPE m={m} n={n} k={k} -> \
             {tflops:.3} TFLOPS [{valid}] | vs M3.6 {M36_CACHED_AB_TFLOPS} -> {ratio_m36:.3}x \
             [{gate}] | vs llama {LLAMA_AB_TFLOPS}",
            v.label
        );
    }

    Some(tflops)
}

fn prepare_variant_handle(
    ctx: &VulkanContext,
    v: &Variant,
) -> Option<(KernelHandle, ParamBindingPlan, u32)> {
    let (bytes, meta) = match compile_source_with_assignments(v.src, &v.assignments) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("bigrb [{}]: compile failed: {e:?}", v.label);
            return None;
        }
    };
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let handle = match ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, meta.coopmat.as_ref(), v.kernel_name, meta.shared_memory_bytes,
    ) {
        Ok(h) => h,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("bigrb [{}]: CoopMatUnsupported (typed-skip): {reason}", v.label);
            return None;
        }
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
            eprintln!("bigrb [{}]: DeviceFeatureUnsupported {feature}/{kernel} (typed-skip)", v.label);
            return None;
        }
        Err(e) => {
            eprintln!("bigrb [{}]: prepare failed: {e}", v.label);
            return None;
        }
    };
    Some((handle, meta.binding_plan.clone(), meta.shared_memory_bytes))
}

fn bench_resident_q4km_matmul_rb_f32acc_cached_bigrb(c: &mut Criterion) {
    const BENCH_ID: &str = "dispatch_resident_q4km_matmul_rb_f32acc_cached_bigrb";
    if !gpu_benches_enabled() {
        eprintln!("{BENCH_ID}: AXC_ENABLE_GPU_BENCHES not set; skipping");
        c.bench_function(BENCH_ID, |b| b.iter(|| {}));
        return;
    }
    let ctx = match VulkanContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("{BENCH_ID}: no Vulkan: {e}");
            c.bench_function(BENCH_ID, |b| b.iter(|| {}));
            return;
        }
    };
    if !ctx.coopmat_support().feature_present {
        eprintln!("{BENCH_ID}: coopmat unsupported on {}; typed-skip", ctx.physical_device_name());
        c.bench_function(BENCH_ID, |b| b.iter(|| {}));
        return;
    }
    if ctx.subgroup_size() != 32 {
        eprintln!("{BENCH_ID}: subgroup_size()={} != 32; typed-skip", ctx.subgroup_size());
        c.bench_function(BENCH_ID, |b| b.iter(|| {}));
        return;
    }

    let v4x2 = variant_4x2();
    let v4x4 = variant_4x4();

    let Some((h4x2, plan4x2, shared4x2)) = prepare_variant_handle(&ctx, &v4x2) else {
        c.bench_function(BENCH_ID, |b| b.iter(|| {}));
        return;
    };
    let Some((h4x4, plan4x4, shared4x4)) = prepare_variant_handle(&ctx, &v4x4) else {
        c.bench_function(BENCH_ID, |b| b.iter(|| {}));
        return;
    };
    eprintln!(
        "{BENCH_ID}: 4x2 shared_memory_bytes={shared4x2} (expect 7168); \
         4x4 shared_memory_bytes={shared4x4} (expect 8192)"
    );

    let mut group = c.benchmark_group("resident_q4km_matmul_rb_f32acc_cached_bigrb");
    group.sample_size(10);
    group.bench_function(BENCH_ID, |b| {
        b.iter_custom(|iters| {
            let mut total_ns: u64 = 0;
            for _ in 0..iters {
                // AT-2004: honest multi-size TFLOPS at 256/512/768/1024 cube for BOTH variants.
                for &sz in &[256usize, 512, 768, 1024] {
                    let _ = measure_bigrb_tflops(&ctx, &h4x2, &plan4x2, &v4x2, sz, sz, sz, false);
                    let _ = measure_bigrb_tflops(&ctx, &h4x4, &plan4x4, &v4x4, sz, sz, sz, false);
                }

                // AT-2005: SAME-SHAPE A/B entry — emits the two AXC_Q4KM_AB_F32ACC_BIGRB_* lines.
                let ab_4x2 = measure_bigrb_tflops(&ctx, &h4x2, &plan4x2, &v4x2, AB_M, AB_N, AB_K, true);
                let ab_4x4 = measure_bigrb_tflops(&ctx, &h4x4, &plan4x4, &v4x4, AB_M, AB_N, AB_K, true);

                // Winner among {M3.6=42.86, 4x2, 4x4}.
                let mut best_label = "M3.6 (2x2 cached)";
                let mut best_tflops = M36_CACHED_AB_TFLOPS;
                if let Some(t) = ab_4x2 {
                    if t > best_tflops { best_tflops = t; best_label = "4x2"; }
                }
                if let Some(t) = ab_4x4 {
                    if t > best_tflops { best_tflops = t; best_label = "4x4"; }
                }
                let gate = if best_tflops >= M38_GATE_TFLOPS {
                    "GATE-MET(>=49.3)"
                } else {
                    "HONEST-NEGATIVE(<49.3 — register-pressure/occupancy; M3.6 remains leader)"
                };
                eprintln!(
                    "{BENCH_ID}: WINNER at A/B = {best_label} @ {best_tflops:.3} TFLOPS \
                     (4x2={:?}, 4x4={:?}, M3.6={M36_CACHED_AB_TFLOPS}) [{gate}] | \
                     shared: 4x2={shared4x2}B 4x4={shared4x4}B",
                    ab_4x2, ab_4x4
                );

                // Criterion duration: re-time the 4x2 256³ size.
                let q = make_q4km_weights(256, 1, 0xC0FFEE ^ 256);
                let xb: Vec<u8> = make_x_f16(256, 256, 0xBADF00D ^ 256)
                    .iter().flat_map(|&b| b.to_le_bytes()).collect();
                let (ns_256, _) = resident_min_of_n(
                    &ctx, &h4x2,
                    &[&q, &xb, &vec![0u8; 256 * 256 * 4]],
                    &[0, 0, (256 * 256 * 4) as u64],
                    ((256 / v4x2.tile_cols) as u32, (256 / v4x2.tile_rows) as u32, 1),
                    assemble_pc(&plan4x2, 256, 256, 256, 1),
                );
                total_ns += ns_256;
            }
            std::time::Duration::from_nanos(total_ns)
        });
    });
    group.finish();
}

criterion_group!(
    resident_q4km_matmul_rb_f32acc_cached_bigrb_benches,
    bench_resident_q4km_matmul_rb_f32acc_cached_bigrb
);
criterion_main!(resident_q4km_matmul_rb_f32acc_cached_bigrb_benches);
