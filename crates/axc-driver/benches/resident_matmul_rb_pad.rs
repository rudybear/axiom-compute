//! AT-2406 + AT-2410 (M3.10a): honest effective-TFLOPS bench for the BANK-PADDED plain-f32 RB
//! coopmat matmul (examples/matmul_rb_coopmat_pad.axc) vs the unpadded M3.3c RB leader
//! (examples/matmul_rb_coopmat.axc).
//!
//! Bench ID: `dispatch_resident_matmul_rb_pad`.
//!
//! ## AT-2406 — padded-vs-base TFLOPS at 256/512/768/1024 cube
//! PRIMARY GATE: padded 768³ TFLOPS >= 1.15x the UNPADDED base at 768³. The gate is PRINTED +
//! CHECKED, NEVER asserted-loose. HONEST-NEGATIVE ARMED: if padding does NOT reach 1.15x (the M3.6
//! tile may already be conflict-free on Blackwell — OQ-1; OR the +pad shared footprint drops
//! occupancy and net-regresses — the M3.7/M3.3d mechanism), the MEASURED per-PAD ratio + per-size
//! shared_memory_bytes + an occupancy/bank diagnosis are reported, the gate is NOT loosened, and
//! the unpadded RB remains the production leader.
//!
//! ## AT-2410 — pad-hole sweep PAD in {1,2,4,8}
//! Enumerates PAD, reports each variant's 768³ TFLOPS + ratio + shared_memory_bytes; the winning
//! pad is the MEASURED winner (NOT hardcoded blind). The pinned ship pad is whichever wins here.
//!
//! Gated on AXC_ENABLE_GPU_BENCHES=1 + a responsive Vulkan ICD. Typed-skip on CoopMatUnsupported
//! (Lavapipe) / DeviceFeatureUnsupported / subgroup_size() != 32.

#![allow(dead_code)]

use criterion::{criterion_group, criterion_main, Criterion};
use axc_driver::compile_source_with_assignments;
use axc_runtime::{
    VulkanContext, DispatchError, KernelHandle,
    ResidentBenchConfig, ResidentTimingSource,
};
use std::collections::BTreeMap;

const PAD_SRC: &str = include_str!("../../../examples/matmul_rb_coopmat_pad.axc");
const BASE_SRC: &str = include_str!("../../../examples/matmul_rb_coopmat.axc");

const CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000: f64 = 125.0_f64;
const COMPETITIVE_PCT_THRESHOLD: f64 = 25.0_f64;
/// PRIMARY gate ratio: padded must beat the unpadded base by this factor at 768³ (NEVER loosened).
const PAD_GATE_RATIO: f64 = 1.15_f64;

const N_WARMUP: usize = 2;
const N_MEASURED: usize = 10;
const FIFTY_GB: u64 = 50 * 1024 * 1024 * 1024;

/// The swept pad candidate set (AT-2410). The winner is MEASURED, not hardcoded.
const PAD_CANDIDATES: [i64; 4] = [1, 2, 4, 8];

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

fn effective_tflops(m: usize, n: usize, k: usize, kernel_ns: u64) -> f64 {
    if kernel_ns == 0 {
        return 0.0;
    }
    let secs = kernel_ns as f64 * 1e-9_f64;
    2.0_f64 * (m as f64) * (n as f64) * (k as f64) / secs / 1e12_f64
}

/// Unpadded base RB 2×2 assignments.
fn base_assignments() -> BTreeMap<String, i64> {
    let mut m = BTreeMap::new();
    m.insert("rb_m".to_owned(), 2_i64);
    m.insert("rb_n".to_owned(), 2_i64);
    m.insert("tile_k".to_owned(), 16_i64);
    m.insert("a_block_size".to_owned(), 512_i64);
    m.insert("b_block_size".to_owned(), 512_i64);
    m
}

/// Padded RB assignments for a given PAD (applied to both A and B free dims).
/// a_pad_stride=16+p, a_pad_size=32*(16+p), a_pad_mat1off=16*(16+p) (SCALES with pad),
/// b_pad_stride=32+p, b_pad_size=16*(32+p).
fn pad_assignments(pad: i64) -> BTreeMap<String, i64> {
    let mut m = base_assignments();
    m.insert("a_pad_stride".to_owned(), 16 + pad);
    m.insert("a_pad_size".to_owned(), 32 * (16 + pad));
    m.insert("a_pad_mat1off".to_owned(), 16 * (16 + pad));
    m.insert("b_pad_stride".to_owned(), 32 + pad);
    m.insert("b_pad_size".to_owned(), 16 * (32 + pad));
    m
}

/// Build f16 LE bytes for an integer-pattern matrix of `len` elements.
fn f16_fill(len: usize, modv: usize) -> Vec<u8> {
    use half::f16;
    (0..len).flat_map(|i| f16::from_f32(((i % modv) + 1) as f32).to_le_bytes()).collect()
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

/// Measure one kernel handle at (m,n,k). Returns (tflops, min_ns, timing_source), or None on a
/// grid/cap/skip failure.
fn measure_tflops(
    ctx: &VulkanContext,
    handle: &KernelHandle,
    label: &str,
    m: usize,
    n: usize,
    k: usize,
) -> Option<(f64, u64, ResidentTimingSource)> {
    if !m.is_multiple_of(32) || !n.is_multiple_of(32) {
        eprintln!("{label}: M={m}/N={n} not multiples of 32 — skip");
        return None;
    }
    let wg_x = (n / 32) as u64;
    let wg_y = (m / 32) as u64;
    let max_wg = ctx.max_compute_work_group_count();
    if wg_x > max_wg[0] as u64 || wg_y > max_wg[1] as u64 {
        eprintln!("{label}: grid ({wg_x},{wg_y}) exceeds limits at M={m} N={n} K={k} — skip");
        return None;
    }
    let total_bytes: u64 = 2 * (m as u64) * (k as u64)
        + 2 * (k as u64) * (n as u64)
        + 2 * (m as u64) * (n as u64);
    if total_bytes >= FIFTY_GB {
        eprintln!("{label}: 50 GB cap exceeded at M={m} N={n} K={k} — skip");
        return None;
    }

    let a_bytes = f16_fill(m * k, 4);
    let b_bytes = f16_fill(k * n, 3);
    let c_size = m * n * 2; // f16 output
    let mut pc = Vec::with_capacity(12);
    pc.extend_from_slice(&(m as u32).to_le_bytes());
    pc.extend_from_slice(&(n as u32).to_le_bytes());
    pc.extend_from_slice(&(k as u32).to_le_bytes());
    let workgroups = (wg_x as u32, wg_y as u32, 1);

    let (min_ns, ts) = resident_min_of_n(
        ctx, handle, &[&a_bytes, &b_bytes, &vec![0u8; c_size]],
        &[0, 0, c_size as u64], workgroups, pc,
    );
    let tflops = effective_tflops(m, n, k, min_ns);
    Some((tflops, min_ns, ts))
}

/// Prepare a kernel handle from source+assignments. Returns (handle, shared_memory_bytes), or
/// None on a typed-skip.
fn prepare(
    ctx: &VulkanContext,
    src: &str,
    assignments: &BTreeMap<String, i64>,
    kernel_name: &str,
    label: &str,
) -> Option<(KernelHandle, u32)> {
    let (bytes, meta) = match compile_source_with_assignments(src, assignments) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{label}: compile failed: {e:?}");
            return None;
        }
    };
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    match ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, meta.coopmat.as_ref(), kernel_name, meta.shared_memory_bytes,
    ) {
        Ok(h) => Some((h, meta.shared_memory_bytes)),
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("{label}: CoopMatUnsupported (typed-skip): {reason}");
            None
        }
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
            eprintln!("{label}: DeviceFeatureUnsupported {feature}/{kernel} (typed-skip)");
            None
        }
        Err(e) => {
            eprintln!("{label}: prepare failed: {e}");
            None
        }
    }
}

fn bench_resident_matmul_rb_pad(c: &mut Criterion) {
    if !gpu_benches_enabled() {
        eprintln!("resident_matmul_rb_pad: AXC_ENABLE_GPU_BENCHES not set; skipping");
        c.bench_function("dispatch_resident_matmul_rb_pad", |b| b.iter(|| {}));
        return;
    }
    let ctx = match VulkanContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("resident_matmul_rb_pad: no Vulkan: {e}");
            c.bench_function("dispatch_resident_matmul_rb_pad", |b| b.iter(|| {}));
            return;
        }
    };
    if !ctx.coopmat_support().feature_present {
        eprintln!("resident_matmul_rb_pad: coopmat unsupported on {}; typed-skip", ctx.physical_device_name());
        c.bench_function("dispatch_resident_matmul_rb_pad", |b| b.iter(|| {}));
        return;
    }
    if ctx.subgroup_size() != 32 {
        eprintln!("resident_matmul_rb_pad: subgroup_size()={} != 32; typed-skip", ctx.subgroup_size());
        c.bench_function("dispatch_resident_matmul_rb_pad", |b| b.iter(|| {}));
        return;
    }

    // ── Base (unpadded) reference at every size ─────────────────────────────────
    let Some((base_handle, base_shared)) = prepare(
        &ctx, BASE_SRC, &base_assignments(), "matmul_rb_coopmat", "resident_matmul_rb_pad[base]",
    ) else {
        c.bench_function("dispatch_resident_matmul_rb_pad", |b| b.iter(|| {}));
        return;
    };
    eprintln!("resident_matmul_rb_pad: base shared_memory_bytes={base_shared}");

    let sizes = [256usize, 512, 768, 1024];
    let mut base_768: f64 = 0.0;
    for &s in &sizes {
        if let Some((tf, ns, ts)) = measure_tflops(&ctx, &base_handle, "resident_matmul_rb_pad[base]", s, s, s) {
            let pct = tf / CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000 * 100.0;
            eprintln!(
                "resident_matmul_rb_pad[base] = {tf:.3} TFLOPS ({pct:.2}%) | {ns} ns ({}) | \
                 {s}^3 | shared={base_shared} B",
                timing_source_label(ts)
            );
            if s == 768 { base_768 = tf; }
        }
    }

    // ── Padded variants: sweep PAD in {1,2,4,8} (AT-2410) ───────────────────────
    // Track the best (winning) pad by 768³ TFLOPS.
    let mut best: Option<(i64, f64, f64, u32)> = None; // (pad, tflops_768, ratio_768, shared)
    for &pad in &PAD_CANDIDATES {
        let label = format!("resident_matmul_rb_pad[PAD={pad}]");
        let Some((handle, shared)) = prepare(
            &ctx, PAD_SRC, &pad_assignments(pad), "matmul_rb_coopmat_pad", &label,
        ) else { continue; };

        let mut pad_768: f64 = 0.0;
        for &s in &sizes {
            if let Some((tf, ns, ts)) = measure_tflops(&ctx, &handle, &label, s, s, s) {
                let pct = tf / CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000 * 100.0;
                eprintln!(
                    "{label} = {tf:.3} TFLOPS ({pct:.2}%) | {ns} ns ({}) | {s}^3 | shared={shared} B",
                    timing_source_label(ts)
                );
                if s == 768 { pad_768 = tf; }
            }
        }
        if base_768 > 0.0 && pad_768 > 0.0 {
            let ratio = pad_768 / base_768;
            eprintln!(
                "{label}: 768^3 padded={pad_768:.3} vs base={base_768:.3} -> {ratio:.4}x \
                 (gate >= {PAD_GATE_RATIO}x) | shared {shared} B vs base {base_shared} B \
                 (+{} B)",
                shared as i64 - base_shared as i64
            );
            match best {
                Some((_, btf, _, _)) if pad_768 <= btf => {}
                _ => best = Some((pad, pad_768, ratio, shared)),
            }
        }
    }

    // ── PRIMARY GATE evaluation (printed + checked, NEVER asserted-loose) ────────
    if let Some((pad, tf, ratio, shared)) = best {
        let gate_met = ratio >= PAD_GATE_RATIO;
        if gate_met {
            eprintln!(
                "resident_matmul_rb_pad: AT-2406 GATE-MET — winning PAD={pad} at 768^3 = {tf:.3} \
                 TFLOPS = {ratio:.4}x the unpadded base (>= {PAD_GATE_RATIO}x). shared={shared} B."
            );
        } else {
            // HONEST-NEGATIVE: report the measured ratio + the occupancy/bank diagnosis.
            // static-shared-fits != occupancy-neutral: +pad shared can cut resident warps/SM even
            // though it stays within the portable floor (the M3.7/M3.3d mechanism).
            eprintln!(
                "resident_matmul_rb_pad: AT-2406 HONEST-NEGATIVE — best PAD={pad} at 768^3 = {tf:.3} \
                 TFLOPS = {ratio:.4}x the unpadded base (< {PAD_GATE_RATIO}x gate, NOT loosened). \
                 DIAGNOSIS: padded shared={shared} B (+{} B vs base {base_shared} B). static-shared- \
                 fits (<= 16384 portable floor) does NOT prove occupancy-neutral: the extra shared \
                 can cut resident-warps/SM (M3.7/M3.3d occupancy mechanism), OR the M3.6 16-wide f16 \
                 coopmat tile-load already swizzles the conflict away on Blackwell (OQ-1). The \
                 unpadded RB remains the production leader; the 2.39x close likely needs M3.10b \
                 vec-loads, not bank padding.",
                shared as i64 - base_shared as i64
            );
        }
        // The gate is CHECKED here (printed verdict) but NOT asserted-loose — the bench asserts
        // only that the measurement is well-formed, so an honest-negative does not fail the run.
        assert!(tf > 0.0 && tf.is_finite(), "AT-2406: padded tflops must be > 0 and finite");
    } else {
        eprintln!(
            "resident_matmul_rb_pad: AT-2406 — no padded 768^3 measurement (typed-skip or grid/cap); \
             nothing to gate."
        );
    }

    // Criterion duration: re-time the base 256³.
    let mut group = c.benchmark_group("resident_matmul_rb_pad");
    group.sample_size(10);
    group.bench_function("dispatch_resident_matmul_rb_pad", |b| {
        b.iter_custom(|iters| {
            let mut total_ns: u64 = 0;
            for _ in 0..iters {
                if let Some((_, ns, _)) = measure_tflops(&ctx, &base_handle, "crit", 256, 256, 256) {
                    total_ns += ns;
                } else {
                    total_ns += 1;
                }
            }
            std::time::Duration::from_nanos(total_ns)
        });
    });
    group.finish();
}

criterion_group!(resident_matmul_rb_pad_benches, bench_resident_matmul_rb_pad);
criterion_main!(resident_matmul_rb_pad_benches);
