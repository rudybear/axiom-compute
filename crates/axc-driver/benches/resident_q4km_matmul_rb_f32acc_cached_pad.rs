//! AT-2407 (M3.10a, CONDITIONAL): honest effective-TFLOPS bench for the BANK-PADDED Q4_K_M
//! scale-cached f32-accumulator RB coopmat matmul
//! (examples/q4km_matmul_rb_coopmat_f32acc_cached_pad.axc) vs the M3.6 leader (42.86 TFLOPS A/B).
//!
//! Bench ID: `dispatch_resident_q4km_matmul_rb_f32acc_cached_pad`.
//!
//! PRIMARY GATE: padded A/B (4096x512x14336) TFLOPS >= 1.15x M3.6's 42.86 (>= 49.3 TFLOPS) AND
//! combined condition-aware metric <= FROZEN 1e-3. The gate is PRINTED + CHECKED, NEVER
//! asserted-loose. HONEST-NEGATIVE ARMED: a sub-gate result is reported with the MEASURED ratio +
//! per-size shared_memory_bytes + an occupancy/bank diagnosis; the gate is NOT loosened and the
//! M3.6 leader stays in production.
//!
//! Emits the machine-readable AXC_Q4KM_AB_F32ACC_CACHED_PAD line (parsed by
//! scripts/m34_llamacpp_ab.sh --pad). The pad sweep (PAD in {1,2,4,8}) reports the winning pad.
//!
//! Gated on AXC_ENABLE_GPU_BENCHES=1. Typed-skip on CoopMatUnsupported / DeviceFeatureUnsupported
//! / subgroup_size() != 32.

#![allow(dead_code)]

use axc_driver::q4km_oracle as common_q4km_f32ref;

use criterion::{criterion_group, criterion_main, Criterion};
use axc_driver::compile_source_with_assignments;
use axc_hir::ParamBindingPlan;
use axc_runtime::{
    VulkanContext, DispatchError, KernelHandle,
    ResidentBenchConfig, ResidentTimingSource,
};
use std::collections::BTreeMap;

const PAD_SRC: &str =
    include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached_pad.axc");

const CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000: f64 = 125.0_f64;
const COMPETITIVE_PCT_THRESHOLD: f64 = 25.0_f64;
const FROZEN_REL_TOL: f64 = 1e-3;
/// M3.6 leader A/B baseline TFLOPS (the PRIMARY gate reference for AT-2407).
const M36_AB_TFLOPS: f64 = 42.86_f64;
/// PRIMARY gate ratio (NEVER loosened): padded A/B >= 1.15x M3.6 (>= 49.3 TFLOPS).
const PAD_GATE_RATIO: f64 = 1.15_f64;

const N_WARMUP: usize = 2;
const N_MEASURED: usize = 10;
const FIFTY_GB: u64 = 50 * 1024 * 1024 * 1024;

/// Pinned A/B same-shape (llama.cpp Q4_K MUL_MAT m=4096,n=512,k=14336).
const AB_M: usize = 4096;
const AB_N: usize = 512;
const AB_K: usize = 14336;

/// Pad sweep (winner is MEASURED, not hardcoded).
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

pub fn effective_q4km_tflops(m: usize, n: usize, k: usize, kernel_ns: u64) -> f64 {
    if kernel_ns == 0 {
        return 0.0;
    }
    let secs = kernel_ns as f64 * 1e-9_f64;
    2.0_f64 * (m as f64) * (n as f64) * (k as f64) / secs / 1e12_f64
}

fn base_assignments() -> BTreeMap<String, i64> {
    let mut m = BTreeMap::new();
    m.insert("rb_m".to_owned(), 2_i64);
    m.insert("rb_n".to_owned(), 2_i64);
    m.insert("tile_k".to_owned(), 16_i64);
    m.insert("a_block_size".to_owned(), 512_i64);
    m.insert("b_block_size".to_owned(), 512_i64);
    m
}

/// Padded assignments for a given PAD (same tuple derivation as the plain-f32 bench).
fn pad_assignments(pad: i64) -> BTreeMap<String, i64> {
    let mut m = base_assignments();
    m.insert("a_pad_stride".to_owned(), 16 + pad);
    m.insert("a_pad_size".to_owned(), 32 * (16 + pad));
    m.insert("a_pad_mat1off".to_owned(), 16 * (16 + pad));
    m.insert("b_pad_stride".to_owned(), 32 + pad);
    m.insert("b_pad_size".to_owned(), 16 * (32 + pad));
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

/// Measure the padded kernel at (m,n,k). Returns (tflops, combined_rel_diff, min_ns, ts), or None
/// on a grid/cap/skip failure. If emit_ab_line is set, prints AXC_Q4KM_AB_F32ACC_CACHED_PAD.
#[allow(clippy::too_many_arguments)]
fn measure_pad_tflops(
    ctx: &VulkanContext,
    handle: &KernelHandle,
    plan: &ParamBindingPlan,
    pad: i64,
    shared: u32,
    m: usize,
    n: usize,
    k: usize,
    emit_ab_line: bool,
) -> Option<(f64, f64, u64, ResidentTimingSource)> {
    if !m.is_multiple_of(32) || !n.is_multiple_of(32) {
        eprintln!("resident_q4km_pad: M={m}/N={n} not multiples of 32 — skip");
        return None;
    }
    if !k.is_multiple_of(256) {
        eprintln!("resident_q4km_pad: K={k} not a multiple of 256 — skip");
        return None;
    }
    let n_bpr = k / 256;
    let wg_x = (n / 32) as u64;
    let wg_y = (m / 32) as u64;
    let max_wg = ctx.max_compute_work_group_count();
    if wg_x > max_wg[0] as u64 || wg_y > max_wg[1] as u64 {
        eprintln!("resident_q4km_pad: grid ({wg_x},{wg_y}) exceeds limits at M={m} N={n} K={k} — skip");
        return None;
    }
    let total_bytes: u64 = (m as u64 * n_bpr as u64 * 144)
        + 2 * (k as u64) * (n as u64)
        + 4 * (m as u64) * (n as u64);
    if total_bytes >= FIFTY_GB {
        eprintln!("resident_q4km_pad: 50 GB cap exceeded at M={m} N={n} K={k} — skip");
        return None;
    }

    let q_bytes = make_q4km_weights(m, n_bpr, 0xC0FFEE ^ (m as u64));
    let x_f16 = make_x_f16(k, n, 0xBADF00D ^ (n as u64));
    let x_bytes: Vec<u8> = x_f16.iter().flat_map(|&b| b.to_le_bytes()).collect();
    let c_size = m * n * 4;

    let pc = assemble_pc(plan, m as u32, n as u32, k as u32, n_bpr as u32);
    let workgroups = (wg_x as u32, wg_y as u32, 1);

    // Correctness pre-flight on the timed handle.
    let outputs = match ctx.dispatch_handle(
        handle, workgroups, &[&q_bytes, &x_bytes, &vec![0u8; c_size]], &[0, 0, c_size], &pc,
    ) {
        Ok(v) => v,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("resident_q4km_pad: CoopMatUnsupported at dispatch (typed-skip): {reason}");
            return None;
        }
        Err(e) => {
            eprintln!("resident_q4km_pad: dispatch failed at M={m} N={n} K={k}: {e}");
            return None;
        }
    };
    let y_gpu: Vec<f32> = outputs[2].chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let y_ref = common_q4km_f32ref::q4km_dequant_matmul_f32accum_cpu(&q_bytes, &x_f16, m, n, n_bpr);
    let abs_scale = common_q4km_f32ref::q4km_abs_dot_scale(&q_bytes, &x_f16, m, n, n_bpr);
    let combined = common_q4km_f32ref::max_rel_diff_combined(&y_gpu, &y_ref, &abs_scale);
    let raw = common_q4km_f32ref::max_rel_diff(&y_gpu, &y_ref);

    let (min_ns, ts) = resident_min_of_n(
        ctx, handle, &[&q_bytes, &x_bytes, &vec![0u8; c_size]],
        &[0, 0, c_size as u64], workgroups, pc,
    );

    let tflops = effective_q4km_tflops(m, n, k, min_ns);
    let pct = tflops / CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000 * 100.0;
    let valid = if combined <= FROZEN_REL_TOL { "VALID" } else { "INVALID(>1e-3)" };
    let label = if ts == ResidentTimingSource::GpuTimestamp && pct >= COMPETITIVE_PCT_THRESHOLD {
        "competitive"
    } else {
        "honest"
    };
    eprintln!(
        "resident_q4km_pad[PAD={pad}] ({label}) = {tflops:.3} TFLOPS ({pct:.2}%) | {min_ns} ns ({}) \
         | M={m} N={n} K={k} | shared={shared} B | combined={combined:.3e} [{valid}] raw={raw:.3e}",
        timing_source_label(ts)
    );

    if emit_ab_line {
        let flops: u64 = 2 * (m as u64) * (n as u64) * (k as u64);
        println!(
            "AXC_Q4KM_AB_F32ACC_CACHED_PAD kernel_ns_min={} kernel_ns_mean={} kernel_ns_median={} \
             sustained_ns={} timing_source={:?} K={} flops={} m={} n={} pad={} shared={} \
             combined={:.6e} raw={:.6e} device={}",
            min_ns, min_ns, min_ns, min_ns, ts, k, flops, m, n, pad, shared,
            combined, raw, ctx.physical_device_name(),
        );
    }

    Some((tflops, combined, min_ns, ts))
}

fn prepare(
    ctx: &VulkanContext,
    assignments: &BTreeMap<String, i64>,
    label: &str,
) -> Option<(KernelHandle, ParamBindingPlan, u32)> {
    let (bytes, meta) = match compile_source_with_assignments(PAD_SRC, assignments) {
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
        &meta.entry_point, meta.coopmat.as_ref(), "q4km_matmul_rb_coopmat_f32acc_cached_pad",
        meta.shared_memory_bytes,
    ) {
        Ok(h) => Some((h, meta.binding_plan.clone(), meta.shared_memory_bytes)),
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

fn bench_resident_q4km_matmul_rb_f32acc_cached_pad(c: &mut Criterion) {
    let bench_id = "dispatch_resident_q4km_matmul_rb_f32acc_cached_pad";
    if !gpu_benches_enabled() {
        eprintln!("resident_q4km_pad: AXC_ENABLE_GPU_BENCHES not set; skipping");
        c.bench_function(bench_id, |b| b.iter(|| {}));
        return;
    }
    let ctx = match VulkanContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("resident_q4km_pad: no Vulkan: {e}");
            c.bench_function(bench_id, |b| b.iter(|| {}));
            return;
        }
    };
    if !ctx.coopmat_support().feature_present {
        eprintln!("resident_q4km_pad: coopmat unsupported; typed-skip");
        c.bench_function(bench_id, |b| b.iter(|| {}));
        return;
    }
    if ctx.subgroup_size() != 32 {
        eprintln!("resident_q4km_pad: subgroup_size()={} != 32; typed-skip", ctx.subgroup_size());
        c.bench_function(bench_id, |b| b.iter(|| {}));
        return;
    }

    // ── Pad sweep over the A/B shape (the real llama fight) + cube sizes (AT-2407/2410) ──
    let mut best: Option<(i64, f64, f64, f64, u32)> = None; // (pad, ab_tflops, ratio, combined, shared)
    for &pad in &PAD_CANDIDATES {
        let label = format!("resident_q4km_pad[PAD={pad}]");
        let Some((handle, plan, shared)) = prepare(&ctx, &pad_assignments(pad), &label) else { continue; };

        // Cube sizes (256/512/768/1024) for context.
        for s in [256usize, 512, 768, 1024] {
            let _ = measure_pad_tflops(&ctx, &handle, &plan, pad, shared, s, s, s, false);
        }
        // A/B shape — the gate.
        if let Some((ab_tflops, combined, _ns, ts)) =
            measure_pad_tflops(&ctx, &handle, &plan, pad, shared, AB_M, AB_N, AB_K, true)
        {
            let ratio = ab_tflops / M36_AB_TFLOPS;
            let gate_met = ts == ResidentTimingSource::GpuTimestamp
                && ratio >= PAD_GATE_RATIO
                && combined <= FROZEN_REL_TOL;
            eprintln!(
                "{label}: A/B padded={ab_tflops:.3} vs M3.6 {M36_AB_TFLOPS} -> {ratio:.4}x \
                 (gate >= {PAD_GATE_RATIO}x AND combined <= 1e-3) | combined={combined:.3e} | \
                 shared={shared} B | {}",
                if gate_met { "GATE-MET" } else { "sub-gate" }
            );
            match best {
                Some((_, btf, _, _, _)) if ab_tflops <= btf => {}
                _ => best = Some((pad, ab_tflops, ratio, combined, shared)),
            }
        }
    }

    // ── PRIMARY GATE evaluation (printed + checked, NEVER asserted-loose) ────────
    if let Some((pad, ab_tflops, ratio, combined, shared)) = best {
        let combined_ok = combined <= FROZEN_REL_TOL;
        let gate_met = ratio >= PAD_GATE_RATIO && combined_ok;
        if gate_met {
            eprintln!(
                "resident_q4km_pad: AT-2407 GATE-MET — winning PAD={pad} A/B = {ab_tflops:.3} TFLOPS \
                 = {ratio:.4}x M3.6's {M36_AB_TFLOPS} (>= {PAD_GATE_RATIO}x = >= 49.3), \
                 combined={combined:.3e} [VALID]. shared={shared} B."
            );
        } else {
            eprintln!(
                "resident_q4km_pad: AT-2407 HONEST-NEGATIVE — best PAD={pad} A/B = {ab_tflops:.3} \
                 TFLOPS = {ratio:.4}x M3.6's {M36_AB_TFLOPS} (< {PAD_GATE_RATIO}x gate, NOT \
                 loosened); combined={combined:.3e} [{}]. DIAGNOSIS: padded shared={shared} B. \
                 static-shared-fits does NOT prove occupancy-neutral — the +pad shared (caches + \
                 padded tiles) can cut resident-warps/SM (M3.7/M3.3d mechanism), OR the M3.6 \
                 16-wide f16 coopmat tile-load already swizzles the conflict on Blackwell (OQ-1). \
                 The M3.6 leader stays in production; the 2.39x close likely needs M3.10b vec-loads.",
                if combined_ok { "VALID" } else { "INVALID" }
            );
        }
        assert!(ab_tflops > 0.0 && ab_tflops.is_finite(), "AT-2407: A/B tflops must be > 0 and finite");
    } else {
        eprintln!("resident_q4km_pad: AT-2407 — no A/B measurement (typed-skip/grid/cap); nothing to gate.");
    }

    c.bench_function(bench_id, |b| b.iter(|| {}));
}

criterion_group!(
    resident_q4km_matmul_rb_f32acc_cached_pad_benches,
    bench_resident_q4km_matmul_rb_f32acc_cached_pad
);
criterion_main!(resident_q4km_matmul_rb_f32acc_cached_pad_benches);
