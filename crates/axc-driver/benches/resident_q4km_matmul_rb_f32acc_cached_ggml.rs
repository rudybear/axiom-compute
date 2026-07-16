//! M4.2a — AT-2996: honest effective-TFLOPS bench for the ggml-ABI-MATCHED variant
//! (examples/q4km_matmul_rb_coopmat_f32acc_cached_ggml.axc), run ALONGSIDE the M3.6 leader
//! (examples/q4km_matmul_rb_coopmat_f32acc_cached.axc) at the SAME sizes, on the SAME device.
//!
//! Bench ID: `dispatch_resident_q4km_matmul_rb_f32acc_cached_ggml`.
//!
//! SIBLING of resident_q4km_matmul_rb_f32acc_cached.rs — clones its resident-dispatch/timing
//! discipline (MIN-of-10, GpuTimestamp, the 50 GB cap, the 2D-grid pre-check) but (a) dispatches
//! BOTH kernels so the ggml variant's cost is measured directly against the unperturbed leader,
//! (b) feeds the ggml variant B as ggml's `[N,K]` row-major f32 (transposed + widened from the
//! SAME logical x_f16 fixture — §5.1/§5.2 of the milestone spec) and reads its D column-major
//! (de-transposed for the correctness readout), and (c) dispatches it on the ggml grid-axis
//! convention `(M/32, N/32, 1)`.
//!
//! HONESTY CONTRACT (identical to every M3.x sibling): the COMBINED condition-aware metric
//! (`|gpu-ref|/max(|ref|,Sum|wk xk|) <= frozen 1e-3`) drives VALID/INVALID; RAW forward error is
//! reported for transparency. **NO ratio gate** — M3.13 concluded the NVIDIA throughput campaign;
//! this bench records the extra per-B-element f32->f16 staging convert cost + the
//! transposed-B/column-major-D addressing cost HONESTLY (the fixed-32x32-tile vs ggml's l/m/s
//! selection caveat is a REPRODUCE.md/A-B note, not gated).
//!
//! AT-2996 asserts only that BOTH kernels run and produce a finite positive TFLOPS number.
//!
//! Gated on AXC_ENABLE_GPU_BENCHES=1 + a responsive Vulkan ICD. Typed-skip on
//! CoopMatUnsupported (Lavapipe) / DeviceFeatureUnsupported / subgroup_size() != 32.

#![allow(dead_code)]

use axc_driver::q4km_oracle as common_q4km_f32ref;

use criterion::{criterion_group, criterion_main, Criterion};
use axc_driver::compile_source_with_assignments;
use axc_hir::ParamBindingPlan;
use axc_runtime::{
    VulkanContext, KernelHandle,
    ResidentBenchConfig, ResidentTimingSource,
};
use std::collections::BTreeMap;
use std::sync::Mutex;

/// Per-(kernel, m, n, k) memo of the correctness pre-flight (combined, raw). The CPU oracle +
/// abs_scale at the A/B shape cost MINUTES each; criterion re-invokes the measurement closure
/// many times, and recomputing them every invocation made the bench take hours. The oracle IS
/// still computed — exactly ONCE per (kernel, shape) per process — on the FIRST invocation (the
/// fixture seeds are identical every invocation, so the value cannot differ); subsequent
/// invocations reuse the memoized correctness value and re-time only the GPU dispatch (which is
/// the thing criterion is actually sampling).
#[allow(clippy::type_complexity)]
static CORRECTNESS_MEMO: Mutex<BTreeMap<(u8, usize, usize, usize), (f64, f64)>> =
    Mutex::new(BTreeMap::new());
const MEMO_LEADER: u8 = 0;
const MEMO_GGML: u8 = 1;

const LEADER_SRC: &str =
    include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached.axc");
const GGML_SRC: &str =
    include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached_ggml.axc");

/// Frozen relative tolerance (AT-1520/AT-1521 value — NOT loosened).
const FROZEN_REL_TOL: f64 = 1e-3;

const N_WARMUP: usize = 2;
const N_MEASURED: usize = 10;

/// 50 GB executable-memory cap.
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

fn assemble_pc_leader(plan: &ParamBindingPlan, m: u32, n: u32, k: u32, n_bpr: u32) -> Vec<u8> {
    let mut pc: Vec<u8> = vec![0u8; plan.push_constant_total_bytes as usize];
    for s in &plan.scalars {
        let val: u32 = match s.name.as_str() {
            "M" => m,
            "N" => n,
            "K" => k,
            "n_blocks_per_row" => n_bpr,
            other => panic!("unexpected leader scalar {other}"),
        };
        pc[s.offset as usize..s.offset as usize + 4].copy_from_slice(&val.to_le_bytes());
    }
    pc
}

fn assemble_pc_ggml(plan: &ParamBindingPlan, m: u32, n: u32, k: u32, stride_a: u32, stride_b: u32, stride_d: u32) -> Vec<u8> {
    let mut pc: Vec<u8> = vec![0u8; plan.push_constant_total_bytes as usize];
    for s in &plan.scalars {
        let val: u32 = match s.name.as_str() {
            "M" => m,
            "N" => n,
            "K" => k,
            "stride_a" => stride_a,
            "stride_b" => stride_b,
            "stride_d" => stride_d,
            other => panic!("unexpected ggml scalar {other}"),
        };
        pc[s.offset as usize..s.offset as usize + 4].copy_from_slice(&val.to_le_bytes());
    }
    pc
}

/// Non-symmetric Q4_K_M weight matrix fixture (SAME seed scheme as the M3.6 bench).
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

/// §5.2 f16-EXACT widen: x_ggml_f32[col*k_total + k] = f32(x_f16[k*n + col]) — exact.
fn transpose_widen_b(x_f16: &[u16], k_total: usize, n: usize) -> Vec<f32> {
    use half::f16;
    let mut out = vec![0.0_f32; k_total * n];
    for k in 0..k_total {
        for col in 0..n {
            out[col * k_total + k] = f16::from_bits(x_f16[k * n + col]).to_f32();
        }
    }
    out
}

fn detranspose_d(d_col_major: &[f32], m: usize, n: usize, stride_d: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; m * n];
    for row in 0..m {
        for col in 0..n {
            out[row * n + col] = d_col_major[col * stride_d + row];
        }
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

/// Measure the M3.6 LEADER at (m,n,k). Returns (tflops, min_ns, timing_source, combined) or None
/// on typed-skip / grid / cap failure.
#[allow(clippy::too_many_arguments)]
fn measure_leader(
    ctx: &VulkanContext, handle: &KernelHandle, plan: &ParamBindingPlan,
    m: usize, n: usize, k: usize,
) -> Option<(f64, u64, ResidentTimingSource, f64, f64)> {
    if !m.is_multiple_of(32) || !n.is_multiple_of(32) || !k.is_multiple_of(256) {
        return None;
    }
    let n_bpr = k / 256;
    let wg_x = (n / 32) as u64;
    let wg_y = (m / 32) as u64;
    let max_wg = ctx.max_compute_work_group_count();
    if wg_x > max_wg[0] as u64 || wg_y > max_wg[1] as u64 {
        return None;
    }
    let total_bytes: u64 = (m as u64 * n_bpr as u64 * 144)
        + 2 * (k as u64) * (n as u64)
        + 4 * (m as u64) * (n as u64);
    if total_bytes >= FIFTY_GB {
        return None;
    }

    let q_bytes = make_q4km_weights(m, n_bpr, 0xC0FFEE ^ (m as u64));
    let x_f16 = make_x_f16(k, n, 0xBADF00D ^ (n as u64));
    let x_bytes: Vec<u8> = x_f16.iter().flat_map(|&b| b.to_le_bytes()).collect();
    let c_size = m * n * 4;

    let pc = assemble_pc_leader(plan, m as u32, n as u32, k as u32, n_bpr as u32);
    let workgroups = (wg_x as u32, wg_y as u32, 1);

    // Correctness pre-flight (memoized — the oracle runs ONCE per shape per process).
    let memo_key = (MEMO_LEADER, m, n, k);
    let cached = CORRECTNESS_MEMO.lock().unwrap().get(&memo_key).copied();
    let (combined, raw) = match cached {
        Some(v) => v,
        None => {
            let outputs = ctx.dispatch_handle(
                handle, workgroups,
                &[&q_bytes, &x_bytes, &vec![0u8; c_size]],
                &[0, 0, c_size],
                &pc,
            ).ok()?;
            let y_gpu: Vec<f32> = outputs[2].chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let y_ref = common_q4km_f32ref::q4km_dequant_matmul_f32accum_cpu(&q_bytes, &x_f16, m, n, n_bpr);
            let abs_scale = common_q4km_f32ref::q4km_abs_dot_scale(&q_bytes, &x_f16, m, n, n_bpr);
            let combined = common_q4km_f32ref::max_rel_diff_combined(&y_gpu, &y_ref, &abs_scale);
            let raw = common_q4km_f32ref::max_rel_diff(&y_gpu, &y_ref);
            CORRECTNESS_MEMO.lock().unwrap().insert(memo_key, (combined, raw));
            (combined, raw)
        }
    };

    let (min_ns, ts) = resident_min_of_n(
        ctx, handle, &[&q_bytes, &x_bytes, &vec![0u8; c_size]], &[0, 0, c_size as u64], workgroups, pc,
    );
    let tflops = effective_q4km_tflops(m, n, k, min_ns);
    Some((tflops, min_ns, ts, combined, raw))
}

/// Measure the ggml VARIANT at (m,n,k). Returns (tflops, min_ns, timing_source, combined, raw)
/// or None on typed-skip / grid / cap failure.
#[allow(clippy::too_many_arguments)]
fn measure_ggml(
    ctx: &VulkanContext, handle: &KernelHandle, plan: &ParamBindingPlan,
    m: usize, n: usize, k: usize,
) -> Option<(f64, u64, ResidentTimingSource, f64, f64)> {
    if !m.is_multiple_of(32) || !n.is_multiple_of(32) || !k.is_multiple_of(256) {
        return None;
    }
    let n_bpr = k / 256;
    let wg_x = (m / 32) as u64; // ggml axis swap: x -> M
    let wg_y = (n / 32) as u64; // y -> N
    let max_wg = ctx.max_compute_work_group_count();
    if wg_x > max_wg[0] as u64 || wg_y > max_wg[1] as u64 {
        return None;
    }
    // B is f32 here (4 B/elem, not 2): total is slightly larger than the leader's.
    let total_bytes: u64 = (m as u64 * n_bpr as u64 * 144)
        + 4 * (k as u64) * (n as u64)
        + 4 * (m as u64) * (n as u64);
    if total_bytes >= FIFTY_GB {
        return None;
    }

    let q_bytes = make_q4km_weights(m, n_bpr, 0xC0FFEE ^ (m as u64));
    let x_f16 = make_x_f16(k, n, 0xBADF00D ^ (n as u64));
    let x_ggml_f32 = transpose_widen_b(&x_f16, k, n);
    let x_bytes: Vec<u8> = x_ggml_f32.iter().flat_map(|&v| v.to_le_bytes()).collect();
    let c_size = m * n * 4;

    let (stride_a, stride_b, stride_d) = (k as u32, k as u32, m as u32);
    let pc = assemble_pc_ggml(plan, m as u32, n as u32, k as u32, stride_a, stride_b, stride_d);
    let workgroups = (wg_x as u32, wg_y as u32, 1);

    // Correctness pre-flight (memoized — the oracle runs ONCE per shape per process).
    let memo_key = (MEMO_GGML, m, n, k);
    let cached = CORRECTNESS_MEMO.lock().unwrap().get(&memo_key).copied();
    let (combined, raw) = match cached {
        Some(v) => v,
        None => {
            let outputs = ctx.dispatch_handle(
                handle, workgroups,
                &[&q_bytes, &x_bytes, &vec![0u8; c_size]],
                &[0, 0, c_size],
                &pc,
            ).ok()?;
            let d_col_major: Vec<f32> = outputs[2].chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let y_gpu = detranspose_d(&d_col_major, m, n, stride_d as usize);
            let y_ref = common_q4km_f32ref::q4km_dequant_matmul_f32accum_cpu(&q_bytes, &x_f16, m, n, n_bpr);
            let abs_scale = common_q4km_f32ref::q4km_abs_dot_scale(&q_bytes, &x_f16, m, n, n_bpr);
            let combined = common_q4km_f32ref::max_rel_diff_combined(&y_gpu, &y_ref, &abs_scale);
            let raw = common_q4km_f32ref::max_rel_diff(&y_gpu, &y_ref);
            CORRECTNESS_MEMO.lock().unwrap().insert(memo_key, (combined, raw));
            (combined, raw)
        }
    };

    let (min_ns, ts) = resident_min_of_n(
        ctx, handle, &[&q_bytes, &x_bytes, &vec![0u8; c_size]], &[0, 0, c_size as u64], workgroups, pc,
    );
    let tflops = effective_q4km_tflops(m, n, k, min_ns);
    Some((tflops, min_ns, ts, combined, raw))
}

/// Measure BOTH kernels at (m,n,k), print the honest A/B, and (if `emit_ab_line`) print the
/// machine-readable AXC_Q4KM_AB_F32ACC_CACHED_GGML line for scripts/m34_llamacpp_ab.sh.
#[allow(clippy::too_many_arguments)]
fn measure_pair(
    ctx: &VulkanContext,
    leader_handle: &KernelHandle, leader_plan: &ParamBindingPlan,
    ggml_handle: &KernelHandle, ggml_plan: &ParamBindingPlan,
    m: usize, n: usize, k: usize, emit_ab_line: bool,
) -> (bool, bool) {
    let leader = measure_leader(ctx, leader_handle, leader_plan, m, n, k);
    let ggml = measure_ggml(ctx, ggml_handle, ggml_plan, m, n, k);

    let leader_ok = leader.is_some();
    let ggml_ok = ggml.is_some();

    if let (Some((l_tf, l_ns, l_ts, l_comb, l_raw)), Some((g_tf, g_ns, g_ts, g_comb, g_raw))) = (&leader, &ggml) {
        let ratio = g_tf / l_tf;
        let leader_valid = if *l_comb <= FROZEN_REL_TOL { "VALID" } else { "INVALID(>1e-3)" };
        let ggml_valid = if *g_comb <= FROZEN_REL_TOL { "VALID" } else { "INVALID(>1e-3)" };
        eprintln!(
            "resident_q4km_ggml_ab: M={m} N={n} K={k} | leader={l_tf:.3} TFLOPS ({l_ns} ns, {}, \
             combined={l_comb:.3e} [{leader_valid}], raw={l_raw:.3e}) | ggml={g_tf:.3} TFLOPS \
             ({g_ns} ns, {}, combined={g_comb:.3e} [{ggml_valid}], raw={g_raw:.3e}) | \
             ggml/leader={ratio:.3}x (honest, NO gate — the f32->f16 B-staging convert + \
             transposed/column-major addressing cost, measured)",
            timing_source_label(*l_ts), timing_source_label(*g_ts),
        );

        if emit_ab_line {
            let flops: u64 = 2 * (m as u64) * (n as u64) * (k as u64);
            println!(
                "AXC_Q4KM_AB_F32ACC_CACHED_GGML kernel_ns_min={g_ns} leader_kernel_ns_min={l_ns} \
                 timing_source={:?} K={k} flops={flops} m={m} n={n} combined={g_comb:.6e} \
                 raw={g_raw:.6e} leader_combined={l_comb:.6e} ggml_tflops={g_tf:.6} \
                 leader_tflops={l_tf:.6} ratio_ggml_over_leader={ratio:.6} device={}",
                g_ts, ctx.physical_device_name(),
            );
        }
    } else {
        eprintln!(
            "resident_q4km_ggml_ab: M={m} N={n} K={k} | leader_ok={leader_ok} ggml_ok={ggml_ok} \
             (at least one typed-skip / grid / cap failure)"
        );
    }

    (leader_ok, ggml_ok)
}

fn bench_resident_q4km_matmul_rb_f32acc_cached_ggml(c: &mut Criterion) {
    if !gpu_benches_enabled() {
        eprintln!("resident_q4km_matmul_rb_f32acc_cached_ggml: AXC_ENABLE_GPU_BENCHES not set; skipping");
        c.bench_function("dispatch_resident_q4km_matmul_rb_f32acc_cached_ggml", |b| b.iter(|| {}));
        return;
    }

    let ctx = match VulkanContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("resident_q4km_matmul_rb_f32acc_cached_ggml: no Vulkan: {e}");
            c.bench_function("dispatch_resident_q4km_matmul_rb_f32acc_cached_ggml", |b| b.iter(|| {}));
            return;
        }
    };
    if !ctx.coopmat_support().feature_present {
        eprintln!("resident_q4km_matmul_rb_f32acc_cached_ggml: coopmat unsupported on {}; typed-skip", ctx.physical_device_name());
        c.bench_function("dispatch_resident_q4km_matmul_rb_f32acc_cached_ggml", |b| b.iter(|| {}));
        return;
    }
    if ctx.subgroup_size() != 32 {
        eprintln!("resident_q4km_matmul_rb_f32acc_cached_ggml: subgroup_size()={} != 32; typed-skip", ctx.subgroup_size());
        c.bench_function("dispatch_resident_q4km_matmul_rb_f32acc_cached_ggml", |b| b.iter(|| {}));
        return;
    }

    let assignments = rb2x2_assignments();
    let (leader_bytes, leader_meta) = match compile_source_with_assignments(LEADER_SRC, &assignments) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("resident_q4km_matmul_rb_f32acc_cached_ggml: leader compile failed: {e:?}");
            c.bench_function("dispatch_resident_q4km_matmul_rb_f32acc_cached_ggml", |b| b.iter(|| {}));
            return;
        }
    };
    let (ggml_bytes, ggml_meta) = match compile_source_with_assignments(GGML_SRC, &assignments) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("resident_q4km_matmul_rb_f32acc_cached_ggml: ggml compile failed: {e:?}");
            c.bench_function("dispatch_resident_q4km_matmul_rb_f32acc_cached_ggml", |b| b.iter(|| {}));
            return;
        }
    };
    let leader_words: Vec<u32> = leader_bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let ggml_words: Vec<u32> = ggml_bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let leader_handle = match ctx.prepare_kernel_checked(
        &leader_words, &leader_meta.binding_plan, leader_meta.push_constant_total_bytes,
        &leader_meta.entry_point, leader_meta.coopmat.as_ref(), "q4km_matmul_rb_coopmat_f32acc_cached",
        leader_meta.shared_memory_bytes,
    ) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("resident_q4km_matmul_rb_f32acc_cached_ggml: leader prepare failed/typed-skip: {e}");
            c.bench_function("dispatch_resident_q4km_matmul_rb_f32acc_cached_ggml", |b| b.iter(|| {}));
            return;
        }
    };
    let ggml_handle = match ctx.prepare_kernel_checked(
        &ggml_words, &ggml_meta.binding_plan, ggml_meta.push_constant_total_bytes,
        &ggml_meta.entry_point, ggml_meta.coopmat.as_ref(), "q4km_matmul_rb_coopmat_f32acc_cached_ggml",
        ggml_meta.shared_memory_bytes,
    ) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("resident_q4km_matmul_rb_f32acc_cached_ggml: ggml prepare failed/typed-skip: {e}");
            c.bench_function("dispatch_resident_q4km_matmul_rb_f32acc_cached_ggml", |b| b.iter(|| {}));
            return;
        }
    };

    let leader_plan = leader_meta.binding_plan.clone();
    let ggml_plan = ggml_meta.binding_plan.clone();

    let mut group = c.benchmark_group("resident_q4km_matmul_rb_f32acc_cached_ggml");
    group.sample_size(10);
    group.bench_function("dispatch_resident_q4km_matmul_rb_f32acc_cached_ggml", |b| {
        b.iter_custom(|iters| {
            let mut total_ns: u64 = 0;
            let mut any_measured = false;
            for _ in 0..iters {
                // AT-2996: cube sizes (both kernels), + the A/B same-shape line.
                for &(m, n, k) in &[(256usize, 256usize, 256usize), (512, 512, 512), (768, 768, 768), (1024, 1024, 1024)] {
                    let (l_ok, g_ok) = measure_pair(&ctx, &leader_handle, &leader_plan, &ggml_handle, &ggml_plan, m, n, k, false);
                    any_measured |= l_ok && g_ok;
                }
                let (l_ok, g_ok) = measure_pair(&ctx, &leader_handle, &leader_plan, &ggml_handle, &ggml_plan, AB_M, AB_N, AB_K, true);
                any_measured |= l_ok && g_ok;

                // Criterion duration: re-time the ggml variant at 256^3.
                let n_bpr = 1usize;
                let q = make_q4km_weights(256, n_bpr, 0xC0FFEE ^ 256);
                let x_f16 = make_x_f16(256, 256, 0xBADF00D ^ 256);
                let xb: Vec<u8> = transpose_widen_b(&x_f16, 256, 256).iter().flat_map(|&v| v.to_le_bytes()).collect();
                let (ns_256, _) = resident_min_of_n(
                    &ctx, &ggml_handle,
                    &[&q, &xb, &vec![0u8; 256 * 256 * 4]],
                    &[0, 0, (256 * 256 * 4) as u64],
                    (8, 8, 1),
                    assemble_pc_ggml(&ggml_plan, 256, 256, 256, 256, 256, 256),
                );
                total_ns += ns_256;
            }
            assert!(any_measured || iters == 0, "AT-2996: at least one (m,n,k) pair must have measured both kernels");
            std::time::Duration::from_nanos(total_ns)
        });
    });
    group.finish();
}

criterion_group!(resident_q4km_matmul_rb_f32acc_cached_ggml_benches, bench_resident_q4km_matmul_rb_f32acc_cached_ggml);
criterion_main!(resident_q4km_matmul_rb_f32acc_cached_ggml_benches);
