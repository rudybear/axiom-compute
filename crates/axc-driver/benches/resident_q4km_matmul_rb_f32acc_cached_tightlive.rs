//! M3.13 PRONG A — AT-2802 + AT-2804: resident TFLOPS for the LIVE-RANGE-TIGHTENED kernel
//! (examples/q4km_matmul_rb_coopmat_f32acc_cached_tightlive.axc) vs the M3.6 leader
//! (examples/q4km_matmul_rb_coopmat_f32acc_cached.axc, 42.86 TFLOPS @ A/B).
//!
//! Bench ID: `dispatch_resident_q4km_matmul_rb_f32acc_cached_tightlive`.
//!
//! Clones resident_q4km_matmul_rb_f32acc_cached.rs and runs BOTH kernels back-to-back on the SAME
//! fixture at the SAME sizes (768³ and the SAME-SHAPE A/B m=4096,n=512,k=14336), MIN-of-10 /
//! GpuTimestamp, so the tightlive-vs-leader ratio is directly comparable. It keeps the SAME honesty
//! contract: the COMBINED condition-aware metric (`|gpu-ref|/max(|ref|,Σ|wₖxₖ|) <= frozen 1e-3`)
//! drives VALID/INVALID; the RAW forward error is reported separately. The 50 GB cap, 2D-grid
//! pre-check, and f32-accumulator oracle are REUSED verbatim.
//!
//! ## AT-2802 — the perf gate + the ARMED DOUBLE honest-negative
//! Prints the tightlive/leader TFLOPS RATIO at 768³ and at A/B, the >=1.15x gate verdict, and the
//! per-variant A-staging temp-count proxy (AT-2804). The gate is NEVER loosened. The LIKELY base
//! case is a FLAT ratio (AXIOM has no scheduler/register-allocator pass; the NVIDIA driver
//! re-allocates SASS registers from the dataflow graph, which a SOURCE let-reorder does NOT change).
//! Two armed honest-negatives, both reported: (a) the proxy does not drop (no codegen leverage); or
//! (b) the proxy drops but TFLOPS is flat (register pressure is not a SOURCE-ADDRESSABLE lever). A
//! clean >=1.15x win is the welcome upside (tightlive could become the new leader). The orchestrator
//! evaluates the gate from the printed AXC_Q4KM_TIGHTLIVE lines.
//!
//! ## AT-2804 — proxy recorded alongside TFLOPS
//! Each variant's A-staging temp-count proxy (the M3.12 r3 instrument, re-measured from the AS-EMITTED
//! module) is printed next to its TFLOPS so the proxy-vs-TFLOPS relationship is visible.
//!
//! Gated on AXC_ENABLE_GPU_BENCHES=1 + a responsive Vulkan ICD. Typed-skip on CoopMatUnsupported
//! (Lavapipe) / DeviceFeatureUnsupported / subgroup_size() != 32.

#![allow(dead_code)]

use axc_driver::q4km_oracle as common_q4km_f32ref;

use criterion::{criterion_group, criterion_main, Criterion};
use axc_driver::compile_source_with_assignments;
use axc_hir::ParamBindingPlan;
use axc_runtime::{
    VulkanContext, DispatchError, KernelHandle,
    ResidentBenchConfig, ResidentTimingSource,
};
use std::collections::{BTreeMap, BTreeSet};

const LEADER_SRC: &str =
    include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached.axc");
const TIGHTLIVE_SRC: &str =
    include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached_tightlive.axc");

const CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000: f64 = 125.0_f64;
const FROZEN_REL_TOL: f64 = 1e-3;
/// The >=1.15x PRONG-A gate (NEVER loosened).
const TIGHTLIVE_GATE: f64 = 1.15_f64;

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

/// Measure one prepared kernel at (m, n, k); returns (tflops, combined, raw, ns, timing_source) or
/// None on a typed-skip / grid / cap failure.
#[allow(clippy::too_many_arguments)]
fn measure(
    ctx: &VulkanContext,
    handle: &KernelHandle,
    plan: &ParamBindingPlan,
    label: &str,
    m: usize,
    n: usize,
    k: usize,
) -> Option<(f64, f64, f64, u64, ResidentTimingSource)> {
    if !m.is_multiple_of(32) || !n.is_multiple_of(32) || !k.is_multiple_of(256) {
        eprintln!("{label}: M={m}/N={n}/K={k} not a valid Q4_K_M shape — skip");
        return None;
    }
    let n_bpr = k / 256;
    let wg_x = (n / 32) as u64;
    let wg_y = (m / 32) as u64;
    let max_wg = ctx.max_compute_work_group_count();
    if wg_x > max_wg[0] as u64 || wg_y > max_wg[1] as u64 {
        eprintln!("{label}: grid ({wg_x},{wg_y}) exceeds limits at M={m} N={n} K={k} — skip");
        return None;
    }
    let total_bytes: u64 = (m as u64 * n_bpr as u64 * 144)
        + 2 * (k as u64) * (n as u64)
        + 4 * (m as u64) * (n as u64);
    if total_bytes >= FIFTY_GB {
        eprintln!("{label}: 50 GB cap exceeded at M={m} N={n} K={k} — skip");
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
            eprintln!("{label}: CoopMatUnsupported at dispatch (typed-skip): {reason}");
            return None;
        }
        Err(e) => {
            eprintln!("{label}: dispatch failed at M={m} N={n} K={k}: {e}");
            return None;
        }
    };
    let y_gpu: Vec<f32> = outputs[2].chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let y_ref = common_q4km_f32ref::q4km_dequant_matmul_f32accum_cpu(&q_bytes, &x_f16, m, n, n_bpr);
    let abs_scale = common_q4km_f32ref::q4km_abs_dot_scale(&q_bytes, &x_f16, m, n, n_bpr);
    let combined = common_q4km_f32ref::max_rel_diff_combined(&y_gpu, &y_ref, &abs_scale);
    let raw = common_q4km_f32ref::max_rel_diff(&y_gpu, &y_ref);

    let (min_ns, ts) = resident_min_of_n(
        ctx, handle,
        &[&q_bytes, &x_bytes, &vec![0u8; c_size]],
        &[0, 0, c_size as u64],
        workgroups,
        pc,
    );
    let tflops = effective_q4km_tflops(m, n, k, min_ns);
    let valid = if combined <= FROZEN_REL_TOL { "VALID" } else { "INVALID(>1e-3)" };
    eprintln!(
        "{label} = {tflops:.3} TFLOPS ({:.2}%) | {min_ns} ns ({}) | M={m} N={n} K={k} \
         combined={combined:.3e} [{valid}] raw={raw:.3e}",
        tflops / CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000 * 100.0,
        timing_source_label(ts),
    );
    Some((tflops, combined, raw, min_ns, ts))
}

/// The A-staging temp-count proxy from the AS-EMITTED module (AT-2804) — the M3.12 r3 instrument.
/// Counts result-producing instructions in the f32_to_f16→a_tile-store region.
fn a_staging_temp_count_proxy(src: &str) -> Option<usize> {
    let (bytes, _meta) = compile_source_with_assignments(src, &rb2x2_assignments()).ok()?;
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let insts = decode(&words);
    let ti = build_type_info(&insts);
    Some(a_staging_chain_inst_count(&insts, &ti))
}

// ── raw-word-stream harness (m312 proxy) ───────────────────────────────────────
#[derive(Clone)]
struct Inst { op: u32, operands: Vec<u32> }
fn decode(words: &[u32]) -> Vec<Inst> {
    let mut out = Vec::new();
    let mut i = 5usize;
    while i < words.len() {
        let head = words[i];
        let wc = (head >> 16) as usize;
        let op = head & 0xFFFF;
        if wc == 0 || i + wc > words.len() { break; }
        out.push(Inst { op, operands: words[i + 1..i + wc].to_vec() });
        i += wc;
    }
    out
}
struct TypeInfo { ptr_kind: BTreeMap<u32, bool> } // true = Workgroup (shared)
fn build_type_info(insts: &[Inst]) -> TypeInfo {
    use spirv::Op;
    let sc_wg = spirv::StorageClass::Workgroup as u32;
    let mut pointers: BTreeMap<u32, u32> = BTreeMap::new(); // ptr-type -> storage class
    let mut ptr_kind: BTreeMap<u32, bool> = BTreeMap::new();
    let op_type_pointer = Op::TypePointer as u32;
    let op_variable = Op::Variable as u32;
    let op_access_chain = Op::AccessChain as u32;
    for inst in insts {
        if inst.op == op_type_pointer && inst.operands.len() >= 3 {
            pointers.insert(inst.operands[0], inst.operands[1]);
        } else if inst.op == op_variable && inst.operands.len() >= 2 {
            let is_wg = pointers.get(&inst.operands[0]).copied() == Some(sc_wg);
            ptr_kind.insert(inst.operands[1], is_wg);
        } else if inst.op == op_access_chain && inst.operands.len() >= 3 {
            let from_type = pointers.get(&inst.operands[0]).copied() == Some(sc_wg);
            let from_base = ptr_kind.get(&inst.operands[2]).copied().unwrap_or(false);
            ptr_kind.insert(inst.operands[1], from_type || from_base);
        }
    }
    TypeInfo { ptr_kind }
}
fn a_staging_chain_inst_count(insts: &[Inst], ti: &TypeInfo) -> usize {
    use spirv::Op;
    let op_store = Op::Store as u32;
    let op_fconvert = Op::FConvert as u32;
    let op_label = Op::Label as u32;
    let op_loop_merge = Op::LoopMerge as u32;
    let op_load = Op::Load as u32;
    let mut tainted_ids: BTreeSet<u32> = BTreeSet::new();
    let mut tainted_ptrs: BTreeSet<u32> = BTreeSet::new();
    for inst in insts {
        if inst.op == op_fconvert {
            if let Some(&rid) = inst.operands.get(1) { tainted_ids.insert(rid); }
        } else if inst.op == op_store {
            if let (Some(&ptr), Some(&obj)) = (inst.operands.first(), inst.operands.get(1)) {
                if tainted_ids.contains(&obj) { tainted_ptrs.insert(ptr); }
            }
        } else if inst.op == op_load {
            if let (Some(&rid), Some(&ptr)) = (inst.operands.get(1), inst.operands.get(2)) {
                if tainted_ptrs.contains(&ptr) { tainted_ids.insert(rid); }
            }
        }
    }
    let mut a_store_idx: Option<usize> = None;
    for (idx, inst) in insts.iter().enumerate() {
        if inst.op != op_store { continue; }
        let (Some(&ptr), Some(&obj)) = (inst.operands.first(), inst.operands.get(1)) else { continue };
        if ti.ptr_kind.get(&ptr).copied() == Some(true) && tainted_ids.contains(&obj) {
            a_store_idx = Some(idx);
            break;
        }
    }
    let Some(a_store_idx) = a_store_idx else { return 0 };
    let mut last_lm: Option<usize> = None;
    for (j, inst) in insts.iter().enumerate().take(a_store_idx) {
        if inst.op == op_loop_merge { last_lm = Some(j); }
    }
    let mut body_start = 0usize;
    if let Some(lm) = last_lm {
        for (j, inst) in insts.iter().enumerate().skip(lm + 1).take(a_store_idx - lm) {
            if inst.op == op_label { body_start = j; break; }
        }
    }
    let slice = &insts[body_start..a_store_idx + 1];
    let non_result = |op: u32| -> bool {
        op == op_store || op == Op::Branch as u32 || op == Op::BranchConditional as u32
            || op == op_loop_merge || op == Op::SelectionMerge as u32 || op == op_label
    };
    slice.iter().filter(|i| !non_result(i.op)).count()
}

/// Prepare a kernel handle from a source; returns (handle, plan) or None on typed-skip.
fn prepare(ctx: &VulkanContext, src: &str, kernel_name: &str)
    -> Option<(KernelHandle, ParamBindingPlan)>
{
    let assignments = rb2x2_assignments();
    let (bytes, meta) = compile_source_with_assignments(src, &assignments)
        .unwrap_or_else(|e| panic!("{kernel_name}: compile failed: {e:?}"));
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let handle = match ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, meta.coopmat.as_ref(), kernel_name, meta.shared_memory_bytes,
    ) {
        Ok(h) => h,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("{kernel_name}: CoopMatUnsupported (typed-skip): {reason}");
            return None;
        }
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
            eprintln!("{kernel_name}: DeviceFeatureUnsupported {feature}/{kernel} (typed-skip)");
            return None;
        }
        Err(e) => { eprintln!("{kernel_name}: prepare failed: {e}"); return None; }
    };
    Some((handle, meta.binding_plan.clone()))
}

fn bench_resident_q4km_tightlive(c: &mut Criterion) {
    let bench_id = "dispatch_resident_q4km_matmul_rb_f32acc_cached_tightlive";
    if !gpu_benches_enabled() {
        eprintln!("{bench_id}: AXC_ENABLE_GPU_BENCHES not set; skipping");
        c.bench_function(bench_id, |b| b.iter(|| {}));
        return;
    }
    let ctx = match VulkanContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("{bench_id}: no Vulkan: {e}");
            c.bench_function(bench_id, |b| b.iter(|| {}));
            return;
        }
    };
    if !ctx.coopmat_support().feature_present || ctx.subgroup_size() != 32 {
        eprintln!("{bench_id}: coopmat/subgroup typed-skip on {}", ctx.physical_device_name());
        c.bench_function(bench_id, |b| b.iter(|| {}));
        return;
    }

    // AT-2804: the per-variant A-staging temp-count proxy (re-measured from the AS-EMITTED module).
    let leader_proxy = a_staging_temp_count_proxy(LEADER_SRC).unwrap_or(0);
    let tl_proxy = a_staging_temp_count_proxy(TIGHTLIVE_SRC).unwrap_or(0);
    let proxy_dropped = tl_proxy < leader_proxy;
    eprintln!(
        "AT-2804 proxy: M3.6 leader A-staging temp-count = {leader_proxy} (§3.1.40 = 86); \
         tightlive = {tl_proxy}; delta = {} ({})",
        leader_proxy as i64 - tl_proxy as i64,
        if proxy_dropped { "DROP" } else { "no-drop" },
    );

    let Some((leader_h, leader_plan)) =
        prepare(&ctx, LEADER_SRC, "q4km_matmul_rb_coopmat_f32acc_cached") else {
        c.bench_function(bench_id, |b| b.iter(|| {}));
        return;
    };
    let Some((tl_h, tl_plan)) =
        prepare(&ctx, TIGHTLIVE_SRC, "q4km_matmul_rb_coopmat_f32acc_cached_tightlive") else {
        c.bench_function(bench_id, |b| b.iter(|| {}));
        return;
    };

    // Measure BOTH at 768³ and at A/B; print the ratio + the >=1.15x gate verdict.
    for (m, n, k, tag) in [(768usize, 768usize, 768usize, "768^3"), (AB_M, AB_N, AB_K, "AB")] {
        let leader = measure(&ctx, &leader_h, &leader_plan, &format!("LEADER[{tag}]"), m, n, k);
        let tl = measure(&ctx, &tl_h, &tl_plan, &format!("TIGHTLIVE[{tag}]"), m, n, k);
        if let (Some((lt, _, _, _, lts)), Some((tt, tc, tr, tns, tts))) = (leader, tl) {
            let ratio = if lt > 0.0 { tt / lt } else { 0.0 };
            let both_gpu = lts == ResidentTimingSource::GpuTimestamp
                && tts == ResidentTimingSource::GpuTimestamp;
            let verdict = if both_gpu && ratio >= TIGHTLIVE_GATE {
                "GATE-MET(>=1.15x) — PRONG A WIN (tightlive could become the new leader)"
            } else if both_gpu {
                if proxy_dropped {
                    "HONEST-NEGATIVE-(b): proxy DROPPED but TFLOPS FLAT — register pressure is NOT a \
                     SOURCE-ADDRESSABLE lever (driver re-allocates from dataflow); NARROWLY closes \
                     that, does NOT falsify the register-pressure pinpoint"
                } else {
                    "HONEST-NEGATIVE-(a): proxy did NOT drop AND TFLOPS flat — no codegen leverage \
                     (AXIOM's naive SSA is source-order-insensitive)"
                }
            } else {
                "INCONCLUSIVE (non-GpuTimestamp timing — re-run on real NVIDIA)"
            };
            // Machine-readable line for scripts/m34_llamacpp_ab.sh --tightlive (ab_results_tightlive.json).
            println!(
                "AXC_Q4KM_TIGHTLIVE size={tag} leader_tflops={lt:.3} tightlive_tflops={tt:.3} \
                 ratio={ratio:.4} gate={TIGHTLIVE_GATE} leader_proxy={leader_proxy} \
                 tightlive_proxy={tl_proxy} proxy_dropped={proxy_dropped} tightlive_ns={tns} \
                 combined={tc:.6e} raw={tr:.6e} device={}",
                ctx.physical_device_name(),
            );
            eprintln!(
                "AT-2802 [{tag}]: tightlive {tt:.3} / leader {lt:.3} = {ratio:.4}x [{verdict}] \
                 (proxy: leader={leader_proxy} tightlive={tl_proxy})"
            );
        }
    }

    let mut group = c.benchmark_group("resident_q4km_matmul_rb_f32acc_cached_tightlive");
    group.sample_size(10);
    group.bench_function(bench_id, |b| {
        b.iter_custom(|iters| {
            let mut total_ns: u64 = 0;
            for _ in 0..iters {
                let n_bpr = 1usize;
                let q = make_q4km_weights(256, n_bpr, 0xC0FFEE ^ 256);
                let xb: Vec<u8> = make_x_f16(256, 256, 0xBADF00D ^ 256)
                    .iter().flat_map(|&b| b.to_le_bytes()).collect();
                let (ns_256, _) = resident_min_of_n(
                    &ctx, &tl_h,
                    &[&q, &xb, &vec![0u8; 256 * 256 * 4]],
                    &[0, 0, (256 * 256 * 4) as u64],
                    (8, 8, 1),
                    assemble_pc(&tl_plan, 256, 256, 256, n_bpr as u32),
                );
                total_ns += ns_256;
            }
            std::time::Duration::from_nanos(total_ns)
        });
    });
    group.finish();
}

criterion_group!(resident_q4km_tightlive_benches, bench_resident_q4km_tightlive);
criterion_main!(resident_q4km_tightlive_benches);
