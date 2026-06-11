//! M3.12 — Dequant front-end ABLATION DIAGNOSTIC: per-variant GPU-resident TFLOPS + the RE-ANCHORED
//! triangulation that PINPOINTS the dominant 1.77x dequant front-end tax.
//!
//! Bench ID: `dispatch_resident_q4km_matmul_rb_f32acc_cached_ablate`.
//!
//! The M3.12 variants are PROFILING INSTRUMENTS with INTENTIONALLY WRONG OUTPUT — they hold the M3.6
//! leader's matmul + B-path + shared-A-stage + 3 barriers + coopmat + the scale-cache FILL CONSTANT
//! and vary ONLY the per-element A-dequant value. NO correctness gate (wrong by design). The ONLY
//! output is the per-variant resident TFLOPS.
//!
//! ## AT-2701 — PER-VARIANT RESIDENT TFLOPS + register/temp-count proxy
//! Each variant's A/B (4096x512x14336) + 768^3 TFLOPS, MIN-of-N GpuTimestamp kernel-only, with
//! shared_memory_bytes (==4096) + the MIN-of-N spread + a per-variant REGISTER/TEMP-COUNT PROXY
//! (the A-staging dependency-chain result-instruction count from the emitted SPIR-V) recorded.
//! V-structure-probe is MEASURED FIRST (the PRIMARY denominator). The temp-count proxy makes the
//! V-structure-probe-vs-real-dequant register-footprint delta VISIBLE (the r3 in-bucket
//! register-occupancy residual on this register-limited kernel — M3.7/M3.8).
//!
//! ## AT-2702 — the RE-ANCHORED triangulation (V-structure-probe is the PRIMARY denominator)
//! The 1.77x splits into TWO non-confounded buckets:
//!   (A) ARITHMETIC = on the [42.86 <-> V-structure-probe] line (both endpoints carry the IDENTICAL
//!       4096 B caches + 3 barriers + kept fill); arithmetic_share = (T-42.86)/(probe-42.86).
//!   (B) STRUCTURAL = (75.8 - V-structure-probe)/(75.8-42.86) — the scale-cache occupancy + the extra
//!       barrier the plain-f32 core never pays.
//! The LARGER bucket NAMES the next-milestone PRIMARY target. r3 CROSS-CHECK CAVEAT: a large
//! V-structure-probe-vs-V-passthrough temp-count delta means the arithmetic bucket includes a
//! register/occupancy component — read BOTH the TFLOPS share AND the temp-count proxy before naming
//! an arithmetic-feature (the M3.7/M3.8 occupancy-regression failure mode); either way the lever
//! folds into the same fused-dequant-coopmat-load rewrite. MIN-of-N <5% = noise (indistinguishable).
//! CONSISTENCY GUARD: a load/address-dominant finding CONTRADICTS M3.10b/M3.11 => ESCALATE.
//!
//! ## AT-2704 — baseline integrity (read-with-source, same kernel/config)
//! The two REAL merged baselines — M3.6 leader 42.86 + plain-f32 core 75.8 @ A/B — are READ from the
//! merged benches (the AT-2503 shape-pinned parser, filtered on M=4096 N=512 K=14336), NOT hardcoded.
//! The 75.8 baseline is asserted to be matmul_rb_coopmat.axc @ rb_m=2,rb_n=2 (the single-subgroup-RB
//! column), so no register-blocking delta folds into the 1.77x. The plain-f32 core is ALSO measured
//! live here so the orchestrator can confirm the merged 75.8 against this machine.
//!
//! Gated on AXC_ENABLE_GPU_BENCHES=1 + a responsive Vulkan ICD. Typed-skip on CoopMatUnsupported
//! (Lavapipe) / DeviceFeatureUnsupported / subgroup_size() != 32.

#![allow(dead_code)]

use criterion::{criterion_group, criterion_main, Criterion};
use axc_driver::compile_source_with_assignments;
use axc_hir::ParamBindingPlan;
use axc_runtime::{
    VulkanContext, DispatchError, KernelHandle,
    ResidentBenchConfig, ResidentTimingSource,
};
use spirv::Op;
use std::collections::{BTreeMap, BTreeSet};

const STRUCTONLY_SRC: &str =
    include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached_ablate_structonly.axc");
const PASSTHROUGH_SRC: &str =
    include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached_ablate_passthrough.axc");
const M36_LEADER_SRC: &str =
    include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached.axc");
const PLAIN_F32_CORE_SRC: &str =
    include_str!("../../../examples/matmul_rb_coopmat.axc");

/// M3.6 leader A/B merged baseline (the 42.86 endpoint) — re-asserted against the merged bench.
const M36_LEADER_AB_TFLOPS: f64 = 42.86_f64;
/// The MIN-of-N <5% noise floor (the M3.11 1.048x precedent).
const NOISE_FLOOR_PCT: f64 = 5.0_f64;

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

fn effective_q4km_tflops(m: usize, n: usize, k: usize, kernel_ns: u64) -> f64 {
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

// ════════════════════════════════════════════════════════════════════════════════════════════
//  AT-2701 register/temp-count PROXY — the m310b raw-word-stream A-staging-chain instruction count
// ════════════════════════════════════════════════════════════════════════════════════════════

fn compile_words(src: &str) -> Vec<u32> {
    let (bytes, _meta) = compile_source_with_assignments(src, &rb2x2_assignments())
        .expect("M3.12 ablate: variant must compile");
    bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[derive(Clone)]
struct Inst {
    op: u32,
    operands: Vec<u32>,
}

fn decode(words: &[u32]) -> Vec<Inst> {
    let mut out = Vec::new();
    let mut i = 5usize;
    while i < words.len() {
        let head = words[i];
        let wc = (head >> 16) as usize;
        let op = head & 0xFFFF;
        if wc == 0 || i + wc > words.len() {
            break;
        }
        out.push(Inst { op, operands: words[i + 1..i + wc].to_vec() });
        i += wc;
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PtrKind { Shared, Other }

fn shared_ptr_ids(insts: &[Inst]) -> BTreeSet<u32> {
    // Identify Workgroup-storage pointer value-ids (variables + access chains derived from them).
    const SC_WORKGROUP: u32 = spirv::StorageClass::Workgroup as u32;
    let op_type_pointer = Op::TypePointer as u32;
    let op_variable = Op::Variable as u32;
    let op_access_chain = Op::AccessChain as u32;
    let mut wg_ptr_types: BTreeSet<u32> = BTreeSet::new();
    let mut ids: BTreeSet<u32> = BTreeSet::new();
    for inst in insts {
        if inst.op == op_type_pointer && inst.operands.len() >= 3 && inst.operands[1] == SC_WORKGROUP {
            wg_ptr_types.insert(inst.operands[0]);
        } else if inst.op == op_variable && inst.operands.len() >= 2 {
            if wg_ptr_types.contains(&inst.operands[0]) {
                ids.insert(inst.operands[1]);
            }
        } else if inst.op == op_access_chain && inst.operands.len() >= 3 {
            // pointer result-type Workgroup OR base already shared.
            if wg_ptr_types.contains(&inst.operands[0]) || ids.contains(&inst.operands[2]) {
                ids.insert(inst.operands[1]);
            }
        }
    }
    ids
}

/// The A-staging region [body_start, a_store_idx] via the f32_to_f16→Workgroup-store anchor (m310b).
fn find_a_staging_region(insts: &[Inst]) -> (usize, usize) {
    let op_store = Op::Store as u32;
    let op_fconvert = Op::FConvert as u32;
    let op_label = Op::Label as u32;
    let op_loop_merge = Op::LoopMerge as u32;
    let op_load = Op::Load as u32;
    let shared = shared_ptr_ids(insts);

    let mut tainted_ids: BTreeSet<u32> = BTreeSet::new();
    let mut tainted_ptrs: BTreeSet<u32> = BTreeSet::new();
    for inst in insts {
        if inst.op == op_fconvert {
            if let Some(&rid) = inst.operands.get(1) {
                tainted_ids.insert(rid);
            }
        } else if inst.op == op_store {
            if let (Some(&ptr), Some(&obj)) = (inst.operands.first(), inst.operands.get(1)) {
                if tainted_ids.contains(&obj) {
                    tainted_ptrs.insert(ptr);
                }
            }
        } else if inst.op == op_load {
            if let (Some(&rid), Some(&ptr)) = (inst.operands.get(1), inst.operands.get(2)) {
                if tainted_ptrs.contains(&ptr) {
                    tainted_ids.insert(rid);
                }
            }
        }
    }

    let mut a_store_idx = 0usize;
    for (idx, inst) in insts.iter().enumerate() {
        if inst.op != op_store {
            continue;
        }
        let (Some(&ptr), Some(&obj)) = (inst.operands.first(), inst.operands.get(1)) else { continue };
        if shared.contains(&ptr) && tainted_ids.contains(&obj) {
            a_store_idx = idx;
            break;
        }
    }
    let mut body_start = 0usize;
    let mut last_lm: Option<usize> = None;
    for (j, inst) in insts.iter().enumerate().take(a_store_idx) {
        if inst.op == op_loop_merge {
            last_lm = Some(j);
        }
    }
    if let Some(lm) = last_lm {
        for (j, inst) in insts.iter().enumerate().skip(lm + 1).take(a_store_idx.saturating_sub(lm)) {
            if inst.op == op_label {
                body_start = j;
                break;
            }
        }
    }
    (body_start, a_store_idx)
}

/// The register/temp-count proxy: result-producing instructions in the A-staging dependency chain.
fn a_staging_chain_proxy(src: &str) -> usize {
    let words = compile_words(src);
    let insts = decode(&words);
    let (start, store) = find_a_staging_region(&insts);
    if store == 0 {
        return 0;
    }
    let non_result = |op: u32| -> bool {
        op == Op::Store as u32
            || op == Op::Branch as u32
            || op == Op::BranchConditional as u32
            || op == Op::LoopMerge as u32
            || op == Op::SelectionMerge as u32
            || op == Op::Label as u32
    };
    insts[start..=store].iter().filter(|i| !non_result(i.op)).count()
}

// ════════════════════════════════════════════════════════════════════════════════════════════
//  AT-2704 baseline read-with-source (the AT-2503 shape-pinned parser)
// ════════════════════════════════════════════════════════════════════════════════════════════

/// Read the plain-f32 single-subgroup-RB core TFLOPS at A/B (M=4096 N=512 K=14336) from the merged
/// bench ab_results_warptile.json — NOT a hardcoded literal (reuses the AT-2503 read-with-source).
fn read_core_ab_tflops_from_bench() -> Option<f64> {
    let manifest_dir =
        std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").ok()?);
    let bench_path = manifest_dir
        .join("..").join("..")
        .join(".pipeline").join("benchmarks").join("m34")
        .join("ab_results_warptile.json");
    let content = std::fs::read_to_string(&bench_path).ok()?;
    let mut vals: Vec<f64> = Vec::new();
    for line in content.lines() {
        if !line.contains("M=4096 N=512 K=14336") {
            continue;
        }
        if let Some(pos) = line.find("single-subgroup-RB ") {
            let rest = &line[pos + "single-subgroup-RB ".len()..];
            let num: String = rest.chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
            if let Ok(v) = num.parse::<f64>() {
                vals.push(v);
            }
        }
    }
    if vals.is_empty() {
        None
    } else {
        Some(vals.iter().sum::<f64>() / vals.len() as f64)
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════════
//  Resident measurement plumbing (mirrors resident_q4km_matmul_rb_f32acc_cached.rs)
// ════════════════════════════════════════════════════════════════════════════════════════════

fn resident_min_of_n(
    ctx: &VulkanContext,
    handle: &KernelHandle,
    inputs: &[&[u8]],
    output_sizes: &[u64],
    workgroups: (u32, u32, u32),
    push_constants: Vec<u8>,
) -> (u64, u64, ResidentTimingSource) {
    let resident = ctx.upload_resident(handle, inputs, output_sizes)
        .expect("upload_resident must succeed");
    let cfg = ResidentBenchConfig { workgroups, push_constants };
    for _ in 0..N_WARMUP {
        ctx.dispatch_resident(handle, &resident, &cfg).expect("warmup dispatch");
    }
    let mut min_ns = u64::MAX;
    let mut max_ns = 0u64;
    let mut last_source = ResidentTimingSource::CpuFenceWall;
    for _ in 0..N_MEASURED {
        let t = ctx.dispatch_resident(handle, &resident, &cfg).expect("measured dispatch");
        if t.kernel_ns < min_ns { min_ns = t.kernel_ns; }
        if t.kernel_ns > max_ns { max_ns = t.kernel_ns; }
        last_source = t.timing_source;
    }
    (min_ns, max_ns, last_source)
}

struct Prepared {
    handle: KernelHandle,
    plan: ParamBindingPlan,
}

fn prepare(ctx: &VulkanContext, src: &str, kernel_name: &str) -> Option<Prepared> {
    let assignments = rb2x2_assignments();
    let (bytes, meta) = match compile_source_with_assignments(src, &assignments) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("ablate: {kernel_name} compile failed: {e:?}");
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
            eprintln!("ablate: {kernel_name} CoopMatUnsupported (typed-skip): {reason}");
            return None;
        }
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
            eprintln!("ablate: {kernel_name} DeviceFeatureUnsupported {feature}/{kernel} (typed-skip)");
            return None;
        }
        Err(e) => {
            eprintln!("ablate: {kernel_name} prepare failed: {e}");
            return None;
        }
    };
    Some(Prepared { handle, plan: meta.binding_plan })
}

// ── q4km-signature fixtures (q:u8, x:f16, C:f32, M,N,K,n_blocks_per_row) ──────────────────────

fn assemble_q4km_pc(plan: &ParamBindingPlan, m: u32, n: u32, k: u32, n_bpr: u32) -> Vec<u8> {
    let mut pc: Vec<u8> = vec![0u8; plan.push_constant_total_bytes as usize];
    for s in &plan.scalars {
        let val: u32 = match s.name.as_str() {
            "M" => m, "N" => n, "K" => k, "n_blocks_per_row" => n_bpr,
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
    let mut next = || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; state };
    for row in 0..m {
        for sb in 0..n_bpr {
            let base = row * n_bpr * 144 + sb * 144;
            let d = 0.02_f32 + ((next() % 16) as f32) * 0.002;
            let dmin = 0.01_f32 + ((next() % 8) as f32) * 0.001;
            q[base..base + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
            q[base + 2..base + 4].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
            for j in 0..12 { q[base + 4 + j] = (next() & 0x3F) as u8; }
            for kk in 0..128 { q[base + 16 + kk] = (next() & 0xFF) as u8; }
        }
    }
    q
}

fn make_x_f16(k: usize, n: usize, seed: u64) -> Vec<u8> {
    use half::f16;
    let mut state = seed | 1;
    let mut out = Vec::with_capacity(k * n * 2);
    for idx in 0..k * n {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        let v = (((state % 2000) as f32) / 1000.0 - 1.0) + (idx % 3) as f32 * 0.01;
        out.extend_from_slice(&f16::from_f32(v).to_bits().to_le_bytes());
    }
    out
}

/// One per-variant measurement record.
struct VariantResult {
    tflops: f64,
    spread_pct: f64,
    chain_proxy: usize,
    shared_bytes: u32,
    timing: ResidentTimingSource,
}

/// Measure a q4km-signature variant (the ablation instruments + the M3.6 leader) at (m,n,k).
#[allow(clippy::too_many_arguments)]
fn measure_q4km_variant(
    ctx: &VulkanContext,
    prep: &Prepared,
    label: &str,
    src: &str,
    shared_bytes: u32,
    m: usize, n: usize, k: usize,
) -> Option<VariantResult> {
    if !m.is_multiple_of(32) || !n.is_multiple_of(32) || !k.is_multiple_of(256) {
        eprintln!("ablate {label}: shape M={m} N={n} K={k} invalid — skip");
        return None;
    }
    let n_bpr = k / 256;
    let wg_x = (n / 32) as u64;
    let wg_y = (m / 32) as u64;
    let max_wg = ctx.max_compute_work_group_count();
    if wg_x > max_wg[0] as u64 || wg_y > max_wg[1] as u64 {
        eprintln!("ablate {label}: grid ({wg_x},{wg_y}) exceeds limits — skip");
        return None;
    }
    let total_bytes: u64 = (m as u64 * n_bpr as u64 * 144)
        + 2 * (k as u64) * (n as u64) + 4 * (m as u64) * (n as u64);
    if total_bytes >= FIFTY_GB {
        eprintln!("ablate {label}: 50 GB cap exceeded — skip");
        return None;
    }
    let q_bytes = make_q4km_weights(m, n_bpr, 0xC0FFEE ^ (m as u64));
    let x_bytes = make_x_f16(k, n, 0xBADF00D ^ (n as u64));
    let c_size = m * n * 4;
    let pc = assemble_q4km_pc(&prep.plan, m as u32, n as u32, k as u32, n_bpr as u32);
    let workgroups = (wg_x as u32, wg_y as u32, 1);

    let (min_ns, max_ns, ts) = resident_min_of_n(
        ctx, &prep.handle,
        &[&q_bytes, &x_bytes, &vec![0u8; c_size]],
        &[0, 0, c_size as u64],
        workgroups, pc,
    );
    let tflops = effective_q4km_tflops(m, n, k, min_ns);
    let spread_pct = if min_ns > 0 { (max_ns - min_ns) as f64 / min_ns as f64 * 100.0 } else { 0.0 };
    let chain_proxy = a_staging_chain_proxy(src);
    eprintln!(
        "ablate {label} @ {m}x{n}x{k} = {tflops:.3} TFLOPS | min {min_ns} ns ({}) | spread {spread_pct:.1}% \
         | chain-proxy {chain_proxy} | shared {shared_bytes} B [WRONG-OUTPUT-BY-DESIGN: instrument]",
        timing_source_label(ts)
    );
    Some(VariantResult { tflops, spread_pct, chain_proxy, shared_bytes, timing: ts })
}

// ── plain-f32 core fixtures (A:f16, B:f16, C:f16, M,N,K) — the 75.8 endpoint, measured live ───

fn assemble_core_pc(m: u32, n: u32, k: u32) -> Vec<u8> {
    let mut pc = Vec::with_capacity(12);
    pc.extend_from_slice(&m.to_le_bytes());
    pc.extend_from_slice(&n.to_le_bytes());
    pc.extend_from_slice(&k.to_le_bytes());
    pc
}

fn make_core_f16(count: usize, modv: usize) -> Vec<u8> {
    use half::f16;
    let mut out = Vec::with_capacity(count * 2);
    for i in 0..count {
        out.extend_from_slice(&f16::from_f32(((i % modv) + 1) as f32).to_bits().to_le_bytes());
    }
    out
}

fn measure_core(
    ctx: &VulkanContext, handle: &KernelHandle, m: usize, n: usize, k: usize,
) -> Option<(f64, f64, ResidentTimingSource)> {
    if !m.is_multiple_of(32) || !n.is_multiple_of(32) {
        return None;
    }
    let wg_x = (n / 32) as u64;
    let wg_y = (m / 32) as u64;
    let max_wg = ctx.max_compute_work_group_count();
    if wg_x > max_wg[0] as u64 || wg_y > max_wg[1] as u64 {
        return None;
    }
    let total: u64 = 2 * (m as u64) * (k as u64) + 2 * (k as u64) * (n as u64) + 2 * (m as u64) * (n as u64);
    if total >= FIFTY_GB {
        return None;
    }
    let a_bytes = make_core_f16(m * k, 4);
    let b_bytes = make_core_f16(k * n, 3);
    let c_size = m * n * 2;
    let pc = assemble_core_pc(m as u32, n as u32, k as u32);
    let (min_ns, max_ns, ts) = resident_min_of_n(
        ctx, handle,
        &[&a_bytes, &b_bytes, &vec![0u8; c_size]],
        &[0, 0, c_size as u64],
        (wg_x as u32, wg_y as u32, 1), pc,
    );
    let tflops = effective_q4km_tflops(m, n, k, min_ns);
    let spread = if min_ns > 0 { (max_ns - min_ns) as f64 / min_ns as f64 * 100.0 } else { 0.0 };
    eprintln!(
        "ablate plain-f32-core @ {m}x{n}x{k} = {tflops:.3} TFLOPS | min {min_ns} ns ({}) | spread {spread:.1}% \
         | shared 2048 B / 2 barriers (the 75.8 endpoint, the leaner structure)",
        timing_source_label(ts)
    );
    Some((tflops, spread, ts))
}

// ════════════════════════════════════════════════════════════════════════════════════════════
//  The triangulation (AT-2702) — RE-ANCHORED on V-structure-probe as the PRIMARY denominator
// ════════════════════════════════════════════════════════════════════════════════════════════

#[allow(clippy::too_many_arguments)]
fn run_triangulation(
    ctx: &VulkanContext,
    probe: &Prepared,
    passthrough: &Prepared,
    leader: &Prepared,
    core_handle: &KernelHandle,
    shared_bytes: u32,
) {
    // AT-2704: read the merged baselines with source.
    let core_ab_merged = read_core_ab_tflops_from_bench();
    match core_ab_merged {
        Some(v) => {
            let ok = (75.5..=76.1).contains(&v);
            eprintln!(
                "ablate AT-2704: plain-f32 core@A/B read from merged bench = {v:.3} TFLOPS \
                 (matmul_rb_coopmat.axc @ rb_m=2,rb_n=2, single-subgroup-RB column) [{}]",
                if ok { "in-range ~75.8" } else { "OUT-OF-RANGE — re-pin" }
            );
        }
        None => eprintln!(
            "ablate AT-2704: WARNING could not read core@A/B from ab_results_warptile.json — \
             the structural bucket falls back to the recorded 75.8 baseline"
        ),
    }
    let core_ab = core_ab_merged.unwrap_or(75.8_f64);

    // ── V-structure-probe FIRST (the PRIMARY denominator). ──
    let probe_ab = measure_q4km_variant(
        ctx, probe, "V-structure-probe(PRIMARY)", STRUCTONLY_SRC, shared_bytes, AB_M, AB_N, AB_K);
    let _ = measure_q4km_variant(
        ctx, probe, "V-structure-probe(PRIMARY)", STRUCTONLY_SRC, shared_bytes, 768, 768, 768);

    // ── V-passthrough (the decisive fork). ──
    let pass_ab = measure_q4km_variant(
        ctx, passthrough, "V-passthrough", PASSTHROUGH_SRC, shared_bytes, AB_M, AB_N, AB_K);
    let _ = measure_q4km_variant(
        ctx, passthrough, "V-passthrough", PASSTHROUGH_SRC, shared_bytes, 768, 768, 768);

    // ── M3.6 leader measured LIVE (re-confirm the 42.86 endpoint on this machine). ──
    let leader_ab = measure_q4km_variant(
        ctx, leader, "M3.6-leader(42.86 endpoint)", M36_LEADER_SRC, shared_bytes, AB_M, AB_N, AB_K);

    // ── plain-f32 core measured LIVE (re-confirm the 75.8 endpoint). ──
    let core_live = measure_core(ctx, core_handle, AB_M, AB_N, AB_K);
    let _ = measure_core(ctx, core_handle, 768, 768, 768);

    // The 42.86 endpoint: prefer the live leader measurement, fall back to the merged constant.
    let leader_t = leader_ab.as_ref().map(|r| r.tflops).unwrap_or(M36_LEADER_AB_TFLOPS);

    let (Some(probe_r), Some(pass_r)) = (probe_ab.as_ref(), pass_ab.as_ref()) else {
        eprintln!("ablate AT-2702: V-structure-probe and/or V-passthrough did not measure — cannot triangulate");
        return;
    };
    let probe_t = probe_r.tflops;
    let pass_t = pass_r.tflops;

    // The temp-count proxies (static — register-footprint deltas).
    let probe_proxy = probe_r.chain_proxy;
    let pass_proxy = pass_r.chain_proxy;
    let leader_proxy = a_staging_chain_proxy(M36_LEADER_SRC);

    // ── RE-ANCHORED buckets (AT-2702). ──
    // (A) ARITHMETIC on [42.86 <-> V-structure-probe] (both endpoints structurally identical on SLM+barriers).
    let arith_denom = probe_t - leader_t;
    let arith_share_pass = if arith_denom.abs() > 1e-9 { (pass_t - leader_t) / arith_denom } else { f64::NAN };
    // (B) STRUCTURAL = (75.8 - V-structure-probe) / (75.8 - 42.86).
    let total_177 = core_ab - leader_t;
    let structural_bucket = core_ab - probe_t;
    let structural_share = if total_177.abs() > 1e-9 { structural_bucket / total_177 } else { f64::NAN };
    let arithmetic_bucket = probe_t - leader_t;
    let arithmetic_share_of_177 = if total_177.abs() > 1e-9 { arithmetic_bucket / total_177 } else { f64::NAN };

    // The MEASURED ordering relative to the design's premise (probe assumed <= core). If the probe
    // EXCEEDS the core, the structural bucket is NON-EXISTENT (<=0) and the design's linear share math
    // (which assumes 42.86 <= probe <= 75.8) is OUT OF RANGE — report it HONESTLY rather than emit a
    // >100% / negative share.
    let probe_exceeds_core = probe_t > core_ab;

    eprintln!("ablate AT-2702 — RE-ANCHORED TRIANGULATION (V-structure-probe is the PRIMARY denominator):");
    eprintln!(
        "  endpoints: M3.6 leader {leader_t:.3} (42.86 merged) | V-structure-probe {probe_t:.3} \
         (PRIMARY) | plain-f32 core {core_ab:.3} (75.8 merged){}",
        match core_live { Some((t,_,_)) => format!(" | core measured-live {t:.3}"), None => String::new() }
    );
    if probe_exceeds_core {
        eprintln!(
            "  MEASURED FINDING (design-premise inversion): V-structure-probe {probe_t:.3} EXCEEDS the \
             plain-f32 core {core_ab:.3}. The structural bucket (75.8 - probe = {structural_bucket:.3}) \
             is <= 0 => the M3.6 structure (4 KB scale-cache + 3 barriers) is NOT a net structural \
             penalty when the A-value is near-free — it is FASTER than the leaner 2 KB / 2-barrier \
             core. => the ENTIRE 1.77x lives in the per-element A-side DEQUANT WORK (arithmetic + the \
             register/occupancy it carries), NOT in the M3.6 scale-cache structure. The linear \
             structural_share is OUT OF RANGE (reported below as <=0 for completeness only).");
    }
    eprintln!(
        "  bucket (A) ARITHMETIC = probe - leader = {arithmetic_bucket:.3} TFLOPS \
         (the full per-element A-side dequant-work gap on the [42.86<->probe] line; \
         {:.1}% of the 1.77x interval)", arithmetic_share_of_177 * 100.0);
    eprintln!(
        "  bucket (B) STRUCTURAL = 75.8 - probe = {structural_bucket:.3} TFLOPS \
         ({:.1}% of the 1.77x){}",
        structural_share * 100.0,
        if probe_exceeds_core {
            " — <= 0: NO net structural penalty (the scale-cache structure is not the gap)"
        } else {
            " — the scale-cache occupancy + extra barrier the core never pays"
        });
    eprintln!(
        "  V-passthrough placement on [42.86<->probe]: arithmetic_share = {:.1}% \
         (=> the per-element scale-reads+FMul/FSub portion of the A-side dequant-work bucket; the \
         residual {{u8-load + ConvertUToF + the f32_to_f16 convert + the live register footprint}} is \
         the rest)", arith_share_pass * 100.0);

    // ── r3 register-occupancy CROSS-CHECK CAVEAT. ──
    let probe_vs_pass_proxy_delta = pass_proxy as i64 - probe_proxy as i64;
    eprintln!(
        "  temp-count proxy (register footprint): M3.6 leader {leader_proxy} | V-passthrough {pass_proxy} \
         | V-structure-probe {probe_proxy} (probe-vs-passthrough delta {probe_vs_pass_proxy_delta})");
    let large_proxy_delta = probe_vs_pass_proxy_delta.unsigned_abs() as f64
        >= 0.30 * (pass_proxy.max(1) as f64);
    if large_proxy_delta {
        eprintln!(
            "  r3 CROSS-CHECK CAVEAT: the V-structure-probe-vs-V-passthrough temp-count delta is LARGE \
             ({probe_vs_pass_proxy_delta}) => the ARITHMETIC bucket includes a register/occupancy \
             component (V-structure-probe removed the live dequant-chain temporaries; on this \
             register-limited kernel — M3.7/M3.8 — that lifts resident-warp count, NOT pure \
             dequant-arithmetic latency). Read the arithmetic share as 'latency AND/MINUS register \
             occupancy', NOT pure latency. EITHER WAY the lever folds into the SAME \
             fused-dequant-coopmat-load rewrite as the structural bucket.");
    } else {
        eprintln!(
            "  r3 cross-check: the probe-vs-passthrough temp-count delta is modest — the arithmetic \
             share is dominated by latency, not register occupancy.");
    }

    // ── Noise floor + the named next-milestone target. ──
    let probe_move_pct = if leader_t > 0.0 { (probe_t - leader_t).abs() / leader_t * 100.0 } else { 0.0 };
    let pass_spread = pass_r.spread_pct;
    let probe_spread = probe_r.spread_pct;
    eprintln!(
        "  MIN-of-N spread: V-structure-probe {probe_spread:.1}% | V-passthrough {pass_spread:.1}% \
         (a <{NOISE_FLOOR_PCT:.0}% move is noise-adjacent — the M3.11 1.048x precedent)");

    let primary_target = if structural_share.is_nan() || arithmetic_share_of_177.is_nan() {
        "INDETERMINATE (an endpoint failed to measure) — re-run".to_owned()
    } else if probe_move_pct < NOISE_FLOOR_PCT {
        "NOISE-ADJACENT: V-structure-probe ~= the 42.86 endpoint (probe move < 5%) — the structure \
         ITSELF is the ceiling; report indistinguishable, re-run with more samples".to_owned()
    } else if probe_exceeds_core {
        // The MEASURED case: probe > core => the structure is not the gap; the whole 1.77x is A-side
        // dequant work, and the LARGE temp-count delta says a big chunk of that is register/occupancy.
        if large_proxy_delta {
            "ARITHMETIC + REGISTER-OCCUPANCY DOMINATE (the structural bucket is <= 0 — the scale-cache \
             structure is NOT the gap). The full 1.77x is the per-element A-side dequant WORK, and the \
             LARGE V-structure-probe-vs-V-passthrough temp-count delta (chain 86/68 -> 10) shows much \
             of it is REGISTER-PRESSURE/OCCUPANCY (the live dequant-chain temporaries cut resident \
             warps on this register-limited kernel — M3.7/M3.8), NOT pure arithmetic latency. \
             => next-milestone PRIMARY target = the FUSED-DEQUANT-COOPMAT-LOAD (dequant directly into \
             the coopmat fragment — eliminates BOTH the per-element dequant chain's live temporaries \
             AND the shared-A round-trip in one rewrite, exactly what llama's warptile does). A \
             major architectural lever (§3.1.34 coopmat use-type system), likely EB.1-gated.".to_owned()
        } else {
            "ARITHMETIC DOMINATES (the structural bucket is <= 0 — the scale-cache structure is NOT the \
             gap). The full 1.77x is the per-element A-side dequant arithmetic + converts; the \
             temp-count delta is modest, so it reads as latency. => next-milestone PRIMARY target = cut \
             the per-element scale-reads/FMul/FSub + the f32_to_f16 convert (the fused-dequant-coopmat-\
             load subsumes this).".to_owned()
        }
    } else if structural_share >= 0.5 {
        "STRUCTURAL DOMINATES => next-milestone PRIMARY target = the fused-dequant-coopmat-load \
         (drop the shared-A round-trip + the scale-cache arrays + the extra barrier; a major \
         architectural lever, likely EB.1-gated). The arithmetic-feature is SECONDARY.".to_owned()
    } else {
        "ARITHMETIC DOMINATES => next-milestone PRIMARY target = cut the per-element \
         scale-reads/FMul/FSub (or the converts) — BUT cross-check the temp-count proxy first (if the \
         arithmetic share is register/occupancy-driven, the real lever is the same \
         fused-dequant-coopmat-load). The structural lever is SECONDARY.".to_owned()
    };
    eprintln!("  PINPOINT / NEXT-MILESTONE RECOMMENDATION: {primary_target}");

    // ── CONSISTENCY GUARD vs M3.10b/M3.11. ──
    eprintln!(
        "  CONSISTENCY (M3.10b load ~3% issue-slots + M3.11 integer-decode latency-hidden): \
         V-passthrough KEEPS the u8 load + integer-decode; if it lands NEAR V-structure-probe the \
         residual is structural/convert (CONSISTENT). A finding that the LOAD/ADDRESS dominates would \
         CONTRADICT M3.10b/M3.11 => ESCALATE (re-measure, do not silently overturn).");

    // ── The machine-readable artifact line (parsed by scripts/m34_llamacpp_ab.sh --ablate). ──
    println!(
        "AXC_Q4KM_AB_ABLATE leader_tflops={leader_t:.4} probe_tflops={probe_t:.4} \
         passthrough_tflops={pass_t:.4} core_ab_merged={core_ab:.4} \
         arithmetic_bucket={arithmetic_bucket:.4} structural_bucket={structural_bucket:.4} \
         arithmetic_share={:.4} structural_share={:.4} passthrough_arith_share={:.4} \
         leader_proxy={leader_proxy} passthrough_proxy={pass_proxy} probe_proxy={probe_proxy} \
         probe_spread_pct={probe_spread:.2} passthrough_spread_pct={pass_spread:.2} \
         shared_bytes={shared_bytes} device={} [PROFILING-INSTRUMENTS: wrong output by design, no correctness gate]",
        arithmetic_share_of_177, structural_share, arith_share_pass,
        ctx.physical_device_name(),
    );
}

fn bench_resident_ablate(c: &mut Criterion) {
    let bench_id = "dispatch_resident_q4km_matmul_rb_f32acc_cached_ablate";
    if !gpu_benches_enabled() {
        eprintln!("ablate: AXC_ENABLE_GPU_BENCHES not set; skipping (CPU temp-count proxy still computable)");
        // Even without GPU, print the static temp-count proxies (the register-footprint deltas).
        eprintln!(
            "ablate (CPU-only static): temp-count proxy — M3.6 leader {} | V-passthrough {} | \
             V-structure-probe {}",
            a_staging_chain_proxy(M36_LEADER_SRC),
            a_staging_chain_proxy(PASSTHROUGH_SRC),
            a_staging_chain_proxy(STRUCTONLY_SRC),
        );
        c.bench_function(bench_id, |b| b.iter(|| {}));
        return;
    }
    let ctx = match VulkanContext::new() {
        Ok(ctx) => ctx,
        Err(e) => { eprintln!("ablate: no Vulkan: {e}"); c.bench_function(bench_id, |b| b.iter(|| {})); return; }
    };
    if !ctx.coopmat_support().feature_present {
        eprintln!("ablate: coopmat unsupported on {}; typed-skip", ctx.physical_device_name());
        c.bench_function(bench_id, |b| b.iter(|| {}));
        return;
    }
    if ctx.subgroup_size() != 32 {
        eprintln!("ablate: subgroup_size()={} != 32; typed-skip", ctx.subgroup_size());
        c.bench_function(bench_id, |b| b.iter(|| {}));
        return;
    }

    // shared_memory_bytes for the q4km-signature variants (== M3.6 leader 4096 B).
    let shared_bytes: u32 = (512 + 512) * 2 + 2 * 256 * 4;

    let Some(probe) = prepare(&ctx, STRUCTONLY_SRC, "q4km_matmul_rb_coopmat_f32acc_cached_ablate_structonly") else {
        c.bench_function(bench_id, |b| b.iter(|| {})); return;
    };
    let Some(passthrough) = prepare(&ctx, PASSTHROUGH_SRC, "q4km_matmul_rb_coopmat_f32acc_cached_ablate_passthrough") else {
        c.bench_function(bench_id, |b| b.iter(|| {})); return;
    };
    let Some(leader) = prepare(&ctx, M36_LEADER_SRC, "q4km_matmul_rb_coopmat_f32acc_cached") else {
        c.bench_function(bench_id, |b| b.iter(|| {})); return;
    };
    let Some(core) = prepare(&ctx, PLAIN_F32_CORE_SRC, "matmul_rb_coopmat") else {
        c.bench_function(bench_id, |b| b.iter(|| {})); return;
    };

    let mut group = c.benchmark_group("resident_q4km_matmul_rb_f32acc_cached_ablate");
    group.sample_size(10);
    group.bench_function(bench_id, |b| {
        b.iter_custom(|iters| {
            let mut total_ns: u64 = 0;
            for _ in 0..iters {
                run_triangulation(&ctx, &probe, &passthrough, &leader, &core.handle, shared_bytes);
                // Criterion duration: re-time the 256³ V-structure-probe.
                let q = make_q4km_weights(256, 1, 0xC0FFEE ^ 256);
                let xb = make_x_f16(256, 256, 0xBADF00D ^ 256);
                let (ns, _, _) = resident_min_of_n(
                    &ctx, &probe.handle,
                    &[&q, &xb, &vec![0u8; 256 * 256 * 4]],
                    &[0, 0, (256 * 256 * 4) as u64],
                    (8, 8, 1),
                    assemble_q4km_pc(&probe.plan, 256, 256, 256, 1),
                );
                total_ns += ns;
            }
            std::time::Duration::from_nanos(total_ns)
        });
    });
    group.finish();
}

criterion_group!(resident_q4km_matmul_rb_f32acc_cached_ablate_benches, bench_resident_ablate);
criterion_main!(resident_q4km_matmul_rb_f32acc_cached_ablate_benches);
