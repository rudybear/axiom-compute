//! AT-2306/2307: Honest effective-TFLOPS bench for the M3.9 multi-subgroup warptile matmul.
//!
//! Measures THREE kernels under IDENTICAL resident methodology (upload-once / N_WARMUP=2 /
//! MIN-of-10 / GpuTimestamp):
//! matmul_warptile_coopmat (K=4, @workgroup(128), 64x64 tile) is the M3.9 leader;
//! matmul_warptile_coopmat_2sg (K=2, @workgroup(64), 32x64 tile) is the A/B opponent;
//! matmul_rb_coopmat (single-subgroup RB 2x2, @workgroup(32)) is the K=1 baseline.
//! Sizes: 256/512/768 (+1024 if <50GB) AND the A/B shape (m=4096, n=512, k=14336).
//!
//! PRIMARY GATE (AT-2306): warptile K=4 >= 1.3x single-subgroup RB at 768^3 OR at the A/B
//! shape. The gate is the MILESTONE acceptance evaluated against the recorded numbers, NOT a
//! hard assert in the bench (the bench asserts only tflops>0 && finite). HONEST-NEGATIVE
//! armed (AT-2307): the M3.3d/M3.7/M3.8 triad shows the kernel is occupancy-bound; if K=4
//! does not beat single-subgroup, the per-size TFLOPS + shared_memory_bytes + workgroup-count
//! and warps-in-flight are recorded for the bottleneck diagnosis (M3.10 vectorized loads and
//! bank padding). NO ratio asserted; the gate is NOT loosened to manufacture a win.
//!
//! ALL kernels TYPED-SKIP when subgroup_size()!=32 / CoopMatUnsupported (Lavapipe).
//! Bench ID: dispatch_resident_matmul_warptile.

#![allow(dead_code)]

use criterion::{criterion_group, criterion_main, Criterion};
use axc_driver::compile_source_with_assignments;
use axc_runtime::{
    VulkanContext, DispatchError,
    ResidentBenchConfig, ResidentTimingSource,
};
use std::collections::BTreeMap;

const MATMUL_WARPTILE_SRC: &str = include_str!("../../../examples/matmul_warptile_coopmat.axc");
const MATMUL_WARPTILE_2SG_SRC: &str = include_str!("../../../examples/matmul_warptile_coopmat_2sg.axc");
const MATMUL_RB_COOPMAT_SRC: &str = include_str!("../../../examples/matmul_rb_coopmat.axc");

const CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000: f64 = 125.0_f64;
const N_WARMUP: usize = 2;
const N_MEASURED: usize = 10;

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
    if kernel_ns == 0 { return 0.0; }
    let s = kernel_ns as f64 * 1e-9_f64;
    2.0_f64 * (m as f64) * (n as f64) * (k as f64) / s / 1e12_f64
}

fn warptile_k4_assignments() -> BTreeMap<String, i64> {
    let mut m = BTreeMap::new();
    m.insert("wg_threads".to_owned(), 128_i64);
    m.insert("n_sg".to_owned(), 4_i64);
    m.insert("bm".to_owned(), 64_i64);
    m.insert("bn".to_owned(), 64_i64);
    m.insert("rb_m".to_owned(), 2_i64);
    m.insert("rb_n".to_owned(), 2_i64);
    m.insert("tile_k".to_owned(), 16_i64);
    m.insert("a_block_size".to_owned(), 1024_i64);
    m.insert("b_block_size".to_owned(), 1024_i64);
    m
}

fn warptile_k2_assignments() -> BTreeMap<String, i64> {
    let mut m = BTreeMap::new();
    m.insert("wg_threads".to_owned(), 64_i64);
    m.insert("n_sg".to_owned(), 2_i64);
    m.insert("bm".to_owned(), 32_i64);
    m.insert("bn".to_owned(), 64_i64);
    m.insert("rb_m".to_owned(), 2_i64);
    m.insert("rb_n".to_owned(), 2_i64);
    m.insert("tile_k".to_owned(), 16_i64);
    m.insert("a_block_size".to_owned(), 512_i64);
    m.insert("b_block_size".to_owned(), 1024_i64);
    m
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

/// Tile geometry per kernel: (tile_n, tile_m, n_warps_per_wg, label).
#[derive(Clone, Copy)]
struct KernelGeom {
    tile_n: usize,   // C cols per workgroup
    tile_m: usize,   // C rows per workgroup
    warps_per_wg: usize,
    label: &'static str,
}

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
    for _ in 0..N_WARMUP {
        ctx.dispatch_resident(handle, &resident, &cfg).expect("warmup dispatch");
    }
    let mut min_ns = u64::MAX;
    let mut last = ResidentTimingSource::CpuFenceWall;
    for _ in 0..N_MEASURED {
        let t = ctx.dispatch_resident(handle, &resident, &cfg).expect("measured dispatch");
        if t.kernel_ns < min_ns { min_ns = t.kernel_ns; }
        last = t.timing_source;
    }
    (min_ns, last)
}

/// f16 LE bytes for A[i]=(i%4)+1, B[i]=(i%3)+1 patterns.
fn fill_f16(len: usize, modulus: usize) -> Vec<u8> {
    use half::f16;
    let mut out = Vec::with_capacity(len * 2);
    for i in 0..len {
        out.extend_from_slice(&f16::from_f32(((i % modulus) + 1) as f32).to_le_bytes());
    }
    out
}

/// Measure one kernel at one (m,n,k) shape; print an honest label + the occupancy artifact
/// (shared_memory_bytes + workgroup-count + warps-in-flight). Returns Some(tflops) or None
/// (typed-skip / grid / 50GB / divisibility). `shared_bytes` is the kernel's reported shared.
fn measure(
    ctx: &VulkanContext,
    handle: &axc_runtime::KernelHandle,
    geom: KernelGeom,
    shared_bytes: u32,
    m: usize, n: usize, k: usize,
) -> Option<f64> {
    // Divisibility: m % tile_m, n % tile_n, k % 16.
    if !m.is_multiple_of(geom.tile_m) || !n.is_multiple_of(geom.tile_n) || !k.is_multiple_of(16) {
        eprintln!(
            "resident_warptile[{}]: M={m}/N={n}/K={k} not divisible by tile ({}x{}) / 16 — skip",
            geom.label, geom.tile_m, geom.tile_n
        );
        return None;
    }
    let wg_x = (n / geom.tile_n) as u64;
    let wg_y = (m / geom.tile_m) as u64;
    let max_wg = ctx.max_compute_work_group_count();
    if wg_x > max_wg[0] as u64 || wg_y > max_wg[1] as u64 {
        eprintln!(
            "resident_warptile[{}]: grid ({wg_x},{wg_y},1) exceeds limits at M={m} N={n} K={k} — skip",
            geom.label
        );
        return None;
    }
    let total_bytes: u64 = 2 * (m as u64) * (k as u64)
        + 2 * (k as u64) * (n as u64) + 2 * (m as u64) * (n as u64);
    const FIFTY_GB: u64 = 50 * 1024 * 1024 * 1024;
    if total_bytes >= FIFTY_GB {
        eprintln!("resident_warptile[{}]: 50 GB cap at M={m} N={n} K={k} — skip", geom.label);
        return None;
    }

    let a_bytes = fill_f16(m * k, 4);
    let b_bytes = fill_f16(k * n, 3);
    let c_size = m * n * 2;
    let mut pc = Vec::with_capacity(12);
    pc.extend_from_slice(&(m as u32).to_le_bytes());
    pc.extend_from_slice(&(n as u32).to_le_bytes());
    pc.extend_from_slice(&(k as u32).to_le_bytes());

    let (min_ns, ts) = resident_min_of_n(
        ctx, handle,
        &[&a_bytes, &b_bytes, &vec![0u8; c_size]],
        &[0, 0, c_size as u64],
        (wg_x as u32, wg_y as u32, 1),
        pc,
    );
    let tflops = effective_tflops(m, n, k, min_ns);
    let pct = tflops / CUBLAS_F32_GEMM_TFLOPS_RTX_PRO_6000 * 100.0_f64;
    let n_wgs = wg_x * wg_y;
    let warps = n_wgs * geom.warps_per_wg as u64;

    eprintln!(
        "resident_warptile[{}]: M={m} N={n} K={k} | TFLOPS={tflops:.3} ({pct:.1}% of 125) \
         | {min_ns} ns ({}) | grid=({wg_x},{wg_y},1)={n_wgs} WGs x {} warps = {warps} warps-in-flight \
         | shared_memory_bytes={shared_bytes}",
        geom.label, timing_source_label(ts), geom.warps_per_wg
    );
    assert!(tflops > 0.0 && tflops.is_finite(), "effective_tflops must be > 0 and finite");
    Some(tflops)
}

/// Prepare a kernel handle (or None on typed-skip). Returns (handle, shared_memory_bytes).
fn prepare(
    ctx: &VulkanContext,
    src: &str,
    assignments: &BTreeMap<String, i64>,
    name: &str,
) -> Option<(axc_runtime::KernelHandle, u32)> {
    let (bytes, meta) = match compile_source_with_assignments(src, assignments) {
        Ok(r) => r,
        Err(e) => { eprintln!("resident_warptile: {name} compile failed: {e:?}"); return None; }
    };
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    match ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, meta.coopmat.as_ref(), name, meta.shared_memory_bytes,
    ) {
        Ok(h) => Some((h, meta.shared_memory_bytes)),
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("resident_warptile: {name} CoopMatUnsupported (typed-skip): {reason}");
            None
        }
        Err(DispatchError::DeviceFeatureUnsupported { feature, .. }) => {
            eprintln!("resident_warptile: {name} DeviceFeatureUnsupported({feature}) — typed-skip");
            None
        }
        Err(e) => { eprintln!("resident_warptile: {name} prepare failed: {e}"); None }
    }
}

/// The shapes: cube sizes + the A/B shape (m=4096, n=512, k=14336).
const SHAPES: &[(usize, usize, usize)] = &[
    (256, 256, 256),
    (512, 512, 512),
    (768, 768, 768),
    (1024, 1024, 1024),
    (4096, 512, 14336), // the A/B shape (occupancy-preserving headline)
];

fn bench_warptile(c: &mut Criterion) {
    if !gpu_benches_enabled() {
        eprintln!("resident_matmul_warptile: AXC_ENABLE_GPU_BENCHES not set; skipping");
        return;
    }
    let ctx = match VulkanContext::new() {
        Ok(c) => c,
        Err(e) => { eprintln!("resident_matmul_warptile: no Vulkan: {e}"); return; }
    };
    if ctx.subgroup_size() != 32 {
        eprintln!(
            "resident_matmul_warptile: subgroup_size={} != 32; skipping (wave32 guard)",
            ctx.subgroup_size()
        );
        return;
    }

    let k4 = prepare(&ctx, MATMUL_WARPTILE_SRC, &warptile_k4_assignments(), "matmul_warptile_coopmat");
    let k2 = prepare(&ctx, MATMUL_WARPTILE_2SG_SRC, &warptile_k2_assignments(), "matmul_warptile_coopmat_2sg");
    let rb = prepare(&ctx, MATMUL_RB_COOPMAT_SRC, &rb2x2_assignments(), "matmul_rb_coopmat");

    let k4_geom = KernelGeom { tile_n: 64, tile_m: 64, warps_per_wg: 4, label: "K=4-warptile" };
    let k2_geom = KernelGeom { tile_n: 64, tile_m: 32, warps_per_wg: 2, label: "K=2-warptile" };
    let rb_geom = KernelGeom { tile_n: 32, tile_m: 32, warps_per_wg: 1, label: "single-subgroup-RB" };

    let mut group = c.benchmark_group("resident_matmul_warptile");
    group.sample_size(10);

    group.bench_function("dispatch_resident_matmul_warptile", |b| {
        b.iter_custom(|iters| {
            let mut total_ns: u64 = 0;
            for _ in 0..iters {
                for &(m, n, k) in SHAPES {
                    eprintln!("── resident_matmul_warptile shape M={m} N={n} K={k} ──");
                    let mut t_k4: Option<f64> = None;
                    let mut t_rb: Option<f64> = None;
                    if let Some((h, sh)) = &k4 {
                        t_k4 = measure(&ctx, h, k4_geom, *sh, m, n, k);
                    }
                    if let Some((h, sh)) = &k2 {
                        let _ = measure(&ctx, h, k2_geom, *sh, m, n, k);
                    }
                    if let Some((h, sh)) = &rb {
                        t_rb = measure(&ctx, h, rb_geom, *sh, m, n, k);
                    }
                    // Honest ratio line (the AT-2306 gate is evaluated against this, NOT asserted).
                    if let (Some(a), Some(b)) = (t_k4, t_rb) {
                        let ratio = a / b;
                        let verdict = if ratio >= 1.3 { "WIN (>=1.3x gate met)" }
                                      else { "HONEST-NEGATIVE (< 1.3x; see occupancy artifact above)" };
                        eprintln!(
                            "resident_matmul_warptile RATIO[M={m} N={n} K={k}]: \
                             K=4-warptile {a:.3} / single-subgroup-RB {b:.3} = {ratio:.3}x — {verdict}"
                        );
                    }
                }
                // Criterion duration accounting: use a tiny 256^3 K=4 (or RB) sample.
                if let Some((h, _)) = &k4 {
                    let a = fill_f16(256 * 256, 4);
                    let bb = fill_f16(256 * 256, 3);
                    let mut pc = Vec::with_capacity(12);
                    for v in [256u32, 256, 256] { pc.extend_from_slice(&v.to_le_bytes()); }
                    let (ns, _) = resident_min_of_n(
                        &ctx, h, &[&a, &bb, &vec![0u8; 256 * 256 * 2]],
                        &[0, 0, (256 * 256 * 2) as u64], (4, 4, 1), pc,
                    );
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

criterion_group!(resident_matmul_warptile_benches, bench_warptile);
criterion_main!(resident_matmul_warptile_benches);
