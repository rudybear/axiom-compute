//! M3.9 — multi-subgroup warptile coopmat matmul GPU dispatch correctness tests.
//!
//! AT-2300: K=4 warptile bit-exact vs CPU oracle at M=N=64, K=32 (1 WG, 4 subgroups, 2 k_blocks).
//! AT-2301: K=4 warptile bit-exact at M=N=128, K=48 (2x2 grid, 3 k_blocks -> >1 WAR barrier;
//!          inner staging iters K-independent == 8).
//! AT-2302: K=4 warptile bit-exact at rectangular M=128, N=64, K=32 (asymmetric grid (1,2,1)).
//! AT-2303: K=2 opponent kernel bit-exact at M=N=64, K=32 (controlled A/B baseline).
//! AT-2305 (REQUIRED — the cross-subgroup-staging-race detector): cross-kernel BIT-IDENTITY
//!          warptile-K=4 vs single-subgroup RB at 768^3 on an ASYMMETRIC fixture
//!          (make_transpose_fixture_a/b tiled across 768x768 = 144 workgroups). n_diff==0.
//!
//! ALL tests TYPED-SKIP when subgroup_size()!=32 (the kernel is hard-wired to the wave32
//! 4-subgroup partition; mirror dispatch_matmul_tile.rs:231 / the M3.3d guard) and on
//! CoopMatUnsupported (Lavapipe). #[ignore]+AXC_ENABLE_GPU_TESTS gated.

#[path = "common_matmul.rs"]
mod common_matmul;

use std::collections::BTreeMap;
use axc_driver::compile_source_with_assignments;
use axc_runtime::{VulkanContext, DispatchError};

const MATMUL_WARPTILE_SRC: &str = include_str!("../../../examples/matmul_warptile_coopmat.axc");
const MATMUL_WARPTILE_2SG_SRC: &str = include_str!("../../../examples/matmul_warptile_coopmat_2sg.axc");
const MATMUL_RB_COOPMAT_SRC: &str = include_str!("../../../examples/matmul_rb_coopmat.axc");

fn gpu_tests_enabled() -> bool {
    std::env::var("AXC_ENABLE_GPU_TESTS").as_deref() == Ok("1")
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn f32_slice_to_f16_le_bytes(vals: &[f32]) -> Vec<u8> {
    use half::f16;
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        out.extend_from_slice(&f16::from_f32(v).to_le_bytes());
    }
    out
}

fn f16_le_bytes_to_f32_slice(bytes: &[u8]) -> Vec<f32> {
    use half::f16;
    bytes.chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

fn push_mnk(m: u32, n: u32, k: u32) -> Vec<u8> {
    let mut pc = Vec::with_capacity(12);
    pc.extend_from_slice(&m.to_le_bytes());
    pc.extend_from_slice(&n.to_le_bytes());
    pc.extend_from_slice(&k.to_le_bytes());
    pc
}

/// CPU oracle: row-major f16 matmul via f32 accumulation. For the small test shapes
/// (K<=48, integer-valued f16 inputs) the result is f16-integer-exact (M3.3c AT-1731).
fn cpu_matmul(a_f32: &[f32], b_f32: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0_f32;
            for ki in 0..k {
                sum += a_f32[row * k + ki] * b_f32[ki * n + col];
            }
            c[row * n + col] = sum;
        }
    }
    c
}

/// K=4 warptile assignments (wg_threads=128, 4 subgroups, 64x64 tile).
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

/// K=2 opponent assignments (wg_threads=64, 2 subgroups, 32x64 tile).
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

/// Single-subgroup RB 2x2 assignments (matmul_rb_coopmat.axc).
fn rb2x2_assignments() -> BTreeMap<String, i64> {
    let mut m = BTreeMap::new();
    m.insert("rb_m".to_owned(), 2_i64);
    m.insert("rb_n".to_owned(), 2_i64);
    m.insert("tile_k".to_owned(), 16_i64);
    m.insert("a_block_size".to_owned(), 512_i64);
    m.insert("b_block_size".to_owned(), 512_i64);
    m
}

/// Prepare a kernel handle; return None (with a logged typed-skip) on CoopMatUnsupported /
/// DeviceFeatureUnsupported (Lavapipe).
fn prepare_or_skip(
    ctx: &VulkanContext,
    src: &str,
    assignments: &BTreeMap<String, i64>,
    name: &str,
    at: &str,
) -> Option<axc_runtime::KernelHandle> {
    let (bytes, meta) = compile_source_with_assignments(src, assignments)
        .unwrap_or_else(|e| panic!("{at}: {name} must compile: {e:?}"));
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    match ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, meta.coopmat.as_ref(), name, meta.shared_memory_bytes,
    ) {
        Ok(h) => Some(h),
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("{at}: CoopMatUnsupported (typed-skip on Lavapipe): {reason}");
            None
        }
        Err(DispatchError::DeviceFeatureUnsupported { feature, .. }) => {
            eprintln!("{at}: DeviceFeatureUnsupported({feature}) — typed-skip");
            None
        }
        Err(e) => panic!("{at}: prepare_kernel_checked failed: {e}"),
    }
}

/// Run the warptile-K=4 bit-exact-vs-CPU check at one shape (grid (N/64, M/64, 1)).
fn run_k4_bit_exact(at: &str, m: usize, n: usize, k: usize) {
    if !gpu_tests_enabled() {
        eprintln!("{at}: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    let ctx = VulkanContext::new().unwrap_or_else(|e| panic!("{at}: VulkanContext init: {e}"));
    eprintln!("{at}: device={}", ctx.physical_device_name());
    if ctx.subgroup_size() != 32 {
        eprintln!(
            "{at}: subgroup_size={} != 32; skipping (wave32 guard — kernel REQUIRES sg_size==32)",
            ctx.subgroup_size()
        );
        return;
    }

    let assignments = warptile_k4_assignments();
    let handle = match prepare_or_skip(&ctx, MATMUL_WARPTILE_SRC, &assignments,
        "matmul_warptile_coopmat", at) {
        Some(h) => h,
        None => return,
    };

    // Non-symmetric integer-f16 fixture (A in {1..4}, B in {1..3}); max element K*4*3 <= 2048.
    let a_f32: Vec<f32> = (0..m * k).map(|idx| ((idx % 4) + 1) as f32).collect();
    let b_f32: Vec<f32> = (0..k * n).map(|idx| ((idx % 3) + 1) as f32).collect();
    let a_bytes = f32_slice_to_f16_le_bytes(&a_f32);
    let b_bytes = f32_slice_to_f16_le_bytes(&b_f32);
    let c_size = m * n * 2;

    let wg_x = (n / 64) as u32;
    let wg_y = (m / 64) as u32;
    let pc = push_mnk(m as u32, n as u32, k as u32);

    let outputs = ctx.dispatch_handle(
        &handle, (wg_x, wg_y, 1),
        &[&a_bytes, &b_bytes, &vec![0u8; c_size]],
        &[0, 0, c_size],
        &pc,
    ).unwrap_or_else(|e| panic!("{at}: dispatch failed: {e}"));

    let gpu_c = f16_le_bytes_to_f32_slice(&outputs[2]);
    let cpu_c = cpu_matmul(&a_f32, &b_f32, m, n, k);

    let mut max_diff = 0.0_f32;
    for (g, c) in gpu_c.iter().zip(cpu_c.iter()) {
        let d = (g - c).abs();
        if d > max_diff { max_diff = d; }
    }
    eprintln!(
        "{at}: M={m} N={n} K={k}, grid=({wg_x},{wg_y},1), max_diff={max_diff}, \
         first4 GPU={:?} CPU={:?}",
        &gpu_c[..4.min(gpu_c.len())], &cpu_c[..4.min(cpu_c.len())]
    );
    assert!(
        max_diff == 0.0,
        "{at}: warptile K=4 FAILED — max_diff={max_diff} != 0.\n\
         HONESTY GATE: do NOT relax tolerance or shrink fixture. \
         grid=({wg_x},{wg_y},1) M={m} N={n} K={k}. First4 GPU {:?} CPU {:?}",
        &gpu_c[..4.min(gpu_c.len())], &cpu_c[..4.min(cpu_c.len())]
    );
    eprintln!(
        "{at} PASS: warptile K=4 bit-exact (max_diff=0.0) on {} — M={m} N={n} K={k} grid=({wg_x},{wg_y},1)",
        ctx.physical_device_name()
    );
}

// ── AT-2300/2301/2302: K=4 warptile bit-exact ────────────────────────────────

#[test]
#[ignore]
fn at_2300_warptile_64cube_k32_bit_exact() {
    // 1 workgroup (4 subgroups), 2 k_blocks.
    run_k4_bit_exact("AT-2300", 64, 64, 32);
}

#[test]
#[ignore]
fn at_2301_warptile_128_k48_bit_exact() {
    // 2x2 grid, 3 k_blocks (>1 WAR barrier); inner staging iters K-independent == 8.
    run_k4_bit_exact("AT-2301", 128, 128, 48);
}

#[test]
#[ignore]
fn at_2302_warptile_rect_bit_exact() {
    // Rectangular M=128, N=64, K=32 -> asymmetric grid (N/64, M/64, 1) = (1, 2, 1).
    run_k4_bit_exact("AT-2302", 128, 64, 32);
}

// ── AT-2303: K=2 opponent bit-exact ──────────────────────────────────────────

#[test]
#[ignore]
fn at_2303_warptile_2sg_bit_exact() {
    let at = "AT-2303";
    if !gpu_tests_enabled() {
        eprintln!("{at}: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    const M: usize = 64;
    const N: usize = 64;
    const K: usize = 32;

    let ctx = VulkanContext::new().unwrap_or_else(|e| panic!("{at}: VulkanContext init: {e}"));
    eprintln!("{at}: device={}", ctx.physical_device_name());
    if ctx.subgroup_size() != 32 {
        eprintln!("{at}: subgroup_size={} != 32; skipping (wave32 guard)", ctx.subgroup_size());
        return;
    }

    let assignments = warptile_k2_assignments();
    let handle = match prepare_or_skip(&ctx, MATMUL_WARPTILE_2SG_SRC, &assignments,
        "matmul_warptile_coopmat_2sg", at) {
        Some(h) => h,
        None => return,
    };

    let a_f32: Vec<f32> = (0..M * K).map(|idx| ((idx % 4) + 1) as f32).collect();
    let b_f32: Vec<f32> = (0..K * N).map(|idx| ((idx % 3) + 1) as f32).collect();
    let a_bytes = f32_slice_to_f16_le_bytes(&a_f32);
    let b_bytes = f32_slice_to_f16_le_bytes(&b_f32);
    let c_size = M * N * 2;

    // K=2 opponent grid: (N/64, M/32, 1) = (1, 2, 1).
    let wg_x = (N / 64) as u32;
    let wg_y = (M / 32) as u32;
    let pc = push_mnk(M as u32, N as u32, K as u32);

    let outputs = ctx.dispatch_handle(
        &handle, (wg_x, wg_y, 1),
        &[&a_bytes, &b_bytes, &vec![0u8; c_size]],
        &[0, 0, c_size],
        &pc,
    ).unwrap_or_else(|e| panic!("{at}: dispatch failed: {e}"));

    let gpu_c = f16_le_bytes_to_f32_slice(&outputs[2]);
    let cpu_c = cpu_matmul(&a_f32, &b_f32, M, N, K);
    let mut max_diff = 0.0_f32;
    for (g, c) in gpu_c.iter().zip(cpu_c.iter()) {
        let d = (g - c).abs();
        if d > max_diff { max_diff = d; }
    }
    eprintln!(
        "{at}: M={M} N={N} K={K}, grid=({wg_x},{wg_y},1) = 2 subgroups, max_diff={max_diff}"
    );
    assert!(
        max_diff == 0.0,
        "{at}: warptile K=2 FAILED — max_diff={max_diff} != 0. HONESTY GATE: no tol relax."
    );
    eprintln!("{at} PASS: warptile K=2 bit-exact (max_diff=0.0) on {}", ctx.physical_device_name());
}

// ── AT-2305: cross-kernel BIT-IDENTITY warptile-K=4 vs single-subgroup RB ─────

/// Build an ASYMMETRIC (per-row/col-varying) MxK A fixture by tiling the 16x16
/// make_transpose_fixture_a primitive across the full MxK extent. A swapped/transposed
/// sub-tile or a missing cross-subgroup barrier cannot pass by symmetry.
fn tiled_fixture_a(m: usize, k: usize) -> Vec<f32> {
    let base = common_matmul::make_transpose_fixture_a(); // 16x16 f16 bit patterns
    let mut out = vec![0.0_f32; m * k];
    for i in 0..m {
        for j in 0..k {
            let bi = i % 16;
            let bj = j % 16;
            out[i * k + j] = common_matmul::f16_bits_to_f32(base[bi * 16 + bj]);
        }
    }
    out
}

/// Build an ASYMMETRIC KxN B fixture by tiling make_transpose_fixture_b across KxN.
fn tiled_fixture_b(k: usize, n: usize) -> Vec<f32> {
    let base = common_matmul::make_transpose_fixture_b();
    let mut out = vec![0.0_f32; k * n];
    for i in 0..k {
        for j in 0..n {
            let bi = i % 16;
            let bj = j % 16;
            out[i * n + j] = common_matmul::f16_bits_to_f32(base[bi * 16 + bj]);
        }
    }
    out
}

/// AT-2305 (REQUIRED): cross-kernel BIT-IDENTITY — the warptile is a pure RE-PARTITION of
/// the SAME coopmat MACs in the SAME per-element accumulation order, so its output MUST be
/// BIT-IDENTICAL to the single-subgroup RB kernel on identical inputs. ANY divergence is a
/// partition bug (swapped sub-tile) or a missing/mis-scoped cross-subgroup barrier — and on
/// NVIDIA a missing cross-subgroup barrier WILL manifest (unlike single-subgroup lockstep).
///
/// FIXTURE IS ASYMMETRIC (make_transpose_fixture_a/b tiled — per-row/col-varying) so a
/// swapped/transposed sub-tile cannot pass by data symmetry. SHAPE 768^3 -> warptile grid
/// (12,12)=144 workgroups (>1 workgroup: multiple 64x64 tiles + cross-subgroup cooperative
/// staging exercised, not a single-tile degenerate case).
#[test]
#[ignore]
fn at_2305_warptile_vs_rb_bit_identity() {
    let at = "AT-2305";
    if !gpu_tests_enabled() {
        eprintln!("{at}: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    const M: usize = 768;
    const N: usize = 768;
    const K: usize = 768;

    let ctx = VulkanContext::new().unwrap_or_else(|e| panic!("{at}: VulkanContext init: {e}"));
    eprintln!("{at}: device={}", ctx.physical_device_name());
    if ctx.subgroup_size() != 32 {
        eprintln!("{at}: subgroup_size={} != 32; skipping (wave32 guard)", ctx.subgroup_size());
        return;
    }

    let warptile = match prepare_or_skip(&ctx, MATMUL_WARPTILE_SRC, &warptile_k4_assignments(),
        "matmul_warptile_coopmat", at) {
        Some(h) => h,
        None => return,
    };
    let rb = match prepare_or_skip(&ctx, MATMUL_RB_COOPMAT_SRC, &rb2x2_assignments(),
        "matmul_rb_coopmat", at) {
        Some(h) => h,
        None => return,
    };

    // ASYMMETRIC per-row/col-varying fixture (tiled 16x16 transpose-distinguishing primitives).
    let a_f32 = tiled_fixture_a(M, K);
    let b_f32 = tiled_fixture_b(K, N);
    let a_bytes = f32_slice_to_f16_le_bytes(&a_f32);
    let b_bytes = f32_slice_to_f16_le_bytes(&b_f32);
    let c_size = M * N * 2;
    let pc = push_mnk(M as u32, N as u32, K as u32);

    // Warptile K=4 grid: (N/64, M/64, 1) = (12, 12, 1) = 144 workgroups.
    let wt_out = ctx.dispatch_handle(
        &warptile, ((N / 64) as u32, (M / 64) as u32, 1),
        &[&a_bytes, &b_bytes, &vec![0u8; c_size]],
        &[0, 0, c_size],
        &pc,
    ).unwrap_or_else(|e| panic!("{at}: warptile dispatch failed: {e}"));

    // Single-subgroup RB grid: (N/32, M/32, 1) = (24, 24, 1) = 576 workgroups.
    let rb_out = ctx.dispatch_handle(
        &rb, ((N / 32) as u32, (M / 32) as u32, 1),
        &[&a_bytes, &b_bytes, &vec![0u8; c_size]],
        &[0, 0, c_size],
        &pc,
    ).unwrap_or_else(|e| panic!("{at}: rb dispatch failed: {e}"));

    let wt_c = f16_le_bytes_to_f32_slice(&wt_out[2]);
    let rb_c = f16_le_bytes_to_f32_slice(&rb_out[2]);
    assert_eq!(wt_c.len(), rb_c.len(), "{at}: output length mismatch");

    let mut n_diff = 0usize;
    let mut first_diff: Option<(usize, f32, f32)> = None;
    for (idx, (w, r)) in wt_c.iter().zip(rb_c.iter()).enumerate() {
        if w.to_bits() != r.to_bits() {
            n_diff += 1;
            if first_diff.is_none() { first_diff = Some((idx, *w, *r)); }
        }
    }
    eprintln!(
        "{at}: 768^3 asymmetric fixture, warptile grid (12,12,1)=144 WGs vs RB (24,24,1)=576 WGs, \
         n_diff={n_diff}/{} first_diff={:?}",
        wt_c.len(), first_diff
    );
    assert_eq!(
        n_diff, 0,
        "{at}: warptile K=4 must be BIT-IDENTICAL to single-subgroup RB (pure re-partition); \
         n_diff={n_diff}. A non-zero diff is a sub-tile/partition bug or a missing/mis-scoped \
         CROSS-SUBGROUP barrier (the load-bearing race detector). first_diff={first_diff:?}. \
         HONESTY GATE: do NOT relax — debug the sub-tile addressing + the barriers."
    );
    eprintln!(
        "{at} PASS: warptile K=4 BIT-IDENTICAL to single-subgroup RB (n_diff=0) on {} — \
         768^3 asymmetric fixture, cross-subgroup staging + 2x2 sub-tile partition correct",
        ctx.physical_device_name()
    );
}
