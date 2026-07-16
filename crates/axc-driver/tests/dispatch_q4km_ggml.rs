//! M4.2a — GPU correctness tests for the ggml-ABI-matched Q4_K_M variant
//! (examples/q4km_matmul_rb_coopmat_f32acc_cached_ggml.axc) vs the UNCHANGED f32-accumulator
//! oracle (axc_driver::q4km_oracle), via a thin TRANSPOSING + WIDENING wrapper (spec §5.1):
//!
//!   1. build the SAME logical fixtures as the M3.6 dispatch tests (`make_q4km_weights`,
//!      `make_x_f16` — SAME seed scheme, so AT-2992..94 can reuse them for cross-kernel
//!      bit-identity);
//!   2. transpose ONLY the buffer LAYOUTS fed to / read from the GPU: pack B as ggml's `[N,K]`
//!      row-major AND WIDEN f16->f32 (`x_ggml_f32[n*K+k] = f32(x_f16[k*N+n])`, exact — §5.2);
//!      read the device D column-major (`y_gpu[m*N+n] = D[n*stride_d+m]`) back into logical
//!      `[M,N]`;
//!   3. call the UNCHANGED oracle on the logical `[M,K]x[K,N]` problem and compare.
//!
//! AT-2989/90/91: combined condition-aware diff <= FROZEN 1e-3 at K=256/512/14336, M=N=64/128.
//! AT-3002: the REAL rectangular shape (M=4096, N=512, K=14336) — a non-square dispatch so an
//!          axis-swap bug (x<->y) cannot pass silently (OOB or wrong-tile, not a coincidental
//!          match).
//!
//! Push constants for the contiguous case: stride_a=K (n_blocks_per_row=stride_a/256), stride_b=K
//! (B row length, ggml `[N,K]` row-major), stride_d=M (D column stride, `mul_mm.comp:404`,
//! `ggml-vulkan.cpp:8175`: `stride_d = dst->nb[1]/type_size = M` for a contiguous, non-batched D).
//!
//! Dispatch grid = (M/32, N/32, 1) — SWAPPED vs the M3.6 leader's (N/32, M/32, 1) — the ggml
//! grid-axis convention (§2.8, delta (e)).
//!
//! NVIDIA #[ignore]-gated (AXC_ENABLE_GPU_TESTS=1). Typed-skip on Lavapipe
//! (CoopMatUnsupported / DeviceFeatureUnsupported) and on subgroup_size() != 32.

use axc_driver::q4km_oracle as common_q4km_f32ref;

use std::collections::BTreeMap;
use axc_driver::compile_source_with_assignments;
use axc_hir::ParamBindingPlan;
use axc_runtime::{VulkanContext, DispatchError};

const GGML_SRC: &str =
    include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached_ggml.axc");

/// Frozen relative tolerance (AT-1520/AT-1521 value — NOT loosened).
const FROZEN_REL_TOL: f64 = 1e-3;

fn gpu_tests_enabled() -> bool {
    std::env::var("AXC_ENABLE_GPU_TESTS").as_deref() == Ok("1")
}

/// RB 2×2 strategy assignments (matching the M3.6 leader; a_block_size PINNED at 512).
fn rb2x2_assignments() -> BTreeMap<String, i64> {
    let mut m = BTreeMap::new();
    m.insert("rb_m".to_owned(), 2_i64);
    m.insert("rb_n".to_owned(), 2_i64);
    m.insert("tile_k".to_owned(), 16_i64);
    m.insert("a_block_size".to_owned(), 512_i64);
    m.insert("b_block_size".to_owned(), 512_i64);
    m
}

/// Assemble push constants by scalar name: ggml's leading six (M,N,K,stride_a,stride_b,stride_d).
fn assemble_pc(plan: &ParamBindingPlan, m: u32, n: u32, k: u32, stride_a: u32, stride_b: u32, stride_d: u32) -> Vec<u8> {
    let mut pc: Vec<u8> = vec![0u8; plan.push_constant_total_bytes as usize];
    for s in &plan.scalars {
        let val: u32 = match s.name.as_str() {
            "M" => m,
            "N" => n,
            "K" => k,
            "stride_a" => stride_a,
            "stride_b" => stride_b,
            "stride_d" => stride_d,
            other => panic!("unexpected scalar param {other}"),
        };
        let start = s.offset as usize;
        pc[start..start + 4].copy_from_slice(&val.to_le_bytes());
    }
    pc
}

/// IDENTICAL seed scheme to dispatch_q4km_matmul_rb_f32acc_cached.rs / dispatch_q4km_f32acc_cached_equiv.rs.
pub fn make_q4km_weights(m: usize, n_bpr: usize, seed: u64) -> Vec<u8> {
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

/// IDENTICAL seed scheme; logical layout x_f16[k*n_cols + col] (row-major [K,N]).
pub fn make_x_f16(k: usize, n: usize, seed: u64) -> Vec<u16> {
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

/// §5.1/§5.2: transpose the logical x_f16[K,N] into ggml's B layout `[N,K]` row-major, WIDENING
/// f16->f32 (exact — the f16-exact fixture, §5.2). `x_ggml_f32[n*k_total + k] = f32(x_f16[k*n + col])`.
pub fn transpose_widen_b(x_f16: &[u16], k_total: usize, n: usize) -> Vec<f32> {
    use half::f16;
    let mut out = vec![0.0_f32; k_total * n];
    for k in 0..k_total {
        for col in 0..n {
            out[col * k_total + k] = f16::from_bits(x_f16[k * n + col]).to_f32();
        }
    }
    out
}

/// §5.1: de-transpose the GPU's column-major D output back into logical `[M,N]` row-major.
/// `y_gpu[m*n + col] = d_col_major[col*stride_d + m]`.
pub fn detranspose_d(d_col_major: &[f32], m: usize, n: usize, stride_d: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; m * n];
    for row in 0..m {
        for col in 0..n {
            out[row * n + col] = d_col_major[col * stride_d + row];
        }
    }
    out
}

struct MeasuredErr {
    raw_rel: f64,
    combined: f64,
}

/// Core: dispatch the ggml variant on a contiguous (M,N,K) fixture and compare vs the f32
/// oracle via the transposing+widening wrapper.
fn run_ggml(at: &str, m: usize, n: usize, n_bpr: usize) -> Option<MeasuredErr> {
    if !gpu_tests_enabled() {
        eprintln!("{at}: AXC_ENABLE_GPU_TESTS not set; skipping");
        return None;
    }
    let ctx = VulkanContext::new().expect("VulkanContext must init");
    eprintln!("{at}: device={}", ctx.physical_device_name());

    if !ctx.coopmat_support().feature_present {
        eprintln!("{at}: coopmat not supported on {}; typed-skip", ctx.physical_device_name());
        return None;
    }
    if ctx.subgroup_size() != 32 {
        eprintln!("{at}: subgroup_size()={} != 32; typed-skip (kernel requires wave32)", ctx.subgroup_size());
        return None;
    }

    let k: usize = n_bpr * 256;
    assert!(m.is_multiple_of(32) && n.is_multiple_of(32), "{at}: M and N must be multiples of 32");

    // Grid pre-check (M/32, N/32, 1) — the ggml axis-swapped grid (delta e).
    let wg_x = (m / 32) as u64;
    let wg_y = (n / 32) as u64;
    let max_wg = ctx.max_compute_work_group_count();
    if wg_x > max_wg[0] as u64 || wg_y > max_wg[1] as u64 {
        eprintln!("{at}: grid ({wg_x},{wg_y}) exceeds device limits at M={m} N={n} K={k} — skip");
        return None;
    }

    // Logical fixtures — SAME seed scheme as the M3.6 dispatch tests.
    let q_bytes = make_q4km_weights(m, n_bpr, 0xC0FFEE ^ (m as u64));
    let x_f16 = make_x_f16(k, n, 0xBADF00D ^ (n as u64));

    // f32-accumulator CPU reference (UNCHANGED oracle, logical [M,K]x[K,N]).
    let y_ref = common_q4km_f32ref::q4km_dequant_matmul_f32accum_cpu(&q_bytes, &x_f16, m, n, n_bpr);

    // Transpose + widen B into ggml's [N,K] row-major f32 layout (§5.1/§5.2).
    let x_ggml_f32 = transpose_widen_b(&x_f16, k, n);
    let x_bytes: Vec<u8> = x_ggml_f32.iter().flat_map(|&v| v.to_le_bytes()).collect();

    let assignments = rb2x2_assignments();
    let (bytes, meta) = compile_source_with_assignments(GGML_SRC, &assignments)
        .expect("q4km_matmul_rb_coopmat_f32acc_cached_ggml.axc must compile");
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let handle = match ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, meta.coopmat.as_ref(), "q4km_matmul_rb_coopmat_f32acc_cached_ggml",
        meta.shared_memory_bytes,
    ) {
        Ok(h) => h,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("{at}: CoopMatUnsupported (typed-skip): {reason}");
            return None;
        }
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
            eprintln!("{at}: DeviceFeatureUnsupported {feature}/{kernel} (typed-skip)");
            return None;
        }
        Err(e) => panic!("{at}: prepare_kernel_checked failed: {e:?}"),
    };

    // Contiguous case: stride_a=K, stride_b=K, stride_d=M (§2.3-2.8).
    let (stride_a, stride_b, stride_d) = (k as u32, k as u32, m as u32);
    let pc = assemble_pc(&meta.binding_plan, m as u32, n as u32, k as u32, stride_a, stride_b, stride_d);
    let c_size: usize = m * n * 4; // f32 output, column-major (stride_d=M).
    // Grid = (M/32, N/32, 1) — ggml's axis convention (delta e).
    let workgroups = (wg_x as u32, wg_y as u32, 1u32);

    let outputs = match ctx.dispatch_handle(
        &handle, workgroups,
        &[&q_bytes, &x_bytes, &vec![0u8; c_size]],
        &[0, 0, c_size],
        &pc,
    ) {
        Ok(v) => v,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("{at}: CoopMatUnsupported at dispatch (typed-skip): {reason}");
            return None;
        }
        Err(e) => panic!("{at}: dispatch failed: {e:?}"),
    };

    let c_bytes: &[u8] = &outputs[2];
    assert_eq!(c_bytes.len(), c_size, "{at}: C output size mismatch");
    let d_col_major: Vec<f32> = c_bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let y_gpu = detranspose_d(&d_col_major, m, n, stride_d as usize);

    let raw_rel = common_q4km_f32ref::max_rel_diff(&y_gpu, &y_ref);
    let abs_scale = common_q4km_f32ref::q4km_abs_dot_scale(&q_bytes, &x_f16, m, n, n_bpr);
    let combined = common_q4km_f32ref::max_rel_diff_combined(&y_gpu, &y_ref, &abs_scale);
    eprintln!(
        "{at}: MEASURED raw max_rel_diff={raw_rel:.3e} | COMBINED (condition-aware, the GATE) \
         ={combined:.3e} (M={m} N={n} K={k}, frozen rtol={FROZEN_REL_TOL:.0e}) on {} \
         dispatch=({wg_x},{wg_y},1)",
        ctx.physical_device_name()
    );
    Some(MeasuredErr { raw_rel, combined })
}

/// AT-2989: K=256 (1 superblock), M=N=64.
#[test]
#[ignore]
fn at_2989_q4km_ggml_within_tol_k256() {
    let Some(e) = run_ggml("at_2989", 64, 64, 1) else { return; };
    assert!(
        e.combined <= FROZEN_REL_TOL,
        "AT-2989: K=256 ggml combined (condition-aware) max_rel_diff={:.3e} exceeds frozen rtol \
         {FROZEN_REL_TOL:.0e} (raw={:.3e})",
        e.combined, e.raw_rel
    );
    eprintln!("at_2989: PASS — ggml K=256 within frozen 1e-3 (combined={:.3e})", e.combined);
}

/// AT-2990: K=512 (2 superblocks) — the M3.5 f16-accumulator failure case; must hold with f32
/// acc + caching + the ggml ABI deltas.
#[test]
#[ignore]
fn at_2990_q4km_ggml_within_tol_k512() {
    let Some(e) = run_ggml("at_2990", 64, 128, 2) else { return; };
    assert!(
        e.combined <= FROZEN_REL_TOL,
        "AT-2990: K=512 ggml combined (condition-aware) max_rel_diff={:.3e} exceeds frozen rtol \
         {FROZEN_REL_TOL:.0e} (raw={:.3e})",
        e.combined, e.raw_rel
    );
    eprintln!("at_2990: PASS — ggml K=512 within frozen 1e-3 (combined={:.3e})", e.combined);
}

/// AT-2991: K=14336 (56 superblocks, inference K) — catches an r1-style silent-zeros reset.
#[test]
#[ignore]
fn at_2991_q4km_ggml_within_tol_k14336() {
    let Some(e) = run_ggml("at_2991", 64, 64, 56) else { return; };
    assert!(
        e.combined <= FROZEN_REL_TOL,
        "AT-2991: K=14336 ggml combined (condition-aware) max_rel_diff={:.3e} exceeds frozen \
         rtol {FROZEN_REL_TOL:.0e} (raw={:.3e})",
        e.combined, e.raw_rel
    );
    eprintln!("at_2991: PASS — ggml K=14336 within frozen 1e-3 (combined={:.3e})", e.combined);
}

/// AT-3002: the REAL rectangular shape (M=4096, N=512, K=14336 — the A/B inference shape). A
/// non-square M != N dispatch so an axis-swap error (x<->y) CANNOT pass silently (it would
/// either go OOB or land in the wrong tile, not coincidentally match).
#[test]
#[ignore]
fn at_3002_q4km_ggml_rectangular_m4096_n512() {
    let Some(e) = run_ggml("at_3002", 4096, 512, 56) else { return; };
    assert!(
        e.combined <= FROZEN_REL_TOL,
        "AT-3002: rectangular M=4096 N=512 K=14336 ggml combined (condition-aware) \
         max_rel_diff={:.3e} exceeds frozen rtol {FROZEN_REL_TOL:.0e} — a non-square shape is the \
         DIRECT detector of a grid-axis-swap bug (raw={:.3e})",
        e.combined, e.raw_rel
    );
    eprintln!(
        "at_3002: PASS — rectangular M=4096 N=512 K=14336 within frozen 1e-3 \
         (combined={:.3e}); the grid-axis swap is correct at a non-square shape",
        e.combined
    );
}
