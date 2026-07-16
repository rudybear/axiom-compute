//! M4.2a — AT-2992/93/94: CROSS-KERNEL BIT-IDENTITY between the M3.6 leader
//! (q4km_matmul_rb_coopmat_f32acc_cached.axc) and the ggml-ABI-matched variant
//! (q4km_matmul_rb_coopmat_f32acc_cached_ggml.axc), de-transposed, on an f16-EXACT B fixture.
//!
//! WHY BIT-IDENTITY IS THE CORRECT ANCHOR (spec §5.2): the ggml variant's B buffer is f32 and is
//! staged f32->f16 via `f32_to_f16`; this round-trips EXACTLY to the SAME f16 bits the leader
//! reads directly IFF the fixture is f16-EXACT — i.e. every B value is already f16-representable.
//! The fixture here is built by generating f16 values FIRST (the `make_x_f16` seeded [-1,1]
//! grid — finite, no inf/nan/overflow) and WIDENING to f32 for the ggml variant
//! (`x_ggml_f32[i] = f32(x_f16[i])`). Both the widen and the staging convert are then TOTAL and
//! EXACT, so the coopmat B tiles are bit-identical to the leader's, the accumulator receives the
//! IDENTICAL sequence of f32 adds, and the two GPU outputs (leader row-major, ggml column-major
//! de-transposed) MUST match bit-for-bit. A non-zero diff on this fixture is a BUG (an addressing
//! slip, a mis-set stride_a/256, a column-major-store bug, an axis-swap error, or a non-exact
//! convert), NOT a permissible f32 reorder — per AT-1803's STOP-and-investigate rule.
//!
//! GPU-gated (#[ignore] + AXC_ENABLE_GPU_TESTS=1), typed-skip on Lavapipe / subgroup != 32.

use std::collections::BTreeMap;
use axc_driver::compile_source_with_assignments;
use axc_runtime::{VulkanContext, DispatchError};

const LEADER_SRC: &str = include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached.axc");
const GGML_SRC: &str =
    include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached_ggml.axc");

fn gpu_tests_enabled() -> bool {
    std::env::var("AXC_ENABLE_GPU_TESTS").as_deref() == Ok("1")
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

/// IDENTICAL seed scheme to dispatch_q4km_f32acc_cached_equiv.rs / dispatch_q4km_ggml.rs, so
/// BOTH kernels see the SAME logical weight/activation matrices.
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

/// §5.2 f16-EXACT widen: x_ggml_f32[col*k_total + k] = f32(x_f16[k*n + col]) — exact, no rounding.
fn transpose_widen_b_exact(x_f16: &[u16], k_total: usize, n: usize) -> Vec<f32> {
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

/// Dispatch the M3.6 LEADER; returns logical [M,N] row-major f32 output, or None on typed-skip.
#[allow(clippy::too_many_arguments)]
fn dispatch_leader(at: &str, q_bytes: &[u8], x_f16: &[u16], m: usize, n: usize, k: usize, n_bpr: usize, ctx: &VulkanContext) -> Option<Vec<f32>> {
    let assignments = rb2x2_assignments();
    let (bytes, meta) = compile_source_with_assignments(LEADER_SRC, &assignments)
        .unwrap_or_else(|e| panic!("{at}: leader must compile: {e:?}"));
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let handle = match ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, meta.coopmat.as_ref(), "q4km_matmul_rb_coopmat_f32acc_cached",
        meta.shared_memory_bytes,
    ) {
        Ok(h) => h,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("{at}: leader CoopMatUnsupported (typed-skip): {reason}");
            return None;
        }
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
            eprintln!("{at}: leader DeviceFeatureUnsupported {feature}/{kernel} (typed-skip)");
            return None;
        }
        Err(e) => panic!("{at}: leader prepare_kernel_checked failed: {e:?}"),
    };

    let mut pc: Vec<u8> = vec![0u8; meta.push_constant_total_bytes as usize];
    for s in &meta.binding_plan.scalars {
        let val: u32 = match s.name.as_str() {
            "M" => m as u32,
            "N" => n as u32,
            "K" => k as u32,
            "n_blocks_per_row" => n_bpr as u32,
            other => panic!("unexpected leader scalar {other}"),
        };
        pc[s.offset as usize..s.offset as usize + 4].copy_from_slice(&val.to_le_bytes());
    }

    let x_bytes: Vec<u8> = x_f16.iter().flat_map(|&b| b.to_le_bytes()).collect();
    let c_size: usize = m * n * 4;
    let workgroups = ((n / 32) as u32, (m / 32) as u32, 1u32);

    let outputs = match ctx.dispatch_handle(
        &handle, workgroups,
        &[q_bytes, &x_bytes, &vec![0u8; c_size]],
        &[0, 0, c_size],
        &pc,
    ) {
        Ok(v) => v,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("{at}: leader CoopMatUnsupported at dispatch (typed-skip): {reason}");
            return None;
        }
        Err(e) => panic!("{at}: leader dispatch failed: {e:?}"),
    };
    let c_bytes: &[u8] = &outputs[2];
    Some(c_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// Dispatch the ggml VARIANT; returns logical [M,N] row-major f32 output (de-transposed), or
/// None on typed-skip.
fn dispatch_ggml(at: &str, q_bytes: &[u8], x_f16: &[u16], m: usize, n: usize, k: usize, ctx: &VulkanContext) -> Option<Vec<f32>> {
    let assignments = rb2x2_assignments();
    let (bytes, meta) = compile_source_with_assignments(GGML_SRC, &assignments)
        .unwrap_or_else(|e| panic!("{at}: ggml variant must compile: {e:?}"));
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let handle = match ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, meta.coopmat.as_ref(), "q4km_matmul_rb_coopmat_f32acc_cached_ggml",
        meta.shared_memory_bytes,
    ) {
        Ok(h) => h,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("{at}: ggml CoopMatUnsupported (typed-skip): {reason}");
            return None;
        }
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
            eprintln!("{at}: ggml DeviceFeatureUnsupported {feature}/{kernel} (typed-skip)");
            return None;
        }
        Err(e) => panic!("{at}: ggml prepare_kernel_checked failed: {e:?}"),
    };

    let (stride_a, stride_b, stride_d) = (k as u32, k as u32, m as u32);
    let mut pc: Vec<u8> = vec![0u8; meta.push_constant_total_bytes as usize];
    for s in &meta.binding_plan.scalars {
        let val: u32 = match s.name.as_str() {
            "M" => m as u32,
            "N" => n as u32,
            "K" => k as u32,
            "stride_a" => stride_a,
            "stride_b" => stride_b,
            "stride_d" => stride_d,
            other => panic!("unexpected ggml scalar {other}"),
        };
        pc[s.offset as usize..s.offset as usize + 4].copy_from_slice(&val.to_le_bytes());
    }

    let x_ggml_f32 = transpose_widen_b_exact(x_f16, k, n);
    let x_bytes: Vec<u8> = x_ggml_f32.iter().flat_map(|&v| v.to_le_bytes()).collect();
    let c_size: usize = m * n * 4;
    let workgroups = ((m / 32) as u32, (n / 32) as u32, 1u32);

    let outputs = match ctx.dispatch_handle(
        &handle, workgroups,
        &[q_bytes, &x_bytes, &vec![0u8; c_size]],
        &[0, 0, c_size],
        &pc,
    ) {
        Ok(v) => v,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("{at}: ggml CoopMatUnsupported at dispatch (typed-skip): {reason}");
            return None;
        }
        Err(e) => panic!("{at}: ggml dispatch failed: {e:?}"),
    };
    let c_bytes: &[u8] = &outputs[2];
    let d_col_major: Vec<f32> = c_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    Some(detranspose_d(&d_col_major, m, n, stride_d as usize))
}

/// Core: dispatch BOTH kernels on the SAME logical fixture, assert bit-identical (de-transposed) output.
fn run_equiv(at: &str, m: usize, n: usize, n_bpr: usize) {
    if !gpu_tests_enabled() {
        eprintln!("{at}: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    let ctx = VulkanContext::new().expect("VulkanContext must init");
    eprintln!("{at}: device={}", ctx.physical_device_name());

    if !ctx.coopmat_support().feature_present {
        eprintln!("{at}: coopmat not supported on {}; typed-skip", ctx.physical_device_name());
        return;
    }
    if ctx.subgroup_size() != 32 {
        eprintln!("{at}: subgroup_size()={} != 32; typed-skip", ctx.subgroup_size());
        return;
    }

    let k: usize = n_bpr * 256;
    assert!(m.is_multiple_of(32) && n.is_multiple_of(32), "{at}: M and N must be multiples of 32");

    // SAME fixture (same seed scheme) fed to BOTH kernels — the M3.6-equiv seed scheme.
    let q_bytes = make_q4km_weights(m, n_bpr, 0xC0FFEE ^ (m as u64));
    let x_f16 = make_x_f16(k, n, 0xBADF00D ^ (n as u64));

    let Some(y_leader) = dispatch_leader(at, &q_bytes, &x_f16, m, n, k, n_bpr, &ctx) else { return; };
    let Some(y_ggml) = dispatch_ggml(at, &q_bytes, &x_f16, m, n, k, &ctx) else { return; };

    assert_eq!(
        y_leader.len(), y_ggml.len(),
        "{at}: leader/ggml output length mismatch ({} vs {})", y_leader.len(), y_ggml.len()
    );

    let mut n_diff = 0usize;
    let mut first_diff: Option<(usize, f32, f32)> = None;
    for (i, (&a, &b)) in y_leader.iter().zip(y_ggml.iter()).enumerate() {
        if a.to_bits() != b.to_bits() {
            n_diff += 1;
            if first_diff.is_none() {
                first_diff = Some((i, a, b));
            }
        }
    }

    assert!(
        n_diff == 0,
        "AT-2992/93/94: ggml variant output (de-transposed) is NOT bit-identical to the M3.6 \
         leader's at M={m} N={n} K={k}: {n_diff} of {} elements differ; first at idx {:?} \
         (leader={:?}, ggml={:?}). The f16-exact fixture design guarantees the f32->f16 B \
         staging convert round-trips to the IDENTICAL f16 bits the leader reads directly, so a \
         non-zero diff is a BUG (addressing slip, stride_a/256 mis-set, column-major-store bug, \
         or axis-swap error), NOT a permissible reorder. STOP and investigate.",
        y_leader.len(),
        first_diff.map(|(i, _, _)| i),
        first_diff.map(|(_, a, _)| a),
        first_diff.map(|(_, _, b)| b),
    );

    eprintln!(
        "{at}: PASS — ggml (de-transposed) == leader BIT-IDENTICAL at M={m} N={n} K={k} \
         ({} elements) on {}",
        y_leader.len(), ctx.physical_device_name()
    );
}

/// AT-2992: bit-identity at K=256 (1 superblock), M=N=64.
#[test]
#[ignore]
fn at_2992_q4km_ggml_equiv_leader_k256() {
    run_equiv("at_2992", 64, 64, 1);
}

/// AT-2993: bit-identity at K=512 (2 superblocks), M=64, N=128.
#[test]
#[ignore]
fn at_2993_q4km_ggml_equiv_leader_k512() {
    run_equiv("at_2993", 64, 128, 2);
}

/// AT-2994: bit-identity at K=14336 (56 superblocks, inference K), M=N=64.
#[test]
#[ignore]
fn at_2994_q4km_ggml_equiv_leader_k14336() {
    run_equiv("at_2994", 64, 64, 56);
}
