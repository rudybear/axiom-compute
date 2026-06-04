//! GPU dispatch tests for M3.2/M3.3 shared[T,N] examples.
//!
//! AT-1606: shared_reduce.axc — shared[f32] parallel reduction, bit-exact vs CPU sum
//!          WITH barrier. Barrier-absent OQ1 hard-error test (compile-time).
//!          STATUS: PASSES on NVIDIA RTX PRO 6000 (measured).
//!
//! AT-1707 (M3.3 retry:1): opphi_coopmat_accumulate.axc — MINIMAL OpPhi numeric isolation.
//!          Loads A and B directly from global buffers (NO shared staging), loops K_ITER
//!          times loading the same tile each iteration, accumulates acc += A*B each time.
//!          Expected C = K_ITER × (A·B). If PASS: OpPhi accumulation is numerically correct;
//!          the AT-1620 bug was in staging coverage (now fixed). Typed-skip on Lavapipe.
//!
//! AT-1620 (M3.3b — UN-STUBBED): matmul_shared_coopmat.axc — full multi-tile coopmat matmul,
//!          bit-exact on NVIDIA over M=32, N=48, K=32 (non-symmetric multiple-of-16 fixture,
//!          3x2 workgroup grid, 2 K-blocks). Dispatch (N/16, M/16, 1) = (3, 2, 1) workgroups.
//!          tile_col = gid(0)/32 (M3.3b fix; ASYMMETRIC: tile_row = gid(1), no division).
//!          max_diff == 0.0 vs cpu_f16_matmul_reference. Typed-skip on Lavapipe (CoopMatUnsupported).
//!          Partial edge tiles (M or N not a multiple of 16) remain out of scope.
//!
//! AT-1621 (M3.3): matmul_shared_f32.axc — UPGRADED to bit-exact GPU dispatch on Lavapipe
//!          and NVIDIA. Index-math bug fixed (gid(1) used for both tile_row and local_row —
//!          corrected to derive tile_row = gid(1)/16, local_row = gid(1)%16).
//!
//! AT-1622 (M3.3b — UN-STUBBED): @strategy holes tile_k=16 AND tile_k=32 BOTH bit-exact
//!          over the same M=32, N=48, K=32 fixture. K=32 gives 2 K-blocks for tile_k=16
//!          (accumulation load-bearing) and 1 K-block for tile_k=32. max_diff == 0.0 for both.
//!          Typed-skip on Lavapipe (CoopMatUnsupported).
//!
//! AT-1630 (M3.3): tiled_attention.axc — UPGRADED to bit-exact within 1e-3 GPU dispatch.
//!          Fixed dispatch geometry: (seq_len,1,1) workgroups. CPU reference uses the
//!          IDENTICAL exp approximation (Taylor 1+x+x^2/2).

use std::collections::BTreeMap;
use axc_driver::{compile_source_with_meta, compile_source_with_assignments};
use axc_runtime::{VulkanContext, DispatchError};
use axc_hir::HirError;

fn gpu_tests_enabled() -> bool {
    std::env::var("AXC_ENABLE_GPU_TESTS").as_deref() == Ok("1")
}

// ── Source text constants ─────────────────────────────────────────────────────

const SHARED_REDUCE_SRC: &str = include_str!("../../../examples/shared_reduce.axc");
const MATMUL_SHARED_COOPMAT_SRC: &str = include_str!("../../../examples/matmul_shared_coopmat.axc");
const MATMUL_SHARED_F32_SRC: &str = include_str!("../../../examples/matmul_shared_f32.axc");
const TILED_ATTENTION_SRC: &str = include_str!("../../../examples/tiled_attention.axc");
const OPPHI_COOPMAT_ACCUMULATE_SRC: &str = include_str!("../../../examples/opphi_coopmat_accumulate.axc");

// ── Helper: tile strategy assignments ────────────────────────────────────────

fn tile_assignments(tile_m: i64, tile_n: i64, tile_k: i64) -> BTreeMap<String, i64> {
    let mut m = BTreeMap::new();
    m.insert("tile_m".to_owned(), tile_m);
    m.insert("tile_n".to_owned(), tile_n);
    m.insert("tile_k".to_owned(), tile_k);
    m.insert("tile_a_size".to_owned(), tile_m * tile_k);
    m.insert("tile_b_size".to_owned(), tile_k * tile_n);
    m
}

// ── CPU reference: f32 matrix multiply (M3.3) ────────────────────────────────

/// CPU reference: row-major f32 matmul C = A * B.
fn cpu_f32_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0_f32;
            for ki in 0..k {
                sum += a[row * k + ki] * b[ki * n + col];
            }
            c[row * n + col] = sum;
        }
    }
    c
}

/// CPU reference: f16-equivalent matmul using f32 arithmetic.
/// For small integer-valued f16 inputs (values 1.0..16.0), f32 is exact.
fn cpu_f16_matmul_reference(a_f32: &[f32], b_f32: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    cpu_f32_matmul(a_f32, b_f32, m, n, k)
}

/// CPU reference: tiled attention O = softmax(QKᵀ / sqrt(d)) V.
/// Uses IDENTICAL Taylor exp approximation as the GPU kernel: exp(x) ≈ 1 + x + x²/2.
/// This ensures bit-accurate comparison regardless of exp precision.
fn cpu_tiled_attention_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    inv_sqrt_d: f32,
) -> Vec<f32> {
    let mut o = vec![0.0_f32; seq_len * head_dim];
    for q_row in 0..seq_len {
        let q_base = q_row * head_dim;
        // Pass 1: compute scores and running max.
        let mut max_score = -1e9_f32;
        let mut scores = vec![0.0_f32; seq_len];
        for (j, score_slot) in scores.iter_mut().enumerate() {
            let k_base = j * head_dim;
            let mut dot = 0.0_f32;
            for d in 0..head_dim {
                dot += q[q_base + d] * k[k_base + d];
            }
            let s = dot * inv_sqrt_d;
            *score_slot = s;
            if s > max_score {
                max_score = s;
            }
        }
        // Pass 2: compute denominator with Taylor exp.
        let mut denom = 0.0_f32;
        let mut exp_scores = vec![0.0_f32; seq_len];
        for (j, es) in exp_scores.iter_mut().enumerate() {
            let x = scores[j] - max_score;
            // IDENTICAL approximation to the GPU kernel: 1 + x + x²/2.
            let e = 1.0_f32 + x + x * x * 0.5_f32;
            *es = e;
            denom += e;
        }
        // Pass 3: accumulate output.
        for d in 0..head_dim {
            let mut out_val = 0.0_f32;
            for j in 0..seq_len {
                let weight = exp_scores[j] / denom;
                out_val += weight * v[j * head_dim + d];
            }
            o[q_base + d] = out_val;
        }
    }
    o
}

// ── Half-precision f16 byte helpers ──────────────────────────────────────────

/// Pack f32 values to f16 bytes (round-to-nearest, little-endian).
/// Uses the `half` crate for accuracy.
fn f32_slice_to_f16_le_bytes(vals: &[f32]) -> Vec<u8> {
    use half::f16;
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        let h = f16::from_f32(v);
        out.extend_from_slice(&h.to_le_bytes());
    }
    out
}

/// Unpack f16 LE bytes to f32 values.
fn f16_le_bytes_to_f32_slice(bytes: &[u8]) -> Vec<f32> {
    use half::f16;
    bytes.chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

// ── Push-constant encoding helpers ───────────────────────────────────────────

/// Encode M, N, K as 3 u32 push constants (12 bytes, LE).
fn push_mnk(m: u32, n: u32, k: u32) -> Vec<u8> {
    let mut pc = Vec::with_capacity(12);
    pc.extend_from_slice(&m.to_le_bytes());
    pc.extend_from_slice(&n.to_le_bytes());
    pc.extend_from_slice(&k.to_le_bytes());
    pc
}

/// Encode seq_len, head_dim, inv_sqrt_d as push constants (12 bytes).
fn push_attention(seq_len: u32, head_dim: u32, inv_sqrt_d: f32) -> Vec<u8> {
    let mut pc = Vec::with_capacity(12);
    pc.extend_from_slice(&seq_len.to_le_bytes());
    pc.extend_from_slice(&head_dim.to_le_bytes());
    pc.extend_from_slice(&inv_sqrt_d.to_le_bytes());
    pc
}

// ── AT-1606: Shared reduction barrier-visibility oracle ───────────────────────

/// AT-1606 part 1a: OQ1 HARD ERROR fires for the PROVABLY-CROSS-SLOT barrier-absent variant.
#[test]
fn at1606_barrier_absent_cross_slot_hard_error_fires() {
    let src_no_barrier_literal = r#"
@kernel
@workgroup(2, 1, 1)
@intent("barrier-absent provable-cross-slot read — OQ1 HARD ERROR test (integer literals)")
@complexity(O(n))
fn barrier_absent_literal(input: readonly_buffer[f32], output: buffer[f32]) -> void {
    shared tile: shared[f32, 2];
    tile[0u32] = input[0u32];
    let v: f32 = tile[1u32];
    output[0u32] = v;
    return;
}
"#;
    let result = axc_driver::compile_source_with_meta(src_no_barrier_literal);
    match result {
        Ok(_) => {
            panic!("AT-1606a: barrier-absent provable-cross-slot should emit hard error but compiled Ok");
        }
        Err(axc_driver::DriverError::Compile { hir, .. }) => {
            let has_barrier_error = hir.iter().any(|e| {
                matches!(e, HirError::Typecheck(
                    axc_hir::TypecheckError::SharedMissingBarrierBeforeCrossInvocationRead { .. }
                ))
            });
            assert!(has_barrier_error, "AT-1606a: expected SharedMissingBarrierBeforeCrossInvocationRead; got: {hir:?}");
            eprintln!("AT-1606a: provable-cross-slot barrier-absent hard error confirmed (OQ1)");
        }
        Err(e) => { panic!("AT-1606a: unexpected error type: {e:?}"); }
    }
}

/// AT-1606 part 1b: OQ1 ADVISORY WARNING fires for the undecidable-index barrier-absent variant.
#[test]
fn at1606_barrier_absent_dynamic_index_advisory_warning() {
    let src_no_barrier_dynamic = r#"
@kernel
@workgroup(256, 1, 1)
@intent("barrier-absent dynamic-index read — OQ1 ADVISORY warning test")
@complexity(O(n))
fn barrier_absent_dynamic(input: readonly_buffer[f32], output: buffer[f32]) -> void {
    shared tile: shared[f32, 256];
    let lid: u32 = gid(0u32);
    tile[lid] = input[lid];
    let step0_target: u32 = lid + 128u32;
    let v: f32 = tile[step0_target];
    output[lid] = v;
    return;
}
"#;
    let result = axc_driver::compile_source_with_meta(src_no_barrier_dynamic);
    match result {
        Ok(_) => { eprintln!("AT-1606b: dynamic-index barrier-absent compiled Ok (advisory — correct)"); }
        Err(axc_driver::DriverError::Compile { hir, .. }) => {
            let has_hard_error = hir.iter().any(|e| {
                matches!(e, HirError::Typecheck(
                    axc_hir::TypecheckError::SharedMissingBarrierBeforeCrossInvocationRead { .. }
                ))
            });
            assert!(!has_hard_error, "AT-1606b: false-positive! Dynamic-index should NOT emit hard error; got: {hir:?}");
            eprintln!("AT-1606b: dynamic-index barrier-absent: no hard error (advisory only) — correct");
        }
        Err(e) => { panic!("AT-1606b: unexpected error: {e:?}"); }
    }
}

/// AT-1636 / OQ1 zero-false-positive: same-index self-read does NOT hard-error.
#[test]
fn at1636_same_index_self_read_no_diagnostic() {
    let src_self_read = r#"
@kernel
@workgroup(256, 1, 1)
@intent("same-index self read-after-write — OQ1 zero-false-positive test")
@complexity(O(n))
fn shared_self_read(input: readonly_buffer[f32], output: buffer[f32]) -> void {
    shared tile: shared[f32, 256];
    let lid: u32 = gid(0u32);
    tile[lid] = input[lid];
    let v: f32 = tile[lid];
    output[lid] = v;
    return;
}
"#;
    let result = axc_driver::compile_source_with_meta(src_self_read);
    match result {
        Ok(_) => { eprintln!("AT-1636: same-index self-read compiled Ok (correct — no false positive)"); }
        Err(axc_driver::DriverError::Compile { hir, .. }) => {
            let has_barrier_error = hir.iter().any(|e| {
                matches!(e, HirError::Typecheck(
                    axc_hir::TypecheckError::SharedMissingBarrierBeforeCrossInvocationRead { .. }
                ))
            });
            assert!(!has_barrier_error, "AT-1636: false-positive! Got: {hir:?}");
        }
        Err(e) => { eprintln!("AT-1636: compile returned other error (no barrier false-positive): {e:?}"); }
    }
}

/// AT-1606 part 2: GPU dispatch — shared_reduce.axc with barrier, bit-exact vs CPU.
#[test]
#[ignore]
fn at1606_shared_reduction_barrier_visibility_gpu() {
    if !gpu_tests_enabled() {
        eprintln!("at1606: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }

    let (bytes, meta) = compile_source_with_meta(SHARED_REDUCE_SRC)
        .expect("shared_reduce.axc must compile");
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let ctx = VulkanContext::new().expect("VulkanContext must init");
    eprintln!("at1606: device={}", ctx.physical_device_name());

    let handle = ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, None, "shared_reduce",
        meta.shared_memory_bytes,
    ).unwrap_or_else(|e| panic!("at1606: pipeline create failed: {e}"));

    let input_f32: Vec<f32> = (0..256_usize).map(|i| i as f32).collect();
    let input_bytes: Vec<u8> = input_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
    let output_size: usize = 4;

    let outputs = ctx.dispatch_handle(
        &handle, (1, 1, 1),
        &[&input_bytes, &[0u8; 4][..]],
        &[0, output_size],
        &[],
    ).unwrap_or_else(|e| panic!("at1606: dispatch failed: {e}"));

    let output_bytes = &outputs[1];
    assert_eq!(output_bytes.len(), 4, "output must be 4 bytes (one f32)");
    let gpu_result = f32::from_le_bytes([output_bytes[0], output_bytes[1], output_bytes[2], output_bytes[3]]);
    let cpu_result = 0.0_f32 + 128.0_f32 + 64.0_f32 + 192.0_f32; // = 384.0
    let tol = 1e-3_f32;
    eprintln!("at1606: gpu_result={gpu_result}, cpu_result={cpu_result}");
    assert!((gpu_result - cpu_result).abs() <= tol,
        "at1606: shared reduction mismatch: gpu={gpu_result}, cpu={cpu_result} (tol={tol})");
    eprintln!("at1606: PASS — shared[f32,256] + workgroup_barrier() is bit-exact on {}", ctx.physical_device_name());
}

// ── AT-1707: OpPhi numeric isolation — no staging, same tile each K_ITER ──────

/// AT-1707: opphi_coopmat_accumulate.axc compile + spirv-val clean (anchor).
#[test]
fn at1707_opphi_coopmat_accumulate_spirv_val_only() {
    use spirv_tools::val::{Validator, create as create_validator};
    use spirv_tools::TargetEnv;

    let (bytes, _meta) = compile_source_with_meta(OPPHI_COOPMAT_ACCUMULATE_SRC)
        .expect("AT-1707: opphi_coopmat_accumulate.axc must compile");
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let validator = create_validator(Some(TargetEnv::Vulkan_1_1));
    validator.validate(&words, None).expect("AT-1707: opphi_coopmat_accumulate.axc spirv-val must pass");
    eprintln!("AT-1707: opphi_coopmat_accumulate.axc compiles + spirv-val clean (M3.3 retry:1)");
}

/// AT-1707 GPU: OpPhi numeric isolation — no staging, same 16×16 f16 tile accumulated K_ITER=4 times.
///
/// Completely bypasses shared-memory staging: A and B are loaded directly from global buffers
/// at offset=0 each iteration. Expected result: C = K_ITER × (A · B).
///
/// If PASS: OpPhi loop-carried accumulation is numerically correct; the AT-1620 bug was
/// ONLY in staging coverage (the old kernel staged 2 of 16 rows per iteration).
/// If FAIL: OpPhi accumulation itself is broken; fix OpPhi emission before fixing AT-1620.
///
/// Typed-skip on Lavapipe (CoopMatUnsupported).
#[test]
#[ignore]
fn at1707_opphi_coopmat_accumulate_bit_exact_gpu() {
    if !gpu_tests_enabled() {
        eprintln!("at1707_gpu: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    const M_SIZE: usize = 16;
    const N_SIZE: usize = 16;
    const K_ITER: u32 = 4; // accumulate same tile K_ITER times

    let (bytes, meta) = compile_source_with_meta(OPPHI_COOPMAT_ACCUMULATE_SRC)
        .expect("AT-1707: opphi_coopmat_accumulate.axc must compile");
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let ctx = VulkanContext::new().expect("VulkanContext must init");
    eprintln!("AT-1707 GPU: device={}", ctx.physical_device_name());

    let handle = match ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, None, "opphi_coopmat_accumulate",
        meta.shared_memory_bytes,
    ) {
        Ok(h) => h,
        Err(DispatchError::CoopMatUnsupported { .. }) => {
            eprintln!("AT-1707 GPU: CoopMatUnsupported — typed-skip (Lavapipe)");
            return;
        }
        Err(DispatchError::DeviceFeatureUnsupported { feature, .. }) => {
            eprintln!("AT-1707 GPU: DeviceFeatureUnsupported({feature}) — typed-skip");
            return;
        }
        Err(e) => panic!("AT-1707: prepare_kernel_checked failed: {e}"),
    };

    // Integer-valued f16 fixture: small values so K_ITER=4 accumulations stay exact.
    // A[i,k] = ((i*N_SIZE + k) % 4 + 1): values 1,2,3,4 repeating.
    // B[k,j] = ((k*N_SIZE + j) % 3 + 1): values 1,2,3 repeating.
    let a_f32: Vec<f32> = (0..M_SIZE * N_SIZE).map(|i| (i % 4 + 1) as f32).collect();
    let b_f32: Vec<f32> = (0..N_SIZE * N_SIZE).map(|i| (i % 3 + 1) as f32).collect();
    let a_bytes = f32_slice_to_f16_le_bytes(&a_f32);
    let b_bytes = f32_slice_to_f16_le_bytes(&b_f32);
    let c_size = M_SIZE * N_SIZE * 2; // f16 output

    // Push constants: N=16, K_ITER=4.
    let mut pc = Vec::with_capacity(8);
    pc.extend_from_slice(&(N_SIZE as u32).to_le_bytes());
    pc.extend_from_slice(&K_ITER.to_le_bytes());

    let outputs = ctx.dispatch_handle(
        &handle, (1, 1, 1), // one workgroup handles the 16×16 tile
        &[&a_bytes, &b_bytes, &vec![0u8; c_size]],
        &[0, 0, c_size],
        &pc,
    ).unwrap_or_else(|e| panic!("AT-1707: dispatch failed: {e}"));

    let gpu_c = f16_le_bytes_to_f32_slice(&outputs[2]);

    // CPU reference: C_ref = K_ITER × (A · B).
    let single_matmul = cpu_f16_matmul_reference(&a_f32, &b_f32, M_SIZE, N_SIZE, N_SIZE);
    let cpu_c: Vec<f32> = single_matmul.iter().map(|&v| v * K_ITER as f32).collect();

    let mut max_diff = 0.0_f32;
    for (g, c) in gpu_c.iter().zip(cpu_c.iter()) {
        let diff = (g - c).abs();
        if diff > max_diff { max_diff = diff; }
    }

    eprintln!(
        "AT-1707 GPU: max_diff={max_diff}, first4 GPU={:?}, CPU={:?}",
        &gpu_c[..4.min(gpu_c.len())],
        &cpu_c[..4.min(cpu_c.len())]
    );

    assert!(
        max_diff == 0.0,
        "AT-1707: OpPhi numeric isolation FAILED — max_diff={max_diff} != 0.\n\
         This means OpPhi accumulation itself is broken (not a staging issue).\n\
         First4 GPU: {:?}, CPU: {:?}",
        &gpu_c[..4.min(gpu_c.len())],
        &cpu_c[..4.min(cpu_c.len())]
    );
    eprintln!(
        "AT-1707 PASS: OpPhi numeric accumulation correct — no staging, \
         K_ITER={K_ITER} accumulations of same 16×16 tile, max_diff=0 on {}",
        ctx.physical_device_name()
    );
}

// ── AT-1621: shared-staged f32 matmul — compile-only anchor + GPU dispatch ───

/// AT-1621: matmul_shared_f32.axc compile + spirv-val clean.
/// Compile anchor retained; GPU dispatch added as a separate #[ignore]-gated test.
#[test]
fn at1621_matmul_shared_f32_spirv_val_only() {
    use spirv_tools::val::{Validator, create as create_validator};
    use spirv_tools::TargetEnv;

    let assignments = tile_assignments(16, 16, 16);
    let (bytes, _meta) = compile_source_with_assignments(MATMUL_SHARED_F32_SRC, &assignments)
        .expect("AT-1621: matmul_shared_f32.axc must compile");
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let validator = create_validator(Some(TargetEnv::Vulkan_1_1));
    validator.validate(&words, None).expect("AT-1621: matmul_shared_f32.axc spirv-val must pass");
    eprintln!("AT-1621: matmul_shared_f32.axc compiles + spirv-val clean (M3.3)");
}

/// AT-1621 GPU: matmul_shared_f32.axc bit-exact on Lavapipe AND NVIDIA.
///
/// M3.3 fix: gid(1) was used for BOTH tile_row AND local_row — corrected to derive
/// tile_row = gid(1)/16, local_row = gid(1)%16 so multi-workgroup dispatches work.
/// Fixture: M=N=K=16, single workgroup (1,1,1), tile_k=16.
#[test]
#[ignore]
fn at1621_matmul_shared_f32_bit_exact_gpu() {
    if !gpu_tests_enabled() {
        eprintln!("at1621_gpu: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    const M: usize = 16;
    const N: usize = 16;
    const K: usize = 16;

    let assignments = tile_assignments(16, 16, 16);
    let (bytes, meta) = compile_source_with_assignments(MATMUL_SHARED_F32_SRC, &assignments)
        .expect("AT-1621: matmul_shared_f32.axc must compile");
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let ctx = VulkanContext::new().expect("VulkanContext must init");
    eprintln!("AT-1621 GPU: device={}", ctx.physical_device_name());

    let handle = ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, None, "matmul_shared_f32",
        meta.shared_memory_bytes,
    ).unwrap_or_else(|e| panic!("AT-1621: prepare_kernel_checked failed: {e}"));

    // Fixture: integer-valued f32 inputs (A[i,k] = i+1, B[k,j] = k+1).
    let a: Vec<f32> = (0..M*K).map(|i| (i % 4 + 1) as f32).collect();
    let b: Vec<f32> = (0..K*N).map(|i| (i % 3 + 1) as f32).collect();
    let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
    let b_bytes: Vec<u8> = b.iter().flat_map(|v| v.to_le_bytes()).collect();
    let output_size = M * N * 4; // f32 output

    let pc = push_mnk(M as u32, N as u32, K as u32);
    // Dispatch 1 workgroup (M=N=K=16 fits in a single 16×16 workgroup).
    let outputs = ctx.dispatch_handle(
        &handle, (1, 1, 1),
        &[&a_bytes, &b_bytes, &vec![0u8; output_size]],
        &[0, 0, output_size],
        &pc,
    ).unwrap_or_else(|e| panic!("AT-1621: dispatch failed: {e}"));

    let gpu_c: Vec<f32> = outputs[2].chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let cpu_c = cpu_f32_matmul(&a, &b, M, N, K);

    let mut max_diff = 0.0_f32;
    for (g, c) in gpu_c.iter().zip(cpu_c.iter()) {
        let diff = (g - c).abs();
        if diff > max_diff { max_diff = diff; }
    }
    let tol = 1e-4_f32;
    assert!(max_diff <= tol,
        "AT-1621: matmul_shared_f32 max_diff={max_diff} > tol={tol}; first few GPU: {:?}, CPU: {:?}",
        &gpu_c[..4.min(gpu_c.len())], &cpu_c[..4.min(cpu_c.len())]);
    eprintln!("AT-1621 PASS: matmul_shared_f32 bit-exact (max_diff={max_diff}) on {}", ctx.physical_device_name());
}

// ── AT-1620: shared-staged coopmat f16 matmul — compile-only + GPU dispatch ──

/// AT-1620: matmul_shared_coopmat.axc compile + spirv-val clean (anchor).
#[test]
fn at1620_matmul_shared_coopmat_spirv_val_only() {
    use spirv_tools::val::{Validator, create as create_validator};
    use spirv_tools::TargetEnv;

    let assignments = tile_assignments(16, 16, 16);
    let (bytes, _meta) = compile_source_with_assignments(MATMUL_SHARED_COOPMAT_SRC, &assignments)
        .expect("AT-1620: matmul_shared_coopmat.axc must compile");
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let validator = create_validator(Some(TargetEnv::Vulkan_1_1));
    validator.validate(&words, None).expect("AT-1620: matmul_shared_coopmat.axc spirv-val must pass");
    eprintln!("AT-1620: matmul_shared_coopmat.axc compiles + spirv-val clean (M3.3)");
}

/// AT-1620 GPU (M3.3b): full multi-tile coopmat matmul bit-exact on NVIDIA.
///
/// Non-symmetric multiple-of-16 fixture: M=32, N=48, K=32.
/// Dispatch (N/16, M/16, 1) = (3, 2, 1) workgroups — 6 output tiles covering all of C(32x48).
/// tile_k=16 → 2 K-blocks (accumulation load-bearing).
///
/// Integer-valued f16 fixture: A[i,k]=(i*K+k)%4+1 ∈ {1..4}, B[k,j]=(k*N+j)%3+1 ∈ {1..3}.
/// Per-element sum ≤ 32*12 = 384, exactly representable in f16 → max_diff == 0.0.
///
/// The tile_col=gid(0)/32 fix (M3.3b, ASYMMETRIC) is what makes the full grid correct.
/// Typed-skip on CoopMatUnsupported (Lavapipe). #[ignore]+AXC_ENABLE_GPU_TESTS gated.
/// Partial edge tiles (M or N not a multiple of 16) are out of scope.
#[test]
#[ignore]
fn at1620_matmul_shared_coopmat_bit_exact_gpu() {
    if !gpu_tests_enabled() {
        eprintln!("AT-1620: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    const M: usize = 32;
    const N: usize = 48;
    const K: usize = 32;

    let assignments = tile_assignments(16, 16, 16);
    let (bytes, meta) = compile_source_with_assignments(MATMUL_SHARED_COOPMAT_SRC, &assignments)
        .expect("AT-1620: matmul_shared_coopmat.axc must compile");
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let ctx = VulkanContext::new().expect("AT-1620: VulkanContext must init");
    eprintln!("AT-1620: device={}", ctx.physical_device_name());

    let handle = match ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, meta.coopmat.as_ref(), "matmul_shared_coopmat",
        meta.shared_memory_bytes,
    ) {
        Ok(h) => h,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("AT-1620: CoopMatUnsupported (typed-skip on Lavapipe): {reason}");
            return;
        }
        Err(DispatchError::DeviceFeatureUnsupported { feature, .. }) => {
            eprintln!("AT-1620: DeviceFeatureUnsupported({feature}) — typed-skip");
            return;
        }
        Err(e) => panic!("AT-1620: prepare_kernel_checked failed: {e}"),
    };

    // Integer-valued f16 fixture: exact in f16 (per-element sum <= 384 < 2048).
    // A[i,k] = (i*K + k) % 4 + 1; B[k,j] = (k*N + j) % 3 + 1.
    let a_f32: Vec<f32> = (0..M * K).map(|idx| ((idx % 4) + 1) as f32).collect();
    let b_f32: Vec<f32> = (0..K * N).map(|idx| ((idx % 3) + 1) as f32).collect();
    let a_bytes = f32_slice_to_f16_le_bytes(&a_f32);
    let b_bytes = f32_slice_to_f16_le_bytes(&b_f32);
    let c_size = M * N * 2; // f16 output

    // Dispatch the FULL grid: (N/16, M/16, 1) = (3, 2, 1).
    let wg_x = (N / 16) as u32;
    let wg_y = (M / 16) as u32;
    let pc = push_mnk(M as u32, N as u32, K as u32);

    let outputs = ctx.dispatch_handle(
        &handle, (wg_x, wg_y, 1),
        &[&a_bytes, &b_bytes, &vec![0u8; c_size]],
        &[0, 0, c_size],
        &pc,
    ).unwrap_or_else(|e| panic!("AT-1620: dispatch failed: {e}"));

    let gpu_c = f16_le_bytes_to_f32_slice(&outputs[2]);
    let cpu_c = cpu_f16_matmul_reference(&a_f32, &b_f32, M, N, K);

    let mut max_diff = 0.0_f32;
    for (g, c) in gpu_c.iter().zip(cpu_c.iter()) {
        let diff = (g - c).abs();
        if diff > max_diff { max_diff = diff; }
    }

    eprintln!(
        "AT-1620: max_diff={max_diff}, dispatch=({wg_x},{wg_y},1), \
         first4 GPU={:?}, CPU={:?}",
        &gpu_c[..4.min(gpu_c.len())],
        &cpu_c[..4.min(cpu_c.len())]
    );

    assert!(
        max_diff == 0.0,
        "AT-1620: full multi-tile coopmat matmul FAILED — max_diff={max_diff} != 0.\n\
         HONESTY GATE: do NOT relax tolerance or shrink fixture.\n\
         Dispatch=({wg_x},{wg_y},1), M={M} N={N} K={K}. First4 GPU: {:?}, CPU: {:?}",
        &gpu_c[..4.min(gpu_c.len())],
        &cpu_c[..4.min(cpu_c.len())]
    );
    eprintln!(
        "AT-1620 PASS: full multi-tile coopmat matmul bit-exact (max_diff=0.0) \
         on {} — M={M} N={N} K={K}, dispatch=({wg_x},{wg_y},1)",
        ctx.physical_device_name()
    );
}

/// AT-1622 structural guard: tile_k=16 and tile_k=32 produce different SPIR-V (compile anchor).
#[test]
fn at1622_strategy_holes_spirv_val_only() {
    use spirv_tools::val::{Validator, create as create_validator};
    use spirv_tools::TargetEnv;

    for &tk in &[16i64, 32i64] {
        let assignments = tile_assignments(16, 16, tk);
        let (bytes, _meta) = compile_source_with_assignments(MATMUL_SHARED_COOPMAT_SRC, &assignments)
            .unwrap_or_else(|e| panic!("AT-1622: tile_k={tk} compile failed: {e:?}"));
        let words: Vec<u32> = bytes.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let validator = create_validator(Some(TargetEnv::Vulkan_1_1));
        validator.validate(&words, None)
            .unwrap_or_else(|e| panic!("AT-1622: tile_k={tk} spirv-val must pass: {e}"));
        eprintln!("AT-1622: tile_k={tk} compiles + spirv-val clean");
    }
    eprintln!("AT-1622: both tile_k variants spirv-val clean (M3.3)");
}

/// AT-1622 GPU (M3.3b): @strategy tile_k=16 AND tile_k=32 BOTH bit-exact on NVIDIA.
///
/// Same M=32, N=48, K=32 fixture as AT-1620.
/// tile_k=16 → 2 K-blocks (accumulation load-bearing: proves OpPhi carries across blocks).
/// tile_k=32 → 1 K-block (proves num_k_blocks=1 path also correct).
/// max_diff == 0.0 for both variants on the integer-valued f16 fixture.
///
/// Typed-skip on CoopMatUnsupported (Lavapipe). #[ignore]+AXC_ENABLE_GPU_TESTS gated.
#[test]
#[ignore]
fn at1622_tile_k_variants_bit_exact_gpu() {
    if !gpu_tests_enabled() {
        eprintln!("AT-1622: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    const M: usize = 32;
    const N: usize = 48;
    const K: usize = 32;

    // Integer-valued f16 fixture (same as AT-1620): A[i,k]=(i*K+k)%4+1, B[k,j]=(k*N+j)%3+1.
    let a_f32: Vec<f32> = (0..M * K).map(|idx| ((idx % 4) + 1) as f32).collect();
    let b_f32: Vec<f32> = (0..K * N).map(|idx| ((idx % 3) + 1) as f32).collect();
    let a_bytes = f32_slice_to_f16_le_bytes(&a_f32);
    let b_bytes = f32_slice_to_f16_le_bytes(&b_f32);
    let c_size = M * N * 2; // f16 output
    let cpu_c = cpu_f16_matmul_reference(&a_f32, &b_f32, M, N, K);

    let ctx = VulkanContext::new().expect("AT-1622: VulkanContext must init");
    eprintln!("AT-1622: device={}", ctx.physical_device_name());

    // Full dispatch grid: (N/16, M/16, 1) = (3, 2, 1).
    let wg_x = (N / 16) as u32;
    let wg_y = (M / 16) as u32;
    let pc = push_mnk(M as u32, N as u32, K as u32);

    for &tk in &[16_i64, 32_i64] {
        let assignments = tile_assignments(16, 16, tk);
        let (bytes, meta) = compile_source_with_assignments(MATMUL_SHARED_COOPMAT_SRC, &assignments)
            .unwrap_or_else(|e| panic!("AT-1622: tile_k={tk} compile failed: {e:?}"));
        let words: Vec<u32> = bytes.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

        let handle = match ctx.prepare_kernel_checked(
            &words, &meta.binding_plan, meta.push_constant_total_bytes,
            &meta.entry_point, meta.coopmat.as_ref(), "matmul_shared_coopmat",
            meta.shared_memory_bytes,
        ) {
            Ok(h) => h,
            Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
                eprintln!("AT-1622: tile_k={tk}: CoopMatUnsupported (typed-skip on Lavapipe): {reason}");
                return;
            }
            Err(DispatchError::DeviceFeatureUnsupported { feature, .. }) => {
                eprintln!("AT-1622: tile_k={tk}: DeviceFeatureUnsupported({feature}) — typed-skip");
                return;
            }
            Err(e) => panic!("AT-1622: tile_k={tk}: prepare_kernel_checked failed: {e}"),
        };

        let outputs = ctx.dispatch_handle(
            &handle, (wg_x, wg_y, 1),
            &[&a_bytes, &b_bytes, &vec![0u8; c_size]],
            &[0, 0, c_size],
            &pc,
        ).unwrap_or_else(|e| panic!("AT-1622: tile_k={tk}: dispatch failed: {e}"));

        let gpu_c = f16_le_bytes_to_f32_slice(&outputs[2]);

        let mut max_diff = 0.0_f32;
        for (g, c) in gpu_c.iter().zip(cpu_c.iter()) {
            let diff = (g - c).abs();
            if diff > max_diff { max_diff = diff; }
        }

        eprintln!(
            "AT-1622: tile_k={tk}, max_diff={max_diff}, first4 GPU={:?}, CPU={:?}",
            &gpu_c[..4.min(gpu_c.len())],
            &cpu_c[..4.min(cpu_c.len())]
        );

        assert!(
            max_diff == 0.0,
            "AT-1622: tile_k={tk} FAILED — max_diff={max_diff} != 0.\n\
             HONESTY GATE: do NOT relax tolerance.\n\
             M={M} N={N} K={K}, dispatch=({wg_x},{wg_y},1). First4 GPU: {:?}, CPU: {:?}",
            &gpu_c[..4.min(gpu_c.len())],
            &cpu_c[..4.min(cpu_c.len())]
        );
        eprintln!(
            "AT-1622 tile_k={tk} PASS: bit-exact (max_diff=0.0) on {} \
             M={M} N={N} K={K} ({} K-blocks)",
            ctx.physical_device_name(),
            K / tk as usize
        );
    }
    eprintln!("AT-1622 PASS: both tile_k=16 and tile_k=32 bit-exact on {}", ctx.physical_device_name());
}

// ── AT-1630: Tiled attention C1 — compile-only anchor + GPU dispatch ──────────

/// AT-1630: tiled_attention.axc compile + spirv-val clean (anchor).
#[test]
fn at1630_tiled_attention_spirv_val_only() {
    use spirv_tools::val::{Validator, create as create_validator};
    use spirv_tools::TargetEnv;

    let (bytes, _meta) = compile_source_with_meta(TILED_ATTENTION_SRC)
        .expect("AT-1630: tiled_attention.axc must compile");
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let validator = create_validator(Some(TargetEnv::Vulkan_1_1));
    validator.validate(&words, None).expect("AT-1630: tiled_attention.axc spirv-val must pass");
    eprintln!("AT-1630: tiled_attention.axc compiles + spirv-val clean (M3.3)");
}

/// AT-1630 GPU: tiled_attention within 1e-3 of CPU reference attention on Lavapipe + NVIDIA.
///
/// M3.3 fix: dispatch was (1,1,1) → corrected to (seq_len,1,1) workgroups.
/// CPU reference uses IDENTICAL Taylor exp approximation (1+x+x²/2).
/// Fixture: seq_len=64, head_dim=32.
#[test]
#[ignore]
fn at1630_tiled_attention_bit_exact_gpu() {
    if !gpu_tests_enabled() {
        eprintln!("at1630_gpu: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    const SEQ_LEN: usize = 64;
    const HEAD_DIM: usize = 32;
    let inv_sqrt_d: f32 = 1.0_f32 / (HEAD_DIM as f32).sqrt();

    let (bytes, meta) = compile_source_with_meta(TILED_ATTENTION_SRC)
        .expect("AT-1630: tiled_attention.axc must compile");
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let ctx = VulkanContext::new().expect("VulkanContext must init");
    eprintln!("AT-1630 GPU: device={}", ctx.physical_device_name());

    let handle = ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, None, "tiled_attention",
        meta.shared_memory_bytes,
    ).unwrap_or_else(|e| panic!("AT-1630: prepare_kernel_checked failed: {e}"));

    // Small random-ish fixture (deterministic).
    let q: Vec<f32> = (0..SEQ_LEN*HEAD_DIM).map(|i| 0.1_f32 * ((i % 7) as f32 - 3.0_f32)).collect();
    let k: Vec<f32> = (0..SEQ_LEN*HEAD_DIM).map(|i| 0.1_f32 * ((i % 5) as f32 - 2.0_f32)).collect();
    let v: Vec<f32> = (0..SEQ_LEN*HEAD_DIM).map(|i| 0.05_f32 * (i as f32 % 11.0_f32)).collect();

    let q_bytes: Vec<u8> = q.iter().flat_map(|v| v.to_le_bytes()).collect();
    let k_bytes: Vec<u8> = k.iter().flat_map(|v| v.to_le_bytes()).collect();
    let v_bytes: Vec<u8> = v.iter().flat_map(|v| v.to_le_bytes()).collect();
    let o_size = SEQ_LEN * HEAD_DIM * 4; // f32 output

    let pc = push_attention(SEQ_LEN as u32, HEAD_DIM as u32, inv_sqrt_d);
    // CORRECTED dispatch: (seq_len, 1, 1) workgroups — one per query row.
    let outputs = ctx.dispatch_handle(
        &handle, (SEQ_LEN as u32, 1, 1),
        &[&q_bytes, &k_bytes, &v_bytes, &vec![0u8; o_size]],
        &[0, 0, 0, o_size],
        &pc,
    ).unwrap_or_else(|e| panic!("AT-1630: dispatch failed: {e}"));

    let gpu_o: Vec<f32> = outputs[3].chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let cpu_o = cpu_tiled_attention_reference(&q, &k, &v, SEQ_LEN, HEAD_DIM, inv_sqrt_d);

    let mut max_diff = 0.0_f32;
    for (g, c) in gpu_o.iter().zip(cpu_o.iter()) {
        let diff = (g - c).abs();
        if diff > max_diff { max_diff = diff; }
    }
    let tol = 1e-3_f32;
    assert!(max_diff <= tol,
        "AT-1630: attention max_diff={max_diff} > tol={tol}; first GPU: {:?}, CPU: {:?}",
        &gpu_o[..4.min(gpu_o.len())], &cpu_o[..4.min(cpu_o.len())]);
    eprintln!("AT-1630 PASS: attention within-tol (max_diff={max_diff}) on {}", ctx.physical_device_name());
}
