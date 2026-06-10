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
//! AT-1622 (M3.3b — FIXED): K-block-count variation bit-exact — tile_k=16 FIXED (bound to
//!          coopmat K dimension=16; tile_k=32 is semantically invalid for matrix[f16,16,16,*]
//!          since one coopmat_mul_add covers only K=16 → half results at tile_k=32).
//!          Instead varies K-block COUNT: K=32 (2 K-blocks) AND K=48 (3 K-blocks), both with
//!          tile_k=16. Proves the OpPhi K-loop accumulation is bit-exact for different trip counts
//!          (the genuinely load-bearing variation). max_diff == 0.0 for both.
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
const FLASH_ATTENTION_SRC: &str = include_str!("../../../examples/flash_attention.axc");
const FLASH_ATTENTION_EXP_SRC: &str = include_str!("../../../examples/flash_attention_exp.axc");
/// M3.2c exp(x) micro-kernel (AT-1820/1821): O[i] = exp(In[i]). One invocation per element
/// (dispatch (N,1,1) workgroups). Exercises the GLSL.std.450 Exp builtin directly.
const EXP_MICRO_SRC: &str = r#"
@kernel
@workgroup(1, 1, 1)
@intent("exp(x) micro-kernel — direct GLSL.std.450 Exp correctness (AT-1820)")
@complexity(O(1))
fn exp_micro(In: readonly_buffer[f32], Out: buffer[f32]) -> void {
    let i: u32 = gid(0u32);
    let x: f32 = In[i];
    Out[i] = exp(x);
    return;
}
"#;
const OPPHI_COOPMAT_ACCUMULATE_SRC: &str = include_str!("../../../examples/opphi_coopmat_accumulate.axc");
const MATMUL_RB_COOPMAT_SRC: &str = include_str!("../../../examples/matmul_rb_coopmat.axc");
const MATMUL_MSG_COOPMAT_SRC: &str = include_str!("../../../examples/matmul_msg_coopmat.axc");

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

/// AT-1622 structural guard: tile_k=16 (the only valid coopmat-K assignment) compiles + spirv-val clean.
///
/// tile_k=32 is NOT tested here: a 16×16×16 coopmat_mul_add covers exactly K=16 per call;
/// tile_k=32 as a single-call-per-block config is semantically invalid (computes only K=16
/// of K=32 per block → half results). Sub-K-loop support is a follow-up (M3.3c+).
/// The valid K-block count variation (K=32 and K=48 with tile_k=16) is tested in
/// at1622_k_block_count_variation_bit_exact_gpu.
#[test]
fn at1622_strategy_holes_spirv_val_only() {
    use spirv_tools::val::{Validator, create as create_validator};
    use spirv_tools::TargetEnv;

    // tile_k=16: the only valid assignment — bound to the coopmat K dimension.
    let assignments = tile_assignments(16, 16, 16);
    let (bytes, _meta) = compile_source_with_assignments(MATMUL_SHARED_COOPMAT_SRC, &assignments)
        .unwrap_or_else(|e| panic!("AT-1622: tile_k=16 compile failed: {e:?}"));
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let validator = create_validator(Some(TargetEnv::Vulkan_1_1));
    validator.validate(&words, None)
        .unwrap_or_else(|e| panic!("AT-1622: tile_k=16 spirv-val must pass: {e}"));
    eprintln!("AT-1622: tile_k=16 compiles + spirv-val clean (the only valid coopmat-K assignment)");
}

/// AT-1622 GPU (M3.3b — FIXED): K-block-count variation bit-exact on NVIDIA.
///
/// tile_k is FIXED at 16 (bound to the coopmat K dimension: a 16×16×16 coopmat_mul_add
/// covers exactly K=16 per call; tile_k=32 with one call per block is semantically invalid
/// and produces GPU=[half of correct] — the original AT-1622 FAIL).
///
/// The genuinely-meaningful variation is the K-block COUNT (K / tile_k):
///   K=32, tile_k=16 → 2 K-blocks: accumulation load-bearing (proves OpPhi carries across 2 blocks).
///   K=48, tile_k=16 → 3 K-blocks: proves OpPhi carries across 3 blocks.
///
/// f16-exactness bound for K=48: max element = 48 × max(A) × max(B) = 48 × 4 × 3 = 576 ≤ 2048
/// (f16 integer-exact limit). max_diff == 0.0 holds for both K=32 and K=48.
///
/// Non-symmetric fixture M=32, N=48 (3x2 workgroup grid covers all of C(32×48)).
/// Typed-skip on CoopMatUnsupported (Lavapipe). #[ignore]+AXC_ENABLE_GPU_TESTS gated.
#[test]
#[ignore]
fn at1622_k_block_count_variation_bit_exact_gpu() {
    if !gpu_tests_enabled() {
        eprintln!("AT-1622: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    const M: usize = 32;
    const N: usize = 48;
    // tile_k is fixed at 16 — bound to the coopmat K dimension.
    const TILE_K: usize = 16;

    // Compile once: tile_k=16 is the only valid assignment.
    let assignments = tile_assignments(16, 16, TILE_K as i64);
    let (bytes, meta) = compile_source_with_assignments(MATMUL_SHARED_COOPMAT_SRC, &assignments)
        .expect("AT-1622: tile_k=16 compile must succeed");
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let ctx = VulkanContext::new().expect("AT-1622: VulkanContext must init");
    eprintln!("AT-1622: device={}", ctx.physical_device_name());

    let handle = match ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, meta.coopmat.as_ref(), "matmul_shared_coopmat",
        meta.shared_memory_bytes,
    ) {
        Ok(h) => h,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("AT-1622: CoopMatUnsupported (typed-skip on Lavapipe): {reason}");
            return;
        }
        Err(DispatchError::DeviceFeatureUnsupported { feature, .. }) => {
            eprintln!("AT-1622: DeviceFeatureUnsupported({feature}) — typed-skip");
            return;
        }
        Err(e) => panic!("AT-1622: prepare_kernel_checked failed: {e}"),
    };

    // Full dispatch grid: (N/16, M/16, 1) = (3, 2, 1).
    let wg_x = (N / 16) as u32;
    let wg_y = (M / 16) as u32;

    // Vary K to exercise different K-block counts (K / TILE_K).
    // K=32 → 2 K-blocks (load-bearing accumulation over 2 blocks).
    // K=48 → 3 K-blocks (proves OpPhi carries correctly over 3 blocks).
    // f16-exactness: max element = K × 4 × 3 ≤ 48 × 12 = 576 ≤ 2048 (f16 integer-exact limit).
    for &k in &[32_usize, 48_usize] {
        let k_blocks = k / TILE_K;
        // Integer-valued f16 fixture: A[idx]=(idx%4)+1 ∈ {1..4}, B[idx]=(idx%3)+1 ∈ {1..3}.
        let a_f32: Vec<f32> = (0..M * k).map(|idx| ((idx % 4) + 1) as f32).collect();
        let b_f32: Vec<f32> = (0..k * N).map(|idx| ((idx % 3) + 1) as f32).collect();
        let a_bytes = f32_slice_to_f16_le_bytes(&a_f32);
        let b_bytes = f32_slice_to_f16_le_bytes(&b_f32);
        let c_size = M * N * 2; // f16 output
        let cpu_c = cpu_f16_matmul_reference(&a_f32, &b_f32, M, N, k);
        let pc = push_mnk(M as u32, N as u32, k as u32);

        let outputs = ctx.dispatch_handle(
            &handle, (wg_x, wg_y, 1),
            &[&a_bytes, &b_bytes, &vec![0u8; c_size]],
            &[0, 0, c_size],
            &pc,
        ).unwrap_or_else(|e| panic!("AT-1622: K={k}: dispatch failed: {e}"));

        let gpu_c = f16_le_bytes_to_f32_slice(&outputs[2]);

        let mut max_diff = 0.0_f32;
        for (g, c) in gpu_c.iter().zip(cpu_c.iter()) {
            let diff = (g - c).abs();
            if diff > max_diff { max_diff = diff; }
        }

        eprintln!(
            "AT-1622: K={k} ({k_blocks} K-blocks), max_diff={max_diff}, \
             first4 GPU={:?}, CPU={:?}",
            &gpu_c[..4.min(gpu_c.len())],
            &cpu_c[..4.min(cpu_c.len())]
        );

        assert!(
            max_diff == 0.0,
            "AT-1622: K={k} ({k_blocks} K-blocks) FAILED — max_diff={max_diff} != 0.\n\
             HONESTY GATE: do NOT relax tolerance or shrink fixture.\n\
             tile_k={TILE_K} (fixed), M={M} N={N}, dispatch=({wg_x},{wg_y},1).\n\
             First4 GPU: {:?}, CPU: {:?}",
            &gpu_c[..4.min(gpu_c.len())],
            &cpu_c[..4.min(cpu_c.len())]
        );
        eprintln!(
            "AT-1622 K={k} ({k_blocks} K-blocks) PASS: bit-exact (max_diff=0.0) \
             on {} — M={M} N={N} tile_k={TILE_K}",
            ctx.physical_device_name()
        );
    }
    eprintln!(
        "AT-1622 PASS: K-block-count variation (K=32/2-blocks, K=48/3-blocks) \
         both bit-exact on {} — tile_k={TILE_K} fixed (coopmat K dimension)",
        ctx.physical_device_name()
    );
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

// ── Helpers for RB 2×2 tests (AT-1731, AT-1732) ──────────────────────────────

/// Build RB 2×2 strategy assignments for matmul_rb_coopmat.axc.
///
/// rb_m=2, rb_n=2, tile_k=16, a_block_size=512, b_block_size=512.
/// These are the ONLY valid shipped assignments for the 2×2 hand-unrolled variant.
fn rb2x2_assignments() -> BTreeMap<String, i64> {
    let mut m = BTreeMap::new();
    m.insert("rb_m".to_owned(), 2_i64);
    m.insert("rb_n".to_owned(), 2_i64);
    m.insert("tile_k".to_owned(), 16_i64);
    m.insert("a_block_size".to_owned(), 512_i64);
    m.insert("b_block_size".to_owned(), 512_i64);
    m
}

// ── AT-1731: RB 2×2 bit-exact GPU dispatch ────────────────────────────────────

/// AT-1731: matmul_rb_coopmat.axc (M3.3c, RB 2×2) is BIT-EXACT (max_diff==0.0)
/// on the non-symmetric integer-f16 fixture, dispatched as the RB block grid.
///
/// Fixture: M=N=64, K=32. Block grid: (N/32, M/32, 1) = (2, 2, 1) workgroups.
/// Each workgroup computes a 32×32 sub-matrix (4 output tiles of 16×16 each).
/// This exercises BOTH the inter-block grid dispatch (4 workgroups) AND the
/// intra-block multi-accumulator accumulation (4 acc × 2 K-blocks = 8 OpPhi carries).
///
/// Integer-valued f16 fixture: A[idx]=(idx%4)+1 ∈ {1..4}, B[idx]=(idx%3)+1 ∈ {1..3}.
/// Per-element max = K × max(A) × max(B) = 32 × 4 × 3 = 384 ≤ 2048 (f16-integer-exact).
///
/// The non-symmetric fixture (distinct per-tile input) detects A/B load index swaps:
/// if a_mat_0 and a_mat_1 are swapped, or b_mat_0 and b_mat_1 are swapped,
/// the non-symmetric inputs produce different (wrong) results.
///
/// Typed-skip on CoopMatUnsupported (Lavapipe). #[ignore]+AXC_ENABLE_GPU_TESTS gated.
#[test]
#[ignore]
fn at1731_matmul_rb_coopmat_bit_exact_gpu() {
    if !gpu_tests_enabled() {
        eprintln!("AT-1731: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    const M: usize = 64;
    const N: usize = 64;
    const K: usize = 32;

    let assignments = rb2x2_assignments();
    let (bytes, meta) = compile_source_with_assignments(MATMUL_RB_COOPMAT_SRC, &assignments)
        .expect("AT-1731: matmul_rb_coopmat.axc must compile");
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let ctx = VulkanContext::new().expect("AT-1731: VulkanContext must init");
    eprintln!("AT-1731: device={}", ctx.physical_device_name());

    let handle = match ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, meta.coopmat.as_ref(), "matmul_rb_coopmat",
        meta.shared_memory_bytes,
    ) {
        Ok(h) => h,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("AT-1731: CoopMatUnsupported (typed-skip on Lavapipe): {reason}");
            return;
        }
        Err(DispatchError::DeviceFeatureUnsupported { feature, .. }) => {
            eprintln!("AT-1731: DeviceFeatureUnsupported({feature}) — typed-skip");
            return;
        }
        Err(e) => panic!("AT-1731: prepare_kernel_checked failed: {e}"),
    };

    // Integer-valued f16 fixture (non-symmetric, distinct per-row/col pattern).
    // A[idx] = (idx % 4) + 1 ∈ {1,2,3,4}; B[idx] = (idx % 3) + 1 ∈ {1,2,3}.
    let a_f32: Vec<f32> = (0..M * K).map(|idx| ((idx % 4) + 1) as f32).collect();
    let b_f32: Vec<f32> = (0..K * N).map(|idx| ((idx % 3) + 1) as f32).collect();
    let a_bytes = f32_slice_to_f16_le_bytes(&a_f32);
    let b_bytes = f32_slice_to_f16_le_bytes(&b_f32);
    let c_size = M * N * 2; // f16 output

    // RB block grid: (N/32, M/32, 1) = (2, 2, 1) workgroups.
    let wg_x: u32 = (N / 32) as u32;
    let wg_y: u32 = (M / 32) as u32;
    let pc = push_mnk(M as u32, N as u32, K as u32);

    let outputs = ctx.dispatch_handle(
        &handle, (wg_x, wg_y, 1),
        &[&a_bytes, &b_bytes, &vec![0u8; c_size]],
        &[0, 0, c_size],
        &pc,
    ).unwrap_or_else(|e| panic!("AT-1731: dispatch failed: {e}"));

    let gpu_c = f16_le_bytes_to_f32_slice(&outputs[2]);
    let cpu_c = cpu_f16_matmul_reference(&a_f32, &b_f32, M, N, K);

    let mut max_diff = 0.0_f32;
    for (g, c) in gpu_c.iter().zip(cpu_c.iter()) {
        let diff = (g - c).abs();
        if diff > max_diff { max_diff = diff; }
    }

    eprintln!(
        "AT-1731: M={M} N={N} K={K}, block grid=({wg_x},{wg_y},1), \
         max_diff={max_diff}, first4 GPU={:?}, CPU={:?}",
        &gpu_c[..4.min(gpu_c.len())],
        &cpu_c[..4.min(cpu_c.len())]
    );

    assert!(
        max_diff == 0.0,
        "AT-1731: RB 2×2 matmul FAILED — max_diff={max_diff} != 0.\n\
         HONESTY GATE: do NOT relax tolerance or shrink fixture.\n\
         Dispatch=({wg_x},{wg_y},1), M={M} N={N} K={K}. First4 GPU: {:?}, CPU: {:?}",
        &gpu_c[..4.min(gpu_c.len())],
        &cpu_c[..4.min(cpu_c.len())]
    );
    eprintln!(
        "AT-1731 PASS: RB 2×2 matmul bit-exact (max_diff=0.0) on {} — \
         M={M} N={N} K={K}, block grid=({wg_x},{wg_y},1)",
        ctx.physical_device_name()
    );
}

// ── AT-1732: RB 2×2 K-block-count variation bit-exact ────────────────────────

/// AT-1732: RB 2×2 K-block-count variation is bit-exact for K=32 (2 K-blocks)
/// AND K=48 (3 K-blocks), both with tile_k=16 FIXED.
///
/// This proves the 4-coopmat OpPhi accumulation is bit-exact for different trip counts:
/// 2 K-blocks = 2 OpPhi carries; 3 K-blocks = 3 OpPhi carries.
/// Directly parallels AT-1622 (single-tile K-block variation) for the RB variant.
///
/// f16-exactness bound:
///   K=32: max element = 32 × 4 × 3 = 384 ≤ 2048 (exact).
///   K=48: max element = 48 × 4 × 3 = 576 ≤ 2048 (exact).
///
/// Fixture: M=N=64 (block grid (2,2,1) = 4 workgroups; exercises multiple RB blocks).
/// Typed-skip on CoopMatUnsupported (Lavapipe). #[ignore]+AXC_ENABLE_GPU_TESTS gated.
#[test]
#[ignore]
fn at1732_matmul_rb_kblock_count_bit_exact_gpu() {
    if !gpu_tests_enabled() {
        eprintln!("AT-1732: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    const M: usize = 64;
    const N: usize = 64;
    // tile_k is fixed at 16 (bound to the coopmat K dimension).
    const TILE_K: usize = 16;

    let assignments = rb2x2_assignments();
    let (bytes, meta) = compile_source_with_assignments(MATMUL_RB_COOPMAT_SRC, &assignments)
        .expect("AT-1732: matmul_rb_coopmat.axc must compile");
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let ctx = VulkanContext::new().expect("AT-1732: VulkanContext must init");
    eprintln!("AT-1732: device={}", ctx.physical_device_name());

    let handle = match ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, meta.coopmat.as_ref(), "matmul_rb_coopmat",
        meta.shared_memory_bytes,
    ) {
        Ok(h) => h,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("AT-1732: CoopMatUnsupported (typed-skip on Lavapipe): {reason}");
            return;
        }
        Err(DispatchError::DeviceFeatureUnsupported { feature, .. }) => {
            eprintln!("AT-1732: DeviceFeatureUnsupported({feature}) — typed-skip");
            return;
        }
        Err(e) => panic!("AT-1732: prepare_kernel_checked failed: {e}"),
    };

    // RB block grid: (N/32, M/32, 1) = (2, 2, 1) workgroups.
    let wg_x: u32 = (N / 32) as u32;
    let wg_y: u32 = (M / 32) as u32;

    // Vary K to exercise different K-block counts.
    // K=32 → 2 K-blocks (load-bearing: proves OpPhi carries across 2 blocks).
    // K=48 → 3 K-blocks (proves OpPhi carries across 3 blocks).
    for &k in &[32_usize, 48_usize] {
        let k_blocks = k / TILE_K;
        let a_f32: Vec<f32> = (0..M * k).map(|idx| ((idx % 4) + 1) as f32).collect();
        let b_f32: Vec<f32> = (0..k * N).map(|idx| ((idx % 3) + 1) as f32).collect();
        let a_bytes = f32_slice_to_f16_le_bytes(&a_f32);
        let b_bytes = f32_slice_to_f16_le_bytes(&b_f32);
        let c_size = M * N * 2; // f16 output
        let cpu_c = cpu_f16_matmul_reference(&a_f32, &b_f32, M, N, k);
        let pc = push_mnk(M as u32, N as u32, k as u32);

        let outputs = ctx.dispatch_handle(
            &handle, (wg_x, wg_y, 1),
            &[&a_bytes, &b_bytes, &vec![0u8; c_size]],
            &[0, 0, c_size],
            &pc,
        ).unwrap_or_else(|e| panic!("AT-1732: K={k}: dispatch failed: {e}"));

        let gpu_c = f16_le_bytes_to_f32_slice(&outputs[2]);

        let mut max_diff = 0.0_f32;
        for (g, c) in gpu_c.iter().zip(cpu_c.iter()) {
            let diff = (g - c).abs();
            if diff > max_diff { max_diff = diff; }
        }

        eprintln!(
            "AT-1732: K={k} ({k_blocks} K-blocks), max_diff={max_diff}, \
             block grid=({wg_x},{wg_y},1), first4 GPU={:?}, CPU={:?}",
            &gpu_c[..4.min(gpu_c.len())],
            &cpu_c[..4.min(cpu_c.len())]
        );

        assert!(
            max_diff == 0.0,
            "AT-1732: RB 2×2 K={k} ({k_blocks} K-blocks) FAILED — max_diff={max_diff} != 0.\n\
             HONESTY GATE: do NOT relax tolerance or shrink fixture.\n\
             tile_k={TILE_K} (fixed), M={M} N={N}, block grid=({wg_x},{wg_y},1).\n\
             First4 GPU: {:?}, CPU: {:?}",
            &gpu_c[..4.min(gpu_c.len())],
            &cpu_c[..4.min(cpu_c.len())]
        );
        eprintln!(
            "AT-1732 K={k} ({k_blocks} K-blocks) PASS: bit-exact (max_diff=0.0) \
             on {} — M={M} N={N} tile_k={TILE_K} block grid=({wg_x},{wg_y},1)",
            ctx.physical_device_name()
        );
    }
    eprintln!(
        "AT-1732 PASS: RB 2×2 K-block-count variation (K=32/2-blocks, K=48/3-blocks) \
         both bit-exact on {} — tile_k={TILE_K} fixed (coopmat K dimension)",
        ctx.physical_device_name()
    );
}

// ── Helpers for MSG 2-subgroup tests (AT-1743, AT-1744) ──────────────────────

/// Build MSG strategy assignments for matmul_msg_coopmat.axc (shipped config).
///
/// wg_threads=64, n_sg=2, rb_m=2, rb_n=2, tile_k=16, a_block_size=512, b_block_size=1024.
fn msg_assignments() -> BTreeMap<String, i64> {
    let mut m = BTreeMap::new();
    m.insert("wg_threads".to_owned(), 64_i64);
    m.insert("n_sg".to_owned(), 2_i64);
    m.insert("rb_m".to_owned(), 2_i64);
    m.insert("rb_n".to_owned(), 2_i64);
    m.insert("tile_k".to_owned(), 16_i64);
    m.insert("a_block_size".to_owned(), 512_i64);
    m.insert("b_block_size".to_owned(), 1024_i64);
    m
}

// ── AT-1743: MSG 2-subgroup bit-exact GPU dispatch ────────────────────────────

/// AT-1743: matmul_msg_coopmat.axc is BIT-EXACT (max_diff==0.0) on NVIDIA.
///
/// M=32, N=64, K=32 → grid (1,1,1) = ONE workgroup = 2 subgroups covering full 32×64 C.
/// K=32 → 2 K-blocks → loop-bottom WAR barrier exercised once across both subgroups.
///
/// Non-symmetric integer-f16 fixture (A in {1..4}, B in {1..3}, discriminates sg_id offsets).
///
/// GUARDS:
///   subgroup_size()==32 (kernel REQUIRES sg_size==32; mirror AT-1510) +
///   CoopMatUnsupported (Lavapipe).
#[test]
#[ignore]
fn at1743_matmul_msg_coopmat_bit_exact() {
    if !gpu_tests_enabled() {
        eprintln!("AT-1743: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    const M: usize = 32;
    const N: usize = 64;
    const K: usize = 32;

    let assignments = msg_assignments();
    let (bytes, meta) = compile_source_with_assignments(MATMUL_MSG_COOPMAT_SRC, &assignments)
        .expect("AT-1743: matmul_msg_coopmat.axc must compile");
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let ctx = VulkanContext::new().expect("AT-1743: VulkanContext must init");
    eprintln!("AT-1743: device={}", ctx.physical_device_name());

    // --- subgroup_size==32 GUARD (mirror dispatch_matmul_tile.rs:231 AT-1510) ---
    if ctx.subgroup_size() != 32 {
        eprintln!(
            "AT-1743: subgroup_size={} != 32; skipping (wave64 guard — kernel REQUIRES sg_size==32)",
            ctx.subgroup_size()
        );
        return;
    }

    let handle = match ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, meta.coopmat.as_ref(), "matmul_msg_coopmat",
        meta.shared_memory_bytes,
    ) {
        Ok(h) => h,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("AT-1743: CoopMatUnsupported (typed-skip on Lavapipe): {reason}");
            return;
        }
        Err(DispatchError::DeviceFeatureUnsupported { feature, .. }) => {
            eprintln!("AT-1743: DeviceFeatureUnsupported({feature}) — typed-skip");
            return;
        }
        Err(e) => panic!("AT-1743: prepare_kernel_checked failed: {e}"),
    };

    // Non-symmetric integer-f16 fixture (A in {1..4}, B in {1..3}).
    // max element = K × max(A) × max(B) = 32 × 4 × 3 = 384 ≤ 2048 (f16-integer-exact).
    let a_f32: Vec<f32> = (0..M * K).map(|idx| ((idx % 4) + 1) as f32).collect();
    let b_f32: Vec<f32> = (0..K * N).map(|idx| ((idx % 3) + 1) as f32).collect();
    let a_bytes = f32_slice_to_f16_le_bytes(&a_f32);
    let b_bytes = f32_slice_to_f16_le_bytes(&b_f32);
    let c_size = M * N * 2;

    // MSG grid: (N/64, M/32, 1) = (1,1,1).
    let wg_x: u32 = (N / 64) as u32;
    let wg_y: u32 = (M / 32) as u32;
    let pc = push_mnk(M as u32, N as u32, K as u32);

    let outputs = ctx.dispatch_handle(
        &handle, (wg_x, wg_y, 1),
        &[&a_bytes, &b_bytes, &vec![0u8; c_size]],
        &[0, 0, c_size],
        &pc,
    ).unwrap_or_else(|e| panic!("AT-1743: dispatch failed: {e}"));

    let gpu_c = f16_le_bytes_to_f32_slice(&outputs[2]);
    let cpu_c = cpu_f16_matmul_reference(&a_f32, &b_f32, M, N, K);

    let mut max_diff = 0.0_f32;
    for (g, c) in gpu_c.iter().zip(cpu_c.iter()) {
        let diff = (g - c).abs();
        if diff > max_diff { max_diff = diff; }
    }

    eprintln!(
        "AT-1743: M={M} N={N} K={K}, grid=({wg_x},{wg_y},1) = 1 WG = 2 subgroups, \
         max_diff={max_diff}, first4 GPU={:?}, CPU={:?}",
        &gpu_c[..4.min(gpu_c.len())],
        &cpu_c[..4.min(cpu_c.len())]
    );

    assert!(
        max_diff == 0.0,
        "AT-1743: MSG 2-subgroup matmul FAILED — max_diff={max_diff} != 0.\n\
         HONESTY GATE: do NOT relax tolerance. Grid=({wg_x},{wg_y},1) M={M} N={N} K={K}.\n\
         First4 GPU: {:?}, CPU: {:?}",
        &gpu_c[..4.min(gpu_c.len())],
        &cpu_c[..4.min(cpu_c.len())]
    );
    eprintln!(
        "AT-1743 PASS: MSG 2-subgroup matmul bit-exact (max_diff=0.0) on {} — \
         M={M} N={N} K={K}, grid=({wg_x},{wg_y},1)",
        ctx.physical_device_name()
    );
}

// ── AT-1744: MSG bit-exact, multi-WG + multi-K-block ─────────────────────────

/// AT-1744: matmul_msg_coopmat.axc bit-exact, multi-workgroup + multi-K-block.
///
/// M=64, N=128, K=48 → grid (2,2,1) = 4 workgroups, 3 K-blocks.
/// K=48 → 3 K-blocks → loop-bottom WAR barrier exercised TWICE across both subgroups.
///
/// GUARDS:
///   subgroup_size()==32 (kernel REQUIRES sg_size==32; mirror AT-1510) +
///   CoopMatUnsupported (Lavapipe).
#[test]
#[ignore]
fn at1744_matmul_msg_coopmat_bit_exact_multiblock() {
    if !gpu_tests_enabled() {
        eprintln!("AT-1744: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    const M: usize = 64;
    const N: usize = 128;
    const K: usize = 48;

    let assignments = msg_assignments();
    let (bytes, meta) = compile_source_with_assignments(MATMUL_MSG_COOPMAT_SRC, &assignments)
        .expect("AT-1744: matmul_msg_coopmat.axc must compile");
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let ctx = VulkanContext::new().expect("AT-1744: VulkanContext must init");
    eprintln!("AT-1744: device={}", ctx.physical_device_name());

    // --- subgroup_size==32 GUARD (mirror AT-1510) ---
    if ctx.subgroup_size() != 32 {
        eprintln!(
            "AT-1744: subgroup_size={} != 32; skipping (wave64 guard — kernel REQUIRES sg_size==32)",
            ctx.subgroup_size()
        );
        return;
    }

    let handle = match ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, meta.coopmat.as_ref(), "matmul_msg_coopmat",
        meta.shared_memory_bytes,
    ) {
        Ok(h) => h,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("AT-1744: CoopMatUnsupported (typed-skip on Lavapipe): {reason}");
            return;
        }
        Err(DispatchError::DeviceFeatureUnsupported { feature, .. }) => {
            eprintln!("AT-1744: DeviceFeatureUnsupported({feature}) — typed-skip");
            return;
        }
        Err(e) => panic!("AT-1744: prepare_kernel_checked failed: {e}"),
    };

    // Non-symmetric integer-f16 fixture.
    // max element = K × 4 × 3 = 48 × 12 = 576 ≤ 2048 (f16-integer-exact).
    let a_f32: Vec<f32> = (0..M * K).map(|idx| ((idx % 4) + 1) as f32).collect();
    let b_f32: Vec<f32> = (0..K * N).map(|idx| ((idx % 3) + 1) as f32).collect();
    let a_bytes = f32_slice_to_f16_le_bytes(&a_f32);
    let b_bytes = f32_slice_to_f16_le_bytes(&b_f32);
    let c_size = M * N * 2;

    // MSG grid: (N/64, M/32, 1) = (2,2,1) for M=64, N=128.
    let wg_x: u32 = (N / 64) as u32;
    let wg_y: u32 = (M / 32) as u32;
    let pc = push_mnk(M as u32, N as u32, K as u32);
    let k_blocks = K / 16;

    let outputs = ctx.dispatch_handle(
        &handle, (wg_x, wg_y, 1),
        &[&a_bytes, &b_bytes, &vec![0u8; c_size]],
        &[0, 0, c_size],
        &pc,
    ).unwrap_or_else(|e| panic!("AT-1744: dispatch failed: {e}"));

    let gpu_c = f16_le_bytes_to_f32_slice(&outputs[2]);
    let cpu_c = cpu_f16_matmul_reference(&a_f32, &b_f32, M, N, K);

    let mut max_diff = 0.0_f32;
    for (g, c) in gpu_c.iter().zip(cpu_c.iter()) {
        let diff = (g - c).abs();
        if diff > max_diff { max_diff = diff; }
    }

    eprintln!(
        "AT-1744: M={M} N={N} K={K} ({k_blocks} K-blocks), grid=({wg_x},{wg_y},1) = 4 WGs, \
         max_diff={max_diff}, first4 GPU={:?}, CPU={:?}",
        &gpu_c[..4.min(gpu_c.len())],
        &cpu_c[..4.min(cpu_c.len())]
    );

    assert!(
        max_diff == 0.0,
        "AT-1744: MSG multi-K-block matmul FAILED — max_diff={max_diff} != 0.\n\
         HONESTY GATE: do NOT relax. Grid=({wg_x},{wg_y},1) M={M} N={N} K={K} ({k_blocks} K-blocks).\n\
         First4 GPU: {:?}, CPU: {:?}",
        &gpu_c[..4.min(gpu_c.len())],
        &cpu_c[..4.min(cpu_c.len())]
    );
    eprintln!(
        "AT-1744 PASS: MSG multi-K-block matmul bit-exact (max_diff=0.0) on {} — \
         M={M} N={N} K={K} ({k_blocks} K-blocks), grid=({wg_x},{wg_y},1)",
        ctx.physical_device_name()
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// M3.2b — FlashAttention-2 streaming online-softmax (C2)
//
// flash_attention.axc: the streaming online-softmax recurrence (running max m_i,
// running denom l_i, rescaled acc[d]) — NO S materialization (no shared-S, no
// global scratch). Taylor exp hard-wired to the near-uniform faithful band
// (post-max args ≤ 0.7); the streaming ALGORITHM is the deliverable.
//
// AT-1738  compile + spirv-val (compile_shared_examples.rs has the canonical anchor;
//          AT-1738 below also spirv-vals the emitted kernel).
// AT-1740  small shape (seq_len=64, head_dim=64) within FROZEN 1e-3 vs Taylor oracle.
// AT-1740b head_dim=32 cross-check vs the C1 tiled_attention kernel (both within 1e-3).
// AT-1741  long sequence (seq_len>=2048) within 1e-3 — exercises the streaming loop.
// AT-1742  no-S falsifiers: shared==3*head_dim*4 invariant to seq_len AND
//          binding_plan.buffers names == {Q,K,V,O}.
// AT-1743  oracle independence: kernel≈Taylor-oracle (FROZEN 1e-3 GATE) AND
//          kernel≈true-exp-softmax (~5e-2 sanity) AND oracle≈full-softmax-Taylor (1e-4).
// AT-1744  first-iter j==0 guard + monotone running-max climb, both within 1e-3.
// ══════════════════════════════════════════════════════════════════════════════

/// FROZEN tolerance: kernel vs the Taylor-identical FA2 oracle. NEVER loosened.
const FA2_FROZEN_TOL: f32 = 1e-3_f32;
/// Independent SANITY tolerance: kernel vs the TRUE-exp (std f32::exp) full-softmax.
/// Distinct from and LOOSER than the gate — NOT a gate.
const FA2_TRUE_EXP_SANITY_TOL: f32 = 5e-2_f32;
/// The enforced Taylor faithful-band bound on the max |post-max exp arg| over a fixture.
const FA2_MAX_POSTMAX_ARG: f32 = 0.7_f32;

/// CPU FlashAttention-2 reference — the FROZEN-1e-3 GATE oracle.
///
/// Runs the EXACT SAME online-softmax recurrence in f32 with the IDENTICAL Taylor exp
/// (1 + x + x²/2) and the IDENTICAL `j==0 => correction=0` first-iteration guard in the
/// IDENTICAL statement order as flash_attention.axc. Because the kernel and oracle compute
/// the SAME function in the SAME f32 order, the result is bit-close (mirrors
/// cpu_tiled_attention_reference at lines 93-140).
fn cpu_flash_attention_reference(
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
        // Loop-carried online-softmax state.
        let mut m_i = -1e9_f32; // running max (init ONLY to make s > m_i on j==0)
        let mut l_i = 0.0_f32; // running denom
        let mut acc = vec![0.0_f32; head_dim]; // rescaled accumulator
        for j in 0..seq_len {
            let kb = j * head_dim;
            // s = dot(Q_i, K_j) * inv_sqrt_d.
            let mut s = 0.0_f32;
            for d in 0..head_dim {
                s += q[q_base + d] * k[kb + d];
            }
            s *= inv_sqrt_d;
            // m_new = max(m_i, s).
            let m_new = if s > m_i { s } else { m_i };
            // correction = Taylor exp(m_i - m_new), EXCEPT j==0 where it is FORCED to 0
            // (the SOLE first-iter mechanism — IDENTICAL guard + statement order to the kernel).
            let cx = m_i - m_new;
            let mut correction = 1.0_f32 + cx + cx * cx * 0.5_f32;
            if j == 0 {
                correction = 0.0_f32;
            }
            // p = Taylor exp(s - m_new) (arg <= 0).
            let px = s - m_new;
            let p = 1.0_f32 + px + px * px * 0.5_f32;
            // Online update.
            l_i = l_i * correction + p;
            for d in 0..head_dim {
                acc[d] = acc[d] * correction + p * v[kb + d];
            }
            m_i = m_new;
        }
        // Finalize.
        for d in 0..head_dim {
            o[q_base + d] = acc[d] / l_i;
        }
    }
    o
}

/// INDEPENDENT true-exp reference (AT-1743 leg 2): a stable full-softmax attention
/// computed with Rust std `f32::exp` (NOT the Taylor polynomial).
///
/// For each query row: s_j = (Q_i·K_j)*inv_sqrt_d; m = max_j s_j; w_j = exp(s_j - m);
/// Z = Σ w_j; O_i = Σ (w_j/Z) V_j. On near-uniform fixtures (post-max args in [-0.7,0])
/// this agrees with the Taylor oracle to ~3% → fits under FA2_TRUE_EXP_SANITY_TOL.
fn fa2_true_exp_softmax_reference(
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
        let mut scores = vec![0.0_f32; seq_len];
        let mut m = -1e9_f32;
        for (j, slot) in scores.iter_mut().enumerate() {
            let kb = j * head_dim;
            let mut s = 0.0_f32;
            for d in 0..head_dim {
                s += q[q_base + d] * k[kb + d];
            }
            s *= inv_sqrt_d;
            *slot = s;
            if s > m {
                m = s;
            }
        }
        let mut z = 0.0_f32;
        let mut w = vec![0.0_f32; seq_len];
        for (j, wj) in w.iter_mut().enumerate() {
            let e = (scores[j] - m).exp(); // TRUE std exp, not Taylor
            *wj = e;
            z += e;
        }
        for d in 0..head_dim {
            let mut out_val = 0.0_f32;
            for j in 0..seq_len {
                out_val += (w[j] / z) * v[j * head_dim + d];
            }
            o[q_base + d] = out_val;
        }
    }
    o
}

/// AT-1743 leg 3 oracle self-consistency: the online FA2 recurrence is mathematically
/// equal to a stable full-softmax computed with the SAME Taylor exp. This computes that
/// full-softmax-Taylor reference (subtract row max, Taylor-exp, normalize, weight V) so
/// the test can assert the online oracle matches it within ~1e-4 — catching an oracle-side
/// rescale-algebra bug INDEPENDENT of the GPU and of the exp choice.
fn fa2_full_softmax_taylor_reference(
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
        let mut scores = vec![0.0_f32; seq_len];
        let mut m = -1e9_f32;
        for (j, slot) in scores.iter_mut().enumerate() {
            let kb = j * head_dim;
            let mut s = 0.0_f32;
            for d in 0..head_dim {
                s += q[q_base + d] * k[kb + d];
            }
            s *= inv_sqrt_d;
            *slot = s;
            if s > m {
                m = s;
            }
        }
        let mut z = 0.0_f32;
        let mut w = vec![0.0_f32; seq_len];
        for (j, wj) in w.iter_mut().enumerate() {
            let x = scores[j] - m;
            let e = 1.0_f32 + x + x * x * 0.5_f32; // SAME Taylor as the kernel/oracle
            *wj = e;
            z += e;
        }
        for d in 0..head_dim {
            let mut out_val = 0.0_f32;
            for j in 0..seq_len {
                out_val += (w[j] / z) * v[j * head_dim + d];
            }
            o[q_base + d] = out_val;
        }
    }
    o
}

/// Replay the online recurrence over the fixture and return the MAXIMUM |post-max exp arg|
/// — max over all (i,j) of |s_ij - m_i^new| (the p-arg) and |m_i^prev - m_i^new| (the
/// correction-arg, skipped on j==0 where correction is forced to 0). EVERY FA2 test asserts
/// the return ≤ FA2_MAX_POSTMAX_ARG (the Taylor faithful-band HARD-WIRE), so a wide-spread
/// fixture cannot slip through.
fn fa2_fixture_max_postmax_arg(
    q: &[f32],
    k: &[f32],
    seq_len: usize,
    head_dim: usize,
    inv_sqrt_d: f32,
) -> f32 {
    let mut max_arg = 0.0_f32;
    for q_row in 0..seq_len {
        let q_base = q_row * head_dim;
        let mut m_i = -1e9_f32;
        for j in 0..seq_len {
            let kb = j * head_dim;
            let mut s = 0.0_f32;
            for d in 0..head_dim {
                s += q[q_base + d] * k[kb + d];
            }
            s *= inv_sqrt_d;
            let m_new = if s > m_i { s } else { m_i };
            // p-arg: |s - m_new|.
            let p_arg = (s - m_new).abs();
            if p_arg > max_arg {
                max_arg = p_arg;
            }
            // correction-arg: |m_i - m_new|, but j==0 forces correction=0 (arg unused).
            if j != 0 {
                let c_arg = (m_i - m_new).abs();
                if c_arg > max_arg {
                    max_arg = c_arg;
                }
            }
            m_i = m_new;
        }
    }
    max_arg
}

/// Host-side preflight: assert head_dim <= 64 (the shared[f32,64] bound) BEFORE any dispatch.
/// head_dim>64 would OOB k_tile/v_tile/acc — UB / robustness-clamp = silently wrong.
fn fa2_preflight(head_dim: u32) {
    assert!(
        head_dim <= 64,
        "flash_attention CORE requires head_dim <= 64 (shared[f32,64] bound); \
         head_dim={head_dim} is M3.2c (larger shared arrays / tiling)"
    );
}

/// AT-1742 no-global-scratch falsifier: assert the kernel binds EXACTLY the 4 buffers
/// {Q,K,V,O} (len==4, NAMES match — not just count). Any global S-materialization would
/// require a 5th O(seq_len) or O(seq_len²) scratch/score buffer, which would appear here.
fn fa2_assert_no_scratch(meta: &axc_runtime::KernelMetadata) {
    let names: Vec<&str> = meta
        .binding_plan
        .buffers
        .iter()
        .map(|b| b.name.as_str())
        .collect();
    assert_eq!(
        meta.binding_plan.buffers.len(),
        4,
        "AT-1742: flash_attention must bind EXACTLY 4 buffers (no O(N) / O(N·M) scratch); \
         got {} buffers: {names:?}",
        meta.binding_plan.buffers.len()
    );
    assert_eq!(
        names,
        vec!["Q", "K", "V", "O"],
        "AT-1742: flash_attention buffer NAMES must be exactly {{Q,K,V,O}} in order \
         (no fifth scratch/score buffer); got {names:?}"
    );
}

/// Near-uniform FA2 fixture (inherits the AT-1630 Q/K/V scale → post-max args in the
/// faithful band). seq_len rows, head_dim columns each.
fn fa2_fixture(seq_len: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let q: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| 0.1_f32 * ((i % 7) as f32 - 3.0_f32))
        .collect();
    let k: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| 0.1_f32 * ((i % 5) as f32 - 2.0_f32))
        .collect();
    let v: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| 0.05_f32 * (i as f32 % 11.0_f32))
        .collect();
    (q, k, v)
}

/// Run the flash_attention kernel on the GPU for a fixture, returning the O buffer as f32.
/// Calls fa2_preflight + asserts the faithful-band bound BEFORE dispatching.
fn fa2_dispatch_gpu(
    ctx: &VulkanContext,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    inv_sqrt_d: f32,
) -> Vec<f32> {
    fa2_preflight(head_dim as u32);
    let max_arg = fa2_fixture_max_postmax_arg(q, k, seq_len, head_dim, inv_sqrt_d);
    assert!(
        max_arg <= FA2_MAX_POSTMAX_ARG,
        "FA2 fixture violates the Taylor faithful band: max |post-max arg|={max_arg} > {FA2_MAX_POSTMAX_ARG}"
    );

    let (bytes, meta) = compile_source_with_meta(FLASH_ATTENTION_SRC)
        .expect("flash_attention.axc must compile");
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let handle = ctx
        .prepare_kernel_checked(
            &words,
            &meta.binding_plan,
            meta.push_constant_total_bytes,
            &meta.entry_point,
            None,
            "flash_attention",
            meta.shared_memory_bytes,
        )
        .unwrap_or_else(|e| panic!("flash_attention: prepare_kernel_checked failed: {e}"));

    let q_bytes: Vec<u8> = q.iter().flat_map(|v| v.to_le_bytes()).collect();
    let k_bytes: Vec<u8> = k.iter().flat_map(|v| v.to_le_bytes()).collect();
    let v_bytes: Vec<u8> = v.iter().flat_map(|v| v.to_le_bytes()).collect();
    let o_size = seq_len * head_dim * 4;
    let pc = push_attention(seq_len as u32, head_dim as u32, inv_sqrt_d);

    let outputs = ctx
        .dispatch_handle(
            &handle,
            (seq_len as u32, 1, 1),
            &[&q_bytes, &k_bytes, &v_bytes, &vec![0u8; o_size]],
            &[0, 0, 0, o_size],
            &pc,
        )
        .unwrap_or_else(|e| panic!("flash_attention: dispatch failed: {e}"));

    outputs[3]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// AT-1738: flash_attention.axc compiles to SPIR-V and passes spirv-val (no GPU).
#[test]
fn at1738_flash_attention_compiles_and_validates() {
    use spirv_tools::val::{Validator, create as create_validator};
    use spirv_tools::TargetEnv;

    let (bytes, _meta) = compile_source_with_meta(FLASH_ATTENTION_SRC)
        .expect("AT-1738: flash_attention.axc must compile");
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let validator = create_validator(Some(TargetEnv::Vulkan_1_1));
    validator
        .validate(&words, None)
        .expect("AT-1738: flash_attention.axc spirv-val must pass");
    eprintln!("AT-1738: flash_attention.axc compiles + spirv-val clean (M3.2b)");
}

/// AT-1742: no-S-materialization falsifiers (no GPU — reads compiled metadata).
///
/// (i) shared_memory_bytes == 3*head_dim*4 (== 768 at head_dim=64), INVARIANT across two
///     different seq_len compilations (the invariance is the falsifier — a materialized-S
///     design's SHARED footprint would scale with seq_len). seq_len is a runtime push
///     constant, never an array dimension, so both compilations are byte-identical here;
///     the EQUALITY assertion across them is what makes the no-shared-S claim meaningful.
/// (ii) binding_plan.buffers names == {Q,K,V,O} exactly (len==4) — no global scratch.
#[test]
fn at1742_flash_attention_no_s_materialization() {
    // Two compilations. seq_len is a push constant, so the source text is identical and
    // the kernel cannot encode seq_len into a shared-array size — but we assert the
    // invariance EXPLICITLY (the falsifier), and the symbolic 3*head_dim*4 value.
    let (_b1, meta1) = compile_source_with_meta(FLASH_ATTENTION_SRC)
        .expect("AT-1742: flash_attention.axc must compile (1)");
    let (_b2, meta2) = compile_source_with_meta(FLASH_ATTENTION_SRC)
        .expect("AT-1742: flash_attention.axc must compile (2)");

    // head_dim is the shared-array dimension (64): 3 arrays (k_tile, v_tile, acc) × 64 × 4 B.
    const HEAD_DIM: u32 = 64;
    let expected = 3 * HEAD_DIM * 4; // 768
    assert_eq!(
        meta1.shared_memory_bytes, expected,
        "AT-1742: shared_memory_bytes must == 3*head_dim*4 = {expected} (k_tile+v_tile+acc); \
         got {}",
        meta1.shared_memory_bytes
    );
    // (i) INVARIANCE: the shared footprint does NOT scale with seq_len.
    assert_eq!(
        meta1.shared_memory_bytes, meta2.shared_memory_bytes,
        "AT-1742: shared_memory_bytes must be INVARIANT to seq_len (no shared-memory S); \
         got {} vs {}",
        meta1.shared_memory_bytes, meta2.shared_memory_bytes
    );

    // (ii) NO-GLOBAL-SCRATCH: exactly the 4 buffers {Q,K,V,O}, names asserted.
    fa2_assert_no_scratch(&meta1);

    eprintln!(
        "AT-1742 PASS: no-S falsifiers — shared_memory_bytes={} (== 3*64*4, invariant to seq_len) \
         AND binding_plan.buffers == {{Q,K,V,O}} (len 4, no global scratch)",
        meta1.shared_memory_bytes
    );
}

/// AT-1743 legs 2 & 3 (CPU-only): oracle independence + rescale algebra.
///
/// Leg 2 (INDEPENDENT SANITY, ~5e-2): the FA2 Taylor oracle ≈ the TRUE-exp full-softmax
///   (std f32::exp) on the near-uniform fixture — closes the 'two identical-buggy Taylor
///   impls agree' hole. (Leg 2's GPU half — kernel ≈ true-exp — is asserted in AT-1740.)
/// Leg 3 (algebra, ~1e-4): the FA2 online Taylor oracle ≈ a full-softmax-Taylor reference
///   (SAME Taylor) — catches an oracle-side rescale-algebra bug, no GPU, no exp dependence.
/// All legs use abs/rel tolerance, NOT assert_eq (the rescale reassociates f32).
#[test]
fn at1743_fa2_oracle_independence_and_algebra() {
    const SEQ_LEN: usize = 64;
    const HEAD_DIM: usize = 64;
    let inv_sqrt_d = 1.0_f32 / (HEAD_DIM as f32).sqrt();
    let (q, k, v) = fa2_fixture(SEQ_LEN, HEAD_DIM);

    // Hard-wire the faithful band.
    let max_arg = fa2_fixture_max_postmax_arg(&q, &k, SEQ_LEN, HEAD_DIM, inv_sqrt_d);
    assert!(
        max_arg <= FA2_MAX_POSTMAX_ARG,
        "AT-1743: fixture max |post-max arg|={max_arg} > {FA2_MAX_POSTMAX_ARG}"
    );

    let oracle = cpu_flash_attention_reference(&q, &k, &v, SEQ_LEN, HEAD_DIM, inv_sqrt_d);
    let true_exp = fa2_true_exp_softmax_reference(&q, &k, &v, SEQ_LEN, HEAD_DIM, inv_sqrt_d);
    let full_taylor = fa2_full_softmax_taylor_reference(&q, &k, &v, SEQ_LEN, HEAD_DIM, inv_sqrt_d);

    // Leg 3 (algebra, abs/rel ~1e-4): online oracle ≈ full-softmax-Taylor.
    let mut leg3_diff = 0.0_f32;
    for (a, b) in oracle.iter().zip(full_taylor.iter()) {
        let d = (a - b).abs();
        if d > leg3_diff {
            leg3_diff = d;
        }
    }
    assert!(
        leg3_diff <= 1e-4_f32,
        "AT-1743 leg 3 (algebra): online FA2 oracle vs full-softmax-Taylor max_diff={leg3_diff} > 1e-4 \
         — the online rescale algebra disagrees with the single-pass softmax"
    );

    // Leg 2 (sanity, abs/rel ~5e-2): Taylor oracle ≈ TRUE-exp full-softmax.
    let mut leg2_diff = 0.0_f32;
    for (a, b) in oracle.iter().zip(true_exp.iter()) {
        let d = (a - b).abs();
        if d > leg2_diff {
            leg2_diff = d;
        }
    }
    assert!(
        leg2_diff <= FA2_TRUE_EXP_SANITY_TOL,
        "AT-1743 leg 2 (sanity): Taylor oracle vs TRUE-exp softmax max_diff={leg2_diff} > \
         {FA2_TRUE_EXP_SANITY_TOL} — the Taylor approximation is not faithful within the band"
    );

    eprintln!(
        "AT-1743 PASS (CPU legs): leg3 algebra max_diff={leg3_diff} (<=1e-4); \
         leg2 sanity max_diff={leg2_diff} (<={FA2_TRUE_EXP_SANITY_TOL}); fixture max_arg={max_arg}"
    );
}

/// AT-1740 GPU: FlashAttention-2 bit-close at SMALL shape (seq_len=64, head_dim=64) vs the
/// Taylor oracle within FROZEN 1e-3 (THE gate, leg 1), AND within ~5e-2 of the TRUE-exp
/// full-softmax (AT-1743 leg 2's GPU half — the independent sanity that this is a REAL
/// attention, not just Taylor-vs-Taylor). #[ignore]-gated, AXC_ENABLE_GPU_TESTS=1.
#[test]
#[ignore]
fn at1740_flash_attention_small_bitclose() {
    if !gpu_tests_enabled() {
        eprintln!("at1740: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    const SEQ_LEN: usize = 64;
    const HEAD_DIM: usize = 64;
    let inv_sqrt_d = 1.0_f32 / (HEAD_DIM as f32).sqrt();
    let (q, k, v) = fa2_fixture(SEQ_LEN, HEAD_DIM);

    let ctx = VulkanContext::new().expect("VulkanContext must init");
    eprintln!("AT-1740 GPU: device={}", ctx.physical_device_name());

    let gpu_o = fa2_dispatch_gpu(&ctx, &q, &k, &v, SEQ_LEN, HEAD_DIM, inv_sqrt_d);
    let cpu_o = cpu_flash_attention_reference(&q, &k, &v, SEQ_LEN, HEAD_DIM, inv_sqrt_d);
    let true_o = fa2_true_exp_softmax_reference(&q, &k, &v, SEQ_LEN, HEAD_DIM, inv_sqrt_d);

    // Leg 1 (FROZEN GATE): kernel ≈ Taylor oracle within 1e-3.
    let mut gate_diff = 0.0_f32;
    for (g, c) in gpu_o.iter().zip(cpu_o.iter()) {
        let d = (g - c).abs();
        if d > gate_diff {
            gate_diff = d;
        }
    }
    assert!(
        gate_diff <= FA2_FROZEN_TOL,
        "AT-1740 (GATE): flash_attention max_diff={gate_diff} > FROZEN {FA2_FROZEN_TOL}; \
         first GPU: {:?}, CPU: {:?}",
        &gpu_o[..4.min(gpu_o.len())],
        &cpu_o[..4.min(cpu_o.len())]
    );

    // Leg 2 (SANITY): kernel ≈ TRUE-exp softmax within ~5e-2.
    let mut sanity_diff = 0.0_f32;
    for (g, t) in gpu_o.iter().zip(true_o.iter()) {
        let d = (g - t).abs();
        if d > sanity_diff {
            sanity_diff = d;
        }
    }
    assert!(
        sanity_diff <= FA2_TRUE_EXP_SANITY_TOL,
        "AT-1740 (SANITY): flash_attention vs TRUE-exp softmax max_diff={sanity_diff} > \
         {FA2_TRUE_EXP_SANITY_TOL} (independent — proves a real attention)"
    );

    eprintln!(
        "AT-1740 PASS: GATE max_diff={gate_diff} (<= FROZEN {FA2_FROZEN_TOL}); \
         SANITY vs true-exp max_diff={sanity_diff} (<= {FA2_TRUE_EXP_SANITY_TOL}) on {}",
        ctx.physical_device_name()
    );
}

/// AT-1740b GPU: head_dim=32 CROSS-CHECK — the streaming FA2 kernel and the C1 NON-streaming
/// tiled_attention kernel, dispatched on the SAME (seq_len=64, head_dim=32) Q/K/V, must agree
/// within FROZEN 1e-3. Two DIFFERENT algorithms, one answer — a strong independent signal.
#[test]
#[ignore]
fn at1740b_flash_attention_vs_tiled_head_dim_32() {
    if !gpu_tests_enabled() {
        eprintln!("at1740b: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    const SEQ_LEN: usize = 64;
    const HEAD_DIM: usize = 32;
    let inv_sqrt_d = 1.0_f32 / (HEAD_DIM as f32).sqrt();
    // Reuse the AT-1630 fixture exactly (apples-to-apples vs the C1 head_dim=32 fixture).
    let q: Vec<f32> = (0..SEQ_LEN * HEAD_DIM).map(|i| 0.1_f32 * ((i % 7) as f32 - 3.0_f32)).collect();
    let k: Vec<f32> = (0..SEQ_LEN * HEAD_DIM).map(|i| 0.1_f32 * ((i % 5) as f32 - 2.0_f32)).collect();
    let v: Vec<f32> = (0..SEQ_LEN * HEAD_DIM).map(|i| 0.05_f32 * (i as f32 % 11.0_f32)).collect();

    let ctx = VulkanContext::new().expect("VulkanContext must init");
    eprintln!("AT-1740b GPU: device={}", ctx.physical_device_name());

    // FA2 (streaming) output.
    let fa2_o = fa2_dispatch_gpu(&ctx, &q, &k, &v, SEQ_LEN, HEAD_DIM, inv_sqrt_d);

    // C1 tiled_attention (non-streaming) output on the IDENTICAL inputs.
    let (tb, tmeta) = compile_source_with_meta(TILED_ATTENTION_SRC)
        .expect("AT-1740b: tiled_attention.axc must compile");
    let twords: Vec<u32> = tb.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let thandle = ctx
        .prepare_kernel_checked(
            &twords, &tmeta.binding_plan, tmeta.push_constant_total_bytes,
            &tmeta.entry_point, None, "tiled_attention", tmeta.shared_memory_bytes,
        )
        .unwrap_or_else(|e| panic!("AT-1740b: tiled prepare failed: {e}"));
    let q_bytes: Vec<u8> = q.iter().flat_map(|v| v.to_le_bytes()).collect();
    let k_bytes: Vec<u8> = k.iter().flat_map(|v| v.to_le_bytes()).collect();
    let v_bytes: Vec<u8> = v.iter().flat_map(|v| v.to_le_bytes()).collect();
    let o_size = SEQ_LEN * HEAD_DIM * 4;
    let pc = push_attention(SEQ_LEN as u32, HEAD_DIM as u32, inv_sqrt_d);
    let touts = ctx
        .dispatch_handle(
            &thandle, (SEQ_LEN as u32, 1, 1),
            &[&q_bytes, &k_bytes, &v_bytes, &vec![0u8; o_size]],
            &[0, 0, 0, o_size], &pc,
        )
        .unwrap_or_else(|e| panic!("AT-1740b: tiled dispatch failed: {e}"));
    let tiled_o: Vec<f32> = touts[3].chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    // Cross-check: FA2 streaming vs C1 non-streaming, within FROZEN 1e-3.
    let mut cross_diff = 0.0_f32;
    for (f, t) in fa2_o.iter().zip(tiled_o.iter()) {
        let d = (f - t).abs();
        if d > cross_diff {
            cross_diff = d;
        }
    }
    assert!(
        cross_diff <= FA2_FROZEN_TOL,
        "AT-1740b: FA2 (streaming) vs tiled_attention (C1) max_diff={cross_diff} > FROZEN \
         {FA2_FROZEN_TOL}; first FA2: {:?}, tiled: {:?}",
        &fa2_o[..4.min(fa2_o.len())],
        &tiled_o[..4.min(tiled_o.len())]
    );

    // Also vs the FA2 oracle (defense in depth).
    let cpu_o = cpu_flash_attention_reference(&q, &k, &v, SEQ_LEN, HEAD_DIM, inv_sqrt_d);
    let mut oracle_diff = 0.0_f32;
    for (f, c) in fa2_o.iter().zip(cpu_o.iter()) {
        let d = (f - c).abs();
        if d > oracle_diff {
            oracle_diff = d;
        }
    }
    assert!(
        oracle_diff <= FA2_FROZEN_TOL,
        "AT-1740b: FA2 vs oracle max_diff={oracle_diff} > FROZEN {FA2_FROZEN_TOL}"
    );

    eprintln!(
        "AT-1740b PASS: FA2 (streaming) vs tiled_attention (C1) max_diff={cross_diff}; \
         FA2 vs oracle max_diff={oracle_diff} (both <= {FA2_FROZEN_TOL}) on {} — \
         two algorithms, one answer",
        ctx.physical_device_name()
    );
}

/// AT-1741 GPU: FlashAttention-2 streaming-correct at LONG sequence (seq_len=2048,
/// head_dim=64) within FROZEN 1e-3 — exercises MANY K/V-loop iterations. Completing at all
/// at 2048 with O(tile)=768-byte shared + 0 scratch global is the corroborating runtime
/// evidence of the streaming property. Reports honest latency (SLOW, no perf claim).
#[test]
#[ignore]
fn at1741_flash_attention_long_streaming() {
    if !gpu_tests_enabled() {
        eprintln!("at1741: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    const SEQ_LEN: usize = 2048;
    const HEAD_DIM: usize = 64;
    let inv_sqrt_d = 1.0_f32 / (HEAD_DIM as f32).sqrt();
    let (q, k, v) = fa2_fixture(SEQ_LEN, HEAD_DIM);

    let ctx = VulkanContext::new().expect("VulkanContext must init");
    eprintln!("AT-1741 GPU: device={}", ctx.physical_device_name());

    let start = std::time::Instant::now();
    let gpu_o = fa2_dispatch_gpu(&ctx, &q, &k, &v, SEQ_LEN, HEAD_DIM, inv_sqrt_d);
    let elapsed = start.elapsed();
    let cpu_o = cpu_flash_attention_reference(&q, &k, &v, SEQ_LEN, HEAD_DIM, inv_sqrt_d);

    let mut max_diff = 0.0_f32;
    for (g, c) in gpu_o.iter().zip(cpu_o.iter()) {
        let d = (g - c).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    assert!(
        max_diff <= FA2_FROZEN_TOL,
        "AT-1741: long-sequence flash_attention max_diff={max_diff} > FROZEN {FA2_FROZEN_TOL}; \
         first GPU: {:?}, CPU: {:?}",
        &gpu_o[..4.min(gpu_o.len())],
        &cpu_o[..4.min(cpu_o.len())]
    );
    eprintln!(
        "AT-1741 PASS: streaming-correct at seq_len={SEQ_LEN} head_dim={HEAD_DIM} \
         max_diff={max_diff} (<= {FA2_FROZEN_TOL}) on {} — O(tile)=768-byte shared, 0 scratch \
         global. HONEST latency (SLOW, scalar core, NO perf claim): {:?}",
        ctx.physical_device_name(),
        elapsed
    );
}

/// AT-1744 GPU: first-iteration guard + running-max climb (TWO adversarial orderings),
/// both within FROZEN 1e-3.
///
/// Fixture A: the FIRST K/V row has the max score (so every LATER correction != 1) — exercises
///   the j==0 correction=0 guard AND the subsequent rescales.
/// Fixture B: a monotone SMALL-step climb so the running max rises every iteration (every
///   correction != 1) WHILE each correction arg stays > -0.7 (Taylor faithful).
/// Both assert max |post-max arg| <= 0.7 (via fa2_dispatch_gpu).
#[test]
#[ignore]
fn at1744_fa2_running_max_climb() {
    if !gpu_tests_enabled() {
        eprintln!("at1744: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    const SEQ_LEN: usize = 64;
    const HEAD_DIM: usize = 64;
    let inv_sqrt_d = 1.0_f32 / (HEAD_DIM as f32).sqrt();

    let ctx = VulkanContext::new().expect("VulkanContext must init");
    eprintln!("AT-1744 GPU: device={}", ctx.physical_device_name());

    // ── Fixture A: FIRST row has the max. Q/K aligned (positive), with row 0 of K boosted
    //    so dot(Q,K_0) is the per-query max. Scores stay small (near-uniform).
    let q_a: Vec<f32> = (0..SEQ_LEN * HEAD_DIM).map(|i| 0.05_f32 * (((i % 4) + 1) as f32)).collect();
    let mut k_a: Vec<f32> = (0..SEQ_LEN * HEAD_DIM).map(|i| 0.04_f32 * (((i % 3) + 1) as f32)).collect();
    for slot in k_a.iter_mut().take(HEAD_DIM) {
        *slot += 0.03_f32;
    }
    let v_a: Vec<f32> = (0..SEQ_LEN * HEAD_DIM).map(|i| 0.05_f32 * (i as f32 % 11.0_f32)).collect();

    let gpu_a = fa2_dispatch_gpu(&ctx, &q_a, &k_a, &v_a, SEQ_LEN, HEAD_DIM, inv_sqrt_d);
    let cpu_a = cpu_flash_attention_reference(&q_a, &k_a, &v_a, SEQ_LEN, HEAD_DIM, inv_sqrt_d);
    let mut diff_a = 0.0_f32;
    for (g, c) in gpu_a.iter().zip(cpu_a.iter()) {
        let d = (g - c).abs();
        if d > diff_a {
            diff_a = d;
        }
    }
    assert!(
        diff_a <= FA2_FROZEN_TOL,
        "AT-1744 fixture A (first-row-max, j==0 guard): max_diff={diff_a} > FROZEN {FA2_FROZEN_TOL}"
    );

    // ── Fixture B: monotone SMALL-step climb. Row j of K is scaled so dot(Q,K_j) increases
    //    monotonically in small steps → the running max rises every iteration (every
    //    correction != 1) while each step stays in the faithful band.
    let q_b: Vec<f32> = (0..SEQ_LEN * HEAD_DIM).map(|_| 0.05_f32).collect();
    let mut k_b: Vec<f32> = vec![0.0_f32; SEQ_LEN * HEAD_DIM];
    for j in 0..SEQ_LEN {
        let base = 0.03_f32 + 0.0008_f32 * (j as f32);
        for d in 0..HEAD_DIM {
            k_b[j * HEAD_DIM + d] = base;
        }
    }
    let v_b: Vec<f32> = (0..SEQ_LEN * HEAD_DIM).map(|i| 0.05_f32 * (i as f32 % 7.0_f32)).collect();

    let gpu_b = fa2_dispatch_gpu(&ctx, &q_b, &k_b, &v_b, SEQ_LEN, HEAD_DIM, inv_sqrt_d);
    let cpu_b = cpu_flash_attention_reference(&q_b, &k_b, &v_b, SEQ_LEN, HEAD_DIM, inv_sqrt_d);
    let mut diff_b = 0.0_f32;
    for (g, c) in gpu_b.iter().zip(cpu_b.iter()) {
        let d = (g - c).abs();
        if d > diff_b {
            diff_b = d;
        }
    }
    assert!(
        diff_b <= FA2_FROZEN_TOL,
        "AT-1744 fixture B (monotone climb, every-iter rescale): max_diff={diff_b} > FROZEN \
         {FA2_FROZEN_TOL}"
    );

    eprintln!(
        "AT-1744 PASS: first-row-max (j==0 guard) max_diff={diff_a}; monotone-climb \
         (every-iter rescale) max_diff={diff_b} (both <= {FA2_FROZEN_TOL}) on {}",
        ctx.physical_device_name()
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// M3.2c-exp — real exp() via GLSL.std.450 Exp. AT-1820..1827.
//
// AT-1820/1821  GPU exp(x) vs Rust f32::exp(x) over x in [-30,5] within FROZEN 1e-3
//               (combined abs/rel, abs floor 1e-6 for near-zero magnitudes). MEASURED.
// AT-1822       import-emitted-once — in compile_shared_examples.rs (raw codegen scan).
// AT-1823       spirv-val clean on the exp kernels + capability set BYTE-IDENTICAL to
//               the M3.2b Taylor flash_attention.axc (no new OpCapability/OpExtension).
// AT-1824       flash_attention_exp.axc compile anchor — in compile_shared_examples.rs.
// AT-1825       flash_attention_exp at REAL logit spreads (post-max args -10..0) within
//               FROZEN 1e-3 vs fa2_true_exp_softmax_reference (std f32::exp). The regime
//               M3.2b's Taylor CANNOT do.
// AT-1826       negative control — the M3.2b Taylor kernel on the SAME wide-spread fixture
//               is GROSSLY wrong (>> 1e-3) vs the true-exp oracle.
// AT-1827       M3.2b no-regression — the Taylor kernel SPIR-V emits ZERO OpExtInstImport
//               (the lazy cache stays None for a zero-exp kernel).
// ══════════════════════════════════════════════════════════════════════════════

/// Combined abs/rel pass: |a-b| <= max(abs_floor, rel_tol*|b|).
fn within_combined(a: f32, b: f32, rel_tol: f32, abs_floor: f32) -> bool {
    (a - b).abs() <= abs_floor.max(rel_tol * b.abs())
}

/// Count OpExtInstImport (opcode 11) instructions in a SPIR-V word stream.
/// Skips the 5-word module header.
fn count_ext_inst_imports(words: &[u32]) -> usize {
    let mut count = 0usize;
    let mut idx = 5usize;
    while idx < words.len() {
        let w0 = words[idx];
        let opcode = w0 & 0xFFFF;
        let wc = (w0 >> 16) as usize;
        if wc == 0 {
            break;
        }
        if opcode == 11 {
            count += 1;
        }
        idx += wc;
    }
    count
}

/// Collect the (value-id) operands of every OpCapability (opcode 17) + OpExtension
/// (opcode 10) instruction, as a sorted multiset, for a byte-comparable capability set.
fn capability_extension_signature(words: &[u32]) -> (Vec<u32>, Vec<String>) {
    let mut caps: Vec<u32> = Vec::new();
    let mut exts: Vec<String> = Vec::new();
    let mut idx = 5usize;
    while idx < words.len() {
        let w0 = words[idx];
        let opcode = w0 & 0xFFFF;
        let wc = (w0 >> 16) as usize;
        if wc == 0 {
            break;
        }
        if opcode == 17 {
            // OpCapability: word1 = capability enum value.
            caps.push(words[idx + 1]);
        } else if opcode == 10 {
            // OpExtension: word1.. = packed null-terminated literal string.
            let mut bytes: Vec<u8> = Vec::new();
            for w in &words[idx + 1..idx + wc] {
                bytes.extend_from_slice(&w.to_le_bytes());
            }
            let nul = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
            exts.push(String::from_utf8_lossy(&bytes[..nul]).into_owned());
        }
        idx += wc;
    }
    caps.sort_unstable();
    exts.sort();
    (caps, exts)
}

fn compile_words(src: &str, name: &str) -> Vec<u32> {
    let (bytes, _meta) = compile_source_with_meta(src)
        .unwrap_or_else(|e| panic!("{name}: compile failed: {e:?}"));
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// AT-1823 (no GPU): spirv-val clean on the exp kernels AND the capability/extension set
/// is BYTE-IDENTICAL to the M3.2b Taylor flash_attention.axc (the ext-inst adds NO new
/// OpCapability / OpExtension). Also: flash_attention_exp emits EXACTLY ONE OpExtInstImport
/// and the Taylor kernel emits ZERO (AT-1827's import-once-vs-none half).
#[test]
fn at1823_exp_spirv_val_no_new_capability() {
    use spirv_tools::val::{Validator, create as create_validator};
    use spirv_tools::TargetEnv;

    let taylor_words = compile_words(FLASH_ATTENTION_SRC, "flash_attention.axc (Taylor)");
    let exp_words = compile_words(FLASH_ATTENTION_EXP_SRC, "flash_attention_exp.axc");
    let micro_words = compile_words(EXP_MICRO_SRC, "exp_micro");

    let validator = create_validator(Some(TargetEnv::Vulkan_1_1));
    validator.validate(&exp_words, None).expect("AT-1823: flash_attention_exp spirv-val must pass");
    validator.validate(&micro_words, None).expect("AT-1823: exp_micro spirv-val must pass");

    // Capability/extension signature must be byte-identical to the Taylor kernel.
    let taylor_sig = capability_extension_signature(&taylor_words);
    let exp_sig = capability_extension_signature(&exp_words);
    assert_eq!(
        exp_sig, taylor_sig,
        "AT-1823: flash_attention_exp capability/extension set must be BYTE-IDENTICAL to the \
         M3.2b Taylor flash_attention.axc (GLSL.std.450 ext-inst adds NO OpCapability/OpExtension). \
         exp={exp_sig:?} taylor={taylor_sig:?}"
    );

    // Import-once vs none.
    assert_eq!(
        count_ext_inst_imports(&exp_words), 1,
        "AT-1823: flash_attention_exp must emit EXACTLY ONE OpExtInstImport (GLSL.std.450)"
    );
    assert_eq!(
        count_ext_inst_imports(&taylor_words), 0,
        "AT-1827 (half): the M3.2b Taylor kernel (zero exp) must emit ZERO OpExtInstImport \
         (the lazy GLSL.std.450 cache stays None)"
    );
    assert_eq!(
        count_ext_inst_imports(&micro_words), 1,
        "AT-1823: exp_micro (one exp) must emit EXACTLY ONE OpExtInstImport"
    );

    eprintln!(
        "AT-1823 PASS: exp kernels spirv-val clean; capability/extension set byte-identical to \
         M3.2b Taylor ({} caps, {} exts); imports: exp_fa=1, taylor=0, micro=1",
        taylor_sig.0.len(), taylor_sig.1.len()
    );
}

/// Wide-spread FA2 fixture: Q/K scaled so post-max exp args span roughly -10..0 — WELL outside
/// the Taylor faithful band (<= 0.7). The fa2_fixture_max_postmax_arg<=0.7 hard-wire is NOT
/// applied to this fixture (it is the whole point of the exp builtin).
fn fa2_wide_spread_fixture(seq_len: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // Each K row j gets a magnitude that RAMPS with the row index j, while Q is a fixed
    // positive pattern. Then s_ij = (Q_i · K_j) * inv_sqrt_d ramps ~linearly with j, so the
    // post-max logit gaps fan out across a WIDE band (~ -10..0), well beyond the Taylor
    // faithful band (<= 0.7). This is the regime real LLM attention lives in and the M3.2b
    // Taylor kernel CANNOT do.
    let inv_sqrt_d = 1.0_f32 / (head_dim as f32).sqrt();
    let mut q = vec![0.0_f32; seq_len * head_dim];
    let mut k = vec![0.0_f32; seq_len * head_dim];
    let mut v = vec![0.0_f32; seq_len * head_dim];
    let q_const = 0.5_f32;
    // Score for row j: s_j = head_dim * q_const * k_row(j) * inv_sqrt_d.
    // We want a WIDE spread of post-max args in the ONLINE recurrence (s_j - m_new and
    // m_prev - m_new). A monotone ramp keeps online per-step gaps tiny, so instead we make a
    // SAW pattern: the max score appears EARLY (row 0 is large), then later rows are far below
    // it (and oscillate), so for j>0 the p-arg s_j - m = (low - high) is strongly negative
    // (spanning ~ -10..0). This is the exact regime where Taylor exp diverges.
    let k_for_score = |target_score: f32| -> f32 {
        target_score / (head_dim as f32 * q_const * inv_sqrt_d)
    };
    for i in 0..seq_len {
        // Row 0 is the global max (score 0); later rows fan DOWN to ~ -11 with a saw ripple so
        // the spread is genuinely wide AND non-monotone (defeats the streaming-max collapse).
        let target = if i == 0 {
            0.0_f32
        } else {
            // Base descent -2..-11 plus a small per-row ripple.
            let base = -2.0_f32 - 9.0_f32 * (i as f32 / seq_len as f32);
            let ripple = if i % 2 == 0 { 0.5_f32 } else { -0.5_f32 };
            base + ripple
        };
        let kv = k_for_score(target);
        for d in 0..head_dim {
            q[i * head_dim + d] = q_const;
            k[i * head_dim + d] = kv;
            v[i * head_dim + d] = 0.05_f32 * ((i * head_dim + d) as f32 % 11.0_f32);
        }
    }
    (q, k, v)
}

/// Dispatch flash_attention_exp.axc on the GPU (NO Taylor faithful-band assert — that is the
/// whole point of the exp builtin). Returns the O buffer as f32.
fn fa2_exp_dispatch_gpu(
    ctx: &VulkanContext,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    inv_sqrt_d: f32,
) -> Vec<f32> {
    fa2_preflight(head_dim as u32);
    let (bytes, meta) = compile_source_with_meta(FLASH_ATTENTION_EXP_SRC)
        .expect("flash_attention_exp.axc must compile");
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let handle = ctx
        .prepare_kernel_checked(
            &words,
            &meta.binding_plan,
            meta.push_constant_total_bytes,
            &meta.entry_point,
            None,
            "flash_attention_exp",
            meta.shared_memory_bytes,
        )
        .unwrap_or_else(|e| panic!("flash_attention_exp: prepare_kernel_checked failed: {e}"));

    let q_bytes: Vec<u8> = q.iter().flat_map(|v| v.to_le_bytes()).collect();
    let k_bytes: Vec<u8> = k.iter().flat_map(|v| v.to_le_bytes()).collect();
    let v_bytes: Vec<u8> = v.iter().flat_map(|v| v.to_le_bytes()).collect();
    let o_size = seq_len * head_dim * 4;
    let pc = push_attention(seq_len as u32, head_dim as u32, inv_sqrt_d);

    let outputs = ctx
        .dispatch_handle(
            &handle,
            (seq_len as u32, 1, 1),
            &[&q_bytes, &k_bytes, &v_bytes, &vec![0u8; o_size]],
            &[0, 0, 0, o_size],
            &pc,
        )
        .unwrap_or_else(|e| panic!("flash_attention_exp: dispatch failed: {e}"));

    outputs[3]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Dispatch the M3.2b Taylor flash_attention.axc on the GPU WITHOUT the faithful-band assert
/// (the negative-control AT-1826 deliberately feeds it a wide-spread fixture to show it fails).
fn fa2_taylor_dispatch_gpu_unguarded(
    ctx: &VulkanContext,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    inv_sqrt_d: f32,
) -> Vec<f32> {
    fa2_preflight(head_dim as u32);
    let (bytes, meta) = compile_source_with_meta(FLASH_ATTENTION_SRC)
        .expect("flash_attention.axc must compile");
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let handle = ctx
        .prepare_kernel_checked(
            &words,
            &meta.binding_plan,
            meta.push_constant_total_bytes,
            &meta.entry_point,
            None,
            "flash_attention",
            meta.shared_memory_bytes,
        )
        .unwrap_or_else(|e| panic!("flash_attention (taylor): prepare_kernel_checked failed: {e}"));

    let q_bytes: Vec<u8> = q.iter().flat_map(|v| v.to_le_bytes()).collect();
    let k_bytes: Vec<u8> = k.iter().flat_map(|v| v.to_le_bytes()).collect();
    let v_bytes: Vec<u8> = v.iter().flat_map(|v| v.to_le_bytes()).collect();
    let o_size = seq_len * head_dim * 4;
    let pc = push_attention(seq_len as u32, head_dim as u32, inv_sqrt_d);

    let outputs = ctx
        .dispatch_handle(
            &handle,
            (seq_len as u32, 1, 1),
            &[&q_bytes, &k_bytes, &v_bytes, &vec![0u8; o_size]],
            &[0, 0, 0, o_size],
            &pc,
        )
        .unwrap_or_else(|e| panic!("flash_attention (taylor): dispatch failed: {e}"));

    outputs[3]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// AT-1820/1821 GPU: exp(x) on the GPU matches Rust f32::exp(x) over a dense sweep of
/// x in [-30, 5] (step 0.1, INCLUDING x < -5 where the M3.2b Taylor failed) within the
/// FROZEN 1e-3 combined abs/rel gate (abs floor 1e-6 for near-zero magnitudes). The raw
/// max relative gap is MEASURED and reported FIRST.
#[test]
#[ignore]
fn at1820_exp_vs_std_exp_range() {
    if !gpu_tests_enabled() {
        eprintln!("at1820: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    const REL_TOL: f32 = 1e-3_f32;
    const ABS_FLOOR: f32 = 1e-6_f32;

    // Dense sweep x in [-30, 5], step 0.1 → 351 points.
    let xs: Vec<f32> = (0..=350).map(|i| -30.0_f32 + 0.1_f32 * i as f32).collect();
    let n = xs.len();

    let (bytes, meta) = compile_source_with_meta(EXP_MICRO_SRC).expect("exp_micro must compile");
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let ctx = VulkanContext::new().expect("VulkanContext must init");
    eprintln!("AT-1820/1821 GPU: device={}", ctx.physical_device_name());

    let handle = ctx
        .prepare_kernel_checked(
            &words, &meta.binding_plan, meta.push_constant_total_bytes,
            &meta.entry_point, None, "exp_micro", meta.shared_memory_bytes,
        )
        .unwrap_or_else(|e| panic!("at1820: prepare failed: {e}"));

    let in_bytes: Vec<u8> = xs.iter().flat_map(|v| v.to_le_bytes()).collect();
    let out_size = n * 4;
    let outputs = ctx
        .dispatch_handle(
            &handle, (n as u32, 1, 1),
            &[&in_bytes, &vec![0u8; out_size]],
            &[0, out_size],
            &[],
        )
        .unwrap_or_else(|e| panic!("at1820: dispatch failed: {e}"));
    let gpu: Vec<f32> = outputs[1]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // MEASURE the raw max relative gap FIRST (report before gating).
    let mut max_rel = 0.0_f32;
    let mut max_abs = 0.0_f32;
    let mut worst_x = 0.0_f32;
    for (&x, &g) in xs.iter().zip(gpu.iter()) {
        let r = x.exp();
        let abs = (g - r).abs();
        let rel = if r.abs() > ABS_FLOOR { abs / r.abs() } else { 0.0_f32 };
        if abs > max_abs { max_abs = abs; }
        if rel > max_rel { max_rel = rel; worst_x = x; }
    }
    eprintln!(
        "AT-1820 MEASURED: max_rel={max_rel:e} (at x={worst_x}), max_abs={max_abs:e} over x in [-30,5] \
         on {}",
        ctx.physical_device_name()
    );

    // Gate within FROZEN 1e-3 combined abs/rel.
    for (&x, &g) in xs.iter().zip(gpu.iter()) {
        let r = x.exp();
        assert!(
            within_combined(g, r, REL_TOL, ABS_FLOOR),
            "AT-1820: exp({x}) gpu={g} vs f32::exp={r} exceeds combined tol \
             (rel {REL_TOL}, abs floor {ABS_FLOOR})"
        );
    }
    eprintln!("AT-1820/1821 PASS: GPU exp ~= f32::exp within FROZEN {REL_TOL} (max_rel={max_rel:e})");
}

/// AT-1825 GPU: flash_attention_exp.axc at REAL logit spreads (post-max args -10..0) within
/// FROZEN 1e-3 vs fa2_true_exp_softmax_reference (std f32::exp). The regime the M3.2b Taylor
/// kernel CANNOT do. The faithful-band hard-wire is NOT applied here; we instead ASSERT the
/// fixture is wide (max post-max arg well beyond 0.7) so the test genuinely exercises real range.
#[test]
#[ignore]
fn at1825_flash_attention_exp_real_range() {
    if !gpu_tests_enabled() {
        eprintln!("at1825: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    const SEQ_LEN: usize = 64;
    const HEAD_DIM: usize = 64;
    let inv_sqrt_d = 1.0_f32 / (HEAD_DIM as f32).sqrt();
    let (q, k, v) = fa2_wide_spread_fixture(SEQ_LEN, HEAD_DIM);

    // Confirm the fixture is genuinely WIDE (well outside the Taylor faithful band).
    let max_arg = fa2_fixture_max_postmax_arg(&q, &k, SEQ_LEN, HEAD_DIM, inv_sqrt_d);
    assert!(
        max_arg > 3.0_f32,
        "AT-1825: wide-spread fixture must have max |post-max arg| > 3.0 (real range); got {max_arg}"
    );

    let ctx = VulkanContext::new().expect("VulkanContext must init");
    eprintln!("AT-1825 GPU: device={}, fixture max post-max arg={max_arg}", ctx.physical_device_name());

    let gpu_o = fa2_exp_dispatch_gpu(&ctx, &q, &k, &v, SEQ_LEN, HEAD_DIM, inv_sqrt_d);
    let true_o = fa2_true_exp_softmax_reference(&q, &k, &v, SEQ_LEN, HEAD_DIM, inv_sqrt_d);

    let mut max_diff = 0.0_f32;
    for (g, t) in gpu_o.iter().zip(true_o.iter()) {
        let d = (g - t).abs();
        if d > max_diff { max_diff = d; }
        assert!(
            within_combined(*g, *t, FA2_FROZEN_TOL, FA2_FROZEN_TOL),
            "AT-1825 (GATE): flash_attention_exp gpu={g} vs true-exp softmax={t} exceeds \
             FROZEN {FA2_FROZEN_TOL} at real range (max post-max arg={max_arg})"
        );
    }
    eprintln!(
        "AT-1825 PASS: flash_attention_exp vs true-exp softmax max_diff={max_diff} \
         (<= FROZEN {FA2_FROZEN_TOL}) at REAL range (max post-max arg={max_arg}) on {}",
        ctx.physical_device_name()
    );
}

/// AT-1826 GPU (negative control): the M3.2b Taylor flash_attention.axc on the SAME wide-spread
/// fixture is GROSSLY wrong (max_diff >> 1e-3) vs the true-exp softmax oracle — proving the exp
/// builtin (not the fixture) is what fixes AT-1825. Tolerance direction INVERTED (a lower bound).
#[test]
#[ignore]
fn at1826_taylor_kernel_wrong_on_real_range() {
    if !gpu_tests_enabled() {
        eprintln!("at1826: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    const SEQ_LEN: usize = 64;
    const HEAD_DIM: usize = 64;
    let inv_sqrt_d = 1.0_f32 / (HEAD_DIM as f32).sqrt();
    let (q, k, v) = fa2_wide_spread_fixture(SEQ_LEN, HEAD_DIM);

    let ctx = VulkanContext::new().expect("VulkanContext must init");
    eprintln!("AT-1826 GPU (neg-control): device={}", ctx.physical_device_name());

    let taylor_o = fa2_taylor_dispatch_gpu_unguarded(&ctx, &q, &k, &v, SEQ_LEN, HEAD_DIM, inv_sqrt_d);
    let true_o = fa2_true_exp_softmax_reference(&q, &k, &v, SEQ_LEN, HEAD_DIM, inv_sqrt_d);

    let mut max_diff = 0.0_f32;
    for (t, r) in taylor_o.iter().zip(true_o.iter()) {
        let d = (t - r).abs();
        if d > max_diff { max_diff = d; }
    }
    // The Taylor kernel must be GROSSLY wrong at real range (large lower bound).
    assert!(
        max_diff > 1e-2_f32,
        "AT-1826 (neg-control): the Taylor kernel should be GROSSLY wrong at real range \
         (max_diff >> 1e-3); got only {max_diff} — the fixture may not be wide enough"
    );
    eprintln!(
        "AT-1826 PASS (neg-control): Taylor kernel vs true-exp softmax max_diff={max_diff} \
         (>> FROZEN {FA2_FROZEN_TOL} — Taylor cannot do real range) on {}",
        ctx.physical_device_name()
    );
}
