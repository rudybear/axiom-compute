//! AT-515: Dispatch vector_add.axc and compare output to CPU reference.
//!
//! Compiles examples/vector_add.axc via axc_driver::compile_source_with_meta,
//! dispatches on Lavapipe (or any available Vulkan device), and verifies
//! each output element against `a[i] + b[i]` within 1e-6 absolute tolerance.

use axc_runtime::{VulkanContext, DispatchRequest, probe_vulkan_available, gpu_tests_enabled};
use axc_hir::ScalarTy;

const ABS_TOL: f32 = 1e-6;

fn f32_slice_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 4, 0, "output length must be 4-byte aligned");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// AT-515: Dispatch vector_add.axc, verify output matches CPU reference within 1e-6.
///
/// vector_add: C[i] = A[i] + B[i]
///
/// Test parameters:
/// - N = 2048
/// - a[i] = i * 0.125  (exact in f32)
/// - b[i] = i * 0.375  (exact in f32)
#[test]
#[ignore] // GPU-gated: requires AXC_ENABLE_GPU_TESTS=1 + Vulkan ICD
fn dispatch_vector_add_matches_cpu_reference() {
    if !gpu_tests_enabled() {
        eprintln!("skipping dispatch_vector_add (AXC_ENABLE_GPU_TESTS != 1)");
        return;
    }
    if !probe_vulkan_available() {
        eprintln!("skipping dispatch_vector_add (no Vulkan ICD available)");
        return;
    }

    // ── Compile vector_add.axc ────────────────────────────────────────────────
    let src: &str = include_str!("../../../examples/vector_add.axc");
    let (spirv_bytes, meta) = axc_driver::compile_source_with_meta(src)
        .expect("vector_add.axc must compile without errors");
    let spirv_words: Vec<u32> = bytes_to_words(&spirv_bytes);

    eprintln!("vector_add metadata: kernel='{}', wg={:?}, buffers={}, scalars={}",
        meta.kernel_name, meta.workgroup_size,
        meta.binding_plan.buffers.len(),
        meta.binding_plan.scalars.len());

    // ── Build test data ──────────────────────────────────────────────────────
    const N: u32 = 2048;

    let a_data: Vec<f32> = (0..N).map(|i| i as f32 * 0.125_f32).collect();
    let b_data: Vec<f32> = (0..N).map(|i| i as f32 * 0.375_f32).collect();

    let a_bytes: Vec<u8> = f32_slice_to_bytes(&a_data);
    let b_bytes: Vec<u8> = f32_slice_to_bytes(&b_data);
    let buf_size: usize = N as usize * 4;
    // Output buffer (c) has no meaningful input; zero-initialize.
    let c_input: Vec<u8> = vec![0u8; buf_size];

    // ── Assemble push constants per AT-514a discipline ───────────────────────
    let mut pc_bytes: Vec<u8> = vec![0u8; meta.binding_plan.push_constant_total_bytes as usize];
    for scalar in &meta.binding_plan.scalars {
        let start: usize = scalar.offset as usize;
        match scalar.ty {
            ScalarTy::U32 => {
                pc_bytes[start..start + 4].copy_from_slice(&N.to_le_bytes());
            }
            ScalarTy::I32 => {
                pc_bytes[start..start + 4].copy_from_slice(&(N as i32).to_le_bytes());
            }
            ScalarTy::F32 => {
                // vector_add has no f32 scalars, but handle gracefully.
                pc_bytes[start..start + 4].copy_from_slice(&(N as f32).to_le_bytes());
            }
            ScalarTy::U64 => {
                pc_bytes[start..start + 8].copy_from_slice(&(N as u64).to_le_bytes());
            }
            ScalarTy::I64 => {
                pc_bytes[start..start + 8].copy_from_slice(&(N as i64).to_le_bytes());
            }
            _ => {}
        }
    }

    // ── Dispatch ─────────────────────────────────────────────────────────────
    let workgroups: [u32; 3] = [N.div_ceil(64), 1, 1];
    let ctx = VulkanContext::new().expect("VulkanContext::new() must succeed");
    eprintln!("  device: {}", ctx.physical_device_name());

    // vector_add has 3 buffers: a (binding=0), b (binding=1), c (binding=2).
    let req = DispatchRequest {
        spirv: &spirv_words,
        binding_plan: &meta.binding_plan,
        workgroups,
        inputs: &[&a_bytes, &b_bytes, &c_input],
        output_sizes: &[buf_size, buf_size, buf_size],
        push_constants: &pc_bytes,
        entry_point: &meta.entry_point,
    };

    let outputs: Vec<Vec<u8>> = ctx.dispatch(req)
        .expect("dispatch must succeed on Lavapipe");

    // ── Correctness oracle ───────────────────────────────────────────────────
    // vector_add: c[i] = a[i] + b[i]
    let c_out: Vec<f32> = bytes_to_f32_vec(&outputs[2]);

    let mut max_err: f32 = 0.0_f32;
    for i in 0..N as usize {
        let cpu_ref: f32 = a_data[i] + b_data[i];
        let gpu_val: f32 = c_out[i];
        let err: f32 = (gpu_val - cpu_ref).abs();
        if err > max_err {
            max_err = err;
        }
        assert!(
            err < ABS_TOL,
            "vector_add[{i}]: GPU={gpu_val}, CPU={cpu_ref}, err={err} >= ABS_TOL={ABS_TOL}"
        );
    }
    eprintln!("  vector_add: max absolute error = {max_err:.2e} (tolerance {ABS_TOL:.2e})");
}
