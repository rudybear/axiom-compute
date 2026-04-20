//! AT-514: Dispatch saxpy.axc and compare output to CPU reference.
//!
//! Compiles examples/saxpy.axc via axc_driver::compile_source_with_meta,
//! dispatches on Lavapipe (or any available Vulkan device), and verifies
//! each output element against `alpha * x[i] + y[i]` within 1e-6 absolute
//! tolerance.
//!
//! ## Push-constant assembly discipline (AT-514a)
//!
//! Per AT-514a (C6 fix from M1.5 rev 1), push-constant bytes are assembled
//! by iterating `binding_plan.scalars` in stored order and writing each value
//! at `scalar.offset`. Never hardcoded.

use axc_runtime::{VulkanContext, DispatchRequest, probe_vulkan_available, gpu_tests_enabled};
use axc_hir::ScalarTy;

const ABS_TOL: f32 = 1e-6;

/// Convert `&[f32]` to `Vec<u8>` for dispatch input.
fn f32_slice_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Parse `&[u8]` output back to `Vec<f32>`.
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 4, 0, "output length must be 4-byte aligned");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Convert SPIR-V bytes to u32 word slice.
fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// AT-514: Dispatch saxpy.axc, verify output matches CPU reference within 1e-6.
///
/// saxpy: Y[i] = alpha * X[i] + Y[i]
///
/// Test parameters (chosen to avoid denormals):
/// - N = 1024
/// - alpha = 2.5
/// - x[i] = i * 0.25  (exact in f32)
/// - y[i] = i * 0.0625 (exact in f32, used as initial Y)
///
/// AT-514a discipline: push-constant bytes assembled via binding_plan.scalars
/// iteration, NOT by hardcoding [N bytes, alpha bytes].
#[test]
#[ignore] // GPU-gated: requires AXC_ENABLE_GPU_TESTS=1 + Vulkan ICD
fn dispatch_saxpy_matches_cpu_reference() {
    if !gpu_tests_enabled() {
        eprintln!("skipping dispatch_saxpy (AXC_ENABLE_GPU_TESTS != 1)");
        return;
    }
    if !probe_vulkan_available() {
        eprintln!("skipping dispatch_saxpy (no Vulkan ICD available)");
        return;
    }

    // ── Compile saxpy.axc ────────────────────────────────────────────────────
    let saxpy_src: &str = include_str!("../../../examples/saxpy.axc");
    let (spirv_bytes, meta) = axc_driver::compile_source_with_meta(saxpy_src)
        .expect("saxpy.axc must compile without errors");
    let spirv_words: Vec<u32> = bytes_to_words(&spirv_bytes);

    eprintln!("saxpy metadata: kernel='{}', wg={:?}, buffers={}, scalars={}",
        meta.kernel_name, meta.workgroup_size,
        meta.binding_plan.buffers.len(),
        meta.binding_plan.scalars.len());

    // ── Build test data ──────────────────────────────────────────────────────
    const N: u32 = 1024;
    let alpha: f32 = 2.5_f32;

    let x_data: Vec<f32> = (0..N).map(|i| i as f32 * 0.25_f32).collect();
    let y_data: Vec<f32> = (0..N).map(|i| i as f32 * 0.0625_f32).collect();

    let x_bytes: Vec<u8> = f32_slice_to_bytes(&x_data);
    let y_bytes: Vec<u8> = f32_slice_to_bytes(&y_data);
    let buf_size: usize = N as usize * 4;

    // ── Assemble push constants per AT-514a discipline ───────────────────────
    // Iterate binding_plan.scalars in stored order; write each value at its offset.
    // NEVER hardcode layout (e.g. do NOT assume scalars[0] is at offset 0).
    let mut pc_bytes: Vec<u8> = vec![0u8; meta.binding_plan.push_constant_total_bytes as usize];
    for scalar in &meta.binding_plan.scalars {
        let start: usize = scalar.offset as usize;
        match scalar.ty {
            ScalarTy::U32 => {
                let value_bytes: [u8; 4] = N.to_le_bytes();
                pc_bytes[start..start + 4].copy_from_slice(&value_bytes);
            }
            ScalarTy::F32 => {
                let value_bytes: [u8; 4] = alpha.to_le_bytes();
                pc_bytes[start..start + 4].copy_from_slice(&value_bytes);
            }
            ScalarTy::I32 => {
                let value_bytes: [u8; 4] = (N as i32).to_le_bytes();
                pc_bytes[start..start + 4].copy_from_slice(&value_bytes);
            }
            ScalarTy::U64 => {
                let value_bytes: [u8; 8] = (N as u64).to_le_bytes();
                pc_bytes[start..start + 8].copy_from_slice(&value_bytes);
            }
            ScalarTy::I64 => {
                let value_bytes: [u8; 8] = (N as i64).to_le_bytes();
                pc_bytes[start..start + 8].copy_from_slice(&value_bytes);
            }
            ScalarTy::F64 => {
                let value_bytes: [u8; 8] = (alpha as f64).to_le_bytes();
                pc_bytes[start..start + 8].copy_from_slice(&value_bytes);
            }
            _ => {
                // Other scalar types (Bool, I8, I16, U8, U16, F16, Bf16) are not
                // used in saxpy; zero-fill is a safe default for test stability.
            }
        }
    }

    // AT-514a assertion: verify we wrote at the correct offsets (plan-driven, not hardcoded).
    if meta.binding_plan.scalars.len() >= 2 {
        let n_slot = &meta.binding_plan.scalars[0];
        let a_slot = &meta.binding_plan.scalars[1];
        let n_written: u32 = u32::from_le_bytes(
            pc_bytes[n_slot.offset as usize..n_slot.offset as usize + 4].try_into().unwrap()
        );
        let a_written: f32 = f32::from_le_bytes(
            pc_bytes[a_slot.offset as usize..a_slot.offset as usize + 4].try_into().unwrap()
        );
        assert_eq!(n_written, N, "n push constant must equal N at plan offset");
        assert!((a_written - alpha).abs() < 1e-10,
            "alpha push constant must equal {alpha} at plan offset; got {a_written}");
    }

    // ── Dispatch ─────────────────────────────────────────────────────────────
    let workgroups: [u32; 3] = [N.div_ceil(64), 1, 1];
    let ctx = VulkanContext::new().expect("VulkanContext::new() must succeed");
    eprintln!("  device: {}", ctx.physical_device_name());

    let req = DispatchRequest {
        spirv: &spirv_words,
        binding_plan: &meta.binding_plan,
        workgroups,
        inputs: &[&x_bytes, &y_bytes],
        output_sizes: &[buf_size, buf_size],
        push_constants: &pc_bytes,
        entry_point: &meta.entry_point,
    };

    let outputs: Vec<Vec<u8>> = ctx.dispatch(req)
        .expect("dispatch must succeed on Lavapipe");

    // ── Correctness oracle ───────────────────────────────────────────────────
    // saxpy: y_out[i] = alpha * x[i] + y[i]
    let y_out: Vec<f32> = bytes_to_f32_vec(&outputs[1]);

    let mut max_err: f32 = 0.0_f32;
    for i in 0..N as usize {
        let cpu_ref: f32 = alpha * x_data[i] + y_data[i];
        let gpu_val: f32 = y_out[i];
        let err: f32 = (gpu_val - cpu_ref).abs();
        if err > max_err {
            max_err = err;
        }
        assert!(
            err < ABS_TOL,
            "saxpy[{i}]: GPU={gpu_val}, CPU={cpu_ref}, err={err} >= ABS_TOL={ABS_TOL}"
        );
    }
    eprintln!("  saxpy: max absolute error = {max_err:.2e} (tolerance {ABS_TOL:.2e})");
}
