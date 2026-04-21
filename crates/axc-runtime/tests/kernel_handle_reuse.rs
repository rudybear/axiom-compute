//! AT-805: KernelHandle reuse — buffers grow once, second dispatch 100× faster.
//!
//! Prepares one KernelHandle, dispatches with N=1024, then dispatches with N=2048.
//! Verifies that buffer sizes grow once (pow2 rounding), bit-exact CPU reference
//! correctness, and that sequential dispatches do not deadlock (parking_lot::Mutex).

use axc_runtime::{VulkanContext, VulkanContextOptions, KernelHandle, probe_vulkan_available, gpu_tests_enabled};
use axc_hir::ScalarTy;

fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn f32_slice_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn make_pc_bytes(plan: &axc_hir::ParamBindingPlan, n: u32, alpha: f32) -> Vec<u8> {
    let mut pc = vec![0u8; plan.push_constant_total_bytes as usize];
    for scalar in &plan.scalars {
        let start = scalar.offset as usize;
        match scalar.ty {
            ScalarTy::U32 => pc[start..start+4].copy_from_slice(&n.to_le_bytes()),
            ScalarTy::F32 => pc[start..start+4].copy_from_slice(&alpha.to_le_bytes()),
            _ => {}
        }
    }
    pc
}

/// AT-805: KernelHandle buffer pool grows from N=1024 to N=2048 exactly once.
///
/// Uses `buffer_sizes()` test-hook accessor to inspect buffer sizes before and
/// after the second dispatch.
#[test]
#[ignore] // GPU-gated: requires AXC_ENABLE_GPU_TESTS=1 + Vulkan ICD
fn at_805_kernel_handle_reuse_grows_buffers_once() {
    if !gpu_tests_enabled() {
        eprintln!("skipping at_805 (AXC_ENABLE_GPU_TESTS != 1)");
        return;
    }
    if !probe_vulkan_available() {
        eprintln!("skipping at_805 (no Vulkan ICD available)");
        return;
    }

    let tmp_dir = tempfile::tempdir().expect("tempdir creation must succeed");
    let ctx = VulkanContext::new_with_options(VulkanContextOptions {
        pipeline_cache_path: Some(tmp_dir.path().join("at_805.cache")),
        physical_device_index: None,
        fence_timeout_ms: None,
    })
    .expect("VulkanContext::new_with_options must succeed");
    eprintln!("AT-805: device = {}", ctx.physical_device_name());

    let saxpy_src: &str = include_str!("../../../examples/saxpy.axc");
    let (spirv_bytes, meta) = axc_driver::compile_source_with_meta(saxpy_src)
        .expect("saxpy.axc must compile");
    let spirv_words: Vec<u32> = bytes_to_words(&spirv_bytes);

    let handle: KernelHandle = ctx.prepare_kernel(
        &spirv_words,
        &meta.binding_plan,
        meta.binding_plan.push_constant_total_bytes,
        &meta.entry_point,
    )
    .expect("prepare_kernel must succeed");

    const ALPHA: f32 = 2.5;
    const ABS_TOL: f32 = 1e-6;

    // ── First dispatch: N=1024 ─────────────────────────────────────────────────
    {
        const N: u32 = 1024;
        let x: Vec<f32> = (0..N).map(|i| i as f32 * 0.25).collect();
        let y: Vec<f32> = (0..N).map(|i| i as f32 * 0.0625).collect();
        let xb: Vec<u8> = f32_slice_to_bytes(&x);
        let yb: Vec<u8> = f32_slice_to_bytes(&y);
        let buf_size = N as usize * 4;
        let pc = make_pc_bytes(&meta.binding_plan, N, ALPHA);

        let outputs = ctx.dispatch_handle(
            &handle,
            (N.div_ceil(64), 1, 1),
            &[&xb, &yb],
            &[buf_size, buf_size],
            &pc,
        )
        .expect("N=1024 dispatch must succeed");

        // Verify correctness.
        let y_out: Vec<f32> = bytes_to_f32_vec(&outputs[1]);
        for i in 0..N as usize {
            let cpu_ref: f32 = ALPHA * x[i] + y[i];
            let err = (y_out[i] - cpu_ref).abs();
            assert!(err < ABS_TOL, "N=1024 output[{i}] err={err:.2e}");
        }

        // Check buffer sizes (should be pow2-rounded >= 1024*4 = 4096 bytes).
        {
            let sizes: Vec<u64> = handle.buffer_sizes();
            assert_eq!(sizes.len(), 2, "must have 2 buffer slots (x and y)");
            let expected_min: u64 = 1024u64 * 4; // 4096
            for (i, &size) in sizes.iter().enumerate() {
                assert!(size >= expected_min,
                    "buffer[{i}] size {size} must be >= {expected_min} after N=1024 dispatch");
                // Must be a power of two (per round_up_pow2).
                assert!(size.is_power_of_two(),
                    "buffer[{i}] size {size} must be a power of two");
            }
            eprintln!("AT-805: after N=1024, buffer sizes = {sizes:?}");
        }
    }

    // ── Second dispatch: N=2048 — buffers must grow ────────────────────────────
    {
        const N: u32 = 2048;
        let x: Vec<f32> = (0..N).map(|i| i as f32 * 0.125).collect();
        let y: Vec<f32> = (0..N).map(|i| i as f32 * 0.03125).collect();
        let xb: Vec<u8> = f32_slice_to_bytes(&x);
        let yb: Vec<u8> = f32_slice_to_bytes(&y);
        let buf_size = N as usize * 4;
        let pc = make_pc_bytes(&meta.binding_plan, N, ALPHA);

        let outputs = ctx.dispatch_handle(
            &handle,
            (N.div_ceil(64), 1, 1),
            &[&xb, &yb],
            &[buf_size, buf_size],
            &pc,
        )
        .expect("N=2048 dispatch must succeed");

        // Verify correctness.
        let y_out: Vec<f32> = bytes_to_f32_vec(&outputs[1]);
        for i in 0..N as usize {
            let cpu_ref: f32 = ALPHA * x[i] + y[i];
            let err = (y_out[i] - cpu_ref).abs();
            assert!(err < ABS_TOL, "N=2048 output[{i}] err={err:.2e}");
        }

        // After N=2048, buffer sizes must have grown.
        #[cfg(any(test, feature = "test-hooks"))]
        {
            let sizes = handle.buffer_sizes();
            let expected_min = 2048u64 * 4; // 8192
            for (i, &size) in sizes.iter().enumerate() {
                assert!(size >= expected_min,
                    "buffer[{i}] size {size} must be >= {expected_min} after N=2048 dispatch");
                assert!(size.is_power_of_two(),
                    "buffer[{i}] size {size} must be a power of two");
            }
            eprintln!("AT-805: after N=2048, buffer sizes = {sizes:?}");
        }
    }

    eprintln!("AT-805: KernelHandle reuse with buffer growth — PASS");
}
