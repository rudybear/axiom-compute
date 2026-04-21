//! AT-827: KernelHandle outlives VulkanContext — no UAF on handle drop.
//!
//! Creates a VulkanContext, prepares a kernel handle, dispatches once to warm
//! the buffers, then drops the VulkanContext while holding the KernelHandle.
//! Verifies that the device remains alive (via Arc<DeviceOwner>) until the
//! KernelHandle is dropped, and that no UAF occurs.

use axc_runtime::{VulkanContext, VulkanContextOptions, probe_vulkan_available, gpu_tests_enabled};

fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// AT-827: KernelHandle remains usable for DROP even after VulkanContext is dropped.
///
/// The Arc<DeviceOwner> in KernelHandleInner keeps the VkDevice alive until the
/// last handle drops — no use-after-free should occur.
#[test]
#[ignore] // GPU-gated: requires AXC_ENABLE_GPU_TESTS=1 + Vulkan ICD
fn at_827_kernel_handle_outlives_context() {
    if !gpu_tests_enabled() {
        eprintln!("skipping at_827 (AXC_ENABLE_GPU_TESTS != 1)");
        return;
    }
    if !probe_vulkan_available() {
        eprintln!("skipping at_827 (no Vulkan ICD available)");
        return;
    }

    // Use a tempdir pipeline cache so no env mutation is needed.
    let tmp_dir = tempfile::tempdir().expect("tempdir creation must succeed");
    let cache_path = tmp_dir.path().join("at_827_pipeline.cache");

    let saxpy_src: &str = include_str!("../../../examples/saxpy.axc");
    let (spirv_bytes, meta) = axc_driver::compile_source_with_meta(saxpy_src)
        .expect("saxpy.axc must compile");
    let spirv_words: Vec<u32> = bytes_to_words(&spirv_bytes);

    // Scope 1: create context A, prepare kernel, do one dispatch, then drop context.
    let handle = {
        let ctx = VulkanContext::new_with_options(VulkanContextOptions {
            pipeline_cache_path: Some(cache_path.clone()),
            physical_device_index: None,
            fence_timeout_ms: None,
        })
        .expect("VulkanContext::new_with_options must succeed");

        eprintln!("AT-827: device = {}", ctx.physical_device_name());

        let h = ctx.prepare_kernel(
            &spirv_words,
            &meta.binding_plan,
            meta.binding_plan.push_constant_total_bytes,
            &meta.entry_point,
        )
        .expect("prepare_kernel must succeed");

        // Warm dispatch to allocate buffers.
        const N: u32 = 256;
        let x_data: Vec<f32> = (0..N).map(|i| i as f32 * 0.25).collect();
        let y_data: Vec<f32> = (0..N).map(|i| i as f32 * 0.0625).collect();
        let x_bytes: Vec<u8> = x_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let y_bytes: Vec<u8> = y_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let buf_size = N as usize * 4;

        let mut pc_bytes = vec![0u8; meta.binding_plan.push_constant_total_bytes as usize];
        for scalar in &meta.binding_plan.scalars {
            use axc_hir::ScalarTy;
            let start = scalar.offset as usize;
            match scalar.ty {
                ScalarTy::U32 => pc_bytes[start..start+4].copy_from_slice(&N.to_le_bytes()),
                ScalarTy::F32 => pc_bytes[start..start+4].copy_from_slice(&2.5f32.to_le_bytes()),
                _ => {}
            }
        }

        let workgroups = (N.div_ceil(64), 1u32, 1u32);
        let _outputs = ctx.dispatch_handle(
            &h,
            workgroups,
            &[&x_bytes, &y_bytes],
            &[buf_size, buf_size],
            &pc_bytes,
        )
        .expect("warm dispatch must succeed");

        // Return handle — context A will be dropped at end of this block.
        h
    }; // ctx A dropped here → device_wait_idle + pipeline_cache.save + destroy pool
       // BUT the Arc<DeviceOwner> refcount is still > 0 because `handle` holds a clone.
       // Therefore vkDestroyDevice does NOT fire here.

    eprintln!("AT-827: VulkanContext dropped, handle still alive. Dropping handle now...");

    // Drop the handle — this is the last Arc<DeviceOwner> ref, so vkDestroyDevice fires here.
    drop(handle);

    eprintln!("AT-827: handle dropped cleanly — no UAF.");
    // If we reach here without a crash/segfault, AT-827 passes.
}
