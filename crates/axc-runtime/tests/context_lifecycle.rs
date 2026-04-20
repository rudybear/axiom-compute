//! AT-516: VulkanContext create + drop repeated OK.

use axc_runtime::{VulkanContext, probe_vulkan_available, gpu_tests_enabled};

/// AT-516: Create and drop VulkanContext a dozen times in a loop without leaks.
///
/// Tests that:
/// 1. VulkanContext::new() succeeds when Vulkan is available.
/// 2. VulkanContext::drop() runs vkDeviceWaitIdle + destroys all handles cleanly.
/// 3. Multiple sequential context creations/drops are safe (no handle leaks or
///    double-frees that would be caught by validation layers or address sanitizer).
#[test]
#[ignore] // GPU-gated: requires AXC_ENABLE_GPU_TESTS=1 + Vulkan ICD
fn context_new_and_drop_repeated_ok() {
    if !gpu_tests_enabled() {
        eprintln!("skipping context_lifecycle (AXC_ENABLE_GPU_TESTS != 1)");
        return;
    }
    if !probe_vulkan_available() {
        eprintln!("skipping context_lifecycle (no Vulkan ICD available)");
        return;
    }

    for i in 0..12 {
        let ctx = VulkanContext::new()
            .unwrap_or_else(|e| panic!("VulkanContext::new() failed at iteration {i}: {e}"));
        // Drop the context at the end of this block, exercising Drop logic.
        let name: &str = ctx.physical_device_name();
        eprintln!("  iteration {i}: device = {name}");
        // ctx drops here
    }
}
