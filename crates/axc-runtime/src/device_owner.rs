//! `DeviceOwner` — RAII wrapper around `ash::Device` for Arc-based lifetime.
//!
//! M2.3a introduces prepare-once/dispatch-many via `KernelHandle`. A `KernelHandle`
//! may outlive its originating `VulkanContext`. Both the context AND every
//! `KernelHandleInner` hold an `Arc<DeviceOwner>`. `vkDestroyDevice` fires exactly
//! once: when the LAST `Arc<DeviceOwner>` is dropped.
//!
//! ## Drop ordering invariant
//!
//! `VulkanContext::drop` calls `device_wait_idle()` before dropping its
//! `Arc<DeviceOwner>`. `KernelHandleInner::drop` only drops its Arc after all
//! per-dispatch fence waits have completed (no in-flight GPU work). Therefore,
//! when `DeviceOwner::drop` finally fires, the VkDevice has no outstanding GPU
//! work and `destroy_device` is safe.

/// RAII holder for the logical Vulkan device.
///
/// Held via `Arc<DeviceOwner>` by both `VulkanContext` and every
/// `KernelHandleInner`. The underlying `VkDevice` is destroyed exactly once,
/// when the last `Arc` reference is released.
///
/// # Deref
///
/// Implements `Deref<Target = ash::Device>` so callers can call any `ash::Device`
/// method directly on `&DeviceOwner` without unwrapping.
pub(crate) struct DeviceOwner {
    /// The ash logical device. Public within the crate for direct Vulkan calls.
    pub(crate) device: ash::Device,
}

impl std::ops::Deref for DeviceOwner {
    type Target = ash::Device;

    fn deref(&self) -> &ash::Device {
        &self.device
    }
}

impl Drop for DeviceOwner {
    fn drop(&mut self) {
        // SAFETY: This Drop fires only when the LAST Arc<DeviceOwner> is released.
        // Invariant: both VulkanContext and KernelHandleInner guarantee that no
        // GPU work is in-flight at this point:
        //   - VulkanContext::drop calls device_wait_idle() before dropping its Arc.
        //   - KernelHandleInner::drop only drops its Arc after vkWaitForFences
        //     has returned (all submitted GPU work is complete).
        // Therefore, all VkQueue work has completed and it is safe to call
        // vkDestroyDevice. `None` allocator matches the allocator used in
        // instance.create_device (also None).
        unsafe { self.device.destroy_device(None); }
    }
}

#[cfg(test)]
mod tests {
    // DeviceOwner can only be meaningfully tested with a real Vulkan device.
    // Structural tests (drop fires exactly once, deref works) are verified
    // indirectly by AT-827 (kernel_handle_outlives_context integration test).
}
