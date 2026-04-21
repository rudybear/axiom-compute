//! `InstanceOwner` — RAII wrapper around `ash::Instance` for Arc-based lifetime.
//!
//! M2.3a introduces `KernelHandle` instances that may outlive `VulkanContext`.
//! Both the context AND every `KernelHandleInner` hold an `Arc<InstanceOwner>`.
//! `vkDestroyInstance` fires exactly once: when the LAST `Arc<InstanceOwner>` is
//! dropped — ensuring the VkInstance lives at least as long as the VkDevice
//! (VkDevice is a child object of VkInstance per Vulkan spec §3.3.3).
//!
//! ## Drop ordering invariant
//!
//! `DeviceOwner::drop` (which calls `vkDestroyDevice`) fires before
//! `InstanceOwner::drop` (which calls `vkDestroyInstance`) because:
//! - `DeviceOwner` is held inside `KernelHandleInner`, which is shared by
//!   `VulkanContext` and any live `KernelHandle`s.
//! - `VulkanContext::drop` explicitly drops its `Arc<DeviceOwner>` (via
//!   `ManuallyDrop::take`) before dropping its `Arc<InstanceOwner>`.
//! - `KernelHandleInner::drop` destroys all per-handle Vulkan objects, then
//!   drops its `Arc<DeviceOwner>`, then drops its `Arc<InstanceOwner>`.
//!
//! The last of {VulkanContext, all KernelHandleInner} to be dropped triggers
//! both `vkDestroyDevice` (via `Arc<DeviceOwner>`) and `vkDestroyInstance`
//! (via `Arc<InstanceOwner>`) in the correct order.

/// RAII holder for the Vulkan instance and entry (function loader).
///
/// Held via `Arc<InstanceOwner>` by both `VulkanContext` and every
/// `KernelHandleInner`. The underlying `VkInstance` is destroyed exactly once,
/// when the last `Arc` reference is released.
///
/// # Drop ordering
///
/// `drop()` calls `destroy_instance` then implicitly drops `entry` (which
/// unloads the Vulkan shared library). This ordering is required: the instance
/// must be destroyed before the function-pointer table is freed.
pub(crate) struct InstanceOwner {
    /// Vulkan instance (VkInstance wrapper).
    pub(crate) instance: ash::Instance,
    /// Entry: holds the Vulkan library handle and function pointers.
    /// Must outlive `instance` (implicit via field order in Drop).
    #[allow(dead_code)]
    pub(crate) entry: ash::Entry,
}

impl Drop for InstanceOwner {
    fn drop(&mut self) {
        // SAFETY: destroy_instance must be called before the Entry (Vulkan library)
        // is unloaded. Rust drops fields in reverse declaration order after drop()
        // returns, so `entry` is dropped AFTER `instance`. But `destroy_instance`
        // must run while the function pointer (loaded via `entry`) is still valid.
        //
        // We call destroy_instance explicitly here in drop(). After this call,
        // `instance` still exists in memory until Rust drops it, but it's now
        // a destroyed/invalid handle. `entry` will then be dropped by Rust,
        // unloading the library — which is correct since instance is already gone.
        //
        // Invariant: when this Drop fires, all VkDevice objects created from this
        // VkInstance have already been destroyed (enforced by the DeviceOwner Arc
        // drop ordering in VulkanContext::drop and KernelHandleInner::drop).
        unsafe { self.instance.destroy_instance(None); }
    }
}
