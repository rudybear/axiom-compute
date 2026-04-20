//! `VulkanContext` — Vulkan device lifecycle for AXIOM-Compute dispatch.
//!
//! `VulkanContext::new()` initializes the full Vulkan stack:
//! - Entry (loads the Vulkan shared library)
//! - Instance (Vulkan API 1.1, no validation layers by default)
//! - Physical device selection (first device with a compute queue, or env override)
//! - Logical device + queue
//! - Command pool with `RESET_COMMAND_BUFFER`
//! - Cached device limits (`max_compute_work_group_count`) for pre-validation
//! - Cached memory properties for buffer allocation
//!
//! ## Drop behavior
//!
//! `VulkanContext::drop` calls `vkDeviceWaitIdle` before destroying the device.
//! This is critical on Lavapipe to avoid a `VK_ERROR_DEVICE_LOST` shutdown race
//! when previously-submitted commands have not yet completed.
//!
//! ## Device selection
//!
//! 1. If `AXC_PHYSICAL_DEVICE_INDEX` is set and in range, use that index.
//! 2. Otherwise, iterate physical devices in enumeration order and pick the first
//!    that has a compute queue family.

use ash::vk;
use crate::error::DispatchError;

/// Initialized Vulkan context for compute dispatch.
///
/// One context per process is typical. Multiple contexts are safe (AT-516) but
/// each has its own device and command pool — useful for testing, not for production.
pub struct VulkanContext {
    /// Vulkan entry (function table loader).
    #[allow(dead_code)]
    pub(crate) entry: ash::Entry,
    /// Vulkan instance.
    pub(crate) instance: ash::Instance,
    /// Selected physical device.
    #[allow(dead_code)]
    pub(crate) physical_device: vk::PhysicalDevice,
    /// Logical device.
    pub(crate) device: ash::Device,
    /// Compute queue handle.
    pub(crate) queue: vk::Queue,
    /// Queue family index used for the command pool and queue.
    #[allow(dead_code)]
    pub(crate) queue_family_index: u32,
    /// Command pool with `RESET_COMMAND_BUFFER` flag.
    pub(crate) command_pool: vk::CommandPool,
    /// Cached physical device memory properties (used for buffer allocation).
    pub(crate) memory_properties: vk::PhysicalDeviceMemoryProperties,
    /// Cached `max_compute_work_group_count` from device limits (used for pre-validation).
    pub(crate) max_compute_work_group_count: [u32; 3],
    /// Human-readable device name for diagnostics.
    device_name: String,
}

impl VulkanContext {
    /// Initialize Vulkan and create a compute-capable device context.
    ///
    /// Returns `Err(DispatchError::NoSupportedDevice)` if no physical device
    /// with a compute queue family is found.
    pub fn new() -> Result<Self, DispatchError> {
        // ── Step 1: Entry (load Vulkan library) ───────────────────────────────
        // SAFETY: Entry::load() loads the Vulkan shared library via the platform
        // search path. The returned Entry holds function pointers valid for the
        // lifetime of this process.
        let entry: ash::Entry = unsafe { ash::Entry::load() }
            .map_err(|e| DispatchError::VulkanEntryFailed(e.to_string()))?;

        // ── Step 2: Instance ──────────────────────────────────────────────────
        let app_name = std::ffi::CString::new("axc-runtime").unwrap();
        let engine_name = std::ffi::CString::new("axc-compute").unwrap();

        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(0)
            .engine_name(&engine_name)
            .engine_version(0)
            .api_version(vk::API_VERSION_1_1);

        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info);

        // SAFETY: instance_info is valid for the duration of this call. No validation
        // layers are enabled by default (opt-in via AXC_VULKAN_VALIDATION in M2+).
        let instance: ash::Instance =
            unsafe { entry.create_instance(&instance_info, None) }
                .map_err(|e| DispatchError::NoVulkanInstance(e.to_string()))?;

        // ── Step 3: Physical device selection ─────────────────────────────────
        // SAFETY: instance is valid; enumerate_physical_devices is a read-only query.
        let physical_devices: Vec<vk::PhysicalDevice> =
            unsafe { instance.enumerate_physical_devices() }
                .map_err(|e| {
                    // SAFETY: must clean up the instance on error.
                    unsafe { instance.destroy_instance(None); }
                    DispatchError::NoVulkanInstance(format!("enumerate_physical_devices: {e}"))
                })?;

        if physical_devices.is_empty() {
            // SAFETY: no devices found; destroy instance before returning error.
            unsafe { instance.destroy_instance(None); }
            return Err(DispatchError::NoSupportedDevice);
        }

        // Determine which physical device to use.
        let device_index_override: Option<usize> =
            std::env::var("AXC_PHYSICAL_DEVICE_INDEX")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&i| i < physical_devices.len());

        let (physical_device, queue_family_index): (vk::PhysicalDevice, u32) =
            match device_index_override {
                Some(idx) => {
                    let pd: vk::PhysicalDevice = physical_devices[idx];
                    // SAFETY: pd is a valid physical device.
                    let qf_idx = find_compute_queue_family(&instance, pd)
                        .ok_or_else(|| {
                            // SAFETY: destroy instance on error.
                            unsafe { instance.destroy_instance(None); }
                            DispatchError::NoComputeQueue
                        })?;
                    (pd, qf_idx)
                }
                None => {
                    let mut found: Option<(vk::PhysicalDevice, u32)> = None;
                    for &pd in &physical_devices {
                        // SAFETY: pd is a valid physical device.
                        if let Some(qf) = find_compute_queue_family(&instance, pd) {
                            found = Some((pd, qf));
                            break;
                        }
                    }
                    found.ok_or_else(|| {
                        // SAFETY: destroy instance; no usable device found.
                        unsafe { instance.destroy_instance(None); }
                        DispatchError::NoSupportedDevice
                    })?
                }
            };

        // ── Step 4: Device name (for diagnostics) ─────────────────────────────
        // SAFETY: physical_device is valid; get_physical_device_properties is read-only.
        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        // SAFETY: props.device_name is a null-terminated C string guaranteed by the
        // Vulkan spec (VkPhysicalDeviceProperties::deviceName is a char[256] with a NUL).
        let device_name: String = unsafe {
            std::ffi::CStr::from_ptr(props.device_name.as_ptr())
                .to_string_lossy()
                .into_owned()
        };

        // ── Step 5: Logical device + queue ────────────────────────────────────
        let queue_priorities: [f32; 1] = [1.0f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_create_info));

        // SAFETY: device_create_info is valid; no extensions or features that require
        // runtime checks are enabled (M1.5 only uses f32 arithmetic which is core 1.1).
        let device: ash::Device =
            unsafe { instance.create_device(physical_device, &device_create_info, None) }
                .map_err(|e| {
                    // SAFETY: destroy instance on device creation failure.
                    unsafe { instance.destroy_instance(None); }
                    DispatchError::DeviceCreationFailed(e.to_string())
                })?;

        // SAFETY: device is valid; queue was created with queue_family_index and index 0.
        let queue: vk::Queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        // ── Step 6: Command pool ──────────────────────────────────────────────
        let cp_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        // SAFETY: cp_info is valid; device is alive for the lifetime of the pool.
        let command_pool: vk::CommandPool =
            unsafe { device.create_command_pool(&cp_info, None) }
                .map_err(|e| {
                    // SAFETY: destroy device + instance on command pool creation failure.
                    unsafe {
                        device.destroy_device(None);
                        instance.destroy_instance(None);
                    }
                    DispatchError::DeviceCreationFailed(format!("create_command_pool: {e}"))
                })?;

        // ── Step 7: Cache memory properties ──────────────────────────────────
        // SAFETY: physical_device is valid; this is a read-only query.
        let memory_properties: vk::PhysicalDeviceMemoryProperties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        // ── Step 8: Cache device limits ───────────────────────────────────────
        // SAFETY: physical_device is valid; this is a read-only query.
        let limits: vk::PhysicalDeviceLimits = props.limits;
        let max_compute_work_group_count: [u32; 3] = limits.max_compute_work_group_count;

        Ok(Self {
            entry,
            instance,
            physical_device,
            device,
            queue,
            queue_family_index,
            command_pool,
            memory_properties,
            max_compute_work_group_count,
            device_name,
        })
    }

    /// Return the human-readable name of the selected physical device.
    ///
    /// Useful for diagnostic output in test failures.
    pub fn physical_device_name(&self) -> &str {
        &self.device_name
    }

    /// Return the cached `max_compute_work_group_count` device limit.
    ///
    /// Used by `validate_request` to pre-check workgroup counts before any
    /// Vulkan resource allocation.
    pub fn max_compute_work_group_count(&self) -> [u32; 3] {
        self.max_compute_work_group_count
    }
}

/// Find the index of the first queue family with `COMPUTE` support.
///
/// Returns `None` if no such family exists.
fn find_compute_queue_family(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Option<u32> {
    // SAFETY: physical_device is valid; this is a read-only query.
    let families =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

    for (i, family) in families.iter().enumerate() {
        if family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
            return Some(i as u32);
        }
    }
    None
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        // SAFETY: device is valid; device_wait_idle blocks until all submitted
        // commands complete — required before destroying the device on Lavapipe
        // to avoid VK_ERROR_DEVICE_LOST shutdown races.
        let _ = unsafe { self.device.device_wait_idle() };

        // SAFETY: command_pool was created from this device; it is valid.
        unsafe { self.device.destroy_command_pool(self.command_pool, None); }

        // SAFETY: device was created from this instance; no outstanding objects remain
        // (device_wait_idle ensured command completion; DispatchResources RAII cleaned
        // up per-dispatch handles before returning to callers).
        unsafe { self.device.destroy_device(None); }

        // SAFETY: instance was created in new(); the device using it has been destroyed.
        unsafe { self.instance.destroy_instance(None); }
    }
}
