//! `VulkanContext` — Vulkan device lifecycle for AXIOM-Compute dispatch.
//!
//! `VulkanContext::new()` initializes the full Vulkan stack:
//! - Entry (loads the Vulkan shared library)
//! - Instance (Vulkan API 1.1, no validation layers by default)
//! - Physical device selection (first device with a compute queue family, or env override)
//! - Logical device + queue
//! - Command pool with `RESET_COMMAND_BUFFER`
//! - Cached device limits (`max_compute_work_group_count`) for pre-validation
//! - Cached memory properties for buffer allocation
//!
//! ## M2.3a additions
//!
//! - `Arc<DeviceOwner>` for shared device lifetime between context and `KernelHandle`s.
//! - `Arc<InstanceOwner>` for shared instance lifetime — ensures VkInstance outlives
//!   VkDevice even when `KernelHandle`s outlive `VulkanContext` (AT-827).
//! - `PipelineCache` for Vulkan pipeline caching (disk-backed via options).
//! - `parking_lot::Mutex<BTreeMap<KernelCacheKey, Weak<KernelHandleInner>>>` in-process
//!   kernel cache (no HashMap — anti-pattern #14).
//! - `prepare_kernel` / `dispatch_handle` prepare-once/dispatch-many API.
//!
//! ## Drop behavior
//!
//! `VulkanContext::drop` sequence:
//! 1. `device_wait_idle()` (blocks until all submitted commands complete)
//! 2. `pipeline_cache.save()` (non-fatal, logs via `tracing::warn!` on error)
//! 3. Destroy `vk::PipelineCache` handle
//! 4. Destroy command pool
//! 5. Drop `ManuallyDrop<Arc<DeviceOwner>>` explicitly → `vkDestroyDevice` if last ref
//! 6. Drop `ManuallyDrop<Arc<InstanceOwner>>` explicitly → `vkDestroyInstance` if last ref
//!
//! Steps 5 and 6 use `ManuallyDrop` to enforce VkDevice-before-VkInstance ordering
//! (Vulkan spec §3.3.3). Without `ManuallyDrop`, Rust field-drop order would run
//! AFTER the drop body, meaning step 6 (`destroy_instance`) would execute in the body
//! while `device_owner` had not yet been dropped by the field auto-drop.
//!
//! When `KernelHandle`s outlive the context (AT-827), both Arcs have `strong_count > 1`
//! at context drop time, so neither `vkDestroyDevice` nor `vkDestroyInstance` fires
//! until the last `KernelHandle` drops its Arc.
//!
//! ## Device selection
//!
//! 1. If `VulkanContextOptions::physical_device_index` is set and in range, use that.
//! 2. If `AXC_PHYSICAL_DEVICE_INDEX` is set and in range (via `from_env()`), use that.
//! 3. Otherwise, iterate physical devices in enumeration order and pick the first
//!    that has a compute queue family.

use std::collections::BTreeMap;
use std::mem::ManuallyDrop;
use std::path::PathBuf;
use std::sync::{Arc, Weak};

use ash::vk;
use axc_hir::ParamBindingPlan;
use parking_lot::Mutex;

use crate::device_owner::DeviceOwner;
use crate::instance_owner::InstanceOwner;
use crate::error::DispatchError;
use crate::pipeline_cache::{PipelineCache, resolve_pipeline_cache_path_from_env};
use crate::pipeline::build_compute_pipeline;
use crate::kernel_handle::{
    KernelHandle, KernelHandleInner, KernelCacheKey,
    make_cache_key, allocate_descriptor_pool_and_set,
    ensure_buffers_fit_with_mem_props, record_and_submit_dispatch,
    DispatchQueueCtx,
};
use crate::dispatch::validate_request;
use crate::dispatch::DispatchRequest;

/// Configuration for `VulkanContext::new_with_options`.
///
/// `VulkanContext::new()` delegates to `new_with_options(VulkanContextOptions::from_env())`.
/// Tests use `new_with_options` directly to supply explicit paths and indices without
/// mutating the process environment.
pub struct VulkanContextOptions {
    /// Path to the on-disk pipeline cache file.
    ///
    /// `None` disables the pipeline cache. Tests pass an explicit tempdir path.
    /// `VulkanContext::new()` resolves this via `resolve_pipeline_cache_path_from_env()`.
    pub pipeline_cache_path: Option<PathBuf>,
    /// Physical device index override.
    ///
    /// `None` falls back to the first compute-capable device. Tests may pass an
    /// explicit index to select a specific device.
    pub physical_device_index: Option<usize>,
    /// Fence timeout in milliseconds for `dispatch_handle`.
    ///
    /// `None` reads `AXC_FENCE_TIMEOUT_MS` from the environment, defaulting to 10,000 ms.
    pub fence_timeout_ms: Option<u64>,
}

impl VulkanContextOptions {
    /// Build options from environment variables.
    ///
    /// Reads `AXC_PHYSICAL_DEVICE_INDEX`, `AXC_FENCE_TIMEOUT_MS`, and
    /// `resolve_pipeline_cache_path_from_env()`.
    pub fn from_env() -> Self {
        Self {
            pipeline_cache_path: resolve_pipeline_cache_path_from_env(),
            physical_device_index: std::env::var("AXC_PHYSICAL_DEVICE_INDEX")
                .ok()
                .and_then(|v: String| v.parse::<usize>().ok()),
            fence_timeout_ms: std::env::var("AXC_FENCE_TIMEOUT_MS")
                .ok()
                .and_then(|v: String| v.parse::<u64>().ok()),
        }
    }
}

/// Initialized Vulkan context for compute dispatch.
///
/// One context per process is typical. Multiple contexts are safe (AT-516) but
/// each has its own device and command pool — useful for testing, not for production.
pub struct VulkanContext {
    /// Shared ownership of the Vulkan instance and entry.
    ///
    /// Wrapped in `ManuallyDrop` so `VulkanContext::drop` can explicitly take
    /// and drop it AFTER explicitly dropping `device_owner`. This satisfies
    /// Vulkan spec §3.3.3: VkDevice before VkInstance.
    ///
    /// `KernelHandleInner` also holds an `Arc<InstanceOwner>` clone, so the
    /// instance is not destroyed until all KernelHandles have been dropped.
    pub(crate) instance_owner: ManuallyDrop<Arc<InstanceOwner>>,
    /// Selected physical device.
    #[allow(dead_code)]
    pub(crate) physical_device: vk::PhysicalDevice,
    /// Arc-wrapped device owner — shared with any KernelHandles created from this context.
    ///
    /// Wrapped in `ManuallyDrop` so `VulkanContext::drop` can take ownership and
    /// explicitly drop it **before** dropping `instance_owner`. Vulkan spec §3.3.3
    /// requires all VkDevice objects to be destroyed before their parent VkInstance.
    pub(crate) device_owner: ManuallyDrop<Arc<DeviceOwner>>,
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
    /// On-disk and in-memory Vulkan pipeline cache.
    pipeline_cache: PipelineCache,
    /// Process-local kernel cache: maps KernelCacheKey → weak ref to KernelHandleInner.
    ///
    /// Weak refs allow handles to be freed normally; the cache simply doesn't serve
    /// stale entries. BTreeMap enforces no-HashMap invariant (#14).
    in_mem_kernel_cache: Mutex<BTreeMap<KernelCacheKey, Weak<KernelHandleInner>>>,
    /// Fence timeout in milliseconds, resolved at context creation time.
    /// Stored for use in `dispatch_handle`; shadowed by env var if set after context init.
    #[allow(dead_code)]
    fence_timeout_ms: u64,
}

impl VulkanContext {
    /// Initialize Vulkan and create a compute-capable device context.
    ///
    /// Equivalent to `new_with_options(VulkanContextOptions::from_env())`.
    ///
    /// Returns `Err(DispatchError::NoSupportedDevice)` if no physical device
    /// with a compute queue family is found.
    pub fn new() -> Result<Self, DispatchError> {
        Self::new_with_options(VulkanContextOptions::from_env())
    }

    /// Initialize Vulkan with explicit options (M2.3a).
    ///
    /// Preferred over `new()` in tests: pass an explicit `pipeline_cache_path`
    /// (e.g., a tempdir) to avoid env-based path resolution and serial_test.
    pub fn new_with_options(opts: VulkanContextOptions) -> Result<Self, DispatchError> {
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
        let raw_instance: ash::Instance =
            unsafe { entry.create_instance(&instance_info, None) }
                .map_err(|e| DispatchError::NoVulkanInstance(e.to_string()))?;

        // Wrap entry + instance in Arc<InstanceOwner> so they can be shared with
        // KernelHandles that outlive this VulkanContext (AT-827). vkDestroyInstance
        // fires when the last Arc<InstanceOwner> drops.
        let instance_owner: Arc<InstanceOwner> = Arc::new(InstanceOwner {
            instance: raw_instance,
            entry,
        });

        // Alias for brevity in the remainder of this function.
        let instance: &ash::Instance = &instance_owner.instance;

        // ── Step 3: Physical device selection ─────────────────────────────────
        // SAFETY: instance is valid; enumerate_physical_devices is a read-only query.
        let physical_devices: Vec<vk::PhysicalDevice> =
            unsafe { instance.enumerate_physical_devices() }
                .map_err(|e| {
                    // Drop Arc<InstanceOwner> — fires vkDestroyInstance.
                    drop(Arc::clone(&instance_owner));
                    DispatchError::NoVulkanInstance(format!("enumerate_physical_devices: {e}"))
                })?;

        if physical_devices.is_empty() {
            // InstanceOwner drops here → vkDestroyInstance.
            return Err(DispatchError::NoSupportedDevice);
        }

        // Determine which physical device to use.
        let device_index_override: Option<usize> = opts.physical_device_index
            .filter(|&i| i < physical_devices.len())
            .or_else(|| {
                std::env::var("AXC_PHYSICAL_DEVICE_INDEX")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .filter(|&i| i < physical_devices.len())
            });

        let (physical_device, queue_family_index): (vk::PhysicalDevice, u32) =
            match device_index_override {
                Some(idx) => {
                    let pd: vk::PhysicalDevice = physical_devices[idx];
                    // SAFETY: pd is a valid physical device.
                    let qf_idx = find_compute_queue_family(instance, pd)
                        .ok_or(DispatchError::NoComputeQueue)?;
                    (pd, qf_idx)
                }
                None => {
                    let mut found: Option<(vk::PhysicalDevice, u32)> = None;
                    for &pd in &physical_devices {
                        // SAFETY: pd is a valid physical device.
                        if let Some(qf) = find_compute_queue_family(instance, pd) {
                            found = Some((pd, qf));
                            break;
                        }
                    }
                    found.ok_or(DispatchError::NoSupportedDevice)?
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
        let raw_device: ash::Device =
            unsafe { instance.create_device(physical_device, &device_create_info, None) }
                .map_err(|e| DispatchError::DeviceCreationFailed(e.to_string()))?;

        // SAFETY: raw_device is valid; queue was created with queue_family_index and index 0.
        let queue: vk::Queue = unsafe { raw_device.get_device_queue(queue_family_index, 0) };

        // Wrap device in Arc<DeviceOwner> inside ManuallyDrop — vkDestroyDevice fires
        // when the last Arc drops. ManuallyDrop allows VulkanContext::drop to explicitly
        // take ownership and drop the Arc before instance_owner (Vulkan spec §3.3.3).
        let device_owner: ManuallyDrop<Arc<DeviceOwner>> =
            ManuallyDrop::new(Arc::new(DeviceOwner { device: raw_device }));

        // Wrap instance_owner in ManuallyDrop so Drop can explicitly take it AFTER
        // device_owner has been dropped, satisfying VkDevice-before-VkInstance ordering.
        let instance_owner: ManuallyDrop<Arc<InstanceOwner>> = ManuallyDrop::new(instance_owner);

        // ── Step 6: Command pool ──────────────────────────────────────────────
        let cp_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        // SAFETY: cp_info is valid; device is alive for the lifetime of the pool.
        let command_pool: vk::CommandPool =
            unsafe { device_owner.create_command_pool(&cp_info, None) }
                .map_err(|e| {
                    DispatchError::DeviceCreationFailed(format!("create_command_pool: {e}"))
                })?;

        // ── Step 7: Cache memory properties ──────────────────────────────────
        // SAFETY: physical_device is valid; this is a read-only query.
        let memory_properties: vk::PhysicalDeviceMemoryProperties =
            unsafe { instance_owner.instance.get_physical_device_memory_properties(physical_device) };

        // ── Step 8: Cache device limits ───────────────────────────────────────
        // SAFETY: physical_device is valid; this is a read-only query.
        let limits: vk::PhysicalDeviceLimits = props.limits;
        let max_compute_work_group_count: [u32; 3] = limits.max_compute_work_group_count;

        // ── Step 9: Pipeline cache ────────────────────────────────────────────
        let pipeline_cache: PipelineCache =
            PipelineCache::new(&device_owner.device, opts.pipeline_cache_path)
                .unwrap_or_else(|e| {
                    tracing::warn!(reason = %e, "pipeline cache init failed — using disabled cache");
                    // Fallback: create with None (disabled). Should not fail since it just
                    // calls vkCreatePipelineCache with 0 initial data — but if it does,
                    // we cannot recover here. The unwrap below is intentional in that
                    // catastrophic case; in practice Vulkan always allows empty cache creation.
                    PipelineCache::new(&device_owner.device, None)
                        .expect("empty pipeline cache creation must not fail")
                });

        // ── Step 10: Fence timeout ────────────────────────────────────────────
        let fence_timeout_ms: u64 = opts.fence_timeout_ms
            .or_else(|| {
                std::env::var("AXC_FENCE_TIMEOUT_MS")
                    .ok()
                    .and_then(|v: String| v.parse::<u64>().ok())
            })
            .unwrap_or(crate::dispatch::DEFAULT_FENCE_TIMEOUT_MS);

        Ok(Self {
            instance_owner,
            physical_device,
            device_owner,
            queue,
            queue_family_index,
            command_pool,
            memory_properties,
            max_compute_work_group_count,
            device_name,
            pipeline_cache,
            in_mem_kernel_cache: Mutex::new(BTreeMap::new()),
            fence_timeout_ms,
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

    /// Prepare (compile) a kernel and return a reusable `KernelHandle`.
    ///
    /// The handle caches: shader module, DSL (optional for 0-buffer kernels),
    /// pipeline layout, pipeline, descriptor pool+set (optional), and a reusable
    /// fence. Subsequent `dispatch_handle` calls reuse all of these.
    ///
    /// ## Double-checked locking (W-3)
    ///
    /// 1. Lock the kernel cache and look up `key`.
    /// 2. If a live `Arc` is found, return it immediately.
    /// 3. **Drop the lock** before calling `build_compute_pipeline` (may take 5–20 ms).
    /// 4. Re-lock and re-check: another thread may have inserted while we compiled.
    /// 5. If still absent, insert the freshly compiled handle.
    ///
    /// If another thread won the race, the freshly-built `KernelHandleInner` is
    /// dropped (its Drop cleans up all Vulkan objects). Wasted compile work is
    /// bounded to one per cold-miss race.
    pub fn prepare_kernel(
        &self,
        spirv: &[u32],
        binding_plan: &ParamBindingPlan,
        push_constant_total_bytes: u32,
        entry_point: &str,
    ) -> Result<KernelHandle, DispatchError> {
        let key: KernelCacheKey = make_cache_key(spirv, binding_plan, push_constant_total_bytes);

        // Phase 1: check cache under lock.
        {
            let guard = self.in_mem_kernel_cache.lock();
            if let Some(weak) = guard.get(&key) {
                if let Some(arc) = weak.upgrade() {
                    return Ok(KernelHandle { inner: arc });
                }
                // Stale entry (KernelHandle was dropped); fall through.
            }
        } // Guard released here — compile outside the lock (W-3).

        // Phase 2: compile pipeline OUTSIDE the lock.
        let compiled = build_compute_pipeline(
            &self.device_owner.device,
            spirv,
            binding_plan,
            entry_point,
            self.pipeline_cache.vk(),
        )?;

        // P-5: allocate descriptor pool + set only for kernels with buffer bindings.
        let (descriptor_pool, descriptor_set): (Option<vk::DescriptorPool>, Option<vk::DescriptorSet>) =
            if binding_plan.buffers.is_empty() {
                (None, None)
            } else {
                let dsl: vk::DescriptorSetLayout = compiled.descriptor_set_layout
                    .expect("non-empty buffers must produce a DSL");
                let (pool, set) = allocate_descriptor_pool_and_set(
                    &self.device_owner.device,
                    dsl,
                    binding_plan.buffers.len(),
                )?;
                (Some(pool), Some(set))
            };

        // Create the reusable fence (P-2).
        let fence_info = vk::FenceCreateInfo::default();
        let fence: vk::Fence =
            // SAFETY: fence_info is valid; the fence will be destroyed in KernelHandleInner::drop.
            unsafe { self.device_owner.create_fence(&fence_info, None) }
                .map_err(|e| {
                    // SAFETY: clean up compiled pipeline resources on fence creation failure.
                    unsafe {
                        if let Some(pool) = descriptor_pool {
                            self.device_owner.destroy_descriptor_pool(pool, None);
                        }
                        self.device_owner.destroy_pipeline(compiled.pipeline, None);
                        self.device_owner.destroy_pipeline_layout(compiled.pipeline_layout, None);
                        if let Some(dsl) = compiled.descriptor_set_layout {
                            self.device_owner.destroy_descriptor_set_layout(dsl, None);
                        }
                        self.device_owner.destroy_shader_module(compiled.shader_module, None);
                    }
                    DispatchError::CommandBufferRecordFailed(format!("create_fence: {e}"))
                })?;

        let inner_fresh: Arc<KernelHandleInner> = Arc::new(KernelHandleInner {
            _instance_owner: Arc::clone(&self.instance_owner),
            device: Arc::clone(&self.device_owner),
            shader_module: compiled.shader_module,
            descriptor_set_layout: compiled.descriptor_set_layout,
            pipeline_layout: compiled.pipeline_layout,
            pipeline: compiled.pipeline,
            descriptor_pool,
            descriptor_set,
            fence,
            _entry_point_cstr: compiled.entry_point_cstr,
            binding_plan: binding_plan.clone(),
            buffers: Mutex::new(Vec::new()),
            cache_key: key.clone(),
            spirv_word_count: spirv.len(),
        });

        // Phase 3: re-lock and re-check (lost-race detection, W-3).
        let mut guard = self.in_mem_kernel_cache.lock();
        if let Some(weak) = guard.get(&key) {
            if let Some(arc) = weak.upgrade() {
                // Another thread compiled and inserted while we held the lock.
                // Discard our fresh build (its Drop cleans up Vulkan objects).
                drop(inner_fresh);
                return Ok(KernelHandle { inner: arc });
            }
        }
        guard.insert(key, Arc::downgrade(&inner_fresh));
        Ok(KernelHandle { inner: inner_fresh })
    }

    /// Execute a prepared kernel and return output buffer bytes.
    ///
    /// Acquires the per-handle buffer mutex, grows buffers if needed,
    /// uploads inputs, records + submits a command buffer, waits on the
    /// reusable fence, and reads back outputs.
    ///
    /// ## Concurrency
    ///
    /// Concurrent `dispatch_handle` calls on the SAME handle serialize at the
    /// `parking_lot::Mutex` (P-4). Calls on DIFFERENT handles run in parallel.
    pub fn dispatch_handle(
        &self,
        handle: &KernelHandle,
        workgroups: (u32, u32, u32),
        inputs: &[&[u8]],
        output_sizes: &[usize],
        push_constants: &[u8],
    ) -> Result<Vec<Vec<u8>>, DispatchError> {
        // Validate arguments before acquiring the buffer mutex.
        let dummy_req = DispatchRequest {
            spirv: &[],
            binding_plan: &handle.inner.binding_plan,
            workgroups: [workgroups.0, workgroups.1, workgroups.2],
            inputs,
            output_sizes,
            push_constants,
            entry_point: "",
        };
        validate_request(&dummy_req, self.max_compute_work_group_count)?;

        // Acquire the per-handle buffer mutex for the entire grow→submit→readback sequence (P-4).
        let mut buffers_guard = handle.inner.buffers.lock();

        // Grow buffers if needed (ensure_buffers_fit_with_mem_props).
        ensure_buffers_fit_with_mem_props(
            &mut buffers_guard,
            &handle.inner,
            inputs,
            output_sizes,
            &self.memory_properties,
        )?;

        // Record, submit, wait, readback.
        let queue_ctx = DispatchQueueCtx {
            command_pool: self.command_pool,
            queue: self.queue,
        };
        let outputs = record_and_submit_dispatch(
            &handle.inner,
            &queue_ctx,
            &buffers_guard,
            inputs,
            output_sizes,
            push_constants,
            workgroups,
        )?;

        drop(buffers_guard);
        Ok(outputs)
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
        // Step 1: wait for all GPU work to complete.
        // SAFETY: device is valid; device_wait_idle blocks until all submitted
        // commands complete — required before destroying resources on Lavapipe
        // to avoid VK_ERROR_DEVICE_LOST shutdown races.
        let _ = unsafe { self.device_owner.device_wait_idle() };

        // Step 2: save pipeline cache to disk (non-fatal).
        if let Err(e) = self.pipeline_cache.save(&self.device_owner.device) {
            tracing::warn!(reason = %e, "pipeline cache save failed on context drop");
        }

        // Step 3: destroy the vk::PipelineCache handle.
        // SAFETY: pipeline_cache.vk_handle was created from this device and is valid.
        unsafe {
            self.device_owner
                .destroy_pipeline_cache(self.pipeline_cache.vk_handle, None);
        }

        // Step 4: destroy command pool.
        // SAFETY: command_pool was created from this device; it is valid.
        unsafe { self.device_owner.destroy_command_pool(self.command_pool, None); }

        // Step 5: explicitly drop Arc<DeviceOwner> via ManuallyDrop::take.
        //
        // This fires vkDestroyDevice HERE if this context holds the last Arc ref.
        // If KernelHandles still hold Arcs, strong_count > 1 and the device lives
        // until the last KernelHandle drops.
        //
        // SAFETY: device_owner was initialized in new_with_options() and has not
        // been taken before (this Drop impl runs exactly once per VulkanContext).
        // After ManuallyDrop::take, self.device_owner is uninitialized — no code
        // below accesses it.
        let owned_device: Arc<DeviceOwner> =
            unsafe { ManuallyDrop::take(&mut self.device_owner) };
        drop(owned_device); // → vkDestroyDevice if last Arc ref.

        // Step 6: explicitly drop Arc<InstanceOwner> via ManuallyDrop::take.
        //
        // This fires vkDestroyInstance HERE if this context holds the last Arc ref.
        // Both Arcs must be taken explicitly because ManuallyDrop fields are NOT
        // auto-dropped by Rust — the code here IS the only destructor that runs.
        //
        // SAFETY: instance_owner was initialized in new_with_options() and has not
        // been taken before. After ManuallyDrop::take, self.instance_owner is
        // uninitialized — this is the last operation in the drop body.
        //
        // Ordering guarantee: step 5 (vkDestroyDevice) executes before step 6
        // (vkDestroyInstance), satisfying Vulkan spec §3.3.3 (VkDevice-before-VkInstance).
        // If KernelHandles outlive this context, both vkDestroyDevice AND vkDestroyInstance
        // are deferred to when the last KernelHandle drops (which holds Arcs to both).
        let owned_instance: Arc<InstanceOwner> =
            unsafe { ManuallyDrop::take(&mut self.instance_owner) };
        drop(owned_instance); // → vkDestroyInstance if last Arc ref.
    }
}
