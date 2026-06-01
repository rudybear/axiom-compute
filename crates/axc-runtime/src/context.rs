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
use crate::transfer_queue::{
    select_queue_families, queue_mode, build_device_queue_create_infos,
    concurrent_family_indices, QueueMode, QueueFamilySelection, TransferQueueInfo,
};
use crate::sync::{detect_sync_mode, create_handoff, SyncMode};

/// Configuration for `VulkanContext::new_with_options`.
///
/// `VulkanContext::new()` delegates to `new_with_options(VulkanContextOptions::from_env())`.
/// Tests use `new_with_options` directly to supply explicit paths and indices without
/// mutating the process environment.
///
/// ## Backward compatibility
///
/// Use `VulkanContextOptions { ..Default::default() }` to add only the fields you care
/// about without breaking when new fields are added.
#[derive(Default)]
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
    // ── M3.0 force options (AT-1409, AT-1418) ─────────────────────────────────
    /// Force single-queue mode regardless of hardware capabilities.
    ///
    /// `None` = auto-detect. `Some(true)` = always single-queue.
    /// Reads `AXC_FORCE_SINGLE_QUEUE=1` from env in `from_env()`.
    pub force_single_queue: Option<bool>,
    /// Force binary semaphores even on Vulkan 1.2 / timeline-capable devices.
    ///
    /// `None` = auto-detect. `Some(true)` = always binary.
    /// Reads `AXC_FORCE_BINARY_SEMAPHORES=1` from env in `from_env()`.
    pub force_binary_semaphores: Option<bool>,
    /// Force NonCoherent flush/invalidate even on coherent staging memory.
    ///
    /// `None` = auto. `Some(true)` = treat all staging as NonCoherent for CI coverage.
    /// Reads `AXC_FORCE_NONCOHERENT_STAGING=1` from env in `from_env()`.
    pub force_noncoherent_staging: Option<bool>,
}

impl VulkanContextOptions {
    /// Build options from environment variables.
    ///
    /// Reads `AXC_PHYSICAL_DEVICE_INDEX`, `AXC_FENCE_TIMEOUT_MS`,
    /// `AXC_FORCE_SINGLE_QUEUE`, `AXC_FORCE_BINARY_SEMAPHORES`,
    /// `AXC_FORCE_NONCOHERENT_STAGING`, and `resolve_pipeline_cache_path_from_env()`.
    pub fn from_env() -> Self {
        let read_bool_flag = |var: &str| -> Option<bool> {
            std::env::var(var).ok().map(|v| v == "1")
        };
        Self {
            pipeline_cache_path: resolve_pipeline_cache_path_from_env(),
            physical_device_index: std::env::var("AXC_PHYSICAL_DEVICE_INDEX")
                .ok()
                .and_then(|v: String| v.parse::<usize>().ok()),
            fence_timeout_ms: std::env::var("AXC_FENCE_TIMEOUT_MS")
                .ok()
                .and_then(|v: String| v.parse::<u64>().ok()),
            force_single_queue: read_bool_flag("AXC_FORCE_SINGLE_QUEUE"),
            force_binary_semaphores: read_bool_flag("AXC_FORCE_BINARY_SEMAPHORES"),
            force_noncoherent_staging: read_bool_flag("AXC_FORCE_NONCOHERENT_STAGING"),
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
    // ── M3.0 fields ───────────────────────────────────────────────────────────
    /// Optional dedicated transfer queue + command pool (DedicatedTransfer mode).
    transfer: Option<TransferQueueInfo>,
    /// Queue mode resolved at init time.
    queue_mode: QueueMode,
    /// Semaphore synchronization mode resolved at init time.
    sync_mode: SyncMode,
    /// `nonCoherentAtomSize` from device limits (WARNING-1: never hardcoded).
    non_coherent_atom_size: u64,
    /// Whether `force_noncoherent_staging` was set (AT-1409 / AT-1418).
    force_noncoherent_staging: bool,
    /// Context-level queue-submit lock (CRITICAL-5: atomic-group serialization).
    queue_submit_lock: Arc<Mutex<()>>,
    /// Per-pool lock for the compute command pool (WARNING-5).
    compute_pool_lock: Arc<Mutex<()>>,
    /// Per-pool lock for the transfer command pool (WARNING-5); `None` in SingleQueue.
    transfer_pool_lock: Option<Arc<Mutex<()>>>,
    /// Resolved queue family selection (stored for concurrent_families).
    queue_family_sel: QueueFamilySelection,
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

    /// Initialize Vulkan with explicit options (M3.0).
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

        // ── Step 2: Instance (M3.0: request Vulkan 1.2 if available) ─────────
        let app_name = std::ffi::CString::new("axc-runtime").unwrap();
        let engine_name = std::ffi::CString::new("axc-compute").unwrap();

        // Bump API version to min(reported, 1.2) so timeline semaphores are reachable.
        // We NEVER fail on 1.1 — binary semaphore fallback handles 1.1-only ICDs.
        // Use try_enumerate_instance_version (available in ash 0.38) which returns None
        // on Vulkan 1.0 implementations that do not support the call.
        // SAFETY: try_enumerate_instance_version is a read-only query on the entry.
        let instance_api_version: u32 = unsafe {
            entry.try_enumerate_instance_version()
                .ok()
                .flatten()
                .unwrap_or(vk::API_VERSION_1_1)
        };
        let requested_api_version: u32 = if instance_api_version >= vk::API_VERSION_1_2 {
            vk::API_VERSION_1_2
        } else {
            vk::API_VERSION_1_1
        };

        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(0)
            .engine_name(&engine_name)
            .engine_version(0)
            .api_version(requested_api_version);

        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info);

        // SAFETY: instance_info is valid for the duration of this call.
        let raw_instance: ash::Instance =
            unsafe { entry.create_instance(&instance_info, None) }
                .map_err(|e| DispatchError::NoVulkanInstance(e.to_string()))?;

        let instance_owner: Arc<InstanceOwner> = Arc::new(InstanceOwner {
            instance: raw_instance,
            entry,
        });
        let instance: &ash::Instance = &instance_owner.instance;

        // ── Step 3: Physical device selection ─────────────────────────────────
        // SAFETY: instance is valid.
        let physical_devices: Vec<vk::PhysicalDevice> =
            unsafe { instance.enumerate_physical_devices() }
                .map_err(|e| {
                    DispatchError::NoVulkanInstance(format!("enumerate_physical_devices: {e}"))
                })?;

        if physical_devices.is_empty() {
            return Err(DispatchError::NoSupportedDevice);
        }

        let device_index_override: Option<usize> = opts.physical_device_index
            .filter(|&i| i < physical_devices.len())
            .or_else(|| {
                std::env::var("AXC_PHYSICAL_DEVICE_INDEX")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .filter(|&i| i < physical_devices.len())
            });

        // Select physical device + queue family selection (M3.0: uses transfer_queue::select).
        let force_single_queue: bool = opts.force_single_queue.unwrap_or(false);
        let (physical_device, queue_family_sel): (vk::PhysicalDevice, QueueFamilySelection) =
            match device_index_override {
                Some(idx) => {
                    let pd = physical_devices[idx];
                    let sel = select_queue_families(instance, pd, force_single_queue)?;
                    (pd, sel)
                }
                None => {
                    let mut found: Option<(vk::PhysicalDevice, QueueFamilySelection)> = None;
                    for &pd in &physical_devices {
                        if let Ok(sel) = select_queue_families(instance, pd, force_single_queue) {
                            found = Some((pd, sel));
                            break;
                        }
                    }
                    found.ok_or(DispatchError::NoSupportedDevice)?
                }
            };

        let queue_family_index: u32 = queue_family_sel.compute_family;
        let q_mode: QueueMode = queue_mode(&queue_family_sel);

        // ── Step 4: Device name + properties ──────────────────────────────────
        // SAFETY: physical_device is valid.
        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        // SAFETY: props.device_name is a null-terminated C string per Vulkan spec
        // (VkPhysicalDeviceProperties::deviceName is char[256] with a NUL terminator).
        let device_name: String = unsafe {
            std::ffi::CStr::from_ptr(props.device_name.as_ptr())
                .to_string_lossy()
                .into_owned()
        };

        // ── Step 4b: Detect sync mode ─────────────────────────────────────────
        let force_binary: bool = opts.force_binary_semaphores.unwrap_or(false);
        let s_mode: SyncMode = detect_sync_mode(
            instance, physical_device, requested_api_version, force_binary,
        );

        // ── Step 5: Logical device + queues ───────────────────────────────────
        let queue_priorities: [f32; 1] = [1.0f32];
        let queue_create_infos = build_device_queue_create_infos(&queue_family_sel, &queue_priorities);

        // Chain Vulkan12Features for timelineSemaphore when applicable (WARNING-4).
        // The local binding MUST outlive the DeviceCreateInfo used in create_device.
        let mut vk12_features = vk::PhysicalDeviceVulkan12Features::default();
        let device_create_info: vk::DeviceCreateInfo<'_> =
            if s_mode == SyncMode::Timeline && requested_api_version >= vk::API_VERSION_1_2 {
                vk12_features.timeline_semaphore = vk::TRUE;
                vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queue_create_infos)
                    .push_next(&mut vk12_features)
            } else {
                vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queue_create_infos)
            };

        // SAFETY: device_create_info is valid; vk12_features outlives this call.
        let raw_device: ash::Device =
            unsafe { instance.create_device(physical_device, &device_create_info, None) }
                .map_err(|e| DispatchError::DeviceCreationFailed(e.to_string()))?;

        // SAFETY: queue was requested at index 0 of the compute family.
        let queue: vk::Queue = unsafe { raw_device.get_device_queue(queue_family_index, 0) };

        let device_owner: ManuallyDrop<Arc<DeviceOwner>> =
            ManuallyDrop::new(Arc::new(DeviceOwner { device: raw_device }));
        let instance_owner: ManuallyDrop<Arc<InstanceOwner>> = ManuallyDrop::new(instance_owner);

        // ── Step 6: Compute command pool ──────────────────────────────────────
        let cp_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: cp_info is valid.
        let command_pool: vk::CommandPool =
            unsafe { device_owner.create_command_pool(&cp_info, None) }
                .map_err(|e| {
                    DispatchError::DeviceCreationFailed(format!("create_command_pool: {e}"))
                })?;

        // ── Step 6b: Transfer command pool + queue (DedicatedTransfer mode) ───
        let transfer: Option<TransferQueueInfo> = if let Some(tf) = queue_family_sel.transfer_family {
            let tf_pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(tf)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            // SAFETY: tf_pool_info is valid.
            let tf_pool = unsafe { device_owner.create_command_pool(&tf_pool_info, None) }
                .map_err(|e| {
                    // SAFETY: compute pool was created above; destroy on error.
                    unsafe { device_owner.destroy_command_pool(command_pool, None); }
                    DispatchError::DeviceCreationFailed(format!("create_transfer_command_pool: {e}"))
                })?;
            // SAFETY: queue was requested at index 0 of the transfer family.
            let tf_queue = unsafe { device_owner.get_device_queue(tf, 0) };
            Some(TransferQueueInfo { family: tf, queue: tf_queue, command_pool: tf_pool })
        } else {
            None
        };

        // ── Step 7: Cache memory properties ──────────────────────────────────
        // SAFETY: physical_device is valid.
        let memory_properties: vk::PhysicalDeviceMemoryProperties =
            unsafe { instance_owner.instance.get_physical_device_memory_properties(physical_device) };

        // ── Step 8: Cache device limits ───────────────────────────────────────
        let limits: vk::PhysicalDeviceLimits = props.limits;
        let max_compute_work_group_count: [u32; 3] = limits.max_compute_work_group_count;
        let non_coherent_atom_size: u64 = limits.non_coherent_atom_size;

        // ── Step 9: Pipeline cache ────────────────────────────────────────────
        let pipeline_cache: PipelineCache =
            PipelineCache::new(&device_owner.device, opts.pipeline_cache_path)
                .unwrap_or_else(|e| {
                    tracing::warn!(reason = %e, "pipeline cache init failed — using disabled cache");
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

        let force_noncoherent: bool = opts.force_noncoherent_staging.unwrap_or(false);
        let transfer_pool_lock: Option<Arc<Mutex<()>>> =
            if transfer.is_some() { Some(Arc::new(Mutex::new(()))) } else { None };

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
            transfer,
            queue_mode: q_mode,
            sync_mode: s_mode,
            non_coherent_atom_size,
            force_noncoherent_staging: force_noncoherent,
            queue_submit_lock: Arc::new(Mutex::new(())),
            compute_pool_lock: Arc::new(Mutex::new(())),
            transfer_pool_lock,
            queue_family_sel,
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

        // Create handoff semaphores for the dedicated-transfer path (M3.0).
        // In SingleQueue mode, handoff = None (only the fence is used).
        let handoff = match self.queue_mode {
            crate::transfer_queue::QueueMode::DedicatedTransfer => {
                Some(
                    create_handoff(&self.device_owner.device, self.sync_mode)
                        .inspect_err(|_| {
                            // SAFETY: all of these handles were successfully created above;
                            // clean up on semaphore creation failure.
                            unsafe {
                                if let Some(pool) = descriptor_pool {
                                    self.device_owner.destroy_descriptor_pool(pool, None);
                                }
                                self.device_owner.destroy_fence(fence, None);
                                self.device_owner.destroy_pipeline(compiled.pipeline, None);
                                self.device_owner.destroy_pipeline_layout(compiled.pipeline_layout, None);
                                if let Some(dsl) = compiled.descriptor_set_layout {
                                    self.device_owner.destroy_descriptor_set_layout(dsl, None);
                                }
                                self.device_owner.destroy_shader_module(compiled.shader_module, None);
                            }
                        })?,
                )
            }
            crate::transfer_queue::QueueMode::SingleQueue => None,
        };

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
            // M3.0 fields:
            queue_mode: self.queue_mode,
            sync_mode: self.sync_mode,
            handoff,
            non_coherent_atom_size: self.non_coherent_atom_size,
            concurrent_families: concurrent_family_indices(&self.queue_family_sel),
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

        // Build the extended DispatchQueueCtx (M3.0).
        let (transfer_queue, transfer_command_pool) = match &self.transfer {
            Some(ti) => (ti.queue, ti.command_pool),
            None => (self.queue, self.command_pool), // SingleQueue: reuse compute handles
        };
        let queue_ctx = DispatchQueueCtx {
            command_pool: self.command_pool,
            queue: self.queue,
            transfer_queue,
            transfer_command_pool,
            non_coherent_atom_size: self.non_coherent_atom_size,
            queue_submit_lock: Arc::clone(&self.queue_submit_lock),
            compute_pool_lock: Arc::clone(&self.compute_pool_lock),
            transfer_pool_lock: self.transfer_pool_lock.as_ref().map(Arc::clone),
            force_noncoherent: self.force_noncoherent_staging,
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

        // Step 4a: destroy transfer command pool if it exists (M3.0).
        if let Some(ref ti) = self.transfer {
            // SAFETY: transfer pool was created from this device; device_wait_idle
            // ensures no commands are in-flight using this pool.
            unsafe { self.device_owner.destroy_command_pool(ti.command_pool, None); }
        }

        // Step 4b: destroy compute command pool.
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
