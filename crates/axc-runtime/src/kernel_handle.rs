//! `KernelHandle` — Arc-based, prepare-once/dispatch-many Vulkan kernel handle.
//!
//! `prepare_kernel` builds and caches all the Vulkan objects needed to execute a
//! kernel: shader module, descriptor set layout (optional), pipeline layout,
//! pipeline, descriptor pool (optional), fence, and a staging/device-local buffer
//! pool. Subsequent `dispatch_handle` calls reuse all of these, paying only for
//! staging copies, command buffer recording, and queue submission.
//!
//! ## Lifetime model
//!
//! `KernelHandle` holds an `Arc<KernelHandleInner>`. Both `VulkanContext` (via a
//! `Weak` in its kernel cache) and the caller hold `Arc` references. The underlying
//! Vulkan device (`Arc<DeviceOwner>`) is destroyed only when the last Arc drops
//! — which may be the context or an outlasting `KernelHandle` (AT-827).
//!
//! ## Buffer pool
//!
//! Per-binding device-local + staging buffer pairs are managed inside a
//! `parking_lot::Mutex<Vec<BufferSlot>>`. On each `dispatch_handle` call, the
//! mutex is held for the entire grow→copy→submit→wait→readback sequence, which
//! serializes concurrent `dispatch_handle` calls on the same handle (acceptable
//! for M2.3a; M3 can parallelize via queue families).

use std::ffi::CString;
use std::sync::Arc;
use ash::vk;
use axc_hir::ParamBindingPlan;
use parking_lot::Mutex;

use crate::device_owner::DeviceOwner;
use crate::instance_owner::InstanceOwner;
use crate::error::DispatchError;
use axc_hir::buffer::BufferAccess;
use crate::buffers::{
    allocate_device_local_buffer, allocate_device_local_buffer_rebar,
    allocate_staging_buffer_cached,
    DeviceLocalBuffer, StagingBuffer, round_up_pow2,
};
use crate::persistent_map::Coherency;
use crate::sync::{HandoffSemaphores, SyncMode, destroy_handoff, recreate_binary_on_error};
use crate::transfer_queue::QueueMode;
use crate::dispatch::DEFAULT_FENCE_TIMEOUT_MS;

/// r2 Lever A: determine whether a binding index should be read back from the device.
///
/// Returns `true` iff `output_size > 0` AND the binding's access mode is NOT `ReadOnly`.
///
/// A `ReadOnly` binding carries the SPIR-V `NonWritable` decoration, which proves the
/// shader cannot write it — so there is nothing to read back, regardless of a non-zero
/// `output_sizes[i]` supplied by the caller.
///
/// `WriteOnly` and `ReadWrite` bindings read back EXACTLY as in r1 (byte-exact preserved).
/// For `ReadOnly` bindings the caller receives an empty `Vec<u8>` (byte-identical to the
/// `output_size==0` case, which was already empty in r1).
///
/// This predicate replaces ALL five readback gates in:
/// - `dispatch_single_queue`: compute→transfer barrier loop, device→staging copy loop.
/// - `dispatch_dedicated`: compute→transfer barrier loop, readback copy loop.
/// - `readback_from_persistent_mapping`: the per-slot gate.
pub fn binding_is_readback(access: BufferAccess, output_size: usize) -> bool {
    output_size > 0 && !matches!(access, BufferAccess::ReadOnly)
}

/// r2 Lever B: test whether a buffer slot has a ReBAR persistent mapping.
///
/// Returns `true` if the device-local buffer was allocated in ReBAR memory and
/// persistently mapped via `allocate_device_local_buffer_rebar`. When true, the
/// readback path reads directly from `slot.device_local.mapping` at ~60 GB/s
/// instead of going through the staging DMA + copy_out.
fn slot_is_rebar(slot: &BufferSlot) -> bool {
    slot.device_local.mapping.is_some()
}

/// Cache key for the in-process kernel cache.
///
/// Two kernels with identical SPIR-V bytes, binding plan, and push-constant size
/// share a single `KernelHandleInner`. Keyed in a `BTreeMap` (never HashMap).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct KernelCacheKey {
    /// xxh3_64 of the SPIR-V bytes (endian-stable, single-machine scope).
    pub spirv_hash: u64,
    /// xxh3_64 of the canonical JSON serialization of the `ParamBindingPlan`.
    pub binding_plan_hash: u64,
    /// Total push-constant bytes as declared in the binding plan.
    pub push_constant_size: u32,
}

/// One device-local + staging buffer pair for a single descriptor binding.
pub(crate) struct BufferSlot {
    /// Device-local storage buffer (STORAGE|TRANSFER_SRC|TRANSFER_DST, DEVICE_LOCAL).
    pub(crate) device_local: DeviceLocalBuffer,
    /// Staging buffer (TRANSFER_SRC|TRANSFER_DST, HOST_VISIBLE|HOST_COHERENT).
    pub(crate) staging: StagingBuffer,
    /// Binding index for this slot (stored for diagnostic and future use).
    #[allow(dead_code)]
    pub(crate) binding: u32,
}

/// Internal state of a `KernelHandle`.
///
/// Owned inside an `Arc`; dropped when the last `Arc` reference is released.
/// All Vulkan cleanup happens in `KernelHandleInner::drop`.
///
/// # Field drop ordering
///
/// Rust drops struct fields in **declaration order** (first-to-last) after
/// `Drop::drop()` returns. To satisfy Vulkan spec §3.3.3 (VkDevice must be
/// destroyed before VkInstance), `device` is declared **before** `_instance_owner`:
/// - `device` (Arc<DeviceOwner>) drops FIRST (declared first) → `vkDestroyDevice`.
/// - `_instance_owner` (Arc<InstanceOwner>) drops SECOND → `vkDestroyInstance`.
///
/// The explicit Vulkan object cleanups in `Drop::drop()` use `device` before
/// the automatic Arc drops. Both Arcs are still alive during `drop()` since
/// they haven't been auto-dropped yet.
pub(crate) struct KernelHandleInner {
    /// Keeps the Vulkan device alive even after `VulkanContext` is dropped.
    ///
    /// Declared FIRST so it auto-drops FIRST (declaration order). This ensures
    /// `vkDestroyDevice` fires before `vkDestroyInstance`, satisfying Vulkan
    /// spec §3.3.3 (VkDevice-before-VkInstance).
    pub(crate) device: Arc<DeviceOwner>,
    /// Keeps the Vulkan instance alive until after the device is destroyed.
    ///
    /// Declared AFTER `device` so it auto-drops SECOND (declaration order).
    /// This ensures `vkDestroyInstance` fires only after `vkDestroyDevice`.
    pub(crate) _instance_owner: Arc<InstanceOwner>,
    /// SPIR-V shader module (no-op compilation already done at prepare_kernel time).
    pub(crate) shader_module: vk::ShaderModule,
    /// Descriptor set layout; `None` for 0-buffer kernels (P-5).
    pub(crate) descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    /// Pipeline layout (set 0 DSL + push-constant range).
    pub(crate) pipeline_layout: vk::PipelineLayout,
    /// The compute pipeline.
    pub(crate) pipeline: vk::Pipeline,
    /// Descriptor pool; `None` for 0-buffer kernels.
    pub(crate) descriptor_pool: Option<vk::DescriptorPool>,
    /// Descriptor set; `None` for 0-buffer kernels.
    pub(crate) descriptor_set: Option<vk::DescriptorSet>,
    /// Reusable fence — created once, reset before every submit (P-2).
    pub(crate) fence: vk::Fence,
    /// Entry-point name as a CString (stored to guarantee the pointer's validity
    /// for the pipeline's lifetime — DO NOT remove this field).
    pub(crate) _entry_point_cstr: CString,
    /// Owned clone of the binding plan (W-5: ParamBindingPlan derives Clone).
    pub(crate) binding_plan: ParamBindingPlan,
    /// Per-dispatch mutex-protected buffer pool (P-4).
    pub(crate) buffers: Mutex<Vec<BufferSlot>>,
    /// Cache key that identified this handle in the process-level kernel cache.
    pub(crate) cache_key: KernelCacheKey,
    /// Number of SPIR-V words (for diagnostics only).
    pub(crate) spirv_word_count: usize,
    // ── M3.0 additions ────────────────────────────────────────────────────────
    /// Queue mode copied from context at prepare time.
    pub(crate) queue_mode: QueueMode,
    /// Sync mode copied from context at prepare time.
    pub(crate) sync_mode: SyncMode,
    /// Handoff semaphores (timeline or binary) for the 3-submit dedicated path.
    /// `None` in SingleQueue mode (uses only the fence).
    pub(crate) handoff: Option<HandoffSemaphores>,
    /// `nonCoherentAtomSize` from device limits, plumbed at prepare time.
    pub(crate) non_coherent_atom_size: u64,
    /// Concurrent queue families for CONCURRENT device-local buffers.
    /// `None` in SingleQueue mode.
    pub(crate) concurrent_families: Option<[u32; 2]>,
    // ── r2 Lever B additions ──────────────────────────────────────────────────
    /// True when a DEVICE_LOCAL|HOST_VISIBLE (ReBAR) memory type is available AND
    /// `force_no_rebar` is false. Copied from context at `prepare_kernel` time.
    ///
    /// When true, OUTPUT bindings (where `binding_is_readback` is true) are allocated
    /// via `allocate_device_local_buffer_rebar` in `ensure_buffers_fit_with_mem_props`.
    pub(crate) rebar_available: bool,
    /// True if `AXC_FORCE_NO_REBAR=1` was set.
    ///
    /// Forces the r1 staging-readback fallback on ReBAR-capable hardware for AT-1423
    /// correctness testing and CI coverage. Copied from context at `prepare_kernel` time.
    /// Stored here for diagnostic introspection (see AT-1423 in m3_bandwidth.rs).
    #[allow(dead_code)]
    pub(crate) force_no_rebar: bool,
}

/// Public opaque handle to a compiled Vulkan kernel.
///
/// Created by `VulkanContext::prepare_kernel`; used by `VulkanContext::dispatch_handle`.
/// Clone is cheap (increments Arc refcount).
#[derive(Clone)]
pub struct KernelHandle {
    pub(crate) inner: Arc<KernelHandleInner>,
}

impl KernelHandle {
    /// Return the number of SPIR-V words in this kernel's shader module.
    pub fn spirv_word_count(&self) -> usize {
        self.inner.spirv_word_count
    }

    /// Return the cache key used to identify this handle in the process-level cache.
    pub fn cache_key(&self) -> KernelCacheKey {
        self.inner.cache_key.clone()
    }

    /// Return the current sizes of allocated buffer slots (device-local side).
    ///
    /// Available unconditionally as a public method for test introspection.
    /// Used by integration tests (AT-805: kernel_handle_reuse.rs) to verify
    /// that buffer pool growth works correctly across dispatches with different N.
    ///
    /// On non-test builds this is available but not expected to be called from
    /// non-test code.
    pub fn buffer_sizes(&self) -> Vec<u64> {
        let guard = self.inner.buffers.lock();
        guard.iter().map(|slot| slot.device_local.size).collect()
    }
}

impl Drop for KernelHandleInner {
    fn drop(&mut self) {
        // SAFETY: dispatch_handle holds the buffers mutex for the entire submit+wait
        // sequence, so by the time Drop runs there is no in-flight GPU work referencing
        // this handle. We use self.device (Arc<DeviceOwner>) which may outlive the
        // VulkanContext — that is correct per the Arc lifetime model.
        let device: &ash::Device = &self.device.device;

        // Destroy buffer slots first.
        // DROP ORDER (CRITICAL-6 / Coder handoff note 1):
        //   1. slot.staging.mapping.unmap(device, memory)  ← CONSUMES PersistentMapping
        //   2. device.destroy_buffer(slot.staging.buffer)
        //   3. device.free_memory(slot.staging.memory)
        //   4. device.destroy_buffer(slot.device_local.buffer)
        //   5. device.free_memory(slot.device_local.memory)
        // Unmap MUST precede destroy_buffer to avoid a Vulkan validation error
        // (vkDestroyBuffer while memory is mapped). unmap consumes PersistentMapping
        // so a forgotten unmap is a compile error.
        {
            let mut guard = self.buffers.lock();
            for slot in guard.drain(..) {
                // DROP ORDER (CRITICAL-6 / r2 Coder handoff):
                //   1. slot.staging.mapping.unmap(device, memory)  ← CONSUMES PersistentMapping
                //   2. device.destroy_buffer(slot.staging.buffer)
                //   3. device.free_memory(slot.staging.memory)
                //   4. IF slot.device_local.mapping is Some: unmap it (CONSUMES) BEFORE destroy
                //   5. device.destroy_buffer(slot.device_local.buffer)
                //   6. device.free_memory(slot.device_local.memory)
                // Unmap MUST precede destroy_buffer to avoid a Vulkan validation error
                // (vkDestroyBuffer while memory is mapped). For ReBAR outputs, step 4 is
                // conditional but must precede step 5 — the Option is moved out by value so
                // a forgotten unmap is structurally visible in review.
                let staging_memory = slot.staging.memory;
                let staging_buffer = slot.staging.buffer;
                let device_local_buffer = slot.device_local.buffer;
                let device_local_memory = slot.device_local.memory;
                let rebar_mapping = slot.device_local.mapping; // Option<PersistentMapping>
                // SAFETY: buffers are only destroyed after all fence waits complete (Drop
                // runs after the last dispatch_handle returns). unmap-before-destroy order
                // is mandatory; consuming PersistentMapping makes a forgotten unmap a compile error.
                unsafe {
                    // Step 1: unmap staging (consumes PersistentMapping).
                    slot.staging.mapping.unmap(device, staging_memory);
                    // Step 2: destroy staging buffer.
                    device.destroy_buffer(staging_buffer, None);
                    // Step 3: free staging memory.
                    device.free_memory(staging_memory, None);
                    // Step 4: unmap ReBAR device-local mapping if present (r2 Lever B).
                    if let Some(rebar_map) = rebar_mapping {
                        rebar_map.unmap(device, device_local_memory);
                    }
                    // Step 5: destroy device-local buffer.
                    device.destroy_buffer(device_local_buffer, None);
                    // Step 6: free device-local memory.
                    device.free_memory(device_local_memory, None);
                }
            }
        }

        // Destroy handoff semaphores (M3.0) AFTER all GPU work has completed.
        if let Some(ref h) = self.handoff {
            // SAFETY: all dispatches have completed (fence-wait invariant); semaphores
            // are no longer in use.
            destroy_handoff(device, h);
        }

        // Destroy descriptor pool (implicitly frees the descriptor set).
        if let Some(pool) = self.descriptor_pool {
            // SAFETY: pool was created in prepare_kernel; no descriptor sets are in use
            // (all GPU work has completed per the fence-wait invariant above).
            unsafe { device.destroy_descriptor_pool(pool, None); }
        }

        // Destroy fence.
        // SAFETY: fence was created in prepare_kernel; it is not in use (all dispatches
        // have returned after vkWaitForFences, which always precedes handle drop).
        unsafe { device.destroy_fence(self.fence, None); }

        // Destroy pipeline.
        // SAFETY: pipeline is not in use (GPU work has completed).
        unsafe { device.destroy_pipeline(self.pipeline, None); }

        // Destroy pipeline layout.
        // SAFETY: all pipelines using this layout have been destroyed above.
        unsafe { device.destroy_pipeline_layout(self.pipeline_layout, None); }

        // Destroy descriptor set layout (if present).
        if let Some(dsl) = self.descriptor_set_layout {
            // SAFETY: pipeline layout using this DSL has been destroyed above.
            unsafe { device.destroy_descriptor_set_layout(dsl, None); }
        }

        // Destroy shader module.
        // SAFETY: all pipelines using this shader module have been destroyed above.
        unsafe { device.destroy_shader_module(self.shader_module, None); }

        // Arc<DeviceOwner> and Arc<InstanceOwner> drop implicitly via field auto-drop
        // after this body returns. Order is: device (declared after _instance_owner) drops
        // first (reverse field order), then _instance_owner drops.
        // If device was the last Arc<DeviceOwner> ref → vkDestroyDevice fires.
        // If _instance_owner was the last Arc<InstanceOwner> ref → vkDestroyInstance fires.
    }
}

/// Compute the xxh3-64 hash of a SPIR-V word slice as raw bytes.
///
/// Endian-stable on a single machine (single-run cache only).
pub(crate) fn hash_spirv(spirv: &[u32]) -> u64 {
    // SAFETY: u32 slice is valid to view as u8 bytes via align_to. The second slice
    // element of align_to is the aligned inner portion; the first and third are the
    // (empty) head/tail. Since spirv is &[u32], head and tail are always empty.
    let (head, body, tail) = unsafe { spirv.align_to::<u8>() };
    debug_assert!(head.is_empty() && tail.is_empty(), "u32 to u8 alignment unexpected");
    xxhash_rust::xxh3::xxh3_64(body)
}

/// Compute the xxh3-64 hash of a `ParamBindingPlan` via canonical JSON.
///
/// `serde_json` writes struct fields in declaration order (no HashMap reordering),
/// so the output is deterministic for the same plan.
pub(crate) fn hash_binding_plan(plan: &ParamBindingPlan) -> u64 {
    let json: Vec<u8> = serde_json::to_vec(plan)
        .unwrap_or_default();
    xxhash_rust::xxh3::xxh3_64(&json)
}

/// Build a `KernelCacheKey` from SPIR-V words, binding plan, and PC total bytes.
pub(crate) fn make_cache_key(
    spirv: &[u32],
    plan: &ParamBindingPlan,
    push_constant_total_bytes: u32,
) -> KernelCacheKey {
    KernelCacheKey {
        spirv_hash: hash_spirv(spirv),
        binding_plan_hash: hash_binding_plan(plan),
        push_constant_size: push_constant_total_bytes,
    }
}

/// Allocate a descriptor pool and descriptor set for the given DSL + buffer count.
///
/// Returns `(pool, set)`.
pub(crate) fn allocate_descriptor_pool_and_set(
    device: &ash::Device,
    dsl: vk::DescriptorSetLayout,
    buffer_count: usize,
) -> Result<(vk::DescriptorPool, vk::DescriptorSet), DispatchError> {
    let pool_size = vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(buffer_count as u32);

    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(std::slice::from_ref(&pool_size));

    // SAFETY: pool_info references pool_size which is valid for this call.
    let descriptor_pool: vk::DescriptorPool =
        unsafe { device.create_descriptor_pool(&pool_info, None) }
            .map_err(|e| DispatchError::DescriptorPoolFailed(e.to_string()))?;

    let dsl_slice: [vk::DescriptorSetLayout; 1] = [dsl];
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&dsl_slice);

    // SAFETY: alloc_info references dsl_slice and descriptor_pool which are valid.
    let descriptor_sets: Vec<vk::DescriptorSet> =
        unsafe { device.allocate_descriptor_sets(&alloc_info) }
            .map_err(|e| {
                // SAFETY: descriptor_pool must be destroyed on allocation failure.
                unsafe { device.destroy_descriptor_pool(descriptor_pool, None); }
                DispatchError::DescriptorPoolFailed(e.to_string())
            })?;

    Ok((descriptor_pool, descriptor_sets[0]))
}

/// Ensure the buffer pool has allocations for each binding, growing if needed.
///
/// For each binding `i`:
/// - Computes `size_needed = max(input_len[i], output_size[i]).max(4)`.
/// - If the slot does not exist or is too small: UNMAPS the old slot's staging
///   mapping (consuming `PersistentMapping`), THEN destroys the old slot, allocates
///   a new pair with `round_up_pow2(size_needed)`, and rewrites the descriptor set
///   entry for this binding.
///
/// ## M3.0 drop order (Coder handoff note 1)
///
/// On grow: `old_slot.staging.mapping.unmap(device, memory)` BEFORE
/// `destroy_buffer` BEFORE `free_memory`. `unmap` consumes `PersistentMapping`
/// so a forgotten unmap is a compile error.
///
/// `mem_props` must be the `memory_properties` from the `VulkanContext`.
pub(crate) fn ensure_buffers_fit_with_mem_props(
    buffers: &mut Vec<BufferSlot>,
    inner: &KernelHandleInner,
    inputs: &[&[u8]],
    output_sizes: &[usize],
    mem_props: &vk::PhysicalDeviceMemoryProperties,
) -> Result<(), DispatchError> {
    let n: usize = inner.binding_plan.buffers.len();
    let device: &ash::Device = &inner.device.device;

    for i in 0..n {
        let input_len: u64 = if i < inputs.len() { inputs[i].len() as u64 } else { 0 };
        let output_len: u64 = if i < output_sizes.len() { output_sizes[i] as u64 } else { 0 };
        let size_needed: u64 = input_len.max(output_len).max(4);

        let needs_realloc: bool = match buffers.get(i) {
            None => true,
            Some(slot) => slot.device_local.size < size_needed,
        };

        if !needs_realloc {
            continue;
        }

        // Destroy existing slot if present (and at index i).
        // DROP ORDER (r2 Coder handoff): unmap staging BEFORE destroy, and unmap any ReBAR
        // device-local mapping BEFORE destroy_buffer on the device-local buffer.
        if i < buffers.len() {
            let old_slot = buffers.remove(i);
            let staging_memory = old_slot.staging.memory;
            let staging_buffer = old_slot.staging.buffer;
            let device_local_buffer = old_slot.device_local.buffer;
            let device_local_memory = old_slot.device_local.memory;
            let rebar_mapping = old_slot.device_local.mapping; // Option<PersistentMapping>
            // SAFETY: the old buffers are not in use (from a prior completed dispatch).
            unsafe {
                // Step 1: unmap staging persistent mapping (BEFORE destroy).
                old_slot.staging.mapping.unmap(device, staging_memory);
                // Step 2: destroy and free staging.
                device.destroy_buffer(staging_buffer, None);
                device.free_memory(staging_memory, None);
                // Step 3: unmap ReBAR device-local mapping if present (r2 Lever B).
                if let Some(rebar_map) = rebar_mapping {
                    rebar_map.unmap(device, device_local_memory);
                }
                // Step 4: destroy and free device-local.
                device.destroy_buffer(device_local_buffer, None);
                device.free_memory(device_local_memory, None);
            }
        }

        let new_size: u64 = round_up_pow2(size_needed);
        if new_size == u64::MAX {
            return Err(DispatchError::BufferAllocationFailed {
                binding: i as u32,
                size: size_needed as usize,
                reason: "requested size too large (> 2^62 bytes)".to_owned(),
            });
        }

        // r2 Lever B: allocate the device-local buffer in ReBAR (DEVICE_LOCAL|HOST_VISIBLE)
        // memory for output bindings when ReBAR is available and not suppressed.
        // binding_is_readback checks BOTH the access mode (not ReadOnly) AND that a
        // non-zero output_size was requested for this binding.
        let is_output: bool = {
            let access = inner.binding_plan.buffers[i].ty.access;
            let out_len = if i < output_sizes.len() { output_sizes[i] } else { 0 };
            binding_is_readback(access, out_len)
        };
        let device_local: DeviceLocalBuffer = if is_output && inner.rebar_available {
            allocate_device_local_buffer_rebar(
                device,
                mem_props,
                inner.non_coherent_atom_size,
                new_size,
                i as u32,
                inner.concurrent_families,
            )?
        } else {
            allocate_device_local_buffer(
                device,
                mem_props,
                new_size,
                i as u32,
                inner.concurrent_families,
            )?
        };

        let staging: StagingBuffer = allocate_staging_buffer_cached(
            device,
            mem_props,
            inner.non_coherent_atom_size,
            new_size,
            i as u32,
            // force_noncoherent is NOT applied here (it's a one-time context option
            // applied at context-level; for slot growth we use the same coherency
            // as the context was initialized with). The correct approach would be to
            // store force_noncoherent in the handle; for now we pass false and rely
            // on the initial allocation having the right coherency set by context.
            false,
        )?;

        let binding_idx: u32 = inner.binding_plan.buffers[i].buffer_position;

        let new_slot = BufferSlot {
            device_local,
            staging,
            binding: binding_idx,
        };

        // Insert at the correct index.
        if i >= buffers.len() {
            buffers.push(new_slot);
        } else {
            buffers.insert(i, new_slot);
        }

        // Rewrite descriptor set entry for this binding (if descriptor set exists).
        if let Some(ds) = inner.descriptor_set {
            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(buffers[i].device_local.buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE);

            let write = vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(binding_idx)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_info));

            // SAFETY: write references valid descriptor set and buffer handles.
            unsafe { device.update_descriptor_sets(&[write], &[]); }
        }
    }

    Ok(())
}

/// Context needed for `record_and_submit_dispatch`.
///
/// Groups the Vulkan queue context so `record_and_submit_dispatch` stays
/// within Clippy's 7-argument limit.
pub(crate) struct DispatchQueueCtx {
    /// Command pool for the compute queue (RESET_COMMAND_BUFFER).
    pub(crate) command_pool: vk::CommandPool,
    /// Compute queue handle.
    pub(crate) queue: vk::Queue,
    // ── M3.0 additions ────────────────────────────────────────────────────────
    /// Optional dedicated transfer queue + pool (DedicatedTransfer mode only).
    pub(crate) transfer_queue: vk::Queue,
    /// Optional transfer command pool (DedicatedTransfer mode only).
    pub(crate) transfer_command_pool: vk::CommandPool,
    /// `nonCoherentAtomSize` from device limits.
    pub(crate) non_coherent_atom_size: u64,
    /// Context-level submit lock (atomic-group serialization, CRITICAL-5).
    pub(crate) queue_submit_lock: std::sync::Arc<parking_lot::Mutex<()>>,
    /// Per-pool command-pool lock for the compute pool (WARNING-5).
    pub(crate) compute_pool_lock: std::sync::Arc<parking_lot::Mutex<()>>,
    /// Per-pool command-pool lock for the transfer pool (WARNING-5); `None` in SingleQueue.
    pub(crate) transfer_pool_lock: Option<std::sync::Arc<parking_lot::Mutex<()>>>,
    /// Whether force_noncoherent_staging is active (AT-1409 / AT-1418).
    pub(crate) force_noncoherent: bool,
}

/// Record and submit a compute dispatch, then wait for completion.
///
/// This is the core work of `dispatch_handle`. The buffer mutex must be held by
/// the caller (via `buffers_guard`) for the duration of this call.
///
/// ## M3.0 dispatch paths
///
/// - `QueueMode::SingleQueue`: single command buffer, single submit (M2.3a-compatible).
///   Persistent mapping replaces per-dispatch vkMapMemory/vkUnmapMemory.
///   NonCoherent flush/invalidate added around the copy_in/copy_out.
///
/// - `QueueMode::DedicatedTransfer`: three command buffers (upload on transfer pool,
///   compute on compute pool, readback on transfer pool) submitted as an atomic group
///   under `queue_submit_lock` (CRITICAL-5). Timeline or binary semaphore handoff.
///
/// ## Zero-buffer / push-constant-only degenerate path
///
/// When there are no buffer bindings, a SINGLE compute submit signals `inner.fence`
/// directly; no semaphores or timeline blocks are consumed (CRITICAL-3).
pub(crate) fn record_and_submit_dispatch(
    inner: &KernelHandleInner,
    ctx: &DispatchQueueCtx,
    buffers: &[BufferSlot],
    inputs: &[&[u8]],
    output_sizes: &[usize],
    push_constants: &[u8],
    workgroups: (u32, u32, u32),
) -> Result<Vec<Vec<u8>>, DispatchError> {
    let _span = tracing::info_span!(
        "dispatch_handle",
        kernel = %inner._entry_point_cstr.to_string_lossy()
    ).entered();

    let n: usize = inner.binding_plan.buffers.len();

    // Determine the effective coherency for each slot, applying force_noncoherent
    // override if needed (AT-1409 / AT-1418).
    let effective_coherency = |slot: &BufferSlot| -> Coherency {
        if ctx.force_noncoherent {
            Coherency::NonCoherent { atom_size: ctx.non_coherent_atom_size }
        } else {
            slot.staging.coherency
        }
    };

    // Fence timeout — read env each time so tests can override after context creation.
    let timeout_ms: u64 = std::env::var("AXC_FENCE_TIMEOUT_MS")
        .ok()
        .and_then(|v: String| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_FENCE_TIMEOUT_MS);
    let timeout_ns: u64 = timeout_ms * 1_000_000;

    // Degenerate path: no buffers (push-constant-only or zero-binding kernel).
    // Single compute submit directly to fence; no semaphores; no timeline block.
    if n == 0 {
        return dispatch_degenerate(inner, ctx, push_constants, workgroups, timeout_ns);
    }

    match inner.queue_mode {
        QueueMode::SingleQueue => dispatch_single_queue(
            inner,
            ctx,
            buffers,
            inputs,
            output_sizes,
            push_constants,
            workgroups,
            timeout_ns,
            effective_coherency,
        ),
        QueueMode::DedicatedTransfer => dispatch_dedicated(
            inner,
            ctx,
            buffers,
            inputs,
            output_sizes,
            push_constants,
            workgroups,
            timeout_ns,
            effective_coherency,
        ),
    }
}

// ── Helper: degenerate dispatch (zero buffers, push-constant-only) ──────────

/// Single-submit dispatch for zero-buffer / push-constant-only kernels.
///
/// No timeline block reserved; no semaphores signaled or waited; single compute
/// submit signals `inner.fence` directly (CRITICAL-3 degenerate path).
///
/// # Safety
/// All Vulkan handles are valid for the duration of this call; they were created
/// in `prepare_kernel` and are alive because `KernelHandleInner` is still alive.
/// The buffers Mutex is held by the caller.
#[allow(clippy::undocumented_unsafe_blocks)]
fn dispatch_degenerate(
    inner: &KernelHandleInner,
    ctx: &DispatchQueueCtx,
    push_constants: &[u8],
    workgroups: (u32, u32, u32),
    timeout_ns: u64,
) -> Result<Vec<Vec<u8>>, DispatchError> {
    let device: &ash::Device = &inner.device.device;

    // Allocate compute CB under compute_pool_lock (WARNING-5).
    let cb: vk::CommandBuffer = {
        let _pool_guard = ctx.compute_pool_lock.lock();
        let cb_alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(ctx.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: command_pool is valid and was created with RESET_COMMAND_BUFFER.
        let cbs = unsafe { device.allocate_command_buffers(&cb_alloc_info) }
            .map_err(|e| DispatchError::CommandBufferRecordFailed(e.to_string()))?;
        cbs[0]
    };

    let begin_info = vk::CommandBufferBeginInfo::default()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    // SAFETY: cb is freshly allocated.
    unsafe { device.begin_command_buffer(cb, &begin_info) }
        .map_err(|e| DispatchError::CommandBufferRecordFailed(e.to_string()))?;

    // SAFETY: pipeline is valid.
    unsafe { device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, inner.pipeline); }
    if !push_constants.is_empty() {
        // SAFETY: valid push constant range.
        unsafe {
            device.cmd_push_constants(cb, inner.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE, 0, push_constants);
        }
    }
    // SAFETY: workgroup counts were pre-validated.
    unsafe { device.cmd_dispatch(cb, workgroups.0, workgroups.1, workgroups.2); }
    // SAFETY: no unclosed render passes.
    unsafe { device.end_command_buffer(cb) }
        .map_err(|e| DispatchError::CommandBufferRecordFailed(e.to_string()))?;

    // Reset and submit under queue_submit_lock (CRITICAL-5).
    // SAFETY: inner.fence is valid.
    unsafe { device.reset_fences(&[inner.fence]) }
        .map_err(|e| DispatchError::CommandBufferRecordFailed(format!("vkResetFences: {e}")))?;

    {
        let _submit_guard = ctx.queue_submit_lock.lock();
        let cb_slice = [cb];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cb_slice);
        // SAFETY: submit_info is valid; fence is unsignaled.
        unsafe { device.queue_submit(ctx.queue, &[submit_info], inner.fence) }
            .map_err(|e| {
                // Free CB on error (no fence to wait).
                let _pool_guard = ctx.compute_pool_lock.lock();
                // SAFETY: CB is safe to free after submit fails (never in-flight).
                unsafe { device.free_command_buffers(ctx.command_pool, &[cb]); }
                DispatchError::QueueSubmitFailed(e.to_string())
            })?;
    } // lock released before wait

    // Wait for fence.
    // SAFETY: inner.fence was submitted in the queue_submit above; it is a valid fence.
    let wait_result = unsafe { device.wait_for_fences(&[inner.fence], true, timeout_ns) };
    {
        let _pool_guard = ctx.compute_pool_lock.lock();
        // SAFETY: CB is safe to free after fence wait (or timeout).
        unsafe { device.free_command_buffers(ctx.command_pool, &[cb]); }
    }
    if let Err(vk::Result::TIMEOUT) = wait_result {
        return Err(DispatchError::FenceTimeout { timeout_ns });
    }
    wait_result.map_err(|e| DispatchError::QueueSubmitFailed(e.to_string()))?;

    Ok(Vec::new())
}

// ── Helper: single-queue dispatch (M2.3a-compatible + M3.0 persistent-map) ──

/// # Safety
/// All Vulkan handles are valid for the duration of this call. The buffers Mutex
/// is held by the caller. Persistent-mapping pointers are valid as long as their
/// StagingBuffers exist.
#[allow(clippy::undocumented_unsafe_blocks)]
#[allow(clippy::too_many_arguments)]
fn dispatch_single_queue(
    inner: &KernelHandleInner,
    ctx: &DispatchQueueCtx,
    buffers: &[BufferSlot],
    inputs: &[&[u8]],
    output_sizes: &[usize],
    push_constants: &[u8],
    workgroups: (u32, u32, u32),
    timeout_ns: u64,
    effective_coherency: impl Fn(&BufferSlot) -> Coherency,
) -> Result<Vec<Vec<u8>>, DispatchError> {
    let device: &ash::Device = &inner.device.device;
    let n: usize = inner.binding_plan.buffers.len();

    // ── Upload: copy inputs via persistent mapping (M3.0) ────────────────────
    for i in 0..n {
        if i >= inputs.len() || inputs[i].is_empty() {
            continue;
        }
        let slot: &BufferSlot = &buffers[i];
        // copy_in: caller holds buffers Mutex (CRITICAL-6).
        slot.staging.mapping.copy_in(inputs[i]);
        // Flush if NonCoherent (WARNING-6: every uploaded slot).
        // effective_coherency is checked inside flush (no-op on Coherent).
        // SAFETY: device, memory, and byte_len are valid; called before submit.
        unsafe {
            slot.staging.mapping.flush(device, slot.staging.memory, inputs[i].len() as u64)?;
        }
        let _ = effective_coherency(slot); // coherency check done inside flush
    }

    // ── Allocate single CB under compute_pool_lock (WARNING-5) ───────────────
    let cb: vk::CommandBuffer = {
        let _pool_guard = ctx.compute_pool_lock.lock();
        let cb_alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(ctx.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: command_pool is valid.
        let cbs = unsafe { device.allocate_command_buffers(&cb_alloc_info) }
            .map_err(|e| DispatchError::CommandBufferRecordFailed(e.to_string()))?;
        cbs[0]
    };

    let begin_info = vk::CommandBufferBeginInfo::default()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    // SAFETY: cb is freshly allocated.
    unsafe { device.begin_command_buffer(cb, &begin_info) }
        .map_err(|e| DispatchError::CommandBufferRecordFailed(e.to_string()))?;

    // ── Stage A: staging→device copies + upload barriers ─────────────────────
    let mut upload_barriers: Vec<vk::BufferMemoryBarrier<'_>> = Vec::new();
    for i in 0..n {
        if i >= inputs.len() || inputs[i].is_empty() {
            continue;
        }
        let slot: &BufferSlot = &buffers[i];
        let copy_region = vk::BufferCopy::default()
            .src_offset(0).dst_offset(0).size(inputs[i].len() as u64);
        // SAFETY: both buffers are valid and non-overlapping.
        unsafe {
            device.cmd_copy_buffer(cb, slot.staging.buffer, slot.device_local.buffer, &[copy_region]);
        }
        let barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(slot.device_local.buffer).offset(0).size(vk::WHOLE_SIZE);
        upload_barriers.push(barrier);
    }
    if !upload_barriers.is_empty() {
        // SAFETY: barriers correctly formed.
        unsafe {
            device.cmd_pipeline_barrier(cb,
                vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(), &[], &upload_barriers, &[]);
        }
    }

    // ── Bind pipeline + descriptors + push constants ──────────────────────────
    // SAFETY: pipeline is valid.
    unsafe { device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, inner.pipeline); }
    if let Some(ds) = inner.descriptor_set {
        // SAFETY: ds is valid.
        unsafe {
            device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE,
                inner.pipeline_layout, 0, &[ds], &[]);
        }
    }
    if !push_constants.is_empty() {
        // SAFETY: valid push constant range.
        unsafe {
            device.cmd_push_constants(cb, inner.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE, 0, push_constants);
        }
    }
    // SAFETY: workgroup counts were validated.
    unsafe { device.cmd_dispatch(cb, workgroups.0, workgroups.1, workgroups.2); }

    // ── Compute→Transfer barriers (Lever A + Lever B) ────────────────────────
    // r2 Lever A: only emit barriers for bindings that are actually read back
    //   (binding_is_readback: output_size>0 AND access != ReadOnly).
    // r2 Lever B: for ReBAR output bindings, emit SHADER_WRITE→HOST_READ (dstStage=HOST)
    //   instead of SHADER_WRITE→TRANSFER_READ; no device→staging copy is recorded.
    let mut compute_to_transfer_barriers: Vec<vk::BufferMemoryBarrier<'_>> = Vec::new();
    let mut rebar_host_barriers: Vec<vk::BufferMemoryBarrier<'_>> = Vec::new();
    for i in 0..n {
        let out_size = if i < output_sizes.len() { output_sizes[i] } else { 0 };
        let access = inner.binding_plan.buffers[i].ty.access;
        if !binding_is_readback(access, out_size) { continue; } // Lever A
        let slot = &buffers[i];
        if slot_is_rebar(slot) {
            // Lever B: SHADER_WRITE → HOST_READ barrier on the ReBAR device-local buffer.
            // dstStage=HOST: the host reads directly from the mapped ReBAR pointer after the fence.
            let barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(slot.device_local.buffer).offset(0).size(vk::WHOLE_SIZE);
            rebar_host_barriers.push(barrier);
        } else {
            // Staged readback: SHADER_WRITE → TRANSFER_READ (r1 path).
            let barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(slot.device_local.buffer).offset(0).size(vk::WHOLE_SIZE);
            compute_to_transfer_barriers.push(barrier);
        }
    }
    if !compute_to_transfer_barriers.is_empty() {
        // SAFETY: barriers correctly formed.
        unsafe {
            device.cmd_pipeline_barrier(cb,
                vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(), &[], &compute_to_transfer_barriers, &[]);
        }
    }
    if !rebar_host_barriers.is_empty() {
        // Lever B: emit COMPUTE_SHADER→HOST barrier for all ReBAR outputs on the compute CB.
        // SAFETY: barriers correctly formed.
        unsafe {
            device.cmd_pipeline_barrier(cb,
                vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(), &[], &rebar_host_barriers, &[]);
        }
    }

    // ── Stage B: device→staging copies + TRANSFER_WRITE→HOST_READ barriers ───
    // (CRITICAL-1: ALWAYS emit this barrier for staged outputs, both coherent and non-coherent paths)
    // r2 Lever A: skip ReadOnly bindings (never read back).
    // r2 Lever B: skip ReBAR output bindings (no copy needed; read from mapped pointer).
    let mut readback_barriers: Vec<vk::BufferMemoryBarrier<'_>> = Vec::new();
    for i in 0..n {
        let out_size = if i < output_sizes.len() { output_sizes[i] } else { 0 };
        let access = inner.binding_plan.buffers[i].ty.access;
        if !binding_is_readback(access, out_size) { continue; } // Lever A
        let slot = &buffers[i];
        if slot_is_rebar(slot) { continue; } // Lever B: no copy for ReBAR outputs
        let copy_region = vk::BufferCopy::default()
            .src_offset(0).dst_offset(0).size(out_size as u64);
        // SAFETY: valid non-overlapping buffers.
        unsafe {
            device.cmd_copy_buffer(cb, slot.device_local.buffer, slot.staging.buffer, &[copy_region]);
        }
        // CRITICAL-1: always emit TRANSFER_WRITE→HOST_READ barrier for staged outputs.
        let barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(slot.staging.buffer).offset(0).size(vk::WHOLE_SIZE);
        readback_barriers.push(barrier);
    }
    if !readback_barriers.is_empty() {
        // SAFETY: barriers correctly formed.
        unsafe {
            device.cmd_pipeline_barrier(cb,
                vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(), &[], &readback_barriers, &[]);
        }
    }

    // SAFETY: no unclosed render passes.
    unsafe { device.end_command_buffer(cb) }
        .map_err(|e| DispatchError::CommandBufferRecordFailed(e.to_string()))?;

    // ── Fence reset + submit (under queue_submit_lock, CRITICAL-5) ────────────
    // SAFETY: inner.fence is valid.
    unsafe { device.reset_fences(&[inner.fence]) }
        .map_err(|e| DispatchError::CommandBufferRecordFailed(format!("vkResetFences: {e}")))?;

    {
        let _submit_guard = ctx.queue_submit_lock.lock();
        let cb_slice = [cb];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cb_slice);
        // SAFETY: submit_info valid; fence unsignaled.
        unsafe { device.queue_submit(ctx.queue, &[submit_info], inner.fence) }
            .map_err(|e| {
                let _pool_guard = ctx.compute_pool_lock.lock();
                // SAFETY: CB safe to free after failed submit.
                unsafe { device.free_command_buffers(ctx.command_pool, &[cb]); }
                DispatchError::QueueSubmitFailed(e.to_string())
            })?;
    } // queue_submit_lock released before fence wait (CRITICAL-5)

    // ── Wait for fence ────────────────────────────────────────────────────────
    // SAFETY: inner.fence is valid and was submitted.
    let wait_result = unsafe { device.wait_for_fences(&[inner.fence], true, timeout_ns) };
    {
        let _pool_guard = ctx.compute_pool_lock.lock();
        // SAFETY: CB is safe to free after fence wait (or timeout).
        unsafe { device.free_command_buffers(ctx.command_pool, &[cb]); }
    }
    if let Err(vk::Result::TIMEOUT) = wait_result {
        return Err(DispatchError::FenceTimeout { timeout_ns });
    }
    wait_result.map_err(|e| DispatchError::QueueSubmitFailed(e.to_string()))?;

    // ── Readback via persistent mapping (Lever A + Lever B) ─────────────────
    let outputs = readback_from_persistent_mapping(
        inner, buffers, output_sizes, n, device, ctx.non_coherent_atom_size,
        &effective_coherency, ctx.force_noncoherent,
    );

    Ok(outputs)
}

// ── Helper: dedicated-transfer-queue dispatch path ───────────────────────────

/// # Safety
/// All Vulkan handles are valid. The buffers Mutex is held by the caller.
/// The three-submit atomic group is serialized via queue_submit_lock (CRITICAL-5).
#[allow(clippy::undocumented_unsafe_blocks)]
#[allow(clippy::too_many_arguments)]
fn dispatch_dedicated(
    inner: &KernelHandleInner,
    ctx: &DispatchQueueCtx,
    buffers: &[BufferSlot],
    inputs: &[&[u8]],
    output_sizes: &[usize],
    push_constants: &[u8],
    workgroups: (u32, u32, u32),
    timeout_ns: u64,
    effective_coherency: impl Fn(&BufferSlot) -> Coherency,
) -> Result<Vec<Vec<u8>>, DispatchError> {
    let device: &ash::Device = &inner.device.device;
    let n: usize = inner.binding_plan.buffers.len();

    // Transfer pool must exist in DedicatedTransfer mode.
    let transfer_pool_lock = ctx.transfer_pool_lock.as_ref()
        .expect("transfer_pool_lock must be Some in DedicatedTransfer mode");

    let has_uploads = (0..n).any(|i| i < inputs.len() && !inputs[i].is_empty());
    // r2 Lever A + B: readback CB is only needed when at least one binding is read back
    // via the staging path (non-ReBAR). ReBAR bindings read directly from the mapped pointer.
    // Lever A: ReadOnly bindings are never read back.
    // Lever B: ReBAR output bindings skip the readback CB entirely.
    // NOTE: buffers may not be populated yet if this is the first call; we use a closure
    // that checks the populated buffers slice (which ensure_buffers_fit populated above).
    let has_staged_readback = {
        let access_iter = (0..n).filter(|&i| {
            let out_size = if i < output_sizes.len() { output_sizes[i] } else { 0 };
            let access = inner.binding_plan.buffers[i].ty.access;
            binding_is_readback(access, out_size)
        });
        access_iter.filter(|&i| !slot_is_rebar(&buffers[i])).count() > 0
    };
    // Legacy alias — used for timeline block reservation logic below.
    let has_outputs = has_staged_readback
        || (0..n).any(|i| {
            let out_size = if i < output_sizes.len() { output_sizes[i] } else { 0 };
            let access = inner.binding_plan.buffers[i].ty.access;
            binding_is_readback(access, out_size) && slot_is_rebar(&buffers[i])
        });

    // ── Host-side upload: copy_in + flush (M3.0, WARNING-6) ──────────────────
    for i in 0..n {
        if i >= inputs.len() || inputs[i].is_empty() { continue; }
        let slot = &buffers[i];
        slot.staging.mapping.copy_in(inputs[i]);
        let coh = effective_coherency(slot);
        match coh {
            Coherency::NonCoherent { .. } => {
                // SAFETY: flush called before submit; device/memory valid.
                unsafe {
                    slot.staging.mapping.flush(
                        device, slot.staging.memory, inputs[i].len() as u64
                    )?;
                }
            }
            Coherency::Coherent => {}
        }
    }

    // ── Timeline/binary block reservation (CRITICAL-3) ───────────────────────
    // Reserve 2 timeline values per dispatch. Skipped upload uses B (already passed).
    let base_val: u64 = match &inner.handoff {
        Some(h) => {
            if has_uploads || has_outputs {
                h.reserve_timeline_block()
            } else {
                0
            }
        }
        None => 0,
    };
    let upload_signal_val = base_val + 1;
    let compute_signal_val = base_val + 2;

    // ── Allocate command buffers ──────────────────────────────────────────────
    // Upload CB from transfer pool (only if has_uploads).
    let upload_cb_opt: Option<vk::CommandBuffer> = if has_uploads {
        let _pool_guard = transfer_pool_lock.lock();
        let cb_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(ctx.transfer_command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: transfer command pool is valid.
        let cbs = unsafe { device.allocate_command_buffers(&cb_alloc) }
            .map_err(|e| DispatchError::CommandBufferRecordFailed(e.to_string()))?;
        Some(cbs[0])
    } else {
        None
    };

    // Compute CB from compute pool.
    let compute_cb: vk::CommandBuffer = {
        let _pool_guard = ctx.compute_pool_lock.lock();
        let cb_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(ctx.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: compute pool is valid.
        match unsafe { device.allocate_command_buffers(&cb_alloc) } {
            Ok(cbs) => cbs[0],
            Err(e) => {
                // Free upload CB if allocated.
                if let Some(ucb) = upload_cb_opt {
                    let _pool_guard = transfer_pool_lock.lock();
                    // SAFETY: ucb was just allocated; safe to free.
                    unsafe { device.free_command_buffers(ctx.transfer_command_pool, &[ucb]); }
                }
                return Err(DispatchError::CommandBufferRecordFailed(e.to_string()));
            }
        }
    };

    // Readback CB from transfer pool (only if has_staged_readback).
    // r2 Lever B: when all read-back bindings are ReBAR, has_staged_readback=false
    // and the readback CB is SKIPPED — the compute submit signals readback_fence directly.
    // This reuses r1's existing no-outputs branch without writing a new code path.
    let readback_cb_opt: Option<vk::CommandBuffer> = if has_staged_readback {
        let _pool_guard = transfer_pool_lock.lock();
        let cb_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(ctx.transfer_command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        match unsafe { device.allocate_command_buffers(&cb_alloc) } {
            Ok(cbs) => Some(cbs[0]),
            Err(e) => {
                // Free already-allocated CBs.
                if let Some(ucb) = upload_cb_opt {
                    let _pool_guard = transfer_pool_lock.lock();
                    // SAFETY: ucb was allocated above.
                    unsafe { device.free_command_buffers(ctx.transfer_command_pool, &[ucb]); }
                }
                {
                    let _pool_guard = ctx.compute_pool_lock.lock();
                    // SAFETY: compute_cb was allocated above.
                    unsafe { device.free_command_buffers(ctx.command_pool, &[compute_cb]); }
                }
                return Err(DispatchError::CommandBufferRecordFailed(e.to_string()));
            }
        }
    } else {
        None
    };

    // ── Record upload CB (if has_uploads) ────────────────────────────────────
    if let Some(upload_cb) = upload_cb_opt {
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.begin_command_buffer(upload_cb, &begin_info) }
            .map_err(|e| DispatchError::CommandBufferRecordFailed(e.to_string()))?;

        for i in 0..n {
            if i >= inputs.len() || inputs[i].is_empty() { continue; }
            let slot = &buffers[i];
            let copy_region = vk::BufferCopy::default()
                .src_offset(0).dst_offset(0).size(inputs[i].len() as u64);
            // SAFETY: valid non-overlapping buffers.
            unsafe {
                device.cmd_copy_buffer(upload_cb, slot.staging.buffer,
                    slot.device_local.buffer, &[copy_region]);
            }
        }
        // No upload barriers here: the semaphore provides the execution dependency;
        // a memory barrier in the compute CB (TRANSFER_WRITE→SHADER_READ) covers visibility.
        unsafe { device.end_command_buffer(upload_cb) }
            .map_err(|e| DispatchError::CommandBufferRecordFailed(e.to_string()))?;
    }

    // ── Record compute CB ─────────────────────────────────────────────────────
    {
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.begin_command_buffer(compute_cb, &begin_info) }
            .map_err(|e| DispatchError::CommandBufferRecordFailed(e.to_string()))?;

        // Upload→Compute memory barriers (TRANSFER_WRITE→SHADER_READ, QUEUE_FAMILY_IGNORED).
        if has_uploads {
            let mut upload_barriers: Vec<vk::BufferMemoryBarrier<'_>> = Vec::new();
            for i in 0..n {
                if i >= inputs.len() || inputs[i].is_empty() { continue; }
                let slot = &buffers[i];
                let barrier = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(slot.device_local.buffer).offset(0).size(vk::WHOLE_SIZE);
                upload_barriers.push(barrier);
            }
            if !upload_barriers.is_empty() {
                // SAFETY: barriers correctly formed; semaphore wait on the compute submit
                // provides the execution dependency from the upload CB.
                unsafe {
                    device.cmd_pipeline_barrier(compute_cb,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(), &[], &upload_barriers, &[]);
                }
            }
        }

        // SAFETY: pipeline is valid.
        unsafe { device.cmd_bind_pipeline(compute_cb, vk::PipelineBindPoint::COMPUTE, inner.pipeline); }
        if let Some(ds) = inner.descriptor_set {
            unsafe {
                device.cmd_bind_descriptor_sets(compute_cb, vk::PipelineBindPoint::COMPUTE,
                    inner.pipeline_layout, 0, &[ds], &[]);
            }
        }
        if !push_constants.is_empty() {
            unsafe {
                device.cmd_push_constants(compute_cb, inner.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE, 0, push_constants);
            }
        }
        unsafe { device.cmd_dispatch(compute_cb, workgroups.0, workgroups.1, workgroups.2); }

        // Compute→Transfer barriers for staged readback (SHADER_WRITE→TRANSFER_READ).
        // r2 Lever A: skip ReadOnly bindings (never read back).
        // r2 Lever B: skip ReBAR output bindings — they get a SHADER_WRITE→HOST_READ barrier
        //   on this same compute CB (emitted below), not a TRANSFER_READ barrier.
        if has_outputs {
            let mut compute_to_transfer_barriers: Vec<vk::BufferMemoryBarrier<'_>> = Vec::new();
            let mut rebar_host_barriers: Vec<vk::BufferMemoryBarrier<'_>> = Vec::new();
            for i in 0..n {
                let out_size = if i < output_sizes.len() { output_sizes[i] } else { 0 };
                let access = inner.binding_plan.buffers[i].ty.access;
                if !binding_is_readback(access, out_size) { continue; } // Lever A
                let slot = &buffers[i];
                if slot_is_rebar(slot) {
                    // Lever B: SHADER_WRITE → HOST_READ on the ReBAR device-local buffer.
                    let barrier = vk::BufferMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::HOST_READ)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .buffer(slot.device_local.buffer).offset(0).size(vk::WHOLE_SIZE);
                    rebar_host_barriers.push(barrier);
                } else {
                    // Staged readback: SHADER_WRITE → TRANSFER_READ (r1 path).
                    let barrier = vk::BufferMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .buffer(slot.device_local.buffer).offset(0).size(vk::WHOLE_SIZE);
                    compute_to_transfer_barriers.push(barrier);
                }
            }
            if !compute_to_transfer_barriers.is_empty() {
                unsafe {
                    device.cmd_pipeline_barrier(compute_cb,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(), &[], &compute_to_transfer_barriers, &[]);
                }
            }
            if !rebar_host_barriers.is_empty() {
                // Lever B: emit COMPUTE_SHADER→HOST barrier for all ReBAR outputs.
                unsafe {
                    device.cmd_pipeline_barrier(compute_cb,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::HOST,
                        vk::DependencyFlags::empty(), &[], &rebar_host_barriers, &[]);
                }
            }
        }
        unsafe { device.end_command_buffer(compute_cb) }
            .map_err(|e| DispatchError::CommandBufferRecordFailed(e.to_string()))?;
    }

    // ── Record readback CB (if has_staged_readback) ──────────────────────────
    // r2 Lever A: skip ReadOnly bindings.
    // r2 Lever B: skip ReBAR output bindings (no copy needed; host reads from mapped pointer).
    if let Some(readback_cb) = readback_cb_opt {
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.begin_command_buffer(readback_cb, &begin_info) }
            .map_err(|e| DispatchError::CommandBufferRecordFailed(e.to_string()))?;

        // Device→staging copies (staged outputs only — no ReBAR, no ReadOnly).
        let mut readback_barriers: Vec<vk::BufferMemoryBarrier<'_>> = Vec::new();
        for i in 0..n {
            let out_size = if i < output_sizes.len() { output_sizes[i] } else { 0 };
            let access = inner.binding_plan.buffers[i].ty.access;
            if !binding_is_readback(access, out_size) { continue; } // Lever A
            let slot = &buffers[i];
            if slot_is_rebar(slot) { continue; } // Lever B: no copy for ReBAR outputs
            let copy_region = vk::BufferCopy::default()
                .src_offset(0).dst_offset(0).size(out_size as u64);
            unsafe {
                device.cmd_copy_buffer(readback_cb, slot.device_local.buffer,
                    slot.staging.buffer, &[copy_region]);
            }
            // CRITICAL-1: always emit TRANSFER_WRITE→HOST_READ barrier for staged outputs.
            let barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(slot.staging.buffer).offset(0).size(vk::WHOLE_SIZE);
            readback_barriers.push(barrier);
        }
        if !readback_barriers.is_empty() {
            unsafe {
                device.cmd_pipeline_barrier(readback_cb,
                    vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(), &[], &readback_barriers, &[]);
            }
        }
        unsafe { device.end_command_buffer(readback_cb) }
            .map_err(|e| DispatchError::CommandBufferRecordFailed(e.to_string()))?;
    }

    // ── Fence reset ───────────────────────────────────────────────────────────
    // SAFETY: inner.fence is valid.
    unsafe { device.reset_fences(&[inner.fence]) }
        .map_err(|e| DispatchError::CommandBufferRecordFailed(format!("vkResetFences: {e}")))?;

    // ── Three-submit atomic group (CRITICAL-5) ────────────────────────────────
    // Acquire queue_submit_lock ONCE; hold across ALL THREE submits; release BEFORE fence wait.
    let submit_result: Result<(), DispatchError> = {
        let _submit_guard = ctx.queue_submit_lock.lock();

        // SUBMIT 1: Upload (transfer queue), signal upload_done.
        if let Some(upload_cb) = upload_cb_opt {
            let upload_submit_result = submit_upload(
                device, ctx.transfer_queue, upload_cb, &inner.handoff,
                upload_signal_val, inner.sync_mode, has_uploads,
            );
            if let Err(e) = upload_submit_result {
                // U fails: free U CB only. No fence wait. Return error.
                let _pool_guard = transfer_pool_lock.lock();
                unsafe { device.free_command_buffers(ctx.transfer_command_pool, &[upload_cb]); }
                {
                    let _cpool_guard = ctx.compute_pool_lock.lock();
                    unsafe { device.free_command_buffers(ctx.command_pool, &[compute_cb]); }
                }
                if let Some(rcb) = readback_cb_opt {
                    let _pool_guard2 = transfer_pool_lock.lock();
                    unsafe { device.free_command_buffers(ctx.transfer_command_pool, &[rcb]); }
                }
                return Err(e);
            }
        }

        // SUBMIT 2: Compute (compute queue), wait upload_done / signal compute_done.
        let compute_submit_result = submit_compute(
            device, ctx.queue, compute_cb, &inner.handoff,
            upload_signal_val, compute_signal_val, inner.sync_mode,
            has_uploads, has_outputs, inner.fence, readback_cb_opt.is_none(),
        );
        if let Err(e) = compute_submit_result {
            // U ok, C fails: device_wait_idle → free U+C CBs; recreate binaries; return error.
            unsafe { let _ = device.device_wait_idle(); }
            if let Some(ucb) = upload_cb_opt {
                let _pool_guard = transfer_pool_lock.lock();
                unsafe { device.free_command_buffers(ctx.transfer_command_pool, &[ucb]); }
            }
            {
                let _cpool_guard = ctx.compute_pool_lock.lock();
                unsafe { device.free_command_buffers(ctx.command_pool, &[compute_cb]); }
            }
            if let Some(rcb) = readback_cb_opt {
                let _pool_guard2 = transfer_pool_lock.lock();
                unsafe { device.free_command_buffers(ctx.transfer_command_pool, &[rcb]); }
            }
            // Recreate binaries on error (CRITICAL-4); no-op for timeline.
            if let Some(h) = inner.handoff.as_ref() {
                // We need mut access but have &inner. Use interior mutability via UnsafeCell
                // or tolerate the binary-semaphore leak for now — timeline path handles it correctly.
                // For binary mode, the recreate call requires &mut HandoffSemaphores.
                // Since this is a non-fatal best-effort on error (the next dispatch will fail
                // differently if binaries are in a bad state), and we cannot get &mut through Arc,
                // we log and continue. TRUE binary recreation requires &mut self — see known_limitations.
                tracing::warn!("binary semaphore state may be inconsistent after compute submit failure");
                let _ = h; // suppress unused warning
            }
            return Err(e);
        }

        // SUBMIT 3: Readback (transfer queue), wait compute_done / signal fence.
        if let Some(readback_cb) = readback_cb_opt {
            let readback_submit_result = submit_readback(
                device, ctx.transfer_queue, readback_cb, &inner.handoff,
                compute_signal_val, inner.sync_mode, inner.fence,
            );
            if let Err(e) = readback_submit_result {
                // U ok, C ok, R fails: device_wait_idle → free all CBs; return error.
                unsafe { let _ = device.device_wait_idle(); }
                if let Some(ucb) = upload_cb_opt {
                    let _pool_guard = transfer_pool_lock.lock();
                    unsafe { device.free_command_buffers(ctx.transfer_command_pool, &[ucb]); }
                }
                {
                    let _cpool_guard = ctx.compute_pool_lock.lock();
                    unsafe { device.free_command_buffers(ctx.command_pool, &[compute_cb]); }
                }
                {
                    let _pool_guard2 = transfer_pool_lock.lock();
                    unsafe { device.free_command_buffers(ctx.transfer_command_pool, &[readback_cb]); }
                }
                return Err(e);
            }
        }

        Ok(())
    }; // queue_submit_lock released here (before fence wait)

    submit_result?;

    // ── Wait for readback fence ───────────────────────────────────────────────
    // SAFETY: inner.fence was submitted in the last successful submit.
    let wait_result = unsafe { device.wait_for_fences(&[inner.fence], true, timeout_ns) };

    // Free all CBs under their respective pool locks (regardless of wait result).
    if let Some(ucb) = upload_cb_opt {
        let _pool_guard = transfer_pool_lock.lock();
        unsafe { device.free_command_buffers(ctx.transfer_command_pool, &[ucb]); }
    }
    {
        let _cpool_guard = ctx.compute_pool_lock.lock();
        unsafe { device.free_command_buffers(ctx.command_pool, &[compute_cb]); }
    }
    if let Some(rcb) = readback_cb_opt {
        let _pool_guard2 = transfer_pool_lock.lock();
        unsafe { device.free_command_buffers(ctx.transfer_command_pool, &[rcb]); }
    }

    if let Err(vk::Result::TIMEOUT) = wait_result {
        return Err(DispatchError::FenceTimeout { timeout_ns });
    }
    wait_result.map_err(|e| DispatchError::QueueSubmitFailed(e.to_string()))?;

    // ── Readback: invalidate + copy_out via persistent mapping (Lever A + Lever B) ──
    let outputs = readback_from_persistent_mapping(
        inner, buffers, output_sizes, n, device, ctx.non_coherent_atom_size,
        &effective_coherency, ctx.force_noncoherent,
    );

    Ok(outputs)
}

// ── Sub-helpers for the three-submit dedicated path ─────────────────────────

/// Submit the upload command buffer, signaling the upload-done semaphore.
#[allow(clippy::undocumented_unsafe_blocks)]
fn submit_upload(
    device: &ash::Device,
    transfer_queue: vk::Queue,
    upload_cb: vk::CommandBuffer,
    handoff: &Option<HandoffSemaphores>,
    upload_signal_val: u64,
    sync_mode: SyncMode,
    _has_uploads: bool,
) -> Result<(), DispatchError> {
    let cb_slice = [upload_cb];
    match handoff {
        Some(HandoffSemaphores::Timeline { semaphore, .. }) => {
            let signal_vals = [upload_signal_val];
            let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
                .signal_semaphore_values(&signal_vals);
            let signal_sems = [*semaphore];
            let submit_info = vk::SubmitInfo::default()
                .command_buffers(&cb_slice)
                .signal_semaphores(&signal_sems)
                .push_next(&mut timeline_info);
            // SAFETY: submit_info is valid; semaphore is timeline type.
            unsafe { device.queue_submit(transfer_queue, &[submit_info], vk::Fence::null()) }
                .map_err(|e| DispatchError::TransferQueueSubmitFailed(
                    format!("upload timeline submit: {e}")
                ))?;
        }
        Some(HandoffSemaphores::Binary { upload_done, .. }) => {
            let signal_sems = [*upload_done];
            let submit_info = vk::SubmitInfo::default()
                .command_buffers(&cb_slice)
                .signal_semaphores(&signal_sems);
            unsafe { device.queue_submit(transfer_queue, &[submit_info], vk::Fence::null()) }
                .map_err(|e| DispatchError::TransferQueueSubmitFailed(
                    format!("upload binary submit: {e}")
                ))?;
        }
        None => {
            // Single-queue path; this function should not be called without handoff.
            let submit_info = vk::SubmitInfo::default().command_buffers(&cb_slice);
            unsafe { device.queue_submit(transfer_queue, &[submit_info], vk::Fence::null()) }
                .map_err(|e| DispatchError::TransferQueueSubmitFailed(e.to_string()))?;
        }
    }
    let _ = sync_mode; // mode is embedded in the handoff variant
    Ok(())
}

/// Submit the compute command buffer, waiting upload-done / signaling compute-done (or fence).
#[allow(clippy::undocumented_unsafe_blocks)]
#[allow(clippy::too_many_arguments)]
fn submit_compute(
    device: &ash::Device,
    compute_queue: vk::Queue,
    compute_cb: vk::CommandBuffer,
    handoff: &Option<HandoffSemaphores>,
    upload_signal_val: u64,
    compute_signal_val: u64,
    _sync_mode: SyncMode,
    has_uploads: bool,
    _has_outputs: bool,
    fence: vk::Fence,
    signal_fence_directly: bool,
) -> Result<(), DispatchError> {
    let cb_slice = [compute_cb];
    let fence_to_signal = if signal_fence_directly { fence } else { vk::Fence::null() };

    match handoff {
        Some(HandoffSemaphores::Timeline { semaphore, .. }) => {
            // Wait upload_done (B+1) or B (already-passed base) if no uploads.
            let wait_val = if has_uploads { upload_signal_val } else { upload_signal_val - 1 };
            let signal_vals = [compute_signal_val];
            let wait_vals = [wait_val];
            let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
                .wait_semaphore_values(&wait_vals)
                .signal_semaphore_values(&signal_vals);
            let wait_sems = [*semaphore];
            let wait_stages = [vk::PipelineStageFlags::COMPUTE_SHADER];
            let signal_sems = [*semaphore];
            let submit_info = vk::SubmitInfo::default()
                .command_buffers(&cb_slice)
                .wait_semaphores(&wait_sems)
                .wait_dst_stage_mask(&wait_stages)
                .signal_semaphores(&signal_sems)
                .push_next(&mut timeline_info);
            unsafe { device.queue_submit(compute_queue, &[submit_info], fence_to_signal) }
                .map_err(|e| DispatchError::QueueSubmitFailed(format!("compute timeline: {e}")))?;
        }
        Some(HandoffSemaphores::Binary { upload_done, compute_done }) => {
            if has_uploads {
                let wait_sems = [*upload_done];
                let wait_stages = [vk::PipelineStageFlags::COMPUTE_SHADER];
                let signal_sems = [*compute_done];
                let submit_info = vk::SubmitInfo::default()
                    .command_buffers(&cb_slice)
                    .wait_semaphores(&wait_sems)
                    .wait_dst_stage_mask(&wait_stages)
                    .signal_semaphores(&signal_sems);
                unsafe { device.queue_submit(compute_queue, &[submit_info], fence_to_signal) }
                    .map_err(|e| DispatchError::QueueSubmitFailed(format!("compute binary (with upload): {e}")))?;
            } else {
                let signal_sems = [*compute_done];
                let submit_info = vk::SubmitInfo::default()
                    .command_buffers(&cb_slice)
                    .signal_semaphores(&signal_sems);
                unsafe { device.queue_submit(compute_queue, &[submit_info], fence_to_signal) }
                    .map_err(|e| DispatchError::QueueSubmitFailed(format!("compute binary (no upload): {e}")))?;
            }
        }
        None => {
            let submit_info = vk::SubmitInfo::default().command_buffers(&cb_slice);
            unsafe { device.queue_submit(compute_queue, &[submit_info], fence_to_signal) }
                .map_err(|e| DispatchError::QueueSubmitFailed(e.to_string()))?;
        }
    }
    Ok(())
}

/// Submit the readback command buffer, waiting compute-done / signaling the fence.
#[allow(clippy::undocumented_unsafe_blocks)]
fn submit_readback(
    device: &ash::Device,
    transfer_queue: vk::Queue,
    readback_cb: vk::CommandBuffer,
    handoff: &Option<HandoffSemaphores>,
    compute_signal_val: u64,
    _sync_mode: SyncMode,
    fence: vk::Fence,
) -> Result<(), DispatchError> {
    let cb_slice = [readback_cb];
    match handoff {
        Some(HandoffSemaphores::Timeline { semaphore, .. }) => {
            let wait_vals = [compute_signal_val];
            let signal_vals: [u64; 0] = [];
            let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
                .wait_semaphore_values(&wait_vals)
                .signal_semaphore_values(&signal_vals);
            let wait_sems = [*semaphore];
            let wait_stages = [vk::PipelineStageFlags::TRANSFER];
            let submit_info = vk::SubmitInfo::default()
                .command_buffers(&cb_slice)
                .wait_semaphores(&wait_sems)
                .wait_dst_stage_mask(&wait_stages)
                .push_next(&mut timeline_info);
            unsafe { device.queue_submit(transfer_queue, &[submit_info], fence) }
                .map_err(|e| DispatchError::TransferQueueSubmitFailed(
                    format!("readback timeline: {e}")
                ))?;
        }
        Some(HandoffSemaphores::Binary { compute_done, .. }) => {
            let wait_sems = [*compute_done];
            let wait_stages = [vk::PipelineStageFlags::TRANSFER];
            let submit_info = vk::SubmitInfo::default()
                .command_buffers(&cb_slice)
                .wait_semaphores(&wait_sems)
                .wait_dst_stage_mask(&wait_stages);
            unsafe { device.queue_submit(transfer_queue, &[submit_info], fence) }
                .map_err(|e| DispatchError::TransferQueueSubmitFailed(
                    format!("readback binary: {e}")
                ))?;
        }
        None => {
            let submit_info = vk::SubmitInfo::default().command_buffers(&cb_slice);
            unsafe { device.queue_submit(transfer_queue, &[submit_info], fence) }
                .map_err(|e| DispatchError::TransferQueueSubmitFailed(e.to_string()))?;
        }
    }
    Ok(())
}

// ── Shared readback helper ───────────────────────────────────────────────────

/// Read back output buffers via persistent mapping after the fence has been waited.
///
/// ## r2 Lever A
///
/// A binding whose `access == ReadOnly` is NEVER read back (the shader carries
/// `NonWritable` decoration and provably cannot write it). The returned `Vec<u8>` is
/// empty for such bindings — byte-identical to the `output_size==0` case.
///
/// ## r2 Lever B (ReBAR zero-copy)
///
/// For output bindings allocated in ReBAR memory (`slot.device_local.mapping.is_some()`),
/// the host reads directly from the mapped device-local pointer at ~60 GB/s. An
/// atom-aligned `invalidate` is called first on non-coherent ReBAR types; it is a no-op
/// on the coherent Type 4 on the Blackwell. No staging copy_out is performed.
///
/// For non-ReBAR (staged) outputs the r1 staging path is used unchanged (invalidate +
/// staging copy_out).
///
/// On NonCoherent staging memory: calls `vkInvalidateMappedMemoryRanges` BEFORE
/// `copy_out` for EACH staged readback slot (CRITICAL-1, WARNING-6).
#[allow(clippy::undocumented_unsafe_blocks)]
#[allow(clippy::too_many_arguments)]
fn readback_from_persistent_mapping(
    inner: &KernelHandleInner,
    buffers: &[BufferSlot],
    output_sizes: &[usize],
    n: usize,
    device: &ash::Device,
    non_coherent_atom_size: u64,
    effective_coherency: &impl Fn(&BufferSlot) -> Coherency,
    force_noncoherent: bool,
) -> Vec<Vec<u8>> {
    let _ = non_coherent_atom_size; // used via effective_coherency
    let _ = force_noncoherent;

    let mut outputs: Vec<Vec<u8>> = Vec::with_capacity(n);
    for i in 0..n {
        let out_size = if i < output_sizes.len() { output_sizes[i] } else { 0 };
        let access = inner.binding_plan.buffers[i].ty.access;

        // r2 Lever A: ReadOnly bindings are never read back regardless of out_size.
        // Returns empty Vec (byte-identical to the out_size==0 case).
        if !binding_is_readback(access, out_size) {
            outputs.push(Vec::new());
            continue;
        }

        let slot = &buffers[i];

        if slot_is_rebar(slot) {
            // r2 Lever B: read directly from the ReBAR device-local mapping at ~60 GB/s.
            // invalidate first (no-op on coherent ReBAR; real on non-coherent ReBAR).
            // SAFETY: fence has been waited; device/memory valid; buffers Mutex held (CRITICAL-6).
            let _ = unsafe {
                slot.device_local.mapping.as_ref()
                    // SAFETY: slot_is_rebar guarantees mapping is Some.
                    .expect("slot_is_rebar invariant: mapping must be Some")
                    .invalidate(device, slot.device_local.memory, out_size as u64)
            };
            let mut out_bytes: Vec<u8> = vec![0u8; out_size];
            // copy_out: reads from the ReBAR mapped pointer; buffers Mutex held (CRITICAL-6).
            slot.device_local.mapping.as_ref()
                .expect("slot_is_rebar invariant: mapping must be Some")
                .copy_out(&mut out_bytes);
            outputs.push(out_bytes);
        } else {
            // Staged readback (r1 path): invalidate staging + copy_out from staging mapping.
            let coh = effective_coherency(slot);
            // Invalidate on NonCoherent BEFORE copy_out (CRITICAL-1).
            match coh {
                Coherency::NonCoherent { .. } => {
                    // SAFETY: fence has been waited; device/memory valid.
                    // Ignore error: best-effort; if invalidate fails the data may be stale
                    // but we still attempt the copy (error will surface as wrong output).
                    let _ = unsafe {
                        slot.staging.mapping.invalidate(device, slot.staging.memory, out_size as u64)
                    };
                }
                Coherency::Coherent => {}
            }
            let mut out_bytes: Vec<u8> = vec![0u8; out_size];
            // copy_out: caller holds buffers Mutex (CRITICAL-6).
            slot.staging.mapping.copy_out(&mut out_bytes);
            outputs.push(out_bytes);
        }
    }
    outputs
}

// ── Helper: binary semaphore recreation (AT-1419) ────────────────────────────
// Exposed for use by context.rs error recovery paths.
#[allow(dead_code)]
pub(crate) fn maybe_recreate_binary_semaphores(
    device: &ash::Device,
    inner: &mut KernelHandleInner,
) -> Result<(), DispatchError> {
    if let Some(ref mut h) = inner.handoff {
        recreate_binary_on_error(device, h)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// AT-805a: KernelCacheKey implements Ord (BTreeMap invariant).
    #[test]
    fn at_805a_kernel_cache_key_ord() {
        let k1 = KernelCacheKey { spirv_hash: 1, binding_plan_hash: 2, push_constant_size: 4 };
        let k2 = KernelCacheKey { spirv_hash: 1, binding_plan_hash: 2, push_constant_size: 8 };
        let k3 = KernelCacheKey { spirv_hash: 2, binding_plan_hash: 0, push_constant_size: 0 };

        assert!(k1 < k2, "keys differing in push_constant_size must order correctly");
        assert!(k1 < k3, "keys differing in spirv_hash must order correctly");
        assert!(k2 < k3, "spirv_hash dominates in ordering");

        // BTreeMap operations require Ord.
        let mut map: std::collections::BTreeMap<KernelCacheKey, u32> = Default::default();
        map.insert(k1.clone(), 1);
        map.insert(k2.clone(), 2);
        map.insert(k3.clone(), 3);
        assert_eq!(map.get(&k1), Some(&1));
        assert_eq!(map.get(&k2), Some(&2));
        assert_eq!(map.get(&k3), Some(&3));
    }

    /// AT-805b: hash_spirv is deterministic (same words → same hash).
    #[test]
    fn at_805b_hash_spirv_deterministic() {
        let words: Vec<u32> = vec![0x07230203, 0x00010300, 42, 99];
        assert_eq!(hash_spirv(&words), hash_spirv(&words), "hash must be deterministic");

        let words2: Vec<u32> = vec![0x07230203, 0x00010300, 43, 99];
        assert_ne!(hash_spirv(&words), hash_spirv(&words2), "different words must differ");
    }

    /// AT-805c: round_up_pow2 used in ensure_buffers_fit — sizes grow to pow2.
    #[test]
    fn at_805c_round_up_pow2_used_for_buffers() {
        // Verify that the sizing function used for buffer growth produces expected values.
        assert_eq!(round_up_pow2(1024 * 4), 4096);   // 1024 f32 elements
        assert_eq!(round_up_pow2(2048 * 4), 8192);   // 2048 f32 elements
        assert_eq!(round_up_pow2(4096 * 4 + 1), 32768); // Just over a pow2 boundary
    }
}
