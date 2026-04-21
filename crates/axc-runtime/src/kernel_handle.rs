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
use crate::error::{DispatchError, CopyDirection};
use crate::buffers::{
    allocate_device_local_buffer, allocate_staging_buffer,
    DeviceLocalBuffer, StagingBuffer, round_up_pow2,
};
use crate::dispatch::DEFAULT_FENCE_TIMEOUT_MS;

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

        // Destroy buffer slots first (device-local buf→mem, then staging buf→mem).
        {
            let mut guard = self.buffers.lock();
            for slot in guard.drain(..) {
                // SAFETY: buffers are only destroyed after all fence waits complete.
                unsafe {
                    device.destroy_buffer(slot.staging.buffer, None);
                    device.free_memory(slot.staging.memory, None);
                    device.destroy_buffer(slot.device_local.buffer, None);
                    device.free_memory(slot.device_local.memory, None);
                }
            }
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
/// - If the slot does not exist or is too small: destroys the old slot (if any),
///   allocates a new pair with `round_up_pow2(size_needed)`, and rewrites the
///   descriptor set entry for this binding.
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
        if i < buffers.len() {
            // We remove the slot at i by truncating + re-inserting — use swap_remove
            // since order within the Vec doesn't need to be preserved (each index
            // maps 1:1 to a binding).
            let old_slot = buffers.remove(i);
            // SAFETY: the old buffers are not in use (they were from a prior dispatch
            // that completed its fence wait before this dispatch_handle call began).
            unsafe {
                device.destroy_buffer(old_slot.staging.buffer, None);
                device.free_memory(old_slot.staging.memory, None);
                device.destroy_buffer(old_slot.device_local.buffer, None);
                device.free_memory(old_slot.device_local.memory, None);
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

        let device_local: DeviceLocalBuffer = allocate_device_local_buffer(
            device,
            mem_props,
            new_size,
            i as u32,
        )?;

        let staging: StagingBuffer = allocate_staging_buffer(
            device,
            mem_props,
            new_size,
            i as u32,
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
    /// Command pool from the `VulkanContext` (RESET_COMMAND_BUFFER).
    pub(crate) command_pool: vk::CommandPool,
    /// Compute queue handle from the `VulkanContext`.
    pub(crate) queue: vk::Queue,
}

/// Record and submit a compute dispatch, then wait for completion.
///
/// This is the core work of `dispatch_handle`. The buffer mutex must be held by
/// the caller (via `buffers_guard`) for the duration of this call.
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

    let device: &ash::Device = &inner.device.device;
    let n: usize = inner.binding_plan.buffers.len();

    // ── Upload: copy inputs from host to staging buffers ─────────────────────
    for i in 0..n {
        if i >= inputs.len() || inputs[i].is_empty() {
            continue;
        }
        let slot: &BufferSlot = &buffers[i];
        // SAFETY: staging memory is HOST_VISIBLE|HOST_COHERENT; mapping offset 0
        // with size = full allocation is valid. No flush needed (HOST_COHERENT).
        let ptr: *mut std::ffi::c_void = unsafe {
            device.map_memory(
                slot.staging.memory,
                0,
                slot.staging.size,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(|e| DispatchError::StagingCopyFailed {
            binding: i as u32,
            direction: CopyDirection::HostToDevice,
            reason: format!("vkMapMemory: {e}"),
        })?;

        // SAFETY: ptr is valid mapped host memory of size >= inputs[i].len().
        unsafe {
            std::ptr::copy_nonoverlapping(
                inputs[i].as_ptr(),
                ptr as *mut u8,
                inputs[i].len(),
            );
        }

        // SAFETY: memory was successfully mapped above.
        unsafe { device.unmap_memory(slot.staging.memory); }
    }

    // ── Allocate command buffer ───────────────────────────────────────────────
    let cb_alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(ctx.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    // SAFETY: command_pool is valid and was created with RESET_COMMAND_BUFFER.
    let cbs: Vec<vk::CommandBuffer> =
        unsafe { device.allocate_command_buffers(&cb_alloc_info) }
            .map_err(|e| DispatchError::CommandBufferRecordFailed(e.to_string()))?;
    let cb: vk::CommandBuffer = cbs[0];

    let begin_info = vk::CommandBufferBeginInfo::default()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    // SAFETY: cb is freshly allocated and not yet recording.
    unsafe { device.begin_command_buffer(cb, &begin_info) }
        .map_err(|e| DispatchError::CommandBufferRecordFailed(e.to_string()))?;

    // ── Stage A: copy staging→device for bindings with non-empty input ────────
    let mut upload_barriers: Vec<vk::BufferMemoryBarrier<'_>> = Vec::new();
    for i in 0..n {
        if i >= inputs.len() || inputs[i].is_empty() {
            continue;
        }
        let slot: &BufferSlot = &buffers[i];
        let copy_region = vk::BufferCopy::default()
            .src_offset(0)
            .dst_offset(0)
            .size(inputs[i].len() as u64);

        // SAFETY: both buffers are valid and non-overlapping.
        unsafe {
            device.cmd_copy_buffer(
                cb,
                slot.staging.buffer,
                slot.device_local.buffer,
                &[copy_region],
            );
        }

        // Build per-buffer barrier: TRANSFER_WRITE → SHADER_READ|SHADER_WRITE.
        let barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            )
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(slot.device_local.buffer)
            .offset(0)
            .size(vk::WHOLE_SIZE);
        upload_barriers.push(barrier);
    }

    if !upload_barriers.is_empty() {
        // SAFETY: barriers are correctly formed; command buffer is recording.
        unsafe {
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &upload_barriers,
                &[],
            );
        }
    }

    // ── Bind pipeline + descriptors + push constants ──────────────────────────
    // SAFETY: pipeline is valid.
    unsafe { device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, inner.pipeline); }

    // P-5: skip descriptor binding for 0-buffer kernels.
    if let Some(ds) = inner.descriptor_set {
        // SAFETY: ds is valid; pipeline_layout matches the pipeline.
        unsafe {
            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                inner.pipeline_layout,
                0,
                &[ds],
                &[],
            );
        }
    }

    if !push_constants.is_empty() {
        // SAFETY: pipeline_layout was created with a push-constant range covering
        // offset 0..push_constant_total_bytes, stage COMPUTE.
        unsafe {
            device.cmd_push_constants(
                cb,
                inner.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_constants,
            );
        }
    }

    // SAFETY: workgroup counts were validated in validate_dispatch_handle_args.
    unsafe { device.cmd_dispatch(cb, workgroups.0, workgroups.1, workgroups.2); }

    // ── Compute→Transfer barrier (SHADER_WRITE → TRANSFER_READ) ──────────────
    let mut compute_barriers: Vec<vk::BufferMemoryBarrier<'_>> = Vec::new();
    for i in 0..n {
        if i >= output_sizes.len() || output_sizes[i] == 0 {
            continue;
        }
        let slot: &BufferSlot = &buffers[i];
        let barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(slot.device_local.buffer)
            .offset(0)
            .size(vk::WHOLE_SIZE);
        compute_barriers.push(barrier);
    }

    if !compute_barriers.is_empty() {
        // SAFETY: barriers are correctly formed; command buffer is recording.
        unsafe {
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &compute_barriers,
                &[],
            );
        }
    }

    // ── Stage B: copy device→staging for bindings with non-zero output_size ───
    let mut readback_barriers: Vec<vk::BufferMemoryBarrier<'_>> = Vec::new();
    for i in 0..n {
        if i >= output_sizes.len() || output_sizes[i] == 0 {
            continue;
        }
        let slot: &BufferSlot = &buffers[i];
        let copy_region = vk::BufferCopy::default()
            .src_offset(0)
            .dst_offset(0)
            .size(output_sizes[i] as u64);

        // SAFETY: both buffers are valid and non-overlapping.
        unsafe {
            device.cmd_copy_buffer(
                cb,
                slot.device_local.buffer,
                slot.staging.buffer,
                &[copy_region],
            );
        }

        // Build TRANSFER_WRITE → HOST_READ barrier for staging.
        let barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(slot.staging.buffer)
            .offset(0)
            .size(vk::WHOLE_SIZE);
        readback_barriers.push(barrier);
    }

    if !readback_barriers.is_empty() {
        // SAFETY: barriers are correctly formed.
        unsafe {
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                &readback_barriers,
                &[],
            );
        }
    }

    // SAFETY: command buffer has a valid begin; no unclosed render passes.
    unsafe { device.end_command_buffer(cb) }
        .map_err(|e| DispatchError::CommandBufferRecordFailed(e.to_string()))?;

    // ── Fence reset + queue submit ────────────────────────────────────────────
    // P-2: reset the reusable fence before every submit.
    // SAFETY: fence is valid and was waited on by the previous dispatch (or just
    // created in prepare_kernel, in which case it is unsignaled and reset is a no-op).
    unsafe { device.reset_fences(&[inner.fence]) }
        .map_err(|e| DispatchError::CommandBufferRecordFailed(
            format!("vkResetFences: {e}")
        ))?;

    let cb_slice: [vk::CommandBuffer; 1] = [cb];
    let submit_info = vk::SubmitInfo::default().command_buffers(&cb_slice);

    // SAFETY: submit_info references cb which is a valid, recorded command buffer.
    // inner.fence is a valid, unsignaled fence (just reset above).
    unsafe { device.queue_submit(ctx.queue, &[submit_info], inner.fence) }
        .map_err(|e| DispatchError::QueueSubmitFailed(e.to_string()))?;

    // ── Wait for fence ────────────────────────────────────────────────────────
    let timeout_ms: u64 = std::env::var("AXC_FENCE_TIMEOUT_MS")
        .ok()
        .and_then(|v: String| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_FENCE_TIMEOUT_MS);
    let timeout_ns: u64 = timeout_ms * 1_000_000;

    // SAFETY: inner.fence is valid and was submitted above.
    let wait_result = unsafe { device.wait_for_fences(&[inner.fence], true, timeout_ns) };
    if let Err(vk::Result::TIMEOUT) = wait_result {
        // Free the command buffer before returning the error.
        // SAFETY: cb is no longer in flight (timeout means GPU may be stuck, but
        // we must free the command buffer to avoid a resource leak).
        unsafe { device.free_command_buffers(ctx.command_pool, &[cb]); }
        return Err(DispatchError::FenceTimeout { timeout_ns });
    }
    wait_result.map_err(|e| DispatchError::QueueSubmitFailed(e.to_string()))?;

    // Free the per-dispatch command buffer.
    // SAFETY: cb has completed (fence waited); freeing is safe.
    unsafe { device.free_command_buffers(ctx.command_pool, &[cb]); }

    // ── Readback: map staging buffers and copy to Vec<u8> ────────────────────
    let mut outputs: Vec<Vec<u8>> = Vec::with_capacity(n);

    for i in 0..n {
        let out_size: usize = if i < output_sizes.len() { output_sizes[i] } else { 0 };
        if out_size == 0 {
            outputs.push(Vec::new());
            continue;
        }

        let slot: &BufferSlot = &buffers[i];
        // SAFETY: staging memory is HOST_VISIBLE|HOST_COHERENT; fence has been
        // waited on, so GPU writes (via the readback copy) are visible to the host.
        let ptr: *mut std::ffi::c_void = unsafe {
            device.map_memory(
                slot.staging.memory,
                0,
                slot.staging.size,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(|e| DispatchError::StagingCopyFailed {
            binding: i as u32,
            direction: CopyDirection::DeviceToHost,
            reason: format!("vkMapMemory: {e}"),
        })?;

        let mut out_bytes: Vec<u8> = vec![0u8; out_size];

        // SAFETY: ptr is valid mapped memory; out_size <= slot.staging.size by construction.
        unsafe {
            std::ptr::copy_nonoverlapping(
                ptr as *const u8,
                out_bytes.as_mut_ptr(),
                out_size,
            );
        }

        // SAFETY: memory was successfully mapped above.
        unsafe { device.unmap_memory(slot.staging.memory); }

        outputs.push(out_bytes);
    }

    Ok(outputs)
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
