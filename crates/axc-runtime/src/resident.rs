// Partial-failure cleanup closures in upload_resident use map_err(|e| { cleanup; e })
// which clippy suggests rewriting as inspect_err. The cleanup code calls cleanup_partial_resident
// which requires moving Vec state — the manual pattern is clearer for the CORRECTNESS-2 audit.
// needless_range_loop: the upload loops use indexed access for synchronous Vulkan state
// across multiple Vecs (transient_staging[i], device_local_bufs[i], inputs[i]) — enumerate
// would require multiple zip chains that obscure the CORRECTNESS-2 cleanup logic.
#![allow(clippy::manual_inspect)]
#![allow(clippy::drain_collect)]
#![allow(clippy::needless_range_loop)]

//! GPU-resident benchmark primitive — M3.1.5.
//!
//! The thesis-relevant metric for llama.cpp inference is: upload weights ONCE to
//! resident VRAM, dispatch N times, measure the GPU dispatch duration. This module
//! provides that primitive.
//!
//! ## API overview
//!
//! ```rust,ignore
//! let resident = ctx.upload_resident(&handle, &inputs, &output_sizes)?;
//! let timing = ctx.dispatch_resident(&handle, &resident, &cfg)?;
//! let outputs = ctx.readback_resident(&handle, &resident, &output_sizes)?;
//! ```
//!
//! `upload_resident` copies inputs into device-local buffers once and binds an
//! INDEPENDENT descriptor set to those buffers (NOT handle.inner.descriptor_set —
//! one KernelHandle can back dispatch_handle + multiple ResidentBuffers). Each
//! `dispatch_resident` records ONLY the compute submit, giving GPU dispatch duration
//! timing that EXCLUDES host upload/readback.
//!
//! ## Timing methodology (TIMING-1, honest)
//!
//! The measured quantity is the **GPU dispatch duration**: the TOP_OF_PIPE →
//! BOTTOM_OF_PIPE timestamp span over the single dispatch. This span:
//! - **EXCLUDES** host upload (one-time, before the loop) and readback (after the loop).
//! - **INCLUDES** dispatch launch + pipeline latency / drain.
//!
//! It is NOT isolated ALU time. Do NOT call it "kernel-only ALU". The exclusion of
//! upload/readback is the entire value over `dispatch_handle`.
//!
//! **Warmup-discard is MANDATORY:** always discard the first 2 dispatch_resident
//! measurements (pipeline warm-up, first-touch device-local page-in, clock ramp).
//! Report MIN-of-N (N=10 bench, N≥3 for AT-1530).
//!
//! **CpuFenceWall fallback:** when `timestampValidBits==0` OR `timestampPeriod==0`
//! (e.g. Lavapipe), timing falls back to CPU fence-wall. This number INCLUDES submit
//! scheduling + driver overhead and is NEVER a GPU kernel time. Always print
//! `timing_source` and append "— scheduling-inclusive, NOT a GPU kernel time" for
//! CpuFenceWall measurements. On this path NO query pool is created and NO
//! vkCmdWriteTimestamp is recorded (recording on a 0-valid-bits queue is UB).
//!
//! ## Timing when `timestampValidBits > 0` and `timestampPeriod > 0`:
//!
//! - `vkCmdWriteTimestamp(TOP_OF_PIPE, pool, 0)` before dispatch
//! - `vkCmdWriteTimestamp(BOTTOM_OF_PIPE, pool, 1)` after dispatch
//! - `elapsed_ns = masked_timestamp_delta(begin, end, valid_bits) * timestamp_period`
//! - `timing_source = GpuTimestamp`
//!
//! The `cmd_reset_query_pool` is recorded BEFORE the first `write_timestamp` in the
//! SAME submission (per HN-6). The N-iteration loop REUSES one command buffer, fence,
//! and query pool (no per-iter allocation — AT-1571).
//!
//! ## Safety
//!
//! All `unsafe` blocks have `// SAFETY:` comments.

use ash::vk;
use std::sync::Arc;
use crate::device_owner::DeviceOwner;
use crate::buffers::{DeviceLocalBuffer, StagingBuffer, allocate_device_local_buffer,
                     allocate_staging_buffer_cached, round_up_pow2};
use crate::error::{DispatchError, DispatchResult};
use crate::kernel_handle::{KernelHandle, allocate_descriptor_pool_and_set, binding_is_readback};
use crate::context::VulkanContext;
use crate::dispatch::DEFAULT_FENCE_TIMEOUT_MS;

/// Configuration for a single `dispatch_resident` call.
pub struct ResidentBenchConfig {
    /// Workgroup dispatch dimensions (x, y, z).
    pub workgroups: (u32, u32, u32),
    /// Push constant bytes to push before dispatch.
    pub push_constants: Vec<u8>,
}

/// Source of the kernel timing measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentTimingSource {
    /// GPU timestamp query (vkCmdWriteTimestamp TOP_OF_PIPE→BOTTOM_OF_PIPE).
    ///
    /// Measures the GPU dispatch duration: the single-dispatch span from before
    /// bind/push/dispatch to after all operations have drained the pipeline.
    /// Excludes host upload (one-time, pre-loop) and readback (post-loop).
    /// Includes dispatch launch latency and pipeline drain.
    GpuTimestamp,
    /// CPU fence-wall time (std::time::Instant around submit+wait). Scheduling-inclusive.
    ///
    /// Used when `timestampValidBits==0` OR `timestampPeriod==0` (e.g. Lavapipe).
    /// This number INCLUDES submit overhead, fence scheduling, and driver overhead.
    /// It is NOT a GPU kernel time and MUST NOT be presented as one.
    CpuFenceWall,
}

/// Timing result from `dispatch_resident`.
#[derive(Debug, Clone)]
pub struct ResidentDispatchTiming {
    /// GPU dispatch duration (or CPU fence-wall time) in nanoseconds.
    ///
    /// Interpretation depends on `timing_source`:
    /// - `GpuTimestamp`: TOP_OF_PIPE→BOTTOM_OF_PIPE span, excludes upload/readback.
    /// - `CpuFenceWall`: CPU wall time around submit+wait, scheduling-inclusive.
    pub kernel_ns: u64,
    /// Source of the timing measurement.
    pub timing_source: ResidentTimingSource,
}

/// GPU-resident buffer set for repeated dispatch.
///
/// Created by `VulkanContext::upload_resident`. Owns device-local buffers,
/// a persistent descriptor set bound to those buffers (INDEPENDENT of the
/// `KernelHandle`'s own descriptor set — AT-1570 guards a regression here),
/// one reusable command buffer, one fence, and optionally a 2-entry timestamp
/// query pool (only when `timestampValidBits > 0` AND `timestampPeriod > 0`).
///
/// Destroyed when dropped — resources freed in Drop (reverse creation order per
/// Vulkan spec §3.3.3). Drop fence-waits to ensure no in-flight work.
#[allow(dead_code)] // Fields used in Drop + dispatch_resident implementation
pub struct ResidentBuffers {
    /// Arc to keep the device alive.
    pub(crate) device: Arc<DeviceOwner>,
    /// Device-local input/output buffers (indexed by binding slot).
    pub(crate) device_local_bufs: Vec<DeviceLocalBuffer>,
    /// Retained staging buffers for readback (output slots only; upload uses a separate
    /// TRANSIENT staging buffer — OPT-NOTE-2: no aliasing between upload and readback staging).
    pub(crate) staging_bufs: Vec<Option<StagingBuffer>>,
    /// Descriptor set bound to the RESIDENT device-local buffers.
    ///
    /// NOTE: This is an INDEPENDENT set, NOT handle.inner.descriptor_set. One KernelHandle
    /// can back dispatch_handle + multiple ResidentBuffers simultaneously. The pipeline and
    /// layout come from the handle; the buffers and descriptor set are owned here (HANDOFF-9).
    pub(crate) descriptor_set: Option<vk::DescriptorSet>,
    /// Descriptor pool owning the descriptor set.
    pub(crate) descriptor_pool: Option<vk::DescriptorPool>,
    /// Reusable command buffer for dispatch recording.
    pub(crate) command_buffer: vk::CommandBuffer,
    /// Command pool owning the command buffer.
    pub(crate) command_pool: vk::CommandPool,
    /// Reusable fence for GPU sync. Reset before each submit (P-2).
    pub(crate) fence: vk::Fence,
    /// 2-entry timestamp query pool for kernel timing.
    ///
    /// `vk::QueryPool::null()` on the CpuFenceWall path (timestampValidBits==0 or period==0).
    /// NEVER call vkCmdWriteTimestamp when this is null — UB per Vulkan spec.
    pub(crate) query_pool: vk::QueryPool,
    /// Timestamp period in nanoseconds per tick (0.0 if unavailable).
    pub(crate) timestamp_period: f32,
    /// Number of valid timestamp bits (0 if unavailable).
    pub(crate) timestamp_valid_bits: u32,
    /// Queue family index for command pool and queue submission.
    pub(crate) queue_family_index: u32,
    /// Compute queue handle (cached at upload time for dispatch use).
    pub(crate) queue: vk::Queue,
    /// Queue-submit lock (Arc clone from VulkanContext — CRITICAL-5 serialization).
    pub(crate) queue_submit_lock: Arc<parking_lot::Mutex<()>>,
    /// Diagnostic-only: whether these buffers back a coopmat kernel.
    ///
    /// Set false in M3.1.5 — the handle carries no coopmat flag in this milestone.
    /// Retained for future milestones that store coopmat shape on KernelHandleInner.
    #[allow(dead_code)]
    pub(crate) uses_coopmat: bool,
}

/// Compute the masked timestamp delta, handling wrap across the valid-bit boundary.
///
/// Both `begin` and `end` are masked to `valid_bits` FIRST, then `end.wrapping_sub(begin)`
/// is computed. This handles the case where the counter wraps between begin and end.
///
/// Per HN-6: mask EACH endpoint first, then wrapping_sub (AT-1535).
#[allow(dead_code)] // Used in dispatch_resident timing (AT-1535); tests confirm correctness
pub(crate) fn masked_timestamp_delta(begin: u64, end: u64, valid_bits: u32) -> u64 {
    if valid_bits == 0 || valid_bits >= 64 {
        // If all 64 bits are valid (or the field is zero), no masking needed.
        return end.wrapping_sub(begin);
    }
    let mask: u64 = (1u64 << valid_bits).wrapping_sub(1);
    let masked_begin = begin & mask;
    let masked_end = end & mask;
    masked_end.wrapping_sub(masked_begin) & mask
}

/// Pure helper: select timing source from device timestamp capabilities.
///
/// Returns `GpuTimestamp` when both `valid_bits > 0` AND `period > 0`.
/// Otherwise returns `CpuFenceWall` (scheduling-inclusive; NOT a GPU kernel time).
///
/// Decided ONCE at upload time from snapshotted values; a `ResidentBuffers` never
/// flips paths mid-life (no stale-result hazard across the N-iteration loop).
/// AT-1573 exercises this function in isolation (pure unit test, no GPU).
pub(crate) fn select_timing_source(valid_bits: u32, period: f32) -> ResidentTimingSource {
    if valid_bits > 0 && period > 0.0 {
        ResidentTimingSource::GpuTimestamp
    } else {
        ResidentTimingSource::CpuFenceWall
    }
}

impl ResidentBuffers {
    /// Return the reusable command buffer handle (for AT-1571 invariance check).
    pub fn command_buffer_handle(&self) -> vk::CommandBuffer {
        self.command_buffer
    }

    /// Return the reusable fence handle (for AT-1571 invariance check).
    pub fn fence_handle(&self) -> vk::Fence {
        self.fence
    }

    /// Return the timestamp query pool handle (for AT-1571 invariance check).
    /// Returns `vk::QueryPool::null()` on the CpuFenceWall path.
    pub fn query_pool_handle(&self) -> vk::QueryPool {
        self.query_pool
    }

    /// Return the device-local buffer VkBuffer handles (for AT-1571 invariance check).
    pub fn device_local_buf_handles(&self) -> Vec<vk::Buffer> {
        self.device_local_bufs.iter().map(|b| b.buffer).collect()
    }
}

impl Drop for ResidentBuffers {
    fn drop(&mut self) {
        let device: &ash::Device = &self.device.device;
        // SAFETY: all GPU work has completed (fence-wait invariant on callers);
        // resources are owned exclusively by this struct.
        // Reverse creation order per Vulkan spec §3.3.3.
        unsafe {
            // Wait for any in-flight work before destroying anything.
            let _ = device.wait_for_fences(&[self.fence], true, u64::MAX);

            // Destroy query pool (null-safe: vkDestroyQueryPool with null handle is no-op
            // per Vulkan spec §3.3.3 — any dispatchable command may receive null).
            if self.query_pool != vk::QueryPool::null() {
                device.destroy_query_pool(self.query_pool, None);
            }

            // Destroy fence.
            device.destroy_fence(self.fence, None);

            // Destroy descriptor pool (implicitly frees descriptor set).
            if let Some(pool) = self.descriptor_pool {
                device.destroy_descriptor_pool(pool, None);
            }

            // Destroy command pool (implicitly frees command buffer).
            device.destroy_command_pool(self.command_pool, None);

            // Free staging buffers (readback-only; unmap not needed — StagingBuffer
            // allocated via allocate_staging_buffer_cached which handles its own mapping).
            for stg in self.staging_bufs.iter().flatten() {
                device.destroy_buffer(stg.buffer, None);
                device.free_memory(stg.memory, None);
            }

            // Free device-local buffers.
            for buf in &self.device_local_bufs {
                device.destroy_buffer(buf.buffer, None);
                device.free_memory(buf.memory, None);
            }
        }
    }
}

/// Helper: free any partially-created persistent Vulkan objects in reverse creation order.
///
/// Called at every early-return Err in upload_resident AFTER the first allocation
/// (CORRECTNESS-2 / AT-1580). Each parameter is an Option so it is safe to pass
/// whatever has been created so far (None entries are no-ops).
///
/// Reverse creation order (Vulkan spec §3.3.3):
/// query_pool → fence → descriptor_pool → command_pool → staging_bufs → device_local_bufs
fn cleanup_partial_resident(
    device: &ash::Device,
    query_pool: vk::QueryPool,
    fence: Option<vk::Fence>,
    descriptor_pool: Option<vk::DescriptorPool>,
    command_pool: Option<vk::CommandPool>,
    staging_bufs: &[Option<StagingBuffer>],
    device_local_bufs: &[DeviceLocalBuffer],
) {
    // SAFETY: every handle passed here was successfully created and is not in use
    // (we are in an error path before any GPU submission).
    unsafe {
        // 1. query_pool (created last among persistent objects).
        if query_pool != vk::QueryPool::null() {
            device.destroy_query_pool(query_pool, None);
        }
        // 2. fence.
        if let Some(f) = fence {
            device.destroy_fence(f, None);
        }
        // 3. descriptor_pool (implicitly frees descriptor set).
        if let Some(dp) = descriptor_pool {
            device.destroy_descriptor_pool(dp, None);
        }
        // 4. command_pool (implicitly frees command buffer).
        if let Some(cp) = command_pool {
            device.destroy_command_pool(cp, None);
        }
        // 5. staging_bufs (readback staging; freed before device_local).
        for stg in staging_bufs.iter().flatten() {
            device.destroy_buffer(stg.buffer, None);
            device.free_memory(stg.memory, None);
        }
        // 6. device_local_bufs (created first, freed last).
        for buf in device_local_bufs {
            device.destroy_buffer(buf.buffer, None);
            device.free_memory(buf.memory, None);
        }
    }
}

impl VulkanContext {
    /// Upload inputs to device-local VRAM ONCE and build a reusable dispatch context.
    ///
    /// Steps:
    /// 1. Validate `inputs.len()` and `output_sizes.len()` against the binding plan.
    /// 2. For each binding: allocate a device-local buffer; for output bindings also
    ///    allocate a RETAINED staging buffer for `readback_resident`. Input copy uses
    ///    a SEPARATE TRANSIENT staging buffer (OPT-NOTE-2: no aliasing).
    /// 3. Copy all inputs to device-local via a one-shot upload CB (transient staging
    ///    → device-local); fence-wait; free transient upload resources.
    /// 4. Allocate an INDEPENDENT descriptor pool+set for the resident buffers and
    ///    write all device-local buffers into it (NOT handle.inner.descriptor_set).
    /// 5. Create a reusable command pool+CB, fence (unsignaled), and — ONLY when
    ///    `timestampValidBits>0 && timestampPeriod>0` — a 2-entry timestamp query pool.
    ///
    /// On any early-return `Err` after the first allocation, `cleanup_partial_resident`
    /// frees every already-created persistent Vulkan object (CORRECTNESS-2 / AT-1580).
    ///
    /// # Call-order contract
    ///
    /// `upload_resident` → `dispatch_resident` (N times) → `readback_resident`.
    pub fn upload_resident(
        &self,
        handle: &KernelHandle,
        inputs: &[&[u8]],
        output_sizes: &[u64],
    ) -> DispatchResult<ResidentBuffers> {
        let device: &ash::Device = &self.device_owner.device;
        let n: usize = handle.inner.binding_plan.buffers.len();

        // ── Step 1: Validate counts (fail fast before any allocation) ────────────
        if inputs.len() != n {
            return Err(DispatchError::BindingCountMismatch { expected: n, provided: inputs.len() });
        }
        if output_sizes.len() != n {
            return Err(DispatchError::BindingCountMismatch { expected: n, provided: output_sizes.len() });
        }

        // ── Step 2: Allocate device-local + readback-staging buffers ─────────────
        let mut device_local_bufs: Vec<DeviceLocalBuffer> = Vec::with_capacity(n);
        let mut staging_bufs: Vec<Option<StagingBuffer>> = Vec::with_capacity(n);

        for i in 0..n {
            let input_len: u64 = inputs[i].len() as u64;
            let output_len: u64 = output_sizes[i];
            let size_needed: u64 = input_len.max(output_len).max(4);
            let alloc_size: u64 = round_up_pow2(size_needed);
            if alloc_size == u64::MAX {
                // size overflow — cleanup what we have so far.
                let mut staging_refs: Vec<Option<StagingBuffer>> = staging_bufs.drain(..).collect();
                cleanup_partial_resident(
                    device, vk::QueryPool::null(), None, None, None,
                    &staging_refs, &device_local_bufs,
                );
                // Zero out to avoid double-free in any future code path.
                staging_refs.clear();
                device_local_bufs.clear();
                return Err(DispatchError::BufferAllocationFailed {
                    binding: i as u32,
                    size: size_needed as usize,
                    reason: "size overflow (> 2^62 bytes)".to_owned(),
                });
            }

            // Allocate device-local buffer (STORAGE|TRANSFER_SRC|TRANSFER_DST, DEVICE_LOCAL).
            let dl_buf = allocate_device_local_buffer(
                device,
                &self.memory_properties,
                alloc_size,
                i as u32,
                None, // SingleQueue mode: EXCLUSIVE sharing (resident path uses the compute queue)
            ).map_err(|e| {
                let staging_refs: Vec<Option<StagingBuffer>> = staging_bufs.drain(..).collect();
                cleanup_partial_resident(
                    device, vk::QueryPool::null(), None, None, None,
                    &staging_refs, &device_local_bufs,
                );
                e
            })?;
            device_local_bufs.push(dl_buf);

            // Allocate retained staging buffer for readback (output bindings only).
            // OPT-NOTE-2: this is the READBACK staging; upload uses a SEPARATE TRANSIENT
            // staging buffer. No aliasing between upload and readback staging.
            let access = handle.inner.binding_plan.buffers[i].ty.access;
            let is_readback = binding_is_readback(access, output_sizes[i] as usize);
            let stg = if is_readback {
                let stg_buf = allocate_staging_buffer_cached(
                    device,
                    &self.memory_properties,
                    self.non_coherent_atom_size(),
                    alloc_size,
                    i as u32,
                    false, // force_noncoherent: use context default
                ).map_err(|e| {
                    // staging_bufs so far + new dl_buf already pushed.
                    let staging_refs: Vec<Option<StagingBuffer>> = staging_bufs.drain(..).collect();
                    cleanup_partial_resident(
                        device, vk::QueryPool::null(), None, None, None,
                        &staging_refs, &device_local_bufs,
                    );
                    e
                })?;
                Some(stg_buf)
            } else {
                None
            };
            staging_bufs.push(stg);
        }

        // ── Step 3: One-time input upload via TRANSIENT staging + CB ─────────────
        // Allocate one transient staging buffer per non-empty input (NOT the retained
        // readback staging — they are separate: OPT-NOTE-2).
        // We use a single transient command pool for the upload.
        let upload_command_pool: vk::CommandPool = {
            let cp_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(self.queue_family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            // SAFETY: cp_info is valid.
            unsafe { device.create_command_pool(&cp_info, None) }
                .map_err(|e| {
                    cleanup_partial_resident(
                        device, vk::QueryPool::null(), None, None, None,
                        &staging_bufs, &device_local_bufs,
                    );
                    DispatchError::CommandBufferRecordFailed(format!("create upload command_pool: {e}"))
                })?
        };

        // Allocate transient upload staging buffers (one per non-empty input).
        let mut transient_staging: Vec<Option<StagingBuffer>> = Vec::with_capacity(n);
        for i in 0..n {
            if inputs[i].is_empty() {
                transient_staging.push(None);
                continue;
            }
            let size_needed: u64 = (inputs[i].len() as u64).max(4);
            let alloc_size: u64 = round_up_pow2(size_needed);
            let stg = allocate_staging_buffer_cached(
                device,
                &self.memory_properties,
                self.non_coherent_atom_size(),
                alloc_size,
                i as u32,
                false,
            ).map_err(|e| {
                // Free transient staging allocated so far.
                for s in transient_staging.iter().flatten() {
                    // SAFETY: these were just allocated; not in use.
                    unsafe {
                        device.destroy_buffer(s.buffer, None);
                        device.free_memory(s.memory, None);
                    }
                }
                // SAFETY: upload_command_pool was just created; not in use.
                unsafe { device.destroy_command_pool(upload_command_pool, None); }
                cleanup_partial_resident(
                    device, vk::QueryPool::null(), None, None, None,
                    &staging_bufs, &device_local_bufs,
                );
                e
            })?;
            // Copy input data into transient staging.
            stg.mapping.copy_in(inputs[i]);
            transient_staging.push(Some(stg));
        }

        // Record and submit the upload CB (staging→device-local copies + barriers).
        if n > 0 && (0..n).any(|i| !inputs[i].is_empty()) {
            let upload_result = self.record_and_submit_upload(
                device,
                upload_command_pool,
                &transient_staging,
                &device_local_bufs,
                inputs,
            );
            if let Err(e) = upload_result {
                for s in transient_staging.iter().flatten() {
                    // SAFETY: not submitted (error path).
                    unsafe {
                        device.destroy_buffer(s.buffer, None);
                        device.free_memory(s.memory, None);
                    }
                }
                // SAFETY: upload_command_pool was just created.
                unsafe { device.destroy_command_pool(upload_command_pool, None); }
                cleanup_partial_resident(
                    device, vk::QueryPool::null(), None, None, None,
                    &staging_bufs, &device_local_bufs,
                );
                return Err(e);
            }
        }

        // Free transient upload staging + command pool (not needed after copy).
        // SAFETY: upload CB has completed (fence-waited in record_and_submit_upload);
        // transient staging is no longer in use.
        unsafe {
            device.destroy_command_pool(upload_command_pool, None);
        }
        for s in transient_staging.into_iter().flatten() {
            // SAFETY: transient staging is not in use after upload.
            unsafe {
                device.destroy_buffer(s.buffer, None);
                device.free_memory(s.memory, None);
            }
        }

        // ── Step 4: Allocate independent descriptor pool+set ─────────────────────
        // NOTE (HANDOFF-9): this is NOT handle.inner.descriptor_set. The resident set
        // is bound to the RESIDENT device-local buffers; it is independent of the handle's
        // dispatch_handle set. AT-1570 guards any regression to using the handle's set.
        let (descriptor_pool_opt, descriptor_set_opt): (Option<vk::DescriptorPool>, Option<vk::DescriptorSet>) =
            if n == 0 {
                (None, None)
            } else {
                let dsl = match handle.inner.descriptor_set_layout {
                    Some(dsl) => dsl,
                    None => {
                        cleanup_partial_resident(
                            device, vk::QueryPool::null(), None, None, None,
                            &staging_bufs, &device_local_bufs,
                        );
                        return Err(DispatchError::DescriptorSetLayoutFailed(
                            "non-empty binding plan has no descriptor set layout".to_owned()
                        ));
                    }
                };
                let (pool, set) = allocate_descriptor_pool_and_set(device, dsl, n)
                    .map_err(|e| {
                        cleanup_partial_resident(
                            device, vk::QueryPool::null(), None, None, None,
                            &staging_bufs, &device_local_bufs,
                        );
                        e
                    })?;

                // Write each device-local buffer into the resident descriptor set.
                // Mirrors ensure_buffers_fit_with_mem_props (kernel_handle.rs ~439-453)
                // at binding index buffer_position (HANDOFF-1).
                for i in 0..n {
                    let binding_idx: u32 = handle.inner.binding_plan.buffers[i].buffer_position;
                    let buffer_info = vk::DescriptorBufferInfo::default()
                        .buffer(device_local_bufs[i].buffer)
                        .offset(0)
                        .range(vk::WHOLE_SIZE);
                    let write = vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(binding_idx)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&buffer_info));
                    // SAFETY: write references valid descriptor set and buffer handles.
                    unsafe { device.update_descriptor_sets(&[write], &[]); }
                }

                (Some(pool), Some(set))
            };

        // ── Step 5: Create reusable command pool+CB, fence, and (maybe) query pool ─
        let resident_command_pool: vk::CommandPool = {
            let cp_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(self.queue_family_index)
                // RESET_COMMAND_BUFFER: allows reset_command_buffer (HANDOFF-4).
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            // SAFETY: cp_info is valid.
            unsafe { device.create_command_pool(&cp_info, None) }
                .map_err(|e| {
                    cleanup_partial_resident(
                        device, vk::QueryPool::null(), None,
                        descriptor_pool_opt, None,
                        &staging_bufs, &device_local_bufs,
                    );
                    DispatchError::CommandBufferRecordFailed(format!("create resident command_pool: {e}"))
                })?
        };

        let resident_cb: vk::CommandBuffer = {
            let cb_alloc = vk::CommandBufferAllocateInfo::default()
                .command_pool(resident_command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            // SAFETY: resident_command_pool is valid.
            let cbs = unsafe { device.allocate_command_buffers(&cb_alloc) }
                .map_err(|e| {
                    cleanup_partial_resident(
                        device, vk::QueryPool::null(), None,
                        descriptor_pool_opt, Some(resident_command_pool),
                        &staging_bufs, &device_local_bufs,
                    );
                    DispatchError::CommandBufferRecordFailed(format!("allocate resident CB: {e}"))
                })?;
            cbs[0]
        };

        let fence_info = vk::FenceCreateInfo::default();
        // SAFETY: fence_info is valid; the fence will be destroyed in Drop or cleanup.
        let resident_fence: vk::Fence = unsafe { device.create_fence(&fence_info, None) }
            .map_err(|e| {
                cleanup_partial_resident(
                    device, vk::QueryPool::null(), None,
                    descriptor_pool_opt, Some(resident_command_pool),
                    &staging_bufs, &device_local_bufs,
                );
                DispatchError::CommandBufferRecordFailed(format!("create resident fence: {e}"))
            })?;

        // Create 2-entry timestamp query pool ONLY on the GpuTimestamp path.
        // (HN-6 / HANDOFF-3: never create or use timestamp queries on 0-valid-bits queues.)
        let valid_bits: u32 = self.compute_timestamp_valid_bits();
        let period: f32 = self.timestamp_period();
        let query_pool: vk::QueryPool =
            if select_timing_source(valid_bits, period) == ResidentTimingSource::GpuTimestamp {
                let qp_info = vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2);
                // SAFETY: qp_info is valid.
                unsafe { device.create_query_pool(&qp_info, None) }
                    .map_err(|e| {
                        cleanup_partial_resident(
                            device, vk::QueryPool::null(), Some(resident_fence),
                            descriptor_pool_opt, Some(resident_command_pool),
                            &staging_bufs, &device_local_bufs,
                        );
                        DispatchError::CommandBufferRecordFailed(format!("create query_pool: {e}"))
                    })?
            } else {
                // CpuFenceWall path: no query pool created.
                vk::QueryPool::null()
            };

        Ok(ResidentBuffers {
            device: Arc::clone(&self.device_owner),
            device_local_bufs,
            staging_bufs,
            descriptor_set: descriptor_set_opt,
            descriptor_pool: descriptor_pool_opt,
            command_buffer: resident_cb,
            command_pool: resident_command_pool,
            fence: resident_fence,
            query_pool,
            timestamp_period: period,
            timestamp_valid_bits: valid_bits,
            queue_family_index: self.queue_family_index,
            queue: self.queue,
            queue_submit_lock: self.queue_submit_lock(),
            uses_coopmat: false, // diagnostic-only; handle carries no coopmat flag in M3.1.5
        })
    }

    /// Record and submit a one-shot upload command buffer (transient staging → device-local).
    ///
    /// Submits under `queue_submit_lock` (CRITICAL-5) and fence-waits on `self.fence`
    /// (the context's MAIN fence — distinct from the resident fence). The transient
    /// staging buffers and upload command pool are freed by the caller after this returns.
    fn record_and_submit_upload(
        &self,
        device: &ash::Device,
        upload_command_pool: vk::CommandPool,
        transient_staging: &[Option<StagingBuffer>],
        device_local_bufs: &[DeviceLocalBuffer],
        inputs: &[&[u8]],
    ) -> DispatchResult<()> {
        let n = inputs.len();
        // Use a transient fence for the upload (not the resident fence).
        let upload_fence_info = vk::FenceCreateInfo::default();
        // SAFETY: valid fence create info.
        let upload_fence: vk::Fence = unsafe { device.create_fence(&upload_fence_info, None) }
            .map_err(|e| DispatchError::CommandBufferRecordFailed(format!("upload fence: {e}")))?;

        let cb_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(upload_command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: upload_command_pool is valid.
        let upload_cb = match unsafe { device.allocate_command_buffers(&cb_alloc) } {
            Ok(cbs) => cbs[0],
            Err(e) => {
                // SAFETY: upload_fence just created; not in use.
                unsafe { device.destroy_fence(upload_fence, None); }
                return Err(DispatchError::CommandBufferRecordFailed(format!("alloc upload CB: {e}")));
            }
        };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: upload_cb is freshly allocated.
        if let Err(e) = unsafe { device.begin_command_buffer(upload_cb, &begin_info) } {
            // SAFETY: CB and fence not in use.
            unsafe { device.destroy_fence(upload_fence, None); }
            return Err(DispatchError::CommandBufferRecordFailed(format!("begin upload CB: {e}")));
        }

        // Record staging→device-local copy commands + upload barriers.
        let mut upload_barriers: Vec<vk::BufferMemoryBarrier<'_>> = Vec::new();
        for i in 0..n {
            if inputs[i].is_empty() {
                continue;
            }
            let Some(stg) = &transient_staging[i] else { continue; };
            let copy_region = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(inputs[i].len() as u64);
            // SAFETY: both buffers are valid and non-overlapping.
            unsafe {
                device.cmd_copy_buffer(upload_cb, stg.buffer, device_local_bufs[i].buffer, &[copy_region]);
            }
            let barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(device_local_bufs[i].buffer)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            upload_barriers.push(barrier);
        }
        if !upload_barriers.is_empty() {
            // SAFETY: barriers correctly formed.
            unsafe {
                device.cmd_pipeline_barrier(
                    upload_cb,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[], &upload_barriers, &[],
                );
            }
        }

        // SAFETY: no unclosed render passes.
        if let Err(e) = unsafe { device.end_command_buffer(upload_cb) } {
            // SAFETY: CB not submitted; fence not in use.
            unsafe { device.destroy_fence(upload_fence, None); }
            return Err(DispatchError::CommandBufferRecordFailed(format!("end upload CB: {e}")));
        }

        // Submit under queue_submit_lock (CRITICAL-5: serialize all queue submissions).
        {
            let submit_lock = self.queue_submit_lock();
            let _submit_guard = submit_lock.lock();
            let cb_slice = [upload_cb];
            let submit_info = vk::SubmitInfo::default().command_buffers(&cb_slice);
            // SAFETY: submit_info is valid; fence is unsignaled.
            if let Err(e) = unsafe { device.queue_submit(self.queue, &[submit_info], upload_fence) } {
                // Lock released by dropping _submit_guard.
                // SAFETY: fence not signaled (submit failed).
                unsafe { device.destroy_fence(upload_fence, None); }
                return Err(DispatchError::QueueSubmitFailed(format!("upload submit: {e}")));
            }
        } // queue_submit_lock released before fence wait (CRITICAL-5)

        // Wait for upload to complete.
        let timeout_ms: u64 = std::env::var("AXC_FENCE_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(DEFAULT_FENCE_TIMEOUT_MS);
        let timeout_ns: u64 = timeout_ms * 1_000_000;
        // SAFETY: upload_fence was submitted above.
        let wait_result = unsafe { device.wait_for_fences(&[upload_fence], true, timeout_ns) };
        // SAFETY: upload fence done; destroy it.
        unsafe { device.destroy_fence(upload_fence, None); }
        if let Err(vk::Result::TIMEOUT) = wait_result {
            return Err(DispatchError::FenceTimeout { timeout_ns });
        }
        wait_result.map_err(|e| DispatchError::QueueSubmitFailed(format!("upload fence wait: {e}")))?;

        Ok(())
    }

    /// Execute one timed dispatch using the resident buffers.
    ///
    /// FIRST validates `cfg.workgroups` against `max_compute_work_group_count`
    /// (CORRECTNESS-1 / AT-1579) — returns `WorkgroupCountExceedsDeviceLimit` before
    /// any GPU work on out-of-range workgroups.
    ///
    /// Reuses the resident CB/fence/query-pool across calls (no per-iter allocation —
    /// AT-1571). Records into the reusable CB:
    /// - GpuTimestamp path: `cmd_reset_query_pool` (HN-6 — same submission, first) →
    ///   `write_timestamp(TOP_OF_PIPE, 0)` → bind pipeline/RESIDENT descriptor/push →
    ///   `dispatch` → `write_timestamp(BOTTOM_OF_PIPE, 1)`.
    /// - CpuFenceWall path: no timestamp commands; CPU Instant::now() around submit+wait.
    ///
    /// ## Concurrency
    ///
    /// `dispatch_resident` on the SAME `ResidentBuffers` from two threads concurrently
    /// is undefined behavior (shared CB/fence/query pool). Documented caller contract;
    /// `queue_submit_lock` still serializes the submit globally.
    pub fn dispatch_resident(
        &self,
        handle: &KernelHandle,
        resident: &ResidentBuffers,
        cfg: &ResidentBenchConfig,
    ) -> DispatchResult<ResidentDispatchTiming> {
        // CORRECTNESS-1 (HANDOFF-7): validate workgroups FIRST, before any GPU work.
        let max: [u32; 3] = self.max_compute_work_group_count;
        let requested: [u32; 3] = [cfg.workgroups.0, cfg.workgroups.1, cfg.workgroups.2];
        for axis in 0..3_usize {
            if requested[axis] > max[axis] {
                return Err(DispatchError::WorkgroupCountExceedsDeviceLimit { requested, max });
            }
        }

        let device: &ash::Device = &self.device_owner.device;
        let cb: vk::CommandBuffer = resident.command_buffer;
        let fence: vk::Fence = resident.fence;

        // Decide timing source ONCE from snapshotted values (HANDOFF-3).
        let use_gpu_ts: bool = resident.query_pool != vk::QueryPool::null()
            && select_timing_source(resident.timestamp_valid_bits, resident.timestamp_period)
                == ResidentTimingSource::GpuTimestamp;

        let timeout_ms: u64 = std::env::var("AXC_FENCE_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(DEFAULT_FENCE_TIMEOUT_MS);
        let timeout_ns: u64 = timeout_ms * 1_000_000;

        // Reset the reusable CB before re-recording (HANDOFF-4).
        // SAFETY: CB is not in-flight (prior dispatch fence-waited, Drop fence-waits).
        unsafe { device.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty()) }
            .map_err(|e| DispatchError::CommandBufferRecordFailed(format!("reset resident CB: {e}")))?;

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: cb is valid and not in-flight.
        unsafe { device.begin_command_buffer(cb, &begin_info) }
            .map_err(|e| DispatchError::CommandBufferRecordFailed(format!("begin resident CB: {e}")))?;

        // GpuTimestamp path: reset query pool FIRST (HN-6 — same submission, before any write),
        // then write TOP_OF_PIPE timestamp.
        if use_gpu_ts {
            // SAFETY: query_pool is valid (non-null) on this path.
            unsafe {
                device.cmd_reset_query_pool(cb, resident.query_pool, 0, 2);
                device.cmd_write_timestamp(cb, vk::PipelineStageFlags::TOP_OF_PIPE,
                    resident.query_pool, 0);
            }
        }

        // Bind pipeline + RESIDENT descriptor set + push constants + dispatch.
        // NOTE (HANDOFF-9): bind resident.descriptor_set, NOT handle.inner.descriptor_set.
        // SAFETY: pipeline is valid.
        unsafe { device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, handle.inner.pipeline); }
        if let Some(ds) = resident.descriptor_set {
            // SAFETY: ds is the resident descriptor set, valid for this handle's layout.
            unsafe {
                device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE,
                    handle.inner.pipeline_layout, 0, &[ds], &[]);
            }
        }
        if !cfg.push_constants.is_empty() {
            // SAFETY: valid push constant range.
            unsafe {
                device.cmd_push_constants(cb, handle.inner.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE, 0, &cfg.push_constants);
            }
        }
        // SAFETY: workgroup counts validated above.
        unsafe { device.cmd_dispatch(cb, cfg.workgroups.0, cfg.workgroups.1, cfg.workgroups.2); }

        // GpuTimestamp path: write BOTTOM_OF_PIPE timestamp after dispatch.
        if use_gpu_ts {
            // SAFETY: query_pool is valid (non-null).
            unsafe {
                device.cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    resident.query_pool, 1);
            }
        }

        // SAFETY: no unclosed render passes.
        unsafe { device.end_command_buffer(cb) }
            .map_err(|e| DispatchError::CommandBufferRecordFailed(format!("end resident CB: {e}")))?;

        // Reset the fence before submit (P-2). SAFETY: fence is not in use.
        unsafe { device.reset_fences(&[fence]) }
            .map_err(|e| DispatchError::CommandBufferRecordFailed(format!("reset resident fence: {e}")))?;

        // Capture CPU t0 BEFORE submit (for CpuFenceWall path).
        let t0 = std::time::Instant::now();

        // Submit under queue_submit_lock (CRITICAL-5 / HANDOFF-5).
        {
            let _submit_guard = resident.queue_submit_lock.lock();
            let cb_slice = [cb];
            let submit_info = vk::SubmitInfo::default().command_buffers(&cb_slice);
            // SAFETY: submit_info valid; fence is unsignaled.
            unsafe { device.queue_submit(resident.queue, &[submit_info], fence) }
                .map_err(|e| DispatchError::QueueSubmitFailed(format!("resident dispatch submit: {e}")))?;
        } // queue_submit_lock released before fence wait (CRITICAL-5 / HANDOFF-5)

        // Wait for dispatch to complete.
        // SAFETY: fence was submitted above.
        let wait_result = unsafe { device.wait_for_fences(&[fence], true, timeout_ns) };
        if let Err(vk::Result::TIMEOUT) = wait_result {
            return Err(DispatchError::FenceTimeout { timeout_ns });
        }
        wait_result.map_err(|e| DispatchError::QueueSubmitFailed(format!("resident fence wait: {e}")))?;

        // Compute kernel_ns based on timing source.
        let (kernel_ns, timing_source) = if use_gpu_ts {
            // GpuTimestamp: read 2 64-bit timestamp values (HANDOFF-6).
            let mut results: [u64; 2] = [0; 2];
            // SAFETY: query_pool is valid; results slice is valid for 2 × u64.
            unsafe {
                device.get_query_pool_results(
                    resident.query_pool,
                    0,
                    &mut results,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
            }.map_err(|e| DispatchError::QueueSubmitFailed(format!("get_query_pool_results: {e}")))?;

            let ticks = masked_timestamp_delta(results[0], results[1], resident.timestamp_valid_bits);
            let ns = (ticks as f64 * resident.timestamp_period as f64) as u64;
            (ns, ResidentTimingSource::GpuTimestamp)
        } else {
            // CpuFenceWall: CPU wall time around submit+wait (scheduling-inclusive).
            // MUST NOT be presented as a GPU kernel time (AT-1582).
            let wall_ns = t0.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            (wall_ns, ResidentTimingSource::CpuFenceWall)
        };

        Ok(ResidentDispatchTiming { kernel_ns, timing_source })
    }

    /// Copy device-local output buffers to host (once, after the dispatch loop).
    ///
    /// For each binding `i`:
    /// - If NOT a readback binding (Lever A / ReadOnly access or zero output size):
    ///   push an empty `Vec<u8>` (NEVER copy — staging_bufs[i] is None for these).
    /// - For readback bindings: copy device-local → retained staging → host.
    ///
    /// ## Call-order contract (OPT-NOTE-3)
    ///
    /// Must be called AFTER `upload_resident` → `dispatch_resident` (N times).
    /// The reusable CB is reset before re-recording; Drop fence-waits; no in-flight hazard.
    pub fn readback_resident(
        &self,
        handle: &KernelHandle,
        resident: &ResidentBuffers,
        output_sizes: &[u64],
    ) -> DispatchResult<Vec<Vec<u8>>> {
        let device: &ash::Device = &self.device_owner.device;
        let n: usize = handle.inner.binding_plan.buffers.len();

        // Build the result vector: empty for non-readback, populated for readback slots.
        let mut results: Vec<Vec<u8>> = Vec::with_capacity(n);

        // Identify which slots need readback.
        let needs_readback: Vec<bool> = (0..n).map(|i| {
            let access = handle.inner.binding_plan.buffers[i].ty.access;
            let out_size = if i < output_sizes.len() { output_sizes[i] as usize } else { 0 };
            binding_is_readback(access, out_size)
        }).collect();

        if !needs_readback.iter().any(|&b| b) {
            // No readback needed — return all-empty (Lever A).
            for _ in 0..n {
                results.push(Vec::new());
            }
            return Ok(results);
        }

        // Reset the reusable CB and record a readback copy (device-local → retained staging).
        let cb: vk::CommandBuffer = resident.command_buffer;
        // SAFETY: CB is not in-flight (prior dispatch fence-waited; Drop fence-waits).
        unsafe { device.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty()) }
            .map_err(|e| DispatchError::CommandBufferRecordFailed(format!("reset CB for readback: {e}")))?;

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: cb is valid and not in-flight.
        unsafe { device.begin_command_buffer(cb, &begin_info) }
            .map_err(|e| DispatchError::CommandBufferRecordFailed(format!("begin readback CB: {e}")))?;

        // Record SHADER_WRITE→TRANSFER_READ barriers + device→staging copies.
        let mut compute_barriers: Vec<vk::BufferMemoryBarrier<'_>> = Vec::new();
        for i in 0..n {
            if !needs_readback[i] { continue; }
            let barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(resident.device_local_bufs[i].buffer)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            compute_barriers.push(barrier);
        }
        if !compute_barriers.is_empty() {
            // SAFETY: barriers correctly formed.
            unsafe {
                device.cmd_pipeline_barrier(cb,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(), &[], &compute_barriers, &[]);
            }
        }

        let mut readback_barriers: Vec<vk::BufferMemoryBarrier<'_>> = Vec::new();
        for i in 0..n {
            if !needs_readback[i] { continue; }
            let out_size = if i < output_sizes.len() { output_sizes[i] } else { 0 };
            let Some(stg) = &resident.staging_bufs[i] else { continue; };
            let copy_region = vk::BufferCopy::default()
                .src_offset(0).dst_offset(0).size(out_size);
            // SAFETY: valid non-overlapping buffers.
            unsafe {
                device.cmd_copy_buffer(cb, resident.device_local_bufs[i].buffer, stg.buffer, &[copy_region]);
            }
            let barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(stg.buffer)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            readback_barriers.push(barrier);
        }
        if !readback_barriers.is_empty() {
            // SAFETY: barriers correctly formed.
            unsafe {
                device.cmd_pipeline_barrier(cb,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(), &[], &readback_barriers, &[]);
            }
        }

        // SAFETY: no unclosed render passes.
        unsafe { device.end_command_buffer(cb) }
            .map_err(|e| DispatchError::CommandBufferRecordFailed(format!("end readback CB: {e}")))?;

        // Reset the fence before submit.
        let fence: vk::Fence = resident.fence;
        // SAFETY: fence is not in use.
        unsafe { device.reset_fences(&[fence]) }
            .map_err(|e| DispatchError::CommandBufferRecordFailed(format!("reset fence for readback: {e}")))?;

        // Submit under queue_submit_lock (CRITICAL-5 / HANDOFF-5).
        {
            let _submit_guard = resident.queue_submit_lock.lock();
            let cb_slice = [cb];
            let submit_info = vk::SubmitInfo::default().command_buffers(&cb_slice);
            // SAFETY: submit_info valid; fence is unsignaled.
            unsafe { device.queue_submit(resident.queue, &[submit_info], fence) }
                .map_err(|e| DispatchError::QueueSubmitFailed(format!("readback submit: {e}")))?;
        }

        let timeout_ms: u64 = std::env::var("AXC_FENCE_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(DEFAULT_FENCE_TIMEOUT_MS);
        let timeout_ns: u64 = timeout_ms * 1_000_000;
        // SAFETY: fence was submitted above.
        let wait_result = unsafe { device.wait_for_fences(&[fence], true, timeout_ns) };
        if let Err(vk::Result::TIMEOUT) = wait_result {
            return Err(DispatchError::FenceTimeout { timeout_ns });
        }
        wait_result.map_err(|e| DispatchError::QueueSubmitFailed(format!("readback fence wait: {e}")))?;

        // Copy from retained staging to host Vec (with invalidate for NonCoherent).
        for i in 0..n {
            if !needs_readback[i] {
                results.push(Vec::new());
                continue;
            }
            let out_size = if i < output_sizes.len() { output_sizes[i] as usize } else { 0 };
            let Some(stg) = &resident.staging_bufs[i] else {
                results.push(Vec::new());
                continue;
            };
            // Invalidate NonCoherent staging before reading (CRITICAL-1 equivalent).
            // SAFETY: device and memory valid; fence has been waited.
            let _ = unsafe { stg.mapping.invalidate(device, stg.memory, out_size as u64) };
            let mut out_bytes = vec![0u8; out_size];
            stg.mapping.copy_out(&mut out_bytes);
            results.push(out_bytes);
        }

        Ok(results)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// AT-1535: masked_timestamp_delta handles wrap across the valid-bit boundary.
    ///
    /// Example: valid_bits=32, begin near 2^32-1, end is small (wrapped around).
    #[test]
    fn at_1535_masked_timestamp_delta_boundary() {
        // Case: valid_bits=32, counter wraps from near max to small value.
        let valid_bits: u32 = 32;
        let begin: u64 = 0xFFFF_FFFF - 10; // near max of 32-bit range
        let end: u64 = 5;                   // after wrap

        // Without masking: end.wrapping_sub(begin) would be a huge number.
        // With masking: (5 - (0xFFFFFFFF - 10)) mod 2^32 = 16 (small, sane).
        let delta = masked_timestamp_delta(begin, end, valid_bits);
        assert_eq!(delta, 16, "wrapped delta must be 16 (10 + 1 + 5)");

        // Case: no wrap (end > begin in the valid range).
        let begin2: u64 = 100;
        let end2: u64 = 200;
        let delta2 = masked_timestamp_delta(begin2, end2, valid_bits);
        assert_eq!(delta2, 100, "simple delta must be 100");

        // Case: valid_bits=0 → no masking, plain wrapping_sub.
        let delta3 = masked_timestamp_delta(200, 100, 0);
        assert_eq!(delta3, u64::MAX - 99, "valid_bits=0 means no mask, plain wrapping_sub");

        // Case: valid_bits=64 → no masking (>= 64 means full 64-bit counter).
        let delta4 = masked_timestamp_delta(100, 200, 64);
        assert_eq!(delta4, 100, "valid_bits=64 means plain wrapping_sub");
    }

    /// AT-1573: select_timing_source — pure unit test, no GPU (HANDOFF-3).
    ///
    /// GpuTimestamp when valid_bits>0 AND period>0; CpuFenceWall otherwise.
    #[test]
    fn at_1573_resident_timing_source_selection() {
        // GpuTimestamp: both valid_bits and period are positive.
        assert_eq!(
            select_timing_source(32, 1.0),
            ResidentTimingSource::GpuTimestamp,
            "valid_bits=32, period=1.0 → GpuTimestamp"
        );
        assert_eq!(
            select_timing_source(64, 0.5),
            ResidentTimingSource::GpuTimestamp,
            "valid_bits=64, period=0.5 → GpuTimestamp"
        );

        // CpuFenceWall: valid_bits == 0.
        assert_eq!(
            select_timing_source(0, 1.0),
            ResidentTimingSource::CpuFenceWall,
            "valid_bits=0, period=1.0 → CpuFenceWall"
        );

        // CpuFenceWall: period == 0.0.
        assert_eq!(
            select_timing_source(32, 0.0),
            ResidentTimingSource::CpuFenceWall,
            "valid_bits=32, period=0.0 → CpuFenceWall"
        );

        // CpuFenceWall: both zero.
        assert_eq!(
            select_timing_source(0, 0.0),
            ResidentTimingSource::CpuFenceWall,
            "valid_bits=0, period=0.0 → CpuFenceWall"
        );
    }
}
