//! Single-shot Vulkan compute dispatch.
//!
//! `VulkanContext::dispatch` takes a `DispatchRequest` and returns
//! `Vec<Vec<u8>>` (one output byte slice per buffer binding). It is the
//! primary user-facing API for M1.5 GPU execution.
//!
//! ## Dispatch pipeline
//!
//! 1. Validate request (binding count, push-constant size, workgroup limits)
//! 2. Build compute pipeline (shader module + DSL + pipeline layout + pipeline)
//! 3. Allocate descriptor pool + set
//! 4. Allocate host-visible coherent buffers; map + memcpy inputs
//! 5. Update descriptor sets
//! 6. Record command buffer: bind → push constants → dispatch → pipeline barrier
//! 7. Submit to queue with a fence; wait up to `AXC_FENCE_TIMEOUT_MS` ms
//! 8. Map output buffers; copy bytes; unmap
//! 9. `DispatchResources` RAII cleans up all handles on return (success or error)

use ash::vk;
use axc_hir::ParamBindingPlan;
use crate::context::VulkanContext;
use crate::error::DispatchError;
use crate::resources::{DispatchResources, ResourceHandle};
use crate::pipeline::build_compute_pipeline;
use crate::buffers::{allocate_host_visible_buffer, HostVisibleBuffer};

/// Default fence timeout in nanoseconds (10 seconds).
const DEFAULT_FENCE_TIMEOUT_MS: u64 = 10_000;

/// A single-shot compute dispatch request.
///
/// All slices are borrowed from the caller for the duration of the dispatch call.
/// The caller retains ownership; no data is copied beyond what is needed for
/// the GPU upload.
pub struct DispatchRequest<'a> {
    /// SPIR-V word slice to create the shader module from.
    pub spirv: &'a [u32],
    /// Binding plan from the compiled kernel (descriptor + push-constant layout).
    pub binding_plan: &'a ParamBindingPlan,
    /// Workgroup dispatch dimensions (X, Y, Z).
    pub workgroups: [u32; 3],
    /// Input byte slices, one per buffer binding (in binding order).
    /// Length must equal `binding_plan.buffers.len()`.
    pub inputs: &'a [&'a [u8]],
    /// Output buffer sizes in bytes, one per buffer binding.
    /// Length must equal `binding_plan.buffers.len()`.
    pub output_sizes: &'a [usize],
    /// Push-constant byte blob. Length must equal `binding_plan.push_constant_total_bytes`.
    pub push_constants: &'a [u8],
    /// SPIR-V entry-point name (typically `"main"`).
    pub entry_point: &'a str,
}

/// Validate a `DispatchRequest` before any Vulkan resource allocation.
///
/// All checks run against the binding plan and device limits. This function is
/// `pub(crate)` so unit tests can exercise it without constructing a `VulkanContext`.
pub(crate) fn validate_request(
    req: &DispatchRequest<'_>,
    max_workgroup_count: [u32; 3],
) -> Result<(), DispatchError> {
    // Check 1 & 2: inputs and output_sizes must have the same length as binding_plan.buffers.
    let expected: usize = req.binding_plan.buffers.len();
    if req.inputs.len() != expected {
        return Err(DispatchError::BindingCountMismatch {
            expected,
            provided: req.inputs.len(),
        });
    }
    if req.output_sizes.len() != expected {
        return Err(DispatchError::BindingCountMismatch {
            expected,
            provided: req.output_sizes.len(),
        });
    }

    // Check 3: push-constant bytes length must match binding_plan.push_constant_total_bytes.
    let expected_pc: usize = req.binding_plan.push_constant_total_bytes as usize;
    if req.push_constants.len() != expected_pc {
        return Err(DispatchError::PushConstantSizeMismatch {
            expected: expected_pc,
            provided: req.push_constants.len(),
        });
    }

    // Check 4: workgroup count must not exceed device maximums (W3 fix, rev 1).
    for axis in 0..3_usize {
        if req.workgroups[axis] > max_workgroup_count[axis] {
            return Err(DispatchError::WorkgroupCountExceedsDeviceLimit {
                requested: req.workgroups,
                max: max_workgroup_count,
            });
        }
    }

    Ok(())
}

impl VulkanContext {
    /// Execute a single compute dispatch and return the output buffer bytes.
    ///
    /// Returns `Vec<Vec<u8>>` — one entry per buffer binding in `binding_plan.buffers` order.
    /// On error, all Vulkan resources are cleaned up via `DispatchResources` RAII before
    /// the error is returned to the caller.
    pub fn dispatch(&self, req: DispatchRequest<'_>) -> Result<Vec<Vec<u8>>, DispatchError> {
        // ── Pre-validation (no Vulkan calls) ─────────────────────────────────
        validate_request(&req, self.max_compute_work_group_count)?;

        let binding_count: usize = req.binding_plan.buffers.len();

        // ── RAII resource holder ──────────────────────────────────────────────
        // Every handle is pushed here immediately after creation.
        // DispatchResources::drop() will destroy everything in dependency-correct order.
        let mut res: DispatchResources<'_> = DispatchResources::new(&self.device);

        // ── Step 1-4: Build compute pipeline ─────────────────────────────────
        let compiled = build_compute_pipeline(
            &self.device,
            req.spirv,
            req.binding_plan,
            req.entry_point,
        )?;
        res.push(ResourceHandle::ShaderModule(compiled.shader_module));
        res.push(ResourceHandle::DescriptorSetLayout(compiled.descriptor_set_layout));
        res.push(ResourceHandle::PipelineLayout(compiled.pipeline_layout));
        res.push(ResourceHandle::Pipeline(compiled.pipeline));

        // ── Step 5: Descriptor pool + set ─────────────────────────────────────
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(binding_count as u32);

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(std::slice::from_ref(&pool_size));

        // SAFETY: pool_info references pool_size which is valid for this call.
        let descriptor_pool: vk::DescriptorPool =
            unsafe { self.device.create_descriptor_pool(&pool_info, None) }
                .map_err(|e| DispatchError::DescriptorPoolFailed(e.to_string()))?;
        res.push(ResourceHandle::DescriptorPool(descriptor_pool));

        let dsl_slice: [vk::DescriptorSetLayout; 1] = [compiled.descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&dsl_slice);

        // SAFETY: alloc_info references dsl_slice and descriptor_pool which are valid.
        let descriptor_sets: Vec<vk::DescriptorSet> =
            unsafe { self.device.allocate_descriptor_sets(&alloc_info) }
                .map_err(|e| DispatchError::DescriptorPoolFailed(e.to_string()))?;
        let descriptor_set: vk::DescriptorSet = descriptor_sets[0];
        // Descriptor sets are implicitly freed when the descriptor pool is destroyed.

        // ── Step 6: Allocate host-visible buffers, upload inputs ───────────────
        let mut hvbs: Vec<HostVisibleBuffer> = Vec::with_capacity(binding_count);

        for i in 0..binding_count {
            let input_len: u64 = req.inputs[i].len() as u64;
            let output_len: u64 = req.output_sizes[i] as u64;
            let buf_size: u64 = input_len.max(output_len).max(4);

            let hvb: HostVisibleBuffer = allocate_host_visible_buffer(
                &self.device,
                &self.memory_properties,
                buf_size,
                i as u32,
            )?;
            res.push(ResourceHandle::Buffer(hvb.buffer));
            res.push(ResourceHandle::DeviceMemory(hvb.memory));

            // Upload input data if non-empty.
            if !req.inputs[i].is_empty() {
                // SAFETY: memory is HOST_VISIBLE|HOST_COHERENT; mapping offset 0 with
                // size = allocated size is valid. No flush needed (HOST_COHERENT).
                let ptr = unsafe {
                    self.device.map_memory(hvb.memory, 0, hvb.size, vk::MemoryMapFlags::empty())
                }
                .map_err(|e| DispatchError::MemoryMapFailed(e.to_string()))?;

                // SAFETY: ptr points to at least `hvb.size` bytes of mapped GPU-visible
                // host memory. `req.inputs[i].len()` <= `hvb.size` by construction above.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        req.inputs[i].as_ptr(),
                        ptr as *mut u8,
                        req.inputs[i].len(),
                    );
                }

                // SAFETY: memory was successfully mapped above; unmap is always safe after map.
                unsafe { self.device.unmap_memory(hvb.memory); }
            }

            hvbs.push(hvb);
        }

        // ── Step 7: Update descriptor sets ────────────────────────────────────
        let buffer_infos: Vec<vk::DescriptorBufferInfo> = hvbs
            .iter()
            .map(|hvb| {
                vk::DescriptorBufferInfo::default()
                    .buffer(hvb.buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            })
            .collect();

        let writes: Vec<vk::WriteDescriptorSet> = req
            .binding_plan
            .buffers
            .iter()
            .enumerate()
            .map(|(i, slot)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(slot.buffer_position)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&buffer_infos[i]))
            })
            .collect();

        // SAFETY: writes reference valid descriptor set and buffer handles.
        unsafe { self.device.update_descriptor_sets(&writes, &[]); }

        // ── Step 8: Allocate and record command buffer ─────────────────────────
        let cb_alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        // SAFETY: command_pool is valid; we push the command buffer into res for cleanup.
        let cbs: Vec<vk::CommandBuffer> =
            unsafe { self.device.allocate_command_buffers(&cb_alloc_info) }
                .map_err(|e| DispatchError::CommandBufferRecordFailed(e.to_string()))?;
        let cb: vk::CommandBuffer = cbs[0];
        res.push(ResourceHandle::CommandBuffer {
            pool: self.command_pool,
            buffer: cb,
        });

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        // SAFETY: cb is a freshly allocated PRIMARY command buffer; begin is valid.
        unsafe { self.device.begin_command_buffer(cb, &begin_info) }
            .map_err(|e| DispatchError::CommandBufferRecordFailed(e.to_string()))?;

        // SAFETY: pipeline and descriptor set are valid; pipeline_layout matches.
        unsafe {
            self.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, compiled.pipeline);
            self.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                compiled.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
        }

        // Push constants if the kernel has any.
        if !req.push_constants.is_empty() {
            // SAFETY: pipeline_layout was created with a push-constant range covering
            // offset 0..push_constant_total_bytes, stage COMPUTE. The slice length
            // was validated against push_constant_total_bytes in validate_request.
            unsafe {
                self.device.cmd_push_constants(
                    cb,
                    compiled.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    req.push_constants,
                );
            }
        }

        // SAFETY: workgroup counts were validated against device limits in validate_request.
        unsafe {
            self.device.cmd_dispatch(
                cb,
                req.workgroups[0],
                req.workgroups[1],
                req.workgroups[2],
            );
        }

        // Pipeline barrier: SHADER_WRITE → HOST_READ.
        // Required so host-visible memory reads after fence wait observe GPU writes
        // (necessary on non-coherent-cache architectures; harmless on Lavapipe).
        let memory_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);

        // SAFETY: memory_barrier is correctly formed; command buffer is recording.
        unsafe {
            self.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[memory_barrier],
                &[],
                &[],
            );
        }

        // SAFETY: command buffer has a valid begin; no unclosed render passes.
        unsafe { self.device.end_command_buffer(cb) }
            .map_err(|e| DispatchError::CommandBufferRecordFailed(e.to_string()))?;

        // ── Step 9: Fence creation + queue submit ─────────────────────────────
        let fence_info = vk::FenceCreateInfo::default();

        // SAFETY: fence_info is valid; fence will be pushed into res for cleanup.
        let fence: vk::Fence = unsafe { self.device.create_fence(&fence_info, None) }
            .map_err(|e| DispatchError::CommandBufferRecordFailed(format!("create_fence: {e}")))?;
        res.push(ResourceHandle::Fence(fence));

        let cb_slice: [vk::CommandBuffer; 1] = [cb];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cb_slice);

        // SAFETY: submit_info references cb which is a valid, completed command buffer.
        // fence is a valid, unsignaled fence created above.
        unsafe { self.device.queue_submit(self.queue, &[submit_info], fence) }
            .map_err(|e| DispatchError::QueueSubmitFailed(e.to_string()))?;

        // ── Step 10: Wait for fence ────────────────────────────────────────────
        let timeout_ms: u64 = std::env::var("AXC_FENCE_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(DEFAULT_FENCE_TIMEOUT_MS);
        let timeout_ns: u64 = timeout_ms * 1_000_000;

        // SAFETY: fence is valid and was submitted with the command buffer above.
        let wait_result = unsafe {
            self.device.wait_for_fences(&[fence], true, timeout_ns)
        };
        if let Err(vk::Result::TIMEOUT) = wait_result {
            return Err(DispatchError::FenceTimeout { timeout_ns });
        }
        wait_result.map_err(|e| DispatchError::QueueSubmitFailed(e.to_string()))?;

        // ── Step 11: Readback output buffers ──────────────────────────────────
        let mut outputs: Vec<Vec<u8>> = Vec::with_capacity(binding_count);

        for (i, hvb) in hvbs.iter().enumerate() {
            let out_size: usize = req.output_sizes[i];
            if out_size == 0 {
                outputs.push(Vec::new());
                continue;
            }

            // SAFETY: memory is HOST_VISIBLE|HOST_COHERENT; fence has been waited on,
            // so GPU writes are visible to the host.
            let ptr = unsafe {
                self.device.map_memory(hvb.memory, 0, hvb.size, vk::MemoryMapFlags::empty())
            }
            .map_err(|e| DispatchError::ReadbackFailed {
                binding: i as u32,
                reason: e.to_string(),
            })?;

            let mut out_bytes: Vec<u8> = vec![0u8; out_size];

            // SAFETY: ptr is valid mapped memory; out_size <= hvb.size (by construction
            // in the buffer allocation step above).
            unsafe {
                std::ptr::copy_nonoverlapping(
                    ptr as *const u8,
                    out_bytes.as_mut_ptr(),
                    out_size,
                );
            }

            // SAFETY: memory was successfully mapped above.
            unsafe { self.device.unmap_memory(hvb.memory); }

            outputs.push(out_bytes);
        }

        // DispatchResources drops here, cleaning up all GPU handles.
        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axc_hir::{ParamBindingPlan, BufferBindingSlot, ScalarPushConstantSlot, BufferTy, ScalarTy};
    use axc_hir::buffer::BufferAccess;
    use axc_lexer::Span;

    /// Build a minimal 2-buffer no-scalar binding plan for tests.
    fn two_buffer_plan() -> ParamBindingPlan {
        ParamBindingPlan {
            buffers: vec![
                BufferBindingSlot {
                    name: "a".to_owned(),
                    ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadOnly },
                    position: 0,
                    buffer_position: 0,
                    span: Span::default(),
                },
                BufferBindingSlot {
                    name: "b".to_owned(),
                    ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadWrite },
                    position: 1,
                    buffer_position: 1,
                    span: Span::default(),
                },
            ],
            scalars: Vec::new(),
            push_constant_total_bytes: 0,
        }
    }

    fn scalar_plan_8bytes() -> ParamBindingPlan {
        ParamBindingPlan {
            buffers: Vec::new(),
            scalars: vec![
                ScalarPushConstantSlot {
                    name: "n".to_owned(),
                    ty: ScalarTy::U32,
                    offset: 0,
                    member_index: 0,
                    position: 0,
                    span: Span::default(),
                },
                ScalarPushConstantSlot {
                    name: "alpha".to_owned(),
                    ty: ScalarTy::F32,
                    offset: 4,
                    member_index: 1,
                    position: 1,
                    span: Span::default(),
                },
            ],
            push_constant_total_bytes: 8,
        }
    }

    /// AT-506: validate_request rejects binding count mismatch (inputs.len() != plan.buffers.len()).
    #[test]
    fn at_506_dispatch_rejects_binding_count_mismatch_without_vulkan() {
        let plan: ParamBindingPlan = two_buffer_plan();
        // Only 1 input instead of 2.
        let input_data: Vec<u8> = vec![0u8; 4];
        let spirv: Vec<u32> = vec![0u32; 8];
        let req = DispatchRequest {
            spirv: &spirv,
            binding_plan: &plan,
            workgroups: [1, 1, 1],
            inputs: &[&input_data],          // 1 input, plan expects 2
            output_sizes: &[4, 4],
            push_constants: &[],
            entry_point: "main",
        };

        let result = validate_request(&req, [65535, 65535, 65535]);
        match result {
            Err(DispatchError::BindingCountMismatch { expected: 2, provided: 1 }) => {}
            other => panic!("expected BindingCountMismatch{{2,1}}, got: {other:?}"),
        }
    }

    /// AT-507: validate_request rejects push-constant size mismatch.
    #[test]
    fn at_507_dispatch_rejects_push_constant_size_mismatch_without_vulkan() {
        let plan: ParamBindingPlan = scalar_plan_8bytes();
        let spirv: Vec<u32> = vec![0u32; 8];
        // Provide 4 bytes instead of 8.
        let wrong_pc: Vec<u8> = vec![0u8; 4];
        let req = DispatchRequest {
            spirv: &spirv,
            binding_plan: &plan,
            workgroups: [1, 1, 1],
            inputs: &[],
            output_sizes: &[],
            push_constants: &wrong_pc,      // 4 bytes, plan expects 8
            entry_point: "main",
        };

        let result = validate_request(&req, [65535, 65535, 65535]);
        match result {
            Err(DispatchError::PushConstantSizeMismatch { expected: 8, provided: 4 }) => {}
            other => panic!("expected PushConstantSizeMismatch{{8,4}}, got: {other:?}"),
        }
    }
}
