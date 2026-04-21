//! Single-shot Vulkan compute dispatch (legacy one-shot path).
//!
//! `VulkanContext::dispatch` takes a `DispatchRequest` and returns
//! `Vec<Vec<u8>>` (one output byte slice per buffer binding). It is the
//! backward-compatible API preserved from M1.5.
//!
//! ## M2.3a note
//!
//! `dispatch()` is now a thin wrapper around `prepare_kernel` + `dispatch_handle`.
//! The handle is dropped at function return, so sequential one-shots pay the full
//! pipeline-compile + buffer-alloc cost each time. For repeated dispatches with the
//! same kernel, use `prepare_kernel` / `dispatch_handle` directly.
//!
//! ## Dispatch pipeline (one-shot)
//!
//! 1. Validate request (binding count, push-constant size, workgroup limits)
//! 2. prepare_kernel: build compute pipeline (shader module + DSL + pipeline layout)
//! 3. dispatch_handle: allocate staging buffers, upload, dispatch, readback
//! 4. Handle dropped at return → all Vulkan handles freed

use axc_hir::ParamBindingPlan;
use crate::context::VulkanContext;
use crate::error::DispatchError;

/// Default fence timeout in milliseconds (10 seconds).
pub(crate) const DEFAULT_FENCE_TIMEOUT_MS: u64 = 10_000;

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
    ///
    /// ## M2.3a note
    ///
    /// This method is now a thin wrapper around `prepare_kernel` + `dispatch_handle`.
    /// The `KernelHandle` is created and destroyed on every call, so sequential
    /// one-shots pay pipeline-compile + buffer-alloc cost each time. For
    /// repeated dispatches with the same kernel, use `prepare_kernel` /
    /// `dispatch_handle` directly.
    pub fn dispatch(&self, req: DispatchRequest<'_>) -> Result<Vec<Vec<u8>>, DispatchError> {
        // Pre-validation (no Vulkan calls).
        validate_request(&req, self.max_compute_work_group_count)?;

        // Delegate to prepare_kernel + dispatch_handle.
        // The handle is dropped at function return → all Vulkan handles freed.
        let handle = self.prepare_kernel(
            req.spirv,
            req.binding_plan,
            req.binding_plan.push_constant_total_bytes,
            req.entry_point,
        )?;

        let out = self.dispatch_handle(
            &handle,
            (req.workgroups[0], req.workgroups[1], req.workgroups[2]),
            req.inputs,
            req.output_sizes,
            req.push_constants,
        )?;

        // handle dropped here → pipeline + DSL + PL + fence + buffers destroyed
        Ok(out)
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
