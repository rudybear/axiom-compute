//! Compute pipeline construction helpers.
//!
//! Builds the Vulkan compute pipeline from:
//! - A SPIR-V word slice (shader module)
//! - A `ParamBindingPlan` (descriptor set layout + push-constant range)
//! - An entry-point name
//! - An optional `vk::PipelineCache` handle (M2.3a; `VK_NULL_HANDLE` in legacy path)
//!
//! Supports 0-buffer kernels (P-5 fix): when `binding_plan.buffers.is_empty()`,
//! no descriptor set layout is created and `descriptor_set_layout` is `None`.
//! The pipeline layout uses only the push-constant range in that case.

use ash::vk;
use axc_hir::ParamBindingPlan;
use crate::error::DispatchError;

/// A fully built compute pipeline with all its constituent Vulkan handles.
///
/// Owned by `DispatchResources` (legacy path) or `KernelHandleInner` (M2.3a path).
/// Destroyed in dependency-correct order on Drop.
pub(crate) struct CompiledPipeline {
    /// SPIR-V shader module.
    pub(crate) shader_module: vk::ShaderModule,
    /// Descriptor set layout for binding set 0.
    /// `None` when `binding_plan.buffers.is_empty()` (push-constant-only kernel, P-5).
    pub(crate) descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    /// Pipeline layout (combines descriptor set layout + push-constant range).
    pub(crate) pipeline_layout: vk::PipelineLayout,
    /// The compute pipeline itself.
    pub(crate) pipeline: vk::Pipeline,
    /// The entry-point name as a CString (stored to ensure the pointer remains valid
    /// for the pipeline's lifetime when used in KernelHandleInner).
    pub(crate) entry_point_cstr: std::ffi::CString,
}

/// Build a `CompiledPipeline` from a SPIR-V word slice, binding plan, and optional cache.
///
/// Steps (per spec §4):
/// 1. Create `ShaderModule` from `spirv` words.
/// 2. Build `DescriptorSetLayout` from `binding_plan.buffers` (all `STORAGE_BUFFER`, stage `COMPUTE`).
///    Skipped when `binding_plan.buffers.is_empty()` (returns `None` for DSL, P-5 fix).
/// 3. Build `PipelineLayout` with an optional push-constant range.
/// 4. Create `ComputePipeline` with the provided `pipeline_cache` handle
///    (`vk::PipelineCache::null()` for the legacy path; a real handle for M2.3a).
///
/// Binding numbers are taken directly from `BufferBindingSlot::buffer_position`.
/// No HashMap is used to drive the binding order (invariant from spec §14).
pub(crate) fn build_compute_pipeline(
    device: &ash::Device,
    spirv: &[u32],
    binding_plan: &ParamBindingPlan,
    entry_point: &str,
    pipeline_cache: vk::PipelineCache,
) -> Result<CompiledPipeline, DispatchError> {
    // ── Step 1: Shader module ──────────────────────────────────────────────
    let shader_module_info = vk::ShaderModuleCreateInfo::default()
        .code(spirv);

    // SAFETY: shader_module_info.code references `spirv` which is valid for this call's
    // duration. `create_shader_module` copies the SPIR-V words internally.
    let shader_module: vk::ShaderModule =
        unsafe { device.create_shader_module(&shader_module_info, None) }
            .map_err(|e| DispatchError::ShaderModuleCreationFailed(e.to_string()))?;

    // ── Step 2: Descriptor set layout ─────────────────────────────────────
    // P-5 fix: skip DSL when there are no buffer bindings (push-constant-only kernel).
    let descriptor_set_layout: Option<vk::DescriptorSetLayout> =
        if binding_plan.buffers.is_empty() {
            None
        } else {
            // Build one `VkDescriptorSetLayoutBinding` per buffer in the binding plan.
            // Binding numbers come from `buffer_position` (invariant: no HashMap, no reordering).
            let dsl_bindings: Vec<vk::DescriptorSetLayoutBinding> = binding_plan
                .buffers
                .iter()
                .map(|slot| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(slot.buffer_position)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();

            let dsl_info = vk::DescriptorSetLayoutCreateInfo::default()
                .bindings(&dsl_bindings);

            // SAFETY: dsl_info references dsl_bindings which are valid for this call's duration.
            let dsl: vk::DescriptorSetLayout =
                unsafe { device.create_descriptor_set_layout(&dsl_info, None) }
                    .map_err(|e| {
                        // SAFETY: shader_module was successfully created; destroy it on error.
                        unsafe { device.destroy_shader_module(shader_module, None); }
                        DispatchError::DescriptorSetLayoutFailed(e.to_string())
                    })?;
            Some(dsl)
        };

    // ── Step 3: Pipeline layout ────────────────────────────────────────────
    let push_constant_range: Option<vk::PushConstantRange> =
        if binding_plan.push_constant_total_bytes > 0 {
            Some(
                vk::PushConstantRange::default()
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .offset(0)
                    .size(binding_plan.push_constant_total_bytes),
            )
        } else {
            None
        };

    let pc_ranges: &[vk::PushConstantRange] = match &push_constant_range {
        Some(r) => std::slice::from_ref(r),
        None => &[],
    };

    // P-5 fix: use an empty set_layouts slice when there is no DSL.
    let dsl_as_array: Option<[vk::DescriptorSetLayout; 1]> =
        descriptor_set_layout.map(|dsl| [dsl]);
    let dsl_slice: &[vk::DescriptorSetLayout] = match &dsl_as_array {
        Some(arr) => arr.as_slice(),
        None => &[],
    };

    let pl_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(dsl_slice)
        .push_constant_ranges(pc_ranges);

    // SAFETY: pl_info references dsl_slice and pc_ranges which are valid for this call.
    let pipeline_layout: vk::PipelineLayout =
        unsafe { device.create_pipeline_layout(&pl_info, None) }
            .map_err(|e| {
                // SAFETY: destroy already-created handles on error.
                unsafe {
                    if let Some(dsl) = descriptor_set_layout {
                        device.destroy_descriptor_set_layout(dsl, None);
                    }
                    device.destroy_shader_module(shader_module, None);
                }
                DispatchError::PipelineLayoutFailed(e.to_string())
            })?;

    // ── Step 4: Compute pipeline ───────────────────────────────────────────
    // Caller provides pipeline_cache: VK_NULL_HANDLE for legacy path,
    // a real PipelineCache handle for M2.3a (speeds up repeated prepare_kernel calls).
    let entry_cstr: std::ffi::CString = std::ffi::CString::new(entry_point)
        .unwrap_or_else(|_| std::ffi::CString::new("main").unwrap());

    let stage_info = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(&entry_cstr);

    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(stage_info)
        .layout(pipeline_layout);

    // SAFETY: pipeline_info and stage_info reference valid handles. pipeline_cache
    // may be VK_NULL_HANDLE (legal per Vulkan spec §10.6) or a valid pipeline cache.
    let pipelines: Vec<vk::Pipeline> =
        unsafe {
            device.create_compute_pipelines(pipeline_cache, &[pipeline_info], None)
        }
        .map_err(|(_, e)| {
            // SAFETY: destroy already-created handles on error.
            unsafe {
                device.destroy_pipeline_layout(pipeline_layout, None);
                if let Some(dsl) = descriptor_set_layout {
                    device.destroy_descriptor_set_layout(dsl, None);
                }
                device.destroy_shader_module(shader_module, None);
            }
            DispatchError::PipelineCreationFailed(e.to_string())
        })?;

    let pipeline: vk::Pipeline = pipelines.into_iter().next()
        .ok_or_else(|| {
            // SAFETY: destroy handles if pipeline vector is unexpectedly empty.
            unsafe {
                device.destroy_pipeline_layout(pipeline_layout, None);
                if let Some(dsl) = descriptor_set_layout {
                    device.destroy_descriptor_set_layout(dsl, None);
                }
                device.destroy_shader_module(shader_module, None);
            }
            DispatchError::PipelineCreationFailed("create_compute_pipelines returned empty vec".to_owned())
        })?;

    Ok(CompiledPipeline {
        shader_module,
        descriptor_set_layout,
        pipeline_layout,
        pipeline,
        entry_point_cstr: entry_cstr,
    })
}
