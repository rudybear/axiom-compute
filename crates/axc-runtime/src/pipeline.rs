//! Compute pipeline construction helpers.
//!
//! Builds the Vulkan compute pipeline from:
//! - A SPIR-V word slice (shader module)
//! - A `ParamBindingPlan` (descriptor set layout + push-constant range)
//! - An entry-point name
//!
//! `VK_NULL_HANDLE` is used for the pipeline cache (M1.5 does not cache pipelines).
//! M2 may add a per-context `VkPipelineCache` for performance.

use ash::vk;
use axc_hir::ParamBindingPlan;
use crate::error::DispatchError;

/// A fully built compute pipeline with all its constituent Vulkan handles.
///
/// Owned by `DispatchResources` and destroyed in dependency-correct order on Drop.
pub(crate) struct CompiledPipeline {
    /// SPIR-V shader module.
    pub(crate) shader_module: vk::ShaderModule,
    /// Descriptor set layout for binding set 0.
    pub(crate) descriptor_set_layout: vk::DescriptorSetLayout,
    /// Pipeline layout (combines descriptor set layout + push-constant range).
    pub(crate) pipeline_layout: vk::PipelineLayout,
    /// The compute pipeline itself.
    pub(crate) pipeline: vk::Pipeline,
}

/// Build a `CompiledPipeline` from a SPIR-V word slice and a binding plan.
///
/// Steps (per spec §4):
/// 1. Create `ShaderModule` from `spirv` words.
/// 2. Build `DescriptorSetLayout` from `binding_plan.buffers` (all `STORAGE_BUFFER`, stage `COMPUTE`).
/// 3. Build `PipelineLayout` with an optional push-constant range.
/// 4. Create `ComputePipeline` with `vk::PipelineCache::null()` (no caching in M1.5, W2 fix).
///
/// Binding numbers are taken directly from `BufferBindingSlot::buffer_position`.
/// No HashMap is used to drive the binding order (invariant from spec §14).
pub(crate) fn build_compute_pipeline(
    device: &ash::Device,
    spirv: &[u32],
    binding_plan: &ParamBindingPlan,
    entry_point: &str,
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
    let descriptor_set_layout: vk::DescriptorSetLayout =
        unsafe { device.create_descriptor_set_layout(&dsl_info, None) }
            .map_err(|e| {
                // SAFETY: shader_module was successfully created; destroy it on error.
                unsafe { device.destroy_shader_module(shader_module, None); }
                DispatchError::DescriptorSetLayoutFailed(e.to_string())
            })?;

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

    let dsl_slice: [vk::DescriptorSetLayout; 1] = [descriptor_set_layout];
    let pl_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&dsl_slice)
        .push_constant_ranges(pc_ranges);

    // SAFETY: pl_info references dsl_slice and pc_ranges which are valid for this call.
    let pipeline_layout: vk::PipelineLayout =
        unsafe { device.create_pipeline_layout(&pl_info, None) }
            .map_err(|e| {
                // SAFETY: destroy already-created handles on error.
                unsafe {
                    device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                    device.destroy_shader_module(shader_module, None);
                }
                DispatchError::PipelineLayoutFailed(e.to_string())
            })?;

    // ── Step 4: Compute pipeline ───────────────────────────────────────────
    // Pipeline cache is VK_NULL_HANDLE in M1.5 (W2 fix — no pipeline caching).
    let entry_cstr: std::ffi::CString = std::ffi::CString::new(entry_point)
        .unwrap_or_else(|_| std::ffi::CString::new("main").unwrap());

    let stage_info = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(&entry_cstr);

    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(stage_info)
        .layout(pipeline_layout);

    // SAFETY: pipeline_info and stage_info reference valid handles. `VK_NULL_HANDLE`
    // for pipeline_cache is explicitly legal per Vulkan spec §10.6.
    let pipelines: Vec<vk::Pipeline> =
        unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        }
        .map_err(|(_, e)| {
            // SAFETY: destroy already-created handles on error.
            unsafe {
                device.destroy_pipeline_layout(pipeline_layout, None);
                device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                device.destroy_shader_module(shader_module, None);
            }
            DispatchError::PipelineCreationFailed(e.to_string())
        })?;

    let pipeline: vk::Pipeline = pipelines.into_iter().next()
        .ok_or_else(|| {
            // SAFETY: destroy handles if pipeline vector is unexpectedly empty.
            unsafe {
                device.destroy_pipeline_layout(pipeline_layout, None);
                device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                device.destroy_shader_module(shader_module, None);
            }
            DispatchError::PipelineCreationFailed("create_compute_pipelines returned empty vec".to_owned())
        })?;

    Ok(CompiledPipeline {
        shader_module,
        descriptor_set_layout,
        pipeline_layout,
        pipeline,
    })
}
