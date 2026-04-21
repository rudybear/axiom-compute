//! RAII carrier for per-dispatch Vulkan resource handles.
//!
//! `DispatchResources` accumulates handles as they are created during a
//! `dispatch()` call. Its `Drop` implementation destroys all handles in
//! dependency-correct order (dependents before dependencies), ensuring that
//! resources are freed on both success paths, `?`-early-returns, and panics.
//!
//! ## M2.3a note
//!
//! `DeviceOwner` lifetime contract: `VulkanContext` holds `Arc<DeviceOwner>`;
//! `KernelHandleInner` clones it. `vkDestroyDevice` fires only when the last
//! Arc drops. `DispatchResources` holds a plain reference `&'dev ash::Device`,
//! which is borrowed from the `DeviceOwner` via Deref — safe because
//! `DispatchResources` is always dropped before the context that created it.
//!
//! ## Dependency-correct destruction order
//!
//! This is NOT strictly reverse-of-creation — it is the order Vulkan requires
//! per VK spec §3.3.3 object lifetimes (W6 fix from M1.5 rev 1):
//!
//! 1. Fence (no dependencies on other per-dispatch resources)
//! 2. CommandBuffer (freed by destroying its pool; must be freed before pool)
//! 3. DescriptorPool (frees all descriptor sets allocated from it)
//! 4. Pipeline (depends on pipeline layout and shader module)
//! 5. PipelineLayout (depends on descriptor set layout)
//! 6. DescriptorSetLayout (no further dependencies)
//! 7. ShaderModule (no further dependencies)
//! 8. Buffer (must be destroyed before its backing DeviceMemory)
//! 9. DeviceMemory (freed after its buffer)

use ash::vk;

/// One Vulkan resource handle, tagged by kind for type-safe destruction.
///
/// Retained for the legacy one-shot dispatch path and future use.
#[allow(dead_code)]
pub(crate) enum ResourceHandle {
    Fence(vk::Fence),
    CommandBuffer {
        pool: vk::CommandPool,
        buffer: vk::CommandBuffer,
    },
    DescriptorPool(vk::DescriptorPool),
    Pipeline(vk::Pipeline),
    PipelineLayout(vk::PipelineLayout),
    DescriptorSetLayout(vk::DescriptorSetLayout),
    ShaderModule(vk::ShaderModule),
    Buffer(vk::Buffer),
    DeviceMemory(vk::DeviceMemory),
}

/// RAII holder for per-dispatch Vulkan handles.
///
/// Create at the top of `dispatch()`, push each handle immediately after
/// successful creation, and let Drop clean everything up.
///
/// On error paths, handles pushed before the failing step are still cleaned up
/// by Drop when the local variable goes out of scope (via `?` or early return).
///
/// Retained for legacy compatibility and potential future use in M3+.
#[allow(dead_code)]
pub(crate) struct DispatchResources<'dev> {
    /// Reference to the logical device used for destruction.
    device: &'dev ash::Device,
    /// Accumulated handles in creation order (NOT in destruction order).
    handles: Vec<ResourceHandle>,
}

impl<'dev> DispatchResources<'dev> {
    /// Create an empty resource holder for the given logical device.
    #[allow(dead_code)]
    pub(crate) fn new(device: &'dev ash::Device) -> Self {
        Self {
            device,
            handles: Vec::new(),
        }
    }

    /// Record a new handle. The handle will be destroyed in `Drop`.
    #[allow(dead_code)]
    pub(crate) fn push(&mut self, h: ResourceHandle) {
        self.handles.push(h);
    }
}

impl<'dev> Drop for DispatchResources<'dev> {
    fn drop(&mut self) {
        // Destroy in dependency-correct order (Vulkan spec §3.3.3).
        // We do a multi-pass scan: each pass destroys all handles of one class,
        // in the class priority defined below. This is equivalent to sorting by
        // priority then destroying, but avoids a sort allocation.

        // Helper closures defined inline to share `self.device` borrow.
        let device = self.device;

        macro_rules! destroy_all {
            ($pat:pat => $body:expr) => {
                for h in &self.handles {
                    #[allow(irrefutable_let_patterns)]
                    if let $pat = h {
                        // SAFETY: each variant's handle was successfully created and
                        // has not been destroyed yet. The device outlives this scope
                        // because DispatchResources holds a reference to it.
                        unsafe { $body }
                    }
                }
            };
        }

        // Pass 1: Fences
        destroy_all!(ResourceHandle::Fence(fence) => device.destroy_fence(*fence, None));

        // Pass 2: CommandBuffers (must free before pool destruction if pool is separate,
        //         but here pool is the VulkanContext's CommandPool — we only free the buffer)
        destroy_all!(ResourceHandle::CommandBuffer { pool, buffer } => {
            device.free_command_buffers(*pool, &[*buffer]);
        });

        // Pass 3: DescriptorPools (frees all descriptor sets allocated from them)
        destroy_all!(ResourceHandle::DescriptorPool(pool) => device.destroy_descriptor_pool(*pool, None));

        // Pass 4: Pipelines (depends on PipelineLayout + ShaderModule)
        destroy_all!(ResourceHandle::Pipeline(pipeline) => device.destroy_pipeline(*pipeline, None));

        // Pass 5: PipelineLayouts (depends on DescriptorSetLayout)
        destroy_all!(ResourceHandle::PipelineLayout(layout) => device.destroy_pipeline_layout(*layout, None));

        // Pass 6: DescriptorSetLayouts
        destroy_all!(ResourceHandle::DescriptorSetLayout(layout) => device.destroy_descriptor_set_layout(*layout, None));

        // Pass 7: ShaderModules
        destroy_all!(ResourceHandle::ShaderModule(module) => device.destroy_shader_module(*module, None));

        // Pass 8: Buffers (must be destroyed before their backing memory)
        destroy_all!(ResourceHandle::Buffer(buffer) => device.destroy_buffer(*buffer, None));

        // Pass 9: DeviceMemory (freed after its buffers)
        destroy_all!(ResourceHandle::DeviceMemory(memory) => device.free_memory(*memory, None));
    }
}
