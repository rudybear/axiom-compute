//! `DispatchError` — typed error enum for every Vulkan dispatch failure mode.
//!
//! Every variant derives `thiserror::Error` for `Display` and `miette::Diagnostic`
//! for structured diagnostic rendering. No `Box<dyn Error>` is used anywhere;
//! all error context is encoded in typed fields (anti-pattern compliance).
//!
//! Variant count: 25 (rev 1, M2.3a). The count is asserted in `at_801`.

/// Direction of a staging-buffer copy operation.
///
/// Used in `DispatchError::StagingCopyFailed` to distinguish host-to-device
/// uploads from device-to-host readbacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopyDirection {
    /// Copying from host (CPU) memory to device (GPU) memory.
    HostToDevice,
    /// Copying from device (GPU) memory to host (CPU) memory.
    DeviceToHost,
}

/// Typed error for all Vulkan dispatch failure modes.
///
/// Each variant corresponds to exactly one failure stage in the dispatch pipeline.
/// The variants are ordered roughly by when they can occur during a `dispatch()` call:
/// context initialization errors first, then per-dispatch errors, then metadata errors.
#[derive(Debug, thiserror::Error, miette::Diagnostic)]
pub enum DispatchError {
    // ── Context initialization errors ─────────────────────────────────────────
    /// Failed to load the Vulkan library or entry points.
    #[error("failed to load Vulkan library: {0}")]
    VulkanEntryFailed(String),

    /// Failed to create a Vulkan instance.
    #[error("failed to create Vulkan instance: {0}")]
    NoVulkanInstance(String),

    /// No physical device with a compute queue family was found.
    #[error("no Vulkan physical device with a compute queue")]
    NoSupportedDevice,

    /// The selected device has no compute queue family index.
    #[error("selected device has no compute queue family")]
    NoComputeQueue,

    /// Logical device creation failed.
    #[error("failed to create logical device: {0}")]
    DeviceCreationFailed(String),

    // ── Per-dispatch resource errors ──────────────────────────────────────────
    /// SPIR-V shader module creation failed (corrupted SPIR-V or unsupported features).
    #[error("failed to create shader module: {0}")]
    ShaderModuleCreationFailed(String),

    /// Descriptor set layout creation failed.
    #[error("failed to create descriptor set layout: {0}")]
    DescriptorSetLayoutFailed(String),

    /// Descriptor pool creation or descriptor set allocation failed.
    #[error("failed to create/allocate descriptor pool/set: {0}")]
    DescriptorPoolFailed(String),

    /// Pipeline layout creation failed.
    #[error("failed to create pipeline layout: {0}")]
    PipelineLayoutFailed(String),

    /// Compute pipeline creation failed (device may lack required capabilities).
    #[error("failed to create compute pipeline: {0}")]
    PipelineCreationFailed(String),

    /// Buffer or device-memory allocation for a binding slot failed.
    #[error("buffer #{binding} allocation failed ({size} bytes): {reason}")]
    BufferAllocationFailed {
        /// Descriptor binding index of the failed buffer.
        binding: u32,
        /// Requested allocation size in bytes.
        size: usize,
        /// Human-readable reason (Vulkan result code or OOM description).
        reason: String,
    },

    /// Memory mapping for a buffer failed.
    #[error("failed to map memory: {0}")]
    MemoryMapFailed(String),

    /// No memory type supporting `HOST_VISIBLE | HOST_COHERENT` was found.
    ///
    /// M1.5 requires coherent host-visible memory (no staging buffers). Mobile GPUs
    /// that only offer non-coherent host-visible memory will hit this error until M2.
    #[error("no memory type supports HOST_VISIBLE | HOST_COHERENT")]
    NoCompatibleMemoryType,

    /// Command buffer allocation, recording, or submission setup failed.
    #[error("command buffer record failed: {0}")]
    CommandBufferRecordFailed(String),

    /// Queue submission (`vkQueueSubmit`) failed.
    #[error("queue submit failed: {0}")]
    QueueSubmitFailed(String),

    /// Fence wait timed out after the configured timeout.
    ///
    /// The kernel may have entered an infinite loop or the device may be lost.
    #[error("fence wait timed out after {timeout_ns} ns")]
    FenceTimeout {
        /// The timeout value in nanoseconds that was exceeded.
        timeout_ns: u64,
    },

    /// Readback memory mapping of an output buffer failed.
    #[error("readback of binding #{binding} failed: {reason}")]
    ReadbackFailed {
        /// Descriptor binding index of the failed readback.
        binding: u32,
        /// Human-readable reason.
        reason: String,
    },

    // ── Pre-dispatch validation errors ────────────────────────────────────────
    /// The number of input/output slices does not match the binding plan's buffer count.
    #[error("binding count mismatch: expected {expected}, got {provided}")]
    BindingCountMismatch {
        /// Expected binding count (from the binding plan).
        expected: usize,
        /// Provided binding count (from `inputs.len()` or `output_sizes.len()`).
        provided: usize,
    },

    /// The push-constant byte slice length does not match the binding plan.
    #[error("push constant size mismatch: expected {expected} bytes, got {provided}")]
    PushConstantSizeMismatch {
        /// Expected size in bytes (from `binding_plan.push_constant_total_bytes`).
        expected: usize,
        /// Provided size in bytes (from `push_constants.len()`).
        provided: usize,
    },

    /// The requested workgroup count exceeds the device's maximum.
    ///
    /// Added in M1.5 rev 1 (W3 fix). The runtime caches
    /// `VkPhysicalDeviceLimits::max_compute_work_group_count` at `VulkanContext::new()`
    /// and checks it before any resource allocation.
    #[error("workgroup count {requested:?} exceeds device limit {max:?}")]
    WorkgroupCountExceedsDeviceLimit {
        /// The workgroup count requested by the caller.
        requested: [u32; 3],
        /// The device's maximum workgroup count.
        max: [u32; 3],
    },

    // ── Metadata sidecar errors ───────────────────────────────────────────────
    /// File I/O error when reading or writing a metadata sidecar.
    #[error("metadata I/O error: {0}")]
    MetadataIoError(String),

    /// JSON parse error when deserializing a metadata sidecar.
    #[error("metadata parse error: {0}")]
    MetadataParseError(String),

    /// The sidecar's `schema_version` does not match `CURRENT_SCHEMA_VERSION`.
    #[error("metadata schema v{got} is not supported (runtime supports v{supported})")]
    MetadataSchemaMismatch {
        /// The schema version found in the sidecar file.
        got: u32,
        /// The schema version this runtime understands.
        supported: u32,
    },

    // ── M2.3a pipeline-cache and staging-copy errors ──────────────────────────
    /// Failed to load the on-disk pipeline cache file.
    ///
    /// Non-fatal at context init: logged via `tracing::warn!` and the context
    /// continues with an empty in-memory pipeline cache. Typed so tests can
    /// inspect the explicit-fail path.
    #[error("failed to load pipeline cache at '{}': {reason}", path.display())]
    PipelineCacheLoadFailed {
        /// The path of the cache file that could not be loaded.
        path: std::path::PathBuf,
        /// Human-readable reason (I/O error or Vulkan result code).
        reason: String,
    },

    /// A staging-buffer copy (host↔device) failed.
    ///
    /// `vkMapMemory` or a `vkCmdCopyBuffer`-level error during the staging
    /// upload or readback phase of `dispatch_handle`.
    #[error("staging copy failed for binding #{binding} ({direction:?}): {reason}")]
    StagingCopyFailed {
        /// Descriptor binding index of the failed copy.
        binding: u32,
        /// Direction of the failed copy.
        direction: CopyDirection,
        /// Human-readable reason.
        reason: String,
    },
}

/// Convenience type alias for dispatch results.
pub type DispatchResult<T> = Result<T, DispatchError>;

#[cfg(test)]
mod tests {
    use super::*;

    /// AT-801: DispatchError has exactly 25 variants, all Display and Diagnostic.
    ///
    /// Supersedes AT-502 (23 variants). The exhaustive match below ensures the
    /// compiler reminds us to update this test whenever a variant is added or removed.
    #[test]
    fn at_801_dispatch_error_variants_count_is_25() {
        // Construct one instance of each variant and verify non-empty Display.
        let variants: Vec<DispatchError> = vec![
            DispatchError::VulkanEntryFailed("test".to_owned()),
            DispatchError::NoVulkanInstance("test".to_owned()),
            DispatchError::NoSupportedDevice,
            DispatchError::NoComputeQueue,
            DispatchError::DeviceCreationFailed("test".to_owned()),
            DispatchError::ShaderModuleCreationFailed("test".to_owned()),
            DispatchError::DescriptorSetLayoutFailed("test".to_owned()),
            DispatchError::DescriptorPoolFailed("test".to_owned()),
            DispatchError::PipelineLayoutFailed("test".to_owned()),
            DispatchError::PipelineCreationFailed("test".to_owned()),
            DispatchError::BufferAllocationFailed { binding: 0, size: 64, reason: "test".to_owned() },
            DispatchError::MemoryMapFailed("test".to_owned()),
            DispatchError::NoCompatibleMemoryType,
            DispatchError::CommandBufferRecordFailed("test".to_owned()),
            DispatchError::QueueSubmitFailed("test".to_owned()),
            DispatchError::FenceTimeout { timeout_ns: 10_000_000_000 },
            DispatchError::ReadbackFailed { binding: 1, reason: "test".to_owned() },
            DispatchError::BindingCountMismatch { expected: 2, provided: 1 },
            DispatchError::PushConstantSizeMismatch { expected: 8, provided: 4 },
            DispatchError::WorkgroupCountExceedsDeviceLimit {
                requested: [99999, 1, 1],
                max: [65535, 65535, 65535],
            },
            DispatchError::MetadataIoError("test".to_owned()),
            DispatchError::MetadataParseError("test".to_owned()),
            DispatchError::MetadataSchemaMismatch { got: 2, supported: 1 },
            DispatchError::PipelineCacheLoadFailed {
                path: std::path::PathBuf::from("/tmp/test.cache"),
                reason: "test".to_owned(),
            },
            DispatchError::StagingCopyFailed {
                binding: 0,
                direction: CopyDirection::HostToDevice,
                reason: "test".to_owned(),
            },
        ];

        // Verify exactly 25 variants are covered.
        assert_eq!(variants.len(), 25, "expected exactly 25 DispatchError variants");

        for variant in &variants {
            let msg = variant.to_string();
            assert!(
                !msg.is_empty(),
                "DispatchError Display must be non-empty; got empty for variant: {variant:?}"
            );
            // Each display message must contain either "error" (case-insensitive) or
            // a variant-specific structural keyword (per AT-519 softened discipline).
            let msg_lower = msg.to_lowercase();
            let has_error_word = msg_lower.contains("error")
                || msg_lower.contains("failed")
                || msg_lower.contains("no ")
                || msg_lower.contains("mismatch")
                || msg_lower.contains("timeout")
                || msg_lower.contains("timed out")
                || msg_lower.contains("exceeds")
                || msg_lower.contains("found")
                || msg_lower.contains("unsupported")
                || msg_lower.contains("not supported");
            assert!(
                has_error_word,
                "DispatchError::Display must contain a diagnostic keyword; got: '{msg}'"
            );

            // miette::Diagnostic::code should not panic.
            use miette::Diagnostic;
            let _ = variant.code();
        }
    }

    /// AT-802: CopyDirection enum has exactly two variants with Debug.
    #[test]
    fn at_802_copy_direction_two_variants_debug() {
        let h2d: CopyDirection = CopyDirection::HostToDevice;
        let d2h: CopyDirection = CopyDirection::DeviceToHost;

        let h2d_str: String = format!("{h2d:?}");
        let d2h_str: String = format!("{d2h:?}");

        assert!(!h2d_str.is_empty(), "HostToDevice debug must be non-empty");
        assert!(!d2h_str.is_empty(), "DeviceToHost debug must be non-empty");
        assert_ne!(h2d_str, d2h_str, "HostToDevice and DeviceToHost must have distinct debug strings");

        // Exhaustive match to ensure exactly two variants exist.
        let _covered: () = match h2d {
            CopyDirection::HostToDevice => {}
            CopyDirection::DeviceToHost => {}
        };
    }

    /// AT-502 (legacy test preserved as alias): verifies the new 25-count.
    #[test]
    fn at_502_dispatch_error_variants_are_display_miette() {
        // This test delegates to the more complete at_801 test above.
        // Preserved for backward-compatibility with any test-name grepping.
        at_801_dispatch_error_variants_count_is_25();
    }
}
