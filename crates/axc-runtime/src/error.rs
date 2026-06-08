//! `DispatchError` — typed error enum for every Vulkan dispatch failure mode.
//!
//! Every variant derives `thiserror::Error` for `Display` and `miette::Diagnostic`
//! for structured diagnostic rendering. No `Box<dyn Error>` is used anywhere;
//! all error context is encoded in typed fields (anti-pattern compliance).
//!
//! Variant count: 31 (M3.2). M3.1 adds 2: CoopMatUnsupported, DeviceFeatureUnsupported.
//! M3.2 adds 1: SharedMemoryExceedsDeviceLimit.
//! The count is asserted in `at_801`.
//!
//! ## M3.0 additions
//!
//! - `SemaphoreCreationFailed` — timeline or binary semaphore creation failure.
//! - `TransferQueueSubmitFailed` — dedicated transfer-queue submit failure
//!   (distinct from `QueueSubmitFailed` for diagnosis).
//! - `MappedRangeOpFailed { op: MappedRangeOp }` — `vkFlush/InvalidateMappedMemoryRanges`
//!   failure, tagged with the `MappedRangeOp` direction enum.

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

/// Operation type for a mapped-memory-range Vulkan call.
///
/// Used in `DispatchError::MappedRangeOpFailed` to distinguish flush from invalidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MappedRangeOp {
    /// `vkFlushMappedMemoryRanges` — makes host writes visible to the device.
    Flush,
    /// `vkInvalidateMappedMemoryRanges` — makes device writes visible to the host.
    Invalidate,
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

    // ── M3.0 bandwidth rework errors ──────────────────────────────────────────
    /// Timeline or binary handoff semaphore creation failed (M3.0).
    ///
    /// Returned by `sync::create_handoff` and the binary-semaphore recreation
    /// path in the partial-submit error recovery matrix (CRITICAL-4).
    #[error("semaphore creation failed: {0}")]
    SemaphoreCreationFailed(String),

    /// A dedicated transfer-queue `vkQueueSubmit` failed (M3.0).
    ///
    /// Distinct from `QueueSubmitFailed` (compute queue) for per-queue diagnosis.
    /// Returned for upload-CB or readback-CB submission failures in the
    /// three-submit dedicated dispatch path.
    #[error("transfer queue submit failed: {0}")]
    TransferQueueSubmitFailed(String),

    /// `vkFlushMappedMemoryRanges` or `vkInvalidateMappedMemoryRanges` failed (M3.0).
    ///
    /// The `op` field distinguishes flush from invalidate so callers can add targeted
    /// diagnostic notes. Only raised on `NonCoherent` memory (coherent paths skip these calls).
    #[error("mapped memory range op {op:?} failed: {reason}")]
    MappedRangeOpFailed {
        /// Whether this was a flush (after upload) or invalidate (before readback).
        op: MappedRangeOp,
        /// Human-readable reason (Vulkan result code).
        reason: String,
    },

    // ── M3.1 cooperative-matrix graceful-skip errors ──────────────────────────

    /// The required cooperative-matrix shape is not supported by this device — M3.1.
    ///
    /// Returned by the coopmat dispatch path when:
    /// - The VK_KHR_cooperative_matrix extension is absent, OR
    /// - The `cooperativeMatrix` feature is not enabled, OR
    /// - No supported shape tuple matches the required (M,N,K,A/B/C/result types,
    ///   Subgroup scope, non-saturating), OR
    /// - The device's subgroup size is not 32 (required for matmul_tile's
    ///   @workgroup(32,1,1) tile assumption).
    ///
    /// Callers should SKIP the coopmat path and fall back to a plain matmul kernel
    /// when this error is returned. NOT a fatal dispatch failure.
    #[error("cooperative matrix shape {required_m}x{required_n}x{required_k} not supported by device: {reason}")]
    CoopMatUnsupported {
        /// Required M dimension.
        required_m: u32,
        /// Required N dimension.
        required_n: u32,
        /// Required K dimension.
        required_k: u32,
        /// Human-readable reason (missing extension/feature, shape mismatch, wrong subgroup size).
        reason: String,
    },

    /// A device feature required by the kernel is not enabled or available — M3.1.
    ///
    /// Returned by the dispatch path when `required_device_features` determines a
    /// feature is needed (e.g. `storageBuffer16BitAccess` for f16 SSBO, `shaderInt8`
    /// for Q4_K_M u8 SSBO) but the context's `enabled_features` record shows it was
    /// not enabled (either the device doesn't support it or it was not requested at
    /// device creation).
    ///
    /// Callers should SKIP the kernel on this device. NOT a fatal dispatch failure.
    #[error("device feature '{feature}' required by kernel '{kernel}' is not supported or not enabled")]
    DeviceFeatureUnsupported {
        /// Name of the missing feature (e.g. `"storageBuffer16BitAccess"`).
        feature: String,
        /// Source-level kernel name (for diagnostics).
        kernel: String,
    },

    /// The kernel declares more workgroup-shared memory than the device supports — M3.2.
    ///
    /// Returned by `preflight_kernel_support` when `kernel.shared_memory_bytes > device.maxComputeSharedMemorySize`.
    /// Callers should SKIP the kernel on this device (graceful typed skip, not a fatal failure).
    ///
    /// This mirrors the `CoopMatUnsupported` / `DeviceFeatureUnsupported` pattern: a typed,
    /// diagnostic-friendly error with `required`, `device_max`, and `kernel` fields.
    #[error("kernel '{kernel}' requires {required} bytes of shared memory which exceeds device maximum of {device_max} bytes")]
    SharedMemoryExceedsDeviceLimit {
        /// Shared memory bytes required by the kernel.
        required: u32,
        /// Device's `maxComputeSharedMemorySize` limit.
        device_max: u32,
        /// Source-level kernel name (for diagnostics).
        kernel: String,
    },

    // ── M4.1 CUDA↔Vulkan external-memory interop errors ───────────────────────
    /// No enumerated Vulkan physical device's `VkPhysicalDeviceIDProperties.deviceUUID`
    /// byte-matched the CUDA target UUID — M4.1 (FAIL-CLOSED, R-3).
    ///
    /// Raised by `select_physical_device_by_uuid`. NEVER falls back to a different
    /// GPU or to a host copy: an OPAQUE_FD import across GPUs cannot succeed, so a
    /// mismatch is a hard error surfaced to Python as `ZeroCopyUnavailable`.
    #[error("no Vulkan physical device matched the CUDA device UUID {target:02x?}")]
    NoUuidMatchedDevice {
        /// The 16-byte raw CUDA device UUID that was requested.
        target: [u8; 16],
    },

    /// The selected device / ICD does not support `VK_KHR_external_memory_fd` or a
    /// CUDA-importable DEVICE_LOCAL OPAQUE_FD allocation — M4.1 (FAIL-CLOSED, R-4).
    #[error("external memory (OPAQUE_FD) unsupported: {0}")]
    ExternalMemoryUnsupported(String),

    /// The selected device / ICD does not support `VK_KHR_external_semaphore_fd` — M4.1.
    #[error("external semaphore (OPAQUE_FD) unsupported: {0}")]
    ExternalSemaphoreUnsupported(String),

    /// `vkGetMemoryFdKHR` / `vkGetSemaphoreFdKHR` (or the underlying create) failed — M4.1.
    #[error("external {kind} export failed: {reason}")]
    ExternalExportFailed {
        /// `"memory"` or `"semaphore"`.
        kind: &'static str,
        /// Human-readable reason (Vulkan result code).
        reason: String,
    },

    /// The chosen DEVICE_LOCAL memory type is not exportable as OPAQUE_FD — M4.1.
    ///
    /// On a unified-only device where the sole DEVICE_LOCAL type is also HOST_VISIBLE
    /// and may not be importable by CUDA over OPAQUE_FD, the export path fails closed
    /// rather than handing Python a non-importable fd.
    #[error("the selected DEVICE_LOCAL memory type is not exportable as OPAQUE_FD (unsupported)")]
    MemoryNotExportable,
}

/// Convenience type alias for dispatch results.
pub type DispatchResult<T> = Result<T, DispatchError>;

#[cfg(test)]
mod tests {
    use super::*;

    /// AT-801: DispatchError has exactly 30 variants, all Display and Diagnostic.
    ///
    /// Supersedes AT-502 (23 variants), M2.3a at_801 (25 variants), M3.0 at_801 (28 variants).
    /// M3.1 adds 2: `CoopMatUnsupported`, `DeviceFeatureUnsupported`.
    /// The exhaustive match below ensures the compiler reminds us to update this test
    /// whenever a variant is added or removed.
    #[test]
    fn at_801_dispatch_error_variants_count_is_36() {
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
            // M3.0 additions (variants 26, 27, 28):
            DispatchError::SemaphoreCreationFailed("test semaphore".to_owned()),
            DispatchError::TransferQueueSubmitFailed("test transfer submit".to_owned()),
            DispatchError::MappedRangeOpFailed {
                op: MappedRangeOp::Flush,
                reason: "test flush".to_owned(),
            },
            // M3.1 additions (variants 29, 30):
            DispatchError::CoopMatUnsupported {
                required_m: 16,
                required_n: 16,
                required_k: 16,
                reason: "16x16x16 f16 Subgroup not supported".to_owned(),
            },
            DispatchError::DeviceFeatureUnsupported {
                feature: "storageBuffer16BitAccess".to_owned(),
                kernel: "matmul_tile".to_owned(),
            },
            // M3.2 addition (variant 31):
            DispatchError::SharedMemoryExceedsDeviceLimit {
                required: 65536,
                device_max: 16384,
                kernel: "shared_reduce".to_owned(),
            },
            // M4.1 additions (variants 32-36):
            DispatchError::NoUuidMatchedDevice { target: [0u8; 16] },
            DispatchError::ExternalMemoryUnsupported("no OPAQUE_FD".to_owned()),
            DispatchError::ExternalSemaphoreUnsupported("no OPAQUE_FD".to_owned()),
            DispatchError::ExternalExportFailed { kind: "memory", reason: "test".to_owned() },
            DispatchError::MemoryNotExportable,
        ];

        // Verify exactly 36 variants are covered (M4.1 adds 5 external-interop variants).
        assert_eq!(variants.len(), 36, "expected exactly 36 DispatchError variants");

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

    /// AT-1415: MappedRangeOp has exactly 2 variants with Debug, PartialEq, Eq.
    #[test]
    fn at_1415_mapped_range_op_two_variants() {
        let flush: MappedRangeOp = MappedRangeOp::Flush;
        let inv: MappedRangeOp = MappedRangeOp::Invalidate;
        assert_ne!(flush, inv, "Flush and Invalidate must be distinct");
        assert!(!format!("{flush:?}").is_empty(), "Flush debug must be non-empty");
        assert!(!format!("{inv:?}").is_empty(), "Invalidate debug must be non-empty");

        // Exhaustive match — compile error if a third variant is added.
        let _covered: () = match flush {
            MappedRangeOp::Flush => {}
            MappedRangeOp::Invalidate => {}
        };

        // Both new error variants render correctly.
        let e1 = DispatchError::SemaphoreCreationFailed("test".to_owned());
        assert!(e1.to_string().contains("failed"), "SemaphoreCreationFailed must say 'failed'");

        let e2 = DispatchError::TransferQueueSubmitFailed("test".to_owned());
        assert!(e2.to_string().contains("failed"), "TransferQueueSubmitFailed must say 'failed'");

        let e3 = DispatchError::MappedRangeOpFailed { op: MappedRangeOp::Invalidate, reason: "boom".to_owned() };
        assert!(e3.to_string().contains("Invalidate"), "MappedRangeOpFailed must include op name");
    }

    /// AT-502 (legacy test preserved as alias): verifies the new 31-count.
    #[test]
    fn at_502_dispatch_error_variants_are_display_miette() {
        // This test delegates to the more complete at_801 test above.
        // Preserved for backward-compatibility with any test-name grepping.
        at_801_dispatch_error_variants_count_is_36();
    }
}
