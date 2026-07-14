//! AXIOM-Compute Vulkan 1.2 runtime dispatcher — M3.0.
//!
//! This crate provides the real GPU execution layer for AXIOM-Compute kernels.
//!
//! # Pipeline (prepare-once / dispatch-many — M3.0)
//!
//! ```text
//! .axc source ──► axc_driver::compile_source_with_meta
//!                   ├─► .spv (SPIR-V bytes)
//!                   └─► KernelMetadata (binding plan, workgroup size, entry point)
//!                         │
//!                         ▼
//!               VulkanContext::new_with_options(VulkanContextOptions { ... })
//!                   │
//!                   ├─► prepare_kernel(spirv, plan, pc_bytes, entry_point) ──► KernelHandle
//!                   │                                                              │
//!                   └─► dispatch_handle(&handle, workgroups, inputs, output_sizes, push_constants)
//!                                                                               └─► Vec<Vec<u8>>
//! ```
//!
//! # M3.0 dispatch bandwidth rework
//!
//! Four orthogonal levers:
//! 1. Persistent-mapped staging (drop per-dispatch vkMapMemory/vkUnmapMemory).
//! 2. HOST_CACHED staging + flush/invalidate on non-coherent memory.
//! 3. Dedicated DMA transfer queue (CONCURRENT device-local buffers; no ownership barriers).
//! 4. Timeline-semaphore (primary) / binary-semaphore (fallback) transfer/compute overlap.
//!
//! # Legacy (one-shot) API
//!
//! `VulkanContext::dispatch(DispatchRequest)` is preserved for backward compatibility.
//! It now internally calls `prepare_kernel` + `dispatch_handle` and drops the handle
//! on return, paying pipeline-compile cost on every call. The per-call handle drop
//! unmaps its staging exactly once (consuming `PersistentMapping`), so there is no
//! double-unmap.
//!
//! # GPU test gating
//!
//! Integration tests are `#[ignore]` by default. They run when:
//! - `AXC_ENABLE_GPU_TESTS=1` is set in the environment
//! - `probe_vulkan_available()` returns `true`
//! - `cargo test -- --ignored` is used
//!
//! CI sets `AXC_ENABLE_GPU_TESTS=1` and uses Lavapipe (Mesa software Vulkan)
//! via `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json`.
//!
//! # Force options (M3.0)
//!
//! Three environment variables override auto-detection for CI coverage:
//! - `AXC_FORCE_SINGLE_QUEUE=1` — disable dedicated transfer queue.
//! - `AXC_FORCE_BINARY_SEMAPHORES=1` — use binary semaphores even on 1.2 devices.
//! - `AXC_FORCE_NONCOHERENT_STAGING=1` — emit flush/invalidate on coherent memory.
//!
//! # Safety
//!
//! Every `unsafe` block has a `// SAFETY:` comment explaining the Vulkan spec
//! invariants being honored. The crate-level lint below enforces this.
//!
//! `PersistentMapping` is `unsafe impl Send` ONLY (never `Sync`). See
//! `persistent_map` module for the full safety argument (CRITICAL-6).

#![warn(clippy::undocumented_unsafe_blocks)]

pub mod error;
pub mod context;
pub mod dispatch;
pub mod metadata;
pub mod icd;
pub(crate) mod buffers;
pub(crate) mod pipeline;
pub(crate) mod resources;
pub(crate) mod device_owner;
pub(crate) mod instance_owner;
pub(crate) mod pipeline_cache;
pub mod kernel_handle;
// M3.0 new modules (crate-private)
pub(crate) mod persistent_map;
pub(crate) mod sync;
pub(crate) mod transfer_queue;
// M3.1 new modules
pub mod coopmat;
pub(crate) mod resident;
// M4.1 CUDA↔Vulkan external-memory interop modules
pub mod external_memory;
pub mod device_uuid;

pub use error::{DispatchError, DispatchResult, CopyDirection};
pub use context::{VulkanContext, VulkanContextOptions};
pub use dispatch::{DispatchRequest, DebugDispatchOutcome};
pub use metadata::{KernelMetadata, load_kernel_metadata, CURRENT_SCHEMA_VERSION,
                   CoopMatShapeMeta, CoopMatScalarMeta, CoopMatScopeMeta,
                   DebugChecksMeta, DebugConditionMeta, DebugCheckKind,
                   debug_augmented_binding_plan, decode_debug_flags};
pub use icd::{probe_vulkan_available, gpu_tests_enabled, captured_icd_path};
pub use kernel_handle::{KernelHandle, KernelCacheKey};
pub use coopmat::{
    CoopMatSupport, CoopMatShapeSupport, CoopMatComponentType, CoopMatScope,
    CoopMatRequiredShape, coopmat_shape_supported, coopmat_required_shape_from_meta,
    EnabledDeviceFeatures,
};
pub use resident::{ResidentBuffers, ResidentDispatchTiming, ResidentTimingSource, ResidentBenchConfig};
pub use external_memory::{
    ExternalBuffer, ExternalTimelineSemaphore, ExternalBinarySemaphore, ExternalInteropCaps,
    SharedBufferSet, allocate_exportable_device_local_buffer, export_memory_fd,
    create_exportable_timeline_semaphore, create_exportable_binary_semaphore_pair,
    export_semaphore_fd,
};
pub use context::ExternalInteropOptions;
pub use device_uuid::{select_physical_device_by_uuid, physical_device_uuid, physical_device_driver_uuid};
