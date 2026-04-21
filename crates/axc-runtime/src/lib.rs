//! AXIOM-Compute Vulkan 1.1 runtime dispatcher — M2.3a.
//!
//! This crate provides the real GPU execution layer for AXIOM-Compute kernels.
//!
//! # Pipeline (prepare-once / dispatch-many — M2.3a)
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
//! # Legacy (one-shot) API
//!
//! `VulkanContext::dispatch(DispatchRequest)` is preserved for backward compatibility.
//! It now internally calls `prepare_kernel` + `dispatch_handle` and drops the handle
//! on return, paying pipeline-compile cost on every call.
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
//! # Safety
//!
//! Every `unsafe` block has a `// SAFETY:` comment explaining the Vulkan spec
//! invariants being honored. The crate-level lint below enforces this.

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
pub(crate) mod pipeline_cache;
pub mod kernel_handle;

pub use error::{DispatchError, DispatchResult, CopyDirection};
pub use context::{VulkanContext, VulkanContextOptions};
pub use dispatch::DispatchRequest;
pub use metadata::{KernelMetadata, load_kernel_metadata, CURRENT_SCHEMA_VERSION};
pub use icd::{probe_vulkan_available, gpu_tests_enabled, captured_icd_path};
pub use kernel_handle::{KernelHandle, KernelCacheKey};
