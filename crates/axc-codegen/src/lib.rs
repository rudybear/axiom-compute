//! AXIOM-Compute SPIR-V codegen.
//!
//! Emits Vulkan-flavor SPIR-V 1.3 from a validated HIR module using
//! `rspirv::dr::Builder`. The LLVM SPIR-V backend path is deferred to M2+.

pub mod emit;
pub mod body;
pub mod buffers;
pub mod subgroup;

pub use emit::{emit_module, emit_module_bytes, CodegenError, CodegenOptions};
pub use body::{ScalarTypeCache, CapabilitiesRequired, BodyCodegenError, KernelResources, emit_kernel_body};
pub use buffers::{
    emit_buffer_globals, emit_push_constant_block, emit_gid_variable,
    BufferBindings, PushConstantBlock, GlobalInvocationIdVar,
};
pub use subgroup::{SubgroupBuiltinVars, SubgroupVote, SubgroupReduceOp, SubgroupCodegenError};
