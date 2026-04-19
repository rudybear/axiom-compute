//! AXIOM-Compute SPIR-V codegen.
//!
//! Emits Vulkan-flavor SPIR-V 1.3 from a validated HIR module using
//! `rspirv::dr::Builder`. The LLVM SPIR-V backend path is deferred to M2+.

pub mod emit;

pub use emit::{emit_module, emit_module_bytes, CodegenError, CodegenOptions};
