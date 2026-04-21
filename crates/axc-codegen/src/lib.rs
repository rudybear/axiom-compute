//! AXIOM-Compute SPIR-V codegen.
//!
//! Emits Vulkan-flavor SPIR-V 1.3 from a validated HIR module using
//! `rspirv::dr::Builder`. The LLVM SPIR-V backend path is deferred to M2+.

pub mod emit;
pub mod body;
pub mod buffers;
pub mod subgroup;
pub mod coopmat;
pub mod q4_0;

pub use emit::{emit_module, emit_module_bytes, CodegenError, CodegenOptions};
pub use body::{ScalarTypeCache, CapabilitiesRequired, BodyCodegenError, KernelResources, emit_kernel_body};
pub use buffers::{
    emit_buffer_globals, emit_push_constant_block, emit_gid_variable,
    BufferBindings, PushConstantBlock, GlobalInvocationIdVar,
};
pub use subgroup::{SubgroupBuiltinVars, SubgroupVote, SubgroupReduceOp, SubgroupCodegenError};

/// Extract the workgroup dimensions from the first kernel in a HIR module.
///
/// Returns `[1, 1, 1]` if the module contains no kernels (safe fallback).
/// Used by `axc_driver::compile_source_with_meta` to populate `KernelMetadata`.
///
/// # Examples
/// ```no_run
/// // Requires a compiled HirModule.
/// // let dims = axc_codegen::extract_workgroup_dims(&hir_module);
/// // assert_eq!(dims, [64, 1, 1]);
/// ```
pub fn extract_workgroup_dims(module: &axc_hir::HirModule) -> [u32; 3] {
    match module.kernels.first() {
        Some(kernel) => {
            let wg = &kernel.annotations.workgroup;
            [wg.x, wg.y, wg.z]
        }
        None => [1, 1, 1],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// AT-511: extract_workgroup_dims returns the correct dimensions for a real kernel.
    #[test]
    fn at_511_extract_workgroup_dims_saxpy() {
        let src = concat!(
            "@kernel\n",
            "@workgroup(64, 1, 1)\n",
            "@intent(\"test\")\n",
            "@complexity(O(n))\n",
            "fn saxpy(n: u32, alpha: f32, x: readonly_buffer[f32], y: buffer[f32]) -> void {\n",
            "    let i: u32 = gid(0);\n",
            "    return;\n",
            "}\n",
        );
        let (tokens, _) = axc_lexer::tokenize(src);
        let mut parser = axc_parser::Parser::new(&tokens);
        let (ast, _) = parser.parse_module();
        let (hir, errs, _) = axc_hir::lower_module(&ast);
        assert!(errs.is_empty(), "hir errors: {errs:?}");
        assert_eq!(extract_workgroup_dims(&hir), [64, 1, 1]);
    }

    /// AT-511 fallback: extract_workgroup_dims on empty module returns [1, 1, 1].
    #[test]
    fn extract_workgroup_dims_empty_module_fallback() {
        let hir = axc_hir::HirModule { kernels: Vec::new() };
        assert_eq!(extract_workgroup_dims(&hir), [1, 1, 1]);
    }
}
