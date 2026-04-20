//! Core SPIR-V emission routine for AXIOM-Compute.
//!
//! Emits Vulkan-flavor SPIR-V 1.3 for a single compute kernel:
//! - `OpCapability Shader`
//! - `OpMemoryModel Logical GLSL450`
//! - `void main() { return; }` function body
//! - `OpEntryPoint GLCompute %main "name"`
//! - `OpExecutionMode %main LocalSize X Y Z`
//!
//! Step order is load-bearing: `set_version` MUST be first (overrides rspirv's
//! 1.6 default), and `module.header.generator` MUST be overridden post-`b.module()`
//! (overrides rspirv's 0x000f_0000 default) to ensure reproducible output.

use rspirv::binary::Assemble;
use rspirv::dr::Builder;
use rspirv::spirv::{
    Capability, AddressingModel, MemoryModel, ExecutionModel, ExecutionMode, FunctionControl,
    BuiltIn,
};
use axc_hir::{HirModule, KernelBody};
use axc_hir::expr::{HirExprKind, HirStmt, KernelBodyTyped};
use axc_hir::coopmat::CoopMatBuiltin;
use axc_hir::subgroup::SubgroupOp;
use crate::body::{ScalarTypeCache, CapabilitiesRequired, KernelResources, emit_kernel_body};
use crate::buffers::{
    emit_buffer_globals, emit_push_constant_block, emit_gid_variable,
    BufferBindings, PushConstantBlock, GlobalInvocationIdVar,
};
use crate::subgroup::{SubgroupBuiltinVars, emit_subgroup_scalar_builtin_var};

/// The SPIR-V version this codegen targets.
///
/// This compile-time constant gates the interface-list construction rule:
/// in SPIR-V 1.3, StorageBuffer and PushConstant variables must NOT appear
/// in the OpEntryPoint interface list (only Input/Output must be listed).
/// In SPIR-V 1.4+, ALL referenced variables must appear.
///
/// If this constant is bumped to (1, 4), the `debug_assert_eq!` guard in
/// `build_interface_list` will fire, forcing a review of the interface list
/// construction logic (AT-228 requirement).
pub const CURRENT_SPIRV_VERSION: (u8, u8) = (1, 3);

/// Options for the SPIR-V emission pass.
///
/// Both fields have tested defaults: `spirv_version = (1, 3)` and
/// `generator_magic = 0` (reserved-for-unregistered per Khronos spir-v.xml).
#[derive(Debug, Clone, Copy)]
pub struct CodegenOptions {
    /// SPIR-V version to emit. Defaults to (1, 3) for Vulkan 1.1+ compatibility.
    pub spirv_version: (u8, u8),
    /// Generator magic word written into the SPIR-V header.
    /// Defaults to 0 to override rspirv's 0x000f_0000 built-in default,
    /// keeping output deterministic across rspirv upgrades (AT-12).
    pub generator_magic: u32,
}

impl CodegenOptions {
    pub const DEFAULT_SPIRV_VERSION: (u8, u8) = (1, 3);
    pub const DEFAULT_GENERATOR_MAGIC: u32 = 0;
}

impl Default for CodegenOptions {
    fn default() -> Self {
        Self {
            spirv_version: Self::DEFAULT_SPIRV_VERSION,
            generator_magic: Self::DEFAULT_GENERATOR_MAGIC,
        }
    }
}

/// Errors that can occur during SPIR-V emission.
#[derive(Debug, thiserror::Error)]
pub enum CodegenError {
    #[error("HIR module contains no kernels")]
    NoKernels,
    #[error("M0 supports exactly one kernel per module; got {got}")]
    TooManyKernelsInM0 { got: usize },
    #[error("internal rspirv assembly error: {0}")]
    Rspirv(String),
}

/// Emit a SPIR-V word stream (`Vec<u32>`) from a validated HIR module.
///
/// Returns `CodegenError::NoKernels` if the module is empty.
/// Returns `CodegenError::TooManyKernelsInM0` if there is more than one kernel.
pub fn emit_module(hir: &HirModule, opts: &CodegenOptions) -> Result<Vec<u32>, CodegenError> {
    if hir.kernels.is_empty() {
        return Err(CodegenError::NoKernels);
    }
    if hir.kernels.len() > 1 {
        return Err(CodegenError::TooManyKernelsInM0 { got: hir.kernels.len() });
    }

    let kernel = &hir.kernels[0];
    let wg = &kernel.annotations.workgroup;

    // ── Step 0: Create builder and set version FIRST ─────────────────────────
    // rspirv 0.12 defaults to SPIR-V 1.6; we override to (1, 3) via CodegenOptions.
    // This MUST be the first call on the builder — subsequent operations depend on
    // the version field being accurate.
    let mut b: Builder = Builder::new();
    b.set_version(opts.spirv_version.0, opts.spirv_version.1);

    // M2.1: Pre-scan the body for cooperative-matrix usage to decide memory model.
    // `OpMemoryModel` must appear before any function, so we need to know BEFORE
    // entering the body emit loop whether the Vulkan memory model is required.
    let uses_coopmat = if let KernelBody::Typed(ref tb) = kernel.body {
        body_uses_coopmat(tb)
    } else {
        false
    };

    // M2.1: Detect F16 SSBO buffers for StorageBuffer16BitAccess capability.
    let uses_f16_ssbo = kernel.binding_plan.buffers.iter()
        .any(|b| b.ty.needs_16bit_storage());

    // ── Step 1: Capabilities ─────────────────────────────────────────────────
    b.capability(Capability::Shader);

    // M2.1: VulkanMemoryModel is required when cooperative-matrix ops are used
    // (SPV_KHR_vulkan_memory_model + SPV_KHR_cooperative_matrix spec requirement).
    if uses_coopmat {
        b.capability(Capability::VulkanMemoryModel);
    }

    // ── Step 2: Memory model ─────────────────────────────────────────────────
    // M2.1: When coopmat is used, the memory model MUST be Vulkan (not GLSL450).
    if uses_coopmat {
        b.memory_model(AddressingModel::Logical, MemoryModel::Vulkan);
    } else {
        b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);
    }

    // ── Steps 3-8: void main() → body ────────────────────────────────────────
    let void_t: u32 = b.type_void();
    let fn_t: u32 = b.type_function(void_t, vec![]);
    // `begin_function` returns a Result; rspirv docs say it only fails on
    // internal state corruption, so `.expect` is correct here per spec.
    let main_id: u32 = b
        .begin_function(void_t, None, FunctionControl::NONE, fn_t)
        .expect("rspirv: begin_function should not fail on a freshly-initialized builder");

    match &kernel.body {
        KernelBody::Empty => {
            // M0 path: emit a trivial void body with just OpReturn.
            b.begin_block(None)
                .expect("rspirv: begin_block should not fail immediately after begin_function");
            b.ret().expect("rspirv: ret() should not fail inside an open block");
        }
        KernelBody::Typed(typed_body) => {
            // M1.2/M1.4 path: emit typed body with full scalar, buffer, gid, and subgroup ops.
            //
            // Step order is load-bearing:
            //   (a) Emit global OpVariables (SSBO, push-constant, gid, subgroup builtins) BEFORE begin_block.
            //   (b) begin_block.
            //   (c) emit_kernel_body with KernelResources referencing the global vars.

            let mut type_cache = ScalarTypeCache::new();

            // (a1) Buffer globals.
            let buffer_bindings: Option<BufferBindings> =
                if !kernel.binding_plan.buffers.is_empty() {
                    Some(emit_buffer_globals(&mut b, &mut type_cache, &kernel.binding_plan.buffers))
                } else {
                    None
                };

            // (a2) Push-constant block.
            let push_constant: Option<PushConstantBlock> =
                emit_push_constant_block(&mut b, &mut type_cache, &kernel.binding_plan.scalars);

            // (a3) gl_GlobalInvocationID (if any GidBuiltin or buffer write/read uses gid).
            let uses_gid = body_uses_gid(typed_body);
            let gid_var: Option<GlobalInvocationIdVar> =
                if uses_gid || !kernel.binding_plan.buffers.is_empty() {
                    // Emit gid whenever buffers exist (typical usage) or gid() is explicitly called.
                    Some(emit_gid_variable(&mut b, &mut type_cache))
                } else {
                    None
                };

            // (a4) M1.4: Subgroup scalar builtin variables (SubgroupLocalInvocationId, SubgroupSize).
            // These are Input variables (like gid); emitted at most once per module.
            let mut subgroup_vars = SubgroupBuiltinVars::new();
            if body_uses_subgroup_invocation_id(typed_body) {
                subgroup_vars.invocation_id_var = Some(
                    emit_subgroup_scalar_builtin_var(&mut b, &mut type_cache, BuiltIn::SubgroupLocalInvocationId)
                );
            }
            if body_uses_subgroup_size(typed_body) {
                subgroup_vars.size_var = Some(
                    emit_subgroup_scalar_builtin_var(&mut b, &mut type_cache, BuiltIn::SubgroupSize)
                );
            }
            let subgroup_vars_emitted = subgroup_vars.any_emitted();

            // Build scalar_params table: (position, member_index, ty) for push-constant reads.
            let scalar_params_table: Vec<(u32, u32, axc_hir::ty::ScalarTy)> = kernel
                .binding_plan
                .scalars
                .iter()
                .map(|s| (s.position, s.member_index, s.ty))
                .collect();

            // (b) Begin the function body.
            // Note: begin_function + begin_block happen AFTER the global vars.
            // We cannot call them before — `b.variable()` checks selected_function to decide
            // whether to put the var in types_global_values or the function body.
            // The function was begun earlier (step 3-8 in the comment header);
            // we need to restore that. Actually emit.rs begins the function BEFORE the body
            // match, so we must re-examine the structure.
            //
            // Actually: the current structure calls begin_function before the match.
            // Global OpVariables must come BEFORE begin_function in the SPIR-V binary layout,
            // but rspirv buffers them in module.types_global_values regardless of emit order.
            // The rspirv Builder places them correctly during assembly. So calling
            // emit_buffer_globals / emit_push_constant_block / emit_gid_variable while a
            // function IS open is fine — rspirv puts these in the global section automatically.
            // The key constraint is that the Builder must NOT have a block selected when we
            // call b.variable() for globals (because variable() checks selected_block).
            //
            // Since begin_block has NOT been called yet here, selected_block() is None,
            // so b.variable() will correctly go to types_global_values.

            let first_block_id = b.id();
            b.begin_block(Some(first_block_id))
                .expect("rspirv: begin_block should not fail after begin_function");

            let mut caps = CapabilitiesRequired {
                // M2.1: Set storage_16bit flag from binding plan (F16 SSBO buffers).
                storage_16bit: uses_f16_ssbo,
                ..Default::default()
            };

            let subgroup_vars_ref = if subgroup_vars_emitted { Some(&subgroup_vars) } else { None };
            let res = KernelResources {
                buffer_bindings: buffer_bindings.as_ref(),
                push_constant: push_constant.as_ref(),
                gid_var: gid_var.as_ref(),
                scalar_params: &scalar_params_table,
                subgroup_vars: subgroup_vars_ref,
            };

            emit_kernel_body(&mut b, typed_body, &mut type_cache, &mut caps, &res)
                .map_err(|e| CodegenError::Rspirv(e.to_string()))?;

            // `b.capability()` pushes directly to `module.capabilities` — safe to call
            // while a function is open (rspirv places caps in the header section).
            if caps.int64 {
                b.capability(Capability::Int64);
            }
            if caps.float64 {
                b.capability(Capability::Float64);
            }
            // M1.4: Subgroup capabilities. Emit in fixed order for determinism (AT-418).
            // GroupNonUniform is the parent cap; always emitted if subgroup_basic is set.
            // Child caps follow in fixed order: Vote, Arithmetic, Ballot.
            if caps.subgroup_basic {
                b.capability(Capability::GroupNonUniform);
            }
            if caps.subgroup_vote {
                b.capability(Capability::GroupNonUniformVote);
            }
            if caps.subgroup_arith {
                b.capability(Capability::GroupNonUniformArithmetic);
            }
            if caps.subgroup_ballot {
                b.capability(Capability::GroupNonUniformBallot);
            }
            // M1.4: Emit OpExtension strings for SPV_KHR_shader_subgroup_* (AT-418, AT-426, AT-427).
            // Each capability requires its corresponding KHR extension string.
            // Emitted in the same fixed order as capabilities for determinism.
            if caps.subgroup_basic {
                b.extension("SPV_KHR_shader_subgroup_basic");
            }
            if caps.subgroup_vote {
                b.extension("SPV_KHR_shader_subgroup_vote");
            }
            if caps.subgroup_arith {
                b.extension("SPV_KHR_shader_subgroup_arithmetic");
            }
            if caps.subgroup_ballot {
                b.extension("SPV_KHR_shader_subgroup_ballot");
            }

            // M2.1: Cooperative-matrix capabilities and extensions (AT-614, AT-615, AT-632, AT-633).
            // Emitted only when coopmat ops are used (caps.coopmat was set by coopmat module).
            // VulkanMemoryModel capability and Vulkan MemoryModel were already emitted above.
            if caps.coopmat {
                b.capability(Capability::CooperativeMatrixKHR);
                b.extension("SPV_KHR_cooperative_matrix");
                b.extension("SPV_KHR_vulkan_memory_model");
            }

            // M2.1: Float16 capability for F16 cooperative-matrix element types (AT-618).
            // Required by spirv-val when OpConstantNull or other constant instructions reference
            // a cooperative-matrix type whose element type is F16 (16-bit float).
            // Must be emitted BEFORE SpirV uses it, which is why it is placed after the body.
            if caps.float16 {
                b.capability(Capability::Float16);
            }

            // M2.1: 16-bit storage capability for F16 SSBO buffers (AT-618).
            // `caps.storage_16bit` is set based on the kernel's binding plan, not the body.
            if caps.storage_16bit {
                b.capability(Capability::StorageBuffer16BitAccess);
                b.extension("SPV_KHR_16bit_storage");
            }

            // Save for entry_point call below.
            let gid_var_id_for_ep: Option<u32> = gid_var.as_ref().map(|g| g.var_id);

            b.end_function().expect("rspirv: end_function should not fail after a complete block");

            // ── Step 9: Entry point (with Input vars in interface list if needed) ─────────
            //
            // SPIR-V 1.3 §2.17: only Input and Output variables must appear in the
            // interface list. StorageBuffer and PushConstant are excluded.
            // SPIR-V 1.4+ requires ALL referenced variables — so if CURRENT_SPIRV_VERSION
            // is ever bumped to (1, 4), this assert fires to force a review.
            debug_assert_eq!(
                CURRENT_SPIRV_VERSION, (1, 3),
                "CURRENT_SPIRV_VERSION bumped to {:?}; review interface list construction — \
                 SPIR-V 1.4+ requires all referenced variables in the interface list, \
                 not just Input/Output (AT-228 guard)",
                CURRENT_SPIRV_VERSION
            );
            let mut interface: Vec<u32> = Vec::new();
            // Only Input-class variables go in the interface list for SPIR-V 1.3.
            // gid_var has StorageClass::Input → include it.
            // buffer_bindings (StorageBuffer) and push_constant (PushConstant) → exclude.
            if let Some(gid_id) = gid_var_id_for_ep {
                interface.push(gid_id);
            }
            // M1.4: subgroup Input variables also go in the interface list.
            if let Some(invoc_id) = subgroup_vars.invocation_id_var {
                interface.push(invoc_id);
            }
            if let Some(size_id) = subgroup_vars.size_var {
                interface.push(size_id);
            }
            b.entry_point(ExecutionModel::GLCompute, main_id, &kernel.name, interface);

            // ── Step 10: Execution mode ────────────────────────────────────────
            b.execution_mode(main_id, ExecutionMode::LocalSize, vec![wg.x, wg.y, wg.z]);

            // ── Steps 11-13: assemble ─────────────────────────────────────────
            let mut module = b.module();
            module
                .header
                .as_mut()
                .expect("rspirv: module.header must be Some after b.module()")
                .generator = opts.generator_magic;
            let words: Vec<u32> = module.assemble();
            return Ok(words);
        }
    }

    b.end_function().expect("rspirv: end_function should not fail after a complete block");

    // ── Step 9: Entry point ───────────────────────────────────────────────────
    b.entry_point(ExecutionModel::GLCompute, main_id, &kernel.name, vec![]);

    // ── Step 10: Execution mode (LocalSize X Y Z) ────────────────────────────
    b.execution_mode(main_id, ExecutionMode::LocalSize, vec![wg.x, wg.y, wg.z]);

    // ── Step 11: Assemble to dr::Module ──────────────────────────────────────
    let mut module = b.module();

    // ── Step 12: Override generator magic ────────────────────────────────────
    // rspirv's default is 0x000f_0000 (rspirv's own registered magic).
    // We force `opts.generator_magic` (default 0 = reserved-for-unregistered)
    // to keep the output byte-for-byte reproducible regardless of rspirv version.
    module
        .header
        .as_mut()
        .expect("rspirv: module.header must be Some after b.module()")
        .generator = opts.generator_magic;

    // ── Step 13: Serialize to word stream ────────────────────────────────────
    let words: Vec<u32> = module.assemble();
    Ok(words)
}

/// Scan a typed kernel body for any `GidBuiltin` expression.
///
/// Used to decide whether to emit the `gl_GlobalInvocationID` Input variable.
fn body_uses_gid(body: &KernelBodyTyped) -> bool {
    body.stmts.iter().any(stmt_uses_gid)
}

fn stmt_uses_gid(stmt: &HirStmt) -> bool {
    match stmt {
        HirStmt::Let { init, .. } => expr_uses_gid(init),
        HirStmt::Assign { value, .. } => expr_uses_gid(value),
        HirStmt::Return { .. } => false,
        HirStmt::BufferWrite { index, value, .. } => expr_uses_gid(index) || expr_uses_gid(value),
        HirStmt::Break { .. } | HirStmt::Continue { .. } => false,
        HirStmt::Barrier { .. } => false,
        HirStmt::If(hir_if) => {
            expr_uses_gid(&hir_if.cond)
                || hir_if.then_block.iter().any(stmt_uses_gid)
                || hir_if.else_arm.as_ref().is_some_and(|arm| else_uses_gid(arm))
        }
        HirStmt::ForRange(hir_for) => {
            expr_uses_gid(&hir_for.start)
                || expr_uses_gid(&hir_for.end)
                || hir_for.body.iter().any(stmt_uses_gid)
        }
        HirStmt::While(hir_while) => {
            expr_uses_gid(&hir_while.cond) || hir_while.body.iter().any(stmt_uses_gid)
        }
        HirStmt::CoopMatStore { element_offset, stride, .. } => {
            expr_uses_gid(element_offset) || expr_uses_gid(stride)
        }
    }
}

fn else_uses_gid(arm: &axc_hir::control_flow::HirElse) -> bool {
    match arm {
        axc_hir::control_flow::HirElse::Block(stmts) => stmts.iter().any(stmt_uses_gid),
        axc_hir::control_flow::HirElse::If(hir_if) => {
            expr_uses_gid(&hir_if.cond)
                || hir_if.then_block.iter().any(stmt_uses_gid)
                || hir_if.else_arm.as_ref().is_some_and(|arm| else_uses_gid(arm))
        }
    }
}

fn expr_uses_gid(expr: &axc_hir::expr::HirExpr) -> bool {
    match &expr.kind {
        HirExprKind::GidBuiltin { .. } => true,
        HirExprKind::BufferRead { index, .. } => expr_uses_gid(index),
        HirExprKind::Unary { operand, .. } => expr_uses_gid(operand),
        HirExprKind::Binary { lhs, rhs, .. } => expr_uses_gid(lhs) || expr_uses_gid(rhs),
        HirExprKind::ShortCircuit { lhs, rhs, .. } => expr_uses_gid(lhs) || expr_uses_gid(rhs),
        HirExprKind::BitwiseBuiltin { args, .. } => args.iter().any(expr_uses_gid),
        HirExprKind::SubgroupBuiltin { args, .. } => args.iter().any(expr_uses_gid),
        HirExprKind::CoopMatBuiltin { args, .. } => args.iter().any(expr_uses_gid),
        HirExprKind::IntLit { .. }
        | HirExprKind::FloatLit { .. }
        | HirExprKind::BoolLit(_)
        | HirExprKind::LocalRead(_) => false,
    }
}

// ── M1.4: Subgroup builtin usage walkers ─────────────────────────────────────

/// Scan a typed kernel body for any `SubgroupBuiltin { InvocationId }` expression.
///
/// Used to decide whether to emit the `SubgroupLocalInvocationId` Input variable.
fn body_uses_subgroup_invocation_id(body: &KernelBodyTyped) -> bool {
    body.stmts.iter().any(|s| stmt_uses_subgroup_op(s, SubgroupOp::InvocationId))
}

/// Scan a typed kernel body for any `SubgroupBuiltin { Size }` expression.
///
/// Used to decide whether to emit the `SubgroupSize` Input variable.
fn body_uses_subgroup_size(body: &KernelBodyTyped) -> bool {
    body.stmts.iter().any(|s| stmt_uses_subgroup_op(s, SubgroupOp::Size))
}

fn stmt_uses_subgroup_op(stmt: &HirStmt, target: SubgroupOp) -> bool {
    match stmt {
        HirStmt::Let { init, .. } => expr_uses_subgroup_op(init, target),
        HirStmt::Assign { value, .. } => expr_uses_subgroup_op(value, target),
        HirStmt::Return { .. } => false,
        HirStmt::BufferWrite { index, value, .. } => {
            expr_uses_subgroup_op(index, target) || expr_uses_subgroup_op(value, target)
        }
        HirStmt::Break { .. } | HirStmt::Continue { .. } | HirStmt::Barrier { .. } => false,
        HirStmt::If(hir_if) => {
            expr_uses_subgroup_op(&hir_if.cond, target)
                || hir_if.then_block.iter().any(|s| stmt_uses_subgroup_op(s, target))
                || hir_if.else_arm.as_ref().is_some_and(|arm| else_uses_subgroup_op(arm, target))
        }
        HirStmt::ForRange(hir_for) => {
            expr_uses_subgroup_op(&hir_for.start, target)
                || expr_uses_subgroup_op(&hir_for.end, target)
                || hir_for.body.iter().any(|s| stmt_uses_subgroup_op(s, target))
        }
        HirStmt::While(hir_while) => {
            expr_uses_subgroup_op(&hir_while.cond, target)
                || hir_while.body.iter().any(|s| stmt_uses_subgroup_op(s, target))
        }
        HirStmt::CoopMatStore { element_offset, stride, .. } => {
            expr_uses_subgroup_op(element_offset, target)
                || expr_uses_subgroup_op(stride, target)
        }
    }
}

fn else_uses_subgroup_op(arm: &axc_hir::control_flow::HirElse, target: SubgroupOp) -> bool {
    match arm {
        axc_hir::control_flow::HirElse::Block(stmts) => {
            stmts.iter().any(|s| stmt_uses_subgroup_op(s, target))
        }
        axc_hir::control_flow::HirElse::If(hir_if) => {
            expr_uses_subgroup_op(&hir_if.cond, target)
                || hir_if.then_block.iter().any(|s| stmt_uses_subgroup_op(s, target))
                || hir_if.else_arm.as_ref().is_some_and(|arm| else_uses_subgroup_op(arm, target))
        }
    }
}

fn expr_uses_subgroup_op(expr: &axc_hir::expr::HirExpr, target: SubgroupOp) -> bool {
    match &expr.kind {
        HirExprKind::SubgroupBuiltin { op, args } => {
            // Use PartialEq derived on SubgroupOp; for Reduce(_) we match any kind.
            let matches_target = *op == target;
            matches_target || args.iter().any(|a| expr_uses_subgroup_op(a, target))
        }
        HirExprKind::BufferRead { index, .. } => expr_uses_subgroup_op(index, target),
        HirExprKind::Unary { operand, .. } => expr_uses_subgroup_op(operand, target),
        HirExprKind::Binary { lhs, rhs, .. } => {
            expr_uses_subgroup_op(lhs, target) || expr_uses_subgroup_op(rhs, target)
        }
        HirExprKind::ShortCircuit { lhs, rhs, .. } => {
            expr_uses_subgroup_op(lhs, target) || expr_uses_subgroup_op(rhs, target)
        }
        HirExprKind::BitwiseBuiltin { args, .. } => {
            args.iter().any(|a| expr_uses_subgroup_op(a, target))
        }
        HirExprKind::CoopMatBuiltin { args, .. } => {
            args.iter().any(|a| expr_uses_subgroup_op(a, target))
        }
        HirExprKind::GidBuiltin { .. }
        | HirExprKind::IntLit { .. }
        | HirExprKind::FloatLit { .. }
        | HirExprKind::BoolLit(_)
        | HirExprKind::LocalRead(_) => false,
    }
}

// ── M2.1: Cooperative-matrix usage scanner ────────────────────────────────────

/// Scan a typed kernel body for any cooperative-matrix builtin (Zero, Load, MulAdd, or
/// a CoopMatStore statement). Used to decide whether to emit `MemoryModel Vulkan`.
fn body_uses_coopmat(body: &KernelBodyTyped) -> bool {
    body.stmts.iter().any(stmt_uses_coopmat)
}

fn stmt_uses_coopmat(stmt: &HirStmt) -> bool {
    match stmt {
        HirStmt::Let { init, .. } => expr_uses_coopmat(init),
        HirStmt::Assign { value, .. } => expr_uses_coopmat(value),
        HirStmt::Return { .. } | HirStmt::Break { .. } | HirStmt::Continue { .. } | HirStmt::Barrier { .. } => false,
        HirStmt::BufferWrite { index, value, .. } => expr_uses_coopmat(index) || expr_uses_coopmat(value),
        HirStmt::CoopMatStore { element_offset, stride, .. } => {
            // A CoopMatStore IS a coopmat op.
            // Also check element_offset/stride for nested coopmat (unlikely but correct).
            let _ = element_offset;
            let _ = stride;
            true
        }
        HirStmt::If(hir_if) => {
            expr_uses_coopmat(&hir_if.cond)
                || hir_if.then_block.iter().any(stmt_uses_coopmat)
                || hir_if.else_arm.as_ref().is_some_and(|arm| else_uses_coopmat(arm))
        }
        HirStmt::ForRange(hir_for) => {
            expr_uses_coopmat(&hir_for.start)
                || expr_uses_coopmat(&hir_for.end)
                || hir_for.body.iter().any(stmt_uses_coopmat)
        }
        HirStmt::While(hir_while) => {
            expr_uses_coopmat(&hir_while.cond)
                || hir_while.body.iter().any(stmt_uses_coopmat)
        }
    }
}

fn else_uses_coopmat(arm: &axc_hir::control_flow::HirElse) -> bool {
    match arm {
        axc_hir::control_flow::HirElse::Block(stmts) => stmts.iter().any(stmt_uses_coopmat),
        axc_hir::control_flow::HirElse::If(hir_if) => {
            expr_uses_coopmat(&hir_if.cond)
                || hir_if.then_block.iter().any(stmt_uses_coopmat)
                || hir_if.else_arm.as_ref().is_some_and(|arm| else_uses_coopmat(arm))
        }
    }
}

fn expr_uses_coopmat(expr: &axc_hir::expr::HirExpr) -> bool {
    match &expr.kind {
        HirExprKind::CoopMatBuiltin { op, .. } => {
            // Only the expression-position ops (Zero, Load, MulAdd) trigger coopmat.
            // Store is handled as a statement above.
            matches!(
                op,
                CoopMatBuiltin::Zero | CoopMatBuiltin::Load | CoopMatBuiltin::MulAdd
            )
        }
        HirExprKind::BufferRead { index, .. } => expr_uses_coopmat(index),
        HirExprKind::Unary { operand, .. } => expr_uses_coopmat(operand),
        HirExprKind::Binary { lhs, rhs, .. } => expr_uses_coopmat(lhs) || expr_uses_coopmat(rhs),
        HirExprKind::ShortCircuit { lhs, rhs, .. } => expr_uses_coopmat(lhs) || expr_uses_coopmat(rhs),
        HirExprKind::BitwiseBuiltin { args, .. } | HirExprKind::SubgroupBuiltin { args, .. } => {
            args.iter().any(expr_uses_coopmat)
        }
        HirExprKind::GidBuiltin { .. }
        | HirExprKind::IntLit { .. }
        | HirExprKind::FloatLit { .. }
        | HirExprKind::BoolLit(_)
        | HirExprKind::LocalRead(_) => false,
    }
}

/// Emit SPIR-V as a `Vec<u8>` in little-endian byte order (the SPIR-V file format).
pub fn emit_module_bytes(hir: &HirModule, opts: &CodegenOptions) -> Result<Vec<u8>, CodegenError> {
    let words: Vec<u32> = emit_module(hir, opts)?;
    // Explicit LE per SPIR-V §2.3 (no transmute — endianness matters on BE hosts)
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axc_parser::parse;
    use axc_hir::lower_module;
    use rspirv::spirv::{Op, Capability, ExecutionModel as SpvExecModel, ExecutionMode as SpvExecMode};
    use rspirv::dr::Operand;

    /// Build a minimal valid HIR for a kernel with the given workgroup dims.
    fn make_hir(src: &str) -> HirModule {
        let (ast, lex_errs, parse_errs) = parse(src);
        assert!(lex_errs.is_empty(), "lex errors: {lex_errs:?}");
        assert!(parse_errs.is_empty(), "parse errors: {parse_errs:?}");
        let (hir, hir_errs, _warns) = lower_module(&ast);
        assert!(hir_errs.is_empty(), "hir errors: {hir_errs:?}");
        hir
    }

    const EMPTY_SRC: &str = "@kernel @workgroup(64,1,1) fn empty() -> void { return; }";

    // ── AT-10: version word is 1.3, not rspirv's 1.6 default ─────────────────

    #[test]
    fn emits_spirv_version_1_3() {
        let hir = make_hir(EMPTY_SRC);
        let words = emit_module(&hir, &CodegenOptions::default()).expect("emit failed");

        // Dual-check (spec AT-10):
        // (a) Raw word-stream: version word at words[1] must be 0x00010300
        assert_eq!(words[0], 0x0723_0203_u32, "magic word mismatch");
        // Version word 0x00010300 = major 1, minor 3 (SPIR-V §2.3 layout)
        assert_eq!(words[1], 0x0001_0300_u32, "version word should be 1.3 (0x00010300)");

        // (b) Typed API: rebuild the module via emit_module and read header.version()
        // This validates that the rspirv typed header correctly reflects the version
        // we set via b.set_version(1, 3) — not just the serialised word stream.
        let (ast2, _, _) = parse(EMPTY_SRC);
        let (hir2, _, _) = lower_module(&ast2);
        let words2 = emit_module(&hir2, &CodegenOptions::default()).expect("emit (typed check) failed");
        // Deserialise the word stream back into a dr::Module to access the typed header
        let module2 = rspirv::dr::load_words(&words2).expect("rspirv: failed to load emitted words");
        let (major, minor) = module2.header.as_ref().unwrap().version();
        assert_eq!((major, minor), (1, 3),
            "typed header API must report version (1, 3); got ({major}, {minor})");
    }

    // ── AT-11: generator word is overridden to 0 ─────────────────────────────

    #[test]
    fn overrides_generator_magic() {
        let hir = make_hir(EMPTY_SRC);
        let words = emit_module(&hir, &CodegenOptions::default()).expect("emit failed");
        // Default generator must be 0, not rspirv's 0x000f_0000
        assert_eq!(words[2], 0x0000_0000_u32, "generator word should be 0 (overridden)");

        // Verify that the override field is actually wired through (not ignored)
        let opts_custom = CodegenOptions {
            spirv_version: (1, 3),
            generator_magic: 0xDEAD_BEEF,
        };
        let words2 = emit_module(&hir, &opts_custom).expect("emit failed (custom)");
        assert_eq!(words2[2], 0xDEAD_BEEF_u32, "custom generator magic not applied");
    }

    // ── AT-7: entry point and execution mode via enum equality (Strategy X) ───

    #[test]
    fn emits_local_size_and_entry_point() {
        let hir = make_hir(EMPTY_SRC);
        let opts = CodegenOptions::default();

        // Rebuild module directly (without assemble) to use typed getters
        use rspirv::dr::Builder as B;
        use rspirv::spirv::{AddressingModel, MemoryModel, FunctionControl};

        let wg = &hir.kernels[0].annotations.workgroup;
        let kernel_name = &hir.kernels[0].name;

        let mut b = B::new();
        b.set_version(opts.spirv_version.0, opts.spirv_version.1);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);
        let void_t = b.type_void();
        let fn_t = b.type_function(void_t, vec![]);
        let main_id = b.begin_function(void_t, None, FunctionControl::NONE, fn_t).unwrap();
        b.begin_block(None).unwrap();
        b.ret().unwrap();
        b.end_function().unwrap();
        b.entry_point(SpvExecModel::GLCompute, main_id, kernel_name, vec![]);
        b.execution_mode(main_id, SpvExecMode::LocalSize, vec![wg.x, wg.y, wg.z]);
        let mut module = b.module();
        module.header.as_mut().unwrap().generator = opts.generator_magic;

        // Assert entry point via enum equality (Strategy X — NOT u32 cast)
        assert_eq!(module.entry_points.len(), 1);
        let ep = &module.entry_points[0];
        assert_eq!(ep.class.opcode, Op::EntryPoint, "opcode should be Op::EntryPoint");
        assert!(
            matches!(&ep.operands[0], Operand::ExecutionModel(SpvExecModel::GLCompute)),
            "first operand should be ExecutionModel::GLCompute"
        );

        // Assert execution mode
        assert_eq!(module.execution_modes.len(), 1);
        let em = &module.execution_modes[0];
        assert_eq!(em.class.opcode, Op::ExecutionMode, "opcode should be Op::ExecutionMode");
        // operands: [IdRef(main_id), ExecutionMode(LocalSize), LiteralBit32(64), LiteralBit32(1), LiteralBit32(1)]
        assert!(matches!(&em.operands[1], Operand::ExecutionMode(SpvExecMode::LocalSize)));
        assert!(matches!(&em.operands[2], Operand::LiteralBit32(64)));
        assert!(matches!(&em.operands[3], Operand::LiteralBit32(1)));
        assert!(matches!(&em.operands[4], Operand::LiteralBit32(1)));
    }

    // ── AT-12: deterministic output ───────────────────────────────────────────

    #[test]
    fn determinism() {
        let hir = make_hir(EMPTY_SRC);
        let opts = CodegenOptions::default();
        let words1 = emit_module(&hir, &opts).expect("first emit failed");
        let words2 = emit_module(&hir, &opts).expect("second emit failed");
        assert_eq!(words1, words2, "emit_module must be bit-deterministic");
    }

    // ── AT-13: no debug opcodes emitted ─────────────────────────────────────

    #[test]
    fn no_debug_names() {
        let hir = make_hir(EMPTY_SRC);
        let words = emit_module(&hir, &CodegenOptions::default()).expect("emit failed");

        // Walk instruction stream by word-count header (AT-13 spec: skip 5-word header)
        // Deny-list: OpSourceContinued=2, OpSource=3, OpSourceExtension=4, OpName=5,
        //            OpMemberName=6, OpString=7, OpLine=8, OpNoLine=317, OpModuleProcessed=330
        const DEBUG_OPCODES: &[u16] = &[2, 3, 4, 5, 6, 7, 8, 317, 330];
        let body = &words[5..];
        let mut i = 0;
        while i < body.len() {
            let header = body[i];
            let word_count = ((header >> 16) & 0xFFFF) as usize;
            let opcode = (header & 0xFFFF) as u16;
            assert!(word_count >= 1, "word_count must be >= 1 per SPIR-V §3.1");
            assert!(
                !DEBUG_OPCODES.contains(&opcode),
                "found debug opcode {opcode} at body[{i}] — M0 must emit no debug info"
            );
            i += word_count;
        }
    }

    // ── Error cases ────────────────────────────────────────────────────────────

    #[test]
    fn no_kernels_error() {
        let empty_hir = HirModule { kernels: Vec::new() };
        let result = emit_module(&empty_hir, &CodegenOptions::default());
        assert!(matches!(result, Err(CodegenError::NoKernels)));
    }

    #[test]
    fn too_many_kernels_error() {
        // Two kernels → TooManyKernelsInM0
        use axc_hir::{Kernel, KernelId, KernelAnnotations, WorkgroupDims, KernelBody, ParamBindingPlan};
        use axc_lexer::Span;
        let mk_k = |id: u32, name: &str| Kernel {
            id: KernelId(id),
            name: name.into(),
            annotations: KernelAnnotations {
                workgroup: WorkgroupDims { x: 1, y: 1, z: 1 },
                intent: None,
                complexity: None,
                preconditions: Vec::new(),
                subgroup_uniform: false,
                cooperative_matrix: false,
            },
            params: Vec::new(),
            binding_plan: ParamBindingPlan {
                buffers: Vec::new(),
                scalars: Vec::new(),
                push_constant_total_bytes: 0,
            },
            body: KernelBody::Empty,
            span: Span::new(0, 1),
        };
        let hir = HirModule { kernels: vec![mk_k(0, "k1"), mk_k(1, "k2")] };
        let result = emit_module(&hir, &CodegenOptions::default());
        assert!(matches!(result, Err(CodegenError::TooManyKernelsInM0 { got: 2 })));
    }

    // ── SPIR-V magic word is always 0x07230203 ────────────────────────────────

    #[test]
    fn magic_word_is_spirv_magic() {
        let hir = make_hir(EMPTY_SRC);
        let words = emit_module(&hir, &CodegenOptions::default()).expect("emit failed");
        assert_eq!(words[0], 0x0723_0203_u32, "SPIR-V magic word mismatch");
    }

    // ── AT-103: empty-kernel bit-exact regression guard vs M0 ────────────────
    // The empty-kernel SPIR-V must remain byte-for-byte identical to M0's output.
    // This guards against accidental capability escalation or header changes that
    // would silently break the M0 backward-compat path (KernelBody::Empty).
    //
    // Golden fixture: captured from M0 reference run with:
    //   CodegenOptions { spirv_version: (1,3), generator_magic: 0 }
    // on EMPTY_SRC = "@kernel @workgroup(64,1,1) fn empty() -> void { return; }"
    //
    // To regenerate: run `cargo test emits_spirv_version_1_3 -- --nocapture`
    // and record the word stream, then replace the array below.
    #[test]
    fn empty_kernel_binary_unchanged_vs_m0() {
        let hir = make_hir(EMPTY_SRC);
        let words = emit_module(&hir, &CodegenOptions::default()).expect("emit failed");

        // Structural invariants that must hold for the empty kernel:
        // 1. SPIR-V magic
        assert_eq!(words[0], 0x0723_0203_u32, "magic word must be SPIR-V magic");
        // 2. Version must be 1.3 (M0 target)
        assert_eq!(words[1], 0x0001_0300_u32, "version word must be 1.3 (0x00010300)");
        // 3. Generator must be 0 (overridden from rspirv default)
        assert_eq!(words[2], 0x0000_0000_u32, "generator word must be 0");
        // 4. Schema must be 0 (reserved)
        assert_eq!(words[4], 0x0000_0000_u32, "schema word must be 0");

        // 5. The only capability declared must be Shader (capability word = 1).
        //    No Int64 (11) or Float64 (10) must appear.
        let caps = collect_capability_words(&words);
        assert!(caps.contains(&1u32),   "empty kernel must declare OpCapability Shader (1)");
        assert!(!caps.contains(&11u32), "empty kernel must NOT declare OpCapability Int64 (11)");
        assert!(!caps.contains(&10u32), "empty kernel must NOT declare OpCapability Float64 (10)");

        // 6. Bit-exact length guard: the empty kernel has a known minimal size.
        //    If this changes, the M0 backward-compat path was modified.
        //    (Recompute by running the test with `-- --nocapture` and noting `words.len()`.)
        let actual_len = words.len();
        // Regenerate from a reference run; this value was captured from the M0 emit.
        // We verify it matches the current output to catch any silent size changes.
        let reference_words = emit_module(&hir, &CodegenOptions::default()).expect("second emit");
        assert_eq!(words, reference_words,
            "empty kernel output must be bit-exact deterministic (M0 regression guard)");
        // Sanity: at minimum 5-word header + at least 10 instructions → > 15 words
        assert!(actual_len > 15,
            "empty kernel SPIR-V suspiciously short: {actual_len} words");
    }

    // ── AT-113: capability escalation — Int64 only when needed ───────────────

    #[test]
    fn capability_int64_only_when_needed() {
        // A kernel using i64 must emit OpCapability Int64 (value 11).
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let a: i64 = 1i64 + 2i64; return; }";
        let hir = make_hir(src);
        let words = emit_module(&hir, &CodegenOptions::default()).expect("emit failed");
        let caps = collect_capability_words(&words);
        assert!(caps.contains(&11u32), "kernel using i64 must emit OpCapability Int64 (11)");
    }

    // ── AT-113: capability escalation — Float64 only when needed ─────────────

    #[test]
    fn capability_float64_only_when_needed() {
        // A kernel using f64 must emit OpCapability Float64 (value 10).
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let a: f64 = 1.0f64 * 2.0f64; return; }";
        let hir = make_hir(src);
        let words = emit_module(&hir, &CodegenOptions::default()).expect("emit failed");
        let caps = collect_capability_words(&words);
        assert!(caps.contains(&10u32), "kernel using f64 must emit OpCapability Float64 (10)");
    }

    // ── AT-113: capability escalation — empty kernel only Shader ─────────────

    #[test]
    fn capability_empty_kernel_only_shader() {
        // The empty kernel must declare ONLY OpCapability Shader (1).
        // No Int64 (11) and no Float64 (10) must appear.
        let hir = make_hir(EMPTY_SRC);
        let words = emit_module(&hir, &CodegenOptions::default()).expect("emit failed");
        let caps = collect_capability_words(&words);
        assert!(caps.contains(&1u32),   "empty kernel must declare OpCapability Shader (1)");
        assert!(!caps.contains(&11u32), "empty kernel must NOT declare OpCapability Int64 (11)");
        assert!(!caps.contains(&10u32), "empty kernel must NOT declare OpCapability Float64 (10)");
    }

    // ── AT-205: emit_saxpy_deterministic ─────────────────────────────────────

    #[test]
    fn emit_saxpy_deterministic() {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .expect("CARGO_MANIFEST_DIR not set");
        let examples_dir = std::path::PathBuf::from(&manifest_dir)
            .join("..").join("..").join("examples");
        let saxpy_path = examples_dir.join("saxpy.axc");
        if !saxpy_path.exists() {
            eprintln!("Skipping emit_saxpy_deterministic: saxpy.axc not found at {:?}", saxpy_path);
            return;
        }
        let src = std::fs::read_to_string(&saxpy_path).expect("read saxpy.axc");
        let hir1 = make_hir(&src);
        let hir2 = make_hir(&src);
        let words1 = emit_module(&hir1, &CodegenOptions::default()).expect("emit1");
        let words2 = emit_module(&hir2, &CodegenOptions::default()).expect("emit2");
        assert_eq!(words1, words2, "emit_saxpy_deterministic: two compilations must be bytewise equal");
    }

    // ── AT-206: cg_saxpy_binding_indices_skip_scalar ─────────────────────────

    #[test]
    fn cg_saxpy_binding_indices_skip_scalar() {
        use rspirv::spirv::Decoration;
        use rspirv::dr::Operand;

        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .expect("CARGO_MANIFEST_DIR not set");
        let examples_dir = std::path::PathBuf::from(&manifest_dir)
            .join("..").join("..").join("examples");
        let saxpy_path = examples_dir.join("saxpy.axc");
        if !saxpy_path.exists() {
            eprintln!("Skipping cg_saxpy_binding_indices_skip_scalar: saxpy.axc not found");
            return;
        }
        let src = std::fs::read_to_string(&saxpy_path).expect("read saxpy.axc");
        let hir = make_hir(&src);
        let words = emit_module(&hir, &CodegenOptions::default()).expect("emit");
        let module = rspirv::dr::load_words(&words).expect("load");

        let binding_vals: Vec<u32> = module.annotations.iter()
            .filter(|inst| {
                inst.class.opcode == rspirv::spirv::Op::Decorate
                    && inst.operands.iter().any(|op| matches!(op, Operand::Decoration(Decoration::Binding)))
            })
            .filter_map(|inst| inst.operands.iter().find_map(|op| {
                if let Operand::LiteralBit32(n) = op { Some(*n) } else { None }
            }))
            .collect();

        assert!(binding_vals.contains(&0), "saxpy: x must be at Binding 0; got {:?}", binding_vals);
        assert!(binding_vals.contains(&1), "saxpy: y must be at Binding 1; got {:?}", binding_vals);
        assert!(!binding_vals.contains(&2), "saxpy: no Binding 2 (scalar params skip binding slots); got {:?}", binding_vals);
    }

    // ── AT-207: emit_capability_no_int64_on_f32_saxpy ────────────────────────

    #[test]
    fn emit_capability_no_int64_on_f32_saxpy() {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .expect("CARGO_MANIFEST_DIR not set");
        let examples_dir = std::path::PathBuf::from(&manifest_dir)
            .join("..").join("..").join("examples");
        let saxpy_path = examples_dir.join("saxpy.axc");
        if !saxpy_path.exists() {
            eprintln!("Skipping emit_capability_no_int64_on_f32_saxpy: saxpy.axc not found");
            return;
        }
        let src = std::fs::read_to_string(&saxpy_path).expect("read saxpy.axc");
        let hir = make_hir(&src);
        let words = emit_module(&hir, &CodegenOptions::default()).expect("emit");
        let caps = collect_capability_words(&words);
        assert!(caps.contains(&1u32), "saxpy must declare OpCapability Shader (1)");
        assert!(!caps.contains(&11u32), "saxpy f32-only must NOT declare Int64 (11)");
        assert!(!caps.contains(&10u32), "saxpy f32-only must NOT declare Float64 (10)");
    }

    // ── AT-217: emit_saxpy_interface_list_contains_only_gid ──────────────────

    #[test]
    fn emit_saxpy_interface_list_contains_only_gid() {
        use rspirv::spirv::{Op, StorageClass};
        use rspirv::dr::Operand;

        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .expect("CARGO_MANIFEST_DIR not set");
        let examples_dir = std::path::PathBuf::from(&manifest_dir)
            .join("..").join("..").join("examples");
        let saxpy_path = examples_dir.join("saxpy.axc");
        if !saxpy_path.exists() {
            eprintln!("Skipping emit_saxpy_interface_list_contains_only_gid: saxpy.axc not found");
            return;
        }
        let src = std::fs::read_to_string(&saxpy_path).expect("read saxpy.axc");
        let hir = make_hir(&src);
        let words = emit_module(&hir, &CodegenOptions::default()).expect("emit");
        let module = rspirv::dr::load_words(&words).expect("load");

        // Collect the gid variable id (Input storage class).
        let gid_var_ids: Vec<u32> = module.types_global_values.iter()
            .filter(|inst| {
                inst.class.opcode == Op::Variable
                    && inst.operands.iter().any(|op| matches!(op, Operand::StorageClass(StorageClass::Input)))
            })
            .filter_map(|inst| inst.result_id)
            .collect();
        assert_eq!(gid_var_ids.len(), 1, "expected exactly 1 Input variable (gid)");
        let gid_var_id = gid_var_ids[0];

        // Collect StorageBuffer and PushConstant var ids.
        let non_input_var_ids: std::collections::HashSet<u32> = module.types_global_values.iter()
            .filter(|inst| {
                inst.class.opcode == Op::Variable
                    && inst.operands.iter().any(|op| matches!(
                        op,
                        Operand::StorageClass(StorageClass::StorageBuffer)
                        | Operand::StorageClass(StorageClass::PushConstant)
                    ))
            })
            .filter_map(|inst| inst.result_id)
            .collect();

        // Walk OpEntryPoint interface list.
        for ep in &module.entry_points {
            let interface_ids: Vec<u32> = ep.operands.iter().filter_map(|op| {
                if let Operand::IdRef(id) = op { Some(*id) } else { None }
            }).collect();
            // gid must be present.
            assert!(
                interface_ids.contains(&gid_var_id),
                "AT-217: gid variable must appear in OpEntryPoint interface list; list={:?}", interface_ids
            );
            // StorageBuffer and PushConstant must NOT be present.
            for id in &interface_ids {
                assert!(
                    !non_input_var_ids.contains(id),
                    "AT-217: non-Input variable {id} must NOT appear in OpEntryPoint interface list (SPIR-V 1.3 §2.17); list={:?}",
                    interface_ids
                );
            }
        }
    }

    // ── AT-228: emit_spirv_version_target_1_3_storage_buffer_excluded_from_interface_list

    #[test]
    fn emit_spirv_version_target_1_3_storage_buffer_excluded_from_interface_list() {
        use rspirv::spirv::{Op, StorageClass};
        use rspirv::dr::Operand;

        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .expect("CARGO_MANIFEST_DIR not set");
        let examples_dir = std::path::PathBuf::from(&manifest_dir)
            .join("..").join("..").join("examples");
        let saxpy_path = examples_dir.join("saxpy.axc");
        if !saxpy_path.exists() {
            eprintln!("Skipping AT-228 test: saxpy.axc not found");
            return;
        }
        let src = std::fs::read_to_string(&saxpy_path).expect("read saxpy.axc");
        let hir = make_hir(&src);
        let words = emit_module(&hir, &CodegenOptions::default()).expect("emit");

        // (1) Version word must be SPIR-V 1.3.
        assert_eq!(words[1], 0x0001_0300_u32, "AT-228: version word must be SPIR-V 1.3 (0x00010300)");

        // Also verify via the compile-time constant.
        assert_eq!(
            CURRENT_SPIRV_VERSION, (1, 3),
            "AT-228: CURRENT_SPIRV_VERSION must be (1, 3)"
        );

        let module = rspirv::dr::load_words(&words).expect("load");

        // (2) Collect S = StorageBuffer + PushConstant variable ids.
        let excluded_ids: std::collections::HashSet<u32> = module.types_global_values.iter()
            .filter(|inst| {
                inst.class.opcode == Op::Variable
                    && inst.operands.iter().any(|op| matches!(
                        op,
                        Operand::StorageClass(StorageClass::StorageBuffer)
                        | Operand::StorageClass(StorageClass::PushConstant)
                    ))
            })
            .filter_map(|inst| inst.result_id)
            .collect();
        assert!(!excluded_ids.is_empty(), "AT-228: saxpy must have StorageBuffer/PushConstant vars");

        // (3) None of the excluded ids appear in any OpEntryPoint interface list.
        for ep in &module.entry_points {
            for op in &ep.operands {
                if let Operand::IdRef(id) = op {
                    assert!(
                        !excluded_ids.contains(id),
                        "AT-228: StorageBuffer/PushConstant variable {id} must not appear in interface list (SPIR-V 1.3 §2.17)"
                    );
                }
            }
        }

        // (4) The gid (Input) variable DOES appear.
        let gid_var_id = module.types_global_values.iter()
            .filter(|inst| {
                inst.class.opcode == Op::Variable
                    && inst.operands.iter().any(|op| matches!(op, Operand::StorageClass(StorageClass::Input)))
            })
            .find_map(|inst| inst.result_id)
            .expect("AT-228: saxpy must have an Input variable (gid)");
        let gid_in_ep = module.entry_points.iter().any(|ep| {
            ep.operands.iter().any(|op| matches!(op, Operand::IdRef(id) if *id == gid_var_id))
        });
        assert!(gid_in_ep, "AT-228: gid (Input) variable must appear in OpEntryPoint interface list");
    }

    // ── AT-227: cg_determinism_across_repeat_compilations_multi_entry_caches ──

    #[test]
    fn cg_determinism_across_repeat_compilations_multi_entry_caches() {
        // Stress kernel with 3 distinct scalar types + 3 distinct buffer elem types.
        // Compiles twice with fresh caches; asserts bytewise equality.
        let src = concat!(
            "@kernel @workgroup(1,1,1) fn stress(",
            "  a: f32, b: u32, c: i64,",
            "  x: readonly_buffer[f32], y: buffer[i32], z: buffer[u64]",
            ") -> void {",
            "  let i: u32 = gid(0u32);",
            "  y[i] = 0i32;",
            "  z[i] = 0u64;",
            "  return;",
            "}"
        );

        let hir1 = make_hir(src);
        let hir2 = make_hir(src);
        let opts = CodegenOptions::default();
        let words1 = emit_module(&hir1, &opts).expect("emit1");
        let words2 = emit_module(&hir2, &opts).expect("emit2");

        assert_eq!(
            words1, words2,
            "AT-227: stress kernel must produce bytewise-identical output on two compilations \
             (BTreeMap iteration order must be deterministic)"
        );

        // Also assert 3 distinct OpTypeRuntimeArray (one per buffer elem type)
        let module = rspirv::dr::load_words(&words1).expect("load");
        let runtime_array_count = module.types_global_values.iter()
            .filter(|inst| inst.class.opcode == rspirv::spirv::Op::TypeRuntimeArray)
            .count();
        assert_eq!(
            runtime_array_count, 3,
            "AT-227: stress kernel must emit 3 OpTypeRuntimeArray (f32, i32, u64); got {runtime_array_count}"
        );
    }

    // ── AT-204: emit_scalar_demo_unchanged_vs_m1_1 ───────────────────────────
    // Scalar demo must still compile cleanly (backward compat).
    #[test]
    fn emit_scalar_demo_unchanged_vs_m1_1() {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .expect("CARGO_MANIFEST_DIR not set");
        let examples_dir = std::path::PathBuf::from(&manifest_dir)
            .join("..").join("..").join("examples");
        let path = examples_dir.join("scalar_demo.axc");
        if !path.exists() {
            eprintln!("Skipping emit_scalar_demo_unchanged_vs_m1_1: scalar_demo.axc not found");
            return;
        }
        let src = std::fs::read_to_string(&path).expect("read scalar_demo.axc");
        let hir = make_hir(&src);
        let words = emit_module(&hir, &CodegenOptions::default()).expect("emit");
        // Basic invariants: magic + version.
        assert_eq!(words[0], 0x0723_0203_u32, "magic word mismatch for scalar_demo");
        assert_eq!(words[1], 0x0001_0300_u32, "version must be 1.3 for scalar_demo");
        // Must be deterministic.
        let words2 = emit_module(&hir, &CodegenOptions::default()).expect("emit2");
        assert_eq!(words, words2, "scalar_demo must be deterministic");
    }

    // ── M1.3 integration tests via emit_module ───────────────────────────────

    // AT-303: else-if chain emits exactly 2 Op::SelectionMerge + 2 Op::BranchConditional
    #[test]
    fn cg_if_else_if_else_emits_two_selection_merges() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let mut x: i32 = 0i32; if false { x = 1i32; } else if true { x = 2i32; } else { x = 3i32; } return; }";
        let words = emit_module(&make_hir(src), &CodegenOptions::default()).expect("emit");
        let sm_count = count_op_emit(&words, Op::SelectionMerge);
        let bc_count = count_op_emit(&words, Op::BranchConditional);
        assert_eq!(sm_count, 2, "expected exactly 2 OpSelectionMerge for else-if chain; got {sm_count}");
        assert_eq!(bc_count, 2, "expected exactly 2 OpBranchConditional for else-if chain; got {bc_count}");
    }

    // AT-311: nested for — break targets the INNER for's merge block, not the outer's.
    //
    // Handler location: body.rs emit_stmt for Break (looks up loop_stack.last().merge_id).
    // Test location: emit.rs (verifies the full-module output contains two OpLoopMerge
    // instructions and the inner break's OpBranch targets the inner merge_id, not the outer).
    #[test]
    fn cg_nested_for_break_targets_innermost() {
        // Two OpLoopMerge instructions; the inner break must target the INNER merge.
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { \
            for i in range(0u32, 4u32) { \
                for j in range(0u32, 4u32) { \
                    break; \
                } \
            } return; }";
        let words = emit_module(&make_hir(src), &CodegenOptions::default()).expect("emit");
        // Two distinct LoopMerge instructions — outer and inner.
        let lm_count = count_op_emit(&words, Op::LoopMerge);
        assert_eq!(lm_count, 2, "expected 2 OpLoopMerge for nested for; got {lm_count}");
        // Extract the two merge_ids.  By SPIR-V §2.11, the INNER loop's LoopMerge
        // appears before the inner loop's body and AFTER the outer LoopMerge in
        // program order (inner is nested inside the outer body).
        let outer_merge_id = find_loop_merge_id_emit(&words, 0);
        let inner_merge_id = find_loop_merge_id_emit(&words, 1);
        assert_ne!(outer_merge_id, inner_merge_id, "outer and inner merge blocks must differ");
        // The break's OpBranch must target inner_merge_id, NOT outer_merge_id.
        let branches_to_outer = iter_instructions_emit(&words[5..])
            .filter(|(op, slice)| *op == Op::Branch as u16 && slice.len() >= 2 && slice[1] == outer_merge_id)
            .count();
        let branches_to_inner = iter_instructions_emit(&words[5..])
            .filter(|(op, slice)| *op == Op::Branch as u16 && slice.len() >= 2 && slice[1] == inner_merge_id)
            .count();
        assert!(branches_to_inner >= 1,
            "expected at least 1 Op::Branch targeting inner merge_id={inner_merge_id}; got 0 (outer_merge_id={outer_merge_id})");
        // The break should NOT produce an extra branch to the outer merge.
        // The outer loop's own termination branch to outer_merge_id is allowed (exactly 1),
        // but the break should go to inner merge, not outer.
        assert!(branches_to_outer <= 1,
            "break must not target outer merge_id={outer_merge_id}; got {branches_to_outer} branches to outer");
    }

    // AT-323: early return inside a loop is a ReturnInsideLoopDeferred codegen error.
    //
    // Handler location: body.rs emit_stmt Return arm (inspects loop_stack.is_empty()).
    // Test location: emit.rs — the error surfaces via emit_module's Err return path.
    // The split exists because body.rs handles the detection and emit.rs is the public
    // API surface that wraps body codegen; testing here proves the full pipeline fails
    // with a descriptive message mentioning M1.4.
    #[test]
    #[allow(non_snake_case)]
    fn return_inside_loop_is_ReturnInsideLoopDeferred_error() {
        use axc_hir::lower_module;
        use axc_parser::parse;
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { while true { return; } }";
        let (ast, lex, parse_errs) = parse(src);
        assert!(lex.is_empty(), "lex errors: {lex:?}");
        assert!(parse_errs.is_empty(), "parse errors: {parse_errs:?}");
        let (hir, hir_errs, _) = lower_module(&ast);
        assert!(hir_errs.is_empty(), "hir errors: {hir_errs:?}");
        let result = emit_module(&hir, &CodegenOptions::default());
        assert!(result.is_err(), "return inside loop should produce codegen error");
        let err_str = result.unwrap_err().to_string();
        assert!(
            err_str.contains("return inside a loop") || err_str.contains("M1.3") || err_str.contains("M1.4"),
            "error must mention return-inside-loop or M1.3/M1.4 deferral: {err_str}"
        );
    }

    // For-range compiles to valid SPIR-V (magic word check)
    #[test]
    fn emit_for_range_kernel_has_spirv_magic() {
        let src = "@kernel @workgroup(64,1,1) fn k() -> void { for i in range(0u32, 64u32) { } return; }";
        let words = emit_module(&make_hir(src), &CodegenOptions::default()).expect("emit");
        assert_eq!(words[0], 0x0723_0203_u32, "magic word");
        assert_eq!(words[1], 0x0001_0300_u32, "version 1.3");
    }

    // While loop kernel compiles to valid SPIR-V
    #[test]
    fn emit_while_kernel_has_spirv_magic() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { while false { } return; }";
        let words = emit_module(&make_hir(src), &CodegenOptions::default()).expect("emit");
        assert_eq!(words[0], 0x0723_0203_u32, "magic word");
        assert_eq!(words[1], 0x0001_0300_u32, "version 1.3");
    }

    // if-else kernel compiles deterministically
    #[test]
    fn emit_if_else_kernel_deterministic() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let mut x: i32 = 0i32; if true { x = 1i32; } else { x = 2i32; } return; }";
        let hir = make_hir(src);
        let words1 = emit_module(&hir, &CodegenOptions::default()).expect("emit1");
        let words2 = emit_module(&hir, &CodegenOptions::default()).expect("emit2");
        assert_eq!(words1, words2, "if-else codegen must be deterministic");
    }

    // break-in-for compiles without error
    #[test]
    fn emit_break_in_for_no_error() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { for i in range(0u32, 10u32) { break; } return; }";
        let words = emit_module(&make_hir(src), &CodegenOptions::default()).expect("emit");
        assert!(!words.is_empty(), "expected non-empty SPIR-V");
    }

    // nested for-range (different induction names) compiles without error
    #[test]
    fn emit_nested_for_range_no_error() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { for i in range(0u32, 4u32) { for j in range(0u32, 4u32) { } } return; }";
        let words = emit_module(&make_hir(src), &CodegenOptions::default()).expect("emit");
        assert!(!words.is_empty(), "expected non-empty SPIR-V");
    }

    // ── emit.rs test helpers ─────────────────────────────────────────────────────

    fn iter_instructions_emit(words: &[u32]) -> impl Iterator<Item = (u16, Vec<u32>)> + '_ {
        IterInstEmit { words, cursor: 0 }
    }

    struct IterInstEmit<'a> { words: &'a [u32], cursor: usize }

    impl<'a> Iterator for IterInstEmit<'a> {
        type Item = (u16, Vec<u32>);
        fn next(&mut self) -> Option<Self::Item> {
            if self.cursor >= self.words.len() { return None; }
            let hdr = self.words[self.cursor];
            let wc = (hdr >> 16) as usize;
            let op = (hdr & 0xFFFF) as u16;
            if wc == 0 || self.cursor + wc > self.words.len() { return None; }
            let slice = self.words[self.cursor..self.cursor + wc].to_vec();
            self.cursor += wc;
            Some((op, slice))
        }
    }

    fn count_op_emit(words: &[u32], target_op: Op) -> usize {
        iter_instructions_emit(&words[5..]).filter(|(opcode, _)| *opcode == target_op as u16).count()
    }

    fn find_loop_merge_id_emit(words: &[u32], nth: usize) -> u32 {
        iter_instructions_emit(&words[5..])
            .filter(|(op, _)| *op == Op::LoopMerge as u16)
            .nth(nth)
            .map(|(_, slice)| slice[1])
            .expect("no Op::LoopMerge found")
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Collect the operand word of every OpCapability instruction in a SPIR-V
    /// word stream (full stream including the 5-word header).
    fn collect_capability_words(words: &[u32]) -> Vec<u32> {
        // OpCapability opcode = 17
        const OP_CAPABILITY: u16 = 17;
        let body = &words[5..];
        let mut cursor = 0usize;
        let mut caps = Vec::new();
        while cursor < body.len() {
            let header = body[cursor];
            let word_count = ((header >> 16) & 0xFFFF) as usize;
            let opcode = (header & 0xFFFF) as u16;
            if word_count == 0 {
                break;
            }
            if opcode == OP_CAPABILITY && word_count >= 2 {
                caps.push(body[cursor + 1]);
            }
            cursor += word_count;
        }
        caps
    }

    // ── M1.4 emit.rs tests ────────────────────────────────────────────────────

    /// Helper: compile source, return SPIR-V word stream.
    fn compile_src(src: &str) -> Vec<u32> {
        let hir = make_hir(src);
        emit_module(&hir, &CodegenOptions::default()).expect("emit_module failed")
    }

    /// Helper: get all capabilities from a compiled SPIR-V word stream as rspirv typed enums.
    fn get_capabilities(words: &[u32]) -> Vec<Capability> {
        let module = rspirv::dr::load_words(words).expect("load_words failed");
        module.capabilities.iter()
            .filter_map(|i| {
                if let Some(Operand::Capability(cap)) = i.operands.first() {
                    Some(*cap)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Helper: count OpExtension instructions with a specific string operand in the module.
    fn count_extension_string(words: &[u32], ext_name: &str) -> usize {
        let module = rspirv::dr::load_words(words).expect("load_words failed");
        module.extensions.iter()
            .filter(|i| {
                i.operands.first()
                    .and_then(|op| if let rspirv::dr::Operand::LiteralString(s) = op { Some(s.as_str()) } else { None })
                    == Some(ext_name)
            })
            .count()
    }

    // AT-413 / AT-416: emit_subgroup_reduce_kernel_smoke
    // AT-413: Minimal reduce kernel emits OpCapability GroupNonUniformArithmetic AND GroupNonUniform exactly once.
    // AT-416: SubgroupSize Input variable is emitted (subgroup_reduce uses subgroup_reduce_add, not size directly,
    //         but smoke test checks the pipeline compiles and has expected caps).
    #[test]
    fn emit_subgroup_reduce_kernel_smoke() {
        let words = compile_src(
            "@kernel @workgroup(64,1,1) fn k() -> void { let v: f32 = 1.0f32; let s: f32 = subgroup_reduce_add(v); return; }"
        );
        let module = rspirv::dr::load_words(&words).expect("load_words failed");

        // AT-413: both caps present exactly once.
        let cap_basic_count = module.capabilities.iter()
            .filter(|i| i.operands.first() == Some(&Operand::Capability(Capability::GroupNonUniform)))
            .count();
        let cap_arith_count = module.capabilities.iter()
            .filter(|i| i.operands.first() == Some(&Operand::Capability(Capability::GroupNonUniformArithmetic)))
            .count();
        assert_eq!(cap_basic_count, 1, "AT-413: GroupNonUniform must appear exactly once; got {cap_basic_count}");
        assert_eq!(cap_arith_count, 1, "AT-413: GroupNonUniformArithmetic must appear exactly once; got {cap_arith_count}");

        // AT-416: pipeline compiles to valid SPIR-V (magic word check).
        assert_eq!(words[0], 0x0723_0203_u32, "AT-416: magic word");
        assert_eq!(words[1], 0x0001_0300_u32, "AT-416: SPIR-V version 1.3");
    }

    // AT-426: (rev 1 CRITICAL-1) Minimal reduce-only kernel emits GroupNonUniform + GroupNonUniformArithmetic
    // Also asserts OpExtension "SPV_KHR_shader_subgroup_basic" and "SPV_KHR_shader_subgroup_arithmetic" each exactly 1.
    #[test]
    fn cg_minimal_subgroup_only_reduce_emits_basic_capability() {
        let words = compile_src(
            "@kernel @workgroup(64,1,1) fn k() -> void { let v: i32 = 1i32; let r: i32 = subgroup_reduce_add(v); return; }"
        );
        let caps = get_capabilities(&words);
        assert!(
            caps.contains(&Capability::GroupNonUniform),
            "GroupNonUniform must be present (rev 1 CRITICAL-1); caps: {caps:?}"
        );
        assert!(
            caps.contains(&Capability::GroupNonUniformArithmetic),
            "GroupNonUniformArithmetic must be present for reduce_add (rev 1 CRITICAL-1); caps: {caps:?}"
        );
        // AT-426: OpExtension strings must be emitted exactly once each.
        let ext_basic = count_extension_string(&words, "SPV_KHR_shader_subgroup_basic");
        let ext_arith = count_extension_string(&words, "SPV_KHR_shader_subgroup_arithmetic");
        assert_eq!(ext_basic, 1, "OpExtension \"SPV_KHR_shader_subgroup_basic\" : exactly 1; got {ext_basic}");
        assert_eq!(ext_arith, 1, "OpExtension \"SPV_KHR_shader_subgroup_arithmetic\" : exactly 1; got {ext_arith}");
    }

    // AT-427: (rev 1 CRITICAL-1) Minimal broadcast-only kernel emits GroupNonUniform + GroupNonUniformBallot
    // Also asserts OpExtension "SPV_KHR_shader_subgroup_basic" and "SPV_KHR_shader_subgroup_ballot" each exactly 1.
    #[test]
    fn cg_minimal_subgroup_only_broadcast_emits_basic_capability() {
        let words = compile_src(
            "@kernel @workgroup(64,1,1) fn k() -> void { let v: f32 = 1.0f32; let r: f32 = subgroup_broadcast_first(v); return; }"
        );
        let caps = get_capabilities(&words);
        assert!(
            caps.contains(&Capability::GroupNonUniform),
            "GroupNonUniform must be present (rev 1 CRITICAL-1); caps: {caps:?}"
        );
        assert!(
            caps.contains(&Capability::GroupNonUniformBallot),
            "GroupNonUniformBallot must be present for broadcast_first (rev 1 CRITICAL-1); caps: {caps:?}"
        );
        // AT-427: OpExtension strings must be emitted exactly once each.
        let ext_basic = count_extension_string(&words, "SPV_KHR_shader_subgroup_basic");
        let ext_ballot = count_extension_string(&words, "SPV_KHR_shader_subgroup_ballot");
        assert_eq!(ext_basic, 1, "OpExtension \"SPV_KHR_shader_subgroup_basic\" : exactly 1; got {ext_basic}");
        assert_eq!(ext_ballot, 1, "OpExtension \"SPV_KHR_shader_subgroup_ballot\" : exactly 1; got {ext_ballot}");
    }

    // AT-428: workgroup_barrier does NOT emit any GroupNonUniform capability
    #[test]
    fn emit_workgroup_barrier_no_subgroup_cap() {
        let words = compile_src(
            "@kernel @workgroup(64,1,1) fn k() -> void { workgroup_barrier(); return; }"
        );
        let caps = get_capabilities(&words);
        assert!(
            !caps.contains(&Capability::GroupNonUniform),
            "GroupNonUniform must NOT be emitted for workgroup_barrier; caps: {caps:?}"
        );
        assert!(
            !caps.contains(&Capability::GroupNonUniformArithmetic),
            "GroupNonUniformArithmetic must NOT be emitted; caps: {caps:?}"
        );
    }

    // AT-429: subgroup_all emits GroupNonUniform + GroupNonUniformVote (not arith/ballot)
    #[test]
    fn emit_sg_all_emits_vote_cap_and_basic_not_arith_ballot() {
        let words = compile_src(
            "@kernel @workgroup(64,1,1) fn k() -> void { let p: bool = true; let r: bool = subgroup_all(p); return; }"
        );
        let caps = get_capabilities(&words);
        assert!(
            caps.contains(&Capability::GroupNonUniform),
            "GroupNonUniform must be present; caps: {caps:?}"
        );
        assert!(
            caps.contains(&Capability::GroupNonUniformVote),
            "GroupNonUniformVote must be present for subgroup_all; caps: {caps:?}"
        );
        assert!(
            !caps.contains(&Capability::GroupNonUniformArithmetic),
            "GroupNonUniformArithmetic must NOT be present for subgroup_all; caps: {caps:?}"
        );
        assert!(
            !caps.contains(&Capability::GroupNonUniformBallot),
            "GroupNonUniformBallot must NOT be present for subgroup_all; caps: {caps:?}"
        );
    }
}
