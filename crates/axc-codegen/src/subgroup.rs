//! SPIR-V emission helpers for subgroup operations and workgroup barriers (M1.4).
//!
//! Each helper takes a mutable `Builder`, a mutable `ScalarTypeCache`, a mutable
//! `CapabilitiesRequired`, and the operand word(s) needed. They return the result
//! `Word` (or `Result<(), SubgroupCodegenError>` for the barrier).
//!
//! **Capability dependency invariant (rev 1, CRITICAL-1):**
//! Every helper that sets `caps.subgroup_vote`, `caps.subgroup_arith`, or
//! `caps.subgroup_ballot` ALSO sets `caps.subgroup_basic = true`, because
//! `GroupNonUniformVote`, `GroupNonUniformArithmetic`, and `GroupNonUniformBallot`
//! all implicitly require `GroupNonUniform` per SPIR-V §3.31 capability table.
//!
//! `emit_workgroup_barrier` sets NO capability beyond base `Shader` (already
//! emitted at module top).

use rspirv::dr::Builder;
use rspirv::spirv::{Word, BuiltIn, StorageClass, Decoration, GroupOperation};
use axc_hir::ty::ScalarTy;
use crate::body::{ScalarTypeCache, CapabilitiesRequired};

/// SPIR-V literal for `ExecutionScope::Subgroup == 3`.
///
/// See SPIR-V spec §3.27 (Execution Scope).
pub const EXECUTION_SCOPE_SUBGROUP: u32 = 3;

/// SPIR-V literal for `ExecutionScope::Workgroup == 2`.
///
/// See SPIR-V spec §3.27 (Execution Scope).
pub const EXECUTION_SCOPE_WORKGROUP: u32 = 2;

/// SPIR-V literal for `MemorySemantics::AcquireRelease (0x8) | MemorySemantics::WorkgroupMemory (0x100) = 0x108`.
///
/// See SPIR-V spec §3.25 (Memory Semantics). Used for `workgroup_barrier()`.
pub const MEMORY_SEMANTICS_WORKGROUP_ACQ_REL: u32 = 0x108;

/// Subgroup vote direction (All or Any).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubgroupVote {
    All,
    Any,
}

/// Subgroup reduce operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubgroupReduceOp {
    Add,
    Min,
    Max,
}

/// Errors from subgroup SPIR-V emission.
#[derive(Debug, thiserror::Error)]
pub enum SubgroupCodegenError {
    #[error("unsupported element type `{ty}` for subgroup_reduce_{op}; expected i32, u32, f32, or f64")]
    UnsupportedReduceType {
        op: &'static str,
        ty: &'static str,
    },
    #[error("internal rspirv error: {0}")]
    Rspirv(String),
}

// ── u32 constant cache ────────────────────────────────────────────────────────

/// Get or create an `OpConstant u32` for the given literal value.
///
/// Uses a `BTreeMap` (not `HashMap`) for deterministic iteration order per AT-418.
/// This function is the single point where scope/semantics constants are emitted.
pub fn get_or_emit_u32_const(
    b: &mut Builder,
    tc: &mut ScalarTypeCache,
    value: u32,
) -> Word {
    tc.get_or_emit_u32_const(b, value)
}

// ── SubgroupBuiltinVars ───────────────────────────────────────────────────────

/// Holds the emitted `Input` variable IDs for `subgroup_invocation_id` and `subgroup_size`.
///
/// Both are emitted at most once per module (analogous to `gid`). `None` means
/// the corresponding builtin was not used by this kernel.
#[derive(Debug, Default)]
pub struct SubgroupBuiltinVars {
    /// OpVariable for `SubgroupLocalInvocationId` (u32 Input).
    pub invocation_id_var: Option<Word>,
    /// OpVariable for `SubgroupSize` (u32 Input).
    pub size_var: Option<Word>,
}

impl SubgroupBuiltinVars {
    /// Create a new empty `SubgroupBuiltinVars`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` if either builtin variable has been emitted.
    pub fn any_emitted(&self) -> bool {
        self.invocation_id_var.is_some() || self.size_var.is_some()
    }
}

/// Emit a single scalar `Input` builtin variable (u32).
///
/// Used for `SubgroupLocalInvocationId` and `SubgroupSize`.
/// Emits:
/// ```text
/// OpTypePointer %ptr Input %u32
/// OpVariable %var Input
/// OpDecorate %var BuiltIn <builtin>
/// ```
pub fn emit_subgroup_scalar_builtin_var(
    b: &mut Builder,
    type_cache: &mut ScalarTypeCache,
    builtin: BuiltIn,
) -> Word {
    let u32_ty = type_cache.scalar_id(b, ScalarTy::U32);

    // Input pointer type for u32.
    let u32_input_ptr = b.type_pointer(None, StorageClass::Input, u32_ty);

    // OpVariable in Input storage class.
    let var_id = b.variable(u32_input_ptr, None, StorageClass::Input, None);

    // Decorate with the appropriate builtin.
    b.decorate(var_id, Decoration::BuiltIn, [rspirv::dr::Operand::BuiltIn(builtin)]);

    var_id
}

// ── emit_subgroup_elect ───────────────────────────────────────────────────────

/// Emit `OpGroupNonUniformElect %bool %scope`.
///
/// Sets `caps.subgroup_basic = true` (requires `GroupNonUniform`).
pub fn emit_subgroup_elect(
    b: &mut Builder,
    type_cache: &mut ScalarTypeCache,
    caps: &mut CapabilitiesRequired,
) -> Word {
    let bool_ty = type_cache.scalar_id(b, ScalarTy::Bool);
    let scope_id = get_or_emit_u32_const(b, type_cache, EXECUTION_SCOPE_SUBGROUP);
    let result = b
        .group_non_uniform_elect(bool_ty, None, scope_id)
        .expect("rspirv: group_non_uniform_elect should not fail");
    caps.subgroup_basic = true;
    result
}

// ── emit_subgroup_vote ────────────────────────────────────────────────────────

/// Emit `OpGroupNonUniformAll` or `OpGroupNonUniformAny` with `%bool %scope %pred`.
///
/// Sets `caps.subgroup_vote = true` AND `caps.subgroup_basic = true`
/// (rev 1 CRITICAL-1: `GroupNonUniformVote` implicitly requires `GroupNonUniform`).
pub fn emit_subgroup_vote(
    b: &mut Builder,
    type_cache: &mut ScalarTypeCache,
    caps: &mut CapabilitiesRequired,
    which: SubgroupVote,
    pred: Word,
) -> Word {
    let bool_ty = type_cache.scalar_id(b, ScalarTy::Bool);
    let scope_id = get_or_emit_u32_const(b, type_cache, EXECUTION_SCOPE_SUBGROUP);
    let result = match which {
        SubgroupVote::All => b
            .group_non_uniform_all(bool_ty, None, scope_id, pred)
            .expect("rspirv: group_non_uniform_all should not fail"),
        SubgroupVote::Any => b
            .group_non_uniform_any(bool_ty, None, scope_id, pred)
            .expect("rspirv: group_non_uniform_any should not fail"),
    };
    // rev 1 CRITICAL-1: child sets parent too.
    caps.subgroup_vote = true;
    caps.subgroup_basic = true;
    result
}

// ── emit_subgroup_reduce ──────────────────────────────────────────────────────

/// Emit `OpGroupNonUniform{I,F}{Add,SMin,UMin,FMin,SMax,UMax,FMax}`.
///
/// Opcode selection by `(SubgroupReduceOp, ScalarTy)`:
///
/// | op  | i32  | u32  | f32/f64 |
/// |-----|------|------|---------|
/// | Add | IAdd | IAdd | FAdd    |
/// | Min | SMin | UMin | FMin    |
/// | Max | SMax | UMax | FMax    |
///
/// Sets `caps.subgroup_arith = true` AND `caps.subgroup_basic = true`
/// (rev 1 CRITICAL-1: `GroupNonUniformArithmetic` implicitly requires `GroupNonUniform`).
///
/// Uses `GroupOperation::Reduce` (typed rspirv enum; no u32 literal fallback — W2).
pub fn emit_subgroup_reduce(
    b: &mut Builder,
    type_cache: &mut ScalarTypeCache,
    caps: &mut CapabilitiesRequired,
    op: SubgroupReduceOp,
    elem_ty: ScalarTy,
    v: Word,
) -> Result<Word, SubgroupCodegenError> {
    let ty_id = type_cache.scalar_id(b, elem_ty);
    let scope_id = get_or_emit_u32_const(b, type_cache, EXECUTION_SCOPE_SUBGROUP);
    let reduce = GroupOperation::Reduce;

    let result = match (op, elem_ty) {
        (SubgroupReduceOp::Add, ScalarTy::I32) | (SubgroupReduceOp::Add, ScalarTy::U32) => {
            b.group_non_uniform_i_add(ty_id, None, scope_id, reduce, v, None)
                .map_err(|e| SubgroupCodegenError::Rspirv(e.to_string()))?
        }
        (SubgroupReduceOp::Add, ScalarTy::F32) | (SubgroupReduceOp::Add, ScalarTy::F64) => {
            b.group_non_uniform_f_add(ty_id, None, scope_id, reduce, v, None)
                .map_err(|e| SubgroupCodegenError::Rspirv(e.to_string()))?
        }
        (SubgroupReduceOp::Min, ScalarTy::I32) => {
            b.group_non_uniform_s_min(ty_id, None, scope_id, reduce, v, None)
                .map_err(|e| SubgroupCodegenError::Rspirv(e.to_string()))?
        }
        (SubgroupReduceOp::Min, ScalarTy::U32) => {
            b.group_non_uniform_u_min(ty_id, None, scope_id, reduce, v, None)
                .map_err(|e| SubgroupCodegenError::Rspirv(e.to_string()))?
        }
        (SubgroupReduceOp::Min, ScalarTy::F32) | (SubgroupReduceOp::Min, ScalarTy::F64) => {
            b.group_non_uniform_f_min(ty_id, None, scope_id, reduce, v, None)
                .map_err(|e| SubgroupCodegenError::Rspirv(e.to_string()))?
        }
        (SubgroupReduceOp::Max, ScalarTy::I32) => {
            b.group_non_uniform_s_max(ty_id, None, scope_id, reduce, v, None)
                .map_err(|e| SubgroupCodegenError::Rspirv(e.to_string()))?
        }
        (SubgroupReduceOp::Max, ScalarTy::U32) => {
            b.group_non_uniform_u_max(ty_id, None, scope_id, reduce, v, None)
                .map_err(|e| SubgroupCodegenError::Rspirv(e.to_string()))?
        }
        (SubgroupReduceOp::Max, ScalarTy::F32) | (SubgroupReduceOp::Max, ScalarTy::F64) => {
            b.group_non_uniform_f_max(ty_id, None, scope_id, reduce, v, None)
                .map_err(|e| SubgroupCodegenError::Rspirv(e.to_string()))?
        }
        (op, ty) => {
            let op_name = match op {
                SubgroupReduceOp::Add => "add",
                SubgroupReduceOp::Min => "min",
                SubgroupReduceOp::Max => "max",
            };
            return Err(SubgroupCodegenError::UnsupportedReduceType {
                op: op_name,
                ty: ty.display_name(),
            });
        }
    };

    // rev 1 CRITICAL-1: set both arith and basic.
    caps.subgroup_arith = true;
    caps.subgroup_basic = true;
    Ok(result)
}

// ── emit_subgroup_broadcast_first ─────────────────────────────────────────────

/// Emit `OpGroupNonUniformBroadcastFirst %elem_ty %scope %v`.
///
/// Sets `caps.subgroup_ballot = true` AND `caps.subgroup_basic = true`
/// (rev 1 CRITICAL-1: `GroupNonUniformBallot` implicitly requires `GroupNonUniform`).
pub fn emit_subgroup_broadcast_first(
    b: &mut Builder,
    type_cache: &mut ScalarTypeCache,
    caps: &mut CapabilitiesRequired,
    elem_ty: ScalarTy,
    v: Word,
) -> Word {
    let ty_id = type_cache.scalar_id(b, elem_ty);
    let scope_id = get_or_emit_u32_const(b, type_cache, EXECUTION_SCOPE_SUBGROUP);
    let result = b
        .group_non_uniform_broadcast_first(ty_id, None, scope_id, v)
        .expect("rspirv: group_non_uniform_broadcast_first should not fail");
    // rev 1 CRITICAL-1: ballot + basic.
    caps.subgroup_ballot = true;
    caps.subgroup_basic = true;
    result
}

// ── emit_workgroup_barrier ────────────────────────────────────────────────────

/// Emit `OpControlBarrier %wg_scope %wg_scope %semantics`.
///
/// Both execution scope and memory scope are `Workgroup (2)`.
/// Memory semantics = `AcquireRelease (0x8) | WorkgroupMemory (0x100) = 0x108`.
///
/// **This is NOT a block terminator** — subsequent instructions continue in the same
/// block. `body.rs` MUST NOT set `current_block_terminated` after calling this.
///
/// No capability beyond base `Shader` is required.
pub fn emit_workgroup_barrier(
    b: &mut Builder,
    type_cache: &mut ScalarTypeCache,
) -> Result<(), SubgroupCodegenError> {
    let scope_wg = get_or_emit_u32_const(b, type_cache, EXECUTION_SCOPE_WORKGROUP);
    let semantics = get_or_emit_u32_const(b, type_cache, MEMORY_SEMANTICS_WORKGROUP_ACQ_REL);
    b.control_barrier(scope_wg, scope_wg, semantics)
        .map_err(|e| SubgroupCodegenError::Rspirv(e.to_string()))?;
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rspirv::binary::Assemble;
    use rspirv::dr::Builder;
    use rspirv::spirv::{
        Op, Capability, FunctionControl, AddressingModel, MemoryModel, ExecutionModel, ExecutionMode,
        GroupOperation as SpvGroupOperation,
    };
    use rspirv::dr::Operand;
    use axc_hir::ty::ScalarTy;

    /// Build a minimal SPIR-V function context for unit-testing subgroup emitters.
    ///
    /// Returns `(builder, type_cache, caps)` with the function opened and a block started.
    fn make_test_context() -> (Builder, ScalarTypeCache, CapabilitiesRequired) {
        let mut b = Builder::new();
        b.set_version(1, 3);
        b.capability(Capability::Shader);
        b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);
        let void_t = b.type_void();
        let fn_t = b.type_function(void_t, vec![]);
        b.begin_function(void_t, None, FunctionControl::NONE, fn_t)
            .expect("begin_function");
        b.begin_block(None).expect("begin_block");
        let tc = ScalarTypeCache::new();
        let caps = CapabilitiesRequired::default();
        (b, tc, caps)
    }

    /// Assemble + load: returns the module for inspection.
    fn assemble_and_load(mut b: Builder) -> rspirv::dr::Module {
        b.ret().expect("ret");
        b.end_function().expect("end_function");
        let main_id = 1u32; // typically 1 in a fresh builder
        b.entry_point(ExecutionModel::GLCompute, main_id, "test", vec![]);
        b.execution_mode(main_id, ExecutionMode::LocalSize, vec![1, 1, 1]);
        let module = b.module();
        let words = module.assemble();
        rspirv::dr::load_words(&words).expect("load_words")
    }

    /// Count instructions with a given opcode in the module's functions.
    fn count_op_in_functions(module: &rspirv::dr::Module, target_op: Op) -> usize {
        module.functions.iter().flat_map(|f| f.blocks.iter()).flat_map(|block| block.instructions.iter())
            .filter(|instr| instr.class.opcode == target_op)
            .count()
    }

    // ── AT-sg-1: elect sets subgroup_basic ────────────────────────────────────

    #[test]
    fn sg_emit_elect_returns_nonzero_word_and_sets_subgroup_basic() {
        let (mut b, mut tc, mut caps) = make_test_context();
        let result = emit_subgroup_elect(&mut b, &mut tc, &mut caps);
        assert_ne!(result, 0, "elect result word should be non-zero");
        assert!(caps.subgroup_basic, "subgroup_basic must be true after emit_subgroup_elect");
    }

    // ── AT-420 / AT-421: vote sets both flags ─────────────────────────────────

    #[test]
    fn sg_emit_vote_all_sets_subgroup_vote_flag() {
        let (mut b, mut tc, mut caps) = make_test_context();
        let bool_ty = tc.scalar_id(&mut b, ScalarTy::Bool);
        let pred = b.constant_true(bool_ty);
        let result = emit_subgroup_vote(&mut b, &mut tc, &mut caps, SubgroupVote::All, pred);
        assert_ne!(result, 0, "vote all result should be non-zero");
        assert!(caps.subgroup_vote, "subgroup_vote must be true after emit_subgroup_vote All");
        // rev 1 CRITICAL-1: parent also set
        assert!(caps.subgroup_basic, "subgroup_basic must be true (GroupNonUniformVote requires GroupNonUniform)");
    }

    #[test]
    fn sg_emit_vote_any_sets_subgroup_vote_flag() {
        let (mut b, mut tc, mut caps) = make_test_context();
        let bool_ty = tc.scalar_id(&mut b, ScalarTy::Bool);
        let pred = b.constant_true(bool_ty);
        let result = emit_subgroup_vote(&mut b, &mut tc, &mut caps, SubgroupVote::Any, pred);
        assert_ne!(result, 0, "vote any result should be non-zero");
        assert!(caps.subgroup_vote, "subgroup_vote must be true after emit_subgroup_vote Any");
        // rev 1 CRITICAL-1: parent also set
        assert!(caps.subgroup_basic, "subgroup_basic must be true (GroupNonUniformVote requires GroupNonUniform)");
    }

    // ── AT-417: reduce add f32 emits FAdd ────────────────────────────────────

    #[test]
    fn sg_emit_reduce_add_f32_emits_op_group_non_uniform_f_add() {
        let (mut b, mut tc, mut caps) = make_test_context();
        let f32_ty = tc.scalar_id(&mut b, ScalarTy::F32);
        let v = b.constant_bit32(f32_ty, 0x3f800000u32); // 1.0f32
        let result = emit_subgroup_reduce(&mut b, &mut tc, &mut caps, SubgroupReduceOp::Add, ScalarTy::F32, v)
            .expect("emit_subgroup_reduce f32");
        assert_ne!(result, 0, "reduce add f32 result should be non-zero");
        assert!(caps.subgroup_arith, "subgroup_arith must be true");
        // rev 1 CRITICAL-1: parent also set
        assert!(caps.subgroup_basic, "subgroup_basic must be true (GroupNonUniformArithmetic requires GroupNonUniform)");

        // Inspect emitted opcode.
        let module = assemble_and_load(b);
        let fadd_count = count_op_in_functions(&module, Op::GroupNonUniformFAdd);
        assert_eq!(fadd_count, 1, "expected exactly 1 OpGroupNonUniformFAdd; got {fadd_count}");

        // Verify GroupOperation operand is typed Reduce enum (AT-414, W2).
        let fadd_instr = module.functions.iter().flat_map(|f| f.blocks.iter())
            .flat_map(|blk| blk.instructions.iter())
            .find(|i| i.class.opcode == Op::GroupNonUniformFAdd)
            .expect("no FAdd found");
        // Operands: [result_type, result_id (implicit), scope_id, group_op, value, ...]
        // In rspirv dr::Instruction, operands[0] = scope IdRef, operands[1] = GroupOperation, operands[2] = value
        let group_op_operand = &fadd_instr.operands[1];
        assert!(
            matches!(group_op_operand, Operand::GroupOperation(SpvGroupOperation::Reduce)),
            "GroupOperation operand must be Reduce enum (not LiteralBit32); got: {group_op_operand:?}"
        );
    }

    // ── AT-417: reduce add i32 emits IAdd ────────────────────────────────────

    #[test]
    fn sg_emit_reduce_add_i32_emits_op_group_non_uniform_i_add() {
        let (mut b, mut tc, mut caps) = make_test_context();
        let i32_ty = tc.scalar_id(&mut b, ScalarTy::I32);
        let v = b.constant_bit32(i32_ty, 1u32);
        let result = emit_subgroup_reduce(&mut b, &mut tc, &mut caps, SubgroupReduceOp::Add, ScalarTy::I32, v)
            .expect("emit_subgroup_reduce i32");
        assert_ne!(result, 0);
        assert!(caps.subgroup_arith);
        assert!(caps.subgroup_basic);
        let module = assemble_and_load(b);
        let iadd_count = count_op_in_functions(&module, Op::GroupNonUniformIAdd);
        assert_eq!(iadd_count, 1, "expected 1 OpGroupNonUniformIAdd; got {iadd_count}");
    }

    // ── reduce min u32 emits UMin ─────────────────────────────────────────────

    #[test]
    fn sg_emit_reduce_min_u32_emits_op_group_non_uniform_u_min() {
        let (mut b, mut tc, mut caps) = make_test_context();
        let u32_ty = tc.scalar_id(&mut b, ScalarTy::U32);
        let v = b.constant_bit32(u32_ty, 42u32);
        let result = emit_subgroup_reduce(&mut b, &mut tc, &mut caps, SubgroupReduceOp::Min, ScalarTy::U32, v)
            .expect("emit_subgroup_reduce u32 min");
        assert_ne!(result, 0);
        let module = assemble_and_load(b);
        let umin_count = count_op_in_functions(&module, Op::GroupNonUniformUMin);
        assert_eq!(umin_count, 1, "expected 1 OpGroupNonUniformUMin; got {umin_count}");
    }

    // ── AT-420: broadcast_first sets ballot + basic, NOT arith/vote ──────────

    #[test]
    fn sg_emit_broadcast_first_f32_sets_subgroup_ballot_flag() {
        let (mut b, mut tc, mut caps) = make_test_context();
        let f32_ty = tc.scalar_id(&mut b, ScalarTy::F32);
        let v = b.constant_bit32(f32_ty, 0x3f800000u32);
        let result = emit_subgroup_broadcast_first(&mut b, &mut tc, &mut caps, ScalarTy::F32, v);
        assert_ne!(result, 0);
        // rev 1 CRITICAL-1: ballot + basic set, arith/vote must be false.
        assert!(caps.subgroup_ballot, "subgroup_ballot must be true");
        assert!(caps.subgroup_basic, "subgroup_basic must be true (GroupNonUniformBallot requires GroupNonUniform)");
        assert!(!caps.subgroup_arith, "subgroup_arith must be false (BroadcastFirst does not need Arithmetic)");
        assert!(!caps.subgroup_vote, "subgroup_vote must be false (BroadcastFirst does not need Vote)");
    }

    // ── AT-415: workgroup barrier emits OpControlBarrier with scope=2 semant=0x108

    #[test]
    fn sg_emit_workgroup_barrier_emits_op_control_barrier_with_expected_constants() {
        let (mut b, mut tc, mut _caps) = make_test_context();
        emit_workgroup_barrier(&mut b, &mut tc).expect("emit_workgroup_barrier");

        let module = assemble_and_load(b);

        // Find the one OpControlBarrier.
        let barrier_instrs: Vec<_> = module.functions.iter().flat_map(|f| f.blocks.iter())
            .flat_map(|blk| blk.instructions.iter())
            .filter(|i| i.class.opcode == Op::ControlBarrier)
            .collect();
        assert_eq!(barrier_instrs.len(), 1, "expected exactly 1 OpControlBarrier; got {}", barrier_instrs.len());

        // Find all OpConstant instructions for scope/semantics lookup.
        let constants: Vec<(Word, u32)> = module.types_global_values.iter()
            .filter(|i| i.class.opcode == Op::Constant)
            .filter_map(|i| {
                let id = i.result_id?;
                if let Operand::LiteralBit32(v) = i.operands.first()? { Some((id, *v)) } else { None }
            })
            .collect();

        let resolve_const = |id: Word| -> Option<u32> {
            constants.iter().find(|(cid, _)| *cid == id).map(|(_, v)| *v)
        };

        let barrier = &barrier_instrs[0];
        // rspirv uses Operand::IdScope for scope IDs and Operand::IdMemorySemantics for semantics.
        // This differs from Operand::IdRef — check both to be defensive.
        let resolve_op_id = |op: &Operand| -> Option<u32> {
            match op {
                Operand::IdRef(id) => resolve_const(*id),
                Operand::IdScope(id) => resolve_const(*id),
                Operand::IdMemorySemantics(id) => resolve_const(*id),
                _ => None,
            }
        };
        let exec_scope_val = barrier.operands.get(0).and_then(resolve_op_id);
        let mem_scope_val = barrier.operands.get(1).and_then(resolve_op_id);
        let semantics_val = barrier.operands.get(2).and_then(resolve_op_id);

        assert_eq!(exec_scope_val, Some(2), "execution scope must be Workgroup (2)");
        assert_eq!(mem_scope_val, Some(2), "memory scope must be Workgroup (2)");
        assert_eq!(semantics_val, Some(0x108), "semantics must be AcquireRelease|WorkgroupMemory (0x108)");
    }

    // ── SubgroupLocalInvocationId unit test (CRITICAL-2 fix) ─────────────────
    // This test drives the CRITICAL-2 correctness requirement: compiling a minimal
    // kernel that uses ONLY `subgroup_invocation_id()` must emit exactly 1 OpLoad
    // of SubgroupLocalInvocationId and must NOT emit GroupNonUniformArithmetic/Vote/Ballot.
    #[test]
    fn sg_subgroup_invocation_id_emits_opload_of_correct_builtin() {
        use axc_parser::parse;
        use axc_hir::lower_module;
        use crate::emit::{emit_module, CodegenOptions};
        use rspirv::spirv::BuiltIn;
        use rspirv::dr::Operand;

        let src = concat!(
            "@kernel @workgroup(64,1,1) ",
            "fn k() -> void { let id: u32 = subgroup_invocation_id(); return; }"
        );
        let (ast, lex_errs, parse_errs) = parse(src);
        assert!(lex_errs.is_empty(), "lex errors: {lex_errs:?}");
        assert!(parse_errs.is_empty(), "parse errors: {parse_errs:?}");
        let (hir, hir_errs, _warns) = lower_module(&ast);
        assert!(hir_errs.is_empty(), "hir errors: {hir_errs:?}");
        let words = emit_module(&hir, &CodegenOptions::default()).expect("emit_module failed");
        let module = rspirv::dr::load_words(&words).expect("load_words failed");

        // Find the SubgroupLocalInvocationId variable.
        let invoc_id_vars: Vec<Word> = module.types_global_values.iter()
            .filter(|i| i.class.opcode == Op::Variable)
            .filter(|i| i.operands.iter().any(|op| matches!(op, Operand::StorageClass(StorageClass::Input))))
            .filter_map(|i| {
                let var_id = i.result_id?;
                // Check if this var is decorated with SubgroupLocalInvocationId
                let is_sg_invoc = module.annotations.iter().any(|ann| {
                    ann.class.opcode == Op::Decorate
                        && ann.operands.first().and_then(|op| if let Operand::IdRef(id) = op { Some(*id) } else { None }) == Some(var_id)
                        && ann.operands.iter().any(|op| matches!(op, Operand::Decoration(Decoration::BuiltIn)))
                        && ann.operands.iter().any(|op| matches!(op, Operand::BuiltIn(BuiltIn::SubgroupLocalInvocationId)))
                });
                if is_sg_invoc { Some(var_id) } else { None }
            })
            .collect();
        assert_eq!(invoc_id_vars.len(), 1, "expected exactly 1 SubgroupLocalInvocationId variable; got {}", invoc_id_vars.len());

        // Exactly 1 OpLoad of that variable.
        let load_count = module.functions.iter().flat_map(|f| f.blocks.iter())
            .flat_map(|blk| blk.instructions.iter())
            .filter(|i| {
                i.class.opcode == Op::Load
                    && i.operands.first().and_then(|op| if let Operand::IdRef(id) = op { Some(*id) } else { None })
                    == invoc_id_vars.first().copied()
            })
            .count();
        assert_eq!(load_count, 1, "expected exactly 1 OpLoad of SubgroupLocalInvocationId; got {load_count}");

        // GroupNonUniform capability is present, but NOT Arithmetic/Vote/Ballot.
        let caps_present: Vec<Capability> = module.capabilities.iter()
            .filter_map(|i| {
                if let Some(Operand::Capability(cap)) = i.operands.first() { Some(*cap) } else { None }
            })
            .collect();
        assert!(caps_present.contains(&Capability::GroupNonUniform), "GroupNonUniform must be present");
        assert!(!caps_present.contains(&Capability::GroupNonUniformArithmetic), "GroupNonUniformArithmetic must be absent");
        assert!(!caps_present.contains(&Capability::GroupNonUniformVote), "GroupNonUniformVote must be absent");
        assert!(!caps_present.contains(&Capability::GroupNonUniformBallot), "GroupNonUniformBallot must be absent");
    }
}
