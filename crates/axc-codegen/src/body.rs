//! SPIR-V body emission for AXIOM-Compute kernel bodies.
//!
//! Translates `KernelBodyTyped` (HIR) into SPIR-V instructions via rspirv.
//!
//! Key design invariants:
//! - ALL OpVariable declarations for bindings are emitted as a PRELUDE in the
//!   FIRST block (SPIR-V §2.16.1). No OpVariable may appear after any
//!   non-variable instruction.
//! - Short-circuit `and`/`or` lower to a structured diamond:
//!   OpSelectionMerge + OpBranchConditional + OpPhi. Block label IDs are
//!   PRE-ALLOCATED via `b.id()` before any block is opened, because
//!   `b.selected_block()` returns an INDEX into the function's block vec
//!   (not a SPIR-V Word), so we cannot read the label word back after
//!   `branch_conditional` ends the block (§5.7 of the architect spec).
//! - Constants are cached by `(ScalarTy, bits)` to avoid duplicate definitions.
//! - Scalar types are cached in `ScalarTypeCache` to satisfy the rspirv
//!   0.12 requirement that types be unique (rspirv does not dedup internally).
//! - `CapabilitiesRequired` accumulates Int64 / Float64 flags; the caller
//!   (`emit.rs`) inserts the corresponding `OpCapability` after the body.
//! - M1.2: `BufferRead`, `BufferWrite`, and `GidBuiltin` nodes are handled via
//!   references to pre-emitted global variables (BufferBindings, PushConstantBlock,
//!   GlobalInvocationIdVar) that `emit.rs` creates before calling this function.

use std::collections::HashMap;
use rspirv::dr::Builder;
use rspirv::spirv::{
    Word, StorageClass, SelectionControl,
};
use axc_hir::expr::{
    KernelBodyTyped, HirExpr, HirExprKind, HirStmt, BinOp, UnaryOp,
    ShortCircuitOp, BitwiseOp, BindingId,
};
use axc_hir::ty::ScalarTy;
use crate::buffers::{BufferBindings, PushConstantBlock, GlobalInvocationIdVar};

/// References to pre-emitted global IR structures for M1.2 buffer/scalar/gid support.
///
/// All three are optional: a kernel with no params uses `None` for all fields.
/// The `emit_kernel_body` function will panic at runtime (not at compile time) if
/// a `BufferRead`/`BufferWrite`/`GidBuiltin` HIR node is encountered but the
/// corresponding resource is `None` — this is a compiler bug, not a user error.
pub struct KernelResources<'r> {
    /// SSBO buffer globals emitted before the function.
    pub buffer_bindings: Option<&'r BufferBindings>,
    /// Push-constant block for scalar params.
    pub push_constant: Option<&'r PushConstantBlock>,
    /// gl_GlobalInvocationID Input variable.
    pub gid_var: Option<&'r GlobalInvocationIdVar>,
    /// Scalar param info: maps position → (member_index, ScalarTy) for load-from-PC.
    pub scalar_params: &'r [(u32, u32, ScalarTy)],  // (position, member_index, ty)
}

/// Errors from SPIR-V body emission.
#[derive(Debug, thiserror::Error)]
pub enum BodyCodegenError {
    #[error("internal rspirv error: {0}")]
    Rspirv(String),
    #[error("unexpected HIR node in body codegen: {0}")]
    UnexpectedHir(&'static str),
}

/// Cache of SPIR-V type IDs, keyed by `ScalarTy`.
///
/// rspirv 0.12 does not deduplicate type declarations internally; we must do so.
#[derive(Debug, Default)]
pub struct ScalarTypeCache {
    scalar_ids: HashMap<ScalarTy, Word>,
    pointer_ids: HashMap<ScalarTy, Word>,
}

impl ScalarTypeCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or create the SPIR-V type id for a scalar type.
    pub fn scalar_id(&mut self, b: &mut Builder, ty: ScalarTy) -> Word {
        if let Some(&id) = self.scalar_ids.get(&ty) {
            return id;
        }
        let id: Word = match ty {
            ScalarTy::I8   => b.type_int(8,  1),
            ScalarTy::I16  => b.type_int(16, 1),
            ScalarTy::I32  => b.type_int(32, 1),
            ScalarTy::I64  => b.type_int(64, 1),
            ScalarTy::U8   => b.type_int(8,  0),
            ScalarTy::U16  => b.type_int(16, 0),
            ScalarTy::U32  => b.type_int(32, 0),
            ScalarTy::U64  => b.type_int(64, 0),
            ScalarTy::F32  => b.type_float(32),
            ScalarTy::F64  => b.type_float(64),
            ScalarTy::Bool => b.type_bool(),
        };
        self.scalar_ids.insert(ty, id);
        id
    }

    /// Get or create the SPIR-V Function-storage-class pointer type for a scalar type.
    pub fn function_ptr_id(&mut self, b: &mut Builder, pointee: ScalarTy) -> Word {
        if let Some(&id) = self.pointer_ids.get(&pointee) {
            return id;
        }
        let base_id = self.scalar_id(b, pointee);
        let id = b.type_pointer(None, StorageClass::Function, base_id);
        self.pointer_ids.insert(pointee, id);
        id
    }
}

/// Tracks which 64-bit capabilities are required by the emitted body.
#[derive(Debug, Default, Clone, Copy)]
pub struct CapabilitiesRequired {
    /// True if any i64 or u64 type was used — requires `OpCapability Int64`.
    pub int64: bool,
    /// True if any f64 type was used — requires `OpCapability Float64`.
    pub float64: bool,
}

impl CapabilitiesRequired {
    fn observe_type(&mut self, ty: ScalarTy) {
        match ty {
            ScalarTy::I64 | ScalarTy::U64 => { self.int64 = true; }
            ScalarTy::F64 => { self.float64 = true; }
            _ => {}
        }
    }
}

// ── Constant cache key ────────────────────────────────────────────────────────

/// A key for caching constants: (ScalarTy, bits).
/// Bool constants are keyed as (Bool, 0) for false and (Bool, 1) for true.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ConstKey(ScalarTy, u64);

// ── Emission context ──────────────────────────────────────────────────────────

struct BodyEmitter<'a> {
    b: &'a mut Builder,
    type_cache: &'a mut ScalarTypeCache,
    caps: &'a mut CapabilitiesRequired,
    /// Maps BindingId → SPIR-V variable id (OpVariable in Function storage).
    var_ids: HashMap<BindingId, Word>,
    /// Cached constants: (ScalarTy, bits) → result id.
    const_cache: HashMap<ConstKey, Word>,
    /// References to pre-emitted global M1.2 resources (buffers, push-constants, gid).
    res: &'a KernelResources<'a>,
}

impl<'a> BodyEmitter<'a> {
    fn type_id(&mut self, ty: ScalarTy) -> Word {
        self.caps.observe_type(ty);
        self.type_cache.scalar_id(self.b, ty)
    }

    fn ptr_type_id(&mut self, ty: ScalarTy) -> Word {
        self.caps.observe_type(ty);
        self.type_cache.function_ptr_id(self.b, ty)
    }

    /// Get or create an integer/float constant.
    fn get_const_int(&mut self, ty: ScalarTy, bits: u64) -> Word {
        let key = ConstKey(ty, bits);
        if let Some(&id) = self.const_cache.get(&key) {
            return id;
        }
        let ty_id = self.type_id(ty);
        let id: Word = match ty.bit_width() {
            32 => self.b.constant_bit32(ty_id, bits as u32),
            64 => self.b.constant_bit64(ty_id, bits),
            8  => self.b.constant_bit32(ty_id, bits as u32),
            16 => self.b.constant_bit32(ty_id, bits as u32),
            _ => unreachable!("unexpected int bit width in get_const_int"),
        };
        self.const_cache.insert(key, id);
        id
    }

    fn get_const_float32(&mut self, bits: u32) -> Word {
        let key = ConstKey(ScalarTy::F32, bits as u64);
        if let Some(&id) = self.const_cache.get(&key) {
            return id;
        }
        let ty_id = self.type_id(ScalarTy::F32);
        let id = self.b.constant_bit32(ty_id, bits);
        self.const_cache.insert(key, id);
        id
    }

    fn get_const_float64(&mut self, bits: u64) -> Word {
        let key = ConstKey(ScalarTy::F64, bits);
        if let Some(&id) = self.const_cache.get(&key) {
            return id;
        }
        let ty_id = self.type_id(ScalarTy::F64);
        let id = self.b.constant_bit64(ty_id, bits);
        self.const_cache.insert(key, id);
        id
    }

    fn get_const_bool(&mut self, value: bool) -> Word {
        let key = ConstKey(ScalarTy::Bool, value as u64);
        if let Some(&id) = self.const_cache.get(&key) {
            return id;
        }
        let ty_id = self.type_id(ScalarTy::Bool);
        let id = if value {
            self.b.constant_true(ty_id)
        } else {
            self.b.constant_false(ty_id)
        };
        self.const_cache.insert(key, id);
        id
    }

    /// Get the label id of the currently-selected block.
    ///
    /// Per §5.7.2: `b.selected_block()` returns Option<usize> — an INDEX into
    /// the function's block vec. We pre-allocated our label ids via `b.id()` and
    /// passed them to `begin_block(Some(id))`, so we can read the label back from
    /// `b.module_ref().functions[f].blocks[idx].label.result_id`.
    fn current_block_label(&self) -> Result<Word, BodyCodegenError> {
        let func_idx = match self.b.selected_function() {
            Some(f) => f,
            None => return Err(BodyCodegenError::Rspirv("no function selected".to_owned())),
        };
        let block_idx = match self.b.selected_block() {
            Some(i) => i,
            None => return Err(BodyCodegenError::Rspirv("no block selected".to_owned())),
        };
        let module = self.b.module_ref();
        let label_id = module.functions[func_idx].blocks[block_idx]
            .label
            .as_ref()
            .and_then(|instr| instr.result_id)
            .ok_or_else(|| BodyCodegenError::Rspirv("block label has no result_id".to_owned()))?;
        Ok(label_id)
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Emit SPIR-V for a typed kernel body.
///
/// The caller MUST have already called `begin_function` and allocated/opened
/// the first block with `begin_block(Some(first_block_label))` BEFORE calling
/// this function.
///
/// Returns the id of the first block (for use in OpBranchConditional / OpPhi
/// by outer calling code, though in practice the driver calls this unconditionally).
///
/// After this call the current block is the last merge/straight-line block
/// (containing OpReturn).
pub fn emit_kernel_body(
    b: &mut Builder,
    body: &KernelBodyTyped,
    type_cache: &mut ScalarTypeCache,
    caps: &mut CapabilitiesRequired,
    res: &KernelResources<'_>,
) -> Result<Word, BodyCodegenError> {
    // The first block's label is the one that was opened before we were called.
    let first_block_label = match (b.selected_function(), b.selected_block()) {
        (Some(fi), Some(bi)) => {
            b.module_ref().functions[fi].blocks[bi]
                .label
                .as_ref()
                .and_then(|instr| instr.result_id)
                .ok_or_else(|| BodyCodegenError::Rspirv("first block has no label".to_owned()))?
        }
        _ => return Err(BodyCodegenError::Rspirv("no block open when emit_kernel_body called".to_owned())),
    };

    let mut emitter = BodyEmitter {
        b,
        type_cache,
        caps,
        var_ids: HashMap::new(),
        const_cache: HashMap::new(),
        res,
    };

    // ── Prelude: emit ALL OpVariable declarations in the first block ──────────
    // SPIR-V §2.16.1: all OpVariable instructions for a function MUST appear in
    // the first block of the function, before any other instructions.
    for binding in &body.bindings {
        emitter.caps.observe_type(binding.ty);
        let ptr_ty = emitter.ptr_type_id(binding.ty);
        let var_id = emitter.b
            .variable(ptr_ty, None, StorageClass::Function, None);
        emitter.var_ids.insert(binding.id, var_id);
    }

    // ── Emit statements ───────────────────────────────────────────────────────
    for stmt in &body.stmts {
        emit_stmt(&mut emitter, stmt)?;
    }

    Ok(first_block_label)
}

// ── Statement emission ────────────────────────────────────────────────────────

fn emit_stmt(em: &mut BodyEmitter<'_>, stmt: &HirStmt) -> Result<(), BodyCodegenError> {
    match stmt {
        HirStmt::Let { binding, init, .. } => {
            let init_id = emit_expr(em, init)?;
            let var_id = em.var_ids.get(binding).copied()
                .ok_or(BodyCodegenError::UnexpectedHir("Let binding not in var_ids"))?;
            let init_ty = em.type_id(init.ty);
            em.b.store(var_id, init_id, None, None)
                .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;
            let _ = init_ty;
            Ok(())
        }
        HirStmt::Assign { binding, value, .. } => {
            let val_id = emit_expr(em, value)?;
            let var_id = em.var_ids.get(binding).copied()
                .ok_or(BodyCodegenError::UnexpectedHir("Assign binding not in var_ids"))?;
            em.b.store(var_id, val_id, None, None)
                .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;
            Ok(())
        }
        HirStmt::Return { .. } => {
            em.b.ret().map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;
            Ok(())
        }
        HirStmt::BufferWrite { buffer_binding, index, value, .. } => {
            emit_buffer_write(em, *buffer_binding, index, value)
        }
    }
}

// ── Expression emission ───────────────────────────────────────────────────────

fn emit_expr(em: &mut BodyEmitter<'_>, expr: &HirExpr) -> Result<Word, BodyCodegenError> {
    match &expr.kind {
        HirExprKind::IntLit { value } => {
            let id = em.get_const_int(value.ty, value.bits);
            Ok(id)
        }
        HirExprKind::FloatLit { value } => {
            let id = match value.ty {
                ScalarTy::F32 => em.get_const_float32(value.bits as u32),
                ScalarTy::F64 => em.get_const_float64(value.bits),
                _ => return Err(BodyCodegenError::UnexpectedHir("float lit with non-float type")),
            };
            Ok(id)
        }
        HirExprKind::BoolLit(b) => {
            let id = em.get_const_bool(*b);
            Ok(id)
        }
        HirExprKind::LocalRead(bid) => {
            // Sentinel BindingId(u32::MAX - pos) signals a scalar push-constant read.
            // Any id >= 0x8000_0000 is treated as a sentinel (params can't have >2B positions).
            if bid.0 >= 0x8000_0000 {
                let param_position: u32 = u32::MAX - bid.0;
                return emit_scalar_param_read(em, param_position, expr.ty);
            }
            let var_id = em.var_ids.get(bid).copied()
                .ok_or(BodyCodegenError::UnexpectedHir("LocalRead: binding not in var_ids"))?;
            let ty_id = em.type_id(expr.ty);
            let id = em.b.load(ty_id, None, var_id, None, None)
                .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;
            Ok(id)
        }
        HirExprKind::Unary { op, operand } => {
            emit_unary(em, *op, operand, expr.ty)
        }
        HirExprKind::Binary { op, lhs, rhs } => {
            emit_binary(em, *op, lhs, rhs, expr.ty)
        }
        HirExprKind::ShortCircuit { op, lhs, rhs } => {
            emit_short_circuit(em, *op, lhs, rhs)
        }
        HirExprKind::BitwiseBuiltin { op, args } => {
            emit_bitwise(em, *op, args, expr.ty)
        }
        HirExprKind::BufferRead { buffer_binding, index, .. } => {
            emit_buffer_read(em, *buffer_binding, index, expr.ty)
        }
        HirExprKind::GidBuiltin { axis } => {
            emit_gid_component(em, *axis)
        }
    }
}

// ── Unary ops ─────────────────────────────────────────────────────────────────

fn emit_unary(
    em: &mut BodyEmitter<'_>,
    op: UnaryOp,
    operand: &HirExpr,
    result_ty: ScalarTy,
) -> Result<Word, BodyCodegenError> {
    let operand_id = emit_expr(em, operand)?;
    let ty_id = em.type_id(result_ty);
    let id = match op {
        UnaryOp::Neg => {
            if result_ty.is_signed_integer() {
                em.b.s_negate(ty_id, None, operand_id)
                    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
            } else if result_ty.is_float() {
                em.b.f_negate(ty_id, None, operand_id)
                    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
            } else {
                return Err(BodyCodegenError::UnexpectedHir("Neg on non-signed/float type reached codegen"));
            }
        }
        UnaryOp::LogicalNot => {
            em.b.logical_not(ty_id, None, operand_id)
                .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
        }
    };
    Ok(id)
}

// ── Binary ops (§5.4 table) ───────────────────────────────────────────────────

fn emit_binary(
    em: &mut BodyEmitter<'_>,
    op: BinOp,
    lhs: &HirExpr,
    rhs: &HirExpr,
    result_ty: ScalarTy,
) -> Result<Word, BodyCodegenError> {
    let lhs_id = emit_expr(em, lhs)?;
    let rhs_id = emit_expr(em, rhs)?;
    let ty_id = em.type_id(result_ty);
    let operand_ty = lhs.ty;

    let id: Word = match op {
        BinOp::Add => {
            if operand_ty.is_integer() {
                em.b.i_add(ty_id, None, lhs_id, rhs_id)
                    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
            } else {
                em.b.f_add(ty_id, None, lhs_id, rhs_id)
                    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
            }
        }
        BinOp::Sub => {
            if operand_ty.is_integer() {
                em.b.i_sub(ty_id, None, lhs_id, rhs_id)
                    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
            } else {
                em.b.f_sub(ty_id, None, lhs_id, rhs_id)
                    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
            }
        }
        BinOp::Mul => {
            if operand_ty.is_integer() {
                em.b.i_mul(ty_id, None, lhs_id, rhs_id)
                    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
            } else {
                em.b.f_mul(ty_id, None, lhs_id, rhs_id)
                    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
            }
        }
        BinOp::Div => {
            if operand_ty.is_signed_integer() {
                em.b.s_div(ty_id, None, lhs_id, rhs_id)
                    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
            } else if operand_ty.is_unsigned_integer() {
                em.b.u_div(ty_id, None, lhs_id, rhs_id)
                    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
            } else {
                em.b.f_div(ty_id, None, lhs_id, rhs_id)
                    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
            }
        }
        BinOp::Rem => {
            if operand_ty.is_signed_integer() {
                em.b.s_rem(ty_id, None, lhs_id, rhs_id)
                    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
            } else if operand_ty.is_unsigned_integer() {
                em.b.u_mod(ty_id, None, lhs_id, rhs_id)
                    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
            } else {
                // Float: OpFRem (sign of result == sign of dividend, per §5.4 note).
                em.b.f_rem(ty_id, None, lhs_id, rhs_id)
                    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
            }
        }
        // Comparison ops: result type is always Bool.
        BinOp::Eq => {
            let bool_ty = em.type_id(ScalarTy::Bool);
            if operand_ty.is_bool() {
                // Bool == Bool uses OpLogicalEqual (NOT OpIEqual — see §5.4 note).
                em.b.logical_equal(bool_ty, None, lhs_id, rhs_id)
                    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
            } else if operand_ty.is_integer() {
                em.b.i_equal(bool_ty, None, lhs_id, rhs_id)
                    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
            } else {
                em.b.f_ord_equal(bool_ty, None, lhs_id, rhs_id)
                    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
            }
        }
        BinOp::Neq => {
            let bool_ty = em.type_id(ScalarTy::Bool);
            if operand_ty.is_bool() {
                em.b.logical_not_equal(bool_ty, None, lhs_id, rhs_id)
                    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
            } else if operand_ty.is_integer() {
                em.b.i_not_equal(bool_ty, None, lhs_id, rhs_id)
                    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
            } else {
                em.b.f_ord_not_equal(bool_ty, None, lhs_id, rhs_id)
                    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
            }
        }
        BinOp::Lt => {
            let bool_ty = em.type_id(ScalarTy::Bool);
            if operand_ty.is_signed_integer() {
                em.b.s_less_than(bool_ty, None, lhs_id, rhs_id)
                    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
            } else if operand_ty.is_unsigned_integer() {
                em.b.u_less_than(bool_ty, None, lhs_id, rhs_id)
                    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
            } else {
                em.b.f_ord_less_than(bool_ty, None, lhs_id, rhs_id)
                    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
            }
        }
        BinOp::LtEq => {
            let bool_ty = em.type_id(ScalarTy::Bool);
            if operand_ty.is_signed_integer() {
                em.b.s_less_than_equal(bool_ty, None, lhs_id, rhs_id)
                    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
            } else if operand_ty.is_unsigned_integer() {
                em.b.u_less_than_equal(bool_ty, None, lhs_id, rhs_id)
                    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
            } else {
                em.b.f_ord_less_than_equal(bool_ty, None, lhs_id, rhs_id)
                    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
            }
        }
        BinOp::Gt => {
            let bool_ty = em.type_id(ScalarTy::Bool);
            if operand_ty.is_signed_integer() {
                em.b.s_greater_than(bool_ty, None, lhs_id, rhs_id)
                    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
            } else if operand_ty.is_unsigned_integer() {
                em.b.u_greater_than(bool_ty, None, lhs_id, rhs_id)
                    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
            } else {
                em.b.f_ord_greater_than(bool_ty, None, lhs_id, rhs_id)
                    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
            }
        }
        BinOp::GtEq => {
            let bool_ty = em.type_id(ScalarTy::Bool);
            if operand_ty.is_signed_integer() {
                em.b.s_greater_than_equal(bool_ty, None, lhs_id, rhs_id)
                    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
            } else if operand_ty.is_unsigned_integer() {
                em.b.u_greater_than_equal(bool_ty, None, lhs_id, rhs_id)
                    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
            } else {
                em.b.f_ord_greater_than_equal(bool_ty, None, lhs_id, rhs_id)
                    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
            }
        }
    };
    let _ = ty_id; // already used above
    Ok(id)
}

// ── Short-circuit and/or diamond lowering (§5.7) ─────────────────────────────

fn emit_short_circuit(
    em: &mut BodyEmitter<'_>,
    op: ShortCircuitOp,
    lhs: &HirExpr,
    rhs: &HirExpr,
) -> Result<Word, BodyCodegenError> {
    // 1. Pre-allocate label ids for the RHS block and the merge block.
    let rhs_label: Word = em.b.id();
    let merge_label: Word = em.b.id();

    // 2. Evaluate LHS in the current block.
    let lhs_id = emit_expr(em, lhs)?;

    // 3. Capture the CURRENT block's label AFTER evaluating LHS.
    //    (LHS may itself have been a nested short-circuit that opened/closed blocks.)
    let entry_label = em.current_block_label()?;

    // 4. Emit OpSelectionMerge + OpBranchConditional.
    //    For `and`: true -> rhs, false -> merge (short-circuit with false).
    //    For `or`:  true -> merge, false -> rhs (short-circuit with true).
    em.b.selection_merge(merge_label, SelectionControl::NONE)
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;

    let (true_label, false_label) = match op {
        ShortCircuitOp::And => (rhs_label,   merge_label),
        ShortCircuitOp::Or  => (merge_label, rhs_label),
    };
    em.b.branch_conditional(lhs_id, true_label, false_label, [])
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;

    // 5. Open the RHS block with the pre-allocated label.
    em.b.begin_block(Some(rhs_label))
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;

    // 6. Evaluate RHS (may itself create nested blocks via nested short-circuit).
    let rhs_id = emit_expr(em, rhs)?;

    // 7. Capture the current block's label AFTER evaluating RHS
    //    (may differ from rhs_label if RHS contained nested short-circuit CF).
    let rhs_end_label = em.current_block_label()?;

    // 8. Branch to merge.
    em.b.branch(merge_label)
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;

    // 9. Open the merge block.
    em.b.begin_block(Some(merge_label))
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;

    // 10. Emit OpPhi.
    //     For `and`: short-circuit path (false) = entry_label, RHS path = rhs_end_label.
    //     For `or`:  short-circuit path (true)  = entry_label, RHS path = rhs_end_label.
    let bool_ty = em.type_id(ScalarTy::Bool);
    let (sc_val_id, sc_pred_label, rhs_pred_label) = match op {
        ShortCircuitOp::And => {
            // If we didn't evaluate RHS (LHS was false), the result is false.
            let false_id = em.get_const_bool(false);
            (false_id, entry_label, rhs_end_label)
        }
        ShortCircuitOp::Or => {
            // If we didn't evaluate RHS (LHS was true), the result is true.
            let true_id = em.get_const_bool(true);
            (true_id, entry_label, rhs_end_label)
        }
    };

    let phi_id = em.b.phi(
        bool_ty,
        None,
        [(rhs_id, rhs_pred_label), (sc_val_id, sc_pred_label)],
    ).map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;

    Ok(phi_id)
}

// ── Bitwise builtins (§5.5) ───────────────────────────────────────────────────

fn emit_bitwise(
    em: &mut BodyEmitter<'_>,
    op: BitwiseOp,
    args: &[HirExpr],
    result_ty: ScalarTy,
) -> Result<Word, BodyCodegenError> {
    let ty_id = em.type_id(result_ty);

    let id: Word = match op {
        BitwiseOp::Band => {
            let a = emit_expr(em, &args[0])?;
            let b = emit_expr(em, &args[1])?;
            em.b.bitwise_and(ty_id, None, a, b)
                .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
        }
        BitwiseOp::Bor => {
            let a = emit_expr(em, &args[0])?;
            let b = emit_expr(em, &args[1])?;
            em.b.bitwise_or(ty_id, None, a, b)
                .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
        }
        BitwiseOp::Bxor => {
            let a = emit_expr(em, &args[0])?;
            let b = emit_expr(em, &args[1])?;
            em.b.bitwise_xor(ty_id, None, a, b)
                .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
        }
        BitwiseOp::Bnot => {
            let a = emit_expr(em, &args[0])?;
            em.b.not(ty_id, None, a)
                .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
        }
        BitwiseOp::Shl => {
            let a = emit_expr(em, &args[0])?;
            let n = emit_expr(em, &args[1])?;
            em.b.shift_left_logical(ty_id, None, a, n)
                .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
        }
        BitwiseOp::Shr => {
            // OpShiftRightArithmetic: sign-preserving, used for SIGNED integers.
            let a = emit_expr(em, &args[0])?;
            let n = emit_expr(em, &args[1])?;
            em.b.shift_right_arithmetic(ty_id, None, a, n)
                .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
        }
        BitwiseOp::Lshr => {
            // OpShiftRightLogical: sign-stripping, used for UNSIGNED integers.
            let a = emit_expr(em, &args[0])?;
            let n = emit_expr(em, &args[1])?;
            em.b.shift_right_logical(ty_id, None, a, n)
                .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?
        }
    };
    Ok(id)
}

// ── Scalar push-constant read (M1.2) ─────────────────────────────────────────

/// Emit a load from the push-constant block for a scalar kernel parameter.
///
/// Decodes `param_position` to find the `member_index`, then emits:
///   %ptr = OpAccessChain member_ptr_ty %pc_var %member_index_const
///   %val = OpLoad ty %ptr
fn emit_scalar_param_read(
    em: &mut BodyEmitter<'_>,
    param_position: u32,
    result_ty: ScalarTy,
) -> Result<Word, BodyCodegenError> {
    let pc = em.res.push_constant
        .ok_or(BodyCodegenError::UnexpectedHir("scalar param read with no push-constant block"))?;

    // Look up member_index from scalar_params table.
    let member_index: u32 = em.res.scalar_params
        .iter()
        .find(|(pos, _, _)| *pos == param_position)
        .map(|(_, mi, _)| *mi)
        .ok_or(BodyCodegenError::UnexpectedHir("scalar param not found in scalar_params table"))?;

    let member_ptr_ty = *pc.member_ptr_ids.get(&member_index)
        .ok_or(BodyCodegenError::UnexpectedHir("member_index not in push_constant.member_ptr_ids"))?;

    let mi_const = em.get_const_int(ScalarTy::U32, member_index as u64);
    let chain_id = em.b.access_chain(member_ptr_ty, None, pc.var_id, [mi_const])
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;

    let ty_id = em.type_id(result_ty);
    let load_id = em.b.load(ty_id, None, chain_id, None, None)
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;
    Ok(load_id)
}

// ── Buffer read/write and gid (M1.2) ─────────────────────────────────────────

/// Emit `OpAccessChain` + `OpLoad` to read one element from an SSBO.
///
/// Pattern:
///   %ptr_to_elem = OpAccessChain ptr_to_elem_ty  %var  %0_const  %index
///   %result      = OpLoad        elem_ty          %ptr_to_elem
fn emit_buffer_read(
    em: &mut BodyEmitter<'_>,
    buffer_binding: u32,
    index: &HirExpr,
    result_ty: ScalarTy,
) -> Result<Word, BodyCodegenError> {
    let bindings = em.res.buffer_bindings
        .ok_or(BodyCodegenError::UnexpectedHir("BufferRead with no BufferBindings in resources"))?;
    let var_id = *bindings.var_ids.get(&buffer_binding)
        .ok_or(BodyCodegenError::UnexpectedHir("BufferRead: buffer_binding not in var_ids"))?;
    let elem_ptr_ty = *bindings.elem_ptr_ids.get(&buffer_binding)
        .ok_or(BodyCodegenError::UnexpectedHir("BufferRead: buffer_binding not in elem_ptr_ids"))?;

    // Member 0 of the SSBO struct is the runtime array; we need a constant 0.
    let u32_ty_id = em.type_id(ScalarTy::U32);
    let zero_id = em.get_const_int(ScalarTy::U32, 0);
    let index_id = emit_expr(em, index)?;

    // OpAccessChain %elem_ptr_ty %var %zero %index
    let chain_id = em.b.access_chain(elem_ptr_ty, None, var_id, [zero_id, index_id])
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;

    // OpLoad %result_ty %chain_id
    let elem_ty_id = em.type_id(result_ty);
    let load_id = em.b.load(elem_ty_id, None, chain_id, None, None)
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;
    let _ = u32_ty_id;
    Ok(load_id)
}

/// Emit `OpAccessChain` + `OpStore` to write one element to an SSBO.
///
/// Pattern:
///   %ptr_to_elem = OpAccessChain ptr_to_elem_ty  %var  %0_const  %index
///   OpStore %ptr_to_elem  %value
fn emit_buffer_write(
    em: &mut BodyEmitter<'_>,
    buffer_binding: u32,
    index: &HirExpr,
    value: &HirExpr,
) -> Result<(), BodyCodegenError> {
    let bindings = em.res.buffer_bindings
        .ok_or(BodyCodegenError::UnexpectedHir("BufferWrite with no BufferBindings in resources"))?;
    let var_id = *bindings.var_ids.get(&buffer_binding)
        .ok_or(BodyCodegenError::UnexpectedHir("BufferWrite: buffer_binding not in var_ids"))?;
    let elem_ptr_ty = *bindings.elem_ptr_ids.get(&buffer_binding)
        .ok_or(BodyCodegenError::UnexpectedHir("BufferWrite: buffer_binding not in elem_ptr_ids"))?;

    let zero_id = em.get_const_int(ScalarTy::U32, 0);
    let index_id = emit_expr(em, index)?;
    let value_id = emit_expr(em, value)?;

    let chain_id = em.b.access_chain(elem_ptr_ty, None, var_id, [zero_id, index_id])
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;
    em.b.store(chain_id, value_id, None, None)
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;
    Ok(())
}

/// Emit extraction of one component from `gl_GlobalInvocationID`.
///
/// Pattern:
///   %ptr = OpAccessChain u32_ptr_type_id %gid_var %axis_const
///   %val = OpLoad u32 %ptr
fn emit_gid_component(
    em: &mut BodyEmitter<'_>,
    axis: u32,
) -> Result<Word, BodyCodegenError> {
    let gid = em.res.gid_var
        .ok_or(BodyCodegenError::UnexpectedHir("GidBuiltin with no gid_var in resources"))?;
    let gid_var_id = gid.var_id;
    let u32_ptr_ty = gid.u32_ptr_type_id;

    let axis_id = em.get_const_int(ScalarTy::U32, axis as u64);
    let chain_id = em.b.access_chain(u32_ptr_ty, None, gid_var_id, [axis_id])
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;

    let u32_ty_id = em.type_id(ScalarTy::U32);
    let load_id = em.b.load(u32_ty_id, None, chain_id, None, None)
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;
    Ok(load_id)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use axc_parser::parse;
    use axc_hir::lower_module;
    use rspirv::spirv::Op;

    /// Compile a source snippet to SPIR-V words.
    fn compile_to_words(src: &str) -> Vec<u32> {
        use crate::emit::{emit_module, CodegenOptions};
        let (ast, lex, parse_errs) = parse(src);
        assert!(lex.is_empty(), "lex: {lex:?}");
        assert!(parse_errs.is_empty(), "parse: {parse_errs:?}");
        let (hir, hir_errs, _) = lower_module(&ast);
        assert!(hir_errs.is_empty(), "hir: {hir_errs:?}");
        emit_module(&hir, &CodegenOptions::default()).expect("codegen failed")
    }

    fn has_op(words: &[u32], target_op: Op) -> bool {
        iter_instructions(&words[5..]).any(|(opcode, _)| opcode == target_op as u16)
    }

    fn lacks_op(words: &[u32], target_op: Op) -> bool {
        !has_op(words, target_op)
    }

    fn has_capability(words: &[u32], cap_word: u32) -> bool {
        iter_instructions(&words[5..]).any(|(op, slice)| {
            op == Op::Capability as u16 && slice.len() >= 2 && slice[1] == cap_word
        })
    }

    // 1. cg_emit_i32_add_uses_opiadd
    #[test]
    fn cg_emit_i32_add_uses_opiadd() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let a: i32 = 1i32 + 2i32; return; }";
        let words = compile_to_words(src);
        assert!(has_op(&words, Op::IAdd), "expected IAdd in SPIR-V");
    }

    // 2. cg_emit_u32_div_uses_opudiv
    #[test]
    fn cg_emit_u32_div_uses_opudiv() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let a: u32 = 10u32 / 3u32; return; }";
        let words = compile_to_words(src);
        assert!(has_op(&words, Op::UDiv), "expected UDiv in SPIR-V");
        assert!(lacks_op(&words, Op::SDiv), "should not have SDiv for u32");
    }

    // 3. cg_emit_i32_div_uses_opsdiv
    #[test]
    fn cg_emit_i32_div_uses_opsdiv() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let a: i32 = 10i32 / 3i32; return; }";
        let words = compile_to_words(src);
        assert!(has_op(&words, Op::SDiv), "expected SDiv in SPIR-V");
        assert!(lacks_op(&words, Op::UDiv), "should not have UDiv for i32");
    }

    // 4. cg_emit_f32_add_uses_opfadd
    #[test]
    fn cg_emit_f32_add_uses_opfadd() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let a: f32 = 1.0f32 + 2.0f32; return; }";
        let words = compile_to_words(src);
        assert!(has_op(&words, Op::FAdd), "expected FAdd in SPIR-V");
    }

    // 5. cg_emit_f64_mul_uses_opfmul_and_float64_cap
    #[test]
    fn cg_emit_f64_mul_uses_opfmul_and_float64_cap() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let a: f64 = 1.0f64 * 2.0f64; return; }";
        let words = compile_to_words(src);
        assert!(has_op(&words, Op::FMul), "expected FMul in SPIR-V");
        // Float64 capability = 10 (per SPIR-V spec §3.31 Capability)
        assert!(has_capability(&words, 10), "expected Float64 capability (10)");
    }

    // 6. cg_emit_i64_add_uses_int64_cap
    #[test]
    fn cg_emit_i64_add_uses_int64_cap() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let a: i64 = 1i64 + 2i64; return; }";
        let words = compile_to_words(src);
        // Int64 capability = 11 (per SPIR-V spec)
        assert!(has_capability(&words, 11), "expected Int64 capability (11)");
    }

    // 7. cg_emit_u32_lt_uses_opulessthan
    #[test]
    fn cg_emit_u32_lt_uses_opulessthan() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let b: bool = 1u32 < 2u32; return; }";
        let words = compile_to_words(src);
        assert!(has_op(&words, Op::ULessThan), "expected ULessThan");
    }

    // 8. cg_emit_i32_lt_uses_opslessthan
    #[test]
    fn cg_emit_i32_lt_uses_opslessthan() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let b: bool = 1i32 < 2i32; return; }";
        let words = compile_to_words(src);
        assert!(has_op(&words, Op::SLessThan), "expected SLessThan");
    }

    // 9. cg_emit_f32_lt_uses_opfordlessthan
    #[test]
    fn cg_emit_f32_lt_uses_opfordlessthan() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let b: bool = 1.0f32 < 2.0f32; return; }";
        let words = compile_to_words(src);
        assert!(has_op(&words, Op::FOrdLessThan), "expected FOrdLessThan");
    }

    // 10. cg_short_circuit_and_emits_diamond
    #[test]
    fn cg_short_circuit_and_emits_diamond() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let b: bool = true and false; return; }";
        let words = compile_to_words(src);
        let sel_merge_count = iter_instructions(&words[5..]).filter(|(op, _)| *op == Op::SelectionMerge as u16).count();
        let br_cond_count = iter_instructions(&words[5..]).filter(|(op, _)| *op == Op::BranchConditional as u16).count();
        let phi_count = iter_instructions(&words[5..]).filter(|(op, _)| *op == Op::Phi as u16).count();
        assert_eq!(sel_merge_count, 1, "expected 1 SelectionMerge");
        assert_eq!(br_cond_count, 1, "expected 1 BranchConditional");
        assert_eq!(phi_count, 1, "expected 1 Phi");
        assert!(lacks_op(&words, Op::LogicalAnd), "LogicalAnd must NOT be emitted for `and`");
    }

    // 11. cg_scalar_type_cache_dedups
    #[test]
    fn cg_scalar_type_cache_dedups() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let a: i32 = 1i32; let b: i32 = 2i32; return; }";
        let words = compile_to_words(src);
        // Count OpTypeInt 32 1 occurrences — should be exactly 1.
        let type_int_count = iter_instructions(&words[5..]).filter(|(op, _)| *op == Op::TypeInt as u16).count();
        assert_eq!(type_int_count, 1, "ScalarTypeCache should dedup: expected 1 OpTypeInt, got {type_int_count}");
    }

    // 12. cg_function_var_prelude_in_first_block
    #[test]
    fn cg_function_var_prelude_in_first_block() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let a: i32 = 1i32; return; }";
        let words = compile_to_words(src);
        // In the instruction stream, all OpVariable must precede any arithmetic.
        let body_words: &[u32] = &words[5..];
        let mut seen_non_var: bool = false;
        let mut var_after_non_var: bool = false;
        for (opcode, _) in iter_instructions(body_words) {
            if opcode == Op::Variable as u16 {
                if seen_non_var {
                    var_after_non_var = true;
                }
            } else if opcode != Op::Label as u16
                && opcode != Op::Capability as u16
                && opcode != Op::MemoryModel as u16
                && opcode != Op::EntryPoint as u16
                && opcode != Op::ExecutionMode as u16
                && opcode != Op::TypeVoid as u16
                && opcode != Op::TypeFunction as u16
                && opcode != Op::TypeInt as u16
                && opcode != Op::TypeFloat as u16
                && opcode != Op::TypeBool as u16
                && opcode != Op::TypePointer as u16
                && opcode != Op::Constant as u16
                && opcode != Op::ConstantTrue as u16
                && opcode != Op::ConstantFalse as u16
                && opcode != Op::Function as u16
                && opcode != Op::FunctionEnd as u16
            {
                seen_non_var = true;
            }
        }
        assert!(!var_after_non_var, "OpVariable must precede all non-var instructions in first block");
    }

    // 13. cg_emit_bool_eq_uses_oplogicalequal
    #[test]
    fn cg_emit_bool_eq_uses_oplogicalequal() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let b: bool = true == false; return; }";
        let words = compile_to_words(src);
        assert!(has_op(&words, Op::LogicalEqual), "expected LogicalEqual for bool==bool");
        assert!(lacks_op(&words, Op::IEqual), "IEqual must NOT be used for bool==bool");
    }

    // 14. cg_emit_bool_neq_uses_oplogicalnotequal
    #[test]
    fn cg_emit_bool_neq_uses_oplogicalnotequal() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let b: bool = true != false; return; }";
        let words = compile_to_words(src);
        assert!(has_op(&words, Op::LogicalNotEqual), "expected LogicalNotEqual for bool!=bool");
    }

    // 15. cg_emit_f32_rem_uses_opfrem (AT-123)
    #[test]
    fn cg_emit_f32_rem_uses_opfrem() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let r: f32 = 5.0f32 % -3.0f32; return; }";
        let words = compile_to_words(src);
        assert!(has_op(&words, Op::FRem), "expected FRem (not FMod) for float %");
        assert!(lacks_op(&words, Op::FMod), "FMod must NOT be used for float %");
    }

    // 16. cg_nested_short_circuit_and_then_or_phi_predecessors (AT-117)
    #[test]
    fn cg_nested_short_circuit_and_then_or_phi_predecessors() {
        // (a and b) or c — outer `or`'s false-path phi predecessor should be the
        // inner `and`'s merge block, NOT the function's first block.
        let src = "@kernel @workgroup(1,1,1) fn k() -> void {
            let a: bool = true;
            let b: bool = false;
            let c: bool = true;
            let r: bool = a and b or c;
            return;
        }";
        let words = compile_to_words(src);
        // spirv-val will catch malformed phi predecessors; we also count for shape.
        let phi_count = iter_instructions(&words[5..]).filter(|(op, _)| *op == Op::Phi as u16).count();
        assert_eq!(phi_count, 2, "expected 2 Phi instructions (one per and/or diamond): got {phi_count}");
    }

    // 17. cg_nested_short_circuit_or_inside_and_phi_predecessors (AT-118)
    #[test]
    fn cg_nested_short_circuit_or_inside_and_phi_predecessors() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void {
            let a: bool = true;
            let b: bool = false;
            let c: bool = true;
            let r: bool = a or (b and c);
            return;
        }";
        let words = compile_to_words(src);
        let phi_count = iter_instructions(&words[5..]).filter(|(op, _)| *op == Op::Phi as u16).count();
        assert_eq!(phi_count, 2, "expected 2 Phi instructions");
    }

    // 18. cg_no_int64_or_float64_cap_on_i32_f32_only_kernel (AT-119)
    #[test]
    fn cg_no_int64_or_float64_cap_on_i32_f32_only_kernel() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let a: i32 = 1i32; let b: f32 = 2.0f32; return; }";
        let words = compile_to_words(src);
        // Int64 cap = 11, Float64 cap = 10
        assert!(!has_capability(&words, 11), "must NOT emit Int64 cap for i32-only kernel");
        assert!(!has_capability(&words, 10), "must NOT emit Float64 cap for f32-only kernel");
    }

    // 19. cg_scalar_demo_deterministic
    #[test]
    fn cg_scalar_demo_deterministic() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let a: i32 = 1i32 + 2i32; return; }";
        let words1 = compile_to_words(src);
        let words2 = compile_to_words(src);
        assert_eq!(words1, words2, "codegen must be deterministic");
    }

    // Utility: iterate instructions in word stream (skip 5-word header).
    fn iter_instructions(words: &[u32]) -> impl Iterator<Item = (u16, &[u32])> {
        IterInst { words, cursor: 0 }
    }
    struct IterInst<'a> { words: &'a [u32], cursor: usize }
    impl<'a> Iterator for IterInst<'a> {
        type Item = (u16, &'a [u32]);
        fn next(&mut self) -> Option<Self::Item> {
            if self.cursor >= self.words.len() { return None; }
            let hdr = self.words[self.cursor];
            let wc = (hdr >> 16) as usize;
            let op = (hdr & 0xFFFF) as u16;
            if wc == 0 || self.cursor + wc > self.words.len() { return None; }
            let slice = &self.words[self.cursor..self.cursor + wc];
            self.cursor += wc;
            Some((op, slice))
        }
    }
}

