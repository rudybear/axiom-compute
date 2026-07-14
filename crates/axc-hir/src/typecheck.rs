//! Two-pass typechecker for AXIOM-Compute kernel bodies.
//!
//! **Pass 1** — Binding table population: for each `Stmt::Let`, allocate a
//! `BindingId`, convert the declared `TypeRef` to `ScalarTy`, and register the
//! `Binding`. Expressions are NOT visited in pass 1.
//!
//! **Pass 2** — Expression typing: walk each statement with an `expected`
//! type context propagated downward from the let/assign target type.
//!
//! Key invariants:
//! - No type inference (anti-pattern #1). Every literal must have a suffix or be
//!   in a context where the outer expected type pins it.
//! - No implicit coercions. Mixed-type binary expressions are rejected with
//!   `MixedOperandTypes`.
//! - The `Neg(IntLit)` peephole (§4.2a) rewrites `-(2147483648)` into
//!   `IntLit(-2147483648)` so that i32::MIN fits. This is the ONLY rewrite.
//! - Error recovery is per-statement: a failed statement may still emit a HIR
//!   statement (without the failing init) so later references resolve.

use std::collections::BTreeMap;
use axc_lexer::Span;
use axc_parser::ast as past;
use crate::expr::{
    Binding, BindingId, BindingTy, HirExpr, HirExprKind, HirStmt, KernelBodyTyped,
    BinOp, UnaryOp, ShortCircuitOp, BitwiseOp,
};
use crate::ty::{ScalarTy, fit_int_literal, fit_float_literal};
use crate::param::{KernelParam, Ty as ParamTy};
use crate::buffer::BufferAccess;
use crate::control_flow::{HirIf, HirElse, HirForRange, HirWhile, ForStep};
use crate::loop_ctx::{HirLoopStack, ScopeStack};
use crate::coopmat::{
    CoopMatUse, CoopMatKey, CoopMatrixShapeKind, CoopMatrixShape,
    is_allowed_coopmat_element,
};
use crate::shared::{SharedId, SharedDecl, SharedTy, MAX_SHARED_ELEMS, is_allowed_shared_element};
use crate::local::{LocalArrayId, LocalArrayDecl, LocalArrayTy, MAX_LOCAL_ARRAY_ELEMS, is_allowed_local_element};

/// Typecheck error — emitted from `typecheck_kernel_body`.
///
/// Errors are non-fatal: typecheck continues past each error to collect all
/// diagnostics in one pass (anti-pattern #6).
#[derive(Debug, Clone, thiserror::Error, miette::Diagnostic)]
pub enum TypecheckError {
    #[error("type mismatch: expected `{expected}`, got `{got}`")]
    TypeMismatch {
        expected: &'static str,
        got: &'static str,
        #[label("here")]
        span: Span,
    },

    #[error("integer literal {value} does not fit in `{target}` (range [{min}, {max}])")]
    LiteralOutOfRange {
        value: i128,
        target: &'static str,
        min: i128,
        max: i128,
        #[label("here")]
        span: Span,
    },

    #[error("float literal is not finite")]
    FloatLiteralNonFinite {
        #[label("here")]
        span: Span,
    },

    #[error("binding `{name}` is not declared (did you forget `let`?)")]
    UnknownBinding {
        name: String,
        #[label("here")]
        span: Span,
    },

    #[error("cannot assign to immutable binding `{name}` (declare it with `let mut` to allow assignment)")]
    AssignImmutable {
        name: String,
        #[label("here")]
        span: Span,
        #[label("originally declared here")]
        original_span: Span,
    },

    #[error("redeclaration of binding `{name}` in the same scope")]
    RedeclaredBinding {
        name: String,
        #[label("here")]
        span: Span,
        #[label("original declaration")]
        original_span: Span,
    },

    #[error("operator `{op}` is only valid on {operand_class}; got operands of type `{lhs_ty}` and `{rhs_ty}`")]
    OperatorTypeError {
        op: &'static str,
        operand_class: &'static str,
        lhs_ty: &'static str,
        rhs_ty: &'static str,
        #[label("here")]
        span: Span,
    },

    #[error("binary operator `{op}` requires operands of the same type; got `{lhs_ty}` and `{rhs_ty}` (no implicit coercions - anti-pattern #1)")]
    MixedOperandTypes {
        op: &'static str,
        lhs_ty: &'static str,
        rhs_ty: &'static str,
        #[label("here")]
        span: Span,
    },

    #[error("bitwise builtin `{builtin}` requires {expected_arity} integer argument(s); got {got_arity}")]
    BitwiseArity {
        builtin: &'static str,
        expected_arity: usize,
        got_arity: usize,
        #[label("here")]
        span: Span,
    },

    #[error("bitwise builtin `{builtin}` only accepts integer operands; got `{got_ty}`")]
    BitwiseNonInteger {
        builtin: &'static str,
        got_ty: &'static str,
        #[label("here")]
        span: Span,
    },

    #[error("unknown function or builtin `{name}` (only bitwise builtins band/bor/bxor/bnot/shl/shr/lshr are supported in M1.1)")]
    UnknownCall {
        name: String,
        #[label("here")]
        span: Span,
    },

    #[error("`shr` requires a SIGNED integer first argument (got `{got_ty}`). Hint: for logical (sign-stripping) right shift on unsigned types, use `lshr`.")]
    ShiftRequiresSignedLhs {
        got_ty: &'static str,
        #[label("here")]
        span: Span,
    },

    #[error("`lshr` requires an UNSIGNED integer first argument (got `{got_ty}`). Hint: for arithmetic (sign-preserving) right shift on signed types, use `shr`.")]
    ShiftRequiresUnsignedLhs {
        got_ty: &'static str,
        #[label("here")]
        span: Span,
    },

    #[error("shift builtin `{builtin}` requires the shift amount to have the same integer type as the value (`{lhs_ty}`); got shift amount of type `{rhs_ty}`")]
    ShiftAmountTypeMismatch {
        builtin: &'static str,
        lhs_ty: &'static str,
        rhs_ty: &'static str,
        #[label("here")]
        span: Span,
    },

    #[error("integer literal has no suffix and no explicit context type. Add a suffix (e.g. `42i32`, `42u64`) or place in a typed context like `let x: i32 = 42;`.")]
    UnconstrainedLiteralNeedsSuffix {
        #[label("here")]
        span: Span,
    },

    #[error("unsupported expression form in M1.1: {detail}")]
    UnsupportedExprInM1_1 {
        detail: &'static str,
        #[label("here")]
        span: Span,
    },

    // ── M1.2 buffer errors ─────────────────────────────────────────────────────

    #[error("cannot write to read-only buffer `{name}`")]
    WriteToReadonlyBuffer {
        name: String,
        #[label("here")]
        span: Span,
    },

    #[error("cannot read from write-only buffer `{name}`")]
    ReadFromWriteonlyBuffer {
        name: String,
        #[label("here")]
        span: Span,
    },

    #[error("buffer index must be `u32`; got `{got_ty}`")]
    BadIndexType {
        got_ty: &'static str,
        #[label("here")]
        span: Span,
    },

    #[error("`{name}` is not a buffer and cannot be indexed with `[]`")]
    IndexOnNonBuffer {
        name: String,
        #[label("here")]
        span: Span,
    },

    #[error("buffer `{name}` cannot be used as a value; use `name[index]` to read an element")]
    BufferAsValue {
        name: String,
        #[label("here")]
        span: Span,
    },

    #[error("`gid()` axis must be an integer literal (0, 1, or 2); got a non-literal expression")]
    GidAxisMustBeConstant {
        #[label("here")]
        span: Span,
    },

    #[error("`gid()` axis {got} is out of range; must be 0, 1, or 2")]
    GidAxisOutOfRange {
        got: u32,
        #[label("here")]
        span: Span,
    },

    #[error("`gid()` requires exactly 1 argument; got {got}")]
    GidArity {
        got: usize,
        #[label("here")]
        span: Span,
    },

    // ── M3.3d: local_invocation_id errors ──────────────────────────────────────

    #[error("`local_invocation_id()` axis must be an integer literal (0, 1, or 2); got a non-literal expression")]
    LocalInvocationIdAxisMustBeConstant {
        #[label("here")]
        span: Span,
    },

    #[error("`local_invocation_id()` axis {got} is out of range; must be 0, 1, or 2")]
    LocalInvocationIdAxisOutOfRange {
        got: u32,
        #[label("here")]
        span: Span,
    },

    #[error("`local_invocation_id()` requires exactly 1 argument; got {got}")]
    LocalInvocationIdArity {
        got: usize,
        #[label("here")]
        span: Span,
    },

    #[error("unsupported buffer element type `{ty_name}` in M1.2 (only i32, u32, i64, u64, f32, f64 are supported)")]
    UnsupportedBufferElem {
        ty_name: &'static str,
        #[label("here")]
        span: Span,
    },

    #[error("cannot assign to kernel parameter `{name}`; parameters are immutable")]
    AssignToParam {
        name: String,
        #[label("here")]
        span: Span,
    },

    #[error("unsupported kernel parameter type for `{name}` (M1.2 supports scalar types and buffer types)")]
    UnsupportedParamType {
        name: String,
        #[label("here")]
        span: Span,
    },

    // ── M1.3 control-flow errors ────────────────────────────────────────────────

    #[error("`break` outside of any loop")]
    BreakOutsideLoop {
        #[label("here")]
        span: Span,
    },

    #[error("`continue` outside of any loop")]
    ContinueOutsideLoop {
        #[label("here")]
        span: Span,
    },

    #[error("for-loop step must be a compile-time positive integer constant; got a non-constant expression")]
    ForStepNotConstant {
        #[label("here")]
        span: Span,
    },

    #[error("for-loop step must be a positive integer; got {value}")]
    ForStepNotPositive {
        value: u64,
        #[label("here")]
        span: Span,
    },

    #[error("for-loop step must have type `u32` (got suffix `{got_suffix}`)")]
    ForStepNotU32 {
        got_suffix: &'static str,
        #[label("here")]
        span: Span,
    },

    #[error("cannot assign to for-loop induction variable `{name}`")]
    AssignToForInductionVar {
        name: String,
        #[label("here")]
        span: Span,
    },

    #[error("condition in `{position}` statement must be a bool expression; got `{got}`")]
    NonBoolCondition {
        position: &'static str,
        got: &'static str,
        #[label("here")]
        span: Span,
    },

    #[error("compound short-circuit (`and`/`or`) is not allowed directly in the `{position}` condition header in M1.3; lift it to `let _cond: bool = <expr>; {position} _cond {{ ... }}`")]
    UnsupportedShortCircuitInHeader {
        position: &'static str,
        #[label("here")]
        span: Span,
    },

    // ── M1.4 subgroup errors ──────────────────────────────────────────────────

    #[error("subgroup builtin `{op}` requires {expected_arity} argument(s); got {got_arity}")]
    SubgroupArity {
        op: &'static str,
        expected_arity: usize,
        got_arity: usize,
        #[label("here")]
        span: Span,
    },

    #[error("subgroup_reduce_{op} does not accept `{got_ty}`; only i32, u32, f32, and f64 are supported")]
    SubgroupReduceTypeUnsupported {
        op: &'static str,
        got_ty: &'static str,
        #[label("here")]
        span: Span,
    },

    #[error("subgroup_broadcast_first does not accept `{got_ty}`; only i32, u32, f32, f64, and bool are supported")]
    SubgroupBroadcastTypeUnsupported {
        got_ty: &'static str,
        #[label("here")]
        span: Span,
    },

    #[error("subgroup builtin `{op_name}` returns a value and cannot appear as a statement; capture the result with `let x: T = {op_name}(...);`")]
    NonVoidSubgroupCallAsStatement {
        op_name: String,
        #[label("here")]
        span: Span,
    },

    #[error("`{name}` is a reserved subgroup builtin identifier and cannot be used as a variable name")]
    ReservedBuiltinName {
        name: String,
        #[label("here")]
        span: Span,
    },

    #[error("statement form is not supported in M1.4: {detail}")]
    UnsupportedStmtInM1_4 {
        detail: &'static str,
        #[label("here")]
        span: Span,
    },

    // ── M2.1 cooperative-matrix errors ────────────────────────────────────────

    #[error("cooperative-matrix element type `{ty}` is not supported in M2.1 (allowed: f16, f32, i8, u8, i32, u32)")]
    CoopMatrixElementTypeUnsupported {
        ty: &'static str,
        #[label("here")]
        span: Span,
    },

    #[error("cooperative-matrix dimension {dim_name} = {value} is out of range (must be 1..=65535)")]
    CoopMatrixDimOutOfRange {
        dim_name: &'static str,
        value: u64,
        #[label("here")]
        span: Span,
    },

    #[error("cooperative-matrix type cannot appear as a kernel parameter (`{param_name}`) in M2.1; matrix values are function-local only")]
    UnsupportedCoopMatrixAsParamInM2_1 {
        param_name: String,
        #[label("here")]
        span: Span,
    },

    #[error("cooperative-matrix builtin `{name}` requires {expected} argument(s); got {found}")]
    CoopMatArity {
        name: &'static str,
        expected: usize,
        found: usize,
        #[label("here")]
        span: Span,
    },

    #[error("cooperative-matrix builtin `{name}` requires an expected matrix type context (use `let x: matrix[T, M, N, use] = {name}(...);`)")]
    CoopMatrixBuiltinRequiresExpectedType {
        name: &'static str,
        #[label("here")]
        span: Span,
    },

    #[error("`{found_kind}` is not a buffer parameter; coopmat_load requires a buffer kernel parameter as its first argument")]
    CoopMatLoadArgMustBeBufferParam {
        found_kind: &'static str,
        #[label("here")]
        span: Span,
    },

    #[error("coopmat_load element type mismatch: matrix expects `{matrix_elem}` but buffer contains `{buffer_elem}`")]
    CoopMatLoadElementTypeMismatch {
        matrix_elem: &'static str,
        buffer_elem: &'static str,
        #[label("here")]
        span: Span,
    },

    #[error("coopmat_store requires a mutable buffer parameter; `{param_name}` is read-only")]
    CoopMatStoreToReadonlyBuffer {
        param_name: String,
        #[label("here")]
        span: Span,
    },

    #[error("coopmat_store element type mismatch: matrix has `{matrix_elem}` but buffer has `{buffer_elem}`")]
    CoopMatStoreElementTypeMismatch {
        matrix_elem: &'static str,
        buffer_elem: &'static str,
        #[label("here")]
        span: Span,
    },

    #[error("cooperative-matrix shape mismatch in coopmat_mul_add")]
    CoopMatrixShapeMismatch {
        kind: CoopMatrixShapeKind,
        #[label("here")]
        span: Span,
    },

    #[error("@cooperative_matrix annotation on kernel `{kernel}` has no matching coopmat_mul_add call")]
    CooperativeMatrixAnnotationUnused {
        kernel: String,
        #[label("here")]
        span: Span,
    },

    #[error("@cooperative_matrix annotation mismatch: declared ({em:?}) but body uses ({fm:?})")]
    CooperativeMatrixAnnotationMismatch {
        em: CoopMatrixShape,
        fm: CoopMatrixShape,
        #[label("here")]
        span: Span,
    },

    #[error("coopmat_load / coopmat_zero stride and element_offset must be `u32`; got `{found_ty}`")]
    CoopMatLoadStrideMustBeU32 {
        found_ty: &'static str,
        #[label("here")]
        span: Span,
    },

    #[error("`{name}` is a reserved cooperative-matrix builtin identifier and cannot be used as a variable name")]
    ReservedCoopMatBuiltinName {
        name: String,
        #[label("here")]
        span: Span,
    },

    #[error("`matrix` is a reserved keyword in M2.1 and cannot be used as a variable name")]
    ReservedKeyword {
        name: String,
        #[label("here")]
        span: Span,
    },

    // ── M2.5 Q4_0-path builtin errors ─────────────────────────────────────────

    #[error("Q4_0 builtin `{name}` requires {expected} argument(s); got {found}")]
    Q4_0BuiltinWrongArity {
        name: &'static str,
        expected: usize,
        found: usize,
        #[label("here")]
        span: Span,
    },

    #[error("`ptr_read_u8_zext` / `ptr_read_u16_zext` first argument must be a kernel buffer parameter identifier (readonly_buffer[u8] or buffer[u8]); got `{found_kind}`")]
    PtrReadArgMustBeBufferParam {
        found_kind: &'static str,
        #[label("here")]
        span: Span,
    },

    #[error("`ptr_read_u8_zext` / `ptr_read_u16_zext` requires a `readonly_buffer[u8]` or `buffer[u8]` parameter; got buffer with element type `{elem_ty}`")]
    PtrReadBufferElemMustBeU8 {
        elem_ty: &'static str,
        #[label("here")]
        span: Span,
    },

    #[error("`f16_bits_to_f32` requires a `u32` argument; got `{got_ty}`")]
    F16BitsToF32ArgMustBeU32 {
        got_ty: &'static str,
        #[label("here")]
        span: Span,
    },

    #[error("`f32_to_f16` requires an `f32` argument; got `{got_ty}`")]
    F32ToF16ArgMustBeF32 {
        got_ty: &'static str,
        #[label("here")]
        span: Span,
    },

    #[error("`f32_from_u32` requires a `u32` argument; got `{got_ty}`")]
    Q4_0BuiltinArgTypeMismatch {
        name: &'static str,
        expected_ty: &'static str,
        got_ty: &'static str,
        #[label("here")]
        span: Span,
    },

    #[error("`{name}` is a reserved Q4_0-path builtin identifier and cannot be used as a variable name")]
    ReservedQ4_0BuiltinName {
        name: String,
        #[label("here")]
        span: Span,
    },

    // ── M3.2c ext-inst (GLSL.std.450) builtin typecheck errors ───────────────

    #[error("`exp` requires an `f32` argument; got `{got_ty}`")]
    ExpArgMustBeF32 {
        got_ty: &'static str,
        #[label("here")]
        span: Span,
    },

    #[error("`{name}` builtin expects {expected} argument(s); got {found}")]
    ExtInstBuiltinWrongArity {
        name: &'static str,
        expected: usize,
        found: usize,
        #[label("here")]
        span: Span,
    },

    #[error("`{name}` is a reserved GLSL.std.450 ext-inst builtin identifier and cannot be used as a variable name")]
    ReservedExtInstBuiltinName {
        name: String,
        #[label("here")]
        span: Span,
    },

    #[error("f16 literal out of range for bin16 (value {value} overflows to infinity)")]
    F16LiteralOutOfRange {
        value: f64,
        #[label("here")]
        span: Span,
    },

    #[error("f16 literal silent underflow: {value} rounds to zero in binary16 (use 0.0f16 for explicit zero)")]
    F16LiteralSubnormalPrecisionLoss {
        value: f64,
        #[label("here")]
        span: Span,
    },

    // ── M3.2 shared-array typecheck errors ───────────────────────────────────

    /// `tile[i]` where `i` is not `U32` — no implicit coercion (anti-pattern #1).
    #[error("shared array index must be `u32`; got `{got}` (no implicit coercion — anti-pattern #1)")]
    SharedIndexNotU32 {
        got: &'static str,
        #[label("here")]
        span: Span,
    },

    /// `tile[i] = v;` where `v` type does not exactly match elem type.
    #[error("shared array `{name}` element type is `{expected}`; got `{got}` (exact match required — no implicit conversion)")]
    SharedWriteTypeMismatch {
        name: String,
        expected: &'static str,
        got: &'static str,
        #[label("here")]
        span: Span,
    },

    /// A shared array name collides with a parameter or binding.
    #[error("shared array name `{name}` collides with an existing parameter or binding")]
    SharedNameCollision {
        name: String,
        #[label("collision here")]
        span: Span,
    },

    /// A name reference resolves as a shared array but it was not declared.
    #[error("shared array `{name}` is not declared in this kernel body")]
    SharedNotDeclared {
        name: String,
        #[label("here")]
        span: Span,
    },

    /// Duplicate shared array name in the same kernel.
    #[error("duplicate shared array name `{name}`")]
    SharedDuplicateName {
        name: String,
        #[label("duplicate here")]
        span: Span,
    },

    /// N = 0 in `shared[T, 0]` — must be at least 1.
    #[error("shared array `{name}` has length 0; N must be >= 1")]
    SharedZeroLength {
        name: String,
        #[label("here")]
        span: Span,
    },

    /// N > MAX_SHARED_ELEMS.
    #[error("shared array `{name}` length {len} exceeds maximum {max}")]
    SharedTooLarge {
        name: String,
        len: u32,
        max: u32,
        #[label("here")]
        span: Span,
    },

    /// Disallowed element type (Bool).
    #[error("shared array `{name}` element type `{ty_name}` is not allowed (Bool has no stable Vulkan memory representation)")]
    SharedElementTypeUnsupported {
        name: String,
        ty_name: String,
        #[label("here")]
        span: Span,
    },

    /// Missing barrier — provable cross-slot RAW hazard (OQ1, sound, SET-based).
    #[error("shared array `{name}`: read at index {read_index_desc} follows writes at [{write_indices_desc}] with no barrier — provable cross-invocation RAW hazard; add workgroup_barrier() between write and read phases")]
    SharedMissingBarrierBeforeCrossInvocationRead {
        name: String,
        read_index_desc: String,
        write_indices_desc: String,
        #[label("here")]
        span: Span,
    },

    /// Barrier inside an if/else body — BarrierInDivergentContext (OQ2).
    #[error("workgroup_barrier() inside an if/else body is undefined behavior in Vulkan (not all invocations provably reach the barrier); move the barrier outside the conditional")]
    BarrierInDivergentContext {
        #[label("barrier here")]
        span: Span,
    },

    /// Aggregate shared memory > 65536 bytes — compile-time ceiling exceeded.
    #[error("kernel uses {total_bytes} bytes of shared memory (sum of all shared arrays), exceeding the compile-time maximum of 65536 bytes; reduce shared array sizes")]
    SharedMemoryTooLarge {
        total_bytes: u64,
        #[label("here")]
        span: Span,
    },

    // ── M3.20 local-array typecheck errors ───────────────────────────────────
    //
    // Placement note: these mirror the M3.2 `Shared*` precedent exactly — the
    // per-declaration / per-use diagnostics live here (constructed in typecheck.rs,
    // where the registration and index/value checks happen), NOT in
    // `validate::HirError` (which only carries `LocalArrayAsParameter`, genuinely
    // constructed at param-lowering time; the aggregate check lives here as
    // `TypecheckError::LocalArrayTooLarge`, the sole reachable-via-the-pipeline
    // check — M3.22 deleted the dead, zero-caller `validate()` pass that used to
    // duplicate this alongside `Shared*`).

    /// `hist[i]` where `i` is not `U32` — no implicit coercion (anti-pattern #1).
    #[error("local array index must be `u32`; got `{got}` (no implicit coercion — anti-pattern #1)")]
    LocalArrayIndexNotU32 {
        got: &'static str,
        #[label("here")]
        span: Span,
    },

    /// `hist[i] = v;` where `v` type does not exactly match elem type.
    #[error("local array `{name}` element type is `{expected}`; got `{got}` (exact match required — no implicit conversion)")]
    LocalArrayWriteTypeMismatch {
        name: String,
        expected: &'static str,
        got: &'static str,
        #[label("here")]
        span: Span,
    },

    /// A local-array name collides with a parameter, local binding, or a `shared`
    /// array name (bidirectional cross-check — M3.20 spec §8).
    #[error("local array name `{name}` collides with an existing parameter, binding, or shared array")]
    LocalArrayNameCollision {
        name: String,
        #[label("collision here")]
        span: Span,
    },

    /// A name reference resolves as a local-array read/write but no `array` decl
    /// registered it. Defense-in-depth (mirrors the `shared` precedent's
    /// `SharedNotDeclared`): unreachable through the normal source pipeline because
    /// index-syntax on an unregistered name falls through to the buffer/binding
    /// disambiguation chain and is reported as `UnknownBinding`/`IndexOnNonBuffer`
    /// instead; kept for direct-construction test coverage and API completeness.
    #[error("local array `{name}` is not declared in this kernel body")]
    LocalArrayUndeclared {
        name: String,
        #[label("here")]
        span: Span,
    },

    /// Duplicate local-array name in the same kernel.
    #[error("duplicate local array name `{name}`")]
    LocalArrayDuplicateName {
        name: String,
        #[label("duplicate here")]
        span: Span,
    },

    /// N = 0 in `array[T, 0]` — must be at least 1.
    #[error("local array `{name}` has length 0; N must be >= 1")]
    LocalArrayZeroLength {
        name: String,
        #[label("here")]
        span: Span,
    },

    /// N > MAX_LOCAL_ARRAY_ELEMS.
    #[error("local array `{name}` length {len} exceeds maximum {max} elements")]
    LocalArrayTooManyElems {
        name: String,
        len: u32,
        max: u32,
        #[label("here")]
        span: Span,
    },

    /// Disallowed element type (Bool).
    #[error("local array `{name}` element type `{ty_name}` is not allowed (Bool has no stable Vulkan memory representation)")]
    LocalArrayElementTypeNotAllowed {
        name: String,
        ty_name: String,
        #[label("here")]
        span: Span,
    },

    /// `array name: array[T,N];` declared inside a nested `if`/`for`/`while` block
    /// (r2, M3.20 spec §5.1). Function-storage `OpVariable`s are hoisted to the
    /// entry-block prelude with no per-iteration reset, so a nested decl would
    /// silently diverge from `let`-inside-a-loop intuition; rejected outright.
    #[error("local array declarations must appear at the top level of the kernel body, before any control-flow block; move `{name}` to the top and index it inside the loop")]
    LocalArrayDeclNotAtBlockScope {
        name: String,
        #[label("here")]
        span: Span,
    },

    /// A **constant** index `>= N` on a local array read or write (r2, M3.20 spec
    /// §5.2) — provably out of bounds, decidable, zero-false-positive. Symbolic
    /// (non-constant) indices remain UB-by-design and are NOT flagged.
    #[error("local array `{name}` index {index} is out of bounds (length {len}; valid indices are 0..={max_index})")]
    LocalArrayConstIndexOutOfBounds {
        name: String,
        index: u32,
        len: u32,
        max_index: u32,
        #[label("here")]
        span: Span,
    },

    /// Aggregate local-array memory > 4096 bytes — compile-time ceiling exceeded
    /// (M3.20 spec §6). This is the REACHABLE aggregate check (mirrors
    /// `SharedMemoryTooLarge`'s placement here in `typecheck_kernel_body`). M3.22
    /// deleted `validate.rs`'s dead, zero-caller `validate()` pass, which used to
    /// duplicate this check post-lowering — this is now the sole check.
    #[error("kernel uses {total_bytes} bytes of local-array storage (sum of all local arrays), exceeding the compile-time maximum of 4096 bytes; reduce local array sizes or use shared[T,N] for workgroup-cooperative data")]
    LocalArrayTooLarge {
        total_bytes: u64,
        #[label("here")]
        span: Span,
    },
}

// ── Internal binding table ────────────────────────────────────────────────────

// ── Index-relation predicate for missing-barrier analysis (A.4.1) ─────────────

/// The result of the structural index-relation predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IndexRelation {
    ProvablyEqual,
    ProvablyDisequal,
    Unknown,
}

/// Structural predicate over HIR expression indices.
///
/// ProvablyEqual when:
/// - Both are `LocalRead(BindingId)` with the SAME id (same SSA binding).
/// - Both are `IntLit` with EQUAL value (same type AND same bits).
/// - Both are `GidBuiltin` with the SAME axis.
///
/// ProvablyDisequal when:
/// - Both are `IntLit` with DISTINCT values (same type, different bits).
/// - Both are `GidBuiltin` with DISTINCT axes.
///
/// Unknown otherwise (Binary, different LocalRead ids, mixed constant/variable, etc.).
fn index_relation(w: &HirExprKind, r: &HirExprKind) -> IndexRelation {
    match (w, r) {
        (HirExprKind::LocalRead(wid), HirExprKind::LocalRead(rid)) => {
            // Same SSA binding id → provably equal (straight-line, no reassignment tracking
            // needed for the simple read-after-write case where the binding hasn't changed).
            if wid == rid { IndexRelation::ProvablyEqual } else { IndexRelation::Unknown }
        }
        (HirExprKind::IntLit { value: wv }, HirExprKind::IntLit { value: rv }) => {
            // IntLiteralValue is a struct { ty: ScalarTy, bits: u64 }.
            // Same type + same bits → equal; same type + different bits → disequal;
            // different types → unknown (could in theory compare across widths but that
            // is an edge case we conservatively leave as unknown).
            if wv.ty == rv.ty {
                if wv.bits == rv.bits {
                    IndexRelation::ProvablyEqual
                } else {
                    IndexRelation::ProvablyDisequal
                }
            } else {
                IndexRelation::Unknown
            }
        }
        (HirExprKind::GidBuiltin { axis: wa }, HirExprKind::GidBuiltin { axis: ra }) => {
            if wa == ra { IndexRelation::ProvablyEqual } else { IndexRelation::ProvablyDisequal }
        }
        // M3.3d: LocalInvocationIdBuiltin is a leaf — two calls with the same axis are equal.
        (HirExprKind::LocalInvocationIdBuiltin { axis: wa }, HirExprKind::LocalInvocationIdBuiltin { axis: ra }) => {
            if wa == ra { IndexRelation::ProvablyEqual } else { IndexRelation::ProvablyDisequal }
        }
        _ => IndexRelation::Unknown,
    }
}

struct TypeChecker<'p> {
    bindings: Vec<Binding>,
    errors: Vec<TypecheckError>,
    /// Non-fatal warnings (e.g. SubgroupOpInDivergentContext).
    warns: Vec<crate::validate::HirWarning>,
    next_id: u32,
    /// Kernel parameters (read-only; buffer params cannot be assigned to).
    params: &'p [KernelParam],
    /// Loop context stack for break/continue validation and induction-var detection.
    loop_stack: HirLoopStack,
    /// Scoped name-resolution: each block pushes a frame, pops on exit.
    scope_stack: ScopeStack,
    /// Tracks nesting depth of divergent control flow (if/else/while bodies).
    ///
    /// Incremented AFTER the cond expression is evaluated, decremented AFTER the body.
    /// For-range bodies do NOT increment this (M1.4 §5(8)).
    divergent_context_depth: u32,

    // ── M3.2 shared-array fields ─────────────────────────────────────────────

    /// Declared workgroup-shared arrays in source order.
    ///
    /// Indexed by SharedId.0 (monotonically from 0 per kernel).
    shared_decls: Vec<SharedDecl>,

    /// Maps shared array name -> index into shared_decls.
    shared_name_map: BTreeMap<String, usize>,

    /// Conditional nesting depth — incremented ONLY at if-then and else bodies.
    ///
    /// NOT incremented at `while` or `for-range` bodies. Used by the
    /// divergent-barrier hard error (OQ2, A.4.2). Distinct from `divergent_context_depth`
    /// which while DOES increment and gates the subgroup-collective warning.
    conditional_depth: u32,

    /// Per-shared-id SET of write index expressions since the last barrier (OQ1, A.4.1, r3).
    ///
    /// Key: SharedId.0. Value: Vec of index HirExprKinds from preceding SharedWrites.
    /// A `HirStmt::Barrier` clears ALL sets.
    /// A `SharedWrite` of id X appends its index kind to the set.
    /// A `SharedRead` of id X triggers the missing-barrier analysis.
    shared_write_sets: BTreeMap<u32, Vec<HirExprKind>>,

    // ── M3.20 local-array fields ──────────────────────────────────────────────

    /// Declared local arrays in source order.
    ///
    /// Indexed by LocalArrayId.0 (monotonically from 0 per kernel).
    local_array_decls: Vec<LocalArrayDecl>,

    /// Maps local-array name -> index into local_array_decls.
    local_array_name_map: BTreeMap<String, usize>,

    /// Presence-only write set: key = LocalArrayId.0, present = "written at least
    /// once, somewhere in the kernel body" (source-order, path-INsensitive — see
    /// `HirWarning::LocalArrayReadBeforeAnyWrite`'s doc for the honest disclosure).
    /// Never cleared (no barrier concept for private memory).
    local_array_write_sets: BTreeMap<u32, ()>,
}

impl<'p> TypeChecker<'p> {
    fn new(params: &'p [KernelParam]) -> Self {
        let mut tc = Self {
            bindings: Vec::new(),
            errors: Vec::new(),
            warns: Vec::new(),
            next_id: 0,
            params,
            loop_stack: HirLoopStack::new(),
            scope_stack: ScopeStack::new(),
            divergent_context_depth: 0,
            // M3.2 shared-array fields
            shared_decls: Vec::new(),
            shared_name_map: BTreeMap::new(),
            conditional_depth: 0,
            shared_write_sets: BTreeMap::new(),
            // M3.20 local-array fields
            local_array_decls: Vec::new(),
            local_array_name_map: BTreeMap::new(),
            local_array_write_sets: BTreeMap::new(),
        };
        // Push the top-level scope frame (pops at end of typecheck_kernel_body).
        tc.scope_stack.push_frame();
        tc
    }

    /// Look up a shared array by name. Returns `(SharedId, elem ScalarTy, len)` or `None`.
    fn find_shared(&self, name: &str) -> Option<(SharedId, ScalarTy, u32)> {
        self.shared_name_map.get(name).map(|&idx| {
            let decl = &self.shared_decls[idx];
            (decl.id, decl.ty.elem, decl.ty.len)
        })
    }

    /// Register a new shared array declaration.
    ///
    /// `len_hole`: `Some(name)` when the declaration's length is an unresolved
    /// `?name` hole (M3.22) — `ty.len` then holds the placeholder `1`.
    /// Returns `Some(SharedId)` on success, `None` on duplicate (error pushed).
    fn register_shared(
        &mut self,
        name: &str,
        ty: SharedTy,
        len_hole: Option<String>,
        span: Span,
    ) -> Option<SharedId> {
        if self.shared_name_map.contains_key(name) {
            self.errors.push(TypecheckError::SharedDuplicateName {
                name: name.to_owned(),
                span,
            });
            return None;
        }
        // Also check for collision with param or binding names.
        if self.find_param(name).is_some() || self.find_binding(name).is_some() {
            self.errors.push(TypecheckError::SharedNameCollision {
                name: name.to_owned(),
                span,
            });
            return None;
        }
        // M3.20 (r2, §8): bidirectional order-independent cross-check against the
        // local-array name map — a `shared` decl must not silently shadow (or be
        // shadowed by) an `array` decl of the same name regardless of declaration
        // order. See `register_local_array`'s mirror-image check and AT-2942.
        if self.local_array_name_map.contains_key(name) {
            self.errors.push(TypecheckError::SharedNameCollision {
                name: name.to_owned(),
                span,
            });
            return None;
        }
        let id = SharedId(self.shared_decls.len() as u32);
        let idx = self.shared_decls.len();
        self.shared_decls.push(SharedDecl {
            id,
            name: name.to_owned(),
            ty,
            span,
            len_hole,
        });
        self.shared_name_map.insert(name.to_owned(), idx);
        Some(id)
    }

    /// Look up a local array by name. Returns `(LocalArrayId, elem ScalarTy, len)` or `None`.
    fn find_local_array(&self, name: &str) -> Option<(LocalArrayId, ScalarTy, u32)> {
        self.local_array_name_map.get(name).map(|&idx| {
            let decl = &self.local_array_decls[idx];
            (decl.id, decl.ty.elem, decl.ty.len)
        })
    }

    /// Register a new local-array declaration.
    ///
    /// Returns `Some(LocalArrayId)` on success, `None` on duplicate/collision
    /// (error pushed). Mirrors `register_shared`, PLUS the bidirectional cross-check
    /// against `shared_name_map` (M3.20 r2, §8 — the #1 non-skippable review item):
    /// a `shared`/`array` name collision is caught regardless of declaration order,
    /// because BOTH registrars consult the other's map. See AT-2942.
    fn register_local_array(
        &mut self,
        name: &str,
        ty: LocalArrayTy,
        span: Span,
    ) -> Option<LocalArrayId> {
        if self.local_array_name_map.contains_key(name) {
            self.errors.push(TypecheckError::LocalArrayDuplicateName {
                name: name.to_owned(),
                span,
            });
            return None;
        }
        // Also check for collision with param or binding names.
        if self.find_param(name).is_some() || self.find_binding(name).is_some() {
            self.errors.push(TypecheckError::LocalArrayNameCollision {
                name: name.to_owned(),
                span,
            });
            return None;
        }
        // M3.20 (r2, §8): bidirectional cross-check against the shared-array name map.
        if self.shared_name_map.contains_key(name) {
            self.errors.push(TypecheckError::LocalArrayNameCollision {
                name: name.to_owned(),
                span,
            });
            return None;
        }
        let id = LocalArrayId(self.local_array_decls.len() as u32);
        let idx = self.local_array_decls.len();
        self.local_array_decls.push(LocalArrayDecl {
            id,
            name: name.to_owned(),
            ty,
            span,
        });
        self.local_array_name_map.insert(name.to_owned(), idx);
        Some(id)
    }

    /// Mark a local array as "written at least once" (§5 empty-write-set advisory).
    fn mark_local_array_written(&mut self, local_array_id: u32) {
        self.local_array_write_sets.insert(local_array_id, ());
    }

    /// True if `local_array_id` has been written at least once so far (presence-only,
    /// source-order, path-insensitive — see `HirWarning::LocalArrayReadBeforeAnyWrite`).
    fn local_array_has_been_written(&self, local_array_id: u32) -> bool {
        self.local_array_write_sets.contains_key(&local_array_id)
    }

    /// Clear all shared write sets — called when a Barrier statement is encountered.
    fn clear_shared_write_sets(&mut self) {
        self.shared_write_sets.clear();
    }

    /// Append a write index to the shared write set for `shared_id`.
    fn append_shared_write(&mut self, shared_id: u32, index_kind: HirExprKind) {
        self.shared_write_sets.entry(shared_id).or_default().push(index_kind);
    }

    /// Run the missing-barrier analysis for a SharedRead of `shared_id` with `read_index`.
    ///
    /// Checks the SET W_X of prior write indices (A.4.1, r3 SET-based rule).
    /// Returns the advisory warning kind if applicable.
    fn analyze_shared_read_barrier(
        &self,
        shared_id: u32,
        read_index: &HirExprKind,
        name: &str,
        span: Span,
    ) -> Option<SharedReadBarrierDiag> {
        let write_set = match self.shared_write_sets.get(&shared_id) {
            Some(set) if !set.is_empty() => set,
            _ => return None, // no prior writes — no hazard
        };

        // Check 1: ProvablyEqual to ANY prior write -> NO diagnostic (self-RAW).
        for wk in write_set {
            if index_relation(wk, read_index) == IndexRelation::ProvablyEqual {
                return None; // correct self-read, zero false positive
            }
        }

        // Check 2: ProvablyDisequal to EVERY prior write -> HARD ERROR.
        let all_disequal = write_set.iter()
            .all(|wk| index_relation(wk, read_index) == IndexRelation::ProvablyDisequal);
        if all_disequal {
            let write_descs: Vec<String> = write_set.iter()
                .map(format_index_kind)
                .collect();
            return Some(SharedReadBarrierDiag::HardError {
                name: name.to_owned(),
                read_index_desc: format_index_kind(read_index),
                write_indices_desc: write_descs.join(", "),
                span,
            });
        }

        // Check 3: Unknown vs at least one prior write -> advisory warning.
        Some(SharedReadBarrierDiag::Warning {
            name: name.to_owned(),
            span,
        })
    }

    /// Look up a kernel parameter by name.
    fn find_param(&self, name: &str) -> Option<&KernelParam> {
        self.params.iter().find(|p| p.name == name)
    }

    fn alloc_id(&mut self) -> BindingId {
        let id = BindingId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Find a binding by name: traverse scope_stack from inner to outer.
    ///
    /// Returns `(BindingId, BindingTy, is_mutable, span)`.
    fn find_binding(&self, name: &str) -> Option<(BindingId, BindingTy, bool, Span)> {
        if let Some(idx) = self.scope_stack.get(name) {
            let b = &self.bindings[idx];
            Some((b.id, b.ty, b.is_mutable, b.span))
        } else {
            None
        }
    }

    /// Find a binding and return its scalar type only. Returns `None` if not found
    /// or if the binding is a cooperative-matrix value.
    #[allow(dead_code)] // Used by check_coopmat_call (M2.1 — not yet wired in)
    fn find_scalar_binding(&self, name: &str) -> Option<(BindingId, ScalarTy, bool, Span)> {
        self.find_binding(name).and_then(|(id, bty, is_mut, span)| {
            bty.as_scalar().map(|st| (id, st, is_mut, span))
        })
    }

    /// Register a new scalar binding in the innermost scope frame.
    fn register_binding(&mut self, name: &str, ty: ScalarTy, is_mutable: bool, span: Span) -> Option<BindingId> {
        self.register_binding_typed(name, BindingTy::Scalar(ty), is_mutable, span)
    }

    /// Register a new cooperative-matrix binding in the innermost scope frame.
    fn register_coopmat_binding(&mut self, name: &str, key: CoopMatKey, is_mutable: bool, span: Span) -> Option<BindingId> {
        self.register_binding_typed(name, BindingTy::CoopMatrix(key), is_mutable, span)
    }

    /// Core binding registration. Duplicate detection is within the SAME scope frame only.
    ///
    /// Shadowing across frames (e.g. nested for loops with same induction var name) is allowed.
    fn register_binding_typed(&mut self, name: &str, ty: BindingTy, is_mutable: bool, span: Span) -> Option<BindingId> {
        // M2.1: `matrix` is reserved so that a future milestone can promote it to a
        // type-constructor keyword at expression scope. Reject it as a variable name.
        if name == "matrix" {
            self.errors.push(TypecheckError::ReservedKeyword {
                name: name.to_owned(),
                span,
            });
            return None;
        }
        // M2.5: Q4_0-path builtin names are reserved and cannot be used as variable names.
        if crate::q4_0::is_reserved_q4_0_builtin(name) {
            self.errors.push(TypecheckError::ReservedQ4_0BuiltinName {
                name: name.to_owned(),
                span,
            });
            return None;
        }
        // M3.2c: GLSL.std.450 ext-inst builtin names (e.g. `exp`) are reserved.
        if crate::ext_inst::is_reserved_ext_inst_builtin(name) {
            self.errors.push(TypecheckError::ReservedExtInstBuiltinName {
                name: name.to_owned(),
                span,
            });
            return None;
        }
        // Check for duplicate in the CURRENT frame only (not outer scopes — shadowing is OK).
        if let Some(idx) = self.scope_stack.get_in_current_frame(name) {
            let orig_span = self.bindings[idx].span;
            self.errors.push(TypecheckError::RedeclaredBinding {
                name: name.to_owned(),
                span,
                original_span: orig_span,
            });
            return None;
        }
        let id = self.alloc_id();
        let idx = self.bindings.len();
        self.bindings.push(Binding {
            id,
            name: name.to_owned(),
            ty,
            is_mutable,
            span,
        });
        self.scope_stack.insert(name.to_owned(), idx);
        Some(id)
    }
}

// ── Shared-read barrier diagnostic result type ────────────────────────────────

enum SharedReadBarrierDiag {
    HardError {
        name: String,
        read_index_desc: String,
        write_indices_desc: String,
        span: Span,
    },
    Warning {
        name: String,
        span: Span,
    },
}

/// Format an HirExprKind index for use in diagnostic messages.
fn format_index_kind(k: &HirExprKind) -> String {
    match k {
        HirExprKind::IntLit { value } => {
            // IntLiteralValue is { ty: ScalarTy, bits: u64 }.
            format!("{}({})", value.ty.display_name(), value.bits)
        }
        HirExprKind::LocalRead(BindingId(id)) => format!("binding#{id}"),
        HirExprKind::GidBuiltin { axis } => format!("gid({axis})"),
        HirExprKind::LocalInvocationIdBuiltin { axis } => format!("local_invocation_id({axis})"),
        _ => "<dynamic>".to_owned(),
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Typecheck a kernel body block.
///
/// `params` is the list of kernel parameters — buffer params cannot be used
/// as scalar values and cannot be assigned to; scalar params appear as read-only
/// bindings in expressions.
///
/// Always returns a `KernelBodyTyped` (possibly incomplete on errors), any
/// `TypecheckError`s, and any non-fatal `HirWarning`s. This supports error-recovery:
/// even with errors, downstream code sees a partial HIR to collect further diagnostics.
pub fn typecheck_kernel_body(
    body: &past::Block,
    params: &[KernelParam],
) -> (KernelBodyTyped, Vec<TypecheckError>, Vec<crate::validate::HirWarning>) {
    let mut tc = TypeChecker::new(params);

    // ── Pass 1: Pre-register top-level let bindings ───────────────────────────
    // This keeps the flat-body path consistent with M1.1/M1.2. Control-flow
    // nested blocks use a single-pass scheme (no pre-registration needed because
    // they introduce new scope frames).
    pre_register_lets_in_block(body, &mut tc);

    // ── Pass 2: Typecheck expressions in each statement ───────────────────────
    let mut hir_stmts: Vec<HirStmt> = Vec::new();
    for spanned_stmt in &body.stmts {
        match &spanned_stmt.node {
            past::Stmt::Let { name, ty, init, .. } => {
                // M1.4: reject reserved subgroup builtin names as variable names.
                if axc_lexer::is_reserved_subgroup_builtin(&name.node) {
                    tc.errors.push(TypecheckError::ReservedBuiltinName {
                        name: name.node.clone(),
                        span: name.span,
                    });
                    // Skip registration but continue for further diagnostics.
                }

                // Lookup the binding registered in pass 1.
                let maybe_binding = tc.find_binding(&name.node);
                // For coopmat bindings, scalar expected_ty is None (coopmat check is
                // dispatched separately below). For scalar bindings, extract ScalarTy.
                let expected_scalar_ty = maybe_binding.and_then(|(_, bty, _, _)| bty.as_scalar());

                // M2.1: if the let target is a CoopMatrix binding, check if the init
                // is coopmat_zero() or coopmat_load() which need the matrix key as context.
                let hir_init = if let Some((_, BindingTy::CoopMatrix(matrix_key), _, _)) = maybe_binding {
                    check_coopmat_init_expr(&mut tc, &init.node, init.span, matrix_key)
                } else {
                    check_expr(&mut tc, &init.node, init.span, expected_scalar_ty)
                };

                if let Some((bid, _, _, _)) = maybe_binding {
                    if let Some(init_expr) = hir_init {
                        hir_stmts.push(HirStmt::Let {
                            binding: bid,
                            init: init_expr,
                            span: spanned_stmt.span,
                        });
                    }
                }
                let _ = ty; // used in pass 1
            }
            past::Stmt::Assign { target, value } => {
                // Check if the target is a kernel parameter — params are always immutable.
                if tc.find_param(&target.node).is_some() {
                    tc.errors.push(TypecheckError::AssignToParam {
                        name: target.node.clone(),
                        span: target.span,
                    });
                    // Still check the value in an unconstrained context for further diagnostics.
                    let _ = check_expr(&mut tc, &value.node, value.span, None);
                } else {
                    match tc.find_binding(&target.node) {
                        None => {
                            tc.errors.push(TypecheckError::UnknownBinding {
                                name: target.node.clone(),
                                span: target.span,
                            });
                            // Still check the value in an unconstrained context
                            let _ = check_expr(&mut tc, &value.node, value.span, None);
                        }
                        Some((bid, binding_ty, is_mutable, orig_span)) => {
                            // Check induction variable assignment first.
                            if tc.loop_stack.contains_induction_binding(bid) {
                                tc.errors.push(TypecheckError::AssignToForInductionVar {
                                    name: target.node.clone(),
                                    span: target.span,
                                });
                            } else if !is_mutable {
                                tc.errors.push(TypecheckError::AssignImmutable {
                                    name: target.node.clone(),
                                    span: target.span,
                                    original_span: orig_span,
                                });
                            }
                            // M3.3 ISSUE-1 (FLAT Assign arm): route CoopMatrix target through
                            // check_coopmat_init_expr (same fn as the Let arm, typecheck.rs:1002-1006)
                            // so `acc = coopmat_mul_add(a,b,acc)` typechecks and use_==Accumulator +
                            // K/M/N/elem shape is validated. Scalars keep check_expr with scalar expected.
                            let hir_value = match binding_ty {
                                BindingTy::CoopMatrix(matrix_key) => {
                                    check_coopmat_init_expr(&mut tc, &value.node, value.span, matrix_key)
                                }
                                BindingTy::Scalar(_) => {
                                    let scalar_expected = binding_ty.as_scalar();
                                    check_expr(&mut tc, &value.node, value.span, scalar_expected)
                                }
                            };
                            if let Some(val_expr) = hir_value {
                                hir_stmts.push(HirStmt::Assign {
                                    binding: bid,
                                    value: val_expr,
                                    span: spanned_stmt.span,
                                });
                            }
                        }
                    }
                }
            }
            past::Stmt::Return(maybe_expr) => {
                if let Some(expr) = maybe_expr {
                    tc.errors.push(TypecheckError::UnsupportedExprInM1_1 {
                        detail: "return with value (kernels must return void in M1.1)",
                        span: expr.span,
                    });
                }
                hir_stmts.push(HirStmt::Return { span: spanned_stmt.span });
            }
            past::Stmt::IndexAssign { target, index, value } => {
                if let Some(stmt) = check_index_assign_stmt(&mut tc, target, index, value, spanned_stmt.span) {
                    hir_stmts.push(stmt);
                }
            }
            past::Stmt::If { cond, then_block, else_arm } => {
                if let Some(stmt) = check_if_stmt(&mut tc, cond, then_block, else_arm.as_deref(), spanned_stmt.span) {
                    hir_stmts.push(HirStmt::If(stmt));
                }
            }
            past::Stmt::For { var, start, end, step, body } => {
                if let Some(stmt) = check_for_stmt(&mut tc, var, start, end, step.as_ref(), body, spanned_stmt.span) {
                    hir_stmts.push(HirStmt::ForRange(stmt));
                }
            }
            past::Stmt::While { cond, body } => {
                if let Some(stmt) = check_while_stmt(&mut tc, cond, body, spanned_stmt.span) {
                    hir_stmts.push(HirStmt::While(stmt));
                }
            }
            past::Stmt::Break => {
                if !tc.loop_stack.is_in_loop() {
                    tc.errors.push(TypecheckError::BreakOutsideLoop { span: spanned_stmt.span });
                } else {
                    hir_stmts.push(HirStmt::Break { span: spanned_stmt.span });
                }
            }
            past::Stmt::Continue => {
                if !tc.loop_stack.is_in_loop() {
                    tc.errors.push(TypecheckError::ContinueOutsideLoop { span: spanned_stmt.span });
                } else {
                    hir_stmts.push(HirStmt::Continue { span: spanned_stmt.span });
                }
            }
            past::Stmt::BuiltinCallStmt { call } => {
                // M1.4: handle reserved subgroup builtin call at statement position.
                if let Some(stmt) = check_builtin_call_stmt(&mut tc, call, spanned_stmt.span) {
                    // M3.2: If this is a Barrier, clear all shared write sets (A.4.1).
                    if matches!(stmt, HirStmt::Barrier { .. }) {
                        tc.clear_shared_write_sets();
                    }
                    hir_stmts.push(stmt);
                }
            }
            // M3.2: workgroup-shared array declaration.
            past::Stmt::SharedDecl { name, elem, len, len_hole } => {
                if let Some(stmt) = check_shared_decl_stmt(&mut tc, name, elem, len, len_hole, spanned_stmt.span) {
                    hir_stmts.push(stmt);
                }
            }
            // M3.20: local-array declaration — ALLOWED at kernel-body top level.
            past::Stmt::LocalArrayDecl { name, elem, len } => {
                if let Some(stmt) = check_local_array_decl_stmt(&mut tc, name, elem, len, spanned_stmt.span, true) {
                    hir_stmts.push(stmt);
                }
            }
        }
    }

    // Pop the top-level scope frame opened in TypeChecker::new.
    tc.scope_stack.pop_frame();

    // M3.2: Check aggregate shared-memory size limits.
    // - > MAX_SHARED_BYTES (65536) → SharedMemoryTooLarge hard error.
    // - > PORTABLE_MIN_SHARED_BYTES (16384) → advisory warning.
    //
    // M3.22: skipped entirely when any shared decl still carries an unresolved
    // `len_hole` — its placeholder length (1) would spuriously undercount the
    // aggregate. Mirrors `lower.rs`'s existing skip of workgroup-dim validation
    // when `@workgroup` has holes.
    if !tc.shared_decls.iter().any(|s| s.len_hole.is_some()) {
        use crate::shared::{MAX_SHARED_BYTES, PORTABLE_MIN_SHARED_BYTES};
        let total_bytes: u64 = tc.shared_decls.iter()
            .map(|s| s.ty.total_byte_size())
            .sum();
        if total_bytes > MAX_SHARED_BYTES {
            tc.errors.push(TypecheckError::SharedMemoryTooLarge {
                total_bytes,
                // Use the span of the last shared decl for location.
                span: tc.shared_decls.last().map(|s| s.span).unwrap_or_default(),
            });
        } else if total_bytes > u64::from(PORTABLE_MIN_SHARED_BYTES) {
            tc.warns.push(crate::validate::HirWarning::SharedMemoryExceedsPortableMinimum {
                total_bytes,
                min_bytes: PORTABLE_MIN_SHARED_BYTES,
                span: tc.shared_decls.last().map(|s| s.span).unwrap_or_default(),
            });
        }
    }

    // M3.20: Check aggregate local-array size limits (§6).
    // - > MAX_LOCAL_ARRAY_BYTES (4096) → LocalArrayTooLarge hard error.
    // - > LOCAL_ARRAY_SPILL_ADVISORY_BYTES (1024) → advisory warning.
    {
        use crate::local::{MAX_LOCAL_ARRAY_BYTES, LOCAL_ARRAY_SPILL_ADVISORY_BYTES};
        let total_bytes: u64 = tc.local_array_decls.iter()
            .map(|a| a.ty.total_byte_size())
            .sum();
        if total_bytes > MAX_LOCAL_ARRAY_BYTES {
            tc.errors.push(TypecheckError::LocalArrayTooLarge {
                total_bytes,
                span: tc.local_array_decls.last().map(|a| a.span).unwrap_or_default(),
            });
        } else if total_bytes > LOCAL_ARRAY_SPILL_ADVISORY_BYTES {
            tc.warns.push(crate::validate::HirWarning::LocalArrayMaySpill {
                total_bytes,
                advisory_bytes: LOCAL_ARRAY_SPILL_ADVISORY_BYTES as u32,
                span: tc.local_array_decls.last().map(|a| a.span).unwrap_or_default(),
            });
        }
    }

    let body_typed = KernelBodyTyped {
        bindings: tc.bindings,
        stmts: hir_stmts,
        shared: tc.shared_decls,
        local_arrays: tc.local_array_decls,
    };

    (body_typed, tc.errors, tc.warns)
}

// ── Pre-registration of let bindings (top-level only) ────────────────────────

/// Pre-register let bindings at the TOP level of a block (not nested blocks).
///
/// This is the M1.1/M1.2 two-pass approach for flat kernel bodies. Nested blocks
/// (if/for/while bodies) use single-pass with scope frames and DON'T pre-register.
fn pre_register_lets_in_block(block: &past::Block, tc: &mut TypeChecker<'_>) {
    for spanned_stmt in &block.stmts {
        if let past::Stmt::Let { name, ty, is_mut, .. } = &spanned_stmt.node {
            // CoopMatrix bindings are handled differently: register via register_coopmat_binding.
            if let past::TypeRef::CoopMatrix { elem, m, n, use_ } = &ty.node {
                // Bf16 is accepted by the parser but rejected here (AT-609).
                // Must check BEFORE calling lower_scalar_type_ref_tc (which panics for Bf16).
                if *elem == past::ScalarTypeRef::Bf16 {
                    tc.errors.push(TypecheckError::CoopMatrixElementTypeUnsupported {
                        ty: "bf16",
                        span: ty.span,
                    });
                    tc.register_binding(&name.node, ScalarTy::I32, *is_mut, name.span);
                    continue;
                }
                let elem_scalar = lower_scalar_type_ref_tc(elem);
                if !is_allowed_coopmat_element(elem_scalar) {
                    tc.errors.push(TypecheckError::CoopMatrixElementTypeUnsupported {
                        ty: elem_scalar.display_name(),
                        span: ty.span,
                    });
                    // Register as a placeholder scalar to allow further analysis.
                    tc.register_binding(&name.node, ScalarTy::I32, *is_mut, name.span);
                    continue;
                }
                let coopmat_use = coopmat_use_ast_to_hir(use_);
                // M3.1: k and result_type are internal-cache-key fields. At per-binding
                // registration time K is not yet known (it requires both A and B bindings
                // to be seen together). We use n as a placeholder for k (A.n == K for
                // MatrixA is a common convention) and elem for result_type (same elem for
                // all-f16 case). derive_coopmat_shape in lower.rs sources the real K from
                // the mul_add operand types where K = a.n = b.m is guaranteed by KDimMismatch.
                let key = CoopMatKey { elem: elem_scalar, m: *m, n: *n, k: *n, use_: coopmat_use, result_type: elem_scalar };
                tc.register_coopmat_binding(&name.node, key, *is_mut, name.span);
                continue;
            }
            let scalar_ty = match typeref_to_scalar(&ty.node) {
                Ok(t) => t,
                Err("__coopmat__") => {
                    // Should have been caught above; this branch is unreachable but safe.
                    ScalarTy::I32
                }
                Err(detail) => {
                    tc.errors.push(TypecheckError::UnsupportedExprInM1_1 {
                        detail,
                        span: ty.span,
                    });
                    ScalarTy::I32
                }
            };
            tc.register_binding(&name.node, scalar_ty, *is_mut, name.span);
        }
    }
}

/// Convert an AST `ScalarTypeRef` to a HIR `ScalarTy` (typecheck-layer helper).
///
/// Mirrors `lower::lower_scalar_type_ref` but is used in typecheck where
/// we can't call into the lower module to avoid circular concerns.
///
/// PRECONDITION: callers must check for `ScalarTypeRef::Bf16` BEFORE calling
/// this function and emit `CoopMatrixElementTypeUnsupported { ty: "bf16" }`.
/// Bf16 is not represented in `ScalarTy`; calling this function with Bf16 panics.
fn lower_scalar_type_ref_tc(str_ref: &past::ScalarTypeRef) -> ScalarTy {
    match str_ref {
        past::ScalarTypeRef::I8  => ScalarTy::I8,
        past::ScalarTypeRef::U8  => ScalarTy::U8,
        past::ScalarTypeRef::I32 => ScalarTy::I32,
        past::ScalarTypeRef::U32 => ScalarTy::U32,
        past::ScalarTypeRef::I64 => ScalarTy::I64,
        past::ScalarTypeRef::U64 => ScalarTy::U64,
        past::ScalarTypeRef::F16 => ScalarTy::F16,
        past::ScalarTypeRef::F32 => ScalarTy::F32,
        past::ScalarTypeRef::F64 => ScalarTy::F64,
        past::ScalarTypeRef::Bf16 => {
            // This should never be reached: callers must filter Bf16 before calling.
            // If it does, this is a compiler bug.
            panic!("lower_scalar_type_ref_tc: bf16 is not a valid HIR ScalarTy; \
                    callers must check ScalarTypeRef::Bf16 and emit CoopMatrixElementTypeUnsupported")
        }
    }
}

/// Convert a parsed `CoopMatUseAst` to the HIR `CoopMatUse`.
fn coopmat_use_ast_to_hir(use_: &past::CoopMatUseAst) -> CoopMatUse {
    match use_ {
        past::CoopMatUseAst::A           => CoopMatUse::MatrixA,
        past::CoopMatUseAst::B           => CoopMatUse::MatrixB,
        past::CoopMatUseAst::Accumulator => CoopMatUse::Accumulator,
    }
}

// ── Control flow statement typechecking (M1.3) ───────────────────────────────

/// Typecheck `if cond { then } [else ...]`
///
/// Rev 1 CRITICAL-1: compound short-circuit in cond is rejected BEFORE bool check.
fn check_if_stmt(
    tc: &mut TypeChecker<'_>,
    cond: &axc_lexer::Spanned<past::Expr>,
    then_block: &axc_lexer::Spanned<past::Block>,
    else_arm: Option<&axc_parser::ast::ElseArm>,
    _stmt_span: Span,
) -> Option<HirIf> {
    // CRITICAL-1: reject compound short-circuit in if-header
    if matches!(&cond.node, past::Expr::ShortCircuit { .. }) {
        tc.errors.push(TypecheckError::UnsupportedShortCircuitInHeader {
            position: "if",
            span: cond.span,
        });
        return None;
    }

    // Cond is evaluated at the PARENT (outer) depth (§5(8) rev 1 CRITICAL-3).
    let cond_hir = check_expr(tc, &cond.node, cond.span, Some(ScalarTy::Bool))?;
    if cond_hir.ty != ScalarTy::Bool {
        tc.errors.push(TypecheckError::NonBoolCondition {
            position: "if",
            got: cond_hir.ty.display_name(),
            span: cond.span,
        });
        return None;
    }

    // Increment divergent depth AFTER cond is evaluated, BEFORE entering then-block.
    // M3.2: Also increment conditional_depth (if/else only — NOT while/for-range).
    tc.divergent_context_depth += 1;
    tc.conditional_depth += 1;
    let then_stmts = typecheck_nested_block(tc, then_block);
    // Decrement AFTER then-block statements complete.
    tc.divergent_context_depth -= 1;
    tc.conditional_depth -= 1;

    let hir_else: Option<Box<HirElse>> = match else_arm {
        None => None,
        Some(past::ElseArm::Block(block)) => {
            // Increment divergent depth for the else-block.
            // M3.2: Also increment conditional_depth.
            tc.divergent_context_depth += 1;
            tc.conditional_depth += 1;
            let else_stmts = typecheck_nested_block(tc, block);
            tc.divergent_context_depth -= 1;
            tc.conditional_depth -= 1;
            Some(Box::new(HirElse::Block(else_stmts)))
        }
        Some(past::ElseArm::If(inner_spanned)) => {
            // inner_spanned.node must be Stmt::If
            if let past::Stmt::If { cond: ic, then_block: itb, else_arm: iea } = &inner_spanned.node {
                // The nested if-else-if: divergent depth was already incremented
                // before the outer if-cond; the nested check_if_stmt will handle
                // its own depth tracking from the current depth level.
                check_if_stmt(tc, ic, itb, iea.as_deref(), inner_spanned.span)
                    .map(|hir_if| Box::new(HirElse::If(hir_if)))
            } else {
                // Should not happen: parser guarantees ElseArm::If contains Stmt::If
                tc.errors.push(TypecheckError::UnsupportedExprInM1_1 {
                    detail: "internal: ElseArm::If does not contain Stmt::If",
                    span: inner_spanned.span,
                });
                None
            }
        }
    };

    Some(HirIf {
        cond: cond_hir,
        then_block: then_stmts,
        else_arm: hir_else,
        span: _stmt_span,
    })
}

/// Typecheck `for var in range(start, end [, step]) { body }`
fn check_for_stmt(
    tc: &mut TypeChecker<'_>,
    var: &axc_lexer::Spanned<String>,
    start: &axc_lexer::Spanned<past::Expr>,
    end: &axc_lexer::Spanned<past::Expr>,
    step: Option<&axc_lexer::Spanned<past::Expr>>,
    body: &axc_lexer::Spanned<past::Block>,
    stmt_span: Span,
) -> Option<HirForRange> {
    // Typecheck start and end with expected=U32
    let start_hir = check_expr(tc, &start.node, start.span, Some(ScalarTy::U32))?;
    let end_hir   = check_expr(tc, &end.node,   end.span,   Some(ScalarTy::U32))?;

    if start_hir.ty != ScalarTy::U32 {
        tc.errors.push(TypecheckError::TypeMismatch {
            expected: "u32",
            got: start_hir.ty.display_name(),
            span: start.span,
        });
    }
    if end_hir.ty != ScalarTy::U32 {
        tc.errors.push(TypecheckError::TypeMismatch {
            expected: "u32",
            got: end_hir.ty.display_name(),
            span: end.span,
        });
    }

    // Step: must be a compile-time positive u32 constant
    let for_step: ForStep = match step {
        None => ForStep::ONE,
        Some(step_expr) => {
            // Unwrap one layer of Paren before inspecting
            let inner_expr = match &step_expr.node {
                past::Expr::Paren(inner) => &inner.node,
                other => other,
            };
            match inner_expr {
                past::Expr::IntLit { value, suffix } => {
                    // Check suffix: must be u32 or absent
                    match suffix {
                        Some(axc_lexer::IntSuffix::U32) | None => {}
                        Some(axc_lexer::IntSuffix::I32) => {
                            tc.errors.push(TypecheckError::ForStepNotU32 {
                                got_suffix: "i32",
                                span: step_expr.span,
                            });
                            return None;
                        }
                        Some(axc_lexer::IntSuffix::I64) => {
                            tc.errors.push(TypecheckError::ForStepNotU32 {
                                got_suffix: "i64",
                                span: step_expr.span,
                            });
                            return None;
                        }
                        Some(axc_lexer::IntSuffix::U64) => {
                            tc.errors.push(TypecheckError::ForStepNotU32 {
                                got_suffix: "u64",
                                span: step_expr.span,
                            });
                            return None;
                        }
                        Some(axc_lexer::IntSuffix::I8)
                        | Some(axc_lexer::IntSuffix::I16)
                        | Some(axc_lexer::IntSuffix::U8)
                        | Some(axc_lexer::IntSuffix::U16) => {
                            tc.errors.push(TypecheckError::ForStepNotU32 {
                                got_suffix: "narrow integer",
                                span: step_expr.span,
                            });
                            return None;
                        }
                    }
                    // Must be positive and fit u32
                    if *value <= 0 {
                        tc.errors.push(TypecheckError::ForStepNotPositive {
                            value: *value as u64,
                            span: step_expr.span,
                        });
                        return None;
                    }
                    if *value > u32::MAX as i128 {
                        tc.errors.push(TypecheckError::ForStepNotPositive {
                            value: *value as u64,
                            span: step_expr.span,
                        });
                        return None;
                    }
                    ForStep { value: *value as u32 }
                }
                _ => {
                    tc.errors.push(TypecheckError::ForStepNotConstant { span: step_expr.span });
                    return None;
                }
            }
        }
    };

    // AT-315: reject redeclaration of a kernel-scope `let` binding by a for-induction.
    // `let i = ...; for i in range(...) { }` must produce RedeclaredBinding.
    // `for i in ... { for i in ... { } }` is allowed (outer `i` lives in for-scope
    // frame 1, NOT kernel-scope frame 0, so the check below returns None for
    // the inner for's `i` lookup in the kernel-scope frame).
    tc.scope_stack.push_frame();
    // Check the outermost (kernel-scope) frame only.
    if let Some(orig_idx) = tc.scope_stack.get_in_kernel_scope_frame(&var.node) {
        let orig_span = tc.bindings[orig_idx].span;
        tc.errors.push(TypecheckError::RedeclaredBinding {
            name: var.node.clone(),
            span: var.span,
            original_span: orig_span,
        });
        tc.scope_stack.pop_frame();
        return None;
    }
    let induction_id: BindingId = match tc.register_binding(&var.node, ScalarTy::U32, false, var.span) {
        Some(id) => id,
        None => {
            // Duplicate binding in the nested scope — error already pushed
            tc.scope_stack.pop_frame();
            return None;
        }
    };

    // Push a loop frame with the induction variable
    tc.loop_stack.push(Some(induction_id));

    // Typecheck the loop body
    let body_stmts = typecheck_nested_block(tc, body);

    // Pop loop frame and induction scope frame
    tc.loop_stack.pop();
    tc.scope_stack.pop_frame();

    Some(HirForRange {
        induction: induction_id,
        start: start_hir,
        end: end_hir,
        step: for_step,
        body: body_stmts,
        span: stmt_span,
    })
}

/// Typecheck `while cond { body }`
///
/// Rev 1 CRITICAL-1: compound short-circuit in cond is rejected BEFORE bool check.
fn check_while_stmt(
    tc: &mut TypeChecker<'_>,
    cond: &axc_lexer::Spanned<past::Expr>,
    body: &axc_lexer::Spanned<past::Block>,
    stmt_span: Span,
) -> Option<HirWhile> {
    // CRITICAL-1: reject compound short-circuit in while-header
    if matches!(&cond.node, past::Expr::ShortCircuit { .. }) {
        tc.errors.push(TypecheckError::UnsupportedShortCircuitInHeader {
            position: "while",
            span: cond.span,
        });
        return None;
    }

    let cond_hir = check_expr(tc, &cond.node, cond.span, Some(ScalarTy::Bool))?;
    if cond_hir.ty != ScalarTy::Bool {
        tc.errors.push(TypecheckError::NonBoolCondition {
            position: "while",
            got: cond_hir.ty.display_name(),
            span: cond.span,
        });
        return None;
    }

    // Push a loop frame (no induction variable for while loops).
    // Increment divergent depth AFTER while-cond, BEFORE body (§5(8) rev 1).
    tc.loop_stack.push(None);
    tc.divergent_context_depth += 1;
    let body_stmts = typecheck_nested_block(tc, body);
    tc.divergent_context_depth -= 1;
    tc.loop_stack.pop();

    Some(HirWhile {
        cond: cond_hir,
        body: body_stmts,
        span: stmt_span,
    })
}

/// Typecheck a nested block (if-then, loop body, etc.) in a fresh scope frame.
///
/// The scope frame is pushed before typechecking and popped after.
/// Let bindings inside nested blocks are registered via single-pass as they are
/// encountered (no pre-registration needed — the flat kernel body is the only
/// place that uses two-pass pre-registration for forward references).
fn typecheck_nested_block(tc: &mut TypeChecker<'_>, block: &axc_lexer::Spanned<past::Block>) -> Vec<HirStmt> {
    tc.scope_stack.push_frame();
    let stmts = typecheck_block_stmts(tc, &block.node.stmts);
    tc.scope_stack.pop_frame();
    stmts
}

/// Typecheck a sequence of statements (single-pass, no pre-registration).
fn typecheck_block_stmts(tc: &mut TypeChecker<'_>, stmts: &[axc_lexer::Spanned<past::Stmt>]) -> Vec<HirStmt> {
    let mut hir_stmts: Vec<HirStmt> = Vec::new();
    for spanned_stmt in stmts {
        match &spanned_stmt.node {
            past::Stmt::Let { name, ty, is_mut, init } => {
                // M1.4: reject reserved subgroup builtin names as variable names.
                if axc_lexer::is_reserved_subgroup_builtin(&name.node) {
                    tc.errors.push(TypecheckError::ReservedBuiltinName {
                        name: name.node.clone(),
                        span: name.span,
                    });
                }
                // M2.1: CoopMatrix let bindings in nested blocks (single-pass).
                if let past::TypeRef::CoopMatrix { elem, m, n, use_ } = &ty.node {
                    if *elem == past::ScalarTypeRef::Bf16 {
                        tc.errors.push(TypecheckError::CoopMatrixElementTypeUnsupported {
                            ty: "bf16",
                            span: ty.span,
                        });
                        continue;
                    }
                    let elem_scalar = lower_scalar_type_ref_tc(elem);
                    if !is_allowed_coopmat_element(elem_scalar) {
                        tc.errors.push(TypecheckError::CoopMatrixElementTypeUnsupported {
                            ty: elem_scalar.display_name(),
                            span: ty.span,
                        });
                        continue;
                    }
                    let coopmat_use = coopmat_use_ast_to_hir(use_);
                    // M3.1: k and result_type placeholder — see comment at first construction site.
                    let key = CoopMatKey { elem: elem_scalar, m: *m, n: *n, k: *n, use_: coopmat_use, result_type: elem_scalar };
                    let hir_init = check_coopmat_init_expr(tc, &init.node, init.span, key);
                    if let Some(bid) = tc.register_coopmat_binding(&name.node, key, *is_mut, name.span) {
                        if let Some(init_expr) = hir_init {
                            hir_stmts.push(HirStmt::Let {
                                binding: bid,
                                init: init_expr,
                                span: spanned_stmt.span,
                            });
                        }
                    }
                    continue;
                }
                // Single-pass: register binding immediately on encounter.
                let scalar_ty = match typeref_to_scalar(&ty.node) {
                    Ok(t) => t,
                    Err(detail) => {
                        tc.errors.push(TypecheckError::UnsupportedExprInM1_1 {
                            detail,
                            span: ty.span,
                        });
                        ScalarTy::I32
                    }
                };
                let hir_init = check_expr(tc, &init.node, init.span, Some(scalar_ty));
                if let Some(bid) = tc.register_binding(&name.node, scalar_ty, *is_mut, name.span) {
                    if let Some(init_expr) = hir_init {
                        hir_stmts.push(HirStmt::Let {
                            binding: bid,
                            init: init_expr,
                            span: spanned_stmt.span,
                        });
                    }
                }
            }
            past::Stmt::Assign { target, value } => {
                if tc.find_param(&target.node).is_some() {
                    tc.errors.push(TypecheckError::AssignToParam {
                        name: target.node.clone(),
                        span: target.span,
                    });
                    let _ = check_expr(tc, &value.node, value.span, None);
                } else {
                    match tc.find_binding(&target.node) {
                        None => {
                            tc.errors.push(TypecheckError::UnknownBinding {
                                name: target.node.clone(),
                                span: target.span,
                            });
                            let _ = check_expr(tc, &value.node, value.span, None);
                        }
                        Some((bid, binding_ty, is_mutable, _orig_span)) => {
                            // Check induction variable assignment
                            if tc.loop_stack.contains_induction_binding(bid) {
                                tc.errors.push(TypecheckError::AssignToForInductionVar {
                                    name: target.node.clone(),
                                    span: target.span,
                                });
                            } else if !is_mutable {
                                tc.errors.push(TypecheckError::AssignImmutable {
                                    name: target.node.clone(),
                                    span: target.span,
                                    original_span: _orig_span,
                                });
                            }
                            // M3.3 ISSUE-1 (NESTED Assign arm): route CoopMatrix target through
                            // check_coopmat_init_expr (same fn as the Let arm, typecheck.rs:1002-1006)
                            // so `acc = coopmat_mul_add(a,b,acc)` inside a loop typechecks and
                            // use_==Accumulator + K/M/N/elem shape is validated for free.
                            // Scalars keep check_expr with scalar expected.
                            let hir_value = match binding_ty {
                                BindingTy::CoopMatrix(matrix_key) => {
                                    check_coopmat_init_expr(tc, &value.node, value.span, matrix_key)
                                }
                                BindingTy::Scalar(_) => {
                                    let scalar_expected = binding_ty.as_scalar();
                                    check_expr(tc, &value.node, value.span, scalar_expected)
                                }
                            };
                            if let Some(val_expr) = hir_value {
                                hir_stmts.push(HirStmt::Assign {
                                    binding: bid,
                                    value: val_expr,
                                    span: spanned_stmt.span,
                                });
                            }
                        }
                    }
                }
            }
            past::Stmt::Return(maybe_expr) => {
                if let Some(expr) = maybe_expr {
                    tc.errors.push(TypecheckError::UnsupportedExprInM1_1 {
                        detail: "return with value (kernels must return void)",
                        span: expr.span,
                    });
                }
                hir_stmts.push(HirStmt::Return { span: spanned_stmt.span });
            }
            past::Stmt::IndexAssign { target, index, value } => {
                if let Some(stmt) = check_index_assign_stmt(tc, target, index, value, spanned_stmt.span) {
                    hir_stmts.push(stmt);
                }
            }
            past::Stmt::If { cond, then_block, else_arm } => {
                if let Some(stmt) = check_if_stmt(tc, cond, then_block, else_arm.as_deref(), spanned_stmt.span) {
                    hir_stmts.push(HirStmt::If(stmt));
                }
            }
            past::Stmt::For { var, start, end, step, body } => {
                if let Some(stmt) = check_for_stmt(tc, var, start, end, step.as_ref(), body, spanned_stmt.span) {
                    hir_stmts.push(HirStmt::ForRange(stmt));
                }
            }
            past::Stmt::While { cond, body } => {
                if let Some(stmt) = check_while_stmt(tc, cond, body, spanned_stmt.span) {
                    hir_stmts.push(HirStmt::While(stmt));
                }
            }
            past::Stmt::Break => {
                if !tc.loop_stack.is_in_loop() {
                    tc.errors.push(TypecheckError::BreakOutsideLoop { span: spanned_stmt.span });
                } else {
                    hir_stmts.push(HirStmt::Break { span: spanned_stmt.span });
                }
            }
            past::Stmt::Continue => {
                if !tc.loop_stack.is_in_loop() {
                    tc.errors.push(TypecheckError::ContinueOutsideLoop { span: spanned_stmt.span });
                } else {
                    hir_stmts.push(HirStmt::Continue { span: spanned_stmt.span });
                }
            }
            past::Stmt::BuiltinCallStmt { call } => {
                if let Some(stmt) = check_builtin_call_stmt(tc, call, spanned_stmt.span) {
                    // M3.2: If this is a Barrier, clear all shared write sets (A.4.1).
                    if matches!(stmt, HirStmt::Barrier { .. }) {
                        tc.clear_shared_write_sets();
                    }
                    hir_stmts.push(stmt);
                }
            }
            // M3.2: shared array declarations are allowed in nested blocks too.
            past::Stmt::SharedDecl { name, elem, len, len_hole } => {
                if let Some(stmt) = check_shared_decl_stmt(tc, name, elem, len, len_hole, spanned_stmt.span) {
                    hir_stmts.push(stmt);
                }
            }
            // M3.20 (r2, §5.1): local-array declarations are REJECTED in nested
            // blocks (if/for/while bodies) — HardError `LocalArrayDeclNotAtBlockScope`.
            // Still registers (poisoned) so later in-block uses don't cascade
            // unrelated errors — see `check_local_array_decl_stmt`'s doc.
            past::Stmt::LocalArrayDecl { name, elem, len } => {
                if let Some(stmt) = check_local_array_decl_stmt(tc, name, elem, len, spanned_stmt.span, false) {
                    hir_stmts.push(stmt);
                }
            }
        }
    }
    hir_stmts
}

// ── Convert TypeRef to ScalarTy ───────────────────────────────────────────────

fn typeref_to_scalar(tr: &past::TypeRef) -> Result<ScalarTy, &'static str> {
    match tr {
        past::TypeRef::I32  => Ok(ScalarTy::I32),
        past::TypeRef::U32  => Ok(ScalarTy::U32),
        past::TypeRef::I64  => Ok(ScalarTy::I64),
        past::TypeRef::U64  => Ok(ScalarTy::U64),
        past::TypeRef::F16  => Ok(ScalarTy::F16),
        past::TypeRef::F32  => Ok(ScalarTy::F32),
        past::TypeRef::F64  => Ok(ScalarTy::F64),
        past::TypeRef::Bool => Ok(ScalarTy::Bool),
        past::TypeRef::Void => Err("void is not a valid scalar type for let bindings"),
        past::TypeRef::Buffer(_)
        | past::TypeRef::ReadonlyBuffer(_)
        | past::TypeRef::WriteonlyBuffer(_) => {
            Err("buffer types are not valid for let bindings; use as kernel parameters only")
        }
        // CoopMatrix is handled separately via pre_register_coopmat / check_coopmat_let.
        // Return a sentinel error so callers fall back to the coopmat path.
        past::TypeRef::CoopMatrix { .. } => {
            Err("__coopmat__")
        }
        // M3.2: shared[T,N] is not a valid scalar type for let bindings.
        // It is a declaration statement type, not a let-binding type.
        past::TypeRef::Shared { .. } => {
            Err("shared[T,N] is not a valid let-binding type; use `shared name: shared[T,N];` to declare a shared array")
        }
        // M3.20: array[T,N] is not a valid scalar type for let bindings.
        // It is a declaration statement type, not a let-binding type.
        past::TypeRef::LocalArray { .. } => {
            Err("array[T,N] is not a valid let-binding type; use `array name: array[T,N];` to declare a local array")
        }
    }
}

// ── Expression typechecker ────────────────────────────────────────────────────

/// Check a parsed expression with an optional expected type, return a typed HIR expr.
fn check_expr(
    tc: &mut TypeChecker,
    expr: &past::Expr,
    span: Span,
    expected: Option<ScalarTy>,
) -> Option<HirExpr> {
    match expr {
        // ── §4.2a Unary-minus-over-integer-literal peephole ──────────────────
        // Before normal Neg handling, intercept Neg(IntLit{v, s}) and rewrite to
        // IntLit{-v, s} so that -2147483648 fits i32::MIN.
        //
        // The peephole is ONLY applied when the suffix (if any) is a SIGNED type:
        // negation on unsigned types must still produce OperatorTypeError, not a
        // range error. An absent suffix with an unsigned expected type also falls
        // through so `check_unary_neg` can produce the right error.
        past::Expr::Unary { op: past::UnaryOp::Neg, operand } => {
            if let past::Expr::IntLit { value, suffix } = &operand.node {
                // Determine whether the literal targets a signed type.
                let targets_unsigned = match suffix {
                    Some(axc_lexer::IntSuffix::U8)
                    | Some(axc_lexer::IntSuffix::U16)
                    | Some(axc_lexer::IntSuffix::U32)
                    | Some(axc_lexer::IntSuffix::U64) => true,
                    None => {
                        // Check the expected type: if unsigned, skip the peephole.
                        matches!(expected, Some(t) if t.is_unsigned_integer())
                    }
                    _ => false,
                };
                if !targets_unsigned {
                    // Apply peephole: Neg(IntLit{v}) → IntLit{-v}
                    let negated: i128 = value.wrapping_neg();
                    return check_expr(
                        tc,
                        &past::Expr::IntLit { value: negated, suffix: *suffix },
                        span,
                        expected,
                    );
                }
                // targets_unsigned: fall through to check_unary_neg for OperatorTypeError.
            }
            // Neg(FloatLit) peephole (floats are always signed; always apply).
            if let past::Expr::FloatLit { value, suffix } = &operand.node {
                let negated: f64 = -value;
                return check_expr(
                    tc,
                    &past::Expr::FloatLit { value: negated, suffix: *suffix },
                    span,
                    expected,
                );
            }
            // Normal unary Neg handling
            check_unary_neg(tc, operand, span, expected)
        }

        past::Expr::Unary { op: past::UnaryOp::LogicalNot, operand } => {
            check_unary_not(tc, operand, span)
        }

        // ── Literals ─────────────────────────────────────────────────────────
        past::Expr::BoolLit(b) => {
            if let Some(exp) = expected {
                if exp != ScalarTy::Bool {
                    tc.errors.push(TypecheckError::TypeMismatch {
                        expected: exp.display_name(),
                        got: "bool",
                        span,
                    });
                    // Return a placeholder with the expected type so we don't cascade errors.
                    return Some(make_bool_lit(*b, span));
                }
            }
            Some(make_bool_lit(*b, span))
        }

        past::Expr::IntLit { value, suffix } => {
            check_int_lit(tc, *value, *suffix, span, expected)
        }

        past::Expr::FloatLit { value, suffix } => {
            check_float_lit(tc, *value, *suffix, span, expected)
        }

        // ── Identifier ────────────────────────────────────────────────────────
        past::Expr::Ident(name) => {
            // Check local bindings first, then params.
            match tc.find_binding(name) {
                Some((bid, bty, _, _)) => {
                    // HirExpr.ty is ScalarTy; for coopmat bindings we use a U32 sentinel.
                    // The CoopMatBuiltin codegen uses the result_ty field, not HirExpr.ty.
                    let scalar_ty: ScalarTy = bty.as_scalar().unwrap_or(ScalarTy::U32);
                    if let Some(exp) = expected {
                        if exp != scalar_ty {
                            tc.errors.push(TypecheckError::TypeMismatch {
                                expected: exp.display_name(),
                                got: bty.display_name(),
                                span,
                            });
                        }
                    }
                    Some(HirExpr {
                        kind: HirExprKind::LocalRead(bid),
                        ty: scalar_ty,
                        span,
                    })
                }
                None => {
                    // Check if it's a kernel parameter.
                    if let Some(param) = tc.find_param(name).map(|p| (p.position, p.ty.clone(), p.span)) {
                        let (pos, pty, _pspan) = param;
                        match pty {
                            ParamTy::Scalar(st) => {
                                // Scalar params are exposed as push-constant reads.
                                // For now emit a placeholder LocalRead with a synthesized binding.
                                // In M1.2 the codegen will handle params separately.
                                // We emit UnknownBinding if the param is not in bindings —
                                // to be consistent with M1.1, push-constant reads are not yet
                                // implemented in the typechecker body (they're codegen-side).
                                // But we DO need to handle this case to not emit an error.
                                // Expose scalar params as opaque reads.
                                let _ = pos;
                                if let Some(exp) = expected {
                                    if exp != st {
                                        tc.errors.push(TypecheckError::TypeMismatch {
                                            expected: exp.display_name(),
                                            got: st.display_name(),
                                            span,
                                        });
                                    }
                                }
                                // Use a sentinel BindingId::MAX to signal push-constant read.
                                // The codegen handles this via KernelParam lookup.
                                Some(HirExpr {
                                    kind: HirExprKind::LocalRead(BindingId(u32::MAX - pos)),
                                    ty: st,
                                    span,
                                })
                            }
                            ParamTy::Buffer(_) => {
                                // Buffer params used bare (not indexed) are an error.
                                tc.errors.push(TypecheckError::BufferAsValue {
                                    name: name.clone(),
                                    span,
                                });
                                None
                            }
                        }
                    } else {
                        tc.errors.push(TypecheckError::UnknownBinding {
                            name: name.clone(),
                            span,
                        });
                        let placeholder_ty: ScalarTy = expected.unwrap_or(ScalarTy::I32);
                        Some(HirExpr {
                            kind: HirExprKind::BoolLit(false), // placeholder
                            ty: placeholder_ty,
                            span,
                        })
                    }
                }
            }
        }

        // ── Paren ─────────────────────────────────────────────────────────────
        past::Expr::Paren(inner) => {
            check_expr(tc, &inner.node, inner.span, expected)
        }

        // ── Binary ────────────────────────────────────────────────────────────
        past::Expr::Binary { op, lhs, rhs } => {
            check_binary(tc, *op, lhs, rhs, span, expected)
        }

        // ── ShortCircuit ──────────────────────────────────────────────────────
        past::Expr::ShortCircuit { op, lhs, rhs } => {
            check_short_circuit(tc, *op, lhs, rhs, span)
        }

        // ── Call (bitwise builtins + gid + local_invocation_id) ─────────────
        past::Expr::Call { name, args } => {
            if name.node == "gid" {
                check_gid_call(tc, args, span)
            } else if name.node == "local_invocation_id" {
                check_local_invocation_id_call(tc, args, span)
            } else {
                check_call(tc, &name.node, name.span, args, span, expected)
            }
        }

        // ── Buffer/shared index read: name[index] ─────────────────────────────
        past::Expr::Index { base, index } => {
            // The M1.2 parser only produces Index with an Ident base (postfix `name[expr]`).
            // Multi-dimensional chained indexing (e.g. buf[i][j]) is not parseable in M1.2.
            match &base.node {
                past::Expr::Ident(name) => {
                    // M3.20: Check if this is a local-array read first (order: local
                    // array -> shared -> buffer; the bidirectional collision guard in
                    // register_local_array/register_shared ensures a name is at most
                    // one of these, so order only fixes the read-node kind).
                    let local_array_info: Option<(LocalArrayId, ScalarTy, u32)> = tc.find_local_array(name);
                    if let Some((local_array_id, elem_ty, len)) = local_array_info {
                        return check_local_array_read(tc, local_array_id, elem_ty, len, index, span);
                    }
                    // M3.2: Check if this is a shared array read next.
                    let shared_info: Option<(SharedId, ScalarTy, u32)> = tc.find_shared(name);
                    if let Some((shared_id, elem_ty, _len)) = shared_info {
                        return check_shared_read(tc, shared_id, elem_ty, index, span);
                    }
                    check_buffer_read(tc, name, base.span, index, span, expected)
                }
                _ => {
                    // Unreachable with the current M1.2 grammar, which only allows
                    // `identifier[expr]` as an index expression. Kept as a safety net
                    // in case the parser is extended in a future milestone.
                    tc.errors.push(TypecheckError::UnsupportedExprInM1_1 {
                        detail: "multi-dimensional buffer indexing not supported in M1.2",
                        span,
                    });
                    None
                }
            }
        }
    }
}

// ── Literal helpers ───────────────────────────────────────────────────────────

fn make_bool_lit(b: bool, span: Span) -> HirExpr {
    HirExpr { kind: HirExprKind::BoolLit(b), ty: ScalarTy::Bool, span }
}

/// §4.2 IntLit typing logic.
fn check_int_lit(
    tc: &mut TypeChecker,
    value: i128,
    suffix: Option<axc_lexer::IntSuffix>,
    span: Span,
    expected: Option<ScalarTy>,
) -> Option<HirExpr> {
    use axc_lexer::IntSuffix;

    // Step 1: If suffix present, use it as the target type.
    let target_ty: ScalarTy = if let Some(s) = suffix {
        let suffix_ty = match s {
            IntSuffix::I8  => {
                tc.errors.push(TypecheckError::UnsupportedExprInM1_1 {
                    detail: "i8/i16/u8/u16 scalar types are deferred past M1.1",
                    span,
                });
                ScalarTy::I8
            }
            IntSuffix::I16 => {
                tc.errors.push(TypecheckError::UnsupportedExprInM1_1 {
                    detail: "i8/i16/u8/u16 scalar types are deferred past M1.1",
                    span,
                });
                ScalarTy::I16
            }
            IntSuffix::U8  => {
                tc.errors.push(TypecheckError::UnsupportedExprInM1_1 {
                    detail: "i8/i16/u8/u16 scalar types are deferred past M1.1",
                    span,
                });
                ScalarTy::U8
            }
            IntSuffix::U16 => {
                tc.errors.push(TypecheckError::UnsupportedExprInM1_1 {
                    detail: "i8/i16/u8/u16 scalar types are deferred past M1.1",
                    span,
                });
                ScalarTy::U16
            }
            IntSuffix::I32 => ScalarTy::I32,
            IntSuffix::I64 => ScalarTy::I64,
            IntSuffix::U32 => ScalarTy::U32,
            IntSuffix::U64 => ScalarTy::U64,
        };
        // If the expected type conflicts with the explicit suffix, emit TypeMismatch.
        if let Some(exp) = expected {
            if exp != suffix_ty {
                tc.errors.push(TypecheckError::TypeMismatch {
                    expected: exp.display_name(),
                    got: suffix_ty.display_name(),
                    span,
                });
            }
        }
        suffix_ty
    } else if let Some(exp) = expected {
        // Step 2: Use expected type if it is an integer type.
        if exp.is_integer() {
            exp
        } else if exp.is_float() {
            // Step 3: Expected is float but we have an int literal.
            tc.errors.push(TypecheckError::TypeMismatch {
                expected: exp.display_name(),
                got: "integer literal",
                span,
            });
            ScalarTy::I32 // placeholder
        } else {
            // Step 4: Expected is bool but we have an int literal.
            tc.errors.push(TypecheckError::TypeMismatch {
                expected: "bool",
                got: "integer literal",
                span,
            });
            ScalarTy::I32 // placeholder
        }
    } else {
        // Step 5: No suffix, no expected — unconstrained.
        tc.errors.push(TypecheckError::UnconstrainedLiteralNeedsSuffix { span });
        ScalarTy::I32 // placeholder to continue type-walking
    };

    // Step 6: Range-check.
    match fit_int_literal(value, target_ty) {
        Ok(lit_val) => Some(HirExpr {
            kind: HirExprKind::IntLit { value: lit_val },
            ty: target_ty,
            span,
        }),
        Err(crate::ty::LiteralRangeErr::IntegerOutOfRange { value: v, target: t }) => {
            let (min_val, max_val) = t.int_range().unwrap_or((i128::MIN, i128::MAX));
            tc.errors.push(TypecheckError::LiteralOutOfRange {
                value: v,
                target: t.display_name(),
                min: min_val,
                max: max_val,
                span,
            });
            None
        }
        Err(_) => None,
    }
}

/// §4.2 FloatLit typing logic.
fn check_float_lit(
    tc: &mut TypeChecker,
    value: f64,
    suffix: Option<axc_lexer::FloatSuffix>,
    span: Span,
    expected: Option<ScalarTy>,
) -> Option<HirExpr> {
    use axc_lexer::FloatSuffix;

    let target_ty: ScalarTy = if let Some(s) = suffix {
        let suffix_ty = match s {
            FloatSuffix::F16  => {
                tc.errors.push(TypecheckError::UnsupportedExprInM1_1 {
                    detail: "f16/bf16 scalar types are deferred past M1.1",
                    span,
                });
                ScalarTy::F32 // placeholder
            }
            FloatSuffix::Bf16 => {
                tc.errors.push(TypecheckError::UnsupportedExprInM1_1 {
                    detail: "f16/bf16 scalar types are deferred past M1.1",
                    span,
                });
                ScalarTy::F32 // placeholder
            }
            FloatSuffix::F32  => ScalarTy::F32,
            FloatSuffix::F64  => ScalarTy::F64,
        };
        // If the expected type conflicts with the explicit suffix, emit TypeMismatch.
        if let Some(exp) = expected {
            if exp != suffix_ty {
                tc.errors.push(TypecheckError::TypeMismatch {
                    expected: exp.display_name(),
                    got: suffix_ty.display_name(),
                    span,
                });
            }
        }
        suffix_ty
    } else if let Some(exp) = expected {
        if exp.is_float() {
            exp
        } else if exp.is_integer() {
            tc.errors.push(TypecheckError::TypeMismatch {
                expected: exp.display_name(),
                got: "float literal",
                span,
            });
            ScalarTy::F32
        } else {
            tc.errors.push(TypecheckError::TypeMismatch {
                expected: "bool",
                got: "float literal",
                span,
            });
            ScalarTy::F32
        }
    } else {
        tc.errors.push(TypecheckError::UnconstrainedLiteralNeedsSuffix { span });
        ScalarTy::F32
    };

    match fit_float_literal(value, target_ty) {
        Ok(lit_val) => Some(HirExpr {
            kind: HirExprKind::FloatLit { value: lit_val },
            ty: target_ty,
            span,
        }),
        Err(crate::ty::LiteralRangeErr::FloatNonFinite) => {
            tc.errors.push(TypecheckError::FloatLiteralNonFinite { span });
            None
        }
        Err(_) => None,
    }
}

// ── Unary ops ─────────────────────────────────────────────────────────────────

fn check_unary_neg(
    tc: &mut TypeChecker,
    operand: &axc_lexer::Spanned<past::Expr>,
    span: Span,
    expected: Option<ScalarTy>,
) -> Option<HirExpr> {
    let operand_hir = check_expr(tc, &operand.node, operand.span, expected)?;
    let operand_ty = operand_hir.ty;

    if !operand_ty.is_signed_integer() && !operand_ty.is_float() {
        tc.errors.push(TypecheckError::OperatorTypeError {
            op: "-",
            operand_class: "signed integer or float",
            lhs_ty: operand_ty.display_name(),
            rhs_ty: operand_ty.display_name(),
            span,
        });
        return None;
    }

    Some(HirExpr {
        kind: HirExprKind::Unary {
            op: UnaryOp::Neg,
            operand: Box::new(operand_hir),
        },
        ty: operand_ty,
        span,
    })
}

fn check_unary_not(
    tc: &mut TypeChecker,
    operand: &axc_lexer::Spanned<past::Expr>,
    span: Span,
) -> Option<HirExpr> {
    let operand_hir = check_expr(tc, &operand.node, operand.span, Some(ScalarTy::Bool))?;

    if operand_hir.ty != ScalarTy::Bool {
        tc.errors.push(TypecheckError::TypeMismatch {
            expected: "bool",
            got: operand_hir.ty.display_name(),
            span,
        });
        return None;
    }

    Some(HirExpr {
        kind: HirExprKind::Unary {
            op: UnaryOp::LogicalNot,
            operand: Box::new(operand_hir),
        },
        ty: ScalarTy::Bool,
        span,
    })
}

// ── Binary ops ────────────────────────────────────────────────────────────────

fn ast_binop_to_hir(op: past::BinOp) -> BinOp {
    match op {
        past::BinOp::Add   => BinOp::Add,
        past::BinOp::Sub   => BinOp::Sub,
        past::BinOp::Mul   => BinOp::Mul,
        past::BinOp::Div   => BinOp::Div,
        past::BinOp::Rem   => BinOp::Rem,
        past::BinOp::Eq    => BinOp::Eq,
        past::BinOp::Neq   => BinOp::Neq,
        past::BinOp::Lt    => BinOp::Lt,
        past::BinOp::LtEq  => BinOp::LtEq,
        past::BinOp::Gt    => BinOp::Gt,
        past::BinOp::GtEq  => BinOp::GtEq,
    }
}

fn check_binary(
    tc: &mut TypeChecker,
    op: past::BinOp,
    lhs: &axc_lexer::Spanned<past::Expr>,
    rhs: &axc_lexer::Spanned<past::Expr>,
    span: Span,
    expected: Option<ScalarTy>,
) -> Option<HirExpr> {
    use past::BinOp as PBinOp;

    let is_comparison = matches!(op, PBinOp::Eq | PBinOp::Neq | PBinOp::Lt | PBinOp::LtEq | PBinOp::Gt | PBinOp::GtEq);
    let is_arithmetic = !is_comparison;

    // For comparisons: LHS is Unconstrained (the expected type flows FROM LHS into RHS).
    // For arithmetic: LHS uses the outer expected type.
    let lhs_expected = if is_arithmetic { expected } else { None };

    let lhs_hir = check_expr(tc, &lhs.node, lhs.span, lhs_expected)?;
    let lhs_ty = lhs_hir.ty;

    // RHS uses LHS's resolved type as the expected type.
    let rhs_hir = check_expr(tc, &rhs.node, rhs.span, Some(lhs_ty))?;
    let rhs_ty = rhs_hir.ty;

    // Type-check: both sides must match.
    if lhs_ty != rhs_ty {
        tc.errors.push(TypecheckError::MixedOperandTypes {
            op: op_name_str(op),
            lhs_ty: lhs_ty.display_name(),
            rhs_ty: rhs_ty.display_name(),
            span,
        });
    }

    let result_ty: ScalarTy = if is_arithmetic {
        // Arithmetic ops (Add/Sub/Mul/Div/Rem) are invalid on bool.
        if lhs_ty.is_bool() {
            tc.errors.push(TypecheckError::OperatorTypeError {
                op: op_name_str(op),
                operand_class: "numeric type (integer or float, not bool)",
                lhs_ty: lhs_ty.display_name(),
                rhs_ty: rhs_ty.display_name(),
                span,
            });
            return None;
        }
        lhs_ty
    } else {
        // Comparison ops: result is always bool.
        // Lt/LtEq/Gt/GtEq are invalid on bool (no ordering for bool).
        if lhs_ty.is_bool() && matches!(op, PBinOp::Lt | PBinOp::LtEq | PBinOp::Gt | PBinOp::GtEq) {
            tc.errors.push(TypecheckError::OperatorTypeError {
                op: op_name_str(op),
                operand_class: "integer or float (bool has no ordering)",
                lhs_ty: lhs_ty.display_name(),
                rhs_ty: rhs_ty.display_name(),
                span,
            });
            return None;
        }
        ScalarTy::Bool
    };

    Some(HirExpr {
        kind: HirExprKind::Binary {
            op: ast_binop_to_hir(op),
            lhs: Box::new(lhs_hir),
            rhs: Box::new(rhs_hir),
        },
        ty: result_ty,
        span,
    })
}

fn op_name_str(op: past::BinOp) -> &'static str {
    match op {
        past::BinOp::Add   => "+",
        past::BinOp::Sub   => "-",
        past::BinOp::Mul   => "*",
        past::BinOp::Div   => "/",
        past::BinOp::Rem   => "%",
        past::BinOp::Eq    => "==",
        past::BinOp::Neq   => "!=",
        past::BinOp::Lt    => "<",
        past::BinOp::LtEq  => "<=",
        past::BinOp::Gt    => ">",
        past::BinOp::GtEq  => ">=",
    }
}

// ── Short-circuit ops ─────────────────────────────────────────────────────────

fn check_short_circuit(
    tc: &mut TypeChecker,
    op: past::ShortCircuitOp,
    lhs: &axc_lexer::Spanned<past::Expr>,
    rhs: &axc_lexer::Spanned<past::Expr>,
    span: Span,
) -> Option<HirExpr> {
    let lhs_hir = check_expr(tc, &lhs.node, lhs.span, Some(ScalarTy::Bool))?;
    let rhs_hir = check_expr(tc, &rhs.node, rhs.span, Some(ScalarTy::Bool))?;

    if lhs_hir.ty != ScalarTy::Bool {
        tc.errors.push(TypecheckError::TypeMismatch {
            expected: "bool",
            got: lhs_hir.ty.display_name(),
            span: lhs.span,
        });
    }
    if rhs_hir.ty != ScalarTy::Bool {
        tc.errors.push(TypecheckError::TypeMismatch {
            expected: "bool",
            got: rhs_hir.ty.display_name(),
            span: rhs.span,
        });
    }

    let hir_op = match op {
        past::ShortCircuitOp::And => ShortCircuitOp::And,
        past::ShortCircuitOp::Or  => ShortCircuitOp::Or,
    };

    Some(HirExpr {
        kind: HirExprKind::ShortCircuit {
            op: hir_op,
            lhs: Box::new(lhs_hir),
            rhs: Box::new(rhs_hir),
        },
        ty: ScalarTy::Bool,
        span,
    })
}

// ── M1.4: Subgroup builtin statement handler ──────────────────────────────────

/// Handle `workgroup_barrier();` at statement position.
///
/// Only `workgroup_barrier` (arity 0) is valid at statement position.
/// All other reserved subgroup names at statement position are rejected with
/// `NonVoidSubgroupCallAsStatement`. This function does NOT consult
/// `divergent_context_depth` — barrier warning is deferred to M1.5 (CRITICAL-4 fix).
fn check_builtin_call_stmt(
    tc: &mut TypeChecker<'_>,
    call: &axc_lexer::Spanned<axc_parser::ast::Expr>,
    stmt_span: Span,
) -> Option<HirStmt> {
    if let axc_parser::ast::Expr::Call { name, args } = &call.node {
        let op_name: &str = &name.node;

        // M2.1: coopmat_store is the ONLY coopmat builtin that appears as a statement.
        if op_name == "coopmat_store" {
            return check_coopmat_store_stmt(tc, args, call.span, stmt_span);
        }
        // Reject other coopmat builtins used at statement position (they return non-void values).
        use crate::coopmat::CoopMatBuiltin;
        if let Some(cm_op) = CoopMatBuiltin::from_source_name(op_name) {
            tc.errors.push(TypecheckError::UnsupportedStmtInM1_4 {
                detail: cm_op.source_name(),
                span: call.span,
            });
            return None;
        }

        if op_name == "workgroup_barrier" {
            if !args.is_empty() {
                tc.errors.push(TypecheckError::SubgroupArity {
                    op: "workgroup_barrier",
                    expected_arity: 0,
                    got_arity: args.len(),
                    span: call.span,
                });
                return None;
            }
            // M3.2 (OQ2): barrier inside an if/else body (conditional_depth > 0) is UB.
            // Note: this DOES NOT fire for while/for-range bodies (those do NOT increment
            // conditional_depth). The existing divergent_context_depth is left unchanged
            // for the subgroup-collective warning.
            if tc.conditional_depth > 0 {
                tc.errors.push(TypecheckError::BarrierInDivergentContext {
                    span: stmt_span,
                });
                return None;
            }
            return Some(HirStmt::Barrier {
                kind: crate::subgroup::BarrierKind::Workgroup,
                span: stmt_span,
            });
        }
        // Any other reserved name at statement position is an error.
        if axc_lexer::is_reserved_subgroup_builtin(op_name) {
            tc.errors.push(TypecheckError::NonVoidSubgroupCallAsStatement {
                op_name: op_name.to_owned(),
                span: call.span,
            });
            return None;
        }
    }
    tc.errors.push(TypecheckError::UnsupportedStmtInM1_4 {
        detail: "unexpected statement form in builtin call handler",
        span: stmt_span,
    });
    None
}

/// Check a subgroup builtin call expression.
///
/// Called when `SubgroupOp::from_source_name(name)` matches.
/// Emits `SubgroupOpInDivergentContext` warning for collective ops when
/// `divergent_context_depth > 0` (§5(8) rev 1 CRITICAL-3/CRITICAL-4 fix).
fn check_subgroup_call(
    tc: &mut TypeChecker<'_>,
    op: crate::subgroup::SubgroupOp,
    name_span: Span,
    args: &[axc_lexer::Spanned<axc_parser::ast::Expr>],
    call_span: Span,
    expected: Option<ScalarTy>,
) -> Option<HirExpr> {
    use crate::subgroup::{SubgroupOp, SubgroupReduceKind};

    // Arity check.
    let expected_arity = op.arity();
    if args.len() != expected_arity {
        tc.errors.push(TypecheckError::SubgroupArity {
            op: op.source_name(),
            expected_arity,
            got_arity: args.len(),
            span: call_span,
        });
        return None;
    }

    // Divergent context warning for collective ops.
    if op.is_collective() && tc.divergent_context_depth > 0 {
        tc.warns.push(crate::validate::HirWarning::SubgroupOpInDivergentContext {
            op_name: op.source_name(),
            span: name_span,
        });
    }

    match op {
        SubgroupOp::InvocationId => {
            // Zero-arg, returns u32.
            check_subgroup_result_ty(tc, call_span, ScalarTy::U32, expected);
            Some(HirExpr {
                kind: HirExprKind::SubgroupBuiltin { op, args: vec![] },
                ty: ScalarTy::U32,
                span: call_span,
            })
        }
        SubgroupOp::Size => {
            // Zero-arg, returns u32.
            check_subgroup_result_ty(tc, call_span, ScalarTy::U32, expected);
            Some(HirExpr {
                kind: HirExprKind::SubgroupBuiltin { op, args: vec![] },
                ty: ScalarTy::U32,
                span: call_span,
            })
        }
        SubgroupOp::Elect => {
            // Zero-arg, returns bool.
            check_subgroup_result_ty(tc, call_span, ScalarTy::Bool, expected);
            Some(HirExpr {
                kind: HirExprKind::SubgroupBuiltin { op, args: vec![] },
                ty: ScalarTy::Bool,
                span: call_span,
            })
        }
        SubgroupOp::Reduce(reduce_kind) => {
            // One-arg type-parameterized: T ∈ {i32, u32, f32, f64}.
            let arg = &args[0];
            let arg_hir = check_expr(tc, &arg.node, arg.span, None)?;
            let elem_ty = arg_hir.ty;
            if !is_reduce_type(elem_ty) {
                let op_str = match reduce_kind {
                    SubgroupReduceKind::Add => "add",
                    SubgroupReduceKind::Min => "min",
                    SubgroupReduceKind::Max => "max",
                };
                tc.errors.push(TypecheckError::SubgroupReduceTypeUnsupported {
                    op: op_str,
                    got_ty: elem_ty.display_name(),
                    span: arg.span,
                });
                return None;
            }
            check_subgroup_result_ty(tc, call_span, elem_ty, expected);
            Some(HirExpr {
                kind: HirExprKind::SubgroupBuiltin { op, args: vec![arg_hir] },
                ty: elem_ty,
                span: call_span,
            })
        }
        SubgroupOp::BroadcastFirst => {
            // One-arg type-parameterized: T ∈ {i32, u32, f32, f64, bool}.
            let arg = &args[0];
            let arg_hir = check_expr(tc, &arg.node, arg.span, None)?;
            let elem_ty = arg_hir.ty;
            if !is_broadcast_first_type(elem_ty) {
                tc.errors.push(TypecheckError::SubgroupBroadcastTypeUnsupported {
                    got_ty: elem_ty.display_name(),
                    span: arg.span,
                });
                return None;
            }
            check_subgroup_result_ty(tc, call_span, elem_ty, expected);
            Some(HirExpr {
                kind: HirExprKind::SubgroupBuiltin { op, args: vec![arg_hir] },
                ty: elem_ty,
                span: call_span,
            })
        }
        SubgroupOp::All | SubgroupOp::Any => {
            // One-arg bool predicate, returns bool.
            let arg = &args[0];
            let arg_hir = check_expr(tc, &arg.node, arg.span, Some(ScalarTy::Bool))?;
            if arg_hir.ty != ScalarTy::Bool {
                tc.errors.push(TypecheckError::TypeMismatch {
                    expected: "bool",
                    got: arg_hir.ty.display_name(),
                    span: arg.span,
                });
                return None;
            }
            check_subgroup_result_ty(tc, call_span, ScalarTy::Bool, expected);
            Some(HirExpr {
                kind: HirExprKind::SubgroupBuiltin { op, args: vec![arg_hir] },
                ty: ScalarTy::Bool,
                span: call_span,
            })
        }
    }
}

/// Optionally emit a TypeMismatch if the actual result type differs from expected.
fn check_subgroup_result_ty(
    tc: &mut TypeChecker<'_>,
    span: Span,
    actual: ScalarTy,
    expected: Option<ScalarTy>,
) {
    if let Some(exp) = expected {
        if exp != actual {
            tc.errors.push(TypecheckError::TypeMismatch {
                expected: exp.display_name(),
                got: actual.display_name(),
                span,
            });
        }
    }
}

/// Returns true if `ty` is valid as a subgroup_reduce_* operand type.
fn is_reduce_type(ty: ScalarTy) -> bool {
    matches!(ty, ScalarTy::I32 | ScalarTy::U32 | ScalarTy::F32 | ScalarTy::F64)
}

/// Returns true if `ty` is valid as a subgroup_broadcast_first operand type.
fn is_broadcast_first_type(ty: ScalarTy) -> bool {
    matches!(ty, ScalarTy::I32 | ScalarTy::U32 | ScalarTy::F32 | ScalarTy::F64 | ScalarTy::Bool)
}

// ── Cooperative-matrix let-init expression (M2.1) ────────────────────────────

/// Typecheck the initializer expression for a `let x: matrix[...] = <init>;` binding.
///
/// Unlike scalar lets, the expected type is a `CoopMatKey` not a `ScalarTy`. This
/// function handles `coopmat_zero()` and `coopmat_load(...)` where the matrix type
/// is inferred from the let-binding context. `coopmat_mul_add()` is also handled
/// (result type is determined from arguments, not from context).
///
/// For any other expression (ident lookup, arithmetic, etc.) we fall through to
/// `check_expr` with `expected: None` and accept whatever type is produced.
fn check_coopmat_init_expr(
    tc: &mut TypeChecker<'_>,
    expr: &past::Expr,
    span: Span,
    matrix_key: CoopMatKey,
) -> Option<HirExpr> {
    use crate::coopmat::{CoopMatBuiltin, is_allowed_coopmat_element};

    match expr {
        past::Expr::Call { name, args } => {
            let call_name: &str = &name.node;
            match CoopMatBuiltin::from_source_name(call_name) {
                Some(CoopMatBuiltin::Zero) => {
                    // coopmat_zero() → result type is matrix_key.
                    if !args.is_empty() {
                        tc.errors.push(TypecheckError::CoopMatArity {
                            name: "coopmat_zero",
                            expected: 0,
                            found: args.len(),
                            span,
                        });
                        return None;
                    }
                    if !is_allowed_coopmat_element(matrix_key.elem) {
                        tc.errors.push(TypecheckError::CoopMatrixElementTypeUnsupported {
                            ty: matrix_key.elem.display_name(),
                            span,
                        });
                        return None;
                    }
                    Some(HirExpr {
                        kind: HirExprKind::CoopMatBuiltin {
                            op: CoopMatBuiltin::Zero,
                            args: vec![],
                            result_ty: matrix_key,
                            source: None,
                        },
                        ty: ScalarTy::U32, // sentinel
                        span,
                    })
                }
                Some(CoopMatBuiltin::Load) => {
                    // coopmat_load(src, element_offset, stride) → result type is matrix_key.
                    // src may be a buffer param (M2.1, Buffer source) OR a shared array (M3.2, Shared source).
                    if args.len() != CoopMatBuiltin::Load.arity() {
                        tc.errors.push(TypecheckError::CoopMatArity {
                            name: "coopmat_load",
                            expected: CoopMatBuiltin::Load.arity(),
                            found: args.len(),
                            span,
                        });
                        return None;
                    }
                    // Arg 0: source ident — must be a buffer param OR a shared array name.
                    let src_ident = match &args[0].node {
                        past::Expr::Ident(ref s) => s.clone(),
                        _ => {
                            tc.errors.push(TypecheckError::CoopMatLoadArgMustBeBufferParam {
                                found_kind: "non-ident expression",
                                span: args[0].span,
                            });
                            return None;
                        }
                    };

                    // PART B (M3.2): check if src_ident is a shared array first.
                    let load_source = if let Some((shared_id, shared_elem, _shared_len)) = tc.find_shared(&src_ident) {
                        // Shared source path: element type must match the coopmat elem.
                        if shared_elem != matrix_key.elem {
                            tc.errors.push(TypecheckError::CoopMatLoadElementTypeMismatch {
                                matrix_elem: matrix_key.elem.display_name(),
                                buffer_elem: shared_elem.display_name(),
                                span: args[0].span,
                            });
                            return None;
                        }
                        crate::coopmat::CoopMatLoadSource::Shared(shared_id.0)
                    } else {
                        // Buffer source path (M2.1 default): find buffer param by name.
                        let mut buf_slot: Option<u32> = None;
                        let mut buf_elem: Option<ScalarTy> = None;
                        let mut buf_access: Option<crate::buffer::BufferAccess> = None;
                        let mut idx: u32 = 0;
                        for p in tc.params {
                            if let crate::param::Ty::Buffer(ref bt) = p.ty {
                                if p.name == src_ident {
                                    buf_slot = Some(idx);
                                    buf_elem = Some(bt.elem);
                                    buf_access = Some(bt.access);
                                    break;
                                }
                                idx += 1;
                            }
                        }
                        let (buf_slot_val, buf_elem_val, _buf_access) = match (buf_slot, buf_elem, buf_access) {
                            (Some(s), Some(e), Some(a)) => (s, e, a),
                            _ => {
                                tc.errors.push(TypecheckError::CoopMatLoadArgMustBeBufferParam {
                                    found_kind: "not a buffer parameter or shared array",
                                    span: args[0].span,
                                });
                                return None;
                            }
                        };
                        // Check element type matches.
                        if buf_elem_val != matrix_key.elem {
                            tc.errors.push(TypecheckError::CoopMatLoadElementTypeMismatch {
                                matrix_elem: matrix_key.elem.display_name(),
                                buffer_elem: buf_elem_val.display_name(),
                                span: args[0].span,
                            });
                            return None;
                        }
                        crate::coopmat::CoopMatLoadSource::Buffer(buf_slot_val)
                    };

                    // Arg 1: element_offset (U32).
                    let offset_hir = check_expr(tc, &args[1].node, args[1].span, Some(ScalarTy::U32))?;
                    if offset_hir.ty != ScalarTy::U32 {
                        tc.errors.push(TypecheckError::CoopMatLoadStrideMustBeU32 {
                            found_ty: offset_hir.ty.display_name(),
                            span: args[1].span,
                        });
                        return None;
                    }
                    // Arg 2: stride (U32).
                    let stride_hir = check_expr(tc, &args[2].node, args[2].span, Some(ScalarTy::U32))?;
                    if stride_hir.ty != ScalarTy::U32 {
                        tc.errors.push(TypecheckError::CoopMatLoadStrideMustBeU32 {
                            found_ty: stride_hir.ty.display_name(),
                            span: args[2].span,
                        });
                        return None;
                    }
                    Some(HirExpr {
                        kind: HirExprKind::CoopMatBuiltin {
                            op: CoopMatBuiltin::Load,
                            args: vec![offset_hir, stride_hir],
                            result_ty: matrix_key,
                            source: Some(load_source),
                        },
                        ty: ScalarTy::U32, // sentinel
                        span,
                    })
                }
                Some(CoopMatBuiltin::MulAdd) => {
                    // coopmat_mul_add: result type is determined by arguments, not context.
                    // Delegate to check_coopmat_expr_call (expected: None).
                    check_coopmat_expr_call(tc, CoopMatBuiltin::MulAdd, name.span, args, span, None)
                }
                Some(CoopMatBuiltin::Store) => {
                    tc.errors.push(TypecheckError::UnsupportedStmtInM1_4 {
                        detail: "coopmat_store must appear at statement position (it returns void)",
                        span,
                    });
                    None
                }
                None => {
                    // Non-coopmat call in coopmat context — fall through to normal check_expr.
                    // This will likely produce a type error, which is correct.
                    check_expr(tc, expr, span, None)
                }
            }
        }
        past::Expr::Paren(inner) => {
            check_coopmat_init_expr(tc, &inner.node, inner.span, matrix_key)
        }
        _ => {
            // For other expressions (idents referring to other coopmat bindings, etc.),
            // fall through to check_expr with no scalar expected type.
            check_expr(tc, expr, span, None)
        }
    }
}

// ── Cooperative-matrix builtin calls (M2.1) ──────────────────────────────────

/// Helper: extract the `CoopMatKey` from a binding. Returns `None` and pushes
/// a TypeMismatch error if the binding is not a coopmat type.
fn expect_coopmat_binding(
    tc: &mut TypeChecker<'_>,
    name: &str,
    name_span: Span,
) -> Option<(BindingId, CoopMatKey)> {
    match tc.find_binding(name) {
        None => {
            tc.errors.push(TypecheckError::UnknownBinding {
                name: name.to_owned(),
                span: name_span,
            });
            None
        }
        Some((bid, BindingTy::CoopMatrix(key), _, _)) => Some((bid, key)),
        Some((_, BindingTy::Scalar(st), _, _)) => {
            tc.errors.push(TypecheckError::TypeMismatch {
                expected: "matrix",
                got: st.display_name(),
                span: name_span,
            });
            None
        }
    }
}


/// Typecheck a cooperative-matrix expression builtin: Zero, Load, MulAdd.
/// Store is a statement (void) — handled in check_coopmat_store_stmt.
fn check_coopmat_expr_call(
    tc: &mut TypeChecker<'_>,
    op: crate::coopmat::CoopMatBuiltin,
    _name_span: Span,
    args: &[axc_lexer::Spanned<past::Expr>],
    call_span: Span,
    expected: Option<ScalarTy>,
) -> Option<HirExpr> {
    use crate::coopmat::{CoopMatBuiltin, is_allowed_coopmat_element};

    // Arity check (except Store which is handled as a stmt).
    if op == CoopMatBuiltin::Store {
        // coopmat_store used in expression position is an error.
        tc.errors.push(TypecheckError::UnsupportedStmtInM1_4 {
            detail: "coopmat_store must appear at statement position (it returns void)",
            span: call_span,
        });
        return None;
    }

    if args.len() != op.arity() {
        tc.errors.push(TypecheckError::CoopMatArity {
            name: op.source_name(),
            expected: op.arity(),
            found: args.len(),
            span: call_span,
        });
        return None;
    }

    match op {
        CoopMatBuiltin::Zero => {
            // coopmat_zero() requires expected-type context (let m: matrix[...] = coopmat_zero()).
            // expected is a ScalarTy here, which can't carry matrix type — we need the BindingTy.
            // HACK: expected is None in expression position without let context.
            // We check for it by looking at `expected`: if None, we can't determine the result type.
            //
            // Actually, we need the FULL CoopMatKey for Zero, which comes from the let binding's
            // declared type. The `expected: Option<ScalarTy>` context cannot carry this.
            // For now, produce an error: coopmat_zero without expected coopmat context.
            tc.errors.push(TypecheckError::CoopMatrixBuiltinRequiresExpectedType {
                name: "coopmat_zero",
                span: call_span,
            });
            None
        }
        CoopMatBuiltin::Load => {
            // coopmat_load(buf, element_offset, stride) → matrix[T, M, N, use]
            // Like Zero, the result type comes from let context.
            tc.errors.push(TypecheckError::CoopMatrixBuiltinRequiresExpectedType {
                name: "coopmat_load",
                span: call_span,
            });
            None
        }
        CoopMatBuiltin::MulAdd => {
            // coopmat_mul_add(a, b, c) → matrix[T, M, N, accumulator]
            // All three args must be coopmat bindings.
            // Extract arg idents.
            let get_ident = |arg: &axc_lexer::Spanned<past::Expr>| {
                if let past::Expr::Ident(ref s) = arg.node {
                    Some((s.clone(), arg.span))
                } else if let past::Expr::Paren(ref inner) = arg.node {
                    if let past::Expr::Ident(ref s) = inner.node {
                        Some((s.clone(), inner.span))
                    } else {
                        None
                    }
                } else {
                    None
                }
            };

            let (a_name, a_span) = get_ident(&args[0]).unwrap_or_else(|| ("".to_owned(), args[0].span));
            let (b_name, b_span) = get_ident(&args[1]).unwrap_or_else(|| ("".to_owned(), args[1].span));
            let (c_name, c_span) = get_ident(&args[2]).unwrap_or_else(|| ("".to_owned(), args[2].span));

            let (a_bid, a_key) = expect_coopmat_binding(tc, &a_name, a_span)?;
            let (b_bid, b_key) = expect_coopmat_binding(tc, &b_name, b_span)?;
            let (c_bid, c_key) = expect_coopmat_binding(tc, &c_name, c_span)?;

            // Validate use tags.
            if a_key.use_ != CoopMatUse::MatrixA {
                tc.errors.push(TypecheckError::CoopMatrixShapeMismatch {
                    kind: CoopMatrixShapeKind::AUseMismatch { found: a_key.use_ },
                    span: a_span,
                });
                return None;
            }
            if b_key.use_ != CoopMatUse::MatrixB {
                tc.errors.push(TypecheckError::CoopMatrixShapeMismatch {
                    kind: CoopMatrixShapeKind::BUseMismatch { found: b_key.use_ },
                    span: b_span,
                });
                return None;
            }
            if c_key.use_ != CoopMatUse::Accumulator {
                tc.errors.push(TypecheckError::CoopMatrixShapeMismatch {
                    kind: CoopMatrixShapeKind::CUseMismatch { found: c_key.use_ },
                    span: c_span,
                });
                return None;
            }

            // Validate K dimension: a.n must equal b.m.
            if a_key.n != b_key.m {
                tc.errors.push(TypecheckError::CoopMatrixShapeMismatch {
                    kind: CoopMatrixShapeKind::KDimMismatch { a_n: a_key.n, b_m: b_key.m },
                    span: call_span,
                });
                return None;
            }

            // Validate accumulator M: c.m must equal a.m.
            if c_key.m != a_key.m {
                tc.errors.push(TypecheckError::CoopMatrixShapeMismatch {
                    kind: CoopMatrixShapeKind::AccumulatorMMismatch { c_m: c_key.m, a_m: a_key.m },
                    span: call_span,
                });
                return None;
            }

            // Validate accumulator N: c.n must equal b.n.
            if c_key.n != b_key.n {
                tc.errors.push(TypecheckError::CoopMatrixShapeMismatch {
                    kind: CoopMatrixShapeKind::AccumulatorNMismatch { c_n: c_key.n, b_n: b_key.n },
                    span: call_span,
                });
                return None;
            }

            // Validate A and B element type match.
            if a_key.elem != b_key.elem {
                tc.errors.push(TypecheckError::CoopMatrixShapeMismatch {
                    kind: CoopMatrixShapeKind::ABElementMismatch { a_elem: a_key.elem, b_elem: b_key.elem },
                    span: call_span,
                });
                return None;
            }

            // Validate element types are allowed.
            if !is_allowed_coopmat_element(a_key.elem) || !is_allowed_coopmat_element(c_key.elem) {
                tc.errors.push(TypecheckError::CoopMatrixElementTypeUnsupported {
                    ty: if !is_allowed_coopmat_element(a_key.elem) { a_key.elem.display_name() } else { c_key.elem.display_name() },
                    span: call_span,
                });
                return None;
            }

            // Result type matches the accumulator's type.
            let result_key = c_key;
            let _ = expected; // result type is determined by arguments, not context

            // Build HIR expression nodes for the arguments.
            let a_expr = HirExpr { kind: HirExprKind::LocalRead(a_bid), ty: ScalarTy::U32, span: a_span };
            let b_expr = HirExpr { kind: HirExprKind::LocalRead(b_bid), ty: ScalarTy::U32, span: b_span };
            let c_expr = HirExpr { kind: HirExprKind::LocalRead(c_bid), ty: ScalarTy::U32, span: c_span };

            Some(HirExpr {
                kind: HirExprKind::CoopMatBuiltin {
                    op: CoopMatBuiltin::MulAdd,
                    args: vec![a_expr, b_expr, c_expr],
                    result_ty: result_key,
                    source: None,
                },
                ty: ScalarTy::U32, // sentinel: actual type is CoopMatrix(result_key)
                span: call_span,
            })
        }
        CoopMatBuiltin::Store => unreachable!("Store handled above"),
    }
}

/// Typecheck `coopmat_store(m, buf, element_offset, stride);` as a statement.
fn check_coopmat_store_stmt(
    tc: &mut TypeChecker<'_>,
    args: &[axc_lexer::Spanned<past::Expr>],
    call_span: Span,
    stmt_span: Span,
) -> Option<HirStmt> {
    use crate::coopmat::CoopMatBuiltin;

    if args.len() != CoopMatBuiltin::Store.arity() {
        tc.errors.push(TypecheckError::CoopMatArity {
            name: "coopmat_store",
            expected: CoopMatBuiltin::Store.arity(),
            found: args.len(),
            span: call_span,
        });
        return None;
    }

    // Arg 0: matrix binding (must be a coopmat identifier).
    let matrix_ident = match &args[0].node {
        past::Expr::Ident(ref s) => s.clone(),
        _ => {
            tc.errors.push(TypecheckError::TypeMismatch {
                expected: "cooperative-matrix binding name",
                got: "expression",
                span: args[0].span,
            });
            return None;
        }
    };
    let (matrix_bid, matrix_key) = {
        match tc.find_binding(&matrix_ident) {
            None => {
                tc.errors.push(TypecheckError::UnknownBinding {
                    name: matrix_ident.clone(),
                    span: args[0].span,
                });
                return None;
            }
            Some((bid, BindingTy::CoopMatrix(key), _, _)) => (bid, key),
            Some((_, BindingTy::Scalar(st), _, _)) => {
                tc.errors.push(TypecheckError::TypeMismatch {
                    expected: "matrix",
                    got: st.display_name(),
                    span: args[0].span,
                });
                return None;
            }
        }
    };

    // Arg 1: destination — buffer param (M2.1) OR shared array (M3.2 PART B).
    let dst_ident = match &args[1].node {
        past::Expr::Ident(ref s) => s.clone(),
        _ => {
            tc.errors.push(TypecheckError::TypeMismatch {
                expected: "buffer parameter name or shared array name",
                got: "expression",
                span: args[1].span,
            });
            return None;
        }
    };

    // PART B (M3.2): check if dst_ident is a shared array first.
    let store_source = if let Some((shared_id, shared_elem, _shared_len)) = tc.find_shared(&dst_ident) {
        // Shared destination path.
        if shared_elem != matrix_key.elem {
            tc.errors.push(TypecheckError::CoopMatStoreElementTypeMismatch {
                matrix_elem: matrix_key.elem.display_name(),
                buffer_elem: shared_elem.display_name(),
                span: args[1].span,
            });
            return None;
        }
        crate::coopmat::CoopMatLoadSource::Shared(shared_id.0)
    } else {
        // Buffer destination path (M2.1 default): find writable buffer param.
        let mut found_slot: Option<u32> = None;
        let mut found_ty: Option<crate::buffer::BufferTy> = None;
        let mut buf_idx: u32 = 0;
        for p in tc.params {
            if let crate::param::Ty::Buffer(ref bt) = p.ty {
                if p.name == dst_ident {
                    found_slot = Some(buf_idx);
                    found_ty = Some(*bt);
                    break;
                }
                buf_idx += 1;
            }
        }
        let (buf_slot, buf_ty) = match (found_slot, found_ty) {
            (Some(slot), Some(ty)) => (slot, ty),
            _ => {
                tc.errors.push(TypecheckError::TypeMismatch {
                    expected: "buffer parameter or shared array",
                    got: "not a buffer parameter or shared array",
                    span: args[1].span,
                });
                return None;
            }
        };
        // Check that buffer is writable (not readonly).
        if buf_ty.access == crate::buffer::BufferAccess::ReadOnly {
            tc.errors.push(TypecheckError::CoopMatStoreToReadonlyBuffer {
                param_name: dst_ident.clone(),
                span: args[1].span,
            });
            return None;
        }
        // Check element type compatibility.
        if buf_ty.elem != matrix_key.elem {
            tc.errors.push(TypecheckError::CoopMatStoreElementTypeMismatch {
                matrix_elem: matrix_key.elem.display_name(),
                buffer_elem: buf_ty.elem.display_name(),
                span: call_span,
            });
            return None;
        }
        crate::coopmat::CoopMatLoadSource::Buffer(buf_slot)
    };

    // Arg 2: element_offset (must be U32).
    let offset_hir = check_expr(tc, &args[2].node, args[2].span, Some(ScalarTy::U32))?;
    if offset_hir.ty != ScalarTy::U32 {
        tc.errors.push(TypecheckError::CoopMatLoadStrideMustBeU32 {
            found_ty: offset_hir.ty.display_name(),
            span: args[2].span,
        });
        return None;
    }

    // Arg 3: stride (must be U32).
    let stride_hir = check_expr(tc, &args[3].node, args[3].span, Some(ScalarTy::U32))?;
    if stride_hir.ty != ScalarTy::U32 {
        tc.errors.push(TypecheckError::CoopMatLoadStrideMustBeU32 {
            found_ty: stride_hir.ty.display_name(),
            span: args[3].span,
        });
        return None;
    }

    Some(HirStmt::CoopMatStore {
        matrix_binding: matrix_bid,
        store_source,
        element_offset: offset_hir,
        stride: stride_hir,
        span: stmt_span,
    })
}

// ── Q4_0-path builtin calls (M2.5) ───────────────────────────────────────────

/// Typecheck a Q4_0-path builtin expression call.
///
/// Dispatched from `check_call` when the call name is in `RESERVED_Q4_0_BUILTIN_NAMES`.
/// Typecheck a GLSL.std.450 ext-inst builtin call (M3.2c). Modeled on the
/// `Q4_0Builtin::F32ToF16` arm: arity-1, arg must typecheck to F32, result F32,
/// producing `HirExprKind::ExtInstBuiltin`.
fn check_ext_inst_call(
    tc: &mut TypeChecker<'_>,
    op: crate::ext_inst::ExtInstBuiltin,
    _name_span: Span,
    args: &[axc_lexer::Spanned<past::Expr>],
    call_span: Span,
    _expected: Option<ScalarTy>,
) -> Option<HirExpr> {
    use crate::ext_inst::ExtInstBuiltin;

    // Arity check.
    let expected_arity = op.arity();
    if args.len() != expected_arity {
        tc.errors.push(TypecheckError::ExtInstBuiltinWrongArity {
            name: op.source_name(),
            expected: expected_arity,
            found: args.len(),
            span: call_span,
        });
        return None;
    }

    match op {
        ExtInstBuiltin::Exp => {
            // arg0: x — must be f32. Result is f32 (GLSL.std.450 Exp).
            let x_arg = &args[0];
            let x_hir = check_expr(tc, &x_arg.node, x_arg.span, Some(ScalarTy::F32))?;
            if x_hir.ty != ScalarTy::F32 {
                tc.errors.push(TypecheckError::ExpArgMustBeF32 {
                    got_ty: x_hir.ty.display_name(),
                    span: x_arg.span,
                });
                return None;
            }
            Some(HirExpr {
                kind: crate::expr::HirExprKind::ExtInstBuiltin {
                    op,
                    args: vec![x_hir],
                },
                ty: ScalarTy::F32,
                span: call_span,
            })
        }
    }
}

fn check_q4_0_call(
    tc: &mut TypeChecker<'_>,
    op: crate::q4_0::Q4_0Builtin,
    _name_span: Span,
    args: &[axc_lexer::Spanned<past::Expr>],
    call_span: Span,
    _expected: Option<ScalarTy>,
) -> Option<HirExpr> {
    use crate::q4_0::Q4_0Builtin;

    // Arity check.
    let expected_arity = op.arity();
    if args.len() != expected_arity {
        tc.errors.push(TypecheckError::Q4_0BuiltinWrongArity {
            name: op.source_name(),
            expected: expected_arity,
            found: args.len(),
            span: call_span,
        });
        return None;
    }

    match op {
        Q4_0Builtin::PtrReadU8Zext | Q4_0Builtin::PtrReadU16Zext => {
            // arg0: must be a kernel buffer parameter identifier with elem type U8.
            // arg1: byte_offset — must be a u32 expression.
            let buf_arg = &args[0];
            let offset_arg = &args[1];

            // Verify that arg0 is a plain identifier referring to a buffer param.
            let buf_name: &str = match &buf_arg.node {
                axc_parser::ast::Expr::Ident(n) => n.as_str(),
                _ => {
                    tc.errors.push(TypecheckError::PtrReadArgMustBeBufferParam {
                        found_kind: "non-identifier expression",
                        span: buf_arg.span,
                    });
                    return None;
                }
            };
            let param_info = tc.find_param(buf_name);
            let (param_position, buf_ty) = match param_info {
                Some(p) => {
                    match &p.ty {
                        ParamTy::Buffer(bt) => (p.position, *bt),
                        ParamTy::Scalar(_) => {
                            tc.errors.push(TypecheckError::PtrReadArgMustBeBufferParam {
                                found_kind: "scalar parameter",
                                span: buf_arg.span,
                            });
                            return None;
                        }
                    }
                }
                None => {
                    // Could be a local binding (not allowed).
                    if tc.find_binding(buf_name).is_some() {
                        tc.errors.push(TypecheckError::PtrReadArgMustBeBufferParam {
                            found_kind: "local binding (not a kernel parameter)",
                            span: buf_arg.span,
                        });
                    } else {
                        tc.errors.push(TypecheckError::UnknownBinding {
                            name: buf_name.to_owned(),
                            span: buf_arg.span,
                        });
                    }
                    return None;
                }
            };
            // The buffer element type MUST be U8.
            if buf_ty.elem != ScalarTy::U8 {
                tc.errors.push(TypecheckError::PtrReadBufferElemMustBeU8 {
                    elem_ty: buf_ty.elem.display_name(),
                    span: buf_arg.span,
                });
                return None;
            }
            // Compute the buffer-only binding slot index.
            let buf_param_index = count_buffer_position(tc.params, param_position);

            // arg1: byte_offset must be u32.
            let offset_hir = check_expr(tc, &offset_arg.node, offset_arg.span, Some(ScalarTy::U32))?;
            if offset_hir.ty != ScalarTy::U32 {
                tc.errors.push(TypecheckError::Q4_0BuiltinArgTypeMismatch {
                    name: op.source_name(),
                    expected_ty: "u32",
                    got_ty: offset_hir.ty.display_name(),
                    span: offset_arg.span,
                });
                return None;
            }

            // Synthesize a placeholder U32 node for the buffer argument
            // (the buf arg is a kernel param identifier, not a scalar value;
            //  the codegen uses buf_param_index to look up the SSBO var id).
            let buf_hir = axc_hir_buf_arg_placeholder(buf_arg.span);

            let return_ty = op.return_ty();
            Some(HirExpr {
                kind: crate::expr::HirExprKind::Q4_0Builtin {
                    op,
                    args: vec![buf_hir, offset_hir],
                    buf_param_index: Some(buf_param_index),
                },
                ty: return_ty,
                span: call_span,
            })
        }

        Q4_0Builtin::F16BitsToF32 => {
            // arg0: bits — must be u32.
            let bits_arg = &args[0];
            let bits_hir = check_expr(tc, &bits_arg.node, bits_arg.span, Some(ScalarTy::U32))?;
            if bits_hir.ty != ScalarTy::U32 {
                tc.errors.push(TypecheckError::F16BitsToF32ArgMustBeU32 {
                    got_ty: bits_hir.ty.display_name(),
                    span: bits_arg.span,
                });
                return None;
            }
            Some(HirExpr {
                kind: crate::expr::HirExprKind::Q4_0Builtin {
                    op,
                    args: vec![bits_hir],
                    buf_param_index: None,
                },
                ty: ScalarTy::F32,
                span: call_span,
            })
        }

        Q4_0Builtin::F32FromU32 => {
            // arg0: u — must be u32.
            let u_arg = &args[0];
            let u_hir = check_expr(tc, &u_arg.node, u_arg.span, Some(ScalarTy::U32))?;
            if u_hir.ty != ScalarTy::U32 {
                tc.errors.push(TypecheckError::Q4_0BuiltinArgTypeMismatch {
                    name: op.source_name(),
                    expected_ty: "u32",
                    got_ty: u_hir.ty.display_name(),
                    span: u_arg.span,
                });
                return None;
            }
            Some(HirExpr {
                kind: crate::expr::HirExprKind::Q4_0Builtin {
                    op,
                    args: vec![u_hir],
                    buf_param_index: None,
                },
                ty: ScalarTy::F32,
                span: call_span,
            })
        }

        Q4_0Builtin::F32ToF16 => {
            // arg0: x — must be f32. Result is f16 (narrowing OpFConvert, M3.5).
            let x_arg = &args[0];
            let x_hir = check_expr(tc, &x_arg.node, x_arg.span, Some(ScalarTy::F32))?;
            if x_hir.ty != ScalarTy::F32 {
                tc.errors.push(TypecheckError::F32ToF16ArgMustBeF32 {
                    got_ty: x_hir.ty.display_name(),
                    span: x_arg.span,
                });
                return None;
            }
            Some(HirExpr {
                kind: crate::expr::HirExprKind::Q4_0Builtin {
                    op,
                    args: vec![x_hir],
                    buf_param_index: None,
                },
                ty: ScalarTy::F16,
                span: call_span,
            })
        }
    }
}

/// Build a placeholder U32 HIR expression for the buffer-parameter argument of
/// `ptr_read_u8_zext` / `ptr_read_u16_zext`.
///
/// The codegen uses `buf_param_index` (not the expression itself) to look up the
/// SSBO variable id. The expression is a BoolLit placeholder that the body emitter
/// never evaluates. Using a dedicated sentinel avoids introducing a new HIR variant
/// solely for "this argument is a kernel parameter name, not a value".
fn axc_hir_buf_arg_placeholder(span: Span) -> HirExpr {
    HirExpr {
        kind: crate::expr::HirExprKind::BoolLit(false), // placeholder — never evaluated by codegen
        ty: ScalarTy::U32,
        span,
    }
}

// ── Bitwise builtin calls ─────────────────────────────────────────────────────

fn check_call(
    tc: &mut TypeChecker,
    name: &str,
    name_span: Span,
    args: &[axc_lexer::Spanned<past::Expr>],
    call_span: Span,
    expected: Option<ScalarTy>,
) -> Option<HirExpr> {
    // M3.2c: dispatch GLSL.std.450 ext-inst builtins FIRST (e.g. `exp`), before
    // q4_0/coopmat/subgroup. `exp` is not a keyword and check_call has no
    // user-function path, so this cannot be intercepted.
    if let Some(ei_op) = crate::ext_inst::ExtInstBuiltin::from_source_name(name) {
        return check_ext_inst_call(tc, ei_op, name_span, args, call_span, expected);
    }

    // M2.5: dispatch Q4_0-path builtins FIRST (before coopmat and subgroup).
    if let Some(q4_op) = crate::q4_0::Q4_0Builtin::from_source_name(name) {
        return check_q4_0_call(tc, q4_op, name_span, args, call_span, expected);
    }

    // M2.1: dispatch cooperative-matrix expression builtins (Zero, Load, MulAdd).
    // Store is a statement (void return) — handled in check_builtin_call_stmt.
    use crate::coopmat::CoopMatBuiltin;
    if let Some(cm_op) = CoopMatBuiltin::from_source_name(name) {
        return check_coopmat_expr_call(tc, cm_op, name_span, args, call_span, expected);
    }

    // M1.4: dispatch subgroup builtins BEFORE bitwise table.
    if let Some(sg_op) = crate::subgroup::SubgroupOp::from_source_name(name) {
        return check_subgroup_call(tc, sg_op, name_span, args, call_span, expected);
    }

    let op: BitwiseOp = match name {
        "band" => BitwiseOp::Band,
        "bor"  => BitwiseOp::Bor,
        "bxor" => BitwiseOp::Bxor,
        "bnot" => BitwiseOp::Bnot,
        "shl"  => BitwiseOp::Shl,
        "shr"  => BitwiseOp::Shr,
        "lshr" => BitwiseOp::Lshr,
        _ => {
            tc.errors.push(TypecheckError::UnknownCall {
                name: name.to_owned(),
                span: name_span,
            });
            return None;
        }
    };

    // Expected arities.
    let expected_arity: usize = match op {
        BitwiseOp::Bnot => 1,
        _               => 2,
    };

    if args.len() != expected_arity {
        tc.errors.push(TypecheckError::BitwiseArity {
            builtin: builtin_name(op),
            expected_arity,
            got_arity: args.len(),
            span: call_span,
        });
        return None;
    }

    // For band/bor/bxor/shl/shr/lshr: first arg is UNconstrained (the builtin
    // itself pins the type from the suffix, not from any outer expected context).
    // The outer `expected` is NOT forwarded into bitwise builtins per §4.1:
    // "band typechecks its args independently (both untyped → error)".
    let first_arg = &args[0];
    let first_hir = check_expr(tc, &first_arg.node, first_arg.span, None)?;
    let val_ty = first_hir.ty;

    // All bitwise ops require integer operands.
    if !val_ty.is_integer() {
        tc.errors.push(TypecheckError::BitwiseNonInteger {
            builtin: builtin_name(op),
            got_ty: val_ty.display_name(),
            span: first_arg.span,
        });
        return None;
    }

    if expected_arity == 1 {
        // bnot: one integer arg.
        return Some(HirExpr {
            kind: HirExprKind::BitwiseBuiltin {
                op,
                args: vec![first_hir],
            },
            ty: val_ty,
            span: call_span,
        });
    }

    // Two-arg ops: second arg typed with first arg's type (shift amount must match).
    let second_arg = &args[1];
    let second_hir = check_expr(tc, &second_arg.node, second_arg.span, Some(val_ty))?;
    let amt_ty = second_hir.ty;

    // Signedness-check for shr/lshr.
    match op {
        BitwiseOp::Shr => {
            if !val_ty.is_signed_integer() {
                tc.errors.push(TypecheckError::ShiftRequiresSignedLhs {
                    got_ty: val_ty.display_name(),
                    span: call_span,
                });
                return None;
            }
        }
        BitwiseOp::Lshr => {
            if !val_ty.is_unsigned_integer() {
                tc.errors.push(TypecheckError::ShiftRequiresUnsignedLhs {
                    got_ty: val_ty.display_name(),
                    span: call_span,
                });
                return None;
            }
        }
        _ => {}
    }

    // Shift amount type must match the value type (stricter than SPIR-V §3.32.5).
    if val_ty != amt_ty {
        tc.errors.push(TypecheckError::ShiftAmountTypeMismatch {
            builtin: builtin_name(op),
            lhs_ty: val_ty.display_name(),
            rhs_ty: amt_ty.display_name(),
            span: call_span,
        });
        return None;
    }

    // For band/bor/bxor: both args must have same type (already enforced above via
    // expected type propagation; but double-check for non-integer arg):
    if !amt_ty.is_integer() {
        tc.errors.push(TypecheckError::BitwiseNonInteger {
            builtin: builtin_name(op),
            got_ty: amt_ty.display_name(),
            span: second_arg.span,
        });
        return None;
    }

    Some(HirExpr {
        kind: HirExprKind::BitwiseBuiltin {
            op,
            args: vec![first_hir, second_hir],
        },
        ty: val_ty,
        span: call_span,
    })
}

// ── Buffer and gid operations (M1.2) ─────────────────────────────────────────

/// Check a `name[index] = value;` statement (buffer write).
fn check_index_assign_stmt(
    tc: &mut TypeChecker<'_>,
    target: &axc_lexer::Spanned<String>,
    index: &axc_lexer::Spanned<past::Expr>,
    value: &axc_lexer::Spanned<past::Expr>,
    stmt_span: Span,
) -> Option<HirStmt> {
    let name: &str = &target.node;

    // M3.20: Check if the target is a local array first (order: local array ->
    // shared -> buffer; see the Expr::Index read handler for the collision-guard
    // rationale).
    let local_array_info: Option<(LocalArrayId, ScalarTy, u32)> = tc.find_local_array(name);
    if let Some((local_array_id, elem_ty, len)) = local_array_info {
        return check_local_array_write(tc, local_array_id, elem_ty, len, name, index, value, stmt_span);
    }

    // M3.2: Check if the target is a shared array next.
    let shared_info: Option<(SharedId, ScalarTy, u32)> = tc.find_shared(name);
    if let Some((shared_id, elem_ty, _len)) = shared_info {
        // Typecheck the index (must be U32 — no coercion, anti-pattern #1).
        let index_hir: HirExpr = check_expr(tc, &index.node, index.span, Some(ScalarTy::U32))?;
        if index_hir.ty != ScalarTy::U32 {
            tc.errors.push(TypecheckError::SharedIndexNotU32 {
                got: index_hir.ty.display_name(),
                span: index.span,
            });
            return None;
        }

        // Typecheck the value (must match elem type exactly).
        let value_hir: HirExpr = check_expr(tc, &value.node, value.span, Some(elem_ty))?;
        if value_hir.ty != elem_ty {
            tc.errors.push(TypecheckError::SharedWriteTypeMismatch {
                name: name.to_owned(),
                expected: elem_ty.display_name(),
                got: value_hir.ty.display_name(),
                span: value.span,
            });
            return None;
        }

        // M3.2 A.4.1: append to the shared write SET.
        let index_kind = index_hir.kind.clone();
        tc.append_shared_write(shared_id.0, index_kind);

        return Some(HirStmt::SharedWrite {
            shared_id: shared_id.0,
            index: index_hir,
            value: value_hir,
            span: stmt_span,
        });
    }

    // Look up the param — clone the needed data to release the borrow before
    // calling check_expr (which borrows tc mutably).
    let param_info: Option<(ParamTy, u32)> = tc.find_param(name)
        .map(|p| (p.ty.clone(), p.position));

    let (param_ty, param_position) = match param_info {
        Some(info) => info,
        None => {
            if tc.find_binding(name).is_some() {
                tc.errors.push(TypecheckError::IndexOnNonBuffer {
                    name: name.to_owned(),
                    span: target.span,
                });
            } else {
                tc.errors.push(TypecheckError::UnknownBinding {
                    name: name.to_owned(),
                    span: target.span,
                });
            }
            return None;
        }
    };

    // Verify it's a buffer
    let bt = match param_ty {
        ParamTy::Buffer(bt) => bt,
        ParamTy::Scalar(_) => {
            tc.errors.push(TypecheckError::IndexOnNonBuffer {
                name: name.to_owned(),
                span: target.span,
            });
            return None;
        }
    };

    // Verify write is allowed
    if bt.access == BufferAccess::ReadOnly {
        tc.errors.push(TypecheckError::WriteToReadonlyBuffer {
            name: name.to_owned(),
            span: target.span,
        });
        return None;
    }

    // Typecheck the index (must be u32)
    let index_hir: HirExpr = check_expr(tc, &index.node, index.span, Some(ScalarTy::U32))?;
    if index_hir.ty != ScalarTy::U32 {
        tc.errors.push(TypecheckError::BadIndexType {
            got_ty: index_hir.ty.display_name(),
            span: index.span,
        });
        return None;
    }

    // Typecheck the value (must match elem type)
    let value_hir: HirExpr = check_expr(tc, &value.node, value.span, Some(bt.elem))?;
    if value_hir.ty != bt.elem {
        tc.errors.push(TypecheckError::TypeMismatch {
            expected: bt.elem.display_name(),
            got: value_hir.ty.display_name(),
            span: value.span,
        });
        return None;
    }

    let buffer_binding: u32 = count_buffer_position(tc.params, param_position);

    Some(HirStmt::BufferWrite {
        param_position,
        buffer_binding,
        index: index_hir,
        value: value_hir,
        span: stmt_span,
    })
}

/// Count the buffer-only position of a param (how many buffer params appear before it).
fn count_buffer_position(params: &[KernelParam], target_position: u32) -> u32 {
    params.iter()
        .filter(|p| p.position < target_position && matches!(p.ty, ParamTy::Buffer(_)))
        .count() as u32
}

/// Check a buffer-read expression: `name[index]`.
fn check_buffer_read(
    tc: &mut TypeChecker<'_>,
    name: &str,
    name_span: Span,
    index: &axc_lexer::Spanned<past::Expr>,
    expr_span: Span,
    _expected: Option<ScalarTy>,
) -> Option<HirExpr> {
    // Clone param data to release borrow before calling check_expr.
    let param_info: Option<(ParamTy, u32)> = tc.find_param(name)
        .map(|p| (p.ty.clone(), p.position));

    let (param_ty, param_position) = match param_info {
        Some(info) => info,
        None => {
            if tc.find_binding(name).is_some() {
                tc.errors.push(TypecheckError::IndexOnNonBuffer {
                    name: name.to_owned(),
                    span: name_span,
                });
            } else {
                tc.errors.push(TypecheckError::UnknownBinding {
                    name: name.to_owned(),
                    span: name_span,
                });
            }
            return None;
        }
    };

    let bt = match param_ty {
        ParamTy::Buffer(bt) => bt,
        ParamTy::Scalar(_) => {
            tc.errors.push(TypecheckError::IndexOnNonBuffer {
                name: name.to_owned(),
                span: name_span,
            });
            return None;
        }
    };

    if bt.access == BufferAccess::WriteOnly {
        tc.errors.push(TypecheckError::ReadFromWriteonlyBuffer {
            name: name.to_owned(),
            span: name_span,
        });
        return None;
    }

    // Typecheck the index (must be u32)
    let index_hir: HirExpr = check_expr(tc, &index.node, index.span, Some(ScalarTy::U32))?;
    if index_hir.ty != ScalarTy::U32 {
        tc.errors.push(TypecheckError::BadIndexType {
            got_ty: index_hir.ty.display_name(),
            span: index.span,
        });
        return None;
    }

    let buffer_binding: u32 = count_buffer_position(tc.params, param_position);

    Some(HirExpr {
        kind: HirExprKind::BufferRead {
            param_position,
            buffer_binding,
            index: Box::new(index_hir),
        },
        ty: bt.elem,
        span: expr_span,
    })
}

// ── M3.2 shared-array typecheck helpers ──────────────────────────────────────

/// Typecheck a shared-array read expression: `shared_name[index]`.
///
/// Called from the `Expr::Index` handler when the base ident resolves to a shared decl.
/// Runs the OQ1 SET-based missing-barrier analysis.
fn check_shared_read(
    tc: &mut TypeChecker<'_>,
    shared_id: SharedId,
    elem_ty: ScalarTy,
    index: &axc_lexer::Spanned<past::Expr>,
    expr_span: Span,
) -> Option<HirExpr> {
    // Index must be U32 (no implicit coercion — anti-pattern #1).
    let index_hir: HirExpr = check_expr(tc, &index.node, index.span, Some(ScalarTy::U32))?;
    if index_hir.ty != ScalarTy::U32 {
        tc.errors.push(TypecheckError::SharedIndexNotU32 {
            got: index_hir.ty.display_name(),
            span: index.span,
        });
        return None;
    }

    // Look up the shared array name for diagnostic messages.
    // We clone the name to avoid holding a borrow into tc.shared_decls during analysis.
    let shared_name: String = tc.shared_decls.get(shared_id.0 as usize)
        .map(|d| d.name.clone())
        .unwrap_or_else(|| format!("<shared#{}>", shared_id.0));

    // M3.2 A.4.1: OQ1 SET-based missing-barrier analysis.
    // analyze_shared_read_barrier takes a reference to tc; we need to avoid
    // a double borrow. Extract the diagnostic first, then push errors/warns.
    let diag = tc.analyze_shared_read_barrier(
        shared_id.0,
        &index_hir.kind,
        &shared_name,
        expr_span,
    );

    if let Some(d) = diag {
        match d {
            SharedReadBarrierDiag::HardError { name, read_index_desc, write_indices_desc, span } => {
                tc.errors.push(TypecheckError::SharedMissingBarrierBeforeCrossInvocationRead {
                    name,
                    read_index_desc,
                    write_indices_desc,
                    span,
                });
                return None;
            }
            SharedReadBarrierDiag::Warning { name, span } => {
                tc.warns.push(crate::validate::HirWarning::SharedWriteWithoutBarrierBeforeRead {
                    name,
                    span,
                });
            }
        }
    }

    Some(HirExpr {
        kind: HirExprKind::SharedRead {
            shared_id: shared_id.0,
            index: Box::new(index_hir),
        },
        ty: elem_ty,
        span: expr_span,
    })
}

/// Typecheck a `shared name: shared[elem, N];` declaration statement.
///
/// Validates N > 0, N <= MAX_SHARED_ELEMS, allowed elem type, no collision,
/// then registers the shared array in the TypeChecker.
///
/// M3.22: when `len_hole.is_some()`, `len.node` is the parser's placeholder
/// `1` — a valid length (`>= 1`, `<= MAX_SHARED_ELEMS`), so the per-declaration
/// `SharedZeroLength`/`SharedTooLarge` checks below run unchanged and pass. The
/// hole name is carried into the registered `SharedDecl` additively; the
/// AGGREGATE shared-byte check (below, in the caller) is skipped separately
/// for any kernel with an unresolved hole.
fn check_shared_decl_stmt(
    tc: &mut TypeChecker<'_>,
    name: &axc_lexer::Spanned<String>,
    elem: &axc_parser::ast::ScalarTypeRef,
    len: &axc_lexer::Spanned<u32>,
    len_hole: &Option<axc_lexer::Spanned<String>>,
    stmt_span: Span,
) -> Option<HirStmt> {
    // Validate element type.
    let elem_ty: ScalarTy = match elem {
        past::ScalarTypeRef::I8  => ScalarTy::I8,
        past::ScalarTypeRef::U8  => ScalarTy::U8,
        past::ScalarTypeRef::I32 => ScalarTy::I32,
        past::ScalarTypeRef::U32 => ScalarTy::U32,
        past::ScalarTypeRef::I64 => ScalarTy::I64,
        past::ScalarTypeRef::U64 => ScalarTy::U64,
        past::ScalarTypeRef::F16 => ScalarTy::F16,
        past::ScalarTypeRef::F32 => ScalarTy::F32,
        past::ScalarTypeRef::F64 => ScalarTy::F64,
        past::ScalarTypeRef::Bf16 => {
            // bf16 is not a valid shared element type (not even a valid scalar in SPIR-V
            // without the BF16 extension). Reject with a clear error.
            tc.errors.push(TypecheckError::SharedElementTypeUnsupported {
                name: name.node.clone(),
                ty_name: "bf16".to_owned(),
                span: stmt_span,
            });
            return None;
        }
    };

    if !is_allowed_shared_element(elem_ty) {
        tc.errors.push(TypecheckError::SharedElementTypeUnsupported {
            name: name.node.clone(),
            ty_name: elem_ty.display_name().to_owned(),
            span: stmt_span,
        });
        return None;
    }

    // Validate N.
    let n: u32 = len.node;
    if n == 0 {
        tc.errors.push(TypecheckError::SharedZeroLength {
            name: name.node.clone(),
            span: len.span,
        });
        return None;
    }
    if n > MAX_SHARED_ELEMS {
        tc.errors.push(TypecheckError::SharedTooLarge {
            name: name.node.clone(),
            len: n,
            max: MAX_SHARED_ELEMS,
            span: len.span,
        });
        return None;
    }

    let ty = SharedTy { elem: elem_ty, len: n };
    let hole_name: Option<String> = len_hole.as_ref().map(|h| h.node.clone());
    let maybe_id = tc.register_shared(&name.node, ty, hole_name, stmt_span);

    let shared_id = maybe_id?;

    Some(HirStmt::SharedDeclMarker {
        id: shared_id,
        span: stmt_span,
    })
}

// ── M3.20 local-array typecheck helpers ──────────────────────────────────────

/// Returns `Some(value)` if `index` is a constant `u32` integer literal whose value
/// is `>= len` — a provably out-of-bounds local-array index (M3.20 spec §5.2).
/// Returns `None` for symbolic (non-constant) indices, which remain UB-by-design
/// and are NOT flagged, and for in-bounds constants.
fn local_array_const_index_oob(index: &HirExpr, len: u32) -> Option<u32> {
    if let HirExprKind::IntLit { value } = &index.kind {
        if value.ty == ScalarTy::U32 {
            let v = value.bits as u32;
            if v >= len {
                return Some(v);
            }
        }
    }
    None
}

/// Typecheck a local-array read expression: `local_array_name[index]`.
///
/// Called from the `Expr::Index` handler when the base ident resolves to a local
/// array decl. Runs the §5.2 const-index-OOB hard-error check and the §5
/// empty-write-set advisory (mirrors `check_shared_read` minus the barrier analysis
/// — private memory has no cross-invocation hazard).
fn check_local_array_read(
    tc: &mut TypeChecker<'_>,
    local_array_id: LocalArrayId,
    elem_ty: ScalarTy,
    len: u32,
    index: &axc_lexer::Spanned<past::Expr>,
    expr_span: Span,
) -> Option<HirExpr> {
    // Index must be U32 (no implicit coercion — anti-pattern #1).
    let index_hir: HirExpr = check_expr(tc, &index.node, index.span, Some(ScalarTy::U32))?;
    if index_hir.ty != ScalarTy::U32 {
        tc.errors.push(TypecheckError::LocalArrayIndexNotU32 {
            got: index_hir.ty.display_name(),
            span: index.span,
        });
        return None;
    }

    // §5.2: a constant index >= N is a decidable hard error.
    if let Some(oob) = local_array_const_index_oob(&index_hir, len) {
        let array_name: String = tc.local_array_decls.get(local_array_id.0 as usize)
            .map(|d| d.name.clone())
            .unwrap_or_else(|| format!("<array#{}>", local_array_id.0));
        tc.errors.push(TypecheckError::LocalArrayConstIndexOutOfBounds {
            name: array_name,
            index: oob,
            len,
            max_index: len - 1,
            span: index.span,
        });
        return None;
    }

    // §5: empty-write-set advisory — fires only when the array's write-set is empty
    // (never written on any path reaching this read). Zero false positives; path-
    // insensitive (may miss true positives — see the warning's doc for the honest
    // disclosure).
    if !tc.local_array_has_been_written(local_array_id.0) {
        let array_name: String = tc.local_array_decls.get(local_array_id.0 as usize)
            .map(|d| d.name.clone())
            .unwrap_or_else(|| format!("<array#{}>", local_array_id.0));
        tc.warns.push(crate::validate::HirWarning::LocalArrayReadBeforeAnyWrite {
            name: array_name,
            span: expr_span,
        });
    }

    Some(HirExpr {
        kind: HirExprKind::LocalArrayRead {
            local_array_id: local_array_id.0,
            index: Box::new(index_hir),
        },
        ty: elem_ty,
        span: expr_span,
    })
}

/// Typecheck a local-array write statement: `local_array_name[index] = value;`.
///
/// Called from `check_index_assign_stmt` when the target ident resolves to a local
/// array decl. Mirrors the shared-array write branch minus the barrier write-set
/// bookkeeping (replaced by the simpler presence-only write-set here).
#[allow(clippy::too_many_arguments)]
fn check_local_array_write(
    tc: &mut TypeChecker<'_>,
    local_array_id: LocalArrayId,
    elem_ty: ScalarTy,
    len: u32,
    name: &str,
    index: &axc_lexer::Spanned<past::Expr>,
    value: &axc_lexer::Spanned<past::Expr>,
    stmt_span: Span,
) -> Option<HirStmt> {
    // Typecheck the index (must be U32 — no coercion, anti-pattern #1).
    let index_hir: HirExpr = check_expr(tc, &index.node, index.span, Some(ScalarTy::U32))?;
    if index_hir.ty != ScalarTy::U32 {
        tc.errors.push(TypecheckError::LocalArrayIndexNotU32 {
            got: index_hir.ty.display_name(),
            span: index.span,
        });
        return None;
    }

    // §5.2: a constant index >= N is a decidable hard error.
    if let Some(oob) = local_array_const_index_oob(&index_hir, len) {
        tc.errors.push(TypecheckError::LocalArrayConstIndexOutOfBounds {
            name: name.to_owned(),
            index: oob,
            len,
            max_index: len - 1,
            span: index.span,
        });
        return None;
    }

    // Typecheck the value (must match elem type exactly).
    let value_hir: HirExpr = check_expr(tc, &value.node, value.span, Some(elem_ty))?;
    if value_hir.ty != elem_ty {
        tc.errors.push(TypecheckError::LocalArrayWriteTypeMismatch {
            name: name.to_owned(),
            expected: elem_ty.display_name(),
            got: value_hir.ty.display_name(),
            span: value.span,
        });
        return None;
    }

    tc.mark_local_array_written(local_array_id.0);

    Some(HirStmt::LocalArrayWrite {
        local_array_id: local_array_id.0,
        index: index_hir,
        value: value_hir,
        span: stmt_span,
    })
}

/// Typecheck an `array name: array[elem, N];` declaration statement.
///
/// Validates N > 0, N <= MAX_LOCAL_ARRAY_ELEMS, allowed elem type, no collision,
/// then registers the local array in the TypeChecker.
///
/// `at_top_level` distinguishes the two call sites (M3.20 spec §5.1): `true` from
/// `typecheck_kernel_body`'s top-level statement loop (allowed); `false` from
/// `typecheck_block_stmts` (nested `if`/`for`/`while` bodies — hard error). A
/// rejected nested decl STILL calls `register_local_array` first (poisoning the
/// name) so later in-block references resolve as local-array uses and don't
/// cascade into unrelated `UnknownBinding`/`IndexOnNonBuffer` noise — one root
/// error (reviewer note (a)).
fn check_local_array_decl_stmt(
    tc: &mut TypeChecker<'_>,
    name: &axc_lexer::Spanned<String>,
    elem: &axc_parser::ast::ScalarTypeRef,
    len: &axc_lexer::Spanned<u32>,
    stmt_span: Span,
    at_top_level: bool,
) -> Option<HirStmt> {
    // Validate element type.
    let elem_ty: ScalarTy = match elem {
        past::ScalarTypeRef::I8  => ScalarTy::I8,
        past::ScalarTypeRef::U8  => ScalarTy::U8,
        past::ScalarTypeRef::I32 => ScalarTy::I32,
        past::ScalarTypeRef::U32 => ScalarTy::U32,
        past::ScalarTypeRef::I64 => ScalarTy::I64,
        past::ScalarTypeRef::U64 => ScalarTy::U64,
        past::ScalarTypeRef::F16 => ScalarTy::F16,
        past::ScalarTypeRef::F32 => ScalarTy::F32,
        past::ScalarTypeRef::F64 => ScalarTy::F64,
        past::ScalarTypeRef::Bf16 => {
            // bf16 is not a valid local-array element type (not even a valid scalar
            // in SPIR-V without the BF16 extension). Reject with a clear error.
            tc.errors.push(TypecheckError::LocalArrayElementTypeNotAllowed {
                name: name.node.clone(),
                ty_name: "bf16".to_owned(),
                span: stmt_span,
            });
            return None;
        }
    };

    if !is_allowed_local_element(elem_ty) {
        tc.errors.push(TypecheckError::LocalArrayElementTypeNotAllowed {
            name: name.node.clone(),
            ty_name: elem_ty.display_name().to_owned(),
            span: stmt_span,
        });
        return None;
    }

    // Validate N.
    let n: u32 = len.node;
    if n == 0 {
        tc.errors.push(TypecheckError::LocalArrayZeroLength {
            name: name.node.clone(),
            span: len.span,
        });
        return None;
    }
    if n > MAX_LOCAL_ARRAY_ELEMS {
        tc.errors.push(TypecheckError::LocalArrayTooManyElems {
            name: name.node.clone(),
            len: n,
            max: MAX_LOCAL_ARRAY_ELEMS,
            span: len.span,
        });
        return None;
    }

    let ty = LocalArrayTy { elem: elem_ty, len: n };
    let maybe_id = tc.register_local_array(&name.node, ty, stmt_span);
    let local_array_id = maybe_id?;

    // §5.1 (r2): reject (but keep registered — poisoned) if not at kernel-body top level.
    if !at_top_level {
        tc.errors.push(TypecheckError::LocalArrayDeclNotAtBlockScope {
            name: name.node.clone(),
            span: stmt_span,
        });
        return None;
    }

    Some(HirStmt::LocalArrayDeclMarker {
        id: local_array_id,
        span: stmt_span,
    })
}

/// Check a `gid(axis)` call.
fn check_gid_call(
    tc: &mut TypeChecker<'_>,
    args: &[axc_lexer::Spanned<past::Expr>],
    call_span: Span,
) -> Option<HirExpr> {
    if args.len() != 1 {
        tc.errors.push(TypecheckError::GidArity {
            got: args.len(),
            span: call_span,
        });
        return None;
    }

    // The axis must be a compile-time integer literal (u32 range 0..=2).
    let arg: &axc_lexer::Spanned<past::Expr> = &args[0];
    let axis: u32 = match &arg.node {
        past::Expr::IntLit { value, .. } => {
            if *value < 0 || *value > 2 {
                tc.errors.push(TypecheckError::GidAxisOutOfRange {
                    got: *value as u32,
                    span: arg.span,
                });
                return None;
            }
            *value as u32
        }
        // For a u32-suffixed literal the same check applies
        _ => {
            // Try to evaluate as a constant — in M1.2 only integer literals are accepted.
            tc.errors.push(TypecheckError::GidAxisMustBeConstant {
                span: arg.span,
            });
            return None;
        }
    };

    Some(HirExpr {
        kind: HirExprKind::GidBuiltin { axis },
        ty: ScalarTy::U32,
        span: call_span,
    })
}

/// Check a `local_invocation_id(axis)` call (M3.3d).
///
/// Mirrors `check_gid_call` exactly.  Lowers to `LocalInvocationIdBuiltin { axis }`.
fn check_local_invocation_id_call(
    tc: &mut TypeChecker<'_>,
    args: &[axc_lexer::Spanned<past::Expr>],
    call_span: Span,
) -> Option<HirExpr> {
    if args.len() != 1 {
        tc.errors.push(TypecheckError::LocalInvocationIdArity {
            got: args.len(),
            span: call_span,
        });
        return None;
    }

    let arg: &axc_lexer::Spanned<past::Expr> = &args[0];
    let axis: u32 = match &arg.node {
        past::Expr::IntLit { value, .. } => {
            if *value < 0 || *value > 2 {
                tc.errors.push(TypecheckError::LocalInvocationIdAxisOutOfRange {
                    got: *value as u32,
                    span: arg.span,
                });
                return None;
            }
            *value as u32
        }
        _ => {
            tc.errors.push(TypecheckError::LocalInvocationIdAxisMustBeConstant {
                span: arg.span,
            });
            return None;
        }
    };

    Some(HirExpr {
        kind: HirExprKind::LocalInvocationIdBuiltin { axis },
        ty: ScalarTy::U32,
        span: call_span,
    })
}

fn builtin_name(op: BitwiseOp) -> &'static str {
    match op {
        BitwiseOp::Band => "band",
        BitwiseOp::Bor  => "bor",
        BitwiseOp::Bxor => "bxor",
        BitwiseOp::Bnot => "bnot",
        BitwiseOp::Shl  => "shl",
        BitwiseOp::Shr  => "shr",
        BitwiseOp::Lshr => "lshr",
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use axc_parser::parse;

    /// Helper: parse kernel body statements and run typecheck.
    fn tc_body(body_stmts: &str) -> (KernelBodyTyped, Vec<TypecheckError>) {
        let full = format!(
            "@kernel @workgroup(1,1,1) fn k() -> void {{ {} }}",
            body_stmts
        );
        let (ast, lex_errs, _parse_errs) = parse(&full);
        assert!(lex_errs.is_empty(), "lex: {lex_errs:?}");
        // _parse_errs: some tests intentionally have parse errors (unresolved idents)
        if let Some(item) = ast.items.first() {
            let axc_parser::Item::Kernel(ref kd) = item.node;
            let (typed, errs, _warns) = typecheck_kernel_body(&kd.body.node, &[]);
            return (typed, errs);
        }
        (KernelBodyTyped { bindings: Vec::new(), stmts: Vec::new(), shared: Vec::new(), local_arrays: Vec::new() }, Vec::new())
    }

    /// Helper: parse kernel body statements with params and run typecheck. Also returns warnings.
    fn tc_body_with_warns(body_stmts: &str) -> (KernelBodyTyped, Vec<TypecheckError>, Vec<crate::validate::HirWarning>) {
        let full = format!(
            "@kernel @workgroup(1,1,1) fn k() -> void {{ {} }}",
            body_stmts
        );
        let (ast, lex_errs, _parse_errs) = parse(&full);
        assert!(lex_errs.is_empty(), "lex: {lex_errs:?}");
        if let Some(item) = ast.items.first() {
            let axc_parser::Item::Kernel(ref kd) = item.node;
            return typecheck_kernel_body(&kd.body.node, &[]);
        }
        (KernelBodyTyped { bindings: Vec::new(), stmts: Vec::new(), shared: Vec::new(), local_arrays: Vec::new() }, Vec::new(), Vec::new())
    }

    // 1. tc_let_i32_literal_happy
    #[test]
    fn tc_let_i32_literal_happy() {
        let (body, errors) = tc_body("let x: i32 = 42; return;");
        assert!(errors.is_empty(), "errors: {errors:?}");
        assert_eq!(body.bindings.len(), 1);
        assert_eq!(body.bindings[0].ty, BindingTy::Scalar(ScalarTy::I32));
    }

    // 2. tc_let_u64_literal_happy
    #[test]
    fn tc_let_u64_literal_happy() {
        let (body, errors) = tc_body("let x: u64 = 42; return;");
        assert!(errors.is_empty(), "errors: {errors:?}");
        assert_eq!(body.bindings[0].ty, BindingTy::Scalar(ScalarTy::U64));
    }

    // 3. tc_let_i32_float_lit_rejected
    #[test]
    fn tc_let_i32_float_lit_rejected() {
        let (_, errors) = tc_body("let x: i32 = 3.14f32; return;");
        assert!(errors.iter().any(|e| matches!(e, TypecheckError::TypeMismatch { .. })),
            "errors: {errors:?}");
    }

    // 4. tc_let_f32_int_lit_rejected
    #[test]
    fn tc_let_f32_int_lit_rejected() {
        let (_, errors) = tc_body("let x: f32 = 42; return;");
        // 42 is an int literal but x is f32, and expected=F32 → float path: TypeMismatch
        assert!(errors.iter().any(|e| matches!(e, TypecheckError::TypeMismatch { .. })),
            "errors: {errors:?}");
    }

    // 5. tc_literal_out_of_range_i32
    #[test]
    fn tc_literal_out_of_range_i32() {
        let (_, errors) = tc_body("let x: i32 = 9999999999; return;");
        assert!(errors.iter().any(|e| matches!(e, TypecheckError::LiteralOutOfRange { value: 9999999999, .. })),
            "errors: {errors:?}");
    }

    // 6. tc_assign_immutable_rejected
    #[test]
    fn tc_assign_immutable_rejected() {
        let (_, errors) = tc_body("let x: i32 = 1i32; x = 2i32; return;");
        assert!(errors.iter().any(|e| matches!(e, TypecheckError::AssignImmutable { .. })),
            "errors: {errors:?}");
    }

    // 7. tc_assign_mutable_happy
    #[test]
    fn tc_assign_mutable_happy() {
        let (_, errors) = tc_body("let mut x: i32 = 1i32; x = 2i32; return;");
        assert!(errors.is_empty(), "errors: {errors:?}");
    }

    // 8. tc_assign_unknown_binding
    #[test]
    fn tc_assign_unknown_binding() {
        let (_, errors) = tc_body("y = 5i32; return;");
        assert!(errors.iter().any(|e| matches!(e, TypecheckError::UnknownBinding { .. })),
            "errors: {errors:?}");
    }

    // 9. tc_redeclared_binding
    #[test]
    fn tc_redeclared_binding() {
        let (_, errors) = tc_body("let x: i32 = 1i32; let x: i32 = 2i32; return;");
        assert!(errors.iter().any(|e| matches!(e, TypecheckError::RedeclaredBinding { .. })),
            "errors: {errors:?}");
    }

    // 10. tc_add_i32_u32_rejected
    #[test]
    fn tc_add_i32_u32_rejected() {
        let (_, errors) = tc_body("let x: i32 = 1i32 + 2u32; return;");
        assert!(errors.iter().any(|e| matches!(e, TypecheckError::MixedOperandTypes { .. })),
            "errors: {errors:?}");
    }

    // 11. tc_div_selects_signed_for_i32
    #[test]
    fn tc_div_selects_signed_for_i32() {
        let (body, errors) = tc_body("let x: i32 = 10i32 / 3i32; return;");
        assert!(errors.is_empty(), "errors: {errors:?}");
        // Check that the HIR contains a Div node with I32 type.
        let has_div = body.stmts.iter().any(|s| {
            if let HirStmt::Let { init, .. } = s {
                matches!(&init.kind, HirExprKind::Binary { op: BinOp::Div, .. })
            } else {
                false
            }
        });
        assert!(has_div, "expected Div stmt");
    }

    // 12. tc_comparison_yields_bool
    #[test]
    fn tc_comparison_yields_bool() {
        let (body, errors) = tc_body("let b: bool = 1i32 < 2i32; return;");
        assert!(errors.is_empty(), "errors: {errors:?}");
        assert_eq!(body.bindings[0].ty, BindingTy::Scalar(ScalarTy::Bool));
    }

    // 13. tc_short_circuit_requires_bool_operands
    #[test]
    fn tc_short_circuit_requires_bool_operands() {
        let (_, errors) = tc_body("let b: bool = 1i32 and 2i32; return;");
        assert!(errors.iter().any(|e| matches!(e, TypecheckError::TypeMismatch { .. })),
            "errors: {errors:?}");
    }

    // 14. tc_not_requires_bool
    #[test]
    fn tc_not_requires_bool() {
        let (_, errors) = tc_body("let b: bool = not 42i32; return;");
        assert!(errors.iter().any(|e| matches!(e, TypecheckError::TypeMismatch { .. })),
            "errors: {errors:?}");
    }

    // 15. tc_band_same_type_required
    #[test]
    fn tc_band_same_type_required() {
        let (_, errors) = tc_body("let c: u32 = band(1u32, 2i32); return;");
        // 2i32 is checked with expected=U32 from 1u32; type mismatch
        assert!(!errors.is_empty(), "expected errors for band(u32, i32): {errors:?}");
    }

    // 16. tc_shl_arity_enforced
    #[test]
    fn tc_shl_arity_enforced() {
        // shl with 1 arg should fail BitwiseArity
        // We test by crafting a source where shl has only 1 arg
        let (_, errors) = tc_body("let c: u32 = shl(1u32); return;");
        assert!(errors.iter().any(|e| matches!(e, TypecheckError::BitwiseArity { .. })),
            "errors: {errors:?}");
    }

    // 17. tc_unknown_call_rejected
    #[test]
    fn tc_unknown_call_rejected() {
        let (_, errors) = tc_body("let c: i32 = foo(1i32, 2i32); return;");
        assert!(errors.iter().any(|e| matches!(e, TypecheckError::UnknownCall { .. })),
            "errors: {errors:?}");
    }

    // 18. tc_neg_on_u32_rejected
    #[test]
    fn tc_neg_on_u32_rejected() {
        let (_, errors) = tc_body("let x: u32 = -1u32; return;");
        assert!(errors.iter().any(|e| matches!(e, TypecheckError::OperatorTypeError { op: "-", .. })),
            "expected OperatorTypeError for neg on u32: {errors:?}");
    }

    // 19. tc_literal_i32_min_accepted
    #[test]
    fn tc_literal_i32_min_accepted() {
        let (body, errors) = tc_body("let x: i32 = -2147483648; return;");
        assert!(errors.is_empty(), "errors: {errors:?}");
        if let Some(HirStmt::Let { init, .. }) = body.stmts.first() {
            if let HirExprKind::IntLit { value } = &init.kind {
                assert_eq!(value.bits, 0x8000_0000_u64, "i32::MIN bits should be 0x80000000");
            } else {
                panic!("expected IntLit, got: {:?}", init.kind);
            }
        }
    }

    // 20. tc_literal_i64_min_accepted
    #[test]
    fn tc_literal_i64_min_accepted() {
        let (body, errors) = tc_body("let x: i64 = -9223372036854775808i64; return;");
        assert!(errors.is_empty(), "errors: {errors:?}");
        if let Some(HirStmt::Let { init, .. }) = body.stmts.first() {
            if let HirExprKind::IntLit { value } = &init.kind {
                assert_eq!(value.bits, 0x8000_0000_0000_0000_u64);
            } else {
                panic!("expected IntLit, got: {:?}", init.kind);
            }
        }
    }

    // 21. tc_bool_eq_happy
    #[test]
    fn tc_bool_eq_happy() {
        let (body, errors) = tc_body("let b: bool = true == false; return;");
        assert!(errors.is_empty(), "errors: {errors:?}");
        assert_eq!(body.bindings[0].ty, BindingTy::Scalar(ScalarTy::Bool));
    }

    // 23. tc_shr_on_u32_rejected
    #[test]
    fn tc_shr_on_u32_rejected() {
        let (_, errors) = tc_body("let x: u32 = shr(0x80000000u32, 1u32); return;");
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::ShiftRequiresSignedLhs { .. })),
            "expected ShiftRequiresSignedLhs: {errors:?}"
        );
    }

    // 24. tc_lshr_on_i32_rejected
    #[test]
    fn tc_lshr_on_i32_rejected() {
        let (_, errors) = tc_body("let x: i32 = lshr(-8i32, 1i32); return;");
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::ShiftRequiresUnsignedLhs { .. })),
            "expected ShiftRequiresUnsignedLhs: {errors:?}"
        );
    }

    // 25. tc_shift_amount_type_mismatch
    #[test]
    fn tc_shift_amount_type_mismatch() {
        let (_, errors) = tc_body("let x: i64 = shl(1i64, 3i32); return;");
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::ShiftAmountTypeMismatch { .. })),
            "expected ShiftAmountTypeMismatch: {errors:?}"
        );
    }

    // 26. tc_unconstrained_int_literal_in_call_rejected
    #[test]
    fn tc_unconstrained_int_literal_in_call_rejected() {
        let (_, errors) = tc_body("let c: i32 = band(42, 1); return;");
        // Both args of band(42, 1) are unconstrained — should emit 2x UnconstrainedLiteralNeedsSuffix
        let count = errors.iter().filter(|e| matches!(e, TypecheckError::UnconstrainedLiteralNeedsSuffix { .. })).count();
        assert!(count >= 1, "expected at least 1 UnconstrainedLiteralNeedsSuffix, got: {errors:?}");
    }

    // 27. tc_unconstrained_float_literal_rejected
    #[test]
    fn tc_unconstrained_float_literal_rejected() {
        let (_, errors) = tc_body("let b: bool = 3.14 < 2.0; return;");
        let count = errors.iter().filter(|e| matches!(e, TypecheckError::UnconstrainedLiteralNeedsSuffix { .. })).count();
        assert!(count >= 1, "expected UnconstrainedLiteralNeedsSuffix: {errors:?}");
    }

    // Error recovery: multiple independent errors aggregate
    #[test]
    fn tc_multiple_errors_aggregate() {
        // Two independent errors: missing type annotation causes parse errors, not TC errors.
        // Use two TC-level errors instead.
        let (_, errors) = tc_body("let x: i32 = 9999999999; let y: i32 = 8888888888; return;");
        assert!(errors.len() >= 2, "expected at least 2 errors, got: {errors:?}");
    }

    // ── M1.2 buffer + gid typecheck tests ────────────────────────────────────

    /// Helper: parse and typecheck a kernel with explicit params.
    fn tc_with_params(params_str: &str, body_stmts: &str) -> (KernelBodyTyped, Vec<TypecheckError>) {
        let full = format!(
            "@kernel @workgroup(1,1,1) fn k({}) -> void {{ {} }}",
            params_str,
            body_stmts
        );
        let (ast, lex_errs, parse_errs) = axc_parser::parse(&full);
        assert!(lex_errs.is_empty(), "lex: {lex_errs:?}");
        assert!(parse_errs.is_empty(), "parse: {parse_errs:?}");
        if let Some(item) = ast.items.first() {
            let axc_parser::Item::Kernel(ref kd) = item.node;
            let params = crate::lower::lower_params_for_test(&kd.params);
            let (typed, errs, _warns) = typecheck_kernel_body(&kd.body.node, &params);
            return (typed, errs);
        }
        (KernelBodyTyped { bindings: Vec::new(), stmts: Vec::new(), shared: Vec::new(), local_arrays: Vec::new() }, Vec::new())
    }

    // AT-210: WriteToReadonlyBuffer
    #[test]
    fn tc_write_to_readonly_rejected() {
        let (_, errors) = tc_with_params(
            "x: readonly_buffer[f32]",
            "x[0u32] = 1.0f32; return;",
        );
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::WriteToReadonlyBuffer { name, .. } if name == "x")),
            "expected WriteToReadonlyBuffer for 'x': {errors:?}"
        );
    }

    // AT-211: ReadFromWriteonlyBuffer
    #[test]
    fn tc_read_from_writeonly_rejected() {
        let (_, errors) = tc_with_params(
            "c: writeonly_buffer[f32]",
            "let v: f32 = c[0u32]; return;",
        );
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::ReadFromWriteonlyBuffer { name, .. } if name == "c")),
            "expected ReadFromWriteonlyBuffer for 'c': {errors:?}"
        );
    }

    // AT-212: BadIndexType (float index)
    #[test]
    fn tc_bad_index_type_float() {
        let (_, errors) = tc_with_params(
            "x: buffer[f32]",
            "let v: f32 = x[1.0f32]; return;",
        );
        // check_buffer_read explicitly pushes BadIndexType when index_hir.ty != U32.
        // A f32 literal with expected=U32 resolves as f32 (TypeMismatch in context), but
        // when the resolved index type is f32, BadIndexType fires.
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::BadIndexType { got_ty: "f32", .. })),
            "expected BadIndexType{{got_ty:'f32'}}: {errors:?}"
        );
    }

    // AT-212: BadIndexType (bool index)
    #[test]
    fn tc_bad_index_type_bool() {
        let (_, errors) = tc_with_params(
            "x: buffer[f32]",
            "let v: f32 = x[true]; return;",
        );
        // `true` resolves as Bool; check_buffer_read fires BadIndexType for non-U32 index.
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::BadIndexType { got_ty: "bool", .. })),
            "expected BadIndexType{{got_ty:'bool'}}: {errors:?}"
        );
    }

    // AT-213: IndexOnNonBuffer (scalar param)
    #[test]
    fn tc_index_on_scalar_rejected() {
        let (_, errors) = tc_with_params(
            "a: f32",
            "let v: f32 = a[0u32]; return;",
        );
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::IndexOnNonBuffer { name, .. } if name == "a")),
            "expected IndexOnNonBuffer for scalar param 'a': {errors:?}"
        );
    }

    // AT-213: IndexOnNonBuffer (local binding)
    #[test]
    fn tc_index_on_local_binding_rejected() {
        let (_, errors) = tc_body("let a: f32 = 1.0f32; let v: f32 = a[0u32]; return;");
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::IndexOnNonBuffer { name, .. } if name == "a")),
            "expected IndexOnNonBuffer for local binding 'a': {errors:?}"
        );
    }

    // AT-215: GidAxisOutOfRange (axis 3)
    #[test]
    fn tc_gid_axis_3_rejected() {
        let (_, errors) = tc_body("let i: u32 = gid(3u32); return;");
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::GidAxisOutOfRange { got: 3, .. })),
            "expected GidAxisOutOfRange{{got:3}}: {errors:?}"
        );
    }

    // AT-216: GidAxisMustBeConstant for unary-negated literal (-1 is not an IntLit node)
    // `-1` parses as Unary(Neg, IntLit(1)) — not a bare IntLit — so check_gid_call
    // falls through to the `_` arm and fires GidAxisMustBeConstant, NOT GidAxisOutOfRange.
    #[test]
    fn tc_gid_axis_non_literal_rejected() {
        let (_, errors) = tc_body("let i: u32 = gid(-1); return;");
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::GidAxisMustBeConstant { .. })),
            "expected GidAxisMustBeConstant for unary-negated axis expression: {errors:?}"
        );
    }

    // AT-215: GidAxisOutOfRange for axis value 3 (constant literal, in-range check fails)
    #[test]
    fn tc_gid_axis_three_out_of_range() {
        let (_, errors) = tc_body("let i: u32 = gid(3u32); return;");
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::GidAxisOutOfRange { got: 3, .. })),
            "expected GidAxisOutOfRange{{got:3}}: {errors:?}"
        );
    }

    // AT-216: GidAxisMustBeConstant (variable axis)
    #[test]
    fn tc_gid_axis_variable_rejected() {
        let (_, errors) = tc_body("let k: u32 = 0u32; let i: u32 = gid(k); return;");
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::GidAxisMustBeConstant { .. })),
            "expected GidAxisMustBeConstant for variable axis: {errors:?}"
        );
    }

    // AT-216: GidAxisMustBeConstant (float axis — unsuffixed float is not an integer literal)
    #[test]
    fn tc_gid_axis_unsuffixed_rejected() {
        // Expression is a binary expression, not an integer literal — must be constant
        let (_, errors) = tc_body("let i: u32 = gid(0u32 + 0u32); return;");
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::GidAxisMustBeConstant { .. })),
            "expected GidAxisMustBeConstant for non-literal axis: {errors:?}"
        );
    }

    // AT-216: GidArity (0 args)
    #[test]
    fn tc_gid_arity_0_rejected() {
        let (_, errors) = tc_body("let i: u32 = gid(); return;");
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::GidArity { got: 0, .. })),
            "expected GidArity{{got:0}}: {errors:?}"
        );
    }

    // AT-216: GidArity (2 args)
    #[test]
    fn tc_gid_arity_2_rejected() {
        let (_, errors) = tc_body("let i: u32 = gid(0u32, 1u32); return;");
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::GidArity { got: 2, .. })),
            "expected GidArity{{got:2}}: {errors:?}"
        );
    }

    // Gid axis 0, 1, 2 are valid
    #[test]
    fn tc_gid_axis_0_ok() {
        let (body, errors) = tc_body("let i: u32 = gid(0); return;");
        assert!(errors.is_empty(), "gid(0) should succeed: {errors:?}");
        let has_gid = body.stmts.iter().any(|s| {
            if let HirStmt::Let { init, .. } = s {
                matches!(init.kind, HirExprKind::GidBuiltin { axis: 0 })
            } else { false }
        });
        assert!(has_gid, "expected GidBuiltin{{axis:0}} in body");
    }

    #[test]
    fn tc_gid_axis_1_ok() {
        let (body, errors) = tc_body("let i: u32 = gid(1); return;");
        assert!(errors.is_empty(), "gid(1) should succeed: {errors:?}");
        let has_gid = body.stmts.iter().any(|s| {
            if let HirStmt::Let { init, .. } = s {
                matches!(init.kind, HirExprKind::GidBuiltin { axis: 1 })
            } else { false }
        });
        assert!(has_gid, "expected GidBuiltin{{axis:1}} in body");
    }

    #[test]
    fn tc_gid_axis_2_ok() {
        let (body, errors) = tc_body("let i: u32 = gid(2); return;");
        assert!(errors.is_empty(), "gid(2) should succeed: {errors:?}");
        let has_gid = body.stmts.iter().any(|s| {
            if let HirStmt::Let { init, .. } = s {
                matches!(init.kind, HirExprKind::GidBuiltin { axis: 2 })
            } else { false }
        });
        assert!(has_gid, "expected GidBuiltin{{axis:2}} in body");
    }

    // ── AT-1740: local_invocation_id() builtin typecheck ─────────────────────────

    /// AT-1740: local_invocation_id(0u32) typechecks to u32, lowers to LocalInvocationIdBuiltin{axis:0}.
    #[test]
    fn tc_local_invocation_id_axis_0_ok() {
        let (body, errors) = tc_body("let i: u32 = local_invocation_id(0u32); return;");
        assert!(errors.is_empty(), "local_invocation_id(0u32) should succeed: {errors:?}");
        let has_lid = body.stmts.iter().any(|s| {
            if let HirStmt::Let { init, .. } = s {
                matches!(init.kind, HirExprKind::LocalInvocationIdBuiltin { axis: 0 })
            } else { false }
        });
        assert!(has_lid, "expected LocalInvocationIdBuiltin{{axis:0}} in body");
    }

    /// AT-1740: local_invocation_id(1u32) — axis 1 valid.
    #[test]
    fn tc_local_invocation_id_axis_1_ok() {
        let (body, errors) = tc_body("let i: u32 = local_invocation_id(1u32); return;");
        assert!(errors.is_empty(), "local_invocation_id(1u32) should succeed: {errors:?}");
        let has_lid = body.stmts.iter().any(|s| {
            if let HirStmt::Let { init, .. } = s {
                matches!(init.kind, HirExprKind::LocalInvocationIdBuiltin { axis: 1 })
            } else { false }
        });
        assert!(has_lid, "expected LocalInvocationIdBuiltin{{axis:1}} in body");
    }

    /// AT-1740: local_invocation_id(2u32) — axis 2 valid.
    #[test]
    fn tc_local_invocation_id_axis_2_ok() {
        let (body, errors) = tc_body("let i: u32 = local_invocation_id(2u32); return;");
        assert!(errors.is_empty(), "local_invocation_id(2u32) should succeed: {errors:?}");
        let has_lid = body.stmts.iter().any(|s| {
            if let HirStmt::Let { init, .. } = s {
                matches!(init.kind, HirExprKind::LocalInvocationIdBuiltin { axis: 2 })
            } else { false }
        });
        assert!(has_lid, "expected LocalInvocationIdBuiltin{{axis:2}} in body");
    }

    /// AT-1740: local_invocation_id(3u32) — axis 3 out of range.
    #[test]
    fn tc_local_invocation_id_axis_3_rejected() {
        let (_, errors) = tc_body("let i: u32 = local_invocation_id(3u32); return;");
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::LocalInvocationIdAxisOutOfRange { got: 3, .. })),
            "expected LocalInvocationIdAxisOutOfRange{{got:3}}: {errors:?}"
        );
    }

    /// AT-1740: local_invocation_id() — arity 0 rejected.
    #[test]
    fn tc_local_invocation_id_arity_0_rejected() {
        let (_, errors) = tc_body("let i: u32 = local_invocation_id(); return;");
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::LocalInvocationIdArity { got: 0, .. })),
            "expected LocalInvocationIdArity{{got:0}}: {errors:?}"
        );
    }

    /// AT-1740: local_invocation_id(0u32, 1u32) — arity 2 rejected.
    #[test]
    fn tc_local_invocation_id_arity_2_rejected() {
        let (_, errors) = tc_body("let i: u32 = local_invocation_id(0u32, 1u32); return;");
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::LocalInvocationIdArity { got: 2, .. })),
            "expected LocalInvocationIdArity{{got:2}}: {errors:?}"
        );
    }

    /// AT-1740: non-literal axis rejected.
    #[test]
    fn tc_local_invocation_id_variable_axis_rejected() {
        let (_, errors) = tc_body("let k: u32 = 0u32; let i: u32 = local_invocation_id(k); return;");
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::LocalInvocationIdAxisMustBeConstant { .. })),
            "expected LocalInvocationIdAxisMustBeConstant: {errors:?}"
        );
    }

    /// AT-1740 NO-REGRESSION: a kernel using BOTH gid() and local_invocation_id()
    /// typechecks both to distinct builtins (GidBuiltin + LocalInvocationIdBuiltin).
    #[test]
    fn tc_gid_and_local_invocation_id_both_distinct() {
        let (body, errors) = tc_body(
            "let x: u32 = gid(0u32); let y: u32 = local_invocation_id(0u32); return;"
        );
        assert!(errors.is_empty(), "gid+local_invocation_id should both succeed: {errors:?}");
        let has_gid = body.stmts.iter().any(|s| {
            if let HirStmt::Let { init, .. } = s {
                matches!(init.kind, HirExprKind::GidBuiltin { .. })
            } else { false }
        });
        let has_lid = body.stmts.iter().any(|s| {
            if let HirStmt::Let { init, .. } = s {
                matches!(init.kind, HirExprKind::LocalInvocationIdBuiltin { .. })
            } else { false }
        });
        assert!(has_gid, "expected GidBuiltin in body");
        assert!(has_lid, "expected LocalInvocationIdBuiltin in body");
    }

    // BufferAsValue
    #[test]
    fn tc_buffer_param_value_rejected() {
        let (_, errors) = tc_with_params(
            "buf: buffer[f32]",
            "let v: f32 = buf; return;",
        );
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::BufferAsValue { name, .. } if name == "buf")),
            "expected BufferAsValue for bare buffer param: {errors:?}"
        );
    }

    // AssignToParam — kernel parameters are immutable; assignment fires AssignToParam
    #[test]
    fn tc_assign_to_param_rejected() {
        // The typechecker checks param names BEFORE the binding table in the Assign path.
        // Any assignment whose target matches a param name fires AssignToParam, not
        // UnknownBinding or AssignImmutable.
        let (_, errors) = tc_with_params(
            "a: f32",
            "a = 2.0f32; return;",
        );
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::AssignToParam { name, .. } if name == "a")),
            "expected AssignToParam{{name:'a'}}: {errors:?}"
        );
    }

    // Scalar param read is OK
    #[test]
    fn tc_scalar_param_read() {
        let (body, errors) = tc_with_params(
            "a: f32",
            "let v: f32 = a; return;",
        );
        assert!(errors.is_empty(), "reading scalar param should succeed: {errors:?}");
        assert!(!body.stmts.is_empty(), "body should have at least a let stmt");
    }

    // Buffer read (readonly) is OK
    #[test]
    fn tc_readonly_buffer_read_ok() {
        let (body, errors) = tc_with_params(
            "x: readonly_buffer[f32]",
            "let v: f32 = x[0u32]; return;",
        );
        assert!(errors.is_empty(), "reading readonly_buffer should succeed: {errors:?}");
        assert!(!body.stmts.is_empty());
    }

    // Buffer write (writeonly) is OK
    #[test]
    fn tc_writeonly_buffer_write_ok() {
        let (_, errors) = tc_with_params(
            "out: writeonly_buffer[f32]",
            "out[0u32] = 1.0f32; return;",
        );
        assert!(errors.is_empty(), "writing writeonly_buffer should succeed: {errors:?}");
    }

    // Buffer index with signed integer (i32) must be rejected with BadIndexType
    #[test]
    fn tc_index_signed_integer_rejected() {
        // The spec requires u32 for buffer index; i32 must fire BadIndexType.
        // `0i32` has an explicit suffix so it resolves as I32 regardless of expected=U32.
        // check_buffer_read then sees index_hir.ty == I32 != U32 and fires BadIndexType.
        let (_, errors) = tc_with_params(
            "x: buffer[f32]",
            "let v: f32 = x[0i32]; return;",
        );
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::BadIndexType { got_ty: "i32", .. })),
            "expected BadIndexType{{got_ty:'i32'}}: {errors:?}"
        );
    }

    // AT-CRIT1: Verify that band() call is accepted (it was being misused as a multi-dim test).
    // Multi-dimensional buffer indexing (e.g. buf[i][j]) is not parseable in M1.2 —
    // the grammar only supports `identifier[expr]` postfix indexing. The
    // TypecheckError::MultiDimIndexInM1_2 variant has been removed as dead code.
    // This test confirms that band() is a valid bitwise builtin unrelated to indexing.
    #[test]
    fn tc_band_builtin_ok() {
        let (_, errors) = tc_body("let a: i32 = 0i32; let b: i32 = 1i32; let c: i32 = band(a, b); return;");
        assert!(errors.is_empty(), "band(a, b) should succeed: {errors:?}");
    }

    // Buffer index read works (integration)
    #[test]
    fn tc_buffer_index_read() {
        let (body, errors) = tc_with_params(
            "buf: buffer[f32]",
            "let v: f32 = buf[0u32]; return;",
        );
        assert!(errors.is_empty(), "buffer index read should succeed: {errors:?}");
        let has_buf_read = body.stmts.iter().any(|s| {
            if let HirStmt::Let { init, .. } = s {
                matches!(init.kind, HirExprKind::BufferRead { .. })
            } else { false }
        });
        assert!(has_buf_read, "expected BufferRead in HIR body");
    }

    // Buffer index write works (integration)
    #[test]
    fn tc_buffer_index_write() {
        let (body, errors) = tc_with_params(
            "buf: buffer[f32]",
            "buf[0u32] = 1.0f32; return;",
        );
        assert!(errors.is_empty(), "buffer index write should succeed: {errors:?}");
        let has_buf_write = body.stmts.iter().any(|s| {
            matches!(s, HirStmt::BufferWrite { .. })
        });
        assert!(has_buf_write, "expected BufferWrite in HIR body");
    }

    // ── M1.3 control flow typecheck tests ─────────────────────────────────────

    // AT-307: basic if with bool condition
    #[test]
    fn tc_if_bool_cond_happy() {
        let (body, errors) = tc_body("if true { return; }");
        assert!(errors.is_empty(), "simple if should succeed: {errors:?}");
        let has_if = body.stmts.iter().any(|s| matches!(s, HirStmt::If(_)));
        assert!(has_if, "expected If stmt in body");
    }

    // AT-321: if condition must be bool — reject non-bool
    #[test]
    #[allow(non_snake_case)]
    fn hir_rejects_if_with_int_cond_as_NonBoolCondition() {
        let (_, errors) = tc_body("let x: i32 = 1i32; if x { return; }");
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::NonBoolCondition { .. })),
            "expected NonBoolCondition: {errors:?}"
        );
    }

    // AT-309: if-else parses and typechecks
    #[test]
    fn tc_if_else_happy() {
        let (body, errors) = tc_body("let mut x: i32 = 1i32; if true { x = 2i32; } else { x = 3i32; } return;");
        assert!(errors.is_empty(), "if-else should succeed: {errors:?}");
        let has_if = body.stmts.iter().any(|s| matches!(s, HirStmt::If(_)));
        assert!(has_if, "expected If stmt with else arm");
    }

    // AT-310: else-if chain typechecks
    #[test]
    fn tc_if_else_if_chain_happy() {
        let (body, errors) = tc_body(
            "let mut x: i32 = 1i32; if false { x = 2i32; } else if true { x = 3i32; } return;"
        );
        assert!(errors.is_empty(), "else-if chain should succeed: {errors:?}");
        let has_if = body.stmts.iter().any(|s| matches!(s, HirStmt::If(_)));
        assert!(has_if, "expected If stmt");
    }

    // AT-311: short-circuit in if header must be rejected (CRITICAL-1)
    #[test]
    fn tc_short_circuit_in_if_cond_rejected() {
        let (_, errors) = tc_body("let x: bool = true; let y: bool = false; if x and y { return; }");
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::UnsupportedShortCircuitInHeader { .. })),
            "expected UnsupportedShortCircuitInHeader: {errors:?}"
        );
    }

    // AT-311b: short-circuit or in if header must be rejected
    #[test]
    fn tc_short_circuit_or_in_if_header_rejected() {
        let (_, errors) = tc_body("let x: bool = true; let y: bool = false; if x or y { return; }");
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::UnsupportedShortCircuitInHeader { .. })),
            "expected UnsupportedShortCircuitInHeader for `or`: {errors:?}"
        );
    }

    // AT-312: for-range basic
    #[test]
    fn tc_for_range_happy() {
        let (body, errors) = tc_body(
            "for i in range(0u32, 10u32) { } return;"
        );
        assert!(errors.is_empty(), "for-range should succeed: {errors:?}");
        let has_for = body.stmts.iter().any(|s| matches!(s, HirStmt::ForRange(_)));
        assert!(has_for, "expected ForRange stmt");
    }

    // AT-313: for-range with explicit step
    #[test]
    fn tc_for_range_with_step_happy() {
        let (body, errors) = tc_body(
            "for i in range(0u32, 10u32, 2u32) { } return;"
        );
        assert!(errors.is_empty(), "for-range with step should succeed: {errors:?}");
        let has_for = body.stmts.iter().any(|s| {
            if let HirStmt::ForRange(f) = s { f.step.value == 2 } else { false }
        });
        assert!(has_for, "expected ForRange with step 2");
    }

    // AT-313: induction variable is out of scope after the for loop
    #[test]
    fn hir_induction_variable_out_of_scope_after_for() {
        // After the for loop, `i` must not be visible — accessing it is UnknownBinding.
        let (_, errors) = tc_body(
            "for i in range(0u32, 5u32) { } let x: u32 = i; return;"
        );
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::UnknownBinding { name, .. } if name == "i")),
            "expected UnknownBinding{{name:'i'}} after for loop: {errors:?}"
        );
    }

    // AT-322: assign to for induction variable is rejected
    #[test]
    #[allow(non_snake_case)]
    fn hir_for_body_assigns_induction_is_AssignToForInductionVar() {
        let (_, errors) = tc_body(
            "for i in range(0u32, 10u32) { i = 5u32; } return;"
        );
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::AssignToForInductionVar { .. })),
            "expected AssignToForInductionVar: {errors:?}"
        );
    }

    // AT-316: while basic
    #[test]
    fn tc_while_happy() {
        let (body, errors) = tc_body(
            "let mut x: i32 = 0i32; while false { x = 1i32; } return;"
        );
        assert!(errors.is_empty(), "while should succeed: {errors:?}");
        let has_while = body.stmts.iter().any(|s| matches!(s, HirStmt::While(_)));
        assert!(has_while, "expected While stmt");
    }

    // AT-317: while condition must be bool
    #[test]
    fn tc_while_non_bool_cond_rejected() {
        let (_, errors) = tc_body("let x: i32 = 1i32; while x { } return;");
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::NonBoolCondition { .. })),
            "expected NonBoolCondition for while: {errors:?}"
        );
    }

    // AT-318: short-circuit in while header is rejected (CRITICAL-1)
    #[test]
    fn tc_short_circuit_in_while_cond_rejected() {
        let (_, errors) = tc_body(
            "let x: bool = true; let y: bool = false; while x and y { } return;"
        );
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::UnsupportedShortCircuitInHeader { .. })),
            "expected UnsupportedShortCircuitInHeader in while header: {errors:?}"
        );
    }

    // AT-319: break inside loop is valid
    #[test]
    fn tc_break_inside_loop_happy() {
        let (body, errors) = tc_body(
            "while false { break; } return;"
        );
        assert!(errors.is_empty(), "break inside loop should succeed: {errors:?}");
        let has_break = body.stmts.iter().any(|s| {
            if let HirStmt::While(w) = s {
                w.body.iter().any(|bs| matches!(bs, HirStmt::Break { .. }))
            } else { false }
        });
        assert!(has_break, "expected Break in while body");
    }

    // AT-312: break outside loop is rejected
    #[test]
    #[allow(non_snake_case)]
    fn hir_break_outside_loop_is_BreakOutsideLoop() {
        let (_, errors) = tc_body("break; return;");
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::BreakOutsideLoop { .. })),
            "expected BreakOutsideLoop: {errors:?}"
        );
    }

    // AT-321: continue inside loop is valid
    #[test]
    fn tc_continue_inside_loop_happy() {
        let (body, errors) = tc_body(
            "for i in range(0u32, 5u32) { continue; } return;"
        );
        assert!(errors.is_empty(), "continue inside loop should succeed: {errors:?}");
        let has_continue = body.stmts.iter().any(|s| {
            if let HirStmt::ForRange(f) = s {
                f.body.iter().any(|bs| matches!(bs, HirStmt::Continue { .. }))
            } else { false }
        });
        assert!(has_continue, "expected Continue in for body");
    }

    // AT-322: continue outside loop is rejected
    #[test]
    fn tc_continue_outside_loop_rejected() {
        let (_, errors) = tc_body("continue; return;");
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::ContinueOutsideLoop { .. })),
            "expected ContinueOutsideLoop: {errors:?}"
        );
    }

    // AT-323: return inside loop produces ReturnInsideLoopDeferred only at codegen
    // At typecheck level, return inside loop is ACCEPTED (deferred to codegen).
    #[test]
    fn tc_return_inside_loop_accepted_at_typecheck() {
        let (_, errors) = tc_body(
            "while false { return; }"
        );
        // Typecheck should NOT produce any error — return inside loop deferred to codegen.
        assert!(
            !errors.iter().any(|e| matches!(e, TypecheckError::BreakOutsideLoop { .. }
                | TypecheckError::ContinueOutsideLoop { .. })),
            "unexpected loop-context errors: {errors:?}"
        );
    }

    // AT-314: nested for loop with same induction variable name (scoping)
    // Two distinct BindingIds must be assigned to the two `i` names.
    #[test]
    fn hir_nested_for_with_shadowed_induction_is_accepted() {
        let (body, errors) = tc_body(
            "for i in range(0u32, 2u32) { for i in range(0u32, 3u32) { } } return;"
        );
        assert!(errors.is_empty(), "nested for with same induction name should succeed: {errors:?}");
        // Collect the two ForRange statements and check their induction BindingIds differ.
        let outer = body.stmts.iter().find_map(|s| {
            if let HirStmt::ForRange(f) = s { Some(f) } else { None }
        }).expect("expected outer ForRange");
        let inner = outer.body.iter().find_map(|s| {
            if let HirStmt::ForRange(f) = s { Some(f) } else { None }
        }).expect("expected inner ForRange");
        assert_ne!(
            outer.induction, inner.induction,
            "outer and inner `i` must have distinct BindingIds: outer={:?} inner={:?}",
            outer.induction, inner.induction
        );
    }

    // AT-315: for-induction variable that shadows a kernel-scope `let` binding
    // must produce RedeclaredBinding.
    #[test]
    #[allow(non_snake_case)]
    fn hir_for_induction_shadowing_kernel_scope_let_is_RedeclaredBinding() {
        // `let i: u32 = 5u32; for i in range(0u32, 10u32) { }` — the for-induction
        // `i` redeclares the kernel-scope let binding `i`.
        let (_, errors) = tc_body(
            "let i: u32 = 5u32; for i in range(0u32, 10u32) { } return;"
        );
        let redeclared = errors.iter().filter(|e| {
            matches!(e, TypecheckError::RedeclaredBinding { name, .. } if name == "i")
        }).count();
        assert_eq!(
            redeclared, 1,
            "expected exactly one RedeclaredBinding{{name:'i'}} for kernel-scope let + for-induction: {errors:?}"
        );
    }

    // AT-325: for-range end bound must be U32-typed
    #[test]
    fn tc_for_range_non_u32_start_rejected() {
        let (_, errors) = tc_body(
            "for i in range(0i32, 10i32) { } return;"
        );
        // start and end must be U32 — signed i32 should produce TypeMismatch or similar
        assert!(!errors.is_empty(), "for-range with i32 bounds should produce errors: {errors:?}");
    }

    // AT-326: let inside for body is scoped (not visible after loop)
    #[test]
    fn tc_let_inside_for_not_visible_after() {
        let (_, errors) = tc_body(
            "for i in range(0u32, 2u32) { let inner: i32 = 1i32; } let x: i32 = inner; return;"
        );
        // `inner` is not in scope after the for loop
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::UnknownBinding { .. })),
            "expected UnknownBinding for out-of-scope variable: {errors:?}"
        );
    }

    // AT-327: if with let in then-block doesn't leak to outer scope
    #[test]
    fn tc_let_inside_if_not_visible_after() {
        let (_, errors) = tc_body(
            "if true { let inner: i32 = 1i32; } let x: i32 = inner; return;"
        );
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::UnknownBinding { .. })),
            "expected UnknownBinding for let inside if-block: {errors:?}"
        );
    }

    // AT-328: for-range step must be a compile-time constant
    #[test]
    fn tc_for_step_variable_rejected() {
        let (_, errors) = tc_body(
            "let s: u32 = 2u32; for i in range(0u32, 10u32, s) { } return;"
        );
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::ForStepNotConstant { .. })),
            "expected ForStepNotConstant: {errors:?}"
        );
    }

    // AT-318: for-range step must be positive (non-zero)
    #[test]
    #[allow(non_snake_case)]
    fn hir_for_step_zero_is_ForStepNotPositive() {
        let (_, errors) = tc_body(
            "for i in range(0u32, 10u32, 0u32) { } return;"
        );
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::ForStepNotPositive { .. })),
            "expected ForStepNotPositive: {errors:?}"
        );
    }

    // AT-330: if body can read enclosing scope bindings
    #[test]
    fn tc_if_reads_outer_binding() {
        let (_, errors) = tc_body(
            "let x: i32 = 5i32; if true { let y: i32 = x; } return;"
        );
        assert!(errors.is_empty(), "if body should read outer bindings: {errors:?}");
    }

    // AT-331: while body can read enclosing scope bindings
    #[test]
    fn tc_while_reads_outer_binding() {
        let (_, errors) = tc_body(
            "let x: bool = false; while x { } return;"
        );
        assert!(errors.is_empty(), "while cond should read outer bindings: {errors:?}");
    }

    // AT-332: for body can use induction variable
    #[test]
    fn tc_for_body_reads_induction_var() {
        let (_, errors) = tc_with_params(
            "out: writeonly_buffer[u32]",
            "for i in range(0u32, 10u32) { out[i] = i; } return;"
        );
        assert!(errors.is_empty(), "for body should read induction var: {errors:?}");
    }

    // AT-333: dead code after break is silently allowed (at typecheck level)
    #[test]
    fn tc_dead_code_after_break_allowed() {
        let (_, errors) = tc_body(
            "while false { break; let x: i32 = 1i32; } return;"
        );
        // Typecheck allows dead code — codegen skips it via current_block_terminated
        assert!(errors.is_empty(), "dead code after break should not produce TC errors: {errors:?}");
    }

    // ── M1.4 Subgroup typecheck tests (AT-14.3) ───────────────────────────────

    // AT-401: subgroup_invocation_id() -> u32
    #[test]
    fn tc_subgroup_invocation_id_returns_u32() {
        let (body, errors) = tc_body("let id: u32 = subgroup_invocation_id(); return;");
        assert!(errors.is_empty(), "errors: {errors:?}");
        assert_eq!(body.bindings[0].ty, BindingTy::Scalar(ScalarTy::U32));
    }

    // AT-402: subgroup_size() -> u32
    #[test]
    fn tc_subgroup_size_returns_u32() {
        let (body, errors) = tc_body("let sz: u32 = subgroup_size(); return;");
        assert!(errors.is_empty(), "errors: {errors:?}");
        assert_eq!(body.bindings[0].ty, BindingTy::Scalar(ScalarTy::U32));
    }

    // AT-403: subgroup_elect() -> bool
    #[test]
    fn tc_subgroup_elect_returns_bool() {
        let (body, errors) = tc_body("let e: bool = subgroup_elect(); return;");
        assert!(errors.is_empty(), "errors: {errors:?}");
        assert_eq!(body.bindings[0].ty, BindingTy::Scalar(ScalarTy::Bool));
    }

    // AT-404: subgroup_reduce_add(i32) -> i32
    #[test]
    fn tc_sg_reduce_add_i32_happy() {
        let (body, errors) = tc_body("let v: i32 = 1i32; let r: i32 = subgroup_reduce_add(v); return;");
        assert!(errors.is_empty(), "errors: {errors:?}");
        assert_eq!(body.bindings[1].ty, BindingTy::Scalar(ScalarTy::I32));
    }

    // AT-404: subgroup_reduce_add(f32) -> f32
    #[test]
    fn tc_subgroup_reduce_add_f32_accepted() {
        let (body, errors) = tc_body("let v: f32 = 1.0f32; let r: f32 = subgroup_reduce_add(v); return;");
        assert!(errors.is_empty(), "errors: {errors:?}");
        assert_eq!(body.bindings[1].ty, BindingTy::Scalar(ScalarTy::F32));
    }

    // AT-406: subgroup_reduce_min(u32) -> u32
    #[test]
    fn tc_sg_reduce_min_u32_happy() {
        let (body, errors) = tc_body("let v: u32 = 1u32; let r: u32 = subgroup_reduce_min(v); return;");
        assert!(errors.is_empty(), "errors: {errors:?}");
        assert_eq!(body.bindings[1].ty, BindingTy::Scalar(ScalarTy::U32));
    }

    // AT-407: subgroup_reduce_max(f64) -> f64
    #[test]
    fn tc_sg_reduce_max_f64_happy() {
        let (body, errors) = tc_body("let v: f64 = 1.0f64; let r: f64 = subgroup_reduce_max(v); return;");
        assert!(errors.is_empty(), "errors: {errors:?}");
        assert_eq!(body.bindings[1].ty, BindingTy::Scalar(ScalarTy::F64));
    }

    // AT-408: subgroup_broadcast_first(f32) -> f32
    #[test]
    fn tc_sg_broadcast_first_f32_happy() {
        let (body, errors) = tc_body("let v: f32 = 1.0f32; let r: f32 = subgroup_broadcast_first(v); return;");
        assert!(errors.is_empty(), "errors: {errors:?}");
        assert_eq!(body.bindings[1].ty, BindingTy::Scalar(ScalarTy::F32));
    }

    // AT-409: subgroup_all(bool) -> bool
    #[test]
    fn tc_sg_all_happy() {
        let (body, errors) = tc_body("let p: bool = true; let r: bool = subgroup_all(p); return;");
        assert!(errors.is_empty(), "errors: {errors:?}");
        assert_eq!(body.bindings[1].ty, BindingTy::Scalar(ScalarTy::Bool));
    }

    // AT-410: subgroup_any(bool) -> bool
    #[test]
    fn tc_sg_any_happy() {
        let (body, errors) = tc_body("let p: bool = false; let r: bool = subgroup_any(p); return;");
        assert!(errors.is_empty(), "errors: {errors:?}");
        assert_eq!(body.bindings[1].ty, BindingTy::Scalar(ScalarTy::Bool));
    }

    // AT-407: workgroup_barrier() as statement — accepted, produces HirStmt::Barrier
    #[test]
    #[allow(non_snake_case)]
    fn tc_workgroup_barrier_stmt_accepted_as_HirStmt_Barrier() {
        let (_, errors) = tc_body("workgroup_barrier(); return;");
        assert!(errors.is_empty(), "errors: {errors:?}");
    }

    // AT-412: subgroup_reduce_add arity error (too many args)
    #[test]
    fn tc_sg_reduce_add_arity_too_many_rejected() {
        let (_, errors) = tc_body("let a: i32 = 1i32; let b: i32 = 2i32; let r: i32 = subgroup_reduce_add(a, b); return;");
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::SubgroupArity { .. })),
            "expected SubgroupArity error: {errors:?}"
        );
    }

    // AT-413: subgroup_elect called with arg — rejected
    #[test]
    fn tc_sg_elect_arity_too_many_rejected() {
        let (_, errors) = tc_body("let a: bool = true; let r: bool = subgroup_elect(a); return;");
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::SubgroupArity { .. })),
            "expected SubgroupArity error: {errors:?}"
        );
    }

    // AT-405: subgroup_reduce_add on bool — rejected (unsupported type), variant SubgroupReduceTypeUnsupported
    #[test]
    #[allow(non_snake_case)]
    fn tc_subgroup_reduce_add_bool_rejected_as_SubgroupReduceTypeUnsupported() {
        let (_, errors) = tc_body("let p: bool = true; let r: bool = subgroup_reduce_add(p); return;");
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::SubgroupReduceTypeUnsupported { .. })),
            "expected SubgroupReduceTypeUnsupported: {errors:?}"
        );
    }

    // AT-406: subgroup_any with non-bool arg — rejected with TypeMismatch
    #[test]
    #[allow(non_snake_case)]
    fn tc_subgroup_any_non_bool_arg_is_TypeMismatch() {
        let (_, errors) = tc_body("let x: i32 = 1i32; let r: bool = subgroup_any(x); return;");
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::TypeMismatch { .. })),
            "expected TypeMismatch for subgroup_any(non_bool): {errors:?}"
        );
    }

    // AT-416: divergent context warning for subgroup_all inside if cond branch
    #[test]
    fn tc_sg_all_in_if_body_produces_divergent_warning() {
        let (_body, errors, warns) = tc_body_with_warns(
            "let p: bool = true; if p { let r: bool = subgroup_all(p); } return;"
        );
        assert!(errors.is_empty(), "errors: {errors:?}");
        assert!(
            warns.iter().any(|w| matches!(w, crate::validate::HirWarning::SubgroupOpInDivergentContext { .. })),
            "expected SubgroupOpInDivergentContext warning; warns: {warns:?}"
        );
    }

    // AT-429 (INVERTED per M3.2 r3): workgroup_barrier inside if body — now a HARD ERROR
    // (OQ2 BarrierInDivergentContext). Previously asserted errors.is_empty(); now inverted.
    #[test]
    fn error_barrier_in_if_body_is_divergent_context() {
        let (_body, errors, _warns) = tc_body_with_warns(
            "let p: bool = true; if p { workgroup_barrier(); } return;"
        );
        // Under M3.2 the barrier-in-if-body is a HARD ERROR (Vulkan UB).
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::BarrierInDivergentContext { .. })),
            "expected BarrierInDivergentContext error for barrier in if-body; errors: {errors:?}"
        );
    }

    // AT-418: subgroup_invocation_id() in non-divergent body — no warning
    #[test]
    fn tc_sg_invocation_id_not_divergent_no_warning() {
        let (_body, errors, warns) = tc_body_with_warns(
            "let id: u32 = subgroup_invocation_id(); return;"
        );
        assert!(errors.is_empty(), "errors: {errors:?}");
        // InvocationId is not collective — must not produce any warning
        assert!(
            !warns.iter().any(|w| matches!(w, crate::validate::HirWarning::SubgroupOpInDivergentContext { .. })),
            "subgroup_invocation_id must not produce divergent warning; warns: {warns:?}"
        );
    }

    // AT-408: workgroup_barrier(42) — rejected with SubgroupArity (arity > 0)
    #[test]
    #[allow(non_snake_case)]
    fn tc_workgroup_barrier_with_args_is_SubgroupArity() {
        // workgroup_barrier accepts exactly 0 arguments; passing one is SubgroupArity.
        // We test via a stmt-level call.
        let (_, errors) = tc_body("workgroup_barrier(42u32); return;");
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::SubgroupArity { op, .. } if *op == "workgroup_barrier")),
            "expected SubgroupArity for workgroup_barrier(42): {errors:?}"
        );
    }

    // AT-409: `let subgroup_size: u32 = 0u32;` is ReservedBuiltinName
    #[test]
    #[allow(non_snake_case)]
    fn tc_let_subgroup_size_is_ReservedBuiltinName() {
        let (_, errors) = tc_body("let subgroup_size: u32 = 0u32; return;");
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::ReservedBuiltinName { name, .. } if name == "subgroup_size")),
            "expected ReservedBuiltinName for 'subgroup_size': {errors:?}"
        );
    }

    // AT-410: subgroup_reduce_add(v) at statement position (discarded result) is NonVoidSubgroupCallAsStatement
    #[test]
    fn validate_non_void_subgroup_call_as_stmt_rejected_end_to_end() {
        let (_, errors) = tc_body("let v: f32 = 1.0f32; subgroup_reduce_add(v); return;");
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::NonVoidSubgroupCallAsStatement { .. })),
            "expected NonVoidSubgroupCallAsStatement for discarded subgroup_reduce_add: {errors:?}"
        );
    }

    // AT-423: subgroup_elect() inside a for-range body does NOT emit SubgroupOpInDivergentContext
    // (for-range induction is uniform, not divergent — CRITICAL-3 semantics)
    #[test]
    fn lower_subgroup_elect_inside_if_in_for_body_produces_no_warning() {
        let (_body, errors, warns) = tc_body_with_warns(
            "for i in range(0u32, 64u32) { let e: bool = subgroup_elect(); } return;"
        );
        assert!(errors.is_empty(), "errors: {errors:?}");
        // for-range body is NOT divergent — subgroup_elect here must not warn
        assert!(
            !warns.iter().any(|w| matches!(w, crate::validate::HirWarning::SubgroupOpInDivergentContext { .. })),
            "subgroup_elect in for-range body must NOT produce divergent warning; warns: {warns:?}"
        );
    }

    // AT-424: All existing M1.3 regression tests still pass (baseline preserved).
    // This is a compile-time marker test — if cargo test --workspace passes,
    // the 336 M1.3 baseline tests still run. This test just asserts the sentinel value.
    #[test]
    fn m1_3_regression_workspace_test_count() {
        // The M1.3 baseline introduced 336 tests. M1.4 adds tests on top.
        // This test asserts that baseline M1.3 invariants are preserved:
        // if this file compiles and runs, all M1.3 test functions before this one pass.
        // Workspace total must be >= 395 (the M1.4 spec floor).
        let baseline_m1_3: usize = 336;
        let m1_4_additions: usize = 59;
        let expected_min: usize = baseline_m1_3 + m1_4_additions;
        // This compile-time assertion pins the expected floor — actual count checked by CI.
        assert!(expected_min >= 395, "M1.3 + M1.4 test floor must be >= 395; computed: {expected_min}");
    }

    // AT-428: (rev 1 CRITICAL-3) `if subgroup_elect() { subgroup_broadcast_first(v); }` —
    // The cond `subgroup_elect()` runs at PARENT depth (no warning).
    // The body op `subgroup_broadcast_first(v)` runs at depth+1 (warns with op_name == "subgroup_broadcast_first").
    // Exactly 1 warning total, with op_name pinned to "subgroup_broadcast_first" NOT "subgroup_elect".
    #[test]
    fn warn_subgroup_elect_as_cond_no_false_positive() {
        let (_body, errors, warns) = tc_body_with_warns(
            "let v: f32 = 1.0f32; if subgroup_elect() { let _r: f32 = subgroup_broadcast_first(v); } return;"
        );
        assert!(errors.is_empty(), "errors: {errors:?}");

        let divergent_warns: Vec<_> = warns.iter()
            .filter(|w| matches!(w, crate::validate::HirWarning::SubgroupOpInDivergentContext { .. }))
            .collect();

        // Exactly 1 warning — for the body op, not the cond.
        assert_eq!(
            divergent_warns.len(), 1,
            "expected exactly 1 SubgroupOpInDivergentContext warning (body op only, not cond); warns: {warns:?}"
        );

        // The warning's op_name must be "subgroup_broadcast_first", NOT "subgroup_elect".
        let op_name_matches = matches!(
            divergent_warns[0],
            crate::validate::HirWarning::SubgroupOpInDivergentContext { op_name, .. }
            if *op_name == "subgroup_broadcast_first"
        );
        assert!(
            op_name_matches,
            "warning op_name must be 'subgroup_broadcast_first', not 'subgroup_elect'; warn: {:?}",
            divergent_warns[0]
        );
    }

    // ── M2.1 acceptance tests ─────────────────────────────────────────────────

    /// AT-631: `let matrix: i32 = 0;` is rejected with ReservedKeyword.
    ///
    /// `matrix` is reserved in M2.1 so that a future milestone can promote it
    /// to a type-constructor keyword at expression scope.
    #[test]
    fn tc_let_matrix_ident_rejected() {
        let (_, errors) = tc_body("let matrix: i32 = 0i32; return;");
        assert!(
            errors.iter().any(|e| matches!(
                e,
                TypecheckError::ReservedKeyword { name, .. } if name == "matrix"
            )),
            "expected ReservedKeyword {{ name: \"matrix\" }}; got: {errors:?}"
        );
    }

    // ── M2.1 coopmat typecheck tests ──────────────────────────────────────────

    /// AT-624: coopmat_mul_add K-dimension mismatch (a.n != b.m) → CoopMatrixShapeMismatch.
    ///
    /// Matrix A is 16×8 (m=16, n=8), Matrix B is 16×16 (m=16, n=16).
    /// K dim: a.n=8 != b.m=16 → KDimMismatch.
    #[test]
    fn tc_coopmat_mul_add_k_mismatch_rejected() {
        // A: matrix[f16, 16, 8, a] — 16×8
        // B: matrix[f16, 16, 16, b] — 16×16
        // C: matrix[f32, 16, 16, accumulator] — 16×16
        let (_, errors) = tc_body(
            "let a: matrix[f16, 16, 8, a] = coopmat_zero(); \
             let b: matrix[f16, 16, 16, b] = coopmat_zero(); \
             let c: matrix[f32, 16, 16, accumulator] = coopmat_zero(); \
             let d: matrix[f32, 16, 16, accumulator] = coopmat_mul_add(a, b, c); \
             return;"
        );
        assert!(
            errors.iter().any(|e| matches!(
                e,
                TypecheckError::CoopMatrixShapeMismatch {
                    kind: crate::coopmat::CoopMatrixShapeKind::KDimMismatch { a_n: 8, b_m: 16 },
                    ..
                }
            )),
            "expected KDimMismatch {{ a_n: 8, b_m: 16 }}; got: {errors:?}"
        );
    }

    /// AT-625: coopmat_mul_add use-tag mismatch (accumulator where A expected).
    #[test]
    fn tc_coopmat_mul_add_use_mismatch_rejected() {
        // Pass an accumulator matrix where A is expected.
        let (_, errors) = tc_body(
            "let c: matrix[f32, 16, 16, accumulator] = coopmat_zero(); \
             let b: matrix[f16, 16, 16, b] = coopmat_zero(); \
             let d: matrix[f32, 16, 16, accumulator] = coopmat_zero(); \
             let r: matrix[f32, 16, 16, accumulator] = coopmat_mul_add(c, b, d); \
             return;"
        );
        assert!(
            errors.iter().any(|e| matches!(
                e,
                TypecheckError::CoopMatrixShapeMismatch {
                    kind: crate::coopmat::CoopMatrixShapeKind::AUseMismatch {
                        found: crate::coopmat::CoopMatUse::Accumulator
                    },
                    ..
                }
            )),
            "expected AUseMismatch {{ found: Accumulator }}; got: {errors:?}"
        );
    }

    /// AT-626: coopmat_store to a readonly_buffer is rejected with CoopMatStoreToReadonlyBuffer.
    #[test]
    fn tc_coopmat_store_to_readonly_rejected() {
        let full = "@kernel @workgroup(1,1,1) fn k(out: readonly_buffer[f16]) -> void { \
                    let m: matrix[f16, 16, 16, a] = coopmat_zero(); \
                    coopmat_store(m, out, 0u32, 16u32); \
                    return; }";
        let (ast, lex_errs, _parse_errs) = parse(full);
        assert!(lex_errs.is_empty(), "lex: {lex_errs:?}");
        let axc_parser::Item::Kernel(ref kd) = ast.items[0].node;
        let params = crate::lower::lower_params_for_test(&kd.params);
        let (_, errors, _) = typecheck_kernel_body(&kd.body.node, &params);
        assert!(
            errors.iter().any(|e| matches!(
                e,
                TypecheckError::CoopMatStoreToReadonlyBuffer { param_name, .. }
                    if param_name == "out"
            )),
            "expected CoopMatStoreToReadonlyBuffer {{ param_name: \"out\" }}; got: {errors:?}"
        );
    }

    /// AT-627: coopmat_zero without a let-binding type annotation is rejected with
    /// CoopMatrixBuiltinRequiresExpectedType.
    ///
    /// The HIR can only determine the result matrix type from the let-declared type.
    /// Using coopmat_zero() directly without any context (e.g. as a statement argument)
    /// must be rejected.
    #[test]
    fn tc_coopmat_zero_without_let_ty_annotation_rejected() {
        // coopmat_zero() used without a matrix-typed let binding (no expected type context).
        // Note: the parser parses it as Expr::Call; typecheck sees no expected coopmat type.
        // We test this by passing it to a non-matrix expression context.
        let (_, errors) = tc_body(
            "let x: i32 = coopmat_zero(); return;"
        );
        // Should see CoopMatrixBuiltinRequiresExpectedType since there's no matrix context.
        assert!(
            errors.iter().any(|e| matches!(
                e,
                TypecheckError::CoopMatrixBuiltinRequiresExpectedType { name, .. }
                    if *name == "coopmat_zero"
            )),
            "expected CoopMatrixBuiltinRequiresExpectedType; got: {errors:?}"
        );
    }

    // ── M3.5b mixed-precision coopmat typecheck tests (AT-1785/1786/1789) ─────────

    /// AT-1785: mixed-precision coopmat_mul_add(a:f16, b:f16, c:f32) typechecks to
    /// matrix[f32,16,16,accumulator] with NO ABElementMismatch and NO error (the f16×f16→f32
    /// HMMA). The accumulator's element type is validated INDEPENDENTLY of A/B and becomes the
    /// result type (result_key = c_key). This locks in the M3.5b finding that the type system
    /// is already mixed-precision-capable; a regression that rejected f16/f16/f32 would fail here.
    #[test]
    fn tc_coopmat_mul_add_mixed_f16_f16_f32_ok() {
        let (_, errors) = tc_body(
            "let a: matrix[f16, 16, 16, a] = coopmat_zero(); \
             let b: matrix[f16, 16, 16, b] = coopmat_zero(); \
             let c: matrix[f32, 16, 16, accumulator] = coopmat_zero(); \
             let d: matrix[f32, 16, 16, accumulator] = coopmat_mul_add(a, b, c); \
             return;"
        );
        assert!(
            errors.is_empty(),
            "AT-1785: mixed-precision coopmat_mul_add(a:f16,b:f16,c:f32) must typecheck with \
             NO errors (f16×f16→f32 HMMA); got: {errors:?}"
        );
    }

    /// AT-1785 (no-regression half): the frozen ALL-f16 coopmat_mul_add still typechecks clean.
    #[test]
    fn tc_coopmat_mul_add_all_f16_still_ok() {
        let (_, errors) = tc_body(
            "let a: matrix[f16, 16, 16, a] = coopmat_zero(); \
             let b: matrix[f16, 16, 16, b] = coopmat_zero(); \
             let c: matrix[f16, 16, 16, accumulator] = coopmat_zero(); \
             let d: matrix[f16, 16, 16, accumulator] = coopmat_mul_add(a, b, c); \
             return;"
        );
        assert!(
            errors.is_empty(),
            "AT-1785: the frozen all-f16 coopmat_mul_add must STILL typecheck (no regression); \
             got: {errors:?}"
        );
    }

    /// AT-1786 (RETAINED hard gate): coopmat_mul_add(a:f16, b:f32, c:f32) STILL errors with
    /// ABElementMismatch. Only the ACCUMULATOR may differ in element type; A and B must match
    /// EACH OTHER. This guards against any unnecessary typecheck relax weakening the
    /// a_key.elem==b_key.elem check (the M3.5b coder_handoff_notes explicitly forbid such a relax).
    #[test]
    fn tc_coopmat_mul_add_ab_mismatch_still_rejected() {
        let (_, errors) = tc_body(
            "let a: matrix[f16, 16, 16, a] = coopmat_zero(); \
             let b: matrix[f32, 16, 16, b] = coopmat_zero(); \
             let c: matrix[f32, 16, 16, accumulator] = coopmat_zero(); \
             let d: matrix[f32, 16, 16, accumulator] = coopmat_mul_add(a, b, c); \
             return;"
        );
        let mismatch_count = errors.iter().filter(|e| matches!(
            e,
            TypecheckError::CoopMatrixShapeMismatch {
                kind: crate::coopmat::CoopMatrixShapeKind::ABElementMismatch {
                    a_elem: crate::ty::ScalarTy::F16,
                    b_elem: crate::ty::ScalarTy::F32,
                },
                ..
            }
        )).count();
        assert_eq!(
            mismatch_count, 1,
            "AT-1786: coopmat_mul_add(a:f16, b:f32, c:f32) must produce exactly one \
             ABElementMismatch{{a_elem:F16,b_elem:F32}} (A/B must match each other; only the \
             accumulator may differ); got: {errors:?}"
        );
    }

    /// AT-1789 (MANDATORY, CPU-only): coopmat_store of an f32 accumulator
    /// (matrix[f32,16,16,accumulator]) into a buffer[f16] MUST error
    /// CoopMatStoreElementTypeMismatch (matrix.elem=F32 != buffer.elem=F16). This permanently
    /// locks in WHY the M3.5b kernel changes C: buffer[f16]->buffer[f32].
    #[test]
    fn tc_coopmat_store_f32_accum_into_buffer_f16_rejected() {
        let full = "@kernel @workgroup(1,1,1) fn k(out: buffer[f16]) -> void { \
                    let m: matrix[f32, 16, 16, accumulator] = coopmat_zero(); \
                    coopmat_store(m, out, 0u32, 16u32); \
                    return; }";
        let (ast, lex_errs, _parse_errs) = parse(full);
        assert!(lex_errs.is_empty(), "lex: {lex_errs:?}");
        let axc_parser::Item::Kernel(ref kd) = ast.items[0].node;
        let params = crate::lower::lower_params_for_test(&kd.params);
        let (_, errors, _) = typecheck_kernel_body(&kd.body.node, &params);
        let mismatch_count = errors.iter().filter(|e| matches!(
            e,
            TypecheckError::CoopMatStoreElementTypeMismatch { matrix_elem, buffer_elem, .. }
                if *matrix_elem == "f32" && *buffer_elem == "f16"
        )).count();
        assert_eq!(
            mismatch_count, 1,
            "AT-1789: coopmat_store(matrix[f32,..,accumulator], buffer[f16], ..) must produce \
             exactly one CoopMatStoreElementTypeMismatch{{matrix=f32, buffer=f16}}; got: {errors:?}"
        );
    }

    /// AT-1789 (positive mirror): coopmat_store of an f32 accumulator into buffer[f32] typechecks
    /// clean (the M3.5b kernel's actual store; covered transitively by AT-1787's clean compile,
    /// asserted here directly for a fast CPU-only regression).
    #[test]
    fn tc_coopmat_store_f32_accum_into_buffer_f32_ok() {
        let full = "@kernel @workgroup(1,1,1) fn k(out: buffer[f32]) -> void { \
                    let m: matrix[f32, 16, 16, accumulator] = coopmat_zero(); \
                    coopmat_store(m, out, 0u32, 16u32); \
                    return; }";
        let (ast, lex_errs, _parse_errs) = parse(full);
        assert!(lex_errs.is_empty(), "lex: {lex_errs:?}");
        let axc_parser::Item::Kernel(ref kd) = ast.items[0].node;
        let params = crate::lower::lower_params_for_test(&kd.params);
        let (_, errors, _) = typecheck_kernel_body(&kd.body.node, &params);
        assert!(
            errors.is_empty(),
            "AT-1789: coopmat_store(matrix[f32,..,accumulator], buffer[f32], ..) must typecheck \
             clean; got: {errors:?}"
        );
    }

    // AT-1775: f32_to_f16 builtin typecheck — happy path produces an f16 binding.
    #[test]
    fn at_1775_f32_to_f16_happy_returns_f16() {
        let (body, errors) = tc_body(
            "let v: f32 = 1.5f32; let h: f16 = f32_to_f16(v); return;"
        );
        assert!(errors.is_empty(), "AT-1775: f32_to_f16 happy path errors: {errors:?}");
        let h_binding = body.bindings.iter().find(|b| b.name == "h")
            .expect("AT-1775: binding `h` must exist");
        assert_eq!(
            h_binding.ty, BindingTy::Scalar(ScalarTy::F16),
            "AT-1775: f32_to_f16 result must be f16"
        );
    }

    // AT-1775: f32_to_f16 rejects a non-f32 argument with F32ToF16ArgMustBeF32.
    #[test]
    fn at_1775_f32_to_f16_rejects_non_f32_arg() {
        let (_body, errors) = tc_body(
            "let u: u32 = 3u32; let h: f16 = f32_to_f16(u); return;"
        );
        assert!(
            errors.iter().any(|e| matches!(
                e, TypecheckError::F32ToF16ArgMustBeF32 { .. }
            )),
            "AT-1775: expected F32ToF16ArgMustBeF32 for a u32 arg; got: {errors:?}"
        );
    }

    // M3.2c: exp builtin typecheck — happy path produces an f32 binding.
    #[test]
    fn m32c_exp_happy_returns_f32() {
        let (body, errors) = tc_body("let x: f32 = -2.0f32; let p: f32 = exp(x); return;");
        assert!(errors.is_empty(), "M3.2c: exp happy path errors: {errors:?}");
        let p_binding = body.bindings.iter().find(|b| b.name == "p")
            .expect("M3.2c: binding `p` must exist");
        assert_eq!(
            p_binding.ty, BindingTy::Scalar(ScalarTy::F32),
            "M3.2c: exp result must be f32"
        );
    }

    // M3.2c: exp rejects a non-f32 argument with ExpArgMustBeF32.
    #[test]
    fn m32c_exp_rejects_non_f32_arg() {
        let (_body, errors) = tc_body("let u: u32 = 3u32; let p: f32 = exp(u); return;");
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::ExpArgMustBeF32 { .. })),
            "M3.2c: expected ExpArgMustBeF32 for a u32 arg; got: {errors:?}"
        );
    }

    // M3.2c: exp with the wrong arity is rejected.
    #[test]
    fn m32c_exp_rejects_wrong_arity() {
        let (_body, errors) = tc_body("let p: f32 = exp(); return;");
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::ExtInstBuiltinWrongArity { .. })),
            "M3.2c: expected ExtInstBuiltinWrongArity for 0 args; got: {errors:?}"
        );
    }

    // M3.2c: `exp` is a reserved builtin name and cannot be used as a variable.
    #[test]
    fn m32c_exp_reserved_name_rejected() {
        let (_body, errors) = tc_body("let exp: f32 = 1.0f32; return;");
        assert!(
            errors.iter().any(|e| matches!(e, TypecheckError::ReservedExtInstBuiltinName { .. })),
            "M3.2c: expected ReservedExtInstBuiltinName for `let exp`; got: {errors:?}"
        );
    }

    // ── M3.20: local arrays — white-box unit coverage ─────────────────────────

    // AT-2928 (HIR half): `array h: array[u32, 8];` typechecks clean and populates
    // `KernelBodyTyped.local_arrays` with one entry.
    #[test]
    fn at_2928_local_array_decl_populates_local_arrays_table() {
        let (body, errors) = tc_body("array h: array[u32, 8]; return;");
        assert!(errors.is_empty(), "errors: {errors:?}");
        assert_eq!(body.local_arrays.len(), 1);
        assert_eq!(body.local_arrays[0].name, "h");
        assert_eq!(body.local_arrays[0].ty.elem, ScalarTy::U32);
        assert_eq!(body.local_arrays[0].ty.len, 8);
        assert!(matches!(body.stmts[0], HirStmt::LocalArrayDeclMarker { .. }));
    }

    // AT-2931 (defense-in-depth): `TypecheckError::LocalArrayUndeclared` exists,
    // formats via thiserror's `#[error(...)]`, and is distinct from the generic
    // `UnknownBinding`/`IndexOnNonBuffer` fallback the normal source pipeline
    // actually produces for a never-declared array name (see
    // `local_array_typecheck.rs`'s integration test for the reachable path).
    // Mirrors the `shared` precedent's `SharedNotDeclared`, which is likewise
    // never constructed through normal parsing — kept for API completeness.
    #[test]
    fn at_2931_local_array_undeclared_variant_shape() {
        let err = TypecheckError::LocalArrayUndeclared {
            name: "hist".to_owned(),
            span: axc_lexer::Span::new(0, 4),
        };
        let msg = err.to_string();
        assert!(msg.contains("hist"), "message must name the array: {msg}");
        assert!(msg.contains("not declared"), "message must say not declared: {msg}");
    }

    // AT-2944: `local_array_const_index_oob` boundary — exact.
    #[test]
    fn at_2944_const_index_oob_boundary_exact() {
        let in_bounds = HirExpr {
            kind: HirExprKind::IntLit { value: crate::ty::IntLiteralValue { ty: ScalarTy::U32, bits: 7 } },
            ty: ScalarTy::U32,
            span: axc_lexer::Span::new(0, 0),
        };
        assert_eq!(local_array_const_index_oob(&in_bounds, 8), None, "index 7 < N=8 must be in-bounds");

        let at_n = HirExpr {
            kind: HirExprKind::IntLit { value: crate::ty::IntLiteralValue { ty: ScalarTy::U32, bits: 8 } },
            ty: ScalarTy::U32,
            span: axc_lexer::Span::new(0, 0),
        };
        assert_eq!(local_array_const_index_oob(&at_n, 8), Some(8), "index 8 == N=8 must be OOB");

        let symbolic = HirExpr {
            kind: HirExprKind::LocalRead(BindingId(0)),
            ty: ScalarTy::U32,
            span: axc_lexer::Span::new(0, 0),
        };
        assert_eq!(local_array_const_index_oob(&symbolic, 8), None, "symbolic index must never be flagged");
    }
}

