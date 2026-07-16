//! Typed expression and statement IR for AXIOM-Compute HIR.
//!
//! Every `HirExpr` carries a resolved `ScalarTy` on every node — no type
//! placeholders, no inference. This is the invariant produced by the two-pass
//! typechecker in `typecheck.rs`.
//!
//! M1.2 adds `BufferRead`, `GidBuiltin`, and the `BufferWrite` / `BufferWriteStmt`
//! statement kinds for buffer I/O.
//! M1.3 adds `If`, `ForRange`, `While`, `Break`, `Continue` for structured control flow.
//! M1.4 adds `SubgroupBuiltin` expression kind and `Barrier` statement kind.
//! M2.1 adds `CoopMatBuiltin` expression kind and `CoopMatStore` statement kind for
//!       cooperative-matrix operations.
//! M2.5 adds `Q4_0Builtin` expression kind for byte-level access and f16 conversion.
//! M3.2 adds `SharedRead` expression kind and `SharedWrite`/`SharedDecl` statement kinds,
//!       plus the `shared` field on `KernelBodyTyped`.

use axc_lexer::Span;
use crate::ty::{ScalarTy, IntLiteralValue, FloatLiteralValue};
use crate::coopmat::CoopMatKey;

/// Opaque identifier for a local variable binding within a kernel body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BindingId(pub u32);

/// The type of a local variable binding.
///
/// M2.1 adds `CoopMatrix` as a valid binding type alongside scalars.
/// Cooperative-matrix values are let-binding-only in M2.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingTy {
    /// A scalar (numeric or bool) binding.
    Scalar(ScalarTy),
    /// A cooperative-matrix binding (M2.1+).
    CoopMatrix(CoopMatKey),
}

impl BindingTy {
    /// If this is a scalar binding, return the `ScalarTy`.
    pub fn as_scalar(&self) -> Option<ScalarTy> {
        match self {
            BindingTy::Scalar(s) => Some(*s),
            BindingTy::CoopMatrix(_) => None,
        }
    }

    /// If this is a coop-matrix binding, return the `CoopMatKey`.
    pub fn as_coopmat(&self) -> Option<CoopMatKey> {
        match self {
            BindingTy::Scalar(_) => None,
            BindingTy::CoopMatrix(k) => Some(*k),
        }
    }

    /// Human-readable name for error messages.
    pub fn display_name(&self) -> &'static str {
        match self {
            BindingTy::Scalar(s) => s.display_name(),
            BindingTy::CoopMatrix(_) => "matrix",
        }
    }
}

impl From<ScalarTy> for BindingTy {
    fn from(s: ScalarTy) -> Self {
        BindingTy::Scalar(s)
    }
}

impl From<CoopMatKey> for BindingTy {
    fn from(k: CoopMatKey) -> Self {
        BindingTy::CoopMatrix(k)
    }
}

/// A typed local variable binding.
#[derive(Debug, Clone)]
pub struct Binding {
    pub id: BindingId,
    pub name: String,
    pub ty: BindingTy,
    pub is_mutable: bool,
    pub span: Span,
}

/// A fully-typed HIR expression. Every node has a resolved scalar type.
#[derive(Debug, Clone)]
pub struct HirExpr {
    pub kind: HirExprKind,
    pub ty: ScalarTy,
    pub span: Span,
}

/// The kind of a typed HIR expression.
#[derive(Debug, Clone)]
pub enum HirExprKind {
    IntLit { value: IntLiteralValue },
    FloatLit { value: FloatLiteralValue },
    BoolLit(bool),
    LocalRead(BindingId),
    Unary {
        op: UnaryOp,
        operand: Box<HirExpr>,
    },
    Binary {
        op: BinOp,
        lhs: Box<HirExpr>,
        rhs: Box<HirExpr>,
    },
    ShortCircuit {
        op: ShortCircuitOp,
        lhs: Box<HirExpr>,
        rhs: Box<HirExpr>,
    },
    BitwiseBuiltin {
        op: BitwiseOp,
        args: Vec<HirExpr>,
    },
    /// Read one element from a buffer parameter: `buf[index]`.
    ///
    /// `param_position` is the 0-based position in the kernel's param list.
    /// `buffer_binding` is the 0-based buffer-only binding slot index.
    /// `index` must have type `U32`.
    BufferRead {
        param_position: u32,
        buffer_binding: u32,
        index: Box<HirExpr>,
    },
    /// `gid(axis)` — extract one component of `gl_GlobalInvocationID`.
    ///
    /// `axis` must be 0, 1, or 2 and is a constant resolved at compile time.
    /// Result type is always `U32`.
    GidBuiltin {
        axis: u32,
    },
    /// `local_invocation_id(axis)` — extract one component of `gl_LocalInvocationID` (M3.3d).
    ///
    /// Lowers to SPIR-V BuiltIn `LocalInvocationId` (a `uvec3` Input `OpVariable` +
    /// `OpLoad` + `OpCompositeExtract` by literal `axis`).
    ///
    /// `axis` must be 0, 1, or 2 and is a compile-time constant (identical restriction to `gid`).
    /// Result type is always `U32`.
    ///
    /// `LocalInvocationId` is a **core Shader** builtin: no new `OpCapability` is added.
    /// The var is emitted **only when this builtin is used** (unlike `gid` which is emitted
    /// whenever buffers are present).
    LocalInvocationIdBuiltin {
        axis: u32,
    },
    /// Subgroup builtin call (M1.4).
    ///
    /// Covers all subgroup operations except `workgroup_barrier`, which is a
    /// statement (`HirStmt::Barrier`). See `crate::subgroup::SubgroupOp` for variants.
    SubgroupBuiltin {
        op: crate::subgroup::SubgroupOp,
        args: Vec<HirExpr>,
    },
    /// Cooperative-matrix builtin call (M2.1 / M3.2).
    ///
    /// Covers `coopmat_zero`, `coopmat_load`, and `coopmat_mul_add`.
    /// `coopmat_store` is a STATEMENT (`HirStmt::CoopMatStore`).
    ///
    /// `result_ty` carries the resolved matrix key.
    /// - `Zero` / `Load`: resolved from the let-binding's expected type.
    /// - `MulAdd`: determined by the c-argument type.
    ///
    /// `source` discriminates the load source (M3.2):
    /// - `Some(Buffer(slot))`: SSBO buffer param (M2.1 default, byte-identical emit path).
    /// - `Some(Shared(id))`: Workgroup shared array (M3.2 PART B, single-index emit path).
    /// - `None`: Zero/MulAdd (not a load).
    CoopMatBuiltin {
        op: crate::coopmat::CoopMatBuiltin,
        args: Vec<HirExpr>,
        result_ty: CoopMatKey,
        /// Load source discriminator (M3.2). `Some` for Load (Buffer or Shared). `None` for Zero/MulAdd.
        source: Option<crate::coopmat::CoopMatLoadSource>,
    },
    /// Q4_0-path builtin call (M2.5).
    ///
    /// Covers the five byte-access and conversion primitives:
    /// - `ptr_read_u8_zext(buf, byte_offset) -> u32`
    /// - `ptr_read_u16_zext(buf, byte_offset) -> u32`
    /// - `f16_bits_to_f32(bits: u32) -> f32`
    /// - `f32_from_u32(u: u32) -> f32`
    /// - `f32_to_f16(x: f32) -> f16`  (M3.5 — narrowing OpFConvert, RNE)
    ///
    /// For the `ptr_read_*` variants, `buf_param_index` is the 0-based buffer-only
    /// binding slot used by the SPIR-V emitter to look up the SSBO variable id.
    /// For the conversion variants (`F16BitsToF32`, `F32FromU32`, `F32ToF16`),
    /// `buf_param_index` is `None`.
    Q4_0Builtin {
        op: crate::q4_0::Q4_0Builtin,
        args: Vec<HirExpr>,
        /// Buffer-parameter binding slot (0-based); `Some` for ptr_read_* builtins.
        buf_param_index: Option<u32>,
    },
    /// GLSL.std.450 extended-instruction builtin call (M3.2c).
    ///
    /// Covers the transcendental ext-inst builtins lowering to a single
    /// `OpExtInst %f32 %glsl450_set <opcode> %x`:
    /// - `exp(x: f32) -> f32`  (GLSL.std.450 Exp = 27)
    ///
    /// No `buf_param_index` (scalar arg, not a buffer). The GLSL.std.450
    /// `OpExtInstImport` is emitted ONCE per module (cached set-id in codegen).
    ExtInstBuiltin {
        op: crate::ext_inst::ExtInstBuiltin,
        args: Vec<HirExpr>,
    },
    /// Read one element from a workgroup-shared array: `tile[index]` (M3.2).
    ///
    /// `shared_id` is the 0-based index into `KernelBodyTyped.shared`.
    /// `index` must have type `U32` (no implicit coercion — anti-pattern #1).
    /// Result type is the shared array's element `ScalarTy`.
    SharedRead {
        /// Id of the shared array being read.
        shared_id: u32,
        /// Index expression — must be U32.
        index: Box<HirExpr>,
    },
    /// Read one element from a per-invocation local array: `hist[index]` (M3.20).
    ///
    /// `local_array_id` is the 0-based index into `KernelBodyTyped.local_arrays`.
    /// `index` must have type `U32` (no implicit coercion — anti-pattern #1).
    /// Result type is the local array's element `ScalarTy`.
    LocalArrayRead {
        /// Id of the local array being read.
        local_array_id: u32,
        /// Index expression — must be U32.
        index: Box<HirExpr>,
    },
}

/// Unary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// Arithmetic negation of a signed integer or float (`-x`).
    Neg,
    /// Logical NOT of a bool (`not x`).
    LogicalNot,
}

/// Binary arithmetic or comparison operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Eq,
    Neq,
    Lt,
    LtEq,
    Gt,
    GtEq,
}

/// Short-circuit logical operator (structured SPIR-V diamond lowering).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShortCircuitOp {
    /// `a and b` — evaluates RHS only if LHS is true.
    And,
    /// `a or b` — evaluates RHS only if LHS is false.
    Or,
}

/// Bitwise builtin operator (call syntax in source).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitwiseOp {
    Band,
    Bor,
    Bxor,
    Bnot,
    Shl,
    /// Arithmetic right shift (signed integers only; rejected by HIR for unsigned).
    Shr,
    /// Logical right shift (unsigned integers only; rejected by HIR for signed).
    Lshr,
}

/// A typed HIR statement.
///
/// M1.3 adds `If`, `ForRange`, `While`, `Break`, `Continue`.
#[derive(Debug, Clone)]
pub enum HirStmt {
    Let {
        binding: BindingId,
        init: HirExpr,
        span: Span,
    },
    Assign {
        binding: BindingId,
        value: HirExpr,
        span: Span,
    },
    /// `return;` — only void return is valid in M1.1/M1.2/M1.3.
    Return { span: Span },
    /// Write one element to a buffer parameter: `buf[index] = value`.
    ///
    /// `param_position` is the 0-based position in the kernel's param list.
    /// `buffer_binding` is the 0-based buffer-only binding slot index.
    /// `index` must have type `U32`.
    BufferWrite {
        param_position: u32,
        buffer_binding: u32,
        index: HirExpr,
        value: HirExpr,
        span: Span,
    },
    /// `if cond { then } [else ...]` — structured selection (M1.3).
    If(crate::control_flow::HirIf),
    /// `for i in range(start, end [, step]) { body }` — structured iteration (M1.3).
    ForRange(crate::control_flow::HirForRange),
    /// `while cond { body }` — structured iteration (M1.3).
    While(crate::control_flow::HirWhile),
    /// `break;` — targets the innermost enclosing loop's merge block.
    Break { span: Span },
    /// `continue;` — targets the innermost enclosing loop's continue block.
    Continue { span: Span },
    /// `workgroup_barrier();` — OpControlBarrier with Workgroup scope (M1.4).
    ///
    /// NOT a block terminator — subsequent statements continue in the same block.
    /// Barrier-in-divergent-control-flow warning is deferred to M1.5.
    Barrier {
        kind: crate::subgroup::BarrierKind,
        span: Span,
    },
    /// `coopmat_store(m, buf_or_shared, element_offset, stride);` (M2.1 / M3.2) or
    /// `coopmat_store_col(...)` (M4.2a, same shape, column-major layout).
    ///
    /// Lowers to `OpAccessChain` + `OpCooperativeMatrixStoreKHR`.
    /// Only valid as a statement (void return type).
    /// `matrix_binding` is the BindingId of the matrix variable being stored.
    /// `store_source` discriminates Buffer (SSBO) vs Shared (Workgroup) destination (M3.2).
    CoopMatStore {
        /// BindingId of the cooperative-matrix value to store.
        matrix_binding: BindingId,
        /// Store destination discriminator (M3.2): Buffer(slot) for SSBO, Shared(id) for workgroup array.
        store_source: crate::coopmat::CoopMatLoadSource,
        /// Element offset argument (must be U32).
        element_offset: HirExpr,
        /// Stride argument (must be U32).
        stride: HirExpr,
        /// SPIR-V layout operand (M4.2a): `RowMajor` for `coopmat_store` (the M2.1 default),
        /// `ColMajor` for `coopmat_store_col`. Orthogonal to typing — codegen-only discriminant.
        layout: crate::coopmat::CoopMatStoreLayout,
        span: Span,
    },
    /// Write one element to a workgroup-shared array: `tile[index] = value;` (M3.2).
    ///
    /// `shared_id` is the 0-based index into `KernelBodyTyped.shared`.
    /// `index` must have type `U32` (no implicit coercion).
    /// `value` must match the shared array's element type exactly.
    SharedWrite {
        /// Id of the shared array being written.
        shared_id: u32,
        /// Index expression — must be U32.
        index: HirExpr,
        /// Value to write — must match elem ScalarTy exactly.
        value: HirExpr,
        span: Span,
    },
    /// No-op marker preserving shared-array declaration order in the statement list (M3.2).
    ///
    /// The actual `OpVariable Workgroup` is emitted by `emit_shared_globals` from the
    /// `KernelBodyTyped.shared` table, NOT from this statement. This marker exists to
    /// preserve lexical ordering of declarations in diagnostic messages and preserves
    /// the correspondence between source order and emission order.
    SharedDeclMarker {
        /// Id of the declared shared array.
        id: crate::shared::SharedId,
        span: Span,
    },
    /// Write one element to a per-invocation local array: `hist[index] = value;` (M3.20).
    ///
    /// `local_array_id` is the 0-based index into `KernelBodyTyped.local_arrays`.
    /// `index` must have type `U32` (no implicit coercion).
    /// `value` must match the local array's element type exactly.
    LocalArrayWrite {
        /// Id of the local array being written.
        local_array_id: u32,
        /// Index expression — must be U32.
        index: HirExpr,
        /// Value to write — must match elem ScalarTy exactly.
        value: HirExpr,
        span: Span,
    },
    /// No-op marker preserving local-array declaration order in the statement list (M3.20).
    ///
    /// The actual `OpVariable Function` is emitted in the function entry-block prelude
    /// (`body.rs`) from the `KernelBodyTyped.local_arrays` table, NOT from this statement.
    /// This marker exists to preserve lexical ordering of declarations in diagnostic
    /// messages and preserves the correspondence between source order and emission order.
    /// Mirrors `SharedDeclMarker`.
    LocalArrayDeclMarker {
        /// Id of the declared local array.
        id: crate::local::LocalArrayId,
        span: Span,
    },
}

/// The typed body of a kernel: a binding table plus ordered statements.
///
/// M3.2 adds `shared`: the ordered list of workgroup-shared array declarations,
/// analogous to `bindings` for local variables and the buffer table in HIR params.
#[derive(Debug, Clone)]
pub struct KernelBodyTyped {
    pub bindings: Vec<Binding>,
    pub stmts: Vec<HirStmt>,
    /// Workgroup-shared array declarations in source order (M3.2).
    ///
    /// Each entry is a `SharedDecl` with a unique `SharedId`. The codegen uses this
    /// table to emit `OpVariable Workgroup` before the function body. The typecheck
    /// resolves `name[i]` references into `SharedRead`/`SharedWrite` HIR nodes
    /// using the ids from this table.
    pub shared: Vec<crate::shared::SharedDecl>,
    /// Per-invocation local array declarations in source order (M3.20).
    ///
    /// Each entry is a `LocalArrayDecl` with a unique `LocalArrayId`. The codegen uses
    /// this table to emit `OpVariable Function` in the function entry-block prelude. The
    /// typecheck resolves `name[i]` references into `LocalArrayRead`/`LocalArrayWrite`
    /// HIR nodes using the ids from this table.
    pub local_arrays: Vec<crate::local::LocalArrayDecl>,
}
