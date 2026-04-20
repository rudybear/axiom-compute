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
    /// Subgroup builtin call (M1.4).
    ///
    /// Covers all subgroup operations except `workgroup_barrier`, which is a
    /// statement (`HirStmt::Barrier`). See `crate::subgroup::SubgroupOp` for variants.
    SubgroupBuiltin {
        op: crate::subgroup::SubgroupOp,
        args: Vec<HirExpr>,
    },
    /// Cooperative-matrix builtin call (M2.1).
    ///
    /// Covers `coopmat_zero`, `coopmat_load`, and `coopmat_mul_add`.
    /// `coopmat_store` is a STATEMENT (`HirStmt::CoopMatStore`).
    ///
    /// `result_ty` carries the resolved matrix key.
    /// - `Zero` / `Load`: resolved from the let-binding's expected type.
    /// - `MulAdd`: determined by the c-argument type.
    ///
    /// `buf_param_index` is `Some(n)` for `Load` where `n` is the 0-based
    /// buffer-parameter index used for `OpAccessChain`. `None` for Zero/MulAdd.
    CoopMatBuiltin {
        op: crate::coopmat::CoopMatBuiltin,
        args: Vec<HirExpr>,
        result_ty: CoopMatKey,
        /// Buffer-parameter binding slot (0-based) used by Load to synthesize OpAccessChain.
        buf_param_index: Option<u32>,
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
    /// `coopmat_store(m, buf, element_offset, stride);` (M2.1).
    ///
    /// Lowers to `OpAccessChain` + `OpCooperativeMatrixStoreKHR`.
    /// Only valid as a statement (void return type).
    /// `matrix_binding` is the BindingId of the matrix variable being stored.
    /// `buf_param_index` is the 0-based buffer-parameter binding slot.
    CoopMatStore {
        /// BindingId of the cooperative-matrix value to store.
        matrix_binding: BindingId,
        /// 0-based buffer-parameter slot (for OpAccessChain).
        buf_param_index: u32,
        /// Element offset argument (must be U32).
        element_offset: HirExpr,
        /// Stride argument (must be U32).
        stride: HirExpr,
        span: Span,
    },
}

/// The typed body of a kernel: a binding table plus ordered statements.
#[derive(Debug, Clone)]
pub struct KernelBodyTyped {
    pub bindings: Vec<Binding>,
    pub stmts: Vec<HirStmt>,
}
