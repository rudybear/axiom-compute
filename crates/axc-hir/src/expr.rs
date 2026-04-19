//! Typed expression and statement IR for AXIOM-Compute HIR.
//!
//! Every `HirExpr` carries a resolved `ScalarTy` on every node — no type
//! placeholders, no inference. This is the invariant produced by the two-pass
//! typechecker in `typecheck.rs`.

use axc_lexer::Span;
use crate::ty::{ScalarTy, IntLiteralValue, FloatLiteralValue};

/// Opaque identifier for a local variable binding within a kernel body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BindingId(pub u32);

/// A typed local variable binding.
#[derive(Debug, Clone)]
pub struct Binding {
    pub id: BindingId,
    pub name: String,
    pub ty: ScalarTy,
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
    /// `return;` — only void return is valid in M1.1.
    Return { span: Span },
}

/// The typed body of a kernel: a binding table plus ordered statements.
#[derive(Debug, Clone)]
pub struct KernelBodyTyped {
    pub bindings: Vec<Binding>,
    pub stmts: Vec<HirStmt>,
}
