//! Typed HIR data structures for structured control flow (M1.3).
//!
//! These types are populated by the typechecker (`typecheck.rs`) and consumed
//! by the SPIR-V body emitter (`body.rs`).
//!
//! Design decisions:
//! - `HirIf.cond` is GUARANTEED by HIR gating to NOT be `HirExprKind::ShortCircuit`.
//!   Any `Expr::ShortCircuit` in an if/while header position produces
//!   `UnsupportedShortCircuitInHeader` at typecheck time, BEFORE this struct is built.
//! - `ForStep.value` is a compile-time u32 constant, not an expression.
//!   Codegen emits `OpConstant u32 <value>` directly.
//! - `Break` and `Continue` carry no loop-id — resolution happens at codegen via
//!   `BodyEmitter.loop_stack`.

use axc_lexer::Span;
use crate::expr::{HirExpr, HirStmt, BindingId};

/// A typed `if [else if]* [else]` chain.
///
/// `cond` is guaranteed by HIR to be a bool-typed non-short-circuit expression.
#[derive(Debug, Clone)]
pub struct HirIf {
    pub cond: HirExpr,
    pub then_block: Vec<HirStmt>,
    /// `None` = no else arm; `Some(HirElse::Block(...))` = else block;
    /// `Some(HirElse::If(...))` = else-if (recursive).
    pub else_arm: Option<Box<HirElse>>,
    pub span: Span,
}

/// The else arm of an if: either a plain block or an else-if chain.
#[derive(Debug, Clone)]
pub enum HirElse {
    Block(Vec<HirStmt>),
    If(HirIf),
}

/// A typed `for i in range(start, end [, step]) { body }` loop.
///
/// `induction` is a BindingId allocated in `KernelBodyTyped.bindings`.
/// The prelude in `body.rs` allocates an `OpVariable Function u32` for it.
#[derive(Debug, Clone)]
pub struct HirForRange {
    /// The induction variable's binding id.
    pub induction: BindingId,
    /// Loop start bound (must be U32-typed).
    pub start: HirExpr,
    /// Loop end bound (must be U32-typed; exclusive: runs while `i < end`).
    pub end: HirExpr,
    /// Compile-time step. Defaults to 1 when no explicit step was written.
    pub step: ForStep,
    /// Loop body statements.
    pub body: Vec<HirStmt>,
    pub span: Span,
}

/// A typed `while cond { body }` loop.
///
/// `cond` is guaranteed by HIR to be a bool-typed non-short-circuit expression.
#[derive(Debug, Clone)]
pub struct HirWhile {
    pub cond: HirExpr,
    pub body: Vec<HirStmt>,
    pub span: Span,
}

/// Compile-time resolved for-loop step.
///
/// Stored as a `u32` constant so that codegen can emit `OpConstant u32 <value>`
/// directly. Variable step is rejected at typecheck with `ForStepNotConstant`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForStep {
    pub value: u32,
}

impl ForStep {
    /// The default step used when no explicit step is written.
    pub const ONE: ForStep = ForStep { value: 1 };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn for_step_one_is_one() {
        assert_eq!(ForStep::ONE.value, 1);
    }

    #[test]
    fn for_step_default_matches_spec() {
        // The spec says default step == 1.
        let s = ForStep { value: 1 };
        assert_eq!(s, ForStep::ONE);
    }

    #[test]
    fn for_step_two_is_distinct() {
        let s = ForStep { value: 2 };
        assert_ne!(s, ForStep::ONE);
    }
}
