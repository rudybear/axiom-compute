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

use axc_lexer::Span;
use axc_parser::ast as past;
use crate::expr::{
    Binding, BindingId, HirExpr, HirExprKind, HirStmt, KernelBodyTyped,
    BinOp, UnaryOp, ShortCircuitOp, BitwiseOp,
};
use crate::ty::{ScalarTy, fit_int_literal, fit_float_literal};
use crate::param::{KernelParam, Ty as ParamTy};
use crate::buffer::BufferAccess;
use crate::control_flow::{HirIf, HirElse, HirForRange, HirWhile, ForStep};
use crate::loop_ctx::{HirLoopStack, ScopeStack};

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
}

// ── Internal binding table ────────────────────────────────────────────────────

struct TypeChecker<'p> {
    bindings: Vec<Binding>,
    errors: Vec<TypecheckError>,
    next_id: u32,
    /// Kernel parameters (read-only; buffer params cannot be assigned to).
    params: &'p [KernelParam],
    /// Loop context stack for break/continue validation and induction-var detection.
    loop_stack: HirLoopStack,
    /// Scoped name-resolution: each block pushes a frame, pops on exit.
    scope_stack: ScopeStack,
}

impl<'p> TypeChecker<'p> {
    fn new(params: &'p [KernelParam]) -> Self {
        let mut tc = Self {
            bindings: Vec::new(),
            errors: Vec::new(),
            next_id: 0,
            params,
            loop_stack: HirLoopStack::new(),
            scope_stack: ScopeStack::new(),
        };
        // Push the top-level scope frame (pops at end of typecheck_kernel_body).
        tc.scope_stack.push_frame();
        tc
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
    fn find_binding(&self, name: &str) -> Option<(BindingId, ScalarTy, bool, Span)> {
        if let Some(idx) = self.scope_stack.get(name) {
            let b = &self.bindings[idx];
            Some((b.id, b.ty, b.is_mutable, b.span))
        } else {
            None
        }
    }

    /// Register a new binding in the innermost scope frame.
    ///
    /// Duplicate detection is within the SAME scope frame only.
    /// Shadowing across frames (e.g. nested for loops with same induction var name) is allowed.
    fn register_binding(&mut self, name: &str, ty: ScalarTy, is_mutable: bool, span: Span) -> Option<BindingId> {
        // Check for duplicate in the CURRENT frame only (not outer scopes — shadowing is OK).
        // Using get_in_current_frame ensures that nested scopes (e.g. nested for loops with
        // the same induction variable name) do not falsely trigger RedeclaredBinding.
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

// ── Public entry point ────────────────────────────────────────────────────────

/// Typecheck a kernel body block.
///
/// `params` is the list of kernel parameters — buffer params cannot be used
/// as scalar values and cannot be assigned to; scalar params appear as read-only
/// bindings in expressions.
///
/// Always returns a `KernelBodyTyped` (possibly incomplete on errors) plus any
/// `TypecheckError`s. This supports error-recovery: even with errors, downstream
/// code sees a partial HIR to collect further diagnostics.
pub fn typecheck_kernel_body(
    body: &past::Block,
    params: &[KernelParam],
) -> (KernelBodyTyped, Vec<TypecheckError>) {
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
                // Lookup the binding registered in pass 1.
                let maybe_binding = tc.find_binding(&name.node);
                let expected_ty = maybe_binding.map(|(_, t, _, _)| t);

                let hir_init = check_expr(&mut tc, &init.node, init.span, expected_ty);

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
                            let hir_value = check_expr(&mut tc, &value.node, value.span, Some(binding_ty));
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
        }
    }

    // Pop the top-level scope frame opened in TypeChecker::new.
    tc.scope_stack.pop_frame();

    let body_typed = KernelBodyTyped {
        bindings: tc.bindings,
        stmts: hir_stmts,
    };

    (body_typed, tc.errors)
}

// ── Pre-registration of let bindings (top-level only) ────────────────────────

/// Pre-register let bindings at the TOP level of a block (not nested blocks).
///
/// This is the M1.1/M1.2 two-pass approach for flat kernel bodies. Nested blocks
/// (if/for/while bodies) use single-pass with scope frames and DON'T pre-register.
fn pre_register_lets_in_block(block: &past::Block, tc: &mut TypeChecker<'_>) {
    for spanned_stmt in &block.stmts {
        if let past::Stmt::Let { name, ty, is_mut, .. } = &spanned_stmt.node {
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
            tc.register_binding(&name.node, scalar_ty, *is_mut, name.span);
        }
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

    let cond_hir = check_expr(tc, &cond.node, cond.span, Some(ScalarTy::Bool))?;
    if cond_hir.ty != ScalarTy::Bool {
        tc.errors.push(TypecheckError::NonBoolCondition {
            position: "if",
            got: cond_hir.ty.display_name(),
            span: cond.span,
        });
        return None;
    }

    let then_stmts = typecheck_nested_block(tc, then_block);

    let hir_else: Option<Box<HirElse>> = match else_arm {
        None => None,
        Some(past::ElseArm::Block(block)) => {
            let else_stmts = typecheck_nested_block(tc, block);
            Some(Box::new(HirElse::Block(else_stmts)))
        }
        Some(past::ElseArm::If(inner_spanned)) => {
            // inner_spanned.node must be Stmt::If
            if let past::Stmt::If { cond: ic, then_block: itb, else_arm: iea } = &inner_spanned.node {
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

    // Push a loop frame (no induction variable for while loops)
    tc.loop_stack.push(None);
    let body_stmts = typecheck_nested_block(tc, body);
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
                            let hir_value = check_expr(tc, &value.node, value.span, Some(binding_ty));
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
        past::TypeRef::F32  => Ok(ScalarTy::F32),
        past::TypeRef::F64  => Ok(ScalarTy::F64),
        past::TypeRef::Bool => Ok(ScalarTy::Bool),
        past::TypeRef::Void => Err("void is not a valid scalar type for let bindings"),
        past::TypeRef::Buffer(_)
        | past::TypeRef::ReadonlyBuffer(_)
        | past::TypeRef::WriteonlyBuffer(_) => {
            Err("buffer types are not valid for let bindings; use as kernel parameters only")
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
                Some((bid, ty, _, _)) => {
                    if let Some(exp) = expected {
                        if exp != ty {
                            tc.errors.push(TypecheckError::TypeMismatch {
                                expected: exp.display_name(),
                                got: ty.display_name(),
                                span,
                            });
                        }
                    }
                    Some(HirExpr {
                        kind: HirExprKind::LocalRead(bid),
                        ty,
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

        // ── Call (bitwise builtins + gid) ────────────────────────────────────
        past::Expr::Call { name, args } => {
            if name.node == "gid" {
                check_gid_call(tc, args, span)
            } else {
                check_call(tc, &name.node, name.span, args, span, expected)
            }
        }

        // ── Buffer index read: name[index] ────────────────────────────────────
        past::Expr::Index { base, index } => {
            // The M1.2 parser only produces Index with an Ident base (postfix `name[expr]`).
            // Multi-dimensional chained indexing (e.g. buf[i][j]) is not parseable in M1.2.
            match &base.node {
                past::Expr::Ident(name) => {
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

// ── Bitwise builtin calls ─────────────────────────────────────────────────────

fn check_call(
    tc: &mut TypeChecker,
    name: &str,
    name_span: Span,
    args: &[axc_lexer::Spanned<past::Expr>],
    call_span: Span,
    _expected: Option<ScalarTy>,
) -> Option<HirExpr> {
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
            return typecheck_kernel_body(&kd.body.node, &[]);
        }
        (KernelBodyTyped { bindings: Vec::new(), stmts: Vec::new() }, Vec::new())
    }

    // 1. tc_let_i32_literal_happy
    #[test]
    fn tc_let_i32_literal_happy() {
        let (body, errors) = tc_body("let x: i32 = 42; return;");
        assert!(errors.is_empty(), "errors: {errors:?}");
        assert_eq!(body.bindings.len(), 1);
        assert_eq!(body.bindings[0].ty, ScalarTy::I32);
    }

    // 2. tc_let_u64_literal_happy
    #[test]
    fn tc_let_u64_literal_happy() {
        let (body, errors) = tc_body("let x: u64 = 42; return;");
        assert!(errors.is_empty(), "errors: {errors:?}");
        assert_eq!(body.bindings[0].ty, ScalarTy::U64);
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
        assert_eq!(body.bindings[0].ty, ScalarTy::Bool);
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
        assert_eq!(body.bindings[0].ty, ScalarTy::Bool);
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
            return typecheck_kernel_body(&kd.body.node, &params);
        }
        (KernelBodyTyped { bindings: Vec::new(), stmts: Vec::new() }, Vec::new())
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
}

