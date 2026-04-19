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
}

// ── Internal binding table ────────────────────────────────────────────────────

struct TypeChecker {
    bindings: Vec<Binding>,
    // Maps binding name -> index in `bindings` (for duplicate detection + lookup).
    binding_lookup: std::collections::HashMap<String, usize>,
    errors: Vec<TypecheckError>,
    next_id: u32,
}

impl TypeChecker {
    fn new() -> Self {
        Self {
            bindings: Vec::new(),
            binding_lookup: std::collections::HashMap::new(),
            errors: Vec::new(),
            next_id: 0,
        }
    }

    fn alloc_id(&mut self) -> BindingId {
        let id = BindingId(self.next_id);
        self.next_id += 1;
        id
    }

    fn find_binding(&self, name: &str) -> Option<(BindingId, ScalarTy, bool, Span)> {
        if let Some(&idx) = self.binding_lookup.get(name) {
            let b = &self.bindings[idx];
            Some((b.id, b.ty, b.is_mutable, b.span))
        } else {
            None
        }
    }

    fn register_binding(&mut self, name: &str, ty: ScalarTy, is_mutable: bool, span: Span) -> Option<BindingId> {
        if let Some(&existing_idx) = self.binding_lookup.get(name) {
            let orig_span = self.bindings[existing_idx].span;
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
        self.binding_lookup.insert(name.to_owned(), idx);
        Some(id)
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Typecheck a kernel body block.
///
/// Always returns a `KernelBodyTyped` (possibly incomplete on errors) plus any
/// `TypecheckError`s. This supports error-recovery: even with errors, downstream
/// code sees a partial HIR to collect further diagnostics.
pub fn typecheck_kernel_body(
    body: &past::Block,
) -> (KernelBodyTyped, Vec<TypecheckError>) {
    let mut tc = TypeChecker::new();
    let mut hir_stmts: Vec<HirStmt> = Vec::new();

    // ── Pass 1: Register all let bindings into the binding table ─────────────
    for spanned_stmt in &body.stmts {
        if let past::Stmt::Let { name, ty, is_mut, .. } = &spanned_stmt.node {
            let scalar_ty = match typeref_to_scalar(&ty.node) {
                Ok(t) => t,
                Err(detail) => {
                    tc.errors.push(TypecheckError::UnsupportedExprInM1_1 {
                        detail,
                        span: ty.span,
                    });
                    // Use a placeholder so future references resolve to something.
                    ScalarTy::I32
                }
            };
            // register_binding handles duplicate detection.
            tc.register_binding(&name.node, scalar_ty, *is_mut, name.span);
        }
    }

    // ── Pass 2: Typecheck expressions in each statement ───────────────────────
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
                        if !is_mutable {
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
            past::Stmt::Return(maybe_expr) => {
                if let Some(expr) = maybe_expr {
                    tc.errors.push(TypecheckError::UnsupportedExprInM1_1 {
                        detail: "return with value (kernels must return void in M1.1)",
                        span: expr.span,
                    });
                }
                hir_stmts.push(HirStmt::Return { span: spanned_stmt.span });
            }
        }
    }

    let body_typed = KernelBodyTyped {
        bindings: tc.bindings,
        stmts: hir_stmts,
    };

    (body_typed, tc.errors)
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
            match tc.find_binding(name) {
                None => {
                    tc.errors.push(TypecheckError::UnknownBinding {
                        name: name.clone(),
                        span,
                    });
                    // Placeholder with the expected type (or I32 if unconstrained).
                    let placeholder_ty = expected.unwrap_or(ScalarTy::I32);
                    Some(HirExpr {
                        kind: HirExprKind::BoolLit(false), // placeholder
                        ty: placeholder_ty,
                        span,
                    })
                }
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

        // ── Call (bitwise builtins) ────────────────────────────────────────────
        past::Expr::Call { name, args } => {
            check_call(tc, &name.node, name.span, args, span, expected)
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
            return typecheck_kernel_body(&kd.body.node);
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
}

