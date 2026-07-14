//! Lowering from `axc_parser::ast::Module` to `hir::Module`.
//!
//! Maps annotation argument lists into structured HIR types.
//! Overflow-safe: u32 dimensions are range-checked before conversion.
//! `@complexity` is mapped to `ComplexityForm` enum values (never a String).

use axc_lexer::Span;
use axc_parser::ast::{Module as AstModule, Item, AnnotationArg, Stmt, TypeRef, ScalarTypeRef};
use crate::hir::{
    Module as HirModule, Kernel, KernelId, KernelAnnotations, WorkgroupDims,
    KernelBody, ComplexityForm, ComplexityVar, PreconditionTrivial, StrategyHoles,
    DebugCheck, DebugCheckKind, DebugCompareOp, DebugOperand,
};
use crate::validate::{HirError, HirWarning};
use crate::typecheck::typecheck_kernel_body;
use crate::ty::ScalarTy;
use crate::buffer::{BufferAccess, BufferTy};
use crate::param::{KernelParam, Ty as ParamTy, ParamBindingPlan, compute_binding_plan};
use crate::coopmat::{CoopMatrixShape, CoopMatScopeHir, CoopMatUse};
use crate::expr::{KernelBodyTyped, HirStmt, HirExprKind};
use crate::control_flow::HirElse;

/// Lower an AST module to HIR, producing structured errors and warnings.
///
/// Always runs to completion — never short-circuits on the first error.
pub fn lower_module(ast: &AstModule) -> (HirModule, Vec<HirError>, Vec<HirWarning>) {
    let mut kernels: Vec<Kernel> = Vec::new();
    let mut errors: Vec<HirError> = Vec::new();
    let mut warnings: Vec<HirWarning> = Vec::new();
    let mut next_id: u32 = 0;

    for item in &ast.items {
        match &item.node {
            Item::Kernel(kd) => {
                // Collect annotation data for this kernel
                let mut workgroup_opt: Option<WorkgroupDims> = None;
                let mut workgroup_span: Span = item.span;
                let mut intent: Option<String> = None;
                let mut complexity: Option<ComplexityForm> = None;
                let mut preconditions: Vec<PreconditionTrivial> = Vec::new();
                let mut subgroup_uniform: bool = false;
                let mut cooperative_matrix: bool = false;
                let mut seen_workgroup: bool = false;
                let mut seen_kernel: bool = false;
                // M2.3: strategy holes
                let mut strategy_holes_opt: Option<StrategyHoles> = None;
                // M3.17 (FG.4): raw Call-form @precondition/@postcondition bodies.
                // Resolution is deferred until after `lower_params` (below) because
                // operand resolution needs the kernel's scalar parameter table.
                let mut precondition_calls: Vec<(String, Vec<axc_lexer::Spanned<AnnotationArg>>, Span)> = Vec::new();
                let mut postcondition_calls: Vec<(String, Vec<axc_lexer::Spanned<AnnotationArg>>, Span)> = Vec::new();

                for ann in &kd.annotations {
                    let name: &str = &ann.node.name.node;
                    match name {
                        "kernel" => {
                            if seen_kernel {
                                errors.push(HirError::DuplicateAnnotation {
                                    name: "kernel".into(),
                                    kernel: kd.name.node.clone(),
                                    span: ann.span,
                                });
                            }
                            seen_kernel = true;
                        }
                        "workgroup" => {
                            if seen_workgroup {
                                errors.push(HirError::DuplicateAnnotation {
                                    name: "workgroup".into(),
                                    kernel: kd.name.node.clone(),
                                    span: ann.span,
                                });
                                continue;
                            }
                            seen_workgroup = true;
                            workgroup_span = ann.span;
                            let args: &[_] = &ann.node.args;
                            if args.len() != 3 {
                                errors.push(HirError::BadWorkgroupArity {
                                    got: args.len(),
                                    span: ann.span,
                                });
                            } else {
                                // M2.3: workgroup args may be HoleRefs; defer validation
                                // (the HoleRef → integer resolution happens in axc-optimize).
                                // Only attempt to extract as integer if they are Int args.
                                let x_res: Result<u32, HirError> = extract_workgroup_dim(&args[0].node, ann.span);
                                let y_res: Result<u32, HirError> = extract_workgroup_dim(&args[1].node, ann.span);
                                let z_res: Result<u32, HirError> = extract_workgroup_dim(&args[2].node, ann.span);
                                match (x_res, y_res, z_res) {
                                    (Ok(x), Ok(y), Ok(z)) => {
                                        workgroup_opt = Some(WorkgroupDims { x, y, z });
                                    }
                                    (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => {
                                        errors.push(e);
                                    }
                                }
                            }
                        }
                        "intent" => {
                            if ann.node.args.len() == 1 {
                                if let AnnotationArg::String(s) = &ann.node.args[0].node {
                                    intent = Some(s.clone());
                                }
                            }
                        }
                        "complexity" => {
                            if ann.node.args.len() == 1 {
                                match lower_complexity(&ann.node.args[0].node, ann.span) {
                                    Ok(form) => complexity = Some(form),
                                    Err(e) => errors.push(e),
                                }
                            } else {
                                errors.push(HirError::UnsupportedComplexityInM0 { span: ann.span });
                            }
                        }
                        "precondition" => {
                            if ann.node.args.len() == 1 {
                                match &ann.node.args[0].node {
                                    AnnotationArg::Bool(true) => {
                                        preconditions.push(PreconditionTrivial { span: ann.span });
                                    }
                                    // M3.17 (FG.4): a whitelisted Call-form predicate
                                    // (`gt(n, 0)`, etc.) — resolution deferred until params
                                    // are known (see `lower_debug_checks` below).
                                    AnnotationArg::Call { name: call_name, args: call_args } => {
                                        precondition_calls.push((call_name.clone(), call_args.clone(), ann.span));
                                    }
                                    other => {
                                        errors.push(HirError::UnsupportedPrecondition {
                                            detail: format!("{other:?}"),
                                            span: ann.span,
                                        });
                                    }
                                }
                            }
                        }
                        // M3.17 (FG.4): @postcondition follows the same Call-form DSL as
                        // @precondition, plus `is_finite`. `true` is accepted for symmetry
                        // (back-compat parity) and lowers to nothing, like @precondition(true).
                        "postcondition" => {
                            if ann.node.args.len() == 1 {
                                match &ann.node.args[0].node {
                                    AnnotationArg::Bool(true) => {}
                                    AnnotationArg::Call { name: call_name, args: call_args } => {
                                        postcondition_calls.push((call_name.clone(), call_args.clone(), ann.span));
                                    }
                                    other => {
                                        errors.push(HirError::UnsupportedDebugPredicate {
                                            name: format!("{other:?}"),
                                            span: ann.span,
                                        });
                                    }
                                }
                            }
                        }
                        "subgroup_uniform" => {
                            subgroup_uniform = true;
                        }
                        "cooperative_matrix" => {
                            cooperative_matrix = true;
                        }
                        // M2.5: @strict is a documentation/lint annotation that
                        // enforces @intent + @complexity + @precondition are all present.
                        // The HIR lowerer accepts and ignores it (validation of the
                        // strict discipline is done by the QA agent, not the compiler).
                        "strict" => {}
                        // M2.3: @strategy { k: ?[v1, v2, ...] } — strategy holes.
                        "strategy" => {
                            let holes: StrategyHoles = lower_strategy_annotation(
                                &ann.node.args,
                                ann.span,
                                &mut errors,
                            );
                            if ann.node.args.is_empty() {
                                warnings.push(HirWarning::EmptyStrategyBlock { span: ann.span });
                            }
                            strategy_holes_opt = Some(holes);
                        }
                        other => {
                            errors.push(HirError::UnknownAnnotationInM0 {
                                name: other.to_owned(),
                                span: ann.span,
                            });
                        }
                    }
                }

                // M2.3: Validate HoleRefs across all annotations on this kernel.
                validate_hole_refs(
                    &kd.annotations,
                    strategy_holes_opt.as_ref(),
                    &kd.params,
                    item.span,
                    &mut errors,
                    &mut warnings,
                );

                // M2.3: If @workgroup has HoleRefs, we cannot determine actual dims yet.
                // Use placeholder dims (1,1,1) so the kernel can still be placed into HIR.
                // The enumerator will produce concrete workgroup dims after substitution.
                let wg_has_holes: bool = kd.annotations.iter().any(|a| {
                    a.node.name.node == "workgroup"
                        && a.node.args.iter().any(|arg| matches!(&arg.node, AnnotationArg::HoleRef(_)))
                });

                // Check that @workgroup was present
                let wg: WorkgroupDims = if wg_has_holes {
                    // Placeholder: actual dims come from the enumerator-resolved source.
                    workgroup_opt.unwrap_or(WorkgroupDims { x: 1, y: 1, z: 1 })
                } else {
                    match workgroup_opt {
                        Some(w) => w,
                        None => {
                            errors.push(HirError::MissingWorkgroup {
                                name: kd.name.node.clone(),
                                span: item.span,
                            });
                            // Use a placeholder so we can still emit the kernel into HIR
                            WorkgroupDims { x: 1, y: 1, z: 1 }
                        }
                    }
                };

                // Check return type
                if kd.return_type.node != axc_parser::ast::TypeRef::Void {
                    errors.push(HirError::BadKernelReturnType {
                        got: format!("{:?}", kd.return_type.node),
                        span: kd.return_type.span,
                    });
                }

                // Validate workgroup dimensions (zero check) — skip if holes are present
                // since placeholder dims would spuriously trigger the zero-dim error.
                if !wg_has_holes {
                    let _ = validate_workgroup_dims(&wg, workgroup_span, &kd.name.node, &mut errors, &mut warnings);
                }

                // Lower kernel parameters.
                let (params, binding_plan) = lower_params(
                    &kd.params,
                    item.span,
                    &mut errors,
                );

                // M3.17 (FG.4): resolve the raw @precondition/@postcondition Call bodies
                // now that the kernel's scalar parameter table is known.
                let debug_checks: Vec<DebugCheck> = lower_debug_checks(
                    &precondition_calls,
                    &postcondition_calls,
                    &params,
                    &mut errors,
                    &mut warnings,
                );

                // Determine body: Empty if trivial (only void return), Typed otherwise.
                let body = lower_kernel_body(&kd.body.node, &params, &mut errors, &mut warnings);

                // M3.1: Derive coopmat shape from the typechecked body.
                // Only attempted when the @cooperative_matrix flag is set.
                let coop_matrix: Option<CoopMatrixShape> = if cooperative_matrix {
                    match &body {
                        KernelBody::Typed(typed_body) => {
                            let shape = derive_coopmat_shape(typed_body);
                            if shape.is_none() {
                                warnings.push(HirWarning::CoopMatFlagWithoutOps {
                                    kernel_name: kd.name.node.clone(),
                                });
                            }
                            shape
                        }
                        KernelBody::Empty => {
                            warnings.push(HirWarning::CoopMatFlagWithoutOps {
                                kernel_name: kd.name.node.clone(),
                            });
                            None
                        }
                    }
                } else {
                    None
                };

                let annotations: KernelAnnotations = KernelAnnotations {
                    workgroup: wg,
                    intent,
                    complexity,
                    preconditions,
                    subgroup_uniform,
                    cooperative_matrix,
                    coop_matrix,
                    strategy: strategy_holes_opt,
                    debug_checks,
                };

                kernels.push(Kernel {
                    id: KernelId(next_id),
                    name: kd.name.node.clone(),
                    annotations,
                    params,
                    binding_plan,
                    body,
                    span: item.span,
                });
                next_id += 1;
            }
        }
    }

    (HirModule { kernels }, errors, warnings)
}

/// Determine whether a block is an empty kernel body (only `return;` or completely
/// empty), and lower it to the appropriate `KernelBody` variant.
///
/// Uses `KernelBody::Empty` if:
/// - The block has no statements AND there are no parameters, OR
/// - The block has exactly one statement that is `Stmt::Return(None)` AND no parameters.
///
/// This preserves the M0 empty-kernel bit-exact SPIR-V output (AT-103).
fn lower_kernel_body(
    block: &axc_parser::ast::Block,
    params: &[KernelParam],
    errors: &mut Vec<HirError>,
    warnings: &mut Vec<crate::validate::HirWarning>,
) -> KernelBody {
    let stmts = &block.stmts;

    let is_trivial_stmts = stmts.is_empty()
        || (stmts.len() == 1 && matches!(stmts[0].node, Stmt::Return(None)));

    // Only use Empty path if there are no params AND the body is trivial.
    // With params, even a body that is just `return;` must go through Typed
    // so that the codegen knows about the param bindings.
    let is_empty_body = is_trivial_stmts && params.is_empty();

    if is_empty_body {
        return KernelBody::Empty;
    }

    // Non-trivial body or body with params — run the typechecker.
    let (typed_body, tc_errors, tc_warns) = typecheck_kernel_body(block, params);
    for e in tc_errors {
        errors.push(HirError::Typecheck(e));
    }
    for w in tc_warns {
        warnings.push(w);
    }
    KernelBody::Typed(typed_body)
}

/// Lower AST kernel parameters to HIR `KernelParam`s and compute the binding plan.
///
/// Returns `(params, binding_plan)`. On error, a placeholder empty plan is returned
/// so codegen can continue and collect further diagnostics.
fn lower_params(
    ast_params: &[axc_lexer::Spanned<axc_parser::ast::Param>],
    kernel_span: Span,
    errors: &mut Vec<HirError>,
) -> (Vec<KernelParam>, ParamBindingPlan) {
    let mut params: Vec<KernelParam> = Vec::new();

    for (pos, spanned) in ast_params.iter().enumerate() {
        let ast_param = &spanned.node;
        let name: &str = &ast_param.name.node;
        let span: Span = spanned.span;

        let ty: ParamTy = match lower_type_ref(&ast_param.ty.node, span, name, errors) {
            Some(t) => t,
            None => continue, // error already pushed
        };

        params.push(KernelParam {
            name: name.to_owned(),
            ty,
            position: pos as u32,
            span,
        });
    }

    match compute_binding_plan(&params, kernel_span) {
        Ok(plan) => (params, plan),
        Err(e) => {
            errors.push(HirError::BindingPlan(e));
            // Return params but empty plan so codegen doesn't crash.
            (params, ParamBindingPlan {
                buffers: Vec::new(),
                scalars: Vec::new(),
                push_constant_total_bytes: 0,
            })
        }
    }
}

// ── M3.17 (FG.4): @precondition/@postcondition Call-form predicate DSL ─────────

/// Classification of one annotation-arg operand inside a debug-check Call.
enum RawDebugOperand {
    Lit(i64),
    ScalarRef(String),
    /// `elem(buf)` — buffer-element operand. Hard-rejected in a precondition,
    /// deferred (`PostconditionNotLowerable`) in a postcondition (defer-list §9.1).
    BufElem(String),
    Invalid,
}

/// Classify a single annotation-arg node as a debug-check operand shape.
fn classify_debug_operand(arg: &AnnotationArg) -> RawDebugOperand {
    match arg {
        AnnotationArg::Int(v) => RawDebugOperand::Lit(*v),
        AnnotationArg::Ident(name) => RawDebugOperand::ScalarRef(name.clone()),
        AnnotationArg::Call { name, args } if name == "elem" && args.len() == 1 => {
            match &args[0].node {
                AnnotationArg::Ident(buf) => RawDebugOperand::BufElem(buf.clone()),
                _ => RawDebugOperand::Invalid,
            }
        }
        _ => RawDebugOperand::Invalid,
    }
}

/// Resolve a scalar param reference to its `ScalarTy`, restricted to u32/i32/f32
/// (§3 of the spec). `None` if not a scalar param, or an out-of-set type.
fn resolve_scalar_param_ty(params: &[KernelParam], name: &str) -> Option<ScalarTy> {
    params.iter().find(|p| p.name == name).and_then(|p| match &p.ty {
        ParamTy::Scalar(t @ (ScalarTy::U32 | ScalarTy::I32 | ScalarTy::F32)) => Some(*t),
        _ => None,
    })
}

/// Lower a classified operand to the codegen-ready `DebugOperand`. `None` for
/// `BufElem`/`Invalid` (callers must reject/defer those before calling this).
fn to_debug_operand(raw: &RawDebugOperand, params: &[KernelParam]) -> Option<DebugOperand> {
    match raw {
        RawDebugOperand::Lit(v) => Some(DebugOperand::Lit(*v)),
        RawDebugOperand::ScalarRef(name) => {
            let position = params.iter().find(|p| p.name == *name)?.position;
            Some(DebugOperand::ScalarRef { name: name.clone(), position })
        }
        RawDebugOperand::BufElem(_) | RawDebugOperand::Invalid => None,
    }
}

/// Human-readable rendering of a raw operand for `DebugCheck::text`.
fn render_debug_operand(raw: &RawDebugOperand) -> String {
    match raw {
        RawDebugOperand::Lit(v) => v.to_string(),
        RawDebugOperand::ScalarRef(n) => n.clone(),
        RawDebugOperand::BufElem(b) => format!("elem({b})"),
        RawDebugOperand::Invalid => "?".to_string(),
    }
}

/// Lower every collected raw `@precondition`/`@postcondition` Call body into
/// fully-resolved `DebugCheck`s (dense 0-based bit index per kind). Deferred until
/// after `lower_params` — operand resolution needs the scalar parameter table.
fn lower_debug_checks(
    precondition_calls: &[(String, Vec<axc_lexer::Spanned<AnnotationArg>>, Span)],
    postcondition_calls: &[(String, Vec<axc_lexer::Spanned<AnnotationArg>>, Span)],
    params: &[KernelParam],
    errors: &mut Vec<HirError>,
    warnings: &mut Vec<HirWarning>,
) -> Vec<DebugCheck> {
    let mut out: Vec<DebugCheck> = Vec::new();

    if precondition_calls.len() > 32 {
        errors.push(HirError::TooManyDebugConditions {
            kind: DebugCheckKind::Pre,
            got: precondition_calls.len(),
            span: precondition_calls.first().map(|c| c.2).unwrap_or_default(),
        });
    } else {
        let mut bit: u32 = 0;
        for (name, args, span) in precondition_calls {
            if let Some(check) = lower_one_debug_predicate(
                DebugCheckKind::Pre, name, args, *span, params, bit, errors, warnings,
            ) {
                bit += 1;
                out.push(check);
            }
        }
    }

    if postcondition_calls.len() > 32 {
        errors.push(HirError::TooManyDebugConditions {
            kind: DebugCheckKind::Post,
            got: postcondition_calls.len(),
            span: postcondition_calls.first().map(|c| c.2).unwrap_or_default(),
        });
    } else {
        let mut bit: u32 = 0;
        for (name, args, span) in postcondition_calls {
            if let Some(check) = lower_one_debug_predicate(
                DebugCheckKind::Post, name, args, *span, params, bit, errors, warnings,
            ) {
                bit += 1;
                out.push(check);
            }
        }
    }

    out
}

/// Lower one `@precondition`/`@postcondition` Call body to a `DebugCheck`.
///
/// Returns `None` when the predicate was rejected (an `HirError` was pushed) or is
/// a not-lowerable buffer-operand postcondition (an `HirWarning` was pushed) — in
/// neither case does the caller advance its bit counter.
#[allow(clippy::too_many_arguments)]
fn lower_one_debug_predicate(
    kind: DebugCheckKind,
    name: &str,
    args: &[axc_lexer::Spanned<AnnotationArg>],
    span: Span,
    params: &[KernelParam],
    bit: u32,
    errors: &mut Vec<HirError>,
    warnings: &mut Vec<HirWarning>,
) -> Option<DebugCheck> {
    let op: DebugCompareOp = match name {
        "gt" => DebugCompareOp::Gt,
        "ge" => DebugCompareOp::Ge,
        "lt" => DebugCompareOp::Lt,
        "le" => DebugCompareOp::Le,
        "eq" => DebugCompareOp::Eq,
        "ne" => DebugCompareOp::Ne,
        "is_finite" if kind == DebugCheckKind::Post => {
            // is_finite is buffer-only in v1 (§3 DSL table) — always deferred, never
            // hard-rejected, as long as its one argument is a well-formed elem(buf).
            if args.len() == 1 {
                if let RawDebugOperand::BufElem(buf) = classify_debug_operand(&args[0].node) {
                    warnings.push(HirWarning::PostconditionNotLowerable {
                        buf,
                        reason: "is_finite(elem(buf)) buffer postconditions are deferred in M3.17 v1 (defer-list §9.1)".to_owned(),
                        span,
                    });
                    return None;
                }
            }
            errors.push(HirError::UnsupportedDebugPredicate { name: name.to_owned(), span });
            return None;
        }
        _ => {
            errors.push(HirError::UnsupportedDebugPredicate { name: name.to_owned(), span });
            return None;
        }
    };

    if args.len() != 2 {
        errors.push(HirError::UnsupportedDebugPredicate { name: name.to_owned(), span });
        return None;
    }

    let lhs_raw = classify_debug_operand(&args[0].node);
    let rhs_raw = classify_debug_operand(&args[1].node);

    // A buffer-element operand: hard error in a precondition (thread-uniform-only),
    // deferred warning in a postcondition (v1 never lowers buffer postconditions).
    for raw in [&lhs_raw, &rhs_raw] {
        if let RawDebugOperand::BufElem(buf) = raw {
            return match kind {
                DebugCheckKind::Pre => {
                    errors.push(HirError::PreconditionOperandNotScalar { span });
                    None
                }
                DebugCheckKind::Post => {
                    warnings.push(HirWarning::PostconditionNotLowerable {
                        buf: buf.clone(),
                        reason: "elem(buf) buffer postconditions are deferred in M3.17 v1 (defer-list §9.1)".to_owned(),
                        span,
                    });
                    None
                }
            };
        }
    }

    if matches!(lhs_raw, RawDebugOperand::Invalid) || matches!(rhs_raw, RawDebugOperand::Invalid) {
        errors.push(HirError::UnsupportedDebugPredicate { name: name.to_owned(), span });
        return None;
    }

    // Resolve the shared comparison type: at least one operand must be a ScalarRef
    // naming a u32/i32/f32 param. Two ScalarRefs must agree; two Lits are ambiguous
    // (v1 infers no default numeric type) and rejected.
    let resolved_ty: Option<ScalarTy> = match (&lhs_raw, &rhs_raw) {
        (RawDebugOperand::ScalarRef(n), RawDebugOperand::ScalarRef(m)) => {
            match (resolve_scalar_param_ty(params, n), resolve_scalar_param_ty(params, m)) {
                (Some(a), Some(b)) if a == b => Some(a),
                _ => None,
            }
        }
        (RawDebugOperand::ScalarRef(n), RawDebugOperand::Lit(_))
        | (RawDebugOperand::Lit(_), RawDebugOperand::ScalarRef(n)) => resolve_scalar_param_ty(params, n),
        _ => None,
    };

    let ty = match resolved_ty {
        Some(t) => t,
        None => {
            errors.push(HirError::UnsupportedDebugPredicate { name: name.to_owned(), span });
            return None;
        }
    };

    let (Some(lhs), Some(rhs)) = (to_debug_operand(&lhs_raw, params), to_debug_operand(&rhs_raw, params)) else {
        errors.push(HirError::UnsupportedDebugPredicate { name: name.to_owned(), span });
        return None;
    };

    if bit >= 32 {
        errors.push(HirError::TooManyDebugConditions { kind, got: bit as usize + 1, span });
        return None;
    }

    let text = format!("{name}({}, {})", render_debug_operand(&lhs_raw), render_debug_operand(&rhs_raw));

    Some(DebugCheck { kind, op, ty, lhs, rhs, bit, text, span })
}

/// Convert an AST `TypeRef` to a HIR `ParamTy`.
///
/// Returns `None` and pushes an error for unsupported types (void, bool, etc.).
fn lower_type_ref(
    tr: &TypeRef,
    span: Span,
    param_name: &str,
    errors: &mut Vec<HirError>,
) -> Option<ParamTy> {
    match tr {
        TypeRef::I32 => Some(ParamTy::Scalar(ScalarTy::I32)),
        TypeRef::U32 => Some(ParamTy::Scalar(ScalarTy::U32)),
        TypeRef::I64 => Some(ParamTy::Scalar(ScalarTy::I64)),
        TypeRef::U64 => Some(ParamTy::Scalar(ScalarTy::U64)),
        TypeRef::F32 => Some(ParamTy::Scalar(ScalarTy::F32)),
        TypeRef::F64 => Some(ParamTy::Scalar(ScalarTy::F64)),
        TypeRef::Buffer(elem)         => Some(ParamTy::Buffer(BufferTy {
            elem: lower_scalar_type_ref(elem),
            access: BufferAccess::ReadWrite,
        })),
        TypeRef::ReadonlyBuffer(elem) => Some(ParamTy::Buffer(BufferTy {
            elem: lower_scalar_type_ref(elem),
            access: BufferAccess::ReadOnly,
        })),
        TypeRef::WriteonlyBuffer(elem) => Some(ParamTy::Buffer(BufferTy {
            elem: lower_scalar_type_ref(elem),
            access: BufferAccess::WriteOnly,
        })),
        TypeRef::F16 => Some(ParamTy::Scalar(ScalarTy::F16)),
        // CoopMatrix is not a valid kernel parameter type in M2.1 (only let-bindings).
        TypeRef::CoopMatrix { .. } => {
            errors.push(HirError::UnsupportedCoopMatrixAsParamInM2_1 {
                param_name: param_name.to_owned(),
                span,
            });
            None
        }
        TypeRef::Void | TypeRef::Bool => {
            errors.push(HirError::UnsupportedParamType {
                ty_name: format!("{:?}", tr),
                param_name: param_name.to_owned(),
                span,
            });
            None
        }
        // M3.2: shared[T,N] as a parameter is a hard error.
        TypeRef::Shared { .. } => {
            errors.push(HirError::SharedTypeNotAllowedAsParam {
                param_name: param_name.to_owned(),
                span,
            });
            None
        }
    }
}

/// Convert an AST `ScalarTypeRef` to an HIR `ScalarTy`.
///
/// `Bf16` is accepted by the parser but has no HIR representation; it should
/// never appear in a BUFFER element type (the parser's `parse_buffer_elem` does
/// not accept it). If encountered here it is a compiler bug.
fn lower_scalar_type_ref(str_ref: &ScalarTypeRef) -> ScalarTy {
    match str_ref {
        ScalarTypeRef::I8  => ScalarTy::I8,
        ScalarTypeRef::U8  => ScalarTy::U8,
        ScalarTypeRef::I32 => ScalarTy::I32,
        ScalarTypeRef::U32 => ScalarTy::U32,
        ScalarTypeRef::I64 => ScalarTy::I64,
        ScalarTypeRef::U64 => ScalarTy::U64,
        ScalarTypeRef::F16 => ScalarTy::F16,
        ScalarTypeRef::F32 => ScalarTy::F32,
        ScalarTypeRef::F64 => ScalarTy::F64,
        ScalarTypeRef::Bf16 => {
            // Bf16 cannot appear in buffer element types (parser rejects it there).
            // If we reach this, it is a compiler bug.
            panic!("lower_scalar_type_ref: bf16 has no HIR ScalarTy representation; \
                    this path should be unreachable for buffer element types")
        }
    }
}

/// M3.1: Derive the cooperative-matrix shape from a typechecked kernel body.
///
/// Walks the body for the FIRST `coopmat_mul_add` operation (HirExprKind::CoopMatBuiltin
/// with op == MulAdd). Extracts M, N, K, element types, and scope from the operand
/// bindings' `CoopMatKey` values (resolved through `body.bindings`).
///
/// Returns `None` if no `coopmat_mul_add` op is found (degenerate kernel).
///
/// Relies on M2.1 typecheck guarantees:
/// - a.n == b.m (KDimMismatch check) — K is derivable from a.n
/// - All operand uses are correct (AUseMismatch, BUseMismatch, CUseMismatch checks)
///
/// The pessimistic reviewer note: operands are `LocalRead(BindingId)` in matmul_tile,
/// so we must resolve them through `body.bindings`, not just read `HirExpr.ty`.
fn derive_coopmat_shape(body: &KernelBodyTyped) -> Option<CoopMatrixShape> {
    // Find the first coopmat_mul_add by searching all statements recursively.
    let mul_add_expr = find_first_mul_add_in_stmts(&body.stmts)?;

    // Extract the three argument expressions (a, b, c).
    let (a_key, b_key, c_key) = match mul_add_expr {
        HirExprKind::CoopMatBuiltin { op: crate::coopmat::CoopMatBuiltin::MulAdd, args, .. } => {
            if args.len() != 3 {
                return None;
            }
            let a_key = resolve_coopmat_key_from_expr(&args[0], body)?;
            let b_key = resolve_coopmat_key_from_expr(&args[1], body)?;
            let c_key = resolve_coopmat_key_from_expr(&args[2], body)?;
            (a_key, b_key, c_key)
        }
        _ => return None,
    };

    // Validate: a must be MatrixA, b must be MatrixB, c must be Accumulator.
    // (These should be guaranteed by typecheck, but we check defensively.)
    if a_key.use_ != CoopMatUse::MatrixA {
        return None;
    }
    if b_key.use_ != CoopMatUse::MatrixB {
        return None;
    }
    if c_key.use_ != CoopMatUse::Accumulator {
        return None;
    }

    // M = a.m, N = b.n, K = a.n (== b.m per KDimMismatch typecheck guarantee).
    let m = a_key.m;
    let n = b_key.n;
    let k = a_key.n; // == b_key.m guaranteed by typecheck

    Some(CoopMatrixShape {
        m,
        n,
        k,
        a_elem: a_key.elem,
        b_elem: b_key.elem,
        c_elem: c_key.elem,
        result_type: c_key.elem, // accumulator elem IS the result type
        scope: CoopMatScopeHir::Subgroup, // M3.1 always emits Subgroup scope
    })
}

/// Resolve the `CoopMatKey` for a cooperative-matrix expression.
///
/// Handles the common case where the expression is a `LocalRead(BindingId)` — the
/// typical form in matmul_tile (`let a: matrix[...] = ...; coopmat_mul_add(a, b, acc)`).
/// The binding's type (a `BindingTy::CoopMatrix(CoopMatKey)`) carries M, N, use, elem.
fn resolve_coopmat_key_from_expr(
    expr: &crate::expr::HirExpr,
    body: &KernelBodyTyped,
) -> Option<crate::coopmat::CoopMatKey> {
    match &expr.kind {
        HirExprKind::LocalRead(binding_id) => {
            // Resolve through the bindings table.
            body.bindings.iter().find(|b| b.id == *binding_id).and_then(|binding| {
                binding.ty.as_coopmat()
            })
        }
        HirExprKind::CoopMatBuiltin { result_ty, .. } => {
            // Inline coopmat expression (less common but supported).
            Some(*result_ty)
        }
        _ => None,
    }
}

/// Recursively search a list of HIR statements for the first `coopmat_mul_add`.
///
/// Returns a reference to the `HirExprKind` of the first MulAdd found.
fn find_first_mul_add_in_stmts(stmts: &[HirStmt]) -> Option<&HirExprKind> {
    for stmt in stmts {
        if let Some(kind) = find_mul_add_in_stmt(stmt) {
            return Some(kind);
        }
    }
    None
}

/// Search a `HirElse` arm for the first `coopmat_mul_add`.
fn find_mul_add_in_else_arm(else_arm: &HirElse) -> Option<&HirExprKind> {
    match else_arm {
        HirElse::Block(stmts) => find_first_mul_add_in_stmts(stmts),
        HirElse::If(hir_if) => {
            find_mul_add_in_expr(&hir_if.cond)
                .or_else(|| find_first_mul_add_in_stmts(&hir_if.then_block))
                .or_else(|| match &hir_if.else_arm {
                    Some(inner) => find_mul_add_in_else_arm(inner),
                    None => None,
                })
        }
    }
}

/// Search a single HIR statement for the first `coopmat_mul_add`.
fn find_mul_add_in_stmt(stmt: &HirStmt) -> Option<&HirExprKind> {
    match stmt {
        HirStmt::Let { init, .. } => find_mul_add_in_expr(init),
        HirStmt::Assign { value, .. } => find_mul_add_in_expr(value),
        HirStmt::If(hir_if) => {
            find_mul_add_in_expr(&hir_if.cond)
                .or_else(|| find_first_mul_add_in_stmts(&hir_if.then_block))
                .or_else(|| match &hir_if.else_arm {
                    Some(else_arm) => find_mul_add_in_else_arm(else_arm),
                    None => None,
                })
        }
        HirStmt::ForRange(for_range) => {
            find_first_mul_add_in_stmts(&for_range.body)
        }
        HirStmt::While(while_) => {
            find_mul_add_in_expr(&while_.cond)
                .or_else(|| find_first_mul_add_in_stmts(&while_.body))
        }
        HirStmt::CoopMatStore { .. }
        | HirStmt::BufferWrite { .. }
        | HirStmt::Return { .. }
        | HirStmt::Break { .. }
        | HirStmt::Continue { .. }
        | HirStmt::Barrier { .. }
        // M3.2: SharedWrite / SharedDeclMarker don't contain coopmat_mul_add.
        | HirStmt::SharedWrite { .. }
        | HirStmt::SharedDeclMarker { .. } => None,
    }
}

/// Search a HIR expression for the first `coopmat_mul_add`.
fn find_mul_add_in_expr(expr: &crate::expr::HirExpr) -> Option<&HirExprKind> {
    match &expr.kind {
        HirExprKind::CoopMatBuiltin { op: crate::coopmat::CoopMatBuiltin::MulAdd, .. } => {
            Some(&expr.kind)
        }
        HirExprKind::CoopMatBuiltin { args, .. } => {
            args.iter().find_map(find_mul_add_in_expr)
        }
        HirExprKind::Unary { operand, .. } => find_mul_add_in_expr(operand),
        HirExprKind::Binary { lhs, rhs, .. } => {
            find_mul_add_in_expr(lhs).or_else(|| find_mul_add_in_expr(rhs))
        }
        HirExprKind::ShortCircuit { lhs, rhs, .. } => {
            find_mul_add_in_expr(lhs).or_else(|| find_mul_add_in_expr(rhs))
        }
        HirExprKind::BitwiseBuiltin { args, .. } => {
            args.iter().find_map(find_mul_add_in_expr)
        }
        HirExprKind::BufferRead { index, .. } => find_mul_add_in_expr(index),
        HirExprKind::SubgroupBuiltin { args, .. } => {
            args.iter().find_map(find_mul_add_in_expr)
        }
        HirExprKind::Q4_0Builtin { args, .. } => {
            args.iter().find_map(find_mul_add_in_expr)
        }
        // M3.2c: ext-inst builtin args (e.g. exp(x)) may nest a coopmat op — recurse.
        HirExprKind::ExtInstBuiltin { args, .. } => {
            args.iter().find_map(find_mul_add_in_expr)
        }
        HirExprKind::IntLit { .. }
        | HirExprKind::FloatLit { .. }
        | HirExprKind::BoolLit(_)
        | HirExprKind::LocalRead(_)
        | HirExprKind::GidBuiltin { .. }
        // M3.3d: LocalInvocationIdBuiltin is a leaf — no coopmat ops nested in it.
        | HirExprKind::LocalInvocationIdBuiltin { .. }
        // M3.2: SharedRead carries an index expression; scan it for completeness (unlikely
        // to contain coopmat_mul_add but must be exhaustive).
        | HirExprKind::SharedRead { .. } => None,
    }
}

/// Public accessor for tests: given an AST module, return the first kernel's body block.
///
/// Used only in `typecheck::tests`.
#[doc(hidden)]
pub fn get_kernel_body_block(ast: &AstModule) -> Option<&axc_lexer::Spanned<axc_parser::ast::Block>> {
    ast.items.first().map(|item| {
        let axc_parser::ast::Item::Kernel(ref kd) = item.node;
        &kd.body
    })
}

/// Public test helper: lower AST params to `KernelParam`s without computing the
/// full binding plan. Used by `typecheck::tests` to construct param lists for
/// `typecheck_kernel_body` test helpers.
///
/// Errors are silently dropped (tests that exercise param errors use `lower_module`).
#[doc(hidden)]
pub fn lower_params_for_test(
    ast_params: &[axc_lexer::Spanned<axc_parser::ast::Param>],
) -> Vec<KernelParam> {
    let mut params: Vec<KernelParam> = Vec::new();
    let mut errors: Vec<crate::validate::HirError> = Vec::new();
    for (pos, spanned) in ast_params.iter().enumerate() {
        let ast_param = &spanned.node;
        let span: axc_lexer::Span = spanned.span;
        let name: &str = &ast_param.name.node;
        if let Some(ty) = lower_type_ref(&ast_param.ty.node, span, name, &mut errors) {
            params.push(KernelParam {
                name: name.to_owned(),
                ty,
                position: pos as u32,
                span,
            });
        }
    }
    params
}

/// Extract a single workgroup dimension from an `AnnotationArg::Int` or `HoleRef`.
///
/// `HoleRef` args produce a placeholder value of 1 so HIR construction succeeds;
/// the actual value is resolved by the enumerator at axc-optimize time. The caller
/// is responsible for not running the workgroup-dimension validator when holes are
/// present.
///
/// Validates that integer values are in `[1, u32::MAX]` (zero is caught by workgroup
/// validation; negative overflows u32).
fn extract_workgroup_dim(arg: &AnnotationArg, span: Span) -> Result<u32, HirError> {
    match arg {
        AnnotationArg::Int(v) => {
            if *v < 0 || *v > i64::from(u32::MAX) {
                Err(HirError::WorkgroupDimOverflow { got: *v, span })
            } else {
                Ok(*v as u32)
            }
        }
        // M2.3: HoleRef — placeholder; actual value comes from enumerator.
        AnnotationArg::HoleRef(_) => Ok(1),
        _ => Err(HirError::BadWorkgroupArity { got: 0, span }),
    }
}

/// Validate individual dimensions and product against the two-tier cap.
fn validate_workgroup_dims(
    wg: &WorkgroupDims,
    span: Span,
    kernel_name: &str,
    errors: &mut Vec<HirError>,
    warnings: &mut Vec<HirWarning>,
) -> bool {
    use crate::hir::{PORTABLE_MIN_WORKGROUP_INVOCATIONS, DESKTOP_MAX_WORKGROUP_INVOCATIONS};

    // Zero-dimension check (any dimension must be >= 1)
    if wg.x == 0 || wg.y == 0 || wg.z == 0 {
        errors.push(HirError::BadWorkgroupDim {
            x: wg.x as i64,
            y: wg.y as i64,
            z: wg.z as i64,
            product: wg.product(),
            max: DESKTOP_MAX_WORKGROUP_INVOCATIONS,
            span,
        });
        return false;
    }

    let product: u64 = wg.product();

    if product > DESKTOP_MAX_WORKGROUP_INVOCATIONS as u64 {
        // Hard error: exceeds the desktop ceiling
        errors.push(HirError::BadWorkgroupDim {
            x: wg.x as i64,
            y: wg.y as i64,
            z: wg.z as i64,
            product,
            max: DESKTOP_MAX_WORKGROUP_INVOCATIONS,
            span,
        });
        return false;
    }

    if product > PORTABLE_MIN_WORKGROUP_INVOCATIONS as u64 {
        // Warning: exceeds the Vulkan 1.1 portability floor but within desktop ceiling
        warnings.push(HirWarning::WorkgroupExceedsPortableFloor {
            name: kernel_name.to_owned(),
            product,
            floor: PORTABLE_MIN_WORKGROUP_INVOCATIONS,
            span,
        });
    }

    true
}

/// Map `AnnotationArg::Call { name, args }` to a `ComplexityForm` enum.
///
/// M0 whitelist: `O(1)`, `O(n)`, `Theta(n)`, `Omega(n)`.
/// Everything else yields `HirError::UnsupportedComplexityInM0`.
fn lower_complexity(arg: &AnnotationArg, span: Span) -> Result<ComplexityForm, HirError> {
    match arg {
        AnnotationArg::Call { name, args } => {
            if args.len() != 1 {
                return Err(HirError::UnsupportedComplexityInM0 { span });
            }
            let var: ComplexityVar = match &args[0].node {
                AnnotationArg::Int(1) => ComplexityVar::One,
                AnnotationArg::Ident(n) if n == "n" => ComplexityVar::N,
                _ => return Err(HirError::UnsupportedComplexityInM0 { span }),
            };
            match name.as_str() {
                "O"     => Ok(ComplexityForm::O(var)),
                "Theta" => Ok(ComplexityForm::Theta(var)),
                "Omega" => Ok(ComplexityForm::Omega(var)),
                _       => Err(HirError::UnsupportedComplexityInM0 { span }),
            }
        }
        _ => Err(HirError::UnsupportedComplexityInM0 { span }),
    }
}

/// M2.3: Lower `@strategy` annotation args to a `StrategyHoles` struct.
///
/// The args are in the uniform Call-wrapped form (whether the user wrote the
/// sugar `{ k: ?[...] }` or the explicit `(k(?[...]))` form). Each arg is:
///   `Call { name: "hole_name", args: [Hole { name: "hole_name", candidates }] }`
///
/// Validation:
/// - Duplicate hole names → `HirError::DuplicateStrategyHoleName` (defense-in-depth)
/// - Empty candidate list → `HirError::StrategyHoleEmptyCandidates`
/// - Non-positive candidate → `HirError::StrategyHoleNonPositive`
/// - HoleRef inside strategy block → `HirError::HoleRefInStrategyBlock`
fn lower_strategy_annotation(
    args: &[axc_lexer::Spanned<AnnotationArg>],
    ann_span: Span,
    errors: &mut Vec<HirError>,
) -> StrategyHoles {
    let mut holes: StrategyHoles = StrategyHoles::new();

    for arg in args {
        match &arg.node {
            AnnotationArg::Call { name, args: inner_args } => {
                // Duplicate check
                if holes.map.contains_key(name.as_str()) {
                    errors.push(HirError::DuplicateStrategyHoleName {
                        name: name.clone(),
                        span: arg.span,
                    });
                    continue;
                }

                // Inner arg should be a Hole
                if inner_args.len() != 1 {
                    errors.push(HirError::UnsupportedComplexityInM0 { span: arg.span });
                    continue;
                }

                match &inner_args[0].node {
                    AnnotationArg::Hole { candidates, .. } => {
                        // Defense-in-depth: validate candidates
                        if candidates.is_empty() {
                            errors.push(HirError::StrategyHoleEmptyCandidates {
                                name: name.clone(),
                                span: arg.span,
                            });
                            continue;
                        }
                        for &v in candidates.iter() {
                            if v < 1 {
                                errors.push(HirError::StrategyHoleNonPositive {
                                    name: name.clone(),
                                    value: v,
                                    span: arg.span,
                                });
                            }
                        }
                        holes.map.insert(name.clone(), candidates.clone());
                    }
                    AnnotationArg::HoleRef(ref_name) => {
                        errors.push(HirError::HoleRefInStrategyBlock {
                            name: ref_name.clone(),
                            span: inner_args[0].span,
                        });
                    }
                    other => {
                        errors.push(HirError::UnsupportedComplexityInM0 { span: arg.span });
                        let _ = other;
                    }
                }
            }
            AnnotationArg::HoleRef(ref_name) => {
                errors.push(HirError::HoleRefInStrategyBlock {
                    name: ref_name.clone(),
                    span: arg.span,
                });
            }
            _ => {
                // Unknown arg shape inside @strategy — skip with a noop.
                // Parser should have caught malformed strategy blocks.
                let _ = ann_span;
            }
        }
    }

    holes
}

/// M2.3: Validate that every HoleRef across all annotations refers to a declared hole,
/// and emit UnusedStrategyHole warnings for holes never referenced.
fn validate_hole_refs(
    annotations: &[axc_lexer::Spanned<axc_parser::ast::Annotation>],
    strategy: Option<&StrategyHoles>,
    params: &[axc_lexer::Spanned<axc_parser::ast::Param>],
    kernel_span: Span,
    errors: &mut Vec<HirError>,
    warnings: &mut Vec<HirWarning>,
) {
    let _ = kernel_span; // currently unused; may be needed for future diagnostics
    // Collect all hole names referenced via HoleRef across non-strategy annotations.
    let mut referenced_holes: std::collections::HashSet<String> = std::collections::HashSet::new();

    for ann in annotations {
        if ann.node.name.node == "strategy" {
            // Don't validate inside the @strategy block itself — only non-strategy annotations.
            continue;
        }
        collect_hole_refs_in_args(&ann.node.args, &mut referenced_holes);
    }

    // For each referenced hole, check it is declared in @strategy.
    for ann in annotations {
        if ann.node.name.node == "strategy" {
            continue;
        }
        validate_hole_refs_in_args(&ann.node.args, strategy, errors);
    }

    // UnusedStrategyHole: declared holes that were never referenced.
    if let Some(holes) = strategy {
        for (hole_name, _) in holes.map.iter() {
            if !referenced_holes.contains(hole_name.as_str()) {
                warnings.push(HirWarning::UnusedStrategyHole {
                    name: hole_name.clone(),
                    span: Span::new(0, 0), // exact span not tracked in HIR for unused holes
                });
            }
        }

        // StrategyHoleShadowsKernelParam: check if any hole name matches a kernel param.
        for (hole_name, _) in holes.map.iter() {
            let shadows = params.iter().any(|p| p.node.name.node == *hole_name);
            if shadows {
                warnings.push(HirWarning::StrategyHoleShadowsKernelParam {
                    name: hole_name.clone(),
                    span: Span::new(0, 0),
                });
            }
        }
    }
}

/// Recursively collect all HoleRef names from an annotation arg list.
fn collect_hole_refs_in_args(
    args: &[axc_lexer::Spanned<AnnotationArg>],
    out: &mut std::collections::HashSet<String>,
) {
    for arg in args {
        match &arg.node {
            AnnotationArg::HoleRef(name) => {
                out.insert(name.clone());
            }
            AnnotationArg::Call { args: inner, .. } => {
                collect_hole_refs_in_args(inner, out);
            }
            _ => {}
        }
    }
}

/// Recursively validate that HoleRef names are all declared in `strategy`.
fn validate_hole_refs_in_args(
    args: &[axc_lexer::Spanned<AnnotationArg>],
    strategy: Option<&StrategyHoles>,
    errors: &mut Vec<HirError>,
) {
    for arg in args {
        match &arg.node {
            AnnotationArg::HoleRef(name) => {
                let declared = strategy
                    .map(|s| s.map.contains_key(name.as_str()))
                    .unwrap_or(false);
                if !declared {
                    errors.push(HirError::UndefinedStrategyHole {
                        name: name.clone(),
                        span: arg.span,
                    });
                }
            }
            AnnotationArg::Call { args: inner, .. } => {
                validate_hole_refs_in_args(inner, strategy, errors);
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axc_parser::parse;
    use crate::hir::{ComplexityForm, ComplexityVar};

    fn lower_src(src: &str) -> (HirModule, Vec<HirError>, Vec<HirWarning>) {
        let (ast, lex_errs, parse_errs) = parse(src);
        assert!(lex_errs.is_empty(), "lex errors: {lex_errs:?}");
        assert!(parse_errs.is_empty(), "parse errors: {parse_errs:?}");
        lower_module(&ast)
    }

    // ── Happy path ────────────────────────────────────────────────────────────

    #[test]
    fn happy_empty_kernel_clean() {
        let (hir, errors, warnings) = lower_src(
            "@kernel @workgroup(64,1,1) fn empty() -> void { return; }",
        );
        assert!(errors.is_empty(), "hir errors: {errors:?}");
        assert!(warnings.is_empty(), "hir warnings: {warnings:?}");
        assert_eq!(hir.kernels.len(), 1);
        assert_eq!(hir.kernels[0].name, "empty");
        assert_eq!(hir.kernels[0].annotations.workgroup.x, 64);
    }

    // ── AT-4: missing @workgroup ─────────────────────────────────────────────

    #[test]
    fn missing_workgroup_rejected() {
        let (ast, _, _) = parse("@kernel fn empty() -> void { return; }");
        let (_, errors, _) = lower_module(&ast);
        let found = errors.iter().any(|e| matches!(e, HirError::MissingWorkgroup { name, .. } if name == "empty"));
        assert!(found, "expected MissingWorkgroup: {errors:?}");
    }

    // ── AT-5: two-tier workgroup cap ─────────────────────────────────────────

    #[test]
    fn workgroup_zero_dim_is_error() {
        let (ast, _, _) = parse("@kernel @workgroup(0,1,1) fn k() -> void { return; }");
        let (_, errors, _) = lower_module(&ast);
        assert!(errors.iter().any(|e| matches!(e, HirError::BadWorkgroupDim { .. })),
            "expected BadWorkgroupDim: {errors:?}");
    }

    #[test]
    fn workgroup_product_128_is_clean() {
        // product == 128 (at portable floor): clean
        let (hir, errors, warnings) = lower_src(
            "@kernel @workgroup(8,8,2) fn k() -> void { return; }",
        );
        assert!(errors.is_empty(), "errors: {errors:?}");
        assert!(warnings.is_empty(), "warnings: {warnings:?}");
        assert_eq!(hir.kernels[0].annotations.workgroup.product(), 128);
    }

    #[test]
    fn workgroup_product_129_is_warning() {
        // product == 129 (first over portable floor): WARN
        let (_, errors, warnings) = lower_src(
            "@kernel @workgroup(129,1,1) fn k() -> void { return; }",
        );
        assert!(errors.is_empty(), "errors: {errors:?}");
        assert!(!warnings.is_empty(), "expected a warning for product=129: {warnings:?}");
    }

    #[test]
    fn workgroup_product_1024_is_warning() {
        // product == 1024 (at desktop ceiling, boundary clarification from spec rev 2): WARN
        let (_, errors, warnings) = lower_src(
            "@kernel @workgroup(16,8,8) fn k() -> void { return; }",
        );
        assert!(errors.is_empty(), "errors: {errors:?}");
        assert!(!warnings.is_empty(), "expected warning at product=1024: {warnings:?}");
    }

    #[test]
    fn workgroup_product_1025_is_error() {
        // product == 1025 (over desktop ceiling): ERROR
        let (ast, _, _) = parse("@kernel @workgroup(1025,1,1) fn k() -> void { return; }");
        let (_, errors, _) = lower_module(&ast);
        assert!(errors.iter().any(|e| matches!(e, HirError::BadWorkgroupDim { product, .. } if *product == 1025)),
            "expected BadWorkgroupDim with product=1025: {errors:?}");
    }

    #[test]
    fn workgroup_product_256_is_warning() {
        let (_, errors, warnings) = lower_src(
            "@kernel @workgroup(16,8,2) fn k() -> void { return; }",
        );
        assert!(errors.is_empty(), "errors: {errors:?}");
        assert!(!warnings.is_empty(), "expected warning for product=256: {warnings:?}");
    }

    #[test]
    fn workgroup_bad_arity_4() {
        // 4 arguments instead of 3
        let (ast, _, _) = parse("@kernel @workgroup(64,1,1,1) fn k() -> void { return; }");
        let (_, errors, _) = lower_module(&ast);
        assert!(errors.iter().any(|e| matches!(e, HirError::BadWorkgroupArity { got: 4, .. })),
            "expected BadWorkgroupArity{{got:4}}: {errors:?}");
    }

    #[test]
    fn workgroup_bad_arity_2() {
        let (ast, _, _) = parse("@kernel @workgroup(64,1) fn k() -> void { return; }");
        let (_, errors, _) = lower_module(&ast);
        assert!(errors.iter().any(|e| matches!(e, HirError::BadWorkgroupArity { got: 2, .. })),
            "expected BadWorkgroupArity{{got:2}}: {errors:?}");
    }

    #[test]
    fn duplicate_workgroup_annotation() {
        let (ast, _, _) = parse("@kernel @workgroup(64,1,1) @workgroup(32,1,1) fn k() -> void { return; }");
        let (_, errors, _) = lower_module(&ast);
        assert!(errors.iter().any(|e| matches!(e, HirError::DuplicateAnnotation { name, .. } if name == "workgroup")),
            "expected DuplicateAnnotation: {errors:?}");
    }

    #[test]
    fn unknown_annotation_in_m0() {
        let (ast, _, _) = parse("@kernel @workgroup(64,1,1) @coalesced fn k() -> void { return; }");
        let (_, errors, _) = lower_module(&ast);
        assert!(errors.iter().any(|e| matches!(e, HirError::UnknownAnnotationInM0 { name, .. } if name == "coalesced")),
            "expected UnknownAnnotationInM0: {errors:?}");
    }

    #[test]
    fn non_void_return_type_rejected() {
        // Parser accepts i32 return type; HIR must reject it
        let (ast, _, _) = parse("@kernel @workgroup(1,1,1) fn k() -> i32 { return; }");
        let (_, errors, _) = lower_module(&ast);
        assert!(errors.iter().any(|e| matches!(e, HirError::BadKernelReturnType { .. })),
            "expected BadKernelReturnType: {errors:?}");
    }

    // ── complexity_o1_lowers_to_structured_enum (anti-pattern #7 guard) ──────

    #[test]
    fn complexity_o1_lowers_to_enum() {
        let (hir, errors, _) = lower_src(
            "@kernel @workgroup(64,1,1) @complexity(O(1)) fn k() -> void { return; }",
        );
        assert!(errors.is_empty(), "errors: {errors:?}");
        assert_eq!(
            hir.kernels[0].annotations.complexity,
            Some(ComplexityForm::O(ComplexityVar::One)),
            "expected ComplexityForm::O(ComplexityVar::One)"
        );
    }

    #[test]
    fn complexity_on_lowers_to_enum() {
        let (hir, errors, _) = lower_src(
            "@kernel @workgroup(64,1,1) @complexity(O(n)) fn k() -> void { return; }",
        );
        assert!(errors.is_empty(), "errors: {errors:?}");
        assert_eq!(
            hir.kernels[0].annotations.complexity,
            Some(ComplexityForm::O(ComplexityVar::N))
        );
    }

    #[test]
    fn complexity_theta_n_lowers_to_enum() {
        let (hir, errors, _) = lower_src(
            "@kernel @workgroup(64,1,1) @complexity(Theta(n)) fn k() -> void { return; }",
        );
        assert!(errors.is_empty(), "errors: {errors:?}");
        assert_eq!(
            hir.kernels[0].annotations.complexity,
            Some(ComplexityForm::Theta(ComplexityVar::N))
        );
    }

    #[test]
    fn negative_workgroup_dim_overflow() {
        // -1 does not fit in u32 → WorkgroupDimOverflow
        let (ast, _, _) = parse("@kernel @workgroup(-1,1,1) fn k() -> void { return; }");
        let (_, errors, _) = lower_module(&ast);
        assert!(errors.iter().any(|e| matches!(e, HirError::WorkgroupDimOverflow { .. })),
            "expected WorkgroupDimOverflow: {errors:?}");
    }

    // ── M1.2 lower tests ──────────────────────────────────────────────────────

    #[test]
    fn lower_buffer_param_assigns_binding_0() {
        let (hir, errors, _) = lower_src(
            "@kernel @workgroup(1,1,1) fn k(x: readonly_buffer[f32]) -> void { return; }",
        );
        assert!(errors.is_empty(), "errors: {errors:?}");
        let plan = &hir.kernels[0].binding_plan;
        assert_eq!(plan.buffers.len(), 1, "expected 1 buffer binding");
        assert_eq!(plan.buffers[0].buffer_position, 0, "first buffer must be at binding 0");
        assert_eq!(plan.buffers[0].name, "x");
    }

    #[test]
    fn lower_saxpy_params_binding_layout() {
        // saxpy(n: u32, alpha: f32, x: readonly_buffer[f32], y: buffer[f32])
        // x → binding 0, y → binding 1; n and alpha → push constant
        let (hir, errors, _) = lower_src(concat!(
            "@kernel @workgroup(64,1,1) fn saxpy(",
            "n: u32, alpha: f32, x: readonly_buffer[f32], y: buffer[f32]",
            ") -> void { return; }",
        ));
        assert!(errors.is_empty(), "errors: {errors:?}");
        let plan = &hir.kernels[0].binding_plan;
        assert_eq!(plan.buffers.len(), 2);
        assert_eq!(plan.buffers[0].name, "x");
        assert_eq!(plan.buffers[0].buffer_position, 0);
        assert_eq!(plan.buffers[1].name, "y");
        assert_eq!(plan.buffers[1].buffer_position, 1);
        assert_eq!(plan.scalars.len(), 2);
        assert_eq!(plan.scalars[0].name, "n");
        assert_eq!(plan.scalars[0].member_index, 0);
        assert_eq!(plan.scalars[1].name, "alpha");
        assert_eq!(plan.scalars[1].member_index, 1);
    }

    #[test]
    fn lower_scalar_param_populates_kernel() {
        let (hir, errors, _) = lower_src(
            "@kernel @workgroup(1,1,1) fn k(a: f32) -> void { return; }",
        );
        assert!(errors.is_empty(), "errors: {errors:?}");
        let plan = &hir.kernels[0].binding_plan;
        assert_eq!(plan.scalars.len(), 1);
        assert_eq!(plan.scalars[0].name, "a");
        assert_eq!(plan.scalars[0].ty, crate::ty::ScalarTy::F32);
        assert_eq!(plan.scalars[0].offset, 0);
        assert_eq!(plan.scalars[0].member_index, 0);
        assert_eq!(plan.push_constant_total_bytes, 4);
    }

    #[test]
    fn lower_pushconstant_over_128_rejected() {
        // 17 × f64 = 136 bytes > 128 → PushConstantTooLarge
        let params: String = (0..17).map(|i| format!("p{i}: f64")).collect::<Vec<_>>().join(", ");
        let src = format!("@kernel @workgroup(1,1,1) fn k({params}) -> void {{ return; }}");
        let (ast, lex_errs, parse_errs) = parse(&src);
        assert!(lex_errs.is_empty(), "lex: {lex_errs:?}");
        assert!(parse_errs.is_empty(), "parse: {parse_errs:?}");
        let (_, errors, _) = lower_module(&ast);
        assert!(
            errors.iter().any(|e| matches!(e, HirError::BindingPlan(
                crate::param::BindingPlanError::PushConstantTooLarge { got, .. }
            ) if *got > 128)),
            "expected BindingPlan::PushConstantTooLarge: {errors:?}"
        );
    }

    #[test]
    fn lower_pushconstant_over_128_points_at_param32() {
        // 33 × u32 = 132 bytes; param p32 (index 32) is the overflow trigger.
        let params: String = (0..33).map(|i| format!("p{i}: u32")).collect::<Vec<_>>().join(", ");
        let src = format!("@kernel @workgroup(1,1,1) fn k({params}) -> void {{ return; }}");
        let (ast, lex_errs, parse_errs) = parse(&src);
        assert!(lex_errs.is_empty(), "lex: {lex_errs:?}");
        assert!(parse_errs.is_empty(), "parse: {parse_errs:?}");
        let (_, errors, _) = lower_module(&ast);

        let overflow_error = errors.iter().find_map(|e| {
            if let HirError::BindingPlan(crate::param::BindingPlanError::PushConstantTooLarge {
                got, limit, overflowing_param_name, span, ..
            }) = e {
                Some((*got, *limit, overflowing_param_name.clone(), *span))
            } else {
                None
            }
        });
        let (got, limit, name, span) = overflow_error
            .expect("expected PushConstantTooLarge error");

        assert_eq!(got, 132, "132 bytes = 33 × 4");
        assert_eq!(limit, 128, "Vulkan minimum is 128 bytes");
        assert_eq!(name, "p32", "p32 (index 32) is the overflow param, NOT p0");

        // The span of p32 must differ from p0's span.
        // We re-parse to get the span of p0 from the params list.
        let kd = ast.items.first().map(|item| {
            let axc_parser::ast::Item::Kernel(ref kd) = item.node; kd
        }).unwrap();
        let p0_span = kd.params[0].span;
        assert_ne!(span, p0_span, "error span must point at p32, not p0");
    }

    #[test]
    fn lower_scalar_param_bool_rejected() {
        // bool is not a valid scalar param type
        let (ast, _, _) = parse("@kernel @workgroup(1,1,1) fn k(b: bool) -> void { return; }");
        let (_, errors, _) = lower_module(&ast);
        assert!(
            errors.iter().any(|e| matches!(e, HirError::UnsupportedParamType { .. })),
            "expected UnsupportedParamType for bool param: {errors:?}"
        );
    }

    #[test]
    fn lower_buffer_elem_bool_rejected() {
        // buffer[bool] is not supported in M1.2 — bool is not a ScalarTypeRef in the parser
        // The parser grammar only allows i32/u32/i64/u64/f32/f64 as buffer element types.
        // This test verifies that attempting to parse buffer[bool] fails at parse level.
        // (Bool is not a valid buffer element type per the parser grammar.)
        let (ast, lex_errs, parse_errs) = parse(
            "@kernel @workgroup(1,1,1) fn k(x: buffer[bool]) -> void { return; }",
        );
        assert!(lex_errs.is_empty(), "lex: {lex_errs:?}");
        // The parser rejects bool as a buffer element type (grammar doesn't allow it)
        // OR the HIR rejects it via UnsupportedParamType.
        if parse_errs.is_empty() {
            let (_, errors, _) = lower_module(&ast);
            assert!(
                !errors.is_empty(),
                "expected HIR error for buffer[bool]: {errors:?}"
            );
        } else {
            // Parser rejected it — that's fine too
            assert!(!parse_errs.is_empty(), "parse should reject buffer[bool]");
        }
    }

    #[test]
    fn lower_nested_buffer_rejected() {
        // buffer[buffer[f32]] — nested buffers are not valid in M1.2
        // The parser grammar uses ScalarTypeRef (not TypeRef) for buffer elements,
        // so buffer[buffer[f32]] would be a parse error.
        let (_, lex_errs, parse_errs) = parse(
            "@kernel @workgroup(1,1,1) fn k(x: buffer[buffer[f32]]) -> void { return; }",
        );
        assert!(lex_errs.is_empty(), "lex: {lex_errs:?}");
        // Either parser or HIR must reject nested buffers
        assert!(!parse_errs.is_empty(), "parser should reject buffer[buffer[f32]]");
    }

    #[test]
    fn lower_type_ref_scalar_types() {
        // All 6 M1.2 scalar param types should lower cleanly
        for ty_str in &["i32", "u32", "i64", "u64", "f32", "f64"] {
            let src = format!(
                "@kernel @workgroup(1,1,1) fn k(x: {ty_str}) -> void {{ return; }}"
            );
            let (hir, errors, _) = lower_src(&src);
            assert!(errors.is_empty(), "errors for {ty_str}: {errors:?}");
            assert_eq!(hir.kernels[0].binding_plan.scalars.len(), 1,
                "expected 1 scalar for {ty_str}");
        }
    }

    // ── M1.3 lower.rs tests (AT-319 / AT-320 and related) ────────────────────

    use crate::expr::HirStmt;
    use crate::hir::KernelBody;

    /// Helper: compile source and return the typed HIR statements from the first kernel.
    fn lower_stmts(src: &str) -> Vec<HirStmt> {
        let (hir, errors, _) = lower_src(src);
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        let kernel = &hir.kernels[0];
        match &kernel.body {
            KernelBody::Typed(typed) => typed.stmts.clone(),
            KernelBody::Empty => panic!("expected Typed body, got Empty"),
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn lower_if_stmt_produces_HirStmt_If() {
        let stmts = lower_stmts(
            "@kernel @workgroup(1,1,1) fn k() -> void { if true { } return; }",
        );
        assert!(
            stmts.iter().any(|s| matches!(s, HirStmt::If(_))),
            "expected HirStmt::If: {stmts:?}"
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn lower_for_range_produces_HirStmt_ForRange_with_induction_binding() {
        let stmts = lower_stmts(
            "@kernel @workgroup(1,1,1) fn k() -> void { for i in range(0u32, 5u32) { } return; }",
        );
        let found = stmts.iter().any(|s| matches!(s, HirStmt::ForRange(_)));
        assert!(found, "expected HirStmt::ForRange: {stmts:?}");
        // Verify the induction BindingId is non-zero (was allocated).
        if let Some(HirStmt::ForRange(f)) = stmts.iter().find(|s| matches!(s, HirStmt::ForRange(_))) {
            assert_eq!(f.step.value, 1, "default step must be 1");
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn lower_while_produces_HirStmt_While() {
        let stmts = lower_stmts(
            "@kernel @workgroup(1,1,1) fn k() -> void { while false { } return; }",
        );
        assert!(
            stmts.iter().any(|s| matches!(s, HirStmt::While(_))),
            "expected HirStmt::While: {stmts:?}"
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn lower_break_in_while_produces_HirStmt_Break() {
        let stmts = lower_stmts(
            "@kernel @workgroup(1,1,1) fn k() -> void { while false { break; } return; }",
        );
        let w = stmts.iter().find_map(|s| if let HirStmt::While(w) = s { Some(w) } else { None })
            .expect("expected While");
        assert!(
            w.body.iter().any(|s| matches!(s, HirStmt::Break { .. })),
            "expected Break inside while body"
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn lower_continue_in_for_produces_HirStmt_Continue() {
        let stmts = lower_stmts(
            "@kernel @workgroup(1,1,1) fn k() -> void { for i in range(0u32, 3u32) { continue; } return; }",
        );
        let f = stmts.iter().find_map(|s| if let HirStmt::ForRange(f) = s { Some(f) } else { None })
            .expect("expected ForRange");
        assert!(
            f.body.iter().any(|s| matches!(s, HirStmt::Continue { .. })),
            "expected Continue inside for body"
        );
    }

    #[test]
    fn lower_nested_if_inside_for_has_break_targeting_for() {
        // `for i in range(...) { if true { break; } }` — break should be inside
        // the for body, inside the if then-block.
        let stmts = lower_stmts(
            "@kernel @workgroup(1,1,1) fn k() -> void { for i in range(0u32, 5u32) { if true { break; } } return; }",
        );
        let f = stmts.iter().find_map(|s| if let HirStmt::ForRange(f) = s { Some(f) } else { None })
            .expect("expected ForRange");
        // The if stmt must be in the for body.
        let has_if_with_break = f.body.iter().any(|s| {
            if let HirStmt::If(hir_if) = s {
                hir_if.then_block.iter().any(|ts| matches!(ts, HirStmt::Break { .. }))
            } else {
                false
            }
        });
        assert!(has_if_with_break, "expected If containing Break inside ForRange body");
    }

    // AT-319: reduction.axc lowers to HIR with exactly one HirStmt::ForRange
    #[test]
    fn lower_reduction_example_produces_one_for_range() {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .expect("CARGO_MANIFEST_DIR not set");
        let examples_dir = std::path::PathBuf::from(&manifest_dir)
            .join("..").join("..").join("examples");
        let source_path = examples_dir.join("reduction.axc");
        assert!(source_path.exists(), "examples/reduction.axc not found at {:?}", source_path);

        let src = std::fs::read_to_string(&source_path)
            .expect("failed to read reduction.axc");
        let (hir, errors, _) = lower_src(&src);
        assert!(errors.is_empty(), "expected no HirErrors for reduction.axc: {errors:?}");

        let kernel = &hir.kernels[0];
        let stmts = match &kernel.body {
            KernelBody::Typed(t) => &t.stmts,
            KernelBody::Empty => panic!("expected Typed body for reduction.axc"),
        };
        let for_count = stmts.iter().filter(|s| matches!(s, HirStmt::ForRange(_))).count();
        assert_eq!(for_count, 1, "expected exactly 1 HirStmt::ForRange in reduction.axc; got {for_count}");
    }

    // AT-320: vector_axpy.axc lowers to HIR with exactly one HirStmt::ForRange
    // and a HirStmt::BufferWrite inside the loop body.
    #[test]
    fn lower_vector_axpy_example_produces_one_for_range() {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .expect("CARGO_MANIFEST_DIR not set");
        let examples_dir = std::path::PathBuf::from(&manifest_dir)
            .join("..").join("..").join("examples");
        let source_path = examples_dir.join("vector_axpy.axc");
        assert!(source_path.exists(), "examples/vector_axpy.axc not found at {:?}", source_path);

        let src = std::fs::read_to_string(&source_path)
            .expect("failed to read vector_axpy.axc");
        let (hir, errors, _) = lower_src(&src);
        assert!(errors.is_empty(), "expected no HirErrors for vector_axpy.axc: {errors:?}");

        let kernel = &hir.kernels[0];
        let stmts = match &kernel.body {
            KernelBody::Typed(t) => &t.stmts,
            KernelBody::Empty => panic!("expected Typed body for vector_axpy.axc"),
        };
        // Exactly one ForRange at the top level.
        let for_count = stmts.iter().filter(|s| matches!(s, HirStmt::ForRange(_))).count();
        assert_eq!(for_count, 1, "expected exactly 1 HirStmt::ForRange in vector_axpy.axc; got {for_count}");
        // The for loop body must contain at least one HirStmt::BufferWrite.
        let for_stmt = stmts.iter().find_map(|s| if let HirStmt::ForRange(f) = s { Some(f) } else { None })
            .expect("expected ForRange");
        let has_buffer_write = for_stmt.body.iter().any(|s| matches!(s, HirStmt::BufferWrite { .. }));
        assert!(has_buffer_write, "expected HirStmt::BufferWrite inside the for loop body of vector_axpy.axc");
    }

    // ── M1.4 Lower tests (AT-14.4) ───────────────────────────────────────────

    // AT-421: subgroup_invocation_id() lowers to HirExprKind::SubgroupBuiltin { op: InvocationId }
    #[test]
    fn lower_sg_invocation_id_produces_subgroup_builtin_node() {
        use crate::subgroup::SubgroupOp;
        use crate::expr::HirExprKind;
        let (hir, errors, _) = lower_src(
            "@kernel @workgroup(64,1,1) fn k() -> void { let id: u32 = subgroup_invocation_id(); return; }"
        );
        assert!(errors.is_empty(), "errors: {errors:?}");
        let stmts = match &hir.kernels[0].body {
            KernelBody::Typed(t) => &t.stmts,
            KernelBody::Empty => panic!("expected Typed"),
        };
        let let_stmt = stmts.iter().find(|s| matches!(s, HirStmt::Let { .. }))
            .expect("expected Let stmt");
        if let HirStmt::Let { init, .. } = let_stmt {
            assert!(
                matches!(&init.kind, HirExprKind::SubgroupBuiltin { op: SubgroupOp::InvocationId, .. }),
                "expected SubgroupBuiltin::InvocationId; got {:?}", &init.kind
            );
        } else {
            panic!("expected HirStmt::Let");
        }
    }

    // AT-422: workgroup_barrier() lowers to HirStmt::Barrier { kind: Workgroup }
    #[test]
    fn lower_workgroup_barrier_produces_barrier_stmt() {
        use crate::subgroup::BarrierKind;
        let (hir, errors, _) = lower_src(
            "@kernel @workgroup(64,1,1) fn k() -> void { workgroup_barrier(); return; }"
        );
        assert!(errors.is_empty(), "errors: {errors:?}");
        let stmts = match &hir.kernels[0].body {
            KernelBody::Typed(t) => &t.stmts,
            KernelBody::Empty => panic!("expected Typed"),
        };
        assert!(
            stmts.iter().any(|s| matches!(s, HirStmt::Barrier { kind: BarrierKind::Workgroup, .. })),
            "expected HirStmt::Barrier{{Workgroup}} in stmts"
        );
    }

    // AT-423: subgroup_reduce_add(f32) lowers to Reduce(Add) with correct binding type
    #[test]
    fn lower_sg_reduce_add_f32_produces_correct_binding_ty() {
        use crate::subgroup::{SubgroupOp, SubgroupReduceKind};
        use crate::expr::HirExprKind;
        let (hir, errors, _) = lower_src(
            "@kernel @workgroup(32,1,1) fn k() -> void { let v: f32 = 1.0f32; let r: f32 = subgroup_reduce_add(v); return; }"
        );
        assert!(errors.is_empty(), "errors: {errors:?}");
        let typed = match &hir.kernels[0].body {
            KernelBody::Typed(t) => t,
            KernelBody::Empty => panic!("expected Typed"),
        };
        // The second binding is the reduce result.
        assert_eq!(typed.bindings[1].ty, crate::expr::BindingTy::Scalar(crate::ty::ScalarTy::F32), "reduce result must be f32");
        // Find the Let stmt for `r` and check the SubgroupBuiltin node.
        let r_stmt = typed.stmts.iter()
            .filter_map(|s| if let HirStmt::Let { init, .. } = s { Some(init) } else { None })
            .nth(1)
            .expect("expected second Let stmt");
        assert!(
            matches!(&r_stmt.kind, HirExprKind::SubgroupBuiltin { op: SubgroupOp::Reduce(SubgroupReduceKind::Add), .. }),
            "expected SubgroupBuiltin::Reduce(Add); got {:?}", &r_stmt.kind
        );
    }

    // ── M2.3: Strategy holes HIR tests (AT-1008..AT-1012, AT-1051) ────────────

    /// AT-1008: HIR lowers @strategy into StrategyHoles with BTreeMap ordering
    /// (alphabetical, not source order).
    #[test]
    fn at_1008_hir_lowers_strategy_to_btreemap() {
        let src = "@strategy { b: ?[1], a: ?[2] } @kernel @workgroup(1, 1, 1) fn k() -> void { return; }";
        let (hir, errors, _warns) = lower_src(src);
        assert!(errors.is_empty(), "errors: {errors:?}");
        let kernel = &hir.kernels[0];
        let strat = kernel.annotations.strategy.as_ref().expect("strategy should be Some");
        // Keys must be in alphabetical order (BTreeMap)
        let keys: Vec<&String> = strat.map.keys().collect();
        assert_eq!(keys, vec!["a", "b"], "BTreeMap must yield alphabetical order: {keys:?}");
        assert_eq!(strat.map["a"], vec![2i64]);
        assert_eq!(strat.map["b"], vec![1i64]);
    }

    /// AT-1009: HIR rejects undefined hole reference with UndefinedStrategyHole.
    #[test]
    fn at_1009_hir_rejects_undefined_strategy_hole() {
        let src = "@kernel @workgroup(?missing_hole, 1, 1) fn k() -> void { return; }";
        let (_, errors, _) = lower_src(src);
        assert!(
            errors.iter().any(|e| matches!(
                e,
                HirError::UndefinedStrategyHole { name, .. } if name == "missing_hole"
            )),
            "expected UndefinedStrategyHole {{ name: \"missing_hole\" }}: {errors:?}"
        );
    }

    /// AT-1010: HIR rejects duplicate hole name in @strategy.
    #[test]
    fn at_1010_hir_rejects_duplicate_hole_name() {
        // Use the explicit Call form where duplicates are expressible.
        let src = "@strategy(x(?[1]), x(?[2])) @kernel @workgroup(1, 1, 1) fn k() -> void { return; }";
        let (_, errors, _) = lower_src(src);
        assert!(
            errors.iter().any(|e| matches!(
                e,
                HirError::DuplicateStrategyHoleName { name, .. } if name == "x"
            )),
            "expected DuplicateStrategyHoleName {{ name: \"x\" }}: {errors:?}"
        );
    }

    /// AT-1011: HIR emits UnusedStrategyHole warning for declared-but-unreferenced holes.
    #[test]
    fn at_1011_hir_warns_on_unused_strategy_hole() {
        let src = "@strategy { x: ?[1], y: ?[2] } @kernel @workgroup(?x, 1, 1) fn k() -> void { return; }";
        let (_, errors, warns) = lower_src(src);
        assert!(errors.is_empty(), "errors: {errors:?}");
        assert!(
            warns.iter().any(|w| matches!(w, HirWarning::UnusedStrategyHole { name, .. } if name == "y")),
            "expected UnusedStrategyHole {{ name: \"y\" }}: {warns:?}"
        );
        // x is referenced, so no warning for x
        assert!(
            !warns.iter().any(|w| matches!(w, HirWarning::UnusedStrategyHole { name, .. } if name == "x")),
            "should NOT warn about x (it is referenced): {warns:?}"
        );
    }

    /// AT-1012: HIR whitelist accepts 'strategy' annotation name.
    #[test]
    fn at_1012_hir_whitelist_accepts_strategy() {
        let src = "@strategy { x: ?[1] } @kernel @workgroup(?x, 1, 1) fn k() -> void { return; }";
        let (_, errors, _) = lower_src(src);
        // Should NOT produce UnknownAnnotationInM0
        assert!(
            !errors.iter().any(|e| matches!(e, HirError::UnknownAnnotationInM0 { name, .. } if name == "strategy")),
            "should NOT produce UnknownAnnotationInM0 for @strategy: {errors:?}"
        );
    }

    /// AT-1051: HirWarning::StrategyHoleShadowsKernelParam fires when hole name
    /// equals a kernel parameter name.
    #[test]
    fn at_1051_hir_warns_on_hole_shadows_param() {
        // @strategy declares hole `n`, kernel has param `n: u32`.
        let src = "@strategy { n: ?[1024] } @kernel @workgroup(?n, 1, 1) fn saxpy(n: u32) -> void { return; }";
        let (_, errors, warns) = lower_src(src);
        // Should produce the shadow warning (non-fatal)
        assert!(
            warns.iter().any(|w| matches!(w, HirWarning::StrategyHoleShadowsKernelParam { name, .. } if name == "n")),
            "expected StrategyHoleShadowsKernelParam {{ name: \"n\" }}: {warns:?}"
        );
        // Should NOT produce an error (just a warning)
        let _ = errors; // might have other errors; this test only checks the warning
    }

    // HIR defense-in-depth: empty strategy block emits EmptyStrategyBlock warning.
    #[test]
    fn hir_warning_empty_strategy_block() {
        // Use call form for empty strategy: @strategy()
        // Note: the sugar @strategy{} with no args requires a different parser path;
        // we use the call form for simplicity.
        let src = "@strategy() @kernel @workgroup(1, 1, 1) fn k() -> void { return; }";
        let (_, errors, warns) = lower_src(src);
        let _ = errors; // don't care about other errors
        assert!(
            warns.iter().any(|w| matches!(w, HirWarning::EmptyStrategyBlock { .. })),
            "expected EmptyStrategyBlock warning: {warns:?}"
        );
    }

    // ── M3.17 (FG.4): @precondition/@postcondition debug-check tests ────────────

    /// AT-2861 (parser no-change anchor): `@precondition(gt(n,0))` parses to
    /// `Call{gt,[Ident(n),Int(0)]}` with EXISTING tokens (no new AnnotationArg
    /// variant, no new token kind); `@precondition(true)` still parses.
    #[test]
    fn at_2861_precondition_call_form_parses_with_existing_grammar() {
        let src = "@kernel @workgroup(1,1,1) @precondition(gt(n, 0)) fn k(n: u32) -> void { return; }";
        let (ast, lex_errs, parse_errs) = axc_parser::parse(src);
        assert!(lex_errs.is_empty(), "lex errors: {lex_errs:?}");
        assert!(parse_errs.is_empty(), "parse errors: {parse_errs:?}");
        let axc_parser::ast::Item::Kernel(kd) = &ast.items[0].node;
        let ann = kd.annotations.iter().find(|a| a.node.name.node == "precondition").expect("precondition annotation");
        match &ann.node.args[0].node {
            AnnotationArg::Call { name, args } => {
                assert_eq!(name, "gt");
                assert_eq!(args.len(), 2);
                assert!(matches!(&args[0].node, AnnotationArg::Ident(n) if n == "n"));
                assert!(matches!(&args[1].node, AnnotationArg::Int(0)));
            }
            other => panic!("expected Call{{gt,[Ident(n),Int(0)]}}; got {other:?}"),
        }

        // @precondition(true) still parses via the existing Bool variant.
        let src2 = "@kernel @workgroup(1,1,1) @precondition(true) fn k() -> void { return; }";
        let (_, lex_errs2, parse_errs2) = axc_parser::parse(src2);
        assert!(lex_errs2.is_empty() && parse_errs2.is_empty(), "@precondition(true) must still parse");
    }

    /// AT-2862 (HIR lower): the comparison whitelist lowers pre + post to structured
    /// `DebugCheck` nodes with distinct bits; `@postcondition(...)` is recognized (no
    /// longer `UnknownAnnotationInM0`).
    #[test]
    fn at_2862_precondition_postcondition_lower_with_distinct_bits() {
        let src = concat!(
            "@kernel @workgroup(1,1,1) @intent(\"t\") @complexity(O(n)) ",
            "@precondition(gt(n, 0)) @postcondition(lt(n, 1000)) ",
            "fn k(n: u32) -> void { return; }",
        );
        let (hir, errors, _warns) = lower_src(src);
        assert!(errors.is_empty(), "errors: {errors:?}");
        let checks = &hir.kernels[0].annotations.debug_checks;
        assert_eq!(checks.len(), 2, "expected 1 pre + 1 post: {checks:?}");
        let pre = checks.iter().find(|c| c.kind == DebugCheckKind::Pre).expect("pre check");
        let post = checks.iter().find(|c| c.kind == DebugCheckKind::Post).expect("post check");
        assert_eq!(pre.bit, 0);
        assert_eq!(post.bit, 0, "pre and post bits are counted independently per flag word");
        assert_eq!(pre.op, DebugCompareOp::Gt);
        assert_eq!(post.op, DebugCompareOp::Lt);
    }

    /// AT-2863 (HIR reject): buffer operand in a precondition →
    /// `PreconditionOperandNotScalar`; a non-whitelisted call → `UnsupportedDebugPredicate`;
    /// 33 conditions of one kind → `TooManyDebugConditions`.
    #[test]
    fn at_2863_debug_predicate_rejections() {
        // Buffer operand in a precondition.
        let src1 = "@kernel @workgroup(1,1,1) @precondition(gt(elem(buf), 0)) fn k(buf: buffer[u32]) -> void { return; }";
        let (_, errors1, _) = lower_src(src1);
        assert!(
            errors1.iter().any(|e| matches!(e, HirError::PreconditionOperandNotScalar { .. })),
            "expected PreconditionOperandNotScalar: {errors1:?}"
        );

        // Non-whitelisted call.
        let src2 = "@kernel @workgroup(1,1,1) @precondition(foo(n)) fn k(n: u32) -> void { return; }";
        let (_, errors2, _) = lower_src(src2);
        assert!(
            errors2.iter().any(|e| matches!(e, HirError::UnsupportedDebugPredicate { name, .. } if name == "foo")),
            "expected UnsupportedDebugPredicate{{name:\"foo\"}}: {errors2:?}"
        );

        // 33 preconditions of one kind.
        let anns: String = (0..33).map(|_| "@precondition(gt(n, 0)) ".to_string()).collect();
        let src3 = format!("@kernel @workgroup(1,1,1) {anns}fn k(n: u32) -> void {{ return; }}");
        let (_, errors3, _) = lower_src(&src3);
        assert!(
            errors3.iter().any(|e| matches!(e, HirError::TooManyDebugConditions { kind: DebugCheckKind::Pre, got: 33, .. })),
            "expected TooManyDebugConditions{{kind:Pre,got:33}}: {errors3:?}"
        );
    }

    /// AT-2864 (HIR warn): `@postcondition(ge(elem(out),0))` → `PostconditionNotLowerable`,
    /// compile succeeds, no check emitted (v1 never lowers buffer postconditions —
    /// defer-list §9.1, which also removes AT-2877's dominance-analysis surface).
    #[test]
    fn at_2864_buffer_postcondition_not_lowerable() {
        let src = "@kernel @workgroup(1,1,1) @postcondition(ge(elem(out), 0)) fn k(out: buffer[u32]) -> void { out[gid(0)] = 1u32; return; }";
        let (hir, errors, warns) = lower_src(src);
        assert!(errors.is_empty(), "compilation must succeed: {errors:?}");
        assert!(
            warns.iter().any(|w| matches!(w, HirWarning::PostconditionNotLowerable { buf, .. } if buf == "out")),
            "expected PostconditionNotLowerable{{buf:\"out\"}}: {warns:?}"
        );
        assert!(hir.kernels[0].annotations.debug_checks.is_empty(), "no check must be emitted for the deferred postcondition");
    }

    /// AT-2865 (back-compat): `@precondition(true)` is accepted and emits no
    /// runtime check (empty `debug_checks`, non-empty legacy `preconditions`).
    #[test]
    fn at_2865_precondition_true_back_compat_emits_no_check() {
        let src = "@kernel @workgroup(1,1,1) @precondition(true) fn k() -> void { return; }";
        let (hir, errors, _warns) = lower_src(src);
        assert!(errors.is_empty(), "errors: {errors:?}");
        assert!(hir.kernels[0].annotations.debug_checks.is_empty(), "true precondition must emit no DebugCheck");
        assert_eq!(hir.kernels[0].annotations.preconditions.len(), 1, "legacy PreconditionTrivial marker must still be recorded");
    }

    /// AT-2877 shed (defer-list §9.1): buffer postconditions are ALWAYS deferred in
    /// v1 (never lowered), regardless of whether the write is guarded or
    /// unconditional — this is a stricter, simpler, and still-sound superset of the
    /// r2 spec's conditional-dominance gate (shedding it removes the CRITICAL-2
    /// surface entirely, as the spec's own defer-list explicitly sanctions).
    #[test]
    fn at_2877_shed_guarded_and_unconditional_buffer_postconditions_both_deferred() {
        let guarded = concat!(
            "@kernel @workgroup(1,1,1) @postcondition(is_finite(elem(buf))) ",
            "fn k(n: u32, buf: buffer[f32]) -> void { ",
            "let i: u32 = gid(0); if i < n { buf[i] = 1.0f32; } return; }",
        );
        let (hir_g, errors_g, warns_g) = lower_src(guarded);
        assert!(errors_g.is_empty(), "errors: {errors_g:?}");
        assert!(warns_g.iter().any(|w| matches!(w, HirWarning::PostconditionNotLowerable { .. })));
        assert!(hir_g.kernels[0].annotations.debug_checks.is_empty());

        let unconditional = concat!(
            "@kernel @workgroup(1,1,1) @postcondition(is_finite(elem(buf))) ",
            "fn k(buf: buffer[f32]) -> void { buf[gid(0)] = 1.0f32; return; }",
        );
        let (hir_u, errors_u, warns_u) = lower_src(unconditional);
        assert!(errors_u.is_empty(), "errors: {errors_u:?}");
        assert!(warns_u.iter().any(|w| matches!(w, HirWarning::PostconditionNotLowerable { .. })));
        assert!(hir_u.kernels[0].annotations.debug_checks.is_empty(),
            "v1 shed scope: even an UNCONDITIONAL buffer-element postcondition is never lowered (defer-list §9.1)");
    }
}
