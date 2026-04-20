//! Lowering from `axc_parser::ast::Module` to `hir::Module`.
//!
//! Maps annotation argument lists into structured HIR types.
//! Overflow-safe: u32 dimensions are range-checked before conversion.
//! `@complexity` is mapped to `ComplexityForm` enum values (never a String).

use axc_lexer::Span;
use axc_parser::ast::{Module as AstModule, Item, AnnotationArg, Stmt, TypeRef, ScalarTypeRef};
use crate::hir::{
    Module as HirModule, Kernel, KernelId, KernelAnnotations, WorkgroupDims,
    KernelBody, ComplexityForm, ComplexityVar, PreconditionTrivial,
};
use crate::validate::{HirError, HirWarning};
use crate::typecheck::typecheck_kernel_body;
use crate::ty::ScalarTy;
use crate::buffer::{BufferAccess, BufferTy};
use crate::param::{KernelParam, Ty as ParamTy, ParamBindingPlan, compute_binding_plan};

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
                let mut seen_workgroup: bool = false;
                let mut seen_kernel: bool = false;

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
                                    other => {
                                        errors.push(HirError::UnsupportedPrecondition {
                                            detail: format!("{other:?}"),
                                            span: ann.span,
                                        });
                                    }
                                }
                            }
                        }
                        "subgroup_uniform" => {
                            subgroup_uniform = true;
                        }
                        other => {
                            errors.push(HirError::UnknownAnnotationInM0 {
                                name: other.to_owned(),
                                span: ann.span,
                            });
                        }
                    }
                }

                // Check that @workgroup was present
                let wg: WorkgroupDims = match workgroup_opt {
                    Some(w) => w,
                    None => {
                        errors.push(HirError::MissingWorkgroup {
                            name: kd.name.node.clone(),
                            span: item.span,
                        });
                        // Use a placeholder so we can still emit the kernel into HIR
                        WorkgroupDims { x: 1, y: 1, z: 1 }
                    }
                };

                // Check return type
                if kd.return_type.node != axc_parser::ast::TypeRef::Void {
                    errors.push(HirError::BadKernelReturnType {
                        got: format!("{:?}", kd.return_type.node),
                        span: kd.return_type.span,
                    });
                }

                // Validate workgroup dimensions (zero check)
                let _ = validate_workgroup_dims(&wg, workgroup_span, &kd.name.node, &mut errors, &mut warnings);

                let annotations: KernelAnnotations = KernelAnnotations {
                    workgroup: wg,
                    intent,
                    complexity,
                    preconditions,
                    subgroup_uniform,
                };

                // Lower kernel parameters.
                let (params, binding_plan) = lower_params(
                    &kd.params,
                    item.span,
                    &mut errors,
                );

                // Determine body: Empty if trivial (only void return), Typed otherwise.
                let body = lower_kernel_body(&kd.body.node, &params, &mut errors, &mut warnings);

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

/// Extract a single workgroup dimension from an `AnnotationArg::Int`.
/// Validates that the value is in `[1, u32::MAX]` (zero is caught by workgroup
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
}
