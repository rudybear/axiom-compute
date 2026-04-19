//! Lowering from `axc_parser::ast::Module` to `hir::Module`.
//!
//! Maps annotation argument lists into structured HIR types.
//! Overflow-safe: u32 dimensions are range-checked before conversion.
//! `@complexity` is mapped to `ComplexityForm` enum values (never a String).

use axc_lexer::Span;
use axc_parser::ast::{Module as AstModule, Item, AnnotationArg};
use crate::hir::{
    Module as HirModule, Kernel, KernelId, KernelAnnotations, WorkgroupDims,
    KernelBody, ComplexityForm, ComplexityVar, PreconditionTrivial,
};
use crate::validate::{HirError, HirWarning};

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

                kernels.push(Kernel {
                    id: KernelId(next_id),
                    name: kd.name.node.clone(),
                    annotations,
                    body: KernelBody::Empty,
                    span: item.span,
                });
                next_id += 1;
            }
        }
    }

    (HirModule { kernels }, errors, warnings)
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
}
