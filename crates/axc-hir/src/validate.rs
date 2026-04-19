//! Post-lowering validation rules for the HIR.
//!
//! - Rule M0-V1: every `@kernel` must carry exactly one `@workgroup(X,Y,Z)` with
//!   X, Y, Z >= 1.
//! - Rule M0-V2 (two-tier cap, per spec rev 2 boundary clarification):
//!   - product in [1, 128]: CLEAN
//!   - product in [129, 1024]: WARN (`WorkgroupExceedsPortableFloor`)
//!   - product > 1024: ERROR (`BadWorkgroupDim`)
//! - Rule M0-V3: unknown annotations are rejected (whitelist in `lower.rs`).

use axc_lexer::Span;
use crate::hir::{
    Module as HirModule,
    PORTABLE_MIN_WORKGROUP_INVOCATIONS,
    DESKTOP_MAX_WORKGROUP_INVOCATIONS,
};
use crate::typecheck::TypecheckError;
use crate::param::BindingPlanError;

/// Diagnostic error from HIR validation.
#[derive(Debug, thiserror::Error, miette::Diagnostic)]
pub enum HirError {
    #[error("kernel `{name}` is missing required @workgroup(X,Y,Z) annotation")]
    MissingWorkgroup {
        name: String,
        #[label("kernel defined here")]
        span: Span,
    },
    #[error("@workgroup requires exactly 3 integer arguments (X, Y, Z); got {got}")]
    BadWorkgroupArity {
        got: usize,
        #[label("here")]
        span: Span,
    },
    #[error("@workgroup dimensions must be in [1, {max}]; got ({x},{y},{z}) product={product}. See VkPhysicalDeviceLimits::maxComputeWorkGroupInvocations.")]
    BadWorkgroupDim {
        x: i64,
        y: i64,
        z: i64,
        product: u64,
        max: u32,
        #[label("here")]
        span: Span,
    },
    #[error("@workgroup dimension value {got} does not fit in u32")]
    WorkgroupDimOverflow {
        got: i64,
        #[label("here")]
        span: Span,
    },
    #[error("duplicate annotation `@{name}` on kernel `{kernel}`")]
    DuplicateAnnotation {
        name: String,
        kernel: String,
        #[label("duplicate here")]
        span: Span,
    },
    #[error("unknown annotation `@{name}` (M0 allows: @kernel, @workgroup, @intent, @complexity, @precondition, @subgroup_uniform)")]
    UnknownAnnotationInM0 {
        name: String,
        #[label("here")]
        span: Span,
    },
    #[error("return type of @kernel must be `void` in M0; got `{got}`")]
    BadKernelReturnType {
        got: String,
        #[label("here")]
        span: Span,
    },
    #[error("@precondition argument is not a literal `true` (M0 limitation): {detail}")]
    UnsupportedPrecondition {
        detail: String,
        #[label("here")]
        span: Span,
    },
    #[error("@complexity form is not supported in M0 (accepted: O(1), O(n), Theta(n), Omega(n); O(n^2) is reserved for M1 once exponent grammar lands)")]
    UnsupportedComplexityInM0 {
        #[label("here")]
        span: Span,
    },

    /// A typechecking error from the M1.1 kernel body.
    #[error(transparent)]
    #[diagnostic(transparent)]
    Typecheck(TypecheckError),

    /// A binding plan error (push constant overflow or unsupported param type).
    #[error(transparent)]
    #[diagnostic(transparent)]
    BindingPlan(BindingPlanError),

    /// Unsupported parameter type (void, bool not allowed as kernel params).
    #[error("unsupported parameter type `{ty_name}` for kernel parameter `{param_name}`")]
    UnsupportedParamType {
        ty_name: String,
        param_name: String,
        #[label("here")]
        span: Span,
    },
}

/// Non-fatal diagnostic warning from HIR validation.
///
/// Warnings do not block compilation.
#[derive(Debug, Clone)]
pub enum HirWarning {
    /// Workgroup product exceeds the Vulkan 1.1 guaranteed portability floor (128)
    /// but is within the observed desktop-class ceiling (1024).
    ///
    /// This means the kernel may not run on all Vulkan 1.1 devices even though
    /// it is accepted by the compiler. A `@target()` annotation (M1) can gate this.
    WorkgroupExceedsPortableFloor {
        name: String,
        product: u64,
        floor: u32,
        span: Span,
    },
}

/// Run post-lowering validation rules on a `HirModule`.
///
/// Returns `(errors, warnings)` — both lists are populated even if the other is
/// non-empty (collect-all, not short-circuit).
pub fn validate(module: &HirModule) -> (Vec<HirError>, Vec<HirWarning>) {
    let mut errors: Vec<HirError> = Vec::new();
    let mut warnings: Vec<HirWarning> = Vec::new();

    for kernel in &module.kernels {
        let wg = &kernel.annotations.workgroup;

        // Rule M0-V1: dimensions >= 1
        if wg.x < 1 || wg.y < 1 || wg.z < 1 {
            errors.push(HirError::BadWorkgroupDim {
                x: wg.x as i64,
                y: wg.y as i64,
                z: wg.z as i64,
                product: wg.product(),
                max: DESKTOP_MAX_WORKGROUP_INVOCATIONS,
                span: kernel.span,
            });
            continue;
        }

        // Rule M0-V2: two-tier cap
        let product: u64 = wg.product();
        if product > DESKTOP_MAX_WORKGROUP_INVOCATIONS as u64 {
            errors.push(HirError::BadWorkgroupDim {
                x: wg.x as i64,
                y: wg.y as i64,
                z: wg.z as i64,
                product,
                max: DESKTOP_MAX_WORKGROUP_INVOCATIONS,
                span: kernel.span,
            });
        } else if product > PORTABLE_MIN_WORKGROUP_INVOCATIONS as u64 {
            warnings.push(HirWarning::WorkgroupExceedsPortableFloor {
                name: kernel.name.clone(),
                product,
                floor: PORTABLE_MIN_WORKGROUP_INVOCATIONS,
                span: kernel.span,
            });
        }
        // product <= PORTABLE_MIN: clean, no action needed
    }

    (errors, warnings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axc_lexer::Span;
    use crate::hir::{
        Module as HirModule, Kernel, KernelId, KernelAnnotations, WorkgroupDims, KernelBody,
    };

    /// Build a minimal HIR module with one kernel and the given workgroup dims.
    fn make_module(x: u32, y: u32, z: u32) -> HirModule {
        use crate::param::ParamBindingPlan;
        HirModule {
            kernels: vec![Kernel {
                id: KernelId(0),
                name: "test_kernel".into(),
                annotations: KernelAnnotations {
                    workgroup: WorkgroupDims { x, y, z },
                    intent: None,
                    complexity: None,
                    preconditions: Vec::new(),
                    subgroup_uniform: false,
                },
                params: Vec::new(),
                binding_plan: ParamBindingPlan {
                    buffers: Vec::new(),
                    scalars: Vec::new(),
                    push_constant_total_bytes: 0,
                },
                body: KernelBody::Empty,
                span: Span::new(0, 10),
            }],
        }
    }

    // ── AT-5: two-tier workgroup cap (all sub-cases) ──────────────────────────

    #[test]
    fn workgroup_dim_tiers() {
        // (a) product == 0 (zero dimension) → ERROR
        let m = make_module(0, 1, 1);
        let (errors, warnings) = validate(&m);
        assert!(!errors.is_empty(), "(a) expected error for zero dim");
        assert!(warnings.is_empty(), "(a) expected no warnings");

        // (b) @workgroup(8,8,2) product==128 (at portable floor) → CLEAN
        let m = make_module(8, 8, 2);
        let (errors, warnings) = validate(&m);
        assert!(errors.is_empty(), "(b) expected no errors: {errors:?}");
        assert!(warnings.is_empty(), "(b) expected no warnings: {warnings:?}");

        // (c) @workgroup(129,1,1) product==129 (first over floor) → WARN
        let m = make_module(129, 1, 1);
        let (errors, warnings) = validate(&m);
        assert!(errors.is_empty(), "(c) expected no errors: {errors:?}");
        assert!(!warnings.is_empty(), "(c) expected warning for product=129");

        // (d) @workgroup(16,8,2) product==256 → WARN
        let m = make_module(16, 8, 2);
        let (errors, warnings) = validate(&m);
        assert!(errors.is_empty(), "(d) expected no errors: {errors:?}");
        assert!(!warnings.is_empty(), "(d) expected warning for product=256");

        // (e) @workgroup(16,8,8) product==1024 (at desktop ceiling) → WARN (rev-2 boundary fix)
        let m = make_module(16, 8, 8);
        let (errors, warnings) = validate(&m);
        assert!(errors.is_empty(), "(e) expected no errors at product=1024: {errors:?}");
        assert!(!warnings.is_empty(), "(e) expected warning at product=1024 (boundary)");
        let HirWarning::WorkgroupExceedsPortableFloor { product, .. } = &warnings[0];
        assert_eq!(*product, 1024);

        // (f) @workgroup(16,8,9) product==1152 → ERROR
        let m = make_module(16, 8, 9);
        let (errors, _warnings) = validate(&m);
        assert!(errors.iter().any(|e| matches!(e, HirError::BadWorkgroupDim { product, .. } if *product == 1152)),
            "(f) expected BadWorkgroupDim{{product:1152}}: {errors:?}");
    }

    // ── Boundary: product == 128 is the last CLEAN value ─────────────────────

    #[test]
    fn product_128_is_clean_no_warning() {
        let m = make_module(8, 8, 2); // 8*8*2 == 128
        let (errors, warnings) = validate(&m);
        assert!(errors.is_empty(), "errors: {errors:?}");
        assert!(warnings.is_empty(), "expected no warning at product=128");
    }

    // ── Large but valid product (overflow-safe u64 check) ────────────────────

    #[test]
    fn product_over_ceiling_is_error() {
        let m = make_module(1025, 1, 1);
        let (errors, _) = validate(&m);
        assert!(!errors.is_empty());
    }
}
