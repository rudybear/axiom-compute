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
    KernelBody,
    PORTABLE_MIN_WORKGROUP_INVOCATIONS,
    DESKTOP_MAX_WORKGROUP_INVOCATIONS,
};
use crate::typecheck::TypecheckError;
use crate::param::BindingPlanError;
use crate::shared::{PORTABLE_MIN_SHARED_BYTES, MAX_SHARED_BYTES};

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
    #[error("unknown annotation `@{name}` (M0 allows: @kernel, @workgroup, @intent, @complexity, @precondition, @postcondition, @strict, @subgroup_uniform, @cooperative_matrix)")]
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

    /// M2.1: cooperative-matrix type used as a kernel parameter (not allowed).
    ///
    /// Cooperative-matrix values are function-local only in M2.1.
    #[error("cooperative-matrix type cannot appear as a kernel parameter (`{param_name}`) in M2.1; matrix values are function-local only")]
    UnsupportedCoopMatrixAsParamInM2_1 {
        param_name: String,
        #[label("here")]
        span: Span,
    },

    // ── M2.3: strategy-hole HIR errors ──────────────────────────────────────

    /// A `?name` HoleRef in an annotation references a hole name that was not
    /// declared in `@strategy { name: ?[...] }` on this kernel.
    #[error("unresolved strategy hole `?{name}`; declare it in `@strategy {{ {name}: ?[...] }}` on this kernel")]
    UndefinedStrategyHole {
        name: String,
        #[label("referenced here")]
        span: Span,
    },

    /// `@strategy` declares the same hole name twice.
    ///
    /// Duplicate detection is always active even though the curly-brace sugar
    /// `{ k: ?[...], k: ?[...] }` cannot naturally express duplicates — the
    /// explicit Call form `@strategy(k(?[1]), k(?[2]))` can.
    #[error("strategy hole `{name}` is declared more than once in `@strategy`")]
    DuplicateStrategyHoleName {
        name: String,
        #[label("duplicate declaration here")]
        span: Span,
    },

    /// Defense-in-depth: a hole's candidate list is empty at HIR time.
    ///
    /// The parser already rejects this with `EmptyHoleCandidateList`; this HIR
    /// error catches direct AST construction in tests and other code paths.
    #[error("strategy hole `{name}` has an empty candidate list; provide at least one value")]
    StrategyHoleEmptyCandidates {
        name: String,
        #[label("here")]
        span: Span,
    },

    /// Defense-in-depth: a hole candidate is non-positive.
    ///
    /// The parser already rejects this with `NonPositiveStrategyCandidate`; this
    /// HIR error catches direct AST construction in tests.
    #[error("strategy hole `{name}` candidate {value} is not positive (>= 1)")]
    StrategyHoleNonPositive {
        name: String,
        value: i64,
        #[label("here")]
        span: Span,
    },

    /// A `?[...]` (HoleRef) appears inside a `@strategy { ... }` block itself —
    /// recursive holes are not supported.
    #[error("a HoleRef `?{name}` may not appear inside a @strategy block (recursive holes are not allowed)")]
    HoleRefInStrategyBlock {
        name: String,
        #[label("here")]
        span: Span,
    },

    // ── M3.2: shared[T,N] workgroup-memory errors ────────────────────────────

    /// `shared[T, 0]` — length must be at least 1.
    #[error("shared array `{name}` has length 0; N must be at least 1")]
    SharedZeroLength {
        name: String,
        #[label("here")]
        span: Span,
    },

    /// `shared[T, N]` with N > MAX_SHARED_ELEMS (65536).
    #[error("shared array `{name}` length {len} exceeds maximum {max} elements")]
    SharedTooLarge {
        name: String,
        len: u32,
        max: u32,
        #[label("here")]
        span: Span,
    },

    /// `shared[bool, N]` — Bool is not allowed as shared array element type.
    #[error("shared array `{name}`: element type `{ty_name}` is not allowed (Bool has no stable Vulkan memory representation; use i32/u32 instead)")]
    SharedElementTypeUnsupported {
        name: String,
        ty_name: String,
        #[label("here")]
        span: Span,
    },

    /// Two shared arrays in the same kernel body have the same name.
    #[error("duplicate shared array name `{name}` in kernel body")]
    SharedDuplicateName {
        name: String,
        #[label("duplicate here")]
        span: Span,
        #[label("original declaration")]
        original_span: Span,
    },

    /// `shared[T, N]` used as a kernel parameter type — not allowed.
    #[error("shared-array type `shared[T, N]` cannot be used as a kernel parameter (`{param_name}`); shared arrays are kernel-local and consume no descriptor")]
    SharedTypeNotAllowedAsParam {
        param_name: String,
        #[label("here")]
        span: Span,
    },

    /// `tile[index]` where `index` is not `U32`.
    ///
    /// Part of anti-pattern #1 compliance — no silent coercion to U32.
    #[error("shared array index must be `u32`; got `{got}` (no implicit coercion — anti-pattern #1)")]
    SharedIndexNotU32 {
        got: String,
        #[label("here")]
        span: Span,
    },

    /// `tile[i] = value;` where `value` type does not match the element type.
    #[error("shared array `{name}` element type is `{expected}`; got `{got}` (no implicit conversion — exact match required)")]
    SharedWriteTypeMismatch {
        name: String,
        expected: String,
        got: String,
        #[label("here")]
        span: Span,
    },

    /// `shared` name collides with a kernel parameter or local binding name.
    #[error("shared array name `{name}` collides with an existing parameter or binding")]
    SharedNameCollision {
        name: String,
        #[label("collision here")]
        span: Span,
    },

    /// Shared array name used but not declared.
    #[error("shared array `{name}` is not declared (did you forget `shared name: shared[T, N];`?)")]
    SharedNotDeclared {
        name: String,
        #[label("here")]
        span: Span,
    },

    /// A SharedRead in the same basic block with no barrier accesses a PROVABLY DIFFERENT
    /// slot than any of the prior SharedWrites since the last barrier.
    ///
    /// This is the sound provable-cross-slot hard error (OQ1, r3 — SET-based):
    /// the read index is ProvablyDisequal to EVERY prior write index in W_X.
    /// A ProvablyEqual self-read emits NO diagnostic; unknown index emits advisory warning.
    ///
    /// Note: the diagnostic name says "missing barrier" because the correct fix is to add
    /// a `workgroup_barrier()` between the cross-invocation write and read phases.
    #[error("shared array `{name}`: a read at index {read_index_desc} follows writes at {write_indices_desc} with no barrier between them; this invocation provably reads a slot written by a DIFFERENT invocation (cross-invocation RAW hazard — add workgroup_barrier() between write and read phases)")]
    SharedMissingBarrierBeforeCrossInvocationRead {
        name: String,
        read_index_desc: String,
        write_indices_desc: String,
        #[label("here")]
        span: Span,
    },

    /// `workgroup_barrier()` inside an `if`/`else` body (conditional_depth > 0).
    ///
    /// Barriers in non-uniform control flow are UB in Vulkan — not all invocations
    /// will reach the barrier, causing the hardware to deadlock or produce incorrect results.
    ///
    /// Barriers inside `for`-range or `while` loop bodies ARE permitted (conditional_depth 0).
    #[error("workgroup_barrier() inside an if/else body is undefined behavior (not all invocations provably enter); move the barrier outside the conditional or restructure the kernel")]
    BarrierInDivergentContext {
        #[label("barrier here")]
        span: Span,
    },

    /// Aggregate shared bytes exceed the compiler's static maximum (65536 bytes).
    ///
    /// This is a compile-time ceiling. At runtime, `SharedMemoryExceedsDeviceLimit`
    /// may fire even below this threshold if the device's `maxComputeSharedMemorySize`
    /// is smaller (e.g. a 16 KiB mobile GPU).
    #[error("kernel uses {total_bytes} bytes of shared memory (sum of all shared arrays), exceeding the compile-time maximum of 65536 bytes; reduce shared array sizes")]
    SharedMemoryTooLarge {
        total_bytes: u64,
        #[label("here")]
        span: Span,
    },

    // ── M3.17 (FG.4): runtime debug-check errors ─────────────────────────────

    /// A `@precondition` operand references a buffer element (`elem(buf)`).
    /// Preconditions are thread-uniform-only in v1 — a hard rejection (unlike a
    /// postcondition buffer operand, which is a deferred `PostconditionNotLowerable`).
    #[error("@precondition operand references a buffer (`elem(...)`) — preconditions are thread-uniform-only in M3.17 v1; use a scalar push-constant parameter instead")]
    PreconditionOperandNotScalar {
        #[label("here")]
        span: Span,
    },

    /// A `@precondition`/`@postcondition` predicate is not in the v1 whitelist, or
    /// an operand is malformed (wrong arity, non-scalar/non-literal, ambiguous
    /// literal-only comparison with no type-bearing operand, etc.).
    #[error("unsupported debug-check predicate `{name}` (M3.17 v1 whitelist: gt/ge/lt/le/eq/ne over u32/i32/f32 scalar params or integer literals for @precondition/@postcondition; is_finite(elem(buf)) for @postcondition only)")]
    UnsupportedDebugPredicate {
        name: String,
        #[label("here")]
        span: Span,
    },

    /// More than 32 lowerable `@precondition` (or `@postcondition`) conditions were
    /// declared on one kernel — the flag word only has 32 bits.
    #[error("kernel declares {got} {kind:?} debug conditions, exceeding the 32-bit flag-word capacity")]
    TooManyDebugConditions {
        kind: crate::hir::DebugCheckKind,
        got: usize,
        #[label("here")]
        span: Span,
    },

    // ── M3.19 (FG.3): @optimization_log block-level errors ──────────────────

    /// Block-level `@optimization_log { ... }` malformation: missing/duplicate/
    /// negative/non-`Int` `version`, or (for `version == SUPPORTED_OPT_LOG_VERSION`)
    /// an unknown block-level key.
    #[error("malformed @optimization_log block: {detail}")]
    BadOptimizationLog {
        detail: String,
        #[label("here")]
        span: Span,
    },

    /// One `entry { ... }` malformation: missing/duplicate/wrong-shape field,
    /// out-of-set `unit`/`verdict`, a negative `metric` under a non-negative-only
    /// unit, more than `MAX_OPT_LOG_ENTRIES` entries, or (for `version ==
    /// SUPPORTED_OPT_LOG_VERSION`) an unknown entry-level key.
    #[error("malformed @optimization_log entry: {detail}")]
    BadOptimizationLogEntry {
        detail: String,
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

    /// A collective subgroup op (Elect, All, Any, Reduce, BroadcastFirst) appears
    /// inside a divergent context (if/else/while body).
    ///
    /// Non-fatal in M1.4. M1.5 may promote to error with uniform analysis.
    /// `workgroup_barrier` does NOT trigger this warning in M1.4 (CRITICAL-4 fix).
    SubgroupOpInDivergentContext {
        op_name: &'static str,
        span: Span,
    },

    // ── M2.3: strategy-hole HIR warnings ─────────────────────────────────

    /// A hole was declared in `@strategy` but never referenced via `?name` in
    /// any other annotation on this kernel. Non-fatal — the user may be staging
    /// an upcoming edit or reserving the name for a future annotation.
    UnusedStrategyHole {
        name: String,
        span: Span,
    },

    /// `@strategy {}` with no hole declarations — the block is a no-op.
    EmptyStrategyBlock {
        span: Span,
    },

    /// A strategy hole name matches a kernel parameter name (e.g. `?n` and `n: u32`).
    ///
    /// The names live in disjoint namespaces (`?n` vs bare `n`), so this is not
    /// an error. However, it can be confusing to readers.
    StrategyHoleShadowsKernelParam {
        name: String,
        span: Span,
    },

    /// M3.1: `@cooperative_matrix` flag is set but the kernel body contains no
    /// `coopmat_mul_add` op. The `coop_matrix` field in `KernelAnnotations` will be
    /// `None`; the kernel is treated as non-coopmat at runtime.
    CoopMatFlagWithoutOps {
        kernel_name: String,
    },

    // ── M3.2: shared-memory warnings ────────────────────────────────────────

    /// Aggregate shared bytes exceed the Vulkan portable minimum (16 KiB) but
    /// stay within the compile-time ceiling (65536 bytes).
    ///
    /// The kernel will fail preflight on devices with `maxComputeSharedMemorySize < total_bytes`
    /// (e.g. some mobile GPUs). Use `@target` annotations (future) or reduce shared array sizes.
    SharedMemoryExceedsPortableMinimum {
        total_bytes: u64,
        min_bytes: u32,
        span: Span,
    },

    /// A SharedWrite of id X is followed (in the same basic block, no intervening barrier)
    /// by a SharedRead of the SAME id X, and the index relation is UNDECIDABLE
    /// (the read index is neither provably equal NOR provably disequal to all prior write
    /// indices — e.g. dynamic/arithmetic index).
    ///
    /// Advisory only: cross-invocation aliasing through dynamic indices is undecidable
    /// in general. Add `workgroup_barrier()` if cross-invocation visibility is required.
    SharedWriteWithoutBarrierBeforeRead {
        name: String,
        span: Span,
    },

    /// M3.17 (FG.4): a `@postcondition(...)` predicate referenced a buffer element
    /// (`elem(buf)` or `is_finite(elem(buf))`). v1 never lowers buffer postconditions
    /// (defer-list §9.1, which also removes the CRITICAL-2 dominance-analysis surface) —
    /// the condition is dropped, compilation succeeds, and no runtime check is emitted.
    PostconditionNotLowerable {
        buf: String,
        reason: String,
        span: Span,
    },

    // ── M3.19 (FG.3): @optimization_log warnings ─────────────────────────────

    /// `@optimization_log { version: N }` with zero entries — the block is a
    /// no-op (mirrors `EmptyStrategyBlock`).
    EmptyOptimizationLog {
        span: Span,
    },

    /// §1.4a forward-compat: a `version > SUPPORTED_OPT_LOG_VERSION` block or
    /// entry carried a key unknown to this compiler; it was preserved into
    /// `forward_unknown` rather than rejected (an old compiler must not
    /// hard-fail on a newer, additively-extended block).
    UnknownOptimizationLogKeyForward {
        version: i64,
        key: String,
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

        // M3.2: Validate aggregate shared memory size.
        if let KernelBody::Typed(ref tb) = kernel.body {
            let total_bytes: u64 = tb.shared.iter()
                .map(|s| s.ty.total_byte_size())
                .sum();
            if total_bytes > MAX_SHARED_BYTES {
                errors.push(HirError::SharedMemoryTooLarge {
                    total_bytes,
                    span: kernel.span,
                });
            } else if total_bytes > u64::from(PORTABLE_MIN_SHARED_BYTES) {
                warnings.push(HirWarning::SharedMemoryExceedsPortableMinimum {
                    total_bytes,
                    min_bytes: PORTABLE_MIN_SHARED_BYTES,
                    span: kernel.span,
                });
            }
        }
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
                    cooperative_matrix: false,
                    coop_matrix: None,
                    strategy: None,
                    debug_checks: Vec::new(),
                    opt_log: None,
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
        if let HirWarning::WorkgroupExceedsPortableFloor { product, .. } = &warnings[0] {
            assert_eq!(*product, 1024);
        } else {
            panic!("expected WorkgroupExceedsPortableFloor warning, got: {:?}", &warnings[0]);
        }

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

    // ── M1.3 validate.rs tests ────────────────────────────────────────────────

    use axc_parser::parse;
    use crate::lower::lower_module;
    use crate::typecheck::TypecheckError;

    /// Run full pipeline (parse → lower → validate) and return HirErrors.
    fn pipeline_errors(src: &str) -> Vec<HirError> {
        let (ast, lex_errs, parse_errs) = parse(src);
        assert!(lex_errs.is_empty(), "lex errors: {lex_errs:?}");
        assert!(parse_errs.is_empty(), "parse errors: {parse_errs:?}");
        let (_, errors, _) = lower_module(&ast);
        errors
    }

    #[test]
    fn validate_break_outside_loop_rejected_end_to_end() {
        // Full pipeline: bare `break;` outside any loop must produce
        // exactly one HirError wrapping TypecheckError::BreakOutsideLoop.
        let errors = pipeline_errors(
            "@kernel @workgroup(1,1,1) fn k() -> void { break; return; }",
        );
        let found = errors.iter().any(|e| {
            if let HirError::Typecheck(tc_err) = e {
                matches!(tc_err, TypecheckError::BreakOutsideLoop { .. })
            } else {
                false
            }
        });
        assert!(found, "expected HirError::Typecheck(BreakOutsideLoop): {errors:?}");
    }

    #[test]
    fn validate_continue_outside_loop_rejected_end_to_end() {
        // Full pipeline: bare `continue;` outside any loop must produce
        // exactly one HirError wrapping TypecheckError::ContinueOutsideLoop.
        let errors = pipeline_errors(
            "@kernel @workgroup(1,1,1) fn k() -> void { continue; return; }",
        );
        let found = errors.iter().any(|e| {
            if let HirError::Typecheck(tc_err) = e {
                matches!(tc_err, TypecheckError::ContinueOutsideLoop { .. })
            } else {
                false
            }
        });
        assert!(found, "expected HirError::Typecheck(ContinueOutsideLoop): {errors:?}");
    }

    // ── M1.4 validate tests (AT-14.5) ────────────────────────────────────────

    // AT-424: subgroup_invocation_id() in non-divergent context — no HirError, no warning
    #[test]
    fn validate_sg_invocation_id_no_error_no_warning() {
        use crate::lower_module;
        let src = "@kernel @workgroup(64,1,1) fn k() -> void { let id: u32 = subgroup_invocation_id(); return; }";
        let (ast, lex_errs, parse_errs) = axc_parser::parse(src);
        assert!(lex_errs.is_empty());
        assert!(parse_errs.is_empty());
        let (hir, hir_errs, hir_warns) = lower_module(&ast);
        assert!(hir_errs.is_empty(), "errors: {hir_errs:?}");
        assert!(
            !hir_warns.iter().any(|w| matches!(w, HirWarning::SubgroupOpInDivergentContext { .. })),
            "must not produce divergent warning in non-divergent context; warns: {hir_warns:?}"
        );
        let _ = hir;
    }

    // AT-425: subgroup_any() inside if body — produces SubgroupOpInDivergentContext warning end-to-end
    #[test]
    fn validate_sg_any_in_if_body_produces_warning() {
        use crate::lower_module;
        let src = "@kernel @workgroup(64,1,1) fn k() -> void { let p: bool = true; if p { let r: bool = subgroup_any(p); } return; }";
        let (ast, lex_errs, parse_errs) = axc_parser::parse(src);
        assert!(lex_errs.is_empty());
        assert!(parse_errs.is_empty());
        let (_hir, hir_errs, hir_warns) = lower_module(&ast);
        assert!(hir_errs.is_empty(), "errors: {hir_errs:?}");
        assert!(
            hir_warns.iter().any(|w| matches!(w, HirWarning::SubgroupOpInDivergentContext { op_name, .. } if *op_name == "subgroup_any")),
            "expected SubgroupOpInDivergentContext{{op_name:\"subgroup_any\"}}; warns: {hir_warns:?}"
        );
    }

    // AT-426: subgroup_reduce_add with wrong arg count is a hard error end-to-end
    #[test]
    fn validate_sg_reduce_add_arity_error_is_fatal() {
        use crate::lower_module;
        let src = "@kernel @workgroup(32,1,1) fn k() -> void { let a: i32 = 1i32; let b: i32 = 2i32; let r: i32 = subgroup_reduce_add(a, b); return; }";
        let (ast, lex_errs, parse_errs) = axc_parser::parse(src);
        assert!(lex_errs.is_empty());
        assert!(parse_errs.is_empty());
        let (_hir, hir_errs, _hir_warns) = lower_module(&ast);
        assert!(
            hir_errs.iter().any(|e| matches!(e, HirError::Typecheck(TypecheckError::SubgroupArity { .. }))),
            "expected HirError::Typecheck(SubgroupArity): {hir_errs:?}"
        );
    }

    // ── M2.1 acceptance tests ─────────────────────────────────────────────────

    /// AT-609: HIR typechecker rejects `matrix[bf16, 16, 16, a]` element type
    /// with CoopMatrixElementTypeUnsupported.
    #[test]
    fn tc_coopmat_type_bf16_element_rejected() {
        // bf16 is accepted by the parser (ScalarTypeRef::Bf16) but rejected by the HIR.
        let errors = pipeline_errors(
            "@kernel @workgroup(1,1,1) fn k() -> void { \
             let m: matrix[bf16, 16, 16, a] = coopmat_zero(); return; }",
        );
        assert!(
            errors.iter().any(|e| matches!(
                e,
                HirError::Typecheck(TypecheckError::CoopMatrixElementTypeUnsupported { ty, .. })
                    if *ty == "bf16"
            )),
            "expected CoopMatrixElementTypeUnsupported {{ ty: \"bf16\" }}; got: {errors:?}"
        );
    }

    /// AT-610: HIR rejects cooperative-matrix type used as a kernel parameter.
    ///
    /// Cooperative-matrix values are function-local only in M2.1.
    #[test]
    fn tc_coopmat_as_kernel_param_rejected() {
        let errors = pipeline_errors(
            "@kernel @workgroup(1,1,1) fn k(m: matrix[f16, 16, 16, a]) -> void { return; }",
        );
        assert!(
            errors.iter().any(|e| matches!(
                e,
                HirError::UnsupportedCoopMatrixAsParamInM2_1 { param_name, .. }
                    if param_name == "m"
            )),
            "expected UnsupportedCoopMatrixAsParamInM2_1 {{ param_name: \"m\" }}; got: {errors:?}"
        );
    }
}
