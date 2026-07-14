//! `HirError` / `HirWarning`: the diagnostic types produced while lowering and
//! typechecking a `HirModule`.
//!
//! M3.22 deleted this module's `pub fn validate()` post-lowering pass: it had
//! ZERO callers workspace-wide (only its own tests), and every check it ran
//! duplicated a REACHABLE equivalent that already fires through the normal
//! compile pipeline:
//! - Rule M0-V1/V2 (workgroup dims `>= 1`, two-tier cap) → `lower.rs`'s
//!   `validate_workgroup_dims`, called from `lower_module`.
//! - Aggregate shared-memory size → `TypecheckError::SharedMemoryTooLarge`,
//!   checked in `typecheck.rs`'s `typecheck_kernel_body`.
//! - Aggregate local-array size → `TypecheckError::LocalArrayTooLarge`, same
//!   placement.
//!
//! (Rule M0-V3, unknown annotations, is enforced by `lower.rs`'s whitelist —
//! it was never part of this module's `validate()` pass either.)

use axc_lexer::Span;
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

    // ── M3.20: array[T,N] local-array errors ─────────────────────────────────
    //
    // `LocalArrayAsParameter` is constructed at lower.rs param-lowering time — a
    // REAL, reachable-via-`lower_module` check (mirrors `SharedTypeNotAllowedAsParam`).
    //
    // The aggregate local-array size check does NOT live here: it is
    // `TypecheckError::LocalArrayTooLarge` in typecheck.rs's `typecheck_kernel_body`
    // (mirrors `TypecheckError::SharedMemoryTooLarge`'s placement) — the SOLE
    // aggregate check, wrapped as `HirError::Typecheck(..)` and reachable through
    // the normal compile pipeline. (M3.22 deleted this module's dead, zero-caller
    // `validate()` pass, which used to duplicate both aggregate checks here.)
    //
    // All other per-declaration/per-use local-array diagnostics live in
    // `TypecheckError` (typecheck.rs) — see that enum's M3.20 section for why.

    /// `array[T, N]` used as a kernel parameter type — not allowed.
    #[error("local-array type `array[T, N]` cannot be used as a kernel parameter (`{param_name}`); local arrays are kernel-local and consume no descriptor")]
    LocalArrayAsParameter {
        param_name: String,
        #[label("here")]
        span: Span,
    },

    // ── M3.21 (FG.9): multi-kernel modules ───────────────────────────────────

    /// Two `@kernel` items in the same module declare the same function name.
    ///
    /// This would collide on both the emitted `OpEntryPoint` name and (for
    /// `axc compile --all`) the per-kernel output file path — fail closed
    /// rather than let the driver silently alias two distinct kernels.
    #[error("duplicate kernel name `{name}`; two @kernel items in this module share the same function name")]
    DuplicateKernelName {
        name: String,
        #[label("duplicate declared here")]
        span: Span,
        #[label("original declaration here")]
        original_span: Span,
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

    // ── M3.20: array[T,N] local-array warnings ───────────────────────────────

    /// Aggregate local-array bytes exceed the non-blocking spill-risk advisory
    /// threshold (1024 bytes) but stay within the compile-time ceiling (4096 bytes).
    ///
    /// A `Function`-storage array the driver cannot keep in registers spills to
    /// off-chip "local memory" (M3.12 register-pressure/occupancy lesson — M3.20
    /// spec §6/§15). Consider `shared[T,N]` for workgroup-cooperative data instead.
    LocalArrayMaySpill {
        total_bytes: u64,
        advisory_bytes: u32,
        span: Span,
    },

    /// A `LocalArrayRead` of array X is reached while X's write-set is EMPTY (the
    /// array has been declared but never written on any path to this read).
    ///
    /// Decidable, zero-FALSE-POSITIVE (never warns on a genuinely-written array).
    /// Honest disclosure (M3.20 spec §5): the write-set is a source-order PRESENCE
    /// set, not a path-sensitive definite-assignment lattice — it can MISS true
    /// positives (e.g. a write nested inside an `if` populates the write-set
    /// unconditionally, so a later read on an uncovered `else` path is NOT
    /// flagged). Uninitialized-read remains UB-by-design on any path; this catches
    /// only the common "forgot to zero it at all" foot-gun.
    LocalArrayReadBeforeAnyWrite {
        name: String,
        span: Span,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

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
