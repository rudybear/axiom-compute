//! Integration tests for `array[T,N]` local-array lower + typecheck (M3.20).
//!
//! Covers AT-2930 (HIR half: param/return-type rejection), AT-2931 (validate
//! errors), AT-2932 (aggregate size cap), AT-2933 (init-hazard advisory),
//! AT-2934 (index/value typing), AT-2942 (bidirectional name collision, both
//! orders), AT-2943 (nested-decl rejection), AT-2944 (const-index OOB).
//!
//! Mirrors `shared_typecheck.rs`'s helpers and style.

use axc_hir::{lower_module, HirError, HirWarning};
use axc_hir::typecheck::TypecheckError;
use axc_parser::parse;

/// Build a kernel source string from body statements.
fn kernel_src(body: &str) -> String {
    format!(
        "@kernel @workgroup(256,1,1) @intent(\"test\") @complexity(O(n))\nfn k(out: buffer[f32]) -> void {{\n{}\n}}",
        body
    )
}

/// Build a kernel source string with a parameter list.
fn kernel_src_with_params(params: &str, body: &str) -> String {
    format!(
        "@kernel @workgroup(256,1,1) @intent(\"test\") @complexity(O(n))\nfn k({}) -> void {{\n{}\n}}",
        params, body
    )
}

/// Compile and lower a kernel source string. Returns (hir_errors, warnings).
fn lower(src: &str) -> (Vec<HirError>, Vec<HirWarning>) {
    let (ast, lex_errs, parse_errs) = parse(src);
    assert!(lex_errs.is_empty(), "lex errors: {lex_errs:?}");
    assert!(parse_errs.is_empty(), "parse errors: {parse_errs:?}");
    let (_, hir_errs, hir_warns) = lower_module(&ast);
    (hir_errs, hir_warns)
}

// ── AT-2930 (HIR half): array[T,N] as parameter / return type ────────────────

/// AT-2930: `array[T,N]` in parameter position -> HirError::LocalArrayAsParameter.
#[test]
fn at2930_local_array_as_parameter_rejected() {
    let src = "@kernel @workgroup(1,1,1) fn k(a: array[u32, 4]) -> void { return; }";
    let (errs, _warns) = lower(src);
    assert!(
        errs.iter().any(|e| matches!(e, HirError::LocalArrayAsParameter { .. })),
        "expected LocalArrayAsParameter; errors: {errs:?}"
    );
}

/// AT-2930: `array[T,N]` as a return type -> BadKernelReturnType (any non-void
/// return type is rejected; array is not special-cased, matching `shared`).
#[test]
fn at2930_local_array_as_return_type_rejected() {
    let src = "@kernel @workgroup(1,1,1) fn k() -> array[u32, 4] { return; }";
    let (errs, _warns) = lower(src);
    assert!(
        errs.iter().any(|e| matches!(e, HirError::BadKernelReturnType { .. })),
        "expected BadKernelReturnType; errors: {errs:?}"
    );
}

// ── AT-2931: validate errors (ZeroLength, ElementTypeNotAllowed, DuplicateName,
//             NameCollision, Undeclared) ─────────────────────────────────────

/// AT-2931: N=0 -> LocalArrayZeroLength.
#[test]
fn at2931_zero_length_rejected() {
    let src = "@kernel @workgroup(1,1,1) fn k() -> void { array a: array[u32, 0]; return; }";
    let (_ast, _lex, parse_errs) = parse(src);
    // N=0 is caught at parse time (parse_local_array_len rejects value <= 0), mirroring shared.
    assert!(!parse_errs.is_empty(), "N=0 must be rejected at parse time; parse_errs: {parse_errs:?}");
}

/// AT-2931: `array[bool,4]` -> LocalArrayElementTypeNotAllowed.
///
/// `bool` is not in the parser's local-array elem-type token set (mirrors shared),
/// so this is caught as a parse error — no `TokenKind::Bool` arm in
/// `parse_local_array_elem_type`.
#[test]
fn at2931_bool_element_rejected_at_parse() {
    let src = "@kernel @workgroup(1,1,1) fn k() -> void { array a: array[bool, 4]; return; }";
    let (_ast, _lex, parse_errs) = parse(src);
    assert!(!parse_errs.is_empty(), "bool element must be rejected at parse time; parse_errs: {parse_errs:?}");
}

/// AT-2931: duplicate local-array name -> LocalArrayDuplicateName.
#[test]
fn at2931_duplicate_name_rejected() {
    let src = kernel_src("array a: array[u32, 4]; array a: array[u32, 8]; return;");
    let (errs, _warns) = lower(&src);
    assert!(
        errs.iter().any(|e| matches!(e, HirError::Typecheck(TypecheckError::LocalArrayDuplicateName { .. }))),
        "expected LocalArrayDuplicateName; errors: {errs:?}"
    );
}

/// AT-2931: local-array name == a param name -> LocalArrayNameCollision.
#[test]
fn at2931_name_collision_with_param_rejected() {
    let src = kernel_src_with_params(
        "hist: buffer[u32]",
        "array hist: array[u32, 4]; return;"
    );
    let (errs, _warns) = lower(&src);
    assert!(
        errs.iter().any(|e| matches!(e, HirError::Typecheck(TypecheckError::LocalArrayNameCollision { .. }))),
        "expected LocalArrayNameCollision (param); errors: {errs:?}"
    );
}

/// AT-2931: local-array name == an existing `let` binding name -> LocalArrayNameCollision.
#[test]
fn at2931_name_collision_with_binding_rejected() {
    let src = kernel_src("let x: u32 = 0u32; array x: array[u32, 4]; return;");
    let (errs, _warns) = lower(&src);
    assert!(
        errs.iter().any(|e| matches!(e, HirError::Typecheck(TypecheckError::LocalArrayNameCollision { .. }))),
        "expected LocalArrayNameCollision (binding); errors: {errs:?}"
    );
}

/// AT-2931: use of an undeclared array. `hist[0u32]` where `hist` was never
/// declared via `array` falls through the local-array/shared/buffer/binding
/// disambiguation chain exactly like an undeclared `shared` name does (mirrors
/// `SharedNotDeclared`'s precedent) — reported as UnknownBinding, not a
/// LocalArray-specific variant, because nothing at the use site distinguishes
/// "meant to be an array" from "meant to be a buffer" for a name that was never
/// declared at all. `LocalArrayUndeclared` itself is defense-in-depth (see its
/// doc comment in typecheck.rs) and is exercised directly at the unit level in
/// `axc-hir/src/typecheck.rs`'s own `#[cfg(test)]` module.
#[test]
fn at2931_reference_to_never_declared_array_name_is_unknown_binding() {
    let src = kernel_src("let _v: u32 = hist[0u32]; return;");
    let (errs, _warns) = lower(&src);
    assert!(
        errs.iter().any(|e| matches!(e, HirError::Typecheck(TypecheckError::UnknownBinding { .. }))),
        "expected UnknownBinding for a name with no array/shared/buffer/binding decl at all; errors: {errs:?}"
    );
}

// ── AT-2932: aggregate byte cap (4096 hard, 1024 advisory) ───────────────────

/// AT-2932: aggregate local-array bytes > 4096 -> LocalArrayTooLarge hard error.
/// `array[f32, 1025]` = 4100 bytes > 4096.
///
/// Reachable via `TypecheckError::LocalArrayTooLarge` (constructed during lowering,
/// mirrors `SharedMemoryTooLarge`'s ACTUAL reachable placement in
/// `typecheck_kernel_body` — the sole check since M3.22 deleted `validate.rs`'s
/// dead, zero-caller `validate()` pass that used to duplicate it).
#[test]
fn at2932_local_array_too_large_hard_error() {
    let src = "@kernel @workgroup(1,1,1) fn k() -> void { array big: array[f32, 1025]; return; }";
    let (errs, _warns) = lower(src);
    assert!(
        errs.iter().any(|e| matches!(e, HirError::Typecheck(TypecheckError::LocalArrayTooLarge { .. }))),
        "expected LocalArrayTooLarge for 4100 bytes; errors: {errs:?}"
    );
}

/// AT-2932: aggregate bytes > 1024 (spill advisory) and <= 4096 -> warning only, no hard error.
/// `array[f32, 512]` = 2048 bytes.
#[test]
fn at2932_local_array_may_spill_warning_only() {
    let src = "@kernel @workgroup(1,1,1) fn k() -> void { array med: array[f32, 512]; return; }";
    let (errs, warns) = lower(src);
    assert!(
        !errs.iter().any(|e| matches!(e, HirError::Typecheck(TypecheckError::LocalArrayTooLarge { .. }))),
        "must NOT have LocalArrayTooLarge for 2048 bytes; errors: {errs:?}"
    );
    assert!(
        warns.iter().any(|w| matches!(w, HirWarning::LocalArrayMaySpill { .. })),
        "expected LocalArrayMaySpill warning; warns: {warns:?}"
    );
}

/// AT-2932: aggregate bytes <= 1024 -> clean, no error, no warning.
/// `array[f32, 8]` = 32 bytes.
#[test]
fn at2932_local_array_small_is_clean() {
    let src = "@kernel @workgroup(1,1,1) fn k() -> void { array small: array[f32, 8]; return; }";
    let (errs, warns) = lower(src);
    assert!(
        !errs.iter().any(|e| matches!(e, HirError::Typecheck(TypecheckError::LocalArrayTooLarge { .. }))),
        "errors: {errs:?}"
    );
    assert!(
        !warns.iter().any(|w| matches!(w, HirWarning::LocalArrayMaySpill { .. })),
        "warns: {warns:?}"
    );
}

// ── AT-2933: init-hazard advisory (read-before-any-write) ────────────────────

/// AT-2933: a LocalArrayRead reached with the array's write-set EMPTY ->
/// LocalArrayReadBeforeAnyWrite advisory.
#[test]
fn at2933_read_before_any_write_warns() {
    let src = kernel_src(
        "array hist: array[u32, 4]; let v: u32 = hist[0u32]; return;"
    );
    let (errs, warns) = lower(&src);
    assert!(errs.is_empty(), "unexpected errors: {errs:?}");
    assert!(
        warns.iter().any(|w| matches!(w, HirWarning::LocalArrayReadBeforeAnyWrite { .. })),
        "expected LocalArrayReadBeforeAnyWrite; warns: {warns:?}"
    );
}

/// AT-2933: write-then-read -> NO warning (zero false positives), even for a
/// symbolic (non-constant) index on both the write and the read.
#[test]
fn at2933_write_then_read_no_warning() {
    let src = kernel_src_with_params(
        "idx: readonly_buffer[u32]",
        "array hist: array[u32, 4]; let i: u32 = idx[0u32]; hist[i] = 1u32; let v: u32 = hist[i]; return;"
    );
    let (errs, warns) = lower(&src);
    assert!(errs.is_empty(), "unexpected errors: {errs:?}");
    assert!(
        !warns.iter().any(|w| matches!(w, HirWarning::LocalArrayReadBeforeAnyWrite { .. })),
        "write-then-read (even symbolic index) must NOT warn; warns: {warns:?}"
    );
}

// ── AT-2934: index/value typing ───────────────────────────────────────────────

/// AT-2934: index of non-U32 type -> LocalArrayIndexNotU32 (no coercion).
#[test]
fn at2934_index_not_u32_rejected() {
    let src = kernel_src(
        "array hist: array[u32, 4]; let i: i32 = 0i32; let v: u32 = hist[i]; return;"
    );
    let (errs, _warns) = lower(&src);
    assert!(
        errs.iter().any(|e| matches!(e, HirError::Typecheck(TypecheckError::LocalArrayIndexNotU32 { .. }))),
        "expected LocalArrayIndexNotU32; errors: {errs:?}"
    );
}

/// AT-2934: write value type != elem type -> LocalArrayWriteTypeMismatch.
#[test]
fn at2934_write_type_mismatch_rejected() {
    let src = kernel_src(
        "array hist: array[u32, 4]; hist[0u32] = 1i32; return;"
    );
    let (errs, _warns) = lower(&src);
    assert!(
        errs.iter().any(|e| matches!(e, HirError::Typecheck(TypecheckError::LocalArrayWriteTypeMismatch { .. }))),
        "expected LocalArrayWriteTypeMismatch; errors: {errs:?}"
    );
}

/// AT-2934: matched index/value types compile clean (happy path).
#[test]
fn at2934_matched_types_clean() {
    let src = kernel_src(
        "array hist: array[u32, 4]; hist[0u32] = 1u32; let v: u32 = hist[0u32]; return;"
    );
    let (errs, _warns) = lower(&src);
    assert!(errs.is_empty(), "unexpected errors: {errs:?}");
}

// ── AT-2942 (r2): bidirectional name-collision cross-check, BOTH orders ──────

/// AT-2942: `shared` declared first, `array` second, same name -> collision (either
/// direction's error variant is acceptable — the guard fires from whichever side
/// registers second).
#[test]
fn at2942_collision_shared_then_array() {
    let src = kernel_src("shared x: shared[u32, 4]; array x: array[u32, 4]; return;");
    let (errs, _warns) = lower(&src);
    let collision = errs.iter().any(|e| matches!(
        e,
        HirError::Typecheck(TypecheckError::LocalArrayNameCollision { .. })
            | HirError::Typecheck(TypecheckError::SharedNameCollision { .. })
    ));
    assert!(collision, "expected a name-collision error (shared-then-array order); errors: {errs:?}");
}

/// AT-2942: `array` declared first, `shared` second, same name -> collision
/// (proves the bidirectional cross-check — `register_shared` MUST also consult
/// `local_array_name_map`, not just the reverse).
#[test]
fn at2942_collision_array_then_shared() {
    let src = kernel_src("array x: array[u32, 4]; shared x: shared[u32, 4]; return;");
    let (errs, _warns) = lower(&src);
    let collision = errs.iter().any(|e| matches!(
        e,
        HirError::Typecheck(TypecheckError::LocalArrayNameCollision { .. })
            | HirError::Typecheck(TypecheckError::SharedNameCollision { .. })
    ));
    assert!(collision, "expected a name-collision error (array-then-shared order); errors: {errs:?}");
}

/// AT-2942: `array` colliding with a param/binding name -> LocalArrayNameCollision
/// (already covered by AT-2931's param/binding tests above; restated here under
/// the AT-2942 banner per the milestone's AT table for direct traceability).
#[test]
fn at2942_array_param_collision_is_local_array_name_collision() {
    let src = kernel_src_with_params(
        "n: u32",
        "array n: array[u32, 4]; return;"
    );
    let (errs, _warns) = lower(&src);
    assert!(
        errs.iter().any(|e| matches!(e, HirError::Typecheck(TypecheckError::LocalArrayNameCollision { .. }))),
        "expected LocalArrayNameCollision for array/param collision; errors: {errs:?}"
    );
}

// ── AT-2943 (r2): nested-decl scope restriction ───────────────────────────────

/// AT-2943: `array` decl inside a `for` loop body -> LocalArrayDeclNotAtBlockScope.
#[test]
fn at2943_nested_decl_in_for_rejected() {
    let src = kernel_src(
        "for i in range(0u32, 4u32) { array a: array[f32, 4]; } return;"
    );
    let (errs, _warns) = lower(&src);
    assert!(
        errs.iter().any(|e| matches!(e, HirError::Typecheck(TypecheckError::LocalArrayDeclNotAtBlockScope { .. }))),
        "expected LocalArrayDeclNotAtBlockScope (for-body); errors: {errs:?}"
    );
}

/// AT-2943: `array` decl inside an `if` block -> LocalArrayDeclNotAtBlockScope.
#[test]
fn at2943_nested_decl_in_if_rejected() {
    let src = kernel_src(
        "let p: bool = true; if p { array a: array[f32, 4]; } return;"
    );
    let (errs, _warns) = lower(&src);
    assert!(
        errs.iter().any(|e| matches!(e, HirError::Typecheck(TypecheckError::LocalArrayDeclNotAtBlockScope { .. }))),
        "expected LocalArrayDeclNotAtBlockScope (if-body); errors: {errs:?}"
    );
}

/// AT-2943: a rejected nested decl still poison-registers the name — an in-block
/// USE of that name resolves as a local-array reference (no additional cascade
/// error beyond the one root LocalArrayDeclNotAtBlockScope).
#[test]
fn at2943_nested_decl_rejection_does_not_cascade() {
    let src = kernel_src(
        "for i in range(0u32, 4u32) { array a: array[f32, 4]; a[0u32] = 1.0f32; } return;"
    );
    let (errs, _warns) = lower(&src);
    let scope_errors: usize = errs.iter()
        .filter(|e| matches!(e, HirError::Typecheck(TypecheckError::LocalArrayDeclNotAtBlockScope { .. })))
        .count();
    assert_eq!(scope_errors, 1, "expected exactly ONE root-cause scope error; errors: {errs:?}");
    let cascade = errs.iter().any(|e| matches!(
        e,
        HirError::Typecheck(TypecheckError::UnknownBinding { .. })
            | HirError::Typecheck(TypecheckError::IndexOnNonBuffer { .. })
    ));
    assert!(!cascade, "the in-block use must NOT cascade into UnknownBinding/IndexOnNonBuffer; errors: {errs:?}");
}

/// AT-2943: a TOP-LEVEL decl indexed INSIDE a loop compiles clean (proves the
/// restriction blocks only the decl, not loop-body USE).
#[test]
fn at2943_top_level_decl_used_inside_loop_is_clean() {
    let src = kernel_src(
        "array acc: array[f32, 4];
         for i in range(0u32, 4u32) {
             acc[i] = acc[i] + 1.0f32;
         }
         return;"
    );
    let (errs, _warns) = lower(&src);
    assert!(errs.is_empty(), "top-level decl indexed inside a loop must compile clean; errors: {errs:?}");
}

// ── AT-2944 (r2): const-index-OOB hard error ──────────────────────────────────

/// AT-2944: constant WRITE index == N -> LocalArrayConstIndexOutOfBounds.
#[test]
fn at2944_const_write_index_equal_n_rejected() {
    let src = kernel_src("array h: array[u32, 8]; h[8u32] = 1u32; return;");
    let (errs, _warns) = lower(&src);
    assert!(
        errs.iter().any(|e| matches!(e, HirError::Typecheck(TypecheckError::LocalArrayConstIndexOutOfBounds { .. }))),
        "expected LocalArrayConstIndexOutOfBounds (write, index==N); errors: {errs:?}"
    );
}

/// AT-2944: constant READ index == N -> LocalArrayConstIndexOutOfBounds.
#[test]
fn at2944_const_read_index_equal_n_rejected() {
    let src = kernel_src("array h: array[u32, 8]; h[0u32] = 1u32; let c: u32 = h[8u32]; return;");
    let (errs, _warns) = lower(&src);
    assert!(
        errs.iter().any(|e| matches!(e, HirError::Typecheck(TypecheckError::LocalArrayConstIndexOutOfBounds { .. }))),
        "expected LocalArrayConstIndexOutOfBounds (read, index==N); errors: {errs:?}"
    );
}

/// AT-2944: constant index N-1 (the last valid index) compiles clean.
#[test]
fn at2944_const_index_n_minus_1_clean() {
    let src = kernel_src("array h: array[u32, 8]; h[7u32] = 1u32; let c: u32 = h[7u32]; return;");
    let (errs, _warns) = lower(&src);
    assert!(errs.is_empty(), "index N-1 must compile clean; errors: {errs:?}");
}

/// AT-2944: a symbolic (non-constant) index of any value compiles clean —
/// UB-by-design, not flagged (mirrors `shared`'s dynamic-index philosophy).
#[test]
fn at2944_symbolic_index_not_flagged() {
    let src = kernel_src_with_params(
        "idx: readonly_buffer[u32]",
        "array h: array[u32, 8]; let i: u32 = idx[0u32]; h[i] = 1u32; let c: u32 = h[i]; return;"
    );
    let (errs, _warns) = lower(&src);
    assert!(
        !errs.iter().any(|e| matches!(e, HirError::Typecheck(TypecheckError::LocalArrayConstIndexOutOfBounds { .. }))),
        "symbolic index must NOT be flagged (UB-by-design); errors: {errs:?}"
    );
}
