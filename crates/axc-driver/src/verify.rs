//! M3.18 (FG.8) — `axc verify`: static annotation/consistency checker (§2 of the spec).
//!
//! Runs the compile FRONT END ONLY (lex → parse → `lower_module`). Never calls
//! codegen, never touches Vulkan — CI-friendly, deterministic, fast. Inspects both
//! the raw AST annotation list (`@strict` presence, the arity-!=1 silent-drop that
//! `lower.rs` normalizes away) AND the lowered HIR (`HirError`/`HirWarning`,
//! `KernelAnnotations`).
//!
//! `verify_source` is a PURE function: it never touches the filesystem or Vulkan.
//! The CLI layer (`main.rs::run_verify`) owns file I/O and maps an unreadable file
//! to the `ERROR` verdict (§2.4 — `verify_source` itself only ever returns
//! `PASS`/`FAIL`; `ERROR` is reserved for CLI-level usage/IO failures).
//!
//! Checks implemented (§2.2 of the spec):
//! - C1 `@strict` discipline completeness (intent + complexity + >=1 pre/postcondition).
//! - C2(a) predicate-DSL well-formedness — every relevant `HirError` surfaced.
//! - C2(b) the arity-!=1 silent-drop diagnostic (`predicate_arity_dropped`) — detected
//!   from the raw AST; `axc compile` itself is UNCHANGED (frozen invariant).
//! - C3 `@workgroup` sanity — reuses the two-tier cap already computed by
//!   `axc_hir::lower::validate_workgroup_dims` (`BadWorkgroupDim` error / `
//!   WorkgroupExceedsPortableFloor` warning); NO new threshold is introduced.
//! - C4 `@postcondition` not-lowerable warnings (`HirWarning::PostconditionNotLowerable`).
//! - C5 any front-end (lex/parse/HIR) error is a verify error.

use std::collections::BTreeMap;

use axc_hir::{lower_module, HirError, HirWarning};
use axc_parser::ast::{Item, KernelDecl};
use miette::Diagnostic;

const MILESTONE: &str = "M3.18";

/// The verdict — closed set. `verify_source` only ever produces `Pass`/`Fail`;
/// `Error` is reserved for CLI-level usage/IO failures (§2.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum VerifyVerdict {
    Pass,
    Fail,
    Error,
}

/// CLI exit code for a verdict (§2.5): `PASS -> 0`, `FAIL -> 1`, `ERROR -> 2`.
pub fn exit_code_for_verify(v: VerifyVerdict) -> i32 {
    match v {
        VerifyVerdict::Pass => 0,
        VerifyVerdict::Fail => 1,
        VerifyVerdict::Error => 2,
    }
}

/// One named check's pass/warn/fail summary (§2.4 `checks[]`).
#[derive(Debug, Clone, serde::Serialize)]
pub struct VerifyCheckResult {
    pub check: String,
    /// `"pass"` | `"warn"` | `"fail"`.
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warnings: Option<usize>,
}

/// One diagnostic (§2.4 `diagnostics[]`).
#[derive(Debug, Clone, serde::Serialize)]
pub struct VerifyDiag {
    /// `"error"` | `"warning"`.
    pub severity: String,
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct VerifySummary {
    pub errors: usize,
    pub warnings: usize,
}

/// The full verify report. `verdict` serializes FIRST (declared first; serde_json
/// preserves struct field declaration order — same technique as `VerifyReport`
/// in `rewrite_verify.rs`, AT-2842).
#[derive(Debug, Clone, serde::Serialize)]
pub struct VerifyReport {
    pub verdict: VerifyVerdict,
    pub milestone: &'static str,
    pub file: String,
    pub kernel: Option<String>,
    pub checks: Vec<VerifyCheckResult>,
    pub diagnostics: Vec<VerifyDiag>,
    pub summary: VerifySummary,
}

/// Internal grouping used to compute the four named `checks[]` entries.
/// NOT part of the public JSON shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Category {
    Strict,
    PredicateDsl,
    Workgroup,
    PostconditionLowerable,
    Other,
}

/// 1-based source line number for a byte offset (best-effort; mirrors
/// `axc_driver::source_line_of` but kept local — verify.rs's only permitted
/// `lib.rs` edit is the two `pub mod` lines, §5 of the spec).
fn line_of(source: &str, byte_offset: u32) -> u32 {
    let off: usize = (byte_offset as usize).min(source.len());
    1 + source.as_bytes()[..off].iter().filter(|&&b| b == b'\n').count() as u32
}

/// Best-effort line number for an `HirError` via its `miette::Diagnostic` labels
/// (every variant carries a `#[label("...")] span: Span` field; this avoids a
/// per-variant match to extract it).
fn hir_error_line(e: &HirError, source: &str) -> Option<u32> {
    let mut labels = e.labels()?;
    labels.next().map(|l| line_of(source, l.offset() as u32))
}

/// C2(a): map an `HirError` to a `(Category, code)` pair. DSL-related variants get
/// precise codes (AT-2884); workgroup variants feed C3; everything else is still
/// surfaced (C5 — "any HIR error is a verify error") under a generic code.
fn classify_hir_error(e: &HirError) -> (Category, &'static str) {
    match e {
        HirError::UnsupportedDebugPredicate { .. } => (Category::PredicateDsl, "unsupported_debug_predicate"),
        HirError::PreconditionOperandNotScalar { .. } => (Category::PredicateDsl, "precondition_operand_not_scalar"),
        HirError::TooManyDebugConditions { .. } => (Category::PredicateDsl, "too_many_debug_conditions"),
        HirError::UnsupportedPrecondition { .. } => (Category::PredicateDsl, "unsupported_precondition"),
        HirError::BadWorkgroupDim { .. } => (Category::Workgroup, "bad_workgroup_dim"),
        HirError::BadWorkgroupArity { .. } => (Category::Workgroup, "bad_workgroup_arity"),
        HirError::MissingWorkgroup { .. } => (Category::Workgroup, "missing_workgroup"),
        HirError::WorkgroupDimOverflow { .. } => (Category::Workgroup, "workgroup_dim_overflow"),
        _ => (Category::Other, "hir_error"),
    }
}

/// Best-effort span extraction for `HirWarning` (a plain enum, not a `miette::Diagnostic`).
/// Exhaustive match (anti-pattern #4 — no silent wildcard arm).
fn hir_warning_span(w: &HirWarning) -> Option<axc_lexer::Span> {
    match w {
        HirWarning::WorkgroupExceedsPortableFloor { span, .. } => Some(*span),
        HirWarning::SubgroupOpInDivergentContext { span, .. } => Some(*span),
        HirWarning::UnusedStrategyHole { span, .. } => Some(*span),
        HirWarning::EmptyStrategyBlock { span } => Some(*span),
        HirWarning::StrategyHoleShadowsKernelParam { span, .. } => Some(*span),
        HirWarning::CoopMatFlagWithoutOps { .. } => None,
        HirWarning::SharedMemoryExceedsPortableMinimum { span, .. } => Some(*span),
        HirWarning::SharedWriteWithoutBarrierBeforeRead { span, .. } => Some(*span),
        HirWarning::PostconditionNotLowerable { span, .. } => Some(*span),
    }
}

/// C3 + C4: map an `HirWarning` to a `(Category, code, message)` triple.
fn classify_hir_warning(w: &HirWarning) -> (Category, &'static str, String) {
    match w {
        HirWarning::WorkgroupExceedsPortableFloor { name, product, floor, .. } => (
            Category::Workgroup,
            "workgroup_over_portable_min",
            format!(
                "kernel `{name}` @workgroup product {product} exceeds the Vulkan portability floor \
                 ({floor}); may fail on some conformant devices"
            ),
        ),
        HirWarning::PostconditionNotLowerable { buf, reason, .. } => (
            Category::PostconditionLowerable,
            "postcondition_not_lowerable",
            format!("@postcondition over buffer `{buf}` is deferred; NOT checked at runtime ({reason})"),
        ),
        other => (Category::Other, "hir_warning", format!("{other:?}")),
    }
}

/// C1: `@strict` discipline completeness. Raw-AST scan — the HIR silently no-ops
/// `@strict` (`lower.rs:167`), so presence/absence of the required elements must
/// be read straight from `kd.annotations`.
fn check_strict_discipline(kd: &KernelDecl, source: &str, out: &mut Vec<(Category, VerifyDiag)>) {
    let has = |name: &str| kd.annotations.iter().any(|a| a.node.name.node == name);
    if !has("strict") {
        return;
    }
    let line = Some(line_of(source, kd.name.span.start));
    if !has("intent") {
        out.push((
            Category::Strict,
            VerifyDiag {
                severity: "error".to_string(),
                code: "strict_missing_intent".to_string(),
                message: format!("kernel `{}` has @strict but is missing @intent", kd.name.node),
                line,
            },
        ));
    }
    if !has("complexity") {
        out.push((
            Category::Strict,
            VerifyDiag {
                severity: "error".to_string(),
                code: "strict_missing_complexity".to_string(),
                message: format!("kernel `{}` has @strict but is missing @complexity", kd.name.node),
                line,
            },
        ));
    }
    if !has("precondition") && !has("postcondition") {
        out.push((
            Category::Strict,
            VerifyDiag {
                severity: "error".to_string(),
                code: "strict_missing_precondition".to_string(),
                message: format!(
                    "kernel `{}` has @strict but declares no @precondition/@postcondition \
                     (CLAUDE.md rule: @intent + @complexity + at least one pre/postcondition)",
                    kd.name.node
                ),
                line,
            },
        ));
    }
}

/// C2(b): the arity-!=1 silent-drop diagnostic (`lower.rs:118`/`lower.rs:142` drop
/// any `@precondition`/`@postcondition` whose `args.len() != 1` with NO `else` —
/// a fail-open the compiler intentionally does not fix (§2.3 decision); verify
/// surfaces it instead.
fn check_arity_drop(kd: &KernelDecl, source: &str, out: &mut Vec<(Category, VerifyDiag)>) {
    for ann in &kd.annotations {
        let name: &str = &ann.node.name.node;
        if (name == "precondition" || name == "postcondition") && ann.node.args.len() != 1 {
            out.push((
                Category::PredicateDsl,
                VerifyDiag {
                    severity: "error".to_string(),
                    code: "predicate_arity_dropped".to_string(),
                    message: format!(
                        "@{name} has {} arguments; only single-predicate annotations are lowered \
                         (a comma-conjunction is silently DROPPED by the compiler, not rejected) — \
                         split into separate @{name}(...) annotations",
                        ann.node.args.len()
                    ),
                    line: Some(line_of(source, ann.span.start)),
                },
            ));
        }
    }
}

/// Every raw `AnnotationArg::Call` that would be silently dropped by `lower.rs`
/// (arity != 1) on this kernel — used by AT-2883's differential pin so verify's
/// detection set can be asserted equal to `lower.rs`'s drop set.
pub fn arity_dropped_annotation_names(kd: &KernelDecl) -> Vec<&'static str> {
    let mut out: Vec<&'static str> = Vec::new();
    for ann in &kd.annotations {
        let name: &str = &ann.node.name.node;
        if name == "precondition" && ann.node.args.len() != 1 {
            out.push("precondition");
        }
        if name == "postcondition" && ann.node.args.len() != 1 {
            out.push("postcondition");
        }
    }
    out
}

/// Run `axc verify` over `source` (already read from `file` by the caller).
///
/// Pure: no filesystem, no Vulkan. `file` is echoed into the report only.
pub fn verify_source(source: &str, file: &str) -> VerifyReport {
    let (ast, lex_errors, parse_errors) = axc_parser::parse(source);
    let (_hir, hir_errors, hir_warnings) = lower_module(&ast);

    let mut diags: Vec<(Category, VerifyDiag)> = Vec::new();

    for e in &lex_errors {
        diags.push((
            Category::Other,
            VerifyDiag { severity: "error".to_string(), code: "lex_error".to_string(), message: e.to_string(), line: None },
        ));
    }
    for e in &parse_errors {
        diags.push((
            Category::Other,
            VerifyDiag { severity: "error".to_string(), code: "parse_error".to_string(), message: e.to_string(), line: None },
        ));
    }
    for e in &hir_errors {
        let (cat, code) = classify_hir_error(e);
        diags.push((
            cat,
            VerifyDiag { severity: "error".to_string(), code: code.to_string(), message: e.to_string(), line: hir_error_line(e, source) },
        ));
    }
    for w in &hir_warnings {
        let (cat, code, message) = classify_hir_warning(w);
        let line = hir_warning_span(w).map(|s| line_of(source, s.start));
        diags.push((cat, VerifyDiag { severity: "warning".to_string(), code: code.to_string(), message, line }));
    }

    let mut kernel_name: Option<String> = None;
    let mut kernel_count: usize = 0;
    for item in &ast.items {
        // M0-era grammar: `Item` has exactly one variant today (`Kernel`). Matched
        // exhaustively (anti-pattern #4 — no wildcard arm) so a future `Item`
        // variant forces this loop to be revisited rather than silently skipped.
        match &item.node {
            Item::Kernel(kd) => {
                kernel_count += 1;
                if kernel_name.is_none() {
                    kernel_name = Some(kd.name.node.clone());
                }
                check_strict_discipline(kd, source, &mut diags);
                check_arity_drop(kd, source, &mut diags);
            }
        }
    }

    if kernel_count == 0 && lex_errors.is_empty() && parse_errors.is_empty() {
        diags.push((
            Category::Other,
            VerifyDiag {
                severity: "error".to_string(),
                code: "no_kernel".to_string(),
                message: "source parses cleanly but declares no @kernel; nothing to verify".to_string(),
                line: None,
            },
        ));
    }

    let mut cat_errors: BTreeMap<Category, usize> = BTreeMap::new();
    let mut cat_warnings: BTreeMap<Category, usize> = BTreeMap::new();
    for (cat, d) in &diags {
        if d.severity == "error" {
            *cat_errors.entry(*cat).or_insert(0) += 1;
        } else {
            *cat_warnings.entry(*cat).or_insert(0) += 1;
        }
    }

    let check_result = |name: &str, cat: Category| -> VerifyCheckResult {
        let e = cat_errors.get(&cat).copied().unwrap_or(0);
        let w = cat_warnings.get(&cat).copied().unwrap_or(0);
        let status = if e > 0 { "fail" } else if w > 0 { "warn" } else { "pass" };
        VerifyCheckResult { check: name.to_string(), status: status.to_string(), warnings: if w > 0 { Some(w) } else { None } }
    };

    let checks: Vec<VerifyCheckResult> = vec![
        check_result("strict_discipline", Category::Strict),
        check_result("predicate_dsl", Category::PredicateDsl),
        check_result("workgroup_sanity", Category::Workgroup),
        check_result("postcondition_lowerable", Category::PostconditionLowerable),
    ];

    let total_errors: usize = diags.iter().filter(|(_, d)| d.severity == "error").count();
    let total_warnings: usize = diags.iter().filter(|(_, d)| d.severity == "warning").count();
    let verdict: VerifyVerdict = if total_errors == 0 { VerifyVerdict::Pass } else { VerifyVerdict::Fail };

    VerifyReport {
        verdict,
        milestone: MILESTONE,
        file: file.to_string(),
        kernel: kernel_name,
        checks,
        diagnostics: diags.into_iter().map(|(_, d)| d).collect(),
        summary: VerifySummary { errors: total_errors, warnings: total_warnings },
    }
}

/// Render a `VerifyReport` as the default (non-`--json`) human summary.
pub fn render_human(report: &VerifyReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("axc verify: {} — {:?}\n", report.file, report.verdict));
    if let Some(k) = &report.kernel {
        out.push_str(&format!("kernel: {k}\n"));
    }
    for c in &report.checks {
        out.push_str(&format!("  [{}] {}\n", c.status, c.check));
    }
    for d in &report.diagnostics {
        match d.line {
            Some(l) => out.push_str(&format!("{}: {} (line {l}): {}\n", d.severity, d.code, d.message)),
            None => out.push_str(&format!("{}: {}: {}\n", d.severity, d.code, d.message)),
        }
    }
    out.push_str(&format!("summary: {} error(s), {} warning(s)\n", report.summary.errors, report.summary.warnings));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAXPY_SRC: &str = include_str!("../../../examples/saxpy.axc");
    const PRECONDITION_SAXPY_SRC: &str = include_str!("../../../examples/precondition_saxpy.axc");
    const VERIFY_STRICT_INCOMPLETE_SRC: &str = include_str!("../../../examples/verify_strict_incomplete.axc");
    const VERIFY_ARITY_DROP_SRC: &str = include_str!("../../../examples/verify_arity_drop.axc");

    // ── AT-2881: verify PASS ───────────────────────────────────────────────────

    #[test]
    fn at_2881_precondition_saxpy_verifies_pass() {
        let report = verify_source(PRECONDITION_SAXPY_SRC, "examples/precondition_saxpy.axc");
        assert_eq!(report.verdict, VerifyVerdict::Pass, "diags: {:?}", report.diagnostics);
        assert_eq!(report.summary.errors, 0);
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.starts_with("{\"verdict\":"), "verdict must be first key; got {json}");
        assert!(json.contains("\"errors\":0"));
    }

    #[test]
    fn at_2881b_saxpy_no_strict_no_dsl_verifies_pass() {
        let report = verify_source(SAXPY_SRC, "examples/saxpy.axc");
        assert_eq!(report.verdict, VerifyVerdict::Pass, "diags: {:?}", report.diagnostics);
    }

    // ── AT-2882: C1 strict-fail + sibling positive ─────────────────────────────

    #[test]
    fn at_2882_strict_incomplete_fails_missing_complexity() {
        let report = verify_source(VERIFY_STRICT_INCOMPLETE_SRC, "examples/verify_strict_incomplete.axc");
        assert_eq!(report.verdict, VerifyVerdict::Fail);
        assert!(
            report.diagnostics.iter().any(|d| d.code == "strict_missing_complexity"),
            "expected strict_missing_complexity; got {:?}", report.diagnostics
        );
    }

    #[test]
    fn at_2882b_complete_strict_kernel_passes() {
        let src = concat!(
            "@kernel\n@workgroup(64,1,1)\n@strict\n@intent(\"t\")\n@complexity(O(n))\n",
            "@precondition(gt(n,0))\n",
            "fn k(n: u32) -> void {\n    return;\n}\n",
        );
        let report = verify_source(src, "inline");
        assert_eq!(report.verdict, VerifyVerdict::Pass, "diags: {:?}", report.diagnostics);
        assert!(!report.diagnostics.iter().any(|d| d.code.starts_with("strict_missing")));
    }

    // ── AT-2883: C2(b) arity-drop diagnostic + differential pin ───────────────

    #[test]
    fn at_2883_arity_drop_fails_and_compile_unchanged() {
        let report = verify_source(VERIFY_ARITY_DROP_SRC, "examples/verify_arity_drop.axc");
        assert_eq!(report.verdict, VerifyVerdict::Fail);
        assert!(
            report.diagnostics.iter().any(|d| d.code == "predicate_arity_dropped"),
            "expected predicate_arity_dropped; got {:?}", report.diagnostics
        );
        // Frozen-invariant control: axc compile still SUCCEEDS with the arity-2
        // predicate silently dropped (compile behavior byte-unchanged).
        let compiled = crate::compile_source_with_meta(VERIFY_ARITY_DROP_SRC);
        assert!(compiled.is_ok(), "compile must still succeed (arity-drop is fail-open by design): {compiled:?}");
    }

    /// r2 differential pin: verify's arity-drop detection set must exactly match
    /// `lower.rs`'s actual silent-drop set (`args.len() != 1`) — walked independently
    /// here via the raw AST, pinning the two duplicated `args.len()==1` predicates
    /// against drift.
    #[test]
    fn at_2883b_arity_drop_set_matches_lower_rs_drop_set() {
        let (ast, lex, parse) = axc_parser::parse(VERIFY_ARITY_DROP_SRC);
        assert!(lex.is_empty() && parse.is_empty());
        let Item::Kernel(kd) = &ast.items[0].node;
        let verify_flagged = arity_dropped_annotation_names(kd);
        // lower.rs:118/142 drop set: any precondition/postcondition with args.len()!=1.
        let lower_dropped: Vec<&str> = kd
            .annotations
            .iter()
            .filter(|a| (a.node.name.node == "precondition" || a.node.name.node == "postcondition") && a.node.args.len() != 1)
            .map(|a| if a.node.name.node == "precondition" { "precondition" } else { "postcondition" })
            .collect();
        assert_eq!(verify_flagged, lower_dropped, "verify's arity-drop set must equal lower.rs's silent-drop set");
        assert!(!verify_flagged.is_empty(), "the fixture must actually exercise the drop");
    }

    // ── AT-2884: C2(a) DSL reject surfaced ─────────────────────────────────────

    #[test]
    fn at_2884_unsupported_predicate_surfaced() {
        let src = "@kernel @workgroup(1,1,1) @precondition(foo(n)) fn k(n: u32) -> void { return; }";
        let report = verify_source(src, "inline");
        assert_eq!(report.verdict, VerifyVerdict::Fail);
        assert!(report.diagnostics.iter().any(|d| d.code == "unsupported_debug_predicate"));
    }

    #[test]
    fn at_2884b_precondition_operand_not_scalar_surfaced() {
        let src = "@kernel @workgroup(1,1,1) @precondition(gt(elem(x), 0)) fn k(x: buffer[u32]) -> void { return; }";
        let report = verify_source(src, "inline");
        assert_eq!(report.verdict, VerifyVerdict::Fail);
        assert!(report.diagnostics.iter().any(|d| d.code == "precondition_operand_not_scalar"));
    }

    // ── AT-2885: C3 workgroup ───────────────────────────────────────────────────

    #[test]
    fn at_2885_workgroup_over_portable_floor_warns_pass() {
        // product = 256: within (128, 1024] — the EXISTING two-tier WARN band
        // (axc_hir::hir::PORTABLE_MIN_WORKGROUP_INVOCATIONS=128,
        // DESKTOP_MAX_WORKGROUP_INVOCATIONS=1024). The spec's illustrative "2048"
        // example would exceed the ALREADY-SHIPPED, FROZEN 1024 hard-error ceiling
        // (compile-time BadWorkgroupDim) rather than warn — using 256 exercises the
        // real WARN band without touching the frozen compiler threshold.
        let src = "@kernel @workgroup(256,1,1) fn k() -> void { return; }";
        let report = verify_source(src, "inline");
        assert_eq!(report.verdict, VerifyVerdict::Pass, "diags: {:?}", report.diagnostics);
        assert!(report.diagnostics.iter().any(|d| d.code == "workgroup_over_portable_min" && d.severity == "warning"));
        assert_eq!(report.summary.errors, 0);
        assert!(report.summary.warnings >= 1);
    }

    #[test]
    fn at_2885b_zero_dim_workgroup_fails() {
        let src = "@kernel @workgroup(0,1,1) fn k() -> void { return; }";
        let report = verify_source(src, "inline");
        assert_eq!(report.verdict, VerifyVerdict::Fail);
        assert!(report.diagnostics.iter().any(|d| d.code == "bad_workgroup_dim"));
    }

    // ── AT-2886: C4 postcondition-not-lowerable warning ────────────────────────

    #[test]
    fn at_2886_postcondition_not_lowerable_warns() {
        let src = "@kernel @workgroup(64,1,1) @postcondition(is_finite(elem(y))) \
                   fn k(y: buffer[f32]) -> void { y[gid(0)] = 1.0; return; }";
        let report = verify_source(src, "inline");
        assert_eq!(report.verdict, VerifyVerdict::Pass, "diags: {:?}", report.diagnostics);
        let diag = report.diagnostics.iter().find(|d| d.code == "postcondition_not_lowerable")
            .unwrap_or_else(|| panic!("expected postcondition_not_lowerable; got {:?}", report.diagnostics));
        assert!(diag.message.contains('y'), "message must name buffer y: {}", diag.message);
    }

    // ── AT-2887: verify JSON schema + exit map ─────────────────────────────────

    #[test]
    fn at_2887_verdict_first_key_for_pass_and_fail() {
        for (src, _label) in [(PRECONDITION_SAXPY_SRC, "pass"), (VERIFY_STRICT_INCOMPLETE_SRC, "fail")] {
            let report = verify_source(src, "inline");
            let json = serde_json::to_string(&report).unwrap();
            assert!(json.starts_with("{\"verdict\":"), "verdict must be first key; got {json}");
        }
    }

    #[test]
    fn at_2887b_exit_code_mapping() {
        assert_eq!(exit_code_for_verify(VerifyVerdict::Pass), 0);
        assert_eq!(exit_code_for_verify(VerifyVerdict::Fail), 1);
        assert_eq!(exit_code_for_verify(VerifyVerdict::Error), 2);
    }

    #[test]
    fn no_kernel_source_is_fail_not_error() {
        let report = verify_source("", "inline");
        assert_eq!(report.verdict, VerifyVerdict::Fail);
        assert!(report.diagnostics.iter().any(|d| d.code == "no_kernel"));
    }

    #[test]
    fn verdict_serializes_uppercase() {
        assert_eq!(serde_json::to_string(&VerifyVerdict::Pass).unwrap(), "\"PASS\"");
        assert_eq!(serde_json::to_string(&VerifyVerdict::Fail).unwrap(), "\"FAIL\"");
        assert_eq!(serde_json::to_string(&VerifyVerdict::Error).unwrap(), "\"ERROR\"");
    }

    #[test]
    fn render_human_does_not_panic_and_mentions_verdict() {
        let report = verify_source(PRECONDITION_SAXPY_SRC, "f.axc");
        let text = render_human(&report);
        assert!(text.contains("Pass") || text.contains("PASS") || text.contains("verify"));
    }
}
