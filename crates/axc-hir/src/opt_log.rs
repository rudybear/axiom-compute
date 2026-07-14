//! FG.3 — `@optimization_log { … }` schema + validator (M3.19).
//!
//! # Why this exists (the falsifiable value case, spec §0.3)
//!
//! The external optimization history (M2.4) is keyed by `xxh3_64(source_bytes)`
//! (`.pipeline/history/<hash>.jsonl`), so it **orphans on every source edit** —
//! exactly the operation the agent self-optimize loop performs. This module is
//! the embedded, rewrite-surviving alternative: a first-class, typed block
//! (anti-pattern #7 — no string blobs) that travels *inside* the kernel source
//! and round-trips through the whole toolchain unchanged.
//!
//! # Codegen-inert
//!
//! `OptimizationLog` lives on `KernelAnnotations` only; codegen never reads it.
//! SPIR-V is byte-identical with and without the block (golden-gated at AT-2903).
//!
//! # AST → schema pipeline
//!
//! `axc_parser::parser::parse_optimization_log_block` produces a flat
//! `Vec<Spanned<AnnotationArg>>` per kernel annotation (see
//! `AnnotationArg::Record`'s doc comment for the exact shape). `axc_hir::lower`
//! separates that list into block-level `RecordField`s (currently just
//! `version`) and per-`entry` field lists, then calls [`validate_block`] once
//! and [`validate_entry`] once per entry — this module never touches the AST
//! directly, only the already-separated `RecordField` slices.

use axc_lexer::Span;
use axc_parser::ast::{RecordField, RecordValue};

use crate::validate::HirError;

/// Bounds source growth / merge noise. The minimal-build writer (`axc log-opt`,
/// `crates/axc-driver/src/log_opt.rs`) fails closed (`LogOptError::Full`)
/// rather than evicting the oldest entry once this cap is reached — rotation
/// is deferred (spec §1.8).
pub const MAX_OPT_LOG_ENTRIES: usize = 64;

/// This compiler's supported `@optimization_log` schema version.
///
/// `version == SUPPORTED_OPT_LOG_VERSION` blocks are validated exhaustively —
/// an unknown key is a REJECT (typo protection). `version >
/// SUPPORTED_OPT_LOG_VERSION` blocks preserve unknown keys into
/// `forward_unknown` and emit `HirWarning::UnknownOptimizationLogKeyForward`
/// instead of rejecting (§1.4a forward-compat: an old compiler must not
/// hard-fail on a newer, additively-extended block).
pub const SUPPORTED_OPT_LOG_VERSION: i64 = 1;

/// A fully-validated `@optimization_log { ... }` block.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizationLog {
    /// Required block-level field; `== SUPPORTED_OPT_LOG_VERSION` for a block
    /// this compiler considers "current" (unknown-key-rejecting); `>
    /// SUPPORTED_OPT_LOG_VERSION` for a forward-compat block.
    pub version: i64,
    /// `0..=MAX_OPT_LOG_ENTRIES` validated entries, in source order.
    pub entries: Vec<OptimizationLogEntry>,
    /// Block-level keys unknown to this compiler, preserved verbatim ONLY when
    /// `version > SUPPORTED_OPT_LOG_VERSION` (§1.4a). Always empty for a
    /// `version == SUPPORTED_OPT_LOG_VERSION` block (an unknown key there is a
    /// REJECT, never reaches this field).
    ///
    /// NOTE (parser-reachability): §1.3a's EBNF pins
    /// `block_item := version_field | entry_block` as CLOSED — THIS
    /// compiler's parser can therefore never produce a block-level
    /// `RecordField` other than `version` from real source text, so this
    /// field is always empty in practice today. It exists as forward-looking
    /// infrastructure for a FUTURE parser version that extends `block_item`
    /// with additional keys while reusing this same validator unchanged.
    /// (Entry-level `forward_unknown`, on `OptimizationLogEntry`, has NO such
    /// restriction — `field := ident ":" record_value` admits any key
    /// syntactically, so it IS reachable via real source today.)
    pub forward_unknown: Vec<(String, RecordValue)>,
}

/// One validated `entry { ... }`.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizationLogEntry {
    /// Required; non-empty; RFC3339-shape-validated only (not calendar-checked).
    pub ts: String,
    /// Required; non-empty.
    pub kernel: String,
    /// Required; sign legality is unit-dependent (see [`OptLogUnit::allows_negative`]).
    pub metric: i64,
    /// Required; closed ident set.
    pub unit: OptLogUnit,
    /// Required; closed ident set.
    pub verdict: OptLogVerdict,
    /// Optional free-form note.
    pub note: Option<String>,
    /// Entry-level unknown keys, preserved verbatim ONLY when
    /// `version > SUPPORTED_OPT_LOG_VERSION` (§1.4a).
    pub forward_unknown: Vec<(String, RecordValue)>,
}

/// Frozen `unit` vocabulary (§1.4 of the spec) — the `_milli` fixed-point scale
/// lives in the UNIT NAME, not the field name (`metric` is unit-neutral).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLogUnit {
    /// `mtflops` — throughput, TFLOPS × 1000.
    MilliTflops,
    /// `pct_milli` — percentage, % × 1000. The only unit allowing a negative
    /// `metric` (deltas).
    MilliPercent,
    /// `ns` — latency, raw nanoseconds.
    Nanoseconds,
    /// `us` — latency, raw microseconds.
    Microseconds,
    /// `ms` — latency, raw milliseconds.
    Milliseconds,
    /// `bytes` — size/bandwidth counter, raw bytes.
    Bytes,
}

impl OptLogUnit {
    /// Parse the closed ident set; `None` for anything else (HIR rejects unknown units).
    pub fn from_ident(s: &str) -> Option<Self> {
        match s {
            "mtflops" => Some(Self::MilliTflops),
            "pct_milli" => Some(Self::MilliPercent),
            "ns" => Some(Self::Nanoseconds),
            "us" => Some(Self::Microseconds),
            "ms" => Some(Self::Milliseconds),
            "bytes" => Some(Self::Bytes),
            _ => None,
        }
    }

    /// The canonical source-text ident for this unit (round-trip formatting).
    pub fn as_ident(self) -> &'static str {
        match self {
            Self::MilliTflops => "mtflops",
            Self::MilliPercent => "pct_milli",
            Self::Nanoseconds => "ns",
            Self::Microseconds => "us",
            Self::Milliseconds => "ms",
            Self::Bytes => "bytes",
        }
    }

    /// `true` only for `pct_milli` (deltas may be negative); every other unit
    /// rejects a negative `metric` as nonsensical (§1.4 sign-rule table).
    pub fn allows_negative(self) -> bool {
        matches!(self, Self::MilliPercent)
    }
}

/// Frozen `verdict` vocabulary (§1.4/§1.5): the six real
/// `rewrite_verify::Verdict` variants verbatim in SCREAMING_SNAKE_CASE, plus
/// grid-search's `REGRESS` (which has no `rewrite_verify` source).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLogVerdict {
    Pass,
    Fail,
    Reject,
    Skipped,
    NondeterministicOracle,
    Error,
    Regress,
}

impl OptLogVerdict {
    /// Parse the closed ident set; `None` for anything else (HIR rejects unknown verdicts).
    pub fn from_ident(s: &str) -> Option<Self> {
        match s {
            "PASS" => Some(Self::Pass),
            "FAIL" => Some(Self::Fail),
            "REJECT" => Some(Self::Reject),
            "SKIPPED" => Some(Self::Skipped),
            "NONDETERMINISTIC_ORACLE" => Some(Self::NondeterministicOracle),
            "ERROR" => Some(Self::Error),
            "REGRESS" => Some(Self::Regress),
            _ => None,
        }
    }

    /// The canonical source-text ident for this verdict (round-trip formatting).
    pub fn as_ident(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Fail => "FAIL",
            Self::Reject => "REJECT",
            Self::Skipped => "SKIPPED",
            Self::NondeterministicOracle => "NONDETERMINISTIC_ORACLE",
            Self::Error => "ERROR",
            Self::Regress => "REGRESS",
        }
    }
}

/// Validate the block-level fields (currently only `version`) against the
/// schema (§1.4a). Returns an [`OptimizationLog`] with EMPTY `entries` — the
/// caller (`axc_hir::lower::lower_optimization_log_annotation`) validates and
/// appends each `entry { ... }` separately via [`validate_entry`], threading
/// this return value's `version` through so entry-level forward-compat honors
/// the block's declared version.
///
/// Rejects (`HirError::BadOptimizationLog`): `version` absent, duplicated,
/// non-`Int`, or `< 1`; for `version == SUPPORTED_OPT_LOG_VERSION`, any
/// unknown block-level key (typo protection). For `version >
/// SUPPORTED_OPT_LOG_VERSION`, unknown keys are preserved into
/// `forward_unknown` instead of rejecting (the caller emits the
/// corresponding `HirWarning`).
pub fn validate_block(block_fields: &[RecordField], span: Span) -> Result<OptimizationLog, HirError> {
    let mut version: Option<i64> = None;
    let mut version_seen_count: usize = 0;
    let mut forward_unknown: Vec<(String, RecordValue)> = Vec::new();
    let mut unknown_keys: Vec<String> = Vec::new();

    for f in block_fields {
        match f.key.node.as_str() {
            "version" => {
                version_seen_count += 1;
                match &f.value.node {
                    RecordValue::Int(v) => version = Some(*v),
                    other => {
                        return Err(HirError::BadOptimizationLog {
                            detail: format!("`version` must be an integer literal; got {other:?}"),
                            span,
                        });
                    }
                }
            }
            other_key => {
                unknown_keys.push(other_key.to_string());
                forward_unknown.push((other_key.to_string(), f.value.node.clone()));
            }
        }
    }

    if version_seen_count == 0 {
        return Err(HirError::BadOptimizationLog {
            detail: "missing required block-level `version` field".to_string(),
            span,
        });
    }
    if version_seen_count > 1 {
        return Err(HirError::BadOptimizationLog {
            detail: format!("block-level `version` field is declared {version_seen_count} times; exactly one is required"),
            span,
        });
    }
    let version: i64 = version.expect("version_seen_count==1 implies version is Some (Int arm sets it)");
    if version < 1 {
        return Err(HirError::BadOptimizationLog {
            detail: format!("`version` must be >= 1; got {version}"),
            span,
        });
    }

    if version == SUPPORTED_OPT_LOG_VERSION {
        if !unknown_keys.is_empty() {
            return Err(HirError::BadOptimizationLog {
                detail: format!(
                    "unknown block-level key(s) for version {SUPPORTED_OPT_LOG_VERSION}: {}",
                    unknown_keys.join(", ")
                ),
                span,
            });
        }
        // A v1 (== SUPPORTED) block is fully known; forward_unknown is always
        // empty here (unknown keys already rejected above).
        return Ok(OptimizationLog { version, entries: Vec::new(), forward_unknown: Vec::new() });
    }

    // version > SUPPORTED: preserve-and-warn (§1.4a). The warning itself is
    // emitted by the caller, which has access to the `HirWarning` sink.
    Ok(OptimizationLog { version, entries: Vec::new(), forward_unknown })
}

/// Validate one `entry { ... }`'s fields against the schema (§1.4).
///
/// `version` is the block's ALREADY-VALIDATED version (from [`validate_block`]),
/// threaded through so forward-compat applies consistently across every entry
/// in the same block.
///
/// Rejects (`HirError::BadOptimizationLogEntry`): missing required field,
/// duplicate key, wrong `RecordValue` shape for a field, out-of-set
/// `unit`/`verdict` ident, a negative `metric` under a `>= 0`-only unit, and —
/// for `version == SUPPORTED_OPT_LOG_VERSION` — any unknown key. For `version
/// > SUPPORTED_OPT_LOG_VERSION`, unknown keys are preserved into
/// `forward_unknown` instead.
pub fn validate_entry(fields: &[RecordField], version: i64, span: Span) -> Result<OptimizationLogEntry, HirError> {
    let mut ts: Option<String> = None;
    let mut kernel: Option<String> = None;
    let mut metric: Option<i64> = None;
    let mut unit: Option<OptLogUnit> = None;
    let mut verdict: Option<OptLogVerdict> = None;
    let mut note: Option<String> = None;
    let mut forward_unknown: Vec<(String, RecordValue)> = Vec::new();
    let mut seen_keys: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

    for f in fields {
        let key: &str = f.key.node.as_str();
        if !seen_keys.insert(key.to_string()) {
            return Err(HirError::BadOptimizationLogEntry {
                detail: format!("duplicate key `{key}` in entry"),
                span,
            });
        }
        match key {
            "ts" => match &f.value.node {
                RecordValue::Str(s) if s.is_empty() => {
                    return Err(HirError::BadOptimizationLogEntry { detail: "`ts` must be non-empty".to_string(), span });
                }
                RecordValue::Str(s) => {
                    if !looks_like_rfc3339(s) {
                        return Err(HirError::BadOptimizationLogEntry {
                            detail: format!("`ts` does not look like an RFC3339 timestamp: {s:?}"),
                            span,
                        });
                    }
                    ts = Some(s.clone());
                }
                other => {
                    return Err(HirError::BadOptimizationLogEntry { detail: format!("`ts` must be a string; got {other:?}"), span });
                }
            },
            "kernel" => match &f.value.node {
                RecordValue::Str(s) if s.is_empty() => {
                    return Err(HirError::BadOptimizationLogEntry { detail: "`kernel` must be non-empty".to_string(), span });
                }
                RecordValue::Str(s) => kernel = Some(s.clone()),
                other => {
                    return Err(HirError::BadOptimizationLogEntry { detail: format!("`kernel` must be a string; got {other:?}"), span });
                }
            },
            "metric" => match &f.value.node {
                RecordValue::Int(v) => metric = Some(*v),
                other => {
                    return Err(HirError::BadOptimizationLogEntry { detail: format!("`metric` must be an integer; got {other:?}"), span });
                }
            },
            "unit" => match &f.value.node {
                RecordValue::Ident(s) => match OptLogUnit::from_ident(s) {
                    Some(u) => unit = Some(u),
                    None => {
                        return Err(HirError::BadOptimizationLogEntry {
                            detail: format!("unknown `unit` ident `{s}`; expected one of mtflops, pct_milli, ns, us, ms, bytes"),
                            span,
                        });
                    }
                },
                other => {
                    return Err(HirError::BadOptimizationLogEntry { detail: format!("`unit` must be an identifier; got {other:?}"), span });
                }
            },
            "verdict" => match &f.value.node {
                RecordValue::Ident(s) => match OptLogVerdict::from_ident(s) {
                    Some(v) => verdict = Some(v),
                    None => {
                        return Err(HirError::BadOptimizationLogEntry {
                            detail: format!(
                                "unknown `verdict` ident `{s}`; expected one of PASS, FAIL, REJECT, SKIPPED, \
                                 NONDETERMINISTIC_ORACLE, ERROR, REGRESS"
                            ),
                            span,
                        });
                    }
                },
                other => {
                    return Err(HirError::BadOptimizationLogEntry { detail: format!("`verdict` must be an identifier; got {other:?}"), span });
                }
            },
            "note" => match &f.value.node {
                RecordValue::Str(s) => note = Some(s.clone()),
                other => {
                    return Err(HirError::BadOptimizationLogEntry { detail: format!("`note` must be a string; got {other:?}"), span });
                }
            },
            other_key => {
                if version == SUPPORTED_OPT_LOG_VERSION {
                    return Err(HirError::BadOptimizationLogEntry {
                        detail: format!("unknown key `{other_key}` for version {SUPPORTED_OPT_LOG_VERSION}"),
                        span,
                    });
                }
                forward_unknown.push((other_key.to_string(), f.value.node.clone()));
            }
        }
    }

    let ts: String = ts.ok_or_else(|| HirError::BadOptimizationLogEntry { detail: "missing required field `ts`".to_string(), span })?;
    let kernel: String = kernel.ok_or_else(|| HirError::BadOptimizationLogEntry { detail: "missing required field `kernel`".to_string(), span })?;
    let metric: i64 = metric.ok_or_else(|| HirError::BadOptimizationLogEntry { detail: "missing required field `metric`".to_string(), span })?;
    let unit: OptLogUnit = unit.ok_or_else(|| HirError::BadOptimizationLogEntry { detail: "missing required field `unit`".to_string(), span })?;
    let verdict: OptLogVerdict = verdict.ok_or_else(|| HirError::BadOptimizationLogEntry { detail: "missing required field `verdict`".to_string(), span })?;

    if metric < 0 && !unit.allows_negative() {
        return Err(HirError::BadOptimizationLogEntry {
            detail: format!("`metric` {metric} is negative, which is not allowed for unit `{}`", unit.as_ident()),
            span,
        });
    }

    Ok(OptimizationLogEntry { ts, kernel, metric, unit, verdict, note, forward_unknown })
}

/// Minimal RFC3339 SHAPE check (§1.4: "RFC3339 shape-validated only") — not a
/// full calendar-correctness parser (no leap-year/day-count validation), just
/// the fixed positional grammar `YYYY-MM-DDTHH:MM:SS` with an optional
/// fractional-seconds/offset tail (e.g. `"2026-07-14T12:00:00.000Z"`).
fn looks_like_rfc3339(s: &str) -> bool {
    let b: &[u8] = s.as_bytes();
    if b.len() < 19 {
        return false;
    }
    let digit = |i: usize| b.get(i).is_some_and(u8::is_ascii_digit);
    digit(0) && digit(1) && digit(2) && digit(3)
        && b[4] == b'-'
        && digit(5) && digit(6)
        && b[7] == b'-'
        && digit(8) && digit(9)
        && (b[10] == b'T' || b[10] == b't')
        && digit(11) && digit(12)
        && b[13] == b':'
        && digit(14) && digit(15)
        && b[16] == b':'
        && digit(17) && digit(18)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axc_lexer::Spanned;

    fn span() -> Span {
        Span::new(0, 1)
    }

    fn field(key: &str, value: RecordValue) -> RecordField {
        RecordField {
            key: Spanned::new(key.to_string(), span()),
            value: Spanned::new(value, span()),
        }
    }

    fn valid_entry_fields() -> Vec<RecordField> {
        vec![
            field("ts", RecordValue::Str("2026-07-14T12:00:00.000Z".to_string())),
            field("kernel", RecordValue::Str("q4km_matmul".to_string())),
            field("metric", RecordValue::Int(42860)),
            field("unit", RecordValue::Ident("mtflops".to_string())),
            field("verdict", RecordValue::Ident("PASS".to_string())),
        ]
    }

    // ── AT-2901 / AT-2902: well-formed accept, typed fields ──────────────────

    #[test]
    fn well_formed_entry_is_accepted_with_typed_fields() {
        let entry = validate_entry(&valid_entry_fields(), 1, span()).expect("must validate");
        assert_eq!(entry.ts, "2026-07-14T12:00:00.000Z");
        assert_eq!(entry.kernel, "q4km_matmul");
        assert_eq!(entry.metric, 42860);
        assert_eq!(entry.unit, OptLogUnit::MilliTflops);
        assert_eq!(entry.verdict, OptLogVerdict::Pass);
        assert!(entry.note.is_none());
    }

    #[test]
    fn optional_note_is_accepted() {
        let mut fields = valid_entry_fields();
        fields.push(field("note", RecordValue::Str("leader".to_string())));
        let entry = validate_entry(&fields, 1, span()).expect("must validate");
        assert_eq!(entry.note.as_deref(), Some("leader"));
    }

    #[test]
    fn missing_required_field_rejected() {
        let mut fields = valid_entry_fields();
        fields.retain(|f| f.key.node != "kernel");
        let err = validate_entry(&fields, 1, span()).unwrap_err();
        assert!(matches!(err, HirError::BadOptimizationLogEntry { .. }));
    }

    #[test]
    fn duplicate_key_rejected() {
        let mut fields = valid_entry_fields();
        fields.push(field("ts", RecordValue::Str("2026-07-14T12:00:00.000Z".to_string())));
        let err = validate_entry(&fields, 1, span()).unwrap_err();
        assert!(matches!(err, HirError::BadOptimizationLogEntry { .. }));
    }

    #[test]
    fn wrong_shape_rejected() {
        let mut fields = valid_entry_fields();
        fields.retain(|f| f.key.node != "metric");
        fields.push(field("metric", RecordValue::Str("not an int".to_string())));
        let err = validate_entry(&fields, 1, span()).unwrap_err();
        assert!(matches!(err, HirError::BadOptimizationLogEntry { .. }));
    }

    #[test]
    fn out_of_set_unit_rejected() {
        let mut fields = valid_entry_fields();
        fields.retain(|f| f.key.node != "unit");
        fields.push(field("unit", RecordValue::Ident("gigaflops".to_string())));
        let err = validate_entry(&fields, 1, span()).unwrap_err();
        assert!(matches!(err, HirError::BadOptimizationLogEntry { .. }));
    }

    #[test]
    fn out_of_set_verdict_rejected() {
        let mut fields = valid_entry_fields();
        fields.retain(|f| f.key.node != "verdict");
        fields.push(field("verdict", RecordValue::Ident("MAYBE".to_string())));
        let err = validate_entry(&fields, 1, span()).unwrap_err();
        assert!(matches!(err, HirError::BadOptimizationLogEntry { .. }));
    }

    // ── AT-2907 (schema half): exhaustive verdict ident round-trip ───────────

    #[test]
    fn at_2907_all_seven_verdict_idents_round_trip() {
        for ident in ["PASS", "FAIL", "REJECT", "SKIPPED", "NONDETERMINISTIC_ORACLE", "ERROR", "REGRESS"] {
            let v = OptLogVerdict::from_ident(ident).unwrap_or_else(|| panic!("must parse {ident}"));
            assert_eq!(v.as_ident(), ident, "round-trip must be identity for {ident}");
        }
    }

    #[test]
    fn all_six_units_round_trip() {
        for ident in ["mtflops", "pct_milli", "ns", "us", "ms", "bytes"] {
            let u = OptLogUnit::from_ident(ident).unwrap_or_else(|| panic!("must parse {ident}"));
            assert_eq!(u.as_ident(), ident);
        }
    }

    // ── AT-2920: sign rule per unit ───────────────────────────────────────────

    #[test]
    fn at_2920_negative_metric_rejected_for_mtflops() {
        let mut fields = valid_entry_fields();
        fields.retain(|f| f.key.node != "metric");
        fields.push(field("metric", RecordValue::Int(-5)));
        let err = validate_entry(&fields, 1, span()).unwrap_err();
        assert!(matches!(err, HirError::BadOptimizationLogEntry { .. }));
    }

    #[test]
    fn at_2920_negative_metric_accepted_for_pct_milli() {
        let mut fields = valid_entry_fields();
        fields.retain(|f| f.key.node != "metric" && f.key.node != "unit");
        fields.push(field("metric", RecordValue::Int(-5)));
        fields.push(field("unit", RecordValue::Ident("pct_milli".to_string())));
        let entry = validate_entry(&fields, 1, span()).expect("negative pct_milli must be accepted");
        assert_eq!(entry.metric, -5);
    }

    // ── AT-2923 / AT-2924: version + forward-compat ───────────────────────────

    #[test]
    fn at_2923_v1_missing_version_rejected() {
        let fields: Vec<RecordField> = Vec::new();
        let err = validate_block(&fields, span()).unwrap_err();
        assert!(matches!(err, HirError::BadOptimizationLog { .. }));
    }

    #[test]
    fn at_2923_v1_duplicate_version_rejected() {
        let fields = vec![field("version", RecordValue::Int(1)), field("version", RecordValue::Int(1))];
        let err = validate_block(&fields, span()).unwrap_err();
        assert!(matches!(err, HirError::BadOptimizationLog { .. }));
    }

    #[test]
    fn at_2923_v1_negative_version_rejected() {
        let fields = vec![field("version", RecordValue::Int(-1))];
        let err = validate_block(&fields, span()).unwrap_err();
        assert!(matches!(err, HirError::BadOptimizationLog { .. }));
    }

    #[test]
    fn at_2923_v1_non_int_version_rejected() {
        let fields = vec![field("version", RecordValue::Str("1".to_string()))];
        let err = validate_block(&fields, span()).unwrap_err();
        assert!(matches!(err, HirError::BadOptimizationLog { .. }));
    }

    #[test]
    fn at_2923_v1_unknown_block_key_rejected() {
        let fields = vec![field("version", RecordValue::Int(1)), field("future_field", RecordValue::Int(1))];
        let err = validate_block(&fields, span()).unwrap_err();
        assert!(matches!(err, HirError::BadOptimizationLog { .. }));
    }

    #[test]
    fn at_2923_v1_unknown_entry_key_rejected() {
        let mut fields = valid_entry_fields();
        fields.push(field("future_entry_field", RecordValue::Int(1)));
        let err = validate_entry(&fields, 1, span()).unwrap_err();
        assert!(matches!(err, HirError::BadOptimizationLogEntry { .. }));
    }

    #[test]
    fn at_2924_v2_unknown_block_key_preserved_not_rejected() {
        let fields = vec![field("version", RecordValue::Int(2)), field("future_field", RecordValue::Int(7))];
        let log = validate_block(&fields, span()).expect("v2 unknown key must NOT reject");
        assert_eq!(log.version, 2);
        assert_eq!(log.forward_unknown, vec![("future_field".to_string(), RecordValue::Int(7))]);
    }

    #[test]
    fn at_2924_v2_unknown_entry_key_preserved_known_fields_still_validated() {
        let mut fields = valid_entry_fields();
        fields.push(field("future_entry_field", RecordValue::Str("x".to_string())));
        let entry = validate_entry(&fields, 2, span()).expect("v2 unknown key must NOT reject");
        assert_eq!(entry.kernel, "q4km_matmul");
        assert_eq!(entry.forward_unknown, vec![("future_entry_field".to_string(), RecordValue::Str("x".to_string()))]);
    }

    #[test]
    fn v1_well_formed_block_validates() {
        let fields = vec![field("version", RecordValue::Int(1))];
        let log = validate_block(&fields, span()).expect("must validate");
        assert_eq!(log.version, 1);
        assert!(log.entries.is_empty());
        assert!(log.forward_unknown.is_empty());
    }

    // ── RFC3339 shape check ────────────────────────────────────────────────

    #[test]
    fn rfc3339_shape_accepts_and_rejects() {
        assert!(looks_like_rfc3339("2026-07-14T12:00:00.000Z"));
        assert!(looks_like_rfc3339("2026-07-14T12:00:00Z"));
        assert!(!looks_like_rfc3339("not-a-timestamp"));
        assert!(!looks_like_rfc3339("2026/07/14 12:00:00"));
    }
}
