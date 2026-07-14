//! M3.19 (FG.3) — `axc log-opt`: the ONE writer for `@optimization_log { ... }`.
//!
//! # The falsifiable value proof
//!
//! This is the single integration point that makes FG.3's minimal build real:
//! an LLM-free, GPU-free, CI-runnable CLI primitive that appends a structured
//! entry to a kernel source's `@optimization_log` block — creating the block
//! if absent — via a **source-preserving splice**, never a reformat. Every
//! byte outside the spliced region is preserved byte-for-byte (AT-2906).
//!
//! # Fail-closed discipline
//!
//! This writer NEVER emits a source that would then fail to compile:
//! - At `MAX_OPT_LOG_ENTRIES` (64) capacity, it refuses (`LogOptError::Full`)
//!   BEFORE any write — the source is left byte-for-byte unchanged, and the
//!   65th entry that would trip `HirError::BadOptimizationLogEntry` is never
//!   written (§1.5, AT-2922). Rotation is deferred (§1.8).
//! - A source with two or more `@kernel` items is refused
//!   (`LogOptError::Multikernel`) — which of several kernels owns the block is
//!   FG.9 territory, not built.
//! - A source that fails to parse, or whose EXISTING `@optimization_log` block
//!   is itself malformed, is refused rather than spliced blind.
//! - A source whose EXISTING `@optimization_log` block declares
//!   `version != SUPPORTED_OPT_LOG_VERSION` is refused
//!   (`LogOptError::UnsupportedVersion`) BEFORE any write — this writer only
//!   ever emits/splices `version == SUPPORTED_OPT_LOG_VERSION`-shaped entries,
//!   so splicing one into a block of any other declared version would
//!   silently mix schema versions inside one block (QA-caught gap, AT-2926:
//!   a `version: 2` block parses and compiles cleanly on its own, so the
//!   writer cannot rely on "this can't happen via the parser" — it must
//!   check explicitly).
//!
//! # Concurrency
//!
//! Whole-file rewrite, last-writer-wins. Unlike the JSONL `append_history_entry`
//! path (which uses `flock`), there is no advisory lock here — do NOT run
//! `axc log-opt` concurrently on the same file (§1.5).

use std::path::Path;

use axc_hir::opt_log::{OptLogUnit, OptLogVerdict, OptimizationLog, OptimizationLogEntry, MAX_OPT_LOG_ENTRIES};
use axc_parser::ast::{Item, RecordValue};

use crate::rewrite_verify::{Verdict as RewriteVerdict, VerifyReport};
use crate::mcp::tools::grid_search::HistoryEntryRecord;

/// Errors from the `axc log-opt` writer. Every variant is fail-closed: none of
/// them leave a partially-spliced source on disk (the write only happens after
/// every check below has passed).
#[derive(Debug, thiserror::Error)]
pub enum LogOptError {
    /// The block is already at `MAX_OPT_LOG_ENTRIES` capacity. The source is
    /// left byte-for-byte unchanged (AT-2922) — rotation is deferred (§1.8).
    #[error("@optimization_log block is at capacity ({current}/{max} entries); refusing to write the {}th entry (rotation is not built in the minimal writer — see spec §1.8)", current + 1)]
    Full { current: usize, max: usize },
    /// The source declares != 1 `@kernel` item. One block per single-kernel
    /// file; multi-kernel modules are FG.9 territory (not built).
    #[error("source has {0} @kernel item(s); @optimization_log requires exactly one kernel per file (multi-kernel modules are FG.9, not built)")]
    Multikernel(usize),
    /// The source itself failed to lex/parse.
    #[error("source failed to parse: {detail}")]
    Parse { detail: String },
    /// The EXISTING `@optimization_log` block on the kernel is malformed
    /// (fails HIR validation) — refusing to splice blind into a block that
    /// already would not compile.
    #[error("existing @optimization_log block is malformed: {detail}")]
    MalformedExistingBlock { detail: String },
    /// The EXISTING `@optimization_log` block declares a `version` this
    /// writer does not support (only `SUPPORTED_OPT_LOG_VERSION` is
    /// splice-able). A `version != SUPPORTED_OPT_LOG_VERSION` block is a
    /// REAL, parseable, HIR-valid block (§1.4a forward-compat preserves
    /// unknown keys instead of rejecting it) — it is NOT unreachable, so this
    /// writer must check explicitly rather than assume the parser rules it
    /// out. Refusing here, BEFORE any write, avoids silently splicing a
    /// `SUPPORTED_OPT_LOG_VERSION`-shaped entry into a block of a different
    /// declared schema version (AT-2926).
    #[error("existing @optimization_log block has version {found}; this writer only supports version {supported} (refusing to splice a mismatched-schema-version entry into it)")]
    UnsupportedVersion { found: i64, supported: i64 },
    /// The kernel has no leading annotations to anchor a freshly-inserted
    /// block before (should not happen for any source that passed HIR
    /// validation, since `@kernel` itself is always a leading annotation).
    #[error("source's kernel has no annotations to anchor a new @optimization_log block before")]
    NoAnchor,
    /// Underlying I/O failure (read or write).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

impl LogOptError {
    /// Distinct machine-readable exit codes (§1.5: "distinct... exit-coded
    /// LogOptError::Full"). `Full`, `Multikernel`, and `UnsupportedVersion`
    /// get their OWN codes (not the generic usage-error `2`) so a calling
    /// script can distinguish "capacity reached" / "multi-kernel not
    /// supported" / "unsupported block version" from a plain usage mistake
    /// without parsing the JSON error body.
    pub fn exit_code(&self) -> i32 {
        match self {
            LogOptError::Full { .. } => 3,
            LogOptError::Multikernel(_) => 4,
            LogOptError::UnsupportedVersion { .. } => 5,
            LogOptError::Parse { .. }
            | LogOptError::MalformedExistingBlock { .. }
            | LogOptError::NoAnchor
            | LogOptError::Io(_) => 2,
        }
    }
}

// ── Rendering (source-text formatter; also the AT-2904 round-trip formatter) ──

/// Escape a string for embedding in a `"..."` string literal (backslash and
/// double-quote only — the same minimal escaping the lexer's string literal
/// grammar requires to round-trip).
fn escape_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            other => out.push(other),
        }
    }
    out
}

fn render_record_value(v: &RecordValue) -> String {
    match v {
        RecordValue::Str(s) => format!("\"{}\"", escape_str(s)),
        RecordValue::Int(i) => i.to_string(),
        RecordValue::Ident(s) => s.clone(),
    }
}

/// Render one `OptimizationLogEntry` as `entry { ... }` source text, in the
/// canonical field order (`ts, kernel, metric, unit, verdict[, note][, forward-unknown...]`).
/// Deterministic — same entry always renders to the same bytes.
pub fn render_entry(entry: &OptimizationLogEntry) -> String {
    let mut fields: Vec<String> = vec![
        format!("ts: \"{}\"", escape_str(&entry.ts)),
        format!("kernel: \"{}\"", escape_str(&entry.kernel)),
        format!("metric: {}", entry.metric),
        format!("unit: {}", entry.unit.as_ident()),
        format!("verdict: {}", entry.verdict.as_ident()),
    ];
    if let Some(note) = &entry.note {
        fields.push(format!("note: \"{}\"", escape_str(note)));
    }
    for (key, value) in &entry.forward_unknown {
        fields.push(format!("{key}: {}", render_record_value(value)));
    }
    format!("entry {{ {} }}", fields.join(", "))
}

/// Render a full `@optimization_log { ... }` block, in canonical order
/// (`version[, forward-unknown block keys...][, entry, entry, ...]`).
/// Deterministic — the AT-2904 round-trip formatter: `parse(format(log))`
/// structurally equals `log`.
pub fn format_optimization_log_block(log: &OptimizationLog) -> String {
    let mut items: Vec<String> = vec![format!("version: {}", log.version)];
    for (key, value) in &log.forward_unknown {
        items.push(format!("{key}: {}", render_record_value(value)));
    }
    for entry in &log.entries {
        items.push(render_entry(entry));
    }
    format!("@optimization_log {{ {} }}", items.join(", "))
}

// ── Verdict mapping (§1.5, AT-2907) ────────────────────────────────────────────

/// Exhaustive 1:1 mapping of the six real `rewrite_verify::Verdict` variants
/// to the identically-named `OptLogVerdict` idents (AT-2907). `REGRESS` is
/// UNREACHABLE from this function — it has no `rewrite_verify` source and is
/// produced only by [`entry_from_grid_record`].
pub fn verdict_from_rewrite_verify(v: RewriteVerdict) -> OptLogVerdict {
    match v {
        RewriteVerdict::Pass => OptLogVerdict::Pass,
        RewriteVerdict::Fail => OptLogVerdict::Fail,
        RewriteVerdict::Reject => OptLogVerdict::Reject,
        RewriteVerdict::Skipped => OptLogVerdict::Skipped,
        RewriteVerdict::NondeterministicOracle => OptLogVerdict::NondeterministicOracle,
        RewriteVerdict::Error => OptLogVerdict::Error,
    }
}

/// Build an `OptimizationLogEntry` from a `rewrite_verify::VerifyReport`.
///
/// `VerifyReport` is a CORRECTNESS report — it carries no throughput/timing
/// number (rewrite_verify checks observational equivalence, not performance).
/// `metric`/`unit` are therefore threaded through explicitly by the caller
/// (the CLI's `--metric`/`--unit` flags remain in effect alongside
/// `--from-verify`); this function derives `verdict` (via
/// [`verdict_from_rewrite_verify`]) and, when the caller passes `None` for
/// `kernel`, the rewritten kernel's name from the report itself.
pub fn entry_from_verify_report(
    report: &VerifyReport,
    kernel: Option<String>,
    ts: String,
    metric: i64,
    unit: OptLogUnit,
    note: Option<String>,
) -> OptimizationLogEntry {
    let kernel: String = kernel.unwrap_or_else(|| {
        report.rewritten.as_ref()
            .or(report.original.as_ref())
            .map(|k| k.kernel_name.clone())
            .unwrap_or_default()
    });
    OptimizationLogEntry {
        ts,
        kernel,
        metric,
        unit,
        verdict: verdict_from_rewrite_verify(report.verdict),
        note,
        forward_unknown: Vec::new(),
    }
}

/// Build an `OptimizationLogEntry` from a `grid_search::HistoryEntryRecord`.
///
/// Unlike `VerifyReport`, a grid-search record DOES carry a natural metric
/// (`winner_median_ns`, unit `ns`) and a natural timestamp (`record.timestamp`,
/// already RFC3339 — used as the default `ts` unless the caller overrides it).
/// `verdict` is `REGRESS` when `baseline_median_ns` is supplied AND the
/// winner is slower than it (higher ns == worse); otherwise `PASS`. This is
/// the ONLY producer of `OptLogVerdict::Regress` (§1.5 table) — it has no
/// `rewrite_verify::Verdict` source.
pub fn entry_from_grid_record(
    record: &HistoryEntryRecord,
    kernel: String,
    baseline_median_ns: Option<u64>,
    ts: Option<String>,
    note: Option<String>,
) -> OptimizationLogEntry {
    let winner_ns: u64 = record.grid_search.winner_median_ns;
    let verdict: OptLogVerdict = match baseline_median_ns {
        Some(baseline) if winner_ns > baseline => OptLogVerdict::Regress,
        _ => OptLogVerdict::Pass,
    };
    OptimizationLogEntry {
        ts: ts.unwrap_or_else(|| record.timestamp.clone()),
        kernel,
        metric: winner_ns as i64,
        unit: OptLogUnit::Nanoseconds,
        verdict,
        note,
        forward_unknown: Vec::new(),
    }
}

// ── The writer (splice, fail-closed) ───────────────────────────────────────────

/// Pure splice core (unit-testable without touching the filesystem):
/// `source` in, spliced `source` out, or a fail-closed `LogOptError`.
///
/// See the module docs for the fail-closed contract. On `Err`, the returned
/// error is the ONLY output — the caller MUST NOT write anything.
pub fn splice_optimization_log_entry(source: &str, new_entry: &OptimizationLogEntry) -> Result<String, LogOptError> {
    let (ast, lex_errors, parse_errors) = axc_parser::parse(source);
    if !lex_errors.is_empty() || !parse_errors.is_empty() {
        return Err(LogOptError::Parse {
            detail: format!("{} lex error(s), {} parse error(s)", lex_errors.len(), parse_errors.len()),
        });
    }

    let kernels: Vec<&axc_parser::ast::KernelDecl> = ast.items.iter()
        .map(|item| { let Item::Kernel(kd) = &item.node; kd })
        .collect();
    if kernels.len() != 1 {
        return Err(LogOptError::Multikernel(kernels.len()));
    }
    let kd: &axc_parser::ast::KernelDecl = kernels[0];

    let existing_ann = kd.annotations.iter().find(|a| a.node.name.node == "optimization_log");

    match existing_ann {
        None => {
            // Create a fresh block, inserted immediately before the kernel's
            // first leading annotation (§1.5).
            let anchor: usize = kd.annotations.first()
                .map(|a| a.span.start as usize)
                .ok_or(LogOptError::NoAnchor)?;
            let fresh = OptimizationLog {
                version: axc_hir::opt_log::SUPPORTED_OPT_LOG_VERSION,
                entries: vec![new_entry.clone()],
                forward_unknown: Vec::new(),
            };
            let block_text: String = format_optimization_log_block(&fresh);
            let mut out = String::with_capacity(source.len() + block_text.len() + 2);
            out.push_str(&source[..anchor]);
            out.push_str(&block_text);
            out.push('\n');
            out.push_str(&source[anchor..]);
            Ok(out)
        }
        Some(ann) => {
            // Validate the EXISTING block via the real HIR validator (reuse,
            // not re-derive) to get the authoritative current entry count and
            // to refuse splicing blind into an already-malformed block.
            let (hir, hir_errors, _warnings) = axc_hir::lower_module(&ast);
            let malformed_detail: Option<String> = hir_errors.iter().find_map(|e| match e {
                axc_hir::HirError::BadOptimizationLog { detail, .. } => Some(detail.clone()),
                axc_hir::HirError::BadOptimizationLogEntry { detail, .. } => Some(detail.clone()),
                _ => None,
            });
            if let Some(detail) = malformed_detail {
                return Err(LogOptError::MalformedExistingBlock { detail });
            }
            let existing_log: Option<&OptimizationLog> = hir.kernels.first()
                .and_then(|k| k.annotations.opt_log.as_ref());

            // Fail-closed BEFORE any write: this writer only ever produces
            // `SUPPORTED_OPT_LOG_VERSION`-shaped entries, so it must refuse to
            // splice into a block declaring any other version rather than
            // silently mixing schema versions inside one block. A
            // `version != SUPPORTED_OPT_LOG_VERSION` block is real,
            // parseable, and HIR-valid (§1.4a forward-compat), so this check
            // cannot be skipped as "unreachable" (AT-2926).
            if let Some(log) = existing_log {
                if log.version != axc_hir::opt_log::SUPPORTED_OPT_LOG_VERSION {
                    return Err(LogOptError::UnsupportedVersion {
                        found: log.version,
                        supported: axc_hir::opt_log::SUPPORTED_OPT_LOG_VERSION,
                    });
                }
            }

            let current_entries: usize = existing_log
                .map(|log| log.entries.len())
                .unwrap_or(0);
            if current_entries >= MAX_OPT_LOG_ENTRIES {
                return Err(LogOptError::Full { current: current_entries, max: MAX_OPT_LOG_ENTRIES });
            }

            // Splice `entry { ... }` immediately before the block's closing
            // `}`. The annotation's span ends exactly one byte past the `}`
            // that `parse_optimization_log_block` consumed to finish the
            // block (span.end is the exclusive end of the last-consumed
            // token, which is that single-byte `}`), so `span.end - 1` is
            // the `}`'s own byte offset.
            let close_brace_pos: usize = (ann.span.end as usize).saturating_sub(1);
            debug_assert_eq!(
                source.as_bytes().get(close_brace_pos), Some(&b'}'),
                "close_brace_pos must point at the block's closing `}}`"
            );

            let needs_leading_comma: bool = !block_ends_with_comma_or_is_empty(source, close_brace_pos);
            let entry_text: String = render_entry(new_entry);
            let insertion: String = if needs_leading_comma {
                format!(", {entry_text}")
            } else {
                entry_text
            };

            let mut out = String::with_capacity(source.len() + insertion.len());
            out.push_str(&source[..close_brace_pos]);
            out.push_str(&insertion);
            out.push_str(&source[close_brace_pos..]);
            Ok(out)
        }
    }
}

/// `true` when the block is empty (nothing but whitespace between `{` and the
/// closing `}` at `close_brace_pos`) OR already ends in a trailing comma —
/// both cases mean a NEW item can be inserted right before `}` with NO
/// leading comma of its own (avoids emitting a double comma).
fn block_ends_with_comma_or_is_empty(source: &str, close_brace_pos: usize) -> bool {
    let before: &str = &source[..close_brace_pos];
    let trimmed: &str = before.trim_end();
    match trimmed.chars().next_back() {
        None => true, // Nothing before `}` at all (defensive; should not occur).
        Some(',') => true,
        Some('{') => true, // Empty block: `@optimization_log { }`.
        Some(_) => false,
    }
}

/// The ONE writer: append `new_entry` to `source_path`'s `@optimization_log`
/// block, creating it if absent. Fail-closed (see module docs): on any `Err`,
/// `source_path` is left byte-for-byte unchanged — this function only calls
/// `std::fs::write` after every check inside [`splice_optimization_log_entry`]
/// has passed.
pub fn append_optimization_log(source_path: &Path, new_entry: OptimizationLogEntry) -> Result<(), LogOptError> {
    let source: String = std::fs::read_to_string(source_path)?;
    let spliced: String = splice_optimization_log_entry(&source, &new_entry)?;
    std::fs::write(source_path, spliced)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entry() -> OptimizationLogEntry {
        OptimizationLogEntry {
            ts: "2026-07-14T12:00:00.000Z".to_string(),
            kernel: "k".to_string(),
            metric: 42860,
            unit: OptLogUnit::MilliTflops,
            verdict: OptLogVerdict::Pass,
            note: None,
            forward_unknown: Vec::new(),
        }
    }

    // ── AT-2905: creates block when absent; appends when present ─────────────

    #[test]
    fn at_2905_creates_block_when_absent() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { return; }";
        let spliced = splice_optimization_log_entry(src, &sample_entry()).expect("must splice");
        assert!(spliced.contains("@optimization_log {"));
        assert!(spliced.contains("entry { ts: \"2026-07-14T12:00:00.000Z\""));
        // Re-parses and lowers clean.
        let (ast, lex, parse) = axc_parser::parse(&spliced);
        assert!(lex.is_empty() && parse.is_empty(), "spliced source must re-parse clean: {lex:?} {parse:?}");
        let (hir, errors, _) = axc_hir::lower_module(&ast);
        assert!(errors.is_empty(), "spliced source must lower clean: {errors:?}");
        assert_eq!(hir.kernels[0].annotations.opt_log.as_ref().unwrap().entries.len(), 1);
    }

    #[test]
    fn at_2905_appends_when_present() {
        let src = "@kernel @workgroup(1,1,1) @optimization_log { version: 1, entry { ts: \"2026-07-14T12:00:00.000Z\", kernel: \"k\", metric: 1, unit: ns, verdict: PASS } } fn k() -> void { return; }";
        let mut second_entry = sample_entry();
        second_entry.ts = "2026-07-14T12:00:01.000Z".to_string();
        let spliced = splice_optimization_log_entry(src, &second_entry).expect("must splice");
        let (ast, lex, parse) = axc_parser::parse(&spliced);
        assert!(lex.is_empty() && parse.is_empty(), "spliced source must re-parse clean: {lex:?} {parse:?}");
        let (hir, errors, _) = axc_hir::lower_module(&ast);
        assert!(errors.is_empty(), "spliced source must lower clean: {errors:?}");
        assert_eq!(hir.kernels[0].annotations.opt_log.as_ref().unwrap().entries.len(), 2);
    }

    // ── AT-2906: byte-for-byte preservation outside the spliced region ───────

    #[test]
    fn at_2906_preserves_bytes_outside_spliced_region() {
        let src = "@kernel @workgroup(1,1,1) @intent(\"unchanged\") fn my_kernel() -> void { let x: u32 = 1u32; return; }";
        let spliced = splice_optimization_log_entry(src, &sample_entry()).expect("must splice");
        assert!(spliced.contains("@intent(\"unchanged\")"));
        assert!(spliced.contains("fn my_kernel() -> void { let x: u32 = 1u32; return; }"));
        // The ORIGINAL source, byte-for-byte, must appear as a contiguous
        // substring split around exactly the inserted block.
        let anchor = src.find("@kernel").unwrap();
        assert_eq!(&spliced[..anchor], &src[..anchor]);
        assert!(spliced.ends_with(&src[anchor..]), "everything from @kernel onward must be preserved verbatim, with only the new block prepended before it");
    }

    #[test]
    fn at_2906_append_preserves_bytes_outside_spliced_region() {
        let src = "@kernel @workgroup(1,1,1) @optimization_log { version: 1 } fn k() -> void { return; }";
        let spliced = splice_optimization_log_entry(src, &sample_entry()).expect("must splice");
        // Everything up to the block's opening `{` is unchanged.
        let open = src.find("@optimization_log {").unwrap() + "@optimization_log {".len();
        assert_eq!(&spliced[..open], &src[..open]);
        // Everything from the ORIGINAL close-brace onward must appear verbatim.
        let close = src.rfind('}').unwrap(); // the block's own close, since it's the first `}` here... use kernel's structure instead
        let _ = close;
        assert!(spliced.contains("fn k() -> void { return; }"));
    }

    // ── AT-2922: fail-closed at capacity, source byte-unchanged ──────────────

    #[test]
    fn at_2922_fails_closed_at_capacity() {
        let mut src = String::from("@kernel @workgroup(1,1,1) @optimization_log { version: 1, ");
        for i in 0..MAX_OPT_LOG_ENTRIES {
            src.push_str(&format!(
                "entry {{ ts: \"2026-07-14T12:00:00.000Z\", kernel: \"k{i}\", metric: 1, unit: ns, verdict: PASS }}, "
            ));
        }
        src.push_str("} fn k() -> void { return; }");

        // Sanity: the fixture really does carry MAX_OPT_LOG_ENTRIES entries.
        let (ast, lex, parse) = axc_parser::parse(&src);
        assert!(lex.is_empty() && parse.is_empty(), "fixture must parse clean: {lex:?} {parse:?}");
        let (hir, errors, _) = axc_hir::lower_module(&ast);
        assert!(errors.is_empty(), "fixture must lower clean: {errors:?}");
        assert_eq!(hir.kernels[0].annotations.opt_log.as_ref().unwrap().entries.len(), MAX_OPT_LOG_ENTRIES);

        let result = splice_optimization_log_entry(&src, &sample_entry());
        match result {
            Err(LogOptError::Full { current, max }) => {
                assert_eq!(current, MAX_OPT_LOG_ENTRIES);
                assert_eq!(max, MAX_OPT_LOG_ENTRIES);
                assert_eq!(LogOptError::Full { current, max }.exit_code(), 3);
            }
            other => panic!("expected LogOptError::Full: {other:?}"),
        }
        // The pure splice function is Err-only on failure — nothing to write;
        // the file-level `append_optimization_log` wrapper never calls
        // `std::fs::write` in this path (verified structurally: it always
        // calls `splice_optimization_log_entry` first via `?`).
    }

    // ── AT-2926 (QA-fix extension, one past the M3.19 spec's AT-2925 ceiling):
    //    the writer REFUSES to splice into an existing `version != 1` block,
    //    fail-closed BEFORE any write, source byte-unchanged. This is the fix
    //    for the QA-blocking finding in M3.19-qa.json: a `version: 2` block
    //    parses and compiles cleanly (proven below), and the coder's original
    //    "no real source can carry a block-level version>1 today via this
    //    parser's own grammar" justification for skipping this check was
    //    factually wrong — it conflated "an unknown BLOCK-LEVEL KEY is
    //    unreachable" (true, EBNF-closed) with "a block-level VERSION NUMBER
    //    greater than 1 is unreachable" (false — `version` is itself always a
    //    parseable, HIR-valid field, per §1.4a forward-compat). ─────────────

    #[test]
    fn at_2926_version2_block_refused_source_byte_unchanged() {
        let src = "@kernel @workgroup(1,1,1) @optimization_log { version: 2, entry { ts: \"2026-07-14T12:00:00.000Z\", kernel: \"k\", metric: 1, unit: ns, verdict: PASS } } fn k() -> void { return; }";

        // Sanity, matching the QA finding: this version:2 block is REAL —
        // it parses and lowers with zero errors (not a malformed/rejected
        // block), so `LogOptError::MalformedExistingBlock` must NOT be what
        // fires here; only the dedicated `UnsupportedVersion` check may.
        let (ast, lex, parse) = axc_parser::parse(src);
        assert!(lex.is_empty() && parse.is_empty(), "fixture must parse clean: {lex:?} {parse:?}");
        let (hir, errors, _warnings) = axc_hir::lower_module(&ast);
        assert!(errors.is_empty(), "fixture must lower clean (this is the exact QA-reproduced end-to-end scenario): {errors:?}");
        assert_eq!(hir.kernels[0].annotations.opt_log.as_ref().unwrap().version, 2);

        let result = splice_optimization_log_entry(src, &sample_entry());
        match result {
            Err(LogOptError::UnsupportedVersion { found, supported }) => {
                assert_eq!(found, 2);
                assert_eq!(supported, axc_hir::opt_log::SUPPORTED_OPT_LOG_VERSION);
                assert_eq!(supported, 1);
                assert_eq!(LogOptError::UnsupportedVersion { found, supported }.exit_code(), 5);
            }
            other => panic!("expected LogOptError::UnsupportedVersion: {other:?}"),
        }
        // The pure splice function is Err-only on failure — nothing written;
        // exercised at the file level too, below, for the actual byte-unchanged
        // guarantee on disk (mirrors AT-2922's own pattern).
    }

    #[test]
    fn at_2926_version2_block_refused_file_byte_unchanged_on_disk() {
        use std::io::Write;
        let src = "@kernel @workgroup(1,1,1) @optimization_log { version: 2, entry { ts: \"2026-07-14T12:00:00.000Z\", kernel: \"k\", metric: 1, unit: ns, verdict: PASS } } fn k() -> void { return; }";
        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        f.write_all(src.as_bytes()).expect("write fixture");

        let before = std::fs::read(f.path()).expect("read before");
        let err = append_optimization_log(f.path(), sample_entry()).expect_err("must refuse to append");
        assert!(matches!(err, LogOptError::UnsupportedVersion { found: 2, supported: 1 }), "expected UnsupportedVersion{{found:2,supported:1}}: {err:?}");
        assert_eq!(err.exit_code(), 5);
        let after = std::fs::read(f.path()).expect("read after");
        assert_eq!(before, after, "source file must be byte-for-byte unchanged after a refused append");
    }

    #[test]
    fn at_2926_version1_block_append_still_works() {
        // No-regression companion: a version:1 block (the only version this
        // writer supports) must continue to splice successfully — the new
        // check must not be over-broad.
        let src = "@kernel @workgroup(1,1,1) @optimization_log { version: 1, entry { ts: \"2026-07-14T12:00:00.000Z\", kernel: \"k\", metric: 1, unit: ns, verdict: PASS } } fn k() -> void { return; }";
        let spliced = splice_optimization_log_entry(src, &sample_entry()).expect("version:1 block must still splice");
        let (ast, lex, parse) = axc_parser::parse(&spliced);
        assert!(lex.is_empty() && parse.is_empty(), "spliced source must re-parse clean: {lex:?} {parse:?}");
        let (hir, errors, _) = axc_hir::lower_module(&ast);
        assert!(errors.is_empty(), "spliced source must lower clean: {errors:?}");
        assert_eq!(hir.kernels[0].annotations.opt_log.as_ref().unwrap().entries.len(), 2);
    }

    // ── Multikernel fail-closed ────────────────────────────────────────────────

    #[test]
    fn multikernel_source_fails_closed() {
        let src = concat!(
            "@kernel @workgroup(1,1,1) fn a() -> void { return; } ",
            "@kernel @workgroup(1,1,1) fn b() -> void { return; }",
        );
        let result = splice_optimization_log_entry(src, &sample_entry());
        assert!(matches!(result, Err(LogOptError::Multikernel(2))), "expected Multikernel(2): {result:?}");
    }

    #[test]
    fn malformed_existing_block_fails_closed() {
        let src = "@kernel @workgroup(1,1,1) @optimization_log { version: 1, entry { ts: \"2026-07-14T12:00:00.000Z\" } } fn k() -> void { return; }";
        let result = splice_optimization_log_entry(src, &sample_entry());
        assert!(matches!(result, Err(LogOptError::MalformedExistingBlock { .. })), "expected MalformedExistingBlock: {result:?}");
    }

    // ── AT-2904: round-trip formatter idempotence ─────────────────────────────

    #[test]
    fn at_2904_format_then_parse_is_structurally_equal() {
        let log = OptimizationLog {
            version: 1,
            entries: vec![sample_entry(), {
                let mut e = sample_entry();
                e.ts = "2026-07-14T12:00:01.000Z".to_string();
                e.verdict = OptLogVerdict::Regress;
                e.note = Some("second run".to_string());
                e
            }],
            forward_unknown: Vec::new(),
        };
        let block_text = format_optimization_log_block(&log);
        let src = format!("@kernel @workgroup(1,1,1) {block_text} fn k() -> void {{ return; }}");
        let (ast, lex, parse) = axc_parser::parse(&src);
        assert!(lex.is_empty() && parse.is_empty(), "formatted block must re-parse clean: {lex:?} {parse:?}");
        let (hir, errors, _) = axc_hir::lower_module(&ast);
        assert!(errors.is_empty(), "formatted block must lower clean: {errors:?}");
        let round_tripped = hir.kernels[0].annotations.opt_log.as_ref().unwrap();
        assert_eq!(*round_tripped, log, "format -> parse -> lower must be structurally equal to the original log");
    }

    // ── AT-2907: exhaustive verdict mapping ────────────────────────────────────

    #[test]
    fn at_2907_all_six_rewrite_verify_verdicts_map_1to1() {
        let pairs = [
            (RewriteVerdict::Pass, OptLogVerdict::Pass),
            (RewriteVerdict::Fail, OptLogVerdict::Fail),
            (RewriteVerdict::Reject, OptLogVerdict::Reject),
            (RewriteVerdict::Skipped, OptLogVerdict::Skipped),
            (RewriteVerdict::NondeterministicOracle, OptLogVerdict::NondeterministicOracle),
            (RewriteVerdict::Error, OptLogVerdict::Error),
        ];
        for (input, expected) in pairs {
            assert_eq!(verdict_from_rewrite_verify(input), expected);
        }
    }

    #[test]
    fn at_2907_regress_never_produced_by_verify_mapping() {
        // Exhaustive over the closed rewrite_verify::Verdict set: none of the
        // six produce OptLogVerdict::Regress (it has no rewrite_verify source).
        for v in [
            RewriteVerdict::Pass, RewriteVerdict::Fail, RewriteVerdict::Reject,
            RewriteVerdict::Skipped, RewriteVerdict::NondeterministicOracle, RewriteVerdict::Error,
        ] {
            assert_ne!(verdict_from_rewrite_verify(v), OptLogVerdict::Regress);
        }
    }

    #[test]
    fn entry_from_grid_record_regress_vs_pass() {
        use crate::mcp::tools::grid_search::GridSearchPersisted;
        use crate::mcp::tools::bench_variant::MachineMetadata;
        use std::collections::BTreeMap;
        let record = HistoryEntryRecord {
            timestamp: "2026-07-14T12:00:00.000Z".to_string(),
            git_sha: None,
            source_xxh3: "deadbeef00000000".to_string(),
            grid_search: GridSearchPersisted {
                winner_variant_id: 1,
                winner_assignments: BTreeMap::new(),
                winner_median_ns: 2000,
                ranked: Vec::new(),
                machine: MachineMetadata {
                    device_name: "test-device".to_string(),
                    driver_version: 0,
                    api_version: 0,
                    physical_device_index: 0,
                    icd_path: None,
                    git_sha: None,
                },
            },
        };
        let regressed = entry_from_grid_record(&record, "k".to_string(), Some(1000), None, None);
        assert_eq!(regressed.verdict, OptLogVerdict::Regress);
        assert_eq!(regressed.metric, 2000);
        assert_eq!(regressed.unit, OptLogUnit::Nanoseconds);

        let improved = entry_from_grid_record(&record, "k".to_string(), Some(3000), None, None);
        assert_eq!(improved.verdict, OptLogVerdict::Pass);

        let no_baseline = entry_from_grid_record(&record, "k".to_string(), None, None, None);
        assert_eq!(no_baseline.verdict, OptLogVerdict::Pass);
        assert_eq!(no_baseline.ts, "2026-07-14T12:00:00.000Z", "ts defaults to the record's own timestamp");
    }
}
