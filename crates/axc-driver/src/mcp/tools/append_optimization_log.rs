//! `append_optimization_log` tool (M3.19 / FG.3) — a THIN JSON-RPC wrapper
//! over `log_opt::append_optimization_log`, mirroring the M3.16
//! CLI-primitive-plus-MCP-tool pattern (`verify_rewrite` / `rewrite-verify`).
//!
//! This is the on-thesis consumer: the agent loop's MCP server can write
//! self-describing rewrite history directly into the `.axc` source — no
//! shell-out to `axc log-opt`, no side-channel file to agree on.
//!
//! Raw-fields only (no `--from-verify`/`--from-grid` JSON ingestion here —
//! an MCP caller already has the verdict/metric in hand from its own prior
//! `verify_rewrite`/`grid_search` call in the SAME process; re-serializing
//! to a temp file just to re-ingest it would be pure overhead for this
//! surface, unlike the standalone CLI which genuinely reads a JSON artifact
//! off disk).
//!
//! # `--ts` reproducibility (§1.5)
//!
//! Mirrors the CLI rule exactly: the default is wall-clock `now()` (fine for
//! production agent use), and it is the CALLER's responsibility to pass an
//! explicit `ts` for any reproducible/golden-fixture call — this wrapper does
//! not special-case that; it simply never overrides a caller-supplied `ts`.

use std::path::PathBuf;

use axc_hir::opt_log::{OptLogUnit, OptLogVerdict, OptimizationLogEntry};

use crate::log_opt::{append_optimization_log, LogOptError};
use crate::mcp::dispatch::McpToolError;
use crate::mcp::format_rfc3339_utc;

/// Request for the `append_optimization_log` tool.
#[derive(Debug, serde::Deserialize)]
pub struct AppendOptimizationLogRequest {
    /// Path to the `.axc` source file to append to (creates the block if absent).
    pub source_path: PathBuf,
    pub kernel: String,
    pub metric: i64,
    /// `mtflops` | `pct_milli` | `ns` | `us` | `ms` | `bytes`.
    pub unit: String,
    /// `PASS` | `FAIL` | `REJECT` | `SKIPPED` | `NONDETERMINISTIC_ORACLE` | `ERROR` | `REGRESS`.
    pub verdict: String,
    /// Explicit RFC3339 timestamp. `None` ⇒ wall-clock `now()` (non-reproducible;
    /// reproducible callers MUST pass this explicitly — §1.5).
    #[serde(default)]
    pub ts: Option<String>,
    #[serde(default)]
    pub note: Option<String>,
}

/// Response from the `append_optimization_log` tool.
#[derive(Debug, serde::Serialize)]
pub struct AppendOptimizationLogResponse {
    pub source_path: PathBuf,
    /// The entry that was written (echoed back for the caller's audit trail).
    pub kernel: String,
    pub metric: i64,
    pub unit: String,
    pub verdict: String,
    pub ts: String,
}

/// M3.23 (M3.19c): per-note byte cap on the untrusted-agent MCP surface — an
/// unbounded `note` lets an agent splice an arbitrarily long string into the
/// source. Measured in BYTES via `str::len()` (cheap, unambiguous; documented
/// as bytes rather than switched to `chars().count()` — see the milestone
/// spec §2.4). A note of exactly `MAX_NOTE_LEN` bytes is accepted (at-cap).
const MAX_NOTE_LEN: usize = 512;

/// Handle an `append_optimization_log` request.
pub(crate) fn handle(req: AppendOptimizationLogRequest) -> Result<AppendOptimizationLogResponse, McpToolError> {
    if let Some(ref note) = req.note {
        if note.len() > MAX_NOTE_LEN {
            return Err(McpToolError::InvalidParams(format!(
                "note exceeds MAX_NOTE_LEN (512 bytes): {} bytes",
                note.len()
            )));
        }
    }
    let unit: OptLogUnit = OptLogUnit::from_ident(&req.unit).ok_or_else(|| {
        McpToolError::InvalidParams(format!(
            "invalid `unit` `{}`; expected one of mtflops, pct_milli, ns, us, ms, bytes",
            req.unit
        ))
    })?;
    let verdict: OptLogVerdict = OptLogVerdict::from_ident(&req.verdict).ok_or_else(|| {
        McpToolError::InvalidParams(format!(
            "invalid `verdict` `{}`; expected one of PASS, FAIL, REJECT, SKIPPED, NONDETERMINISTIC_ORACLE, ERROR, REGRESS",
            req.verdict
        ))
    })?;
    let ts: String = req.ts.unwrap_or_else(|| format_rfc3339_utc(std::time::SystemTime::now()));

    let entry = OptimizationLogEntry {
        ts: ts.clone(),
        kernel: req.kernel.clone(),
        metric: req.metric,
        unit,
        verdict,
        note: req.note,
        forward_unknown: Vec::new(),
    };

    append_optimization_log(&req.source_path, entry).map_err(log_opt_error_to_mcp)?;

    Ok(AppendOptimizationLogResponse {
        source_path: req.source_path,
        kernel: req.kernel,
        metric: req.metric,
        unit: unit.as_ident().to_string(),
        verdict: verdict.as_ident().to_string(),
        ts,
    })
}

/// Map a fail-closed `LogOptError` to an `McpToolError`. Every variant maps
/// to `InvalidParams` (the caller's request could not be satisfied against
/// the given source) EXCEPT the underlying I/O failure, which is `Io`.
fn log_opt_error_to_mcp(e: LogOptError) -> McpToolError {
    match e {
        LogOptError::Io(io_err) => McpToolError::Io(io_err),
        other => McpToolError::InvalidParams(other.to_string()),
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_axc_file(contents: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        f.write_all(contents.as_bytes()).expect("write");
        f
    }

    #[test]
    fn invalid_unit_is_invalid_params() {
        let f = temp_axc_file("@kernel @workgroup(1,1,1) fn k() -> void { return; }");
        let req = AppendOptimizationLogRequest {
            source_path: f.path().to_path_buf(),
            kernel: "k".to_string(),
            metric: 1,
            unit: "gigaflops".to_string(),
            verdict: "PASS".to_string(),
            ts: Some("2026-07-14T12:00:00.000Z".to_string()),
            note: None,
        };
        let err = handle(req).unwrap_err();
        assert!(matches!(err, McpToolError::InvalidParams(_)));
    }

    #[test]
    fn invalid_verdict_is_invalid_params() {
        let f = temp_axc_file("@kernel @workgroup(1,1,1) fn k() -> void { return; }");
        let req = AppendOptimizationLogRequest {
            source_path: f.path().to_path_buf(),
            kernel: "k".to_string(),
            metric: 1,
            unit: "ns".to_string(),
            verdict: "MAYBE".to_string(),
            ts: Some("2026-07-14T12:00:00.000Z".to_string()),
            note: None,
        };
        let err = handle(req).unwrap_err();
        assert!(matches!(err, McpToolError::InvalidParams(_)));
    }

    /// AT-3012 (M3.19c): a note over `MAX_NOTE_LEN` (512 bytes) is rejected with
    /// `InvalidParams`; a note of EXACTLY 512 bytes is accepted (at-cap).
    #[test]
    fn at_3012_note_len_cap_512_bytes() {
        let base_req = |note: Option<String>| AppendOptimizationLogRequest {
            source_path: PathBuf::new(),
            kernel: "k".to_string(),
            metric: 1,
            unit: "ns".to_string(),
            verdict: "PASS".to_string(),
            ts: Some("2026-07-14T12:00:00.000Z".to_string()),
            note,
        };

        // 513 bytes: over cap, rejected BEFORE any file is touched (source_path
        // is intentionally empty/invalid — proves the cap check runs first).
        let over = "a".repeat(MAX_NOTE_LEN + 1);
        let err = handle(base_req(Some(over))).unwrap_err();
        match err {
            McpToolError::InvalidParams(detail) => {
                assert!(detail.contains("MAX_NOTE_LEN"), "detail: {detail}");
                assert!(detail.contains("513"), "detail: {detail}");
            }
            other => panic!("expected InvalidParams; got {other:?}"),
        }

        // Exactly 512 bytes: accepted (at-cap) — splice against a real temp file.
        let f = temp_axc_file("@kernel @workgroup(1,1,1) fn k() -> void { return; }");
        let mut at_cap_req = base_req(Some("a".repeat(MAX_NOTE_LEN)));
        at_cap_req.source_path = f.path().to_path_buf();
        let resp = handle(at_cap_req).expect("exactly-512-byte note must be accepted");
        assert_eq!(resp.kernel, "k");
    }

    /// AT-2912: MCP `append_optimization_log` produces a byte-identical result
    /// to the CLI (`axc log-opt`) for the same entry — verified here at the
    /// handler level: both go through the identical `log_opt::append_optimization_log`
    /// writer with the identical resolved `OptimizationLogEntry`, so the two
    /// surfaces cannot diverge (mirrors AT-2912's intent; the full
    /// process-boundary CLI-vs-MCP comparison lives in
    /// `tests/mcp_roundtrip.rs`, per the `at_1114`-pattern integration style).
    #[test]
    fn at_2912_mcp_and_writer_produce_identical_splice() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { return; }";
        let f = temp_axc_file(src);
        let req = AppendOptimizationLogRequest {
            source_path: f.path().to_path_buf(),
            kernel: "k".to_string(),
            metric: 42860,
            unit: "mtflops".to_string(),
            verdict: "PASS".to_string(),
            ts: Some("2026-07-14T12:00:00.000Z".to_string()),
            note: Some("leader".to_string()),
        };
        let resp = handle(req).expect("must succeed");
        assert_eq!(resp.kernel, "k");
        assert_eq!(resp.metric, 42860);
        assert_eq!(resp.unit, "mtflops");
        assert_eq!(resp.verdict, "PASS");
        assert_eq!(resp.ts, "2026-07-14T12:00:00.000Z");

        let via_mcp: String = std::fs::read_to_string(f.path()).expect("read spliced file");

        // Now compute the SAME splice directly via the writer core (the CLI's
        // code path), starting from the SAME original source, and assert the
        // two outputs are byte-identical.
        let entry = OptimizationLogEntry {
            ts: "2026-07-14T12:00:00.000Z".to_string(),
            kernel: "k".to_string(),
            metric: 42860,
            unit: OptLogUnit::MilliTflops,
            verdict: OptLogVerdict::Pass,
            note: Some("leader".to_string()),
            forward_unknown: Vec::new(),
        };
        let via_cli: String = crate::log_opt::splice_optimization_log_entry(src, &entry).expect("must splice");
        assert_eq!(via_mcp, via_cli, "MCP and CLI writer paths must produce byte-identical output for the same entry");
    }
}
