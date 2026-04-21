//! `optimization_history` tool — read JSONL history for a source file.
//!
//! Reads entries from `.pipeline/history/<xxh3_hex16(source)>.jsonl` in file
//! order (oldest first). Malformed lines are skipped with a `skipped_lines`
//! counter increment.

use std::collections::BTreeMap;
use std::io::BufRead;
use std::path::PathBuf;

use xxhash_rust::xxh3::xxh3_64;

use crate::mcp::dispatch::{McpContext, McpToolError};
use crate::mcp::tools::bench_variant::MachineMetadata;
use crate::mcp::tools::grid_search::{RankedVariant, HistoryEntryRecord, history_path_for_source, hex16};

// ── Request / Response types ──────────────────────────────────────────────────

/// Request for the `optimization_history` tool.
#[derive(Debug, serde::Deserialize)]
pub struct OptHistoryRequest {
    /// Path to the `.axc` source file whose history to retrieve.
    pub source_path: PathBuf,
}

/// Response from the `optimization_history` tool.
#[derive(Debug, serde::Serialize)]
pub struct OptHistoryResponse {
    /// xxh3_hex16 of the source file content.
    pub source_xxh3: String,
    /// Path to the JSONL history file.
    pub history_path: PathBuf,
    /// History entries in file order (oldest first).
    pub entries: Vec<HistoryEntry>,
    /// Number of malformed lines skipped.
    pub skipped_lines: u32,
}

/// On-wire entry for the `optimization_history` tool.
///
/// Mirrors `HistoryEntryRecord` but is a distinct struct to decouple the
/// wire format from the on-disk format.
#[derive(Debug, serde::Serialize)]
pub struct HistoryEntry {
    /// RFC3339 UTC timestamp with millisecond resolution.
    pub timestamp: String,
    /// Best-effort git HEAD SHA (7 chars), or `None`.
    pub git_sha: Option<String>,
    /// xxh3_hex16 of the source.
    pub source_xxh3: String,
    /// Winner variant_id.
    pub winner_variant_id: u64,
    /// Winner assignments (BTreeMap for determinism).
    pub winner_assignments: BTreeMap<String, i64>,
    /// Winner median timing in ns.
    pub winner_median_ns: u64,
    /// Full ranked list from this grid_search run.
    pub ranked: Vec<RankedVariant>,
    /// Host machine metadata.
    pub machine: MachineMetadata,
}

// ── Handler ───────────────────────────────────────────────────────────────────

/// Handle an `optimization_history` request.
pub(crate) fn handle(
    req: OptHistoryRequest,
    ctx: &mut McpContext,
) -> Result<OptHistoryResponse, McpToolError> {
    let source: String = std::fs::read_to_string(&req.source_path)
        .map_err(McpToolError::Io)?;

    let source_xxh3: String = hex16(xxh3_64(source.as_bytes()));
    let history_path: PathBuf = history_path_for_source(&ctx.history_dir, &source);

    if !history_path.exists() {
        return Ok(OptHistoryResponse {
            source_xxh3,
            history_path,
            entries: Vec::new(),
            skipped_lines: 0,
        });
    }

    let file = std::fs::File::open(&history_path).map_err(McpToolError::Io)?;
    let reader = std::io::BufReader::new(file);

    let mut entries: Vec<HistoryEntry> = Vec::new();
    let mut skipped_lines: u32 = 0;

    for line_result in reader.lines() {
        let line: String = match line_result {
            Ok(l) => l,
            Err(e) => {
                tracing::warn!(reason = %e, "optimization_history: error reading line, skipping");
                skipped_lines += 1;
                continue;
            }
        };
        let trimmed: &str = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match serde_json::from_str::<HistoryEntryRecord>(trimmed) {
            Ok(record) => {
                entries.push(record_to_entry(record));
            }
            Err(e) => {
                tracing::warn!(
                    reason = %e,
                    line_preview = %&trimmed[..trimmed.len().min(80)],
                    "optimization_history: skipping malformed line"
                );
                skipped_lines += 1;
            }
        }
    }

    Ok(OptHistoryResponse {
        source_xxh3,
        history_path,
        entries,
        skipped_lines,
    })
}

/// Convert a `HistoryEntryRecord` (on-disk) to a `HistoryEntry` (on-wire).
fn record_to_entry(r: HistoryEntryRecord) -> HistoryEntry {
    HistoryEntry {
        timestamp: r.timestamp,
        git_sha: r.git_sha,
        source_xxh3: r.source_xxh3,
        winner_variant_id: r.grid_search.winner_variant_id,
        winner_assignments: r.grid_search.winner_assignments,
        winner_median_ns: r.grid_search.winner_median_ns,
        ranked: r.grid_search.ranked,
        machine: r.grid_search.machine,
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::tools::grid_search::{HistoryEntryRecord, GridSearchPersisted};

    /// AT-1120: optimization_history skips malformed lines and reports skipped_lines count.
    #[test]
    fn at_1120_history_read_skips_malformed_lines() {
        use std::io::Write;
        let mut tmp = tempfile::NamedTempFile::new().expect("tempfile");

        // Write 3 lines: valid, garbage, valid
        let valid1 = serde_json::to_string(&make_record("r1")).unwrap();
        let valid2 = serde_json::to_string(&make_record("r2")).unwrap();
        writeln!(tmp, "{valid1}").unwrap();
        writeln!(tmp, "{{not valid json at all }}}}}}").unwrap();
        writeln!(tmp, "{valid2}").unwrap();
        tmp.flush().unwrap();

        // Manually read using the internal logic
        let file = std::fs::File::open(tmp.path()).unwrap();
        let reader = std::io::BufReader::new(file);
        let mut entries: Vec<HistoryEntry> = Vec::new();
        let mut skipped: u32 = 0;

        for line_result in reader.lines() {
            let line = line_result.unwrap();
            let trimmed = line.trim();
            if trimmed.is_empty() { continue; }
            match serde_json::from_str::<HistoryEntryRecord>(trimmed) {
                Ok(r) => entries.push(record_to_entry(r)),
                Err(_) => skipped += 1,
            }
        }

        assert_eq!(entries.len(), 2, "must have 2 valid entries");
        assert_eq!(skipped, 1, "must have 1 skipped line");
    }

    fn make_record(label: &str) -> HistoryEntryRecord {
        HistoryEntryRecord {
            timestamp: "2026-04-18T00:00:00.000Z".to_string(),
            git_sha: None,
            source_xxh3: format!("{label}_hash"),
            grid_search: GridSearchPersisted {
                winner_variant_id: 1,
                winner_assignments: BTreeMap::new(),
                winner_median_ns: 1000,
                ranked: vec![],
                machine: MachineMetadata {
                    device_name: "test".to_string(),
                    driver_version: 0,
                    api_version: 0,
                    physical_device_index: 0,
                    icd_path: None,
                    git_sha: None,
                },
            },
        }
    }
}
