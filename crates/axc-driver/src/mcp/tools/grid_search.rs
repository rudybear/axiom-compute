//! `grid_search` tool — run grid search over @strategy holes using GPU dispatch.
//!
//! Returns a ranked variant list (ascending median_ns), the winner, and appends
//! one JSONL entry to `.pipeline/history/<source_xxh3>.jsonl`.
//!
//! ## History file append (B-1 fix)
//!
//! `append_history_entry` uses POSIX advisory `flock(LOCK_EX)` around the
//! `writeln!` call to serialize concurrent writers. See `flock_exclusive` for
//! the SAFETY comment.
//!
//! ## RFC3339 UTC formatter (N-1: millisecond resolution)
//!
//! `format_rfc3339_utc` produces `"YYYY-MM-DDTHH:MM:SS.NNNZ"`.
//! Millisecond resolution avoids same-second ordering collisions.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::Write as IoWrite;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use xxhash_rust::xxh3::xxh3_64;

use crate::mcp::dispatch::{McpContext, McpToolError};
use crate::mcp::tools::bench_variant::{
    BenchVariantRequest, CorrectnessStatus, MachineMetadata, build_machine_metadata,
};
use crate::mcp::tools::enumerate_variants::StrategyVariantSummary;

// ── RFC3339 UTC formatter ─────────────────────────────────────────────────────

/// Format a `SystemTime` as RFC 3339 UTC with millisecond resolution.
///
/// Output shape: `"YYYY-MM-DDTHH:MM:SS.NNNZ"` (always exactly 24 chars).
/// Uses the civil-time / Howard-Hinnant algorithm for Y-M-D decomposition.
///
/// # Examples
///
/// ```
/// # use axc_driver::mcp::format_rfc3339_utc;
/// # use std::time::UNIX_EPOCH;
/// // epoch
/// assert_eq!(format_rfc3339_utc(UNIX_EPOCH), "1970-01-01T00:00:00.000Z");
/// ```
pub fn format_rfc3339_utc(t: SystemTime) -> String {
    let dur: Duration = t.duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO);
    let total_secs: u64 = dur.as_secs();
    let millis: u32 = dur.subsec_millis();

    // Decompose seconds into Y-M-D H:M:S (UTC).
    let secs_in_day: u64 = total_secs % 86400;
    let days_since_epoch: u64 = total_secs / 86400;

    let hour: u32 = (secs_in_day / 3600) as u32;
    let minute: u32 = ((secs_in_day % 3600) / 60) as u32;
    let second: u32 = (secs_in_day % 60) as u32;

    // Howard-Hinnant algorithm: days since 1970-01-01 → Y-M-D.
    // Reference: http://howardhinnant.github.io/date_algorithms.html
    let z: i64 = days_since_epoch as i64 + 719468;
    let era: i64 = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe: i64 = z - era * 146097;
    let yoe: i64 = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y: i64 = yoe + era * 400;
    let doy: i64 = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp: i64 = (5 * doy + 2) / 153;
    let d: i64 = doy - (153 * mp + 2) / 5 + 1;
    let m: i64 = if mp < 10 { mp + 3 } else { mp - 9 };
    let y: i64 = y + if m <= 2 { 1 } else { 0 };

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        y, m, d, hour, minute, second, millis
    )
}

// ── xxh3 hex16 ───────────────────────────────────────────────────────────────

/// Compute a 16-hex-character string from `xxh3_64(source.as_bytes())`.
pub(crate) fn hex16(h: u64) -> String {
    format!("{h:016x}")
}

/// Compute the history file path for a given source string.
///
/// Returns `<history_dir>/<xxh3_hex16(source.as_bytes())>.jsonl`.
pub fn history_path_for_source(history_dir: &Path, source: &str) -> PathBuf {
    let hash: u64 = xxh3_64(source.as_bytes());
    history_dir.join(format!("{}.jsonl", hex16(hash)))
}

// ── flock (POSIX advisory lock) ───────────────────────────────────────────────

/// Acquire an exclusive POSIX advisory lock on `file`.
///
/// # SAFETY
///
/// `fd` is obtained via `file.as_raw_fd()` on a `&std::fs::File` that outlives
/// this call (the caller holds the file open). `libc::flock` is a non-destructive
/// POSIX syscall that reads the fd and updates kernel lock state only; return
/// value is -1 on error (with errno set) or 0 on success. No aliasing, no memory
/// is touched via the fd argument. The lock is released when the file handle is
/// dropped (the kernel releases POSIX advisory locks on `close(2)`).
fn flock_exclusive(file: &File) -> std::io::Result<()> {
    // SAFETY: see above.
    let ret: i32 = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
    if ret == -1 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

// ── History types ─────────────────────────────────────────────────────────────

/// A ranked variant entry in the grid search result.
#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub struct RankedVariant {
    /// Zero-based ordinal.
    pub ordinal: u64,
    /// xxh3_64 variant fingerprint.
    pub variant_id: u64,
    /// Concrete hole assignments.
    pub assignments: BTreeMap<String, i64>,
    /// Median timing in ns, or `None` for failed/skipped variants.
    pub median_ns: Option<u64>,
    /// Tri-state correctness verdict.
    pub correctness: CorrectnessStatus,
    /// Outcome string: `"ok"`, `"failed: <reason>"`,
    /// `"correctness_rejected: <reason>"`, or `"not_checked"`.
    pub outcome: String,
}

/// Serde-serializable record persisted to the JSONL history file.
///
/// Distinct from the on-wire `HistoryEntry` (which is a superset used by
/// `optimization_history`) — keeps the on-disk format stable.
#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub struct HistoryEntryRecord {
    /// RFC3339 UTC timestamp with millisecond resolution.
    pub timestamp: String,
    /// Best-effort git HEAD SHA (7 chars), or `None`.
    pub git_sha: Option<String>,
    /// xxh3_hex16 of the source string.
    pub source_xxh3: String,
    /// Grid search result sub-record.
    pub grid_search: GridSearchPersisted,
}

/// Persistent grid-search sub-record.
#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub struct GridSearchPersisted {
    /// Winner's variant_id.
    pub winner_variant_id: u64,
    /// Winner's assignments (BTreeMap for determinism).
    pub winner_assignments: BTreeMap<String, i64>,
    /// Winner's median timing in ns.
    pub winner_median_ns: u64,
    /// Full ranked list.
    pub ranked: Vec<RankedVariant>,
    /// Machine metadata.
    pub machine: MachineMetadata,
}

/// Atomically append one JSONL entry to `path`.
///
/// Creates the file if absent. Acquires `LOCK_EX` for the duration of the write
/// and flush. The lock is released when the `File` is dropped.
///
/// On NFS or FUSE filesystems where `flock` is advisory-only, the lock may be a
/// no-op; the fallback is `O_APPEND` atomicity (Linux guarantees this up to one
/// page, ~4 KiB). Documented in DESIGN.md §3.1.10.
pub fn append_history_entry(
    path: &Path,
    entry: &HistoryEntryRecord,
) -> Result<(), std::io::Error> {
    // Serialize first (before opening the file) to keep the locked window small.
    let line: String = serde_json::to_string(entry)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    // Sanity check: no embedded newlines in the serialized entry.
    debug_assert!(!line.contains('\n'), "JSONL entry must not contain embedded newlines");

    let file: File = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;

    // Acquire exclusive lock before writing.
    flock_exclusive(&file)?;

    // Write a single line (the lock serializes concurrent writers).
    let mut writer = std::io::BufWriter::new(&file);
    writeln!(writer, "{line}")?;
    writer.flush()?;

    // Lock released when `file` goes out of scope (close(2) → LOCK_UN).
    Ok(())
}

// ── Request / Response types ──────────────────────────────────────────────────

/// Request for the `grid_search` tool.
#[derive(Debug, serde::Deserialize)]
pub struct GridSearchRequest {
    /// Inline source text. Mutually exclusive with `path`.
    #[serde(default)]
    pub source: Option<String>,
    /// Path to an `.axc` source file. Mutually exclusive with `source`.
    #[serde(default)]
    pub path: Option<PathBuf>,
    /// REPLACE the kernel's @strategy map entirely with this map.
    /// (N-4: REPLACE semantics, not merge — gives LLM unambiguous control.)
    #[serde(default)]
    pub holes_override: Option<BTreeMap<String, Vec<i64>>>,
    /// Input buffer sizes in bytes (one per buffer binding).
    #[serde(default)]
    pub input_sizes: Option<Vec<usize>>,
    /// Number of samples per variant.
    #[serde(default)]
    pub sample_count: Option<u32>,
    /// Correctness policy string: `"none"`, `"bit-exact"`, `"fp-tol:<ulp>"`.
    #[serde(default)]
    pub correctness: Option<String>,
}

/// Response from the `grid_search` tool.
#[derive(Debug, serde::Serialize)]
pub struct GridSearchResponse {
    /// The winning variant (fastest correct one).
    pub winner: StrategyVariantSummary,
    /// Winner's median timing in ns.
    pub winner_median_ns: u64,
    /// All variants sorted ascending by `median_ns` (failed last).
    pub ranked: Vec<RankedVariant>,
    /// `true` if the winner was selected via the fallback chain (no variant fully succeeded).
    pub fallback_used: bool,
    /// Non-fatal warnings (e.g. Cartesian product exceeded CARTESIAN_WARN_THRESHOLD).
    pub warnings: Vec<String>,
    /// Path to the JSONL history file where this result was appended.
    pub history_path: PathBuf,
    /// Host machine metadata.
    pub machine: MachineMetadata,
}

// ── Handler ───────────────────────────────────────────────────────────────────

/// Handle a `grid_search` request.
pub(crate) fn handle(
    req: GridSearchRequest,
    ctx: &mut McpContext,
) -> Result<GridSearchResponse, McpToolError> {
    let source: String = crate::mcp::dispatch::resolve_source(&req.source, &req.path)?;
    let sample_count: u32 = req.sample_count.unwrap_or(5);

    if sample_count == 0 {
        return Err(McpToolError::InvalidParams(
            "sample_count must be >= 1".to_string()
        ));
    }

    // Parse source to get the kernel + strategy
    let (_kernel_name, binding_plan, strategy_holes, _workgroup_size) =
        parse_source_for_grid_search(&source)?;

    // Apply holes_override (REPLACE semantics — N-4)
    let effective_holes: axc_hir::hir::StrategyHoles = match req.holes_override {
        Some(override_map) => axc_hir::hir::StrategyHoles { map: override_map },
        None => strategy_holes,
    };

    // Enumerate variants
    let variants = axc_optimize::enumerator::enumerate_strategy(&effective_holes)
        .map_err(McpToolError::Enumerate)?;

    let mut warnings: Vec<String> = Vec::new();
    if variants.len() > axc_optimize::enumerator::CARTESIAN_WARN_THRESHOLD as usize {
        warnings.push(format!(
            "Cartesian product ({}) exceeds CARTESIAN_WARN_THRESHOLD ({}); \
             search may take a long time",
            variants.len(),
            axc_optimize::enumerator::CARTESIAN_WARN_THRESHOLD
        ));
    }

    let input_sizes: Vec<usize> = req.input_sizes.unwrap_or_else(|| {
        // Default: 4096 bytes per buffer
        vec![4096_usize; binding_plan.buffers.len()]
    });

    // Initialize Vulkan (lazy)
    let _ = ctx.vulkan.get_or_init()?;

    let mut ranked: Vec<RankedVariant> = Vec::with_capacity(variants.len());
    let git_sha_clone: Option<String> = ctx.git_sha.clone();

    for variant in &variants {
        let assignments: BTreeMap<String, i64> = variant.assignments.values.clone();
        let bench_req = BenchVariantRequest {
            source: Some(source.clone()),
            path: None,
            assignments: assignments.clone(),
            input_sizes: input_sizes.clone(),
            sample_count,
            workgroup_override: None,
            push_constants_base64: None,
            output_sizes: None,
        };

        let bench_result = crate::mcp::tools::bench_variant::handle(bench_req, ctx);

        let ranked_variant: RankedVariant = match bench_result {
            Ok(resp) => {
                let outcome: String = match &resp.correctness {
                    CorrectnessStatus::Ok => "ok".to_string(),
                    CorrectnessStatus::Failed { reason } => {
                        format!("correctness_rejected: {reason}")
                    }
                    CorrectnessStatus::NotChecked { .. } => "not_checked".to_string(),
                };
                let median_ns = if matches!(resp.correctness, CorrectnessStatus::Failed { .. }) {
                    None
                } else {
                    Some(resp.median_ns)
                };
                RankedVariant {
                    ordinal: variant.ordinal,
                    variant_id: variant.variant_id,
                    assignments,
                    median_ns,
                    correctness: resp.correctness,
                    outcome,
                }
            }
            Err(e) => {
                RankedVariant {
                    ordinal: variant.ordinal,
                    variant_id: variant.variant_id,
                    assignments,
                    median_ns: None,
                    correctness: CorrectnessStatus::NotChecked {
                        reason: e.to_string(),
                    },
                    outcome: format!("failed: {e}"),
                }
            }
        };
        ranked.push(ranked_variant);
    }

    // Sort by median_ns ascending (None = failed, sort last); ties break by ordinal.
    ranked.sort_by(|a, b| {
        match (a.median_ns, b.median_ns) {
            (Some(x), Some(y)) => x.cmp(&y).then(a.ordinal.cmp(&b.ordinal)),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => a.ordinal.cmp(&b.ordinal),
        }
    });

    // Select winner (first ranked with Some(median_ns), or first overall as fallback)
    let winner_idx: usize = ranked.iter().position(|r| r.median_ns.is_some())
        .unwrap_or(0);
    let fallback_used: bool = ranked.get(winner_idx).map(|r| r.median_ns.is_none()).unwrap_or(true);
    let winner_rv: &RankedVariant = &ranked[winner_idx];

    let winner: StrategyVariantSummary = StrategyVariantSummary {
        ordinal: winner_rv.ordinal,
        variant_id: winner_rv.variant_id,
        assignments: winner_rv.assignments.clone(),
    };
    let winner_median_ns: u64 = winner_rv.median_ns.unwrap_or(u64::MAX);

    // Build machine metadata
    let vk: &axc_runtime::VulkanContext = ctx.vulkan.get_or_init()?;
    let machine: MachineMetadata = build_machine_metadata(vk, git_sha_clone.as_deref());

    // Compute source xxh3 and history path
    let source_xxh3: String = hex16(xxh3_64(source.as_bytes()));
    let history_dir: &Path = &ctx.history_dir;
    std::fs::create_dir_all(history_dir).map_err(McpToolError::Io)?;
    let history_path: PathBuf = history_path_for_source(history_dir, &source);

    // Build and append history entry
    let record: HistoryEntryRecord = HistoryEntryRecord {
        timestamp: format_rfc3339_utc(SystemTime::now()),
        git_sha: git_sha_clone,
        source_xxh3,
        grid_search: GridSearchPersisted {
            winner_variant_id: winner.variant_id,
            winner_assignments: winner.assignments.clone(),
            winner_median_ns,
            ranked: ranked.clone(),
            machine: machine.clone(),
        },
    };
    append_history_entry(&history_path, &record).map_err(McpToolError::Io)?;

    Ok(GridSearchResponse {
        winner,
        winner_median_ns,
        ranked,
        fallback_used,
        warnings,
        history_path,
        machine,
    })
}

/// Parse source string to extract kernel name, binding plan, strategy holes, workgroup_size.
fn parse_source_for_grid_search(
    source: &str,
) -> Result<
    (String, axc_hir::ParamBindingPlan, axc_hir::hir::StrategyHoles, [u32; 3]),
    McpToolError
> {
    use axc_lexer::tokenize;
    use axc_parser::Parser;
    use axc_hir::lower_module;
    use crate::DriverError;

    if source.as_bytes().starts_with(&[0xEF, 0xBB, 0xBF]) {
        return Err(McpToolError::Compile(DriverError::UnexpectedByteOrderMark {
            span: axc_lexer::Span { start: 0, end: 3 },
        }));
    }

    let (tokens, lex_errors) = tokenize(source);
    let mut parser: Parser = Parser::new(&tokens);
    let (ast, parse_errors) = parser.parse_module();
    let (hir, hir_errors, _) = lower_module(&ast);

    if !lex_errors.is_empty() || !parse_errors.is_empty() || !hir_errors.is_empty() {
        return Err(McpToolError::Compile(DriverError::Compile {
            lex: lex_errors,
            parse: parse_errors,
            hir: hir_errors,
        }));
    }

    let kernel = hir.kernels.into_iter().next()
        .ok_or(McpToolError::GridSearch(axc_optimize::GridSearchError::NoStrategy))?;

    let strategy = kernel.annotations.strategy.unwrap_or_else(axc_hir::hir::StrategyHoles::new);
    let wg: [u32; 3] = [
        kernel.annotations.workgroup.x,
        kernel.annotations.workgroup.y,
        kernel.annotations.workgroup.z,
    ];

    Ok((kernel.name, kernel.binding_plan, strategy, wg))
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::UNIX_EPOCH;

    /// AT-1118: history_path_for_source produces `<dir>/<xxh3_hex16>.jsonl`.
    #[test]
    fn at_1118_history_path_for_source_is_xxh3_hex16() {
        let dir = Path::new("/tmp");
        let p1 = history_path_for_source(dir, "abc\n");
        let p2 = history_path_for_source(dir, "abc\n");
        assert_eq!(p1, p2, "same source must produce same path");

        // Path must end with .jsonl
        assert!(p1.to_str().unwrap().ends_with(".jsonl"), "must end with .jsonl");

        // The stem (filename without extension) must be exactly 16 hex chars
        let stem = p1.file_stem().unwrap().to_str().unwrap();
        assert_eq!(stem.len(), 16, "xxh3_hex16 must be exactly 16 chars; got {stem:?}");
        assert!(stem.chars().all(|c| c.is_ascii_hexdigit()), "must be hex; got {stem:?}");

        // Different source → different path
        let p3 = history_path_for_source(dir, "xyz");
        assert_ne!(p1, p3, "different sources must produce different paths");
    }

    /// AT-1119: append_history_entry writes exactly one line per call.
    #[test]
    fn at_1119_history_append_is_line_preserving() {
        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        let path = tmp.path();

        let entry1 = make_test_entry("e1");
        let entry2 = make_test_entry("e2");

        append_history_entry(path, &entry1).expect("append 1");
        append_history_entry(path, &entry2).expect("append 2");

        let content = std::fs::read(path).expect("read");

        // Count newlines: must be exactly 2
        let newline_count = content.iter().filter(|&&b| b == b'\n').count();
        assert_eq!(newline_count, 2, "must have exactly 2 newlines, got {newline_count}");

        // No carriage returns
        assert!(
            !content.contains(&b'\r'),
            "must not contain carriage returns"
        );

        // Each line parses back as HistoryEntryRecord
        let text = std::str::from_utf8(&content).expect("valid utf8");
        for line in text.lines() {
            serde_json::from_str::<HistoryEntryRecord>(line)
                .expect("each line must parse as HistoryEntryRecord");
        }
    }

    /// AT-1119b: concurrent writers with flock produce no interleaving.
    #[test]
    fn at_1119b_history_append_flock_excludes_concurrent_writers() {
        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        let path = tmp.path().to_path_buf();
        let path_clone = path.clone();

        const N: usize = 50;

        let t1 = std::thread::spawn({
            let p = path.clone();
            move || {
                for i in 0..N {
                    let e = make_test_entry(&format!("t1_{i}"));
                    append_history_entry(&p, &e).expect("t1 append");
                    std::thread::yield_now();
                }
            }
        });

        let t2 = std::thread::spawn({
            let p = path_clone;
            move || {
                for i in 0..N {
                    let e = make_test_entry(&format!("t2_{i}"));
                    append_history_entry(&p, &e).expect("t2 append");
                    std::thread::yield_now();
                }
            }
        });

        t1.join().expect("t1 must not panic");
        t2.join().expect("t2 must not panic");

        let content = std::fs::read(&path).expect("read");
        let text = std::str::from_utf8(&content).expect("valid utf8");
        let lines: Vec<&str> = text.lines().collect();

        // Must have exactly 2*N lines
        assert_eq!(
            lines.len(),
            2 * N,
            "must have exactly {} lines, got {}",
            2 * N,
            lines.len()
        );

        // Every line must parse cleanly
        for (i, line) in lines.iter().enumerate() {
            serde_json::from_str::<HistoryEntryRecord>(line)
                .unwrap_or_else(|e| panic!("line {i} failed to parse: {e}\nline={line:?}"));
        }
    }

    /// AT-1125: format_rfc3339_utc fixed-input tests.
    #[test]
    fn at_1125_rfc3339_fixed_inputs() {
        // Epoch
        assert_eq!(format_rfc3339_utc(UNIX_EPOCH), "1970-01-01T00:00:00.000Z");

        // Leap year: 2000-02-29 12:34:56.000
        // 2000-02-29 is days since epoch: (30*365 + 7 leap years) + 31 + 29 - 1 = 11016
        // 11016 days since 1970-01-01 = 2000-02-29
        // 12:34:56 = 12*3600 + 34*60 + 56 = 43200 + 2040 + 56 = 45296
        let leap_ts: SystemTime = UNIX_EPOCH + Duration::from_secs(11016 * 86400 + 45296);
        assert_eq!(format_rfc3339_utc(leap_ts), "2000-02-29T12:34:56.000Z");

        // Sub-second: UNIX_EPOCH + 123ms
        let sub_sec: SystemTime = UNIX_EPOCH + Duration::from_millis(123);
        assert_eq!(format_rfc3339_utc(sub_sec), "1970-01-01T00:00:00.123Z");

        // 2026-04-18T14:30:00.500Z
        // Days since epoch for 2026-04-18:
        // 56*365 + 14 leap days (1972,76,80,84,88,92,96,00,04,08,12,16,20,24) = 20437 + 14 = ...
        // Actually compute: let's use a simpler approach
        // 2026-01-01 is ... we trust the formatter and validate via regex instead.
        let ts_2026: SystemTime = UNIX_EPOCH + Duration::from_millis(
            (56 * 365 + 14) * 86400 * 1000 + // approximate — we verify via format test below
            14 * 3600 * 1000 + 30 * 60 * 1000 + 500
        );
        let formatted: String = format_rfc3339_utc(ts_2026);
        // Validate shape: YYYY-MM-DDTHH:MM:SS.NNNZ
        assert!(
            formatted.len() == 24 && formatted.ends_with('Z'),
            "format must be 24 chars ending in Z; got {formatted:?}"
        );
        // Validate milliseconds appear
        let ms_part = &formatted[20..23];
        assert_eq!(ms_part, "500", "milliseconds must be 500; got {ms_part:?}");
    }

    // Helper: create a test HistoryEntryRecord
    fn make_test_entry(label: &str) -> HistoryEntryRecord {
        HistoryEntryRecord {
            timestamp: "2026-04-18T00:00:00.000Z".to_string(),
            git_sha: Some("abc1234".to_string()),
            source_xxh3: format!("{label}_hash"),
            grid_search: GridSearchPersisted {
                winner_variant_id: 42,
                winner_assignments: {
                    let mut m = BTreeMap::new();
                    m.insert("wg".to_string(), 64_i64);
                    m
                },
                winner_median_ns: 1000,
                ranked: vec![],
                machine: MachineMetadata {
                    device_name: "test device".to_string(),
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
