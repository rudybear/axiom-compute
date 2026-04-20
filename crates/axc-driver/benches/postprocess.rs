//! Post-processing bench (M2.2): reads Criterion `estimates.json` files,
//! builds a `baselines.json` candidate, and optionally blesses it.
//!
//! This is a thin Criterion bench whose actual work is post-processing rather
//! than measuring a hot loop.  It runs last (fourth `[[bench]]` entry in
//! Cargo.toml — AT-708) so all other groups have already written their
//! Criterion output.
//!
//! # Blessing workflow (AT-711)
//!
//! - Without `AXC_BLESS_BASELINES=1`: writes candidate to
//!   `target/axc-bench/candidate-baselines.json` (gitignored).
//! - With `AXC_BLESS_BASELINES=1`: additionally overwrites
//!   `.pipeline/benchmarks/baselines.json` (git-tracked).
//!
//! # Schema v1 (AT-710)
//!
//! Serialized via BTreeMap (not HashMap) to guarantee field-order stability.
//! The `benchmarks` array is sorted by `(group, bench)` before serialization.
//!
//! # AT-710d: sanity gate
//!
//! Every recorded bench must have `median_ns > 0`.  An unexpected Criterion
//! schema shape surfaces a clear error.

#[path = "common.rs"]
mod common;

use criterion::{criterion_group, criterion_main, Criterion};
use serde::{Deserialize, Serialize};
use std::io::Write;

// ── Schema v1 structs (BTreeMap for stable serialization order) ────────────────

/// Full baselines.json document (schema version 1).
#[derive(Debug, Serialize, Deserialize)]
struct BaselineFile {
    schema_version: u32,
    generated: String,
    git_sha: String,
    machine: MachineMeta,
    benchmarks: Vec<BenchEntry>,
}

/// Machine metadata sub-object (AT-709).
///
/// All fields are populated; empty string for unavailable probes (never null).
#[derive(Debug, Serialize, Deserialize)]
struct MachineMeta {
    os: String,
    rustc: String,
    vulkan_icd: String,
    vulkan_device: String,
    cpu_model: String,
    axc_version: String,
}

/// One bench entry in the `benchmarks` array.
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
struct BenchEntry {
    group: String,
    bench: String,
    median_ns: u64,
    low_ns: u64,
    high_ns: u64,
}

// ── Criterion estimates.json shape (what Criterion writes) ─────────────────────

/// The subset of Criterion's `estimates.json` we consume.
///
/// Shape-verified for forward compatibility (AT-710d): if `mean` or its
/// `point_estimate` field is missing, we emit a clear error.
#[derive(Debug, Deserialize)]
struct CriterionEstimates {
    mean: Option<CriterionStat>,
    median: Option<CriterionStat>,
    slope: Option<CriterionStat>,
}

#[derive(Debug, Deserialize, Clone)]
struct CriterionStat {
    confidence_interval: Option<CriterionCI>,
    point_estimate: Option<f64>,
}

#[derive(Debug, Deserialize, Clone)]
struct CriterionCI {
    lower_bound: Option<f64>,
    upper_bound: Option<f64>,
}

// ── Known bench groups and their Criterion output directory names ──────────────

/// Each entry: (group_name, criterion_bench_name_on_disk, bench_field_name_in_json)
const KNOWN_BENCHES: &[(&str, &str, &str)] = &[
    // compile_pipeline group
    ("compile_pipeline", "compile/compile_saxpy", "compile_saxpy"),
    ("compile_pipeline", "compile/compile_vector_add", "compile_vector_add"),
    // cpu_reference group
    ("cpu_reference", "cpu_reference/cpu_saxpy_1024", "cpu_saxpy_1024"),
    ("cpu_reference", "cpu_reference/cpu_saxpy_1m", "cpu_saxpy_1m"),
    ("cpu_reference", "cpu_reference/cpu_vector_add_1024", "cpu_vector_add_1024"),
    ("cpu_reference", "cpu_reference/cpu_vector_add_1m", "cpu_vector_add_1m"),
    // dispatch_gpu group
    ("dispatch_gpu", "dispatch/dispatch_saxpy_1024", "dispatch_saxpy_1024"),
    ("dispatch_gpu", "dispatch/dispatch_saxpy_1m", "dispatch_saxpy_1m"),
    ("dispatch_gpu", "dispatch/dispatch_vector_add_1024", "dispatch_vector_add_1024"),
    ("dispatch_gpu", "dispatch/dispatch_vector_add_1m", "dispatch_vector_add_1m"),
];

// ── Helper: probe git SHA ──────────────────────────────────────────────────────

/// Probe the short git SHA via `git rev-parse --short HEAD`.
///
/// Returns empty string on any failure.
fn probe_git_sha() -> String {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output();
    match out {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().to_owned(),
        _ => String::new(),
    }
}

/// Probe `rustc --version`.
///
/// Returns empty string on any failure.
fn probe_rustc_version() -> String {
    let out = std::process::Command::new("rustc").arg("--version").output();
    match out {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().to_owned(),
        _ => String::new(),
    }
}

/// Probe OS name string.
fn probe_os() -> &'static str {
    if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else {
        "other"
    }
}

/// Probe Vulkan ICD path from env vars (AT-709).
fn probe_vulkan_icd() -> String {
    std::env::var("VK_DRIVER_FILES")
        .or_else(|_| std::env::var("VK_ICD_FILENAMES"))
        .unwrap_or_default()
}

/// Probe Vulkan device name by constructing a temporary VulkanContext.
///
/// Returns empty string if Vulkan is unavailable or construction fails.
fn probe_vulkan_device() -> String {
    match axc_runtime::VulkanContext::new() {
        Ok(ctx) => ctx.physical_device_name().to_owned(),
        Err(_) => String::new(),
    }
}

// ── Main postprocess function ──────────────────────────────────────────────────

fn run_postprocess() {
    // Locate target directory (CARGO_TARGET_DIR or default `target/`).
    let manifest_dir: String = std::env::var("CARGO_MANIFEST_DIR")
        .unwrap_or_else(|_| ".".to_owned());
    let target_dir: std::path::PathBuf =
        std::env::var("CARGO_TARGET_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                std::path::PathBuf::from(&manifest_dir)
                    .join("..")
                    .join("..")
                    .join("target")
            });

    // Collect bench entries from Criterion output.
    let mut entries: Vec<BenchEntry> = Vec::new();
    let mut errors: Vec<String> = Vec::new();

    for &(group, criterion_path, bench) in KNOWN_BENCHES {
        let estimates_path = target_dir
            .join("criterion")
            .join(criterion_path)
            .join("new")
            .join("estimates.json");

        if !estimates_path.exists() {
            // Bench was skipped (e.g. dispatch_gpu without Vulkan) — skip entry.
            eprintln!("postprocess: skipping '{}' (no estimates.json)", criterion_path);
            continue;
        }

        let raw: String = match std::fs::read_to_string(&estimates_path) {
            Ok(s) => s,
            Err(e) => {
                errors.push(format!("read {}: {e}", estimates_path.display()));
                continue;
            }
        };

        let est: CriterionEstimates = match serde_json::from_str(&raw) {
            Ok(v) => v,
            Err(e) => {
                errors.push(format!(
                    "AT-710d: parse {}: {e}\nraw: {}",
                    estimates_path.display(),
                    &raw[..raw.len().min(200)]
                ));
                continue;
            }
        };

        let stat: CriterionStat = match est.median.or(est.slope).or(est.mean) {
            Some(s) => s,
            None => {
                errors.push(format!(
                    "AT-710d: '{}' has no median/slope/mean — unexpected Criterion schema",
                    estimates_path.display()
                ));
                continue;
            }
        };

        let median_ns: u64 = match stat.point_estimate {
            Some(p) => p.round() as u64,
            None => {
                errors.push(format!(
                    "AT-710d: '{}' stat has no point_estimate",
                    estimates_path.display()
                ));
                continue;
            }
        };

        // AT-710d sanity.
        if median_ns == 0 {
            errors.push(format!(
                "AT-710d: bench '{}' has median_ns=0 — degenerate estimates",
                estimates_path.display()
            ));
            continue;
        }

        let (low_ns, high_ns): (u64, u64) = if let Some(ci) = stat.confidence_interval {
            let low = ci.lower_bound.unwrap_or(median_ns as f64).round() as u64;
            let high = ci.upper_bound.unwrap_or(median_ns as f64).round() as u64;
            (low, high)
        } else {
            (median_ns, median_ns)
        };

        entries.push(BenchEntry {
            group: group.to_owned(),
            bench: bench.to_owned(),
            median_ns,
            low_ns,
            high_ns,
        });
    }

    if !errors.is_empty() {
        for e in &errors {
            eprintln!("postprocess ERROR: {e}");
        }
        panic!("postprocess: {} error(s) reading Criterion output — see above", errors.len());
    }

    if entries.is_empty() {
        eprintln!("postprocess: no Criterion entries found (all benches skipped?) — writing empty benchmarks array");
    }

    // Sort for stable output (AT-710, BTreeMap discipline).
    entries.sort();

    // Build machine metadata.
    let git_sha: String = probe_git_sha();
    let machine: MachineMeta = MachineMeta {
        os: probe_os().to_owned(),
        rustc: probe_rustc_version(),
        vulkan_icd: probe_vulkan_icd(),
        vulkan_device: probe_vulkan_device(),
        cpu_model: common::cpu_model_probe(),
        axc_version: env!("CARGO_PKG_VERSION").to_owned(),
    };

    // Use BTreeMap to control JSON key order (AT-710).
    // We serialize the struct directly — serde will emit fields in declaration order.
    // The struct fields are already ordered per schema v1.
    let baseline = BaselineFile {
        schema_version: 1,
        generated: chrono_now_rfc3339(),
        git_sha: git_sha.clone(),
        machine,
        benchmarks: entries,
    };

    let json: String = serde_json::to_string_pretty(&baseline)
        .expect("postprocess: failed to serialize baselines JSON");

    // Write candidate (always).
    let candidate_dir: std::path::PathBuf = target_dir.join("axc-bench");
    std::fs::create_dir_all(&candidate_dir)
        .expect("postprocess: failed to create target/axc-bench/");
    let candidate_path: std::path::PathBuf = candidate_dir.join("candidate-baselines.json");
    let mut f = std::fs::File::create(&candidate_path)
        .expect("postprocess: failed to create candidate-baselines.json");
    f.write_all(json.as_bytes())
        .expect("postprocess: failed to write candidate-baselines.json");
    eprintln!("postprocess: wrote candidate to {}", candidate_path.display());

    // Bless if AXC_BLESS_BASELINES=1 (AT-711).
    if std::env::var("AXC_BLESS_BASELINES").as_deref() == Ok("1") {
        let repo_root: std::path::PathBuf = std::path::PathBuf::from(&manifest_dir)
            .join("..")
            .join("..");
        let blessed_dir: std::path::PathBuf = repo_root.join(".pipeline").join("benchmarks");
        std::fs::create_dir_all(&blessed_dir)
            .expect("postprocess: failed to create .pipeline/benchmarks/");
        let blessed_path: std::path::PathBuf = blessed_dir.join("baselines.json");
        let mut bf = std::fs::File::create(&blessed_path)
            .expect("postprocess: failed to create baselines.json for blessing");
        bf.write_all(json.as_bytes())
            .expect("postprocess: failed to write blessed baselines.json");
        eprintln!("postprocess: BLESSED baselines to {}", blessed_path.display());
    } else {
        eprintln!(
            "postprocess: set AXC_BLESS_BASELINES=1 to promote candidate to .pipeline/benchmarks/baselines.json"
        );
    }
}

/// Return the current UTC time as an RFC 3339 string.
///
/// Uses only `std` (no `chrono` dep) — format: `2026-04-18T12:00:00Z`.
fn chrono_now_rfc3339() -> String {
    // Use UNIX timestamp via SystemTime.
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs: u64 = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Simple proleptic Gregorian calendar calculation.
    let s = secs % 60;
    let m = (secs / 60) % 60;
    let h = (secs / 3600) % 24;
    let days = secs / 86400;
    // Calculate year and day-of-year using civil calendar algorithm (Howard Hinnant).
    let z = days as i64 + 719468;
    let era: i64 = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe: i64 = z - era * 146097;
    let yoe: i64 = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y: i64 = yoe + era * 400;
    let doy: i64 = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp: i64 = (5 * doy + 2) / 153;
    let d: i64 = doy - (153 * mp + 2) / 5 + 1;
    let mo: i64 = if mp < 10 { mp + 3 } else { mp - 9 };
    let yr: i64 = if mo <= 2 { y + 1 } else { y };
    format!("{yr:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

/// Criterion bench function wrapping the postprocess logic.
///
/// This is a "bench" only in the Criterion harness sense; it runs once and
/// does not produce meaningful timing output.  Its purpose is to be the last
/// `[[bench]]` target executed by `cargo bench -p axc-driver`.
fn postprocess_bench(c: &mut Criterion) {
    c.bench_function("postprocess_baselines", |b| {
        b.iter(|| {
            run_postprocess();
        });
    });
}

criterion_group!(postprocess_group, postprocess_bench);
criterion_main!(postprocess_group);
