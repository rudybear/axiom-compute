//! Regression gate for CPU reference benches (M2.2).
//!
//! Reads `.pipeline/benchmarks/baselines.json` (schema v1), runs the
//! `cpu_reference` subset `WARM_SAMPLES` times with wall-clock timing, takes
//! the median, and compares against the baseline.  Fails if any bench exceeds
//! `REGRESSION_THRESHOLD_PCT`%.
//!
//! # Environment gate (AT-712)
//!
//! The test body short-circuits when `AXC_ENABLE_BENCH_REGRESSION != "1"`.
//! This prevents the regression gate from running in every `cargo test` —
//! it is too slow and too noise-prone for every-change CI.
//!
//! # Statistical design (AT-713, C1 resolution)
//!
//! `WARM_SAMPLES = 11` (odd for a unique single-valued median).  Under the
//! assumption that cpu_reference noise is ~Gaussian with σ ≈ 8% of the mean
//! (from Lavapipe CI history), the 11-sample median has >95% power to detect
//! a true 15% mean shift vs ~60-70% for the 5-sample variant.
//!
//! # Fault-injection (AT-714)
//!
//! When `bench_regression_fixture_slowdown` is enabled, `saxpy_cpu_reference`
//! runs 100M extra accumulator iterations (common.rs).  This test MUST detect
//! the slowdown and panic with the AT-712 message pattern.

#[path = "../benches/common.rs"]
mod common;

use serde::Deserialize;

/// Number of warm samples for the regression gate (AT-713).
///
/// Odd so the median is a single unique value (no two-sample averaging).
const WARM_SAMPLES: usize = 11;

/// Regression threshold percentage (AT-712).
///
/// If `current_median > baseline.median_ns * (1 + REGRESSION_THRESHOLD_PCT / 100)`,
/// the test fails.
const REGRESSION_THRESHOLD_PCT: u32 = 15;

// ── Baseline JSON schema v1 (deserialization) ─────────────────────────────────

#[derive(Debug, Deserialize)]
struct BaselineFile {
    schema_version: u32,
    benchmarks: Vec<BenchEntry>,
}

#[derive(Debug, Deserialize)]
struct BenchEntry {
    group: String,
    bench: String,
    median_ns: u64,
}

// ── Regression outcome enum ────────────────────────────────────────────────────

#[derive(Debug, PartialEq)]
enum RegressionOutcome {
    Ok,
    PossibleSpeedup { pct: f64 },
    Regression { pct: f64 },
}

/// Compare current median against baseline; return the outcome.
fn check_regression(
    bench_name: &str,
    current_ns: u64,
    baseline_ns: u64,
    threshold_pct: u32,
) -> RegressionOutcome {
    if baseline_ns == 0 {
        // Cannot reason about a zero baseline; skip this bench.
        eprintln!("bench_regression: bench '{bench_name}' has baseline_ns=0 — skipping");
        return RegressionOutcome::Ok;
    }

    let ratio: f64 = current_ns as f64 / baseline_ns as f64;
    let pct: f64 = (ratio - 1.0) * 100.0;

    if ratio > 1.0 + (threshold_pct as f64 / 100.0) {
        return RegressionOutcome::Regression { pct };
    }
    if ratio < 1.0 - (threshold_pct as f64 / 100.0) {
        return RegressionOutcome::PossibleSpeedup { pct: pct.abs() };
    }
    RegressionOutcome::Ok
}

// ── Warm-sample timing helpers ────────────────────────────────────────────────

/// Run `common::saxpy_cpu_reference` `WARM_SAMPLES` times and return the
/// median wall-clock duration in nanoseconds.
fn run_cpu_saxpy_warm(n: usize, seed: u64) -> u64 {
    let (x, y, alpha) = common::make_saxpy_inputs(n, seed);
    let mut samples: Vec<u64> = Vec::with_capacity(WARM_SAMPLES);

    for _ in 0..WARM_SAMPLES {
        let t0 = std::time::Instant::now();
        let _result = common::saxpy_cpu_reference(&x, &y, alpha);
        let elapsed_ns: u64 = t0.elapsed().as_nanos() as u64;
        samples.push(elapsed_ns);
    }

    samples.sort_unstable();
    // WARM_SAMPLES = 11 is odd → index 5 is the unique median.
    samples[WARM_SAMPLES / 2]
}

/// Run `common::vector_add_cpu_reference` `WARM_SAMPLES` times and return the
/// median wall-clock duration in nanoseconds.
fn run_cpu_vector_add_warm(n: usize, seed: u64) -> u64 {
    let (a, b) = common::make_vector_add_inputs(n, seed);
    let mut samples: Vec<u64> = Vec::with_capacity(WARM_SAMPLES);

    for _ in 0..WARM_SAMPLES {
        let t0 = std::time::Instant::now();
        let _result = common::vector_add_cpu_reference(&a, &b);
        let elapsed_ns: u64 = t0.elapsed().as_nanos() as u64;
        samples.push(elapsed_ns);
    }

    samples.sort_unstable();
    samples[WARM_SAMPLES / 2]
}

// ── Bench name → (kernel, N) mapping ─────────────────────────────────────────

/// Dispatch a `cpu_reference` bench name to its wall-clock timing function.
///
/// Returns `None` if the name is not recognised (future-proofing).
fn time_cpu_reference_bench(bench: &str) -> Option<u64> {
    match bench {
        "cpu_saxpy_1024" => Some(run_cpu_saxpy_warm(1_024, common::SEED)),
        "cpu_saxpy_1m" => Some(run_cpu_saxpy_warm(1_048_576, common::SEED)),
        "cpu_vector_add_1024" => Some(run_cpu_vector_add_warm(1_024, common::SEED)),
        "cpu_vector_add_1m" => Some(run_cpu_vector_add_warm(1_048_576, common::SEED)),
        _ => {
            // Unknown bench name — future-proofing; skip gracefully.
            None
        }
    }
}

// ── Load baselines.json ────────────────────────────────────────────────────────

fn load_baselines() -> BaselineFile {
    let manifest_dir: String = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR not set");
    let baselines_path: std::path::PathBuf = std::path::PathBuf::from(&manifest_dir)
        .join("..")
        .join("..")
        .join(".pipeline")
        .join("benchmarks")
        .join("baselines.json");

    let raw: String = std::fs::read_to_string(&baselines_path).unwrap_or_else(|e| {
        panic!(
            "bench_regression: cannot read baselines at '{}': {e}\n\
             Run `AXC_BLESS_BASELINES=1 cargo bench -p axc-driver` first.",
            baselines_path.display()
        )
    });

    let file: BaselineFile = serde_json::from_str(&raw).unwrap_or_else(|e| {
        panic!(
            "bench_regression: failed to parse '{}': {e}",
            baselines_path.display()
        )
    });

    assert_eq!(
        file.schema_version, 1,
        "bench_regression: baselines.json schema_version must be 1, got {}",
        file.schema_version
    );

    file
}

// ── Main regression test ───────────────────────────────────────────────────────

/// AT-712: Run the regression gate.
///
/// Short-circuits when `AXC_ENABLE_BENCH_REGRESSION != "1"` (AT-712).
/// Fails if any `cpu_reference` bench exceeds the baseline by >15% (AT-713).
/// When `bench_regression_fixture_slowdown` is enabled, saxpy MUST trigger a
/// regression — the test asserts the gate is functional (AT-714).
#[test]
fn bench_regression_detects_over_threshold_slowdown() {
    if std::env::var("AXC_ENABLE_BENCH_REGRESSION").as_deref() != Ok("1") {
        eprintln!("skipping bench regression (AXC_ENABLE_BENCH_REGRESSION != 1)");
        return;
    }

    let baselines: BaselineFile = load_baselines();

    let mut regressions: Vec<String> = Vec::new();

    for entry in &baselines.benchmarks {
        // Only gate on cpu_reference benches (AT-712 intentional scope).
        // dispatch_gpu is too variable on Lavapipe; compile_pipeline has OS-noise dominance.
        if entry.group != "cpu_reference" {
            continue;
        }

        let current_ns: u64 = match time_cpu_reference_bench(&entry.bench) {
            Some(ns) => ns,
            None => {
                eprintln!(
                    "bench_regression: unknown bench '{}' in group '{}' — skipping",
                    entry.bench, entry.group
                );
                continue;
            }
        };

        let outcome: RegressionOutcome =
            check_regression(&entry.bench, current_ns, entry.median_ns, REGRESSION_THRESHOLD_PCT);

        match outcome {
            RegressionOutcome::Regression { pct } => {
                let msg: String = format!(
                    "regression: bench `{}` median {} ns exceeds baseline {} ns by {:.1}% (>{}% threshold)",
                    entry.bench, current_ns, entry.median_ns, pct, REGRESSION_THRESHOLD_PCT
                );
                eprintln!("{msg}");
                regressions.push(msg);
            }
            RegressionOutcome::PossibleSpeedup { pct } => {
                eprintln!(
                    "possible speedup: bench `{}` median {} ns is {:.1}% faster than baseline {} ns \
                     — consider blessing a new baseline",
                    entry.bench, current_ns, pct, entry.median_ns
                );
            }
            RegressionOutcome::Ok => {
                eprintln!(
                    "bench `{}` OK: {} ns (baseline {} ns)",
                    entry.bench, current_ns, entry.median_ns
                );
            }
        }
    }

    // AT-714: when fault-injection is enabled, the regressions list MUST be non-empty.
    #[cfg(feature = "bench_regression_fixture_slowdown")]
    {
        assert!(
            !regressions.is_empty(),
            "AT-714 VIOLATION: bench_regression_fixture_slowdown is enabled but NO regression \
             was detected — the regression gate is silently broken!"
        );
        // With fault-injection the test is EXPECTED to "fail" (regressions detected).
        // We return early after confirming detection works, so the test passes overall
        // (the CI fault-injection job asserts exit=1 at the shell level, not cargo level).
        eprintln!(
            "AT-714: fault-injection detected {} regression(s) as expected — gate is functional",
            regressions.len()
        );
        // Intentionally panic to produce exit=1 that the CI job checks.
        panic!(
            "AT-714 fault-injection: {} regression(s) detected (expected) — {}",
            regressions.len(),
            regressions.join("; ")
        );
    }

    #[cfg(not(feature = "bench_regression_fixture_slowdown"))]
    if !regressions.is_empty() {
        panic!(
            "bench_regression: {} regression(s) detected:\n{}",
            regressions.len(),
            regressions.join("\n")
        );
    }
}

// ── Unit tests for regression gate internals ──────────────────────────────────

#[cfg(test)]
mod unit_tests {
    use super::*;

    /// check_regression returns Ok when ratio is within threshold.
    #[test]
    fn check_regression_ok_within_threshold() {
        let outcome = check_regression("test_bench", 110, 100, 15);
        assert_eq!(outcome, RegressionOutcome::Ok, "10% above baseline must be Ok");
    }

    /// check_regression returns Regression when ratio exceeds threshold.
    #[test]
    fn check_regression_fails_above_threshold() {
        let outcome = check_regression("test_bench", 120, 100, 15);
        match outcome {
            RegressionOutcome::Regression { pct } => {
                assert!(pct > 15.0, "pct must be > 15%, got {pct:.2}%");
            }
            other => panic!("expected Regression, got {other:?}"),
        }
    }

    /// check_regression returns PossibleSpeedup when ratio is below threshold.
    #[test]
    fn check_regression_speedup_below_threshold() {
        let outcome = check_regression("test_bench", 80, 100, 15);
        match outcome {
            RegressionOutcome::PossibleSpeedup { pct } => {
                assert!(pct > 15.0, "speedup pct must be > 15%, got {pct:.2}%");
            }
            other => panic!("expected PossibleSpeedup, got {other:?}"),
        }
    }

    /// check_regression handles zero baseline gracefully (returns Ok, not divide-by-zero).
    #[test]
    fn check_regression_zero_baseline_is_skipped() {
        let outcome = check_regression("test_bench", 100, 0, 15);
        assert_eq!(outcome, RegressionOutcome::Ok, "zero baseline must return Ok (skipped)");
    }

    /// WARM_SAMPLES is 11 (AT-713).
    #[test]
    fn warm_samples_is_11() {
        assert_eq!(WARM_SAMPLES, 11, "WARM_SAMPLES must be 11 per AT-713");
    }

    /// REGRESSION_THRESHOLD_PCT is 15 (AT-713).
    #[test]
    fn regression_threshold_is_15() {
        assert_eq!(
            REGRESSION_THRESHOLD_PCT, 15,
            "REGRESSION_THRESHOLD_PCT must be 15 per AT-713"
        );
    }
}
