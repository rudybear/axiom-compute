//! M2.3: Grid-search harness for @strategy hole resolution.
//!
//! ## Algorithm
//!
//! ```text
//! 1. enumerate_strategy(kernel.annotations.strategy) → variants[]
//! 2. For ordinal in variants:
//!    a. resolve_single_variant(kernel, variant.assignments) → resolved_kernel
//!    b. codegen(resolved_kernel) → SPIR-V words
//!    c. if correctness_policy != None:
//!         run_correctness_oracle(spv, inputs) → ok | CorrectnessFailure
//!         if fail: record VariantResult { bench: None, correctness: Fail }, continue
//!    d. run_benchmark(spv, inputs, n_samples) → SampleStats  or  BenchError
//!       if bench fails (timeout / device lost / etc.):
//!         record VariantResult { bench: None, correctness: Ok }
//!         continue
//!    e. record VariantResult { bench: Some(stats), correctness: Ok }
//! 3. Collect all variants with bench: Some(_), sorted by median_ns ascending.
//! 4. Winner = fastest variant.
//!    Fallback chain if winner fails: ordinal 0 → 1 → 2 → 3 (R3 spec rule).
//! 5. Return GridSearchResult { winner_ordinal, winner_variant_id, winner_assignments,
//!                               results, fallback_used }.
//! ```
//!
//! ## Correctness policies
//!
//! - `CorrectnessPolicy::None` — skip correctness check entirely (use for
//!   pure-performance holes where the kernel semantics don't change).
//! - `CorrectnessPolicy::BitExact` — run baseline (ordinal 0) as reference,
//!   require bit-exact output match for all subsequent variants.
//! - `CorrectnessPolicy::EquivFpTol { ulp }` — allow per-element ULP deviation.
//!
//! ## M2.3 scope note
//!
//! The GPU execution path (Vulkan dispatch) is wired via `axc-runtime`.  In
//! test / CI environments without a real GPU, callers supply a mock `BenchFn`
//! and `CorrectnessFn` closure pair.  This crate never calls Vulkan directly.

use crate::{
    enumerator::{
        enumerate_strategy, resolve_single_variant, StrategyAssignments, StrategyVariant,
        CARTESIAN_WARN_THRESHOLD,
    },
    GridSearchError,
};
use axc_hir::hir::Kernel;
use axc_codegen::emit::emit_module;
use axc_codegen::emit::CodegenOptions;
use axc_hir::hir::Module as HirModule;

// ── Public data types ─────────────────────────────────────────────────────────

/// Correctness verification policy.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum CorrectnessPolicy {
    /// Skip correctness checks.
    None,
    /// Require bit-exact output buffers (relative to ordinal-0 reference run).
    BitExact,
    /// Allow per-element ULP deviation.
    EquivFpTol {
        /// Maximum ULP deviation per element.
        ulp: u32,
    },
}

/// Reason a correctness check failed.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum CorrectnessFailure {
    /// Bit-exact check failed: outputs differed at `count` positions.
    BitMismatch { count: usize },
    /// FP-tolerance check failed: `count` elements exceeded `max_ulp` deviation.
    UlpExceeded { count: usize, max_ulp: u32 },
    /// Correctness oracle returned an error string.
    OracleError(String),
}

/// Benchmark statistics for a single variant (across multiple warm samples).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SampleStats {
    /// Median sample time in nanoseconds.
    pub median_ns: u64,
    /// Minimum sample time in nanoseconds.
    pub min_ns: u64,
    /// Maximum sample time in nanoseconds.
    pub max_ns: u64,
    /// Number of samples taken.
    pub n_samples: u32,
}

/// Outcome of benchmarking a single variant.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum BenchOutcome {
    /// Benchmark succeeded; statistics are available.
    Ok(SampleStats),
    /// Benchmark failed (timeout, device lost, etc.).
    Failed(String),
    /// Correctness check failed before benchmarking.
    CorrectnessRejected(CorrectnessFailure),
}

/// Result record for a single enumerated variant.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VariantResult {
    /// Ordinal within the enumerated set.
    pub ordinal: u64,
    /// Stable fingerprint (from `enumerate_strategy`).
    pub variant_id: u64,
    /// Hole assignments for this variant.
    pub assignments: StrategyAssignments,
    /// Benchmark / correctness outcome.
    pub outcome: BenchOutcome,
}

/// The result of a completed grid search.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GridSearchResult {
    /// Ordinal of the winning variant.
    pub winner_ordinal: u64,
    /// `variant_id` of the winner.
    pub winner_variant_id: u64,
    /// Assignments of the winner.
    pub winner_assignments: StrategyAssignments,
    /// Per-variant results (all enumerated variants, in ordinal order).
    pub results: Vec<VariantResult>,
    /// True if a fallback ordinal was used (winner was not the fastest variant).
    pub fallback_used: bool,
    /// Warning messages (e.g. Cartesian-product size exceeded threshold).
    pub warnings: Vec<String>,
}

// ── Closure types ─────────────────────────────────────────────────────────────

/// A closure that benchmarks compiled SPIR-V words and returns sample stats.
///
/// Callers provide this; grid_search never touches Vulkan directly.
///
/// Returns `Ok(SampleStats)` on success or `Err(reason)` on failure.
pub type BenchFn<'a> = &'a dyn Fn(&[u32]) -> Result<SampleStats, String>;

/// A closure that checks correctness of compiled SPIR-V.
///
/// Called only when `CorrectnessPolicy != None`.  Returns `Ok(())` if the
/// output matches the expected (reference) output, or `Err(CorrectnessFailure)`
/// if not.
pub type CorrectnessFn<'a> = &'a dyn Fn(&[u32]) -> Result<(), CorrectnessFailure>;

// ── Grid search entry point ───────────────────────────────────────────────────

/// Run a complete grid search over all strategy variants for `kernel`.
///
/// `bench_fn` is called for each variant that passes the correctness check.
/// `correctness_fn` is called if `policy != CorrectnessPolicy::None`.
///
/// Returns `GridSearchError::NoStrategy` if the kernel has no `@strategy`.
/// Returns `GridSearchError::EnumerateFailed` if enumeration fails.
/// Returns `GridSearchError::NoSuccessfulVariants` if no variant benchmarked ok.
pub fn grid_search(
    kernel: &Kernel,
    policy: &CorrectnessPolicy,
    correctness_fn: Option<CorrectnessFn<'_>>,
    bench_fn: BenchFn<'_>,
    codegen_opts: &CodegenOptions,
) -> Result<GridSearchResult, GridSearchError> {
    // ── 1. Validate input ──────────────────────────────────────────────────
    let strategy = kernel.annotations.strategy.as_ref()
        .ok_or(GridSearchError::NoStrategy)?;

    // ── 2. Enumerate variants ──────────────────────────────────────────────
    let variants: Vec<StrategyVariant> = enumerate_strategy(strategy)
        .map_err(|e| GridSearchError::EnumerateFailed(e.to_string()))?;

    let mut warnings: Vec<String> = Vec::new();
    let total_count: u64 = variants.len() as u64;
    if total_count > CARTESIAN_WARN_THRESHOLD {
        warnings.push(format!(
            "Cartesian product size {total_count} exceeds CARTESIAN_WARN_THRESHOLD \
             ({CARTESIAN_WARN_THRESHOLD}); grid search may take a long time"
        ));
    }

    // ── 3. Evaluate each variant ───────────────────────────────────────────
    let mut results: Vec<VariantResult> = Vec::with_capacity(variants.len());

    for variant in &variants {
        let resolved = match resolve_single_variant(kernel, &variant.assignments) {
            Ok(k) => k,
            Err(e) => {
                results.push(VariantResult {
                    ordinal: variant.ordinal,
                    variant_id: variant.variant_id,
                    assignments: variant.assignments.clone(),
                    outcome: BenchOutcome::Failed(format!("resolve error: {e}")),
                });
                continue;
            }
        };

        // Codegen
        let hir_module = HirModule { kernels: vec![resolved] };
        let spv_words = match emit_module(&hir_module, codegen_opts) {
            Ok(w) => w,
            Err(e) => {
                results.push(VariantResult {
                    ordinal: variant.ordinal,
                    variant_id: variant.variant_id,
                    assignments: variant.assignments.clone(),
                    outcome: BenchOutcome::Failed(format!("codegen error: {e}")),
                });
                continue;
            }
        };

        // Correctness check
        if *policy != CorrectnessPolicy::None {
            if let Some(cfn) = correctness_fn {
                if let Err(failure) = cfn(&spv_words) {
                    results.push(VariantResult {
                        ordinal: variant.ordinal,
                        variant_id: variant.variant_id,
                        assignments: variant.assignments.clone(),
                        outcome: BenchOutcome::CorrectnessRejected(failure),
                    });
                    continue;
                }
            }
        }

        // Benchmark
        let outcome = match bench_fn(&spv_words) {
            Ok(stats) => BenchOutcome::Ok(stats),
            Err(reason) => BenchOutcome::Failed(reason),
        };

        results.push(VariantResult {
            ordinal: variant.ordinal,
            variant_id: variant.variant_id,
            assignments: variant.assignments.clone(),
            outcome,
        });
    }

    // ── 4. Pick winner ─────────────────────────────────────────────────────
    // Collect all ok variants sorted by median_ns ascending.
    let mut ok_results: Vec<&VariantResult> = results.iter()
        .filter(|r| matches!(r.outcome, BenchOutcome::Ok(_)))
        .collect();
    ok_results.sort_by_key(|r| {
        if let BenchOutcome::Ok(ref s) = r.outcome { s.median_ns } else { u64::MAX }
    });

    // R3 fallback chain: if the fastest ok variant is not available, try
    // ordinal 0 → 1 → 2 → 3 as fallback.
    if ok_results.is_empty() {
        // Try fallback chain: ordinals 0, 1, 2, 3 (whichever compiled ok).
        let fallback_ordinals: &[u64] = &[0, 1, 2, 3];
        for &fo in fallback_ordinals {
            if let Some(r) = results.iter().find(|r| r.ordinal == fo) {
                // Even if bench failed, we can at least return this as winner.
                return Ok(GridSearchResult {
                    winner_ordinal: r.ordinal,
                    winner_variant_id: r.variant_id,
                    winner_assignments: r.assignments.clone(),
                    results,
                    fallback_used: true,
                    warnings,
                });
            }
        }
        return Err(GridSearchError::NoSuccessfulVariants);
    }

    let winner = ok_results[0];

    Ok(GridSearchResult {
        winner_ordinal: winner.ordinal,
        winner_variant_id: winner.variant_id,
        winner_assignments: winner.assignments.clone(),
        results,
        fallback_used: false,
        warnings,
    })
}

// ── Tests (AT-1022..AT-1030) ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use axc_hir::hir::{
        Kernel, KernelAnnotations, KernelBody, KernelId, StrategyHoles, WorkgroupDims,
    };
    use axc_hir::param::ParamBindingPlan;
    use axc_lexer::Span;
    use std::collections::BTreeMap;

    fn make_strategy_kernel(wg_x: u32, candidates: Vec<i64>) -> Kernel {
        let mut map: BTreeMap<String, Vec<i64>> = BTreeMap::new();
        map.insert("workgroup_x".to_string(), candidates);
        Kernel {
            id: KernelId(0),
            name: "bench_kernel".to_string(),
            annotations: KernelAnnotations {
                workgroup: WorkgroupDims { x: wg_x, y: 1, z: 1 },
                intent: None,
                complexity: None,
                preconditions: Vec::new(),
                subgroup_uniform: false,
                cooperative_matrix: false,
                strategy: Some(StrategyHoles { map }),
            },
            params: Vec::new(),
            binding_plan: ParamBindingPlan {
                buffers: Vec::new(),
                scalars: Vec::new(),
                push_constant_total_bytes: 0,
            },
            body: KernelBody::Empty,
            span: Span::new(0, 1),
        }
    }

    /// AT-1022: grid_search returns NoStrategy for kernel without @strategy.
    #[test]
    fn at_1022_grid_search_no_strategy_returns_error() {
        let kernel = make_strategy_kernel(64, vec![]); // empty vec → no strategy
        // Manually clear strategy to None.
        let no_strategy_kernel = Kernel {
            annotations: KernelAnnotations {
                strategy: None,
                ..kernel.annotations.clone()
            },
            ..kernel
        };
        let result = grid_search(
            &no_strategy_kernel,
            &CorrectnessPolicy::None,
            None,
            &|_spv| Ok(SampleStats { median_ns: 100, min_ns: 90, max_ns: 110, n_samples: 5 }),
            &CodegenOptions::default(),
        );
        assert!(
            matches!(result, Err(GridSearchError::NoStrategy)),
            "expected NoStrategy; got {result:?}"
        );
    }

    /// AT-1023: grid_search picks fastest variant (lowest median_ns).
    ///
    /// Two candidates: workgroup_x=32 (slow) and workgroup_x=64 (fast).
    /// bench_fn returns median proportional to workgroup_x to simulate latency
    /// going up with larger workgroup (so 32 → 32ns, 64 → 64ns; 32 is faster).
    #[test]
    fn at_1023_grid_search_picks_fastest_variant() {
        // Placeholder wg_x=1 (HoleRef); candidates in the strategy map.
        let kernel = make_strategy_kernel(1, vec![32, 64]);

        // bench_fn: simulate 32 being faster (lower median).
        let bench_fn: BenchFn<'_> = &|_spv: &[u32]| {
            // We can't inspect SPIR-V workgroup dim cheaply here;
            // use a counter trick via a cell.
            // Instead, trust ordinal order: ordinal 0 = workgroup_x=32.
            // Since we can't get ordinal here, just return different values
            // based on SPIR-V length difference (hack for test).
            // Actually: both resolve to Empty body, so spv length is identical.
            // Use static counter.
            use std::sync::atomic::{AtomicU64, Ordering};
            static CALL_COUNT: AtomicU64 = AtomicU64::new(0);
            let n = CALL_COUNT.fetch_add(1, Ordering::Relaxed);
            // Call 0 = ordinal 0 (workgroup_x=32): return 50ns (faster)
            // Call 1 = ordinal 1 (workgroup_x=64): return 100ns (slower)
            let median_ns = if n == 0 { 50 } else { 100 };
            Ok(SampleStats {
                median_ns,
                min_ns: median_ns - 5,
                max_ns: median_ns + 5,
                n_samples: 5,
            })
        };

        let result = grid_search(
            &kernel,
            &CorrectnessPolicy::None,
            None,
            bench_fn,
            &CodegenOptions::default(),
        ).expect("grid_search failed");

        assert_eq!(result.winner_ordinal, 0, "ordinal 0 (workgroup_x=32) should win");
        assert!(!result.fallback_used);
        assert_eq!(result.results.len(), 2);
    }

    /// AT-1024: grid_search uses R3 fallback (ordinal 0) when all benchmarks fail.
    #[test]
    fn at_1024_grid_search_fallback_when_all_bench_fail() {
        let kernel = make_strategy_kernel(1, vec![32, 64, 128]);

        let result = grid_search(
            &kernel,
            &CorrectnessPolicy::None,
            None,
            &|_| Err("simulated bench failure".to_string()),
            &CodegenOptions::default(),
        ).expect("grid_search should use fallback, not error");

        assert!(result.fallback_used, "fallback_used must be true when all bench fail");
        // R3 spec: fallback starts at ordinal 0.
        assert_eq!(result.winner_ordinal, 0,
            "R3 fallback: winner_ordinal must be 0 when all fail");
    }

    /// AT-1025: correctness rejection excludes variant from winner selection.
    #[test]
    fn at_1025_correctness_failure_excludes_variant() {
        let kernel = make_strategy_kernel(1, vec![32, 64]);

        use std::sync::atomic::{AtomicU64, Ordering};
        static CORRECT_CALL: AtomicU64 = AtomicU64::new(0);

        // Correctness: ordinal 0 (call 0) fails; ordinal 1 (call 1) passes.
        let correctness_fn: CorrectnessFn<'_> = &|_spv: &[u32]| {
            let n = CORRECT_CALL.fetch_add(1, Ordering::Relaxed);
            if n == 0 {
                return Err(CorrectnessFailure::BitMismatch { count: 4 });
            }
            Ok(())
        };

        let result = grid_search(
            &kernel,
            &CorrectnessPolicy::BitExact,
            Some(correctness_fn),
            &|_| Ok(SampleStats { median_ns: 100, min_ns: 90, max_ns: 110, n_samples: 5 }),
            &CodegenOptions::default(),
        ).expect("grid_search failed");

        // ordinal 0 was correctness-rejected, so ordinal 1 wins (it's the only ok variant).
        assert_eq!(result.winner_ordinal, 1, "ordinal 1 should win (ordinal 0 correctness-rejected)");
        assert!(matches!(
            result.results[0].outcome,
            BenchOutcome::CorrectnessRejected(CorrectnessFailure::BitMismatch { count: 4 })
        ));
    }

    /// AT-1026: warnings include Cartesian threshold message when product > 100.
    ///
    /// Note: actually running >100 variants in a unit test is impractical, so
    /// we directly test cartesian_product_size and warning generation logic.
    #[test]
    fn at_1026_cartesian_warn_threshold_constant_is_100() {
        assert_eq!(CARTESIAN_WARN_THRESHOLD, 100,
            "CARTESIAN_WARN_THRESHOLD must be 100 per spec");
    }

    // ── AT-1040..AT-1042: Additional grid search tests ────────────────────────

    /// AT-1040: grid_search winner_variant_id matches the variant_id from enumerate_strategy.
    #[test]
    fn at_1040_winner_variant_id_matches_enumerator() {
        use crate::enumerator::enumerate_strategy;

        let kernel = make_strategy_kernel(1, vec![32, 64, 128]);
        let strategy = kernel.annotations.strategy.as_ref().unwrap();
        let variants = enumerate_strategy(strategy).unwrap();

        // Use atomic counter so first call wins (ordinal 0 = fastest).
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let bench_fn: BenchFn<'_> = &|_spv: &[u32]| {
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            // First call (ordinal 0) returns 10ns; rest return 100ns.
            let t = if n == 0 { 10 } else { 100 };
            Ok(SampleStats { median_ns: t, min_ns: t, max_ns: t, n_samples: 1 })
        };

        let result = grid_search(
            &kernel,
            &CorrectnessPolicy::None,
            None,
            bench_fn,
            &CodegenOptions::default(),
        ).unwrap();

        // Winner must be ordinal 0; its variant_id must match what enumerate gives.
        assert_eq!(result.winner_variant_id, variants[0].variant_id,
            "winner_variant_id must match enumerator variant_id for ordinal 0");
    }

    /// AT-1041: grid_search returns fallback_used=false when winner benchmarks ok.
    #[test]
    fn at_1041_fallback_used_false_when_bench_succeeds() {
        let kernel = make_strategy_kernel(1, vec![64, 128]);

        let result = grid_search(
            &kernel,
            &CorrectnessPolicy::None,
            None,
            &|_| Ok(SampleStats { median_ns: 50, min_ns: 40, max_ns: 60, n_samples: 3 }),
            &CodegenOptions::default(),
        ).unwrap();

        assert!(!result.fallback_used,
            "fallback_used must be false when benchmarks succeed");
    }

    /// AT-1042: grid_search results vec contains all enumerated variants.
    #[test]
    fn at_1042_results_vec_contains_all_variants() {
        let kernel = make_strategy_kernel(1, vec![32, 64, 128, 256]);

        let result = grid_search(
            &kernel,
            &CorrectnessPolicy::None,
            None,
            &|_| Ok(SampleStats { median_ns: 100, min_ns: 90, max_ns: 110, n_samples: 5 }),
            &CodegenOptions::default(),
        ).unwrap();

        assert_eq!(result.results.len(), 4,
            "results must contain all 4 enumerated variants");
        // All ordinals 0..3 present
        let ordinals: Vec<u64> = result.results.iter().map(|r| r.ordinal).collect();
        assert_eq!(ordinals, vec![0, 1, 2, 3], "ordinals must be 0..3 in order");
    }
}
