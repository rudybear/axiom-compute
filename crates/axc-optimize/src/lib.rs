//! `axc-optimize` — AXIOM-Compute autotuner for M2.3 @strategy holes.
//!
//! This crate provides:
//! - [`enumerator`] — Cartesian product enumeration of strategy holes.
//! - [`grid_search`] — Grid-search harness that evaluates all variants and
//!   selects the fastest correct one.
//!
//! # Typical usage
//!
//! ```rust,ignore
//! use axc_optimize::{enumerator::enumerate_strategy, grid_search::grid_search};
//! use axc_optimize::grid_search::{CorrectnessPolicy, SampleStats};
//! use axc_codegen::emit::CodegenOptions;
//!
//! // kernel is an axc_hir::hir::Kernel with Some(strategy).
//! let result = grid_search(
//!     &kernel,
//!     &CorrectnessPolicy::None,
//!     None,
//!     &|spv| Ok(SampleStats { median_ns: 100, min_ns: 90, max_ns: 110, n_samples: 5 }),
//!     &CodegenOptions::default(),
//! ).expect("grid_search failed");
//!
//! println!("winner ordinal: {}", result.winner_ordinal);
//! println!("winner assignments: {:?}", result.winner_assignments);
//! ```

pub mod enumerator;
pub mod grid_search;

// ── Error types ───────────────────────────────────────────────────────────────

/// Errors from the strategy hole enumerator.
#[derive(Debug, thiserror::Error)]
pub enum EnumerateError {
    #[error("strategy map is empty (no holes to enumerate)")]
    EmptyStrategy,
    #[error("unknown strategy hole name `{name}`")]
    UnknownHole { name: String },
    #[error("workgroup dimension `{key}` resolved to non-positive value {value}")]
    NegativeWorkgroupDim { key: String, value: i64 },
}

/// Errors from the grid search harness.
#[derive(Debug, thiserror::Error)]
pub enum GridSearchError {
    #[error("kernel has no @strategy annotation")]
    NoStrategy,
    #[error("strategy enumeration failed: {0}")]
    EnumerateFailed(String),
    #[error("no variant completed benchmarking successfully")]
    NoSuccessfulVariants,
}

#[cfg(test)]
mod tests {
    // Smoke test: confirm modules are reachable.
    #[test]
    fn modules_are_present() {
        // If this compiles, the modules are wired up.
        let _ = crate::enumerator::CARTESIAN_WARN_THRESHOLD;
    }
}
