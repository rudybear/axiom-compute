//! M2.3: `axc optimize` subcommand implementation.
//!
//! Reads an .axc source, extracts @strategy holes, runs grid search using a
//! no-op mock bench (GPU execution arrives in M2.4), and writes the winning
//! SPIR-V binary and a JSON strategy result sidecar.

use std::collections::BTreeMap;
use std::path::Path;

use axc_optimize::grid_search::{
    grid_search, CorrectnessPolicy, GridSearchResult, SampleStats,
};
use axc_codegen::emit::CodegenOptions;
use axc_hir::lower_module;
use axc_parser::parse;

use crate::DriverError;

/// Errors specific to the optimize subcommand.
#[derive(Debug, thiserror::Error)]
pub enum OptimizeError {
    #[error("compilation failed: {0}")]
    Compile(#[from] DriverError),
    #[error("no @strategy annotation found in kernel")]
    NoStrategy,
    #[error("grid search failed: {0}")]
    GridSearch(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json serialization failed: {0}")]
    Json(#[from] serde_json::Error),
}

/// Run grid search on the source file, write winning SPIR-V to `output`,
/// and write a JSON sidecar to `output.axc.strategy.json`.
///
/// In M2.3, the bench closure is a no-op mock that returns a fixed latency.
/// Real GPU benchmarking via Vulkan dispatch is wired in M2.4.
///
/// `correctness_str`: one of `"none"`, `"bit-exact"`, `"fp-tol:<ulp>"`.
pub fn run_optimize(
    input: &Path,
    output: &Path,
    correctness_str: &str,
) -> Result<(), OptimizeError> {
    let source: String = std::fs::read_to_string(input).map_err(OptimizeError::Io)?;

    // Parse the correctness policy from the string flag.
    let policy: CorrectnessPolicy = parse_correctness_policy(correctness_str);

    // Run the full lexer/parser/HIR pipeline to get a kernel with @strategy.
    let (ast, lex_errs, parse_errs) = parse(&source);
    if !lex_errs.is_empty() || !parse_errs.is_empty() {
        return Err(OptimizeError::Compile(DriverError::Compile {
            lex: lex_errs,
            parse: parse_errs,
            hir: Vec::new(),
        }));
    }
    let (hir, hir_errs, _warns) = lower_module(&ast);
    if !hir_errs.is_empty() {
        return Err(OptimizeError::Compile(DriverError::Compile {
            lex: Vec::new(),
            parse: Vec::new(),
            hir: hir_errs,
        }));
    }

    let kernel = hir.kernels.first().ok_or(OptimizeError::NoStrategy)?;

    if kernel.annotations.strategy.is_none() {
        return Err(OptimizeError::NoStrategy);
    }

    // M2.3: mock bench — returns a fixed 1000ns for every variant.
    // Real GPU dispatch is wired in M2.4 via axc-runtime Vulkan path.
    let bench_fn = &|_spv: &[u32]| -> Result<SampleStats, String> {
        Ok(SampleStats {
            median_ns: 1000,
            min_ns: 990,
            max_ns: 1010,
            n_samples: 1,
        })
    };

    let result: GridSearchResult = grid_search(
        kernel,
        &policy,
        None,
        bench_fn,
        &CodegenOptions::default(),
    ).map_err(|e| OptimizeError::GridSearch(e.to_string()))?;

    // Build winner assignments and compile the winning variant.
    let mut winner_assignments: BTreeMap<String, i64> = BTreeMap::new();
    for (name, value) in &result.winner_assignments.values {
        winner_assignments.insert(name.clone(), *value);
    }

    let (spv_bytes, _meta) = crate::compile_source_with_assignments(&source, &winner_assignments)
        .map_err(OptimizeError::Compile)?;

    // Write the winning SPIR-V binary.
    std::fs::write(output, &spv_bytes).map_err(OptimizeError::Io)?;

    // Write the JSON strategy sidecar.
    let sidecar_path = strategy_sidecar_path(output);
    let sidecar_json: String = serde_json::to_string_pretty(&result)?;
    std::fs::write(&sidecar_path, sidecar_json.as_bytes()).map_err(OptimizeError::Io)?;

    Ok(())
}

/// Compute the strategy sidecar path from an output path.
///
/// Appends `.axc.strategy.json` to the full filename.
/// Example: `out.spv` → `out.spv.axc.strategy.json`
pub fn strategy_sidecar_path(output: &Path) -> std::path::PathBuf {
    let filename: String = output
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".to_owned());
    let sidecar_name: String = format!("{filename}.axc.strategy.json");
    output
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(sidecar_name)
}

/// Parse `--correctness` flag value into a `CorrectnessPolicy`.
///
/// Accepts: `"none"`, `"bit-exact"`, `"fp-tol:<ulp>"` (e.g. `"fp-tol:4"`).
/// Unrecognised strings default to `CorrectnessPolicy::None`.
pub(crate) fn parse_correctness_policy(s: &str) -> CorrectnessPolicy {
    match s {
        "none" => CorrectnessPolicy::None,
        "bit-exact" => CorrectnessPolicy::BitExact,
        s if s.starts_with("fp-tol:") => {
            let ulp_str: &str = &s["fp-tol:".len()..];
            let ulp: u32 = ulp_str.parse().unwrap_or(4);
            CorrectnessPolicy::EquivFpTol { ulp }
        }
        _ => CorrectnessPolicy::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// AT-1027: parse_correctness_policy parses "none" correctly.
    #[test]
    fn at_1027_parse_correctness_none() {
        assert_eq!(parse_correctness_policy("none"), CorrectnessPolicy::None);
    }

    /// AT-1028: parse_correctness_policy parses "bit-exact" correctly.
    #[test]
    fn at_1028_parse_correctness_bit_exact() {
        assert_eq!(parse_correctness_policy("bit-exact"), CorrectnessPolicy::BitExact);
    }

    /// AT-1029: parse_correctness_policy parses "fp-tol:8" correctly.
    #[test]
    fn at_1029_parse_correctness_fp_tol() {
        assert_eq!(
            parse_correctness_policy("fp-tol:8"),
            CorrectnessPolicy::EquivFpTol { ulp: 8 }
        );
    }

    /// AT-1030: strategy_sidecar_path appends .axc.strategy.json.
    #[test]
    fn at_1030_strategy_sidecar_path() {
        let p = strategy_sidecar_path(std::path::Path::new("out.spv"));
        assert_eq!(p, std::path::PathBuf::from("out.spv.axc.strategy.json"));
    }
}
