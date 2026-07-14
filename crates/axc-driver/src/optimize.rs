//! M2.3: `axc optimize` subcommand implementation.
//!
//! Reads an .axc source, extracts @strategy holes, runs grid search using a
//! no-op mock bench (GPU execution arrives in M2.4), and writes the winning
//! SPIR-V binary and a JSON strategy result sidecar.
//!
//! M3.21 (FG.9) adds module-global hole discovery (F3 — `union_module_holes`,
//! so a hole declared in a SIBLING kernel's `@strategy` block is available for
//! enumeration even though the benched `--kernel` doesn't declare it itself)
//! and the F4 winner cross-validator: before ANY output is written, the
//! winning assignment is cross-compiled across EVERY kernel in the module via
//! `compile_module_all` — a candidate that is valid for the benched kernel but
//! breaks a non-benched sibling (e.g. overflows its shared-memory budget)
//! fails closed here, naming the broken sibling where determinable (r3
//! orchestrator patch: this check runs before EITHER the `.spv` or the
//! strategy sidecar is written — a sibling failure leaves ZERO artifacts).

use std::collections::BTreeMap;
use std::path::Path;

use axc_optimize::grid_search::{
    grid_search, CorrectnessPolicy, GridSearchResult, SampleStats,
};
use axc_codegen::emit::CodegenOptions;
use axc_hir::hir::{Kernel, Module as HirModule};
use axc_hir::lower_module;
use axc_optimize::enumerator::union_module_holes;
use axc_parser::parse;

use crate::DriverError;

/// Errors specific to the optimize subcommand.
#[derive(Debug, thiserror::Error)]
pub enum OptimizeError {
    #[error("compilation failed: {0}")]
    Compile(#[from] DriverError),
    #[error("no @strategy annotation found in kernel")]
    NoStrategy,
    #[error("strategy hole discovery failed: {0}")]
    Enumerate(#[from] axc_optimize::EnumerateError),
    #[error("grid search failed: {0}")]
    GridSearch(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json serialization failed: {0}")]
    Json(#[from] serde_json::Error),
    /// M3.21 (FG.9) F4: the grid-search winner compiles/validates for the
    /// benched `--kernel` but fails module-wide — cross-compiling it across
    /// ALL kernels (via `compile_module_all`) surfaced a failure. NO output
    /// (`.spv` or strategy sidecar) was written (r3 patch).
    #[error(
        "grid-search winner is invalid module-wide{}: {detail}",
        broken_kernel.as_ref().map(|k| format!(" (sibling kernel `{k}` failed)")).unwrap_or_default()
    )]
    WinnerInvalidModuleWide {
        broken_kernel: Option<String>,
        detail: String,
    },
}

/// Run grid search on the source file, write winning SPIR-V to `output`,
/// and write a JSON sidecar to `output.axc.strategy.json`.
///
/// In M2.3, the bench closure is a no-op mock that returns a fixed latency.
/// Real GPU benchmarking via Vulkan dispatch is wired in M2.4.
///
/// `correctness_str`: one of `"none"`, `"bit-exact"`, `"fp-tol:<ulp>"`.
///
/// `kernel`: M3.21 (FG.9) — the kernel grid-search dispatches+times. `None`
/// on an ambiguous (2+ kernel) source fails closed (`DriverError::AmbiguousKernel`
/// via `OptimizeError::Compile`). Ignored on a single-kernel source. The
/// shared hole still binds MODULE-WIDE regardless of which kernel is benched
/// (F3 — `union_module_holes`); F4 re-validates the winner across every
/// kernel before any output is written.
pub fn run_optimize(
    input: &Path,
    output: &Path,
    correctness_str: &str,
    kernel: Option<&str>,
) -> Result<(), OptimizeError> {
    let source: String = std::fs::read_to_string(input).map_err(OptimizeError::Io)?;

    // Parse the correctness policy from the string flag.
    let policy: CorrectnessPolicy = parse_correctness_policy(correctness_str);

    // Run the full lexer/parser/HIR pipeline to get every kernel + its holes.
    let (ast, lex_errs, parse_errs) = parse(&source);
    if !lex_errs.is_empty() || !parse_errs.is_empty() {
        return Err(OptimizeError::Compile(DriverError::Compile {
            lex: lex_errs,
            parse: parse_errs,
            hir: Vec::new(),
        }));
    }
    let (hir, hir_errs, _warns): (HirModule, _, _) = lower_module(&ast);
    if !hir_errs.is_empty() {
        return Err(OptimizeError::Compile(DriverError::Compile {
            lex: Vec::new(),
            parse: Vec::new(),
            hir: hir_errs,
        }));
    }

    // F3: module-global hole discovery — union across ALL kernels' @strategy
    // blocks (conflict-checked + prefix-collision-checked), NOT just the
    // benched kernel's own block.
    let union_holes = union_module_holes(&hir)?;
    if union_holes.map.is_empty() {
        return Err(OptimizeError::NoStrategy);
    }

    // Fail-closed bench-target selection (never a silent kernels[0] pick).
    let selected: &Kernel = crate::select_kernel(&hir.kernels, kernel)
        .map_err(OptimizeError::Compile)?;

    // The benched kernel is codegen'd+timed with its OWN annotations, except
    // `strategy` is swapped for the module-global UNION so a hole declared
    // only in a sibling kernel still enumerates here (the sharing mechanism).
    let bench_kernel: Kernel = Kernel {
        annotations: axc_hir::hir::KernelAnnotations {
            strategy: Some(union_holes),
            ..selected.annotations.clone()
        },
        ..selected.clone()
    };

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
        &bench_kernel,
        &policy,
        None,
        bench_fn,
        &CodegenOptions::default(),
    ).map_err(|e| OptimizeError::GridSearch(e.to_string()))?;

    // Build winner assignments.
    let mut winner_assignments: BTreeMap<String, i64> = BTreeMap::new();
    for (name, value) in &result.winner_assignments.values {
        winner_assignments.insert(name.clone(), *value);
    }

    // F4 / r3: cross-validate the winner across EVERY kernel BEFORE writing
    // ANY output (neither the `.spv` nor the strategy sidecar exists if this
    // fails — AT-2970).
    let resolved_source: String = crate::substitute_strategy_holes(&source, &winner_assignments);
    cross_validate_winner_module_wide(&resolved_source)?;

    // Compile the winning variant for the benched kernel (the artifact `axc
    // optimize` actually writes to `--output`).
    let (spv_bytes, _meta) = crate::compile_source_with_assignments_kernel(
        &source, &winner_assignments, Some(selected.name.as_str()),
    ).map_err(OptimizeError::Compile)?;

    // Write the winning SPIR-V binary.
    std::fs::write(output, &spv_bytes).map_err(OptimizeError::Io)?;

    // Write the JSON strategy sidecar.
    let sidecar_path = strategy_sidecar_path(output);
    let sidecar_json: String = serde_json::to_string_pretty(&result)?;
    std::fs::write(&sidecar_path, sidecar_json.as_bytes()).map_err(OptimizeError::Io)?;

    Ok(())
}

/// M3.21 (FG.9) F4: cross-compile `resolved_source` (already hole-substituted
/// + `@strategy`-stripped) across EVERY kernel via `compile_module_all`.
///
/// On failure, re-lowers `resolved_source` (fresh — the ORIGINAL unsubstituted
/// `hir` built at the top of `run_optimize` has stale byte offsets once holes
/// are substituted, since `?WG` -> `8` changes the source's byte length) to
/// attribute the failure to a specific kernel where determinable, satisfying
/// AT-2970's "surfacing which sibling broke".
fn cross_validate_winner_module_wide(resolved_source: &str) -> Result<(), OptimizeError> {
    let Err(e) = crate::compile_module_all(resolved_source, false) else {
        return Ok(());
    };
    let broken_kernel: Option<String> = {
        let (ast, _lex, _parse) = axc_parser::parse(resolved_source);
        let (hir, _hir_errs, _warns) = lower_module(&ast);
        name_broken_kernel(&e, &hir)
    };
    Err(OptimizeError::WinnerInvalidModuleWide {
        broken_kernel,
        detail: e.to_string(),
    })
}

/// Best-effort "which kernel owns this failure" attribution (r3's "surfacing
/// which sibling broke").
///
/// Two cases, by where `compile_module_all` actually failed:
///
/// - A `DriverError::Codegen` failure happens INSIDE `compile_module_all`'s
///   own per-kernel emit loop, so `axc_codegen::CodegenError::UnresolvedStrategyHole`
///   already carries the exact `kernel_name` — used directly, no guessing.
/// - A `DriverError::Compile { hir, .. }` failure is a WHOLE-MODULE HIR
///   validation error (HIR lowers every kernel in one pass, so a bad
///   `@workgroup` on kernel N surfaces before any per-kernel emit even
///   starts) — attributed via the error's `miette::Diagnostic` span: every
///   `HirError` variant carries a `#[label] span` pointing at the offending
///   annotation/expression, which — because annotations always precede the
///   `fn name(...)` they modify — falls between the PREVIOUS kernel's name
///   token and the OWNING kernel's name token. So: find the kernel whose
///   name-span START is the smallest one still `>=` the error span's start
///   (an error inside a kernel's own annotations attributes forward to that
///   kernel); fall back to the LAST kernel by name-span start (a body-level
///   error, which lands strictly after its own kernel's name) when no such
///   kernel exists.
fn name_broken_kernel(e: &DriverError, hir: &HirModule) -> Option<String> {
    use miette::Diagnostic;

    if let DriverError::Codegen(axc_codegen::CodegenError::UnresolvedStrategyHole { kernel_name, .. }) = e {
        return Some(kernel_name.clone());
    }

    let DriverError::Compile { hir: hir_errors, .. } = e else {
        return None;
    };
    let error_offset: u32 = hir_errors.first()?.labels()?.next()?.offset() as u32;

    let mut by_start: Vec<&Kernel> = hir.kernels.iter().collect();
    by_start.sort_by_key(|k| k.span.start);

    by_start.iter()
        .find(|k| k.span.start >= error_offset)
        .or_else(|| by_start.last())
        .map(|k| k.name.clone())
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
