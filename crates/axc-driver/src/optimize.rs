//! M2.3: `axc optimize` subcommand implementation.
//!
//! Reads an .axc source, extracts @strategy holes, runs grid search, and
//! writes the winning SPIR-V binary and a JSON strategy result sidecar.
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
//!
//! M3.23 replaces the M2.3 mock bench (fixed 1000ns for every variant, never
//! dispatches, never checks correctness) with a REAL bench: the resident
//! `GpuTimestamp` MIN-of-N harness (`prepare_kernel` / `upload_resident` /
//! `dispatch_resident`, same as every `resident_*.rs` TFLOPS bench) plus a
//! variant-vs-variant differential DETECTOR layered under `bench_variant`'s
//! device-keyed CPU oracle ATTRIBUTOR (§1.5.1 of the M3.23 spec). A caller
//! without a GPU gets `OptimizeError::GpuUnavailable` — never a fake winner.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::path::Path;

use axc_optimize::grid_search::{
    grid_search, BenchFn, CorrectnessFailure, CorrectnessFn, CorrectnessPolicy, GridSearchResult,
    SampleStats,
};
use axc_codegen::emit::CodegenOptions;
use axc_hir::hir::{Kernel, Module as HirModule};
use axc_hir::lower_module;
use axc_optimize::enumerator::union_module_holes;
use axc_parser::parse;
use axc_runtime::{ResidentBenchConfig, ResidentTimingSource, VulkanContext};

use crate::mcp::tools::bench_variant::{self, CorrectnessStatus};
use crate::DriverError;

/// Default per-buffer size (bytes) used to seed inputs / size outputs for the
/// real bench, matching the MCP `grid_search` default (`mcp/tools/grid_search.rs`).
const DEFAULT_BENCH_BYTES: usize = 4096;
/// Fixed seed so every variant sees IDENTICAL input bytes (fair timing +
/// comparable outputs for the differential/oracle).
const SEED: u64 = 0;
/// Discarded warmup dispatches before the measured loop (TIMING-1 / HANDOFF-12).
const N_WARMUP: usize = 2;
/// Measured dispatches per variant; the reported stat is MIN-of-N.
const N_MEASURED: usize = 10;

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
    /// M3.23: no Vulkan device is available to bench variants for real. NEVER
    /// falls back to a mock/fake winner — fails closed BEFORE any artifact is
    /// written (checked after the cheap CPU-only diagnostics, before
    /// `grid_search` — §1.3.1 of the spec).
    #[error("GPU unavailable: `axc optimize` requires a Vulkan device to bench variants \
             (no fake/mock winner is emitted); {0}")]
    GpuUnavailable(String),
    /// M3.23 (§1.5.1, r3): the variant-vs-variant differential DETECTED that
    /// two or more variants produced different output on identical seeded
    /// input, and no registered CPU oracle could ATTRIBUTE which side is
    /// correct. Refuses to crown anyone — ZERO artifacts (`.spv`/sidecar).
    /// Direction-independent: fires identically whether the garbage variant
    /// seeded the differential's reference or diverged from it.
    #[error("variants diverged with no ground-truth oracle to attribute correctness: \
             ordinals {ordinals:?} produced different output on identical seeded input; \
             refusing to crown a winner (register a CPU reference for kernel `{kernel}` to disambiguate)")]
    VariantsDiverge { ordinals: Vec<usize>, kernel: String },
}

/// Run grid search on the source file, write winning SPIR-V to `output`,
/// and write a JSON sidecar to `output.axc.strategy.json`.
///
/// M3.23: benches with the REAL resident `GpuTimestamp` MIN-of-N harness (via
/// `vk`) and checks correctness with a variant-vs-variant differential
/// DETECTOR (§1.5.1) layered under `bench_variant`'s device-keyed CPU oracle
/// ATTRIBUTOR. `vk == None` fails closed with `OptimizeError::GpuUnavailable`
/// — never a fake/mock winner (checked AFTER the cheap CPU-only diagnostics,
/// BEFORE `grid_search` — §1.3.1, so a headless caller still gets `NoStrategy`
/// / `AmbiguousKernel` where applicable).
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
    vk: Option<&VulkanContext>,
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

    // §1.3.1 (PINNED ordering): the GPU-less fail-closed check runs AFTER the
    // cheap CPU-only diagnostics above (parse / NoStrategy / AmbiguousKernel)
    // and BEFORE the real bench below — a headless caller still gets the
    // useful diagnostic on a malformed request, and only pays GpuUnavailable
    // once a real bench is genuinely unavoidable.
    let vk: &VulkanContext = vk.ok_or_else(|| {
        OptimizeError::GpuUnavailable(
            "pass a Vulkan device (probe_vulkan_available() + VulkanContext::new(), \
             or run under Lavapipe via VK_DRIVER_FILES in CI)".to_string(),
        )
    })?;

    // ── Invariant metadata (§1.4): shared/identical across every variant — a
    // @strategy hole is an integer knob on shared/tile sizes, never the buffer
    // signature or push-constant layout, so this is captured ONCE.
    let entry: String = bench_kernel.name.clone();
    let plan = &bench_kernel.binding_plan;
    let pc_bytes: u32 = plan.push_constant_total_bytes;
    let workgroup_size: [u32; 3] = [
        selected.annotations.workgroup.x,
        selected.annotations.workgroup.y,
        selected.annotations.workgroup.z,
    ];
    let input_sizes: Vec<usize> = vec![DEFAULT_BENCH_BYTES; plan.buffers.len()];
    let input_data: Vec<Vec<u8>> = bench_variant::seeded_inputs(&input_sizes, SEED);
    let input_refs: Vec<&[u8]> = input_data.iter().map(Vec::as_slice).collect();
    let output_sizes_u64: Vec<u64> = input_sizes.iter().map(|&s| s as u64).collect();
    let workgroups: [u32; 3] = bench_variant::derive_workgroups(&input_sizes, workgroup_size, None);
    let push_constants: Vec<u8> = vec![0_u8; pc_bytes as usize];
    let device_name: String = vk.physical_device_name().to_string();
    let ulp_tol: u32 = bench_variant::ulp_tolerance_for_device(&device_name);

    // ── The differential's bookkeeping cells (§1.5, §1.5.1) — read AFTER
    // `grid_search` returns (the closures' borrows end at their last use,
    // inside `grid_search`).
    let ordinal: Cell<usize> = Cell::new(0);
    let reference_ord: Cell<usize> = Cell::new(0);
    let reference_outputs: RefCell<Option<Vec<Vec<u8>>>> = RefCell::new(None);
    let divergence: RefCell<Option<Vec<usize>>> = RefCell::new(None);
    let oracle_attributed: Cell<bool> = Cell::new(false);
    let saw_not_checked: Cell<bool> = Cell::new(false);
    let last_timing_source: Cell<ResidentTimingSource> = Cell::new(ResidentTimingSource::CpuFenceWall);

    // (1) The variant-vs-variant differential DETECTOR + (2) the registered
    // CPU oracle ATTRIBUTOR (§1.5, §1.5.1) — TWO faces run per variant here.
    let correctness_fn: CorrectnessFn<'_> = &|words: &[u32]| -> Result<(), CorrectnessFailure> {
        let ord: usize = ordinal.get();
        ordinal.set(ord + 1);
        let handle = vk.prepare_kernel(words, plan, pc_bytes, &entry)
            .map_err(|e| CorrectnessFailure::OracleError(e.to_string()))?;
        let resident = vk.upload_resident(&handle, &input_refs, &output_sizes_u64)
            .map_err(|e| CorrectnessFailure::OracleError(e.to_string()))?;
        let cfg = ResidentBenchConfig {
            workgroups: (workgroups[0], workgroups[1], workgroups[2]),
            push_constants: push_constants.clone(),
        };
        let _ = vk.dispatch_resident(&handle, &resident, &cfg)
            .map_err(|e| CorrectnessFailure::OracleError(e.to_string()))?;
        let outputs = vk.readback_resident(&handle, &resident, &output_sizes_u64)
            .map_err(|e| CorrectnessFailure::OracleError(e.to_string()))?;

        // (1) DETECT: compare against the first-seen reference (NOT ground
        // truth — a mismatch is RECORDED, never rejected on its own word).
        if let Some(reference) = reference_outputs.borrow().as_ref() {
            if bench_variant::differential_mismatch(reference, &outputs, ulp_tol).is_some() {
                divergence.borrow_mut().get_or_insert_with(Vec::new)
                    .extend_from_slice(&[reference_ord.get(), ord]);
            }
        }

        // (2) ATTRIBUTE: the registered CPU oracle, when one exists.
        let oracle = bench_variant::check_correctness(
            &entry, plan, &input_data, Some(&outputs), &BTreeMap::new(), &device_name,
        );
        match oracle {
            CorrectnessStatus::Ok => oracle_attributed.set(true),
            CorrectnessStatus::Failed { reason } => {
                oracle_attributed.set(true);
                return Err(CorrectnessFailure::OracleError(reason));
            }
            CorrectnessStatus::NotChecked { .. } => saw_not_checked.set(true),
        }

        // Seed the reference from the FIRST variant that reaches here.
        if reference_outputs.borrow().is_none() {
            *reference_outputs.borrow_mut() = Some(outputs);
            reference_ord.set(ord);
        }
        Ok(())
    };

    // The real resident GpuTimestamp MIN-of-N bench (§1.4) — same harness
    // every `resident_*.rs` TFLOPS bench uses.
    let bench_fn: BenchFn<'_> = &|words: &[u32]| -> Result<SampleStats, String> {
        let handle = vk.prepare_kernel(words, plan, pc_bytes, &entry).map_err(|e| e.to_string())?;
        let resident = vk.upload_resident(&handle, &input_refs, &output_sizes_u64).map_err(|e| e.to_string())?;
        let cfg = ResidentBenchConfig {
            workgroups: (workgroups[0], workgroups[1], workgroups[2]),
            push_constants: push_constants.clone(),
        };
        for _ in 0..N_WARMUP {
            vk.dispatch_resident(&handle, &resident, &cfg).map_err(|e| e.to_string())?;
        }
        let (mut min_ns, mut max_ns): (u64, u64) = (u64::MAX, 0);
        for _ in 0..N_MEASURED {
            let t = vk.dispatch_resident(&handle, &resident, &cfg).map_err(|e| e.to_string())?;
            min_ns = min_ns.min(t.kernel_ns);
            max_ns = max_ns.max(t.kernel_ns);
            last_timing_source.set(t.timing_source);
        }
        Ok(SampleStats { median_ns: min_ns, min_ns, max_ns, n_samples: N_MEASURED as u32 })
    };

    let mut result: GridSearchResult = grid_search(
        &bench_kernel,
        &policy,
        Some(correctness_fn),
        bench_fn,
        &CodegenOptions::default(),
    ).map_err(|e| OptimizeError::GridSearch(e.to_string()))?;

    // Post-grid_search decision (§1.5.1, §4): unattributable divergence fails
    // closed with ZERO artifacts — attributed divergence proceeds (the oracle
    // already rejected the wrong variant(s)); no divergence proceeds normally.
    if let Some(ordinals) = divergence.into_inner() {
        if !oracle_attributed.get() {
            return Err(OptimizeError::VariantsDiverge { ordinals, kernel: selected.name.clone() });
        }
    }
    if saw_not_checked.get() {
        result.warnings.push(format!(
            "correctness policy requested but no CPU ground-truth oracle is registered for \
             kernel `{}`; all benched variants AGREED with each other via the variant-vs-variant \
             differential (within {ulp_tol} ULP) on identical seeded input, but the shared answer \
             was NOT confirmed against an absolute reference (NotChecked)",
            selected.name,
        ));
    }
    eprintln!(
        "axc optimize: real-bench timing_source = {:?} (device = {device_name:?})",
        last_timing_source.get()
    );

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

    // ── AT-3006: headless (vk=None) fails closed, zero artifacts ──────────────

    /// AT-3006 (replaces the M3.22 AT-2984·A mock-completes leg): `run_optimize`
    /// on the shared-len-hole fixture with `vk == None` returns
    /// `OptimizeError::GpuUnavailable` and writes ZERO artifacts (no `.spv`,
    /// no sidecar) — the real bench cannot run headless, and M3.23 fails
    /// closed rather than emitting a fake/mock winner.
    #[test]
    fn at_3006_headless_gpu_unavailable_zero_artifacts() {
        let src = "@kernel @workgroup(64,1,1) @strategy { WG: ?[32, 64] } \
                   @intent(\"AT-3006 shared-len hole fixture\") @complexity(O(1)) \
                   fn k(out: buffer[i32]) -> void { \
                   shared t: shared[i32, ?WG]; \
                   let lid: u32 = gid(0u32); \
                   t[lid] = out[lid]; \
                   workgroup_barrier(); \
                   out[lid] = t[lid]; \
                   return; }";

        let dir = tempfile::tempdir().expect("tempdir");
        let input_path = dir.path().join("fixture.axc");
        std::fs::write(&input_path, src).expect("write fixture");
        let output_path = dir.path().join("fixture.spv");

        let result = run_optimize(&input_path, &output_path, "none", None, None);
        assert!(
            matches!(result, Err(OptimizeError::GpuUnavailable(_))),
            "headless run_optimize must fail closed with GpuUnavailable; got {result:?}"
        );
        assert!(!output_path.exists(), "no .spv may be written headless");
        assert!(!strategy_sidecar_path(&output_path).exists(), "no sidecar may be written headless");
    }

    // ── AT-3015: F4 fires + sibling-naming, decoupled from timing ─────────────

    /// AT-3015 (r3 MINOR): an in-crate test calling the PRIVATE
    /// `cross_validate_winner_module_wide` directly (in-module visibility via
    /// `use super::*` — no `pub(crate)` promotion needed) on a hand-substituted
    /// sibling-breaking resolved source (`?BAD` -> 8) fails closed naming the
    /// broken sibling. Fully deterministic, no GPU, no bench dependence —
    /// preserves the F4-fires + sibling-naming coverage of the old
    /// `at_2970_winner_invalid_for_sibling_fails_closed_zero_artifacts`
    /// without depending on the (now real, non-deterministic) bench timing to
    /// force a specific winner.
    #[test]
    fn at_3015_cross_validate_winner_module_wide_names_broken_sibling() {
        let src_text = concat!(
            "@kernel\n",
            "@workgroup(?BAD, 1, 1)\n",
            "@strategy { BAD: ?[8, 2] }\n",
            "@intent(\"bench target: legal for both candidates as a 1-D workgroup\")\n",
            "@complexity(O(n))\n",
            "fn bench_target(x: buffer[u32]) -> void {\n",
            "    let i: u32 = gid(0u32);\n",
            "    x[i] = i;\n",
            "    return;\n",
            "}\n",
            "@kernel\n",
            "@workgroup(?BAD, 16, 16)\n",
            "@strategy { BAD: ?[8, 2] }\n",
            "@intent(\"sibling: BAD=8 overflows the 1024-invocation ceiling (8*16*16=2048)\")\n",
            "@complexity(O(n))\n",
            "fn sibling(y: buffer[u32]) -> void {\n",
            "    let i: u32 = gid(0u32);\n",
            "    y[i] = i;\n",
            "    return;\n",
            "}\n",
        );
        let mut bad: BTreeMap<String, i64> = BTreeMap::new();
        bad.insert("BAD".to_string(), 8);
        let resolved: String = crate::substitute_strategy_holes(src_text, &bad);

        let result = cross_validate_winner_module_wide(&resolved);
        match result {
            Err(OptimizeError::WinnerInvalidModuleWide { broken_kernel, detail }) => {
                assert_eq!(broken_kernel.as_deref(), Some("sibling"), "F4 must name the broken sibling; detail={detail}");
            }
            other => panic!("expected WinnerInvalidModuleWide naming `sibling`; got {other:?}"),
        }
    }
}
