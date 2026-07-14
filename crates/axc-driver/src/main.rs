//! `axc` binary entry point.
//!
//! Parses the CLI, dispatches to `compile_file` or a debug lex dump,
//! renders diagnostics via miette, and exits non-zero on error.
//!
//! M2.4 adds `axc mcp [--log stderr|null]` — a JSON-RPC 2.0 stdio MCP server.
//!
//! M3.15 (EB.3) adds `axc bench [--filter NAME] [--bless]` — a discoverable
//! wrapper over `cargo bench -p axc-driver`. The argv/env mapping is the pure,
//! unit-tested `build_bench_command` (lib.rs); this file only spawns it.

use std::path::PathBuf;

use clap::Parser as ClapParser;
use axc_driver::{Cli, Command, build_bench_command};
use axc_driver::optimize::run_optimize;
use axc_driver::mcp::{run_mcp_server, LogTarget};

fn main() -> miette::Result<()> {
    let cli: Cli = Cli::parse();

    match cli.command {
        Command::Compile { input, output, strategy_values, debug, kernel, all } => {
            run_compile(input, output, strategy_values, debug, kernel, all)
        }
        Command::Lex { input } => {
            let source: String = std::fs::read_to_string(&input).map_err(|e| {
                miette::miette!("io error reading {:?}: {}", input, e)
            })?;
            let (tokens, _errors) = axc_lexer::tokenize(&source);
            for tok in &tokens {
                println!("{:?}", tok.kind);
            }
            Ok(())
        }
        Command::Optimize { input, output, correctness, kernel } => {
            run_optimize(&input, &output, &correctness, kernel.as_deref())
                .map_err(|e| miette::miette!("{}", e))
        }
        Command::Mcp { log } => {
            // Parse log target; unknown values fall back to stderr with a warning.
            let target: LogTarget = match log.as_str() {
                "null" => LogTarget::Null,
                "stderr" => LogTarget::Stderr,
                other => {
                    eprintln!("axc mcp: unknown --log value {:?}, defaulting to stderr", other);
                    LogTarget::Stderr
                }
            };
            run_mcp_server(
                std::io::BufReader::new(std::io::stdin().lock()),
                std::io::stdout().lock(),
                target,
            ).map_err(|e| miette::miette!("mcp server: {}", e))
        }
        Command::Bench { filter, bless } => run_bench(filter.as_deref(), bless),
        Command::RewriteVerify {
            original, rewritten, tol, buffer_sizes, size, output_sizes,
            workgroups, push_constants_base64, seed, strategy_values, kernel,
        } => {
            run_rewrite_verify(
                original, rewritten, tol, buffer_sizes, size, output_sizes,
                workgroups, push_constants_base64, seed, strategy_values, kernel,
            )
        }
        Command::Verify { input, json } => run_verify(input, json),
        Command::Test {
            input, fuzz, runs, seed, debug_oracle, no_debug_oracle, violate,
            buffer_sizes, size, workgroups, scalar_window, strategy_values, kernel,
        } => run_test(
            input, fuzz, runs, seed, debug_oracle, no_debug_oracle, violate,
            buffer_sizes, size, workgroups, scalar_window, strategy_values, kernel,
        ),
        Command::LogOpt {
            source, kernel, metric, unit, verdict, ts, note,
            from_verify, from_grid, baseline_median_ns,
        } => run_log_opt(
            source, kernel, metric, unit, verdict, ts, note,
            from_verify, from_grid, baseline_median_ns,
        ),
    }
}

/// M3.19 (FG.3): `axc log-opt` handler.
///
/// Resolves the CLI surface into one `OptimizationLogEntry` (raw flags,
/// `--from-verify`, or `--from-grid` — see `cli.rs::Command::LogOpt`'s doc
/// comment for the exact three-way grammar), calls the fail-closed writer,
/// prints a small JSON status to stdout, and exits: `0` on success,
/// `LogOptError::exit_code()` on a writer failure (`3` = Full, `4` =
/// Multikernel, `5` = UnsupportedVersion, `2` = any other writer error), `2`
/// on a CLI-layer usage error (bad flag combination, unreadable/malformed
/// `--from-*` JSON, bad `--unit`/`--verdict` string). Never returns.
#[allow(clippy::too_many_arguments)]
fn run_log_opt(
    source: PathBuf,
    kernel: Option<String>,
    metric: Option<i64>,
    unit: Option<String>,
    verdict: Option<String>,
    ts: Option<String>,
    note: Option<String>,
    from_verify: Option<PathBuf>,
    from_grid: Option<PathBuf>,
    baseline_median_ns: Option<u64>,
) -> ! {
    use axc_driver::log_opt::{append_optimization_log, entry_from_grid_record};
    use axc_hir::opt_log::{OptLogUnit, OptLogVerdict, OptimizationLogEntry};
    use axc_driver::mcp::format_rfc3339_utc;

    if from_verify.is_some() && from_grid.is_some() {
        log_opt_usage_error("--from-verify and --from-grid are mutually exclusive".to_string());
    }

    // §1.5: default `now()` is non-reproducible and production-only; every
    // golden/round-trip test fixture MUST pass an explicit `--ts`.
    let resolved_ts: String = ts.unwrap_or_else(|| format_rfc3339_utc(std::time::SystemTime::now()));

    let entry: OptimizationLogEntry = if let Some(verify_path) = from_verify {
        // Minimal on-disk-JSON shape (deliberately NOT `rewrite_verify::VerifyReport`
        // itself, which carries a `milestone: &'static str` field that cannot be
        // `Deserialize`d from an arbitrary runtime buffer): only `verdict` and the
        // two optional `kernel_name`s are needed here. `verdict`'s JSON string is
        // ALREADY the identical SCREAMING_SNAKE_CASE ident `OptLogVerdict` uses
        // (`rewrite_verify::Verdict`'s `#[serde(rename_all = "SCREAMING_SNAKE_CASE")]`),
        // so `OptLogVerdict::from_ident` on it IS `verdict_from_rewrite_verify`
        // (the same exhaustive §1.5 mapping, applied to the wire format instead of
        // the Rust enum — `log_opt::verdict_from_rewrite_verify` is exercised
        // directly by the AT-2907 unit tests for the in-process Rust-value path).
        #[derive(serde::Deserialize)]
        struct VerifyReportJson {
            verdict: String,
            original: Option<KernelSummaryJson>,
            rewritten: Option<KernelSummaryJson>,
        }
        #[derive(serde::Deserialize)]
        struct KernelSummaryJson {
            kernel_name: String,
        }

        let text: String = match std::fs::read_to_string(&verify_path) {
            Ok(t) => t,
            Err(e) => log_opt_usage_error(format!("io error reading {verify_path:?}: {e}")),
        };
        let report: VerifyReportJson = match serde_json::from_str(&text) {
            Ok(r) => r,
            Err(e) => log_opt_usage_error(format!("--from-verify: failed to parse {verify_path:?} as a VerifyReport: {e}")),
        };
        let resolved_verdict: OptLogVerdict = match OptLogVerdict::from_ident(&report.verdict) {
            Some(v) => v,
            None => log_opt_usage_error(format!("--from-verify: unrecognized verdict `{}` in {verify_path:?}", report.verdict)),
        };
        let resolved_kernel: String = kernel
            .or_else(|| report.rewritten.map(|k| k.kernel_name))
            .or_else(|| report.original.map(|k| k.kernel_name))
            .unwrap_or_default();
        let resolved_unit: OptLogUnit = match unit.as_deref() {
            Some(u) => match OptLogUnit::from_ident(u) {
                Some(u) => u,
                None => log_opt_usage_error(format!("invalid --unit `{u}`; expected one of mtflops, pct_milli, ns, us, ms, bytes")),
            },
            None => OptLogUnit::Nanoseconds,
        };
        OptimizationLogEntry {
            ts: resolved_ts,
            kernel: resolved_kernel,
            metric: metric.unwrap_or(0),
            unit: resolved_unit,
            verdict: resolved_verdict,
            note,
            forward_unknown: Vec::new(),
        }
    } else if let Some(grid_path) = from_grid {
        let Some(kernel_name) = kernel else {
            log_opt_usage_error("--from-grid requires --kernel (a HistoryEntryRecord carries no kernel name)".to_string());
        };
        let text: String = match std::fs::read_to_string(&grid_path) {
            Ok(t) => t,
            Err(e) => log_opt_usage_error(format!("io error reading {grid_path:?}: {e}")),
        };
        let record: axc_driver::mcp::HistoryEntryRecord = match serde_json::from_str(&text) {
            Ok(r) => r,
            Err(e) => log_opt_usage_error(format!("--from-grid: failed to parse {grid_path:?} as a HistoryEntryRecord: {e}")),
        };
        entry_from_grid_record(&record, kernel_name, baseline_median_ns, Some(resolved_ts), note)
    } else {
        let (Some(kernel_name), Some(metric_val), Some(unit_str), Some(verdict_str)) = (kernel, metric, unit, verdict) else {
            log_opt_usage_error(
                "raw mode requires --kernel, --metric, --unit, and --verdict (or use --from-verify / --from-grid)".to_string(),
            );
        };
        let resolved_unit: OptLogUnit = match OptLogUnit::from_ident(&unit_str) {
            Some(u) => u,
            None => log_opt_usage_error(format!("invalid --unit `{unit_str}`; expected one of mtflops, pct_milli, ns, us, ms, bytes")),
        };
        let resolved_verdict: OptLogVerdict = match OptLogVerdict::from_ident(&verdict_str) {
            Some(v) => v,
            None => log_opt_usage_error(format!(
                "invalid --verdict `{verdict_str}`; expected one of PASS, FAIL, REJECT, SKIPPED, NONDETERMINISTIC_ORACLE, ERROR, REGRESS"
            )),
        };
        OptimizationLogEntry {
            ts: resolved_ts,
            kernel: kernel_name,
            metric: metric_val,
            unit: resolved_unit,
            verdict: resolved_verdict,
            note,
            forward_unknown: Vec::new(),
        }
    };

    match append_optimization_log(&source, entry) {
        Ok(()) => {
            println!("{}", serde_json::json!({ "status": "ok", "source": source.display().to_string() }));
            std::process::exit(0);
        }
        Err(e) => {
            let exit_code: i32 = e.exit_code();
            println!("{}", serde_json::json!({ "status": "error", "detail": e.to_string() }));
            std::process::exit(exit_code);
        }
    }
}

/// Emit a minimal usage-error JSON to stdout and exit 2 (CLI-layer failures
/// that occur BEFORE an `OptimizationLogEntry` can even be built — mirrors
/// `emit_usage_error`'s role for `rewrite-verify`).
fn log_opt_usage_error(detail: String) -> ! {
    println!("{}", serde_json::json!({ "status": "error", "detail": detail }));
    std::process::exit(2);
}

/// M3.18 (FG.8): `axc verify` handler.
///
/// Pure front-end (no GPU, no Vulkan probe at all). Reads the source, calls
/// `verify::verify_source`, prints the human summary (default) or `--json`, and
/// exits per `exit_code_for_verify`. An unreadable file is the ONLY way
/// `verify_source` itself never sees — mapped directly to the `ERROR` verdict here.
fn run_verify(input: PathBuf, json: bool) -> ! {
    use axc_driver::verify::{exit_code_for_verify, verify_source, VerifyReport, VerifyVerdict, VerifySummary};

    let source: String = match std::fs::read_to_string(&input) {
        Ok(s) => s,
        Err(e) => {
            let report = VerifyReport {
                verdict: VerifyVerdict::Error,
                milestone: "M3.18",
                file: input.display().to_string(),
                kernel: None,
                kernels: Vec::new(),
                checks: Vec::new(),
                diagnostics: vec![axc_driver::verify::VerifyDiag {
                    severity: "error".to_string(),
                    code: "io_error".to_string(),
                    message: format!("io error reading {input:?}: {e}"),
                    line: None,
                }],
                summary: VerifySummary { errors: 1, warnings: 0 },
            };
            print_verify_report(&report, json);
            std::process::exit(exit_code_for_verify(VerifyVerdict::Error));
        }
    };

    let report = verify_source(&source, &input.display().to_string());
    print_verify_report(&report, json);
    std::process::exit(exit_code_for_verify(report.verdict));
}

fn print_verify_report(report: &axc_driver::verify::VerifyReport, json: bool) {
    if json {
        let s: String = serde_json::to_string_pretty(report).unwrap_or_else(|e| format!("{{\"error\":\"serialize failed: {e}\"}}"));
        println!("{s}");
    } else {
        print!("{}", axc_driver::verify::render_human(report));
    }
}

/// M3.18 (FG.8): `axc test --fuzz` handler.
///
/// `--fuzz` is currently the ONLY mode; its absence is a usage ERROR (§3.1).
/// Resolves the CLI surface into a `fuzz::FuzzRequest`, probes for a local
/// Vulkan device (absence -> `SKIPPED`/`gpu_unavailable`, first-class per §3.1
/// step 5), calls the core, prints the `FuzzReport` JSON, and exits per
/// `exit_code_for_fuzz`. Never returns.
///
/// QA-fix (post-M3.18): missing `--buffer-sizes`/`--size` is NOT gated here
/// before calling `fuzz_kernel` -- doing so pre-empted `fuzz_kernel`'s own
/// `solve()`-based UNSATISFIABLE short-circuit, which must run first (§3.1
/// step 2 precedes step 5). An empty `Vec` is threaded through instead;
/// `fuzz_kernel` raises the equivalent usage ERROR/exit 2 itself, but only
/// AFTER `solve()` has had first refusal.
#[allow(clippy::too_many_arguments)]
fn run_test(
    input: PathBuf,
    fuzz_mode: bool,
    runs: u32,
    seed: u64,
    debug_oracle_flag: bool,
    no_debug_oracle_flag: bool,
    violate: bool,
    buffer_sizes_arg: Vec<usize>,
    size: Option<usize>,
    workgroups: Option<Vec<u32>>,
    scalar_window: i64,
    strategy_values: Vec<axc_driver::cli::StrategyValue>,
    kernel: Option<String>,
) -> ! {
    use axc_driver::fuzz::{exit_code_for_fuzz, fuzz_kernel, FuzzRequest};

    if !fuzz_mode {
        emit_fuzz_usage_error("--fuzz is the only axc test mode in v1; pass --fuzz (e.g. `axc test --fuzz <file>`)".to_string());
    }
    if runs == 0 {
        emit_fuzz_usage_error("--runs must be >= 1".to_string());
    }

    let source: String = match std::fs::read_to_string(&input) {
        Ok(s) => s,
        Err(e) => emit_fuzz_usage_error(format!("io error reading {input:?}: {e}")),
    };

    let mut assignments: std::collections::BTreeMap<String, i64> = std::collections::BTreeMap::new();
    for sv in strategy_values {
        assignments.insert(sv.name, sv.value);
    }

    let resolved_buffer_sizes: Vec<usize> = if !buffer_sizes_arg.is_empty() {
        buffer_sizes_arg
    } else if let Some(n) = size {
        match resolve_size_shortcut(&source, &assignments, n, kernel.as_deref()) {
            Ok(sizes) => sizes,
            Err(e) => emit_fuzz_usage_error(e),
        }
    } else {
        // Neither `--buffer-sizes` nor `--size` was supplied. Do NOT usage-error
        // here: `fuzz_kernel`'s own internal ordering resolves UNSATISFIABLE (via
        // `solve()` over the kernel's `@precondition`s) BEFORE buffer sizes are
        // ever consulted (§3.1: constraint solving precedes dispatch prep).
        // Pass an empty `Vec` through unconditionally; if the kernel turns out
        // to be satisfiable and genuinely needs to dispatch, `fuzz_kernel`'s own
        // `buffer_sizes.len()` mismatch check (which fires AFTER `solve()`)
        // raises the identical usage ERROR/exit 2 -- this only fixes the
        // refusal ORDER, not the outcome, for kernels that would dispatch.
        Vec::new()
    };

    let workgroups_arr: Option<[u32; 3]> = match workgroups {
        None => None,
        Some(v) if v.len() == 3 => Some([v[0], v[1], v[2]]),
        Some(v) => emit_fuzz_usage_error(format!("--workgroups must have exactly 3 comma-separated values (x,y,z); got {}", v.len())),
    };

    let debug_oracle: Option<bool> = match (debug_oracle_flag, no_debug_oracle_flag) {
        (true, false) => Some(true),
        (false, true) => Some(false),
        (false, false) => None,
        (true, true) => emit_fuzz_usage_error("--debug-oracle and --no-debug-oracle are mutually exclusive".to_string()),
    };

    let req = FuzzRequest {
        source,
        runs,
        seed,
        debug_oracle,
        violate,
        buffer_sizes: resolved_buffer_sizes,
        workgroups: workgroups_arr,
        scalar_window,
        strategy_assignments: assignments,
        kernel,
    };

    let vk: Option<axc_runtime::VulkanContext> = if axc_runtime::probe_vulkan_available() {
        axc_runtime::VulkanContext::new().ok()
    } else {
        None
    };

    let report = fuzz_kernel(&req, vk.as_ref());
    let json: String = serde_json::to_string_pretty(&report).unwrap_or_else(|e| format!("{{\"error\":\"serialize failed: {e}\"}}"));
    println!("{json}");
    std::process::exit(exit_code_for_fuzz(report.verdict));
}

/// Emit a minimal `FuzzReport`-shaped `ERROR` JSON to stdout and exit 2. Mirrors
/// `emit_usage_error` (rewrite-verify's CLI-layer usage-failure convention).
fn emit_fuzz_usage_error(detail: String) -> ! {
    use axc_driver::fuzz::{exit_code_for_fuzz, FuzzReason, FuzzReport, FuzzVerdict};
    let report = FuzzReport {
        verdict: FuzzVerdict::Error,
        milestone: "M3.18",
        kernel: String::new(),
        seed: 0,
        runs_requested: 0,
        runs_executed: 0,
        debug_oracle: false,
        violate_leg: false,
        device: None,
        constraints: Vec::new(),
        counterexample: None,
        reason: Some(FuzzReason { kind: "usage".to_string(), detail: Some(serde_json::json!({ "detail": detail })) }),
        host_recheck: None,
        notes: Vec::new(),
    };
    let json: String = serde_json::to_string_pretty(&report).unwrap_or_else(|e| format!("{{\"error\":\"serialize failed: {e}\"}}"));
    println!("{json}");
    std::process::exit(exit_code_for_fuzz(FuzzVerdict::Error));
}

/// M3.16 (FG.1): `axc rewrite-verify` handler.
///
/// Resolves the CLI surface into a `rewrite_verify::VerifyRequest`, probes for
/// a local Vulkan device (absence ⇒ `SKIPPED`/`gpu_unavailable`, a first-class
/// CI outcome — NOT an error), calls the core, prints the `VerifyReport` JSON
/// to stdout, and exits with the verdict's mapped code. Never returns.
#[allow(clippy::too_many_arguments)]
fn run_rewrite_verify(
    original_path: PathBuf,
    rewritten_path: PathBuf,
    tol: String,
    buffer_sizes_arg: Vec<usize>,
    size: Option<usize>,
    output_sizes: Option<Vec<usize>>,
    workgroups: Option<Vec<u32>>,
    push_constants_base64: Option<String>,
    seed: u64,
    strategy_values: Vec<axc_driver::cli::StrategyValue>,
    kernel: Option<String>,
) -> ! {
    use axc_driver::rewrite_verify::{
        exit_code_for_verdict, verify_rewrite, TolerancePolicy, VerifyRequest,
    };

    let tolerance: TolerancePolicy = match tol.parse() {
        Ok(t) => t,
        Err(e) => emit_usage_error(&tol, format!("invalid --tol: {e}")),
    };

    let original_src: String = match std::fs::read_to_string(&original_path) {
        Ok(s) => s,
        Err(e) => emit_usage_error(&tol, format!("io error reading {original_path:?}: {e}")),
    };
    let rewritten_src: String = match std::fs::read_to_string(&rewritten_path) {
        Ok(s) => s,
        Err(e) => emit_usage_error(&tol, format!("io error reading {rewritten_path:?}: {e}")),
    };

    let mut assignments: std::collections::BTreeMap<String, i64> = std::collections::BTreeMap::new();
    for sv in strategy_values {
        assignments.insert(sv.name, sv.value);
    }

    let resolved_buffer_sizes: Vec<usize> = if !buffer_sizes_arg.is_empty() {
        buffer_sizes_arg
    } else if let Some(n) = size {
        match resolve_size_shortcut(&original_src, &assignments, n, kernel.as_deref()) {
            Ok(sizes) => sizes,
            Err(e) => emit_usage_error(&tol, e),
        }
    } else {
        emit_usage_error(&tol, "must supply --buffer-sizes or --size".to_string());
    };

    let workgroups_arr: Option<[u32; 3]> = match workgroups {
        None => None,
        Some(v) if v.len() == 3 => Some([v[0], v[1], v[2]]),
        Some(v) => emit_usage_error(
            &tol,
            format!("--workgroups must have exactly 3 comma-separated values (x,y,z); got {}", v.len()),
        ),
    };

    let push_constants: Option<Vec<u8>> = match push_constants_base64 {
        None => None,
        Some(b64) => match axc_driver::mcp::base64_decode(&b64) {
            Ok(bytes) => Some(bytes),
            Err(e) => emit_usage_error(&tol, format!("invalid --push-constants-base64: {e}")),
        },
    };

    let req = VerifyRequest {
        original: original_src,
        rewritten: rewritten_src,
        assignments,
        tolerance,
        buffer_sizes: resolved_buffer_sizes,
        output_sizes,
        workgroups: workgroups_arr,
        push_constants,
        seed,
        kernel,
    };

    // Absence of a usable Vulkan device is a first-class SKIPPED outcome for
    // the CLI/CI path (§3.1 step 5) — NOT a usage error.
    let vk: Option<axc_runtime::VulkanContext> = if axc_runtime::probe_vulkan_available() {
        axc_runtime::VulkanContext::new().ok()
    } else {
        None
    };

    let report = verify_rewrite(&req, vk.as_ref());
    let json: String = serde_json::to_string_pretty(&report)
        .unwrap_or_else(|e| format!("{{\"error\":\"serialize failed: {e}\"}}"));
    println!("{json}");
    std::process::exit(exit_code_for_verdict(report.verdict));
}

/// `--size N` eligibility check + expansion (§3.2): only for element-wise
/// kernels whose buffers are all the SAME `ScalarTy`, with no coopmat and no
/// shared memory (a matmul with differently-shaped buffers would otherwise
/// silently get equal-and-wrong sizes and a trivial partial-coverage PASS).
///
/// QA-fix (post-M3.21, MEDIUM finding): `kernel` MUST be threaded into the
/// same kernel-aware compile surface (`compile_source_with_assignments_kernel`)
/// that the non-`--size` `--buffer-sizes` path already uses, not the
/// kernel-agnostic `compile_source_with_assignments`. Before this fix, any
/// `--size` invocation on a 2+-kernel source always failed closed with
/// `AmbiguousKernel` regardless of `--kernel`, silently making `--kernel`
/// inert for this one flag combination (fails closed, not a correctness bug,
/// but a real functional gap — see M3.21-qa.json discrepancy #1).
fn resolve_size_shortcut(
    original_src: &str,
    assignments: &std::collections::BTreeMap<String, i64>,
    n: usize,
    kernel: Option<&str>,
) -> Result<Vec<usize>, String> {
    let (_bytes, meta) = axc_driver::compile_source_with_assignments_kernel(original_src, assignments, kernel)
        .map_err(|e| format!("--size: failed to compile original to determine buffer layout: {e}"))?;
    if meta.coopmat.is_some() || meta.shared_memory_bytes > 0 {
        return Err(
            "--size is restricted to non-coopmat, non-shared, element-wise kernels; supply explicit --buffer-sizes".to_string()
        );
    }
    let buffers = &meta.binding_plan.buffers;
    if buffers.is_empty() {
        return Err("--size: kernel has no buffer bindings".to_string());
    }
    let first_elem = buffers[0].ty.elem;
    if !buffers.iter().all(|b| b.ty.elem == first_elem) {
        return Err(
            "--size is restricted to kernels whose buffers are all the SAME ScalarTy; supply explicit --buffer-sizes".to_string()
        );
    }
    let elem_bytes: usize = (first_elem.bit_width() as usize).div_ceil(8);
    let size_bytes: usize = n * elem_bytes;
    Ok(vec![size_bytes; buffers.len()])
}

/// Emit a minimal `VerifyReport`-shaped `ERROR` JSON to stdout and exit 2.
///
/// Used for CLI-level usage failures that occur BEFORE a `VerifyRequest` can
/// even be built (bad `--tol` grammar, unreadable source file, malformed
/// `--workgroups`/`--push-constants-base64`, missing `--buffer-sizes`/`--size`).
/// Keeps the "verdict JSON always on stdout" contract (§5) uniform across every
/// exit path, not just the ones that reach `verify_rewrite`.
fn emit_usage_error(policy: &str, detail: String) -> ! {
    use axc_driver::rewrite_verify::{Reason, Verdict, VerifyReport};
    let report = VerifyReport {
        verdict: Verdict::Error,
        milestone: "M3.16",
        policy: policy.to_string(),
        tolerance_overridden: false,
        seed: 0,
        stage: "preflight".to_string(),
        original: None,
        rewritten: None,
        device: None,
        buffers_compared: Vec::new(),
        reason: Some(Reason { kind: "infra_error".to_string(), detail: Some(serde_json::json!({ "detail": detail })) }),
        notes: Vec::new(),
    };
    let json: String = serde_json::to_string_pretty(&report)
        .unwrap_or_else(|e| format!("{{\"error\":\"serialize failed: {e}\"}}"));
    println!("{json}");
    std::process::exit(2);
}

/// M3.21 (FG.9): `axc compile` handler — routes to the plain single-output
/// path (byte-identical to pre-M3.21 when `kernel`/`all` are both absent),
/// the `--kernel NAME` single-select path, or the `--all` multi-emit path.
/// `--kernel` and `--all` are mutually exclusive at the clap layer
/// (`conflicts_with`), so at most one of `kernel`/`all` is set here.
fn run_compile(
    input: PathBuf,
    output: PathBuf,
    strategy_values: Vec<axc_driver::cli::StrategyValue>,
    debug: bool,
    kernel: Option<String>,
    all: bool,
) -> miette::Result<()> {
    let mut assignments: std::collections::BTreeMap<String, i64> = std::collections::BTreeMap::new();
    for sv in strategy_values {
        assignments.insert(sv.name, sv.value);
    }

    if all {
        let source: String = std::fs::read_to_string(&input)
            .map_err(|e| miette::miette!("io error reading {:?}: {}", input, e))?;
        let substituted: String = if assignments.is_empty() {
            source
        } else {
            axc_driver::substitute_strategy_holes(&source, &assignments)
        };
        let compiled = axc_driver::compile_module_all(&substituted, debug)
            .map_err(|e| miette::miette!("{}", e))?;
        let stem: PathBuf = axc_driver::all_output_stem(&output);
        for ck in &compiled {
            let spv_path: PathBuf = axc_driver::per_kernel_output_path(&stem, &ck.name);
            std::fs::write(&spv_path, &ck.spirv)
                .map_err(|e| miette::miette!("io error writing {:?}: {}", spv_path, e))?;
            let sidecar_path: PathBuf = axc_driver::metadata_sidecar_path(&spv_path);
            ck.metadata.save(&sidecar_path)
                .map_err(|e| miette::miette!("metadata emit failed: {}", e))?;
        }
        return Ok(());
    }

    if assignments.is_empty() {
        // Byte-identical to the pre-M3.21 single-kernel path when kernel==None
        // (golden-identity gate, §7 of the spec).
        axc_driver::compile_file_kernel(&input, &output, kernel.as_deref(), debug)
            .map_err(|e| miette::miette!("{}", e))
    } else {
        // M2.3: per-variant compilation with explicit hole assignments.
        let source: String = std::fs::read_to_string(&input)
            .map_err(|e| miette::miette!("io error reading {:?}: {}", input, e))?;
        let (bytes, _meta) = axc_driver::compile_source_with_assignments_kernel(&source, &assignments, kernel.as_deref())
            .map_err(|e| miette::miette!("{}", e))?;
        std::fs::write(&output, &bytes)
            .map_err(|e| miette::miette!("io error writing {:?}: {}", output, e))?;
        Ok(())
    }
}

/// M3.15 (EB.3): spawn `cargo bench -p axc-driver` per `build_bench_command`'s
/// pure argv/env mapping, with **inherited** stdio (child bench output streams
/// straight to the user's terminal).
///
/// This function shells out, so it is intentionally NOT unit-tested — only
/// `build_bench_command` (the argv/env mapping) is (AT-2828). See that
/// function's doc-comment for the "installed binary, not `cargo run`"
/// self-deadlock footgun this wrapper exists to sidestep for its own callers.
fn run_bench(filter: Option<&str>, bless: bool) -> miette::Result<()> {
    let (program, args, env_overrides) = build_bench_command(filter, bless);

    let status = std::process::Command::new(&program)
        .args(&args)
        .envs(env_overrides)
        .status();

    match status {
        Ok(status) if status.success() => Ok(()),
        Ok(status) => Err(miette::miette!(
            "axc bench: `{program} {}` exited with {status}",
            args.join(" "),
        )),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(miette::miette!(
            "axc bench: `cargo` not found on PATH — install a Rust toolchain"
        )),
        Err(e) => Err(miette::miette!("axc bench: failed to spawn `{program}`: {e}")),
    }
}
