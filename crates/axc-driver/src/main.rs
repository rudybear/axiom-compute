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
use axc_driver::{Cli, Command, compile_file_debug, build_bench_command};
use axc_driver::optimize::run_optimize;
use axc_driver::mcp::{run_mcp_server, LogTarget};

fn main() -> miette::Result<()> {
    let cli: Cli = Cli::parse();

    match cli.command {
        Command::Compile { input, output, strategy_values, debug } => {
            if strategy_values.is_empty() {
                compile_file_debug(&input, &output, debug).map_err(|e| {
                    miette::miette!("{}", e)
                })
            } else {
                // M2.3: per-variant compilation with explicit hole assignments.
                let mut assignments: std::collections::BTreeMap<String, i64> =
                    std::collections::BTreeMap::new();
                for sv in strategy_values {
                    assignments.insert(sv.name, sv.value);
                }
                let source: String = std::fs::read_to_string(&input)
                    .map_err(|e| miette::miette!("io error reading {:?}: {}", input, e))?;
                let (bytes, _meta) = axc_driver::compile_source_with_assignments(&source, &assignments)
                    .map_err(|e| miette::miette!("{}", e))?;
                std::fs::write(&output, &bytes)
                    .map_err(|e| miette::miette!("io error writing {:?}: {}", output, e))?;
                Ok(())
            }
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
        Command::Optimize { input, output, correctness } => {
            run_optimize(&input, &output, &correctness)
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
            workgroups, push_constants_base64, seed, strategy_values,
        } => {
            run_rewrite_verify(
                original, rewritten, tol, buffer_sizes, size, output_sizes,
                workgroups, push_constants_base64, seed, strategy_values,
            )
        }
        Command::Verify { input, json } => run_verify(input, json),
        Command::Test {
            input, fuzz, runs, seed, debug_oracle, no_debug_oracle, violate,
            buffer_sizes, size, workgroups, scalar_window, strategy_values,
        } => run_test(
            input, fuzz, runs, seed, debug_oracle, no_debug_oracle, violate,
            buffer_sizes, size, workgroups, scalar_window, strategy_values,
        ),
    }
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
        match resolve_size_shortcut(&source, &assignments, n) {
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
        match resolve_size_shortcut(&original_src, &assignments, n) {
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
fn resolve_size_shortcut(
    original_src: &str,
    assignments: &std::collections::BTreeMap<String, i64>,
    n: usize,
) -> Result<Vec<usize>, String> {
    let (_bytes, meta) = axc_driver::compile_source_with_assignments(original_src, assignments)
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
