//! Command-line interface definition for `axc`.
//!
//! M0 ships two subcommands:
//! - `axc compile <input.axc> -o <output.spv>`: full pipeline to SPIR-V
//! - `axc lex <input.axc>`: debug token dump
//!
//! M2.3 adds:
//! - `axc optimize <input.axc> -o <output.spv>`: run grid search over @strategy
//!   holes and emit the winning SPIR-V.
//! - `--strategy-value name=value` on `compile`: compile with a specific hole
//!   assignment (bypasses grid search, used by the optimizer internally).
//!
//! M2.4 adds:
//! - `axc mcp [--log stderr|null]`: start a JSON-RPC 2.0 stdio MCP server.
//!
//! M3.15 (EB.3) adds:
//! - `axc bench [--filter NAME] [--bless]`: discoverable wrapper over
//!   `cargo bench -p axc-driver`.
//!
//! M3.16 (FG.1) adds:
//! - `axc rewrite-verify <original.axc> <rewritten.axc> [--tol ...] ...`: the
//!   LLM structural-rewrite verifier CLI primitive (verify, don't generate).

use std::path::PathBuf;

/// Root CLI entry point.
#[derive(clap::Parser)]
#[command(name = "axc", about = "AXIOM-Compute compiler")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

/// A single strategy hole assignment, as parsed from `--strategy-value name=value`.
///
/// # Semantics vs `--holes`
///
/// `--strategy-value` assigns a *specific* concrete integer to one hole name.
/// Multiple `--strategy-value` flags may be supplied to assign several holes.
/// This is how the optimizer internally calls `axc compile` for each variant —
/// one `--strategy-value` per hole, with the candidate selected by the grid search.
///
/// It is NOT the same as `--holes` (which would enumerate candidates).  There is no
/// `--holes` flag on `compile`; enumeration is internal to `axc optimize`.
#[derive(Debug, Clone)]
pub struct StrategyValue {
    /// Hole name.
    pub name: String,
    /// Concrete candidate value.
    pub value: i64,
}

impl std::str::FromStr for StrategyValue {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (name, val_str) = s.split_once('=')
            .ok_or_else(|| format!(
                "invalid --strategy-value `{s}`; expected `name=value` (e.g. `workgroup_x=64`)"
            ))?;
        if name.is_empty() {
            return Err(format!(
                "invalid --strategy-value `{s}`; hole name must not be empty"
            ));
        }
        let value: i64 = val_str.parse().map_err(|_| format!(
            "invalid --strategy-value `{s}`; `{val_str}` is not a valid integer"
        ))?;
        Ok(StrategyValue { name: name.to_string(), value })
    }
}

/// Available subcommands.
#[derive(clap::Subcommand)]
pub enum Command {
    /// Compile an .axc source file to a SPIR-V binary.
    ///
    /// When `--strategy-value` flags are supplied, the named holes are substituted
    /// with their concrete values before codegen.  This is the per-variant
    /// compilation path used internally by `axc optimize`.
    ///
    /// Note: `--strategy-value` assigns a *specific* value to one hole (e.g.
    /// `--strategy-value workgroup_x=64`), NOT a list of candidates.  Candidate
    /// enumeration is internal to `axc optimize`.
    Compile {
        /// Input source file (`.axc`)
        input: PathBuf,
        /// Output SPIR-V file (`.spv`)
        #[arg(short, long)]
        output: PathBuf,
        /// Assign a concrete value to a @strategy hole: `name=value`.
        ///
        /// May be repeated for multiple holes.  Each flag assigns exactly one
        /// hole.  Example: `--strategy-value workgroup_x=64 --strategy-value
        /// workgroup_y=4`.
        ///
        /// Semantics: substitutes the given integer for the named hole before
        /// HIR construction and codegen.  Does NOT enumerate candidates.
        #[arg(long = "strategy-value", value_name = "name=value")]
        strategy_values: Vec<StrategyValue>,
        /// Enable runtime `@precondition`/`@postcondition` checks (M3.17 / FG.4).
        ///
        /// Injects a debug-flag SSBO + atomic-violation checks into the emitted
        /// SPIR-V and writes schema-v4 metadata carrying `debug_checks`. Defaults to
        /// `false` (release): the emitted SPIR-V is then byte-identical to a build
        /// with no debug-check support at all (the golden-identity gate). No other
        /// subcommand (`optimize`, `rewrite-verify`, `mcp`) exposes this flag.
        #[arg(long)]
        debug: bool,
    },
    /// Dump the lexed token stream (debug / diagnostic use).
    Lex {
        /// Input source file (`.axc`)
        input: PathBuf,
    },
    /// Enumerate @strategy holes and run grid search to find the fastest variant.
    ///
    /// Reads the source file, extracts @strategy hole declarations, runs the
    /// Cartesian product of all candidates through codegen + benchmarking
    /// (using a no-op mock bench in this M2.3 version), and writes the winning
    /// SPIR-V binary to `--output`.
    ///
    /// A JSON result sidecar (`<output>.axc.strategy.json`) is written alongside
    /// the SPIR-V containing the winner ordinal, variant_id, assignments, and
    /// per-variant results.
    Optimize {
        /// Input source file (`.axc`)
        input: PathBuf,
        /// Output SPIR-V file (`.spv`) for the winning variant
        #[arg(short, long)]
        output: PathBuf,
        /// Correctness policy: `none`, `bit-exact`, or `fp-tol:<ulp>`.
        ///
        /// Defaults to `none`.
        #[arg(long, default_value = "none")]
        correctness: String,
    },
    /// Start a JSON-RPC 2.0 stdio MCP server for LLM agent integration.
    ///
    /// Reads NDJSON requests from stdin and writes NDJSON responses to stdout.
    /// Logging goes to stderr (or /dev/null with `--log null`).
    ///
    /// Tools exposed: initialize, load_source, enumerate_variants,
    /// compile_variant, bench_variant, grid_search, optimization_history.
    Mcp {
        /// Log destination: `stderr` (default) or `null` (discard all logs).
        ///
        /// Unknown values fall back to `stderr` with a warning.
        #[arg(long, default_value = "stderr")]
        log: String,
    },
    /// Run the AXIOM-Compute benchmark suite (wrapper over `cargo bench -p axc-driver`).
    ///
    /// Examples:
    ///   axc bench                       # run all bench groups
    ///   axc bench --filter cpu_saxpy    # run only benches whose name matches `cpu_saxpy`
    ///   axc bench --bless               # run all + promote candidate → baselines.json
    ///
    /// NOTE: run the INSTALLED/built `axc` binary, NOT `cargo run -p axc-driver -- bench`.
    /// `axc bench` spawns an inner `cargo bench` that needs the workspace target-dir build
    /// lock; launching it via `cargo run` (same target dir) makes the inner cargo block
    /// forever on the lock the outer `cargo run` holds → a self-deadlock.
    Bench {
        /// Only run benches whose Criterion name contains this substring.
        ///
        /// Passed through to `cargo bench -p axc-driver -- <FILTER>` (Criterion's
        /// positional name filter).
        #[arg(long, value_name = "NAME")]
        filter: Option<String>,

        /// Promote the measured candidate to `.pipeline/benchmarks/baselines.json`
        /// for THIS machine's key (sets `AXC_BLESS_BASELINES=1` for the child).
        #[arg(long)]
        bless: bool,
    },
    /// Verify that a rewritten kernel is observationally equivalent to the
    /// original (M3.16 / FG.1 — the LLM structural-rewrite verifier).
    ///
    /// This is the LLM-free, CI-runnable verification primitive: it never
    /// generates a rewrite, it only checks one. Writes the `VerifyReport` JSON
    /// to stdout (always, regardless of exit code) and exits `0` for
    /// PASS/SKIPPED/NONDETERMINISTIC_ORACLE, `1` for FAIL/REJECT, `2` for a
    /// usage/under-specified-request/I-O ERROR.
    RewriteVerify {
        /// Original kernel source file (`.axc`).
        original: PathBuf,
        /// Rewritten kernel source file (`.axc`) to verify against `original`.
        rewritten: PathBuf,
        /// Differential-comparison tolerance policy: `bit-exact` (default),
        /// `ulp:N`, or `rel:EPS`.
        #[arg(long, default_value = "bit-exact")]
        tol: String,
        /// One size (bytes) per buffer binding. Required unless `--size` is used.
        #[arg(long, value_delimiter = ',')]
        buffer_sizes: Vec<usize>,
        /// Uniform-element-count shortcut for same-`ScalarTy`, non-coopmat,
        /// non-shared element-wise kernels (e.g. saxpy/vector_add class).
        #[arg(long)]
        size: Option<usize>,
        /// Output-buffer sizes (bytes). Defaults to `buffer_sizes`.
        #[arg(long, value_delimiter = ',')]
        output_sizes: Option<Vec<usize>>,
        /// Explicit dispatch grid `x,y,z`. Mandatory for coopmat/shared kernels.
        #[arg(long, value_delimiter = ',')]
        workgroups: Option<Vec<u32>>,
        /// Base64-encoded (RFC 4648 §4 standard alphabet) push-constant bytes.
        #[arg(long)]
        push_constants_base64: Option<String>,
        /// Seed for deterministic input generation.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Shared `@strategy` hole assignment applied to BOTH sources: `name=value`.
        #[arg(long = "strategy-value", value_name = "name=value")]
        strategy_values: Vec<StrategyValue>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{CommandFactory, Parser};

    // ── AT-2875: `--debug` on `compile` (M3.17 FG.4) ───────────────────────────

    /// AT-2875a: `axc compile a.axc -o b.spv --debug` parses `debug: true`.
    #[test]
    fn at_2875a_compile_debug_flag_parses_true() {
        let cli = Cli::parse_from(["axc", "compile", "a.axc", "-o", "b.spv", "--debug"]);
        match cli.command {
            Command::Compile { debug, .. } => assert!(debug, "--debug must parse to true"),
            other => panic!("expected Command::Compile, got: {:?}", std::mem::discriminant(&other)),
        }
    }

    /// AT-2875b: `axc compile a.axc -o b.spv` (no `--debug`) defaults to `debug: false` (release).
    #[test]
    fn at_2875b_compile_debug_flag_defaults_false() {
        let cli = Cli::parse_from(["axc", "compile", "a.axc", "-o", "b.spv"]);
        match cli.command {
            Command::Compile { debug, .. } => assert!(!debug, "default must be release (debug: false)"),
            other => panic!("expected Command::Compile, got: {:?}", std::mem::discriminant(&other)),
        }
    }

    /// AT-2875c: no OTHER subcommand (`optimize`, `mcp`, `rewrite-verify`, `lex`,
    /// `bench`) exposes a `--debug` flag — its rendered help must not mention it.
    #[test]
    fn at_2875c_no_other_subcommand_exposes_debug() {
        let mut cmd = Cli::command();
        for name in ["optimize", "mcp", "rewrite-verify", "lex", "bench"] {
            let sub = cmd.find_subcommand_mut(name)
                .unwrap_or_else(|| panic!("Cli must have a `{name}` subcommand"));
            let help: String = sub.render_long_help().to_string();
            assert!(
                !help.contains("--debug"),
                "`{name}` must NOT expose --debug; got help:\n{help}"
            );
        }
    }

    // ── AT-2826/2827: `axc bench` CLI parsing (EB.3, M3.15) ────────────────────

    /// AT-2826: `axc bench` (no flags) parses to `Command::Bench { filter: None, bless: false }`.
    #[test]
    fn at_2826_bench_parse_default() {
        let cli = Cli::parse_from(["axc", "bench"]);
        match cli.command {
            Command::Bench { filter, bless } => {
                assert_eq!(filter, None, "default filter must be None");
                assert!(!bless, "default bless must be false");
            }
            other => panic!("expected Command::Bench, got a different variant: {:?}",
                std::mem::discriminant(&other)),
        }
    }

    /// AT-2827: `axc bench --filter cpu_saxpy --bless` parses both flags correctly.
    #[test]
    fn at_2827_bench_parse_flags() {
        let cli = Cli::parse_from(["axc", "bench", "--filter", "cpu_saxpy", "--bless"]);
        match cli.command {
            Command::Bench { filter, bless } => {
                assert_eq!(filter, Some("cpu_saxpy".to_owned()));
                assert!(bless);
            }
            other => panic!("expected Command::Bench, got a different variant: {:?}",
                std::mem::discriminant(&other)),
        }
    }

    // ── AT-2841: `axc rewrite-verify` CLI parsing (M3.16) ──────────────────────

    /// AT-2841: `axc rewrite-verify a.axc b.axc --tol rel:1e-3 --buffer-sizes 256,256`
    /// parses to `Command::RewriteVerify` with the correct fields.
    #[test]
    fn at_2841_rewrite_verify_parse_buffer_sizes() {
        let cli = Cli::parse_from([
            "axc", "rewrite-verify", "a.axc", "b.axc",
            "--tol", "rel:1e-3",
            "--buffer-sizes", "256,256",
            "--seed", "7",
        ]);
        match cli.command {
            Command::RewriteVerify { original, rewritten, tol, buffer_sizes, size, seed, .. } => {
                assert_eq!(original, PathBuf::from("a.axc"));
                assert_eq!(rewritten, PathBuf::from("b.axc"));
                assert_eq!(tol, "rel:1e-3");
                assert_eq!(buffer_sizes, vec![256, 256]);
                assert_eq!(size, None);
                assert_eq!(seed, 7);
            }
            other => panic!("expected Command::RewriteVerify, got: {:?}", std::mem::discriminant(&other)),
        }
    }

    /// AT-2841: `--size` shortcut parses; default `--tol` is `bit-exact`; default `--seed` is 0.
    #[test]
    fn at_2841b_rewrite_verify_parse_size_shortcut_and_defaults() {
        let cli = Cli::parse_from(["axc", "rewrite-verify", "a.axc", "b.axc", "--size", "1024"]);
        match cli.command {
            Command::RewriteVerify { tol, buffer_sizes, size, seed, workgroups, push_constants_base64, .. } => {
                assert_eq!(tol, "bit-exact", "default --tol must be bit-exact");
                assert!(buffer_sizes.is_empty());
                assert_eq!(size, Some(1024));
                assert_eq!(seed, 0, "default --seed must be 0");
                assert_eq!(workgroups, None);
                assert_eq!(push_constants_base64, None);
            }
            other => panic!("expected Command::RewriteVerify, got: {:?}", std::mem::discriminant(&other)),
        }
    }

    /// AT-2829: the `bench` subcommand's rendered clap help mentions `--filter`,
    /// `--bless`, and the word `baseline` — guards that help text stays useful
    /// (discoverability was EB.3's whole point).
    #[test]
    fn at_2829_bench_help_mentions_flags_and_baseline() {
        let mut cmd = Cli::command();
        let bench_cmd = cmd
            .find_subcommand_mut("bench")
            .expect("Cli must have a `bench` subcommand");
        let help: String = bench_cmd.render_long_help().to_string();
        assert!(help.contains("--filter"), "help must mention --filter; got:\n{help}");
        assert!(help.contains("--bless"), "help must mention --bless; got:\n{help}");
        assert!(
            help.to_lowercase().contains("baseline"),
            "help must mention 'baseline'; got:\n{help}"
        );
    }
}
