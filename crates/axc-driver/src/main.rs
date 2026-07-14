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

use clap::Parser as ClapParser;
use axc_driver::{Cli, Command, compile_file, build_bench_command};
use axc_driver::optimize::run_optimize;
use axc_driver::mcp::{run_mcp_server, LogTarget};

fn main() -> miette::Result<()> {
    let cli: Cli = Cli::parse();

    match cli.command {
        Command::Compile { input, output, strategy_values } => {
            if strategy_values.is_empty() {
                compile_file(&input, &output).map_err(|e| {
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
