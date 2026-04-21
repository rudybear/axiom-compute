//! `axc` binary entry point.
//!
//! Parses the CLI, dispatches to `compile_file` or a debug lex dump,
//! renders diagnostics via miette, and exits non-zero on error.
//!
//! M2.4 adds `axc mcp [--log stderr|null]` — a JSON-RPC 2.0 stdio MCP server.

use clap::Parser as ClapParser;
use axc_driver::{Cli, Command, compile_file};
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
    }
}
