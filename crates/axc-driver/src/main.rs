//! `axc` binary entry point.
//!
//! Parses the CLI, dispatches to `compile_file` or a debug lex dump,
//! renders diagnostics via miette, and exits non-zero on error.

use clap::Parser as ClapParser;
use axc_driver::{Cli, Command, compile_file};

fn main() -> miette::Result<()> {
    let cli: Cli = Cli::parse();

    match cli.command {
        Command::Compile { input, output } => {
            compile_file(&input, &output).map_err(|e| {
                miette::miette!("{}", e)
            })
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
    }
}
