//! Command-line interface definition for `axc`.
//!
//! M0 ships two subcommands:
//! - `axc compile <input.axc> -o <output.spv>`: full pipeline to SPIR-V
//! - `axc lex <input.axc>`: debug token dump

use std::path::PathBuf;

/// Root CLI entry point.
#[derive(clap::Parser)]
#[command(name = "axc", about = "AXIOM-Compute compiler")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

/// Available subcommands.
#[derive(clap::Subcommand)]
pub enum Command {
    /// Compile an .axc source file to a SPIR-V binary.
    Compile {
        /// Input source file (`.axc`)
        input: PathBuf,
        /// Output SPIR-V file (`.spv`)
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Dump the lexed token stream (debug / diagnostic use).
    Lex {
        /// Input source file (`.axc`)
        input: PathBuf,
    },
}
