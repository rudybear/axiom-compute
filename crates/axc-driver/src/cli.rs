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
}
