//! Tool submodules for the MCP server.
//!
//! Each module implements one JSON-RPC tool:
//! - `load_source`        — parse + HIR-lower, return kernel metadata
//! - `enumerate_variants` — Cartesian-product enumeration of @strategy holes
//! - `compile_variant`    — compile to SPIR-V + validate + base64-encode
//! - `bench_variant`      — GPU dispatch + timing + correctness oracle
//! - `grid_search`        — full grid search + history append
//! - `optimization_history` — read JSONL history for a source file

pub mod load_source;
pub mod enumerate_variants;
pub mod compile_variant;
pub mod bench_variant;
pub mod grid_search;
pub mod optimization_history;
