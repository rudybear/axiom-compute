//! Shared helpers for the axc-driver bench suite (M2.2).
//!
//! This module is re-included via `#[path = "common.rs"] mod common;` in each
//! bench file (compile.rs, cpu_reference.rs, dispatch.rs, postprocess.rs).
//! Unit tests for these helpers live in `crates/axc-driver/tests/common_helpers.rs`
//! (C9 resolution: avoids the #[cfg(test)] duplicate-registration fragility).
//!
//! # Determinism contract (AT-707)
//!
//! Every input-generation function takes an explicit `seed: u64`.  Callers
//! MUST pass `SEED` (42) and re-seed each bench function independently so that
//! bench-order does not affect the data.
//!
//! # Fault-injection (AT-714)
//!
//! When the `bench_regression_fixture_slowdown` feature is enabled (only in the
//! dedicated CI fault-injection job), `saxpy_cpu_reference` appends an
//! accumulator loop of 100_000_000 iterations — a guaranteed >15% slowdown
//! detectable by the regression gate.

// Each bench target (#[path = "common.rs"] mod common;) includes this module
// independently, so items used in one target appear "dead" to another's analysis.
#![allow(dead_code)]

use rand::SeedableRng;
use rand::Rng;
use rand::rngs::StdRng;
use axc_hir::ParamBindingPlan;
use axc_hir::ScalarTy;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Fixed seed for all deterministic bench inputs (AT-707).
pub const SEED: u64 = 42;

/// Fixed alpha for saxpy benches.
pub const ALPHA: f32 = 2.5_f32;

/// Absolute tolerance for GPU vs CPU correctness checks.
pub const ABS_TOL: f32 = 1e-6;

// ── Source text constants ──────────────────────────────────────────────────────

pub const SAXPY_SRC: &str = include_str!("../../../examples/saxpy.axc");
pub const VECTOR_ADD_SRC: &str = include_str!("../../../examples/vector_add.axc");

// ── Input generation (AT-707: deterministic StdRng) ───────────────────────────

/// Build saxpy inputs: (x, y, alpha) of length `n`.
///
/// Values drawn from `[-1.0, 1.0]` uniform to avoid denormals and overflow.
/// `alpha` is fixed at `ALPHA` (2.5) regardless of seed.
///
/// Re-seeding here (not at module level) ensures bench-order independence.
pub fn make_saxpy_inputs(n: usize, seed: u64) -> (Vec<f32>, Vec<f32>, f32) {
    let mut rng: StdRng = StdRng::seed_from_u64(seed);
    let x: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0_f32..=1.0_f32)).collect();
    let y: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0_f32..=1.0_f32)).collect();
    (x, y, ALPHA)
}

/// Build vector_add inputs: (a, b) of length `n`.
///
/// Values drawn from `[-1.0, 1.0]` uniform to avoid denormals and overflow.
pub fn make_vector_add_inputs(n: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut rng: StdRng = StdRng::seed_from_u64(seed);
    let a: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0_f32..=1.0_f32)).collect();
    let b: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0_f32..=1.0_f32)).collect();
    (a, b)
}

// ── Byte / word conversion helpers ────────────────────────────────────────────

/// Convert `&[f32]` to little-endian `Vec<u8>`.
pub fn f32_slice_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Parse little-endian `&[u8]` back to `Vec<f32>`.
///
/// Panics if `bytes.len()` is not a multiple of 4.
pub fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 4, 0, "output length must be 4-byte aligned");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Convert SPIR-V byte blob to word slice.
///
/// Panics if `bytes.len()` is not a multiple of 4.
pub fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    assert_eq!(bytes.len() % 4, 0, "SPIR-V length must be 4-byte aligned");
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ── CPU reference implementations ─────────────────────────────────────────────

/// CPU reference for saxpy: computes `alpha * x[i] + y[i]` for each element.
///
/// # Fault-injection (AT-714)
///
/// When `bench_regression_fixture_slowdown` is enabled, an additional
/// accumulator loop runs after the computation, artificially slowing this
/// function by ~10000x.  The loop uses `black_box` to prevent dead-code
/// elimination by future rustc/LLVM versions.
pub fn saxpy_cpu_reference(x: &[f32], y: &[f32], alpha: f32) -> Vec<f32> {
    assert_eq!(x.len(), y.len(), "saxpy: x and y must have equal length");
    let result: Vec<f32> = x.iter().zip(y.iter()).map(|(&xi, &yi)| alpha * xi + yi).collect();

    #[cfg(feature = "bench_regression_fixture_slowdown")]
    {
        // Accumulator-carrying loop that defeats theoretical DCE regressions.
        // The induction variable is carried through black_box so a future rustc
        // LLVM release cannot eliminate the loop even if it proves the result
        // is unused.  This is the C10-resolution pattern from the architect spec.
        let mut acc: u64 = 0;
        for i in 0..100_000_000_u64 {
            acc = acc.wrapping_add(std::hint::black_box(i));
        }
        std::hint::black_box(acc);
    }

    result
}

/// CPU reference for vector_add: computes `a[i] + b[i]` for each element.
pub fn vector_add_cpu_reference(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "vector_add: a and b must have equal length");
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai + bi).collect()
}

// ── Q4_0 dequantization CPU reference (M2.5) ──────────────────────────────────
//
// Q4_0 block layout (matches llama.cpp gguf format):
//   18 bytes per block, 32 f32 elements per block
//   byte  0..1:  f16 scale (little-endian IEEE 754 half-precision)
//   bytes 2..17: 16 packed nibble pairs
//                byte k (k in 0..16):
//                  low  4 bits = weight at index k
//                  high 4 bits = weight at index k + 16
//                (This is the canonical llama.cpp / GGUF Q4_0 convention, NOT
//                the interleaved `[2*k, 2*k+1]` layout. See DESIGN.md §3.1.8
//                and examples/q4_0_dequant_matvec.axc.)
//
// Dequant formula: value_i = (nibble_i - 8) * f16_to_f32(scale)
// where nibble is unsigned 0..15 and the offset 8 centers it at zero.
//
// This is the canonical CPU reference for Q4_0 dequant+matvec used in
// integration tests and benchmarks (AT-901..AT-918, acceptance criterion 7).

/// Bytes per Q4_0 block.
pub const Q4_0_BLOCK_BYTES: usize = 18;

/// Elements per Q4_0 block.
pub const Q4_0_BLOCK_ELEMS: usize = 32;

/// Build random Q4_0 quantized weight data and a matching f32 x-vector.
///
/// Returns `(q_bytes, x_vec)` where:
///   - `q_bytes` is `n_blocks * Q4_0_BLOCK_BYTES` bytes of synthetic Q4_0 data
///   - `x_vec` is `n_blocks * Q4_0_BLOCK_ELEMS` f32 values in `[-1.0, 1.0]`
///
/// The f16 scales are chosen as small positive f32 values that round-trip cleanly
/// through f16 (values in `[0.1, 1.0]` rounded to nearest f16).
/// The nibble bytes are random `u8` values (0x00..0xFF packed nibble pairs).
pub fn make_q4_0_inputs(n_blocks: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
    let mut rng: StdRng = StdRng::seed_from_u64(seed);
    let mut q_bytes: Vec<u8> = Vec::with_capacity(n_blocks * Q4_0_BLOCK_BYTES);
    let mut x_vec: Vec<f32> = Vec::with_capacity(n_blocks * Q4_0_BLOCK_ELEMS);

    for _ in 0..n_blocks {
        // Scale: random f32 in (0.1, 1.0) → round to f16 → store as 2 LE bytes.
        let scale_f32: f32 = rng.gen_range(0.1_f32..1.0_f32);
        let scale_f16 = half::f16::from_f32(scale_f32);
        let scale_bits: u16 = scale_f16.to_bits();
        q_bytes.push((scale_bits & 0xFF) as u8);
        q_bytes.push((scale_bits >> 8) as u8);

        // 16 packed nibble bytes.
        for _ in 0..16 {
            q_bytes.push(rng.gen::<u8>());
        }

        // 32 x-values for this block.
        for _ in 0..Q4_0_BLOCK_ELEMS {
            x_vec.push(rng.gen_range(-1.0_f32..=1.0_f32));
        }
    }

    (q_bytes, x_vec)
}

/// CPU reference for Q4_0 dequantize + matrix-vector multiply.
///
/// Computes the dot product of the single-invocation kernel:
///   y = sum over all blocks and elements of dequant(weight_k) * x[k]
///
/// where `dequant(nibble) = (nibble - 8) * scale`.
///
/// # Parameters
/// - `q`: raw Q4_0 bytes (must be `n_blocks * 18` bytes long)
/// - `x`: input f32 vector (must be `n_blocks * 32` f32 values long)
/// - `n_blocks`: number of Q4_0 blocks
///
/// Returns a single `f32` accumulator (the scalar output of the matvec).
///
/// # Panics
/// Panics if `q.len() != n_blocks * 18` or `x.len() != n_blocks * 32`.
pub fn q4_0_dequant_matvec_cpu(q: &[u8], x: &[f32], n_blocks: usize) -> f32 {
    assert_eq!(
        q.len(), n_blocks * Q4_0_BLOCK_BYTES,
        "q4_0_dequant_matvec_cpu: q length mismatch: expected {}, got {}",
        n_blocks * Q4_0_BLOCK_BYTES, q.len()
    );
    assert_eq!(
        x.len(), n_blocks * Q4_0_BLOCK_ELEMS,
        "q4_0_dequant_matvec_cpu: x length mismatch: expected {}, got {}",
        n_blocks * Q4_0_BLOCK_ELEMS, x.len()
    );

    let mut acc: f32 = 0.0_f32;

    for block_idx in 0..n_blocks {
        let block_byte_offset: usize = block_idx * Q4_0_BLOCK_BYTES;

        // Decode f16 scale from the first 2 bytes (little-endian).
        let scale_lo: u8 = q[block_byte_offset];
        let scale_hi: u8 = q[block_byte_offset + 1];
        let scale_bits: u16 = (scale_lo as u16) | ((scale_hi as u16) << 8);
        let scale: f32 = half::f16::from_bits(scale_bits).to_f32();

        // 16 packed nibble bytes → 32 nibble values.
        //
        // GGUF Q4_0 layout: for k in 0..16, byte k encodes two weights:
        //   lo nibble = weight at index k
        //   hi nibble = weight at index k + 16
        //
        // This matches `examples/q4_0_dequant_matvec.axc` and DESIGN.md §3.1.8.
        let x_base: usize = block_idx * Q4_0_BLOCK_ELEMS;
        for byte_k in 0..16_usize {
            let packed: u8 = q[block_byte_offset + 2 + byte_k];
            let lo_nibble: u8 = packed & 0x0F;
            let hi_nibble: u8 = (packed >> 4) & 0x0F;

            // Element byte_k: low nibble.
            let w_lo: f32 = (lo_nibble as f32 - 8.0_f32) * scale;
            acc += w_lo * x[x_base + byte_k];

            // Element byte_k + 16: high nibble.
            let w_hi: f32 = (hi_nibble as f32 - 8.0_f32) * scale;
            acc += w_hi * x[x_base + byte_k + 16];
        }
    }

    acc
}

// ── M3.1: F16 byte helpers ────────────────────────────────────────────────────

/// Convert a slice of f16 bit patterns (u16) to a byte Vec (little-endian per element).
///
/// Used by the coopmat dispatch tests to prepare f16 SSBO inputs.
pub fn f16_slice_to_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|&b| b.to_le_bytes()).collect()
}

/// Convert a byte slice to a Vec<u16> of f16 bit patterns (little-endian).
///
/// Panics if `bytes.len()` is not a multiple of 2.
pub fn bytes_to_f16_vec(bytes: &[u8]) -> Vec<u16> {
    assert_eq!(bytes.len() % 2, 0, "f16 output must be 2-byte aligned");
    bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

// ── Push-constant assembly (AT-514a discipline from M1.5) ─────────────────────
//
// Callers MUST iterate `plan.scalars` in stored order and dispatch on scalar.ty.
// Never hardcode layout — future milestones may add alignment padding or reorder scalars.

/// Assemble push-constant bytes for saxpy: writes `n` and `alpha` at the
/// offsets dictated by `plan.scalars` (AT-514a discipline).
///
/// Returns a `Vec<u8>` of length `plan.push_constant_total_bytes`.
pub fn assemble_saxpy_push_constants(plan: &ParamBindingPlan, n: u32, alpha: f32) -> Vec<u8> {
    let mut pc: Vec<u8> = vec![0u8; plan.push_constant_total_bytes as usize];
    for scalar in &plan.scalars {
        let start: usize = scalar.offset as usize;
        match scalar.ty {
            ScalarTy::U32 => {
                pc[start..start + 4].copy_from_slice(&n.to_le_bytes());
            }
            ScalarTy::F32 => {
                pc[start..start + 4].copy_from_slice(&alpha.to_le_bytes());
            }
            ScalarTy::I32 => {
                pc[start..start + 4].copy_from_slice(&(n as i32).to_le_bytes());
            }
            ScalarTy::U64 => {
                pc[start..start + 8].copy_from_slice(&(n as u64).to_le_bytes());
            }
            ScalarTy::I64 => {
                pc[start..start + 8].copy_from_slice(&(n as i64).to_le_bytes());
            }
            ScalarTy::F64 => {
                pc[start..start + 8].copy_from_slice(&(alpha as f64).to_le_bytes());
            }
            _ => {
                // Other types not used in saxpy; zero-fill is safe for test stability.
            }
        }
    }
    pc
}

/// Assemble push-constant bytes for vector_add: writes `n` at the offset
/// dictated by `plan.scalars` (AT-514a discipline).
///
/// Returns a `Vec<u8>` of length `plan.push_constant_total_bytes`.
pub fn assemble_vector_add_push_constants(plan: &ParamBindingPlan, n: u32) -> Vec<u8> {
    let mut pc: Vec<u8> = vec![0u8; plan.push_constant_total_bytes as usize];
    for scalar in &plan.scalars {
        let start: usize = scalar.offset as usize;
        match scalar.ty {
            ScalarTy::U32 => {
                pc[start..start + 4].copy_from_slice(&n.to_le_bytes());
            }
            ScalarTy::I32 => {
                pc[start..start + 4].copy_from_slice(&(n as i32).to_le_bytes());
            }
            ScalarTy::U64 => {
                pc[start..start + 8].copy_from_slice(&(n as u64).to_le_bytes());
            }
            ScalarTy::I64 => {
                pc[start..start + 8].copy_from_slice(&(n as i64).to_le_bytes());
            }
            _ => {
                // Other types not used in vector_add; zero-fill.
            }
        }
    }
    pc
}

// ── Platform probe (AT-709, C8 resolution) ────────────────────────────────────

/// Probe the CPU model string from the OS.
///
/// - Linux: reads the first `model name` line from `/proc/cpuinfo`.
/// - macOS: invokes `sysctl machdep.cpu.brand_string` via `std::process::Command`.
/// - Windows / other: returns an empty string (AT-709b explicit limitation for M2.2).
///
/// Always returns a `String` (possibly empty); never panics.
pub fn cpu_model_probe() -> String {
    cpu_model_probe_impl()
}

#[cfg(target_os = "linux")]
fn cpu_model_probe_impl() -> String {
    // Read /proc/cpuinfo and extract the first "model name" line.
    let contents: String = match std::fs::read_to_string("/proc/cpuinfo") {
        Ok(s) => s,
        Err(_) => return String::new(),
    };
    for line in contents.lines() {
        if let Some(rest) = line.strip_prefix("model name") {
            // Format: "model name\t: Intel(R) Core(TM) ..."
            if let Some(value) = rest.strip_prefix('\t').or_else(|| rest.strip_prefix(' ')) {
                if let Some(value) = value.strip_prefix(':') {
                    return value.trim().to_owned();
                }
            }
        }
    }
    String::new()
}

#[cfg(target_os = "macos")]
fn cpu_model_probe_impl() -> String {
    // macOS: `sysctl -n machdep.cpu.brand_string` returns the brand string.
    let output = match std::process::Command::new("sysctl")
        .arg("-n")
        .arg("machdep.cpu.brand_string")
        .output()
    {
        Ok(o) => o,
        Err(_) => return String::new(),
    };
    if output.status.success() {
        return String::from_utf8_lossy(&output.stdout).trim().to_owned();
    }
    String::new()
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn cpu_model_probe_impl() -> String {
    // Windows and other platforms: empty string (AT-709b).
    String::new()
}

// ════════════════════════════════════════════════════════════════════════════
// EB.2 — Per-machine bench-regression baselines (schema v2).
//
// This is the SINGLE source of truth for the machine key, the v2 schema, and
// the bless/gate decision logic.  Both the writer (postprocess.rs) and the gate
// (bench_regression.rs, via `#[path]`-include) call THESE functions so the key
// derivation and struct layout can never drift between writer and gate.
//
// Pure-fn unit tests (AT-EB2-01..04/08/10/11) live in
// `crates/axc-driver/tests/common_helpers.rs` (the C9 convention, common.rs:5),
// NOT as `#[cfg(test)]` here — common.rs is `#[path]`-included by every bench
// target + the gate and a `#[cfg(test)]` block would compile into each.
// ════════════════════════════════════════════════════════════════════════════

/// Maximum length (chars) of a sanitized machine-key token.
pub const MACHINE_KEY_MAX_LEN: usize = 80;

/// Sanitize an arbitrary device/CPU string into a JSON-key / filesystem-safe
/// token: lowercase ASCII; every char NOT in `[a-z0-9]` → `_`; collapse runs of
/// `_`; trim leading/trailing `_`; truncate to [`MACHINE_KEY_MAX_LEN`] chars.
///
/// Total and deterministic (never panics).  May return an empty string (the
/// caller — [`machine_key`] — supplies the non-empty fallback).
///
/// # Examples
/// ```ignore
/// assert_eq!(sanitize("NVIDIA RTX PRO 6000 Blackwell Workstation Edition"),
///            "nvidia_rtx_pro_6000_blackwell_workstation_edition");
/// assert_eq!(sanitize("Intel(R) Core(TM) i9-14900KF"), "intel_r_core_tm_i9_14900kf");
/// ```
pub fn sanitize(s: &str) -> String {
    let mut out: String = String::with_capacity(s.len());
    let mut last_was_underscore: bool = false;
    for ch in s.chars() {
        // Lowercase ASCII letters and digits pass through; everything else → '_'.
        let mapped: char = if ch.is_ascii_alphanumeric() {
            ch.to_ascii_lowercase()
        } else {
            '_'
        };
        if mapped == '_' {
            // Collapse runs of '_'.
            if !last_was_underscore {
                out.push('_');
                last_was_underscore = true;
            }
        } else {
            out.push(mapped);
            last_was_underscore = false;
        }
    }
    // Trim leading/trailing '_'.
    let trimmed: &str = out.trim_matches('_');
    // Truncate to MACHINE_KEY_MAX_LEN chars (char-boundary safe — all bytes are
    // ASCII here, but use char_indices for total safety), then re-trim a
    // trailing '_' the truncation may have exposed.
    let capped: String = trimmed.chars().take(MACHINE_KEY_MAX_LEN).collect();
    capped.trim_matches('_').to_owned()
}

/// Derive the per-machine baseline key from the Vulkan device name and the host
/// CPU model.
///
/// EB.2 decision: the key is the sanitized **device name only** (the `cpu_model`
/// is recorded in [`MachineMeta`] for provenance and is promotable into the key
/// later if a same-GPU/different-CPU collision ever mis-gates the `cpu_reference`
/// benches).
///
/// Empty device (Lavapipe-less / no-Vulkan CI) → the documented fallback:
/// `cpu_only__<sanitized cpu>`; if the CPU is also empty → `unknown_machine`.
///
/// # Examples
/// ```ignore
/// assert_eq!(machine_key("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "whatever"),
///            "nvidia_rtx_pro_6000_blackwell_workstation_edition");
/// assert_eq!(machine_key("", "Intel(R) Core(TM) i9-14900KF"), "cpu_only__intel_r_core_tm_i9_14900kf");
/// assert_eq!(machine_key("", ""), "unknown_machine");
/// ```
pub fn machine_key(device_name: &str, cpu_model: &str) -> String {
    let device_key: String = sanitize(device_name);
    if !device_key.is_empty() {
        return device_key;
    }
    // Empty device → fall back to a CPU-only key so a no-GPU host has a
    // deterministic key that never collides with a real GPU machine.
    let cpu_key: String = sanitize(cpu_model);
    if cpu_key.is_empty() {
        "unknown_machine".to_owned()
    } else {
        format!("cpu_only__{cpu_key}")
    }
}

// ── Schema v2 structs (writer + gate share ONE definition) ─────────────────────

/// Machine metadata sub-object (AT-709).  Unchanged from schema v1.
///
/// All fields are populated; empty string for unavailable probes (never null).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MachineMeta {
    pub os: String,
    pub rustc: String,
    pub vulkan_icd: String,
    pub vulkan_device: String,
    pub cpu_model: String,
    pub axc_version: String,
}

/// One bench entry in a machine block's `benchmarks` array.  Unchanged from v1.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct BenchEntry {
    pub group: String,
    pub bench: String,
    pub median_ns: u64,
    pub low_ns: u64,
    pub high_ns: u64,
}

/// A single machine's baseline block — the old schema-v1 body minus the (now
/// top-level) `schema_version`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MachineBlock {
    pub generated: String,
    pub git_sha: String,
    pub machine: MachineMeta,
    pub benchmarks: Vec<BenchEntry>,
}

/// The full baselines.json document (schema v2): one nested file keyed by the
/// per-machine [`machine_key`].  `BTreeMap` gives a deterministic, reviewable
/// key order in the committed diff.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BaselinesV2 {
    pub schema_version: u32,
    pub machines: BTreeMap<String, MachineBlock>,
}

/// The legacy schema-v1 document shape — parsed ONLY by the bless-side upgrade
/// path (`merge_blessed`) so an operator with a stale v1 file is auto-migrated
/// rather than crashing.  The gate NEVER accepts v1 (it is a hard error there).
#[derive(Debug, Deserialize)]
struct BaselinesV1 {
    schema_version: u32,
    generated: String,
    git_sha: String,
    machine: MachineMeta,
    benchmarks: Vec<BenchEntry>,
}

/// Error returned by [`merge_blessed`] when it refuses to bless.
#[derive(Debug)]
pub enum MergeError {
    /// The existing baselines file is present but cannot be parsed as either
    /// schema v2 or schema v1 — corrupt/truncated.  The bless MUST fail loudly
    /// and MUST NOT start-from-empty (which would drop every other machine's
    /// block on the next write — the exact clobber EB.2 prevents).
    CorruptExistingFile { detail: String },
}

impl std::fmt::Display for MergeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MergeError::CorruptExistingFile { detail } => write!(
                f,
                "refusing to bless over a corrupt/unparseable baselines file \
                 (neither schema v2 nor v1): {detail}. Fix or remove the file before blessing; \
                 NEVER auto-restart-from-empty (that would drop every other machine's baseline)."
            ),
        }
    }
}

impl std::error::Error for MergeError {}

/// Pure read-modify-write merge for the bless path (BLOCKER 1).
///
/// - `existing_json == None` (file ABSENT): start from an empty machines map —
///   the first bless on a fresh repo.  This is the ONLY case that starts empty.
/// - `Some(parseable v2)`: insert-or-replace ONLY `map[current_key]`; every
///   other machine's block is preserved byte-for-byte (modulo deterministic
///   BTreeMap re-ordering).
/// - `Some(parseable v1)`: auto-upgrade — re-key the single v1 block under
///   `machine_key(its device, its cpu)`, then insert-or-replace `current_key`.
/// - `Some(UNPARSEABLE)`: `Err(MergeError::CorruptExistingFile)` — fail loud,
///   NEVER an empty map (BLOCKER 1).
pub fn merge_blessed(
    existing_json: Option<&str>,
    current_key: &str,
    current_block: &MachineBlock,
) -> Result<BaselinesV2, MergeError> {
    let mut machines: BTreeMap<String, MachineBlock> = match existing_json {
        None => BTreeMap::new(),
        Some(raw) => {
            // Try v2 first (the live schema), then v1 (auto-upgrade), then fail loud.
            if let Ok(v2) = serde_json::from_str::<BaselinesV2>(raw) {
                v2.machines
            } else if let Ok(v1) = serde_json::from_str::<BaselinesV1>(raw) {
                if v1.schema_version != 1 {
                    return Err(MergeError::CorruptExistingFile {
                        detail: format!(
                            "v1-shaped file with schema_version={} (expected 1)",
                            v1.schema_version
                        ),
                    });
                }
                let mut upgraded: BTreeMap<String, MachineBlock> = BTreeMap::new();
                let v1_key: String =
                    machine_key(&v1.machine.vulkan_device, &v1.machine.cpu_model);
                upgraded.insert(
                    v1_key,
                    MachineBlock {
                        generated: v1.generated,
                        git_sha: v1.git_sha,
                        machine: v1.machine,
                        benchmarks: v1.benchmarks,
                    },
                );
                upgraded
            } else {
                return Err(MergeError::CorruptExistingFile {
                    detail: "serde failed to parse the file as schema v2 or v1".to_owned(),
                });
            }
        }
    };

    // Insert-or-replace ONLY the current machine's key; all others untouched.
    machines.insert(current_key.to_owned(), current_block.clone());

    Ok(BaselinesV2 {
        schema_version: 2,
        machines,
    })
}

/// The gate's decision for which baseline block to compare the live run against
/// (BLOCKER 2 two-case skip + the no-Vulkan single-machine fallback).
#[derive(Debug)]
pub enum SelectOutcome<'a> {
    /// The current machine's block was found — run the 15% gate against it.
    Gate(&'a MachineBlock),
    /// Genuinely-empty baselines (no machine ever blessed) → QUIET skip.
    QuietSkipEmpty,
    /// A real device was probed AND the file is non-empty AND the current key is
    /// absent → LOUD skip: log the current key + sorted known keys (surfaces a
    /// driver-rename key-drift on an EXISTING machine).
    LoudSkipKeyAbsent { current: String, known: Vec<String> },
    /// No Vulkan device probed AND exactly one machine exists → fall back to it
    /// (preserves today's Lavapipe-less CI gate).
    SingleMachineFallback(&'a MachineBlock),
    /// No Vulkan device probed AND >1 machine exists → cannot disambiguate → skip.
    SkipAmbiguous,
}

/// Pure block-selection for the gate (BLOCKER 2).  `device_probed` is `true`
/// when the live Vulkan device name was non-empty (a real GPU was queried).
pub fn select_block<'a>(
    machines: &'a BTreeMap<String, MachineBlock>,
    current_key: &str,
    device_probed: bool,
) -> SelectOutcome<'a> {
    if let Some(block) = machines.get(current_key) {
        return SelectOutcome::Gate(block);
    }

    if !device_probed {
        // No Vulkan device in the gate env (Lavapipe-less CI).
        return match machines.len() {
            0 => SelectOutcome::QuietSkipEmpty,
            1 => {
                // Safe: len == 1.
                let block: &MachineBlock = machines.values().next().expect("len==1");
                SelectOutcome::SingleMachineFallback(block)
            }
            _ => SelectOutcome::SkipAmbiguous,
        };
    }

    // A real device WAS probed but its key is absent.
    if machines.is_empty() {
        // No machine ever blessed → quiet skip (a new repo, not a key-drift).
        SelectOutcome::QuietSkipEmpty
    } else {
        // Non-empty file + probed device + key absent → LOUD: surface possible
        // driver-rename key-drift on an EXISTING machine.
        let known: Vec<String> = machines.keys().cloned().collect();
        SelectOutcome::LoudSkipKeyAbsent {
            current: current_key.to_owned(),
            known,
        }
    }
}

/// Atomically write `bytes` to `dir/final_name` (BLOCKER 1).
///
/// Serializes to a TEMP file in the SAME directory (`<final_name>.tmp.<pid>`),
/// `write_all` + `sync_all`, then `std::fs::rename` over the final path.  Rename
/// is atomic on POSIX within one filesystem, so a crash mid-write leaves the OLD
/// file fully intact — only the temp file is ever a casualty.  The same-dir
/// requirement is load-bearing (a cross-filesystem rename is NOT atomic).
///
/// On any failure before the rename, the temp file is best-effort removed and
/// the original `dir/final_name` is left untouched.
pub fn atomic_write(dir: &std::path::Path, final_name: &str, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write as _;
    let final_path: std::path::PathBuf = dir.join(final_name);
    let tmp_name: String = format!("{final_name}.tmp.{}", std::process::id());
    let tmp_path: std::path::PathBuf = dir.join(&tmp_name);

    // Scope the file handle so it is closed before the rename.
    let write_result: std::io::Result<()> = (|| {
        let mut f = std::fs::File::create(&tmp_path)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        Ok(())
    })();

    if let Err(e) = write_result {
        // Best-effort cleanup of the partial temp file; the original is intact.
        let _ = std::fs::remove_file(&tmp_path);
        return Err(e);
    }

    // Atomic replace.  If the rename fails, the original is still intact and the
    // temp file is cleaned up.
    if let Err(e) = std::fs::rename(&tmp_path, &final_path) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(e);
    }

    Ok(())
}
