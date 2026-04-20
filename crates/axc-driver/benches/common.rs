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
