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

// ── Q4_0 dequantization CPU reference (M2.5) ──────────────────────────────────
//
// Q4_0 block layout (matches llama.cpp gguf format):
//   18 bytes per block, 32 f32 elements per block
//   byte  0..1:  f16 scale (little-endian IEEE 754 half-precision)
//   bytes 2..17: 16 packed nibble pairs (byte k holds nibble k*2 in low 4 bits,
//                nibble k*2+1 in high 4 bits)
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
        let x_base: usize = block_idx * Q4_0_BLOCK_ELEMS;
        for byte_k in 0..16_usize {
            let packed: u8 = q[block_byte_offset + 2 + byte_k];
            let lo_nibble: u8 = packed & 0x0F;
            let hi_nibble: u8 = (packed >> 4) & 0x0F;

            // Element 2*byte_k: low nibble.
            let w0: f32 = (lo_nibble as f32 - 8.0_f32) * scale;
            acc += w0 * x[x_base + 2 * byte_k];

            // Element 2*byte_k + 1: high nibble.
            let w1: f32 = (hi_nibble as f32 - 8.0_f32) * scale;
            acc += w1 * x[x_base + 2 * byte_k + 1];
        }
    }

    acc
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
