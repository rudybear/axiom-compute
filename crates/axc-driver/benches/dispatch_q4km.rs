//! Criterion bench binary for the `dispatch_gpu_q4km` group (M2.6).
//!
//! Two benches:
//!   - `dispatch_q4km_128`: n_superblocks=128 (128 × 144 = 18 KB weights,
//!     128 × 256 × 4 = 131072 f32 x-values = 512 KB input)
//!   - `dispatch_q4km_512`: n_superblocks=512 (512 × 144 = 72 KB weights,
//!     512 × 256 × 4 = 2 MB x-values input)
//!
//! # GPU availability gate
//!
//! Both benches are gated on `AXC_ENABLE_GPU_BENCHES=1` AND `probe_vulkan_available()`.
//! When either condition is false, a skip message is emitted and the bench runs
//! a single dummy iteration to keep Criterion's bench-name table consistent.
//!
//! # Env override
//!
//! `AXC_Q4KM_BENCH_MAX_SUPERBLOCKS`: caps n_superblocks for CI runners with small VRAM.
//! The cap is applied between 128 (minimum) and the intended bench size.
//! Lavapipe default is unmodified.
//!
//! # Pre-flight correctness (AT-705 continuity)
//!
//! Before each bench's measurement loop, one dispatch is performed and verified against
//! the CPU reference within 1e-3 relative tolerance.  Divergence panics.
//!
//! # VulkanContext lifetime
//!
//! One `GpuStateQ4KM` is constructed once via `OnceLock` and shared across all benches in
//! this binary.  This is a SEPARATE `OnceLock` from the one in `dispatch.rs` (dispatch_q4_0)
//! to preserve per-bench-binary Criterion state isolation.
//!
//! # BatchSize rationale
//!
//! `BatchSize::PerIteration` for both: the kernel produces one f32 output; the dominant
//! cost is GPU round-trip latency (CB record + queue submit + fence wait + 4-byte readback),
//! not data size.  PerIteration avoids Criterion pre-allocating batches of request builders
//! holding borrowed slices.

// Each bench binary includes common.rs independently via #[path]; clippy sees multiple
// binaries loading the same file and flags it.  This is the established pattern in the
// axc-driver bench suite (same as dispatch.rs, compile.rs, cpu_reference.rs).
#[allow(clippy::duplicate_mod)]
#[path = "common.rs"]
mod common;

// Re-use the CPU reference and fixture generator from the integration test helpers.
// Dead-code allow: common_helpers.rs has test-only helper functions unused in bench context.
#[allow(clippy::duplicate_mod, dead_code)]
#[path = "../tests/common_helpers.rs"]
mod helpers;

use axc_runtime::{VulkanContext, VulkanContextOptions, DispatchRequest, probe_vulkan_available};
use axc_hir::ParamBindingPlan;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use std::sync::OnceLock;

/// n_superblocks for the "small" bench.
const N_SUPERBLOCKS_SMALL: u32 = 128;

/// n_superblocks for the "large" bench.
const N_SUPERBLOCKS_LARGE: u32 = 512;

/// Deterministic fixture seed (different from dispatch.rs for bench-order independence).
const FIXTURE_SEED: u64 = 0xCAFE_BABE_u64;

/// Q4_K_M kernel source.
const Q4KM_SRC: &str = include_str!("../../../examples/q4km_dequant_matvec.axc");

// ── GPU state ─────────────────────────────────────────────────────────────────

/// Lazy-initialized GPU state for Q4_K_M dispatch benches.
/// Separate from dispatch.rs's GpuState to preserve per-binary Criterion isolation.
struct GpuStateQ4KM {
    ctx: VulkanContext,
    spirv: Vec<u32>,
    plan: ParamBindingPlan,
}

static GPU_STATE: OnceLock<GpuStateQ4KM> = OnceLock::new();

/// Returns `true` if GPU benches should run (env var + ICD probe).
fn gpu_benches_enabled() -> bool {
    std::env::var("AXC_ENABLE_GPU_BENCHES").as_deref() == Ok("1")
        && probe_vulkan_available()
}

/// Read `AXC_Q4KM_BENCH_MAX_SUPERBLOCKS` env var and cap `default` to [128, default].
///
/// Returns `default` if the env var is unset, unparseable, or out of [128, default].
fn cap_superblocks_from_env(default: u32) -> u32 {
    match std::env::var("AXC_Q4KM_BENCH_MAX_SUPERBLOCKS") {
        Ok(s) => match s.parse::<u32>() {
            Ok(n) if n >= 128 && n <= default => n,
            _ => default,
        },
        Err(_) => default,
    }
}

/// Initialize GPU state once.  Panics if Vulkan setup or kernel compilation fails.
fn gpu_state() -> &'static GpuStateQ4KM {
    GPU_STATE.get_or_init(|| {
        let ctx: VulkanContext = VulkanContext::new_with_options(VulkanContextOptions {
            pipeline_cache_path: None,
            physical_device_index: None,
            fence_timeout_ms: None,
        })
        .expect("dispatch_q4km bench: VulkanContext::new_with_options must succeed");

        eprintln!(
            "dispatch_gpu_q4km: using device '{}'",
            ctx.physical_device_name()
        );

        // Compile the Q4_K_M kernel.
        let (spirv_bytes, meta) =
            axc_driver::compile_source_with_meta(Q4KM_SRC)
                .expect("dispatch_q4km bench: q4km_dequant_matvec.axc compile failed");
        let spirv: Vec<u32> = common::bytes_to_words(&spirv_bytes);
        let plan: ParamBindingPlan = meta.binding_plan;

        GpuStateQ4KM { ctx, spirv, plan }
    })
}

// ── Push-constant assembly ─────────────────────────────────────────────────────

/// Assemble push-constant block for q4km_dequant_matvec (n_superblocks: u32).
fn assemble_q4km_push_constants(plan: &ParamBindingPlan, n_superblocks: u32) -> Vec<u8> {
    let mut pc: Vec<u8> = vec![0u8; plan.push_constant_total_bytes as usize];
    for scalar in &plan.scalars {
        let start: usize = scalar.offset as usize;
        if scalar.ty == axc_hir::ScalarTy::U32 {
            pc[start..start + 4].copy_from_slice(&n_superblocks.to_le_bytes());
        }
    }
    pc
}

// ── Pre-flight correctness oracle ─────────────────────────────────────────────

/// Dispatch the Q4_K_M kernel once and verify GPU output matches CPU reference
/// within 1e-3 relative tolerance.  Panics on divergence (AT-705 continuity).
fn pre_flight_q4km(state: &GpuStateQ4KM, n_superblocks: u32) {
    let n: usize = n_superblocks as usize;
    let (q_bytes, x_vec) = helpers::generate_q4km_test_data(n_superblocks, FIXTURE_SEED);
    let cpu_result: f32 = helpers::q4km_dequant_matvec_cpu(&q_bytes, &x_vec, n);

    let x_bytes: Vec<u8> = common::f32_slice_to_bytes(&x_vec);
    let y_init: Vec<u8> = vec![0u8; 4];
    let pc: Vec<u8> = assemble_q4km_push_constants(&state.plan, n_superblocks);

    let q_len: usize = n * helpers::Q4KM_SUPERBLOCK_BYTES;
    let x_len: usize = n * helpers::Q4KM_SUPERBLOCK_ELEMS * 4;

    let req: DispatchRequest = DispatchRequest {
        spirv: &state.spirv,
        binding_plan: &state.plan,
        workgroups: [1, 1, 1],
        inputs: &[q_bytes.as_slice(), x_bytes.as_slice(), y_init.as_slice()],
        output_sizes: &[q_len, x_len, 4],
        push_constants: &pc,
        entry_point: "q4km_dequant_matvec",
    };

    let outputs: Vec<Vec<u8>> = state
        .ctx
        .dispatch(req)
        .expect("pre_flight_q4km: dispatch must succeed");

    let y_bytes: &[u8] = &outputs[2];
    assert!(y_bytes.len() >= 4, "pre_flight_q4km: y output too short");
    let gpu_result: f32 = f32::from_le_bytes([y_bytes[0], y_bytes[1], y_bytes[2], y_bytes[3]]);

    let rel_err: f32 = if cpu_result.abs() > 1e-10_f32 {
        (gpu_result - cpu_result).abs() / cpu_result.abs()
    } else {
        (gpu_result - cpu_result).abs()
    };

    assert!(
        rel_err < 1e-3_f32,
        "pre_flight_q4km(n_superblocks={n_superblocks}): GPU={gpu_result:.6}, \
         CPU={cpu_result:.6}, rel_err={rel_err:.2e} >= 1e-3"
    );
}

// ── dispatch_q4km_128 bench ───────────────────────────────────────────────────

/// Bench `dispatch_q4km_128`: Q4_K_M dequant+matvec dispatch, n_superblocks=128.
///
/// 128 superblocks × 144 bytes = 18 KB q-weights; 128 × 256 × 4 = 512 KB x-vector.
/// BatchSize::PerIteration — single f32 output; overhead is dominated by GPU
/// round-trip latency (CB record + submit + fence wait + 4-byte readback).
fn bench_dispatch_q4km_128(c: &mut Criterion) {
    let n_superblocks: u32 = cap_superblocks_from_env(N_SUPERBLOCKS_SMALL);

    if !gpu_benches_enabled() {
        eprintln!(
            "skipping dispatch_q4km_128: AXC_ENABLE_GPU_BENCHES != 1 or no Vulkan ICD"
        );
        c.bench_function("dispatch_q4km_128", |b| {
            b.iter(|| {});
        });
        return;
    }

    let state: &GpuStateQ4KM = gpu_state();
    pre_flight_q4km(state, n_superblocks);

    let n: usize = n_superblocks as usize;
    let (q_bytes, x_vec) = helpers::generate_q4km_test_data(n_superblocks, FIXTURE_SEED);
    let x_bytes: Vec<u8> = common::f32_slice_to_bytes(&x_vec);
    let y_init: Vec<u8> = vec![0u8; 4];
    let pc: Vec<u8> = assemble_q4km_push_constants(&state.plan, n_superblocks);

    let q_len: usize = n * helpers::Q4KM_SUPERBLOCK_BYTES;
    let x_len: usize = n * helpers::Q4KM_SUPERBLOCK_ELEMS * 4;

    c.bench_function("dispatch_q4km_128", |b| {
        b.iter_batched(
            || {},
            |()| {
                let req: DispatchRequest = DispatchRequest {
                    spirv: &state.spirv,
                    binding_plan: &state.plan,
                    workgroups: [1, 1, 1],
                    inputs: &[q_bytes.as_slice(), x_bytes.as_slice(), y_init.as_slice()],
                    output_sizes: &[q_len, x_len, 4],
                    push_constants: &pc,
                    entry_point: "q4km_dequant_matvec",
                };
                let _outputs = state
                    .ctx
                    .dispatch(req)
                    .expect("dispatch_q4km_128: dispatch must succeed");
            },
            BatchSize::PerIteration,
        );
    });
}

// ── dispatch_q4km_512 bench ───────────────────────────────────────────────────

/// Bench `dispatch_q4km_512`: Q4_K_M dequant+matvec dispatch, n_superblocks=512.
///
/// 512 superblocks × 144 bytes = 72 KB q-weights; 512 × 256 × 4 = 2 MB x-vector.
/// BatchSize::PerIteration — single f32 output; overhead dominated by GPU round-trip.
/// Optional cap via AXC_Q4KM_BENCH_MAX_SUPERBLOCKS (must be ≥ 128).
fn bench_dispatch_q4km_512(c: &mut Criterion) {
    let n_superblocks: u32 = cap_superblocks_from_env(N_SUPERBLOCKS_LARGE);

    if !gpu_benches_enabled() {
        eprintln!(
            "skipping dispatch_q4km_512: AXC_ENABLE_GPU_BENCHES != 1 or no Vulkan ICD"
        );
        c.bench_function("dispatch_q4km_512", |b| {
            b.iter(|| {});
        });
        return;
    }

    let state: &GpuStateQ4KM = gpu_state();
    pre_flight_q4km(state, n_superblocks);

    let n: usize = n_superblocks as usize;
    let (q_bytes, x_vec) = helpers::generate_q4km_test_data(n_superblocks, FIXTURE_SEED);
    let x_bytes: Vec<u8> = common::f32_slice_to_bytes(&x_vec);
    let y_init: Vec<u8> = vec![0u8; 4];
    let pc: Vec<u8> = assemble_q4km_push_constants(&state.plan, n_superblocks);

    let q_len: usize = n * helpers::Q4KM_SUPERBLOCK_BYTES;
    let x_len: usize = n * helpers::Q4KM_SUPERBLOCK_ELEMS * 4;

    c.bench_function("dispatch_q4km_512", |b| {
        b.iter_batched(
            || {},
            |()| {
                let req: DispatchRequest = DispatchRequest {
                    spirv: &state.spirv,
                    binding_plan: &state.plan,
                    workgroups: [1, 1, 1],
                    inputs: &[q_bytes.as_slice(), x_bytes.as_slice(), y_init.as_slice()],
                    output_sizes: &[q_len, x_len, 4],
                    push_constants: &pc,
                    entry_point: "q4km_dequant_matvec",
                };
                let _outputs = state
                    .ctx
                    .dispatch(req)
                    .expect("dispatch_q4km_512: dispatch must succeed");
            },
            BatchSize::PerIteration,
        );
    });
}

criterion_group!(dispatch_gpu_q4km, bench_dispatch_q4km_128, bench_dispatch_q4km_512);
criterion_main!(dispatch_gpu_q4km);
