//! M3.4 — AXIOM-side A/B measurement bench for the llama.cpp Vulkan Q4_K_M
//! head-to-head (the DESIGN §5 kill-criterion run).
//!
//! Bench ID: `dispatch_q4km_ab`.
//!
//! ## What this measures
//!
//! AXIOM's FROZEN M2.6 single-row Q4_K_M dequant+matvec (examples/q4km_dequant_matvec.axc)
//! measured via the resident path (upload-once → warmup → N timed dispatches), with REAL
//! 144-byte Q4_K_M superblocks (helpers::generate_q4km_test_data) so the dequant ALU runs.
//!
//! ## Measurement-boundary comparability (CRITICAL-1)
//!
//! STEP 0 (M3.4-coder) read vendor/llama.cpp/tests/test-backend-ops.cpp at the pinned SHA
//! (b9542 / 6b80c74f285390368b3c99c5e750f19e9b096e98). Its `perf` path:
//!   - brackets `ggml_backend_graph_compute` with CPU wall-clock (`ggml_time_us()`),
//!   - over a graph that DUPLICATES the op `n_runs` times (target ~100 GFLOP of work),
//!   - loops the whole thing until `total_time_us >= 1s` (sustained),
//!   - reports `avg_time_us = total_time_us / total_runs` (per-op MEAN, amortized),
//!   - discards ONE warmup `ggml_backend_graph_compute`, subtracts NO overhead.
//!
//! So llama.cpp is CPU-WALL, BATCHED-AMORTIZED, MEAN, sustained. To be comparable, the
//! AXIOM side reports BOTH:
//!   1. on-device GpuTimestamp MIN / MEAN / MEDIAN of N timed dispatches (the cleanest
//!      kernel-only number — covers the min-vs-mean axis), AND
//!   2. a SUSTAINED/BATCHED CPU-wall number: N_SUSTAINED back-to-back dispatches whose
//!      loop structure mirrors llama.cpp's amortized-over-N timing.
//!
//! The shell driver (scripts/m34_llamacpp_ab.sh) picks the LIKE-FOR-LIKE pair for the
//! headline ratio: AXIOM-sustained-CPU-wall vs llama.cpp-sustained-CPU-wall (the matched
//! boundary), and records the GpuTimestamp framing alongside for disclosure.
//!
//! ## Fairness caveat (the apples-to-oranges note)
//!
//! AXIOM produces ONE output per dispatch (the `if i >= 1 return` guard → ONE workgroup,
//! ~1 of ~188 SMs). llama.cpp's MUL_MAT Q4_K n==1 case computes m=4096 output rows over
//! k=14336 contraction, tiled across all SMs. The contraction K is matched (14336); the
//! output-row count is NOT (AXIOM 1 vs llama 4096). This is the documented structural gap.
//!
//! ## Gating
//!
//! Gated on `AXC_ENABLE_GPU_BENCHES=1` + a responsive Vulkan ICD. Typed-skip on
//! DeviceFeatureUnsupported. CpuFenceWall (Lavapipe) marks the run NON-GPU-KERNEL-TIME
//! (not the kill-criterion machine).

// Each bench binary includes common.rs independently via #[path]; clippy flags the repeat.
#[allow(clippy::duplicate_mod)]
#[path = "common.rs"]
mod common;

// Re-use the CPU reference + REAL 144-byte fixture generator from the integration helpers.
#[allow(clippy::duplicate_mod, dead_code)]
#[path = "../tests/common_helpers.rs"]
mod helpers;

use axc_driver::compile_source_with_meta;
use axc_runtime::{
    DispatchError, KernelHandle, ResidentBenchConfig, ResidentTimingSource, VulkanContext,
};
use criterion::{criterion_group, criterion_main, Criterion};
use std::time::Instant;

/// Q4_K_M single-row dequant+matvec kernel source (FROZEN M2.6).
const Q4KM_MATVEC_SRC: &str = include_str!("../../../examples/q4km_dequant_matvec.axc");

/// Deterministic fixture seed (distinct from dispatch_q4km.rs for bench-order independence).
const FIXTURE_SEED: u64 = 0x5A11_00AB_u64;

/// Number of superblocks → K = N_SUPERBLOCKS * 256 = 14336 contraction elements,
/// MATCHING llama.cpp's Q4_K MUL_MAT perf case (m=4096, n=1, k=14336): 14336/256 = 56.
const N_SUPERBLOCKS: u32 = 56;

/// Contraction length K (must equal N_SUPERBLOCKS * 256).
const K_CONTRACTION: usize = N_SUPERBLOCKS as usize * 256;

/// Warmup dispatches discarded before timing (TIMING-1 / mirrors resident_matmul.rs).
const N_WARMUP: usize = 2;
/// Timed dispatches for MIN / MEAN / MEDIAN.
const N_MEASURED: usize = 10;
/// Back-to-back dispatches for the SUSTAINED/BATCHED CPU-wall number (mirrors llama.cpp's
/// amortized-over-N loop). 200 ≈ enough to amortize per-dispatch host overhead.
const N_SUSTAINED: usize = 200;

/// 50 GB executable-memory cap (anti-pattern #1 / HANDOFF-17). The matvec fixture is tiny
/// (56 × 144 = 8 KB weights + 56 × 256 × 4 = 56 KB x), far under the cap; asserted anyway.
const FIFTY_GB: u64 = 50 * 1024 * 1024 * 1024;

fn gpu_benches_enabled() -> bool {
    std::env::var("AXC_ENABLE_GPU_BENCHES").as_deref() == Ok("1")
}

/// Format timing_source for printing (CpuFenceWall must carry the qualifier).
fn timing_source_label(ts: ResidentTimingSource) -> &'static str {
    match ts {
        ResidentTimingSource::GpuTimestamp => "GpuTimestamp",
        ResidentTimingSource::CpuFenceWall => {
            "CpuFenceWall — scheduling-inclusive, NOT a GPU kernel time"
        }
    }
}

/// Assemble the push-constant block for q4km_dequant_matvec (n_superblocks: u32).
fn assemble_push_constants(plan: &axc_hir::ParamBindingPlan, n_superblocks: u32) -> Vec<u8> {
    let mut pc: Vec<u8> = vec![0u8; plan.push_constant_total_bytes as usize];
    for scalar in &plan.scalars {
        if scalar.ty == axc_hir::ScalarTy::U32 {
            let start: usize = scalar.offset as usize;
            pc[start..start + 4].copy_from_slice(&n_superblocks.to_le_bytes());
        }
    }
    pc
}

/// Result of the AXIOM-side measurement.
struct AbMeasurement {
    kernel_ns_min: u64,
    kernel_ns_mean: u64,
    kernel_ns_median: u64,
    sustained_ns: u64,
    timing_source: ResidentTimingSource,
}

/// Measure the Q4_K_M matvec kernel via the resident path.
///
/// Returns MIN/MEAN/MEDIAN of N_MEASURED GpuTimestamp dispatches AND a sustained
/// CPU-wall per-dispatch number (N_SUSTAINED back-to-back dispatches / N_SUSTAINED).
fn axiom_q4km_kernel_measurement(
    ctx: &VulkanContext,
    handle: &KernelHandle,
    inputs: &[&[u8]],
    output_sizes: &[u64],
    push_constants: Vec<u8>,
) -> AbMeasurement {
    let resident = ctx
        .upload_resident(handle, inputs, output_sizes)
        .expect("upload_resident must succeed");
    let cfg = ResidentBenchConfig {
        workgroups: (1, 1, 1),
        push_constants,
    };

    // Warmup (discarded).
    for _ in 0..N_WARMUP {
        ctx.dispatch_resident(handle, &resident, &cfg)
            .expect("warmup dispatch must succeed");
    }

    // Timed loop: collect N_MEASURED on-device GpuTimestamp samples.
    let mut samples: Vec<u64> = Vec::with_capacity(N_MEASURED);
    let mut last_source = ResidentTimingSource::CpuFenceWall;
    for _ in 0..N_MEASURED {
        let t = ctx
            .dispatch_resident(handle, &resident, &cfg)
            .expect("measured dispatch must succeed");
        samples.push(t.kernel_ns);
        last_source = t.timing_source;
    }
    samples.sort_unstable();
    let kernel_ns_min = samples[0];
    let kernel_ns_median = samples[N_MEASURED / 2];
    let kernel_ns_mean = (samples.iter().sum::<u64>()) / (N_MEASURED as u64);

    // Sustained/batched CPU-wall number: N_SUSTAINED back-to-back dispatches bracketed by
    // CPU wall-clock, / N_SUSTAINED. Mirrors llama.cpp's amortized-over-N perf loop. Each
    // dispatch_resident already fence-waits internally, so the wall time captures the
    // sustained per-dispatch cost (the AXIOM analogue of llama.cpp's CPU-wall/total_runs).
    let sustained_start = Instant::now();
    for _ in 0..N_SUSTAINED {
        ctx.dispatch_resident(handle, &resident, &cfg)
            .expect("sustained dispatch must succeed");
    }
    let sustained_elapsed = sustained_start.elapsed();
    let sustained_ns = (sustained_elapsed.as_nanos() / (N_SUSTAINED as u128)) as u64;

    AbMeasurement {
        kernel_ns_min,
        kernel_ns_mean,
        kernel_ns_median,
        sustained_ns,
        timing_source: last_source,
    }
}

/// Pre-flight correctness oracle (AT-1761): dispatch once with real fixture data and
/// assert rel_err < 1e-3 vs the ggml CPU reference BEFORE timing. A wrong-but-fast kernel
/// must not pollute the A/B.
fn pre_flight_bit_exact(ctx: &VulkanContext, handle: &KernelHandle, plan: &axc_hir::ParamBindingPlan) {
    let n: usize = N_SUPERBLOCKS as usize;
    let (q_bytes, x_vec) = helpers::generate_q4km_test_data(N_SUPERBLOCKS, FIXTURE_SEED);
    let cpu_result: f32 = helpers::q4km_dequant_matvec_cpu(&q_bytes, &x_vec, n);

    let x_bytes: Vec<u8> = common::f32_slice_to_bytes(&x_vec);
    let y_init: Vec<u8> = vec![0u8; 4];
    let pc: Vec<u8> = assemble_push_constants(plan, N_SUPERBLOCKS);

    // Use the resident path for the oracle so we exercise the SAME handle being timed.
    let resident = ctx
        .upload_resident(
            handle,
            &[q_bytes.as_slice(), x_bytes.as_slice(), y_init.as_slice()],
            &[0, 0, 4],
        )
        .expect("oracle upload_resident must succeed");
    let cfg = ResidentBenchConfig {
        workgroups: (1, 1, 1),
        push_constants: pc.clone(),
    };
    ctx.dispatch_resident(handle, &resident, &cfg)
        .expect("oracle dispatch must succeed");
    let outputs = ctx
        .readback_resident(handle, &resident, &[0, 0, 4])
        .expect("oracle readback must succeed");

    let y_bytes: &[u8] = &outputs[2];
    assert!(y_bytes.len() >= 4, "oracle: y output too short");
    let gpu_result: f32 = f32::from_le_bytes([y_bytes[0], y_bytes[1], y_bytes[2], y_bytes[3]]);
    let rel_err: f32 = if cpu_result.abs() > 1e-10 {
        (gpu_result - cpu_result).abs() / cpu_result.abs()
    } else {
        (gpu_result - cpu_result).abs()
    };
    assert!(
        rel_err < 1e-3,
        "AT-1761 pre-flight: Q4_K_M GPU={gpu_result:.6} vs CPU={cpu_result:.6}, \
         rel_err={rel_err:.2e} >= 1e-3 (kernel not bit-exact — A/B aborted)"
    );
    eprintln!(
        "dispatch_q4km_ab: AT-1761 pre-flight OK (GPU={gpu_result:.6}, CPU={cpu_result:.6}, \
         rel_err={rel_err:.2e})"
    );
}

fn bench_q4km_ab(c: &mut Criterion) {
    if !gpu_benches_enabled() {
        eprintln!("dispatch_q4km_ab: AXC_ENABLE_GPU_BENCHES != 1; skipping");
        c.bench_function("dispatch_q4km_ab", |b| b.iter(|| {}));
        return;
    }

    let ctx = match VulkanContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("dispatch_q4km_ab: no Vulkan: {e}; skipping");
            c.bench_function("dispatch_q4km_ab", |b| b.iter(|| {}));
            return;
        }
    };
    let device_name: String = ctx.physical_device_name().to_string();
    eprintln!("dispatch_q4km_ab: device '{device_name}'");

    // 50 GB cap assertion (anti-pattern #1).
    let n: usize = N_SUPERBLOCKS as usize;
    let total_bytes: u64 =
        (n * helpers::Q4KM_SUPERBLOCK_BYTES + n * helpers::Q4KM_SUPERBLOCK_ELEMS * 4 + 4) as u64;
    assert!(total_bytes < FIFTY_GB, "dispatch_q4km_ab: fixture exceeds 50 GB cap");

    let (bytes, meta) = match compile_source_with_meta(Q4KM_MATVEC_SRC) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("dispatch_q4km_ab: compile failed: {e}; skipping");
            c.bench_function("dispatch_q4km_ab", |b| b.iter(|| {}));
            return;
        }
    };
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let handle = match ctx.prepare_kernel_checked(
        &words,
        &meta.binding_plan,
        meta.push_constant_total_bytes,
        &meta.entry_point,
        None,
        "q4km_dequant_matvec",
        meta.shared_memory_bytes,
    ) {
        Ok(h) => h,
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
            eprintln!("dispatch_q4km_ab: DeviceFeatureUnsupported {feature}/{kernel}; skipping");
            c.bench_function("dispatch_q4km_ab", |b| b.iter(|| {}));
            return;
        }
        Err(e) => {
            eprintln!("dispatch_q4km_ab: prepare failed: {e}; skipping");
            c.bench_function("dispatch_q4km_ab", |b| b.iter(|| {}));
            return;
        }
    };

    // AT-1761: bit-exactness BEFORE timing.
    pre_flight_bit_exact(&ctx, &handle, &meta.binding_plan);

    // Build the timed inputs (real Q4_K_M weights — NOT zeros, REAL 144-byte superblocks).
    let (q_bytes, x_vec) = helpers::generate_q4km_test_data(N_SUPERBLOCKS, FIXTURE_SEED);
    let x_bytes: Vec<u8> = common::f32_slice_to_bytes(&x_vec);
    let y_init: Vec<u8> = vec![0u8; 4];
    let pc: Vec<u8> = assemble_push_constants(&meta.binding_plan, N_SUPERBLOCKS);

    let mut group = c.benchmark_group("dispatch_q4km_ab");
    group.sample_size(10);

    group.bench_function("dispatch_q4km_ab", |b| {
        b.iter_custom(|iters| {
            let mut total_ns: u64 = 0;
            for _ in 0..iters {
                let m = axiom_q4km_kernel_measurement(
                    &ctx,
                    &handle,
                    &[q_bytes.as_slice(), x_bytes.as_slice(), y_init.as_slice()],
                    &[0, 0, 4],
                    pc.clone(),
                );

                // FLOPs convention: matmul MACs only (2 * M * K), M=1, dequant EXCLUDED —
                // matches llama.cpp's op_flops(MUL_MAT)=2*m*n*k counting (per-row analogue).
                let flops: u64 = 2 * K_CONTRACTION as u64;
                let ts_label = timing_source_label(m.timing_source);

                // Machine-readable line for scripts/m34_llamacpp_ab.sh (anchored prefix).
                println!(
                    "AXC_Q4KM_AB kernel_ns_min={} kernel_ns_mean={} kernel_ns_median={} \
                     sustained_ns={} timing_source={:?} n_superblocks={} K={} M=1 flops={} \
                     device={}",
                    m.kernel_ns_min,
                    m.kernel_ns_mean,
                    m.kernel_ns_median,
                    m.sustained_ns,
                    m.timing_source,
                    N_SUPERBLOCKS,
                    K_CONTRACTION,
                    flops,
                    device_name,
                );
                eprintln!(
                    "dispatch_q4km_ab: min={} ns mean={} ns median={} ns sustained={} ns/op \
                     ({}) | K={} M=1 flops={}",
                    m.kernel_ns_min,
                    m.kernel_ns_mean,
                    m.kernel_ns_median,
                    m.sustained_ns,
                    ts_label,
                    K_CONTRACTION,
                    flops,
                );

                assert!(
                    m.kernel_ns_min > 0 && (m.kernel_ns_min as f64).is_finite(),
                    "kernel_ns_min must be > 0 and finite"
                );
                assert!(m.sustained_ns > 0, "sustained_ns must be > 0");
                total_ns += m.kernel_ns_min;
            }
            std::time::Duration::from_nanos(total_ns)
        });
    });

    group.finish();
}

criterion_group!(dispatch_q4km_ab_benches, bench_q4km_ab);
criterion_main!(dispatch_q4km_ab_benches);
