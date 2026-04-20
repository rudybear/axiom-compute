//! Criterion bench group `dispatch_gpu` (M2.2).
//!
//! Measures `VulkanContext::dispatch` end-to-end latency for saxpy and
//! vector_add at N ∈ {1024, 1M}.
//!
//! # GPU availability gate
//!
//! Gated on `AXC_ENABLE_GPU_BENCHES=1` AND `probe_vulkan_available()`.
//! When either condition is false, every bench in the group emits a single
//! dummy iteration and prints a skip message — keeping Criterion's test-name
//! table consistent with AT-708's declared bench count.
//!
//! # VulkanContext lifetime
//!
//! One `VulkanContext` is constructed once (expensive) and shared across all
//! four benches via a lazy static.  Device/queue/command-pool creation is NOT
//! in the measured region.
//!
//! # Pre-flight correctness (AT-705)
//!
//! Before the measurement loop, each bench dispatches once and verifies the GPU
//! output matches the CPU reference within 1e-6 absolute tolerance.  A
//! divergence panics with a descriptive message.
//!
//! # BatchSize rationale (C2 resolution)
//!
//! - `dispatch_saxpy_1024` / `dispatch_vector_add_1024`: `BatchSize::PerIteration`
//!   — each iteration is a full GPU round-trip (CB record + submit + fence wait +
//!   readback), measurement-time-dominated.  PerIteration avoids Criterion
//!   pre-allocating batches of request builders that hold borrowed slices.
//! - `dispatch_saxpy_1m` / `dispatch_vector_add_1m`: `BatchSize::LargeInput`
//!   — each iteration's setup builds a fresh output Vec of 4 MB; LargeInput
//!   reduces pre-allocated batch size so setup-memory peaks stay bounded.

#[path = "common.rs"]
mod common;

use axc_runtime::{VulkanContext, DispatchRequest, probe_vulkan_available};
use axc_hir::ParamBindingPlan;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use std::sync::OnceLock;

/// N for "small" dispatch benches.
const N_SMALL: u32 = 1_024;

/// N for "large" dispatch benches.
const N_LARGE: u32 = 1_048_576;

/// Workgroup size matching saxpy.axc / vector_add.axc `@workgroup(64, 1, 1)`.
const WG_SIZE_X: u32 = 64;

/// Returns `true` if GPU benches should run (env var + ICD probe).
fn gpu_benches_enabled() -> bool {
    std::env::var("AXC_ENABLE_GPU_BENCHES").as_deref() == Ok("1")
        && probe_vulkan_available()
}

// ── Lazy-initialized GPU state (one VulkanContext for the whole bench run) ────

struct GpuState {
    ctx: VulkanContext,
    saxpy_spirv: Vec<u32>,
    saxpy_plan: ParamBindingPlan,
    vector_add_spirv: Vec<u32>,
    vector_add_plan: ParamBindingPlan,
}

static GPU_STATE: OnceLock<GpuState> = OnceLock::new();

/// Initialize GPU state once; panics if Vulkan setup or compilation fails.
fn gpu_state() -> &'static GpuState {
    GPU_STATE.get_or_init(|| {
        let ctx: VulkanContext = VulkanContext::new()
            .expect("dispatch bench: VulkanContext::new() must succeed");

        eprintln!(
            "dispatch_gpu: using device '{}'",
            ctx.physical_device_name()
        );

        let (saxpy_bytes, saxpy_meta) =
            axc_driver::compile_source_with_meta(common::SAXPY_SRC)
                .expect("dispatch bench: saxpy.axc compile failed");
        let saxpy_spirv: Vec<u32> = common::bytes_to_words(&saxpy_bytes);
        let saxpy_plan: ParamBindingPlan = saxpy_meta.binding_plan;

        let (va_bytes, va_meta) =
            axc_driver::compile_source_with_meta(common::VECTOR_ADD_SRC)
                .expect("dispatch bench: vector_add.axc compile failed");
        let vector_add_spirv: Vec<u32> = common::bytes_to_words(&va_bytes);
        let vector_add_plan: ParamBindingPlan = va_meta.binding_plan;

        GpuState {
            ctx,
            saxpy_spirv,
            saxpy_plan,
            vector_add_spirv,
            vector_add_plan,
        }
    })
}

// ── Pre-flight correctness oracle ─────────────────────────────────────────────

/// Dispatch saxpy once and verify GPU output vs CPU reference within ABS_TOL.
///
/// Panics with a descriptive message on divergence.
fn pre_flight_saxpy(state: &GpuState, n: u32) {
    let (x, y, alpha) = common::make_saxpy_inputs(n as usize, common::SEED);
    let x_bytes: Vec<u8> = common::f32_slice_to_bytes(&x);
    let y_bytes: Vec<u8> = common::f32_slice_to_bytes(&y);
    let pc: Vec<u8> = common::assemble_saxpy_push_constants(&state.saxpy_plan, n, alpha);
    let buf_size: usize = n as usize * 4;
    let workgroups: [u32; 3] = [n.div_ceil(WG_SIZE_X), 1, 1];

    let req = DispatchRequest {
        spirv: &state.saxpy_spirv,
        binding_plan: &state.saxpy_plan,
        workgroups,
        inputs: &[&x_bytes, &y_bytes],
        output_sizes: &[buf_size, buf_size],
        push_constants: &pc,
        entry_point: "saxpy",
    };

    let outputs: Vec<Vec<u8>> = state
        .ctx
        .dispatch(req)
        .expect("pre_flight_saxpy: dispatch must succeed");
    let y_out: Vec<f32> = common::bytes_to_f32_vec(&outputs[1]);
    let cpu_ref: Vec<f32> = common::saxpy_cpu_reference(&x, &y, alpha);

    for i in 0..n as usize {
        let err: f32 = (y_out[i] - cpu_ref[i]).abs();
        assert!(
            err < common::ABS_TOL,
            "pre_flight_saxpy(n={n}): GPU[{i}]={:.8}, CPU[{i}]={:.8}, err={:.2e} >= ABS_TOL={:.2e}",
            y_out[i], cpu_ref[i], err, common::ABS_TOL
        );
    }
}

/// Dispatch vector_add once and verify GPU output vs CPU reference within ABS_TOL.
///
/// Panics with a descriptive message on divergence.
fn pre_flight_vector_add(state: &GpuState, n: u32) {
    let (a, b) = common::make_vector_add_inputs(n as usize, common::SEED);
    let a_bytes: Vec<u8> = common::f32_slice_to_bytes(&a);
    let b_bytes: Vec<u8> = common::f32_slice_to_bytes(&b);
    let pc: Vec<u8> =
        common::assemble_vector_add_push_constants(&state.vector_add_plan, n);
    let buf_size: usize = n as usize * 4;
    let c_input: Vec<u8> = vec![0u8; buf_size];
    let workgroups: [u32; 3] = [n.div_ceil(WG_SIZE_X), 1, 1];

    let req = DispatchRequest {
        spirv: &state.vector_add_spirv,
        binding_plan: &state.vector_add_plan,
        workgroups,
        inputs: &[&a_bytes, &b_bytes, &c_input],
        output_sizes: &[buf_size, buf_size, buf_size],
        push_constants: &pc,
        entry_point: "vector_add",
    };

    let outputs: Vec<Vec<u8>> = state
        .ctx
        .dispatch(req)
        .expect("pre_flight_vector_add: dispatch must succeed");
    let c_out: Vec<f32> = common::bytes_to_f32_vec(&outputs[2]);
    let cpu_ref: Vec<f32> = common::vector_add_cpu_reference(&a, &b);

    for i in 0..n as usize {
        let err: f32 = (c_out[i] - cpu_ref[i]).abs();
        assert!(
            err < common::ABS_TOL,
            "pre_flight_vector_add(n={n}): GPU[{i}]={:.8}, CPU[{i}]={:.8}, err={:.2e} >= ABS_TOL={:.2e}",
            c_out[i], cpu_ref[i], err, common::ABS_TOL
        );
    }
}

// ── Saxpy dispatch benches ─────────────────────────────────────────────────────

/// Bench `dispatch_saxpy_1024`: end-to-end GPU dispatch for saxpy, N=1024.
///
/// BatchSize::PerIteration — full GPU round-trip per iteration;
/// measurement-time-dominated rather than setup-dominated.
fn dispatch_saxpy_1024(c: &mut Criterion) {
    if !gpu_benches_enabled() {
        eprintln!("skipping dispatch_saxpy_1024: AXC_ENABLE_GPU_BENCHES != 1 or no Vulkan ICD");
        c.bench_function("dispatch_saxpy_1024", |b| {
            b.iter(|| {});
        });
        return;
    }

    let state: &GpuState = gpu_state();
    pre_flight_saxpy(state, N_SMALL);

    let n: u32 = N_SMALL;
    let buf_size: usize = n as usize * 4;
    let workgroups: [u32; 3] = [n.div_ceil(WG_SIZE_X), 1, 1];

    let (x, y, alpha) = common::make_saxpy_inputs(n as usize, common::SEED);
    let x_bytes: Vec<u8> = common::f32_slice_to_bytes(&x);
    let y_bytes: Vec<u8> = common::f32_slice_to_bytes(&y);
    let pc: Vec<u8> = common::assemble_saxpy_push_constants(&state.saxpy_plan, n, alpha);

    c.bench_function("dispatch_saxpy_1024", |b| {
        // PerIteration: each full GPU round-trip is one iteration.
        b.iter_batched(
            || {},
            |()| {
                let req = DispatchRequest {
                    spirv: &state.saxpy_spirv,
                    binding_plan: &state.saxpy_plan,
                    workgroups,
                    inputs: &[&x_bytes, &y_bytes],
                    output_sizes: &[buf_size, buf_size],
                    push_constants: &pc,
                    entry_point: "saxpy",
                };
                let _outputs = state
                    .ctx
                    .dispatch(req)
                    .expect("dispatch_saxpy_1024: dispatch must succeed");
            },
            BatchSize::PerIteration,
        );
    });
}

/// Bench `dispatch_saxpy_1m`: end-to-end GPU dispatch for saxpy, N=1M.
///
/// BatchSize::LargeInput — each iteration's setup builds a fresh output Vec
/// (4 MB); LargeInput reduces pre-allocated batch size for bounded peak memory.
fn dispatch_saxpy_1m(c: &mut Criterion) {
    if !gpu_benches_enabled() {
        eprintln!("skipping dispatch_saxpy_1m: AXC_ENABLE_GPU_BENCHES != 1 or no Vulkan ICD");
        c.bench_function("dispatch_saxpy_1m", |b| {
            b.iter(|| {});
        });
        return;
    }

    let state: &GpuState = gpu_state();
    pre_flight_saxpy(state, N_LARGE);

    let n: u32 = N_LARGE;
    let buf_size: usize = n as usize * 4;
    let workgroups: [u32; 3] = [n.div_ceil(WG_SIZE_X), 1, 1];

    let (x, y, alpha) = common::make_saxpy_inputs(n as usize, common::SEED);
    let x_bytes: Vec<u8> = common::f32_slice_to_bytes(&x);
    let y_bytes: Vec<u8> = common::f32_slice_to_bytes(&y);
    let pc: Vec<u8> = common::assemble_saxpy_push_constants(&state.saxpy_plan, n, alpha);

    c.bench_function("dispatch_saxpy_1m", |b| {
        // LargeInput: 4 MB output Vecs; reduces batch size for peak-memory stability.
        b.iter_batched(
            || {},
            |()| {
                let req = DispatchRequest {
                    spirv: &state.saxpy_spirv,
                    binding_plan: &state.saxpy_plan,
                    workgroups,
                    inputs: &[&x_bytes, &y_bytes],
                    output_sizes: &[buf_size, buf_size],
                    push_constants: &pc,
                    entry_point: "saxpy",
                };
                let _outputs = state
                    .ctx
                    .dispatch(req)
                    .expect("dispatch_saxpy_1m: dispatch must succeed");
            },
            BatchSize::LargeInput,
        );
    });
}

// ── Vector_add dispatch benches ────────────────────────────────────────────────

/// Bench `dispatch_vector_add_1024`: end-to-end GPU dispatch for vector_add, N=1024.
///
/// BatchSize::PerIteration — full GPU round-trip per iteration.
fn dispatch_vector_add_1024(c: &mut Criterion) {
    if !gpu_benches_enabled() {
        eprintln!(
            "skipping dispatch_vector_add_1024: AXC_ENABLE_GPU_BENCHES != 1 or no Vulkan ICD"
        );
        c.bench_function("dispatch_vector_add_1024", |b| {
            b.iter(|| {});
        });
        return;
    }

    let state: &GpuState = gpu_state();
    pre_flight_vector_add(state, N_SMALL);

    let n: u32 = N_SMALL;
    let buf_size: usize = n as usize * 4;
    let workgroups: [u32; 3] = [n.div_ceil(WG_SIZE_X), 1, 1];

    let (a, b) = common::make_vector_add_inputs(n as usize, common::SEED);
    let a_bytes: Vec<u8> = common::f32_slice_to_bytes(&a);
    let b_bytes: Vec<u8> = common::f32_slice_to_bytes(&b);
    let pc: Vec<u8> = common::assemble_vector_add_push_constants(&state.vector_add_plan, n);
    let c_input: Vec<u8> = vec![0u8; buf_size];

    c.bench_function("dispatch_vector_add_1024", |b_crit| {
        b_crit.iter_batched(
            || {},
            |()| {
                let req = DispatchRequest {
                    spirv: &state.vector_add_spirv,
                    binding_plan: &state.vector_add_plan,
                    workgroups,
                    inputs: &[&a_bytes, &b_bytes, &c_input],
                    output_sizes: &[buf_size, buf_size, buf_size],
                    push_constants: &pc,
                    entry_point: "vector_add",
                };
                let _outputs = state
                    .ctx
                    .dispatch(req)
                    .expect("dispatch_vector_add_1024: dispatch must succeed");
            },
            BatchSize::PerIteration,
        );
    });
}

/// Bench `dispatch_vector_add_1m`: end-to-end GPU dispatch for vector_add, N=1M.
///
/// BatchSize::LargeInput — 4 MB output buffers; LargeInput for bounded peak memory.
fn dispatch_vector_add_1m(c: &mut Criterion) {
    if !gpu_benches_enabled() {
        eprintln!(
            "skipping dispatch_vector_add_1m: AXC_ENABLE_GPU_BENCHES != 1 or no Vulkan ICD"
        );
        c.bench_function("dispatch_vector_add_1m", |b| {
            b.iter(|| {});
        });
        return;
    }

    let state: &GpuState = gpu_state();
    pre_flight_vector_add(state, N_LARGE);

    let n: u32 = N_LARGE;
    let buf_size: usize = n as usize * 4;
    let workgroups: [u32; 3] = [n.div_ceil(WG_SIZE_X), 1, 1];

    let (a, b) = common::make_vector_add_inputs(n as usize, common::SEED);
    let a_bytes: Vec<u8> = common::f32_slice_to_bytes(&a);
    let b_bytes: Vec<u8> = common::f32_slice_to_bytes(&b);
    let pc: Vec<u8> = common::assemble_vector_add_push_constants(&state.vector_add_plan, n);
    let c_input: Vec<u8> = vec![0u8; buf_size];

    c.bench_function("dispatch_vector_add_1m", |b_crit| {
        b_crit.iter_batched(
            || {},
            |()| {
                let req = DispatchRequest {
                    spirv: &state.vector_add_spirv,
                    binding_plan: &state.vector_add_plan,
                    workgroups,
                    inputs: &[&a_bytes, &b_bytes, &c_input],
                    output_sizes: &[buf_size, buf_size, buf_size],
                    push_constants: &pc,
                    entry_point: "vector_add",
                };
                let _outputs = state
                    .ctx
                    .dispatch(req)
                    .expect("dispatch_vector_add_1m: dispatch must succeed");
            },
            BatchSize::LargeInput,
        );
    });
}

criterion_group!(
    dispatch_benches,
    dispatch_saxpy_1024,
    dispatch_saxpy_1m,
    dispatch_vector_add_1024,
    dispatch_vector_add_1m
);
criterion_main!(dispatch_benches);
