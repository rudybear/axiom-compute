//! Criterion bench groups `dispatch_gpu`, `dispatch_gpu_amortized`, and
//! `dispatch_gpu_q4_0` (M2.2 / M2.3a / M2.5).
//!
//! `dispatch_gpu`: Measures `VulkanContext::dispatch` end-to-end latency for
//! saxpy and vector_add at N ∈ {1024, 1M}.
//!
//! `dispatch_gpu_amortized`: Measures `VulkanContext::dispatch_handle` latency
//! for a pre-compiled `KernelHandle` (prepare-once/dispatch-many path).
//! Added in M2.3a. Benches one warm-up dispatch, then measures single
//! `dispatch_handle` calls on the same handle.  Demonstrates that the
//! per-dispatch cost is dominated by CB record + fence wait (not pipeline
//! compilation or buffer allocation).
//!
//! `dispatch_gpu_q4_0` (M2.5): Measures end-to-end GPU dispatch latency for the
//! Q4_0 dequant+matvec kernel at n_blocks ∈ {128, 1024}.  The kernel produces
//! a single f32 output (single-invocation kernel with 64-wide workgroup).
//! Pre-flight correctness is verified against `q4_0_dequant_matvec_cpu` within
//! an absolute tolerance of 1e-3 (matches AT-918 tolerance for f16 scale values).
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
//! benches via a lazy static.  Device/queue/command-pool creation is NOT
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
//! - `dispatch_handle_saxpy_1m`: `BatchSize::PerIteration` — handle is pre-warmed,
//!   each iteration calls `dispatch_handle` directly; buffer pool is already allocated.
//! - `dispatch_gpu_q4_0_128` / `dispatch_gpu_q4_0_1024`: `BatchSize::PerIteration`
//!   — single-invocation kernel, tiny output; per-iteration overhead is the
//!   dominant cost (CB record + queue submit + fence wait + single f32 readback).

#[path = "common.rs"]
mod common;

use axc_runtime::{VulkanContext, VulkanContextOptions, KernelHandle, DispatchRequest, probe_vulkan_available};
use axc_hir::{ParamBindingPlan, ScalarTy};
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
    /// Pre-compiled KernelHandle for the amortized dispatch bench (M2.3a).
    saxpy_handle: KernelHandle,
    /// M2.5: Q4_0 dequant+matvec SPIR-V and binding plan.
    q4_0_spirv: Vec<u32>,
    q4_0_plan: ParamBindingPlan,
}

static GPU_STATE: OnceLock<GpuState> = OnceLock::new();

/// Initialize GPU state once; panics if Vulkan setup or compilation fails.
fn gpu_state() -> &'static GpuState {
    GPU_STATE.get_or_init(|| {
        let ctx: VulkanContext = VulkanContext::new_with_options(VulkanContextOptions {
            pipeline_cache_path: None,
            physical_device_index: None,
            fence_timeout_ms: None,
        })
        .expect("dispatch bench: VulkanContext::new_with_options must succeed");

        eprintln!(
            "dispatch_gpu: using device '{}'",
            ctx.physical_device_name()
        );

        let (saxpy_bytes, saxpy_meta) =
            axc_driver::compile_source_with_meta(common::SAXPY_SRC)
                .expect("dispatch bench: saxpy.axc compile failed");
        let saxpy_spirv: Vec<u32> = common::bytes_to_words(&saxpy_bytes);
        let saxpy_plan: ParamBindingPlan = saxpy_meta.binding_plan.clone();

        // Pre-compile KernelHandle for the amortized bench (M2.3a).
        let saxpy_handle: KernelHandle = ctx.prepare_kernel(
            &saxpy_spirv,
            &saxpy_plan,
            saxpy_meta.binding_plan.push_constant_total_bytes,
            &saxpy_meta.entry_point,
        )
        .expect("dispatch bench: prepare_kernel for saxpy must succeed");

        let (va_bytes, va_meta) =
            axc_driver::compile_source_with_meta(common::VECTOR_ADD_SRC)
                .expect("dispatch bench: vector_add.axc compile failed");
        let vector_add_spirv: Vec<u32> = common::bytes_to_words(&va_bytes);
        let vector_add_plan: ParamBindingPlan = va_meta.binding_plan;

        // M2.5: Compile q4_0_dequant_matvec kernel.
        let q4_0_src: &str = include_str!("../../../examples/q4_0_dequant_matvec.axc");
        let (q4_0_bytes, q4_0_meta) =
            axc_driver::compile_source_with_meta(q4_0_src)
                .expect("dispatch bench: q4_0_dequant_matvec.axc compile failed");
        let q4_0_spirv: Vec<u32> = common::bytes_to_words(&q4_0_bytes);
        let q4_0_plan: ParamBindingPlan = q4_0_meta.binding_plan;

        GpuState {
            ctx,
            saxpy_spirv,
            saxpy_plan,
            vector_add_spirv,
            vector_add_plan,
            saxpy_handle,
            q4_0_spirv,
            q4_0_plan,
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

// ── Amortized dispatch_handle bench (M2.3a) ────────────────────────────────────

/// Bench `dispatch_handle_saxpy_1m`: amortized `dispatch_handle` on a
/// pre-compiled `KernelHandle`, saxpy N=1M.
///
/// The handle is prepared once in `gpu_state()` (pipeline compile + buffer
/// alloc happen during first `dispatch_handle` call, outside the measured
/// region).  Each Criterion iteration calls `dispatch_handle` directly:
/// fence reset, CB record, queue submit, fence wait, readback.
///
/// BatchSize::PerIteration — each iteration is a complete GPU round-trip.
fn dispatch_handle_saxpy_1m(c: &mut Criterion) {
    if !gpu_benches_enabled() {
        eprintln!(
            "skipping dispatch_handle_saxpy_1m: AXC_ENABLE_GPU_BENCHES != 1 or no Vulkan ICD"
        );
        c.bench_function("dispatch_handle_saxpy_1m", |b| {
            b.iter(|| {});
        });
        return;
    }

    let state: &GpuState = gpu_state();

    let n: u32 = N_LARGE;
    let buf_size: usize = n as usize * 4;
    let workgroups: (u32, u32, u32) = (n.div_ceil(WG_SIZE_X), 1, 1);

    let (x, y, alpha) = common::make_saxpy_inputs(n as usize, common::SEED);
    let x_bytes: Vec<u8> = common::f32_slice_to_bytes(&x);
    let y_bytes: Vec<u8> = common::f32_slice_to_bytes(&y);
    let pc: Vec<u8> = common::assemble_saxpy_push_constants(&state.saxpy_plan, n, alpha);

    // Warm-up dispatch: allocates buffers and verifies correctness.
    // This first call is NOT in the measured region.
    {
        let outputs: Vec<Vec<u8>> = state.ctx.dispatch_handle(
            &state.saxpy_handle,
            workgroups,
            &[&x_bytes, &y_bytes],
            &[buf_size, buf_size],
            &pc,
        )
        .expect("dispatch_handle_saxpy_1m: warm-up dispatch must succeed");
        let y_out: Vec<f32> = common::bytes_to_f32_vec(&outputs[1]);
        let cpu_ref: Vec<f32> = common::saxpy_cpu_reference(&x, &y, alpha);
        for i in 0..n as usize {
            let err: f32 = (y_out[i] - cpu_ref[i]).abs();
            assert!(
                err < common::ABS_TOL,
                "dispatch_handle_saxpy_1m warm-up: GPU[{i}]={:.8}, CPU[{i}]={:.8}, err={:.2e}",
                y_out[i], cpu_ref[i], err
            );
        }
    }

    c.bench_function("dispatch_handle_saxpy_1m", |b| {
        b.iter_batched(
            || {},
            |()| {
                let _outputs = state.ctx.dispatch_handle(
                    &state.saxpy_handle,
                    workgroups,
                    &[&x_bytes, &y_bytes],
                    &[buf_size, buf_size],
                    &pc,
                )
                .expect("dispatch_handle_saxpy_1m: dispatch must succeed");
            },
            BatchSize::PerIteration,
        );
    });
}

// ── Q4_0 dequant+matvec dispatch benches (M2.5) ───────────────────────────────

/// Pre-flight correctness check for Q4_0 dispatch.
///
/// Dispatches the Q4_0 kernel once and verifies `y[0]` matches the CPU
/// reference within `ABS_TOL_Q4_0 = 1e-3` (f16 scale precision limit).
fn pre_flight_q4_0(state: &GpuState, n_blocks: usize) {
    /// Absolute tolerance for Q4_0: f16 scale has ~3 decimal digits of precision.
    const ABS_TOL_Q4_0: f32 = 1e-3;

    let (q_bytes, x_vec) = common::make_q4_0_inputs(n_blocks, common::SEED);
    let cpu_result: f32 = common::q4_0_dequant_matvec_cpu(&q_bytes, &x_vec, n_blocks);
    let x_bytes: Vec<u8> = common::f32_slice_to_bytes(&x_vec);
    let y_init: Vec<u8> = vec![0u8; 4];

    // Build push-constant block for n_blocks.
    let mut pc: Vec<u8> = vec![0u8; state.q4_0_plan.push_constant_total_bytes as usize];
    let n_blocks_u32: u32 = n_blocks as u32;
    for scalar in &state.q4_0_plan.scalars {
        let start: usize = scalar.offset as usize;
        if scalar.ty == ScalarTy::U32 {
            pc[start..start + 4].copy_from_slice(&n_blocks_u32.to_le_bytes());
        }
    }

    let q_len: usize = n_blocks * common::Q4_0_BLOCK_BYTES;
    let x_len: usize = n_blocks * common::Q4_0_BLOCK_ELEMS * 4;

    let req = DispatchRequest {
        spirv: &state.q4_0_spirv,
        binding_plan: &state.q4_0_plan,
        workgroups: [1, 1, 1],
        inputs: &[q_bytes.as_slice(), x_bytes.as_slice(), y_init.as_slice()],
        output_sizes: &[q_len, x_len, 4],
        push_constants: &pc,
        entry_point: "q4_0_dequant_matvec",
    };

    let outputs: Vec<Vec<u8>> = state
        .ctx
        .dispatch(req)
        .expect("pre_flight_q4_0: dispatch must succeed");

    let y_bytes: &[u8] = &outputs[2];
    assert!(y_bytes.len() >= 4, "pre_flight_q4_0: y output too short");
    let gpu_result: f32 = f32::from_le_bytes([y_bytes[0], y_bytes[1], y_bytes[2], y_bytes[3]]);

    let abs_err: f32 = (gpu_result - cpu_result).abs();
    assert!(
        abs_err < ABS_TOL_Q4_0,
        "pre_flight_q4_0(n_blocks={n_blocks}): GPU={gpu_result:.6}, CPU={cpu_result:.6}, abs_err={abs_err:.2e} >= 1e-3"
    );
}

/// Bench `dispatch_gpu_q4_0_128`: Q4_0 dequant+matvec dispatch, n_blocks=128 (128 * 18 = 2304 weight bytes).
///
/// BatchSize::PerIteration — full GPU round-trip per iteration.
fn dispatch_gpu_q4_0_128(c: &mut Criterion) {
    if !gpu_benches_enabled() {
        eprintln!("skipping dispatch_gpu_q4_0_128: AXC_ENABLE_GPU_BENCHES != 1 or no Vulkan ICD");
        c.bench_function("dispatch_gpu_q4_0_128", |b| {
            b.iter(|| {});
        });
        return;
    }

    let state: &GpuState = gpu_state();
    let n_blocks: usize = 128;
    pre_flight_q4_0(state, n_blocks);

    let (q_bytes, x_vec) = common::make_q4_0_inputs(n_blocks, common::SEED);
    let x_bytes: Vec<u8> = common::f32_slice_to_bytes(&x_vec);
    let y_init: Vec<u8> = vec![0u8; 4];

    let mut pc: Vec<u8> = vec![0u8; state.q4_0_plan.push_constant_total_bytes as usize];
    let n_blocks_u32: u32 = n_blocks as u32;
    for scalar in &state.q4_0_plan.scalars {
        let start: usize = scalar.offset as usize;
        if scalar.ty == ScalarTy::U32 {
            pc[start..start + 4].copy_from_slice(&n_blocks_u32.to_le_bytes());
        }
    }

    let q_len: usize = n_blocks * common::Q4_0_BLOCK_BYTES;
    let x_len: usize = n_blocks * common::Q4_0_BLOCK_ELEMS * 4;

    c.bench_function("dispatch_gpu_q4_0_128", |b| {
        b.iter_batched(
            || {},
            |()| {
                let req = DispatchRequest {
                    spirv: &state.q4_0_spirv,
                    binding_plan: &state.q4_0_plan,
                    workgroups: [1, 1, 1],
                    inputs: &[q_bytes.as_slice(), x_bytes.as_slice(), y_init.as_slice()],
                    output_sizes: &[q_len, x_len, 4],
                    push_constants: &pc,
                    entry_point: "q4_0_dequant_matvec",
                };
                let _outputs = state
                    .ctx
                    .dispatch(req)
                    .expect("dispatch_gpu_q4_0_128: dispatch must succeed");
            },
            BatchSize::PerIteration,
        );
    });
}

/// Bench `dispatch_gpu_q4_0_1024`: Q4_0 dequant+matvec dispatch, n_blocks=1024.
///
/// BatchSize::PerIteration — single-output kernel; overhead is dominated by
/// GPU round-trip latency (CB record + submit + fence wait), not data size.
fn dispatch_gpu_q4_0_1024(c: &mut Criterion) {
    if !gpu_benches_enabled() {
        eprintln!("skipping dispatch_gpu_q4_0_1024: AXC_ENABLE_GPU_BENCHES != 1 or no Vulkan ICD");
        c.bench_function("dispatch_gpu_q4_0_1024", |b| {
            b.iter(|| {});
        });
        return;
    }

    let state: &GpuState = gpu_state();
    let n_blocks: usize = 1024;
    pre_flight_q4_0(state, n_blocks);

    let (q_bytes, x_vec) = common::make_q4_0_inputs(n_blocks, common::SEED);
    let x_bytes: Vec<u8> = common::f32_slice_to_bytes(&x_vec);
    let y_init: Vec<u8> = vec![0u8; 4];

    let mut pc: Vec<u8> = vec![0u8; state.q4_0_plan.push_constant_total_bytes as usize];
    let n_blocks_u32: u32 = n_blocks as u32;
    for scalar in &state.q4_0_plan.scalars {
        let start: usize = scalar.offset as usize;
        if scalar.ty == ScalarTy::U32 {
            pc[start..start + 4].copy_from_slice(&n_blocks_u32.to_le_bytes());
        }
    }

    let q_len: usize = n_blocks * common::Q4_0_BLOCK_BYTES;
    let x_len: usize = n_blocks * common::Q4_0_BLOCK_ELEMS * 4;

    c.bench_function("dispatch_gpu_q4_0_1024", |b| {
        b.iter_batched(
            || {},
            |()| {
                let req = DispatchRequest {
                    spirv: &state.q4_0_spirv,
                    binding_plan: &state.q4_0_plan,
                    workgroups: [1, 1, 1],
                    inputs: &[q_bytes.as_slice(), x_bytes.as_slice(), y_init.as_slice()],
                    output_sizes: &[q_len, x_len, 4],
                    push_constants: &pc,
                    entry_point: "q4_0_dequant_matvec",
                };
                let _outputs = state
                    .ctx
                    .dispatch(req)
                    .expect("dispatch_gpu_q4_0_1024: dispatch must succeed");
            },
            BatchSize::PerIteration,
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
criterion_group!(
    dispatch_gpu_amortized,
    dispatch_handle_saxpy_1m,
);
criterion_group!(
    dispatch_gpu_q4_0,
    dispatch_gpu_q4_0_128,
    dispatch_gpu_q4_0_1024,
);
criterion_main!(dispatch_benches, dispatch_gpu_amortized, dispatch_gpu_q4_0);
