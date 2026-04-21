//! AT-816: dispatch_handle amortization — 100 dispatches < 10× first dispatch time.
//!
//! Verifies that the prepare-once/dispatch-many path amortizes the pipeline-compile
//! and buffer-alloc cost. After the first dispatch, subsequent dispatches reuse the
//! cached KernelHandle (pipeline, fence, buffers) and pay only for staging copies,
//! command buffer recording, and queue submission.

use axc_runtime::{VulkanContext, VulkanContextOptions, probe_vulkan_available, gpu_tests_enabled};
use axc_hir::ScalarTy;

fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn f32_slice_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// AT-816: 100 amortized dispatches total < 10× single first-dispatch time.
///
/// Measures: first dispatch_handle wall time, then 100 subsequent dispatches.
/// Asserts sum(subsequent 100) < 10 × first_dispatch_time.
/// Also verifies GPU correctness against CPU reference on last dispatch.
#[test]
#[ignore] // GPU-gated: requires AXC_ENABLE_GPU_TESTS=1 + Vulkan ICD
fn at_816_dispatch_handle_saxpy_amortized_100x() {
    if !gpu_tests_enabled() {
        eprintln!("skipping at_816 (AXC_ENABLE_GPU_TESTS != 1)");
        return;
    }
    if !probe_vulkan_available() {
        eprintln!("skipping at_816 (no Vulkan ICD available)");
        return;
    }

    let tmp_dir = tempfile::tempdir().expect("tempdir creation must succeed");
    let ctx = VulkanContext::new_with_options(VulkanContextOptions {
        pipeline_cache_path: Some(tmp_dir.path().join("at_816.cache")),
        physical_device_index: None,
        fence_timeout_ms: None,
    })
    .expect("VulkanContext::new_with_options must succeed");
    eprintln!("AT-816: device = {}", ctx.physical_device_name());

    let saxpy_src: &str = include_str!("../../../examples/saxpy.axc");
    let (spirv_bytes, meta) = axc_driver::compile_source_with_meta(saxpy_src)
        .expect("saxpy.axc must compile");
    let spirv_words: Vec<u32> = bytes_to_words(&spirv_bytes);

    // Prepare kernel ONCE (outside measured region).
    let handle = ctx.prepare_kernel(
        &spirv_words,
        &meta.binding_plan,
        meta.binding_plan.push_constant_total_bytes,
        &meta.entry_point,
    )
    .expect("prepare_kernel must succeed");

    const N: u32 = 1_048_576; // 1M elements
    let alpha: f32 = 2.5;
    let x_data: Vec<f32> = (0..N).map(|i| i as f32 * 0.25).collect();
    let y_data: Vec<f32> = (0..N).map(|i| i as f32 * 0.0625).collect();
    let x_bytes: Vec<u8> = f32_slice_to_bytes(&x_data);
    let y_bytes: Vec<u8> = f32_slice_to_bytes(&y_data);
    let buf_size: usize = N as usize * 4;
    let workgroups = (N.div_ceil(64), 1u32, 1u32);

    // Assemble push constants.
    let mut pc_bytes: Vec<u8> = vec![0u8; meta.binding_plan.push_constant_total_bytes as usize];
    for scalar in &meta.binding_plan.scalars {
        let start = scalar.offset as usize;
        match scalar.ty {
            ScalarTy::U32 => pc_bytes[start..start+4].copy_from_slice(&N.to_le_bytes()),
            ScalarTy::F32 => pc_bytes[start..start+4].copy_from_slice(&alpha.to_le_bytes()),
            _ => {}
        }
    }

    // ── Measure first dispatch ─────────────────────────────────────────────────
    let t0 = std::time::Instant::now();
    let _first_outputs = ctx.dispatch_handle(
        &handle,
        workgroups,
        &[&x_bytes, &y_bytes],
        &[buf_size, buf_size],
        &pc_bytes,
    )
    .expect("first dispatch must succeed");
    let first_ns: u64 = t0.elapsed().as_nanos() as u64;
    eprintln!("AT-816: first dispatch: {} ns ({} ms)", first_ns, first_ns / 1_000_000);

    // ── Measure 100 subsequent dispatches ─────────────────────────────────────
    let t_start = std::time::Instant::now();
    let mut last_outputs: Vec<Vec<u8>> = Vec::new();
    for i in 0..100 {
        let out = ctx.dispatch_handle(
            &handle,
            workgroups,
            &[&x_bytes, &y_bytes],
            &[buf_size, buf_size],
            &pc_bytes,
        )
        .unwrap_or_else(|e| panic!("dispatch {} must succeed: {e}", i + 2));
        last_outputs = out;
    }
    let subsequent_total_ns: u64 = t_start.elapsed().as_nanos() as u64;
    eprintln!(
        "AT-816: 100 subsequent dispatches total: {} ns ({} ms), avg: {} ns",
        subsequent_total_ns,
        subsequent_total_ns / 1_000_000,
        subsequent_total_ns / 100,
    );

    // ── Amortization assertion ─────────────────────────────────────────────────
    // Sum of 100 subsequent < 10 × first dispatch time.
    // Generous threshold: even if first dispatch is slow due to pipeline compile
    // on a cold machine, 100 fence-reuse dispatches should be << 10× first.
    assert!(
        subsequent_total_ns < first_ns * 10,
        "AT-816 FAIL: 100 subsequent dispatches ({} ns) must be < 10 × first ({} ns = {})",
        subsequent_total_ns,
        first_ns,
        first_ns * 10
    );
    eprintln!(
        "AT-816: amortization ratio = {:.2}× (must be < 10×)",
        subsequent_total_ns as f64 / first_ns as f64
    );

    // ── Correctness oracle (last dispatch) ────────────────────────────────────
    const ABS_TOL: f32 = 1e-6;
    let y_out: Vec<f32> = bytes_to_f32_vec(&last_outputs[1]);
    for i in 0..N as usize {
        let cpu_ref: f32 = alpha * x_data[i] + y_data[i];
        let err: f32 = (y_out[i] - cpu_ref).abs();
        assert!(
            err < ABS_TOL,
            "AT-816 correctness: output[{i}] GPU={} CPU={} err={:.2e}",
            y_out[i], cpu_ref, err
        );
    }
    eprintln!("AT-816: correctness verified on last dispatch — AT-816 PASS");
}
