//! Resident-timing integration tests — M3.1.5.
//!
//! Tests for the three resident methods (upload_resident, dispatch_resident,
//! readback_resident) and the prepare_kernel_checked preflight.
//!
//! ## GPU test gating
//!
//! All tests that require actual GPU dispatch are `#[ignore]`-gated and run only when:
//! - `AXC_ENABLE_GPU_TESTS=1` is set
//! - `cargo test -- --ignored` is used
//!
//! Non-ignored tests (AT-1573, AT-1576, AT-1578) are pure unit tests with no GPU.
//!
//! ## AT coverage
//!
//! - AT-1530: resident dispatch_ns < host-round-trip on GpuTimestamp path (warmup mandatory)
//! - AT-1531: resident dispatch works for saxpy + coopmat matmul_tile (typed-skip on Lavapipe)
//! - AT-1570: upload/dispatch/readback round-trip matches dispatch_handle outputs
//! - AT-1571: CB/fence/query-pool AND device-local buffers are REUSED across N dispatches
//! - AT-1572: readback Lever A — readonly bindings produce empty Vec
//! - AT-1579: dispatch_resident rejects out-of-range workgroups (WorkgroupCountExceedsDeviceLimit)

#[allow(unused_imports)]
use axc_driver::compile_source_with_meta;
#[allow(unused_imports)]
use axc_runtime::{
    VulkanContext, VulkanContextOptions, DispatchError,
    ResidentBenchConfig, ResidentTimingSource,
};

// ── AT-1530: resident dispatch span < host round-trip on GpuTimestamp ────────

/// AT-1530: On a timestamp-capable GPU, dispatch_resident produces a GpuTimestamp-sourced
/// kernel_ns (MIN of N>=5 post-warmup samples, 2 warm-ups discarded) that is DISTINCTLY
/// SMALLER than the host-round-trip dispatch_handle time for the same kernel.
///
/// On Lavapipe (timestampValidBits==0): timing_source==CpuFenceWall, assert only
/// kernel_ns>0 and print the CpuFenceWall qualifier (no smaller-than claim).
///
/// Warmup-discard is MANDATORY (TIMING-1 / HANDOFF-12).
#[test]
#[ignore]
fn at_1530_resident_dispatch_span_smaller_than_host_roundtrip() {
    if !axc_runtime::gpu_tests_enabled() { return; }

    const SAXPY_SRC: &str = include_str!("../../../examples/saxpy.axc");
    const N: usize = 1024;
    const N_WARMUP: usize = 2;
    const N_MEASURED: usize = 5; // N >= 5 (optimistic reviewer recommendation)

    let ctx = VulkanContext::new().expect("VulkanContext must create");

    let (bytes, meta) = compile_source_with_meta(SAXPY_SRC)
        .expect("saxpy must compile");
    let words: Vec<u32> = bytes.chunks_exact(4).map(|c| {
        u32::from_le_bytes([c[0], c[1], c[2], c[3]])
    }).collect();

    let handle = ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, meta.coopmat.as_ref(), "saxpy",
        meta.shared_memory_bytes,
    ).expect("saxpy prepare must succeed");

    // Prepare inputs.
    let alpha: f32 = 2.0_f32;
    let x: Vec<f32> = (0..N).map(|i| i as f32 * 0.1_f32).collect();
    let y: Vec<f32> = (0..N).map(|i| i as f32 * 0.5_f32).collect();
    let alpha_bytes = alpha.to_le_bytes();
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let y_bytes: Vec<u8> = y.iter().flat_map(|v| v.to_le_bytes()).collect();
    let n_bytes = (N as u32).to_le_bytes();
    let mut pc = Vec::new();
    pc.extend_from_slice(&n_bytes);
    pc.extend_from_slice(&alpha_bytes);

    let output_size = y_bytes.len() as u64;
    let inputs: &[&[u8]] = &[&x_bytes, &y_bytes];
    let output_sizes: &[u64] = &[0, output_size];

    // Measure host-round-trip dispatch_handle time as baseline.
    let t0 = std::time::Instant::now();
    let _ = ctx.dispatch_handle(
        &handle,
        ((N as u32).div_ceil(64), 1, 1),
        inputs,
        &[0, output_size as usize],
        &pc,
    ).expect("dispatch_handle must succeed");
    let host_round_trip_ns = t0.elapsed().as_nanos() as u64;

    // Upload resident buffers once.
    let resident = ctx.upload_resident(&handle, inputs, output_sizes)
        .expect("upload_resident must succeed");

    let workgroups = ((N as u32).div_ceil(64), 1, 1);
    let cfg = ResidentBenchConfig { workgroups, push_constants: pc.clone() };

    // MANDATORY warmup: 2 untimed dispatches discarded (TIMING-1 / HANDOFF-12).
    for _ in 0..N_WARMUP {
        let _ = ctx.dispatch_resident(&handle, &resident, &cfg)
            .expect("warmup dispatch must succeed");
    }

    // Measure N_MEASURED dispatches, report MIN.
    let mut min_ns: u64 = u64::MAX;
    let mut timing_source = ResidentTimingSource::CpuFenceWall;
    for _ in 0..N_MEASURED {
        let t = ctx.dispatch_resident(&handle, &resident, &cfg)
            .expect("measured dispatch must succeed");
        if t.kernel_ns < min_ns {
            min_ns = t.kernel_ns;
        }
        timing_source = t.timing_source;
    }

    assert!(min_ns > 0, "min kernel_ns must be > 0");

    // AT-1582 / AT-1530: conditional assertions based on timing source.
    match timing_source {
        ResidentTimingSource::GpuTimestamp => {
            eprintln!(
                "AT-1530 [GpuTimestamp]: saxpy min resident dispatch = {} ns vs host round-trip = {} ns",
                min_ns, host_round_trip_ns
            );
            // resident dispatch_ns must be distinctly smaller than host round-trip
            // (upload + readback excluded). Use 50% headroom to survive clock ramp.
            assert!(
                min_ns < host_round_trip_ns / 2,
                "GpuTimestamp min resident ({} ns) must be < 50% of host round-trip ({} ns); \
                 the resident path excludes upload/readback",
                min_ns, host_round_trip_ns
            );
        }
        ResidentTimingSource::CpuFenceWall => {
            // Lavapipe path: CpuFenceWall — scheduling-inclusive, NOT a GPU kernel time.
            // Only assert kernel_ns > 0 (no smaller-than claim on this path — AT-1582).
            eprintln!(
                "AT-1530 [CpuFenceWall — scheduling-inclusive, NOT a GPU kernel time]: \
                 saxpy min resident = {} ns",
                min_ns
            );
            assert!(min_ns > 0, "CpuFenceWall kernel_ns must be > 0");
        }
    }
}

// ── AT-1531: resident dispatch over existing real kernels ─────────────────────

/// AT-1531: dispatch_resident works for saxpy + coopmat matmul_tile (typed-skip
/// on Lavapipe). On coopmat-capable+subgroup32 device: kernel_ns > 0 finite.
/// On unsupported device: prepare_kernel_checked returns CoopMatUnsupported (no crash).
#[test]
#[ignore]
fn at_1531_resident_dispatch_existing_kernels() {
    if !axc_runtime::gpu_tests_enabled() { return; }

    const SAXPY_SRC: &str = include_str!("../../../examples/saxpy.axc");
    const MATMUL_TILE_SRC: &str = include_str!("../../../examples/matmul_tile.axc");
    const N: usize = 256;

    let ctx = VulkanContext::new().expect("VulkanContext must create");

    // Test 1: saxpy (always supported).
    {
        let (bytes, meta) = compile_source_with_meta(SAXPY_SRC).expect("saxpy compile");
        let words: Vec<u32> = bytes.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
        let handle = ctx.prepare_kernel_checked(
            &words, &meta.binding_plan, meta.push_constant_total_bytes,
            &meta.entry_point, None, "saxpy",
            meta.shared_memory_bytes,
        ).expect("saxpy prepare_kernel_checked must succeed");

        let alpha: f32 = 1.0_f32;
        let x: Vec<f32> = vec![1.0_f32; N];
        let y: Vec<f32> = vec![0.0_f32; N];
        let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        let y_bytes: Vec<u8> = y.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut pc = Vec::new();
        pc.extend_from_slice(&(N as u32).to_le_bytes());
        pc.extend_from_slice(&alpha.to_le_bytes());

        let output_size = (N * 4) as u64;
        let resident = ctx.upload_resident(
            &handle, &[&x_bytes, &y_bytes], &[0, output_size],
        ).expect("saxpy upload_resident must succeed");

        let cfg = ResidentBenchConfig {
            workgroups: ((N as u32).div_ceil(64), 1, 1),
            push_constants: pc,
        };
        let timing = ctx.dispatch_resident(&handle, &resident, &cfg)
            .expect("saxpy dispatch_resident must succeed");

        assert!(timing.kernel_ns > 0 && (timing.kernel_ns as f64).is_finite(),
            "saxpy kernel_ns must be > 0 and finite, got {}", timing.kernel_ns);

        let ts_label = match timing.timing_source {
            ResidentTimingSource::GpuTimestamp => "GpuTimestamp",
            ResidentTimingSource::CpuFenceWall => "CpuFenceWall — scheduling-inclusive, NOT a GPU kernel time",
        };
        eprintln!("AT-1531 saxpy: {} ns ({})", timing.kernel_ns, ts_label);
    }

    // Test 2: matmul_tile (coopmat) — typed-skip on Lavapipe/non-32-subgroup.
    {
        let (bytes, meta) = compile_source_with_meta(MATMUL_TILE_SRC)
            .expect("matmul_tile compile");
        let words: Vec<u32> = bytes.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();

        match ctx.prepare_kernel_checked(
            &words, &meta.binding_plan, meta.push_constant_total_bytes,
            &meta.entry_point, meta.coopmat.as_ref(), "matmul_tile",
            meta.shared_memory_bytes,
        ) {
            Ok(handle) => {
                let a_f16: Vec<u8> = vec![0u8; 512]; // 16x16 f16 zeros
                let b_f16: Vec<u8> = vec![0u8; 512];
                let c_f16: Vec<u8> = vec![0u8; 512];
                let pc: Vec<u8> = 0u32.to_le_bytes().to_vec();
                let output_size = 512u64;

                let resident = ctx.upload_resident(
                    &handle, &[&a_f16, &b_f16, &c_f16], &[0, 0, output_size],
                ).expect("matmul_tile upload_resident must succeed");

                let cfg = ResidentBenchConfig {
                    workgroups: (1, 1, 1),
                    push_constants: pc,
                };
                let timing = ctx.dispatch_resident(&handle, &resident, &cfg)
                    .expect("matmul_tile dispatch_resident must succeed");

                assert!(timing.kernel_ns > 0, "matmul_tile kernel_ns must be > 0");
                eprintln!("AT-1531 matmul_tile: {} ns ({:?})", timing.kernel_ns, timing.timing_source);
            }
            Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
                eprintln!("AT-1531 matmul_tile: CoopMatUnsupported (graceful skip): {reason}");
                // Graceful skip — no assertion needed; typed skip is the correct behavior.
            }
            Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
                eprintln!("AT-1531 matmul_tile: DeviceFeatureUnsupported {feature}/{kernel} (graceful skip)");
            }
            Err(e) => {
                panic!("AT-1531 matmul_tile: unexpected error (must be graceful skip or success): {e}");
            }
        }
    }
}

// ── AT-1570: resident round-trip matches dispatch_handle ─────────────────────

/// AT-1570: upload/dispatch/readback round-trip produces the SAME outputs as
/// dispatch_handle for a non-coopmat kernel WITH at least one ReadOnly binding
/// (so Lever A is exercised). AT-1572 (Lever A) is also covered here.
///
/// Bit-exact for this integer-free saxpy-like kernel (f32 accumulate is close
/// enough for 1e-3 relative tolerance, but saxpy outputs differ only by floating
/// point rounding which we check at 1e-3 relative).
#[test]
#[ignore]
fn at_1570_resident_roundtrip_matches_dispatch_handle() {
    if !axc_runtime::gpu_tests_enabled() { return; }

    const SAXPY_SRC: &str = include_str!("../../../examples/saxpy.axc");
    const N: usize = 128;

    let ctx = VulkanContext::new().expect("VulkanContext must create");
    let (bytes, meta) = compile_source_with_meta(SAXPY_SRC).expect("compile");
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
    let handle = ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, None, "saxpy",
        meta.shared_memory_bytes,
    ).expect("prepare");

    let alpha: f32 = 3.0_f32;
    let x: Vec<f32> = (0..N).map(|i| i as f32 * 0.1_f32).collect();
    let y: Vec<f32> = (0..N).map(|i| i as f32 * 0.5_f32).collect();
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let y_bytes: Vec<u8> = y.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut pc = Vec::new();
    pc.extend_from_slice(&(N as u32).to_le_bytes());
    pc.extend_from_slice(&alpha.to_le_bytes());

    let output_size = (N * 4) as u64;
    let workgroups = ((N as u32).div_ceil(64), 1, 1);

    // Reference: dispatch_handle.
    let ref_outputs = ctx.dispatch_handle(
        &handle, workgroups,
        &[&x_bytes, &y_bytes],
        &[0, output_size as usize],
        &pc,
    ).expect("dispatch_handle reference");

    // Resident path.
    let resident = ctx.upload_resident(
        &handle, &[&x_bytes, &y_bytes], &[0, output_size],
    ).expect("upload_resident");

    let cfg = ResidentBenchConfig { workgroups, push_constants: pc };
    ctx.dispatch_resident(&handle, &resident, &cfg).expect("dispatch_resident");

    let res_outputs = ctx.readback_resident(
        &handle, &resident, &[0, output_size],
    ).expect("readback_resident");

    // x is ReadOnly → Lever A → empty Vec.
    // AT-1572 (Lever A): readonly bindings produce empty Vec.
    assert!(res_outputs[0].is_empty(),
        "AT-1572: ReadOnly binding (x) must produce empty Vec in readback_resident");

    // y (output) must match dispatch_handle output.
    let ref_y = &ref_outputs[1];
    let res_y = &res_outputs[1];
    assert_eq!(ref_y.len(), res_y.len(), "output byte lengths must match");

    // Check values within 1e-3 relative tolerance.
    let n_vals = ref_y.len() / 4;
    for i in 0..n_vals {
        let ref_f = f32::from_le_bytes(ref_y[i*4..(i+1)*4].try_into().unwrap());
        let res_f = f32::from_le_bytes(res_y[i*4..(i+1)*4].try_into().unwrap());
        let rel_diff = if ref_f.abs() > 1e-6 { (ref_f - res_f).abs() / ref_f.abs() } else { (ref_f - res_f).abs() };
        assert!(
            rel_diff < 1e-3,
            "AT-1570: output[{i}] resident={res_f} vs dispatch_handle={ref_f} (rel diff={rel_diff:.2e} > 1e-3)"
        );
    }
    eprintln!("AT-1570: resident round-trip matches dispatch_handle ({n_vals} f32 values, tol 1e-3)");
}

// ── AT-1571: CB/fence/query-pool reused across N dispatches ──────────────────

/// AT-1571: dispatch_resident reuses one CB, fence, query pool, AND the same
/// device-local buffer handles across N>=4 iterations (no per-iteration allocation).
#[test]
#[ignore]
fn at_1571_resident_reuses_cb_fence_querypool() {
    if !axc_runtime::gpu_tests_enabled() { return; }

    const SAXPY_SRC: &str = include_str!("../../../examples/saxpy.axc");
    const N: usize = 64;
    const N_ITERS: usize = 4;

    let ctx = VulkanContext::new().expect("VulkanContext must create");
    let (bytes, meta) = compile_source_with_meta(SAXPY_SRC).expect("compile");
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
    let handle = ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, None, "saxpy",
        meta.shared_memory_bytes,
    ).expect("prepare");

    let x: Vec<f32> = vec![1.0_f32; N];
    let y: Vec<f32> = vec![0.0_f32; N];
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let y_bytes: Vec<u8> = y.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut pc = Vec::new();
    pc.extend_from_slice(&(N as u32).to_le_bytes());
    pc.extend_from_slice(&1.0_f32.to_le_bytes());

    let output_size = (N * 4) as u64;
    let resident = ctx.upload_resident(
        &handle, &[&x_bytes, &y_bytes], &[0, output_size],
    ).expect("upload_resident");

    // Capture handles before dispatching (using public accessor methods for AT-1571).
    let cb_before = resident.command_buffer_handle();
    let fence_before = resident.fence_handle();
    let qp_before = resident.query_pool_handle();
    let dl_bufs_before = resident.device_local_buf_handles();

    let workgroups = ((N as u32).div_ceil(64), 1, 1);
    let cfg = ResidentBenchConfig { workgroups, push_constants: pc };

    for _ in 0..N_ITERS {
        ctx.dispatch_resident(&handle, &resident, &cfg)
            .expect("dispatch must succeed");
    }

    // Verify handles are unchanged (no per-iter allocation).
    assert_eq!(resident.command_buffer_handle(), cb_before, "command_buffer handle must not change");
    assert_eq!(resident.fence_handle(), fence_before, "fence handle must not change");
    assert_eq!(resident.query_pool_handle(), qp_before, "query_pool handle must not change");
    let dl_bufs_after = resident.device_local_buf_handles();
    assert_eq!(dl_bufs_before, dl_bufs_after, "device_local_bufs handles must not change");

    eprintln!("AT-1571: CB/fence/query_pool/bufs all unchanged after {} dispatches", N_ITERS);
}

// ── AT-1572: readback Lever A standalone ─────────────────────────────────────

/// AT-1572: readback_resident honors Lever A — readonly bindings produce empty Vec.
/// (Also tested as part of AT-1570; this is a dedicated standalone test.)
#[test]
#[ignore]
fn at_1572_readback_resident_lever_a_skips_readonly() {
    if !axc_runtime::gpu_tests_enabled() { return; }

    const SAXPY_SRC: &str = include_str!("../../../examples/saxpy.axc");
    const N: usize = 32;

    let ctx = VulkanContext::new().expect("VulkanContext must create");
    let (bytes, meta) = compile_source_with_meta(SAXPY_SRC).expect("compile");
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
    let handle = ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, None, "saxpy",
        meta.shared_memory_bytes,
    ).expect("prepare");

    let x: Vec<f32> = vec![1.0_f32; N];
    let y: Vec<f32> = vec![2.0_f32; N];
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let y_bytes: Vec<u8> = y.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut pc = Vec::new();
    pc.extend_from_slice(&(N as u32).to_le_bytes());
    pc.extend_from_slice(&1.0_f32.to_le_bytes());

    let output_size = (N * 4) as u64;
    let resident = ctx.upload_resident(
        &handle, &[&x_bytes, &y_bytes], &[0, output_size],
    ).expect("upload_resident");

    let cfg = ResidentBenchConfig {
        workgroups: ((N as u32).div_ceil(64), 1, 1),
        push_constants: pc,
    };
    ctx.dispatch_resident(&handle, &resident, &cfg).expect("dispatch");

    let outputs = ctx.readback_resident(&handle, &resident, &[0, output_size])
        .expect("readback_resident");

    // Binding 0 (x) is readonly_buffer → Lever A → empty Vec.
    assert!(outputs[0].is_empty(),
        "AT-1572: ReadOnly binding (x) must produce empty Vec; got {} bytes", outputs[0].len());
    // Binding 1 (y) is output → must have data.
    assert_eq!(outputs[1].len(), output_size as usize,
        "AT-1572: output binding (y) must have {} bytes, got {}", output_size, outputs[1].len());

    eprintln!("AT-1572: Lever A correctly skips readonly binding");
}

// ── AT-1579: dispatch_resident rejects out-of-range workgroups ───────────────

/// AT-1579: dispatch_resident rejects an out-of-range cfg.workgroups with
/// WorkgroupCountExceedsDeviceLimit (library-side guard, CORRECTNESS-1).
///
/// This test requires upload_resident (needs a real GPU), hence `#[ignore]`.
/// The guard itself is exercised before any GPU work.
#[test]
#[ignore]
fn at_1579_dispatch_resident_rejects_oversize_workgroups() {
    if !axc_runtime::gpu_tests_enabled() { return; }

    const SAXPY_SRC: &str = include_str!("../../../examples/saxpy.axc");

    let ctx = VulkanContext::new().expect("VulkanContext must create");
    let (bytes, meta) = compile_source_with_meta(SAXPY_SRC).expect("compile");
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
    let handle = ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, None, "saxpy",
        meta.shared_memory_bytes,
    ).expect("prepare");

    let x: Vec<f32> = vec![1.0_f32; 64];
    let y: Vec<f32> = vec![0.0_f32; 64];
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let y_bytes: Vec<u8> = y.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut pc = Vec::new();
    pc.extend_from_slice(&64u32.to_le_bytes());
    pc.extend_from_slice(&1.0_f32.to_le_bytes());

    let output_size = (64 * 4) as u64;
    let resident = ctx.upload_resident(
        &handle, &[&x_bytes, &y_bytes], &[0, output_size],
    ).expect("upload_resident");

    // Out-of-range workgroup count (u32::MAX should exceed any device limit).
    let cfg = ResidentBenchConfig {
        workgroups: (u32::MAX, 1, 1),
        push_constants: pc,
    };

    let result = ctx.dispatch_resident(&handle, &resident, &cfg);
    assert!(
        matches!(result, Err(DispatchError::WorkgroupCountExceedsDeviceLimit { .. })),
        "AT-1579: out-of-range workgroups must return WorkgroupCountExceedsDeviceLimit; got: {result:?}"
    );
    eprintln!("AT-1579: WorkgroupCountExceedsDeviceLimit correctly returned for wg_x=u32::MAX");
}
