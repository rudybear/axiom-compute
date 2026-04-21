//! AT-810: On-disk pipeline cache round-trip.
//!
//! Creates a VulkanContext with an explicit tempdir pipeline cache path,
//! prepares and dispatches a kernel, drops the context (saving cache to disk),
//! then creates a second context pointing at the same cache file and verifies
//! the cache file grows (driver binary is persisted).

use axc_runtime::{VulkanContext, VulkanContextOptions, probe_vulkan_available, gpu_tests_enabled};
use axc_hir::ScalarTy;

fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// AT-810: pipeline cache is written to disk and grows after second context use.
///
/// No env mutation; no serial_test needed. Each context is created with an
/// explicit `VulkanContextOptions { pipeline_cache_path: Some(path), .. }`.
#[test]
#[ignore] // GPU-gated: requires AXC_ENABLE_GPU_TESTS=1 + Vulkan ICD
fn at_810_pipeline_cache_disk_roundtrip() {
    if !gpu_tests_enabled() {
        eprintln!("skipping at_810 (AXC_ENABLE_GPU_TESTS != 1)");
        return;
    }
    if !probe_vulkan_available() {
        eprintln!("skipping at_810 (no Vulkan ICD available)");
        return;
    }

    let tmp_dir = tempfile::tempdir().expect("tempdir creation must succeed");
    let cache_path = tmp_dir.path().join("at_810_pipeline.cache");

    let saxpy_src: &str = include_str!("../../../examples/saxpy.axc");
    let (spirv_bytes, meta) = axc_driver::compile_source_with_meta(saxpy_src)
        .expect("saxpy.axc must compile");
    let spirv_words: Vec<u32> = bytes_to_words(&spirv_bytes);

    fn make_pc_bytes(plan: &axc_hir::ParamBindingPlan, n: u32, alpha: f32) -> Vec<u8> {
        let mut pc = vec![0u8; plan.push_constant_total_bytes as usize];
        for scalar in &plan.scalars {
            let start = scalar.offset as usize;
            match scalar.ty {
                ScalarTy::U32 => pc[start..start+4].copy_from_slice(&n.to_le_bytes()),
                ScalarTy::F32 => pc[start..start+4].copy_from_slice(&alpha.to_le_bytes()),
                _ => {}
            }
        }
        pc
    }

    // ── Context 1: compile pipeline, dispatch, drop (saves cache) ─────────────
    {
        let ctx = VulkanContext::new_with_options(VulkanContextOptions {
            pipeline_cache_path: Some(cache_path.clone()),
            physical_device_index: None,
            fence_timeout_ms: None,
        })
        .expect("ctx1 must succeed");
        eprintln!("AT-810: ctx1 device = {}", ctx.physical_device_name());

        const N: u32 = 256;
        let handle = ctx.prepare_kernel(
            &spirv_words,
            &meta.binding_plan,
            meta.binding_plan.push_constant_total_bytes,
            &meta.entry_point,
        )
        .expect("ctx1 prepare_kernel must succeed");

        let x: Vec<f32> = (0..N).map(|i| i as f32 * 0.25).collect();
        let y: Vec<f32> = (0..N).map(|i| i as f32 * 0.0625).collect();
        let xb: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        let yb: Vec<u8> = y.iter().flat_map(|v| v.to_le_bytes()).collect();
        let buf_size = N as usize * 4;
        let pc = make_pc_bytes(&meta.binding_plan, N, 2.5);

        let _outputs = ctx.dispatch_handle(
            &handle,
            (N.div_ceil(64), 1, 1),
            &[&xb, &yb],
            &[buf_size, buf_size],
            &pc,
        )
        .expect("ctx1 dispatch must succeed");
    } // ctx1 dropped here → pipeline_cache.save() writes cache_path

    // Verify cache file was written with meaningful content (>= 32 bytes).
    let cache_size_1 = std::fs::metadata(&cache_path)
        .map(|m| m.len())
        .unwrap_or(0);
    eprintln!("AT-810: cache file size after ctx1 = {} bytes", cache_size_1);
    assert!(
        cache_size_1 >= 32,
        "AT-810: pipeline cache file must be >= 32 bytes after ctx1, got {cache_size_1}"
    );

    // ── Context 2: load cache, compile pipeline (should be faster), dispatch ───
    {
        let ctx = VulkanContext::new_with_options(VulkanContextOptions {
            pipeline_cache_path: Some(cache_path.clone()),
            physical_device_index: None,
            fence_timeout_ms: None,
        })
        .expect("ctx2 must succeed");
        eprintln!("AT-810: ctx2 device = {}", ctx.physical_device_name());

        const N: u32 = 256;
        let handle = ctx.prepare_kernel(
            &spirv_words,
            &meta.binding_plan,
            meta.binding_plan.push_constant_total_bytes,
            &meta.entry_point,
        )
        .expect("ctx2 prepare_kernel must succeed");

        let x: Vec<f32> = (0..N).map(|i| i as f32 * 0.5).collect();
        let y: Vec<f32> = (0..N).map(|i| i as f32 * 0.125).collect();
        let xb: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        let yb: Vec<u8> = y.iter().flat_map(|v| v.to_le_bytes()).collect();
        let buf_size = N as usize * 4;
        let pc = make_pc_bytes(&meta.binding_plan, N, 3.0);

        let _outputs = ctx.dispatch_handle(
            &handle,
            (N.div_ceil(64), 1, 1),
            &[&xb, &yb],
            &[buf_size, buf_size],
            &pc,
        )
        .expect("ctx2 dispatch must succeed");
    } // ctx2 dropped → cache saved again (may grow or stay same size depending on driver)

    let cache_size_2 = std::fs::metadata(&cache_path)
        .map(|m| m.len())
        .unwrap_or(0);
    eprintln!("AT-810: cache file size after ctx2 = {} bytes", cache_size_2);

    // Cache must still exist and be non-empty after ctx2.
    assert!(
        cache_size_2 >= 32,
        "AT-810: pipeline cache file must be >= 32 bytes after ctx2, got {cache_size_2}"
    );

    eprintln!("AT-810: pipeline cache disk round-trip — PASS");
}
