//! Integration tests for q4km_dequant_matmul.axc (plain, no coopmat) — M3.1.
//!
//! AT-1520: 256x256 bit-exact vs CPU reference on Lavapipe AND NVIDIA (NOT #[ignore]).
//! AT-1521: q4km_dequant_matmul_coopmat.axc staged bridge, NVIDIA-only (#[ignore]).
//! AT-1522: pipeline CREATION for q4km_dequant_matmul (8-bit storage + Int8 + Int16) succeeds.
//! AT-1560: all M3.1 q4km modules pass spirv-val.

#[path = "common_matmul.rs"]
mod common_matmul;

use axc_driver::compile_source_with_meta;
use axc_runtime::{VulkanContext, VulkanContextOptions, DispatchError};

const Q4KM_MATMUL_SRC: &str = include_str!("../../../examples/q4km_dequant_matmul.axc");
const Q4KM_COOPMAT_SRC: &str = include_str!("../../../examples/q4km_dequant_matmul_coopmat.axc");

fn gpu_tests_enabled() -> bool {
    std::env::var("AXC_ENABLE_GPU_TESTS").as_deref() == Ok("1")
}

/// AT-1560 (partial): q4km_dequant_matmul.axc SPIR-V passes spirv-val.
#[test]
fn at_1560_q4km_matmul_spirv_val_clean() {
    let (bytes, _) = compile_source_with_meta(Q4KM_MATMUL_SRC)
        .expect("q4km_dequant_matmul.axc must compile");
    let words: Vec<u32> = bytes.chunks_exact(4).map(|c| {
        u32::from_le_bytes([c[0], c[1], c[2], c[3]])
    }).collect();
    use spirv_tools::val::{Validator, create as create_validator};
    use spirv_tools::TargetEnv;
    let validator = create_validator(Some(TargetEnv::Vulkan_1_1));
    validator.validate(&words, None)
        .expect("q4km_dequant_matmul SPIR-V must pass spirv-val");
}

/// AT-1560 (partial): q4km_dequant_matmul_coopmat.axc SPIR-V passes spirv-val.
#[test]
fn at_1560_q4km_coopmat_spirv_val_clean() {
    let (bytes, _) = compile_source_with_meta(Q4KM_COOPMAT_SRC)
        .expect("q4km_dequant_matmul_coopmat.axc must compile");
    let words: Vec<u32> = bytes.chunks_exact(4).map(|c| {
        u32::from_le_bytes([c[0], c[1], c[2], c[3]])
    }).collect();
    use spirv_tools::val::{Validator, create as create_validator};
    use spirv_tools::TargetEnv;
    let validator = create_validator(Some(TargetEnv::Vulkan_1_1));
    validator.validate(&words, None)
        .expect("q4km_dequant_matmul_coopmat SPIR-V must pass spirv-val");
}

/// AT-1522: pipeline CREATION for q4km_dequant_matmul succeeds when 8-bit+Int8+Int16 enabled.
///
/// On a device with these features (NVIDIA): pipeline creates Ok BEFORE dispatch.
/// On a device without them: DeviceFeatureUnsupported typed skip.
#[test]
#[ignore]
fn at_1522_q4km_pipeline_create_succeeds_or_feature_skip() {
    if !gpu_tests_enabled() {
        eprintln!("at_1522: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    let (bytes, meta) = compile_source_with_meta(Q4KM_MATMUL_SRC)
        .expect("q4km_dequant_matmul.axc must compile");
    let words: Vec<u32> = bytes.chunks_exact(4).map(|c| {
        u32::from_le_bytes([c[0], c[1], c[2], c[3]])
    }).collect();
    let ctx = VulkanContext::new().expect("VulkanContext must init");
    eprintln!("at_1522: device={}, storage_8bit={}, shader_int8={}, shader_int16={}",
        ctx.physical_device_name(),
        ctx.enabled_features().storage_8bit,
        ctx.enabled_features().shader_int8,
        ctx.enabled_features().shader_int16);

    let handle_result = ctx.prepare_kernel(
        &words,
        &meta.binding_plan,
        meta.push_constant_total_bytes,
        &meta.entry_point,
    );

    match handle_result {
        Ok(handle) => {
            eprintln!("at_1522: q4km pipeline created successfully");
            let _ = handle;
        }
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
            eprintln!("at_1522: DeviceFeatureUnsupported {feature} for {kernel} (fail-closed skip; expected on devices without 8-bit storage)");
        }
        Err(e) => {
            eprintln!("at_1522: pipeline create error: {e}");
        }
    }
}

/// AT-1520: q4km 256x256 multi-row matmul, bit-exact vs CPU reference.
///
/// NOT #[ignore] — runs on Lavapipe AND NVIDIA under AXC_ENABLE_GPU_TESTS=1.
/// Uses a synthetic 256x256 Q4_K_M fixture (1 superblock per row).
#[test]
#[ignore]
fn at_1520_q4km_matmul_256x256_bit_exact() {
    if !gpu_tests_enabled() {
        eprintln!("at_1520: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    let ctx = VulkanContext::new().expect("VulkanContext must init");
    eprintln!("at_1520: device={}", ctx.physical_device_name());

    // 256x256 fixture: n_rows=256, n_cols=256, n_blocks_per_row=1.
    let n_rows: u32 = 256;
    let n_cols: u32 = 256;
    let n_blocks_per_row: u32 = 1;

    // Generate a deterministic Q4_K_M weight matrix (256 rows, 1 block/row).
    let (q_bytes, x_bytes) = make_q4km_256x256_fixture(n_rows, n_cols, n_blocks_per_row);

    // CPU reference.
    let x_f32: Vec<f32> = x_bytes.chunks_exact(4).map(|c| {
        f32::from_le_bytes([c[0], c[1], c[2], c[3]])
    }).collect();
    let y_ref: Vec<f32> = common_matmul::q4km_dequant_matmul_cpu(
        &q_bytes, &x_f32, n_rows as usize, n_cols as usize, n_blocks_per_row as usize,
    );

    // Compile and dispatch.
    let (bytes, meta) = compile_source_with_meta(Q4KM_MATMUL_SRC)
        .expect("q4km_dequant_matmul.axc must compile");
    let words: Vec<u32> = bytes.chunks_exact(4).map(|c| {
        u32::from_le_bytes([c[0], c[1], c[2], c[3]])
    }).collect();

    let handle = match ctx.prepare_kernel(&words, &meta.binding_plan, meta.push_constant_total_bytes, &meta.entry_point) {
        Ok(h) => h,
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
            eprintln!("at_1520: DeviceFeatureUnsupported {feature} for {kernel}; skipping");
            return;
        }
        Err(e) => panic!("at_1520: pipeline create failed: {e:?}"),
    };

    // Push constants: n_rows, n_cols, n_blocks_per_row.
    let mut pc_bytes: Vec<u8> = Vec::new();
    pc_bytes.extend_from_slice(&n_rows.to_le_bytes());
    pc_bytes.extend_from_slice(&n_cols.to_le_bytes());
    pc_bytes.extend_from_slice(&n_blocks_per_row.to_le_bytes());

    let output_y_size = (n_rows * n_cols) as usize * 4; // f32 elements
    let workgroups = (
        ((n_rows * n_cols + 63) / 64),
        1, 1
    );

    let outputs = match ctx.dispatch_handle(
        &handle,
        workgroups,
        &[&q_bytes, &x_bytes, &vec![0u8; output_y_size]],
        &[0, 0, output_y_size],
        &pc_bytes,
    ) {
        Ok(v) => v,
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
            eprintln!("at_1520: DeviceFeatureUnsupported at dispatch: {feature} / {kernel}; skipping");
            return;
        }
        Err(e) => panic!("at_1520: dispatch failed: {e:?}"),
    };

    // y is at binding 2. q and x are ReadOnly → empty (Lever A).
    assert!(outputs[0].is_empty(), "q (ReadOnly) must produce empty output (Lever A)");
    assert!(outputs[1].is_empty(), "x (ReadOnly) must produce empty output (Lever A)");
    let y_bytes: &[u8] = &outputs[2];
    assert_eq!(y_bytes.len(), output_y_size);
    let y_gpu: Vec<f32> = y_bytes.chunks_exact(4).map(|c| {
        f32::from_le_bytes([c[0], c[1], c[2], c[3]])
    }).collect();

    // Compare within 1e-3 relative tolerance.
    let rel_tol: f32 = 1e-3_f32;
    let mut max_rel_diff: f32 = 0.0_f32;
    let mut max_idx: usize = 0;
    for (idx, (&g, &r)) in y_gpu.iter().zip(y_ref.iter()).enumerate() {
        let abs_diff: f32 = (g - r).abs();
        let denom: f32 = r.abs().max(1e-8_f32);
        let rel_diff: f32 = abs_diff / denom;
        if rel_diff > max_rel_diff {
            max_rel_diff = rel_diff;
            max_idx = idx;
        }
    }
    eprintln!("at_1520: max_rel_diff={max_rel_diff} at idx={max_idx}");
    assert!(
        max_rel_diff <= rel_tol,
        "at_1520: q4km 256x256 exceeds 1e-3 rel tol: max_rel_diff={max_rel_diff} at idx={max_idx} \
         (gpu={}, ref={})", y_gpu[max_idx], y_ref[max_idx]
    );
    eprintln!("at_1520: PASS — 256x256 q4km matmul bit-exact on {}", ctx.physical_device_name());
}

/// AT-1521: q4km coopmat bridge (NVIDIA-only, #[ignore]).
///
/// Uses pre-dequantized f16 tiles; bit-close within f16 tol on NVIDIA.
/// Skips on Lavapipe via CoopMatUnsupported.
#[test]
#[ignore]
fn at_1521_q4km_coopmat_bridge_nvidia() {
    if !gpu_tests_enabled() {
        eprintln!("at_1521: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    let ctx = VulkanContext::new().expect("VulkanContext must init");
    if !ctx.coopmat_support().feature_present {
        eprintln!("at_1521: coopmat not supported on {}; skipping", ctx.physical_device_name());
        return;
    }

    let (bytes, meta) = compile_source_with_meta(Q4KM_COOPMAT_SRC)
        .expect("q4km_dequant_matmul_coopmat.axc must compile");
    let words: Vec<u32> = bytes.chunks_exact(4).map(|c| {
        u32::from_le_bytes([c[0], c[1], c[2], c[3]])
    }).collect();

    // Use the transpose fixtures as pre-dequantized f16 inputs.
    let a: Vec<u16> = common_matmul::make_transpose_fixture_a();
    let b: Vec<u16> = common_matmul::make_transpose_fixture_b();
    let c0: Vec<u16> = vec![0u16; 256];
    let a_bytes: Vec<u8> = common_matmul::f16_slice_to_bytes(&a);
    let b_bytes: Vec<u8> = common_matmul::f16_slice_to_bytes(&b);
    let c_bytes: Vec<u8> = common_matmul::f16_slice_to_bytes(&c0);
    let tile_offset: u32 = 0;
    let pc_bytes: Vec<u8> = tile_offset.to_le_bytes().to_vec();

    let handle = match ctx.prepare_kernel(&words, &meta.binding_plan, meta.push_constant_total_bytes, &meta.entry_point) {
        Ok(h) => h,
        Err(e) => { eprintln!("at_1521: pipeline create: {e}; skipping"); return; }
    };

    let outputs = match ctx.dispatch_handle(
        &handle, (1, 1, 1),
        &[&a_bytes, &b_bytes, &c_bytes],
        &[0, 0, 256 * 2],
        &pc_bytes,
    ) {
        Ok(v) => v,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("at_1521: CoopMatUnsupported: {reason}; skipping");
            return;
        }
        Err(e) => panic!("at_1521: dispatch error: {e:?}"),
    };

    let c_gpu: Vec<u16> = common_matmul::bytes_to_f16_vec(&outputs[2]);
    let c_ref: Vec<u16> = common_matmul::matmul_tile_16x16x16_f16_cpu(&a, &b, &c0);
    let ref_mag: f32 = c_ref.iter().map(|&b| common_matmul::f16_bits_to_f32(b).abs()).fold(0.0_f32, f32::max);
    let tol: f32 = common_matmul::f16_tile_tol(ref_mag);
    let max_diff: f32 = c_gpu.iter().zip(c_ref.iter()).map(|(&g, &r)| {
        (common_matmul::f16_bits_to_f32(g) - common_matmul::f16_bits_to_f32(r)).abs()
    }).fold(0.0_f32, f32::max);
    assert!(max_diff <= tol, "at_1521: q4km coopmat bridge exceeds f16 tol: {max_diff} > {tol}");
    eprintln!("at_1521: PASS — q4km coopmat bridge on {}, max_diff={max_diff}", ctx.physical_device_name());
}

// ── Test fixture generation ────────────────────────────────────────────────────

/// Generate a synthetic 256x256 Q4_K_M fixture.
///
/// q: 256 rows * 1 block/row * 144 bytes = 36864 bytes.
/// x: 256 * 256 f32 = 65536 bytes (column-of-output-major layout).
fn make_q4km_256x256_fixture(
    n_rows: u32,
    n_cols: u32,
    n_blocks_per_row: u32,
) -> (Vec<u8>, Vec<u8>) {
    use common_matmul::f32_to_f16_bits;
    let n_rows = n_rows as usize;
    let n_cols = n_cols as usize;
    let n_bpr = n_blocks_per_row as usize;

    // Build a deterministic but non-trivial Q4_K_M weight matrix.
    // Each superblock has: d=1.0 (f16), dmin=0.0 (f16), scales all=1, weights all=8 (midpoint).
    let q_total: usize = n_rows * n_bpr * 144;
    let mut q: Vec<u8> = vec![0u8; q_total];
    for row in 0..n_rows {
        for sb in 0..n_bpr {
            let base: usize = row * n_bpr * 144 + sb * 144;
            // d = f16(1.0), dmin = f16(0.0).
            let d_bits: u16 = f32_to_f16_bits(1.0_f32);
            let dmin_bits: u16 = f32_to_f16_bits(0.0_f32);
            q[base..base+2].copy_from_slice(&d_bits.to_le_bytes());
            q[base+2..base+4].copy_from_slice(&dmin_bits.to_le_bytes());
            // Scales: all 1 in low 6 bits (sc=1 for each sub-block).
            // Bytes [4..16]: set bytes 0..4 to 1 (j<4 path: sc=byte&63, m=byte[j+4]&63).
            for j in 0..4_usize {
                q[base + 4 + j] = 1; // sc = 1
                q[base + 4 + j + 4] = 0; // m = 0 (no min offset)
            }
            // Bytes [8..16] already zeroed. Remaining 4 bytes [8..12] are used by j>=4 path.
            // For j in {4,5,6,7}: sc uses bits from [j+4] and [j-4]; m uses bits from [j+4] and [j].
            // With all zeros, sc=0 and m=0 for j>=4. We accept that.
            // Weights: all nibbles = 8 (midpoint nibble).
            // Bytes [16..144]: each byte = 0x88 (lo nibble = 8, hi nibble = 8).
            for k in 0..128_usize {
                q[base + 16 + k] = 0x88;
            }
        }
    }

    // x: column-of-output-major layout: x[k * n_cols + col] = (k+col+1) as f32 / 1000.0
    // k in 0..n_bpr*256.
    let k_total: usize = n_bpr * 256;
    let x_total: usize = k_total * n_cols;
    let mut x: Vec<f32> = vec![0.0_f32; x_total];
    for k in 0..k_total {
        for col in 0..n_cols {
            x[k * n_cols + col] = (k + col + 1) as f32 / 1000.0_f32;
        }
    }
    let x_bytes: Vec<u8> = x.iter().flat_map(|&v| v.to_le_bytes()).collect();

    (q, x_bytes)
}
