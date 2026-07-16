//! M4.2a — AT-2986: GPU bit-exact test for the `coopmat_store_col` codegen feature.
//!
//! Dispatches TWO minimal single-tile coopmat kernels (byte-copies of `examples/matmul_tile.axc`,
//! one using `coopmat_store`, the other `coopmat_store_col`) on the SAME A/B fixture, and asserts
//! the column-major output is the EXACT TRANSPOSE of the row-major output:
//! `D_col[c*stride + r] == D_row[r*stride + c]` for all `r, c` in `0..16` (bit-exact, not a
//! tolerance — both stores write the SAME accumulator VALUES computed by the SAME
//! `coopmat_mul_add`; only the memory destination differs).
//!
//! NVIDIA #[ignore]-gated (AXC_ENABLE_GPU_TESTS=1). Typed-skip on Lavapipe
//! (CoopMatUnsupported / DeviceFeatureUnsupported) and on subgroup_size() != 32.

use axc_driver::compile_source_with_meta;
use axc_runtime::{VulkanContext, DispatchError};

/// Row-major sibling — byte-identical to `examples/matmul_tile.axc`.
const ROW_MAJOR_SRC: &str = r#"
@kernel
@workgroup(32,1,1)
@cooperative_matrix
@intent("AT-2986: row-major coopmat_store fixture")
fn at2986_store_row(
    tile_offset: u32,
    a_buf: readonly_buffer[f16],
    b_buf: readonly_buffer[f16],
    c_buf: buffer[f16]
) -> void {
    let stride: u32 = 16u32;
    let a: matrix[f16, 16, 16, a] = coopmat_load(a_buf, tile_offset, stride);
    let b: matrix[f16, 16, 16, b] = coopmat_load(b_buf, tile_offset, stride);
    let acc: matrix[f16, 16, 16, accumulator] = coopmat_zero();
    let result: matrix[f16, 16, 16, accumulator] = coopmat_mul_add(a, b, acc);
    coopmat_store(result, c_buf, tile_offset, stride);
    return;
}
"#;

/// Column-major sibling — the ONLY delta is `coopmat_store` -> `coopmat_store_col` (M4.2a).
const COL_MAJOR_SRC: &str = r#"
@kernel
@workgroup(32,1,1)
@cooperative_matrix
@intent("AT-2986: column-major coopmat_store_col fixture")
fn at2986_store_col(
    tile_offset: u32,
    a_buf: readonly_buffer[f16],
    b_buf: readonly_buffer[f16],
    c_buf: buffer[f16]
) -> void {
    let stride: u32 = 16u32;
    let a: matrix[f16, 16, 16, a] = coopmat_load(a_buf, tile_offset, stride);
    let b: matrix[f16, 16, 16, b] = coopmat_load(b_buf, tile_offset, stride);
    let acc: matrix[f16, 16, 16, accumulator] = coopmat_zero();
    let result: matrix[f16, 16, 16, accumulator] = coopmat_mul_add(a, b, acc);
    coopmat_store_col(result, c_buf, tile_offset, stride);
    return;
}
"#;

fn gpu_tests_enabled() -> bool {
    std::env::var("AXC_ENABLE_GPU_TESTS").as_deref() == Ok("1")
}

/// Deterministic, NON-symmetric 16x16 f16 fixture (xorshift), scaled to [-1,1] so a transpose
/// bug is distinguishable from a correct store (a symmetric fixture would hide it).
fn make_tile_f16(seed: u64) -> Vec<u16> {
    use half::f16;
    let mut state = seed | 1;
    let mut out = Vec::with_capacity(256);
    for idx in 0..256usize {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let v = (((state % 2000) as f32) / 1000.0 - 1.0) + (idx as f32) * 0.0011;
        out.push(f16::from_f32(v).to_bits());
    }
    out
}

/// Dispatch one of the two fixture kernels; returns the 256-element f16-bits output, or None on
/// a typed-skip.
fn dispatch_one(at: &str, src: &str, kernel_name: &str, a_bytes: &[u8], b_bytes: &[u8], ctx: &VulkanContext) -> Option<Vec<u16>> {
    let (bytes, meta) = compile_source_with_meta(src)
        .unwrap_or_else(|e| panic!("{at}: {kernel_name} must compile: {e:?}"));
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let handle = match ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, meta.coopmat.as_ref(), kernel_name, meta.shared_memory_bytes,
    ) {
        Ok(h) => h,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("{at}: {kernel_name} CoopMatUnsupported (typed-skip): {reason}");
            return None;
        }
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
            eprintln!("{at}: {kernel_name} DeviceFeatureUnsupported {feature}/{kernel} (typed-skip)");
            return None;
        }
        Err(e) => panic!("{at}: {kernel_name} prepare_kernel_checked failed: {e:?}"),
    };

    // tile_offset (u32 scalar) push constant — the plan has exactly one scalar.
    let mut pc: Vec<u8> = vec![0u8; meta.push_constant_total_bytes as usize];
    for s in &meta.binding_plan.scalars {
        if s.name == "tile_offset" {
            let start = s.offset as usize;
            pc[start..start + 4].copy_from_slice(&0u32.to_le_bytes());
        }
    }

    let c_size: usize = 256 * 2; // 256 f16 elements.
    let outputs = match ctx.dispatch_handle(
        &handle, (1, 1, 1),
        &[a_bytes, b_bytes, &vec![0u8; c_size]],
        &[0, 0, c_size],
        &pc,
    ) {
        Ok(v) => v,
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("{at}: {kernel_name} CoopMatUnsupported at dispatch (typed-skip): {reason}");
            return None;
        }
        Err(e) => panic!("{at}: {kernel_name} dispatch failed: {e:?}"),
    };

    let c_bytes = &outputs[2];
    assert_eq!(c_bytes.len(), c_size, "{at}: {kernel_name} C output size mismatch");
    Some(c_bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
}

/// AT-2986: `coopmat_store_col` output is the EXACT TRANSPOSE of `coopmat_store`'s.
#[test]
#[ignore]
fn at_2986_coopmat_store_col_is_transpose_of_row_major() {
    if !gpu_tests_enabled() {
        eprintln!("at_2986: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    let ctx = VulkanContext::new().expect("VulkanContext must init");
    eprintln!("at_2986: device={}", ctx.physical_device_name());

    if !ctx.coopmat_support().feature_present {
        eprintln!("at_2986: coopmat not supported on {}; typed-skip", ctx.physical_device_name());
        return;
    }
    if ctx.subgroup_size() != 32 {
        eprintln!("at_2986: subgroup_size()={} != 32; typed-skip", ctx.subgroup_size());
        return;
    }

    let a_f16 = make_tile_f16(0xC0FFEE);
    let b_f16 = make_tile_f16(0xBADF00D);
    let a_bytes: Vec<u8> = a_f16.iter().flat_map(|&v| v.to_le_bytes()).collect();
    let b_bytes: Vec<u8> = b_f16.iter().flat_map(|&v| v.to_le_bytes()).collect();

    let Some(d_row) = dispatch_one("at_2986", ROW_MAJOR_SRC, "at2986_store_row", &a_bytes, &b_bytes, &ctx) else { return; };
    let Some(d_col) = dispatch_one("at_2986", COL_MAJOR_SRC, "at2986_store_col", &a_bytes, &b_bytes, &ctx) else { return; };

    assert_eq!(d_row.len(), 256, "at_2986: row-major output must have 256 elements");
    assert_eq!(d_col.len(), 256, "at_2986: col-major output must have 256 elements");

    let stride: usize = 16;
    let mut n_diff = 0usize;
    let mut first_diff: Option<(usize, usize)> = None;
    for r in 0..stride {
        for c in 0..stride {
            let row_bits = d_row[r * stride + c];
            let col_bits = d_col[c * stride + r];
            if row_bits != col_bits {
                n_diff += 1;
                if first_diff.is_none() {
                    first_diff = Some((r, c));
                }
            }
        }
    }

    assert!(
        n_diff == 0,
        "AT-2986: coopmat_store_col output is NOT the bit-exact transpose of coopmat_store's: \
         {n_diff} of 256 elements differ; first at (r,c)={:?}. Both kernels compute the IDENTICAL \
         accumulator (same coopmat_mul_add on the same A/B tile); only the store destination \
         layout differs, so D_col[c*stride+r] MUST equal D_row[r*stride+c] bit-for-bit.",
        first_diff
    );

    eprintln!(
        "at_2986: PASS — coopmat_store_col output is the exact transpose of coopmat_store's \
         (256 elements, 0 diffs) on {}",
        ctx.physical_device_name()
    );
}
