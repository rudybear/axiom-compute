//! AT-1560: spirv-val clean on ALL emitted M3.1 modules.
//!
//! matmul_tile (coopmat), q4km_dequant_matmul (plain), q4km_dequant_matmul_coopmat (coopmat),
//! matmul_f32_tiled — all must pass spirv-tools Vulkan_1_1 validation.

use axc_driver::compile_source_with_meta;
use spirv_tools::val::{Validator, create as create_validator};
use spirv_tools::TargetEnv;

fn validate_src(label: &str, src: &str) {
    let (bytes, _) = compile_source_with_meta(src)
        .unwrap_or_else(|e| panic!("at_1560: {label} compile failed: {e}"));
    let words: Vec<u32> = bytes.chunks_exact(4).map(|c| {
        u32::from_le_bytes([c[0], c[1], c[2], c[3]])
    }).collect();
    let validator = create_validator(Some(TargetEnv::Vulkan_1_1));
    validator.validate(&words, None)
        .unwrap_or_else(|e| panic!("at_1560: {label} spirv-val FAILED: {e}"));
}

/// AT-1560: matmul_tile (coopmat) passes spirv-val.
#[test]
fn at_1560_matmul_tile_spirv_val() {
    validate_src("matmul_tile", include_str!("../../../examples/matmul_tile.axc"));
}

/// AT-1560: q4km_dequant_matmul (plain) passes spirv-val.
#[test]
fn at_1560_q4km_dequant_matmul_spirv_val() {
    validate_src("q4km_dequant_matmul", include_str!("../../../examples/q4km_dequant_matmul.axc"));
}

/// AT-1560: q4km_dequant_matmul_coopmat (coopmat bridge) passes spirv-val.
#[test]
fn at_1560_q4km_dequant_matmul_coopmat_spirv_val() {
    validate_src("q4km_dequant_matmul_coopmat", include_str!("../../../examples/q4km_dequant_matmul_coopmat.axc"));
}

/// AT-1560 / AT-1576: matmul_f32_tiled (no coopmat, no @strategy holes) passes spirv-val.
///
/// The @strategy { tile_m, tile_n } holes were removed (M3.1.5 / HANDOFF-0) because
/// they were inert (never referenced in the kernel body). This test now compiles via
/// compile_source_with_meta (no assignments needed), consistent with AT-1576.
#[test]
fn at_1560_matmul_f32_tiled_spirv_val() {
    let src = include_str!("../../../examples/matmul_f32_tiled.axc");
    // No @strategy block remains — compile_source_with_meta is sufficient (AT-1576).
    validate_src("matmul_f32_tiled (no strategy holes)", src);
}
