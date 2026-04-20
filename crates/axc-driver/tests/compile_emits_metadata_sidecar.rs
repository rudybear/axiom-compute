//! AT-510, AT-522: axc_driver::compile_file produces both .spv and .axc.meta.json sidecar.
//!
//! AT-510: saxpy + vector_add metadata sidecar emission.
//! AT-522: CLAUDE.md and DESIGN.md document M1.5 runtime dispatch.

use axc_runtime::{load_kernel_metadata, CURRENT_SCHEMA_VERSION};
use std::path::PathBuf;

/// AT-510: compile_file emits both .spv and .axc.meta.json for saxpy and vector_add.
///
/// Verifies that:
/// 1. The .spv file is written.
/// 2. The .axc.meta.json sidecar is written.
/// 3. The sidecar parses into a valid `KernelMetadata`.
/// 4. The binding plan matches expected layout.
#[test]
fn compile_saxpy_emits_metadata_sidecar() {
    let tmp_dir = std::env::temp_dir();

    // ── saxpy ────────────────────────────────────────────────────────────────
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let saxpy_src_path = PathBuf::from(&manifest_dir)
        .join("..")
        .join("..")
        .join("examples")
        .join("saxpy.axc");
    let saxpy_spv_path = tmp_dir.join("at_510_saxpy.spv");
    let saxpy_meta_path = tmp_dir.join("at_510_saxpy.spv.axc.meta.json");

    // Compile saxpy.axc → .spv + .axc.meta.json
    axc_driver::compile_file(&saxpy_src_path, &saxpy_spv_path)
        .unwrap_or_else(|e| panic!("compile_file(saxpy.axc) failed: {e}"));

    // Verify .spv was written.
    assert!(saxpy_spv_path.exists(), ".spv file must exist after compile_file");
    let spv_bytes = std::fs::read(&saxpy_spv_path).expect("read .spv");
    assert_eq!(&spv_bytes[0..4], &[0x03, 0x02, 0x23, 0x07], "SPIR-V magic mismatch");

    // Verify sidecar was written.
    assert!(saxpy_meta_path.exists(), ".axc.meta.json sidecar must exist after compile_file");

    // Load and verify sidecar content.
    let meta = load_kernel_metadata(&saxpy_meta_path)
        .unwrap_or_else(|e| panic!("load_kernel_metadata failed for saxpy: {e}"));

    assert_eq!(meta.schema_version, CURRENT_SCHEMA_VERSION);
    assert_eq!(meta.kernel_name, "saxpy");
    assert_eq!(meta.workgroup_size, [64, 1, 1]);
    // Entry point name equals kernel name (codegen writes `OpEntryPoint
    // GLCompute %main "saxpy"`); runtime passes this to vkCreateComputePipelines.
    // Hardcoding "main" would raise VUID-VkPipelineShaderStageCreateInfo-pName-00707.
    assert_eq!(meta.entry_point, "saxpy");
    assert_eq!(meta.binding_plan.buffers.len(), 2,
        "saxpy has 2 buffer bindings (x, y)");
    assert_eq!(meta.binding_plan.scalars.len(), 2,
        "saxpy has 2 scalar push constants (n, alpha)");
    assert_eq!(meta.push_constant_total_bytes, 8,
        "n:u32(4) + alpha:f32(4) = 8 bytes");

    // ── vector_add ───────────────────────────────────────────────────────────
    let vadd_src_path = PathBuf::from(&manifest_dir)
        .join("..")
        .join("..")
        .join("examples")
        .join("vector_add.axc");
    let vadd_spv_path = tmp_dir.join("at_510_vector_add.spv");
    let vadd_meta_path = tmp_dir.join("at_510_vector_add.spv.axc.meta.json");

    axc_driver::compile_file(&vadd_src_path, &vadd_spv_path)
        .unwrap_or_else(|e| panic!("compile_file(vector_add.axc) failed: {e}"));

    assert!(vadd_spv_path.exists(), "vector_add .spv must exist");
    assert!(vadd_meta_path.exists(), "vector_add .axc.meta.json must exist");

    let vadd_meta = load_kernel_metadata(&vadd_meta_path)
        .unwrap_or_else(|e| panic!("load_kernel_metadata failed for vector_add: {e}"));

    assert_eq!(vadd_meta.kernel_name, "vector_add");
    assert_eq!(vadd_meta.workgroup_size, [64, 1, 1]);
    assert_eq!(vadd_meta.binding_plan.buffers.len(), 3,
        "vector_add has 3 buffer bindings (a, b, c)");
    assert_eq!(vadd_meta.binding_plan.scalars.len(), 1,
        "vector_add has 1 scalar push constant (n)");

    // Clean up.
    let _ = std::fs::remove_file(&saxpy_spv_path);
    let _ = std::fs::remove_file(&saxpy_meta_path);
    let _ = std::fs::remove_file(&vadd_spv_path);
    let _ = std::fs::remove_file(&vadd_meta_path);
}

/// AT-522: CLAUDE.md and DESIGN.md document M1.5 runtime dispatch.
///
/// Verifies that:
/// 1. CLAUDE.md contains the GPU test execution section (AXC_ENABLE_GPU_TESTS).
/// 2. DESIGN.md contains the §3.1.6 Runtime dispatch section.
///
/// Path: `env!("CARGO_MANIFEST_DIR")/../../CLAUDE.md` (two levels up from
/// crates/axc-driver to workspace root), per AT-522 W8 fix from M1.5 rev 1.
#[test]
fn at_522_design_and_claude_md_document_m1_5_runtime_dispatch() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");

    // CLAUDE.md
    let claude_md_path = PathBuf::from(&manifest_dir)
        .join("..")
        .join("..")
        .join("CLAUDE.md");
    let claude_content = std::fs::read_to_string(&claude_md_path)
        .unwrap_or_else(|e| panic!("failed to read CLAUDE.md at {claude_md_path:?}: {e}"));

    assert!(
        claude_content.contains("AXC_ENABLE_GPU_TESTS"),
        "CLAUDE.md must document AXC_ENABLE_GPU_TESTS env var"
    );
    assert!(
        claude_content.contains("GPU test execution") || claude_content.contains("M1.5"),
        "CLAUDE.md must document GPU test execution or M1.5 completion"
    );

    // DESIGN.md
    let design_md_path = PathBuf::from(&manifest_dir)
        .join("..")
        .join("..")
        .join("DESIGN.md");
    let design_content = std::fs::read_to_string(&design_md_path)
        .unwrap_or_else(|e| panic!("failed to read DESIGN.md at {design_md_path:?}: {e}"));

    assert!(
        design_content.contains("M1.5") || design_content.contains("Runtime dispatch"),
        "DESIGN.md must document M1.5 runtime dispatch"
    );
    assert!(
        design_content.contains("VulkanContext") || design_content.contains("DispatchRequest"),
        "DESIGN.md must mention VulkanContext or DispatchRequest API"
    );
}
