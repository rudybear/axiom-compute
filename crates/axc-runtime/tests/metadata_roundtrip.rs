//! AT-512, AT-513: KernelMetadata serde roundtrip and schema version rejection.

use axc_runtime::{KernelMetadata, load_kernel_metadata, CURRENT_SCHEMA_VERSION};
use axc_hir::{ParamBindingPlan, BufferBindingSlot, ScalarPushConstantSlot, BufferTy, ScalarTy};
use axc_hir::buffer::BufferAccess;
use axc_lexer::Span;

fn saxpy_plan() -> ParamBindingPlan {
    ParamBindingPlan {
        buffers: vec![
            BufferBindingSlot {
                name: "x".to_owned(),
                ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadOnly },
                position: 2,
                buffer_position: 0,
                span: Span::default(),
            },
            BufferBindingSlot {
                name: "y".to_owned(),
                ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadWrite },
                position: 3,
                buffer_position: 1,
                span: Span::default(),
            },
        ],
        scalars: vec![
            ScalarPushConstantSlot {
                name: "n".to_owned(),
                ty: ScalarTy::U32,
                offset: 0,
                member_index: 0,
                position: 0,
                span: Span::default(),
            },
            ScalarPushConstantSlot {
                name: "alpha".to_owned(),
                ty: ScalarTy::F32,
                offset: 4,
                member_index: 1,
                position: 1,
                span: Span::default(),
            },
        ],
        push_constant_total_bytes: 8,
    }
}

/// AT-512: KernelMetadata save/load roundtrip preserves all bindings.
///
/// Constructs metadata, saves to a tempfile, loads back, asserts equality
/// via PartialEq (which works because Span fields are skipped in serde and
/// both sides use Span::default()).
#[test]
fn metadata_save_load_roundtrip_preserves_bindings() {
    let plan: ParamBindingPlan = saxpy_plan();
    // AT-512a: entry_point field holds the SPIR-V OpEntryPoint name, which the
    // codegen sets to the kernel name (e.g. `"saxpy"`). Hardcoding `"main"`
    // caused VUID-VkPipelineShaderStageCreateInfo-pName-00707 on Lavapipe.
    let original: KernelMetadata = KernelMetadata::new(
        "saxpy".to_owned(),
        [64, 1, 1],
        plan.clone(),
        "saxpy".to_owned(),
    );

    // Write to a temporary file.
    let tmp_dir = std::env::temp_dir();
    let sidecar_path = tmp_dir.join("at_512_saxpy_test.axc.meta.json");

    original.save(&sidecar_path)
        .expect("save should succeed");

    // Load back.
    let loaded: KernelMetadata = load_kernel_metadata(&sidecar_path)
        .expect("load should succeed");

    // Verify key fields.
    assert_eq!(loaded.kernel_name, "saxpy");
    assert_eq!(loaded.workgroup_size, [64, 1, 1]);
    assert_eq!(loaded.entry_point, "saxpy");
    assert_eq!(loaded.schema_version, CURRENT_SCHEMA_VERSION);
    assert_eq!(loaded.push_constant_total_bytes, 8);

    // Verify binding plan equality (Span fields will both be Span::default() = {0,0}).
    assert_eq!(loaded.binding_plan.buffers.len(), 2);
    assert_eq!(loaded.binding_plan.buffers[0].name, "x");
    assert_eq!(loaded.binding_plan.buffers[0].buffer_position, 0);
    assert_eq!(loaded.binding_plan.buffers[1].name, "y");
    assert_eq!(loaded.binding_plan.buffers[1].buffer_position, 1);
    assert_eq!(loaded.binding_plan.scalars.len(), 2);
    assert_eq!(loaded.binding_plan.scalars[0].offset, 0);
    assert_eq!(loaded.binding_plan.scalars[1].offset, 4);

    // Full PartialEq comparison (requires PartialEq on ParamBindingPlan etc.).
    assert_eq!(loaded.binding_plan, plan,
        "roundtripped binding_plan must equal original (with Span::default() for skipped fields)");

    // Clean up.
    let _ = std::fs::remove_file(&sidecar_path);
}

/// AT-513: load_kernel_metadata rejects metadata with wrong schema_version.
#[test]
fn metadata_schema_mismatch_rejected() {
    let tmp_dir = std::env::temp_dir();
    let path = tmp_dir.join("at_513_wrong_schema.axc.meta.json");

    // Write a JSON file with schema_version = 99 (unsupported).
    let json = r#"{
        "schema_version": 99,
        "kernel_name": "test",
        "workgroup_size": [64, 1, 1],
        "binding_plan": {
            "buffers": [],
            "scalars": [],
            "push_constant_total_bytes": 0
        },
        "push_constant_total_bytes": 0,
        "entry_point": "main"
    }"#;
    std::fs::write(&path, json).expect("write test file");

    let result = load_kernel_metadata(&path);
    match result {
        Err(axc_runtime::DispatchError::MetadataSchemaMismatch { got: 99, supported: 1 }) => {}
        other => panic!("expected MetadataSchemaMismatch{{got:99, supported:1}}, got: {other:?}"),
    }

    let _ = std::fs::remove_file(&path);
}
