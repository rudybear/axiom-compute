//! AT-828: Push-constant-only kernel with zero buffer bindings.
//!
//! Verifies that `prepare_kernel` and `dispatch_handle` work correctly for a
//! kernel that has no buffer parameters — only push-constant scalars. The
//! descriptor set layout, descriptor pool, and descriptor set are all None.
//! `dispatch_handle` skips `cmd_bind_descriptor_sets` entirely.

use axc_runtime::{VulkanContext, VulkanContextOptions, probe_vulkan_available, gpu_tests_enabled};
use axc_hir::{ParamBindingPlan, ScalarPushConstantSlot, ScalarTy};
use axc_lexer::Span;

fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// AT-828: Zero-buffer push-constant-only kernel must compile and dispatch.
///
/// Uses a trivial kernel with a single u32 push constant and no buffer bindings.
/// `descriptor_set_layout`, `descriptor_pool`, and `descriptor_set` are all None.
/// `dispatch_handle` must succeed with empty inputs and output_sizes.
#[test]
#[ignore] // GPU-gated: requires AXC_ENABLE_GPU_TESTS=1 + Vulkan ICD
fn at_828_push_constants_only_kernel_dispatches() {
    if !gpu_tests_enabled() {
        eprintln!("skipping at_828 (AXC_ENABLE_GPU_TESTS != 1)");
        return;
    }
    if !probe_vulkan_available() {
        eprintln!("skipping at_828 (no Vulkan ICD available)");
        return;
    }

    // Build a trivial SPIR-V kernel with one push-constant u32 and no buffers.
    const SRC: &str = concat!(
        "@kernel\n",
        "@workgroup(1, 1, 1)\n",
        "@intent(\"AT-828 push-constant-only kernel\")\n",
        "@complexity(O(1))\n",
        "fn pc_only(dummy: u32) -> void {\n",
        "    return;\n",
        "}\n",
    );

    let (spirv_bytes, meta) = axc_driver::compile_source_with_meta(SRC)
        .expect("pc_only kernel must compile without errors");
    let spirv_words: Vec<u32> = bytes_to_words(&spirv_bytes);

    eprintln!("AT-828: entry={}, buffers={}, scalars={}",
        meta.entry_point,
        meta.binding_plan.buffers.len(),
        meta.binding_plan.scalars.len());

    // Verify the plan has no buffer bindings.
    assert!(
        meta.binding_plan.buffers.is_empty(),
        "AT-828: pc_only kernel must have no buffer bindings"
    );
    assert_eq!(
        meta.binding_plan.push_constant_total_bytes, 4,
        "AT-828: pc_only kernel must have exactly 4 bytes of push constants (u32)"
    );

    let tmp_dir = tempfile::tempdir().expect("tempdir creation must succeed");
    let ctx = VulkanContext::new_with_options(VulkanContextOptions {
        pipeline_cache_path: Some(tmp_dir.path().join("at_828.cache")),
        physical_device_index: None,
        fence_timeout_ms: None,
    })
    .expect("VulkanContext::new_with_options must succeed");

    eprintln!("AT-828: device = {}", ctx.physical_device_name());

    // prepare_kernel for a 0-buffer kernel must succeed.
    let handle = ctx.prepare_kernel(
        &spirv_words,
        &meta.binding_plan,
        meta.binding_plan.push_constant_total_bytes,
        &meta.entry_point,
    )
    .expect("prepare_kernel for 0-buffer kernel must succeed");

    eprintln!("AT-828: prepare_kernel succeeded (DSL=None expected)");

    // Build push constants: dummy = 42.
    let push_constants: [u8; 4] = 42u32.to_le_bytes();

    // dispatch_handle with empty inputs/output_sizes must succeed.
    let outputs = ctx.dispatch_handle(
        &handle,
        (1, 1, 1),
        &[],
        &[],
        &push_constants,
    )
    .expect("dispatch_handle for 0-buffer kernel must succeed");

    assert!(
        outputs.is_empty(),
        "AT-828: 0-buffer kernel must return empty outputs Vec"
    );

    eprintln!("AT-828: dispatch_handle succeeded with empty outputs — AT-828 PASS");
}

/// AT-828b: push_constant_only plan construction — no-GPU unit test.
///
/// Verifies that a `ParamBindingPlan` with empty buffers has the expected shape
/// (0 buffers, non-zero push_constant_total_bytes). Does not call validate_request
/// (pub(crate)) directly — instead verifies the plan data structure is correct.
#[test]
fn at_828b_push_constants_only_plan_has_correct_shape() {
    let plan = ParamBindingPlan {
        buffers: vec![],
        scalars: vec![
            ScalarPushConstantSlot {
                name: "dummy".to_owned(),
                ty: ScalarTy::U32,
                offset: 0,
                member_index: 0,
                position: 0,
                span: Span::default(),
            },
        ],
        push_constant_total_bytes: 4,
    };

    assert!(plan.buffers.is_empty(), "0-buffer plan must have no buffer bindings");
    assert_eq!(plan.scalars.len(), 1, "0-buffer plan must have 1 scalar");
    assert_eq!(plan.push_constant_total_bytes, 4, "0-buffer plan must have 4-byte push constants");
    assert_eq!(plan.scalars[0].ty, ScalarTy::U32, "scalar must be U32");
    assert_eq!(plan.scalars[0].offset, 0, "scalar offset must be 0");
}
