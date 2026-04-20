//! Negative-path dispatch tests: AT-517, AT-518, AT-506a, AT-519.
//!
//! AT-517 and AT-518 require a real Vulkan context to exercise the dispatch
//! path (not just the pure `validate_request` function).
//! AT-506a tests the device-limit workgroup count check via the actual device.
//! AT-519 (Display smoke) does NOT require GPU — it only checks error messages.

use axc_runtime::{
    VulkanContext, DispatchRequest, DispatchError,
    probe_vulkan_available, gpu_tests_enabled,
};
use axc_hir::{ParamBindingPlan, BufferBindingSlot, BufferTy, ScalarTy};
use axc_hir::buffer::BufferAccess;
use axc_lexer::Span;

/// Build a minimal valid SPIR-V module for error-path tests.
/// Uses the axc-driver to compile a trivial kernel.
fn compile_trivial_kernel() -> Vec<u8> {
    const SRC: &str = concat!(
        "@kernel\n",
        "@workgroup(1, 1, 1)\n",
        "@intent(\"error path test\")\n",
        "@complexity(O(1))\n",
        "fn trivial(x: readonly_buffer[f32], y: buffer[f32]) -> void {\n",
        "    return;\n",
        "}\n",
    );
    axc_driver::compile_source_to_spirv(SRC)
        .expect("trivial kernel compile should succeed")
}

fn two_buffer_plan() -> ParamBindingPlan {
    ParamBindingPlan {
        buffers: vec![
            BufferBindingSlot {
                name: "x".to_owned(),
                ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadOnly },
                position: 0,
                buffer_position: 0,
                span: Span::default(),
            },
            BufferBindingSlot {
                name: "y".to_owned(),
                ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadWrite },
                position: 1,
                buffer_position: 1,
                span: Span::default(),
            },
        ],
        scalars: Vec::new(),
        push_constant_total_bytes: 0,
    }
}

/// AT-517: dispatch rejects binding count mismatch (GPU-gated via real context).
#[test]
#[ignore]
fn dispatch_rejects_binding_count_mismatch() {
    if !gpu_tests_enabled() {
        eprintln!("skipping dispatch_rejects_binding_count_mismatch (AXC_ENABLE_GPU_TESTS != 1)");
        return;
    }
    if !probe_vulkan_available() {
        eprintln!("skipping dispatch_rejects_binding_count_mismatch (no Vulkan ICD)");
        return;
    }

    let ctx = VulkanContext::new().expect("VulkanContext::new() must succeed");
    let spirv_bytes: Vec<u8> = compile_trivial_kernel();
    let spirv_words: Vec<u32> = bytes_to_words(&spirv_bytes);
    let plan: ParamBindingPlan = two_buffer_plan();

    let data: Vec<u8> = vec![0u8; 16];

    // Only 1 input instead of 2.
    // Entry point name must match OpEntryPoint in the compiled SPIR-V, which
    // is the kernel name — "trivial" for compile_trivial_kernel — not "main".
    let req = DispatchRequest {
        spirv: &spirv_words,
        binding_plan: &plan,
        workgroups: [1, 1, 1],
        inputs: &[&data],
        output_sizes: &[16, 16],
        push_constants: &[],
        entry_point: "trivial",
    };

    match ctx.dispatch(req) {
        Err(DispatchError::BindingCountMismatch { expected: 2, provided: 1 }) => {}
        other => panic!("expected BindingCountMismatch{{2,1}}, got: {other:?}"),
    }
}

/// AT-518: dispatch rejects push-constant size mismatch (GPU-gated via real context).
#[test]
#[ignore]
fn dispatch_rejects_push_constant_size_mismatch() {
    if !gpu_tests_enabled() {
        eprintln!("skipping dispatch_rejects_push_constant_size_mismatch (AXC_ENABLE_GPU_TESTS != 1)");
        return;
    }
    if !probe_vulkan_available() {
        eprintln!("skipping dispatch_rejects_push_constant_size_mismatch (no Vulkan ICD)");
        return;
    }

    let ctx = VulkanContext::new().expect("VulkanContext::new() must succeed");

    // Kernel with 1 scalar push constant (f32, 4 bytes).
    const SRC: &str = concat!(
        "@kernel\n",
        "@workgroup(1, 1, 1)\n",
        "@intent(\"push constant test\")\n",
        "@complexity(O(1))\n",
        "fn k(a: f32) -> void {\n",
        "    return;\n",
        "}\n",
    );
    let (spirv_bytes, meta) = axc_driver::compile_source_with_meta(SRC)
        .expect("compile should succeed");
    let spirv_words: Vec<u32> = bytes_to_words(&spirv_bytes);

    // Provide 0 bytes instead of 4.
    // Use meta.entry_point (kernel.name) to match the SPIR-V OpEntryPoint; the
    // older hardcoded "main" would raise VUID-...-pName-00707 before the push-
    // constant size check can report.
    let req = DispatchRequest {
        spirv: &spirv_words,
        binding_plan: &meta.binding_plan,
        workgroups: [1, 1, 1],
        inputs: &[],
        output_sizes: &[],
        push_constants: &[],  // 0 bytes, plan expects 4
        entry_point: &meta.entry_point,
    };

    match ctx.dispatch(req) {
        Err(DispatchError::PushConstantSizeMismatch { expected: 4, provided: 0 }) => {}
        other => panic!("expected PushConstantSizeMismatch{{4,0}}, got: {other:?}"),
    }
}

/// AT-506a: dispatch rejects workgroup count over device limit (GPU-gated).
///
/// Requests workgroup count that exceeds device maximum in at least one axis.
/// This is pre-dispatch validation so no resources are leaked.
#[test]
#[ignore]
fn at_506a_dispatch_rejects_workgroup_count_over_device_limit() {
    if !gpu_tests_enabled() {
        eprintln!("skipping at_506a (AXC_ENABLE_GPU_TESTS != 1)");
        return;
    }
    if !probe_vulkan_available() {
        eprintln!("skipping at_506a (no Vulkan ICD)");
        return;
    }

    let ctx = VulkanContext::new().expect("VulkanContext::new() must succeed");
    let max_wg: [u32; 3] = ctx.max_compute_work_group_count();
    eprintln!("  device max_compute_work_group_count: {max_wg:?}");

    let spirv_bytes: Vec<u8> = compile_trivial_kernel();
    let spirv_words: Vec<u32> = bytes_to_words(&spirv_bytes);
    let plan: ParamBindingPlan = two_buffer_plan();
    let data: Vec<u8> = vec![0u8; 16];

    // Request max+1 in x-axis (always exceeds).
    let requested: [u32; 3] = [max_wg[0].saturating_add(1), 1, 1];

    // Entry point matches OpEntryPoint in compile_trivial_kernel output.
    let req = DispatchRequest {
        spirv: &spirv_words,
        binding_plan: &plan,
        workgroups: requested,
        inputs: &[&data, &data],
        output_sizes: &[16, 16],
        push_constants: &[],
        entry_point: "trivial",
    };

    match ctx.dispatch(req) {
        Err(DispatchError::WorkgroupCountExceedsDeviceLimit { .. }) => {}
        other => panic!("expected WorkgroupCountExceedsDeviceLimit, got: {other:?}"),
    }
}

/// AT-519: Every DispatchError variant's Display message contains a diagnostic keyword.
///
/// NOT GPU-gated — only tests error message formatting.
#[test]
fn dispatch_error_variants_are_display_miette() {
    let variants: Vec<DispatchError> = vec![
        DispatchError::VulkanEntryFailed("test".to_owned()),
        DispatchError::NoVulkanInstance("test".to_owned()),
        DispatchError::NoSupportedDevice,
        DispatchError::NoComputeQueue,
        DispatchError::DeviceCreationFailed("test".to_owned()),
        DispatchError::ShaderModuleCreationFailed("test".to_owned()),
        DispatchError::DescriptorSetLayoutFailed("test".to_owned()),
        DispatchError::DescriptorPoolFailed("test".to_owned()),
        DispatchError::PipelineLayoutFailed("test".to_owned()),
        DispatchError::PipelineCreationFailed("test".to_owned()),
        DispatchError::BufferAllocationFailed { binding: 0, size: 64, reason: "test".to_owned() },
        DispatchError::MemoryMapFailed("test".to_owned()),
        DispatchError::NoCompatibleMemoryType,
        DispatchError::CommandBufferRecordFailed("test".to_owned()),
        DispatchError::QueueSubmitFailed("test".to_owned()),
        DispatchError::FenceTimeout { timeout_ns: 10_000_000_000 },
        DispatchError::ReadbackFailed { binding: 1, reason: "test".to_owned() },
        DispatchError::BindingCountMismatch { expected: 2, provided: 1 },
        DispatchError::PushConstantSizeMismatch { expected: 8, provided: 4 },
        DispatchError::WorkgroupCountExceedsDeviceLimit { requested: [99999, 1, 1], max: [65535, 65535, 65535] },
        DispatchError::MetadataIoError("test".to_owned()),
        DispatchError::MetadataParseError("test".to_owned()),
        DispatchError::MetadataSchemaMismatch { got: 2, supported: 1 },
    ];

    for variant in &variants {
        let msg: String = variant.to_string();
        assert!(!msg.is_empty(),
            "DispatchError Display must be non-empty; variant: {variant:?}");

        let msg_lower: String = msg.to_lowercase();
        let has_keyword: bool = msg_lower.contains("error")
            || msg_lower.contains("failed")
            || msg_lower.contains("no ")
            || msg_lower.contains("mismatch")
            || msg_lower.contains("timeout")
            || msg_lower.contains("timed out")
            || msg_lower.contains("exceeds")
            || msg_lower.contains("found")
            || msg_lower.contains("unsupported")
            || msg_lower.contains("not supported");
        assert!(has_keyword,
            "DispatchError::Display must contain a diagnostic keyword; msg='{msg}'");
    }
}

/// Convert a `Vec<u8>` SPIR-V binary to a `Vec<u32>` word stream.
fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    assert_eq!(bytes.len() % 4, 0, "SPIR-V bytes must be 4-byte aligned");
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}
