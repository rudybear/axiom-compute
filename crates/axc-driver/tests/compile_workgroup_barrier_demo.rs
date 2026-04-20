//! AT-412: Integration test — compile `examples/workgroup_barrier_demo.axc` to SPIR-V
//! and validate with in-process spirv-tools (Vulkan_1_1). MANDATORY — no PATH dependency, no silent skip.
//!
//! Exercises M1.4 workgroup barrier end-to-end via the axc-driver public API:
//!   - workgroup_barrier() → OpControlBarrier with exec=Workgroup(2), mem=Workgroup(2), sem=0x108
//!   - NO GroupNonUniform capabilities (barrier is not a subgroup op)
//!   - NO OpExtension SPV_KHR_shader_subgroup_* (barrier needs none)
//!
//! AT-412-a: SPIR-V header words correct (magic=0x07230203, version=1.3)
//! AT-412-b: in-process spirv-tools validation passes (Vulkan_1_1)
//! AT-412-c: OpControlBarrier present in SPIR-V output
//! AT-412-d: Capability::GroupNonUniform is ABSENT (barrier does not require it)
//! AT-412-e: No OpExtension SPV_KHR_shader_subgroup_* strings in output

use std::path::PathBuf;
use rspirv::spirv::{Op, Capability};
use rspirv::dr::Operand;

/// Validate SPIR-V words using the in-process spirv-tools crate.
/// MANDATORY — no PATH dependency, no silent skip. PANIC on failure.
fn validate_spirv(words: &[u32], label: &str) {
    use spirv_tools::val::{Validator, create as create_validator};
    use spirv_tools::TargetEnv;
    let validator = create_validator(Some(TargetEnv::Vulkan_1_1));
    validator.validate(words, None)
        .unwrap_or_else(|e| panic!("AT-412-b: spirv-tools rejected {label} SPIR-V: {e}"));
}

fn load_words(bytes: &[u8]) -> Vec<u32> {
    assert!(
        bytes.len().is_multiple_of(4),
        "SPIR-V byte length must be divisible by 4; got {}",
        bytes.len()
    );
    let n = bytes.len() / 4;
    let mut words = Vec::with_capacity(n);
    for i in 0..n {
        words.push(u32::from_le_bytes([
            bytes[4 * i],
            bytes[4 * i + 1],
            bytes[4 * i + 2],
            bytes[4 * i + 3],
        ]));
    }
    words
}

fn count_op_in_module(module: &rspirv::dr::Module, target_op: Op) -> usize {
    module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|blk| blk.instructions.iter())
        .filter(|i| i.class.opcode == target_op)
        .count()
}

fn has_extension_string_prefix(module: &rspirv::dr::Module, prefix: &str) -> bool {
    module.extensions.iter().any(|i| {
        i.operands
            .first()
            .and_then(|op| {
                if let Operand::LiteralString(s) = op {
                    Some(s.as_str())
                } else {
                    None
                }
            })
            .map(|s| s.starts_with(prefix))
            .unwrap_or(false)
    })
}

/// AT-412: Compile examples/workgroup_barrier_demo.axc and validate.
#[test]
fn test_compile_workgroup_barrier_demo_produces_valid_spirv() {
    let manifest_dir = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"),
    );
    let examples_dir = manifest_dir.join("..").join("..").join("examples");
    let source_path = examples_dir.join("workgroup_barrier_demo.axc");

    assert!(
        source_path.exists(),
        "examples/workgroup_barrier_demo.axc not found at {:?}",
        source_path
    );

    let tmp_dir = std::env::temp_dir();
    let out_path = tmp_dir.join("axc_test_workgroup_barrier_demo.spv");

    axc_driver::compile_file(&source_path, &out_path)
        .expect("compile_file should succeed for examples/workgroup_barrier_demo.axc");

    let spv_bytes = std::fs::read(&out_path).expect("failed to read output .spv");
    assert!(
        spv_bytes.len() >= 20,
        "SPIR-V too short: {} bytes",
        spv_bytes.len()
    );

    let words = load_words(&spv_bytes);

    // AT-412-a: Header words.
    assert_eq!(words[0], 0x0723_0203_u32, "AT-412-a: magic word mismatch");
    assert_eq!(words[1], 0x0001_0300_u32, "AT-412-a: version must be 1.3");
    assert_eq!(words[2], 0x0000_0000_u32, "AT-412-a: generator must be 0");

    // Load into rspirv for typed assertions.
    let module = rspirv::dr::load_words(&words).expect("rspirv failed to load emitted words");

    // AT-412-b: in-process spirv-tools validation (MANDATORY, no silent skip).
    validate_spirv(&words, "workgroup_barrier_demo");

    // AT-412-c: OpControlBarrier present (workgroup_barrier()).
    let barrier_count = count_op_in_module(&module, Op::ControlBarrier);
    assert_eq!(
        barrier_count, 1,
        "AT-412-c: expected exactly 1 OpControlBarrier; got {barrier_count}"
    );

    // AT-412-d: Capability::GroupNonUniform is ABSENT (workgroup_barrier does not require it).
    let caps_present: Vec<Capability> = module
        .capabilities
        .iter()
        .filter_map(|i| {
            if let Some(Operand::Capability(cap)) = i.operands.first() {
                Some(*cap)
            } else {
                None
            }
        })
        .collect();

    assert!(
        caps_present.contains(&Capability::Shader),
        "Capability::Shader must be present; caps: {caps_present:?}"
    );
    assert!(
        !caps_present.contains(&Capability::GroupNonUniform),
        "AT-412-d: Capability::GroupNonUniform must be ABSENT for workgroup_barrier; caps: {caps_present:?}"
    );
    assert!(
        !caps_present.contains(&Capability::GroupNonUniformArithmetic),
        "AT-412-d: Capability::GroupNonUniformArithmetic must be ABSENT; caps: {caps_present:?}"
    );
    assert!(
        !caps_present.contains(&Capability::GroupNonUniformBallot),
        "AT-412-d: Capability::GroupNonUniformBallot must be ABSENT; caps: {caps_present:?}"
    );
    assert!(
        !caps_present.contains(&Capability::GroupNonUniformVote),
        "AT-412-d: Capability::GroupNonUniformVote must be ABSENT; caps: {caps_present:?}"
    );

    // AT-412-e: No OpExtension SPV_KHR_shader_subgroup_* strings.
    let has_sg_ext = has_extension_string_prefix(&module, "SPV_KHR_shader_subgroup_");
    assert!(
        !has_sg_ext,
        "AT-412-e: no SPV_KHR_shader_subgroup_* OpExtension expected for workgroup_barrier_demo"
    );

    let _ = std::fs::remove_file(&out_path);
}
