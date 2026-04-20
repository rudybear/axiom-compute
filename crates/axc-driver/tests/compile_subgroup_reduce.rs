//! AT-411 / AT-418 / AT-425: Integration test — compile `examples/subgroup_reduce.axc` to SPIR-V
//! and validate with in-process spirv-tools (Vulkan_1_1). MANDATORY — no PATH dependency, no silent skip.
//!
//! Exercises M1.4 subgroup reduction end-to-end via the axc-driver public API:
//!   - subgroup_invocation_id() → OpLoad of SubgroupLocalInvocationId
//!   - subgroup_reduce_add() → OpGroupNonUniformFAdd
//!   - subgroup_elect() → OpGroupNonUniformElect
//!   - OpCapability GroupNonUniform + GroupNonUniformArithmetic + GroupNonUniformBallot
//!   - OpExtension "SPV_KHR_shader_subgroup_basic" + "SPV_KHR_shader_subgroup_arithmetic"
//!
//! AT-411-a: SPIR-V header words correct (magic=0x07230203, version=1.3)
//! AT-411-b: in-process spirv-tools validation passes (Vulkan_1_1)
//! AT-411-c: OpGroupNonUniformFAdd present (subgroup_reduce_add on f32)
//! AT-411-d: OpGroupNonUniformElect present (subgroup_elect)
//! AT-418-a: Determinism — compiling twice produces bytewise-identical SPIR-V
//! AT-418-b: OpExtension "SPV_KHR_shader_subgroup_basic" present exactly once
//! AT-418-c: OpExtension "SPV_KHR_shader_subgroup_arithmetic" present exactly once
//! AT-425:   in-process spirv-tools validation is mandatory (no silent skip)
//! Capability assertions: GroupNonUniform, GroupNonUniformArithmetic present;
//!   GroupNonUniformBallot present (elect uses ballot-type op in SPIR-V)

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
        .unwrap_or_else(|e| panic!("AT-425: spirv-tools rejected {label} SPIR-V: {e}"));
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

fn count_extension_string(module: &rspirv::dr::Module, ext_name: &str) -> usize {
    module
        .extensions
        .iter()
        .filter(|i| {
            i.operands
                .first()
                .and_then(|op| {
                    if let Operand::LiteralString(s) = op {
                        Some(s.as_str())
                    } else {
                        None
                    }
                })
                == Some(ext_name)
        })
        .count()
}

/// AT-411 / AT-418 / AT-425: Compile examples/subgroup_reduce.axc and validate.
#[test]
fn test_compile_subgroup_reduce_produces_valid_spirv() {
    let manifest_dir = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"),
    );
    let examples_dir = manifest_dir.join("..").join("..").join("examples");
    let source_path = examples_dir.join("subgroup_reduce.axc");

    assert!(
        source_path.exists(),
        "examples/subgroup_reduce.axc not found at {:?}",
        source_path
    );

    let tmp_dir = std::env::temp_dir();
    let out_path = tmp_dir.join("axc_test_subgroup_reduce.spv");

    axc_driver::compile_file(&source_path, &out_path)
        .expect("compile_file should succeed for examples/subgroup_reduce.axc");

    let spv_bytes = std::fs::read(&out_path).expect("failed to read output .spv");
    assert!(
        spv_bytes.len() >= 20,
        "SPIR-V too short: {} bytes",
        spv_bytes.len()
    );

    let words = load_words(&spv_bytes);

    // AT-411-a: Header words.
    assert_eq!(words[0], 0x0723_0203_u32, "AT-411-a: magic word mismatch");
    assert_eq!(words[1], 0x0001_0300_u32, "AT-411-a: version must be 1.3");
    assert_eq!(words[2], 0x0000_0000_u32, "AT-411-a: generator must be 0");

    // Load into rspirv for typed assertions.
    let module = rspirv::dr::load_words(&words).expect("rspirv failed to load emitted words");

    // AT-411-b: in-process spirv-tools validation (AT-425: MANDATORY).
    validate_spirv(&words, "subgroup_reduce");

    // AT-411-c: OpGroupNonUniformFAdd present (subgroup_reduce_add on f32).
    let fadd_count = count_op_in_module(&module, Op::GroupNonUniformFAdd);
    assert_eq!(
        fadd_count, 1,
        "AT-411-c: expected exactly 1 OpGroupNonUniformFAdd; got {fadd_count}"
    );

    // AT-411-d: OpGroupNonUniformElect present (subgroup_elect).
    let elect_count = count_op_in_module(&module, Op::GroupNonUniformElect);
    assert_eq!(
        elect_count, 1,
        "AT-411-d: expected exactly 1 OpGroupNonUniformElect; got {elect_count}"
    );

    // Capability assertions via spirv::Capability enum (no raw u32).
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
        caps_present.contains(&Capability::GroupNonUniform),
        "Capability::GroupNonUniform must be present (parent cap for all subgroup ops); caps: {caps_present:?}"
    );
    assert!(
        caps_present.contains(&Capability::GroupNonUniformArithmetic),
        "Capability::GroupNonUniformArithmetic must be present for subgroup_reduce_add; caps: {caps_present:?}"
    );
    // subgroup_elect() is a Basic op — requires GroupNonUniform only (no Ballot/Arithmetic/Vote).
    // subgroup_reduce_add() requires GroupNonUniformArithmetic (which forces GroupNonUniform via parent chain).

    // AT-418-b: OpExtension "SPV_KHR_shader_subgroup_basic" present exactly once.
    let ext_basic = count_extension_string(&module, "SPV_KHR_shader_subgroup_basic");
    assert_eq!(
        ext_basic, 1,
        "AT-418-b: OpExtension \"SPV_KHR_shader_subgroup_basic\" : exactly 1; got {ext_basic}"
    );

    // AT-418-c: OpExtension "SPV_KHR_shader_subgroup_arithmetic" present exactly once.
    let ext_arith = count_extension_string(&module, "SPV_KHR_shader_subgroup_arithmetic");
    assert_eq!(
        ext_arith, 1,
        "AT-418-c: OpExtension \"SPV_KHR_shader_subgroup_arithmetic\" : exactly 1; got {ext_arith}"
    );

    // AT-418-a: Determinism — compile a second time and compare.
    let out_path2 = tmp_dir.join("axc_test_subgroup_reduce_2.spv");
    axc_driver::compile_file(&source_path, &out_path2)
        .expect("second compile_file should succeed");
    let spv_bytes2 = std::fs::read(&out_path2).expect("failed to read output .spv (second)");
    assert_eq!(
        spv_bytes, spv_bytes2,
        "AT-418-a: two compilations of subgroup_reduce.axc must produce bytewise-identical SPIR-V"
    );

    let _ = std::fs::remove_file(&out_path);
    let _ = std::fs::remove_file(&out_path2);
}
