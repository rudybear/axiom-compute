//! AT-324: Integration test — compile `examples/reduction.axc` to SPIR-V
//! and validate with the spirv-tools crate (in-process, always mandatory).
//!
//! Exercises M1.3 for-loop codegen end-to-end via the axc-driver public API:
//!   - n: u32 push-constant scalar
//!   - src: readonly_buffer[f32] (binding=0)
//!   - dst: buffer[f32] (binding=1)
//!   - for-loop in kernel body → OpLoopMerge must appear in SPIR-V output
//!
//! Structural assertions (using rspirv Op enum, NO raw u32):
//!   AT-324-a: SPIR-V header words correct (magic=0x07230203, version=1.3)
//!   AT-324-b: OpLoopMerge present in output (for-loop lowering)
//!   AT-324-c: OpIAdd present (loop increment)
//!   AT-324-d: OpULessThan present (loop condition)
//!   AT-324-e: OpVariable StorageBuffer present (SSBO globals)
//!   AT-324-f: OpVariable PushConstant present (n parameter)
//!   AT-324-g: in-process spirv-tools validation (always mandatory, no PATH dependency)

use std::path::PathBuf;
use rspirv::spirv::{Op, StorageClass};
use rspirv::dr::Operand;

fn load_words(bytes: &[u8]) -> Vec<u32> {
    assert!(bytes.len().is_multiple_of(4), "SPIR-V byte length must be divisible by 4");
    let n = bytes.len() / 4;
    let mut words = Vec::with_capacity(n);
    for i in 0..n {
        words.push(u32::from_le_bytes([bytes[4 * i], bytes[4 * i + 1], bytes[4 * i + 2], bytes[4 * i + 3]]));
    }
    words
}

/// Validate SPIR-V words using the in-process spirv-tools crate.
/// This is always mandatory — no PATH dependency, no silent skip.
fn validate_spirv(words: &[u32], label: &str) {
    use spirv_tools::val::{Validator, create as create_validator};
    use spirv_tools::TargetEnv;
    let validator = create_validator(Some(TargetEnv::Vulkan_1_1));
    validator.validate(words, None)
        .unwrap_or_else(|e| panic!("AT-324-g: spirv-tools rejected {label} SPIR-V: {e}"));
}

fn iter_instructions(words: &[u32]) -> impl Iterator<Item = (u16, &[u32])> {
    IterInst { words, cursor: 0 }
}

struct IterInst<'a> {
    words: &'a [u32],
    cursor: usize,
}

impl<'a> Iterator for IterInst<'a> {
    type Item = (u16, &'a [u32]);
    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.words.len() {
            return None;
        }
        let hdr = self.words[self.cursor];
        let wc = (hdr >> 16) as usize;
        let op = (hdr & 0xFFFF) as u16;
        if wc == 0 || self.cursor + wc > self.words.len() {
            return None;
        }
        let slice = &self.words[self.cursor..self.cursor + wc];
        self.cursor += wc;
        Some((op, slice))
    }
}

fn has_op_in_words(words: &[u32], target_op: Op) -> bool {
    iter_instructions(&words[5..]).any(|(op, _)| op == target_op as u16)
}

#[test]
fn test_compile_reduction_produces_valid_spirv() {
    let manifest_dir = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"),
    );
    let examples_dir = manifest_dir.join("..").join("..").join("examples");
    let source_path = examples_dir.join("reduction.axc");

    assert!(
        source_path.exists(),
        "examples/reduction.axc not found at {:?}", source_path
    );

    let tmp_dir = std::env::temp_dir();
    let out_path = tmp_dir.join("axc_test_reduction.spv");

    axc_driver::compile_file(&source_path, &out_path)
        .expect("compile_file should succeed for examples/reduction.axc");

    let spv_bytes = std::fs::read(&out_path).expect("failed to read output .spv");
    assert!(spv_bytes.len() >= 20, "SPIR-V too short: {} bytes", spv_bytes.len());

    let words = load_words(&spv_bytes);

    // AT-324-a: Header words.
    assert_eq!(words[0], 0x0723_0203_u32, "AT-324-a: magic word mismatch");
    assert_eq!(words[1], 0x0001_0300_u32, "AT-324-a: version must be 1.3");
    assert_eq!(words[2], 0x0000_0000_u32, "AT-324-a: generator must be 0");

    // AT-324-b: OpLoopMerge present (for-loop).
    assert!(
        has_op_in_words(&words, Op::LoopMerge),
        "AT-324-b: expected Op::LoopMerge in reduction SPIR-V (for-loop lowering)"
    );

    // AT-324-c: OpIAdd present (loop increment i = i + 1).
    assert!(
        has_op_in_words(&words, Op::IAdd),
        "AT-324-c: expected Op::IAdd in reduction SPIR-V (loop increment)"
    );

    // AT-324-d: OpULessThan present (loop condition i < n).
    assert!(
        has_op_in_words(&words, Op::ULessThan),
        "AT-324-d: expected Op::ULessThan in reduction SPIR-V (loop condition)"
    );

    // Load into rspirv for typed assertions.
    let module = rspirv::dr::load_words(&words).expect("rspirv failed to load emitted words");

    // AT-324-e: OpVariable StorageBuffer present (SSBO globals for src and dst).
    let has_storage_buffer_var = module.types_global_values.iter().any(|inst| {
        inst.class.opcode == Op::Variable
            && inst.operands.iter().any(|op| matches!(op, Operand::StorageClass(StorageClass::StorageBuffer)))
    });
    assert!(has_storage_buffer_var, "AT-324-e: expected Op::Variable StorageBuffer for SSBO");

    // AT-324-f: OpVariable PushConstant present (n parameter).
    let has_push_const_var = module.types_global_values.iter().any(|inst| {
        inst.class.opcode == Op::Variable
            && inst.operands.iter().any(|op| matches!(op, Operand::StorageClass(StorageClass::PushConstant)))
    });
    assert!(has_push_const_var, "AT-324-f: expected Op::Variable PushConstant for scalar params");

    // AT-324-g: in-process SPIR-V validation via spirv-tools crate (always mandatory).
    validate_spirv(&words, "reduction");

    let _ = std::fs::remove_file(&out_path);
}
