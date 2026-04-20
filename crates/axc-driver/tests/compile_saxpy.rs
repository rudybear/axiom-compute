//! AT-201: Integration test — compile `examples/saxpy.axc` to SPIR-V
//! and validate it with `spirv-val` (if available).
//!
//! Exercises M1.2 buffer bindings end-to-end:
//!   - 2 buffer params (readonly_buffer + buffer → bindings 0, 1)
//!   - 2 scalar push-constant params (u32 n, f32 alpha)
//!   - gid(0) GlobalInvocationID builtin
//!   - Buffer read (x[i]) and write (y[i] = ...)
//!
//! Structural assertions:
//!   AT-201-a: SPIR-V header words correct (magic=0x07230203, version=1.3)
//!   AT-201-b: OpVariable StorageBuffer present (SSBO globals)
//!   AT-201-c: OpVariable PushConstant present
//!   AT-201-d: OpVariable Input present (GlobalInvocationID)
//!   AT-201-e: OpDecorate Block present (struct decoration)
//!   AT-201-f: OpDecorate Binding 0 and Binding 1 present
//!   AT-201-g: OpDecorate ArrayStride present
//!   AT-201-h: No debug opcodes
//!   AT-201-i: spirv-val accepts output (if spirv-val on PATH)

use std::path::PathBuf;
use rspirv::spirv::{Op, StorageClass, Decoration};
use rspirv::dr::Operand;

/// Validate SPIR-V words using the in-process spirv-tools crate.
/// This is always mandatory — no PATH dependency, no silent skip.
fn validate_spirv(words: &[u32], label: &str) {
    use spirv_tools::val::{Validator, create as create_validator};
    use spirv_tools::TargetEnv;
    let validator = create_validator(Some(TargetEnv::Vulkan_1_1));
    validator.validate(words, None)
        .unwrap_or_else(|e| panic!("AT-201-i: spirv-tools rejected {label} SPIR-V: {e}"));
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

fn load_words(bytes: &[u8]) -> Vec<u32> {
    assert!(bytes.len().is_multiple_of(4), "SPIR-V byte length must be divisible by 4");
    let n = bytes.len() / 4;
    let mut words = Vec::with_capacity(n);
    for i in 0..n {
        words.push(u32::from_le_bytes([bytes[4*i], bytes[4*i+1], bytes[4*i+2], bytes[4*i+3]]));
    }
    words
}

#[test]
fn test_compile_saxpy_produces_valid_spirv() {
    let manifest_dir = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"),
    );
    let examples_dir = manifest_dir.join("..").join("..").join("examples");
    let source_path = examples_dir.join("saxpy.axc");

    assert!(
        source_path.exists(),
        "examples/saxpy.axc not found at {:?}", source_path
    );

    let tmp_dir = std::env::temp_dir();
    let out_path = tmp_dir.join("axc_test_saxpy.spv");

    axc_driver::compile_file(&source_path, &out_path)
        .expect("compile_file should succeed for examples/saxpy.axc");

    let spv_bytes = std::fs::read(&out_path).expect("failed to read output .spv");
    assert!(spv_bytes.len() >= 20, "SPIR-V too short: {} bytes", spv_bytes.len());

    let words = load_words(&spv_bytes);

    // AT-201-a: Header words.
    assert_eq!(words[0], 0x0723_0203_u32, "magic word mismatch");
    assert_eq!(words[1], 0x0001_0300_u32, "version must be 1.3");
    assert_eq!(words[2], 0x0000_0000_u32, "generator must be 0");

    // Deserialize into rspirv module for typed assertions.
    let module = rspirv::dr::load_words(&words).expect("rspirv failed to load emitted words");

    // AT-201-b: OpVariable StorageBuffer present.
    let has_storage_buffer_var = module.types_global_values.iter().any(|inst| {
        inst.class.opcode == Op::Variable
            && inst.operands.iter().any(|op| matches!(op, Operand::StorageClass(StorageClass::StorageBuffer)))
    });
    assert!(has_storage_buffer_var, "AT-201-b: expected OpVariable StorageBuffer for SSBO");

    // AT-201-c: OpVariable PushConstant present.
    let has_push_const_var = module.types_global_values.iter().any(|inst| {
        inst.class.opcode == Op::Variable
            && inst.operands.iter().any(|op| matches!(op, Operand::StorageClass(StorageClass::PushConstant)))
    });
    assert!(has_push_const_var, "AT-201-c: expected OpVariable PushConstant for scalar params");

    // AT-201-d: OpVariable Input present (GlobalInvocationID).
    let has_input_var = module.types_global_values.iter().any(|inst| {
        inst.class.opcode == Op::Variable
            && inst.operands.iter().any(|op| matches!(op, Operand::StorageClass(StorageClass::Input)))
    });
    assert!(has_input_var, "AT-201-d: expected OpVariable Input for gl_GlobalInvocationID");

    // AT-201-e: OpDecorate Block present.
    let has_block_deco = module.annotations.iter().any(|inst| {
        inst.class.opcode == Op::Decorate
            && inst.operands.iter().any(|op| matches!(op, Operand::Decoration(Decoration::Block)))
    });
    assert!(has_block_deco, "AT-201-e: expected OpDecorate Block");

    // AT-201-f: OpDecorate Binding 0 and Binding 1 present.
    let binding_vals: Vec<u32> = module.annotations.iter()
        .filter(|inst| {
            inst.class.opcode == Op::Decorate
                && inst.operands.iter().any(|op| matches!(op, Operand::Decoration(Decoration::Binding)))
        })
        .filter_map(|inst| inst.operands.iter().find_map(|op| {
            if let Operand::LiteralBit32(n) = op { Some(*n) } else { None }
        }))
        .collect();
    assert!(binding_vals.contains(&0), "AT-201-f: expected Binding 0; got {:?}", binding_vals);
    assert!(binding_vals.contains(&1), "AT-201-f: expected Binding 1; got {:?}", binding_vals);

    // AT-201-g: OpDecorate ArrayStride present.
    let has_array_stride = module.annotations.iter().any(|inst| {
        inst.class.opcode == Op::Decorate
            && inst.operands.iter().any(|op| matches!(op, Operand::Decoration(Decoration::ArrayStride)))
    });
    assert!(has_array_stride, "AT-201-g: expected OpDecorate ArrayStride");

    // AT-201-h: No debug opcodes.
    const DEBUG_OPCODES: &[u16] = &[2, 3, 4, 5, 6, 7, 8, 317, 330];
    for (opcode, _) in iter_instructions(&words[5..]) {
        assert!(
            !DEBUG_OPCODES.contains(&opcode),
            "AT-201-h: found debug opcode {opcode} — must emit no debug info"
        );
    }

    // AT-201-i: in-process SPIR-V validation via spirv-tools crate (always mandatory).
    validate_spirv(&words, "saxpy");

    let _ = std::fs::remove_file(&out_path);
}

// ── AT-224: design_md_m1_2_docs_present ──────────────────────────────────────
// Verifies DESIGN.md contains the required M1.2 section headers.
#[test]
fn design_md_m1_2_docs_present() {
    let manifest_dir = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"),
    );
    let design_md_path = manifest_dir.join("..").join("..").join("DESIGN.md");
    let content = std::fs::read_to_string(&design_md_path)
        .unwrap_or_else(|e| panic!("failed to read DESIGN.md at {:?}: {e}", design_md_path));

    assert!(
        content.contains("### 3.1.1 Buffer types (M1.2)"),
        "DESIGN.md must contain '### 3.1.1 Buffer types (M1.2)'"
    );
    assert!(
        content.contains("### 3.1.2 Scalar kernel parameters (M1.2)"),
        "DESIGN.md must contain '### 3.1.2 Scalar kernel parameters (M1.2)'"
    );
    assert!(
        content.contains("### 3.1.3 Global invocation ID (M1.2)"),
        "DESIGN.md must contain '### 3.1.3 Global invocation ID (M1.2)'"
    );
    assert!(
        content.contains("M1.2 (this architect run): buffers, array indexing, gid."),
        "DESIGN.md must contain 'M1.2 (this architect run): buffers, array indexing, gid.'"
    );
}

// ── AT-229: design_md_m1_2_saxpy_binding_example_present ─────────────────────
// Verifies DESIGN.md contains the saxpy binding example with correct labels.
#[test]
fn design_md_m1_2_saxpy_binding_example_present() {
    let manifest_dir = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"),
    );
    let design_md_path = manifest_dir.join("..").join("..").join("DESIGN.md");
    let content = std::fs::read_to_string(&design_md_path)
        .unwrap_or_else(|e| panic!("failed to read DESIGN.md at {:?}: {e}", design_md_path));

    assert!(
        content.contains("saxpy(a: f32, x: readonly_buffer[f32], y: buffer[f32])"),
        "DESIGN.md must contain 'saxpy(a: f32, x: readonly_buffer[f32], y: buffer[f32])'"
    );
    assert!(
        content.contains("x -> descriptor binding 0"),
        "DESIGN.md must contain 'x -> descriptor binding 0'"
    );
    assert!(
        content.contains("y -> descriptor binding 1"),
        "DESIGN.md must contain 'y -> descriptor binding 1'"
    );
    assert!(
        content.contains("a -> push-constant member 0"),
        "DESIGN.md must contain 'a -> push-constant member 0'"
    );
}
