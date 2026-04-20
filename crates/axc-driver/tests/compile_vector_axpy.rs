//! AT-325: Integration test — compile `examples/vector_axpy.axc` to SPIR-V
//! and validate with `spirv-val` (if available).
//!
//! Exercises M1.3 for-loop codegen with buffer reads and writes inside the loop:
//!   - k: u32, stride: u32, alpha: f32 push-constant scalars
//!   - x: readonly_buffer[f32] (binding=0)
//!   - y: buffer[f32] (binding=1)
//!   - for-loop with buffer write inside the loop body
//!
//! Structural assertions (using rspirv Op enum, NO raw u32):
//!   AT-325-a: SPIR-V header words correct (magic=0x07230203, version=1.3)
//!   AT-325-b: OpLoopMerge present (for-loop lowering)
//!   AT-325-c: OpIAdd present (loop increment + arithmetic)
//!   AT-325-d: OpStore present (buffer write y[idx] = result inside loop)
//!   AT-325-e: OpLoad present (buffer reads inside loop)
//!   AT-325-f: OpVariable StorageBuffer present (SSBO for x and y)
//!   AT-325-g: OpVariable PushConstant present (scalar params k, stride, alpha)
//!   AT-325-h: spirv-val accepts output (if AXC_REQUIRE_SPIRV_VAL or spirv-val on PATH)

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

fn which_spirv_val() -> Option<PathBuf> {
    let path_var: std::ffi::OsString = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join("spirv-val");
        if candidate.is_file() {
            return Some(candidate);
        }
        let candidate_exe = dir.join("spirv-val.exe");
        if candidate_exe.is_file() {
            return Some(candidate_exe);
        }
    }
    None
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
fn test_compile_vector_axpy_produces_valid_spirv() {
    let manifest_dir = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"),
    );
    let examples_dir = manifest_dir.join("..").join("..").join("examples");
    let source_path = examples_dir.join("vector_axpy.axc");

    assert!(
        source_path.exists(),
        "examples/vector_axpy.axc not found at {:?}", source_path
    );

    let tmp_dir = std::env::temp_dir();
    let out_path = tmp_dir.join("axc_test_vector_axpy.spv");

    axc_driver::compile_file(&source_path, &out_path)
        .expect("compile_file should succeed for examples/vector_axpy.axc");

    let spv_bytes = std::fs::read(&out_path).expect("failed to read output .spv");
    assert!(spv_bytes.len() >= 20, "SPIR-V too short: {} bytes", spv_bytes.len());

    let words = load_words(&spv_bytes);

    // AT-325-a: Header words.
    assert_eq!(words[0], 0x0723_0203_u32, "AT-325-a: magic word mismatch");
    assert_eq!(words[1], 0x0001_0300_u32, "AT-325-a: version must be 1.3");
    assert_eq!(words[2], 0x0000_0000_u32, "AT-325-a: generator must be 0");

    // AT-325-b: OpLoopMerge present (for-loop).
    assert!(
        has_op_in_words(&words, Op::LoopMerge),
        "AT-325-b: expected Op::LoopMerge in vector_axpy SPIR-V (for-loop lowering)"
    );

    // AT-325-c: OpIAdd present (loop increment + index arithmetic).
    assert!(
        has_op_in_words(&words, Op::IAdd),
        "AT-325-c: expected Op::IAdd in vector_axpy SPIR-V (loop increment/arithmetic)"
    );

    // AT-325-d: OpStore present (buffer write y[idx] = result).
    assert!(
        has_op_in_words(&words, Op::Store),
        "AT-325-d: expected Op::Store in vector_axpy SPIR-V (buffer write y[idx])"
    );

    // AT-325-e: OpLoad present (buffer reads x[idx], y[idx]).
    assert!(
        has_op_in_words(&words, Op::Load),
        "AT-325-e: expected Op::Load in vector_axpy SPIR-V (buffer reads)"
    );

    // Load into rspirv for typed assertions.
    let module = rspirv::dr::load_words(&words).expect("rspirv failed to load emitted words");

    // AT-325-f: OpVariable StorageBuffer present (SSBO globals for x and y).
    let has_storage_buffer_var = module.types_global_values.iter().any(|inst| {
        inst.class.opcode == Op::Variable
            && inst.operands.iter().any(|op| matches!(op, Operand::StorageClass(StorageClass::StorageBuffer)))
    });
    assert!(has_storage_buffer_var, "AT-325-f: expected Op::Variable StorageBuffer for SSBO");

    // AT-325-g: OpVariable PushConstant present (k, stride, alpha parameters).
    let has_push_const_var = module.types_global_values.iter().any(|inst| {
        inst.class.opcode == Op::Variable
            && inst.operands.iter().any(|op| matches!(op, Operand::StorageClass(StorageClass::PushConstant)))
    });
    assert!(has_push_const_var, "AT-325-g: expected Op::Variable PushConstant for scalar params");

    // AT-325-h: spirv-val (optional, guarded by AXC_REQUIRE_SPIRV_VAL).
    match which_spirv_val() {
        Some(sv_path) => {
            let output = std::process::Command::new(&sv_path)
                .arg("--target-env")
                .arg("vulkan1.1")
                .arg(&out_path)
                .output()
                .expect("failed to execute spirv-val");
            assert!(
                output.status.success(),
                "AT-325-h: spirv-val rejected vector_axpy output:\nstdout: {}\nstderr: {}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
        }
        None => {
            if std::env::var("AXC_REQUIRE_SPIRV_VAL").as_deref() == Ok("1") {
                panic!(
                    "spirv-val required by CI (AXC_REQUIRE_SPIRV_VAL=1) but not found on PATH; \
                    install SPIRV-Tools or set AXC_REQUIRE_SPIRV_VAL=0"
                );
            } else {
                eprintln!("note: spirv-val not found on PATH; skipping spirv-val validation for vector_axpy.");
            }
        }
    }

    let _ = std::fs::remove_file(&out_path);
}
