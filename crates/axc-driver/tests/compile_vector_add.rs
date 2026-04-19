//! AT-202: Integration test — compile `examples/vector_add.axc` to SPIR-V.
//!
//! Exercises M1.2 buffer bindings with 3 buffer params:
//!   - a: readonly_buffer[f32] (binding=0, NonWritable)
//!   - b: readonly_buffer[f32] (binding=1, NonWritable)
//!   - c: writeonly_buffer[f32] (binding=2, NonReadable)
//!   - n: u32 push-constant scalar
//!
//! Structural assertions:
//!   - SPIR-V header: magic=0x07230203, version=1.3
//!   - Exactly 3 OpVariable StorageBuffer (one per buffer param)
//!   - 2 of them carry OpDecorate NonWritable (a and b are readonly)
//!   - 1 carries NonReadable (c is writeonly)
//!   - Bindings 0, 1, 2 all at DescriptorSet 0
//!   - spirv-val accepts output (if spirv-val on PATH)

use std::path::PathBuf;

fn load_words(bytes: &[u8]) -> Vec<u32> {
    assert!(bytes.len().is_multiple_of(4), "SPIR-V byte length must be divisible by 4");
    let n = bytes.len() / 4;
    let mut words = Vec::with_capacity(n);
    for i in 0..n {
        words.push(u32::from_le_bytes([bytes[4*i], bytes[4*i+1], bytes[4*i+2], bytes[4*i+3]]));
    }
    words
}

fn which_spirv_val() -> Option<PathBuf> {
    let path_var: std::ffi::OsString = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate: PathBuf = dir.join("spirv-val");
        if candidate.is_file() {
            return Some(candidate);
        }
        let candidate_exe: PathBuf = dir.join("spirv-val.exe");
        if candidate_exe.is_file() {
            return Some(candidate_exe);
        }
    }
    None
}

#[test]
fn test_compile_vector_add_produces_valid_spirv() {
    use rspirv::spirv::{Op, StorageClass, Decoration};
    use rspirv::dr::Operand;

    let manifest_dir = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"),
    );
    let examples_dir = manifest_dir.join("..").join("..").join("examples");
    let source_path = examples_dir.join("vector_add.axc");

    assert!(
        source_path.exists(),
        "examples/vector_add.axc not found at {:?}", source_path
    );

    let tmp_dir = std::env::temp_dir();
    let out_path = tmp_dir.join("axc_test_vector_add.spv");

    axc_driver::compile_file(&source_path, &out_path)
        .expect("compile_file should succeed for examples/vector_add.axc");

    let spv_bytes = std::fs::read(&out_path).expect("failed to read output .spv");
    assert!(spv_bytes.len() >= 20, "SPIR-V too short: {} bytes", spv_bytes.len());

    let words = load_words(&spv_bytes);

    // Header checks.
    assert_eq!(words[0], 0x0723_0203_u32, "magic word mismatch");
    assert_eq!(words[1], 0x0001_0300_u32, "version must be 1.3");
    assert_eq!(words[2], 0x0000_0000_u32, "generator must be 0");

    let module = rspirv::dr::load_words(&words).expect("rspirv failed to load emitted words");

    // AT-202: Exactly 3 OpVariable StorageBuffer.
    let storage_buffer_vars: Vec<_> = module.types_global_values.iter()
        .filter(|inst| {
            inst.class.opcode == Op::Variable
                && inst.operands.iter().any(|op| {
                    matches!(op, Operand::StorageClass(StorageClass::StorageBuffer))
                })
        })
        .collect();
    assert_eq!(
        storage_buffer_vars.len(), 3,
        "AT-202: expected exactly 3 StorageBuffer OpVariables (a, b, c); got {}",
        storage_buffer_vars.len()
    );

    // Collect the var_ids of all StorageBuffer vars.
    let sb_var_ids: std::collections::HashSet<u32> = storage_buffer_vars.iter()
        .filter_map(|inst| inst.result_id)
        .collect();

    // Count NonWritable decorations (should be 2: a and b are readonly).
    let non_writable_count = module.annotations.iter()
        .filter(|inst| {
            inst.class.opcode == Op::Decorate
                && inst.operands.iter().any(|op| matches!(op, Operand::Decoration(Decoration::NonWritable)))
                && inst.operands.first().and_then(|op| if let Operand::IdRef(id) = op { Some(*id) } else { None })
                    .map(|id| sb_var_ids.contains(&id))
                    .unwrap_or(false)
        })
        .count();
    assert_eq!(
        non_writable_count, 2,
        "AT-202: expected 2 StorageBuffer vars with NonWritable (readonly a, b); got {non_writable_count}"
    );

    // Count NonReadable decorations (should be 1: c is writeonly).
    let non_readable_count = module.annotations.iter()
        .filter(|inst| {
            inst.class.opcode == Op::Decorate
                && inst.operands.iter().any(|op| matches!(op, Operand::Decoration(Decoration::NonReadable)))
                && inst.operands.first().and_then(|op| if let Operand::IdRef(id) = op { Some(*id) } else { None })
                    .map(|id| sb_var_ids.contains(&id))
                    .unwrap_or(false)
        })
        .count();
    assert_eq!(
        non_readable_count, 1,
        "AT-202: expected 1 StorageBuffer var with NonReadable (writeonly c); got {non_readable_count}"
    );

    // Bindings 0, 1, 2 all at DescriptorSet 0.
    let binding_vals: Vec<u32> = module.annotations.iter()
        .filter(|inst| {
            inst.class.opcode == Op::Decorate
                && inst.operands.iter().any(|op| matches!(op, Operand::Decoration(Decoration::Binding)))
        })
        .filter_map(|inst| inst.operands.iter().find_map(|op| {
            if let Operand::LiteralBit32(n) = op { Some(*n) } else { None }
        }))
        .collect();
    assert!(binding_vals.contains(&0), "AT-202: expected Binding 0; got {:?}", binding_vals);
    assert!(binding_vals.contains(&1), "AT-202: expected Binding 1; got {:?}", binding_vals);
    assert!(binding_vals.contains(&2), "AT-202: expected Binding 2; got {:?}", binding_vals);

    let descriptor_set_vals: Vec<u32> = module.annotations.iter()
        .filter(|inst| {
            inst.class.opcode == Op::Decorate
                && inst.operands.iter().any(|op| matches!(op, Operand::Decoration(Decoration::DescriptorSet)))
        })
        .filter_map(|inst| inst.operands.iter().find_map(|op| {
            if let Operand::LiteralBit32(n) = op { Some(*n) } else { None }
        }))
        .collect();
    assert!(
        descriptor_set_vals.iter().all(|&v| v == 0),
        "AT-202: all buffers must be in DescriptorSet 0; got {:?}", descriptor_set_vals
    );

    // spirv-val (optional).
    if let Some(sv_path) = which_spirv_val() {
        let output = std::process::Command::new(&sv_path)
            .arg("--target-env")
            .arg("vulkan1.1")
            .arg(&out_path)
            .output()
            .expect("failed to execute spirv-val");
        assert!(
            output.status.success(),
            "AT-202: spirv-val rejected vector_add output:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    let _ = std::fs::remove_file(&out_path);
}
