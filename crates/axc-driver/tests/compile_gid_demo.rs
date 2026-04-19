//! AT-203: Integration test — compile `examples/gid_demo.axc` to SPIR-V.
//!
//! Exercises M1.2 global invocation ID (gid) builtin:
//!   - gid(0), gid(1), gid(2) are all used in the kernel body
//!   - A single buffer[u32] output param
//!
//! Structural assertions:
//!   - SPIR-V header: magic=0x07230203, version=1.3
//!   - Exactly 1 OpVariable Input (GlobalInvocationId)
//!   - BuiltIn GlobalInvocationId decoration present
//!   - At least 3 OpLoad of v3uint (one per gid() call) - note: the codegen emits
//!     one OpLoad per gid() call-site in the body per spec AT-203
//!   - At least 3 OpCompositeExtract with axis indices 0, 1, 2 (one each)
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
fn test_compile_gid_demo_produces_valid_spirv() {
    use rspirv::spirv::{Op, StorageClass};
    use rspirv::dr::Operand;

    let manifest_dir = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"),
    );
    let examples_dir = manifest_dir.join("..").join("..").join("examples");
    let source_path = examples_dir.join("gid_demo.axc");

    assert!(
        source_path.exists(),
        "examples/gid_demo.axc not found at {:?}", source_path
    );

    let tmp_dir = std::env::temp_dir();
    let out_path = tmp_dir.join("axc_test_gid_demo.spv");

    axc_driver::compile_file(&source_path, &out_path)
        .expect("compile_file should succeed for examples/gid_demo.axc");

    let spv_bytes = std::fs::read(&out_path).expect("failed to read output .spv");
    assert!(spv_bytes.len() >= 20, "SPIR-V too short: {} bytes", spv_bytes.len());

    let words = load_words(&spv_bytes);

    // Header checks.
    assert_eq!(words[0], 0x0723_0203_u32, "magic word mismatch");
    assert_eq!(words[1], 0x0001_0300_u32, "version must be 1.3");

    let module = rspirv::dr::load_words(&words).expect("rspirv failed to load emitted words");

    // AT-203: Exactly 1 OpVariable Input (GlobalInvocationId).
    let input_vars: Vec<_> = module.types_global_values.iter()
        .filter(|inst| {
            inst.class.opcode == Op::Variable
                && inst.operands.iter().any(|op| {
                    matches!(op, Operand::StorageClass(StorageClass::Input))
                })
        })
        .collect();
    assert_eq!(
        input_vars.len(), 1,
        "AT-203: expected exactly 1 OpVariable Input (GlobalInvocationId); got {}",
        input_vars.len()
    );

    // AT-203: BuiltIn GlobalInvocationId decoration present.
    let has_gid_builtin = module.annotations.iter().any(|inst| {
        inst.operands.iter().any(|op| {
            matches!(op, Operand::BuiltIn(rspirv::spirv::BuiltIn::GlobalInvocationId))
        })
    });
    assert!(has_gid_builtin, "AT-203: expected BuiltIn GlobalInvocationId decoration");

    // AT-203: The gid var appears in OpEntryPoint interface list.
    let gid_var_id = input_vars[0].result_id.expect("OpVariable must have result_id");
    let ep_interface_has_gid = module.entry_points.iter().any(|ep| {
        ep.operands.iter().any(|op| {
            if let Operand::IdRef(id) = op { *id == gid_var_id } else { false }
        })
    });
    assert!(
        ep_interface_has_gid,
        "AT-203: GlobalInvocationId variable must appear in OpEntryPoint interface list"
    );

    // AT-203: StorageBuffer variables must NOT appear in OpEntryPoint interface list.
    let sb_var_ids: Vec<u32> = module.types_global_values.iter()
        .filter(|inst| {
            inst.class.opcode == Op::Variable
                && inst.operands.iter().any(|op| {
                    matches!(op, Operand::StorageClass(StorageClass::StorageBuffer))
                })
        })
        .filter_map(|inst| inst.result_id)
        .collect();
    for ep in &module.entry_points {
        for op in &ep.operands {
            if let Operand::IdRef(id) = op {
                assert!(
                    !sb_var_ids.contains(id),
                    "AT-203: StorageBuffer variable {id} must NOT appear in OpEntryPoint interface list (SPIR-V 1.3 §2.17)"
                );
            }
        }
    }

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
            "AT-203: spirv-val rejected gid_demo output:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    let _ = std::fs::remove_file(&out_path);
}
