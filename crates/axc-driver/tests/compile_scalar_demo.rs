//! AT-101: Integration test — compile `examples/scalar_demo.axc` to SPIR-V
//! and validate it with `spirv-val` (if available).
//!
//! Exercises M1.1 scalar ops end-to-end: let/let-mut, arithmetic, comparison,
//! short-circuit (and/or), bitwise builtins, 64-bit types, and mutable assignment.
//!
//! This test also checks:
//! - The output contains no HIR errors (no type errors on the demo source).
//! - The SPIR-V header magic, version, and generator fields are correct.
//! - No debug opcodes are emitted in the output (AT-13 in integration form).
//! - spirv-val accepts the output (when spirv-val is on PATH).

use std::path::{Path, PathBuf};
use std::process::Output;

/// Locate `spirv-val` by walking `PATH` without shelling out.
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

/// Run `spirv-val --target-env vulkan1.1 <path>` and return the process output.
fn run_spirv_val(spirv_val: &Path, spv_path: &Path) -> std::io::Result<Output> {
    std::process::Command::new(spirv_val)
        .arg("--target-env")
        .arg("vulkan1.1")
        .arg(spv_path)
        .output()
}

/// Iterate SPIR-V instructions in a word stream (caller must skip 5-word header).
fn iter_instructions(words: &[u32]) -> impl Iterator<Item = (u16, &[u32])> {
    IterInstructions { words, cursor: 0 }
}

struct IterInstructions<'a> {
    words: &'a [u32],
    cursor: usize,
}

impl<'a> Iterator for IterInstructions<'a> {
    type Item = (u16, &'a [u32]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.words.len() {
            return None;
        }
        let header: u32 = self.words[self.cursor];
        let word_count: usize = ((header >> 16) & 0xFFFF) as usize;
        let opcode: u16 = (header & 0xFFFF) as u16;
        if word_count == 0 || self.cursor + word_count > self.words.len() {
            return None;
        }
        let slice: &[u32] = &self.words[self.cursor..self.cursor + word_count];
        self.cursor += word_count;
        Some((opcode, slice))
    }
}

#[test]
fn test_compile_scalar_demo_produces_valid_spirv() {
    // 1. Locate examples/scalar_demo.axc relative to CARGO_MANIFEST_DIR
    let manifest_dir: PathBuf = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"),
    );
    // crates/axc-driver is two levels below the repo root
    let examples_dir: PathBuf = manifest_dir.join("..").join("..").join("examples");
    let source_path: PathBuf = examples_dir.join("scalar_demo.axc");

    assert!(
        source_path.exists(),
        "examples/scalar_demo.axc not found at {:?}",
        source_path
    );

    // 2. Compile to a temporary file
    let tmp_dir: PathBuf = std::env::temp_dir();
    let out_path: PathBuf = tmp_dir.join("axc_test_scalar_demo.spv");

    axc_driver::compile_file(&source_path, &out_path)
        .expect("compile_file should succeed for examples/scalar_demo.axc");

    // 3. Read and verify SPIR-V header words
    let spv_bytes: Vec<u8> = std::fs::read(&out_path).expect("failed to read output .spv");
    assert!(
        spv_bytes.len() >= 20,
        "SPIR-V output is too short ({} bytes); expected at least 5 header words",
        spv_bytes.len()
    );

    let to_word = |i: usize| -> u32 {
        u32::from_le_bytes([spv_bytes[i], spv_bytes[i+1], spv_bytes[i+2], spv_bytes[i+3]])
    };
    assert_eq!(to_word(0), 0x0723_0203_u32, "word[0] must be SPIR-V magic");
    assert_eq!(to_word(4), 0x0001_0300_u32, "word[1] must be SPIR-V 1.3");
    assert_eq!(to_word(8), 0x0000_0000_u32, "word[2] must be generator=0");
    assert_eq!(to_word(16), 0x0000_0000_u32, "word[4] must be schema=0");

    // 4. Reconstruct word stream and check no debug opcodes
    let n_words: usize = spv_bytes.len() / 4;
    let mut all_words: Vec<u32> = Vec::with_capacity(n_words);
    for i in 0..n_words {
        all_words.push(to_word(i * 4));
    }
    const DEBUG_OPCODES: &[u16] = &[2, 3, 4, 5, 6, 7, 8, 317, 330];
    for (opcode, _) in iter_instructions(&all_words[5..]) {
        assert!(
            !DEBUG_OPCODES.contains(&opcode),
            "found debug opcode {opcode} in scalar_demo output"
        );
    }

    // 5. Check Int64 capability (11) is present (scalar_demo uses i64/u64)
    // OpCapability = opcode 17 = 0x11
    let has_int64_cap: bool = iter_instructions(&all_words[5..]).any(|(op, slice)| {
        op == 17u16 && slice.len() >= 2 && slice[1] == 11u32
    });
    assert!(
        has_int64_cap,
        "scalar_demo uses i64/u64, so Int64 capability (11) must be present"
    );

    // 6. Run spirv-val if available
    match which_spirv_val() {
        Some(sv_path) => {
            let output: Output = run_spirv_val(&sv_path, &out_path)
                .expect("failed to execute spirv-val");
            assert!(
                output.status.success(),
                "spirv-val rejected scalar_demo output:\nstdout: {}\nstderr: {}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
        }
        None => {
            if std::env::var("AXC_REQUIRE_SPIRV_VAL").as_deref() == Ok("1") {
                panic!(
                    "spirv-val required by CI but not found on PATH; \
                    install SPIRV-Tools or set AXC_REQUIRE_SPIRV_VAL=0 to skip"
                );
            } else {
                eprintln!(
                    "note: spirv-val not found on PATH; \
                    skipping spirv-val validation for scalar_demo."
                );
            }
        }
    }

    // 7. Assert required arithmetic opcodes are present in the SPIR-V output.
    //    AT-101 expected_behavior: the scalar_demo output must contain at least
    //    one of each: Op::IAdd (128), Op::IMul (130), Op::Phi (245),
    //    Op::BitwiseAnd (196).
    //
    //    These opcode numbers come from the SPIR-V spec §3.32 / rspirv::spirv::Op.
    //    We use raw u16 values to avoid importing rspirv in the integration test
    //    (the helper already works with raw opcodes).
    //
    //    scalar_demo.axc exercises:
    //    - IAdd:       `counter + 1i32` (mutable binding increment)
    //    - IMul:       `10i32 * 3i32`
    //    - Phi:        `true and false` / `false or true` (short-circuit diamonds)
    //    - BitwiseAnd: `band(0x0Fi32, 0x3Fi32)`

    // IAdd = 128 per SPIR-V §3.32.14
    const OP_IADD:       u16 = 128;
    // IMul = 130
    const OP_IMUL:       u16 = 130;
    // Phi  = 245
    const OP_PHI:        u16 = 245;
    // BitwiseAnd = 196
    const OP_BITWISE_AND: u16 = 196;

    let has_iadd = iter_instructions(&all_words[5..]).any(|(op, _)| op == OP_IADD);
    let has_imul = iter_instructions(&all_words[5..]).any(|(op, _)| op == OP_IMUL);
    let has_phi  = iter_instructions(&all_words[5..]).any(|(op, _)| op == OP_PHI);
    let has_band = iter_instructions(&all_words[5..]).any(|(op, _)| op == OP_BITWISE_AND);

    assert!(has_iadd, "scalar_demo must contain Op::IAdd (128); not found in SPIR-V output");
    assert!(has_imul, "scalar_demo must contain Op::IMul (130); not found in SPIR-V output");
    assert!(has_phi,  "scalar_demo must contain Op::Phi (245) from short-circuit and/or; not found");
    assert!(has_band, "scalar_demo must contain Op::BitwiseAnd (196) from band(); not found");

    // 8. Clean up temp file
    let _ = std::fs::remove_file(&out_path);
}
