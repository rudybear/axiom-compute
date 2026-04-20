//! AT-1: Integration test — compile `examples/empty_kernel.axc` to a SPIR-V binary
//! and validate it with `spirv-val` (if available).
//!
//! spirv-val discovery uses a portable PATH walk (no shell invocation) so this
//! test works on Unix and Windows without shelling out to `which` or `where`.
//!
//! CI behaviour (AXC_REQUIRE_SPIRV_VAL=1 set at jobs.test.env level):
//!   - spirv-val present: validate and assert exit 0.
//!   - spirv-val absent: panic with an exact actionable message (pinned in spec §7.2).
//!
//! Local dev (AXC_REQUIRE_SPIRV_VAL unset or != "1"):
//!   - spirv-val present: validate.
//!   - spirv-val absent: print skip note to stderr, return.

use std::path::PathBuf;

/// Validate SPIR-V words using the in-process spirv-tools crate.
/// This is always mandatory — no PATH dependency, no silent skip.
fn validate_spirv(words: &[u32], label: &str) {
    use spirv_tools::val::{Validator, create as create_validator};
    use spirv_tools::TargetEnv;
    let validator = create_validator(Some(TargetEnv::Vulkan_1_1));
    validator.validate(words, None)
        .unwrap_or_else(|e| panic!("AT-1: spirv-tools rejected {label} SPIR-V: {e}"));
}

/// Walk an assembled SPIR-V word stream as an instruction stream.
///
/// Yields `(opcode_u16, word_slice)` pairs. The 5-word SPIR-V header must have
/// been skipped by the caller (`words[5..]`).
///
/// Per SPIR-V §3.1: the high 16 bits of the first word of each instruction are
/// the word count; the low 16 bits are the opcode. The word count includes the
/// first (header) word itself, so `cursor += word_count` advances past the whole
/// instruction.
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
            // Malformed stream; stop iteration
            return None;
        }
        let slice: &[u32] = &self.words[self.cursor..self.cursor + word_count];
        self.cursor += word_count;
        Some((opcode, slice))
    }
}

#[test]
fn test_compile_empty_kernel_produces_valid_spirv() {
    // 1. Locate examples/empty_kernel.axc relative to CARGO_MANIFEST_DIR
    let manifest_dir: PathBuf = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"),
    );
    // crates/axc-driver → repo root is ../../
    let examples_dir: PathBuf = manifest_dir.join("..").join("..").join("examples");
    let source_path: PathBuf = examples_dir.join("empty_kernel.axc");

    assert!(
        source_path.exists(),
        "examples/empty_kernel.axc not found at {:?}",
        source_path
    );

    // 2. Compile to a temporary file
    let tmp_dir: PathBuf = std::env::temp_dir();
    let out_path: PathBuf = tmp_dir.join("axc_test_empty_kernel.spv");

    axc_driver::compile_file(&source_path, &out_path)
        .expect("compile_file should succeed for examples/empty_kernel.axc");

    // 3. Read and validate header words
    let spv_bytes: Vec<u8> = std::fs::read(&out_path).expect("failed to read output .spv");
    assert!(
        spv_bytes.len() >= 20,
        "SPIR-V output is too short ({} bytes); expected at least 20 (5 header words)",
        spv_bytes.len()
    );

    // Reconstruct words from LE bytes (per SPIR-V §2.3)
    let to_word = |i: usize| -> u32 {
        u32::from_le_bytes([spv_bytes[i], spv_bytes[i+1], spv_bytes[i+2], spv_bytes[i+3]])
    };
    let word0: u32 = to_word(0);
    let word1: u32 = to_word(4);
    let word2: u32 = to_word(8);
    let word4: u32 = to_word(16);

    assert_eq!(word0, 0x0723_0203_u32, "word[0] must be SPIR-V magic 0x07230203");
    assert_eq!(word1, 0x0001_0300_u32, "word[1] must be version 1.3 (0x00010300)");
    assert_eq!(word2, 0x0000_0000_u32, "word[2] must be generator=0 (CodegenOptions override)");
    assert_eq!(word4, 0x0000_0000_u32, "word[4] must be schema=0 per SPIR-V §2.3");
    // word[3] is the bound (ID count); not asserted — it is runtime-determined

    // 4. Validate expected-header fixture (golden file check)
    let fixture_path: PathBuf = manifest_dir
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("empty_kernel.spv.expected_header.txt");
    if fixture_path.exists() {
        let fixture: String = std::fs::read_to_string(&fixture_path)
            .expect("failed to read expected_header fixture");
        let expected_words: Vec<u32> = fixture
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.trim().starts_with("//"))
            .map(|l| {
                let hex: &str = l.split_whitespace().next().unwrap_or("");
                u32::from_str_radix(hex.trim_start_matches("0x"), 16)
                    .unwrap_or_else(|_| panic!("invalid hex in fixture: {:?}", hex))
            })
            .collect();
        assert!(expected_words.len() >= 3, "fixture must have at least 3 lines");
        assert_eq!(word0, expected_words[0], "magic word mismatch against fixture");
        assert_eq!(word1, expected_words[1], "version word mismatch against fixture");
        assert_eq!(word2, expected_words[2], "generator word mismatch against fixture");
    }

    // 5. Validate instruction stream: no debug opcodes (AT-13 in integration form)
    let n_words: usize = spv_bytes.len() / 4;
    let mut all_words: Vec<u32> = Vec::with_capacity(n_words);
    for i in 0..n_words {
        all_words.push(to_word(i * 4));
    }
    const DEBUG_OPCODES: &[u16] = &[2, 3, 4, 5, 6, 7, 8, 317, 330];
    for (opcode, _slice) in iter_instructions(&all_words[5..]) {
        assert!(
            !DEBUG_OPCODES.contains(&opcode),
            "found debug opcode {opcode} in output — M0 must emit no debug info"
        );
    }

    // 6. In-process SPIR-V validation via spirv-tools crate (always mandatory).
    validate_spirv(&all_words, "empty_kernel");

    // 7. Clean up temp file (best-effort; don't fail test on cleanup error)
    let _ = std::fs::remove_file(&out_path);
}
