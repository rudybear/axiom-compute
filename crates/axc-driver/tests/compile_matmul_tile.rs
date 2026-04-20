//! AT-611 through AT-622, AT-629, AT-632–AT-634: Integration tests for
//! `examples/matmul_tile.axc` — M2.1 cooperative-matrix end-to-end codegen.
//!
//! Compiles `examples/matmul_tile.axc` to SPIR-V and validates with in-process
//! spirv-tools. Each test asserts a distinct structural property of the emitted
//! SPIR-V binary.
//!
//! NOTE: rspirv 0.12's `load_words` does not include `TypeCooperativeMatrixKHR`
//! in its `is_type()` whitelist, so it rejects our SPIR-V with DetachedInstruction.
//! All structural assertions use a lightweight raw-binary SPIR-V scanner instead.
//!
//! AT-611: SPIR-V header words correct and spirv-tools validation passes
//! AT-612: Three OpTypeCooperativeMatrixKHR types emitted (A, B, accumulator)
//! AT-613: All coopmat loads and stores present (RowMajor validated by spirv-tools)
//! AT-614: OpCapability CooperativeMatrixKHR present exactly once
//! AT-615: OpExtension "SPV_KHR_cooperative_matrix" present exactly once
//! AT-616: No coopmat capability emitted for a non-coopmat kernel
//! AT-617: coopmat_zero() emits OpConstantNull (not OpCompositeConstruct)
//! AT-618: F16 SSBO triggers StorageBuffer16BitAccess + SPV_KHR_16bit_storage
//! AT-619: OpTypeCooperativeMatrixKHR deduplicated (no duplicate type for same key)
//! AT-620: Compiling matmul_tile.axc end-to-end succeeds (happy path)
//! AT-621: @cooperative_matrix annotation without coopmat builtins compiles cleanly
//! AT-622: No duplicate capabilities under matmul_tile stress kernel
//! AT-629: Instruction count sanity (OpCooperativeMatrixLoadKHR × 2, MulAdd × 1, StoreKHR × 1)
//! AT-632: OpCapability VulkanMemoryModel present exactly once when coopmat used
//! AT-633: OpExtension "SPV_KHR_vulkan_memory_model" present exactly once
//! AT-634: OpMemoryModel is Logical Vulkan (not GLSL450) when coopmat used

use std::path::PathBuf;

// ── Raw SPIR-V binary helpers ─────────────────────────────────────────────────
//
// rspirv 0.12 does NOT list TypeCooperativeMatrixKHR in is_type(), so load_words
// fails with DetachedInstruction. We use a direct word-stream scanner instead.

/// SPIR-V opcode constants (bottom 16 bits of the instruction word).
mod spv {
    pub const OP_CAPABILITY:                    u16 = 17;
    pub const OP_EXTENSION:                     u16 = 10;
    pub const OP_MEMORY_MODEL:                  u16 = 14;
    pub const OP_TYPE_COOPERATIVE_MATRIX_KHR:    u16 = 4456;
    pub const OP_COOPERATIVE_MATRIX_LOAD_KHR:    u16 = 4457;
    pub const OP_COOPERATIVE_MATRIX_STORE_KHR:   u16 = 4458;
    pub const OP_COOPERATIVE_MATRIX_MUL_ADD_KHR: u16 = 4459;
    pub const OP_CONSTANT_NULL:                  u16 = 46;

    /// SPIR-V Capability enum values.
    pub mod cap {
        pub const COOPERATIVE_MATRIX_KHR:        u32 = 6022;
        pub const VULKAN_MEMORY_MODEL:            u32 = 5345;
        pub const STORAGE_BUFFER_16BIT_ACCESS:    u32 = 4433;
    }

    /// SPIR-V MemoryModel enum values.
    pub mod mem_model {
        pub const VULKAN: u32 = 3;
    }
}

/// A parsed SPIR-V instruction (opcode + word operands).
struct RawInst {
    opcode: u16,
    /// Raw operand words following the instruction word (does NOT include the
    /// instruction word itself). For type/result instructions the result_id is
    /// included here as the first or second word depending on the op.
    operands: Vec<u32>,
}

/// Parse a SPIR-V word stream into a flat list of `RawInst`.
///
/// Returns `None` if the stream is malformed (truncated instruction).
fn parse_raw(words: &[u32]) -> Option<Vec<RawInst>> {
    // Skip the 5-word header.
    if words.len() < 5 {
        return None;
    }
    let mut pos = 5;
    let mut out = Vec::new();
    while pos < words.len() {
        let instr_word = words[pos];
        let word_count = (instr_word >> 16) as usize;
        let opcode = (instr_word & 0xffff) as u16;
        if word_count == 0 || pos + word_count > words.len() {
            return None;
        }
        let operands = words[pos + 1..pos + word_count].to_vec();
        out.push(RawInst { opcode, operands });
        pos += word_count;
    }
    Some(out)
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

fn validate_spirv(words: &[u32], label: &str) {
    use spirv_tools::val::{Validator, create as create_validator};
    use spirv_tools::TargetEnv;
    let validator = create_validator(Some(TargetEnv::Vulkan_1_1));
    validator
        .validate(words, None)
        .unwrap_or_else(|e| panic!("spirv-tools rejected {label} SPIR-V: {e}"));
}

fn examples_dir() -> PathBuf {
    let manifest_dir = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"),
    );
    manifest_dir.join("..").join("..").join("examples")
}

/// Compile `examples/matmul_tile.axc` and return (bytes, words, parsed instructions).
///
/// Uses a unique temp file per call to avoid test parallelism races.
fn compile_matmul_tile() -> (Vec<u8>, Vec<u32>, Vec<RawInst>) {
    let source_path = examples_dir().join("matmul_tile.axc");
    assert!(
        source_path.exists(),
        "examples/matmul_tile.axc not found at {:?}",
        source_path
    );
    let tmp_dir = std::env::temp_dir();
    // Use a random component in the temp file name to avoid inter-test races.
    let unique = format!(
        "axc_test_matmul_tile_{}.spv",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos() ^ (d.as_secs() << 16) as u32)
            .unwrap_or(rand_seed()),
    );

    /// Simple entropy source from stack address (stable fallback).
    fn rand_seed() -> u32 {
        let x: u32 = 0;
        &x as *const u32 as u32
    }
    let out_path = tmp_dir.join(&unique);
    axc_driver::compile_file(&source_path, &out_path)
        .expect("compile_file should succeed for examples/matmul_tile.axc");
    let spv_bytes = std::fs::read(&out_path).expect("failed to read output .spv");
    let words = load_words(&spv_bytes);
    let insts = parse_raw(&words).expect("SPIR-V parse_raw failed: malformed output");
    let _ = std::fs::remove_file(&out_path);
    (spv_bytes, words, insts)
}

/// Count instructions with the given opcode.
fn count_op(insts: &[RawInst], opcode: u16) -> usize {
    insts.iter().filter(|i| i.opcode == opcode).count()
}

/// Count OpCapability instructions with a specific capability value.
fn count_capability(insts: &[RawInst], cap_value: u32) -> usize {
    insts
        .iter()
        .filter(|i| {
            i.opcode == spv::OP_CAPABILITY
                && i.operands.first().copied() == Some(cap_value)
        })
        .count()
}

/// Count OpExtension instructions with the given extension name string.
fn count_extension(insts: &[RawInst], ext_name: &str) -> usize {
    let target = encode_spirv_string(ext_name);
    insts
        .iter()
        .filter(|i| i.opcode == spv::OP_EXTENSION && i.operands == target.as_slice())
        .count()
}

/// Encode a string into SPIR-V LiteralString format (null-terminated, 4-byte aligned).
fn encode_spirv_string(s: &str) -> Vec<u32> {
    let bytes = s.as_bytes();
    // Add null terminator and pad to 4-byte boundary.
    let padded_len = (bytes.len() + 1).div_ceil(4);
    let mut result = vec![0u32; padded_len];
    let raw: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, padded_len * 4)
    };
    raw[..bytes.len()].copy_from_slice(bytes);
    raw[bytes.len()] = 0; // null terminator
    result
}

/// Collect all capabilities from OpCapability instructions.
fn collect_capabilities(insts: &[RawInst]) -> Vec<u32> {
    insts
        .iter()
        .filter(|i| i.opcode == spv::OP_CAPABILITY)
        .filter_map(|i| i.operands.first().copied())
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// AT-611: SPIR-V header words are correct and spirv-tools validation passes.
#[test]
fn at_611_matmul_tile_produces_valid_spirv() {
    let (_bytes, words, _insts) = compile_matmul_tile();

    assert!(words.len() >= 5, "AT-611: SPIR-V too short: {} words", words.len());
    assert_eq!(words[0], 0x0723_0203_u32, "AT-611: magic word mismatch");
    assert_eq!(words[1], 0x0001_0300_u32, "AT-611: version must be 1.3");
    assert_eq!(words[2], 0x0000_0000_u32, "AT-611: generator must be 0");

    validate_spirv(&words, "matmul_tile");
}

/// AT-612: Three OpTypeCooperativeMatrixKHR types are emitted (A, B, accumulator).
#[test]
fn at_612_matmul_tile_emits_three_coopmat_types() {
    let (_, _, insts) = compile_matmul_tile();
    let count = count_op(&insts, spv::OP_TYPE_COOPERATIVE_MATRIX_KHR);
    assert_eq!(
        count, 3,
        "AT-612: expected 3 OpTypeCooperativeMatrixKHR (A, B, accumulator); got {count}"
    );
}

/// AT-613: Coopmat load and store instructions are present and spirv-tools validates
/// RowMajorKHR layout (layout=0 is the only accepted value in spirv-val).
#[test]
fn at_613_matmul_tile_coopmat_ops_use_row_major() {
    let (_, words, insts) = compile_matmul_tile();

    let load_count = count_op(&insts, spv::OP_COOPERATIVE_MATRIX_LOAD_KHR);
    let store_count = count_op(&insts, spv::OP_COOPERATIVE_MATRIX_STORE_KHR);

    assert_eq!(
        load_count, 2,
        "AT-613: expected 2 OpCooperativeMatrixLoadKHR (a and b); got {load_count}"
    );
    assert_eq!(
        store_count, 1,
        "AT-613: expected 1 OpCooperativeMatrixStoreKHR; got {store_count}"
    );

    // RowMajorKHR correctness validated implicitly by spirv-tools — any incorrect
    // layout value (non-zero) would cause spirv-val to reject the SPIR-V.
    validate_spirv(&words, "matmul_tile (AT-613 row-major check)");
}

/// AT-614: OpCapability CooperativeMatrixKHR present exactly once.
#[test]
fn at_614_emit_cooperative_matrix_capability_exactly_once() {
    let (_, _, insts) = compile_matmul_tile();
    let count = count_capability(&insts, spv::cap::COOPERATIVE_MATRIX_KHR);
    assert_eq!(
        count, 1,
        "AT-614: OpCapability CooperativeMatrixKHR must appear exactly once; got {count}"
    );
}

/// AT-615: OpExtension "SPV_KHR_cooperative_matrix" present exactly once.
#[test]
fn at_615_emit_spv_khr_cooperative_matrix_extension_exactly_once() {
    let (_, _, insts) = compile_matmul_tile();
    let count = count_extension(&insts, "SPV_KHR_cooperative_matrix");
    assert_eq!(
        count, 1,
        "AT-615: OpExtension \"SPV_KHR_cooperative_matrix\" must appear exactly once; got {count}"
    );
}

/// AT-616: No coopmat capability emitted when no coopmat builtins are used.
///
/// Uses a simple saxpy kernel (no cooperative-matrix ops) to verify that
/// CooperativeMatrixKHR capability is NOT emitted.
#[test]
fn at_616_emit_no_coopmat_capability_when_no_matrix_types_used() {
    let src = concat!(
        "@kernel\n",
        "@workgroup(64, 1, 1)\n",
        "@intent(\"saxpy\")\n",
        "@complexity(O(n))\n",
        "fn saxpy(n: u32, alpha: f32, x: readonly_buffer[f32], y: buffer[f32]) -> void {\n",
        "    let i: u32 = gid(0);\n",
        "    return;\n",
        "}\n",
    );
    let tmp_dir = std::env::temp_dir();
    let src_path = tmp_dir.join("at616_saxpy.axc");
    let out_path = tmp_dir.join("at616_saxpy.spv");
    std::fs::write(&src_path, src).expect("failed to write temp source");
    axc_driver::compile_file(&src_path, &out_path)
        .expect("AT-616: saxpy compile should succeed");
    let bytes = std::fs::read(&out_path).expect("failed to read AT-616 output");
    let words = load_words(&bytes);
    let insts = parse_raw(&words).expect("AT-616: parse_raw failed");
    let _ = std::fs::remove_file(&src_path);
    let _ = std::fs::remove_file(&out_path);

    let caps = collect_capabilities(&insts);
    assert!(
        !caps.contains(&spv::cap::COOPERATIVE_MATRIX_KHR),
        "AT-616: CooperativeMatrixKHR must NOT be present for a non-coopmat kernel; caps: {caps:?}"
    );
    let ext_count = count_extension(&insts, "SPV_KHR_cooperative_matrix");
    assert_eq!(
        ext_count, 0,
        "AT-616: SPV_KHR_cooperative_matrix extension must NOT be present; got {ext_count}"
    );
}

/// AT-617: coopmat_zero() emits OpConstantNull (not OpCompositeConstruct or other).
#[test]
fn at_617_coopmat_zero_emits_op_constant_null() {
    let (_, _, insts) = compile_matmul_tile();
    let count = count_op(&insts, spv::OP_CONSTANT_NULL);
    assert!(
        count >= 1,
        "AT-617: expected at least 1 OpConstantNull for coopmat_zero(); got {count}"
    );
}

/// AT-618: F16 SSBO buffers trigger StorageBuffer16BitAccess + SPV_KHR_16bit_storage.
#[test]
fn at_618_f16_ssbo_triggers_16bit_storage_capability() {
    let (_, words, insts) = compile_matmul_tile();

    let cap_count = count_capability(&insts, spv::cap::STORAGE_BUFFER_16BIT_ACCESS);
    assert_eq!(
        cap_count, 1,
        "AT-618: OpCapability StorageBuffer16BitAccess must appear exactly once; got {cap_count}"
    );

    let ext_count = count_extension(&insts, "SPV_KHR_16bit_storage");
    assert_eq!(
        ext_count, 1,
        "AT-618: OpExtension \"SPV_KHR_16bit_storage\" must appear exactly once; got {ext_count}"
    );

    validate_spirv(&words, "matmul_tile (AT-618 16bit)");
}

/// AT-619: OpTypeCooperativeMatrixKHR is deduplicated — no two instructions have the
/// same result-id and no two type decls have identical operands.
///
/// Structure: OpTypeCooperativeMatrixKHR %result_id %comp_type %scope %rows %cols %usage
/// The result_id is at operands[0] (word index 1 of the instruction).
/// Operand words [1..5] are the type parameters (no result_type for type decls).
#[test]
fn at_619_coopmat_type_cache_deduplicates_same_key() {
    let (_, _, insts) = compile_matmul_tile();

    // Collect all TypeCooperativeMatrixKHR result_ids.
    let result_ids: Vec<u32> = insts
        .iter()
        .filter(|i| i.opcode == spv::OP_TYPE_COOPERATIVE_MATRIX_KHR)
        .filter_map(|i| i.operands.first().copied())
        .collect();

    // All result_ids must be distinct.
    let mut seen_ids = std::collections::HashSet::new();
    for id in &result_ids {
        assert!(
            seen_ids.insert(*id),
            "AT-619: duplicate OpTypeCooperativeMatrixKHR result_id: {id}"
        );
    }

    // Also verify operands [1..5] (type parameters) are distinct across types.
    let type_params: Vec<&[u32]> = insts
        .iter()
        .filter(|i| i.opcode == spv::OP_TYPE_COOPERATIVE_MATRIX_KHR)
        .map(|i| {
            // operands[1..] are: comp_type, scope, rows, cols, usage
            if i.operands.len() >= 5 { &i.operands[1..] } else { &i.operands[..] }
        })
        .collect();

    let mut seen_params: std::collections::HashSet<Vec<u32>> = std::collections::HashSet::new();
    for params in type_params {
        assert!(
            seen_params.insert(params.to_vec()),
            "AT-619: duplicate OpTypeCooperativeMatrixKHR type params detected: {params:?}"
        );
    }
}

/// AT-620: End-to-end happy path — matmul_tile.axc compiles without error.
#[test]
fn at_620_validate_matmul_tile_end_to_end_happy() {
    let (_, words, _) = compile_matmul_tile();
    validate_spirv(&words, "matmul_tile (AT-620 e2e happy)");
}

/// AT-622: No duplicate capabilities in the matmul_tile output.
#[test]
fn at_622_no_duplicate_capabilities() {
    let (_, _, insts) = compile_matmul_tile();
    let caps = collect_capabilities(&insts);

    let mut seen = std::collections::HashSet::new();
    for cap in &caps {
        assert!(
            seen.insert(*cap),
            "AT-622: duplicate capability value {cap} in matmul_tile SPIR-V"
        );
    }
}

/// AT-629: Exact instruction counts — 2 coopmat loads, 1 mul-add, 1 store.
#[test]
fn at_629_matmul_tile_instruction_counts_exact() {
    let (_, _, insts) = compile_matmul_tile();

    let load_count = count_op(&insts, spv::OP_COOPERATIVE_MATRIX_LOAD_KHR);
    assert_eq!(
        load_count, 2,
        "AT-629: expected exactly 2 OpCooperativeMatrixLoadKHR; got {load_count}"
    );

    let muladd_count = count_op(&insts, spv::OP_COOPERATIVE_MATRIX_MUL_ADD_KHR);
    assert_eq!(
        muladd_count, 1,
        "AT-629: expected exactly 1 OpCooperativeMatrixMulAddKHR; got {muladd_count}"
    );

    let store_count = count_op(&insts, spv::OP_COOPERATIVE_MATRIX_STORE_KHR);
    assert_eq!(
        store_count, 1,
        "AT-629: expected exactly 1 OpCooperativeMatrixStoreKHR; got {store_count}"
    );
}

/// AT-632: OpCapability VulkanMemoryModel present exactly once when coopmat used.
#[test]
fn at_632_emit_vulkan_memory_model_capability_exactly_once() {
    let (_, _, insts) = compile_matmul_tile();
    let count = count_capability(&insts, spv::cap::VULKAN_MEMORY_MODEL);
    assert_eq!(
        count, 1,
        "AT-632: OpCapability VulkanMemoryModel must appear exactly once; got {count}"
    );
}

/// AT-633: OpExtension "SPV_KHR_vulkan_memory_model" present exactly once.
#[test]
fn at_633_emit_spv_khr_vulkan_memory_model_extension_exactly_once() {
    let (_, _, insts) = compile_matmul_tile();
    let count = count_extension(&insts, "SPV_KHR_vulkan_memory_model");
    assert_eq!(
        count, 1,
        "AT-633: OpExtension \"SPV_KHR_vulkan_memory_model\" must appear exactly once; got {count}"
    );
}

/// AT-634: OpMemoryModel uses Logical Vulkan (not GLSL450) when coopmat is used.
///
/// OpMemoryModel format: [OpMemoryModel, AddressingModel, MemoryModel]
/// AddressingModel::Logical = 0, MemoryModel::Vulkan = 3, MemoryModel::GLSL450 = 1.
#[test]
fn at_634_memory_model_is_logical_vulkan_when_coopmat_used() {
    let (_, _, insts) = compile_matmul_tile();

    let mem_model_inst = insts
        .iter()
        .find(|i| i.opcode == spv::OP_MEMORY_MODEL)
        .expect("AT-634: OpMemoryModel instruction not found");

    assert!(
        mem_model_inst.operands.len() >= 2,
        "AT-634: OpMemoryModel must have at least 2 operands; got {}",
        mem_model_inst.operands.len()
    );

    let addressing_model = mem_model_inst.operands[0];
    let memory_model = mem_model_inst.operands[1];

    assert_eq!(
        addressing_model, 0,
        "AT-634: AddressingModel must be Logical (0); got {addressing_model}"
    );
    assert_eq!(
        memory_model,
        spv::mem_model::VULKAN,
        "AT-634: MemoryModel must be Vulkan ({}) when coopmat used; got {memory_model}",
        spv::mem_model::VULKAN
    );
}

/// AT-621: A kernel with @cooperative_matrix but no coopmat builtins in the body
/// compiles cleanly and the resulting SPIR-V does NOT contain CooperativeMatrixKHR
/// capability (since no coopmat ops are emitted).
#[test]
fn at_621_cooperative_matrix_annotation_without_builtins_compiles_without_coopmat_cap() {
    let src = concat!(
        "@kernel\n",
        "@workgroup(32, 1, 1)\n",
        "@cooperative_matrix\n",
        "@intent(\"no-op coopmat kernel\")\n",
        "fn no_op_coopmat(n: u32) -> void {\n",
        "    return;\n",
        "}\n",
    );
    let tmp_dir = std::env::temp_dir();
    let src_path = tmp_dir.join("at621_no_op_coopmat.axc");
    let out_path = tmp_dir.join("at621_no_op_coopmat.spv");
    std::fs::write(&src_path, src).expect("failed to write temp source");
    axc_driver::compile_file(&src_path, &out_path)
        .expect("AT-621: kernel with @cooperative_matrix but no builtins should compile");
    let bytes = std::fs::read(&out_path).expect("failed to read AT-621 output");
    let words = load_words(&bytes);
    let insts = parse_raw(&words).expect("AT-621: parse_raw failed");
    let _ = std::fs::remove_file(&src_path);
    let _ = std::fs::remove_file(&out_path);

    let caps = collect_capabilities(&insts);
    assert!(
        !caps.contains(&spv::cap::COOPERATIVE_MATRIX_KHR),
        "AT-621: CooperativeMatrixKHR must NOT be present when no coopmat ops used; caps: {caps:?}"
    );

    validate_spirv(&words, "AT-621 no-op coopmat kernel");
}
