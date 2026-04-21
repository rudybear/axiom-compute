//! M2.5 integration tests for `examples/q4_0_dequant_matvec.axc`.
//!
//! AT-901: q4_0_dequant_matvec compiles to valid SPIR-V (header words correct)
//! AT-902: emitted SPIR-V passes in-process spirv-tools validation (Vulkan 1.1)
//! AT-903: OpCapability Int8 is present in emitted SPIR-V
//! AT-904: OpCapability StorageBuffer8BitAccess is present in emitted SPIR-V
//! AT-905: OpExtension "SPV_KHR_8bit_storage" is present in emitted SPIR-V
//! AT-906: OpCapability Int16 is present in emitted SPIR-V (for f16_bits_to_f32)
//! AT-907: OpCapability Float16 is present in emitted SPIR-V (for OpFConvert f16→f32)
//! AT-908: Exactly 3 OpVariable StorageBuffer (q, x, y buffers)
//! AT-909: q (binding=0) has NonWritable decoration (readonly_buffer)
//! AT-910: x (binding=1) has NonWritable decoration (readonly_buffer)
//! AT-911: y (binding=2) has no NonWritable / NonReadable decoration (readwrite)
//! AT-912: Push-constant block present with u32 n_blocks scalar
//! AT-913: OpFConvert instruction is present (f16 → f32 widening for scale)
//! AT-914: No duplicate OpCapability Int8 (emitted exactly once)
//! AT-915: No duplicate OpExtension "SPV_KHR_8bit_storage" (emitted exactly once)
//! AT-916: OpLoad with StorageBuffer pointer is present (ptr_read_u8_zext emits OpLoad)
//! AT-917: No OpCapability StorageBuffer16BitAccess (no f16 SSBOs in this kernel)
//! AT-918: GPU dispatch test (AXC_ENABLE_GPU_TESTS=1 required)

use std::path::PathBuf;

// ── Raw SPIR-V binary scanner (same approach as compile_matmul_tile.rs) ─────────
// Avoids rspirv::load_words which may reject new opcodes.

/// SPIR-V opcode constants.
mod spv {
    pub const OP_CAPABILITY:          u16 = 17;
    pub const OP_EXTENSION:           u16 = 10;
    pub const OP_TYPE_INT:            u16 = 21;
    pub const OP_TYPE_FLOAT:          u16 = 22;
    pub const OP_VARIABLE:            u16 = 59;
    pub const OP_F_CONVERT:           u16 = 115;
    pub const OP_LOAD:                u16 = 61;

    /// Capability enum values.
    pub mod cap {
        pub const SHADER:                        u32 = 1;
        pub const INT8:                          u32 = 39;
        pub const INT16:                         u32 = 22;
        pub const INT64:                         u32 = 11;
        pub const FLOAT16:                       u32 = 9;
        pub const FLOAT64:                       u32 = 10;
        pub const STORAGE_BUFFER_8BIT_ACCESS:    u32 = 4448;
        pub const STORAGE_BUFFER_16BIT_ACCESS:   u32 = 4433;
    }

    /// StorageClass enum values.
    pub mod sc {
        pub const STORAGE_BUFFER: u32 = 12;
        pub const PUSH_CONSTANT:  u32 = 9;
    }

    /// Decoration enum values.
    pub mod dec {
        pub const NON_WRITABLE:  u32 = 24;
        pub const NON_READABLE:  u32 = 25;
        pub const BINDING:       u32 = 33;
        pub const DESCRIPTOR_SET: u32 = 34;
    }
}

/// A parsed SPIR-V instruction.
struct RawInst {
    opcode: u16,
    operands: Vec<u32>,
}

/// Parse a SPIR-V word stream into a flat list of `RawInst`.
fn parse_raw(words: &[u32]) -> Vec<RawInst> {
    // Skip the 5-word header.
    assert!(words.len() >= 5, "SPIR-V too short for header");
    let mut pos = 5;
    let mut out = Vec::new();
    while pos < words.len() {
        let instr_word = words[pos];
        let word_count = (instr_word >> 16) as usize;
        let opcode = (instr_word & 0xFFFF) as u16;
        assert!(word_count > 0 && pos + word_count <= words.len(),
            "malformed SPIR-V instruction at word {pos}");
        let operands: Vec<u32> = words[pos + 1..pos + word_count].to_vec();
        out.push(RawInst { opcode, operands });
        pos += word_count;
    }
    out
}

/// Decode a packed SPIR-V string operand (one or more u32 words, NUL-terminated).
fn decode_spirv_string(words: &[u32]) -> String {
    let mut bytes: Vec<u8> = Vec::new();
    'outer: for &w in words {
        for byte in w.to_le_bytes() {
            if byte == 0 {
                break 'outer;
            }
            bytes.push(byte);
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Load words from a SPIR-V byte blob.
fn load_words(bytes: &[u8]) -> Vec<u32> {
    assert!(bytes.len() % 4 == 0, "SPIR-V byte length must be divisible by 4");
    let n = bytes.len() / 4;
    let mut words = Vec::with_capacity(n);
    for i in 0..n {
        words.push(u32::from_le_bytes([bytes[4*i], bytes[4*i+1], bytes[4*i+2], bytes[4*i+3]]));
    }
    words
}

/// Validate SPIR-V words using the in-process spirv-tools crate.
fn validate_spirv(words: &[u32], label: &str) {
    use spirv_tools::val::{Validator, create as create_validator};
    use spirv_tools::TargetEnv;
    let validator = create_validator(Some(TargetEnv::Vulkan_1_1));
    validator.validate(words, None)
        .unwrap_or_else(|e| panic!("spirv-tools rejected {label} SPIR-V: {e}"));
}

/// Compile `examples/q4_0_dequant_matvec.axc` and return the SPIR-V word stream.
fn compile_q4_0() -> Vec<u32> {
    let manifest_dir = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"),
    );
    let examples_dir = manifest_dir.join("..").join("..").join("examples");
    let source_path = examples_dir.join("q4_0_dequant_matvec.axc");

    assert!(
        source_path.exists(),
        "examples/q4_0_dequant_matvec.axc not found at {:?}", source_path
    );

    let tmp_dir = std::env::temp_dir();
    // Use a unique component to avoid inter-test races when tests run in parallel.
    let unique = format!(
        "axc_test_q4_0_{}.spv",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos() ^ ((d.as_secs() << 16) as u32))
            .unwrap_or_else(|_| {
                // Fallback: use stack address as entropy.
                let x: u32 = 0;
                &x as *const u32 as usize as u32
            }),
    );
    let out_path = tmp_dir.join(&unique);

    axc_driver::compile_file(&source_path, &out_path)
        .expect("compile_file should succeed for examples/q4_0_dequant_matvec.axc");

    let spv_bytes = std::fs::read(&out_path).expect("failed to read output .spv");
    let _ = std::fs::remove_file(&out_path);
    assert!(spv_bytes.len() >= 20, "SPIR-V too short: {} bytes", spv_bytes.len());
    load_words(&spv_bytes)
}

// ── AT-901: Header words correct ─────────────────────────────────────────────

/// AT-901: q4_0_dequant_matvec compiles to valid SPIR-V with correct header.
#[test]
fn at_901_q4_0_compiles_to_valid_spirv_header() {
    let words = compile_q4_0();
    assert_eq!(words[0], 0x0723_0203_u32, "AT-901: magic word mismatch");
    assert_eq!(words[1], 0x0001_0300_u32, "AT-901: version must be SPIR-V 1.3");
    assert_eq!(words[2], 0x0000_0000_u32, "AT-901: generator must be 0");
}

// ── AT-902: In-process spirv-tools validation ─────────────────────────────────

/// AT-902: emitted SPIR-V passes in-process spirv-tools validation (Vulkan 1.1).
#[test]
fn at_902_q4_0_spirv_passes_validator() {
    let words = compile_q4_0();
    validate_spirv(&words, "q4_0_dequant_matvec");
}

// ── AT-903: OpCapability Int8 ─────────────────────────────────────────────────

/// AT-903: OpCapability Int8 is present in emitted SPIR-V.
#[test]
fn at_903_q4_0_emits_capability_int8() {
    let words = compile_q4_0();
    let insts = parse_raw(&words);
    let has_int8 = insts.iter().any(|inst| {
        inst.opcode == spv::OP_CAPABILITY
            && inst.operands.first() == Some(&spv::cap::INT8)
    });
    assert!(has_int8, "AT-903: expected OpCapability Int8 in q4_0 SPIR-V");
}

// ── AT-904: OpCapability StorageBuffer8BitAccess ──────────────────────────────

/// AT-904: OpCapability StorageBuffer8BitAccess is present in emitted SPIR-V.
#[test]
fn at_904_q4_0_emits_capability_storage_buffer_8bit_access() {
    let words = compile_q4_0();
    let insts = parse_raw(&words);
    let has_cap = insts.iter().any(|inst| {
        inst.opcode == spv::OP_CAPABILITY
            && inst.operands.first() == Some(&spv::cap::STORAGE_BUFFER_8BIT_ACCESS)
    });
    assert!(has_cap, "AT-904: expected OpCapability StorageBuffer8BitAccess in q4_0 SPIR-V");
}

// ── AT-905: OpExtension "SPV_KHR_8bit_storage" ───────────────────────────────

/// AT-905: OpExtension "SPV_KHR_8bit_storage" is present in emitted SPIR-V.
#[test]
fn at_905_q4_0_emits_extension_spv_khr_8bit_storage() {
    let words = compile_q4_0();
    let insts = parse_raw(&words);
    let has_ext = insts.iter().any(|inst| {
        inst.opcode == spv::OP_EXTENSION
            && decode_spirv_string(&inst.operands) == "SPV_KHR_8bit_storage"
    });
    assert!(has_ext, "AT-905: expected OpExtension \"SPV_KHR_8bit_storage\" in q4_0 SPIR-V");
}

// ── AT-906: OpCapability Int16 ────────────────────────────────────────────────

/// AT-906: OpCapability Int16 is present (for f16_bits_to_f32 intermediate u16).
#[test]
fn at_906_q4_0_emits_capability_int16() {
    let words = compile_q4_0();
    let insts = parse_raw(&words);
    let has_int16 = insts.iter().any(|inst| {
        inst.opcode == spv::OP_CAPABILITY
            && inst.operands.first() == Some(&spv::cap::INT16)
    });
    assert!(has_int16, "AT-906: expected OpCapability Int16 in q4_0 SPIR-V");
}

// ── AT-907: OpCapability Float16 ─────────────────────────────────────────────

/// AT-907: OpCapability Float16 is present (for OpBitcast u16→f16 and OpFConvert).
#[test]
fn at_907_q4_0_emits_capability_float16() {
    let words = compile_q4_0();
    let insts = parse_raw(&words);
    let has_float16 = insts.iter().any(|inst| {
        inst.opcode == spv::OP_CAPABILITY
            && inst.operands.first() == Some(&spv::cap::FLOAT16)
    });
    assert!(has_float16, "AT-907: expected OpCapability Float16 in q4_0 SPIR-V");
}

// ── AT-908: Exactly 3 OpVariable StorageBuffer ────────────────────────────────

/// AT-908: Exactly 3 OpVariable StorageBuffer (q binding=0, x binding=1, y binding=2).
#[test]
fn at_908_q4_0_has_three_ssbo_variables() {
    let words = compile_q4_0();
    let insts = parse_raw(&words);
    // OpVariable has operands [result_type, result_id?, storage_class, ...]
    // In SPIR-V binary: OpVariable words = [opcode+wc, result_type, result_id, storage_class]
    // In our parser: operands[0]=result_type, operands[1]=result_id (implicit), operands[2]=storage_class
    // Actually for OpVariable: result_type_id, result_id, storage_class [, initializer]
    // Our RawInst operands = words after the opcode/wc word, so:
    // operands[0] = result_type, operands[1] = result_id, operands[2] = storage_class
    let ssbo_var_count = insts.iter().filter(|inst| {
        inst.opcode == spv::OP_VARIABLE
            && inst.operands.get(2) == Some(&spv::sc::STORAGE_BUFFER)
    }).count();
    assert_eq!(
        ssbo_var_count, 3,
        "AT-908: expected exactly 3 StorageBuffer OpVariables (q, x, y); got {ssbo_var_count}"
    );
}

// ── AT-909: q (binding=0) has NonWritable ────────────────────────────────────

/// AT-909: q is a readonly_buffer → NonWritable decoration must be present.
///
/// This test checks that at least 2 NonWritable decorations exist (q and x are both readonly).
#[test]
fn at_909_q4_0_q_and_x_have_nonwritable_decoration() {
    let words = compile_q4_0();
    let insts = parse_raw(&words);
    // OpDecorate: operands[0]=target, operands[1]=decoration value
    let non_writable_count = insts.iter().filter(|inst| {
        // OpDecorate opcode = 71
        inst.opcode == 71
            && inst.operands.get(1) == Some(&spv::dec::NON_WRITABLE)
    }).count();
    assert!(
        non_writable_count >= 2,
        "AT-909: expected ≥2 NonWritable decorations (q and x are readonly_buffer); got {non_writable_count}"
    );
}

// ── AT-910: x (binding=1) has NonWritable ────────────────────────────────────

/// AT-910: x is a readonly_buffer → its NonWritable decoration is included in AT-909 count.
/// This test verifies there are exactly 2 NonWritable decorations (q and x only).
#[test]
fn at_910_q4_0_exactly_two_nonwritable_decorations() {
    let words = compile_q4_0();
    let insts = parse_raw(&words);
    let non_writable_count = insts.iter().filter(|inst| {
        inst.opcode == 71
            && inst.operands.get(1) == Some(&spv::dec::NON_WRITABLE)
    }).count();
    assert_eq!(
        non_writable_count, 2,
        "AT-910: expected exactly 2 NonWritable decorations (q and x); got {non_writable_count}"
    );
}

// ── AT-911: y (binding=2) has no access decoration ───────────────────────────

/// AT-911: y is a readwrite buffer → no NonReadable decoration in the module.
#[test]
fn at_911_q4_0_no_nonreadable_decoration() {
    let words = compile_q4_0();
    let insts = parse_raw(&words);
    let non_readable_count = insts.iter().filter(|inst| {
        inst.opcode == 71
            && inst.operands.get(1) == Some(&spv::dec::NON_READABLE)
    }).count();
    assert_eq!(
        non_readable_count, 0,
        "AT-911: expected 0 NonReadable decorations (y is readwrite buffer); got {non_readable_count}"
    );
}

// ── AT-912: Push-constant block with n_blocks scalar ─────────────────────────

/// AT-912: Push-constant block is present (one PushConstant OpVariable for n_blocks).
#[test]
fn at_912_q4_0_has_push_constant_variable() {
    let words = compile_q4_0();
    let insts = parse_raw(&words);
    let pc_var_count = insts.iter().filter(|inst| {
        inst.opcode == spv::OP_VARIABLE
            && inst.operands.get(2) == Some(&spv::sc::PUSH_CONSTANT)
    }).count();
    assert_eq!(
        pc_var_count, 1,
        "AT-912: expected exactly 1 PushConstant OpVariable (n_blocks); got {pc_var_count}"
    );
}

// ── AT-913: OpFConvert present ────────────────────────────────────────────────

/// AT-913: OpFConvert instruction is present (f16 → f32 widening in f16_bits_to_f32).
#[test]
fn at_913_q4_0_emits_op_fconvert() {
    let words = compile_q4_0();
    let insts = parse_raw(&words);
    let has_fconvert = insts.iter().any(|inst| inst.opcode == spv::OP_F_CONVERT);
    assert!(has_fconvert, "AT-913: expected OpFConvert in q4_0 SPIR-V (f16→f32 for scale)");
}

// ── AT-914: No duplicate OpCapability Int8 ────────────────────────────────────

/// AT-914: OpCapability Int8 must appear exactly once.
#[test]
fn at_914_q4_0_capability_int8_exactly_once() {
    let words = compile_q4_0();
    let insts = parse_raw(&words);
    let int8_count = insts.iter().filter(|inst| {
        inst.opcode == spv::OP_CAPABILITY
            && inst.operands.first() == Some(&spv::cap::INT8)
    }).count();
    assert_eq!(int8_count, 1, "AT-914: OpCapability Int8 must appear exactly once; got {int8_count}");
}

// ── AT-915: No duplicate OpExtension "SPV_KHR_8bit_storage" ──────────────────

/// AT-915: OpExtension "SPV_KHR_8bit_storage" must appear exactly once.
#[test]
fn at_915_q4_0_extension_8bit_storage_exactly_once() {
    let words = compile_q4_0();
    let insts = parse_raw(&words);
    let ext_count = insts.iter().filter(|inst| {
        inst.opcode == spv::OP_EXTENSION
            && decode_spirv_string(&inst.operands) == "SPV_KHR_8bit_storage"
    }).count();
    assert_eq!(ext_count, 1, "AT-915: OpExtension SPV_KHR_8bit_storage must appear exactly once; got {ext_count}");
}

// ── AT-916: OpLoad for StorageBuffer pointer ──────────────────────────────────

/// AT-916: At least one OpLoad instruction is present (ptr_read_u8_zext emits OpLoad).
#[test]
fn at_916_q4_0_has_op_load_instructions() {
    let words = compile_q4_0();
    let insts = parse_raw(&words);
    let load_count = insts.iter().filter(|inst| inst.opcode == spv::OP_LOAD).count();
    assert!(
        load_count >= 1,
        "AT-916: expected at least one OpLoad instruction (ptr_read_u8_zext); got {load_count}"
    );
}

// ── AT-917: No StorageBuffer16BitAccess ──────────────────────────────────────

/// AT-917: No OpCapability StorageBuffer16BitAccess (no f16 SSBOs in this kernel;
/// the f16 scale is accessed via ptr_read_u16_zext which uses Int8 not 16bit storage).
#[test]
fn at_917_q4_0_no_storage_16bit_access_capability() {
    let words = compile_q4_0();
    let insts = parse_raw(&words);
    let has_16bit = insts.iter().any(|inst| {
        inst.opcode == spv::OP_CAPABILITY
            && inst.operands.first() == Some(&spv::cap::STORAGE_BUFFER_16BIT_ACCESS)
    });
    assert!(
        !has_16bit,
        "AT-917: q4_0 kernel must NOT emit OpCapability StorageBuffer16BitAccess \
         (no f16 SSBOs; scale is read via u8 ptr_read + bit conversion)"
    );
}

// ── AT-918: GPU dispatch test ─────────────────────────────────────────────────

/// AT-918: GPU dispatch — Q4_0 dequant+matvec result matches CPU reference within 1e-3.
///
/// Requires `AXC_ENABLE_GPU_TESTS=1` environment variable to run.
/// Uses Lavapipe (software Vulkan) as the CI fallback.
///
/// The test compiles `q4_0_dequant_matvec.axc`, dispatches a single workgroup
/// (64 invocations, single-output kernel), and checks that `y[0]` matches the
/// CPU reference within a relative tolerance of `1e-3`.
#[test]
#[ignore]
fn at_918_q4_0_dispatch_matches_cpu_reference() {
    if std::env::var("AXC_ENABLE_GPU_TESTS").unwrap_or_default() != "1" {
        eprintln!("AT-918: skipped (AXC_ENABLE_GPU_TESTS != 1)");
        return;
    }

    // Load the kernel source.
    let manifest_dir = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"),
    );
    let examples_dir = manifest_dir.join("..").join("..").join("examples");
    let source_path = examples_dir.join("q4_0_dequant_matvec.axc");
    let src = std::fs::read_to_string(&source_path)
        .expect("AT-918: failed to read q4_0_dequant_matvec.axc");

    // Generate test inputs.
    #[path = "../benches/common.rs"]
    mod common;

    const N_BLOCKS: usize = 128;
    const SEED: u64 = 42;
    let (q_bytes, x_vec) = common::make_q4_0_inputs(N_BLOCKS, SEED);
    let cpu_result: f32 = common::q4_0_dequant_matvec_cpu(&q_bytes, &x_vec, N_BLOCKS);

    // Compile the kernel and lower to HIR (need binding plan for DispatchRequest).
    let (tokens, _) = axc_lexer::tokenize(&src);
    let mut parser = axc_parser::Parser::new(&tokens);
    let (ast, _) = parser.parse_module();
    let (hir, errs, _warnings) = axc_hir::lower_module(&ast);
    assert!(errs.is_empty(), "AT-918: HIR errors: {errs:?}");

    let spv_words = axc_codegen::emit_module(&hir, &Default::default())
        .expect("AT-918: codegen failed");

    let kernel = hir.kernels.first().expect("AT-918: HIR must have at least one kernel");
    let binding_plan = &kernel.binding_plan;

    // Set up Vulkan runtime.
    use axc_runtime::{VulkanContext, DispatchRequest};

    let ctx = match VulkanContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("AT-918: skipped — VulkanContext::new() failed: {e}");
            return;
        }
    };

    // Buffer sizes (bytes).
    let q_buf_size: usize = N_BLOCKS * common::Q4_0_BLOCK_BYTES;
    let x_buf_size: usize = N_BLOCKS * common::Q4_0_BLOCK_ELEMS * 4;
    let y_buf_size: usize = 4; // single f32 output

    // Build push-constant bytes (n_blocks: u32 at offset 0).
    let n_blocks_val: u32 = N_BLOCKS as u32;
    let mut push_data: Vec<u8> = vec![0u8; binding_plan.push_constant_total_bytes as usize];
    for scalar in &binding_plan.scalars {
        let start: usize = scalar.offset as usize;
        match scalar.ty {
            axc_hir::ScalarTy::U32 => {
                push_data[start..start + 4].copy_from_slice(&n_blocks_val.to_le_bytes());
            }
            _ => {}
        }
    }

    // Input data bytes.
    let x_bytes: Vec<u8> = x_vec.iter().flat_map(|v| v.to_le_bytes()).collect();
    let y_bytes_init: Vec<u8> = vec![0u8; y_buf_size];

    // Single workgroup covers all 64 invocations (single-output kernel).
    let workgroups: [u32; 3] = [1, 1, 1];

    let req = DispatchRequest {
        spirv: &spv_words,
        binding_plan,
        workgroups,
        inputs: &[q_bytes.as_slice(), x_bytes.as_slice(), y_bytes_init.as_slice()],
        output_sizes: &[q_buf_size, x_buf_size, y_buf_size],
        push_constants: &push_data,
        entry_point: "q4_0_dequant_matvec",
    };

    let outputs: Vec<Vec<u8>> = ctx
        .dispatch(req)
        .expect("AT-918: GPU dispatch failed");

    // y is the third buffer (index 2).
    let y_out: &[u8] = &outputs[2];
    assert!(y_out.len() >= 4, "AT-918: y output too short: {} bytes", y_out.len());
    let gpu_result = f32::from_le_bytes([y_out[0], y_out[1], y_out[2], y_out[3]]);

    // Check relative tolerance (1e-3).
    let rel_err = if cpu_result.abs() > 1e-10 {
        (gpu_result - cpu_result).abs() / cpu_result.abs()
    } else {
        (gpu_result - cpu_result).abs()
    };

    assert!(
        rel_err < 1e-3,
        "AT-918: GPU result {gpu_result} vs CPU {cpu_result}: relative error {rel_err:.6} exceeds 1e-3"
    );
}
