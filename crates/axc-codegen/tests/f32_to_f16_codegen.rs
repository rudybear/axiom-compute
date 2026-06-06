//! AT-1776: codegen + spirv-val for the `f32_to_f16` builtin (M3.5).
//!
//! A minimal kernel using `f32_to_f16(x)` must:
//!   - emit exactly one `OpFConvert` whose result type is f16 and operand is f32,
//!   - declare `OpCapability Float16`,
//!   - add NO new extension (Float16 needs none),
//!   - pass spirv-val.

use axc_codegen::{emit_module, CodegenOptions};
use axc_hir::lower_module;
use axc_parser::parse;
use rspirv::binary::Disassemble;

/// Compile a source string to SPIR-V words.
fn compile(src: &str) -> Vec<u32> {
    let (ast, lex_errs, parse_errs) = parse(src);
    assert!(lex_errs.is_empty(), "lex errors: {lex_errs:?}");
    assert!(parse_errs.is_empty(), "parse errors: {parse_errs:?}");
    let (hir, hir_errs, _warns) = lower_module(&ast);
    assert!(hir_errs.is_empty(), "hir errors: {hir_errs:?}");
    emit_module(&hir, &CodegenOptions::default()).expect("codegen must succeed")
}

/// Disassemble words to text.
fn disasm(words: &[u32]) -> String {
    use rspirv::dr::Loader;
    let mut loader = Loader::new();
    rspirv::binary::parse_words(words, &mut loader).expect("rspirv parse");
    loader.module().disassemble()
}

/// Minimal kernel: read f32 from `inp`, narrow to f16 via the builtin, store to `out`.
const F32_TO_F16_SRC: &str = "
    @kernel @workgroup(1,1,1) @intent(\"f32_to_f16 codegen anchor\") @complexity(O(1))
    fn narrow(inp: readonly_buffer[f32], out: buffer[f16]) -> void {
        let v: f32 = inp[0u32];
        let h: f16 = f32_to_f16(v);
        out[0u32] = h;
        return;
    }
";

/// AT-1776: f32_to_f16 emits exactly one OpFConvert (f32→f16), declares Float16,
/// adds no extension, and passes spirv-val.
#[test]
fn at_1776_f32_to_f16_emits_fconvert_and_validates() {
    let words = compile(F32_TO_F16_SRC);
    let asm = disasm(&words);

    // The disassembly MUST contain an OpFConvert.
    assert!(
        asm.contains("OpFConvert"),
        "AT-1776: f32_to_f16 must emit OpFConvert; disasm:\n{asm}"
    );

    // OpCapability Float16 MUST be present (the only capability the builtin needs).
    assert!(
        asm.contains("OpCapability Float16"),
        "AT-1776: f32_to_f16 kernel must declare OpCapability Float16; disasm:\n{asm}"
    );

    // Structural raw-word check: OpFConvert (opcode 115) with an f16 result type.
    // First find the f16 type id (OpTypeFloat width 16). OpTypeFloat = opcode 22.
    const OP_TYPE_FLOAT: u32 = 22;
    const OP_FCONVERT: u32 = 115;
    let mut f16_type_id: Option<u32> = None;
    let mut i = 5usize; // skip the 5-word SPIR-V header
    while i < words.len() {
        let opcode = words[i] & 0xFFFF;
        let wc = (words[i] >> 16) as usize;
        if wc == 0 {
            break;
        }
        if opcode == OP_TYPE_FLOAT && wc == 3 && words[i + 2] == 16 {
            // OpTypeFloat %result 16  → result id is words[i+1].
            f16_type_id = Some(words[i + 1]);
        }
        i += wc;
    }
    let f16_type_id = f16_type_id.expect("AT-1776: kernel must declare an f16 OpTypeFloat(16)");

    // Count OpFConvert instructions whose result type == the f16 type id.
    let mut fconvert_to_f16_count = 0usize;
    let mut j = 5usize;
    while j < words.len() {
        let opcode = words[j] & 0xFFFF;
        let wc = (words[j] >> 16) as usize;
        if wc == 0 {
            break;
        }
        // OpFConvert: [wc<<16|115, result_type, result_id, operand].
        if opcode == OP_FCONVERT && wc == 4 && words[j + 1] == f16_type_id {
            fconvert_to_f16_count += 1;
        }
        j += wc;
    }
    assert_eq!(
        fconvert_to_f16_count, 1,
        "AT-1776: exactly one OpFConvert producing the f16 type expected (the narrowing); \
         got {fconvert_to_f16_count}; disasm:\n{asm}"
    );

    // NO new extension: the only extensions a Float16-only kernel may emit are NONE
    // for f32_to_f16 itself. Assert the builtin did not pull a storage/cooperative
    // extension (this minimal kernel has no coopmat / 8-bit buffer).
    assert!(
        !asm.contains("OpExtension \"SPV_KHR_cooperative_matrix\""),
        "AT-1776: f32_to_f16 must NOT require the cooperative-matrix extension"
    );
    assert!(
        !asm.contains("OpExtension \"SPV_KHR_8bit_storage\""),
        "AT-1776: f32_to_f16 must NOT require the 8-bit-storage extension"
    );

    // spirv-val must pass.
    use spirv_tools::val::{create as create_validator, Validator};
    use spirv_tools::TargetEnv;
    let validator = create_validator(Some(TargetEnv::Vulkan_1_1));
    validator
        .validate(&words, None)
        .expect("AT-1776: f32_to_f16 SPIR-V must pass spirv-val");
}
