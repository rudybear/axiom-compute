//! spirv-val-gated codegen tests for local arrays (M3.20).
//!
//! Covers AT-2935..AT-2936. Mirrors `shared_codegen.rs`'s style — structural
//! disassembly checks are the in-process guard here; real spirv-tools validation
//! runs in the `axc-driver` integration tests (`compile_local_array_examples.rs`).

use axc_codegen::{emit_module, CodegenOptions};
use axc_hir::lower_module;
use axc_parser::parse;
use rspirv::binary::Disassemble;

/// Compile a source string and return the SPIR-V words.
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

/// Extract the set of `OpCapability` values from a SPIR-V word stream.
fn capability_set(words: &[u32]) -> std::collections::BTreeSet<u32> {
    let mut caps = std::collections::BTreeSet::new();
    let mut i = 5usize; // skip the 5-word module header
    while i < words.len() {
        let opcode = words[i] & 0xFFFF;
        let wc = (words[i] >> 16) as usize;
        if wc == 0 {
            break;
        }
        if opcode == 17 && wc == 2 {
            caps.insert(words[i + 1]);
        }
        i += wc;
    }
    caps
}

// ── AT-2935: structural — OpTypeArray, 2x OpTypePointer Function, OpVariable
//    precedes all non-var in first block, single-index access chain, zero OpPhi,
//    capability set byte-identical to a plain-let-scalar equivalent ────────────

#[test]
fn at2935_u32_array_structural_shape() {
    let src = "
        @kernel @workgroup(1,1,1) @intent(\"test\") @complexity(O(1))
        fn k(out: buffer[u32]) -> void {
            array hist: array[u32, 8];
            hist[0u32] = 1u32;
            let v: u32 = hist[0u32];
            out[0u32] = v;
            return;
        }
    ";
    let words = compile(src);
    let asm = disasm(&words);

    assert!(asm.contains("OpTypeArray"), "must have OpTypeArray; asm:\n{asm}");
    let ptr_function_count = asm.lines()
        .filter(|l| l.contains("OpTypePointer") && l.contains("Function"))
        .count();
    assert!(
        ptr_function_count >= 2,
        "must have >= 2 OpTypePointer Function (array-ptr + elem-ptr); asm:\n{asm}"
    );
    assert!(asm.contains("OpAccessChain"), "must have OpAccessChain; asm:\n{asm}");
    assert!(!asm.contains("OpPhi"), "local arrays must never emit OpPhi; asm:\n{asm}");

    // No new capability vs the SAME kernel with a plain `let` scalar instead of
    // the local array — Function storage is core Shader (on-demand emission).
    let src_plain = "
        @kernel @workgroup(1,1,1) @intent(\"test\") @complexity(O(1))
        fn k(out: buffer[u32]) -> void {
            let v: u32 = 1u32;
            out[0u32] = v;
            return;
        }
    ";
    let plain_words = compile(src_plain);
    assert_eq!(
        capability_set(&words), capability_set(&plain_words),
        "u32 local-array kernel must have a BYTE-IDENTICAL capability set to the plain-let-scalar equivalent"
    );
}

/// AT-2935: the `OpVariable Function` for the local array precedes ALL
/// non-`OpVariable` instructions in the function's first block (SPIR-V S2.16.1).
#[test]
fn at2935_opvariable_precedes_non_variable_in_first_block() {
    let src = "
        @kernel @workgroup(1,1,1) @intent(\"test\") @complexity(O(1))
        fn k(out: buffer[u32]) -> void {
            array hist: array[u32, 8];
            hist[0u32] = 1u32;
            out[0u32] = hist[0u32];
            return;
        }
    ";
    let words = compile(src);
    let asm = disasm(&words);

    // Within the function body (after OpFunction), collect instruction opcodes in
    // order and verify no non-OpVariable line precedes an OpVariable line.
    let mut in_function = false;
    let mut seen_non_variable = false;
    for line in asm.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("%") && trimmed.contains("= OpFunction") {
            in_function = true;
            continue;
        }
        if !in_function {
            continue;
        }
        if trimmed.contains("OpFunctionEnd") {
            break;
        }
        if trimmed.contains("OpLabel") || trimmed.contains("OpFunctionParameter") {
            continue;
        }
        if trimmed.contains("OpVariable") {
            assert!(
                !seen_non_variable,
                "an OpVariable appeared after a non-OpVariable instruction in the first block; asm:\n{asm}"
            );
        } else if !trimmed.is_empty() {
            seen_non_variable = true;
        }
    }
    assert!(in_function, "did not find OpFunction in disassembly; asm:\n{asm}");
}

// ── AT-2936: capability flags — f16 sets Float16, u8 sets Int8; no
//    StorageBuffer8/16BitAccess / SPV_KHR_*_storage (Function is non-StorageBuffer) ──

#[test]
fn at2936_f16_array_sets_float16_no_storage_buffer_cap() {
    let src = "
        @kernel @workgroup(1,1,1) @intent(\"test\") @complexity(O(1))
        fn k() -> void {
            array a: array[f16, 4];
            return;
        }
    ";
    let words = compile(src);
    let asm = disasm(&words);

    assert!(
        asm.contains("OpCapability Float16"),
        "array[f16] declaration must emit OpCapability Float16; asm:\n{asm}"
    );
    assert!(
        !asm.contains("StorageBuffer16BitAccess"),
        "array[f16] (Function storage) must NOT emit StorageBuffer16BitAccess; asm:\n{asm}"
    );
    assert!(
        !asm.contains("SPV_KHR_16bit_storage"),
        "array[f16] must NOT emit SPV_KHR_16bit_storage; asm:\n{asm}"
    );
}

#[test]
fn at2936_u8_array_sets_int8_no_storage_buffer_cap() {
    let src = "
        @kernel @workgroup(1,1,1) @intent(\"test\") @complexity(O(1))
        fn k() -> void {
            array a: array[u8, 4];
            return;
        }
    ";
    let words = compile(src);
    let asm = disasm(&words);

    assert!(
        asm.contains("OpCapability Int8"),
        "array[u8] declaration must emit OpCapability Int8; asm:\n{asm}"
    );
    assert!(
        !asm.contains("StorageBuffer8BitAccess"),
        "array[u8] (Function storage) must NOT emit StorageBuffer8BitAccess; asm:\n{asm}"
    );
    assert!(
        !asm.contains("SPV_KHR_8bit_storage"),
        "array[u8] must NOT emit SPV_KHR_8bit_storage; asm:\n{asm}"
    );
}

// ── On-demand emission: a kernel with NO local arrays is byte-identical to the
//    pre-M3.20 baseline shape (no OpTypeArray/local-array-only OpTypePointer noise) ──

#[test]
fn no_local_arrays_emits_no_array_types() {
    let src = "
        @kernel @workgroup(1,1,1) @intent(\"test\") @complexity(O(1))
        fn k(out: buffer[u32]) -> void {
            let v: u32 = 1u32;
            out[0u32] = v;
            return;
        }
    ";
    let words = compile(src);
    let asm = disasm(&words);
    assert!(
        !asm.contains("OpTypeArray"),
        "a kernel with no local arrays must not emit OpTypeArray; asm:\n{asm}"
    );
}
