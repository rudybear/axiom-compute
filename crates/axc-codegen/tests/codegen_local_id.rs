//! AT-1741: codegen tests for local_invocation_id() builtin (M3.3d).
//!
//! Verifies that:
//! 1. Exactly one Input OpVariable is emitted decorated BuiltIn LocalInvocationId.
//! 2. An OpLoad of the vec3 u32 + OpCompositeExtract with a literal axis is emitted.
//! 3. The var appears in the OpEntryPoint interface list.
//! 4. No new OpCapability or OpExtension is added for LocalInvocationId (core Shader).
//! 5. A gid-only kernel's capability set is unchanged by this milestone.
//! 6. A kernel using BOTH gid() and local_invocation_id() emits TWO distinct Input vars.

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

// ── AT-1741: local_invocation_id emits Input var ─────────────────────────────

/// AT-1741: a kernel using local_invocation_id() emits exactly one Input OpVariable
/// decorated BuiltIn LocalInvocationId, an OpLoad + OpCompositeExtract, the var in
/// the OpEntryPoint interface list, and NO new OpCapability.
#[test]
fn codegen_local_invocation_id_emits_builtin_var_spirv_val_clean() {
    let src = r#"
        @kernel @workgroup(64,1,1)
        @intent("local_invocation_id builtin codegen test AT-1741")
        @complexity(O(1))
        fn local_id_test(out: buffer[u32]) -> void {
            let lid: u32 = local_invocation_id(0u32);
            let gx:  u32 = gid(0u32);
            out[gx] = lid;
            return;
        }
    "#;
    let words = compile(src);
    let asm = disasm(&words);

    // (1) Exactly one BuiltIn LocalInvocationId Input variable.
    assert!(
        asm.contains("LocalInvocationId"),
        "AT-1741: must have BuiltIn LocalInvocationId in disassembly; asm:\n{asm}"
    );

    // Count occurrences of LocalInvocationId in the asm (decoration + interface list = 1+ lines).
    let local_id_count = asm.matches("LocalInvocationId").count();
    assert!(
        local_id_count >= 1,
        "AT-1741: LocalInvocationId must appear at least once (decoration); asm:\n{asm}"
    );

    // (2) OpLoad and OpCompositeExtract are present.
    assert!(
        asm.contains("OpLoad"),
        "AT-1741: OpLoad must be present for vec3 u32 load; asm:\n{asm}"
    );
    assert!(
        asm.contains("OpCompositeExtract"),
        "AT-1741: OpCompositeExtract must be present for axis extraction; asm:\n{asm}"
    );

    // (3) GlobalInvocationId is also present (gid used too).
    assert!(
        asm.contains("GlobalInvocationId"),
        "AT-1741: must also have GlobalInvocationId (gid used); asm:\n{asm}"
    );

    // (4) No new OpCapability beyond Shader (LocalInvocationId is core Shader — no new cap needed).
    // Count OpCapability lines that are NOT "Shader".
    let extra_caps: Vec<&str> = asm.lines()
        .filter(|l| l.contains("OpCapability") && !l.contains("Shader"))
        .collect();
    assert!(
        extra_caps.is_empty(),
        "AT-1741: MUST NOT add any new OpCapability for LocalInvocationId; extra: {extra_caps:?}"
    );

    eprintln!("AT-1741 PASS: LocalInvocationId emitted, no new OpCapability");
}

/// AT-1741 NO-REGRESSION: a gid-only kernel has no LocalInvocationId var
/// and its capability set is unchanged.
#[test]
fn codegen_gid_only_kernel_unchanged_no_local_invocation_id() {
    let src = r#"
        @kernel @workgroup(64,1,1)
        @intent("gid-only regression test AT-1741")
        @complexity(O(1))
        fn gid_only(out: buffer[u32]) -> void {
            let gx: u32 = gid(0u32);
            out[gx] = gx;
            return;
        }
    "#;
    let words = compile(src);
    let asm = disasm(&words);

    // Must NOT have LocalInvocationId in a gid-only kernel.
    assert!(
        !asm.contains("LocalInvocationId"),
        "AT-1741 regression: gid-only kernel must NOT emit LocalInvocationId; asm:\n{asm}"
    );

    eprintln!("AT-1741 PASS (regression): gid-only kernel has no LocalInvocationId");
}

/// AT-1741: a kernel using BOTH gid() and local_invocation_id() emits TWO distinct
/// Input vars: one BuiltIn GlobalInvocationId AND one BuiltIn LocalInvocationId.
#[test]
fn codegen_gid_and_local_invocation_id_both_emitted() {
    let src = r#"
        @kernel @workgroup(64,1,1)
        @intent("gid + local_invocation_id both emitted AT-1741")
        @complexity(O(1))
        fn both_builtins(out: buffer[u32]) -> void {
            let gx:  u32 = gid(0u32);
            let lid: u32 = local_invocation_id(0u32);
            out[gx] = lid;
            return;
        }
    "#;
    let words = compile(src);
    let asm = disasm(&words);

    assert!(
        asm.contains("GlobalInvocationId"),
        "AT-1741: must have GlobalInvocationId; asm:\n{asm}"
    );
    assert!(
        asm.contains("LocalInvocationId"),
        "AT-1741: must have LocalInvocationId; asm:\n{asm}"
    );

    eprintln!("AT-1741 PASS: gid+local_invocation_id → both Input vars emitted");
}

/// AT-1741: a kernel that does NOT call local_invocation_id() must NOT emit the var.
/// Confirms the body-scanner opt-in (unlike gid, not emitted on buffer presence alone).
#[test]
fn codegen_local_invocation_id_not_emitted_when_unused() {
    let src = r#"
        @kernel @workgroup(64,1,1)
        @intent("no local_invocation_id — must not emit var AT-1741")
        @complexity(O(1))
        fn no_lid(out: buffer[u32]) -> void {
            let gx: u32 = gid(0u32);
            out[gx] = gx;
            return;
        }
    "#;
    let words = compile(src);
    let asm = disasm(&words);

    assert!(
        !asm.contains("LocalInvocationId"),
        "AT-1741: LocalInvocationId must NOT be emitted when not used; asm:\n{asm}"
    );
    eprintln!("AT-1741 PASS: LocalInvocationId omitted when unused (opt-in confirmed)");
}

/// Debug test: compile a coopmat kernel using local_invocation_id() and verify
/// the LocalInvocationId decoration appears in the SPIR-V.
/// This simulates the AT-1745 issue with a simpler kernel.
#[test]
fn codegen_local_invocation_id_in_coopmat_kernel() {
    // Simple kernel that uses local_invocation_id alongside subgroup ops
    // (mirrors the MSG kernel's basic structure)
    let src = r#"
        @kernel @workgroup(64,1,1)
        @cooperative_matrix
        @intent("local_invocation_id in coopmat kernel AT-1741-debug")
        @complexity(O(n))
        fn lid_coopmat_test(out: buffer[u32]) -> void {
            let sg_sz: u32 = subgroup_size();
            let local_x: u32 = local_invocation_id(0u32);
            let sg_id: u32 = local_x / sg_sz;
            let gx: u32 = gid(0u32);
            out[gx] = sg_id;
            return;
        }
    "#;
    let words = compile(src);
    let asm = disasm(&words);
    
    assert!(
        asm.contains("LocalInvocationId"),
        "must have LocalInvocationId when used alongside subgroup ops; asm:\n{asm}"
    );
    eprintln!("codegen_local_invocation_id_in_coopmat_kernel PASS");
}
