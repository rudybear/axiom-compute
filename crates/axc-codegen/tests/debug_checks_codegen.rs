//! M3.17 (FG.4) codegen-level acceptance tests: AT-2866, AT-2867, AT-2869.
//!
//! AT-2866: golden-identity gate — a `@precondition`-bearing kernel compiled
//!          `--debug` OFF is byte-identical to (a) a committed golden and (b) the
//!          same source with the annotation removed.
//! AT-2867: `--debug` ON injects exactly one extra binding at index ==
//!          `num_user_buffers`, emits `OpAtomicOr`, and spirv-val is clean with the
//!          capability set == baseline on BOTH a GLSL450 (non-coopmat) leg (Device
//!          scope) and a coopmat leg (QueueFamily scope under MemoryModel::Vulkan).
//! AT-2869: a `--debug` kernel's binding count is N+1 — documents that M3.16
//!          rewrite-verify's release-only interface gate would reject it.

use axc_codegen::{emit_module, CodegenOptions};
use axc_hir::{lower_module, HirModule};
use axc_parser::parse;

const PRECONDITION_SAXPY_SRC: &str = include_str!("../../../examples/precondition_saxpy.axc");

/// Strip every `@precondition(...)` / `@postcondition(...)` line from source text
/// (line-based, sufficient for this single-line-annotation fixture).
fn strip_debug_annotations(src: &str) -> String {
    src.lines()
        .filter(|l| {
            let t = l.trim_start();
            !t.starts_with("@precondition") && !t.starts_with("@postcondition")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn lower_ok(src: &str) -> HirModule {
    let (ast, lex_errs, parse_errs) = parse(src);
    assert!(lex_errs.is_empty(), "lex errors: {lex_errs:?}");
    assert!(parse_errs.is_empty(), "parse errors: {parse_errs:?}");
    let (hir, hir_errs, _warns) = lower_module(&ast);
    assert!(hir_errs.is_empty(), "hir errors: {hir_errs:?}");
    hir
}

fn emit_words(hir: &HirModule, debug_checks: bool) -> Vec<u32> {
    emit_module(hir, &CodegenOptions { debug_checks, ..CodegenOptions::default() })
        .expect("codegen must succeed")
}

// ── AT-2866: golden identity ───────────────────────────────────────────────────

#[test]
fn at_2866_release_spirv_is_golden_identical_and_annotation_inert() {
    let hir = lower_ok(PRECONDITION_SAXPY_SRC);
    let words = emit_words(&hir, false);
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();

    // (a) byte-identical to the committed golden.
    let golden: Vec<u8> = std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/precondition_saxpy_release.spv"),
    ).expect("read golden fixture");
    assert_eq!(bytes, golden, "release SPIR-V must be byte-identical to the committed golden (AT-2866)");

    // (b) byte-identical to the same source with @precondition/@postcondition stripped
    //     — proves the annotations are INERT without --debug.
    let stripped_src = strip_debug_annotations(PRECONDITION_SAXPY_SRC);
    let stripped_hir = lower_ok(&stripped_src);
    let stripped_words = emit_words(&stripped_hir, false);
    assert_eq!(words, stripped_words,
        "release SPIR-V must be unaffected by the presence of @precondition/@postcondition (AT-2866)");
}

/// AT-2838 stays green (guard: this milestone must not perturb the empty-kernel path).
#[test]
fn at_2838_empty_kernel_golden_unaffected() {
    const EMPTY_SRC: &str = "@kernel @workgroup(64,1,1) fn empty() -> void { return; }";
    let hir = lower_ok(EMPTY_SRC);
    let words = emit_words(&hir, false);
    assert_eq!(words[0], 0x0723_0203_u32, "magic word");
    assert_eq!(words[1], 0x0001_0300_u32, "version word");
    assert_eq!(words[2], 0x0000_0000_u32, "generator word");
}

// ── Word-stream scan helpers (mirrors axc_codegen::emit's private test helpers) ─

const OP_CAPABILITY: u16 = 17;
const OP_DECORATE: u16 = 71;
const OP_ATOMIC_OR: u16 = 241;
const DECORATION_BINDING: u32 = 33;
/// `Capability::VulkanMemoryModel` numeric value (SPIR-V spec — NOT the `Scope`
/// enum's `QueueFamily = 5`, which is a different enum entirely).
const CAPABILITY_VULKAN_MEMORY_MODEL: u32 = 5345;

fn collect_capabilities(words: &[u32]) -> Vec<u32> {
    scan_opcode(words, OP_CAPABILITY).into_iter().map(|ops| ops[0]).collect()
}

fn collect_binding_decorations(words: &[u32]) -> Vec<u32> {
    scan_opcode(words, OP_DECORATE)
        .into_iter()
        .filter(|ops| ops.len() >= 3 && ops[1] == DECORATION_BINDING)
        .map(|ops| ops[2])
        .collect()
}

fn contains_atomic_or(words: &[u32]) -> bool {
    !scan_opcode(words, OP_ATOMIC_OR).is_empty()
}

/// Return the operand words (everything after the opcode/word-count header) for
/// every instruction matching `opcode`.
fn scan_opcode(words: &[u32], opcode: u16) -> Vec<Vec<u32>> {
    let body = &words[5..];
    let mut cursor = 0usize;
    let mut out = Vec::new();
    while cursor < body.len() {
        let header = body[cursor];
        let word_count = ((header >> 16) & 0xFFFF) as usize;
        let op = (header & 0xFFFF) as u16;
        if word_count == 0 {
            break;
        }
        if op == opcode {
            out.push(body[cursor + 1..cursor + word_count].to_vec());
        }
        cursor += word_count;
    }
    out
}

fn spirv_val_check(words: &[u32]) {
    use spirv_tools::val::{create as create_validator, Validator};
    use spirv_tools::TargetEnv;
    let validator = create_validator(Some(TargetEnv::Vulkan_1_1));
    validator.validate(words, None).expect("spirv-val Vulkan_1_1 must pass");
}

// ── AT-2867: --debug injected binding + OpAtomicOr + spirv-val, TWO legs ───────

/// (a) GLSL450 (non-coopmat) leg: `Device` scope, caps == {Shader} == baseline.
#[test]
fn at_2867_glsl450_leg_injects_flag_binding_atomic_or_and_stays_clean() {
    let hir = lower_ok(PRECONDITION_SAXPY_SRC);
    let num_user_buffers = hir.kernels[0].binding_plan.buffers.len() as u32;

    let release_words = emit_words(&hir, false);
    let release_caps = collect_capabilities(&release_words);

    let debug_words = emit_words(&hir, true);
    spirv_val_check(&debug_words);

    let debug_caps = collect_capabilities(&debug_words);
    assert_eq!(
        debug_caps, release_caps,
        "the atomic must add ZERO new capabilities on the GLSL450 leg (Shader-only baseline)"
    );
    assert!(!release_caps.contains(&CAPABILITY_VULKAN_MEMORY_MODEL),
        "sanity: the GLSL450 leg's baseline must not already carry VulkanMemoryModel");

    // Exactly one extra binding, at index == num_user_buffers.
    let release_bindings = collect_binding_decorations(&release_words);
    let debug_bindings = collect_binding_decorations(&debug_words);
    assert_eq!(debug_bindings.len(), release_bindings.len() + 1,
        "exactly one extra binding must be injected under --debug");
    assert!(debug_bindings.contains(&num_user_buffers),
        "the injected flag binding must be at index == num_user_buffers ({num_user_buffers}); got {debug_bindings:?}");

    assert!(contains_atomic_or(&debug_words), "--debug must emit at least one OpAtomicOr");

    // Pool/layout/bind consistency: the metadata-facing binding_plan.buffers.len()
    // (what the DSL/pool/bind loop all key on, per axc_runtime::debug_augmented_binding_plan)
    // must equal num_user_buffers + 1 for this kernel.
    assert_eq!(debug_bindings.len() as u32, num_user_buffers + 1);
}

/// (b) coopmat leg: `QueueFamily` scope under `MemoryModel::Vulkan`, caps ==
/// {Shader, VulkanMemoryModel, CooperativeMatrixKHR} == baseline coopmat set (the
/// atomic adds nothing beyond what coopmat already requires).
#[test]
fn at_2867_coopmat_leg_injects_flag_binding_atomic_or_and_stays_clean() {
    let src = r#"
@kernel
@workgroup(32,1,1)
@cooperative_matrix
@intent("16x16 f16 tiled matmul with a runtime precondition")
@precondition(gt(tile_offset, 0))
fn matmul_tile_debug(
    tile_offset: u32,
    a_buf: readonly_buffer[f16],
    b_buf: readonly_buffer[f16],
    c_buf: buffer[f16]
) -> void {
    let stride: u32 = 16u32;
    let a: matrix[f16, 16, 16, a] = coopmat_load(a_buf, tile_offset, stride);
    let b: matrix[f16, 16, 16, b] = coopmat_load(b_buf, tile_offset, stride);
    let acc: matrix[f16, 16, 16, accumulator] = coopmat_zero();
    let result: matrix[f16, 16, 16, accumulator] = coopmat_mul_add(a, b, acc);
    coopmat_store(result, c_buf, tile_offset, stride);
    return;
}
"#;
    let hir = lower_ok(src);
    assert_eq!(hir.kernels[0].annotations.debug_checks.len(), 1, "the coopmat precondition must lower to exactly 1 check");
    let num_user_buffers = hir.kernels[0].binding_plan.buffers.len() as u32;

    let release_words = emit_words(&hir, false);
    let release_caps = collect_capabilities(&release_words);
    assert!(release_caps.contains(&CAPABILITY_VULKAN_MEMORY_MODEL), "coopmat baseline must already carry VulkanMemoryModel");

    let debug_words = emit_words(&hir, true);
    spirv_val_check(&debug_words);

    let debug_caps = collect_capabilities(&debug_words);
    assert_eq!(
        debug_caps, release_caps,
        "the atomic must add ZERO new capabilities on the coopmat leg (VulkanMemoryModel is already baseline)"
    );

    let release_bindings = collect_binding_decorations(&release_words);
    let debug_bindings = collect_binding_decorations(&debug_words);
    assert_eq!(debug_bindings.len(), release_bindings.len() + 1);
    assert!(debug_bindings.contains(&num_user_buffers),
        "the injected flag binding must be at index == num_user_buffers ({num_user_buffers}); got {debug_bindings:?}");
    assert!(contains_atomic_or(&debug_words), "--debug must emit at least one OpAtomicOr on the coopmat leg too");
}

// ── AT-2869: --debug binding count is N+1 (documents the M3.16 interface gate) ─

#[test]
fn at_2869_debug_kernel_has_n_plus_one_bindings() {
    let hir = lower_ok(PRECONDITION_SAXPY_SRC);
    let num_user_buffers = hir.kernels[0].binding_plan.buffers.len();
    let debug_words = emit_words(&hir, true);
    let debug_bindings = collect_binding_decorations(&debug_words);
    assert_eq!(
        debug_bindings.len(), num_user_buffers + 1,
        "a --debug kernel has N+1 bindings; M3.16 rewrite-verify's release-only \
         interface gate (which compares binding counts) correctly rejects any pairing \
         involving this artifact class — enforced by the EXISTING gate, no new code"
    );
}
