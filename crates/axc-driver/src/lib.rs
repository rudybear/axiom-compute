//! Library form of the AXIOM-Compute driver.
//!
//! # BOM pre-check
//! `compile_source_to_spirv` rejects UTF-8 BOM-prefixed files before any other
//! processing. BOMs drift spans by 3 bytes, breaking diagnostic rendering and
//! downstream tooling (anti-pattern #1 — no implicit behavior).
//!
//! # Collect-all error aggregation (anti-pattern #6)
//! All three pipeline phases (lex, parse, HIR) always run to completion.
//! Errors from every phase are aggregated into `DriverError::Compile { lex, parse, hir }`.
//! Codegen runs ONLY when all three error lists are empty.

pub mod cli;
pub mod optimize;

pub use cli::{Command, Cli};

use std::path::{Path, PathBuf};
use axc_lexer::{tokenize, Span};
use axc_parser::{Parser, ParseError};
use axc_hir::{lower_module, HirError};
use axc_codegen::{emit_module_bytes, CodegenOptions, extract_workgroup_dims};
use axc_runtime::KernelMetadata;

/// Errors that can occur during compilation.
#[derive(Debug, thiserror::Error, miette::Diagnostic)]
pub enum DriverError {
    /// One or more compilation phases produced errors.
    /// All phases ran to completion; this aggregates their outputs.
    #[error("compilation failed: {lex} lex errors, {parse} parse errors, {hir} hir errors",
        lex = lex.len(), parse = parse.len(), hir = hir.len())]
    Compile {
        lex: Vec<axc_lexer::LexError>,
        parse: Vec<ParseError>,
        hir: Vec<HirError>,
    },
    /// The source file begins with a UTF-8 byte-order mark (U+FEFF, bytes EF BB BF).
    ///
    /// axc rejects BOMs rather than silently stripping them, because stripping
    /// would drift all downstream spans by 3 bytes — breaking miette diagnostics
    /// and any tooling that maps span offsets back to raw file positions.
    #[error("file begins with a UTF-8 byte-order mark (U+FEFF); re-save the file without a BOM or strip it explicitly before passing to axc. axc rejects BOMs to keep source spans raw-file-aligned.")]
    UnexpectedByteOrderMark {
        #[label("BOM occupies these 3 bytes")]
        span: Span,
    },
    #[error("codegen error: {0}")]
    Codegen(#[from] axc_codegen::CodegenError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// Metadata sidecar emission failed.
    #[error("metadata emit failed: {0}")]
    MetadataEmitFailed(axc_runtime::DispatchError),
}

impl DriverError {
    /// Returns `true` if the error originated in a pipeline phase (lex/parse/hir).
    pub fn has_phase_errors(&self) -> bool {
        matches!(self, DriverError::Compile { .. })
    }
}

/// Full pipeline: source string → (SPIR-V bytes, KernelMetadata).
///
/// This is the primary M1.5 compilation API. It returns both the compiled
/// SPIR-V bytes and a `KernelMetadata` describing the kernel's binding plan,
/// workgroup size, and entry-point name.
///
/// Step 0: BOM pre-check (reject immediately if present).
/// Steps 1-3: Lex, parse, HIR all run to completion even on errors.
/// Step 4: Codegen runs only if all three error lists are empty.
/// Step 5: Build KernelMetadata from HIR module.
pub fn compile_source_with_meta(source: &str) -> Result<(Vec<u8>, KernelMetadata), DriverError> {
    // ── Step 0: BOM rejection (anti-pattern #1) ─────────────────────────────
    // UTF-8 BOM is EF BB BF. We check raw bytes, not the char value, to keep
    // the logic portable across platforms and consistent with the span semantics
    // (spans are byte offsets into the raw source, not char offsets).
    if source.as_bytes().starts_with(&[0xEF, 0xBB, 0xBF]) {
        return Err(DriverError::UnexpectedByteOrderMark {
            span: Span { start: 0, end: 3 },
        });
    }

    // ── Phase 1: Lex (always runs to EOF) ────────────────────────────────────
    let (tokens, lex_errors) = tokenize(source);

    // ── Phase 2: Parse (always runs to EOF, skips Error tokens silently) ─────
    let mut parser: Parser = Parser::new(&tokens);
    let (ast, parse_errors) = parser.parse_module();

    // ── Phase 3: HIR (always runs to completion) ──────────────────────────────
    let (hir, hir_errors, hir_warnings) = lower_module(&ast);

    // Print warnings (non-blocking)
    for w in &hir_warnings {
        eprintln!("warning: {:?}", w);
    }

    // If any phase produced errors, aggregate and return early (before codegen)
    if !lex_errors.is_empty() || !parse_errors.is_empty() || !hir_errors.is_empty() {
        return Err(DriverError::Compile {
            lex: lex_errors,
            parse: parse_errors,
            hir: hir_errors,
        });
    }

    // ── Phase 4: Codegen (only when all phases are clean) ────────────────────
    let bytes: Vec<u8> = emit_module_bytes(&hir, &CodegenOptions::default())?;

    // ── Phase 5: Build metadata from HIR ─────────────────────────────────────
    let workgroup_size: [u32; 3] = extract_workgroup_dims(&hir);
    let kernel = hir.kernels.first()
        .expect("codegen succeeded, so at least one kernel exists");
    // The SPIR-V emitter writes `OpEntryPoint GLCompute %main "<kernel.name>"`
    // (see axc-codegen/src/emit.rs — `b.entry_point(..., &kernel.name, ...)`),
    // so the runtime must pass `kernel.name` as `pName` to vkCreateComputePipelines
    // or Lavapipe/validation layers raise
    // `VUID-VkPipelineShaderStageCreateInfo-pName-00707` (pName `main` entrypoint
    // not found). Keep this consistent with codegen rather than hardcoding "main".
    let metadata: KernelMetadata = KernelMetadata::new(
        kernel.name.clone(),
        workgroup_size,
        kernel.binding_plan.clone(),
        kernel.name.clone(),
    );

    Ok((bytes, metadata))
}

/// Full pipeline: source string → SPIR-V bytes.
///
/// Preserved verbatim from M1.4 for backward compatibility with all existing callers.
/// Internally forwards to `compile_source_with_meta` and discards the metadata.
///
/// Step 0: BOM pre-check (reject immediately if present).
/// Steps 1-3: Lex, parse, HIR all run to completion even on errors.
/// Step 4: Codegen runs only if all three error lists are empty.
pub fn compile_source_to_spirv(source: &str) -> Result<Vec<u8>, DriverError> {
    let (bytes, _meta) = compile_source_with_meta(source)?;
    Ok(bytes)
}

/// M2.3: Full pipeline with strategy hole pre-resolution.
///
/// Like `compile_source_with_meta` but accepts explicit hole assignments that
/// substitute named `?hole` references with concrete integer values before
/// HIR construction.
///
/// This is the per-variant compilation path called by `axc optimize` (and
/// `axc compile --strategy-value`).  Each call compiles one strategy variant.
///
/// `assignments`: map of hole name → concrete value.  Holes not listed here
/// retain whatever default the source provides (first candidate).
///
/// Returns `DriverError::Compile` if any pipeline phase fails after substitution.
pub fn compile_source_with_assignments(
    source: &str,
    assignments: &std::collections::BTreeMap<String, i64>,
) -> Result<(Vec<u8>, KernelMetadata), DriverError> {
    // Strategy substitution: rewrite `?hole` references in the source text to
    // their concrete integer values before lexing.  This is source-text
    // substitution (NOT AST rewrite) for byte-identical SPIR-V reproducibility.
    //
    // The substituted source is fed to the normal pipeline.
    let substituted: String = substitute_strategy_holes(source, assignments);
    compile_source_with_meta(&substituted)
}

/// M2.3: Substitute strategy hole references in source text and strip @strategy block.
///
/// Performs two passes:
///
/// 1. For each `(name, value)` in `assignments`, replaces all occurrences of
///    `?name` in the source with the decimal representation of `value`.
///    (Alphabetical BTreeMap order for determinism.)
///
/// 2. Strips the entire `@strategy { ... }` annotation block from the result.
///    This is necessary because after hole-ref substitution, the @strategy block
///    would still contain `?[...]` candidate lists that HIR would lower to
///    StrategyHoles — triggering the codegen UnresolvedStrategyHole backstop.
///    Stripping it signals that the holes are fully resolved.
///
/// The strip is simple-text-level: finds the first `@strategy` followed by
/// whitespace and `{`, then removes through the matching `}`.  This handles
/// the common single-level case.  Nested braces are not supported (M2.3 scope).
pub(crate) fn substitute_strategy_holes(
    source: &str,
    assignments: &std::collections::BTreeMap<String, i64>,
) -> String {
    // Pass 1: substitute ?name references.
    let mut result: String = source.to_string();
    for (name, value) in assignments {
        let hole_ref: String = format!("?{name}");
        let replacement: String = value.to_string();
        result = result.replace(&hole_ref, &replacement);
    }

    // Pass 2: strip @strategy { ... } block.
    // After Pass 1 the hole refs in @workgroup are integers, but @strategy still
    // has ?[...] candidate lists. Removing the block prevents HIR from lowering
    // the residual candidates into StrategyHoles (which would trip the backstop).
    result = strip_strategy_annotation_block(&result);

    result
}

/// Strip the first `@strategy { ... }` annotation block from source text.
///
/// Finds `@strategy` followed by optional whitespace and `{`, then removes
/// the entire span including the closing `}`.  Content between the braces is
/// discarded.  Only single-level brace nesting is handled (sufficient for M2.3).
///
/// If no `@strategy {` is found, the source is returned unchanged.
pub(crate) fn strip_strategy_annotation_block(source: &str) -> String {
    // Find `@strategy` in the text.
    let Some(at_pos) = source.find("@strategy") else {
        return source.to_string();
    };

    // After `@strategy`, skip whitespace and look for `{`.
    let after_keyword: &str = &source[at_pos + "@strategy".len()..];
    let trimmed: &str = after_keyword.trim_start_matches([' ', '\t', '\r', '\n']);
    if !trimmed.starts_with('{') {
        // Not the curly-brace form; leave unchanged (e.g. @strategy(...) call form).
        return source.to_string();
    }

    // Find the opening brace position in the original source.
    let open_brace_offset: usize = at_pos
        + "@strategy".len()
        + (after_keyword.len() - trimmed.len());

    // Scan forward to find the matching closing brace (depth tracking).
    let bytes: &[u8] = source.as_bytes();
    let mut depth: usize = 0;
    let mut close_pos: Option<usize> = None;
    for (i, &byte) in bytes.iter().enumerate().skip(open_brace_offset) {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    close_pos = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }

    let close_pos: usize = match close_pos {
        Some(p) => p,
        None => return source.to_string(), // Unmatched brace — leave unchanged.
    };

    // Build the stripped source: everything before `@strategy` + everything after `}`.
    let before: &str = &source[..at_pos];
    let after: &str = &source[close_pos + 1..];
    format!("{before}{after}")
}

/// Compute the metadata sidecar path from an output path.
///
/// Appends `.axc.meta.json` to the full filename (including any extension).
/// Examples:
/// - `out.spv` → `out.spv.axc.meta.json`
/// - `kernel` → `kernel.axc.meta.json`
/// - `/a/b.c/d.spv` → `/a/b.c/d.spv.axc.meta.json`
pub(crate) fn metadata_sidecar_path(output: &Path) -> PathBuf {
    let filename: String = output
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".to_owned());
    let sidecar_name: String = format!("{filename}.axc.meta.json");
    output
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(sidecar_name)
}

/// Full pipeline: read source from `input`, write SPIR-V to `output` and metadata sidecar.
///
/// Writes two files:
/// - `output`: SPIR-V binary
/// - `output.axc.meta.json`: JSON metadata sidecar
pub fn compile_file(input: &Path, output: &Path) -> Result<(), DriverError> {
    let source: String = std::fs::read_to_string(input)?;
    let (bytes, metadata) = compile_source_with_meta(&source)?;
    std::fs::write(output, &bytes)?;
    let sidecar_path: PathBuf = metadata_sidecar_path(output);
    metadata.save(&sidecar_path)
        .map_err(DriverError::MetadataEmitFailed)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EMPTY_KERNEL_SRC: &str = concat!(
        "@kernel\n",
        "@workgroup(64, 1, 1)\n",
        "@intent(\"smoke-test: smallest valid kernel for SPIR-V emission\")\n",
        "@complexity(O(1))\n",
        "@precondition(true)\n",
        "fn empty() -> void {\n",
        "    return;\n",
        "}\n",
    );

    // ── Happy path ────────────────────────────────────────────────────────────

    #[test]
    fn compile_source_happy_path() {
        let bytes = compile_source_to_spirv(EMPTY_KERNEL_SRC)
            .expect("compile should succeed for valid kernel");
        assert!(!bytes.is_empty(), "output bytes should not be empty");
        // First 4 bytes are SPIR-V magic (LE)
        assert_eq!(&bytes[0..4], &[0x03, 0x02, 0x23, 0x07], "SPIR-V magic mismatch");
    }

    // ── AT-14: BOM rejection ─────────────────────────────────────────────────

    #[test]
    fn bom_prefix_is_rejected() {
        // Source with leading UTF-8 BOM (EF BB BF) followed by a valid kernel
        let bom_src: String = format!("\u{FEFF}{}", EMPTY_KERNEL_SRC);
        let result = compile_source_to_spirv(&bom_src);
        match result {
            Err(DriverError::UnexpectedByteOrderMark { span }) => {
                assert_eq!(span.start, 0);
                assert_eq!(span.end, 3);
            }
            other => panic!("expected UnexpectedByteOrderMark, got: {other:?}"),
        }

        // Verify the non-BOM variant compiles cleanly (BOM is the only reason to fail)
        let result_ok = compile_source_to_spirv(EMPTY_KERNEL_SRC);
        assert!(result_ok.is_ok(), "non-BOM source should compile: {result_ok:?}");
    }

    // ── AT-8: multi-phase error aggregation ───────────────────────────────────

    #[test]
    fn multi_phase_error_report() {
        // Source: emoji mid-token (lex error) + struct at top level (parse error)
        let src = "💥 struct Foo {}";
        let result = compile_source_to_spirv(src);
        match result {
            Err(DriverError::Compile { lex, parse, hir }) => {
                assert!(!lex.is_empty() && !parse.is_empty(),
                    "expected BOTH lex AND parse errors (collect-all aggregation); lex={lex:?}, parse={parse:?}, hir={hir:?}");
            }
            other => panic!("expected DriverError::Compile, got: {other:?}"),
        }
    }

    // ── Lex-error-only surfaces correctly ─────────────────────────────────────

    #[test]
    fn lex_error_only_surfacing() {
        // An unterminated string with no other content; should produce lex error(s)
        let src = r#"@kernel @workgroup(1,1,1) fn k() -> void { return; } "unterminated"#;
        let result = compile_source_to_spirv(src);
        match result {
            Err(DriverError::Compile { lex, .. }) => {
                assert!(!lex.is_empty(), "expected lex errors: {lex:?}");
            }
            Ok(_) => panic!("expected error for unterminated string"),
            other => panic!("unexpected result: {other:?}"),
        }
    }

    // ── has_phase_errors helper ───────────────────────────────────────────────

    #[test]
    fn has_phase_errors_returns_true_for_compile_variant() {
        let e = DriverError::Compile { lex: Vec::new(), parse: Vec::new(), hir: Vec::new() };
        assert!(e.has_phase_errors());
    }

    // ── AT-124: DESIGN.md documents integer division UB ──────────────────────
    //
    // Spec (CRITICAL-2 fix): DESIGN.md must contain a paragraph documenting that
    // OpSDiv / OpSRem with INT_MIN / -1 is UNDEFINED BEHAVIOR per SPIR-V §3.32.14,
    // and that AXIOM-Compute does NOT emit runtime checks for this case.
    //
    // This test triple-checks the presence of the required text to guard against
    // accidental deletion during future DESIGN.md edits.
    #[test]
    fn design_md_documents_int_division_ub() {
        // Locate DESIGN.md relative to the workspace root.
        // CARGO_MANIFEST_DIR for axc-driver is <repo>/crates/axc-driver.
        // Go up two levels to reach the repo root.
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .expect("CARGO_MANIFEST_DIR not set");
        let design_md_path = std::path::PathBuf::from(&manifest_dir)
            .join("..")
            .join("..")
            .join("DESIGN.md");
        let content = std::fs::read_to_string(&design_md_path)
            .unwrap_or_else(|e| panic!("failed to read DESIGN.md at {:?}: {e}", design_md_path));

        // Triple-check: all three substrings must be present.
        assert!(
            content.contains("INT_MIN / -1"),
            "DESIGN.md must document INT_MIN / -1 as UB; substring 'INT_MIN / -1' not found"
        );
        assert!(
            content.to_uppercase().contains("UNDEFINED BEHAVIOR"),
            "DESIGN.md must use the phrase 'UNDEFINED BEHAVIOR' (case-insensitive); not found"
        );
        assert!(
            content.contains("OpSDiv") || content.contains("OpSRem"),
            "DESIGN.md must mention OpSDiv or OpSRem in the UB section; neither found"
        );
    }

    // ── AT-521: metadata_sidecar_path helper ─────────────────────────────────

    #[test]
    fn at_521_metadata_sidecar_path_appends_suffix() {
        use super::metadata_sidecar_path;
        use std::path::Path;

        // .spv extension → appends .axc.meta.json after the .spv
        let p = metadata_sidecar_path(Path::new("out.spv"));
        assert_eq!(p, std::path::PathBuf::from("out.spv.axc.meta.json"));

        // No extension
        let p = metadata_sidecar_path(Path::new("kernel"));
        assert_eq!(p, std::path::PathBuf::from("kernel.axc.meta.json"));

        // Dots in directory component should not affect result
        let p = metadata_sidecar_path(Path::new("/a/b.c/d.spv"));
        assert_eq!(p, std::path::PathBuf::from("/a/b.c/d.spv.axc.meta.json"));
    }

    // ── AT-510a: compile_source_with_meta returns bytes + metadata ────────────

    #[test]
    fn at_510a_compile_source_with_meta_returns_bytes_and_metadata() {
        let src = concat!(
            "@kernel\n",
            "@workgroup(64, 1, 1)\n",
            "@intent(\"test\")\n",
            "@complexity(O(n))\n",
            "fn saxpy(n: u32, alpha: f32, x: readonly_buffer[f32], y: buffer[f32]) -> void {\n",
            "    let i: u32 = gid(0);\n",
            "    return;\n",
            "}\n",
        );

        let (bytes, meta) = compile_source_with_meta(src)
            .expect("compile should succeed");

        // SPIR-V magic bytes
        assert_eq!(&bytes[0..4], &[0x03, 0x02, 0x23, 0x07], "SPIR-V magic mismatch");

        // Metadata fields
        assert_eq!(meta.kernel_name, "saxpy");
        assert_eq!(meta.workgroup_size, [64, 1, 1]);
        // Entry point name equals kernel name (matches OpEntryPoint emitted by
        // axc-codegen; avoids VUID-VkPipelineShaderStageCreateInfo-pName-00707).
        assert_eq!(meta.entry_point, "saxpy");
        assert_eq!(meta.schema_version, axc_runtime::CURRENT_SCHEMA_VERSION);

        // Binding plan: saxpy has 2 buffers + 2 scalars
        assert_eq!(meta.binding_plan.buffers.len(), 2, "saxpy should have 2 buffer bindings");
        assert_eq!(meta.binding_plan.scalars.len(), 2, "saxpy should have 2 scalar push constants");
        assert_eq!(meta.push_constant_total_bytes, 8, "n:u32 + alpha:f32 = 8 bytes");

        // Buffer names and positions
        assert_eq!(meta.binding_plan.buffers[0].name, "x");
        assert_eq!(meta.binding_plan.buffers[0].buffer_position, 0);
        assert_eq!(meta.binding_plan.buffers[1].name, "y");
        assert_eq!(meta.binding_plan.buffers[1].buffer_position, 1);

        // Scalar offsets per AT-514a discipline
        assert_eq!(meta.binding_plan.scalars[0].name, "n");
        assert_eq!(meta.binding_plan.scalars[0].offset, 0);
        assert_eq!(meta.binding_plan.scalars[1].name, "alpha");
        assert_eq!(meta.binding_plan.scalars[1].offset, 4);
    }

    // ── AT-116: tri-phase error aggregation for scalar code ───────────────────
    //
    // Spec requires: feed a source with lex errors AND parse errors AND HIR errors;
    // assert DriverError::Compile { lex, parse, hir } has ALL THREE non-empty.
    //
    // Construction:
    //   - Lex error:   emoji 💥 in the token stream (axc-lexer rejects non-ASCII
    //                  codepoints that are not inside string literals).
    //   - Parse error: `struct Foo {}` at top level (axc-parser does not parse
    //                  struct declarations; produces a parse error on `struct`).
    //   - HIR error:   a valid-looking kernel that has a type mismatch.
    //                  Because lex/parse errors propagate partial ASTs, we need the
    //                  HIR error to survive even when there are upstream errors.
    //                  Strategy: include a second kernel-like fragment that parses
    //                  successfully but contains a type mismatch (e.g. assigning a
    //                  float literal to an i32 binding without a suffix).
    //
    // Note: because the collect-all design runs all three phases unconditionally,
    // the lex and parse phases will produce errors from the emoji / struct tokens,
    // while the HIR phase sees the partially-parsed AST.  To guarantee a HIR error
    // independent of lex/parse, we construct a source where the bad fragment that
    // causes a lex error is isolated from a separately valid (but type-mismatched)
    // kernel that will be parsed and HIR-checked.
    //
    // After careful analysis: the three phases run on the SAME source string.
    // - Phase 1 tokenizes everything and records lex errors for bad tokens.
    // - Phase 2 parses the full (possibly error-containing) token stream.
    // - Phase 3 runs HIR on whatever partial AST Phase 2 produced.
    //
    // Therefore the simplest approach that reliably hits all three:
    //   src = "<emoji> <struct-decl> @kernel fn k() -> void { let x: i32 = 1.0f32; return; }"
    //
    // - emoji → LexError (lex phase rejects non-ASCII outside string literals)
    // - struct → ParseError (top-level struct is not in the grammar)
    // - let x: i32 = 1.0f32 → HirError::TypeMismatch (f32 literal to i32 binding)
    //
    // The kernel `fn k() -> void` has no `@kernel` / `@workgroup` annotation
    // so the HIR will also produce a validation error.  Any HirError makes
    // hir.len() > 0, which is the assertion we need.
    #[test]
    fn driver_scalar_demo_aggregates_errors() {
        // 💥 triggers a LexError (non-ASCII codepoint outside a string literal).
        // `struct Foo {}` triggers a ParseError (not a valid top-level item in axc grammar).
        // `let x: i32 = 1.0f32;` inside a kernel triggers a HirError (f32 ≠ i32).
        let src = concat!(
            "💥 struct Foo {}\n",
            "@kernel @workgroup(1,1,1) fn k() -> void {\n",
            "    let x: i32 = 1.0f32;\n",
            "    return;\n",
            "}\n",
        );

        let result = compile_source_to_spirv(src);
        match result {
            Err(DriverError::Compile { lex, parse, hir }) => {
                assert!(
                    !lex.is_empty(),
                    "expected lex errors (emoji in source); got none. lex={lex:?}"
                );
                assert!(
                    !parse.is_empty(),
                    "expected parse errors (struct at top level); got none. parse={parse:?}"
                );
                assert!(
                    !hir.is_empty(),
                    "expected HIR errors (f32 assigned to i32); got none. hir={hir:?}"
                );
            }
            other => panic!(
                "expected DriverError::Compile with all three phases non-empty; got: {other:?}"
            ),
        }
    }

    // ── AT-1043..AT-1045: Strategy hole source-text substitution tests ────────

    /// AT-1043: substitute_strategy_holes replaces ?name with integer value.
    #[test]
    fn at_1043_substitute_strategy_holes_basic() {
        let mut assignments: std::collections::BTreeMap<String, i64> =
            std::collections::BTreeMap::new();
        assignments.insert("wg".to_string(), 64);

        let src = "@workgroup(?wg, 1, 1)";
        let result = substitute_strategy_holes(src, &assignments);
        // After substitution, ?wg → 64; @strategy block stripped (none present here).
        assert_eq!(result, "@workgroup(64, 1, 1)",
            "substitute_strategy_holes must replace ?wg with 64");
    }

    /// AT-1044: strip_strategy_annotation_block removes @strategy { ... } from source.
    #[test]
    fn at_1044_strip_strategy_block_removes_annotation() {
        let src = "@kernel @workgroup(64,1,1) @strategy { x: ?[32, 64] } fn k() -> void { return; }";
        let result = strip_strategy_annotation_block(src);
        assert!(
            !result.contains("@strategy"),
            "strip_strategy_annotation_block must remove @strategy block; got {result:?}"
        );
        assert!(
            result.contains("@kernel"),
            "rest of source must be preserved; got {result:?}"
        );
        assert!(
            result.contains("fn k()"),
            "kernel declaration must be preserved; got {result:?}"
        );
    }

    /// AT-1045: compile_source_with_assignments + strategy strips correctly produce valid SPIR-V.
    #[test]
    fn at_1045_compile_with_assignments_end_to_end() {
        let src = concat!(
            "@kernel @workgroup(?wg_x, 1, 1)\n",
            "@strategy { wg_x: ?[32, 64, 128] }\n",
            "fn tune() -> void { return; }\n",
        );

        let mut assignments: std::collections::BTreeMap<String, i64> =
            std::collections::BTreeMap::new();
        assignments.insert("wg_x".to_string(), 128);

        let (bytes, meta) = compile_source_with_assignments(src, &assignments)
            .expect("compile must succeed with wg_x=128");

        assert_eq!(&bytes[0..4], &[0x03, 0x02, 0x23, 0x07],
            "SPIR-V magic must be correct");
        assert_eq!(meta.workgroup_size, [128, 1, 1],
            "workgroup_size must reflect resolved wg_x=128");
    }
}
