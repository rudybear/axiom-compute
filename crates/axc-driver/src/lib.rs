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

pub use cli::{Command, Cli};

use std::path::Path;
use axc_lexer::{tokenize, Span};
use axc_parser::{Parser, ParseError};
use axc_hir::{lower_module, HirError};
use axc_codegen::{emit_module_bytes, CodegenOptions};

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
}

impl DriverError {
    /// Returns `true` if the error originated in a pipeline phase (lex/parse/hir).
    pub fn has_phase_errors(&self) -> bool {
        matches!(self, DriverError::Compile { .. })
    }
}

/// Full pipeline: source string → SPIR-V bytes.
///
/// Step 0: BOM pre-check (reject immediately if present).
/// Steps 1-3: Lex, parse, HIR all run to completion even on errors.
/// Step 4: Codegen runs only if all three error lists are empty.
pub fn compile_source_to_spirv(source: &str) -> Result<Vec<u8>, DriverError> {
    // ── Step 0: BOM rejection (NEW rev 2 / anti-pattern #1) ─────────────────
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
    Ok(bytes)
}

/// Full pipeline: read source from `input`, write SPIR-V to `output`.
pub fn compile_file(input: &Path, output: &Path) -> Result<(), DriverError> {
    let source: String = std::fs::read_to_string(input)?;
    let bytes: Vec<u8> = compile_source_to_spirv(&source)?;
    std::fs::write(output, &bytes)?;
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
}
