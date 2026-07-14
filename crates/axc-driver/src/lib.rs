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
pub mod mcp;
pub mod q4km_oracle;
pub mod q6k_oracle;
pub mod q5k_oracle;
/// M3.16 (FG.1): `verify_rewrite` — the LLM structural-rewrite verifier core.
pub mod rewrite_verify;
/// M3.18 (FG.8): `axc verify` — static annotation/consistency checker (front-end only).
pub mod verify;
/// M3.18 (FG.8): `axc test --fuzz` — precondition-satisfying input fuzzer with a
/// runtime debug-check oracle.
pub mod fuzz;

pub use cli::{Command, Cli};

use std::path::{Path, PathBuf};
use axc_lexer::{tokenize, Span};
use axc_parser::{Parser, ParseError};
use axc_hir::{lower_module, HirError};
use axc_codegen::{emit_module_bytes, CodegenOptions, extract_workgroup_dims};
use axc_runtime::{KernelMetadata, CoopMatShapeMeta, CoopMatScalarMeta, CoopMatScopeMeta};
use axc_runtime::{DebugChecksMeta, DebugConditionMeta, DebugCheckKind as DebugMetaCheckKind};
use axc_hir::coopmat::{CoopMatrixShape, CoopMatScopeHir};
use axc_hir::ty::ScalarTy;

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
    /// A `ScalarTy` that is not a valid coopmat element type was passed to
    /// `scalar_ty_to_coopmat_scalar_meta` (D, M3.1.5).
    ///
    /// This path is effectively unreachable (gated by `is_allowed_coopmat_element`
    /// in HIR) but returns a typed error rather than panicking (Hard rule: no panic
    /// in library code).
    #[error("scalar type '{ty}' is not a valid cooperative-matrix element type")]
    UnsupportedCoopmatScalar {
        /// Debug representation of the unsupported `ScalarTy`.
        ty: String,
    },
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
///
/// Compiles in RELEASE mode (`--debug` OFF). Forwards to `compile_source_with_meta_debug`.
pub fn compile_source_with_meta(source: &str) -> Result<(Vec<u8>, KernelMetadata), DriverError> {
    compile_source_with_meta_debug(source, false)
}

/// M3.17 (FG.4): full pipeline with an explicit `--debug` flag. Identical to
/// `compile_source_with_meta` except `debug_checks` threads to `CodegenOptions` and
/// the metadata sidecar; `debug_checks == false` is byte-identical to release
/// (golden-identity gate, §6 of the spec).
pub fn compile_source_with_meta_debug(
    source: &str,
    debug_checks: bool,
) -> Result<(Vec<u8>, KernelMetadata), DriverError> {
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
    let bytes: Vec<u8> = emit_module_bytes(&hir, &CodegenOptions { debug_checks, ..CodegenOptions::default() })?;

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
    //
    // M3.1: Fill coopmat shape from HIR (CRITICAL-1). The runtime reads the required
    // shape FROM METADATA — nothing is hardcoded in the runtime or tests (HN-10/AT-1552).
    // M3.1.5 (D): hir_coopmat_shape_to_meta now returns Result to avoid panic.
    let coopmat_meta: Option<CoopMatShapeMeta> = kernel.annotations.coop_matrix.as_ref()
        .map(hir_coopmat_shape_to_meta)
        .transpose()?;

    // M3.2: Compute aggregate shared_memory_bytes from the typed body's shared decls.
    // This is Σ total_byte_size() for all shared[T,N] declarations in the kernel body.
    let shared_memory_bytes: u32 = match &kernel.body {
        axc_hir::KernelBody::Typed(tb) => {
            let total: u64 = tb.shared.iter().map(|s| s.ty.total_byte_size()).sum();
            // Saturate to u32::MAX if somehow overflow occurs (should not in practice
            // since the HIR validator rejects > 65536 bytes at compile time).
            total.min(u64::from(u32::MAX)) as u32
        }
        axc_hir::KernelBody::Empty => 0,
    };

    // M3.17 (FG.4): debug-check metadata, present only under `--debug` (§7/§6 —
    // `skip_serializing_if` on the field keeps a release sidecar unperturbed).
    let debug_checks_meta: Option<DebugChecksMeta> = if debug_checks {
        Some(build_debug_checks_meta(kernel, source))
    } else {
        None
    };

    let metadata: KernelMetadata = KernelMetadata::new(
        kernel.name.clone(),
        workgroup_size,
        kernel.binding_plan.clone(),
        kernel.name.clone(),
    ).with_coopmat(coopmat_meta)
     .with_shared_memory_bytes(shared_memory_bytes)
     .with_debug_checks(debug_checks_meta);

    Ok((bytes, metadata))
}

/// M3.17 (FG.4): build the sidecar's `DebugChecksMeta` from a compiled kernel's
/// lowered `debug_checks` (§7 of the spec — the flag binding is one past the last
/// user buffer; `text`/`line` are for host-side violation messages).
fn build_debug_checks_meta(kernel: &axc_hir::Kernel, source: &str) -> DebugChecksMeta {
    DebugChecksMeta {
        flag_binding: kernel.binding_plan.buffers.len() as u32,
        flag_len_words: 2,
        conditions: kernel.annotations.debug_checks.iter().map(|c| DebugConditionMeta {
            kind: match c.kind {
                axc_hir::DebugCheckKind::Pre => DebugMetaCheckKind::Pre,
                axc_hir::DebugCheckKind::Post => DebugMetaCheckKind::Post,
            },
            bit: c.bit,
            text: c.text.clone(),
            line: source_line_of(source, c.span.start),
        }).collect(),
    }
}

/// 1-based source line number for a byte offset (best-effort; used only for the
/// human-readable `DebugConditionMeta.line`, not for correctness).
fn source_line_of(source: &str, byte_offset: u32) -> u32 {
    let off: usize = (byte_offset as usize).min(source.len());
    1 + source.as_bytes()[..off].iter().filter(|&&b| b == b'\n').count() as u32
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

/// M3.1: Map a HIR `CoopMatrixShape` to a `CoopMatShapeMeta` for the metadata sidecar.
///
/// Called by `compile_source_with_meta` to fill `KernelMetadata.coopmat` from the
/// HIR-derived shape. The runtime reads the shape from the sidecar — nothing is
/// hardcoded in the runtime (AT-1552/HN-10).
///
/// M3.1.5 (D): returns `Result` so `scalar_ty_to_coopmat_scalar_meta` errors propagate
/// without panicking (Hard rule: no panic in library code).
fn hir_coopmat_shape_to_meta(shape: &CoopMatrixShape) -> Result<CoopMatShapeMeta, DriverError> {
    Ok(CoopMatShapeMeta {
        m: shape.m,
        n: shape.n,
        k: shape.k,
        a_type: scalar_ty_to_coopmat_scalar_meta(shape.a_elem)?,
        b_type: scalar_ty_to_coopmat_scalar_meta(shape.b_elem)?,
        c_type: scalar_ty_to_coopmat_scalar_meta(shape.c_elem)?,
        result_type: scalar_ty_to_coopmat_scalar_meta(shape.result_type)?,
        scope: match shape.scope {
            CoopMatScopeHir::Subgroup => CoopMatScopeMeta::Subgroup,
            CoopMatScopeHir::Workgroup => CoopMatScopeMeta::Workgroup,
        },
    })
}

/// Map an HIR `ScalarTy` to the restricted `CoopMatScalarMeta` enum.
///
/// Returns `Err(DriverError::UnsupportedCoopmatScalar)` if the scalar type is not
/// a valid coopmat element type. This path is effectively unreachable because
/// `is_allowed_coopmat_element` gates HIR coopmat type creation, but we return a
/// typed error rather than panicking (Hard rule: no panic in library code — M3.1.5 D).
fn scalar_ty_to_coopmat_scalar_meta(ty: ScalarTy) -> Result<CoopMatScalarMeta, DriverError> {
    match ty {
        ScalarTy::F16 => Ok(CoopMatScalarMeta::F16),
        ScalarTy::F32 => Ok(CoopMatScalarMeta::F32),
        ScalarTy::I8  => Ok(CoopMatScalarMeta::I8),
        ScalarTy::U8  => Ok(CoopMatScalarMeta::U8),
        ScalarTy::I32 => Ok(CoopMatScalarMeta::I32),
        ScalarTy::U32 => Ok(CoopMatScalarMeta::U32),
        other => Err(DriverError::UnsupportedCoopmatScalar {
            ty: format!("{other:?}"),
        }),
    }
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
/// Anchored on the **lexer token stream**, not a raw-text `find` (M3.15 EB.3
/// footgun fix). The lexer already skips `//` comments and string-literal
/// contents, so a `@strategy` occurring inside a comment or inside an
/// `@intent("...")` string produces no `Annotation` token — it can no longer
/// be mistaken for the real block. The anchor is the first
/// `TokenKind::Annotation("strategy")` token's span.
///
/// From that anchor, skips optional whitespace and looks for `{`, then
/// removes the entire span including the closing `}`.  Content between the
/// braces is discarded, with depth-tracked brace matching (handles nested
/// `{ }` inside the block, e.g. a value list containing braces).
///
/// If no `@strategy {` is found (either no `Annotation("strategy")` token at
/// all, or it's not followed by `{` — e.g. the `@strategy(...)` call form),
/// the source is returned unchanged.
pub(crate) fn strip_strategy_annotation_block(source: &str) -> String {
    // Find the first `Annotation("strategy")` token via the real lexer — this
    // is comment/string-literal safe (M3.14 footgun: a naive `source.find`
    // would latch onto a decoy inside a `//` comment or `@intent(...)` prose).
    let (tokens, _errs) = axc_lexer::tokenize(source);
    let Some(strategy_tok) = tokens.iter().find(|t| {
        matches!(&t.kind, axc_lexer::TokenKind::Annotation(name) if name == "strategy")
    }) else {
        return source.to_string();
    };

    // The Annotation token's span starts at the `@` sigil itself (verified
    // against the lexer: `lex_annotation` is called with `start` = the
    // position BEFORE the `@` byte is consumed). Guard this with a
    // debug_assert (AT-2832 pins the byte-offset behavior in release too via
    // the assertion on stripped output) rather than assuming silently.
    let mut at_pos: usize = strategy_tok.span.start as usize;
    if !source[at_pos..].starts_with("@strategy") {
        // Defensive fallback in case a future lexer change moves the span to
        // start at the name instead of the sigil — back up one byte to `@`.
        at_pos = at_pos.saturating_sub(1);
    }
    debug_assert!(
        source[at_pos..].starts_with("@strategy"),
        "strip_strategy_annotation_block: computed at_pos {at_pos} does not point at `@strategy`"
    );

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
///
/// Compiles in RELEASE mode (`--debug` OFF). Forwards to `compile_file_debug`.
pub fn compile_file(input: &Path, output: &Path) -> Result<(), DriverError> {
    compile_file_debug(input, output, false)
}

/// M3.17 (FG.4): `compile_file` with an explicit `--debug` flag. See
/// `compile_source_with_meta_debug` for the behavioral contract.
pub fn compile_file_debug(input: &Path, output: &Path, debug_checks: bool) -> Result<(), DriverError> {
    let source: String = std::fs::read_to_string(input)?;
    let (bytes, metadata) = compile_source_with_meta_debug(&source, debug_checks)?;
    std::fs::write(output, &bytes)?;
    let sidecar_path: PathBuf = metadata_sidecar_path(output);
    metadata.save(&sidecar_path)
        .map_err(DriverError::MetadataEmitFailed)?;
    Ok(())
}

/// M3.15 (EB.3): Build the child-process invocation for `axc bench`.
///
/// Returns `(program, args, env_overrides)`.  Pure — does NOT spawn.  This is the
/// unit-test seam (AT-2828) so the argv/env mapping is verified without running cargo.
///
/// Mapping (exact, test-pinned):
/// - `program = "cargo"`.
/// - base `args = ["bench", "-p", "axc-driver"]`.
/// - if `filter = Some(f)`: append `["--", f]` (Criterion positional filter after the `--`).
/// - if `bless`: `env_overrides = [("AXC_BLESS_BASELINES".into(), "1".into())]`; else `[]`.
///
/// NOTE (self-deadlock footgun): the caller MUST invoke this via the
/// installed/built `axc` binary, NOT `cargo run -p axc-driver -- bench` — the
/// spawned `cargo bench` needs the workspace target-dir build lock, and an outer
/// `cargo run` already holds it, so the inner cargo would block forever.
pub fn build_bench_command(
    filter: Option<&str>,
    bless: bool,
) -> (String, Vec<String>, Vec<(String, String)>) {
    let program: String = "cargo".to_owned();
    let mut args: Vec<String> = vec!["bench".to_owned(), "-p".to_owned(), "axc-driver".to_owned()];
    if let Some(f) = filter {
        args.push("--".to_owned());
        args.push(f.to_owned());
    }
    let env_overrides: Vec<(String, String)> = if bless {
        vec![("AXC_BLESS_BASELINES".to_owned(), "1".to_owned())]
    } else {
        Vec::new()
    };
    (program, args, env_overrides)
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

    // ── AT-2830..2835: strip_strategy_annotation_block robust fix (M3.15) ──────
    //
    // These live here (not in a `tests/` integration file) because
    // `strip_strategy_annotation_block` is `pub(crate)` — the M3.15 spec's
    // suggested integration-test location (`tests/strategy_saxpy_autotune.rs`
    // or a new `tests/strip_strategy_robust.rs`) is not reachable from an
    // external test binary; unit tests here follow the same convention as the
    // pre-existing AT-1043/AT-1044 tests for the same function.

    /// AT-2830: line-comment decoy — the real M3.14 footgun. A `@strategy`
    /// inside a `//` comment must not be mistaken for the real block. The OLD
    /// `source.find("@strategy")` algorithm would latch onto the comment
    /// occurrence, see it isn't followed by `{`, and return the source
    /// UNCHANGED — leaving the real block's `?[...]` candidates unresolved
    /// (bug reproduced by AT-2835's `naive_strip_reference`; fixed here by
    /// anchoring on the lexer's `Annotation` token stream, which emits no
    /// token for text inside a comment).
    #[test]
    fn at_2830_line_comment_decoy_does_not_block_real_strip() {
        let src = "// tune @strategy here\n@strategy { x: ?[1] }\n@kernel fn k() -> void { return; }\n";
        let result = strip_strategy_annotation_block(src);
        assert!(
            !result.contains("@strategy {"),
            "the real @strategy block must be stripped; got {result:?}"
        );
        assert!(
            !result.contains("?["),
            "the ?[...] candidate list must be gone after stripping; got {result:?}"
        );
        assert!(
            result.contains("// tune @strategy here"),
            "the comment line must survive verbatim; got {result:?}"
        );
        assert!(
            result.contains("@kernel fn k()"),
            "the kernel body must survive; got {result:?}"
        );
    }

    /// AT-2831: `@intent("...")` prose decoy before the real block. A
    /// `@strategy` substring inside an `@intent("...")` string literal is part
    /// of a `String` token, not an `Annotation` token, so it must not be
    /// mistaken for the real block either.
    #[test]
    fn at_2831_intent_prose_decoy_does_not_block_real_strip() {
        let src = "@intent(\"explores @strategy space\")\n@strategy { x: ?[1] }\n@kernel fn k() -> void { return; }\n";
        let result = strip_strategy_annotation_block(src);
        assert!(
            !result.contains("@strategy {"),
            "the real @strategy block must be stripped; got {result:?}"
        );
        assert!(
            result.contains("@intent(\"explores @strategy space\")"),
            "the @intent(...) prose (including its @strategy substring) must survive verbatim; got {result:?}"
        );
    }

    /// AT-2832: real block alone, behavior preserved — pins the `@`-sigil byte
    /// offset by asserting the result is byte-identical to `source` with
    /// exactly the `@strategy { ... }` span removed (no off-by-one).
    #[test]
    fn at_2832_real_block_alone_sigil_offset_correct() {
        let src = "@kernel @workgroup(64,1,1) @strategy { rb_m: ?[2] } fn k() -> void { return; }";
        let result = strip_strategy_annotation_block(src);
        let expected = "@kernel @workgroup(64,1,1)  fn k() -> void { return; }";
        assert_eq!(
            result, expected,
            "stripped result must equal source with exactly the `@strategy {{...}}` span removed"
        );
    }

    /// AT-2833: no `@strategy` anywhere → source returned unchanged (identity).
    #[test]
    fn at_2833_no_strategy_is_identity() {
        let src = "@kernel @workgroup(64,1,1) fn k() -> void { return; }";
        let result = strip_strategy_annotation_block(src);
        assert_eq!(result, src, "source with no @strategy must be returned unchanged");
    }

    /// AT-2834: call-form `@strategy(foo)` (annotation followed by `(`, not
    /// `{`) → unchanged. The `{`-guard escape hatch still holds after the
    /// lexer-anchor rewrite.
    #[test]
    fn at_2834_call_form_is_unchanged() {
        let src = "@kernel @strategy(foo) fn k() -> void { return; }";
        let result = strip_strategy_annotation_block(src);
        assert_eq!(result, src, "@strategy(...) call form must be left unchanged");
    }

    /// AT-2835: corpus regression anchor. `naive_strip_reference` reproduces
    /// the OLD (pre-M3.15) `source.find("@strategy")` algorithm VERBATIM —
    /// this half is load-bearing and MUST NOT be "improved". For every
    /// `examples/*.axc` containing `@strategy`, asserts the NEW lexer-anchored
    /// `strip_strategy_annotation_block` is BYTE-IDENTICAL to the old
    /// algorithm's output (zero behavior change on the real corpus), and
    /// separately asserts no residual LIVE `@strategy {` remains — scoped to
    /// non-comment lines only, exactly mirroring the existing AT-1576
    /// precedent below (`examples/matmul_f32_tiled.axc:13` carries a
    /// brace-form `@strategy {` INSIDE a `//` comment, which correctly
    /// survives the lexer-anchored strip; an unscoped residual check would
    /// wrongly flag that file). The non-comment scoping applies ONLY to the
    /// residual clause, never to the `== naive_strip_reference` equality.
    #[test]
    fn at_2835_corpus_regression_anchor_matches_naive_reference() {
        /// Verbatim reproduction of the OLD `source.find("@strategy")`
        /// algorithm (pre-M3.15). Do NOT "fix" this — it is the regression
        /// baseline the new implementation is pinned against.
        fn naive_strip_reference(source: &str) -> String {
            let Some(at_pos) = source.find("@strategy") else {
                return source.to_string();
            };
            let after_keyword: &str = &source[at_pos + "@strategy".len()..];
            let trimmed: &str = after_keyword.trim_start_matches([' ', '\t', '\r', '\n']);
            if !trimmed.starts_with('{') {
                return source.to_string();
            }
            let open_brace_offset: usize = at_pos
                + "@strategy".len()
                + (after_keyword.len() - trimmed.len());
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
                None => return source.to_string(),
            };
            let before: &str = &source[..at_pos];
            let after: &str = &source[close_pos + 1..];
            format!("{before}{after}")
        }

        let manifest_dir: PathBuf = PathBuf::from(
            std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"),
        );
        let examples_dir: PathBuf = manifest_dir.join("..").join("..").join("examples");
        let mut checked: usize = 0;
        for entry in std::fs::read_dir(&examples_dir)
            .unwrap_or_else(|e| panic!("failed to read {examples_dir:?}: {e}"))
        {
            let entry = entry.expect("dir entry read failed");
            let path: PathBuf = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("axc") {
                continue;
            }
            let src: String = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("failed to read {path:?}: {e}"));
            if !src.contains("@strategy") {
                continue;
            }
            checked += 1;

            let naive: String = naive_strip_reference(&src);
            let robust: String = strip_strategy_annotation_block(&src);
            assert_eq!(
                robust, naive,
                "strip_strategy_annotation_block must be byte-identical to the old \
                 algorithm on {path:?} (zero behavior change on the real corpus)"
            );

            // Residual `@strategy {` check, scoped to NON-COMMENT lines only —
            // mirrors AT-1576 above. An unscoped check would wrongly flag
            // matmul_f32_tiled.axc's comment-form decoy at line 13.
            let has_live_strategy_block: bool = robust
                .lines()
                .filter(|line| !line.trim_start().starts_with("//"))
                .any(|line| line.contains("@strategy {"));
            assert!(
                !has_live_strategy_block,
                "{path:?}: a live (non-comment) `@strategy {{` block survived stripping"
            );
        }
        assert!(
            checked > 0,
            "expected at least one examples/*.axc containing @strategy to exercise this regression anchor"
        );
    }

    // ── AT-1575: scalar_ty_to_coopmat_scalar_meta — no panic (M3.1.5 D) ─────────

    /// AT-1575: Valid CoopMat scalar types map to Ok; invalid types return Err without panic.
    ///
    /// The effective error path is unreachable in production (guarded by is_allowed_coopmat_element
    /// in HIR), but the function must NOT panic — it must return a typed Err (M3.1.5 D).
    #[test]
    fn at_1575_coopmat_scalar_meta_no_panic() {
        // Valid types (F16, F32, I8, U8, I32, U32) must all map to Ok.
        assert!(matches!(scalar_ty_to_coopmat_scalar_meta(ScalarTy::F16), Ok(CoopMatScalarMeta::F16)));
        assert!(matches!(scalar_ty_to_coopmat_scalar_meta(ScalarTy::F32), Ok(CoopMatScalarMeta::F32)));
        assert!(matches!(scalar_ty_to_coopmat_scalar_meta(ScalarTy::I8),  Ok(CoopMatScalarMeta::I8)));
        assert!(matches!(scalar_ty_to_coopmat_scalar_meta(ScalarTy::U8),  Ok(CoopMatScalarMeta::U8)));
        assert!(matches!(scalar_ty_to_coopmat_scalar_meta(ScalarTy::I32), Ok(CoopMatScalarMeta::I32)));
        assert!(matches!(scalar_ty_to_coopmat_scalar_meta(ScalarTy::U32), Ok(CoopMatScalarMeta::U32)));
        // Invalid type (Bool is not a coopmat element type) must return Err, not panic.
        let result = scalar_ty_to_coopmat_scalar_meta(ScalarTy::Bool);
        assert!(
            matches!(result, Err(DriverError::UnsupportedCoopmatScalar { .. })),
            "Bool is not a valid coopmat element type; expected UnsupportedCoopmatScalar, got: {result:?}"
        );
    }

    // ── AT-1576: matmul_f32_tiled compiles without @strategy holes (M3.1.5) ─────

    /// AT-1576: matmul_f32_tiled.axc compiles via compile_source_with_meta after the
    /// inert @strategy holes are removed (HANDOFF-0). No UnresolvedStrategyHole, no
    /// compile_source_with_assignments needed.
    #[test]
    fn at_1576_matmul_f32_tiled_compiles_without_strategy_holes() {
        let src = include_str!("../../../examples/matmul_f32_tiled.axc");
        // Must NOT contain a live @strategy annotation (non-comment lines only).
        // Comments documenting the removal are fine; the annotation itself must be gone.
        let has_strategy_annotation = src.lines()
            .filter(|line| !line.trim_start().starts_with("//"))
            .any(|line| line.contains("@strategy"));
        assert!(
            !has_strategy_annotation,
            "matmul_f32_tiled.axc must NOT contain @strategy annotation (non-comment) after M3.1.5 HANDOFF-0"
        );
        let result = compile_source_with_meta(src);
        assert!(
            result.is_ok(),
            "matmul_f32_tiled.axc must compile via compile_source_with_meta (no assignments): {result:?}"
        );
        let (bytes, meta) = result.unwrap();
        assert_eq!(&bytes[0..4], &[0x03, 0x02, 0x23, 0x07], "SPIR-V magic must be correct");
        // Push constants: m, n, k as 3 × u32 = 12 bytes.
        assert_eq!(meta.push_constant_total_bytes, 12,
            "matmul_f32_tiled must have 12 push constant bytes (m, n, k as u32)");
        // 3 buffer bindings: a_buf (readonly), b_buf (readonly), c_buf (readwrite).
        assert_eq!(meta.binding_plan.buffers.len(), 3,
            "matmul_f32_tiled must have 3 buffer bindings");
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

    // ── AT-2828: build_bench_command — pure argv/env mapping (EB.3, M3.15) ─────

    /// AT-2828: build_bench_command produces the exact test-pinned argv/env mapping.
    #[test]
    fn at_2828_build_bench_command_mapping() {
        assert_eq!(
            build_bench_command(None, false),
            (
                "cargo".to_owned(),
                vec!["bench".to_owned(), "-p".to_owned(), "axc-driver".to_owned()],
                Vec::<(String, String)>::new(),
            ),
            "no filter, no bless: base argv, no env overrides"
        );

        assert_eq!(
            build_bench_command(Some("cpu_saxpy"), true),
            (
                "cargo".to_owned(),
                vec![
                    "bench".to_owned(),
                    "-p".to_owned(),
                    "axc-driver".to_owned(),
                    "--".to_owned(),
                    "cpu_saxpy".to_owned(),
                ],
                vec![("AXC_BLESS_BASELINES".to_owned(), "1".to_owned())],
            ),
            "filter + bless: filter appended after `--`, AXC_BLESS_BASELINES=1 set"
        );
    }

    // ── AT-2874: @strict + a REAL precondition interaction (M3.17 FG.4) ────────

    /// AT-2874: a `@strict` kernel (`@intent`+`@complexity`+`@precondition`) with a
    /// REAL (non-`true`) precondition compiles; `--debug` lowers the precondition;
    /// the M2.5 strict-discipline path is unaffected; and the RELEASE artifact is
    /// byte-unchanged by the presence of `@precondition` (ties to AT-2866).
    #[test]
    fn at_2874_strict_kernel_with_real_precondition() {
        const STRICT_SRC: &str = concat!(
            "@kernel\n",
            "@workgroup(64, 1, 1)\n",
            "@intent(\"strict kernel with a real precondition\")\n",
            "@complexity(O(n))\n",
            "@precondition(gt(n, 0))\n",
            "@strict\n",
            "fn strict_k(n: u32) -> void { return; }\n",
        );
        const STRICT_SRC_NO_PRECONDITION: &str = concat!(
            "@kernel\n",
            "@workgroup(64, 1, 1)\n",
            "@intent(\"strict kernel with a real precondition\")\n",
            "@complexity(O(n))\n",
            "@strict\n",
            "fn strict_k(n: u32) -> void { return; }\n",
        );

        // Release: compiles clean, no debug_checks in metadata.
        let (release_bytes, release_meta) = compile_source_with_meta_debug(STRICT_SRC, false)
            .expect("release compile must succeed for a @strict kernel with a real precondition");
        assert!(release_meta.debug_checks.is_none(), "release metadata must have no debug_checks");

        // Release artifact is byte-unchanged by the presence of @precondition (AT-2866 tie-in).
        let (release_bytes_no_pc, _) = compile_source_with_meta_debug(STRICT_SRC_NO_PRECONDITION, false)
            .expect("release compile (no precondition) must succeed");
        assert_eq!(release_bytes, release_bytes_no_pc,
            "release SPIR-V for a @strict kernel must be byte-unchanged by @precondition's presence");

        // --debug: lowers the precondition into metadata.
        let (_debug_bytes, debug_meta) = compile_source_with_meta_debug(STRICT_SRC, true)
            .expect("--debug compile must succeed for a @strict kernel");
        let dbg = debug_meta.debug_checks.expect("--debug metadata must carry debug_checks");
        assert_eq!(dbg.conditions.len(), 1, "exactly the one real precondition must be lowered");
        assert_eq!(dbg.conditions[0].text, "gt(n, 0)");
    }
}
