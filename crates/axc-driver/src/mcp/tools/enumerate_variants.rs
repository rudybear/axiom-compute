//! `enumerate_variants` tool — enumerate the Cartesian product of @strategy holes.
//!
//! Returns per-variant ordinal, stable `variant_id`, and `BTreeMap` assignments.
//! Returns `-32002 ENUMERATE_ERROR` if the kernel has no `@strategy` block.

use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::mcp::dispatch::McpToolError;

// ── Request / Response types ──────────────────────────────────────────────────

/// Request for the `enumerate_variants` tool.
#[derive(Debug, serde::Deserialize)]
pub struct EnumerateVariantsRequest {
    /// Inline source text. Mutually exclusive with `path`.
    #[serde(default)]
    pub source: Option<String>,
    /// Path to an `.axc` source file. Mutually exclusive with `source`.
    #[serde(default)]
    pub path: Option<PathBuf>,
}

/// Response from the `enumerate_variants` tool.
#[derive(Debug, serde::Serialize)]
pub struct EnumerateVariantsResponse {
    /// All enumerated strategy variants in mixed-radix order.
    pub variants: Vec<StrategyVariantSummary>,
}

/// A single enumerated strategy variant.
#[derive(Debug, serde::Serialize)]
pub struct StrategyVariantSummary {
    /// Zero-based ordinal in the Cartesian product.
    pub ordinal: u64,
    /// Stable xxh3_64 fingerprint of the canonical key-value encoding.
    pub variant_id: u64,
    /// Concrete hole assignments (alphabetical key order).
    pub assignments: BTreeMap<String, i64>,
}

// ── Handler ───────────────────────────────────────────────────────────────────

/// Handle an `enumerate_variants` request.
pub(crate) fn handle(
    req: EnumerateVariantsRequest,
) -> Result<EnumerateVariantsResponse, McpToolError> {
    let source: String = crate::mcp::dispatch::resolve_source(&req.source, &req.path)?;
    enumerate_source(&source)
}

/// Shared helper: run the pipeline and enumerate strategy variants.
pub(crate) fn enumerate_source(source: &str) -> Result<EnumerateVariantsResponse, McpToolError> {
    use axc_lexer::tokenize;
    use axc_parser::Parser;
    use axc_hir::lower_module;
    use axc_optimize::enumerator::enumerate_strategy;
    use crate::DriverError;

    // BOM check
    if source.as_bytes().starts_with(&[0xEF, 0xBB, 0xBF]) {
        return Err(McpToolError::Compile(DriverError::UnexpectedByteOrderMark {
            span: axc_lexer::Span { start: 0, end: 3 },
        }));
    }

    let (tokens, lex_errors) = tokenize(source);
    let mut parser: Parser = Parser::new(&tokens);
    let (ast, parse_errors) = parser.parse_module();
    let (hir, hir_errors, _warns) = lower_module(&ast);

    if !lex_errors.is_empty() || !parse_errors.is_empty() || !hir_errors.is_empty() {
        return Err(McpToolError::Compile(DriverError::Compile {
            lex: lex_errors,
            parse: parse_errors,
            hir: hir_errors,
        }));
    }

    let kernel = hir.kernels.into_iter().next()
        .ok_or_else(|| McpToolError::InvalidParams(
            "source must declare exactly one @kernel function".to_string()
        ))?;

    let strategy = kernel.annotations.strategy
        .ok_or(McpToolError::Enumerate(
            axc_optimize::EnumerateError::EmptyStrategy
        ))?;

    let raw_variants = enumerate_strategy(&strategy)
        .map_err(McpToolError::Enumerate)?;

    let variants: Vec<StrategyVariantSummary> = raw_variants
        .into_iter()
        .map(|v| StrategyVariantSummary {
            ordinal: v.ordinal,
            variant_id: v.variant_id,
            assignments: v.assignments.values,
        })
        .collect();

    Ok(EnumerateVariantsResponse { variants })
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SAXPY_STRATEGY_SRC: &str = concat!(
        "@kernel @workgroup(64, 1, 1)\n",
        "@strategy { wg: ?[32, 64, 128] }\n",
        "fn saxpy(n: u32, alpha: f32, x: readonly_buffer[f32], y: buffer[f32]) -> void {\n",
        "    let i: u32 = gid(0);\n",
        "    return;\n",
        "}\n",
    );

    #[test]
    fn enumerate_saxpy_returns_3_variants() {
        let resp: EnumerateVariantsResponse = enumerate_source(SAXPY_STRATEGY_SRC)
            .expect("enumerate_source must succeed for saxpy strategy");

        assert_eq!(resp.variants.len(), 3);
        assert_eq!(resp.variants[0].ordinal, 0);
        assert_eq!(resp.variants[0].assignments["wg"], 32);
        assert_eq!(resp.variants[1].assignments["wg"], 64);
        assert_eq!(resp.variants[2].assignments["wg"], 128);
    }

    #[test]
    fn enumerate_no_strategy_returns_enumerate_error() {
        let src = "@kernel @workgroup(64, 1, 1) fn k() -> void { return; }\n";
        let err = enumerate_source(src).unwrap_err();
        assert!(
            matches!(err, McpToolError::Enumerate(axc_optimize::EnumerateError::EmptyStrategy)),
            "expected EnumerateError::EmptyStrategy, got {err:?}"
        );
    }

    #[test]
    fn enumerate_two_holes_cartesian_product() {
        let src = concat!(
            "@kernel @workgroup(64, 1, 1)\n",
            "@strategy { a: ?[1, 2], b: ?[10, 20] }\n",
            "fn k() -> void { return; }\n",
        );
        let resp = enumerate_source(src).expect("must succeed");
        // 2 × 2 = 4 variants
        assert_eq!(resp.variants.len(), 4);
    }

    #[test]
    fn enumerate_variant_ids_are_deterministic() {
        let resp1 = enumerate_source(SAXPY_STRATEGY_SRC).unwrap();
        let resp2 = enumerate_source(SAXPY_STRATEGY_SRC).unwrap();
        for (v1, v2) in resp1.variants.iter().zip(resp2.variants.iter()) {
            assert_eq!(v1.variant_id, v2.variant_id, "variant_id must be deterministic");
        }
    }
}
