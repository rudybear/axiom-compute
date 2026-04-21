//! `load_source` tool — parse + HIR-lower, return kernel metadata without codegen.
//!
//! Accepts either `source` (inline string) or `path` (filesystem path).
//! Returns structured kernel metadata: name, workgroup size, binding plan summary,
//! strategy holes map, complexity, intent.

use std::collections::BTreeMap;
use std::path::PathBuf;

use axc_hir::buffer::BufferAccess;

use crate::mcp::dispatch::McpToolError;

// ── Request / Response types ──────────────────────────────────────────────────

/// Request for the `load_source` tool.
#[derive(Debug, serde::Deserialize)]
pub struct LoadSourceRequest {
    /// Inline source text. Mutually exclusive with `path`.
    #[serde(default)]
    pub source: Option<String>,
    /// Path to an `.axc` source file. Mutually exclusive with `source`.
    #[serde(default)]
    pub path: Option<PathBuf>,
}

/// Response from the `load_source` tool.
#[derive(Debug, serde::Serialize)]
pub struct LoadSourceResponse {
    /// Kernel function name.
    pub kernel_name: String,
    /// Resolved workgroup dimensions `[x, y, z]`.
    pub workgroup_size: [u32; 3],
    /// Binding plan summary (buffers + scalars).
    pub binding_plan_summary: BindingPlanSummary,
    /// Strategy holes: hole name → sorted list of candidate values.
    /// `BTreeMap` for deterministic JSON key order.
    pub strategy_holes: BTreeMap<String, Vec<i64>>,
    /// `@complexity(...)` in human-readable form, e.g. `"O(n)"`.
    pub complexity: Option<String>,
    /// `@intent("...")` free-form prose.
    pub intent: Option<String>,
}

/// Summary of the kernel's parameter binding plan.
#[derive(Debug, serde::Serialize)]
pub struct BindingPlanSummary {
    /// SSBO buffer bindings, ordered by `buffer_position`.
    pub buffers: Vec<BufferBindingSummary>,
    /// Push-constant scalar members, ordered by `member_index`.
    pub scalars: Vec<ScalarBindingSummary>,
    /// Total push-constant block size in bytes.
    pub push_constant_total_bytes: u32,
}

/// Single buffer binding summary.
#[derive(Debug, serde::Serialize)]
pub struct BufferBindingSummary {
    /// Parameter name.
    pub name: String,
    /// 0-based binding slot.
    pub buffer_position: u32,
    /// Access mode: `"readonly"`, `"writeonly"`, or `"readwrite"`.
    pub access: String,
    /// Element type name, e.g. `"f32"`, `"u32"`.
    pub element_ty: String,
}

/// Single scalar push-constant member summary.
#[derive(Debug, serde::Serialize)]
pub struct ScalarBindingSummary {
    /// Parameter name.
    pub name: String,
    /// Byte offset within the push-constant block.
    pub offset: u32,
    /// Type name, e.g. `"f32"`, `"u32"`.
    pub ty: String,
}

// ── Handler ───────────────────────────────────────────────────────────────────

/// Handle a `load_source` request.
///
/// Runs lex → parse → HIR on the resolved source and returns kernel metadata.
/// Returns `McpToolError::Compile` if any pipeline phase fails.
pub(crate) fn handle(req: LoadSourceRequest) -> Result<LoadSourceResponse, McpToolError> {
    let source: String = crate::mcp::dispatch::resolve_source(&req.source, &req.path)?;
    load_source_str(&source)
}

/// Shared helper: run the pipeline on a source string, return `LoadSourceResponse`.
///
/// Called directly by other tools that need kernel metadata before further processing.
pub(crate) fn load_source_str(source: &str) -> Result<LoadSourceResponse, McpToolError> {
    use axc_lexer::tokenize;
    use axc_parser::Parser;
    use axc_hir::lower_module;
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
        .ok_or_else(|| McpToolError::Compile(DriverError::Compile {
            lex: Vec::new(),
            parse: Vec::new(),
            hir: Vec::new(),
        }))?;

    let wg: [u32; 3] = [
        kernel.annotations.workgroup.x,
        kernel.annotations.workgroup.y,
        kernel.annotations.workgroup.z,
    ];

    let buffers: Vec<BufferBindingSummary> = kernel.binding_plan.buffers
        .iter()
        .map(|b| BufferBindingSummary {
            name: b.name.clone(),
            buffer_position: b.buffer_position,
            access: access_str(b.ty.access),
            element_ty: b.ty.elem.display_name().to_string(),
        })
        .collect();

    let scalars: Vec<ScalarBindingSummary> = kernel.binding_plan.scalars
        .iter()
        .map(|s| ScalarBindingSummary {
            name: s.name.clone(),
            offset: s.offset,
            ty: s.ty.display_name().to_string(),
        })
        .collect();

    let strategy_holes: BTreeMap<String, Vec<i64>> = kernel.annotations.strategy
        .as_ref()
        .map(|s| s.map.clone())
        .unwrap_or_default();

    let complexity: Option<String> = kernel.annotations.complexity
        .as_ref()
        .map(render_complexity);

    let intent: Option<String> = kernel.annotations.intent.clone();

    Ok(LoadSourceResponse {
        kernel_name: kernel.name,
        workgroup_size: wg,
        binding_plan_summary: BindingPlanSummary {
            buffers,
            scalars,
            push_constant_total_bytes: kernel.binding_plan.push_constant_total_bytes,
        },
        strategy_holes,
        complexity,
        intent,
    })
}

/// Render `BufferAccess` to the MCP wire string.
fn access_str(access: BufferAccess) -> String {
    match access {
        BufferAccess::ReadOnly  => "readonly".to_string(),
        BufferAccess::WriteOnly => "writeonly".to_string(),
        BufferAccess::ReadWrite => "readwrite".to_string(),
    }
}

/// Render a `ComplexityVar` to string.
fn render_complexity_var(v: &axc_hir::hir::ComplexityVar) -> &'static str {
    use axc_hir::hir::ComplexityVar;
    match v {
        ComplexityVar::One => "1",
        ComplexityVar::N => "n",
        ComplexityVar::NSquared => "n^2",
    }
}

/// Render a `ComplexityForm` to a human-readable string.
fn render_complexity(c: &axc_hir::hir::ComplexityForm) -> String {
    use axc_hir::hir::ComplexityForm;
    match c {
        ComplexityForm::O(v) => format!("O({})", render_complexity_var(v)),
        ComplexityForm::Theta(v) => format!("Theta({})", render_complexity_var(v)),
        ComplexityForm::Omega(v) => format!("Omega({})", render_complexity_var(v)),
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SAXPY_SRC: &str = concat!(
        "@kernel @workgroup(64, 1, 1)\n",
        "@intent(\"saxpy test\")\n",
        "@complexity(O(n))\n",
        "@strategy { wg: ?[32, 64, 128] }\n",
        "fn saxpy(n: u32, alpha: f32, x: readonly_buffer[f32], y: buffer[f32]) -> void {\n",
        "    let i: u32 = gid(0);\n",
        "    return;\n",
        "}\n",
    );

    #[test]
    fn load_saxpy_returns_correct_metadata() {
        let resp: LoadSourceResponse = load_source_str(SAXPY_SRC)
            .expect("load_source_str must succeed for saxpy");

        assert_eq!(resp.kernel_name, "saxpy");
        // workgroup_size is the unresolved HIR dims (holes stored as 1 placeholder
        // OR actual value — for wg axis with ?wg the HIR stores the *first candidate*
        // as the live dim; the exact value depends on HIR lowering).
        assert_eq!(resp.workgroup_size[1], 1);
        assert_eq!(resp.workgroup_size[2], 1);

        // Strategy holes
        assert_eq!(resp.strategy_holes["wg"], vec![32_i64, 64, 128]);

        // Complexity and intent
        assert_eq!(resp.complexity.as_deref(), Some("O(n)"));
        assert_eq!(resp.intent.as_deref(), Some("saxpy test"));

        // Binding plan: 2 buffers, 2 scalars
        assert_eq!(resp.binding_plan_summary.buffers.len(), 2);
        assert_eq!(resp.binding_plan_summary.scalars.len(), 2);
        assert_eq!(resp.binding_plan_summary.buffers[0].name, "x");
        assert_eq!(resp.binding_plan_summary.buffers[0].access, "readonly");
        assert_eq!(resp.binding_plan_summary.buffers[1].name, "y");
        assert_eq!(resp.binding_plan_summary.buffers[1].access, "readwrite");
        assert_eq!(resp.binding_plan_summary.scalars[0].name, "n");
        assert_eq!(resp.binding_plan_summary.scalars[1].name, "alpha");
        assert_eq!(resp.binding_plan_summary.push_constant_total_bytes, 8);
    }

    #[test]
    fn load_source_with_no_strategy_returns_empty_map() {
        let src = concat!(
            "@kernel @workgroup(64, 1, 1)\n",
            "fn k() -> void { return; }\n",
        );
        let resp: LoadSourceResponse = load_source_str(src)
            .expect("load_source_str must succeed");
        assert!(resp.strategy_holes.is_empty(), "no @strategy → empty map");
    }

    #[test]
    fn load_source_returns_compile_error_for_bad_source() {
        let err = load_source_str("💥 this is invalid").unwrap_err();
        assert!(
            matches!(err, McpToolError::Compile(_)),
            "expected McpToolError::Compile, got {err:?}"
        );
    }

    #[test]
    fn access_str_roundtrips() {
        assert_eq!(access_str(BufferAccess::ReadOnly), "readonly");
        assert_eq!(access_str(BufferAccess::WriteOnly), "writeonly");
        assert_eq!(access_str(BufferAccess::ReadWrite), "readwrite");
    }
}
