//! `verify_rewrite` MCP tool — Layer 2 (M3.16 / FG.1): a thin JSON-RPC wrapper
//! over the `rewrite_verify` core. The agent supplies BOTH `original` and
//! `rewritten` sources; the tool returns a machine-checked `VerifyReport`.
//!
//! # Tolerance governance (r2/CRIT-2)
//!
//! The tool's default tolerance policy is the SERVER-OWNED `BitExact`
//! (`default_bit_exact`), never a caller default. A caller-supplied tolerance
//! looser than the server default (i.e. anything non-`BitExact`) is accepted
//! only with a loud `tolerance_overridden: true` stamp + a `notes` entry — the
//! verdict consumer (the outer supervisor agent, pipeline rule #3) audits the
//! verdict JSON, so a loosened bar cannot reach a merge decision silently.
//!
//! # Vulkan-unavailable divergence (documented, intentional)
//!
//! Unlike the CLI's local `VulkanContext::new().ok()` (which turns
//! Vulkan-unavailable into a `SKIPPED` verdict for CI ergonomics), the MCP path
//! treats Vulkan-unavailable as a hard RPC error (`VULKAN_UNAVAILABLE`,
//! -32004) via `McpContext::vulkan.get_or_init()`: the caller explicitly asked
//! to verify and the server could not.

use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::mcp::dispatch::{resolve_source, McpContext, McpToolError};
use crate::mcp::tools::compile_variant::base64_decode;
use crate::rewrite_verify::{verify_rewrite, TolerancePolicy, VerifyReport, VerifyRequest};

fn default_bit_exact() -> String {
    "bit-exact".to_string()
}

/// Request for the `verify_rewrite` tool.
///
/// Exactly one of `original_source`/`original_path` and exactly one of
/// `rewritten_source`/`rewritten_path` must be supplied (the existing
/// `resolve_source` exactly-one-of rule, applied independently per side).
#[derive(Debug, serde::Deserialize)]
pub struct VerifyRewriteRequest {
    /// Inline original source text. Mutually exclusive with `original_path`.
    #[serde(default)]
    pub original_source: Option<String>,
    /// Path to the original `.axc` source file. Mutually exclusive with `original_source`.
    #[serde(default)]
    pub original_path: Option<PathBuf>,
    /// Inline rewritten source text. Mutually exclusive with `rewritten_path`.
    #[serde(default)]
    pub rewritten_source: Option<String>,
    /// Path to the rewritten `.axc` source file. Mutually exclusive with `rewritten_source`.
    #[serde(default)]
    pub rewritten_path: Option<PathBuf>,
    /// Shared `@strategy` hole resolution applied to BOTH sources (default empty).
    #[serde(default)]
    pub assignments: BTreeMap<String, i64>,
    /// Tolerance grammar: `"bit-exact"` | `"ulp:N"` | `"rel:EPS"`. SERVER-OWNED
    /// default (`bit-exact`) — never a caller default (r2/CRIT-2).
    #[serde(default = "default_bit_exact")]
    pub tolerance: String,
    /// One size (bytes) per buffer binding. Required.
    pub buffer_sizes: Vec<usize>,
    /// Output-buffer sizes (bytes). Defaults to `buffer_sizes`.
    #[serde(default)]
    pub output_sizes: Option<Vec<usize>>,
    /// Explicit dispatch grid. Mandatory for coopmat/shared kernels.
    #[serde(default)]
    pub workgroups: Option<[u32; 3]>,
    /// Base64-encoded (RFC 4648 §4 standard alphabet) push-constant bytes.
    #[serde(default)]
    pub push_constants_base64: Option<String>,
    /// Seed for deterministic input generation. Default `0`.
    #[serde(default)]
    pub seed: u64,
}

/// Handle a `verify_rewrite` request.
///
/// Vulkan-unavailable is a hard `McpToolError::Vulkan` here (RPC error
/// -32004), NOT a `SKIPPED` verdict — the CLI path is where GPU-optional
/// `SKIPPED` ergonomics live.
pub(crate) fn handle(req: VerifyRewriteRequest, ctx: &mut McpContext) -> Result<VerifyReport, McpToolError> {
    let original: String = resolve_source(&req.original_source, &req.original_path)?;
    let rewritten: String = resolve_source(&req.rewritten_source, &req.rewritten_path)?;

    let requested: TolerancePolicy = req
        .tolerance
        .parse()
        .map_err(McpToolError::InvalidParams)?;
    // r2/CRIT-2: ANY non-BitExact caller tolerance is a loosening past the
    // server-owned default, regardless of its actual strictness — flagged
    // unconditionally so the stamp can never be silently skipped.
    let overridden: bool = requested != TolerancePolicy::BitExact;

    let push_constants: Option<Vec<u8>> = match &req.push_constants_base64 {
        Some(b64) => Some(
            base64_decode(b64)
                .map_err(|e| McpToolError::InvalidParams(format!("push_constants_base64: {e}")))?,
        ),
        None => None,
    };

    let vk_req = VerifyRequest {
        original,
        rewritten,
        assignments: req.assignments,
        tolerance: requested,
        buffer_sizes: req.buffer_sizes,
        output_sizes: req.output_sizes,
        workgroups: req.workgroups,
        push_constants,
        seed: req.seed,
    };

    // MCP callers want a hard error on Vulkan-unavailable (documented divergence
    // from the CLI's SKIPPED ergonomics — see module docs).
    let vk: &axc_runtime::VulkanContext = ctx.vulkan.get_or_init()?;

    let mut report: VerifyReport = verify_rewrite(&vk_req, Some(vk));
    if overridden {
        report.tolerance_overridden = true;
        report.notes.push(format!("tolerance_loosened_by_caller: {requested}"));
    }
    Ok(report)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_tolerance_is_bit_exact() {
        assert_eq!(default_bit_exact(), "bit-exact");
    }

    /// AT-2849 (CPU-checkable half): a request-level deserialize round-trip
    /// confirms `tolerance` defaults to the server-set `"bit-exact"` when the
    /// caller omits the field entirely.
    #[test]
    fn verify_rewrite_request_tolerance_defaults_to_bit_exact() {
        let json = serde_json::json!({
            "original_source": "@kernel fn k() -> void { return; }",
            "rewritten_source": "@kernel fn k() -> void { return; }",
            "buffer_sizes": [],
        });
        let req: VerifyRewriteRequest = serde_json::from_value(json).expect("must deserialize");
        assert_eq!(req.tolerance, "bit-exact");
    }

    /// AT-2849: missing `buffer_sizes` (a required field) fails deserialization
    /// — the caller sees `INVALID_PARAMS` via the generic `narrow_and_run_ctx`
    /// path (no per-field custom handling needed).
    #[test]
    fn verify_rewrite_request_missing_buffer_sizes_fails_deserialize() {
        let json = serde_json::json!({
            "original_source": "@kernel fn k() -> void { return; }",
            "rewritten_source": "@kernel fn k() -> void { return; }",
        });
        let result: Result<VerifyRewriteRequest, _> = serde_json::from_value(json);
        assert!(result.is_err(), "buffer_sizes is required; missing field must fail to deserialize");
    }

    /// AT-2849 (CPU-checkable parse-level half): `tolerance: "rel:1e-3"` is a
    /// non-BitExact caller override — asserted purely on the parse logic (no
    /// GPU needed). This is the pure-`TolerancePolicy` complement only; the
    /// full end-to-end assertion that a real `verify_rewrite` MCP call with
    /// this tolerance string stamps `tolerance_overridden: true` + a
    /// `tolerance_loosened_by_caller` notes entry on the returned report
    /// lives in `at_2849_verify_rewrite_tolerance_overridden_stamp_on_lavapipe`
    /// in `crates/axc-driver/tests/mcp_roundtrip.rs` (drives the real
    /// `handle()` through the actual MCP dispatch path on Lavapipe).
    #[test]
    fn non_bit_exact_tolerance_string_parses_and_is_flagged_overridden() {
        let policy: TolerancePolicy = "rel:1e-3".parse().expect("must parse");
        assert_ne!(policy, TolerancePolicy::BitExact);
    }
}
