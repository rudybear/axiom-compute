//! JSON-RPC 2.0 dispatcher — routes method names to typed tool handlers.
//!
//! This module is the central narrowing point: `params: serde_json::Value` from
//! `RpcEnvelope` is deserialized into per-method typed request structs here,
//! then forwarded to the handler in the appropriate `tools::` submodule.
//!
//! ## Notification handling (JSON-RPC 2.0 §4.1)
//!
//! When `env.id` is `None` (the `id` key was absent from the JSON object), the
//! envelope is a *notification*. The handler is still invoked for side effects
//! (e.g. initialize could cache some state), but the return value is `None` —
//! the server loop must NOT write anything to stdout.
//!
//! `Some(Value::Null)` is a well-formed request (id == explicit null) and MUST
//! produce a response with `id: null`.

use serde_json::Value;

use crate::mcp::protocol::{
    ErrorCode, RpcEnvelope, RpcError, RpcResponse,
    make_error_response, make_result_response, response_id,
};

// ── Method name constants ─────────────────────────────────────────────────────

pub const METHOD_INITIALIZE: &str = "initialize";
pub const METHOD_LOAD_SOURCE: &str = "load_source";
pub const METHOD_ENUMERATE_VARIANTS: &str = "enumerate_variants";
pub const METHOD_COMPILE_VARIANT: &str = "compile_variant";
pub const METHOD_BENCH_VARIANT: &str = "bench_variant";
pub const METHOD_GRID_SEARCH: &str = "grid_search";
pub const METHOD_OPTIMIZATION_HISTORY: &str = "optimization_history";

/// All registered method names in alphabetical order (for error messages and initialize).
pub const ALL_METHODS: &[&str] = &[
    METHOD_INITIALIZE,
    METHOD_LOAD_SOURCE,
    METHOD_ENUMERATE_VARIANTS,
    METHOD_COMPILE_VARIANT,
    METHOD_BENCH_VARIANT,
    METHOD_GRID_SEARCH,
    METHOD_OPTIMIZATION_HISTORY,
];

// ── McpContext ────────────────────────────────────────────────────────────────

/// Per-session server state.
///
/// Created once per `run_mcp_server` invocation. Owns lazy-initialized Vulkan,
/// the history directory path, and the best-effort git SHA.
pub(crate) struct McpContext {
    /// Lazy-initialized Vulkan context (first bench/grid call triggers init).
    pub(crate) vulkan: OnceVulkan,
    /// Directory where JSONL history files are written.
    pub(crate) history_dir: std::path::PathBuf,
    /// Best-effort git HEAD SHA (first 7 chars), read once at startup.
    pub(crate) git_sha: Option<String>,
}

impl McpContext {
    /// Create a new `McpContext` with the given history directory.
    pub(crate) fn new(history_dir: std::path::PathBuf) -> Self {
        let git_sha: Option<String> = read_git_head_sha_best_effort();
        Self {
            vulkan: OnceVulkan::NotTried,
            history_dir,
            git_sha,
        }
    }
}

/// Lazy-initialized `VulkanContext`.
///
/// States:
/// - `NotTried` — first bench/grid call will attempt initialization.
/// - `Available(ctx)` — Vulkan is usable.
/// - `Unavailable(reason)` — probe failed; same reason returned on all subsequent calls.
pub(crate) enum OnceVulkan {
    NotTried,
    /// Boxed to reduce enum size (VulkanContext is ~680 bytes).
    Available(Box<axc_runtime::VulkanContext>),
    Unavailable(String),
}

impl OnceVulkan {
    /// Get or initialize the `VulkanContext`.
    ///
    /// On first call: probes `probe_vulkan_available()` then `VulkanContext::new()`.
    /// Caches the outcome (success or failure) for all subsequent calls.
    pub(crate) fn get_or_init(&mut self) -> Result<&axc_runtime::VulkanContext, McpToolError> {
        match self {
            OnceVulkan::Available(ctx) => Ok(ctx),
            OnceVulkan::Unavailable(reason) => {
                Err(McpToolError::Vulkan(axc_runtime::DispatchError::DeviceCreationFailed(
                    reason.clone()
                )))
            }
            OnceVulkan::NotTried => {
                if !axc_runtime::probe_vulkan_available() {
                    let reason: String = "Vulkan is not available on this host (probe returned false); \
                        set VK_DRIVER_FILES or install a Vulkan ICD".to_string();
                    *self = OnceVulkan::Unavailable(reason.clone());
                    return Err(McpToolError::Vulkan(
                        axc_runtime::DispatchError::DeviceCreationFailed(reason)
                    ));
                }
                match axc_runtime::VulkanContext::new() {
                    Ok(ctx) => {
                        *self = OnceVulkan::Available(Box::new(ctx));
                        match self {
                            OnceVulkan::Available(ctx) => Ok(ctx),
                            _ => unreachable!(),
                        }
                    }
                    Err(e) => {
                        let reason: String = e.to_string();
                        *self = OnceVulkan::Unavailable(reason.clone());
                        Err(McpToolError::Vulkan(e))
                    }
                }
            }
        }
    }
}

// ── McpToolError ─────────────────────────────────────────────────────────────

/// Uniform error type for tool handlers.
///
/// Each variant maps to a specific `ErrorCode` constant.
#[derive(Debug)]
#[allow(dead_code)] // `Internal` reserved for future tool handlers
pub(crate) enum McpToolError {
    /// Compilation phase failure (lex/parse/HIR/codegen).
    Compile(crate::DriverError),
    /// Strategy enumeration failure.
    Enumerate(axc_optimize::EnumerateError),
    /// Grid search failure.
    GridSearch(axc_optimize::GridSearchError),
    /// Vulkan unavailable or dispatch failed.
    Vulkan(axc_runtime::DispatchError),
    /// I/O error (source file, history file, etc.).
    Io(std::io::Error),
    /// In-process spirv-val rejected the compiled SPIR-V.
    SpirvVal(String),
    /// Invalid parameters from the caller.
    InvalidParams(String),
    /// Unexpected internal failure.
    Internal(String),
}

impl std::fmt::Display for McpToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            McpToolError::Compile(e)       => write!(f, "compile error: {e}"),
            McpToolError::Enumerate(e)     => write!(f, "enumerate error: {e}"),
            McpToolError::GridSearch(e)    => write!(f, "grid search error: {e}"),
            McpToolError::Vulkan(e)        => write!(f, "vulkan error: {e}"),
            McpToolError::Io(e)            => write!(f, "io error: {e}"),
            McpToolError::SpirvVal(s)      => write!(f, "spirv-val: {s}"),
            McpToolError::InvalidParams(s) => write!(f, "invalid params: {s}"),
            McpToolError::Internal(s)      => write!(f, "internal error: {s}"),
        }
    }
}

impl McpToolError {
    /// Convert to a JSON-RPC `RpcError` with the appropriate error code.
    pub(crate) fn to_rpc_error(&self) -> RpcError {
        let (code, message, data) = match self {
            McpToolError::Compile(e) => {
                let (lex_count, parse_count, hir_count) = match e {
                    crate::DriverError::Compile { lex, parse, hir } => {
                        (lex.len(), parse.len(), hir.len())
                    }
                    _ => (0, 0, 0),
                };
                let data: Value = serde_json::json!({
                    "category": "compile_error",
                    "detail": e.to_string(),
                    "lex_count": lex_count,
                    "parse_count": parse_count,
                    "hir_count": hir_count,
                });
                (ErrorCode::COMPILE_ERROR, e.to_string(), Some(data))
            }
            McpToolError::Enumerate(e) => {
                let data: Value = serde_json::json!({
                    "category": "enumerate_error",
                    "detail": e.to_string(),
                });
                (ErrorCode::ENUMERATE_ERROR, e.to_string(), Some(data))
            }
            McpToolError::GridSearch(e) => {
                let data: Value = serde_json::json!({
                    "category": "grid_search_error",
                    "detail": e.to_string(),
                });
                (ErrorCode::GRID_SEARCH_ERROR, e.to_string(), Some(data))
            }
            McpToolError::Vulkan(e) => {
                let data: Value = serde_json::json!({
                    "category": "vulkan_unavailable",
                    "detail": e.to_string(),
                });
                (ErrorCode::VULKAN_UNAVAILABLE, e.to_string(), Some(data))
            }
            McpToolError::Io(e) => {
                let data: Value = serde_json::json!({
                    "category": "io_error",
                    "detail": e.to_string(),
                });
                (ErrorCode::IO_ERROR, e.to_string(), Some(data))
            }
            McpToolError::SpirvVal(s) => {
                let data: Value = serde_json::json!({
                    "category": "spirv_val_failed",
                    "detail": s,
                });
                (ErrorCode::SPIRV_VAL_FAILED, format!("spirv-val failed: {s}"), Some(data))
            }
            McpToolError::InvalidParams(s) => {
                let data: Value = serde_json::json!({
                    "category": "invalid_params",
                    "detail": s,
                });
                (ErrorCode::INVALID_PARAMS, s.clone(), Some(data))
            }
            McpToolError::Internal(s) => {
                let data: Value = serde_json::json!({
                    "category": "internal_error",
                    "detail": s,
                });
                (ErrorCode::INTERNAL_ERROR, format!("internal error: {s}"), Some(data))
            }
        };
        RpcError { code, message, data }
    }
}

// ── resolve_source ────────────────────────────────────────────────────────────

/// Resolve source from either `source` (inline) or `path` (file).
///
/// Exactly one of `source` / `path` must be `Some`; both-None or both-Some
/// returns `McpToolError::InvalidParams`.
///
/// N-3: Does NOT enforce path-traversal policy (absolute paths and `..` are
/// permitted under the stdio single-user trust boundary). Emits `tracing::warn!`
/// on suspicious path shapes (leading `/` or containing `..`) for audit trail.
pub(crate) fn resolve_source(
    source: &Option<String>,
    path: &Option<std::path::PathBuf>,
) -> Result<String, McpToolError> {
    match (source.as_ref(), path.as_ref()) {
        (None, None) => Err(McpToolError::InvalidParams(
            "must supply exactly one of `source` or `path`".to_string()
        )),
        (Some(_), Some(_)) => Err(McpToolError::InvalidParams(
            "must supply exactly one of `source` or `path`, not both".to_string()
        )),
        (Some(s), None) => Ok(s.clone()),
        (None, Some(p)) => {
            // Audit log for suspicious paths (N-3).
            let path_str: &str = p.to_str().unwrap_or("");
            if path_str.starts_with('/') {
                tracing::warn!(
                    path = %path_str,
                    "load_source: absolute path supplied — proceeding (single-user trust boundary)"
                );
            }
            if path_str.contains("..") {
                tracing::warn!(
                    path = %path_str,
                    "load_source: path contains '..' — proceeding (single-user trust boundary)"
                );
            }
            std::fs::read_to_string(p)
                .map_err(McpToolError::Io)
        }
    }
}

// ── dispatch_request ─────────────────────────────────────────────────────────

/// Dispatch one parsed `RpcEnvelope` to the right tool handler.
///
/// Returns `Some(RpcResponse)` to write to stdout, or `None` for notifications
/// (JSON-RPC 2.0 §4.1 — the handler still runs for side effects).
///
/// Handler failures for id-bearing requests return error responses (not `Err`),
/// so the server loop continues running.
pub(crate) fn dispatch_request(
    env: RpcEnvelope,
    ctx: &mut McpContext,
) -> Option<RpcResponse> {
    let is_notification: bool = env.id.is_none();
    let resp_id: Value = response_id(env.id.as_ref());

    let response: RpcResponse = route(env, ctx, resp_id);

    if is_notification {
        None // JSON-RPC 2.0 §4.1: never emit a response for notifications
    } else {
        Some(response)
    }
}

/// Internal routing: call the handler and convert its result to `RpcResponse`.
fn route(env: RpcEnvelope, ctx: &mut McpContext, resp_id: Value) -> RpcResponse {
    match env.method.as_str() {
        METHOD_INITIALIZE => {
            handle_initialize(resp_id)
        }
        METHOD_LOAD_SOURCE => {
            narrow_and_run(env.params, resp_id, |req| {
                crate::mcp::tools::load_source::handle(req)
            })
        }
        METHOD_ENUMERATE_VARIANTS => {
            narrow_and_run(env.params, resp_id, |req| {
                crate::mcp::tools::enumerate_variants::handle(req)
            })
        }
        METHOD_COMPILE_VARIANT => {
            narrow_and_run(env.params, resp_id, |req| {
                crate::mcp::tools::compile_variant::handle(req)
            })
        }
        METHOD_BENCH_VARIANT => {
            narrow_and_run_ctx(env.params, resp_id, ctx, |req, c| {
                crate::mcp::tools::bench_variant::handle(req, c)
            })
        }
        METHOD_GRID_SEARCH => {
            narrow_and_run_ctx(env.params, resp_id, ctx, |req, c| {
                crate::mcp::tools::grid_search::handle(req, c)
            })
        }
        METHOD_OPTIMIZATION_HISTORY => {
            narrow_and_run_ctx(env.params, resp_id, ctx, |req, c| {
                crate::mcp::tools::optimization_history::handle(req, c)
            })
        }
        unknown => {
            let data: Value = serde_json::json!({
                "category": "method_not_found",
                "available": ALL_METHODS,
            });
            make_error_response(
                resp_id,
                ErrorCode::METHOD_NOT_FOUND,
                format!("method not found: {unknown}"),
                Some(data),
            )
        }
    }
}

/// Handle `initialize` — returns server info and tool list.
fn handle_initialize(resp_id: Value) -> RpcResponse {
    #[derive(serde::Serialize)]
    struct InitResult {
        server: &'static str,
        version: &'static str,
        tools: Vec<&'static str>,
    }
    let result = InitResult {
        server: "axc-mcp",
        version: "0.1.0",
        tools: vec![
            METHOD_LOAD_SOURCE,
            METHOD_ENUMERATE_VARIANTS,
            METHOD_COMPILE_VARIANT,
            METHOD_BENCH_VARIANT,
            METHOD_GRID_SEARCH,
            METHOD_OPTIMIZATION_HISTORY,
        ],
    };
    match make_result_response(resp_id.clone(), &result) {
        Ok(r) => r,
        Err(e) => make_error_response(resp_id, e.code, e.message, e.data),
    }
}

/// Deserialize params to `Req`, run `f`, convert result to `RpcResponse`.
fn narrow_and_run<Req, Res, F>(
    params: Value,
    resp_id: Value,
    f: F,
) -> RpcResponse
where
    Req: serde::de::DeserializeOwned,
    Res: serde::Serialize,
    F: FnOnce(Req) -> Result<Res, McpToolError>,
{
    let req: Req = match serde_json::from_value(params) {
        Ok(r) => r,
        Err(e) => {
            return make_error_response(
                resp_id,
                ErrorCode::INVALID_PARAMS,
                format!("invalid params: {e}"),
                Some(serde_json::json!({ "category": "invalid_params", "detail": e.to_string() })),
            );
        }
    };
    match f(req) {
        Ok(res) => match make_result_response(resp_id.clone(), &res) {
            Ok(r) => r,
            Err(e) => make_error_response(resp_id, e.code, e.message, e.data),
        },
        Err(e) => {
            let rpc_err: RpcError = e.to_rpc_error();
            make_error_response(resp_id, rpc_err.code, rpc_err.message, rpc_err.data)
        }
    }
}

/// Like `narrow_and_run` but also passes `McpContext` to the handler.
fn narrow_and_run_ctx<Req, Res, F>(
    params: Value,
    resp_id: Value,
    ctx: &mut McpContext,
    f: F,
) -> RpcResponse
where
    Req: serde::de::DeserializeOwned,
    Res: serde::Serialize,
    F: FnOnce(Req, &mut McpContext) -> Result<Res, McpToolError>,
{
    let req: Req = match serde_json::from_value(params) {
        Ok(r) => r,
        Err(e) => {
            return make_error_response(
                resp_id,
                ErrorCode::INVALID_PARAMS,
                format!("invalid params: {e}"),
                Some(serde_json::json!({ "category": "invalid_params", "detail": e.to_string() })),
            );
        }
    };
    match f(req, ctx) {
        Ok(res) => match make_result_response(resp_id.clone(), &res) {
            Ok(r) => r,
            Err(e) => make_error_response(resp_id, e.code, e.message, e.data),
        },
        Err(e) => {
            let rpc_err: RpcError = e.to_rpc_error();
            make_error_response(resp_id, rpc_err.code, rpc_err.message, rpc_err.data)
        }
    }
}

// ── git SHA best-effort ───────────────────────────────────────────────────────

/// Read the current git HEAD SHA best-effort.
///
/// Walks up from `CARGO_MANIFEST_DIR` to find `.git/HEAD`. Returns the first
/// 7 characters of the SHA when successful. All errors are silently swallowed.
pub(crate) fn read_git_head_sha_best_effort() -> Option<String> {
    let manifest_dir: String = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let mut dir: std::path::PathBuf = std::path::PathBuf::from(manifest_dir);

    // Walk up to find `.git/HEAD`.
    for _ in 0..10 {
        let head_path: std::path::PathBuf = dir.join(".git").join("HEAD");
        if head_path.exists() {
            let content: String = std::fs::read_to_string(&head_path).ok()?;
            let content: &str = content.trim();
            // Packed ref: `ref: refs/heads/<branch>`
            if let Some(branch_ref) = content.strip_prefix("ref: ") {
                let ref_path: std::path::PathBuf = dir.join(".git").join(branch_ref);
                if let Ok(sha) = std::fs::read_to_string(&ref_path) {
                    let sha: &str = sha.trim();
                    if sha.len() >= 7 {
                        return Some(sha[..7].to_string());
                    }
                }
            } else if content.len() >= 7 {
                // Detached HEAD: content is the SHA directly.
                return Some(content[..7].to_string());
            }
            return None;
        }
        if !dir.pop() {
            break;
        }
    }
    None
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// AT-1127: resolve_source enforces exactly-one-of.
    #[test]
    fn at_1127_resolve_source_exactly_one() {
        // Neither source nor path
        let err = resolve_source(&None, &None).unwrap_err();
        let msg: String = err.to_string();
        assert!(msg.contains("exactly one"), "error must mention 'exactly one'; got: {msg}");

        // Both source and path
        let err2 = resolve_source(
            &Some("src".to_string()),
            &Some(std::path::PathBuf::from("/some/path")),
        ).unwrap_err();
        let msg2: String = err2.to_string();
        assert!(msg2.contains("exactly one"), "error must mention 'exactly one'; got: {msg2}");

        // Only source → ok
        let result = resolve_source(&Some("@kernel fn k() -> void {}".to_string()), &None);
        assert!(result.is_ok(), "source only must succeed");

        // Only path with non-existent file → Io error
        let result2 = resolve_source(&None, &Some(std::path::PathBuf::from("/nonexistent_file_xyz")));
        assert!(matches!(result2, Err(McpToolError::Io(_))), "non-existent file must return Io error");
    }

    /// AT-1128: OnceVulkan caches Unavailable state (unit test via env scrubbing).
    #[test]
    fn at_1128_once_vulkan_caches_unavailable() {
        // This test probes OnceVulkan state without actually calling VulkanContext::new,
        // by testing the caching of the Unavailable branch.
        let mut ov: OnceVulkan = OnceVulkan::Unavailable("test reason".to_string());
        let r1 = ov.get_or_init();
        assert!(r1.is_err(), "Unavailable must return Err");
        // A second call must also return Err (the state did not change).
        let r2 = ov.get_or_init();
        assert!(r2.is_err(), "Unavailable must still return Err on second call");
    }

    #[test]
    fn handle_initialize_returns_correct_server_info() {
        let resp: RpcResponse = handle_initialize(Value::from(1_i64));
        assert!(resp.error.is_none(), "initialize must not return an error");
        let result: &Value = resp.result.as_ref().unwrap();
        assert_eq!(result["server"], "axc-mcp");
        assert_eq!(result["version"], "0.1.0");
        let tools: &Value = &result["tools"];
        assert!(tools.is_array());
        let tool_list: Vec<&str> = tools.as_array().unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(tool_list.contains(&"load_source"));
        assert!(tool_list.contains(&"enumerate_variants"));
        assert!(tool_list.contains(&"compile_variant"));
        assert!(tool_list.contains(&"bench_variant"));
        assert!(tool_list.contains(&"grid_search"));
        assert!(tool_list.contains(&"optimization_history"));
        assert_eq!(tool_list.len(), 6);
    }

    #[test]
    fn unknown_method_returns_32601() {
        let env = RpcEnvelope {
            jsonrpc: "2.0".to_string(),
            id: Some(Value::from(99_i64)),
            method: "fake_method".to_string(),
            params: Value::Null,
        };
        let history_dir = std::path::PathBuf::from(
            std::env::var("CARGO_MANIFEST_DIR")
                .unwrap_or_else(|_| "/tmp".to_string())
        );
        let mut ctx = McpContext::new(history_dir);
        let resp = dispatch_request(env, &mut ctx).expect("non-notification must return Some");
        assert!(resp.result.is_none());
        let e = resp.error.as_ref().unwrap();
        assert_eq!(e.code, ErrorCode::METHOD_NOT_FOUND);
    }

    #[test]
    fn notification_returns_none() {
        let env = RpcEnvelope {
            jsonrpc: "2.0".to_string(),
            id: None, // notification: no id key
            method: METHOD_INITIALIZE.to_string(),
            params: Value::Null,
        };
        let history_dir = std::path::PathBuf::from("/tmp");
        let mut ctx = McpContext::new(history_dir);
        let resp = dispatch_request(env, &mut ctx);
        assert!(resp.is_none(), "notification must produce None (no response)");
    }

    #[test]
    fn explicit_null_id_produces_response_with_null_id() {
        let env = RpcEnvelope {
            jsonrpc: "2.0".to_string(),
            id: Some(Value::Null), // explicit null id → must respond with id=null
            method: METHOD_INITIALIZE.to_string(),
            params: Value::Null,
        };
        let history_dir = std::path::PathBuf::from("/tmp");
        let mut ctx = McpContext::new(history_dir);
        let resp = dispatch_request(env, &mut ctx).expect("must return Some for explicit null id");
        assert_eq!(resp.id, Value::Null, "response id must be null for explicit null id request");
    }
}
