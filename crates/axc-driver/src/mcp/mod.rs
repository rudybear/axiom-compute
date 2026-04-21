//! MCP server — JSON-RPC 2.0 stdio bridge for LLM agents.
//!
//! # Usage
//!
//! ```no_run
//! use axc_driver::mcp::{run_mcp_server, LogTarget};
//! let result = run_mcp_server(
//!     std::io::BufReader::new(std::io::stdin().lock()),
//!     std::io::stdout().lock(),
//!     LogTarget::Stderr,
//! );
//! ```
//!
//! # Protocol
//!
//! NDJSON (newline-delimited JSON). Each input line is a JSON-RPC 2.0 request or
//! notification; each output line is a JSON-RPC 2.0 response or error.
//! Notifications (requests with no `id` field) produce no output line.
//!
//! # Tools
//!
//! `initialize`, `load_source`, `enumerate_variants`, `compile_variant`,
//! `bench_variant`, `grid_search`, `optimization_history`.

pub mod protocol;
pub mod dispatch;
pub mod tools;

pub use protocol::{RpcEnvelope, RpcResponse, RpcError, ErrorCode};
pub use tools::load_source::{
    LoadSourceRequest, LoadSourceResponse, BindingPlanSummary,
    BufferBindingSummary, ScalarBindingSummary,
};
pub use tools::enumerate_variants::{EnumerateVariantsRequest, EnumerateVariantsResponse, StrategyVariantSummary};
pub use tools::compile_variant::{CompileVariantRequest, CompileVariantResponse};
pub use tools::bench_variant::{BenchVariantRequest, BenchVariantResponse, MachineMetadata, CorrectnessStatus};
pub use tools::grid_search::{GridSearchRequest, GridSearchResponse, RankedVariant};
pub use tools::optimization_history::{OptHistoryRequest, OptHistoryResponse, HistoryEntry};

/// Maximum inbound line size: 8 MiB.
///
/// Lines larger than this are rejected with PARSE_ERROR and the server
/// continues running. Outbound responses have no hard cap (documented in
/// DESIGN.md §3.1.10, N-2).
const MAX_LINE_BYTES: usize = 8 * 1024 * 1024;

/// Log target for the MCP server.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogTarget {
    /// Log to stderr (default).
    Stderr,
    /// Discard all log output.
    Null,
}

/// Top-level server error (separate from per-RPC error codes, which are in-band).
#[derive(Debug, thiserror::Error)]
pub enum McpServerError {
    /// Underlying I/O failure on the stdio streams.
    #[error("mcp server io error: {0}")]
    Io(#[from] std::io::Error),
    /// Server shut down cleanly on EOF.
    #[error("mcp server shut down (EOF)")]
    Shutdown,
}

/// Run the MCP server loop.
///
/// Reads NDJSON from `input`, dispatches each request, writes responses to
/// `output`. Returns `Ok(())` on clean EOF. Returns `Err` only for
/// unrecoverable I/O errors on the transport itself.
///
/// Per-request errors (parse errors, invalid params, etc.) are sent as
/// JSON-RPC error responses in-band and do not terminate the loop.
///
/// # Log target
///
/// `LogTarget::Null` installs a no-op tracing subscriber (no stderr output).
/// `LogTarget::Stderr` installs a minimal `fmt` subscriber writing to stderr.
pub fn run_mcp_server<R, W>(
    mut input: R,
    mut output: W,
    log_target: LogTarget,
) -> Result<(), McpServerError>
where
    R: std::io::BufRead,
    W: std::io::Write,
{
    // Install tracing subscriber based on log_target.
    match log_target {
        LogTarget::Stderr => {
            // Best-effort install; ignore error if a subscriber is already set.
            let _ = tracing_subscriber::fmt()
                .with_writer(std::io::stderr)
                .try_init();
        }
        LogTarget::Null => {
            let _ = tracing_subscriber::fmt()
                .with_writer(std::io::sink)
                .try_init();
        }
    }

    // Build the McpContext with history_dir.
    let history_dir: std::path::PathBuf = resolve_history_dir();
    let mut ctx: dispatch::McpContext = dispatch::McpContext::new(history_dir);

    let mut line: String = String::new();

    loop {
        line.clear();
        // Step 1: read one line (NDJSON framing).
        let n_bytes: usize = input.read_line(&mut line)?;
        if n_bytes == 0 {
            // EOF — clean shutdown.
            return Ok(());
        }

        // Step 2: check for oversized line (N-2 inbound cap).
        if line.len() > MAX_LINE_BYTES {
            let resp = protocol::make_error_response(
                serde_json::Value::Null,
                ErrorCode::PARSE_ERROR,
                "line exceeds 8 MiB".to_string(),
                Some(serde_json::json!({ "reason": "line exceeds 8 MiB" })),
            );
            write_response(&mut output, &resp)?;
            continue;
        }

        // Step 3: trim trailing newline/CR.
        let trimmed: &str = line.trim_end_matches(['\r', '\n']);

        // Step 4: skip blank lines.
        if trimmed.is_empty() {
            continue;
        }

        // Step 5: parse JSON.
        let env: protocol::RpcEnvelope = match serde_json::from_str(trimmed) {
            Ok(e) => e,
            Err(e) => {
                let resp = protocol::make_error_response(
                    serde_json::Value::Null,
                    ErrorCode::PARSE_ERROR,
                    format!("invalid JSON: {e}"),
                    Some(serde_json::json!({ "category": "parse_error", "detail": e.to_string() })),
                );
                write_response(&mut output, &resp)?;
                continue;
            }
        };

        // Step 6a: B-5 — validate jsonrpc field.
        if env.jsonrpc != "2.0" {
            let resp_id: serde_json::Value = protocol::response_id(env.id.as_ref());
            let resp = protocol::make_error_response(
                resp_id,
                ErrorCode::INVALID_REQUEST,
                format!("jsonrpc must be \"2.0\", got \"{}\"", env.jsonrpc),
                Some(serde_json::json!({
                    "category": "invalid_request",
                    "received": env.jsonrpc,
                })),
            );
            write_response(&mut output, &resp)?;
            continue;
        }

        // Step 6b: B-5 — validate method non-empty.
        if env.method.is_empty() {
            let resp_id: serde_json::Value = protocol::response_id(env.id.as_ref());
            let resp = protocol::make_error_response(
                resp_id,
                ErrorCode::METHOD_NOT_FOUND,
                "method must not be empty".to_string(),
                Some(serde_json::json!({
                    "category": "method_not_found",
                    "available": dispatch::ALL_METHODS,
                })),
            );
            write_response(&mut output, &resp)?;
            continue;
        }

        // Step 7: dispatch.
        if let Some(resp) = dispatch::dispatch_request(env, &mut ctx) {
            // Warn if outbound response is large (N-2 soft warn).
            let json: String = serde_json::to_string(&resp)
                .unwrap_or_else(|e| format!("{{\"error\":\"serialize failed: {e}\"}}"));
            if json.len() > 1024 * 1024 {
                tracing::warn!(
                    size_bytes = json.len(),
                    "mcp: outbound response exceeds 1 MiB"
                );
            }
            output.write_all(json.as_bytes())?;
            output.write_all(b"\n")?;
            // Step 8: flush after every response.
            output.flush()?;
        }
        // Notification: nothing written, no flush needed.
    }
}

/// Write an `RpcResponse` as a single NDJSON line.
fn write_response<W: std::io::Write>(
    output: &mut W,
    resp: &protocol::RpcResponse,
) -> Result<(), McpServerError> {
    let json: String = serde_json::to_string(resp)
        .unwrap_or_else(|e| format!("{{\"error\":\"serialize failed: {e}\"}}"));
    output.write_all(json.as_bytes())?;
    output.write_all(b"\n")?;
    output.flush()?;
    Ok(())
}

/// Resolve the history directory.
///
/// Priority:
/// 1. `AXC_MCP_HISTORY_DIR` environment variable (for tests).
/// 2. Walk up from `CARGO_MANIFEST_DIR` to find workspace root (has `[workspace]`
///    in its Cargo.toml), then append `.pipeline/history/`.
/// 3. Fall back to `./pipeline/history/` in the current directory.
fn resolve_history_dir() -> std::path::PathBuf {
    // 1. Env var override (for tests).
    if let Ok(dir) = std::env::var("AXC_MCP_HISTORY_DIR") {
        return std::path::PathBuf::from(dir);
    }

    // 2. Walk up from CARGO_MANIFEST_DIR.
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let mut dir: std::path::PathBuf = std::path::PathBuf::from(manifest_dir);
        for _ in 0..10 {
            let cargo_toml: std::path::PathBuf = dir.join("Cargo.toml");
            if cargo_toml.exists() {
                if let Ok(content) = std::fs::read_to_string(&cargo_toml) {
                    if content.contains("[workspace]") {
                        return dir.join(".pipeline").join("history");
                    }
                }
            }
            if !dir.pop() {
                break;
            }
        }
    }

    // 3. Fallback.
    std::path::PathBuf::from(".pipeline/history")
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_mcp_server_eof_returns_ok() {
        // Empty input → immediate EOF → Ok(())
        let input = std::io::Cursor::new(b"");
        let mut output: Vec<u8> = Vec::new();
        let result = run_mcp_server(input, &mut output, LogTarget::Null);
        assert!(result.is_ok(), "EOF must return Ok; got {result:?}");
        assert!(output.is_empty(), "no output on EOF");
    }

    #[test]
    fn run_mcp_server_parse_error_continues() {
        let input = std::io::Cursor::new(b"{not json\n{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\"}\n");
        let mut output: Vec<u8> = Vec::new();
        let _result = run_mcp_server(input, &mut output, LogTarget::Null);
        let text = String::from_utf8(output).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        // Must have 2 lines: PARSE_ERROR + initialize response
        assert_eq!(lines.len(), 2, "must have 2 output lines; got: {lines:?}");

        let err_resp: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(err_resp["error"]["code"], -32700, "first must be parse error");

        let init_resp: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(init_resp["result"]["server"], "axc-mcp");
    }

    #[test]
    fn run_mcp_server_invalid_jsonrpc_version() {
        let input = std::io::Cursor::new(
            b"{\"jsonrpc\":\"1.0\",\"id\":1,\"method\":\"initialize\"}\n"
        );
        let mut output: Vec<u8> = Vec::new();
        let _result = run_mcp_server(input, &mut output, LogTarget::Null);
        let text = String::from_utf8(output).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 1);
        let resp: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(resp["error"]["code"], -32600, "must be InvalidRequest");
        assert_eq!(resp["error"]["data"]["received"], "1.0");
    }

    #[test]
    fn run_mcp_server_empty_method() {
        let input = std::io::Cursor::new(
            b"{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"\"}\n"
        );
        let mut output: Vec<u8> = Vec::new();
        let _result = run_mcp_server(input, &mut output, LogTarget::Null);
        let text = String::from_utf8(output).unwrap();
        let resp: serde_json::Value = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(resp["error"]["code"], -32601, "empty method must return MethodNotFound");
    }
}
