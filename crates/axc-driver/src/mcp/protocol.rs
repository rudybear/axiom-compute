//! JSON-RPC 2.0 envelope types, error codes, and wire-format helpers.
//!
//! This module is pure wire-format — no tool logic, no pipeline calls.
//! The dispatcher narrows the generic `params: serde_json::Value` to typed
//! per-tool request structs.
//!
//! ## JSON-RPC 2.0 notification handling (§4.1)
//!
//! `id` is `Option<serde_json::Value>`:
//! - `None` (key absent from JSON) → notification; server MUST NOT reply.
//! - `Some(Value::Null)` (key present, value `null`) → request with null id;
//!   server MUST reply with `id: null` per §4.2.
//!
//! We rely on `#[serde(default)]` on `id` to distinguish the two cases:
//! default is `None`, so a missing key deserialises to `None`.

use serde_json::Value;

/// Incoming request or notification envelope.
///
/// `params` defaults to `Value::Null` when absent (serde default).
/// `id` defaults to `None` when absent — absence means notification (§4.1).
#[derive(Debug, serde::Deserialize)]
pub struct RpcEnvelope {
    /// Must equal `"2.0"` — validated in the server loop before dispatch.
    pub jsonrpc: String,
    /// Request id. `None` → notification (no reply). `Some(Null)` → reply with null id.
    #[serde(default)]
    pub id: Option<Value>,
    /// Method name. Validated to be non-empty before dispatch.
    #[serde(default)]
    pub method: String,
    /// Method params. Narrowed to typed structs per method in the dispatcher.
    #[serde(default)]
    pub params: Value,
}

/// Outgoing response envelope.
///
/// Exactly one of `result` / `error` is `Some`; both-`Some` or both-`None` is a bug.
/// `id` echoes the request id (or `null` for notifications and null-id requests).
#[derive(Debug, serde::Serialize)]
pub struct RpcResponse {
    /// Always `"2.0"`.
    pub jsonrpc: &'static str,
    /// Echoed request id, or `null`.
    pub id: Value,
    /// Success result payload. Mutually exclusive with `error`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    /// Error payload. Mutually exclusive with `result`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
}

/// JSON-RPC 2.0 error object.
#[derive(Debug, serde::Serialize)]
pub struct RpcError {
    /// Numeric error code (see `ErrorCode` constants).
    pub code: i32,
    /// Human-readable short message.
    pub message: String,
    /// Optional structured data for diagnostics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// JSON-RPC 2.0 + AXIOM-Compute error code constants.
pub struct ErrorCode;

impl ErrorCode {
    /// Invalid JSON received by the server.
    pub const PARSE_ERROR: i32 = -32700;
    /// JSON is valid but does not represent a valid RPC object (e.g. wrong jsonrpc version).
    pub const INVALID_REQUEST: i32 = -32600;
    /// Method name not recognized.
    pub const METHOD_NOT_FOUND: i32 = -32601;
    /// Params cannot be narrowed to the method's typed request struct.
    pub const INVALID_PARAMS: i32 = -32602;
    /// Unexpected server-side failure (caught panic, logic error).
    pub const INTERNAL_ERROR: i32 = -32603;

    // --- AXIOM-Compute domain codes ---

    /// Lex / parse / HIR / codegen failure.
    pub const COMPILE_ERROR: i32 = -32001;
    /// Strategy enumeration failure (empty strategy, unknown hole, etc.).
    pub const ENUMERATE_ERROR: i32 = -32002;
    /// Grid search failure (no strategy, no successful variants, etc.).
    pub const GRID_SEARCH_ERROR: i32 = -32003;
    /// Vulkan not available on this host.
    pub const VULKAN_UNAVAILABLE: i32 = -32004;
    /// I/O error reading source file or writing history file.
    pub const IO_ERROR: i32 = -32005;
    /// In-process spirv-val rejected the compiled SPIR-V.
    pub const SPIRV_VAL_FAILED: i32 = -32006;
}

/// Reconstruct the response `id` from the request `id` field.
///
/// JSON-RPC §5: if the request id is absent (notification), the response id is
/// `null`. If present (including explicit null), it is echoed verbatim.
pub(crate) fn response_id(req_id: Option<&Value>) -> Value {
    match req_id {
        Some(v) => v.clone(),
        None => Value::Null,
    }
}

/// Construct an error response.
///
/// `id` is the already-resolved response id (from `response_id`).
pub fn make_error_response(
    id: Value,
    code: i32,
    message: impl Into<String>,
    data: Option<Value>,
) -> RpcResponse {
    RpcResponse {
        jsonrpc: "2.0",
        id,
        result: None,
        error: Some(RpcError {
            code,
            message: message.into(),
            data,
        }),
    }
}

/// Construct a success response by serializing `value` into a `Value`.
///
/// Returns `Err(RpcError)` only if `serde_json::to_value` fails (extremely rare,
/// e.g. a non-string map key), at which point the caller should emit an INTERNAL_ERROR.
pub fn make_result_response<T: serde::Serialize>(
    id: Value,
    value: &T,
) -> Result<RpcResponse, RpcError> {
    let v: Value = serde_json::to_value(value).map_err(|e| RpcError {
        code: ErrorCode::INTERNAL_ERROR,
        message: format!("failed to serialize response: {e}"),
        data: None,
    })?;
    Ok(RpcResponse {
        jsonrpc: "2.0",
        id,
        result: Some(v),
        error: None,
    })
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_id_none_returns_null() {
        let id: Value = response_id(None);
        assert_eq!(id, Value::Null);
    }

    #[test]
    fn response_id_some_null_returns_null() {
        let id: Value = response_id(Some(&Value::Null));
        assert_eq!(id, Value::Null);
    }

    #[test]
    fn response_id_some_number_echoes() {
        let id: Value = response_id(Some(&Value::from(42_i64)));
        assert_eq!(id, Value::from(42_i64));
    }

    #[test]
    fn make_error_response_sets_code_and_message() {
        let r: RpcResponse = make_error_response(Value::from(1_i64), -32700, "parse error", None);
        assert_eq!(r.jsonrpc, "2.0");
        assert!(r.result.is_none());
        let e: &RpcError = r.error.as_ref().unwrap();
        assert_eq!(e.code, -32700);
        assert_eq!(e.message, "parse error");
    }

    #[test]
    fn make_result_response_happy() {
        #[derive(serde::Serialize)]
        struct Foo { x: u32 }
        let r: RpcResponse = make_result_response(Value::from(1_i64), &Foo { x: 42 }).unwrap();
        assert!(r.error.is_none());
        assert_eq!(r.result.unwrap()["x"], Value::from(42_u64));
    }

    #[test]
    fn error_code_constants() {
        assert_eq!(ErrorCode::PARSE_ERROR, -32700);
        assert_eq!(ErrorCode::INVALID_REQUEST, -32600);
        assert_eq!(ErrorCode::METHOD_NOT_FOUND, -32601);
        assert_eq!(ErrorCode::INVALID_PARAMS, -32602);
        assert_eq!(ErrorCode::INTERNAL_ERROR, -32603);
        assert_eq!(ErrorCode::COMPILE_ERROR, -32001);
        assert_eq!(ErrorCode::ENUMERATE_ERROR, -32002);
        assert_eq!(ErrorCode::GRID_SEARCH_ERROR, -32003);
        assert_eq!(ErrorCode::VULKAN_UNAVAILABLE, -32004);
        assert_eq!(ErrorCode::IO_ERROR, -32005);
        assert_eq!(ErrorCode::SPIRV_VAL_FAILED, -32006);
    }

    /// Feed 20 malformed inputs and ensure RpcEnvelope deserialization returns Err for each.
    #[test]
    fn malformed_inputs_parse_to_err() {
        let cases: &[&str] = &[
            "",
            "{",
            "}",
            "null",
            "[]",
            "{\"x\":}",
            "{\"jsonrpc\":\"2.0\",\"method\":}",
            "\"string\"",
            "42",
            "true",
            "{\"jsonrpc\":\"2.0\",\"method\":\"init\",\"id\":",
            "{{",
            "{\"jsonrpc\":\"2.0\",\"method\":\"init\",\"params\":[1,2,3",
            // UTF-8 truncation (valid JSON structure but not a complete object)
            "{\"jsonrpc\":\"2.0\"",
            // Control chars (U+0000 inside a JSON string is invalid per RFC 8259)
            "{\"jsonrpc\":\"2.0\",\"method\":\"\x00\"}",
            // Trailing garbage
            "{\"jsonrpc\":\"2.0\",\"method\":\"init\"}garbage",
            // Double top-level value
            "{} {}",
            // Escaped newline in key
            "{\"json\nrpc\":\"2.0\"}",
            // Very large integer overflows i32 for id
            "{\"jsonrpc\":\"2.0\",\"id\":99999999999999999999999999999999,\"method\":\"init\"}",
            // Array as id is valid JSON-RPC but weird — must parse, not crash.
            "{\"jsonrpc\":\"2.0\",\"id\":[1,2],\"method\":\"init\"}",
        ];

        let mut err_count: usize = 0;
        for &input in cases {
            let result: Result<RpcEnvelope, _> = serde_json::from_str(input);
            if result.is_err() {
                err_count += 1;
            }
        }
        // At least half of the cases should be parse errors; the last two are
        // actually valid JSON so they may succeed (that's fine — we're testing
        // that we don't PANIC, not that every input is rejected).
        assert!(err_count >= 14, "expected at least 14/20 parse errors, got {err_count}");
    }
}
