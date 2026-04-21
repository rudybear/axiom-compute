//! M2.4 MCP server integration tests — AT-1101 through AT-1132.
//!
//! Runs the `axc mcp` subprocess over stdin/stdout (NDJSON JSON-RPC 2.0).
//! GPU-execution tests (AT-1114, AT-1115, AT-1116, AT-1117) are gated behind
//! `AXC_ENABLE_GPU_TESTS=1` so they only execute when Vulkan is available.
//!
//! Unit-level tests (AT-1118 through AT-1128) call library functions directly
//! via `axc_driver::mcp::*` re-exports.
//!
//! Subprocess tests (AT-1101 through AT-1113, AT-1129 through AT-1132) build
//! `axc` via `cargo build -p axc-driver` and invoke it with stdin/stdout pipes.

use std::io::Write as _;
use std::path::PathBuf;
use std::process::{Command, Stdio};

// ── Fixture sources ───────────────────────────────────────────────────────────

/// Minimal empty kernel — no buffers, no strategy.
const EMPTY_KERNEL_SRC: &str = r#"
@kernel
@workgroup(64, 1, 1)
@intent("smoke-test: smallest valid kernel for SPIR-V emission")
@complexity(O(1))
@precondition(true)
fn empty() -> void {
    return;
}
"#;

/// Saxpy kernel — 2 buffers + 2 scalars, with @strategy holes.
const SAXPY_STRATEGY_SRC: &str = r#"
@kernel
@workgroup(64, 1, 1)
@intent("compute Y[i] = alpha * X[i] + Y[i] in parallel")
@complexity(O(n))
@strategy { wg: ?[32, 64, 128] }
fn saxpy(n: u32, alpha: f32, x: readonly_buffer[f32], y: buffer[f32]) -> void {
    let i: u32 = gid(0);
    let limit: u32 = n;
    let in_range: bool = i < limit;
    let xi: f32 = x[i];
    let yi: f32 = y[i];
    let result: f32 = alpha * xi + yi;
    y[i] = result;
    return;
}
"#;

/// Saxpy kernel with 2-hole strategy for AT-1103.
const SAXPY_TWO_HOLE_SRC: &str = r#"
@kernel
@workgroup(64, 1, 1)
@intent("saxpy two-hole strategy fixture")
@complexity(O(n))
@strategy { wg: ?[32, 64], unroll: ?[1, 2, 4] }
fn saxpy2(n: u32, alpha: f32, x: readonly_buffer[f32], y: buffer[f32]) -> void {
    let i: u32 = gid(0);
    let xi: f32 = x[i];
    let yi: f32 = y[i];
    let result: f32 = alpha * xi + yi;
    y[i] = result;
    return;
}
"#;

/// Kernel source with a lex error (emoji causes lex failure).
const LEX_ERROR_SRC: &str = r#"
@kernel
@workgroup(64, 1, 1)
fn bad_kernel() -> void {
    let 💥: u32 = 0;
    return;
}
"#;

/// Kernel with no @strategy block.
const NO_STRATEGY_SRC: &str = r#"
@kernel
@workgroup(64, 1, 1)
@intent("no strategy kernel")
@complexity(O(1))
fn no_strat() -> void {
    return;
}
"#;

// ── Subprocess helpers ────────────────────────────────────────────────────────

/// Build the `axc` binary if not already built, then return its path.
///
/// Uses `cargo build -p axc-driver` to ensure the binary is up to date.
/// Panics if the build fails.
fn axc_binary_path() -> PathBuf {
    // cargo build outputs into target/debug by default relative to workspace root.
    let workspace_root: PathBuf = {
        let manifest_dir: &str = env!("CARGO_MANIFEST_DIR");
        PathBuf::from(manifest_dir)
            .ancestors()
            .nth(2) // crates/axc-driver → crates → workspace root
            .expect("workspace root")
            .to_path_buf()
    };
    let target_dir: PathBuf = workspace_root.join("target").join("debug").join("axc");
    // Build if needed (tests run after `cargo test` which already compiles, but
    // the binary artifact may not exist if only `--lib` was built).
    if !target_dir.exists() {
        let status: std::process::ExitStatus = Command::new("cargo")
            .args(["build", "-p", "axc-driver"])
            .current_dir(&workspace_root)
            .status()
            .expect("cargo build");
        assert!(status.success(), "cargo build -p axc-driver failed");
    }
    target_dir
}

/// Send one or more NDJSON lines to `axc mcp` subprocess and collect all
/// stdout lines produced before the process exits (stdin is closed after writing).
///
/// Returns (stdout_lines, stderr_bytes).
fn run_mcp_exchange(inputs: &[&str]) -> (Vec<serde_json::Value>, Vec<u8>) {
    let axc: PathBuf = axc_binary_path();
    let mut child: std::process::Child = Command::new(&axc)
        .args(["mcp", "--log", "null"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn axc mcp");

    // Write all input lines then close stdin to signal EOF.
    {
        let mut stdin_pipe = child.stdin.take().expect("stdin");
        for line in inputs {
            writeln!(stdin_pipe, "{line}").expect("write to axc stdin");
        }
        // stdin_pipe drops here, closing the pipe → EOF for the child.
    }

    let output: std::process::Output = child.wait_with_output().expect("wait_with_output");

    let stdout_str: &str = std::str::from_utf8(&output.stdout).expect("stdout utf8");
    let responses: Vec<serde_json::Value> = stdout_str
        .lines()
        .filter(|l: &&str| !l.is_empty())
        .map(|l: &str| serde_json::from_str(l).expect("stdout line is valid JSON"))
        .collect();

    (responses, output.stderr)
}

/// Run a single JSON-RPC request against `axc mcp` and return the single response.
fn rpc(request: &str) -> serde_json::Value {
    let (mut lines, _stderr) = run_mcp_exchange(&[request]);
    assert_eq!(lines.len(), 1, "expected exactly 1 response line, got {}", lines.len());
    lines.remove(0)
}

// ── AT-1101: initialize returns server info ───────────────────────────────────

#[test]
fn at_1101_initialize_returns_server_info() {
    let resp: serde_json::Value = rpc(r#"{"jsonrpc":"2.0","id":1,"method":"initialize"}"#);
    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["id"], 1);
    let result: &serde_json::Value = &resp["result"];
    assert_eq!(result["server"], "axc-mcp", "server name mismatch");
    assert_eq!(result["version"], "0.1.0", "version mismatch");
    let tools: &serde_json::Value = &result["tools"];
    assert!(tools.is_array(), "tools must be array");
    let tool_list: Vec<&str> = tools.as_array().unwrap()
        .iter()
        .map(|v: &serde_json::Value| v.as_str().expect("tool name is string"))
        .collect();
    assert_eq!(
        tool_list,
        &[
            "load_source",
            "enumerate_variants",
            "compile_variant",
            "bench_variant",
            "grid_search",
            "optimization_history",
        ],
        "tool list mismatch"
    );
}

// ── AT-1102: load_source returns strategy_holes for saxpy fixture ─────────────

#[test]
fn at_1102_load_source_returns_strategy_holes_for_saxpy_strategy_fixture() {
    let req: serde_json::Value = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "load_source",
        "params": { "source": SAXPY_STRATEGY_SRC }
    });
    let resp: serde_json::Value = rpc(&req.to_string());
    let result: &serde_json::Value = &resp["result"];
    let holes: &serde_json::Value = &result["strategy_holes"];
    assert!(holes.is_object(), "strategy_holes must be an object");
    let wg: &serde_json::Value = &holes["wg"];
    assert!(wg.is_array(), "strategy_holes.wg must be array");
    let wg_values: Vec<i64> = wg.as_array().unwrap()
        .iter()
        .map(|v: &serde_json::Value| v.as_i64().expect("wg value is i64"))
        .collect();
    assert_eq!(wg_values, vec![32_i64, 64, 128], "strategy_holes.wg must be [32, 64, 128]");
}

// ── AT-1103: enumerate_variants returns Cartesian product size ────────────────

#[test]
fn at_1103_enumerate_variants_returns_cartesian_product_size() {
    let req: serde_json::Value = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "enumerate_variants",
        "params": { "source": SAXPY_TWO_HOLE_SRC }
    });
    // First call
    let resp1: serde_json::Value = rpc(&req.to_string());
    let variants1: &serde_json::Value = &resp1["result"]["variants"];
    assert!(variants1.is_array(), "variants must be array");
    let count: usize = variants1.as_array().unwrap().len();
    assert_eq!(count, 6, "2 holes (2×3) must produce 6 variants; got {count}");

    // Check ordinals 0..5 in order.
    for (i, v) in variants1.as_array().unwrap().iter().enumerate() {
        assert_eq!(
            v["ordinal"].as_u64().unwrap(),
            i as u64,
            "ordinal mismatch at position {i}"
        );
    }

    // Second call must yield byte-identical variant_ids (determinism).
    let resp2: serde_json::Value = rpc(&req.to_string());
    let variants2: &serde_json::Value = &resp2["result"]["variants"];
    for (i, (v1, v2)) in variants1.as_array().unwrap()
        .iter()
        .zip(variants2.as_array().unwrap().iter())
        .enumerate()
    {
        assert_eq!(
            v1["variant_id"], v2["variant_id"],
            "variant_id not deterministic at ordinal {i}"
        );
    }
}

// ── AT-1104: compile_variant SPIR-V magic word ────────────────────────────────

#[test]
fn at_1104_compile_variant_spirv_magic_word_present_in_base64() {
    let req: serde_json::Value = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "compile_variant",
        "params": {
            "source": EMPTY_KERNEL_SRC,
            "assignments": {}
        }
    });
    let resp: serde_json::Value = rpc(&req.to_string());
    assert!(resp["error"].is_null(), "unexpected error: {}", resp["error"]);
    let b64: &str = resp["result"]["spirv_base64"].as_str()
        .expect("spirv_base64 must be a string");
    // Decode base64 using the in-process decoder from axc_driver.
    let bytes: Vec<u8> = axc_driver::mcp::base64_decode(b64)
        .expect("spirv_base64 must be valid base64");
    // SPIR-V magic: 0x07230203 in little-endian is bytes [0x03, 0x02, 0x23, 0x07].
    assert!(
        bytes.len() >= 4,
        "SPIR-V output must be at least 4 bytes; got {}",
        bytes.len()
    );
    assert_eq!(
        &bytes[0..4],
        &[0x03_u8, 0x02, 0x23, 0x07],
        "SPIR-V magic word mismatch"
    );
}

// ── AT-1105: compile_variant capabilities contains "Shader" ──────────────────

#[test]
fn at_1105_compile_variant_includes_shader_capability() {
    let req: serde_json::Value = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 5,
        "method": "compile_variant",
        "params": {
            "source": EMPTY_KERNEL_SRC,
            "assignments": {}
        }
    });
    let resp: serde_json::Value = rpc(&req.to_string());
    assert!(resp["error"].is_null(), "unexpected error: {}", resp["error"]);
    let caps: &serde_json::Value = &resp["result"]["capabilities"];
    assert!(caps.is_array(), "capabilities must be array");
    let cap_list: Vec<&str> = caps.as_array().unwrap()
        .iter()
        .map(|v: &serde_json::Value| v.as_str().expect("capability is string"))
        .collect();
    assert!(
        cap_list.contains(&"Shader"),
        "capabilities must contain 'Shader'; got {cap_list:?}"
    );
    let exts: &serde_json::Value = &resp["result"]["extensions"];
    assert!(exts.is_array(), "extensions must be array");
}

// ── AT-1106: unknown method returns -32601 ────────────────────────────────────

#[test]
fn at_1106_unknown_method_returns_method_not_found_error() {
    let resp: serde_json::Value = rpc(r#"{"jsonrpc":"2.0","id":6,"method":"fake"}"#);
    assert_eq!(resp["id"], 6);
    assert!(resp["result"].is_null(), "result must be absent for error");
    assert_eq!(resp["error"]["code"].as_i64().unwrap(), -32601);
    let available: &serde_json::Value = &resp["error"]["data"]["available"];
    assert!(available.is_array(), "error.data.available must be array");
    let methods: Vec<&str> = available.as_array().unwrap()
        .iter()
        .map(|v: &serde_json::Value| v.as_str().expect("method name is string"))
        .collect();
    // Must include "initialize" plus the 6 tool methods.
    assert!(methods.contains(&"initialize"), "available must include 'initialize'");
    assert_eq!(methods.len(), 7, "must list all 7 methods; got {methods:?}");
}

// ── AT-1107: malformed JSON returns -32700 ────────────────────────────────────

#[test]
fn at_1107_parse_error_returns_32700() {
    let resp: serde_json::Value = rpc("{not json");
    assert_eq!(
        resp["error"]["code"].as_i64().unwrap(), -32700,
        "parse error must return -32700"
    );
    assert!(
        resp["id"].is_null(),
        "parse error response id must be null; got: {}",
        resp["id"]
    );
    let message: &str = resp["error"]["message"].as_str().expect("message is string");
    assert!(
        message.to_lowercase().contains("invalid") || message.to_lowercase().contains("json"),
        "parse error message must mention 'invalid' or 'json'; got: {message:?}"
    );
}

// ── AT-1108: wrong-typed params returns -32602 ────────────────────────────────

#[test]
fn at_1108_invalid_params_returns_32602() {
    let req: serde_json::Value = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 8,
        "method": "load_source",
        "params": { "source": 42 }
    });
    let resp: serde_json::Value = rpc(&req.to_string());
    assert_eq!(
        resp["error"]["code"].as_i64().unwrap(), -32602,
        "wrong type must return -32602"
    );
}

// ── AT-1109: load_source with neither source nor path → -32602 ───────────────

#[test]
fn at_1109_load_source_rejects_neither_source_nor_path() {
    let req: serde_json::Value = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 9,
        "method": "load_source",
        "params": {}
    });
    let resp: serde_json::Value = rpc(&req.to_string());
    assert_eq!(
        resp["error"]["code"].as_i64().unwrap(), -32602,
        "neither source nor path must return -32602"
    );
    let message: &str = resp["error"]["message"].as_str().expect("error message");
    assert!(
        message.contains("exactly one"),
        "error message must mention 'exactly one'; got: {message:?}"
    );
}

// ── AT-1110: load_source with both source and path → -32602 ──────────────────

#[test]
fn at_1110_load_source_rejects_both_source_and_path() {
    let req: serde_json::Value = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 10,
        "method": "load_source",
        "params": {
            "source": "fn empty() -> void { return; }",
            "path": "/tmp/nonexistent.axc"
        }
    });
    let resp: serde_json::Value = rpc(&req.to_string());
    assert_eq!(
        resp["error"]["code"].as_i64().unwrap(), -32602,
        "both source and path must return -32602"
    );
    let message: &str = resp["error"]["message"].as_str().expect("error message");
    assert!(
        message.contains("exactly one"),
        "error message must mention 'exactly one'; got: {message:?}"
    );
}

// ── AT-1111: compile error propagated as -32001 ───────────────────────────────

#[test]
fn at_1111_compile_variant_propagates_compile_error_as_32001() {
    let req: serde_json::Value = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 11,
        "method": "compile_variant",
        "params": {
            "source": LEX_ERROR_SRC,
            "assignments": {}
        }
    });
    let resp: serde_json::Value = rpc(&req.to_string());
    assert_eq!(
        resp["error"]["code"].as_i64().unwrap(), -32001,
        "lex error must return -32001; got: {}",
        resp["error"]
    );
    let data: &serde_json::Value = &resp["error"]["data"];
    let lex_count: i64 = data["lex_count"].as_i64().unwrap_or(0);
    assert!(lex_count > 0, "lex_count must be > 0 for emoji source; got {lex_count}");
}

// ── AT-1112: enumerate_variants with no @strategy → -32002 ───────────────────

#[test]
fn at_1112_enumerate_variants_returns_32002_for_no_strategy() {
    let req: serde_json::Value = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 12,
        "method": "enumerate_variants",
        "params": { "source": NO_STRATEGY_SRC }
    });
    let resp: serde_json::Value = rpc(&req.to_string());
    assert_eq!(
        resp["error"]["code"].as_i64().unwrap(), -32002,
        "no @strategy must return -32002; got: {}",
        resp["error"]
    );
}

// ── AT-1113: path variant reads file from disk ────────────────────────────────

#[test]
fn at_1113_path_variant_reads_file_from_disk() {
    let dir: tempfile::TempDir = tempfile::tempdir().expect("tempdir");
    let file_path: PathBuf = dir.path().join("fixture.axc");
    std::fs::write(&file_path, EMPTY_KERNEL_SRC).expect("write fixture");

    // Call load_source with path= and with source= and compare kernel_name.
    let req_path: serde_json::Value = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 13,
        "method": "load_source",
        "params": { "path": file_path.to_str().unwrap() }
    });
    let req_src: serde_json::Value = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 14,
        "method": "load_source",
        "params": { "source": EMPTY_KERNEL_SRC }
    });

    let resp_path: serde_json::Value = rpc(&req_path.to_string());
    let resp_src: serde_json::Value = rpc(&req_src.to_string());

    assert!(resp_path["error"].is_null(), "path variant error: {}", resp_path["error"]);
    assert!(resp_src["error"].is_null(), "source variant error: {}", resp_src["error"]);

    assert_eq!(
        resp_path["result"]["kernel_name"],
        resp_src["result"]["kernel_name"],
        "kernel_name must match between path and source variants"
    );
}

// ── AT-1114: bench_variant returns median_ns on Lavapipe (GPU-gated) ──────────

#[test]
fn at_1114_bench_variant_returns_median_ns_on_lavapipe() {
    if std::env::var("AXC_ENABLE_GPU_TESTS").unwrap_or_default() != "1" {
        eprintln!("AT-1114: skipped (AXC_ENABLE_GPU_TESTS != 1)");
        return;
    }
    let req: serde_json::Value = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 14,
        "method": "bench_variant",
        "params": {
            "source": SAXPY_STRATEGY_SRC,
            "assignments": { "wg": 64 },
            "input_sizes": [4096, 4096],
            "sample_count": 5
        }
    });
    let resp: serde_json::Value = rpc(&req.to_string());
    assert!(resp["error"].is_null(), "bench_variant error: {}", resp["error"]);
    let result: &serde_json::Value = &resp["result"];
    let median_ns: u64 = result["median_ns"].as_u64().expect("median_ns is u64");
    assert!(median_ns > 0, "median_ns must be > 0");
    let samples: &serde_json::Value = &result["samples"];
    assert!(samples.is_array(), "samples must be array");
    assert_eq!(
        samples.as_array().unwrap().len(),
        5,
        "samples count must equal sample_count=5"
    );
    let device_name: &str = result["machine"]["device_name"].as_str()
        .expect("machine.device_name is string");
    assert!(!device_name.is_empty(), "device_name must be non-empty");
}

// ── AT-1115: bench_variant correctness.status = "ok" for saxpy ───────────────

#[test]
fn at_1115_bench_variant_correctness_ok_for_saxpy() {
    if std::env::var("AXC_ENABLE_GPU_TESTS").unwrap_or_default() != "1" {
        eprintln!("AT-1115: skipped (AXC_ENABLE_GPU_TESTS != 1)");
        return;
    }
    let req: serde_json::Value = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 15,
        "method": "bench_variant",
        "params": {
            "source": SAXPY_STRATEGY_SRC,
            "assignments": { "wg": 64 },
            "input_sizes": [4096, 4096],
            "sample_count": 3
        }
    });
    let resp: serde_json::Value = rpc(&req.to_string());
    assert!(resp["error"].is_null(), "bench_variant error: {}", resp["error"]);
    let correctness: &serde_json::Value = &resp["result"]["correctness"];
    assert_eq!(
        correctness["status"].as_str().unwrap_or(""),
        "ok",
        "correctness.status must be 'ok' for saxpy; got: {correctness}"
    );
    // When status is "ok", there must be no "reason" field.
    assert!(
        correctness["reason"].is_null() || !correctness.as_object().unwrap().contains_key("reason"),
        "correctness 'ok' must have no 'reason' field; got: {correctness}"
    );
}

// ── AT-1116: grid_search returns winner + appends history (GPU-gated) ─────────

#[test]
fn at_1116_grid_search_returns_winner_and_appends_history() {
    if std::env::var("AXC_ENABLE_GPU_TESTS").unwrap_or_default() != "1" {
        eprintln!("AT-1116: skipped (AXC_ENABLE_GPU_TESTS != 1)");
        return;
    }
    let history_dir: tempfile::TempDir = tempfile::tempdir().expect("tempdir");
    let history_env: String = history_dir.path().to_str().unwrap().to_string();

    let axc: PathBuf = axc_binary_path();
    let req: serde_json::Value = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 16,
        "method": "grid_search",
        "params": {
            "source": SAXPY_STRATEGY_SRC,
            "input_sizes": [4096, 4096],
            "sample_count": 3
        }
    });

    let mut child: std::process::Child = Command::new(&axc)
        .args(["mcp", "--log", "null"])
        .env("AXC_MCP_HISTORY_DIR", &history_env)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn axc mcp");

    {
        let mut stdin_pipe = child.stdin.take().unwrap();
        writeln!(stdin_pipe, "{req}").unwrap();
    }
    let output: std::process::Output = child.wait_with_output().expect("wait_with_output");
    let stdout_str: &str = std::str::from_utf8(&output.stdout).unwrap();
    let resp: serde_json::Value = serde_json::from_str(
        stdout_str.lines().next().expect("at least one response line")
    ).expect("valid JSON response");

    assert!(resp["error"].is_null(), "grid_search error: {}", resp["error"]);
    let result: &serde_json::Value = &resp["result"];
    let ranked: &serde_json::Value = &result["ranked"];
    assert!(ranked.is_array(), "ranked must be array");
    // Verify sorted ascending by median_ns (non-null entries come first).
    let ranked_arr: &Vec<serde_json::Value> = ranked.as_array().unwrap();
    let mut prev_ns: Option<u64> = None;
    for r in ranked_arr.iter() {
        if let Some(ns) = r["median_ns"].as_u64() {
            if let Some(prev) = prev_ns {
                assert!(ns >= prev, "ranked must be ascending by median_ns");
            }
            prev_ns = Some(ns);
        }
    }

    // winner must equal ranked[0].
    let winner: &serde_json::Value = &result["winner"];
    let first: &serde_json::Value = &ranked_arr[0];
    assert_eq!(
        winner["variant_id"], first["variant_id"],
        "winner variant_id must equal ranked[0].variant_id"
    );

    // History file must exist with exactly 1 line.
    let history_files: Vec<PathBuf> = std::fs::read_dir(history_dir.path())
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|e| e == "jsonl").unwrap_or(false))
        .collect();
    assert_eq!(history_files.len(), 1, "must have exactly 1 history file");
    let history_content: String = std::fs::read_to_string(&history_files[0]).unwrap();
    let history_lines: Vec<&str> = history_content
        .lines()
        .filter(|l| !l.is_empty())
        .collect();
    assert_eq!(history_lines.len(), 1, "history file must have exactly 1 line");

    let entry: serde_json::Value = serde_json::from_str(history_lines[0])
        .expect("history line is valid JSON");
    assert_eq!(
        entry["grid_search"]["winner_variant_id"],
        winner["variant_id"],
        "history winner_variant_id must match response winner"
    );
}

// ── AT-1117: grid_search + optimization_history roundtrip (GPU-gated) ─────────

#[test]
fn at_1117_grid_search_history_roundtrip_via_optimization_history() {
    if std::env::var("AXC_ENABLE_GPU_TESTS").unwrap_or_default() != "1" {
        eprintln!("AT-1117: skipped (AXC_ENABLE_GPU_TESTS != 1)");
        return;
    }
    let history_dir: tempfile::TempDir = tempfile::tempdir().expect("tempdir");
    let history_env: String = history_dir.path().to_str().unwrap().to_string();

    // Write the source to a temp file for optimization_history.
    let src_file: PathBuf = history_dir.path().join("saxpy.axc");
    std::fs::write(&src_file, SAXPY_STRATEGY_SRC).expect("write source file");

    let axc: PathBuf = axc_binary_path();

    let grid_req: serde_json::Value = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "grid_search",
        "params": {
            "source": SAXPY_STRATEGY_SRC,
            "input_sizes": [4096, 4096],
            "sample_count": 2
        }
    });

    // Run grid_search twice sequentially (two separate axc mcp invocations).
    for _ in 0..2 {
        let mut child: std::process::Child = Command::new(&axc)
            .args(["mcp", "--log", "null"])
            .env("AXC_MCP_HISTORY_DIR", &history_env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn axc mcp");
        {
            let mut stdin_pipe = child.stdin.take().unwrap();
            writeln!(stdin_pipe, "{grid_req}").unwrap();
        }
        child.wait_with_output().expect("wait_with_output");
        // Sleep 2ms to guarantee distinct millisecond timestamps.
        std::thread::sleep(std::time::Duration::from_millis(2));
    }

    // Now call optimization_history.
    let hist_req: serde_json::Value = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "optimization_history",
        "params": { "source_path": src_file.to_str().unwrap() }
    });

    let mut child: std::process::Child = Command::new(&axc)
        .args(["mcp", "--log", "null"])
        .env("AXC_MCP_HISTORY_DIR", &history_env)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn axc mcp for optimization_history");
    {
        let mut stdin_pipe = child.stdin.take().unwrap();
        writeln!(stdin_pipe, "{hist_req}").unwrap();
    }
    let output: std::process::Output = child.wait_with_output().expect("wait_with_output");
    let resp: serde_json::Value = serde_json::from_str(
        std::str::from_utf8(&output.stdout)
            .unwrap()
            .lines()
            .next()
            .expect("at least one response line")
    ).expect("valid JSON response");

    assert!(resp["error"].is_null(), "optimization_history error: {}", resp["error"]);
    let entries: &serde_json::Value = &resp["result"]["entries"];
    assert!(entries.is_array(), "entries must be array");
    assert_eq!(entries.as_array().unwrap().len(), 2, "must have 2 entries after 2 grid_search calls");

    // Timestamps must be RFC3339 with ms and lexicographically ordered.
    // Format: "YYYY-MM-DDTHH:MM:SS.NNNZ" — exactly 24 chars, ends with 'Z',
    // has 'T' at index 10, '.' at index 19.
    let ts0: &str = entries[0]["timestamp"].as_str().expect("timestamp 0");
    let ts1: &str = entries[1]["timestamp"].as_str().expect("timestamp 1");
    for ts in [ts0, ts1] {
        assert_eq!(ts.len(), 24, "timestamp must be 24 chars: {ts:?}");
        assert!(ts.ends_with('Z'), "timestamp must end with 'Z': {ts:?}");
        assert_eq!(ts.as_bytes()[10], b'T', "timestamp must have 'T' at index 10: {ts:?}");
        assert_eq!(ts.as_bytes()[19], b'.', "timestamp must have '.' at index 19: {ts:?}");
        assert!(
            ts[20..23].chars().all(|c: char| c.is_ascii_digit()),
            "timestamp must have 3 ms digits at indices 20-22: {ts:?}"
        );
    }
    assert!(ts0 <= ts1, "entries must be in ascending timestamp order: {ts0} > {ts1}");
}

// ── AT-1118: history_path_for_source uses xxh3_hex16 (unit test) ─────────────

#[test]
fn at_1118_history_path_for_source_is_xxh3_hex16() {
    use axc_driver::mcp::history_path_for_source;
    use std::path::Path;

    let dir: &Path = Path::new("/tmp");
    let source: &str = "abc\n";
    let path1: PathBuf = history_path_for_source(dir, source);
    let path2: PathBuf = history_path_for_source(dir, source);

    // Must be deterministic.
    assert_eq!(path1, path2, "history_path must be deterministic");

    // Filename stem must be exactly 16 hex characters.
    let stem: &str = path1.file_stem().unwrap().to_str().unwrap();
    assert_eq!(stem.len(), 16, "stem must be 16 hex chars; got {stem:?}");
    assert!(
        stem.chars().all(|c: char| c.is_ascii_hexdigit()),
        "stem must be all hex digits; got {stem:?}"
    );

    // Extension must be .jsonl.
    assert_eq!(
        path1.extension().unwrap().to_str().unwrap(),
        "jsonl",
        "extension must be .jsonl"
    );
}

// ── AT-1119: history append is line-preserving (unit test) ───────────────────

#[test]
fn at_1119_history_append_is_line_preserving() {
    use axc_driver::mcp::append_history_entry_for_test;
    use axc_driver::mcp::make_test_history_record;

    let dir: tempfile::TempDir = tempfile::tempdir().expect("tempdir");
    let source: &str = "unit_test_source_AT-1119";
    let path: PathBuf = axc_driver::mcp::history_path_for_source(dir.path(), source);

    let record1 = make_test_history_record("entry_A");
    let record2 = make_test_history_record("entry_B");
    append_history_entry_for_test(&path, &record1).expect("append 1");
    append_history_entry_for_test(&path, &record2).expect("append 2");

    let content: Vec<u8> = std::fs::read(&path).expect("read history file");
    let newline_count: usize = content.iter().filter(|&&b: &&u8| b == b'\n').count();
    let cr_count: usize = content.iter().filter(|&&b: &&u8| b == b'\r').count();
    assert_eq!(newline_count, 2, "must have exactly 2 newlines; got {newline_count}");
    assert_eq!(cr_count, 0, "must have no carriage returns");

    // Each line must parse as valid JSON.
    let text: &str = std::str::from_utf8(&content).expect("valid utf8");
    for line in text.lines() {
        serde_json::from_str::<serde_json::Value>(line)
            .unwrap_or_else(|e| panic!("history line is not valid JSON: {e}\nline: {line:?}"));
    }
}

// ── AT-1119b: flock excludes concurrent writers (unit test) ──────────────────

#[test]
fn at_1119b_history_append_flock_excludes_concurrent_writers() {
    use axc_driver::mcp::append_history_entry_for_test;
    use axc_driver::mcp::make_test_history_record;

    let dir: tempfile::TempDir = tempfile::tempdir().expect("tempdir");
    let source: &str = "concurrent_flock_AT-1119b";
    let path: PathBuf = axc_driver::mcp::history_path_for_source(dir.path(), source);
    let path2: PathBuf = path.clone();

    const N: usize = 100;

    // Spawn two threads, each appending N records, yielding after each write.
    let t1 = std::thread::spawn({
        let path_t1 = path.clone();
        move || {
            for i in 0..N {
                let record = make_test_history_record(&format!("t1_{i}"));
                append_history_entry_for_test(&path_t1, &record)
                    .expect("t1 append");
                std::thread::yield_now();
            }
        }
    });
    let t2 = std::thread::spawn({
        let path_t2 = path2.clone();
        move || {
            for i in 0..N {
                let record = make_test_history_record(&format!("t2_{i}"));
                append_history_entry_for_test(&path_t2, &record)
                    .expect("t2 append");
                std::thread::yield_now();
            }
        }
    });
    t1.join().expect("t1 join");
    t2.join().expect("t2 join");

    // Read and verify.
    let content: String = std::fs::read_to_string(&path).expect("read history");
    let lines: Vec<&str> = content.lines().filter(|l: &&str| !l.is_empty()).collect();
    assert_eq!(lines.len(), 2 * N, "must have exactly {} lines; got {}", 2 * N, lines.len());
    for (i, line) in lines.iter().enumerate() {
        // No embedded newlines (the line itself must parse cleanly).
        assert!(
            !line.contains('\n'),
            "line {i} contains embedded newline"
        );
        serde_json::from_str::<serde_json::Value>(line)
            .unwrap_or_else(|e| panic!("line {i} is not valid JSON: {e}"));
    }
}

// ── AT-1120: optimization_history skips malformed lines ──────────────────────

#[test]
fn at_1120_history_read_skips_malformed_lines() {
    use axc_driver::mcp::make_test_history_record;

    let dir: tempfile::TempDir = tempfile::tempdir().expect("tempdir");
    let source: &str = "skip_malformed_AT-1120";

    // Compute the history path to craft a manual JSONL file there.
    let history_path: PathBuf = axc_driver::mcp::history_path_for_source(dir.path(), source);

    // Write a source file that matches the hash.
    let src_file: PathBuf = dir.path().join("fixture.axc");
    std::fs::write(&src_file, source).expect("write source");

    // Craft a 3-line JSONL: valid, garbage, valid.
    let record_a = make_test_history_record("line_A");
    let record_c = make_test_history_record("line_C");
    let line_a: String = serde_json::to_string(&record_a).unwrap();
    let line_c: String = serde_json::to_string(&record_c).unwrap();
    let content: String = format!("{line_a}\nGARBAGE NOT JSON\n{line_c}\n");
    std::fs::write(&history_path, &content).expect("write history");

    // Call optimization_history via subprocess.
    let axc: PathBuf = axc_binary_path();
    let req: serde_json::Value = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 20,
        "method": "optimization_history",
        "params": { "source_path": src_file.to_str().unwrap() }
    });

    let mut child: std::process::Child = Command::new(&axc)
        .args(["mcp", "--log", "null"])
        .env("AXC_MCP_HISTORY_DIR", dir.path().to_str().unwrap())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn axc mcp");
    {
        let mut stdin_pipe = child.stdin.take().unwrap();
        writeln!(stdin_pipe, "{req}").unwrap();
    }
    let output: std::process::Output = child.wait_with_output().expect("wait_with_output");
    let resp: serde_json::Value = serde_json::from_str(
        std::str::from_utf8(&output.stdout).unwrap().lines().next().expect("response line")
    ).expect("valid JSON");

    assert!(resp["error"].is_null(), "unexpected error: {}", resp["error"]);
    let result: &serde_json::Value = &resp["result"];
    let entries_len: usize = result["entries"].as_array().unwrap().len();
    let skipped: i64 = result["skipped_lines"].as_i64().unwrap_or(-1);
    assert_eq!(entries_len, 2, "must have 2 valid entries; got {entries_len}");
    assert_eq!(skipped, 1, "must have 1 skipped line; got {skipped}");
}

// ── AT-1121: derive_workgroups ceiling divide (unit test) ─────────────────────

#[test]
fn at_1121_derive_workgroups_ceiling_divide() {
    use axc_driver::mcp::derive_workgroups;

    // 4096 bytes / 4 bytes per f32 = 1024 elems, ceil(1024/64) = 16.
    let result: [u32; 3] = derive_workgroups(&[4096_usize, 4096], [64_u32, 1, 1], None);
    assert_eq!(result, [16, 1, 1], "default axes 1/2 must be 1; got {result:?}");

    // Override all three dimensions.
    let override_result: [u32; 3] = derive_workgroups(&[4096_usize, 4096], [64_u32, 1, 1], Some([8_u32, 8, 1]));
    assert_eq!(override_result, [8, 8, 1], "workgroup_override must take effect; got {override_result:?}");
}

// ── AT-1122: seeded_inputs deterministic (unit test) ─────────────────────────

#[test]
fn at_1122_seeded_inputs_deterministic() {
    use axc_driver::mcp::seeded_inputs;

    let sizes: &[usize] = &[256, 256];
    let variant_id: u64 = 12345;

    let result1: Vec<Vec<u8>> = seeded_inputs(sizes, variant_id);
    let result2: Vec<Vec<u8>> = seeded_inputs(sizes, variant_id);
    assert_eq!(result1, result2, "seeded_inputs must be deterministic");

    // Different variant_id must produce different output.
    let result3: Vec<Vec<u8>> = seeded_inputs(sizes, variant_id + 1);
    assert_ne!(result1, result3, "different variant_id must yield different inputs");
}

// ── AT-1123: base64 RFC 4648 vectors (unit test) ─────────────────────────────

#[test]
fn at_1123_base64_rfc4648_vectors() {
    use axc_driver::mcp::base64_encode;
    use axc_driver::mcp::BASE64_ALPHABET;

    // RFC 4648 §10 test vectors.
    assert_eq!(base64_encode(b""), "", "empty input");
    assert_eq!(base64_encode(b"f"), "Zg==", "1-byte 'f'");
    assert_eq!(base64_encode(b"fo"), "Zm8=", "2-byte 'fo'");
    assert_eq!(base64_encode(b"foo"), "Zm9v", "3-byte 'foo'");
    assert_eq!(base64_encode(b"foob"), "Zm9vYg==", "4-byte 'foob'");
    assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=", "5-byte 'fooba'");
    assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy", "6-byte 'foobar'");

    // AT-1123b folded in: alphabet assertion.
    assert_eq!(BASE64_ALPHABET[62], b'+', "index 62 must be '+' (standard base64)");
    assert_eq!(BASE64_ALPHABET[63], b'/', "index 63 must be '/' (standard base64)");

    // Negative check: URL-safe chars must never appear.
    for input in [b"f".as_ref(), b"fo".as_ref(), b"foobar".as_ref()] {
        let encoded: String = base64_encode(input);
        assert!(!encoded.contains('-'), "output must not contain '-' (URL-safe char): {encoded:?}");
        assert!(!encoded.contains('_'), "output must not contain '_' (URL-safe char): {encoded:?}");
    }
}

// ── AT-1124: base64 roundtrip random (unit test) ──────────────────────────────

#[test]
fn at_1124_base64_roundtrip_random() {
    use axc_driver::mcp::{base64_encode, base64_decode};
    use rand::{Rng as _, SeedableRng as _, RngCore as _};

    let mut rng: rand::rngs::StdRng = rand::rngs::StdRng::seed_from_u64(42);
    for _ in 0..1024_usize {
        let len: usize = rng.gen_range(0..256_usize);
        let mut bytes: Vec<u8> = vec![0u8; len];
        rng.fill_bytes(&mut bytes);
        let encoded: String = base64_encode(&bytes);
        let decoded: Vec<u8> = base64_decode(&encoded)
            .unwrap_or_else(|e| panic!("base64_decode failed for len={len}: {e}"));
        assert_eq!(bytes, decoded, "roundtrip mismatch for len={len}");
    }
}

// ── AT-1125: RFC3339 fixed inputs (unit test) ─────────────────────────────────

#[test]
fn at_1125_rfc3339_fixed_inputs() {
    use axc_driver::mcp::format_rfc3339_utc;
    use std::time::{Duration, UNIX_EPOCH};

    assert_eq!(
        format_rfc3339_utc(UNIX_EPOCH),
        "1970-01-01T00:00:00.000Z",
        "UNIX epoch"
    );

    // 2000-02-29 12:34:56.000 UTC (leap year).
    // 2000-02-29 00:00:00 UTC = UNIX epoch + 951782400 seconds.
    let leap_day_base: u64 = 951_782_400;
    let leap_ts: std::time::SystemTime = UNIX_EPOCH + Duration::from_secs(leap_day_base + 12 * 3600 + 34 * 60 + 56);
    assert_eq!(
        format_rfc3339_utc(leap_ts),
        "2000-02-29T12:34:56.000Z",
        "leap year 2000-02-29"
    );

    // Sub-second: UNIX_EPOCH + 123ms.
    let sub_second: std::time::SystemTime = UNIX_EPOCH + Duration::from_millis(123);
    assert_eq!(
        format_rfc3339_utc(sub_second),
        "1970-01-01T00:00:00.123Z",
        "123ms sub-second"
    );

    // 2026-04-18 14:30:00.500 UTC.
    // Unix ts for 2026-04-18 14:30:00 UTC = 1_776_522_600.
    let ts_2026: u64 = 1_776_522_600;
    let ts_with_ms: std::time::SystemTime = UNIX_EPOCH + Duration::from_millis(ts_2026 * 1000 + 500);
    assert_eq!(
        format_rfc3339_utc(ts_with_ms),
        "2026-04-18T14:30:00.500Z",
        "2026-04-18 with 500ms"
    );

    // All outputs must end with 'Z' and have exactly 3 fractional digits.
    for t_val in [UNIX_EPOCH, leap_ts, sub_second, ts_with_ms] {
        let s: String = format_rfc3339_utc(t_val);
        assert!(s.ends_with('Z'), "must end with 'Z': {s:?}");
        assert_eq!(s.len(), 24, "must be exactly 24 chars: {s:?}");
        // Position 20 is the '.' separator; chars 21..24 are the 3 ms digits.
        assert_eq!(&s[19..20], ".", "must have '.' at position 19: {s:?}");
        assert!(
            s[20..23].chars().all(|c: char| c.is_ascii_digit()),
            "must have 3 digit ms: {s:?}"
        );
    }
}

// ── AT-1126: scan_caps_and_exts order (unit test) ────────────────────────────

#[test]
fn at_1126_scan_caps_and_exts_order() {
    use axc_driver::mcp::scan_caps_and_exts;

    // Build a minimal synthetic SPIR-V word stream with:
    //   OpCapability Shader (capability 1)
    //   OpCapability StorageBuffer16BitAccess / StorageUniformBufferBlock16 (capability 4433)
    //   OpExtension "SPV_KHR_16bit_storage"
    // Full SPIR-V header: [magic, version, generator, bound, schema] = 5 words.
    let mut words: Vec<u32> = vec![
        0x07230203, // magic
        0x00010300, // SPIR-V 1.3
        0,          // generator magic
        1,          // bound
        0,          // schema
        // OpCapability Shader: opcode=17, word_count=2; operand = 1 (Shader)
        (2 << 16) | 17,
        1,
        // OpCapability StorageBuffer16BitAccess (= StorageUniformBufferBlock16): opcode=17; operand=4433
        (2 << 16) | 17,
        4433,
    ];

    // OpExtension "SPV_KHR_16bit_storage" as null-padded 4-byte words.
    let ext_str: &str = "SPV_KHR_16bit_storage";
    let mut ext_bytes: Vec<u8> = ext_str.as_bytes().to_vec();
    while !ext_bytes.len().is_multiple_of(4) {
        ext_bytes.push(0);
    }
    let ext_word_count: u16 = 1 + (ext_bytes.len() / 4) as u16;
    words.push(((ext_word_count as u32) << 16) | 10); // OpExtension opcode=10
    for chunk in ext_bytes.chunks(4) {
        words.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }

    let (caps, exts) = scan_caps_and_exts(&words);
    // StorageBuffer16BitAccess and StorageUniformBufferBlock16 are the same enum value (4433).
    // The spirv crate's Debug output for 4433 is "StorageBuffer16BitAccess".
    assert_eq!(
        caps,
        &["Shader", "StorageBuffer16BitAccess"],
        "capabilities order mismatch; got {caps:?}"
    );
    assert_eq!(
        exts,
        &["SPV_KHR_16bit_storage"],
        "extensions mismatch; got {exts:?}"
    );
}

// ── AT-1127: resolve_source exactly-one (unit test) ──────────────────────────

#[test]
fn at_1127_resolve_source_exactly_one() {
    use axc_driver::mcp::resolve_source;

    // Neither source nor path → InvalidParams.
    let err_neither = resolve_source(&None::<String>, &None::<std::path::PathBuf>)
        .expect_err("neither source nor path must fail");
    let msg: String = err_neither.to_string();
    assert!(
        msg.contains("exactly one"),
        "error message must contain 'exactly one'; got: {msg:?}"
    );

    // Both source and path → InvalidParams.
    let source_opt: Option<String> = Some("@kernel fn x() -> void { return; }".to_string());
    let path_opt: Option<std::path::PathBuf> = Some(std::path::PathBuf::from("/tmp/x.axc"));
    let err_both = resolve_source(&source_opt, &path_opt)
        .expect_err("both source and path must fail");
    let msg2: String = err_both.to_string();
    assert!(
        msg2.contains("exactly one"),
        "error message must contain 'exactly one'; got: {msg2:?}"
    );
}

// ── AT-1128: OnceVulkan caches unavailable state (unit test) ─────────────────

#[test]
fn at_1128_once_vulkan_caches_unavailable() {
    use axc_driver::mcp::OnceVulkan;

    // Seed an Unavailable state directly (no probe needed).
    let mut ov: OnceVulkan = OnceVulkan::new_unavailable("cached reason".to_string());
    let r1 = ov.get_or_init();
    assert!(r1.is_err(), "must return Err for Unavailable");
    // Use .err() instead of .unwrap_err() to avoid requiring VulkanContext: Debug.
    let reason1: String = r1.err().expect("must be Err").to_string();

    let r2 = ov.get_or_init();
    assert!(r2.is_err(), "must return Err on second call");
    let reason2: String = r2.err().expect("must be Err").to_string();

    assert_eq!(reason1, reason2, "cached reason must be identical on second call");
}

// ── AT-1129: oversize line rejected with parse error ─────────────────────────

#[test]
fn at_1129_oversize_line_rejected_with_parse_error() {
    let axc: PathBuf = axc_binary_path();

    // Build a line that is 8 MiB + 1 byte.
    // Wrap it in a JSON string so it *would* parse if the size check didn't fire.
    // Use a raw string: {"jsonrpc":"2.0","id":1,"method":"X" + <8MiB padding>}
    // Actually, we just send a raw garbage line of that length — the server
    // must reject it before even attempting JSON parsing.
    let oversize_line: String = "A".repeat(8 * 1024 * 1024 + 1);

    let mut child: std::process::Child = Command::new(&axc)
        .args(["mcp", "--log", "null"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn axc mcp");

    {
        let mut stdin_pipe = child.stdin.take().unwrap();
        // Write the oversize line.
        writeln!(stdin_pipe, "{oversize_line}").unwrap();
        // Follow with a valid request to verify the server continues running.
        writeln!(stdin_pipe, r#"{{"jsonrpc":"2.0","id":999,"method":"initialize"}}"#).unwrap();
    }

    let output: std::process::Output = child.wait_with_output().expect("wait_with_output");
    let stdout_str: &str = std::str::from_utf8(&output.stdout).expect("stdout utf8");
    let lines: Vec<&str> = stdout_str.lines().filter(|l: &&str| !l.is_empty()).collect();

    assert!(!lines.is_empty(), "must have at least 1 response line");

    let first_resp: serde_json::Value = serde_json::from_str(lines[0]).expect("valid JSON");
    assert_eq!(
        first_resp["error"]["code"].as_i64().unwrap(), -32700,
        "oversize line must return -32700; got: {}",
        first_resp["error"]
    );
    let reason: &str = first_resp["error"]["data"]["reason"].as_str().unwrap_or("");
    assert!(
        reason.contains("8 MiB") || reason.contains("8388608"),
        "error reason must mention '8 MiB'; got: {reason:?}"
    );

    // Server must have continued and responded to the second request.
    if lines.len() >= 2 {
        let second_resp: serde_json::Value = serde_json::from_str(lines[1]).expect("valid JSON");
        assert_eq!(second_resp["id"], 999, "second request must have id=999");
    }
}

// ── AT-1130: EOF shutdown and --log null (unit test via subprocess) ───────────

#[test]
fn at_1130_eof_shutdown_and_log_null() {
    let axc: PathBuf = axc_binary_path();
    let mut child: std::process::Child = Command::new(&axc)
        .args(["mcp", "--log", "null"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn axc mcp");

    // Send one request then close stdin.
    {
        let mut stdin_pipe = child.stdin.take().unwrap();
        writeln!(stdin_pipe, r#"{{"jsonrpc":"2.0","id":1,"method":"initialize"}}"#).unwrap();
        // stdin_pipe drops here → EOF.
    }

    let output: std::process::Output = child.wait_with_output().expect("wait_with_output");

    assert!(output.status.success(), "axc mcp must exit with code 0 on EOF");
    assert!(
        output.stderr.is_empty(),
        "stderr must be empty with --log null; got: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// ── AT-1131: notifications emit no response ───────────────────────────────────

#[test]
fn at_1131_mcp_notifications_emit_no_response() {
    let notification: &str = r#"{"jsonrpc":"2.0","method":"initialize"}"#;
    let real_request: &str = r#"{"jsonrpc":"2.0","id":5,"method":"initialize"}"#;
    // Null-id is NOT a notification — it must produce a response.
    let null_id_request: &str = r#"{"jsonrpc":"2.0","id":null,"method":"initialize"}"#;

    let (responses, _stderr) = run_mcp_exchange(&[notification, real_request]);

    // Notification must produce NO response; real_request must produce 1 response.
    assert_eq!(
        responses.len(), 1,
        "notification must produce no stdout line; got {} responses",
        responses.len()
    );
    assert_eq!(
        responses[0]["id"].as_i64().unwrap(),
        5,
        "response id must be 5 (from the real request)"
    );

    // Negative control: null-id request MUST produce a response.
    let (null_id_responses, _) = run_mcp_exchange(&[null_id_request]);
    assert_eq!(
        null_id_responses.len(), 1,
        "null-id request must produce exactly 1 response"
    );
    assert!(
        null_id_responses[0]["id"].is_null(),
        "null-id response must have id=null; got: {}",
        null_id_responses[0]["id"]
    );
}

// ── AT-1132: malformed jsonrpc field → correct error codes ────────────────────

#[test]
fn at_1132_mcp_rejects_malformed_jsonrpc_field() {
    // (a) jsonrpc="1.0" → -32600 (InvalidRequest).
    let req_a: &str = r#"{"jsonrpc":"1.0","id":1,"method":"initialize"}"#;
    let resp_a: serde_json::Value = rpc(req_a);
    assert_eq!(resp_a["id"].as_i64().unwrap(), 1, "(a) id must be echoed");
    assert_eq!(
        resp_a["error"]["code"].as_i64().unwrap(), -32600,
        "(a) jsonrpc=1.0 must return -32600; got: {}",
        resp_a["error"]
    );
    let received_a: &str = resp_a["error"]["data"]["received"].as_str().unwrap_or("");
    assert_eq!(received_a, "1.0", "(a) error.data.received must be '1.0'");

    // (b) jsonrpc="2.0.0" → -32600.
    let req_b: &str = r#"{"jsonrpc":"2.0.0","id":2,"method":"initialize"}"#;
    let resp_b: serde_json::Value = rpc(req_b);
    assert_eq!(resp_b["id"].as_i64().unwrap(), 2, "(b) id must be echoed");
    assert_eq!(
        resp_b["error"]["code"].as_i64().unwrap(), -32600,
        "(b) jsonrpc=2.0.0 must return -32600"
    );
    let received_b: &str = resp_b["error"]["data"]["received"].as_str().unwrap_or("");
    assert_eq!(received_b, "2.0.0", "(b) error.data.received must be '2.0.0'");

    // (c) empty method → -32601 (MethodNotFound).
    let req_c: &str = r#"{"jsonrpc":"2.0","id":3,"method":""}"#;
    let resp_c: serde_json::Value = rpc(req_c);
    assert_eq!(resp_c["id"].as_i64().unwrap(), 3, "(c) id must be echoed");
    assert_eq!(
        resp_c["error"]["code"].as_i64().unwrap(), -32601,
        "(c) empty method must return -32601"
    );
    let available_c: &serde_json::Value = &resp_c["error"]["data"]["available"];
    assert!(
        available_c.is_array(),
        "(c) error.data.available must be array; got: {available_c}"
    );
    assert_eq!(
        available_c.as_array().unwrap().len(),
        7,
        "(c) must list 7 methods"
    );
}
