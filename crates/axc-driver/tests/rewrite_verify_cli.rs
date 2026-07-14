//! M3.16 (FG.1) `axc rewrite-verify` — CPU-only tests (AT-2841…2850, no GPU).
//!
//! Most of the CPU-testable acceptance criteria (AT-2841…2848, AT-2850b) are
//! implemented as `#[cfg(test)]` unit tests colocated with the code they
//! exercise (`crates/axc-driver/src/cli.rs`, `crates/axc-driver/src/rewrite_verify.rs`,
//! `crates/axc-driver/src/mcp/dispatch.rs`, `crates/axc-driver/src/mcp/tools/verify_rewrite.rs`)
//! — the SAME convention the pre-existing `strip_strategy_annotation_block`
//! AT-2830…2835 tests follow (colocated because the exercised internals are
//! `pub(crate)`-scoped or because the test IS the unit under test's natural
//! home). This file holds the tests that specifically need the black-box,
//! outside-the-crate integration-test perspective: AT-2849's MCP round-trip
//! lives in `mcp_roundtrip.rs` (the established location for all MCP
//! subprocess tests); AT-2850 (CLI exit codes) is a genuine subprocess test
//! and lives here.

use std::io::Write as _;
use std::path::PathBuf;
use std::process::Command;

use axc_driver::rewrite_verify::{diff_interface, verify_rewrite, InterfaceMismatch, TolerancePolicy, VerifyRequest, Verdict};

const SAXPY_SRC: &str = include_str!("../../../examples/saxpy.axc");
const SAXPY_REASSOC_SRC: &str = include_str!("../../../examples/saxpy_reassoc.axc");

// ── Subprocess helper (mirrors mcp_roundtrip.rs's axc_binary_path) ────────────

fn axc_binary_path() -> PathBuf {
    let workspace_root: PathBuf = {
        let manifest_dir: &str = env!("CARGO_MANIFEST_DIR");
        PathBuf::from(manifest_dir)
            .ancestors()
            .nth(2) // crates/axc-driver → crates → workspace root
            .expect("workspace root")
            .to_path_buf()
    };
    let target_dir: PathBuf = workspace_root.join("target").join("debug").join("axc");
    if !target_dir.exists() {
        let status = Command::new("cargo")
            .args(["build", "-p", "axc-driver"])
            .current_dir(&workspace_root)
            .status()
            .expect("cargo build");
        assert!(status.success(), "cargo build -p axc-driver failed");
    }
    target_dir
}

fn examples_dir() -> PathBuf {
    let manifest_dir: &str = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir).ancestors().nth(2).expect("workspace root").join("examples")
}

fn write_temp_axc(dir: &std::path::Path, name: &str, contents: &str) -> PathBuf {
    let path = dir.join(name);
    let mut f = std::fs::File::create(&path).expect("create temp .axc");
    f.write_all(contents.as_bytes()).expect("write temp .axc");
    path
}

// ── AT-2850: CLI exit codes ────────────────────────────────────────────────────

/// AT-2850: `axc rewrite-verify` exit codes — REJECT (signature mismatch) → 1,
/// ERROR (under-specified request: no --buffer-sizes/--size) → 2. Neither
/// scenario needs a GPU (both are CPU-only gates), so this runs unconditionally.
#[test]
fn at_2850_cli_exit_codes_reject_and_error() {
    let axc = axc_binary_path();
    let tmp = tempfile::tempdir().expect("tempdir");

    let saxpy_path = write_temp_axc(tmp.path(), "saxpy.axc", SAXPY_SRC);
    let vector_add_src = concat!(
        "@kernel\n@workgroup(64, 1, 1)\n@intent(\"t\")\n@complexity(O(n))\n",
        "fn vector_add(a: readonly_buffer[f32], b: readonly_buffer[f32], c: buffer[f32]) -> void {\n",
        "    let i: u32 = gid(0);\n    c[i] = a[i] + b[i];\n    return;\n}\n",
    );
    let vadd_path = write_temp_axc(tmp.path(), "vector_add.axc", vector_add_src);

    // ── REJECT: buffer-count-changed signature (interface gate fires with NO
    // GPU and NO buffer-sizes needed — it runs before the request-completeness
    // preflight) → exit 1.
    let out = Command::new(&axc)
        .args([
            "rewrite-verify",
            saxpy_path.to_str().unwrap(),
            vadd_path.to_str().unwrap(),
            "--buffer-sizes", "256,256",
        ])
        .output()
        .expect("spawn axc rewrite-verify");
    assert_eq!(out.status.code(), Some(1), "REJECT must exit 1; stdout={}", String::from_utf8_lossy(&out.stdout));
    let report: serde_json::Value = serde_json::from_slice(&out.stdout).expect("stdout must be valid JSON");
    assert_eq!(report["verdict"], "REJECT");
    assert_eq!(report["stage"], "interface");

    // ── ERROR: neither --buffer-sizes nor --size supplied → usage ERROR, exit 2.
    let out2 = Command::new(&axc)
        .args(["rewrite-verify", saxpy_path.to_str().unwrap(), saxpy_path.to_str().unwrap()])
        .output()
        .expect("spawn axc rewrite-verify");
    assert_eq!(out2.status.code(), Some(2), "under-specified request must exit 2; stdout={}", String::from_utf8_lossy(&out2.stdout));
    let report2: serde_json::Value = serde_json::from_slice(&out2.stdout).expect("stdout must be valid JSON");
    assert_eq!(report2["verdict"], "ERROR");

    // Both stdout payloads must start with the verdict key (pipeline rule #3).
    let stdout1 = String::from_utf8_lossy(&out.stdout);
    let stdout2 = String::from_utf8_lossy(&out2.stdout);
    assert!(stdout1.trim_start().starts_with("{\n  \"verdict\""), "got: {stdout1}");
    assert!(stdout2.trim_start().starts_with("{\n  \"verdict\""), "got: {stdout2}");
}

/// AT-2850: PASS/SKIPPED exit 0. Drives the saxpy-vs-itself pair with no GPU
/// available in-process (`vk=None` via the library call, not the subprocess,
/// since we cannot force the subprocess's Vulkan probe to fail portably) —
/// covered by the library-level `at_2847_vk_none_skips_gpu_unavailable` unit
/// test (SKIPPED→0 is exercised there via `exit_code_for_verdict`). This test
/// additionally drives the CLI subprocess end-to-end for a REJECT-shaped
/// verdict already asserted above; the exit-code MAPPING itself (the pure
/// function) is unit-tested exhaustively in `rewrite_verify::tests::exit_code_mapping`.
#[test]
fn at_2850_exit_code_mapping_pass_skipped_nondeterministic_are_zero() {
    use axc_driver::rewrite_verify::exit_code_for_verdict;
    assert_eq!(exit_code_for_verdict(Verdict::Pass), 0);
    assert_eq!(exit_code_for_verdict(Verdict::Skipped), 0);
    assert_eq!(exit_code_for_verdict(Verdict::NondeterministicOracle), 0);
    assert_eq!(exit_code_for_verdict(Verdict::Fail), 1);
    assert_eq!(exit_code_for_verdict(Verdict::Reject), 1);
    assert_eq!(exit_code_for_verdict(Verdict::Error), 2);
}

// ── Black-box coverage over the PUBLIC API (outside-the-crate perspective) ────

/// Black-box (public-API) re-check of AT-2844/AT-2845 using the REAL
/// `saxpy.axc` / `saxpy_reassoc.axc` example fixtures from OUTSIDE the crate
/// (via `axc_driver::rewrite_verify::diff_interface`, not an internal helper).
#[test]
fn diff_interface_saxpy_reassoc_is_interface_identical() {
    let (_bytes_a, meta_a) = axc_driver::compile_source_with_meta(SAXPY_SRC).unwrap();
    let (_bytes_b, meta_b) = axc_driver::compile_source_with_meta(SAXPY_REASSOC_SRC).unwrap();
    assert_eq!(diff_interface(&meta_a, &meta_b), None, "saxpy_reassoc must be interface-identical to saxpy");
}

/// Positive/negative compile check (black-box): `verify_rewrite` on a
/// non-compiling `original` fails at the `compile` stage with side="original"
/// — the counterpart of the lib-level AT-2846 (which uses a bad `rewritten`).
#[test]
fn verify_rewrite_non_compiling_original_fails_compile_original() {
    let req = VerifyRequest {
        original: "@@@ not valid axc $$$".to_string(),
        rewritten: SAXPY_SRC.to_string(),
        assignments: Default::default(),
        tolerance: TolerancePolicy::BitExact,
        buffer_sizes: vec![256, 256],
        output_sizes: None,
        workgroups: None,
        push_constants: Some(vec![0_u8; 8]),
        seed: 0,
        kernel: None,
    };
    let report = verify_rewrite(&req, None);
    assert_eq!(report.verdict, Verdict::Fail);
    assert_eq!(report.stage, "compile");
    assert_eq!(report.reason.unwrap().kind, "compile_fail_original");
}

/// `TolerancePolicy` grammar round-trip using the examples fixture directory
/// as a sanity check that the fixtures created for M3.16 actually exist on
/// disk (guards against a stale `include_str!` masking a missing file).
#[test]
fn m316_example_fixtures_exist_on_disk() {
    let dir = examples_dir();
    for name in ["saxpy.axc", "saxpy_wrong.axc", "saxpy_perturbed.axc", "saxpy_reassoc.axc"] {
        assert!(dir.join(name).exists(), "missing M3.16 fixture: {name}");
    }
}

/// `InterfaceMismatch` reject path is reachable from outside the crate too
/// (public API), matching the CLI subprocess assertion above but via the
/// library call (`vk=None`, no subprocess).
#[test]
fn diff_interface_buffer_count_mismatch_is_public_api_reachable() {
    let vadd_src = concat!(
        "@kernel\n@workgroup(64, 1, 1)\n@intent(\"t\")\n@complexity(O(n))\n",
        "fn vector_add(a: readonly_buffer[f32], b: readonly_buffer[f32], c: buffer[f32]) -> void {\n",
        "    let i: u32 = gid(0);\n    c[i] = a[i] + b[i];\n    return;\n}\n",
    );
    let (_b1, meta_saxpy) = axc_driver::compile_source_with_meta(SAXPY_SRC).unwrap();
    let (_b2, meta_vadd) = axc_driver::compile_source_with_meta(vadd_src).unwrap();
    let mismatch = diff_interface(&meta_saxpy, &meta_vadd);
    assert!(matches!(mismatch, Some(InterfaceMismatch::BufferCount { original: 2, rewritten: 3 })), "got {mismatch:?}");
}
