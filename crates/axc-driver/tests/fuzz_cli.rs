//! M3.18 (FG.8) `axc test --fuzz` — CLI-layer subprocess coverage (AT-2893).
//!
//! QA-fix (post-M3.18): the coder's original AT-2893 unit test
//! (`axc-driver/src/fuzz.rs::at_2893_fuzz_unsat_library_level_fuzz_kernel_no_gpu_needed`)
//! called `fuzz_kernel()` directly with a hand-supplied `buffer_sizes`, which
//! bypassed `main.rs::run_test`'s own CLI-level buffer-size handling entirely.
//! That gap hid a real bug: `run_test` used to require `--buffer-sizes`/`--size`
//! BEFORE ever calling `fuzz_kernel`, so the literal spec command for AT-2893
//! (`axc test --fuzz examples/fuzz_unsat.axc`, no other flags) returned a usage
//! `ERROR`/exit 2 instead of the spec-mandated `UNSATISFIABLE`/exit 3 --
//! `fuzz_kernel`'s own internal ordering resolves `solve()`/UNSATISFIABLE
//! before buffer sizes are ever consulted, but the CLI gate pre-empted it.
//!
//! Fixed by threading an empty `Vec` through to `fuzz_kernel` when neither
//! `--buffer-sizes` nor `--size` is supplied, letting `fuzz_kernel`'s own
//! `solve()` short-circuit run first; `fuzz_kernel`'s existing
//! `buffer_sizes.len()` mismatch check still raises the equivalent usage
//! ERROR/exit 2 for kernels that turn out to be satisfiable and would
//! genuinely need to dispatch (see `at_2893b_...` below).
//!
//! This file drives the REAL built `axc` binary as a subprocess, the same
//! convention `rewrite_verify_cli.rs`'s `at_2850_cli_exit_codes_reject_and_error`
//! uses, so it exercises the exact code path a user (or the spec's own AT
//! table) invokes -- not an internal shortcut.

use std::path::PathBuf;
use std::process::Command;

fn axc_binary_path() -> PathBuf {
    let workspace_root: PathBuf = {
        let manifest_dir: &str = env!("CARGO_MANIFEST_DIR");
        PathBuf::from(manifest_dir)
            .ancestors()
            .nth(2) // crates/axc-driver -> crates -> workspace root
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

/// AT-2893 (CLI-layer): the literal spec command -- `axc test --fuzz
/// examples/fuzz_unsat.axc` with NO `--buffer-sizes`/`--size`/any other flag
/// -- must report `UNSATISFIABLE`/exit 3, WITHOUT ever needing a buffer size,
/// because `fuzz_unsat.axc`'s two `@precondition`s (`gt(n,10)` and `lt(n,5)`)
/// are unsatisfiable at the `solve()` stage, which runs before dispatch prep.
#[test]
fn at_2893_cli_fuzz_unsat_no_size_flags_end_to_end() {
    let axc = axc_binary_path();
    let fuzz_unsat_path = examples_dir().join("fuzz_unsat.axc");
    assert!(fuzz_unsat_path.exists(), "missing fixture: {fuzz_unsat_path:?}");

    let out = Command::new(&axc)
        .args(["test", "--fuzz", fuzz_unsat_path.to_str().unwrap()])
        .output()
        .expect("spawn axc test --fuzz");

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(
        out.status.code(),
        Some(3),
        "UNSATISFIABLE must exit 3 with NO size flags supplied; stdout={stdout}"
    );
    let report: serde_json::Value = serde_json::from_slice(&out.stdout).expect("stdout must be valid JSON");
    assert_eq!(report["verdict"], "UNSATISFIABLE");
    assert_eq!(report["runs_executed"], 0, "no dispatch must be attempted");
    // verdict must serialize FIRST (pipeline rule #3).
    assert!(stdout.trim_start().starts_with("{\n  \"verdict\""), "got: {stdout}");
}

/// Companion regression guard for the QA fix's "preserve existing behavior"
/// requirement: a kernel that IS satisfiable (and would therefore genuinely
/// need to dispatch) still gets a usage `ERROR`/exit 2 when neither
/// `--buffer-sizes` nor `--size` is supplied -- the CLI-layer buffer-size
/// requirement is not silently dropped, it is only re-ordered to run AFTER
/// the UNSATISFIABLE check.
#[test]
fn at_2893b_cli_fuzz_satisfiable_kernel_missing_size_flags_is_still_usage_error() {
    let axc = axc_binary_path();
    let saxpy_path = examples_dir().join("precondition_saxpy.axc");
    assert!(saxpy_path.exists(), "missing fixture: {saxpy_path:?}");

    let out = Command::new(&axc)
        .args(["test", "--fuzz", saxpy_path.to_str().unwrap()])
        .output()
        .expect("spawn axc test --fuzz");

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(
        out.status.code(),
        Some(2),
        "a satisfiable kernel missing --buffer-sizes/--size must still usage-error/exit 2; stdout={stdout}"
    );
    let report: serde_json::Value = serde_json::from_slice(&out.stdout).expect("stdout must be valid JSON");
    assert_eq!(report["verdict"], "ERROR");
    assert_eq!(report["reason"]["kind"], "usage");
}
