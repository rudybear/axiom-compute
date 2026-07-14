//! M3.21 (FG.9) — `axc compile` multi-kernel CLI acceptance tests
//! (AT-2949…AT-2952). Genuine subprocess tests (spawns the built `axc`
//! binary), mirroring the established convention in `rewrite_verify_cli.rs`
//! for CLI-level exit-code / file-written / argv behavior that cannot be
//! exercised by calling library functions directly (the naming/dispatch glue
//! lives in `main.rs`, a separate binary crate).
//!
//! Uses a hole-free inline 2-kernel fixture (written to a temp `.axc` file)
//! rather than `examples/two_pass_reduce.axc` — that file's `?WG` hole
//! requires `--strategy-value` resolution first, which would conflate "the
//! kernel-selection/naming CLI behavior under test" with "hole resolution",
//! two independent concerns.

use std::path::PathBuf;
use std::process::Command;

const MULTI_SRC: &str = concat!(
    "@kernel\n",
    "@workgroup(64, 1, 1)\n",
    "@intent(\"stage 1\")\n",
    "@complexity(O(n))\n",
    "fn partial_reduce(x: buffer[u32]) -> void {\n",
    "    let i: u32 = gid(0u32);\n",
    "    x[i] = i;\n",
    "    return;\n",
    "}\n",
    "@kernel\n",
    "@workgroup(32, 1, 1)\n",
    "@intent(\"stage 2\")\n",
    "@complexity(O(n))\n",
    "fn final_reduce(y: buffer[u32]) -> void {\n",
    "    let i: u32 = gid(0u32);\n",
    "    y[i] = i;\n",
    "    return;\n",
    "}\n",
);

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

/// Unique-per-test temp subdirectory (avoids cross-test filename collisions
/// under parallel `cargo test` execution).
fn tempfile_dir(label: &str) -> PathBuf {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("axc_{label}_{nanos}"));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn write_multi_fixture(dir: &std::path::Path) -> PathBuf {
    let path = dir.join("multi.axc");
    std::fs::write(&path, MULTI_SRC).expect("write multi.axc fixture");
    path
}

/// AT-2949: `axc compile multi.axc -o out.spv` (no `--kernel`/`--all`) on a
/// 2-kernel source -> `AmbiguousKernel` listing both names, NO file written,
/// non-zero exit.
#[test]
fn at_2949_ambiguous_multi_kernel_no_selector_fails_closed_no_file() {
    let bin = axc_binary_path();
    let dir = tempfile_dir("at_2949");
    let src = write_multi_fixture(&dir);
    let out = dir.join("out.spv");
    let sidecar = axc_driver::metadata_sidecar_path(&out);

    let output = Command::new(&bin)
        .args(["compile", src.to_str().unwrap(), "-o", out.to_str().unwrap()])
        .output()
        .expect("spawn axc compile");

    assert!(!output.status.success(), "ambiguous compile must exit non-zero");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("partial_reduce") && stderr.contains("final_reduce"),
        "error must list both kernel names; stderr={stderr}"
    );
    assert!(!out.exists(), "AT-2949: NO .spv file may be written on an ambiguous compile");
    assert!(!sidecar.exists(), "AT-2949: NO sidecar may be written on an ambiguous compile");

    let _ = std::fs::remove_dir_all(&dir);
}

/// AT-2950: `axc compile multi.axc -o out.spv --kernel final_reduce` emits
/// the final_reduce module + sidecar, `entry_point == "final_reduce"`; the
/// SPIR-V passes spirv-val.
#[test]
fn at_2950_kernel_selector_emits_named_kernel_plus_sidecar() {
    let bin = axc_binary_path();
    let dir = tempfile_dir("at_2950");
    let src = write_multi_fixture(&dir);
    let out = dir.join("out.spv");
    let sidecar = axc_driver::metadata_sidecar_path(&out);

    let output = Command::new(&bin)
        .args(["compile", src.to_str().unwrap(), "-o", out.to_str().unwrap(), "--kernel", "final_reduce"])
        .output()
        .expect("spawn axc compile");

    assert!(output.status.success(), "AT-2950: --kernel compile must succeed; stderr={}", String::from_utf8_lossy(&output.stderr));
    assert!(out.exists(), "AT-2950: .spv must be written");
    assert!(sidecar.exists(), "AT-2950: sidecar must be written");

    let sidecar_text = std::fs::read_to_string(&sidecar).expect("read sidecar");
    assert!(sidecar_text.contains("\"final_reduce\""), "sidecar must name final_reduce; got {sidecar_text}");

    // spirv-val
    let bytes = std::fs::read(&out).expect("read spv");
    let words: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    use spirv_tools::val::{Validator, create as create_validator};
    use spirv_tools::TargetEnv;
    create_validator(Some(TargetEnv::Vulkan_1_1)).validate(&words, None)
        .expect("AT-2950: spirv-val must accept the emitted final_reduce module");

    let _ = std::fs::remove_dir_all(&dir);
}

/// AT-2951: `axc compile multi.axc -o build/scan.spv --all` -> one `.spv` +
/// sidecar pair PER kernel, named `<stem>.<kernel>.spv`.
#[test]
fn at_2951_all_flag_multi_emits_one_pair_per_kernel() {
    let bin = axc_binary_path();
    let dir = tempfile_dir("at_2951");
    let src = write_multi_fixture(&dir);
    let out = dir.join("scan.spv");

    let output = Command::new(&bin)
        .args(["compile", src.to_str().unwrap(), "-o", out.to_str().unwrap(), "--all"])
        .output()
        .expect("spawn axc compile --all");
    assert!(output.status.success(), "AT-2951: --all compile must succeed; stderr={}", String::from_utf8_lossy(&output.stderr));

    let partial_spv = dir.join("scan.partial_reduce.spv");
    let final_spv = dir.join("scan.final_reduce.spv");
    assert!(partial_spv.exists(), "AT-2951: scan.partial_reduce.spv must exist");
    assert!(final_spv.exists(), "AT-2951: scan.final_reduce.spv must exist");
    assert!(axc_driver::metadata_sidecar_path(&partial_spv).exists(), "AT-2951: partial_reduce sidecar must exist");
    assert!(axc_driver::metadata_sidecar_path(&final_spv).exists(), "AT-2951: final_reduce sidecar must exist");
    assert!(!out.exists(), "AT-2951: the plain (non-suffixed) scan.spv must NOT be written by --all");

    // Distinct SPIR-V (different @workgroup dims: 64,1,1 vs 32,1,1).
    let partial_bytes = std::fs::read(&partial_spv).unwrap();
    let final_bytes = std::fs::read(&final_spv).unwrap();
    assert_ne!(partial_bytes, final_bytes, "AT-2951: distinct kernels must not collide on identical bytes");

    let _ = std::fs::remove_dir_all(&dir);
}

/// AT-2952: `--kernel nonexistent` -> `UnknownKernel` listing available names,
/// non-zero exit, no file written.
#[test]
fn at_2952_unknown_kernel_name_fails_closed() {
    let bin = axc_binary_path();
    let dir = tempfile_dir("at_2952");
    let src = write_multi_fixture(&dir);
    let out = dir.join("out.spv");

    let output = Command::new(&bin)
        .args(["compile", src.to_str().unwrap(), "-o", out.to_str().unwrap(), "--kernel", "nonexistent"])
        .output()
        .expect("spawn axc compile");

    assert!(!output.status.success(), "unknown --kernel must exit non-zero");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("nonexistent"), "error must name the unknown kernel; stderr={stderr}");
    assert!(
        stderr.contains("partial_reduce") && stderr.contains("final_reduce"),
        "error must list available kernel names; stderr={stderr}"
    );
    assert!(!out.exists(), "AT-2952: NO file may be written on an unknown --kernel");

    let _ = std::fs::remove_dir_all(&dir);
}

/// `--kernel` on the REAL `examples/two_pass_reduce.axc` (with its `?WG`
/// hole resolved via `--strategy-value`) also succeeds end-to-end through
/// the actual binary — the real-file corroboration of AT-2950.
#[test]
fn m321_real_two_pass_reduce_kernel_selector_with_strategy_value() {
    let bin = axc_binary_path();
    let src = examples_dir().join("two_pass_reduce.axc");
    let dir = tempfile_dir("m321_real");
    let out = dir.join("out.spv");

    let output = Command::new(&bin)
        .args([
            "compile", src.to_str().unwrap(), "-o", out.to_str().unwrap(),
            "--kernel", "final_reduce", "--strategy-value", "WG=64",
        ])
        .output()
        .expect("spawn axc compile");
    assert!(output.status.success(), "stderr={}", String::from_utf8_lossy(&output.stderr));
    assert!(out.exists());

    let _ = std::fs::remove_dir_all(&dir);
}

/// Single-kernel file through `--kernel`/`--all`-aware CLI is
/// behavior-identical to the plain (no-flag) path — the golden gate holds
/// end-to-end through the actual binary, not just the library.
#[test]
fn m321_single_kernel_compile_unaffected_by_new_flags() {
    let bin = axc_binary_path();
    let src = examples_dir().join("saxpy.axc");
    let dir = tempfile_dir("m321_single");
    let out_plain = dir.join("saxpy_plain.spv");
    let out_flagged = dir.join("saxpy_flagged.spv");

    let status_plain = Command::new(&bin)
        .args(["compile", src.to_str().unwrap(), "-o", out_plain.to_str().unwrap()])
        .status().expect("spawn");
    assert!(status_plain.success());

    let status_flagged = Command::new(&bin)
        .args(["compile", src.to_str().unwrap(), "-o", out_flagged.to_str().unwrap(), "--kernel", "saxpy"])
        .status().expect("spawn");
    assert!(status_flagged.success());

    let bytes_plain = std::fs::read(&out_plain).unwrap();
    let bytes_flagged = std::fs::read(&out_flagged).unwrap();
    assert_eq!(bytes_plain, bytes_flagged, "single-kernel --kernel selection must be byte-identical to the plain path");

    let _ = std::fs::remove_dir_all(&dir);
}
