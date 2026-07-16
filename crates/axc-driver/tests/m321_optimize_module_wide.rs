//! M3.21 (FG.9) — `axc optimize` module-wide hole discovery (F3) and winner
//! cross-validation (F4 / r3 orchestrator patch). AT-3013, AT-3014, AT-3016.
//!
//! M3.23 migrates all three `run_optimize` call sites to the new 5-arg
//! signature (`vk: Option<&VulkanContext>`) and rewrites the two callers that
//! depended on the OLD mock bench's "ordinal 0 always ties/wins" determinism
//! into GPU-gated (`#[ignore]` + `AXC_ENABLE_GPU_TESTS=1`, Lavapipe-runnable)
//! real-bench legs — the mock bench is gone (M3.23 §1).

use std::path::PathBuf;

use axc_driver::optimize::{run_optimize, OptimizeError};

fn gpu_tests_enabled() -> bool {
    std::env::var("AXC_ENABLE_GPU_TESTS").as_deref() == Ok("1")
}

/// A 2-kernel source sharing `?WG` (module-global, F3): BOTH kernels declare
/// `@strategy { WG: ?[32, 64] }` with a BYTE-IDENTICAL candidate list (§4
/// rule 3 — "shared = same name" requires identical lists; HIR's per-kernel
/// `validate_hole_refs_in_args`, pre-existing M2.3 code untouched by M3.21,
/// requires a `?name` reference to be declared in THAT SAME kernel's own
/// `@strategy` block — so "declared-in-one-kernel-only, referenced in a
/// sibling with no local declaration" is not achievable without also
/// changing HIR's per-kernel HoleRef validation, which is out of scope here;
/// `two_pass_reduce.axc` uses the identical both-declare pattern). F3's
/// module-global UNION still does real work here: it is what lets ONE grid
/// search (benching only `bench_target`) size the Cartesian product AND
/// (via F4) validate the resolved value against `sibling` too.
const SHARED_HOLE_SRC: &str = concat!(
    "@kernel\n",
    "@workgroup(?WG, 1, 1)\n",
    "@strategy { WG: ?[32, 64] }\n",
    "@intent(\"bench target\")\n",
    "@complexity(O(n))\n",
    "fn bench_target(x: buffer[u32]) -> void {\n",
    "    let i: u32 = gid(0u32);\n",
    "    x[i] = i;\n",
    "    return;\n",
    "}\n",
    "@kernel\n",
    "@workgroup(?WG, 1, 1)\n",
    "@strategy { WG: ?[32, 64] }\n",
    "@intent(\"sibling, shares the SAME hole name+candidates\")\n",
    "@complexity(O(n))\n",
    "fn sibling(y: buffer[u32]) -> void {\n",
    "    let i: u32 = gid(0u32);\n",
    "    y[i] = i;\n",
    "    return;\n",
    "}\n",
);

fn tempfile_dir(label: &str) -> PathBuf {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("axc_{label}_{nanos}"));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

/// Probe a real `VulkanContext` (Lavapipe-runnable) the same way `main.rs`
/// wires `axc optimize`. `None` if no Vulkan device is available.
fn probe_vk() -> Option<axc_runtime::VulkanContext> {
    if axc_runtime::probe_vulkan_available() {
        axc_runtime::VulkanContext::new().ok()
    } else {
        None
    }
}

/// AT-3014 (rewritten from AT-2957): `axc optimize multi.axc --kernel
/// bench_target` real-benches over the UNION holes (F3), and the winner is
/// cross-compiled across ALL kernels (F4) BEFORE the winner sidecar is
/// written. GPU-gated (Lavapipe-runnable, non-coopmat fixture) — does NOT
/// assert which candidate wins (real timings).
#[test]
#[ignore]
fn at_3014_optimize_union_holes_and_f4_cross_validate_before_write() {
    if !gpu_tests_enabled() {
        eprintln!("AT-3014: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    let Some(vk) = probe_vk() else {
        eprintln!("AT-3014: no Vulkan device available; skipping");
        return;
    };

    let dir = tempfile_dir("at_3014");
    let src = dir.join("multi.axc");
    std::fs::write(&src, SHARED_HOLE_SRC).unwrap();
    let out = dir.join("out.spv");

    let result = run_optimize(&src, &out, "none", Some("bench_target"), Some(&vk));
    assert!(result.is_ok(), "optimize must succeed: {result:?}");
    assert!(out.exists(), "winning .spv must be written");
    let sidecar = axc_driver::optimize::strategy_sidecar_path(&out);
    assert!(sidecar.exists(), "strategy sidecar must be written");

    let sidecar_text = std::fs::read_to_string(&sidecar).unwrap();
    // The union has one axis (WG ∈ {32,64}); winner_assignments must carry it.
    assert!(sidecar_text.contains("\"WG\""), "sidecar must record the WG assignment: {sidecar_text}");

    let _ = std::fs::remove_dir_all(&dir);
}

/// AT-3013: `axc optimize` with no `--kernel` on an ambiguous (2+ kernel)
/// source fails closed (never a silent kernels[0] pick) — HEADLESS
/// (`vk=None`), proving the AmbiguousKernel diagnostic fires at
/// `select_kernel`, BEFORE the `vk==None` GpuUnavailable check (§1.3.1
/// ordering pin). Arity-only migration from the pre-M3.23 test.
#[test]
fn at_3013_optimize_no_kernel_on_ambiguous_source_fails_closed() {
    let dir = tempfile_dir("optimize_ambiguous");
    let src = dir.join("multi.axc");
    std::fs::write(&src, SHARED_HOLE_SRC).unwrap();
    let out = dir.join("out.spv");

    let result = run_optimize(&src, &out, "none", None, None);
    match result {
        Err(OptimizeError::Compile(axc_driver::DriverError::AmbiguousKernel { available })) => {
            assert_eq!(available, vec!["bench_target".to_string(), "sibling".to_string()]);
        }
        other => panic!("expected AmbiguousKernel; got {other:?}"),
    }
    assert!(!out.exists(), "no output on an ambiguous bench-target selection");

    let _ = std::fs::remove_dir_all(&dir);
}

/// AT-3016 (rewritten from AT-2970): a SINGLE-candidate `@strategy { BAD: ?[8] }`
/// forces `BAD=8` as the winner bench-independently (only candidate; even the
/// R3 fallback crowns ordinal 0) — `bench_target` benches fine (1-D,
/// product=8), but F4 then rejects the winner because `sibling`
/// (`@workgroup(?BAD,16,16)`, 8*16*16=2048 > 1024) breaks module-wide =>
/// `WinnerInvalidModuleWide` naming `sibling`; NEITHER `.spv` NOR sidecar
/// exists. GPU-gated (Lavapipe-runnable) — the real bench must actually run
/// (and succeed) on `bench_target` before F4 fires.
#[test]
#[ignore]
fn at_3016_winner_invalid_for_sibling_fails_closed_zero_artifacts() {
    if !gpu_tests_enabled() {
        eprintln!("AT-3016: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    let Some(vk) = probe_vk() else {
        eprintln!("AT-3016: no Vulkan device available; skipping");
        return;
    };

    let src_text = concat!(
        "@kernel\n",
        "@workgroup(?BAD, 1, 1)\n",
        "@strategy { BAD: ?[8] }\n",
        "@intent(\"bench target: legal for the sole candidate as a 1-D workgroup\")\n",
        "@complexity(O(n))\n",
        "fn bench_target(x: buffer[u32]) -> void {\n",
        "    let i: u32 = gid(0u32);\n",
        "    x[i] = i;\n",
        "    return;\n",
        "}\n",
        "@kernel\n",
        "@workgroup(?BAD, 16, 16)\n",
        "@strategy { BAD: ?[8] }\n",
        "@intent(\"sibling: BAD=8 overflows the 1024-invocation ceiling (8*16*16=2048)\")\n",
        "@complexity(O(n))\n",
        "fn sibling(y: buffer[u32]) -> void {\n",
        "    let i: u32 = gid(0u32);\n",
        "    y[i] = i;\n",
        "    return;\n",
        "}\n",
    );
    let dir = tempfile_dir("at_3016");
    let src = dir.join("multi.axc");
    std::fs::write(&src, src_text).unwrap();
    let out = dir.join("out.spv");
    let sidecar = axc_driver::optimize::strategy_sidecar_path(&out);

    // Fixture sanity: BAD=8 really does break `sibling` module-wide.
    let mut bad = std::collections::BTreeMap::new();
    bad.insert("BAD".to_string(), 8i64);
    let resolved = axc_driver::substitute_strategy_holes(src_text, &bad);
    assert!(
        axc_driver::compile_module_all(&resolved, false).is_err(),
        "fixture sanity: BAD=8 must break `sibling` module-wide"
    );

    // The actual F4 firing, inside run_optimize's own real-bench winner path.
    let result = run_optimize(&src, &out, "none", Some("bench_target"), Some(&vk));
    match result {
        Err(OptimizeError::WinnerInvalidModuleWide { broken_kernel, detail }) => {
            assert_eq!(broken_kernel.as_deref(), Some("sibling"), "F4 must name the broken sibling; detail={detail}");
        }
        other => panic!("expected WinnerInvalidModuleWide naming `sibling`; got {other:?}"),
    }

    // r3: NEITHER the .spv NOR the sidecar may exist after an F4 failure.
    assert!(!out.exists(), "AT-3016 (r3): NO .spv may be written on an F4 failure");
    assert!(!sidecar.exists(), "AT-3016 (r3): NO strategy sidecar may be written on an F4 failure");

    let _ = std::fs::remove_dir_all(&dir);
}
