//! M3.18 (FG.8) GPU acceptance tests: AT-2896, AT-2897, AT-2898, AT-2899.
//!
//! Exercises `axc_driver::fuzz::fuzz_kernel` end-to-end against
//! `examples/precondition_saxpy.axc` on a REAL Vulkan device — the M3.17 debug-flag
//! oracle (`VulkanContext::dispatch_debug_checked`) is GPU-proven
//! (`axc-runtime/tests/dispatch_debug_checks.rs`, AT-2870..2873); this suite asserts
//! `fuzz_kernel`'s ORCHESTRATION on top of it.
//!
//! GPU-gated: requires `AXC_ENABLE_GPU_TESTS=1` + a responsive Vulkan ICD. Run on
//! BOTH real NVIDIA (no `VK_DRIVER_FILES`) and Lavapipe
//! (`VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json`) per the coder mandate —
//! `precondition_saxpy` is non-coopmat/non-shared, so every AT here runs on both.

use std::collections::BTreeMap;

use axc_driver::fuzz::{exit_code_for_fuzz, fuzz_kernel, FuzzRequest, FuzzVerdict};
use axc_runtime::{gpu_tests_enabled, probe_vulkan_available, VulkanContext};

const PRECONDITION_SAXPY_SRC: &str = include_str!("../../../examples/precondition_saxpy.axc");

fn skip_unless_gpu() -> bool {
    if !gpu_tests_enabled() {
        eprintln!("skipping (AXC_ENABLE_GPU_TESTS != 1)");
        return true;
    }
    if !probe_vulkan_available() {
        eprintln!("skipping (no Vulkan ICD available)");
        return true;
    }
    false
}

fn base_request() -> FuzzRequest {
    FuzzRequest {
        source: PRECONDITION_SAXPY_SRC.to_string(),
        runs: 16,
        seed: 0,
        debug_oracle: Some(true),
        violate: false,
        buffer_sizes: vec![256, 256],
        workgroups: None,
        scalar_window: 256,
        strategy_assignments: BTreeMap::new(),
    }
}

/// AT-2896 (fuzz PASS with debug-oracle): satisfier-soundness invariant — every
/// run's constrained precondition bits stay clean (flag_words[0]==0, r2/M-1 strict-
/// interior open-bound sampler) — AND the postcondition `lt(alpha,1000)` holds
/// (default window 256 keeps alpha well under 1000). Assert `counterexample==null`.
#[test]
#[ignore] // GPU-gated
fn at_2896_fuzz_pass_with_debug_oracle_default_window() {
    if skip_unless_gpu() {
        return;
    }
    let vk = VulkanContext::new().expect("VulkanContext::new()");
    let req = base_request();
    let report = fuzz_kernel(&req, Some(&vk));

    assert_eq!(report.verdict, FuzzVerdict::Pass, "report: {report:#?}");
    assert_eq!(exit_code_for_fuzz(report.verdict), 0);
    assert!(report.counterexample.is_none());
    assert!(report.debug_oracle, "precondition_saxpy has a lowered Post check; debug_oracle must auto-ON");
    assert_eq!(report.runs_executed, req.runs);
}

/// AT-2897 (fuzz FAIL — postcondition counterexample; THE ALWAYS-PASS-ORACLE
/// GUARD, r2/C-1): reuse `precondition_saxpy`, widen `--scalar-window` so `alpha`'s
/// effective window reaches >= 1000, making a precondition-satisfying input
/// (`alpha > 0`) VIOLATE the lowered postcondition `lt(alpha, 1000)`.
///
/// r3/fix-5 PROVEN-FIRING TRIPLE, measured once at implementation time via a
/// throwaway CPU-only harness over the (already-pub) `solve`/`sample_all`
/// functions (no GPU needed to find it — the sampler is deterministic): with
/// `AT2897_WINDOW=1500`, `AT2897_SEED=0`, the very FIRST run (run 0) draws
/// `alpha >= 1000`, so `AT2897_RUNS=4` is ample margin and the fire is NOT
/// merely probable — it is guaranteed for this pinned triple. Re-measure and
/// re-pin if the standard library RNG stream ever changes under a toolchain bump.
#[test]
#[ignore] // GPU-gated
fn at_2897_fuzz_fail_postcondition_counterexample_always_pass_oracle_guard() {
    if skip_unless_gpu() {
        return;
    }
    const AT2897_SEED: u64 = 0;
    const AT2897_WINDOW: i64 = 1500;
    const AT2897_RUNS: u32 = 4;

    let vk = VulkanContext::new().expect("VulkanContext::new()");
    let req = FuzzRequest {
        source: PRECONDITION_SAXPY_SRC.to_string(),
        runs: AT2897_RUNS,
        seed: AT2897_SEED,
        debug_oracle: Some(true),
        violate: false,
        buffer_sizes: vec![256, 256],
        workgroups: None,
        scalar_window: AT2897_WINDOW,
        strategy_assignments: BTreeMap::new(),
    };
    let report = fuzz_kernel(&req, Some(&vk));

    assert_eq!(report.verdict, FuzzVerdict::Fail, "report: {report:#?}");
    assert_eq!(exit_code_for_fuzz(report.verdict), 1);
    let reason = report.reason.expect("FAIL must carry a reason");
    assert_eq!(reason.kind, "postcondition_violation");
    let ce = report.counterexample.expect("FAIL must carry a counterexample");
    assert_eq!(ce.run, 0, "the pinned triple fires on run 0 (measured at implementation time)");
    let alpha = ce.scalars.get("alpha").and_then(serde_json::Value::as_f64).expect("counterexample must name alpha");
    assert!(alpha >= 1000.0, "counterexample alpha must actually violate lt(alpha,1000); got {alpha}");
}

/// AT-2898 (fuzz determinism leg): the leg's presence + green IS the assertion —
/// no nondeterministic fixture is contrived (determinism is the expected result).
/// A `notes` entry records the leg ran and which artifact (DEBUG, since
/// `debug_oracle=true` here) it certified (r2/W-2 honesty note).
#[test]
#[ignore] // GPU-gated
fn at_2898_fuzz_determinism_leg_present_and_green() {
    if skip_unless_gpu() {
        return;
    }
    let vk = VulkanContext::new().expect("VulkanContext::new()");
    let req = base_request();
    let report = fuzz_kernel(&req, Some(&vk));

    assert_eq!(report.verdict, FuzzVerdict::Pass, "report: {report:#?}");
    assert!(
        report.notes.iter().any(|n| n.contains("determinism leg (c)") && n.contains("byte-identical")),
        "expected a determinism-leg note; got {:?}", report.notes
    );
    assert!(
        report.notes.iter().any(|n| n.contains("DEBUG artifact")),
        "expected the DEBUG-artifact honesty note (r2/W-2); got {:?}", report.notes
    );
}

/// AT-2899 (violate leg): targets the PINNED non-size-coupled scalar `alpha`
/// (r2/W-1 — NOT `n`), generates `alpha <= 0` (violates `gt(alpha,0)`), and
/// asserts the precondition `flag_words[0]` bit for `gt(alpha,0)` fires. `alpha`
/// bounds no buffer, so this cannot device-fault. Non-coopmat/non-shared, so this
/// is a rare cross-vendor GPU AT that does NOT need NVIDIA (Lavapipe runs it too).
#[test]
#[ignore] // GPU-gated
fn at_2899_violate_leg_fires_alpha_precondition() {
    if skip_unless_gpu() {
        return;
    }
    let vk = VulkanContext::new().expect("VulkanContext::new()");
    let mut req = base_request();
    req.violate = true;
    let report = fuzz_kernel(&req, Some(&vk));

    assert!(report.violate_leg, "violate_leg must be recorded true");
    // The violate leg PASSING means: no counterexample naming
    // `precondition_check_did_not_fire`, and a note confirming the fire.
    let did_not_fire = report.counterexample.as_ref().map(|c| c.reason == "precondition_check_did_not_fire").unwrap_or(false);
    assert!(!did_not_fire, "violate leg must have FIRED the gt(alpha,0) check; report: {report:#?}");
    assert!(
        report.notes.iter().any(|n| n.contains("violate leg") && n.contains("fired as expected") && n.contains("alpha")),
        "expected a violate-leg fired note naming alpha; got {:?}", report.notes
    );
}
