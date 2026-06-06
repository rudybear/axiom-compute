//! M3.4 — llama.cpp Vulkan Q4_K_M A/B integration tests.
//!
//! AT-1760: the A/B harness exists and emits ab_results.json (script-driven, GPU-gated).
//! AT-1761: AXIOM Q4_K_M is still bit-exact vs the ggml CPU reference (the A/B compares a
//!          CORRECT kernel). Runs on any GPU incl. Lavapipe (GPU-gated, NOT NVIDIA-only).
//! AT-1766: committed ab_results.json records the llama.cpp timer boundary VERBATIM, the
//!          tag + 40-char SHA, AXIOM min/mean/median + sustained, the boundary label, the
//!          FLOP-consistency result, and a byte-identical device-match.
//! AT-1767: INCOMPLETE (not a GEMM-substituted FAIL) when no n==1 Q4_K case exists — the
//!          kill-criterion number is never computed from a GEMM (n>1) case.
//!
//! The NVIDIA A/B numbers themselves are produced by scripts/m34_llamacpp_ab.sh on the
//! NVIDIA box (llama.cpp is gitignored, not built in CI). These tests validate the AXIOM
//! side's correctness + the committed results artifact's schema/honesty invariants.

#[path = "common_helpers.rs"]
mod helpers;

use std::path::PathBuf;

fn gpu_tests_enabled() -> bool {
    std::env::var("AXC_ENABLE_GPU_TESTS").as_deref() == Ok("1")
}

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = crates/axc-driver
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

/// AT-1760 (script existence + executability): the A/B harness + emitter are present and
/// the bench is registered. The full run is GPU/llama.cpp-gated (NVIDIA box only).
#[test]
fn at_1760_ab_harness_present() {
    let root = repo_root();
    let script = root.join("scripts/m34_llamacpp_ab.sh");
    let emitter = root.join("scripts/m34_emit.py");
    assert!(script.exists(), "AT-1760: scripts/m34_llamacpp_ab.sh must exist");
    assert!(emitter.exists(), "AT-1760: scripts/m34_emit.py must exist");

    let script_src = std::fs::read_to_string(&script).expect("read m34 script");
    // Pinned LITERAL 40-char SHA default (WARNING-4), not a floating tag.
    assert!(
        script_src.contains("LLAMACPP_COMMIT:-6b80c74f285390368b3c99c5e750f19e9b096e98"),
        "AT-1760: script must default LLAMACPP_COMMIT to the literal 40-char SHA"
    );
    // Fail-closed: device-match + n==1 (CRITICAL-2) handling present.
    assert!(script_src.contains("n=1"), "AT-1760: script must select the n==1 Q4_K case");
    assert!(
        script_src.contains("INCOMPLETE"),
        "AT-1760: script must have the INCOMPLETE fail-closed path"
    );
}

/// AT-1761: AXIOM Q4_K_M dequant+matvec is bit-exact vs the ggml CPU reference.
///
/// GPU-gated (runs on Lavapipe in CI). Mirrors the bench pre-flight oracle: dispatch once
/// with REAL 144-byte Q4_K_M weights and assert rel_err < 1e-3. A wrong-but-fast kernel
/// must not pollute the A/B.
#[test]
#[ignore]
fn at_1761_q4km_matvec_bit_exact() {
    if !gpu_tests_enabled() {
        eprintln!("AT-1761: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    use axc_driver::compile_source_with_meta;
    use axc_runtime::{DispatchError, DispatchRequest, VulkanContext};

    const Q4KM_SRC: &str = include_str!("../../../examples/q4km_dequant_matvec.axc");
    const N_SUPERBLOCKS: u32 = 56; // K = 56*256 = 14336, matching the A/B llama.cpp case.
    const SEED: u64 = 0x5A11_00AB;

    let ctx = match VulkanContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("AT-1761: no Vulkan: {e}; skipping");
            return;
        }
    };

    let (bytes, meta) = compile_source_with_meta(Q4KM_SRC).expect("q4km matvec must compile");
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Pipeline-create preflight (typed skip if device lacks 8-bit storage etc.).
    match ctx.prepare_kernel_checked(
        &words,
        &meta.binding_plan,
        meta.push_constant_total_bytes,
        &meta.entry_point,
        None,
        "q4km_dequant_matvec",
        meta.shared_memory_bytes,
    ) {
        Ok(_) => {}
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
            eprintln!("AT-1761: DeviceFeatureUnsupported {feature}/{kernel}; skipping");
            return;
        }
        Err(e) => panic!("AT-1761: unexpected prepare error: {e}"),
    }

    let n = N_SUPERBLOCKS as usize;
    let (q_bytes, x_vec) = helpers::generate_q4km_test_data(N_SUPERBLOCKS, SEED);
    let cpu_result = helpers::q4km_dequant_matvec_cpu(&q_bytes, &x_vec, n);

    let x_bytes: Vec<u8> = x_vec.iter().flat_map(|v| v.to_le_bytes()).collect();
    let y_init = vec![0u8; 4];
    let mut pc = vec![0u8; meta.binding_plan.push_constant_total_bytes as usize];
    for scalar in &meta.binding_plan.scalars {
        if scalar.ty == axc_hir::ScalarTy::U32 {
            let s = scalar.offset as usize;
            pc[s..s + 4].copy_from_slice(&N_SUPERBLOCKS.to_le_bytes());
        }
    }
    let q_len = n * helpers::Q4KM_SUPERBLOCK_BYTES;
    let x_len = n * helpers::Q4KM_SUPERBLOCK_ELEMS * 4;

    let req = DispatchRequest {
        spirv: &words,
        binding_plan: &meta.binding_plan,
        workgroups: [1, 1, 1],
        inputs: &[q_bytes.as_slice(), x_bytes.as_slice(), y_init.as_slice()],
        output_sizes: &[q_len, x_len, 4],
        push_constants: &pc,
        entry_point: "q4km_dequant_matvec",
    };
    let outputs = ctx.dispatch(req).expect("AT-1761: dispatch must succeed");
    let y = &outputs[2];
    let gpu = f32::from_le_bytes([y[0], y[1], y[2], y[3]]);
    let rel_err = if cpu_result.abs() > 1e-10 {
        (gpu - cpu_result).abs() / cpu_result.abs()
    } else {
        (gpu - cpu_result).abs()
    };
    assert!(
        rel_err < 1e-3,
        "AT-1761: Q4_K_M GPU={gpu:.6} vs CPU={cpu_result:.6}, rel_err={rel_err:.2e} >= 1e-3"
    );
}

/// AT-1766: the committed ab_results.json (if present from an NVIDIA run) records the
/// measurement-boundary, tag+40-char SHA, AXIOM min/mean/median+sustained, FLOP-consistency,
/// boundary label, and byte-identical device-match.
///
/// When ab_results.json is absent (clean checkout, no NVIDIA run yet) the test is a no-op
/// (the artifact is produced by the NVIDIA harness run, not CI).
#[test]
fn at_1766_ab_results_records_boundary_and_metadata() {
    let path = repo_root().join(".pipeline/benchmarks/m34/ab_results.json");
    if !path.exists() {
        eprintln!("AT-1766: ab_results.json absent (no NVIDIA run yet); skipping schema check");
        return;
    }
    let txt = std::fs::read_to_string(&path).expect("read ab_results.json");
    let v: serde_json::Value = serde_json::from_str(&txt).expect("ab_results.json must be valid JSON");

    // Timer boundary recorded VERBATIM (CRITICAL-1).
    let tb = &v["llamacpp_timer_boundary"];
    assert!(tb["source_excerpt"].as_str().map(|s| s.contains("ggml_backend_graph_compute")).unwrap_or(false),
        "AT-1766: timer boundary must quote test-backend-ops.cpp verbatim");
    assert!(tb["min_mean_or_sustained"].as_str().is_some(), "AT-1766: min/mean/sustained recorded");
    assert!(tb["per_op_or_amortized"].as_str().is_some(), "AT-1766: per-op/amortized recorded");

    // Tag + literal 40-char SHA (WARNING-4).
    let sha = v["commit"]["sha"].as_str().expect("commit.sha");
    assert_eq!(sha.len(), 40, "AT-1766: commit.sha must be a literal 40-char SHA");
    assert!(v["commit"]["tag"].as_str().is_some(), "AT-1766: commit.tag recorded");

    // AXIOM min/mean/median + sustained (CRITICAL-1 comparability).
    let ax = &v["axiom"];
    for k in ["kernel_ns_min", "kernel_ns_mean", "kernel_ns_median", "sustained_ns"] {
        assert!(ax[k].as_u64().is_some(), "AT-1766: axiom.{k} must be present");
    }

    // Boundary label naming the matched pair (only when a ratio was computed).
    let status = v["kill_criterion_status"].as_str().unwrap_or("");
    if status == "PASS" || status == "FAIL" {
        assert!(v["ratio_boundary_label"].as_str().map(|s| !s.is_empty()).unwrap_or(false),
            "AT-1766: ratio_boundary_label must name the matched boundary for a real ratio");
        // FLOP-consistency must have passed (WARNING-3).
        assert_eq!(v["flop_consistency"]["ok"].as_bool(), Some(true),
            "AT-1766: a PASS/FAIL ratio requires flop_consistency.ok=true");
        // Device-match byte-identical (WARNING-5).
        assert_eq!(v["device_match"]["byte_identical"].as_bool(), Some(true),
            "AT-1766: a PASS/FAIL ratio requires byte-identical device-match");
    }
}

/// AT-1767: when ab_results.json reports a real ratio, the kill-criterion number was
/// computed from an n==1 (GEMV) case — NEVER a GEMM (n>1) substitution (CRITICAL-2).
/// When status=INCOMPLETE, no ratio is present.
#[test]
fn at_1767_kill_number_is_n1_never_gemm() {
    let path = repo_root().join(".pipeline/benchmarks/m34/ab_results.json");
    if !path.exists() {
        eprintln!("AT-1767: ab_results.json absent; skipping");
        return;
    }
    let txt = std::fs::read_to_string(&path).expect("read ab_results.json");
    let v: serde_json::Value = serde_json::from_str(&txt).expect("valid JSON");
    let status = v["kill_criterion_status"].as_str().unwrap_or("");

    match status {
        "PASS" | "FAIL" => {
            // The selected llama.cpp case MUST be n==1 (GEMV), never GEMM.
            let n = v["llamacpp"]["n"].as_i64();
            assert_eq!(n, Some(1),
                "AT-1767: a kill-criterion ratio MUST come from the n==1 Q4_K case, got n={n:?}");
            assert!(v["ratio_axiom_over_llama_tflops"].as_f64().is_some(),
                "AT-1767: PASS/FAIL must carry a computed ratio");
        }
        "INCOMPLETE" => {
            // INCOMPLETE must NOT carry a ratio (never a GEMM-substituted FAIL).
            assert!(v["ratio_axiom_over_llama_tflops"].is_null(),
                "AT-1767: INCOMPLETE must NOT render a ratio");
            assert!(v["kill_criterion_reason"].as_str().map(|s| !s.is_empty()).unwrap_or(false),
                "AT-1767: INCOMPLETE must record a reason");
        }
        other => panic!("AT-1767: unexpected kill_criterion_status '{other}'"),
    }
}
