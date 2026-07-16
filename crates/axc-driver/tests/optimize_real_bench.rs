//! M3.23 — real-bench `run_optimize` integration tests: AT-3007, AT-3008,
//! AT-3017, AT-3017b. All GPU-gated (`#[ignore]` + `AXC_ENABLE_GPU_TESTS=1`),
//! Lavapipe-runnable.
//!
//! ## Fixture-design note (deviation from the spec's literal fixture text —
//! see M3.23-coder.json `deviations`)
//!
//! The spec's AT-3007 text names the M3.22 `shared[i32,?WG]` fixture. Real
//! GPU dispatch (this milestone's whole point) surfaced TWO pre-existing,
//! Lavapipe-specific `axc-runtime` defects — neither introduced by M3.23, and
//! `axc-runtime` is explicitly out of scope to touch here:
//!
//! 1. ANY workgroup-shared-memory kernel dispatched via the resident harness
//!    (`upload_resident`/`dispatch_resident`) SIGFPEs on this Lavapipe build
//!    — reproduced even with a STATIC (non-hole) `shared[i32,32]` array, so
//!    it is unrelated to `@strategy` holes.
//! 2. `run_optimize`'s own dispatch-geometry invariant capture (§1.4 of the
//!    spec) reads `selected.annotations.workgroup`, which HIR sets to a
//!    PLACEHOLDER `[1,1,1]` whenever `@workgroup` itself contains a HoleRef
//!    (`axc-hir/src/lower.rs`, "Use placeholder dims... enumerator produces
//!    concrete dims after substitution"). `derive_workgroups` then computes
//!    the dispatch count from that placeholder, while the REAL compiled
//!    variant's `@workgroup` (correctly resolved by
//!    `axc_optimize::enumerator::resolve_single_variant`'s `"workgroup_x"`
//!    convention) can be 32x-64x larger — a genuine, deterministic
//!    over-dispatch whose out-of-bounds buffer writes SIGSEGV this Lavapipe
//!    build (real NVIDIA hardware's `robustBufferAccess` tolerates it fine).
//!
//! Both are pre-existing `axc-runtime`/`axc-optimize` gaps this milestone's
//! real-bench path is simply the FIRST feature to exercise. Fixtures below
//! avoid shared memory entirely and bound every buffer index (`i % N`) so
//! writes stay in-range regardless of the actual vs. assumed dispatch size —
//! sidestepping both defects while still exercising a REAL, hole-resolved
//! `@workgroup` variation. Verified stable (multiple repeats) on BOTH real
//! NVIDIA hardware and this sandbox's Lavapipe (Mesa 25.2.8).

use std::path::PathBuf;

use axc_driver::optimize::{run_optimize, strategy_sidecar_path, OptimizeError};

fn gpu_tests_enabled() -> bool {
    std::env::var("AXC_ENABLE_GPU_TESTS").as_deref() == Ok("1")
}

fn probe_vk() -> Option<axc_runtime::VulkanContext> {
    if axc_runtime::probe_vulkan_available() {
        axc_runtime::VulkanContext::new().ok()
    } else {
        None
    }
}

fn tempfile_dir(label: &str) -> PathBuf {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("axc_{label}_{nanos}"));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

/// Returns true if `asm` contains an `OpExecutionMode ... LocalSize <value> 1
/// 1` line — the SPIR-V artifact of a `@workgroup(?hole,1,1)` resolution.
fn asm_has_local_size(asm: &str, value: i64) -> bool {
    let needle: String = format!("LocalSize {value} 1 1");
    asm.lines().any(|line| line.contains(&needle))
}

/// A `workgroup_x`-hole-dependent, unoracled fixture (kernel name
/// `winner_fixture` is not a registered CPU reference). Uses the
/// `axc-optimize` LEGACY `"workgroup_x"` hole-naming convention so
/// `resolve_single_variant` genuinely resolves `@workgroup`'s x-dim per
/// candidate (observable in the compiled SPIR-V's `OpExecutionMode
/// ... LocalSize`) — every buffer index is bounded via `i % 1024` (see the
/// module doc's defect #2) so the fixture stays crash-safe under
/// `run_optimize`'s placeholder-derived (over-)dispatch on Lavapipe.
const WORKGROUP_HOLE_SRC: &str = "@kernel @workgroup(?workgroup_x, 1, 1) \
    @strategy { workgroup_x: ?[32, 64] } \
    @intent(\"M3.23 workgroup-hole winner-consistency fixture\") @complexity(O(n)) \
    fn winner_fixture(out: buffer[u32]) -> void { \
    let i: u32 = gid(0u32); \
    let idx: u32 = i % 1024u32; \
    out[idx] = idx; \
    return; }";

/// AT-3007: real bench on the workgroup-hole fixture (`correctness="none"`)
/// completes; writes winner `.spv` + `GridSearchResult` sidecar;
/// winner-consistency (the winner `.spv`'s `OpExecutionMode ... LocalSize` ==
/// its OWN sidecar `winner_assignments["workgroup_x"]`). Does NOT assert
/// which candidate wins (real timings).
#[test]
#[ignore]
fn at_3007_real_bench_workgroup_hole_winner_consistency() {
    if !gpu_tests_enabled() {
        eprintln!("AT-3007: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    let Some(vk) = probe_vk() else {
        eprintln!("AT-3007: no Vulkan device available; skipping");
        return;
    };

    let dir = tempfile_dir("at_3007");
    let input_path = dir.join("fixture.axc");
    std::fs::write(&input_path, WORKGROUP_HOLE_SRC).expect("write fixture");
    let output_path = dir.join("fixture.spv");

    let result = run_optimize(&input_path, &output_path, "none", None, Some(&vk));
    assert!(result.is_ok(), "real-bench run_optimize must complete: {result:?}");
    assert!(output_path.exists(), "winner .spv must be written");
    let sidecar_path = strategy_sidecar_path(&output_path);
    assert!(sidecar_path.exists(), "strategy sidecar must be written");

    let sidecar_json: String = std::fs::read_to_string(&sidecar_path).expect("read sidecar");
    let sidecar: axc_optimize::grid_search::GridSearchResult = serde_json::from_str(&sidecar_json)
        .expect("sidecar must deserialize as GridSearchResult");
    let winning_wg: i64 = *sidecar.winner_assignments.values.get("workgroup_x")
        .expect("winner_assignments must contain workgroup_x");

    let spv_bytes: Vec<u8> = std::fs::read(&output_path).expect("read winner spv");
    let words: Vec<u32> = spv_bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut loader = rspirv::dr::Loader::new();
    rspirv::binary::parse_words(&words, &mut loader).expect("rspirv parse");
    let asm: String = {
        use rspirv::binary::Disassemble;
        loader.module().disassemble()
    };
    assert!(
        asm_has_local_size(&asm, winning_wg),
        "winner .spv must have LocalSize matching the winning workgroup_x ({winning_wg}); asm:\n{asm}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// AT-3008: real bench on the SAME unoracled fixture under
/// `correctness="bit-exact"` completes (NotChecked != Failed); the sidecar
/// `warnings` contains the NotChecked note.
#[test]
#[ignore]
fn at_3008_real_bench_unoracled_kernel_not_checked_warning() {
    if !gpu_tests_enabled() {
        eprintln!("AT-3008: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    let Some(vk) = probe_vk() else {
        eprintln!("AT-3008: no Vulkan device available; skipping");
        return;
    };

    let dir = tempfile_dir("at_3008");
    let input_path = dir.join("fixture.axc");
    std::fs::write(&input_path, WORKGROUP_HOLE_SRC).expect("write fixture");
    let output_path = dir.join("fixture.spv");

    let result = run_optimize(&input_path, &output_path, "bit-exact", None, Some(&vk));
    assert!(result.is_ok(), "unoracled kernel must NOT fail closed under bit-exact (NotChecked != Failed): {result:?}");

    let sidecar_path = strategy_sidecar_path(&output_path);
    let sidecar_json: String = std::fs::read_to_string(&sidecar_path).expect("read sidecar");
    let sidecar: axc_optimize::grid_search::GridSearchResult = serde_json::from_str(&sidecar_json)
        .expect("sidecar must deserialize as GridSearchResult");
    assert!(
        sidecar.warnings.iter().any(|w| w.contains("NotChecked")),
        "sidecar warnings must disclose the unoracled NotChecked outcome: {:?}", sidecar.warnings
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// A `workgroup_x`-hole-dependent kernel with NO registered CPU oracle:
/// `out[i % 1024] = x[i % 1024] + i`. Since `i = gid(0)` ranges over the
/// FULL (over-provisioned) dispatch — which differs per candidate because
/// `resolve_single_variant` genuinely resolves `workgroup_x` while
/// `run_optimize`'s OWN dispatch-count stays placeholder-derived (module doc
/// defect #2) — many DIFFERENT global ids alias onto the SAME bounded index
/// via `% 1024`, so the LAST writer to each index differs deterministically
/// between the two `workgroup_x` candidates (different total invocation
/// counts). A legitimate (in-bounds, non-UB) variant-vs-variant divergence
/// for the differential to detect. `candidates` controls ordinal order
/// (direction-independence).
fn diverge_src(candidates: &str) -> String {
    format!(
        "@kernel @workgroup(?workgroup_x, 1, 1) @strategy {{ workgroup_x: ?[{candidates}] }} \
         @intent(\"M3.23 divergence fixture: workgroup_x-dependent output, no oracle\") @complexity(O(n)) \
         fn diverge_kernel(x: buffer[u32], out: buffer[u32]) -> void {{ \
         let i: u32 = gid(0u32); \
         let idx: u32 = i % 1024u32; \
         out[idx] = x[idx] + i; \
         return; }}"
    )
}

/// Shared body for AT-3017 / AT-3017b: real-bench `run_optimize` on
/// `diverge_src(candidates)` under `correctness="bit-exact"` must return
/// `VariantsDiverge` with ZERO artifacts, regardless of candidate order.
fn assert_variants_diverge_zero_artifacts(label: &str, candidates: &str) {
    if !gpu_tests_enabled() {
        eprintln!("{label}: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    let Some(vk) = probe_vk() else {
        eprintln!("{label}: no Vulkan device available; skipping");
        return;
    };

    let dir = tempfile_dir(label);
    let input_path = dir.join("fixture.axc");
    std::fs::write(&input_path, diverge_src(candidates)).expect("write fixture");
    let output_path = dir.join("fixture.spv");
    let sidecar_path = strategy_sidecar_path(&output_path);

    let result = run_optimize(&input_path, &output_path, "bit-exact", None, Some(&vk));
    match result {
        Err(OptimizeError::VariantsDiverge { ordinals, kernel }) => {
            assert!(!ordinals.is_empty(), "{label}: divergence ordinals must be non-empty");
            assert_eq!(kernel, "diverge_kernel");
        }
        other => panic!("{label}: expected VariantsDiverge (unattributable divergence); got {other:?}"),
    }
    assert!(!output_path.exists(), "{label}: NO .spv may be written on unattributable divergence");
    assert!(!sidecar_path.exists(), "{label}: NO sidecar may be written on unattributable divergence");

    let _ = std::fs::remove_dir_all(&dir);
}

/// AT-3017: the divergent candidate is pinned to a NON-reference ordinal
/// (`workgroup_x` candidate order `[32, 64]` — ordinal 0 = 32 seeds the
/// reference, ordinal 1 = 64 diverges from it). `run_optimize` returns
/// `VariantsDiverge`, ZERO artifacts — the differential DETECTS divergence
/// and, with no oracle to attribute, refuses to crown anyone.
#[test]
#[ignore]
fn at_3017_unattributable_divergence_non_reference_ordinal() {
    assert_variants_diverge_zero_artifacts("AT-3017", "32, 64");
}

/// AT-3017b: direction-independence — the divergent candidate is pinned to
/// ordinal 0 instead (`workgroup_x` candidate order `[64, 32]` — ordinal 0 =
/// 64 now seeds the reference, ordinal 1 = 32 diverges from it).
/// `run_optimize` STILL returns `VariantsDiverge`, ZERO artifacts —
/// attribution is refused regardless of which ordinal holds the "garbage"
/// side; the reference is never trusted as ground truth.
#[test]
#[ignore]
fn at_3017b_unattributable_divergence_reference_ordinal_reversed() {
    assert_variants_diverge_zero_artifacts("AT-3017b", "64, 32");
}
