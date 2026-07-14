//! M3.21 (FG.9) — `axc optimize` module-wide hole discovery (F3) and winner
//! cross-validation (F4 / r3 orchestrator patch). AT-2957, AT-2970.
//!
//! `run_optimize`'s bench closure is the pre-existing M2.3 mock (fixed
//! 1000ns for every variant, no GPU) — these tests exercise the CPU-only
//! discovery/selection/cross-validation logic, not GPU timing.

use std::path::PathBuf;

use axc_driver::optimize::{run_optimize, OptimizeError};

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

/// AT-2957: `axc optimize multi.axc --kernel bench_target` runs grid_search
/// over the UNION holes (F3), and the winner is cross-compiled across ALL
/// kernels (F4) BEFORE the winner sidecar is written.
#[test]
fn at_2957_optimize_union_holes_and_f4_cross_validate_before_write() {
    let dir = tempfile_dir("at_2957");
    let src = dir.join("multi.axc");
    std::fs::write(&src, SHARED_HOLE_SRC).unwrap();
    let out = dir.join("out.spv");

    let result = run_optimize(&src, &out, "none", Some("bench_target"));
    assert!(result.is_ok(), "optimize must succeed: {result:?}");
    assert!(out.exists(), "winning .spv must be written");
    let sidecar = axc_driver::optimize::strategy_sidecar_path(&out);
    assert!(sidecar.exists(), "strategy sidecar must be written");

    let sidecar_text = std::fs::read_to_string(&sidecar).unwrap();
    // The union has one axis (WG ∈ {32,64}); winner_assignments must carry it.
    assert!(sidecar_text.contains("\"WG\""), "sidecar must record the WG assignment: {sidecar_text}");

    let _ = std::fs::remove_dir_all(&dir);
}

/// `axc optimize` with no `--kernel` on an ambiguous (2+ kernel) source
/// fails closed (never a silent kernels[0] pick).
#[test]
fn optimize_no_kernel_on_ambiguous_source_fails_closed() {
    let dir = tempfile_dir("optimize_ambiguous");
    let src = dir.join("multi.axc");
    std::fs::write(&src, SHARED_HOLE_SRC).unwrap();
    let out = dir.join("out.spv");

    let result = run_optimize(&src, &out, "none", None);
    match result {
        Err(OptimizeError::Compile(axc_driver::DriverError::AmbiguousKernel { available })) => {
            assert_eq!(available, vec!["bench_target".to_string(), "sibling".to_string()]);
        }
        other => panic!("expected AmbiguousKernel; got {other:?}"),
    }
    assert!(!out.exists(), "no output on an ambiguous bench-target selection");

    let _ = std::fs::remove_dir_all(&dir);
}

/// AT-2970 / r3: a candidate that is valid for the BENCHED kernel but makes a
/// SIBLING kernel fail to compile -> `run_optimize` fails closed BEFORE
/// writing EITHER the `.spv` or the strategy sidecar (r3 orchestrator patch —
/// zero partial artifacts on an injected sibling failure).
///
/// `bench_target` uses `?BAD` only as its OWN workgroup x-dim with y=z=1 (a
/// 1-D workgroup, legal for ANY of these candidates); `sibling` multiplies
/// the SAME value by two more dimensions of 16, so `?BAD=8` overflows the
/// portable 1024-invocation ceiling for `sibling` (8*16*16=2048 ->
/// `HirError::BadWorkgroupDim`) while staying perfectly legal for
/// `bench_target` alone (product=8). `?BAD`'s candidate order is `[8, 2]` —
/// the mock bench (`run_optimize`'s fixed-1000ns-for-every-variant closure)
/// always crowns ORDINAL 0 (ties broken by original enumeration order via a
/// stable sort), so this deterministically drives `run_optimize`'s OWN
/// winner selection into the sibling-breaking candidate — a genuine internal
/// F4 firing, not just an external proof that F4 *would* catch it.
#[test]
fn at_2970_winner_invalid_for_sibling_fails_closed_zero_artifacts() {
    let src_text = concat!(
        "@kernel\n",
        "@workgroup(?BAD, 1, 1)\n",
        "@strategy { BAD: ?[8, 2] }\n",
        "@intent(\"bench target: legal for both candidates as a 1-D workgroup\")\n",
        "@complexity(O(n))\n",
        "fn bench_target(x: buffer[u32]) -> void {\n",
        "    let i: u32 = gid(0u32);\n",
        "    x[i] = i;\n",
        "    return;\n",
        "}\n",
        "@kernel\n",
        "@workgroup(?BAD, 16, 16)\n",
        "@strategy { BAD: ?[8, 2] }\n",
        "@intent(\"sibling: BAD=8 overflows the 1024-invocation ceiling (8*16*16=2048)\")\n",
        "@complexity(O(n))\n",
        "fn sibling(y: buffer[u32]) -> void {\n",
        "    let i: u32 = gid(0u32);\n",
        "    y[i] = i;\n",
        "    return;\n",
        "}\n",
    );
    let dir = tempfile_dir("at_2970");
    let src = dir.join("multi.axc");
    std::fs::write(&src, src_text).unwrap();
    let out = dir.join("out.spv");
    let sidecar = axc_driver::optimize::strategy_sidecar_path(&out);

    // Sanity: confirm BAD=8 really does break `sibling` (the scenario F4
    // exists to catch) and BAD=2 does not — otherwise this fixture would be
    // vacuous.
    let mut bad_first = std::collections::BTreeMap::new();
    bad_first.insert("BAD".to_string(), 8i64);
    let resolved_first = axc_driver::substitute_strategy_holes(src_text, &bad_first);
    assert!(
        axc_driver::compile_module_all(&resolved_first, false).is_err(),
        "fixture sanity: BAD=8 must break `sibling` module-wide"
    );
    let mut bad_second = std::collections::BTreeMap::new();
    bad_second.insert("BAD".to_string(), 2i64);
    let resolved_second = axc_driver::substitute_strategy_holes(src_text, &bad_second);
    assert!(
        axc_driver::compile_module_all(&resolved_second, false).is_ok(),
        "fixture sanity: BAD=2 must be valid module-wide"
    );

    // The actual F4 firing, inside run_optimize's own winner-selection path.
    let result = run_optimize(&src, &out, "none", Some("bench_target"));
    match result {
        Err(OptimizeError::WinnerInvalidModuleWide { broken_kernel, detail }) => {
            assert_eq!(broken_kernel.as_deref(), Some("sibling"), "F4 must name the broken sibling; detail={detail}");
        }
        other => panic!("expected WinnerInvalidModuleWide naming `sibling`; got {other:?}"),
    }

    // r3: NEITHER the .spv NOR the sidecar may exist after an F4 failure.
    assert!(!out.exists(), "AT-2970 (r3): NO .spv may be written on an F4 failure");
    assert!(!sidecar.exists(), "AT-2970 (r3): NO strategy sidecar may be written on an F4 failure");

    let _ = std::fs::remove_dir_all(&dir);
}
