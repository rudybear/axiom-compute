//! M3.15 (EB.4): committed `.pipeline/benchmarks/baselines.json` round-trip +
//! two-machine-shape invariant test.
//!
//! Reuses the shared v2 structs from `benches/common.rs` (the SAME definitions
//! the writer, `postprocess.rs`, and the gate, `bench_regression.rs`, use — see
//! that file's header comment), mirroring `bench_regression.rs`'s
//! `#[path]`-include parser-reuse pattern so this test cannot drift from the
//! live schema.

#[path = "../benches/common.rs"]
mod common;

use common::{sanitize, BaselinesV2};
use std::path::PathBuf;

/// Absolute path to the committed baselines file.
fn baselines_path() -> PathBuf {
    let manifest_dir: PathBuf = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"),
    );
    manifest_dir
        .join("..")
        .join("..")
        .join(".pipeline")
        .join("benchmarks")
        .join("baselines.json")
}

fn load_baselines() -> BaselinesV2 {
    let path: PathBuf = baselines_path();
    let raw: String = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read committed baselines at {path:?}: {e}"));
    serde_json::from_str::<BaselinesV2>(&raw)
        .unwrap_or_else(|e| panic!("committed baselines at {path:?} do not parse as schema v2: {e}"))
}

/// AT-2836: round-trip stability + shape.
///
/// - `schema_version == 2`.
/// - Every machine block is non-empty, and every entry has `median_ns > 0` and
///   `low_ns <= median_ns <= high_ns`.
/// - `serialize -> deserialize -> serialize -> deserialize` is structurally
///   stable: the twice-deserialized document equals the once-deserialized one
///   (the "pruned file round-trips" gate).
#[test]
fn at_2836_baselines_roundtrip_stability_and_shape() {
    let doc: BaselinesV2 = load_baselines();

    assert_eq!(doc.schema_version, 2, "committed baselines must be schema_version 2");

    assert!(
        !doc.machines.is_empty(),
        "committed baselines must have at least one machine block"
    );

    for (key, block) in &doc.machines {
        assert!(
            !block.benchmarks.is_empty(),
            "machine block {key:?} must have at least one benchmark entry"
        );
        for entry in &block.benchmarks {
            assert!(
                entry.median_ns > 0,
                "machine {key:?} bench {:?}/{:?}: median_ns must be > 0, got {}",
                entry.group, entry.bench, entry.median_ns
            );
            assert!(
                entry.low_ns <= entry.median_ns && entry.median_ns <= entry.high_ns,
                "machine {key:?} bench {:?}/{:?}: expected low_ns <= median_ns <= high_ns, \
                 got low={} median={} high={}",
                entry.group, entry.bench, entry.low_ns, entry.median_ns, entry.high_ns
            );
        }
    }

    // Round-trip: serialize -> reparse -> serialize -> reparse must yield an
    // equal BaselinesV2 (structural stability of the pruned file).
    let serialized_once: String = serde_json::to_string_pretty(&doc)
        .expect("BaselinesV2 must serialize");
    let reparsed_once: BaselinesV2 = serde_json::from_str(&serialized_once)
        .expect("serialized BaselinesV2 must reparse");
    let serialized_twice: String = serde_json::to_string_pretty(&reparsed_once)
        .expect("reparsed BaselinesV2 must re-serialize");
    let reparsed_twice: BaselinesV2 = serde_json::from_str(&serialized_twice)
        .expect("twice-serialized BaselinesV2 must reparse");

    assert_eq!(
        reparsed_once, reparsed_twice,
        "serialize -> deserialize -> serialize -> deserialize must be structurally stable"
    );
    assert_eq!(
        doc, reparsed_once,
        "the committed file's parsed form must equal its own reparse (no lossy round-trip)"
    );
}

/// AT-2837: prune / expected-key invariant.
///
/// Asserts the committed file holds EXACTLY two machine keys: one
/// `nvidia_rtx_pro_6000_blackwell_workstation_edition` (exact match) and one
/// `llvmpipe*`-prefixed key (checked by prefix to stay robust to the
/// LLVM-version-dependent suffix Lavapipe's `deviceName` carries). Also
/// asserts every key equals its own `sanitize()` (no unsanitized/legacy key
/// leaked).
///
/// Per the M3.15 spec (r2 risk-flag resolution): this assertion is GREEN
/// before merge because the correct `lvp_icd.json` ICD was verified present on
/// this box and the Lavapipe bless was run and landed successfully. The
/// `len()==2` / `starts_with("llvmpipe")` clauses must NEVER be weakened to
/// `>= 1` — see the coder run JSON for the executed bless commands.
#[test]
fn at_2837_baselines_exactly_two_machine_keys() {
    let doc: BaselinesV2 = load_baselines();

    assert_eq!(
        doc.machines.len(),
        2,
        "expected EXACTLY two machine keys (one nvidia_*, one llvmpipe_*); got {:?}",
        doc.machines.keys().collect::<Vec<_>>()
    );

    assert!(
        doc.machines.contains_key("nvidia_rtx_pro_6000_blackwell_workstation_edition"),
        "expected the nvidia_rtx_pro_6000_blackwell_workstation_edition key to be present; got {:?}",
        doc.machines.keys().collect::<Vec<_>>()
    );

    let llvmpipe_keys: Vec<&String> = doc
        .machines
        .keys()
        .filter(|k| k.starts_with("llvmpipe"))
        .collect();
    assert_eq!(
        llvmpipe_keys.len(),
        1,
        "expected exactly one llvmpipe*-prefixed key; got {:?}",
        doc.machines.keys().collect::<Vec<_>>()
    );

    for key in doc.machines.keys() {
        assert_eq!(
            key, &sanitize(key),
            "machine key {key:?} must equal its own sanitize() output (no unsanitized/legacy key)"
        );
    }
}
