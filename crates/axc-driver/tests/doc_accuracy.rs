//! M3.15 (Item 6): doc-accuracy sweep guard.
//!
//! Two ROADMAP polish rows are already resolved in the current tree (the stale
//! `Float64 cap = 6` comment and the stale BENCHMARKS.md "(M2.2)" heading) —
//! this is a cheap regression fence so they don't silently regress, plus a
//! guard for the M3.15 freshness note and the CLAUDE.md Lavapipe ICD-name
//! environment-dependence note (AT-2840). Also covers the CHANGELOG.md
//! milestone-level backfill presence guard (AT-2839).

use std::path::PathBuf;

/// Absolute path to the repo root (two levels up from this crate's manifest dir:
/// `crates/axc-driver` → repo root).
fn repo_root() -> PathBuf {
    let manifest_dir: PathBuf = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"),
    );
    manifest_dir.join("..").join("..")
}

/// AT-2839: `CHANGELOG.md` exists at repo root, is non-empty, contains an
/// `M3.14` bullet and an `M3.6` bullet (spot-check that the backfill covers
/// both a recent and a mid milestone), and mentions `M3.15`.
#[test]
fn at_2839_changelog_exists_and_covers_backfill_range() {
    let path: PathBuf = repo_root().join("CHANGELOG.md");
    let content: String = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("AT-2839: CHANGELOG.md must exist at repo root ({path:?}): {e}"));
    assert!(!content.trim().is_empty(), "AT-2839: CHANGELOG.md must be non-empty");
    assert!(
        content.contains("M3.14"),
        "AT-2839: CHANGELOG.md must contain an M3.14 bullet (recent milestone spot-check)"
    );
    assert!(
        content.contains("M3.6"),
        "AT-2839: CHANGELOG.md must contain an M3.6 bullet (mid milestone spot-check)"
    );
    assert!(
        content.contains("M3.15"),
        "AT-2839: CHANGELOG.md must mention M3.15 (this bundle's own entry)"
    );
}

/// AT-2840: doc-accuracy grep-fence guard.
///
/// 1. `crates/axc-codegen/src/body.rs` must contain no literal `cap = 6` and
///    no `Float64 cap = 6` (the stale ROADMAP polish row — verified already
///    resolved; SPIR-V §3.31 Float64 capability value is 10, not 6).
/// 2. `BENCHMARKS.md` must contain the `Last updated: M3.15` freshness note.
/// 3. `CLAUDE.md` must contain BOTH `lvp_icd.x86_64.json` (the CI-canonical
///    name `ci.yml` depends on — must NOT be removed) AND `lvp_icd.json` (the
///    box-local alternate named by the environment-dependent note). Note
///    `lvp_icd.json` is not a substring of `lvp_icd.x86_64.json`, so a plain
///    `.contains("lvp_icd.json")` only passes when the env-note is actually
///    present — this asserts PRESENCE of the both-names note (option B), not
///    absence of the CI-canonical string.
#[test]
fn at_2840_doc_accuracy_grep_fence() {
    let root: PathBuf = repo_root();

    // ── 1. body.rs: no stale `Float64 cap = 6` ──────────────────────────────
    let body_rs_path: PathBuf = root.join("crates").join("axc-codegen").join("src").join("body.rs");
    let body_rs: String = std::fs::read_to_string(&body_rs_path)
        .unwrap_or_else(|e| panic!("AT-2840: failed to read {body_rs_path:?}: {e}"));
    assert!(
        !body_rs.contains("cap = 6"),
        "AT-2840: {body_rs_path:?} must not contain the stale literal `cap = 6`"
    );
    assert!(
        !body_rs.contains("Float64 cap = 6"),
        "AT-2840: {body_rs_path:?} must not contain the stale `Float64 cap = 6` comment \
         (SPIR-V §3.31 Float64 capability value is 10)"
    );

    // ── 2. BENCHMARKS.md: freshness note present ────────────────────────────
    let benchmarks_md_path: PathBuf = root.join("BENCHMARKS.md");
    let benchmarks_md: String = std::fs::read_to_string(&benchmarks_md_path)
        .unwrap_or_else(|e| panic!("AT-2840: failed to read {benchmarks_md_path:?}: {e}"));
    assert!(
        benchmarks_md.contains("Last updated: M3.15"),
        "AT-2840: BENCHMARKS.md must contain the 'Last updated: M3.15' freshness note"
    );

    // ── 3. CLAUDE.md: both Lavapipe ICD names present (env-dependence note) ─
    let claude_md_path: PathBuf = root.join("CLAUDE.md");
    let claude_md: String = std::fs::read_to_string(&claude_md_path)
        .unwrap_or_else(|e| panic!("AT-2840: failed to read {claude_md_path:?}: {e}"));
    assert!(
        claude_md.contains("lvp_icd.x86_64.json"),
        "AT-2840: CLAUDE.md must still contain `lvp_icd.x86_64.json` (the CI-canonical name \
         `ci.yml` depends on — must NOT be removed)"
    );
    assert!(
        claude_md.contains("lvp_icd.json"),
        "AT-2840: CLAUDE.md must contain `lvp_icd.json` (the box-local alternate name), proving \
         the environment-dependent ICD-name note is present"
    );
}
