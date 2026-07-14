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

// ── M3.19 (FG.2/FG.3): doc-fence + codegen grep-fence ──────────────────────────

/// AT-2917: the FG.2 `@transfer` revisit trigger is recorded verbatim in
/// DESIGN.md `## 3.1.43` AND in ROADMAP.md's FG.2 row — both copies must
/// name the SAME two concrete triggering conditions (an MCP consumer tool,
/// or the 7-agent supervisor consuming in-source `@transfer`).
#[test]
fn at_2917_transfer_revisit_trigger_recorded_verbatim() {
    let root: PathBuf = repo_root();

    let design_md: String = std::fs::read_to_string(root.join("DESIGN.md"))
        .expect("AT-2917: DESIGN.md must exist");
    assert!(
        design_md.contains("## 3.1.43 M3.19"),
        "AT-2917: DESIGN.md must contain the `## 3.1.43 M3.19` subsection"
    );
    assert!(
        design_md.contains("agent-orchestration MCP tool (a 9th tool, e.g. `next_step`/`route`)"),
        "AT-2917: DESIGN.md §3.1.43 must record the exact MCP-consumer revisit trigger"
    );
    assert!(
        design_md.contains("7-agent supervisor is changed to consume in-source `@transfer` handoff"),
        "AT-2917: DESIGN.md §3.1.43 must record the exact supervisor-consumption revisit trigger"
    );

    let roadmap_md: String = std::fs::read_to_string(root.join("ROADMAP.md"))
        .expect("AT-2917: ROADMAP.md must exist");
    assert!(
        roadmap_md.contains("FG.2") && roadmap_md.contains("NO-BUILD"),
        "AT-2917: ROADMAP.md's FG.2 row must record the NO-BUILD verdict"
    );
    assert!(
        roadmap_md.contains("agent-orchestration MCP tool (a 9th tool, e.g. `next_step`/`route`)"),
        "AT-2917: ROADMAP.md's FG.2 row must record the exact MCP-consumer revisit trigger"
    );
    assert!(
        roadmap_md.contains("7-agent supervisor is changed to consume in-source `@transfer` handoff"),
        "AT-2917: ROADMAP.md's FG.2 row must record the exact supervisor-consumption revisit trigger"
    );
}

/// AT-2918: whole-milestone codegen invariance. `crates/axc-codegen/src` has
/// an EMPTY diff against the pre-M3.19 tree, with ONE documented mechanical
/// exception: the single `opt_log: None,` line forced into a `#[cfg(test)]`
/// `KernelAnnotations` literal by the new field on the shared HIR struct
/// (test-only, non-behavioral). Complements — does NOT substitute for — the
/// real binding proof, AT-2903's byte-identical golden SPIR-V gate.
///
/// NOTE (mirrors AT-2607's own documented caveat): this diffs the CURRENT
/// working tree against `HEAD` (the commit this milestone branched from,
/// since the Coder does not commit). Once this milestone is committed, a
/// later reviewer should consider re-pinning this to fixed base/head SHAs
/// the way AT-2607 does for M3.11a, so the fence keeps working after `HEAD`
/// moves.
#[test]
fn at_2918_codegen_empty_diff_except_one_mechanical_test_line() {
    let root: PathBuf = repo_root();
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(&root)
        .args(["diff", "-U0", "--", "crates/axc-codegen/src"])
        .output()
        .expect("AT-2918: failed to run git diff");
    assert!(output.status.success(), "AT-2918: git diff failed: {:?}", output.status);

    let diff_text: String = String::from_utf8_lossy(&output.stdout).to_string();
    // Every ADDED line (starting with a single `+`, not `+++`) must be
    // exactly the one mechanical field addition; every REMOVED line
    // (starting with a single `-`, not `---`) must not exist at all (this
    // milestone only ADDS the new field, never removes codegen lines).
    for line in diff_text.lines() {
        if line.starts_with("+++") || line.starts_with("---") {
            continue;
        }
        if let Some(added) = line.strip_prefix('+') {
            assert_eq!(
                added.trim(),
                "opt_log: None,",
                "AT-2918: crates/axc-codegen/src must have an EMPTY diff except the one \
                 mechanical `opt_log: None,` test-fixture line; found an unexpected addition: {added:?}"
            );
        } else if line.starts_with('-') {
            panic!("AT-2918: crates/axc-codegen/src must have ZERO removed lines; found: {line:?}");
        }
    }
}
