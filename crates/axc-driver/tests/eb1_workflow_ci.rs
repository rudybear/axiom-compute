//! EB.1-prep — CPU-only structural/security validation of the cross-vendor CI
//! workflow files + `docs/EB1_RUNNER_SETUP.md` (AT-3019..3035).
//!
//! Runs on ordinary `ubuntu-latest` CI with no GPU. It is the CPU-only
//! validatable half of EB.1: it proves the workflow YAML is well-formed and
//! carries the activation/fork-guard/least-privilege/timeout invariants the
//! milestone spec requires — it does NOT (and cannot) prove the vendor jobs
//! actually pass on real AMD/Intel hardware, which is EB.1-proper.
//!
//! ## §6.3 no-vacuous-pass contract (AT-3033)
//!
//! The YAML parser (`saphyr`, a `[dev-dependencies]`-only crate — zero
//! `crates/axc-*/src` touches) is imported unconditionally at the top of this
//! file. There is deliberately **no** `#[cfg(feature = ...)]` gate around it:
//! if the crate is unavailable, the whole test binary fails to COMPILE (a
//! harder failure than a runtime skip). Every `Yaml::load_from_str` call below
//! `.unwrap_or_else(|e| panic!(...))`s on a parse error — a malformed workflow
//! file panics the test, it never silently no-ops. This guarantees the
//! security assertions (AT-3020/3021/3024/3025/3026/3032) can never compile or
//! run down to a vacuous green pass.
//!
//! ## AT-3032 / COND-1 / COND-2 (the regression-proof security scan)
//!
//! [`scan_self_hosted_jobs`] enumerates every job across a **glob** of
//! `.github/workflows/*.yml` (not a hardcoded file list — see
//! [`all_workflow_files`]), fails CLOSED on any job whose `runs-on:` is not a
//! literal string / literal string-sequence (an expression- or
//! matrix-derived `runs-on:` is a hard test failure demanding literal labels
//! — COND-1), and cross-checks the resulting self-hosted job count against an
//! INDEPENDENT raw-text line-scan implementation ([`grep_self_hosted_job_count`])
//! that does not go through `saphyr` at all — COND-2. Both counts must agree
//! and both must meet [`KNOWN_SELF_HOSTED_JOB_FLOOR`]; an empty/zero result
//! from either mechanism is a hard failure, not a vacuous pass.

use saphyr::{LoadableYamlNode, Yaml};
use std::path::PathBuf;

// ════════════════════════════════════════════════════════════════════════════
// Shared helpers
// ════════════════════════════════════════════════════════════════════════════

fn repo_root() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("eb1_workflow_ci: CARGO_MANIFEST_DIR not set");
    PathBuf::from(manifest_dir).join("..").join("..")
}

fn read_repo_file(rel: &str) -> String {
    let path = repo_root().join(rel);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("eb1_workflow_ci: cannot read {}: {e}", path.display()))
}

/// Unconditional parse (§6.3 no-vacuous-pass contract) — panics on any parse
/// failure, never a silent skip.
fn parse_yaml_docs<'a>(content: &'a str, label: &str) -> Vec<Yaml<'a>> {
    Yaml::load_from_str(content)
        .unwrap_or_else(|e| panic!("eb1_workflow_ci: {label} failed to parse as YAML: {e}"))
}

fn normalize_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The 3 workflow files this milestone CREATES.
const NEW_WORKFLOW_FILES: [&str; 3] = [
    ".github/workflows/cross-vendor-ci.yml",
    ".github/workflows/cross-vendor-ab.yml",
    ".github/workflows/cross-vendor-bless.yml",
];

const CI_YML: &str = ".github/workflows/ci.yml";

/// The FLOOR count of self-hosted jobs known to exist across the workflow set
/// at the time this milestone was authored: `pytorch-interop` (ci.yml,
/// pre-existing) + `amd-gpu`/`intel-gpu` (cross-vendor-ci.yml) +
/// `ab-amd`/`ab-intel` (cross-vendor-ab.yml) + `bless-amd`/`bless-intel`
/// (cross-vendor-bless.yml) = 7. AT-3032/COND-2: both counting mechanisms
/// must reach at least this floor — a result of 0 (or any count below this)
/// is a hard failure, never a vacuous pass.
const KNOWN_SELF_HOSTED_JOB_FLOOR: usize = 7;

/// Glob `.github/workflows/*.{yml,yaml}` (NOT a hardcoded list) — AT-3032
/// requires every self-hosted job across every workflow file, including any
/// future addition, to be caught by this scan.
///
/// PESS-1 must-fix: GitHub Actions honors BOTH `.yml` and `.yaml` as valid
/// workflow-file extensions (they are fully equivalent to the Actions
/// runner). A `.yml`-only filter let a `.yaml` workflow bypass the entire
/// security suite — proven during code review with a scratch
/// `zz_scratch_hole.yaml` carrying a guardless self-hosted job, which the
/// `.yml`-only scan reported as `test result: ok`. The accepted extension set
/// is now exactly `{yml, yaml}` (see `WORKFLOW_FILE_EXTENSIONS` /
/// [`at_3037_workflow_file_extension_set_is_exactly_yml_or_yaml`]).
const WORKFLOW_FILE_EXTENSIONS: [&str; 2] = ["yml", "yaml"];

fn all_workflow_files() -> Vec<PathBuf> {
    let dir = repo_root().join(".github/workflows");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("eb1_workflow_ci: cannot read {}: {e}", dir.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .is_some_and(|ext| WORKFLOW_FILE_EXTENSIONS.contains(&ext))
        })
        .collect();
    files.sort();
    assert!(
        !files.is_empty(),
        "eb1_workflow_ci: the .github/workflows glob found ZERO .yml/.yaml files — \
         an empty selector must fail, not pass vacuously"
    );
    files
}

fn get_str<'y>(y: &'y Yaml, key: &str) -> Option<&'y str> {
    y.as_mapping_get(key).and_then(|v| v.as_str())
}

fn get_i64(y: &Yaml, key: &str) -> Option<i64> {
    y.as_mapping_get(key).and_then(|v| v.as_integer())
}

/// The top-level `jobs:` mapping as `(name, job-body)` pairs. Panics
/// (never skips) if `jobs:` is absent or not a mapping — part of the
/// AT-3033 no-vacuous-pass meta-assertion.
fn jobs_of<'y>(doc: &'y Yaml) -> Vec<(String, &'y Yaml<'y>)> {
    doc.as_mapping_get("jobs")
        .unwrap_or_else(|| panic!("eb1_workflow_ci: AT-3033: no top-level 'jobs' key"))
        .as_mapping()
        .unwrap_or_else(|| panic!("eb1_workflow_ci: AT-3033: 'jobs' is not a mapping"))
        .iter()
        .map(|(k, v)| (k.as_str().unwrap_or("<non-string-job-key>").to_owned(), v))
        .collect()
}

fn is_workflow_dispatch_only(doc: &Yaml) -> bool {
    match doc.as_mapping_get("on") {
        Some(Yaml::Mapping(m)) => {
            m.len() == 1 && m.keys().next().and_then(|k| k.as_str()) == Some("workflow_dispatch")
        }
        Some(y) => y.as_str() == Some("workflow_dispatch"),
        None => false,
    }
}

/// Literal-runs-on classification (COND-1: fail closed on anything else).
enum RunsOnKind {
    Absent,
    /// A literal string, or literal sequence-of-strings, with no `${{ }}`
    /// expression markers anywhere in it.
    Literal(Vec<String>),
    /// Anything else: an expression (`${{ ... }}`, incl. matrix-derived), a
    /// non-string sequence element, a mapping (`group:`/`labels:` form), etc.
    NonLiteral,
}

fn runs_on_kind(job: &Yaml) -> RunsOnKind {
    match job.as_mapping_get("runs-on") {
        None => RunsOnKind::Absent,
        Some(Yaml::Sequence(seq)) => {
            let mut out = Vec::with_capacity(seq.len());
            for item in seq {
                match item.as_str() {
                    Some(s) if !s.contains("${{") => out.push(s.to_owned()),
                    _ => return RunsOnKind::NonLiteral,
                }
            }
            RunsOnKind::Literal(out)
        }
        Some(y) => match y.as_str() {
            Some(s) if !s.contains("${{") => RunsOnKind::Literal(vec![s.to_owned()]),
            _ => RunsOnKind::NonLiteral,
        },
    }
}

fn permissions_no_broader_than_contents_read(job: &Yaml) -> bool {
    match job.as_mapping_get("permissions").and_then(|p| p.as_mapping()) {
        Some(m) => {
            m.len() == 1
                && m.iter()
                    .all(|(k, v)| k.as_str() == Some("contents") && v.as_str() == Some("read"))
        }
        None => false,
    }
}

/// Independent (non-saphyr) raw-text line scan for `runs-on:` blocks that
/// contain the literal label `self-hosted`, handling both flow-sequence
/// (`runs-on: [self-hosted, foo]`) and block-sequence
/// (`runs-on:\n  - self-hosted\n  - foo`) styles. Deliberately a DIFFERENT
/// algorithm from the saphyr AST walk in [`scan_self_hosted_jobs`] — COND-2's
/// cross-check.
fn grep_self_hosted_job_count(content: &str) -> usize {
    let lines: Vec<&str> = content.lines().collect();
    let mut count = 0usize;
    let mut i = 0usize;
    while i < lines.len() {
        let trimmed = lines[i].trim_start();
        if let Some(rest) = trimmed.strip_prefix("runs-on:") {
            let rest = rest.trim();
            if rest.contains("self-hosted") {
                count += 1;
                i += 1;
                continue;
            }
            if rest.is_empty() {
                let mut j = i + 1;
                let mut found = false;
                while j < lines.len() {
                    let t = lines[j].trim_start();
                    if let Some(item) = t.strip_prefix("- ") {
                        if item.trim().contains("self-hosted") {
                            found = true;
                        }
                        j += 1;
                    } else {
                        break;
                    }
                }
                if found {
                    count += 1;
                }
                i = j;
                continue;
            }
        }
        i += 1;
    }
    count
}

/// Raw-text top-level job-key names under `jobs:` (2-space indent), used as
/// an independent duplicate-key detector (AT-3019): `saphyr`'s underlying
/// `LinkedHashMap` silently overwrites a duplicate key on insert, so a
/// text-level count is required to actually CATCH one.
fn raw_top_level_job_names(content: &str) -> Vec<String> {
    let mut in_jobs = false;
    let mut names = Vec::new();
    for line in content.lines() {
        if line.starts_with("jobs:") {
            in_jobs = true;
            continue;
        }
        if !in_jobs {
            continue;
        }
        if line.is_empty() {
            continue;
        }
        if line.starts_with("  ") && !line.starts_with("    ") {
            let trimmed = line.trim_start();
            if trimmed.starts_with('#') {
                continue;
            }
            let key_part = trimmed.split(':').next().unwrap_or("").trim();
            if !key_part.is_empty() {
                names.push(key_part.to_owned());
            }
        } else if !line.starts_with(' ') {
            break;
        }
    }
    names
}

struct SelfHostedJobInfo {
    file: String,
    job: String,
    has_fork_guard_or_dispatch_only: bool,
    permissions_ok: bool,
    timeout_minutes: Option<i64>,
}

fn scan_self_hosted_jobs() -> Vec<SelfHostedJobInfo> {
    let mut out = Vec::new();
    for path in all_workflow_files() {
        let content = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("eb1_workflow_ci: cannot read {}: {e}", path.display()));
        let rel = path
            .strip_prefix(repo_root())
            .unwrap_or(&path)
            .display()
            .to_string();
        let docs = parse_yaml_docs(&content, &rel);
        assert!(
            !docs.is_empty(),
            "eb1_workflow_ci: AT-3033: {rel} parsed to zero YAML documents"
        );
        let doc = &docs[0];
        let dispatch_only = is_workflow_dispatch_only(doc);
        for (job_name, job) in jobs_of(doc) {
            match runs_on_kind(job) {
                RunsOnKind::NonLiteral => panic!(
                    "eb1_workflow_ci: AT-3032/COND-1 FAIL-CLOSED: job '{job_name}' in {rel} has \
                     a non-literal (expression/matrix-derived, or otherwise unrecognized) \
                     `runs-on:`. Self-hosted jobs on this repo MUST declare a literal \
                     [self-hosted, <label>] list so this security scan can verify the \
                     fork-guard/permissions/timeout invariants — rewrite with literal labels."
                ),
                RunsOnKind::Absent => {}
                RunsOnKind::Literal(labels) => {
                    if labels.iter().any(|l| l == "self-hosted") {
                        let if_str = get_str(job, "if").map(normalize_ws).unwrap_or_default();
                        let fork_guard =
                            if_str.contains("head.repo.full_name == github.repository");
                        out.push(SelfHostedJobInfo {
                            file: rel.clone(),
                            job: job_name.clone(),
                            has_fork_guard_or_dispatch_only: fork_guard || dispatch_only,
                            permissions_ok: permissions_no_broader_than_contents_read(job),
                            timeout_minutes: get_i64(job, "timeout-minutes"),
                        });
                    }
                }
            }
        }
    }
    out
}

// ════════════════════════════════════════════════════════════════════════════
// AT-3019: all 4 files parse as valid YAML; no tabs; no duplicate keys.
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn at_3019_workflow_files_parse_as_valid_yaml() {
    for rel in NEW_WORKFLOW_FILES.iter().chain(std::iter::once(&CI_YML)) {
        let content = read_repo_file(rel);
        assert!(!content.contains('\t'), "AT-3019: {rel} contains a literal tab character");

        let docs = parse_yaml_docs(&content, rel);
        assert!(!docs.is_empty(), "AT-3019: {rel} parsed to zero YAML documents");
        let doc = &docs[0];

        // Duplicate-key detection: saphyr's LinkedHashMap silently overwrites a
        // duplicate key on insert, so compare the parsed job count against an
        // INDEPENDENT raw-text count of top-level job-key lines.
        let raw_names = raw_top_level_job_names(&content);
        let mut sorted = raw_names.clone();
        sorted.sort();
        let mut deduped = sorted.clone();
        deduped.dedup();
        assert_eq!(
            sorted.len(),
            deduped.len(),
            "AT-3019: {rel} has a duplicate top-level job key (raw text scan): {raw_names:?}"
        );
        assert_eq!(
            raw_names.len(),
            jobs_of(doc).len(),
            "AT-3019: {rel} raw job-name count ({}) != saphyr-parsed job count ({}) — a \
             duplicate key may have been silently overwritten by the parser",
            raw_names.len(),
            jobs_of(doc).len()
        );
    }
}

// ════════════════════════════════════════════════════════════════════════════
// AT-3020 / AT-3021: activation-var gate + fork guard on cross-vendor-ci.yml.
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn at_3020_vendor_jobs_have_activation_var_guard() {
    let content = read_repo_file(".github/workflows/cross-vendor-ci.yml");
    let docs = parse_yaml_docs(&content, "cross-vendor-ci.yml");
    let doc = &docs[0];
    let jobs: std::collections::HashMap<String, &Yaml> = jobs_of(doc).into_iter().collect();

    let amd_if = normalize_ws(get_str(jobs["amd-gpu"], "if").expect("amd-gpu must have `if:`"));
    assert!(
        amd_if.contains("vars.AMD_RUNNER_ONLINE == 'true'"),
        "AT-3020: amd-gpu `if:` missing the activation-var guard: {amd_if}"
    );

    let intel_if =
        normalize_ws(get_str(jobs["intel-gpu"], "if").expect("intel-gpu must have `if:`"));
    assert!(
        intel_if.contains("vars.INTEL_RUNNER_ONLINE == 'true'"),
        "AT-3020: intel-gpu `if:` missing the activation-var guard: {intel_if}"
    );
}

#[test]
fn at_3021_vendor_jobs_have_fork_guard() {
    let content = read_repo_file(".github/workflows/cross-vendor-ci.yml");
    let docs = parse_yaml_docs(&content, "cross-vendor-ci.yml");
    let doc = &docs[0];
    let jobs: std::collections::HashMap<String, &Yaml> = jobs_of(doc).into_iter().collect();

    for name in ["amd-gpu", "intel-gpu"] {
        let if_str = normalize_ws(get_str(jobs[name], "if").expect("must have `if:`"));
        assert!(
            if_str.contains("head.repo.full_name == github.repository"),
            "AT-3021: {name} `if:` missing the same-repo fork guard: {if_str}"
        );
    }
}

// ════════════════════════════════════════════════════════════════════════════
// AT-3022: exact runs-on label sets.
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn at_3022_vendor_runs_on_labels_exact() {
    let content = read_repo_file(".github/workflows/cross-vendor-ci.yml");
    let docs = parse_yaml_docs(&content, "cross-vendor-ci.yml");
    let doc = &docs[0];
    let jobs: std::collections::HashMap<String, &Yaml> = jobs_of(doc).into_iter().collect();

    match runs_on_kind(jobs["amd-gpu"]) {
        RunsOnKind::Literal(labels) => {
            assert_eq!(labels, vec!["self-hosted".to_owned(), "amd-rdna3".to_owned()])
        }
        _ => panic!("AT-3022: amd-gpu runs-on is not a literal label list"),
    }
    match runs_on_kind(jobs["intel-gpu"]) {
        RunsOnKind::Literal(labels) => {
            assert_eq!(labels, vec!["self-hosted".to_owned(), "intel-arc".to_owned()])
        }
        _ => panic!("AT-3022: intel-gpu runs-on is not a literal label list"),
    }
}

// ════════════════════════════════════════════════════════════════════════════
// AT-3023: A/B + bless workflows are workflow_dispatch-only.
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn at_3023_ab_and_bless_are_workflow_dispatch_only() {
    for rel in [
        ".github/workflows/cross-vendor-ab.yml",
        ".github/workflows/cross-vendor-bless.yml",
    ] {
        let content = read_repo_file(rel);
        let docs = parse_yaml_docs(&content, rel);
        assert!(
            is_workflow_dispatch_only(&docs[0]),
            "AT-3023: {rel} must trigger on workflow_dispatch ONLY (no push/pull_request)"
        );
    }
}

// ════════════════════════════════════════════════════════════════════════════
// AT-3024: pull_request_target absent from the new workflow files.
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn at_3024_no_pull_request_target_in_new_files() {
    for rel in NEW_WORKFLOW_FILES {
        let content = read_repo_file(rel);
        assert!(
            !content.contains("pull_request_target"),
            "AT-3024: {rel} must never use `pull_request_target` (fork-secrets footgun)"
        );
    }
}

// ════════════════════════════════════════════════════════════════════════════
// AT-3025: least privilege on every new job; no `secrets.*` references.
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn at_3025_new_jobs_least_privilege_no_secrets() {
    for rel in NEW_WORKFLOW_FILES {
        let content = read_repo_file(rel);
        assert!(
            !content.contains("secrets."),
            "AT-3025: {rel} must not reference secrets.* (vendor jobs need none)"
        );

        let docs = parse_yaml_docs(&content, rel);
        for (name, job) in jobs_of(&docs[0]) {
            assert!(
                permissions_no_broader_than_contents_read(job),
                "AT-3025: job '{name}' in {rel} does not declare `permissions: contents: read` \
                 (no broader)"
            );
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// AT-3026: the per-PR bench-regression gate never sets AXC_BLESS_BASELINES: "1".
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn at_3026_cross_vendor_ci_never_blesses() {
    let content = read_repo_file(".github/workflows/cross-vendor-ci.yml");
    assert!(
        !content.contains("AXC_BLESS_BASELINES: \"1\""),
        "AT-3026/AT-714b: cross-vendor-ci.yml (the per-PR gate) must NEVER set \
         AXC_BLESS_BASELINES to \"1\""
    );
    // Sanity: bless.yml legitimately DOES (it's the dedicated, human-triggered,
    // workflow_dispatch-only bless job — not the per-PR gate).
    let bless_content = read_repo_file(".github/workflows/cross-vendor-bless.yml");
    assert!(
        bless_content.contains("AXC_BLESS_BASELINES: \"1\""),
        "sanity: cross-vendor-bless.yml is expected to set AXC_BLESS_BASELINES: \"1\""
    );
}

// ════════════════════════════════════════════════════════════════════════════
// AT-3027: llama.cpp pin matches scripts/m34_llamacpp_ab.sh (M3.4).
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn at_3027_llama_pin_matches_script() {
    let script = read_repo_file("scripts/m34_llamacpp_ab.sh");
    let content = read_repo_file(".github/workflows/cross-vendor-ab.yml");
    let docs = parse_yaml_docs(&content, "cross-vendor-ab.yml");
    let doc = &docs[0];
    let env = doc
        .as_mapping_get("env")
        .expect("cross-vendor-ab.yml must declare a workflow-level env: block");
    let tag = get_str(env, "LLAMACPP_TAG").expect("LLAMACPP_TAG must be set");
    let commit = get_str(env, "LLAMACPP_COMMIT").expect("LLAMACPP_COMMIT must be set");

    assert_eq!(tag, "b9542", "AT-3027: workflow LLAMACPP_TAG must equal the M3.4 pin");
    assert_eq!(
        commit, "6b80c74f285390368b3c99c5e750f19e9b096e98",
        "AT-3027: workflow LLAMACPP_COMMIT must equal the M3.4 pin"
    );
    assert!(
        script.contains(&format!("LLAMACPP_TAG=\"{tag}\"")),
        "AT-3027: scripts/m34_llamacpp_ab.sh does not contain LLAMACPP_TAG=\"{tag}\" — pin drift"
    );
    assert!(
        script.contains(commit),
        "AT-3027: scripts/m34_llamacpp_ab.sh does not contain the pinned SHA {commit} — pin drift"
    );
}

// ════════════════════════════════════════════════════════════════════════════
// AT-3028: vendor GPU-test step parity with the Lavapipe `test` job command.
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn at_3028_vendor_gpu_test_step_parity() {
    let content = read_repo_file(".github/workflows/cross-vendor-ci.yml");
    assert!(
        content.contains("cargo test --workspace -- --ignored"),
        "AT-3028: cross-vendor-ci.yml must invoke `cargo test --workspace -- --ignored`"
    );

    let docs = parse_yaml_docs(&content, "cross-vendor-ci.yml");
    let doc = &docs[0];
    let jobs: std::collections::HashMap<String, &Yaml> = jobs_of(doc).into_iter().collect();

    let amd_env = jobs["amd-gpu"].as_mapping_get("env").expect("amd-gpu env");
    assert_eq!(get_str(amd_env, "AXC_ENABLE_GPU_TESTS"), Some("1"));
    assert!(get_str(amd_env, "VK_DRIVER_FILES").unwrap_or("").contains("radeon_icd"));

    let intel_env = jobs["intel-gpu"].as_mapping_get("env").expect("intel-gpu env");
    assert_eq!(get_str(intel_env, "AXC_ENABLE_GPU_TESTS"), Some("1"));
    assert!(get_str(intel_env, "VK_DRIVER_FILES").unwrap_or("").contains("intel_icd"));
}

// ════════════════════════════════════════════════════════════════════════════
// AT-3029: docs/EB1_RUNNER_SETUP.md completeness anchors.
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn at_3029_docs_completeness_anchors() {
    let raw = read_repo_file("docs/EB1_RUNNER_SETUP.md");
    let doc = normalize_ws(&raw);

    let required_substrings = [
        "RADV",
        "AMDVLK",
        "ANV",
        "amd-rdna3",
        "intel-arc",
        "AMD_RUNNER_ONLINE",
        "workflow_dispatch",
        "Fork-PR security",
        "AXC_BLESS_BASELINES",
        "required status check",
        "paths:",
        "Ephemeral",
        "non-root",
        "REQUIRED",
        "Mesa",
        "25.2.8",
    ];
    for needle in required_substrings {
        assert!(
            doc.contains(needle),
            "AT-3029: docs/EB1_RUNNER_SETUP.md missing required anchor: {needle:?}"
        );
    }
}

// ════════════════════════════════════════════════════════════════════════════
// AT-3030: ci.yml has lint-workflows (actionlint) + the mesa-version probe.
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn at_3030_ci_yml_has_lint_workflows_and_mesa_probe() {
    let content = read_repo_file(CI_YML);
    assert!(content.contains("lint-workflows"), "AT-3030: ci.yml missing the lint-workflows job");
    assert!(content.contains("actionlint"), "AT-3030: lint-workflows must run actionlint");
    assert!(
        content.contains("sha256sum"),
        "AT-3030: lint-workflows must checksum-verify the actionlint download (§6.4)"
    );
    assert!(
        content.contains("mesa-version probe") || content.contains("driverInfo"),
        "AT-3030: ci.yml missing the non-fatal mesa-version probe step"
    );

    let docs = parse_yaml_docs(&content, "ci.yml");
    let jobs: std::collections::HashMap<String, &Yaml> = jobs_of(&docs[0]).into_iter().collect();
    assert!(jobs.contains_key("lint-workflows"), "AT-3030: 'lint-workflows' job must exist");
}

// ════════════════════════════════════════════════════════════════════════════
// AT-3031: vendor-independence linkage — the bench-gate step is byte-identical
// (modulo the ICD env) across Lavapipe (ci.yml) and both vendor jobs.
// ════════════════════════════════════════════════════════════════════════════

fn find_step_run(job: &Yaml, needle: &str) -> Option<String> {
    let steps = job.as_mapping_get("steps")?.as_sequence()?;
    for step in steps {
        if let Some(run) = get_str(step, "run") {
            if run.contains(needle) {
                return Some(normalize_ws(run));
            }
        }
    }
    None
}

#[test]
fn at_3031_vendor_independence_bench_gate_parity() {
    let ci_content = read_repo_file(CI_YML);
    let ci_docs = parse_yaml_docs(&ci_content, "ci.yml");
    let ci_jobs: std::collections::HashMap<String, &Yaml> =
        jobs_of(&ci_docs[0]).into_iter().collect();
    let lavapipe_run = find_step_run(ci_jobs["bench-regression"], "--test bench_regression")
        .expect("ci.yml bench-regression job must have a bench_regression run step");

    let cv_content = read_repo_file(".github/workflows/cross-vendor-ci.yml");
    let cv_docs = parse_yaml_docs(&cv_content, "cross-vendor-ci.yml");
    let cv_jobs: std::collections::HashMap<String, &Yaml> =
        jobs_of(&cv_docs[0]).into_iter().collect();

    for name in ["amd-gpu", "intel-gpu"] {
        let vendor_run = find_step_run(cv_jobs[name], "--test bench_regression")
            .unwrap_or_else(|| panic!("{name} must have a bench_regression run step"));
        assert_eq!(
            vendor_run, lavapipe_run,
            "AT-3031: {name}'s bench-regression run command differs from the Lavapipe \
             ci.yml bench-regression job — vendor-independence requires the SAME entrypoint \
             (device selection is delegated to EB.2's machine_key, not a per-vendor script)"
        );
    }
}

// ════════════════════════════════════════════════════════════════════════════
// AT-3036: every job whose steps invoke `--test bench_regression` has
// AXC_ENABLE_BENCH_REGRESSION: "1" somewhere in its effective env
// (job-level or step-level) — the no-vacuous-gate regression this milestone's
// own review found: bench_regression.rs:209 short-circuits to a silent
// pass without it, and neither AT-3028 nor AT-3031 inspected the enclosing
// env: block (they only checked the `run:` command string / other vars). A
// GLOB over every workflow file (not a hardcoded amd-gpu/intel-gpu list) so
// any future job that runs the bench_regression test inherits the same
// no-vacuous-gate guarantee AT-3032 gives the self-hosted security posture.
// ════════════════════════════════════════════════════════════════════════════

fn find_step<'y>(job: &'y Yaml, needle: &str) -> Option<&'y Yaml<'y>> {
    let steps = job.as_mapping_get("steps")?.as_sequence()?;
    steps.iter().find(|step| get_str(step, "run").is_some_and(|run| run.contains(needle)))
}

/// True if `AXC_ENABLE_BENCH_REGRESSION: "1"` appears in the step's own
/// `env:` block OR the enclosing job's `env:` block (GitHub Actions env
/// resolution: step-level overrides job-level, but either location makes the
/// gate live).
fn bench_regression_enabled(job: &Yaml, step: &Yaml) -> bool {
    let step_val = step.as_mapping_get("env").and_then(|e| get_str(e, "AXC_ENABLE_BENCH_REGRESSION"));
    if step_val == Some("1") {
        return true;
    }
    let job_val = job.as_mapping_get("env").and_then(|e| get_str(e, "AXC_ENABLE_BENCH_REGRESSION"));
    job_val == Some("1")
}

#[test]
fn at_3036_every_bench_regression_step_has_the_enabling_var() {
    let mut checked = 0usize;
    for path in all_workflow_files() {
        let content = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("eb1_workflow_ci: cannot read {}: {e}", path.display()));
        let rel = path.strip_prefix(repo_root()).unwrap_or(&path).display().to_string();
        let docs = parse_yaml_docs(&content, &rel);
        for (job_name, job) in jobs_of(&docs[0]) {
            if let Some(step) = find_step(job, "--test bench_regression") {
                checked += 1;
                assert!(
                    bench_regression_enabled(job, step),
                    "AT-3036: job '{job_name}' in {rel} runs `--test bench_regression` but \
                     neither its job-level nor step-level `env:` sets \
                     AXC_ENABLE_BENCH_REGRESSION: \"1\" — bench_regression.rs:209 short-circuits \
                     to a silent no-op without it, so this step would pass vacuously"
                );
            }
        }
    }
    assert!(
        checked > 0,
        "AT-3036: found ZERO jobs invoking `--test bench_regression` across all workflow \
         files — an empty selector must fail, not pass vacuously"
    );
}

// ════════════════════════════════════════════════════════════════════════════
// AT-3037: the workflow-file extension set all_workflow_files() accepts is
// exactly {yml, yaml} — PESS-1 regression guard (a `.yml`-only filter let a
// `.yaml` workflow bypass the entire security suite; proven by scratch
// injection during code review).
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn at_3037_workflow_file_extension_set_is_exactly_yml_or_yaml() {
    let mut sorted = WORKFLOW_FILE_EXTENSIONS.to_vec();
    sorted.sort_unstable();
    assert_eq!(
        sorted,
        vec!["yaml", "yml"],
        "AT-3037/PESS-1: all_workflow_files()'s accepted extension set must be exactly \
         {{yml, yaml}} — GitHub Actions treats both as valid workflow-file extensions, so \
         narrowing this set re-opens the .yaml security-scan bypass"
    );

    // Positive + negative probes directly against the classifier the scan
    // actually uses, so a future refactor of `all_workflow_files()`'s filter
    // predicate that silently narrows/widens the set trips this test even if
    // no physical .yaml file exists on disk.
    let is_workflow_ext = |ext: &str| WORKFLOW_FILE_EXTENSIONS.contains(&ext);
    assert!(is_workflow_ext("yml"), "AT-3037: '.yml' must be accepted");
    assert!(is_workflow_ext("yaml"), "AT-3037: '.yaml' must be accepted");
    assert!(!is_workflow_ext("yml.bak"), "AT-3037: '.yml.bak' must NOT be accepted");
    assert!(!is_workflow_ext("txt"), "AT-3037: '.txt' must NOT be accepted");
    assert!(!is_workflow_ext(""), "AT-3037: an empty/absent extension must NOT be accepted");
}

// ════════════════════════════════════════════════════════════════════════════
// AT-3032: the ALL-self-hosted-jobs security scan (COND-1 + COND-2).
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn at_3032_all_self_hosted_jobs_security_scan() {
    let jobs = scan_self_hosted_jobs();
    assert!(
        !jobs.is_empty(),
        "AT-3032: the structural self-hosted job scan found ZERO jobs — an empty selector \
         must fail, not pass vacuously (COND-2)"
    );
    assert!(
        jobs.len() >= KNOWN_SELF_HOSTED_JOB_FLOOR,
        "AT-3032/COND-2: expected >= {KNOWN_SELF_HOSTED_JOB_FLOOR} self-hosted jobs, \
         structurally found {}",
        jobs.len()
    );

    // Independent cross-check (COND-2): a DIFFERENT (raw-text, non-saphyr)
    // algorithm must agree on the count.
    let mut grep_total = 0usize;
    for path in all_workflow_files() {
        let content = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("eb1_workflow_ci: cannot read {}: {e}", path.display()));
        grep_total += grep_self_hosted_job_count(&content);
    }
    assert!(
        grep_total >= KNOWN_SELF_HOSTED_JOB_FLOOR,
        "AT-3032/COND-2: independent grep-style count {grep_total} is below the known floor \
         {KNOWN_SELF_HOSTED_JOB_FLOOR} — must fail, not pass vacuously"
    );
    assert_eq!(
        jobs.len(),
        grep_total,
        "AT-3032/COND-2: structural (saphyr) self-hosted job count {} != independent \
         grep-style count {grep_total} — one counting mechanism missed a job",
        jobs.len()
    );

    for j in &jobs {
        assert!(
            j.has_fork_guard_or_dispatch_only,
            "AT-3032(a): self-hosted job '{}' in {} has neither the same-repo fork guard in \
             its `if:` nor is its containing workflow workflow_dispatch-only",
            j.job, j.file
        );
        assert!(
            j.permissions_ok,
            "AT-3032(b): self-hosted job '{}' in {} does not declare \
             `permissions: contents: read` (no broader)",
            j.job, j.file
        );
        assert!(
            j.timeout_minutes.is_some(),
            "AT-3032(c): self-hosted job '{}' in {} has no `timeout-minutes`",
            j.job, j.file
        );
    }
}

// ════════════════════════════════════════════════════════════════════════════
// AT-3033: no-vacuous-pass meta-assertion — the parse actually ran and
// produced structure, for every workflow file (glob), independent of AT-3032.
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn at_3033_parse_meta_assertion_non_empty_and_jobs_resolved() {
    for path in all_workflow_files() {
        let content = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("eb1_workflow_ci: cannot read {}: {e}", path.display()));
        let rel = path.strip_prefix(repo_root()).unwrap_or(&path).display().to_string();
        let docs = parse_yaml_docs(&content, &rel);
        assert!(!docs.is_empty(), "AT-3033: {rel} parsed to zero YAML documents");
        let jobs = jobs_of(&docs[0]);
        assert!(!jobs.is_empty(), "AT-3033: {rel} 'jobs' resolved to zero jobs");
    }
}

// ════════════════════════════════════════════════════════════════════════════
// AT-3034: self-hosted job timeout-minutes sane bounds.
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn at_3034_self_hosted_timeout_bounds() {
    let jobs = scan_self_hosted_jobs();
    assert!(!jobs.is_empty(), "AT-3034: no self-hosted jobs found to bound-check");
    for j in &jobs {
        let n = j
            .timeout_minutes
            .unwrap_or_else(|| panic!("AT-3034: '{}' in {} has no timeout-minutes", j.job, j.file));
        assert!(
            n > 0 && n <= 90,
            "AT-3034: '{}' in {} timeout-minutes={n} outside the sane bound (0, 90]",
            j.job,
            j.file
        );
        if j.job == "pytorch-interop" {
            assert_eq!(n, 30, "AT-3034: pytorch-interop timeout-minutes must == 30");
        }
        if j.job == "amd-gpu" || j.job == "intel-gpu" || j.job.starts_with("bless-") {
            assert!(
                n <= 30,
                "AT-3034: per-PR test/bless job '{}' timeout-minutes={n} must be <= 30",
                j.job
            );
        }
        if j.job.starts_with("ab-") {
            assert!(n <= 90, "AT-3034: A/B job '{}' timeout-minutes={n} must be <= 90", j.job);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// AT-3035: Mesa honesty — known-bad 25.2.8, no fabricated known-good version.
// ════════════════════════════════════════════════════════════════════════════

/// Crude, dependency-free "does this look like a dotted version number"
/// scanner (no `regex` crate needed): true if `s` contains a run of digits,
/// a literal `.`, and more digits (e.g. "24.1", "25.2.8").
fn looks_like_version_number(s: &str) -> bool {
    let bytes = s.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            let start = i;
            let mut j = i;
            let mut saw_dot = false;
            while j < bytes.len() && (bytes[j].is_ascii_digit() || bytes[j] == b'.') {
                if bytes[j] == b'.' {
                    saw_dot = true;
                }
                j += 1;
            }
            if saw_dot && j > start + 2 {
                return true;
            }
            i = j.max(i + 1);
        } else {
            i += 1;
        }
    }
    false
}

#[test]
fn at_3035_mesa_honesty_no_fabricated_known_good_version() {
    let raw = read_repo_file("docs/EB1_RUNNER_SETUP.md");
    let doc = normalize_ws(&raw);
    let doc_lower = doc.to_lowercase();

    assert!(doc.contains("25.2.8"), "AT-3035: docs must mention Mesa 25.2.8");
    assert!(
        doc_lower.contains("known-bad"),
        "AT-3035: docs must frame 25.2.8 as KNOWN-BAD"
    );
    assert!(
        doc.contains("UNKNOWN") || doc.contains("TBD") || doc.contains("untested until EB.1 hardware"),
        "AT-3035: docs must contain an explicit UNKNOWN/TBD/untested-until-EB.1-hardware \
         framing for the known-good RADV/ANV Mesa version"
    );

    // No fabricated "known-good X.Y[.Z]" claim: scan every "known-good"
    // occurrence and assert no version-shaped token appears in the following
    // window.
    let lower_bytes = doc_lower.as_bytes();
    let needle = "known-good";
    let mut search_from = 0usize;
    let mut occurrences = 0usize;
    while let Some(rel_idx) = find_subslice(&lower_bytes[search_from..], needle.as_bytes()) {
        occurrences += 1;
        let idx = search_from + rel_idx;
        let window_end = (idx + needle.len() + 60).min(doc.len());
        let window = &doc[idx + needle.len()..window_end];
        assert!(
            !looks_like_version_number(window),
            "AT-3035: a version-shaped token follows 'known-good' within 60 chars — this looks \
             like a FABRICATED known-good version claim, which is forbidden: {window:?}"
        );
        search_from = idx + needle.len();
    }
    assert!(
        occurrences > 0,
        "AT-3035: docs must discuss 'known-good' (to then explicitly say it is unknown)"
    );
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}
