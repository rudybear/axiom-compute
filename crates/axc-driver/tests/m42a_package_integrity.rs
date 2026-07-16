//! M4.2a — AT-2999 (package reference-integrity) + AT-3004 (the committed `.spv` is
//! git-TRACKED, catching HIGH #4 — a `.gitignore` `*.spv` blanket rule silently un-tracking it).

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir.parent().unwrap().parent().unwrap().to_path_buf()
}

/// The M4.2a spec's SIX-deltas paragraph fixes the exact trigger-paragraph text (§7.5 review):
/// this is the EXACT byte content of `UPSTREAM_PR_PLAN.md`'s "## THE TRIGGER" paragraph, captured
/// BEFORE this milestone's edits, so a byte-compare proves the M4.2a doc update did NOT perturb it.
const TRIGGER_PARAGRAPH: &str = "Run the same single, unchanged AXIOM `.axc` source on an **AMD APU (RDNA3+) or Intel Arc** and\nproduce a same-machine, same-ICD, kernel-only A/B vs llama.cpp's Vulkan Q4_K MUL_MAT at the\ninference shape (m=4096, n=512, k=14336). **If AXIOM shows parity-or-better there** (plausible —\nthose are llama.cpp's documented weak spots: ggml-org/llama.cpp #16230, #17273; ollama #15601;\nggml-org/llama.cpp #21517), the story becomes genuinely novel: *one portable annotated source,\nbehind on NVIDIA but ahead on AMD/Intel*. That is an RFC worth a maintainer's time. If AMD/Intel\n**also** loses, do not submit at all. (This is the EB.1 hardware item — the same blocker gates the\nproject's own cross-vendor kill-criterion.)";

const STATUS_MARKER: &str = "PREPARED — HOLD";

/// AT-2999: every artifact path referenced by `upstream/REPRODUCE.md` / `UPSTREAM_PR_PLAN.md`
/// exists; the status line still reads "PREPARED — HOLD"; the trigger paragraph is
/// BYTE-IDENTICAL to its pre-M4.2a text (proving the doc update did not perturb it).
#[test]
fn at_2999_package_reference_integrity_and_trigger_unchanged() {
    let root = repo_root();

    // Every artifact this milestone's docs reference must exist.
    let referenced_paths = [
        "examples/q4km_matmul_rb_coopmat_f32acc_cached_ggml.axc",
        "examples/q4km_matmul_rb_coopmat_f32acc_cached.axc",
        "upstream/matmul_q4_k_f32_cm1_axiom.spv",
        "upstream/matmul_q4_k_f32_cm1_axiom_data.hpp.fragment",
        "upstream/ggml-vulkan-axiom-q4k.patch",
        "upstream/REPRODUCE.md",
        "crates/axc-driver/tests/dispatch_q4km_ggml.rs",
        "crates/axc-driver/tests/dispatch_q4km_ggml_equiv.rs",
        "crates/axc-driver/tests/m42a_ggml_abi_golden.rs",
        "crates/axc-driver/tests/m42a_patch_applies.rs",
        "crates/axc-codegen/src/coopmat.rs",
        ".pipeline/milestones/M4.2a-ggml-abi.md",
        "UPSTREAM_PR_PLAN.md",
    ];
    let mut missing = Vec::new();
    for rel in referenced_paths {
        if !root.join(rel).exists() {
            missing.push(rel);
        }
    }
    assert!(missing.is_empty(), "AT-2999: referenced artifact(s) missing: {missing:?}");

    // Status + trigger integrity in UPSTREAM_PR_PLAN.md.
    let plan_text = std::fs::read_to_string(root.join("UPSTREAM_PR_PLAN.md"))
        .expect("UPSTREAM_PR_PLAN.md must be readable");
    assert!(
        plan_text.contains(STATUS_MARKER),
        "AT-2999: UPSTREAM_PR_PLAN.md status line must still contain \"{STATUS_MARKER}\""
    );
    assert!(
        plan_text.contains(TRIGGER_PARAGRAPH),
        "AT-2999: UPSTREAM_PR_PLAN.md's TRIGGER paragraph must be BYTE-IDENTICAL to its \
         pre-M4.2a text — the M4.2a doc update may ADD content around it but must not perturb \
         the trigger condition itself. Expected exact paragraph:\n---\n{TRIGGER_PARAGRAPH}\n---"
    );
    // The M4.2a pointer must exist (spec §7.4: "add a one-line pointer to the milestone doc").
    assert!(
        plan_text.contains(".pipeline/milestones/M4.2a-ggml-abi.md"),
        "AT-2999: UPSTREAM_PR_PLAN.md must point at .pipeline/milestones/M4.2a-ggml-abi.md"
    );

    // REPRODUCE.md must reference the honest NVIDIA A/B numbers + the pinned SHA + the build flag.
    let reproduce_text = std::fs::read_to_string(root.join("upstream/REPRODUCE.md"))
        .expect("upstream/REPRODUCE.md must be readable");
    assert!(
        reproduce_text.contains("6b80c74f285390368b3c99c5e750f19e9b096e98"),
        "AT-2999: REPRODUCE.md must record the pinned SHA"
    );
    assert!(
        reproduce_text.contains("GGML_VULKAN_AXIOM_Q4K"),
        "AT-2999: REPRODUCE.md must reference the GGML_VULKAN_AXIOM_Q4K build flag"
    );
    assert!(
        reproduce_text.contains("42.86") && reproduce_text.contains("102.49"),
        "AT-2999: REPRODUCE.md must record the honest NVIDIA A/B (42.86 vs 102.49 = 2.39x behind)"
    );

    eprintln!(
        "AT-2999 PASS: {} referenced artifacts exist; status line + trigger paragraph in \
         UPSTREAM_PR_PLAN.md are intact (byte-compared); REPRODUCE.md records the pinned SHA + \
         build flag + honest A/B",
        referenced_paths.len()
    );
}

/// AT-3004 (catches HIGH #4): `upstream/matmul_q4_k_f32_cm1_axiom.spv` is git-TRACKED
/// (`git ls-files --error-unmatch`), NOT merely present on disk. `.gitignore`'s blanket `*.spv`
/// rule would silently un-track it without the `!upstream/*.spv` negation (placed AFTER the
/// `*.spv` line).
#[test]
fn at_3004_committed_spv_is_git_tracked() {
    let root = repo_root();
    let spv_path = root.join("upstream/matmul_q4_k_f32_cm1_axiom.spv");
    assert!(spv_path.is_file(), "AT-3004: {} must exist on disk", spv_path.display());

    // git check-ignore: the file must NOT be ignored (the !upstream/*.spv negation must win).
    let ignore_check = Command::new("git")
        .args(["-C", root.to_str().unwrap(), "check-ignore", "upstream/matmul_q4_k_f32_cm1_axiom.spv"])
        .output()
        .expect("git check-ignore must run");
    assert!(
        !ignore_check.status.success(),
        "AT-3004: upstream/matmul_q4_k_f32_cm1_axiom.spv is STILL gitignored — the \
         `!upstream/*.spv` negation (added AFTER the `*.spv` line, HIGH #4) is missing or \
         mis-ordered in .gitignore"
    );

    // git ls-files --error-unmatch: the file must be actually TRACKED (staged/committed), not
    // merely un-ignored (a file can be un-ignored yet never `git add`-ed).
    let ls_files = Command::new("git")
        .args(["-C", root.to_str().unwrap(), "ls-files", "--error-unmatch", "upstream/matmul_q4_k_f32_cm1_axiom.spv"])
        .output()
        .expect("git ls-files must run");
    assert!(
        ls_files.status.success(),
        "AT-3004: upstream/matmul_q4_k_f32_cm1_axiom.spv is NOT git-tracked (`git ls-files \
         --error-unmatch` failed) — it must be `git add`-ed, not merely present on disk and \
         un-ignored:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&ls_files.stdout),
        String::from_utf8_lossy(&ls_files.stderr)
    );

    eprintln!("AT-3004 PASS: upstream/matmul_q4_k_f32_cm1_axiom.spv is un-ignored AND git-tracked");
}

/// The `.gitignore` negation line itself is present and correctly ordered AFTER `*.spv`.
#[test]
fn at_3004_gitignore_negation_present_and_ordered() {
    let root = repo_root();
    let gitignore = std::fs::read_to_string(root.join(".gitignore")).expect(".gitignore must be readable");
    let star_spv_line = gitignore.lines().position(|l| l.trim() == "*.spv");
    let negation_line = gitignore.lines().position(|l| l.trim() == "!upstream/*.spv");
    assert!(star_spv_line.is_some(), "AT-3004: .gitignore must contain a bare `*.spv` rule");
    assert!(negation_line.is_some(), "AT-3004: .gitignore must contain `!upstream/*.spv`");
    assert!(
        negation_line.unwrap() > star_spv_line.unwrap(),
        "AT-3004: `!upstream/*.spv` must come AFTER `*.spv` (gitignore negations only work when \
         ordered after the rule they negate) — the M3.17 lesson"
    );
}
