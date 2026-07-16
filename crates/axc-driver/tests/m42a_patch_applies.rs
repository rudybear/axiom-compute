//! M4.2a — AT-2998: `upstream/ggml-vulkan-axiom-q4k.patch` applies cleanly (`git apply --check`)
//! against the pinned llama.cpp SHA `6b80c74f285390368b3c99c5e750f19e9b096e98`, AND its hunk
//! context anchors INSIDE the coopmat block `[:4090,:4211]` (creation) and at/near the
//! `ggml_vk_guess_matmul_pipeline` call `:8239` (override) — a static hunk-context check so a
//! patch drifting to the wrong device tier (e.g. the `:4429` `if(device->fp16)` fallback) or
//! elsewhere fails CI cheaply.
//!
//! `vendor/` is gitignored (never committed) — this test is a **typed-skip** if
//! `vendor/llama.cpp` is absent (e.g. a fresh CI checkout that never ran
//! `scripts/m34_llamacpp_ab.sh`). This mirrors the AT-2998 convention noted in the milestone's
//! CI-runnable table ("2998(skip)").
//!
//! NEVER applies the patch persistently — `git apply --check` only; the vendor tree is read-only.

use std::path::PathBuf;
use std::process::Command;

const PINNED_SHA: &str = "6b80c74f285390368b3c99c5e750f19e9b096e98";

/// The coopmat block's line range (`ggml-vulkan.cpp:4090-4211`) — the CREATE_MM block.
const COOPMAT_BLOCK_START: u32 = 4090;
const COOPMAT_BLOCK_END: u32 = 4211;
/// A window around the `ggml_vk_guess_matmul_pipeline` call site (`:8239`) that a valid
/// override hunk must anchor within — generous enough to tolerate a hunk header starting a few
/// lines of context before the exact call line, tight enough to reject the `:4429` fp16 tier or
/// anything outside the selection-guard region.
const OVERRIDE_SITE_WINDOW: std::ops::RangeInclusive<u32> = 8200..=8242;

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = <repo>/crates/axc-driver
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir.parent().unwrap().parent().unwrap().to_path_buf()
}

fn vendor_llamacpp_path() -> PathBuf {
    repo_root().join("vendor/llama.cpp")
}

fn patch_path() -> PathBuf {
    repo_root().join("upstream/ggml-vulkan-axiom-q4k.patch")
}

/// Parse `@@ -<start>,<count> +...` hunk headers from a unified diff, returning the ORIGINAL-file
/// (pre-image) `(start_line, count)` pairs. Naive but sufficient for this single-file patch.
fn parse_hunk_starts(patch_text: &str) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    for line in patch_text.lines() {
        if let Some(rest) = line.strip_prefix("@@ -") {
            // rest looks like "8238,6 +8261,35 @@ ..." — take up to the first space.
            if let Some(minus_part) = rest.split(' ').next() {
                let mut it = minus_part.splitn(2, ',');
                let start: Option<u32> = it.next().and_then(|s| s.parse().ok());
                let count: u32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(1);
                if let Some(start) = start {
                    out.push((start, count));
                }
            }
        }
    }
    out
}

#[test]
fn at_2998_patch_applies_cleanly_and_hunks_anchor_in_range() {
    let vendor = vendor_llamacpp_path();
    if !vendor.join(".git").is_dir() {
        eprintln!(
            "AT-2998: vendor/llama.cpp not present (gitignored, not fetched in this checkout) — \
             typed-skip. Run scripts/m34_llamacpp_ab.sh (or clone+pin manually) to exercise this \
             test locally."
        );
        return;
    }

    // Verify we're checking against the PINNED SHA (a mismatched tree could apply "cleanly" by
    // coincidence and give false confidence).
    let head = Command::new("git")
        .args(["-C", vendor.to_str().unwrap(), "rev-parse", "HEAD"])
        .output()
        .expect("git rev-parse must run");
    let head_sha = String::from_utf8_lossy(&head.stdout).trim().to_string();
    if head_sha != PINNED_SHA {
        eprintln!(
            "AT-2998: vendor/llama.cpp HEAD={head_sha} != pinned SHA {PINNED_SHA} — typed-skip \
             (the patch is verified against the PINNED SHA specifically; a drifted tree is not \
             a meaningful apply-check)."
        );
        return;
    }

    let patch = patch_path();
    assert!(patch.is_file(), "AT-2998: {} must exist", patch.display());
    let patch_text = std::fs::read_to_string(&patch).expect("patch file must be readable UTF-8");

    // (1) git apply --check — NEVER applies persistently.
    let check = Command::new("git")
        .args(["-C", vendor.to_str().unwrap(), "apply", "--check", patch.to_str().unwrap()])
        .output()
        .expect("git apply --check must run");
    assert!(
        check.status.success(),
        "AT-2998: `git apply --check` failed against the pinned SHA {PINNED_SHA}:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&check.stdout),
        String::from_utf8_lossy(&check.stderr)
    );

    // (2) hunk-context anchoring: at least one hunk starts inside the coopmat block
    // [4090,4211] (creation), and at least one hunk overlaps the override-site window
    // [8200,8242] (the ggml_vk_guess_matmul_pipeline call region).
    let hunks = parse_hunk_starts(&patch_text);
    assert!(!hunks.is_empty(), "AT-2998: patch must contain at least one hunk");

    let creation_anchor = hunks.iter().any(|&(start, count)| {
        let end = start + count;
        // overlap with [COOPMAT_BLOCK_START, COOPMAT_BLOCK_END]
        start <= COOPMAT_BLOCK_END && end >= COOPMAT_BLOCK_START
    });
    assert!(
        creation_anchor,
        "AT-2998: no patch hunk anchors inside the coopmat block [{COOPMAT_BLOCK_START},\
         {COOPMAT_BLOCK_END}] — a patch that drifted to the wrong device tier (e.g. the :4429 \
         if(device->fp16) fallback) must fail this check. Hunks: {hunks:?}"
    );

    let override_anchor = hunks.iter().any(|&(start, count)| {
        let end = start + count;
        end >= *OVERRIDE_SITE_WINDOW.start() && start <= *OVERRIDE_SITE_WINDOW.end()
    });
    assert!(
        override_anchor,
        "AT-2998: no patch hunk anchors at/near the ggml_vk_guess_matmul_pipeline call site \
         window [{},{}] — the dispatch-override hunk must sit there. Hunks: {hunks:?}",
        OVERRIDE_SITE_WINDOW.start(), OVERRIDE_SITE_WINDOW.end()
    );

    eprintln!(
        "AT-2998 PASS: patch applies cleanly (git apply --check, read-only) against pinned SHA \
         {PINNED_SHA}; hunks anchor in [{COOPMAT_BLOCK_START},{COOPMAT_BLOCK_END}] (creation) \
         and near :8239 (override). Hunk starts: {hunks:?}"
    );
}

/// Sanity: the patch file targets the real file path (defense against a copy-paste path typo).
#[test]
fn at_2998_patch_targets_ggml_vulkan_cpp() {
    let patch_text = std::fs::read_to_string(patch_path())
        .expect("upstream/ggml-vulkan-axiom-q4k.patch must exist and be readable");
    assert!(
        patch_text.contains("+++ b/ggml/src/ggml-vulkan/ggml-vulkan.cpp"),
        "AT-2998: patch must target ggml/src/ggml-vulkan/ggml-vulkan.cpp"
    );
    assert!(
        patch_text.contains("GGML_VULKAN_AXIOM_Q4K"),
        "AT-2998: every patch hunk must be behind the GGML_VULKAN_AXIOM_Q4K opt-in flag"
    );
}
