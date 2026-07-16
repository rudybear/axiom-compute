# EB.1-prep — Self-hosted AMD/Intel runner workflows + setup docs

**Milestone ID:** `EB1prep-cross-vendor-ci`
**Verdict:** PROPOSED (Architect) — **revision r2**
**Depends on:** EB.2 (per-machine-keyed baselines, MERGED), M3.15/EB.4 (two blessed keys), M3.23 (Mesa-pin discovery).
**Blocks / enables:** EB.1 measurement (hardware-gated), M4.2 upstream RFC trigger (`UPSTREAM_PR_PLAN.md §7.5`).

> **r2 status:** revised after optimistic APPROVE + pessimistic NEEDS_REVISION. Both CRITICAL blockers closed (the LIVE `pytorch-interop` fork hole is now in scope and hardened; the YAML parse mechanism is PINNED to a maintained crate with the silent-no-op fallback FORBIDDEN), plus timeouts, Mesa-honesty, branch-protection posture, and the folded optimistic rulings. AT range grows to **AT-3019 .. AT-3035**. See the **r2-changelog** at the end.

---

## 0. Scope and non-scope

EB.1 is "self-hosted GitHub Actions runners for AMD (RDNA3 + Vulkan ICD) + Intel (Arc + ANV ICD); acceptance = all GPU tests + the bench regression gate run on AMD and Intel in CI on every PR." That acceptance is **hardware-gated** — this box has only NVIDIA, no AMD/Intel runner can register today.

This milestone delivers the **preparable half**: the CI workflows, the runner setup docs, and a CPU-only validation harness — everything that can be authored, schema-validated, and smoke-run **without** the AMD/Intel hardware, so that when a runner physically registers the cross-vendor A/B and the per-PR gates go live by flipping **one repository variable** (no code change).

**In scope (this milestone):**
- 3 new GitHub Actions workflow files (per-PR vendor gates; the `workflow_dispatch` A/B evidence job; the `workflow_dispatch` manual bless job).
- **Hardening of the ALREADY-LIVE self-hosted job** — `ci.yml`'s `pytorch-interop` (`[self-hosted, nvidia, cuda]`, `on: pull_request`) TODAY has no fork guard and no `permissions:` pin (see §2.3). This milestone adds the same-repo `if:` guard, `permissions: { contents: read }`, and `timeout-minutes` to it, and the security AT is a **scan of ALL self-hosted jobs in ALL workflow files** — not a per-file list — so any future self-hosted job that regresses the guard is caught.
- `docs/EB1_RUNNER_SETUP.md` — per-vendor driver/ICD install, runner registration, labels, activation switch, security constraints, first-run bless procedure, branch-protection posture.
- A CPU-only validation test (`crates/axc-driver/tests/eb1_workflow_ci.rs`) using a **PINNED maintained YAML parser dev-dependency** (§6.3) + an `actionlint` lint step folded into the existing CI — the AT-3019.. test plan.
- Fold-in of the M3.23 Mesa-pin recommendation where the existing CI (and the RADV runner) is exposed.

**Explicitly NOT in scope:**
- **No compiler / driver / runtime code changes.** Zero touches to `crates/axc-*/src`. (The only Rust added is a CPU-only integration *test* that parses the YAML + anchors the docs; the only manifest change is one **test-only dev-dependency** — §6.3.)
- **No actual AMD/Intel measurement** — that is EB.1 proper, hardware-gated. This milestone makes it one-command-runnable, not run.
- **No Windows/AMD path** (open-question #3, RULED out-of-scope): the beachhead is Linux + Mesa (RADV/ANV); a Windows runner needs a different ICD stack (native AMD/Intel Vulkan, no Mesa) and its own security review — genuinely new scope, deferred, and flagged as a deliberate non-goal (not an oversight).
- No change to the FROZEN 1e-3 tolerance, no change to `baselines.json`, no re-bless of existing keys.

---

## 1. The activation mechanism — avoiding "queue-forever"

### 1.1 The failure mode we must NOT ship

A job whose `runs-on:` targets a label set that no registered runner satisfies **does not skip — it queues indefinitely** (status: *pending / waiting for a runner*). On a `pull_request` trigger a pending required check **blocks the merge button forever**. Naïvely writing:

```yaml
amd-gpu:
  runs-on: [self-hosted, amd-rdna3]   # WRONG while no such runner exists
```

would make every PR un-mergeable the moment this file lands. This is the single most important correctness constraint of the milestone.

### 1.2 The chosen mechanism: an `if:` gate on a repository variable

Each vendor job is **guarded by a repository variable** evaluated at job level:

```yaml
amd-gpu:
  if: >-
    vars.AMD_RUNNER_ONLINE == 'true'
    && (github.event_name != 'pull_request'
        || github.event.pull_request.head.repo.full_name == github.repository)
  runs-on: [self-hosted, amd-rdna3]
```

- When `vars.AMD_RUNNER_ONLINE` is **unset or `'false'`** (today, and any time the runner is offline), the `if` is false → GitHub **skips** the job. A skipped job is *neutral/success* for branch-protection purposes — it never queues and never blocks a merge. This is the honest, verified difference between *skip* (`if:` false) and *queue-forever* (`runs-on:` unsatisfiable).
- The **activation switch** is literally: repo **Settings → Secrets and variables → Actions → Variables →** set `AMD_RUNNER_ONLINE = true` (and `INTEL_RUNNER_ONLINE = true`) the moment the corresponding runner is registered and healthy. One variable per vendor. No workflow edit, no merge, no code change. Documented step-by-step in `EB1_RUNNER_SETUP.md §Activation`.
- **Turning a vendor off** (runner down for maintenance) = flip the variable back to `false`; PRs immediately stop waiting on it.

The second clause of the `if` (the same-repo guard) is the fork-PR security control — see §2.

### 1.4 Branch-protection posture (r2 — the skip-vs-pending distinction, and the paths-filter trap)

The design's safety rests on a **precise** GitHub semantic that must be documented so a future edit cannot silently break it:

- **`if:`-level skip == PASS.** A job that is *skipped* because its job-level `if:` evaluated false — inside a workflow that **did** trigger — reports its required status check as **successful** (GitHub's since-2022 required-check semantics). This is why the vendor jobs are safe to leave in place while offline: they skip green, they never queue.
- **Never-triggered == PENDING-FOREVER.** The dangerous, *different* case is a required check whose **workflow never triggered at all** (e.g. a `paths:` or branch filter excluded the PR's changed files). GitHub then leaves that required check **Pending indefinitely**, and a pending required check **blocks the merge button forever** — the exact "queue-forever" failure §1.1 exists to avoid, arriving through a different door.
- **Therefore (documented in `EB1_RUNNER_SETUP.md`):**
  1. Do **NOT** mark `amd-gpu` / `intel-gpu` (or the A/B / bless jobs) as **required status checks** until *after* the first successful post-activation run on real hardware. A vendor job made "required" while its runner is offline is fine under job-level-skip semantics, but making it required *before it has ever produced a check* risks the pending trap on tooling that treats an absent check as pending.
  2. **Never add a `paths:` or path-ignore filter to these workflows.** A paths-filtered workflow that doesn't trigger yields the pending-forever trap for any required check it declares. The workflows trigger on `push`/`pull_request` unconditionally (job-level `if:` does the gating, not workflow-level filters) — this is deliberate and must be preserved.
  3. `main` currently has **no branch protection at all** (verified: `gh api …/branches/main/protection` → 404), so none of this blocks today; the guidance is forward-defensive for whenever protection is enabled.

### 1.3 The manual-only workflows use `workflow_dispatch`

The A/B evidence job (§3) and the first-run bless job (§4) are **`workflow_dispatch`-only** (plus the repo-variable guard). They never run automatically, so they *cannot* queue on a PR; they are one-click jobs a maintainer runs when a runner is online. This also matches their semantics: the A/B is measurement-not-a-gate, and blessing is a deliberate human act (never automatic in CI — the AT-714b discipline).

---

## 2. Fork-PR security (self-hosted runners on a PUBLIC repo)

`axiom-compute` is public. A self-hosted runner is a **persistent, non-ephemeral machine on the maintainer's network** holding a real GPU. The canonical, severe risk: a `pull_request` from a **fork** carries attacker-controlled workflow/build/test code; if that code executes on a self-hosted runner it is **arbitrary code execution on the maintainer's hardware** (secret theft, lateral movement, runner persistence). GitHub's own docs flag this as the primary reason self-hosted runners on public repos are dangerous.

### 2.1 Layered mitigations (all applied)

1. **Fork PRs never reach the runner (primary control).** The same-repo clause in every vendor job's `if:`
   `github.event.pull_request.head.repo.full_name == github.repository`
   is **false for any fork PR** → the vendor job is *skipped*, so the fork's code is never checked out or executed on the self-hosted machine. Same-repo PRs (branches pushed by collaborators with write access, who are already trusted) run normally once the runner is online.
2. **Never `pull_request_target`.** These workflows use `pull_request` (restricted context: read-only `GITHUB_TOKEN`, no secrets for fork events) and **never** `pull_request_target` (which would run the *trusted* workflow with repo secrets against fork-checked-out code — the classic footgun). A negative test (AT-3024) asserts the string `pull_request_target` appears in **none** of the new workflow files.
3. **Optional stricter tier — environment approval.** For maintainers who *want* fork PRs to run after a human clicks "approve", the docs describe wrapping the vendor jobs in a GitHub **`environment:` with required reviewers** (deployment-protection rules pause the job *before* a runner picks it up). We ship the same-repo-skip default (fork code never runs) and document the environment-approval upgrade as an alternative — not both, to avoid a confusing double gate.
4. **Least privilege.** Every vendor job sets `permissions: { contents: read }` and consumes **no repo secrets** (the tests + bench gate + A/B need none — llama.cpp and the ICD are installed on the runner, not fetched with a token). This bounds blast radius even for same-repo events.
5. **Runner hardening (docs).** `EB1_RUNNER_SETUP.md` mandates: register at **repo scope** (not org), run the runner service as a **non-root, unprivileged** user, prefer an **ephemeral / just-in-time** runner or a disposable VM/container per job, keep no long-lived credentials on the box, and enable **Settings → Actions → "Require approval for all outside collaborators"** as a belt-and-suspenders backstop.

### 2.2 Honest limit

The same-repo-skip means the **per-PR AMD/Intel gate covers collaborator (same-repo) PRs, not fork PRs** — an intentional security/utility tradeoff. Fork contributions still get the full Lavapipe CI (which is safe on `ubuntu-latest` hosted runners); their cross-vendor coverage happens after a maintainer pushes the branch to the repo or runs the `workflow_dispatch` A/B. This is the correct posture for a public repo and is stated plainly in the docs.

### 2.3 The ALREADY-LIVE hole: `pytorch-interop` (CRITICAL, r2 — closes BLOCKER-1)

The milestone's premise is "secure self-hosted runners on a public repo," yet the existing `pytorch-interop` job (`.github/workflows/ci.yml:98`, `runs-on: [self-hosted, nvidia, cuda]`) already ships the exact hole §2.1 mitigates for the NEW jobs:

- it is reached by the workflow-level `on: pull_request`;
- it has **no same-repo fork guard** → a fork PR's `working-directory: py` build + `maturin develop --release` + `pytest` executes **attacker-controlled code on the maintainer's real NVIDIA GPU box TODAY** (arbitrary code execution, secret theft, runner persistence);
- it has **no `permissions:` pin** (inherits the workflow-default token scope);
- it has **no `timeout-minutes`** (a wedged build can hold the sole GPU box for GitHub's 360-min default).

Because `ci.yml` is already in `files_to_modify`, this milestone **fixes it in place**:

```yaml
pytorch-interop:
  runs-on: [self-hosted, nvidia, cuda]
  continue-on-error: true            # kept (M4.1 mergeability posture)
  timeout-minutes: 30                 # r2: bound GPU-box lockup
  permissions:
    contents: read                    # r2: least privilege
  if: >-                              # r2: fork code never runs on the box
    github.event_name != 'pull_request'
    || github.event.pull_request.head.repo.full_name == github.repository
```

The `continue-on-error` posture (M4.1) is preserved — it only affects the PR *status*, not the security guard. `AXC_ENABLE_GPU_TESTS`/`VK_DRIVER_FILES` and the steps are unchanged.

### 2.4 Security AT is a SCAN, not a per-file list (regression-proof)

To ensure the next self-hosted job cannot silently re-open the hole, the security assertion (AT-3032) **enumerates every self-hosted-labelled job across ALL `.github/workflows/*.yml`** (glob, not a hardcoded 4-file list — including `pytorch-interop` and any future addition) and asserts each carries (a) the same-repo fork guard **or** is `workflow_dispatch`-only, (b) a `permissions:` block no broader than `contents: read`, and (c) a `timeout-minutes`. A new self-hosted job that forgets any of the three **fails CI**.

---

## 3. The llama.cpp A/B one-command job (the RFC trigger evidence)

`UPSTREAM_PR_PLAN.md §7.5` states the upstream RFC trigger is the **AMD/Intel A/B parity-or-better** measurement — currently the single UNMEASURED, EB.1-gated precondition. This milestone makes producing that evidence **one click**.

### 3.1 Job shape (`cross-vendor-ab.yml`, `workflow_dispatch`-only)

Inputs:
- `vendor` (choice: `amd` | `intel`) → selects `runs-on` label + the ICD env.
- `variant` (string, default `--fused-f32acc-cached`) → the AXIOM fused Q4_K_M kernel flag passed to `scripts/m34_llamacpp_ab.sh` (the M3.6 leader is the default).
- `skip_llama_build` (boolean, default false) → reuse a cached llama.cpp checkout.

Job attributes: `timeout-minutes: 90` (r2 — the llama.cpp `test-backend-ops` source build dominates; 90 bounds a wedged build/dispatch on the sole GPU box), `permissions: { contents: read }`, `workflow_dispatch`-only.

Steps (one vendor job, gated by `vars.<VENDOR>_RUNNER_ONLINE == 'true'`):
1. `actions/checkout`.
2. `dtolnay/rust-toolchain@stable`; `cargo build --release -p axc-driver` (the AXIOM side).
3. Clone llama.cpp **pinned** at tag `b9542` / SHA `6b80c74f285390368b3c99c5e750f19e9b096e98` (the exact pin M3.4 established), configure with the Vulkan backend, build `test-backend-ops`. **`skip_llama_build` guard (r2):** when `skip_llama_build=true`, a preflight step asserts the cached build exists AND records its checked-out SHA equals the pin (`git -C vendor/llama.cpp rev-parse HEAD == 6b80c74…`); if the cache is absent or the SHA drifted, the job **fails loudly** rather than the obscure downstream "missing `test-backend-ops`" error.
4. Run `scripts/m34_llamacpp_ab.sh ${{ inputs.variant }}` with `VK_DRIVER_FILES` set to the vendor ICD (RADV for AMD, ANV for Intel) and `AXC_ENABLE_GPU_BENCHES=1` — the **same-machine, same-ICD, fence-synchronized** A/B the NVIDIA half already runs.
5. `actions/upload-artifact` the produced `ab_results*.json` + the run log + `vulkaninfo --summary` (device provenance) → **this artifact is the RFC-trigger evidence** feeding `UPSTREAM_PR_PLAN.md §7.5`'s AMD/Intel row.

**Honesty contract (mirrors the script's own):** this job **has no ratio pass/fail gate**. M3.13 concluded the NVIDIA *throughput* campaign; this run is *measurement/evidence*, not a CI gate. It records whatever the numbers are (parity, win, or loss) with no cherry-picking. Whether the numbers fire the RFC trigger is a **human** reading of the artifact against §7.5, not a green check.

**Evidence provenance (r2 — a bounded integrity anchor, not a proof):** because this artifact becomes `UPSTREAM_PR_PLAN.md §7.5` RFC evidence and anyone with `workflow_dispatch` rights or runner access could in principle produce falsified numbers, the docs record the **provenance chain** that scopes the trust: the dispatching actor is recorded in the run metadata (`github.actor`), the full runner log is uploaded alongside the JSON, and the `vulkaninfo --summary` device string is embedded in the artifact (ties numbers to a real device). This is stated honestly as a **bounded limit** — it deters and audits, it does not cryptographically prove — matching the §7.5 evidence chain.

---

## 4. Per-vendor per-PR jobs: the 6 GPU tests + bench regression gate

### 4.1 What runs (`cross-vendor-ci.yml`)

Per vendor (`amd-gpu`, `intel-gpu`), mirroring the existing Lavapipe `test` + `bench-regression` jobs but on the self-hosted label and with the vendor ICD. Each job sets `timeout-minutes: 30` (r2 — the GPU test + bench pass, no source-build of llama; 30 bounds a wedged GPU dispatch) and `permissions: { contents: read }`:

- **GPU tests:** `cargo test --workspace && cargo test --workspace -- --ignored` under
  `AXC_ENABLE_GPU_TESTS=1`, `AXC_REQUIRE_SPIRV_VAL=1`, and the vendor `VK_DRIVER_FILES`.
  This is the **exact command the existing CI uses** and it runs the full GPU-dispatch suite — the honest superset of "the 6 GPU tests." We deliberately do **not** hard-code six test names (brittle); we run the whole `--ignored` set and assert **zero failures**.

  **Honest cross-vendor caveat (documented, not hidden):** a large part of the `#[ignore]` GPU suite is **coopmat / subgroup-size-32-gated** and **typed-skips by construction** on non-wave32 devices (see `DESIGN.md §754`: the kernels *miscompute* on wave64/SIMD16 so their tests typed-skip when `subgroup_size() != 32`). On **AMD RDNA3 (wave64 default)** and **Intel Arc (SIMD8/16)** those tests will typed-skip, not fail. The portable-everywhere subset (saxpy, vector_add, plain multi-tile matmul, the non-`#[ignore]` `dispatch_q6k_matmul` K=256) executes on all three vendors and is the real cross-vendor correctness signal. The job asserts **no failures**; the run log records **which tests typed-skipped**, so "green on AMD" is never silently over-read as "all coopmat kernels ran on AMD." AMD RDNA3 exposes `VK_KHR_cooperative_matrix`; whether AXIOM's wave32-hard-wired coopmat kernels can be made wave64-portable is a genuine EB.1 follow-up, flagged not solved here.

- **Bench regression gate:** `cargo test --release -p axc-driver --test bench_regression` with the vendor ICD and `AXC_ENABLE_BENCH_REGRESSION=1`, **`AXC_BLESS_BASELINES=""`** (never auto-bless — AT-714b). This reuses EB.2's per-machine keying verbatim: `select_block` keys off `sanitize(deviceName)`; a brand-new AMD/Intel device is an **absent key**, so the gate returns `SelectOutcome::LoudSkipKeyAbsent` and **SKIPS with a loud warning** ("re-bless with AXC_BLESS_BASELINES=1") rather than failing. That is exactly the EB.2-designed first-run behavior — a new machine is never hard-blocked. The gate only starts *gating* AMD/Intel after a human blesses the first baseline (§4.2).

### 4.2 First-run bless procedure — who blesses, and how

Blessing is **never automatic** (AT-714b). A dedicated **`workflow_dispatch`-only** job `cross-vendor-bless.yml` performs it:

- Input `vendor` (amd | intel). Runs on the vendor label, gated by the runner-online variable. `timeout-minutes: 30`, `permissions: { contents: read }`, `workflow_dispatch`-only.
- Steps: checkout → build → run the resident/cpu-reference benches with `AXC_BLESS_BASELINES=1` → the harness adds a **new machine block** keyed by `sanitize(deviceName)` (EB.2's `merge_blessed`, which preserves existing keys — it does *not* overwrite NVIDIA/Lavapipe) → upload the updated `baselines.json` as an artifact.
- A **maintainer** then reviews the diff (one new key, correct device name, sane numbers) and **commits it via a normal PR**. The runner never writes to the repo; blessing is a human-reviewed commit. Documented as a numbered checklist in `EB1_RUNNER_SETUP.md §First-run bless`.

After that commit lands, the AMD/Intel key exists in `baselines.json` → the per-PR `bench-regression` gate transitions from `LoudSkipKeyAbsent` (skip) to `Gate` (enforce) automatically on the next run. No workflow change.

---

## 5. Driver / ICD choices (docs)

### 5.1 AMD — choose **RADV** (Mesa), justified
- **RADV** (Mesa's open-source Vulkan driver) over **AMDVLK** (AMD's open source) / the proprietary `amdvlk-pro`, because: RADV is the driver **llama.cpp's own Vulkan CI and the overwhelming majority of AMD Vulkan users run** (so the A/B compares AXIOM against the *same* stack llama is tuned/measured on — an apples-to-apples portability claim); it ships in-distro (`mesa-vulkan-drivers`, trivial `apt`/`dnf` install, no out-of-tree package); it has the most mature `VK_KHR_cooperative_matrix` support on RDNA3; and it is what the kill-criterion audience (the DESIGN §5 "AMD APU / Intel Arc weak-spot" users, ollama #15601) actually uses. AMDVLK is documented as a fallback if a specific coopmat path regresses on RADV.
- ICD JSON: `/usr/share/vulkan/icd.d/radeon_icd.x86_64.json`. Select the AMD device with `AXC_PHYSICAL_DEVICE_INDEX` if the box also has a probe GPU.

### 5.2 Intel — **ANV** (Mesa)
- **ANV** (Mesa's Intel Vulkan driver) is the only practical open Arc Vulkan driver on Linux; ships as `mesa-vulkan-drivers`. ICD JSON: `/usr/share/vulkan/icd.d/intel_icd.x86_64.json`. Arc needs a recent Mesa (≥ 24.x) for stable `VK_KHR_cooperative_matrix` / Xe-cores — pinned per §6.

### 5.3 Mesa-version pin (folding in the M3.23 discovery)
M3.23 root-caused a **Lavapipe SIGSEGV/heap-corruption on Mesa 25.2.8** when workgroup-**shared**-memory kernels are dispatched via the **resident** harness (ROADMAP §412; NVIDIA unaffected). RADV and ANV are **also Mesa** and the A/B job (§3) dispatches exactly those shared-memory resident kernels. Therefore:
- **Mesa honesty (r2 — no invented ranges):** the ONLY evidence we have is that **Mesa 25.2.8 is known-BAD** for the resident shared-mem path (measured on Lavapipe; NVIDIA unaffected). There is **NO known-good Mesa version on RADV/ANV** — that path is entirely hardware-gated and untested. `EB1_RUNNER_SETUP.md` therefore records **`25.2.8` as known-bad** and states explicitly that **the known-good RADV/ANV Mesa version is UNKNOWN / TBD until EB.1 hardware — do not assume or pin a fabricated "good" version.** It instructs the operator to (a) avoid 25.2.8, and (b) capture the working Mesa version at first successful EB.1 run and record it *then* as the empirically-good pin. No version range is invented here.
- A **`mesa-version` probe step** (`vulkaninfo | grep driverInfo`, non-fatal) is added to the vendor jobs *and* folded into the existing Lavapipe `test` job so a bad Mesa bump surfaces in the log instead of a mystery segfault. (We do not hard-*pin* `ubuntu-latest`'s Mesa in `ci.yml` — that runner is hosted and we don't control its apt mirror — but we surface the version and document the risk. Fold-in is a probe + doc, not a silent apt version hold that could rot.)

---

## 6. Files

### 6.1 Create

| Path | Purpose |
|---|---|
| `.github/workflows/cross-vendor-ci.yml` | Per-PR (same-repo) AMD + Intel jobs: full `--ignored` GPU test suite + bench-regression gate. `if:`-gated on `vars.<VENDOR>_RUNNER_ONLINE` (skip, never queue) AND the same-repo fork guard. `permissions: contents: read`. Triggers: `push` (repo branches) + `pull_request` + `workflow_dispatch`. Includes the mesa-version probe step. |
| `.github/workflows/cross-vendor-ab.yml` | `workflow_dispatch`-only llama.cpp A/B evidence job (inputs: vendor, variant, skip_llama_build). Builds AXIOM + pinned llama.cpp `b9542`, runs `scripts/m34_llamacpp_ab.sh`, uploads `ab_results*.json` + logs. NO ratio gate. The RFC-trigger evidence for `UPSTREAM_PR_PLAN.md §7.5`. |
| `.github/workflows/cross-vendor-bless.yml` | `workflow_dispatch`-only first-run baseline bless job (input: vendor). Runs benches with `AXC_BLESS_BASELINES=1`, uploads the merged `baselines.json` artifact for a human to commit. |
| `docs/EB1_RUNNER_SETUP.md` | Per-vendor driver/ICD install (RADV justified, ANV), runner registration + labels (`[self-hosted, amd-rdna3]`, `[self-hosted, intel-arc]`), the **Activation** switch (repo variables), the **fork-PR security** section, the **first-run bless** checklist, the **Mesa pin** guidance, and a "validate without hardware" appendix. |
| `crates/axc-driver/tests/eb1_workflow_ci.rs` | CPU-only AT-3019..3035 test plan: YAML validity of the 3 workflow files + `ci.yml` (via the PINNED parser, §6.3), structural/security anchors (activation-var gate present, same-repo guard present, `pull_request_target` absent, `runs-on` self-hosted labels, `permissions: contents: read`, `timeout-minutes` present, `AXC_BLESS_BASELINES` never `1` in CI-gate job), the **ALL-self-hosted-jobs scan** across every workflow file (incl. `pytorch-interop`), docs-completeness + Mesa-honesty anchors, and the llama pin match. Runs in normal `ubuntu-latest` CI — no hardware. **No `#[cfg]`-guarded no-op path** — if the parser is unavailable the test PANICS (§6.3), never silently passes. |

### 6.2 Modify

| Path | Change |
|---|---|
| `.github/workflows/ci.yml` | **(r2) Harden `pytorch-interop`:** add the same-repo `if:` fork guard, `permissions: { contents: read }`, and `timeout-minutes: 30` (§2.3). **Add a CPU-only `lint-workflows` job** (§6.4). Add the non-fatal **mesa-version probe** step to the existing `test` job. |
| `crates/axc-driver/Cargo.toml` | **(r2)** Add `saphyr` to `[dev-dependencies]` (test-only, §6.3) + the resulting `Cargo.lock` update. No `src/` touch, no runtime dependency, no default-member/feature change. |
| `ROADMAP.md` | Flip the EB.1 row note to "prep MERGED (workflows + docs authored, `AMD/INTEL_RUNNER_ONLINE` activation ready); measurement remains hardware-gated." (Documentation only.) |
| `UPSTREAM_PR_PLAN.md` | §7.5: note the A/B trigger is now **one `workflow_dispatch` click** (`cross-vendor-ab.yml`) once a runner registers. (Documentation only.) |

### 6.3 The PINNED YAML parse mechanism (r2 — closes BLOCKER-2, no vacuous passes)

**Fact established at review:** no YAML parser is in `Cargo.lock` today, and `serde_yaml` is **deprecated/archived** by its author — it must NOT be introduced. The r1 hedge ("whatever is already vendored … `#[cfg]`-guarded no-op when absent") is FORBIDDEN because it lets the security ATs (fork-guard, no-`pull_request_target`, least-privilege, never-auto-bless) compile to **vacuous green passes** when the parser/python is missing — strictly worse than no check.

**Ruling — pin exactly one mechanism:** add **`saphyr`** as a **`[dev-dependencies]` of `axc-driver`** (test-only; zero touches to any `src/`, satisfies the "no runtime code" non-scope). Justification for `saphyr` over the alternatives:
- **`serde_yaml` — rejected:** archived/deprecated by dtolnay; do not introduce a dead dependency.
- **`saphyr` — chosen:** the actively-maintained current-generation pure-Rust YAML crate from the `yaml-rust2` maintainer (`saphyr` is the successor project; `yaml-rust2` is its maintenance-mode predecessor). It exposes a real YAML AST (`saphyr::Yaml`), which is what the structural ATs need — they inspect `if:` / `runs-on:` / `permissions:` / `on:` / `timeout-minutes` **as parsed nodes**, not fragile substring greps. Either `saphyr` or `yaml-rust2` is acceptable to the Coder; `saphyr` is the recommendation on maintenance-status grounds.

**No-vacuous-pass contract (AT-3033):** the test loads the parser unconditionally. If a file fails to parse, the test **panics/fails** — there is no `#[cfg]` skip, no "parser absent → pass" branch. A dedicated meta-assertion confirms the parse actually ran (e.g. asserts the parsed doc set is non-empty and the expected top-level `jobs` key resolved), so a security AT can never silently no-op. `actionlint` (§6.4) is now **belt-and-suspenders schema-linting only**; the authoritative structural/security checks live in this Rust test with the pinned parser.

### 6.4 The `lint-workflows` CI job (actionlint, r2 supply-chain-pinned)

`ci.yml`'s new `lint-workflows` job runs `actionlint` over `.github/workflows/*.yml` for schema-level linting (bad `runs-on`, unknown keys, expression errors). To avoid an unpinned network binary fetch on CI, actionlint is **pinned by version AND checksum-verified** (download the pinned `rhysd/actionlint` release, verify its SHA-256 against a committed checksum before executing). If that verification is undesirable overhead, actionlint may be **dropped entirely** — the authoritative parse/structure/security coverage is the `saphyr`-backed `eb1_workflow_ci.rs` (§6.3), which already validates parse-ability. actionlint is never load-bearing for the security ATs.

**Dependency graph:** YAML + Markdown + one CPU-only Rust test + one **test-only** dev-dependency (`saphyr`). No crate-graph edges into any `src/`; no runtime-code change. `eb1_workflow_ci.rs` reads files from the repo root (via `CARGO_MANIFEST_DIR/../..`) and parses them with `saphyr` — unconditionally, no fallback, no silent skip.

---

## 7. Validation without hardware (the honest testable surface)

### 7.1 What IS validatable on CPU-only CI (this milestone's test plan)
- **YAML schema validity** of all 3 new workflow files + `ci.yml` — `actionlint` (job-level, understands the Actions schema, catches bad `runs-on`, unknown keys, expression errors — supply-chain-pinned per §6.4) and an in-test **`saphyr`** parse (§6.3, unconditional — no silent no-op).
- **Structural / security invariants** by parsing the YAML: activation-var `if` present on every vendor job; same-repo fork guard present on **every self-hosted job in every workflow file** (incl. `pytorch-interop`); `pull_request_target` absent everywhere; `permissions: contents: read`; `timeout-minutes` present on every self-hosted job; A/B + bless jobs are `workflow_dispatch`-only; bench-gate job never sets `AXC_BLESS_BASELINES: "1"`.
- **Vendor-independence proof (the stand-in for real AMD/Intel):** the vendor job's *script body* is proven device-agnostic by running the **same** `bench_regression` + `--ignored` commands on the **existing Lavapipe (CI) / local NVIDIA** runner and observing the **EB.2 machine-key mechanism** select the right block by `deviceName`. Because the gate keys off the probed device, the *identical* job YAML that gates NVIDIA/Lavapipe will gate AMD/Intel with no per-vendor script logic — the machine-key indirection is the vendor-independence, and it is already green (AT-2836/2837, AT-EB2-01). We assert this linkage rather than claiming to have run on AMD.
- **Docs completeness anchors:** grep `EB1_RUNNER_SETUP.md` for the required section headers (RADV justification, ANV, labels, Activation variables, fork-PR security, first-run bless, Mesa pin).
- **llama pin match:** the A/B workflow's pinned SHA/tag equals the one in `scripts/m34_llamacpp_ab.sh` / M3.4 (`b9542` / `6b80c74…`).

### 7.2 What is NOT validatable without hardware (stated plainly)
- Actual AMD/Intel test execution, the resident bench numbers, the A/B ratio, the RADV/ANV coopmat behavior, the real Mesa segfault on RADV. These are EB.1-proper, hardware-gated. This milestone asserts the *files are correct and the activation is wired*, not that AMD/Intel passed.
- **`act`** is assessed and **rejected** as a validation tool here: it cannot emulate self-hosted-runner label matching, the repo-variable/environment gates, or a real GPU — it would give false confidence. `actionlint` (schema) + the in-repo parse/structure test + the Lavapipe/NVIDIA script smoke are the honest tools.

---

## 8. Acceptance tests (AT-3019 .. AT-3035) — all CPU-only, run in normal CI

| ID | Location | Assertion |
|---|---|---|
| **AT-3019** | `eb1_workflow_ci.rs` | All of `cross-vendor-ci.yml`, `cross-vendor-ab.yml`, `cross-vendor-bless.yml`, and the modified `ci.yml` **parse as valid YAML** via the pinned **`saphyr`** parser (§6.3); no tabs, no duplicate keys. |
| **AT-3020** | `eb1_workflow_ci.rs` | Every vendor job in `cross-vendor-ci.yml` has an `if:` containing `vars.AMD_RUNNER_ONLINE == 'true'` / `vars.INTEL_RUNNER_ONLINE == 'true'` — the **anti-queue-forever** guard. (Parse the `if` string, assert the token present.) |
| **AT-3021** | `eb1_workflow_ci.rs` | Every vendor job's `if:` also contains the same-repo fork guard `head.repo.full_name == github.repository` (or the job is `workflow_dispatch`-only) — the **fork-PR** control. |
| **AT-3022** | `eb1_workflow_ci.rs` | `runs-on` of the vendor jobs are exactly `[self-hosted, amd-rdna3]` and `[self-hosted, intel-arc]`; labels match the docs. |
| **AT-3023** | `eb1_workflow_ci.rs` | `cross-vendor-ab.yml` and `cross-vendor-bless.yml` declare `on:` == **`workflow_dispatch` only** (no `pull_request`/`push`), so they cannot queue on a PR. |
| **AT-3024** | `eb1_workflow_ci.rs` | The string `pull_request_target` appears in **none** of the new workflow files (negative security anchor). |
| **AT-3025** | `eb1_workflow_ci.rs` | Every new job sets `permissions:` with `contents: read` (or narrower) and no vendor job references `secrets.*` (least-privilege). |
| **AT-3026** | `eb1_workflow_ci.rs` | The per-PR bench-regression job **never** sets `AXC_BLESS_BASELINES: "1"` (AT-714b discipline preserved on the cross-vendor gate). |
| **AT-3027** | `eb1_workflow_ci.rs` | The A/B workflow pins llama.cpp to tag `b9542` / SHA `6b80c74f285390368b3c99c5e750f19e9b096e98` — **equals** the pin in `scripts/m34_llamacpp_ab.sh`/M3.4 (read both, assert equality; guards pin drift). |
| **AT-3028** | `eb1_workflow_ci.rs` | The vendor GPU-test step invokes `cargo test --workspace -- --ignored` with `AXC_ENABLE_GPU_TESTS=1` and a vendor `VK_DRIVER_FILES` (radeon_icd / intel_icd) — parity with the Lavapipe `test` job command. |
| **AT-3029** | `eb1_workflow_ci.rs` | `docs/EB1_RUNNER_SETUP.md` exists and contains all required section anchors: `RADV`, `AMDVLK` (as fallback), `ANV`, `amd-rdna3`, `intel-arc`, `AMD_RUNNER_ONLINE`, `workflow_dispatch`, fork/security wording, `AXC_BLESS_BASELINES`, a **branch-protection / required-checks** guidance anchor (§1.4: do-not-mark-required-pre-activation + no-`paths:`-filter), an **ephemeral/JIT + non-root REQUIRED** anchor, and `Mesa` / `25.2.8`. |
| **AT-3030** | `eb1_workflow_ci.rs` | `ci.yml` contains the `lint-workflows` job (actionlint, supply-chain-pinned per §6.4) and the non-fatal `mesa`-version probe step (Mesa-pin fold-in surfaced). |
| **AT-3031** | `eb1_workflow_ci.rs` | **Vendor-independence linkage:** assert the vendor bench-gate step and the existing Lavapipe bench-gate step invoke the **same** `cargo test --release -p axc-driver --test bench_regression` entrypoint (byte-equal modulo the ICD env), proving no per-vendor script branch exists — the device selection is delegated entirely to EB.2's `machine_key`. |
| **AT-3032** | `eb1_workflow_ci.rs` | **ALL-self-hosted-jobs security scan (closes BLOCKER-1, regression-proof):** enumerate **every** job with a `self-hosted` `runs-on` label across **all** `.github/workflows/*.yml` (glob — incl. the pre-existing `pytorch-interop`). Each MUST carry (a) the same-repo fork guard `head.repo.full_name == github.repository` **or** be `workflow_dispatch`-only, (b) a `permissions:` block no broader than `contents: read`, and (c) a `timeout-minutes`. Fails if any self-hosted job (present or future) omits any of the three. |
| **AT-3033** | `eb1_workflow_ci.rs` | **No-vacuous-pass meta-assertion (closes BLOCKER-2):** the `saphyr` parse ran and produced structure — assert the parsed workflow set is non-empty and the top-level `jobs` node resolved for each file. There is **no `#[cfg]` skip / parser-absent branch**; a missing parser is a compile error, a parse failure panics. Guarantees the security ATs (3020/3021/3024/3025/3026/3032) can never silently no-op. |
| **AT-3034** | `eb1_workflow_ci.rs` | Every self-hosted job across all workflow files declares `timeout-minutes` within sane bounds (per-PR test/bless ≤ ~30; A/B ≤ ~90; `pytorch-interop` == 30) — bounds sole-GPU-box lockup. (Overlaps AT-3032(c); asserted explicitly with the value bounds.) |
| **AT-3035** | `eb1_workflow_ci.rs` | **Mesa honesty:** `EB1_RUNNER_SETUP.md` frames Mesa as **`25.2.8` known-bad** AND contains an explicit `UNKNOWN`/`TBD`/`untested until EB.1 hardware` framing for the known-**good** RADV/ANV version — asserts NO fabricated good-version range was invented (no hardcoded "known-good X.Y.Z" claim). |

**QA note:** every AT above runs green on `ubuntu-latest` with **no GPU** — the milestone's exit gate is CPU-only-verifiable. The hardware ATs (AMD/Intel tests actually passing, the A/B numbers) are explicitly **deferred to EB.1-proper** and listed as `EB.1-HW-01..` placeholders in the spec, not claimed here.

---

## 9. Correctness oracle

Not a numerical milestone — no `@equiv_fp_tol`. The "oracle" is: (1) the workflow files are schema-valid and carry the security/activation invariants (AT-3019..3035, machine-checked); (2) the vendor jobs reuse the *already-proven* EB.2 machine-key gate and the *already-proven* `scripts/m34_llamacpp_ab.sh` verbatim — so correctness of the *measurement* is inherited, not re-implemented. No new tolerance, no FROZEN-1e-3 change.

## 10. Edge cases

- **Runner variable set but runner offline:** `if` passes → job queues briefly then GitHub errors "no runner" after the org timeout. Mitigation: docs instruct flipping the variable to `false` whenever the runner is taken down (the variable is the source of truth, kept in sync with runner health).
- **Both a probe GPU and the target GPU present on the runner:** `AXC_PHYSICAL_DEVICE_INDEX` selects the target; documented.
- **AMD wave64:** coopmat/subgroup-32 tests typed-skip (not fail); log records skips (§4.1). Not an error.
- **New device key drift** (driver rename changes `deviceName`): EB.2's `LoudSkipKeyAbsent` loudly skips + instructs re-bless; no silent gate loss.
- **Fork PR:** vendor jobs skip (fork code never runs); Lavapipe CI still covers the PR.
- **actionlint binary unavailable / checksum-mismatch in CI:** actionlint is schema-lint-only (§6.4) and may be dropped; the authoritative parse/structure/security coverage is the `saphyr`-backed `eb1_workflow_ci.rs` (§6.3), which is unconditional and panics on failure — never a silent skip.
- **`skip_llama_build=true` with no/stale cached build:** the §3 preflight guard fails loudly (asserts the cached checkout exists AND its SHA == the pin) instead of the obscure downstream "missing `test-backend-ops`".

## 11. Open questions — RESOLVED in r2 (design-review rulings folded)
1. **Ephemeral vs long-lived runner → REQUIRED (upgraded from "strongly recommend").** Per the optimistic ruling (and reinforced by BLOCKER-1's demonstration of an already-exposed persistent runner): the same-repo `if:` guard is the *sole* primary control preventing fork code from executing on a persistent GPU box, and there is no `environment:` approval layer and no branch protection on `main` today. `EB1_RUNNER_SETUP.md` therefore **mandates** an ephemeral / just-in-time runner (or a disposable VM/container per job) **and non-root** operation **and** "Require approval for all outside collaborators" — stated as requirements, not recommendations, so a hobbyist single-box setup cannot casually skip the only backstop. (AT-3029 anchors the ephemeral+non-root requirement wording.)
2. **Environment-approval tier → DOCS-ONLY (confirmed).** Do **not** wire an `environment:` into `cross-vendor-ci.yml` now. Rationale (optimistic ruling): the same-repo guard already fully excludes fork-PR code from reaching the runner; a required-reviewer `environment:` on top is a redundant second gate for already-trusted same-repo collaborators and adds merge friction with no runner yet registered to validate it against. It is documented as the upgrade path for maintainers who *want* fork PRs to run post-approval — one gate, not two.
3. **Windows/AMD path → OUT OF SCOPE (confirmed, now explicit non-goal in §0).** Linux + Mesa (RADV/ANV) only, matching the beachhead; a Windows runner needs a different ICD stack and its own security review. Recorded as a deliberate non-goal in §0, not an oversight.

---

## 12. LOC estimate (restated for r2)

Non-code milestone (YAML + Markdown + one CPU-only test + one dev-dependency line). Net additions:

| Artifact | Est. LOC |
|---|---|
| `cross-vendor-ci.yml` | ~110 |
| `cross-vendor-ab.yml` | ~90 |
| `cross-vendor-bless.yml` | ~60 |
| `ci.yml` edits (pytorch-interop hardening + `lint-workflows` job + mesa probe) | ~40 |
| `crates/axc-driver/tests/eb1_workflow_ci.rs` (17 ATs, saphyr-backed) | ~520 |
| `crates/axc-driver/Cargo.toml` dev-dep | ~1 |
| `docs/EB1_RUNNER_SETUP.md` | ~430 |
| `ROADMAP.md` / `UPSTREAM_PR_PLAN.md` doc edits | ~15 |
| **Total (net additions)** | **~1,266 LOC** |

**Zero** LOC in any `crates/axc-*/src` — the invariant holds (the only Rust is a test; the only manifest change is a test-only dev-dependency).

---

## r2-changelog (surgical revision after design review)

**From:** r1 — optimistic APPROVE (with rulings); pessimistic NEEDS_REVISION (2 CRITICAL + 3 must-fix + warnings).

1. **CRITICAL / BLOCKER-1 — the LIVE `pytorch-interop` fork hole (closed).** Added §2.3: `ci.yml`'s existing `[self-hosted, nvidia, cuda]` job runs fork PR code on the maintainer's GPU box TODAY (no fork guard, no `permissions:`, no timeout). Now hardened in-place (same-repo `if:` + `permissions: contents: read` + `timeout-minutes: 30`, `continue-on-error` preserved). Added §2.4 + **AT-3032**: the security assertion is a **scan of every self-hosted job in every `.github/workflows/*.yml`** (glob, incl. `pytorch-interop` and any future job) — not a per-file list — so a regressed guard fails CI. Added to §0 scope.
2. **CRITICAL / BLOCKER-2 — vacuous security ATs (closed).** Added §6.3: no YAML parser was vendored and `serde_yaml` is deprecated. **Mechanism PINNED to `saphyr`** (maintained successor to `yaml-rust2`, real AST) as a **test-only `[dev-dependencies]`** of `axc-driver`. The `#[cfg]`-guarded silent-no-op fallback is **FORBIDDEN** — the parse is unconditional; a missing parser is a compile error, a parse failure panics. Added **AT-3033** (no-vacuous-pass meta-assertion). serde_yaml explicitly rejected.
3. **Timeouts (must-fix).** `timeout-minutes` on every self-hosted job: per-PR GPU test/bless = **30**, A/B (llama source build) = **90**, `pytorch-interop` = **30**. §§2.3/3.1/4.1/4.2 + **AT-3034**.
4. **Mesa honesty (vendor must-fix).** §5.3 reworded: **`25.2.8` known-bad ONLY**; known-good RADV/ANV Mesa is **UNKNOWN/TBD until EB.1 hardware** — no invented range; capture the empirically-good version at first EB.1 run. **AT-3035** asserts the honesty framing.
5. **Branch-protection posture (must-fix).** New §1.4: `if:`-level-skip == PASS vs never-triggered == PENDING-FOREVER (the `paths:`-filter trap); don't mark vendor jobs required pre-activation; **never add `paths:` filters** to these workflows. Anchored in AT-3029.
6. **Optimistic rulings folded (§11 resolved).** Ephemeral/JIT + non-root now **REQUIRED** (not recommended); environment-approval **docs-only** (why: redundant second gate for trusted same-repo PRs); **Windows out-of-scope** as an explicit non-goal in §0.
7. **Warnings lowered.** §3: A/B **evidence-provenance** chain documented (dispatch `github.actor` + runner logs + embedded `vulkaninfo` device string; bounded, not a proof); **`skip_llama_build` guarded** (preflight asserts cached build present AND SHA == pin). §6.4: `actionlint` **checksum-pinned** or dropped in favor of the `saphyr` parser (never load-bearing for security ATs).
8. **AT range grown** AT-3019..**AT-3035** (added 3032–3035; ceiling AT-3035 as authorized). **LOC restated** in §12: **~1,266 net additions, zero `src/` LOC**.
