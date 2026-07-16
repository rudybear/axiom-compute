# EB.1 Runner Setup — AMD (RDNA3/RADV) + Intel (Arc/ANV) self-hosted GitHub Actions runners

**Status:** EB.1-prep (workflows + docs authored, activation wired). Actual AMD/Intel
measurement is hardware-gated — this document is the operational checklist for the day a
runner physically registers. See `.pipeline/milestones/EB1prep-cross-vendor-ci.md` for the
full design rationale and `ROADMAP.md` / `UPSTREAM_PR_PLAN.md` for how this feeds the
cross-vendor kill-criterion and the M4.2 upstream RFC trigger.

`axiom-compute` is a **public repository**. Everything in this document assumes a
self-hosted runner is a real, persistent GPU machine on the maintainer's network — read
the **Fork-PR security** section before registering anything.

---

## 1. Overview — what gets registered, and why two vendors

Two independent self-hosted runner "slots":

| Vendor | GPU family | Vulkan driver | Runner labels |
|---|---|---|---|
| AMD | RDNA3 (or newer) | **RADV** (Mesa) | `[self-hosted, amd-rdna3]` |
| Intel | Arc (Xe-cores) | **ANV** (Mesa) | `[self-hosted, intel-arc]` |

Both drivers are **Mesa**, both are Linux-only in this milestone (Windows is an explicit
non-goal — see §8). Neither runner needs to exist for this repository's CI to stay green:
every job that targets them is gated by a repository variable that defaults to "off" (§3).

---

## 2. AMD — driver choice: RADV (justified), AMDVLK as fallback

Install **RADV**, Mesa's open-source AMD Vulkan driver, as the primary driver:

```sh
sudo apt-get install -y mesa-vulkan-drivers vulkan-tools libvulkan-dev
```

ICD JSON: `/usr/share/vulkan/icd.d/radeon_icd.x86_64.json`.

**Why RADV over AMDVLK / `amdvlk-pro`:**
- It is the driver **llama.cpp's own Vulkan CI and the overwhelming majority of AMD Vulkan
  users run** — comparing AXIOM against RADV is an apples-to-apples portability claim
  against the same stack llama.cpp is tuned and measured on.
- It ships in-distro (`mesa-vulkan-drivers`), no out-of-tree package to track.
- It has the most mature `VK_KHR_cooperative_matrix` support on RDNA3.
- It is what the kill-criterion audience actually uses (ggml-org/llama.cpp #16230,
  ollama #15601 — AMD users leaving throughput on the table on the stack they already run).

**AMDVLK** (AMD's own open-source driver) is documented here as the **fallback** if a
specific coopmat code path regresses on RADV: `sudo apt-get install -y amdvlk` (or the
GitHub release tarball), ICD JSON typically at `/etc/vulkan/icd.d/amd_icd64.json`. Select
between ICDs with `VK_DRIVER_FILES` / `VK_ICD_FILENAMES` — do not install both
simultaneously with `AXC_PHYSICAL_DEVICE_INDEX` unset (device enumeration can double-count
the GPU with two loaded ICDs targeting it).

## 3. Intel Arc — driver choice: ANV

Install **ANV**, Mesa's Intel Vulkan driver — the only practical open Arc Vulkan driver on
Linux:

```sh
sudo apt-get install -y mesa-vulkan-drivers vulkan-tools libvulkan-dev
```

ICD JSON: `/usr/share/vulkan/icd.d/intel_icd.x86_64.json`. Arc needs a **recent Mesa
(>= 24.x)** for stable `VK_KHR_cooperative_matrix` / Xe-cores support — see §7 (Mesa
version pin) before assuming any specific version is safe.

If the box has an integrated Intel GPU alongside the Arc discrete GPU, select the Arc
device explicitly with `AXC_PHYSICAL_DEVICE_INDEX` (see §6).

---

## 4. Runner registration + labels

Register **at repo scope** (not org scope — narrower blast radius, easier to reason about
who can dispatch jobs against it):

```sh
# From the repo's Settings -> Actions -> Runners -> New self-hosted runner, or via API:
./config.sh --url https://github.com/rudybear/axiom-compute --token <REG_TOKEN> \
    --labels amd-rdna3 --name amd-rdna3-01          # AMD box
./config.sh --url https://github.com/rudybear/axiom-compute --token <REG_TOKEN> \
    --labels intel-arc --name intel-arc-01          # Intel box
```

The workflows in `.github/workflows/` target `[self-hosted, amd-rdna3]` and
`[self-hosted, intel-arc]` — the label **must** match exactly (a typo'd label is
indistinguishable from "no runner" and queues forever — see §5).

---

## 5. The activation mechanism (avoiding "queue-forever")

A job whose `runs-on:` targets a label set that no registered runner satisfies **does not
skip — it queues indefinitely**. On `pull_request`, a pending required check **blocks the
merge button forever**. This is why none of the EB.1-prep workflows put the vendor gate in
`runs-on:` — the gate is a job-level `if:` on a **repository variable**:

```yaml
amd-gpu:
  runs-on: [self-hosted, amd-rdna3]   # ALWAYS this literal label set
  if: vars.AMD_RUNNER_ONLINE == 'true' && ...     # the actual gate
```

### Activation switch (do this the moment a runner is registered and healthy)

Repo **Settings -> Secrets and variables -> Actions -> Variables -> New repository
variable**:

| Name | Value | Effect |
|---|---|---|
| `AMD_RUNNER_ONLINE` | `true` | `cross-vendor-ci.yml`'s `amd-gpu` job (and the AMD side of the A/B / bless workflows) starts running instead of skipping |
| `INTEL_RUNNER_ONLINE` | `true` | same, for `intel-arc` |

No workflow edit, no merge, no code change — flipping the variable is the entire activation
step. **Turning a vendor off** (runner down for maintenance): flip the variable back to
`false`. PRs immediately stop waiting on it (jobs go back to skipping).

**Keep the variable in sync with runner health.** If the variable is `true` but the runner
is offline, the job queues until GitHub's own runner-pickup timeout errors it out — the
variable, not the runner's actual uptime, is the source of truth CI relies on.

---

## 6. Device selection

If a box has more than one Vulkan-capable GPU (e.g. an integrated + a discrete GPU),
`AXC_PHYSICAL_DEVICE_INDEX=<N>` selects which one AXIOM's `VulkanContext` opens. Set it in
the runner's own environment (not per-workflow) if the index is stable, or as a job `env:`
override otherwise.

---

## 7. Mesa version — what we actually know (honesty, no invented ranges)

M3.23 root-caused a **Lavapipe SIGSEGV / heap corruption on Mesa 25.2.8** when
workgroup-**shared**-memory kernels are dispatched via the **resident** harness (NVIDIA is
unaffected — this is a Mesa-side bug, not an AXIOM correctness bug). RADV and ANV are
**also Mesa**, and the `cross-vendor-ab.yml` A/B job dispatches exactly those shared-memory
resident kernels.

- **`25.2.8` is KNOWN-BAD.** Do not run the vendor runners on this Mesa version.
- **There is NO known-good RADV/ANV Mesa version documented here.** That claim would be
  fabricated — the only evidence we have is the Lavapipe measurement above; RADV/ANV
  behavior on the same code path is **UNKNOWN / TBD until EB.1 hardware** actually runs it.
  Do not assume, guess, or pin a "known-good" version from thin air.
- **Operator instructions:**
  1. Avoid Mesa `25.2.8` on the vendor runner.
  2. At the **first successful** EB.1 run on real AMD/Intel hardware, capture
     `vulkaninfo | grep driverInfo` output and record that exact Mesa version **here** as
     the empirically-good pin, replacing this paragraph. Until that happens, treat every
     Mesa version other than the confirmed-bad `25.2.8` as **untested, not "safe."**
- Both the vendor CI workflows and the existing Lavapipe `test` job in `ci.yml` run a
  **non-fatal** `vulkaninfo | grep driverInfo` probe step so a bad Mesa bump shows up in
  the log instead of a mystery segfault.

---

## 8. Fork-PR security (self-hosted runners on a PUBLIC repo)

A self-hosted runner is a **persistent machine on the maintainer's network holding a real
GPU**. The canonical, severe risk on a public repo: a `pull_request` from a **fork**
carries attacker-controlled workflow/build/test code. If that code executes on a
self-hosted runner, it is **arbitrary code execution on the maintainer's hardware**
(secret theft, lateral movement, runner persistence). This is GitHub's own stated primary
reason self-hosted runners on public repos are dangerous.

### Layered mitigations (all applied in this repo's workflows)

1. **Fork PRs never reach the runner (primary control).** Every self-hosted job's `if:`
   includes the same-repo clause `github.event.pull_request.head.repo.full_name ==
   github.repository`, which is **false for any fork PR** — the job is *skipped*, so
   fork-controlled code is never checked out or executed on the machine. Same-repo PRs
   (pushed by collaborators with write access, already trusted) run normally once the
   runner is online.

   > **Forward-defensive note (this is a fork guard, not a trust guard):** the same-repo
   > check only asks "did this PR come from a branch of THIS repository?" — it does **not**
   > vet the identity or intent of whoever pushed that branch. **Dependabot PRs are
   > same-repo** (they land as branches on `axiom-compute` itself, not forks) and **would
   > pass this guard** even though their diff content (a bumped dependency's build script,
   > a transitive `Cargo.lock`/`package-lock.json` change) is not authored by a human
   > maintainer. If Dependabot (or any other bot that opens same-repo PRs) is ever enabled
   > on this repository, review its PRs before merge with the same scrutiny as external
   > code — the fork guard will not stop it from reaching the self-hosted runner.

2. **Never `pull_request_target`.** All workflows use `pull_request` (restricted context:
   read-only `GITHUB_TOKEN`, no secrets on fork events), never `pull_request_target` (which
   runs the *trusted* workflow with repo secrets against fork-checked-out code — the
   classic footgun). Verified by AT-3024: the string `pull_request_target` appears in none
   of the new workflow files.

3. **Optional stricter tier — environment approval.** Maintainers who want fork PRs to run
   after a human clicks "approve" can wrap the vendor jobs in a GitHub `environment:` with
   required reviewers (deployment-protection rules pause the job before a runner picks it
   up). This repo ships the same-repo-skip default instead (fork code never runs at all) —
   deliberately **one gate, not two**, since a required-reviewer `environment:` on top of
   an already-fully-excluding same-repo guard is redundant friction for already-trusted
   same-repo collaborators.

4. **Least privilege.** Every self-hosted job sets `permissions: { contents: read }` (no
   broader) and none of the EB.1-prep vendor jobs reference `secrets.*` — the tests, the
   bench gate, and the A/B harness need no secrets (llama.cpp and the ICD live on the
   runner already).

5. **Runner hardening — REQUIRED, not optional:**
   - **Ephemeral / just-in-time runner, or a disposable VM/container per job — REQUIRED.**
     There is currently no `environment:` approval layer and no branch protection on
     `main` (verified: `gh api …/branches/main/protection` -> 404) — the same-repo `if:`
     guard is the **sole** primary control keeping fork code off this machine. A
     long-lived, always-on runner with no other backstop is not an acceptable
     configuration for this repo; use GitHub's ephemeral runner mode or tear the VM/
     container down and rebuild it between job runs.
   - **Non-root — REQUIRED.** Run the runner service as an unprivileged, non-root user.
     Never register a runner running as `root`.
   - Register at **repo scope**, not org scope (§4).
   - Keep no long-lived credentials on the box.
   - Enable **Settings -> Actions -> "Require approval for all outside collaborators"** as
     a belt-and-suspenders backstop on top of the same-repo guard.

### Honest limit

The same-repo-skip means the **per-PR AMD/Intel gate covers collaborator (same-repo) PRs,
not fork PRs** — a deliberate security/utility tradeoff. Fork contributions still get full
Lavapipe CI coverage (safe on hosted `ubuntu-latest` runners); their cross-vendor coverage
happens after a maintainer pushes the branch into the repo, or a maintainer runs the
`workflow_dispatch` A/B manually.

### Non-goal: Windows/AMD path

This document and the EB.1-prep workflows cover **Linux + Mesa (RADV/ANV) only**, matching
the project's beachhead. A Windows runner needs a different ICD stack (native AMD/Intel
Vulkan, no Mesa) and its own, separate security review — this is a deliberate non-goal
recorded here, not an oversight.

---

## 9. Branch-protection posture (read before enabling required checks)

GitHub's required-status-check semantics have two behaviors that look similar but are
**not** — conflating them is exactly how a "safe skip" design turns into a merge-blocking
trap:

- **`if:`-level skip == PASS.** A job that is *skipped* because its job-level `if:`
  evaluated false — inside a workflow that **did** trigger — reports its required status
  check as **successful**. This is why the vendor jobs are safe to leave in the repo while
  offline: they skip green, they never queue.
- **Never-triggered == PENDING-FOREVER.** The dangerous, *different* case is a required
  check whose **workflow never triggered at all** (e.g. a `paths:` filter excluded the
  PR's changed files). GitHub then leaves that required check **Pending indefinitely**, and
  a pending required check **blocks the merge button forever**.

**Therefore:**

1. **Do NOT mark `amd-gpu` / `intel-gpu` (or the A/B / bless jobs) as required status
   checks until AFTER the first successful post-activation run on real hardware.** A
   vendor job made "required" while its runner is offline is fine under job-level-skip
   semantics, but making it required *before it has ever produced a check* risks the
   pending trap on any tooling that treats an absent check as pending rather than skipped.
2. **Never add a `paths:` or path-ignore filter to `cross-vendor-ci.yml`,
   `cross-vendor-ab.yml`, or `cross-vendor-bless.yml`.** A paths-filtered workflow that
   doesn't trigger on a given PR yields the pending-forever trap for any required check it
   declares. These workflows trigger on `push`/`pull_request`/`workflow_dispatch`
   **unconditionally** — the job-level `if:` does the gating, not a workflow-level filter —
   and this must be preserved on any future edit.
3. `main` currently has **no branch protection at all** — none of this blocks merges
   today; this guidance is forward-defensive for whenever protection is enabled.

---

## 10. What runs per PR (once activated)

Per vendor (`amd-gpu`, `intel-gpu` in `cross-vendor-ci.yml`), `timeout-minutes: 30`:

- **GPU tests:** the full `cargo test --workspace -- --ignored` GPU-dispatch suite under
  `AXC_ENABLE_GPU_TESTS=1`, `AXC_REQUIRE_SPIRV_VAL=1`, and the vendor `VK_DRIVER_FILES`.
  We do not hard-code individual test names; we run the whole `--ignored` set and require
  zero failures.

  **Honest cross-vendor caveat:** a large part of the `#[ignore]` GPU suite is coopmat /
  subgroup-size-32-gated and **typed-skips by construction** on non-wave32 devices (see
  `DESIGN.md §754`). On **AMD RDNA3 (wave64 default)** and **Intel Arc (SIMD8/16)** those
  tests will typed-skip, not fail — that is expected, not a regression. The
  portable-everywhere subset (saxpy, vector_add, plain multi-tile matmul, the
  non-`#[ignore]` `dispatch_q6k_matmul` K=256) is the real cross-vendor correctness signal
  on all three vendors. "Green on AMD" should never be read as "every coopmat kernel ran
  on AMD" — check the run log for which tests typed-skipped.

- **Bench regression gate:** `cargo test --release -p axc-driver --test bench_regression`
  with the vendor ICD, `AXC_ENABLE_BENCH_REGRESSION=1`, and `AXC_BLESS_BASELINES=""`
  (**never** `"1"` on this per-PR path — AT-714b discipline). A brand-new AMD/Intel device
  is an absent EB.2 machine key: the gate **skips with a loud warning** rather than
  failing, until a human blesses the first baseline (§11).

---

## 11. First-run bless procedure

Blessing is **never automatic**. `cross-vendor-bless.yml` is a dedicated
`workflow_dispatch`-only job — the ONLY place `AXC_BLESS_BASELINES` is ever set to `"1"` in
this repo's CI.

1. Confirm the vendor's `*_RUNNER_ONLINE` variable is `true` and the runner is healthy.
2. From the **Actions** tab, run **"Cross-vendor baseline bless (manual, first-run)"**,
   choosing `vendor: amd` or `vendor: intel`.
3. The job runs `cargo bench -p axc-driver` with `AXC_BLESS_BASELINES=1` on the vendor
   runner. EB.2's `merge_blessed` performs a read-modify-write: it adds a **new** machine
   block keyed by `sanitize(deviceName)` and **preserves every other existing key**
   (NVIDIA/Lavapipe blocks are never overwritten).
4. Download the `baselines.json` artifact the job uploads.
5. **A maintainer reviews the diff** — confirm exactly one new top-level machine key was
   added, the device name matches the vendor, and the numbers are plausible (not zero, not
   absurdly small/large) — then commits it via a **normal PR**.
6. The runner **never** writes to the repository directly; the commit is always a
   human-reviewed action.

After that PR merges, the AMD/Intel key exists in `baselines.json` -> the per-PR
`bench-regression` gate transitions from "loud skip" to actually enforcing on the next run,
automatically, with no workflow change.

---

## 12. The llama.cpp A/B evidence job

`cross-vendor-ab.yml` is `workflow_dispatch`-only and produces the artifact that feeds
`UPSTREAM_PR_PLAN.md §7.5`'s AMD/Intel RFC-trigger row: pinned llama.cpp `b9542` /
`6b80c74f285390368b3c99c5e750f19e9b096e98`, `scripts/m34_llamacpp_ab.sh`, uploaded
`ab_results*.json` + run log + `vulkaninfo --summary`. It has **no** pass/fail ratio gate —
it is measurement, not a CI gate; it records whatever the numbers are.

---

## 13. Validating this setup WITHOUT the hardware (what CI already checks today)

Everything above is validated at the file/schema level on ordinary `ubuntu-latest` CI, with
no GPU:

- `crates/axc-driver/tests/eb1_workflow_ci.rs` parses every workflow file with the pinned
  `saphyr` YAML parser (unconditional — no silent skip if the parser is missing, that would
  be a compile error) and asserts the activation gate, the fork guard, the absence of
  `pull_request_target`, `permissions: contents: read`, `timeout-minutes`, and the
  never-`AXC_BLESS_BASELINES: "1"`-on-the-per-PR-path invariant.
- `ci.yml`'s `lint-workflows` job runs a checksum-pinned `actionlint` over
  `.github/workflows/*.yml` for schema-level linting.
- **`act` is deliberately not used** — it cannot emulate self-hosted-runner label matching,
  repo-variable/environment gates, or a real GPU; it would give false confidence. The
  `saphyr`-backed test + `actionlint` + the Lavapipe/NVIDIA script smoke are the honest
  validation surface until real hardware registers.
