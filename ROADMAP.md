# AXIOM-Compute Roadmap

This document is the **comprehensive remaining work plan** as of 2026-04-28. The project's first 13 milestones (M0 → M2.6) are merged on `main`; this plan covers what's left to reach the thesis claims in `DESIGN.md` §5 (kill criteria) and beyond.

Last updated: 2026-04-28. Test count baseline: **713**.

---

## Status snapshot

| Capability | Status |
|---|---|
| Compile `.axc` → SPIR-V → spirv-tools::val | ✅ done (M0–M1.4) |
| Vulkan dispatch on real GPU | ✅ done (M1.5) |
| `@strategy` holes + grid-search autotuner | ✅ done (M2.3) |
| MCP server for LLM agents | ✅ done (M2.4) |
| Q4_0 + Q4_K_M kernels bit-exact on NVIDIA RTX PRO 6000 | ✅ done (M2.5–M2.6) |
| Cooperative matrix codegen | ✅ done (M2.1) — dispatch on tensor-core HW deferred |
| Bench harness + measured baselines | ✅ done (M2.2) |
| **FlashAttention-2** | ❌ M3.1 |
| **KernelBench-Vulkan public submission** | ❌ M3.2 |
| **PyTorch frontend + upstream adoption** | ❌ M4 |
| Bandwidth optimization (pinned memory, concurrent transfer) | ❌ M3.0 |
| Multi-row tiled matmul (cooperative_matrix on real workloads) | ❌ M3.0 |
| Cross-vendor real GPU CI (AMD RDNA3+, Intel Arc) | ❌ infra |
| **llama.cpp Vulkan head-to-head A/B** | ❌ thesis-closing milestone |

---

## Phase M3 — Performance + ML kernels (the thesis-closing phase)

Goal: prove the DESIGN.md §5 kill-criteria gates with publishable numbers, not just bit-exact correctness.

### M3.0 — Dispatch bandwidth rework (PREREQUISITE for everything else)

**Why first:** every measured GPU number on real hardware (NVIDIA RTX PRO 6000) is staging-bound at 1 M+ elements. saxpy_1m at 23 ms is ~100× off theoretical PCIe peak. Without fixing this, M3.1/M3.2/M4 numbers are dominated by infrastructure overhead, not kernel quality.

**Scope in:**
- Pinned host memory via `VK_EXT_host_memory_alloc_placement` or fallback `mmap` + `vkMapMemory` with `HOST_VISIBLE | HOST_CACHED`
- Concurrent transfer queue: dedicated transfer queue family if present, with semaphore handoff to compute queue
- `vkCmdCopyBuffer` overlap with compute (transfer queue submit pre-warms the device-local while compute runs on previous tile)
- `VK_KHR_buffer_device_address` for direct GPU memory access where supported (NVIDIA, AMD)
- Optional: persistent mapped staging ringbuffer (avoid per-dispatch map/unmap cost)

**Acceptance:**
- saxpy_1m on NVIDIA RTX PRO 6000 drops from 23 ms to **< 1 ms** (~25× speedup expected — closes most of the 100× peak gap)
- Q4_K_M_512 drops from 8.84 ms to **< 2 ms**
- Lavapipe perf unchanged (no PCIe → no benefit, but no regression)

**Effort:** ~2500–4000 LOC, 1 milestone cycle.
**Depends on:** M2.3a (already shipped pipeline cache + staging foundation).
**Blocks:** all subsequent perf claims.

### M3.1 — Multi-row tiled matmul + cooperative_matrix dispatch

**Why:** M2.6 Q4_K_M is single-row matvec (1 output per dispatch). Real LLM inference needs N-row matmul with tensor cores. M2.1 added cooperative_matrix codegen but it's compile-only — never dispatched on tensor-core hardware.

**Scope in:**
- Multi-output kernel: `q4km_dequant_matmul(q: ..., x: ..., y: buffer[f32], n_rows: u32, n_cols: u32, n_blocks_per_row: u32)`
- Tile-parallel: each workgroup computes a 16×16 (or 32×32) tile of outputs
- Cooperative_matrix integration for the f16 inner dot product (after dequant)
- Per-vendor `@strategy` holes for tile dimensions (NVIDIA prefers 16×16×16, AMD WMMA is 16×16×16, Intel Xe-Cores want 8×16×16)
- Dispatch on NVIDIA RTX PRO 6000 (Blackwell tensor cores) — requires `VK_KHR_cooperative_matrix` device feature

**Acceptance:**
- 4096×4096 f32 matmul on NVIDIA: **≥ 50% of cuBLAS** (cuBLAS hits ~15 TFLOPS f32, Q4_K_M dequant + matmul should clear 7.5 TFLOPS effective)
- Bit-exact vs CPU reference for 256×256 fixture
- Same `.axc` compiles + runs on Lavapipe (graceful skip via cooperative_matrix preflight)

**Effort:** ~3000–5000 LOC.
**Depends on:** M3.0 (bandwidth), M2.1 cooperative matrix infra.
**Blocks:** M3.2.

### M3.2 — FlashAttention-2 kernel

**Why:** the second-most-cited LLM kernel after matmul. Tests `@strategy` holes on a kernel with non-trivial control flow (online softmax, accumulating denominator).

**Scope in:**
- FA2-shape kernel: `flash_attention_v2(Q: buffer[f16], K: buffer[f16], V: buffer[f16], O: buffer[f16], n_heads: u32, seq_len: u32, head_dim: u32)`
- Block-level streaming softmax with scaling re-correction
- Workgroup-shared K/V tile in shared memory (requires `shared[T, N]` syntax — currently DESIGN.md §3.1 lists it but not implemented; M3.2 adds the language feature)
- `@strategy { tile_q: ?[64, 128], tile_k: ?[64, 128], stages: ?[1, 2, 3] }` on shared-memory tile sizes

**Acceptance:**
- ≥ **80% of FlashAttention-3 cuBLAS+FA3 wrapper** on NVIDIA H100/Blackwell at seq_len=8192, head_dim=128
- ≥ 90% of rocBLAS-flash equivalent on MI300X (if access available)
- Bit-exact vs reference within 1e-3 fp tolerance

**Effort:** ~4000–6000 LOC. Largest single milestone. Includes new language feature (`shared[T, N]`).
**Depends on:** M3.0, M3.1.
**Blocks:** KernelBench submission (M3.3).

### M3.3 — KernelBench-Vulkan public submission

**Why:** Stanford's KernelBench (ICML 2025) is the standard eval for LLM-generated GPU kernels. Nobody has submitted SPIR-V yet. First submission is automatic differentiation in the LLM-for-GPU literature.

**Scope in:**
- Wrapper crate consuming KernelBench's PyTorch task definitions
- Auto-translate KernelBench's "PyTorch reference + CUDA-like kernel slot" into AXIOM-Compute `.axc` template + `@strategy` holes
- Run grid_search on a representative subset (Level 1 = ops, Level 2 = chains, Level 3 = real models)
- Publish results in a public repo / issue thread / arXiv preprint

**Acceptance:**
- Submit ≥ 50 / 250 KernelBench tasks with bit-exact output and measured Vulkan timing
- Beat PyTorch baseline on ≥ 30% (matches DeepSeek-R1 single-shot baseline)
- Beat PyTorch + 10-turn iterative LLM refinement on ≥ 5%

**Effort:** ~2000 LOC + paper writing.
**Depends on:** M3.0, M3.1, M3.2.

### M3.4 — llama.cpp Vulkan head-to-head A/B (the thesis-closing milestone)

**Why:** DESIGN.md §5 kill criterion: *"M2 slip: cannot match llama.cpp Vulkan Q4_K_M within 15% on any vendor."* Currently we have a Q4_K_M kernel that's bit-exact and dispatches, but no comparison.

**Scope in:**
- Side-by-side bench: same Q4_K_M weights, same input vector, same output, run via llama.cpp's Vulkan backend AND via AXIOM-Compute's kernel
- Identical machine, identical Vulkan ICD, fence-synchronized timing
- Multi-row matmul shape (M3.1 prerequisite) — single-row matvec doesn't reflect real inference
- Run across vendor matrix: NVIDIA RTX PRO 6000, AMD RDNA3 (target: Radeon 7900 XTX or MI300X if accessible), Intel Arc

**Acceptance:**
- Match llama.cpp Vulkan Q4_K_M within **5%** on NVIDIA (DESIGN.md §5 says 15%; aim higher)
- **Beat by ≥ 25%** on AMD APU or Intel Arc (DESIGN.md §5 — these are llama.cpp's weak spots per ggml-org/llama.cpp issues #16230, #21517, ollama #15601)
- Single annotated source produces all three variants via `@target` paths

**Effort:** ~1500 LOC + cross-vendor hardware access.
**Depends on:** M3.0, M3.1.
**Blocks:** practical adoption story.

---

## Phase M4 — PyTorch + adoption

Goal: real users.

### M4.1 — PyTorch `torch.library` custom-op frontend

**Scope in:**
- `axiom-compute-py` Python package with PyO3 bindings
- `axc.compile_kernel(source) -> torch.library.Library` registers the kernel as a PyTorch op
- `axc.optimize_kernel(source, sample_inputs) -> CompiledKernel` runs grid_search and caches the winner
- Integrate with `torch.compile` as a custom backend (lower PyTorch graph ops to AXIOM-Compute kernels for the subset we support: matmul, layer norm, softmax, residuals)
- ABI: pass `torch.Tensor` as raw GPU pointer + descriptor (zero-copy when possible)

**Acceptance:**
- `pip install axiom-compute` works on Linux x86_64 with NVIDIA + Vulkan loader
- Drop-in custom-op replacement for `torch.matmul` on a 4096×4096 f32 case, within 50% of native PyTorch (cuBLAS-backed) speed
- 10-line PyTorch user code can call AXIOM kernel

**Effort:** ~3000–5000 LOC.
**Depends on:** M3.0–M3.2.

### M4.2 — Upstream PR to llama.cpp / candle / MLX

**Scope in:**
- Pick the framework with the lowest integration friction (probably `candle` — Rust, Vulkan-curious)
- Port one hot kernel (Q4_K_M matmul) from candle's existing GLSL to AXIOM-Compute output
- Open a PR with measured A/B numbers
- Land it OR get a useful "won't merge because X" reason

**Acceptance:**
- Either: PR merged into a public framework
- Or: a clear, public technical reason from maintainers why the AXIOM-Compute approach is rejected (still valuable signal)

**Effort:** ~1500 LOC + maintainer relationship building.
**Depends on:** M3.4.

---

## Engineering debt (cuts across phases)

### EB.1 — Cross-vendor real GPU CI

**Scope:** self-hosted GitHub Actions runners for AMD (RDNA3 + Vulkan ICD) + Intel (Arc + ANV ICD). Currently CI is Lavapipe-only.

**Why it matters:** the portability thesis is unproven without measurements on AMD and Intel. Dev machine has only NVIDIA.

**Acceptance:** all 6 GPU tests + bench regression gate run on AMD and Intel in CI on every PR.

**Effort:** infra work, ~1 week to set up runners + ~1000 LOC of GitHub Actions YAML.

### EB.2 — Per-machine-keyed baseline format

**Bug:** `baselines.json` has only one machine field. Re-blessing on a different machine overwrites prior entries. We've seen this — Lavapipe runs were overwriting NVIDIA runs and vice versa.

**Fix:** `baselines.json` becomes `{ "machines": { "<machine_id>": { ...current schema... } } }` keyed by `(vulkan_device, cpu_model)` hash. Bench gate looks up entry for current machine; if absent, prints "no baseline; run AXC_BLESS_BASELINES=1".

**Effort:** ~300 LOC.

### EB.3 — `axc bench` CLI subcommand

**Currently:** users run `cargo bench -p axc-driver` directly. Should have an `axc bench [--filter NAME] [--bless]` wrapper that's easier to discover.

**Effort:** ~200 LOC.

### EB.4 — Baseline drift fix in current `baselines.json`

`baselines.json` was reblessed multiple times during the autonomous run; some entries reflect Lavapipe, some NVIDIA, some interleaved. Re-bless on a single machine after EB.2 lands.

---

## Feature gaps (mentioned in DESIGN.md / CLAUDE.md but never built)

### FG.1 — `axc rewrite` source-to-source LLM rewriter

`@strategy` holes are *parameter* tuning. `axc rewrite` is *structural* — the LLM rewrites the kernel body itself (different loop nest, different memory access pattern, etc.) and the compiler verifies via `@equiv_fp_tol`.

**Scope:** new MCP tool `propose_rewrite(source, hint) -> rewritten_source`; LLM-side prompting infrastructure; correctness verification harness that compares rewritten kernel output to original.

**Effort:** ~2000 LOC.

### FG.2 — `@transfer { ... }` blocks for inter-agent handoff

DESIGN.md §3.2 lists `@transfer` as an annotation. Idea: structured handoff between agents with confidence scores ("agent A optimized for tile size; agent B should explore @async_copy"). Currently parsed but no semantics.

**Effort:** ~800 LOC + protocol design.

### FG.3 — `@optimization_log {}` block

Per-kernel embedded history of prior optimization runs. Currently stored externally in `.pipeline/history/<hash>.jsonl` (M2.4). The block embeds it into source so the kernel is self-describing.

**Effort:** ~500 LOC.

### FG.4 — `@precondition` / `@postcondition` runtime checks

Currently parsed and HIR-validated, but never lowered to debug-mode runtime asserts. Spec promises `axc compile --debug` enables them.

**Effort:** ~600 LOC.

### FG.5 — Q5_K_M / Q6_K quantization variants

Straightforward extension of M2.6 Q4_K_M pattern. Q5_K_M adds a 1-bit overlay to Q4_K_M. Q6_K is 6-bit weights with similar superblock structure.

**Effort:** ~500 LOC each (mostly CPU reference + tests; codegen is small).

### FG.6 — `shared[T, N]` workgroup-local memory

DESIGN.md §3.1 lists it. Required for FA2 (M3.2). Lex/parse + HIR + SPIR-V `OpTypePointer Workgroup` codegen.

**Effort:** ~1000 LOC. (Folded into M3.2 above.)

### FG.7 — Sized arrays as locals

`array[T, N]` per DESIGN.md §3.1. Currently only buffer types are supported.

**Effort:** ~600 LOC.

### FG.8 — `axc verify` / `axc test --fuzz`

Parent AXIOM CLAUDE.md mentions these for `@strict` modules. AXIOM-Compute inherits the design but never built them. Verify checks annotation completeness; test --fuzz auto-generates inputs from `@precondition` constraints.

**Effort:** ~1500 LOC.

### FG.9 — Multi-kernel modules

Currently one `@kernel` per file. DESIGN.md hints at multi-kernel modules with cross-kernel `@strategy` (shared holes). Useful for prefix-sum style kernels with multiple stages.

**Effort:** ~1200 LOC.

---

## Polish + bug backlog

| Item | Severity | Effort |
|---|---|---|
| `BENCHMARKS.md` says "(M2.2)" in heading; should reflect current state | low | 5 min |
| `M2.6 dispatch_q4km_128` baseline label is `dispatch_q4km_128` while spec says `dispatch_gpu_q4km_128` — naming inconsistency | low | 30 min |
| `axc lex` output format human-only; spec mentions JSON for M1+ | low | 200 LOC |
| Stale doc-comment `Float64 cap = 6` in `body.rs:902` (caught by M1.5 reviewer) | trivial | 1 line |
| AT-103 empty-kernel "bit-exact" test only does determinism guard, not stored golden bytes | medium | 100 LOC |
| `dispatch_gpu_amortized` has only 1 entry (saxpy_1m); should add q4_0/q4_km amortized too | low | 100 LOC |
| `axc-runtime` exposed surface includes some `pub(crate)` items leaking into `pub` | low | review pass |
| No `CHANGELOG.md` | low | 30 min |

---

## Suggested ordering

Critical path to thesis closure:

```
M3.0 (bandwidth)  →  M3.1 (multi-row + coopmat)  →  M3.4 (llama.cpp A/B)
                                                      ↓
                                                   thesis proven
                                                      ↓
                                  M3.2 (FA2)  →  M3.3 (KernelBench)
                                                      ↓
                                                 publishable
                                                      ↓
                                              M4.1 (PyTorch)  →  M4.2 (upstream PR)
                                                                       ↓
                                                                  real users
```

Engineering debt + feature gaps interleave between milestones based on what unblocks next.

### Single highest-leverage next step

**M3.0 (dispatch bandwidth rework).**

Without it: every M3.x perf number is dominated by staging copy. With it: 1M-element saxpy goes from 23 ms → ~1 ms on NVIDIA, bringing AXIOM-Compute into the regime where comparison against handwritten kernels is meaningful. ~3000 LOC, ~1 milestone cycle through the 7-agent pipeline.

### Lowest-risk acceleration

**EB.2 (per-machine baselines) + EB.1 (cross-vendor CI)** can run in parallel with M3.0 since they touch separate codepaths (bench infrastructure + GitHub Actions vs runtime/dispatch). Frees future milestones from baseline-drift confusion and gives early signal on AMD/Intel portability before M3.4.

---

## Estimated effort to "thesis proven" state

To reach DESIGN.md §5 kill-criteria pass (M3.0 + M3.1 + M3.4):
- ~7000–10000 LOC
- ~3 milestone cycles (each milestone = architect → dual review → coder → QA → dual code review → merge, averaging ~6 hours of agent time per milestone in the autonomous regime)
- Cross-vendor hardware access (AMD RDNA3 + Intel Arc) for M3.4

To reach M4.2 (upstream adoption):
- additional ~5000–8000 LOC
- additional ~2 milestone cycles
- maintainer relationship building (real-world latency)

To finish the full DESIGN.md vision (everything above):
- ~25000–35000 LOC over ~10 milestone cycles
- with feature gaps + cross-vendor CI added
- a year of engineering at the current cadence

---

## How to contribute

- Pick a milestone or engineering-debt item from above
- Open an issue at https://github.com/rudybear/axiom-compute/issues describing your approach
- The 7-agent pipeline templates live at `.pipeline/templates/` — Architect first, dual review, then Coder
- Every PR runs in-process spirv-tools validation + clippy `--all-targets` `-D warnings` + the bench regression gate
