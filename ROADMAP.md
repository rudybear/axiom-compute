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
| Bandwidth optimization (pinned memory, concurrent transfer) | ✅ M3.0 (saxpy_1m 7.5×, saxpy_1024 39×; <1ms gate re-scoped to GPU-resident metric) |
| Multi-row tiled matmul (cooperative_matrix on real workloads) | ✅ M3.1 (first coopmat dispatch on Blackwell, bit-exact; resident-TFLOPS benchmark → M3.2) |
| Cross-vendor real GPU CI (AMD RDNA3+, Intel Arc) | ❌ infra |
| **llama.cpp Vulkan head-to-head A/B** | ❌ thesis-closing milestone |

---

## Phase M3 — Performance + ML kernels (the thesis-closing phase)

Goal: prove the DESIGN.md §5 kill-criteria gates with publishable numbers, not just bit-exact correctness.

### M3.0 — Dispatch bandwidth rework ✅ DONE (gate re-scoped)

**Status (2026-06-01):** Merged. Persistent-mapped HOST_CACHED staging + optional dedicated transfer queue + timeline/binary-semaphore overlap, single-queue fallback byte-identical to M2.3a. Measured on NVIDIA RTX PRO 6000: `dispatch_saxpy_1m` 23 ms → **3.08 ms (7.5×)**, `dispatch_saxpy_1024` 1.22 ms → **31 µs (39×)**, `dispatch_q4km_512` 8.84 ms → **5.76 ms (1.5×)**. All paths byte-exact (AT-1418 four-config oracle on NVIDIA). 747 tests, codegen untouched.

**Gate re-scoped — the original `<1 ms` / `<2 ms` targets were not met and are deferred, for a documented reason.** Profiling showed the residual cost is host-round-trip PCIe transfer + readback, not kernel quality: `saxpy_1m` moves ~12 MB host→device→host every call. A ReBAR zero-copy-readback fix (r2) was attempted and **empirically reverted** — CPU reads from the BAR aperture are write-combined and ~60× *slower* (the real-GPU measurement gate caught it pre-merge; see DESIGN.md §3.1.12 postmortem). The thesis-relevant metric is a **GPU-resident benchmark** (upload once, dispatch N times, measure kernel time), re-scoped to M3.1/M3.4. Two correct-but-unshipped ideas carry forward to M3.1: Lever A (skip readback of `readonly` bindings) and the binary-semaphore-recreation fixup (Dev#1).

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

### M3.1 — Multi-row tiled matmul + cooperative_matrix dispatch ✅ DONE (core proven; perf-methodology carried to M3.2)

**Status (2026-06-01):** Merged. **First-ever cooperative_matrix dispatch on real Blackwell tensor cores** — `matmul_tile.axc` (M2.1's compile-only SPIR-V, now executed) produces a 16×16 C=A·B tile **bit-exact (max_diff=0)** vs CPU reference on the NVIDIA RTX PRO 6000 (AT-1510). Required unlocking three device-feature classes never previously enabled: `VulkanMemoryModel` (the coopmat SPIR-V uses `OpMemoryModel Logical Vulkan`), `16BitStorage` (f16 SSBOs), `8BitStorage`+`shaderInt8/16` (q4km u8 weights) — all enabled **conditionally** (probe-then-enable, fail-closed `DeviceFeatureUnsupported`/`CoopMatUnsupported` skip), so Lavapipe and existing dispatch tests are unregressed. Multi-row `q4km_dequant_matmul` is **bit-exact 256×256** (AT-1520, runs on Lavapipe+NVIDIA); a pre-dequantized coopmat bridge proves coopmat-on-q4km (AT-1521, max_diff=0). Coopmat shape now flows HIR→metadata (schema v1→v2, back-compat). Lever A (skip `readonly` readback, deferred from M3.0 r2) landed cleanly without ReBAR. SPIR-V byte-identical to M2.1; spirv-val clean (AT-1560); a non-symmetric transpose fixture (AT-1512) guards against mis-strided loads. 780 tests.

**M3.1.5 (DONE, 2026-06-01):** completed the **GPU-resident benchmark** (`upload/dispatch/readback_resident` implemented; timestamp-query timing + warmup + CpuFenceWall fallback). Measured kernel-only spans on NVIDIA (all GpuTimestamp): **saxpy 2.0 µs** (vs 3.08 ms host-round-trip), **q4km matvec 131 µs**, **q4km matmul 1.9 µs** — proving the M3.0 dispatches are ~99.9% PCIe transfer, validating the host-round-trip gate re-scope. The dispatch-time preflight is wired via additive `prepare_kernel_checked`. **Honest finding:** `matmul_f32_tiled.axc` had inert `@strategy` holes (removed) — it's a naive un-tiled GEMM (4.5 TFLOPS ≈ 3.6% cuBLAS datasheet estimate, `naive_gemm_harness_validation` bench only).

**Still carried to M3.2 (honestly deferred):** a **competitive tiled/coopmat matmul TFLOPS** (needs `shared[T,N]` workgroup memory, FG.6) — the full **dequant→shared-f16-tile→coopmat fusion**; folding the preflight into raw `prepare_kernel` (opt-in gap remains). The `<2 ms` q4km gate from M3.0 also rides on this.

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

### M3.2 — shared[T,N] workgroup memory ✅ LANDED; competitive matmul + attention → M3.3 (2026-06-03)

**LANDED — the `shared[T,N]` language feature (FG.6):**
- **FG.6 `shared[T,N]` IMPLEMENTED + GPU-VALIDATED**: full lexer→parser→HIR→typecheck→codegen→spirv-val pipeline. OQ1 SET-based missing-barrier analysis (provably zero false positives — verified by the pessimistic reviewer with 5 adversarial patterns). OQ2 `conditional_depth` divergent-barrier hard error. AT-429 inverted. CRITICAL-1/2/3/4 resolved across 3 design-revision cycles. **AT-1606: a shared[f32,256] parallel reduction + `workgroup_barrier()` runs BIT-EXACT on the NVIDIA RTX PRO 6000 (=384.0)** — the feature executes correctly on real hardware, not just compiles.
- **Coopmat-from-shared infrastructure**: `CoopMatLoadSource::{Buffer,Shared}` discriminator + `emit_coopmat_load_shared_inline`/`store_shared_inline` (single-index Workgroup access chain); existing Buffer-source coopmat SPIR-V byte-identical (AT-1613).
- **Metadata schema v3**: `SUPPORTED_SCHEMA_VERSIONS=[1,2,3]` (v1/v2 back-compat), `shared_memory_bytes`, `maxComputeSharedMemorySize` graceful-skip preflight.

**RE-SCOPED to M3.3 (honest — GPU-measured):** the competitive tiled matmul (`matmul_shared_coopmat.axc`, `matmul_shared_f32.axc`) and the tiled attention (`tiled_attention.axc`) **compile + spirv-val clean but compute INCORRECT results (zeros) on real GPU.** Root cause: `emit_for_range` lacks **OpPhi loop-carried SSA**, so a K-loop accumulator can't carry the coopmat/sum value across iterations — without it the matmul is single-tile and the kernels don't accumulate. Their bit-exact GPU tests were converted to compile/spirv-val-only (no zero-computing test ships as passing); the misleading TFLOPS bench was removed; the examples carry `WIP (M3.3)` headers. **M3.3 = OpPhi loop-carried SSA in `emit_for_range` → real multi-tile matmul (competitive TFLOPS) + working tiled attention.** FlashAttention-2 streaming softmax (C2) remains M3.2b after that.

**Acceptance (what passed):** 834 tests green; clippy -D warnings clean; AT-1606 GPU bit-exact on NVIDIA; AT-1613/1614 coopmat byte-identity + shared spirv-val; metadata v1/v2/v3 back-compat. Both code reviewers APPROVE.

**FG.6 status:** `shared[T,N]` is **IMPLEMENTED + GPU-validated** (see DESIGN.md §3.1.14).

**Effort:** ~4800 LOC. The new language feature landed; the kernels exploiting it need OpPhi (M3.3).
**Depends on:** M3.0, M3.1. **Blocks:** M3.3 (OpPhi + competitive matmul + attention), then M3.2b (C2 FA2).

### M3.3 — OpPhi loop-carried SSA (GPU-proven) ✅ LANDED; full coopmat matmul → follow-up (2026-06-03)

**LANDED + GPU-validated on NVIDIA RTX PRO 6000:**
- **PART A — OpPhi loop-carried SSA in `emit_for_range`**: emits an OpPhi at the loop header for loop-carried coopmat (SSA) accumulators (scalars stay on Function storage, unchanged). **AT-1707 PROVES it numerically**: a `acc = coopmat_mul_add(A,B,acc)` K-loop is **bit-exact = K·(A·B)** on NVIDIA — resolving the M3.2 blocker (coopmat accumulators reset to zero each iteration). AT-1701 spirv-val + phi-well-formed; AT-1700 confirms scalars still emit 0 phis. ISSUE-1 (both Assign arms route CoopMatrix targets through `check_coopmat_init_expr`), ISSUE-2 (Assign SSA-rebind branch), break/continue-over-coopmat hard error.
- **PART C — working tiled attention (AT-1630 bit-exact within 1e-3 on NVIDIA)** + **f32 tiled matmul (AT-1621 bit-exact)**: these were ZEROS in M3.2 due to *separate scalar-path kernel-logic bugs* (dispatch geometry / index math — NOT OpPhi). Debugged on the already-working Function-storage path. tiled_attention dispatches (seq_len,1,1); Taylor exp with matching CPU ref.

**DEFERRED to a follow-up (honest — GPU-measured):** the **full competitive coopmat matmul** (`matmul_shared_coopmat.axc`). The OpPhi K-accumulation works (AT-1707), but the kernel computes only a single 16×16 output tile, so a 16×**24** output (the test fixture) is wrong for N>16 — it needs a **multi-N-tile output loop** (and register/multi-warp blocking for real throughput). AT-1620/1622 are therefore **compile + spirv-val only (WIP stubs for the bit-exact GPU assertion)** — no wrong-computing test ships as passing. The competitive-TFLOPS bench (`resident_matmul_competitive.rs`) was **removed**: a TFLOPS number from a wrong-computing kernel would be misleading. **No TFLOPS is reported until the matmul is correct.** This is kernel-tiling work, not a compiler gap.

**Honest finding:** only the coopmat accumulator was OpPhi-blocked; the f32-matmul + attention zeros were independent kernel-logic bugs. 846 tests; both code reviews + QA confirm OpPhi correctness + AT-1707 load-bearing + the deferral honesty.

**Depends on:** M3.2. **Blocks:** full competitive matmul + M3.4 llama.cpp A/B (NVIDIA half needs the working tensor-core matmul).

### M3.3b — multi-tile coopmat matmul: bit-exact full matmul + honest effective-TFLOPS (2026-06-03)

**Root cause (corrected from M3.3 deferral):** M3.3 misdiagnosed the multi-tile gap as "needs a multi-N-tile output loop." The actual causes were: (a) test dispatched (1,1,1) against N=24 (only tile (0,0) ran); (b) kernel recovered `tile_col = gid(0)` directly — the GLOBAL invocation id, not the output-tile index. The idiomatic GPU tiling: ONE workgroup == ONE output tile; dispatch a GRID of workgroups (the grid IS the output-tile loop). No per-kernel output loop needed.

**Kernel fix (one line, ASYMMETRIC):**
- `tile_col = gid(0) / 32` (divide by local_size.x=32; all 32 lanes of a workgroup collapse to one tile_col).
- `tile_row = gid(1)` — UNCHANGED (local_size.y=1 so gid(1)==workgroup_id.y; NO division).
- Dispatch: (N/16, M/16, 1) workgroups.
- NO codegen change. OpUDiv already emitted; all builtins and coopmat paths GPU-proven in M3.2/M3.3.

**AT-1620/1622 UN-STUBBED (bit-exact full matmul on NVIDIA):**
- Non-symmetric multiple-of-16 fixture: M=32, N=48, K=32 (3x2 workgroup grid, 2 K-blocks).
- Integer-valued f16 (A∈{1..4}, B∈{1..3}; per-element sum ≤ 384, f16-exact) → max_diff == 0.0.
- tile_k=16 (2 K-blocks, accumulation load-bearing) AND tile_k=32 (1 K-block) both bit-exact.
- Lavapipe: typed-skip (CoopMatUnsupported); matmul_shared_f32.axc (AT-1621) unaffected.

**AT-1710 honest effective-TFLOPS (resident_matmul_competitive.rs re-added):**
- Large matmul M=N=K=256 (default), full 16x16 tile grid, upload-once/dispatch-N (MIN-of-10, GpuTimestamp).
- Reports effective_tflops = 2·M·N·K / kernel_ns + % of 125-TFLOPS f32 datasheet ESTIMATE.
- NO ratio asserted (only tflops>0 && finite). With single 32-lane subgroup per 16x16 tile, throughput is MODEST — that is the honest deliverable.
- "competitive" label omitted unless measured % >= 25%. CpuFenceWall path omits % + carries scheduling-inclusive qualifier.
- 2D grid pre-check: both max_compute_work_group_count()[0] AND [1] checked; shrink M and N together if needed.

**Out of scope (follow-up M3.3c):** partial edge tiles (M or N not a multiple of 16); K not a multiple of tile_k.

**Depends on:** M3.3. **Blocks:** M3.4 llama.cpp A/B (tensor-core matmul now correctly computes full outputs).

### M3.2b — FlashAttention-2 streaming softmax (C2, deferred from M3.2)

**Why:** FA2's defining contribution — block-streaming ONLINE softmax (running max m_i, running denominator l_i, output rescale O_i ← O_i * exp(m_old-m_new) + P*V) — AVOIDS materializing the SxS score block. Deferred from M3.2 because its online-rescale arithmetic is a separate high-risk bit-exactness surface verified against C1 (`tiled_attention`) as the baseline. C2 earns the `flash_attention_v2` kernel name.

**Acceptance:** ≥ 80% of FlashAttention-3 cuBLAS+FA3 wrapper. @equiv_fp_tol(1e-3) vs C1 baseline. Streaming HIR path fully wired through coopmat_load-from-shared dispatch.

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

### FG.6 — `shared[T, N]` workgroup-local memory ✅ IMPLEMENTED (M3.2, 2026-06-02)

IMPLEMENTED. Full pipeline: lexer (token already existed) → parser (TypeRef::Shared, Stmt::SharedDecl) → HIR (SharedId, SharedTy, SharedDecl, KernelBodyTyped.shared, SharedRead/SharedWrite HIR nodes, OQ1 SET-based missing-barrier analysis, OQ2 conditional_depth divergent-barrier) → codegen (SharedBindings, emit_shared_globals with Float16/Int8/etc. caps, single-index OpAccessChain, Workgroup OpVariable, SPIR-V 1.3 interface list exclusion) → runtime (metadata v3, SharedMemoryExceedsDeviceLimit preflight, maxComputeSharedMemorySize cached). See DESIGN.md §3.1.14. Tests AT-1600..AT-1636 passing.

**Effort:** ~5800 LOC across 19+ files. Landed as part of M3.2.

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
