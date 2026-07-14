# Changelog

All notable changes to AXIOM-Compute, by milestone. This is a **milestone-level**
backfill (not per-commit) — one entry per merged milestone family with a one-line
summary and its AT (acceptance-test) range where applicable. Backfilled from
`ROADMAP.md`'s status snapshot and `git log --oneline` at M3.15. See `ROADMAP.md`
for the living, detailed record and `DESIGN.md` for the design rationale.

---

## [Unreleased] — M3.15 — Engineering-debt + polish bundle

`axc bench` CLI subcommand (EB.3, pure `build_bench_command` seam); robust
lexer-token-anchored `strip_strategy_annotation_block` fix (closes the M3.14
`source.find("@strategy")` comment/string-literal decoy footgun); EB.4 baseline
re-bless (NVIDIA refresh + new Lavapipe `llvmpipe_*` key, two-machine-shape
round-trip test); AT-103 golden full-bytes upgrade for the empty-kernel SPIR-V
regression guard; this `CHANGELOG.md` backfill; doc-accuracy sweep (BENCHMARKS.md
freshness note, CLAUDE.md Lavapipe-ICD-name environment-dependence note,
grep-fence regression test). No kernel, codegen, or perf-campaign work; FROZEN
1e-3 and production kernels untouched. AT-2826..2840.

## M0 → M2.6 — Bootstrap → Q4_K_M bit-exact

The founding arc: 7-crate Cargo workspace skeleton, lexer/parser/HIR/codegen
pipeline, SPIR-V 1.3 emission + `spirv-val`, real Vulkan compute dispatch (`ash`),
cooperative-matrix codegen, the `@strategy { ?holes }` grid-search autotuner, an
MCP JSON-RPC server for LLM-agent integration, and the first LLM kernels: Q4_0
quantized matmul (M2.5) then the Q4_K_M superblock matmul (M2.6) — the
llama.cpp-beachhead kernel proven bit-exact on real NVIDIA hardware.

## M3.0 — Dispatch bandwidth rework

Pinned-memory + concurrent-transfer dispatch path: 7.5× `saxpy_1m`, 39× small
dispatch. The `<1ms` gate was re-scoped to a GPU-resident metric after
discovering host-round-trip benches mostly measure PCIe, not kernel quality.

## M3.1 / M3.1.5 — Cooperative-matrix dispatch + GPU-resident benchmark

First `cooperative_matrix` dispatch on Blackwell tensor cores (bit-exact). The
GPU-resident benchmark (upload-once / dispatch-N / kernel-only timestamp timing)
landed in M3.1.5, measuring saxpy at 2.0 µs kernel-only.

## M3.2 / M3.2b / M3.2c — shared memory, FlashAttention-2, real exp()

`shared[T,N]` workgroup-local memory as a first-class language feature
(GPU-validated). FlashAttention-2 streaming online-softmax landed in M3.2b
(scalar, Taylor-series exp), then M3.2c added a real `exp()` builtin
(GLSL.std.450 — the first SPIR-V extended instruction), coopmat-accelerated QKᵀ
(M3.2c-perf), and coopmat P·V (M3.2c-PV) — both matmuls of attention on tensor
cores.

## M3.3 / M3.3b / M3.3c / M3.3d — OpPhi + register-blocked matmul

`OpPhi` loop-carried SSA (GPU-proven) unlocked a working f32 matmul and
attention. M3.3b delivered a bit-exact multi-tile coopmat matmul (5 TFLOPS).
M3.3c added 2×2 register blocking: 6.2× uplift to 31 TFLOPS (24.96% of cuBLAS
datasheet). M3.3d (`local_invocation_id()` + multi-subgroup) was an honest
negative — no uplift over single-subgroup.

## M3.4 → M3.6 — The llama.cpp Q4_K_M A/B campaign (thesis-closing arc, part 1)

M3.4 established the honest kill-criterion baseline: AXIOM's single-row matvec
was ~87,000× behind llama.cpp's Q4_K GEMV on NVIDIA. M3.5 fused the Q4_K_M
dequant into the register-blocked coopmat matmul, closing the gap to ~9× — but
the f16 accumulator was numerically wrong at production K. M3.5b switched to an
f32 accumulator: numerically valid, gap 9.3×. M3.6 added scale-caching
(`shared[f32,256]` per-superblock scale cache, filled once, reused across 16
k-blocks): gap closed to **2.39×**, bit-identical to M3.5b — the project's
long-standing performance leader.

## M3.7 / M3.8 — Double-buffering and larger register tiles (honest negatives)

M3.7 (software-pipelined double-buffered staging) regressed ~3% — the kernel
was occupancy-bound, not latency-bound. M3.8 (4×2 / 4×4 register tiles)
regressed further (0.61–0.66×) — more registers per thread cut resident warps.
Together these established M3.6's 2×2 tiling as the occupancy/compute sweet
spot.

## M3.9 / M3.10 / M3.10b — Warptile, bank padding, vec-load (the "fix the
compiler" campaign)

M3.9 added a multi-subgroup warptile (4 subgroups cooperating on a 64×64 tile,
bit-identical, pure source) — honest negative, 0.955× A/B. M3.10a (shared-memory
bank padding) also regressed at every pad width, ruling out bank conflicts as
the bottleneck. M3.10b (a DIAGNOSE-FIRST feasibility study, no kernel built)
corrected the gap decomposition: the 2.39× gap is **1.35× matmul-core ×
1.77× dequant front-end**, with the front-end dominant and ALU/issue-bound, not
load-bound — ruling out vectorized loads as a lever.

## M3.11 / M3.12 / M3.13 — Pinpointing and closing the dequant front-end (thesis-
closing arc, part 2)

M3.11 hand-strength-reduced the per-element dequant index-decode (affine
`/256 /64 /32` chain → loop-carried counters) — ALU dropped but TFLOPS stayed
flat (1.048×): the integer decode is latency-hidden under the HMMA pipeline.
M3.12's ablation diagnostic pinpointed the true cause: the 1.77× tax is the
per-element dequant **work**, register-pressure/occupancy-driven, not the
scale-cache structure. M3.13 closed the campaign: PRONG A (live-range
tightening) was a double honest-negative (register pressure isn't
source-addressable in AXIOM's current codegen); PRONG B (fusing dequant directly
into a coopmat fragment) was ruled infeasible — `VK_KHR_cooperative_matrix`
exposes only 4 opaque builtins with no per-element fragment write. **Verdict:**
AXIOM's portable coopmat Q4_K_M kernel sits at the VK_KHR achievable ceiling;
the residual 2.39× is the portable-coopmat tax versus llama's vendor-tuned
in-register fusion. M3.6 remains the production leader.

## M3.14 — Q6_K + Q5_K_M quantization variants (FG.5)

Coverage milestone (no TFLOPS gate — M3.13 closed the perf campaign): real
"Q4_K_M" GGUFs store `output`/`attn_v` tensors as Q6_K, so this closes an actual
beachhead hole. Zero compiler changes — CPU oracles + portable kernels for both
formats, bit-exact on real NVIDIA (AT-2811..2823).

## M4.1 — PyTorch `torch.library` custom-op frontend

Phases 1–4, MERGED. CUDA↔Vulkan zero-copy proven on NVIDIA (external-memory-fd +
timeline external semaphore, no host copy). M3.6's Q4_K_M matmul and the
FlashAttention kernel registered as `torch.ops.axiom.*` custom ops, composing
under `torch.compile(fullgraph=True)` with zero graph breaks. `crates/axc-py`
(PyO3) + `py/axiom_compute` (pip/maturin).

## M4.3 — `torch.compile(backend="axiom")`

Auto-lowers matching `scaled_dot_product_attention` calls to the AXIOM attention
op under `torch.compile`.

## EB.2 — Per-machine-keyed bench-regression baselines

`.pipeline/benchmarks/baselines.json` moved to a `schema_version: 2`,
per-machine-keyed shape (keyed by sanitized Vulkan `deviceName`), making the
bench-regression harness safe to re-bless independently across NVIDIA / AMD /
Intel / Lavapipe machines without clobbering other machines' entries.

---

_Not itemized here (see `ROADMAP.md` for status): EB.1 (cross-vendor real GPU
CI, hardware-gated / open), M4.2 (upstream PR to llama.cpp, prepared-and-held)._
