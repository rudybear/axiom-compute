# AXIOM-Compute -- Master Task List

Living plan. Reviewed at each milestone gate. Last update: 2026-04-18 (M0 pre-architect).

## Phase M0 -- Bootstrap (week 0-2)

- M0.1 Cargo workspace + 7 empty crates
- M0.2 Lexer port from `axiom` (crate: `axc-lexer`) + GPU keyword additions
- M0.3 Minimal parser for `@kernel` + `@workgroup` only (crate: `axc-parser`)
- M0.4 Minimal HIR: validate `@kernel` requires `@workgroup` (crate: `axc-hir`)
- M0.5 Minimal codegen: rspirv direct emission of empty kernel (crate: `axc-codegen`)
- M0.6 Driver: `axc compile` CLI (crate: `axc-driver`)
- M0.7 GitHub Actions CI with spirv-val
- M0.8 `examples/empty_kernel.axc` compiles and validates

**Exit gate:** see `.pipeline/milestones/M0-bootstrap.json`.

## Phase M1 -- MVP backend (month 1-3)

- M1.1 Scalar ops, buffers, shared memory, workgroup dispatch
- M1.2 Subgroup ops (ballot, shuffle, reduce) via `SPV_KHR_shader_subgroup_*`
- M1.3 `axc run` dispatcher on Vulkan via `ash`
- M1.4 Saxpy + vector add + parallel reduction correctness tests on NVIDIA + AMD + Intel + Lavapipe

## Phase M2 -- Cooperative matrix + llama.cpp Q4_K_M (month 3-9)

- M2.1 `@cooperative_matrix` → `SPV_KHR_cooperative_matrix` lowering
- M2.2 `@strategy` hole infrastructure
- M2.3 LLM autotuner (port from AXIOM optimize crate)
- M2.4 MCP server port
- M2.5 Q4_K_M dequant + matmul reference kernel
- M2.6 Bench harness vs current llama.cpp Vulkan baseline
- M2.7 `@equiv_fp_tol` correctness oracle

**Exit gate:** match llama.cpp Vulkan Q4_K_M within 5% on RTX 4090; beat by ≥25% on AMD APU / Intel Arc.

## Phase M3 -- FlashAttention + KernelBench (month 9-15)

- M3.1 FlashAttention-2 shape kernel
- M3.2 KernelBench-Vulkan submission harness
- M3.3 First public SPIR-V KernelBench leaderboard entry

**Exit gate:** ≥80% cuBLAS+FA3 on H100 via Vulkan; ≥90% rocBLAS on MI300X.

## Phase M4 -- PyTorch + adoption (month 15-24)

- M4.1 `torch.library` custom-op integration
- M4.2 Upstream PR to llama.cpp or candle or MLX

**Exit gate:** external adoption by ≥1 inference framework.

## Kill criteria

Pre-registered; see `DESIGN.md §5`.
