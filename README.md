# AXIOM-Compute

**AI-first compute language for SPIR-V.** Sister project to [AXIOM](https://github.com/rudybear/axiom) (CPU via LLVM IR), targeting GPU/GPGPU via SPIR-V → Vulkan compute / OpenCL / WebGPU.

---

## Thesis

LLMs iteratively optimize CUDA kernels today at the source-text level (Sakana AI CUDA Engineer, Kevin, CUDA-L1, EvoEngineer, STARK). Every published system works on raw CUDA where optimization intent is implicit, which is why they reward-hack and produce correctness regressions.

AXIOM-Compute makes intent first-class. `@strategy { workgroup_x: ?[32, 64, 128, 256] }` declares holes the compiler enumerates and a grid-search (or LLM agent via MCP) fills. `@equiv_fp_tol(1e-3)` is machine-checked. The result is portable SPIR-V that downstream vendor drivers finish optimizing — without CUDA lock-in.

## Current status (2026-06-10)

- **~34 milestones merged on `main`**: M0 → M3.2c-PV / M4.1 Phase 4
- **941 workspace tests passing**, clippy `--all-targets` clean, zero SPIR-V validation errors. Every milestone went through the 7-agent adversarial pipeline (Architect → dual design review → Coder → QA → dual code review) with real-GPU verification before merge.
- **Two flagship LLM kernels, from one annotated `.axc` source each → portable SPIR-V, measured on NVIDIA RTX PRO 6000 Blackwell:**
  - **Q4_K_M matmul (the llama.cpp beachhead) — 42.86 TFLOPS, 2.39× behind hand-tuned llama.cpp Vulkan, numerically valid + bit-identical.** The campaign: M3.4 honest A/B (un-optimized matvec = 87,000× behind) → M3.5 fused dequant→coopmat (9×, but *fast-but-wrong*: f16 accumulator invalid at inference K) → **M3.5b f32 accumulator (numerically VALID, combined condition-aware metric ≤ 1e-3 at K=14336)** → **M3.6 dequant scale-caching (the leader): gap collapses 9.3× → 2.39×, bit-identical, pure source.** M3.7 (double-buffering) + M3.8 (larger register tiles) are rigorous **honest-negatives** establishing the M3.6 2×2 kernel as the occupancy/compute sweet spot — both classic GEMM levers regress because the kernel is occupancy-bound.
  - **FlashAttention-2 — streaming online-softmax, fully coopmat-accelerated (QKᵀ *and* P·V on tensor cores), real-range correct.** M3.2b (scalar streaming softmax, no S materialization) → M3.2c-exp (a real `exp()` builtin — the **first GLSL.std.450 extended instruction** in the codegen — making real-range attention correct) → M3.2c-perf (coopmat QKᵀ + 16-row query tile) → **M3.2c-PV (coopmat P·V too)**. Correct within frozen 1e-3 vs a true-exp softmax oracle; the acc-in-shared design keeps the per-row rescale scalar (avoiding a coopmat-diagonal-scale codegen feature).
- **PyTorch frontend with CUDA↔Vulkan zero-copy interop (M4.1, the M4 adoption phase):** a `pip`-installable PyO3 package where a torch **CUDA** tensor feeds an AXIOM **Vulkan** kernel on the **same physical GPU with zero host copies** — via `VK_KHR_external_memory_fd` export → `cudaImportExternalMemory`, a timeline external semaphore handshake, fail-closed device-UUID matching. **Both flagship kernels are registered `torch.library` custom-ops** (`torch.ops.axiom.q4km_matmul`, `torch.ops.axiom.flash_attention`) that **compose with `torch.compile(fullgraph=True)`** (0 graph breaks). Honest: the win is no-host-copy + real torch integration, not beating cuBLAS/SDPA.
- **The thesis, demonstrated:** one annotated source → portable coopmat SPIR-V → correct, competitive-ish kernels callable from the framework the ecosystem actually uses, with every claim independently verified and every honest-negative reported as such (the frozen `@equiv_fp_tol` was never loosened across the campaign).

| Kernel | Status |
|---|---|
| saxpy / vector_add | ✅ bit-exact on NVIDIA (+ zero-copy from PyTorch) |
| reduction / workgroup barrier / subgroup reduce | ✅ bit-exact / validate |
| Q4_0 / Q4_K_M dequant | ✅ bit-exact on NVIDIA + Lavapipe — the llama.cpp beachhead |
| **Q4_K_M matmul (fused, f32-acc, scale-cached)** | ✅ **42.86 TFLOPS, 2.39× behind llama.cpp, numerically valid (M3.6)** |
| cooperative_matrix matmul (register-blocked) | ✅ bit-exact on NVIDIA tensor cores |
| **FlashAttention-2 (coopmat QKᵀ + P·V, real-range exp)** | ✅ **correct within 1e-3 on NVIDIA (M3.2c-PV)** |
| **Both kernels as `torch.library` ops** | ✅ **zero-copy, compose with `torch.compile` (M4.1)** |

## Architecture

```
.axc source (annotated for LLM consumption)       <- LLM agents author here
       │
       ▼
Lexer → Parser → HIR (@strategy validation)
       │
       ▼
axc-optimize: enumerate @strategy holes (Cartesian product of candidates)
       │
       ▼
SPIR-V codegen (rspirv) with lazy capability/extension emission
       │
       ▼
spirv_tools::val (in-process, MANDATORY)
       │
       ▼
Vulkan runtime (ash) + KernelHandle cache + DEVICE_LOCAL buffers + staging
       │
       ▼
Grid search / LLM via MCP → winner picked by median_ns
```

## Crate layout

| Crate | Purpose |
|---|---|
| `axc-lexer` | Tokenizer |
| `axc-parser` | Recursive descent + Pratt expressions |
| `axc-hir` | Type check, annotation validation, strategy holes |
| `axc-codegen` | SPIR-V emission via rspirv (typed enum API) |
| `axc-optimize` | Strategy hole enumeration + grid search autotuner |
| `axc-runtime` | Vulkan dispatch (ash), pipeline cache, staging buffers |
| `axc-driver` | CLI (`axc compile`, `axc optimize`, `axc mcp`) + MCP server |

## Quick start

```bash
# Build
cargo build --release

# Compile a kernel to SPIR-V
./target/release/axc compile examples/saxpy.axc -o /tmp/saxpy.spv

# Run grid search (requires Vulkan-capable GPU)
AXC_ENABLE_GPU_TESTS=1 ./target/release/axc optimize examples/saxpy.axc \
    --output /tmp/saxpy-winner.spv

# Reproduce a specific variant
./target/release/axc compile examples/saxpy.axc \
    --strategy-value workgroup_x=64 -o /tmp/saxpy-64.spv

# Benchmark
AXC_ENABLE_GPU_BENCHES=1 cargo bench -p axc-driver
```

## Running tests

```bash
# All non-GPU tests (~710 unit + integration)
cargo test --workspace

# GPU-gated tests (dispatches on real GPU)
AXC_ENABLE_GPU_TESTS=1 cargo test --workspace --all-targets -- --ignored

# Force Lavapipe (software Vulkan, no hardware needed):
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json \
    AXC_ENABLE_GPU_TESTS=1 cargo test --workspace --all-targets -- --ignored
```

## LLM agent integration (MCP)

AXIOM-Compute ships a Model Context Protocol (MCP) server that exposes the optimization loop to external LLM agents. See [AGENTS.md](AGENTS.md) for the full protocol + example sessions.

```bash
# Start the MCP server over stdio
./target/release/axc mcp

# Smoke test
echo '{"jsonrpc":"2.0","id":1,"method":"initialize"}' | ./target/release/axc mcp
```

Six tools:
- `load_source` — parse a `.axc` file, return strategy holes + binding plan
- `enumerate_variants` — Cartesian product of all `@strategy` candidates
- `compile_variant` — materialize one assignment → SPIR-V (base64)
- `bench_variant` — dispatch + measure on local GPU
- `grid_search` — end-to-end: enumerate + compile + bench + pick winner
- `optimization_history` — JSONL append-only history keyed by source hash

## Key design decisions

1. **Annotations are first-class data**, not decorations. `@strategy { x: ?[...] }` is structured.
2. **No type inference.** Every type is explicit; every `let` has `: type`.
3. **`>>` operator forbidden.** Use `shr()` (arithmetic) or `lshr()` (logical).
4. **All opcodes via typed enum API.** Never raw `u32` values — they drift.
5. **BTreeMap, never HashMap**, for anything driving emission order.
6. **`spirv-tools::val` mandatory.** Every integration test validates in-process; no silent skip when the CLI is missing.
7. **Independent adversarial verification.** Every milestone goes through a 7-agent pipeline: Architect → dual design review → Coder → QA → dual code review. "Agents lie about own work" — cross-check via git diff, not self-reports.

## Benchmarks

On Intel i9-14900KF + NVIDIA RTX PRO 6000 Blackwell Workstation:

| Kernel | Size | Dispatch time |
|---|---|---|
| saxpy | 1 K elements | **38 μs** (was 691 μs pre-M2.3a — 18× speedup from pipeline cache) |
| saxpy | 1 M elements | **3.08 ms** (was 23 ms pre-M3.0 — 7.5× from HOST_CACHED staging + persistent map; host round-trip) |
| Q4_K_M matvec | 128 superblocks (32 K elements) | measured on Lavapipe — see `.pipeline/benchmarks/baselines.json` |

Run `cargo bench -p axc-driver` locally to generate baselines for your machine. Regression gate fails if any metric regresses > 15% with 11-sample median.

## Memory safety

- All `unsafe` blocks carry `// SAFETY:` comments (enforced via `#![warn(clippy::undocumented_unsafe_blocks)]`)
- `Arc<DeviceOwner>` + `Arc<InstanceOwner>` for Vulkan object lifetime
- RAII `DispatchResources` destroys in dependency order with `vkDeviceWaitIdle` guard
- `parking_lot::Mutex` on KernelHandle buffer pool for concurrent dispatch safety

## Related repos

- [axiom](https://github.com/rudybear/axiom) — CPU compiler (parent)
- [axiom-ports](https://github.com/rudybear/axiom-ports) — real-world C ports
- [axiom-engine](https://github.com/rudybear/axiom-engine) — game engine + Q2 port

## License

MIT OR Apache-2.0.

## Acknowledgments

Built through a 7-agent pipeline using Claude (Opus 4.7 for architects + pessimistic reviewers, Sonnet 4.6 for coders + optimistic reviewers). 88 commits across 13 milestones catch a representative sample of real GPU programming bugs: wrong opcode constants, missing `OpMemberDecorate Offset 0` on SSBO structs, misplaced `OpLoopMerge`, uncommitted work masquerading as done. Every bug was caught by the adversarial verification step before merge.
