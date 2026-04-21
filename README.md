# AXIOM-Compute

**AI-first compute language for SPIR-V.** Sister project to [AXIOM](https://github.com/rudybear/axiom) (CPU via LLVM IR), targeting GPU/GPGPU via SPIR-V → Vulkan compute / OpenCL / WebGPU.

---

## Thesis

LLMs iteratively optimize CUDA kernels today at the source-text level (Sakana AI CUDA Engineer, Kevin, CUDA-L1, EvoEngineer, STARK). Every published system works on raw CUDA where optimization intent is implicit, which is why they reward-hack and produce correctness regressions.

AXIOM-Compute makes intent first-class. `@strategy { workgroup_x: ?[32, 64, 128, 256] }` declares holes the compiler enumerates and a grid-search (or LLM agent via MCP) fills. `@equiv_fp_tol(1e-3)` is machine-checked. The result is portable SPIR-V that downstream vendor drivers finish optimizing — without CUDA lock-in.

## Current status (2026-04-21)

- **13 milestones merged on `main`**: M0 → M2.6
- **713 tests passing**, clippy `--all-targets` clean, zero SPIR-V validation errors
- **Real GPU execution** via `ash` 0.38 + Vulkan 1.1+
- **6 GPU-gated tests pass on both NVIDIA RTX PRO 6000 Blackwell AND Lavapipe** (software Vulkan)
- **Q4_K_M kernel dispatches bit-exact** against ggml CPU reference on real GPU — the llama.cpp beachhead

| Kernel | Status |
|---|---|
| saxpy | ✅ bit-exact, 38 μs (1024 elem) on NVIDIA |
| vector_add | ✅ bit-exact |
| reduction / workgroup barrier | ✅ compile + validate |
| subgroup reduce | ✅ compile + validate |
| Q4_0 dequant matvec | ✅ bit-exact on NVIDIA + Lavapipe |
| **Q4_K_M dequant matvec** | ✅ **bit-exact on NVIDIA** — llama.cpp beachhead |
| cooperative matrix matmul | ✅ compile + spirv-val (dispatch requires tensor-core hardware) |
| FlashAttention-2 | 🔜 M3.1 |

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
| saxpy | 1 M elements | 26 ms (staging-bound; bandwidth optimization is M3) |
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
