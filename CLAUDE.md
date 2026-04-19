# AXIOM-Compute -- Project Context

## Project Identity

AXIOM-Compute is the **GPU/GPGPU sister project to [AXIOM](https://github.com/rudybear/axiom)**. Same thesis (AI-first language with semantic annotations that unlock downstream optimizations), different target (SPIR-V → Vulkan compute / OpenCL / WebGPU, instead of LLVM IR → clang -O2 → native CPU).

**This is NOT a shader/graphics language.** It is a compute language for AI agents to produce optimized GPGPU kernels across NVIDIA/AMD/Intel/Apple/mobile without CUDA lock-in.

**Repository:** https://github.com/rudybear/axiom-compute

See `DESIGN.md` for the living design document and `.pipeline/PIPELINE.md` for the 7-agent development workflow.

---

## Thesis (two sentences)

LLMs iteratively optimize CUDA kernels today at the source-text level, but every published system (Sakana AI CUDA Engineer, Kevin, CUDA-L1, EvoEngineer, STARK) operates on raw CUDA where optimization intent is implicit, leaving the rewrites vulnerable to reward hacking and correctness regressions. AXIOM-Compute makes that intent first-class (`@coalesced`, `@occupancy`, `@cooperative_matrix`, `@equiv_fp_tol`) so the compiler *verifies* what the LLM claims, producing portable SPIR-V that downstream vendor drivers can finish optimizing.

---

## What transfers from AXIOM, what doesn't

**Transfers:**
- Rust multi-crate compiler skeleton (lexer → parser → HIR → MIR → codegen → driver)
- `Spanned<T>` AST/IR node wrapper
- Annotations as first-class data
- MCP server for agent integration
- `@strategy { ?holes }` for LLM-driven autotuning
- `@strict` module discipline (`@intent` + `@complexity` + at least one pre/postcondition)
- 7-agent development pipeline
- Self-optimize loop (LLM reads → proposes rewrite → benchmark → iterate)

**Does NOT transfer (CPU-specific):**
- `noalias` on every pointer param (GPU has no pointer aliasing model like LLVM's)
- `nsw` on integer arithmetic (GPU ALUs don't poison like LLVM)
- `fast-math` via `@pure` (shader compilers have their own fast-math)
- Arena allocators (kernels don't have heaps)
- `@lifetime(scope)` heap-to-stack promotion (no stack/heap distinction on GPU)

The GPU optimization vocabulary is different and ~3× larger than CPU. See `DESIGN.md §3.2`.

---

## Beachhead

**llama.cpp Vulkan backend.** Current gaps (2025-2026):
- ggml-org/llama.cpp #16230: Vulkan-vs-CUDA performance regression
- ggml-org/llama.cpp #17273: Vulkan-vs-CUDA gap on A100
- ollama/ollama #15601: AMD users leaving ~56% tok/s on table
- ggml-org/llama.cpp #21517: Intel Arc Q8_0 at 21-24% of peak bandwidth

First target: match hand-tuned llama.cpp Vulkan within 5% on RTX 4090, beat it by ≥25% on AMD APU / Intel Arc, using **a single annotated source** producing portable SPIR-V.

---

## Architecture

```
AXIOM-Compute Source (.axc)        <- AI agents read/write
       |
       v
Lexer / Parser                     <- Port from AXIOM
       |
       v
HIR (annotation validation)
       |
       v
MIR (@strategy holes)              <- Autotuner resolves here
       |
       v
Dual codegen:
   Path A: LLVM IR -> LLVM SPIR-V backend (official Jan 2025, OpenCL flavor)
   Path B: rspirv direct -> Vulkan flavor with decorations
       |
       v
spirv-val  (reject malformed IR)
spirv-opt  (peephole, DCE)
       |
       v
SPIR-V binary -> Vulkan compute (ash) / OpenCL / WebGPU
       |
       v
Correctness oracle (@equiv_fp_tol) + benchmark
       |
       v
Self-optimize loop (LLM reads metrics -> proposes next @strategy values)
```

---

## Technology stack

- **Language:** Rust (target ~30-50K LOC across 7 crates, mirrors AXIOM)
- **SPIR-V emission:** `rspirv` (direct) + LLVM SPIR-V backend (via `.ll` text, mirror AXIOM)
- **GPU runtime:** `ash` for Vulkan, OpenCL ICD for OpenCL compute
- **Validator/optimizer:** `spirv-val`, `spirv-opt` from SPIRV-Tools
- **Build:** Cargo workspace
- **CI:** GitHub Actions (Linux + Lavapipe software fallback; NVIDIA/AMD/Intel smoke tests on self-hosted runners later)
- **Testing:** unit + integration + `spirv-val` + cross-vendor execution smoke + benchmark baselines

---

## Current milestone

**M0 -- Bootstrap.** Cargo workspace, crate skeletons, lexer port, "empty kernel" SPIR-V codegen, CI. Exit gate: `axc compile empty_kernel.axc -o empty.spv && spirv-val empty.spv` clean on CI.

After M0 is approved by both design reviewers → implement via Coder → QA → dual code review → merge.

---

## Hard rules (from parent project + verified incidents)

1. **Memory limits:** test executables must not exceed 50 GB (prior 336 GB crash incident).
2. **Independent verification is mandatory.** Agents lie about their own work. Every implementation claim is verified by a separate agent that reads the actual diff.
3. **Verdict enforcement:** every reviewer emits `{"verdict": "APPROVE|REJECT|NEEDS_REVISION", ...}` as the first key. Supervisor parses and gates the next phase on FAIL.
4. **No coding before design approval:** both design reviewers must APPROVE (via JSON verdict) before Coder starts.
5. **Issues go to Architect, not Coder:** if code review finds design problems, the Architect revises.

---

## 7-Agent pipeline (mirrors AXIOM)

| # | Agent | Model | Role |
|---|---|---|---|
| 1 | Architect | Opus | Design spec + plan + test requirements |
| 2 | Optimistic Design Reviewer | Sonnet | Validate feasibility, completeness |
| 3 | Pessimistic Design Reviewer | Opus | Adversarial review, UB/principle holes |
| 4 | Coder | Sonnet | Implement exactly what was agreed |
| 5 | QA | Sonnet | Verify diff conforms to plan -- never trust, always verify |
| 6 | Optimistic Code Reviewer | Sonnet | Spec compliance, quality |
| 7 | Pessimistic Code Reviewer | Opus | UB, races, correctness holes |

Templates live under `.pipeline/templates/`. Config under `.pipeline/config.json`.

---

## Anti-patterns to avoid

1. **No type inference.** Every type is explicit.
2. **No implicit returns.** Every function has `return`.
3. **No operator overloading.** `+` is always numeric add.
4. **No feature without a test.** Every feature ships with at least one correctness test on at least one GPU vendor.
5. **Annotations are first-class data, not decorations.**
6. **Parser must recover gracefully** -- report ALL errors.
7. **No string types for structured data.**
8. **No `>>` operator.** Use `shr()` / `lshr()` (same rationale as AXIOM).
9. **No mocking GPU execution in tests.** Correctness must be verified against real GPU (Lavapipe software GPU is the CI fallback).

---

## Current feature set

M0 complete: 7 crate skeletons, empty-kernel SPIR-V 1.3 codegen, spirv-val integration test. GPU execution (Lavapipe fallback) begins M1 — anti-pattern #9 is formally relaxed for M0 codegen-only milestone (syntactic validation via spirv-val substitutes).
