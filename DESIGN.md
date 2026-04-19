# AXIOM-Compute -- Design Document (v0.1, Living)

> Working document. Updated at every design-review approval. Last revision: bootstrap (M0, 2026-04-18).

---

## 1. Problem statement

### 1.1 The gap

GPU kernel programming is splintered across CUDA (NVIDIA-locked), ROCm/HIP (AMD), Metal (Apple), Vulkan compute + SPIR-V (cross-vendor), OpenCL (deprecated but reusable), and WebGPU (browser). Every existing shader/compute language was designed for humans:

| Project | Surface | LLM-first annotation layer? |
|---|---|---|
| CUDA C++ | C++ with `<<<>>>` | No |
| HLSL / GLSL | C-like shading | No |
| Slang (Khronos, 2024) | HLSL-superset + autodiff | No (autodiff, not LLM-intent) |
| Rust-GPU | Rust MIR → SPIR-V | No |
| Triton | Python tile DSL | No (tile abstraction, not intent) |
| TileLang / Gluon / ThunderKittens | CUDA-embedded / tile DSLs | No |
| IREE | MLIR → SPIR-V | No (graph-level) |
| MLIR SPIR-V dialect | Compiler plumbing | No |

No project was designed around *LLMs as primary authors* with an explicit semantic-intent annotation vocabulary over SPIR-V.

### 1.2 The opportunity

LLMs iteratively optimize CUDA kernels at the *source-text* level today (Sakana AI CUDA Engineer, Kevin, CUDA-L1, EvoEngineer, STARK, CudaForge, GPU Kernel Scientist, Simon Guo's Metamorph). Every one works on raw CUDA, with optimization intent implicit. The LLM infers `@coalesced` from the access pattern, `@cooperative_matrix` from the tile shape, `@divergence_free` from the branch structure -- but none of that inference is *verified*, which is why Sakana had reward-hacking retractions in 2025.

**AXIOM-Compute's wedge:** make the inference explicit. If the source says `@coalesced`, the compiler *verifies* coalescing or rejects the code. If the source says `@equiv_fp_tol(1e-3)`, rewrite proposals are machine-checked against the baseline. This is exactly the hole Sakana left open.

### 1.3 What AXIOM (CPU) contributes and does not

**Transfers from AXIOM:** Rust-based multi-crate compiler skeleton, lexer/parser/HIR/codegen structure, `Spanned<T>` pattern, `@annotations` as first-class data, MCP server for agent integration, `@strategy` holes for LLM-driven tuning, `@strict` with pre/postconditions, self-optimize loop, 7-agent development pipeline.

**Does NOT transfer:** `noalias` / `nsw` / `fast-math` / arena allocators / `@lifetime` heap-to-stack. These are CPU-LLVM-specific advantages. The GPU optimization vocabulary is different and ~3× larger (see §3.2).

---

## 2. Architecture

```
.axc source (annotated for LLM consumption)     <- LLM agents author here
       |
       v
Lexer (reuse AXIOM recursive descent)
       |
       v
Parser (Pratt expressions, AST)
       |
       v
HIR (GPU annotation validation, type checking, @strict enforcement)
       |
       v
MIR (lowered, auto-tune @strategy holes resolved here)
       |
       v
Codegen: dual-path
    Path A: LLVM IR -> LLVM SPIR-V backend (official Jan 2025)
            -> OpenCL-flavor SPIR-V for oneAPI / OpenCL compute
    Path B: rspirv direct -> Vulkan-flavor SPIR-V with decorations
       |
       v
spirv-val  (validator -- reject malformed IR)
spirv-opt  (peephole, constant-propagation, DCE)
       |
       v
SPIR-V binary
       |
       v
Loader/Dispatcher (ash for Vulkan, OpenCL ICD for OpenCL)
       |
       v
Benchmark + Correctness check (tolerance-aware)
       |
       v
Agent self-optimize loop (LLM proposes new @strategy values -> re-run)
```

### 2.1 Crate layout

```
axiom-compute/
├── crates/
│   ├── axc-lexer/       # Tokenizer (port AXIOM's lexer + GPU-keyword additions)
│   ├── axc-parser/      # Recursive descent + Pratt expressions
│   ├── axc-hir/         # Annotation validation, type checking, @strict enforcement
│   ├── axc-codegen/     # SPIR-V emission (rspirv direct + LLVM SPIR-V backend)
│   ├── axc-optimize/    # @strategy hole resolution, autotuner, LLM bridge
│   ├── axc-driver/      # CLI, MCP server, dispatcher, benchmark harness
│   └── axc-runtime/     # Vulkan/OpenCL loader, kernel launch wrappers
├── spec/                # Formal language spec (grammar, types, annotations)
├── examples/            # First target: llama.cpp Vulkan equivalents
├── benchmarks/          # KernelBench-Vulkan submission + llama.cpp comparisons
└── .pipeline/           # 7-agent development pipeline (adapted from AXIOM)
```

---

## 3. Language

### 3.1 Types

```
Primitives:     i8 i16 i32 i64 u8 u16 u32 u64 f16 bf16 f32 f64 bool
Vectors:        vec2..vec4 (f32), dvec2..dvec4 (f64), ivec2..ivec4 (i32), uvec2..uvec4 (u32)
Buffers:        buffer[T]              // SSBO on Vulkan, cl_mem on OpenCL
                readonly_buffer[T]     // readonly decoration
                writeonly_buffer[T]    // writeonly decoration
Images:         image2d[T] image3d[T]  // opaque image handles
Shared:         shared[T, N]           // workgroup-local memory (SLM)
Matrices:       matrix[T, M, N]        // cooperative_matrix
Subgroup:       subgroup[T]            // subgroup-uniform values
```

### 3.2 GPU-specific annotations (the core value add)

| Annotation | Meaning | Lowering |
|---|---|---|
| `@kernel` | Entry point (compute shader) | `OpEntryPoint GLCompute` + `OpExecutionMode` |
| `@workgroup(X, Y, Z)` | Workgroup dimensions | `LocalSize` execution mode |
| `@subgroup_uniform` | Value invariant across subgroup | `SubgroupUniform` decoration |
| `@uniform_branch` | Control flow uniform across subgroup | Guides structurizer |
| `@divergence_free` | No divergence in this region | Verified + unlocks shuffles |
| `@coalesced(stride=1)` | Adjacent threads access adjacent elements | Verified by access-pattern analysis |
| `@shared_tile(M, N, pad=K)` | SLM tile with bank-conflict padding | Emits `shared[T, M*(N+K)]` |
| `@no_bank_conflict` | Asserted; verified by swizzle/stride analysis | Compile-time check |
| `@cooperative_matrix(M, N, K, A_type, B_type, C_type)` | Tensor-core-style tile op | `OpCooperativeMatrixLoadKHR` / `MulAddKHR` |
| `@occupancy(min=0.5)` | Minimum target occupancy | Register-pressure estimate; rejects if violated |
| `@max_registers(N)` | Register budget | Informs codegen + spill heuristics |
| `@async_copy` | Overlap memory + compute | `cp.async` / equivalent where available |
| `@reduction(op)` | Parallel reduction over subgroup/workgroup | `SubgroupReduce` / tree reduction |
| `@barrier(scope)` | Explicit barrier with scope | `OpControlBarrier` |
| `@equiv_fp_tol(eps)` | FP tolerance for rewrite verification | Correctness oracle |
| `@strategy { tile_m: ?, tile_n: ?, stages: ? }` | LLM autotune hole | Resolved by `axc optimize` |
| `@strict` | Require `@intent` + `@complexity` + pre/post on every kernel | Compile error if missing |
| `@precondition(expr)` / `@postcondition(expr)` | Runtime-checkable contract | Debug-mode assertion |
| `@target(vendor, caps)` | Target capability set | Guards codegen paths |

### 3.3 SPIR-V extensions in scope

**Portable baseline (must work on NVIDIA/AMD/Intel):**
- `SPV_KHR_shader_subgroup*` (ballot, vote, shuffle, reduce)
- `SPV_KHR_cooperative_matrix` (tensor-core portable access, promoted in Vulkan 1.3.300)
- `SPV_KHR_float_controls`
- `SPV_KHR_16bit_storage` / `SPV_KHR_8bit_storage`
- `SPV_KHR_integer_dot_product`
- `SPV_EXT_shader_atomic_float_add`

**Vendor fast paths (path-split, not required):**
- `SPV_NV_cooperative_matrix2` (workgroup-scope matrices, per-element ops) -- NVIDIA only
- `SPV_INTEL_*` -- Intel DPC++ compute

---

## 4. Phased plan

### Phase M0 -- Bootstrap (week 0-2)

- Cargo workspace with 7 empty crates
- Lexer port from AXIOM with GPU keyword additions
- `axc lex` CLI emitting tokens
- CI on GitHub Actions (test + build)
- `.pipeline/` copied and adapted (this document produced)
- First kernel compiles (empty kernel with `@workgroup(64,1,1)`) → SPIR-V → spirv-val clean

**Exit gate M0:** `axc compile empty_kernel.axc -o empty.spv && spirv-val empty.spv` succeeds on Linux CI.

### Phase M1 -- Minimum viable SPIR-V backend (month 1-3)

- Parser + HIR + SPIR-V codegen path for: scalar ops, buffers, workgroup dispatch, barriers, subgroup ops
- `axc run` dispatcher on Vulkan via `ash`
- Validator integration (`spirv-val`, `spirv-opt`)
- Saxpy + vector add + parallel reduction benchmarks running on NVIDIA + AMD + Intel + Lavapipe

**Exit gate M1:** 3-vendor execution of 5 elementwise kernels, correctness verified vs CPU reference.

### Phase M2 -- Cooperative matrix + llama.cpp Q4_K_M matmul (month 3-9)

- `@cooperative_matrix` lowering to `SPV_KHR_cooperative_matrix`
- `@strategy` hole infrastructure + LLM autotuner
- MCP server port from AXIOM
- llama.cpp Q4_K_M dequant + matmul reference kernel
- Bench harness producing `tok/s` comparison vs llama.cpp current Vulkan baseline

**Exit gate M2:** On RTX 4090, within 5% of hand-tuned llama.cpp Vulkan Q4_K_M. On AMD APU or Intel Arc, beat current Vulkan by ≥25%. One annotated source, three targets.

### Phase M3 -- FlashAttention + KernelBench-Vulkan submission (month 9-15)

- Attention kernel (FA2 shape, then FA3 where extensions allow)
- First public SPIR-V KernelBench submission
- `@equiv_fp_tol` correctness oracle (Sakana-proof verification)

**Exit gate M3:** ≥80% of cuBLAS+FA3 on H100 via Vulkan compute; ≥90% of rocBLAS on MI300X. Public KernelBench leaderboard entry.

### Phase M4 -- PyTorch custom-op frontend + adoption (month 15-24)

- `torch.compile` backend or `torch.library` custom-op integration
- Upstream PR to llama.cpp (or candle/MLX) replacing at least one hot kernel with AXIOM-Compute output

**Exit gate M4:** External adoption by ≥1 production inference framework.

---

## 5. Kill criteria (pre-registered)

Stop the project if, at the stated gate:

- **M1 slip (>6 mo):** cannot produce a correctness-verified SPIR-V kernel on 3 vendors. Means the validator/lowering path is structurally broken.
- **M2 slip (>12 mo):** cannot match llama.cpp Vulkan Q4_K_M within 15% on *any* vendor. Means annotation-to-SPIR-V lowering adds no value above handwritten GLSL.
- **M3 slip (>18 mo):** FA variant cannot clear 50% of cuBLAS+FA3 on any GPU. Means LLMs + annotations cannot close the handwritten-vendor-kernel gap -- project thesis is refuted.
- **Ecosystem preempt:** if Slang grows an equivalent annotation sublanguage, or Triton-Vulkan ships and hits M2 targets first, pivot to contribute there instead of competing.

---

## 6. Development workflow

7-agent pipeline inherited from AXIOM (see `.pipeline/PIPELINE.md`):

1. **Architect** (Opus) -- designs spec + plan
2. **Optimistic Design Reviewer** (Sonnet) -- validates feasibility + completeness
3. **Pessimistic Design Reviewer** (Opus) -- adversarial review
4. **Coder** (Sonnet) -- implements agreed plan
5. **QA** (Sonnet) -- verifies test conformance, not-trust-always-verify
6. **Optimistic Code Reviewer** (Sonnet) -- spec compliance + quality
7. **Pessimistic Code Reviewer** (Opus) -- UB, races, correctness

**Verdict enforcement:** every reviewer agent emits `{"verdict": "APPROVE|REJECT|NEEDS_REVISION"}` as the FIRST key in a JSON object. Supervisor parses this and gates the next pipeline phase. Memory-tracked rule: agents lie about their own work, so independent adversarial verification is mandatory.

**Memory limits:** test executables must not exceed 50 GB (from prior incident with a 336 GB crash).

---

## 7. Open questions (M0 -> architect resolves)

1. OpenCL SPIR-V flavor vs Vulkan SPIR-V flavor -- emit both, or OpenCL first?
2. Direct `rspirv` emission vs going through LLVM SPIR-V backend -- which for M1?
3. Naming: `.axc` for source files (for AXIOM-Compute) or `.axm` reused? Current doc assumes `.axc`.
4. Runtime dispatch: start with Kompute (Vulkan-only) or `ash` (more control)? Current doc assumes `ash`.
5. How much should we inherit from `axiom` as a git subtree vs re-implement? Prefer crate-level reuse where APIs are stable.
6. Where is the line between `axc-optimize` (autotuner) and `axc-driver` (MCP server)? Keep them separate, match AXIOM layout.

---

## Revision log

- **2026-04-18:** Initial draft (v0.1), pre-architect review. To be revised through dual design review.
