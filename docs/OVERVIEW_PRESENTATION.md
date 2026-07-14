---
marp: true
theme: default
paginate: true
title: "AXIOM-Compute — State of the Project"
description: "An onboarding deck for GPU/graphics engineers with zero prior context"
---

<!--
Speaker note: audience = graphics/GPU programmers who know Vulkan, SPIR-V,
tensor cores, shaders, occupancy, bank conflicts — but have NEVER heard of this
project. No AXIOM context assumed. Goal: in ~18 slides, convey what it is,
what we built, what we proved, and the one gap that actually matters.
-->

# AXIOM-Compute

### A portable, *intent-verified* compute language for AI-generated GPU kernels

**State of the project — onboarding deck**
For GPU engineers seeing this for the first time

<!-- 2026-06-22 -->

---

## The 10-second version

> You write **one** annotated source file.
> It compiles to **portable SPIR-V**.
> It runs **bit-exact on real tensor cores** — NVIDIA today, AMD/Intel/Apple by design.
> And the **compiler verifies the optimization claims** instead of trusting them.

Think of it as: **a shader compiler, but for GPGPU/ML kernels, where the
optimization *intent* is part of the type system.**

---

## The problem we're attacking

LLMs already optimize GPU kernels (Sakana CUDA Engineer, Kevin, CUDA-L1, …).

But they all operate on **raw CUDA text**:

- Optimization intent is **implicit** — buried in how the code happens to be written.
- The model can **reward-hack**: "make it faster" → silently changes the math.
- Correctness regressions slip through because nothing *checks* the claim.
- And it's **CUDA — single-vendor lock-in.** Nothing portable comes out.

**Graphics analogy:** imagine if the only way to ship a shader was to hand a
text file to an intern who rewrites your inner loop for speed — and nobody
re-runs the reference image to check it still renders correctly.

---

## The idea: make intent *first-class and verifiable*

Optimization intent becomes **annotations the compiler checks**:

| Annotation | Means | Compiler verifies |
|---|---|---|
| `@coalesced` | this access is memory-coalesced | the access pattern actually is |
| `@occupancy(n)` | target N resident warps | register/shared budget allows it |
| `@cooperative_matrix` | use tensor-core MMA | shape/types map to a valid coopmat op |
| `@equiv_fp_tol(1e-3)` | rewrite must match reference within tol | runs both, compares on real GPU |

So when an LLM says *"I fused the dequant into the matmul and it's still
correct,"* the compiler **proves it** (bit-exact / within tolerance on hardware)
instead of taking its word.

---

## Where it sits in the stack

```
        AXIOM-Compute source (.axc)        ← AI agents read/write this
                  │
          Lexer → Parser → HIR → MIR        ← Rust multi-crate compiler
                  │   (annotation validation, @strategy autotuning holes)
                  ▼
        ┌──────── Dual codegen ────────┐
        │ rspirv  (Vulkan flavor)      │   ← direct SPIR-V w/ decorations
        │ LLVM SPIR-V backend (OpenCL) │
        └──────────────┬───────────────┘
                  │
         spirv-val  (reject malformed)
         spirv-opt  (peephole, DCE)
                  │
                  ▼
   SPIR-V binary → Vulkan compute (ash) / OpenCL / WebGPU
                  │
                  ▼
   Correctness oracle (@equiv_fp_tol) + benchmark on REAL GPU
```

**Sister project to AXIOM** (the CPU/LLVM version). Same thesis, GPU target.

---

## What's BUILT — the compiler & language

✅ Full Rust compiler: **lexer → parser → HIR → MIR → dual codegen → driver** (7 crates)
✅ **`@strategy { ?hole }`** tuning holes + **grid-search autotuner**
✅ **MCP server** so an LLM agent can drive compile/tune/benchmark as tools
✅ **`shared[T,N]`** workgroup-local memory (full pipeline + divergent-barrier safety analysis)
✅ **OpPhi loop-carried SSA** — real reduction loops, not just straight-line code
✅ **Cooperative-matrix (tensor-core) codegen** + dispatch
✅ **spirv-val clean**, dual code-review gate, **713+ tests**

**Everything is real:** no mocked GPU execution. Tests run on a real Vulkan
device (NVIDIA RTX PRO 6000 Blackwell) with a **Lavapipe** software fallback in CI.

---

## What's BUILT — the GPU kernels

✅ **Q4_0 + Q4_K_M** quantized weight kernels — **bit-exact vs CPU reference** on real hardware
✅ First-ever **cooperative_matrix dispatch on Blackwell tensor cores** (C = A·B, `max_diff = 0`)
✅ **Register-blocked coopmat matmul** — measured TFLOPS, not just "it compiles"
✅ **FlashAttention-2** — streaming online-softmax, **no S-matrix materialization**, coopmat-accelerated, real `exp()` builtin
✅ **PyTorch integration**: CUDA↔Vulkan **zero-copy** (torch tensor + AXIOM kernel share one physical alloc, no host copy), registered as `torch.ops.axiom.*`, auto-lowered under `torch.compile(backend="axiom")`

The thesis spans **both GEMM and attention** — the two kernels that matter for LLM inference.

---

## The flagship experiment: Q4_K_M vs hand-tuned llama.cpp

**Setup:** same machine, same Vulkan ICD, same op (Q4_K matrix-multiply).
llama.cpp pinned at a fixed tag. Work-normalized TFLOPS, kernel-only timing.

This is the honest A/B that tells us how close a *portable, machine-generated*
kernel gets to a *hand-tuned, vendor-specific* one.

**Why Q4_K_M?** It's the dominant quantization format for local LLM inference —
4-bit weights in a superblock structure with per-block scales. The kernel must
**dequantize on the fly** and feed a matmul. It's exactly where hand-tuning pays off.

---

## The journey: 87,000× → 2.39×

| Milestone | What changed | Gap to llama.cpp |
|---|---|---|
| M3.4 | naive single-row matvec | **87,000×** 😱 |
| M3.5b | f32 accumulation, multi-row | 9.3× |
| **M3.6** | **dequant scale-caching, 2×2 register tile** | **2.39×** ✅ |

**M3.6 = the production leader: 42.86 TFLOPS, bit-identical output.**

From five orders of magnitude behind, to within ~2.4× of a kernel that an
expert hand-wrote for this exact GPU — from **one portable source file**.

---

## Anatomy of the remaining 2.39×

We didn't just stop at 2.39× — we **decomposed it** with same-shape ablation
experiments (measured on NVIDIA):

```
   2.39×  =  1.35×  (matmul core)   ×   1.77×  (dequant front-end)
             └─ near the KHR ceiling     └─ THE dominant gap
```

- **1.35× core**: our portable coopmat matmul vs llama's. Near the achievable
  ceiling for `VK_KHR_cooperative_matrix`.
- **1.77× dequant**: the on-the-fly 4-bit→f16 unpack. Pinned to
  **register pressure / occupancy**, not arithmetic latency, not bank conflicts,
  not load width.

Every number here is from a controlled, bit-identical ablation — not a guess.

---

## The campaign that ruled out *every portable lever*

We ran the classic "fix the compiler" optimization playbook. Each lever was
built, measured on the real GPU, and **merged as a documented experiment** —
including the negatives. Honesty over hype.

| Lever tried | Result |
|---|---|
| Double-buffering (M3.7) | ❌ not latency-bound |
| Larger register tiles 4×4 (M3.8) | ❌ not arithmetic-intensity-bound |
| Multi-subgroup warptile (M3.9) | ❌ not warp-occupancy-bound |
| Bank padding (M3.10a) | ❌ not bank-conflict-bound |
| Vectorized loads (M3.10b) | ❌ front-end is ALU-bound (40:1 ALU:load) |
| Integer strength-reduction (M3.11) | ❌ index math is latency-hidden |
| Live-range tightening (M3.13) | ❌ register pressure not source-addressable |
| Fused dequant→coopmat (M3.13) | ⛔ **blocked by the KHR API itself** |

---

## Why we can't close the last 1.77×: the "portable-coopmat tax"

`VK_KHR_cooperative_matrix` exposes **exactly 4 opaque ops**:
`load`, `store`, `mul_add`, `zero`.

There is **no per-element write into a coopmat fragment.**

So to dequantize 4-bit weights into a tile, we **must** round-trip through
shared memory before `coopmat_load`.

llama.cpp's hand-tuned path uses **vendor extensions (NV coopmat2)** to
dequantize **directly into registers** — skipping that round-trip.

> **The residual gap is the price of portability**, not a missing trick.
> Matching it requires NV-only extensions that **break the cross-vendor thesis**.

**Graphics analogy:** it's like being forced to go through a framebuffer
round-trip because the API won't let you write individual fragments of a
tensor-core tile — while the vendor's private path writes them straight from ALU.

---

## What we have honestly PROVEN

✅ A **single annotated source** compiles to **portable SPIR-V** and runs
   **bit-exact on real tensor cores.**

✅ The intent-verification model works: **every** optimization claim was
   checked bit-exact / within `@equiv_fp_tol` on real hardware.

✅ On NVIDIA, a portable machine-generated Q4_K_M kernel reaches **within 2.39×**
   of an expert hand-tuned vendor-specific kernel — and we can **explain every
   factor** of that gap.

✅ The thesis spans **GEMM + FlashAttention + PyTorch interop** — a real stack,
   not a toy.

---

## The gap that actually matters

⚠️ **The central thesis is cross-vendor — and it's the unmeasured part.**

The whole value proposition is:

> *"From one unchanged source, **beat** llama.cpp Vulkan by ≥25% **where it's
> weak** — AMD APU, Intel Arc — while staying portable."*

On NVIDIA, hand-tuned CUDA-class code wins (the portability tax). **That's
expected.** The bet is that portability wins on the *other* vendors —
and **this development box is NVIDIA-only.**

🚧 **EB.1 (cross-vendor A/B on AMD/Intel)** is **hardware-gated.**
🚧 **M4.2 (upstream llama.cpp PR)** is **prepared and held** — it needs that
   cross-vendor data before it's compelling, plus a go-ahead to submit.

---

## Scorecard

| Dimension | Status |
|---|---|
| Compiler + language | ✅ complete, 713+ tests |
| Portable SPIR-V → real GPU, bit-exact | ✅ proven |
| Intent annotations verified on hardware | ✅ proven |
| Tensor-core (coopmat) GEMM + attention | ✅ proven |
| PyTorch zero-copy + `torch.compile` | ✅ complete |
| NVIDIA Q4_K_M vs hand-tuned llama.cpp | 🟡 2.39× behind — fully explained (portability tax) |
| **Cross-vendor win (AMD/Intel) — the thesis** | 🔴 **UNMEASURED — hardware-gated** |
| Upstream adoption (llama.cpp PR) | 🟡 prepared & held |

---

## Where we are, and the fork ahead

**The autonomous performance campaign is concluded.** Everything provable on the
available hardware is proven; every portable lever was tried and documented.

The project is at an **honest inflection point**:

- 🔴 **The decisive experiment needs hardware** (AMD RDNA3 / Intel Arc).
- 🟡 **The upstream PR is an outward action** awaiting that data + a go-ahead.

**Unblocked work we *can* still do without hardware:**
- More quant formats (Q5_K_M / Q6_K) — broadens the kernel library
- Language/tooling feature gaps (`axc rewrite`, runtime asserts, multi-kernel modules)
- Polish & close-out

---

## One-slide takeaway

> **AXIOM-Compute proves you can generate portable, tensor-core GPU kernels from
> one annotated source, with the compiler verifying every optimization claim
> bit-exact on real hardware.**
>
> On NVIDIA it lands **within a fully-explained 2.39×** of expert hand-tuned code —
> the unavoidable **price of portability** through the KHR coopmat API.
>
> **The thesis-deciding question — does portability *win* on AMD/Intel? —
> is built, instrumented, and waiting on hardware.**

*Questions?*
