# Prepared upstream RFC — AXIOM-Compute Q4_K_M coopmat kernel into llama.cpp Vulkan

**Status: PREPARED — HOLD. Do NOT open until the trigger below is met.** This is a ready-to-open
RFC package; it is intentionally not submitted. (M4.2; see DESIGN.md §5 kill-criterion + the
`.pipeline/benchmarks/m34/` A/B results.)

**M4.2a pointer:** the committable half of this package (the ggml-ABI-matched kernel, the
generated SPIR-V, the host patch, REPRODUCE.md) is now real — see
`.pipeline/milestones/M4.2a-ggml-abi.md` for the full design + vendor citations.

## TL;DR for the future submitter

A complete feasibility study (2026-06-10) found that upstreaming AXIOM's Q4_K_M kernel into
llama.cpp's Vulkan backend is **technically feasible but should not be submitted yet**, because on
the only GPU we can measure (NVIDIA RTX PRO 6000) AXIOM is **2.39× slower** than llama.cpp's own
Q4_K Vulkan kernel, and the compelling value proposition (portability + winning on AMD/Intel, where
llama.cpp's Vulkan is documented-weak) is **entirely unmeasured**. Submitting now would spend
maintainer goodwill on a kernel that is slower on NVIDIA with an unsubstantiated cross-vendor
claim, earning a polite rejection that makes a later, stronger PR harder to land.

## THE TRIGGER (open the RFC only when this is true)

Run the same single, unchanged AXIOM `.axc` source on an **AMD APU (RDNA3+) or Intel Arc** and
produce a same-machine, same-ICD, kernel-only A/B vs llama.cpp's Vulkan Q4_K MUL_MAT at the
inference shape (m=4096, n=512, k=14336). **If AXIOM shows parity-or-better there** (plausible —
those are llama.cpp's documented weak spots: ggml-org/llama.cpp #16230, #17273; ollama #15601;
ggml-org/llama.cpp #21517), the story becomes genuinely novel: *one portable annotated source,
behind on NVIDIA but ahead on AMD/Intel*. That is an RFC worth a maintainer's time. If AMD/Intel
**also** loses, do not submit at all. (This is the EB.1 hardware item — the same blocker gates the
project's own cross-vendor kill-criterion.)

**M4.2a UPDATE (this section now references REAL, committed artifacts — see
`.pipeline/milestones/M4.2a-ggml-abi.md` for the full design + vendor citations at pinned SHA
`6b80c74f285390368b3c99c5e750f19e9b096e98`, tag b9542). Items 2-4 below were "held for hardware"
placeholders; M4.2a did the ABI-matching engineering (it does NOT require the target hardware —
only the AMD/Intel A/B trigger measurement does) and PREPARED the committable half of the package.
Status and the trigger below are UNCHANGED — this remains PREPARE-AND-HOLD.**

## What the PR/RFC would contain (ready to assemble)

1. **Framing:** an RFC/methodology proposal — *"portable single-source cooperative_matrix Q4_K
   kernels via AXIOM-Compute"* — NOT a perf patch. Honest headline: NVIDIA behind, AMD/Intel
   [the measured win].
2. **The kernel:** `examples/q4km_matmul_rb_coopmat_f32acc_cached_ggml.axc` (M4.2a — the
   ggml-ABI-matched variant, VERBATIM from the production leader
   `examples/q4km_matmul_rb_coopmat_f32acc_cached.axc` except the six ABI deltas below) + its
   generated SPIR-V, committed at `upstream/matmul_q4_k_f32_cm1_axiom.spv`.
3. **The integration (injection route B — opt-in, never default), NOW REAL:**
   `upstream/matmul_q4_k_f32_cm1_axiom_data.hpp.fragment` embeds AXIOM's SPIR-V as a byte array
   next to the REAL KHR-coopmat symbol **`matmul_q4_k_f32_cm1_data[]`** (§2.9 of the M4.2a spec —
   an earlier draft targeted the wrong, never-used-on-coopmat-HW scalar symbol
   `matmul_q4_k_f32_data[]`; corrected). `upstream/ggml-vulkan-axiom-q4k.patch` registers a
   STANDALONE `matmul_q4_k_f32_cm1_axiom` pipeline behind `GGML_VULKAN_AXIOM_Q4K` (opt-in), and
   **overrides the RESULT of `ggml_vk_guess_matmul_pipeline`** at its call site
   (`ggml-vulkan.cpp:8239`, inside `ggml_vk_mul_mat_q_f16`) under a 13-condition fail-closed guard
   — ONE interception point covers ALL SIX stock variants (`{s,m,l}×{unaligned,aligned}`),
   alignment-independent. Pipeline creation sits inside the coopmat block
   `ggml-vulkan.cpp:4090-4211` (push struct `:1036-1044`, the `stride_a=ne10` call site
   `ggml_vk_matmul` `:8403-8409`). `git apply --check` verified against the pinned SHA
   (AT-2998; `crates/axc-driver/tests/m42a_patch_applies.rs`).
4. **The `.axc` ABI-matching rewrite — DONE (M4.2a), the six deltas, all GPU-proven on NVIDIA:**
   - 3 bindings A=0/B=1/D=2 — already matched (zero work).
   - **Weight layout already bit-exact** (144-byte `block_q4_K`; AXIOM's dequant was built against
     ggml) — the one clean win, lead with it. UNCHANGED in M4.2a.
   - `n_blocks_per_row` DERIVED from `stride_a / 256u32` (ggml's `stride_a` is ELEMENTS —
     `ggml_vk_matmul(ne01,ne11,ne10,ne10,ne10,stride_d,…)`, so `stride_a=ne10=K`; `/256`, NOT
     `/144`). The kernel consumes ggml's 17-field push struct, reading only the leading six
     (M,N,K,stride_a,stride_b,stride_d at std430 offsets 0..20 — byte-identical by construction).
   - **B (activations) retyped f32** (`readonly_buffer[f32]`, matching ggml's `matmul_q4_k_f32`
     B_TYPE=float), read TRANSPOSED (`x[n*stride_b+k]`, ggml's `[N,K]` row-major), and staged
     f32→f16 via the existing M3.5 `f32_to_f16` builtin. **D stored COLUMN-MAJOR** with
     `stride_d` via the NEW `coopmat_store_col` codegen builtin (mirrors `coopmat_store`, adds
     only the `ColumnMajorKHR` layout OPERAND — no new SPIR-V capability;
     `crates/axc-codegen/src/coopmat.rs`). **Grid axis order SWAPPED** to ggml's convention
     (`gl_WorkGroupID.x`→M-row, `.y`→N-col — the opposite of the leader); the host adapter
     registers `wg_denoms={32,32,1}`.
   - Bit-exactness RE-PROVEN on real NVIDIA hardware: cross-kernel BIT-IDENTITY vs the leader
     (de-transposed, f16-exact B fixture, `max|Δbits|==0` at K=256/512/14336) AND the standing
     combined condition-aware ≤ FROZEN 1e-3 vs the f32 oracle, INCLUDING the real rectangular A/B
     shape (M=4096, N=512, K=14336) so a grid-axis-swap bug could not pass silently. See
     `crates/axc-driver/tests/dispatch_q4km_ggml.rs`,
     `crates/axc-driver/tests/dispatch_q4km_ggml_equiv.rs`.
5. **The honest A/B table:** NVIDIA 42.86 vs 102.49 TFLOPS = 2.39× behind (the M3.6 leader,
   M3.13-concluded), kill-criterion FAIL, same-machine/same-ICD/same-shape, kernel-only (no
   GEMV-vs-GEMM flattering) + the AMD/Intel row from the trigger (still unmeasured). PLUS the NEW
   M4.2a ggml-variant self-A/B (ggml-variant TFLOPS vs the M3.6 leader, same shapes, on NVIDIA —
   records the f32→f16 B-staging convert + transposed/column-major addressing cost honestly, NO
   gate) — see `upstream/REPRODUCE.md` and `.pipeline/benchmarks/m34/ab_results_fused_f32acc_cached_ggml.json`.

## Why AXIOM is behind on NVIDIA (state this honestly in the RFC)

It is architectural, not a tuning knob (M3.7 + M3.8 are merged honest-negatives proving the
occupancy ceiling): llama.cpp's `mul_mm` uses large warptiles (BM/BN up to 128, WMITER/WNITER
register blocking, BK-stepped staging, multi-warp workgroups, split-K, aligned vectorized loads)
and **dequantizes A inline during shared staging**; AXIOM is a single-subgroup `@workgroup(32,1,1)`
2×2-tile kernel paying a dequant-front-end tax. The 2.39× is not quickly closable on NVIDIA. The
thesis bet is that the *portable single source* wins precisely where the hand-tuned GLSL backend is
weakest — AMD/Intel — which is the trigger above.

## RFC-open trigger preconditions (M4.2a §7.5 — appended, the trigger paragraph above is unchanged)

The trigger above (AMD/Intel A/B parity-or-better) is necessary but, per the M4.2a r2 review, NOT
by itself sufficient to open the RFC. Both of the following are HARD preconditions of the
RFC-open trigger:

- [ ] **The AMD/Intel A/B trigger condition** (the paragraph above) — UNMEASURED, EB.1-gated.
- [ ] **The executed real-ggml ABI smoke** — building `llama.cpp` with the
      `GGML_VULKAN_AXIOM_Q4K` patch applied and running actual inference (bit-close output vs
      stock ggml) on the pinned SHA. This is the ONLY test that exercises the REAL bound
      buffers/dispatch/f32 `src1`/column-major D — the static ABI goldens
      (`crates/axc-driver/tests/m42a_ggml_abi_golden.rs`) are the CI-runnable substitute until
      this runs, NOT a replacement for it. NOT executed in M4.2a (explicit out-of-scope); see
      `upstream/REPRODUCE.md` for the deferred build+run steps.

Both boxes must be checked before the RFC referenced by this document is opened.

## Recommendation

PREPARE-AND-HOLD (this document). Engineer the `.axc` rewrite + host patch only when AMD/Intel
hardware is in hand and shows the trigger condition; submit the RFC only if it does.
