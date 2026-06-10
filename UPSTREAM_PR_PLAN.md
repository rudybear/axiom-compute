# Prepared upstream RFC — AXIOM-Compute Q4_K_M coopmat kernel into llama.cpp Vulkan

**Status: PREPARED — HOLD. Do NOT open until the trigger below is met.** This is a ready-to-open
RFC package; it is intentionally not submitted. (M4.2; see DESIGN.md §5 kill-criterion + the
`.pipeline/benchmarks/m34/` A/B results.)

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

## What the PR/RFC would contain (ready to assemble)

1. **Framing:** an RFC/methodology proposal — *"portable single-source cooperative_matrix Q4_K
   kernels via AXIOM-Compute"* — NOT a perf patch. Honest headline: NVIDIA behind, AMD/Intel
   [the measured win].
2. **The kernel:** `examples/q4km_matmul_rb_coopmat_f32acc_cached.axc` + its generated SPIR-V.
3. **The integration (injection route B — opt-in, never default):** embed AXIOM's SPIR-V as a new
   byte array next to `matmul_q4_k_f32_data[]`, register a `matmul_q4_k_f32_axiom` pipeline behind
   a build flag (e.g. `GGML_VULKAN_AXIOM_Q4K`), selected only when the flag is set. Touch points:
   `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (pipeline registration ~:4040-4474, push struct :1036,
   dispatch `ggml_vk_matmul` :7847) + the generated `ggml-vulkan-shaders.hpp`.
4. **The required `.axc` ABI-matching rewrite** (the real engineering, held for hardware):
   - 3 bindings A=0/B=1/D=2 — already match.
   - **Weight layout already bit-exact** (144-byte `block_q4_K`; AXIOM's dequant was built against
     ggml) — the one clean win, lead with it.
   - Re-derive `n_blocks_per_row` from `stride_a / (QUANT_K bytes)` instead of taking it as a
     push constant; accept ggml's 17-field push struct (read M/N/K/stride_b/stride_d; assert
     `k_split==1`, `num_batches==1`).
   - **Flip the activation staging + the C store to ggml's orientations** (B row-major [N,K]
     loaded column-major; D column-major with `stride_d`) — AXIOM currently uses [K,N] / row-major.
   - Re-prove bit-exactness (the existing combined condition-aware metric ≤ frozen 1e-3) against
     the transposed layout before claiming correctness.
5. **The honest A/B table:** NVIDIA 42.86 vs 102.49 TFLOPS = 2.39× behind, kill-criterion FAIL,
   same-machine/same-ICD/same-shape, kernel-only (no GEMV-vs-GEMM flattering) + the AMD/Intel
   row from the trigger.

## Why AXIOM is behind on NVIDIA (state this honestly in the RFC)

It is architectural, not a tuning knob (M3.7 + M3.8 are merged honest-negatives proving the
occupancy ceiling): llama.cpp's `mul_mm` uses large warptiles (BM/BN up to 128, WMITER/WNITER
register blocking, BK-stepped staging, multi-warp workgroups, split-K, aligned vectorized loads)
and **dequantizes A inline during shared staging**; AXIOM is a single-subgroup `@workgroup(32,1,1)`
2×2-tile kernel paying a dequant-front-end tax. The 2.39× is not quickly closable on NVIDIA. The
thesis bet is that the *portable single source* wins precisely where the hand-tuned GLSL backend is
weakest — AMD/Intel — which is the trigger above.

## Recommendation

PREPARE-AND-HOLD (this document). Engineer the `.axc` rewrite + host patch only when AMD/Intel
hardware is in hand and shows the trigger condition; submit the RFC only if it does.
