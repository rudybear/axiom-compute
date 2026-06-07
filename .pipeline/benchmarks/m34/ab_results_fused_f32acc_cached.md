# M3.5b — llama.cpp Vulkan Q4_K_M A/B, f32-ACCUMULATOR FUSED kernel, SAME-SHAPE (NVIDIA RTX PRO 6000)

- llama.cpp: tag `b9542` sha `6b80c74f285390368b3c99c5e750f19e9b096e98`
- device (both): `NVIDIA RTX PRO 6000 Blackwell Workstation Edition` | ICD `/usr/share/vulkan/icd.d/nvidia_icd.json` | device_match byte-identical: True
- K contraction (both): 14336

| metric | AXIOM (f32-accumulator fused Q4_K_M RB coopmat GEMM) | llama.cpp (Q4_K MUL_MAT n=512) |
|---|---|---|
| shape (m,n,k) | (4096,512,14336) | (4096,512,14336) |
| TFLOPS (GpuTimestamp MIN, kernel-only) | 42.862 | 102.49 |
| max-rel-diff vs f32-accum ref (combined, condition-aware — the gate) | 0.000 | (ggml is the ref) |
| raw forward-error vs f32-accum ref (reporting only; ~1e-2 on cancellation outputs, identical-in-kind to llama.cpp's HMMA) | 2.098 | (ggml is the ref) |

- **SAME-SHAPE headline ratio (AXIOM fused GEMM / llama same-shape, basis `same_shape_gemm m=4096,n=512,k=14336`): 0.41821** (llama 2.39x faster)
- Cross-shape CONTEXT (NOT the kill criterion): AXIOM fused GEMM 42.862 TFLOPS vs llama n=1 GEMV 7.38 TFLOPS (ratio 5.808) — labeled cross-shape, never the headline.
- FLOP-consistency: ok=True (recomputed 102.491 vs reported 102.490 TFLOPS)
- **Kill-criterion (DESIGN §5, within 15% on NVIDIA): FAIL**
- Qualifier: M3.5 SAME-SHAPE fused GEMM vs llama's same-shape Q4_K MUL_MAT (102.49 TFLOPS). AXIOM measured 42.86 TFLOPS = ~2x BEHIND on throughput — a MASSIVE improvement from M3.4's ~87,000x cross-shape matvec. DESIGN §5 kill-criterion is 'within 15% on ANY vendor'; this does NOT fire (AXIOM behind). AMD/Intel pending hardware (EB.1).
- Fairness caveat: Same machine, same ICD, kernel-only-vs-kernel-only, SAME SHAPE (m=4096,n=512,k=14336 both sides), FLOP convention identical (2*m*n*k matmul MACs, dequant excluded). AXIOM's fused kernel is numerically VALID at this A/B shape (max_rel_diff=2.05e-06 <= frozen 1e-3); the ratio is a usable-kernel comparison. Ratio reported AS MEASURED for a numerically-VALID result at the A/B K. The n=1 GEMV (llama 7.38 TFLOPS) is cross-shape CONTEXT ONLY — never the headline (the SAME llama kernel runs 13.7x faster at n=512), so headlining AXIOM-GEMM vs llama-n1-GEMV would flatter AXIOM ~13x; that is forbidden (CRITICAL-1).
- Gap-closing path: M3.5 closed the M3.4 ~87,000x cross-shape THROUGHPUT gap to a single-digit same-shape ratio. Remaining gap (if ALU-bound): scale-caching across the 16 tile_k K-blocks of a superblock (OQ-3 -> M3.5b) to cut ALU-bound dequant overhead.

Reproduce: `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json AXC_ENABLE_GPU_BENCHES=1 scripts/m34_llamacpp_ab.sh --fused-f32acc`
