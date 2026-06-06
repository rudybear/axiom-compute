# M3.5 — llama.cpp Vulkan Q4_K_M A/B, FUSED kernel, SAME-SHAPE (NVIDIA RTX PRO 6000)

- llama.cpp: tag `b9542` sha `6b80c74f285390368b3c99c5e750f19e9b096e98`
- device (both): `NVIDIA RTX PRO 6000 Blackwell Workstation Edition` | ICD `/usr/share/vulkan/icd.d/nvidia_icd.json` | device_match byte-identical: True
- K contraction (both): 14336

| metric | AXIOM (fused Q4_K_M RB coopmat GEMM) | llama.cpp (Q4_K MUL_MAT n=512) |
|---|---|---|
| shape (m,n,k) | (4096,512,14336) | (4096,512,14336) |
| TFLOPS (GpuTimestamp MIN, kernel-only) | 11.275 | 101.48 |
| max-rel-diff vs f16-accum ref | 29.069 | (ggml is the ref) |

> **NUMERICALLY INVALID at the A/B shape** (max_rel_diff=29.069 > frozen 0.001). fast-but-WRONG at inference K (f16 accumulator); correct only at K<=256; usable competitive kernel needs an f32-accumulator coopmat shape (M3.5b). At k=14336 the fused kernel's max_rel_diff=29.07 vs the f16-accumulator ggml reference — ~29069x over the frozen 0.001. The f16 coopmat accumulator cannot hold an inference-scale K sum; the output is GARBAGE at this K. Correctness holds ONLY at small K (AT-1770 K=256 max_rel_diff=8.3e-4 PASS; AT-1771 K=512 max_rel_diff=3.6e-3 EXCEEDS 1e-3, gate capped at K=256). The throughput ratio is therefore FAST-BUT-WRONG at inference K — NOT a usable-kernel win. A correct large-K fused kernel needs an f32-accumulator coopmat shape (M3.5b).

- **SAME-SHAPE headline ratio (AXIOM fused GEMM / llama same-shape, basis `same_shape_gemm m=4096,n=512,k=14336`): 0.11110** (llama 9.00x faster, on a NUMERICALLY INVALID AXIOM result — throughput-only, NOT a usable-kernel win)
- Cross-shape CONTEXT (NOT the kill criterion): AXIOM fused GEMM 11.275 TFLOPS vs llama n=1 GEMV 7.42 TFLOPS (ratio 1.520) — labeled cross-shape, never the headline.
- FLOP-consistency: ok=True (recomputed 101.476 vs reported 101.480 TFLOPS)
- **Kill-criterion (DESIGN §5, within 15% on NVIDIA): FAIL**
- Qualifier: M3.5 SAME-SHAPE fused GEMM vs llama's same-shape Q4_K MUL_MAT (101.48 TFLOPS). AXIOM measured 11.27 TFLOPS = ~9x BEHIND on throughput — a MASSIVE improvement from M3.4's ~87,000x cross-shape matvec. BUT this is NOT a usable-kernel win: AXIOM's output is NUMERICALLY INVALID at this k=14336 shape (max_rel_diff=29.07, see numerical_validity) because the f16 coopmat accumulator overflows precision at inference-scale K; correct only at K<=256. So the throughput gap closed dramatically but AXIOM is fast-but-WRONG at inference K (needs an f32-accumulator coopmat, M3.5b). DESIGN §5 kill-criterion is 'within 15% on ANY vendor'; this does NOT fire (AXIOM behind AND numerically invalid). AMD/Intel pending hardware (EB.1).
- Fairness caveat: Same machine, same ICD, kernel-only-vs-kernel-only, SAME SHAPE (m=4096,n=512,k=14336 both sides), FLOP convention identical (2*m*n*k matmul MACs, dequant excluded). CRITICAL CORRECTNESS CAVEAT: AXIOM's fused output is NUMERICALLY INVALID at this A/B shape (k=14336, max_rel_diff=29.07 — see numerical_validity) because the f16 coopmat accumulator overflows precision at inference-scale K; it is correct ONLY at K<=256. So the ~9x throughput gap is fast-but-WRONG at inference K, NOT AXIOM being that far behind a usable kernel of its own. Ratio reported AS MEASURED for a numerically-INVALID result at the A/B K. The n=1 GEMV (llama 7.42 TFLOPS) is cross-shape CONTEXT ONLY — never the headline (the SAME llama kernel runs 13.7x faster at n=512), so headlining AXIOM-GEMM vs llama-n1-GEMV would flatter AXIOM ~13x; that is forbidden (CRITICAL-1).
- Gap-closing path: M3.5 closed the M3.4 ~87,000x cross-shape THROUGHPUT gap to a single-digit same-shape ratio. But the BLOCKING issue is NOT throughput — it is CORRECTNESS: the f16 coopmat accumulator is numerically invalid at inference-scale K (max_rel_diff=29.07 at k=14336). The next milestone (M3.5b) MUST add an f32-accumulator coopmat shape so the fused kernel is CORRECT at large K; only then is the throughput number a usable-kernel comparison. Secondary: scale-caching across the 16 tile_k K-blocks of a superblock (OQ-3) to cut ALU-bound dequant overhead.

Reproduce: `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json AXC_ENABLE_GPU_BENCHES=1 scripts/m34_llamacpp_ab.sh --fused`
