# M3.4 — llama.cpp Vulkan Q4_K_M A/B (NVIDIA RTX PRO 6000)

- llama.cpp: tag `b9542` sha `6b80c74f285390368b3c99c5e750f19e9b096e98`
- device (both): `NVIDIA RTX PRO 6000 Blackwell Workstation Edition` | ICD `/usr/share/vulkan/icd.d/nvidia_icd.json` | device_match byte-identical: True
- K contraction (both): 14336

| metric | AXIOM (M2.6 single-row matvec) | llama.cpp (Q4_K MUL_MAT n=1) |
|---|---|---|
| output rows | 1 | 4096 |
| us/dispatch (GpuTimestamp MIN) | 315.456 | (CPU-wall) |
| us/op (sustained CPU-wall) | 338.668 | 15.89 |
| TFLOPS (GpuTimestamp MIN) | 0.0001 | — |
| TFLOPS (sustained CPU-wall) | 0.0001 | 7.39 |

- **Headline ratio (AXIOM/llama, work-normalized TFLOPS, matched sustained boundary): 0.00001** (llama.cpp 87289.22x faster)
- FLOP-consistency: ok=True (recomputed 7.391 vs reported 7.390 TFLOPS)
- **Kill-criterion (DESIGN §5, within 15% on NVIDIA): FAIL**
- Qualifier: NVIDIA-only FAIL with the current FROZEN M2.6 single-row matvec is the documented baseline; DESIGN §5 kill-criterion is 'within 15% on ANY vendor', so this does NOT fire the project kill-criterion. AMD/Intel halves pending cross-vendor hardware (EB.1).
- Fairness caveat: Same machine, same ICD, kernel-only-vs-kernel-only, identical K=14336 contraction, FLOP convention identical (2*m*n*k matmul MACs, dequant excluded). BOTH kernels are CORRECT (AXIOM bit-exact vs ggml CPU ref; llama.cpp IS ggml). AXIOM is single-row matvec (1 workgroup) vs llama.cpp's tiled multi-row MUL_MAT (all SMs) — ~structurally 100x under-parallelized. Ratio reported AS MEASURED; honest expected FAIL.
- Gap-closing path: Fuse the Q4_K_M dequant front-end onto the M3.3c register-blocked coopmat matmul (dequant -> shared f16 tile -> coopmat mul_add; plain-f16 reached 31.2 TFLOPS = 24.96% of datasheet). Follow-up milestone.

Reproduce: `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json AXC_ENABLE_GPU_BENCHES=1 scripts/m34_llamacpp_ab.sh`
