# AXIOM-Compute

**AI-first compute language for SPIR-V.** Sister project to [AXIOM](https://github.com/rudybear/axiom) (CPU via LLVM IR), targeting GPU/GPGPU via SPIR-V → Vulkan compute / OpenCL / WebGPU.

## Thesis

LLMs cannot reliably optimize GPU kernels written in CUDA/HLSL/GLSL because those surfaces were designed for humans, with optimization intent implicit. AXIOM-Compute encodes GPU-specific intent as first-class annotations -- `@coalesced`, `@occupancy`, `@cooperative_matrix`, `@subgroup_uniform`, `@divergence_free`, pre/postconditions with FP tolerance -- that an agent reads, rewrites, and iteratively optimizes under a verifier that mechanically rejects correctness-breaking rewrites.

## Beachhead

**llama.cpp's Vulkan backend.** Measured performance gaps across AMD, Intel Arc, and Apple Silicon are documented in ggml-org/llama.cpp issues #16230, #17273, #21517 and ollama/ollama #15601. This is the first concrete target: match hand-tuned llama.cpp Vulkan within 5% on NVIDIA, beat it by ≥25% on AMD APU / Intel Arc using a single annotated source.

## Non-goals

- **Not a shading language.** Graphics pipelines are out of scope.
- **Not trying to beat NVCC on CUDA.** Proprietary vendor driver compilers are not a target to out-optimize.
- **Not replacing MLIR / IREE / Triton / Slang.** Occupies the unfilled niche of LLM-first annotation layer above SPIR-V.

## Status

Milestone M0 (bootstrap) in progress. See `DESIGN.md` and `docs/MASTER_TASK_LIST.md`.

## Related

- [axiom](https://github.com/rudybear/axiom) -- CPU compiler (parent project)
- [axiom-ports](https://github.com/rudybear/axiom-ports) -- real-world C ports via axiom
- [axiom-engine](https://github.com/rudybear/axiom-engine) -- game engine + Q2 port

## License

MIT OR Apache-2.0.
