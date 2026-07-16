# REPRODUCE.md — M4.2a ggml-ABI-matched Q4_K_M coopmat-KHR package

This directory is the **committable half** of the held M4.2 upstream RFC
(`UPSTREAM_PR_PLAN.md`, status **PREPARED — HOLD**). It contains everything that can be prepared
WITHOUT AMD/Intel hardware: the ggml-ABI-matched kernel, its generated SPIR-V, a hand-written host
patch against the pinned llama.cpp SHA, and this reproduction guide. **No RFC is opened by this
package** — see `UPSTREAM_PR_PLAN.md`'s trigger + the preconditions checklist appended there.

## Pinned source

- Repository: `https://github.com/ggml-org/llama.cpp`
- Tag: `b9542`
- SHA: `6b80c74f285390368b3c99c5e750f19e9b096e98` (the FULL 40-char SHA — the tag is re-pointable,
  the SHA is not)
- Local clone convention (gitignored, never committed): `vendor/llama.cpp`, cloned/pinned by
  `scripts/m34_llamacpp_ab.sh` (or manually: `git clone ... vendor/llama.cpp && git -C
  vendor/llama.cpp checkout 6b80c74f285390368b3c99c5e750f19e9b096e98`).

## Build flag

`GGML_VULKAN_AXIOM_Q4K` — opt-in, **never default**. Every hunk in `ggml-vulkan-axiom-q4k.patch`
is `#ifdef GGML_VULKAN_AXIOM_Q4K`-guarded; an unpatched or flag-off build is byte-identical to
stock llama.cpp.

## The six ABI deltas (kernel: `examples/q4km_matmul_rb_coopmat_f32acc_cached_ggml.axc`)

VERBATIM from the M3.6 production leader (`examples/q4km_matmul_rb_coopmat_f32acc_cached.axc`)
except:

1. `n_blocks_per_row` DERIVED as `stride_a / 256u32` (ggml's `stride_a` is ELEMENTS, not bytes —
   `ggml_vk_matmul(ne01,ne11,ne10,ne10,ne10,stride_d,…)`, `ggml-vulkan.cpp:8403-8409`).
2. Consumes ggml's 17-field `vk_mat_mat_push_constants` (`ggml-vulkan.cpp:1036-1044`), reading
   only the leading six (M,N,K,stride_a,stride_b,stride_d @ offsets 0,4,8,12,16,20).
3. B (activations) retyped `readonly_buffer[f32]` (ggml's `matmul_q4_k_f32` B_TYPE=float,
   `vulkan-shaders-gen.cpp:582`), read TRANSPOSED (`x[n*stride_b+k]`, ggml's `[N,K]` row-major),
   staged f32→f16 via the existing M3.5 `f32_to_f16` builtin.
4. D stored COLUMN-MAJOR with `stride_d` (`mul_mm.comp:404`, `ggml-vulkan.cpp:8175`) via the NEW
   `coopmat_store_col` codegen builtin.
5. Grid axis order SWAPPED: `gl_WorkGroupID.x`→M-row, `.y`→N-col (`mul_mm.comp:141,166` — the
   OPPOSITE of the M3.6 leader). Host adapter registers `wg_denoms={32,32,1}`.
6. (Implied by #4) — the codegen feature itself: `coopmat_store_col` adds ONLY the
   `ColumnMajorKHR` layout operand to the existing `SPV_KHR_cooperative_matrix` store — no new
   capability/extension.

Full design + every vendor citation: `.pipeline/milestones/M4.2a-ggml-abi.md`.

## How `matmul_q4_k_f32_cm1_axiom.spv` was generated

```
cargo run -p axc-driver --bin axc -- compile \
  examples/q4km_matmul_rb_coopmat_f32acc_cached_ggml.axc \
  -o upstream/matmul_q4_k_f32_cm1_axiom.spv \
  --strategy-value rb_m=2 --strategy-value rb_n=2 --strategy-value tile_k=16 \
  --strategy-value a_block_size=512 --strategy-value b_block_size=512
```

These are the SAME `@strategy` assignments as the M3.6 leader (RB 2×2, tile_k=16,
a_block_size/b_block_size pinned at 512 — see the kernel's `@strategy` line). AT-2997
(`compile_shared_examples.rs`) asserts the committed `.spv` is byte-reproducible from this exact
command against the pinned source. `spirv-val` (Vulkan 1.1 target env, via the `spirv-tools`
crate) is clean — verified in the coder pass (AT-2985/2987/2997) and mirrored by every dispatch
test that compiles this source.

## How the byte array was produced

`upstream/matmul_q4_k_f32_cm1_axiom_data.hpp.fragment` mirrors llama.cpp's own
`vulkan-shaders-gen.cpp:write_output_files()` byte-array emission format exactly (`0x`-hex bytes,
12 per line, `const uint64_t <name>_len` + `const unsigned char <name>_data[<len>]`). To embed:
paste the fragment's body into the generated `ggml-vulkan-shaders.hpp`/`.cpp` pair, alongside the
existing `matmul_q4_k_f32_cm1_data[]` declaration (the REAL KHR-coopmat symbol — NOT the scalar
`matmul_q4_k_f32_data[]`, which is never selected on coopmat hardware; NOT `matmul_q4_k_f32_cm2_*`,
the separate NV coopmat2 path).

## Applying the patch

```
cd vendor/llama.cpp   # at the pinned SHA
git apply --check /path/to/upstream/ggml-vulkan-axiom-q4k.patch   # dry-run
git apply /path/to/upstream/ggml-vulkan-axiom-q4k.patch           # actually apply
```

The patch touches `ggml/src/ggml-vulkan/ggml-vulkan.cpp` in three hunks (two logical sites, per
the milestone spec §7.2): (1) a small `vk_device_struct` member declaration (the standalone
pipeline slot, bundled with the creation site), (2) pipeline CREATION inside the coopmat block
(`:4090-4211`), and (3) dispatch SELECTION — overriding the RESULT of
`ggml_vk_guess_matmul_pipeline` at its call site (`:8239`) under the 13-condition fail-closed
guard (§2.9 of the milestone spec). `git apply --check` is verified against the pinned SHA in the
coder pass (AT-2998, `crates/axc-driver/tests/m42a_patch_applies.rs`) — **read-only, never applied
persistently** in this milestone.

**KNOWN LIMITATION (disclosed, not yet resolved):** the override hunk's guard condition #8
(`split_k==1` + single-batch) calls `ggml_vk_guess_split_k(...)` inline (mirroring the resolution
at `:8252`) since the `split_k` local is not yet declared at the `:8239` interception point. This
is architecturally sound (the function signature is confirmed correct) but has **NOT been
compiled** against the real headers in this milestone — a handful of other local-variable names in
the 13-condition guard (all confirmed present in the pinned source during this coder pass, but not
exhaustively compile-checked) may need small adjustments when the patch is actually built. This is
exactly why the executed real-ggml ABI smoke (below) is a HARD precondition of the RFC-open
trigger, not a formality.

## Rebuild llama.cpp with the patch (POST-HARDWARE — not run in this milestone)

```
cmake -B build -DGGML_VULKAN=ON -DGGML_VULKAN_AXIOM_Q4K=ON ...   # flag threading is a
                                                                    # CMakeLists.txt addition,
                                                                    # not included in this patch
cmake --build build --target llama-cli -j
```

(The patch does not yet include a CMake `option()` wiring `-DGGML_VULKAN_AXIOM_Q4K` to the
preprocessor define — a small, mechanical addition left for the actual build pass, per the
KNOWN LIMITATION above.)

## Running the NVIDIA A/B (this milestone, MEASURED)

```
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json AXC_ENABLE_GPU_BENCHES=1 \
  scripts/m34_llamacpp_ab.sh --fused-f32acc-cached-ggml
```

Runs `resident_q4km_matmul_rb_f32acc_cached_ggml` (BOTH the ggml variant and the M3.6 leader, same
sizes, same device), writes `.pipeline/benchmarks/m34/ab_results_fused_f32acc_cached_ggml.json`.

### The honest NVIDIA numbers (RTX PRO 6000 Blackwell, this coder pass)

**The standing M3.6-vs-llama gap (UNCHANGED by M4.2a — M3.6 remains the production leader):**

| metric | AXIOM (M3.6 leader) | llama.cpp (Q4_K MUL_MAT n=512) |
|---|---|---|
| shape (m,n,k) | 4096 × 512 × 14336 | 4096 × 512 × 14336 |
| TFLOPS (GpuTimestamp MIN) | **42.86** | **102.49** |
| ratio | 2.39× behind | — |

**The NEW ggml-variant self-A/B (this milestone, NO gate — perf sanity only):**

| size (M=N=K) | M3.6 leader TFLOPS | ggml variant TFLOPS | ggml/leader |
|---|---|---|---|
| 256 | 1.57–1.59 | 1.36–1.37 | 0.86–0.87× |
| 512 | 6.77–6.83 | 5.95–5.97 | 0.87–0.88× |
| 768 | 14.20–14.21 | 11.31–11.63 | 0.80–0.82× |
| 1024 | 23.80–23.87 | 17.67–17.68 | 0.74× |
| 4096×512×14336 (A/B, one-shot MIN-of-10 GpuTimestamp measurement) | **44.06** | **22.26** | **0.505×** |

All ggml-variant dispatches are combined-condition-aware ≤ FROZEN 1e-3 VALID (bit-identical to the
leader's combined values at every size — expected, since the accumulator VALUES are bit-identical
to the leader; §5.2). The ggml variant is honestly **0.50×–0.88× the leader's throughput**,
WORST at the A/B (inference) shape — the extra per-B-element f32→f16 staging convert cost scales
with `N*K` (7.3M converts at the A/B shape vs ≤1M at the largest cube), and the B buffer itself is
4× larger (f32 vs f16) — MEASURED, not modeled. **This is a perf-sanity number, not a gate** (M3.13
concluded the NVIDIA throughput campaign — see DESIGN.md §3.1.41); it also does not reflect ggml's
real dispatch, since the fixed-32×32-tile override REPLACES whichever of ggml's six
l/m/s/aligned variants would have been chosen for a given shape (§2.9 spec caveat, NOT gated).
Run via `scripts/m34_llamacpp_ab.sh --fused-f32acc-cached-ggml` (writes the full JSON with the
criterion-measured numbers, which may differ slightly run-to-run from the one-shot numbers above
due to normal GPU-clock/thermal variance — both measurement paths use the same MIN-of-10
GpuTimestamp discipline).

## The executed real-ggml ABI smoke — a HARD precondition (NOT done in this milestone)

Building the patched llama.cpp and running actual inference (bit-close output vs stock ggml, same
prompt, same model, on the pinned SHA) is the ONLY test that exercises the REAL bound
buffers/dispatch/f32 `src1`/column-major D. It is explicitly **out of scope for M4.2a** (see the
milestone spec §7.5) and is a checklist item in `UPSTREAM_PR_PLAN.md`'s appended
"RFC-open trigger preconditions" section — it must pass BEFORE the RFC is opened, independent of
the AMD/Intel A/B trigger.

## The AMD/Intel trigger A/B — the actual RFC-open condition (hardware-gated, EB.1)

Run the SAME single, unchanged `.axc` source (this milestone's ggml variant, once the real-ggml
smoke above has passed) on an AMD APU (RDNA3+) or Intel Arc GPU, and produce a same-machine,
same-ICD, kernel-only A/B vs llama.cpp's Vulkan Q4_K MUL_MAT at the inference shape (m=4096,
n=512, k=14336). See `UPSTREAM_PR_PLAN.md`'s "THE TRIGGER" section (unchanged by this milestone).

## Coopmat2 (`_cm2`) device scope caveat

On a coopmat2-capable NVIDIA device, ggml dispatches the `_cm2` pipeline, NOT `_cm1` — this
substitution is INACTIVE there (guard condition #10: `coopmat_support && !coopmat2`). This is a
documented coverage boundary, not a defect: the substitution covers coopmat-KHR devices without
coopmat2 support.
