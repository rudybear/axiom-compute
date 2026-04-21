# AXIOM-Compute Benchmark Harness

This document describes the Criterion-based benchmark harness for AXIOM-Compute.
See also `DESIGN.md §3.1.7` for the design-level context.

---

## Measured results

Baselines live at `.pipeline/benchmarks/baselines.json`. Run
`AXC_BLESS_BASELINES=1 cargo bench -p axc-driver` on your own machine to
regenerate for your hardware.

### NVIDIA RTX PRO 6000 Blackwell Workstation

Intel i9-14900KF host, driver 580.126.09 / CUDA 13.0, 96 GB VRAM.

| Bench | Size | Median | Notes |
|---|---|---|---|
| `compile_saxpy` | — | **11.8 μs** | Source → SPIR-V codegen time |
| `compile_vector_add` | — | **9.1 μs** | |
| `cpu_saxpy` | 1 K / 1 M | 265 ns / 721 μs | Rust reference |
| `cpu_vector_add` | 1 K / 1 M | 63 ns / 211 μs | Rust reference |
| `dispatch_saxpy` | 1 K | **1.22 ms** | One-shot: includes pipeline compile + staging |
| `dispatch_saxpy` | 1 M | **23.0 ms** | Staging-bound |
| `dispatch_vector_add` | 1 K | 2.58 ms | |
| `dispatch_vector_add` | 1 M | 52.1 ms | |
| `dispatch_handle_saxpy_1m` (amortized) | 1 M | **22.2 ms** | Pipeline cache reused |
| `dispatch_gpu_q4_0_128` | 128 blocks (4 K elem) | **2.56 ms** | Q4_0 dequant + matvec |
| `dispatch_gpu_q4_0_1024` | 1024 blocks (32 K elem) | **3.26 ms** | |
| `dispatch_gpu_q4km_128` | 128 SB (32 K elem) | **3.38 ms** | Q4_K_M — llama.cpp beachhead |
| `dispatch_gpu_q4km_512` | 512 SB (131 K elem) | **8.84 ms** | Bandwidth-bound |

All numbers include the full host-round-trip: `memcpy` → staging → device-local copy → compute → device-local → staging → memcpy + fence wait.

**Correctness**: every GPU kernel produces bit-exact output vs CPU reference within its declared FP tolerance. `AT-1331_gpu_dispatch_nvidia_matches_cpu_reference_within_1e_3` is green for Q4_K_M.

**Ceiling at 1M elements**: ~23 ms is ~100× off theoretical PCIe peak — the staging-buffer copy path dominates. Pinned memory + concurrent transfer (deferred to M3) closes this gap.

**Pipeline cache impact**: `dispatch_saxpy_1m` one-shot (23.0 ms) → amortized (22.2 ms) saves ~800 μs per call on NVIDIA; expected to be much larger on AMD/Intel where shader compile is slower.

### Lavapipe (software Vulkan, CI)

Not a GPU perf signal — validates dispatch plumbing.

| Bench | Lavapipe median |
|---|---|
| `dispatch_saxpy_1024` | ~290 μs |
| `dispatch_saxpy_1m` | ~4.2 ms |
| `dispatch_q4km_128` | ~1.3 ms |

---

## Running benchmarks

```sh
# Run all bench groups (compile_pipeline, cpu_reference, dispatch_gpu, postprocess).
cargo bench -p axc-driver

# Run a single group.
cargo bench --bench compile       -p axc-driver   # source → SPIR-V timing
cargo bench --bench cpu_reference -p axc-driver   # CPU Rust equivalents
cargo bench --bench dispatch      -p axc-driver   # GPU dispatch timing (requires Vulkan)
```

GPU benches require `AXC_ENABLE_GPU_BENCHES=1` and a Vulkan ICD:

```sh
AXC_ENABLE_GPU_BENCHES=1 \
  VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
  cargo bench --bench dispatch -p axc-driver
```

---

## Interpreting output

Criterion reports each benchmark as three numbers inside brackets:

```
compile_saxpy  time:  [11.711 µs  11.754 µs  11.798 µs]
                        ^^^^^^^^   ^^^^^^^^   ^^^^^^^^
                        low CI     median     high CI
                        (2.5%)     (50th)     (97.5%)
```

- **low**: lower bound of the 95% confidence interval.
- **median**: the 50th percentile of all samples (primary comparison value).
- **high**: upper bound of the 95% confidence interval.

A small `[low, high]` spread indicates a stable measurement.  Wide spreads
signal OS noise, thermal throttling, or cold-cache effects.

---

## Metric-to-exit-gate mapping

| Group | Bench | Corresponding DESIGN.md criterion |
|---|---|---|
| `compile_pipeline` | `compile_saxpy`, `compile_vector_add` | Scaffolding only (M2.2); no direct exit gate in M2.2. Compile-time regressions tracked structurally. |
| `cpu_reference` | `cpu_saxpy_*`, `cpu_vector_add_*` | Baseline quality metric: GPU/CPU ratio (dispatch_ns / cpu_ns). Used to report `dispatch_gpu` performance relative to scalar CPU. |
| `dispatch_gpu` | `dispatch_saxpy_*`, `dispatch_vector_add_*` | M1 exit gate 3: 3-vendor execution correctness. M2.5: within 5% of llama.cpp Vulkan Q4\_K\_M on RTX 4090; beat by ≥25% on AMD APU / Intel Arc. |

M2.2 Lavapipe numbers are **structural baselines only** — they prove the harness
works end-to-end.  They are NOT performance targets.

---

## Blessing baselines

`baselines.json` records the machine-specific timing baseline for the regression
gate.  To update it after a performance improvement or hardware change:

```sh
# 1. Run all bench groups (generates Criterion output in target/criterion/).
cargo bench -p axc-driver --bench compile --bench cpu_reference --bench dispatch

# 2. Run postprocess with AXC_BLESS_BASELINES=1 to promote the candidate.
AXC_BLESS_BASELINES=1 cargo bench -p axc-driver --bench postprocess
```

This overwrites `.pipeline/benchmarks/baselines.json` (git-tracked).
Without `AXC_BLESS_BASELINES=1`, postprocess writes to
`target/axc-bench/candidate-baselines.json` (gitignored) and never touches
the committed baseline.

**CI must NEVER set `AXC_BLESS_BASELINES=1`** (AT-714b).  Baselines are always
promoted by a human developer after reviewing the Criterion output.

---

## Regression gate

The regression gate runs a lightweight 11-sample timing of the `cpu_reference`
group and compares the median against `baselines.json`.

```sh
# Enable the regression gate (disabled by default).
AXC_ENABLE_BENCH_REGRESSION=1 \
  cargo test --release -p axc-driver --test bench_regression -- --nocapture
```

**Threshold:** 15%.  If `current_median > baseline.median_ns × 1.15`, the test
fails with:

```
regression: bench `cpu_saxpy_1m` median 12345 ns exceeds baseline 801680 ns by 1440.0% (>15% threshold)
```

If `current_median < baseline.median_ns × 0.85`, the test prints a speedup note
but **passes** — improvements never fail the gate.

Only `cpu_reference` benches are gated (dispatch_gpu is too variable on
Lavapipe; compile_pipeline is OS-noise-dominated in the short-sample regime).

---

## CI matrix

| Job | Feature flag | `AXC_BLESS_BASELINES` | Expected result |
|---|---|---|---|
| `bench-regression` | none | unset | EXIT 0 (gate passes) |
| `bench-regression-fault-injection` | `bench_regression_fixture_slowdown` | unset | EXIT 1 (gate detects slowdown) |

Both jobs use `cargo test --release` to match the timing profile of
`baselines.json` (generated by `cargo bench` at the same optimization level).

---

## Known variance sources

- **Lavapipe software rendering jitter**: Lavapipe's dispatch time is dominated
  by CPU time rather than GPU memory bandwidth.  Measurements are highly
  variable under load or in VMs.  `dispatch_gpu` baselines on Lavapipe are
  structural only.

- **Cold cache vs warm cache**: The first dispatch after a cold start includes
  Vulkan pipeline creation overhead.  The `VulkanContext` is initialized once
  per bench run (outside the measured region), but shader-module creation is
  inside the timed region per spec.

- **Kernel-launch-overhead dominates small-N**: For `N=1024`, the GPU
  round-trip overhead (CB record + submit + fence wait + readback) dwarfs the
  actual compute work.  `dispatch_saxpy_1024` measures dispatch infrastructure
  latency, not compute throughput.

- **`cpu_saxpy_1024` variance**: At 288 ns, this bench is sensitive to cache
  state and scheduler latency.  The 11-sample median in the regression gate has
  >95% power to detect a true 15% mean shift (N=11 odd, σ≈8%) but may log
  "possible speedup" notes on faster machines than the baseline host.
