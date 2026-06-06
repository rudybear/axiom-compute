#!/usr/bin/env bash
# M3.4 — llama.cpp Vulkan Q4_K_M A/B harness (the DESIGN §5 kill-criterion run).
#
# Same-machine, same-ICD, fence-synchronized PERF comparison of AXIOM-Compute's FROZEN
# M2.6 single-row Q4_K_M dequant+matvec vs llama.cpp's Vulkan backend Q4_K_M MUL_MAT
# (test-backend-ops perf), producing an HONEST ratio + the kill-criterion verdict.
#
# HONESTY: expected outcome is AXIOM losing by a large margin on NVIDIA (FAIL). That is the
# documented baseline. The script reports whatever the numbers are. No cherry-picking, no
# zero-weights, no GEMM-substituted kill number.
#
# Usage:
#   VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json AXC_ENABLE_GPU_BENCHES=1 \
#     scripts/m34_llamacpp_ab.sh [--skip-build] [--skip-llama]
#
# Env:
#   LLAMACPP_DIR     (default vendor/llama.cpp)
#   LLAMACPP_COMMIT  (default: the LITERAL 40-char SHA pinned below — NOT a re-pointable tag)
#   AXC_M34_OUTDIR   (default .pipeline/benchmarks/m34)
#   SPIRV_HEADERS_PREFIX (default vendor/spirv-headers/install — needed if libvulkan-dev
#                         does not ship the SPIRV-Headers CMake config)
#
# Fail-closed: device-string / ICD byte-mismatch, missing n==1 Q4_K case, FLOP-consistency
# failure, or parse failure ⇒ kill_criterion_status=INCOMPLETE, non-zero exit, no ratio.

set -euo pipefail

# ── Pinned llama.cpp commit (WARNING-4: LITERAL 40-char SHA, frozen at clone time) ──────
# tag b9542 resolved to this SHA on 2026-06-06. A tag is re-pointable; only the SHA is
# reproducible.
LLAMACPP_TAG="b9542"
LLAMACPP_COMMIT="${LLAMACPP_COMMIT:-6b80c74f285390368b3c99c5e750f19e9b096e98}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMACPP_DIR="${LLAMACPP_DIR:-${REPO_ROOT}/vendor/llama.cpp}"
OUTDIR="${AXC_M34_OUTDIR:-${REPO_ROOT}/.pipeline/benchmarks/m34}"
SPIRV_HEADERS_PREFIX="${SPIRV_HEADERS_PREFIX:-${REPO_ROOT}/vendor/spirv-headers/install}"
ICD="${VK_DRIVER_FILES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"

SKIP_BUILD=0
SKIP_LLAMA=0
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        --skip-llama) SKIP_LLAMA=1 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

mkdir -p "$OUTDIR"
RAW="${OUTDIR}/llamacpp_raw.txt"
RESULTS_JSON="${OUTDIR}/ab_results.json"
RESULTS_MD="${OUTDIR}/ab_results.md"

export VK_DRIVER_FILES="$ICD"

echo "== M3.4 llama.cpp Vulkan Q4_K_M A/B =="
echo "  repo_root  : ${REPO_ROOT}"
echo "  llama dir  : ${LLAMACPP_DIR}"
echo "  llama tag  : ${LLAMACPP_TAG}"
echo "  llama sha  : ${LLAMACPP_COMMIT}"
echo "  ICD        : ${ICD}"
echo "  outdir     : ${OUTDIR}"

# ── 1. Clone + pin llama.cpp (gitignored) ───────────────────────────────────────────────
if [ ! -d "${LLAMACPP_DIR}/.git" ]; then
    echo "-- cloning ggml-org/llama.cpp into ${LLAMACPP_DIR}"
    git clone https://github.com/ggml-org/llama.cpp "${LLAMACPP_DIR}"
fi
echo "-- checking out pinned SHA ${LLAMACPP_COMMIT}"
git -C "${LLAMACPP_DIR}" fetch --tags --quiet || true
git -C "${LLAMACPP_DIR}" checkout --quiet "${LLAMACPP_COMMIT}"
ACTUAL_SHA="$(git -C "${LLAMACPP_DIR}" rev-parse HEAD)"
if [ "${ACTUAL_SHA}" != "${LLAMACPP_COMMIT}" ]; then
    echo "FATAL: checked-out SHA ${ACTUAL_SHA} != pinned ${LLAMACPP_COMMIT}" >&2
    exit 1
fi

# ── 2. Build test-backend-ops (Vulkan) ──────────────────────────────────────────────────
TBO="${LLAMACPP_DIR}/build/bin/test-backend-ops"
if [ "${SKIP_BUILD}" -eq 0 ] || [ ! -x "${TBO}" ]; then
    echo "-- building test-backend-ops (Vulkan)"
    CMAKE_EXTRA=()
    if [ -d "${SPIRV_HEADERS_PREFIX}" ]; then
        # libvulkan-dev on this box does not ship the SPIRV-Headers CMake config; point at
        # a locally-installed copy (vendor/spirv-headers/install) and inject its include dir
        # because ggml-vulkan does not link the SPIRV-Headers::SPIRV-Headers target.
        CMAKE_EXTRA+=("-DCMAKE_PREFIX_PATH=${SPIRV_HEADERS_PREFIX}")
        CMAKE_EXTRA+=("-DCMAKE_CXX_FLAGS=-I${SPIRV_HEADERS_PREFIX}/include")
    fi
    cmake -S "${LLAMACPP_DIR}" -B "${LLAMACPP_DIR}/build" \
        -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON \
        "${CMAKE_EXTRA[@]}"
    cmake --build "${LLAMACPP_DIR}/build" --target test-backend-ops -j"$(nproc)"
fi
if [ ! -x "${TBO}" ]; then
    echo "FATAL: ${TBO} not built. Install libvulkan-dev / glslc / SPIRV-Headers and retry." >&2
    exit 1
fi
echo "-- test-backend-ops: ${TBO}"

# ── 3. Capture device strings (fail-closed device-match, WARNING-5) ──────────────────────
AXIOM_DEVICE=""
LLAMA_DEVICE=""
VULKANINFO_DEVICE="$(VK_DRIVER_FILES="${ICD}" vulkaninfo 2>/dev/null \
    | grep -m1 'deviceName' | sed 's/.*= //' | tr -d '\r' || true)"
echo "-- vulkaninfo device: ${VULKANINFO_DEVICE}"

# ── 4. llama.cpp side: run perf MUL_MAT, parse the Q4_K n==1 case ────────────────────────
if [ "${SKIP_LLAMA}" -eq 0 ]; then
    echo "-- running llama.cpp test-backend-ops perf -o MUL_MAT"
    VK_DRIVER_FILES="${ICD}" "${TBO}" perf -o MUL_MAT > "${RAW}" 2>&1 || {
        echo "FATAL: test-backend-ops perf exited non-zero (see ${RAW})" >&2
        exit 1
    }
fi
if [ ! -s "${RAW}" ]; then
    echo "FATAL: llama.cpp raw output ${RAW} empty; cannot parse" >&2
    exit 1
fi

LLAMA_DEVICE="$(grep -m1 'Device description:' "${RAW}" | sed 's/.*Device description: //' | tr -d '\r' || true)"
echo "-- llama.cpp device: ${LLAMA_DEVICE}"

# Strip ANSI color codes for parsing.
CLEAN_RAW="$(sed 's/\x1b\[[0-9;]*m//g' "${RAW}")"

# Select the Q4_K MUL_MAT n==1 (GEMV) case. CRITICAL-2: if none, INCOMPLETE.
Q4K_N1_LINE="$(printf '%s\n' "${CLEAN_RAW}" \
    | grep 'MUL_MAT(type_a=q4_K' | grep ',n=1,' | head -1 || true)"

KILL_STATUS="FAIL"          # mechanical; expected FAIL on NVIDIA
KILL_REASON=""
INCOMPLETE=0

if [ -z "${Q4K_N1_LINE}" ]; then
    KILL_STATUS="INCOMPLETE"
    KILL_REASON="no type_a=q4_K MUL_MAT n==1 (GEMV) case found in test-backend-ops perf output (CRITICAL-2: GEMM substitution forbidden)"
    INCOMPLETE=1
    LLAMA_US="null"; LLAMA_TFLOPS="null"; LLAMA_M="null"; LLAMA_N="null"; LLAMA_K="null"; LLAMA_RUNS="null"
else
    echo "-- selected Q4_K n==1 line:"
    echo "   ${Q4K_N1_LINE}"
    # Parse m,n,k and us/run and TFLOPS.
    LLAMA_M="$(echo "${Q4K_N1_LINE}" | sed -n 's/.*[,(]m=\([0-9]*\).*/\1/p')"
    LLAMA_N="$(echo "${Q4K_N1_LINE}" | sed -n 's/.*,n=\([0-9]*\),.*/\1/p')"
    LLAMA_K="$(echo "${Q4K_N1_LINE}" | sed -n 's/.*,k=\([0-9]*\),.*/\1/p')"
    LLAMA_RUNS="$(echo "${Q4K_N1_LINE}" | sed -n 's/.*[:)] *\([0-9]*\) runs.*/\1/p')"
    LLAMA_US="$(echo "${Q4K_N1_LINE}" | sed -n 's/.* runs - *\([0-9.]*\) us\/run.*/\1/p')"
    LLAMA_TFLOPS="$(echo "${Q4K_N1_LINE}" | sed -n 's/.* - *\([0-9.]*\) TFLOPS.*/\1/p')"

    if [ -z "${LLAMA_M}" ] || [ -z "${LLAMA_K}" ] || [ -z "${LLAMA_US}" ] || [ -z "${LLAMA_TFLOPS}" ]; then
        KILL_STATUS="INCOMPLETE"
        KILL_REASON="failed to parse m/k/us/TFLOPS from the selected Q4_K line (format drift)"
        INCOMPLETE=1
    fi
fi

# ── 5. AXIOM side: run the dispatch_q4km_ab bench, parse the AXC_Q4KM_AB line ─────────────
echo "-- running AXIOM dispatch_q4km_ab bench"
BENCH_OUT="${OUTDIR}/axiom_bench_raw.txt"
AXC_ENABLE_GPU_BENCHES=1 VK_DRIVER_FILES="${ICD}" \
    cargo bench --manifest-path "${REPO_ROOT}/Cargo.toml" -p axc-driver --bench dispatch_q4km_ab \
    > "${BENCH_OUT}" 2>&1 || {
    echo "FATAL: AXIOM bench exited non-zero (see ${BENCH_OUT})" >&2
    exit 1
}

AXC_LINE="$(grep -m1 '^AXC_Q4KM_AB ' "${BENCH_OUT}" || true)"
if [ -z "${AXC_LINE}" ]; then
    echo "FATAL: no AXC_Q4KM_AB line in bench output (GPU benches may be disabled/skipped)" >&2
    exit 1
fi
echo "-- AXIOM line: ${AXC_LINE}"

axc_field() { echo "${AXC_LINE}" | sed -n "s/.* $1=\([^ ]*\).*/\1/p"; }
AXC_NS_MIN="$(axc_field kernel_ns_min)"
AXC_NS_MEAN="$(axc_field kernel_ns_mean)"
AXC_NS_MEDIAN="$(axc_field kernel_ns_median)"
AXC_SUSTAINED_NS="$(axc_field sustained_ns)"
AXC_TIMING_SRC="$(axc_field timing_source)"
AXC_K="$(axc_field K)"
AXC_FLOPS="$(axc_field flops)"
AXIOM_DEVICE="$(echo "${AXC_LINE}" | sed -n 's/.* device=\(.*\)$/\1/p')"
echo "-- AXIOM device: ${AXIOM_DEVICE}"

# ── 6. Device-match FAIL-CLOSED (WARNING-5): byte-identical device strings ────────────────
DEVICE_MATCH="false"
if [ -n "${AXIOM_DEVICE}" ] && [ -n "${LLAMA_DEVICE}" ] && [ "${AXIOM_DEVICE}" = "${LLAMA_DEVICE}" ]; then
    DEVICE_MATCH="true"
fi

# ── 7. Compute the like-for-like A/B ─────────────────────────────────────────────────────
# Boundary: llama.cpp = CPU-wall, batched-amortized MEAN, sustained (read from
# test-backend-ops.cpp at the SHA, see llamacpp_timer_boundary in the JSON). AXIOM's
# matched boundary is its sustained CPU-wall number; the GpuTimestamp MIN is the cleaner
# kernel-only framing reported alongside for disclosure.
#
# The FAIR metric is work-normalized THROUGHPUT (TFLOPS), because llama.cpp computes 4096
# output rows in its 15.74 us while AXIOM computes 1 row — a raw us comparison is
# apples-to-oranges in AXIOM's favour (AXIOM does 1/4096th the work). TFLOPS normalizes.
PY="$(command -v python3 || true)"
if [ -z "${PY}" ]; then echo "FATAL: python3 required for arithmetic + JSON emit" >&2; exit 1; fi


# Pass all values to Python via environment (robust: no source-substitution of strings).
export M34_INCOMPLETE="${INCOMPLETE}"
export M34_AXC_NS_MIN="${AXC_NS_MIN}"
export M34_AXC_NS_MEAN="${AXC_NS_MEAN}"
export M34_AXC_NS_MEDIAN="${AXC_NS_MEDIAN}"
export M34_AXC_SUSTAINED_NS="${AXC_SUSTAINED_NS}"
export M34_AXC_K="${AXC_K}"
export M34_AXC_FLOPS="${AXC_FLOPS}"
export M34_AXC_TIMING="${AXC_TIMING_SRC}"
export M34_AXC_DEVICE="${AXIOM_DEVICE}"
export M34_LLAMA_DEVICE="${LLAMA_DEVICE}"
export M34_DEVICE_MATCH="${DEVICE_MATCH}"
export M34_KILL_STATUS="${KILL_STATUS}"
export M34_KILL_REASON="${KILL_REASON}"
export M34_LLAMA_US="${LLAMA_US}"
export M34_LLAMA_TFLOPS="${LLAMA_TFLOPS}"
export M34_LLAMA_M="${LLAMA_M}"
export M34_LLAMA_N="${LLAMA_N}"
export M34_LLAMA_K="${LLAMA_K}"
export M34_LLAMA_RUNS="${LLAMA_RUNS}"
export M34_Q4K_LINE="${Q4K_N1_LINE}"
export M34_VULKANINFO_DEVICE="${VULKANINFO_DEVICE}"
export M34_LLAMACPP_TAG="${LLAMACPP_TAG}"
export M34_LLAMACPP_COMMIT="${LLAMACPP_COMMIT}"
export M34_ICD="${ICD}"
export M34_RESULTS_JSON="${RESULTS_JSON}"
export M34_RESULTS_MD="${RESULTS_MD}"

"${PY}" "${REPO_ROOT}/scripts/m34_emit.py"
PY_EXIT=$?

echo ""
echo "-- wrote ${RESULTS_JSON}"
echo "-- wrote ${RESULTS_MD}"
exit ${PY_EXIT}
