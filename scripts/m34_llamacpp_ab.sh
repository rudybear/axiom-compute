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
#     scripts/m34_llamacpp_ab.sh [--skip-build] [--skip-llama] [--fused | --fused-f32acc | --fused-f32acc-cached | --fused-f32acc-db | --fused-f32acc-rb44 | --fused-f32acc-rb42 | --warptile | --pad | --sr | --ablate]
#
#   --pad (M3.10a): runs the BANK-PADDED resident benches — the plain-f32 padded RB
#                    (resident_matmul_rb_pad, gate >=1.15x the unpadded base at 768^3) AND the
#                    Q4_K_M padded leader (resident_q4km_matmul_rb_f32acc_cached_pad, gate >=1.15x
#                    M3.6's 42.86 = >=49.3 TFLOPS AND combined<=1e-3 at the A/B shape). PAD is
#                    swept in {1,2,4,8}; the winner is MEASURED. Short-circuits the llama Q4_K_M
#                    path and writes ab_results_pad.json. HONEST-NEGATIVE armed: a sub-gate result
#                    reports the shared-bytes + occupancy/bank diagnosis and NEVER loosens the gate.
#
#   --warptile (M3.9): runs the PLAIN-f32 multi-subgroup warptile (K=4) resident bench
#                    (resident_matmul_warptile) at 256/512/768/1024 + the A/B shape
#                    (4096x512x14336), measuring K=4 warptile vs the K=2 opponent vs
#                    single-subgroup RB. The PRIMARY M3.9 gate is PLAIN warptile vs PLAIN
#                    single-subgroup RB (>=1.3x at 768^3 OR the A/B shape) — measured by the
#                    bench itself, NOT against the Q4_K_M llama line (that fusion is the
#                    explicit M3.10 follow-on, out of M3.9 scope). This flag short-circuits
#                    the llama Q4_K_M path and writes ab_results_warptile.json. HONEST-NEGATIVE
#                    armed: if K=4 misses 1.3x the per-size TFLOPS + occupancy artifact are
#                    reported for the bottleneck diagnosis (-> M3.10 vectorized loads + bank pad).
#
# Modes:
#   (default)        M3.4 single-row matvec vs llama Q4_K n=1 GEMV (cross-shape baseline).
#   --fused          M3.5 f16-accumulator fused GEMM vs llama Q4_K n=512 same-shape (fast-but-
#                    WRONG at inference K — the f16 coopmat accumulator overflows precision).
#   --fused-f32acc   M3.5b f32-accumulator fused GEMM vs llama Q4_K n=512 same-shape. Now
#                    NUMERICALLY VALID at k=14336 under the condition-aware COMBINED metric
#                    (|gpu-ref|/max(|ref|,Σ|wₖxₖ|) <= frozen 1e-3 — the same backward-stable gate
#                    the GPU dispatch ATs use) — a REAL fast-AND-correct comparison (still behind
#                    on throughput). numerically_valid is driven by the MEASURED COMBINED metric,
#                    never hardcoded; the RAW forward error (~1e-2 on cancellation outputs, a
#                    metric artifact identical-in-kind to llama.cpp's HMMA) is recorded for
#                    transparency but does NOT gate validity.
#   --fused-f32acc-cached
#                    M3.6 DEQUANT-SCALE-CACHED f32-accumulator fused GEMM vs llama Q4_K n=512
#                    same-shape. Pure reassociation of --fused-f32acc (bit-identical output,
#                    AT-1803), so SAME combined-driven validity contract; the cache trades 15/16
#                    of the dequant-scale ALU recompute for +1 unconditional barrier/k_block — the
#                    throughput delta vs --fused-f32acc is the measured outcome (gap-narrowing,
#                    HONEST-NEGATIVE if the barrier/occupancy cost dominates). Runs the
#                    resident_q4km_matmul_rb_f32acc_cached bench (AXC_Q4KM_AB_F32ACC_CACHED line)
#                    and writes ab_results_fused_f32acc_cached.json.
#   --fused-f32acc-db
#                    M3.7 DOUBLE-BUFFERED (software-pipelined) scale-cached f32-accumulator fused
#                    GEMM vs llama Q4_K n=512 same-shape. Pure SCHEDULING of --fused-f32acc-cached
#                    (bit-identical output, AT-1903), so SAME combined-driven validity contract; the
#                    double-buffer overlaps next-tile global loads with current-tile HMMA compute.
#                    The throughput delta vs --fused-f32acc-cached is the measured outcome (the
#                    >=1.15x-over-42.86 = >=49.3 TFLOPS exit gate; HONEST-NEGATIVE if the doubled
#                    shared footprint regresses occupancy or the GEMM is compute-bound). Runs the
#                    resident_q4km_matmul_rb_f32acc_db bench (AXC_Q4KM_AB_F32ACC_DB line) and writes
#                    ab_results_fused_f32acc_db.json.
#   --fused-f32acc-rb44 / --fused-f32acc-rb42
#                    M3.8 LARGER-REGISTER-TILE (4x4 = 16 accumulators / 4x2 = 8 accumulators) scale-
#                    cached f32-accumulator fused GEMM vs llama Q4_K n=512 same-shape. SAME arithmetic
#                    as --fused-f32acc-cached (more output tiles per workgroup, same per-16x16-tile
#                    accumulation order — bit-identical anchor AT-2003), so SAME combined-driven
#                    validity contract; raising the register-tile size raises arithmetic intensity to
#                    amortize the dequant/staging front-end (the M3.7 occupancy/compute redirect). The
#                    throughput delta vs --fused-f32acc-cached is the measured outcome (the
#                    >=1.15x-over-42.86 = >=49.3 TFLOPS exit gate; HONEST-NEGATIVE if register pressure
#                    regresses occupancy). BOTH flags run the SAME
#                    resident_q4km_matmul_rb_f32acc_cached_bigrb bench (it emits both
#                    AXC_Q4KM_AB_F32ACC_BIGRB_4X4 and _4X2 lines); the flag selects which prefix to
#                    parse + the result JSON (ab_results_fused_f32acc_bigrb_4x4.json / _4x2.json).
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
FUSED=0       # M3.5: --fused switches to the SAME-SHAPE fused-kernel A/B (AT-1774).
FUSED_F32ACC=0 # M3.5b: --fused-f32acc switches to the f32-accumulator fused-kernel A/B (AT-1784).
FUSED_F32ACC_CACHED=0 # M3.6: --fused-f32acc-cached -> the dequant-scale-CACHED f32acc A/B (AT-1805).
FUSED_F32ACC_DB=0 # M3.7: --fused-f32acc-db -> the DOUBLE-BUFFERED scale-cached f32acc A/B (AT-1905).
# M3.8: --fused-f32acc-rb44 / --fused-f32acc-rb42 -> the LARGER-REGISTER-TILE A/B (AT-2005).
# Both run the SAME resident_q4km_matmul_rb_f32acc_cached_bigrb bench (it emits BOTH the
# AXC_Q4KM_AB_F32ACC_BIGRB_4X4 and _4X2 lines); the flag selects which prefix to parse + the
# result-JSON name. The honest path computes each variant's throughput vs M3.6's 42.86 and llama's
# 102.49 and applies the >=1.15x / >=49.3 TFLOPS gate (HONEST-NEGATIVE if register pressure regresses).
FUSED_F32ACC_RB44=0
FUSED_F32ACC_RB42=0
WARPTILE=0    # M3.9: --warptile -> PLAIN-f32 multi-subgroup warptile resident bench (no llama path).
PAD=0         # M3.10a: --pad -> the BANK-PADDED resident benches (plain-f32 + Q4_K_M, no llama path).
SR=0          # M3.11a: --sr -> the dequant-index STRENGTH-REDUCED resident bench vs M3.6 (no llama path).
ABLATE=0      # M3.12: --ablate -> the dequant front-end ABLATION DIAGNOSTIC profiling instruments (no llama path).
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        --skip-llama) SKIP_LLAMA=1 ;;
        --fused) FUSED=1 ;;
        --fused-f32acc) FUSED=1; FUSED_F32ACC=1 ;;  # f32-accum is a fused-mode variant
        # M3.6: cached is an f32acc-mode variant (same SAME-SHAPE A/B, same combined/raw schema).
        --fused-f32acc-cached) FUSED=1; FUSED_F32ACC=1; FUSED_F32ACC_CACHED=1 ;;
        # M3.7: double-buffered is an f32acc-mode variant (same SAME-SHAPE A/B, same combined/raw).
        --fused-f32acc-db) FUSED=1; FUSED_F32ACC=1; FUSED_F32ACC_DB=1 ;;
        # M3.8: larger-register-tile variants (same SAME-SHAPE A/B, same combined/raw schema).
        --fused-f32acc-rb44) FUSED=1; FUSED_F32ACC=1; FUSED_F32ACC_RB44=1 ;;
        --fused-f32acc-rb42) FUSED=1; FUSED_F32ACC=1; FUSED_F32ACC_RB42=1 ;;
        # M3.9: PLAIN-f32 warptile resident bench (short-circuits the Q4_K_M/llama path).
        --warptile) WARPTILE=1 ;;
        # M3.10a: BANK-PADDED resident benches (short-circuits the Q4_K_M/llama path).
        --pad) PAD=1 ;;
        # M3.11a: dequant-index STRENGTH-REDUCED resident bench vs M3.6 (short-circuits the llama path).
        --sr) SR=1 ;;
        # M3.12: dequant front-end ABLATION DIAGNOSTIC profiling instruments (short-circuits the llama path).
        --ablate) ABLATE=1 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

mkdir -p "$OUTDIR"

# ── M3.11a --sr: dequant-index STRENGTH-REDUCED resident bench vs the M3.6 leader. ──────────
# Runs resident_q4km_matmul_rb_f32acc_cached_sr (the SR kernel A/B + 768^3 + per-size table, timed
# AGAINST the M3.6 leader at the same shapes). PRIMARY gate >=1.15x M3.6's 42.86 (>=49.3 TFLOPS) at
# the A/B shape; ARMED honest-negative: ALU-count-down (AT-2600, CPU) but TFLOPS-flat => latency-
# hidden under HMMA (the LIKELY base case) => M3.11b NO-GO, M3.6 stays leader, gate NOT loosened.
# Emits AXC_Q4KM_AB_F32ACC_CACHED_SR + the per-size SR-vs-M3.6 ratio. Writes ab_results_sr.json.
# No llama path (the SR kernel is an A/B-vs-the-AXIOM-leader experiment; the llama A/B is M3.6's).
if [ "${SR}" -eq 1 ]; then
    SR_RESULTS="${OUTDIR}/ab_results_sr.json"
    SR_LOG="${OUTDIR}/sr_bench_q4km.txt"
    echo "== M3.11a dequant-index STRENGTH-REDUCED resident bench (SR vs M3.6 leader; no llama path) =="
    echo "  repo_root : ${REPO_ROOT}"
    echo "  ICD       : ${ICD}"
    echo "  outdir    : ${OUTDIR}"
    echo "  gate      : SR >=1.15x M3.6 42.86 (>=49.3 TFLOPS) @ A/B; ARMED honest-negative (latency-hidden flat = the LIKELY base case)"
    VK_DRIVER_FILES="${ICD}" AXC_ENABLE_GPU_BENCHES=1 \
        cargo bench --manifest-path "${REPO_ROOT}/Cargo.toml" -p axc-driver \
        --bench resident_q4km_matmul_rb_f32acc_cached_sr 2>&1 | tee "${SR_LOG}"
    SR_GATE_LINES="$(grep 'resident_sr: AT-2602\|resident_sr: .*\^3 SR=' "${SR_LOG}" || true)"
    AB_LINE="$(grep 'AXC_Q4KM_AB_F32ACC_CACHED_SR' "${SR_LOG}" | tail -1 || true)"
    {
        echo "{"
        echo "  \"milestone\": \"M3.11a-dequant-index-strength-reduction\","
        echo "  \"mode\": \"strength-reduced-vs-leader (SR kernel vs M3.6 cached, same A/B shapes)\","
        echo "  \"gate\": \"SR >=1.15x M3.6 42.86 (>=49.3 TFLOPS) AND combined<=1e-3 @ A/B; ARMED honest-negative\","
        echo "  \"icd\": \"${ICD}\","
        echo "  \"vulkaninfo_device\": \"$(VK_DRIVER_FILES="${ICD}" vulkaninfo 2>/dev/null | grep -m1 'deviceName' | sed 's/.*= //' | tr -d '\r' || true)\","
        echo "  \"note\": \"PURE-SOURCE induction-variable strength-reduction (no codegen change). The per-element integer-index ALU drop is pinned by AT-2600 (CPU). ARMED honest-negative: ALU-down-but-TFLOPS-flat => the dequant integer ALU is latency-hidden under the HMMA pipeline (the LIKELY base case) => M3.11b NO-GO; the 1.77x tax is NOT the integer index-decode. See sr_bench_q4km.txt.\","
        echo "  \"ab_line\": \"$(printf '%s' "${AB_LINE}" | sed 's/"/\\"/g')\","
        echo "  \"sr_gate_lines\": ["
        printf '%s\n' "${SR_GATE_LINES}" | sed '/^$/d' | sed 's/"/\\"/g' | awk 'BEGIN{first=1} {if(!first)printf ",\n"; printf "    \"%s\"", $0; first=0} END{if(!first)printf "\n"}'
        echo "  ]"
        echo "}"
    } > "${SR_RESULTS}"
    echo "-- wrote ${SR_RESULTS}"
    echo "-- full bench log: ${SR_LOG}"
    exit 0
fi

# ── M3.12 --ablate: dequant front-end ABLATION DIAGNOSTIC (PROFILING INSTRUMENTS, no llama path). ──
# Runs resident_q4km_matmul_rb_f32acc_cached_ablate: per-variant resident TFLOPS (V-structure-probe
# FIRST = the PRIMARY denominator, then V-passthrough, the M3.6 leader, and the plain-f32 core) at
# A/B + 768^3, plus the per-variant register/temp-count proxy + the RE-ANCHORED triangulation
# (arithmetic bucket on [42.86 <-> V-structure-probe]; structural = 75.8 - V-structure-probe). The
# variants are PROFILING INSTRUMENTS with WRONG OUTPUT BY DESIGN (NO correctness gate, NO perf gate);
# the deliverable is the PINPOINT of the dominant 1.77x dequant front-end tax + the next-milestone
# target. Emits AXC_Q4KM_AB_ABLATE. Writes ab_results_ablate.json. No llama path (the llama A/B is M3.6's).
if [ "${ABLATE}" -eq 1 ]; then
    ABLATE_RESULTS="${OUTDIR}/ab_results_ablate.json"
    ABLATE_LOG="${OUTDIR}/ablate_bench_q4km.txt"
    echo "== M3.12 dequant front-end ABLATION DIAGNOSTIC (profiling instruments; no llama path) =="
    echo "  repo_root : ${REPO_ROOT}"
    echo "  ICD       : ${ICD}"
    echo "  outdir    : ${OUTDIR}"
    echo "  deliverable: the PINPOINT (arithmetic vs structural bucket) + the next-milestone target; NO perf/correctness gate (wrong output by design)"
    VK_DRIVER_FILES="${ICD}" AXC_ENABLE_GPU_BENCHES=1 \
        cargo bench --manifest-path "${REPO_ROOT}/Cargo.toml" -p axc-driver \
        --bench resident_q4km_matmul_rb_f32acc_cached_ablate 2>&1 | tee "${ABLATE_LOG}"
    ABLATE_PINPOINT_LINES="$(grep 'ablate AT-2702\|ablate .* TFLOPS\|bucket (\|PINPOINT\|CROSS-CHECK\|temp-count proxy\|MEASURED FINDING\|CONSISTENCY' "${ABLATE_LOG}" || true)"
    ABLATE_LINE="$(grep 'AXC_Q4KM_AB_ABLATE' "${ABLATE_LOG}" | tail -1 || true)"
    {
        echo "{"
        echo "  \"milestone\": \"M3.12-dequant-frontend-ablation-diagnostic\","
        echo "  \"mode\": \"ablation-diagnostic (profiling instruments, wrong output by design, no correctness/perf gate)\","
        echo "  \"deliverable\": \"PINPOINT the dominant 1.77x dequant front-end tax (arithmetic vs structural bucket, RE-ANCHORED on V-structure-probe as the PRIMARY denominator) + name the next-milestone target\","
        echo "  \"icd\": \"${ICD}\","
        echo "  \"vulkaninfo_device\": \"$(VK_DRIVER_FILES="${ICD}" vulkaninfo 2>/dev/null | grep -m1 'deviceName' | sed 's/.*= //' | tr -d '\r' || true)\","
        echo "  \"note\": \"PURE-SOURCE profiling instruments (no codegen change). V-structure-probe (near-free A-value, full M3.6 structure) is the PRIMARY denominator measured FIRST; V-passthrough (f32_to_f16-retaining nibble pass-through) forks structure-vs-arithmetic; V-no-scale-read SUBSUMED, V-no-convert DROPPED (no integer-route-to-f16). Per-variant register/temp-count proxy makes the in-bucket register-occupancy residual visible (the r3 cross-check). See ablate_bench_q4km.txt.\","
        echo "  \"ab_line\": \"$(printf '%s' "${ABLATE_LINE}" | sed 's/"/\\"/g')\","
        echo "  \"pinpoint_lines\": ["
        printf '%s\n' "${ABLATE_PINPOINT_LINES}" | sed '/^$/d' | sed 's/"/\\"/g' | awk 'BEGIN{first=1} {if(!first)printf ",\n"; printf "    \"%s\"", $0; first=0} END{if(!first)printf "\n"}'
        echo "  ]"
        echo "}"
    } > "${ABLATE_RESULTS}"
    echo "-- wrote ${ABLATE_RESULTS}"
    echo "-- full bench log: ${ABLATE_LOG}"
    exit 0
fi

# ── M3.9 --warptile: PLAIN-f32 warptile gate (NOT the Q4_K_M kill-criterion). ───────────────
# The M3.9 primary gate is PLAIN multi-subgroup warptile (K=4) vs PLAIN single-subgroup RB
# (>=1.3x at 768^3 OR the A/B shape 4096x512x14336), measured by the resident_matmul_warptile
# bench itself (it prints per-size TFLOPS + the occupancy artifact + the honest ratio line).
# The Q4_K_M warptile fusion + a llama A/B is the EXPLICIT M3.10 follow-on (out of M3.9 scope),
# so this branch does NOT touch the llama Q4_K_M path. HONEST-NEGATIVE armed: a miss reports the
# numbers for the bottleneck diagnosis, never loosens the gate.
if [ "${WARPTILE}" -eq 1 ]; then
    WT_RESULTS="${OUTDIR}/ab_results_warptile.json"
    WT_LOG="${OUTDIR}/warptile_bench.txt"
    echo "== M3.9 PLAIN-f32 multi-subgroup warptile resident bench (no llama Q4_K_M path) =="
    echo "  repo_root : ${REPO_ROOT}"
    echo "  ICD       : ${ICD}"
    echo "  outdir    : ${OUTDIR}"
    echo "  gate      : K=4 warptile >= 1.3x single-subgroup RB at 768^3 OR A/B (4096x512x14336)"
    VK_DRIVER_FILES="${ICD}" AXC_ENABLE_GPU_BENCHES=1 \
        cargo bench --manifest-path "${REPO_ROOT}/Cargo.toml" -p axc-driver \
        --bench resident_matmul_warptile 2>&1 | tee "${WT_LOG}"
    # Extract the per-size RATIO lines emitted by the bench for the artifact.
    RATIO_LINES="$(grep 'resident_matmul_warptile RATIO' "${WT_LOG}" || true)"
    {
        echo "{"
        echo "  \"milestone\": \"M3.9-multi-subgroup-warptile\","
        echo "  \"mode\": \"plain-f32-warptile-vs-plain-single-subgroup-RB\","
        echo "  \"gate\": \"K=4 warptile >= 1.3x single-subgroup RB at 768^3 OR A/B (4096x512x14336)\","
        echo "  \"icd\": \"${ICD}\","
        echo "  \"vulkaninfo_device\": \"$(VK_DRIVER_FILES="${ICD}" vulkaninfo 2>/dev/null | grep -m1 'deviceName' | sed 's/.*= //' | tr -d '\r' || true)\","
        echo "  \"note\": \"Q4_K_M warptile fusion + llama A/B is the M3.10 follow-on (out of M3.9 scope). See warptile_bench.txt for per-size TFLOPS + the occupancy artifact + the honest ratio.\","
        echo "  \"ratio_lines\": ["
        printf '%s\n' "${RATIO_LINES}" | sed '/^$/d' | sed 's/"/\\"/g' | awk 'BEGIN{first=1} {if(!first)printf ",\n"; printf "    \"%s\"", $0; first=0} END{if(!first)printf "\n"}'
        echo "  ]"
        echo "}"
    } > "${WT_RESULTS}"
    echo "-- wrote ${WT_RESULTS}"
    echo "-- full bench log: ${WT_LOG}"
    exit 0
fi

# ── M3.10a --pad: BANK-PADDED resident benches (NOT the Q4_K_M kill-criterion). ─────────────
# Runs BOTH padded benches: the plain-f32 padded RB (resident_matmul_rb_pad, PRIMARY gate
# >=1.15x the unpadded base at 768^3, PAD in {1,2,4,8} swept) and — for the real llama fight —
# the Q4_K_M padded leader (resident_q4km_matmul_rb_f32acc_cached_pad, gate >=1.15x M3.6's 42.86
# = >=49.3 TFLOPS AND combined <=1e-3 at the A/B shape). Both emit per-PAD per-size TFLOPS +
# shared-bytes + the honest ratio; HONEST-NEGATIVE armed (a miss reports the occupancy/bank
# diagnosis, NEVER loosens the gate). Writes ab_results_pad.json. No llama path (the padded
# kernels are A/B-vs-the-AXIOM-leader experiments; the llama A/B is the M3.6 leader's, unchanged).
if [ "${PAD}" -eq 1 ]; then
    PAD_RESULTS="${OUTDIR}/ab_results_pad.json"
    PAD_LOG_PLAIN="${OUTDIR}/pad_bench_plain.txt"
    PAD_LOG_Q4KM="${OUTDIR}/pad_bench_q4km.txt"
    echo "== M3.10a BANK-PADDED resident benches (plain-f32 + Q4_K_M; no llama Q4_K_M path) =="
    echo "  repo_root : ${REPO_ROOT}"
    echo "  ICD       : ${ICD}"
    echo "  outdir    : ${OUTDIR}"
    echo "  gate      : plain-f32 >=1.15x unpadded base @768^3; Q4_K_M >=1.15x M3.6 42.86 (>=49.3) AND combined<=1e-3"
    echo "-- plain-f32 padded RB sweep (resident_matmul_rb_pad)"
    VK_DRIVER_FILES="${ICD}" AXC_ENABLE_GPU_BENCHES=1 \
        cargo bench --manifest-path "${REPO_ROOT}/Cargo.toml" -p axc-driver \
        --bench resident_matmul_rb_pad 2>&1 | tee "${PAD_LOG_PLAIN}"
    echo "-- Q4_K_M padded leader sweep (resident_q4km_matmul_rb_f32acc_cached_pad)"
    VK_DRIVER_FILES="${ICD}" AXC_ENABLE_GPU_BENCHES=1 \
        cargo bench --manifest-path "${REPO_ROOT}/Cargo.toml" -p axc-driver \
        --bench resident_q4km_matmul_rb_f32acc_cached_pad 2>&1 | tee "${PAD_LOG_Q4KM}"
    PLAIN_GATE_LINES="$(grep 'resident_matmul_rb_pad: AT-2406' "${PAD_LOG_PLAIN}" || true)"
    Q4KM_GATE_LINES="$(grep 'resident_q4km_pad: AT-2407' "${PAD_LOG_Q4KM}" || true)"
    AB_LINE="$(grep 'AXC_Q4KM_AB_F32ACC_CACHED_PAD' "${PAD_LOG_Q4KM}" | tail -1 || true)"
    {
        echo "{"
        echo "  \"milestone\": \"M3.10a-bank-padding\","
        echo "  \"mode\": \"bank-padded-vs-leader (plain-f32 vs unpadded RB; Q4_K_M vs M3.6 cached)\","
        echo "  \"gate\": \"plain-f32 >=1.15x unpadded base @768^3; Q4_K_M >=1.15x M3.6 42.86 (>=49.3 TFLOPS) AND combined<=1e-3\","
        echo "  \"icd\": \"${ICD}\","
        echo "  \"vulkaninfo_device\": \"$(VK_DRIVER_FILES="${ICD}" vulkaninfo 2>/dev/null | grep -m1 'deviceName' | sed 's/.*= //' | tr -d '\r' || true)\","
        echo "  \"note\": \"PURE-SOURCE bank padding (no codegen change). PAD swept in {1,2,4,8}; winner is MEASURED. HONEST-NEGATIVE armed: a sub-gate result reports the shared-bytes + occupancy/bank diagnosis and does NOT loosen the gate. See pad_bench_plain.txt / pad_bench_q4km.txt.\","
        echo "  \"ab_line\": \"$(printf '%s' "${AB_LINE}" | sed 's/"/\\"/g')\","
        echo "  \"plain_gate_lines\": ["
        printf '%s\n' "${PLAIN_GATE_LINES}" | sed '/^$/d' | sed 's/"/\\"/g' | awk 'BEGIN{first=1} {if(!first)printf ",\n"; printf "    \"%s\"", $0; first=0} END{if(!first)printf "\n"}'
        echo "  ],"
        echo "  \"q4km_gate_lines\": ["
        printf '%s\n' "${Q4KM_GATE_LINES}" | sed '/^$/d' | sed 's/"/\\"/g' | awk 'BEGIN{first=1} {if(!first)printf ",\n"; printf "    \"%s\"", $0; first=0} END{if(!first)printf "\n"}'
        echo "  ]"
        echo "}"
    } > "${PAD_RESULTS}"
    echo "-- wrote ${PAD_RESULTS}"
    echo "-- full bench logs: ${PAD_LOG_PLAIN} ${PAD_LOG_Q4KM}"
    exit 0
fi

mkdir -p "$OUTDIR"
RAW="${OUTDIR}/llamacpp_raw.txt"
# M3.5 (--fused) and M3.5b (--fused-f32acc) each write a DISTINCT artifact; the frozen-matvec
# ab_results.json and the f16-accum ab_results_fused.json are kept side-by-side.
if [ "${FUSED_F32ACC_RB44}" -eq 1 ]; then
    RESULTS_JSON="${OUTDIR}/ab_results_fused_f32acc_bigrb_4x4.json"
    RESULTS_MD="${OUTDIR}/ab_results_fused_f32acc_bigrb_4x4.md"
elif [ "${FUSED_F32ACC_RB42}" -eq 1 ]; then
    RESULTS_JSON="${OUTDIR}/ab_results_fused_f32acc_bigrb_4x2.json"
    RESULTS_MD="${OUTDIR}/ab_results_fused_f32acc_bigrb_4x2.md"
elif [ "${FUSED_F32ACC_DB}" -eq 1 ]; then
    RESULTS_JSON="${OUTDIR}/ab_results_fused_f32acc_db.json"
    RESULTS_MD="${OUTDIR}/ab_results_fused_f32acc_db.md"
elif [ "${FUSED_F32ACC_CACHED}" -eq 1 ]; then
    RESULTS_JSON="${OUTDIR}/ab_results_fused_f32acc_cached.json"
    RESULTS_MD="${OUTDIR}/ab_results_fused_f32acc_cached.md"
elif [ "${FUSED_F32ACC}" -eq 1 ]; then
    RESULTS_JSON="${OUTDIR}/ab_results_fused_f32acc.json"
    RESULTS_MD="${OUTDIR}/ab_results_fused_f32acc.md"
elif [ "${FUSED}" -eq 1 ]; then
    RESULTS_JSON="${OUTDIR}/ab_results_fused.json"
    RESULTS_MD="${OUTDIR}/ab_results_fused.md"
else
    RESULTS_JSON="${OUTDIR}/ab_results.json"
    RESULTS_MD="${OUTDIR}/ab_results.md"
fi

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

# Helper: parse one Q4_K line into TFLOPS / m / n / k / us / runs (echoed space-separated).
parse_q4k_line() {
    local line="$1"
    local _m _n _k _runs _us _tf
    _m="$(echo "${line}" | sed -n 's/.*[,(]m=\([0-9]*\).*/\1/p')"
    _n="$(echo "${line}" | sed -n 's/.*,n=\([0-9]*\),.*/\1/p')"
    _k="$(echo "${line}" | sed -n 's/.*,k=\([0-9]*\),.*/\1/p')"
    _runs="$(echo "${line}" | sed -n 's/.*[:)] *\([0-9]*\) runs.*/\1/p')"
    _us="$(echo "${line}" | sed -n 's/.* runs - *\([0-9.]*\) us\/run.*/\1/p')"
    _tf="$(echo "${line}" | sed -n 's/.* - *\([0-9.]*\) TFLOPS.*/\1/p')"
    echo "${_m} ${_n} ${_k} ${_runs} ${_us} ${_tf}"
}

# The n==1 (GEMV) case — CROSS-SHAPE CONTEXT in --fused mode; the headline in matvec mode.
Q4K_N1_LINE="$(printf '%s\n' "${CLEAN_RAW}" \
    | grep 'MUL_MAT(type_a=q4_K' | grep ',n=1,' | head -1 || true)"
# The n==512 SAME-SHAPE case — the HEADLINE in --fused mode (m=4096,n=512,k=14336=101 TFLOPS).
Q4K_N512_LINE="$(printf '%s\n' "${CLEAN_RAW}" \
    | grep 'MUL_MAT(type_a=q4_K' | grep ',m=4096,n=512,k=14336,' | head -1 || true)"

KILL_STATUS="FAIL"          # mechanical; expected FAIL on NVIDIA
KILL_REASON=""
INCOMPLETE=0
# Defaults.
LLAMA_US="null"; LLAMA_TFLOPS="null"; LLAMA_M="null"; LLAMA_N="null"; LLAMA_K="null"; LLAMA_RUNS="null"
LLAMA_SAMESHAPE_TFLOPS="null"; LLAMA_GEMV_CONTEXT_TFLOPS="null"

# ── n==1 GEMV parse (kill-criterion headline in matvec mode; context in fused mode) ─
if [ -z "${Q4K_N1_LINE}" ]; then
    if [ "${FUSED}" -eq 0 ]; then
        KILL_STATUS="INCOMPLETE"
        KILL_REASON="no type_a=q4_K MUL_MAT n==1 (GEMV) case found (CRITICAL-2: GEMM substitution forbidden)"
        INCOMPLETE=1
    fi
else
    echo "-- Q4_K n==1 (GEMV) line:"
    echo "   ${Q4K_N1_LINE}"
    read -r N1_M N1_N N1_K N1_RUNS N1_US N1_TF <<< "$(parse_q4k_line "${Q4K_N1_LINE}")"
    LLAMA_GEMV_CONTEXT_TFLOPS="${N1_TF}"
    if [ "${FUSED}" -eq 0 ]; then
        LLAMA_M="${N1_M}"; LLAMA_N="${N1_N}"; LLAMA_K="${N1_K}"
        LLAMA_RUNS="${N1_RUNS}"; LLAMA_US="${N1_US}"; LLAMA_TFLOPS="${N1_TF}"
        if [ -z "${LLAMA_M}" ] || [ -z "${LLAMA_K}" ] || [ -z "${LLAMA_US}" ] || [ -z "${LLAMA_TFLOPS}" ]; then
            KILL_STATUS="INCOMPLETE"
            KILL_REASON="failed to parse m/k/us/TFLOPS from the n=1 Q4_K line (format drift)"
            INCOMPLETE=1
        fi
    fi
fi

# ── n==512 SAME-SHAPE parse (kill-criterion headline in --fused mode, CRITICAL-1) ──
if [ "${FUSED}" -eq 1 ]; then
    if [ -z "${Q4K_N512_LINE}" ]; then
        KILL_STATUS="INCOMPLETE"
        KILL_REASON="no type_a=q4_K MUL_MAT same-shape (m=4096,n=512,k=14336) case found (CRITICAL-1: SAME-SHAPE headline required)"
        INCOMPLETE=1
    else
        echo "-- Q4_K SAME-SHAPE (m=4096,n=512,k=14336) line:"
        echo "   ${Q4K_N512_LINE}"
        read -r S_M S_N S_K S_RUNS S_US S_TF <<< "$(parse_q4k_line "${Q4K_N512_LINE}")"
        LLAMA_M="${S_M}"; LLAMA_N="${S_N}"; LLAMA_K="${S_K}"
        LLAMA_RUNS="${S_RUNS}"; LLAMA_US="${S_US}"; LLAMA_TFLOPS="${S_TF}"
        LLAMA_SAMESHAPE_TFLOPS="${S_TF}"
        if [ -z "${LLAMA_M}" ] || [ -z "${LLAMA_K}" ] || [ -z "${LLAMA_US}" ] || [ -z "${LLAMA_TFLOPS}" ]; then
            KILL_STATUS="INCOMPLETE"
            KILL_REASON="failed to parse m/n/k/us/TFLOPS from the same-shape Q4_K line (format drift)"
            INCOMPLETE=1
        fi
    fi
fi

# ── 5. AXIOM side: run the AXIOM bench, parse its anchored line ───────────────────────────
if [ "${FUSED_F32ACC_RB44}" -eq 1 ]; then
    AXC_BENCH="resident_q4km_matmul_rb_f32acc_cached_bigrb"
    AXC_PREFIX="AXC_Q4KM_AB_F32ACC_BIGRB_4X4"
    AXC_KERNEL="fused_f32acc_bigrb_4x4"
elif [ "${FUSED_F32ACC_RB42}" -eq 1 ]; then
    AXC_BENCH="resident_q4km_matmul_rb_f32acc_cached_bigrb"
    AXC_PREFIX="AXC_Q4KM_AB_F32ACC_BIGRB_4X2"
    AXC_KERNEL="fused_f32acc_bigrb_4x2"
elif [ "${FUSED_F32ACC_DB}" -eq 1 ]; then
    AXC_BENCH="resident_q4km_matmul_rb_f32acc_db"
    AXC_PREFIX="AXC_Q4KM_AB_F32ACC_DB"
    AXC_KERNEL="fused_f32acc_db"
elif [ "${FUSED_F32ACC_CACHED}" -eq 1 ]; then
    AXC_BENCH="resident_q4km_matmul_rb_f32acc_cached"
    AXC_PREFIX="AXC_Q4KM_AB_F32ACC_CACHED"
    AXC_KERNEL="fused_f32acc_cached"
elif [ "${FUSED_F32ACC}" -eq 1 ]; then
    AXC_BENCH="resident_q4km_matmul_rb_f32acc"
    AXC_PREFIX="AXC_Q4KM_AB_F32ACC"
    AXC_KERNEL="fused_f32acc"
elif [ "${FUSED}" -eq 1 ]; then
    AXC_BENCH="resident_q4km_matmul_rb"
    AXC_PREFIX="AXC_Q4KM_AB_FUSED"
    AXC_KERNEL="fused"
else
    AXC_BENCH="dispatch_q4km_ab"
    AXC_PREFIX="AXC_Q4KM_AB"
    AXC_KERNEL="matvec"
fi
echo "-- running AXIOM ${AXC_BENCH} bench (kernel=${AXC_KERNEL})"
BENCH_OUT="${OUTDIR}/axiom_bench_raw.txt"
AXC_ENABLE_GPU_BENCHES=1 VK_DRIVER_FILES="${ICD}" \
    cargo bench --manifest-path "${REPO_ROOT}/Cargo.toml" -p axc-driver --bench "${AXC_BENCH}" \
    > "${BENCH_OUT}" 2>&1 || {
    echo "FATAL: AXIOM bench exited non-zero (see ${BENCH_OUT})" >&2
    exit 1
}

AXC_LINE="$(grep -m1 "^${AXC_PREFIX} " "${BENCH_OUT}" || true)"
if [ -z "${AXC_LINE}" ]; then
    echo "FATAL: no ${AXC_PREFIX} line in bench output (GPU benches may be disabled/skipped)" >&2
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
AXC_DEVICE="$(echo "${AXC_LINE}" | sed -n 's/.* device=\(.*\)$/\1/p')"
AXIOM_DEVICE="${AXC_DEVICE}"
# Fused-only: the AXIOM GEMM output dims (m=rows, n=cols) for the same-shape ratio.
# M3.5b (--fused-f32acc) emits TWO correctness fields: `combined` (condition-aware backward-stable
# metric, the GPU-AT gate) and `raw` (forward error, reporting only). numerically_valid MUST be
# driven by the COMBINED metric — the raw forward error is ~1e-2 on near-zero cancellation outputs
# at the A/B shape (a metric artifact, identical-in-kind to llama.cpp's own HMMA), so keying
# validity off raw would dishonestly under-claim. M3.5 (--fused) emits a single `max_rel_diff`.
if [ "${FUSED_F32ACC}" -eq 1 ]; then
    AXC_M="$(axc_field m)"
    AXC_N="$(axc_field n)"
    AXC_MAX_REL_DIFF="$(axc_field combined)"   # COMBINED drives numerically_valid (≤ frozen 1e-3)
    AXC_RAW_REL_DIFF="$(axc_field raw)"        # raw forward error, recorded for transparency
elif [ "${FUSED}" -eq 1 ]; then
    AXC_M="$(axc_field m)"
    AXC_N="$(axc_field n)"
    AXC_MAX_REL_DIFF="$(axc_field max_rel_diff)"
    AXC_RAW_REL_DIFF="null"
else
    AXC_M="1"; AXC_N="1"; AXC_MAX_REL_DIFF="null"; AXC_RAW_REL_DIFF="null"
fi
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
# M3.5 (--fused) discriminators + same-shape / cross-shape llama numbers (CRITICAL-1).
export M34_AXC_KERNEL="${AXC_KERNEL}"
export M34_AXC_M="${AXC_M}"
export M34_AXC_N="${AXC_N}"
export M34_AXC_MAX_REL_DIFF="${AXC_MAX_REL_DIFF}"
export M34_AXC_RAW_REL_DIFF="${AXC_RAW_REL_DIFF}"
export M34_LLAMA_SAMESHAPE_TFLOPS="${LLAMA_SAMESHAPE_TFLOPS}"
export M34_LLAMA_GEMV_CONTEXT_TFLOPS="${LLAMA_GEMV_CONTEXT_TFLOPS}"
export M34_Q4K_SAMESHAPE_LINE="${Q4K_N512_LINE:-}"
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
