#!/usr/bin/env python3
"""M3.4 A/B results emitter — reads measured values from M34_* env vars, computes the
like-for-like throughput ratio + kill-criterion verdict, and writes ab_results.json +
ab_results.md. Invoked by scripts/m34_llamacpp_ab.sh. Exits non-zero on INCOMPLETE.

HONESTY: the ratio is reported AS MEASURED. The fair metric is work-normalized THROUGHPUT
(TFLOPS) because llama.cpp computes 4096 output rows per op while AXIOM computes 1 — a raw
us comparison is apples-to-oranges. No cherry-picking, no GEMM substitution for the kill
number (CRITICAL-2), no silent best-case-vs-sustained ratio (CRITICAL-1)."""

import json
import os
import sys


def env(name, default=""):
    return os.environ.get(name, default)


def num_or_none(name):
    v = env(name)
    if v in ("", "null", "None"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def int_or_none(name):
    v = num_or_none(name)
    return int(v) if v is not None else None


incomplete = int(env("M34_INCOMPLETE", "0"))
axc_ns_min = int(env("M34_AXC_NS_MIN"))
axc_ns_mean = int(env("M34_AXC_NS_MEAN"))
axc_ns_median = int(env("M34_AXC_NS_MEDIAN"))
axc_sustained_ns = int(env("M34_AXC_SUSTAINED_NS"))
axc_k = int(env("M34_AXC_K"))
axc_flops = int(env("M34_AXC_FLOPS"))
axc_timing = env("M34_AXC_TIMING")
axc_device = env("M34_AXC_DEVICE")
llama_device = env("M34_LLAMA_DEVICE")
device_match = env("M34_DEVICE_MATCH") == "true"
kill_status = env("M34_KILL_STATUS")
kill_reason = env("M34_KILL_REASON")

llama_us = num_or_none("M34_LLAMA_US")
llama_tflops = num_or_none("M34_LLAMA_TFLOPS")
llama_m = int_or_none("M34_LLAMA_M")
llama_n = int_or_none("M34_LLAMA_N")
llama_k = int_or_none("M34_LLAMA_K")
llama_runs = int_or_none("M34_LLAMA_RUNS")
q4k_line = env("M34_Q4K_LINE")

# M3.5 (--fused) / M3.5b (--fused-f32acc) discriminators (CRITICAL-1): SAME-SHAPE headline +
# cross-shape context. Both fused variants share the same SAME-SHAPE A/B machinery; the only
# difference is numerical_validity (f16-accum is invalid at inference K; f32-accum is valid).
axc_kernel = env("M34_AXC_KERNEL", "matvec")  # "matvec" (M3.4) | "fused" (M3.5) | "fused_f32acc" (M3.5b)
is_fused = axc_kernel in ("fused", "fused_f32acc")
is_f32acc = axc_kernel == "fused_f32acc"
axc_m_out = int_or_none("M34_AXC_M") or 1      # AXIOM fused GEMM output rows
axc_n_out = int_or_none("M34_AXC_N") or 1      # AXIOM fused GEMM output cols
# For M3.5b (fused_f32acc), M34_AXC_MAX_REL_DIFF carries the COMBINED (condition-aware,
# backward-stable) metric — this is what DRIVES numerically_valid (the same gate the GPU dispatch
# ATs use). M34_AXC_RAW_REL_DIFF carries the raw forward error (~1e-2 on cancellation outputs, a
# metric artifact identical-in-kind to llama.cpp's HMMA), recorded for transparency only. For
# M3.4/M3.5, M34_AXC_RAW_REL_DIFF is absent (None) and max_rel_diff is the only metric.
axc_max_rel_diff = num_or_none("M34_AXC_MAX_REL_DIFF")
axc_raw_rel_diff = num_or_none("M34_AXC_RAW_REL_DIFF")
llama_sameshape_tflops = num_or_none("M34_LLAMA_SAMESHAPE_TFLOPS")
llama_gemv_context_tflops = num_or_none("M34_LLAMA_GEMV_CONTEXT_TFLOPS")
q4k_sameshape_line = env("M34_Q4K_SAMESHAPE_LINE")

# AXIOM derived numbers (work-normalized TFLOPS, dequant excluded).
axc_us_min = axc_ns_min / 1e3
axc_us_sustained = axc_sustained_ns / 1e3
axc_tflops_min = axc_flops / (axc_ns_min * 1e-9) / 1e12 if axc_ns_min else None
axc_tflops_sustained = axc_flops / (axc_sustained_ns * 1e-9) / 1e12 if axc_sustained_ns else None

# Frozen correctness tolerance (DESIGN §3.1.20; AT-1770/1771). The fused kernel is only
# numerically valid where the measured max_rel_diff stays within this bound.
FROZEN_TOL = 1e-3

flop_consistency = {"checked": False, "ok": None}
ratio_tflops = None
ratio_boundary_label = None
headline_basis = None
cross_shape_context = None
numerical_validity = None

if not incomplete and llama_us is not None and llama_k is not None and llama_m is not None:
    # FLOP-consistency (WARNING-3): recompute llama.cpp GFLOPS from its us + m/n/k vs its
    # self-reported TFLOPS within 2%.
    n_eff = llama_n if llama_n else 1
    # llama_us is MICROSECONDS → seconds is *1e-6 (NOT 1e-9).
    recomputed_tflops = (2 * llama_m * n_eff * llama_k) / (llama_us * 1e-6) / 1e12
    disagree = abs(recomputed_tflops - llama_tflops) / llama_tflops if llama_tflops else 1.0
    flop_consistency = {
        "checked": True,
        "ok": disagree <= 0.02,
        "reported_tflops": llama_tflops,
        "recomputed_tflops": round(recomputed_tflops, 4),
        "disagreement_frac": round(disagree, 5),
        "convention": "2*m*n*k matmul MACs only (dequant excluded), both sides",
    }
    if not flop_consistency["ok"]:
        kill_status = "INCOMPLETE"
        kill_reason = (f"FLOP-consistency failed: recomputed {recomputed_tflops:.3f} vs "
                       f"reported {llama_tflops:.3f} TFLOPS (>2%)")
        incomplete = 1
    elif not device_match:
        kill_status = "INCOMPLETE"
        kill_reason = (f"device-match failed (not byte-identical): "
                       f"AXIOM='{axc_device}' llama='{llama_device}'")
        incomplete = 1
    elif is_fused:
        # M3.5 SAME-SHAPE headline (CRITICAL-1): AXIOM fused GEMM (GpuTimestamp MIN,
        # kernel-only) vs llama Q4_K MUL_MAT at the IDENTICAL (m=4096,n=512,k=14336).
        # The headline + within-15% verdict come from the SAME-SHAPE llama number ONLY.
        denom = llama_sameshape_tflops if llama_sameshape_tflops else llama_tflops
        ratio_tflops = axc_tflops_min / denom
        # Numerical validity at the A/B shape (computed from the MEASURED max_rel_diff, never
        # hand-asserted): the f16 coopmat accumulator overflows precision at inference-scale K.
        ab_valid = (axc_max_rel_diff is not None) and (axc_max_rel_diff <= FROZEN_TOL)
        ratio_boundary_label = (
            "SAME-SHAPE GEMM-vs-GEMM: AXIOM fused GEMM (GpuTimestamp MIN, kernel-only) vs "
            f"llama.cpp Q4_K MUL_MAT at the IDENTICAL m=4096,n=512,k=14336 ({denom:.2f} TFLOPS). "
            "Headline + verdict computed from the same-shape llama number; the n=1 GEMV is "
            "cross-shape context only."
            + ("" if ab_valid else
               f" CAVEAT: AXIOM is numerically INVALID at this k (max_rel_diff="
               f"{axc_max_rel_diff:.2f}) — the ratio is throughput-only, NOT a usable-kernel "
               f"comparison (see numerical_validity)."))
        headline_basis = "same_shape_gemm m=4096,n=512,k=14336"
        within_15 = ratio_tflops >= 0.85
        kill_status = "PASS" if within_15 else "FAIL"
        # numerical_validity (CRITICAL correctness gate): computed from the MEASURED
        # max_rel_diff at the A/B shape, never hand-asserted. valid only when within tol.
        _mrd = axc_max_rel_diff if axc_max_rel_diff is not None else float("nan")
        if ab_valid:
            _nv_verdict = (
                f"numerically VALID at the A/B shape: max_rel_diff={_mrd:.3g} <= frozen "
                f"{FROZEN_TOL:g}. The throughput ratio below is a usable-kernel comparison.")
            _nv_kill = ("Throughput gap closed (M3.4 ~87,000x -> M3.5 same-shape) AND "
                        "numerically valid at the A/B shape — a usable-kernel comparison.")
        else:
            _over = _mrd / FROZEN_TOL if _mrd == _mrd else float("inf")
            _nv_verdict = (
                f"fast-but-WRONG at inference K (f16 accumulator); correct only at K<=256; "
                f"usable competitive kernel needs an f32-accumulator coopmat shape (M3.5b). "
                f"At k={axc_k} the fused kernel's max_rel_diff={_mrd:.4g} vs the "
                f"f16-accumulator ggml reference — ~{_over:.0f}x over the frozen {FROZEN_TOL:g}. "
                f"The f16 coopmat accumulator cannot hold an inference-scale K sum; the output "
                f"is GARBAGE at this K. Correctness holds ONLY at small K (AT-1770 K=256 "
                f"max_rel_diff=8.3e-4 PASS; AT-1771 K=512 max_rel_diff=3.6e-3 EXCEEDS 1e-3, "
                f"gate capped at K=256). The throughput ratio is therefore FAST-BUT-WRONG at "
                f"inference K — NOT a usable-kernel win. A correct large-K fused kernel needs "
                f"an f32-accumulator coopmat shape (M3.5b).")
            _nv_kill = (
                f"Throughput gap closed dramatically (M3.4 ~87,000x -> M3.5 same-shape) BUT "
                f"this is NOT a usable-kernel win: AXIOM's result is numerically invalid at "
                f"inference K.")
        numerical_validity = {
            "valid_at_ab_shape": bool(ab_valid),
            # For M3.5b: ab_shape_max_rel_diff is the COMBINED (condition-aware, backward-stable)
            # metric that DRIVES validity; ab_shape_raw_rel_diff is the raw forward error recorded
            # for transparency (NOT a gate). For M3.4/M3.5 the raw field is null (single metric).
            "ab_shape_max_rel_diff": axc_max_rel_diff,
            "ab_shape_metric": ("combined_condition_aware" if is_f32acc else "raw_relative"),
            "ab_shape_raw_rel_diff": axc_raw_rel_diff,
            "frozen_tol": FROZEN_TOL,
            "verdict": _nv_verdict,
            "kill_criterion_interpretation": _nv_kill,
        }
        # Cross-shape context block (labeled, NEVER the headline / verdict input).
        cross_shape_context = {
            "axiom_gemm_tflops": round(axc_tflops_min, 4) if axc_tflops_min else None,
            "llama_n1_gemv_tflops": llama_gemv_context_tflops,
            "axiom_over_llama_n1_ratio": (
                round(axc_tflops_min / llama_gemv_context_tflops, 4)
                if (axc_tflops_min and llama_gemv_context_tflops) else None),
            "note": ("cross-shape, NOT the kill criterion — llama's n=1 is a memory-bound "
                     "GEMV at ~13.7x lower arithmetic intensity than its own n=512 GEMM; "
                     "headlining AXIOM-GEMM vs llama-n1-GEMV would flatter AXIOM ~13x."),
        }
    else:
        # M3.4 matvec: like-for-like THROUGHPUT ratio at the matched (sustained CPU-wall)
        # boundary; GpuTimestamp-MIN reported alongside for disclosure.
        ratio_tflops = axc_tflops_sustained / llama_tflops
        ratio_boundary_label = (
            "AXIOM-sustained-CPU-wall (matched to llama.cpp sustained CPU-wall MEAN) vs "
            "llama.cpp sustained TFLOPS; GpuTimestamp-MIN framing reported alongside for "
            "disclosure")
        headline_basis = "same_shape_gemv m=4096,n=1,k=14336"
        within_15 = ratio_tflops >= 0.85
        kill_status = "PASS" if within_15 else "FAIL"

verbatim_excerpt = (
    "// run\n"
    "int64_t total_time_us = 0;\n"
    "int total_runs = 0;\n"
    "do {\n"
    "    int64_t start_time = ggml_time_us();\n"
    "    ggml_status status = ggml_backend_graph_compute(backend, gf);\n"
    "    ...\n"
    "    int64_t end_time = ggml_time_us();\n"
    "    total_time_us += end_time - start_time;\n"
    "    total_runs += n_runs;\n"
    "} while (total_time_us < 1000*1000); // run for at least 1 second\n"
    "double avg_time_us = (double) total_time_us / total_runs;\n"
    "double calculated_flops = (op_flops(out) * total_runs) / (total_time_us / 1e6);"
)

out = {
    "milestone": ("M3.5b-llamacpp-ab-fused-f32acc" if is_f32acc
                  else "M3.5-llamacpp-ab-fused" if is_fused
                  else "M3.4-llamacpp-ab"),
    "axiom_kernel": axc_kernel,
    "generated_by": "scripts/m34_llamacpp_ab.sh + scripts/m34_emit.py",
    "commit": {"tag": env("M34_LLAMACPP_TAG"), "sha": env("M34_LLAMACPP_COMMIT")},
    "icd": env("M34_ICD"),
    "device": {
        "axiom": axc_device,
        "llamacpp": llama_device,
        "vulkaninfo": env("M34_VULKANINFO_DEVICE"),
    },
    "device_match": {
        "byte_identical": bool(device_match),
        "rtx_pro_6000_substring": ("RTX PRO 6000" in axc_device) and ("RTX PRO 6000" in llama_device),
    },
    "dims": {
        "K_contraction": axc_k, "llama_m": llama_m, "llama_n": llama_n, "llama_k": llama_k,
        "note": (
            (f"SAME-SHAPE GEMM-vs-GEMM: AXIOM's FUSED kernel computes the full m={axc_m_out}, "
             f"n={axc_n_out} GEMM (axiom.fused_output_m={axc_m_out}, "
             f"fused_output_n={axc_n_out}), and llama.cpp Q4_K MUL_MAT computes the IDENTICAL "
             f"m={llama_m}, n={llama_n}, k={llama_k}. Throughput (TFLOPS) is the work-normalized "
             "metric (2*m*n*k MACs, dequant excluded, both sides).")
            if is_fused else
            ("AXIOM computes M=1 output row; llama.cpp computes m=4096 rows over the "
             "SAME K=14336 contraction (the apples-to-oranges row-count caveat). "
             "Throughput (TFLOPS) is the fair, work-normalized metric.")),
    },
    "llamacpp_timer_boundary": {
        "source_file": "tests/test-backend-ops.cpp",
        "sha": env("M34_LLAMACPP_COMMIT"),
        "measurement": ("CPU wall-clock (ggml_time_us) bracketing ggml_backend_graph_compute "
                        "over a graph that DUPLICATES the op n_runs times"),
        "per_op_or_amortized": "batched-amortized: avg_time_us = total_time_us / total_runs",
        "min_mean_or_sustained": ("sustained MEAN (loops until total_time_us >= 1s; reports "
                                  "the mean per-op time, NOT min)"),
        "warmup": "ONE ggml_backend_graph_compute warmup discarded before timing",
        "overhead_subtracted": "none",
        "n_runs_basis": ("FLOP-targeted for MUL_MAT: ~100 GFLOP target / op_flops, so many "
                         "op copies per graph"),
        "source_excerpt": verbatim_excerpt,
    },
    "axiom": {
        "kernel_ns_min": axc_ns_min, "kernel_ns_mean": axc_ns_mean,
        "kernel_ns_median": axc_ns_median, "sustained_ns": axc_sustained_ns,
        "us_min": round(axc_us_min, 3), "us_sustained": round(axc_us_sustained, 3),
        "tflops_min_gputimestamp": round(axc_tflops_min, 6) if axc_tflops_min else None,
        "tflops_sustained_cpuwall": round(axc_tflops_sustained, 6) if axc_tflops_sustained else None,
        "timing_source": axc_timing,
        "flops_per_dispatch": axc_flops,
        "flops_convention": (
            f"matmul MACs, dequant EXCLUDED (2*M*N*K, M={axc_m_out}, N={axc_n_out})"
            if is_fused else "matmul-equivalent FLOPs, dequant EXCLUDED (2*M*K, M=1)"),
        "fused_output_m": axc_m_out if is_fused else None,
        "fused_output_n": axc_n_out if is_fused else None,
        "fused_max_rel_diff": axc_max_rel_diff if is_fused else None,
        "note": (
            "FUSED Q4_K_M dequant + 2x2 register-blocked coopmat matmul; full (N/32,M/32,1) "
            "grid (NOT the single-row matvec). GpuTimestamp MIN is the kernel-only number."
            if is_fused else
            "single workgroup, ONE output row (the M2.6 'if i>=1 return' guard) — "
            "~1 of ~188 SMs occupied"),
    },
    "llamacpp": {
        # CRITICAL-1: in fused mode the headline TFLOPS come from the SAME-SHAPE (n=512) line,
        # so selected_line MUST be that same-shape line (not the n=1 GEMV context line).
        "selected_line": (q4k_sameshape_line if (is_fused and q4k_sameshape_line) else q4k_line),
        "us_per_run": llama_us, "tflops": llama_tflops, "n_runs": llama_runs,
        "m": llama_m, "n": llama_n, "k": llama_k,
    },
    "flop_consistency": flop_consistency,
    "ratio_boundary_label": ratio_boundary_label,
    "headline_basis": headline_basis,
    "headline_ratio": round(ratio_tflops, 5) if ratio_tflops is not None else None,
    "ratio_axiom_over_llama_tflops": round(ratio_tflops, 5) if ratio_tflops is not None else None,
    "llama_over_axiom_speedup_tflops": round(1.0 / ratio_tflops, 2) if ratio_tflops else None,
    "cross_shape_context": cross_shape_context,
    "axiom_fused_max_rel_diff": axc_max_rel_diff,
    "axiom_fused_raw_rel_diff": axc_raw_rel_diff,
    "kill_criterion_within_15pct": (kill_status == "PASS") if kill_status in ("PASS", "FAIL") else None,
    "kill_criterion_status": kill_status,
    "kill_criterion_reason": kill_reason,
    "numerical_validity": numerical_validity,
    "kill_criterion_qualifier": (
        (f"M3.5 SAME-SHAPE fused GEMM vs llama's same-shape Q4_K MUL_MAT "
         f"({(llama_sameshape_tflops if llama_sameshape_tflops else llama_tflops):.2f} TFLOPS). "
         f"AXIOM measured {axc_tflops_min:.2f} TFLOPS = "
         f"~{(1.0 / ratio_tflops):.0f}x BEHIND on throughput — a MASSIVE improvement from "
         f"M3.4's ~87,000x cross-shape matvec." +
         ("" if (numerical_validity and numerical_validity["valid_at_ab_shape"]) else
          f" BUT this is NOT a usable-kernel win: AXIOM's output is NUMERICALLY INVALID at this "
          f"k={axc_k} shape (max_rel_diff={axc_max_rel_diff:.2f}, see numerical_validity) "
          f"because the f16 coopmat accumulator overflows precision at inference-scale K; "
          f"correct only at K<=256. So the throughput gap closed dramatically but AXIOM is "
          f"fast-but-WRONG at inference K (needs an f32-accumulator coopmat, M3.5b).") +
         f" DESIGN §5 kill-criterion is 'within 15% on ANY vendor'; this does NOT fire "
         f"(AXIOM behind" +
         ("" if (numerical_validity and numerical_validity["valid_at_ab_shape"]) else
          " AND numerically invalid") +
         f"). AMD/Intel pending hardware (EB.1).")
        if is_fused else
        ("NVIDIA-only FAIL with the current FROZEN M2.6 single-row matvec is the documented "
         "baseline; DESIGN §5 kill-criterion is 'within 15% on ANY vendor', so this does NOT "
         "fire the project kill-criterion. AMD/Intel halves pending cross-vendor hardware (EB.1).")),
    "fairness_caveat": (
        ("Same machine, same ICD, kernel-only-vs-kernel-only, SAME SHAPE (m=4096,n=512,k=14336 "
         "both sides), FLOP convention identical (2*m*n*k matmul MACs, dequant excluded). " +
         (("AXIOM's fused kernel is numerically VALID at this A/B shape (max_rel_diff="
           f"{axc_max_rel_diff:.3g} <= frozen 1e-3); the ratio is a usable-kernel comparison.")
          if (numerical_validity and numerical_validity["valid_at_ab_shape"]) else
          ("CRITICAL CORRECTNESS CAVEAT: AXIOM's fused output is NUMERICALLY INVALID at this A/B "
           f"shape (k={axc_k}, max_rel_diff={axc_max_rel_diff:.2f} — see numerical_validity) "
           "because the f16 coopmat accumulator overflows precision at inference-scale K; it is "
           f"correct ONLY at K<=256. So the ~{(1.0 / ratio_tflops):.0f}x throughput gap is "
           "fast-but-WRONG at inference K, NOT AXIOM being that far behind a usable kernel of "
           "its own.")) +
         f" Ratio reported AS MEASURED for a numerically-"
         f"{'VALID' if (numerical_validity and numerical_validity['valid_at_ab_shape']) else 'INVALID'}"
         f" result at the A/B K. The n=1 GEMV (llama "
         f"{(llama_gemv_context_tflops if llama_gemv_context_tflops else 0.0):.2f} TFLOPS) is "
         "cross-shape CONTEXT ONLY — never the headline (the SAME llama kernel runs 13.7x faster "
         "at n=512), so headlining AXIOM-GEMM vs llama-n1-GEMV would flatter AXIOM ~13x; that is "
         "forbidden (CRITICAL-1).")
        if is_fused else
        ("Same machine, same ICD, kernel-only-vs-kernel-only, identical K=14336 contraction, FLOP "
         "convention identical (2*m*n*k matmul MACs, dequant excluded). BOTH kernels are CORRECT "
         "(AXIOM bit-exact vs ggml CPU ref; llama.cpp IS ggml). AXIOM is single-row matvec (1 "
         "workgroup) vs llama.cpp's tiled multi-row MUL_MAT (all SMs) — ~structurally 100x "
         "under-parallelized. Ratio reported AS MEASURED; honest expected FAIL.")),
    "gap_closing_path": (
        ("M3.5 closed the M3.4 ~87,000x cross-shape THROUGHPUT gap to a single-digit same-shape "
         "ratio. " +
         (("Remaining gap (if ALU-bound): scale-caching across the 16 tile_k K-blocks of a "
           "superblock (OQ-3 -> M3.5b) to cut ALU-bound dequant overhead.")
          if (numerical_validity and numerical_validity["valid_at_ab_shape"]) else
          ("But the BLOCKING issue is NOT throughput — it is CORRECTNESS: the f16 coopmat "
           f"accumulator is numerically invalid at inference-scale K (max_rel_diff="
           f"{axc_max_rel_diff:.2f} at k={axc_k}). The next milestone (M3.5b) MUST add an "
           "f32-accumulator coopmat shape so the fused kernel is CORRECT at large K; only then "
           "is the throughput number a usable-kernel comparison. Secondary: scale-caching "
           "across the 16 tile_k K-blocks of a superblock (OQ-3) to cut ALU-bound dequant "
           "overhead.")))
        if is_fused else
        ("Fuse the Q4_K_M dequant front-end onto the M3.3c register-blocked coopmat matmul "
         "(dequant -> shared f16 tile -> coopmat mul_add; plain-f16 reached 31.2 TFLOPS = "
         "24.96% of datasheet). Follow-up milestone (M3.5).")),
}

with open(env("M34_RESULTS_JSON"), "w") as f:
    json.dump(out, f, indent=2)
    f.write("\n")


def fmt(x, d=3):
    if x is None:
        return "n/a"
    return f"{x:.{d}f}" if isinstance(x, float) else str(x)


md = []
if is_f32acc:
    md.append("# M3.5b — llama.cpp Vulkan Q4_K_M A/B, f32-ACCUMULATOR FUSED kernel, SAME-SHAPE "
              "(NVIDIA RTX PRO 6000)\n")
elif is_fused:
    md.append("# M3.5 — llama.cpp Vulkan Q4_K_M A/B, FUSED kernel, SAME-SHAPE (NVIDIA RTX PRO 6000)\n")
else:
    md.append("# M3.4 — llama.cpp Vulkan Q4_K_M A/B (NVIDIA RTX PRO 6000)\n")
md.append(f"- llama.cpp: tag `{out['commit']['tag']}` sha `{out['commit']['sha']}`")
md.append(f"- device (both): `{axc_device}` | ICD `{out['icd']}` | device_match byte-identical: {out['device_match']['byte_identical']}")
md.append(f"- K contraction (both): {axc_k}")
md.append("")
if is_fused:
    _acc = "f32-accum" if is_f32acc else "f16-accum"
    _axiom_label = (f"AXIOM (f32-accumulator fused Q4_K_M RB coopmat GEMM)" if is_f32acc
                    else "AXIOM (fused Q4_K_M RB coopmat GEMM)")
    md.append(f"| metric | {_axiom_label} | llama.cpp (Q4_K MUL_MAT n=512) |")
    md.append("|---|---|---|")
    md.append(f"| shape (m,n,k) | ({axc_m_out},{axc_n_out},{axc_k}) | ({fmt(llama_m)},{fmt(llama_n)},{fmt(llama_k)}) |")
    md.append(f"| TFLOPS (GpuTimestamp MIN, kernel-only) | {fmt(axc_tflops_min, 3)} | {fmt(llama_sameshape_tflops, 2)} |")
    if is_f32acc:
        _mrd_label = "max-rel-diff vs f32-accum ref (combined, condition-aware — the gate)"
    else:
        _mrd_label = f"max-rel-diff vs {_acc} ref"
    md.append(f"| {_mrd_label} | {fmt(axc_max_rel_diff, 3) if axc_max_rel_diff is not None else 'n/a'} | (ggml is the ref) |")
    if is_f32acc and axc_raw_rel_diff is not None:
        md.append(f"| raw forward-error vs f32-accum ref (reporting only; ~1e-2 on cancellation "
                  f"outputs, identical-in-kind to llama.cpp's HMMA) | {fmt(axc_raw_rel_diff, 3)} | "
                  f"(ggml is the ref) |")
    md.append("")
    if numerical_validity is not None and not numerical_validity["valid_at_ab_shape"]:
        md.append(f"> **NUMERICALLY INVALID at the A/B shape** (max_rel_diff="
                  f"{fmt(axc_max_rel_diff, 3)} > frozen {FROZEN_TOL:g}). {numerical_validity['verdict']}")
        md.append("")
    md.append(f"- **SAME-SHAPE headline ratio (AXIOM fused GEMM / llama same-shape, basis `{headline_basis}`): "
              f"{fmt(out['headline_ratio'], 5)}** (llama {fmt(out['llama_over_axiom_speedup_tflops'], 2)}x faster"
              + ("" if (numerical_validity and numerical_validity['valid_at_ab_shape']) else
                 ", on a NUMERICALLY INVALID AXIOM result — throughput-only, NOT a usable-kernel win")
              + ")")
    if cross_shape_context:
        md.append(f"- Cross-shape CONTEXT (NOT the kill criterion): AXIOM fused GEMM {fmt(cross_shape_context.get('axiom_gemm_tflops'),3)} TFLOPS "
                  f"vs llama n=1 GEMV {fmt(cross_shape_context.get('llama_n1_gemv_tflops'),2)} TFLOPS "
                  f"(ratio {fmt(cross_shape_context.get('axiom_over_llama_n1_ratio'),3)}) — labeled cross-shape, never the headline.")
else:
    md.append("| metric | AXIOM (M2.6 single-row matvec) | llama.cpp (Q4_K MUL_MAT n=1) |")
    md.append("|---|---|---|")
    md.append(f"| output rows | 1 | {fmt(llama_m)} |")
    md.append(f"| us/dispatch (GpuTimestamp MIN) | {fmt(axc_us_min)} | (CPU-wall) |")
    md.append(f"| us/op (sustained CPU-wall) | {fmt(axc_us_sustained)} | {fmt(llama_us, 2)} |")
    md.append(f"| TFLOPS (GpuTimestamp MIN) | {fmt(axc_tflops_min, 4)} | — |")
    md.append(f"| TFLOPS (sustained CPU-wall) | {fmt(axc_tflops_sustained, 4)} | {fmt(llama_tflops, 2)} |")
    md.append("")
    md.append(f"- **Headline ratio (AXIOM/llama, work-normalized TFLOPS, matched sustained boundary): {fmt(out['ratio_axiom_over_llama_tflops'], 5)}** (llama.cpp {fmt(out['llama_over_axiom_speedup_tflops'], 2)}x faster)")
md.append(f"- FLOP-consistency: ok={flop_consistency.get('ok')} (recomputed {fmt(flop_consistency.get('recomputed_tflops'), 3)} vs reported {fmt(flop_consistency.get('reported_tflops'), 3)} TFLOPS)")
md.append(f"- **Kill-criterion (DESIGN §5, within 15% on NVIDIA): {kill_status}**" + (f" — {kill_reason}" if kill_reason else ""))
md.append(f"- Qualifier: {out['kill_criterion_qualifier']}")
md.append(f"- Fairness caveat: {out['fairness_caveat']}")
md.append(f"- Gap-closing path: {out['gap_closing_path']}")
md.append("")
_repro_flag = " --fused-f32acc" if is_f32acc else " --fused" if is_fused else ""
md.append(f"Reproduce: `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json AXC_ENABLE_GPU_BENCHES=1 scripts/m34_llamacpp_ab.sh{_repro_flag}`")
md.append("")
with open(env("M34_RESULTS_MD"), "w") as f:
    f.write("\n".join(md))

# Console summary.
_tmin = axc_tflops_min if axc_tflops_min is not None else 0.0
_tsus = axc_tflops_sustained if axc_tflops_sustained is not None else 0.0
print("")
print("=" * 78)
if is_fused:
    _hdr = ("M3.5b A/B RESULT — f32-ACCUMULATOR FUSED kernel, SAME-SHAPE" if is_f32acc
            else "M3.5 A/B RESULT — FUSED kernel, SAME-SHAPE")
    print(f"{_hdr} (NVIDIA RTX PRO 6000, same ICD, kernel-only):")
    print("-" * 78)
    print(f"  AXIOM   : fused GEMM ({axc_m_out}x{axc_n_out}x{axc_k}) = {_tmin:.3f} TFLOPS "
          f"(GpuTs MIN) | max_rel_diff={axc_max_rel_diff if axc_max_rel_diff is not None else 'n/a'}")
    if llama_sameshape_tflops is not None:
        print(f"  llama   : same-shape Q4_K MUL_MAT = {llama_sameshape_tflops:.2f} TFLOPS "
              f"[m={llama_m}, n={llama_n}, k={llama_k}]")
    if ratio_tflops is not None:
        _nv_tag = ("usable-kernel comparison"
                   if (numerical_validity and numerical_validity["valid_at_ab_shape"])
                   else "NUMERICALLY INVALID AXIOM result — fast-but-WRONG at this K, NOT a win")
        print(f"  ratio   : AXIOM/llama SAME-SHAPE (TFLOPS) = {ratio_tflops:.5f}  => llama is "
              f"{1.0 / ratio_tflops:.1f}x faster (HONEST — {_nv_tag})")
    if cross_shape_context and cross_shape_context.get("llama_n1_gemv_tflops"):
        print(f"  context : (NOT the headline) llama n=1 GEMV = "
              f"{cross_shape_context['llama_n1_gemv_tflops']:.2f} TFLOPS — cross-shape only")
else:
    print("M3.4 A/B RESULT (NVIDIA RTX PRO 6000 Blackwell, same ICD, kernel-only):")
    print("-" * 78)
    print(f"  AXIOM   : {axc_us_min:.2f} us (GpuTs MIN) / {_tmin:.4f} TFLOPS | "
          f"sustained {axc_us_sustained:.2f} us / {_tsus:.4f} TFLOPS")
    if llama_us is not None:
        print(f"  llama   : {llama_us:.2f} us/op (CPU-wall sustained) / {llama_tflops:.2f} TFLOPS  "
              f"[m={llama_m}, n={llama_n}, k={llama_k}]")
        if ratio_tflops is not None:
            print(f"  ratio   : AXIOM/llama (TFLOPS) = {ratio_tflops:.5f}  => llama is {1.0 / ratio_tflops:.1f}x faster")
print(f"  KILL-CRITERION (within 15% on NVIDIA): {kill_status}" + (f" ({kill_reason})" if kill_reason else ""))
print("  Qualifier: NVIDIA-only; criterion is any-vendor; AMD/Intel pending hw.")
print("=" * 78)

sys.exit(1 if incomplete else 0)
