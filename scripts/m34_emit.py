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

# AXIOM derived numbers (work-normalized TFLOPS, dequant excluded).
axc_us_min = axc_ns_min / 1e3
axc_us_sustained = axc_sustained_ns / 1e3
axc_tflops_min = axc_flops / (axc_ns_min * 1e-9) / 1e12 if axc_ns_min else None
axc_tflops_sustained = axc_flops / (axc_sustained_ns * 1e-9) / 1e12 if axc_sustained_ns else None

flop_consistency = {"checked": False, "ok": None}
ratio_tflops = None
ratio_boundary_label = None

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
    else:
        # Like-for-like THROUGHPUT ratio. HEADLINE uses the matched (sustained CPU-wall)
        # boundary; the GpuTimestamp-MIN framing is reported alongside for disclosure.
        ratio_tflops = axc_tflops_sustained / llama_tflops
        ratio_boundary_label = (
            "AXIOM-sustained-CPU-wall (matched to llama.cpp sustained CPU-wall MEAN) vs "
            "llama.cpp sustained TFLOPS; GpuTimestamp-MIN framing reported alongside for "
            "disclosure")
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
    "milestone": "M3.4-llamacpp-ab",
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
        "note": ("AXIOM computes M=1 output row; llama.cpp computes m=4096 rows over the "
                 "SAME K=14336 contraction (the apples-to-oranges row-count caveat). "
                 "Throughput (TFLOPS) is the fair, work-normalized metric."),
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
        "flops_convention": "matmul-equivalent FLOPs, dequant EXCLUDED (2*M*K, M=1)",
        "note": ("single workgroup, ONE output row (the M2.6 'if i>=1 return' guard) — "
                 "~1 of ~188 SMs occupied"),
    },
    "llamacpp": {
        "selected_line": q4k_line,
        "us_per_run": llama_us, "tflops": llama_tflops, "n_runs": llama_runs,
        "m": llama_m, "n": llama_n, "k": llama_k,
    },
    "flop_consistency": flop_consistency,
    "ratio_boundary_label": ratio_boundary_label,
    "ratio_axiom_over_llama_tflops": round(ratio_tflops, 5) if ratio_tflops is not None else None,
    "llama_over_axiom_speedup_tflops": round(1.0 / ratio_tflops, 2) if ratio_tflops else None,
    "kill_criterion_within_15pct": (kill_status == "PASS") if kill_status in ("PASS", "FAIL") else None,
    "kill_criterion_status": kill_status,
    "kill_criterion_reason": kill_reason,
    "kill_criterion_qualifier": ("NVIDIA-only FAIL with the current FROZEN M2.6 single-row "
                                 "matvec is the documented baseline; DESIGN §5 kill-criterion "
                                 "is 'within 15% on ANY vendor', so this does NOT fire the "
                                 "project kill-criterion. AMD/Intel halves pending cross-vendor "
                                 "hardware (EB.1)."),
    "fairness_caveat": ("Same machine, same ICD, kernel-only-vs-kernel-only, identical "
                        "K=14336 contraction, FLOP convention identical (2*m*n*k matmul MACs, "
                        "dequant excluded). BOTH kernels are CORRECT (AXIOM bit-exact vs ggml "
                        "CPU ref; llama.cpp IS ggml). AXIOM is single-row matvec (1 workgroup) "
                        "vs llama.cpp's tiled multi-row MUL_MAT (all SMs) — ~structurally 100x "
                        "under-parallelized. Ratio reported AS MEASURED; honest expected FAIL."),
    "gap_closing_path": ("Fuse the Q4_K_M dequant front-end onto the M3.3c register-blocked "
                         "coopmat matmul (dequant -> shared f16 tile -> coopmat mul_add; "
                         "plain-f16 reached 31.2 TFLOPS = 24.96% of datasheet). Follow-up "
                         "milestone."),
}

with open(env("M34_RESULTS_JSON"), "w") as f:
    json.dump(out, f, indent=2)
    f.write("\n")


def fmt(x, d=3):
    if x is None:
        return "n/a"
    return f"{x:.{d}f}" if isinstance(x, float) else str(x)


md = []
md.append("# M3.4 — llama.cpp Vulkan Q4_K_M A/B (NVIDIA RTX PRO 6000)\n")
md.append(f"- llama.cpp: tag `{out['commit']['tag']}` sha `{out['commit']['sha']}`")
md.append(f"- device (both): `{axc_device}` | ICD `{out['icd']}` | device_match byte-identical: {out['device_match']['byte_identical']}")
md.append(f"- K contraction (both): {axc_k}")
md.append("")
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
md.append("Reproduce: `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json AXC_ENABLE_GPU_BENCHES=1 scripts/m34_llamacpp_ab.sh`")
md.append("")
with open(env("M34_RESULTS_MD"), "w") as f:
    f.write("\n".join(md))

# Console summary.
print("")
print("=" * 78)
print("M3.4 A/B RESULT (NVIDIA RTX PRO 6000 Blackwell, same ICD, kernel-only):")
print("-" * 78)
print(f"  AXIOM   : {axc_us_min:.2f} us (GpuTs MIN) / {axc_tflops_min:.4f} TFLOPS | "
      f"sustained {axc_us_sustained:.2f} us / {axc_tflops_sustained:.4f} TFLOPS")
if llama_us is not None:
    print(f"  llama   : {llama_us:.2f} us/op (CPU-wall sustained) / {llama_tflops:.2f} TFLOPS  "
          f"[m={llama_m}, n={llama_n}, k={llama_k}]")
    if ratio_tflops is not None:
        print(f"  ratio   : AXIOM/llama (TFLOPS) = {ratio_tflops:.5f}  => llama is {1.0 / ratio_tflops:.1f}x faster")
print(f"  KILL-CRITERION (within 15% on NVIDIA): {kill_status}" + (f" ({kill_reason})" if kill_reason else ""))
print("  Qualifier: NVIDIA-only; criterion is any-vendor; AMD/Intel pending hw.")
print("=" * 78)

sys.exit(1 if incomplete else 0)
