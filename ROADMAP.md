# AXIOM-Compute Roadmap

This document is the **comprehensive remaining work plan** as of 2026-04-28. The project's first 13 milestones (M0 → M2.6) are merged on `main`; this plan covers what's left to reach the thesis claims in `DESIGN.md` §5 (kill criteria) and beyond.

Last updated: 2026-04-28. Test count baseline: **713**.

---

## Status snapshot

| Capability | Status |
|---|---|
| Compile `.axc` → SPIR-V → spirv-tools::val | ✅ done (M0–M1.4) |
| Vulkan dispatch on real GPU | ✅ done (M1.5) |
| `@strategy` holes + grid-search autotuner | ✅ done (M2.3) |
| MCP server for LLM agents | ✅ done (M2.4) |
| Q4_0 + Q4_K_M kernels bit-exact on NVIDIA RTX PRO 6000 | ✅ done (M2.5–M2.6) |
| Cooperative matrix codegen | ✅ done (M2.1) — dispatch on tensor-core HW deferred |
| Bench harness + measured baselines | ✅ done (M2.2) |
| **FlashAttention-2** | ✅ COMPLETE — streaming online-softmax, fully coopmat-accelerated on NVIDIA. M3.2b (scalar streaming softmax, no S materialization) → M3.2c-exp (real `exp()` builtin, the FIRST GLSL.std.450 ext-inst → real-range correct) → M3.2c-perf (coopmat QKᵀ + 16-row query tile) → M3.2c-PV (coopmat P·V too, 1.385× over scalar-PV). Correct within FROZEN 1e-3 vs a true-exp softmax oracle. M3.2c-optC (full-coopmat acc) RESOLVED as BLOCKED by the KHR coopmat use-type system (§3.1.34) — M3.2c-PV's acc-in-shared is the correct design. Registered as `torch.ops.axiom.flash_attention` (M4.1 Phase 4) + auto-lowered under `torch.compile(backend="axiom")` (M4.3). AT-1738..2235 |
| **KernelBench-Vulkan public submission** | ⏸️ remaining — would need AXIOM kernels per task (we have matmul + attention + a few primitives); a subset harness would largely re-present the existing honest A/Bs. Low marginal value vs the hardware-gated work below |
| **PyTorch frontend + upstream adoption** | ✅ M4.1 COMPLETE (Phases 1-3 MERGED) — CUDA↔Vulkan ZERO-COPY proven on NVIDIA: torch CUDA tensor + AXIOM Vulkan kernel share one dedicated physical alloc (no host copy), via external-memory-fd + timeline external semaphore (spike works on 580.x). saxpy BIT-EXACT zero-copy; M3.6 Q4_K_M matmul as the headline op (AT-2103 combined ≤ 1e-3 at K=256/512/14336; AT-2105 honest latency); race-free; double-free/UAF-safe; fail-closed UUID. `crates/axc-py` PyO3 + `py/axiom_compute` (pip/maturin). ✅ Phase 3 DELIVERED (§3.1.27, r2 — both design reviewers APPROVED; GPU-verified on NVIDIA RTX PRO 6000 / 580.x) — `torch.library.custom_op` (`torch.ops.axiom.q4km_matmul`) + `register_fake` composes under `torch.compile(fullgraph=True)` (graph_break_count=0, combined 5.3e-7): caller (M,N,K) fail-closed-validated so fake==real shape (R1, AT-2111c); op runs on the session's captured stream with a cross-stream hand-back (R3, AT-2111d); ≤10-line user demo (AT-2116); opcheck passes (AT-2117); wait-timeout residual CLOSED on BOTH sides — bounded `wait_completion` + host-signal the timeline to V2 (`vkSignalSemaphore`) releases the dangling CUDA wait so stream S unblocks + poison/rebuild (AT-2114 PASS — the both-sides deadlock fix works on 580.x), happy path unchanged (AT-2115). DEFERRED to M4.2: cross-vendor (AMD/Intel), NVIDIA-only EXCLUSIVE-sync residual, the binary-pair host-signal-release analogue. Honest: ~53% cuBLAS, win is no-host-copy + real torch interop + torch.compile composition not speed; zero-copy claim EAGER-scoped (functionalization forces device_copy under compile). AT-2100..2118 |
| Bandwidth optimization (pinned memory, concurrent transfer) | ✅ M3.0 (saxpy_1m 7.5×, saxpy_1024 39×; <1ms gate re-scoped to GPU-resident metric) |
| Multi-row tiled matmul (cooperative_matrix on real workloads) | ✅ M3.1 (first coopmat dispatch on Blackwell, bit-exact; resident-TFLOPS benchmark → M3.2) |
| Cross-vendor real GPU CI (AMD RDNA3+, Intel Arc) | ⛔ HARDWARE-GATED (EB.1) — this box has only NVIDIA. The SINGLE blocker for BOTH the project's cross-vendor kill-criterion ("beat by ≥25% where llama.cpp Vulkan is weak") AND a compelling upstream PR (M4.2). EB.2 (per-machine baselines, MERGED) made the bench-regression harness cross-vendor-ready (each machine keyed by sanitized deviceName, blessed independently). M4.2 upstream RFC is PREPARED-AND-HELD (`UPSTREAM_PR_PLAN.md`): integration into llama.cpp Vulkan is feasible-but-not-drop-in, but AXIOM is 2.39× behind on NVIDIA (architectural) and the AMD/Intel value prop is unmeasured → don't submit until an AMD/Intel A/B shows parity-or-better from one unchanged source |
| **llama.cpp Vulkan head-to-head A/B** | NVIDIA done (M3.4 matvec; M3.5 fused SAME-SHAPE; M3.5b fused f32-accumulator SAME-SHAPE, now CORRECT). M3.4: AXIOM single-row matvec ≈ 87,000× below llama Q4_K n=1 GEMV (cross-shape baseline). M3.5: fused Q4_K_M dequant→RB coopmat GEMM SAME-SHAPE (AXIOM **11.27 TFLOPS** vs llama **101.48 TFLOPS** @ m=4096,n=512,k=14336) → ~9× behind BUT fast-but-WRONG (f16 accumulator, max-rel-diff 29.07 at k=14336). **M3.5b: f32-accumulator fused GEMM — NUMERICALLY VALID at inference K, the now-fast-AND-correct fight. MEASURED NVIDIA RTX PRO 6000: AXIOM 10.91 TFLOPS vs llama 101.88 @ m=4096,n=512,k=14336 → 9.3× behind (ratio 0.107), combined condition-aware max-rel-diff 2.05e-6 ≤ FROZEN 1e-3 = VALID (f16 was 29.07 garbage at same K; f32 tax only ~3% vs M3.5's 11.27). cube sizes all VALID: 256=0.59, 512=2.74, 768=4.81, 1024=7.93 TFLOPS. `--fused-f32acc` → ab_results_fused_f32acc.json; numerically_valid driven by MEASURED combined metric, never hardcoded.** Kill-criterion still FAIL on NVIDIA (9.3× behind on throughput) but a GENUINE usable baseline; AMD/Intel pending hw |
| **Q4_K_M coopmat fusion (M3.5 → M3.5b)** | ✅ landed — M3.5: `f32_to_f16` builtin + fused `q4km_matmul_rb_coopmat.axc` (f16 accumulator, K-LIMITED: correct only at K≤256, garbage at k=14336). **✅ M3.5b: f32-accumulator fused `q4km_matmul_rb_coopmat_f32acc.axc` (C buffer[f16]→f32, 4 accumulators matrix[f16,..]→matrix[f32,..], else VERBATIM) — the canonical f16×f16→f32 HMMA, NUMERICALLY VALID at inference K (AT-1780/1781/1782 within FROZEN 1e-3 at K=256/512/14336; 1e-3 NOT loosened). AUDIT: ZERO production coopmat code changed (M3.1 design dividend — shapes already carry independent a/b/c/result types); AT-1787 proves no new capability + identical cap set + metadata {16,16,16,F16,F16,F32,F32,Subgroup}. CPU-only regressions AT-1785/1786/1788/1789. f32-accum oracle `common_q4km_f32ref.rs`.** |
| **Multi-subgroup warptile (M3.9)** | ⚠️ HONEST-NEGATIVE (merged as documented experiment, NOT leader) — THE PIVOT's first compiler-feature milestone ("fix the compiler" campaign: AXIOM emits a weaker *algorithm* than llama, not weaker SPIR-V → add the missing structure). `matmul_warptile_coopmat.axc`: 4 subgroups `@workgroup(128,1,1)` cooperating on a 64×64 tile, 2×2 disjoint coopmat sub-tiles (M3.6 register-blocking per subgroup), cross-subgroup cooperative staging (K-independent, all 128 threads) + **Workgroup-scope** barriers (pessimistic code review verified at the SPIR-V level: `OpControlBarrier %uint_2 %uint_2 %uint_264`, depth-0, no divergent context). **PURE SOURCE, zero codegen change** (over the proven M3.3d cross-subgroup + M3.6 RB machinery). **BIT-IDENTICAL to single-subgroup RB (AT-2305 n_diff=0/589824 on an asymmetric 144-workgroup cross-subgroup-race fixture — partition + barriers provably correct).** **MEASURED NVIDIA: 0.955× A/B, 0.98× @768³, 0.86–0.93× elsewhere vs single-subgroup RB — the ≥1.3× gate MISSED at every size, NOT loosened; frozen 1e-3 untouched.** Diagnosis → M3.10: the M3.3d trap was *avoided* (warps-in-flight equalized 2048==2048, staging amortized 4×), so the residual cost is NOT staging volume / warp occupancy — it's cross-subgroup `workgroup_barrier` sync + 4-warp-per-WG SM co-residency + **un-vectorized scalar staging**. The warptile ALONE isn't the win; closing the 2.39× gap needs vectorized loads (`LOAD_VEC`) + shared-memory bank padding (`SHMEM_STRIDE`) = **M3.10**. Single-subgroup RB + M3.6 remain production. AT-2300..2308. |
| **Dequant front-end CLOSE-OUT (M3.13)** | ✅ DIAGNOSE-FIRST two-pronged close-out of the dominant **1.77×** dequant front-end — likely the autonomous performance campaign's **final substantive milestone**. **PRONG A = a PURE-SOURCE live-range-tightened kernel** (`q4km_matmul_rb_coopmat_f32acc_cached_tightlive.axc` = the M3.6 leader VERBATIM except the A-staging inner loop is restructured by SOURCE-LEVEL reorder/inline only: just-in-time `dsc`/`dmm` scale-cache reads [moved down to immediately before the consuming dequant] + inlined single-use `cache_idx`/`qs_base` lets — **BIT-IDENTICAL** to M3.6, ZERO codegen change) testing M3.12's HEDGED register-pressure attribution directly. **MEASURED (CPU, AT-2800): the A-staging temp-count proxy did NOT drop — M3.6 leader 86 → tightlive 87 (ticked UP by 1) ⇒ HONEST-NEGATIVE-(a): AXIOM's naive SSA construction is SOURCE-ORDER-INSENSITIVE (one result-inst per distinct value regardless of `let` order) ⇒ no codegen leverage.** Cap-set BYTE-IDENTICAL to M3.6 (AT-2806), `shared_memory_bytes==4096`, barrier/OpPhi counts == leader, pure-source / empty-codegen-diff (AT-2807), spirv-val clean. GPU gates (orchestrator, real NVIDIA): AT-2803 bit-identity (`n_diff==0` on the asymmetric fixture, K=256/512/14336) + AT-2805 combined ≤ FROZEN 1e-3 + AT-2802 TFLOPS vs M3.6 (gate ≥1.15×, NEVER loosened); given the proxy did not drop AND AXIOM has no scheduler/register-allocator (the driver re-allocates SASS from the dataflow graph), **flat TFLOPS is the predicted base case** — a flat result NARROWLY closes "register-pressure as a SOURCE-ADDRESSABLE lever" but does NOT falsify the register-pressure pinpoint (it stays a hedged static proxy). `--tightlive` → `ab_results_tightlive.json` (AT-2802/2804). **PRONG B = the FUSED-DEQUANT-COOPMAT-LOAD feasibility VERDICT (paper finding, AT-2810, AT-2230-class — NO kernel built): BLOCKED by VK_KHR_cooperative_matrix fragment OPACITY.** AXIOM exposes EXACTLY the 4 coopmat builtins {Zero/Load/Store/MulAdd}; NO per-element coopmat-fragment-write op exists (convert/recast/cast/extract/set/write_elem all parse to None); `coopmat_mul_add` always yields an Accumulator-use result; `coopmat_load` reads FROM a memory source (forces the M3.6 shared round-trip) ⇒ register-resident dequant-into-fragment is UNEXPRESSIBLE portably — identical class to §3.1.34 M3.2c-optC. NV coopmat2 (per-element/explicit-layout) is the ONLY in-register path but is NV-ONLY → OUT of scope (breaks the portability thesis), flagged for the EB.1 cross-vendor future. **CAMPAIGN CONCLUSION (thesis-relevant, honest):** PRONG A no portable register lever + PRONG B KHR-blocked ⇒ AXIOM's portable coopmat Q4_K_M kernel is at the **VK_KHR_cooperative_matrix ACHIEVABLE CEILING**; the residual 2.39× is the **PORTABLE-COOPMAT TAX** (opaque-fragment-forced shared round-trip + dequant register pressure that llama's vendor-tuned fused-into-register avoids); closing it needs non-portable vendor extensions (NV coopmat2) or AMD/Intel hardware (EB.1) where portability is the value prop — NOT a missing simple feature. AXIOM matched every PORTABLE structural lever (M3.3c tiling, M3.6 occupancy + scale-caching, M3.10a bank-padding, M3.11 strength-reduction, M3.9 warptile); the final gap is the portability-vs-vendor-tuning tradeoff. NOT a failure. M3.6 stays leader; FROZEN 1e-3 + production kernels UNCHANGED; PRONG B builds no kernel. AT-2800..2810. |
| **Dequant front-end ABLATION DIAGNOSTIC (M3.12)** | ✅ DIAGNOSE-FIRST, measure-only — PINPOINTS the re-opened dominant **1.77×** dequant front-end tax (M3.11 ruled out the integer index-decode as latency-hidden). **PURE-SOURCE PROFILING INSTRUMENTS (WRONG OUTPUT BY DESIGN, NO correctness gate, ZERO codegen)** hold the M3.6 matmul + B-path + shared-A-stage + 3 barriers + coopmat + the scale-cache FILL CONSTANT and vary ONLY the per-element A-dequant value. Built **V-structure-probe** (`…_ablate_structonly.axc`, near-free per-element-index A-value, full M3.6 structure) + **V-passthrough** (`…_ablate_passthrough.axc`, f32_to_f16-retaining nibble pass-through, drops the scale-reads+FMul/FSub). V-no-scale-read SUBSUMED (same op-mix); V-no-convert DROPPED (no integer-route-to-f16 — no builtin added, hard rule #5; `{convert OR structure}` residual reported honestly). RE-ANCHORED triangulation (V-structure-probe = PRIMARY denominator) + a per-variant **register/temp-count proxy** (the r3 in-bucket register-occupancy cross-check). **MEASURED NVIDIA RTX PRO 6000 (A/B 4096×512×14336, MIN-of-10 GpuTimestamp):** M3.6 leader **42.8** (≡42.86) | V-passthrough **~62.0** (arith_share ~35%) | **V-structure-probe ~96.5 (PRIMARY)** | plain-f32 core 75.5 live / 75.8 merged (read-with-source). temp-count proxy 86/68/**10** (probe-vs-passthrough delta 58 = LARGE). **PINPOINT (a design-premise inversion, reported honestly): V-structure-probe (96.5) EXCEEDS the plain-f32 core (75.8)** ⇒ the **structural bucket is ≤ 0 — the 4 KB scale-cache + 3-barrier structure is NOT the gap** (with a near-free A-value it is *faster* than the leaner core); **the ENTIRE 1.77× is the per-element A-side DEQUANT WORK**, and the LARGE temp-count delta flags that much of it is **REGISTER-PRESSURE/OCCUPANCY** (live dequant-chain temporaries cut resident warps on this register-limited kernel — M3.7/M3.8), not pure arithmetic latency. **CONSISTENT with M3.10b (load ~3%) + M3.11 (index-decode latency-hidden)** — V-passthrough keeps the load+decode and still lands far below V-structure-probe ⇒ residual is the float-dequant chain + scale-reads, NOT the load/address (no escalation). **NEXT-MILESTONE TARGET = the FUSED-DEQUANT-COOPMAT-LOAD** (dequant directly into the coopmat fragment — eliminates BOTH the dequant-chain live temporaries AND the shared-A round-trip in one rewrite, like llama's warptile; both the arithmetic-feature and the register/occupancy lever fold into it; §3.1.34 KHR coopmat use-type, likely EB.1-gated). NO feature built (DIAGNOSE-ONLY); M3.6 stays leader; FROZEN 1e-3 + production kernels UNCHANGED. `--ablate`. AT-2700..2706. |
| **Dequant-index strength-reduction (M3.11)** | ⚠️ HONEST-NEGATIVE for the primary hypothesis (latency-hidden), MERGED — **MEASURED NVIDIA: the strength-reduction worked (emitted ALU 34→29 incl. the ×144 IMul gone, BIT-IDENTICAL to M3.6, pure source) but TFLOPS only 1.048× A/B (gate ≥1.15× MISSED, not loosened; ~1.04-1.07× per-size, never slower).** PROFOUND FINDING: the dequant integer ALU is LATENCY-HIDDEN under the HMMA pipeline (concurrent int/SFU issue) → **the dominant 1.77× tax is NOT the integer index-decode; the M3.10b decomposition RE-OPENS** (remaining suspects: the f32_to_f16 converts / the dequant-A-tile shared stores / the coopmat issue rate). M3.11b (codegen IndVarSimplify pass) = NO-GO (same latency-hidden ceiling). M3.6 stays leader (1.048× sub-gate, not re-baselined). The next milestone targets the re-opened suspects, not the integer ALU. — the FIRST lever on the DOMINANT 1.77× dequant front-end gap (the "fix the compiler" campaign's classic IndVarSimplify / operator-strength-reduction). **M3.11a = pure-source hand-strength-reduced kernel** (`q4km_matmul_rb_coopmat_f32acc_cached_sr.axc` = the M3.6 leader VERBATIM except the per-A-element affine integer index-decode → **3 loop-carried scalar counters**: `sb_idx_c`, `sb_byte_base_c` += 144 on the 16-wrap [the non-pow2 ×144 superblock-stride IMul prize], `wc0_c` += 16 / wrap-to-0). Declared BEFORE the `for k_block` loop, Assign-reassigned ONLY inside (the declare-before-loop carry invariant — a loop-body re-`let` would silently destroy the carry). Rides the proven scalar Function-storage OpVariable carry (NOT the coopmat OpPhi path), so **ZERO codegen/HIR/typecheck/parser/lexer change** (AT-2607). **Static results (CPU): AT-2600 — emitted per-element integer-index ALU dropped 5 ops (M3.6 34→SR 29), incl. OpIMul 13→10 (the ×144 prize + g_k-seed multiplies → carried OpIAdd) + OpUDiv 5→4 (sb division carried); AT-2601 — carried recurrence == div/mod decode over the full K=14336 range incl. every superblock wrap, zero mismatches, tile_k==16 asserted; AT-2606 — spirv-val clean + cap set BYTE-IDENTICAL to the M3.6 leader.** GPU AT-2603 (bit-identity vs M3.6 on the asymmetric fixture) + AT-2605 (combined ≤ FROZEN 1e-3) + AT-2602/2604 (A/B + 768³ TFLOPS gate ≥1.15× / armed honest-negative) are the orchestrator's real-NVIDIA gates. **ARMED HONEST-NEGATIVE (the LIKELY base case):** ALU-down-but-TFLOPS-flat ⇒ the dequant integer ALU is latency-hidden under the HMMA pipeline (issued concurrently on int/SFU) ⇒ NOT on the critical path ⇒ the 1.77× tax is NOT the integer index-decode (decomposition re-opens) ⇒ M3.11b NO-GO; a ≥1.15× win is the surprising upside. **M3.11b = a general codegen IndVarSimplify pass, CONDITIONAL on a measured M3.11a ≥1.15× signal, DEFERRED.** FROZEN 1e-3 untouched. `--sr`. AT-2600..2607. |
| **Memory-subsystem levers (M3.10)** | ⚠️ HONEST-NEGATIVE (M3.10a merged as documented experiment, NOT leader; M3.10b r2 RESOLVED NO-BUILD — corrected same-shape: dequant front-end is the dominant 1.77× gap and is ALU/issue-bound) — the campaign's second compiler-feature milestone, targeting the bottleneck M3.7/M3.8/M3.9 isolated by elimination. **M3.10a = pure-source bank padding** (`matmul_rb_coopmat_pad.axc` + `q4km_..._cached_pad.axc`): padded shared tile + padded write index + padded coopmat_load stride, **ZERO codegen change** (codegen/hir/parser/lexer diff EMPTY — verified). Pessimistic code review confirmed in the EMITTED SPIR-V: `OpTypeCooperativeMatrixKHR` stays **16×16**, only the Stride operand carries the pad (A `stride=24` offsets `0`/`384`; B `stride=40` offsets `0`/**`16`** — B second-tile offset correctly unscaled), write index matches read stride. **BIT-IDENTICAL / combined-VALID** to the leaders (plain-f32 `n_diff=0/589824` asymmetric 768³; Q4_K_M combined **4.13e-7/4.67e-7/1.54e-6** at K=256/512/14336 ≤ FROZEN 1e-3; cap-set byte-identical, no new capability). **MEASURED NVIDIA: every PAD REGRESSES — PAD1=0.79–0.81×, PAD8=0.87–0.89× (best) vs the 31.6-TFLOPS base; ≥1.15× gate MISSED at every PAD, NOT loosened** (QA independently reproduced). **DIAGNOSIS — the matmul-core is NOT shared-bank-conflict-bound on Blackwell** (OQ-1: the coopmat tile-load already swizzles the 16-wide f16 conflict away; +pad footprint cuts occupancy = the M3.7/M3.3d mechanism). **This RULES OUT bank-conflicts** → combined with M3.7 (not latency), M3.8 (not arithmetic-intensity), M3.9/M3.3d (not warp-occupancy/staging-volume), the campaign has eliminated **every pure-source lever**. The one lever still standing was **vectorized-load instruction-count (M3.10b)**. **M3.10b = RESOLVED: NO-BUILD (DIAGNOSE-FIRST, §3.1.38, r2 2026-06-10)** — the packed-q u8 vec-load feature is NOT undertaken, established by a CORRECTED SAME-SHAPE gap decomposition + a static load-vs-ALU diagnostic. **r1 was SPLIT at design review (optimistic APPROVE, pessimistic NEEDS_REVISION) — its "3.24× core-dominant" claim cherry-picked the core's worst shape (768³, 31.2 TFLOPS) against llama@A/B (102.49). r2 RETRACTS it.** VERIFIED from the merged M3.9 bench (`ab_results_warptile.json`): the dequant-FREE plain-f32 core measured at the SAME A/B shape (4096×512×14336) = **~75.8 TFLOPS (60.6% datasheet)**, NOT 24.96%. **Corrected SAME-SHAPE decomposition:** matmul-core deficit = 102.49/75.8 = **1.35×** (near-ceiling — the pure-source negatives established that); dequant front-end tax = 75.8/42.86 = **1.77× (the DOMINANT gap, inverting r1)**. **D1** the M3.6 A-tile is a COMPUTED value `f32_to_f16(dsc·nib − dmm)`, NOT a global load ⇒ no f16 A-buffer to vectorize (only the packed-q u8 bytes are Q4_K_M-vectorizable) — SURVIVES. **D-dequant (decisive):** the packed-q u8 loads live in the dominant 1.77× front-end, but a static count of the A-staging region (`cached.axc:207-242`) shows **1 u8 `OpLoad` (`:232`) vs ~26 dequant ALU ops** (index-decode + `band`/`lshr` extract + `f32_from_u32`/FMul/FSub/`f32_to_f16`) ⇒ the front-end is **ALU/ISSUE-bound, not u8-load-bound**; a `u8vec4` view (16→4 loads) trims ~3% of the path's issue-slots and **cannot** move the 1.77× gap. **D3** M3.7 (0.97×) + M3.10a (0.79–0.89×) rule out the f16-staging vec-load by occupancy (the f16 path, distinct from the u8 path) — SURVIVES. **CAMPAIGN NARROWED, not over-claimed:** every single load/tile/occupancy/bank/vec-load feature is eliminated AT THE RIGHT SHAPE, but the dominant 1.77× dequant front-end is ALU/issue-bound — its remaining lever is reducing the dequant ALU cost via a fused vec-load-dequant **warptile** (a different architecture; M3.9 pure-source regressed at 0.955×), gated behind cross-vendor EB.1 — NOT a vec-load builtin. Frozen-144-byte-layout safety analyzed (qs @offset 16, 32×u8vec4, 4-aligned, M4.1 untouched) for the specified-but-not-taken build. Ships as the AT-2500..2504 static load-vs-ALU diagnostic + AT-2505 no-regression anchor (no codegen change, FROZEN 1e-3 untouched). Unpadded RB + M3.6 remain production. AT-2400..2412 (M3.10a), AT-2500..2505 (M3.10b r2). |
| **Larger register tiles (M3.8)** | ⚠️ HONEST-NEGATIVE (merged as documented experiment, NOT leader) — 4×4 (16 accumulators) + 4×2 (8) variants vs M3.6's 2×2. PURE SOURCE (N-coopmat-phi N-agnostic, OpPhi==16 verified). Both BIT-IDENTICAL to M3.6 (AT-2003 n_diff=0, grid-independent), combined ≤1e-3 VALID. **MEASURED NVIDIA: A/B 4×4=28.19, 4×2=26.31 vs M3.6 42.86 TFLOPS (0.66×/0.61×, gate MISSED); all sizes slower.** 16 accumulators = too much register pressure → fewer resident warps. **COMBINED M3.7+M3.8: M3.6 2×2 is the occupancy/compute SWEET SPOT — both latency-hiding AND arithmetic-intensity levers regress (occupancy-bound); remaining 2.39× not reachable by tile-tuning.** M3.6 remains leader. AT-2000..2006. |
| **Double-buffered staging (M3.7)** | ⚠️ HONEST-NEGATIVE (merged as documented experiment, NOT leader) — `q4km_matmul_rb_coopmat_f32acc_db.axc`: parity-indexed `shared[f16,1024]` A/B double buffers, prologue + prefetch-next, single flat K-loop (FILL→B1→STAGE→B2→load), 2-barrier ping-pong PROVEN race-free (AT-1903 db==M3.6 BIT-IDENTICAL at K=256/512/14336 + formal SPIR-V barrier trace). **PURE SOURCE, no new codegen.** **MEASURED NVIDIA: A/B 41.64 vs M3.6's 42.86 TFLOPS = 0.97× (REGRESSED ~3%; gate ≥1.15× MISSED); 768³ 13.0 vs 13.84.** The kernel is NOT latency-bound — doubled shared (6144 vs 4096 B/wg) cut occupancy, net loss (M3.3d mechanism). **M3.6 cached remains production leader.** Value: proves AXIOM expresses double-buffering correctly + rules out latency-hiding → next opt targets occupancy/compute, not staging. AT-1900..1907. |
| **Q4_K_M dequant scale-caching (M3.6)** | ✅ landed — `q4km_matmul_rb_coopmat_f32acc_cached.axc`: two `shared[f32,256]` caches hold the 8 (d·sc, dmin·m) pairs per superblock, filled ONCE via a workgroup-uniform `if (k_block%16==0)` cooperative fill + a hoisted unconditional RAW barrier (OPT-B; the `BarrierInDivergentContext` guard forbids barriers inside `if`), reused across 16 k_blocks. Flat single K-loop + single-level OpPhi VERBATIM from M3.5b (r1 nested restructure REJECTED at design review — would reset the accumulator per superblock). **PURE SOURCE, no new codegen/capability.** **MEASURED NVIDIA: 768³ 4.81→13.84 TFLOPS (2.88×, gate ≥1.5× CRUSHED); A/B 10.91→42.86 TFLOPS (3.93×) vs llama 102.49 → gap 9.3×→2.39× behind, combined 2.05e-6 VALID, BIT-IDENTICAL to M3.5b (AT-1803, zero correctness cost).** Dequant-bottleneck hypothesis CONFIRMED. AT-1800..1806. `--fused-f32acc-cached`. |

---

## Phase M3 — Performance + ML kernels (the thesis-closing phase)

Goal: prove the DESIGN.md §5 kill-criteria gates with publishable numbers, not just bit-exact correctness.

### M3.0 — Dispatch bandwidth rework ✅ DONE (gate re-scoped)

**Status (2026-06-01):** Merged. Persistent-mapped HOST_CACHED staging + optional dedicated transfer queue + timeline/binary-semaphore overlap, single-queue fallback byte-identical to M2.3a. Measured on NVIDIA RTX PRO 6000: `dispatch_saxpy_1m` 23 ms → **3.08 ms (7.5×)**, `dispatch_saxpy_1024` 1.22 ms → **31 µs (39×)**, `dispatch_q4km_512` 8.84 ms → **5.76 ms (1.5×)**. All paths byte-exact (AT-1418 four-config oracle on NVIDIA). 747 tests, codegen untouched.

**Gate re-scoped — the original `<1 ms` / `<2 ms` targets were not met and are deferred, for a documented reason.** Profiling showed the residual cost is host-round-trip PCIe transfer + readback, not kernel quality: `saxpy_1m` moves ~12 MB host→device→host every call. A ReBAR zero-copy-readback fix (r2) was attempted and **empirically reverted** — CPU reads from the BAR aperture are write-combined and ~60× *slower* (the real-GPU measurement gate caught it pre-merge; see DESIGN.md §3.1.12 postmortem). The thesis-relevant metric is a **GPU-resident benchmark** (upload once, dispatch N times, measure kernel time), re-scoped to M3.1/M3.4. Two correct-but-unshipped ideas carry forward to M3.1: Lever A (skip readback of `readonly` bindings) and the binary-semaphore-recreation fixup (Dev#1).

**Why first:** every measured GPU number on real hardware (NVIDIA RTX PRO 6000) is staging-bound at 1 M+ elements. saxpy_1m at 23 ms is ~100× off theoretical PCIe peak. Without fixing this, M3.1/M3.2/M4 numbers are dominated by infrastructure overhead, not kernel quality.

**Scope in:**
- Pinned host memory via `VK_EXT_host_memory_alloc_placement` or fallback `mmap` + `vkMapMemory` with `HOST_VISIBLE | HOST_CACHED`
- Concurrent transfer queue: dedicated transfer queue family if present, with semaphore handoff to compute queue
- `vkCmdCopyBuffer` overlap with compute (transfer queue submit pre-warms the device-local while compute runs on previous tile)
- `VK_KHR_buffer_device_address` for direct GPU memory access where supported (NVIDIA, AMD)
- Optional: persistent mapped staging ringbuffer (avoid per-dispatch map/unmap cost)

**Acceptance:**
- saxpy_1m on NVIDIA RTX PRO 6000 drops from 23 ms to **< 1 ms** (~25× speedup expected — closes most of the 100× peak gap)
- Q4_K_M_512 drops from 8.84 ms to **< 2 ms**
- Lavapipe perf unchanged (no PCIe → no benefit, but no regression)

**Effort:** ~2500–4000 LOC, 1 milestone cycle.
**Depends on:** M2.3a (already shipped pipeline cache + staging foundation).
**Blocks:** all subsequent perf claims.

### M3.1 — Multi-row tiled matmul + cooperative_matrix dispatch ✅ DONE (core proven; perf-methodology carried to M3.2)

**Status (2026-06-01):** Merged. **First-ever cooperative_matrix dispatch on real Blackwell tensor cores** — `matmul_tile.axc` (M2.1's compile-only SPIR-V, now executed) produces a 16×16 C=A·B tile **bit-exact (max_diff=0)** vs CPU reference on the NVIDIA RTX PRO 6000 (AT-1510). Required unlocking three device-feature classes never previously enabled: `VulkanMemoryModel` (the coopmat SPIR-V uses `OpMemoryModel Logical Vulkan`), `16BitStorage` (f16 SSBOs), `8BitStorage`+`shaderInt8/16` (q4km u8 weights) — all enabled **conditionally** (probe-then-enable, fail-closed `DeviceFeatureUnsupported`/`CoopMatUnsupported` skip), so Lavapipe and existing dispatch tests are unregressed. Multi-row `q4km_dequant_matmul` is **bit-exact 256×256** (AT-1520, runs on Lavapipe+NVIDIA); a pre-dequantized coopmat bridge proves coopmat-on-q4km (AT-1521, max_diff=0). Coopmat shape now flows HIR→metadata (schema v1→v2, back-compat). Lever A (skip `readonly` readback, deferred from M3.0 r2) landed cleanly without ReBAR. SPIR-V byte-identical to M2.1; spirv-val clean (AT-1560); a non-symmetric transpose fixture (AT-1512) guards against mis-strided loads. 780 tests.

**M3.1.5 (DONE, 2026-06-01):** completed the **GPU-resident benchmark** (`upload/dispatch/readback_resident` implemented; timestamp-query timing + warmup + CpuFenceWall fallback). Measured kernel-only spans on NVIDIA (all GpuTimestamp): **saxpy 2.0 µs** (vs 3.08 ms host-round-trip), **q4km matvec 131 µs**, **q4km matmul 1.9 µs** — proving the M3.0 dispatches are ~99.9% PCIe transfer, validating the host-round-trip gate re-scope. The dispatch-time preflight is wired via additive `prepare_kernel_checked`. **Honest finding:** `matmul_f32_tiled.axc` had inert `@strategy` holes (removed) — it's a naive un-tiled GEMM (4.5 TFLOPS ≈ 3.6% cuBLAS datasheet estimate, `naive_gemm_harness_validation` bench only).

**Still carried to M3.2 (honestly deferred):** a **competitive tiled/coopmat matmul TFLOPS** (needs `shared[T,N]` workgroup memory, FG.6) — the full **dequant→shared-f16-tile→coopmat fusion**; folding the preflight into raw `prepare_kernel` (opt-in gap remains). The `<2 ms` q4km gate from M3.0 also rides on this.

**Why:** M2.6 Q4_K_M is single-row matvec (1 output per dispatch). Real LLM inference needs N-row matmul with tensor cores. M2.1 added cooperative_matrix codegen but it's compile-only — never dispatched on tensor-core hardware.

**Scope in:**
- Multi-output kernel: `q4km_dequant_matmul(q: ..., x: ..., y: buffer[f32], n_rows: u32, n_cols: u32, n_blocks_per_row: u32)`
- Tile-parallel: each workgroup computes a 16×16 (or 32×32) tile of outputs
- Cooperative_matrix integration for the f16 inner dot product (after dequant)
- Per-vendor `@strategy` holes for tile dimensions (NVIDIA prefers 16×16×16, AMD WMMA is 16×16×16, Intel Xe-Cores want 8×16×16)
- Dispatch on NVIDIA RTX PRO 6000 (Blackwell tensor cores) — requires `VK_KHR_cooperative_matrix` device feature

**Acceptance:**
- 4096×4096 f32 matmul on NVIDIA: **≥ 50% of cuBLAS** (cuBLAS hits ~15 TFLOPS f32, Q4_K_M dequant + matmul should clear 7.5 TFLOPS effective)
- Bit-exact vs CPU reference for 256×256 fixture
- Same `.axc` compiles + runs on Lavapipe (graceful skip via cooperative_matrix preflight)

**Effort:** ~3000–5000 LOC.
**Depends on:** M3.0 (bandwidth), M2.1 cooperative matrix infra.
**Blocks:** M3.2.

### M3.2 — shared[T,N] workgroup memory ✅ LANDED; competitive matmul + attention → M3.3 (2026-06-03)

**LANDED — the `shared[T,N]` language feature (FG.6):**
- **FG.6 `shared[T,N]` IMPLEMENTED + GPU-VALIDATED**: full lexer→parser→HIR→typecheck→codegen→spirv-val pipeline. OQ1 SET-based missing-barrier analysis (provably zero false positives — verified by the pessimistic reviewer with 5 adversarial patterns). OQ2 `conditional_depth` divergent-barrier hard error. AT-429 inverted. CRITICAL-1/2/3/4 resolved across 3 design-revision cycles. **AT-1606: a shared[f32,256] parallel reduction + `workgroup_barrier()` runs BIT-EXACT on the NVIDIA RTX PRO 6000 (=384.0)** — the feature executes correctly on real hardware, not just compiles.
- **Coopmat-from-shared infrastructure**: `CoopMatLoadSource::{Buffer,Shared}` discriminator + `emit_coopmat_load_shared_inline`/`store_shared_inline` (single-index Workgroup access chain); existing Buffer-source coopmat SPIR-V byte-identical (AT-1613).
- **Metadata schema v3**: `SUPPORTED_SCHEMA_VERSIONS=[1,2,3]` (v1/v2 back-compat), `shared_memory_bytes`, `maxComputeSharedMemorySize` graceful-skip preflight.

**RE-SCOPED to M3.3 (honest — GPU-measured):** the competitive tiled matmul (`matmul_shared_coopmat.axc`, `matmul_shared_f32.axc`) and the tiled attention (`tiled_attention.axc`) **compile + spirv-val clean but compute INCORRECT results (zeros) on real GPU.** Root cause: `emit_for_range` lacks **OpPhi loop-carried SSA**, so a K-loop accumulator can't carry the coopmat/sum value across iterations — without it the matmul is single-tile and the kernels don't accumulate. Their bit-exact GPU tests were converted to compile/spirv-val-only (no zero-computing test ships as passing); the misleading TFLOPS bench was removed; the examples carry `WIP (M3.3)` headers. **M3.3 = OpPhi loop-carried SSA in `emit_for_range` → real multi-tile matmul (competitive TFLOPS) + working tiled attention.** FlashAttention-2 streaming softmax (C2) remains M3.2b after that.

**Acceptance (what passed):** 834 tests green; clippy -D warnings clean; AT-1606 GPU bit-exact on NVIDIA; AT-1613/1614 coopmat byte-identity + shared spirv-val; metadata v1/v2/v3 back-compat. Both code reviewers APPROVE.

**FG.6 status:** `shared[T,N]` is **IMPLEMENTED + GPU-validated** (see DESIGN.md §3.1.14).

**Effort:** ~4800 LOC. The new language feature landed; the kernels exploiting it need OpPhi (M3.3).
**Depends on:** M3.0, M3.1. **Blocks:** M3.3 (OpPhi + competitive matmul + attention), then M3.2b (C2 FA2).

### M3.3 — OpPhi loop-carried SSA (GPU-proven) ✅ LANDED; full coopmat matmul → follow-up (2026-06-03)

**LANDED + GPU-validated on NVIDIA RTX PRO 6000:**
- **PART A — OpPhi loop-carried SSA in `emit_for_range`**: emits an OpPhi at the loop header for loop-carried coopmat (SSA) accumulators (scalars stay on Function storage, unchanged). **AT-1707 PROVES it numerically**: a `acc = coopmat_mul_add(A,B,acc)` K-loop is **bit-exact = K·(A·B)** on NVIDIA — resolving the M3.2 blocker (coopmat accumulators reset to zero each iteration). AT-1701 spirv-val + phi-well-formed; AT-1700 confirms scalars still emit 0 phis. ISSUE-1 (both Assign arms route CoopMatrix targets through `check_coopmat_init_expr`), ISSUE-2 (Assign SSA-rebind branch), break/continue-over-coopmat hard error.
- **PART C — working tiled attention (AT-1630 bit-exact within 1e-3 on NVIDIA)** + **f32 tiled matmul (AT-1621 bit-exact)**: these were ZEROS in M3.2 due to *separate scalar-path kernel-logic bugs* (dispatch geometry / index math — NOT OpPhi). Debugged on the already-working Function-storage path. tiled_attention dispatches (seq_len,1,1); Taylor exp with matching CPU ref.

**DEFERRED to a follow-up (honest — GPU-measured):** the **full competitive coopmat matmul** (`matmul_shared_coopmat.axc`). The OpPhi K-accumulation works (AT-1707), but the kernel computes only a single 16×16 output tile, so a 16×**24** output (the test fixture) is wrong for N>16 — it needs a **multi-N-tile output loop** (and register/multi-warp blocking for real throughput). AT-1620/1622 are therefore **compile + spirv-val only (WIP stubs for the bit-exact GPU assertion)** — no wrong-computing test ships as passing. The competitive-TFLOPS bench (`resident_matmul_competitive.rs`) was **removed**: a TFLOPS number from a wrong-computing kernel would be misleading. **No TFLOPS is reported until the matmul is correct.** This is kernel-tiling work, not a compiler gap.

**Honest finding:** only the coopmat accumulator was OpPhi-blocked; the f32-matmul + attention zeros were independent kernel-logic bugs. 846 tests; both code reviews + QA confirm OpPhi correctness + AT-1707 load-bearing + the deferral honesty.

**Depends on:** M3.2. **Blocks:** full competitive matmul + M3.4 llama.cpp A/B (NVIDIA half needs the working tensor-core matmul).

### M3.3b — multi-tile coopmat matmul: bit-exact full matmul + honest effective-TFLOPS (2026-06-03)

**Root cause (corrected from M3.3 deferral):** M3.3 misdiagnosed the multi-tile gap as "needs a multi-N-tile output loop." The actual causes were: (a) test dispatched (1,1,1) against N=24 (only tile (0,0) ran); (b) kernel recovered `tile_col = gid(0)` directly — the GLOBAL invocation id, not the output-tile index. The idiomatic GPU tiling: ONE workgroup == ONE output tile; dispatch a GRID of workgroups (the grid IS the output-tile loop). No per-kernel output loop needed.

**Kernel fix (one line, ASYMMETRIC):**
- `tile_col = gid(0) / 32` (divide by local_size.x=32; all 32 lanes of a workgroup collapse to one tile_col).
- `tile_row = gid(1)` — UNCHANGED (local_size.y=1 so gid(1)==workgroup_id.y; NO division).
- Dispatch: (N/16, M/16, 1) workgroups.
- NO codegen change. OpUDiv already emitted; all builtins and coopmat paths GPU-proven in M3.2/M3.3.

**AT-1620/1622 UN-STUBBED (bit-exact full matmul on NVIDIA):**
- Non-symmetric multiple-of-16 fixture: M=32, N=48, K=32 (3x2 workgroup grid, 2 K-blocks).
- Integer-valued f16 (A∈{1..4}, B∈{1..3}; per-element sum ≤ 384, f16-exact) → max_diff == 0.0.
- **AT-1620 bit-exact (max_diff=0) on NVIDIA RTX PRO 6000** — the full multi-tile matmul works.
- `tile_k` is bound to the coopmat K dimension (**16**): one `coopmat_mul_add` consumes exactly K=16, so `tile_k=32` is semantically invalid (it computed exactly HALF — a single coopmat op over a 32-wide K-block reads only K=0..15). AT-1622 therefore varies the **K-block COUNT** (K=32 → 2 blocks, K=48 → 3 blocks; tile_k=16 fixed) — both **bit-exact**, genuinely exercising the OpPhi K-loop. A `tile_k>16` sub-K-loop is a follow-up.
- Lavapipe: typed-skip (CoopMatUnsupported); matmul_shared_f32.axc (AT-1621) unaffected.

**AT-1710 honest effective-TFLOPS (resident_matmul_competitive.rs re-added):**
- Large matmul M=N=K=256 (default), full 16x16 tile grid, upload-once/dispatch-N (MIN-of-10, GpuTimestamp).
- Reports effective_tflops = 2·M·N·K / kernel_ns + % of 125-TFLOPS f32 datasheet ESTIMATE.
- NO ratio asserted (only tflops>0 && finite). With single 32-lane subgroup per 16x16 tile, throughput is MODEST — that is the honest deliverable.
- "competitive" label omitted unless measured % >= 25%. CpuFenceWall path omits % + carries scheduling-inclusive qualifier.
- 2D grid pre-check: both max_compute_work_group_count()[0] AND [1] checked; shrink M and N together if needed.

**Out of scope (follow-up M3.3c):** partial edge tiles (M or N not a multiple of 16); K not a multiple of tile_k.

**Depends on:** M3.3. **Blocks:** M3.4 llama.cpp A/B (tensor-core matmul now correctly computes full outputs).

### M3.3c — register-blocked coopmat matmul (bit-exact + measured TFLOPS uplift) (2026-06-04)

**Root cause (M3.3b bottleneck):** ONE workgroup = ONE 16×16 output tile = ONE subgroup = ONE coopmat accumulator. Tensor cores idle on shared staging + barriers (issue/bandwidth-bound, not FLOP-bound). 5.04 TFLOPS = 4.0% of datasheet.

**Register blocking (matmul_rb_coopmat.axc, 2×2, hand-unrolled):**
- ONE workgroup (32 threads) computes a 32×32 output BLOCK (4 tiles = 4 coopmat accumulators).
- 4 loop-carried coopmat accumulators (acc_00..acc_11) in ONE K-loop (AT-1733 pre-gate PASSED).
- Per K-block: stage 32×16 A block + 16×32 B block to shared; load a_mat_0/a_mat_1 ONCE; load b_mat_0/b_mat_1 ONCE; reuse across 4 mul_adds → ~2× arithmetic intensity.
- NO codegen change (detect_loop_carried_coopmat + emit_for_range N-phi path already general; AT-1733 locks N=4 phis in ONE loop, spirv-val clean).
- Dispatch grid: (N/32, M/32, 1). Index: block_col=gid(0)/32, block_row=gid(1), lane=subgroup_invocation_id() — SAME idiom as M3.3b.
- HAND-UNROLLED: AXIOM has no compile-time strategy-unroll and no coopmat array type (SSA). The 2×2 variant hard-codes 4 named accumulators; strategy holes drive shared sizes + bench grid.

**Bit-exact (AT-1731/1732):**
- AT-1731: M=N=64, K=32, block grid (2,2,1) = 4 workgroups → max_diff==0.0 on NVIDIA (expected).
- AT-1732: K=32 (2 K-blocks) AND K=48 (3 K-blocks), tile_k=16 fixed → max_diff==0.0 (expected).
- Non-symmetric fixture detects A/B index swaps. Single-tile AT-1620/1622 RETAINED unchanged.
- Typed-skip on Lavapipe (CoopMatUnsupported).

**Measured TFLOPS (AT-1730) — HONEST, no asserted ratio:**
- Bench: resident_matmul_rb.rs, same methodology as AT-1710 (N_WARMUP=2, MIN-of-10, GpuTimestamp).
- Measured at **256³** (64 workgroups — occupancy-constrained; may be slower than M3.3b) AND **512³** (256 workgroups — better occupancy) AND **768³** (576 workgroups). All reported honestly.
- OCCUPANCY NOTE: At 256³ the RB grid (64 WGs) < ~188 SMs → may under-occupy vs M3.3b (256 WGs). If 256³ regresses but 512³ improves, both are reported. Larger matmuls better realize the register-blocking arithmetic-intensity gain.
- **MEASURED on NVIDIA RTX PRO 6000 Blackwell (QA run, 2026-06-04):** 256³ = **3.1 TFLOPS (2.5%) — REGRESSES** vs M3.3b 5.04 TFLOPS (under-occupied: 64 WGs < 188 SMs, as predicted); 512³ = **14.6 TFLOPS (11.7%)**; 768³ = **31.2 TFLOPS (24.96% of the datasheet estimate; 6.2× over the M3.3b 5.04-TFLOPS baseline)**. 24.96% is *just under* the project's 25% "competitive" threshold, so it is honestly reported as ~25% and NOT labeled competitive (the bench requires pct ≥ 25.0). The 256³ regression is reported honestly (not hidden or cherry-picked).
- 'competitive' label ONLY if measured pct >= 25.0. NO ratio asserted (only tflops>0 && finite).

**Deferred:**
- Multi-subgroup blocking: needs LocalInvocationId / SubgroupId builtin (not in AXIOM; deferred to M3.3d/M3.4).
- Double-buffered shared staging (software pipelining): deferred.
- Partial RB-blocks / edge tiles: masked/predicated coopmat load/store, carried from M3.3b.
- 2×4 and 4×4 RB variants: optional stretch; 2×2 ships; higher dims may spill registers.

**AT-1733 (pre-gate, CI):** compile + spirv-val: 4 coopmat phis in ONE loop header — PASSED. Locked the N-phi single-loop codegen guarantee before kernel work.
**AT-1734 (compile anchor, CI):** matmul_rb_coopmat.axc spirv-val clean with shipped assignments (shared_memory_bytes=2048 B).

**Depends on:** M3.3b. **Blocks:** M3.4 (multi-subgroup blocking, LocalInvocationId builtin).

### M3.3d — local_invocation_id() builtin + multi-subgroup matmul (2026-06-05)

**Two-part milestone.** PART A: language builtin `local_invocation_id(axis: u32) -> u32`. PART B: `matmul_msg_coopmat.axc` multi-subgroup (N_SG=2) register-blocked coopmat matmul.

**PART A — local_invocation_id() builtin:**
- Lowers to SPIR-V BuiltIn LocalInvocationId (vec3 u32 Input OpVariable + OpLoad + OpCompositeExtract by literal axis) — IDENTICAL lowering to gid() (GlobalInvocationId).
- **NO new OpCapability** (LocalInvocationId is core Shader — anti-pattern #7: no silent capability).
- Emitted **only when used** (opt-in; not on buffer presence like gid's shortcut).
- Threaded through all layers: lexer (collision guard), parser (no grammar change), HIR (LocalInvocationIdBuiltin{axis}), codegen (buffers.rs + body.rs + emit.rs).
- AT-1740 (lex/parse/HIR typecheck), AT-1741 (codegen + spirv-val + no-new-capability), AT-1742 (GPU: out[g*64+l]==l — runs on Lavapipe).
- **Durable builtin regardless of PART B perf outcome.**

**PART B — matmul_msg_coopmat.axc (N_SG=2):**
- @workgroup(64,1,1) = 2 subgroups of 32 lanes (NVIDIA wave32). HARD PRECONDITION: sg_size==32.
- sg_id = local_invocation_id(0u32) / subgroup_size(). ALL 64 threads cooperatively stage a_tile[512]/b_tile[1024] (3072 B shared); ONE workgroup_barrier; each subgroup runs its 2x2 RB block on distinct B columns (sg_id*32 offset).
- Cross-subgroup bit-exactness GUARANTEED BY CONSTRUCTION by workgroup-scope OpControlBarrier.
- All GPU tests/bench TYPED-SKIP on subgroup_size()!=32 (wave64 guard; mirror AT-1510).
- AT-1743 (M=32,N=64,K=32, 1 WG = 2 subgroups, max_diff==0.0), AT-1744 (M=64,N=128,K=48, 4 WGs, 3 K-blocks, max_diff==0.0). Non-symmetric fixture (A in {1..4}, B in {1..3}).
- AT-1745 (CI compile anchor: shared_memory_bytes==3072, LocalInvocationId var emitted).

**Measured TFLOPS (AT-1750) — HONEST, no asserted ratio:**
- Bench: resident_matmul_msg.rs (same methodology as AT-1730, N_WARMUP=2, MIN-of-10, GpuTimestamp).
- Grid: (N/64, M/32, 1). sg_size==32 guard before allocation.
- **MEASURED on NVIDIA RTX PRO 6000 Blackwell (QA run, 2026-06-05):** 256³ = **2.5 TFLOPS (2.0%)**; 768³ = **24.0 TFLOPS (19.2% of the 125-TFLOPS datasheet estimate)**.
- **REGRESSION vs M3.3c single-subgroup RB** (31.2 TFLOPS = 24.96% at 768³): halved workgroup count (288 vs 576 at 768³) + cross-subgroup barrier overhead outweigh staging amortization. 19.2% < 25.0% — NOT labeled competitive. Multi-subgroup is SLOWER than single-subgroup RB at every measured size.
- Single-subgroup RB bench (resident_matmul_rb.rs) RETAINED for A/B. **Single-subgroup RB (M3.3c) remains best.**
- 'competitive' label ONLY if pct >= 25.0. NO ratio asserted. Multi-subgroup is an honest negative performance result; the local_invocation_id() builtin is the durable deliverable.

**Deferred to M3.4+.** Double-buffered shared staging; partial/edge tiles; N_SG=4 / strategy-unroll.

**Depends on:** M3.3c. **Blocks:** M3.4 (double-buffered staging, N_SG=4).

### M3.2b — FlashAttention-2 streaming softmax (C2, deferred from M3.2)

**Why:** FA2's defining contribution — block-streaming ONLINE softmax (running max m_i, running denominator l_i, output rescale O_i ← O_i * exp(m_old-m_new) + P*V) — AVOIDS materializing the SxS score block. Deferred from M3.2 because its online-rescale arithmetic is a separate high-risk bit-exactness surface verified against C1 (`tiled_attention`) as the baseline. C2 earns the `flash_attention_v2` kernel name.

**Acceptance:** ≥ 80% of FlashAttention-3 cuBLAS+FA3 wrapper. @equiv_fp_tol(1e-3) vs C1 baseline. Streaming HIR path fully wired through coopmat_load-from-shared dispatch.

### M3.3 — KernelBench-Vulkan public submission

**Why:** Stanford's KernelBench (ICML 2025) is the standard eval for LLM-generated GPU kernels. Nobody has submitted SPIR-V yet. First submission is automatic differentiation in the LLM-for-GPU literature.

**Scope in:**
- Wrapper crate consuming KernelBench's PyTorch task definitions
- Auto-translate KernelBench's "PyTorch reference + CUDA-like kernel slot" into AXIOM-Compute `.axc` template + `@strategy` holes
- Run grid_search on a representative subset (Level 1 = ops, Level 2 = chains, Level 3 = real models)
- Publish results in a public repo / issue thread / arXiv preprint

**Acceptance:**
- Submit ≥ 50 / 250 KernelBench tasks with bit-exact output and measured Vulkan timing
- Beat PyTorch baseline on ≥ 30% (matches DeepSeek-R1 single-shot baseline)
- Beat PyTorch + 10-turn iterative LLM refinement on ≥ 5%

**Effort:** ~2000 LOC + paper writing.
**Depends on:** M3.0, M3.1, M3.2.

### M3.4 — llama.cpp Vulkan head-to-head A/B (the thesis-closing milestone)

**Why:** DESIGN.md §5 kill criterion: *"M2 slip: cannot match llama.cpp Vulkan Q4_K_M within 15% on any vendor."* Currently we have a Q4_K_M kernel that's bit-exact and dispatches, but no comparison.

**Scope in:**
- Side-by-side bench: same Q4_K_M weights, same input vector, same output, run via llama.cpp's Vulkan backend AND via AXIOM-Compute's kernel
- Identical machine, identical Vulkan ICD, fence-synchronized timing
- Multi-row matmul shape (M3.1 prerequisite) — single-row matvec doesn't reflect real inference
- Run across vendor matrix: NVIDIA RTX PRO 6000, AMD RDNA3 (target: Radeon 7900 XTX or MI300X if accessible), Intel Arc

**Acceptance:**
- Match llama.cpp Vulkan Q4_K_M within **5%** on NVIDIA (DESIGN.md §5 says 15%; aim higher)
- **Beat by ≥ 25%** on AMD APU or Intel Arc (DESIGN.md §5 — these are llama.cpp's weak spots per ggml-org/llama.cpp issues #16230, #21517, ollama #15601)
- Single annotated source produces all three variants via `@target` paths

**Effort:** ~1500 LOC + cross-vendor hardware access.
**Depends on:** M3.0, M3.1.
**Blocks:** practical adoption story.

**STATUS — NVIDIA half DONE (2026-06-06).** The same-machine, same-ICD A/B is implemented (`scripts/m34_llamacpp_ab.sh`, `crates/axc-driver/benches/dispatch_q4km_ab.rs`, results in `.pipeline/benchmarks/m34/ab_results.json`). llama.cpp pinned at tag `b9542` / SHA `6b80c74f285390368b3c99c5e750f19e9b096e98`, op = `test-backend-ops perf` Q4_K MUL_MAT n==1 GEMV. **Measured (NVIDIA RTX PRO 6000 Blackwell, work-normalized TFLOPS, kernel-only):** AXIOM ≈ 0.000085 TFLOPS (single-row matvec, 338.7 µs/op sustained, 1 output row) vs llama.cpp 7.39 TFLOPS (15.89 µs/op, 4096 output rows) → ratio ≈ 1e-5, llama ≈ 87,000× faster. **Kill-criterion (within 15%): FAIL on NVIDIA — the HONEST documented baseline.** This does NOT fire the project kill-criterion (DESIGN §5 = "within 15% on ANY vendor"; NVIDIA FAIL with the current FROZEN M2.6 single-row kernel ≠ project kill). The gap is two stacked deficits (1/4096th the rows per dispatch + dispatch-latency-dominated single workgroup); gap-closing path = fuse Q4_K_M dequant onto the M3.3c register-blocked coopmat matmul (follow-up). **AMD APU / Intel Arc halves remain BLOCKED on cross-vendor hardware (EB.1)** — deferred-not-dropped; those are where the portability thesis is strongest. See DESIGN.md §3.1.19, BENCHMARKS.md.

---

## Phase M4 — PyTorch + adoption

Goal: real users.

### M4.1 — PyTorch `torch.library` custom-op frontend

**Scope in:**
- `axiom-compute-py` Python package with PyO3 bindings
- `axc.compile_kernel(source) -> torch.library.Library` registers the kernel as a PyTorch op
- `axc.optimize_kernel(source, sample_inputs) -> CompiledKernel` runs grid_search and caches the winner
- Integrate with `torch.compile` as a custom backend (lower PyTorch graph ops to AXIOM-Compute kernels for the subset we support: matmul, layer norm, softmax, residuals)
- ABI: pass `torch.Tensor` as raw GPU pointer + descriptor (zero-copy when possible)

**Acceptance:**
- `pip install axiom-compute` works on Linux x86_64 with NVIDIA + Vulkan loader ✅ (maturin wheel)
- Registered `torch.ops.axiom.q4km_matmul` custom-op composes under `torch.compile(fullgraph=True)`; correct within FROZEN 1e-3 vs the Rust oracle (Phase 3, §3.1.27). HONEST: ~53% of f32 cuBLAS — the value is real torch interop, NOT a beat-cuBLAS speed claim (the original "within 50% of cuBLAS" target is reported, not gated)
- 10-line PyTorch user code can call the AXIOM kernel under `torch.compile` ✅ (`py/examples/q4km_torch_compile_demo.py`, Phase 3)

**Effort:** ~3000–5000 LOC.
**Depends on:** M3.0–M3.2.

### M4.2 — Upstream PR to llama.cpp / candle / MLX

**Scope in:**
- Pick the framework with the lowest integration friction (probably `candle` — Rust, Vulkan-curious)
- Port one hot kernel (Q4_K_M matmul) from candle's existing GLSL to AXIOM-Compute output
- Open a PR with measured A/B numbers
- Land it OR get a useful "won't merge because X" reason

**Acceptance:**
- Either: PR merged into a public framework
- Or: a clear, public technical reason from maintainers why the AXIOM-Compute approach is rejected (still valuable signal)

**Effort:** ~1500 LOC + maintainer relationship building.
**Depends on:** M3.4.

---

## Engineering debt (cuts across phases)

### EB.1 — Cross-vendor real GPU CI

**Scope:** self-hosted GitHub Actions runners for AMD (RDNA3 + Vulkan ICD) + Intel (Arc + ANV ICD). Currently CI is Lavapipe-only.

**Why it matters:** the portability thesis is unproven without measurements on AMD and Intel. Dev machine has only NVIDIA.

**Acceptance:** all 6 GPU tests + bench regression gate run on AMD and Intel in CI on every PR.

**Effort:** infra work, ~1 week to set up runners + ~1000 LOC of GitHub Actions YAML.

### EB.2 — Per-machine-keyed baseline format

**Bug:** `baselines.json` has only one machine field. Re-blessing on a different machine overwrites prior entries. We've seen this — Lavapipe runs were overwriting NVIDIA runs and vice versa.

**Fix:** `baselines.json` becomes `{ "machines": { "<machine_id>": { ...current schema... } } }` keyed by `(vulkan_device, cpu_model)` hash. Bench gate looks up entry for current machine; if absent, prints "no baseline; run AXC_BLESS_BASELINES=1".

**Effort:** ~300 LOC.

### EB.3 — `axc bench` CLI subcommand

**Currently:** users run `cargo bench -p axc-driver` directly. Should have an `axc bench [--filter NAME] [--bless]` wrapper that's easier to discover.

**Effort:** ~200 LOC.

### EB.4 — Baseline drift fix in current `baselines.json`

`baselines.json` was reblessed multiple times during the autonomous run; some entries reflect Lavapipe, some NVIDIA, some interleaved. Re-bless on a single machine after EB.2 lands.

---

## Feature gaps (mentioned in DESIGN.md / CLAUDE.md but never built)

### FG.1 — `axc rewrite` source-to-source LLM rewriter

`@strategy` holes are *parameter* tuning. `axc rewrite` is *structural* — the LLM rewrites the kernel body itself (different loop nest, different memory access pattern, etc.) and the compiler verifies via `@equiv_fp_tol`.

**Scope:** new MCP tool `propose_rewrite(source, hint) -> rewritten_source`; LLM-side prompting infrastructure; correctness verification harness that compares rewritten kernel output to original.

**Effort:** ~2000 LOC.

### FG.2 — `@transfer { ... }` blocks for inter-agent handoff

DESIGN.md §3.2 lists `@transfer` as an annotation. Idea: structured handoff between agents with confidence scores ("agent A optimized for tile size; agent B should explore @async_copy"). Currently parsed but no semantics.

**Effort:** ~800 LOC + protocol design.

### FG.3 — `@optimization_log {}` block

Per-kernel embedded history of prior optimization runs. Currently stored externally in `.pipeline/history/<hash>.jsonl` (M2.4). The block embeds it into source so the kernel is self-describing.

**Effort:** ~500 LOC.

### FG.4 — `@precondition` / `@postcondition` runtime checks

Currently parsed and HIR-validated, but never lowered to debug-mode runtime asserts. Spec promises `axc compile --debug` enables them.

**Effort:** ~600 LOC.

### FG.5 — Q5_K_M / Q6_K quantization variants

Straightforward extension of M2.6 Q4_K_M pattern. Q5_K_M adds a 1-bit overlay to Q4_K_M. Q6_K is 6-bit weights with similar superblock structure.

**Effort:** ~500 LOC each (mostly CPU reference + tests; codegen is small).

### FG.6 — `shared[T, N]` workgroup-local memory ✅ IMPLEMENTED (M3.2, 2026-06-02)

IMPLEMENTED. Full pipeline: lexer (token already existed) → parser (TypeRef::Shared, Stmt::SharedDecl) → HIR (SharedId, SharedTy, SharedDecl, KernelBodyTyped.shared, SharedRead/SharedWrite HIR nodes, OQ1 SET-based missing-barrier analysis, OQ2 conditional_depth divergent-barrier) → codegen (SharedBindings, emit_shared_globals with Float16/Int8/etc. caps, single-index OpAccessChain, Workgroup OpVariable, SPIR-V 1.3 interface list exclusion) → runtime (metadata v3, SharedMemoryExceedsDeviceLimit preflight, maxComputeSharedMemorySize cached). See DESIGN.md §3.1.14. Tests AT-1600..AT-1636 passing.

**Effort:** ~5800 LOC across 19+ files. Landed as part of M3.2.

### FG.7 — Sized arrays as locals

`array[T, N]` per DESIGN.md §3.1. Currently only buffer types are supported.

**Effort:** ~600 LOC.

### FG.8 — `axc verify` / `axc test --fuzz`

Parent AXIOM CLAUDE.md mentions these for `@strict` modules. AXIOM-Compute inherits the design but never built them. Verify checks annotation completeness; test --fuzz auto-generates inputs from `@precondition` constraints.

**Effort:** ~1500 LOC.

### FG.9 — Multi-kernel modules

Currently one `@kernel` per file. DESIGN.md hints at multi-kernel modules with cross-kernel `@strategy` (shared holes). Useful for prefix-sum style kernels with multiple stages.

**Effort:** ~1200 LOC.

---

## Polish + bug backlog

| Item | Severity | Effort |
|---|---|---|
| `BENCHMARKS.md` says "(M2.2)" in heading; should reflect current state | low | 5 min |
| `M2.6 dispatch_q4km_128` baseline label is `dispatch_q4km_128` while spec says `dispatch_gpu_q4km_128` — naming inconsistency | low | 30 min |
| `axc lex` output format human-only; spec mentions JSON for M1+ | low | 200 LOC |
| Stale doc-comment `Float64 cap = 6` in `body.rs:902` (caught by M1.5 reviewer) | trivial | 1 line |
| AT-103 empty-kernel "bit-exact" test only does determinism guard, not stored golden bytes | medium | 100 LOC |
| `dispatch_gpu_amortized` has only 1 entry (saxpy_1m); should add q4_0/q4_km amortized too | low | 100 LOC |
| `axc-runtime` exposed surface includes some `pub(crate)` items leaking into `pub` | low | review pass |
| No `CHANGELOG.md` | low | 30 min |

---

## Suggested ordering

Critical path to thesis closure:

```
M3.0 (bandwidth)  →  M3.1 (multi-row + coopmat)  →  M3.4 (llama.cpp A/B)
                                                      ↓
                                                   thesis proven
                                                      ↓
                                  M3.2 (FA2)  →  M3.3 (KernelBench)
                                                      ↓
                                                 publishable
                                                      ↓
                                              M4.1 (PyTorch)  →  M4.2 (upstream PR)
                                                                       ↓
                                                                  real users
```

Engineering debt + feature gaps interleave between milestones based on what unblocks next.

### Single highest-leverage next step

**M3.0 (dispatch bandwidth rework).**

Without it: every M3.x perf number is dominated by staging copy. With it: 1M-element saxpy goes from 23 ms → ~1 ms on NVIDIA, bringing AXIOM-Compute into the regime where comparison against handwritten kernels is meaningful. ~3000 LOC, ~1 milestone cycle through the 7-agent pipeline.

### Lowest-risk acceleration

**EB.2 (per-machine baselines) + EB.1 (cross-vendor CI)** can run in parallel with M3.0 since they touch separate codepaths (bench infrastructure + GitHub Actions vs runtime/dispatch). Frees future milestones from baseline-drift confusion and gives early signal on AMD/Intel portability before M3.4.

---

## Estimated effort to "thesis proven" state

To reach DESIGN.md §5 kill-criteria pass (M3.0 + M3.1 + M3.4):
- ~7000–10000 LOC
- ~3 milestone cycles (each milestone = architect → dual review → coder → QA → dual code review → merge, averaging ~6 hours of agent time per milestone in the autonomous regime)
- Cross-vendor hardware access (AMD RDNA3 + Intel Arc) for M3.4

To reach M4.2 (upstream adoption):
- additional ~5000–8000 LOC
- additional ~2 milestone cycles
- maintainer relationship building (real-world latency)

To finish the full DESIGN.md vision (everything above):
- ~25000–35000 LOC over ~10 milestone cycles
- with feature gaps + cross-vendor CI added
- a year of engineering at the current cadence

---

## How to contribute

- Pick a milestone or engineering-debt item from above
- Open an issue at https://github.com/rudybear/axiom-compute/issues describing your approach
- The 7-agent pipeline templates live at `.pipeline/templates/` — Architect first, dual review, then Coder
- Every PR runs in-process spirv-tools validation + clippy `--all-targets` `-D warnings` + the bench regression gate
