//! M3.12 — Dequant front-end ABLATION DIAGNOSTIC: CPU-only STATIC isolation falsifiers.
//!
//! The M3.12 variants are PROFILING INSTRUMENTS with INTENTIONALLY WRONG OUTPUT — they hold the
//! M3.6 leader's matmul + B-path + shared-A-stage + 3 barriers + coopmat + the scale-cache FILL
//! CONSTANT and vary ONLY the per-element A-dequant value, to PINPOINT the dominant 1.77x dequant
//! front-end tax (M3.11 ruled out the integer index-decode as latency-hidden). They are NOT correct
//! kernels and carry NO correctness gate; the perf deliverable (per-variant resident TFLOPS) lives in
//! the bench (resident_q4km_matmul_rb_f32acc_cached_ablate.rs).
//!
//! This file PINS the static premises so a mis-built variant FAILS LOUDLY before any perf read:
//!   * AT-2700 — each variant's A-staging op-mix matches its REALIZED isolation claim (the named op
//!     REMOVED, the kept ops PRESENT), the scale-cache FILL OpStores PRESENT (held constant), and the
//!     V-structure-probe constant a_tile store is INSIDE the k_block loop body (store-hoist guard).
//!   * AT-2705 — each variant file carries the wrong-output PROFILING-INSTRUMENT marker AND is
//!     referenced by NO correctness-gated test (no dispatch_*_equiv / q4km_oracle / 1e-3 comparison).
//!
//! ## Harness reuse
//! Reuses the m310b raw-word-stream opcode model + the def-use taint-trace from the a_tile store
//! (decode / build_type_info / find_a_staging_region / a_store_data_operand_trace) — the SAME
//! technique the m310b/m311 diagnostics use. `rspirv::dr` rejects coopmat modules, so we parse the
//! raw SPIR-V word stream. The default driver pipeline runs NO spirv-opt (AT-1613), so the op-mix is
//! the AS-EMITTED instruction mix.

use std::collections::{BTreeMap, BTreeSet};
use axc_driver::compile_source_with_assignments;
use spirv::Op;
use spirv_tools::val::{Validator, create as create_validator};
use spirv_tools::TargetEnv;

type StrategyMap = BTreeMap<String, i64>;

const STRUCTONLY_SRC: &str =
    include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached_ablate_structonly.axc");
const PASSTHROUGH_SRC: &str =
    include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached_ablate_passthrough.axc");
const M36_LEADER_SRC: &str =
    include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached.axc");

/// The mandatory wrong-output marker every ablation variant header must carry (AT-2705).
const WRONG_OUTPUT_MARKER: &str =
    "PROFILING INSTRUMENT — WRONG OUTPUT BY DESIGN — NOT A CORRECT/SHIPPABLE KERNEL";

fn rb2x2_assignments() -> StrategyMap {
    let mut m = StrategyMap::new();
    m.insert("rb_m".to_owned(), 2);
    m.insert("rb_n".to_owned(), 2);
    m.insert("tile_k".to_owned(), 16);
    m.insert("a_block_size".to_owned(), 512);
    m.insert("b_block_size".to_owned(), 512);
    m
}

/// Compile a variant to SPIR-V words AND spirv-val it (Vulkan 1.1) — the AS-EMITTED, pre-spirv-opt
/// module. spirv-val clean is part of AT-2700 (a mis-built variant must still be valid SPIR-V).
fn compile_words_validated(src: &str, name: &str) -> Vec<u32> {
    let (bytes, _meta) = compile_source_with_assignments(src, &rb2x2_assignments())
        .unwrap_or_else(|e| panic!("M3.12 {name}: kernel must compile: {e:?}"));
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let validator = create_validator(Some(TargetEnv::Vulkan_1_1));
    validator
        .validate(&words, None)
        .unwrap_or_else(|e| panic!("M3.12 {name}: spirv-val FAILED: {e}"));
    words
}

// ── Minimal SPIR-V word-stream model (the m310b/m311 raw-word-stream harness) ─────────────────

#[derive(Clone)]
struct Inst {
    op: u32,
    operands: Vec<u32>,
}

fn decode(words: &[u32]) -> Vec<Inst> {
    let mut out = Vec::new();
    let mut i = 5usize; // skip magic, version, generator, bound, schema
    while i < words.len() {
        let head = words[i];
        let wc = (head >> 16) as usize;
        let op = head & 0xFFFF;
        if wc == 0 || i + wc > words.len() {
            break;
        }
        out.push(Inst { op, operands: words[i + 1..i + wc].to_vec() });
        i += wc;
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PtrKind {
    GlobalU8,
    GlobalF16,
    GlobalOther,
    Shared,
    Other,
}

struct TypeInfo {
    ptr_kind: BTreeMap<u32, PtrKind>,
}

const SC_STORAGE_BUFFER: u32 = spirv::StorageClass::StorageBuffer as u32;
const SC_WORKGROUP: u32 = spirv::StorageClass::Workgroup as u32;

fn build_type_info(insts: &[Inst]) -> TypeInfo {
    let mut u8_types = BTreeSet::new();
    let mut f16_types = BTreeSet::new();
    let mut pointers: BTreeMap<u32, (u32, u32)> = BTreeMap::new();
    let mut ptr_kind: BTreeMap<u32, PtrKind> = BTreeMap::new();

    let op_type_int = Op::TypeInt as u32;
    let op_type_float = Op::TypeFloat as u32;
    let op_type_pointer = Op::TypePointer as u32;
    let op_variable = Op::Variable as u32;
    let op_access_chain = Op::AccessChain as u32;

    for inst in insts {
        if inst.op == op_type_int {
            if inst.operands.len() >= 2 && inst.operands[1] == 8 {
                u8_types.insert(inst.operands[0]);
            }
        } else if inst.op == op_type_float {
            if inst.operands.len() >= 2 && inst.operands[1] == 16 {
                f16_types.insert(inst.operands[0]);
            }
        } else if inst.op == op_type_pointer {
            if inst.operands.len() >= 3 {
                pointers.insert(inst.operands[0], (inst.operands[1], inst.operands[2]));
            }
        } else if inst.op == op_variable {
            if inst.operands.len() >= 2 {
                let ptr_ty = inst.operands[0];
                let var_id = inst.operands[1];
                if let Some(kind) = classify_pointer_type(ptr_ty, &pointers, &u8_types, &f16_types) {
                    ptr_kind.insert(var_id, kind);
                }
            }
        } else if inst.op == op_access_chain && inst.operands.len() >= 3 {
            let ptr_ty = inst.operands[0];
            let res_id = inst.operands[1];
            let base = inst.operands[2];
            let kind = classify_pointer_type(ptr_ty, &pointers, &u8_types, &f16_types)
                .or_else(|| ptr_kind.get(&base).copied())
                .unwrap_or(PtrKind::Other);
            ptr_kind.insert(res_id, kind);
        }
    }

    TypeInfo { ptr_kind }
}

fn classify_pointer_type(
    ptr_ty: u32,
    pointers: &BTreeMap<u32, (u32, u32)>,
    u8_types: &BTreeSet<u32>,
    f16_types: &BTreeSet<u32>,
) -> Option<PtrKind> {
    let (sc, pointee) = *pointers.get(&ptr_ty)?;
    if sc == SC_STORAGE_BUFFER {
        if u8_types.contains(&pointee) {
            Some(PtrKind::GlobalU8)
        } else if f16_types.contains(&pointee) {
            Some(PtrKind::GlobalF16)
        } else {
            Some(PtrKind::GlobalOther)
        }
    } else if sc == SC_WORKGROUP {
        Some(PtrKind::Shared)
    } else {
        Some(PtrKind::Other)
    }
}

fn load_ptr_kind(inst: &Inst, ti: &TypeInfo) -> Option<PtrKind> {
    if inst.op != Op::Load as u32 {
        return None;
    }
    let pointer = *inst.operands.get(2)?;
    Some(ti.ptr_kind.get(&pointer).copied().unwrap_or(PtrKind::Other))
}

struct Region {
    start: usize,
    end: usize,
}

/// Find the A-staging inner-loop body region (m310b technique): the unique `OpStore` of an
/// f32_to_f16 (OpFConvert→half) result INTO a Workgroup pointer (`a_tile[ei_a] = w`). ALL three
/// variants (M3.6 leader, V-structure-probe, V-passthrough) keep the f32_to_f16 sink, so the anchor
/// is stable. The region runs from the loop body's first OpLabel (after the last OpLoopMerge before
/// the store) up to and including the store. Returns the region + the store instruction index.
fn find_a_staging_region(insts: &[Inst], ti: &TypeInfo) -> (Region, usize) {
    let op_store = Op::Store as u32;
    let op_fconvert = Op::FConvert as u32;
    let op_label = Op::Label as u32;
    let op_loop_merge = Op::LoopMerge as u32;
    let op_load = Op::Load as u32;

    // Taint: f32_to_f16 result -> Function ptr (via Store) -> reload (via Load).
    let mut tainted_ids: BTreeSet<u32> = BTreeSet::new();
    let mut tainted_ptrs: BTreeSet<u32> = BTreeSet::new();
    for inst in insts {
        if inst.op == op_fconvert {
            if let Some(&rid) = inst.operands.get(1) {
                tainted_ids.insert(rid);
            }
        } else if inst.op == op_store {
            if let (Some(&ptr), Some(&obj)) = (inst.operands.first(), inst.operands.get(1)) {
                if tainted_ids.contains(&obj) {
                    tainted_ptrs.insert(ptr);
                }
            }
        } else if inst.op == op_load {
            if let (Some(&rid), Some(&ptr)) = (inst.operands.get(1), inst.operands.get(2)) {
                if tainted_ptrs.contains(&ptr) {
                    tainted_ids.insert(rid);
                }
            }
        }
    }

    let mut a_store_idx: Option<usize> = None;
    for (idx, inst) in insts.iter().enumerate() {
        if inst.op != op_store {
            continue;
        }
        let (Some(&ptr), Some(&obj)) = (inst.operands.first(), inst.operands.get(1)) else {
            continue;
        };
        let ptr_is_shared = ti.ptr_kind.get(&ptr).copied() == Some(PtrKind::Shared);
        if ptr_is_shared && tainted_ids.contains(&obj) {
            a_store_idx = Some(idx);
            break;
        }
    }
    let a_store_idx = a_store_idx.expect(
        "AT-2700: expected an a_tile store `OpStore <Workgroup> <OpFConvert result>` \
         (the f32_to_f16 A-tile write) — not found. Every variant keeps the f32_to_f16 sink.",
    );

    let mut body_start = 0usize;
    let mut last_loop_merge: Option<usize> = None;
    for (j, inst) in insts.iter().enumerate().take(a_store_idx) {
        if inst.op == op_loop_merge {
            last_loop_merge = Some(j);
        }
    }
    if let Some(lm) = last_loop_merge {
        for (j, inst) in insts.iter().enumerate().skip(lm + 1).take(a_store_idx - lm) {
            if inst.op == op_label {
                body_start = j;
                break;
            }
        }
    }

    (Region { start: body_start, end: a_store_idx + 1 }, a_store_idx)
}

/// The op-mix of the A-staging region, classified by the suspects the variants fork.
#[derive(Debug)]
struct AStagingOpMix {
    /// packed-q u8 global OpLoads (the `byte` nibble read) on the A-store chain region.
    u8_global_loads: usize,
    /// Workgroup (shared) OpLoads in the region (the per-element dsc/dmm scale-cache reads).
    shared_loads: usize,
    /// f32_to_f16 OpFConvert in the region (the f16 narrowing).
    fconvert: usize,
    /// ConvertUToF in the region (f32_from_u32).
    convert_u_to_f: usize,
    /// FMul in the region (the dsc * nib).
    fmul: usize,
    /// FSub in the region (the ... - dmm).
    fsub: usize,
}

fn count_a_staging_op_mix(insts: &[Inst], ti: &TypeInfo) -> AStagingOpMix {
    let (region, _store) = find_a_staging_region(insts, ti);
    let slice = &insts[region.start..region.end];
    let mut mix = AStagingOpMix {
        u8_global_loads: 0,
        shared_loads: 0,
        fconvert: 0,
        convert_u_to_f: 0,
        fmul: 0,
        fsub: 0,
    };
    for inst in slice {
        match load_ptr_kind(inst, ti) {
            Some(PtrKind::GlobalU8) => mix.u8_global_loads += 1,
            Some(PtrKind::Shared) => mix.shared_loads += 1,
            _ => {}
        }
        if inst.op == Op::FConvert as u32 {
            mix.fconvert += 1;
        } else if inst.op == Op::ConvertUToF as u32 {
            mix.convert_u_to_f += 1;
        } else if inst.op == Op::FMul as u32 {
            mix.fmul += 1;
        } else if inst.op == Op::FSub as u32 {
            mix.fsub += 1;
        }
    }
    mix
}

/// The register/temp-count PROXY (AT-2701, r3) for the A-store dependency chain: the number of
/// result-producing instructions in the A-staging region. A larger value = a deeper live dequant
/// chain (more live SSA temporaries / register pressure). Used to make the V-structure-probe-vs-real
/// register-footprint delta VISIBLE (the in-bucket register-occupancy residual).
fn a_staging_chain_inst_count(insts: &[Inst], ti: &TypeInfo) -> usize {
    let (region, _store) = find_a_staging_region(insts, ti);
    let slice = &insts[region.start..region.end];
    let non_result = |op: u32| -> bool {
        op == Op::Store as u32
            || op == Op::Branch as u32
            || op == Op::BranchConditional as u32
            || op == Op::LoopMerge as u32
            || op == Op::SelectionMerge as u32
            || op == Op::Label as u32
    };
    slice.iter().filter(|i| !non_result(i.op)).count()
}

/// Count the scale-cache FILL OpStores: OpStore into a Workgroup pointer whose stored value comes
/// from an OpFMul on f32 (`d * f32_from_u32(sc)` / `dmin * f32_from_u32(mm)`). The FILL is the ONLY
/// place an FMul-derived value is stored to shared (the per-element dequant FMul is DROPPED in every
/// variant; the FILL FMul is held constant). We count Workgroup stores whose object is produced by an
/// FMul to confirm the fill is PRESENT (held constant).
fn count_scale_cache_fill_stores(insts: &[Inst], ti: &TypeInfo) -> usize {
    let op_store = Op::Store as u32;
    let op_fmul = Op::FMul as u32;

    // value-ids produced by an FMul.
    let mut fmul_ids: BTreeSet<u32> = BTreeSet::new();
    for inst in insts {
        if inst.op == op_fmul {
            if let Some(&rid) = inst.operands.get(1) {
                fmul_ids.insert(rid);
            }
        }
    }
    let mut n = 0usize;
    for inst in insts {
        if inst.op != op_store {
            continue;
        }
        let (Some(&ptr), Some(&obj)) = (inst.operands.first(), inst.operands.get(1)) else {
            continue;
        };
        let ptr_is_shared = ti.ptr_kind.get(&ptr).copied() == Some(PtrKind::Shared);
        if ptr_is_shared && fmul_ids.contains(&obj) {
            n += 1;
        }
    }
    n
}

/// Whether the A-tile store is INSIDE a loop body region (store-hoist guard, r2 Fix 5): there is at
/// least one OpLoopMerge BEFORE the store AND at least one back-edge OpBranch AFTER the store whose
/// target is a label at-or-before the store (the loop continues past the store). Conservatively, we
/// assert the store sits strictly between the LAST OpLoopMerge before it and the loop's merge block —
/// i.e. an OpLoopMerge precedes the store and a branch back to a preceding label follows it.
fn a_store_is_inside_loop(insts: &[Inst], ti: &TypeInfo) -> bool {
    let (_region, store_idx) = find_a_staging_region(insts, ti);
    let op_loop_merge = Op::LoopMerge as u32;
    let op_label = Op::Label as u32;
    let op_branch = Op::Branch as u32;

    // The label index of the loop body (first OpLabel after the last OpLoopMerge before the store).
    let mut last_lm: Option<usize> = None;
    for (j, inst) in insts.iter().enumerate().take(store_idx) {
        if inst.op == op_loop_merge {
            last_lm = Some(j);
        }
    }
    let Some(lm) = last_lm else { return false };

    // The loop-merge target label id (OpLoopMerge operands: [merge-block, continue-block, ...]).
    let merge_label_id = *insts[lm].operands.first().unwrap_or(&0);

    // Collect label ids defined at-or-before the store (the loop's body/header labels). A back-edge
    // OpBranch after the store to one of these labels proves the store is in the loop body, not past
    // the merge. The merge label is EXCLUDED (a branch to it is the loop EXIT, not a back-edge).
    let mut labels_before_store: BTreeSet<u32> = BTreeSet::new();
    for inst in insts.iter().take(store_idx + 1) {
        if inst.op == op_label {
            if let Some(&lid) = inst.operands.first() {
                if lid != merge_label_id {
                    labels_before_store.insert(lid);
                }
            }
        }
    }

    for inst in insts.iter().skip(store_idx + 1) {
        if inst.op == op_branch {
            if let Some(&target) = inst.operands.first() {
                if labels_before_store.contains(&target) {
                    return true;
                }
            }
        }
    }
    false
}

// ════════════════════════════════════════════════════════════════════════════════════════════
//  AT-2700 — STATIC ISOLATION FALSIFIER: each variant isolates exactly its claimed suspect
// ════════════════════════════════════════════════════════════════════════════════════════════

#[test]
fn at_2700_each_variant_isolates_its_suspect_static_op_mix() {
    // ── M3.6 LEADER (the 42.86 endpoint) — the reference op-mix all variants are measured against. ──
    let leader_words = compile_words_validated(M36_LEADER_SRC, "M3.6 leader");
    let leader_insts = decode(&leader_words);
    let leader_ti = build_type_info(&leader_insts);
    let leader_mix = count_a_staging_op_mix(&leader_insts, &leader_ti);
    // The M3.6 leader's A-staging carries the FULL dequant chain: u8 load + 2 scale-reads + FMul +
    // FSub + ConvertUToF + FConvert.
    assert_eq!(
        leader_mix.u8_global_loads, 1,
        "AT-2700: M3.6 leader A-staging must have exactly 1 packed-q u8 load; got {leader_mix:?}"
    );
    assert!(
        leader_mix.shared_loads >= 2,
        "AT-2700: M3.6 leader A-staging must read the 2 per-element dsc/dmm scale caches; got {leader_mix:?}"
    );
    assert!(
        leader_mix.fmul >= 1 && leader_mix.fsub >= 1,
        "AT-2700: M3.6 leader A-staging must carry the dequant FMul + FSub; got {leader_mix:?}"
    );
    assert_eq!(
        leader_mix.fconvert, 1,
        "AT-2700: M3.6 leader A-staging must carry exactly 1 f32_to_f16 FConvert; got {leader_mix:?}"
    );

    // ── V-structure-probe — the PRIMARY DENOMINATOR ──────────────────────────────────────────────
    let probe_words = compile_words_validated(STRUCTONLY_SRC, "V-structure-probe");
    let probe_insts = decode(&probe_words);
    let probe_ti = build_type_info(&probe_insts);
    let probe_mix = count_a_staging_op_mix(&probe_insts, &probe_ti);

    // Claim: NO u8 OpLoad on the A-store chain (the packed-q load is DROPPED).
    assert_eq!(
        probe_mix.u8_global_loads, 0,
        "AT-2700 (V-structure-probe): the A-staging region must have ZERO packed-q u8 global loads \
         (the load is dropped — near-free index-derived value); got {probe_mix:?}"
    );
    // Claim: NO per-element dsc/dmm shared scale-reads.
    assert_eq!(
        probe_mix.shared_loads, 0,
        "AT-2700 (V-structure-probe): the A-staging region must have ZERO per-element shared \
         scale-cache reads (dropped); got {probe_mix:?}"
    );
    // Claim: NO FMul/FSub on the A-store chain (the dequant arithmetic is dropped).
    assert_eq!(
        (probe_mix.fmul, probe_mix.fsub), (0, 0),
        "AT-2700 (V-structure-probe): the A-staging region must have ZERO FMul/FSub (dropped); \
         got {probe_mix:?}"
    );
    // KEEPS the f32_to_f16 FConvert (the only f16 sink) + the a_tile store.
    assert_eq!(
        probe_mix.fconvert, 1,
        "AT-2700 (V-structure-probe): the A-staging region must keep exactly 1 f32_to_f16 FConvert \
         (the only f16 sink); got {probe_mix:?}"
    );
    // The scale-cache FILL OpStores PRESENT (held constant, r2 Fix 2).
    let probe_fill = count_scale_cache_fill_stores(&probe_insts, &probe_ti);
    assert!(
        probe_fill >= 2,
        "AT-2700 (V-structure-probe): the scale-cache FILL OpStores (d*sc / dmin*m -> dsc/dmm cache) \
         must be PRESENT (held constant); got {probe_fill} fill stores"
    );
    // The constant a_tile store is INSIDE the k_block loop body (store-hoist guard, r2 Fix 5).
    assert!(
        a_store_is_inside_loop(&probe_insts, &probe_ti),
        "AT-2700 (V-structure-probe): the index-derived a_tile store MUST stay INSIDE the k_block \
         loop body (store-hoist guard — a loop-invariant store could be hoisted, measuring a \
         structurally different kernel). The value is derived from the per-iter index ei_a so it \
         genuinely depends on the loop."
    );

    // ── V-passthrough — the DECISIVE structure-vs-arithmetic fork (f32_to_f16-RETAINING) ──────────
    let pass_words = compile_words_validated(PASSTHROUGH_SRC, "V-passthrough");
    let pass_insts = decode(&pass_words);
    let pass_ti = build_type_info(&pass_insts);
    let pass_mix = count_a_staging_op_mix(&pass_insts, &pass_ti);

    // KEEPS the u8 load (the integer index-decode + the packed-q load are held constant).
    assert_eq!(
        pass_mix.u8_global_loads, 1,
        "AT-2700 (V-passthrough): the A-staging region must KEEP exactly 1 packed-q u8 global load \
         (held constant); got {pass_mix:?}"
    );
    // DROPS the per-element dsc/dmm shared scale-reads.
    assert_eq!(
        pass_mix.shared_loads, 0,
        "AT-2700 (V-passthrough): the A-staging region must DROP the 2 per-element dsc/dmm shared \
         scale-cache reads; got {pass_mix:?}"
    );
    // DROPS the FMul/FSub.
    assert_eq!(
        (pass_mix.fmul, pass_mix.fsub), (0, 0),
        "AT-2700 (V-passthrough): the A-staging region must DROP the dequant FMul/FSub; got {pass_mix:?}"
    );
    // KEEPS the ConvertUToF (f32_from_u32) + the f32_to_f16 FConvert.
    assert_eq!(
        pass_mix.fconvert, 1,
        "AT-2700 (V-passthrough): the A-staging region must KEEP exactly 1 f32_to_f16 FConvert \
         (f32_to_f16 is the only f16 sink — u32_to_f16_bits is NOT expressible); got {pass_mix:?}"
    );
    assert!(
        pass_mix.convert_u_to_f >= 1,
        "AT-2700 (V-passthrough): the A-staging region must KEEP the f32_from_u32 ConvertUToF; \
         got {pass_mix:?}"
    );
    // The scale-cache FILL OpStores PRESENT (held constant, r2 Fix 2).
    let pass_fill = count_scale_cache_fill_stores(&pass_insts, &pass_ti);
    assert!(
        pass_fill >= 2,
        "AT-2700 (V-passthrough): the scale-cache FILL OpStores must be PRESENT (held constant — \
         only the per-element READS are dropped, NOT the fill); got {pass_fill} fill stores"
    );

    // ── The temp-count proxy (AT-2701, r3): the V-structure-probe chain must be SHORTER than the
    //    real-dequant endpoints (leader / V-passthrough) — making the register-footprint delta the
    //    re-anchored arithmetic line does NOT hold constant VISIBLE. (Recorded here statically; the
    //    bench records it alongside TFLOPS.) ──
    let leader_chain = a_staging_chain_inst_count(&leader_insts, &leader_ti);
    let probe_chain = a_staging_chain_inst_count(&probe_insts, &probe_ti);
    let pass_chain = a_staging_chain_inst_count(&pass_insts, &pass_ti);
    assert!(
        probe_chain < leader_chain,
        "AT-2700/2701: V-structure-probe's A-staging chain ({probe_chain} result-insts) must be \
         SHORTER than the M3.6 leader's ({leader_chain}) — it collapses the dequant chain to a \
         near-free index-derived value, removing the live dequant-chain temporaries. A non-shorter \
         chain means the probe did not actually drop the dequant front-end."
    );
    assert!(
        probe_chain < pass_chain,
        "AT-2700/2701: V-structure-probe's chain ({probe_chain}) must be shorter than V-passthrough's \
         ({pass_chain}) — the temp-count delta is the register-occupancy cross-check signal (AT-2702)."
    );

    eprintln!(
        "AT-2700 PASS: V-structure-probe op-mix {{u8={} shared={} fmul={} fsub={} fconvert={}}} \
         (NO load/scale-read/FMul/FSub, fill present={probe_fill}, store in-loop), \
         V-passthrough op-mix {{u8={} shared={} fmul={} fsub={} convUToF={} fconvert={}}} \
         (KEEP load+convert, DROP scale-read+FMul/FSub, fill present={pass_fill}). \
         temp-count proxy: leader={leader_chain} / V-passthrough={pass_chain} / V-structure-probe={probe_chain} \
         (probe chain shortest -> the register-footprint delta is visible for the AT-2702 cross-check).",
        probe_mix.u8_global_loads, probe_mix.shared_loads, probe_mix.fmul, probe_mix.fsub, probe_mix.fconvert,
        pass_mix.u8_global_loads, pass_mix.shared_loads, pass_mix.fmul, pass_mix.fsub,
        pass_mix.convert_u_to_f, pass_mix.fconvert,
    );
}

// ════════════════════════════════════════════════════════════════════════════════════════════
//  AT-2705 — HONESTY GUARD: variants are marked WRONG OUTPUT and have NO correctness gate
// ════════════════════════════════════════════════════════════════════════════════════════════

/// The on-disk ablation-variant source files (relative to the repo root).
const ABLATE_VARIANT_FILES: &[&str] = &[
    "examples/q4km_matmul_rb_coopmat_f32acc_cached_ablate_structonly.axc",
    "examples/q4km_matmul_rb_coopmat_f32acc_cached_ablate_passthrough.axc",
];

fn repo_root() -> std::path::PathBuf {
    let manifest_dir =
        std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    manifest_dir.join("..").join("..")
}

#[test]
fn at_2705_variants_are_marked_wrong_output_and_have_no_correctness_gate() {
    let root = repo_root();

    // (1) Every built variant file carries the mandatory wrong-output PROFILING-INSTRUMENT marker.
    for rel in ABLATE_VARIANT_FILES {
        let path = root.join(rel);
        let content = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("AT-2705: could not read variant {rel}: {e}"));
        assert!(
            content.contains(WRONG_OUTPUT_MARKER),
            "AT-2705: ablation variant {rel} MUST carry the wrong-output marker \
             '{WRONG_OUTPUT_MARKER}' in its header (these are PROFILING INSTRUMENTS with wrong \
             output by design)."
        );
    }

    // (2) NO correctness GATE references any `*_ablate_*.axc` file. We scan the driver test + bench
    //     trees for a reference to an ablation variant that is in CLOSE PROXIMITY (within a small line
    //     window) to a correctness-gate marker (equiv / oracle / 1e-3 / max_rel_diff / bit-identity).
    //     Proximity — NOT mere co-location in the same file — is what distinguishes an instrument
    //     wired INTO a gate from an instrument's cap-set/compile test that merely lives in a large
    //     shared test file alongside UNRELATED correctness tests (e.g. AT-2703 in
    //     compile_shared_examples.rs, which compiles + cap-set-checks the variants but does NOT gate
    //     their output). The m312 diagnostic file (this file) + the dedicated ablate bench are
    //     EXCLUDED outright (neither gates correctness on the instruments).
    const PROXIMITY_WINDOW: usize = 25; // lines either side of an `_ablate_` reference
    let needle = "_ablate_";
    let correctness_markers = [
        "max_rel_diff", "q4km_oracle", "1e-3", "FROZEN_REL_TOL",
        "bit-identity", "bit_identity", "_equiv", "combined <=", "assert_bit",
        "y_ref", "oracle",
    ];
    let scan_dirs = [
        root.join("crates").join("axc-driver").join("tests"),
        root.join("crates").join("axc-driver").join("benches"),
    ];

    let mut offenders: Vec<String> = Vec::new();
    for dir in &scan_dirs {
        let Ok(entries) = std::fs::read_dir(dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            let fname = path.file_name().and_then(|f| f.to_str()).unwrap_or("");
            if fname == "m312_ablation_diagnostic.rs"
                || fname == "resident_q4km_matmul_rb_f32acc_cached_ablate.rs"
            {
                continue;
            }
            let Ok(content) = std::fs::read_to_string(&path) else { continue };
            let lines: Vec<&str> = content.lines().collect();
            let ablate_lines: Vec<usize> = lines.iter().enumerate()
                .filter(|(_, l)| l.contains(needle))
                .map(|(i, _)| i)
                .collect();
            if ablate_lines.is_empty() {
                continue;
            }
            // For each ablate reference, check whether a correctness marker appears WITHIN the window.
            'refs: for &al in &ablate_lines {
                let lo = al.saturating_sub(PROXIMITY_WINDOW);
                let hi = (al + PROXIMITY_WINDOW + 1).min(lines.len());
                for line in &lines[lo..hi] {
                    if correctness_markers.iter().any(|m| line.contains(m)) {
                        offenders.push(format!("{fname}:{}", al + 1));
                        break 'refs;
                    }
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "AT-2705: ablation variants (`*_ablate_*.axc`) are PROFILING INSTRUMENTS with WRONG OUTPUT \
         and MUST NOT be wired into any correctness-gated test. An `_ablate_` reference sits within \
         {PROXIMITY_WINDOW} lines of a correctness-gate marker at: {offenders:?}. Remove the \
         correctness gate on the instruments (they are wrong by construction; asserting correctness \
         would be a lie)."
    );

    eprintln!(
        "AT-2705 PASS: {} ablation variant(s) carry the wrong-output marker; NO correctness-gated \
         test references any *_ablate_*.axc (the m312 static op-mix file + the ablate bench are \
         excluded — neither gates correctness on the instruments).",
        ABLATE_VARIANT_FILES.len()
    );
}
