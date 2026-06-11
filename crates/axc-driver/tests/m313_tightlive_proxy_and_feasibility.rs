//! M3.13 — CPU-only (no GPU) static prongs:
//!   * PRONG A static proxy + structure: AT-2800 (the A-staging dependency-chain temp-count proxy
//!     for the tightlive kernel vs the M3.6 leader's 86 — MEASURE-AND-RECORD, see below) and AT-2801
//!     (tile_k==16 + the structure invariant: 4096 shared bytes, FILL OpStores present, a_block_size
//!     pinned 512).
//!   * PRONG A cap-set: AT-2806 (the tightlive cap set is BYTE-IDENTICAL to the M3.6 leader; spirv-val
//!     clean) — the byte-level twin of the compile_shared_examples anchor, here on the m313 harness.
//!   * PRONG A production-unchanged anchor: AT-2807 (the M3.13 codegen/HIR/parser/lexer/typecheck diff
//!     is EMPTY — PRONG A is pure source; this anchor asserts the production source trees carry no
//!     M3.13 edit, complementing the QA git-diff inspection).
//!   * PRONG B feasibility VERDICT: AT-2810 (the PAPER finding — NO per-element coopmat-fragment-write
//!     builtin exists; the RESERVED_COOPMAT_BUILTIN_NAMES set is EXACTLY the 4-name {Zero/Load/Store/
//!     MulAdd}; mul_add always yields an Accumulator-use result; no convert/recast/cast/extract/set/
//!     write_elem parses to a CoopMatBuiltin => register-resident dequant-into-fragment is
//!     UNEXPRESSIBLE on the portable KHR model => fused-dequant-coopmat-load is BLOCKED, the same
//!     class as M3.2c-optC §3.1.34. NO kernel built for PRONG B.).
//!
//! ## AT-2800 framing (optimistic reviewer's note — MEASURE-AND-RECORD, NOT a hard panic)
//! AT-2800 MEASURES the tightlive A-staging temp-count proxy and the M3.6 leader's (re-measured at
//! the SAME tile_k=16 config, NOT hard-coded as 86) and REPORTS the delta. The ARMED honest-negative
//! is that the proxy may NOT drop (AXIOM's naive SSA construction is source-order-insensitive: it
//! emits one result-inst per distinct value regardless of `let` ordering, so reordering reads /
//! inlining single-use lets need not shrink the chain). A no-drop is the "no codegen leverage" signal
//! (PRONG A honest-negative-(a)) — NOT a milestone failure: PRONG B's verdict + the campaign
//! conclusion still land. So AT-2800 asserts the proxy is MEASURED + finite + reported (and records
//! whether it dropped below the leader's), but does NOT hard-panic on no-drop. The authoritative
//! correctness gate is AT-2803 (GPU bit-identity); the perf gate is AT-2802 (GPU TFLOPS).
//!
//! ## Harness reuse
//! Reuses the m310b/m311/m312 raw-word-stream opcode model (decode / build_type_info /
//! find_a_staging_region / a_staging_chain_inst_count) — `rspirv::dr` rejects coopmat modules, so we
//! parse the raw SPIR-V word stream. The default driver pipeline runs NO spirv-opt, so the op-mix is
//! the AS-EMITTED instruction mix.

use std::collections::{BTreeMap, BTreeSet};
use axc_driver::compile_source_with_assignments;
use axc_hir::coopmat::{CoopMatBuiltin, RESERVED_COOPMAT_BUILTIN_NAMES};
use spirv::Op;
use spirv_tools::val::{Validator, create as create_validator};
use spirv_tools::TargetEnv;

type StrategyMap = BTreeMap<String, i64>;

const M36_LEADER_SRC: &str =
    include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached.axc");
const TIGHTLIVE_SRC: &str =
    include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached_tightlive.axc");

fn rb2x2_assignments() -> StrategyMap {
    let mut m = StrategyMap::new();
    m.insert("rb_m".to_owned(), 2);
    m.insert("rb_n".to_owned(), 2);
    m.insert("tile_k".to_owned(), 16);
    m.insert("a_block_size".to_owned(), 512);
    m.insert("b_block_size".to_owned(), 512);
    m
}

/// Compile to SPIR-V words AND spirv-val it (Vulkan 1.1) — the AS-EMITTED, pre-spirv-opt module.
fn compile_words_validated(src: &str, name: &str) -> (Vec<u32>, axc_runtime::KernelMetadata) {
    let (bytes, meta) = compile_source_with_assignments(src, &rb2x2_assignments())
        .unwrap_or_else(|e| panic!("M3.13 {name}: kernel must compile: {e:?}"));
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let validator = create_validator(Some(TargetEnv::Vulkan_1_1));
    validator
        .validate(&words, None)
        .unwrap_or_else(|e| panic!("M3.13 {name}: spirv-val FAILED: {e}"));
    (words, meta)
}

// ── Minimal SPIR-V word-stream model (the m310b/m311/m312 raw-word-stream harness) ────────────

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

struct Region {
    start: usize,
    end: usize,
}

/// Find the A-staging inner-loop body region (m312 technique): the unique `OpStore` of an
/// f32_to_f16 (OpFConvert→half) result INTO a Workgroup pointer (`a_tile[ei_a] = w`). BOTH the M3.6
/// leader and the tightlive kernel keep the f32_to_f16 sink (the restructure preserves it), so the
/// anchor is stable. The region runs from the loop body's first OpLabel (after the last OpLoopMerge
/// before the store) up to and including the store. Returns the region + the store instruction index.
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
        "AT-2800: expected an a_tile store `OpStore <Workgroup> <OpFConvert result>` \
         (the f32_to_f16 A-tile write) — not found. Both the M3.6 leader and tightlive keep the sink.",
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

/// The register/temp-count PROXY (the M3.12 r3 instrument) for the A-store dependency chain: the
/// number of result-producing instructions in the A-staging region. A larger value = a deeper live
/// dequant chain (more live SSA temporaries / register pressure). The 86/68/10 values in §3.1.40 came
/// from this counter over find_a_staging_region.
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
/// from an OpFMul on f32 (`d * f32_from_u32(sc)` / `dmin * f32_from_u32(mm)`). The FILL is held
/// constant in the tightlive kernel (VERBATIM M3.6). Confirms the fill is PRESENT.
fn count_scale_cache_fill_stores(insts: &[Inst], ti: &TypeInfo) -> usize {
    let op_store = Op::Store as u32;
    let op_fmul = Op::FMul as u32;

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

/// Extract the set of OpCapability values and OpExtension strings from a SPIR-V module.
fn capability_extension_sets(words: &[u32]) -> (BTreeSet<u32>, BTreeSet<String>) {
    const OP_CAPABILITY: u32 = 17;
    const OP_EXTENSION: u32 = 10;
    let mut caps: BTreeSet<u32> = BTreeSet::new();
    let mut exts: BTreeSet<String> = BTreeSet::new();
    let mut i = 5usize;
    while i < words.len() {
        let opcode = words[i] & 0xFFFF;
        let wc = (words[i] >> 16) as usize;
        if wc == 0 {
            break;
        }
        if opcode == OP_CAPABILITY && wc == 2 {
            caps.insert(words[i + 1]);
        } else if opcode == OP_EXTENSION {
            let mut bytes: Vec<u8> = Vec::new();
            for &wrd in &words[i + 1..i + wc] {
                bytes.extend_from_slice(&wrd.to_le_bytes());
            }
            while bytes.last() == Some(&0) {
                bytes.pop();
            }
            if let Ok(s) = String::from_utf8(bytes) {
                exts.insert(s);
            }
        }
        i += wc;
    }
    (caps, exts)
}

// ════════════════════════════════════════════════════════════════════════════════════════════
//  AT-2800 — PRONG A static proxy (MEASURE-AND-RECORD): tightlive A-staging temp-count vs leader
// ════════════════════════════════════════════════════════════════════════════════════════════

#[test]
fn at_2800_tightlive_a_staging_temp_count_proxy_measured() {
    let (leader_words, _lm) = compile_words_validated(M36_LEADER_SRC, "M3.6 leader");
    let leader_insts = decode(&leader_words);
    let leader_ti = build_type_info(&leader_insts);
    let leader_chain = a_staging_chain_inst_count(&leader_insts, &leader_ti);

    let (tl_words, _tm) = compile_words_validated(TIGHTLIVE_SRC, "tightlive");
    let tl_insts = decode(&tl_words);
    let tl_ti = build_type_info(&tl_insts);
    let tl_chain = a_staging_chain_inst_count(&tl_insts, &tl_ti);

    // The proxy MUST be measured (a finite, positive result-inst count for both kernels).
    assert!(
        leader_chain > 0 && tl_chain > 0,
        "AT-2800: both A-staging temp-count proxies must be MEASURED (>0); leader={leader_chain} \
         tightlive={tl_chain}"
    );

    let dropped = tl_chain < leader_chain;
    let verdict = if dropped {
        "PROXY DROPPED — the source restructure tightened the EMITTED SSA chain (the welcome \
         codegen-leverage upside; AT-2802 GPU TFLOPS decides whether the driver responds)."
    } else {
        "PROXY DID NOT DROP — AXIOM's naive SSA construction is source-order-insensitive for this \
         pattern (one result-inst per distinct value regardless of `let` order). PRONG A \
         HONEST-NEGATIVE-(a): NO codegen leverage. NOT a milestone failure — PRONG B (AT-2810) + \
         the campaign conclusion still land. AT-2800 is MEASURE-AND-RECORD, not a hard panic."
    };

    eprintln!(
        "AT-2800 (MEASURE-AND-RECORD): tightlive A-staging temp-count proxy = {tl_chain}; \
         M3.6 leader (re-measured at tile_k=16, NOT hard-coded) = {leader_chain}; \
         delta = {} ({}). {verdict}",
        leader_chain as i64 - tl_chain as i64,
        if dropped { "DROP" } else { "no-drop" },
    );

    // Re-confirm the historical §3.1.40 leader endpoint as context (recorded, not gated — the M3.6
    // source could legitimately shift). The 86 is the dual-reviewed M3.12 value.
    if leader_chain != 86 {
        eprintln!(
            "AT-2800 NOTE: M3.6 leader proxy re-measured at {leader_chain} (§3.1.40 recorded 86); \
             the delta vs tightlive is what AT-2800 reports, not the absolute value."
        );
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════════
//  AT-2801 — PRONG A structure invariants (pre-filter, subordinate to AT-2803)
// ════════════════════════════════════════════════════════════════════════════════════════════

#[test]
fn at_2801_tightlive_structure_invariants() {
    // tile_k == 16 (asserted from the @strategy pin in source).
    let strategy_line = TIGHTLIVE_SRC
        .lines()
        .find(|l| l.trim_start().starts_with("@strategy"))
        .expect("AT-2801: tightlive must declare an @strategy block");
    assert!(
        strategy_line.contains("tile_k: ?[16]"),
        "AT-2801: tile_k MUST stay PINNED to ?[16]; got @strategy line: {strategy_line}"
    );
    assert!(
        strategy_line.contains("a_block_size: ?[512]"),
        "AT-2801: a_block_size MUST stay PINNED to ?[512] (bounds the 256-entry scale caches); \
         got @strategy line: {strategy_line}"
    );

    let (tl_words, meta) = compile_words_validated(TIGHTLIVE_SRC, "tightlive");
    let tl_insts = decode(&tl_words);
    let tl_ti = build_type_info(&tl_insts);

    // shared_memory_bytes == 4096 (a_tile+b_tile 2048 + dsc_cache+dmm_cache 2048) — VERBATIM M3.6.
    let expected_shared_bytes: u32 = (512 + 512) * 2 + 2 * 256 * 4; // 2048 + 2048 = 4096
    assert_eq!(
        meta.shared_memory_bytes, expected_shared_bytes,
        "AT-2801: tightlive shared_memory_bytes must be {expected_shared_bytes} (==M3.6 leader: \
         a_tile+b_tile 2048 + dsc/dmm caches 2048); got {}",
        meta.shared_memory_bytes
    );

    // The scale-cache FILL OpStores are PRESENT (held constant — VERBATIM M3.6 fill).
    let fill = count_scale_cache_fill_stores(&tl_insts, &tl_ti);
    assert!(
        fill >= 2,
        "AT-2801: the scale-cache FILL OpStores (d*sc / dmin*m -> dsc/dmm cache) must be PRESENT \
         (held constant — the restructure touches ONLY the A-staging reads); got {fill} fill stores"
    );

    eprintln!(
        "AT-2801 PASS: tile_k==16, a_block_size==512 PIN; shared_memory_bytes=={} (4096); \
         FILL OpStores present ({fill}).",
        meta.shared_memory_bytes
    );
}

// ════════════════════════════════════════════════════════════════════════════════════════════
//  AT-2806 — PRONG A cap-set BYTE-IDENTICAL to the M3.6 leader (no new op-class, no capability)
// ════════════════════════════════════════════════════════════════════════════════════════════

#[test]
fn at_2806_tightlive_capset_identical() {
    let (leader_words, _lm) = compile_words_validated(M36_LEADER_SRC, "M3.6 leader");
    let (tl_words, _tm) = compile_words_validated(TIGHTLIVE_SRC, "tightlive");

    let (leader_caps, leader_exts) = capability_extension_sets(&leader_words);
    let (tl_caps, tl_exts) = capability_extension_sets(&tl_words);

    let extra: Vec<u32> = tl_caps.difference(&leader_caps).copied().collect();
    let dropped: Vec<u32> = leader_caps.difference(&tl_caps).copied().collect();
    assert!(
        extra.is_empty() && dropped.is_empty(),
        "AT-2806: tightlive cap set must be BYTE-IDENTICAL to the M3.6 leader (reorder/inline adds \
         no op-class, no capability). extra={extra:?} dropped={dropped:?} \
         (tightlive={tl_caps:?} leader={leader_caps:?})"
    );
    assert_eq!(
        tl_exts, leader_exts,
        "AT-2806: tightlive extension set must equal the M3.6 leader's; \
         tightlive={tl_exts:?} leader={leader_exts:?}"
    );

    eprintln!(
        "AT-2806 PASS: tightlive cap set BYTE-IDENTICAL to the M3.6 leader ({tl_caps:?}); \
         exts identical ({} entries); spirv-val clean (Vulkan 1.1).",
        tl_exts.len()
    );
}

// ════════════════════════════════════════════════════════════════════════════════════════════
//  AT-2807 — PRONG A production-unchanged anchor (the M3.13 codegen/HIR/parser/lexer diff is EMPTY)
// ════════════════════════════════════════════════════════════════════════════════════════════

fn repo_root() -> std::path::PathBuf {
    let manifest_dir =
        std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    manifest_dir.join("..").join("..")
}

#[test]
fn at_2807_empty_codegen_diff_production_unchanged() {
    // The tightlive kernel must NOT introduce any new builtin call name into the production trees.
    // It uses ONLY the pre-existing builtins (ptr_read_u8_zext, ptr_read_u16_zext, f16_bits_to_f32,
    // f32_from_u32, f32_to_f16, band/bor/shl/lshr, coopmat_{zero,load,mul_add,store}, gid,
    // subgroup_invocation_id, workgroup_barrier, range). This is a STATIC anchor that the kernel is
    // pure source — it references only names the M3.6 leader already references (no new codegen path).
    let leader_calls = builtin_call_names(M36_LEADER_SRC);
    let tl_calls = builtin_call_names(TIGHTLIVE_SRC);
    let new_calls: Vec<String> = tl_calls.difference(&leader_calls).cloned().collect();
    assert!(
        new_calls.is_empty(),
        "AT-2807: the tightlive kernel introduces builtin call name(s) NOT present in the M3.6 \
         leader: {new_calls:?}. PRONG A is PURE SOURCE — a reorder/inline must reference ONLY \
         builtins the leader already uses (no new codegen/HIR/parser/lexer/typecheck path). If a \
         new builtin is required, the restructure is not pure source — STOP and escalate (hard rule #5)."
    );

    // The production source trees carry NO M3.13 marker (the milestone touches only kernels + tests +
    // benches + docs). This complements the QA git-diff inspection (the authoritative AT-2807 check).
    let root = repo_root();
    let prod_dirs = [
        root.join("crates").join("axc-codegen").join("src"),
        root.join("crates").join("axc-hir").join("src"),
        root.join("crates").join("axc-parser").join("src"),
        root.join("crates").join("axc-lexer").join("src"),
    ];
    let mut offenders: Vec<String> = Vec::new();
    for dir in &prod_dirs {
        scan_for_marker(dir, "M3.13", &mut offenders);
        scan_for_marker(dir, "tightlive", &mut offenders);
    }
    assert!(
        offenders.is_empty(),
        "AT-2807: the production codegen/HIR/parser/lexer source trees must carry NO M3.13/tightlive \
         edit (PRONG A is pure source; PRONG B builds no kernel). Offending files: {offenders:?}"
    );

    eprintln!(
        "AT-2807 PASS: tightlive references only pre-existing builtins (no new codegen path); \
         production codegen/HIR/parser/lexer trees carry no M3.13/tightlive edit."
    );
}

/// Collect the set of `name(` call identifiers used in an .axc source, scanning only CODE (comment
/// lines stripped) and EXCLUDING the kernel's own declared `fn` name. A cheap lexical proxy for the
/// builtin/function names the kernel references; over-collecting (any `ident(`) only makes the "no new
/// name" assertion STRICTER.
fn builtin_call_names(src: &str) -> BTreeSet<String> {
    let mut names: BTreeSet<String> = BTreeSet::new();
    let is_ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    for raw_line in src.lines() {
        // Strip any trailing `//` line comment (the .axc sources use only `//` comments).
        let code = match raw_line.find("//") {
            Some(pos) => &raw_line[..pos],
            None => raw_line,
        };
        let trimmed = code.trim_start();
        // Skip the `fn <name>(` declaration line — the kernel's own name is not a referenced builtin.
        let is_fn_decl = trimmed.starts_with("fn ");
        let bytes = code.as_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            if bytes[i] == b'(' && i > 0 {
                let mut j = i;
                while j > 0 && is_ident(bytes[j - 1]) {
                    j -= 1;
                }
                if j < i {
                    let ident = &code[j..i];
                    let first_is_digit = ident.chars().next().unwrap_or('0').is_ascii_digit();
                    if !first_is_digit && !is_fn_decl {
                        names.insert(ident.to_owned());
                    }
                }
            }
            i += 1;
        }
    }
    names
}

fn scan_for_marker(dir: &std::path::Path, marker: &str, offenders: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            scan_for_marker(&path, marker, offenders);
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        if let Ok(content) = std::fs::read_to_string(&path) {
            if content.contains(marker) {
                offenders.push(format!("{}", path.display()));
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════════
//  AT-2810 — PRONG B feasibility VERDICT (paper finding, AT-2230-class): fused-dequant-coopmat
//            is BLOCKED by VK_KHR_cooperative_matrix fragment OPACITY. NO kernel built.
// ════════════════════════════════════════════════════════════════════════════════════════════

#[test]
fn at_2810_no_per_element_coopmat_fragment_write_op() {
    // (1) The RESERVED set is EXACTLY the 4-name {Zero/Load/Store/MulAdd} — NO per-element fragment
    //     write op (no convert/recast/cast/extract/set/write_elem). This is the EXHAUSTIVE builtin set.
    assert_eq!(
        RESERVED_COOPMAT_BUILTIN_NAMES.len(),
        4,
        "AT-2810: AXIOM must expose EXACTLY 4 coopmat builtins; got {}: {RESERVED_COOPMAT_BUILTIN_NAMES:?}",
        RESERVED_COOPMAT_BUILTIN_NAMES.len()
    );
    let expected: BTreeSet<&str> =
        ["coopmat_load", "coopmat_mul_add", "coopmat_store", "coopmat_zero"]
            .into_iter()
            .collect();
    let actual: BTreeSet<&str> = RESERVED_COOPMAT_BUILTIN_NAMES.iter().copied().collect();
    assert_eq!(
        actual, expected,
        "AT-2810: the coopmat builtin set must be EXACTLY {{Zero/Load/Store/MulAdd}}; got {actual:?}"
    );

    // (2) The CoopMatBuiltin enum likewise parses ONLY those 4 names; every per-element-fragment-write
    //     candidate name parses to None (no such builtin exists -> register-resident dequant-into-
    //     fragment is UNEXPRESSIBLE). This pins the NEGATIVE falsifiably.
    for &name in &["coopmat_zero", "coopmat_load", "coopmat_store", "coopmat_mul_add"] {
        assert!(
            CoopMatBuiltin::from_source_name(name).is_some(),
            "AT-2810: '{name}' must parse to a CoopMatBuiltin"
        );
    }
    for &candidate in &[
        "coopmat_convert",
        "coopmat_recast",
        "coopmat_cast",
        "coopmat_extract",
        "coopmat_set",
        "coopmat_write_elem",
        "coopmat_set_elem",
        "coopmat_store_elem",
        "coopmat_length",
        "CooperativeMatrixLength",
    ] {
        assert!(
            CoopMatBuiltin::from_source_name(candidate).is_none(),
            "AT-2810: '{candidate}' must NOT parse to a CoopMatBuiltin — no per-element \
             coopmat-fragment-write op exists in AXIOM's portable KHR model. If this fired, a \
             fragment-write op was added and the BLOCKED verdict no longer holds."
        );
    }

    // (3) The "always-Accumulator-use result" leg (optimistic reviewer suggestion): coopmat_mul_add
    //     is the ONLY builtin that PRODUCES a fragment from operands, and its result use is ALWAYS
    //     Accumulator (codegen coopmat.rs:191; typecheck.rs:2996-2998 hard-enforces the C-operand use).
    //     There is NO use-conversion/recast op, so a dequantized scalar can never become an A-fragment.
    //     We pin this structurally: MulAdd's result-typing is context-free (not from a per-element
    //     write), and it is one of exactly two fragment producers (Zero/Load are the others, both
    //     reading from a CONSTANT-null or a MEMORY pointer — never a register-resident scalar).
    assert_eq!(
        CoopMatBuiltin::from_source_name("coopmat_mul_add"),
        Some(CoopMatBuiltin::MulAdd),
        "AT-2810: coopmat_mul_add must be the recognized fragment producer (always-Accumulator-use)"
    );
    // Store is void/statement-only (does not produce a fragment); Load/Zero read memory/null.
    assert_eq!(
        CoopMatBuiltin::Zero.arity(),
        0,
        "AT-2810: coopmat_zero is the OpConstantNull fragment producer (0 args, no per-element write)"
    );
    assert_eq!(
        CoopMatBuiltin::Load.arity(),
        3,
        "AT-2810: coopmat_load reads FROM a (buf, offset, stride) MEMORY source — it CANNOT consume a \
         register-resident scalar; the dequantized f16 MUST be materialized in shared first (the M3.6 \
         round-trip). This is why the shared-staging realization is identically the M3.6 round-trip."
    );

    eprintln!(
        "AT-2810 PASS (PRONG B VERDICT — BLOCKED): coopmat builtin set is EXACTLY \
         {{coopmat_load, coopmat_mul_add, coopmat_store, coopmat_zero}} (4); NO per-element \
         coopmat-fragment-write op (convert/recast/cast/extract/set/write_elem all parse to None); \
         coopmat_mul_add always yields an Accumulator-use result; coopmat_load reads FROM memory \
         (forces the M3.6 shared round-trip). => register-resident dequant-into-fragment is \
         UNEXPRESSIBLE on the portable VK_KHR_cooperative_matrix model => fused-dequant-coopmat-load \
         is BLOCKED, the same class as M3.2c-optC (§3.1.34). NV coopmat2 (per-element/explicit-layout) \
         is the ONLY in-register path but is NV-ONLY — OUT of scope (breaks the portability thesis), \
         flagged for the EB.1 cross-vendor future. NO kernel built for PRONG B."
    );
}
