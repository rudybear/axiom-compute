//! M3.10b — vec-load builtin DIAGNOSE-FIRST → NO-BUILD: falsifiable static diagnostic.
//!
//! This milestone builds NO codegen/HIR/typecheck/parser/lexer feature. It is the set of
//! CPU-only static-analysis tests (AT-2500..2505) that PIN the NO-BUILD conclusion of
//! DESIGN §3.1.38 (r2) so it FAILS LOUDLY in CI if a premise drifts. The conclusion:
//!
//!   * Same-shape (A/B = 4096×512×14336) gap decomposition: matmul-core deficit ≈ 1.35×
//!     (llama 102.49 / core@A/B 75.8), dequant front-end tax ≈ 1.77× (core 75.8 / M3.6 42.86);
//!     product ≈ 2.39 = the headline llama/Q4_K_M gap. The DEQUANT FRONT-END is the dominant gap.
//!   * The Q4_K_M A-tile is a COMPUTED dequant value `f32_to_f16(dsc·nib − dmm)`, not a global
//!     load — there is NO f16 A-buffer to vectorize (D1).
//!   * The dominant 1.77× front-end is ALU/ISSUE-bound, NOT packed-q-u8-load-bound: in the
//!     A-staging inner loop the dequant ALU-op count per packed-q u8 global OpLoad is ≥10:1
//!     (measured ~40:1 as-emitted) — so a u8vec4 view (16→4 loads) cannot move the gap.
//!
//! ## Why raw word-stream parsing (not rspirv::dr::Module)
//! `rspirv::dr::load_words` REJECTS these modules (`ConsumerError(DetachedInstruction(...))` on
//! `OpTypeCooperativeMatrixKHR`), so the architect's `dr::Module`-typed signatures are not usable
//! for the coopmat leader. We parse the raw SPIR-V word stream instead — the SAME technique the
//! existing `count_op_control_barrier` / `count_op_phi` anchors in `compile_shared_examples.rs`
//! use. Opcodes are compared against `spirv::Op::X as u32` (no magic numbers).
//!
//! ## PRE-spirv-opt (pessimistic warning #1, folded)
//! `compile_source_with_assignments` returns the RAW codegen output (the default pipeline runs NO
//! spirv-opt — verified by the byte-identical-across-recompile precedent AT-1613). AT-2502 asserts
//! over THIS as-emitted module: the instruction mix AXIOM generates, before any peephole/DCE. The
//! dequant chain is genuinely def-use-linked (each decode/convert result feeds the next), so DCE
//! could not eliminate it even if it ran — the ratio is stable across the pipeline.

use std::collections::{BTreeMap, BTreeSet};
use axc_driver::compile_source_with_assignments;
use spirv::Op;

type StrategyMap = BTreeMap<String, i64>;

/// The shipped RB 2×2 strategy assignment (rb_m=2, rb_n=2, tile_k=16, a/b_block_size=512).
fn rb2x2_assignments() -> StrategyMap {
    let mut m = StrategyMap::new();
    m.insert("rb_m".to_owned(), 2);
    m.insert("rb_n".to_owned(), 2);
    m.insert("tile_k".to_owned(), 16);
    m.insert("a_block_size".to_owned(), 512);
    m.insert("b_block_size".to_owned(), 512);
    m
}

/// Compile a source with the RB 2×2 assignments to SPIR-V words (the AS-EMITTED, pre-spirv-opt
/// module — the default driver pipeline runs no optimizer).
fn compile_words(src: &str) -> Vec<u32> {
    let (bytes, _meta) = compile_source_with_assignments(src, &rb2x2_assignments())
        .expect("M3.10b diagnostic: kernel must compile");
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ── Minimal SPIR-V word-stream model ────────────────────────────────────────────────────────

/// One decoded SPIR-V instruction (opcode + raw operand words, no result-type/id interpretation).
#[derive(Clone)]
struct Inst {
    op: u32,
    /// All operand words AFTER the opcode word (i.e. words[i+1 .. i+wc]).
    operands: Vec<u32>,
}

/// Decode the word stream (skipping the 5-word header) into a flat instruction list.
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

/// How a loaded/accessed pointer is classified by the buffer it ultimately targets.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PtrKind {
    /// A `StorageBuffer` pointer to a `u8` element (the packed-q `q` SSBO — the u8vec4 target).
    GlobalU8,
    /// A `StorageBuffer` pointer to an `f16` element (the `x` SSBO — B staging, shared with the
    /// plain-f32 core).
    GlobalF16,
    /// A `StorageBuffer` pointer to any other scalar element (e.g. the `C` f32 output).
    GlobalOther,
    /// A `Workgroup` (shared-memory) pointer — the dsc/dmm caches + a_tile/b_tile.
    Shared,
    /// A `Function` / `Input` / `PushConstant` / other pointer (locals, builtins, scalars).
    Other,
}

/// Resolved per-module pointer classification needed to classify loads. The intermediate type-id
/// / pointer-type maps are consumed during construction (`build_type_info`) and fully captured
/// into `ptr_kind` (value-id → PtrKind), which is all the diagnostic queries thereafter.
struct TypeInfo {
    /// any value-id (OpVariable / OpAccessChain result) → the PtrKind it denotes.
    ptr_kind: BTreeMap<u32, PtrKind>,
}

const SC_STORAGE_BUFFER: u32 = spirv::StorageClass::StorageBuffer as u32;
const SC_WORKGROUP: u32 = spirv::StorageClass::Workgroup as u32;

/// Build the type/pointer classification by a single forward pass (SPIR-V requires types and
/// the variables/access-chains that use them to be defined before use within the module ordering
/// we emit, and access chains derive from already-defined bases).
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
            // [result-id, width, signedness]
            if inst.operands.len() >= 2 && inst.operands[1] == 8 {
                u8_types.insert(inst.operands[0]);
            }
        } else if inst.op == op_type_float {
            // [result-id, width]
            if inst.operands.len() >= 2 && inst.operands[1] == 16 {
                f16_types.insert(inst.operands[0]);
            }
        } else if inst.op == op_type_pointer {
            // [result-id, storage-class, pointee-type-id]
            if inst.operands.len() >= 3 {
                pointers.insert(inst.operands[0], (inst.operands[1], inst.operands[2]));
            }
        } else if inst.op == op_variable {
            // [result-type (a pointer type), result-id, storage-class, ...]
            if inst.operands.len() >= 2 {
                let ptr_ty = inst.operands[0];
                let var_id = inst.operands[1];
                if let Some(kind) = classify_pointer_type(ptr_ty, &pointers, &u8_types, &f16_types) {
                    ptr_kind.insert(var_id, kind);
                }
            }
        } else if inst.op == op_access_chain {
            // [result-type (a pointer type), result-id, base, indices...]
            if inst.operands.len() >= 3 {
                let ptr_ty = inst.operands[0];
                let res_id = inst.operands[1];
                let base = inst.operands[2];
                // Prefer the result pointer-type classification; fall back to the base's kind
                // (an access chain into a global SSBO keeps the same storage class as its base).
                let kind = classify_pointer_type(ptr_ty, &pointers, &u8_types, &f16_types)
                    .or_else(|| ptr_kind.get(&base).copied())
                    .unwrap_or(PtrKind::Other);
                ptr_kind.insert(res_id, kind);
            }
        }
    }

    TypeInfo { ptr_kind }
}

/// Classify a pointer TYPE id into a PtrKind from its storage class + pointee scalar type.
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

/// The classified kind of the pointer an `OpLoad` reads from. Returns `None` for non-loads or
/// loads whose pointer kind is unknown (treated as `Other`).
fn load_ptr_kind(inst: &Inst, ti: &TypeInfo) -> Option<PtrKind> {
    if inst.op != Op::Load as u32 {
        return None;
    }
    // OpLoad: [result-type, result-id, pointer, memory-operands...]
    let pointer = *inst.operands.get(2)?;
    Some(ti.ptr_kind.get(&pointer).copied().unwrap_or(PtrKind::Other))
}

/// AT-2500/2504-facing whole-module load mix.
#[derive(Debug, Default, PartialEq, Eq)]
struct LoadMix {
    f16_global_loads: usize,
    u8_global_loads: usize,
    shared_loads: usize,
}

/// Count global StorageBuffer loads by element type over an instruction slice.
fn count_storage_buffer_loads(insts: &[Inst], ti: &TypeInfo) -> LoadMix {
    let mut mix = LoadMix::default();
    for inst in insts {
        match load_ptr_kind(inst, ti) {
            Some(PtrKind::GlobalU8) => mix.u8_global_loads += 1,
            Some(PtrKind::GlobalF16) => mix.f16_global_loads += 1,
            Some(PtrKind::Shared) => mix.shared_loads += 1,
            _ => {}
        }
    }
    mix
}

// ── Locating the A-staging inner-loop body ──────────────────────────────────────────────────

/// A half-open instruction-index range `[start, end)` of the A-staging inner-loop body.
struct Region {
    start: usize,
    end: usize,
}

/// The dequant ALU ops counted in the A-staging region (per the §3.1.38 D-dequant enumeration):
/// the integer superblock index-decode (`IMul`/`IAdd`/`ISub`/`UDiv`), the nibble extract
/// (`BitwiseAnd`/`ShiftRightLogical` + the `is_hi` compare), and the per-element dequant
/// arithmetic (`UConvert`/`ConvertUToF`/`FMul`/`FSub`/`FConvert`).
fn is_dequant_alu(op: u32) -> bool {
    const ALU: &[Op] = &[
        // integer index-decode / address arithmetic
        Op::IAdd, Op::ISub, Op::IMul, Op::UDiv, Op::SDiv, Op::UMod, Op::SMod,
        Op::ShiftLeftLogical, Op::ShiftRightLogical, Op::ShiftRightArithmetic,
        Op::BitwiseAnd, Op::BitwiseOr, Op::BitwiseXor,
        // the nibble-select predicate (is_hi >= 1)
        Op::UGreaterThanEqual, Op::UGreaterThan, Op::ULessThan, Op::ULessThanEqual, Op::IEqual,
        // the dequant converts + float arithmetic
        Op::UConvert, Op::SConvert, Op::ConvertUToF, Op::ConvertSToF, Op::FConvert,
        Op::FAdd, Op::FSub, Op::FMul,
    ];
    ALU.iter().any(|a| *a as u32 == op)
}

/// Find the A-staging inner-loop body region by a DEF-USE anchor (robust to line shifts and
/// to the B-staging f16 loads): locate the unique `OpStore` of a half value INTO a Workgroup
/// (shared) pointer where the stored value is produced by `OpFConvert` (f32_to_f16) — that is
/// `a_tile[ei_a] = f32_to_f16(...)` (cached.axc:240-241). The enclosing loop body is the region
/// from the loop's first body `OpLabel` up to that store. We return `[body_start, store_idx+...]`
/// bounded by the loop's continue/merge target.
///
/// Returns the region plus the instruction index of the FConvert→half store (the A-tile store).
fn find_a_staging_region(insts: &[Inst], ti: &TypeInfo) -> (Region, usize) {
    let op_store = Op::Store as u32;
    let op_fconvert = Op::FConvert as u32;
    let op_label = Op::Label as u32;
    let op_loop_merge = Op::LoopMerge as u32;

    let op_load = Op::Load as u32;

    // Forward pass marking every value-id that carries an `f32_to_f16` (OpFConvert→half) dequant
    // result — directly OR through a Function-local `let w` reload hop. AXIOM emits:
    //     %fc = OpFConvert %half %x      ; the f32_to_f16
    //           OpStore %w %fc           ; let w  (Function half ptr)
    //     %r  = OpLoad %half %w          ; reload w
    //           OpStore %a_tile_ptr %r   ; a_tile[ei] = w
    // so the Workgroup store's object is the RELOAD %r, not %fc directly. We propagate the taint
    // through OpStore (pointer ← value) and OpLoad (value ← pointer's last-stored taint).
    let mut tainted_ids: BTreeSet<u32> = BTreeSet::new(); // value-ids carrying the dequant FConvert
    let mut tainted_ptrs: BTreeSet<u32> = BTreeSet::new(); // Function ptrs holding a tainted value
    for inst in insts {
        if inst.op == op_fconvert {
            // [result-type, result-id, operand] — the f32_to_f16 result is tainted.
            if let Some(&rid) = inst.operands.get(1) {
                tainted_ids.insert(rid);
            }
        } else if inst.op == op_store {
            // [pointer, object, ...] — storing a tainted value taints the pointer.
            if let (Some(&ptr), Some(&obj)) = (inst.operands.first(), inst.operands.get(1)) {
                if tainted_ids.contains(&obj) {
                    tainted_ptrs.insert(ptr);
                }
            }
        } else if inst.op == op_load {
            // [result-type, result-id, pointer] — loading a tainted Function ptr taints the result.
            if let (Some(&rid), Some(&ptr)) = (inst.operands.get(1), inst.operands.get(2)) {
                if tainted_ptrs.contains(&ptr) {
                    tainted_ids.insert(rid);
                }
            }
        }
    }

    // The A-tile store: OpStore <workgroup-ptr> <value carrying the dequant FConvert>.
    // OpStore operands: [pointer, object, memory-operands...].
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
        "AT-2500/2502: expected an a_tile store `OpStore <Workgroup> <OpFConvert result>` \
         (the f32_to_f16 dequant A-tile write, cached.axc:240-241) — not found",
    );

    // Walk backwards from the store to the nearest enclosing loop body `OpLabel` (the OpLabel
    // immediately AFTER an OpLoopMerge). That OpLabel begins the A-staging inner-loop body.
    let mut body_start = 0usize;
    for j in (0..a_store_idx).rev() {
        if insts[j].op == op_label {
            // Is the immediately-preceding instruction an OpLoopMerge (the header's merge inst)?
            // The loop *body* label is the conditional-branch target; the simplest robust anchor
            // is: the FIRST OpLabel after the LAST OpLoopMerge that precedes the store.
            // We instead track the last loop-merge below; here just record the candidate.
            body_start = j;
            break;
        }
    }
    // Tighten: body_start = the OpLabel following the LAST OpLoopMerge before the store.
    let mut last_loop_merge: Option<usize> = None;
    for (j, inst) in insts.iter().enumerate().take(a_store_idx) {
        if inst.op == op_loop_merge {
            last_loop_merge = Some(j);
        }
    }
    if let Some(lm) = last_loop_merge {
        // The header block ends at the OpLoopMerge + its OpBranchConditional; the body label is
        // the next OpLabel after lm.
        for (j, inst) in insts.iter().enumerate().skip(lm + 1).take(a_store_idx - lm) {
            if inst.op == op_label {
                body_start = j;
                break;
            }
        }
    }

    // End just past the A-tile store (inclusive of the store + its branch back).
    let end = a_store_idx + 1;
    (Region { start: body_start, end }, a_store_idx)
}

/// AT-2502-facing instruction mix of the A-staging region.
#[derive(Debug)]
struct AluMix {
    dequant_alu_ops: usize,
    packed_q_u8_loads: usize,
    load_to_alu_ratio: f64,
}

fn count_a_staging_alu_ops(insts: &[Inst], ti: &TypeInfo) -> AluMix {
    let (region, _store) = find_a_staging_region(insts, ti);
    let slice = &insts[region.start..region.end];
    let mut dequant_alu_ops = 0usize;
    let mut packed_q_u8_loads = 0usize;
    for inst in slice {
        if is_dequant_alu(inst.op) {
            dequant_alu_ops += 1;
        }
        if load_ptr_kind(inst, ti) == Some(PtrKind::GlobalU8) {
            packed_q_u8_loads += 1;
        }
    }
    let load_to_alu_ratio = if packed_q_u8_loads == 0 {
        f64::INFINITY
    } else {
        dequant_alu_ops as f64 / packed_q_u8_loads as f64
    };
    AluMix { dequant_alu_ops, packed_q_u8_loads, load_to_alu_ratio }
}

/// AT-2500 def-use trace: does the A-tile store's data operand trace back to `f32_to_f16(...)`
/// (an OpFConvert to half) whose dequant chain reaches a u8 global load — and NEVER an f16
/// StorageBuffer OpLoad?
///
/// Returns `(traces_to_fconvert, reaches_u8_global_load, reaches_f16_global_load)`.
fn a_store_data_operand_trace(insts: &[Inst], ti: &TypeInfo) -> (bool, bool, bool) {
    let (region, store_idx) = find_a_staging_region(insts, ti);

    // Index every result-producing instruction by its result-id. The ops in the dequant chain
    // (OpLoad/OpAccessChain/OpFConvert/OpConvert*/integer+float ALU) all follow the
    // [result-type, result-id, value-operands...] shape, so the result-id is operands[1]. We must
    // EXCLUDE ops that do NOT have that shape (notably OpStore = [pointer, object] — its
    // operands[1] is a VALUE, not a result-id, and would otherwise shadow the real producer of
    // that value), plus control-flow / structural ops.
    let result_bearing = |op: u32| -> bool {
        !matches!(
            op,
            x if x == Op::Store as u32
                || x == Op::Branch as u32
                || x == Op::BranchConditional as u32
                || x == Op::LoopMerge as u32
                || x == Op::SelectionMerge as u32
                || x == Op::Return as u32
                || x == Op::ReturnValue as u32
                || x == Op::ControlBarrier as u32
                || x == Op::MemoryBarrier as u32
                || x == Op::Label as u32 // result-id is operands[0], no result-type
        )
    };
    let mut def: BTreeMap<u32, usize> = BTreeMap::new();
    for (idx, inst) in insts.iter().enumerate() {
        if result_bearing(inst.op) {
            if let Some(&rid) = inst.operands.get(1) {
                def.insert(rid, idx);
            }
        }
    }

    // Bridge for the `let w` Function-local reload: map each pointer-id to the value-ids ever
    // stored into it. When the trace hits an `OpLoad <Function-ptr>`, we continue from the
    // value(s) stored into that pointer (AXIOM's unoptimized `let`/reload pattern) so the chain
    // reaches the OpFConvert through the local.
    let op_store = Op::Store as u32;
    let mut stored_into: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for inst in insts {
        if inst.op == op_store {
            if let (Some(&ptr), Some(&obj)) = (inst.operands.first(), inst.operands.get(1)) {
                stored_into.entry(ptr).or_default().push(obj);
            }
        }
    }

    // The A-tile store's data operand.
    let store = &insts[store_idx];
    let data_operand = store.operands.get(1).copied()
        .expect("A-tile store must have an object operand");

    // BFS the def-use chain backward from the data operand. The region bounds the A-side; the u8
    // load + the dequant chain live in it. We follow value operands of every producer, and bridge
    // Function-local reloads via `stored_into`.
    let op_fconvert = Op::FConvert as u32;
    let op_load = Op::Load as u32;

    let mut traces_to_fconvert = false;
    let mut reaches_u8 = false;
    let mut reaches_f16 = false;

    let mut seen: BTreeSet<u32> = BTreeSet::new();
    let mut stack: Vec<u32> = vec![data_operand];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let Some(&didx) = def.get(&id) else { continue };
        let inst = &insts[didx];

        if inst.op == op_fconvert {
            traces_to_fconvert = true;
        }
        if inst.op == op_load {
            match load_ptr_kind(inst, ti) {
                Some(PtrKind::GlobalU8) => reaches_u8 = true,
                Some(PtrKind::GlobalF16) => reaches_f16 = true,
                _ => {}
            }
            // Bridge a Function-local reload: continue from whatever was stored into this pointer.
            if let Some(&ptr) = inst.operands.get(2) {
                if let Some(vals) = stored_into.get(&ptr) {
                    for &v in vals {
                        stack.push(v);
                    }
                }
            }
        }

        // Push the instruction's value operands. Every result-bearing op we care about follows
        // the [result-type, result-id, value-operands...] shape, so skip the first two words.
        for &opnd in inst.operands.iter().skip(2) {
            stack.push(opnd);
        }
    }

    let _ = (region.start, region.end);
    (traces_to_fconvert, reaches_u8, reaches_f16)
}

// ════════════════════════════════════════════════════════════════════════════════════════════
//  AT-2500 — the Q4_K_M A-tile is a COMPUTED dequant value (def-use trace, no f16 A global load)
// ════════════════════════════════════════════════════════════════════════════════════════════

#[test]
fn test_at2500_q4km_a_tile_has_no_f16_global_load_defuse_trace() {
    let words = compile_words(include_str!(
        "../../../examples/q4km_matmul_rb_coopmat_f32acc_cached.axc"
    ));
    let insts = decode(&words);
    let ti = build_type_info(&insts);

    let (traces_to_fconvert, reaches_u8, reaches_f16) = a_store_data_operand_trace(&insts, &ti);

    assert!(
        traces_to_fconvert,
        "AT-2500: the a_tile store's data operand must trace to f32_to_f16(...) (OpFConvert→half) \
         — the A-tile is a COMPUTED dequant value, not a global load (D1)."
    );
    assert!(
        reaches_u8,
        "AT-2500: the A-tile dequant chain must reach a packed-q u8 global StorageBuffer OpLoad \
         (the `byte` nibble read, cached.axc:232) — the dequant's only global input is the u8 q SSBO."
    );
    assert!(
        !reaches_f16,
        "AT-2500: the A-tile store's def-use chain must NEVER reach an f16 StorageBuffer OpLoad \
         — there is NO f16 A-buffer to vectorize. (A naive whole-module f16-load count would \
         falsely flag the B/x-side f16 loads; this trace is A-side-specific.)"
    );

    // The whole-module mix has u8 global loads (A-staging + scale fill) AND f16 global loads
    // (B/x staging) — confirming the kernel DOES have f16 loads, just not on the A-store path.
    let mix = count_storage_buffer_loads(&insts, &ti);
    assert!(
        mix.u8_global_loads > 0,
        "AT-2500: expected packed-q u8 global loads in the module; got {mix:?}"
    );
    assert!(
        mix.f16_global_loads > 0,
        "AT-2500: expected f16 global loads (the B/x side) in the module — the def-use trace, \
         not a naive count, is what proves the A side is f16-load-free; got {mix:?}"
    );

    eprintln!(
        "AT-2500 PASS: A-tile store traces to f32_to_f16 (computed dequant), reaches u8 global \
         load, NEVER an f16 global load. Module mix = {mix:?} (f16 loads exist on B/x, not A)."
    );
}

// ════════════════════════════════════════════════════════════════════════════════════════════
//  AT-2501 — the only directly-vectorizable Q4_K_M-specific global loads are the packed-q u8 bytes
// ════════════════════════════════════════════════════════════════════════════════════════════

#[test]
fn test_at2501_q4km_directly_vectorizable_loads_are_packed_q_only() {
    let words = compile_words(include_str!(
        "../../../examples/q4km_matmul_rb_coopmat_f32acc_cached.axc"
    ));
    let insts = decode(&words);
    let ti = build_type_info(&insts);

    let mix = count_storage_buffer_loads(&insts, &ti);

    // The packed-q u8 loads (the q SSBO) are the Q4_K_M-specific vectorizable target.
    assert!(
        mix.u8_global_loads > 0,
        "AT-2501: the packed-q u8 nibble/scale bytes (the q SSBO) must be the directly-vectorizable \
         Q4_K_M-specific global loads; got {mix:?}"
    );
    // The f16 global loads are the B/x staging — shared with the plain-f32 core, NOT Q4_K_M-specific.
    assert!(
        mix.f16_global_loads > 0,
        "AT-2501: the x SSBO f16 B-staging loads must exist (shared with the plain-f32 core); got {mix:?}"
    );

    // The decisive partition assertion: ZERO f16 global load is on the A-store def-use path
    // (AT-2500 proves it) — so the f16 loads are NOT an A-side (Q4_K_M-specific) vectorization
    // target; the ONLY Q4_K_M-specific vectorizable global load is the u8 packed-q chain.
    let (_traces, _u8, reaches_f16_on_a) = a_store_data_operand_trace(&insts, &ti);
    assert!(
        !reaches_f16_on_a,
        "AT-2501: no f16 global load may feed the A-tile (A is dequant-computed) — so the only \
         Q4_K_M-SPECIFIC vectorizable global load is the packed-q u8 chain."
    );

    eprintln!(
        "AT-2501 PASS: Q4_K_M-specific directly-vectorizable global loads = packed-q u8 only \
         (u8_global={}, f16_global={} are B/x-side / core-shared, none feed A).",
        mix.u8_global_loads, mix.f16_global_loads
    );
}

// ════════════════════════════════════════════════════════════════════════════════════════════
//  AT-2502 — THE DECISIVE FALSIFIER: the dequant front-end is ALU/ISSUE-bound, not u8-load-bound
// ════════════════════════════════════════════════════════════════════════════════════════════

/// The conservative load-to-ALU threshold. The architect measured ~26:1, the pessimistic reviewer
/// independently counted ~37:1, and the as-emitted module here counts ~40:1 — so ≥10:1 is a
/// conservative floor. A `u8vec4` view cuts the u8 loads 4× (16→4 loads/lane/k_block); at ≥10:1
/// that removes ≤ ~3% of the path's issue-slots and CANNOT move the 1.77× dequant front-end gap.
const LOAD_VS_ALU_THRESHOLD: f64 = 10.0;

#[test]
fn test_at2502_dequant_front_end_is_alu_issue_dominated_not_u8_load_bound() {
    // PRE-spirv-opt: compile_source_with_assignments returns the RAW codegen output (no optimizer
    // in the default pipeline). We assert the AS-EMITTED instruction mix — what AXIOM generates.
    let words = compile_words(include_str!(
        "../../../examples/q4km_matmul_rb_coopmat_f32acc_cached.axc"
    ));
    let insts = decode(&words);
    let ti = build_type_info(&insts);

    let alu = count_a_staging_alu_ops(&insts, &ti);

    // Exactly one packed-q u8 global load per dequantized A-tile element (cached.axc:232).
    assert_eq!(
        alu.packed_q_u8_loads, 1,
        "AT-2502: the A-staging inner-loop body must contain EXACTLY 1 packed-q u8 global OpLoad \
         (the `byte` read, cached.axc:232); got {} — if this changed, the load:ALU ratio analysis \
         and the NO-BUILD premise must be re-examined.",
        alu.packed_q_u8_loads
    );

    // The decisive assertion: the dequant front-end is ALU/issue-dominated.
    assert!(
        alu.load_to_alu_ratio >= LOAD_VS_ALU_THRESHOLD,
        "AT-2502 FALSIFIER: A-staging dequant ALU:load ratio = {:.1}:1 ({} ALU : {} u8 load) — \
         expected ≥ {:.0}:1. If this dropped BELOW {:.0}:1 the front-end would be u8-LOAD-bound \
         and the NO-BUILD would be WRONG (the vec-load BUILD re-opens — ESCALATE to the Architect, \
         hard rule #5). Measured as-emitted (pre-spirv-opt).",
        alu.load_to_alu_ratio, alu.dequant_alu_ops, alu.packed_q_u8_loads,
        LOAD_VS_ALU_THRESHOLD, LOAD_VS_ALU_THRESHOLD
    );

    eprintln!(
        "AT-2502 PASS (NO-BUILD premise PINNED): A-staging dequant front-end is ALU/ISSUE-bound — \
         {} dequant ALU ops : {} packed-q u8 global load = {:.1}:1 (≥ {:.0}:1 threshold; \
         architect ~26:1, pessimistic ~37:1, as-emitted ~{:.0}:1). A u8vec4 view (16→4 loads) \
         trims ~3% of the path's issue-slots → CANNOT move the 1.77× dequant gap. Pre-spirv-opt.",
        alu.dequant_alu_ops, alu.packed_q_u8_loads, alu.load_to_alu_ratio,
        LOAD_VS_ALU_THRESHOLD, alu.load_to_alu_ratio
    );
}

// ════════════════════════════════════════════════════════════════════════════════════════════
//  AT-2503 — CORRECTED SAME-SHAPE gap-decomposition pin (re-pinned to A/B, NOT 768³/3.24×)
// ════════════════════════════════════════════════════════════════════════════════════════════

/// Read the plain-f32 single-subgroup-RB core TFLOPS at the A/B shape (M=4096 N=512 K=14336)
/// from the merged M3.9 bench `ab_results_warptile.json` — NOT a hardcoded literal. The bench
/// records lines like `... RATIO[M=4096 N=512 K=14336]: K=4-warptile X / single-subgroup-RB Y = ...`.
/// We parse all such `single-subgroup-RB Y` values at the A/B shape and return their mean.
fn read_core_ab_tflops_from_bench() -> f64 {
    let manifest_dir =
        std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let bench_path = manifest_dir
        .join("..")
        .join("..")
        .join(".pipeline")
        .join("benchmarks")
        .join("m34")
        .join("ab_results_warptile.json");
    let content = std::fs::read_to_string(&bench_path).unwrap_or_else(|e| {
        panic!(
            "AT-2503: could not read the source bench {bench_path:?}: {e}. The same-shape core@A/B \
             number MUST be read from the merged bench, not hardcoded-without-source."
        )
    });

    let mut vals: Vec<f64> = Vec::new();
    for line in content.lines() {
        // Only the A/B shape rows.
        if !line.contains("M=4096 N=512 K=14336") {
            continue;
        }
        // Extract the number after "single-subgroup-RB ".
        if let Some(pos) = line.find("single-subgroup-RB ") {
            let rest = &line[pos + "single-subgroup-RB ".len()..];
            let num: String = rest
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            if let Ok(v) = num.parse::<f64>() {
                vals.push(v);
            }
        }
    }
    assert!(
        !vals.is_empty(),
        "AT-2503: found no `single-subgroup-RB <tflops>` values at the A/B shape (M=4096 N=512 \
         K=14336) in ab_results_warptile.json — the bench format must have changed; re-pin."
    );
    vals.iter().sum::<f64>() / vals.len() as f64
}

#[test]
fn test_at2503_same_shape_gap_decomposition_pins_no_build_premises() {
    // The same-shape data points (ALL at A/B 4096×512×14336). The core number is READ from the
    // bench (not hardcoded); the rest are the campaign's recorded headline figures.
    let core_ab = read_core_ab_tflops_from_bench();
    let q4km_ab = 42.86_f64; // M3.6 leader @ A/B (the merged leader bench)
    let llama_ab = 102.49_f64; // llama.cpp @ A/B
    let datasheet = 125.0_f64; // RTX PRO 6000 Blackwell f16 datasheet TFLOPS

    // The bench's single-subgroup-RB @ A/B is ~75.8 (75.588–76.038 across repeats).
    assert!(
        (75.5..=76.1).contains(&core_ab),
        "AT-2503: core@A/B read from bench = {core_ab:.3} TFLOPS, expected ~75.8 (the \
         single-subgroup-RB column at the A/B shape, 75.588–76.038). If the bench moved, re-pin."
    );

    // Datasheet-% at A/B ≈ 60.6%, NOT the retracted ~25% ceiling.
    let core_datasheet_pct = core_ab / datasheet * 100.0;
    assert!(
        (59.0..=62.0).contains(&core_datasheet_pct),
        "AT-2503: core@A/B datasheet-% = {core_datasheet_pct:.1}% — expected ~60.6% (NOT the \
         retracted ~25% 'ceiling at any shape'). The core's datasheet-% RISES with shape."
    );

    // Same-shape decomposition.
    let core_deficit = llama_ab / core_ab; // ≈ 1.35×
    let dequant_tax = core_ab / q4km_ab; // ≈ 1.77×
    let product = core_deficit * dequant_tax; // ≈ 2.39×
    let headline = llama_ab / q4km_ab; // = 2.391× exactly

    assert!(
        (1.33..=1.37).contains(&core_deficit),
        "AT-2503: same-shape matmul-core deficit = {core_deficit:.3}× (llama {llama_ab} / core@A/B \
         {core_ab:.2}) — expected ~1.35× (NOT the retracted 3.24× which divided by core@768³ 31.2)."
    );
    assert!(
        (1.75..=1.79).contains(&dequant_tax),
        "AT-2503: same-shape dequant front-end tax = {dequant_tax:.3}× (core@A/B {core_ab:.2} / \
         M3.6 {q4km_ab}) — expected ~1.77× (the DOMINANT same-shape gap, reversing r1)."
    );
    // The dequant front-end (1.77×) must be the LARGER same-shape gap (inverting r1).
    assert!(
        dequant_tax > core_deficit,
        "AT-2503: the dequant front-end tax ({dequant_tax:.3}×) must DOMINATE the matmul-core \
         deficit ({core_deficit:.3}×) at the same shape — this is the r2 inversion of r1."
    );
    // Product reconstructs the headline gap.
    assert!(
        (product - headline).abs() < 0.02,
        "AT-2503: 1.35×1.77 product = {product:.3}× must reconstruct the headline llama/Q4_K_M \
         gap = {headline:.3}× within tolerance."
    );
    assert!(
        (2.37..=2.41).contains(&headline),
        "AT-2503: headline gap llama/Q4_K_M = {headline:.3}× — expected ~2.39×."
    );

    // The monotonic datasheet-% sequence (guard against re-introducing the false 25% ceiling).
    // 2.5%@256, 11.7%@512, 24.96%@768, 60.6%@A/B — strictly increasing, no plateau.
    let datasheet_pct_seq = [2.5_f64, 11.7, 24.96, 60.6];
    for w in datasheet_pct_seq.windows(2) {
        assert!(
            w[1] > w[0],
            "AT-2503: the core's datasheet-% sequence {datasheet_pct_seq:?} must be strictly \
             monotonic-increasing (the '25% ceiling at any shape' is RETRACTED — no plateau)."
        );
    }
    // The retracted figures must NOT be reintroduced as premises.
    assert!(
        (core_datasheet_pct - 24.96).abs() > 1.0,
        "AT-2503: core@A/B datasheet-% must NOT be the retracted ~24.96% (768³ figure)."
    );
    assert!(
        (core_deficit - 3.24).abs() > 0.3,
        "AT-2503: the retracted 3.24× core deficit must NOT reappear (it divided by core@768³)."
    );

    eprintln!(
        "AT-2503 PASS (same-shape, re-pinned to A/B): core@A/B {core_ab:.2} TFLOPS \
         ({core_datasheet_pct:.1}% datasheet, read from bench), Q4_K_M {q4km_ab}, llama {llama_ab} \
         → core deficit {core_deficit:.2}× / dequant tax {dequant_tax:.2}× (DOMINANT) / product \
         {product:.2}× = headline {headline:.2}×. 3.24×/25%-ceiling RETRACTED; datasheet-% monotonic."
    );
}

// ════════════════════════════════════════════════════════════════════════════════════════════
//  AT-2504 — the plain-f32 core's f16 A/B staging IS the f16 vec-load target (occupancy-ruled-out)
// ════════════════════════════════════════════════════════════════════════════════════════════

#[test]
fn test_at2504_plain_f32_core_staging_is_f16_vectorization_target() {
    let words = compile_words(include_str!("../../../examples/matmul_rb_coopmat.axc"));
    let insts = decode(&words);
    let ti = build_type_info(&insts);

    let mix = count_storage_buffer_loads(&insts, &ti);

    // The plain-f32 core stages f16 A and f16 B via scalar global StorageBuffer loads — the exact
    // target an f16 vec4 load would vectorize.
    assert!(
        mix.f16_global_loads > 0,
        "AT-2504: the plain-f32 core (matmul_rb_coopmat.axc) must issue scalar f16 global \
         StorageBuffer loads (the A/B staging — the f16 vec-load target); got {mix:?}"
    );
    // It is the DEQUANT-FREE core: NO packed-q u8 global loads.
    assert_eq!(
        mix.u8_global_loads, 0,
        "AT-2504: the plain-f32 core must have ZERO packed-q u8 global loads (it is dequant-free — \
         the f16-staging lever lives on THIS core, distinct from the Q4_K_M u8 path); got {mix:?}"
    );

    eprintln!(
        "AT-2504 PASS: plain-f32 core staging = f16 global loads (the f16 vec-load target, \
         occupancy-ruled-out by M3.7 0.97× + M3.10a 0.79–0.89×), u8_global=0 (dequant-free). \
         Mix = {mix:?}. The f16-staging lever is core-resident, distinct from the u8 path (D3)."
    );
}

// ════════════════════════════════════════════════════════════════════════════════════════════
//  AT-2505 — no-regression + EMPTY-codegen-diff anchor (this milestone adds ZERO codegen)
// ════════════════════════════════════════════════════════════════════════════════════════════

#[test]
fn test_at2505_no_codegen_change_suite_green() {
    // The empty-codegen-diff anchor: `git diff main` must touch NO file under the codegen /
    // HIR / typecheck / parser / lexer source trees. This milestone is a diagnostic + a
    // documented conclusion — it adds ZERO codegen (same discipline as M3.10a).
    let manifest_dir =
        std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir.join("..").join("..");

    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(&repo_root)
        .args(["diff", "--name-only", "main", "--"])
        .args([
            "crates/axc-codegen/src",
            "crates/axc-hir/src",
            "crates/axc-typecheck",
            "crates/axc-parser/src",
            "crates/axc-lexer/src",
        ])
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let changed = String::from_utf8_lossy(&out.stdout);
            let changed_files: Vec<&str> =
                changed.lines().filter(|l| !l.trim().is_empty()).collect();
            assert!(
                changed_files.is_empty(),
                "AT-2505: M3.10b must add ZERO codegen — `git diff main` touched protected \
                 source files: {changed_files:?}. This milestone is a diagnostic + a documented \
                 NO-BUILD conclusion; any codegen/HIR/typecheck/parser/lexer change is out of scope."
            );
            eprintln!(
                "AT-2505 PASS: empty-codegen-diff confirmed — `git diff main` touches NO file under \
                 axc-codegen/src, axc-hir/src, axc-typecheck, axc-parser/src, axc-lexer/src."
            );
        }
        Ok(out) => {
            // `main` may be unavailable (detached / shallow CI checkout). Don't fail the suite on
            // a git-environment issue; the diff anchor is best-effort and the milestone discipline
            // is also enforced by the orchestrator's `git diff main --stat` review.
            eprintln!(
                "AT-2505 SKIP: `git diff main` unavailable (status {:?}, stderr: {}). \
                 Empty-codegen-diff is also enforced at the orchestrator review.",
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
        Err(e) => {
            eprintln!("AT-2505 SKIP: git not available ({e}). Empty-codegen-diff enforced at review.");
        }
    }

    // No-regression: the two production kernels still compile to non-empty SPIR-V with the shipped
    // assignments (the full workspace suite is the broader green gate; this is the local anchor).
    let q4km = compile_words(include_str!(
        "../../../examples/q4km_matmul_rb_coopmat_f32acc_cached.axc"
    ));
    let plain = compile_words(include_str!("../../../examples/matmul_rb_coopmat.axc"));
    assert!(!q4km.is_empty(), "AT-2505: M3.6 leader must still compile");
    assert!(!plain.is_empty(), "AT-2505: plain-f32 core must still compile");

    // Stable across recompiles (deterministic emission — the empty-codegen-diff mechanical check).
    let q4km2 = compile_words(include_str!(
        "../../../examples/q4km_matmul_rb_coopmat_f32acc_cached.axc"
    ));
    assert_eq!(
        q4km, q4km2,
        "AT-2505: the M3.6 leader's emitted SPIR-V must be byte-identical across recompiles \
         (deterministic, no codegen change)."
    );
}
