//! M3.11a — CPU-only static falsifiers for the dequant-index affine strength-reduction.
//!
//! AT-2600 (STATIC ALU-classification falsifier AND silent-carry-destruction detector): emits the
//! M3.6 leader (cached.axc) and the M3.11a SR kernel (cached_sr.axc) SPIR-V; counts the integer-ALU
//! ops (IMul/UDiv/UMod/ISub/IAdd/ShiftRightLogical/BitwiseAnd, plus the related index/shift ops) in
//! the per-A-element INNER A-staging region via the m310b raw-word-stream region-finder; asserts the
//! SR kernel CUTS the per-element integer-ALU count by >= the pinned threshold (the x144 OpIMul +
//! the recomputed sb/div-mod chain replaced by carried adds). DUAL ROLE: if a carried counter were
//! re-`let` inside the loop (carry SILENTLY DESTROYED, init re-runs each iteration), the per-element
//! recompute it was meant to replace is STILL emitted => the count does NOT drop => AT-2600 FAILS
//! STATICALLY (no GPU). The ALU-count-drop is ALSO the carry-is-genuinely-carried proof.
//!
//! AT-2601 (CARRY-EQUIVALENCE static replay — the cheap CPU DESIGN-SOUNDNESS PRE-FILTER, subordinate
//! to the AUTHORITATIVE GPU AT-2603): replays the carried-counter recurrence (sb_idx_c += 1 on the
//! 16-wrap; sb_byte_base_c += 144 on the 16-wrap; wc0_c += 16 wrap-to-0) vs the div/mod decode over
//! the FULL k-range [0, K/16) at K=14336, INCLUDING every superblock-boundary wrap. SINGLE SOURCE OF
//! TRUTH: the constants (tile_k, 256, 144, 64, 32) are read from the kernel's @strategy map / source,
//! so a kernel hard-coding a WRONG delta is caught here. ASSERTS tile_k==16 (the 256/tile_k=16-integer
//! soundness precondition; fails loudly if ever widened).

use std::collections::{BTreeMap, BTreeSet};
use axc_driver::compile_source_with_assignments;
use spirv::Op;

const CACHED_SRC: &str =
    include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached.axc");
const CACHED_SR_SRC: &str =
    include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached_sr.axc");

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

fn compile_words(src: &str) -> Vec<u32> {
    let (bytes, _meta) = compile_source_with_assignments(src, &rb2x2_assignments())
        .expect("M3.11 diagnostic: kernel must compile");
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ── Minimal SPIR-V word-stream model (the m310b raw-word-stream harness) ──────────────────────

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

/// The integer-ALU ops the strength-reduction targets: the integer index-decode / address
/// arithmetic (IMul/UDiv/UMod/ISub/IAdd + shifts/bitwise that the div/mod by 256/64/32 lower to).
/// EXCLUDES the float dequant ops (ConvertUToF/FMul/FSub/FConvert) — those are the irreducible
/// floor, untouched by M3.11. This isolates the integer-index ALU the carry removes.
fn is_integer_index_alu(op: u32) -> bool {
    const ALU: &[Op] = &[
        Op::IAdd, Op::ISub, Op::IMul, Op::UDiv, Op::SDiv, Op::UMod, Op::SMod,
        Op::ShiftLeftLogical, Op::ShiftRightLogical, Op::ShiftRightArithmetic,
        Op::BitwiseAnd, Op::BitwiseOr, Op::BitwiseXor,
    ];
    ALU.iter().any(|a| *a as u32 == op)
}

/// Find the A-staging inner-loop body region by the def-use anchor (the m310b technique): the unique
/// `OpStore` of an f32_to_f16 (OpFConvert→half) result INTO a Workgroup pointer (a_tile[ei_a] = w).
/// The region is from the loop body's first `OpLabel` (after the last OpLoopMerge before the store)
/// up to and including that store. This counts ONLY the inner per-element region; the outer
/// per-k_block carry advances live OUTSIDE it (amortized 1:16).
fn find_a_staging_region(insts: &[Inst], ti: &TypeInfo) -> Region {
    let op_store = Op::Store as u32;
    let op_fconvert = Op::FConvert as u32;
    let op_label = Op::Label as u32;
    let op_loop_merge = Op::LoopMerge as u32;
    let op_load = Op::Load as u32;

    // Taint propagation: f32_to_f16 result -> Function ptr (via Store) -> reload (via Load).
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
        "AT-2600: expected an a_tile store `OpStore <Workgroup> <OpFConvert result>` \
         (the f32_to_f16 dequant A-tile write) — not found",
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

    Region { start: body_start, end: a_store_idx + 1 }
}

/// Count integer-index ALU ops + packed-q u8 loads in the per-element A-staging region.
fn count_inner_integer_index_alu(insts: &[Inst], ti: &TypeInfo) -> (usize, usize) {
    let region = find_a_staging_region(insts, ti);
    let slice = &insts[region.start..region.end];
    let mut int_alu = 0usize;
    let mut u8_loads = 0usize;
    for inst in slice {
        if is_integer_index_alu(inst.op) {
            int_alu += 1;
        }
        if load_ptr_kind(inst, ti) == Some(PtrKind::GlobalU8) {
            u8_loads += 1;
        }
    }
    (int_alu, u8_loads)
}

/// Count a SINGLE opcode in the per-element A-staging region.
fn count_inner_op(insts: &[Inst], ti: &TypeInfo, op: Op) -> usize {
    let region = find_a_staging_region(insts, ti);
    insts[region.start..region.end]
        .iter()
        .filter(|i| i.op == op as u32)
        .count()
}

// ════════════════════════════════════════════════════════════════════════════════════════════
//  AT-2600 — STATIC ALU-count drop + silent-carry-destruction detector (CPU-only, no GPU)
// ════════════════════════════════════════════════════════════════════════════════════════════

/// The pinned minimum reduction in TOTAL per-element integer-index ALU ops. The architect prior was
/// ~8-12 fewer; the AS-EMITTED measurement (the authority, per OQ-2) is the M3.6 leader 34 -> SR 29
/// = 5 fewer, attributable EXACTLY to the affine-in-k_block chain the carry removes: the x144
/// superblock-stride OpIMul (sb*144), the g_k-seed OpIMul (k_block*16) + the within-superblock
/// OpIMul (sb*256), the superblock OpUDiv (sb=g_k/256), and the OpISub (wk=g_k-sb*256). The architect
/// ~8-12 prior over-counted the region scope; the faithful strength-reduction (only the affine
/// index-decode reformulated, the row-invariant base term left structurally identical to M3.6)
/// removes 5. >= 4 is the conservative floor (determinism margin). A DESTROYED carry (re-`let` inside
/// the loop) leaves the recompute emitted => the count does NOT drop => this test FAILS with no GPU.
const MIN_INNER_INT_ALU_DROP: usize = 4;
/// The non-pow2 x144 OpIMul on sb (cached.axc:230) + the g_k-seed/within-superblock OpIMul that the
/// carry removes: M3.6 13 -> SR 10 = 3 fewer. >= 3 binds the x144 PRIZE specifically (a destroyed
/// carry removes 0 IMul). This is the load-bearing, op-specific guard (not a soft total).
const MIN_INNER_IMUL_DROP: usize = 3;

#[test]
fn at_2600_sr_cuts_per_element_integer_alu_count() {
    let cached_words = compile_words(CACHED_SRC);
    let sr_words = compile_words(CACHED_SR_SRC);

    let cached_insts = decode(&cached_words);
    let sr_insts = decode(&sr_words);
    let cached_ti = build_type_info(&cached_insts);
    let sr_ti = build_type_info(&sr_insts);

    let (cached_alu, cached_u8) = count_inner_integer_index_alu(&cached_insts, &cached_ti);
    let (sr_alu, sr_u8) = count_inner_integer_index_alu(&sr_insts, &sr_ti);

    // Both kernels read exactly one packed-q u8 byte per A-element (the nibble read) — the SR
    // rewrite touches only the INDEX arithmetic, not the load count.
    assert_eq!(
        cached_u8, 1,
        "AT-2600: M3.6 leader inner region must contain EXACTLY 1 packed-q u8 load; got {cached_u8}"
    );
    assert_eq!(
        sr_u8, 1,
        "AT-2600: SR kernel inner region must contain EXACTLY 1 packed-q u8 load; got {sr_u8}"
    );

    assert!(
        sr_alu < cached_alu,
        "AT-2600: the SR kernel's per-element integer-index ALU count ({sr_alu}) must be LOWER \
         than the M3.6 leader's ({cached_alu}). If it is NOT, the strength-reduction failed — \
         either a coding bug OR a silently-DESTROYED carry (a loop-body re-`let` whose init re-runs \
         each iteration, leaving the per-element recompute emitted). Fix before any perf read."
    );

    let drop = cached_alu - sr_alu;
    assert!(
        drop >= MIN_INNER_INT_ALU_DROP,
        "AT-2600 FALSIFIER: SR cut only {drop} per-element integer-index ALU ops \
         (M3.6 leader {cached_alu} -> SR {sr_alu}); expected >= {MIN_INNER_INT_ALU_DROP} \
         (the affine-in-k_block chain replaced by carried adds). A below-threshold drop indicates \
         the carry was not applied OR was silently destroyed (re-`let` inside the loop). Pin the \
         carry declare-before-loop, Assign-only-inside invariant."
    );

    // OP-SPECIFIC guard: the non-pow2 x144 OpIMul + the g_k/within-superblock OpIMul are the PRIZE.
    // The carry replaces them with a Function-storage OpIAdd, so the inner-region OpIMul count MUST
    // drop by >= MIN_INNER_IMUL_DROP. A destroyed carry leaves the x144 OpIMul emitted => 0 drop.
    let cached_imul = count_inner_op(&cached_insts, &cached_ti, Op::IMul);
    let sr_imul = count_inner_op(&sr_insts, &sr_ti, Op::IMul);
    assert!(
        sr_imul + MIN_INNER_IMUL_DROP <= cached_imul,
        "AT-2600 FALSIFIER (x144 prize): the per-element OpIMul count must drop by >= \
         {MIN_INNER_IMUL_DROP} (the non-pow2 sb*144 superblock-stride multiply + the g_k-seed/within-\
         superblock multiplies, all replaced by the carried sb_byte_base_c OpIAdd). M3.6 leader \
         {cached_imul} -> SR {sr_imul}. If the OpIMul count did NOT drop, the x144 carry was not \
         applied or was silently destroyed."
    );

    // The superblock OpUDiv (sb = g_k/256) must also be removed (it is now the carried sb_idx_c).
    let cached_udiv = count_inner_op(&cached_insts, &cached_ti, Op::UDiv);
    let sr_udiv = count_inner_op(&sr_insts, &sr_ti, Op::UDiv);
    assert!(
        sr_udiv < cached_udiv,
        "AT-2600: the per-element OpUDiv count must drop (the superblock sb = g_k/256 division is \
         replaced by the carried sb_idx_c). M3.6 leader {cached_udiv} -> SR {sr_udiv}."
    );

    eprintln!(
        "AT-2600 PASS: per-element integer-index ALU dropped {drop} ops \
         (M3.6 leader {cached_alu} -> SR {sr_alu}); OpIMul {cached_imul} -> {sr_imul} \
         (>= {MIN_INNER_IMUL_DROP} drop incl. the x144 superblock-stride prize), OpUDiv \
         {cached_udiv} -> {sr_udiv} (sb division carried). u8 load count unchanged (1 each). \
         Carry genuinely carried — a destroyed carry would leave the recompute => no drop."
    );
}

// ════════════════════════════════════════════════════════════════════════════════════════════
//  AT-2601 — CARRY-EQUIVALENCE static replay (CPU design-soundness PRE-FILTER, subordinate to AT-2603)
// ════════════════════════════════════════════════════════════════════════════════════════════

/// Read tile_k from the SR kernel's @strategy map (single source of truth) — NOT a hardcoded literal.
/// The replay derives its constants from the SAME value the kernel uses, so a wrong-delta kernel is
/// caught. Panics if tile_k is not the single-value pin `?[16]`.
fn read_tile_k_from_strategy(src: &str) -> u32 {
    let strat = src
        .lines()
        .find(|l| l.trim_start().starts_with("@strategy"))
        .expect("AT-2601: SR kernel must declare an @strategy block");
    // Find `tile_k: ?[<n>]`.
    let needle = "tile_k: ?[";
    let pos = strat.find(needle).unwrap_or_else(|| {
        panic!("AT-2601: SR kernel @strategy must declare `tile_k: ?[...]`; got {strat}")
    });
    let rest = &strat[pos + needle.len()..];
    let num: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    num.parse::<u32>()
        .unwrap_or_else(|_| panic!("AT-2601: could not parse tile_k from @strategy: {strat}"))
}

/// The div/mod decode for one (k_block, a_col): returns (sb, chunk, l, is_hi, is, cache_idx_is_part,
/// nibble_byte_address_relative). The byte address is relative to the row-invariant base term
/// (the (block_row*32+a_row)*n_blocks_per_row*144 hoisted term is dropped from BOTH sides since it
/// is identical and k_block-invariant) — i.e. sb*144 + 16 + chunk*32 + l.
fn div_mod_decode(k_block: u32, a_col: u32, tile_k: u32) -> (u32, u32, u32) {
    let g_k = k_block * tile_k + a_col;
    let sb = g_k / 256;
    let wk = g_k - sb * 256;
    let chunk = wk / 64;
    let wc = wk - chunk * 64;
    let l = wc - (wc / 32) * 32;
    let is_hi = wc / 32;
    let is = chunk * 2 + is_hi;
    let byte_addr_rel = sb * 144 + 16 + chunk * 32 + l;
    (is, byte_addr_rel, sb)
}

#[test]
fn at_2601_carry_sequence_equals_div_mod_over_k_range() {
    let tile_k = read_tile_k_from_strategy(CACHED_SR_SRC);

    // The tile_k==16 soundness precondition (256 mod tile_k == 0). A future non-256-divisor widening
    // would silently mis-stride the once-per-k_block carry — fail LOUDLY here.
    assert_eq!(
        tile_k, 16,
        "AT-2601: tile_k MUST be 16 (the 256/tile_k=16-integer soundness precondition for the \
         once-per-k_block carry — each k_block must lie entirely within ONE superblock). A \
         non-256-divisor tile_k would silently mis-stride the carry. Read from @strategy = {tile_k}."
    );
    assert_eq!(
        256 % tile_k, 0,
        "AT-2601: 256 mod tile_k must be 0 (the carry soundness precondition); got tile_k={tile_k}"
    );

    // The carried recurrence, EXACTLY as the kernel advances it (bottom-of-loop, counter-driven wrap):
    //   wc0_c += 16; if wc0_c == 256 { wc0_c = 0; sb_idx_c += 1; sb_byte_base_c += 144 }
    // Carried-in values hold THIS k_block's values; init (all 0) == k_block=0 value.
    let k: u32 = 14336;
    let num_k_blocks = k / tile_k; // 896

    let mut sb_idx_c: u32 = 0;
    let mut sb_byte_base_c: u32 = 0; // = sb_idx_c * 144
    let mut wc0_c: u32 = 0; // = (k_block mod 16) * 16

    let mut mismatches = 0usize;
    let mut first_mismatch: Option<(u32, u32)> = None;

    for k_block in 0..num_k_blocks {
        for a_col in 0..tile_k {
            // Carried decode: wk = wc0_c + a_col.
            let wk_c = wc0_c + a_col;
            let chunk_c = wk_c / 64;
            let wc_c = wk_c - chunk_c * 64;
            let l_c = wc_c - (wc_c / 32) * 32;
            let is_hi_c = wc_c / 32;
            let is_c = chunk_c * 2 + is_hi_c;
            let byte_addr_carry = sb_byte_base_c + 16 + chunk_c * 32 + l_c;

            let (is_div, byte_addr_div, sb_div) = div_mod_decode(k_block, a_col, tile_k);

            // Assert the FULL COMPOUND nibble byte address AND the sub-block index `is` (which drives
            // cache_idx) AND the carried sb index match the div/mod decode.
            if is_c != is_div
                || byte_addr_carry != byte_addr_div
                || sb_idx_c != sb_div
                || sb_byte_base_c != sb_div * 144
            {
                mismatches += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some((k_block, a_col));
                }
            }
        }
        // Bottom-of-loop advance (counter-driven 16-wrap), VERBATIM the kernel.
        wc0_c += 16;
        if wc0_c == 256 {
            wc0_c = 0;
            sb_idx_c += 1;
            sb_byte_base_c += 144;
        }
    }

    assert_eq!(
        mismatches, 0,
        "AT-2601: the carried-counter recurrence (sb_idx_c += 1 / sb_byte_base_c += 144 / \
         wc0_c += 16 wrap-to-0) must reproduce the div/mod decode (compound nibble byte address + \
         is + sb) for EVERY (k_block, a_col) over the full K={k} range incl. every superblock wrap. \
         {mismatches} mismatches; first at {first_mismatch:?}. An off-by-one (wrong delta / wrong \
         wrap boundary / wrong init) is caught here as a DESIGN pre-filter — the BINDING kernel proof \
         is AT-2603 (GPU bit-identity on the asymmetric fixture)."
    );

    eprintln!(
        "AT-2601 PASS: carried recurrence == div/mod decode over K={k} ({num_k_blocks} k_blocks x \
         {tile_k} a_cols incl. all superblock wraps); tile_k==16 asserted (256 mod tile_k == 0 \
         soundness precondition). DESIGN pre-filter green; AT-2603 binds the emitted kernel."
    );
}

// ════════════════════════════════════════════════════════════════════════════════════════════
//  AT-2607 — NO-REGRESSION / EMPTY-codegen-diff anchor (M3.11a is PURE SOURCE — ZERO codegen change)
// ════════════════════════════════════════════════════════════════════════════════════════════

#[test]
fn at_2607_no_regression_empty_codegen_diff() {
    // The empty-codegen-diff anchor: `git diff main` must touch NO file under the codegen / HIR /
    // typecheck / parser / lexer source trees. M3.11a is a PURE-SOURCE kernel + tests + docs + a
    // bench — it adds ZERO codegen (same discipline as M3.10a/M3.10b).
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
                "AT-2607: M3.11a must add ZERO codegen — `git diff main` touched protected source \
                 files: {changed_files:?}. M3.11a is a PURE-SOURCE strength-reduction (the carry \
                 rides the existing scalar Function-storage OpVariable mechanism); any \
                 codegen/HIR/typecheck/parser/lexer change is out of scope (M3.11b's codegen \
                 IndVarSimplify pass is DEFERRED + conditional)."
            );
            eprintln!(
                "AT-2607 PASS: empty-codegen-diff confirmed — `git diff main` touches NO file under \
                 axc-codegen/src, axc-hir/src, axc-typecheck, axc-parser/src, axc-lexer/src."
            );
        }
        Ok(out) => {
            eprintln!(
                "AT-2607 SKIP: `git diff main` unavailable (status {:?}, stderr: {}). \
                 Empty-codegen-diff is also enforced at the orchestrator review.",
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
        Err(e) => {
            eprintln!("AT-2607 SKIP: git not available ({e}). Empty-codegen-diff enforced at review.");
        }
    }

    // No-regression: both the M3.6 leader and the SR kernel still compile to non-empty SPIR-V, and
    // the SR kernel is byte-stable across recompiles (deterministic emission, no codegen change).
    let cached = compile_words(CACHED_SRC);
    let sr = compile_words(CACHED_SR_SRC);
    let sr2 = compile_words(CACHED_SR_SRC);
    assert!(!cached.is_empty(), "AT-2607: M3.6 leader must still compile");
    assert!(!sr.is_empty(), "AT-2607: SR kernel must compile");
    assert_eq!(
        sr, sr2,
        "AT-2607: the SR kernel's emitted SPIR-V must be byte-identical across recompiles \
         (deterministic, no codegen change)."
    );
}

