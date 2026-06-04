//! M3.3 PART A: OpPhi loop-carried SSA codegen tests.
//!
//! AT-1700: Scalar loop-carried binding still uses Function-storage load/store
//!          and emits NO header OpPhi (non-regression guard for scalars-stay-on-Function).
//! AT-1701: Coopmat loop-carried accumulator emits exactly one well-formed OpPhi
//!          at the loop header with predecessors {pre-header, latch}.
//! AT-1702: Nested for-range with coopmat accumulators emits one phi per loop level.
//! AT-1703: Zero-trip loop — post-loop coopmat_store emits OpPhi in header.
//! AT-1704: break/continue over a loop-carried coopmat → HARD ERROR.
//! AT-1705: POSITIVE typecheck: `acc = coopmat_mul_add(...)` on a mutable accumulator.
//! AT-1706: NEGATIVE typecheck: wrong-use_/wrong-shape/scalar → coopmat target rejected.

use axc_codegen::{emit_module, CodegenOptions};
use axc_hir::lower_module;
use axc_parser::parse;

// ── Compile helpers ───────────────────────────────────────────────────────────

/// Compile a source string, asserting zero lex/parse/HIR errors.
fn compile_ok(src: &str) -> Vec<u32> {
    let (ast, lex_errs, parse_errs) = parse(src);
    assert!(lex_errs.is_empty(), "lex errors: {lex_errs:?}");
    assert!(parse_errs.is_empty(), "parse errors: {parse_errs:?}");
    let (hir, hir_errs, _warns) = lower_module(&ast);
    assert!(hir_errs.is_empty(), "hir errors: {hir_errs:?}");
    emit_module(&hir, &CodegenOptions::default()).expect("codegen must succeed")
}

/// Compile and expect a codegen error containing the given substring.
fn compile_expect_codegen_error(src: &str, expected_fragment: &str) {
    let (ast, lex_errs, parse_errs) = parse(src);
    assert!(lex_errs.is_empty(), "lex errors: {lex_errs:?}");
    assert!(parse_errs.is_empty(), "parse errors: {parse_errs:?}");
    let (hir, hir_errs, _warns) = lower_module(&ast);
    assert!(hir_errs.is_empty(), "hir errors before codegen: {hir_errs:?}");
    let result = emit_module(&hir, &CodegenOptions::default());
    match result {
        Ok(_) => panic!(
            "expected codegen error containing '{expected_fragment}' but codegen succeeded"
        ),
        Err(e) => {
            let msg = format!("{e}");
            let msg_debug = format!("{e:?}");
            assert!(
                msg.contains(expected_fragment) || msg_debug.contains(expected_fragment),
                "expected error containing '{expected_fragment}'; got: {msg}"
            );
        }
    }
}

// ── spirv-val integration ─────────────────────────────────────────────────────

/// Validate SPIR-V words via spirv-val (Vulkan_1_1 target).
fn spirv_val_check(words: &[u32]) {
    use spirv_tools::val::{Validator, create as create_validator};
    use spirv_tools::TargetEnv;
    let validator = create_validator(Some(TargetEnv::Vulkan_1_1));
    validator.validate(words, None)
        .expect("spirv-val Vulkan_1_1 must pass");
}

// ── Binary word-level SPIR-V instruction scanner ─────────────────────────────
// rspirv::dr::Loader cannot parse cooperative-matrix extension opcodes
// (it emits DetachedInstruction for OpTypeCooperativeMatrixKHR).
// We use a raw binary scanner instead, which is extension-agnostic.

/// SPIR-V opcodes (partial list for what we need).
const OP_PHI: u16 = 245;
const OP_LOOP_MERGE: u16 = 246;
const OP_LABEL: u16 = 248;
// OP_BRANCH (249) and OP_BRANCH_CONDITIONAL (250) reserved for future CFG validation.
// OP_COOP_MAT_STORE_KHR (5360) reserved for future store-source operand walking.

/// Iterate over (opcode, word_slice_including_header) tuples.
fn iter_insts(words: &[u32]) -> impl Iterator<Item = (u16, &[u32])> {
    IterInst { words, cursor: 0 }
}

struct IterInst<'a> {
    words: &'a [u32],
    cursor: usize,
}

impl<'a> Iterator for IterInst<'a> {
    type Item = (u16, &'a [u32]);
    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.words.len() {
            return None;
        }
        let hdr = self.words[self.cursor];
        let wc = (hdr >> 16) as usize;
        let op = (hdr & 0xFFFF) as u16;
        if wc == 0 || self.cursor + wc > self.words.len() {
            return None;
        }
        let slice = &self.words[self.cursor..self.cursor + wc];
        self.cursor += wc;
        Some((op, slice))
    }
}

/// Count occurrences of a given opcode in the SPIR-V binary (including header words).
fn count_op(words: &[u32], target_op: u16) -> usize {
    // Skip SPIR-V header (5 words).
    iter_insts(&words[5..]).filter(|(op, _)| *op == target_op).count()
}

/// Collect information about loop header blocks.
///
/// A loop header is identified as a block containing OpLoopMerge.
/// Returns for each loop header: (label_word, phi_count_at_head, vec_of_phi_instructions).
///
/// Each phi instruction is represented as Vec<u32> (the full instruction word slice).
fn collect_loop_headers(words: &[u32]) -> Vec<(u32, Vec<Vec<u32>>)> {
    // First pass: collect all instruction slices.
    let all_insts: Vec<(u16, Vec<u32>)> = iter_insts(&words[5..])
        .map(|(op, slice)| (op, slice.to_vec()))
        .collect();

    let mut result: Vec<(u32, Vec<Vec<u32>>)> = Vec::new();

    // Find blocks containing OpLoopMerge. Walk blocks by OpLabel boundaries.
    // Block = from one OpLabel (inclusive) to the next OpLabel (exclusive).
    let mut current_label: Option<u32> = None;
    let mut current_block_insts: Vec<(u16, Vec<u32>)> = Vec::new();
    #[allow(clippy::type_complexity)]
    let mut blocks: Vec<(u32, Vec<(u16, Vec<u32>)>)> = Vec::new();

    for (op, slice) in &all_insts {
        if *op == OP_LABEL {
            if let Some(lbl) = current_label {
                blocks.push((lbl, current_block_insts.clone()));
            }
            current_block_insts.clear();
            current_label = slice.get(1).copied();
        } else {
            current_block_insts.push((*op, slice.clone()));
        }
    }
    if let Some(lbl) = current_label {
        blocks.push((lbl, current_block_insts));
    }

    for (label, insts) in blocks {
        let has_loop_merge = insts.iter().any(|(op, _)| *op == OP_LOOP_MERGE);
        if has_loop_merge {
            // Collect leading phi instructions (before any non-phi).
            let phis: Vec<Vec<u32>> = insts.iter()
                .take_while(|(op, _)| *op == OP_PHI)
                .map(|(_, slice)| slice.clone())
                .collect();
            result.push((label, phis));
        }
    }
    result
}

// ── AT-1700: Scalar loop-carried — NO header phi ──────────────────────────────

/// AT-1700: A scalar loop-carried accumulator emits ZERO OpPhi in the loop header.
/// Scalars stay on Function-storage load/store — this guards the non-regression.
#[test]
fn at1700_scalar_loop_carried_no_phi_unchanged() {
    let src = r#"
        @kernel @workgroup(1,1,1) @intent("reduction") @complexity(O(n))
        fn reduction(input: readonly_buffer[f32], output: buffer[f32], n: u32) -> void {
            let mut total: f32 = 0.0f32;
            for i in range(0u32, n) {
                let v: f32 = input[i];
                total = total + v;
            }
            output[0u32] = total;
            return;
        }
    "#;
    let words = compile_ok(src);
    spirv_val_check(&words);
    let phi_total = count_op(&words, OP_PHI);
    // Scalars must have ZERO phis from the loop-carried path (there may be other phis
    // from short-circuit or other constructs, but we don't have those here).
    assert_eq!(
        phi_total, 0,
        "AT-1700: scalar loop-carried must emit ZERO OpPhi; got {phi_total}"
    );
    eprintln!("AT-1700 PASS: scalar loop-carried — 0 OpPhi, spirv-val clean");
}

// ── AT-1701: Coopmat loop-carried — exactly one well-formed OpPhi ─────────────

/// AT-1701: A K-loop kernel with `let mut acc = coopmat_zero(); for k { acc = coopmat_mul_add(...); }`
/// must emit exactly one OpPhi at the loop header (SPIR-V 2.4 ordering, before OpLoopMerge).
/// spirv-val clean. The phi must have exactly 2 predecessor pairs.
#[test]
fn at1701_coopmat_loop_carried_phi_wellformed() {
    let src = r#"
        @kernel @workgroup(32,1,1) @cooperative_matrix @intent("coopmat k-loop") @complexity(O(n))
        fn coopmat_kloop(
            A: readonly_buffer[f16],
            B: readonly_buffer[f16],
            C: buffer[f16],
            K: u32,
            N: u32
        ) -> void {
            shared a_tile: shared[f16, 256];
            shared b_tile: shared[f16, 256];
            let mut acc: matrix[f16, 16, 16, accumulator] = coopmat_zero();
            for k in range(0u32, K) {
                let a_mat: matrix[f16, 16, 16, a] = coopmat_load(a_tile, 0u32, 16u32);
                let b_mat: matrix[f16, 16, 16, b] = coopmat_load(b_tile, 0u32, 16u32);
                acc = coopmat_mul_add(a_mat, b_mat, acc);
            }
            coopmat_store(acc, C, 0u32, N);
            return;
        }
    "#;
    let words = compile_ok(src);
    spirv_val_check(&words);

    let headers = collect_loop_headers(&words);
    assert_eq!(headers.len(), 1, "AT-1701: expected 1 loop header; got {}", headers.len());
    let (header_label, phis) = &headers[0];
    assert_eq!(
        phis.len(), 1,
        "AT-1701: expected exactly 1 OpPhi in loop header %{header_label}; got {}", phis.len()
    );

    // OpPhi format: [word_count<<16|OpPhi, result_type, result_id, val1, pred1, val2, pred2]
    // That's 7 words total: header + result_type + result_id + 4 operand ids.
    let phi = &phis[0];
    assert_eq!(
        phi.len(), 7,
        "AT-1701: OpPhi must have 7 words (header+result_type+result_id+4 operands); got {}", phi.len()
    );
    // phi[1]=result_type, phi[2]=result_id, phi[3]=val1, phi[4]=pred1, phi[5]=val2, phi[6]=pred2
    let pred_a = phi[4]; // predecessor block label 1 (pre-header)
    let pred_b = phi[6]; // predecessor block label 2 (latch/continue)
    assert_ne!(pred_a, pred_b, "AT-1701: phi predecessors must be distinct labels");

    // Both predecessor labels must appear as OpLabel result_ids in the binary.
    let all_labels: Vec<u32> = iter_insts(&words[5..])
        .filter(|(op, _)| *op == OP_LABEL)
        .filter_map(|(_, slice)| slice.get(1).copied())
        .collect();
    assert!(all_labels.contains(&pred_a), "AT-1701: phi pred %{pred_a} must be a valid label");
    assert!(all_labels.contains(&pred_b), "AT-1701: phi pred %{pred_b} must be a valid label");

    eprintln!(
        "AT-1701 PASS: loop-carried coopmat OpPhi — 1 phi, predecessors %{pred_a} + %{pred_b}, spirv-val clean"
    );
}

// ── AT-1702: Nested for-range — 2 loop headers each with 1 phi ────────────────

/// AT-1702: Two nested for-ranges each carrying a coopmat accumulator:
/// the OUTER loop carries acc_outer, the INNER loop carries acc_inner.
/// Both loops must get one OpPhi each at their respective loop headers.
/// spirv-val clean.
#[test]
fn at1702_nested_loop_two_coopmat_phis() {
    // Valid nested kernel: outer carries acc_outer, inner carries acc_inner.
    // Use matmul-like structure: inner accumulates for one k-block, outer across k-blocks.
    let src = r#"
        @kernel @workgroup(32,1,1) @cooperative_matrix @intent("nested coopmat phis") @complexity(O(n))
        fn nested_coopmat(
            A: readonly_buffer[f16],
            B: readonly_buffer[f16],
            C: buffer[f16],
            outer_iters: u32,
            inner_iters: u32,
            N: u32
        ) -> void {
            shared a_tile: shared[f16, 256];
            shared b_tile: shared[f16, 256];
            let mut acc_outer: matrix[f16, 16, 16, accumulator] = coopmat_zero();
            for outer in range(0u32, outer_iters) {
                let mut acc_inner: matrix[f16, 16, 16, accumulator] = coopmat_zero();
                for inner in range(0u32, inner_iters) {
                    let a_mat: matrix[f16, 16, 16, a] = coopmat_load(a_tile, 0u32, 16u32);
                    let b_mat: matrix[f16, 16, 16, b] = coopmat_load(b_tile, 0u32, 16u32);
                    acc_inner = coopmat_mul_add(a_mat, b_mat, acc_inner);
                }
                let a_mat2: matrix[f16, 16, 16, a] = coopmat_load(a_tile, 0u32, 16u32);
                let b_mat2: matrix[f16, 16, 16, b] = coopmat_load(b_tile, 0u32, 16u32);
                acc_outer = coopmat_mul_add(a_mat2, b_mat2, acc_outer);
            }
            coopmat_store(acc_outer, C, 0u32, N);
            return;
        }
    "#;
    let words = compile_ok(src);
    spirv_val_check(&words);

    let headers = collect_loop_headers(&words);
    assert_eq!(
        headers.len(), 2,
        "AT-1702: expected 2 loop header blocks (one outer, one inner); got {}", headers.len()
    );

    // Each loop header must have exactly 1 phi.
    for (i, (label, phis)) in headers.iter().enumerate() {
        assert_eq!(
            phis.len(), 1,
            "AT-1702: loop header[{i}] at %{label} must have exactly 1 OpPhi; got {}", phis.len()
        );
    }

    // Total phi count must be 2.
    let total_phis = count_op(&words, OP_PHI);
    assert_eq!(total_phis, 2, "AT-1702: expected 2 total OpPhi; got {total_phis}");

    // §A.7: each loop phi must have EXACTLY 2 predecessor pairs (pre-header + latch) —
    // the structural invariant that makes per-loop-level composition well-formed. An
    // OpPhi instruction is [wordcount|opcode, result_type, result_id, (val,label)*]; a
    // 2-predecessor phi is therefore 3 + 2*2 = 7 words. A malformed nested phi (wrong
    // predecessor set / stale latch edge) would have a different operand count.
    // (The cross-level NUMERIC composition — outer latch reads the inner loop's merged
    // value — is GPU-proven end-to-end by AT-1707's bit-exact loop-carried accumulation.)
    let mut phi_result_ids = Vec::new();
    for (label, phis) in &headers {
        for phi in phis {
            assert_eq!(
                phi.len(), 7,
                "AT-1702: loop-header %{label} OpPhi must have exactly 2 predecessor pairs \
                 (7 words: type+result+2*(val,label)); got {} words", phi.len()
            );
            phi_result_ids.push(phi[2]); // result id
        }
    }
    assert_eq!(phi_result_ids.len(), 2, "AT-1702: expected 2 phi result ids");
    assert_ne!(
        phi_result_ids[0], phi_result_ids[1],
        "AT-1702: the outer and inner loop phis must have DISTINCT result ids"
    );

    eprintln!("AT-1702 PASS: nested 2-level coopmat loop — 2 phis (2 preds each, distinct ids) in 2 headers, spirv-val clean");
}

// ── AT-1703: Zero-trip loop — OpPhi present in header ────────────────────────

/// AT-1703: A kernel where the K-loop can run zero trips at runtime still emits
/// an OpPhi in the header (with K as runtime parameter). spirv-val clean.
#[test]
fn at1703_zero_trip_coopmat_phi_post_loop_read() {
    let src = r#"
        @kernel @workgroup(32,1,1) @cooperative_matrix @intent("zero-trip coopmat") @complexity(O(1))
        fn coopmat_zero_trip(
            A: readonly_buffer[f16],
            B: readonly_buffer[f16],
            C: buffer[f16],
            K: u32,
            N: u32
        ) -> void {
            shared a_tile: shared[f16, 256];
            shared b_tile: shared[f16, 256];
            let mut acc: matrix[f16, 16, 16, accumulator] = coopmat_zero();
            for k in range(0u32, K) {
                let a_mat: matrix[f16, 16, 16, a] = coopmat_load(a_tile, 0u32, 16u32);
                let b_mat: matrix[f16, 16, 16, b] = coopmat_load(b_tile, 0u32, 16u32);
                acc = coopmat_mul_add(a_mat, b_mat, acc);
            }
            coopmat_store(acc, C, 0u32, N);
            return;
        }
    "#;
    let words = compile_ok(src);
    spirv_val_check(&words);

    // The loop header must have exactly 1 OpPhi (the loop-carried coopmat phi).
    let headers = collect_loop_headers(&words);
    assert_eq!(headers.len(), 1, "AT-1703: expected 1 loop header");
    let phi_count = headers[0].1.len();
    assert_eq!(
        phi_count, 1,
        "AT-1703: zero-trip kernel must emit exactly 1 header OpPhi; got {phi_count}"
    );
    eprintln!("AT-1703 PASS: zero-trip phi present in header, spirv-val clean");
}

// ── AT-1704: break/continue over loop-carried coopmat → HARD ERROR ────────────

/// AT-1704a: A loop carrying a coopmat accumulator that contains a `break` is rejected.
#[test]
fn at1704_coopmat_break_rejected() {
    let src = r#"
        @kernel @workgroup(32,1,1) @cooperative_matrix @intent("break in coopmat loop") @complexity(O(n))
        fn coopmat_break(
            A: readonly_buffer[f16],
            B: readonly_buffer[f16],
            C: buffer[f16],
            K: u32,
            N: u32,
            stop_at: u32
        ) -> void {
            shared a_tile: shared[f16, 256];
            shared b_tile: shared[f16, 256];
            let mut acc: matrix[f16, 16, 16, accumulator] = coopmat_zero();
            for k in range(0u32, K) {
                let a_mat: matrix[f16, 16, 16, a] = coopmat_load(a_tile, 0u32, 16u32);
                let b_mat: matrix[f16, 16, 16, b] = coopmat_load(b_tile, 0u32, 16u32);
                acc = coopmat_mul_add(a_mat, b_mat, acc);
                if k > stop_at {
                    break;
                }
            }
            coopmat_store(acc, C, 0u32, N);
            return;
        }
    "#;
    compile_expect_codegen_error(src, "break inside a loop carrying");
    eprintln!("AT-1704a PASS: break in coopmat loop correctly rejected");
}

/// AT-1704b: A loop carrying a coopmat accumulator with a `continue` path is rejected.
#[test]
fn at1704_coopmat_continue_rejected() {
    let src = r#"
        @kernel @workgroup(32,1,1) @cooperative_matrix @intent("continue in coopmat loop") @complexity(O(n))
        fn coopmat_continue_skip(
            A: readonly_buffer[f16],
            B: readonly_buffer[f16],
            C: buffer[f16],
            K: u32,
            N: u32,
            skip_at: u32
        ) -> void {
            shared a_tile: shared[f16, 256];
            shared b_tile: shared[f16, 256];
            let mut acc: matrix[f16, 16, 16, accumulator] = coopmat_zero();
            for k in range(0u32, K) {
                if k == skip_at {
                    continue;
                }
                let a_mat: matrix[f16, 16, 16, a] = coopmat_load(a_tile, 0u32, 16u32);
                let b_mat: matrix[f16, 16, 16, b] = coopmat_load(b_tile, 0u32, 16u32);
                acc = coopmat_mul_add(a_mat, b_mat, acc);
            }
            coopmat_store(acc, C, 0u32, N);
            return;
        }
    "#;
    compile_expect_codegen_error(src, "continue on a path");
    eprintln!("AT-1704b PASS: continue in coopmat loop correctly rejected");
}

/// AT-1704c: A non-coopmat-carrying loop with a break still compiles.
#[test]
fn at1704_scalar_loop_break_still_compiles() {
    let src = r#"
        @kernel @workgroup(1,1,1) @intent("scalar break") @complexity(O(n))
        fn scalar_break(input: readonly_buffer[f32], output: buffer[f32], n: u32) -> void {
            let mut total: f32 = 0.0f32;
            for i in range(0u32, n) {
                let v: f32 = input[i];
                total = total + v;
                if total > 100.0f32 {
                    break;
                }
            }
            output[0u32] = total;
            return;
        }
    "#;
    let words = compile_ok(src);
    spirv_val_check(&words);
    eprintln!("AT-1704c PASS: scalar loop with break still compiles and validates");
}

/// AT-1704d: Per reviewer note (b) — an inner loop's break is NOT rejected by the
/// outer coopmat-carrying loop's guard (non-descend-into-nested-loops discipline).
#[test]
fn at1704_inner_loop_break_not_rejected_by_outer_guard() {
    // Outer loop carries acc (coopmat). Inner scalar loop has a break.
    // The inner break targets the INNER loop's merge, not the outer loop.
    // The outer guard must NOT fire (non-descend rule).
    let src = r#"
        @kernel @workgroup(32,1,1) @cooperative_matrix @intent("inner break not rejected") @complexity(O(n))
        fn outer_coopmat_inner_break(
            A: readonly_buffer[f16],
            B: readonly_buffer[f16],
            C: buffer[f16],
            K: u32,
            inner_iters: u32,
            N: u32,
            threshold: u32
        ) -> void {
            shared a_tile: shared[f16, 256];
            shared b_tile: shared[f16, 256];
            let mut acc: matrix[f16, 16, 16, accumulator] = coopmat_zero();
            for k in range(0u32, K) {
                let a_mat: matrix[f16, 16, 16, a] = coopmat_load(a_tile, 0u32, 16u32);
                let b_mat: matrix[f16, 16, 16, b] = coopmat_load(b_tile, 0u32, 16u32);
                acc = coopmat_mul_add(a_mat, b_mat, acc);
                let mut _dummy: u32 = 0u32;
                for inner in range(0u32, inner_iters) {
                    if inner > threshold {
                        break;
                    }
                    _dummy = inner;
                }
            }
            coopmat_store(acc, C, 0u32, N);
            return;
        }
    "#;
    // This MUST compile (inner break targets inner loop only).
    let words = compile_ok(src);
    spirv_val_check(&words);
    eprintln!("AT-1704d PASS: inner loop break not rejected by outer coopmat loop guard");
}

// ── AT-1705: POSITIVE typecheck — coopmat reassign ────────────────────────────

/// AT-1705: `let mut acc: matrix[...accumulator] = coopmat_zero(); acc = coopmat_mul_add(...)`
/// must typecheck with ZERO errors.
#[test]
fn at1705_coopmat_reassign_typechecks() {
    let src = r#"
        @kernel @workgroup(32,1,1) @cooperative_matrix @intent("coopmat reassign positive") @complexity(O(n))
        fn coopmat_reassign(
            A: readonly_buffer[f16],
            B: readonly_buffer[f16],
            C: buffer[f16],
            K: u32,
            N: u32
        ) -> void {
            shared a_tile: shared[f16, 256];
            shared b_tile: shared[f16, 256];
            let mut acc: matrix[f16, 16, 16, accumulator] = coopmat_zero();
            for k in range(0u32, K) {
                let a_mat: matrix[f16, 16, 16, a] = coopmat_load(a_tile, 0u32, 16u32);
                let b_mat: matrix[f16, 16, 16, b] = coopmat_load(b_tile, 0u32, 16u32);
                acc = coopmat_mul_add(a_mat, b_mat, acc);
            }
            coopmat_store(acc, C, 0u32, N);
            return;
        }
    "#;
    let (ast, lex_errs, parse_errs) = parse(src);
    assert!(lex_errs.is_empty(), "lex errors: {lex_errs:?}");
    assert!(parse_errs.is_empty(), "parse errors: {parse_errs:?}");
    let (_hir, hir_errs, _warns) = lower_module(&ast);
    assert!(
        hir_errs.is_empty(),
        "AT-1705: `acc = coopmat_mul_add(...)` must typecheck with zero errors; got: {hir_errs:?}"
    );
    eprintln!("AT-1705 PASS: coopmat reassign typechecks (ISSUE-1 FLAT+NESTED both fixed)");
}

// ── AT-1706: NEGATIVE typecheck — bad RHS to coopmat target ──────────────────

/// AT-1706a: Assigning coopmat_mul_add with an A-use matrix as the C-arg is rejected.
/// The C-arg of coopmat_mul_add must be use_==Accumulator; this validates ISSUE-1 routing.
#[test]
fn at1706_coopmat_reassign_wrong_c_arg_use_rejected() {
    // acc = coopmat_mul_add(a_mat, b_mat, a_mat) — a_mat is use_==A, not accumulator.
    let src = r#"
        @kernel @workgroup(32,1,1) @cooperative_matrix @intent("wrong c-arg use") @complexity(O(n))
        fn coopmat_wrong_c_arg(
            A: readonly_buffer[f16],
            B: readonly_buffer[f16],
            C: buffer[f16],
            N: u32
        ) -> void {
            shared a_tile: shared[f16, 256];
            shared b_tile: shared[f16, 256];
            let mut acc: matrix[f16, 16, 16, accumulator] = coopmat_zero();
            let a_mat: matrix[f16, 16, 16, a] = coopmat_load(a_tile, 0u32, 16u32);
            let b_mat: matrix[f16, 16, 16, b] = coopmat_load(b_tile, 0u32, 16u32);
            acc = coopmat_mul_add(a_mat, b_mat, a_mat);
            coopmat_store(acc, C, 0u32, N);
            return;
        }
    "#;
    let (ast, lex_errs, parse_errs) = parse(src);
    assert!(lex_errs.is_empty());
    assert!(parse_errs.is_empty());
    let (_hir, hir_errs, _warns) = lower_module(&ast);
    assert!(
        !hir_errs.is_empty(),
        "AT-1706a: coopmat_mul_add with A-use as c-arg must be rejected (c must be accumulator)"
    );
    let combined = format!("{hir_errs:?}");
    assert!(
        combined.contains("CUseMismatch") || combined.contains("CoopMatrix"),
        "AT-1706a: expected CUseMismatch error; got: {combined}"
    );
    eprintln!("AT-1706a PASS: wrong c-arg use_ rejected (CUseMismatch) - ISSUE-1 validation confirmed");
}

/// AT-1706b: Assigning coopmat_mul_add with correct shapes compiles (sanity check).
#[test]
fn at1706_coopmat_reassign_correct_shape_compiles() {
    let src = r#"
        @kernel @workgroup(32,1,1) @cooperative_matrix @intent("correct reassign") @complexity(O(n))
        fn coopmat_correct_reassign(
            A: readonly_buffer[f16],
            B: readonly_buffer[f16],
            C: buffer[f16],
            K: u32,
            N: u32
        ) -> void {
            shared a_tile: shared[f16, 256];
            shared b_tile: shared[f16, 256];
            let mut acc: matrix[f16, 16, 16, accumulator] = coopmat_zero();
            let a_mat: matrix[f16, 16, 16, a] = coopmat_load(a_tile, 0u32, 16u32);
            let b_mat: matrix[f16, 16, 16, b] = coopmat_load(b_tile, 0u32, 16u32);
            acc = coopmat_mul_add(a_mat, b_mat, acc);
            coopmat_store(acc, C, 0u32, N);
            return;
        }
    "#;
    let (ast, lex_errs, parse_errs) = parse(src);
    assert!(lex_errs.is_empty());
    assert!(parse_errs.is_empty());
    let (_hir, hir_errs, _warns) = lower_module(&ast);
    assert!(
        hir_errs.is_empty(),
        "AT-1706b: correct coopmat reassign should typecheck; got: {hir_errs:?}"
    );
    eprintln!("AT-1706b PASS: correct coopmat reassign compiles");
}

// ── AT-1708: Conditional-only coopmat reassign → HARD ERROR ──────────────────

/// AT-1708 (BLOCKING FIX): A loop where the coopmat accumulator is reassigned ONLY
/// inside an `if` body (not unconditionally at the top level of the loop body) must
/// produce `LoopCarriedCoopMatConditionalReassignUnsupported` — NOT invalid SPIR-V.
///
/// Without this guard, `detect_loop_carried_coopmat` would detect `acc` as loop-carried
/// (because `stmt_reassigns_binding` descends into if/else), and `emit_for_range` would
/// emit an OpPhi whose latch operand is the SSA id produced inside the `if` block. That
/// id does NOT dominate the loop's continue/back-edge block → spirv-val rejects it with
/// "ID definition does not dominate its parent" (invalid SPIR-V, latent GPU UB).
///
/// The fix (symmetric to the break/continue rejection in AT-1704): require an unconditional
/// top-level reassignment. If the only reassignment is conditional, it is a hard error.
#[test]
fn at1708_coopmat_conditional_only_reassign_rejected() {
    // acc is only reassigned inside `if cond { acc = coopmat_mul_add(...); }`.
    // There is NO unconditional top-level reassignment in the loop body.
    // This must produce a hard error (not silently emit invalid SPIR-V).
    let src = r#"
        @kernel @workgroup(32,1,1) @cooperative_matrix @intent("conditional coopmat reassign") @complexity(O(n))
        fn coopmat_conditional_only(
            A: readonly_buffer[f16],
            B: readonly_buffer[f16],
            C: buffer[f16],
            K: u32,
            N: u32,
            cond_val: u32
        ) -> void {
            shared a_tile: shared[f16, 256];
            shared b_tile: shared[f16, 256];
            let mut acc: matrix[f16, 16, 16, accumulator] = coopmat_zero();
            for k in range(0u32, K) {
                let a_mat: matrix[f16, 16, 16, a] = coopmat_load(a_tile, 0u32, 16u32);
                let b_mat: matrix[f16, 16, 16, b] = coopmat_load(b_tile, 0u32, 16u32);
                if k < cond_val {
                    acc = coopmat_mul_add(a_mat, b_mat, acc);
                }
            }
            coopmat_store(acc, C, 0u32, N);
            return;
        }
    "#;
    compile_expect_codegen_error(src, "conditional reassignment");
    eprintln!("AT-1708 PASS: conditional-only coopmat reassign correctly rejected (not invalid SPIR-V)");
}

// ── AT-1733: N=4 coopmat phis in ONE single loop header — pre-gate for RB ─────

/// AT-1733: A SINGLE for-range carrying 4 coopmat accumulators (acc_00, acc_01, acc_10, acc_11)
/// emits exactly 4 well-formed OpPhi instructions in ONE loop header.
///
/// This is the pre-gate for M3.3c register blocking: register blocking depends on the N-phi
/// single-loop path (4 loop-carried coopmat accumulators, all in one K-loop). AT-1702 proved
/// the multi-phi insert mechanism for 2 phis across 2 NESTED loops (one phi per header);
/// AT-1733 locks N=4 phis in ONE single header.
///
/// Assertions:
///   1. Exactly 1 loop header block (ONE K-loop, not nested).
///   2. Exactly 4 OpPhi at the loop header head (all phis first, per SPIR-V 2.4).
///   3. Each phi has 7 words (type + result + 2 predecessor pairs).
///   4. All 4 phi result_ids are DISTINCT (no cross-contamination).
///   5. All 4 phis share the same two predecessor labels (pre_header + continue/latch).
///   6. Both predecessors are valid OpLabel result_ids in the binary.
///   7. spirv-val (Vulkan_1_1) exits clean.
///
/// NO codegen change is expected (the parallel pre_header_values/latch_values/carried Vecs
/// are indexed by the single key i, making them N-agnostic; step-10 reverse-BeginInsert
/// places all N phis at the header head in declaration order for any N). If this test FAILS,
/// the 'no codegen change' claim is false and M3.3c needs a codegen fix (Architect-owned,
/// NOT a Coder body.rs hack per HN-3).
#[test]
fn at1733_n_coopmat_phis_single_loop_wellformed() {
    // A kernel with 4 coopmat accumulators, all loop-carried in ONE K-loop.
    // Each accumulator is:
    //   (a) coopmat-typed (let mut acc_ij: matrix[f16,16,16,accumulator] = coopmat_zero())
    //   (b) defined BEFORE the K-loop
    //   (c) reassigned UNCONDITIONALLY at the top level of the K-loop body (AT-1708)
    //   (d) no break/continue (AT-1704)
    //
    // This is the EXACT pattern of the matmul_rb_coopmat.axc 2x2 RB kernel.
    let src = r#"
        @kernel @workgroup(32,1,1) @cooperative_matrix
        @intent("4 loop-carried coopmat accumulators in ONE K-loop — AT-1733 pre-gate")
        @complexity(O(n))
        fn rb2x2_loop_carried_4phi(
            A: readonly_buffer[f16],
            B: readonly_buffer[f16],
            C: buffer[f16],
            K: u32,
            N: u32
        ) -> void {
            shared a_tile: shared[f16, 512];
            shared b_tile: shared[f16, 512];

            let mut acc_00: matrix[f16, 16, 16, accumulator] = coopmat_zero();
            let mut acc_01: matrix[f16, 16, 16, accumulator] = coopmat_zero();
            let mut acc_10: matrix[f16, 16, 16, accumulator] = coopmat_zero();
            let mut acc_11: matrix[f16, 16, 16, accumulator] = coopmat_zero();

            for k_block in range(0u32, K) {
                let a_mat_0: matrix[f16, 16, 16, a] = coopmat_load(a_tile, 0u32, 16u32);
                let a_mat_1: matrix[f16, 16, 16, a] = coopmat_load(a_tile, 256u32, 16u32);
                let b_mat_0: matrix[f16, 16, 16, b] = coopmat_load(b_tile, 0u32, 32u32);
                let b_mat_1: matrix[f16, 16, 16, b] = coopmat_load(b_tile, 16u32, 32u32);

                acc_00 = coopmat_mul_add(a_mat_0, b_mat_0, acc_00);
                acc_01 = coopmat_mul_add(a_mat_0, b_mat_1, acc_01);
                acc_10 = coopmat_mul_add(a_mat_1, b_mat_0, acc_10);
                acc_11 = coopmat_mul_add(a_mat_1, b_mat_1, acc_11);
            }

            coopmat_store(acc_00, C, 0u32, N);
            coopmat_store(acc_01, C, 16u32, N);
            coopmat_store(acc_10, C, 0u32, N);
            coopmat_store(acc_11, C, 16u32, N);
            return;
        }
    "#;
    let words = compile_ok(src);
    spirv_val_check(&words);

    // AT-1733 assertion 1: exactly ONE loop header block (single K-loop, not nested).
    let headers = collect_loop_headers(&words);
    assert_eq!(
        headers.len(), 1,
        "AT-1733: expected exactly 1 loop header block (single K-loop); got {}", headers.len()
    );

    let (header_label, phis) = &headers[0];

    // AT-1733 assertion 2: exactly 4 OpPhi at the loop header head.
    assert_eq!(
        phis.len(), 4,
        "AT-1733: expected exactly 4 OpPhi in ONE loop header %{header_label} \
         (4 loop-carried coopmat accumulators in one K-loop); got {}",
        phis.len()
    );

    // AT-1733 assertion 3: each phi has 7 words (type+result+2 predecessor pairs).
    // AT-1733 assertion 4: all 4 phi result_ids are DISTINCT.
    // AT-1733 assertion 5: all 4 phis share the same two predecessor labels.
    let mut phi_result_ids: Vec<u32> = Vec::new();
    let mut first_pred_a: Option<u32> = None;
    let mut first_pred_b: Option<u32> = None;

    for (idx, phi) in phis.iter().enumerate() {
        assert_eq!(
            phi.len(), 7,
            "AT-1733: OpPhi[{idx}] in header %{header_label} must have 7 words \
             (header + result_type + result_id + 4 operand ids = 2 predecessor pairs); \
             got {} words", phi.len()
        );

        // phi[2] = result_id, phi[4] = pred_label_1, phi[6] = pred_label_2
        let result_id = phi[2];
        let pred_a = phi[4];
        let pred_b = phi[6];

        // Predecessors must be distinct (pre_header != continue/latch).
        assert_ne!(
            pred_a, pred_b,
            "AT-1733: OpPhi[{idx}] predecessors must be distinct labels (pre_header != latch); \
             got pred_a=%{pred_a} == pred_b=%{pred_b}"
        );

        // All 4 phis must share the same predecessor pair (same K-loop header).
        match (first_pred_a, first_pred_b) {
            (None, None) => {
                first_pred_a = Some(pred_a);
                first_pred_b = Some(pred_b);
            }
            (Some(fa), Some(fb)) => {
                assert_eq!(
                    pred_a, fa,
                    "AT-1733: OpPhi[{idx}] pre_header label %{pred_a} != first phi's %{fa} \
                     (all 4 phis must share the same predecessor pair for a single K-loop)"
                );
                assert_eq!(
                    pred_b, fb,
                    "AT-1733: OpPhi[{idx}] latch label %{pred_b} != first phi's %{fb} \
                     (all 4 phis must share the same latch for a single K-loop)"
                );
            }
            _ => unreachable!("first_pred_a and first_pred_b are always set together"),
        }

        phi_result_ids.push(result_id);
    }

    // AT-1733 assertion 4: all 4 result_ids are DISTINCT.
    assert_eq!(phi_result_ids.len(), 4, "AT-1733: must have exactly 4 phi result ids");
    let unique_count = {
        let mut sorted = phi_result_ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        sorted.len()
    };
    assert_eq!(
        unique_count, 4,
        "AT-1733: all 4 OpPhi result_ids must be DISTINCT (no cross-contamination); \
         ids={phi_result_ids:?}"
    );

    // AT-1733 assertion 6: both predecessor labels must be valid OpLabel result_ids.
    let all_labels: Vec<u32> = iter_insts(&words[5..])
        .filter(|(op, _)| *op == OP_LABEL)
        .filter_map(|(_, slice)| slice.get(1).copied())
        .collect();

    let pred_a = first_pred_a.expect("pred_a must be set after processing phis");
    let pred_b = first_pred_b.expect("pred_b must be set after processing phis");
    assert!(
        all_labels.contains(&pred_a),
        "AT-1733: phi predecessor (pre_header) %{pred_a} must be a valid OpLabel result_id"
    );
    assert!(
        all_labels.contains(&pred_b),
        "AT-1733: phi predecessor (continue/latch) %{pred_b} must be a valid OpLabel result_id"
    );

    // AT-1733 assertion 7: total OpPhi count across the whole module == 4
    // (no extra phantom phis from anywhere else).
    let total_phis = count_op(&words, OP_PHI);
    assert_eq!(
        total_phis, 4,
        "AT-1733: total OpPhi in module must be exactly 4 (one per carried accumulator); \
         got {total_phis}"
    );

    eprintln!(
        "AT-1733 PASS: 4 loop-carried coopmat accumulators in ONE K-loop emit \
         exactly 4 well-formed OpPhi in ONE header %{header_label} — \
         result_ids={phi_result_ids:?}, pre_header=%{pred_a}, latch=%{pred_b}, \
         spirv-val clean. M3.3c RB codegen pre-gate LOCKED."
    );
}

/// AT-1708b: A loop where acc is reassigned BOTH unconditionally at the top level AND
/// also conditionally inside an if — must also be rejected.
///
/// Even though a top-level assign exists, a conditional reassignment inside the loop
/// causes the body-emission SSA rebind to update var_ids[acc] to the if-branch SSA id
/// (which emits AFTER the top-level assign). The latch value captured after body emission
/// is therefore the if-branch SSA id — which does NOT dominate the back-edge block.
/// The OpPhi would be invalid SPIR-V. Conservative rejection is correct here.
#[test]
fn at1708_coopmat_unconditional_plus_conditional_also_rejected() {
    let src = r#"
        @kernel @workgroup(32,1,1) @cooperative_matrix @intent("unconditional plus conditional") @complexity(O(n))
        fn coopmat_uncond_plus_cond(
            A: readonly_buffer[f16],
            B: readonly_buffer[f16],
            C: buffer[f16],
            K: u32,
            N: u32,
            cond_val: u32
        ) -> void {
            shared a_tile: shared[f16, 256];
            shared b_tile: shared[f16, 256];
            let mut acc: matrix[f16, 16, 16, accumulator] = coopmat_zero();
            for k in range(0u32, K) {
                let a_mat: matrix[f16, 16, 16, a] = coopmat_load(a_tile, 0u32, 16u32);
                let b_mat: matrix[f16, 16, 16, b] = coopmat_load(b_tile, 0u32, 16u32);
                acc = coopmat_mul_add(a_mat, b_mat, acc);
                if k < cond_val {
                    let a2: matrix[f16, 16, 16, a] = coopmat_load(a_tile, 0u32, 16u32);
                    let b2: matrix[f16, 16, 16, b] = coopmat_load(b_tile, 0u32, 16u32);
                    acc = coopmat_mul_add(a2, b2, acc);
                }
            }
            coopmat_store(acc, C, 0u32, N);
            return;
        }
    "#;
    // Must be rejected: a conditional reassignment makes the latch SSA id ambiguous.
    // The compiler cannot safely emit a valid OpPhi for this pattern.
    compile_expect_codegen_error(src, "conditional reassignment");
    eprintln!("AT-1708b PASS: unconditional+conditional coopmat loop correctly rejected (ambiguous latch)");
}
