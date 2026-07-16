//! M4.2a — STATIC ABI-conformance golden suite (AT-2988, AT-2995, AT-3000, AT-3001, AT-3003,
//! AT-3005): the anti-circularity anchor called out by the r1 pessimistic REJECT.
//!
//! AT-1803-class bit-identity / combined tests are AXIOM-vs-AXIOM: both the leader and the
//! ggml variant run through AXIOM's own runtime/f16-buffers/grid, so neither one alone can see a
//! genuine mismatch against ggml's REAL ABI (that was exactly how r1's CRITICAL #1/#2 slipped
//! past its own bit-identity-only anchor). These goldens instead encode the ggml facts as
//! LITERAL, CITED tables in-repo and FAIL if the kernel / oracle / fixture drift from them.
//!
//! Every fact below is re-verified in THIS coder pass against `vendor/llama.cpp` @ pinned SHA
//! `6b80c74f285390368b3c99c5e750f19e9b096e98` (tag b9542) — file:line citations point at that
//! SHA. These tests do NOT read the vendor tree (CI-runnable without it, per the milestone's
//! CI-runnable list); they assert the literal tables against the COMPILED ggml-variant kernel
//! and its source text.

use std::collections::BTreeMap;
use axc_driver::compile_source_with_assignments;

const GGML_SRC: &str =
    include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached_ggml.axc");
const LEADER_SRC: &str = include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached.axc");

fn rb2x2_assignments() -> BTreeMap<String, i64> {
    let mut m = BTreeMap::new();
    m.insert("rb_m".to_owned(), 2_i64);
    m.insert("rb_n".to_owned(), 2_i64);
    m.insert("tile_k".to_owned(), 16_i64);
    m.insert("a_block_size".to_owned(), 512_i64);
    m.insert("b_block_size".to_owned(), 512_i64);
    m
}

// ── AT-2988: STATIC golden — n_blocks_per_row = stride_a/256 (the /256-not-/144 correction) ──

/// AT-2988. Cites `ggml-vulkan.cpp:8403-8409` (`ggml_vk_matmul(ne01,ne11,ne10,ne10,ne10,
/// stride_d,…)` — `stride_a = ne10 = K` in ELEMENTS, so `n_blocks_per_row = stride_a/256`, NOT
/// `/144`). Asserts (a) the literal derived value equals `K/256` for K in {256,512,14336} (the
/// contiguous case, `stride_a == K`), and (b) the kernel SOURCE contains the literal `stride_a /
/// 256u32` division (catches a drift back to `/144` or to reading a push field directly).
#[test]
fn at_2988_n_blocks_per_row_derivation_is_stride_a_div_256() {
    // (a) literal derivation table, contiguous case (stride_a == K).
    for &k in &[256u32, 512, 14336] {
        let stride_a = k; // contiguous case
        let n_blocks_per_row = stride_a / 256u32;
        assert_eq!(
            n_blocks_per_row, k / 256,
            "AT-2988: n_blocks_per_row derivation must equal K/256 at K={k} \
             (ggml-vulkan.cpp:8403-8409: stride_a = ne10 = K, in ELEMENTS)"
        );
    }
    // (b) source-text anchor: the kernel must derive it via `stride_a / 256u32`, NOT `/144u32`.
    assert!(
        GGML_SRC.contains("stride_a / 256u32"),
        "AT-2988: q4km_matmul_rb_coopmat_f32acc_cached_ggml.axc must derive \
         n_blocks_per_row via `stride_a / 256u32` (ggml-vulkan.cpp:8403-8409) — got source without \
         that literal division; a drift to /144 or to a push field would silently break ABI \
         compatibility"
    );
    assert!(
        !GGML_SRC.contains("/ 144u32"),
        "AT-2988: the ggml variant must NOT divide stride_a by 144 (that was the r1-class bug: \
         144 is the Q4_K_M SUPERBLOCK BYTE size, not the /256 ELEMENT-count divisor)"
    );
    eprintln!("AT-2988 PASS: n_blocks_per_row = stride_a/256 (source + literal table, K in {{256,512,14336}})");
}

// ── AT-2995: STATIC golden — the 17-field push offset table + bindings 0/1/2 ─────────────────

/// One row of ggml's `vk_mat_mat_push_constants` (`ggml-vulkan.cpp:1036-1044`), 17 x u32, 68 B.
struct PushField {
    name: &'static str,
    offset: u32,
}

/// The FULL 17-field literal table, cited from `ggml-vulkan.cpp:1036-1044` at the pinned SHA:
/// ```c
/// struct vk_mat_mat_push_constants {
///     uint32_t M; uint32_t N; uint32_t K;
///     uint32_t stride_a; uint32_t stride_b; uint32_t stride_d;
///     uint32_t batch_stride_a; uint32_t batch_stride_b; uint32_t batch_stride_d;
///     uint32_t base_work_group_z; uint32_t num_batches;
///     uint32_t k_split;
///     uint32_t ne02; uint32_t ne12; uint32_t broadcast2; uint32_t broadcast3;
///     uint32_t padded_N;
/// };
/// ```
/// 17 fields x 4 B = 68 B. AXIOM reads only the LEADING SIX (offsets 0..24); the AXIOM shader's
/// push block (24 B) is legally smaller than the host's pipeline-layout range (68 B).
const GGML_PUSH_TABLE: &[PushField] = &[
    PushField { name: "M", offset: 0 },
    PushField { name: "N", offset: 4 },
    PushField { name: "K", offset: 8 },
    PushField { name: "stride_a", offset: 12 },
    PushField { name: "stride_b", offset: 16 },
    PushField { name: "stride_d", offset: 20 },
    PushField { name: "batch_stride_a", offset: 24 },
    PushField { name: "batch_stride_b", offset: 28 },
    PushField { name: "batch_stride_d", offset: 32 },
    PushField { name: "base_work_group_z", offset: 36 },
    PushField { name: "num_batches", offset: 40 },
    PushField { name: "k_split", offset: 44 },
    PushField { name: "ne02", offset: 48 },
    PushField { name: "ne12", offset: 52 },
    PushField { name: "broadcast2", offset: 56 },
    PushField { name: "broadcast3", offset: 60 },
    PushField { name: "padded_N", offset: 64 },
];

/// AT-2995. Asserts: (1) the literal 17-field table above is internally self-consistent (17
/// fields, 68 B, sequential 4-byte u32 offsets — the r2 fix that added `ne02@48` and corrected
/// the mislabeled tail); (2) the COMPILED ggml-variant kernel's push-constant scalar list matches
/// the table's LEADING SIX names+offsets exactly (M,N,K,stride_a,stride_b,stride_d @ 0,4,8,12,
/// 16,20); (3) buffer bindings q=0, x=1, C=2 (`compute_binding_plan`, declaration order —
/// `ggml-vulkan.cpp:7863`'s descriptor set is `{a,b,d}`).
#[test]
fn at_2995_seventeen_field_push_table_and_bindings() {
    // (1) table self-consistency.
    assert_eq!(GGML_PUSH_TABLE.len(), 17, "AT-2995: ggml's vk_mat_mat_push_constants has 17 fields");
    for (i, f) in GGML_PUSH_TABLE.iter().enumerate() {
        assert_eq!(
            f.offset, (i as u32) * 4,
            "AT-2995: field {i} ({}) must be at offset {} (sequential 4-byte u32 std430 layout)",
            f.name, i * 4
        );
    }
    let total_bytes = GGML_PUSH_TABLE.last().unwrap().offset + 4;
    assert_eq!(total_bytes, 68, "AT-2995: 17 fields x 4 B = 68 B (ggml-vulkan.cpp:1036-1044)");
    assert_eq!(GGML_PUSH_TABLE[12].name, "ne02", "AT-2995: field 12 (offset 48) must be ne02 (r2 fix)");

    // (2) the compiled kernel's leading-six scalars match the table exactly.
    let assignments = rb2x2_assignments();
    let (_bytes, meta) = compile_source_with_assignments(GGML_SRC, &assignments)
        .expect("q4km_matmul_rb_coopmat_f32acc_cached_ggml.axc must compile");
    let leading_six = &GGML_PUSH_TABLE[0..6];
    assert_eq!(
        meta.binding_plan.scalars.len(), 6,
        "AT-2995: the ggml variant must declare EXACTLY six scalar push params \
         (M,N,K,stride_a,stride_b,stride_d) — got {:?}",
        meta.binding_plan.scalars.iter().map(|s| &s.name).collect::<Vec<_>>()
    );
    for (slot, expected) in meta.binding_plan.scalars.iter().zip(leading_six.iter()) {
        assert_eq!(
            slot.name, expected.name,
            "AT-2995: scalar param order must match ggml's leading six"
        );
        assert_eq!(
            slot.offset, expected.offset,
            "AT-2995: `{}` must be at push-constant offset {} (byte-identical to ggml's \
             leading six, ggml-vulkan.cpp:1036-1044)", expected.name, expected.offset
        );
    }

    // (3) buffer bindings q=0, x=1, C=2 (declaration order == ggml's {a,b,d} descriptor set).
    assert_eq!(meta.binding_plan.buffers.len(), 3, "AT-2995: exactly 3 buffer bindings (q,x,C)");
    let expected_buffers = [("q", 0u32), ("x", 1u32), ("C", 2u32)];
    for (slot, (name, binding)) in meta.binding_plan.buffers.iter().zip(expected_buffers.iter()) {
        assert_eq!(&slot.name, name, "AT-2995: buffer param name/order must be q,x,C");
        assert_eq!(
            slot.buffer_position, *binding,
            "AT-2995: `{name}` must bind at slot {binding} (ggml A=0/B=1/D=2, ggml-vulkan.cpp:7863)"
        );
    }

    eprintln!(
        "AT-2995 PASS: 17-field push table self-consistent (68 B, ne02@48); compiled kernel's \
         leading six scalars + q/x/C=0/1/2 bindings match ggml byte-for-byte"
    );
}

// ── AT-3000: STATIC golden — B binding element size == 4 B (f32), the B_TYPE=float catch ────

/// AT-3000 (catches CRITICAL #1). Cites `vulkan-shaders-gen.cpp:582` (`{"B_TYPE","float"}` —
/// `matmul_q4_k_f32` names ggml's B ACTIVATION type as f32, not the A weight type, which is
/// always `DATA_A_Q4_K`). Asserts the ggml variant's B (`x`) buffer element size is 4 B (F32),
/// `!= 2` (F16) — the DIRECT static detector of r1's B-type bug (reading ggml's f32 src1 buffer
/// through an `x: readonly_buffer[f16]` reinterprets every pair of f32 bytes as garbage f16).
#[test]
fn at_3000_b_buffer_element_size_is_f32_matching_ggml_b_type() {
    use axc_hir::ty::ScalarTy;

    let assignments = rb2x2_assignments();
    let (_bytes, ggml_meta) = compile_source_with_assignments(GGML_SRC, &assignments)
        .expect("q4km_matmul_rb_coopmat_f32acc_cached_ggml.axc must compile");
    let (_bytes2, leader_meta) = compile_source_with_assignments(LEADER_SRC, &assignments)
        .expect("q4km_matmul_rb_coopmat_f32acc_cached.axc must compile");

    // x is buffer_position 1 in both kernels (q=0, x=1, C=2).
    let ggml_x = ggml_meta.binding_plan.buffers.iter().find(|b| b.name == "x")
        .expect("AT-3000: ggml variant must declare buffer param `x`");
    let leader_x = leader_meta.binding_plan.buffers.iter().find(|b| b.name == "x")
        .expect("AT-3000: leader must declare buffer param `x`");

    assert_eq!(
        ggml_x.ty.elem, ScalarTy::F32,
        "AT-3000: the ggml variant's B (`x`) buffer element type must be F32 \
         (vulkan-shaders-gen.cpp:582 `matmul_q4_k_f32` instantiates B_TYPE=float) — got {:?}",
        ggml_x.ty.elem
    );
    assert_eq!(
        ggml_x.ty.elem_byte_size(), 4,
        "AT-3000: the ggml variant's B buffer element size must be 4 B (f32) == ggml's \
         matmul_q4_k_f32 B_TYPE — got {} B. A 2 B (f16) B buffer would reinterpret ggml's f32 \
         src1 as garbage f16 pairs (the r1 CRITICAL #1 bug).", ggml_x.ty.elem_byte_size()
    );
    assert_ne!(
        ggml_x.ty.elem_byte_size(), 2,
        "AT-3000: the ggml variant's B buffer must NOT be 2 B (f16) — that is the leader's \
         (non-ggml-ABI) B type, got {:?}", leader_x.ty.elem
    );
    // Regression guard: the LEADER's B stays f16 (unchanged — only the ggml variant retypes B).
    assert_eq!(
        leader_x.ty.elem, ScalarTy::F16,
        "AT-3000: the M3.6 LEADER's B buffer must remain F16 (unchanged) — this golden must not \
         mask a mistaken edit to the leader kernel"
    );

    eprintln!(
        "AT-3000 PASS: ggml variant's B (x) buffer elem_byte_size=4 (F32, matches ggml \
         matmul_q4_k_f32 B_TYPE=float); leader's B stays F16 (2 B, unchanged)"
    );
}

// ── AT-3001: STATIC golden — grid axis mapping gid(0)->M-row / gid(1)->N-col ──────────────────

/// AT-3001 (catches CRITICAL #2). Cites `mul_mm.comp:141` (`const uint ic = gl_WorkGroupID.y;`
/// — y selects the N-column block) and `mul_mm.comp:166` (`const uint ir = gl_WorkGroupID.x %
/// blocks_m;` — x selects the M-row block). Asserts the ggml variant's SOURCE derives
/// `block_row` from `gid(0u32)` and `block_col` from `gid(1u32)` — the OPPOSITE of the M3.6
/// leader (which the golden also pins, as a regression guard against accidentally "fixing" the
/// leader instead of adding the variant).
#[test]
fn at_3001_grid_axis_mapping_matches_ggml_convention() {
    // ggml variant: x -> block_row (M), y -> block_col (N).
    assert!(
        GGML_SRC.contains("let block_row: u32 = gid(0u32) / 32u32;"),
        "AT-3001: the ggml variant must derive block_row from gid(0u32) (mul_mm.comp:166: \
         `ir = gl_WorkGroupID.x % blocks_m` selects the M-row block on the x axis)"
    );
    assert!(
        GGML_SRC.contains("let block_col: u32 = gid(1u32);"),
        "AT-3001: the ggml variant must derive block_col from gid(1u32) (mul_mm.comp:141: \
         `ic = gl_WorkGroupID.y` selects the N-column block on the y axis)"
    );

    // M3.6 leader: the OPPOSITE mapping (x -> block_col/N, y -> block_row/M) — regression guard.
    assert!(
        LEADER_SRC.contains("let block_col: u32 = gid(0u32) / 32u32;"),
        "AT-3001: the M3.6 LEADER must keep its (opposite) mapping x->block_col — this golden \
         guards against accidentally editing the leader instead of adding the ggml variant"
    );
    assert!(
        LEADER_SRC.contains("let block_row: u32 = gid(1u32);"),
        "AT-3001: the M3.6 LEADER must keep block_row from gid(1u32) (unchanged)"
    );

    eprintln!(
        "AT-3001 PASS: ggml variant maps gid(0)->block_row(M)/gid(1)->block_col(N) \
         (mul_mm.comp:141,166); leader keeps the opposite mapping (regression guard)"
    );
}

// ── AT-3003: STATIC golden — D column-major layout + stride_d semantics ──────────────────────

/// AT-3003. Cites `mul_mm.comp:404` (`coopMatStore(cm_dtype, data_d, offsets + (dc + cm_col *
/// TN) * p.stride_d + dr + cm_row * TM, p.stride_d, gl_CooperativeMatrixLayoutColumnMajor)` —
/// address = col*stride_d + row) and `ggml-vulkan.cpp:8175` (`stride_d = dst->nb[1]/type_size`,
/// M elements for a contiguous non-batched D). Asserts the ggml variant's SOURCE (a) uses
/// `coopmat_store_col` (not `coopmat_store`) for all four D writes, and (b) the base-offset
/// formula follows the `col*stride_d + row` pattern (tile_col first, multiplied by stride_d,
/// then tile_row added) for all four 16x16 output tiles.
#[test]
fn at_3003_d_column_major_store_and_stride_d_formula() {
    // (a) all four D stores use coopmat_store_col, NONE use plain coopmat_store.
    let store_col_count = GGML_SRC.matches("coopmat_store_col(").count();
    assert_eq!(
        store_col_count, 4,
        "AT-3003: the ggml variant must store all 4 accumulator tiles via coopmat_store_col \
         (mul_mm.comp:404: ColumnMajor) — found {store_col_count} calls"
    );
    // A bare `coopmat_store(` (not `coopmat_store_col(`) must NOT appear.
    let bare_store_count = GGML_SRC.matches("coopmat_store(").count();
    assert_eq!(
        bare_store_count, 0,
        "AT-3003: the ggml variant must NOT use row-major coopmat_store anywhere — found \
         {bare_store_count} occurrence(s) (ggml's D is column-major)"
    );

    // (b) base-offset formula: `(tile_col_X * 16u32) * stride_d + (tile_row_Y * 16u32)`.
    for (col, row) in [("tile_col_0", "tile_row_0"), ("tile_col_1", "tile_row_0"),
                       ("tile_col_0", "tile_row_1"), ("tile_col_1", "tile_row_1")] {
        let needle = format!("({col} * 16u32) * stride_d + ({row} * 16u32)");
        assert!(
            GGML_SRC.contains(&needle),
            "AT-3003: expected the column-major base-offset formula `{needle}` \
             (ggml-vulkan.cpp:8175 stride_d=dst->nb[1]/type_size=M; mul_mm.comp:404: \
             address = col*stride_d + row) — not found in the ggml variant source"
        );
    }

    eprintln!(
        "AT-3003 PASS: 4/4 D stores use coopmat_store_col; base-offset formula matches \
         col*stride_d+row (mul_mm.comp:404, ggml-vulkan.cpp:8175)"
    );
}

// ── AT-3005: STATIC golden — 144 B block_q4_K + six-variant aligned family + interception ────
//            point + the 13-condition guard table (recorded facts, cited) ────────────────────

/// AT-3005. Records (as literal, cited facts — re-verified against the pinned SHA in this coder
/// pass):
/// - `block_q4_K` = 144 bytes (`ggml-common.h`: `QK_K=256`; d@0/dmin@2/scales[12]@4/qs[128]@16).
/// - the SIX-variant aligned family: `matmul_q4_k_f32_cm1_data[]` (-> l/m/s) +
///   `matmul_q4_k_f32_aligned_cm1_data[]` (-> a_l/a_m/a_s), built by CREATE_MM
///   (`ggml-vulkan.cpp:4092-4104`) inside the coopmat block `[4090,4211]`; Q4_K f32acc
///   registration at `:4135` (CREATE_MM2, gated by `coopmat_acc_f16_support`/`coopmat_acc_f32_support`
///   at `:4108`/`:4111`) or `:4159` (its else) — BOTH inside `[4090,4211]`. `:4429` is the
///   `if (device->fp16)` FALLBACK TIER (`:4212`+), NOT coopmat.
/// - suffix rule `_cm1` (coopmat-KHR) from `vulkan-shaders-gen.cpp:411`
///   (`(coopmat ? "_cm1" : "")`); `_cm2` is the separate NV coopmat2 path (`:4010` block).
/// - `aligned` computed at `:8237`; `ggml_vk_guess_matmul_pipeline` at `:7796` (tail
///   `:7822-7839`) returns `a_l/a_m/a_s` when aligned.
/// - the INTERCEPTION POINT: `:8239`, immediately after the
///   `pipeline = ggml_vk_guess_matmul_pipeline(...)` call — override the RESULT (r3 mechanism;
///   ONE point, all six variants, alignment-independent).
/// - the 64b-indexing swap at `:8241-8242` (guard condition #13).
/// - the 13-condition selection guard (§2.9 of the milestone spec).
// The line-number relationships below are asserted as REGRESSION GUARDS over literal, cited
// constants (a human editing one citation should still trip the check) — clippy's
// assertions-on-constants lint is a false positive for this intentional pattern.
#[allow(clippy::assertions_on_constants)]
#[test]
fn at_3005_block_q4k_size_six_variant_family_interception_and_guard() {
    // block_q4_K = 144 B (d f16 @0, dmin f16 @2, 12 B scales @4, 128 B qs @16).
    const Q4_K_SUPERBLOCK_BYTES: usize = 144;
    assert_eq!(2 + 2 + 12 + 128, Q4_K_SUPERBLOCK_BYTES, "AT-3005: block_q4_K byte layout sums to 144");
    // Cross-check against the shared oracle constant (single source of truth).
    assert_eq!(
        axc_driver::q4km_oracle::Q4KM_SUPERBLOCK_BYTES, Q4_K_SUPERBLOCK_BYTES,
        "AT-3005: the oracle's Q4KM_SUPERBLOCK_BYTES must equal the golden's 144"
    );
    assert_eq!(axc_driver::q4km_oracle::Q4KM_SUPERBLOCK_ELEMS, 256, "AT-3005: QK_K=256 (ggml-common.h)");

    // The six-variant aligned family (byte-array symbol names, l/m/s x unaligned/aligned).
    const CM1_DATA_SYMBOL: &str = "matmul_q4_k_f32_cm1_data";
    const CM1_ALIGNED_DATA_SYMBOL: &str = "matmul_q4_k_f32_aligned_cm1_data";
    const SIX_VARIANT_SLOTS: &[&str] = &["l", "m", "s", "a_l", "a_m", "a_s"];
    assert_eq!(SIX_VARIANT_SLOTS.len(), 6, "AT-3005: CREATE_MM builds SIX slots (ggml-vulkan.cpp:4092-4104)");
    assert!(CM1_DATA_SYMBOL.ends_with("_cm1_data"), "AT-3005: unaligned symbol ends _cm1_data");
    assert!(CM1_ALIGNED_DATA_SYMBOL.ends_with("_aligned_cm1_data"), "AT-3005: aligned symbol ends _aligned_cm1_data");

    // The target coopmat-KHR symbol our upstream package injects against.
    const TARGET_SYMBOL: &str = "matmul_q4_k_f32_cm1";
    assert_eq!(
        format!("{TARGET_SYMBOL}_data"), CM1_DATA_SYMBOL,
        "AT-3005: the patch/fragment target symbol must be matmul_q4_k_f32_cm1 (the KHR coopmat \
         path; vulkan-shaders-gen.cpp:411 `(coopmat ? \"_cm1\" : \"\")`), NOT `_cm2` \
         (coopmat2, :4010 block) or the scalar no-coopmat `matmul_q4_k_f32_data` (:4260)"
    );

    // The coopmat block's line range + the two Q4_K f32acc registration sites (both inside it).
    const COOPMAT_BLOCK_START_LINE: u32 = 4090;
    const COOPMAT_BLOCK_END_LINE: u32 = 4211;
    const Q4K_F32ACC_REG_SITE_A: u32 = 4135; // CREATE_MM2, gated coopmat_acc_f16/f32_support
    const Q4K_F32ACC_REG_SITE_B: u32 = 4159; // its else
    const FP16_FALLBACK_TIER_LINE: u32 = 4429; // NOT coopmat — if(device->fp16) tier, :4212+
    assert!(
        (COOPMAT_BLOCK_START_LINE..=COOPMAT_BLOCK_END_LINE).contains(&Q4K_F32ACC_REG_SITE_A),
        "AT-3005: the :4135 Q4_K f32acc registration must be inside the coopmat block [4090,4211]"
    );
    assert!(
        (COOPMAT_BLOCK_START_LINE..=COOPMAT_BLOCK_END_LINE).contains(&Q4K_F32ACC_REG_SITE_B),
        "AT-3005: the :4159 Q4_K f32acc registration must be inside the coopmat block [4090,4211]"
    );
    assert!(
        FP16_FALLBACK_TIER_LINE > COOPMAT_BLOCK_END_LINE,
        "AT-3005: :4429 (the if(device->fp16) fallback tier) must be OUTSIDE [4090,4211] — r2 \
         wrongly cited it as a coopmat registration; it is a separate device-class tier"
    );

    // The interception point (r3 mechanism): override the RESULT of
    // ggml_vk_guess_matmul_pipeline, right after its call site.
    const GUESS_PIPELINE_CALL_SITE_LINE: u32 = 8239;
    const ALIGNED_CONDITION_LINE: u32 = 8237;
    const SIXTY_FOUR_BIT_INDEXING_SWAP_LINE: u32 = 8241;
    assert!(
        GUESS_PIPELINE_CALL_SITE_LINE > ALIGNED_CONDITION_LINE,
        "AT-3005: the interception point (:8239) must be AFTER the `aligned` computation (:8237)"
    );
    assert!(
        SIXTY_FOUR_BIT_INDEXING_SWAP_LINE > GUESS_PIPELINE_CALL_SITE_LINE,
        "AT-3005: the 64b-indexing swap (:8241-8242, guard condition #13) sits AFTER the \
         interception point (:8239) — the override happens BEFORE that swap"
    );

    // The 13-condition selection guard (§2.9) — recorded as a literal, counted table.
    const GUARD_CONDITIONS: &[&str] = &[
        "src0->type == GGML_TYPE_Q4_K",
        "src1->type == GGML_TYPE_F32 (y_f32_kernel)",
        "!quantize_y",
        "!x_non_contig",
        "!y_non_contig",
        "!qx_needs_dequant",
        "!qy_needs_dequant",
        "split_k==1 && (ne12*ne13)==1 && broadcast2==1 && broadcast3==1",
        "ne01%32==0 && ne11%32==0 && ne10%256==0",
        "device->coopmat_support && !device->coopmat2",
        "device->coopmat_acc_f32_support",
        "(ggml_prec)dst->op_params[0] == GGML_PREC_F32",
        "ggml_nbytes(src0) <= maxStorageBufferRange",
    ];
    assert_eq!(
        GUARD_CONDITIONS.len(), 13,
        "AT-3005: the host-adapter selection guard must enumerate EXACTLY 13 conditions (§2.9)"
    );
    // Condition #11 cites :4111 (coopmat_acc_f32_support, the CREATE_MM2 f32acc gate) — NOT
    // :3299 (coopmat_support_16x16x16_f32acc, FA_COOPMAT1 flash-attention tuning — r2's error,
    // corrected in r3).
    assert!(
        GUARD_CONDITIONS[10].contains("coopmat_acc_f32_support"),
        "AT-3005: condition #11 must be device->coopmat_acc_f32_support (:4111), NOT \
         coopmat_support_16x16x16_f32acc (:3299, FA_COOPMAT1 — unrelated flash-attention tuning)"
    );
    // Condition #13 (the 64b-indexing swap, :8241-8242) is documented UNREACHABLE at the target
    // shapes (Q4_K A weight at M=4096,K=14336 ~= 33 MB << the >=2 GB maxStorageBufferRange) but
    // included for fail-closed safety.
    assert!(
        GUARD_CONDITIONS[12].contains("maxStorageBufferRange"),
        "AT-3005: condition #13 (fail-closed, unreachable at target shapes) must cite \
         maxStorageBufferRange (:8241-8242)"
    );

    eprintln!(
        "AT-3005 PASS: block_q4_K=144B; six-variant aligned family recorded \
         ({CM1_DATA_SYMBOL}=l/m/s, {CM1_ALIGNED_DATA_SYMBOL}=a_l/a_m/a_s); coopmat block \
         [{COOPMAT_BLOCK_START_LINE},{COOPMAT_BLOCK_END_LINE}] (fp16 fallback tier at \
         {FP16_FALLBACK_TIER_LINE} correctly EXCLUDED); interception at \
         :{GUESS_PIPELINE_CALL_SITE_LINE}; 13-condition guard recorded"
    );
}
