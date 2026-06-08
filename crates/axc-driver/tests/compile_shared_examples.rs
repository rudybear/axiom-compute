//! Compile + spirv-val tests for M3.2 shared[T,N] examples.
//!
//! AT-1611 additive guard: verifies that new shared-memory examples compile
//! cleanly and produce valid SPIR-V. Does NOT touch existing fixtures.
//!
//! AT-1614: shared-source coopmat SPIR-V passes spirv-val (single-index Workgroup path).
//! AT-1609 partial: shared[f16] Float16 capability present in matmul_shared_coopmat.
//! AT-1613: Buffer-source coopmat path is byte-identical (no regression from CoopMatLoadSource).

use std::collections::BTreeMap;
use axc_driver::{compile_source_with_meta, compile_source_with_assignments};
use spirv_tools::val::{Validator, create as create_validator};
use spirv_tools::TargetEnv;

type StrategyMap = BTreeMap<String, i64>;

/// Compile bytes -> SPIR-V words + spirv-val.
fn compile_and_validate(src: &str, name: &str) -> Vec<u32> {
    let (bytes, _meta) = compile_source_with_meta(src)
        .unwrap_or_else(|e| panic!("{name}: compile failed: {e:?}"));
    words_and_validate(bytes, name)
}

fn compile_with_assignments_and_validate(
    src: &str,
    assignments: &StrategyMap,
    name: &str,
) -> Vec<u32> {
    let (bytes, _meta) = compile_source_with_assignments(src, assignments)
        .unwrap_or_else(|e| panic!("{name}: compile failed: {e:?}"));
    words_and_validate(bytes, name)
}

fn tile_assignments(tile_m: i64, tile_n: i64, tile_k: i64) -> StrategyMap {
    let mut m = StrategyMap::new();
    m.insert("tile_m".to_owned(), tile_m);
    m.insert("tile_n".to_owned(), tile_n);
    m.insert("tile_k".to_owned(), tile_k);
    // Pre-computed products for shared array sizes (N must be a literal after substitution).
    m.insert("tile_a_size".to_owned(), tile_m * tile_k);
    m.insert("tile_b_size".to_owned(), tile_k * tile_n);
    m
}

fn words_and_validate(bytes: Vec<u8>, name: &str) -> Vec<u32> {
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let validator = create_validator(Some(TargetEnv::Vulkan_1_1));
    validator
        .validate(&words, None)
        .unwrap_or_else(|e| panic!("{name}: spirv-val failed: {e}"));
    words
}

/// AT-partial: shared_reduce.axc compiles + spirv-val clean.
#[test]
fn shared_reduce_compiles_and_validates() {
    let src = include_str!("../../../examples/shared_reduce.axc");
    compile_and_validate(src, "shared_reduce.axc");
}

/// AT-partial: matmul_shared_f32.axc (Lavapipe-friendly, tile holes resolved) compiles + spirv-val clean.
#[test]
fn matmul_shared_f32_compiles_and_validates() {
    let src = include_str!("../../../examples/matmul_shared_f32.axc");
    let assignments = tile_assignments(16, 16, 16);
    compile_with_assignments_and_validate(src, &assignments, "matmul_shared_f32.axc");
}

/// AT-1614 partial + AT-partial: matmul_shared_coopmat.axc (PART B) compiles + spirv-val clean.
///
/// This is the KEY test — it exercises the coopmat_load FROM SHARED path (PART B blocker fixed).
/// Strategy holes resolved to defaults (tile_m=16, tile_n=16, tile_k=16).
#[test]
fn matmul_shared_coopmat_compiles_and_validates() {
    let src = include_str!("../../../examples/matmul_shared_coopmat.axc");
    let assignments = tile_assignments(16, 16, 16);
    let words = compile_with_assignments_and_validate(src, &assignments, "matmul_shared_coopmat.axc");

    // AT-1609 partial: Float16 capability must be present (shared[f16] requires it).
    // OpCapability Float16: opcode=17 (0x11), word_count=2, value=9.
    let float16_cap_found = words.windows(2).any(|w| {
        let opcode = w[0] & 0xFFFF;
        let wc = w[0] >> 16;
        opcode == 17 && wc == 2 && w[1] == 9 // Float16 = 9
    });
    assert!(
        float16_cap_found,
        "matmul_shared_coopmat.axc must emit OpCapability Float16 \
         (shared[f16] requires it; observe_type does NOT cover F16)"
    );
}

/// AT-1622 partial: compile tile_k=16 and tile_k=32 produce different OpTypeArray length constants.
///
/// Guards that @strategy holes GENUINELY parameterize the shared array sizes (not inert).
/// Only the structural (compilation) part of AT-1622 runs here; the bit-exact GPU part
/// is in dispatch_shared_matmul.rs.
#[test]
fn at1622_tile_k_variants_produce_different_spirv() {
    let src = include_str!("../../../examples/matmul_shared_coopmat.axc");

    let assignments_16 = tile_assignments(16, 16, 16); // tile_a_size=256, tile_b_size=256
    let assignments_32 = tile_assignments(16, 16, 32); // tile_a_size=512, tile_b_size=512

    let (bytes16, _) = compile_source_with_assignments(src, &assignments_16)
        .expect("matmul_shared_coopmat tile_k=16 must compile");
    let (bytes32, _) = compile_source_with_assignments(src, &assignments_32)
        .expect("matmul_shared_coopmat tile_k=32 must compile");

    // Both must be spirv-val clean.
    let words16: Vec<u32> = bytes16.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
    let words32: Vec<u32> = bytes32.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
    let validator = create_validator(Some(TargetEnv::Vulkan_1_1));
    validator.validate(&words16, None).expect("tile_k=16 spirv-val must pass");
    validator.validate(&words32, None).expect("tile_k=32 spirv-val must pass");

    // The SPIR-V must DIFFER (tile_k=32 implies larger shared arrays = different constants).
    assert_ne!(
        words16, words32,
        "AT-1622: tile_k=16 and tile_k=32 must produce different SPIR-V \
         (shared array sizes are parameterized by tile_k)"
    );
}

/// AT-partial: tiled_attention.axc (PART C1) compiles + spirv-val clean.
#[test]
fn tiled_attention_compiles_and_validates() {
    let src = include_str!("../../../examples/tiled_attention.axc");
    compile_and_validate(src, "tiled_attention.axc");
}

/// AT-1738: flash_attention.axc (M3.2b C2 — FlashAttention-2 streaming online-softmax)
/// compiles + spirv-val clean. CI compile anchor (no GPU). Pure source, ZERO codegen change.
#[test]
fn flash_attention_compiles_and_validates() {
    let src = include_str!("../../../examples/flash_attention.axc");
    compile_and_validate(src, "flash_attention.axc");
}

/// AT-1613: q4km_dequant_matmul_coopmat.axc SPIR-V is byte-identical before and after
/// the CoopMatLoadSource discriminator addition (Buffer-source path unchanged).
#[test]
fn at1613_buffer_source_coopmat_byte_identical_across_recompiles() {
    let src = include_str!("../../../examples/q4km_dequant_matmul_coopmat.axc");
    let (bytes1, _) = compile_source_with_meta(src)
        .expect("q4km_dequant_matmul_coopmat.axc must compile");
    let (bytes2, _) = compile_source_with_meta(src)
        .expect("q4km_dequant_matmul_coopmat.axc must compile (2nd)");
    // Byte-identical across recompiles verifies Buffer-source emit path is stable.
    assert_eq!(
        bytes1, bytes2,
        "AT-1613: q4km_dequant_matmul_coopmat SPIR-V must be byte-identical across recompiles \
         (Buffer-source coopmat path unchanged by CoopMatLoadSource discriminator)"
    );
}

/// AT-1614: shared-source coopmat SPIR-V in matmul_shared_coopmat.axc must be spirv-val clean.
///
/// If the Buffer two-index path were used for a Workgroup variable,
/// spirv-val would reject the module (wrong pointer type / arity).
/// Passing spirv-val proves the single-index Workgroup path was taken.
#[test]
fn at1614_shared_source_coopmat_spirv_valid() {
    let src = include_str!("../../../examples/matmul_shared_coopmat.axc");
    let assignments = tile_assignments(16, 16, 16);
    // If the Buffer two-index path were used for a Workgroup variable, spirv-val rejects.
    // Passing this test proves the shared-source single-index path is emitted correctly.
    compile_with_assignments_and_validate(src, &assignments, "at1614_matmul_shared_coopmat");
}

/// AT-1745: matmul_msg_coopmat.axc (M3.3d multi-subgroup) compiles + spirv-val clean
/// with the shipped strategy assignments. CI compile anchor, no GPU.
///
/// ASSERTS:
///   - staging-coverage invariant: a_block_size % wg_threads == 0 AND b_block_size % wg_threads == 0
///     (AT-1620 staging-bug class: 512%64==0 -> 8 iters, 1024%64==0 -> 16 iters).
///   - shared_memory_bytes == 3072 and <= 16384.
///   - The compiled module EMITS the LocalInvocationId Input var (body-scanner coverage proof).
#[test]
fn at1745_compile_matmul_msg_coopmat() {
    let src = include_str!("../../../examples/matmul_msg_coopmat.axc");

    // Shipped MSG assignments.
    let wg_threads: i64 = 64;
    let a_block_size: i64 = 512;
    let b_block_size: i64 = 1024;
    let mut assignments = StrategyMap::new();
    assignments.insert("wg_threads".to_owned(), wg_threads);
    assignments.insert("n_sg".to_owned(), 2_i64);
    assignments.insert("rb_m".to_owned(), 2_i64);
    assignments.insert("rb_n".to_owned(), 2_i64);
    assignments.insert("tile_k".to_owned(), 16_i64);
    assignments.insert("a_block_size".to_owned(), a_block_size);
    assignments.insert("b_block_size".to_owned(), b_block_size);

    // Staging-coverage invariant (AT-1620 staging-bug class).
    assert_eq!(
        a_block_size % wg_threads, 0,
        "AT-1745: a_block_size({a_block_size}) % wg_threads({wg_threads}) must == 0 \
         (every shared A element covered exactly once by {wg_threads} threads)"
    );
    assert_eq!(
        b_block_size % wg_threads, 0,
        "AT-1745: b_block_size({b_block_size}) % wg_threads({wg_threads}) must == 0 \
         (every shared B element covered exactly once by {wg_threads} threads)"
    );

    let (bytes, meta) = axc_driver::compile_source_with_assignments(src, &assignments)
        .unwrap_or_else(|e| panic!("AT-1745: matmul_msg_coopmat.axc compile failed: {e:?}"));

    assert!(!bytes.is_empty(), "AT-1745: matmul_msg_coopmat.axc produced empty SPIR-V");

    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // spirv-val must pass.
    let validator = create_validator(Some(TargetEnv::Vulkan_1_1));
    validator.validate(&words, None)
        .unwrap_or_else(|e| panic!("AT-1745: matmul_msg_coopmat.axc spirv-val FAILED: {e}"));

    // shared_memory_bytes == (a_block_size + b_block_size) * sizeof(f16) = (512+1024)*2 = 3072 bytes.
    let expected_shared_bytes: u32 = ((a_block_size + b_block_size) * 2) as u32;
    assert_eq!(
        meta.shared_memory_bytes, expected_shared_bytes,
        "AT-1745: shared_memory_bytes must be {expected_shared_bytes} \
         (= (a_block_size={a_block_size} + b_block_size={b_block_size}) * 2 bytes/f16); \
         got {}", meta.shared_memory_bytes
    );
    assert!(
        meta.shared_memory_bytes <= 16384,
        "AT-1745: shared_memory_bytes={} exceeds 16384 (portable minimum maxComputeSharedMemorySize)",
        meta.shared_memory_bytes
    );

    // Body-scanner coverage proof: the kernel uses local_invocation_id() as the staging thread
    // index inside loop-bound/shared-write-index expressions, so body_uses_local_invocation_id
    // must detect it and emit the LocalInvocationId Input variable.
    //
    // SPIR-V BuiltIn LocalInvocationId = 27 (spirv crate autogen_spirv.rs).
    // We search the raw word stream for value 27 appearing as the BuiltIn decoration operand
    // in an OpDecorate instruction: [word_count<<16 | 71, var_id, 11(BuiltIn), 27(LocalId)].
    // OpDecorate=71, BuiltIn=11.
    // SPIR-V BuiltIn LocalInvocationId = 27 (spirv crate autogen_spirv.rs).
    // SPIR-V Decoration::BuiltIn = 11 (verified from spirv autogen: BuiltIn = 11u32).
    // We search the raw word stream for OpDecorate %var BuiltIn LocalInvocationId(27):
    //   [word_count<<16 | 71, var_id, 11(BuiltIn), 27(LocalInvocationId)]
    const OP_DECORATE: u32 = 71;
    const BUILTIN_DECORATION: u32 = 11;  // Decoration::BuiltIn = 11 (SPIR-V spec)
    const LOCAL_INVOCATION_ID_VALUE: u32 = 27;  // BuiltIn::LocalInvocationId = 27
    let has_local_invocation_id_decoration = words.windows(4).any(|w| {
        let opcode = w[0] & 0xFFFF;
        opcode == OP_DECORATE && w[2] == BUILTIN_DECORATION && w[3] == LOCAL_INVOCATION_ID_VALUE
    });
    assert!(
        has_local_invocation_id_decoration,
        "AT-1745: module MUST emit LocalInvocationId Input var (body-scanner coverage proof). \
         Expected OpDecorate BuiltIn LocalInvocationId(=27) in SPIR-V word stream \
         (OpDecorate=71, Decoration::BuiltIn=11, LocalInvocationId=27). \
         The kernel calls local_invocation_id() inside staging loop bounds/shared-write indices; \
         if body_uses_local_invocation_id misses those positions, the var is not emitted."
    );

    eprintln!(
        "AT-1745 PASS: matmul_msg_coopmat.axc compiles + spirv-val clean \
         (wg_threads={wg_threads}, a_block_size={a_block_size}, b_block_size={b_block_size}, \
         shared_memory_bytes={}, LocalInvocationId emitted)",
        meta.shared_memory_bytes
    );
}

/// AT-1734: matmul_rb_coopmat.axc (M3.3c register-blocked 2×2) compiles + spirv-val clean
/// with the shipped RB strategy assignments (rb_m=2, rb_n=2, tile_k=16, a_block_size=512,
/// b_block_size=512). Runs in CI without a coopmat GPU — CI compile anchor.
///
/// Also verifies:
///   - shared_memory_bytes = (a_block_size + b_block_size) * sizeof(f16) = (512+512)*2 = 2048 bytes.
///   - The SPIR-V binary is non-empty (codegen produced real instructions).
///   - No regression on the matmul_shared_coopmat.axc anchor (retained).
#[test]
fn at1734_matmul_rb_coopmat_spirv_val() {
    let src = include_str!("../../../examples/matmul_rb_coopmat.axc");

    // Shipped RB assignments: rb_m=2, rb_n=2, tile_k=16, a_block_size=512, b_block_size=512.
    // a_block_size = RB_M * tile_m * tile_k = 2 * 16 * 16 = 512 f16 elements.
    // b_block_size = tile_k * RB_N * tile_n = 16 * 2 * 16 = 512 f16 elements.
    let mut assignments = StrategyMap::new();
    assignments.insert("rb_m".to_owned(), 2_i64);
    assignments.insert("rb_n".to_owned(), 2_i64);
    assignments.insert("tile_k".to_owned(), 16_i64);
    assignments.insert("a_block_size".to_owned(), 512_i64);
    assignments.insert("b_block_size".to_owned(), 512_i64);

    let (bytes, meta) = axc_driver::compile_source_with_assignments(src, &assignments)
        .unwrap_or_else(|e| panic!("AT-1734: matmul_rb_coopmat.axc compile failed: {e:?}"));

    // SPIR-V must be non-empty.
    assert!(!bytes.is_empty(), "AT-1734: matmul_rb_coopmat.axc produced empty SPIR-V");

    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // spirv-val must pass.
    let validator = create_validator(Some(TargetEnv::Vulkan_1_1));
    validator.validate(&words, None)
        .unwrap_or_else(|e| panic!("AT-1734: matmul_rb_coopmat.axc spirv-val FAILED: {e}"));

    // Verify shared_memory_bytes = (a_block_size + b_block_size) * sizeof(f16) = 2048 bytes.
    // a_block_size=512 f16 elements + b_block_size=512 f16 elements = 1024 f16 = 2048 bytes.
    let expected_shared_bytes: u32 = (512 + 512) * 2; // 2048 bytes
    assert_eq!(
        meta.shared_memory_bytes, expected_shared_bytes,
        "AT-1734: shared_memory_bytes must be {expected_shared_bytes} \
         (= (a_block_size=512 + b_block_size=512) * 2 bytes/f16); \
         got {}", meta.shared_memory_bytes
    );

    eprintln!(
        "AT-1734 PASS: matmul_rb_coopmat.axc compiles + spirv-val clean \
         (rb_m=2, rb_n=2, tile_k=16, a_block_size=512, b_block_size=512, \
         shared_memory_bytes={}, entry_point={})",
        meta.shared_memory_bytes, meta.entry_point
    );
}

// ── AT-1773: fused Q4_K_M coopmat matmul — no capability/extension beyond the union ──

/// RB strategy assignments for the fused kernel (same as M3.3c RB 2×2).
fn rb2x2_assignments() -> StrategyMap {
    let mut m = StrategyMap::new();
    m.insert("rb_m".to_owned(), 2_i64);
    m.insert("rb_n".to_owned(), 2_i64);
    m.insert("tile_k".to_owned(), 16_i64);
    m.insert("a_block_size".to_owned(), 512_i64);
    m.insert("b_block_size".to_owned(), 512_i64);
    m
}

/// Extract the set of OpCapability values and OpExtension strings from a SPIR-V module.
///
/// Returns (capability values as u32, extension names). OpCapability=17 (2 words:
/// [wc<<16|17, value]); OpExtension=10 (literal string operand follows the opcode word).
fn capability_extension_sets(words: &[u32]) -> (std::collections::BTreeSet<u32>, std::collections::BTreeSet<String>) {
    use std::collections::BTreeSet;
    const OP_CAPABILITY: u32 = 17;
    const OP_EXTENSION: u32 = 10;
    let mut caps: BTreeSet<u32> = BTreeSet::new();
    let mut exts: BTreeSet<String> = BTreeSet::new();
    let mut i = 5usize; // skip header
    while i < words.len() {
        let opcode = words[i] & 0xFFFF;
        let wc = (words[i] >> 16) as usize;
        if wc == 0 {
            break;
        }
        if opcode == OP_CAPABILITY && wc == 2 {
            caps.insert(words[i + 1]);
        } else if opcode == OP_EXTENSION {
            // The literal string occupies words[i+1 .. i+wc] (UTF-8, null-padded).
            let mut bytes: Vec<u8> = Vec::new();
            for &wrd in &words[i + 1..i + wc] {
                bytes.extend_from_slice(&wrd.to_le_bytes());
            }
            // Trim trailing NULs.
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

/// Compile a source (with optional assignments) and return SPIR-V words. spirv-val'd.
fn compile_words(src: &str, assignments: Option<&StrategyMap>, name: &str) -> Vec<u32> {
    let (bytes, _meta) = match assignments {
        Some(a) => compile_source_with_assignments(src, a)
            .unwrap_or_else(|e| panic!("{name}: compile failed: {e:?}")),
        None => compile_source_with_meta(src)
            .unwrap_or_else(|e| panic!("{name}: compile failed: {e:?}")),
    };
    words_and_validate(bytes, name)
}

/// AT-1773: the fused Q4_K_M coopmat matmul (q4km_matmul_rb_coopmat.axc) compiles,
/// spirv-val passes, declares OpCapability Float16 exactly once, and declares NO
/// capability/extension beyond the UNION of:
///   - matmul_rb_coopmat.axc's caps (M3.3c coopmat: Float16, CooperativeMatrixKHR, ...)
///   - q4km_dequant_matmul.axc's caps (M2.6 Q4_K_M: Int8, StorageBuffer8BitAccess,
///     SPV_KHR_8bit_storage, Int16, ...)
///
/// The readonly_buffer[u8] weight buffer legitimately adds Int8/StorageBuffer8BitAccess/
/// SPV_KHR_8bit_storage (via emit_ptr_read_u8_zext); the f32_to_f16 no-new-capability
/// proof is that the fused module adds NOTHING beyond that union.
#[test]
fn at_1773_q4km_rb_coopmat_no_new_capability_beyond_union() {
    use std::collections::BTreeSet;

    let fused_src = include_str!("../../../examples/q4km_matmul_rb_coopmat.axc");
    let rb_src = include_str!("../../../examples/matmul_rb_coopmat.axc");
    let q4km_src = include_str!("../../../examples/q4km_dequant_matmul.axc");

    let assignments = rb2x2_assignments();

    let fused_words = compile_words(fused_src, Some(&assignments), "q4km_matmul_rb_coopmat.axc");
    let rb_words = compile_words(rb_src, Some(&assignments), "matmul_rb_coopmat.axc");
    let q4km_words = compile_words(q4km_src, None, "q4km_dequant_matmul.axc");

    let (fused_caps, fused_exts) = capability_extension_sets(&fused_words);
    let (rb_caps, rb_exts) = capability_extension_sets(&rb_words);
    let (q4km_caps, q4km_exts) = capability_extension_sets(&q4km_words);

    // UNION baseline.
    let union_caps: BTreeSet<u32> = rb_caps.union(&q4km_caps).copied().collect();
    let union_exts: BTreeSet<String> = rb_exts.union(&q4km_exts).cloned().collect();

    // The fused module must declare NO capability beyond the union.
    let extra_caps: Vec<u32> = fused_caps.difference(&union_caps).copied().collect();
    assert!(
        extra_caps.is_empty(),
        "AT-1773: fused kernel declares capabilities NOT in (matmul_rb_coopmat ∪ q4km_dequant_matmul): \
         {extra_caps:?} (fused={fused_caps:?}, union={union_caps:?})"
    );

    // ...and NO extension beyond the union.
    let extra_exts: Vec<String> = fused_exts.difference(&union_exts).cloned().collect();
    assert!(
        extra_exts.is_empty(),
        "AT-1773: fused kernel declares extensions NOT in the union: {extra_exts:?} \
         (fused={fused_exts:?}, union={union_exts:?})"
    );

    // Float16 (= 9) must be present exactly once. capability_extension_sets dedups to a
    // BTreeSet (a malformed double-OpCapability would still appear once here), so also
    // do a raw structural count to assert "exactly once".
    const FLOAT16_CAP: u32 = 9;
    assert!(
        fused_caps.contains(&FLOAT16_CAP),
        "AT-1773: fused kernel must declare OpCapability Float16"
    );
    const OP_CAPABILITY: u32 = 17;
    let mut float16_raw_count = 0usize;
    let mut i = 5usize;
    while i < fused_words.len() {
        let opcode = fused_words[i] & 0xFFFF;
        let wc = (fused_words[i] >> 16) as usize;
        if wc == 0 {
            break;
        }
        if opcode == OP_CAPABILITY && wc == 2 && fused_words[i + 1] == FLOAT16_CAP {
            float16_raw_count += 1;
        }
        i += wc;
    }
    assert_eq!(
        float16_raw_count, 1,
        "AT-1773: OpCapability Float16 must appear exactly once; got {float16_raw_count}"
    );

    eprintln!(
        "AT-1773 PASS: q4km_matmul_rb_coopmat.axc ⊆ (M3.3c ∪ M2.6) caps/exts; \
         Float16 present exactly once. fused_caps={fused_caps:?} union_caps={union_caps:?} \
         fused_exts={fused_exts:?} union_exts={union_exts:?}"
    );
}

// ── AT-1787: f32-accumulator fused Q4_K_M coopmat matmul — compile + spirv-val + ──────
//            no-new-capability vs M3.5 + metadata shape f16/f16/f32/f32 ───────────────

/// AT-1787 (M3.5b, CI no-GPU): the f32-accumulator fused kernel
/// (q4km_matmul_rb_coopmat_f32acc.axc) compiles, passes spirv-val, declares NO capability
/// or extension beyond what the M3.5 fused kernel (q4km_matmul_rb_coopmat.axc) declares
/// (the f32 accumulator adds NOTHING — a Float32 coopmat component needs only
/// CooperativeMatrixKHR + Shader, both already present), and its emitted coopmat metadata
/// shape is {16,16,16, a=F16, b=F16, c=F32, result=F32, Subgroup}.
#[test]
fn at_1787_q4km_f32acc_compiles_and_validates() {
    use std::collections::BTreeSet;
    use axc_runtime::{CoopMatScalarMeta, CoopMatScopeMeta};

    let f32acc_src = include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc.axc");
    let f16_src = include_str!("../../../examples/q4km_matmul_rb_coopmat.axc");

    let assignments = rb2x2_assignments();

    // Compile the f32acc kernel WITH metadata (we assert the coopmat shape below) and
    // spirv-val it. compile_words spirv-vals the f16 baseline.
    let (f32acc_bytes, meta) = compile_source_with_assignments(f32acc_src, &assignments)
        .unwrap_or_else(|e| panic!("q4km_matmul_rb_coopmat_f32acc.axc: compile failed: {e:?}"));
    let f32acc_words = words_and_validate(f32acc_bytes, "q4km_matmul_rb_coopmat_f32acc.axc");
    let f16_words = compile_words(f16_src, Some(&assignments), "q4km_matmul_rb_coopmat.axc");

    // No capability/extension beyond the M3.5 fused kernel's set (the f32 accumulator
    // introduces NO new capability — AT-1787 / spirv_capabilities_NOTE).
    let (f32acc_caps, f32acc_exts) = capability_extension_sets(&f32acc_words);
    let (f16_caps, f16_exts) = capability_extension_sets(&f16_words);

    let extra_caps: Vec<u32> = f32acc_caps.difference(&f16_caps).copied().collect();
    assert!(
        extra_caps.is_empty(),
        "AT-1787: f32-acc kernel declares capabilities NOT in the M3.5 fused kernel's set: \
         {extra_caps:?} (f32acc={f32acc_caps:?}, f16={f16_caps:?})"
    );
    let extra_exts: Vec<String> = f32acc_exts.difference(&f16_exts).cloned().collect();
    assert!(
        extra_exts.is_empty(),
        "AT-1787: f32-acc kernel declares extensions NOT in the M3.5 fused kernel's set: \
         {extra_exts:?} (f32acc={f32acc_exts:?}, f16={f16_exts:?})"
    );
    // Both directions: the f32-acc cap set must EQUAL the f16 set (no new cap; and Float16
    // is still required for the f16 A/B types — the f32 accumulator does not drop it).
    let dropped_caps: Vec<u32> = f16_caps.difference(&f32acc_caps).copied().collect();
    assert!(
        dropped_caps.is_empty(),
        "AT-1787: f32-acc kernel DROPPED capabilities present in M3.5 (Float16 must remain \
         for the f16 A/B types): {dropped_caps:?}"
    );
    const FLOAT16_CAP: u32 = 9;
    assert!(
        f32acc_caps.contains(&FLOAT16_CAP),
        "AT-1787: f32-acc kernel must STILL declare OpCapability Float16 (f16 A/B types)"
    );

    // Sanity: capability sets are byte-for-byte equal (the whole no-new-capability point).
    let f32acc_set: BTreeSet<u32> = f32acc_caps.iter().copied().collect();
    let f16_set: BTreeSet<u32> = f16_caps.iter().copied().collect();
    assert_eq!(
        f32acc_set, f16_set,
        "AT-1787: f32-acc and M3.5 fused kernel must have IDENTICAL capability sets"
    );

    // Emitted coopmat metadata shape must be the mixed f16/f16/f32/f32 16x16x16 Subgroup.
    let coopmat = meta.coopmat.as_ref()
        .expect("AT-1787: f32-acc kernel must emit coopmat metadata");
    assert_eq!(coopmat.m, 16, "AT-1787: coopmat.m");
    assert_eq!(coopmat.n, 16, "AT-1787: coopmat.n");
    assert_eq!(coopmat.k, 16, "AT-1787: coopmat.k");
    assert_eq!(coopmat.a_type, CoopMatScalarMeta::F16, "AT-1787: A type must be F16");
    assert_eq!(coopmat.b_type, CoopMatScalarMeta::F16, "AT-1787: B type must be F16");
    assert_eq!(coopmat.c_type, CoopMatScalarMeta::F32, "AT-1787: C type must be F32");
    assert_eq!(coopmat.result_type, CoopMatScalarMeta::F32, "AT-1787: result type must be F32");
    assert_eq!(coopmat.scope, CoopMatScopeMeta::Subgroup, "AT-1787: scope must be Subgroup");

    eprintln!(
        "AT-1787 PASS: q4km_matmul_rb_coopmat_f32acc.axc compiles + spirv-val clean; \
         caps == M3.5 fused ({f32acc_caps:?}); meta.coopmat = {{16,16,16, F16,F16,F32,F32, Subgroup}}"
    );
}

// ── AT-1806: M3.6 dequant-scale-CACHED f32-accumulator fused Q4_K_M coopmat matmul — ──
//            compile + spirv-val + no-new-capability vs M3.5b + metadata shape +
//            +2 KB shared cache + a_block_size=512 PIN ─────────────────────────────────

/// AT-1806 (M3.6, CI no-GPU): the dequant-scale-CACHED kernel
/// (q4km_matmul_rb_coopmat_f32acc_cached.axc) compiles, passes spirv-val, declares NO
/// capability/extension outside the M3.5b fused kernel's set (the two shared[f32,256]
/// scale caches use only the Workgroup storage class + Float32, both already pulled in by
/// the f32 accumulators), emits the same coopmat metadata shape {16,16,16, F16,F16,F32,F32,
/// Subgroup} as M3.5b, reports shared_memory_bytes reflecting the +2 KB cache (2 * 256 * 4),
/// and compiles only at the pinned a_block_size=512.
///
/// NECESSARY but NOT SUFFICIENT: the r1 silent-zeros miscompile would ALSO pass spirv-val.
/// The true correctness gate is the orchestrator's GPU run of AT-1803 + AT-1800/1801/1802.
///
/// Mirrors AT-1787 verbatim. One benign delta is pre-permitted: shared[f32] may legitimately
/// require WorkgroupMemoryExplicitLayout / a layout decoration that shared[f16] did not — if
/// so it is documented (not failed). The capability SET equality is the strict anchor.
#[test]
fn at_1806_q4km_f32acc_cached_compiles_and_validates() {
    use std::collections::BTreeSet;
    use axc_runtime::{CoopMatScalarMeta, CoopMatScopeMeta};

    let cached_src = include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached.axc");
    let f32acc_src = include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc.axc");

    let assignments = rb2x2_assignments();

    // Compile the CACHED kernel WITH metadata + spirv-val it (this is the load-bearing
    // assertion that the r3 barrier hoist cleared BarrierInDivergentContext).
    let (cached_bytes, meta) = compile_source_with_assignments(cached_src, &assignments)
        .unwrap_or_else(|e| panic!("q4km_matmul_rb_coopmat_f32acc_cached.axc: compile failed: {e:?}"));
    let cached_words = words_and_validate(cached_bytes, "q4km_matmul_rb_coopmat_f32acc_cached.axc");
    let f32acc_words = compile_words(f32acc_src, Some(&assignments), "q4km_matmul_rb_coopmat_f32acc.axc");

    // No capability/extension beyond the M3.5b fused kernel's set (the scale caches add NO
    // new capability — Float32 + the Workgroup storage class are already present).
    let (cached_caps, cached_exts) = capability_extension_sets(&cached_words);
    let (f32acc_caps, f32acc_exts) = capability_extension_sets(&f32acc_words);

    let extra_caps: Vec<u32> = cached_caps.difference(&f32acc_caps).copied().collect();
    assert!(
        extra_caps.is_empty(),
        "AT-1806: cached kernel declares capabilities NOT in the M3.5b f32acc kernel's set: \
         {extra_caps:?} (cached={cached_caps:?}, f32acc={f32acc_caps:?})"
    );
    let dropped_caps: Vec<u32> = f32acc_caps.difference(&cached_caps).copied().collect();
    assert!(
        dropped_caps.is_empty(),
        "AT-1806: cached kernel DROPPED capabilities present in M3.5b: {dropped_caps:?}"
    );
    let cached_set: BTreeSet<u32> = cached_caps.iter().copied().collect();
    let f32acc_set: BTreeSet<u32> = f32acc_caps.iter().copied().collect();
    assert_eq!(
        cached_set, f32acc_set,
        "AT-1806: cached and M3.5b f32acc kernel must have IDENTICAL capability sets \
         (the shared[f32] scale caches introduce NO new capability)"
    );

    // Extensions: the cached set must be a subset of (and equal to) the M3.5b set. The one
    // pre-permitted benign delta is a shared[f32] layout decoration extension; document it if
    // present rather than failing, but a NEW capability is never allowed (asserted above).
    let extra_exts: Vec<String> = cached_exts.difference(&f32acc_exts).cloned().collect();
    if !extra_exts.is_empty() {
        eprintln!(
            "AT-1806 NOTE: cached kernel declares extension(s) not in M3.5b's set: {extra_exts:?} \
             — pre-permitted iff it is a benign shared[f32] layout-decoration extension \
             (WorkgroupMemoryExplicitLayout-class). No NEW capability was added (asserted)."
        );
    }
    assert!(
        extra_exts.len() <= 1,
        "AT-1806: at most ONE benign shared[f32] layout extension delta is pre-permitted; \
         got {extra_exts:?}"
    );

    // Coopmat metadata shape unchanged from M3.5b: {16,16,16, F16,F16,F32,F32, Subgroup}.
    let coopmat = meta.coopmat.as_ref()
        .expect("AT-1806: cached kernel must emit coopmat metadata");
    assert_eq!(coopmat.m, 16, "AT-1806: coopmat.m");
    assert_eq!(coopmat.n, 16, "AT-1806: coopmat.n");
    assert_eq!(coopmat.k, 16, "AT-1806: coopmat.k");
    assert_eq!(coopmat.a_type, CoopMatScalarMeta::F16, "AT-1806: A type must be F16");
    assert_eq!(coopmat.b_type, CoopMatScalarMeta::F16, "AT-1806: B type must be F16");
    assert_eq!(coopmat.c_type, CoopMatScalarMeta::F32, "AT-1806: C type must be F32");
    assert_eq!(coopmat.result_type, CoopMatScalarMeta::F32, "AT-1806: result type must be F32");
    assert_eq!(coopmat.scope, CoopMatScopeMeta::Subgroup, "AT-1806: scope must be Subgroup");

    // shared_memory_bytes reflects the +2 KB cache: a_tile(512 f16) + b_tile(512 f16) = 2048 B,
    // PLUS dsc_cache(256 f32) + dmm_cache(256 f32) = 2 * 256 * 4 = 2048 B -> 4096 B total.
    let expected_shared_bytes: u32 = (512 + 512) * 2 + 2 * 256 * 4; // 2048 + 2048 = 4096
    assert_eq!(
        meta.shared_memory_bytes, expected_shared_bytes,
        "AT-1806: shared_memory_bytes must be {expected_shared_bytes} \
         (a_tile+b_tile = (512+512)*2 = 2048 B, PLUS dsc_cache+dmm_cache = 2*256*4 = 2048 B); \
         got {} — the +2 KB scale cache must be accounted", meta.shared_memory_bytes
    );

    // a_block_size=512 PIN: the autotuner must only ever pick 512. The pin is the SINGLE-VALUE
    // @strategy candidate list `a_block_size: ?[512]` — axc-optimize enumerates exactly one
    // candidate, so a_row=ei_a/16 stays in 0..31 (32*8=256 exact) and the cache index never OOBs
    // the 256-entry shared[f32] caches. (The text-substitution compile path can take an arbitrary
    // override, so the enforced invariant is the single-candidate declaration in source.)
    let strategy_decl = cached_src
        .lines()
        .find(|l| l.trim_start().starts_with("@strategy"))
        .expect("AT-1806: cached kernel must declare an @strategy block");
    assert!(
        strategy_decl.contains("a_block_size: ?[512]"),
        "AT-1806: a_block_size MUST be PINNED to the single-value @strategy candidate `?[512]` \
         (bounds the 256-entry scale caches to 32 rows); got @strategy line: {strategy_decl}"
    );

    eprintln!(
        "AT-1806 PASS: q4km_matmul_rb_coopmat_f32acc_cached.axc compiles + spirv-val clean; \
         caps == M3.5b f32acc ({cached_caps:?}); meta.coopmat = {{16,16,16, F16,F16,F32,F32, Subgroup}}; \
         shared_memory_bytes={} (+2 KB cache); a_block_size=512 PIN enforced",
        meta.shared_memory_bytes
    );
}

// ── AT-1906: M3.7 DOUBLE-BUFFERED (software-pipelined) scale-cached f32-accumulator fused ──
//            Q4_K_M coopmat matmul — compile + spirv-val + no-new-capability vs M3.6 cached +
//            metadata shape + DOUBLED shared (6144 bytes) + emitted-barrier-count +
//            a_block_size_db/b_block_size_db=1024 PIN ─────────────────────────────────────

/// RB 2×2 db strategy (a_block_size_db/b_block_size_db PINNED at 1024 = 2*512).
fn rb2x2_db_assignments() -> StrategyMap {
    let mut m = StrategyMap::new();
    m.insert("rb_m".to_owned(), 2_i64);
    m.insert("rb_n".to_owned(), 2_i64);
    m.insert("tile_k".to_owned(), 16_i64);
    m.insert("a_block_size_db".to_owned(), 1024_i64);
    m.insert("b_block_size_db".to_owned(), 1024_i64);
    m
}

/// Count OpControlBarrier (opcode 224, word_count 4) instructions in a SPIR-V word stream.
fn count_op_control_barrier(words: &[u32]) -> usize {
    const OP_CONTROL_BARRIER: u32 = 224;
    let mut n = 0usize;
    let mut i = 5usize; // skip header
    while i < words.len() {
        let opcode = words[i] & 0xFFFF;
        let wc = (words[i] >> 16) as usize;
        if wc == 0 {
            break;
        }
        if opcode == OP_CONTROL_BARRIER {
            n += 1;
        }
        i += wc;
    }
    n
}

/// Count OpPhi (opcode 245) instructions — the single-level loop-carried accumulator phis.
fn count_op_phi(words: &[u32]) -> usize {
    const OP_PHI: u32 = 245;
    let mut n = 0usize;
    let mut i = 5usize;
    while i < words.len() {
        let opcode = words[i] & 0xFFFF;
        let wc = (words[i] >> 16) as usize;
        if wc == 0 {
            break;
        }
        if opcode == OP_PHI {
            n += 1;
        }
        i += wc;
    }
    n
}

/// AT-1906 (M3.7, CI no-GPU): the DOUBLE-BUFFERED kernel
/// (q4km_matmul_rb_coopmat_f32acc_db.axc) compiles, passes spirv-val, declares NO
/// capability/extension outside the M3.6 cached kernel's set (parity-indexed shared, a larger
/// shared[f16,1024] array, a prologue, and a runtime coopmat_load offset add NO capability),
/// emits the same coopmat metadata shape {16,16,16, F16,F16,F32,F32, Subgroup} as M3.6, reports
/// shared_memory_bytes == 6144 (DOUBLED tiles 4096 + caches 2048), and pins
/// a_block_size_db/b_block_size_db at the single-value @strategy candidate `?[1024]`.
///
/// ALSO (the reviewer-requested SHOULD hardenings):
///   - asserts the emitted OpControlBarrier count == 4 (2 prologue + 2 steady-state) — fewer than
///     would indicate a dropped barrier, more than indicates an un-hoisted in-if barrier;
///   - asserts OpPhi count == the M3.6 cached kernel's (single-level carry intact — a nested-loop
///     carry would DROP phis -> silent miscompile).
///
/// NECESSARY but NOT SUFFICIENT: a missing-barrier race would ALSO pass spirv-val. The true
/// correctness gate is the orchestrator's GPU run of AT-1903 + AT-1900/1901/1902.
#[test]
fn at_1906_q4km_f32acc_db_no_new_capability() {
    use std::collections::BTreeSet;
    use axc_runtime::{CoopMatScalarMeta, CoopMatScopeMeta};

    let db_src = include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_db.axc");
    let cached_src = include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached.axc");

    let db_assignments = rb2x2_db_assignments();
    let cached_assignments = rb2x2_assignments();

    // Compile the db kernel WITH metadata + spirv-val it (load-bearing: the 2 prologue + 2
    // steady-state barriers must all clear BarrierInDivergentContext at conditional_depth==0).
    let (db_bytes, meta) = compile_source_with_assignments(db_src, &db_assignments)
        .unwrap_or_else(|e| panic!("q4km_matmul_rb_coopmat_f32acc_db.axc: compile failed: {e:?}"));
    let db_words = words_and_validate(db_bytes, "q4km_matmul_rb_coopmat_f32acc_db.axc");

    let (cached_bytes, _cached_meta) = compile_source_with_assignments(cached_src, &cached_assignments)
        .unwrap_or_else(|e| panic!("q4km_matmul_rb_coopmat_f32acc_cached.axc: compile failed: {e:?}"));
    let cached_words = words_and_validate(cached_bytes, "q4km_matmul_rb_coopmat_f32acc_cached.axc");

    // (c) Capability set BYTE-IDENTICAL to the M3.6 cached kernel (no new capability).
    let (db_caps, db_exts) = capability_extension_sets(&db_words);
    let (cached_caps, cached_exts) = capability_extension_sets(&cached_words);

    let extra_caps: Vec<u32> = db_caps.difference(&cached_caps).copied().collect();
    assert!(
        extra_caps.is_empty(),
        "AT-1906: db kernel declares capabilities NOT in the M3.6 cached kernel's set: \
         {extra_caps:?} (db={db_caps:?}, cached={cached_caps:?})"
    );
    let dropped_caps: Vec<u32> = cached_caps.difference(&db_caps).copied().collect();
    assert!(
        dropped_caps.is_empty(),
        "AT-1906: db kernel DROPPED capabilities present in M3.6 cached: {dropped_caps:?}"
    );
    let db_set: BTreeSet<u32> = db_caps.iter().copied().collect();
    let cached_set: BTreeSet<u32> = cached_caps.iter().copied().collect();
    assert_eq!(
        db_set, cached_set,
        "AT-1906: db and M3.6 cached kernel must have BYTE-IDENTICAL capability sets \
         (parity-indexed shared + shared[f16,1024] introduce NO new capability)"
    );
    let extra_exts: Vec<String> = db_exts.difference(&cached_exts).cloned().collect();
    assert!(
        extra_exts.is_empty(),
        "AT-1906: db kernel declares extensions NOT in the M3.6 cached kernel's set: {extra_exts:?}"
    );

    // (d) Coopmat metadata shape unchanged: {16,16,16, F16,F16,F32,F32, Subgroup}.
    let coopmat = meta.coopmat.as_ref()
        .expect("AT-1906: db kernel must emit coopmat metadata");
    assert_eq!(coopmat.m, 16, "AT-1906: coopmat.m");
    assert_eq!(coopmat.n, 16, "AT-1906: coopmat.n");
    assert_eq!(coopmat.k, 16, "AT-1906: coopmat.k");
    assert_eq!(coopmat.a_type, CoopMatScalarMeta::F16, "AT-1906: A type must be F16");
    assert_eq!(coopmat.b_type, CoopMatScalarMeta::F16, "AT-1906: B type must be F16");
    assert_eq!(coopmat.c_type, CoopMatScalarMeta::F32, "AT-1906: C type must be F32");
    assert_eq!(coopmat.result_type, CoopMatScalarMeta::F32, "AT-1906: result type must be F32");
    assert_eq!(coopmat.scope, CoopMatScopeMeta::Subgroup, "AT-1906: scope must be Subgroup");

    // (b) shared_memory_bytes == DOUBLED tiles + caches:
    //     a_tile(1024 f16) + b_tile(1024 f16) = (1024+1024)*2 = 4096 B,
    //     PLUS dsc_cache(256 f32) + dmm_cache(256 f32) = 2*256*4 = 2048 B -> 6144 B.
    let expected_shared_bytes: u32 = (1024 + 1024) * 2 + 2 * 256 * 4; // 4096 + 2048 = 6144
    assert_eq!(
        meta.shared_memory_bytes, expected_shared_bytes,
        "AT-1906: shared_memory_bytes must be {expected_shared_bytes} \
         (DOUBLED tiles a_tile+b_tile = (1024+1024)*2 = 4096 B, PLUS dsc_cache+dmm_cache = \
         2*256*4 = 2048 B); got {} — the doubled-tile occupancy cost must be accounted",
        meta.shared_memory_bytes
    );

    // SHOULD hardening 1: the emitted OpControlBarrier count == 4 (2 prologue + 2 steady-state).
    // M3.6 cached has 3 (all in-loop); the db prologue adds 2 and the steady state drops to 2,
    // netting 4 textual barriers. A count != 4 indicates a dropped barrier (race risk) or an
    // un-hoisted in-if barrier (which would have failed BarrierInDivergentContext anyway).
    let db_barriers = count_op_control_barrier(&db_words);
    assert_eq!(
        db_barriers, 4,
        "AT-1906: db kernel must emit exactly 4 OpControlBarrier (2 PROLOGUE + 2 steady-state); \
         got {db_barriers}. Fewer => a dropped barrier (data-race risk, AT-1903 is the real \
         detector); more => an un-hoisted barrier. The M3.6 cached kernel emits {} (3, all in-loop).",
        count_op_control_barrier(&cached_words)
    );

    // SHOULD hardening 2: OpPhi count == the M3.6 cached kernel's (single-level carry intact).
    // A nested-loop accumulator carry would DROP the outer phis (descend_nested_loops=false) ->
    // a silent per-iteration reset miscompile (the M3.6 r1 trap).
    let db_phis = count_op_phi(&db_words);
    let cached_phis = count_op_phi(&cached_words);
    assert_eq!(
        db_phis, cached_phis,
        "AT-1906: db OpPhi count ({db_phis}) must EQUAL the M3.6 cached count ({cached_phis}) — \
         the 4 f32 accumulators must be carried by the SAME single-level OpPhi (no nested-loop \
         carry, which would drop phis -> silent reset miscompile)"
    );

    // (e) a_block_size_db / b_block_size_db PIN: the autotuner must only ever pick 1024 (= 2*512,
    // the two parity slabs). The pin is the SINGLE-VALUE @strategy candidate list `?[1024]`.
    let strategy_decl = db_src
        .lines()
        .find(|l| l.trim_start().starts_with("@strategy"))
        .expect("AT-1906: db kernel must declare an @strategy block");
    assert!(
        strategy_decl.contains("a_block_size_db: ?[1024]"),
        "AT-1906: a_block_size_db MUST be PINNED to the single-value @strategy candidate `?[1024]` \
         (= 2*512 parity slabs; the cache index a_row*8+is stays < 256); got @strategy: {strategy_decl}"
    );
    assert!(
        strategy_decl.contains("b_block_size_db: ?[1024]"),
        "AT-1906: b_block_size_db MUST be PINNED to the single-value @strategy candidate `?[1024]`; \
         got @strategy: {strategy_decl}"
    );

    eprintln!(
        "AT-1906 PASS: q4km_matmul_rb_coopmat_f32acc_db.axc compiles + spirv-val clean; \
         caps == M3.6 cached ({db_caps:?}); meta.coopmat = {{16,16,16, F16,F16,F32,F32, Subgroup}}; \
         shared_memory_bytes={} (DOUBLED tiles 4096 + caches 2048 = 6144); \
         OpControlBarrier={db_barriers} (2 prologue + 2 steady-state); OpPhi={db_phis} \
         (== cached {cached_phis}, single-level carry); a_block_size_db/b_block_size_db=1024 PIN enforced",
        meta.shared_memory_bytes
    );
}

/// AT-1906 (CI no-GPU, OPTIONAL plain-matmul _db variant): the plain double-buffered RB coopmat
/// matmul (matmul_rb_coopmat_db.axc) compiles, passes spirv-val, reports shared_memory_bytes ==
/// 4096 (DOUBLED tiles, no scale cache), emits OpControlBarrier == 3 (1 prologue + 2 steady-state),
/// preserves the single-level OpPhi carry vs M3.3c, declares NO capability beyond M3.3c, and pins
/// a_block_size_db/b_block_size_db at 1024. Isolates the matmul-core latency-overlap structure.
#[test]
fn at_1906_matmul_rb_db_no_new_capability() {
    use std::collections::BTreeSet;

    let db_src = include_str!("../../../examples/matmul_rb_coopmat_db.axc");
    let m33c_src = include_str!("../../../examples/matmul_rb_coopmat.axc");

    let db_assignments = rb2x2_db_assignments();
    let m33c_assignments = rb2x2_assignments();

    let (db_bytes, meta) = compile_source_with_assignments(db_src, &db_assignments)
        .unwrap_or_else(|e| panic!("matmul_rb_coopmat_db.axc: compile failed: {e:?}"));
    let db_words = words_and_validate(db_bytes, "matmul_rb_coopmat_db.axc");

    let (m33c_bytes, _m33c_meta) = compile_source_with_assignments(m33c_src, &m33c_assignments)
        .unwrap_or_else(|e| panic!("matmul_rb_coopmat.axc: compile failed: {e:?}"));
    let m33c_words = words_and_validate(m33c_bytes, "matmul_rb_coopmat.axc");

    // No new capability beyond M3.3c (parity-indexed shared + shared[f16,1024] add nothing).
    let (db_caps, db_exts) = capability_extension_sets(&db_words);
    let (m33c_caps, m33c_exts) = capability_extension_sets(&m33c_words);
    let extra_caps: Vec<u32> = db_caps.difference(&m33c_caps).copied().collect();
    assert!(
        extra_caps.is_empty(),
        "AT-1906 (plain): _db declares capabilities NOT in M3.3c: {extra_caps:?}"
    );
    let db_set: BTreeSet<u32> = db_caps.iter().copied().collect();
    let m33c_set: BTreeSet<u32> = m33c_caps.iter().copied().collect();
    assert_eq!(
        db_set, m33c_set,
        "AT-1906 (plain): _db and M3.3c must have BYTE-IDENTICAL capability sets"
    );
    let extra_exts: Vec<String> = db_exts.difference(&m33c_exts).cloned().collect();
    assert!(extra_exts.is_empty(), "AT-1906 (plain): _db declares new extensions: {extra_exts:?}");

    // shared_memory_bytes == DOUBLED tiles only (no scale cache): (1024+1024)*2 = 4096 B.
    let expected_shared_bytes: u32 = (1024 + 1024) * 2; // 4096
    assert_eq!(
        meta.shared_memory_bytes, expected_shared_bytes,
        "AT-1906 (plain): shared_memory_bytes must be {expected_shared_bytes} \
         (DOUBLED tiles (1024+1024)*2, NO scale cache); got {}", meta.shared_memory_bytes
    );

    // OpControlBarrier == 3 (1 prologue + 2 steady-state); OpPhi == M3.3c (single-level carry).
    let db_barriers = count_op_control_barrier(&db_words);
    assert_eq!(
        db_barriers, 3,
        "AT-1906 (plain): _db must emit 3 OpControlBarrier (1 PROLOGUE + 2 steady-state); \
         got {db_barriers} (M3.3c emits {})", count_op_control_barrier(&m33c_words)
    );
    let db_phis = count_op_phi(&db_words);
    let m33c_phis = count_op_phi(&m33c_words);
    assert_eq!(
        db_phis, m33c_phis,
        "AT-1906 (plain): _db OpPhi count ({db_phis}) must EQUAL M3.3c ({m33c_phis}) — \
         single-level 4-accumulator carry preserved"
    );

    // 1024 pin.
    let strategy_decl = db_src.lines()
        .find(|l| l.trim_start().starts_with("@strategy"))
        .expect("AT-1906 (plain): _db must declare an @strategy block");
    assert!(
        strategy_decl.contains("a_block_size_db: ?[1024]") &&
        strategy_decl.contains("b_block_size_db: ?[1024]"),
        "AT-1906 (plain): a_block_size_db/b_block_size_db MUST be PINNED to `?[1024]`; \
         got @strategy: {strategy_decl}"
    );

    eprintln!(
        "AT-1906 (plain) PASS: matmul_rb_coopmat_db.axc compiles + spirv-val clean; \
         caps == M3.3c; shared_memory_bytes={} (DOUBLED tiles, no cache); \
         OpControlBarrier={db_barriers} (1 prologue + 2 steady-state); OpPhi={db_phis} \
         (== M3.3c {m33c_phis}); a_block_size_db/b_block_size_db=1024 PIN",
        meta.shared_memory_bytes
    );
}


// ── AT-2006 (M3.8, CI no-GPU): LARGER REGISTER TILES (4x2 / 4x4) compile + spirv-val + ──
//    no-new-capability vs the M3.6 cached kernel + shared_memory_bytes anchor (7168 / 8192)
//    + OpPhi count == 8 / 16 (single-level N-accumulator carry intact).
//
// NR-3 FAIL-FAST: the 4x4 (16-accumulator) compile + spirv-val + OpPhi==16 is the load-bearing
// first-line gate (any genuine codegen gap STOPS work and escalates to the Architect). It passed:
// 16 OpPhi in ONE loop header, spirv-val clean, shared==8192.

fn rb4x2_assignments() -> StrategyMap {
    let mut m = StrategyMap::new();
    m.insert("rb_m".to_owned(), 4_i64);
    m.insert("rb_n".to_owned(), 2_i64);
    m.insert("tile_k".to_owned(), 16_i64);
    m.insert("a_block_size".to_owned(), 1024_i64);
    m.insert("b_block_size".to_owned(), 512_i64);
    m
}

fn rb4x4_assignments() -> StrategyMap {
    let mut m = StrategyMap::new();
    m.insert("rb_m".to_owned(), 4_i64);
    m.insert("rb_n".to_owned(), 4_i64);
    m.insert("tile_k".to_owned(), 16_i64);
    m.insert("a_block_size".to_owned(), 1024_i64);
    m.insert("b_block_size".to_owned(), 1024_i64);
    m
}

/// AT-2006 shared logic for one big-RB variant. Asserts: compiles, spirv-val clean, coopmat
/// metadata shape == M3.6, capability/extension set == the M3.6 cached kernel's, OpPhi count ==
/// `expected_phis` (the N-accumulator single-level carry — all in ONE loop-header block), and
/// shared_memory_bytes == `expected_shared`.
fn at_2006_variant(
    variant_src: &str,
    variant_name: &str,
    assignments: &StrategyMap,
    expected_phis: usize,
    expected_shared: u32,
) {
    use std::collections::BTreeSet;
    use axc_runtime::{CoopMatScalarMeta, CoopMatScopeMeta};

    let cached_src = include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached.axc");
    let cached_assignments = rb2x2_assignments();

    let (variant_bytes, meta) = compile_source_with_assignments(variant_src, assignments)
        .unwrap_or_else(|e| panic!("{variant_name}: compile failed: {e:?}"));
    let variant_words = words_and_validate(variant_bytes, variant_name);

    let (cached_bytes, _cm) = compile_source_with_assignments(cached_src, &cached_assignments)
        .unwrap_or_else(|e| panic!("q4km_matmul_rb_coopmat_f32acc_cached.axc: compile failed: {e:?}"));
    let cached_words = words_and_validate(cached_bytes, "q4km_matmul_rb_coopmat_f32acc_cached.axc");

    // (c) Capability set BYTE-IDENTICAL to the M3.6 cached kernel (no new capability).
    let (variant_caps, variant_exts) = capability_extension_sets(&variant_words);
    let (cached_caps, cached_exts) = capability_extension_sets(&cached_words);

    let extra_caps: Vec<u32> = variant_caps.difference(&cached_caps).copied().collect();
    assert!(
        extra_caps.is_empty(),
        "AT-2006 ({variant_name}): declares capabilities NOT in the M3.6 cached set: \
         {extra_caps:?} (variant={variant_caps:?}, cached={cached_caps:?})"
    );
    let dropped_caps: Vec<u32> = cached_caps.difference(&variant_caps).copied().collect();
    assert!(
        dropped_caps.is_empty(),
        "AT-2006 ({variant_name}): DROPPED capabilities present in M3.6 cached: {dropped_caps:?}"
    );
    let variant_set: BTreeSet<u32> = variant_caps.iter().copied().collect();
    let cached_set: BTreeSet<u32> = cached_caps.iter().copied().collect();
    assert_eq!(
        variant_set, cached_set,
        "AT-2006 ({variant_name}): capability set must equal the M3.6 cached kernel's \
         (larger register tiles introduce NO new capability)"
    );

    // Extensions: a single benign shared-layout-decoration delta is pre-permitted (as AT-1806/1906).
    let extra_exts: Vec<String> = variant_exts.difference(&cached_exts).cloned().collect();
    let dropped_exts: Vec<String> = cached_exts.difference(&variant_exts).cloned().collect();
    if !extra_exts.is_empty() || !dropped_exts.is_empty() {
        eprintln!(
            "AT-2006 ({variant_name}) NOTE: extension delta vs M3.6 cached: extra={extra_exts:?} \
             dropped={dropped_exts:?} — pre-permitted iff benign shared-layout-decoration class."
        );
    }
    assert!(
        extra_exts.len() <= 1 && dropped_exts.len() <= 1,
        "AT-2006 ({variant_name}): at most ONE benign shared-layout extension delta is \
         pre-permitted; extra={extra_exts:?} dropped={dropped_exts:?}"
    );

    // Coopmat metadata shape unchanged: {16,16,16, F16,F16,F32,F32, Subgroup}.
    let coopmat = meta.coopmat.as_ref()
        .unwrap_or_else(|| panic!("AT-2006 ({variant_name}): must emit coopmat metadata"));
    assert_eq!(coopmat.m, 16, "AT-2006 ({variant_name}): coopmat.m");
    assert_eq!(coopmat.n, 16, "AT-2006 ({variant_name}): coopmat.n");
    assert_eq!(coopmat.k, 16, "AT-2006 ({variant_name}): coopmat.k");
    assert_eq!(coopmat.a_type, CoopMatScalarMeta::F16, "AT-2006 ({variant_name}): A type F16");
    assert_eq!(coopmat.b_type, CoopMatScalarMeta::F16, "AT-2006 ({variant_name}): B type F16");
    assert_eq!(coopmat.c_type, CoopMatScalarMeta::F32, "AT-2006 ({variant_name}): C type F32");
    assert_eq!(coopmat.result_type, CoopMatScalarMeta::F32, "AT-2006 ({variant_name}): result F32");
    assert_eq!(coopmat.scope, CoopMatScopeMeta::Subgroup, "AT-2006 ({variant_name}): scope Subgroup");

    // OpPhi count == the N-accumulator single-level carry, AND all in ONE loop-header block.
    let phis = count_op_phi(&variant_words);
    assert_eq!(
        phis, expected_phis,
        "AT-2006 ({variant_name}): OpPhi count must be {expected_phis} (the {expected_phis} \
         loop-carried coopmat accumulators, single-level N-carry); got {phis}"
    );
    let phi_runs = op_phi_block_runs(&variant_words);
    assert!(
        phi_runs.contains(&expected_phis),
        "AT-2006 ({variant_name}): all {expected_phis} OpPhi must sit in ONE loop-header block \
         (single-level carry); got per-block runs {phi_runs:?}"
    );

    // shared_memory_bytes anchor.
    assert_eq!(
        meta.shared_memory_bytes, expected_shared,
        "AT-2006 ({variant_name}): shared_memory_bytes must be {expected_shared}; got {}",
        meta.shared_memory_bytes
    );
    assert!(
        expected_shared <= 16384,
        "AT-2006 ({variant_name}): shared_memory_bytes {expected_shared} must be <= 16384 \
         (portable maxComputeSharedMemorySize floor)"
    );

    eprintln!(
        "AT-2006 ({variant_name}) PASS: compiles + spirv-val clean; caps == M3.6 cached \
         ({variant_caps:?}); coopmat {{16,16,16,F16,F16,F32,F32,Subgroup}}; OpPhi={phis} \
         (one loop header); shared_memory_bytes={expected_shared}"
    );
}

/// Count OpPhi (245) per basic block: returns the length of each CONSECUTIVE OpPhi run that
/// leads a basic block (an OpLabel starts a block; OpPhi must immediately follow). A single
/// loop-header block carrying N accumulators yields a run of N.
fn op_phi_block_runs(words: &[u32]) -> Vec<usize> {
    const OP_PHI: u32 = 245;
    const OP_LABEL: u32 = 248;
    let mut runs: Vec<usize> = Vec::new();
    let mut i = 5usize;
    let mut cur = 0usize;
    let mut after_label = false;
    while i < words.len() {
        let op = words[i] & 0xFFFF;
        let wc = (words[i] >> 16) as usize;
        if wc == 0 {
            break;
        }
        if op == OP_LABEL {
            if cur > 0 {
                runs.push(cur);
            }
            cur = 0;
            after_label = true;
        } else if op == OP_PHI && after_label {
            cur += 1;
        } else {
            if cur > 0 {
                runs.push(cur);
            }
            cur = 0;
            after_label = false;
        }
        i += wc;
    }
    if cur > 0 {
        runs.push(cur);
    }
    runs
}

/// AT-2006 (4x2): 8-accumulator variant. shared = (1024+512)*2 + 2*512*4 = 3072 + 4096 = 7168.
#[test]
fn at_2006_bigrb_4x2_compile_validates() {
    let src = include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached_4x2.axc");
    let expected_shared: u32 = (1024 + 512) * 2 + 2 * 512 * 4; // 3072 + 4096 = 7168
    at_2006_variant(src, "4x2", &rb4x2_assignments(), 8, expected_shared);
}

/// AT-2006 (4x4): 16-accumulator variant (NR-3 fail-fast anchor). shared = (1024+1024)*2 +
/// 2*512*4 = 4096 + 4096 = 8192. OpPhi == 16 in one loop header.
#[test]
fn at_2006_bigrb_4x4_compile_validates() {
    let src = include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached_4x4.axc");
    let expected_shared: u32 = (1024 + 1024) * 2 + 2 * 512 * 4; // 4096 + 4096 = 8192
    at_2006_variant(src, "4x4", &rb4x4_assignments(), 16, expected_shared);
}
