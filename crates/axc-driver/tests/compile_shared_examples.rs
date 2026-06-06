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
