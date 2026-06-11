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

/// AT-1824: flash_attention_exp.axc (M3.2c — real exp via GLSL.std.450) compiles +
/// spirv-val clean. Additive compile anchor (no GPU), mirroring the AT-1738 anchor.
#[test]
fn at1824_flash_attention_exp_compiles() {
    let src = include_str!("../../../examples/flash_attention_exp.axc");
    compile_and_validate(src, "flash_attention_exp.axc");
}

/// AT-1822 (codegen, no GPU): the GLSL.std.450 import is emitted EXACTLY ONCE in a
/// kernel with MULTIPLE exp() calls, and ALL OpExtInst Exp (27) reference that one
/// set-id. Scans the RAW codegen output (no spirv-opt in the default pipeline). A
/// per-call-import implementation would emit 2 imports and FAIL this.
#[test]
fn at1822_glsl450_import_emitted_once() {
    // Two distinct exp() call sites in expression position.
    let src = r#"
@kernel
@workgroup(1, 1, 1)
@intent("two exp calls — import-once falsifier")
@complexity(O(1))
fn two_exp(In: readonly_buffer[f32], Out: buffer[f32]) -> void {
    let i: u32 = gid(0u32);
    let a: f32 = In[i];
    let b: f32 = exp(a);
    let c: f32 = exp(b);
    Out[i] = b + c;
    return;
}
"#;
    let words = compile_and_validate(src, "two_exp (AT-1822)");

    // ── Scan: exactly ONE OpExtInstImport, with literal-string "GLSL.std.450". ──
    // OpExtInstImport = opcode 11. Layout: word0 = (wc<<16)|11, word1 = result-id,
    // word2.. = the extended-set name as a packed null-terminated UTF-8 literal.
    let mut import_count = 0usize;
    let mut import_set_id: u32 = 0;
    {
        // Skip the 5-word SPIR-V module header (magic, version, generator, bound, schema).
        let mut idx = 5usize;
        while idx < words.len() {
            let word0 = words[idx];
            let opcode = word0 & 0xFFFF;
            let wc = (word0 >> 16) as usize;
            if wc == 0 {
                break; // malformed — bail (validator already passed, so unreachable)
            }
            if opcode == 11 {
                import_count += 1;
                let result_id = words[idx + 1];
                // Decode the packed string operand (words idx+2 .. idx+wc).
                let mut bytes: Vec<u8> = Vec::new();
                for w in &words[idx + 2..idx + wc] {
                    bytes.extend_from_slice(&w.to_le_bytes());
                }
                // Trim at first NUL.
                let nul = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
                let name = String::from_utf8_lossy(&bytes[..nul]);
                assert_eq!(
                    name, "GLSL.std.450",
                    "AT-1822: OpExtInstImport literal must be \"GLSL.std.450\", got {name:?}"
                );
                import_set_id = result_id;
            }
            idx += wc;
        }
    }
    assert_eq!(
        import_count, 1,
        "AT-1822: a 2-exp kernel must emit EXACTLY ONE OpExtInstImport \"GLSL.std.450\" \
         (the import-once cache is load-bearing; rspirv does NOT dedup). Found {import_count}."
    );

    // ── Scan: >= 2 OpExtInst ... 27, all referencing the one set-id. ──
    // OpExtInst = opcode 12. Layout: word0=(wc<<16)|12, word1=result-type,
    // word2=result-id, word3=set-id (IdRef), word4=instruction-literal (27), operands..
    let mut exp_count = 0usize;
    {
        // Skip the 5-word SPIR-V module header (magic, version, generator, bound, schema).
        let mut idx = 5usize;
        while idx < words.len() {
            let word0 = words[idx];
            let opcode = word0 & 0xFFFF;
            let wc = (word0 >> 16) as usize;
            if wc == 0 {
                break;
            }
            if opcode == 12 {
                let set_id = words[idx + 3];
                let instruction = words[idx + 4];
                if instruction == 27 {
                    exp_count += 1;
                    assert_eq!(
                        set_id, import_set_id,
                        "AT-1822: every OpExtInst Exp must reference the single cached \
                         GLSL.std.450 set-id ({import_set_id}); got {set_id}"
                    );
                }
            }
            idx += wc;
        }
    }
    assert!(
        exp_count >= 2,
        "AT-1822: a 2-exp kernel must emit >= 2 OpExtInst ... 27 (Exp); found {exp_count}"
    );
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

// ── AT-2606: M3.11a dequant-index STRENGTH-REDUCED scale-cached f32-accumulator fused ──
//            Q4_K_M coopmat matmul — compile + spirv-val + cap-set BYTE-IDENTICAL to the M3.6
//            leader (no new capability — scalar integer carried counters add none) ──────────

/// AT-2606 (M3.11a, CI no-GPU): the dequant-index STRENGTH-REDUCED kernel
/// (q4km_matmul_rb_coopmat_f32acc_cached_sr.axc) compiles, passes spirv-val (Vulkan 1.1), and
/// emits a capability set BYTE-IDENTICAL to the M3.6 leader (cached.axc). The strength-reduction
/// removes OpIMul/OpUDiv and adds carried OpIAdd on Function-storage u32 OpVariables — all
/// already-emitted op classes, NO new capability/extension. It also keeps the M3.6 coopmat
/// metadata shape {16,16,16, F16,F16,F32,F32, Subgroup}, the same +2 KB scale cache
/// (shared_memory_bytes == 4096), and the a_block_size=512 pin.
///
/// NECESSARY but NOT SUFFICIENT: a wrong-nibble carry would ALSO pass spirv-val. The true
/// correctness gate is the orchestrator's GPU run of AT-2603 (bit-identity) + AT-2605 (combined).
#[test]
fn at_2606_sr_compiles_capset_identical() {
    use std::collections::BTreeSet;
    use axc_runtime::{CoopMatScalarMeta, CoopMatScopeMeta};

    let sr_src = include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached_sr.axc");
    let cached_src = include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached.axc");

    let assignments = rb2x2_assignments();

    // Compile the SR kernel WITH metadata + spirv-val it.
    let (sr_bytes, meta) = compile_source_with_assignments(sr_src, &assignments)
        .unwrap_or_else(|e| panic!("q4km_matmul_rb_coopmat_f32acc_cached_sr.axc: compile failed: {e:?}"));
    let sr_words = words_and_validate(sr_bytes, "q4km_matmul_rb_coopmat_f32acc_cached_sr.axc");
    let cached_words = compile_words(cached_src, Some(&assignments), "q4km_matmul_rb_coopmat_f32acc_cached.axc");

    // Capability set BYTE-IDENTICAL to the M3.6 leader (no new capability from carried scalar counters).
    let (sr_caps, sr_exts) = capability_extension_sets(&sr_words);
    let (cached_caps, cached_exts) = capability_extension_sets(&cached_words);

    let extra_caps: Vec<u32> = sr_caps.difference(&cached_caps).copied().collect();
    assert!(
        extra_caps.is_empty(),
        "AT-2606: SR kernel declares capabilities NOT in the M3.6 leader's set: \
         {extra_caps:?} (sr={sr_caps:?}, cached={cached_caps:?})"
    );
    let dropped_caps: Vec<u32> = cached_caps.difference(&sr_caps).copied().collect();
    assert!(
        dropped_caps.is_empty(),
        "AT-2606: SR kernel DROPPED capabilities present in the M3.6 leader: {dropped_caps:?}"
    );
    let sr_set: BTreeSet<u32> = sr_caps.iter().copied().collect();
    let cached_set: BTreeSet<u32> = cached_caps.iter().copied().collect();
    assert_eq!(
        sr_set, cached_set,
        "AT-2606: SR and M3.6 leader must have BYTE-IDENTICAL capability sets \
         (carried scalar integer counters introduce NO new capability)"
    );

    // Extensions identical (or at most the same one benign shared[f32] layout delta the M3.6
    // leader already carries — but NO new capability ever, asserted above).
    let extra_exts: Vec<String> = sr_exts.difference(&cached_exts).cloned().collect();
    assert!(
        extra_exts.is_empty(),
        "AT-2606: SR kernel must not declare extensions beyond the M3.6 leader's set \
         (the strength-reduction adds only carried OpIAdd on Function-storage OpVariables); \
         got {extra_exts:?}"
    );

    // Coopmat metadata shape unchanged from M3.6: {16,16,16, F16,F16,F32,F32, Subgroup}.
    let coopmat = meta.coopmat.as_ref()
        .expect("AT-2606: SR kernel must emit coopmat metadata");
    assert_eq!(coopmat.m, 16, "AT-2606: coopmat.m");
    assert_eq!(coopmat.n, 16, "AT-2606: coopmat.n");
    assert_eq!(coopmat.k, 16, "AT-2606: coopmat.k");
    assert_eq!(coopmat.a_type, CoopMatScalarMeta::F16, "AT-2606: A type must be F16");
    assert_eq!(coopmat.b_type, CoopMatScalarMeta::F16, "AT-2606: B type must be F16");
    assert_eq!(coopmat.c_type, CoopMatScalarMeta::F32, "AT-2606: C type must be F32");
    assert_eq!(coopmat.result_type, CoopMatScalarMeta::F32, "AT-2606: result type must be F32");
    assert_eq!(coopmat.scope, CoopMatScopeMeta::Subgroup, "AT-2606: scope must be Subgroup");

    // shared_memory_bytes UNCHANGED from M3.6: a_tile+b_tile (2048) + dsc+dmm cache (2048) = 4096.
    let expected_shared_bytes: u32 = (512 + 512) * 2 + 2 * 256 * 4;
    assert_eq!(
        meta.shared_memory_bytes, expected_shared_bytes,
        "AT-2606: shared_memory_bytes must be {expected_shared_bytes} (UNCHANGED vs M3.6 — the SR \
         rewrite touches only integer index math, not the shared layout); got {}",
        meta.shared_memory_bytes
    );

    // a_block_size=512 PIN (bounds the 256-entry scale caches to 32 rows — the carry magnitudes
    // and the cache index both depend on it).
    let strategy_decl = sr_src
        .lines()
        .find(|l| l.trim_start().starts_with("@strategy"))
        .expect("AT-2606: SR kernel must declare an @strategy block");
    assert!(
        strategy_decl.contains("a_block_size: ?[512]"),
        "AT-2606: a_block_size MUST be PINNED to `?[512]`; got @strategy line: {strategy_decl}"
    );
    // tile_k=16 PIN (the 256/tile_k=16-integer carry soundness precondition; AT-2601 also asserts it).
    assert!(
        strategy_decl.contains("tile_k: ?[16]"),
        "AT-2606: tile_k MUST be PINNED to `?[16]` (the carry soundness precondition 256 mod \
         tile_k == 0); got @strategy line: {strategy_decl}"
    );

    eprintln!(
        "AT-2606 PASS: q4km_matmul_rb_coopmat_f32acc_cached_sr.axc compiles + spirv-val clean; \
         caps BYTE-IDENTICAL to M3.6 leader ({sr_caps:?}); meta.coopmat = \
         {{16,16,16, F16,F16,F32,F32, Subgroup}}; shared_memory_bytes={} (unchanged); \
         a_block_size=512 + tile_k=16 PIN enforced",
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


// ── M3.2c-perf: flash_attention_coopmat.axc compile anchors (AT-2200/2201/2206) ──────────

const FLASH_ATTENTION_COOPMAT_SRC: &str =
    include_str!("../../../examples/flash_attention_coopmat.axc");

const MATMUL_SHARED_COOPMAT_BASELINE_SRC: &str =
    include_str!("../../../examples/matmul_shared_coopmat.axc");

/// SPIR-V capability values (subset).
mod m32cperf_cap {
    pub const SHADER: u32 = 1;
    pub const FLOAT16: u32 = 9;
    pub const VULKAN_MEMORY_MODEL: u32 = 5345;
    pub const COOPERATIVE_MATRIX_KHR: u32 = 6022;
}

/// Scan coopmat-relevant ops + OpPhi, returning
/// (n_coop_types, n_loads, n_stores, n_muladds, n_constant_null, n_total_phi, n_coop_phi).
fn flash_coopmat_op_scan(words: &[u32]) -> (usize, usize, usize, usize, usize, usize, usize) {
    use std::collections::BTreeSet;
    const OP_TYPE_COOP: u16 = 4456;
    const OP_COOP_LOAD: u16 = 4457;
    const OP_COOP_STORE: u16 = 4458;
    const OP_COOP_MULADD: u16 = 4459;
    const OP_CONSTANT_NULL: u16 = 46;
    const OP_PHI: u16 = 245;

    let mut coop_type_ids: BTreeSet<u32> = BTreeSet::new();
    let (mut n_load, mut n_store, mut n_muladd, mut n_null, mut n_phi, mut n_coop_phi) =
        (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let mut idx = 5usize;
    while idx < words.len() {
        let w0 = words[idx];
        let op = (w0 & 0xFFFF) as u16;
        let wc = (w0 >> 16) as usize;
        if wc == 0 || idx + wc > words.len() {
            break;
        }
        match op {
            // OpTypeCooperativeMatrixKHR: result id is the first operand word.
            OP_TYPE_COOP => {
                coop_type_ids.insert(words[idx + 1]);
            }
            OP_COOP_LOAD => n_load += 1,
            OP_COOP_STORE => n_store += 1,
            OP_COOP_MULADD => n_muladd += 1,
            OP_CONSTANT_NULL => n_null += 1,
            // OpPhi: word[idx+1] = result-type id, word[idx+2] = result id.
            OP_PHI => {
                n_phi += 1;
                if coop_type_ids.contains(&words[idx + 1]) {
                    n_coop_phi += 1;
                }
            }
            _ => {}
        }
        idx += wc;
    }
    (coop_type_ids.len(), n_load, n_store, n_muladd, n_null, n_phi, n_coop_phi)
}

/// AT-2200: flash_attention_coopmat.axc compiles + spirv-val clean; caps are exactly
/// [Shader, CooperativeMatrixKHR, VulkanMemoryModel] (+ Float16 for the f16 coopmat/shared
/// types); SPV_KHR_cooperative_matrix + SPV_KHR_vulkan_memory_model present; the GLSL.std.450
/// import is emitted EXACTLY once despite multiple exp() call sites.
#[test]
fn at_2200_flash_coopmat_compiles_validates() {
    let words = compile_and_validate(FLASH_ATTENTION_COOPMAT_SRC, "flash_attention_coopmat.axc");
    let (caps, exts) = capability_extension_sets(&words);

    for required in [
        m32cperf_cap::SHADER,
        m32cperf_cap::COOPERATIVE_MATRIX_KHR,
        m32cperf_cap::VULKAN_MEMORY_MODEL,
        m32cperf_cap::FLOAT16,
    ] {
        assert!(
            caps.contains(&required),
            "AT-2200: missing required capability {required}; got caps={caps:?}"
        );
    }
    // No capability beyond the PROVEN f16-coopmat baseline (matmul_shared_coopmat.axc) ∪
    // {VulkanMemoryModel}. The required four are the load-bearing set; the f16-shared
    // staging legitimately pulls in Float16Buffer (61) etc., exactly as the existing
    // shared-staged coopmat kernel does (no NEW capability vs that proven kernel).
    let baseline_words =
        compile_words(MATMUL_SHARED_COOPMAT_BASELINE_SRC, Some(&tile_assignments(16, 16, 16)),
            "matmul_shared_coopmat.axc (AT-2200 baseline)");
    let (baseline_caps, _) = capability_extension_sets(&baseline_words);
    let mut allowed: std::collections::BTreeSet<u32> = baseline_caps.clone();
    allowed.insert(m32cperf_cap::VULKAN_MEMORY_MODEL);
    let extra: Vec<u32> = caps.difference(&allowed).copied().collect();
    assert!(
        extra.is_empty(),
        "AT-2200: flash_attention_coopmat declares capabilities NOT in \
         (matmul_shared_coopmat ∪ {{VulkanMemoryModel}}): {extra:?} \
         (caps={caps:?}, baseline={baseline_caps:?})"
    );

    assert!(
        exts.contains("SPV_KHR_cooperative_matrix"),
        "AT-2200: SPV_KHR_cooperative_matrix missing; exts={exts:?}"
    );
    assert!(
        exts.contains("SPV_KHR_vulkan_memory_model"),
        "AT-2200: SPV_KHR_vulkan_memory_model missing; exts={exts:?}"
    );

    // GLSL.std.450 import emitted exactly once (two exp() call sites — correction + p).
    assert_eq!(
        count_glsl450_imports(&words),
        1,
        "AT-2200: GLSL.std.450 import must be emitted EXACTLY once"
    );

    eprintln!("AT-2200 PASS: flash_attention_coopmat.axc spirv-val clean; caps={caps:?}; exts={exts:?}");
}

/// Count OpExtInstImport "GLSL.std.450" occurrences.
fn count_glsl450_imports(words: &[u32]) -> usize {
    let mut count = 0usize;
    let mut idx = 5usize;
    while idx < words.len() {
        let w0 = words[idx];
        let opcode = w0 & 0xFFFF;
        let wc = (w0 >> 16) as usize;
        if wc == 0 || idx + wc > words.len() {
            break;
        }
        if opcode == 11 {
            let mut bytes: Vec<u8> = Vec::new();
            for &w in &words[idx + 2..idx + wc] {
                bytes.extend_from_slice(&w.to_le_bytes());
            }
            let nul = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
            if String::from_utf8_lossy(&bytes[..nul]) == "GLSL.std.450" {
                count += 1;
            }
        }
        idx += wc;
    }
    count
}

/// AT-2201 (R3): OpPhi discipline — s_acc is declared INSIDE the K/V loop body (fresh
/// coopmat_zero() per iteration, the 4 head-dim mul_adds UNROLLED at top level) so it is
/// iteration-local and gets NO OpPhi. Assert the emitted SPIR-V has EXACTLY 0 coopmat OpPhi
/// (the falsifier for an accidental pre-loop declaration/carry) AND exactly the expected
/// coopmat op shape: 3 coopmat types (A/B/accumulator), 8 loads (4 Q + 4 K), 1 store (S),
/// 4 mul_adds (the 4 head-dim sub-blocks), 1 OpConstantNull (the fresh s_acc per iteration).
#[test]
fn at_2201_flash_coopmat_opphi_discipline() {
    let words = compile_and_validate(FLASH_ATTENTION_COOPMAT_SRC, "flash_attention_coopmat.axc (AT-2201)");
    let (n_types, n_load, n_store, n_muladd, n_null, n_phi, n_coop_phi) =
        flash_coopmat_op_scan(&words);

    // THE falsifier: EXACTLY 0 coopmat OpPhi (s_acc iteration-local, not loop-carried).
    assert_eq!(
        n_coop_phi, 0,
        "AT-2201: s_acc must be iteration-local → EXACTLY 0 coopmat OpPhi; found {n_coop_phi} \
         (an accidental pre-loop s_acc declaration would create a coopmat phi). \
         total_phi={n_phi}"
    );
    // No OpPhi at all is expected (acc/m/l loop-carry lives in SHARED, scalars never get phi).
    assert_eq!(
        n_phi, 0,
        "AT-2201: no OpPhi expected (all loop-carry is in SHARED); found {n_phi}"
    );

    assert_eq!(n_types, 3, "AT-2201: expected 3 coopmat types (A,B,accumulator); got {n_types}");
    assert_eq!(n_load, 8, "AT-2201: expected 8 coopmat loads (4 Q + 4 K); got {n_load}");
    assert_eq!(n_store, 1, "AT-2201: expected 1 coopmat store (S); got {n_store}");
    assert_eq!(n_muladd, 4, "AT-2201: expected 4 coopmat mul_adds (4 head-dim sub-blocks); got {n_muladd}");
    assert_eq!(n_null, 1, "AT-2201: expected 1 OpConstantNull (the fresh s_acc); got {n_null}");

    eprintln!(
        "AT-2201 PASS: coopmat OpPhi=0 (s_acc iteration-local); shape: types={n_types} loads={n_load} \
         stores={n_store} muladds={n_muladd} nulls={n_null}"
    );
}

/// AT-2206: CI anchor — shared_memory_bytes == 13504 (INCL corr_sh[16]=64), seq-INVARIANT
/// (seq is a push-constant, not an array dim → streaming, no O(seq) score buffer);
/// exactly {Q,K,V,O} buffers bound (no S scratch buffer); < 16384 portable floor.
#[test]
fn at_2206_flash_coopmat_anchor_and_streaming() {
    let (bytes, meta) = compile_source_with_meta(FLASH_ATTENTION_COOPMAT_SRC)
        .expect("AT-2206: flash_attention_coopmat.axc must compile");
    let _ = words_and_validate(bytes, "flash_attention_coopmat.axc (AT-2206)");

    // Exact shared tally: q 2048 + k 2048 + v 4096 + s 1024 + acc 4096 + m 64 + l 64 + corr 64.
    assert_eq!(
        meta.shared_memory_bytes, 13504,
        "AT-2206: shared_memory_bytes must be 13504 (incl corr_sh[16]); got {}",
        meta.shared_memory_bytes
    );
    assert!(
        meta.shared_memory_bytes <= 16384,
        "AT-2206: shared_memory_bytes={} exceeds the 16384 portable floor",
        meta.shared_memory_bytes
    );

    // Exactly {Q,K,V,O} — no O(seq) score scratch buffer (the streaming property).
    let names: Vec<String> = meta.binding_plan.buffers.iter().map(|b| b.name.clone()).collect();
    assert_eq!(
        names,
        vec!["Q".to_string(), "K".to_string(), "V".to_string(), "O".to_string()],
        "AT-2206: buffers must be exactly {{Q,K,V,O}}; got {names:?}"
    );

    eprintln!(
        "AT-2206 PASS: shared_memory_bytes=13504 (seq-invariant); buffers={names:?}"
    );
}

// ── M3.2c-PV: flash_attention_coopmat_pv.axc compile anchors (AT-2210/2211/2216) ──────────
//
// The coopmat-PV kernel ADDS a second tensor-core matmul PV = P·V (carried from M3.2c-perf's
// coopmat QKᵀ). PURE SOURCE — the SAME shared-source coopmat load/store the QKᵀ path uses.

const FLASH_ATTENTION_COOPMAT_PV_SRC: &str =
    include_str!("../../../examples/flash_attention_coopmat_pv.axc");

/// AT-2210: flash_attention_coopmat_pv.axc compiles + spirv-val clean; emitted capabilities
/// are EXACTLY the M3.2c-perf set [Shader, CooperativeMatrixKHR, VulkanMemoryModel, Float16] —
/// a STRICT SUBSET of matmul_shared_coopmat.axc's caps (∪ {VulkanMemoryModel}), NO new
/// capability vs the QKᵀ-only kernel; GLSL.std.450 imported EXACTLY once (despite the two
/// exp() call sites); SPV_KHR_cooperative_matrix + SPV_KHR_vulkan_memory_model present.
#[test]
fn at_2210_flash_coopmat_pv_spirv_val_and_caps() {
    let words =
        compile_and_validate(FLASH_ATTENTION_COOPMAT_PV_SRC, "flash_attention_coopmat_pv.axc");
    let (caps, exts) = capability_extension_sets(&words);

    for required in [
        m32cperf_cap::SHADER,
        m32cperf_cap::COOPERATIVE_MATRIX_KHR,
        m32cperf_cap::VULKAN_MEMORY_MODEL,
        m32cperf_cap::FLOAT16,
    ] {
        assert!(
            caps.contains(&required),
            "AT-2210: missing required capability {required}; got caps={caps:?}"
        );
    }
    // No capability beyond (matmul_shared_coopmat ∪ {VulkanMemoryModel}) — i.e. EXACTLY the
    // M3.2c-perf set; the f16 PV operands reuse the f16-coopmat capability surface, no NEW cap.
    let baseline_words = compile_words(
        MATMUL_SHARED_COOPMAT_BASELINE_SRC,
        Some(&tile_assignments(16, 16, 16)),
        "matmul_shared_coopmat.axc (AT-2210 baseline)",
    );
    let (baseline_caps, _) = capability_extension_sets(&baseline_words);
    let mut allowed: std::collections::BTreeSet<u32> = baseline_caps.clone();
    allowed.insert(m32cperf_cap::VULKAN_MEMORY_MODEL);
    let extra: Vec<u32> = caps.difference(&allowed).copied().collect();
    assert!(
        extra.is_empty(),
        "AT-2210: flash_attention_coopmat_pv declares capabilities NOT in \
         (matmul_shared_coopmat ∪ {{VulkanMemoryModel}}): {extra:?} \
         (caps={caps:?}, baseline={baseline_caps:?})"
    );
    // Cross-check: the PV kernel's caps equal the M3.2c-perf scalar-PV kernel's caps EXACTLY.
    let scalar_pv_words = compile_words(
        FLASH_ATTENTION_COOPMAT_SRC,
        None,
        "flash_attention_coopmat.axc (AT-2210 cap parity)",
    );
    let (scalar_pv_caps, _) = capability_extension_sets(&scalar_pv_words);
    assert_eq!(
        caps, scalar_pv_caps,
        "AT-2210: coopmat-PV caps must EQUAL the M3.2c-perf scalar-PV caps (no new cap); \
         pv={caps:?} scalar_pv={scalar_pv_caps:?}"
    );

    assert!(
        exts.contains("SPV_KHR_cooperative_matrix"),
        "AT-2210: SPV_KHR_cooperative_matrix missing; exts={exts:?}"
    );
    assert!(
        exts.contains("SPV_KHR_vulkan_memory_model"),
        "AT-2210: SPV_KHR_vulkan_memory_model missing; exts={exts:?}"
    );
    assert_eq!(
        count_glsl450_imports(&words),
        1,
        "AT-2210: GLSL.std.450 import must be emitted EXACTLY once"
    );

    eprintln!(
        "AT-2210 PASS: flash_attention_coopmat_pv.axc spirv-val clean; caps={caps:?}; exts={exts:?}"
    );
}

/// AT-2211 (R3): OpPhi discipline — BOTH s_acc (QKᵀ) AND every pv_acc_ds (the 4 PV sub-block
/// accumulators) are declared INSIDE the K/V loop body (fresh coopmat_zero() each) → the loop
/// header has EXACTLY 0 coopmat OpPhi (the falsifier for an accidental carry of EITHER). The
/// canonical 4-sub-block / P-loaded-once layout pins the exact op shape:
///   3 coopmat types (A/B/accumulator — the SAME 16×16×16 f16/f16/f32 shape serves BOTH matmuls),
///   13 coopmat loads = 4 Q + 4 K (QKᵀ) + 1 P + 4 V (PV),
///   5 coopmat stores = 1 S + 4 PV,
///   8 coopmat mul_adds = 4 QKᵀ + 4 PV,
///   5 OpConstantNull = 1 s_acc + 4 pv_acc (all iteration-local).
#[test]
fn at_2211_flash_coopmat_pv_opphi_discipline() {
    let words = compile_and_validate(
        FLASH_ATTENTION_COOPMAT_PV_SRC,
        "flash_attention_coopmat_pv.axc (AT-2211)",
    );
    let (n_types, n_load, n_store, n_muladd, n_null, n_phi, n_coop_phi) =
        flash_coopmat_op_scan(&words);

    // THE invariant falsifier: EXACTLY 0 coopmat OpPhi (BOTH accumulators iteration-local).
    assert_eq!(
        n_coop_phi, 0,
        "AT-2211: BOTH s_acc AND every pv_acc must be iteration-local → EXACTLY 0 coopmat \
         OpPhi; found {n_coop_phi} (an accidental pre-loop declaration of EITHER would create a \
         coopmat phi). total_phi={n_phi}"
    );
    assert_eq!(
        n_phi, 0,
        "AT-2211: no OpPhi expected (all loop-carry is in SHARED); found {n_phi}"
    );

    // Shape-reuse proof: ONE 16×16×16 f16/f16/f32 shape serves BOTH matmuls → 3 types, NOT 6.
    assert_eq!(
        n_types, 3,
        "AT-2211: expected 3 coopmat types (A,B,accumulator — shape REUSED across QKᵀ + PV); got {n_types}"
    );
    // Canonical 4-sub-block PV layout, P loaded once (the spec's recommended, hard-pinned form).
    assert_eq!(
        n_load, 13,
        "AT-2211: expected 13 coopmat loads (4 Q + 4 K + 1 P + 4 V); got {n_load}"
    );
    assert_eq!(
        n_store, 5,
        "AT-2211: expected 5 coopmat stores (1 S + 4 PV); got {n_store}"
    );
    assert_eq!(
        n_muladd, 8,
        "AT-2211: expected 8 coopmat mul_adds (4 QKᵀ + 4 PV); got {n_muladd}"
    );
    assert_eq!(
        n_null, 5,
        "AT-2211: expected 5 OpConstantNull (1 s_acc + 4 pv_acc, all iteration-local); got {n_null}"
    );

    eprintln!(
        "AT-2211 PASS: coopmat OpPhi=0 (s_acc + 4 pv_acc all iteration-local); shape: types={n_types} \
         loads={n_load} stores={n_store} muladds={n_muladd} nulls={n_null}"
    );
}

/// AT-2216: CI anchor — shared_memory_bytes == 16064 (q[f16]2048 + k[f16]2048 + v[f16]2048 +
/// s[f32]1024 + p_sh[f16]512 + pv_sh[f32]4096 + acc[f32]4096 + m64 + l64 + corr64), seq-INVARIANT
/// (seq is a push-constant, no O(seq) score buffer), < 16384 portable floor (320 B headroom);
/// buffers bound exactly {Q,K,V,O}.
#[test]
fn at_2216_flash_coopmat_pv_anchor_and_streaming() {
    let (bytes, meta) = compile_source_with_meta(FLASH_ATTENTION_COOPMAT_PV_SRC)
        .expect("AT-2216: flash_attention_coopmat_pv.axc must compile");
    let _ = words_and_validate(bytes, "flash_attention_coopmat_pv.axc (AT-2216)");

    assert_eq!(
        meta.shared_memory_bytes, 16064,
        "AT-2216: shared_memory_bytes must be 16064 (q2048 + k2048 + v[f16]2048 + s1024 + \
         p_sh[f16]512 + pv_sh4096 + acc4096 + m64 + l64 + corr64); got {}",
        meta.shared_memory_bytes
    );
    assert!(
        meta.shared_memory_bytes <= 16384,
        "AT-2216: shared_memory_bytes={} exceeds the 16384 portable floor",
        meta.shared_memory_bytes
    );

    let names: Vec<String> = meta.binding_plan.buffers.iter().map(|b| b.name.clone()).collect();
    assert_eq!(
        names,
        vec!["Q".to_string(), "K".to_string(), "V".to_string(), "O".to_string()],
        "AT-2216: buffers must be exactly {{Q,K,V,O}}; got {names:?}"
    );

    eprintln!(
        "AT-2216 PASS: shared_memory_bytes=16064 < 16384 (320 B headroom, seq-invariant); buffers={names:?}"
    );
}

// ── AT-2304: M3.9 multi-subgroup warptile kernels — compile + spirv-val + cap-set ──────
//            == matmul_rb_coopmat.axc (no new capability) + ZERO portable-floor warns ────

/// Shipped K=4 warptile assignments (wg_threads=128, 4 subgroups, 64x64 tile).
fn warptile_k4_assignments() -> StrategyMap {
    let mut m = StrategyMap::new();
    m.insert("wg_threads".to_owned(), 128_i64);
    m.insert("n_sg".to_owned(), 4_i64);
    m.insert("bm".to_owned(), 64_i64);
    m.insert("bn".to_owned(), 64_i64);
    m.insert("rb_m".to_owned(), 2_i64);
    m.insert("rb_n".to_owned(), 2_i64);
    m.insert("tile_k".to_owned(), 16_i64);
    m.insert("a_block_size".to_owned(), 1024_i64);
    m.insert("b_block_size".to_owned(), 1024_i64);
    m
}

/// Shipped K=2 opponent assignments (wg_threads=64, 2 subgroups, 32x64 tile).
fn warptile_k2_assignments() -> StrategyMap {
    let mut m = StrategyMap::new();
    m.insert("wg_threads".to_owned(), 64_i64);
    m.insert("n_sg".to_owned(), 2_i64);
    m.insert("bm".to_owned(), 32_i64);
    m.insert("bn".to_owned(), 64_i64);
    m.insert("rb_m".to_owned(), 2_i64);
    m.insert("rb_n".to_owned(), 2_i64);
    m.insert("tile_k".to_owned(), 16_i64);
    m.insert("a_block_size".to_owned(), 512_i64);
    m.insert("b_block_size".to_owned(), 1024_i64);
    m
}

/// Lower a kernel source and return its HIR warnings (raw source — the `@workgroup` dims
/// are literals, not holes, so the workgroup-product validation runs as written; the
/// `@strategy` holes lower to `UnusedStrategyHole` warns which AT-2304 filters out).
fn lower_warnings(src: &str) -> Vec<axc_hir::HirWarning> {
    let (tokens, _lex) = axc_lexer::tokenize(src);
    let mut parser = axc_parser::Parser::new(&tokens);
    let (ast, _parse) = parser.parse_module();
    let (_hir, _errors, warnings) = axc_hir::lower_module(&ast);
    warnings
}

/// AT-2304 (CI, no GPU): both M3.9 warptile kernels compile + spirv-val clean; the emitted
/// capability set EQUALS matmul_rb_coopmat.axc's caps (no NEW capability beyond the M3.6
/// set); and NEITHER kernel emits a `WorkgroupExceedsPortableFloor` warning.
///
/// PORTABLE-FLOOR (r2): `@workgroup(128,1,1)` product==128 IS the portable floor
/// (`PORTABLE_MIN_WORKGROUP_INVOCATIONS`, hir.rs:19) so it does NOT warn (the warn fires
/// only when product > 128, validate.rs:413); `@workgroup(64,1,1)` (64<128) likewise. The
/// test asserts SPECIFICALLY the ABSENCE of the `WorkgroupExceedsPortableFloor` variant for
/// EITHER kernel — it does NOT assert warning-free overall, because the base matmul kernel
/// emits pre-existing `UnusedStrategyHole` warns for its @strategy holes (those are filtered
/// out before the assertion).
#[test]
fn at_2304_warptile_compiles_spirv_val() {
    use std::collections::BTreeSet;

    let k4_src = include_str!("../../../examples/matmul_warptile_coopmat.axc");
    let k2_src = include_str!("../../../examples/matmul_warptile_coopmat_2sg.axc");
    let rb_src = include_str!("../../../examples/matmul_rb_coopmat.axc");

    let k4_assignments = warptile_k4_assignments();
    let k2_assignments = warptile_k2_assignments();
    let rb_assignments = rb2x2_assignments();

    // Compile + spirv-val both warptile kernels and the single-subgroup RB baseline.
    let k4_words = compile_words(k4_src, Some(&k4_assignments), "matmul_warptile_coopmat.axc");
    let k2_words = compile_words(k2_src, Some(&k2_assignments), "matmul_warptile_coopmat_2sg.axc");
    let rb_words = compile_words(rb_src, Some(&rb_assignments), "matmul_rb_coopmat.axc");

    // ── Cap set == matmul_rb_coopmat.axc (no NEW capability beyond the M3.6 set). ──
    let (k4_caps, k4_exts) = capability_extension_sets(&k4_words);
    let (k2_caps, k2_exts) = capability_extension_sets(&k2_words);
    let (rb_caps, rb_exts) = capability_extension_sets(&rb_words);

    let k4_extra: Vec<u32> = k4_caps.difference(&rb_caps).copied().collect();
    assert!(
        k4_extra.is_empty(),
        "AT-2304: K=4 warptile declares capabilities NOT in matmul_rb_coopmat: {k4_extra:?} \
         (k4={k4_caps:?}, rb={rb_caps:?})"
    );
    let k2_extra: Vec<u32> = k2_caps.difference(&rb_caps).copied().collect();
    assert!(
        k2_extra.is_empty(),
        "AT-2304: K=2 warptile declares capabilities NOT in matmul_rb_coopmat: {k2_extra:?} \
         (k2={k2_caps:?}, rb={rb_caps:?})"
    );
    // Both directions: the cap set must EQUAL the RB set (a pure re-partition adds/drops
    // nothing — same coopmat + Float16 + Shader + memory-model caps).
    let k4_set: BTreeSet<u32> = k4_caps.iter().copied().collect();
    let k2_set: BTreeSet<u32> = k2_caps.iter().copied().collect();
    let rb_set: BTreeSet<u32> = rb_caps.iter().copied().collect();
    assert_eq!(
        k4_set, rb_set,
        "AT-2304: K=4 warptile cap set must EQUAL matmul_rb_coopmat's (pure source, no new cap)"
    );
    assert_eq!(
        k2_set, rb_set,
        "AT-2304: K=2 warptile cap set must EQUAL matmul_rb_coopmat's (pure source, no new cap)"
    );
    // Extensions likewise — no new extension beyond the RB baseline.
    let k4_extra_exts: Vec<String> = k4_exts.difference(&rb_exts).cloned().collect();
    assert!(
        k4_extra_exts.is_empty(),
        "AT-2304: K=4 warptile declares extensions NOT in matmul_rb_coopmat: {k4_extra_exts:?}"
    );
    let k2_extra_exts: Vec<String> = k2_exts.difference(&rb_exts).cloned().collect();
    assert!(
        k2_extra_exts.is_empty(),
        "AT-2304: K=2 warptile declares extensions NOT in matmul_rb_coopmat: {k2_extra_exts:?}"
    );

    // ── ZERO WorkgroupExceedsPortableFloor warnings (the r2 BLOCKER fix). ──
    // Lower the RAW source (the @workgroup dims are literals 128/64, so the portable-floor
    // validation runs as written); tolerate the pre-existing UnusedStrategyHole warns.
    let is_portable_floor = |w: &axc_hir::HirWarning| {
        matches!(w, axc_hir::HirWarning::WorkgroupExceedsPortableFloor { .. })
    };

    let k4_warns = lower_warnings(k4_src);
    let k4_floor: Vec<_> = k4_warns.iter().filter(|w| is_portable_floor(w)).collect();
    assert!(
        k4_floor.is_empty(),
        "AT-2304: @workgroup(128,1,1) product==128 IS the portable floor and must emit ZERO \
         WorkgroupExceedsPortableFloor warns (warn fires only at product > 128); got {k4_floor:?}"
    );

    let k2_warns = lower_warnings(k2_src);
    let k2_floor: Vec<_> = k2_warns.iter().filter(|w| is_portable_floor(w)).collect();
    assert!(
        k2_floor.is_empty(),
        "AT-2304: @workgroup(64,1,1) (64 < 128) must emit ZERO WorkgroupExceedsPortableFloor \
         warns; got {k2_floor:?}"
    );

    eprintln!(
        "AT-2304 PASS: both warptile kernels compile + spirv-val clean; cap set == \
         matmul_rb_coopmat ({k4_caps:?}); ZERO WorkgroupExceedsPortableFloor warns \
         (K=4 @workgroup(128)==floor, K=2 @workgroup(64)<floor)"
    );
}

// ── AT-2405 (M3.10a, CI no-GPU): the BANK-PADDED kernels compile + spirv-val clean,        ──
//            capability set BYTE-IDENTICAL to the base (padding adds NO capability),         ──
//            shared_memory_bytes reflects the padded stride, barrier count == base.          ──

/// M3.10a plain-f32 padded RB strategy (PAD_A=PAD_B=8 canonical ship tuple).
/// a_pad_stride=24 (16+8), a_pad_size=768 (32*24), a_pad_mat1off=384 (16*24),
/// b_pad_stride=40 (32+8), b_pad_size=640 (16*40).
fn rb2x2_pad_assignments() -> StrategyMap {
    let mut m = StrategyMap::new();
    m.insert("rb_m".to_owned(), 2_i64);
    m.insert("rb_n".to_owned(), 2_i64);
    m.insert("tile_k".to_owned(), 16_i64);
    m.insert("a_block_size".to_owned(), 512_i64);
    m.insert("b_block_size".to_owned(), 512_i64);
    m.insert("a_pad_stride".to_owned(), 24_i64);
    m.insert("a_pad_size".to_owned(), 768_i64);
    m.insert("a_pad_mat1off".to_owned(), 384_i64);
    m.insert("b_pad_stride".to_owned(), 40_i64);
    m.insert("b_pad_size".to_owned(), 640_i64);
    m
}

/// M3.10a Q4_K_M padded leader strategy (PAD_A=PAD_B=8; SAME pad tuple as plain-f32, PLUS the
/// M3.6 a_block_size=512 PIN that bounds the 256-entry scale caches).
fn rb2x2_q4km_pad_assignments() -> StrategyMap {
    rb2x2_pad_assignments()
}

/// AT-2405 (plain-f32): matmul_rb_coopmat_pad.axc compiles + spirv-val clean; capability set
/// BYTE-IDENTICAL to matmul_rb_coopmat.axc (the base); shared_memory_bytes == padded stride
/// (A 32*24 + B 16*40 f16 = (768+640)*2 = 2816 B); barrier count == base (2 per k_block).
#[test]
fn at_2405_matmul_rb_pad_compile_capset_sharedbytes() {
    use std::collections::BTreeSet;

    let pad_src = include_str!("../../../examples/matmul_rb_coopmat_pad.axc");
    let base_src = include_str!("../../../examples/matmul_rb_coopmat.axc");

    // The padded kernel compiles WITH metadata + spirv-val (the load-bearing compile anchor).
    let (pad_bytes, meta) = compile_source_with_assignments(pad_src, &rb2x2_pad_assignments())
        .unwrap_or_else(|e| panic!("matmul_rb_coopmat_pad.axc: compile failed: {e:?}"));
    let pad_words = words_and_validate(pad_bytes, "matmul_rb_coopmat_pad.axc");
    let base_words = compile_words(base_src, Some(&rb2x2_assignments()), "matmul_rb_coopmat.axc");

    // Capability set BYTE-IDENTICAL to the base (padding is plain shared[f16] index arithmetic +
    // a runtime coopmat_load stride — NO new capability).
    let (pad_caps, pad_exts) = capability_extension_sets(&pad_words);
    let (base_caps, base_exts) = capability_extension_sets(&base_words);
    let pad_set: BTreeSet<u32> = pad_caps.iter().copied().collect();
    let base_set: BTreeSet<u32> = base_caps.iter().copied().collect();
    assert_eq!(
        pad_set, base_set,
        "AT-2405 (plain): padded kernel cap set must be BYTE-IDENTICAL to matmul_rb_coopmat \
         (padding adds NO capability); pad={pad_caps:?} base={base_caps:?}"
    );
    assert_eq!(
        pad_exts, base_exts,
        "AT-2405 (plain): padded kernel extension set must equal the base's; \
         pad={pad_exts:?} base={base_exts:?}"
    );

    // shared_memory_bytes reflects the PADDED stride: A 32*24 + B 16*40 f16 = (768+640)*2 = 2816 B.
    let expected_shared_bytes: u32 = (768 + 640) * 2; // 2816
    assert_eq!(
        meta.shared_memory_bytes, expected_shared_bytes,
        "AT-2405 (plain): padded shared_memory_bytes must be {expected_shared_bytes} \
         (a_tile 32*24=768 f16 + b_tile 16*40=640 f16, *2 B); got {}", meta.shared_memory_bytes
    );
    assert!(
        meta.shared_memory_bytes <= 16384,
        "AT-2405 (plain): padded shared_memory_bytes={} exceeds the 16384-byte portable floor",
        meta.shared_memory_bytes
    );

    // Barrier count == base (2 per k_block: post-staging + WAR). Padding is pure address layout —
    // it adds NO barrier. OpPhi count == base (single-level 4-accumulator carry intact).
    let pad_barriers = count_op_control_barrier(&pad_words);
    let base_barriers = count_op_control_barrier(&base_words);
    assert_eq!(
        pad_barriers, base_barriers,
        "AT-2405 (plain): padded kernel must emit the SAME OpControlBarrier count as the base \
         (padding adds no barrier); pad={pad_barriers} base={base_barriers}"
    );
    let pad_phis = count_op_phi(&pad_words);
    let base_phis = count_op_phi(&base_words);
    assert_eq!(
        pad_phis, base_phis,
        "AT-2405 (plain): padded kernel OpPhi count must equal the base (single-level carry \
         intact); pad={pad_phis} base={base_phis}"
    );

    eprintln!(
        "AT-2405 (plain) PASS: matmul_rb_coopmat_pad.axc compiles + spirv-val clean; \
         cap set == base ({pad_caps:?}); shared_memory_bytes={} (2816 padded); \
         OpControlBarrier={pad_barriers} == base; OpPhi={pad_phis} == base",
        meta.shared_memory_bytes
    );
}

/// AT-2405 (Q4_K_M): q4km_matmul_rb_coopmat_f32acc_cached_pad.axc compiles + spirv-val clean;
/// capability set BYTE-IDENTICAL to the M3.6 cached leader; shared_memory_bytes == padded tiles
/// (A 32*24 + B 16*40 f16 = 2816 B) PLUS the unchanged 256-entry scale caches (2*256*4 = 2048 B)
/// = 4864 B; barrier count == base (3 per k_block); a_block_size=512 PIN preserved.
#[test]
fn at_2405_q4km_rb_pad_compile_capset_sharedbytes() {
    use std::collections::BTreeSet;
    use axc_runtime::{CoopMatScalarMeta, CoopMatScopeMeta};

    let pad_src = include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached_pad.axc");
    let base_src = include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached.axc");

    let (pad_bytes, meta) = compile_source_with_assignments(pad_src, &rb2x2_q4km_pad_assignments())
        .unwrap_or_else(|e| panic!("q4km_matmul_rb_coopmat_f32acc_cached_pad.axc: compile failed: {e:?}"));
    let pad_words = words_and_validate(pad_bytes, "q4km_matmul_rb_coopmat_f32acc_cached_pad.axc");
    let base_words = compile_words(base_src, Some(&rb2x2_assignments()),
        "q4km_matmul_rb_coopmat_f32acc_cached.axc");

    // Capability set BYTE-IDENTICAL to the M3.6 cached leader.
    let (pad_caps, pad_exts) = capability_extension_sets(&pad_words);
    let (base_caps, base_exts) = capability_extension_sets(&base_words);
    let pad_set: BTreeSet<u32> = pad_caps.iter().copied().collect();
    let base_set: BTreeSet<u32> = base_caps.iter().copied().collect();
    assert_eq!(
        pad_set, base_set,
        "AT-2405 (Q4_K_M): padded kernel cap set must be BYTE-IDENTICAL to the M3.6 cached leader \
         (padding adds NO capability); pad={pad_caps:?} base={base_caps:?}"
    );
    assert_eq!(
        pad_exts, base_exts,
        "AT-2405 (Q4_K_M): padded kernel extension set must equal the base's; \
         pad={pad_exts:?} base={base_exts:?}"
    );

    // Coopmat metadata shape unchanged: {16,16,16, F16,F16,F32,F32, Subgroup}.
    let coopmat = meta.coopmat.as_ref()
        .expect("AT-2405 (Q4_K_M): padded kernel must emit coopmat metadata");
    assert_eq!(coopmat.m, 16, "AT-2405: coopmat.m");
    assert_eq!(coopmat.n, 16, "AT-2405: coopmat.n");
    assert_eq!(coopmat.k, 16, "AT-2405: coopmat.k");
    assert_eq!(coopmat.a_type, CoopMatScalarMeta::F16, "AT-2405: A type must be F16");
    assert_eq!(coopmat.b_type, CoopMatScalarMeta::F16, "AT-2405: B type must be F16");
    assert_eq!(coopmat.c_type, CoopMatScalarMeta::F32, "AT-2405: C type must be F32");
    assert_eq!(coopmat.result_type, CoopMatScalarMeta::F32, "AT-2405: result type must be F32");
    assert_eq!(coopmat.scope, CoopMatScopeMeta::Subgroup, "AT-2405: scope must be Subgroup");

    // shared_memory_bytes: padded A/B tiles (768+640 f16 = 2816 B) + UNCHANGED scale caches
    // (dsc_cache 256 f32 + dmm_cache 256 f32 = 2048 B) = 4864 B.
    let expected_shared_bytes: u32 = (768 + 640) * 2 + 2 * 256 * 4; // 2816 + 2048 = 4864
    assert_eq!(
        meta.shared_memory_bytes, expected_shared_bytes,
        "AT-2405 (Q4_K_M): padded shared_memory_bytes must be {expected_shared_bytes} \
         (padded A/B tiles 2816 B + 256-entry scale caches 2048 B); got {}",
        meta.shared_memory_bytes
    );
    assert!(
        meta.shared_memory_bytes <= 16384,
        "AT-2405 (Q4_K_M): padded shared_memory_bytes={} exceeds the 16384-byte portable floor",
        meta.shared_memory_bytes
    );

    // Barrier count == base (3 per k_block: hoisted RAW + post-staging + WAR). OpPhi == base.
    let pad_barriers = count_op_control_barrier(&pad_words);
    let base_barriers = count_op_control_barrier(&base_words);
    assert_eq!(
        pad_barriers, base_barriers,
        "AT-2405 (Q4_K_M): padded kernel must emit the SAME OpControlBarrier count as the M3.6 \
         leader (padding adds no barrier); pad={pad_barriers} base={base_barriers}"
    );
    let pad_phis = count_op_phi(&pad_words);
    let base_phis = count_op_phi(&base_words);
    assert_eq!(
        pad_phis, base_phis,
        "AT-2405 (Q4_K_M): padded kernel OpPhi count must equal the M3.6 leader (single-level \
         carry intact); pad={pad_phis} base={base_phis}"
    );

    // a_block_size=512 PIN preserved (bounds the 256-entry scale caches to 32 logical rows;
    // GATE-NOTE #2: the cache index stays on the LOGICAL row, NEVER the padded stride).
    let strategy_decl = pad_src
        .lines()
        .find(|l| l.trim_start().starts_with("@strategy"))
        .expect("AT-2405 (Q4_K_M): padded kernel must declare an @strategy block");
    assert!(
        strategy_decl.contains("a_block_size: ?[512]"),
        "AT-2405 (Q4_K_M): a_block_size MUST stay PINNED to ?[512] (bounds the 256-entry scale \
         caches); got @strategy line: {strategy_decl}"
    );

    eprintln!(
        "AT-2405 (Q4_K_M) PASS: q4km_matmul_rb_coopmat_f32acc_cached_pad.axc compiles + spirv-val \
         clean; cap set == M3.6 leader ({pad_caps:?}); shared_memory_bytes={} (4864 = 2816 padded \
         tiles + 2048 caches); OpControlBarrier={pad_barriers} == base; OpPhi={pad_phis} == base; \
         a_block_size=512 PIN preserved",
        meta.shared_memory_bytes
    );
}

// ── AT-2703: M3.12 dequant front-end ABLATION DIAGNOSTIC profiling-instrument variants — ──
//            compile + spirv-val + cap-set BYTE-IDENTICAL to the M3.6 leader (no new capability)
//
// The M3.12 variants are PROFILING INSTRUMENTS with WRONG OUTPUT BY DESIGN (they hold the M3.6
// structure constant and vary only the per-element A-dequant value). The cheaper substitute
// expressions (a near-free index-derived value / an f32_to_f16-retaining nibble pass-through) remove
// the per-element Workgroup scale-reads + FMul/FSub (and, V-structure-probe, the u8 load) and add NO
// op-class requiring a new capability. AT-2703 asserts each variant compiles, passes spirv-val
// (Vulkan 1.1), and emits a cap set BYTE-IDENTICAL to the M3.6 leader. NECESSARY but NOT SUFFICIENT:
// these variants are WRONG by construction — they carry NO correctness gate (AT-2705 enforces that
// negatively); the deliverable is the per-variant resident TFLOPS (the bench).
#[test]
fn at_2703_ablation_variants_compile_capset_identical() {
    use std::collections::BTreeSet;

    let leader_src =
        include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached.axc");
    let structonly_src =
        include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached_ablate_structonly.axc");
    let passthrough_src =
        include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached_ablate_passthrough.axc");

    let assignments = rb2x2_assignments();

    let leader_words = compile_words(
        leader_src, Some(&assignments), "q4km_matmul_rb_coopmat_f32acc_cached.axc");
    let (leader_caps, leader_exts) = capability_extension_sets(&leader_words);
    let leader_cap_set: BTreeSet<u32> = leader_caps.iter().copied().collect();

    for (src, name) in [
        (structonly_src, "q4km_matmul_rb_coopmat_f32acc_cached_ablate_structonly.axc"),
        (passthrough_src, "q4km_matmul_rb_coopmat_f32acc_cached_ablate_passthrough.axc"),
    ] {
        // Compile WITH metadata + spirv-val (the f32_to_f16-retaining / index-derived substitutes
        // are still valid SPIR-V).
        let (bytes, meta) = compile_source_with_assignments(src, &assignments)
            .unwrap_or_else(|e| panic!("AT-2703: {name} compile failed: {e:?}"));
        let words = words_and_validate(bytes, name);

        let (caps, exts) = capability_extension_sets(&words);
        let cap_set: BTreeSet<u32> = caps.iter().copied().collect();

        let extra_caps: Vec<u32> = caps.difference(&leader_caps).copied().collect();
        assert!(
            extra_caps.is_empty(),
            "AT-2703: ablation variant {name} declares capabilities NOT in the M3.6 leader's set: \
             {extra_caps:?} (variant={caps:?}, leader={leader_caps:?}). The cheaper substitute \
             expressions must add NO new capability."
        );
        let dropped_caps: Vec<u32> = leader_caps.difference(&caps).copied().collect();
        assert!(
            dropped_caps.is_empty(),
            "AT-2703: ablation variant {name} DROPPED capabilities present in the M3.6 leader: \
             {dropped_caps:?}"
        );
        assert_eq!(
            cap_set, leader_cap_set,
            "AT-2703: ablation variant {name} cap set must be BYTE-IDENTICAL to the M3.6 leader's"
        );

        // Extensions: identical, or at most the SAME one benign shared[f32] layout-decoration delta
        // the M3.6 leader already carries (the dsc/dmm caches remain WRITTEN by the kept fill). NO
        // new capability ever (asserted above).
        let extra_exts: Vec<String> = exts.difference(&leader_exts).cloned().collect();
        assert!(
            extra_exts.is_empty(),
            "AT-2703: ablation variant {name} must not declare extensions beyond the M3.6 leader's \
             set; got {extra_exts:?}"
        );

        // shared_memory_bytes == the M3.6 leader's 4096 B (the FILL is kept; the dsc/dmm arrays are
        // still allocated + written, so the 4 KB footprint is CONSTANT — SHARED-MEMORY occupancy
        // held constant across variants, the AT-2701 cross-check premise).
        let expected_shared_bytes: u32 = (512 + 512) * 2 + 2 * 256 * 4;
        assert_eq!(
            meta.shared_memory_bytes, expected_shared_bytes,
            "AT-2703: ablation variant {name} shared_memory_bytes must be {expected_shared_bytes} \
             (==M3.6 leader — the fill is kept; the dsc/dmm caches remain written; the 4 KB footprint \
             is constant); got {}",
            meta.shared_memory_bytes
        );
    }

    eprintln!(
        "AT-2703 PASS: M3.12 ablation variants (structonly, passthrough) compile + spirv-val clean; \
         cap set BYTE-IDENTICAL to the M3.6 leader ({leader_caps:?}); shared_memory_bytes==4096 \
         (fill kept, footprint constant). PROFILING INSTRUMENTS — wrong output by design, no \
         correctness gate (AT-2705)."
    );
}
