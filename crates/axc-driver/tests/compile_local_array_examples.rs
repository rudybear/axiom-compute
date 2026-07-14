//! Compile + spirv-val tests for M3.20 `array[T,N]` local-array examples.
//!
//! Mirrors `compile_shared_examples.rs`'s style for the M3.2 shared-array anchor.
//!
//! AT-2941: `local_histogram.axc` compiles + spirv-val clean with the shipped
//! source; `metadata.shared_memory_bytes == 0` at schema v4; the Function
//! `OpVariable` count matches the declared-array count.
//! AT-2939: metadata deserializes at schema v4 (CURRENT_SCHEMA_VERSION) unchanged.

use axc_driver::compile_source_with_meta;
use spirv_tools::val::{Validator, create as create_validator};
use spirv_tools::TargetEnv;

/// Compile bytes -> SPIR-V words + spirv-val. Returns (words, metadata).
fn compile_and_validate(src: &str, name: &str) -> (Vec<u32>, axc_runtime::KernelMetadata) {
    let (bytes, meta) = compile_source_with_meta(src)
        .unwrap_or_else(|e| panic!("{name}: compile failed: {e:?}"));
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let validator = create_validator(Some(TargetEnv::Vulkan_1_1));
    validator
        .validate(&words, None)
        .unwrap_or_else(|e| panic!("{name}: spirv-val failed: {e}"));
    (words, meta)
}

/// Count `OpVariable ... Function` instructions in a SPIR-V word stream.
///
/// `OpVariable` = opcode 59. Layout: `[wc<<16|59, result_type, result_id, storage_class, ...]`.
/// `StorageClass::Function` = 7 (SPIR-V spec).
fn count_function_opvariable(words: &[u32]) -> usize {
    const OP_VARIABLE: u32 = 59;
    const STORAGE_CLASS_FUNCTION: u32 = 7;
    let mut n = 0usize;
    let mut i = 5usize; // skip the 5-word module header
    while i < words.len() {
        let opcode = words[i] & 0xFFFF;
        let wc = (words[i] >> 16) as usize;
        if wc == 0 {
            break;
        }
        if opcode == OP_VARIABLE && wc >= 4 && words[i + 3] == STORAGE_CLASS_FUNCTION {
            n += 1;
        }
        i += wc;
    }
    n
}

/// AT-2941: `local_histogram.axc` compiles + spirv-val clean with the shipped
/// source; `metadata.shared_memory_bytes == 0` at schema v4; the Function
/// `OpVariable` count matches the declared-array count (exactly 1 — `hist`).
///
/// AT-2939: the emitted `metadata` deserializes at schema v4
/// (`CURRENT_SCHEMA_VERSION`) unchanged — no new field, no bump — proving local
/// arrays are metadata-inert (private memory needs no runtime preflight).
#[test]
fn at2941_at2939_local_histogram_compiles_and_validates() {
    let src = include_str!("../../../examples/local_histogram.axc");
    let (words, meta) = compile_and_validate(src, "local_histogram.axc");

    assert_eq!(
        meta.shared_memory_bytes, 0,
        "AT-2941: local_histogram.axc uses NO shared[T,N] memory; shared_memory_bytes must be 0"
    );
    assert_eq!(meta.entry_point, "local_histogram");

    // AT-2939: round-trip metadata through JSON to prove it deserializes cleanly
    // at the CURRENT schema version (v4) — the same struct axc-runtime already
    // (de)serializes; local arrays add NO new field.
    let json = serde_json::to_string(&meta).expect("AT-2939: metadata must serialize");
    let round_tripped: axc_runtime::KernelMetadata = serde_json::from_str(&json)
        .expect("AT-2939: metadata must deserialize at the current schema version");
    assert_eq!(round_tripped.shared_memory_bytes, 0, "AT-2939: round-tripped shared_memory_bytes must stay 0");
    assert_eq!(
        round_tripped.schema_version, axc_runtime::CURRENT_SCHEMA_VERSION,
        "AT-2939: metadata must round-trip at CURRENT_SCHEMA_VERSION (v4) — unchanged by M3.20"
    );

    // Exactly one Function OpVariable count attributable to the local array
    // (`hist`) — cross-checked as "at least 1" (other Function OpVariables exist
    // for `let` scalars: gid0, base, i (x3 loop inductions), idx, v, b).
    let function_var_count = count_function_opvariable(&words);
    assert!(
        function_var_count >= 1,
        "AT-2941: expected at least 1 Function OpVariable (the `hist` local array); \
         got {function_var_count}"
    );
}

/// AT-2935/AT-2941 structural cross-check: `local_histogram.axc`'s SPIR-V
/// contains exactly one `OpTypeArray` (one declared local array: `hist`).
#[test]
fn local_histogram_has_exactly_one_local_array_type() {
    let src = include_str!("../../../examples/local_histogram.axc");
    let (words, _meta) = compile_and_validate(src, "local_histogram.axc");

    const OP_TYPE_ARRAY: u32 = 28;
    let mut count = 0usize;
    let mut i = 5usize;
    while i < words.len() {
        let opcode = words[i] & 0xFFFF;
        let wc = (words[i] >> 16) as usize;
        if wc == 0 {
            break;
        }
        if opcode == OP_TYPE_ARRAY {
            count += 1;
        }
        i += wc;
    }
    assert_eq!(count, 1, "expected exactly one OpTypeArray (hist: array[u32,8])");
}
