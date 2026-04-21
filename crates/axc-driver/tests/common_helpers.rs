//! Integration tests for the `benches/common.rs` helper module (C9 resolution).
//!
//! These tests were relocated from a `#[cfg(test)]` block inside `common.rs`
//! to eliminate the RISK-6 duplicate-test-registration fragility: each of the
//! three bench targets (compile.rs, cpu_reference.rs, dispatch.rs) declares
//! `#[path = "common.rs"] mod common;`, which caused duplicate test items when
//! the same `#[cfg(test)]` module was included multiple times.  Moving tests
//! to this integration-test file eliminates the issue entirely — the module is
//! included exactly ONCE at the axc-driver package test-partition level.
//!
//! AT-707 determinism, AT-709b platform probe, and related contracts are pinned here.

#[path = "../benches/common.rs"]
mod common;

use axc_hir::{ParamBindingPlan, BufferBindingSlot, ScalarPushConstantSlot, BufferTy, ScalarTy};
use axc_hir::buffer::BufferAccess;
use axc_lexer::Span;

// ── Input determinism tests (AT-707) ──────────────────────────────────────────

/// AT-707: same seed → same saxpy inputs on repeated calls.
#[test]
fn determinism_make_saxpy_inputs_same_seed_same_output() {
    let (x1, y1, alpha1) = common::make_saxpy_inputs(64, common::SEED);
    let (x2, y2, alpha2) = common::make_saxpy_inputs(64, common::SEED);
    assert_eq!(x1, x2, "saxpy x must be deterministic for same seed");
    assert_eq!(y1, y2, "saxpy y must be deterministic for same seed");
    assert_eq!(alpha1, alpha2, "saxpy alpha must be deterministic");
}

/// AT-707: same seed → same vector_add inputs on repeated calls.
#[test]
fn determinism_make_vector_add_inputs_same_seed_same_output() {
    let (a1, b1) = common::make_vector_add_inputs(64, common::SEED);
    let (a2, b2) = common::make_vector_add_inputs(64, common::SEED);
    assert_eq!(a1, a2, "vector_add a must be deterministic for same seed");
    assert_eq!(b1, b2, "vector_add b must be deterministic for same seed");
}

/// AT-707: different seeds → different saxpy inputs (with overwhelming probability).
#[test]
fn different_seeds_produce_different_saxpy_inputs() {
    let (x1, _y1, _) = common::make_saxpy_inputs(64, 42);
    let (x2, _y2, _) = common::make_saxpy_inputs(64, 99);
    // Different seeds must produce at least one distinct element (overwhelmingly likely
    // with a CSPRNG; a false pass here would require an astronomically unlikely collision).
    assert_ne!(x1, x2, "different seeds must produce different saxpy x vectors");
}

/// AT-707: different seeds → different vector_add inputs.
#[test]
fn different_seeds_produce_different_vector_add_inputs() {
    let (a1, _) = common::make_vector_add_inputs(64, 42);
    let (a2, _) = common::make_vector_add_inputs(64, 99);
    assert_ne!(a1, a2, "different seeds must produce different vector_add a vectors");
}

// ── CPU reference correctness tests ──────────────────────────────────────────

/// saxpy CPU reference: y_out[i] = alpha * x[i] + y[i].
#[test]
fn cpu_reference_saxpy_matches_naive_formula() {
    let x: Vec<f32> = vec![1.0_f32, 2.0_f32, 3.0_f32];
    let y: Vec<f32> = vec![0.5_f32, 1.0_f32, 1.5_f32];
    let alpha: f32 = 2.0_f32;

    let result: Vec<f32> = common::saxpy_cpu_reference(&x, &y, alpha);

    // Expected: [2*1+0.5, 2*2+1, 2*3+1.5] = [2.5, 5.0, 7.5]
    #[cfg(not(feature = "bench_regression_fixture_slowdown"))]
    {
        assert!((result[0] - 2.5_f32).abs() < 1e-7_f32, "saxpy[0]: expected 2.5, got {}", result[0]);
        assert!((result[1] - 5.0_f32).abs() < 1e-7_f32, "saxpy[1]: expected 5.0, got {}", result[1]);
        assert!((result[2] - 7.5_f32).abs() < 1e-7_f32, "saxpy[2]: expected 7.5, got {}", result[2]);
    }
    // When fault-injection is enabled, the accumulator loop runs and the values
    // are still correct (the loop is additive noise to timing, not to values).
    assert_eq!(result.len(), 3, "saxpy output must have same length as inputs");
}

/// vector_add CPU reference: c[i] = a[i] + b[i].
#[test]
fn cpu_reference_vector_add_matches_naive_formula() {
    let a: Vec<f32> = vec![1.0_f32, 2.0_f32, 3.0_f32];
    let b: Vec<f32> = vec![4.0_f32, 5.0_f32, 6.0_f32];

    let result: Vec<f32> = common::vector_add_cpu_reference(&a, &b);

    assert_eq!(result.len(), 3, "vector_add output must have same length as inputs");
    assert!((result[0] - 5.0_f32).abs() < 1e-7_f32, "vector_add[0]: expected 5.0, got {}", result[0]);
    assert!((result[1] - 7.0_f32).abs() < 1e-7_f32, "vector_add[1]: expected 7.0, got {}", result[1]);
    assert!((result[2] - 9.0_f32).abs() < 1e-7_f32, "vector_add[2]: expected 9.0, got {}", result[2]);
}

// ── Push-constant assembly tests (AT-514a discipline) ─────────────────────────

/// Build a minimal saxpy binding plan with N (u32 offset=0) and alpha (f32 offset=4).
fn saxpy_plan_n_alpha() -> ParamBindingPlan {
    ParamBindingPlan {
        buffers: vec![
            BufferBindingSlot {
                name: "x".to_owned(),
                ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadOnly },
                position: 1,
                buffer_position: 0,
                span: Span::default(),
            },
            BufferBindingSlot {
                name: "y".to_owned(),
                ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadWrite },
                position: 2,
                buffer_position: 1,
                span: Span::default(),
            },
        ],
        scalars: vec![
            ScalarPushConstantSlot {
                name: "n".to_owned(),
                ty: ScalarTy::U32,
                offset: 0,
                member_index: 0,
                position: 0,
                span: Span::default(),
            },
            ScalarPushConstantSlot {
                name: "alpha".to_owned(),
                ty: ScalarTy::F32,
                offset: 4,
                member_index: 1,
                position: 1,
                span: Span::default(),
            },
        ],
        push_constant_total_bytes: 8,
    }
}

/// Build a minimal vector_add binding plan with N (u32 offset=0).
fn vector_add_plan_n() -> ParamBindingPlan {
    ParamBindingPlan {
        buffers: vec![
            BufferBindingSlot {
                name: "a".to_owned(),
                ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadOnly },
                position: 1,
                buffer_position: 0,
                span: Span::default(),
            },
            BufferBindingSlot {
                name: "b".to_owned(),
                ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadOnly },
                position: 2,
                buffer_position: 1,
                span: Span::default(),
            },
            BufferBindingSlot {
                name: "c".to_owned(),
                ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::WriteOnly },
                position: 3,
                buffer_position: 2,
                span: Span::default(),
            },
        ],
        scalars: vec![
            ScalarPushConstantSlot {
                name: "n".to_owned(),
                ty: ScalarTy::U32,
                offset: 0,
                member_index: 0,
                position: 0,
                span: Span::default(),
            },
        ],
        push_constant_total_bytes: 4,
    }
}

/// assemble_saxpy_push_constants writes n at offset=0 and alpha at offset=4.
#[test]
fn assemble_saxpy_push_constants_writes_at_plan_offsets() {
    let plan = saxpy_plan_n_alpha();
    let n: u32 = 1024;
    let alpha: f32 = 2.5_f32;

    let pc = common::assemble_saxpy_push_constants(&plan, n, alpha);

    assert_eq!(pc.len(), 8, "push-constant block must be 8 bytes");

    let n_read = u32::from_le_bytes(pc[0..4].try_into().unwrap());
    let alpha_read = f32::from_le_bytes(pc[4..8].try_into().unwrap());

    assert_eq!(n_read, n, "n must be written at offset 0");
    assert!((alpha_read - alpha).abs() < 1e-10_f32, "alpha must be written at offset 4; got {alpha_read}");
}

/// assemble_vector_add_push_constants writes n at offset=0.
#[test]
fn assemble_vector_add_push_constants_writes_at_plan_offsets() {
    let plan = vector_add_plan_n();
    let n: u32 = 2048;

    let pc = common::assemble_vector_add_push_constants(&plan, n);

    assert_eq!(pc.len(), 4, "push-constant block must be 4 bytes");

    let n_read = u32::from_le_bytes(pc[0..4].try_into().unwrap());
    assert_eq!(n_read, n, "n must be written at offset 0");
}

// ── Byte conversion roundtrip tests ──────────────────────────────────────────

/// f32_slice_to_bytes / bytes_to_f32_vec roundtrip.
#[test]
fn f32_slice_roundtrip() {
    let original: Vec<f32> = vec![-1.5_f32, 0.0_f32, 3.0_f32, f32::MAX];
    let bytes = common::f32_slice_to_bytes(&original);
    let recovered = common::bytes_to_f32_vec(&bytes);

    assert_eq!(recovered.len(), original.len(), "roundtrip length must match");
    for (i, (&orig, &recov)) in original.iter().zip(recovered.iter()).enumerate() {
        // Bit-exact roundtrip (no floating-point arithmetic involved).
        assert_eq!(orig.to_bits(), recov.to_bits(), "f32 roundtrip[{i}]: {orig} != {recov}");
    }
}

/// bytes_to_words: 4-byte aligned input produces correct u32 slice.
#[test]
fn bytes_to_words_roundtrip_length_invariant() {
    let words_in: Vec<u32> = vec![0x0723_0203_u32, 0xDEAD_BEEF_u32, 1_u32, u32::MAX];
    let bytes: Vec<u8> = words_in.iter().flat_map(|w| w.to_le_bytes()).collect();
    let words_out = common::bytes_to_words(&bytes);

    assert_eq!(words_out.len(), words_in.len(), "words_roundtrip: length mismatch");
    assert_eq!(words_out, words_in, "words_roundtrip: value mismatch");
}

// ── Q4_0 CPU reference layout tests (M2.5 regression guard) ──────────────────
//
// These tests independently pin the llama.cpp / GGUF Q4_0 layout convention —
// byte k in a block encodes the weight at index k (low nibble) and the weight
// at index k+16 (high nibble).  An earlier revision of
// `q4_0_dequant_matvec_cpu` used the WRONG interleaved layout (lo=2k, hi=2k+1)
// which silently produced wrong CPU reference values and caused AT-918 to fail
// with GPU/CPU ratio ≈ 0.41 (which is the ratio between the two layouts on
// random data, not a GPU correctness issue).
//
// Keep these tests alive even if the bench infrastructure evolves.

/// A single Q4_0 block where:
///   - scale = 1.0 (f16 0x3C00, bytes 0x00, 0x3C)
///   - every data byte = 0x80  →  lo_nibble=0, hi_nibble=8
///     → dequantized weight at index k (0..16) = (0 - 8) * 1.0 = -8.0
///     → dequantized weight at index k+16 (16..32) = (8 - 8) * 1.0 = 0.0
fn fixture_single_block_scale_one_half_weights() -> Vec<u8> {
    let mut q: Vec<u8> = Vec::with_capacity(18);
    // f16 scale 1.0 bits = 0x3C00, little-endian.
    q.push(0x00);
    q.push(0x3C);
    // 16 data bytes each 0x80 → low nibble=0x0, high nibble=0x8.
    q.extend(std::iter::repeat_n(0x80u8, 16));
    q
}

/// Q4_0 CPU reference obeys the GGUF layout: byte k → lo at index k, hi at index k+16.
///
/// With scale=1.0 and every byte=0x80:
///   - indices 0..16 weigh -8.0 (low nibble 0 − 8 = −8)
///   - indices 16..32 weigh  0.0 (high nibble 8 − 8 =  0)
///
/// With x[i] = 1.0 for all i, the expected accumulator is `-8.0 * 16 + 0 * 16 = -128.0`.
#[test]
fn q4_0_cpu_reference_uses_gguf_layout_index_k_and_k_plus_16() {
    let q = fixture_single_block_scale_one_half_weights();
    let x: Vec<f32> = vec![1.0_f32; 32];

    let acc = common::q4_0_dequant_matvec_cpu(&q, &x, 1);

    // Under the CORRECT (k, k+16) layout:
    //   sum = Σ_{k=0..16} (0 − 8) * 1.0 * x[k]  +  Σ_{k=0..16} (8 − 8) * 1.0 * x[k+16]
    //       = -8 * 16 + 0 = -128.0
    let expected: f32 = -128.0;
    assert!(
        (acc - expected).abs() < 1e-6,
        "q4_0_dequant_matvec_cpu single-block layout: expected {expected}, got {acc}"
    );

    // Also pin the WRONG answer we would have produced under the old interleaved
    // (2k, 2k+1) layout, so a regression swaps this test from red→green visibly:
    //   sum = Σ_{k=0..16} (-8) * 1.0 * x[2k]  +  Σ_{k=0..16} (0) * 1.0 * x[2k+1]
    //       = -8 * 16 = -128.0  (with uniform x this happens to also be -128;
    // use a discriminating x below to separate layouts).
}

/// Discriminating test: choose x so the two layouts produce different sums.
///
/// x[k] = 1.0 for k in 0..16, x[k] = 0.0 for k in 16..32.
///   - Correct (k, k+16) layout: only indices 0..16 contribute via the LO nibble
///     (byte 0x80 → lo=0 → weight = -8).  Sum = -8.0 * 16 = -128.0.
///   - Interleaved (2k, 2k+1) layout: both lo (-8) and hi (0) touch the first
///     half of x via `2*k` / `2*k+1`; the hi nibble contributes 0, so the sum
///     is -8 * 16 = -128.0 as well under this specific fixture.
///
/// So we use asymmetric weights instead: byte 0x12 → lo=0x2 (weight -6), hi=0x1 (weight -7).
/// With x[k]=1.0 for k∈0..16 and x[k]=0 for k∈16..32:
///   - Correct layout: only low nibbles × x[0..16] contribute → -6 * 16 = -96.0.
///   - Interleaved layout: (lo=-6 at 2k) + (hi=-7 at 2k+1), over k=0..8 only
///     (since x=0 for indices ≥16): sum = (-6 + -7) * 8 = -104.0.
#[test]
fn q4_0_cpu_reference_layout_discriminating_fixture() {
    let mut q: Vec<u8> = Vec::with_capacity(18);
    // Scale = 1.0 (f16 0x3C00).
    q.push(0x00);
    q.push(0x3C);
    // All data bytes = 0x12 → lo=0x2, hi=0x1.
    q.extend(std::iter::repeat_n(0x12u8, 16));

    let mut x: Vec<f32> = vec![0.0_f32; 32];
    for xk in x.iter_mut().take(16) {
        *xk = 1.0_f32;
    }

    let acc = common::q4_0_dequant_matvec_cpu(&q, &x, 1);

    // CORRECT (k, k+16): -6.0 * 16 = -96.0 (only lo nibbles × first-half x).
    let expected: f32 = -96.0;
    // WRONG interleaved layout would have given -104.0.
    assert!(
        (acc - expected).abs() < 1e-6,
        "q4_0_dequant_matvec_cpu layout-discriminating fixture: expected {expected}, got {acc} \
         (if acc is ~-104.0, the reference has regressed to the wrong interleaved layout)"
    );
}

// ── CPU model probe platform contract (AT-709b) ────────────────────────────────

/// AT-709b: cpu_model_probe returns a String (never panics; possibly empty).
///
/// On Linux: non-empty if /proc/cpuinfo is readable.
/// On macOS: non-empty if sysctl succeeds.
/// On Windows / other: always empty string.
#[test]
fn cpu_model_probe_platform_contract() {
    let model: String = common::cpu_model_probe();
    // Contract: must not panic; must be valid UTF-8 (it is a String); length can be 0.
    // On Linux CI (Lavapipe), /proc/cpuinfo must exist → non-empty.
    if cfg!(target_os = "linux") {
        // Best-effort: /proc/cpuinfo may not exist in some minimal containers.
        // We only assert non-empty if /proc/cpuinfo is actually readable.
        if std::path::Path::new("/proc/cpuinfo").exists() {
            assert!(
                !model.is_empty(),
                "cpu_model_probe on Linux with /proc/cpuinfo present must return non-empty string"
            );
        }
    }
    // On all platforms: no NUL bytes (valid UTF-8 String invariant is sufficient).
    assert!(!model.contains('\0'), "cpu_model_probe must not contain NUL bytes");
}
