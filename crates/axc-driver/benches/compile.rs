//! Criterion bench group `compile_pipeline` (M2.2).
//!
//! Measures source-text → SPIR-V wall time for saxpy and vector_add.
//! The timed region covers the full `axc_driver::compile_source_with_meta`
//! call: lexer + parser + HIR + codegen + in-process spirv-tools validation.
//!
//! # Pre-flight (AT-705)
//!
//! Each bench function asserts `compile_source_with_meta(src).is_ok()` ONCE
//! before entering the measurement loop.  A compile failure surfaces as a loud
//! panic rather than silent timing noise.
//!
//! # BatchSize (C2)
//!
//! No `iter_batched` needed: each compile allocates fresh internal Vecs so
//! there is no per-iteration mutation to undo.  Plain `b.iter` suffices.

#[path = "common.rs"]
mod common;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

/// Bench `compile_saxpy`: measure source→SPIR-V for examples/saxpy.axc.
fn compile_saxpy(c: &mut Criterion) {
    let src: &str = common::SAXPY_SRC;

    // Pre-flight: confirm compilation succeeds before timing (AT-705).
    axc_driver::compile_source_with_meta(src)
        .expect("saxpy.axc pre-flight: compile_source_with_meta must succeed");

    c.bench_function("compile_saxpy", |b| {
        b.iter(|| {
            let _ = axc_driver::compile_source_with_meta(black_box(src))
                .expect("compile_saxpy: unexpected compile failure during bench");
        });
    });
}

/// Bench `compile_vector_add`: measure source→SPIR-V for examples/vector_add.axc.
fn compile_vector_add(c: &mut Criterion) {
    let src: &str = common::VECTOR_ADD_SRC;

    // Pre-flight: confirm compilation succeeds before timing (AT-705).
    axc_driver::compile_source_with_meta(src)
        .expect("vector_add.axc pre-flight: compile_source_with_meta must succeed");

    c.bench_function("compile_vector_add", |b| {
        b.iter(|| {
            let _ = axc_driver::compile_source_with_meta(black_box(src))
                .expect("compile_vector_add: unexpected compile failure during bench");
        });
    });
}

criterion_group!(compile_benches, compile_saxpy, compile_vector_add);
criterion_main!(compile_benches);
