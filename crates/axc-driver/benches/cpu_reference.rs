//! Criterion bench group `cpu_reference` (M2.2).
//!
//! Measures the equivalent Rust loop for each kernel at two problem sizes.
//! Independent of Vulkan availability — runs on any machine (even CI without a
//! GPU driver).
//!
//! # Bench layout
//!
//! | Bench name         | N          | BatchSize     |
//! |--------------------|------------|---------------|
//! | cpu_saxpy_1024     | 1 024      | SmallInput    |
//! | cpu_saxpy_1m       | 1 048 576  | LargeInput    |
//! | cpu_vector_add_1024 | 1 024     | iter (r/o)    |
//! | cpu_vector_add_1m  | 1 048 576  | iter (r/o)    |
//!
//! # BatchSize rationale (C2 resolution)
//!
//! `cpu_saxpy`: saxpy writes to `y` in-place (the output is a fresh Vec, but
//! the conceptual model requires fresh mutable copies).  We use `iter_batched`
//! so each iteration gets a fresh clone of `y`.
//! - N=1024: 2 × 4 KB = 8 KB per setup — well within Criterion's SmallInput
//!   threshold heuristic.  `BatchSize::SmallInput` amortises setup cheaply.
//! - N=1M: 2 × 4 MB = 8 MB per iteration input — far above the ~10 KB
//!   SmallInput threshold.  `BatchSize::LargeInput` reduces iterations-per-batch
//!   so setup allocation does not dominate timing variance.
//!
//! `cpu_vector_add`: both inputs are read-only; the output is a fresh Vec every
//! call.  No per-iteration clone needed — plain `b.iter` suffices for both sizes.

#[path = "common.rs"]
mod common;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};

/// N for "small" benches.
const N_SMALL: usize = 1_024;

/// N for "large" benches.
const N_LARGE: usize = 1_048_576;

// ── saxpy benches ─────────────────────────────────────────────────────────────

/// Bench `cpu_saxpy_1024`: CPU saxpy reference, N=1024.
///
/// BatchSize::SmallInput because N=1024 × 2 vectors × 4 bytes = 8 KB per-iter
/// setup; Criterion can amortise many iterations cheaply.
fn cpu_saxpy_1024(c: &mut Criterion) {
    let (x, y, alpha) = common::make_saxpy_inputs(N_SMALL, common::SEED);

    // Pre-flight: confirm the helper produces a sane output shape.
    let preflight = common::saxpy_cpu_reference(&x, &y, alpha);
    assert_eq!(preflight.len(), N_SMALL, "cpu_saxpy_1024 pre-flight: output length mismatch");

    c.bench_function("cpu_saxpy_1024", |b| {
        // BatchSize::SmallInput: 8 KB per setup, well within Criterion's threshold.
        b.iter_batched(
            || (x.clone(), y.clone()),
            |(x_local, y_local)| {
                black_box(common::saxpy_cpu_reference(
                    black_box(&x_local),
                    black_box(&y_local),
                    black_box(alpha),
                ))
            },
            BatchSize::SmallInput,
        );
    });
}

/// Bench `cpu_saxpy_1m`: CPU saxpy reference, N=1M.
///
/// BatchSize::LargeInput because N=1M × 2 vectors × 4 bytes = 8 MB per-iter
/// setup; SmallInput's aggressive batching causes setup allocation to dominate
/// measurement variance above ~10 KB (Criterion docs recommendation).
fn cpu_saxpy_1m(c: &mut Criterion) {
    let (x, y, alpha) = common::make_saxpy_inputs(N_LARGE, common::SEED);

    // Pre-flight shape check.
    let preflight = common::saxpy_cpu_reference(&x, &y, alpha);
    assert_eq!(preflight.len(), N_LARGE, "cpu_saxpy_1m pre-flight: output length mismatch");

    c.bench_function("cpu_saxpy_1m", |b| {
        // BatchSize::LargeInput: 8 MB per setup; reduces pre-allocated batch size
        // so setup-memory peaks stay bounded and variance stabilises.
        b.iter_batched(
            || (x.clone(), y.clone()),
            |(x_local, y_local)| {
                black_box(common::saxpy_cpu_reference(
                    black_box(&x_local),
                    black_box(&y_local),
                    black_box(alpha),
                ))
            },
            BatchSize::LargeInput,
        );
    });
}

// ── vector_add benches ────────────────────────────────────────────────────────

/// Bench `cpu_vector_add_1024`: CPU vector_add reference, N=1024.
///
/// Inputs are read-only; output is a fresh Vec every call.
/// Plain `b.iter` — no per-iteration clone needed.
fn cpu_vector_add_1024(c: &mut Criterion) {
    let (a, b) = common::make_vector_add_inputs(N_SMALL, common::SEED);

    // Pre-flight shape check.
    let preflight = common::vector_add_cpu_reference(&a, &b);
    assert_eq!(
        preflight.len(),
        N_SMALL,
        "cpu_vector_add_1024 pre-flight: output length mismatch"
    );

    c.bench_function("cpu_vector_add_1024", |b_crit| {
        b_crit.iter(|| {
            black_box(common::vector_add_cpu_reference(black_box(&a), black_box(&b)))
        });
    });
}

/// Bench `cpu_vector_add_1m`: CPU vector_add reference, N=1M.
///
/// Inputs (4 MB vectors) are constructed ONCE outside the iter call — no
/// per-iteration clone, so BatchSize is moot.  Plain `b.iter` is used.
fn cpu_vector_add_1m(c: &mut Criterion) {
    let (a, b) = common::make_vector_add_inputs(N_LARGE, common::SEED);

    // Pre-flight shape check.
    let preflight = common::vector_add_cpu_reference(&a, &b);
    assert_eq!(
        preflight.len(),
        N_LARGE,
        "cpu_vector_add_1m pre-flight: output length mismatch"
    );

    c.bench_function("cpu_vector_add_1m", |b_crit| {
        b_crit.iter(|| {
            black_box(common::vector_add_cpu_reference(black_box(&a), black_box(&b)))
        });
    });
}

criterion_group!(
    cpu_reference_benches,
    cpu_saxpy_1024,
    cpu_saxpy_1m,
    cpu_vector_add_1024,
    cpu_vector_add_1m
);
criterion_main!(cpu_reference_benches);
