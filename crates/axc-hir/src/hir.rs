//! HIR data types for AXIOM-Compute.
//!
//! This is the validated, annotation-resolved form of the source AST.
//! In M0 only models empty compute kernels with required GPU annotations.
//!
//! Design note: `@complexity` is stored as a structured `ComplexityForm` enum
//! rather than `Option<String>`, per anti-pattern #7 (no string types for
//! structured data).

use axc_lexer::Span;

/// Vulkan 1.1 guaranteed minimum `maxComputeWorkGroupInvocations`.
///
/// See Vulkan Specification, "Required Limits" table.
/// Product ≤ this value: CLEAN (no warning).
/// Product in (this, DESKTOP_MAX]: WARN.
pub const PORTABLE_MIN_WORKGROUP_INVOCATIONS: u32 = 128;

/// Observed desktop-class ceiling for `maxComputeWorkGroupInvocations`.
///
/// Applies to NVIDIA, AMD, Intel, Apple M-series as of 2025.
/// Product ≤ this value (and > PORTABLE_MIN): WARN.
/// Product > this value: ERROR — may be rejected by actual hardware drivers.
pub const DESKTOP_MAX_WORKGROUP_INVOCATIONS: u32 = 1024;

/// A validated HIR module containing zero or more compute kernels.
#[derive(Debug, Clone)]
pub struct Module {
    pub kernels: Vec<Kernel>,
}

/// Opaque kernel identifier (monotonically assigned, from 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelId(pub u32);

/// A fully validated compute kernel.
#[derive(Debug, Clone)]
pub struct Kernel {
    pub id: KernelId,
    pub name: String,
    pub annotations: KernelAnnotations,
    pub body: KernelBody,
    pub span: Span,
}

/// Structured representation of the supported annotation set for a kernel.
///
/// All fields are structured types — no `String` is used for data that has
/// an enumerable shape (anti-pattern #7 compliance).
#[derive(Debug, Clone)]
pub struct KernelAnnotations {
    pub workgroup: WorkgroupDims,
    /// `@intent("…")` — user-facing free-form prose.
    /// This DOES NOT violate anti-pattern #7 because the String IS the data;
    /// there is no structure to lose by storing it as text.
    pub intent: Option<String>,
    /// `@complexity(…)` — stored as structured enum (anti-pattern #7).
    pub complexity: Option<ComplexityForm>,
    /// `@precondition(true)` — M0 accepts only literal `true`.
    pub preconditions: Vec<PreconditionTrivial>,
    /// `@subgroup_uniform` flag.
    pub subgroup_uniform: bool,
}

/// Complexity form: the outer function (`O`, `Theta`, `Omega`).
///
/// `NSquared` is defined to lock the enum shape for M1 but is NOT reachable
/// via any valid M0 source (the `^` grammar is absent from M0 ann_args).
/// The `#[allow(dead_code)]` is required to silence clippy on the unreachable variant.
#[derive(Debug, Clone, PartialEq)]
pub enum ComplexityForm {
    O(ComplexityVar),
    Theta(ComplexityVar),
    Omega(ComplexityVar),
}

/// Complexity variable: the inner argument.
///
/// `NSquared` is defined to lock the enum shape for M1 wiring.
#[derive(Debug, Clone, PartialEq)]
pub enum ComplexityVar {
    One,
    N,
    #[allow(dead_code)]
    NSquared,
}

/// M0 only accepts `@precondition(true)`.
///
/// A structural marker (not a String) so HIR remains free of stringly-typed data.
#[derive(Debug, Clone)]
pub struct PreconditionTrivial {
    pub span: Span,
}

/// Workgroup invocation dimensions from `@workgroup(X, Y, Z)`.
#[derive(Debug, Clone)]
pub struct WorkgroupDims {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl WorkgroupDims {
    /// Compute the total number of invocations (product of all three dimensions).
    ///
    /// Uses `u64` to avoid overflow when individual dimensions are near `u32::MAX`.
    pub fn product(&self) -> u64 {
        (self.x as u64) * (self.y as u64) * (self.z as u64)
    }
}

/// The body of a kernel.
///
/// M0 only represents empty kernels (those that `return;` immediately).
/// Richer bodies are an M1 deliverable.
#[derive(Debug, Clone)]
pub enum KernelBody {
    Empty,
}
