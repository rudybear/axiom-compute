//! HIR data types for cooperative-matrix operations (M2.1).
//!
//! This module mirrors `crates/axc-hir/src/subgroup.rs` in layout and discipline.
//! It contains pure HIR data: no codegen dependency, no type-checking logic.
//!
//! M2.1 scope:
//! - `CoopMatUse`: the three matrix roles (A, B, Accumulator).
//! - `CoopMatBuiltin`: the four new builtin call names.
//! - `CoopMatrixShapeKind`: typed shape-mismatch descriptors for diagnostics.
//! - `CoopMatKey`: cache key used by both HIR (expr annotation) and codegen type cache.
//! - `RESERVED_COOPMAT_BUILTIN_NAMES`: sorted slice for binary_search reserved-name checks.
//! - `is_allowed_coopmat_element`: gate for M2.1's permitted element type set.
//!
//! NOTE: axc-lexer does NOT depend on axc-hir (it is the lowest crate). The
//! `RESERVED_COOPMAT_BUILTIN_NAMES` constant is duplicated in
//! `axc_lexer::token` for the lexer-layer reserved-name check. Two copies of a
//! 4-element constant is acceptable; a future single-source-of-truth helper crate
//! is a documented tech debt item.

use crate::ty::ScalarTy;

// ── CoopMatUse ────────────────────────────────────────────────────────────────

/// The `use` tag carried on every cooperative-matrix value.
///
/// Per SPV_KHR_cooperative_matrix:
/// - `MatrixAKHR  = 0`
/// - `MatrixBKHR  = 1`
/// - `MatrixAccumulatorKHR = 2`
///
/// The raw SPIR-V u32 values are NOT spelled out in tests; tests compare against
/// rspirv's typed `CooperativeMatrixUse` enum (via `use_to_rspirv` in axc-codegen).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CoopMatUse {
    MatrixA,
    MatrixB,
    Accumulator,
}

impl CoopMatUse {
    /// Source-level keyword used in the `matrix[T, M, N, use]` type.
    ///
    /// Returns `"a"`, `"b"`, or `"accumulator"`.
    pub fn source_name(self) -> &'static str {
        match self {
            CoopMatUse::MatrixA => "a",
            CoopMatUse::MatrixB => "b",
            CoopMatUse::Accumulator => "accumulator",
        }
    }

    /// Parse the source-level keyword; `None` for unknown strings.
    ///
    /// This is case-sensitive: `"A"` returns `None`.
    pub fn from_source_name(s: &str) -> Option<CoopMatUse> {
        match s {
            "a" => Some(CoopMatUse::MatrixA),
            "b" => Some(CoopMatUse::MatrixB),
            "accumulator" => Some(CoopMatUse::Accumulator),
            _ => None,
        }
    }
}

impl std::fmt::Display for CoopMatUse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.source_name())
    }
}

// ── CoopMatrixShapeKind ───────────────────────────────────────────────────────

/// Descriptor carried in `CoopMatrixShapeMismatch` diagnostic payloads.
///
/// Variant-per-failure-mode rather than a single string — strings are for humans,
/// enums are for machines (§AXIOM-Compute anti-pattern #7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoopMatrixShapeKind {
    /// The K dimension of mul_add did not match: `a.n != b.m`.
    KDimMismatch { a_n: u32, b_m: u32 },
    /// The accumulator M did not match `a.m`.
    AccumulatorMMismatch { c_m: u32, a_m: u32 },
    /// The accumulator N did not match `b.n`.
    AccumulatorNMismatch { c_n: u32, b_n: u32 },
    /// The A argument was not tagged `MatrixA`.
    AUseMismatch { found: CoopMatUse },
    /// The B argument was not tagged `MatrixB`.
    BUseMismatch { found: CoopMatUse },
    /// The C argument was not tagged `Accumulator`.
    CUseMismatch { found: CoopMatUse },
    /// The element types of A and B do not match.
    ABElementMismatch { a_elem: ScalarTy, b_elem: ScalarTy },
}

// ── CoopMatBuiltin ────────────────────────────────────────────────────────────

/// The four new cooperative-matrix builtin call names (M2.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoopMatBuiltin {
    /// `coopmat_zero() -> matrix[T, M, N, use]` — result from expected-type context.
    Zero,
    /// `coopmat_load(buf, element_offset, stride) -> matrix[T, M, N, use]`.
    Load,
    /// `coopmat_store(m, buf, element_offset, stride)` — void; statement-only.
    Store,
    /// `coopmat_mul_add(a, b, c) -> matrix[T, M, N, accumulator]`.
    MulAdd,
}

impl CoopMatBuiltin {
    /// Return the canonical source-code name for this builtin.
    pub fn source_name(self) -> &'static str {
        match self {
            CoopMatBuiltin::Zero => "coopmat_zero",
            CoopMatBuiltin::Load => "coopmat_load",
            CoopMatBuiltin::Store => "coopmat_store",
            CoopMatBuiltin::MulAdd => "coopmat_mul_add",
        }
    }

    /// Parse a source-level call name into a `CoopMatBuiltin`, or `None`.
    pub fn from_source_name(s: &str) -> Option<CoopMatBuiltin> {
        match s {
            "coopmat_zero" => Some(CoopMatBuiltin::Zero),
            "coopmat_load" => Some(CoopMatBuiltin::Load),
            "coopmat_store" => Some(CoopMatBuiltin::Store),
            "coopmat_mul_add" => Some(CoopMatBuiltin::MulAdd),
            _ => None,
        }
    }

    /// Number of source-level arguments accepted by this builtin.
    ///
    /// - `Zero`:    0 args.
    /// - `Load`:    3 args (buf, element_offset, stride).
    /// - `Store`:   4 args (m, buf, element_offset, stride).
    /// - `MulAdd`:  3 args (a, b, c).
    pub fn arity(self) -> usize {
        match self {
            CoopMatBuiltin::Zero => 0,
            CoopMatBuiltin::Load => 3,
            CoopMatBuiltin::Store => 4,
            CoopMatBuiltin::MulAdd => 3,
        }
    }

    /// True if this builtin's return type is determined by expected-type context
    /// (Zero and Load) rather than by arguments (MulAdd) or is void (Store).
    pub fn needs_expected_type(self) -> bool {
        match self {
            CoopMatBuiltin::Zero | CoopMatBuiltin::Load => true,
            CoopMatBuiltin::Store | CoopMatBuiltin::MulAdd => false,
        }
    }
}

// ── RESERVED_COOPMAT_BUILTIN_NAMES ───────────────────────────────────────────

/// Sorted-for-binary-search list of the four cooperative-matrix builtin names.
///
/// Used by the HIR reserved-name check to reject `let coopmat_load = ...` etc.
/// MUST remain sorted lexicographically. The lexer-layer copy in
/// `axc_lexer::token` must stay in sync whenever this list changes.
pub const RESERVED_COOPMAT_BUILTIN_NAMES: &[&str] = &[
    "coopmat_load",
    "coopmat_mul_add",
    "coopmat_store",
    "coopmat_zero",
];

/// Returns `true` if `name` is a reserved cooperative-matrix builtin identifier.
///
/// Uses binary search on the sorted `RESERVED_COOPMAT_BUILTIN_NAMES` slice.
pub fn is_reserved_coopmat_builtin(name: &str) -> bool {
    RESERVED_COOPMAT_BUILTIN_NAMES.binary_search(&name).is_ok()
}

// ── is_allowed_coopmat_element ────────────────────────────────────────────────

/// M2.1 permitted cooperative-matrix element type set: F16, F32, I8, U8, I32, U32.
///
/// F64 and Bool matrix elements are explicitly rejected. bf16 as a matrix element
/// is also rejected (bf16 as a primitive type is out of scope for M2.1).
/// I16 / U16 are also rejected.
pub fn is_allowed_coopmat_element(ty: ScalarTy) -> bool {
    matches!(
        ty,
        ScalarTy::F16 | ScalarTy::F32
        | ScalarTy::I8 | ScalarTy::U8
        | ScalarTy::I32 | ScalarTy::U32
    )
}

// ── CoopMatKey ────────────────────────────────────────────────────────────────

/// Cache key for the codegen-side `CoopMatTypeCache` and `CoopMatNullCache`.
///
/// Placed in HIR because it composes HIR-level fields; codegen reuses it
/// without creating an additional indirection layer.
///
/// `PartialOrd + Ord` are required by `BTreeMap<CoopMatKey, Word>` (determinism
/// invariant: no HashMap-driven emission order — M1.3 precedent).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CoopMatKey {
    pub elem: ScalarTy,
    pub m: u32,
    pub n: u32,
    pub use_: CoopMatUse,
}

// ── CoopMatrixShape ───────────────────────────────────────────────────────────

/// Annotation shape advertised by `@cooperative_matrix(M, N, K, A_type, B_type, C_type)`.
///
/// Used in `CooperativeMatrixAnnotationMismatch` diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoopMatrixShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a_elem: ScalarTy,
    pub b_elem: ScalarTy,
    pub c_elem: ScalarTy,
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coopmat_use_from_source_name_a() {
        assert_eq!(CoopMatUse::from_source_name("a"), Some(CoopMatUse::MatrixA));
    }

    #[test]
    fn coopmat_use_from_source_name_b() {
        assert_eq!(CoopMatUse::from_source_name("b"), Some(CoopMatUse::MatrixB));
    }

    #[test]
    fn coopmat_use_from_source_name_accumulator() {
        assert_eq!(CoopMatUse::from_source_name("accumulator"), Some(CoopMatUse::Accumulator));
    }

    #[test]
    fn coopmat_use_from_source_name_unknown_is_none() {
        // Case-sensitive: uppercase is not a valid use token.
        assert_eq!(CoopMatUse::from_source_name("x"), None);
        assert_eq!(CoopMatUse::from_source_name("A"), None);
        assert_eq!(CoopMatUse::from_source_name("Accumulator"), None);
        assert_eq!(CoopMatUse::from_source_name(""), None);
    }

    #[test]
    fn coopmat_builtin_from_source_name_happy() {
        // All four names round-trip via from_source_name / source_name.
        for &builtin in &[
            CoopMatBuiltin::Zero,
            CoopMatBuiltin::Load,
            CoopMatBuiltin::Store,
            CoopMatBuiltin::MulAdd,
        ] {
            let name = builtin.source_name();
            let parsed = CoopMatBuiltin::from_source_name(name);
            assert_eq!(parsed, Some(builtin), "round-trip failed for {name}");
        }
    }

    #[test]
    fn coopmat_builtin_arity_correct() {
        assert_eq!(CoopMatBuiltin::Zero.arity(), 0);
        assert_eq!(CoopMatBuiltin::Load.arity(), 3);
        assert_eq!(CoopMatBuiltin::Store.arity(), 4);
        assert_eq!(CoopMatBuiltin::MulAdd.arity(), 3);
    }

    #[test]
    fn coopmat_builtin_needs_expected_type() {
        assert!(CoopMatBuiltin::Zero.needs_expected_type(), "Zero needs expected type");
        assert!(CoopMatBuiltin::Load.needs_expected_type(), "Load needs expected type");
        assert!(!CoopMatBuiltin::Store.needs_expected_type(), "Store is void — no expected type");
        assert!(!CoopMatBuiltin::MulAdd.needs_expected_type(), "MulAdd result is from args");
    }

    #[test]
    fn reserved_coopmat_builtin_names_is_sorted_for_binary_search() {
        // The constant MUST be sorted for binary_search to work correctly.
        let mut sorted = RESERVED_COOPMAT_BUILTIN_NAMES.to_vec();
        sorted.sort_unstable();
        assert_eq!(
            RESERVED_COOPMAT_BUILTIN_NAMES,
            sorted.as_slice(),
            "RESERVED_COOPMAT_BUILTIN_NAMES must be sorted lexicographically"
        );
        // All four names are recoverable via binary_search.
        for name in RESERVED_COOPMAT_BUILTIN_NAMES {
            assert!(
                RESERVED_COOPMAT_BUILTIN_NAMES.binary_search(name).is_ok(),
                "{name} not found via binary_search"
            );
        }
    }

    #[test]
    fn is_allowed_coopmat_element_accepts_f16_f32_i8_u8_i32_u32() {
        // Allowed types.
        assert!(is_allowed_coopmat_element(ScalarTy::F16), "F16 should be allowed");
        assert!(is_allowed_coopmat_element(ScalarTy::F32), "F32 should be allowed");
        assert!(is_allowed_coopmat_element(ScalarTy::I8), "I8 should be allowed");
        assert!(is_allowed_coopmat_element(ScalarTy::U8), "U8 should be allowed");
        assert!(is_allowed_coopmat_element(ScalarTy::I32), "I32 should be allowed");
        assert!(is_allowed_coopmat_element(ScalarTy::U32), "U32 should be allowed");

        // Disallowed types.
        assert!(!is_allowed_coopmat_element(ScalarTy::F64), "F64 must be rejected");
        assert!(!is_allowed_coopmat_element(ScalarTy::Bool), "Bool must be rejected");
        assert!(!is_allowed_coopmat_element(ScalarTy::I16), "I16 must be rejected");
        assert!(!is_allowed_coopmat_element(ScalarTy::U16), "U16 must be rejected");
        assert!(!is_allowed_coopmat_element(ScalarTy::I64), "I64 must be rejected");
        assert!(!is_allowed_coopmat_element(ScalarTy::U64), "U64 must be rejected");
    }
}
