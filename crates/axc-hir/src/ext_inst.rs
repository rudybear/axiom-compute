//! HIR data types for the GLSL.std.450 extended-instruction builtin family (M3.2c).
//!
//! This module mirrors `crates/axc-hir/src/q4_0.rs` in layout and discipline.
//! It contains pure HIR data: no codegen dependency, no type-checking logic.
//!
//! These builtins lower to a SPIR-V `OpExtInst` referencing the `GLSL.std.450`
//! extended instruction set (the FIRST `OpExtInstImport` ever emitted by codegen).
//! The import is emitted ONCE per module (cached set-id) — see
//! `axc-codegen/src/ext_inst.rs` and `ScalarTypeCache::get_or_emit_glsl450_set`.
//!
//! M3.2c-exp ships exactly one variant: `Exp` (GLSL.std.450 instruction 27).
//! The machinery is reusable for future ext-inst builtins (exp2=29, log=28,
//! sqrt=31, etc.) but those are OUT OF SCOPE (anti-pattern #4: no feature without
//! a test — each future ext-inst ships with its own correctness test).

use crate::ty::ScalarTy;

// ── ExtInstBuiltin ────────────────────────────────────────────────────────────

/// The GLSL.std.450 extended-instruction builtin family (M3.2c).
///
/// M3.2c-exp ships exactly one variant: `Exp`. Each lowers to a single
/// `OpExtInst %f32 %glsl450_set <opcode> %x` in codegen.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExtInstBuiltin {
    /// `exp(x: f32) -> f32` — natural exponential via GLSL.std.450 `Exp` (=27).
    ///
    /// Lowers to `%r = OpExtInst %f32 %glsl450_set 27 %x`. Needs NO new SPIR-V
    /// capability (`GLSL.std.450` is an `OpExtInstImport`, not an `OpCapability`;
    /// `OpExtInst Exp` on an f32 operand needs only `Shader` + `Float32`, both
    /// baseline). Replaces the M3.2b Taylor `exp` approximation, faithful only for
    /// near-uniform attention (post-max logit gaps ≲ 0.7), with a true exp valid
    /// over the full real attention logit range.
    Exp,
}

impl ExtInstBuiltin {
    /// The canonical source-code name for this builtin.
    pub fn source_name(self) -> &'static str {
        match self {
            ExtInstBuiltin::Exp => "exp",
        }
    }

    /// Parse a source-level call name into an `ExtInstBuiltin`, or `None`.
    pub fn from_source_name(s: &str) -> Option<ExtInstBuiltin> {
        match s {
            "exp" => Some(ExtInstBuiltin::Exp),
            _     => None,
        }
    }

    /// Number of source-level arguments accepted by this builtin.
    ///
    /// - `Exp`: 1 (x: f32)
    pub fn arity(self) -> usize {
        match self {
            ExtInstBuiltin::Exp => 1,
        }
    }

    /// HIR return type for this builtin.
    ///
    /// - `Exp`: F32
    pub fn return_ty(self) -> ScalarTy {
        match self {
            ExtInstBuiltin::Exp => ScalarTy::F32,
        }
    }

    /// The GLSL.std.450 extended-instruction opcode (the literal passed to
    /// `OpExtInst`).
    ///
    /// - `Exp`: 27 (GLSL.std.450 `Exp`; `Log`=28, `Exp2`=29, `Sqrt`=31 are distinct
    ///   and reserved for future builtins).
    pub fn glsl450_opcode(self) -> u32 {
        match self {
            ExtInstBuiltin::Exp => 27,
        }
    }
}

// ── RESERVED_EXT_INST_BUILTIN_NAMES ───────────────────────────────────────────

/// Sorted-for-binary-search list of the GLSL.std.450 ext-inst builtin names.
///
/// Used by the HIR reserved-name check to reject `let exp = ...` etc.
/// MUST remain sorted lexicographically; verified by the unit test below.
pub const RESERVED_EXT_INST_BUILTIN_NAMES: &[&str] = &[
    "exp",
];

/// Returns `true` if `name` is a reserved GLSL.std.450 ext-inst builtin identifier.
///
/// Uses binary search on the sorted `RESERVED_EXT_INST_BUILTIN_NAMES` slice.
pub fn is_reserved_ext_inst_builtin(name: &str) -> bool {
    RESERVED_EXT_INST_BUILTIN_NAMES.binary_search(&name).is_ok()
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ext_inst_builtin_from_source_name_exp() {
        assert_eq!(ExtInstBuiltin::from_source_name("exp"), Some(ExtInstBuiltin::Exp));
    }

    #[test]
    fn ext_inst_builtin_from_source_name_unknown_is_none() {
        assert_eq!(ExtInstBuiltin::from_source_name("exp2"), None);
        assert_eq!(ExtInstBuiltin::from_source_name("log"), None);
        assert_eq!(ExtInstBuiltin::from_source_name(""), None);
        assert_eq!(ExtInstBuiltin::from_source_name("Exp"), None);
    }

    #[test]
    fn ext_inst_builtin_source_name_round_trip() {
        // One variant in M3.2c; assert the round-trip directly (extend the list as future
        // ext-inst builtins — exp2/log/sqrt — are added with their own tests).
        let builtin = ExtInstBuiltin::Exp;
        let name = builtin.source_name();
        assert_eq!(
            ExtInstBuiltin::from_source_name(name),
            Some(builtin),
            "round-trip failed for {name}"
        );
    }

    #[test]
    fn ext_inst_builtin_arity_correct() {
        assert_eq!(ExtInstBuiltin::Exp.arity(), 1, "Exp arity must be 1");
    }

    #[test]
    fn ext_inst_builtin_return_ty_correct() {
        assert_eq!(ExtInstBuiltin::Exp.return_ty(), ScalarTy::F32);
    }

    #[test]
    fn ext_inst_builtin_glsl450_opcode_correct() {
        // GLSL.std.450 Exp = 27 (Log=28, Exp2=29, Sqrt=31 are distinct).
        assert_eq!(ExtInstBuiltin::Exp.glsl450_opcode(), 27);
    }

    #[test]
    fn is_reserved_ext_inst_builtin_positive() {
        assert!(is_reserved_ext_inst_builtin("exp"));
    }

    #[test]
    fn is_reserved_ext_inst_builtin_negative() {
        assert!(!is_reserved_ext_inst_builtin("exp2"));
        assert!(!is_reserved_ext_inst_builtin("log"));
        assert!(!is_reserved_ext_inst_builtin("band"));
        assert!(!is_reserved_ext_inst_builtin(""));
        assert!(!is_reserved_ext_inst_builtin("EXP"));
    }

    #[test]
    fn reserved_ext_inst_builtin_names_is_sorted_for_binary_search() {
        let mut sorted = RESERVED_EXT_INST_BUILTIN_NAMES.to_vec();
        sorted.sort_unstable();
        assert_eq!(
            RESERVED_EXT_INST_BUILTIN_NAMES,
            sorted.as_slice(),
            "RESERVED_EXT_INST_BUILTIN_NAMES must be sorted lexicographically"
        );
        for name in RESERVED_EXT_INST_BUILTIN_NAMES {
            assert!(
                RESERVED_EXT_INST_BUILTIN_NAMES.binary_search(name).is_ok(),
                "{name} not found via binary_search"
            );
        }
        assert!(
            RESERVED_EXT_INST_BUILTIN_NAMES.binary_search(&"band").is_err(),
            "band must NOT be found in RESERVED_EXT_INST_BUILTIN_NAMES"
        );
    }
}
