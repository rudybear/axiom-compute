//! HIR data types for the four Q4_0-path builtins (M2.5).
//!
//! This module mirrors `crates/axc-hir/src/coopmat.rs` in layout and discipline.
//! It contains pure HIR data: no codegen dependency, no type-checking logic.
//!
//! These primitives are named after their use in Q4_0 dequantization, but they
//! are general byte-access and conversion primitives that Q4_K_M (M2.6) will
//! also reuse verbatim.
//!
//! The four builtins are:
//!   - `ptr_read_u8_zext`  — byte load + zero-extend to u32
//!   - `ptr_read_u16_zext` — two-byte little-endian load + zero-extend to u32
//!   - `f16_bits_to_f32`   — reinterpret low 16 bits of u32 as f16, widen to f32
//!   - `f32_from_u32`      — convert u32 to f32 via IEEE-754 RNE

use crate::ty::ScalarTy;

// ── Q4_0Builtin ───────────────────────────────────────────────────────────────

/// The four Q4_0-path builtin call names (M2.5).
///
/// These are NOT specific to Q4_0 — they are general byte-access and conversion
/// primitives that happen to land in M2.5 as the minimum viable set for Q4_0.
/// Q4_K_M (M2.6) will reuse all four verbatim.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Q4_0Builtin {
    /// Read one byte from a buffer[u8] and zero-extend to u32.
    ///
    /// Lowers to `OpAccessChain` + `OpLoad(u8)` + `OpUConvert(u32)`.
    /// Required capabilities: `Int8`, `StorageBuffer8BitAccess`.
    /// Required extension: `SPV_KHR_8bit_storage`.
    PtrReadU8Zext,
    /// Read two consecutive bytes (little-endian) from a buffer[u8] and return
    /// u32 with the low 16 bits set from those two bytes.
    ///
    /// Lowers to two `OpAccessChain + OpLoad(u8) + OpUConvert(u32)` sequences,
    /// then `OpShiftLeftLogical(hi, 8)` + `OpBitwiseOr(lo, shifted_hi)`.
    /// Does NOT issue a native u16 SSBO load — avoids alignment requirement.
    /// Required capabilities: same as PtrReadU8Zext (`Int8` + `StorageBuffer8BitAccess`).
    PtrReadU16Zext,
    /// Reinterpret the low 16 bits of a u32 as IEEE-754 binary16 and widen to f32.
    ///
    /// Lowers to `OpUConvert(u16)` + `OpBitcast(f16)` + `OpFConvert(f32)`.
    /// Required capabilities: `Int16`, `Float16`.
    /// Note: `StorageBuffer16BitAccess` is NOT required — the f16 is synthesized
    /// in-register from integer bits, never SSBO-loaded as a native f16 scalar.
    F16BitsToF32,
    /// Convert u32 to f32 via IEEE-754 round-to-nearest-even.
    ///
    /// Lowers to `OpConvertUToF`.
    /// No new capability side-effects (f32 and u32 are the baseline).
    F32FromU32,
}

impl Q4_0Builtin {
    /// The canonical source-code name for this builtin.
    pub fn source_name(self) -> &'static str {
        match self {
            Q4_0Builtin::PtrReadU8Zext  => "ptr_read_u8_zext",
            Q4_0Builtin::PtrReadU16Zext => "ptr_read_u16_zext",
            Q4_0Builtin::F16BitsToF32   => "f16_bits_to_f32",
            Q4_0Builtin::F32FromU32     => "f32_from_u32",
        }
    }

    /// Parse a source-level call name into a `Q4_0Builtin`, or `None`.
    ///
    /// Uses binary search on `RESERVED_Q4_0_BUILTIN_NAMES` for O(log n) lookup.
    pub fn from_source_name(s: &str) -> Option<Q4_0Builtin> {
        match s {
            "ptr_read_u8_zext"  => Some(Q4_0Builtin::PtrReadU8Zext),
            "ptr_read_u16_zext" => Some(Q4_0Builtin::PtrReadU16Zext),
            "f16_bits_to_f32"   => Some(Q4_0Builtin::F16BitsToF32),
            "f32_from_u32"      => Some(Q4_0Builtin::F32FromU32),
            _                   => None,
        }
    }

    /// Number of source-level arguments accepted by this builtin.
    ///
    /// - `PtrReadU8Zext`:  2 (buf, byte_offset)
    /// - `PtrReadU16Zext`: 2 (buf, byte_offset)
    /// - `F16BitsToF32`:   1 (bits: u32)
    /// - `F32FromU32`:     1 (u: u32)
    pub fn arity(self) -> usize {
        match self {
            Q4_0Builtin::PtrReadU8Zext  => 2,
            Q4_0Builtin::PtrReadU16Zext => 2,
            Q4_0Builtin::F16BitsToF32   => 1,
            Q4_0Builtin::F32FromU32     => 1,
        }
    }

    /// HIR return type for this builtin.
    ///
    /// - `PtrReadU8Zext`:  U32 (byte zero-extended to 32 bits)
    /// - `PtrReadU16Zext`: U32 (two bytes zero-extended to 32 bits)
    /// - `F16BitsToF32`:   F32 (widened from f16)
    /// - `F32FromU32`:     F32 (u32 converted to f32)
    pub fn return_ty(self) -> ScalarTy {
        match self {
            Q4_0Builtin::PtrReadU8Zext  => ScalarTy::U32,
            Q4_0Builtin::PtrReadU16Zext => ScalarTy::U32,
            Q4_0Builtin::F16BitsToF32   => ScalarTy::F32,
            Q4_0Builtin::F32FromU32     => ScalarTy::F32,
        }
    }
}

// ── RESERVED_Q4_0_BUILTIN_NAMES ──────────────────────────────────────────────

/// Sorted-for-binary-search list of the four Q4_0-path builtin names.
///
/// Used by the HIR reserved-name check to reject `let ptr_read_u8_zext = ...` etc.
/// MUST remain sorted lexicographically; verified by the unit test below.
pub const RESERVED_Q4_0_BUILTIN_NAMES: &[&str] = &[
    "f16_bits_to_f32",
    "f32_from_u32",
    "ptr_read_u16_zext",
    "ptr_read_u8_zext",
];

/// Returns `true` if `name` is a reserved Q4_0-path builtin identifier.
///
/// Uses binary search on the sorted `RESERVED_Q4_0_BUILTIN_NAMES` slice for O(log n).
pub fn is_reserved_q4_0_builtin(name: &str) -> bool {
    RESERVED_Q4_0_BUILTIN_NAMES.binary_search(&name).is_ok()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_0_builtin_from_source_name_ptr_read_u8_zext() {
        assert_eq!(
            Q4_0Builtin::from_source_name("ptr_read_u8_zext"),
            Some(Q4_0Builtin::PtrReadU8Zext)
        );
    }

    #[test]
    fn q4_0_builtin_from_source_name_ptr_read_u16_zext() {
        assert_eq!(
            Q4_0Builtin::from_source_name("ptr_read_u16_zext"),
            Some(Q4_0Builtin::PtrReadU16Zext)
        );
    }

    #[test]
    fn q4_0_builtin_from_source_name_f16_bits_to_f32() {
        assert_eq!(
            Q4_0Builtin::from_source_name("f16_bits_to_f32"),
            Some(Q4_0Builtin::F16BitsToF32)
        );
    }

    #[test]
    fn q4_0_builtin_from_source_name_f32_from_u32() {
        assert_eq!(
            Q4_0Builtin::from_source_name("f32_from_u32"),
            Some(Q4_0Builtin::F32FromU32)
        );
    }

    #[test]
    fn q4_0_builtin_from_source_name_unknown_is_none() {
        assert_eq!(Q4_0Builtin::from_source_name("band"), None);
        assert_eq!(Q4_0Builtin::from_source_name(""), None);
        assert_eq!(Q4_0Builtin::from_source_name("PtrReadU8Zext"), None);
    }

    #[test]
    fn q4_0_builtin_source_name_round_trip() {
        for &builtin in &[
            Q4_0Builtin::PtrReadU8Zext,
            Q4_0Builtin::PtrReadU16Zext,
            Q4_0Builtin::F16BitsToF32,
            Q4_0Builtin::F32FromU32,
        ] {
            let name = builtin.source_name();
            let parsed = Q4_0Builtin::from_source_name(name);
            assert_eq!(parsed, Some(builtin), "round-trip failed for {name}");
        }
    }

    #[test]
    fn q4_0_builtin_arity_correct() {
        assert_eq!(Q4_0Builtin::PtrReadU8Zext.arity(), 2, "PtrReadU8Zext arity must be 2");
        assert_eq!(Q4_0Builtin::PtrReadU16Zext.arity(), 2, "PtrReadU16Zext arity must be 2");
        assert_eq!(Q4_0Builtin::F16BitsToF32.arity(), 1, "F16BitsToF32 arity must be 1");
        assert_eq!(Q4_0Builtin::F32FromU32.arity(), 1, "F32FromU32 arity must be 1");
    }

    #[test]
    fn q4_0_builtin_return_ty_correct() {
        assert_eq!(Q4_0Builtin::PtrReadU8Zext.return_ty(), ScalarTy::U32);
        assert_eq!(Q4_0Builtin::PtrReadU16Zext.return_ty(), ScalarTy::U32);
        assert_eq!(Q4_0Builtin::F16BitsToF32.return_ty(), ScalarTy::F32);
        assert_eq!(Q4_0Builtin::F32FromU32.return_ty(), ScalarTy::F32);
    }

    #[test]
    fn is_reserved_q4_0_builtin_positive() {
        assert!(is_reserved_q4_0_builtin("ptr_read_u8_zext"));
        assert!(is_reserved_q4_0_builtin("ptr_read_u16_zext"));
        assert!(is_reserved_q4_0_builtin("f16_bits_to_f32"));
        assert!(is_reserved_q4_0_builtin("f32_from_u32"));
    }

    #[test]
    fn is_reserved_q4_0_builtin_negative() {
        assert!(!is_reserved_q4_0_builtin("band"));
        assert!(!is_reserved_q4_0_builtin("gid"));
        assert!(!is_reserved_q4_0_builtin("coopmat_load"));
        assert!(!is_reserved_q4_0_builtin(""));
        assert!(!is_reserved_q4_0_builtin("PTR_READ_U8_ZEXT"));
    }

    #[test]
    fn reserved_q4_0_builtin_names_is_sorted_for_binary_search() {
        // The constant MUST be sorted for binary_search to work correctly.
        let mut sorted = RESERVED_Q4_0_BUILTIN_NAMES.to_vec();
        sorted.sort_unstable();
        assert_eq!(
            RESERVED_Q4_0_BUILTIN_NAMES,
            sorted.as_slice(),
            "RESERVED_Q4_0_BUILTIN_NAMES must be sorted lexicographically"
        );
        // All four names are recoverable via binary_search.
        for name in RESERVED_Q4_0_BUILTIN_NAMES {
            assert!(
                RESERVED_Q4_0_BUILTIN_NAMES.binary_search(name).is_ok(),
                "{name} not found via binary_search"
            );
        }
        // An unrelated name is not found.
        assert!(
            RESERVED_Q4_0_BUILTIN_NAMES.binary_search(&"band").is_err(),
            "band must NOT be found in RESERVED_Q4_0_BUILTIN_NAMES"
        );
    }
}
