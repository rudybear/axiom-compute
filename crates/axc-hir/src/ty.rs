//! Scalar type representation at HIR level.
//!
//! Separate module so it can be reused by buffer/vector types in M1.2+.
//! `ScalarTy` is the only type in scope for M1.1; `Ty` wraps it for extensibility.
//! M2.1 adds `F16` (IEEE 754 binary16) for cooperative-matrix element support.

/// The set of scalar types supported in M1.1+.
///
/// `F16` is added in M2.1 as a cooperative-matrix element type and buffer element type.
/// `bf16` remains deferred to a later milestone (not in M2.1 scope).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ScalarTy {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    F32,
    F64,
    Bool,
}

impl ScalarTy {
    /// True for all integer types (signed and unsigned).
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            ScalarTy::I8 | ScalarTy::I16 | ScalarTy::I32 | ScalarTy::I64
            | ScalarTy::U8 | ScalarTy::U16 | ScalarTy::U32 | ScalarTy::U64
        )
    }

    /// True for signed integer types only (I8, I16, I32, I64).
    pub fn is_signed_integer(&self) -> bool {
        matches!(self, ScalarTy::I8 | ScalarTy::I16 | ScalarTy::I32 | ScalarTy::I64)
    }

    /// True for unsigned integer types only (U8, U16, U32, U64).
    pub fn is_unsigned_integer(&self) -> bool {
        matches!(self, ScalarTy::U8 | ScalarTy::U16 | ScalarTy::U32 | ScalarTy::U64)
    }

    /// True for floating-point types (F16, F32, F64).
    pub fn is_float(&self) -> bool {
        matches!(self, ScalarTy::F16 | ScalarTy::F32 | ScalarTy::F64)
    }

    /// True for the boolean type.
    pub fn is_bool(&self) -> bool {
        matches!(self, ScalarTy::Bool)
    }

    /// Bit width of the type.
    ///
    /// Bool is 1 bit in the abstract model; SPIR-V uses a distinct type for it.
    pub fn bit_width(&self) -> u32 {
        match self {
            ScalarTy::I8  | ScalarTy::U8  => 8,
            ScalarTy::I16 | ScalarTy::U16 | ScalarTy::F16 => 16,
            ScalarTy::I32 | ScalarTy::U32 | ScalarTy::F32 => 32,
            ScalarTy::I64 | ScalarTy::U64 | ScalarTy::F64 => 64,
            ScalarTy::Bool => 1,
        }
    }

    /// Human-readable name matching the source-level type keyword.
    pub fn display_name(&self) -> &'static str {
        match self {
            ScalarTy::I8   => "i8",
            ScalarTy::I16  => "i16",
            ScalarTy::I32  => "i32",
            ScalarTy::I64  => "i64",
            ScalarTy::U8   => "u8",
            ScalarTy::U16  => "u16",
            ScalarTy::U32  => "u32",
            ScalarTy::U64  => "u64",
            ScalarTy::F16  => "f16",
            ScalarTy::F32  => "f32",
            ScalarTy::F64  => "f64",
            ScalarTy::Bool => "bool",
        }
    }

    /// Range for integer types: returns (min_i128, max_i128).
    ///
    /// Signed: [-2^(W-1), 2^(W-1) - 1]. Unsigned: [0, 2^W - 1].
    /// Returns None for non-integer types.
    pub fn int_range(&self) -> Option<(i128, i128)> {
        match self {
            ScalarTy::I8   => Some((i8::MIN as i128, i8::MAX as i128)),
            ScalarTy::I16  => Some((i16::MIN as i128, i16::MAX as i128)),
            ScalarTy::I32  => Some((i32::MIN as i128, i32::MAX as i128)),
            ScalarTy::I64  => Some((i64::MIN as i128, i64::MAX as i128)),
            ScalarTy::U8   => Some((0, u8::MAX as i128)),
            ScalarTy::U16  => Some((0, u16::MAX as i128)),
            ScalarTy::U32  => Some((0, u32::MAX as i128)),
            ScalarTy::U64  => Some((0, u64::MAX as i128)),
            _ => None,
        }
    }
}

impl std::fmt::Display for ScalarTy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.display_name())
    }
}

/// A typed value produced by `fit_int_literal`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntLiteralValue {
    /// The resolved type.
    pub ty: ScalarTy,
    /// Bits of the value in the target type's width, zero-extended to u64.
    /// Two's-complement for signed types (e.g., -1i32 is stored as 0xFFFF_FFFF).
    pub bits: u64,
}

/// A typed value produced by `fit_float_literal`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FloatLiteralValue {
    /// The resolved type (F32 or F64).
    pub ty: ScalarTy,
    /// IEEE 754 bits: f32.to_bits() for F32; f64.to_bits() for F64.
    pub bits: u64,
}

/// Error from range-checking a literal against a target type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LiteralRangeErr {
    IntegerOutOfRange { value: i128, target: ScalarTy },
    FloatTargetInvalid { target: ScalarTy },
    FloatNonFinite,
    /// The f64 value converts to ±infinity in IEEE-754 binary16.
    ///
    /// M2.1: f16 has a much narrower range than f32 (max finite ≈ 65504).
    /// Values that round to +inf/-inf under bin16 conversion are rejected.
    FloatOutOfRangeForF16 { value: f64, target: ScalarTy },
    /// The f64 value is non-zero but rounds to zero (or a subnormal that has
    /// zero bin16 representation), causing silent precision loss in f16.
    ///
    /// M2.1 AT-630: reject `1.0e-6f16` etc. to prevent silent underflow.
    FloatSubnormalPrecisionLoss { value: f64, target: ScalarTy },
}

/// Fit an integer literal `value` into `ty`, checking the type's range.
///
/// # Returns
/// - `Ok(IntLiteralValue)` when the value fits.
/// - `Err(LiteralRangeErr::IntegerOutOfRange)` when it does not.
/// - `Err(LiteralRangeErr::FloatTargetInvalid)` when `ty` is not an integer type.
pub fn fit_int_literal(value: i128, ty: ScalarTy) -> Result<IntLiteralValue, LiteralRangeErr> {
    let (min_val, max_val) = ty.int_range()
        .ok_or(LiteralRangeErr::FloatTargetInvalid { target: ty })?;

    if value < min_val || value > max_val {
        return Err(LiteralRangeErr::IntegerOutOfRange { value, target: ty });
    }

    // Store as u64 two's-complement, width-masked.
    let bits: u64 = match ty.bit_width() {
        8  => (value as u8)  as u64,
        16 => (value as u16) as u64,
        32 => (value as u32) as u64,
        64 => value as u64,
        _  => unreachable!("unexpected integer bit width"),
    };

    Ok(IntLiteralValue { ty, bits })
}

/// Fit a float literal `value` into `ty`.
///
/// Checks that `ty` is F16, F32, or F64, and that `value` is finite.
/// For F16: also rejects values that overflow to infinity or that silently
/// underflow to zero from a non-zero input (AT-630).
pub fn fit_float_literal(value: f64, ty: ScalarTy) -> Result<FloatLiteralValue, LiteralRangeErr> {
    if !ty.is_float() {
        return Err(LiteralRangeErr::FloatTargetInvalid { target: ty });
    }
    if !value.is_finite() {
        return Err(LiteralRangeErr::FloatNonFinite);
    }
    let bits: u64 = match ty {
        ScalarTy::F16 => {
            // Use the half crate for bin16 conversion.
            let h = half::f16::from_f64(value);
            // Reject values that overflowed to ±infinity under bin16.
            if h.is_infinite() {
                return Err(LiteralRangeErr::FloatOutOfRangeForF16 { value, target: ty });
            }
            // Reject non-zero inputs that silently underflow to zero in bin16 (AT-630).
            // A non-zero f64 producing a bin16 representation of zero means we lost ALL
            // information — this is a silent precision loss that must be explicit.
            if value != 0.0 && h.to_bits() == 0 {
                return Err(LiteralRangeErr::FloatSubnormalPrecisionLoss { value, target: ty });
            }
            h.to_bits() as u64
        }
        ScalarTy::F32 => (value as f32).to_bits() as u64,
        ScalarTy::F64 => value.to_bits(),
        _ => unreachable!("fit_float_literal: unexpected non-float ScalarTy"),
    };
    Ok(FloatLiteralValue { ty, bits })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_ty_predicates() {
        assert!(ScalarTy::I32.is_integer());
        assert!(ScalarTy::I32.is_signed_integer());
        assert!(!ScalarTy::I32.is_unsigned_integer());
        assert!(ScalarTy::U64.is_unsigned_integer());
        assert!(!ScalarTy::U64.is_signed_integer());
        assert!(ScalarTy::F32.is_float());
        assert!(!ScalarTy::I64.is_float());
        assert!(ScalarTy::Bool.is_bool());
        assert!(!ScalarTy::I32.is_bool());
    }

    #[test]
    fn scalar_ty_bit_widths() {
        assert_eq!(ScalarTy::I32.bit_width(), 32);
        assert_eq!(ScalarTy::U64.bit_width(), 64);
        assert_eq!(ScalarTy::F32.bit_width(), 32);
        assert_eq!(ScalarTy::F64.bit_width(), 64);
        assert_eq!(ScalarTy::Bool.bit_width(), 1);
    }

    // ── M2.1: F16 tests ───────────────────────────────────────────────────────

    #[test]
    fn scalar_ty_f16_is_float_not_integer() {
        assert!(ScalarTy::F16.is_float(), "F16 must be a float type");
        assert!(!ScalarTy::F16.is_integer(), "F16 is not an integer");
        assert!(!ScalarTy::F16.is_signed_integer(), "F16 is not signed integer");
        assert!(!ScalarTy::F16.is_unsigned_integer(), "F16 is not unsigned integer");
        assert!(!ScalarTy::F16.is_bool(), "F16 is not bool");
    }

    #[test]
    fn scalar_ty_f16_bit_width_16() {
        assert_eq!(ScalarTy::F16.bit_width(), 16, "F16 must have bit width 16");
    }

    #[test]
    fn scalar_ty_f16_display_name() {
        assert_eq!(ScalarTy::F16.display_name(), "f16", "F16 display name must be \"f16\"");
    }

    #[test]
    fn fit_float_literal_f16_happy() {
        // 1.5 is exactly representable in bin16.
        let v = fit_float_literal(1.5, ScalarTy::F16).expect("1.5 should fit in f16");
        assert_eq!(v.ty, ScalarTy::F16);
        let expected_bits: u64 = half::f16::from_f64(1.5).to_bits() as u64;
        assert_eq!(v.bits, expected_bits, "bits must match half::f16::from_f64(1.5)");
    }

    #[test]
    fn fit_float_literal_f16_overflow_rejected() {
        // 1.0e10 overflows to infinity in bin16.
        let e = fit_float_literal(1.0e10, ScalarTy::F16).unwrap_err();
        assert!(
            matches!(e, LiteralRangeErr::FloatOutOfRangeForF16 { .. }),
            "expected FloatOutOfRangeForF16, got {e:?}"
        );
    }

    /// AT-630: f16 subnormal underflow.
    #[test]
    fn fit_float_literal_f16_subnormal_underflow_rejected() {
        // 1.0e-10 is smaller than the smallest f16 subnormal (5.96e-8) and rounds
        // to zero in bin16 — silent underflow, must be rejected.
        let e = fit_float_literal(1.0e-10, ScalarTy::F16).unwrap_err();
        assert!(
            matches!(e, LiteralRangeErr::FloatSubnormalPrecisionLoss { .. }),
            "expected FloatSubnormalPrecisionLoss for 1.0e-10, got {e:?}"
        );

        // Exact zero is always fine.
        let v = fit_float_literal(0.0, ScalarTy::F16).expect("0.0 must be accepted for f16");
        assert_eq!(v.bits, 0, "0.0 in f16 must have bits == 0");

        // 5.96e-8 is approximately the smallest f16 subnormal and has non-zero bin16 bits.
        // half::f16::MIN_POSITIVE_SUBNORMAL is 5.96046e-8 (2^-24).
        let small = half::f16::MIN_POSITIVE_SUBNORMAL.to_f64();
        let v2 = fit_float_literal(small, ScalarTy::F16)
            .expect("smallest f16 subnormal (non-zero bin16 bits) must be accepted");
        assert_ne!(v2.bits, 0, "smallest f16 subnormal must have non-zero bin16 bits");
    }

    #[test]
    fn fit_int_literal_happy() {
        let v = fit_int_literal(42, ScalarTy::I32).expect("should fit");
        assert_eq!(v.ty, ScalarTy::I32);
        assert_eq!(v.bits, 42);
    }

    #[test]
    fn fit_int_literal_i32_min() {
        // i32::MIN = -2147483648 — edge case for the §4.2a peephole
        let v = fit_int_literal(i32::MIN as i128, ScalarTy::I32).expect("i32::MIN should fit");
        assert_eq!(v.bits, 0x8000_0000_u64);
    }

    #[test]
    fn fit_int_literal_out_of_range() {
        let e = fit_int_literal(9999999999, ScalarTy::I32).unwrap_err();
        assert!(matches!(e, LiteralRangeErr::IntegerOutOfRange { value: 9999999999, target: ScalarTy::I32 }));
    }

    #[test]
    fn fit_int_literal_u32_max() {
        let v = fit_int_literal(u32::MAX as i128, ScalarTy::U32).expect("u32::MAX should fit");
        assert_eq!(v.bits, u32::MAX as u64);
    }

    #[test]
    fn fit_int_literal_negative_fails_for_unsigned() {
        let e = fit_int_literal(-1, ScalarTy::U32).unwrap_err();
        assert!(matches!(e, LiteralRangeErr::IntegerOutOfRange { value: -1, target: ScalarTy::U32 }));
    }

    #[test]
    fn fit_float_literal_f32() {
        // Use 1.5 (exact in binary float; avoids clippy::approx_constant on PI-adjacent values)
        let v = fit_float_literal(1.5, ScalarTy::F32).expect("should fit");
        assert_eq!(v.ty, ScalarTy::F32);
        let expected_bits: u64 = (1.5f32).to_bits() as u64;
        assert_eq!(v.bits, expected_bits);
    }

    #[test]
    fn fit_float_literal_f64() {
        // Use 1.5 (exact in binary float; avoids clippy::approx_constant on PI-adjacent values)
        let v = fit_float_literal(1.5, ScalarTy::F64).expect("should fit");
        assert_eq!(v.ty, ScalarTy::F64);
        assert_eq!(v.bits, 1.5f64.to_bits());
    }

    #[test]
    fn fit_float_literal_non_float_type_rejected() {
        let e = fit_float_literal(1.0, ScalarTy::I32).unwrap_err();
        assert!(matches!(e, LiteralRangeErr::FloatTargetInvalid { target: ScalarTy::I32 }));
    }
}
