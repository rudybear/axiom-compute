//! HIR data types for per-invocation (thread-private) local arrays (M3.20).
//!
//! `array[T,N]` is a kernel-local, per-invocation array: declared in the kernel
//! body, allocated as a Function-class `OpVariable` in SPIR-V, private to one
//! invocation. It is the private-memory sibling of `shared[T,N]` (M3.2) — see
//! `crate::shared` for the workgroup-shared analogue this module mirrors.
//!
//! This module contains pure HIR data; no codegen dependency — mirrors shared.rs
//! discipline.

use axc_lexer::Span;
use crate::ty::ScalarTy;

/// Maximum number of elements in a local array (compile-time ceiling, not device limit).
///
/// Private memory is not a Vulkan-limited resource, so this is a compiler-chosen
/// ceiling independent of element size — the aggregate byte cap (`MAX_LOCAL_ARRAY_BYTES`)
/// binds first in practice for any realistic element type.
pub const MAX_LOCAL_ARRAY_ELEMS: u32 = 65536;

/// Maximum total local-array bytes enforced at compile time, PER KERNEL.
///
/// Any single kernel whose aggregate `sum(total_byte_size())` over all declared
/// local arrays exceeds this is rejected with `TypecheckError::LocalArrayTooLarge`.
/// 16x tighter than `MAX_SHARED_BYTES` (65536) because private arrays are
/// replicated PER-THREAD (not per-workgroup) — see M3.20 spec §6.
pub const MAX_LOCAL_ARRAY_BYTES: u64 = 4096;

/// Non-blocking spill-risk advisory threshold, in bytes.
///
/// Kernels whose aggregate local-array bytes exceed this (but stay under
/// `MAX_LOCAL_ARRAY_BYTES`) emit `HirWarning::LocalArrayMaySpill` — register-file
/// pressure / local-memory spill risk (M3.12 lesson; see M3.20 spec §6, §15).
pub const LOCAL_ARRAY_SPILL_ADVISORY_BYTES: u64 = 1024;

/// Opaque identifier for a local array within a kernel body.
///
/// Monotonically assigned from 0 per kernel; used as the key into the
/// local-array tables in codegen and typecheck.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LocalArrayId(pub u32);

/// The type of a local array.
///
/// Corresponds to `array[elem, N]` in source where `elem` is a scalar type and
/// `N` is a compile-time positive integer literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalArrayTy {
    /// Element scalar type (Bool is not allowed).
    pub elem: ScalarTy,
    /// Number of elements (must be >= 1 and <= MAX_LOCAL_ARRAY_ELEMS).
    pub len: u32,
}

impl LocalArrayTy {
    /// Byte size of a single element.
    ///
    /// Matches `SharedTy::elem_byte_size` / `BufferTy::elem_byte_size` for consistency.
    pub fn elem_byte_size(&self) -> u32 {
        match self.elem.bit_width() {
            8  => 1,
            16 => 2,
            32 => 4,
            64 => 8,
            w  => panic!("unsupported local-array elem bit width {w}"),
        }
    }

    /// Total byte size of the array: `len as u64 * elem_byte_size as u64`.
    ///
    /// Returns `u64` to avoid overflow for large local arrays.
    pub fn total_byte_size(&self) -> u64 {
        (self.len as u64) * (self.elem_byte_size() as u64)
    }

    /// True if this local-array element type requires the `Float16` SPIR-V capability.
    pub fn needs_float16(&self) -> bool {
        self.elem == ScalarTy::F16
    }

    /// True if this local-array element type requires the `Int8` SPIR-V capability.
    pub fn needs_int8(&self) -> bool {
        matches!(self.elem, ScalarTy::I8 | ScalarTy::U8)
    }

    /// True if this local-array element type requires the `Int16` SPIR-V capability.
    pub fn needs_int16(&self) -> bool {
        matches!(self.elem, ScalarTy::I16 | ScalarTy::U16)
    }

    /// True if this local-array element type requires the `Int64` SPIR-V capability.
    pub fn needs_int64(&self) -> bool {
        matches!(self.elem, ScalarTy::I64 | ScalarTy::U64)
    }

    /// True if this local-array element type requires the `Float64` SPIR-V capability.
    pub fn needs_float64(&self) -> bool {
        self.elem == ScalarTy::F64
    }
}

/// A validated local-array declaration.
///
/// Each declaration in a kernel body produces one `LocalArrayDecl` with a unique
/// `LocalArrayId`. The declarations are collected into `KernelBodyTyped.local_arrays`
/// in source order and lowered to `OpVariable Function` in SPIR-V (in the function
/// entry-block prelude — see `crate::expr::HirStmt::LocalArrayDeclMarker`).
#[derive(Debug, Clone)]
pub struct LocalArrayDecl {
    /// Unique identifier within the kernel (monotonic from 0).
    pub id: LocalArrayId,
    /// Source-level name (e.g. `"hist"`).
    pub name: String,
    /// Element type and length.
    pub ty: LocalArrayTy,
    /// Source span of the full declaration for error messages.
    pub span: Span,
}

/// Returns `true` if `ty` is an allowed local-array element type.
///
/// All scalar types except `Bool` are allowed — identical rule to `shared`'s
/// `is_allowed_shared_element` (Bool has no stable Vulkan memory representation).
pub fn is_allowed_local_element(ty: ScalarTy) -> bool {
    !matches!(ty, ScalarTy::Bool)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_array_ty_elem_byte_size_u32() {
        let ty: LocalArrayTy = LocalArrayTy { elem: ScalarTy::U32, len: 8 };
        assert_eq!(ty.elem_byte_size(), 4);
    }

    #[test]
    fn local_array_ty_total_byte_size_u32_8() {
        let ty: LocalArrayTy = LocalArrayTy { elem: ScalarTy::U32, len: 8 };
        assert_eq!(ty.total_byte_size(), 32);
    }

    #[test]
    fn local_array_ty_total_byte_size_f16_1024() {
        let ty: LocalArrayTy = LocalArrayTy { elem: ScalarTy::F16, len: 1024 };
        assert_eq!(ty.total_byte_size(), 2048);
    }

    #[test]
    fn local_array_ty_needs_float16_for_f16() {
        assert!(LocalArrayTy { elem: ScalarTy::F16, len: 1 }.needs_float16());
        assert!(!LocalArrayTy { elem: ScalarTy::F32, len: 1 }.needs_float16());
    }

    #[test]
    fn local_array_ty_needs_int8_for_i8_u8() {
        assert!(LocalArrayTy { elem: ScalarTy::I8, len: 1 }.needs_int8());
        assert!(LocalArrayTy { elem: ScalarTy::U8, len: 1 }.needs_int8());
        assert!(!LocalArrayTy { elem: ScalarTy::I32, len: 1 }.needs_int8());
    }

    #[test]
    fn is_allowed_local_element_rejects_bool() {
        assert!(!is_allowed_local_element(ScalarTy::Bool));
    }

    #[test]
    fn is_allowed_local_element_accepts_all_numeric() {
        for ty in [
            ScalarTy::I8, ScalarTy::U8, ScalarTy::I16, ScalarTy::U16,
            ScalarTy::I32, ScalarTy::U32, ScalarTy::I64, ScalarTy::U64,
            ScalarTy::F16, ScalarTy::F32, ScalarTy::F64,
        ] {
            assert!(is_allowed_local_element(ty), "{ty:?} should be allowed");
        }
    }

    #[test]
    fn max_local_array_bytes_is_4096() {
        assert_eq!(MAX_LOCAL_ARRAY_BYTES, 4096);
    }

    #[test]
    fn local_array_spill_advisory_bytes_is_1024() {
        assert_eq!(LOCAL_ARRAY_SPILL_ADVISORY_BYTES, 1024);
    }

    #[test]
    fn local_array_id_ord() {
        assert!(LocalArrayId(0) < LocalArrayId(1));
    }
}
