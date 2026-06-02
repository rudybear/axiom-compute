//! HIR data types for workgroup-shared arrays (M3.2 PART A).
//!
//! `shared[T,N]` is a kernel-wide workgroup-local array: declared in the kernel
//! body, allocated as a Workgroup-class `OpVariable` in SPIR-V, shared by all
//! invocations in the workgroup.
//!
//! This module contains pure HIR data; no codegen dependency — mirrors buffer.rs
//! and subgroup.rs discipline.

use axc_lexer::Span;
use crate::ty::ScalarTy;

/// Maximum number of elements in a shared array (compile-time ceiling, not device limit).
///
/// The device's `maxComputeSharedMemorySize` is the runtime preflight gate.
/// This constant bounds the per-array element count independent of element size.
pub const MAX_SHARED_ELEMS: u32 = 65536;

/// Minimum guaranteed shared memory bytes on Vulkan portable devices (16 KiB).
///
/// Kernels using more than this emit `HirWarning::SharedMemoryExceedsPortableMinimum`.
pub const PORTABLE_MIN_SHARED_BYTES: u32 = 16384;

/// Maximum total shared memory bytes enforced at compile time.
///
/// Any single kernel whose aggregate `sum(total_byte_size())` exceeds this is rejected
/// with `HirError::SharedMemoryTooLarge` — independent of device limits.
pub const MAX_SHARED_BYTES: u64 = 65536;

/// Opaque identifier for a workgroup-shared array within a kernel body.
///
/// Monotonically assigned from 0 per kernel; used as the key into the
/// shared-array tables in codegen and typecheck.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SharedId(pub u32);

/// The type of a workgroup-shared array.
///
/// Corresponds to `shared[elem, N]` in source where `elem` is a scalar type and
/// `N` is a compile-time positive integer literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedTy {
    /// Element scalar type (Bool is not allowed).
    pub elem: ScalarTy,
    /// Number of elements (must be >= 1 and <= MAX_SHARED_ELEMS).
    pub len: u32,
}

impl SharedTy {
    /// Byte size of a single element.
    ///
    /// Matches `BufferTy::elem_byte_size` for consistency.
    pub fn elem_byte_size(&self) -> u32 {
        match self.elem.bit_width() {
            8  => 1,
            16 => 2,
            32 => 4,
            64 => 8,
            w  => panic!("unsupported shared elem bit width {w}"),
        }
    }

    /// Total byte size of the array: `len as u64 * elem_byte_size as u64`.
    ///
    /// Returns `u64` to avoid overflow for large shared arrays.
    pub fn total_byte_size(&self) -> u64 {
        (self.len as u64) * (self.elem_byte_size() as u64)
    }

    /// True if this shared element type requires the `Float16` SPIR-V capability.
    ///
    /// This is NOT StorageBuffer16BitAccess (which gates the StorageBuffer class).
    /// Workgroup arrays only need the Float16 type-level capability.
    pub fn needs_float16(&self) -> bool {
        self.elem == ScalarTy::F16
    }

    /// True if this shared element type requires the `Int8` SPIR-V capability.
    pub fn needs_int8(&self) -> bool {
        matches!(self.elem, ScalarTy::I8 | ScalarTy::U8)
    }

    /// True if this shared element type requires the `Int16` SPIR-V capability.
    pub fn needs_int16(&self) -> bool {
        matches!(self.elem, ScalarTy::I16 | ScalarTy::U16)
    }

    /// True if this shared element type requires the `Int64` SPIR-V capability.
    pub fn needs_int64(&self) -> bool {
        matches!(self.elem, ScalarTy::I64 | ScalarTy::U64)
    }

    /// True if this shared element type requires the `Float64` SPIR-V capability.
    pub fn needs_float64(&self) -> bool {
        self.elem == ScalarTy::F64
    }
}

/// A validated workgroup-shared array declaration.
///
/// Each declaration in a kernel body produces one `SharedDecl` with a unique
/// `SharedId`. The declarations are collected into `KernelBodyTyped.shared`
/// in source order and lowered to `OpVariable Workgroup` in SPIR-V.
#[derive(Debug, Clone)]
pub struct SharedDecl {
    /// Unique identifier within the kernel (monotonic from 0).
    pub id: SharedId,
    /// Source-level name (e.g. `"tile"`).
    pub name: String,
    /// Element type and length.
    pub ty: SharedTy,
    /// Source span of the full declaration for error messages.
    pub span: Span,
}

/// Returns `true` if `ty` is an allowed shared-array element type.
///
/// All scalar types except `Bool` are allowed. Bool is rejected because Vulkan
/// does not have a stable `OpTypeBool` memory representation (it is an abstract
/// type with no guaranteed size or alignment).
///
/// Note: `I16` and `U16` are allowed here (unlike cooperative-matrix elements)
/// because they only need the `Int16` type-level capability, not the
/// StorageBuffer16BitAccess extension.
pub fn is_allowed_shared_element(ty: ScalarTy) -> bool {
    !matches!(ty, ScalarTy::Bool)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_ty_elem_byte_size_f32() {
        let ty: SharedTy = SharedTy { elem: ScalarTy::F32, len: 256 };
        assert_eq!(ty.elem_byte_size(), 4);
    }

    #[test]
    fn shared_ty_total_byte_size_f32_256() {
        let ty: SharedTy = SharedTy { elem: ScalarTy::F32, len: 256 };
        // 256 * 4 = 1024 bytes
        assert_eq!(ty.total_byte_size(), 1024);
    }

    #[test]
    fn shared_ty_total_byte_size_f16_1024() {
        let ty: SharedTy = SharedTy { elem: ScalarTy::F16, len: 1024 };
        // 1024 * 2 = 2048 bytes
        assert_eq!(ty.total_byte_size(), 2048);
    }

    #[test]
    fn shared_ty_needs_float16_for_f16() {
        assert!(SharedTy { elem: ScalarTy::F16, len: 1 }.needs_float16());
        assert!(!SharedTy { elem: ScalarTy::F32, len: 1 }.needs_float16());
    }

    #[test]
    fn shared_ty_needs_int8_for_i8_u8() {
        assert!(SharedTy { elem: ScalarTy::I8, len: 1 }.needs_int8());
        assert!(SharedTy { elem: ScalarTy::U8, len: 1 }.needs_int8());
        assert!(!SharedTy { elem: ScalarTy::I32, len: 1 }.needs_int8());
    }

    #[test]
    fn is_allowed_shared_element_rejects_bool() {
        assert!(!is_allowed_shared_element(ScalarTy::Bool));
    }

    #[test]
    fn is_allowed_shared_element_accepts_all_numeric() {
        for ty in [
            ScalarTy::I8, ScalarTy::U8, ScalarTy::I16, ScalarTy::U16,
            ScalarTy::I32, ScalarTy::U32, ScalarTy::I64, ScalarTy::U64,
            ScalarTy::F16, ScalarTy::F32, ScalarTy::F64,
        ] {
            assert!(is_allowed_shared_element(ty), "{ty:?} should be allowed");
        }
    }

    #[test]
    fn max_shared_elems_is_65536() {
        assert_eq!(MAX_SHARED_ELEMS, 65536);
    }

    #[test]
    fn shared_id_ord() {
        assert!(SharedId(0) < SharedId(1));
    }
}
