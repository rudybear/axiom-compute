//! Buffer type representation at HIR level.
//!
//! Introduced in M1.2. Buffers are SSBO-backed array parameters passed to
//! compute kernels. Each buffer has an element type (`ScalarTy`) and an
//! access mode (`BufferAccess`).

use crate::ty::ScalarTy;

/// Access mode for a buffer parameter.
///
/// Maps directly to Vulkan SPIR-V decorations:
/// - `ReadWrite` → no extra decoration (default SSBO)
/// - `ReadOnly`  → `Decoration::NonWritable`
/// - `WriteOnly` → `Decoration::NonReadable`
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferAccess {
    /// Buffer is readable and writable. No extra decoration.
    ReadWrite,
    /// Buffer is read-only. Emits `Decoration::NonWritable`.
    ReadOnly,
    /// Buffer is write-only. Emits `Decoration::NonReadable`.
    WriteOnly,
}

impl BufferAccess {
    /// Human-readable display name for error messages.
    pub fn display_name(&self) -> &'static str {
        match self {
            BufferAccess::ReadWrite  => "buffer",
            BufferAccess::ReadOnly   => "readonly_buffer",
            BufferAccess::WriteOnly  => "writeonly_buffer",
        }
    }
}

/// The HIR representation of a buffer type.
///
/// Corresponds to `buffer[T]`, `readonly_buffer[T]`, `writeonly_buffer[T]`
/// in the source language.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferTy {
    /// Element type. Only 32-bit and 64-bit scalars are supported in M1.2.
    pub elem: ScalarTy,
    /// Read/write access mode.
    pub access: BufferAccess,
}

impl BufferTy {
    /// Byte size of a single element for stride computation.
    ///
    /// Used to compute `Decoration::ArrayStride`:
    /// - 1 byte for 8-bit types (I8, U8)
    /// - 2 bytes for 16-bit types (F16, I16, U16)
    /// - 4 bytes for 32-bit types (I32, U32, F32)
    /// - 8 bytes for 64-bit types (I64, U64, F64)
    ///
    /// M2.1: F16 buffers require `StorageBuffer16BitAccess` capability and
    /// `SPV_KHR_16bit_storage` extension (emitted by codegen).
    pub fn elem_byte_size(&self) -> u32 {
        match self.elem.bit_width() {
            8  => 1,
            16 => 2,
            32 => 4,
            64 => 8,
            w  => panic!("unsupported buffer elem bit width {w}"),
        }
    }

    /// True if this buffer element type requires the `StorageBuffer16BitAccess`
    /// capability and `SPV_KHR_16bit_storage` extension.
    ///
    /// M2.1: Only F16 SSBO elements trigger this. I8/U8 use the 8-bit storage
    /// extension path (see `needs_8bit_storage`).
    pub fn needs_16bit_storage(&self) -> bool {
        self.elem.bit_width() == 16
    }

    /// True if this buffer element type requires the `StorageBuffer8BitAccess`
    /// capability and `SPV_KHR_8bit_storage` extension.
    ///
    /// M2.5: I8 and U8 SSBO elements trigger this capability path.
    pub fn needs_8bit_storage(&self) -> bool {
        self.elem.bit_width() == 8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_access_display_names() {
        assert_eq!(BufferAccess::ReadWrite.display_name(), "buffer");
        assert_eq!(BufferAccess::ReadOnly.display_name(), "readonly_buffer");
        assert_eq!(BufferAccess::WriteOnly.display_name(), "writeonly_buffer");
    }

    #[test]
    fn buffer_ty_elem_byte_size_f32() {
        let bt: BufferTy = BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadWrite };
        assert_eq!(bt.elem_byte_size(), 4);
    }

    #[test]
    fn buffer_ty_elem_byte_size_f64() {
        let bt: BufferTy = BufferTy { elem: ScalarTy::F64, access: BufferAccess::ReadOnly };
        assert_eq!(bt.elem_byte_size(), 8);
    }

    #[test]
    fn buffer_ty_elem_byte_size_u32() {
        let bt: BufferTy = BufferTy { elem: ScalarTy::U32, access: BufferAccess::WriteOnly };
        assert_eq!(bt.elem_byte_size(), 4);
    }

    #[test]
    fn buffer_ty_elem_byte_size_i64() {
        let bt: BufferTy = BufferTy { elem: ScalarTy::I64, access: BufferAccess::ReadWrite };
        assert_eq!(bt.elem_byte_size(), 8);
    }

    /// AT-618: F16 buffer element byte size is 2 (16 / 8).
    #[test]
    fn buffer_ty_elem_byte_size_f16() {
        let bt: BufferTy = BufferTy { elem: ScalarTy::F16, access: BufferAccess::ReadWrite };
        assert_eq!(bt.elem_byte_size(), 2);
    }

    /// AT-618: F16 buffer needs 16-bit storage capability.
    #[test]
    fn buffer_ty_f16_needs_16bit_storage() {
        let bt_f16: BufferTy = BufferTy { elem: ScalarTy::F16, access: BufferAccess::ReadWrite };
        assert!(bt_f16.needs_16bit_storage(), "F16 buffer must need 16-bit storage");

        let bt_f32: BufferTy = BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadWrite };
        assert!(!bt_f32.needs_16bit_storage(), "F32 buffer must NOT need 16-bit storage");
    }
}
