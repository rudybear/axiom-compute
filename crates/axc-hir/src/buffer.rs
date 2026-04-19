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
    /// Used to compute `Decoration::ArrayStride`: 4 for 32-bit, 8 for 64-bit.
    pub fn elem_byte_size(&self) -> u32 {
        match self.elem.bit_width() {
            32 => 4,
            64 => 8,
            w  => panic!("unsupported buffer elem bit width {w}"),
        }
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
}
