//! SPIR-V emission helpers for Q4_0 dequantization builtins (M2.5).
//!
//! Provides four helpers called from `body.rs` when a `HirExprKind::Q4_0Builtin`
//! node is encountered:
//!
//! - `emit_ptr_read_u8_zext`  — OpAccessChain(u8 SSBO elem) + OpLoad(u8) + OpUConvert(u32)
//! - `emit_ptr_read_u16_zext` — two u8 loads + shift/or reassembly into u32
//! - `emit_f16_bits_to_f32`   — OpUConvert(u32→u16) + OpBitcast(u16→f16) + OpFConvert(f16→f32)
//! - `emit_f32_from_u32`      — OpConvertUToF(u32→f32)
//!
//! **Capability side-effects (set on the `CapabilitiesRequired` struct):**
//! - `emit_ptr_read_u8_zext` / `emit_ptr_read_u16_zext`: set `caps.int8 = true`, `caps.storage_8bit = true`
//! - `emit_f16_bits_to_f32`:  set `caps.int16 = true`, `caps.float16 = true`
//! - `emit_f32_from_u32`:     no new capabilities
//!
//! **Invariants:**
//! - All functions accept `&mut CapabilitiesRequired` and are responsible for
//!   marking the capabilities they need (lazy accumulation pattern from M2.1).
//! - `emit_ptr_read_u16_zext` emits two u8 loads at byte_offset and byte_offset+1,
//!   then assembles little-endian u16: `lo | (hi << 8)`. This matches the
//!   Q4_0 block layout where the f16 scale is stored as two consecutive bytes
//!   in little-endian order in a `buffer[u8]` SSBO.
//! - SAFETY blocks on `unsafe` rspirv calls document the invariant that the
//!   argument IDs were freshly produced by the same Builder context.

use rspirv::dr::Builder;
use rspirv::spirv::Word;
use axc_hir::ty::ScalarTy;
use crate::body::{ScalarTypeCache, CapabilitiesRequired, BodyCodegenError};

/// Emit a zero-extended single-byte load from a `buffer[u8]` SSBO at a given byte offset.
///
/// Emits:
/// 1. `OpAccessChain %ptr_u8_StorageBuffer %buf_var_id %zero %byte_offset_id`
/// 2. `OpLoad %u8 %ptr`
/// 3. `OpUConvert %u32 %byte_value`
///
/// Returns the `u32` result id.
///
/// Sets `caps.int8 = true` and `caps.storage_8bit = true`.
pub fn emit_ptr_read_u8_zext(
    b: &mut Builder,
    tc: &mut ScalarTypeCache,
    caps: &mut CapabilitiesRequired,
    buf_var_id: Word,
    elem_ptr_ty: Word,
    byte_offset_id: Word,
) -> Result<Word, BodyCodegenError> {
    caps.int8 = true;
    caps.storage_8bit = true;

    let u8_ty  = tc.scalar_id(b, ScalarTy::U8);
    let u32_ty = tc.scalar_id(b, ScalarTy::U32);

    // Index zero selects the runtime array inside the SSBO struct.
    let zero_id = tc.get_or_emit_u32_const(b, 0);

    // OpAccessChain into StorageBuffer class — pointer to the u8 element.
    let ptr_id = b.access_chain(elem_ptr_ty, None, buf_var_id, vec![zero_id, byte_offset_id])
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;

    // OpLoad the u8 element.
    let byte_val_id = b.load(u8_ty, None, ptr_id, None, vec![])
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;

    // OpUConvert u8 → u32 (zero-extend).
    let result_id = b.u_convert(u32_ty, None, byte_val_id)
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;

    Ok(result_id)
}

/// Emit a zero-extended two-byte little-endian load from a `buffer[u8]` SSBO.
///
/// Loads bytes at `byte_offset_id` and `byte_offset_id + 1`, then assembles:
/// ```text
/// lo = buf[byte_offset]         (u8 → u32 zext)
/// hi = buf[byte_offset + 1]     (u8 → u32 zext)
/// result = lo | (hi << 8)
/// ```
///
/// Returns the assembled `u32` result id, representing the 16-bit value in the
/// lower 16 bits (bits 15..0), as a u32.
///
/// Sets `caps.int8 = true` and `caps.storage_8bit = true`.
pub fn emit_ptr_read_u16_zext(
    b: &mut Builder,
    tc: &mut ScalarTypeCache,
    caps: &mut CapabilitiesRequired,
    buf_var_id: Word,
    elem_ptr_ty: Word,
    byte_offset_id: Word,
) -> Result<Word, BodyCodegenError> {
    caps.int8 = true;
    caps.storage_8bit = true;

    let u8_ty  = tc.scalar_id(b, ScalarTy::U8);
    let u32_ty = tc.scalar_id(b, ScalarTy::U32);

    let zero_id = tc.get_or_emit_u32_const(b, 0);
    let one_id  = tc.get_or_emit_u32_const(b, 1);
    let eight_id = tc.get_or_emit_u32_const(b, 8);

    // ── load lo byte at byte_offset ──────────────────────────────────────────
    let ptr_lo = b.access_chain(elem_ptr_ty, None, buf_var_id, vec![zero_id, byte_offset_id])
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;
    let byte_lo = b.load(u8_ty, None, ptr_lo, None, vec![])
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;
    let lo_u32 = b.u_convert(u32_ty, None, byte_lo)
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;

    // ── compute byte_offset + 1 for hi byte ──────────────────────────────────
    let offset_plus_one = b.i_add(u32_ty, None, byte_offset_id, one_id)
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;

    // ── load hi byte at byte_offset + 1 ──────────────────────────────────────
    let ptr_hi = b.access_chain(elem_ptr_ty, None, buf_var_id, vec![zero_id, offset_plus_one])
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;
    let byte_hi = b.load(u8_ty, None, ptr_hi, None, vec![])
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;
    let hi_u32 = b.u_convert(u32_ty, None, byte_hi)
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;

    // ── assemble: lo | (hi << 8) ──────────────────────────────────────────────
    // OpShiftLeftLogical %u32 %hi_u32 %8
    let hi_shifted = b.shift_left_logical(u32_ty, None, hi_u32, eight_id)
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;
    // OpBitwiseOr %u32 %lo_u32 %hi_shifted
    let result_id = b.bitwise_or(u32_ty, None, lo_u32, hi_shifted)
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;

    Ok(result_id)
}

/// Emit the conversion from u32 bit-pattern to f32 via f16 reinterpretation.
///
/// Emits:
/// 1. `OpUConvert %u16 %u32_bits_id`    — truncate to 16 bits
/// 2. `OpBitcast  %f16 %u16_val`        — reinterpret bits as f16
/// 3. `OpFConvert %f32 %f16_val`        — widen to f32
///
/// Returns the `f32` result id.
///
/// Sets `caps.int16 = true` and `caps.float16 = true`.
pub fn emit_f16_bits_to_f32(
    b: &mut Builder,
    tc: &mut ScalarTypeCache,
    caps: &mut CapabilitiesRequired,
    u32_bits_id: Word,
) -> Result<Word, BodyCodegenError> {
    caps.int16 = true;
    caps.float16 = true;

    let u16_ty = tc.scalar_id(b, ScalarTy::U16);
    let f16_ty = tc.scalar_id(b, ScalarTy::F16);
    let f32_ty = tc.scalar_id(b, ScalarTy::F32);

    // OpUConvert u32 → u16 (truncate to lower 16 bits).
    let u16_val = b.u_convert(u16_ty, None, u32_bits_id)
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;

    // OpBitcast u16 → f16 (same bits, interpreted as IEEE 754 half-precision).
    let f16_val = b.bitcast(f16_ty, None, u16_val)
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;

    // OpFConvert f16 → f32 (widen to single-precision).
    let f32_val = b.f_convert(f32_ty, None, f16_val)
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;

    Ok(f32_val)
}

/// Emit unsigned-integer-to-float conversion: `OpConvertUToF u32 → f32`.
///
/// Used in the Q4_0 nibble path to convert nibble values (0..15 as u32) to f32.
///
/// Returns the `f32` result id. No new capabilities required.
pub fn emit_f32_from_u32(
    b: &mut Builder,
    tc: &mut ScalarTypeCache,
    u32_id: Word,
) -> Result<Word, BodyCodegenError> {
    let f32_ty = tc.scalar_id(b, ScalarTy::F32);

    let result_id = b.convert_u_to_f(f32_ty, None, u32_id)
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;

    Ok(result_id)
}

