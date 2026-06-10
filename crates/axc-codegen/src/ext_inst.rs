//! SPIR-V emission helpers for the GLSL.std.450 extended-instruction builtins (M3.2c).
//!
//! Mirrors `crates/axc-codegen/src/q4_0.rs::emit_f32_to_f16` (the M3.5 unary-builtin
//! precedent) in shape. Provides `emit_exp`, called from `body.rs` when a
//! `HirExprKind::ExtInstBuiltin { op: Exp, .. }` node is encountered.
//!
//! **The import-once invariant (load-bearing).** GLSL.std.450 is an EXTENDED
//! INSTRUCTION SET requiring exactly ONE `OpExtInstImport "GLSL.std.450"` in the
//! module header, whose result-id (the set-id) is reused by every `OpExtInst`.
//! `emit_exp` takes the ALREADY-CACHED set-id (`glsl450_set`) — the caller
//! (`body.rs`) fetches it via `ScalarTypeCache::get_or_emit_glsl450_set`, which
//! emits the import once and caches the id. rspirv's `ext_inst_import` does NOT
//! deduplicate, so the manual cache is what guarantees exactly one import. AT-1822
//! asserts this on a ≥2-exp kernel.
//!
//! **No new capability.** `GLSL.std.450` is an `OpExtInstImport`, NOT an
//! `OpCapability`/`OpExtension`. `OpExtInst Exp` on an f32 operand needs only
//! `Shader` + `Float32` (both baseline). `emit_exp` therefore takes NO
//! `&mut CapabilitiesRequired` (unlike `emit_f32_to_f16`, which sets `caps.float16`).

use rspirv::dr::{Builder, Operand};
use rspirv::spirv::Word;
use axc_hir::ty::ScalarTy;
use crate::body::{ScalarTypeCache, BodyCodegenError};

/// Emit `exp(x)` via GLSL.std.450 `Exp` (instruction 27).
///
/// Emits `%r = OpExtInst %f32 %glsl450_set 27 %x` via
/// `Builder::ext_inst(f32_ty, None, glsl450_set, 27, [Operand::IdRef(x_id)])`.
///
/// - `glsl450_set` is the cached `OpExtInstImport "GLSL.std.450"` set-id (see
///   `ScalarTypeCache::get_or_emit_glsl450_set`).
/// - `x_id` is the LOADED f32 VALUE id (the caller emits the arg first via
///   `emit_expr`, which returns loaded scalar values, never raw pointers).
///
/// Returns the `f32` result id. No new capability.
pub fn emit_exp(
    b: &mut Builder,
    tc: &mut ScalarTypeCache,
    glsl450_set: Word,
    x_id: Word,
) -> Result<Word, BodyCodegenError> {
    let f32_ty = tc.scalar_id(b, ScalarTy::F32);

    // %r = OpExtInst %f32 %glsl450_set 27 %x  (27 = GLSL.std.450 Exp).
    let result_id = b
        .ext_inst(f32_ty, None, glsl450_set, 27, [Operand::IdRef(x_id)])
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;

    Ok(result_id)
}
