//! SPIR-V emission for workgroup-shared arrays (M3.2 PART A).
//!
//! Emits:
//! ```text
//! %elem    = <scalar type via ScalarTypeCache>
//! %n_const = OpConstant %u32 <N>         ; array length is an OpConstant id (SPIR-V 3.32.6)
//! %arr     = OpTypeArray %elem %n_const  ; SIZED array (not RuntimeArray)
//! %ptr_wg  = OpTypePointer Workgroup %arr
//! %var     = OpVariable %ptr_wg Workgroup ; global, NO initializer
//! %ptr_e   = OpTypePointer Workgroup %elem ; shared across arrays of the same elem type
//! ```
//!
//! # Access chain arity (CRITICAL)
//!
//! The shared-array access chain uses a SINGLE index:
//! ```text
//! %ptr = OpAccessChain %ptr_e %var %idx   ; ONE index
//! ```
//! This contrasts with the SSBO two-index chain (`%var %0 %idx`) because the SSBO
//! wraps the runtime array in an `OpTypeStruct` (buffers.rs:111), while the shared
//! array has NO wrapping struct. Do NOT copy the SSBO pattern.
//!
//! # Capabilities
//!
//! emit_shared_globals explicitly sets capability flags for the element types it emits.
//! This is REQUIRED because body.rs's `observe_type` does NOT handle F16 (it falls into
//! the `_ => {}` arm at body.rs:213-218). Without this, spirv-val rejects f16 OpLoad/OpStore.
//! The Workgroup StorageClass requires NO new capability or extension.

use std::collections::BTreeMap;
use rspirv::dr::Builder;
use rspirv::spirv::{Word, StorageClass};
use axc_hir::shared::SharedDecl;
use axc_hir::ty::ScalarTy;
use crate::body::{ScalarTypeCache, CapabilitiesRequired, BodyCodegenError};

/// Emitted Workgroup global variable ids, keyed by SharedId.0.
///
/// Uses `BTreeMap` for deterministic iteration order (no HashMap — AT-227).
pub struct SharedBindings {
    /// Maps SharedId.0 → SPIR-V Workgroup OpVariable id.
    pub var_ids: BTreeMap<u32, Word>,
    /// Maps SharedId.0 → Workgroup-class pointer-to-elem type id.
    ///
    /// Used by the shared-read/write access chains AND by the shared-source
    /// coopmat emit path (emit_coopmat_load_shared_inline).
    pub elem_ptr_ids: BTreeMap<u32, Word>,
}

impl SharedBindings {
    /// Create a new empty `SharedBindings`.
    pub fn new() -> Self {
        Self {
            var_ids: BTreeMap::new(),
            elem_ptr_ids: BTreeMap::new(),
        }
    }

    /// Return the Workgroup-class elem-ptr type id for the given SharedId.
    ///
    /// Returns `None` if the id is not present (compiler bug — caller must check).
    pub fn workgroup_elem_ptr_ty(&self, shared_id: u32) -> Option<Word> {
        self.elem_ptr_ids.get(&shared_id).copied()
    }

    /// Return the Workgroup OpVariable id for the given SharedId.
    ///
    /// Returns `None` if the id is not present (compiler bug — caller must check).
    pub fn var_id(&self, shared_id: u32) -> Option<Word> {
        self.var_ids.get(&shared_id).copied()
    }
}

impl Default for SharedBindings {
    fn default() -> Self {
        Self::new()
    }
}

/// Emit all Workgroup global variables for the shared arrays in a kernel body.
///
/// Must be called BEFORE `begin_function` — `OpVariable` with Workgroup storage class
/// must be in the module's types/globals section, NOT inside the function body block.
/// This mirrors `emit_buffer_globals` in buffers.rs.
///
/// Sets the appropriate capability flags on `caps`:
/// - Float16 for F16 elements (observe_type does NOT cover F16; this is the only gate).
/// - Int8 for I8/U8 elements.
/// - Int16 for I16/U16 elements (note: no StorageBuffer16BitAccess — that gates
///   StorageBuffer class only; Workgroup only needs the Int16 type-level cap).
/// - Int64 for I64/U64 elements.
/// - Float64 for F64 elements.
///
/// NO StorageBuffer8/16BitAccess or SPV_KHR_8/16bit_storage for Workgroup arrays.
///
/// # DEDUP
///
/// Two arrays with the SAME elem type SHARE the same elem-ptr type
/// (`%ptr_e = OpTypePointer Workgroup %elem`) — rspirv deduplicates OpTypePointer
/// by (StorageClass, pointee) internally. However, each array gets its OWN
/// OpTypeArray (keyed by SharedId, not by (elem,N)) and its OWN OpVariable.
/// AT-1615 tests this: two arrays of the same elem type must have distinct var_ids.
pub fn emit_shared_globals(
    b: &mut Builder,
    type_cache: &mut ScalarTypeCache,
    caps: &mut CapabilitiesRequired,
    shared: &[SharedDecl],
) -> SharedBindings {
    let mut out = SharedBindings::new();

    for decl in shared {
        let sid: u32 = decl.id.0;
        let elem_ty: ScalarTy = decl.ty.elem;
        let len: u32 = decl.ty.len;

        // Set capability flags for non-trivial element types.
        // observe_type does NOT set Float16 (body.rs:213 leaves F16 in `_ => {}`).
        // emit_shared_globals MUST set it explicitly (AT-1609).
        match elem_ty {
            ScalarTy::F16  => { caps.float16 = true; }
            ScalarTy::I8 | ScalarTy::U8   => { caps.int8 = true; }
            ScalarTy::I16 | ScalarTy::U16 => { caps.int16 = true; }
            ScalarTy::I64 | ScalarTy::U64 => { caps.int64 = true; }
            ScalarTy::F64  => { caps.float64 = true; }
            ScalarTy::I32 | ScalarTy::U32 | ScalarTy::F32 | ScalarTy::Bool => {}
        }

        // 1. Scalar element type.
        let elem_type_id: Word = type_cache.scalar_id(b, elem_ty);

        // 2. OpConstant u32 <N> — array length MUST be an OpConstant id, NOT a literal.
        //    (SPIR-V 3.32.6: OpTypeArray length operand is a constant id.)
        let n_const_id: Word = type_cache.get_or_emit_u32_const(b, len);

        // 3. OpTypeArray %elem %n_const — SIZED array (not RuntimeArray).
        //    Each shared array gets its own distinct OpTypeArray id (keyed by SharedId),
        //    even if N and elem are identical to another array. AT-1615.
        let arr_type_id: Word = b.id();
        let _ = b.type_array_id(Some(arr_type_id), elem_type_id, n_const_id);

        // 4. OpTypePointer Workgroup %arr — pointer to the whole array.
        let ptr_to_arr_id: Word = b.type_pointer(None, StorageClass::Workgroup, arr_type_id);

        // 5. OpVariable %ptr_to_arr Workgroup — global, NO initializer.
        //    (Vulkan forbids Workgroup initializers without VK_KHR_zero_initialize_workgroup_memory.)
        let var_id: Word = b.variable(ptr_to_arr_id, None, StorageClass::Workgroup, None);

        // 6. OpTypePointer Workgroup %elem — pointer to a single element.
        //    rspirv deduplicates by (StorageClass, pointee), so two arrays of the same
        //    elem type share this type id. That is correct and intended (AT-1615 passes).
        let elem_ptr_id: Word = b.type_pointer(None, StorageClass::Workgroup, elem_type_id);

        out.var_ids.insert(sid, var_id);
        out.elem_ptr_ids.insert(sid, elem_ptr_id);
    }

    out
}

/// Emit a SharedRead: `tile[index]`.
///
/// Emits `OpAccessChain %ptr_e %var %idx` (SINGLE index) + `OpLoad`.
/// Returns the SPIR-V id of the loaded element value.
pub fn emit_shared_read(
    b: &mut Builder,
    bindings: &SharedBindings,
    shared_id: u32,
    index_id: Word,
    elem_ty: Word,
) -> Result<Word, BodyCodegenError> {
    let var_id: Word = bindings.var_id(shared_id)
        .ok_or(BodyCodegenError::UnexpectedHir("emit_shared_read: missing var_id for shared_id"))?;
    let elem_ptr_ty: Word = bindings.workgroup_elem_ptr_ty(shared_id)
        .ok_or(BodyCodegenError::UnexpectedHir("emit_shared_read: missing elem_ptr_id for shared_id"))?;

    // SINGLE-index access chain: %var %idx (NO leading struct-member-0).
    let ptr_id: Word = b.access_chain(elem_ptr_ty, None, var_id, [index_id])
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;

    let val_id: Word = b.load(elem_ty, None, ptr_id, None, std::iter::empty())
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;

    Ok(val_id)
}

/// Emit a SharedWrite: `tile[index] = value`.
///
/// Emits `OpAccessChain %ptr_e %var %idx` (SINGLE index) + `OpStore`.
pub fn emit_shared_write(
    b: &mut Builder,
    bindings: &SharedBindings,
    shared_id: u32,
    index_id: Word,
    value_id: Word,
) -> Result<(), BodyCodegenError> {
    let var_id: Word = bindings.var_id(shared_id)
        .ok_or(BodyCodegenError::UnexpectedHir("emit_shared_write: missing var_id for shared_id"))?;
    let elem_ptr_ty: Word = bindings.workgroup_elem_ptr_ty(shared_id)
        .ok_or(BodyCodegenError::UnexpectedHir("emit_shared_write: missing elem_ptr_id for shared_id"))?;

    // SINGLE-index access chain: %var %idx.
    let ptr_id: Word = b.access_chain(elem_ptr_ty, None, var_id, [index_id])
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;

    b.store(ptr_id, value_id, None, std::iter::empty())
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_bindings_new_is_empty() {
        let sb: SharedBindings = SharedBindings::new();
        assert!(sb.var_ids.is_empty());
        assert!(sb.elem_ptr_ids.is_empty());
        assert!(sb.workgroup_elem_ptr_ty(0).is_none());
        assert!(sb.var_id(0).is_none());
    }
}
