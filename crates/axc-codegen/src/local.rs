//! SPIR-V emission for per-invocation local arrays (M3.20).
//!
//! Private-memory sibling of `shared.rs` (M3.2 workgroup-shared arrays). Emits:
//! ```text
//! %elem    = <scalar type via ScalarTypeCache>
//! %n_const = OpConstant %u32 <N>          ; array length is an OpConstant id (SPIR-V 3.32.6)
//! %arr     = OpTypeArray %elem %n_const   ; SIZED array (not RuntimeArray), one per LocalArrayId
//! %ptr_fn  = OpTypePointer Function %arr  ; pointer to the whole array (globals/types section)
//! %ptr_e   = OpTypePointer Function %elem ; pointer to one element (reuses the `let`-scalar cache)
//! ; --- and, in the FUNCTION ENTRY-BLOCK PRELUDE (body.rs), NOT here ---
//! %var     = OpVariable %ptr_fn Function  ; NO initializer
//! ```
//!
//! # Key divergence from `shared.rs`: OpVariable placement
//!
//! Unlike Workgroup-class shared arrays (whose `OpVariable` is a MODULE GLOBAL,
//! emitted before `begin_function`), a `Function`-storage local array's
//! `OpVariable` MUST live in the function's FIRST BLOCK, before any non-variable
//! instruction (SPIR-V §2.16.1). `emit_local_array_types` runs BEFORE
//! `begin_block` (like `emit_shared_globals`) and creates the TYPES only —
//! `OpTypeArray` + both `OpTypePointer Function` — but NOT the `OpVariable`.
//! `body.rs`'s entry-block prelude loop (which already emits `let`-scalar
//! `Function` `OpVariable`s) creates the `OpVariable` for each declared array and
//! records its id into `LocalArrayBindings.var_ids`.
//!
//! # Access chain arity
//!
//! Single-index, exactly like `shared.rs` (no wrapping struct — see that module's
//! doc for why this differs from the SSBO two-index chain).
//!
//! # Capabilities
//!
//! Mirrors `emit_shared_globals`'s capability-flag logic exactly (Float16/Int8/
//! Int16/Int64/Float64 set when the element type demands it). `Function` storage
//! is core Shader — no new capability or extension, ever.

use std::collections::BTreeMap;
use rspirv::dr::Builder;
use rspirv::spirv::{Word, StorageClass};
use axc_hir::local::LocalArrayDecl;
use axc_hir::ty::ScalarTy;
use crate::body::{ScalarTypeCache, CapabilitiesRequired, BodyCodegenError};

/// Emitted local-array type/variable ids, keyed by LocalArrayId.0.
///
/// Uses `BTreeMap` for deterministic iteration order (mirrors `SharedBindings`,
/// AT-227 discipline).
pub struct LocalArrayBindings {
    /// Maps LocalArrayId.0 -> Function-storage OpVariable id.
    ///
    /// Populated by `emit_local_array_types` is EMPTY here — the `OpVariable`
    /// itself is created in `body.rs`'s entry-block prelude loop, which fills
    /// this map in afterward (see this module's doc for why).
    pub var_ids: BTreeMap<u32, Word>,
    /// Maps LocalArrayId.0 -> Function-class pointer-to-elem type id.
    pub elem_ptr_ids: BTreeMap<u32, Word>,
    /// Maps LocalArrayId.0 -> Function-class pointer-to-whole-array type id.
    ///
    /// Consumed by `body.rs`'s prelude loop to create the `OpVariable`.
    pub arr_ptr_ids: BTreeMap<u32, Word>,
}

impl LocalArrayBindings {
    /// Create a new empty `LocalArrayBindings`.
    pub fn new() -> Self {
        Self {
            var_ids: BTreeMap::new(),
            elem_ptr_ids: BTreeMap::new(),
            arr_ptr_ids: BTreeMap::new(),
        }
    }

    /// Return the Function-class elem-ptr type id for the given LocalArrayId.
    pub fn function_elem_ptr_ty(&self, local_array_id: u32) -> Option<Word> {
        self.elem_ptr_ids.get(&local_array_id).copied()
    }

    /// Return the Function-class array-ptr type id for the given LocalArrayId.
    pub fn arr_ptr_ty(&self, local_array_id: u32) -> Option<Word> {
        self.arr_ptr_ids.get(&local_array_id).copied()
    }

    /// Return the Function OpVariable id for the given LocalArrayId.
    pub fn var_id(&self, local_array_id: u32) -> Option<Word> {
        self.var_ids.get(&local_array_id).copied()
    }
}

impl Default for LocalArrayBindings {
    fn default() -> Self {
        Self::new()
    }
}

/// Emit all Function-storage TYPES (NOT the `OpVariable`s) for the local arrays
/// in a kernel body.
///
/// Called BEFORE `begin_block`, mirroring `emit_shared_globals`'s call site —
/// `OpTypeArray`/`OpTypePointer` always land in the module's types/globals
/// section regardless of block state, so creating them before `begin_block` is
/// safe (and keeps this call symmetric with the shared-array pre-scan). The
/// `OpVariable` itself is deliberately NOT created here — see this module's doc.
///
/// The elem-ptr type is obtained via `type_cache.function_ptr_id`, the SAME
/// cache `let`-scalar bindings use — a kernel with both plain `let`s and local
/// arrays of the same elem type shares one `OpTypePointer Function %elem` id
/// (on-demand emission: a kernel using NO local arrays never calls this
/// function, so its SPIR-V is byte-identical to before M3.20).
pub fn emit_local_array_types(
    b: &mut Builder,
    type_cache: &mut ScalarTypeCache,
    caps: &mut CapabilitiesRequired,
    arrays: &[LocalArrayDecl],
) -> LocalArrayBindings {
    let mut out = LocalArrayBindings::new();

    for decl in arrays {
        let aid: u32 = decl.id.0;
        let elem_ty: ScalarTy = decl.ty.elem;
        let len: u32 = decl.ty.len;

        // Set capability flags for non-trivial element types (mirrors
        // emit_shared_globals — observe_type does NOT set Float16).
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

        // 2. OpConstant u32 <N> — array length MUST be an OpConstant id.
        let n_const_id: Word = type_cache.get_or_emit_u32_const(b, len);

        // 3. OpTypeArray %elem %n_const — SIZED array, one per LocalArrayId.
        let arr_type_id: Word = b.id();
        let _ = b.type_array_id(Some(arr_type_id), elem_type_id, n_const_id);

        // 4. OpTypePointer Function %arr — pointer to the whole array.
        let ptr_to_arr_id: Word = b.type_pointer(None, StorageClass::Function, arr_type_id);

        // 5. OpTypePointer Function %elem — reuses the `let`-scalar Function-ptr cache.
        let elem_ptr_id: Word = type_cache.function_ptr_id(b, elem_ty);

        out.arr_ptr_ids.insert(aid, ptr_to_arr_id);
        out.elem_ptr_ids.insert(aid, elem_ptr_id);
    }

    out
}

/// Emit a LocalArrayRead: `hist[index]`.
///
/// Emits `OpAccessChain %ptr_e %var %idx` (SINGLE index) + `OpLoad`.
/// Returns the SPIR-V id of the loaded element value.
pub fn emit_local_array_read(
    b: &mut Builder,
    bindings: &LocalArrayBindings,
    local_array_id: u32,
    index_id: Word,
    elem_ty: Word,
) -> Result<Word, BodyCodegenError> {
    let var_id: Word = bindings.var_id(local_array_id)
        .ok_or(BodyCodegenError::UnexpectedHir("emit_local_array_read: missing var_id for local_array_id"))?;
    let elem_ptr_ty: Word = bindings.function_elem_ptr_ty(local_array_id)
        .ok_or(BodyCodegenError::UnexpectedHir("emit_local_array_read: missing elem_ptr_id for local_array_id"))?;

    let ptr_id: Word = b.access_chain(elem_ptr_ty, None, var_id, [index_id])
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;

    let val_id: Word = b.load(elem_ty, None, ptr_id, None, std::iter::empty())
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;

    Ok(val_id)
}

/// Emit a LocalArrayWrite: `hist[index] = value`.
///
/// Emits `OpAccessChain %ptr_e %var %idx` (SINGLE index) + `OpStore`.
pub fn emit_local_array_write(
    b: &mut Builder,
    bindings: &LocalArrayBindings,
    local_array_id: u32,
    index_id: Word,
    value_id: Word,
) -> Result<(), BodyCodegenError> {
    let var_id: Word = bindings.var_id(local_array_id)
        .ok_or(BodyCodegenError::UnexpectedHir("emit_local_array_write: missing var_id for local_array_id"))?;
    let elem_ptr_ty: Word = bindings.function_elem_ptr_ty(local_array_id)
        .ok_or(BodyCodegenError::UnexpectedHir("emit_local_array_write: missing elem_ptr_id for local_array_id"))?;

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
    fn local_array_bindings_new_is_empty() {
        let lb: LocalArrayBindings = LocalArrayBindings::new();
        assert!(lb.var_ids.is_empty());
        assert!(lb.elem_ptr_ids.is_empty());
        assert!(lb.arr_ptr_ids.is_empty());
        assert!(lb.function_elem_ptr_ty(0).is_none());
        assert!(lb.arr_ptr_ty(0).is_none());
        assert!(lb.var_id(0).is_none());
    }
}
