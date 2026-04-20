//! SPIR-V cooperative-matrix codegen for AXIOM-Compute M2.1.
//!
//! Implements code generation for the four cooperative-matrix builtins:
//!   - `coopmat_zero()`  → `OpConstantNull %mat_type_id`
//!   - `coopmat_load(buf, element_offset, stride)` → `OpAccessChain` + `OpCooperativeMatrixLoadKHR`
//!   - `coopmat_mul_add(a, b, c)` → `OpCooperativeMatrixMulAddKHR`
//!   - `coopmat_store(m, buf, element_offset, stride)` → `OpAccessChain` + `OpCooperativeMatrixStoreKHR`
//!
//! # Caller contract
//!
//! Body.rs calls `emit_expr` on argument expressions FIRST, then passes the
//! resulting SPIR-V `Word` ids to these functions. The coopmat module only
//! needs `b`, `type_cache`, and `caps` — it never calls back into `emit_expr`.
//!
//! # SPIR-V extension requirements
//!
//! Any use of a cooperative-matrix builtin sets `CapabilitiesRequired::coopmat = true`.
//! The emit.rs pass observes this flag and emits:
//!   - `OpCapability CooperativeMatrixKHR`
//!   - `OpCapability VulkanMemoryModel`
//!   - `OpExtension "SPV_KHR_cooperative_matrix"`
//!   - `OpExtension "SPV_KHR_vulkan_memory_model"`
//!   - Changes `OpMemoryModel Logical Vulkan` (instead of GLSL450)
//!
//! # Scope
//!
//! All cooperative-matrix ops use Subgroup scope (3) per the SPIR-V 1.3 spec.
//! The scope is emitted as an OpConstant u32 (cached via `ScalarTypeCache`).
//!
//! # Matrix type caching (AT-619)
//!
//! `OpTypeCooperativeMatrixKHR` must not be duplicated for the same logical type.
//! `CoopMatTypeCache` deduplicates by `CoopMatKey` using a `BTreeMap`.
//!
//! # Memory layout
//!
//! All loads and stores use `RowMajorKHR` (0) layout (AT-613).
//!
//! # MulAdd operands
//!
//! For floating-point matrix multiply-add, `CooperativeMatrixOperands::NONE_KHR`
//! is passed (no signed-component or saturating flags).

use std::collections::BTreeMap;
use rspirv::dr::Builder;
use rspirv::spirv::{Word, CooperativeMatrixOperands};
use axc_hir::coopmat::{CoopMatKey, CoopMatUse};
use crate::body::{BodyCodegenError, ScalarTypeCache, CapabilitiesRequired};

// ── SPIR-V constants ──────────────────────────────────────────────────────────

/// SPIR-V Scope::Subgroup = 3.
pub const SUBGROUP_SCOPE_VALUE: u32 = 3;

/// CooperativeMatrixUse::MatrixAKHR = 0.
const MATRIX_USE_A: u32 = 0;
/// CooperativeMatrixUse::MatrixBKHR = 1.
const MATRIX_USE_B: u32 = 1;
/// CooperativeMatrixUse::MatrixAccumulatorKHR = 2.
const MATRIX_USE_ACCUMULATOR: u32 = 2;

/// CooperativeMatrixLayout::RowMajorKHR = 0.
/// Used for all loads and stores (AT-613: all loads/stores use row major).
const ROW_MAJOR_LAYOUT: u32 = 0;

// ── CoopMatTypeCache ──────────────────────────────────────────────────────────

/// Cache of SPIR-V `OpTypeCooperativeMatrixKHR` type IDs, keyed by `CoopMatKey`.
///
/// Cooperative-matrix types must be deduplicated (AT-619).
/// `BTreeMap` is used for deterministic iteration order (AT-418 / AT-622).
#[derive(Debug, Default)]
pub struct CoopMatTypeCache {
    ids: BTreeMap<CoopMatKey, Word>,
}

impl CoopMatTypeCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or emit the `OpTypeCooperativeMatrixKHR` type ID for a given key.
    ///
    /// Parameters from the SPIR-V spec (SPIR-V 1.3 + SPV_KHR_cooperative_matrix):
    /// - `component_type`: scalar element type ID
    /// - `scope`: u32 constant ID for Subgroup scope (= 3)
    /// - `rows`, `columns`: u32 constant IDs for M, N dimensions
    /// - `usage`: u32 constant ID for MatrixAKHR/MatrixBKHR/MatrixAccumulatorKHR (0/1/2)
    pub fn get_or_emit(
        &mut self,
        b: &mut Builder,
        type_cache: &mut ScalarTypeCache,
        key: CoopMatKey,
    ) -> Word {
        if let Some(&id) = self.ids.get(&key) {
            return id;
        }

        let elem_type_id = type_cache.scalar_id(b, key.elem);

        // Subgroup scope = 3.
        let scope_id = type_cache.get_or_emit_u32_const(b, SUBGROUP_SCOPE_VALUE);

        // M and N dimensions as u32 constants.
        let rows_id = type_cache.get_or_emit_u32_const(b, key.m);
        let cols_id = type_cache.get_or_emit_u32_const(b, key.n);

        // Usage: 0=MatrixA, 1=MatrixB, 2=MatrixAccumulator.
        let usage_val: u32 = match key.use_ {
            CoopMatUse::MatrixA     => MATRIX_USE_A,
            CoopMatUse::MatrixB     => MATRIX_USE_B,
            CoopMatUse::Accumulator => MATRIX_USE_ACCUMULATOR,
        };
        let usage_id = type_cache.get_or_emit_u32_const(b, usage_val);

        // rspirv's type_cooperative_matrix_khr deduplicates by value internally.
        let id = b.type_cooperative_matrix_khr(
            elem_type_id,
            scope_id,
            rows_id,
            cols_id,
            usage_id,
        );

        self.ids.insert(key, id);
        id
    }
}

// ── coopmat_zero ─────────────────────────────────────────────────────────────

/// Emit `coopmat_zero()` → `OpConstantNull %mat_type_id`.
///
/// `result_ty` is the cooperative-matrix key from the let-binding context.
/// Sets `caps.coopmat = true`.
pub fn emit_coopmat_zero(
    b: &mut Builder,
    type_cache: &mut ScalarTypeCache,
    coopmat_cache: &mut CoopMatTypeCache,
    caps: &mut CapabilitiesRequired,
    result_ty: CoopMatKey,
) -> Word {
    caps.coopmat = true;
    let mat_type_id = coopmat_cache.get_or_emit(b, type_cache, result_ty);
    b.constant_null(mat_type_id)
}

// ── coopmat_mul_add ──────────────────────────────────────────────────────────

/// Emit `coopmat_mul_add(a, b, c)` → `OpCooperativeMatrixMulAddKHR`.
///
/// `a_id`, `b_id`, `c_id` are pre-evaluated SPIR-V result ids.
/// `result_ty` is the accumulator type (from the HIR, validated to match `c`).
///
/// Sets `caps.coopmat = true`.
#[allow(clippy::too_many_arguments)]
pub fn emit_coopmat_mul_add(
    b: &mut Builder,
    type_cache: &mut ScalarTypeCache,
    coopmat_cache: &mut CoopMatTypeCache,
    caps: &mut CapabilitiesRequired,
    result_ty: CoopMatKey,
    a_id: Word,
    b_id: Word,
    c_id: Word,
) -> Result<Word, BodyCodegenError> {
    caps.coopmat = true;

    let mat_type_id = coopmat_cache.get_or_emit(b, type_cache, result_ty);

    // For floating-point matrices, no signed-component or saturation flags needed.
    let operands = CooperativeMatrixOperands::NONE_KHR;

    let result_id = b.cooperative_matrix_mul_add_khr(
        mat_type_id,
        None,
        a_id,
        b_id,
        c_id,
        Some(operands),
    )
    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;

    Ok(result_id)
}

// ── coopmat_store ────────────────────────────────────────────────────────────

/// Emit `coopmat_store(m, buf, element_offset, stride)` using pre-extracted buffer ids.
///
/// Used by body.rs which splits the borrow to extract `buf_var_id` and `elem_ptr_ty`
/// from `BufferBindings` before calling this function.
///
/// Lowers to:
/// ```text
/// %zero   = OpConstant u32 0
/// %ptr    = OpAccessChain %elem_ptr_ty %buf_var %zero %element_offset
/// %layout = OpConstant u32 0   ; RowMajorKHR
/// OpCooperativeMatrixStoreKHR %ptr %mat_val %layout %stride None []
/// ```
///
/// Sets `caps.coopmat = true`.
#[allow(clippy::too_many_arguments)]
pub fn emit_coopmat_store_inline(
    b: &mut Builder,
    type_cache: &mut ScalarTypeCache,
    caps: &mut CapabilitiesRequired,
    buf_var_id: Word,
    elem_ptr_ty: Word,
    mat_val_id: Word,
    element_offset_id: Word,
    stride_id: Word,
) -> Result<(), BodyCodegenError> {
    caps.coopmat = true;

    // OpAccessChain into the buffer at element_offset (member 0 of the SSBO struct).
    let zero_id = type_cache.get_or_emit_u32_const(b, 0);
    let ptr_id = b.access_chain(elem_ptr_ty, None, buf_var_id, [zero_id, element_offset_id])
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;

    // RowMajorKHR layout constant.
    let layout_id = type_cache.get_or_emit_u32_const(b, ROW_MAJOR_LAYOUT);

    b.cooperative_matrix_store_khr(
        ptr_id,
        mat_val_id,
        layout_id,
        Some(stride_id),
        None,              // no MemoryAccess flags
        std::iter::empty(),
    )
    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;

    Ok(())
}

/// Emit `coopmat_load` using pre-extracted buffer ids.
///
/// Used by body.rs which splits the borrow to extract `buf_var_id` and `elem_ptr_ty`
/// from `BufferBindings` before calling this function.
///
/// Sets `caps.coopmat = true`.
#[allow(clippy::too_many_arguments)]
pub fn emit_coopmat_load_inline(
    b: &mut Builder,
    type_cache: &mut ScalarTypeCache,
    coopmat_cache: &mut CoopMatTypeCache,
    caps: &mut CapabilitiesRequired,
    result_ty: CoopMatKey,
    buf_var_id: Word,
    elem_ptr_ty: Word,
    element_offset_id: Word,
    stride_id: Word,
) -> Result<Word, BodyCodegenError> {
    caps.coopmat = true;

    let mat_type_id = coopmat_cache.get_or_emit(b, type_cache, result_ty);

    // Access the element at element_offset in the SSBO (member 0 of the struct).
    let zero_id = type_cache.get_or_emit_u32_const(b, 0);
    let ptr_id = b.access_chain(elem_ptr_ty, None, buf_var_id, [zero_id, element_offset_id])
        .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;

    // RowMajorKHR = 0.
    let layout_id = type_cache.get_or_emit_u32_const(b, ROW_MAJOR_LAYOUT);

    let result_id = b.cooperative_matrix_load_khr(
        mat_type_id,
        None,
        ptr_id,
        layout_id,
        Some(stride_id),
        None,                  // no MemoryAccess flags
        std::iter::empty(),
    )
    .map_err(|e| BodyCodegenError::Rspirv(e.to_string()))?;

    Ok(result_id)
}
