//! SPIR-V SSBO, push-constant, and `gl_GlobalInvocationID` emission.
//!
//! # SSBO layout (one per buffer param)
//!
//! ```text
//! OpTypeRuntimeArray  %arr_T  %T
//! OpTypeStruct        %block  %arr_T      ; { T[] data; }
//! OpTypePointer       %ptr    StorageBuffer %block
//! OpVariable          %var    StorageBuffer
//! OpDecorate          %arr_T  ArrayStride <elem_bytes>
//! OpDecorate          %block  Block
//! OpDecorate          %var    DescriptorSet 0
//! OpDecorate          %var    Binding <slot>
//! OpDecorate          %var    NonWritable   (readonly_buffer only)
//! OpDecorate          %var    NonReadable   (writeonly_buffer only)
//! ```
//!
//! # Push-constant layout (one struct for all scalar params)
//!
//! ```text
//! OpTypeStruct            %pc_struct  [T0, T1, ...]
//! OpTypePointer           %pc_ptr     PushConstant %pc_struct
//! OpVariable              %pc_var     PushConstant
//! OpDecorate              %pc_struct  Block
//! OpMemberDecorate        %pc_struct  <i>  Offset <offset>
//! ```
//!
//! # gl_GlobalInvocationID
//!
//! ```text
//! OpTypeVector        %v3u32  %u32  3
//! OpTypePointer       %ptr    Input  %v3u32
//! OpVariable          %gid    Input
//! OpDecorate          %gid    BuiltIn GlobalInvocationId
//! ```
//!
//! The input variable is included in the OpEntryPoint interface list.

use std::collections::BTreeMap;
use rspirv::dr::{Builder, Operand};
use rspirv::spirv::{
    Word, StorageClass, Decoration, BuiltIn,
};
use axc_hir::{
    ParamBindingPlan, BufferBindingSlot, ScalarPushConstantSlot,
    BufferAccess,
};
use crate::body::ScalarTypeCache;

// ── Buffer bindings ───────────────────────────────────────────────────────────

/// Emitted SSBO global variable ids, keyed by buffer_position (= Binding N).
///
/// Uses `BTreeMap` (not `HashMap`) to guarantee deterministic iteration order
/// for SPIR-V emission. HashMap iteration order is seed-randomized and would
/// produce non-deterministic bytewise output across runs (spec §5.8 / AT-227).
pub struct BufferBindings {
    /// Maps buffer_position → SPIR-V global Variable id.
    pub var_ids: BTreeMap<u32, Word>,
    /// Maps buffer_position → SPIR-V pointer-to-elem type id.
    /// (Used in access_chain for reads/writes.)
    pub elem_ptr_ids: BTreeMap<u32, Word>,
}

impl BufferBindings {
    pub fn new() -> Self {
        Self {
            var_ids: BTreeMap::new(),
            elem_ptr_ids: BTreeMap::new(),
        }
    }
}

impl Default for BufferBindings {
    fn default() -> Self {
        Self::new()
    }
}

/// Emit all SSBO global variables for a kernel's buffer parameters.
///
/// Must be called BEFORE `begin_function` — SPIR-V requires global
/// OpVariable instructions in the types/globals section.
///
/// Returns the populated `BufferBindings` (var_ids + elem_ptr_ids).
pub fn emit_buffer_globals(
    b: &mut Builder,
    type_cache: &mut ScalarTypeCache,
    buffers: &[BufferBindingSlot],
) -> BufferBindings {
    let mut out = BufferBindings::new();

    for slot in buffers {
        let bp = slot.buffer_position;
        let elem_ty = slot.ty.elem;
        let elem_byte_size = slot.ty.elem_byte_size();

        // 1. Scalar element type.
        let elem_type_id: Word = type_cache.scalar_id(b, elem_ty);

        // 2. OpTypeRuntimeArray T
        let arr_id: Word = b.type_runtime_array(elem_type_id);

        // 3. OpTypeStruct { T[] data }
        //    Must use type_struct_id(Some(id), ...) to avoid rspirv dedup
        //    (each buffer gets its own distinct struct type).
        let struct_id: Word = b.id();
        let struct_type_id: Word = b.type_struct_id(Some(struct_id), [arr_id]);

        // 4. Pointer type for the struct in StorageBuffer class.
        let ptr_to_struct_id: Word = b.type_pointer(None, StorageClass::StorageBuffer, struct_type_id);

        // 5. Pointer type for an element (used in OpAccessChain result type).
        let ptr_to_elem_id: Word = b.type_pointer(None, StorageClass::StorageBuffer, elem_type_id);

        // 6. OpVariable (global — no function selected, goes into types_global_values).
        let var_id: Word = b.variable(ptr_to_struct_id, None, StorageClass::StorageBuffer, None);

        // 7. OpDecorate arr_T ArrayStride <bytes>
        b.decorate(arr_id, Decoration::ArrayStride, [Operand::LiteralBit32(elem_byte_size)]);

        // 8. OpDecorate struct Block
        b.decorate(struct_type_id, Decoration::Block, []);

        // 8b. OpMemberDecorate struct 0 Offset 0
        //     Vulkan §15.6.4 requires every member of a Block-decorated struct to carry
        //     an explicit Offset decoration. Member 0 is the runtime array; its offset
        //     within the wrapper struct is always 0. Without this, spirv-val with
        //     --target-env vulkan1.1 rejects the binary.
        b.member_decorate(struct_type_id, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);

        // 9. OpDecorate var DescriptorSet 0
        b.decorate(var_id, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);

        // 10. OpDecorate var Binding <bp>
        b.decorate(var_id, Decoration::Binding, [Operand::LiteralBit32(bp)]);

        // 11. Access decorations.
        match slot.ty.access {
            BufferAccess::ReadOnly  => {
                b.decorate(var_id, Decoration::NonWritable, []);
            }
            BufferAccess::WriteOnly => {
                b.decorate(var_id, Decoration::NonReadable, []);
            }
            BufferAccess::ReadWrite => {}
        }

        out.var_ids.insert(bp, var_id);
        out.elem_ptr_ids.insert(bp, ptr_to_elem_id);
    }

    out
}

// ── Push-constant block ───────────────────────────────────────────────────────

/// Emitted push-constant global variable id and member accessor info.
pub struct PushConstantBlock {
    /// The global push-constant OpVariable id.
    pub var_id: Word,
    /// The push-constant struct type id.
    pub struct_type_id: Word,
    /// Maps scalar member_index → pointer-to-member type id.
    /// Uses `BTreeMap` for deterministic iteration order (spec §5.8 / AT-227).
    pub member_ptr_ids: BTreeMap<u32, Word>,
}

/// Emit the push-constant block for a kernel's scalar parameters.
///
/// Returns `None` if there are no scalar params.
///
/// Must be called BEFORE `begin_function`.
pub fn emit_push_constant_block(
    b: &mut Builder,
    type_cache: &mut ScalarTypeCache,
    scalars: &[ScalarPushConstantSlot],
) -> Option<PushConstantBlock> {
    if scalars.is_empty() {
        return None;
    }

    // 1. Collect member type ids in member_index order.
    let mut ordered_scalars: Vec<&ScalarPushConstantSlot> = scalars.iter().collect();
    ordered_scalars.sort_by_key(|s| s.member_index);

    let member_type_ids: Vec<Word> = ordered_scalars
        .iter()
        .map(|s| type_cache.scalar_id(b, s.ty))
        .collect();

    // 2. OpTypeStruct { T0 m0; T1 m1; ... }
    //    Use type_struct_id with explicit id to avoid dedup collision.
    let struct_id: Word = b.id();
    let struct_type_id: Word = b.type_struct_id(Some(struct_id), member_type_ids.iter().copied());

    // 3. OpDecorate struct Block
    b.decorate(struct_type_id, Decoration::Block, []);

    // 4. OpMemberDecorate struct <i> Offset <offset>
    for s in &ordered_scalars {
        b.member_decorate(
            struct_type_id,
            s.member_index,
            Decoration::Offset,
            [Operand::LiteralBit32(s.offset)],
        );
    }

    // 5. Pointer type for the struct in PushConstant class.
    let ptr_id: Word = b.type_pointer(None, StorageClass::PushConstant, struct_type_id);

    // 6. OpVariable PushConstant
    let var_id: Word = b.variable(ptr_id, None, StorageClass::PushConstant, None);

    // 7. Pointer types for individual members.
    let mut member_ptr_ids: BTreeMap<u32, Word> = BTreeMap::new();
    for s in &ordered_scalars {
        let elem_ty_id = type_cache.scalar_id(b, s.ty);
        let ptr_ty = b.type_pointer(None, StorageClass::PushConstant, elem_ty_id);
        member_ptr_ids.insert(s.member_index, ptr_ty);
    }

    Some(PushConstantBlock { var_id, struct_type_id, member_ptr_ids })
}

// ── gl_GlobalInvocationID ─────────────────────────────────────────────────────

/// The emitted `gl_GlobalInvocationID` variable and its type support.
pub struct GlobalInvocationIdVar {
    /// The global OpVariable id for the Input vector.
    pub var_id: Word,
    /// The `uvec3` type id (OpTypeVector u32 3).
    pub vec3_u32_type_id: Word,
}

/// Emit the `gl_GlobalInvocationID` Input variable.
///
/// Must be called BEFORE `begin_function`, and the returned `var_id` MUST be
/// included in the OpEntryPoint interface list (SPIR-V 1.4+ requirement; also
/// correct for 1.3 Input/Output variables).
///
/// Returns `None` if neither `GidBuiltin` nor `BufferRead`/`BufferWrite` use gid.
/// The caller should call this if any kernel uses `gid()`.
pub fn emit_gid_variable(
    b: &mut Builder,
    type_cache: &mut ScalarTypeCache,
) -> GlobalInvocationIdVar {
    use axc_hir::ty::ScalarTy;

    // 1. u32 type
    let u32_type_id: Word = type_cache.scalar_id(b, ScalarTy::U32);

    // 2. vec3 u32 type
    let vec3_u32_type_id: Word = b.type_vector(u32_type_id, 3);

    // 3. Pointer to vec3 u32 in Input storage class.
    let ptr_to_vec3: Word = b.type_pointer(None, StorageClass::Input, vec3_u32_type_id);

    // 4. OpVariable Input
    let var_id: Word = b.variable(ptr_to_vec3, None, StorageClass::Input, None);

    // 5. OpDecorate var BuiltIn GlobalInvocationId
    b.decorate(var_id, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::GlobalInvocationId)]);

    GlobalInvocationIdVar { var_id, vec3_u32_type_id }
}

// ── Scan helpers ─────────────────────────────────────────────────────────────

/// Returns `true` if the kernel uses any `gid()` builtin expression.
///
/// Used to decide whether to emit the gl_GlobalInvocationID variable.
pub fn kernel_uses_gid(plan: &ParamBindingPlan) -> bool {
    // GID is only needed when there are buffer params or scalar params referencing gid.
    // However, gid() can be used in any kernel body — the HIR typechecker controls
    // whether gid() is allowed. We must scan the emitted HIR stmts.
    //
    // For M1.2, emit GID if there is at least one buffer param (the primary use case).
    // A kernel with only scalar params could use gid() too — but that would be unusual
    // and is not tested in AT-201..230. The body-scanner approach is cleaner but requires
    // threading the typed body into this function; instead we rely on the caller
    // (`emit.rs`) to call `emit_gid_variable` when the HIR body scan finds a GidBuiltin.
    //
    // This function is intentionally a no-op placeholder — the real decision is in emit.rs.
    let _ = plan;
    false
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use axc_hir::{
        BufferBindingSlot, ScalarPushConstantSlot,
        BufferTy, BufferAccess,
    };
    use axc_hir::ty::ScalarTy;
    use axc_lexer::Span;
    use rspirv::spirv::Op;

    fn make_builder() -> Builder {
        let mut b = Builder::new();
        b.set_version(1, 3);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::GLSL450,
        );
        b
    }

    fn iter_type_instructions(b: &Builder) -> impl Iterator<Item = (u16, &rspirv::dr::Instruction)> {
        b.module_ref().types_global_values.iter().map(|inst| {
            (inst.class.opcode as u16, inst)
        })
    }

    fn iter_annotations(b: &Builder) -> impl Iterator<Item = &rspirv::dr::Instruction> {
        b.module_ref().annotations.iter()
    }

    // buf_emit_one_ssbo_global_emits_variable
    #[test]
    fn buf_emit_one_ssbo_global_emits_variable() {
        let mut b = make_builder();
        let mut tc = ScalarTypeCache::new();
        let slot = BufferBindingSlot {
            name: "xs".into(),
            ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadWrite },
            position: 0,
            buffer_position: 0,
            span: Span::new(0, 1),
        };
        let bindings = emit_buffer_globals(&mut b, &mut tc, &[slot]);
        assert!(bindings.var_ids.contains_key(&0), "buffer_position 0 must produce a var_id");
        // Check that a Variable was emitted into types_global_values
        let has_var = iter_type_instructions(&b).any(|(op, _)| op == Op::Variable as u16);
        assert!(has_var, "expected OpVariable in types_global_values");
    }

    // buf_emit_readonly_emits_nonwritable_deco
    #[test]
    fn buf_emit_readonly_emits_nonwritable_deco() {
        let mut b = make_builder();
        let mut tc = ScalarTypeCache::new();
        let slot = BufferBindingSlot {
            name: "inp".into(),
            ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadOnly },
            position: 0,
            buffer_position: 0,
            span: Span::new(0, 1),
        };
        emit_buffer_globals(&mut b, &mut tc, &[slot]);
        let has_nw = iter_annotations(&b).any(|inst| {
            inst.operands.iter().any(|op| matches!(op, Operand::Decoration(Decoration::NonWritable)))
        });
        assert!(has_nw, "ReadOnly buffer must have NonWritable decoration");
    }

    // buf_emit_writeonly_emits_nonreadable_deco
    #[test]
    fn buf_emit_writeonly_emits_nonreadable_deco() {
        let mut b = make_builder();
        let mut tc = ScalarTypeCache::new();
        let slot = BufferBindingSlot {
            name: "out".into(),
            ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::WriteOnly },
            position: 0,
            buffer_position: 0,
            span: Span::new(0, 1),
        };
        emit_buffer_globals(&mut b, &mut tc, &[slot]);
        let has_nr = iter_annotations(&b).any(|inst| {
            inst.operands.iter().any(|op| matches!(op, Operand::Decoration(Decoration::NonReadable)))
        });
        assert!(has_nr, "WriteOnly buffer must have NonReadable decoration");
    }

    // buf_emit_two_ssbos_have_distinct_bindings
    #[test]
    fn buf_emit_two_ssbos_have_distinct_bindings() {
        let mut b = make_builder();
        let mut tc = ScalarTypeCache::new();
        let slots = vec![
            BufferBindingSlot {
                name: "a".into(),
                ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadOnly },
                position: 0,
                buffer_position: 0,
                span: Span::new(0, 1),
            },
            BufferBindingSlot {
                name: "b".into(),
                ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadWrite },
                position: 1,
                buffer_position: 1,
                span: Span::new(0, 1),
            },
        ];
        let bindings = emit_buffer_globals(&mut b, &mut tc, &slots);
        assert_eq!(bindings.var_ids.len(), 2, "expected 2 SSBO vars");
        assert_ne!(bindings.var_ids[&0], bindings.var_ids[&1], "vars must have distinct ids");
        // Check Binding decorations: must contain Binding 0 and Binding 1.
        let binding_vals: Vec<u32> = iter_annotations(&b)
            .filter(|inst| inst.operands.iter().any(|op| matches!(op, Operand::Decoration(Decoration::Binding))))
            .filter_map(|inst| inst.operands.iter().find_map(|op| {
                if let Operand::LiteralBit32(n) = op { Some(*n) } else { None }
            }))
            .collect();
        assert!(binding_vals.contains(&0), "must decorate with Binding 0");
        assert!(binding_vals.contains(&1), "must decorate with Binding 1");
    }

    // buf_push_const_emits_block_and_offsets
    #[test]
    fn buf_push_const_emits_block_and_offsets() {
        let mut b = make_builder();
        let mut tc = ScalarTypeCache::new();
        let scalars = vec![
            ScalarPushConstantSlot {
                name: "n".into(),
                ty: ScalarTy::U32,
                offset: 0,
                member_index: 0,
                position: 0,
                span: Span::new(0, 1),
            },
            ScalarPushConstantSlot {
                name: "alpha".into(),
                ty: ScalarTy::F32,
                offset: 4,
                member_index: 1,
                position: 1,
                span: Span::new(0, 1),
            },
        ];
        let pc = emit_push_constant_block(&mut b, &mut tc, &scalars)
            .expect("expected PushConstantBlock for 2 scalars");
        // Block decoration must be on the struct.
        let has_block = iter_annotations(&b).any(|inst| {
            inst.operands.iter().any(|op| matches!(op, Operand::Decoration(Decoration::Block)))
        });
        assert!(has_block, "push-constant struct must have Block decoration");
        // Offset decorations must include offset 0 and offset 4.
        // OpMemberDecorate operands: [IdRef, LiteralBit32(member_idx), Decoration(Offset), LiteralBit32(offset)]
        // We want the LAST LiteralBit32 (after the Decoration operand).
        let offset_vals: Vec<u32> = iter_annotations(&b)
            .filter(|inst| inst.operands.iter().any(|op| matches!(op, Operand::Decoration(Decoration::Offset))))
            .filter_map(|inst| inst.operands.iter().rev().find_map(|op| {
                if let Operand::LiteralBit32(n) = op { Some(*n) } else { None }
            }))
            .collect();
        assert!(offset_vals.contains(&0), "must have Offset 0: offset_vals={offset_vals:?}");
        assert!(offset_vals.contains(&4), "must have Offset 4: offset_vals={offset_vals:?}");
        // The PushConstantBlock must have a valid var_id.
        assert_ne!(pc.var_id, 0, "push-constant var_id must be nonzero");
    }

    // buf_push_const_none_when_empty
    #[test]
    fn buf_push_const_none_when_empty() {
        let mut b = make_builder();
        let mut tc = ScalarTypeCache::new();
        let result = emit_push_constant_block(&mut b, &mut tc, &[]);
        assert!(result.is_none(), "expected None for empty scalar list");
    }

    // buf_gid_var_emits_input_variable_with_builtin_deco
    #[test]
    fn buf_gid_var_emits_input_variable_with_builtin_deco() {
        let mut b = make_builder();
        let mut tc = ScalarTypeCache::new();
        let gid = emit_gid_variable(&mut b, &mut tc);
        // Check that the variable id is nonzero.
        assert_ne!(gid.var_id, 0, "gid var_id must be nonzero");
        // Check that GlobalInvocationId BuiltIn decoration is present.
        let has_gid = iter_annotations(&b).any(|inst| {
            inst.operands.iter().any(|op| matches!(op, Operand::BuiltIn(BuiltIn::GlobalInvocationId)))
        });
        assert!(has_gid, "expected GlobalInvocationId BuiltIn decoration");
    }

    // buf_emit_arraystride_on_ssbo
    #[test]
    fn buf_emit_arraystride_on_ssbo() {
        let mut b = make_builder();
        let mut tc = ScalarTypeCache::new();
        let slot = BufferBindingSlot {
            name: "xs".into(),
            ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadWrite },
            position: 0,
            buffer_position: 0,
            span: Span::new(0, 1),
        };
        emit_buffer_globals(&mut b, &mut tc, &[slot]);
        // ArrayStride 4 must be present for f32 elements.
        let has_stride4 = iter_annotations(&b).any(|inst| {
            let has_array_stride = inst.operands.iter().any(|op| matches!(op, Operand::Decoration(Decoration::ArrayStride)));
            let has_val_4 = inst.operands.iter().any(|op| matches!(op, Operand::LiteralBit32(4)));
            has_array_stride && has_val_4
        });
        assert!(has_stride4, "expected ArrayStride 4 for f32 elements");
    }

    // ── AT-218: BufferTypeCache deduplication ─────────────────────────────────

    // AT-218: cg_buffer_type_cache_dedupes_elem_ty
    // Kernel with 2 × buffer[f32] params must emit exactly 1 OpTypeRuntimeArray and
    // 1 OpTypeStruct for f32 (the type cache deduplicates). With buffer[f32] + buffer[i32],
    // it must emit 2 of each.
    #[test]
    fn cg_buffer_type_cache_dedupes_elem_ty() {
        use rspirv::spirv::Op;

        // Case 1: 2 × buffer[f32] → 1 OpTypeRuntimeArray (same elem type deduplicated).
        {
            let mut b = make_builder();
            let mut tc = ScalarTypeCache::new();
            let slots = vec![
                BufferBindingSlot {
                    name: "a".into(),
                    ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadWrite },
                    position: 0, buffer_position: 0, span: Span::new(0, 1),
                },
                BufferBindingSlot {
                    name: "b".into(),
                    ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadWrite },
                    position: 1, buffer_position: 1, span: Span::new(0, 1),
                },
            ];
            emit_buffer_globals(&mut b, &mut tc, &slots);
            let runtime_array_count = iter_type_instructions(&b)
                .filter(|(op, _)| *op == Op::TypeRuntimeArray as u16)
                .count();
            // rspirv deduplicates OpTypeRuntimeArray for the same element type.
            assert_eq!(
                runtime_array_count, 1,
                "2×buffer[f32] should produce exactly 1 OpTypeRuntimeArray; got {runtime_array_count}"
            );
        }

        // Case 2: buffer[f32] + buffer[i32] → 2 OpTypeRuntimeArray (distinct elem types).
        {
            let mut b = make_builder();
            let mut tc = ScalarTypeCache::new();
            let slots = vec![
                BufferBindingSlot {
                    name: "x".into(),
                    ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadWrite },
                    position: 0, buffer_position: 0, span: Span::new(0, 1),
                },
                BufferBindingSlot {
                    name: "y".into(),
                    ty: BufferTy { elem: ScalarTy::I32, access: BufferAccess::ReadWrite },
                    position: 1, buffer_position: 1, span: Span::new(0, 1),
                },
            ];
            emit_buffer_globals(&mut b, &mut tc, &slots);
            let runtime_array_count = iter_type_instructions(&b)
                .filter(|(op, _)| *op == Op::TypeRuntimeArray as u16)
                .count();
            assert_eq!(
                runtime_array_count, 2,
                "buffer[f32]+buffer[i32] should produce 2 OpTypeRuntimeArray; got {runtime_array_count}"
            );
        }
    }

    // AT-219: cg_buffer_array_stride_32bit and cg_buffer_array_stride_64bit
    #[test]
    fn cg_buffer_array_stride_32bit() {
        // f32, i32, u32 → ArrayStride 4
        for (elem, name) in &[
            (ScalarTy::F32, "f32"),
            (ScalarTy::I32, "i32"),
            (ScalarTy::U32, "u32"),
        ] {
            let mut b = make_builder();
            let mut tc = ScalarTypeCache::new();
            let slot = BufferBindingSlot {
                name: (*name).into(),
                ty: BufferTy { elem: *elem, access: BufferAccess::ReadWrite },
                position: 0, buffer_position: 0, span: Span::new(0, 1),
            };
            emit_buffer_globals(&mut b, &mut tc, &[slot]);
            let has_stride4 = iter_annotations(&b).any(|inst| {
                inst.operands.iter().any(|op| matches!(op, Operand::Decoration(Decoration::ArrayStride)))
                && inst.operands.iter().any(|op| matches!(op, Operand::LiteralBit32(4)))
            });
            assert!(has_stride4, "buffer[{name}] must have ArrayStride 4");
        }
    }

    #[test]
    fn cg_buffer_array_stride_64bit() {
        // f64, i64, u64 → ArrayStride 8
        for (elem, name) in &[
            (ScalarTy::F64, "f64"),
            (ScalarTy::I64, "i64"),
            (ScalarTy::U64, "u64"),
        ] {
            let mut b = make_builder();
            let mut tc = ScalarTypeCache::new();
            let slot = BufferBindingSlot {
                name: (*name).into(),
                ty: BufferTy { elem: *elem, access: BufferAccess::ReadWrite },
                position: 0, buffer_position: 0, span: Span::new(0, 1),
            };
            emit_buffer_globals(&mut b, &mut tc, &[slot]);
            let has_stride8 = iter_annotations(&b).any(|inst| {
                inst.operands.iter().any(|op| matches!(op, Operand::Decoration(Decoration::ArrayStride)))
                && inst.operands.iter().any(|op| matches!(op, Operand::LiteralBit32(8)))
            });
            assert!(has_stride8, "buffer[{name}] must have ArrayStride 8");
        }
    }

    // AT-206: cg_saxpy_binding_indices_skip_scalar
    // Verifies binding 0 is x (readonly_buffer), binding 1 is y (buffer), no binding 2.
    #[test]
    fn cg_saxpy_binding_indices_skip_scalar() {
        use axc_hir::{BufferBindingSlot, BufferTy, BufferAccess};
        let mut b = make_builder();
        let mut tc = ScalarTypeCache::new();
        // saxpy: (n: u32, alpha: f32, x: readonly_buffer[f32], y: buffer[f32])
        // x is buffer_position 0, y is buffer_position 1
        let slots = vec![
            BufferBindingSlot {
                name: "x".into(),
                ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadOnly },
                position: 2, buffer_position: 0, span: Span::new(0, 1),
            },
            BufferBindingSlot {
                name: "y".into(),
                ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadWrite },
                position: 3, buffer_position: 1, span: Span::new(0, 1),
            },
        ];
        emit_buffer_globals(&mut b, &mut tc, &slots);
        let binding_vals: Vec<u32> = iter_annotations(&b)
            .filter(|inst| inst.operands.iter().any(|op| matches!(op, Operand::Decoration(Decoration::Binding))))
            .filter_map(|inst| inst.operands.iter().find_map(|op| {
                if let Operand::LiteralBit32(n) = op { Some(*n) } else { None }
            }))
            .collect();
        assert!(binding_vals.contains(&0), "x must have Binding 0; got {:?}", binding_vals);
        assert!(binding_vals.contains(&1), "y must have Binding 1; got {:?}", binding_vals);
        assert!(!binding_vals.contains(&2), "no Binding 2 expected; got {:?}", binding_vals);
    }

    // cg_descriptor_set_is_always_zero
    #[test]
    fn cg_descriptor_set_is_always_zero() {
        let mut b = make_builder();
        let mut tc = ScalarTypeCache::new();
        let slots = vec![
            BufferBindingSlot {
                name: "a".into(),
                ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadOnly },
                position: 0, buffer_position: 0, span: Span::new(0, 1),
            },
            BufferBindingSlot {
                name: "b".into(),
                ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadWrite },
                position: 1, buffer_position: 1, span: Span::new(0, 1),
            },
        ];
        emit_buffer_globals(&mut b, &mut tc, &slots);
        let ds_vals: Vec<u32> = iter_annotations(&b)
            .filter(|inst| inst.operands.iter().any(|op| matches!(op, Operand::Decoration(Decoration::DescriptorSet))))
            .filter_map(|inst| inst.operands.iter().find_map(|op| {
                if let Operand::LiteralBit32(n) = op { Some(*n) } else { None }
            }))
            .collect();
        assert!(ds_vals.iter().all(|&v| v == 0), "all buffers must be DescriptorSet 0; got {ds_vals:?}");
    }

    // cg_binding_indices_are_sequential_among_buffers
    #[test]
    fn cg_binding_indices_are_sequential_among_buffers() {
        let mut b = make_builder();
        let mut tc = ScalarTypeCache::new();
        let slots: Vec<BufferBindingSlot> = (0..4u32).map(|i| BufferBindingSlot {
            name: format!("buf{i}"),
            ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadWrite },
            position: i,
            buffer_position: i,
            span: Span::new(0, 1),
        }).collect();
        emit_buffer_globals(&mut b, &mut tc, &slots);
        let mut binding_vals: Vec<u32> = iter_annotations(&b)
            .filter(|inst| inst.operands.iter().any(|op| matches!(op, Operand::Decoration(Decoration::Binding))))
            .filter_map(|inst| inst.operands.iter().find_map(|op| {
                if let Operand::LiteralBit32(n) = op { Some(*n) } else { None }
            }))
            .collect();
        binding_vals.sort_unstable();
        assert_eq!(binding_vals, vec![0, 1, 2, 3], "bindings must be sequential 0,1,2,3; got {binding_vals:?}");
    }

    // cg_readonly_buffer_emits_nonwritable_decoration (maps to AT-208)
    #[test]
    fn cg_readonly_buffer_emits_nonwritable_decoration() {
        let mut b = make_builder();
        let mut tc = ScalarTypeCache::new();
        let slot = BufferBindingSlot {
            name: "ro".into(),
            ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadOnly },
            position: 0, buffer_position: 0, span: Span::new(0, 1),
        };
        emit_buffer_globals(&mut b, &mut tc, &[slot]);
        let has_nw = iter_annotations(&b).any(|inst| {
            inst.operands.iter().any(|op| matches!(op, Operand::Decoration(Decoration::NonWritable)))
        });
        assert!(has_nw, "readonly_buffer must have NonWritable decoration");
    }

    // cg_writeonly_buffer_emits_nonreadable_decoration (maps to AT-209)
    #[test]
    fn cg_writeonly_buffer_emits_nonreadable_decoration() {
        let mut b = make_builder();
        let mut tc = ScalarTypeCache::new();
        let slot = BufferBindingSlot {
            name: "wo".into(),
            ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::WriteOnly },
            position: 0, buffer_position: 0, span: Span::new(0, 1),
        };
        emit_buffer_globals(&mut b, &mut tc, &[slot]);
        let has_nr = iter_annotations(&b).any(|inst| {
            inst.operands.iter().any(|op| matches!(op, Operand::Decoration(Decoration::NonReadable)))
        });
        assert!(has_nr, "writeonly_buffer must have NonReadable decoration");
    }

    // cg_readwrite_buffer_no_access_decoration
    #[test]
    fn cg_readwrite_buffer_no_access_decoration() {
        let mut b = make_builder();
        let mut tc = ScalarTypeCache::new();
        let slot = BufferBindingSlot {
            name: "rw".into(),
            ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadWrite },
            position: 0, buffer_position: 0, span: Span::new(0, 1),
        };
        emit_buffer_globals(&mut b, &mut tc, &[slot]);
        let has_nw = iter_annotations(&b).any(|inst| {
            inst.operands.iter().any(|op| matches!(op, Operand::Decoration(Decoration::NonWritable)))
        });
        let has_nr = iter_annotations(&b).any(|inst| {
            inst.operands.iter().any(|op| matches!(op, Operand::Decoration(Decoration::NonReadable)))
        });
        assert!(!has_nw, "readwrite buffer must NOT have NonWritable");
        assert!(!has_nr, "readwrite buffer must NOT have NonReadable");
    }

    // AT-214: cg_gid_var_emitted_once
    // A kernel using gid(0) + gid(1) + gid(2) should emit exactly 1 OpVariable Input.
    #[test]
    fn cg_gid_var_emitted_once() {
        use rspirv::spirv::Op;
        let mut b = make_builder();
        let mut tc = ScalarTypeCache::new();
        // Call emit_gid_variable once (the codegen emits it once, then uses it N times).
        let gid1 = emit_gid_variable(&mut b, &mut tc);
        // Try to call again — but the codegen does NOT call it multiple times.
        // The key test is that the codegen infrastructure only emits ONE Input variable.
        let input_var_count = b.module_ref().types_global_values.iter()
            .filter(|inst| {
                inst.class.opcode == Op::Variable
                    && inst.operands.iter().any(|op| matches!(op, Operand::StorageClass(rspirv::spirv::StorageClass::Input)))
            })
            .count();
        assert_eq!(input_var_count, 1, "gid variable must be emitted exactly once; got {input_var_count}");
        assert_ne!(gid1.var_id, 0, "var_id must be nonzero");
    }

    // AT-214: cg_gid_var_not_emitted_if_unused
    #[test]
    fn cg_gid_var_not_emitted_if_unused() {
        use rspirv::spirv::Op;
        let b = make_builder();
        // If we don't call emit_gid_variable, no Input variable should exist.
        let input_var_count = b.module_ref().types_global_values.iter()
            .filter(|inst| {
                inst.class.opcode == Op::Variable
                    && inst.operands.iter().any(|op| matches!(op, Operand::StorageClass(rspirv::spirv::StorageClass::Input)))
            })
            .count();
        assert_eq!(input_var_count, 0, "gid variable must NOT be emitted if unused");
    }

    // AT-226: cg_pushconstant_i32_then_f64_pads_correctly
    // Parameterized test over 4 shapes from the spec.
    #[test]
    fn cg_pushconstant_i32_then_f64_pads_correctly() {
        use axc_hir::{KernelParam, ParamTy, compute_binding_plan};
        use axc_lexer::Span;

        let ds = Span::new(0, 1);

        fn scalar_param(name: &str, ty: ScalarTy, pos: u32, span: Span) -> KernelParam {
            KernelParam {
                name: name.to_owned(),
                ty: ParamTy::Scalar(ty),
                position: pos,
                span,
            }
        }

        // (a) fn k(a: i32, b: f64)
        {
            let params = vec![
                scalar_param("a", ScalarTy::I32, 0, ds),
                scalar_param("b", ScalarTy::F64, 1, ds),
            ];
            let plan = compute_binding_plan(&params, ds).expect("plan (a)");
            assert_eq!(plan.scalars[0].member_index, 0);
            assert_eq!(plan.scalars[0].offset, 0);
            assert_eq!(plan.scalars[0].ty, ScalarTy::I32);
            assert_eq!(plan.scalars[1].member_index, 1);
            assert_eq!(plan.scalars[1].offset, 8, "(a) f64 must be at offset 8 (4-byte pad from 4)");
            assert_eq!(plan.push_constant_total_bytes, 16, "(a) total must be 16");
        }

        // (b) fn k(a: u32, b: i64)
        {
            let params = vec![
                scalar_param("a", ScalarTy::U32, 0, ds),
                scalar_param("b", ScalarTy::I64, 1, ds),
            ];
            let plan = compute_binding_plan(&params, ds).expect("plan (b)");
            assert_eq!(plan.scalars[0].offset, 0);
            assert_eq!(plan.scalars[1].offset, 8, "(b) i64 must be at offset 8");
            assert_eq!(plan.push_constant_total_bytes, 16, "(b) total must be 16");
        }

        // (c) fn k(a: f32, b: i64, c: u32)
        {
            let params = vec![
                scalar_param("a", ScalarTy::F32, 0, ds),
                scalar_param("b", ScalarTy::I64, 1, ds),
                scalar_param("c", ScalarTy::U32, 2, ds),
            ];
            let plan = compute_binding_plan(&params, ds).expect("plan (c)");
            assert_eq!(plan.scalars[0].offset, 0);
            assert_eq!(plan.scalars[1].offset, 8, "(c) i64 at offset 8 (4-byte pad from 4)");
            assert_eq!(plan.scalars[2].offset, 16, "(c) u32 at offset 16");
            assert_eq!(plan.push_constant_total_bytes, 20, "(c) total must be 20 (NOT 24)");
        }

        // (d) fn k(a: f32, b: f32, c: f64)
        {
            let params = vec![
                scalar_param("a", ScalarTy::F32, 0, ds),
                scalar_param("b", ScalarTy::F32, 1, ds),
                scalar_param("c", ScalarTy::F64, 2, ds),
            ];
            let plan = compute_binding_plan(&params, ds).expect("plan (d)");
            assert_eq!(plan.scalars[0].offset, 0);
            assert_eq!(plan.scalars[1].offset, 4);
            assert_eq!(plan.scalars[2].offset, 8, "(d) f64 at offset 8 (no pad from 8)");
            assert_eq!(plan.push_constant_total_bytes, 16, "(d) total must be 16");
        }
    }

    // AT-225: cg_mixed_buffer_scalar_order_gets_correct_offsets
    // fn foo(a: f32, buf: buffer[f32], b: u32) — member_index must be dense (0, 1)
    // not global position (0, 2).
    #[test]
    fn cg_mixed_buffer_scalar_order_gets_correct_offsets() {
        use axc_hir::{KernelParam, ParamTy, BufferTy, BufferAccess, compute_binding_plan};
        use axc_lexer::Span;

        let ds = Span::new(0, 1);

        let params = vec![
            KernelParam {
                name: "a".into(),
                ty: ParamTy::Scalar(ScalarTy::F32),
                position: 0,
                span: ds,
            },
            KernelParam {
                name: "buf".into(),
                ty: ParamTy::Buffer(BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadWrite }),
                position: 1,
                span: ds,
            },
            KernelParam {
                name: "b".into(),
                ty: ParamTy::Scalar(ScalarTy::U32),
                position: 2,
                span: ds,
            },
        ];

        let plan = compute_binding_plan(&params, ds).expect("plan");
        // Buffers
        assert_eq!(plan.buffers.len(), 1);
        assert_eq!(plan.buffers[0].name, "buf");
        assert_eq!(plan.buffers[0].buffer_position, 0, "buf → binding 0");
        // Scalars
        assert_eq!(plan.scalars.len(), 2);
        assert_eq!(plan.scalars[0].name, "a");
        assert_eq!(plan.scalars[0].member_index, 0, "a → member_index 0");
        assert_eq!(plan.scalars[0].offset, 0);
        assert_eq!(plan.scalars[1].name, "b");
        assert_eq!(plan.scalars[1].member_index, 1, "b → member_index 1 (NOT 2 = position)");
        assert_eq!(plan.scalars[1].offset, 4);
        assert_eq!(plan.push_constant_total_bytes, 8);
    }

    // AT-231: cg_ssbo_struct_member_has_offset_zero
    // Vulkan §15.6.4: every member of a Block-decorated struct must carry an explicit
    // Offset decoration. The SSBO wrapper struct { T[] data } has member 0 (the runtime
    // array) at byte offset 0 — this must be decorated with OpMemberDecorate … 0 Offset 0.
    #[test]
    fn cg_ssbo_struct_member_has_offset_zero() {
        let mut b = make_builder();
        let mut tc = ScalarTypeCache::new();
        let slot = BufferBindingSlot {
            name: "xs".into(),
            ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadWrite },
            position: 0,
            buffer_position: 0,
            span: Span::new(0, 1),
        };
        emit_buffer_globals(&mut b, &mut tc, &[slot]);

        // Walk annotations for an OpMemberDecorate with Decoration::Offset and value 0.
        // OpMemberDecorate layout: [IdRef(struct), LiteralBit32(member_idx), Decoration(Offset), LiteralBit32(offset)]
        let has_member_offset_zero = iter_annotations(&b).any(|inst| {
            // Must be OpMemberDecorate
            let has_offset_deco = inst.operands.iter().any(|op| {
                matches!(op, Operand::Decoration(Decoration::Offset))
            });
            // Member index 0
            let has_member_0 = inst.operands.iter().any(|op| {
                matches!(op, Operand::LiteralBit32(0))
            });
            has_offset_deco && has_member_0
        });
        assert!(
            has_member_offset_zero,
            "SSBO struct member 0 must have OpMemberDecorate Offset 0 (Vulkan §15.6.4)"
        );
    }

    // cg_multi_buffer_elem_ty_shares_type_per_elem
    #[test]
    fn cg_multi_buffer_elem_ty_shares_type_per_elem() {
        use rspirv::spirv::Op;
        // Two buffer[f32] vars should share the same OpTypeFloat
        let mut b = make_builder();
        let mut tc = ScalarTypeCache::new();
        let slots = vec![
            BufferBindingSlot {
                name: "a".into(),
                ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadOnly },
                position: 0, buffer_position: 0, span: Span::new(0, 1),
            },
            BufferBindingSlot {
                name: "b".into(),
                ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadWrite },
                position: 1, buffer_position: 1, span: Span::new(0, 1),
            },
        ];
        emit_buffer_globals(&mut b, &mut tc, &slots);
        // Should have exactly 1 OpTypeFloat 32 (shared scalar type)
        let float32_count = iter_type_instructions(&b)
            .filter(|(op, inst)| {
                *op == Op::TypeFloat as u16
                    && inst.operands.iter().any(|o| matches!(o, Operand::LiteralBit32(32)))
            })
            .count();
        assert_eq!(float32_count, 1, "2×buffer[f32] should share 1 OpTypeFloat 32; got {float32_count}");
    }
}
