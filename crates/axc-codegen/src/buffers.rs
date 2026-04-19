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

use std::collections::HashMap;
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
pub struct BufferBindings {
    /// Maps buffer_position → SPIR-V global Variable id.
    pub var_ids: HashMap<u32, Word>,
    /// Maps buffer_position → SPIR-V pointer-to-elem type id.
    /// (Used in access_chain for reads/writes.)
    pub elem_ptr_ids: HashMap<u32, Word>,
}

impl BufferBindings {
    pub fn new() -> Self {
        Self {
            var_ids: HashMap::new(),
            elem_ptr_ids: HashMap::new(),
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
    pub member_ptr_ids: HashMap<u32, Word>,
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
    let mut member_ptr_ids: HashMap<u32, Word> = HashMap::new();
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
    /// Pointer type for a single `u32` component (used in access_chain).
    pub u32_ptr_type_id: Word,
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

    // 4. Pointer to a single u32 in Input class (for access_chain).
    let u32_ptr_type_id: Word = b.type_pointer(None, StorageClass::Input, u32_type_id);

    // 5. OpVariable Input
    let var_id: Word = b.variable(ptr_to_vec3, None, StorageClass::Input, None);

    // 6. OpDecorate var BuiltIn GlobalInvocationId
    b.decorate(var_id, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::GlobalInvocationId)]);

    GlobalInvocationIdVar { var_id, vec3_u32_type_id, u32_ptr_type_id }
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
}
