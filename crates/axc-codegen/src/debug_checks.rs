//! M3.17 (FG.4): SPIR-V support for the injected runtime debug-flag SSBO.
//!
//! Under `--debug`, one SSBO — `buffer[u32]` of length 2 words (`flag[0]` =
//! precondition violations, `flag[1]` = postcondition violations) — is injected at
//! descriptor binding `num_user_buffers`, built by reusing `emit_buffer_globals` on
//! an EXTENDED slot list so the DSL/pool/bind-loop treat it like any other SSBO
//! (§7 of the spec). Check emission (compare + `OpAtomicOr`) lives in `body.rs`
//! because it needs `BodyEmitter`, which is private to that module.

use rspirv::dr::Builder;
use axc_hir::{BufferAccess, BufferBindingSlot, BufferTy, ScalarTy};
use crate::body::ScalarTypeCache;
use crate::buffers::{emit_buffer_globals, BufferBindings};

/// Descriptor-bookkeeping name for the injected flag SSBO. Never surfaced to user
/// source and never looked up by name at runtime (the bind loop is keyed on
/// `buffer_position`) — this is purely a diagnostic label.
pub const DEBUG_FLAG_BUFFER_NAME: &str = "__axc_debug_flags";

/// Length (in `u32` words) of the injected flag buffer: `[precondition_bits, postcondition_bits]`.
pub const DEBUG_FLAG_LEN_WORDS: u32 = 2;

/// Emit the user SSBOs plus the injected debug-flag SSBO in ONE `emit_buffer_globals`
/// call, so every decoration is produced by the exact same code path as an ordinary
/// buffer. The flag slot is always appended LAST, at `buffer_position == user_buffers.len()`.
pub fn emit_debug_flag_binding(
    b: &mut Builder,
    type_cache: &mut ScalarTypeCache,
    user_buffers: &[BufferBindingSlot],
) -> BufferBindings {
    let flag_position: u32 = user_buffers.len() as u32;
    let mut slots: Vec<BufferBindingSlot> = user_buffers.to_vec();
    slots.push(BufferBindingSlot {
        name: DEBUG_FLAG_BUFFER_NAME.to_owned(),
        ty: BufferTy { elem: ScalarTy::U32, access: BufferAccess::ReadWrite },
        position: flag_position,
        buffer_position: flag_position,
        span: Default::default(),
    });
    emit_buffer_globals(b, type_cache, &slots)
}

/// Raw SPIR-V `Scope` value for the injected `OpAtomicOr` (CRITICAL-1, §2): the
/// Scope/Semantics operands are `<id>` refs to an `OpConstant`, so only the numeric
/// code is needed (no `rspirv::spirv::Scope` dependency). `QueueFamily` (=5) under
/// `MemoryModel::Vulkan` (coopmat — needs no new capability); `Device` (=1) under
/// `MemoryModel::GLSL450`. Keyed on the SAME `uses_coopmat` predicate as `emit.rs`'s
/// memory-model branch, so the net capability set is unchanged either way.
pub fn debug_atomic_scope_value(uses_coopmat: bool) -> u32 {
    if uses_coopmat { 5 } else { 1 }
}

/// Raw SPIR-V memory-semantics value for `Relaxed` (no ordering — the atomic only
/// needs to be atomic and idempotent, spec §2).
pub const DEBUG_ATOMIC_SEMANTICS_RELAXED: u32 = 0;

#[cfg(test)]
mod tests {
    use super::*;
    use rspirv::spirv::{AddressingModel, MemoryModel, Capability};

    fn make_builder() -> Builder {
        let mut b = Builder::new();
        b.set_version(1, 3);
        b.capability(Capability::Shader);
        b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);
        b
    }

    #[test]
    fn flag_binding_appended_at_num_user_buffers() {
        let mut b = make_builder();
        let mut tc = ScalarTypeCache::new();
        let user = vec![BufferBindingSlot {
            name: "x".into(),
            ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadOnly },
            position: 0,
            buffer_position: 0,
            span: Default::default(),
        }];
        let bindings = emit_debug_flag_binding(&mut b, &mut tc, &user);
        assert_eq!(bindings.var_ids.len(), 2, "expected user buffer + injected flag buffer");
        assert!(bindings.var_ids.contains_key(&1), "flag buffer must be at buffer_position 1 (== num_user_buffers)");
    }

    #[test]
    fn scope_value_keyed_on_coopmat() {
        assert_eq!(debug_atomic_scope_value(false), 1, "GLSL450/non-coopmat must use Device scope");
        assert_eq!(debug_atomic_scope_value(true), 5, "Vulkan/coopmat must use QueueFamily scope");
    }
}
