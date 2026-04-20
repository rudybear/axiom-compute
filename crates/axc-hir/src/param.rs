//! Kernel parameter types and binding plan for M1.2.
//!
//! Every `@kernel fn` declaration may carry zero or more parameters. In M1.2,
//! parameters are either:
//! - Scalar values (pushed via the push-constant block)
//! - Buffer parameters (bound as SSBOs at descriptor set 0)
//!
//! The `ParamBindingPlan` is computed by the HIR lower pass and consumed by
//! the codegen to produce the correct SPIR-V decorations and `OpVariable`
//! declarations.

use axc_lexer::Span;
use crate::ty::ScalarTy;
use crate::buffer::BufferTy;

/// The HIR type of a kernel parameter.
///
/// M1.2 supports scalar values and buffer bindings. Other types (images,
/// shared memory, subgroup-uniform values) are deferred to later milestones.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ty {
    /// A scalar value passed via the push-constant block.
    Scalar(ScalarTy),
    /// An SSBO buffer binding.
    Buffer(BufferTy),
}

impl Ty {
    /// Human-readable name for error messages.
    pub fn display_name(&self) -> &'static str {
        match self {
            Ty::Scalar(s) => s.display_name(),
            Ty::Buffer(b) => b.access.display_name(),
        }
    }
}

/// A validated kernel parameter (after HIR lowering).
///
/// `position` is the 0-based index in the kernel's parameter list.
/// `name` is the source-level identifier.
#[derive(Debug, Clone)]
pub struct KernelParam {
    /// Parameter identifier as written in the source.
    pub name: String,
    /// Resolved HIR type.
    pub ty: Ty,
    /// 0-based position in the kernel's parameter list (determines push-constant
    /// offset ordering for scalars, and descriptor binding slot for buffers).
    pub position: u32,
    /// Source span for diagnostic messages.
    pub span: Span,
}

/// A single buffer binding slot in the descriptor layout.
///
/// `buffer_position` is the 0-based index among BUFFER parameters only
/// (not among all params). This is what becomes `Decoration::Binding N`.
///
/// Example: `fn k(a: f32, x: readonly_buffer[f32], y: buffer[f32])`:
///   - `a`  → push-constant scalar (position=0)
///   - `x`  → buffer_position=0, binding=0
///   - `y`  → buffer_position=1, binding=1
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferBindingSlot {
    /// Parameter name (for diagnostics and codegen variable naming).
    pub name: String,
    /// Buffer type (element type + access mode).
    pub ty: BufferTy,
    /// Position in the kernel's full parameter list.
    pub position: u32,
    /// 0-based index among buffer params only. Used as `Decoration::Binding N`.
    pub buffer_position: u32,
    /// Source span — skipped during serde serialization (not meaningful at runtime).
    #[cfg_attr(feature = "serde", serde(skip))]
    pub span: Span,
}

/// A single scalar parameter occupying a slot in the push-constant block.
///
/// `member_index` is the 0-based index among SCALAR push-constant members only,
/// assigned by a dedicated counter that is independent of `position`.
/// This matches SPIR-V `OpMemberDecorate` member indexing (§4.4.2 spec rule).
///
/// Example: `fn k(x: readonly_buffer[f32], a: f32, b: f32)`:
///   - `x`  → buffer (skipped in scalar counter)
///   - `a`  → position=1, member_index=0, offset=0
///   - `b`  → position=2, member_index=1, offset=4
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalarPushConstantSlot {
    /// Parameter name.
    pub name: String,
    /// Scalar type.
    pub ty: ScalarTy,
    /// Byte offset of this member in the push-constant struct (std430 layout).
    pub offset: u32,
    /// 0-based index among push-constant struct members (independent of `position`).
    /// Used as the member index in `OpMemberDecorate`.
    pub member_index: u32,
    /// 0-based position in the kernel's full parameter list.
    pub position: u32,
    /// Source span — skipped during serde serialization (not meaningful at runtime).
    #[cfg_attr(feature = "serde", serde(skip))]
    pub span: Span,
}

/// Maximum total push-constant bytes allowed by Vulkan's guaranteed minimum
/// (`minPushConstantsSize` = 128 bytes per Vulkan spec).
pub const MAX_PUSH_CONSTANT_BYTES: u32 = 128;

/// The complete parameter binding plan for one kernel.
///
/// Computed by `compute_binding_plan` during HIR lowering.
/// Consumed by codegen to emit SPIR-V descriptors and push-constant blocks.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamBindingPlan {
    /// SSBO bindings, ordered by `buffer_position`.
    pub buffers: Vec<BufferBindingSlot>,
    /// Push-constant scalar members, ordered by `member_index`.
    pub scalars: Vec<ScalarPushConstantSlot>,
    /// Total size of the push-constant block in bytes.
    pub push_constant_total_bytes: u32,
}

/// Error variants for binding plan construction.
#[derive(Debug, Clone, thiserror::Error, miette::Diagnostic)]
pub enum BindingPlanError {
    #[error("push constant block exceeds {limit} bytes (Vulkan minPushConstantsSize): parameter `{overflowing_param_name}` would push total to {got} bytes")]
    PushConstantTooLarge {
        got: u32,
        limit: u32,
        overflowing_param_name: String,
        /// Span of the overflowing parameter declaration (NOT param[0]).
        #[label("this parameter causes the overflow")]
        span: Span,
        /// Span of the kernel declaration.
        #[label("kernel declared here")]
        kernel_span: Span,
    },

    #[error("unsupported parameter type `{ty_name}` for kernel parameter `{param_name}` (M1.2 supports: scalar scalars, buffer[T], readonly_buffer[T], writeonly_buffer[T])")]
    UnsupportedParamType {
        ty_name: String,
        param_name: String,
        #[label("here")]
        span: Span,
    },
}

/// Compute the `ParamBindingPlan` from a list of kernel parameters.
///
/// Walk `params` in order:
/// - Buffer params get consecutive `buffer_position` values (0, 1, 2, …).
/// - Scalar params get consecutive `member_index` values (0, 1, 2, …).
/// - Push-constant offsets use std430 layout (aligned to `max(4, elem_size)`).
///
/// Returns `Err(BindingPlanError::PushConstantTooLarge)` if the scalar total
/// exceeds `MAX_PUSH_CONSTANT_BYTES`. The `span` in the error points at the
/// OVERFLOWING parameter, not at param[0].
pub fn compute_binding_plan(
    params: &[KernelParam],
    kernel_span: Span,
) -> Result<ParamBindingPlan, BindingPlanError> {
    let mut buffers: Vec<BufferBindingSlot> = Vec::new();
    let mut scalars: Vec<ScalarPushConstantSlot> = Vec::new();
    let mut next_buffer_position: u32 = 0;
    let mut next_member_index: u32 = 0;
    let mut current_offset: u32 = 0;

    for param in params {
        match &param.ty {
            Ty::Buffer(bt) => {
                buffers.push(BufferBindingSlot {
                    name: param.name.clone(),
                    ty: *bt,
                    position: param.position,
                    buffer_position: next_buffer_position,
                    span: param.span,
                });
                next_buffer_position += 1;
            }
            Ty::Scalar(st) => {
                // std430 alignment: align to the element's byte size (minimum 4).
                let elem_size: u32 = st.bit_width().saturating_add(7) / 8;
                let alignment: u32 = elem_size.max(4);
                // Round current_offset up to alignment boundary.
                let aligned_offset: u32 = (current_offset + alignment - 1) & !(alignment - 1);
                let new_total: u32 = aligned_offset + elem_size;

                if new_total > MAX_PUSH_CONSTANT_BYTES {
                    return Err(BindingPlanError::PushConstantTooLarge {
                        got: new_total,
                        limit: MAX_PUSH_CONSTANT_BYTES,
                        overflowing_param_name: param.name.clone(),
                        span: param.span,
                        kernel_span,
                    });
                }

                scalars.push(ScalarPushConstantSlot {
                    name: param.name.clone(),
                    ty: *st,
                    offset: aligned_offset,
                    member_index: next_member_index,
                    position: param.position,
                    span: param.span,
                });
                next_member_index += 1;
                current_offset = new_total;
            }
        }
    }

    Ok(ParamBindingPlan {
        buffers,
        scalars,
        push_constant_total_bytes: current_offset,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::BufferAccess;
    use axc_lexer::Span;

    /// AT-508: ParamBindingPlan serde roundtrip preserves all bindings.
    ///
    /// Constructs a saxpy-shaped plan (scalars: n:u32 + alpha:f32; buffers: x:f32_ro + y:f32_rw),
    /// serializes to JSON, deserializes back, and asserts equality via PartialEq.
    /// Span fields are skipped (deserialized as Span::default()) per the serde skip rule.
    #[cfg(feature = "serde")]
    #[test]
    fn at_508_param_binding_plan_serde_roundtrip_preserves_bindings() {
        use crate::ty::ScalarTy;
        use crate::buffer::{BufferAccess, BufferTy};

        let plan: ParamBindingPlan = ParamBindingPlan {
            buffers: vec![
                BufferBindingSlot {
                    name: "x".to_owned(),
                    ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadOnly },
                    position: 2,
                    buffer_position: 0,
                    span: Span::default(),
                },
                BufferBindingSlot {
                    name: "y".to_owned(),
                    ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadWrite },
                    position: 3,
                    buffer_position: 1,
                    span: Span::default(),
                },
            ],
            scalars: vec![
                ScalarPushConstantSlot {
                    name: "n".to_owned(),
                    ty: ScalarTy::U32,
                    offset: 0,
                    member_index: 0,
                    position: 0,
                    span: Span::default(),
                },
                ScalarPushConstantSlot {
                    name: "alpha".to_owned(),
                    ty: ScalarTy::F32,
                    offset: 4,
                    member_index: 1,
                    position: 1,
                    span: Span::default(),
                },
            ],
            push_constant_total_bytes: 8,
        };

        let json: String = serde_json::to_string(&plan).expect("serialize should succeed");
        let roundtripped: ParamBindingPlan =
            serde_json::from_str(&json).expect("deserialize should succeed");

        // PartialEq compares all fields; Span fields will be Span::default() after roundtrip.
        assert_eq!(plan, roundtripped, "serde roundtrip must preserve all binding plan fields");
        assert_eq!(roundtripped.buffers[0].name, "x");
        assert_eq!(roundtripped.buffers[1].buffer_position, 1);
        assert_eq!(roundtripped.scalars[0].offset, 0);
        assert_eq!(roundtripped.scalars[1].offset, 4);
        assert_eq!(roundtripped.push_constant_total_bytes, 8);
    }

    fn dummy_span() -> Span {
        Span::new(0, 1)
    }

    fn scalar_param(name: &str, ty: ScalarTy, position: u32) -> KernelParam {
        KernelParam {
            name: name.to_owned(),
            ty: Ty::Scalar(ty),
            position,
            span: dummy_span(),
        }
    }

    fn buffer_param(name: &str, elem: ScalarTy, access: BufferAccess, position: u32) -> KernelParam {
        use crate::buffer::BufferTy;
        KernelParam {
            name: name.to_owned(),
            ty: Ty::Buffer(BufferTy { elem, access }),
            position,
            span: dummy_span(),
        }
    }

    #[test]
    fn empty_params_gives_empty_plan() {
        let plan: ParamBindingPlan = compute_binding_plan(&[], dummy_span()).expect("should succeed");
        assert!(plan.buffers.is_empty());
        assert!(plan.scalars.is_empty());
        assert_eq!(plan.push_constant_total_bytes, 0);
    }

    #[test]
    fn scalar_f32_offset_zero() {
        let params: Vec<KernelParam> = vec![scalar_param("a", ScalarTy::F32, 0)];
        let plan: ParamBindingPlan = compute_binding_plan(&params, dummy_span()).expect("should succeed");
        assert_eq!(plan.scalars.len(), 1);
        assert_eq!(plan.scalars[0].offset, 0);
        assert_eq!(plan.scalars[0].member_index, 0);
        assert_eq!(plan.push_constant_total_bytes, 4);
    }

    #[test]
    fn two_scalars_sequential_offsets() {
        let params: Vec<KernelParam> = vec![
            scalar_param("a", ScalarTy::F32, 0),
            scalar_param("b", ScalarTy::F32, 1),
        ];
        let plan: ParamBindingPlan = compute_binding_plan(&params, dummy_span()).expect("should succeed");
        assert_eq!(plan.scalars[0].offset, 0);
        assert_eq!(plan.scalars[0].member_index, 0);
        assert_eq!(plan.scalars[1].offset, 4);
        assert_eq!(plan.scalars[1].member_index, 1);
        assert_eq!(plan.push_constant_total_bytes, 8);
    }

    #[test]
    fn buffer_gets_binding_position() {
        let params: Vec<KernelParam> = vec![
            scalar_param("a", ScalarTy::F32, 0),
            buffer_param("x", ScalarTy::F32, BufferAccess::ReadOnly, 1),
            buffer_param("y", ScalarTy::F32, BufferAccess::ReadWrite, 2),
        ];
        let plan: ParamBindingPlan = compute_binding_plan(&params, dummy_span()).expect("should succeed");
        assert_eq!(plan.buffers.len(), 2);
        assert_eq!(plan.buffers[0].buffer_position, 0);
        assert_eq!(plan.buffers[0].name, "x");
        assert_eq!(plan.buffers[1].buffer_position, 1);
        assert_eq!(plan.buffers[1].name, "y");
        assert_eq!(plan.scalars.len(), 1);
        assert_eq!(plan.scalars[0].member_index, 0); // independent of position
    }

    #[test]
    fn member_index_independent_of_position_with_buffers() {
        // Spec §4.4.2: member_index is ONLY counted among scalars.
        // fn k(x: readonly_buffer[f32], a: f32, b: f32) — buffer at position 0
        let params: Vec<KernelParam> = vec![
            buffer_param("x", ScalarTy::F32, BufferAccess::ReadOnly, 0),
            scalar_param("a", ScalarTy::F32, 1),
            scalar_param("b", ScalarTy::F32, 2),
        ];
        let plan: ParamBindingPlan = compute_binding_plan(&params, dummy_span()).expect("should succeed");
        assert_eq!(plan.scalars[0].member_index, 0); // NOT 1 (position-independent)
        assert_eq!(plan.scalars[1].member_index, 1);
    }

    #[test]
    fn push_constant_overflow_reports_overflowing_param() {
        // 128 / 4 = 32 scalars fit; the 33rd overflows.
        let mut params: Vec<KernelParam> = (0..32)
            .map(|i: u32| scalar_param(&format!("p{i}"), ScalarTy::F32, i))
            .collect();
        params.push(scalar_param("overflow", ScalarTy::F32, 32));
        let err: BindingPlanError = compute_binding_plan(&params, dummy_span()).unwrap_err();
        match err {
            BindingPlanError::PushConstantTooLarge { overflowing_param_name, .. } => {
                assert_eq!(overflowing_param_name, "overflow");
            }
            BindingPlanError::UnsupportedParamType { .. } => {
                panic!("expected PushConstantTooLarge, got UnsupportedParamType");
            }
        }
    }

    #[test]
    fn f64_scalar_aligned_to_8() {
        // f64 (8 bytes) in push-constant: offset must be 8-aligned.
        let params: Vec<KernelParam> = vec![
            scalar_param("a", ScalarTy::F32, 0),  // offset 0, size 4
            scalar_param("b", ScalarTy::F64, 1),  // must align to 8 → offset 8
        ];
        let plan: ParamBindingPlan = compute_binding_plan(&params, dummy_span()).expect("should succeed");
        assert_eq!(plan.scalars[0].offset, 0);
        assert_eq!(plan.scalars[1].offset, 8); // padded from 4 to 8
        assert_eq!(plan.push_constant_total_bytes, 16);
    }
}
