//! AXIOM-Compute HIR (High-level IR).
//!
//! Lowers `axc_parser::ast::Module` to a validated, structured representation
//! where GPU annotations are fully parsed into typed fields (no raw strings for
//! structured data — anti-pattern #7).
//!
//! M1.1 adds scalar types, let/let-mut bindings, and a full two-pass typechecker.
//! M1.2 adds buffer types, kernel parameters, and push-constant binding plans.
//! M1.3 adds structured control flow: if/else, for-range, while, break, continue.
//! M1.4 adds subgroup operations and workgroup barrier.
//! M2.1 adds cooperative-matrix types, F16 primitive, and four coopmat builtins.
//! M2.5 adds U8 buffer element type and four Q4_0-path builtins for LLM dequant.
//! M3.2 adds shared[T,N] workgroup-local memory (FG.6).

pub mod hir;
pub mod lower;
pub mod validate;
pub mod ty;
pub mod expr;
pub mod typecheck;
pub mod buffer;
pub mod param;
pub mod loop_ctx;
pub mod control_flow;
pub mod subgroup;
pub mod coopmat;
pub mod q4_0;
/// GLSL.std.450 extended-instruction builtins (M3.2c — exp).
pub mod ext_inst;
/// Workgroup-shared array HIR types (M3.2).
pub mod shared;

pub use hir::{
    Module as HirModule,
    Kernel,
    KernelId,
    KernelAnnotations,
    WorkgroupDims,
    KernelBody,
    ComplexityForm,
    ComplexityVar,
    PORTABLE_MIN_WORKGROUP_INVOCATIONS,
    DESKTOP_MAX_WORKGROUP_INVOCATIONS,
    DebugCheck,
    DebugCheckKind,
    DebugCompareOp,
    DebugOperand,
};
pub use lower::lower_module;
pub use validate::{HirError, HirWarning, validate};
pub use ty::{ScalarTy, IntLiteralValue, FloatLiteralValue, LiteralRangeErr, fit_int_literal, fit_float_literal};
pub use expr::{
    KernelBodyTyped, HirExpr, HirExprKind, HirStmt, Binding, BindingId, BindingTy,
    BinOp as HirBinOp, UnaryOp as HirUnaryOp, ShortCircuitOp as HirShortCircuitOp,
    BitwiseOp as HirBitwiseOp,
};
pub use typecheck::{typecheck_kernel_body, TypecheckError};
pub use buffer::{BufferAccess, BufferTy};
pub use param::{
    Ty as ParamTy, KernelParam, BufferBindingSlot, ScalarPushConstantSlot,
    ParamBindingPlan, BindingPlanError, compute_binding_plan, MAX_PUSH_CONSTANT_BYTES,
};
pub use control_flow::{HirIf, HirElse, HirForRange, HirWhile, ForStep};
pub use loop_ctx::{HirLoopStack, ScopeStack};
pub use subgroup::{SubgroupOp, SubgroupReduceKind, BarrierKind};
pub use coopmat::{CoopMatUse, CoopMatBuiltin, CoopMatrixShapeKind, CoopMatKey, CoopMatrixShape};
pub use q4_0::{Q4_0Builtin, RESERVED_Q4_0_BUILTIN_NAMES, is_reserved_q4_0_builtin};
pub use ext_inst::{ExtInstBuiltin, RESERVED_EXT_INST_BUILTIN_NAMES, is_reserved_ext_inst_builtin};
pub use shared::{SharedId, SharedTy, SharedDecl, MAX_SHARED_ELEMS, PORTABLE_MIN_SHARED_BYTES, is_allowed_shared_element};
