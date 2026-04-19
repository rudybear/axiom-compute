//! AXIOM-Compute HIR (High-level IR).
//!
//! Lowers `axc_parser::ast::Module` to a validated, structured representation
//! where GPU annotations are fully parsed into typed fields (no raw strings for
//! structured data — anti-pattern #7).

pub mod hir;
pub mod lower;
pub mod validate;

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
};
pub use lower::lower_module;
pub use validate::{HirError, HirWarning, validate};
