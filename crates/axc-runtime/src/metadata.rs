//! Kernel metadata sidecar — schema v1.
//!
//! `KernelMetadata` is a JSON-serializable struct written alongside every `.spv`
//! file by `axc_driver::compile_file`. It contains everything the runtime needs
//! to dispatch the kernel without re-parsing the `.axc` source.
//!
//! The runtime reads the sidecar via `load_kernel_metadata(path)` and uses the
//! embedded `ParamBindingPlan` to build descriptor set layouts, push-constant
//! ranges, and validate `DispatchRequest` fields at pre-dispatch check time.
//!
//! ## Schema versioning
//!
//! `CURRENT_SCHEMA_VERSION = 1` for M1.5. Future-incompatible changes must bump
//! this integer. `load_kernel_metadata` rejects mismatched versions with
//! `DispatchError::MetadataSchemaMismatch { got, supported }`.
//!
//! ## Span fields
//!
//! `axc_hir::ParamBindingPlan` contains `axc_lexer::Span` fields on its slot
//! types. These fields are annotated with `#[serde(skip)]` in axc-hir so they
//! are excluded from the JSON representation. On deserialization, serde calls
//! `Span::default()` which returns `Span { start: 0, end: 0 }`. Source location
//! information is irrelevant at dispatch time.

use std::path::Path;
use serde::{Deserialize, Serialize};
use axc_hir::ParamBindingPlan;
use crate::error::DispatchError;

/// Schema version for the `.axc.meta.json` sidecar format.
///
/// Increment this constant when making a breaking change to `KernelMetadata`.
/// The runtime rejects sidecars with a different schema version via
/// `DispatchError::MetadataSchemaMismatch`.
pub const CURRENT_SCHEMA_VERSION: u32 = 1;

/// Metadata sidecar for a compiled AXIOM-Compute kernel.
///
/// Written by `axc_driver::compile_file` as `<output>.axc.meta.json` next to
/// the `.spv` file. Read by the runtime via `load_kernel_metadata`.
///
/// All fields needed to dispatch the kernel are present here, so the runtime
/// does not need to re-parse the `.axc` source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelMetadata {
    /// Schema version. Must equal `CURRENT_SCHEMA_VERSION` for this runtime.
    pub schema_version: u32,
    /// Source-level kernel name (e.g. `"saxpy"`).
    pub kernel_name: String,
    /// Workgroup dimensions from `@workgroup(X, Y, Z)`.
    pub workgroup_size: [u32; 3],
    /// Parameter binding plan: buffer bindings + scalar push-constant slots.
    pub binding_plan: ParamBindingPlan,
    /// Total push-constant block size in bytes (std430 layout).
    pub push_constant_total_bytes: u32,
    /// SPIR-V entry-point name (always `"main"` for M1.5).
    pub entry_point: String,
}

impl KernelMetadata {
    /// Construct a new `KernelMetadata` with the given fields.
    ///
    /// `schema_version` is automatically set to `CURRENT_SCHEMA_VERSION`.
    /// `push_constant_total_bytes` is derived from the `binding_plan`.
    pub fn new(
        kernel_name: String,
        workgroup_size: [u32; 3],
        binding_plan: ParamBindingPlan,
        entry_point: String,
    ) -> Self {
        let push_constant_total_bytes: u32 = binding_plan.push_constant_total_bytes;
        Self {
            schema_version: CURRENT_SCHEMA_VERSION,
            kernel_name,
            workgroup_size,
            binding_plan,
            push_constant_total_bytes,
            entry_point,
        }
    }

    /// Serialize this metadata to JSON (pretty-printed) and write it to `path`.
    pub fn save(&self, path: &Path) -> Result<(), DispatchError> {
        let json: String = serde_json::to_string_pretty(self)
            .map_err(|e| DispatchError::MetadataIoError(format!("serialize: {e}")))?;
        std::fs::write(path, json.as_bytes())
            .map_err(|e| DispatchError::MetadataIoError(e.to_string()))?;
        Ok(())
    }
}

/// Load a `KernelMetadata` sidecar from a JSON file.
///
/// Reads the file, deserializes JSON, and checks `schema_version`.
/// Returns `DispatchError::MetadataSchemaMismatch` if the version does not
/// match `CURRENT_SCHEMA_VERSION`.
pub fn load_kernel_metadata(path: &Path) -> Result<KernelMetadata, DispatchError> {
    let text: String = std::fs::read_to_string(path)
        .map_err(|e| DispatchError::MetadataIoError(e.to_string()))?;

    let meta: KernelMetadata = serde_json::from_str(&text)
        .map_err(|e| DispatchError::MetadataParseError(e.to_string()))?;

    if meta.schema_version != CURRENT_SCHEMA_VERSION {
        return Err(DispatchError::MetadataSchemaMismatch {
            got: meta.schema_version,
            supported: CURRENT_SCHEMA_VERSION,
        });
    }

    Ok(meta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axc_hir::{ParamBindingPlan, BufferBindingSlot, ScalarPushConstantSlot, BufferTy, ScalarTy};
    use axc_hir::buffer::BufferAccess;
    use axc_lexer::Span;

    /// Build a minimal saxpy-shaped `ParamBindingPlan` for tests.
    fn saxpy_plan() -> ParamBindingPlan {
        ParamBindingPlan {
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
        }
    }

    /// AT-503: CURRENT_SCHEMA_VERSION equals 1, and KernelMetadata::new sets it.
    #[test]
    fn at_503_metadata_current_schema_is_1_and_new_sets_it() {
        assert_eq!(CURRENT_SCHEMA_VERSION, 1, "CURRENT_SCHEMA_VERSION must be 1 for M1.5");

        let meta: KernelMetadata = KernelMetadata::new(
            "saxpy".to_owned(),
            [64, 1, 1],
            saxpy_plan(),
            "main".to_owned(),
        );

        assert_eq!(meta.schema_version, 1);
        assert_eq!(meta.kernel_name, "saxpy");
        assert_eq!(meta.workgroup_size, [64, 1, 1]);
        assert_eq!(meta.entry_point, "main");
        assert_eq!(meta.push_constant_total_bytes, 8);
    }

    /// AT-501: Manifest test — verify axc-runtime Cargo.toml has the expected dependencies.
    #[test]
    fn at_501_runtime_cargo_has_expected_deps() {
        let cargo_toml_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("Cargo.toml");
        let content: String = std::fs::read_to_string(&cargo_toml_path)
            .unwrap_or_else(|e| panic!("failed to read Cargo.toml: {e}"));

        assert!(content.contains("ash = { workspace = true }"),
            "Cargo.toml must contain `ash = {{ workspace = true }}`; got:\n{content}");
        assert!(content.contains("axc-hir"),
            "Cargo.toml must contain axc-hir dep; got:\n{content}");
        assert!(content.contains("features = [\"serde\"]"),
            "Cargo.toml must contain axc-hir with serde feature; got:\n{content}");
        assert!(content.contains("thiserror = { workspace = true }"),
            "Cargo.toml must contain thiserror; got:\n{content}");
        assert!(content.contains("miette = { workspace = true }"),
            "Cargo.toml must contain miette; got:\n{content}");
        assert!(content.contains("serde = { workspace = true }"),
            "Cargo.toml must contain serde; got:\n{content}");
        assert!(content.contains("serde_json = { workspace = true }"),
            "Cargo.toml must contain serde_json; got:\n{content}");
        assert!(content.contains("rspirv = { workspace = true }"),
            "Cargo.toml must contain rspirv in dev-dependencies; got:\n{content}");
        assert!(content.contains("axc-driver = { path = \"../axc-driver\" }"),
            "Cargo.toml must contain axc-driver in dev-dependencies; got:\n{content}");
    }
}
