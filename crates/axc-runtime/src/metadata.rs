//! Kernel metadata sidecar — schema v2 (M3.1).
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
//! `CURRENT_SCHEMA_VERSION = 2` for M3.1 (bumped from 1). The version check
//! in `load_kernel_metadata` accepts BOTH version 1 AND version 2:
//! - Version 1 sidecars (pre-M3.1) have no `coopmat` field; serde deserializes
//!   `coopmat = None` via `#[serde(default)]`, which is dispatch-identical to
//!   a non-coopmat kernel.
//! - Version 2 sidecars carry the optional `coopmat: Option<CoopMatShapeMeta>`
//!   field populated by the driver from the HIR `CoopMatShape`.
//!
//! Versions >= 3 (or <= 0) are rejected with `MetadataSchemaMismatch`.
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

// ── CoopMat metadata types ────────────────────────────────────────────────────

/// Serializable scalar element type for a cooperative-matrix operand.
///
/// Mirrors `axc_hir::ty::ScalarTy` but restricted to the allowed coopmat element set,
/// serialized as a string tag for human-readable JSON sidecars.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoopMatScalarMeta {
    /// 16-bit float (f16). Most common for Tensor Core workloads.
    F16,
    /// 32-bit float.
    F32,
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 32-bit integer.
    I32,
    /// Unsigned 32-bit integer.
    U32,
}

/// Serializable cooperative-matrix scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoopMatScopeMeta {
    /// Subgroup scope (SPIR-V Scope = 3). All M3.1 kernels use this.
    Subgroup,
    /// Workgroup scope. Reserved for future milestones.
    Workgroup,
}

/// Serializable cooperative-matrix shape metadata.
///
/// Stored in `KernelMetadata.coopmat` (schema v2). The runtime builds
/// `CoopMatRequiredShape` from this struct via `coopmat_required_shape_from_meta`
/// — nothing is hardcoded in the runtime.
///
/// Populated by `axc_driver::compile_source_with_meta` from the HIR's
/// `KernelAnnotations.coop_matrix` field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoopMatShapeMeta {
    /// M dimension (rows of A / rows of C).
    pub m: u32,
    /// N dimension (columns of B / columns of C).
    pub n: u32,
    /// K dimension (columns of A = rows of B; contraction dim).
    pub k: u32,
    /// Element type of the A matrix.
    pub a_type: CoopMatScalarMeta,
    /// Element type of the B matrix.
    pub b_type: CoopMatScalarMeta,
    /// Element type of the C (accumulator input) matrix.
    pub c_type: CoopMatScalarMeta,
    /// Element type of the result / accumulator output matrix.
    pub result_type: CoopMatScalarMeta,
    /// Scope used for cooperative-matrix operations.
    pub scope: CoopMatScopeMeta,
}

/// Schema version for the `.axc.meta.json` sidecar format.
///
/// M3.1 bumped this from 1 to 2 to add `coopmat: Option<CoopMatShapeMeta>`.
/// M3.2 bumps this from 2 to 3 to add `shared_memory_bytes: u32`.
///
/// `load_kernel_metadata` accepts versions 1, 2, and 3 via the `SUPPORTED_SCHEMA_VERSIONS`
/// allowed-set guard (CRITICAL-1 fix). Version 4+ is rejected.
pub const CURRENT_SCHEMA_VERSION: u32 = 3;

/// All schema versions accepted by this runtime (CRITICAL-1 back-compat guard).
///
/// - v1: pre-M3.1 (no coopmat, no shared_memory_bytes). Deserializes with both
///   `coopmat=None` and `shared_memory_bytes=0` via `#[serde(default)]`.
/// - v2: M3.1 coopmat. Deserializes with `shared_memory_bytes=0` via default.
/// - v3: M3.2 shared memory. New; all fields present.
///
/// v4+ (or 0) are rejected with `MetadataSchemaMismatch`.
pub const SUPPORTED_SCHEMA_VERSIONS: [u32; 3] = [1, 2, 3];

/// Metadata sidecar for a compiled AXIOM-Compute kernel.
///
/// Written by `axc_driver::compile_file` as `<output>.axc.meta.json` next to
/// the `.spv` file. Read by the runtime via `load_kernel_metadata`.
///
/// All fields needed to dispatch the kernel are present here, so the runtime
/// does not need to re-parse the `.axc` source.
///
/// ## Schema v2 (M3.1)
///
/// Adds `coopmat: Option<CoopMatShapeMeta>` with `#[serde(default)]` so
/// version 1 sidecars (no field) deserialize to `coopmat = None`.
///
/// ## Schema v3 (M3.2)
///
/// Adds `shared_memory_bytes: u32` with `#[serde(default)]` so v1 and v2
/// sidecars deserialize with `shared_memory_bytes = 0` (no shared memory = 0 bytes).
/// The allowed-set version guard {1, 2, 3} ensures v1 and v2 still load.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelMetadata {
    /// Schema version. Must be in `SUPPORTED_SCHEMA_VERSIONS` [1, 2, 3] for this runtime.
    pub schema_version: u32,
    /// Source-level kernel name (e.g. `"saxpy"`).
    pub kernel_name: String,
    /// Workgroup dimensions from `@workgroup(X, Y, Z)`.
    pub workgroup_size: [u32; 3],
    /// Parameter binding plan: buffer bindings + scalar push-constant slots.
    pub binding_plan: ParamBindingPlan,
    /// Total push-constant block size in bytes (std430 layout).
    pub push_constant_total_bytes: u32,
    /// SPIR-V entry-point name.
    ///
    /// This is the name written to `OpEntryPoint <ExecModel> %id "<name>"` in
    /// the compiled SPIR-V module and is the `pName` passed to
    /// `vkCreateComputePipelines`. The AXIOM-Compute codegen uses the source-
    /// level kernel name as the SPIR-V entry-point name (see
    /// `axc-codegen::emit`), so this field typically equals `kernel_name`.
    /// The runtime MUST pass this value (not a hard-coded `"main"`) to
    /// `VkPipelineShaderStageCreateInfo.pName` — otherwise Vulkan raises
    /// `VUID-VkPipelineShaderStageCreateInfo-pName-00707` ("entrypoint not
    /// found"), surfaced by Lavapipe as `ERROR_UNKNOWN`.
    pub entry_point: String,
    /// Cooperative-matrix shape metadata — M3.1 (schema v2).
    ///
    /// `Some` for kernels that use `@cooperative_matrix`; `None` for all others.
    /// On schema v1 sidecars (no field in JSON), serde deserializes this as `None`
    /// via `#[serde(default)]`, which is dispatch-identical to a non-coopmat kernel.
    /// The runtime builds `CoopMatRequiredShape` from this field via
    /// `coopmat_required_shape_from_meta` — nothing is hardcoded.
    #[serde(default)]
    pub coopmat: Option<CoopMatShapeMeta>,

    /// Total static workgroup-shared memory bytes declared in this kernel — M3.2 (schema v3).
    ///
    /// Sum of `shared_decl.ty.total_byte_size()` for all `shared[T,N]` declarations
    /// in the kernel body. `0` for kernels without shared arrays.
    ///
    /// On schema v1 and v2 sidecars (no field in JSON), serde deserializes this as `0`
    /// via `#[serde(default)]`, which is dispatch-identical to a zero-shared-memory kernel.
    ///
    /// The runtime uses this in `preflight_kernel_support` to check against the device's
    /// `maxComputeSharedMemorySize` limit (CRITICAL-4 wiring).
    #[serde(default)]
    pub shared_memory_bytes: u32,
}

impl KernelMetadata {
    /// Construct a new `KernelMetadata` with the given fields.
    ///
    /// `schema_version` is automatically set to `CURRENT_SCHEMA_VERSION`.
    /// `push_constant_total_bytes` is derived from the `binding_plan`.
    /// `coopmat` defaults to `None`; use `.with_coopmat(...)` to add it.
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
            coopmat: None,
            shared_memory_bytes: 0,
        }
    }

    /// Builder: set the optional cooperative-matrix shape metadata (M3.1).
    ///
    /// Returns `self` with `coopmat` set to the given value. Use `None` for
    /// non-coopmat kernels (the default). Existing callers pass `None` and are unaffected.
    ///
    /// ```rust,ignore
    /// let meta = KernelMetadata::new(...).with_coopmat(Some(shape_meta));
    /// ```
    pub fn with_coopmat(mut self, coopmat: Option<CoopMatShapeMeta>) -> Self {
        self.coopmat = coopmat;
        self
    }

    /// Builder: set the total static workgroup-shared memory bytes (M3.2).
    ///
    /// Populated by the driver from `Σ shared_decl.ty.total_byte_size()`.
    /// Callers with no shared arrays pass `0` (or omit this builder step).
    ///
    /// ```rust,ignore
    /// let meta = KernelMetadata::new(...).with_shared_memory_bytes(4096);
    /// ```
    pub fn with_shared_memory_bytes(mut self, bytes: u32) -> Self {
        self.shared_memory_bytes = bytes;
        self
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

    // CRITICAL-1 back-compat guard: accept an explicit allowed-set {1, 2, 3}.
    // - v1 (pre-M3.1): no coopmat field, no shared_memory_bytes. Both default to 0/None.
    // - v2 (M3.1 coopmat): has coopmat, no shared_memory_bytes. shared defaults to 0.
    // - v3 (M3.2 shared): has both coopmat and shared_memory_bytes.
    // - v4+ (or 0): rejected with MetadataSchemaMismatch.
    if !SUPPORTED_SCHEMA_VERSIONS.contains(&meta.schema_version) {
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

    /// AT-503: CURRENT_SCHEMA_VERSION equals 3 (M3.2 bump), and KernelMetadata::new sets it.
    ///
    /// Updated from "version 2" in M3.1 to "version 3" in M3.2.
    /// Non-coopmat kernels have coopmat==None; non-shared kernels have shared_memory_bytes==0.
    #[test]
    fn at_503_metadata_current_schema_is_3_and_new_sets_it() {
        assert_eq!(CURRENT_SCHEMA_VERSION, 3, "CURRENT_SCHEMA_VERSION must be 3 for M3.2");

        // AT-503a: entry_point is set to the kernel name (matches OpEntryPoint
        // emitted by axc-codegen), not a hard-coded `"main"`. See the fix for
        // VUID-VkPipelineShaderStageCreateInfo-pName-00707.
        let meta: KernelMetadata = KernelMetadata::new(
            "saxpy".to_owned(),
            [64, 1, 1],
            saxpy_plan(),
            "saxpy".to_owned(),
        );

        assert_eq!(meta.schema_version, 3);
        assert_eq!(meta.kernel_name, "saxpy");
        assert_eq!(meta.workgroup_size, [64, 1, 1]);
        assert_eq!(meta.entry_point, "saxpy");
        assert_eq!(meta.push_constant_total_bytes, 8);
        // Non-coopmat kernel has None.
        assert!(meta.coopmat.is_none(), "non-coopmat kernel must have coopmat=None");
        // Non-shared kernel has 0 shared_memory_bytes.
        assert_eq!(meta.shared_memory_bytes, 0, "non-shared kernel must have shared_memory_bytes=0");
    }

    /// AT-1553 (restructured for M3.2): Metadata schema back-compat.
    ///
    /// Tests that:
    /// (a) v1 sidecar (no coopmat/shared fields) deserializes with coopmat=None, shared=0.
    /// (b) v2 sidecar (hand-written literal with schema_version=2, coopmat field) loads correctly.
    ///     NOTE: KernelMetadata::new now stamps version 3 — we use a hand-written v2 JSON literal
    ///     to avoid testing serialization from a v2 instance (which would now produce v3).
    /// (c) v3 sidecar with shared_memory_bytes ACCEPTED (INVERTED from old reject-v3 assertion).
    /// (d) v4 sidecar REJECTED with MetadataSchemaMismatch { supported: 3 }.
    #[test]
    fn at_1553_metadata_schema_v1_v2_back_compat() {
        // (a) V1 sidecar: no coopmat field — must deserialize with coopmat=None, shared=0.
        let v1_json = r#"{
            "schema_version": 1,
            "kernel_name": "saxpy",
            "workgroup_size": [64, 1, 1],
            "binding_plan": {"buffers": [], "scalars": [], "push_constant_total_bytes": 0},
            "push_constant_total_bytes": 0,
            "entry_point": "saxpy"
        }"#;
        let v1_meta: KernelMetadata = serde_json::from_str(v1_json)
            .expect("v1 sidecar must deserialize");
        assert_eq!(v1_meta.schema_version, 1);
        assert!(v1_meta.coopmat.is_none(), "v1 sidecar must have coopmat=None");
        assert_eq!(v1_meta.shared_memory_bytes, 0, "v1 sidecar must have shared_memory_bytes=0");

        // (b) V2 sidecar — hand-written literal with schema_version=2.
        // KernelMetadata::new now stamps version 3 (M3.2 bump), so we write the v2 JSON
        // by hand and verify it round-trips coopmat + shared defaults to 0.
        let v2_json_literal = r#"{
            "schema_version": 2,
            "kernel_name": "matmul_tile",
            "workgroup_size": [32, 1, 1],
            "binding_plan": {"buffers": [], "scalars": [], "push_constant_total_bytes": 0},
            "push_constant_total_bytes": 0,
            "entry_point": "matmul_tile",
            "coopmat": {
                "m": 16, "n": 16, "k": 16,
                "a_type": "F16", "b_type": "F16", "c_type": "F16", "result_type": "F16",
                "scope": "Subgroup"
            }
        }"#;
        let v2_meta: KernelMetadata = serde_json::from_str(v2_json_literal)
            .expect("v2 sidecar must deserialize");
        assert_eq!(v2_meta.schema_version, 2);
        assert!(v2_meta.coopmat.is_some(), "v2 sidecar must have coopmat field");
        assert_eq!(v2_meta.coopmat.as_ref().unwrap().m, 16);
        assert_eq!(v2_meta.shared_memory_bytes, 0, "v2 sidecar must have shared_memory_bytes=0 (default)");

        // (c) V3 sidecar with shared_memory_bytes — now ACCEPTED (INVERTED from old reject-v3).
        let v3_json = r#"{
            "schema_version": 3,
            "kernel_name": "shared_reduce",
            "workgroup_size": [256, 1, 1],
            "binding_plan": {"buffers": [], "scalars": [], "push_constant_total_bytes": 0},
            "push_constant_total_bytes": 0,
            "entry_point": "shared_reduce",
            "shared_memory_bytes": 1024
        }"#;
        let v3_meta: KernelMetadata = serde_json::from_str(v3_json)
            .expect("v3 sidecar must deserialize");
        assert_eq!(v3_meta.schema_version, 3);
        assert_eq!(v3_meta.shared_memory_bytes, 1024, "v3 sidecar shared_memory_bytes must be 1024");
        assert!(v3_meta.coopmat.is_none(), "v3 sidecar without coopmat field must have coopmat=None");

        // (d) V4 sidecar — REJECTED (supported: 3 is the new CURRENT).
        let v4_json = r#"{
            "schema_version": 4,
            "kernel_name": "test",
            "workgroup_size": [1, 1, 1],
            "binding_plan": {"buffers": [], "scalars": [], "push_constant_total_bytes": 0},
            "push_constant_total_bytes": 0,
            "entry_point": "test"
        }"#;
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), v4_json).unwrap();
        let result = load_kernel_metadata(tmp.path());
        assert!(
            matches!(result, Err(DispatchError::MetadataSchemaMismatch { got: 4, supported: 3 })),
            "version 4 must be rejected with supported: 3; got: {result:?}"
        );
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
