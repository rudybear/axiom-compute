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

// ── M3.17 (FG.4): runtime debug-check metadata ───────────────────────────────

/// Serializable mirror of `axc_hir::DebugCheckKind` — which flag word a condition
/// targets (`Pre` → word 0, `Post` → word 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DebugCheckKind {
    Pre,
    Post,
}

/// One lowered `@precondition`/`@postcondition` condition, as recorded in the
/// sidecar for host-side violation decoding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DebugConditionMeta {
    pub kind: DebugCheckKind,
    /// 0-based bit index within `kind`'s flag word.
    pub bit: u32,
    /// Human-readable predicate rendering (e.g. `"gt(n, 0)"`), used to name the
    /// violated condition in `DispatchError::DebugCheckViolation`.
    pub text: String,
    /// 1-based source line number (best-effort; computed from the annotation's
    /// byte span against the original source text at compile time).
    pub line: u32,
}

/// Runtime debug-check metadata (M3.17 FG.4, schema v4). Present in the sidecar
/// ONLY under `--debug` (`None` for a release compile — §6/§7 of the spec).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DebugChecksMeta {
    /// Descriptor binding of the injected flag SSBO (`== num_user_buffers`).
    pub flag_binding: u32,
    /// Length of the flag buffer in `u32` words (always 2: `[pre_bits, post_bits]`).
    pub flag_len_words: u32,
    /// Every lowered condition, in declaration order.
    pub conditions: Vec<DebugConditionMeta>,
}

/// Mirrors `axc_codegen::debug_checks::DEBUG_FLAG_BUFFER_NAME` (duplicated so
/// `axc-runtime` needs no dependency on `axc-codegen`).
const DEBUG_FLAG_BUFFER_NAME: &str = "__axc_debug_flags";

/// M3.17 (FG.4), reviewer nit (b): materialize the injected debug-flag descriptor
/// slot into a `ParamBindingPlan`. ANY consumer dispatching a `--debug` kernel —
/// in-process OR a sidecar-reconstructed path (e.g. an MCP tool that only reads
/// JSON) — MUST call this instead of `meta.binding_plan` directly, so the
/// DSL/pool/bind-loop (keyed on `binding_plan.buffers`) see the flag binding
/// uniformly. Returns `meta.binding_plan` unchanged when `debug_checks` is `None`.
pub fn debug_augmented_binding_plan(meta: &KernelMetadata) -> ParamBindingPlan {
    let mut plan = meta.binding_plan.clone();
    if let Some(dbg) = &meta.debug_checks {
        plan.buffers.push(axc_hir::BufferBindingSlot {
            name: DEBUG_FLAG_BUFFER_NAME.to_owned(),
            ty: axc_hir::BufferTy { elem: axc_hir::ScalarTy::U32, access: axc_hir::BufferAccess::ReadWrite },
            position: dbg.flag_binding,
            buffer_position: dbg.flag_binding,
            span: Default::default(),
        });
    }
    plan
}

/// Decode the injected debug-flag words against `conditions`.
///
/// `flag_words == [0, 0]` ⇒ `Ok(())` (no violation). Otherwise every set bit is
/// matched against `conditions` and returns a typed
/// `DispatchError::DebugCheckViolation` naming every failing condition's `text`
/// (split by `kind` — preconditions vs postconditions).
pub fn decode_debug_flags(
    flag_words: [u32; 2],
    conditions: &[DebugConditionMeta],
) -> Result<(), DispatchError> {
    if flag_words == [0, 0] {
        return Ok(());
    }
    let mut preconditions: Vec<String> = Vec::new();
    let mut postconditions: Vec<String> = Vec::new();
    for c in conditions {
        let word = match c.kind {
            DebugCheckKind::Pre => flag_words[0],
            DebugCheckKind::Post => flag_words[1],
        };
        if word & (1u32 << c.bit) != 0 {
            match c.kind {
                DebugCheckKind::Pre => preconditions.push(c.text.clone()),
                DebugCheckKind::Post => postconditions.push(c.text.clone()),
            }
        }
    }
    Err(DispatchError::DebugCheckViolation { preconditions, postconditions })
}

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
/// M3.17 bumps this from 3 to 4 to add `debug_checks: Option<DebugChecksMeta>`.
///
/// `load_kernel_metadata` accepts versions 1, 2, 3, and 4 via the
/// `SUPPORTED_SCHEMA_VERSIONS` allowed-set guard (CRITICAL-1 fix). Version 5+ is rejected.
pub const CURRENT_SCHEMA_VERSION: u32 = 4;

/// All schema versions accepted by this runtime (CRITICAL-1 back-compat guard).
///
/// - v1: pre-M3.1 (no coopmat, no shared_memory_bytes). Deserializes with both
///   `coopmat=None` and `shared_memory_bytes=0` via `#[serde(default)]`.
/// - v2: M3.1 coopmat. Deserializes with `shared_memory_bytes=0` via default.
/// - v3: M3.2 shared memory. Deserializes with `debug_checks=None` via default.
/// - v4: M3.17 debug checks. New; all fields present when `--debug` was used.
///
/// v5+ (or 0) are rejected with `MetadataSchemaMismatch`.
pub const SUPPORTED_SCHEMA_VERSIONS: [u32; 4] = [1, 2, 3, 4];

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

    /// Runtime debug-check metadata — M3.17 (schema v4).
    ///
    /// `Some` only when compiled with `--debug`. `skip_serializing_if` is REQUIRED so
    /// a release sidecar emits NO `debug_checks` key at all (not `"debug_checks":null`),
    /// keeping "release metadata unchanged from today" literally true (§7/§6 of the spec).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub debug_checks: Option<DebugChecksMeta>,
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
            debug_checks: None,
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

    /// Builder: set the optional runtime debug-check metadata (M3.17).
    ///
    /// Pass `None` for a release compile (the default) — `debug_checks` is then
    /// omitted from the serialized JSON entirely (`skip_serializing_if`).
    pub fn with_debug_checks(mut self, debug_checks: Option<DebugChecksMeta>) -> Self {
        self.debug_checks = debug_checks;
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

    /// AT-503: CURRENT_SCHEMA_VERSION equals 4 (M3.17 bump), and KernelMetadata::new sets it.
    ///
    /// Updated from "version 3" in M3.2 to "version 4" in M3.17.
    /// Non-coopmat kernels have coopmat==None; non-shared kernels have shared_memory_bytes==0;
    /// non-debug kernels have debug_checks==None.
    #[test]
    fn at_503_metadata_current_schema_is_4_and_new_sets_it() {
        assert_eq!(CURRENT_SCHEMA_VERSION, 4, "CURRENT_SCHEMA_VERSION must be 4 for M3.17");

        // AT-503a: entry_point is set to the kernel name (matches OpEntryPoint
        // emitted by axc-codegen), not a hard-coded `"main"`. See the fix for
        // VUID-VkPipelineShaderStageCreateInfo-pName-00707.
        let meta: KernelMetadata = KernelMetadata::new(
            "saxpy".to_owned(),
            [64, 1, 1],
            saxpy_plan(),
            "saxpy".to_owned(),
        );

        assert_eq!(meta.schema_version, 4);
        assert_eq!(meta.kernel_name, "saxpy");
        assert_eq!(meta.workgroup_size, [64, 1, 1]);
        assert_eq!(meta.entry_point, "saxpy");
        assert_eq!(meta.push_constant_total_bytes, 8);
        // Non-coopmat kernel has None.
        assert!(meta.coopmat.is_none(), "non-coopmat kernel must have coopmat=None");
        // Non-shared kernel has 0 shared_memory_bytes.
        assert_eq!(meta.shared_memory_bytes, 0, "non-shared kernel must have shared_memory_bytes=0");
        // Non-debug kernel has None.
        assert!(meta.debug_checks.is_none(), "non-debug kernel must have debug_checks=None");
    }

    /// AT-2868: release metadata (debug_checks=None) serializes with NO `debug_checks`
    /// key at all — not `"debug_checks":null` — asserting the `skip_serializing_if`.
    #[test]
    fn at_2868_release_metadata_omits_debug_checks_key() {
        let meta: KernelMetadata = KernelMetadata::new(
            "saxpy".to_owned(), [64, 1, 1], saxpy_plan(), "saxpy".to_owned(),
        );
        let json: String = serde_json::to_string(&meta).expect("serialize");
        assert!(!json.contains("debug_checks"), "release JSON must omit debug_checks entirely: {json}");
    }

    /// AT-2868: a `--debug` sidecar carries `debug_checks{flag_binding, conditions[]}`,
    /// `CURRENT_SCHEMA_VERSION == 4`, and round-trips through JSON.
    #[test]
    fn at_2868_debug_metadata_round_trips() {
        let dbg = DebugChecksMeta {
            flag_binding: 2,
            flag_len_words: 2,
            conditions: vec![DebugConditionMeta {
                kind: DebugCheckKind::Pre,
                bit: 0,
                text: "gt(n, 0)".to_owned(),
                line: 1,
            }],
        };
        let meta: KernelMetadata = KernelMetadata::new(
            "saxpy".to_owned(), [64, 1, 1], saxpy_plan(), "saxpy".to_owned(),
        ).with_debug_checks(Some(dbg.clone()));
        let json: String = serde_json::to_string(&meta).expect("serialize");
        assert!(json.contains("debug_checks"), "debug JSON must carry debug_checks: {json}");
        let roundtripped: KernelMetadata = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(roundtripped.debug_checks, Some(dbg));
        assert_eq!(roundtripped.schema_version, 4);
    }

    /// M3.17: `debug_augmented_binding_plan` appends the flag slot at
    /// `buffer_position == flag_binding` only when `debug_checks` is `Some`; returns
    /// the plan unchanged (buffer count-wise) otherwise.
    #[test]
    fn debug_augmented_binding_plan_appends_flag_slot_only_when_present() {
        let plan_without_debug = saxpy_plan();
        let meta_release: KernelMetadata = KernelMetadata::new(
            "saxpy".to_owned(), [64, 1, 1], plan_without_debug.clone(), "saxpy".to_owned(),
        );
        let augmented_release = debug_augmented_binding_plan(&meta_release);
        assert_eq!(augmented_release.buffers.len(), plan_without_debug.buffers.len(),
            "release metadata: augmented plan must be unchanged");

        let dbg = DebugChecksMeta {
            flag_binding: plan_without_debug.buffers.len() as u32,
            flag_len_words: 2,
            conditions: Vec::new(),
        };
        let meta_debug: KernelMetadata = KernelMetadata::new(
            "saxpy".to_owned(), [64, 1, 1], plan_without_debug.clone(), "saxpy".to_owned(),
        ).with_debug_checks(Some(dbg.clone()));
        let augmented_debug = debug_augmented_binding_plan(&meta_debug);
        assert_eq!(augmented_debug.buffers.len(), plan_without_debug.buffers.len() + 1,
            "debug metadata: augmented plan must append exactly one flag slot");
        assert_eq!(augmented_debug.buffers.last().unwrap().buffer_position, dbg.flag_binding);
    }

    /// M3.17: `decode_debug_flags` returns `Ok(())` for `[0,0]`, and a typed
    /// violation naming the exact failing condition text for a nonzero word.
    #[test]
    fn decode_debug_flags_ok_and_violation() {
        let conditions = vec![
            DebugConditionMeta { kind: DebugCheckKind::Pre, bit: 0, text: "gt(n, 0)".to_owned(), line: 1 },
            DebugConditionMeta { kind: DebugCheckKind::Post, bit: 0, text: "is_finite(elem(y))".to_owned(), line: 2 },
        ];
        assert!(decode_debug_flags([0, 0], &conditions).is_ok());

        let err = decode_debug_flags([1, 0], &conditions).unwrap_err();
        match err {
            DispatchError::DebugCheckViolation { preconditions, postconditions } => {
                assert_eq!(preconditions, vec!["gt(n, 0)".to_owned()]);
                assert!(postconditions.is_empty());
            }
            other => panic!("expected DebugCheckViolation, got {other:?}"),
        }
    }

    /// AT-1553 (restructured for M3.17): Metadata schema back-compat.
    ///
    /// Tests that:
    /// (a) v1 sidecar (no coopmat/shared/debug_checks fields) deserializes with
    ///     coopmat=None, shared=0, debug_checks=None.
    /// (b) v2 sidecar (hand-written literal with schema_version=2, coopmat field) loads correctly.
    /// (c) v3 sidecar with shared_memory_bytes ACCEPTED, debug_checks defaults to None.
    /// (d) v4 sidecar (with debug_checks) ACCEPTED (M3.17: v4 is now CURRENT).
    /// (e) v5 sidecar REJECTED with MetadataSchemaMismatch { supported: 4 }.
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
        assert!(v3_meta.debug_checks.is_none(), "v3 sidecar without debug_checks field must default to None");

        // (d) V4 sidecar (with debug_checks) — ACCEPTED (M3.17: v4 is now CURRENT).
        let v4_json = r#"{
            "schema_version": 4,
            "kernel_name": "precondition_saxpy",
            "workgroup_size": [64, 1, 1],
            "binding_plan": {"buffers": [], "scalars": [], "push_constant_total_bytes": 0},
            "push_constant_total_bytes": 0,
            "entry_point": "precondition_saxpy",
            "debug_checks": {
                "flag_binding": 2,
                "flag_len_words": 2,
                "conditions": [
                    {"kind": "Pre", "bit": 0, "text": "gt(n, 0)", "line": 3}
                ]
            }
        }"#;
        let v4_meta: KernelMetadata = serde_json::from_str(v4_json)
            .expect("v4 sidecar must deserialize");
        assert_eq!(v4_meta.schema_version, 4);
        assert!(v4_meta.debug_checks.is_some(), "v4 sidecar must carry debug_checks");
        assert_eq!(v4_meta.debug_checks.as_ref().unwrap().flag_binding, 2);

        // (e) V5 sidecar — REJECTED (supported: 4 is the new CURRENT).
        let v5_json = r#"{
            "schema_version": 5,
            "kernel_name": "test",
            "workgroup_size": [1, 1, 1],
            "binding_plan": {"buffers": [], "scalars": [], "push_constant_total_bytes": 0},
            "push_constant_total_bytes": 0,
            "entry_point": "test"
        }"#;
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), v5_json).unwrap();
        let result = load_kernel_metadata(tmp.path());
        assert!(
            matches!(result, Err(DispatchError::MetadataSchemaMismatch { got: 5, supported: 4 })),
            "version 5 must be rejected with supported: 4; got: {result:?}"
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
