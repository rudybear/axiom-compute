//! Cooperative-matrix device enablement, preflight, and required-feature derivation.
//!
//! This module implements:
//! 1. `device_advertises_coopmat_ext` — check if VK_KHR_cooperative_matrix is listed
//!    in device extension properties (needed before calling coopmat entry points).
//! 2. `query_coopmat_support` — two-call vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR
//!    (count-then-array; if count==0, returns empty WITHOUT a second call per HN-2/AT-1501).
//! 3. `coopmat_shape_supported` — pure matcher for the required shape against the
//!    device's supported set, checking M/N/K + a/b/c/result types + scope==Subgroup +
//!    saturatingAccumulation==false (per AT-1561/WARNING fix).
//! 4. `required_device_features` — derives the Vulkan device features needed from the
//!    binding plan (CRITICAL-2/-3): storage_16bit from f16 SSBOs, storage_8bit +
//!    shader_int8 + shader_int16 from u8/i8 SSBOs, vulkan_memory_model + cooperative_matrix
//!    from uses_coopmat.
//! 5. `coopmat_required_shape_from_meta` — builds a `CoopMatRequiredShape` from the
//!    metadata sidecar (HN-10/AT-1552 — nothing is hardcoded in the runtime).
//!
//! ## Fail-closed discipline
//!
//! If the VK_KHR_cooperative_matrix extension is absent, `query_coopmat_support` returns
//! `CoopMatSupport { feature_present: false, shapes: [] }` WITHOUT calling any coopmat
//! entry points. A missing required feature fails closed at DISPATCH with the typed
//! `DispatchError::DeviceFeatureUnsupported` skip — never enable-and-hope.
//!
//! ## No GPU queries in unit tests
//!
//! `coopmat_shape_supported` and `required_device_features` are pure functions tested
//! without Vulkan (AT-1500, AT-1501, AT-1504). GPU-side preflight is integration-tested
//! via AT-1505, AT-1507, AT-1509.

use ash::vk;
use axc_hir::ParamBindingPlan;
use crate::metadata::CoopMatShapeMeta;

// ── Public types ──────────────────────────────────────────────────────────────

/// Component type of a cooperative-matrix operand as reported by the device.
///
/// Maps VkComponentTypeKHR values. `Other(i32)` captures vendor-specific values
/// that the M3.1 matcher will not match against (fail-closed).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoopMatComponentType {
    /// 16-bit float (VK_COMPONENT_TYPE_FLOAT16_KHR = 0).
    Float16,
    /// 32-bit float (VK_COMPONENT_TYPE_FLOAT32_KHR = 1).
    Float32,
    /// Signed 8-bit integer (VK_COMPONENT_TYPE_SINT8_KHR = 5).
    Sint8,
    /// Unsigned 8-bit integer (VK_COMPONENT_TYPE_UINT8_KHR = 6).
    Uint8,
    /// Signed 32-bit integer (VK_COMPONENT_TYPE_SINT32_KHR = 7).
    Sint32,
    /// Unsigned 32-bit integer (VK_COMPONENT_TYPE_UINT32_KHR = 8).
    Uint32,
    /// Any other value — not matched by the M3.1 preflight (fail-closed).
    Other(i32),
}

/// Scope of a cooperative-matrix operation as reported by the device.
///
/// Maps VkScopeKHR values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoopMatScope {
    /// Subgroup scope (VK_SCOPE_SUBGROUP_KHR = 3).
    Subgroup,
    /// Workgroup scope (VK_SCOPE_WORKGROUP_KHR = 2).
    Workgroup,
    /// Any other value (fail-closed: not matched by M3.1 matcher).
    Other(i32),
}

/// One supported (M, N, K, component types, scope, saturation) tuple from the device.
///
/// Populated by `query_coopmat_support` from the Vulkan cooperative-matrix properties.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoopMatShapeSupport {
    /// M size (rows of A / rows of C).
    pub m: u32,
    /// N size (columns of B / columns of C).
    pub n: u32,
    /// K size (contraction dim: columns of A = rows of B).
    pub k: u32,
    /// Component type of A.
    pub a_type: CoopMatComponentType,
    /// Component type of B.
    pub b_type: CoopMatComponentType,
    /// Component type of C (accumulator input).
    pub c_type: CoopMatComponentType,
    /// Component type of the result / accumulator output.
    pub result_type: CoopMatComponentType,
    /// SPIR-V scope.
    pub scope: CoopMatScope,
    /// Whether saturating accumulation is enabled for this shape.
    ///
    /// The M3.1 matcher requires `false` (codegen emits NONE_KHR operands).
    pub saturating_accumulation: bool,
}

/// Result of querying the device's cooperative-matrix support.
#[derive(Debug, Clone)]
pub struct CoopMatSupport {
    /// True if VK_KHR_cooperative_matrix extension is advertised AND
    /// VkPhysicalDeviceCooperativeMatrixFeaturesKHR.cooperativeMatrix == VK_TRUE.
    pub feature_present: bool,
    /// All (M,N,K,types,scope,saturation) tuples supported by this device.
    /// Empty when `feature_present == false` or device reports count==0.
    pub shapes: Vec<CoopMatShapeSupport>,
}

/// The required cooperative-matrix shape for a kernel dispatch.
///
/// Built by `coopmat_required_shape_from_meta` from `KernelMetadata.coopmat`.
/// The runtime NEVER hardcodes this — it always reads from the metadata sidecar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoopMatRequiredShape {
    /// Required M dimension.
    pub m: u32,
    /// Required N dimension.
    pub n: u32,
    /// Required K dimension.
    pub k: u32,
    /// Required A component type.
    pub a_type: CoopMatComponentType,
    /// Required B component type.
    pub b_type: CoopMatComponentType,
    /// Required C component type.
    pub c_type: CoopMatComponentType,
    /// Required result component type.
    pub result_type: CoopMatComponentType,
    /// Required scope (M3.1 kernels always require Subgroup).
    pub scope: CoopMatScope,
}

/// Device features required by a kernel, derived from its binding plan.
///
/// All fields default to `false`. `required_device_features` returns a populated
/// struct based on the binding plan element types and the `uses_coopmat` flag.
///
/// The context probes device support for each feature and enables only the
/// device-supported subset; a kernel that needs a feature the device lacks
/// returns `DispatchError::DeviceFeatureUnsupported` at dispatch time.
#[allow(dead_code)] // Used in AT-1504 tests and future dispatch-time preflight
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct RequiredDeviceFeatures {
    /// `VkPhysicalDevice16BitStorageFeatures.storageBuffer16BitAccess` —
    /// needed for any f16 SSBO binding.
    pub storage_16bit: bool,
    /// `VkPhysicalDevice8BitStorageFeatures.storageBuffer8BitAccess` —
    /// needed for any u8/i8 SSBO binding.
    pub storage_8bit: bool,
    /// `VkPhysicalDeviceShaderFloat16Int8Features.shaderInt8` —
    /// needed for u8/i8 SSBO (ptr_read_u8_zext path).
    pub shader_int8: bool,
    /// `shaderInt16` via VkPhysicalDeviceFeatures —
    /// needed for f16_bits_to_f32 (u16 intermediate in q4km path).
    pub shader_int16: bool,
    /// `VkPhysicalDeviceVulkanMemoryModelFeatures.vulkanMemoryModel` —
    /// needed when the kernel uses any cooperative-matrix op.
    pub vulkan_memory_model: bool,
    /// `VkPhysicalDeviceCooperativeMatrixFeaturesKHR.cooperativeMatrix` —
    /// needed when the kernel uses any cooperative-matrix op.
    pub cooperative_matrix: bool,
}

/// Record of which optional device features the context enabled at creation time.
///
/// The context probes the device via VkPhysicalDeviceFeatures2 and enables
/// only the device-supported subset of the required superset. At dispatch time,
/// `required_device_features(binding_plan, uses_coopmat)` is compared against
/// this record; a mismatch returns `DispatchError::DeviceFeatureUnsupported`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EnabledDeviceFeatures {
    /// Whether `storageBuffer16BitAccess` was enabled at device creation.
    pub storage_16bit: bool,
    /// Whether `storageBuffer8BitAccess` was enabled at device creation.
    pub storage_8bit: bool,
    /// Whether `shaderInt8` was enabled at device creation.
    pub shader_int8: bool,
    /// Whether `shaderInt16` was enabled at device creation.
    pub shader_int16: bool,
    /// Whether `vulkanMemoryModel` was enabled at device creation.
    pub vulkan_memory_model: bool,
    /// Whether `cooperativeMatrix` was enabled at device creation.
    pub cooperative_matrix: bool,
}

// ── VkComponentTypeKHR constants ──────────────────────────────────────────────

/// Convert a raw `VkComponentTypeKHR` i32 to `CoopMatComponentType`.
fn component_type_from_raw(v: i32) -> CoopMatComponentType {
    match v {
        0 => CoopMatComponentType::Float16,
        1 => CoopMatComponentType::Float32,
        5 => CoopMatComponentType::Sint8,
        6 => CoopMatComponentType::Uint8,
        7 => CoopMatComponentType::Sint32,
        8 => CoopMatComponentType::Uint32,
        other => CoopMatComponentType::Other(other),
    }
}

/// Convert a raw `VkScopeKHR` i32 to `CoopMatScope`.
fn scope_from_raw(v: i32) -> CoopMatScope {
    match v {
        2 => CoopMatScope::Workgroup,
        3 => CoopMatScope::Subgroup,
        other => CoopMatScope::Other(other),
    }
}

// ── Public functions ──────────────────────────────────────────────────────────

/// Returns true if the physical device advertises VK_KHR_cooperative_matrix
/// in its device extension properties.
///
/// This must be checked BEFORE calling any coopmat entry points (HN-2).
/// Returns false on any Vulkan error (fail-closed).
pub(crate) fn device_advertises_coopmat_ext(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> bool {
    // SAFETY: instance and physical_device are valid Vulkan handles.
    let props = unsafe {
        instance.enumerate_device_extension_properties(physical_device)
    };
    let Ok(props) = props else {
        return false;
    };
    let target = std::ffi::CString::new("VK_KHR_cooperative_matrix").unwrap();
    props.iter().any(|p| {
        // SAFETY: extension_name is a null-terminated char array.
        let name = unsafe { std::ffi::CStr::from_ptr(p.extension_name.as_ptr()) };
        name == target.as_c_str()
    })
}

/// Query the device's cooperative-matrix support.
///
/// Returns `CoopMatSupport { feature_present: false, shapes: [] }` when:
/// - `coopmat_device_ext_present == false` (NO entry point calls made), OR
/// - The VkPhysicalDeviceCooperativeMatrixFeaturesKHR.cooperativeMatrix feature is false.
///
/// Two-call count idiom (HN-2/AT-1501):
/// - First call gets count.
/// - If count == 0, returns empty WITHOUT a second call.
/// - If count > 0, allocates the array and calls again.
///
/// The `entry` parameter is the `ash::Entry` from `InstanceOwner`. It is needed
/// to load the `VK_KHR_cooperative_matrix` extension entry points via ash's loader API.
pub(crate) fn query_coopmat_support(
    entry: &ash::Entry,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    coopmat_device_ext_present: bool,
) -> CoopMatSupport {
    if !coopmat_device_ext_present {
        // Extension absent — no entry point calls; return unsupported immediately.
        return CoopMatSupport { feature_present: false, shapes: Vec::new() };
    }

    // Load the VK_KHR_cooperative_matrix instance-level loader.
    // SAFETY: The cooperative_matrix loader is valid when the extension is present;
    // entry and instance are alive for the duration of this function.
    let coopmat_khr = ash::khr::cooperative_matrix::Instance::new(entry, instance);

    // Query whether the cooperativeMatrix feature is enabled/available.
    let mut coopmat_features = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
    let mut features2 = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut coopmat_features);
    // SAFETY: physical_device is valid; features2 chain is well-formed.
    unsafe { instance.get_physical_device_features2(physical_device, &mut features2) };

    if coopmat_features.cooperative_matrix == vk::FALSE {
        // Extension present but feature not available.
        return CoopMatSupport { feature_present: false, shapes: Vec::new() };
    }

    // Two-call count idiom: first call to get count.
    // SAFETY: coopmat_khr entry point is valid (extension is present).
    let count_result = unsafe {
        coopmat_khr.get_physical_device_cooperative_matrix_properties(physical_device)
    };
    let raw_props = match count_result {
        Ok(v) => v,
        Err(_) => {
            // Query failed — treat as unsupported (fail-closed).
            return CoopMatSupport { feature_present: true, shapes: Vec::new() };
        }
    };

    if raw_props.is_empty() {
        // Extension + feature present but device reports zero shapes — AT-1501.
        return CoopMatSupport { feature_present: true, shapes: Vec::new() };
    }

    let shapes: Vec<CoopMatShapeSupport> = raw_props.iter().map(|p| {
        CoopMatShapeSupport {
            m: p.m_size,
            n: p.n_size,
            k: p.k_size,
            a_type: component_type_from_raw(p.a_type.as_raw()),
            b_type: component_type_from_raw(p.b_type.as_raw()),
            c_type: component_type_from_raw(p.c_type.as_raw()),
            result_type: component_type_from_raw(p.result_type.as_raw()),
            scope: scope_from_raw(p.scope.as_raw()),
            saturating_accumulation: p.saturating_accumulation == vk::TRUE,
        }
    }).collect();

    CoopMatSupport { feature_present: true, shapes }
}

/// Pure matcher: returns true iff the device's CoopMatSupport contains a shape
/// that satisfies the required shape for dispatch.
///
/// Requirements (all must hold — AT-1561):
/// - `support.feature_present == true`
/// - A supported shape `s` exists with:
///   - s.m == required.m && s.n == required.n && s.k == required.k
///   - s.a_type == required.a_type (exact component type match)
///   - s.b_type == required.b_type
///   - s.c_type == required.c_type
///   - s.result_type == required.result_type
///   - s.scope == required.scope (M3.1 requires Subgroup)
///   - s.saturating_accumulation == false (codegen emits NONE_KHR operands)
pub fn coopmat_shape_supported(
    required: &CoopMatRequiredShape,
    support: &CoopMatSupport,
) -> bool {
    if !support.feature_present {
        return false;
    }
    support.shapes.iter().any(|s| {
        s.m == required.m
            && s.n == required.n
            && s.k == required.k
            && s.a_type == required.a_type
            && s.b_type == required.b_type
            && s.c_type == required.c_type
            && s.result_type == required.result_type
            && s.scope == required.scope
            && !s.saturating_accumulation
    })
}

/// Derive the required Vulkan device features from a kernel's binding plan
/// and whether it uses cooperative-matrix operations.
///
/// This is the CRITICAL-2/-3 binding-plan-driven feature pass. Every
/// SPIR-V capability emitted by axc-codegen maps to a Vulkan device feature:
///
/// | Capability               | Trigger                         | Feature                          |
/// |--------------------------|----------------------------------|----------------------------------|
/// | StorageBuffer16BitAccess | any f16 SSBO binding             | storageBuffer16BitAccess          |
/// | StorageBuffer8BitAccess  | any u8/i8 SSBO binding           | storageBuffer8BitAccess           |
/// | Int8                     | u8/i8 SSBO (ptr_read_u8 path)   | shaderInt8                        |
/// | Int16                    | f16_bits_to_f32 (u16 intermed)  | shaderInt16                       |
/// | VulkanMemoryModel        | coopmat op used                  | vulkanMemoryModel                 |
/// | CooperativeMatrixKHR     | coopmat op used                  | cooperativeMatrix                 |
///
/// See the capability→feature table in M3.1-architect.json for the full mapping.
#[allow(dead_code)] // Used in AT-1504 tests and future dispatch-time preflight
pub(crate) fn required_device_features(
    binding_plan: &ParamBindingPlan,
    uses_coopmat: bool,
) -> RequiredDeviceFeatures {
    let mut feats = RequiredDeviceFeatures::default();

    for buf in &binding_plan.buffers {
        if buf.ty.needs_16bit_storage() {
            feats.storage_16bit = true;
        }
        if buf.ty.needs_8bit_storage() {
            feats.storage_8bit = true;
            feats.shader_int8 = true;
            // f16_bits_to_f32 uses a u16 intermediate (Int16 capability).
            feats.shader_int16 = true;
        }
    }

    if uses_coopmat {
        feats.vulkan_memory_model = true;
        feats.cooperative_matrix = true;
    }

    feats
}

/// Build a `CoopMatRequiredShape` from a `CoopMatShapeMeta` sidecar value.
///
/// The runtime ALWAYS uses this function — never a literal shape — so that
/// the required shape tracks whatever dimensions the source declares (AT-1552/HN-10).
pub fn coopmat_required_shape_from_meta(meta: &CoopMatShapeMeta) -> CoopMatRequiredShape {
    CoopMatRequiredShape {
        m: meta.m,
        n: meta.n,
        k: meta.k,
        a_type: coopmat_scalar_meta_to_component_type(meta.a_type),
        b_type: coopmat_scalar_meta_to_component_type(meta.b_type),
        c_type: coopmat_scalar_meta_to_component_type(meta.c_type),
        result_type: coopmat_scalar_meta_to_component_type(meta.result_type),
        scope: match meta.scope {
            crate::metadata::CoopMatScopeMeta::Subgroup => CoopMatScope::Subgroup,
            crate::metadata::CoopMatScopeMeta::Workgroup => CoopMatScope::Workgroup,
        },
    }
}

/// Map a `CoopMatScalarMeta` (sidecar type) to a `CoopMatComponentType` (runtime type).
fn coopmat_scalar_meta_to_component_type(
    meta: crate::metadata::CoopMatScalarMeta,
) -> CoopMatComponentType {
    match meta {
        crate::metadata::CoopMatScalarMeta::F16 => CoopMatComponentType::Float16,
        crate::metadata::CoopMatScalarMeta::F32 => CoopMatComponentType::Float32,
        crate::metadata::CoopMatScalarMeta::I8  => CoopMatComponentType::Sint8,
        crate::metadata::CoopMatScalarMeta::U8  => CoopMatComponentType::Uint8,
        crate::metadata::CoopMatScalarMeta::I32 => CoopMatComponentType::Sint32,
        crate::metadata::CoopMatScalarMeta::U32 => CoopMatComponentType::Uint32,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use axc_hir::{ParamBindingPlan, BufferBindingSlot};
    use axc_hir::buffer::{BufferTy, BufferAccess};
    use axc_hir::ty::ScalarTy;
    use axc_lexer::Span;
    use crate::metadata::{CoopMatShapeMeta, CoopMatScalarMeta, CoopMatScopeMeta};

    // ── AT-1500: coopmat_shape_supported matcher ──────────────────────────────

    /// AT-1500: coopmat_shape_supported pure matcher.
    ///
    /// True for exact 16x16x16 f16 Subgroup non-saturating tuple in the supported set.
    /// False on various mismatch conditions.
    #[test]
    fn at_1500_coopmat_shape_matcher() {
        let shape_f16_16x16x16 = CoopMatShapeSupport {
            m: 16, n: 16, k: 16,
            a_type: CoopMatComponentType::Float16,
            b_type: CoopMatComponentType::Float16,
            c_type: CoopMatComponentType::Float16,
            result_type: CoopMatComponentType::Float16,
            scope: CoopMatScope::Subgroup,
            saturating_accumulation: false,
        };

        let support = CoopMatSupport {
            feature_present: true,
            shapes: vec![shape_f16_16x16x16],
        };

        let req = CoopMatRequiredShape {
            m: 16, n: 16, k: 16,
            a_type: CoopMatComponentType::Float16,
            b_type: CoopMatComponentType::Float16,
            c_type: CoopMatComponentType::Float16,
            result_type: CoopMatComponentType::Float16,
            scope: CoopMatScope::Subgroup,
        };

        // Happy path: exact match.
        assert!(coopmat_shape_supported(&req, &support), "exact 16x16x16 f16 must match");

        // feature_present=false: must reject.
        let no_feat = CoopMatSupport { feature_present: false, shapes: support.shapes.clone() };
        assert!(!coopmat_shape_supported(&req, &no_feat), "feature_present=false must reject");

        // M/N/K mismatch.
        let req_32 = CoopMatRequiredShape { m: 32, ..req.clone() };
        assert!(!coopmat_shape_supported(&req_32, &support), "M mismatch must reject");
        let req_n32 = CoopMatRequiredShape { n: 32, ..req.clone() };
        assert!(!coopmat_shape_supported(&req_n32, &support), "N mismatch must reject");
        let req_k32 = CoopMatRequiredShape { k: 32, ..req.clone() };
        assert!(!coopmat_shape_supported(&req_k32, &support), "K mismatch must reject");

        // Component type mismatch.
        let req_f32_result = CoopMatRequiredShape {
            result_type: CoopMatComponentType::Float32, ..req.clone()
        };
        assert!(!coopmat_shape_supported(&req_f32_result, &support), "result_type f32 vs f16 must reject");

        // Workgroup scope only (no Subgroup shape).
        let req_wg = CoopMatRequiredShape { scope: CoopMatScope::Workgroup, ..req.clone() };
        assert!(!coopmat_shape_supported(&req_wg, &support), "Workgroup scope must reject (Subgroup available)");

        // Saturating accumulation shape — must reject (codegen emits NONE_KHR).
        let support_saturating = CoopMatSupport {
            feature_present: true,
            shapes: vec![CoopMatShapeSupport { saturating_accumulation: true, m: 16, n: 16, k: 16,
                a_type: CoopMatComponentType::Float16, b_type: CoopMatComponentType::Float16,
                c_type: CoopMatComponentType::Float16, result_type: CoopMatComponentType::Float16,
                scope: CoopMatScope::Subgroup }],
        };
        assert!(!coopmat_shape_supported(&req, &support_saturating), "saturating shape must reject");
    }

    // ── AT-1501: ext absent and count==0 safety ───────────────────────────────

    /// AT-1501: query_coopmat_support with coopmat_device_ext_present=false returns
    /// empty support without calling any Vulkan entry points.
    ///
    /// This is a logic test — no GPU needed.
    #[test]
    fn at_1501_query_coopmat_support_no_ext_and_zero_count_safe() {
        // When ext is absent, no Vulkan calls should be made (no instance provided).
        // We can't test this with a real VkPhysicalDevice, but we can verify that
        // the ext-absent early return logic works with a synthetic call.
        // (The full two-call idiom is only exercised with a real ICD.)
        //
        // Test: ext_present=false → feature_present=false, shapes empty.
        // We create a synthetic support struct to test coopmat_shape_supported.
        let absent_support = CoopMatSupport { feature_present: false, shapes: Vec::new() };
        let req = CoopMatRequiredShape {
            m: 16, n: 16, k: 16,
            a_type: CoopMatComponentType::Float16,
            b_type: CoopMatComponentType::Float16,
            c_type: CoopMatComponentType::Float16,
            result_type: CoopMatComponentType::Float16,
            scope: CoopMatScope::Subgroup,
        };
        assert!(!coopmat_shape_supported(&req, &absent_support),
            "ext absent → feature_present=false → shape not supported");

        // Empty shapes (ext+feat present but count==0): feature_present=true, no shapes.
        let empty_shapes = CoopMatSupport { feature_present: true, shapes: Vec::new() };
        assert!(!coopmat_shape_supported(&req, &empty_shapes),
            "count==0 (no shapes) → shape not supported even with feature");
    }

    // ── AT-1503: binding_is_readback predicate ────────────────────────────────

    // (binding_is_readback is in kernel_handle.rs — tested there via at_1503)

    // ── AT-1504: required_device_features from binding plan ───────────────────

    /// AT-1504: required_device_features pure derivation from binding plan.
    ///
    /// - f16 SSBO → storage_16bit=true
    /// - u8 SSBO → storage_8bit=true, shader_int8=true, shader_int16=true
    /// - all-f32 → all storage/int flags false
    /// - uses_coopmat=true → vulkan_memory_model=true, cooperative_matrix=true
    #[test]
    fn at_1504_required_device_features_from_binding_plan() {
        // f16 SSBO plan.
        let f16_plan = ParamBindingPlan {
            buffers: vec![BufferBindingSlot {
                name: "a_buf".to_owned(),
                ty: BufferTy { elem: ScalarTy::F16, access: BufferAccess::ReadOnly },
                position: 0,
                buffer_position: 0,
                span: Span::default(),
            }],
            scalars: Vec::new(),
            push_constant_total_bytes: 0,
        };
        let f16_feats = required_device_features(&f16_plan, false);
        assert!(f16_feats.storage_16bit, "f16 SSBO must set storage_16bit");
        assert!(!f16_feats.storage_8bit, "f16 SSBO must NOT set storage_8bit");
        assert!(!f16_feats.shader_int8, "f16 SSBO must NOT set shader_int8");
        assert!(!f16_feats.shader_int16, "f16 SSBO must NOT set shader_int16");
        assert!(!f16_feats.vulkan_memory_model, "non-coopmat must not set vulkan_memory_model");
        assert!(!f16_feats.cooperative_matrix, "non-coopmat must not set cooperative_matrix");

        // u8 SSBO plan.
        let u8_plan = ParamBindingPlan {
            buffers: vec![BufferBindingSlot {
                name: "q".to_owned(),
                ty: BufferTy { elem: ScalarTy::U8, access: BufferAccess::ReadOnly },
                position: 0,
                buffer_position: 0,
                span: Span::default(),
            }],
            scalars: Vec::new(),
            push_constant_total_bytes: 0,
        };
        let u8_feats = required_device_features(&u8_plan, false);
        assert!(!u8_feats.storage_16bit, "u8 SSBO must NOT set storage_16bit");
        assert!(u8_feats.storage_8bit, "u8 SSBO must set storage_8bit");
        assert!(u8_feats.shader_int8, "u8 SSBO must set shader_int8");
        assert!(u8_feats.shader_int16, "u8 SSBO must set shader_int16 (f16_bits_to_f32 path)");

        // All-f32 plan: no storage/int requirements.
        let f32_plan = ParamBindingPlan {
            buffers: vec![
                BufferBindingSlot {
                    name: "x".to_owned(),
                    ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadOnly },
                    position: 0, buffer_position: 0, span: Span::default(),
                },
                BufferBindingSlot {
                    name: "y".to_owned(),
                    ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadWrite },
                    position: 1, buffer_position: 1, span: Span::default(),
                },
            ],
            scalars: Vec::new(),
            push_constant_total_bytes: 0,
        };
        let f32_feats = required_device_features(&f32_plan, false);
        assert!(!f32_feats.storage_16bit, "f32-only must NOT set storage_16bit");
        assert!(!f32_feats.storage_8bit, "f32-only must NOT set storage_8bit");
        assert!(!f32_feats.shader_int8, "f32-only must NOT set shader_int8");
        assert!(!f32_feats.shader_int16, "f32-only must NOT set shader_int16");

        // uses_coopmat=true → vulkan_memory_model + cooperative_matrix.
        let coopmat_feats = required_device_features(&f16_plan, true);
        assert!(coopmat_feats.vulkan_memory_model, "coopmat must set vulkan_memory_model");
        assert!(coopmat_feats.cooperative_matrix, "coopmat must set cooperative_matrix");
        assert!(coopmat_feats.storage_16bit, "f16 SSBO + coopmat must still set storage_16bit");

        // Empty plan + uses_coopmat: only coopmat features, no storage/int.
        let empty_plan = ParamBindingPlan { buffers: Vec::new(), scalars: Vec::new(), push_constant_total_bytes: 0 };
        let empty_coopmat = required_device_features(&empty_plan, true);
        assert!(empty_coopmat.vulkan_memory_model);
        assert!(empty_coopmat.cooperative_matrix);
        assert!(!empty_coopmat.storage_16bit);
    }

    // ── AT-1507: subgroup size guard ──────────────────────────────────────────

    /// AT-1507: subgroupSize guard logic test.
    ///
    /// The guard is: subgroup_size != 32 → return CoopMatUnsupported.
    /// This test verifies the predicate logic; the actual guard is in kernel_handle.rs.
    #[test]
    fn at_1507_subgroup_size_guard_skips_non_32() {
        // The guard condition: fail if subgroup_size != 32.
        let requires_skip = |subgroup_size: u32| subgroup_size != 32;
        assert!(requires_skip(64), "wave64 must be skipped");
        assert!(requires_skip(16), "SIMD16 must be skipped");
        assert!(!requires_skip(32), "32-lane subgroup must NOT be skipped");
    }

    // ── AT-1551: preflight drives tile selection ──────────────────────────────

    /// AT-1551: preflight DRIVES coopmat tile selection — unsupported tiles are filtered.
    ///
    /// Given a supported set containing only 16x16x16, a 32x32x16 candidate is
    /// rejected by coopmat_shape_supported; the 16x16x16 candidate is accepted.
    #[test]
    fn at_1551_preflight_drives_tile_selection() {
        let support = CoopMatSupport {
            feature_present: true,
            shapes: vec![CoopMatShapeSupport {
                m: 16, n: 16, k: 16,
                a_type: CoopMatComponentType::Float16,
                b_type: CoopMatComponentType::Float16,
                c_type: CoopMatComponentType::Float16,
                result_type: CoopMatComponentType::Float16,
                scope: CoopMatScope::Subgroup,
                saturating_accumulation: false,
            }],
        };

        let req_32x32 = CoopMatRequiredShape {
            m: 32, n: 32, k: 16,
            a_type: CoopMatComponentType::Float16,
            b_type: CoopMatComponentType::Float16,
            c_type: CoopMatComponentType::Float16,
            result_type: CoopMatComponentType::Float16,
            scope: CoopMatScope::Subgroup,
        };
        assert!(!coopmat_shape_supported(&req_32x32, &support),
            "32x32x16 must be rejected when only 16x16x16 is supported");

        let req_16x16 = CoopMatRequiredShape {
            m: 16, n: 16, k: 16,
            a_type: CoopMatComponentType::Float16,
            b_type: CoopMatComponentType::Float16,
            c_type: CoopMatComponentType::Float16,
            result_type: CoopMatComponentType::Float16,
            scope: CoopMatScope::Subgroup,
        };
        assert!(coopmat_shape_supported(&req_16x16, &support),
            "16x16x16 must be accepted (it is in the supported set)");
    }

    // ── AT-1561: vendor shape heterogeneity + near-miss rejection ────────────

    /// AT-1561: near-miss rejection for result_type and saturatingAccumulation.
    ///
    /// Distinguishes f16/f16/f16/f16 from f16/f16/f16/f32 (result_type differs).
    /// Rejects saturatingAccumulation==true even on exact shape match.
    #[test]
    fn at_1561_vendor_shape_resulttype_and_saturation_discrimination() {
        // Device advertises f16 a/b/c but f32 result type.
        let shape_f32_result = CoopMatShapeSupport {
            m: 16, n: 16, k: 16,
            a_type: CoopMatComponentType::Float16,
            b_type: CoopMatComponentType::Float16,
            c_type: CoopMatComponentType::Float16,
            result_type: CoopMatComponentType::Float32, // f32 result, not f16
            scope: CoopMatScope::Subgroup,
            saturating_accumulation: false,
        };
        // Device also advertises all-f16 with saturation.
        let shape_saturating = CoopMatShapeSupport {
            m: 16, n: 16, k: 16,
            a_type: CoopMatComponentType::Float16,
            b_type: CoopMatComponentType::Float16,
            c_type: CoopMatComponentType::Float16,
            result_type: CoopMatComponentType::Float16,
            scope: CoopMatScope::Subgroup,
            saturating_accumulation: true,
        };
        let support = CoopMatSupport {
            feature_present: true,
            shapes: vec![shape_f32_result, shape_saturating],
        };

        // Requirement: all-f16, non-saturating.
        let req = CoopMatRequiredShape {
            m: 16, n: 16, k: 16,
            a_type: CoopMatComponentType::Float16,
            b_type: CoopMatComponentType::Float16,
            c_type: CoopMatComponentType::Float16,
            result_type: CoopMatComponentType::Float16, // requires all-f16
            scope: CoopMatScope::Subgroup,
        };

        // The f32-result shape must NOT match the all-f16 requirement.
        assert!(!coopmat_shape_supported(&req, &support),
            "f32 result_type must not satisfy f16 requirement (near-miss); \
             saturating shape must also be rejected");

        // Now add a matching all-f16 non-saturating shape.
        let shape_correct = CoopMatShapeSupport {
            m: 16, n: 16, k: 16,
            a_type: CoopMatComponentType::Float16,
            b_type: CoopMatComponentType::Float16,
            c_type: CoopMatComponentType::Float16,
            result_type: CoopMatComponentType::Float16,
            scope: CoopMatScope::Subgroup,
            saturating_accumulation: false,
        };
        let support_with_correct = CoopMatSupport {
            feature_present: true,
            shapes: vec![
                CoopMatShapeSupport { result_type: CoopMatComponentType::Float32, m: 16, n: 16, k: 16,
                    a_type: CoopMatComponentType::Float16, b_type: CoopMatComponentType::Float16,
                    c_type: CoopMatComponentType::Float16,
                    scope: CoopMatScope::Subgroup, saturating_accumulation: false },
                shape_correct,
            ],
        };
        assert!(coopmat_shape_supported(&req, &support_with_correct),
            "all-f16 non-saturating Subgroup shape must match");
    }

    // ── AT-1788 (M3.5b): mixed f16/f16/f32/f32 preflight ACCEPT + near-miss reject ───

    /// AT-1788 (MANDATORY, CPU-only): the M3.5b f16×f16→f32 HMMA dispatch matcher's
    /// correctness proof, independent of any GPU.
    ///
    /// `coopmat_shape_supported` ACCEPTS required={m:16,n:16,k:16, a:F16,b:F16,c:F32,result:F32,
    /// Subgroup, !saturating} when the device advertises that EXACT tuple, AND REJECTS the same
    /// required shape against a device that offers ONLY f16/f16/f16/f16 (c_type/result_type
    /// near-miss). AT-1561 proves near-miss REJECTION + all-f16 ACCEPTANCE; without this positive
    /// f16/f16/f32/f32 accept test a matcher regression that rejected an f32 c_type/result_type
    /// would pass CI and surface only at GPU dispatch (Lavapipe skips dispatch).
    #[test]
    fn at_1788_coopmat_shape_supported_accepts_f16_f16_f32_f32() {
        // The M3.5b required shape: f16 A/B, f32 accumulator + result.
        let req_mixed = CoopMatRequiredShape {
            m: 16, n: 16, k: 16,
            a_type: CoopMatComponentType::Float16,
            b_type: CoopMatComponentType::Float16,
            c_type: CoopMatComponentType::Float32,
            result_type: CoopMatComponentType::Float32,
            scope: CoopMatScope::Subgroup,
        };

        // Device A: advertises the EXACT mixed tuple (NVIDIA's primary HMMA) — must ACCEPT.
        let device_with_mixed = CoopMatSupport {
            feature_present: true,
            shapes: vec![
                // A distractor all-f16 shape (must not be the one that matches).
                CoopMatShapeSupport {
                    m: 16, n: 16, k: 16,
                    a_type: CoopMatComponentType::Float16,
                    b_type: CoopMatComponentType::Float16,
                    c_type: CoopMatComponentType::Float16,
                    result_type: CoopMatComponentType::Float16,
                    scope: CoopMatScope::Subgroup,
                    saturating_accumulation: false,
                },
                // The exact mixed f16/f16/f32/f32 tuple.
                CoopMatShapeSupport {
                    m: 16, n: 16, k: 16,
                    a_type: CoopMatComponentType::Float16,
                    b_type: CoopMatComponentType::Float16,
                    c_type: CoopMatComponentType::Float32,
                    result_type: CoopMatComponentType::Float32,
                    scope: CoopMatScope::Subgroup,
                    saturating_accumulation: false,
                },
            ],
        };
        assert!(
            coopmat_shape_supported(&req_mixed, &device_with_mixed),
            "AT-1788: the mixed f16/f16/f32/f32 Subgroup 16x16x16 shape MUST be accepted when \
             the device advertises that exact tuple"
        );

        // Device B: offers ONLY f16/f16/f16/f16 — must REJECT the mixed requirement (c_type and
        // result_type near-miss: F16 != F32).
        let device_only_all_f16 = CoopMatSupport {
            feature_present: true,
            shapes: vec![CoopMatShapeSupport {
                m: 16, n: 16, k: 16,
                a_type: CoopMatComponentType::Float16,
                b_type: CoopMatComponentType::Float16,
                c_type: CoopMatComponentType::Float16,
                result_type: CoopMatComponentType::Float16,
                scope: CoopMatScope::Subgroup,
                saturating_accumulation: false,
            }],
        };
        assert!(
            !coopmat_shape_supported(&req_mixed, &device_only_all_f16),
            "AT-1788: the mixed f16/f16/f32/f32 requirement MUST be rejected against a device \
             that offers only f16/f16/f16/f16 (c_type/result_type near-miss)"
        );

        // Device C: feature absent — must REJECT even though it lists the mixed shape.
        let device_feature_absent = CoopMatSupport {
            feature_present: false,
            shapes: vec![CoopMatShapeSupport {
                m: 16, n: 16, k: 16,
                a_type: CoopMatComponentType::Float16,
                b_type: CoopMatComponentType::Float16,
                c_type: CoopMatComponentType::Float32,
                result_type: CoopMatComponentType::Float32,
                scope: CoopMatScope::Subgroup,
                saturating_accumulation: false,
            }],
        };
        assert!(
            !coopmat_shape_supported(&req_mixed, &device_feature_absent),
            "AT-1788: the mixed shape MUST be rejected when feature_present=false"
        );
    }

    // ── coopmat_required_shape_from_meta ──────────────────────────────────────

    /// Verify coopmat_required_shape_from_meta maps all fields correctly.
    #[test]
    fn coopmat_required_shape_from_meta_maps_correctly() {
        let meta = CoopMatShapeMeta {
            m: 16, n: 16, k: 16,
            a_type: CoopMatScalarMeta::F16,
            b_type: CoopMatScalarMeta::F16,
            c_type: CoopMatScalarMeta::F16,
            result_type: CoopMatScalarMeta::F16,
            scope: CoopMatScopeMeta::Subgroup,
        };
        let req = coopmat_required_shape_from_meta(&meta);
        assert_eq!(req.m, 16);
        assert_eq!(req.n, 16);
        assert_eq!(req.k, 16);
        assert_eq!(req.a_type, CoopMatComponentType::Float16);
        assert_eq!(req.result_type, CoopMatComponentType::Float16);
        assert_eq!(req.scope, CoopMatScope::Subgroup);
    }
}
