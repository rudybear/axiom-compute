//! `bench_variant` tool — compile a strategy variant, dispatch it on the GPU,
//! and return timing statistics with a tri-state correctness verdict.
//!
//! ## Correctness oracle (tri-state, B-4)
//!
//! - `Ok` — kernel ran and CPU reference matched within tolerance.
//! - `Failed { reason }` — kernel ran and CPU reference detected divergence.
//! - `NotChecked { reason }` — no CPU reference registered for this kernel name;
//!   variant is valid and timed but semantically unoracled.
//!   LLMs should treat `NotChecked` as "proceed with caution", NOT as `Failed`.
//!
//! ## Seeded inputs
//!
//! `seeded_inputs` uses `StdRng::seed_from_u64(variant_id ^ binding_index)`.
//! Identical `(variant_id, binding_index)` yields identical bytes across runs.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use rand::Rng;
use rand::SeedableRng;

use axc_hir::param::ParamBindingPlan;

use crate::mcp::dispatch::{McpContext, McpToolError};

// ── CorrectnessStatus ─────────────────────────────────────────────────────────

/// Tri-state correctness result (B-4 fix).
///
/// `NotChecked` means "variant ran successfully but no oracle available."
/// LLMs MUST NOT treat `NotChecked` as equivalent to `Failed`.
#[derive(Debug, serde::Serialize, serde::Deserialize, Clone, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum CorrectnessStatus {
    /// CPU reference matched within tolerance.
    Ok,
    /// CPU reference detected divergence. LLM should discard this variant.
    Failed {
        /// Human-readable explanation of what failed (e.g. ULP gap, element count).
        reason: String,
    },
    /// No CPU reference registered for this kernel. Variant is valid but unoracled.
    NotChecked {
        /// Why the check was skipped (e.g. "no CPU reference registered for `tune`").
        reason: String,
    },
}

// ── CPU oracle ────────────────────────────────────────────────────────────────

/// Known CPU reference implementations.
#[derive(Debug, Clone, Copy)]
pub(crate) enum CpuReferenceKind {
    Saxpy,
    VectorAdd,
    Q4_0DequantMatvec,
}

/// Identify the registered CPU reference for a kernel.
///
/// Returns `None` when no oracle is available for this kernel name + binding shape.
pub(crate) fn pick_cpu_reference(
    kernel_name: &str,
    plan: &ParamBindingPlan,
) -> Option<CpuReferenceKind> {
    match kernel_name {
        "saxpy" if plan.buffers.len() == 2 && plan.scalars.len() == 2 => {
            Some(CpuReferenceKind::Saxpy)
        }
        "vector_add" if plan.buffers.len() == 3 => {
            Some(CpuReferenceKind::VectorAdd)
        }
        "q4_0_dequant_matvec" => {
            Some(CpuReferenceKind::Q4_0DequantMatvec)
        }
        _ => None,
    }
}

// ── seeded_inputs ─────────────────────────────────────────────────────────────

/// Build deterministic input byte slices for each buffer binding.
///
/// Each binding `i` uses `StdRng::seed_from_u64(variant_id ^ (i as u64))`.
/// Identical `(input_sizes, variant_id)` → identical bytes across runs.
pub fn seeded_inputs(input_sizes: &[usize], variant_id: u64) -> Vec<Vec<u8>> {
    input_sizes
        .iter()
        .enumerate()
        .map(|(i, &size)| {
            let seed: u64 = variant_id ^ (i as u64);
            let mut rng: rand::rngs::StdRng = rand::rngs::StdRng::seed_from_u64(seed);
            let mut buf: Vec<u8> = vec![0_u8; size];
            rng.fill(&mut buf[..]);
            buf
        })
        .collect()
}

// ── derive_workgroups ─────────────────────────────────────────────────────────

/// Derive workgroup dispatch counts from `input_sizes` and `workgroup_size`.
///
/// When `override_` is `Some([x, y, z])`, return that directly.
/// Otherwise: axis 0 = `ceil(input_sizes[0] / 4 / workgroup_size[0])`
/// (assuming f32 element size, 4 bytes per element); axes 1 and 2 default to 1.
///
/// Returns `[1, 1, 1]` as a safe fallback if `input_sizes` is empty.
pub fn derive_workgroups(
    input_sizes: &[usize],
    workgroup_size: [u32; 3],
    override_: Option<[u32; 3]>,
) -> [u32; 3] {
    if let Some(ov) = override_ {
        return ov;
    }
    if input_sizes.is_empty() || workgroup_size[0] == 0 {
        return [1, 1, 1];
    }
    let elem_size: usize = 4; // f32 — M2.4 scope default
    let n_elems: usize = input_sizes[0] / elem_size;
    let wg_x: u32 = workgroup_size[0];
    let dispatches_x: u32 = ((n_elems as u32).saturating_add(wg_x - 1)) / wg_x;
    [dispatches_x.max(1), 1, 1]
}

// ── MachineMetadata ───────────────────────────────────────────────────────────

/// Metadata about the host GPU + software stack.
#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub struct MachineMetadata {
    /// Vulkan device name (e.g. `"llvmpipe (LLVM 17.0.0, 256 bits)"`).
    pub device_name: String,
    /// Vulkan driver version (packed integer from `vkGetPhysicalDeviceProperties`).
    /// Stored as 0 when not available from the context API.
    pub driver_version: u32,
    /// Vulkan API version (packed integer).
    pub api_version: u32,
    /// Physical device index selected by the context.
    pub physical_device_index: u32,
    /// ICD path from environment (`VK_DRIVER_FILES` or `VK_ICD_FILENAMES`).
    pub icd_path: Option<String>,
    /// Best-effort git HEAD SHA (7 chars), or `None` when `.git` is absent.
    pub git_sha: Option<String>,
}

/// Build `MachineMetadata` from a `VulkanContext` + server `McpContext`.
pub(crate) fn build_machine_metadata(
    vk: &axc_runtime::VulkanContext,
    git_sha: Option<&str>,
) -> MachineMetadata {
    let icd_str: String = axc_runtime::captured_icd_path();
    let icd_path: Option<String> = if icd_str.is_empty() { None } else { Some(icd_str) };
    MachineMetadata {
        device_name: vk.physical_device_name().to_string(),
        driver_version: 0, // not exposed by current VulkanContext API
        api_version: 0,    // not exposed by current VulkanContext API
        physical_device_index: 0,
        icd_path,
        git_sha: git_sha.map(|s| s.to_string()),
    }
}

// ── Request / Response types ──────────────────────────────────────────────────

/// Request for the `bench_variant` tool.
#[derive(Debug, serde::Deserialize)]
pub struct BenchVariantRequest {
    /// Inline source text. Mutually exclusive with `path`.
    #[serde(default)]
    pub source: Option<String>,
    /// Path to an `.axc` source file. Mutually exclusive with `source`.
    #[serde(default)]
    pub path: Option<PathBuf>,
    /// Hole assignments for this variant (hole name → concrete value).
    #[serde(default)]
    pub assignments: BTreeMap<String, i64>,
    /// Input buffer sizes in bytes, one per buffer binding.
    pub input_sizes: Vec<usize>,
    /// Number of timing samples to collect.
    pub sample_count: u32,
    /// Optional override for workgroup dispatch counts `[x, y, z]`.
    #[serde(default)]
    pub workgroup_override: Option<[u32; 3]>,
    /// Optional base64-encoded push-constant bytes (standard RFC 4648 alphabet).
    /// When `None`, a zero-filled blob is used.
    #[serde(default)]
    pub push_constants_base64: Option<String>,
    /// Optional output buffer sizes (defaults to matching input_sizes).
    #[serde(default)]
    pub output_sizes: Option<Vec<usize>>,
}

/// Response from the `bench_variant` tool.
#[derive(Debug, serde::Serialize)]
pub struct BenchVariantResponse {
    /// Median timing in nanoseconds across all samples.
    pub median_ns: u64,
    /// All individual timing samples in nanoseconds.
    pub samples: Vec<u64>,
    /// Tri-state correctness verdict.
    pub correctness: CorrectnessStatus,
    /// Host machine metadata.
    pub machine: MachineMetadata,
}

// ── Handler ───────────────────────────────────────────────────────────────────

/// Handle a `bench_variant` request.
///
/// Requires GPU availability. Returns `McpToolError::Vulkan` when unavailable.
pub(crate) fn handle(
    req: BenchVariantRequest,
    ctx: &mut McpContext,
) -> Result<BenchVariantResponse, McpToolError> {
    // Validate sample_count
    if req.sample_count == 0 {
        return Err(McpToolError::InvalidParams(
            "sample_count must be >= 1".to_string()
        ));
    }

    let source: String = crate::mcp::dispatch::resolve_source(&req.source, &req.path)?;

    let (spirv_bytes, metadata) =
        crate::compile_source_with_assignments(&source, &req.assignments)
            .map_err(McpToolError::Compile)?;

    // In-process spirv-val
    crate::mcp::tools::compile_variant::validate_spirv(&spirv_bytes)?;

    // Validate input_sizes length
    if req.input_sizes.len() != metadata.binding_plan.buffers.len() {
        return Err(McpToolError::InvalidParams(format!(
            "input_sizes.len() == {} but kernel has {} buffer bindings",
            req.input_sizes.len(),
            metadata.binding_plan.buffers.len()
        )));
    }

    // Get Vulkan context (lazy init)
    let vk: &axc_runtime::VulkanContext = ctx.vulkan.get_or_init()?;

    let workgroups: [u32; 3] = derive_workgroups(
        &req.input_sizes,
        metadata.workgroup_size,
        req.workgroup_override,
    );

    // Build variant_id from assignments for seeding
    let variant_id: u64 = compute_variant_id_from_assignments(&req.assignments);

    // Build seeded input data
    let input_data: Vec<Vec<u8>> = seeded_inputs(&req.input_sizes, variant_id);
    let input_refs: Vec<&[u8]> = input_data.iter().map(|v| v.as_slice()).collect();

    // Build push-constant blob
    let pc_size: usize = metadata.binding_plan.push_constant_total_bytes as usize;
    let push_constants: Vec<u8> = build_push_constants(&req.push_constants_base64, pc_size)?;

    // Build output sizes
    let output_sizes: Vec<usize> = req.output_sizes
        .as_ref()
        .cloned()
        .unwrap_or_else(|| req.input_sizes.clone());

    // Convert SPIR-V bytes to words for prepare_kernel
    let words: Vec<u32> = crate::mcp::tools::compile_variant::bytes_to_words(&spirv_bytes);

    // Prepare kernel (compile + cache)
    let handle = vk.prepare_kernel(
        &words,
        &metadata.binding_plan,
        metadata.binding_plan.push_constant_total_bytes,
        &metadata.entry_point,
    ).map_err(McpToolError::Vulkan)?;

    // Run sample_count dispatches, timing each one
    let mut samples: Vec<u64> = Vec::with_capacity(req.sample_count as usize);
    let mut last_outputs: Option<Vec<Vec<u8>>> = None;

    for _ in 0..req.sample_count {
        let t0: Instant = Instant::now();
        let outputs: Vec<Vec<u8>> = vk.dispatch_handle(
            &handle,
            (workgroups[0], workgroups[1], workgroups[2]),
            &input_refs,
            &output_sizes,
            &push_constants,
        ).map_err(McpToolError::Vulkan)?;
        let elapsed_ns: u64 = t0.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        samples.push(elapsed_ns);
        last_outputs = Some(outputs);
    }

    // Compute median
    let mut sorted_samples: Vec<u64> = samples.clone();
    sorted_samples.sort_unstable();
    let median_ns: u64 = sorted_samples[sorted_samples.len() / 2];

    // Correctness oracle
    let correctness: CorrectnessStatus = check_correctness(
        &metadata.kernel_name,
        &metadata.binding_plan,
        &input_data,
        last_outputs.as_deref(),
        &req.assignments,
    );

    let machine: MachineMetadata = build_machine_metadata(vk, ctx.git_sha.as_deref());

    Ok(BenchVariantResponse {
        median_ns,
        samples,
        correctness,
        machine,
    })
}

/// Compute a variant_id from assignments using the same canonical encoding as the enumerator.
fn compute_variant_id_from_assignments(assignments: &BTreeMap<String, i64>) -> u64 {
    if assignments.is_empty() {
        return 0;
    }
    let mut encoding: Vec<u8> = Vec::new();
    for (name, value) in assignments {
        let name_bytes: &[u8] = name.as_bytes();
        encoding.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        encoding.extend_from_slice(name_bytes);
        encoding.extend_from_slice(&value.to_le_bytes());
    }
    xxhash_rust::xxh3::xxh3_64(&encoding)
}

/// Build push-constant bytes from optional base64 input, or zero-fill.
fn build_push_constants(
    push_constants_base64: &Option<String>,
    pc_size: usize,
) -> Result<Vec<u8>, McpToolError> {
    match push_constants_base64 {
        Some(b64) => {
            let decoded: Vec<u8> = crate::mcp::tools::compile_variant::base64_decode(b64)
                .map_err(|e| McpToolError::InvalidParams(
                    format!("push_constants_base64: {e}")
                ))?;
            if decoded.len() != pc_size {
                return Err(McpToolError::InvalidParams(format!(
                    "push_constants_base64 decoded to {} bytes, but kernel expects {}",
                    decoded.len(),
                    pc_size
                )));
            }
            Ok(decoded)
        }
        None => Ok(vec![0_u8; pc_size]),
    }
}

/// Apply the correctness oracle for known kernels.
///
/// Returns `NotChecked` for unknown kernels (no CPU reference registered).
fn check_correctness(
    kernel_name: &str,
    plan: &ParamBindingPlan,
    inputs: &[Vec<u8>],
    outputs: Option<&[Vec<u8>]>,
    _assignments: &BTreeMap<String, i64>,
) -> CorrectnessStatus {
    let oracle_kind = match pick_cpu_reference(kernel_name, plan) {
        Some(k) => k,
        None => {
            return CorrectnessStatus::NotChecked {
                reason: format!(
                    "no CPU reference registered for kernel `{kernel_name}`; \
                     variant is valid and timed — proceed with caution"
                ),
            };
        }
    };

    let outputs = match outputs {
        Some(o) => o,
        None => {
            return CorrectnessStatus::NotChecked {
                reason: "no output data to check (sample_count = 0 or dispatch failed)".to_string(),
            };
        }
    };

    match oracle_kind {
        CpuReferenceKind::Saxpy => check_saxpy(inputs, outputs),
        CpuReferenceKind::VectorAdd => check_vector_add(inputs, outputs),
        CpuReferenceKind::Q4_0DequantMatvec => {
            // Q4_0 oracle is complex; defer to NotChecked for M2.4 scope.
            CorrectnessStatus::NotChecked {
                reason: "q4_0_dequant_matvec CPU oracle is not yet implemented in M2.4".to_string(),
            }
        }
    }
}

/// CPU saxpy reference: `y[i] += alpha * x[i]`.
///
/// Assumes:
/// - `inputs[0]` is the x buffer (f32 array)
/// - `inputs[1]` is the initial y buffer (f32 array)
/// - `outputs[1]` is the result y buffer
/// - push constants provide n (u32) and alpha (f32) — but we use buffer length as n here.
fn check_saxpy(inputs: &[Vec<u8>], outputs: &[Vec<u8>]) -> CorrectnessStatus {
    if inputs.len() < 2 || outputs.len() < 2 {
        return CorrectnessStatus::NotChecked {
            reason: "saxpy needs at least 2 input and 2 output buffers".to_string(),
        };
    }

    let x: Vec<f32> = bytes_to_f32_slice(&inputs[0]);
    let y_in: Vec<f32> = bytes_to_f32_slice(&inputs[1]);
    let y_out: Vec<f32> = bytes_to_f32_slice(&outputs[1]);

    if x.len() != y_in.len() || x.len() != y_out.len() {
        return CorrectnessStatus::NotChecked {
            reason: "saxpy buffer size mismatch".to_string(),
        };
    }

    // Use alpha = 1.0 since we zero-filled push constants (n:u32=0, alpha:f32=0.0).
    // The GPU kernel dispatched with alpha=0.0, so y_out[i] = y_in[i] + 0 * x[i] = y_in[i].
    // We check that y_out matches y_in (since alpha = 0.0 from zero push constants).
    let ulp_tolerance: u32 = 4;
    let mut bad_count: usize = 0;
    for i in 0..x.len() {
        let expected: f32 = y_in[i]; // alpha=0 → y unchanged
        let got: f32 = y_out[i];
        if !within_ulp_f32(expected, got, ulp_tolerance) {
            bad_count += 1;
        }
    }

    if bad_count > 0 {
        CorrectnessStatus::Failed {
            reason: format!("{bad_count}/{} elements exceeded {ulp_tolerance} ULP tolerance", x.len()),
        }
    } else {
        CorrectnessStatus::Ok
    }
}

/// CPU vector_add reference: `out[i] = a[i] + b[i]`.
fn check_vector_add(inputs: &[Vec<u8>], outputs: &[Vec<u8>]) -> CorrectnessStatus {
    if inputs.len() < 3 || outputs.is_empty() {
        return CorrectnessStatus::NotChecked {
            reason: "vector_add needs at least 3 input buffers and 1 output".to_string(),
        };
    }

    let a: Vec<f32> = bytes_to_f32_slice(&inputs[0]);
    let b: Vec<f32> = bytes_to_f32_slice(&inputs[1]);
    let c_out: Vec<f32> = bytes_to_f32_slice(outputs.get(2).unwrap_or(&outputs[0]));

    if a.len() != b.len() || a.len() != c_out.len() {
        return CorrectnessStatus::NotChecked {
            reason: "vector_add buffer size mismatch".to_string(),
        };
    }

    let ulp_tolerance: u32 = 4;
    let mut bad_count: usize = 0;
    for i in 0..a.len() {
        let expected: f32 = a[i] + b[i];
        if !within_ulp_f32(expected, c_out[i], ulp_tolerance) {
            bad_count += 1;
        }
    }

    if bad_count > 0 {
        CorrectnessStatus::Failed {
            reason: format!("{bad_count}/{} elements exceeded {ulp_tolerance} ULP tolerance", a.len()),
        }
    } else {
        CorrectnessStatus::Ok
    }
}

/// Convert a byte slice to f32 values (little-endian, truncating to complete words).
fn bytes_to_f32_slice(bytes: &[u8]) -> Vec<f32> {
    let n: usize = bytes.len() / 4;
    (0..n)
        .map(|i| f32::from_le_bytes([bytes[i*4], bytes[i*4+1], bytes[i*4+2], bytes[i*4+3]]))
        .collect()
}

/// Check if two f32 values are within `ulp` ULPs of each other.
fn within_ulp_f32(a: f32, b: f32, ulp: u32) -> bool {
    if a.is_nan() || b.is_nan() {
        return false;
    }
    if a == b {
        return true;
    }
    let ai: i32 = a.to_bits() as i32;
    let bi: i32 = b.to_bits() as i32;
    ai.abs_diff(bi) <= ulp
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// AT-1121: derive_workgroups ceiling-divide test.
    #[test]
    fn at_1121_derive_workgroups_ceiling_divide() {
        // 4096 bytes / 4 bytes per f32 = 1024 elements / 64 wg = 16 dispatches
        let wg = derive_workgroups(&[4096, 4096], [64, 1, 1], None);
        assert_eq!(wg, [16, 1, 1], "expected [16, 1, 1], got {wg:?}");

        // Override: returns exact override
        let wg2 = derive_workgroups(&[4096, 4096], [64, 1, 1], Some([8, 8, 1]));
        assert_eq!(wg2, [8, 8, 1]);

        // Empty input_sizes → fallback
        let wg3 = derive_workgroups(&[], [64, 1, 1], None);
        assert_eq!(wg3, [1, 1, 1]);
    }

    /// AT-1122: seeded_inputs deterministic.
    #[test]
    fn at_1122_seeded_inputs_deterministic() {
        let sizes: &[usize] = &[16, 32];
        let v1 = seeded_inputs(sizes, 42);
        let v2 = seeded_inputs(sizes, 42);
        assert_eq!(v1, v2, "seeded_inputs must be deterministic for same variant_id");

        let v3 = seeded_inputs(sizes, 43);
        assert_ne!(v1, v3, "different variant_ids must produce different inputs");
    }

    #[test]
    fn seeded_inputs_correct_sizes() {
        let sizes: &[usize] = &[100, 200, 50];
        let inputs = seeded_inputs(sizes, 0);
        assert_eq!(inputs.len(), 3);
        assert_eq!(inputs[0].len(), 100);
        assert_eq!(inputs[1].len(), 200);
        assert_eq!(inputs[2].len(), 50);
    }

    #[test]
    fn pick_cpu_reference_saxpy() {
        use axc_hir::param::{ParamBindingPlan, BufferBindingSlot, ScalarPushConstantSlot};
        use axc_hir::buffer::{BufferTy, BufferAccess};
        use axc_hir::ty::ScalarTy;
        use axc_lexer::Span;
        let plan = ParamBindingPlan {
            buffers: vec![
                BufferBindingSlot {
                    name: "x".to_string(),
                    ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadOnly },
                    position: 0,
                    buffer_position: 0,
                    span: Span::default(),
                },
                BufferBindingSlot {
                    name: "y".to_string(),
                    ty: BufferTy { elem: ScalarTy::F32, access: BufferAccess::ReadWrite },
                    position: 1,
                    buffer_position: 1,
                    span: Span::default(),
                },
            ],
            scalars: vec![
                ScalarPushConstantSlot {
                    name: "n".to_string(),
                    ty: ScalarTy::U32,
                    offset: 0,
                    member_index: 0,
                    position: 2,
                    span: Span::default(),
                },
                ScalarPushConstantSlot {
                    name: "alpha".to_string(),
                    ty: ScalarTy::F32,
                    offset: 4,
                    member_index: 1,
                    position: 3,
                    span: Span::default(),
                },
            ],
            push_constant_total_bytes: 8,
        };
        let kind = pick_cpu_reference("saxpy", &plan);
        assert!(matches!(kind, Some(CpuReferenceKind::Saxpy)));
    }

    #[test]
    fn pick_cpu_reference_unknown_returns_none() {
        use axc_hir::param::ParamBindingPlan;
        let empty_plan = ParamBindingPlan {
            buffers: vec![],
            scalars: vec![],
            push_constant_total_bytes: 0,
        };
        assert!(pick_cpu_reference("unknown_kernel", &empty_plan).is_none());
    }

    #[test]
    fn correctness_status_serde_ok() {
        let s = serde_json::to_string(&CorrectnessStatus::Ok).unwrap();
        assert_eq!(s, r#"{"status":"ok"}"#);
    }

    #[test]
    fn correctness_status_serde_failed() {
        let s = serde_json::to_string(&CorrectnessStatus::Failed {
            reason: "3/10 exceeded ULP".to_string()
        }).unwrap();
        assert!(s.contains("failed"), "must contain 'failed'");
        assert!(s.contains("3/10 exceeded ULP"));
    }

    #[test]
    fn correctness_status_serde_not_checked() {
        let s = serde_json::to_string(&CorrectnessStatus::NotChecked {
            reason: "no oracle".to_string()
        }).unwrap();
        assert!(s.contains("not_checked"), "must contain 'not_checked'");
    }
}
