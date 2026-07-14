//! M3.16 (FG.1) — `verify_rewrite`: the LLM structural-rewrite verifier core (Layer 0).
//!
//! # Thesis
//!
//! `@strategy` holes (M2.3) already cover *parameter* tuning. This module is the
//! missing *structural* half: an LLM agent may rewrite a kernel's BODY (different
//! loop nest, different memory-access pattern) and this module machine-checks that
//! the rewrite is observationally equivalent to the original — verify, don't
//! generate (§2 of the M3.16 spec). There is no LLM call anywhere in this module.
//!
//! # Pipeline (§3.1 of the spec)
//!
//! 1. Compile both sources (`compile_source_with_assignments`).
//! 2. `spirv-val` both.
//! 3. Interface gate (`diff_interface`) — CPU-only anti-cheat structural check.
//! 4. Request-completeness preflight (CPU-only) — coopmat/shared kernels require
//!    explicit `workgroups`; float-scalar kernels require explicit `push_constants`
//!    (the alpha=0 annihilation guard, CRIT-1).
//! 5. GPU availability — `vk == None` short-circuits to `SKIPPED`/`gpu_unavailable`
//!    (CPU gates 1-4 have already run, so this is a first-class typed outcome).
//! 6. Deterministic input generation — well-conditioned `[-1,1]` finite fills for
//!    float buffers, raw-byte fills (`seeded_inputs`) for integer/packed buffers.
//! 7. Determinism self-check — the ORIGINAL is dispatched TWICE; self-disagreement
//!    yields `NONDETERMINISTIC_ORACLE` (neither PASS nor FAIL for the candidate).
//! 8. Dispatch the REWRITTEN kernel once.
//! 9. Differential compare (rewritten vs the first original run) under the
//!    verifier-owned `TolerancePolicy`.
//!
//! # Reuse (§3.4 — no duplication)
//!
//! `compile_source_with_assignments`, `mcp::tools::compile_variant::{validate_spirv,
//! bytes_to_words}`, `mcp::tools::bench_variant::{seeded_inputs, derive_workgroups,
//! within_ulp_f32}`, `VulkanContext::{prepare_kernel_checked, dispatch_handle}`,
//! `KernelMetadata`/`ParamBindingPlan`. The NEW code here is orchestration, the
//! interface diff, the element-type-dispatched buffer fill, the relative-tolerance
//! comparator, and verdict serialization.

use std::collections::BTreeMap;

use rand::{RngCore, SeedableRng};

use axc_hir::param::ParamBindingPlan;
use axc_hir::ty::ScalarTy;
use axc_runtime::{DispatchError, KernelMetadata, VulkanContext};

use crate::mcp::tools::bench_variant::{derive_workgroups, seeded_inputs, within_ulp_f32};
use crate::mcp::tools::compile_variant::{bytes_to_words, validate_spirv};

// ── TolerancePolicy ───────────────────────────────────────────────────────────

/// Tolerance policy for the differential output comparison. VERIFIER-OWNED (§4.2):
/// never read from either kernel's source — the comparison bar is set by the
/// verifier / the request, never by the thing being judged.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum TolerancePolicy {
    /// `a.to_bits() == b.to_bits()` per element. The default.
    #[default]
    BitExact,
    /// `within_ulp_f32(a, b, ulp)`.
    Ulp {
        /// Maximum allowed ULP distance.
        ulp: u32,
    },
    /// `|a-b| <= eps * max(|a|,|b|, tiny)` (the DESIGN.md §510 relative-tolerance
    /// form used by the FROZEN 1e-3 Q4_K_M oracle).
    Relative {
        /// Relative error bound.
        eps: f32,
    },
}

impl std::fmt::Display for TolerancePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TolerancePolicy::BitExact => write!(f, "bit-exact"),
            TolerancePolicy::Ulp { ulp } => write!(f, "ulp:{ulp}"),
            TolerancePolicy::Relative { eps } => write!(f, "rel:{eps}"),
        }
    }
}

impl std::str::FromStr for TolerancePolicy {
    type Err = String;

    /// Grammar: `"bit-exact"` | `"ulp:N"` | `"rel:EPS"`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "bit-exact" {
            return Ok(TolerancePolicy::BitExact);
        }
        if let Some(rest) = s.strip_prefix("ulp:") {
            let ulp: u32 = rest.parse().map_err(|_| {
                format!("invalid tolerance policy '{s}': '{rest}' is not a valid u32 ULP count")
            })?;
            return Ok(TolerancePolicy::Ulp { ulp });
        }
        if let Some(rest) = s.strip_prefix("rel:") {
            let eps: f32 = rest.parse().map_err(|_| {
                format!("invalid tolerance policy '{s}': '{rest}' is not a valid f32 epsilon")
            })?;
            return Ok(TolerancePolicy::Relative { eps });
        }
        Err(format!(
            "invalid tolerance policy '{s}'; expected 'bit-exact', 'ulp:N', or 'rel:EPS'"
        ))
    }
}

// ── VerifyRequest ─────────────────────────────────────────────────────────────

/// Fully-specified verification request (shared by CLI + MCP layers).
///
/// Both `Layer 1` (CLI) and `Layer 2` (MCP tool) resolve their own surface
/// request shape into this struct before calling [`verify_rewrite`].
#[derive(Debug, Clone, serde::Deserialize)]
pub struct VerifyRequest {
    /// Original kernel source text.
    pub original: String,
    /// Rewritten kernel source text.
    pub rewritten: String,
    /// Shared `@strategy` hole resolution applied to BOTH sources (default empty).
    #[serde(default)]
    pub assignments: BTreeMap<String, i64>,
    /// Differential-comparison tolerance policy (verifier-owned; default `BitExact`).
    #[serde(default)]
    pub tolerance: TolerancePolicy,
    /// One size (bytes) per buffer binding. Required; length must equal
    /// `meta.binding_plan.buffers.len()` (checked at the CPU preflight step).
    pub buffer_sizes: Vec<usize>,
    /// Output-buffer sizes (bytes). Defaults to `buffer_sizes`.
    #[serde(default)]
    pub output_sizes: Option<Vec<usize>>,
    /// Explicit dispatch grid. Mandatory for coopmat/shared kernels (§3.1 step 4);
    /// otherwise derived via `derive_workgroups`.
    #[serde(default)]
    pub workgroups: Option<[u32; 3]>,
    /// Raw push-constant bytes. `None` ⇒ zero-fill UNLESS a float scalar is
    /// present, in which case zero-fill is forbidden (CRIT-1 — see the preflight
    /// `push_constants_required` gate).
    #[serde(default)]
    pub push_constants: Option<Vec<u8>>,
    /// Seed for deterministic input generation. Default `0`.
    #[serde(default)]
    pub seed: u64,
}

// ── Verdict / report types ───────────────────────────────────────────────────

/// The verdict — CLOSED set (§5). No other value is ever emitted.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Verdict {
    Pass,
    Fail,
    Reject,
    Skipped,
    NondeterministicOracle,
    Error,
}

/// Structural interface mismatch (anti-cheat gate output, §4.1).
///
/// Serializes with an internal `"sub"` tag matching the closed
/// `InterfaceMismatch` sub-kind set in §5 (`buffer_count`, `buffer_name`, …).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "sub", rename_all = "snake_case")]
pub enum InterfaceMismatch {
    BufferCount { original: usize, rewritten: usize },
    BufferName { index: usize, original: String, rewritten: String },
    BufferType { index: usize, detail: String },
    BufferAccess { index: usize, detail: String },
    ScalarMismatch { detail: String },
    PushConstantBytes { original: u32, rewritten: u32 },
    WorkgroupSize { original: [u32; 3], rewritten: [u32; 3] },
}

/// One output buffer's differential statistics.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BufferDiff {
    pub index: usize,
    pub size_bytes: usize,
    pub elem: String,
    pub n_diff: usize,
    pub total: usize,
    pub max_abs_diff: f64,
    pub max_ulp: u32,
    pub max_rel: f64,
}

/// Compact per-kernel summary embedded in the verdict (§5 `original`/`rewritten`).
#[derive(Debug, Clone, serde::Serialize)]
pub struct KernelSummary {
    pub kernel_name: String,
    pub buffers: usize,
    pub scalars: usize,
    pub workgroup: [u32; 3],
}

/// Reason payload for every non-PASS verdict (§5 closed `reason.kind` set).
///
/// `kind` is one of the closed strings in the spec's §5 table. `detail` carries
/// kind-specific structured data (e.g. lex/parse/hir counts for a compile
/// failure, the `InterfaceMismatch` for an interface rejection).
#[derive(Debug, Clone, serde::Serialize)]
pub struct Reason {
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<serde_json::Value>,
}

/// The full verification report. `verdict` serializes FIRST (declared first;
/// serde_json preserves struct field declaration order) per pipeline rule #3.
#[derive(Debug, Clone, serde::Serialize)]
pub struct VerifyReport {
    pub verdict: Verdict,
    pub milestone: &'static str,
    /// Echoes the tolerance policy used (audit trail), e.g. `"bit-exact"`.
    pub policy: String,
    /// `true` iff an MCP caller loosened past the server-owned default (§4.2).
    /// The core never sets this; the MCP handler stamps it after calling
    /// [`verify_rewrite`].
    pub tolerance_overridden: bool,
    pub seed: u64,
    /// Producing stage — closed set: `compile | spirv_val | interface | preflight
    /// | self_check | dispatch | output`.
    pub stage: String,
    pub original: Option<KernelSummary>,
    pub rewritten: Option<KernelSummary>,
    /// Present when dispatched; `None` on CPU-only verdicts.
    pub device: Option<String>,
    pub buffers_compared: Vec<BufferDiff>,
    pub reason: Option<Reason>,
    pub notes: Vec<String>,
}

const MILESTONE: &str = "M3.16";

fn kernel_summary(meta: &KernelMetadata) -> KernelSummary {
    KernelSummary {
        kernel_name: meta.kernel_name.clone(),
        buffers: meta.binding_plan.buffers.len(),
        scalars: meta.binding_plan.scalars.len(),
        workgroup: meta.workgroup_size,
    }
}

fn reason(kind: &str, detail: Option<serde_json::Value>) -> Option<Reason> {
    Some(Reason { kind: kind.to_string(), detail })
}

#[allow(clippy::too_many_arguments)]
fn make_report(
    req: &VerifyRequest,
    verdict: Verdict,
    stage: &str,
    original: Option<KernelSummary>,
    rewritten: Option<KernelSummary>,
    device: Option<String>,
    buffers_compared: Vec<BufferDiff>,
    reason: Option<Reason>,
) -> VerifyReport {
    VerifyReport {
        verdict,
        milestone: MILESTONE,
        policy: req.tolerance.to_string(),
        tolerance_overridden: false,
        seed: req.seed,
        stage: stage.to_string(),
        original,
        rewritten,
        device,
        buffers_compared,
        reason,
        notes: Vec::new(),
    }
}

// ── diff_interface (§4.1) ─────────────────────────────────────────────────────

/// Pure interface gate — CPU-only, fully unit-testable, runs BEFORE any GPU work.
///
/// A rewrite may change `kernel_name`/`entry_point`, `coopmat` shape,
/// `shared_memory_bytes`, SPIR-V capabilities, and body structure — that is
/// exactly what a structural rewrite is allowed to do (§4.1 "NOT gated"). It may
/// NOT change buffer count/name/element-type/access-mode/position, scalar
/// count/name/type/offset, push-constant total bytes, or workgroup size.
pub fn diff_interface(
    original: &KernelMetadata,
    rewritten: &KernelMetadata,
) -> Option<InterfaceMismatch> {
    let ob = &original.binding_plan.buffers;
    let rb = &rewritten.binding_plan.buffers;
    if ob.len() != rb.len() {
        return Some(InterfaceMismatch::BufferCount { original: ob.len(), rewritten: rb.len() });
    }
    for (i, (o, r)) in ob.iter().zip(rb.iter()).enumerate() {
        if o.name != r.name {
            return Some(InterfaceMismatch::BufferName {
                index: i,
                original: o.name.clone(),
                rewritten: r.name.clone(),
            });
        }
        if o.ty.elem != r.ty.elem || o.buffer_position != r.buffer_position {
            return Some(InterfaceMismatch::BufferType {
                index: i,
                detail: format!(
                    "elem original={:?} rewritten={:?}; buffer_position original={} rewritten={}",
                    o.ty.elem, r.ty.elem, o.buffer_position, r.buffer_position
                ),
            });
        }
        if o.ty.access != r.ty.access {
            return Some(InterfaceMismatch::BufferAccess {
                index: i,
                detail: format!("buffer[{i}] access differs: original={:?} rewritten={:?}", o.ty.access, r.ty.access),
            });
        }
    }

    let os = &original.binding_plan.scalars;
    let rs = &rewritten.binding_plan.scalars;
    if os.len() != rs.len() {
        return Some(InterfaceMismatch::ScalarMismatch {
            detail: format!("scalar count differs: original={} rewritten={}", os.len(), rs.len()),
        });
    }
    for (o, r) in os.iter().zip(rs.iter()) {
        if o.name != r.name || o.ty != r.ty || o.offset != r.offset || o.member_index != r.member_index {
            return Some(InterfaceMismatch::ScalarMismatch {
                detail: format!(
                    "scalar '{}' differs: original={{ty:{:?},offset:{},member_index:{}}} rewritten={{name:{},ty:{:?},offset:{},member_index:{}}}",
                    o.name, o.ty, o.offset, o.member_index, r.name, r.ty, r.offset, r.member_index
                ),
            });
        }
    }

    if original.push_constant_total_bytes != rewritten.push_constant_total_bytes {
        return Some(InterfaceMismatch::PushConstantBytes {
            original: original.push_constant_total_bytes,
            rewritten: rewritten.push_constant_total_bytes,
        });
    }

    if original.workgroup_size != rewritten.workgroup_size {
        return Some(InterfaceMismatch::WorkgroupSize {
            original: original.workgroup_size,
            rewritten: rewritten.workgroup_size,
        });
    }

    None
}

// ── Input generation (§3.2) ───────────────────────────────────────────────────

/// Element-type-dispatched deterministic buffer fill (r2/CRIT-3).
///
/// Float buffers (f32/f16) get a well-conditioned, finite, seeded `[-1,1]`
/// fill (denormal/NaN/Inf-free by construction). Integer/packed buffers reuse
/// `seeded_inputs`'s raw-byte fill (every bit pattern is a valid packed payload).
fn build_inputs(plan: &ParamBindingPlan, buffer_sizes: &[usize], seed: u64) -> Vec<Vec<u8>> {
    plan.buffers
        .iter()
        .enumerate()
        .map(|(b, slot)| {
            let size: usize = buffer_sizes[b];
            match slot.ty.elem {
                ScalarTy::F32 => fill_f32(size, seed, b as u64),
                ScalarTy::F16 => fill_f16(size, seed, b as u64),
                _ => {
                    // Integer / packed-quant: identical derivation to seeded_inputs
                    // (StdRng::seed_from_u64(seed ^ b)), so this is a single-buffer
                    // call to the SAME reused helper, not a re-derivation.
                    seeded_inputs(&[size], seed ^ (b as u64))
                        .into_iter()
                        .next()
                        .unwrap_or_default()
                }
            }
        })
        .collect()
}

/// Well-conditioned seeded `[-1,1]` f32 fill (r2/CRIT-3 formula, §3.2).
fn fill_f32(size_bytes: usize, seed: u64, binding_index: u64) -> Vec<u8> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ binding_index);
    let n: usize = size_bytes / 4;
    let mut out: Vec<u8> = Vec::with_capacity(size_bytes);
    for _ in 0..n {
        let u: u32 = rng.next_u32();
        let v: f64 = (u as f64 / u32::MAX as f64) * 2.0 - 1.0;
        out.extend_from_slice(&(v as f32).to_le_bytes());
    }
    out.resize(size_bytes, 0);
    out
}

/// Well-conditioned seeded `[-1,1]` f16 fill (rounds the same `[-1,1]` f32 draw).
fn fill_f16(size_bytes: usize, seed: u64, binding_index: u64) -> Vec<u8> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ binding_index);
    let n: usize = size_bytes / 2;
    let mut out: Vec<u8> = Vec::with_capacity(size_bytes);
    for _ in 0..n {
        let u: u32 = rng.next_u32();
        let v: f64 = (u as f64 / u32::MAX as f64) * 2.0 - 1.0;
        let h = half::f16::from_f32(v as f32);
        out.extend_from_slice(&h.to_le_bytes());
    }
    out.resize(size_bytes, 0);
    out
}

// ── Comparison (§3.3) ─────────────────────────────────────────────────────────

/// Bit-distance between two f32 values (mirrors `within_ulp_f32`'s internal
/// metric). Returns `u32::MAX` when either operand is NaN (documented sentinel;
/// NEVER used as a pass/fail decision directly — see `within_ulp_f32`).
fn ulp_diff_f32(a: f32, b: f32) -> u32 {
    if a.is_nan() || b.is_nan() {
        return u32::MAX;
    }
    if a == b {
        return 0;
    }
    let ai: i32 = a.to_bits() as i32;
    let bi: i32 = b.to_bits() as i32;
    ai.abs_diff(bi)
}

/// Relative-diff denominator floor, matching the codebase's established
/// `q4km_oracle::max_rel_diff` convention.
const REL_TINY: f64 = 1e-8;

fn compare_elems_f32(
    n: usize,
    policy: TolerancePolicy,
    get_a: impl Fn(usize) -> f32,
    get_b: impl Fn(usize) -> f32,
) -> (usize, f64, u32, f64) {
    let mut n_diff: usize = 0;
    let mut max_abs_diff: f64 = 0.0;
    let mut max_ulp: u32 = 0;
    let mut max_rel: f64 = 0.0;

    for i in 0..n {
        let av: f32 = get_a(i);
        let bv: f32 = get_b(i);
        let abs_diff: f64 = (av as f64 - bv as f64).abs();
        if abs_diff > max_abs_diff {
            max_abs_diff = abs_diff;
        }
        let ud: u32 = ulp_diff_f32(av, bv);
        if ud != u32::MAX && ud > max_ulp {
            max_ulp = ud;
        }
        let denom: f64 = (av as f64).abs().max((bv as f64).abs()).max(REL_TINY);
        let rel: f64 = abs_diff / denom;
        if rel > max_rel {
            max_rel = rel;
        }

        let ok: bool = match policy {
            TolerancePolicy::BitExact => av.to_bits() == bv.to_bits(),
            TolerancePolicy::Ulp { ulp } => within_ulp_f32(av, bv, ulp),
            TolerancePolicy::Relative { eps } => {
                // NaN handling: a rewrite cannot launder NaNs by comparison-vacuity.
                !av.is_nan() && !bv.is_nan() && abs_diff <= (eps as f64) * denom
            }
        };
        if !ok {
            n_diff += 1;
        }
    }
    (n_diff, max_abs_diff, max_ulp, max_rel)
}

fn compare_f32_buffer(index: usize, a: &[u8], b: &[u8], size_bytes: usize, policy: TolerancePolicy) -> (BufferDiff, bool) {
    let n: usize = size_bytes / 4;
    let get_a = |i: usize| f32::from_le_bytes([a[i * 4], a[i * 4 + 1], a[i * 4 + 2], a[i * 4 + 3]]);
    let get_b = |i: usize| f32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]]);
    let (n_diff, max_abs_diff, max_ulp, max_rel) = compare_elems_f32(n, policy, get_a, get_b);
    let diff = BufferDiff { index, size_bytes, elem: "f32".to_string(), n_diff, total: n, max_abs_diff, max_ulp, max_rel };
    let pass: bool = n_diff == 0;
    (diff, pass)
}

fn compare_f16_buffer(index: usize, a: &[u8], b: &[u8], size_bytes: usize, policy: TolerancePolicy) -> (BufferDiff, bool) {
    let n: usize = size_bytes / 2;
    if matches!(policy, TolerancePolicy::BitExact) {
        // Raw 16-bit bit-identity (§3.3) — avoids any f32-promotion collapsing
        // distinct f16 bit patterns onto the same promoted value.
        let mut n_diff: usize = 0;
        let mut max_abs_diff: f64 = 0.0;
        for i in 0..n {
            let ab: u16 = u16::from_le_bytes([a[i * 2], a[i * 2 + 1]]);
            let bb: u16 = u16::from_le_bytes([b[i * 2], b[i * 2 + 1]]);
            if ab != bb {
                n_diff += 1;
            }
            let av: f32 = half::f16::from_bits(ab).to_f32();
            let bv: f32 = half::f16::from_bits(bb).to_f32();
            let d: f64 = (av as f64 - bv as f64).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
            }
        }
        let diff = BufferDiff { index, size_bytes, elem: "f16".to_string(), n_diff, total: n, max_abs_diff, max_ulp: 0, max_rel: 0.0 };
        (diff, n_diff == 0)
    } else {
        let get_a = |i: usize| half::f16::from_le_bytes([a[i * 2], a[i * 2 + 1]]).to_f32();
        let get_b = |i: usize| half::f16::from_le_bytes([b[i * 2], b[i * 2 + 1]]).to_f32();
        let (n_diff, max_abs_diff, max_ulp, max_rel) = compare_elems_f32(n, policy, get_a, get_b);
        let diff = BufferDiff { index, size_bytes, elem: "f16".to_string(), n_diff, total: n, max_abs_diff, max_ulp, max_rel };
        (diff, n_diff == 0)
    }
}

/// Fallback for non-f32/f16 element types (§3.3): raw-byte `BitExact` compare,
/// with a `notes` annotation pushed by the caller.
fn compare_raw_bytes(index: usize, elem: ScalarTy, a: &[u8], b: &[u8], size_bytes: usize) -> (BufferDiff, bool) {
    let n: usize = size_bytes.min(a.len()).min(b.len());
    let n_diff: usize = (0..n).filter(|&i| a[i] != b[i]).count() + a.len().abs_diff(b.len());
    let diff = BufferDiff {
        index,
        size_bytes,
        elem: elem.display_name().to_string(),
        n_diff,
        total: size_bytes,
        max_abs_diff: 0.0,
        max_ulp: 0,
        max_rel: 0.0,
    };
    (diff, n_diff == 0)
}

/// Differential compare of every non-skipped output buffer (§3.1 step 9, §3.3).
///
/// A buffer is skipped (not compared) when its `output_sizes[i] == 0` OR when
/// its access mode is `ReadOnly` — mirroring `axc_runtime::kernel_handle::
/// binding_is_readback` (`output_size > 0 && access != ReadOnly`), which is
/// what `dispatch_handle` itself uses to decide whether to read a binding
/// back at all. A `ReadOnly` slot's output entry is ALWAYS an empty `Vec<u8>`
/// regardless of the requested `output_sizes[i]` — comparing it would either
/// be a meaningless always-pass (both empty) or, worse, an out-of-bounds
/// index panic on the element-typed comparators below, so it is excluded
/// here rather than relying on the caller to zero `output_sizes` for it.
fn compare_outputs(
    orig_outputs: &[Vec<u8>],
    rewritten_outputs: &[Vec<u8>],
    plan: &ParamBindingPlan,
    output_sizes: &[usize],
    policy: TolerancePolicy,
    notes: &mut Vec<String>,
) -> (bool, Vec<BufferDiff>) {
    let mut buffers_compared: Vec<BufferDiff> = Vec::new();
    let mut all_pass: bool = true;

    for (i, slot) in plan.buffers.iter().enumerate() {
        let sz: usize = output_sizes.get(i).copied().unwrap_or(0);
        if sz == 0 || slot.ty.access == axc_hir::buffer::BufferAccess::ReadOnly {
            continue;
        }
        let a: &[u8] = orig_outputs.get(i).map(|v| v.as_slice()).unwrap_or(&[]);
        let b: &[u8] = rewritten_outputs.get(i).map(|v| v.as_slice()).unwrap_or(&[]);

        let (diff, pass) = match slot.ty.elem {
            ScalarTy::F32 => compare_f32_buffer(i, a, b, sz, policy),
            ScalarTy::F16 => compare_f16_buffer(i, a, b, sz, policy),
            other => {
                notes.push(format!(
                    "buffer[{i}] element type {other:?} is not f32/f16; compared via raw-byte BitExact fallback"
                ));
                compare_raw_bytes(i, other, a, b, sz)
            }
        };
        if !pass {
            all_pass = false;
        }
        buffers_compared.push(diff);
    }

    (all_pass, buffers_compared)
}

/// Byte-for-byte comparison of the ORIGINAL dispatched twice (§3.1 step 7,
/// r2/WARN-2). Returns `Some(stats)` describing every disagreeing buffer, or
/// `None` when both runs are byte-identical on every non-skipped buffer.
///
/// `pub(crate)` (M3.18 / r2 M-2): reused verbatim by `fuzz.rs`'s determinism leg
/// (c) — avoids duplicating the ~20-LOC byte-diff (reuse-over-duplication,
/// driver-crate-only visibility change, does NOT breach the zero-compiler-change
/// gate).
pub(crate) fn find_nondeterminism(
    run1: &[Vec<u8>],
    run2: &[Vec<u8>],
    output_sizes: &[usize],
) -> Option<Vec<serde_json::Value>> {
    let mut stats: Vec<serde_json::Value> = Vec::new();
    for i in 0..run1.len().min(run2.len()) {
        if output_sizes.get(i).copied().unwrap_or(0) == 0 {
            continue;
        }
        let a: &[u8] = &run1[i];
        let b: &[u8] = &run2[i];
        let n_diff_bytes: usize = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len());
        if n_diff_bytes > 0 {
            stats.push(serde_json::json!({
                "index": i,
                "n_diff_bytes": n_diff_bytes,
                "size_bytes": a.len(),
            }));
        }
    }
    if stats.is_empty() { None } else { Some(stats) }
}

// ── Compile-error extraction ──────────────────────────────────────────────────

fn compile_error_detail(side: &str, e: &crate::DriverError) -> serde_json::Value {
    let (lex, parse, hir) = match e {
        crate::DriverError::Compile { lex, parse, hir } => (lex.len(), parse.len(), hir.len()),
        _ => (0, 0, 0),
    };
    serde_json::json!({
        "side": side,
        "detail": e.to_string(),
        "lex_count": lex,
        "parse_count": parse,
        "hir_count": hir,
    })
}

// ── Preflight (§3.1 step 4) ───────────────────────────────────────────────────

/// Request-completeness preflight (r2/WARN-1 + CRIT-1). CPU-only; runs before
/// any GPU work. Returns `Some(report)` on a hard ERROR, `None` when the
/// request is complete enough to proceed.
fn preflight_check(
    req: &VerifyRequest,
    meta_orig: &KernelMetadata,
    meta_rewritten: &KernelMetadata,
) -> Option<VerifyReport> {
    let needs_workgroups: bool = meta_orig.coopmat.is_some()
        || meta_orig.shared_memory_bytes > 0
        || meta_rewritten.coopmat.is_some()
        || meta_rewritten.shared_memory_bytes > 0;
    if needs_workgroups && req.workgroups.is_none() {
        return Some(make_report(
            req,
            Verdict::Error,
            "preflight",
            Some(kernel_summary(meta_orig)),
            Some(kernel_summary(meta_rewritten)),
            None,
            Vec::new(),
            reason(
                "workgroups_required",
                Some(serde_json::json!({
                    "detail": "kernel uses coopmat or shared memory but no explicit `workgroups` was supplied (derive_workgroups is element-wise-1-D-only)",
                })),
            ),
        ));
    }

    let has_float_scalar: bool = meta_orig.binding_plan.scalars.iter().any(|s| s.ty.is_float())
        || meta_rewritten.binding_plan.scalars.iter().any(|s| s.ty.is_float());
    if has_float_scalar && req.push_constants.is_none() {
        return Some(make_report(
            req,
            Verdict::Error,
            "preflight",
            Some(kernel_summary(meta_orig)),
            Some(kernel_summary(meta_rewritten)),
            None,
            Vec::new(),
            reason(
                "push_constants_required",
                Some(serde_json::json!({
                    "detail": "kernel has a float (f32/f16) push-constant scalar but no push_constants were supplied; the silent all-zero default is FORBIDDEN (CRIT-1 alpha=0 annihilation guard)",
                })),
            ),
        ));
    }

    None
}

// ── verify_rewrite (the orchestration core) ───────────────────────────────────

/// The orchestration core (§3.1). `vk == None` ⇒ CPU-only gates run, then the
/// verdict is `SKIPPED`/`gpu_unavailable`. Never panics: a non-compiling or
/// structurally-invalid rewrite becomes a `FAIL`/`REJECT` report, never an `Err`.
pub fn verify_rewrite(req: &VerifyRequest, vk: Option<&VulkanContext>) -> VerifyReport {
    // ── Step 1: compile both ──────────────────────────────────────────────
    let orig_compiled = crate::compile_source_with_assignments(&req.original, &req.assignments);
    let (orig_bytes, meta_orig) = match orig_compiled {
        Ok(v) => v,
        Err(e) => {
            return make_report(
                req, Verdict::Fail, "compile", None, None, None, Vec::new(),
                reason("compile_fail_original", Some(compile_error_detail("original", &e))),
            );
        }
    };

    let rewritten_compiled = crate::compile_source_with_assignments(&req.rewritten, &req.assignments);
    let (rewritten_bytes, meta_rewritten) = match rewritten_compiled {
        Ok(v) => v,
        Err(e) => {
            return make_report(
                req, Verdict::Fail, "compile",
                Some(kernel_summary(&meta_orig)), None, None, Vec::new(),
                reason("compile_fail_rewritten", Some(compile_error_detail("rewritten", &e))),
            );
        }
    };

    // ── Step 2: spirv-val both ─────────────────────────────────────────────
    if let Err(e) = validate_spirv(&orig_bytes) {
        return make_report(
            req, Verdict::Fail, "spirv_val",
            Some(kernel_summary(&meta_orig)), Some(kernel_summary(&meta_rewritten)), None, Vec::new(),
            reason("val_fail_original", Some(serde_json::json!({"side": "original", "detail": e.to_string()}))),
        );
    }
    if let Err(e) = validate_spirv(&rewritten_bytes) {
        return make_report(
            req, Verdict::Fail, "spirv_val",
            Some(kernel_summary(&meta_orig)), Some(kernel_summary(&meta_rewritten)), None, Vec::new(),
            reason("val_fail_rewritten", Some(serde_json::json!({"side": "rewritten", "detail": e.to_string()}))),
        );
    }

    // ── Step 3: interface gate (anti-cheat, CPU-only) ─────────────────────
    if let Some(mismatch) = diff_interface(&meta_orig, &meta_rewritten) {
        let detail = serde_json::to_value(&mismatch).ok();
        return make_report(
            req, Verdict::Reject, "interface",
            Some(kernel_summary(&meta_orig)), Some(kernel_summary(&meta_rewritten)), None, Vec::new(),
            reason("interface_mismatch", detail),
        );
    }

    // ── Step 4: request-completeness preflight (CPU-only) ─────────────────
    if let Some(report) = preflight_check(req, &meta_orig, &meta_rewritten) {
        return report;
    }

    // buffer_sizes length must match the (now-proven-identical) buffer count.
    if req.buffer_sizes.len() != meta_orig.binding_plan.buffers.len() {
        return make_report(
            req, Verdict::Error, "preflight",
            Some(kernel_summary(&meta_orig)), Some(kernel_summary(&meta_rewritten)), None, Vec::new(),
            reason("infra_error", Some(serde_json::json!({
                "detail": format!(
                    "buffer_sizes.len()=={} but kernel has {} buffer bindings",
                    req.buffer_sizes.len(), meta_orig.binding_plan.buffers.len()
                ),
            }))),
        );
    }

    let output_sizes: Vec<usize> = req.output_sizes.clone().unwrap_or_else(|| req.buffer_sizes.clone());
    if output_sizes.len() != meta_orig.binding_plan.buffers.len() {
        return make_report(
            req, Verdict::Error, "preflight",
            Some(kernel_summary(&meta_orig)), Some(kernel_summary(&meta_rewritten)), None, Vec::new(),
            reason("infra_error", Some(serde_json::json!({
                "detail": format!(
                    "output_sizes.len()=={} but kernel has {} buffer bindings",
                    output_sizes.len(), meta_orig.binding_plan.buffers.len()
                ),
            }))),
        );
    }

    let pc_total: usize = meta_orig.push_constant_total_bytes as usize;
    let push_constants: Vec<u8> = match &req.push_constants {
        Some(bytes) => {
            if bytes.len() != pc_total {
                return make_report(
                    req, Verdict::Error, "preflight",
                    Some(kernel_summary(&meta_orig)), Some(kernel_summary(&meta_rewritten)), None, Vec::new(),
                    reason("infra_error", Some(serde_json::json!({
                        "detail": format!(
                            "push_constants.len()=={} but kernel expects {pc_total} bytes",
                            bytes.len()
                        ),
                    }))),
                );
            }
            bytes.clone()
        }
        None => vec![0_u8; pc_total],
    };

    let workgroups: [u32; 3] = derive_workgroups(&req.buffer_sizes, meta_orig.workgroup_size, req.workgroups);

    // ── Step 5: GPU availability ────────────────────────────────────────────
    let vk: &VulkanContext = match vk {
        Some(vk) => vk,
        None => {
            return make_report(
                req, Verdict::Skipped, "dispatch",
                Some(kernel_summary(&meta_orig)), Some(kernel_summary(&meta_rewritten)), None, Vec::new(),
                reason("gpu_unavailable", None),
            );
        }
    };

    // ── Step 6: deterministic input generation ─────────────────────────────
    let inputs: Vec<Vec<u8>> = build_inputs(&meta_orig.binding_plan, &req.buffer_sizes, req.seed);
    let input_refs: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();

    // ── Step 7: prepare + dispatch ORIGINAL TWICE (determinism self-check) ──
    let orig_words: Vec<u32> = bytes_to_words(&orig_bytes);
    let handle_orig = match vk.prepare_kernel_checked(
        &orig_words, &meta_orig.binding_plan, meta_orig.push_constant_total_bytes,
        &meta_orig.entry_point, meta_orig.coopmat.as_ref(), &meta_orig.kernel_name, meta_orig.shared_memory_bytes,
    ) {
        Ok(h) => h,
        Err(e) => return typed_original_skip(req, &meta_orig, &meta_rewritten, vk, e),
    };

    let run1 = match vk.dispatch_handle(&handle_orig, (workgroups[0], workgroups[1], workgroups[2]), &input_refs, &output_sizes, &push_constants) {
        Ok(o) => o,
        Err(e) => return typed_original_skip(req, &meta_orig, &meta_rewritten, vk, e),
    };
    let run2 = match vk.dispatch_handle(&handle_orig, (workgroups[0], workgroups[1], workgroups[2]), &input_refs, &output_sizes, &push_constants) {
        Ok(o) => o,
        Err(e) => return typed_original_skip(req, &meta_orig, &meta_rewritten, vk, e),
    };

    if let Some(stats) = find_nondeterminism(&run1, &run2, &output_sizes) {
        return make_report(
            req, Verdict::NondeterministicOracle, "self_check",
            Some(kernel_summary(&meta_orig)), Some(kernel_summary(&meta_rewritten)),
            Some(vk.physical_device_name().to_string()), Vec::new(),
            reason("original_nondeterministic", Some(serde_json::json!({ "disagreeing_buffers": stats }))),
        );
    }

    // ── Step 8: dispatch REWRITTEN ──────────────────────────────────────────
    let rewritten_words: Vec<u32> = bytes_to_words(&rewritten_bytes);
    let handle_rewritten = match vk.prepare_kernel_checked(
        &rewritten_words, &meta_rewritten.binding_plan, meta_rewritten.push_constant_total_bytes,
        &meta_rewritten.entry_point, meta_rewritten.coopmat.as_ref(), &meta_rewritten.kernel_name, meta_rewritten.shared_memory_bytes,
    ) {
        Ok(h) => h,
        Err(e) => return typed_rewritten_fail(req, &meta_orig, &meta_rewritten, vk, e),
    };

    let run_rewritten = match vk.dispatch_handle(&handle_rewritten, (workgroups[0], workgroups[1], workgroups[2]), &input_refs, &output_sizes, &push_constants) {
        Ok(o) => o,
        Err(e) => return typed_rewritten_fail(req, &meta_orig, &meta_rewritten, vk, e),
    };

    // ── Step 9: differential compare ────────────────────────────────────────
    let mut notes: Vec<String> = Vec::new();
    let (all_pass, buffers_compared) = compare_outputs(
        &run1, &run_rewritten, &meta_orig.binding_plan, &output_sizes, req.tolerance, &mut notes,
    );

    let mut report = if all_pass {
        make_report(
            req, Verdict::Pass, "output",
            Some(kernel_summary(&meta_orig)), Some(kernel_summary(&meta_rewritten)),
            Some(vk.physical_device_name().to_string()), buffers_compared, None,
        )
    } else {
        let buffers_failed: Vec<usize> = buffers_compared.iter().filter(|b| b.n_diff > 0).map(|b| b.index).collect();
        let worst = buffers_compared.iter().max_by(|a, b| {
            a.n_diff.cmp(&b.n_diff).then(a.max_rel.partial_cmp(&b.max_rel).unwrap_or(std::cmp::Ordering::Equal))
        });
        let detail = serde_json::json!({
            "buffers_failed": buffers_failed,
            "worst": worst.map(|w| serde_json::json!({"index": w.index, "n_diff": w.n_diff, "max_rel": w.max_rel})),
        });
        make_report(
            req, Verdict::Fail, "output",
            Some(kernel_summary(&meta_orig)), Some(kernel_summary(&meta_rewritten)),
            Some(vk.physical_device_name().to_string()), buffers_compared,
            reason("numeric_mismatch", Some(detail)),
        )
    };
    report.notes = notes;
    report
}

/// Map a `prepare_kernel_checked`/`dispatch_handle` error on the ORIGINAL side
/// to its typed verdict (§3.1 step 7, §5): `CoopMatUnsupported` /
/// `DeviceFeatureUnsupported` are typed-skips; anything else is
/// `SKIPPED`/`preflight_fail_original` (device incapacity, not a candidate
/// verdict — the original is not runnable here, so no verdict can be rendered).
fn typed_original_skip(
    req: &VerifyRequest,
    meta_orig: &KernelMetadata,
    meta_rewritten: &KernelMetadata,
    vk: &VulkanContext,
    e: DispatchError,
) -> VerifyReport {
    let device = Some(vk.physical_device_name().to_string());
    let (kind, detail) = match &e {
        DispatchError::CoopMatUnsupported { reason: r, .. } => ("coopmat_unsupported", r.clone()),
        DispatchError::DeviceFeatureUnsupported { feature, kernel } => {
            ("device_feature_unsupported", format!("{feature}/{kernel}"))
        }
        other => ("preflight_fail_original", other.to_string()),
    };
    make_report(
        req, Verdict::Skipped, "dispatch",
        Some(kernel_summary(meta_orig)), Some(kernel_summary(meta_rewritten)), device, Vec::new(),
        reason(kind, Some(serde_json::json!({ "detail": detail }))),
    )
}

/// Map a `prepare_kernel_checked`/`dispatch_handle` error on the REWRITTEN side
/// (§3.1 step 8, §5): `CoopMatUnsupported`/`DeviceFeatureUnsupported` are
/// typed-skips (device incapacity affects both sides equally); anything else
/// is `FAIL`/`preflight_fail_rewritten` — an unrunnable rewrite fails
/// verification.
fn typed_rewritten_fail(
    req: &VerifyRequest,
    meta_orig: &KernelMetadata,
    meta_rewritten: &KernelMetadata,
    vk: &VulkanContext,
    e: DispatchError,
) -> VerifyReport {
    let device = Some(vk.physical_device_name().to_string());
    match &e {
        DispatchError::CoopMatUnsupported { reason: r, .. } => make_report(
            req, Verdict::Skipped, "dispatch",
            Some(kernel_summary(meta_orig)), Some(kernel_summary(meta_rewritten)), device, Vec::new(),
            reason("coopmat_unsupported", Some(serde_json::json!({ "detail": r }))),
        ),
        DispatchError::DeviceFeatureUnsupported { feature, kernel } => make_report(
            req, Verdict::Skipped, "dispatch",
            Some(kernel_summary(meta_orig)), Some(kernel_summary(meta_rewritten)), device, Vec::new(),
            reason("device_feature_unsupported", Some(serde_json::json!({ "detail": format!("{feature}/{kernel}") }))),
        ),
        other => make_report(
            req, Verdict::Fail, "preflight",
            Some(kernel_summary(meta_orig)), Some(kernel_summary(meta_rewritten)), device, Vec::new(),
            reason("preflight_fail_rewritten", Some(serde_json::json!({ "detail": other.to_string() }))),
        ),
    }
}

/// CLI exit code for a verdict (§5).
///
/// `PASS`/`SKIPPED`/`NONDETERMINISTIC_ORACLE` → 0; `FAIL`/`REJECT` → 1.
/// `ERROR` is a usage/under-specified-request/I-O failure and is mapped by the
/// CLI layer separately (this helper covers only verdict-carrying exits).
pub fn exit_code_for_verdict(v: Verdict) -> i32 {
    match v {
        Verdict::Pass | Verdict::Skipped | Verdict::NondeterministicOracle => 0,
        Verdict::Fail | Verdict::Reject => 1,
        Verdict::Error => 2,
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SAXPY_SRC: &str = include_str!("../../../examples/saxpy.axc");
    const VECTOR_ADD_SRC: &str = concat!(
        "@kernel\n",
        "@workgroup(64, 1, 1)\n",
        "@intent(\"vector add\")\n",
        "@complexity(O(n))\n",
        "fn vector_add(a: readonly_buffer[f32], b: readonly_buffer[f32], c: buffer[f32]) -> void {\n",
        "    let i: u32 = gid(0);\n",
        "    c[i] = a[i] + b[i];\n",
        "    return;\n",
        "}\n",
    );

    fn compile(src: &str) -> KernelMetadata {
        let (_bytes, meta) = crate::compile_source_with_meta(src).expect("must compile");
        meta
    }

    fn base_saxpy_request() -> VerifyRequest {
        VerifyRequest {
            original: SAXPY_SRC.to_string(),
            rewritten: SAXPY_SRC.to_string(),
            assignments: BTreeMap::new(),
            tolerance: TolerancePolicy::BitExact,
            buffer_sizes: vec![256, 256],
            output_sizes: None,
            workgroups: None,
            push_constants: None,
            seed: 0,
        }
    }

    // ── AT-2843: TolerancePolicy::from_str round-trip + rejection ────────────

    #[test]
    fn at_2843_tolerance_policy_from_str_roundtrip() {
        assert_eq!("bit-exact".parse::<TolerancePolicy>().unwrap(), TolerancePolicy::BitExact);
        assert_eq!("ulp:4".parse::<TolerancePolicy>().unwrap(), TolerancePolicy::Ulp { ulp: 4 });
        assert_eq!("rel:0.001".parse::<TolerancePolicy>().unwrap(), TolerancePolicy::Relative { eps: 0.001 });

        assert!("garbage".parse::<TolerancePolicy>().is_err());
        assert!("ulp:".parse::<TolerancePolicy>().is_err());
        assert!("rel:x".parse::<TolerancePolicy>().is_err());
    }

    #[test]
    fn at_2843b_tolerance_policy_display_matches_grammar() {
        assert_eq!(TolerancePolicy::BitExact.to_string(), "bit-exact");
        assert_eq!(TolerancePolicy::Ulp { ulp: 4 }.to_string(), "ulp:4");
        assert_eq!(TolerancePolicy::Relative { eps: 0.001 }.to_string(), "rel:0.001");
    }

    // ── AT-2842: VerifyReport serializes with "verdict" as the FIRST key ─────

    #[test]
    fn at_2842_verdict_is_first_serialized_key() {
        for verdict in [
            Verdict::Pass, Verdict::Fail, Verdict::Reject,
            Verdict::Skipped, Verdict::NondeterministicOracle, Verdict::Error,
        ] {
            let report = VerifyReport {
                verdict,
                milestone: MILESTONE,
                policy: "bit-exact".to_string(),
                tolerance_overridden: false,
                seed: 0,
                stage: "output".to_string(),
                original: None,
                rewritten: None,
                device: None,
                buffers_compared: Vec::new(),
                reason: None,
                notes: Vec::new(),
            };
            let json = serde_json::to_string(&report).unwrap();
            // NOTE: inspect the RAW JSON string, not a re-parsed `serde_json::Value`
            // — without the `preserve_order` feature, `Value`'s object map is a
            // `BTreeMap` that re-sorts keys alphabetically on parse, which would
            // hide a field-order regression in the ACTUAL serializer output.
            assert!(
                json.starts_with("{\"verdict\":"),
                "verdict must be the FIRST serialized key; got {json}"
            );
        }
    }

    #[test]
    fn verdict_serializes_screaming_snake_case() {
        assert_eq!(serde_json::to_string(&Verdict::Pass).unwrap(), "\"PASS\"");
        assert_eq!(serde_json::to_string(&Verdict::NondeterministicOracle).unwrap(), "\"NONDETERMINISTIC_ORACLE\"");
        assert_eq!(serde_json::to_string(&Verdict::Reject).unwrap(), "\"REJECT\"");
    }

    // ── AT-2844: diff_interface(kernel, kernel) == None ───────────────────────

    #[test]
    fn at_2844_diff_interface_identical_kernel_is_none() {
        let meta = compile(SAXPY_SRC);
        let meta2 = compile(SAXPY_SRC);
        assert_eq!(diff_interface(&meta, &meta2), None);
    }

    // ── AT-2845: diff_interface detects buffer-count / access / workgroup ────

    #[test]
    fn at_2845_diff_interface_detects_buffer_count() {
        let saxpy = compile(SAXPY_SRC);
        let vadd = compile(VECTOR_ADD_SRC);
        let mismatch = diff_interface(&saxpy, &vadd);
        assert!(matches!(mismatch, Some(InterfaceMismatch::BufferCount { .. })), "got {mismatch:?}");
    }

    #[test]
    fn at_2845b_diff_interface_detects_buffer_access_flip() {
        let saxpy_wo: &str = concat!(
            "@kernel\n@workgroup(64, 1, 1)\n@intent(\"t\")\n@complexity(O(n))\n",
            "fn saxpy(n: u32, alpha: f32, x: readonly_buffer[f32], y: writeonly_buffer[f32]) -> void {\n",
            "    let i: u32 = gid(0);\n    y[i] = alpha * x[i];\n    return;\n}\n",
        );
        let orig = compile(SAXPY_SRC);
        let flipped = compile(saxpy_wo);
        let mismatch = diff_interface(&orig, &flipped);
        assert!(matches!(mismatch, Some(InterfaceMismatch::BufferAccess { .. })), "got {mismatch:?}");
    }

    #[test]
    fn at_2845c_diff_interface_detects_workgroup_size_change() {
        let saxpy_wg128: String = SAXPY_SRC.replace("@workgroup(64, 1, 1)", "@workgroup(128, 1, 1)");
        let orig = compile(SAXPY_SRC);
        let changed = compile(&saxpy_wg128);
        let mismatch = diff_interface(&orig, &changed);
        assert!(matches!(mismatch, Some(InterfaceMismatch::WorkgroupSize { .. })), "got {mismatch:?}");
    }

    // ── AT-2846: verify_rewrite with a non-compiling rewritten → FAIL/compile ─

    #[test]
    fn at_2846_non_compiling_rewritten_fails_no_panic() {
        let mut req = base_saxpy_request();
        req.rewritten = "not valid axc source $$$ @@@".to_string();
        let report = verify_rewrite(&req, None);
        assert_eq!(report.verdict, Verdict::Fail);
        assert_eq!(report.stage, "compile");
        let r = report.reason.expect("reason must be present");
        assert_eq!(r.kind, "compile_fail_rewritten");
    }

    // ── AT-2847: vk=None on a complete request → SKIPPED/gpu_unavailable ─────

    #[test]
    fn at_2847_vk_none_skips_gpu_unavailable() {
        let mut req = base_saxpy_request();
        // Float scalar `alpha` present; must supply push_constants (CRIT-1).
        req.push_constants = Some(vec![0_u8; 8]);
        let report = verify_rewrite(&req, None);
        assert_eq!(report.verdict, Verdict::Skipped);
        assert_eq!(report.stage, "dispatch");
        assert_eq!(report.reason.unwrap().kind, "gpu_unavailable");
        assert!(report.original.is_some());
        assert!(report.rewritten.is_some());
    }

    // ── AT-2848: kernel-name rename is legal; WriteOnly<->ReadWrite flip is not ─

    #[test]
    fn at_2848_kernel_name_rename_is_legal() {
        let renamed: &str = &SAXPY_SRC.replace("fn saxpy(", "fn saxpy_renamed(");
        let orig = compile(SAXPY_SRC);
        let renamed_meta = compile(renamed);
        assert_eq!(diff_interface(&orig, &renamed_meta), None, "a kernel-name rename must be a legal (non-gated) rewrite");
    }

    #[test]
    fn at_2848b_writeonly_to_readwrite_flip_is_gated() {
        let wo_src: &str = concat!(
            "@kernel\n@workgroup(64,1,1)\n@intent(\"t\")\n@complexity(O(n))\n",
            "fn k(a: readonly_buffer[f32], b: writeonly_buffer[f32]) -> void {\n",
            "    let i: u32 = gid(0);\n    b[i] = a[i];\n    return;\n}\n",
        );
        let rw_src: &str = concat!(
            "@kernel\n@workgroup(64,1,1)\n@intent(\"t\")\n@complexity(O(n))\n",
            "fn k(a: readonly_buffer[f32], b: buffer[f32]) -> void {\n",
            "    let i: u32 = gid(0);\n    b[i] = a[i];\n    return;\n}\n",
        );
        let wo = compile(wo_src);
        let rw = compile(rw_src);
        let mismatch = diff_interface(&wo, &rw);
        assert!(matches!(mismatch, Some(InterfaceMismatch::BufferAccess { .. })), "got {mismatch:?}");
    }

    // ── AT-2850b: preflight — workgroups_required / push_constants_required ──

    #[test]
    fn at_2850b_preflight_workgroups_required_for_coopmat() {
        let coopmat_src: &str = include_str!("../../../examples/matmul_rb_coopmat.axc");
        let mut assignments: BTreeMap<String, i64> = BTreeMap::new();
        for (k, v) in [("rb_m", 2), ("rb_n", 2), ("tile_k", 16), ("a_block_size", 512), ("b_block_size", 512)] {
            assignments.insert(k.to_string(), v);
        }
        let req = VerifyRequest {
            original: coopmat_src.to_string(),
            rewritten: coopmat_src.to_string(),
            assignments,
            tolerance: TolerancePolicy::BitExact,
            buffer_sizes: vec![2048, 2048, 2048],
            output_sizes: None,
            workgroups: None, // MISSING — must ERROR
            push_constants: Some(vec![0_u8; 12]),
            seed: 0,
        };
        let report = verify_rewrite(&req, None);
        assert_eq!(report.verdict, Verdict::Error);
        assert_eq!(report.stage, "preflight");
        assert_eq!(report.reason.unwrap().kind, "workgroups_required");
    }

    #[test]
    fn at_2850b_preflight_push_constants_required_for_float_scalar() {
        let req = base_saxpy_request(); // push_constants: None, saxpy has alpha:f32
        let report = verify_rewrite(&req, None);
        assert_eq!(report.verdict, Verdict::Error);
        assert_eq!(report.stage, "preflight");
        assert_eq!(report.reason.unwrap().kind, "push_constants_required");
    }

    // ── Additional core unit coverage ─────────────────────────────────────────

    #[test]
    fn interface_reject_is_reachable_with_vk_none() {
        // AT-2854 precondition check: interface REJECT fires before any GPU need.
        let mut req = base_saxpy_request();
        req.rewritten = VECTOR_ADD_SRC.to_string();
        req.buffer_sizes = vec![256, 256, 256];
        req.push_constants = Some(vec![0_u8; 8]);
        let report = verify_rewrite(&req, None);
        assert_eq!(report.verdict, Verdict::Reject);
        assert_eq!(report.stage, "interface");
        assert_eq!(report.reason.unwrap().kind, "interface_mismatch");
    }

    #[test]
    fn exit_code_mapping() {
        assert_eq!(exit_code_for_verdict(Verdict::Pass), 0);
        assert_eq!(exit_code_for_verdict(Verdict::Skipped), 0);
        assert_eq!(exit_code_for_verdict(Verdict::NondeterministicOracle), 0);
        assert_eq!(exit_code_for_verdict(Verdict::Fail), 1);
        assert_eq!(exit_code_for_verdict(Verdict::Reject), 1);
        assert_eq!(exit_code_for_verdict(Verdict::Error), 2);
    }

    #[test]
    fn compare_f32_buffer_bit_exact_detects_diff() {
        let a: Vec<u8> = 1.0_f32.to_le_bytes().to_vec();
        let b: Vec<u8> = 1.0000001_f32.to_le_bytes().to_vec();
        let (diff, pass) = compare_f32_buffer(0, &a, &b, 4, TolerancePolicy::BitExact);
        assert!(!pass);
        assert_eq!(diff.n_diff, 1);
        let (_diff2, pass2) = compare_f32_buffer(0, &a, &b, 4, TolerancePolicy::Relative { eps: 1e-3 });
        assert!(pass2, "1 ULP-scale perturbation must pass under rel:1e-3");
    }

    #[test]
    fn compare_f16_buffer_bit_exact_raw_16bit() {
        let a: Vec<u8> = half::f16::from_f32(1.0).to_le_bytes().to_vec();
        let b: Vec<u8> = half::f16::from_f32(1.0).to_le_bytes().to_vec();
        let (diff, pass) = compare_f16_buffer(0, &a, &b, 2, TolerancePolicy::BitExact);
        assert!(pass);
        assert_eq!(diff.n_diff, 0);
    }

    #[test]
    fn find_nondeterminism_none_when_identical() {
        let run1 = vec![vec![1, 2, 3], vec![4, 5]];
        let run2 = vec![vec![1, 2, 3], vec![4, 5]];
        assert!(find_nondeterminism(&run1, &run2, &[3, 2]).is_none());
    }

    #[test]
    fn find_nondeterminism_some_when_different() {
        let run1 = vec![vec![1, 2, 3]];
        let run2 = vec![vec![1, 9, 3]];
        let stats = find_nondeterminism(&run1, &run2, &[3]).expect("must detect disagreement");
        assert_eq!(stats.len(), 1);
    }

    #[test]
    fn build_inputs_float_buffers_are_finite_and_in_range() {
        let meta = compile(SAXPY_SRC);
        let inputs = build_inputs(&meta.binding_plan, &[400, 400], 7);
        assert_eq!(inputs.len(), 2);
        for buf in &inputs {
            for chunk in buf.chunks_exact(4) {
                let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                assert!(v.is_finite(), "generated f32 must be finite; got {v}");
                assert!((-1.0..=1.0).contains(&v), "generated f32 must be in [-1,1]; got {v}");
            }
        }
    }

    #[test]
    fn build_inputs_deterministic_for_same_seed() {
        let meta = compile(SAXPY_SRC);
        let a = build_inputs(&meta.binding_plan, &[400, 400], 42);
        let b = build_inputs(&meta.binding_plan, &[400, 400], 42);
        assert_eq!(a, b);
        let c = build_inputs(&meta.binding_plan, &[400, 400], 43);
        assert_ne!(a, c);
    }

    // ── AT-2925 (M3.19 / FG.3): @optimization_log survives the M3.16 rewrite-flow ──

    /// AT-2925: a source carrying an `@optimization_log` block (including a
    /// `version > SUPPORTED` forward-compat block/entry key) passes through
    /// `verify_rewrite` — the SAME `original == rewritten` text compiles and
    /// clears the interface gate cleanly (SKIPPED on vk=None, not
    /// REJECT/FAIL), proving the block does not perturb the M3.16 pipeline —
    /// AND, via the identical compile entrypoint `verify_rewrite` itself uses
    /// (`compile_source_with_assignments`), the block's `forward_unknown`
    /// keys are carried byte-verbatim through HIR (stronger than mere
    /// accept: the exact key/value pairs from source are recoverable).
    #[test]
    fn at_2925_optimization_log_block_survives_rewrite_verify_flow_byte_verbatim() {
        // NOTE: only an ENTRY-level unknown key is reachable via real source
        // text — §1.3a's EBNF pins `block_item := version_field | entry_block`
        // as CLOSED, so THIS compiler's parser can never produce a block-level
        // `RecordField` other than `version` from source (a block-level
        // forward_unknown key is forward-looking infra for a FUTURE parser
        // that extends `block_item`; it is exercised only via direct HIR
        // construction, see `opt_log::tests::at_2924_v2_unknown_block_key_preserved_not_rejected`).
        // `field := ident ":" record_value` at the entry level, by contrast,
        // admits ANY identifier key syntactically, so `future_entry_key` here
        // is real, parser-reachable forward-compat input.
        const SRC_WITH_BLOCK: &str = concat!(
            "@kernel\n@workgroup(64, 1, 1)\n@intent(\"t\")\n@complexity(O(n))\n",
            "@optimization_log {\n",
            "  version: 2,\n",
            "  entry { ts: \"2026-07-14T12:00:00.000Z\", kernel: \"saxpy\", metric: 42860, unit: mtflops, verdict: PASS, future_entry_key: \"x\" }\n",
            "}\n",
            "fn saxpy(n: u32, alpha: f32, x: readonly_buffer[f32], y: buffer[f32]) -> void {\n",
            "    let i: u32 = gid(0);\n    y[i] = alpha * x[i] + y[i];\n    return;\n}\n",
        );

        let mut req = base_saxpy_request();
        req.original = SRC_WITH_BLOCK.to_string();
        req.rewritten = SRC_WITH_BLOCK.to_string();
        req.push_constants = Some(vec![0_u8; 8]); // alpha:f32 present (CRIT-1)

        let report = verify_rewrite(&req, None);
        assert_eq!(
            report.verdict,
            Verdict::Skipped,
            "block-carrying source must clear compile+interface (SKIPPED only for lack of GPU, never REJECT/FAIL): {report:?}"
        );
        assert_eq!(report.stage, "dispatch");

        // Byte-verbatim carry check, through the SAME compile entrypoint
        // verify_rewrite uses internally (compile_source_with_assignments),
        // not a bespoke one.
        let (_bytes, _meta) = crate::compile_source_with_assignments(SRC_WITH_BLOCK, &BTreeMap::new())
            .expect("verify_rewrite's own compile path must succeed for a block-carrying source");
        let (ast, lex_errs, parse_errs) = axc_parser::parse(SRC_WITH_BLOCK);
        assert!(lex_errs.is_empty() && parse_errs.is_empty());
        let (hir, hir_errs, _warnings) = axc_hir::lower_module(&ast);
        assert!(hir_errs.is_empty(), "hir errors: {hir_errs:?}");
        let log = hir.kernels[0].annotations.opt_log.as_ref().expect("opt_log must be Some");
        assert_eq!(log.version, 2);
        assert!(log.forward_unknown.is_empty(), "no block-level unknown key was declared in this source");
        assert_eq!(
            log.entries[0].forward_unknown,
            vec![("future_entry_key".to_string(), axc_parser::ast::RecordValue::Str("x".to_string()))],
            "entry-level forward_unknown must carry the exact source key/value"
        );
    }
}
