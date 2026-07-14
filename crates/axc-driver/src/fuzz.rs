//! M3.18 (FG.8) — `axc test --fuzz`: a precondition-satisfying input fuzzer with a
//! runtime debug-check oracle (§3 of the spec).
//!
//! There is no reference output (that is `rewrite_verify`'s job, a different tool).
//! A no-reference fuzzer can honestly check exactly three things (the "oracle
//! triad", §3.2):
//!   (a) no device fault / timeout on a precondition-satisfying input,
//!   (b) postconditions hold at runtime (via the M3.17 debug-flag oracle), and
//!   (c) the kernel is deterministic on a fixed input (twin-dispatch byte-compare).
//!
//! The ONE substantive new algorithm is the interval satisfier (§3.4): `solve`
//! turns a kernel's lowered `@precondition` `DebugCheck`s into per-scalar interval
//! constraints (or `Unsat`), and `sample` deterministically draws a satisfying
//! value per run. Both are pure, CPU-only, and unit-tested without touching the
//! GPU or even a compiler pass (synthetic `DebugCheck` vectors).
//!
//! `fuzz_kernel` is the orchestration: front-end -> solve -> [GPU unavailable ->
//! SKIPPED] -> compile the needed artifact(s) -> per-run dispatch loop -> the
//! TOTAL 4-quadrant precondition discriminator (`classify_run`, r3/fix-4) ->
//! determinism leg -> optional `--violate` leg -> the severity-ordered verdict
//! (r4: `ERROR > FAIL > UNSATISFIABLE > UNCONSTRAINED_PRECONDITION_FIRED > SKIPPED
//! > PASS`).
//!
//! Reuse (§4 of the spec, "no duplication"): `axc_hir::{DebugCheck, DebugCheckKind,
//! DebugCompareOp, DebugOperand}` (already `pub`, consumed not re-parsed),
//! `axc_hir::lower_module`, `compile_source_with_meta[_debug]`,
//! `VulkanContext::dispatch_debug_checked` (dispatch.rs:144 — NEVER
//! `dispatch_handle`, which never decodes the flag words and would false-PASS
//! every run, r2/C-1), `rewrite_verify::find_nondeterminism` (now `pub(crate)`),
//! `mcp::tools::bench_variant::{seeded_inputs, derive_workgroups}`,
//! `mcp::tools::compile_variant::bytes_to_words`.

use std::collections::{BTreeMap, BTreeSet};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use axc_hir::{lower_module, DebugCheck, DebugCheckKind, DebugCompareOp, DebugOperand, KernelParam, ParamTy, ScalarTy};
use axc_runtime::{DispatchError, DispatchRequest, KernelMetadata, VulkanContext};

use crate::mcp::tools::bench_variant::{derive_workgroups, seeded_inputs};
use crate::mcp::tools::compile_variant::bytes_to_words;

const MILESTONE: &str = "M3.18";

/// Default `--scalar-window` (§3.1 contract).
pub const DEFAULT_SCALAR_WINDOW: i64 = 256;

// ══════════════════════════════════════════════════════════════════════════════
// §3.4 — the interval satisfier (pure, CPU-only, unit-testable)
// ══════════════════════════════════════════════════════════════════════════════

/// An unsatisfiable constraint set (a genuine finding about the *spec*, not a bug).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unsat {
    pub reason: String,
}

/// Exact integer interval domain for one `U32`/`I32` scalar (`i128` arithmetic,
/// §3.4). `new` seeds `[lo,hi]` from `ScalarTy::int_range()` — an UNTOUCHED domain
/// (`lo==tmin && hi==tmax`) is indistinguishable from "no constraint narrowed this
/// scalar" and is treated as the "unbounded" windowing bucket by `sample`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntDomain {
    pub lo: i128,
    pub hi: i128,
    pub excluded: BTreeSet<i128>,
    pub pinned: Option<i128>,
}

impl IntDomain {
    fn new(ty: ScalarTy) -> Self {
        let (lo, hi) = ty.int_range().expect("IntDomain::new requires an integer ScalarTy");
        Self { lo, hi, excluded: BTreeSet::new(), pinned: None }
    }

    /// Apply one predicate; fails fast (Eq-pin conflicts detected immediately,
    /// everything else re-checked via `check_sat` after every apply).
    fn apply(&mut self, op: DebugCompareOp, k: i128) -> Result<(), String> {
        match op {
            DebugCompareOp::Gt => self.lo = self.lo.max(k + 1),
            DebugCompareOp::Ge => self.lo = self.lo.max(k),
            DebugCompareOp::Lt => self.hi = self.hi.min(k - 1),
            DebugCompareOp::Le => self.hi = self.hi.min(k),
            DebugCompareOp::Eq => {
                if let Some(p) = self.pinned {
                    if p != k {
                        return Err(format!("conflicting eq pins {p} and {k}"));
                    }
                }
                self.pinned = Some(k);
            }
            DebugCompareOp::Ne => {
                self.excluded.insert(k);
            }
        }
        self.check_sat()
    }

    fn check_sat(&self) -> Result<(), String> {
        if self.lo > self.hi {
            return Err(format!("interval empty: [{}, {}]", self.lo, self.hi));
        }
        if let Some(p) = self.pinned {
            if p < self.lo || p > self.hi || self.excluded.contains(&p) {
                return Err(format!("eq pin {p} conflicts with interval [{}, {}] / exclusions", self.lo, self.hi));
            }
        }
        if self.lo == self.hi && self.excluded.contains(&self.lo) {
            return Err(format!("single remaining point {} is excluded", self.lo));
        }
        Ok(())
    }
}

/// Float interval domain (`F32`-only, r3/C-1new) with open/closed endpoint flags.
/// A fresh domain has no info: `(-inf, +inf)`, both open (matches "unbounded").
#[derive(Debug, Clone, PartialEq)]
pub struct FloatDomain {
    pub lo: f32,
    pub hi: f32,
    pub lo_open: bool,
    pub hi_open: bool,
    pub pinned: Option<f32>,
}

impl FloatDomain {
    fn new() -> Self {
        Self { lo: f32::NEG_INFINITY, hi: f32::INFINITY, lo_open: true, hi_open: true, pinned: None }
    }

    fn apply(&mut self, op: DebugCompareOp, k: f32) -> Result<(), String> {
        match op {
            DebugCompareOp::Gt => self.tighten_lower(k, true),
            DebugCompareOp::Ge => self.tighten_lower(k, false),
            DebugCompareOp::Lt => self.tighten_upper(k, true),
            DebugCompareOp::Le => self.tighten_upper(k, false),
            DebugCompareOp::Eq => {
                if let Some(p) = self.pinned {
                    if p != k {
                        return Err(format!("conflicting eq pins {p} and {k}"));
                    }
                }
                self.pinned = Some(k);
            }
            DebugCompareOp::Ne => {
                if self.pinned == Some(k) {
                    return Err(format!("eq pin {k} conflicts with ne {k}"));
                }
            }
        }
        self.check_sat()
    }

    fn tighten_lower(&mut self, k: f32, open: bool) {
        if k > self.lo || (k == self.lo && open && !self.lo_open) {
            self.lo = k;
            self.lo_open = open;
        } else if k == self.lo {
            self.lo_open = self.lo_open || open;
        }
    }

    fn tighten_upper(&mut self, k: f32, open: bool) {
        if k < self.hi || (k == self.hi && open && !self.hi_open) {
            self.hi = k;
            self.hi_open = open;
        } else if k == self.hi {
            self.hi_open = self.hi_open || open;
        }
    }

    fn check_sat(&self) -> Result<(), String> {
        if self.lo > self.hi {
            return Err(format!("interval empty: ({}, {})", self.lo, self.hi));
        }
        if self.lo == self.hi && (self.lo_open || self.hi_open) {
            return Err(format!("degenerate open interval at {}", self.lo));
        }
        if let Some(p) = self.pinned {
            let lo_ok = p > self.lo || (p == self.lo && !self.lo_open);
            let hi_ok = p < self.hi || (p == self.hi && !self.hi_open);
            if !lo_ok || !hi_ok {
                return Err(format!("eq pin {p} outside interval"));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ScalarDomain {
    Int(IntDomain),
    Float(FloatDomain),
}

/// The satisfier's output: per-scalar domains (only for scalars that carry at
/// least one CONSTRAINED — single-scalar-vs-literal — `@precondition`), the
/// contributing predicate texts, deferred-relational notes, and the bit-index
/// partition (constrained vs. deferred) used by the fuzz orchestrator's
/// deferred-condition exemption (§3.2(b)).
#[derive(Debug, Clone, Default)]
pub struct Constraints {
    pub scalars: BTreeMap<String, (ScalarTy, ScalarDomain)>,
    pub from: BTreeMap<String, Vec<String>>,
    pub notes: Vec<String>,
    /// `Pre`-kind bit indices belonging to a deferred (two-`ScalarRef`) predicate.
    pub deferred_pre_bits: BTreeSet<u32>,
    /// `Pre`-kind bit indices belonging to a constrained (scalar-vs-literal) predicate.
    pub constrained_pre_bits: BTreeSet<u32>,
}

/// `gt(k,s) -> s<k`, `ge(k,s) -> s<=k`, `lt(k,s) -> s>k`, `le(k,s) -> s>=k`,
/// `eq`/`ne` symmetric (§3.4 flip table).
fn flip_op(op: DebugCompareOp) -> DebugCompareOp {
    match op {
        DebugCompareOp::Gt => DebugCompareOp::Lt,
        DebugCompareOp::Ge => DebugCompareOp::Le,
        DebugCompareOp::Lt => DebugCompareOp::Gt,
        DebugCompareOp::Le => DebugCompareOp::Ge,
        DebugCompareOp::Eq => DebugCompareOp::Eq,
        DebugCompareOp::Ne => DebugCompareOp::Ne,
    }
}

/// Normalize a single-scalar-vs-literal `DebugCheck` to `(scalar_name, op, literal)`
/// with the scalar as the LHS. `None` for a two-`ScalarRef` (deferred) or malformed
/// shape (the latter cannot occur for a resolved `DebugCheck` — HIR guarantees
/// >=1 `ScalarRef` operand, §1.3 of the spec).
fn normalize(c: &DebugCheck) -> Option<(&str, DebugCompareOp, i64)> {
    match (&c.lhs, &c.rhs) {
        (DebugOperand::ScalarRef { name, .. }, DebugOperand::Lit(k)) => Some((name.as_str(), c.op, *k)),
        (DebugOperand::Lit(k), DebugOperand::ScalarRef { name, .. }) => Some((name.as_str(), flip_op(c.op), *k)),
        _ => None,
    }
}

fn is_relational(c: &DebugCheck) -> bool {
    matches!((&c.lhs, &c.rhs), (DebugOperand::ScalarRef { .. }, DebugOperand::ScalarRef { .. }))
}

/// Solve every `Pre`-kind `DebugCheck` into `Constraints`, or `Unsat` on the first
/// detected conflict. Two-`ScalarRef` relational predicates are NOT
/// interval-solvable in v1 — deferred with a `relational_constraint_deferred`
/// note (§3.4), never an error.
pub fn solve(checks: &[DebugCheck]) -> Result<Constraints, Unsat> {
    let mut out = Constraints::default();
    for c in checks.iter().filter(|c| c.kind == DebugCheckKind::Pre) {
        if is_relational(c) {
            out.deferred_pre_bits.insert(c.bit);
            out.notes.push(format!(
                "relational_constraint_deferred: {} (two-scalar predicates are not interval-solvable in v1)",
                c.text
            ));
            continue;
        }
        let Some((name, op, k)) = normalize(c) else {
            // Defensive: cannot happen for a resolved DebugCheck (HIR invariant),
            // but treat as deferred rather than panicking (no unwrap in library code).
            out.deferred_pre_bits.insert(c.bit);
            out.notes.push(format!("unrecognized_operand_shape_deferred: {}", c.text));
            continue;
        };
        out.constrained_pre_bits.insert(c.bit);
        out.from.entry(name.to_string()).or_default().push(c.text.clone());
        let entry = out.scalars.entry(name.to_string()).or_insert_with(|| {
            let dom = if c.ty == ScalarTy::F32 { ScalarDomain::Float(FloatDomain::new()) } else { ScalarDomain::Int(IntDomain::new(c.ty)) };
            (c.ty, dom)
        });
        let apply_result = match &mut entry.1 {
            ScalarDomain::Int(dom) => dom.apply(op, k as i128),
            ScalarDomain::Float(dom) => dom.apply(op, k as f32),
        };
        apply_result.map_err(|reason| Unsat { reason: format!("scalar `{name}`: {reason}") })?;
    }
    Ok(out)
}

// ── Windowing (§3.4 "Value generation") ─────────────────────────────────────────

fn default_int_window(ty: ScalarTy, window: i64) -> (i128, i128) {
    let (tmin, tmax) = ty.int_range().expect("default_int_window requires an integer ScalarTy");
    if ty.is_signed_integer() {
        ((-(window as i128) / 2).max(tmin), (window as i128 / 2).min(tmax))
    } else {
        (0i128.max(tmin), (window as i128).min(tmax))
    }
}

/// Effective windowed sampling range for a constrained int domain (§3.4):
/// both bounds finite -> use directly; one-sided -> `[lo, lo+W]` / `[hi-W, hi]`;
/// untouched (both at type extremes) -> the type-appropriate default window.
fn effective_int_window(ty: ScalarTy, dom: &IntDomain, window: i64) -> (i128, i128) {
    let (tmin, tmax) = ty.int_range().expect("effective_int_window requires an integer ScalarTy");
    let w = window as i128;
    let lower_bounded = dom.lo > tmin;
    let upper_bounded = dom.hi < tmax;
    match (lower_bounded, upper_bounded) {
        (true, true) => (dom.lo, dom.hi),
        (true, false) => (dom.lo, (dom.lo + w).min(tmax)),
        (false, true) => ((dom.hi - w).max(tmin), dom.hi),
        (false, false) => default_int_window(ty, window),
    }
}

/// `f32::next_up`/`next_down` per r3/C-1new — NEVER an `f64` nextafter. The
/// open-lower-bound-at-0 case floors at `f32::MIN_POSITIVE` (FTZ-safe).
fn adjust_open_lower(lo: f32, open: bool) -> f32 {
    if !open {
        return lo;
    }
    if lo == 0.0 { f32::MIN_POSITIVE } else { lo.next_up() }
}

fn adjust_open_upper(hi: f32, open: bool) -> f32 {
    if !open { hi } else { hi.next_down() }
}

/// Effective windowed sampling range for a constrained float domain (§3.4 + r2/M-1
/// + r3/C-1new open-bound strict-interior adjustment).
fn effective_float_window(dom: &FloatDomain, window: i64) -> (f32, f32) {
    let w = window as f32;
    let lower_bounded = dom.lo.is_finite();
    let upper_bounded = dom.hi.is_finite();

    let eff_lo: f32 = if lower_bounded {
        adjust_open_lower(dom.lo, dom.lo_open)
    } else if upper_bounded {
        dom.hi - w
    } else {
        0.0
    };
    let eff_hi: f32 = if upper_bounded {
        adjust_open_upper(dom.hi, dom.hi_open)
    } else if lower_bounded {
        dom.lo + w
    } else {
        w
    };
    (eff_lo, eff_hi)
}

fn int_value_to_le_bytes(ty: ScalarTy, value: i128) -> Vec<u8> {
    let width: usize = (ty.bit_width() as usize).div_ceil(8);
    match axc_hir::fit_int_literal(value, ty) {
        Ok(v) => v.bits.to_le_bytes()[..width].to_vec(),
        Err(_) => vec![0u8; width],
    }
}

fn float_value_to_le_bytes(ty: ScalarTy, value: f64) -> Vec<u8> {
    let width: usize = (ty.bit_width() as usize).div_ceil(8);
    match axc_hir::fit_float_literal(value, ty) {
        Ok(v) => v.bits.to_le_bytes()[..width].to_vec(),
        Err(_) => vec![0u8; width],
    }
}

fn draw_int_inclusive(lo: i128, hi: i128, rng: &mut StdRng) -> i128 {
    if lo >= hi { lo } else { rng.gen_range((lo as i64)..=(hi as i64)) as i128 }
}

fn sample_int_constrained(ty: ScalarTy, dom: &IntDomain, window: i64, rng: &mut StdRng) -> Vec<u8> {
    if let Some(p) = dom.pinned {
        return int_value_to_le_bytes(ty, p);
    }
    let (lo, hi) = effective_int_window(ty, dom, window);
    let mut v = draw_int_inclusive(lo, hi, rng);
    if dom.excluded.contains(&v) && lo < hi {
        let mut tries = 0;
        while dom.excluded.contains(&v) && tries < 64 {
            v = draw_int_inclusive(lo, hi, rng);
            tries += 1;
        }
        if dom.excluded.contains(&v) {
            if let Some(alt) = (lo..=hi).find(|x| !dom.excluded.contains(x)) {
                v = alt;
            }
        }
    }
    int_value_to_le_bytes(ty, v)
}

fn draw_f32_inclusive(lo: f32, hi: f32, rng: &mut StdRng) -> f32 {
    if lo >= hi { lo } else { rng.gen_range(lo..=hi) }
}

fn sample_float_constrained(dom: &FloatDomain, window: i64, rng: &mut StdRng) -> Vec<u8> {
    if let Some(p) = dom.pinned {
        return float_value_to_le_bytes(ScalarTy::F32, p as f64);
    }
    let (lo, hi) = effective_float_window(dom, window);
    let v = draw_f32_inclusive(lo, hi, rng);
    float_value_to_le_bytes(ScalarTy::F32, v as f64)
}

/// Fuzz a scalar with NO precondition at all: within a type-appropriate default
/// window (§3.4 "Honest v1 limitation" / edge case §10 "no preconditions").
/// Reuses `axc_hir::{fit_int_literal, fit_float_literal}` for the final encode.
fn default_fuzz_value(ty: ScalarTy, window: i64, rng: &mut StdRng) -> Vec<u8> {
    if ty.is_bool() {
        return vec![u8::from(rng.gen_bool(0.5))];
    }
    if ty.is_integer() {
        let (lo, hi) = default_int_window(ty, window);
        let v = draw_int_inclusive(lo, hi, rng);
        return int_value_to_le_bytes(ty, v);
    }
    let w = (window as f64).max(0.0);
    let v: f64 = if w <= 0.0 { 0.0 } else { rng.gen_range(0.0..=w) };
    float_value_to_le_bytes(ty, v)
}

/// Sample every scalar param for one run. Constrained scalars (present in
/// `constraints.scalars`) draw within their windowed domain; unconstrained
/// scalars draw within the default window (§3.4). Seeded per
/// `StdRng::seed_from_u64(seed ^ scalar_position ^ run)` — deterministic across
/// re-invocations for the same `(seed, run)` (AT-2891).
pub fn sample_all(constraints: &Constraints, scalar_params: &[KernelParam], window: i64, seed: u64, run: u32) -> BTreeMap<String, Vec<u8>> {
    let mut out: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for p in scalar_params {
        let ParamTy::Scalar(ty) = p.ty else { continue };
        let mut rng = StdRng::seed_from_u64(seed ^ (p.position as u64) ^ (run as u64));
        let bytes = match constraints.scalars.get(&p.name) {
            Some((_, ScalarDomain::Int(dom))) => sample_int_constrained(ty, dom, window, &mut rng),
            Some((_, ScalarDomain::Float(dom))) => sample_float_constrained(dom, window, &mut rng),
            None => default_fuzz_value(ty, window, &mut rng),
        };
        out.insert(p.name.clone(), bytes);
    }
    out
}

/// Encode sampled scalars into a `push_constant_total_bytes`-sized std430 buffer
/// (§3.4 "Emitting push-constants" — the exact layout codegen uses).
pub fn encode_push_constants(scalars: &[axc_hir::ScalarPushConstantSlot], values: &BTreeMap<String, Vec<u8>>, total_bytes: u32) -> Vec<u8> {
    let mut buf = vec![0u8; total_bytes as usize];
    for slot in scalars {
        if let Some(bytes) = values.get(&slot.name) {
            let start = slot.offset as usize;
            let end = start + bytes.len();
            if end <= buf.len() {
                buf[start..end].copy_from_slice(bytes);
            }
        }
    }
    buf
}

/// Negate ONE precondition (§3.3 `--violate`): produce a value that VIOLATES
/// `check` for its target scalar (`target_name` must equal the check's scalar).
pub fn negate_one(check: &DebugCheck, target_name: &str) -> Result<Vec<u8>, String> {
    let (name, op, k) = normalize(check).ok_or_else(|| format!("check `{}` is not a single-scalar-vs-literal predicate", check.text))?;
    if name != target_name {
        return Err(format!("check `{}` does not reference scalar `{target_name}`", check.text));
    }
    match check.ty {
        ScalarTy::F32 => {
            let k = k as f32;
            let v: f32 = match op {
                DebugCompareOp::Gt => k,
                DebugCompareOp::Ge => k - 1.0,
                DebugCompareOp::Lt => k,
                DebugCompareOp::Le => k + 1.0,
                DebugCompareOp::Eq => k + 1.0,
                DebugCompareOp::Ne => k,
            };
            Ok(float_value_to_le_bytes(ScalarTy::F32, v as f64))
        }
        ScalarTy::U32 | ScalarTy::I32 => {
            let (tmin, tmax) = check.ty.int_range().expect("U32/I32 always have an int_range");
            let kk = k as i128;
            let v: i128 = match op {
                DebugCompareOp::Gt => kk,
                DebugCompareOp::Ge => (kk - 1).max(tmin),
                DebugCompareOp::Lt => kk,
                DebugCompareOp::Le => (kk + 1).min(tmax),
                DebugCompareOp::Eq => if kk < tmax { kk + 1 } else { (kk - 1).max(tmin) },
                DebugCompareOp::Ne => kk,
            };
            Ok(int_value_to_le_bytes(check.ty, v))
        }
        other => Err(format!("negate_one: unsupported DebugCheck::ty {other:?} (v1 DSL is u32/i32/f32-only)")),
    }
}

// ── `--violate` eligibility scan (§3.3, r2/W-1) ─────────────────────────────────

const ELEMENT_COUNT_NAMES: &[&str] = &["n", "len", "length", "count", "num", "size", "width", "height", "idx", "index"];

fn is_conventional_element_count(name: &str) -> bool {
    ELEMENT_COUNT_NAMES.contains(&name.to_ascii_lowercase().as_str())
}

/// Conservative textual scan: does `scalar_name` appear as a whole identifier
/// inside ANY `[...]` bracket span of `source`? Over-inclusive by design — a false
/// EXCLUSION (skipping a scalar that was actually safe) is acceptable; a false
/// INCLUSION (violating a scalar that bounds an access) is not (r2/W-1
/// "conservative-when-uncertain").
fn appears_inside_brackets(scalar_name: &str, source: &str) -> bool {
    let bytes = source.as_bytes();
    let mut depth: i32 = 0;
    let mut start: Option<usize> = None;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'[' => {
                if depth == 0 {
                    start = Some(i + 1);
                }
                depth += 1;
            }
            b']' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(s) = start.take() {
                        if word_contains(&source[s..i], scalar_name) {
                            return true;
                        }
                    }
                }
            }
            _ => {}
        }
    }
    false
}

fn word_contains(haystack: &str, needle: &str) -> bool {
    haystack.split(|c: char| !c.is_alphanumeric() && c != '_').any(|w| w == needle)
}

fn is_eligible_for_violate(scalar_name: &str, source: &str) -> bool {
    !is_conventional_element_count(scalar_name) && !appears_inside_brackets(scalar_name, source)
}

/// Select the `--violate` target: the first lowerable, single-scalar-vs-literal
/// `Pre` check whose scalar passes `is_eligible_for_violate` (§3.3).
pub fn select_violate_target<'a>(pre_checks: &'a [DebugCheck], source: &str) -> Option<&'a DebugCheck> {
    pre_checks.iter().find(|c| matches!(normalize(c), Some((name, _, _)) if is_eligible_for_violate(name, source)))
}

// ══════════════════════════════════════════════════════════════════════════════
// §3.2(b) — the TOTAL 4-quadrant precondition discriminator (pure, unit-testable)
// ══════════════════════════════════════════════════════════════════════════════

/// One run's classification (§3.2(b) + r4 severity ordering). Deliberately does
/// NOT include `Unsatisfiable`/`Skipped` — those are whole-request, not per-run,
/// outcomes (computed before/around the run loop, never per-run).
#[derive(Debug, Clone, PartialEq)]
pub enum RunOutcome {
    Pass,
    Fail { reason_kind: &'static str, detail: String },
    Error { reason_kind: &'static str, detail: String },
    UnconstrainedPreconditionFired { detail: String },
}

/// Severity rank: LOWER is MORE severe (`ERROR > FAIL > UNCONSTRAINED... > PASS`,
/// r4 orchestrator resolution). Used both to pick a single run's outcome (when several
/// signals fire in the same run) and to fold multiple runs into one report verdict.
fn severity_rank(o: &RunOutcome) -> u8 {
    match o {
        RunOutcome::Error { .. } => 0,
        RunOutcome::Fail { .. } => 1,
        RunOutcome::UnconstrainedPreconditionFired { .. } => 2,
        RunOutcome::Pass => 3,
    }
}

/// The TOTAL 4-quadrant discriminator (§3.2(b)/r3-fix-4) with the r3/C-2new
/// deferred-condition exemption applied FIRST, and the r4 verdict-precedence rule
/// (a deferred fire is demoted to a note whenever the rest of the run is not clean).
///
/// - `host_constrained_satisfied`: did EVERY CONSTRAINED `Pre` predicate evaluate
///   true on the host, over the SAME bit pattern sent to the GPU (r3/C-1new f32)?
/// - `gpu_pre_bits` / `gpu_post_bits`: the raw `flag_words` (`gpu_post_bits` is
///   `None` when no postcondition oracle ran at all — legs a+c only).
/// - `constrained_bits` / `deferred_bits`: the `Constraints` bit partition.
pub fn classify_run(
    host_constrained_satisfied: bool,
    gpu_pre_bits: u32,
    gpu_post_bits: Option<u32>,
    constrained_bits: &BTreeSet<u32>,
    deferred_bits: &BTreeSet<u32>,
) -> RunOutcome {
    let constrained_mask: u32 = constrained_bits.iter().fold(0u32, |acc, b| acc | (1u32 << b));
    let deferred_mask: u32 = deferred_bits.iter().fold(0u32, |acc, b| acc | (1u32 << b));
    let gpu_constrained_fired = (gpu_pre_bits & constrained_mask) != 0;
    let gpu_deferred_fired = (gpu_pre_bits & deferred_mask) != 0;

    let quadrant: Option<RunOutcome> = match (host_constrained_satisfied, gpu_constrained_fired) {
        (true, false) => None,
        (true, true) => Some(RunOutcome::Fail {
            reason_kind: "precondition_check_miscompiled",
            detail: "host f32 re-check says SATISFIED but the GPU constrained precondition bit fired \
                     (CPU/GPU disagree -> an M3.17 lowering miscompile, not a satisfier bug)"
                .to_string(),
        }),
        (false, true) => Some(RunOutcome::Error {
            reason_kind: "satisfier_unsound",
            detail: "the satisfier emitted a value its own constrained predicate rejects, and the GPU agreed by firing".to_string(),
        }),
        (false, false) => Some(RunOutcome::Error {
            reason_kind: "satisfier_unsound_or_check_inert",
            detail: "host f32 re-check says UNSATISFIED but no GPU bit fired: either the satisfier is unsound \
                     or the M3.17 check is missing/inert/miscompiled-to-never-fire"
                .to_string(),
        }),
    };

    let post_fail: Option<RunOutcome> = match gpu_post_bits {
        Some(bits) if bits != 0 => Some(RunOutcome::Fail {
            reason_kind: "postcondition_violation",
            detail: format!("postcondition flag word fired: {bits:#010x}"),
        }),
        _ => None,
    };

    let deferred_fire: Option<RunOutcome> = if gpu_deferred_fired {
        Some(RunOutcome::UnconstrainedPreconditionFired {
            detail: "a DEFERRED (two-scalar relational) precondition bit fired; the satisfier never constrained it \
                     (expected, not a defect)"
                .to_string(),
        })
    } else {
        None
    };

    let mut candidates: Vec<RunOutcome> = Vec::new();
    if let Some(q) = quadrant {
        candidates.push(q);
    }
    if let Some(p) = post_fail {
        candidates.push(p);
    }
    if let Some(d) = deferred_fire {
        candidates.push(d);
    }
    candidates.sort_by_key(severity_rank);
    candidates.into_iter().next().unwrap_or(RunOutcome::Pass)
}

/// Host `f32`/`i32`/`u32` re-evaluation of ONE predicate over the byte-identical
/// value that was std430-encoded and dispatched (r3/C-1new — never `f64`). The
/// SAME evaluator the discriminator's `host_constrained_satisfied` input is built
/// from (AT-2891's "differential re-check ... the SAME one the discriminator reuses").
pub fn eval_check_host(c: &DebugCheck, sampled: &BTreeMap<String, Vec<u8>>) -> bool {
    let Some((name, op, k)) = normalize(c) else { return true };
    let Some(bytes) = sampled.get(name) else { return true };
    match c.ty {
        ScalarTy::F32 => {
            let Ok(raw) = <[u8; 4]>::try_from(&bytes[..bytes.len().min(4)]) else { return true };
            eval_cmp_f32(f32::from_le_bytes(raw), op, k as f32)
        }
        ScalarTy::U32 => {
            let Ok(raw) = <[u8; 4]>::try_from(&bytes[..bytes.len().min(4)]) else { return true };
            eval_cmp_i128(u32::from_le_bytes(raw) as i128, op, k as i128)
        }
        ScalarTy::I32 => {
            let Ok(raw) = <[u8; 4]>::try_from(&bytes[..bytes.len().min(4)]) else { return true };
            eval_cmp_i128(i32::from_le_bytes(raw) as i128, op, k as i128)
        }
        _ => true,
    }
}

fn eval_cmp_f32(v: f32, op: DebugCompareOp, k: f32) -> bool {
    match op {
        DebugCompareOp::Gt => v > k,
        DebugCompareOp::Ge => v >= k,
        DebugCompareOp::Lt => v < k,
        DebugCompareOp::Le => v <= k,
        DebugCompareOp::Eq => v == k,
        DebugCompareOp::Ne => v != k,
    }
}

fn eval_cmp_i128(v: i128, op: DebugCompareOp, k: i128) -> bool {
    match op {
        DebugCompareOp::Gt => v > k,
        DebugCompareOp::Ge => v >= k,
        DebugCompareOp::Lt => v < k,
        DebugCompareOp::Le => v <= k,
        DebugCompareOp::Eq => v == k,
        DebugCompareOp::Ne => v != k,
    }
}

/// `host_constrained_satisfied`: every CONSTRAINED `Pre` predicate must hold.
pub fn host_recheck_satisfied(pre_checks: &[DebugCheck], constraints: &Constraints, sampled: &BTreeMap<String, Vec<u8>>) -> bool {
    pre_checks.iter().filter(|c| constraints.constrained_pre_bits.contains(&c.bit)).all(|c| eval_check_host(c, sampled))
}

// ══════════════════════════════════════════════════════════════════════════════
// Request / report types (§3.5, §3.6)
// ══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FuzzVerdict {
    Pass,
    Fail,
    Unsatisfiable,
    Skipped,
    UnconstrainedPreconditionFired,
    Error,
}

/// `exit_code_for_fuzz` (§3.6): `PASS|SKIPPED|UNCONSTRAINED_PRECONDITION_FIRED -> 0`,
/// `FAIL -> 1`, `ERROR -> 2`, `UNSATISFIABLE -> 3` (r2: distinct from FAIL).
pub fn exit_code_for_fuzz(v: FuzzVerdict) -> i32 {
    match v {
        FuzzVerdict::Pass | FuzzVerdict::Skipped | FuzzVerdict::UnconstrainedPreconditionFired => 0,
        FuzzVerdict::Fail => 1,
        FuzzVerdict::Error => 2,
        FuzzVerdict::Unsatisfiable => 3,
    }
}

#[derive(Debug, Clone)]
pub struct FuzzRequest {
    pub source: String,
    pub runs: u32,
    pub seed: u64,
    /// `None` = auto (`--debug-oracle` ON only when the kernel has >=1 lowered
    /// `Post` check, §3.2(b)).
    pub debug_oracle: Option<bool>,
    pub violate: bool,
    pub buffer_sizes: Vec<usize>,
    pub workgroups: Option<[u32; 3]>,
    pub scalar_window: i64,
    /// Shared `@strategy` hole assignment (`--strategy-value name=value`, mirrors
    /// `rewrite_verify::VerifyRequest::assignments`). Applied via
    /// `crate::substitute_strategy_holes` BEFORE the front-end runs (same
    /// source-text-substitution technique `compile_source_with_assignments` uses).
    pub strategy_assignments: BTreeMap<String, i64>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ConstraintReport {
    pub scalar: String,
    pub ty: String,
    pub interval: String,
    pub from: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Counterexample {
    pub run: u32,
    pub scalars: serde_json::Value,
    pub reason: String,
    pub detail: serde_json::Value,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct FuzzReason {
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<serde_json::Value>,
}

/// `verdict` serializes FIRST (declared first — same technique as `VerifyReport`).
#[derive(Debug, Clone, serde::Serialize)]
pub struct FuzzReport {
    pub verdict: FuzzVerdict,
    pub milestone: &'static str,
    pub kernel: String,
    pub seed: u64,
    pub runs_requested: u32,
    pub runs_executed: u32,
    pub debug_oracle: bool,
    pub violate_leg: bool,
    pub device: Option<String>,
    pub constraints: Vec<ConstraintReport>,
    pub counterexample: Option<Counterexample>,
    pub reason: Option<FuzzReason>,
    pub host_recheck: Option<serde_json::Value>,
    pub notes: Vec<String>,
}

fn constraint_reports(constraints: &Constraints, window: i64) -> Vec<ConstraintReport> {
    constraints
        .scalars
        .iter()
        .map(|(name, (ty, dom))| {
            let interval = match dom {
                ScalarDomain::Int(d) => {
                    let (lo, hi) = effective_int_window(*ty, d, window);
                    let lo_disp = d.pinned.unwrap_or(lo);
                    let hi_disp = d.pinned.unwrap_or(hi);
                    format!("[{lo_disp}, {hi_disp}]")
                }
                ScalarDomain::Float(d) => {
                    let (win_lo, win_hi) = effective_float_window(d, window);
                    let lo_disp: f64 = d.pinned.map(f64::from).unwrap_or(if d.lo.is_finite() { d.lo as f64 } else { win_lo as f64 });
                    let hi_disp: f64 = d.pinned.map(f64::from).unwrap_or(if d.hi.is_finite() { d.hi as f64 } else { win_hi as f64 });
                    let lo_br = if d.pinned.is_none() && d.lo_open && d.lo.is_finite() { '(' } else { '[' };
                    let hi_br = if d.pinned.is_none() && d.hi_open && d.hi.is_finite() { ')' } else { ']' };
                    format!("{lo_br}{lo_disp}, {hi_disp}{hi_br}")
                }
            };
            ConstraintReport { scalar: name.clone(), ty: ty.display_name().to_string(), interval, from: constraints.from.get(name).cloned().unwrap_or_default() }
        })
        .collect()
}

fn decode_scalar_json(ty: ScalarTy, bytes: &[u8]) -> serde_json::Value {
    match ty {
        ScalarTy::U32 => <[u8; 4]>::try_from(bytes).map(|b| serde_json::json!(u32::from_le_bytes(b))).unwrap_or(serde_json::Value::Null),
        ScalarTy::I32 => <[u8; 4]>::try_from(bytes).map(|b| serde_json::json!(i32::from_le_bytes(b))).unwrap_or(serde_json::Value::Null),
        ScalarTy::F32 => <[u8; 4]>::try_from(bytes).map(|b| serde_json::json!(f32::from_le_bytes(b))).unwrap_or(serde_json::Value::Null),
        ScalarTy::U16 => <[u8; 2]>::try_from(bytes).map(|b| serde_json::json!(u16::from_le_bytes(b))).unwrap_or(serde_json::Value::Null),
        ScalarTy::I16 => <[u8; 2]>::try_from(bytes).map(|b| serde_json::json!(i16::from_le_bytes(b))).unwrap_or(serde_json::Value::Null),
        ScalarTy::U64 => <[u8; 8]>::try_from(bytes).map(|b| serde_json::json!(u64::from_le_bytes(b))).unwrap_or(serde_json::Value::Null),
        ScalarTy::I64 => <[u8; 8]>::try_from(bytes).map(|b| serde_json::json!(i64::from_le_bytes(b))).unwrap_or(serde_json::Value::Null),
        ScalarTy::F64 => <[u8; 8]>::try_from(bytes).map(|b| serde_json::json!(f64::from_le_bytes(b))).unwrap_or(serde_json::Value::Null),
        ScalarTy::U8 => serde_json::json!(bytes.first().copied().unwrap_or(0)),
        ScalarTy::I8 => serde_json::json!(bytes.first().copied().unwrap_or(0) as i8),
        ScalarTy::F16 => <[u8; 2]>::try_from(bytes)
            .map(|b| serde_json::json!(half::f16::from_le_bytes(b).to_f32()))
            .unwrap_or(serde_json::Value::Null),
        ScalarTy::Bool => serde_json::json!(bytes.first().copied().unwrap_or(0) != 0),
    }
}

fn sampled_to_json(kernel_params: &[KernelParam], sampled: &BTreeMap<String, Vec<u8>>) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for p in kernel_params {
        let ParamTy::Scalar(ty) = p.ty else { continue };
        if let Some(bytes) = sampled.get(&p.name) {
            map.insert(p.name.clone(), decode_scalar_json(ty, bytes));
        }
    }
    serde_json::Value::Object(map)
}

// ══════════════════════════════════════════════════════════════════════════════
// Orchestration (§3.1, §3.2, §3.3)
// ══════════════════════════════════════════════════════════════════════════════

fn empty_report(req: &FuzzRequest, kernel_name: &str, verdict: FuzzVerdict, kind: &str, detail: Option<serde_json::Value>, constraints: Vec<ConstraintReport>) -> FuzzReport {
    FuzzReport {
        verdict,
        milestone: MILESTONE,
        kernel: kernel_name.to_string(),
        seed: req.seed,
        runs_requested: req.runs,
        runs_executed: 0,
        debug_oracle: req.debug_oracle.unwrap_or(false),
        violate_leg: false,
        device: None,
        constraints,
        counterexample: None,
        reason: Some(FuzzReason { kind: kind.to_string(), detail }),
        host_recheck: None,
        notes: Vec::new(),
    }
}

fn classify_dispatch_err(e: &DispatchError) -> Option<(&'static str, String)> {
    match e {
        DispatchError::CoopMatUnsupported { reason, .. } => Some(("coopmat_unsupported", reason.clone())),
        DispatchError::DeviceFeatureUnsupported { feature, kernel } => Some(("device_feature_unsupported", format!("{feature}/{kernel}"))),
        _ => None,
    }
}

/// Run one dispatch (debug or release path) and return only the outputs
/// (used by the determinism twin-dispatch, which does not need flag words).
#[allow(clippy::too_many_arguments)]
fn run_once(vk: &VulkanContext, debug_oracle: bool, words: &[u32], meta: &KernelMetadata, workgroups: [u32; 3], input_refs: &[&[u8]], output_sizes: &[usize], pc: &[u8]) -> Result<Vec<Vec<u8>>, DispatchError> {
    if debug_oracle {
        vk.dispatch_debug_checked(words, meta, (workgroups[0], workgroups[1], workgroups[2]), input_refs, output_sizes, pc).map(|o| o.outputs)
    } else {
        vk.dispatch(DispatchRequest { spirv: words, binding_plan: &meta.binding_plan, workgroups, inputs: input_refs, output_sizes, push_constants: pc, entry_point: &meta.entry_point })
    }
}

fn record_worst(worst: &mut Option<(u32, RunOutcome, serde_json::Value)>, run: u32, outcome: RunOutcome, scalars_json: serde_json::Value) {
    let replace = match worst {
        None => true,
        Some((_, cur, _)) => severity_rank(&outcome) < severity_rank(cur),
    };
    if replace {
        *worst = Some((run, outcome, scalars_json));
    }
}

/// The orchestration core (§3.1-§3.3). `vk == None` -> CPU-only gates (front-end +
/// satisfier) run, then `SKIPPED`/`gpu_unavailable`. `req.runs == 0` is a CLI-level
/// usage error (checked by the caller, §10) — `fuzz_kernel` assumes `runs >= 1`.
pub fn fuzz_kernel(req: &FuzzRequest, vk: Option<&VulkanContext>) -> FuzzReport {
    let substituted_source: String = if req.strategy_assignments.is_empty() {
        req.source.clone()
    } else {
        crate::substitute_strategy_holes(&req.source, &req.strategy_assignments)
    };
    // Re-borrow `req.source` for the rest of this function as the (possibly)
    // substituted source, without perturbing the caller-visible `FuzzRequest`.
    let req = &FuzzRequest { source: substituted_source, ..req.clone() };

    let (ast, lex_errs, parse_errs) = axc_parser::parse(&req.source);
    let (hir, hir_errs, _warns) = lower_module(&ast);
    if !lex_errs.is_empty() || !parse_errs.is_empty() || !hir_errs.is_empty() {
        return empty_report(
            req,
            "",
            FuzzVerdict::Error,
            "compile_failed",
            Some(serde_json::json!({"lex_errors": lex_errs.len(), "parse_errors": parse_errs.len(), "hir_errors": hir_errs.len()})),
            Vec::new(),
        );
    }
    let Some(kernel) = hir.kernels.into_iter().next() else {
        return empty_report(req, "", FuzzVerdict::Error, "compile_failed", Some(serde_json::json!({"detail": "no kernel declared"})), Vec::new());
    };
    let kernel_name = kernel.name.clone();

    let pre_checks: Vec<DebugCheck> = kernel.annotations.debug_checks.iter().filter(|c| c.kind == DebugCheckKind::Pre).cloned().collect();
    let has_post = kernel.annotations.debug_checks.iter().any(|c| c.kind == DebugCheckKind::Post);

    let constraints = match solve(&pre_checks) {
        Ok(c) => c,
        Err(unsat) => {
            return empty_report(req, &kernel_name, FuzzVerdict::Unsatisfiable, "unsatisfiable", Some(serde_json::json!({"detail": unsat.reason})), Vec::new());
        }
    };
    let creports = constraint_reports(&constraints, req.scalar_window);

    let Some(vk) = vk else {
        return empty_report(req, &kernel_name, FuzzVerdict::Skipped, "gpu_unavailable", None, creports);
    };

    let debug_oracle = req.debug_oracle.unwrap_or(has_post);

    if req.violate && select_violate_target(&pre_checks, &req.source).is_none() {
        return empty_report(req, &kernel_name, FuzzVerdict::Error, "usage", Some(serde_json::json!({"detail": "nothing safely violable: no eligible (non-size-coupled) precondition scalar"})), creports);
    }

    let (spirv_bytes, meta) = if debug_oracle {
        match crate::compile_source_with_meta_debug(&req.source, true) {
            Ok(v) => v,
            Err(e) => return empty_report(req, &kernel_name, FuzzVerdict::Error, "compile_failed", Some(serde_json::json!({"detail": e.to_string()})), creports),
        }
    } else {
        match crate::compile_source_with_meta(&req.source) {
            Ok(v) => v,
            Err(e) => return empty_report(req, &kernel_name, FuzzVerdict::Error, "compile_failed", Some(serde_json::json!({"detail": e.to_string()})), creports),
        }
    };
    let words = bytes_to_words(&spirv_bytes);

    let needs_workgroups = meta.coopmat.is_some() || meta.shared_memory_bytes > 0;
    if needs_workgroups && req.workgroups.is_none() {
        return empty_report(req, &kernel_name, FuzzVerdict::Error, "usage", Some(serde_json::json!({"detail": "kernel uses coopmat or shared memory but no explicit --workgroups was supplied"})), creports);
    }
    if meta.binding_plan.buffers.len() != req.buffer_sizes.len() {
        return empty_report(
            req,
            &kernel_name,
            FuzzVerdict::Error,
            "usage",
            Some(serde_json::json!({"detail": format!("buffer_sizes.len()=={} but kernel has {} buffer bindings", req.buffer_sizes.len(), meta.binding_plan.buffers.len())})),
            creports,
        );
    }
    let workgroups = derive_workgroups(&req.buffer_sizes, meta.workgroup_size, req.workgroups);

    let device_name = Some(vk.physical_device_name().to_string());
    let mut notes: Vec<String> = Vec::new();
    if debug_oracle && !has_post {
        notes.push("postcondition_oracle_inert: no lowered Post check exercised (debug oracle enabled but the kernel declares zero lowered postconditions)".to_string());
    }
    notes.push(format!(
        "legs {} certify the {} artifact",
        if debug_oracle { "(a)/(b)/(c)" } else { "(a)/(c)" },
        if debug_oracle { "DEBUG" } else { "RELEASE" }
    ));

    let mut worst: Option<(u32, RunOutcome, serde_json::Value)> = None;

    // ── determinism leg (c): run-0 inputs, dispatched TWICE ─────────────────
    let sampled0 = sample_all(&constraints, &kernel.params, req.scalar_window, req.seed, 0);
    let pc0 = encode_push_constants(&meta.binding_plan.scalars, &sampled0, meta.binding_plan.push_constant_total_bytes);
    let buf0 = seeded_inputs(&req.buffer_sizes, req.seed ^ 0x9E37_79B9);
    let input_refs0: Vec<&[u8]> = buf0.iter().map(Vec::as_slice).collect();

    match run_once(vk, debug_oracle, &words, &meta, workgroups, &input_refs0, &req.buffer_sizes, &pc0) {
        Ok(run1) => match run_once(vk, debug_oracle, &words, &meta, workgroups, &input_refs0, &req.buffer_sizes, &pc0) {
            Ok(run2) => {
                if crate::rewrite_verify::find_nondeterminism(&run1, &run2, &req.buffer_sizes).is_some() {
                    record_worst(
                        &mut worst,
                        0,
                        RunOutcome::Fail { reason_kind: "nondeterministic", detail: "twin-dispatch of run 0 disagreed byte-for-byte".to_string() },
                        sampled_to_json(&kernel.params, &sampled0),
                    );
                } else {
                    notes.push("determinism leg (c): run 0 twin-dispatch is byte-identical".to_string());
                }
            }
            Err(e) => {
                if let Some((kind, detail)) = classify_dispatch_err(&e) {
                    return empty_report(req, &kernel_name, FuzzVerdict::Skipped, kind, Some(serde_json::json!({"detail": detail})), creports);
                }
                record_worst(&mut worst, 0, RunOutcome::Fail { reason_kind: "device_fault", detail: e.to_string() }, sampled_to_json(&kernel.params, &sampled0));
            }
        },
        Err(e) => {
            if let Some((kind, detail)) = classify_dispatch_err(&e) {
                return empty_report(req, &kernel_name, FuzzVerdict::Skipped, kind, Some(serde_json::json!({"detail": detail})), creports);
            }
            record_worst(&mut worst, 0, RunOutcome::Fail { reason_kind: "device_fault", detail: e.to_string() }, sampled_to_json(&kernel.params, &sampled0));
        }
    }

    // ── main counterexample loop: legs (a)+(b) ──────────────────────────────
    let mut runs_executed: u32 = 0;
    let mut last_host_recheck: Option<serde_json::Value> = None;
    'runs: for r in 0..req.runs {
        runs_executed = r + 1;
        let sampled = sample_all(&constraints, &kernel.params, req.scalar_window, req.seed, r);
        let pc = encode_push_constants(&meta.binding_plan.scalars, &sampled, meta.binding_plan.push_constant_total_bytes);
        let buf_inputs = seeded_inputs(&req.buffer_sizes, req.seed ^ (r as u64) ^ 0x9E37_79B9);
        let input_refs: Vec<&[u8]> = buf_inputs.iter().map(Vec::as_slice).collect();

        if debug_oracle {
            match vk.dispatch_debug_checked(&words, &meta, (workgroups[0], workgroups[1], workgroups[2]), &input_refs, &req.buffer_sizes, &pc) {
                Ok(outcome) => {
                    let host_sat = host_recheck_satisfied(&pre_checks, &constraints, &sampled);
                    let run_outcome = classify_run(host_sat, outcome.flag_words[0], Some(outcome.flag_words[1]), &constraints.constrained_pre_bits, &constraints.deferred_pre_bits);
                    if r == 0 {
                        last_host_recheck = Some(serde_json::json!({
                            "run": r,
                            "gpu_flag_words0": outcome.flag_words[0],
                            "gpu_flag_words1": outcome.flag_words[1],
                            "host_constrained_satisfied": host_sat,
                        }));
                    }
                    record_worst(&mut worst, r, run_outcome, sampled_to_json(&kernel.params, &sampled));
                }
                Err(e) => {
                    if let Some((kind, detail)) = classify_dispatch_err(&e) {
                        return empty_report(req, &kernel_name, FuzzVerdict::Skipped, kind, Some(serde_json::json!({"detail": detail})), creports);
                    }
                    record_worst(&mut worst, r, RunOutcome::Fail { reason_kind: "device_fault", detail: e.to_string() }, sampled_to_json(&kernel.params, &sampled));
                    break 'runs;
                }
            }
        } else {
            match vk.dispatch(DispatchRequest { spirv: &words, binding_plan: &meta.binding_plan, workgroups, inputs: &input_refs, output_sizes: &req.buffer_sizes, push_constants: &pc, entry_point: &meta.entry_point }) {
                Ok(_) => record_worst(&mut worst, r, RunOutcome::Pass, sampled_to_json(&kernel.params, &sampled)),
                Err(e) => {
                    if let Some((kind, detail)) = classify_dispatch_err(&e) {
                        return empty_report(req, &kernel_name, FuzzVerdict::Skipped, kind, Some(serde_json::json!({"detail": detail})), creports);
                    }
                    record_worst(&mut worst, r, RunOutcome::Fail { reason_kind: "device_fault", detail: e.to_string() }, sampled_to_json(&kernel.params, &sampled));
                    break 'runs;
                }
            }
        }
    }

    // ── §3.3 optional --violate leg ──────────────────────────────────────────
    let violate_ran = req.violate;
    if req.violate {
        if let Some(target) = select_violate_target(&pre_checks, &req.source) {
            let target_name = normalize(target).expect("select_violate_target only returns normalizable checks").0.to_string();
            let mut sampled = sample_all(&constraints, &kernel.params, req.scalar_window, req.seed, req.runs);
            match negate_one(target, &target_name) {
                Ok(violating_bytes) => {
                    sampled.insert(target_name.clone(), violating_bytes);
                    let pc = encode_push_constants(&meta.binding_plan.scalars, &sampled, meta.binding_plan.push_constant_total_bytes);
                    let buf_inputs = seeded_inputs(&req.buffer_sizes, req.seed ^ 0xC0FF_EE00);
                    let input_refs: Vec<&[u8]> = buf_inputs.iter().map(Vec::as_slice).collect();

                    let violate_result = if debug_oracle {
                        vk.dispatch_debug_checked(&words, &meta, (workgroups[0], workgroups[1], workgroups[2]), &input_refs, &req.buffer_sizes, &pc)
                    } else {
                        match crate::compile_source_with_meta_debug(&req.source, true) {
                            Ok((vb, vm)) => {
                                let vw = bytes_to_words(&vb);
                                vk.dispatch_debug_checked(&vw, &vm, (workgroups[0], workgroups[1], workgroups[2]), &input_refs, &req.buffer_sizes, &pc)
                            }
                            Err(e) => return empty_report(req, &kernel_name, FuzzVerdict::Error, "compile_failed", Some(serde_json::json!({"detail": e.to_string()})), creports),
                        }
                    };

                    match violate_result {
                        Ok(outcome) => {
                            let fired = (outcome.flag_words[0] & (1u32 << target.bit)) != 0;
                            if fired {
                                notes.push(format!("violate leg: `{}` fired as expected (flag_words[0] bit {})", target.text, target.bit));
                            } else {
                                record_worst(
                                    &mut worst,
                                    req.runs,
                                    RunOutcome::Fail {
                                        reason_kind: "precondition_check_did_not_fire",
                                        detail: format!("expected bit {} for `{}` to fire; flag_words[0]={:#010x}", target.bit, target.text, outcome.flag_words[0]),
                                    },
                                    sampled_to_json(&kernel.params, &sampled),
                                );
                            }
                        }
                        Err(e) => {
                            if let Some((kind, detail)) = classify_dispatch_err(&e) {
                                return empty_report(req, &kernel_name, FuzzVerdict::Skipped, kind, Some(serde_json::json!({"detail": detail})), creports);
                            }
                            // §3.3 documented residual: a device fault on an intentionally-violating
                            // input is an ACCEPTED (not FAIL) outcome — noted, not escalated.
                            notes.push(format!("violate leg: device fault on intentionally-violating input (accepted residual, not a failure): {e}"));
                        }
                    }
                }
                Err(e) => return empty_report(req, &kernel_name, FuzzVerdict::Error, "usage", Some(serde_json::json!({"detail": e})), creports),
            }
        } else {
            return empty_report(req, &kernel_name, FuzzVerdict::Error, "usage", Some(serde_json::json!({"detail": "nothing safely violable: no eligible (non-size-coupled) precondition scalar"})), creports);
        }
    }

    let (verdict, reason, counterexample) = match &worst {
        None | Some((_, RunOutcome::Pass, _)) => (FuzzVerdict::Pass, None, None),
        Some((run, RunOutcome::Fail { reason_kind, detail }, scalars)) => (
            FuzzVerdict::Fail,
            Some(FuzzReason { kind: reason_kind.to_string(), detail: Some(serde_json::json!({"detail": detail})) }),
            Some(Counterexample { run: *run, scalars: scalars.clone(), reason: reason_kind.to_string(), detail: serde_json::json!({"detail": detail}) }),
        ),
        Some((run, RunOutcome::Error { reason_kind, detail }, scalars)) => (
            FuzzVerdict::Error,
            Some(FuzzReason { kind: reason_kind.to_string(), detail: Some(serde_json::json!({"detail": detail})) }),
            Some(Counterexample { run: *run, scalars: scalars.clone(), reason: reason_kind.to_string(), detail: serde_json::json!({"detail": detail}) }),
        ),
        Some((_run, RunOutcome::UnconstrainedPreconditionFired { detail }, _scalars)) => {
            notes.push(format!("UNCONSTRAINED_PRECONDITION_FIRED: {detail}"));
            (FuzzVerdict::UnconstrainedPreconditionFired, Some(FuzzReason { kind: "unconstrained_precondition_fired".to_string(), detail: Some(serde_json::json!({"detail": detail})) }), None)
        }
    };

    FuzzReport {
        verdict,
        milestone: MILESTONE,
        kernel: kernel_name,
        seed: req.seed,
        runs_requested: req.runs,
        runs_executed,
        debug_oracle,
        violate_leg: violate_ran,
        device: device_name,
        constraints: creports,
        counterexample,
        reason,
        host_recheck: last_host_recheck,
        notes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axc_lexer::Span;

    fn scalar_ref(name: &str) -> DebugOperand {
        DebugOperand::ScalarRef { name: name.to_string(), position: 0 }
    }

    fn check(kind: DebugCheckKind, op: DebugCompareOp, ty: ScalarTy, lhs: DebugOperand, rhs: DebugOperand, bit: u32, text: &str) -> DebugCheck {
        DebugCheck { kind, op, ty, lhs, rhs, bit, text: text.to_string(), span: Span::default() }
    }

    fn pre_i32(op: DebugCompareOp, scalar: &str, lit: i64, bit: u32, text: &str) -> DebugCheck {
        check(DebugCheckKind::Pre, op, ScalarTy::U32, scalar_ref(scalar), DebugOperand::Lit(lit), bit, text)
    }

    // ── AT-2888: satisfier interval, integers ──────────────────────────────────

    #[test]
    fn at_2888_gt_alone_gives_one_to_umax() {
        let cs = solve(&[pre_i32(DebugCompareOp::Gt, "n", 0, 0, "gt(n, 0)")]).unwrap();
        let (ty, dom) = &cs.scalars["n"];
        assert_eq!(*ty, ScalarTy::U32);
        let ScalarDomain::Int(d) = dom else { panic!("expected Int domain") };
        assert_eq!((d.lo, d.hi), (1, u32::MAX as i128));
    }

    #[test]
    fn at_2888_ge_le_intersect() {
        let cs = solve(&[pre_i32(DebugCompareOp::Ge, "n", 4, 0, "ge(n,4)"), pre_i32(DebugCompareOp::Le, "n", 9, 1, "le(n,9)")]).unwrap();
        let ScalarDomain::Int(d) = &cs.scalars["n"].1 else { panic!() };
        assert_eq!((d.lo, d.hi), (4, 9));
    }

    #[test]
    fn at_2888_gt_lt_intersect() {
        let cs = solve(&[pre_i32(DebugCompareOp::Gt, "n", 0, 0, "gt(n,0)"), pre_i32(DebugCompareOp::Lt, "n", 10, 1, "lt(n,10)")]).unwrap();
        let ScalarDomain::Int(d) = &cs.scalars["n"].1 else { panic!() };
        assert_eq!((d.lo, d.hi), (1, 9));
    }

    #[test]
    fn at_2888_rhs_scalar_flip() {
        // lt(0, n) means 0 < n, i.e. n > 0.
        let c = check(DebugCheckKind::Pre, DebugCompareOp::Lt, ScalarTy::U32, DebugOperand::Lit(0), scalar_ref("n"), 0, "lt(0, n)");
        let cs = solve(&[c]).unwrap();
        let ScalarDomain::Int(d) = &cs.scalars["n"].1 else { panic!() };
        assert_eq!(d.lo, 1);
        assert_eq!(d.hi, u32::MAX as i128);
    }

    // ── AT-2889: satisfier unsat detection ─────────────────────────────────────

    #[test]
    fn at_2889_disjoint_interval_unsat() {
        let e = solve(&[pre_i32(DebugCompareOp::Gt, "n", 10, 0, "gt(n,10)"), pre_i32(DebugCompareOp::Lt, "n", 5, 1, "lt(n,5)")]).unwrap_err();
        assert!(!e.reason.is_empty());
    }

    #[test]
    fn at_2889_eq_ne_conflict_unsat() {
        solve(&[pre_i32(DebugCompareOp::Eq, "n", 3, 0, "eq(n,3)"), pre_i32(DebugCompareOp::Ne, "n", 3, 1, "ne(n,3)")]).unwrap_err();
    }

    #[test]
    fn at_2889_eq_eq_conflict_unsat() {
        solve(&[pre_i32(DebugCompareOp::Eq, "n", 3, 0, "eq(n,3)"), pre_i32(DebugCompareOp::Eq, "n", 4, 1, "eq(n,4)")]).unwrap_err();
    }

    // ── AT-2890: satisfier float + exclusion ───────────────────────────────────

    #[test]
    fn at_2890_float_open_lower_sampled_positive() {
        let c = check(DebugCheckKind::Pre, DebugCompareOp::Gt, ScalarTy::F32, scalar_ref("alpha"), DebugOperand::Lit(0), 0, "gt(alpha, 0)");
        let cs = solve(&[c]).unwrap();
        let params = vec![KernelParam { name: "alpha".to_string(), ty: ParamTy::Scalar(ScalarTy::F32), position: 0, span: Span::default() }];
        for run in 0..64 {
            let sampled = sample_all(&cs, &params, DEFAULT_SCALAR_WINDOW, 7, run);
            let bytes = &sampled["alpha"];
            let v = f32::from_le_bytes(bytes[..4].try_into().unwrap());
            assert!(v > 0.0, "sampled alpha must be > 0; got {v} at run {run}");
            assert!(v.is_finite());
        }
    }

    #[test]
    fn at_2890_ne_exclusion_never_sampled() {
        let cs = solve(&[pre_i32(DebugCompareOp::Ne, "n", 5, 0, "ne(n,5)")]).unwrap();
        let params = vec![KernelParam { name: "n".to_string(), ty: ParamTy::Scalar(ScalarTy::U32), position: 0, span: Span::default() }];
        for run in 0..128 {
            let sampled = sample_all(&cs, &params, DEFAULT_SCALAR_WINDOW, 11, run);
            let v = u32::from_le_bytes(sampled["n"][..4].try_into().unwrap());
            assert_ne!(v, 5, "sampled n must never == 5 (ne exclusion) at run {run}");
        }
    }

    // ── AT-2891: satisfier seed determinism + TOTAL 4-quadrant discriminator ───

    #[test]
    fn at_2891_sample_seed_determinism() {
        let cs = solve(&[pre_i32(DebugCompareOp::Gt, "n", 0, 0, "gt(n,0)")]).unwrap();
        let params = vec![KernelParam { name: "n".to_string(), ty: ParamTy::Scalar(ScalarTy::U32), position: 3, span: Span::default() }];
        for run in 0..16 {
            let a = sample_all(&cs, &params, DEFAULT_SCALAR_WINDOW, 42, run);
            let b = sample_all(&cs, &params, DEFAULT_SCALAR_WINDOW, 42, run);
            assert_eq!(a, b, "same (seed,run) must be bit-identical at run {run}");
        }
        let mut any_diff = false;
        for run in 0..16 {
            let a = sample_all(&cs, &params, DEFAULT_SCALAR_WINDOW, 42, run);
            let b = sample_all(&cs, &params, DEFAULT_SCALAR_WINDOW, 43, run);
            if a != b {
                any_diff = true;
            }
        }
        assert!(any_diff, "seed=42 must differ from seed=43 for at least one run");
    }

    #[test]
    fn at_2891_sampled_values_satisfy_constraints_differentially() {
        let checks = vec![pre_i32(DebugCompareOp::Gt, "n", 0, 0, "gt(n,0)"), check(DebugCheckKind::Pre, DebugCompareOp::Gt, ScalarTy::F32, scalar_ref("alpha"), DebugOperand::Lit(0), 1, "gt(alpha,0)")];
        let cs = solve(&checks).unwrap();
        let params = vec![
            KernelParam { name: "n".to_string(), ty: ParamTy::Scalar(ScalarTy::U32), position: 0, span: Span::default() },
            KernelParam { name: "alpha".to_string(), ty: ParamTy::Scalar(ScalarTy::F32), position: 1, span: Span::default() },
        ];
        for run in 0..32 {
            let sampled = sample_all(&cs, &params, DEFAULT_SCALAR_WINDOW, 5, run);
            assert!(host_recheck_satisfied(&checks, &cs, &sampled), "run {run} sampled values must satisfy every constrained predicate: {sampled:?}");
        }
    }

    /// TOTAL 4-quadrant discriminator unit (r2/C-2 + r3/fix-4).
    #[test]
    fn at_2891_quadrant_i_host_sat_gpu_clean_passes() {
        let constrained: BTreeSet<u32> = [0].into_iter().collect();
        let deferred: BTreeSet<u32> = BTreeSet::new();
        let outcome = classify_run(true, 0, Some(0), &constrained, &deferred);
        assert_eq!(outcome, RunOutcome::Pass);
    }

    #[test]
    fn at_2891_quadrant_ii_host_sat_gpu_fired_is_miscompiled_fail() {
        let constrained: BTreeSet<u32> = [0].into_iter().collect();
        let deferred: BTreeSet<u32> = BTreeSet::new();
        let outcome = classify_run(true, 1, Some(0), &constrained, &deferred);
        match outcome {
            RunOutcome::Fail { reason_kind, .. } => assert_eq!(reason_kind, "precondition_check_miscompiled"),
            other => panic!("expected Fail(precondition_check_miscompiled); got {other:?}"),
        }
        assert_eq!(exit_code_for_fuzz(FuzzVerdict::Fail), 1);
    }

    #[test]
    fn at_2891_quadrant_iii_host_unsat_gpu_fired_is_satisfier_unsound_error() {
        let constrained: BTreeSet<u32> = [0].into_iter().collect();
        let deferred: BTreeSet<u32> = BTreeSet::new();
        let outcome = classify_run(false, 1, Some(0), &constrained, &deferred);
        match outcome {
            RunOutcome::Error { reason_kind, .. } => assert_eq!(reason_kind, "satisfier_unsound"),
            other => panic!("expected Error(satisfier_unsound); got {other:?}"),
        }
        assert_eq!(exit_code_for_fuzz(FuzzVerdict::Error), 2);
    }

    #[test]
    fn at_2891_quadrant_iv_host_unsat_gpu_clean_is_inert_error() {
        let constrained: BTreeSet<u32> = [0].into_iter().collect();
        let deferred: BTreeSet<u32> = BTreeSet::new();
        let outcome = classify_run(false, 0, Some(0), &constrained, &deferred);
        match outcome {
            RunOutcome::Error { reason_kind, .. } => assert_eq!(reason_kind, "satisfier_unsound_or_check_inert"),
            other => panic!("expected Error(satisfier_unsound_or_check_inert); got {other:?}"),
        }
    }

    #[test]
    fn at_2891_host_recheck_uses_std430_bytes_actually_sent_f32_not_f64() {
        // The host re-check must decode the SAME f32 bytes that were encoded —
        // never an f64 re-derivation. Encode 0.1f32 (imprecise in f64) and confirm
        // the re-check agrees exactly with the f32 round-trip, not a wider f64 compare.
        let c = check(DebugCheckKind::Pre, DebugCompareOp::Gt, ScalarTy::F32, scalar_ref("alpha"), DebugOperand::Lit(0), 0, "gt(alpha,0)");
        let mut sampled: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        sampled.insert("alpha".to_string(), 0.1f32.to_le_bytes().to_vec());
        assert!(eval_check_host(&c, &sampled));
        let bytes = &sampled["alpha"];
        assert_eq!(f32::from_le_bytes(bytes[..4].try_into().unwrap()), 0.1f32, "must decode the exact f32 bit pattern sent");
    }

    // ── AT-2892: relational defer + deferred-exemption verdict + composition ──

    #[test]
    fn at_2892_relational_two_scalar_defers_with_note() {
        let c = check(DebugCheckKind::Pre, DebugCompareOp::Gt, ScalarTy::U32, scalar_ref("a"), scalar_ref("b"), 0, "gt(a, b)");
        let cs = solve(&[c]).unwrap();
        assert!(cs.scalars.is_empty(), "relational predicates must not constrain either scalar");
        assert!(cs.notes.iter().any(|n| n.starts_with("relational_constraint_deferred")));
        assert!(cs.deferred_pre_bits.contains(&0));
        assert!(!cs.constrained_pre_bits.contains(&0));
    }

    #[test]
    fn at_2892_deferred_bit_fired_yields_unconstrained_precondition_fired() {
        let constrained: BTreeSet<u32> = BTreeSet::new();
        let deferred: BTreeSet<u32> = [0].into_iter().collect();
        let outcome = classify_run(true, 1 << 0, Some(0), &constrained, &deferred);
        match outcome {
            RunOutcome::UnconstrainedPreconditionFired { .. } => {}
            other => panic!("expected UnconstrainedPreconditionFired; got {other:?}"),
        }
        assert_eq!(exit_code_for_fuzz(FuzzVerdict::UnconstrainedPreconditionFired), 0);
    }

    /// r4/orchestrator composition unit: a deferred fire + a REAL postcondition
    /// violation in the SAME run -> the run verdict is FAIL (deferred fire demoted
    /// to a note), proving UNCONSTRAINED_PRECONDITION_FIRED cannot mask a real failure.
    #[test]
    fn at_2892_composition_deferred_fire_does_not_mask_postcondition_fail() {
        let constrained: BTreeSet<u32> = BTreeSet::new();
        let deferred: BTreeSet<u32> = [0].into_iter().collect();
        let outcome = classify_run(true, 1 << 0, Some(1), &constrained, &deferred);
        match outcome {
            RunOutcome::Fail { reason_kind, .. } => assert_eq!(reason_kind, "postcondition_violation"),
            other => panic!("expected Fail(postcondition_violation); the deferred fire must be demoted to a note, got {other:?}"),
        }
        assert_eq!(exit_code_for_fuzz(FuzzVerdict::Fail), 1);
    }

    // ── AT-2893: fuzz UNSATISFIABLE, LIBRARY level (CPU, vk=None) ──────────────
    //
    // This exercises `fuzz_kernel()` directly with a hand-supplied
    // `buffer_sizes`, i.e. it proves the satisfier/orchestration core resolves
    // UNSATISFIABLE correctly and BEFORE any dispatch attempt -- but it does
    // NOT exercise `main.rs::run_test`'s CLI-layer buffer-size handling (a
    // QA-caught gap: the CLI used to usage-error on missing `--buffer-sizes`
    // before ever reaching this code path, so this test alone could not catch
    // that regression). The CLI-level counterpart that drives the literal
    // spec command (`axc test --fuzz examples/fuzz_unsat.axc`, no flags) via a
    // real subprocess lives in `tests/fuzz_cli.rs`
    // (`at_2893_cli_fuzz_unsat_no_size_flags_end_to_end`).

    #[test]
    fn at_2893_fuzz_unsat_library_level_fuzz_kernel_no_gpu_needed() {
        let src = include_str!("../../../examples/fuzz_unsat.axc");
        let req = FuzzRequest { source: src.to_string(), runs: 8, seed: 0, debug_oracle: None, violate: false, buffer_sizes: vec![64], workgroups: None, scalar_window: DEFAULT_SCALAR_WINDOW, strategy_assignments: BTreeMap::new() };
        let report = fuzz_kernel(&req, None);
        assert_eq!(report.verdict, FuzzVerdict::Unsatisfiable);
        assert_eq!(exit_code_for_fuzz(report.verdict), 3);
        assert_eq!(report.runs_executed, 0, "no dispatch must be attempted");
    }

    // ── AT-2895: fuzz vk=None SKIPPED (satisfiable kernel) ──────────────────────

    #[test]
    fn at_2895_satisfiable_kernel_vk_none_is_skipped_with_constraints() {
        let src = include_str!("../../../examples/precondition_saxpy.axc");
        let req = FuzzRequest { source: src.to_string(), runs: 4, seed: 0, debug_oracle: None, violate: false, buffer_sizes: vec![256, 256], workgroups: None, scalar_window: DEFAULT_SCALAR_WINDOW, strategy_assignments: BTreeMap::new() };
        let report = fuzz_kernel(&req, None);
        assert_eq!(report.verdict, FuzzVerdict::Skipped);
        assert_eq!(exit_code_for_fuzz(report.verdict), 0);
        assert_eq!(report.reason.unwrap().kind, "gpu_unavailable");
        assert!(!report.constraints.is_empty(), "constraints must still be reported even when SKIPPED");
        assert!(report.constraints.iter().any(|c| c.scalar == "n"));
        assert!(report.constraints.iter().any(|c| c.scalar == "alpha"));
    }

    // ── satisfier interval-rendering sanity (matches §3.5 schema example) ──────

    #[test]
    fn constraint_report_renders_expected_intervals() {
        let src = include_str!("../../../examples/precondition_saxpy.axc");
        let req = FuzzRequest { source: src.to_string(), runs: 4, seed: 0, debug_oracle: None, violate: false, buffer_sizes: vec![256, 256], workgroups: None, scalar_window: 256, strategy_assignments: BTreeMap::new() };
        let report = fuzz_kernel(&req, None);
        let n = report.constraints.iter().find(|c| c.scalar == "n").unwrap();
        assert_eq!(n.interval, "[1, 257]");
        let alpha = report.constraints.iter().find(|c| c.scalar == "alpha").unwrap();
        assert_eq!(alpha.interval, "(0, 256]");
    }

    // ── negate_one / eligibility sanity ─────────────────────────────────────────

    #[test]
    fn negate_one_gt_alpha_zero_gives_zero() {
        let c = check(DebugCheckKind::Pre, DebugCompareOp::Gt, ScalarTy::F32, scalar_ref("alpha"), DebugOperand::Lit(0), 1, "gt(alpha, 0)");
        let bytes = negate_one(&c, "alpha").unwrap();
        let v = f32::from_le_bytes(bytes[..4].try_into().unwrap());
        assert_eq!(v, 0.0);
        assert!(v <= 0.0, "the negated value must NOT satisfy gt(alpha,0)");
    }

    #[test]
    fn violate_eligibility_alpha_yes_n_no() {
        let src = include_str!("../../../examples/precondition_saxpy.axc");
        assert!(is_eligible_for_violate("alpha", src), "alpha must be eligible (width-independent scalar)");
        assert!(!is_eligible_for_violate("n", src), "n must be EXCLUDED (conventional element-count name)");
    }

    #[test]
    fn select_violate_target_picks_alpha() {
        let src = include_str!("../../../examples/precondition_saxpy.axc");
        let (ast, _, _) = axc_parser::parse(src);
        let (hir, _, _) = lower_module(&ast);
        let kernel = hir.kernels.into_iter().next().unwrap();
        let pre: Vec<DebugCheck> = kernel.annotations.debug_checks.iter().filter(|c| c.kind == DebugCheckKind::Pre).cloned().collect();
        let target = select_violate_target(&pre, src).expect("alpha must be selected");
        let (name, _, _) = normalize(target).unwrap();
        assert_eq!(name, "alpha");
    }

    #[test]
    fn exit_code_mapping() {
        assert_eq!(exit_code_for_fuzz(FuzzVerdict::Pass), 0);
        assert_eq!(exit_code_for_fuzz(FuzzVerdict::Skipped), 0);
        assert_eq!(exit_code_for_fuzz(FuzzVerdict::UnconstrainedPreconditionFired), 0);
        assert_eq!(exit_code_for_fuzz(FuzzVerdict::Fail), 1);
        assert_eq!(exit_code_for_fuzz(FuzzVerdict::Error), 2);
        assert_eq!(exit_code_for_fuzz(FuzzVerdict::Unsatisfiable), 3);
    }

    #[test]
    fn verdict_first_key_and_screaming_snake_case() {
        let src = include_str!("../../../examples/fuzz_unsat.axc");
        let req = FuzzRequest { source: src.to_string(), runs: 1, seed: 0, debug_oracle: None, violate: false, buffer_sizes: vec![64], workgroups: None, scalar_window: DEFAULT_SCALAR_WINDOW, strategy_assignments: BTreeMap::new() };
        let report = fuzz_kernel(&req, None);
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.starts_with("{\"verdict\":"), "verdict must be first key; got {json}");
        assert!(json.contains("\"UNSATISFIABLE\""));
    }
}
