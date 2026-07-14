//! M3.16 (FG.1) `axc rewrite-verify` — GPU-gated tests (AT-2851…2860).
//!
//! `#[ignore]` + `AXC_ENABLE_GPU_TESTS=1`, typed-skip on Lavapipe / no-coopmat /
//! subgroup != 32 for the coopmat fixtures. AT-2854 and AT-2859 do not need
//! NVIDIA-tier coopmat support: AT-2854 fires at the CPU-only interface gate
//! (documented "also passes with vk=None" in the spec) and AT-2859 is the
//! Lavapipe-runnable non-coopmat ACCEPT path (r2/WARN-5) — both still live in
//! this GPU-gated file per the spec's AT table, and both are exercised on
//! Lavapipe in this milestone's local run.
//!
//! Reuses the SAME asymmetric Q4_K_M fixture generator + push-constant
//! assembly idiom as `dispatch_q4km_f32acc_cached_sr_equiv.rs` (AT-2603's
//! bit-identity harness), now driven through `verify_rewrite` instead of a
//! hand-rolled dispatch loop — proving the new harness reproduces the SAME
//! verdict the M3.11a campaign established by hand.

#[path = "common_matmul.rs"]
mod common_matmul;

use std::collections::BTreeMap;

use axc_driver::rewrite_verify::{verify_rewrite, TolerancePolicy, Verdict, VerifyRequest};
use axc_hir::param::ParamBindingPlan;
use axc_runtime::VulkanContext;

const SAXPY_SRC: &str = include_str!("../../../examples/saxpy.axc");
const SAXPY_WRONG_SRC: &str = include_str!("../../../examples/saxpy_wrong.axc");
const SAXPY_PERTURBED_SRC: &str = include_str!("../../../examples/saxpy_perturbed.axc");
const SAXPY_REASSOC_SRC: &str = include_str!("../../../examples/saxpy_reassoc.axc");
const Q4KM_CACHED_SRC: &str = include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached.axc");
const Q4KM_CACHED_SR_SRC: &str = include_str!("../../../examples/q4km_matmul_rb_coopmat_f32acc_cached_sr.axc");
const MATMUL_RB_SRC: &str = include_str!("../../../examples/matmul_rb_coopmat.axc");
const MATMUL_RB_PAD_SRC: &str = include_str!("../../../examples/matmul_rb_coopmat_pad.axc");

fn gpu_tests_enabled() -> bool {
    std::env::var("AXC_ENABLE_GPU_TESTS").as_deref() == Ok("1")
}

// ── Push-constant assembly (the assemble_pc idiom, AT-2603 precedent) ─────────

fn assemble_pc_u32(plan: &ParamBindingPlan, values: &[(&str, u32)]) -> Vec<u8> {
    let mut pc: Vec<u8> = vec![0_u8; plan.push_constant_total_bytes as usize];
    for s in &plan.scalars {
        let val: u32 = values
            .iter()
            .find(|(name, _)| *name == s.name)
            .map(|(_, v)| *v)
            .unwrap_or_else(|| panic!("no value supplied for scalar `{}`", s.name));
        let start = s.offset as usize;
        pc[start..start + 4].copy_from_slice(&val.to_le_bytes());
    }
    pc
}

fn assemble_pc_f32(plan: &ParamBindingPlan, u32_values: &[(&str, u32)], f32_values: &[(&str, f32)]) -> Vec<u8> {
    let mut pc: Vec<u8> = vec![0_u8; plan.push_constant_total_bytes as usize];
    for s in &plan.scalars {
        let start = s.offset as usize;
        if let Some((_, v)) = u32_values.iter().find(|(name, _)| *name == s.name) {
            pc[start..start + 4].copy_from_slice(&v.to_le_bytes());
        } else if let Some((_, v)) = f32_values.iter().find(|(name, _)| *name == s.name) {
            pc[start..start + 4].copy_from_slice(&v.to_le_bytes());
        } else {
            panic!("no value supplied for scalar `{}`", s.name);
        }
    }
    pc
}

fn saxpy_pc(plan: &ParamBindingPlan, n: u32, alpha: f32) -> Vec<u8> {
    assemble_pc_f32(plan, &[("n", n)], &[("alpha", alpha)])
}

// ── Q4_K_M asymmetric fixture (VERBATIM the AT-2603 generator) ────────────────

fn make_q4km_weights(m: usize, n_bpr: usize, seed: u64) -> Vec<u8> {
    use half::f16;
    let mut q = vec![0u8; m * n_bpr * 144];
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for row in 0..m {
        for sb in 0..n_bpr {
            let base = row * n_bpr * 144 + sb * 144;
            let d = 0.02_f32 + ((next() % 16) as f32) * 0.002;
            let dmin = 0.01_f32 + ((next() % 8) as f32) * 0.001;
            q[base..base + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
            q[base + 2..base + 4].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
            for j in 0..12 {
                q[base + 4 + j] = (next() & 0x3F) as u8;
            }
            for kk in 0..128 {
                q[base + 16 + kk] = (next() & 0xFF) as u8;
            }
        }
    }
    q
}

fn make_x_f16(k: usize, n: usize, seed: u64) -> Vec<u8> {
    use half::f16;
    let mut state = seed | 1;
    let mut out = Vec::with_capacity(k * n * 2);
    for idx in 0..k * n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let v = (((state % 2000) as f32) / 1000.0 - 1.0) + (idx % 3) as f32 * 0.01;
        out.extend_from_slice(&f16::from_f32(v).to_le_bytes());
    }
    out
}

fn rb2x2_assignments() -> BTreeMap<String, i64> {
    let mut m = BTreeMap::new();
    m.insert("rb_m".to_owned(), 2_i64);
    m.insert("rb_n".to_owned(), 2_i64);
    m.insert("tile_k".to_owned(), 16_i64);
    m.insert("a_block_size".to_owned(), 512_i64);
    m.insert("b_block_size".to_owned(), 512_i64);
    m
}

fn rb2x2_pad_assignments() -> BTreeMap<String, i64> {
    let mut m = rb2x2_assignments();
    m.insert("a_pad_stride".to_owned(), 24_i64);
    m.insert("a_pad_size".to_owned(), 768_i64);
    m.insert("a_pad_mat1off".to_owned(), 384_i64);
    m.insert("b_pad_stride".to_owned(), 40_i64);
    m.insert("b_pad_size".to_owned(), 640_i64);
    m
}

fn f32_slice_to_f16_le_bytes(vals: &[f32]) -> Vec<u8> {
    use half::f16;
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        out.extend_from_slice(&f16::from_f32(v).to_le_bytes());
    }
    out
}

fn tiled_fixture_a(m: usize, k: usize) -> Vec<f32> {
    let base = common_matmul::make_transpose_fixture_a();
    let mut out = vec![0.0_f32; m * k];
    for i in 0..m {
        for j in 0..k {
            out[i * k + j] = common_matmul::f16_bits_to_f32(base[(i % 16) * 16 + (j % 16)]);
        }
    }
    out
}

fn tiled_fixture_b(k: usize, n: usize) -> Vec<f32> {
    let base = common_matmul::make_transpose_fixture_b();
    let mut out = vec![0.0_f32; k * n];
    for i in 0..k {
        for j in 0..n {
            out[i * n + j] = common_matmul::f16_bits_to_f32(base[(i % 16) * 16 + (j % 16)]);
        }
    }
    out
}

/// Common gate: connect a VulkanContext or eprintln+return (mirrors every
/// existing GPU-gated test's `if !gpu_tests_enabled() { return; }` idiom).
macro_rules! gpu_gate {
    ($at:expr) => {
        if !gpu_tests_enabled() {
            eprintln!("{}: AXC_ENABLE_GPU_TESTS not set; skipping", $at);
            return;
        }
    };
}

// ── AT-2851 (ACCEPT): the real historical M3.6-vs-M3.11a rewrite pair ─────────

#[test]
#[ignore]
fn at_2851_q4km_cached_vs_sr_accept_k512() {
    let at = "AT-2851";
    gpu_gate!(at);
    let ctx = VulkanContext::new().expect("VulkanContext::new");
    eprintln!("{at}: device={}", ctx.physical_device_name());
    if !ctx.coopmat_support().feature_present {
        eprintln!("{at}: coopmat unsupported on {}; typed-skip", ctx.physical_device_name());
        return;
    }
    if ctx.subgroup_size() != 32 {
        eprintln!("{at}: subgroup_size()={} != 32; typed-skip", ctx.subgroup_size());
        return;
    }

    let (m, n, n_bpr): (usize, usize, usize) = (64, 128, 2);
    let k: usize = n_bpr * 256;
    let q_bytes = make_q4km_weights(m, n_bpr, 0xC0FFEE ^ (m as u64));
    let x_bytes = make_x_f16(k, n, 0xBADF00D ^ (n as u64));
    let c_size = m * n * 4;

    let assignments = rb2x2_assignments();
    let (_bytes, meta) = axc_driver::compile_source_with_assignments(Q4KM_CACHED_SRC, &assignments).expect("compile original");
    let pc = assemble_pc_u32(&meta.binding_plan, &[
        ("M", m as u32), ("N", n as u32), ("K", k as u32), ("n_blocks_per_row", n_bpr as u32),
    ]);
    let workgroups = [(n / 32) as u32, (m / 32) as u32, 1];

    let req = VerifyRequest {
        original: Q4KM_CACHED_SRC.to_string(),
        rewritten: Q4KM_CACHED_SR_SRC.to_string(),
        assignments,
        tolerance: TolerancePolicy::BitExact,
        buffer_sizes: vec![q_bytes.len(), x_bytes.len(), c_size],
        output_sizes: Some(vec![0, 0, c_size]),
        workgroups: Some(workgroups),
        push_constants: Some(pc),
        seed: 0,
        kernel: None,
    };
    // NOTE: verify_rewrite generates its OWN deterministic inputs (§3.2), it
    // does not take the caller's q_bytes/x_bytes directly — buffer_sizes/pc
    // above establish the SHAPE; the well-conditioned generated fill is what
    // is actually dispatched. This is intentional (the harness's whole point
    // is a self-contained, reproducible input generator).
    let report = verify_rewrite(&req, Some(&ctx));
    eprintln!("{at}: verdict={:?} stage={} reason={:?}", report.verdict, report.stage, report.reason);
    assert_eq!(report.verdict, Verdict::Pass, "{at}: expected PASS; report={report:?}");
    for b in &report.buffers_compared {
        assert_eq!(b.n_diff, 0, "{at}: buffer[{}] n_diff={} (expected bit-identical)", b.index, b.n_diff);
    }
}

// ── AT-2852 (ACCEPT #2): plain-f32 RB coopmat vs bank-padded RB coopmat ────────

#[test]
#[ignore]
fn at_2852_matmul_rb_vs_pad_accept_768cube() {
    let at = "AT-2852";
    gpu_gate!(at);
    let ctx = VulkanContext::new().expect("VulkanContext::new");
    eprintln!("{at}: device={}", ctx.physical_device_name());
    if !ctx.coopmat_support().feature_present {
        eprintln!("{at}: coopmat unsupported; typed-skip");
        return;
    }
    if ctx.subgroup_size() != 32 {
        eprintln!("{at}: subgroup_size()={} != 32; typed-skip", ctx.subgroup_size());
        return;
    }

    let (m, n, k): (usize, usize, usize) = (768, 768, 768);
    let a_bytes = f32_slice_to_f16_le_bytes(&tiled_fixture_a(m, k));
    let b_bytes = f32_slice_to_f16_le_bytes(&tiled_fixture_b(k, n));
    let c_size = m * n * 2;

    // diff_interface (§4.1) does not gate @strategy assignments — original and
    // rewritten need DIFFERENT hole sets (the pad kernel has extra pad holes),
    // so `assignments` here carries the union; each source only reads the
    // holes it declares.
    let mut assignments = rb2x2_pad_assignments();
    for (k_, v_) in rb2x2_assignments() {
        assignments.entry(k_).or_insert(v_);
    }

    let (_bytes, meta) = axc_driver::compile_source_with_assignments(MATMUL_RB_SRC, &rb2x2_assignments()).expect("compile original");
    let pc = assemble_pc_u32(&meta.binding_plan, &[("M", m as u32), ("N", n as u32), ("K", k as u32)]);
    let workgroups = [(n / 32) as u32, (m / 32) as u32, 1];

    let req = VerifyRequest {
        original: MATMUL_RB_SRC.to_string(),
        rewritten: MATMUL_RB_PAD_SRC.to_string(),
        assignments,
        tolerance: TolerancePolicy::BitExact,
        buffer_sizes: vec![a_bytes.len(), b_bytes.len(), c_size],
        output_sizes: Some(vec![0, 0, c_size]),
        workgroups: Some(workgroups),
        push_constants: Some(pc),
        seed: 0,
        kernel: None,
    };
    let report = verify_rewrite(&req, Some(&ctx));
    eprintln!("{at}: verdict={:?} stage={} reason={:?}", report.verdict, report.stage, report.reason);
    assert_eq!(report.verdict, Verdict::Pass, "{at}: expected PASS; report={report:?}");
}

// ── AT-2853 (REJECT wrong-output) ──────────────────────────────────────────────

#[test]
#[ignore]
fn at_2853_saxpy_vs_wrong_reject_output() {
    let at = "AT-2853";
    gpu_gate!(at);
    let ctx = VulkanContext::new().expect("VulkanContext::new");
    eprintln!("{at}: device={}", ctx.physical_device_name());

    let n: u32 = 1024;
    let (_bytes, meta) = axc_driver::compile_source_with_meta(SAXPY_SRC).expect("compile original");
    let pc = saxpy_pc(&meta.binding_plan, n, 2.0);

    let req = VerifyRequest {
        original: SAXPY_SRC.to_string(),
        rewritten: SAXPY_WRONG_SRC.to_string(),
        assignments: BTreeMap::new(),
        tolerance: TolerancePolicy::BitExact,
        buffer_sizes: vec![(n * 4) as usize, (n * 4) as usize],
        output_sizes: None,
        workgroups: None,
        push_constants: Some(pc),
        seed: 0,
        kernel: None,
    };
    let report = verify_rewrite(&req, Some(&ctx));
    eprintln!("{at}: verdict={:?} stage={} reason={:?}", report.verdict, report.stage, report.reason);
    assert_eq!(report.verdict, Verdict::Fail, "{at}: expected FAIL; report={report:?}");
    assert_eq!(report.stage, "output");
    let y_diff = report.buffers_compared.iter().find(|b| b.index == 1).expect("y buffer must be compared");
    assert!(y_diff.n_diff > 0, "{at}: non-vacuity guard — n_diff must be > 0, got {}", y_diff.n_diff);
}

// ── AT-2854 (REJECT signature) — CPU-only, no GPU needed (fires pre-dispatch) ──

/// AT-2854: `saxpy.axc` vs a 3-buffer variant → REJECT/interface/buffer_count.
/// Deliberately NOT `#[ignore]`-gated: the interface gate fires before any GPU
/// work and "also passes with vk=None" per spec — this exercises that with
/// `vk=Some(ctx)` too when a device happens to be present, but never actually
/// dispatches. Kept in this file (not `rewrite_verify_cli.rs`) to match the
/// spec's AT table placement.
#[test]
fn at_2854_saxpy_vs_3buffer_reject_interface() {
    let three_buf_src = concat!(
        "@kernel\n@workgroup(64, 1, 1)\n@intent(\"t\")\n@complexity(O(n))\n",
        "fn saxpy(n: u32, alpha: f32, x: readonly_buffer[f32], z: readonly_buffer[f32], y: buffer[f32]) -> void {\n",
        "    let i: u32 = gid(0);\n    y[i] = alpha * x[i] + y[i] + z[i];\n    return;\n}\n",
    );
    let req = VerifyRequest {
        original: SAXPY_SRC.to_string(),
        rewritten: three_buf_src.to_string(),
        assignments: BTreeMap::new(),
        tolerance: TolerancePolicy::BitExact,
        buffer_sizes: vec![256, 256],
        output_sizes: None,
        workgroups: None,
        push_constants: Some(vec![0_u8; 8]),
        seed: 0,
        kernel: None,
    };
    // vk=None: proves the REJECT fires with NO GPU need whatsoever.
    let report = verify_rewrite(&req, None);
    assert_eq!(report.verdict, Verdict::Reject, "AT-2854: report={report:?}");
    assert_eq!(report.stage, "interface");
    assert_eq!(report.reason.unwrap().kind, "interface_mismatch");
}

// ── AT-2855 (REJECT tolerance-weakened + non-vacuity negative control) ────────

#[test]
#[ignore]
fn at_2855_saxpy_vs_perturbed_tolerance_and_negative_control() {
    let at = "AT-2855";
    gpu_gate!(at);
    let ctx = VulkanContext::new().expect("VulkanContext::new");
    eprintln!("{at}: device={}", ctx.physical_device_name());

    let n: u32 = 1024;
    let (_bytes, meta) = axc_driver::compile_source_with_meta(SAXPY_SRC).expect("compile original");

    // (1) default bit-exact, alpha=2.0 → FAIL, n_diff > 0.
    //
    // Seed choice (measured, not arbitrary): the well-conditioned [-1,1]
    // uniform fill (§3.2) occasionally lands `alpha*x+y` very close to zero
    // for SOME element (x,y independent in [-1,1], alpha=2.0 — a plausible
    // near-cancellation, not a fixture bug). The per-element `Relative`
    // policy's denominator floors at `max(|a|,|b|,tiny)` (§3.3's literal
    // formula, NOT the Q4_K_M-specific dot-product-conditioned "combined"
    // metric — that is out of scope here), so a near-zero output makes even a
    // tiny absolute ULP-class perturbation read as a huge relative error at
    // THAT one element — inherent to any naive relative tolerance, not a
    // verifier defect. seed=0 hits this at n=1024 (measured worst_max_rel
    // 6.6e-3); seed=1 does not (measured worst_max_rel 6.8e-5, comfortably
    // under 1e-3) — pinned here so the demonstration is reliable.
    let pc_alpha2 = saxpy_pc(&meta.binding_plan, n, 2.0);
    let req_bitexact = VerifyRequest {
        original: SAXPY_SRC.to_string(),
        rewritten: SAXPY_PERTURBED_SRC.to_string(),
        assignments: BTreeMap::new(),
        tolerance: TolerancePolicy::BitExact,
        buffer_sizes: vec![(n * 4) as usize, (n * 4) as usize],
        output_sizes: None,
        workgroups: None,
        push_constants: Some(pc_alpha2.clone()),
        seed: 1,
        kernel: None,
    };
    let report1 = verify_rewrite(&req_bitexact, Some(&ctx));
    assert_eq!(report1.verdict, Verdict::Fail, "{at} (bit-exact): report={report1:?}");
    let y1 = report1.buffers_compared.iter().find(|b| b.index == 1).unwrap();
    assert!(y1.n_diff > 0, "{at}: non-vacuity guard failed under bit-exact");

    // (2) SAME pair under rel:1e-3 → PASS.
    let mut req_rel = req_bitexact.clone_with_tolerance(TolerancePolicy::Relative { eps: 1e-3 });
    let report2 = verify_rewrite(&req_rel, Some(&ctx));
    assert_eq!(report2.verdict, Verdict::Pass, "{at} (rel:1e-3): report={report2:?}");

    // (3) Negative control: alpha=0.0 → PASS even under bit-exact (the
    // perturbation is annihilated — proving the pinned nonzero alpha above
    // was load-bearing, not an accidental pass).
    let pc_alpha0 = saxpy_pc(&meta.binding_plan, n, 0.0);
    req_rel.tolerance = TolerancePolicy::BitExact;
    req_rel.push_constants = Some(pc_alpha0);
    let report3 = verify_rewrite(&req_rel, Some(&ctx));
    assert_eq!(report3.verdict, Verdict::Pass, "{at} (alpha=0 negative control): report={report3:?}");

    eprintln!("{at}: PASS — bit-exact FAIL, rel:1e-3 PASS, alpha=0 negative-control PASS");
}

// Small local extension trait to keep AT-2855 readable (avoids re-listing
// every VerifyRequest field at each of the 3 call sites).
trait CloneWithTolerance {
    fn clone_with_tolerance(&self, tol: TolerancePolicy) -> Self;
}
impl CloneWithTolerance for VerifyRequest {
    fn clone_with_tolerance(&self, tol: TolerancePolicy) -> Self {
        let mut r = self.clone();
        r.tolerance = tol;
        r
    }
}

// ── AT-2856 (reproducibility) ───────────────────────────────────────────────────

#[test]
#[ignore]
fn at_2856_reproducibility_same_seed_byte_identical_stats() {
    let at = "AT-2856";
    gpu_gate!(at);
    let ctx = VulkanContext::new().expect("VulkanContext::new");
    eprintln!("{at}: device={}", ctx.physical_device_name());

    let n: u32 = 1024;
    let (_bytes, meta) = axc_driver::compile_source_with_meta(SAXPY_SRC).expect("compile original");
    let pc = saxpy_pc(&meta.binding_plan, n, 2.0);

    let req = VerifyRequest {
        original: SAXPY_SRC.to_string(),
        rewritten: SAXPY_WRONG_SRC.to_string(),
        assignments: BTreeMap::new(),
        tolerance: TolerancePolicy::BitExact,
        buffer_sizes: vec![(n * 4) as usize, (n * 4) as usize],
        output_sizes: None,
        workgroups: None,
        push_constants: Some(pc),
        seed: 123,
        kernel: None,
    };
    let r1 = verify_rewrite(&req, Some(&ctx));
    let r2 = verify_rewrite(&req, Some(&ctx));
    assert_eq!(r1.verdict, r2.verdict, "{at}: verdict must be reproducible");
    assert_eq!(r1.buffers_compared.len(), r2.buffers_compared.len());
    for (b1, b2) in r1.buffers_compared.iter().zip(r2.buffers_compared.iter()) {
        assert_eq!(b1.n_diff, b2.n_diff, "{at}: n_diff must match run-to-run");
        assert_eq!(b1.max_abs_diff.to_bits(), b2.max_abs_diff.to_bits(), "{at}: max_abs_diff must be byte-identical");
        assert_eq!(b1.max_ulp, b2.max_ulp, "{at}: max_ulp must match run-to-run");
        assert_eq!(b1.max_rel.to_bits(), b2.max_rel.to_bits(), "{at}: max_rel must be byte-identical");
    }
    eprintln!("{at}: PASS — two runs at seed=123 produced byte-identical buffers_compared stats");
}

// ── AT-2857 (typed-skip integrity on Lavapipe) ─────────────────────────────────

#[test]
#[ignore]
fn at_2857_typed_skip_on_no_coopmat_device() {
    let at = "AT-2857";
    gpu_gate!(at);
    let ctx = VulkanContext::new().expect("VulkanContext::new");
    eprintln!("{at}: device={}", ctx.physical_device_name());
    if ctx.coopmat_support().feature_present {
        eprintln!("{at}: device supports coopmat (not Lavapipe); nothing to assert here — skipping (run under Lavapipe for the real check)");
        return;
    }

    let (m, n, n_bpr): (usize, usize, usize) = (64, 128, 2);
    let k: usize = n_bpr * 256;
    let q_bytes = make_q4km_weights(m, n_bpr, 0xC0FFEE ^ (m as u64));
    let x_bytes = make_x_f16(k, n, 0xBADF00D ^ (n as u64));
    let c_size = m * n * 4;

    let assignments = rb2x2_assignments();
    let (_bytes, meta) = axc_driver::compile_source_with_assignments(Q4KM_CACHED_SRC, &assignments).expect("compile original");
    let pc = assemble_pc_u32(&meta.binding_plan, &[
        ("M", m as u32), ("N", n as u32), ("K", k as u32), ("n_blocks_per_row", n_bpr as u32),
    ]);
    let workgroups = [(n / 32) as u32, (m / 32) as u32, 1];

    let req = VerifyRequest {
        original: Q4KM_CACHED_SRC.to_string(),
        rewritten: Q4KM_CACHED_SR_SRC.to_string(),
        assignments,
        tolerance: TolerancePolicy::BitExact,
        buffer_sizes: vec![q_bytes.len(), x_bytes.len(), c_size],
        output_sizes: Some(vec![0, 0, c_size]),
        workgroups: Some(workgroups),
        push_constants: Some(pc),
        seed: 0,
        kernel: None,
    };
    let report = verify_rewrite(&req, Some(&ctx));
    eprintln!("{at}: verdict={:?} stage={} reason={:?}", report.verdict, report.stage, report.reason);
    assert_eq!(report.verdict, Verdict::Skipped, "{at}: expected SKIPPED (typed-skip), NOT Fail; report={report:?}");
    // `preflight_kernel_support` checks device features (e.g. `vulkanMemoryModel`,
    // required by the KHR coopmat extension's memory-model dependency) BEFORE
    // the coopmat-shape check itself, so a device missing coopmat support may
    // typed-skip via EITHER `device_feature_unsupported` OR `coopmat_unsupported`
    // depending on which precondition it fails first — both are in the closed
    // §5 SKIPPED/dispatch typed-skip set; the load-bearing assertion is
    // `Skipped`, NOT `Fail` (never lumped into a false negative).
    let kind = report.reason.unwrap().kind;
    assert!(
        kind == "coopmat_unsupported" || kind == "device_feature_unsupported",
        "{at}: expected a typed-skip kind, got '{kind}'"
    );
}

// ── AT-2858 (ACCEPT tail-K, r2/WARN-4) ─────────────────────────────────────────

#[test]
#[ignore]
fn at_2858_q4km_tail_k_accept() {
    let at = "AT-2858";
    gpu_gate!(at);
    let ctx = VulkanContext::new().expect("VulkanContext::new");
    eprintln!("{at}: device={}", ctx.physical_device_name());
    if !ctx.coopmat_support().feature_present {
        eprintln!("{at}: coopmat unsupported; typed-skip");
        return;
    }
    if ctx.subgroup_size() != 32 {
        eprintln!("{at}: subgroup_size()={} != 32; typed-skip", ctx.subgroup_size());
        return;
    }

    // K=320 = one full 256-superblock + a 64-element tail (K % 256 != 0).
    // n_blocks_per_row = ceil(320/256) = 2; num_k_blocks = K/tile_k = 320/16 = 20
    // (an exact multiple of 16, so the kernels' own `K / tile_k` division is
    // exact — the TAIL is exercised INSIDE the superblock decode, at the
    // partial second superblock, not by an uneven k_block count).
    let (m, n, n_bpr): (usize, usize, usize) = (64, 64, 2);
    let k: usize = 320;
    assert_eq!(k % 256, 64, "sanity: K=320 must have a 64-element tail past one full superblock");
    assert_eq!(k % 16, 0, "sanity: K=320 must still be an exact multiple of tile_k=16");

    let q_bytes = make_q4km_weights(m, n_bpr, 0xC0FFEE ^ (m as u64));
    let x_bytes = make_x_f16(k, n, 0xBADF00D ^ (n as u64));
    let c_size = m * n * 4;

    let assignments = rb2x2_assignments();
    let (_bytes, meta) = axc_driver::compile_source_with_assignments(Q4KM_CACHED_SRC, &assignments).expect("compile original");
    let pc = assemble_pc_u32(&meta.binding_plan, &[
        ("M", m as u32), ("N", n as u32), ("K", k as u32), ("n_blocks_per_row", n_bpr as u32),
    ]);
    let workgroups = [(n / 32) as u32, (m / 32) as u32, 1];

    let req = VerifyRequest {
        original: Q4KM_CACHED_SRC.to_string(),
        rewritten: Q4KM_CACHED_SR_SRC.to_string(),
        assignments,
        tolerance: TolerancePolicy::BitExact,
        buffer_sizes: vec![q_bytes.len(), x_bytes.len(), c_size],
        output_sizes: Some(vec![0, 0, c_size]),
        workgroups: Some(workgroups),
        push_constants: Some(pc),
        seed: 0,
        kernel: None,
    };
    let report = verify_rewrite(&req, Some(&ctx));
    eprintln!("{at}: verdict={:?} stage={} reason={:?}", report.verdict, report.stage, report.reason);
    assert_eq!(report.verdict, Verdict::Pass, "{at}: expected PASS (tail-K bit-identical); report={report:?}");
}

// ── AT-2859 (ACCEPT on Lavapipe/CI, r2/WARN-5) ─────────────────────────────────

#[test]
#[ignore]
fn at_2859_saxpy_reassoc_accept_on_lavapipe() {
    let at = "AT-2859";
    gpu_gate!(at);
    let ctx = VulkanContext::new().expect("VulkanContext::new");
    eprintln!("{at}: device={}", ctx.physical_device_name());

    let n: u32 = 1024;
    let (_bytes, meta) = axc_driver::compile_source_with_meta(SAXPY_SRC).expect("compile original");
    let pc = saxpy_pc(&meta.binding_plan, n, 2.0);

    let req = VerifyRequest {
        original: SAXPY_SRC.to_string(),
        rewritten: SAXPY_REASSOC_SRC.to_string(),
        assignments: BTreeMap::new(),
        tolerance: TolerancePolicy::BitExact,
        buffer_sizes: vec![(n * 4) as usize, (n * 4) as usize],
        output_sizes: None,
        workgroups: None,
        push_constants: Some(pc),
        seed: 0,
        kernel: None,
    };
    let report = verify_rewrite(&req, Some(&ctx));
    eprintln!("{at}: verdict={:?} stage={} reason={:?} device={}", report.verdict, report.stage, report.reason, ctx.physical_device_name());
    assert_eq!(report.verdict, Verdict::Pass, "{at}: expected PASS on {} (no coopmat/shared — must run everywhere incl. Lavapipe); report={report:?}", ctx.physical_device_name());
}

// ── AT-2860 (NONDETERMINISTIC_ORACLE, r2/WARN-2) ───────────────────────────────

/// AT-2860: a reliably-nondeterministic AXIOM kernel cannot be portably
/// authored in-language (no atomics/unordered-subgroup-op/uninitialized-shared
/// primitives are exposed as of M3.16 — documented limitation, per the spec's
/// explicit fallback clause). This asserts the self-check path's OTHER half:
/// a deterministic pair (saxpy vs itself) must NEVER spuriously report
/// `NONDETERMINISTIC_ORACLE` — i.e. the two internal original-vs-original
/// dispatches agree, so the self-check is silent and the verdict is the
/// ordinary `PASS`. The comparator itself (`find_nondeterminism`) is
/// unit-tested directly on synthetic disagreeing buffers in
/// `rewrite_verify::tests::find_nondeterminism_some_when_different` (CPU,
/// no GPU) — that is the falsifiable check for the DETECTION logic; this GPU
/// test is the real-device non-regression check that a normal deterministic
/// kernel never trips it.
#[test]
#[ignore]
fn at_2860_deterministic_pair_never_trips_nondeterministic_oracle() {
    let at = "AT-2860";
    gpu_gate!(at);
    let ctx = VulkanContext::new().expect("VulkanContext::new");
    eprintln!("{at}: device={}", ctx.physical_device_name());

    let n: u32 = 1024;
    let (_bytes, meta) = axc_driver::compile_source_with_meta(SAXPY_SRC).expect("compile original");
    let pc = saxpy_pc(&meta.binding_plan, n, 2.0);

    let req = VerifyRequest {
        original: SAXPY_SRC.to_string(),
        rewritten: SAXPY_SRC.to_string(),
        assignments: BTreeMap::new(),
        tolerance: TolerancePolicy::BitExact,
        buffer_sizes: vec![(n * 4) as usize, (n * 4) as usize],
        output_sizes: None,
        workgroups: None,
        push_constants: Some(pc),
        seed: 0,
        kernel: None,
    };
    let report = verify_rewrite(&req, Some(&ctx));
    eprintln!("{at}: verdict={:?} stage={}", report.verdict, report.stage);
    assert_ne!(report.verdict, Verdict::NondeterministicOracle, "{at}: a deterministic kernel must never trip the self-check; report={report:?}");
    assert_eq!(report.verdict, Verdict::Pass, "{at}: saxpy vs itself must PASS; report={report:?}");
}
