//! M3.17 (FG.4) GPU acceptance tests: AT-2870, AT-2871, AT-2872, AT-2873, AT-2878, AT-2879.
//!
//! Compiles `examples/precondition_saxpy.axc` with `--debug` and dispatches it via
//! `VulkanContext::dispatch_debug_checked` — the injected flag buffer is handled
//! transparently by the EXISTING generic `prepare_kernel_checked`/`dispatch_handle`
//! path (§7 of the spec; see `axc_runtime::dispatch::dispatch_debug_checked` doc).
//!
//! GPU-gated: requires `AXC_ENABLE_GPU_TESTS=1` + a responsive Vulkan ICD. Per the
//! coder mandate this suite is run on BOTH real NVIDIA (no `VK_DRIVER_FILES`) and
//! Lavapipe (`VK_DRIVER_FILES=.../lvp_icd.json`), plus the
//! `AXC_FORCE_NONCOHERENT_STAGING=1` leg (AT-2879).
//!
//! Fixture conditions (see `examples/precondition_saxpy.axc`):
//!   pre  bit 0: gt(n, 0)
//!   pre  bit 1: gt(alpha, 0)      — f32 operand, used for AT-2878 (NaN-fail-loud)
//!   post bit 0: lt(alpha, 1000)

use axc_driver::compile_source_with_meta_debug;
use axc_runtime::{gpu_tests_enabled, probe_vulkan_available, DispatchError, KernelMetadata, VulkanContext};
use axc_hir::ScalarTy;

const PRECONDITION_SAXPY_SRC: &str = include_str!("../../../examples/precondition_saxpy.axc");
const N: u32 = 64; // matches @workgroup(64,1,1); one dispatch of exactly 64 threads.
const BUF_BYTES: usize = (N as usize) * 4;

fn skip_unless_gpu() -> bool {
    if !gpu_tests_enabled() {
        eprintln!("skipping (AXC_ENABLE_GPU_TESTS != 1)");
        return true;
    }
    if !probe_vulkan_available() {
        eprintln!("skipping (no Vulkan ICD available)");
        return true;
    }
    false
}

fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn f32_buf(v: f32) -> Vec<u8> {
    (0..N).flat_map(|_| v.to_le_bytes()).collect()
}

/// Build push-constant bytes for `n`/`alpha`, driven by `meta.binding_plan.scalars`
/// (plan-offset discipline — AT-514a convention, not hardcoded layout).
fn build_pc(meta: &KernelMetadata, n: u32, alpha: f32) -> Vec<u8> {
    let mut pc = vec![0u8; meta.binding_plan.push_constant_total_bytes as usize];
    for scalar in &meta.binding_plan.scalars {
        let start = scalar.offset as usize;
        match (&scalar.name[..], scalar.ty) {
            ("n", ScalarTy::U32) => pc[start..start + 4].copy_from_slice(&n.to_le_bytes()),
            ("alpha", ScalarTy::F32) => pc[start..start + 4].copy_from_slice(&alpha.to_le_bytes()),
            _ => {}
        }
    }
    pc
}

fn compile_debug() -> (Vec<u32>, KernelMetadata) {
    let (bytes, meta) = compile_source_with_meta_debug(PRECONDITION_SAXPY_SRC, true)
        .expect("precondition_saxpy.axc must compile with --debug");
    (bytes_to_words(&bytes), meta)
}

/// AT-2870 (GPU): satisfied precondition ⇒ flag=={0,0} ⇒ Ok, output identical to a
/// release dispatch. Assert the flag-buffer bytes directly, in addition to `Ok`.
#[test]
#[ignore] // GPU-gated
fn at_2870_satisfied_conditions_flag_zero_and_output_matches_release() {
    if skip_unless_gpu() { return; }
    let (words, meta) = compile_debug();
    let ctx = VulkanContext::new().expect("VulkanContext::new()");
    let x = f32_buf(2.0);
    let y = f32_buf(3.0);
    let pc = build_pc(&meta, N, 1.0); // n>0, alpha=1.0>0, alpha<1000: all satisfied

    let outcome = ctx.dispatch_debug_checked(
        &words, &meta, (1, 1, 1), &[&x, &y], &[BUF_BYTES, BUF_BYTES], &pc,
    ).expect("dispatch_debug_checked must succeed");

    assert_eq!(outcome.flag_words, [0, 0], "flag buffer must read back {{0,0}} when all conditions pass");

    // Compare against a release (non-debug) dispatch of the SAME kernel.
    let (release_bytes, release_meta) = axc_driver::compile_source_with_meta(PRECONDITION_SAXPY_SRC)
        .expect("release compile");
    let release_words = bytes_to_words(&release_bytes);
    let release_pc = build_pc(&release_meta, N, 1.0);
    let release_outputs = ctx.dispatch(axc_runtime::DispatchRequest {
        spirv: &release_words,
        binding_plan: &release_meta.binding_plan,
        workgroups: [1, 1, 1],
        inputs: &[&x, &y],
        output_sizes: &[BUF_BYTES, BUF_BYTES],
        push_constants: &release_pc,
        entry_point: &release_meta.entry_point,
    }).expect("release dispatch must succeed");

    let outputs = outcome.into_result().expect("clean flag must decode to Ok");
    assert_eq!(outputs, release_outputs, "--debug output must be identical to the release run when no check fires");
}

/// AT-2871 (GPU): violated precondition (`n == 0`) ⇒ `flag[0]` bit 0 set ⇒ typed
/// `DebugCheckViolation` naming `"gt(n, 0)"`. Assert the exact expected bitmask.
#[test]
#[ignore] // GPU-gated
fn at_2871_violated_precondition_sets_exact_bit_and_typed_error() {
    if skip_unless_gpu() { return; }
    let (words, meta) = compile_debug();
    let ctx = VulkanContext::new().expect("VulkanContext::new()");
    let x = f32_buf(2.0);
    let y = f32_buf(3.0);
    let pc = build_pc(&meta, 0, 1.0); // n==0 violates gt(n,0) (bit 0); alpha ok.

    let outcome = ctx.dispatch_debug_checked(
        &words, &meta, (1, 1, 1), &[&x, &y], &[BUF_BYTES, BUF_BYTES], &pc,
    ).expect("dispatch_debug_checked must succeed (violation is a normal outcome, not a Vulkan error)");

    assert_eq!(outcome.flag_words, [1u32 << 0, 0], "flag[0] must == exactly bit 0 (gt(n,0)); got {:?}", outcome.flag_words);

    match outcome.into_result() {
        Err(DispatchError::DebugCheckViolation { preconditions, postconditions }) => {
            assert_eq!(preconditions, vec!["gt(n, 0)".to_owned()]);
            assert!(postconditions.is_empty());
        }
        other => panic!("expected DebugCheckViolation naming gt(n, 0); got {other:?}"),
    }
}

/// AT-2872 (GPU): a postcondition violation (`alpha >= 1000`) ⇒ `flag[1]` bit 0 set
/// (assert exact bitmask) ⇒ typed violation naming `"lt(alpha, 1000)"`; a passing
/// postcondition (the AT-2870 case) ⇒ `Ok` + `flag=={0,0}`.
#[test]
#[ignore] // GPU-gated
fn at_2872_postcondition_violation_sets_exact_bit_and_typed_error() {
    if skip_unless_gpu() { return; }
    let (words, meta) = compile_debug();
    let ctx = VulkanContext::new().expect("VulkanContext::new()");
    let x = f32_buf(2.0);
    let y = f32_buf(3.0);
    let pc = build_pc(&meta, N, 2000.0); // pre both pass; post lt(alpha,1000) fails.

    let outcome = ctx.dispatch_debug_checked(
        &words, &meta, (1, 1, 1), &[&x, &y], &[BUF_BYTES, BUF_BYTES], &pc,
    ).expect("dispatch_debug_checked must succeed");

    assert_eq!(outcome.flag_words, [0, 1u32 << 0], "flag[1] must == exactly bit 0 (lt(alpha,1000)); got {:?}", outcome.flag_words);

    match outcome.into_result() {
        Err(DispatchError::DebugCheckViolation { preconditions, postconditions }) => {
            assert!(preconditions.is_empty());
            assert_eq!(postconditions, vec!["lt(alpha, 1000)".to_owned()]);
        }
        other => panic!("expected DebugCheckViolation naming lt(alpha, 1000); got {other:?}"),
    }
}

/// AT-2873 (GPU): multiple simultaneous violations ⇒ distinct bits across BOTH
/// words; host decodes ALL failing conditions. Assert the composite flag words ==
/// the exact OR of expected bit masks (n=0 ⇒ pre bit0; alpha=NaN ⇒ pre bit1 AND
/// post bit0, since NaN fails BOTH the `gt(alpha,0)` precondition and the
/// `lt(alpha,1000)` postcondition via the ordered-compare NaN-fail-loud rule).
#[test]
#[ignore] // GPU-gated
fn at_2873_multiple_violations_distinct_bits_composite() {
    if skip_unless_gpu() { return; }
    let (words, meta) = compile_debug();
    let ctx = VulkanContext::new().expect("VulkanContext::new()");
    let x = f32_buf(2.0);
    let y = f32_buf(3.0);
    let pc = build_pc(&meta, 0, f32::NAN);

    let outcome = ctx.dispatch_debug_checked(
        &words, &meta, (1, 1, 1), &[&x, &y], &[BUF_BYTES, BUF_BYTES], &pc,
    ).expect("dispatch_debug_checked must succeed");

    let expected_pre = (1u32 << 0) | (1u32 << 1);
    let expected_post = 1u32 << 0;
    assert_eq!(outcome.flag_words, [expected_pre, expected_post],
        "composite flag words must equal the exact OR of expected bitmasks; got {:?}", outcome.flag_words);

    match outcome.into_result() {
        Err(DispatchError::DebugCheckViolation { mut preconditions, postconditions }) => {
            preconditions.sort();
            assert_eq!(preconditions, vec!["gt(alpha, 0)".to_owned(), "gt(n, 0)".to_owned()]);
            assert_eq!(postconditions, vec!["lt(alpha, 1000)".to_owned()]);
        }
        other => panic!("expected DebugCheckViolation naming all 3 conditions; got {other:?}"),
    }
}

/// AT-2878 (GPU, NaN fail-loud): a NaN-valued f32 scalar precondition operand
/// (`gt(alpha, 0)` with `alpha = NaN`) sets the violation bit — NaN does NOT
/// silently pass. Assert `flag[0]` bit 1 is set (isolated: `n` stays valid so only
/// the alpha-precondition and its dependent postcondition fire — see AT-2873 for
/// the full composite; this test isolates the NaN-fail-loud claim specifically via
/// the precondition bit).
#[test]
#[ignore] // GPU-gated
fn at_2878_nan_precondition_operand_fails_loud() {
    if skip_unless_gpu() { return; }
    let (words, meta) = compile_debug();
    let ctx = VulkanContext::new().expect("VulkanContext::new()");
    let x = f32_buf(2.0);
    let y = f32_buf(3.0);
    let pc = build_pc(&meta, N, f32::NAN); // n valid; alpha=NaN must FAIL gt(alpha,0), not silently pass.

    let outcome = ctx.dispatch_debug_checked(
        &words, &meta, (1, 1, 1), &[&x, &y], &[BUF_BYTES, BUF_BYTES], &pc,
    ).expect("dispatch_debug_checked must succeed");

    assert_eq!(outcome.flag_words[0] & (1u32 << 1), 1u32 << 1,
        "NaN alpha must set the gt(alpha,0) violation bit (fail LOUD, not silently pass); got flag[0]={:#x}", outcome.flag_words[0]);
}

/// AT-2879 (GPU, coherency fail-CLOSED, principle #9): flag readback under
/// `AXC_FORCE_NONCOHERENT_STAGING=1` still observes a real violation — the
/// `vkInvalidateMappedMemoryRanges` path runs; the tool does not fail-OPEN on
/// non-coherent memory.
///
/// Uses `VulkanContext::new_with_options` with `force_noncoherent_staging: Some(true)`
/// rather than mutating the process-wide env var (test-isolation safe).
#[test]
#[ignore] // GPU-gated
fn at_2879_noncoherent_staging_still_observes_violation() {
    if skip_unless_gpu() { return; }
    let (words, meta) = compile_debug();

    let mut opts = axc_runtime::VulkanContextOptions::from_env();
    opts.force_noncoherent_staging = Some(true);
    let ctx = VulkanContext::new_with_options(opts).expect("VulkanContext::new_with_options()");

    let x = f32_buf(2.0);
    let y = f32_buf(3.0);
    let pc = build_pc(&meta, 0, 1.0); // n==0 violates gt(n,0).

    let outcome = ctx.dispatch_debug_checked(
        &words, &meta, (1, 1, 1), &[&x, &y], &[BUF_BYTES, BUF_BYTES], &pc,
    ).expect("dispatch_debug_checked must succeed under forced NonCoherent staging");

    assert_eq!(outcome.flag_words, [1u32 << 0, 0],
        "a real violation must still be observed under AXC_FORCE_NONCOHERENT_STAGING (fail-CLOSED); got {:?}",
        outcome.flag_words);
    assert!(outcome.into_result().is_err(), "the violation must still surface as a typed error");
}

/// AT-2867 reviewer nit (c): the coopmat leg must actually DISPATCH on NVIDIA (not
/// just spirv-val, which `axc-codegen`'s test already covers CPU-side/everywhere).
/// Lavapipe typed-skips only the dispatch (via `CoopMatUnsupported`); the
/// spirv-val/cap-set assertions live in `axc-codegen`'s test and run unconditionally.
/// Uses all-zero f16 tile data (bit pattern `0x0000` for every element) — this
/// exercises the injected-binding + `OpAtomicOr` + coopmat coexistence on real
/// hardware without needing an f16 conversion dependency; numeric matmul
/// correctness on this shape is already covered by AT-1510.
#[test]
#[ignore] // GPU-gated
fn at_2867c_coopmat_debug_dispatch_on_nvidia_lavapipe_typed_skip() {
    if skip_unless_gpu() { return; }
    const SRC: &str = concat!(
        "@kernel @workgroup(32,1,1) @cooperative_matrix ",
        "@intent(\"coopmat debug-check dispatch leg\") @precondition(gt(tile_offset, 0)) ",
        "fn matmul_tile_debug(tile_offset: u32, a_buf: readonly_buffer[f16], ",
        "b_buf: readonly_buffer[f16], c_buf: buffer[f16]) -> void { ",
        "let stride: u32 = 16u32; ",
        "let a: matrix[f16, 16, 16, a] = coopmat_load(a_buf, tile_offset, stride); ",
        "let b: matrix[f16, 16, 16, b] = coopmat_load(b_buf, tile_offset, stride); ",
        "let acc: matrix[f16, 16, 16, accumulator] = coopmat_zero(); ",
        "let result: matrix[f16, 16, 16, accumulator] = coopmat_mul_add(a, b, acc); ",
        "coopmat_store(result, c_buf, tile_offset, stride); return; }",
    );
    let (bytes, meta) = compile_source_with_meta_debug(SRC, true).expect("coopmat debug compile");
    let words = bytes_to_words(&bytes);
    let ctx = VulkanContext::new().expect("VulkanContext::new()");

    let tile_bytes = vec![0u8; 256 * 2]; // 256 f16 elements, all-zero bit pattern.
    let tile_offset: u32 = 0; // violates gt(tile_offset,0) — proves the mechanism, not just the happy path.
    let pc = tile_offset.to_le_bytes().to_vec();

    let outcome = ctx.dispatch_debug_checked(
        &words, &meta, (1, 1, 1),
        &[&tile_bytes, &tile_bytes, &tile_bytes], &[0, 0, tile_bytes.len()], &pc,
    );
    match outcome {
        Ok(o) => {
            assert_eq!(o.flag_words, [1u32 << 0, 0], "gt(tile_offset,0) must be flagged (tile_offset=0)");
            eprintln!("at_2867c: coopmat debug dispatch succeeded on {}", ctx.physical_device_name());
        }
        Err(DispatchError::CoopMatUnsupported { reason, .. }) => {
            eprintln!("at_2867c: typed skip (expected on Lavapipe): {reason}");
        }
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
            eprintln!("at_2867c: typed skip: DeviceFeatureUnsupported {feature}/{kernel}");
        }
        Err(e) => panic!("at_2867c: unexpected dispatch error: {e:?}"),
    }
}
