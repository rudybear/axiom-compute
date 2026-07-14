//! M3.21 (FG.9) — GPU dispatch test for `examples/two_pass_reduce.axc`.
//!
//! AT-2958: spirv-val clean on BOTH emitted kernel modules (runs unconditionally,
//! no GPU required).
//! AT-2959/AT-2960: two-stage sequential GPU dispatch (`partial_reduce` then
//! `final_reduce`, two independent pipelines — the "N separate SPIR-V modules"
//! v1 architecture, §3 of the spec) is bit-exact vs a CPU `i64`-accumulated sum
//! oracle (`max_diff == 0`, i32 addition is associative so grouping cannot
//! perturb the result). Runs on whichever real Vulkan device is available
//! (NVIDIA when `AXC_ENABLE_GPU_TESTS=1` + a discrete GPU; Lavapipe as the CI
//! software-rasterizer fallback) — the SAME test code proves both legs, since
//! this kernel uses only `shared[i32,N]` + `workgroup_barrier()` + `gid()` +
//! `local_invocation_id()`, all shipped since M3.2/M3.3d with no coopmat
//! dependency requiring a typed-skip.

use std::collections::BTreeMap;

use axc_driver::{compile_module_all, compile_source_with_meta_kernel, substitute_strategy_holes};
use axc_runtime::VulkanContext;

const TWO_PASS_REDUCE_SRC: &str = include_str!("../../../examples/two_pass_reduce.axc");

const N: u32 = 4096;
const WG: u32 = 64;
const N_PARTS: u32 = N / WG; // 64, == WG exactly (no padding needed).

fn gpu_tests_enabled() -> bool {
    std::env::var("AXC_ENABLE_GPU_TESTS").as_deref() == Ok("1")
}

fn wg_assignment() -> BTreeMap<String, i64> {
    let mut m = BTreeMap::new();
    m.insert("WG".to_string(), i64::from(WG));
    m
}

fn resolved_source() -> String {
    substitute_strategy_holes(TWO_PASS_REDUCE_SRC, &wg_assignment())
}

/// AT-2945 (locked here too, real-file form): the raw file declares 2 kernels.
#[test]
fn at_2945_two_pass_reduce_has_two_kernels() {
    let (ast, lex_errs, parse_errs) = axc_parser::parse(TWO_PASS_REDUCE_SRC);
    assert!(lex_errs.is_empty() && parse_errs.is_empty(), "lex/parse errors: {lex_errs:?} {parse_errs:?}");
    let (hir, hir_errs, _warns) = axc_hir::lower_module(&ast);
    assert!(hir_errs.is_empty(), "hir errors: {hir_errs:?}");
    assert_eq!(hir.kernels.len(), 2, "two_pass_reduce.axc must declare exactly 2 kernels");
    assert_eq!(hir.kernels[0].name, "partial_reduce");
    assert_eq!(hir.kernels[1].name, "final_reduce");
}

/// AT-2956 (real-file corroboration): resolving ?WG=64 rewrites BOTH kernels'
/// `@workgroup(?WG,1,1)` to `@workgroup(64,1,1)` and the hole disappears.
/// (The comment prose legitimately contains OTHER `?` characters — e.g. the
/// generic term `` `?hole` `` — so this checks the specific `?WG` reference
/// and the live `@strategy {` block, not a blanket "no `?` anywhere".)
#[test]
fn at_2956_resolving_wg_rewrites_both_kernels_workgroup() {
    let resolved = resolved_source();
    assert!(!resolved.contains("?WG"), "no `?WG` hole ref may survive resolution: {resolved}");
    let live_strategy_blocks: usize = resolved
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .filter(|line| line.contains("@strategy {"))
        .count();
    assert_eq!(live_strategy_blocks, 0, "both @strategy {{ blocks must be stripped; got:\n{resolved}");
    let occurrences = resolved.matches("@workgroup(64, 1, 1)").count();
    assert_eq!(occurrences, 2, "both kernels' @workgroup must resolve to (64,1,1); got:\n{resolved}");
}

/// AT-2958: spirv-val clean on BOTH emitted kernel modules. No GPU required.
#[test]
fn at_2958_both_kernels_spirv_val_clean() {
    use spirv_tools::val::{Validator, create as create_validator};
    use spirv_tools::TargetEnv;

    let resolved = resolved_source();
    let compiled = compile_module_all(&resolved, false)
        .expect("two_pass_reduce.axc must compile via compile_module_all");
    assert_eq!(compiled.len(), 2, "expected exactly 2 compiled kernels");

    let validator = create_validator(Some(TargetEnv::Vulkan_1_1));
    for ck in &compiled {
        let words: Vec<u32> = ck.spirv.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        validator.validate(&words, None)
            .unwrap_or_else(|e| panic!("AT-2958: spirv-val rejected kernel `{}`: {e}", ck.name));
    }
}

/// AT-2946: `compile_module_all` on this 2-kernel source produces 2
/// `CompiledKernel`s with distinct names, distinct SPIR-V, distinct metadata.
#[test]
fn at_2946_compile_module_all_distinct_kernels() {
    let resolved = resolved_source();
    let compiled = compile_module_all(&resolved, false).expect("must compile");
    assert_eq!(compiled.len(), 2);
    assert_ne!(compiled[0].name, compiled[1].name);
    assert_ne!(compiled[0].spirv, compiled[1].spirv, "distinct kernels must not compile to identical SPIR-V");
    assert_eq!(compiled[0].metadata.entry_point, compiled[0].name);
    assert_eq!(compiled[1].metadata.entry_point, compiled[1].name);
    assert_ne!(
        compiled[0].metadata.binding_plan.buffers.len(),
        0,
        "partial_reduce must have buffer bindings"
    );
}

/// AT-2959 / AT-2960: dispatch stage1 (`partial_reduce`) then stage2
/// (`final_reduce`) IN SEQUENCE on a real GPU. `out[0] == Sigma(input)`
/// exactly. Runs on NVIDIA (real hardware) or Lavapipe (CI fallback) —
/// whichever `VulkanContext::new()` resolves to.
#[test]
#[ignore]
fn at_2959_2960_two_stage_dispatch_bit_exact_sum() {
    if !gpu_tests_enabled() {
        eprintln!("at_2959_2960: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }

    let resolved = resolved_source();

    let (partial_bytes, partial_meta) = compile_source_with_meta_kernel(&resolved, Some("partial_reduce"), false)
        .expect("partial_reduce must compile");
    let (final_bytes, final_meta) = compile_source_with_meta_kernel(&resolved, Some("final_reduce"), false)
        .expect("final_reduce must compile");

    let partial_words: Vec<u32> = partial_bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let final_words: Vec<u32> = final_bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let ctx = VulkanContext::new().expect("VulkanContext must init");
    eprintln!("at_2959_2960: device={}", ctx.physical_device_name());

    // ── Input + CPU oracle (i64 accumulation avoids any host-side overflow
    // ambiguity; the GPU side stays strictly i32, matching the kernel body) ──
    let input_i32: Vec<i32> = (0..N).map(|i| ((i % 7) as i32) - 3).collect(); // range [-3, 3]
    let cpu_sum: i64 = input_i32.iter().map(|&v| i64::from(v)).sum();
    let input_bytes: Vec<u8> = input_i32.iter().flat_map(|v| v.to_le_bytes()).collect();

    // ── Stage 1: partial_reduce, grid = (N/WG, 1, 1) workgroups ──────────────
    let partial_handle = ctx.prepare_kernel_checked(
        &partial_words, &partial_meta.binding_plan, partial_meta.push_constant_total_bytes,
        &partial_meta.entry_point, None, "partial_reduce", partial_meta.shared_memory_bytes,
    ).unwrap_or_else(|e| panic!("at_2959_2960: partial_reduce pipeline create failed: {e}"));

    let n_pc: Vec<u8> = N.to_le_bytes().to_vec();
    let partials_size: usize = (N_PARTS as usize) * 4;
    let stage1_outputs = ctx.dispatch_handle(
        &partial_handle, (N / WG, 1, 1),
        &[&input_bytes, &vec![0u8; partials_size]],
        &[0, partials_size],
        &n_pc,
    ).unwrap_or_else(|e| panic!("at_2959_2960: partial_reduce dispatch failed: {e}"));

    let partials_bytes: &[u8] = &stage1_outputs[1];
    assert_eq!(partials_bytes.len(), partials_size);

    // ── Stage 2: final_reduce, a SINGLE workgroup of WG threads ──────────────
    let final_handle = ctx.prepare_kernel_checked(
        &final_words, &final_meta.binding_plan, final_meta.push_constant_total_bytes,
        &final_meta.entry_point, None, "final_reduce", final_meta.shared_memory_bytes,
    ).unwrap_or_else(|e| panic!("at_2959_2960: final_reduce pipeline create failed: {e}"));

    let n_parts_pc: Vec<u8> = N_PARTS.to_le_bytes().to_vec();
    let stage2_outputs = ctx.dispatch_handle(
        &final_handle, (1, 1, 1),
        &[partials_bytes, &[0u8; 4]],
        &[0, 4],
        &n_parts_pc,
    ).unwrap_or_else(|e| panic!("at_2959_2960: final_reduce dispatch failed: {e}"));

    let out_bytes: &[u8] = &stage2_outputs[1];
    let gpu_sum: i32 = i32::from_le_bytes([out_bytes[0], out_bytes[1], out_bytes[2], out_bytes[3]]);

    eprintln!("at_2959_2960: gpu_sum={gpu_sum}, cpu_sum={cpu_sum} on {}", ctx.physical_device_name());
    let max_diff: i64 = (i64::from(gpu_sum) - cpu_sum).abs();
    assert!(
        max_diff == 0,
        "AT-2959/2960: two-stage reduce is NOT bit-exact — gpu_sum={gpu_sum}, cpu_sum={cpu_sum}, \
         max_diff={max_diff} on {} (i32 addition is associative; any nonzero diff is a \
         barrier/race/index bug, not a reassociation artifact)",
        ctx.physical_device_name()
    );
    eprintln!(
        "at_2959_2960: PASS — two-pass i32 sum reduction bit-exact on {} (n={N}, wg={WG}, n_parts={N_PARTS})",
        ctx.physical_device_name()
    );
}
