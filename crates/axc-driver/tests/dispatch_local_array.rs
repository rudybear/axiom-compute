//! GPU dispatch tests for M3.20 `array[T,N]` local arrays.
//!
//! AT-2937: local_histogram.axc — per-invocation 8-bin histogram, bit-exact
//!          (`max_diff == 0`) vs a trivial CPU reference. Runs on Lavapipe (CI)
//!          and NVIDIA RTX PRO 6000.
//! AT-2938: a read-modify-write-in-a-loop fixture proves no OpPhi is needed for
//!          local-array carry across loop iterations (the OpVariable persists;
//!          OpLoad/OpStore see the latest store — memory semantics, not SSA).
//!
//! Mirrors `dispatch_shared_m32.rs`'s style: `#[ignore]`-gated, `AXC_ENABLE_GPU_TESTS=1`
//! required at runtime (both the Lavapipe and NVIDIA legs use the SAME gated pattern
//! as every other GPU dispatch test in this suite).

use axc_driver::compile_source_with_meta;
use axc_runtime::VulkanContext;

fn gpu_tests_enabled() -> bool {
    std::env::var("AXC_ENABLE_GPU_TESTS").as_deref() == Ok("1")
}

/// CPU reference for `local_histogram.axc`'s algorithm: for each of
/// `num_invocations` (each owning a 16-element slice starting at `gid*16`),
/// bucket each `u32` value by `v % 8` into 8 per-invocation bins, written out
/// at `gid*8 + bin`.
fn cpu_local_histogram_reference(input: &[u32], num_invocations: usize) -> Vec<u32> {
    let mut out = vec![0u32; num_invocations * 8];
    for gid in 0..num_invocations {
        let base = gid * 16;
        let mut hist = [0u32; 8];
        for i in 0..16 {
            let v = input[base + i];
            let b = (v % 8) as usize;
            hist[b] += 1;
        }
        out[gid * 8..gid * 8 + 8].copy_from_slice(&hist);
    }
    out
}

/// AT-2937: local_histogram.axc bit-exact (`max_diff == 0`) vs the CPU reference.
///
/// `@workgroup(64,1,1)` with a `(1,1,1)` dispatch gives exactly 64 invocations
/// (64 * 16 = 1024 input elements consumed; 64 * 8 = 512 output u32 counts).
#[test]
#[ignore]
fn at2937_local_histogram_bit_exact_gpu() {
    if !gpu_tests_enabled() {
        eprintln!("AT-2937: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }

    let src = include_str!("../../../examples/local_histogram.axc");
    let (bytes, meta) = compile_source_with_meta(src).expect("local_histogram.axc must compile");
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let ctx = VulkanContext::new().expect("VulkanContext must init");
    eprintln!("AT-2937: device={}", ctx.physical_device_name());

    let handle = ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, None, "local_histogram",
        meta.shared_memory_bytes,
    ).unwrap_or_else(|e| panic!("AT-2937: pipeline create failed: {e}"));

    const NUM_INVOCATIONS: usize = 64;
    const NUM_INPUT: usize = NUM_INVOCATIONS * 16;
    const NUM_OUTPUT: usize = NUM_INVOCATIONS * 8;

    // Deterministic, non-trivial input pattern (not all-zero — exercises every bin).
    let input: Vec<u32> = (0..NUM_INPUT as u32).map(|i| i.wrapping_mul(2654435761) & 0xFFFF).collect();
    let input_bytes: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
    let output_bytes_len = NUM_OUTPUT * 4;

    let outputs = ctx.dispatch_handle(
        &handle, (1, 1, 1),
        &[&input_bytes, &vec![0u8; output_bytes_len][..]],
        &[0, output_bytes_len],
        &[],
    ).unwrap_or_else(|e| panic!("AT-2937: dispatch failed: {e}"));

    let output_bytes = &outputs[1];
    assert_eq!(output_bytes.len(), output_bytes_len);
    let gpu_result: Vec<u32> = output_bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let cpu_result = cpu_local_histogram_reference(&input, NUM_INVOCATIONS);
    assert_eq!(gpu_result.len(), cpu_result.len());

    let max_diff: i64 = gpu_result.iter().zip(cpu_result.iter())
        .map(|(&g, &c)| (g as i64 - c as i64).abs())
        .max()
        .unwrap_or(0);
    assert_eq!(
        max_diff, 0,
        "AT-2937: local_histogram must be bit-exact vs CPU reference; max_diff={max_diff}\n\
         gpu[0..8]={:?}\ncpu[0..8]={:?}", &gpu_result[0..8], &cpu_result[0..8]
    );
    eprintln!("AT-2937: PASS — local_histogram.axc is bit-exact on {}", ctx.physical_device_name());
}

/// AT-2938: a read-modify-write-in-a-loop fixture proves no OpPhi is needed for
/// local-array carry across loop iterations — `acc[j] = acc[j] + 1.0f32` run 10
/// times must yield `acc[j] == 10.0` for all j (the `OpVariable` persists across
/// iterations; `OpLoad`/`OpStore` see the latest store).
#[test]
#[ignore]
fn at2938_local_array_rmw_loop_no_opphi_needed_gpu() {
    if !gpu_tests_enabled() {
        eprintln!("AT-2938: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }

    let src = r#"
@kernel
@workgroup(1, 1, 1)
@intent("local-array RMW-in-a-loop — proves no OpPhi needed for array carry (M3.20 AT-2938)")
@complexity(O(1))
fn local_array_rmw_loop(out: buffer[f32]) -> void {
    array acc: array[f32, 4];
    for j in range(0u32, 4u32) {
        acc[j] = 0.0f32;
    }
    for i in range(0u32, 10u32) {
        for j in range(0u32, 4u32) {
            acc[j] = acc[j] + 1.0f32;
        }
    }
    for j in range(0u32, 4u32) {
        out[j] = acc[j];
    }
    return;
}
"#;
    let (bytes, meta) = compile_source_with_meta(src).expect("local_array_rmw_loop must compile");
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    // Zero OpPhi in the compiled module (structural corroboration alongside the
    // GPU numeric proof below).
    {
        use rspirv::dr::Loader;
        use rspirv::binary::Disassemble;
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&words, &mut loader).expect("rspirv parse");
        let asm = loader.module().disassemble();
        assert!(!asm.contains("OpPhi"), "AT-2938: local-array carry must need ZERO OpPhi; asm:\n{asm}");
    }

    let ctx = VulkanContext::new().expect("VulkanContext must init");
    eprintln!("AT-2938: device={}", ctx.physical_device_name());

    let handle = ctx.prepare_kernel_checked(
        &words, &meta.binding_plan, meta.push_constant_total_bytes,
        &meta.entry_point, None, "local_array_rmw_loop",
        meta.shared_memory_bytes,
    ).unwrap_or_else(|e| panic!("AT-2938: pipeline create failed: {e}"));

    let output_bytes_len = 4 * 4;
    let outputs = ctx.dispatch_handle(
        &handle, (1, 1, 1),
        &[&vec![0u8; output_bytes_len][..]],
        &[output_bytes_len],
        &[],
    ).unwrap_or_else(|e| panic!("AT-2938: dispatch failed: {e}"));

    let output_bytes = &outputs[0];
    assert_eq!(output_bytes.len(), output_bytes_len);
    let gpu_result: Vec<f32> = output_bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    for (j, &v) in gpu_result.iter().enumerate() {
        assert_eq!(v, 10.0_f32, "AT-2938: acc[{j}] must be exactly 10.0 after 10 RMW iterations; got {v}");
    }
    eprintln!("AT-2938: PASS — local-array RMW-in-a-loop is bit-exact (no OpPhi) on {}", ctx.physical_device_name());
}
