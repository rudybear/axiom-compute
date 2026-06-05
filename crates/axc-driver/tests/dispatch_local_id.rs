//! GPU dispatch tests for local_invocation_id() builtin (M3.3d).
//!
//! AT-1742: local_id_probe.axc dispatched as 4 workgroups of 64 threads returns
//!          out[g*64+l]==l for all g in 0..4, l in 0..64 (exact integer).
//!          Runs on Lavapipe AND NVIDIA. NOT typed-skipped for coopmat.

use axc_driver::compile_source_with_meta;
use axc_runtime::{VulkanContext, DispatchError};

fn gpu_tests_enabled() -> bool {
    std::env::var("AXC_ENABLE_GPU_TESTS").as_deref() == Ok("1")
}

const LOCAL_ID_PROBE_SRC: &str = include_str!("../../../examples/local_id_probe.axc");

/// AT-1742: local_invocation_id() GPU bit-exact oracle.
///
/// Dispatches local_id_probe.axc as G=4 workgroups of 64 threads (grid (4,1,1)).
/// Asserts out[g*64 + l] == l for all g in 0..4, l in 0..64.
/// This is the real-GPU proof that local_invocation_id() returns the correct
/// per-invocation LocalInvocationId (within-workgroup local id along x).
///
/// Runs on Lavapipe too (no coopmat required). NOT typed-skipped.
#[test]
#[ignore]
fn at1742_local_invocation_id_gpu_bit_exact() {
    if !gpu_tests_enabled() {
        eprintln!("AT-1742: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }

    let (bytes, meta) = compile_source_with_meta(LOCAL_ID_PROBE_SRC)
        .expect("AT-1742: local_id_probe.axc must compile");
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let ctx = VulkanContext::new().expect("AT-1742: VulkanContext must init");
    eprintln!("AT-1742: device={}", ctx.physical_device_name());

    // Prepare kernel — no coopmat check needed (pure scalar u32 output).
    let handle = ctx.prepare_kernel(
        &words,
        &meta.binding_plan,
        meta.push_constant_total_bytes,
        &meta.entry_point,
    ).expect("AT-1742: pipeline create must succeed");

    // G=4 workgroups of 64 threads each → 256 output elements (u32).
    const G: usize = 4;
    const WG_SIZE: usize = 64;
    const N: usize = G * WG_SIZE; // 256 elements

    let output_size_bytes: usize = N * 4; // u32 = 4 bytes each

    let outputs = ctx.dispatch_handle(
        &handle,
        (G as u32, 1, 1),
        &[&vec![0u8; output_size_bytes]],
        &[output_size_bytes],
        &[],
    );

    let outputs = match outputs {
        Ok(v) => v,
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
            eprintln!("AT-1742: DeviceFeatureUnsupported({feature}) for {kernel} — typed-skip");
            return;
        }
        Err(e) => panic!("AT-1742: unexpected dispatch error: {e:?}"),
    };

    let out_bytes = &outputs[0];
    assert_eq!(
        out_bytes.len(), output_size_bytes,
        "AT-1742: output must be {output_size_bytes} bytes ({N} u32 elements)"
    );

    // Parse u32 values from LE bytes.
    let out_u32: Vec<u32> = out_bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(out_u32.len(), N, "AT-1742: expected {N} u32 values");

    // Verify: out[g*64 + l] == l for all g in 0..4, l in 0..64.
    let mut max_error: u32 = 0;
    let mut error_count: usize = 0;
    for g in 0..G {
        for l in 0..WG_SIZE {
            let idx = g * WG_SIZE + l;
            let expected = l as u32;
            let actual = out_u32[idx];
            if actual != expected {
                if error_count < 5 {
                    eprintln!(
                        "AT-1742: out[g={g} l={l}] = {actual} != {expected} (expected local_id={l})"
                    );
                }
                let diff = actual.abs_diff(expected);
                if diff > max_error { max_error = diff; }
                error_count += 1;
            }
        }
    }

    eprintln!(
        "AT-1742: G={G} workgroups, WG_SIZE={WG_SIZE} threads, N={N} total, \
         error_count={error_count}, max_error={max_error}, \
         first4={:?}",
        &out_u32[..4]
    );

    assert!(
        error_count == 0 && max_error == 0,
        "AT-1742: local_invocation_id() GPU FAILED — {error_count} errors, max_error={max_error}.\n\
         HONESTY GATE: do NOT relax — local_invocation_id() must return the exact within-WG id.\n\
         Grid=(4,1,1), WG_SIZE=64, expected out[g*64+l]==l for all g,l."
    );

    eprintln!(
        "AT-1742 PASS: local_invocation_id() returns correct within-WG local id (exact) \
         on {} — G={G} WGs, WG_SIZE={WG_SIZE}, all {N} elements match",
        ctx.physical_device_name()
    );
}
