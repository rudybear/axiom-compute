//! M3.14a — GPU correctness tests for the portable (non-coopmat) Q6_K dequant+matmul kernel
//! (examples/q6k_dequant_matmul.axc).
//!
//! AT-2814: K=256 (n_blocks_per_row=1), bit-exact (max_diff==0) vs
//!          q6k_oracle::q6k_dequant_matmul_f32_cpu. NOT #[ignore] — runs on Lavapipe AND
//!          NVIDIA. The fixture uses an EXACT-INTEGER design (d is a power of two; sc, q, and x
//!          are all small integers) so every partial sum in BOTH the kernel and the CPU oracle
//!          is exactly representable in f32 — additions introduce ZERO rounding error REGARDLESS
//!          of summation order, making max_diff==0 hold robustly (not merely by luck of a
//!          matching op order between GPU codegen and the CPU loop nest).
//! AT-2815: large-K (n_blocks_per_row=8, K=2048) with random signed scales, combined ≤ 1e-3 vs
//!          q6k_oracle::q6k_dequant_matmul_f32accum_cpu (the f16-staged, coopmat-comparable
//!          oracle — per the M3.14 spec §6 table; reusing this oracle avoids a THIRD combined-
//!          metric-compatible reference and the f16-rounding delta it introduces is <1e-3, well
//!          inside the frozen gate). Both metrics reported. NVIDIA-only, #[ignore]-gated.

use axc_driver::{compile_source_with_meta, q6k_oracle};
use axc_runtime::{VulkanContext, DispatchError};
use axc_hir::ParamBindingPlan;
use half::f16;

const Q6K_MATMUL_SRC: &str = include_str!("../../../examples/q6k_dequant_matmul.axc");

fn gpu_tests_enabled() -> bool {
    std::env::var("AXC_ENABLE_GPU_TESTS").as_deref() == Ok("1")
}

/// Assemble push constants by scalar name (robust to declaration-order changes).
fn assemble_pc(plan: &ParamBindingPlan, n_rows: u32, n_cols: u32, n_blocks_per_row: u32) -> Vec<u8> {
    let mut pc: Vec<u8> = vec![0u8; plan.push_constant_total_bytes as usize];
    for s in &plan.scalars {
        let val: u32 = match s.name.as_str() {
            "n_rows" => n_rows,
            "n_cols" => n_cols,
            "n_blocks_per_row" => n_blocks_per_row,
            other => panic!("unexpected scalar param {other}"),
        };
        let start = s.offset as usize;
        pc[start..start + 4].copy_from_slice(&val.to_le_bytes());
    }
    pc
}

/// Deterministic xorshift64 PRNG (matches the seed scheme used throughout the M3.x test suite).
struct Xorshift(u64);
impl Xorshift {
    fn new(seed: u64) -> Self { Xorshift(seed | 1) }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// AT-2814 fixture: EXACT-INTEGER Q6_K weights + x, so f32 accumulation is order-independent.
///
/// Per row: d = 0.25 (power of two, f16-exact); scales[16] = small signed integers in
/// [-8, 8] (includes negatives); ql/qh bytes are pseudo-random (spanning the full q_u range
/// 0..63). x values are small integers in [-6, 6]. Every intermediate product/sum is an exact
/// multiple of 0.25 with magnitude far below 2^24/4, so NO floating-point rounding ever occurs
/// — bit-exactness holds regardless of summation order.
fn make_q6k_exact_integer_fixture(n_rows: u32, n_cols: u32) -> (Vec<u8>, Vec<f32>) {
    let n_rows = n_rows as usize;
    let n_cols = n_cols as usize;
    let n_bpr = 1usize;
    let mut q: Vec<u8> = vec![0u8; n_rows * n_bpr * q6k_oracle::Q6K_SUPERBLOCK_BYTES];

    for row in 0..n_rows {
        let base = row * q6k_oracle::Q6K_SUPERBLOCK_BYTES;
        let mut rng = Xorshift::new(0xC0FFEE_u64 ^ (row as u64) ^ 1);

        // d = 0.25 (f16-exact power of two), fixed across rows to keep magnitude analysis simple.
        let d_bits = f16::from_f32(0.25).to_bits();
        q[base + 208] = d_bits.to_le_bytes()[0];
        q[base + 209] = d_bits.to_le_bytes()[1];

        // scales[16]: signed integers in [-8, 8], deterministic + non-symmetric per row.
        for i in 0..16usize {
            let v: i32 = (((row * 3 + i * 5 + 2) % 17) as i32) - 8; // [-8, 8]
            q[base + 192 + i] = (v as i8) as u8;
        }

        // ql[128] @ base+0, qh[64] @ base+128: pseudo-random bytes -> q_u spans 0..63.
        for i in 0..128usize {
            q[base + i] = (rng.next() & 0xFF) as u8;
        }
        for i in 0..64usize {
            q[base + 128 + i] = (rng.next() & 0xFF) as u8;
        }
    }

    // x: column-of-output-major layout, small integers in [-6, 6] (exact in f32).
    let k_total = n_bpr * q6k_oracle::Q6K_SUPERBLOCK_ELEMS;
    let mut x: Vec<f32> = vec![0.0_f32; k_total * n_cols];
    for k in 0..k_total {
        for col in 0..n_cols {
            let v: i64 = (((k * 7 + col * 3 + 1) % 13) as i64) - 6; // [-6, 6]
            x[k * n_cols + col] = v as f32;
        }
    }

    (q, x)
}

/// AT-2815 fixture: large-K random signed-scale Q6_K weights + f16-exact x (mirrors the M3.x
/// combined-metric fixture style; x is rounded through f16 so the f16accum oracle's x-side
/// rounding introduces ZERO additional delta, isolating the comparison to weight f16-rounding).
fn make_q6k_random_fixture(n_rows: u32, n_bpr: u32) -> Vec<u8> {
    let n_rows = n_rows as usize;
    let n_bpr = n_bpr as usize;
    let mut q: Vec<u8> = vec![0u8; n_rows * n_bpr * q6k_oracle::Q6K_SUPERBLOCK_BYTES];
    let mut rng = Xorshift::new(0xBADC0DE_u64 ^ (n_rows as u64));

    for row in 0..n_rows {
        for sb in 0..n_bpr {
            let base = (row * n_bpr + sb) * q6k_oracle::Q6K_SUPERBLOCK_BYTES;
            let d = 0.01_f32 + ((rng.next() % 16) as f32) * 0.002; // small quantization-like scale
            let d_bits = f16::from_f32(d).to_bits();
            q[base + 208] = d_bits.to_le_bytes()[0];
            q[base + 209] = d_bits.to_le_bytes()[1];
            for i in 0..16usize {
                q[base + 192 + i] = (rng.next() & 0xFF) as u8; // random signed scales
            }
            for i in 0..128usize {
                q[base + i] = (rng.next() & 0xFF) as u8;
            }
            for i in 0..64usize {
                q[base + 128 + i] = (rng.next() & 0xFF) as u8;
            }
        }
    }
    q
}

fn make_x_f16_exact(k: usize, n: usize, seed: u64) -> (Vec<f32>, Vec<u16>) {
    let mut rng = Xorshift::new(seed);
    let mut x_f32: Vec<f32> = Vec::with_capacity(k * n);
    let mut x_f16: Vec<u16> = Vec::with_capacity(k * n);
    for idx in 0..k * n {
        let v = (((rng.next() % 2000) as f32) / 1000.0 - 1.0) + (idx % 3) as f32 * 0.01;
        let bits = f16::from_f32(v).to_bits();
        x_f16.push(bits);
        x_f32.push(f16::from_bits(bits).to_f32()); // f16-exact f32 value fed to the GPU kernel.
    }
    (x_f32, x_f16)
}

/// AT-2814: q6k_dequant_matmul.axc bit-exact (max_diff==0) vs q6k_dequant_matmul_f32_cpu at
/// K=256. NOT #[ignore] — runs on Lavapipe AND NVIDIA under AXC_ENABLE_GPU_TESTS=1.
#[test]
fn at_2814_q6k_matmul_k256_bit_exact() {
    if !gpu_tests_enabled() {
        eprintln!("at_2814: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    let ctx = VulkanContext::new().expect("VulkanContext must init");
    eprintln!("at_2814: device={}", ctx.physical_device_name());

    let n_rows: u32 = 64;
    let n_cols: u32 = 64;
    let n_blocks_per_row: u32 = 1;

    let (q_bytes, x_f32) = make_q6k_exact_integer_fixture(n_rows, n_cols);
    let x_bytes: Vec<u8> = x_f32.iter().flat_map(|&v| v.to_le_bytes()).collect();

    let y_ref = q6k_oracle::q6k_dequant_matmul_f32_cpu(
        &q_bytes, &x_f32, n_rows as usize, n_cols as usize, n_blocks_per_row as usize,
    );

    let (bytes, meta) = compile_source_with_meta(Q6K_MATMUL_SRC)
        .expect("q6k_dequant_matmul.axc must compile");
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let handle = match ctx.prepare_kernel(&words, &meta.binding_plan, meta.push_constant_total_bytes, &meta.entry_point) {
        Ok(h) => h,
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
            eprintln!("at_2814: DeviceFeatureUnsupported {feature} for {kernel}; skipping");
            return;
        }
        Err(e) => panic!("at_2814: pipeline create failed: {e:?}"),
    };

    let pc_bytes = assemble_pc(&meta.binding_plan, n_rows, n_cols, n_blocks_per_row);
    let output_y_size = (n_rows * n_cols) as usize * 4;
    let workgroups = ((n_rows * n_cols).div_ceil(64), 1_u32, 1_u32);

    let outputs = match ctx.dispatch_handle(
        &handle, workgroups,
        &[&q_bytes, &x_bytes, &vec![0u8; output_y_size]],
        &[0, 0, output_y_size],
        &pc_bytes,
    ) {
        Ok(v) => v,
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
            eprintln!("at_2814: DeviceFeatureUnsupported at dispatch: {feature} / {kernel}; skipping");
            return;
        }
        Err(e) => panic!("at_2814: dispatch failed: {e:?}"),
    };

    assert!(outputs[0].is_empty(), "q (ReadOnly) must produce empty output (Lever A)");
    assert!(outputs[1].is_empty(), "x (ReadOnly) must produce empty output (Lever A)");
    let y_bytes: &[u8] = &outputs[2];
    assert_eq!(y_bytes.len(), output_y_size);
    let y_gpu: Vec<f32> = y_bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let mut max_diff: f32 = 0.0_f32;
    let mut max_idx: usize = 0;
    for (idx, (&g, &r)) in y_gpu.iter().zip(y_ref.iter()).enumerate() {
        let d = (g - r).abs();
        if d > max_diff {
            max_diff = d;
            max_idx = idx;
        }
    }
    eprintln!("at_2814: max_diff={max_diff} at idx={max_idx} (gpu={}, ref={})", y_gpu[max_idx], y_ref[max_idx]);
    assert_eq!(
        max_diff, 0.0_f32,
        "AT-2814: q6k_dequant_matmul.axc must be BIT-EXACT (max_diff==0) vs the CPU oracle on \
         the exact-integer fixture; got max_diff={max_diff} at idx={max_idx} \
         (gpu={}, ref={})", y_gpu[max_idx], y_ref[max_idx]
    );
    eprintln!("at_2814: PASS — bit-exact on {}", ctx.physical_device_name());
}

/// AT-2815: q6k_dequant_matmul.axc combined ≤ 1e-3 vs q6k_dequant_matmul_f32accum_cpu on a
/// large-K (n_blocks_per_row=8, K=2048) fixture with random signed scales. NVIDIA-only,
/// #[ignore]-gated.
#[test]
#[ignore]
fn at_2815_q6k_matmul_k2048_combined_within_tol() {
    if !gpu_tests_enabled() {
        eprintln!("at_2815: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    let ctx = VulkanContext::new().expect("VulkanContext must init");
    eprintln!("at_2815: device={}", ctx.physical_device_name());

    let n_rows: u32 = 32;
    let n_cols: u32 = 32;
    let n_blocks_per_row: u32 = 8; // K = 2048

    let q_bytes = make_q6k_random_fixture(n_rows, n_blocks_per_row);
    let k_total = n_blocks_per_row as usize * q6k_oracle::Q6K_SUPERBLOCK_ELEMS;
    let (x_f32, x_f16) = make_x_f16_exact(k_total, n_cols as usize, 0xF00DBEEF);
    let x_bytes: Vec<u8> = x_f32.iter().flat_map(|&v| v.to_le_bytes()).collect();

    let y_ref = q6k_oracle::q6k_dequant_matmul_f32accum_cpu(
        &q_bytes, &x_f16, n_rows as usize, n_cols as usize, n_blocks_per_row as usize,
    );
    let abs_scale = q6k_oracle::q6k_abs_dot_scale(
        &q_bytes, &x_f16, n_rows as usize, n_cols as usize, n_blocks_per_row as usize,
    );

    let (bytes, meta) = compile_source_with_meta(Q6K_MATMUL_SRC)
        .expect("q6k_dequant_matmul.axc must compile");
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let handle = match ctx.prepare_kernel(&words, &meta.binding_plan, meta.push_constant_total_bytes, &meta.entry_point) {
        Ok(h) => h,
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
            eprintln!("at_2815: DeviceFeatureUnsupported {feature} for {kernel}; skipping");
            return;
        }
        Err(e) => panic!("at_2815: pipeline create failed: {e:?}"),
    };

    let pc_bytes = assemble_pc(&meta.binding_plan, n_rows, n_cols, n_blocks_per_row);
    let output_y_size = (n_rows * n_cols) as usize * 4;
    let workgroups = ((n_rows * n_cols).div_ceil(64), 1_u32, 1_u32);

    let outputs = match ctx.dispatch_handle(
        &handle, workgroups,
        &[&q_bytes, &x_bytes, &vec![0u8; output_y_size]],
        &[0, 0, output_y_size],
        &pc_bytes,
    ) {
        Ok(v) => v,
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
            eprintln!("at_2815: DeviceFeatureUnsupported at dispatch: {feature} / {kernel}; skipping");
            return;
        }
        Err(e) => panic!("at_2815: dispatch failed: {e:?}"),
    };

    let y_bytes: &[u8] = &outputs[2];
    assert_eq!(y_bytes.len(), output_y_size);
    let y_gpu: Vec<f32> = y_bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let raw_rel = axc_driver::q4km_oracle::max_rel_diff(&y_gpu, &y_ref);
    let combined = axc_driver::q4km_oracle::max_rel_diff_combined(&y_gpu, &y_ref, &abs_scale);
    eprintln!(
        "at_2815: MEASURED raw max_rel_diff={raw_rel:.3e} | COMBINED (condition-aware, the GATE) \
         ={combined:.3e} (n_rows={n_rows} n_cols={n_cols} K={k_total}) on {}",
        ctx.physical_device_name()
    );
    assert!(
        combined <= 1e-3,
        "AT-2815: q6k_dequant_matmul.axc combined (condition-aware) max_rel_diff={combined:.3e} \
         exceeds frozen rtol 1e-3 at K={k_total} (raw={raw_rel:.3e})"
    );
    eprintln!("at_2815: PASS — combined within frozen 1e-3 on {}", ctx.physical_device_name());
}
