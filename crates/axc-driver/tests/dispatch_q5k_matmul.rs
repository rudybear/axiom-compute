//! M3.14b — GPU correctness tests for the portable (non-coopmat) Q5_K_M dequant+matmul kernel
//! (examples/q5k_dequant_matmul.axc).
//!
//! AT-2819: K=256 (n_blocks_per_row=1) non-symmetric fixture with qh bits SET, bit-exact
//!          (max_diff==0) vs q5k_oracle::q5k_dequant_matmul_f32_cpu (mirror AT-1520). NOT
//!          #[ignore] — runs on Lavapipe AND NVIDIA. All-unsigned, all-integer fixture design
//!          (d is a power of two; sc/m/nib5/x are small integers), so every partial sum is
//!          exactly representable in f32 and additions introduce ZERO rounding error regardless
//!          of summation order.
//! AT-2820: large-K (n_blocks_per_row=8, K=2048), combined ≤ 1e-3 vs
//!          q5k_oracle::q5k_dequant_matmul_f32accum_cpu. NVIDIA-only, #[ignore]-gated.

use axc_driver::{compile_source_with_meta, q5k_oracle};
use axc_runtime::{VulkanContext, DispatchError};
use axc_hir::ParamBindingPlan;
use half::f16;

const Q5K_MATMUL_SRC: &str = include_str!("../../../examples/q5k_dequant_matmul.axc");

fn gpu_tests_enabled() -> bool {
    std::env::var("AXC_ENABLE_GPU_TESTS").as_deref() == Ok("1")
}

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

/// AT-2819 fixture: EXACT-INTEGER Q5_K_M weights + x (all-unsigned, so no sign-reconstruction
/// concerns). d/dmin are powers of two (f16-exact); the packed 6-bit scales/mins are small
/// integers; qh bytes are non-trivial (exercises SET bits); x values are small integers. Every
/// intermediate product/sum is an exact multiple of a fixed power-of-two scale, well below the
/// f32 exact-integer ceiling, so max_diff==0 holds regardless of summation order.
fn make_q5k_exact_integer_fixture(n_rows: u32, n_cols: u32) -> (Vec<u8>, Vec<f32>) {
    let n_rows = n_rows as usize;
    let n_cols = n_cols as usize;
    let n_bpr = 1usize;
    let mut q: Vec<u8> = vec![0u8; n_rows * n_bpr * q5k_oracle::Q5K_SUPERBLOCK_BYTES];

    for row in 0..n_rows {
        let base = row * q5k_oracle::Q5K_SUPERBLOCK_BYTES;

        // d = 0.25, dmin = 0.0625 (both f16-exact powers of two).
        let d_bits = f16::from_f32(0.25).to_bits();
        q[base..base + 2].copy_from_slice(&d_bits.to_le_bytes());
        let dmin_bits = f16::from_f32(0.0625).to_bits();
        q[base + 2..base + 4].copy_from_slice(&dmin_bits.to_le_bytes());

        // scales[12]: j<4 path direct (sc,m in 0..63); keep small + non-symmetric per row.
        for j in 0..4usize {
            let sc: u8 = ((row + j * 5 + 3) % 16) as u8; // 0..15
            let m: u8 = ((row * 2 + j * 3 + 1) % 16) as u8; // 0..15
            q[base + 4 + j] = sc & 0x3F;
            q[base + 4 + j + 4] = m & 0x3F;
        }
        // j>=4 path: lo4_sc/lo4_m packed into bytes[8..12]; keep zero hi-bits (bytes[0..4] hi2=0
        // already satisfied since sc/m above are < 16 < 64, top 2 bits clear).
        for jj in 0..4usize {
            let sc4: u8 = ((row + jj * 7 + 2) % 16) as u8;
            let m4: u8 = ((row * 3 + jj + 1) % 16) as u8;
            q[base + 4 + 8 + jj] = (sc4 & 0x0F) | ((m4 & 0x0F) << 4);
        }

        // qh[32]: non-trivial pattern (exercises SET bits across all 8 positions).
        for i in 0..32usize {
            q[base + 16 + i] = ((row * 13 + i * 5 + 7) & 0xFF) as u8;
        }
        // qs[128]: small deterministic nibble pattern (0..15 both nibbles).
        for i in 0..128usize {
            let lo: u8 = ((row + i * 3) % 16) as u8;
            let hi: u8 = ((row * 2 + i * 7 + 5) % 16) as u8;
            q[base + 48 + i] = lo | (hi << 4);
        }
    }

    // x: column-of-output-major layout, small integers in [-6, 6].
    let k_total = n_bpr * q5k_oracle::Q5K_SUPERBLOCK_ELEMS;
    let mut x: Vec<f32> = vec![0.0_f32; k_total * n_cols];
    for k in 0..k_total {
        for col in 0..n_cols {
            let v: i64 = (((k * 7 + col * 3 + 1) % 13) as i64) - 6;
            x[k * n_cols + col] = v as f32;
        }
    }

    (q, x)
}

fn make_q5k_random_fixture(n_rows: u32, n_bpr: u32) -> Vec<u8> {
    let n_rows = n_rows as usize;
    let n_bpr = n_bpr as usize;
    let mut q: Vec<u8> = vec![0u8; n_rows * n_bpr * q5k_oracle::Q5K_SUPERBLOCK_BYTES];
    let mut rng = Xorshift::new(0xC0FFEE_u64 ^ (n_rows as u64));

    for row in 0..n_rows {
        for sb in 0..n_bpr {
            let base = (row * n_bpr + sb) * q5k_oracle::Q5K_SUPERBLOCK_BYTES;
            let d = 0.02_f32 + ((rng.next() % 16) as f32) * 0.002;
            let dmin = 0.01_f32 + ((rng.next() % 8) as f32) * 0.001;
            q[base..base + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
            q[base + 2..base + 4].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
            for j in 0..12 {
                q[base + 4 + j] = (rng.next() & 0x3F) as u8;
            }
            for i in 0..32 {
                q[base + 16 + i] = (rng.next() & 0xFF) as u8;
            }
            for i in 0..128 {
                q[base + 48 + i] = (rng.next() & 0xFF) as u8;
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
        x_f32.push(f16::from_bits(bits).to_f32());
    }
    (x_f32, x_f16)
}

/// AT-2819: q5k_dequant_matmul.axc bit-exact (max_diff==0) vs q5k_dequant_matmul_f32_cpu at
/// K=256, qh bits set. NOT #[ignore] — runs on Lavapipe AND NVIDIA.
#[test]
fn at_2819_q5k_matmul_k256_bit_exact() {
    if !gpu_tests_enabled() {
        eprintln!("at_2819: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    let ctx = VulkanContext::new().expect("VulkanContext must init");
    eprintln!("at_2819: device={}", ctx.physical_device_name());

    let n_rows: u32 = 64;
    let n_cols: u32 = 64;
    let n_blocks_per_row: u32 = 1;

    let (q_bytes, x_f32) = make_q5k_exact_integer_fixture(n_rows, n_cols);
    let x_bytes: Vec<u8> = x_f32.iter().flat_map(|&v| v.to_le_bytes()).collect();

    // Sanity: fixture actually exercises SET qh bits (else a broken hi=0-always kernel could pass).
    assert!(
        q_bytes.iter().skip(16).take(32).any(|&b| b != 0),
        "AT-2819 fixture sanity: qh bytes must contain at least one SET bit"
    );

    let y_ref = q5k_oracle::q5k_dequant_matmul_f32_cpu(
        &q_bytes, &x_f32, n_rows as usize, n_cols as usize, n_blocks_per_row as usize,
    );

    let (bytes, meta) = compile_source_with_meta(Q5K_MATMUL_SRC)
        .expect("q5k_dequant_matmul.axc must compile");
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let handle = match ctx.prepare_kernel(&words, &meta.binding_plan, meta.push_constant_total_bytes, &meta.entry_point) {
        Ok(h) => h,
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
            eprintln!("at_2819: DeviceFeatureUnsupported {feature} for {kernel}; skipping");
            return;
        }
        Err(e) => panic!("at_2819: pipeline create failed: {e:?}"),
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
            eprintln!("at_2819: DeviceFeatureUnsupported at dispatch: {feature} / {kernel}; skipping");
            return;
        }
        Err(e) => panic!("at_2819: dispatch failed: {e:?}"),
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
    eprintln!("at_2819: max_diff={max_diff} at idx={max_idx} (gpu={}, ref={})", y_gpu[max_idx], y_ref[max_idx]);
    assert_eq!(
        max_diff, 0.0_f32,
        "AT-2819: q5k_dequant_matmul.axc must be BIT-EXACT (max_diff==0) vs the CPU oracle on \
         the exact-integer fixture; got max_diff={max_diff} at idx={max_idx} \
         (gpu={}, ref={})", y_gpu[max_idx], y_ref[max_idx]
    );
    eprintln!("at_2819: PASS — bit-exact on {}", ctx.physical_device_name());
}

/// AT-2820: q5k_dequant_matmul.axc combined ≤ 1e-3 vs q5k_dequant_matmul_f32accum_cpu at
/// large K (n_blocks_per_row=8, K=2048). NVIDIA-only, #[ignore]-gated.
#[test]
#[ignore]
fn at_2820_q5k_matmul_k2048_combined_within_tol() {
    if !gpu_tests_enabled() {
        eprintln!("at_2820: AXC_ENABLE_GPU_TESTS not set; skipping");
        return;
    }
    let ctx = VulkanContext::new().expect("VulkanContext must init");
    eprintln!("at_2820: device={}", ctx.physical_device_name());

    let n_rows: u32 = 32;
    let n_cols: u32 = 32;
    let n_blocks_per_row: u32 = 8; // K = 2048

    let q_bytes = make_q5k_random_fixture(n_rows, n_blocks_per_row);
    let k_total = n_blocks_per_row as usize * q5k_oracle::Q5K_SUPERBLOCK_ELEMS;
    let (x_f32, x_f16) = make_x_f16_exact(k_total, n_cols as usize, 0xF00DBEEF ^ 0x5555);
    let x_bytes: Vec<u8> = x_f32.iter().flat_map(|&v| v.to_le_bytes()).collect();

    let y_ref = q5k_oracle::q5k_dequant_matmul_f32accum_cpu(
        &q_bytes, &x_f16, n_rows as usize, n_cols as usize, n_blocks_per_row as usize,
    );
    let abs_scale = q5k_oracle::q5k_abs_dot_scale(
        &q_bytes, &x_f16, n_rows as usize, n_cols as usize, n_blocks_per_row as usize,
    );

    let (bytes, meta) = compile_source_with_meta(Q5K_MATMUL_SRC)
        .expect("q5k_dequant_matmul.axc must compile");
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let handle = match ctx.prepare_kernel(&words, &meta.binding_plan, meta.push_constant_total_bytes, &meta.entry_point) {
        Ok(h) => h,
        Err(DispatchError::DeviceFeatureUnsupported { feature, kernel }) => {
            eprintln!("at_2820: DeviceFeatureUnsupported {feature} for {kernel}; skipping");
            return;
        }
        Err(e) => panic!("at_2820: pipeline create failed: {e:?}"),
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
            eprintln!("at_2820: DeviceFeatureUnsupported at dispatch: {feature} / {kernel}; skipping");
            return;
        }
        Err(e) => panic!("at_2820: dispatch failed: {e:?}"),
    };

    let y_bytes: &[u8] = &outputs[2];
    assert_eq!(y_bytes.len(), output_y_size);
    let y_gpu: Vec<f32> = y_bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let raw_rel = axc_driver::q4km_oracle::max_rel_diff(&y_gpu, &y_ref);
    let combined = axc_driver::q4km_oracle::max_rel_diff_combined(&y_gpu, &y_ref, &abs_scale);
    eprintln!(
        "at_2820: MEASURED raw max_rel_diff={raw_rel:.3e} | COMBINED (condition-aware, the GATE) \
         ={combined:.3e} (n_rows={n_rows} n_cols={n_cols} K={k_total}) on {}",
        ctx.physical_device_name()
    );
    assert!(
        combined <= 1e-3,
        "AT-2820: q5k_dequant_matmul.axc combined (condition-aware) max_rel_diff={combined:.3e} \
         exceeds frozen rtol 1e-3 at K={k_total} (raw={raw_rel:.3e})"
    );
    eprintln!("at_2820: PASS — combined within frozen 1e-3 on {}", ctx.physical_device_name());
}
