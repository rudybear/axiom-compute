//! GPU-resident benchmark primitive — M3.1.
//!
//! The thesis-relevant metric for llama.cpp inference is: upload weights ONCE to
//! resident VRAM, dispatch N times, measure KERNEL-ONLY time. This module provides
//! that primitive.
//!
//! ## API overview
//!
//! ```rust,ignore
//! let resident = ctx.upload_resident(&handle, &inputs, &output_sizes)?;
//! let timing = ctx.dispatch_resident(&handle, &resident, &cfg)?;
//! let outputs = ctx.readback_resident(&handle, &resident, &output_sizes)?;
//! ```
//!
//! `upload_resident` copies inputs into device-local buffers once and binds the
//! descriptor set to those buffers. Each `dispatch_resident` records ONLY the
//! compute submit with timestamp queries (no upload/readback), giving kernel-only
//! timing.
//!
//! ## Timing methodology
//!
//! When `timestampValidBits > 0` and `timestampPeriod > 0`:
//! - `vkCmdWriteTimestamp(TOP_OF_PIPE, pool, 0)` before dispatch
//! - `vkCmdWriteTimestamp(BOTTOM_OF_PIPE, pool, 1)` after dispatch
//! - `elapsed_ns = masked_timestamp_delta(begin, end, valid_bits) * timestamp_period`
//! - `timing_source = GpuTimestamp`
//!
//! Fallback (invalid bits or zero period):
//! - CPU wall time measured via `std::time::Instant` around queue submit + fence wait
//! - `timing_source = CpuFenceWall`
//!
//! The `cmd_reset_query_pool` is recorded BEFORE the first `write_timestamp` in the
//! SAME submission (per HN-6). The N-iteration loop REUSES one command buffer, fence,
//! and query pool (no per-iter allocation).
//!
//! ## Safety
//!
//! All `unsafe` blocks have `// SAFETY:` comments.

use ash::vk;
use std::sync::Arc;
use crate::device_owner::DeviceOwner;
use crate::buffers::{DeviceLocalBuffer, StagingBuffer};

/// Configuration for a single `dispatch_resident` call.
pub struct ResidentBenchConfig {
    /// Workgroup dispatch dimensions (x, y, z).
    pub workgroups: (u32, u32, u32),
    /// Push constant bytes to push before dispatch.
    pub push_constants: Vec<u8>,
}

/// Source of the kernel timing measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentTimingSource {
    /// GPU timestamp query (vkCmdWriteTimestamp). Accurate kernel-only time.
    GpuTimestamp,
    /// CPU fence-wall time (std::time::Instant around submit+wait). Includes scheduling.
    CpuFenceWall,
}

/// Timing result from `dispatch_resident`.
#[derive(Debug, Clone)]
pub struct ResidentDispatchTiming {
    /// Kernel execution time in nanoseconds.
    pub kernel_ns: u64,
    /// Source of the timing measurement.
    pub timing_source: ResidentTimingSource,
}

/// GPU-resident buffer set for repeated dispatch.
///
/// Created by `VulkanContext::upload_resident`. Owns device-local buffers,
/// a persistent descriptor set bound to those buffers, one reusable command
/// buffer, one fence, and one timestamp query pool.
///
/// Destroyed when dropped — resources freed in Drop.
#[allow(dead_code)] // Fields used in Drop + future dispatch_resident implementation
pub struct ResidentBuffers {
    /// Arc to keep the device alive.
    pub(crate) device: Arc<DeviceOwner>,
    /// Device-local input/output buffers (indexed by binding slot).
    pub(crate) device_local_bufs: Vec<DeviceLocalBuffer>,
    /// Staging buffers for readback (output slots only).
    pub(crate) staging_bufs: Vec<Option<StagingBuffer>>,
    /// Descriptor set bound to the device-local buffers.
    pub(crate) descriptor_set: Option<vk::DescriptorSet>,
    /// Descriptor pool owning the descriptor set.
    pub(crate) descriptor_pool: Option<vk::DescriptorPool>,
    /// Reusable command buffer for dispatch recording.
    pub(crate) command_buffer: vk::CommandBuffer,
    /// Command pool owning the command buffer.
    pub(crate) command_pool: vk::CommandPool,
    /// Reusable fence for GPU sync.
    pub(crate) fence: vk::Fence,
    /// 2-entry timestamp query pool for kernel timing.
    pub(crate) query_pool: vk::QueryPool,
    /// Timestamp period in nanoseconds per tick (0.0 if unavailable).
    pub(crate) timestamp_period: f32,
    /// Number of valid timestamp bits (0 if unavailable).
    pub(crate) timestamp_valid_bits: u32,
    /// Queue family index for command pool.
    pub(crate) queue_family_index: u32,
    /// Whether these buffers need shaderInt16 (for diagnostics).
    #[allow(dead_code)]
    pub(crate) uses_coopmat: bool,
}

/// Compute the masked timestamp delta, handling wrap across the valid-bit boundary.
///
/// Both `begin` and `end` are masked to `valid_bits` FIRST, then `end.wrapping_sub(begin)`
/// is computed. This handles the case where the counter wraps between begin and end.
///
/// Per HN-6: mask EACH endpoint first, then wrapping_sub (AT-1535).
#[allow(dead_code)] // Used in dispatch_resident timing (AT-1535); tests confirm correctness
pub(crate) fn masked_timestamp_delta(begin: u64, end: u64, valid_bits: u32) -> u64 {
    if valid_bits == 0 || valid_bits >= 64 {
        // If all 64 bits are valid (or the field is zero), no masking needed.
        return end.wrapping_sub(begin);
    }
    let mask: u64 = (1u64 << valid_bits).wrapping_sub(1);
    let masked_begin = begin & mask;
    let masked_end = end & mask;
    masked_end.wrapping_sub(masked_begin) & mask
}

impl Drop for ResidentBuffers {
    fn drop(&mut self) {
        let device: &ash::Device = &self.device.device;
        // SAFETY: all GPU work has completed (fence-wait invariant on callers);
        // resources are owned exclusively by this struct.
        unsafe {
            // Wait for any in-flight work.
            let _ = device.wait_for_fences(&[self.fence], true, u64::MAX);

            // Destroy query pool.
            device.destroy_query_pool(self.query_pool, None);

            // Destroy fence.
            device.destroy_fence(self.fence, None);

            // Destroy descriptor pool (implicitly frees descriptor set).
            if let Some(pool) = self.descriptor_pool {
                device.destroy_descriptor_pool(pool, None);
            }

            // Destroy command pool (implicitly frees command buffer).
            device.destroy_command_pool(self.command_pool, None);

            // Free device-local buffers.
            for buf in &self.device_local_bufs {
                device.destroy_buffer(buf.buffer, None);
                device.free_memory(buf.memory, None);
            }

            // Free staging buffers (unmapping not needed — staging here is for readback only).
            for stg in self.staging_bufs.iter().flatten() {
                device.destroy_buffer(stg.buffer, None);
                device.free_memory(stg.memory, None);
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// AT-1535: masked_timestamp_delta handles wrap across the valid-bit boundary.
    ///
    /// Example: valid_bits=32, begin near 2^32-1, end is small (wrapped around).
    #[test]
    fn at_1535_masked_timestamp_delta_boundary() {
        // Case: valid_bits=32, counter wraps from near max to small value.
        let valid_bits: u32 = 32;
        let begin: u64 = 0xFFFF_FFFF - 10; // near max of 32-bit range
        let end: u64 = 5;                   // after wrap

        // Without masking: end.wrapping_sub(begin) would be a huge number.
        // With masking: (5 - (0xFFFFFFFF - 10)) mod 2^32 = 16 (small, sane).
        let delta = masked_timestamp_delta(begin, end, valid_bits);
        assert_eq!(delta, 16, "wrapped delta must be 16 (10 + 1 + 5)");

        // Case: no wrap (end > begin in the valid range).
        let begin2: u64 = 100;
        let end2: u64 = 200;
        let delta2 = masked_timestamp_delta(begin2, end2, valid_bits);
        assert_eq!(delta2, 100, "simple delta must be 100");

        // Case: valid_bits=0 → no masking, plain wrapping_sub.
        let delta3 = masked_timestamp_delta(200, 100, 0);
        assert_eq!(delta3, u64::MAX - 99, "valid_bits=0 means no mask, plain wrapping_sub");

        // Case: valid_bits=64 → no masking (>= 64 means full 64-bit counter).
        let delta4 = masked_timestamp_delta(100, 200, 64);
        assert_eq!(delta4, 100, "valid_bits=64 means plain wrapping_sub");
    }
}
