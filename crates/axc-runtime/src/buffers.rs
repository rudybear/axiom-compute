//! Buffer allocation helpers for M1.5 dispatch and M2.3a staging-buffer dispatch.
//!
//! ## M1.5 (legacy one-shot path)
//!
//! `allocate_host_visible_buffer` allocates a single HOST_VISIBLE|HOST_COHERENT
//! buffer per binding. Used by the legacy `dispatch()` wrapper.
//!
//! ## M2.3a (prepare-once / dispatch-many path)
//!
//! `allocate_device_local_buffer` allocates DEVICE_LOCAL storage (STORAGE +
//! TRANSFER_SRC|DST). `allocate_staging_buffer` allocates HOST_VISIBLE|HOST_COHERENT
//! staging (TRANSFER_SRC|DST). Both are used by `KernelHandleInner` per binding.
//!
//! `round_up_pow2(size)` rounds a byte count up to the next power of two with a
//! minimum of 4 bytes — used for the grow-if-needed buffer pool.

use ash::vk;
use crate::error::DispatchError;

/// A host-visible, host-coherent Vulkan buffer with its backing device memory.
///
/// Created by `allocate_host_visible_buffer`; destroyed by the caller via
/// `DispatchResources` RAII (which calls `destroy_buffer` then `free_memory`
/// in dependency-correct order).
///
/// Retained for legacy compatibility (M1.5 one-shot path).
#[allow(dead_code)]
pub(crate) struct HostVisibleBuffer {
    /// The Vulkan buffer handle.
    pub(crate) buffer: vk::Buffer,
    /// The backing device memory handle.
    pub(crate) memory: vk::DeviceMemory,
    /// Actual allocated size in bytes (may be rounded up to alignment).
    pub(crate) size: u64,
}

/// Allocate a `HOST_VISIBLE | HOST_COHERENT` buffer of at least `size` bytes.
///
/// Steps:
/// 1. Create a `VkBuffer` with `STORAGE_BUFFER` usage.
/// 2. Query memory requirements (to get `alignment` and `memory_type_bits`).
/// 3. Find a compatible `HOST_VISIBLE | HOST_COHERENT` memory type.
/// 4. Allocate `VkDeviceMemory` of `max(size, alignment)` rounded to `alignment`.
/// 5. Bind the buffer to the memory.
///
/// Returns `DispatchError::NoCompatibleMemoryType` if no coherent memory type
/// is available (M2 follow-up: staging-buffer fallback for mobile GPUs).
///
/// Retained for legacy compatibility (M1.5 one-shot dispatch path).
#[allow(dead_code)]
pub(crate) fn allocate_host_visible_buffer(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
    binding: u32,
) -> Result<HostVisibleBuffer, DispatchError> {
    // At least 4 bytes so zero-length buffers don't trip Vulkan validation.
    let requested_size: u64 = size.max(4);

    let buffer_info = vk::BufferCreateInfo::default()
        .size(requested_size)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    // SAFETY: device is valid and buffer_info is correctly populated.
    let buffer: vk::Buffer = unsafe { device.create_buffer(&buffer_info, None) }
        .map_err(|e| DispatchError::BufferAllocationFailed {
            binding,
            size: requested_size as usize,
            reason: e.to_string(),
        })?;

    // SAFETY: buffer was just created successfully.
    let mem_reqs: vk::MemoryRequirements = unsafe { device.get_buffer_memory_requirements(buffer) };

    let memory_type_index: u32 =
        find_host_visible_memory_type_index(mem_props, mem_reqs.memory_type_bits)
            .ok_or(DispatchError::NoCompatibleMemoryType)?;

    // Round up to alignment (NOT a hardcoded 4-byte rounding — W1 fix).
    let aligned_size: u64 = align_up(requested_size, mem_reqs.alignment);

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(aligned_size)
        .memory_type_index(memory_type_index);

    // SAFETY: alloc_info is valid; we destroy this memory in DispatchResources::drop.
    let memory: vk::DeviceMemory = unsafe { device.allocate_memory(&alloc_info, None) }
        .map_err(|e| {
            // SAFETY: buffer was created above; we must destroy it on error to avoid leak.
            unsafe { device.destroy_buffer(buffer, None); }
            DispatchError::BufferAllocationFailed {
                binding,
                size: aligned_size as usize,
                reason: format!("allocate_memory: {e}"),
            }
        })?;

    // SAFETY: memory and buffer are valid. Offset 0 satisfies alignment (whole allocation).
    unsafe { device.bind_buffer_memory(buffer, memory, 0) }
        .map_err(|e| {
            // SAFETY: clean up both handles on bind failure.
            unsafe {
                device.free_memory(memory, None);
                device.destroy_buffer(buffer, None);
            }
            DispatchError::BufferAllocationFailed {
                binding,
                size: aligned_size as usize,
                reason: format!("bind_buffer_memory: {e}"),
            }
        })?;

    Ok(HostVisibleBuffer {
        buffer,
        memory,
        size: aligned_size,
    })
}

/// Find the index of a memory type that is `HOST_VISIBLE | HOST_COHERENT`.
///
/// Iterates `memory_types[0..memory_type_count]` in order and returns the first
/// type index `i` where:
/// - `(type_bits >> i) & 1 == 1` (compatible with the buffer's requirements)
/// - `property_flags` contains `HOST_VISIBLE | HOST_COHERENT`
///
/// Returns `None` if no such type is available (M1.5 hard-requires coherent
/// host-visible memory; non-coherent fallback is an M2 follow-up).
pub(crate) fn find_host_visible_memory_type_index(
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
) -> Option<u32> {
    let required_flags: vk::MemoryPropertyFlags =
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

    for i in 0..mem_props.memory_type_count {
        if (type_bits >> i) & 1 != 1 {
            continue;
        }
        let flags: vk::MemoryPropertyFlags =
            mem_props.memory_types[i as usize].property_flags;
        if flags.contains(required_flags) {
            return Some(i);
        }
    }
    None
}

/// Round `value` up to the next multiple of `alignment`.
///
/// `alignment` must be a power of two (guaranteed by Vulkan spec for
/// `VkMemoryRequirements::alignment`).
fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    (value + alignment - 1) & !(alignment - 1)
}

// ── M2.3a: device-local + staging buffer allocation ──────────────────────────

/// A DEVICE_LOCAL Vulkan buffer and its backing device memory.
///
/// Created by `allocate_device_local_buffer`; destroyed by `KernelHandleInner::drop`
/// (which calls `destroy_buffer` then `free_memory` in dependency-correct order).
pub(crate) struct DeviceLocalBuffer {
    /// The Vulkan buffer handle.
    pub(crate) buffer: vk::Buffer,
    /// The backing device memory handle.
    pub(crate) memory: vk::DeviceMemory,
    /// Actual allocated size in bytes.
    pub(crate) size: u64,
}

/// A HOST_VISIBLE|HOST_COHERENT staging Vulkan buffer and its backing memory.
///
/// Used for host↔device staging copies in the M2.3a dispatch path.
pub(crate) struct StagingBuffer {
    /// The Vulkan buffer handle.
    pub(crate) buffer: vk::Buffer,
    /// The backing device memory handle.
    pub(crate) memory: vk::DeviceMemory,
    /// Actual allocated size in bytes.
    pub(crate) size: u64,
}

/// Round `size` up to the next power of two with a minimum of 4 bytes.
///
/// Used for grow-if-needed buffer pool sizing so buffers are always powers-of-two
/// and at least 4 bytes (prevents Vulkan validation errors on zero-size buffers).
///
/// Saturating behaviour: sizes >= `1 << 63` (more than half of u64::MAX) return
/// `u64::MAX` to avoid panic. The caller should reject such sizes before allocation.
pub(crate) fn round_up_pow2(size: u64) -> u64 {
    if size <= 4 {
        return 4;
    }
    // next_power_of_two on u64 panics for values > 2^63. Saturate instead.
    if size > (1u64 << 62) {
        return u64::MAX;
    }
    size.next_power_of_two()
}

/// Allocate a DEVICE_LOCAL buffer suitable for compute shader storage + DMA transfers.
///
/// The buffer is created with usage `STORAGE_BUFFER | TRANSFER_SRC | TRANSFER_DST`
/// and backed by `DEVICE_LOCAL` memory.
///
/// ## iGPU fallback
///
/// If no purely DEVICE_LOCAL memory type exists but a type with both
/// DEVICE_LOCAL and HOST_VISIBLE bits is available (iGPU unified memory),
/// that type is used. This fallback is logged once per call via `tracing::debug!`.
///
/// Returns `Err(NoCompatibleMemoryType)` if no compatible type exists at all.
pub(crate) fn allocate_device_local_buffer(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
    binding: u32,
) -> Result<DeviceLocalBuffer, DispatchError> {
    let requested_size: u64 = size.max(4);

    let buffer_info: vk::BufferCreateInfo<'_> = vk::BufferCreateInfo::default()
        .size(requested_size)
        .usage(
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    // SAFETY: device is valid; buffer_info is correctly populated.
    let buffer: vk::Buffer = unsafe { device.create_buffer(&buffer_info, None) }
        .map_err(|e| DispatchError::BufferAllocationFailed {
            binding,
            size: requested_size as usize,
            reason: e.to_string(),
        })?;

    // SAFETY: buffer was just created successfully.
    let mem_reqs: vk::MemoryRequirements =
        unsafe { device.get_buffer_memory_requirements(buffer) };

    let memory_type_index: u32 =
        find_device_local_memory_type_index(mem_props, mem_reqs.memory_type_bits)
            .ok_or(DispatchError::NoCompatibleMemoryType)?;

    let aligned_size: u64 = align_up(requested_size, mem_reqs.alignment);

    let alloc_info: vk::MemoryAllocateInfo<'_> = vk::MemoryAllocateInfo::default()
        .allocation_size(aligned_size)
        .memory_type_index(memory_type_index);

    // SAFETY: alloc_info is valid; we destroy this memory in KernelHandleInner::drop.
    let memory: vk::DeviceMemory = unsafe { device.allocate_memory(&alloc_info, None) }
        .map_err(|e| {
            // SAFETY: buffer was created above; destroy on error to avoid leak.
            unsafe { device.destroy_buffer(buffer, None); }
            DispatchError::BufferAllocationFailed {
                binding,
                size: aligned_size as usize,
                reason: format!("allocate_memory: {e}"),
            }
        })?;

    // SAFETY: memory and buffer are valid. Offset 0 satisfies alignment.
    unsafe { device.bind_buffer_memory(buffer, memory, 0) }
        .map_err(|e| {
            // SAFETY: clean up both handles on bind failure.
            unsafe {
                device.free_memory(memory, None);
                device.destroy_buffer(buffer, None);
            }
            DispatchError::BufferAllocationFailed {
                binding,
                size: aligned_size as usize,
                reason: format!("bind_buffer_memory: {e}"),
            }
        })?;

    Ok(DeviceLocalBuffer {
        buffer,
        memory,
        size: aligned_size,
    })
}

/// Allocate a HOST_VISIBLE|HOST_COHERENT staging buffer for DMA transfers.
///
/// The buffer is created with usage `TRANSFER_SRC | TRANSFER_DST`
/// and backed by HOST_VISIBLE|HOST_COHERENT memory (no flush required).
pub(crate) fn allocate_staging_buffer(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
    binding: u32,
) -> Result<StagingBuffer, DispatchError> {
    let requested_size: u64 = size.max(4);

    let buffer_info: vk::BufferCreateInfo<'_> = vk::BufferCreateInfo::default()
        .size(requested_size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    // SAFETY: device is valid; buffer_info is correctly populated.
    let buffer: vk::Buffer = unsafe { device.create_buffer(&buffer_info, None) }
        .map_err(|e| DispatchError::BufferAllocationFailed {
            binding,
            size: requested_size as usize,
            reason: e.to_string(),
        })?;

    // SAFETY: buffer was just created successfully.
    let mem_reqs: vk::MemoryRequirements =
        unsafe { device.get_buffer_memory_requirements(buffer) };

    let memory_type_index: u32 =
        find_host_visible_memory_type_index(mem_props, mem_reqs.memory_type_bits)
            .ok_or(DispatchError::NoCompatibleMemoryType)?;

    let aligned_size: u64 = align_up(requested_size, mem_reqs.alignment);

    let alloc_info: vk::MemoryAllocateInfo<'_> = vk::MemoryAllocateInfo::default()
        .allocation_size(aligned_size)
        .memory_type_index(memory_type_index);

    // SAFETY: alloc_info is valid; we destroy this memory in KernelHandleInner::drop.
    let memory: vk::DeviceMemory = unsafe { device.allocate_memory(&alloc_info, None) }
        .map_err(|e| {
            // SAFETY: buffer was created above; destroy on error to avoid leak.
            unsafe { device.destroy_buffer(buffer, None); }
            DispatchError::BufferAllocationFailed {
                binding,
                size: aligned_size as usize,
                reason: format!("allocate_memory: {e}"),
            }
        })?;

    // SAFETY: memory and buffer are valid. Offset 0 satisfies alignment.
    unsafe { device.bind_buffer_memory(buffer, memory, 0) }
        .map_err(|e| {
            // SAFETY: clean up both handles on bind failure.
            unsafe {
                device.free_memory(memory, None);
                device.destroy_buffer(buffer, None);
            }
            DispatchError::BufferAllocationFailed {
                binding,
                size: aligned_size as usize,
                reason: format!("bind_buffer_memory: {e}"),
            }
        })?;

    Ok(StagingBuffer {
        buffer,
        memory,
        size: aligned_size,
    })
}

/// Find a DEVICE_LOCAL (optionally also HOST_VISIBLE, for iGPU) memory type index.
///
/// Preference order:
/// 1. DEVICE_LOCAL without HOST_VISIBLE (discrete GPU, fast VRAM).
/// 2. DEVICE_LOCAL | HOST_VISIBLE (iGPU unified memory — one allocation serves both).
///
/// Returns `None` if no compatible type is available.
pub(crate) fn find_device_local_memory_type_index(
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
) -> Option<u32> {
    // Prefer purely DEVICE_LOCAL first (discrete GPUs).
    for i in 0..mem_props.memory_type_count {
        if (type_bits >> i) & 1 != 1 {
            continue;
        }
        let flags: vk::MemoryPropertyFlags =
            mem_props.memory_types[i as usize].property_flags;
        if flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            && !flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        {
            return Some(i);
        }
    }

    // Fallback: unified iGPU memory (DEVICE_LOCAL | HOST_VISIBLE).
    for i in 0..mem_props.memory_type_count {
        if (type_bits >> i) & 1 != 1 {
            continue;
        }
        let flags: vk::MemoryPropertyFlags =
            mem_props.memory_types[i as usize].property_flags;
        if flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            && flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        {
            tracing::debug!(
                binding_type_index = i,
                "using unified iGPU memory type for device-local buffer"
            );
            return Some(i);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk;

    /// AT-803: round_up_pow2 — basic cases.
    #[test]
    fn at_803_round_up_pow2_basic() {
        assert_eq!(round_up_pow2(0), 4, "zero must round to minimum 4");
        assert_eq!(round_up_pow2(1), 4, "1 must round to minimum 4");
        assert_eq!(round_up_pow2(4), 4, "4 is already pow2 minimum");
        assert_eq!(round_up_pow2(5), 8, "5 rounds up to 8");
        assert_eq!(round_up_pow2(8), 8, "8 is already pow2");
        assert_eq!(round_up_pow2(9), 16, "9 rounds up to 16");
        assert_eq!(round_up_pow2(1024), 1024, "1024 is pow2");
        assert_eq!(round_up_pow2(1025), 2048, "1025 rounds up to 2048");
        assert_eq!(round_up_pow2(1_048_576), 1_048_576, "1M is pow2");
        assert_eq!(round_up_pow2(1_048_577), 2_097_152, "1M+1 rounds up");
    }

    /// AT-803b: round_up_pow2 saturates on very large values.
    #[test]
    fn at_803b_round_up_pow2_saturates() {
        // Values > 2^62 should return u64::MAX rather than panicking.
        let large: u64 = (1u64 << 62) + 1;
        assert_eq!(round_up_pow2(large), u64::MAX, "over-large must saturate");
    }

    /// AT-804: find_device_local_memory_type_index — discrete GPU (no unified).
    #[test]
    fn at_804_find_device_local_discrete_gpu() {
        let mut memory_types = [vk::MemoryType::default(); vk::MAX_MEMORY_TYPES];
        // type 0: HOST_VISIBLE | HOST_COHERENT only (no DEVICE_LOCAL)
        memory_types[0].property_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        // type 1: DEVICE_LOCAL only (discrete VRAM)
        memory_types[1].property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;

        let mem_props = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: 2,
            memory_types,
            ..Default::default()
        };

        let result: Option<u32> = find_device_local_memory_type_index(&mem_props, 0b11);
        assert_eq!(result, Some(1), "should pick DEVICE_LOCAL type at index 1");
    }

    /// AT-804b: find_device_local_memory_type_index — iGPU unified (DEVICE_LOCAL|HOST_VISIBLE).
    #[test]
    fn at_804b_find_device_local_igpu_unified() {
        let mut memory_types = [vk::MemoryType::default(); vk::MAX_MEMORY_TYPES];
        // type 0: HOST_VISIBLE | HOST_COHERENT only (no DEVICE_LOCAL)
        memory_types[0].property_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        // type 1: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT (iGPU unified)
        memory_types[1].property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL
            | vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT;

        let mem_props = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: 2,
            memory_types,
            ..Default::default()
        };

        // type_bits = 0b11 (both types compatible)
        let result: Option<u32> = find_device_local_memory_type_index(&mem_props, 0b11);
        assert_eq!(result, Some(1), "should pick DEVICE_LOCAL|HOST_VISIBLE unified type at index 1");
    }

    /// AT-804c: find_device_local_memory_type_index — no compatible type → None.
    #[test]
    fn at_804c_find_device_local_none_when_unavailable() {
        let mut memory_types = [vk::MemoryType::default(); vk::MAX_MEMORY_TYPES];
        // type 0: HOST_VISIBLE | HOST_COHERENT only (no DEVICE_LOCAL)
        memory_types[0].property_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

        let mem_props = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: 1,
            memory_types,
            ..Default::default()
        };

        let result: Option<u32> = find_device_local_memory_type_index(&mem_props, 0b1);
        assert_eq!(result, None, "no DEVICE_LOCAL type must return None");
    }

    /// AT-504: find_host_visible_memory_type_index positive and negative cases.
    ///
    /// Positive: type 0 = DEVICE_LOCAL only; type 1 = HOST_VISIBLE|HOST_COHERENT;
    /// type_bits = 0b11 → returns Some(1).
    ///
    /// Negative: type_bits = 0b01 (only type 0 compatible, type 0 is DEVICE_LOCAL) → returns None.
    #[test]
    fn at_504_find_host_visible_memory_type_index_positive_and_negative() {
        let mut memory_types = [vk::MemoryType::default(); vk::MAX_MEMORY_TYPES];
        // type 0: DEVICE_LOCAL only — no HOST_VISIBLE
        memory_types[0].property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        memory_types[0].heap_index = 0;
        // type 1: HOST_VISIBLE | HOST_COHERENT — what we need
        memory_types[1].property_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        memory_types[1].heap_index = 1;

        let mem_props = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: 2,
            memory_types,
            ..Default::default()
        };

        // Positive: both types compatible (type_bits = 0b11), should pick type 1
        let result = find_host_visible_memory_type_index(&mem_props, 0b11);
        assert_eq!(result, Some(1), "should find HOST_VISIBLE|HOST_COHERENT type at index 1");

        // Negative: only type 0 compatible (type_bits = 0b01), type 0 is DEVICE_LOCAL only
        let result = find_host_visible_memory_type_index(&mem_props, 0b01);
        assert_eq!(result, None, "should return None when only DEVICE_LOCAL type is compatible");
    }

    #[test]
    fn align_up_power_of_two() {
        assert_eq!(align_up(0, 256), 0);
        assert_eq!(align_up(1, 256), 256);
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(257, 256), 512);
        assert_eq!(align_up(16, 16), 16);
        assert_eq!(align_up(17, 16), 32);
    }

    #[test]
    fn align_up_zero_alignment_passthrough() {
        // Zero alignment is degenerate but must not panic.
        assert_eq!(align_up(42, 0), 42);
    }
}
