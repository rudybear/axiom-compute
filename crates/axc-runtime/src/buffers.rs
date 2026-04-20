//! Host-visible coherent buffer allocation for M1.5 dispatch.
//!
//! M1.5 uses a simplified memory strategy: one Vulkan buffer per descriptor
//! binding, allocated in HOST_VISIBLE | HOST_COHERENT memory. This avoids
//! the need for staging buffers and explicit `vkFlushMappedMemoryRanges` /
//! `vkInvalidateMappedMemoryRanges` calls.
//!
//! Non-coherent host-visible memory (e.g. some mobile GPUs) is an M2 follow-up;
//! M1.5 returns `DispatchError::NoCompatibleMemoryType` on those devices.

use ash::vk;
use crate::error::DispatchError;

/// A host-visible, host-coherent Vulkan buffer with its backing device memory.
///
/// Created by `allocate_host_visible_buffer`; destroyed by the caller via
/// `DispatchResources` RAII (which calls `destroy_buffer` then `free_memory`
/// in dependency-correct order).
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

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk;

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
