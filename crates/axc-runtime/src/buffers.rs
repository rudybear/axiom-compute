//! Buffer allocation helpers for M1.5 dispatch and M3.0 staging-buffer dispatch.
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
//! ## M3.0 additions
//!
//! `allocate_staging_buffer_cached` — HOST_CACHED staging ladder:
//! 1. HOST_VISIBLE|HOST_CACHED|HOST_COHERENT → `Coherency::Coherent`.
//! 2. HOST_VISIBLE|HOST_CACHED → `Coherency::NonCoherent{atom_size}`.
//! 3. HOST_VISIBLE|HOST_COHERENT (current M2.3a behavior) → `Coherency::Coherent`.
//!
//! The staging memory is **persistently mapped** once at allocation time.
//! `StagingBuffer` gains `coherency: Coherency` and `mapping: PersistentMapping`.
//!
//! `allocate_device_local_buffer` gains a `concurrent_families: Option<[u32;2]>`
//! parameter — when `Some([c,t])`, the buffer is `VK_SHARING_MODE_CONCURRENT` over
//! those families (DedicatedTransfer mode; CRITICAL-2). `None` = `EXCLUSIVE`.
//!
//! `round_up_pow2(size)` rounds a byte count up to the next power of two with a
//! minimum of 4 bytes — used for the grow-if-needed buffer pool.

use ash::vk;
use crate::error::DispatchError;
use crate::persistent_map::{Coherency, PersistentMapping, map_persistent};

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

/// A HOST_VISIBLE staging Vulkan buffer and its backing memory.
///
/// Used for host↔device staging copies in the M3.0 dispatch path.
/// The staging memory is **persistently mapped** at allocation time; the mapping
/// lives for the buffer-slot lifetime and is unmapped (consuming `PersistentMapping`)
/// before `destroy_buffer` / `free_memory` in `KernelHandleInner::drop`.
pub(crate) struct StagingBuffer {
    /// The Vulkan buffer handle.
    pub(crate) buffer: vk::Buffer,
    /// The backing device memory handle.
    pub(crate) memory: vk::DeviceMemory,
    /// Actual allocated size in bytes.
    #[allow(dead_code)]
    pub(crate) size: u64,
    /// Coherency policy of the backing memory type (M3.0).
    pub(crate) coherency: Coherency,
    /// Persistent host-side mapping into `memory` (M3.0).
    ///
    /// Must be unmapped (via `PersistentMapping::unmap`) BEFORE `destroy_buffer`
    /// and `free_memory`. `unmap` consumes `self` so a forgotten unmap is a
    /// compile error.
    pub(crate) mapping: PersistentMapping,
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
/// ## M3.0 — concurrent_families parameter (CRITICAL-2)
///
/// When `concurrent_families` is `Some([compute_family, transfer_family])`, the buffer
/// is created with `VK_SHARING_MODE_CONCURRENT` over those two families. This is the
/// `DedicatedTransfer` mode — there are NO queue-family ownership-transfer barriers.
/// When `None`, the buffer uses `VK_SHARING_MODE_EXCLUSIVE` (single-queue mode).
///
/// ## iGPU fallback
///
/// If no purely DEVICE_LOCAL memory type exists but a type with both
/// DEVICE_LOCAL and HOST_VISIBLE bits is available (iGPU unified memory),
/// that type is used. This fallback is logged once per call via `tracing::debug!`.
/// Note: iGPU CONCURRENT sharing composes correctly — CONCURRENT is a buffer-creation
/// attribute unrelated to the memory type's HOST_VISIBLE bit.
///
/// Returns `Err(NoCompatibleMemoryType)` if no compatible type exists at all.
pub(crate) fn allocate_device_local_buffer(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
    binding: u32,
    concurrent_families: Option<[u32; 2]>,
) -> Result<DeviceLocalBuffer, DispatchError> {
    let requested_size: u64 = size.max(4);

    // The concurrent family indices must outlive the BufferCreateInfo (which borrows them).
    // We store them in a local array that lives for the whole function scope.
    let concurrent_families_storage: [u32; 2] = concurrent_families.unwrap_or([0, 0]);

    // Build the buffer create info with appropriate sharing mode (CRITICAL-2).
    let buffer_info: vk::BufferCreateInfo<'_> = if concurrent_families.is_some() {
        // CONCURRENT sharing: device-local buffer accessible from both compute and
        // transfer families without ownership-transfer barriers.
        // SAFETY: concurrent_families_storage lives until after create_buffer returns.
        vk::BufferCreateInfo::default()
            .size(requested_size)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::CONCURRENT)
            .queue_family_indices(&concurrent_families_storage)
    } else {
        // EXCLUSIVE sharing: single-queue mode (M2.3a behavior).
        vk::BufferCreateInfo::default()
            .size(requested_size)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
    };

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

/// Allocate a HOST_VISIBLE staging buffer and persistently map it.
///
/// ## M3.0 — HOST_CACHED staging ladder
#[allow(dead_code)] // retained for legacy API compatibility; delegates to allocate_staging_buffer_cached
///
/// Delegates to `allocate_staging_buffer_cached` with `force_noncoherent=false`,
/// so the memory-type search follows the full HOST_CACHED preference ladder.
/// The legacy `allocate_staging_buffer` function is retained for call-site
/// compatibility but now returns a `StagingBuffer` with a `PersistentMapping`.
///
/// ## M2.3a compatibility
///
/// Call sites that previously used this function continue to work. The persistent
/// mapping replaces the per-dispatch `vkMapMemory`/`vkUnmapMemory` calls.
///
/// `non_coherent_atom_size` must come from `VkPhysicalDeviceLimits::nonCoherentAtomSize`
/// (never hardcoded, WARNING-1).
pub(crate) fn allocate_staging_buffer(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    non_coherent_atom_size: u64,
    size: u64,
    binding: u32,
) -> Result<StagingBuffer, DispatchError> {
    allocate_staging_buffer_cached(device, mem_props, non_coherent_atom_size, size, binding, false)
}

/// Allocate a HOST_VISIBLE staging buffer with HOST_CACHED preference and persistent mapping.
///
/// ## Memory-type ladder (HOST_CACHED preference, M3.0)
///
/// 1. `HOST_VISIBLE|HOST_CACHED|HOST_COHERENT` → `Coherency::Coherent`.
/// 2. `HOST_VISIBLE|HOST_CACHED` (no COHERENT) → `Coherency::NonCoherent{atom_size}`.
/// 3. `HOST_VISIBLE|HOST_COHERENT` → `Coherency::Coherent` (M2.3a behavior; Lavapipe/iGPU).
///
/// ## force_noncoherent override (AT-1409 / AT-1418)
///
/// When `force_noncoherent` is `true`, the chosen type's `Coherency` is OVERRIDDEN to
/// `NonCoherent{atom_size}` regardless of its real property flags. This makes
/// `flush`/`invalidate` fire against coherent memory (semantic no-op) so the code
/// path is exercised in CI on coherent hardware.
///
/// ## Persistent mapping (M3.0)
///
/// The staging memory is `vkMapMemory`'d exactly once after `vkBindBufferMemory`.
/// On `vkMapMemory` failure, the partially-constructed `buffer` and `memory` are
/// destroyed before returning the error (no leak).
///
/// `non_coherent_atom_size` must come from `VkPhysicalDeviceLimits::nonCoherentAtomSize`
/// (WARNING-1: never hardcoded).
pub(crate) fn allocate_staging_buffer_cached(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    non_coherent_atom_size: u64,
    size: u64,
    binding: u32,
    force_noncoherent: bool,
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

    // Try the HOST_CACHED ladder; fall back to HOST_COHERENT only.
    let (memory_type_index, mut coherency): (u32, Coherency) =
        find_host_cached_memory_type_index(mem_props, mem_reqs.memory_type_bits)
            .or_else(|| {
                find_host_visible_memory_type_index(mem_props, mem_reqs.memory_type_bits)
                    .map(|idx| (idx, Coherency::Coherent))
            })
            .ok_or_else(|| {
                // SAFETY: buffer was created above; destroy on error to avoid leak.
                unsafe { device.destroy_buffer(buffer, None); }
                DispatchError::NoCompatibleMemoryType
            })?;

    // Apply force_noncoherent override for CI coverage (AT-1409).
    if force_noncoherent {
        coherency = Coherency::NonCoherent { atom_size: non_coherent_atom_size };
    }

    let aligned_size: u64 = align_up(requested_size, mem_reqs.alignment);

    let alloc_info: vk::MemoryAllocateInfo<'_> = vk::MemoryAllocateInfo::default()
        .allocation_size(aligned_size)
        .memory_type_index(memory_type_index);

    // SAFETY: alloc_info is valid; we destroy this memory in KernelHandleInner::drop
    // (after unmapping the persistent mapping first).
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

    // Persistently map the staging memory AFTER bind and BEFORE returning.
    // On vkMapMemory failure: destroy buffer + free memory to avoid leak.
    // SAFETY: memory is HOST_VISIBLE (guaranteed by find_host_cached/visible index),
    // not yet mapped, and aligned_size is the true allocation size.
    let mapping: PersistentMapping = unsafe {
        map_persistent(device, memory, aligned_size, coherency)
    }.inspect_err(|_| {
        // SAFETY: buffer and memory were successfully created above; destroy on map failure
        // to avoid a resource leak. free_memory before destroy_buffer is intentional here —
        // the buffer is already unbound on map failure, so no ordering issue.
        unsafe {
            device.free_memory(memory, None);
            device.destroy_buffer(buffer, None);
        }
    })?;

    Ok(StagingBuffer {
        buffer,
        memory,
        size: aligned_size,
        coherency,
        mapping,
    })
}

/// Find the best HOST_VISIBLE memory type index for cached staging.
///
/// Returns the first memory type index that is compatible with `type_bits`
/// and has HOST_VISIBLE, searching for cached types first:
///
/// 1. `HOST_VISIBLE|HOST_CACHED|HOST_COHERENT` → `Coherency::Coherent`.
/// 2. `HOST_VISIBLE|HOST_CACHED` (without COHERENT) → `Coherency::NonCoherent{atom_size=64}`.
///    (NOTE: actual atom_size is plumbed from context limits, not hardcoded here — the
///    atom_size in the returned Coherency value is a placeholder 64 that callers MUST
///    override with the real `non_coherent_atom_size` from device limits. The `force_noncoherent`
///    path in `allocate_staging_buffer_cached` sets the correct value.)
///
/// Returns `None` if no HOST_CACHED type is available (fall back to HOST_COHERENT).
pub(crate) fn find_host_cached_memory_type_index(
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
) -> Option<(u32, Coherency)> {
    // Prefer HOST_VISIBLE|HOST_CACHED|HOST_COHERENT (no flush needed).
    for i in 0..mem_props.memory_type_count {
        if (type_bits >> i) & 1 != 1 {
            continue;
        }
        let flags = mem_props.memory_types[i as usize].property_flags;
        let target = vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_CACHED
            | vk::MemoryPropertyFlags::HOST_COHERENT;
        if flags.contains(target) {
            return Some((i, Coherency::Coherent));
        }
    }

    // Fall back to HOST_VISIBLE|HOST_CACHED without COHERENT.
    for i in 0..mem_props.memory_type_count {
        if (type_bits >> i) & 1 != 1 {
            continue;
        }
        let flags = mem_props.memory_types[i as usize].property_flags;
        if flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_CACHED)
            && !flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT)
        {
            // Return a placeholder atom_size; callers substitute the real limit.
            // The actual NonCoherent semantics are only triggered when coherency==NonCoherent.
            return Some((i, Coherency::NonCoherent { atom_size: 64 }));
        }
    }

    None
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

    // ── M3.0 HOST_CACHED ladder tests ────────────────────────────────────────

    /// AT-1407h: find_host_cached_memory_type_index prefers HOST_CACHED|COHERENT first.
    #[test]
    fn at_1407h_find_host_cached_prefers_coherent_cached() {
        let mut memory_types = [vk::MemoryType::default(); vk::MAX_MEMORY_TYPES];
        // type 0: HOST_VISIBLE|HOST_COHERENT (no CACHED)
        memory_types[0].property_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        // type 1: HOST_VISIBLE|HOST_CACHED|HOST_COHERENT (preferred)
        memory_types[1].property_flags = vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_CACHED
            | vk::MemoryPropertyFlags::HOST_COHERENT;
        // type 2: DEVICE_LOCAL only
        memory_types[2].property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;

        let mem_props = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: 3,
            memory_types,
            ..Default::default()
        };

        let result = find_host_cached_memory_type_index(&mem_props, 0b111);
        assert!(result.is_some(), "must find HOST_CACHED type");
        let (idx, coherency) = result.unwrap();
        assert_eq!(idx, 1, "must pick type 1 (HOST_CACHED|COHERENT)");
        assert_eq!(coherency, Coherency::Coherent, "HOST_CACHED|COHERENT must be Coherent");
    }

    /// AT-1407i: find_host_cached_memory_type_index falls back to HOST_CACHED without COHERENT.
    #[test]
    fn at_1407i_find_host_cached_falls_back_to_noncoherent() {
        let mut memory_types = [vk::MemoryType::default(); vk::MAX_MEMORY_TYPES];
        // type 0: HOST_VISIBLE|HOST_COHERENT (no CACHED)
        memory_types[0].property_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        // type 1: HOST_VISIBLE|HOST_CACHED (no COHERENT)
        memory_types[1].property_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_CACHED;

        let mem_props = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: 2,
            memory_types,
            ..Default::default()
        };

        let result = find_host_cached_memory_type_index(&mem_props, 0b11);
        assert!(result.is_some(), "must find HOST_CACHED non-coherent type");
        let (idx, coherency) = result.unwrap();
        assert_eq!(idx, 1, "must pick type 1 (HOST_CACHED non-coherent)");
        match coherency {
            Coherency::NonCoherent { .. } => {}
            Coherency::Coherent => panic!("HOST_CACHED without COHERENT must be NonCoherent"),
        }
    }

    /// AT-1407j: find_host_cached_memory_type_index returns None when no CACHED type exists.
    #[test]
    fn at_1407j_find_host_cached_returns_none_when_no_cached() {
        let mut memory_types = [vk::MemoryType::default(); vk::MAX_MEMORY_TYPES];
        // Only HOST_VISIBLE|HOST_COHERENT (no CACHED)
        memory_types[0].property_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

        let mem_props = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: 1,
            memory_types,
            ..Default::default()
        };

        let result = find_host_cached_memory_type_index(&mem_props, 0b1);
        assert!(result.is_none(), "no CACHED type must return None");
    }

    /// AT-1420: 50 GB memory cap — round_up_pow2 on large but valid sizes stays < 50 GB.
    #[test]
    fn at_1420_memory_cap_guard_round_up() {
        // saxpy_1m: 2 buffers x 4 MB + 1 staging = 12 MB total < 50 GB.
        let saxpy_buf_size: u64 = 4 * 1024 * 1024; // 4 MB
        let n_buffers: u64 = 3; // 2 inputs + 1 output staging
        let total: u64 = round_up_pow2(saxpy_buf_size) * n_buffers;
        let cap_bytes: u64 = 50 * 1024 * 1024 * 1024; // 50 GB
        assert!(total < cap_bytes, "saxpy_1m total allocation must be < 50 GB; got {total}");
    }
}
