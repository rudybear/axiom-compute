//! Vulkan-side external-memory + external-semaphore EXPORT — M4.1 (Phase 1a).
//!
//! This module is the RUST HALF of the CUDA↔Vulkan zero-copy interop. It knows
//! nothing about CUDA: it only EXPORTS OS file descriptors (OPAQUE_FD) for
//! Vulkan device memory and Vulkan semaphores. The CUDA IMPORT
//! (`cudaImportExternalMemory` / `cudaImportExternalSemaphore`) is done on the
//! Python side, which is the SINGLE owner of the handed-off fds (G-3).
//!
//! ## What it provides
//!
//! 1. [`allocate_exportable_device_local_buffer`] — a DEVICE_LOCAL buffer that is
//!    EXPORTABLE as OPAQUE_FD and is **unconditionally DEDICATED**
//!    (`VkMemoryDedicatedAllocateInfo{buffer}` chained, G-1). One
//!    `VkDeviceMemory` per buffer, whole-allocation offset-0 bind (no
//!    suballocation), so the dedicated export is natural and correct.
//! 2. [`export_memory_fd`] — `vkGetMemoryFdKHR` → an [`OwnedFd`].
//! 3. [`create_exportable_timeline_semaphore`] — a TIMELINE semaphore created with
//!    `VkExportSemaphoreCreateInfo(OPAQUE_FD)` and exported via
//!    `vkGetSemaphoreFdKHR`.
//! 4. [`create_exportable_binary_semaphore_pair`] — the G-6 fallback used when the
//!    timeline-over-fd interop spike fails on the driver.
//!
//! ## G-1 dedicated bit (CRITICAL)
//!
//! The fd does NOT carry the dedication bit. [`ExternalBuffer::dedicated`] returns
//! `true` (always, for now) so the Python side sets
//! `cudaExternalMemoryHandleDesc.flags = cudaExternalMemoryDedicated` to MATCH the
//! Vulkan `VkMemoryDedicatedAllocateInfo`. A mismatch fails the import (fail-closed,
//! OK) or silently aliases wrong memory (NOT OK) — hence the bit is plumbed
//! out-of-band.
//!
//! ## G-3 fd ownership
//!
//! `vkGetMemoryFdKHR` transfers OPAQUE_FD ownership to the app. We hold it as an
//! [`OwnedFd`] until [`ExternalBuffer::take_memory_fd_raw`] hands the raw `i32` to
//! Python via `into_raw_fd` (RELEASING Rust ownership). Rust does NOT Drop-close
//! the handed-off fd. If the fd is never taken (error path before hand-off), the
//! `OwnedFd` closes it on Drop exactly once.

use std::os::fd::{IntoRawFd, OwnedFd, FromRawFd};

use ash::vk;

use crate::buffers::find_device_local_memory_type_index;
use crate::error::DispatchError;

/// Capabilities relevant to CUDA↔Vulkan external interop, queried at context init.
#[derive(Debug, Clone)]
pub struct ExternalInteropCaps {
    /// Whether `VK_KHR_external_memory_fd` is enabled on the device.
    pub external_memory_fd: bool,
    /// Whether `VK_KHR_external_semaphore_fd` is enabled on the device.
    pub external_semaphore_fd: bool,
    /// Whether `timelineSemaphore` was forced ON for the external semaphore.
    pub timeline_semaphore: bool,
    /// The selected device's `VkPhysicalDeviceIDProperties.deviceUUID` (16 raw bytes).
    pub device_uuid: [u8; 16],
    /// The selected device's `VkPhysicalDeviceIDProperties.driverUUID` (16 raw bytes).
    pub driver_uuid: [u8; 16],
}

/// A DEVICE_LOCAL, dedicated, OPAQUE_FD-exportable Vulkan buffer + its exported fd.
///
/// The buffer is bound to its own whole `VkDeviceMemory` at offset 0 and is
/// dedicated (G-1). The exported memory fd is owned by `memory_fd` until
/// [`take_memory_fd_raw`](Self::take_memory_fd_raw) hands it to Python (G-3).
///
/// `vkDestroyBuffer` + `vkFreeMemory` are the caller's responsibility (the PyO3
/// layer frees them in the G-4 teardown order, AFTER the CUDA side is destroyed).
pub struct ExternalBuffer {
    /// The Vulkan buffer handle (bound to `memory` at offset 0).
    pub buffer: vk::Buffer,
    /// The backing dedicated device memory handle.
    pub memory: vk::DeviceMemory,
    /// Actual allocated size in bytes (== requested, aligned).
    allocation_size: u64,
    /// The descriptor binding index this buffer is for.
    pub binding: u32,
    /// The exported OPAQUE_FD for `memory`, owned until handed off to Python (G-3).
    ///
    /// `Some` until `take_memory_fd_raw` is called; `None` afterwards (ownership
    /// transferred — Rust will NOT close it). If still `Some` at Drop, the
    /// `OwnedFd` closes it exactly once.
    memory_fd: Option<OwnedFd>,
    /// G-1: always `true` for now (the allocation is unconditionally dedicated).
    dedicated: bool,
}

impl ExternalBuffer {
    /// G-1: whether the allocation is dedicated (always `true` for now).
    ///
    /// Plumbed to Python so it sets `cudaExternalMemoryHandleDesc.flags =
    /// cudaExternalMemoryDedicated` to MATCH the Vulkan side.
    pub fn dedicated(&self) -> bool {
        self.dedicated
    }

    /// Allocation size in bytes.
    pub fn size(&self) -> u64 {
        self.allocation_size
    }

    /// G-3: take the raw OPAQUE_FD, RELEASING Rust ownership.
    ///
    /// Returns the raw `i32` fd via `into_raw_fd`; Python becomes the single owner
    /// and closes it after `cudaImportExternalMemory` consumes it. Returns `-1` if
    /// the fd was already taken (idempotent guard — a second call must NOT close an
    /// unrelated reopened fd).
    pub fn take_memory_fd_raw(&mut self) -> i32 {
        match self.memory_fd.take() {
            Some(fd) => fd.into_raw_fd(),
            None => -1,
        }
    }
}

/// A TIMELINE semaphore exported as OPAQUE_FD for CUDA import.
///
/// The fd is owned until [`take_fd_raw`](Self::take_fd_raw) hands it to Python (G-3).
pub struct ExternalTimelineSemaphore {
    /// The Vulkan timeline semaphore handle.
    pub semaphore: vk::Semaphore,
    /// The exported OPAQUE_FD, owned until handed off (G-3).
    fd: Option<OwnedFd>,
    /// The initial timeline value the semaphore was created with.
    pub initial_value: u64,
}

impl ExternalTimelineSemaphore {
    /// G-3: take the raw OPAQUE_FD, RELEASING Rust ownership.
    pub fn take_fd_raw(&mut self) -> i32 {
        match self.fd.take() {
            Some(fd) => fd.into_raw_fd(),
            None => -1,
        }
    }
}

/// One direction of the G-6 binary-semaphore-pair fallback.
pub struct ExternalBinarySemaphore {
    /// The Vulkan binary semaphore handle.
    pub semaphore: vk::Semaphore,
    /// The exported OPAQUE_FD, owned until handed off (G-3).
    fd: Option<OwnedFd>,
}

impl ExternalBinarySemaphore {
    /// G-3: take the raw OPAQUE_FD, RELEASING Rust ownership.
    pub fn take_fd_raw(&mut self) -> i32 {
        match self.fd.take() {
            Some(fd) => fd.into_raw_fd(),
            None => -1,
        }
    }
}

/// A set of exportable shared buffers + the shared timeline semaphore for one
/// zero-copy kernel — M4.1 (G-2). Owned by the PyO3 layer; the descriptor
/// pool/set bind the external buffers DIRECTLY as the kernel's SSBOs (no staging).
pub struct SharedBufferSet {
    /// One [`ExternalBuffer`] per descriptor binding (input + output).
    pub buffers: Vec<ExternalBuffer>,
    /// The shared TIMELINE semaphore (exported to CUDA).
    pub timeline: Option<ExternalTimelineSemaphore>,
    /// G-6 fallback binary semaphore pair, present only when timeline interop failed.
    pub binary_pair: Option<(ExternalBinarySemaphore, ExternalBinarySemaphore)>,
    /// The descriptor pool that owns `descriptor_set` (allocated by the zero-copy path).
    pub descriptor_pool: vk::DescriptorPool,
    /// The descriptor set binding the external buffers as SSBOs.
    pub descriptor_set: vk::DescriptorSet,
    /// G-2 instrumentation: number of transient staging buffers allocated on the
    /// zero-copy path. MUST stay 0 — asserted by AT-2109.
    pub staging_buffers_allocated: u32,
    /// G-2 instrumentation: number of `vkCmdCopyBuffer` records on the zero-copy path.
    /// MUST stay 0 — asserted by AT-2109.
    pub copy_buffer_records: u32,
}

impl SharedBufferSet {
    /// Whether timeline-over-fd is the active sync path (vs the binary fallback).
    pub fn timeline_supported(&self) -> bool {
        self.timeline.is_some()
    }

    /// G-3: take the raw OPAQUE_FDs for every buffer (ownership → Python), in order.
    ///
    /// Each fd is handed off via `into_raw_fd`; Rust will NOT Drop-close them. A
    /// second call returns `-1`s.
    pub fn take_memory_fds_raw(&mut self) -> Vec<i32> {
        self.buffers.iter_mut().map(|b| b.take_memory_fd_raw()).collect()
    }

    /// Per-buffer allocation sizes in bytes (binding order).
    pub fn sizes(&self) -> Vec<u64> {
        self.buffers.iter().map(|b| b.size()).collect()
    }

    /// G-1: per-buffer dedication bit (all `true` for now) — Python sets
    /// `cudaExternalMemoryHandleDesc.flags = cudaExternalMemoryDedicated` to MATCH.
    pub fn dedicated(&self) -> Vec<bool> {
        self.buffers.iter().map(|b| b.dedicated()).collect()
    }

    /// G-3: take the raw timeline-semaphore fd (ownership → Python), or `-1` if on the
    /// binary fallback / already taken.
    pub fn take_semaphore_fd_raw(&mut self) -> i32 {
        match self.timeline.as_mut() {
            Some(ts) => ts.take_fd_raw(),
            None => -1,
        }
    }

    /// G-6 fallback: take the two binary-semaphore fds (A, B). Returns `(-1, -1)` on
    /// the timeline path / already taken.
    pub fn take_binary_semaphore_fds_raw(&mut self) -> (i32, i32) {
        match self.binary_pair.as_mut() {
            Some((a, b)) => (a.take_fd_raw(), b.take_fd_raw()),
            None => (-1, -1),
        }
    }
}

fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    (value + alignment - 1) & !(alignment - 1)
}

/// Allocate a DEVICE_LOCAL, dedicated, OPAQUE_FD-exportable buffer and export its fd.
///
/// The buffer is created with `STORAGE_BUFFER | TRANSFER_SRC | TRANSFER_DST` usage
/// and a chained `VkExternalMemoryBufferCreateInfo(OPAQUE_FD)`. Its memory is
/// allocated with `VkExportMemoryAllocateInfo(OPAQUE_FD)` AND
/// `VkMemoryDedicatedAllocateInfo{buffer}` (unconditional, G-1) on a DEVICE_LOCAL
/// memory type, bound at offset 0, then exported via `vkGetMemoryFdKHR`.
///
/// FAIL-CLOSED ([`DispatchError::MemoryNotExportable`]) if the only available
/// DEVICE_LOCAL type is the unified (`DEVICE_LOCAL | HOST_VISIBLE`) iGPU type, which
/// may not be CUDA-importable over OPAQUE_FD — for M4.1 (discrete NVIDIA) we require
/// a pure DEVICE_LOCAL type.
pub fn allocate_exportable_device_local_buffer(
    device: &ash::Device,
    ext_mem_fd_fns: &ash::khr::external_memory_fd::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
    binding: u32,
) -> Result<ExternalBuffer, DispatchError> {
    let requested_size: u64 = size.max(4);

    // Chain VkExternalMemoryBufferCreateInfo(OPAQUE_FD) onto the buffer create info
    // so the buffer is declared exportable at creation.
    let mut ext_buf_info = vk::ExternalMemoryBufferCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
    let buffer_info = vk::BufferCreateInfo::default()
        .size(requested_size)
        .usage(
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(&mut ext_buf_info);

    // SAFETY: device is valid; buffer_info (and its chained ext_buf_info) live for the call.
    let buffer: vk::Buffer = unsafe { device.create_buffer(&buffer_info, None) }
        .map_err(|e| DispatchError::BufferAllocationFailed {
            binding,
            size: requested_size as usize,
            reason: format!("create exportable buffer: {e}"),
        })?;

    // SAFETY: buffer was just created.
    let mem_reqs: vk::MemoryRequirements =
        unsafe { device.get_buffer_memory_requirements(buffer) };

    // Require a PURE DEVICE_LOCAL type (not the unified iGPU fallback) for OPAQUE_FD
    // exportability on discrete NVIDIA. We re-implement the discrete-only scan here
    // rather than reuse the iGPU-tolerant helper.
    let memory_type_index: u32 = match find_pure_device_local_memory_type_index(
        mem_props,
        mem_reqs.memory_type_bits,
    ) {
        Some(i) => i,
        None => {
            // If only a unified type exists, surface MemoryNotExportable (fail-closed).
            // SAFETY: buffer was created above; destroy on error to avoid leak.
            unsafe { device.destroy_buffer(buffer, None); }
            if find_device_local_memory_type_index(mem_props, mem_reqs.memory_type_bits).is_some() {
                return Err(DispatchError::MemoryNotExportable);
            }
            return Err(DispatchError::NoCompatibleMemoryType);
        }
    };

    let aligned_size: u64 = align_up(requested_size, mem_reqs.alignment);

    // G-1: dedicated allocation (VkMemoryDedicatedAllocateInfo{buffer}) +
    // exportable allocation (VkExportMemoryAllocateInfo(OPAQUE_FD)).
    let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
    let mut export_info = vk::ExportMemoryAllocateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(aligned_size)
        .memory_type_index(memory_type_index)
        .push_next(&mut export_info)
        .push_next(&mut dedicated_info);

    // SAFETY: alloc_info (and its chained structs) live for the call; buffer is valid.
    let memory: vk::DeviceMemory = unsafe { device.allocate_memory(&alloc_info, None) }
        .map_err(|e| {
            // SAFETY: buffer was created above; destroy on error to avoid leak.
            unsafe { device.destroy_buffer(buffer, None); }
            DispatchError::BufferAllocationFailed {
                binding,
                size: aligned_size as usize,
                reason: format!("allocate exportable memory: {e}"),
            }
        })?;

    // SAFETY: memory and buffer are valid; offset 0 satisfies the whole-allocation bind.
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
                reason: format!("bind exportable buffer: {e}"),
            }
        })?;

    // Export the OPAQUE_FD for this allocation.
    let memory_fd: OwnedFd = export_memory_fd(ext_mem_fd_fns, memory).inspect_err(|_| {
        // SAFETY: memory + buffer were created above; destroy on export failure.
        unsafe {
            device.free_memory(memory, None);
            device.destroy_buffer(buffer, None);
        }
    })?;

    Ok(ExternalBuffer {
        buffer,
        memory,
        allocation_size: aligned_size,
        binding,
        memory_fd: Some(memory_fd),
        dedicated: true,
    })
}

/// Find a PURE DEVICE_LOCAL (NOT also HOST_VISIBLE) memory type index.
///
/// Unlike [`find_device_local_memory_type_index`], this rejects the unified iGPU
/// `DEVICE_LOCAL | HOST_VISIBLE` fallback, which may not be CUDA-importable over
/// OPAQUE_FD. Returns `None` if no pure DEVICE_LOCAL type is compatible.
fn find_pure_device_local_memory_type_index(
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
) -> Option<u32> {
    for i in 0..mem_props.memory_type_count {
        if (type_bits >> i) & 1 != 1 {
            continue;
        }
        let flags = mem_props.memory_types[i as usize].property_flags;
        if flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            && !flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        {
            return Some(i);
        }
    }
    None
}

/// Export an OPAQUE_FD for `memory` via `vkGetMemoryFdKHR`.
///
/// Returns an [`OwnedFd`] (ownership transferred from Vulkan to the app). The
/// caller hands the raw fd to Python and does NOT Drop-close it (G-3).
pub fn export_memory_fd(
    ext_mem_fd_fns: &ash::khr::external_memory_fd::Device,
    memory: vk::DeviceMemory,
) -> Result<OwnedFd, DispatchError> {
    let get_info = vk::MemoryGetFdInfoKHR::default()
        .memory(memory)
        .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
    // SAFETY: ext_mem_fd_fns was loaded for the same device that owns `memory`;
    // get_info is well-formed; vkGetMemoryFdKHR transfers fd ownership to us.
    let raw: i32 = unsafe { ext_mem_fd_fns.get_memory_fd(&get_info) }
        .map_err(|e| DispatchError::ExternalExportFailed {
            kind: "memory",
            reason: e.to_string(),
        })?;
    if raw < 0 {
        return Err(DispatchError::ExternalExportFailed {
            kind: "memory",
            reason: "vkGetMemoryFdKHR returned a negative fd".to_owned(),
        });
    }
    // SAFETY: `raw` is a fresh, valid, owned fd from vkGetMemoryFdKHR; wrapping it
    // as OwnedFd makes us its sole owner until hand-off.
    Ok(unsafe { OwnedFd::from_raw_fd(raw) })
}

/// Create a TIMELINE semaphore that is exportable as OPAQUE_FD, and export its fd.
///
/// Chains `VkSemaphoreTypeCreateInfo(TIMELINE, initial_value)` +
/// `VkExportSemaphoreCreateInfo(OPAQUE_FD)` into the create info, then exports via
/// `vkGetSemaphoreFdKHR`. Python imports the fd with
/// `cudaImportExternalSemaphore(type = TimelineSemaphoreFd = 9)`.
pub fn create_exportable_timeline_semaphore(
    device: &ash::Device,
    ext_sem_fd_fns: &ash::khr::external_semaphore_fd::Device,
    initial_value: u64,
) -> Result<ExternalTimelineSemaphore, DispatchError> {
    let mut type_info = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(initial_value);
    let mut export_info = vk::ExportSemaphoreCreateInfo::default()
        .handle_types(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);
    let create_info = vk::SemaphoreCreateInfo::default()
        .push_next(&mut type_info)
        .push_next(&mut export_info);

    // SAFETY: device is valid; create_info chain lives for the call.
    let semaphore: vk::Semaphore = unsafe { device.create_semaphore(&create_info, None) }
        .map_err(|e| DispatchError::SemaphoreCreationFailed(format!(
            "exportable timeline semaphore: {e}"
        )))?;

    let fd: OwnedFd = export_semaphore_fd(ext_sem_fd_fns, semaphore).inspect_err(|_| {
        // SAFETY: semaphore was created above; destroy on export failure.
        unsafe { device.destroy_semaphore(semaphore, None); }
    })?;

    Ok(ExternalTimelineSemaphore {
        semaphore,
        fd: Some(fd),
        initial_value,
    })
}

/// G-6 fallback: create a PAIR of binary semaphores exportable as OPAQUE_FD.
///
/// Used only when the timeline-over-fd interop spike (AT-2110) fails on the driver.
/// One semaphore per direction (torch→Vulkan and Vulkan→torch).
pub fn create_exportable_binary_semaphore_pair(
    device: &ash::Device,
    ext_sem_fd_fns: &ash::khr::external_semaphore_fd::Device,
) -> Result<(ExternalBinarySemaphore, ExternalBinarySemaphore), DispatchError> {
    let a = create_exportable_binary_semaphore(device, ext_sem_fd_fns)?;
    let b = create_exportable_binary_semaphore(device, ext_sem_fd_fns).inspect_err(|_| {
        // SAFETY: `a` was created above; destroy it (and close its fd) on the second failure.
        unsafe { device.destroy_semaphore(a.semaphore, None); }
    })?;
    Ok((a, b))
}

fn create_exportable_binary_semaphore(
    device: &ash::Device,
    ext_sem_fd_fns: &ash::khr::external_semaphore_fd::Device,
) -> Result<ExternalBinarySemaphore, DispatchError> {
    let mut export_info = vk::ExportSemaphoreCreateInfo::default()
        .handle_types(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);
    let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut export_info);

    // SAFETY: device is valid; create_info chain lives for the call.
    let semaphore: vk::Semaphore = unsafe { device.create_semaphore(&create_info, None) }
        .map_err(|e| DispatchError::SemaphoreCreationFailed(format!(
            "exportable binary semaphore: {e}"
        )))?;

    let fd: OwnedFd = export_semaphore_fd(ext_sem_fd_fns, semaphore).inspect_err(|_| {
        // SAFETY: semaphore was created above; destroy on export failure.
        unsafe { device.destroy_semaphore(semaphore, None); }
    })?;

    Ok(ExternalBinarySemaphore { semaphore, fd: Some(fd) })
}

/// Export an OPAQUE_FD for `semaphore` via `vkGetSemaphoreFdKHR`.
pub fn export_semaphore_fd(
    ext_sem_fd_fns: &ash::khr::external_semaphore_fd::Device,
    semaphore: vk::Semaphore,
) -> Result<OwnedFd, DispatchError> {
    let get_info = vk::SemaphoreGetFdInfoKHR::default()
        .semaphore(semaphore)
        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);
    // SAFETY: ext_sem_fd_fns was loaded for the same device that owns `semaphore`;
    // get_info is well-formed; vkGetSemaphoreFdKHR transfers fd ownership to us.
    let raw: i32 = unsafe { ext_sem_fd_fns.get_semaphore_fd(&get_info) }
        .map_err(|e| DispatchError::ExternalExportFailed {
            kind: "semaphore",
            reason: e.to_string(),
        })?;
    if raw < 0 {
        return Err(DispatchError::ExternalExportFailed {
            kind: "semaphore",
            reason: "vkGetSemaphoreFdKHR returned a negative fd".to_owned(),
        });
    }
    // SAFETY: `raw` is a fresh, valid, owned fd from vkGetSemaphoreFdKHR.
    Ok(unsafe { OwnedFd::from_raw_fd(raw) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk;

    /// `find_pure_device_local_memory_type_index` rejects the unified iGPU type.
    #[test]
    fn pure_device_local_rejects_unified() {
        let mut memory_types = [vk::MemoryType::default(); vk::MAX_MEMORY_TYPES];
        // type 0: DEVICE_LOCAL | HOST_VISIBLE (unified iGPU)
        memory_types[0].property_flags =
            vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE;
        let mem_props = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: 1,
            memory_types,
            ..Default::default()
        };
        assert_eq!(
            find_pure_device_local_memory_type_index(&mem_props, 0b1),
            None,
            "unified DEVICE_LOCAL|HOST_VISIBLE must NOT be accepted for OPAQUE_FD export"
        );
    }

    /// `find_pure_device_local_memory_type_index` accepts a pure DEVICE_LOCAL type.
    #[test]
    fn pure_device_local_accepts_discrete() {
        let mut memory_types = [vk::MemoryType::default(); vk::MAX_MEMORY_TYPES];
        memory_types[0].property_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        memory_types[1].property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let mem_props = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: 2,
            memory_types,
            ..Default::default()
        };
        assert_eq!(
            find_pure_device_local_memory_type_index(&mem_props, 0b11),
            Some(1),
            "pure DEVICE_LOCAL type at index 1 must be selected"
        );
    }

    /// `take_memory_fd_raw` is idempotent: a second call returns -1, not a stale fd.
    ///
    /// We construct an `ExternalBuffer` with a real owned fd (a pipe read end) so
    /// `into_raw_fd` yields a valid descriptor on the first take, and `-1` on the
    /// second (the G-3 single-owner / no-double-close contract). We close the taken
    /// fd ourselves to avoid leaking it in the test.
    #[test]
    fn take_memory_fd_raw_is_idempotent() {
        // Make a real owned fd via a pipe.
        let mut fds = [0i32; 2];
        // SAFETY: `pipe` writes exactly two fds into `fds`.
        let rc = unsafe { libc_pipe(fds.as_mut_ptr()) };
        assert_eq!(rc, 0, "pipe() must succeed");
        // SAFETY: fds[0] is a fresh owned read end from pipe().
        let read_fd = unsafe { OwnedFd::from_raw_fd(fds[0]) };
        // SAFETY: fds[1] is a fresh owned write end from pipe().
        let _write_fd = unsafe { OwnedFd::from_raw_fd(fds[1]) };

        let mut buf = ExternalBuffer {
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            allocation_size: 4,
            binding: 0,
            memory_fd: Some(read_fd),
            dedicated: true,
        };
        assert!(buf.dedicated(), "G-1: dedicated must be true");
        let raw = buf.take_memory_fd_raw();
        assert!(raw >= 0, "first take returns the valid fd");
        assert_eq!(buf.take_memory_fd_raw(), -1, "second take must return -1 (no double-close)");
        // Close the taken fd ourselves (ownership was transferred to us).
        // SAFETY: `raw` is the valid pipe read end we just took; close it once.
        unsafe { libc_close(raw); }
        // _write_fd drops and closes the write end.
    }

    // Minimal libc bindings for the pipe-based fd test (avoids a libc dep).
    extern "C" {
        #[link_name = "pipe"]
        fn libc_pipe(fds: *mut i32) -> i32;
        #[link_name = "close"]
        fn libc_close(fd: i32) -> i32;
    }
}
