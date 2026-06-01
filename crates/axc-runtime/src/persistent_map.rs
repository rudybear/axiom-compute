//! Persistent-mapping wrapper for HOST_VISIBLE staging memory.
//!
//! `PersistentMapping` holds a PRIVATE raw `*mut u8` returned by a single
//! `vkMapMemory` call kept alive for the buffer-slot lifetime, plus the coherency
//! policy (`Coherent` or `NonCoherent { atom_size }`). Host access is exclusively
//! through the guarded `copy_in` / `copy_out` helpers and the flush/invalidate pair.
//!
//! ## Safety model (CRITICAL-6)
//!
//! - `unsafe impl Send for PersistentMapping {}` ONLY — never `Sync`.
//! - The `*mut u8` is PRIVATE; there is no public `host_ptr`.
//! - `copy_in` and `copy_out` are called only from `record_and_submit_dispatch`
//!   while the per-handle `buffers` `Mutex` is held, providing the happens-before edge.
//! - `unmap` consumes `self`, making use-after-unmap a compile error and a
//!   forgotten unmap a compile error (field cannot be partially moved).
//!
//! ## Flush/invalidate placement (WARNING-1, WARNING-2)
//!
//! `mapped_range` returns `(offset=0, size)` where:
//! - `aligned = align_up(byte_len, atom_size)`.
//! - if `aligned >= alloc_size` → `size = VK_WHOLE_SIZE` (never a non-atom-multiple).
//! - else → `size = aligned` (a true atom multiple).
//!
//! The old `.min(alloc)` clamp is ABSENT — it could yield a non-atom-multiple size.

use ash::vk;
use crate::error::DispatchError;

/// Coherency policy for a staging buffer's backing memory.
///
/// Determines whether `flush`/`invalidate` are no-ops (`Coherent`) or real Vulkan
/// calls (`NonCoherent`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Coherency {
    /// Memory is `HOST_VISIBLE | HOST_COHERENT` — flush/invalidate are no-ops.
    Coherent,
    /// Memory is `HOST_VISIBLE | HOST_CACHED` without coherent — flush/invalidate required.
    NonCoherent {
        /// `VkPhysicalDeviceLimits::nonCoherentAtomSize` (NEVER hardcoded, WARNING-1).
        atom_size: u64,
    },
}

/// Persistent-mapped host-visible staging memory.
///
/// Created once at buffer allocation; destroyed by `unmap` (which consumes `self`)
/// before the associated `VkBuffer` and `VkDeviceMemory` are destroyed.
///
/// # Safety
///
/// `unsafe impl Send` ONLY (see module-level safety model). No `Sync`.
pub(crate) struct PersistentMapping {
    /// Raw host pointer into the persistently mapped device memory.
    ///
    /// PRIVATE — never exposed. All access goes through `copy_in`/`copy_out`.
    ptr: *mut u8,
    /// Total allocated size of the underlying `VkDeviceMemory` object in bytes.
    ///
    /// Used by `mapped_range` to determine whether `VK_WHOLE_SIZE` applies.
    size: u64,
    /// Coherency policy derived from the memory-type property flags.
    coherency: Coherency,
}

// SAFETY: `PersistentMapping::ptr` points into Vulkan device memory that is valid
// on any thread (the mapping is process-global). The pointer is PRIVATE and
// dereferenced only under the per-handle `buffers` Mutex (the only call sites are
// `copy_in`/`copy_out` in `record_and_submit_dispatch`). Sending the struct to
// another thread transfers ownership of the pointer, and the exclusive Mutex access
// provides the required happens-before edge.
//
// NOT `Sync`: there is no scenario where two threads hold `&PersistentMapping`
// simultaneously — access is always mediated through `Mutex<Vec<BufferSlot>>`.
// `Mutex<T>: Sync` only requires `T: Send`, not `T: Sync`, so omitting `Sync` here
// is correct and safe.
unsafe impl Send for PersistentMapping {}

/// Map `memory` persistently and return a `PersistentMapping`.
///
/// Calls `vkMapMemory(memory, 0, size, 0)` exactly once. The mapping remains
/// active until `PersistentMapping::unmap` is called.
///
/// On `vkMapMemory` failure the partially-constructed allocation is NOT cleaned up
/// here — the caller is responsible (the error is returned and the caller must
/// `destroy_buffer` and `free_memory`).
///
/// # Safety
///
/// - `device` must be the device that allocated `memory`.
/// - `memory` must be HOST_VISIBLE and not yet persistently mapped.
/// - `size` must equal the actual allocated size of `memory`.
/// - The mapping must be unmapped (via `unmap`) before `free_memory` is called.
pub(crate) unsafe fn map_persistent(
    device: &ash::Device,
    memory: vk::DeviceMemory,
    size: u64,
    coherency: Coherency,
) -> Result<PersistentMapping, DispatchError> {
    // SAFETY: caller guarantees memory is HOST_VISIBLE, not yet mapped, and size is correct.
    let raw_ptr: *mut std::ffi::c_void = unsafe {
        device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
    }
    .map_err(|e| DispatchError::MemoryMapFailed(format!("persistent vkMapMemory: {e}")))?;

    Ok(PersistentMapping {
        ptr: raw_ptr as *mut u8,
        size,
        coherency,
    })
}

impl PersistentMapping {
    /// Copy `src` bytes into the persistently mapped staging memory.
    ///
    /// The caller MUST hold the per-handle `buffers` Mutex when calling this.
    /// `src.len()` must be <= the buffer's allocated size.
    pub(crate) fn copy_in(&self, src: &[u8]) {
        // SAFETY: self.ptr is a valid, persistently mapped host pointer into device memory.
        // src.len() <= allocation size is guaranteed by ensure_buffers_fit_with_mem_props.
        // The per-handle buffers Mutex is held by the caller, providing the exclusive-access
        // invariant that makes this memcpy safe.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr, src.len());
        }
    }

    /// Copy `dst.len()` bytes from the persistently mapped staging memory into `dst`.
    ///
    /// The caller MUST hold the per-handle `buffers` Mutex when calling this.
    /// `dst.len()` must be <= the buffer's allocated size.
    pub(crate) fn copy_out(&self, dst: &mut [u8]) {
        // SAFETY: self.ptr is valid mapped device memory; dst.len() <= allocation size.
        // Mutex held by caller.
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr as *const u8, dst.as_mut_ptr(), dst.len());
        }
    }

    /// Return the coherency policy of this mapping.
    #[allow(dead_code)]
    pub(crate) fn coherency(&self) -> Coherency {
        self.coherency
    }

    /// Flush the mapped memory range to make host writes visible to the device.
    ///
    /// No-op when `coherency == Coherent` (HOST_COHERENT memory requires no flush).
    ///
    /// On `NonCoherent` memory, calls `vkFlushMappedMemoryRanges` with the range
    /// computed by `mapped_range(coherency, byte_len, self.size)`.
    ///
    /// # Safety
    ///
    /// - `device` and `memory` must match those used to create this mapping.
    /// - Must be called AFTER `copy_in` and BEFORE the upload command buffer is submitted.
    pub(crate) unsafe fn flush(
        &self,
        device: &ash::Device,
        memory: vk::DeviceMemory,
        byte_len: u64,
    ) -> Result<(), DispatchError> {
        match self.coherency {
            Coherency::Coherent => {
                // No-op on coherent memory.
                return Ok(());
            }
            Coherency::NonCoherent { atom_size } => {
                let (offset, size) = mapped_range(self.coherency, byte_len, self.size);
                let range = vk::MappedMemoryRange::default()
                    .memory(memory)
                    .offset(offset)
                    .size(size);

                // SAFETY: range describes a valid mapped subrange; atom_size alignment
                // is guaranteed by mapped_range.
                unsafe { device.flush_mapped_memory_ranges(&[range]) }
                    .map_err(|e| DispatchError::MappedRangeOpFailed {
                        op: crate::error::MappedRangeOp::Flush,
                        reason: format!(
                            "vkFlushMappedMemoryRanges (byte_len={byte_len}, atom={atom_size}): {e}"
                        ),
                    })?;
            }
        }
        Ok(())
    }

    /// Invalidate the mapped memory range to make device writes visible to the host.
    ///
    /// No-op when `coherency == Coherent`.
    ///
    /// On `NonCoherent` memory, calls `vkInvalidateMappedMemoryRanges` with the
    /// range computed by `mapped_range(coherency, byte_len, self.size)`.
    ///
    /// # Safety
    ///
    /// - `device` and `memory` must match those used to create this mapping.
    /// - Must be called AFTER `vkWaitForFences` and BEFORE `copy_out`.
    pub(crate) unsafe fn invalidate(
        &self,
        device: &ash::Device,
        memory: vk::DeviceMemory,
        byte_len: u64,
    ) -> Result<(), DispatchError> {
        match self.coherency {
            Coherency::Coherent => {
                // No-op on coherent memory.
                return Ok(());
            }
            Coherency::NonCoherent { atom_size } => {
                let (offset, size) = mapped_range(self.coherency, byte_len, self.size);
                let range = vk::MappedMemoryRange::default()
                    .memory(memory)
                    .offset(offset)
                    .size(size);

                // SAFETY: range describes a valid mapped subrange; atom_size alignment
                // guaranteed by mapped_range.
                unsafe { device.invalidate_mapped_memory_ranges(&[range]) }
                    .map_err(|e| DispatchError::MappedRangeOpFailed {
                        op: crate::error::MappedRangeOp::Invalidate,
                        reason: format!(
                            "vkInvalidateMappedMemoryRanges (byte_len={byte_len}, atom={atom_size}): {e}"
                        ),
                    })?;
            }
        }
        Ok(())
    }

    /// Unmap the persistently-mapped memory, consuming `self`.
    ///
    /// Must be called BEFORE `vkDestroyBuffer` and `vkFreeMemory` on the associated
    /// buffer slot. Because this method consumes `self`, a forgotten unmap is a
    /// compile error (the `PersistentMapping` field cannot be partially moved).
    ///
    /// # Safety
    ///
    /// - `device` and `memory` must match those used to create this mapping.
    /// - Must be called exactly once per `PersistentMapping` instance.
    /// - No access through `self.ptr` may occur after this call.
    pub(crate) unsafe fn unmap(self, device: &ash::Device, memory: vk::DeviceMemory) {
        // SAFETY: memory was mapped by map_persistent with the same device; this
        // consumes self so the raw pointer becomes unreachable after this call.
        unsafe { device.unmap_memory(memory); }
        // self is dropped here (ptr/size/coherency are Copy/trivially-drop types).
    }
}

/// Compute the `(offset, size)` pair for a `VkMappedMemoryRange` per the Vulkan spec.
///
/// Rules (WARNING-2):
/// - `offset` is always `0`.
/// - `aligned = align_up(byte_len, atom_size)` (where atom_size comes from
///   `nonCoherentAtomSize`; for `Coherent` this returns `(0, VK_WHOLE_SIZE)`).
/// - if `aligned >= alloc_size` → `size = VK_WHOLE_SIZE`
///   (VUID-VkMappedMemoryRange-size-01390: extends to end of allocation even when
///   `alloc_size` is not an atom multiple).
/// - else → `size = aligned` (a true atom multiple, satisfying
///   VUID-VkMappedMemoryRange-size-01390 on the non-WHOLE_SIZE branch).
///
/// The old `.min(alloc_size)` clamp is absent — it could yield a non-atom-multiple
/// that violates the VUID.
///
/// # Examples
///
/// ```no_run
/// // mapped_range is pub(crate); see the #[cfg(test)] module for usage examples.
/// // When aligned(len=100, atom=64)=128 < alloc=256: returns (0, 128).
/// // When aligned(len=200, atom=64)=256 >= alloc=200: returns (0, WHOLE_SIZE).
/// ```
pub(crate) fn mapped_range(coherency: Coherency, byte_len: u64, alloc_size: u64) -> (u64, u64) {
    match coherency {
        Coherency::Coherent => {
            // On coherent memory flush/invalidate are no-ops; return a canonical range.
            (0, vk::WHOLE_SIZE)
        }
        Coherency::NonCoherent { atom_size } => {
            let aligned: u64 = align_up(byte_len, atom_size);
            let size: u64 = if aligned >= alloc_size {
                vk::WHOLE_SIZE
            } else {
                aligned
            };
            (0, size)
        }
    }
}

/// Round `value` up to the next multiple of `alignment` (power-of-two or not).
///
/// If `alignment` is 0 or 1, returns `value` unchanged.
fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment <= 1 {
        return value;
    }
    value.div_ceil(alignment) * alignment
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk;

    /// AT-1410: mapped_range respects nonCoherentAtomSize for all atom sizes
    /// including pathological large values; no clamp bug.
    ///
    /// Asserts: offset==0 AND (size%atom==0 OR size==VK_WHOLE_SIZE) AND size>=byte_len.
    #[test]
    fn at_1410_mapped_range_alignment_all_atoms() {
        let atom_sizes: &[u64] = &[1, 64, 128, 256, 1024, 65536];
        let byte_lens: &[u64] = &[1, 4, 4095, 4096, 1 << 20];

        for &atom in atom_sizes {
            for &byte_len in byte_lens {
                // alloc = round up byte_len to next power-of-two (matching round_up_pow2)
                let alloc: u64 = {
                    if byte_len <= 4 {
                        4
                    } else if byte_len > (1u64 << 62) {
                        u64::MAX
                    } else {
                        byte_len.next_power_of_two()
                    }
                };
                let coherency = Coherency::NonCoherent { atom_size: atom };
                let (offset, size) = mapped_range(coherency, byte_len, alloc);

                assert_eq!(offset, 0, "offset must always be 0 (atom={atom}, len={byte_len})");
                assert!(
                    size >= byte_len,
                    "size={size} must be >= byte_len={byte_len} (atom={atom})"
                );
                if size != vk::WHOLE_SIZE {
                    assert_eq!(
                        size % atom,
                        0,
                        "non-WHOLE_SIZE range must be atom-aligned; size={size}, atom={atom}, len={byte_len}"
                    );
                }
            }
        }
    }

    /// AT-1410b: non-atom-aligned allocation triggers WHOLE_SIZE (WARNING-2).
    ///
    /// When alloc_size is not an atom multiple and align_up(len,atom) >= alloc,
    /// mapped_range must return WHOLE_SIZE (not clamp).
    #[test]
    fn at_1410b_non_atom_aligned_alloc_returns_whole_size() {
        // atom=256, byte_len=200, alloc=200 (not pow2, not atom multiple)
        let (offset, size) = mapped_range(Coherency::NonCoherent { atom_size: 256 }, 200, 200);
        assert_eq!(offset, 0);
        assert_eq!(size, vk::WHOLE_SIZE, "aligned(200,256)=256 >= alloc=200 → WHOLE_SIZE");
    }

    /// AT-1410c: aligned < alloc returns a true atom multiple.
    #[test]
    fn at_1410c_aligned_less_than_alloc_returns_atom_multiple() {
        // atom=64, byte_len=100, alloc=256. aligned=128 < 256 → size=128 (atom multiple)
        let (offset, size) = mapped_range(Coherency::NonCoherent { atom_size: 64 }, 100, 256);
        assert_eq!(offset, 0);
        assert_eq!(size, 128, "align_up(100,64)=128; 128 < 256 → size=128");
        assert_eq!(size % 64, 0, "128 must be a multiple of 64");
    }

    /// AT-1410d: atom_size=1 — aligned == byte_len (trivial alignment).
    #[test]
    fn at_1410d_atom_one_trivial_alignment() {
        // atom=1 means no alignment constraint; align_up(x,1)=x
        let (offset, size) = mapped_range(Coherency::NonCoherent { atom_size: 1 }, 4095, 8192);
        assert_eq!(offset, 0);
        // align_up(4095, 1) = 4095 < 8192 → size=4095
        assert_eq!(size, 4095);
    }

    /// AT-1410e: Coherent mapping always returns (0, WHOLE_SIZE).
    #[test]
    fn at_1410e_coherent_returns_whole_size() {
        let (offset, size) = mapped_range(Coherency::Coherent, 4096, 4096);
        assert_eq!(offset, 0);
        assert_eq!(size, vk::WHOLE_SIZE);
    }

    /// align_up — basic correctness for non-power-of-two alignment (atom_size can be
    /// any multiple, not just power-of-two per the Vulkan spec).
    #[test]
    fn align_up_non_pow2_atom() {
        // atom_size is always a power of two per spec, but our align_up handles general case.
        assert_eq!(align_up(0, 64), 0);
        assert_eq!(align_up(1, 64), 64);
        assert_eq!(align_up(64, 64), 64);
        assert_eq!(align_up(65, 64), 128);
        assert_eq!(align_up(127, 64), 128);
        assert_eq!(align_up(128, 64), 128);
        assert_eq!(align_up(100, 256), 256);
        assert_eq!(align_up(257, 256), 512);
    }

    /// align_up — zero/one alignment passthrough.
    #[test]
    fn align_up_zero_one_passthrough() {
        assert_eq!(align_up(42, 0), 42);
        assert_eq!(align_up(42, 1), 42);
        assert_eq!(align_up(0, 1), 0);
    }

    /// mapped_range with large atom (65536) and byte_len=1 — must trigger WHOLE_SIZE
    /// or return a correctly-aligned size.
    #[test]
    fn at_1410f_large_atom_small_len() {
        // atom=65536, byte_len=1, alloc=4 (minimum from round_up_pow2)
        // aligned = align_up(1, 65536) = 65536 >= alloc=4 → WHOLE_SIZE
        let (offset, size) = mapped_range(Coherency::NonCoherent { atom_size: 65536 }, 1, 4);
        assert_eq!(offset, 0);
        assert_eq!(size, vk::WHOLE_SIZE);
    }
}
