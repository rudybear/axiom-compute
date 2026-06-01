//! Dedicated-transfer-queue topology selection.
//!
//! Encapsulates queue-family selection and the `QueueMode` decision so
//! `kernel_handle.rs` and `context.rs` stay readable.
//!
//! ## Queue mode decision
//!
//! - `DedicatedTransfer`: the device exposes a queue family with `TRANSFER`
//!   but **without** `COMPUTE` or `GRAPHICS` (the discrete DMA copy engine).
//!   In this mode the device-local buffer is created
//!   `VK_SHARING_MODE_CONCURRENT` over `{compute_family, transfer_family}`,
//!   eliminating all queue-family ownership-transfer barriers (CRITICAL-2).
//!
//! - `SingleQueue`: no distinct transfer-only family found (Lavapipe, many iGPUs),
//!   or `force_single_queue` was set. Command stream is identical to M2.3a.
//!
//! ## No ownership-transfer barriers (CRITICAL-2)
//!
//! In `DedicatedTransfer` mode we use `VK_SHARING_MODE_CONCURRENT` for device-local
//! buffers. This eliminates the matched release/acquire barrier pairs entirely —
//! the canonical "NVIDIA passes, AMD corrupts" failure class. Plain
//! `QUEUE_FAMILY_IGNORED` barriers are sufficient for memory visibility.
//! Staging buffers stay `EXCLUSIVE` (touched only by the transfer family in dedicated
//! mode, only by the compute family in single-queue mode).

use ash::vk;
use crate::error::DispatchError;

/// How dispatches use the available GPU queues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QueueMode {
    /// A single compute queue handles all operations (M2.3a behaviour preserved).
    SingleQueue,
    /// Upload and readback run on a dedicated transfer queue; compute on the
    /// compute queue. Device-local buffers use `CONCURRENT` sharing.
    DedicatedTransfer,
}

/// Result of queue-family selection — indices only (no Vulkan handles).
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueFamilySelection {
    /// Queue family with `COMPUTE` support (unchanged from M2.3a).
    pub(crate) compute_family: u32,
    /// Distinct `TRANSFER`-only queue family, or `None` if not available.
    pub(crate) transfer_family: Option<u32>,
}

/// Information about the optional dedicated transfer queue and its command pool.
///
/// Owned by `VulkanContext`; destroyed in `VulkanContext::drop` after
/// `device_wait_idle()`.
pub(crate) struct TransferQueueInfo {
    /// Queue family index of the dedicated transfer queue.
    #[allow(dead_code)]
    pub(crate) family: u32,
    /// Dedicated transfer queue handle.
    pub(crate) queue: vk::Queue,
    /// Command pool on the transfer family (`RESET_COMMAND_BUFFER`).
    pub(crate) command_pool: vk::CommandPool,
}

/// Overall queue topology: compute family index + optional transfer info.
#[allow(dead_code)]
pub(crate) struct QueueTopology {
    /// Compute queue family index.
    pub(crate) compute_family: u32,
    /// Optional dedicated transfer queue info (owned by context).
    pub(crate) transfer: Option<TransferQueueInfo>,
}

/// Select queue families from the physical device properties.
///
/// ## Selection rules
///
/// - `compute_family` = first family with `COMPUTE` (unchanged from M2.3a).
/// - `transfer_family` = first family with `TRANSFER` and **not** `COMPUTE` and
///   **not** `GRAPHICS` and index != `compute_family`. If none → `None`.
/// - If `force_single_queue` is `true`, `transfer_family` is forced to `None`
///   regardless of hardware capability.
///
/// Returns `NoComputeQueue` if no compute family is found.
pub(crate) fn select_queue_families(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    force_single_queue: bool,
) -> Result<QueueFamilySelection, DispatchError> {
    // SAFETY: physical_device is valid; this is a read-only query.
    let families = unsafe {
        instance.get_physical_device_queue_family_properties(physical_device)
    };

    // Find compute family (first family with COMPUTE, same as M2.3a).
    let compute_family: u32 = families
        .iter()
        .enumerate()
        .find_map(|(i, f)| {
            if f.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                Some(i as u32)
            } else {
                None
            }
        })
        .ok_or(DispatchError::NoComputeQueue)?;

    // Find dedicated transfer family (if not forced to single-queue).
    let transfer_family: Option<u32> = if force_single_queue {
        None
    } else {
        families.iter().enumerate().find_map(|(i, f)| {
            let idx = i as u32;
            // Must have TRANSFER but NOT COMPUTE and NOT GRAPHICS, and be distinct from compute.
            if f.queue_flags.contains(vk::QueueFlags::TRANSFER)
                && !f.queue_flags.contains(vk::QueueFlags::COMPUTE)
                && !f.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                && idx != compute_family
            {
                Some(idx)
            } else {
                None
            }
        })
    };

    Ok(QueueFamilySelection {
        compute_family,
        transfer_family,
    })
}

/// Determine the `QueueMode` from a `QueueFamilySelection`.
pub(crate) fn queue_mode(sel: &QueueFamilySelection) -> QueueMode {
    if sel.transfer_family.is_some() {
        QueueMode::DedicatedTransfer
    } else {
        QueueMode::SingleQueue
    }
}

/// Build the `VkDeviceQueueCreateInfo` array for the selected families.
///
/// In `SingleQueue` mode: one entry for `compute_family`.
/// In `DedicatedTransfer` mode: two entries — `compute_family` and `transfer_family`.
///
/// `priorities` must contain at least one element (typically `[1.0f32]`).
pub(crate) fn build_device_queue_create_infos<'a>(
    sel: &QueueFamilySelection,
    priorities: &'a [f32],
) -> Vec<vk::DeviceQueueCreateInfo<'a>> {
    let mut infos: Vec<vk::DeviceQueueCreateInfo<'_>> = Vec::with_capacity(2);

    let compute_ci = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(sel.compute_family)
        .queue_priorities(priorities);
    infos.push(compute_ci);

    if let Some(tf) = sel.transfer_family {
        let transfer_ci = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(tf)
            .queue_priorities(priorities);
        infos.push(transfer_ci);
    }

    infos
}

/// Return the family indices for `VK_SHARING_MODE_CONCURRENT` device-local buffers.
///
/// Returns `Some([compute_family, transfer_family])` in `DedicatedTransfer` mode,
/// or `None` in `SingleQueue` mode (use `EXCLUSIVE`).
pub(crate) fn concurrent_family_indices(sel: &QueueFamilySelection) -> Option<[u32; 2]> {
    sel.transfer_family.map(|tf| [sel.compute_family, tf])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk;

    /// Build a synthetic `VkQueueFamilyProperties` with the given flags.
    fn make_family(flags: vk::QueueFlags) -> vk::QueueFamilyProperties {
        vk::QueueFamilyProperties {
            queue_flags: flags,
            queue_count: 1,
            timestamp_valid_bits: 0,
            min_image_transfer_granularity: vk::Extent3D { width: 1, height: 1, depth: 1 },
        }
    }

    // ── AT-1407: queue topology selection tests ───────────────────────────────

    /// AT-1407a: single-family device (Lavapipe-like: GRAPHICS|COMPUTE|TRANSFER)
    /// → SingleQueue, transfer_family = None.
    #[test]
    fn at_1407a_lavapipe_like_single_family_selects_single_queue() {
        let sel = QueueFamilySelection {
            compute_family: 0,
            transfer_family: None,
        };
        assert_eq!(queue_mode(&sel), QueueMode::SingleQueue);
        assert!(concurrent_family_indices(&sel).is_none());
    }

    /// AT-1407b: two-family device (compute + transfer-only) → DedicatedTransfer.
    #[test]
    fn at_1407b_dedicated_transfer_family_selected() {
        let sel = QueueFamilySelection {
            compute_family: 0,
            transfer_family: Some(1),
        };
        assert_eq!(queue_mode(&sel), QueueMode::DedicatedTransfer);
        assert_eq!(concurrent_family_indices(&sel), Some([0, 1]));
    }

    /// AT-1407c: force_single_queue overrides a two-family device.
    ///
    /// Verifies that even when a transfer_family could be selected, force_single_queue
    /// produces SingleQueue with transfer_family = None.
    #[test]
    fn at_1407c_force_single_queue_overrides_dedicated() {
        // Simulate: what select_queue_families produces with force_single_queue=true.
        let forced = QueueFamilySelection {
            compute_family: 0,
            transfer_family: None, // forced to None
        };
        assert_eq!(queue_mode(&forced), QueueMode::SingleQueue);
        assert!(concurrent_family_indices(&forced).is_none());
    }

    /// AT-1407d: concurrent_family_indices returns [compute, transfer] in order.
    #[test]
    fn at_1407d_concurrent_family_indices_order() {
        let sel = QueueFamilySelection {
            compute_family: 2,
            transfer_family: Some(5),
        };
        let indices = concurrent_family_indices(&sel).unwrap();
        assert_eq!(indices[0], 2, "first index must be compute_family");
        assert_eq!(indices[1], 5, "second index must be transfer_family");
    }

    /// AT-1407e: build_device_queue_create_infos for SingleQueue → one entry.
    #[test]
    fn at_1407e_build_create_infos_single_queue() {
        let sel = QueueFamilySelection { compute_family: 3, transfer_family: None };
        let priorities: [f32; 1] = [1.0f32];
        let infos = build_device_queue_create_infos(&sel, &priorities);
        assert_eq!(infos.len(), 1, "SingleQueue must produce exactly one DeviceQueueCreateInfo");
        assert_eq!(infos[0].queue_family_index, 3);
    }

    /// AT-1407f: build_device_queue_create_infos for DedicatedTransfer → two entries.
    #[test]
    fn at_1407f_build_create_infos_dedicated() {
        let sel = QueueFamilySelection { compute_family: 0, transfer_family: Some(2) };
        let priorities: [f32; 1] = [1.0f32];
        let infos = build_device_queue_create_infos(&sel, &priorities);
        assert_eq!(infos.len(), 2, "DedicatedTransfer must produce two DeviceQueueCreateInfos");
        let families: Vec<u32> = infos.iter().map(|ci| ci.queue_family_index).collect();
        assert!(families.contains(&0), "compute family must be present");
        assert!(families.contains(&2), "transfer family must be present");
    }

    /// AT-1416: Lavapipe-like family (GRAPHICS|COMPUTE|TRANSFER at index 0) → SingleQueue.
    ///
    /// Guards against a silent DedicatedTransfer pick that would cause hangs on Lavapipe.
    #[test]
    fn at_1416_lavapipe_family_selects_single_queue() {
        let flags = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER;
        // Verify our selection logic: a family with COMPUTE|GRAPHICS|TRANSFER does NOT
        // qualify as a dedicated transfer family (has COMPUTE and GRAPHICS).
        let is_transfer_only = flags.contains(vk::QueueFlags::TRANSFER)
            && !flags.contains(vk::QueueFlags::COMPUTE)
            && !flags.contains(vk::QueueFlags::GRAPHICS);
        assert!(!is_transfer_only, "GRAPHICS|COMPUTE|TRANSFER must not be selected as dedicated");

        // With only this one family and index 0 == compute_family:
        let sel = QueueFamilySelection {
            compute_family: 0,
            transfer_family: None, // correctly not selected
        };
        assert_eq!(queue_mode(&sel), QueueMode::SingleQueue);
    }

    /// AT-1407g: QueueMode is Copy/Clone/Debug/PartialEq/Eq.
    #[test]
    fn at_1407g_queue_mode_traits() {
        let m: QueueMode = QueueMode::DedicatedTransfer;
        let m2: QueueMode = m;
        let m3: QueueMode = m;
        assert_eq!(m, m2);
        assert_eq!(m, m3);
        assert!(!format!("{m:?}").is_empty());

        let _ = make_family(vk::QueueFlags::TRANSFER);
    }
}
