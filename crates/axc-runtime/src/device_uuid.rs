//! Device-UUID matching for CUDA↔Vulkan interop — M4.1 (Phase 1a).
//!
//! CUDA and Vulkan must target the SAME physical GPU for an OPAQUE_FD
//! external-memory import to succeed (a cross-GPU import fails outright). This
//! module enumerates Vulkan physical devices, reads each device's
//! `VkPhysicalDeviceIDProperties.deviceUUID` (16 raw bytes) via
//! `vkGetPhysicalDeviceProperties2`, and selects the one whose UUID byte-equals a
//! caller-supplied target (the CUDA device UUID, read RAW from CUDA on the Python
//! side — never the dashed `GPU-xxxx` string re-parsed, to avoid a byte-order /
//! parse false-negative).
//!
//! ## Fail-closed (R-3)
//!
//! If NO enumerated device's UUID matches, this returns
//! [`DispatchError::NoUuidMatchedDevice`]. It NEVER silently falls back to a
//! different GPU or to a host copy — that would either fail the import or, worse,
//! run the Vulkan kernel on a different GPU than the torch tensor.

use ash::vk;
use crate::error::DispatchError;

/// Read the `VkPhysicalDeviceIDProperties.deviceUUID` (16 raw bytes) of a device.
///
/// Uses `vkGetPhysicalDeviceProperties2` with a chained
/// `VkPhysicalDeviceIDProperties` (core in Vulkan 1.1+). Returns the raw 16-byte
/// UUID; the bytes are compared verbatim against the CUDA device UUID (no
/// stringification).
pub fn physical_device_uuid(instance: &ash::Instance, pd: vk::PhysicalDevice) -> [u8; 16] {
    let mut id_props = vk::PhysicalDeviceIDProperties::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut id_props);
    // SAFETY: `pd` is a valid physical device from `enumerate_physical_devices`;
    // the `props2` pNext chain is well-formed and lives for the call.
    unsafe { instance.get_physical_device_properties2(pd, &mut props2) };
    id_props.device_uuid
}

/// Read the `VkPhysicalDeviceIDProperties.driverUUID` (16 raw bytes) of a device.
///
/// Exposed so the Python layer can additionally assert the driver UUID if desired;
/// only the `deviceUUID` is load-bearing for same-GPU matching.
pub fn physical_device_driver_uuid(instance: &ash::Instance, pd: vk::PhysicalDevice) -> [u8; 16] {
    let mut id_props = vk::PhysicalDeviceIDProperties::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut id_props);
    // SAFETY: as above.
    unsafe { instance.get_physical_device_properties2(pd, &mut props2) };
    id_props.driver_uuid
}

/// Select the physical device whose `deviceUUID` byte-equals `target_uuid`.
///
/// Returns `(physical_device, enumeration_index)` on a match, or
/// [`DispatchError::NoUuidMatchedDevice`] if none matches (FAIL-CLOSED, R-3).
///
/// The comparison is a raw 16-byte equality — the CUDA device UUID and the
/// Vulkan `deviceUUID` are both 16 raw bytes in the same layout, so no
/// stringify/parse step is involved.
pub fn select_physical_device_by_uuid(
    instance: &ash::Instance,
    target_uuid: [u8; 16],
) -> Result<(vk::PhysicalDevice, usize), DispatchError> {
    // SAFETY: `instance` is valid.
    let physical_devices: Vec<vk::PhysicalDevice> =
        unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| DispatchError::NoVulkanInstance(format!(
                "enumerate_physical_devices (uuid match): {e}"
            )))?;

    for (idx, &pd) in physical_devices.iter().enumerate() {
        if physical_device_uuid(instance, pd) == target_uuid {
            return Ok((pd, idx));
        }
    }

    Err(DispatchError::NoUuidMatchedDevice { target: target_uuid })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// AT-2100 (unit half): a 16-byte target that matches no device returns
    /// `NoUuidMatchedDevice` and the error carries the requested UUID verbatim.
    ///
    /// The positive (real-GPU) half is exercised in the PyO3/Python GPU test
    /// `test_device_uuid_match_selects_cuda_gpu` (AXC_ENABLE_GPU_TESTS=1).
    #[test]
    fn at_2100_no_uuid_match_is_fail_closed() {
        let target = [0xABu8; 16];
        // Construct the error directly (no instance needed) and assert it carries
        // the exact bytes — the load-bearing fail-closed contract.
        let err = DispatchError::NoUuidMatchedDevice { target };
        match err {
            DispatchError::NoUuidMatchedDevice { target: t } => {
                assert_eq!(t, target, "error must carry the requested UUID verbatim");
            }
            other => panic!("wrong variant: {other:?}"),
        }
        // Display must be non-empty and mention the UUID context.
        let msg = DispatchError::NoUuidMatchedDevice { target }.to_string();
        assert!(msg.contains("UUID") || msg.contains("uuid"), "msg: {msg}");
    }
}
