//! ICD probe helpers for Vulkan availability and GPU test gating.
//!
//! `probe_vulkan_available()` attempts to load the Vulkan entry and create a
//! throwaway instance. It is used by integration tests to skip gracefully when
//! no Vulkan ICD is installed.
//!
//! `gpu_tests_enabled()` reads the `AXC_ENABLE_GPU_TESTS` environment variable
//! and returns true only when it is set to the literal string `"1"`.
//!
//! The three-layer gate used by every integration test:
//! 1. `#[ignore]` attribute (cargo skips by default; `-- --ignored` re-enables)
//! 2. `if !gpu_tests_enabled() { ... return; }` — env var check
//! 3. `if !probe_vulkan_available() { ... return; }` — ICD probe

/// Try to load the Vulkan entry point and create a minimal instance.
///
/// Returns `true` if Vulkan is available and an instance can be created.
/// Returns `false` on any failure (library not found, driver not installed, etc.).
///
/// This probe creates and immediately destroys the instance — it does NOT
/// cache any state. Multiple calls are safe.
pub fn probe_vulkan_available() -> bool {
    // SAFETY: ash Entry::load() loads the Vulkan shared library via
    // the standard platform-specific search path. No invariants to uphold
    // beyond "the library must be loadable" which we handle via the Result.
    let entry = match unsafe { ash::Entry::load() } {
        Ok(e) => e,
        Err(_) => return false,
    };

    let app_info = ash::vk::ApplicationInfo::default()
        .api_version(ash::vk::API_VERSION_1_1);
    let create_info = ash::vk::InstanceCreateInfo::default()
        .application_info(&app_info);

    // SAFETY: create_info is valid for the duration of this call. We pass
    // no enabled layers or extensions that might require special handling.
    let instance = match unsafe { entry.create_instance(&create_info, None) } {
        Ok(inst) => inst,
        Err(_) => return false,
    };

    // SAFETY: instance was just created successfully; destroying it here is safe.
    unsafe { instance.destroy_instance(None); }
    true
}

/// Returns `true` if GPU integration tests should run.
///
/// Reads the `AXC_ENABLE_GPU_TESTS` environment variable. Returns `true` only
/// when the value is the literal string `"1"`. Any other value (including `"0"`,
/// `"true"`, `"yes"`, or unset) returns `false`.
///
/// This intentionally requires an explicit opt-in to prevent accidental GPU
/// allocation on developer laptops or CI runners without Lavapipe.
pub fn gpu_tests_enabled() -> bool {
    std::env::var("AXC_ENABLE_GPU_TESTS").as_deref() == Ok("1")
}

/// Return the Vulkan ICD path that the Loader will use for this process.
///
/// Reads `VK_DRIVER_FILES` first (Vulkan Loader 1.3.207+), then falls back to
/// the legacy `VK_ICD_FILENAMES`.  Returns an empty string when neither is set.
///
/// This is used by the M2.2 bench harness to populate the `vulkan_icd` field
/// in `baselines.json` (AT-709).
///
/// # Examples
///
/// ```
/// // When VK_DRIVER_FILES is set, captured_icd_path returns its value verbatim.
/// std::env::set_var("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json");
/// let path = axc_runtime::captured_icd_path();
/// assert!(!path.is_empty());
/// std::env::remove_var("VK_DRIVER_FILES");
/// ```
pub fn captured_icd_path() -> String {
    std::env::var("VK_DRIVER_FILES")
        .or_else(|_| std::env::var("VK_ICD_FILENAMES"))
        .unwrap_or_default()
}


#[cfg(test)]
mod tests {
    use super::*;

    /// AT-505: gpu_tests_enabled checks the env var for the literal "1".
    ///
    /// Tests all non-"1" values to ensure no false positives.
    #[test]
    fn at_505_gpu_tests_enabled_checks_env_literal_one() {
        // Save and restore the env var to avoid polluting other tests.
        let saved = std::env::var("AXC_ENABLE_GPU_TESTS").ok();

        // Set to "1" → true
        std::env::set_var("AXC_ENABLE_GPU_TESTS", "1");
        assert!(gpu_tests_enabled(), "AXC_ENABLE_GPU_TESTS=1 must return true");

        // Set to "0" → false
        std::env::set_var("AXC_ENABLE_GPU_TESTS", "0");
        assert!(!gpu_tests_enabled(), "AXC_ENABLE_GPU_TESTS=0 must return false");

        // Set to "true" → false (not the literal "1")
        std::env::set_var("AXC_ENABLE_GPU_TESTS", "true");
        assert!(!gpu_tests_enabled(), "AXC_ENABLE_GPU_TESTS=true must return false");

        // Set to "yes" → false
        std::env::set_var("AXC_ENABLE_GPU_TESTS", "yes");
        assert!(!gpu_tests_enabled(), "AXC_ENABLE_GPU_TESTS=yes must return false");

        // Set to "" (empty) → false
        std::env::set_var("AXC_ENABLE_GPU_TESTS", "");
        assert!(!gpu_tests_enabled(), "AXC_ENABLE_GPU_TESTS='' must return false");

        // Unset → false
        std::env::remove_var("AXC_ENABLE_GPU_TESTS");
        assert!(!gpu_tests_enabled(), "AXC_ENABLE_GPU_TESTS unset must return false");

        // Restore
        match saved {
            Some(v) => std::env::set_var("AXC_ENABLE_GPU_TESTS", v),
            None => std::env::remove_var("AXC_ENABLE_GPU_TESTS"),
        }
    }

    /// AT-520: probe_vulkan_available returns a bool without panicking.
    ///
    /// We cannot assert a specific value (depends on whether Lavapipe is installed)
    /// but we can assert it does not panic.
    #[test]
    fn at_520_probe_vulkan_available_does_not_panic() {
        // Just call it; whether it returns true or false depends on the environment.
        let _available: bool = probe_vulkan_available();
        // If we get here, it did not panic.
    }

    /// AT-709 unit test: captured_icd_path returns empty string when neither env var is set.
    #[test]
    fn at_709_captured_icd_path_empty_when_no_env() {
        let saved_driver = std::env::var("VK_DRIVER_FILES").ok();
        let saved_icd = std::env::var("VK_ICD_FILENAMES").ok();

        std::env::remove_var("VK_DRIVER_FILES");
        std::env::remove_var("VK_ICD_FILENAMES");

        let path = captured_icd_path();
        assert_eq!(path, "", "captured_icd_path must return empty string when neither env var is set");

        // Restore
        if let Some(v) = saved_driver { std::env::set_var("VK_DRIVER_FILES", v); }
        if let Some(v) = saved_icd { std::env::set_var("VK_ICD_FILENAMES", v); }
    }

    /// AT-709 unit test: captured_icd_path returns VK_DRIVER_FILES verbatim when set.
    #[test]
    fn at_709_captured_icd_path_returns_vk_driver_files_verbatim() {
        let saved_driver = std::env::var("VK_DRIVER_FILES").ok();
        let saved_icd = std::env::var("VK_ICD_FILENAMES").ok();

        let test_path = "/test/icd/path.json";
        std::env::set_var("VK_DRIVER_FILES", test_path);
        std::env::remove_var("VK_ICD_FILENAMES");

        let path = captured_icd_path();
        assert_eq!(path, test_path, "captured_icd_path must return VK_DRIVER_FILES verbatim");

        // Restore
        match saved_driver {
            Some(v) => std::env::set_var("VK_DRIVER_FILES", v),
            None => std::env::remove_var("VK_DRIVER_FILES"),
        }
        if let Some(v) = saved_icd { std::env::set_var("VK_ICD_FILENAMES", v); }
    }
}
