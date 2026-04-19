//! AXIOM-Compute Vulkan/OpenCL runtime dispatcher — placeholder for M1.
//!
//! Real content (kernel upload, dispatch, synchronization, Lavapipe fallback)
//! arrives at M1. This stub exists to keep the workspace coherent.

/// Placeholder function. Returns a deterministic sentinel string.
///
/// Will be removed when M1 Vulkan dispatcher logic replaces this module.
pub fn placeholder() -> &'static str {
    "axc-runtime placeholder (M1)"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_returns_sentinel() {
        assert_eq!(placeholder(), "axc-runtime placeholder (M1)");
    }
}
