//! AXIOM-Compute autotuner — placeholder for M2.
//!
//! Real content (LLM bridge, @strategy hole resolution, benchmark harness)
//! arrives at M2. This stub exists to keep the workspace coherent and
//! prevent churn when M2 crates are added.

/// Placeholder function. Returns a deterministic sentinel string.
///
/// Will be removed when M2 autotuner logic replaces this module.
pub fn placeholder() -> &'static str {
    "axc-optimize placeholder (M2)"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_returns_sentinel() {
        assert_eq!(placeholder(), "axc-optimize placeholder (M2)");
    }
}
