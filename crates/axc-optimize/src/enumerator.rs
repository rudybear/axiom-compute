//! M2.3: Strategy-hole enumerator for AXIOM-Compute.
//!
//! Enumerates the Cartesian product of `@strategy` hole candidates and
//! produces `StrategyVariant` descriptors that the grid-search harness can
//! dispatch individually.
//!
//! ## Determinism guarantee
//!
//! `StrategyHoles::map` is a `BTreeMap`, so keys are always iterated in
//! alphabetical order.  Candidate lists preserve source-order (index 0 is the
//! baseline assignment).  The mixed-radix counter therefore walks:
//!
//! ```text
//! key₀=candidates₀[0], key₁=candidates₁[0], …   (variant 0, baseline)
//! key₀=candidates₀[0], key₁=candidates₁[1], …   (variant 1)
//! …
//! ```
//!
//! This order is deterministic across runs and independent of Rust hash
//! randomisation.
//!
//! ## `variant_id` encoding
//!
//! For each (name, value) pair in alphabetical key order:
//!
//! ```text
//! name_len_le: u32  (little-endian)
//! name_bytes       (UTF-8)
//! value_le:    i64  (little-endian)
//! ```
//!
//! All bytes are fed into xxh3_64.  This gives a 64-bit fingerprint that is
//! stable across platforms (xxh3 is endian-neutral, but we normalise to LE
//! anyway for portability).

use std::collections::BTreeMap;
use xxhash_rust::xxh3::xxh3_64;
use axc_hir::hir::{Kernel, KernelAnnotations, Module as HirModule, StrategyHoles, WorkgroupDims};
use crate::EnumerateError;

/// Soft limit: warn (but don't fail) when the Cartesian product exceeds this.
pub const CARTESIAN_WARN_THRESHOLD: u64 = 100;

/// A single resolved assignment of strategy holes to concrete values.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StrategyAssignments {
    /// Alphabetically-ordered (hole name → concrete value) map.
    pub values: BTreeMap<String, i64>,
}

/// One enumerated variant: an ordinal index, a stable fingerprint, and the
/// concrete assignments.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StrategyVariant {
    /// Zero-based ordinal within the Cartesian product (stable across runs).
    pub ordinal: u64,
    /// xxh3_64 fingerprint of the canonical encoding (see module doc).
    pub variant_id: u64,
    /// Concrete hole assignments for this variant.
    pub assignments: StrategyAssignments,
}

/// Enumerate all variants from a `StrategyHoles` map.
///
/// Returns `EnumerateError::EmptyStrategy` if the map is empty.
/// Variants are yielded in mixed-radix order (alphabetical keys, source-order
/// candidates within each key).
pub fn enumerate_strategy(holes: &StrategyHoles) -> Result<Vec<StrategyVariant>, EnumerateError> {
    if holes.map.is_empty() {
        return Err(EnumerateError::EmptyStrategy);
    }

    // Collect keys and candidate slices in alphabetical order (BTreeMap iteration).
    let keys: Vec<&str> = holes.map.keys().map(|s| s.as_str()).collect();
    let candidates: Vec<&[i64]> = keys.iter()
        .map(|k| holes.map[*k].as_slice())
        .collect();
    let radices: Vec<usize> = candidates.iter().map(|c| c.len()).collect();

    // Total product (u64 to avoid overflow for large radices).
    let total: u64 = radices.iter().fold(1u64, |acc, &r| acc.saturating_mul(r as u64));

    let mut variants: Vec<StrategyVariant> = Vec::with_capacity(total.min(4096) as usize);

    // Mixed-radix counter: indices[i] is the current index into candidates[i].
    let mut indices: Vec<usize> = vec![0; keys.len()];

    for ordinal in 0..total {
        // Build assignment map and canonical encoding buffer for this variant.
        let mut assignment_map: BTreeMap<String, i64> = BTreeMap::new();
        let mut encoding_buf: Vec<u8> = Vec::with_capacity(keys.len() * 16);

        for (i, key) in keys.iter().enumerate() {
            let value: i64 = candidates[i][indices[i]];
            assignment_map.insert(key.to_string(), value);

            // Canonical encoding: u32-LE name_len + UTF-8 name bytes + i64-LE value.
            let name_bytes: &[u8] = key.as_bytes();
            let name_len: u32 = name_bytes.len() as u32;
            encoding_buf.extend_from_slice(&name_len.to_le_bytes());
            encoding_buf.extend_from_slice(name_bytes);
            encoding_buf.extend_from_slice(&value.to_le_bytes());
        }

        let variant_id: u64 = xxh3_64(&encoding_buf);

        variants.push(StrategyVariant {
            ordinal,
            variant_id,
            assignments: StrategyAssignments { values: assignment_map },
        });

        // Advance mixed-radix counter (least-significant = last key, rightmost).
        let mut carry: bool = true;
        for i in (0..keys.len()).rev() {
            if carry {
                indices[i] += 1;
                if indices[i] < radices[i] {
                    carry = false;
                } else {
                    indices[i] = 0;
                    // carry remains true, propagates left
                }
            }
        }
    }

    Ok(variants)
}

/// Apply `assignments` to a `Kernel`, producing a resolved kernel ready for
/// codegen.
///
/// Resolution currently applies to `@workgroup(x, y, z)` dimensions: any
/// placeholder dimension stored as `1` due to a `HoleRef` is replaced by the
/// concrete value from `assignments`.
///
/// Returns `EnumerateError::UnknownHole` if an assignment key names a hole
/// that isn't present in `kernel.annotations.strategy`.
/// Returns `EnumerateError::NegativeWorkgroupDim` if a candidate resolves to
/// a non-positive workgroup dimension.
///
/// On success, the returned kernel has `annotations.strategy = None` so it
/// passes the codegen backstop guard.
pub fn resolve_single_variant(
    kernel: &Kernel,
    assignments: &StrategyAssignments,
) -> Result<Kernel, EnumerateError> {
    // Verify all assignment keys are known holes.
    if let Some(ref strategy) = kernel.annotations.strategy {
        for name in assignments.values.keys() {
            if !strategy.map.contains_key(name) {
                return Err(EnumerateError::UnknownHole { name: name.clone() });
            }
        }
    }

    // Build resolved workgroup dims by substituting HoleRef placeholders.
    // We re-use the original dims; if holes controlled workgroup dims, those
    // dims were stored as 1 (placeholder) in the HIR.  We detect this by
    // looking at the strategy map for known workgroup-dimension keys.
    let orig_wg: &WorkgroupDims = &kernel.annotations.workgroup;
    let resolved_x: u32 = resolve_wg_dim("workgroup_x", orig_wg.x, assignments)?;
    let resolved_y: u32 = resolve_wg_dim("workgroup_y", orig_wg.y, assignments)?;
    let resolved_z: u32 = resolve_wg_dim("workgroup_z", orig_wg.z, assignments)?;

    let new_annotations: KernelAnnotations = KernelAnnotations {
        workgroup: WorkgroupDims {
            x: resolved_x,
            y: resolved_y,
            z: resolved_z,
        },
        intent: kernel.annotations.intent.clone(),
        complexity: kernel.annotations.complexity.clone(),
        preconditions: kernel.annotations.preconditions.clone(),
        subgroup_uniform: kernel.annotations.subgroup_uniform,
        cooperative_matrix: kernel.annotations.cooperative_matrix,
        coop_matrix: kernel.annotations.coop_matrix,
        // Resolution clears strategy → None so codegen backstop is satisfied.
        strategy: None,
        debug_checks: kernel.annotations.debug_checks.clone(),
        opt_log: kernel.annotations.opt_log.clone(),
    };

    Ok(Kernel {
        id: kernel.id,
        name: kernel.name.clone(),
        annotations: new_annotations,
        params: kernel.params.clone(),
        binding_plan: kernel.binding_plan.clone(),
        body: kernel.body.clone(),
        span: kernel.span,
    })
}

/// Resolve a single workgroup dimension, substituting from `assignments` when
/// the convention key is present, otherwise keeping the original value.
fn resolve_wg_dim(
    key: &str,
    original: u32,
    assignments: &StrategyAssignments,
) -> Result<u32, EnumerateError> {
    if let Some(&value) = assignments.values.get(key) {
        if value <= 0 {
            return Err(EnumerateError::NegativeWorkgroupDim {
                key: key.to_string(),
                value,
            });
        }
        return Ok(value as u32);
    }
    Ok(original)
}

/// M3.21 (FG.9): union the `@strategy` hole maps across ALL kernels of a module.
///
/// A hole name declared in one kernel's `@strategy` block is available
/// module-wide (referenced-but-not-declared is valid — that's the sharing
/// mechanism, §4 of the spec). Two rules enforced here, both fail-closed:
///
/// - The SAME hole name declared (with a candidate list) in >=2 kernels must
///   have a byte-identical candidate list in every declaration, else
///   `EnumerateError::ConflictingHoleCandidates`.
/// - No hole name may be a byte-prefix of another hole name in the union
///   (F1 defense-in-depth — the substitution itself is already
///   prefix-collision-safe; this refuses the ambiguous authoring pattern at
///   discovery time), else `EnumerateError::PrefixCollidingHoleNames`.
///
/// Kernels with no `@strategy` block contribute nothing. A module where NO
/// kernel declares any hole produces an empty `StrategyHoles` (callers that
/// require >=1 hole, e.g. the CLI `optimize` path, raise their own
/// `NoStrategy`-shaped error on the empty result).
pub fn union_module_holes(module: &HirModule) -> Result<StrategyHoles, EnumerateError> {
    let mut union_map: BTreeMap<String, Vec<i64>> = BTreeMap::new();

    for kernel in &module.kernels {
        let Some(ref strategy) = kernel.annotations.strategy else {
            continue;
        };
        for (name, candidates) in &strategy.map {
            match union_map.get(name) {
                Some(existing) if existing != candidates => {
                    return Err(EnumerateError::ConflictingHoleCandidates { name: name.clone() });
                }
                Some(_) => {}
                None => {
                    union_map.insert(name.clone(), candidates.clone());
                }
            }
        }
    }

    // Prefix-collision guard (F1 defense-in-depth): BTreeMap iterates keys in
    // sorted byte order, so any name that is a prefix of another is adjacent
    // to it in iteration order — a single linear scan over consecutive pairs
    // suffices (a prefix relation implies lexicographic adjacency for any
    // strings that share that prefix, since nothing can sort strictly between
    // a prefix and a string it prefixes).
    let names: Vec<&String> = union_map.keys().collect();
    for pair in names.windows(2) {
        let (shorter, longer) = (pair[0], pair[1]);
        if longer.starts_with(shorter.as_str()) {
            return Err(EnumerateError::PrefixCollidingHoleNames {
                shorter: shorter.clone(),
                longer: longer.clone(),
            });
        }
    }

    Ok(StrategyHoles { map: union_map })
}

/// Compute the total Cartesian product size for a set of holes.
///
/// Returns 0 if holes is empty.  Result saturates at `u64::MAX`.
pub fn cartesian_product_size(holes: &StrategyHoles) -> u64 {
    if holes.map.is_empty() {
        return 0;
    }
    holes.map.values().fold(1u64, |acc, candidates| {
        acc.saturating_mul(candidates.len() as u64)
    })
}

// ── Tests (AT-1014..AT-1020) ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use axc_hir::hir::{
        Kernel, KernelAnnotations, KernelBody, KernelId, StrategyHoles, WorkgroupDims,
    };
    use axc_hir::param::ParamBindingPlan;
    use axc_lexer::Span;
    use std::collections::BTreeMap;

    fn make_kernel_with_strategy(wg_x: u32, holes_map: BTreeMap<String, Vec<i64>>) -> Kernel {
        Kernel {
            id: KernelId(0),
            name: "test_kernel".to_string(),
            annotations: KernelAnnotations {
                workgroup: WorkgroupDims { x: wg_x, y: 1, z: 1 },
                intent: None,
                complexity: None,
                preconditions: Vec::new(),
                subgroup_uniform: false,
                cooperative_matrix: false,
                coop_matrix: None,
                strategy: if holes_map.is_empty() {
                    None
                } else {
                    Some(StrategyHoles { map: holes_map })
                },
                debug_checks: Vec::new(),
                opt_log: None,
            },
            params: Vec::new(),
            binding_plan: ParamBindingPlan {
                buffers: Vec::new(),
                scalars: Vec::new(),
                push_constant_total_bytes: 0,
            },
            body: KernelBody::Empty,
            span: Span::new(0, 1),
        }
    }

    /// AT-1014: enumerate_strategy returns error for empty strategy map.
    #[test]
    fn at_1014_enumerate_empty_strategy_returns_error() {
        let holes = StrategyHoles::new();
        let result = enumerate_strategy(&holes);
        assert!(
            matches!(result, Err(EnumerateError::EmptyStrategy)),
            "expected EmptyStrategy; got {result:?}"
        );
    }

    /// AT-1015: enumerate_strategy with single hole [32,64,128] produces 3 variants in order.
    #[test]
    fn at_1015_enumerate_single_hole_three_candidates() {
        let mut map: BTreeMap<String, Vec<i64>> = BTreeMap::new();
        map.insert("workgroup_x".to_string(), vec![32, 64, 128]);
        let holes = StrategyHoles { map };

        let variants = enumerate_strategy(&holes).expect("enumerate failed");
        assert_eq!(variants.len(), 3, "expected 3 variants");

        assert_eq!(variants[0].ordinal, 0);
        assert_eq!(variants[0].assignments.values["workgroup_x"], 32);
        assert_eq!(variants[1].ordinal, 1);
        assert_eq!(variants[1].assignments.values["workgroup_x"], 64);
        assert_eq!(variants[2].ordinal, 2);
        assert_eq!(variants[2].assignments.values["workgroup_x"], 128);
    }

    /// AT-1016: enumerate_strategy Cartesian product with two holes is correct.
    ///
    /// holes: { a: [1, 2], b: [10, 20] } → 4 variants
    /// Order: (a=1,b=10), (a=1,b=20), (a=2,b=10), (a=2,b=20)
    #[test]
    fn at_1016_enumerate_two_holes_cartesian_product() {
        let mut map: BTreeMap<String, Vec<i64>> = BTreeMap::new();
        map.insert("a".to_string(), vec![1, 2]);
        map.insert("b".to_string(), vec![10, 20]);
        let holes = StrategyHoles { map };

        let variants = enumerate_strategy(&holes).expect("enumerate failed");
        assert_eq!(variants.len(), 4);

        // BTreeMap alphabetical: a < b
        let expected: &[(i64, i64)] = &[(1, 10), (1, 20), (2, 10), (2, 20)];
        for (i, (ea, eb)) in expected.iter().enumerate() {
            assert_eq!(variants[i].assignments.values["a"], *ea, "variant {i} a");
            assert_eq!(variants[i].assignments.values["b"], *eb, "variant {i} b");
        }
    }

    /// AT-1017: variant_id is deterministic across calls.
    #[test]
    fn at_1017_variant_id_is_deterministic() {
        let mut map: BTreeMap<String, Vec<i64>> = BTreeMap::new();
        map.insert("workgroup_x".to_string(), vec![64, 128]);

        let v1 = enumerate_strategy(&StrategyHoles { map: map.clone() }).unwrap();
        let v2 = enumerate_strategy(&StrategyHoles { map: map.clone() }).unwrap();

        assert_eq!(v1[0].variant_id, v2[0].variant_id, "variant 0 id must be deterministic");
        assert_eq!(v1[1].variant_id, v2[1].variant_id, "variant 1 id must be deterministic");
    }

    /// AT-1018: variant_id differs between distinct assignments.
    #[test]
    fn at_1018_variant_ids_are_distinct() {
        let mut map: BTreeMap<String, Vec<i64>> = BTreeMap::new();
        map.insert("workgroup_x".to_string(), vec![32, 64, 128]);
        let holes = StrategyHoles { map };

        let variants = enumerate_strategy(&holes).unwrap();
        let id0 = variants[0].variant_id;
        let id1 = variants[1].variant_id;
        let id2 = variants[2].variant_id;
        assert_ne!(id0, id1, "variant 0 and 1 must have distinct ids");
        assert_ne!(id1, id2, "variant 1 and 2 must have distinct ids");
        assert_ne!(id0, id2, "variant 0 and 2 must have distinct ids");
    }

    /// AT-1019: resolve_single_variant applies workgroup_x assignment and clears strategy.
    #[test]
    fn at_1019_resolve_single_variant_applies_workgroup_x() {
        let mut holes_map: BTreeMap<String, Vec<i64>> = BTreeMap::new();
        holes_map.insert("workgroup_x".to_string(), vec![32, 64, 128]);
        // wg_x=1 is the placeholder (HoleRef → 1 in lower.rs).
        let kernel = make_kernel_with_strategy(1, holes_map);

        let mut assignment_values: BTreeMap<String, i64> = BTreeMap::new();
        assignment_values.insert("workgroup_x".to_string(), 64);
        let assignments = StrategyAssignments { values: assignment_values };

        let resolved = resolve_single_variant(&kernel, &assignments)
            .expect("resolve failed");

        assert_eq!(resolved.annotations.workgroup.x, 64,
            "workgroup.x must be resolved to 64");
        assert!(resolved.annotations.strategy.is_none(),
            "strategy must be None after resolution");
    }

    /// AT-1020: resolve_single_variant rejects negative workgroup dim.
    #[test]
    fn at_1020_resolve_rejects_negative_workgroup_dim() {
        let mut holes_map: BTreeMap<String, Vec<i64>> = BTreeMap::new();
        holes_map.insert("workgroup_x".to_string(), vec![-1]);
        let kernel = make_kernel_with_strategy(1, holes_map);

        let mut assignment_values: BTreeMap<String, i64> = BTreeMap::new();
        assignment_values.insert("workgroup_x".to_string(), -1);
        let assignments = StrategyAssignments { values: assignment_values };

        let result = resolve_single_variant(&kernel, &assignments);
        assert!(
            matches!(result, Err(EnumerateError::NegativeWorkgroupDim { .. })),
            "expected NegativeWorkgroupDim; got {result:?}"
        );
    }

    /// AT-1021: ordinal 0 is always the baseline (first candidate of each hole).
    #[test]
    fn at_1021_ordinal_0_is_baseline() {
        let mut map: BTreeMap<String, Vec<i64>> = BTreeMap::new();
        map.insert("workgroup_x".to_string(), vec![32, 64, 128, 256]);
        map.insert("workgroup_y".to_string(), vec![1, 2, 4]);
        let holes = StrategyHoles { map };

        let variants = enumerate_strategy(&holes).unwrap();
        let baseline = &variants[0];
        assert_eq!(baseline.ordinal, 0);
        assert_eq!(baseline.assignments.values["workgroup_x"], 32,
            "baseline workgroup_x must be first candidate (32)");
        assert_eq!(baseline.assignments.values["workgroup_y"], 1,
            "baseline workgroup_y must be first candidate (1)");
    }

    /// cartesian_product_size returns 0 for empty holes.
    #[test]
    fn cartesian_size_empty_is_zero() {
        let holes = StrategyHoles::new();
        assert_eq!(cartesian_product_size(&holes), 0);
    }

    /// cartesian_product_size returns correct product.
    #[test]
    fn cartesian_size_two_holes() {
        let mut map: BTreeMap<String, Vec<i64>> = BTreeMap::new();
        map.insert("a".to_string(), vec![1, 2, 3]);
        map.insert("b".to_string(), vec![10, 20]);
        let holes = StrategyHoles { map };
        assert_eq!(cartesian_product_size(&holes), 6);
    }

    // ── AT-1036..AT-1039: Additional enumerator tests ─────────────────────────

    /// AT-1036: CARTESIAN_WARN_THRESHOLD is 100.
    #[test]
    fn at_1036_cartesian_warn_threshold_is_100() {
        assert_eq!(CARTESIAN_WARN_THRESHOLD, 100);
    }

    /// AT-1037: variant_id encoding is xxh3_64 of canonical bytes.
    ///
    /// Manually compute expected bytes for workgroup_x=64 and verify.
    #[test]
    fn at_1037_variant_id_canonical_encoding() {
        use xxhash_rust::xxh3::xxh3_64;

        // Single hole: workgroup_x=64.
        // Canonical encoding: u32-LE name_len + UTF-8 name + i64-LE value.
        let name = b"workgroup_x";
        let value: i64 = 64;
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&(name.len() as u32).to_le_bytes());
        buf.extend_from_slice(name);
        buf.extend_from_slice(&value.to_le_bytes());
        let expected_id: u64 = xxh3_64(&buf);

        let mut map: BTreeMap<String, Vec<i64>> = BTreeMap::new();
        map.insert("workgroup_x".to_string(), vec![32, 64, 128]);
        let holes = StrategyHoles { map };
        let variants = enumerate_strategy(&holes).unwrap();

        // ordinal 1 = workgroup_x=64
        assert_eq!(variants[1].variant_id, expected_id,
            "variant_id must match manual xxh3_64 computation");
    }

    /// AT-1038: resolve_single_variant with unknown hole returns UnknownHole error.
    #[test]
    fn at_1038_resolve_rejects_unknown_hole_name() {
        let mut holes_map: BTreeMap<String, Vec<i64>> = BTreeMap::new();
        holes_map.insert("workgroup_x".to_string(), vec![64]);
        let kernel = make_kernel_with_strategy(1, holes_map);

        let mut assignment_values: BTreeMap<String, i64> = BTreeMap::new();
        assignment_values.insert("nonexistent_hole".to_string(), 64);
        let assignments = StrategyAssignments { values: assignment_values };

        let result = resolve_single_variant(&kernel, &assignments);
        assert!(
            matches!(result, Err(crate::EnumerateError::UnknownHole { ref name }) if name == "nonexistent_hole"),
            "expected UnknownHole for unknown name; got {result:?}"
        );
    }

    /// AT-1039: enumerate_strategy with three holes produces correct total variant count.
    #[test]
    fn at_1039_three_holes_cartesian_product_count() {
        let mut map: BTreeMap<String, Vec<i64>> = BTreeMap::new();
        map.insert("a".to_string(), vec![1, 2]);
        map.insert("b".to_string(), vec![10, 20, 30]);
        map.insert("c".to_string(), vec![100, 200]);
        let holes = StrategyHoles { map };

        let variants = enumerate_strategy(&holes).unwrap();
        // 2 × 3 × 2 = 12
        assert_eq!(variants.len(), 12, "2×3×2 should produce 12 variants");

        // Verify alphabetical key ordering: a < b < c
        assert_eq!(variants[0].assignments.values["a"], 1);
        assert_eq!(variants[0].assignments.values["b"], 10);
        assert_eq!(variants[0].assignments.values["c"], 100);

        // Last variant: a=2, b=30, c=200
        assert_eq!(variants[11].assignments.values["a"], 2);
        assert_eq!(variants[11].assignments.values["b"], 30);
        assert_eq!(variants[11].assignments.values["c"], 200);
    }

    // ── M3.21 (FG.9): union_module_holes (AT-2953, AT-2954, AT-2972) ──────────

    fn kernel_with_strategy(name: &str, holes_map: BTreeMap<String, Vec<i64>>) -> Kernel {
        Kernel {
            id: KernelId(0),
            name: name.to_string(),
            annotations: KernelAnnotations {
                workgroup: WorkgroupDims { x: 1, y: 1, z: 1 },
                intent: None,
                complexity: None,
                preconditions: Vec::new(),
                subgroup_uniform: false,
                cooperative_matrix: false,
                coop_matrix: None,
                strategy: if holes_map.is_empty() { None } else { Some(StrategyHoles { map: holes_map }) },
                debug_checks: Vec::new(),
                opt_log: None,
            },
            params: Vec::new(),
            binding_plan: axc_hir::param::ParamBindingPlan {
                buffers: Vec::new(),
                scalars: Vec::new(),
                push_constant_total_bytes: 0,
            },
            body: KernelBody::Empty,
            span: Span::new(0, 1),
        }
    }

    /// AT-2953: two kernels both declaring `?WG` with IDENTICAL candidates ->
    /// one merged axis (union has exactly one entry, with that candidate list).
    #[test]
    fn at_2953_union_identical_candidates_merges_to_one_axis() {
        let mut m1: BTreeMap<String, Vec<i64>> = BTreeMap::new();
        m1.insert("WG".to_string(), vec![64, 128, 256]);
        let mut m2: BTreeMap<String, Vec<i64>> = BTreeMap::new();
        m2.insert("WG".to_string(), vec![64, 128, 256]);

        let module = HirModule {
            kernels: vec![
                kernel_with_strategy("partial_reduce", m1),
                kernel_with_strategy("final_reduce", m2),
            ],
        };

        let union = union_module_holes(&module).expect("union must succeed for identical candidates");
        assert_eq!(union.map.len(), 1, "expected exactly one merged hole axis");
        assert_eq!(union.map["WG"], vec![64, 128, 256]);
    }

    /// AT-2954: same `?WG` name, DIFFERING candidate lists -> ConflictingHoleCandidates.
    #[test]
    fn at_2954_union_conflicting_candidates_fails_closed() {
        let mut m1: BTreeMap<String, Vec<i64>> = BTreeMap::new();
        m1.insert("WG".to_string(), vec![64, 128]);
        let mut m2: BTreeMap<String, Vec<i64>> = BTreeMap::new();
        m2.insert("WG".to_string(), vec![64, 256]);

        let module = HirModule {
            kernels: vec![
                kernel_with_strategy("a", m1),
                kernel_with_strategy("b", m2),
            ],
        };

        let result = union_module_holes(&module);
        assert!(
            matches!(result, Err(EnumerateError::ConflictingHoleCandidates { ref name }) if name == "WG"),
            "expected ConflictingHoleCandidates{{name:\"WG\"}}; got {result:?}"
        );
    }

    /// AT-2972: a module declaring both `?WG` and `?WG2` (prefix collision)
    /// fails closed at discovery, even though each individually is a valid hole.
    #[test]
    fn at_2972_union_prefix_colliding_names_fails_closed() {
        let mut m: BTreeMap<String, Vec<i64>> = BTreeMap::new();
        m.insert("WG".to_string(), vec![64]);
        m.insert("WG2".to_string(), vec![32]);

        let module = HirModule {
            kernels: vec![kernel_with_strategy("k", m)],
        };

        let result = union_module_holes(&module);
        assert!(
            matches!(
                result,
                Err(EnumerateError::PrefixCollidingHoleNames { ref shorter, ref longer })
                    if shorter == "WG" && longer == "WG2"
            ),
            "expected PrefixCollidingHoleNames{{shorter:\"WG\",longer:\"WG2\"}}; got {result:?}"
        );
    }

    /// Independent per-kernel holes with distinct names produce distinct axes
    /// (no interference — sharing is opt-in via name-matching only).
    #[test]
    fn union_distinct_hole_names_produce_independent_axes() {
        let mut m1: BTreeMap<String, Vec<i64>> = BTreeMap::new();
        m1.insert("A".to_string(), vec![1, 2]);
        let mut m2: BTreeMap<String, Vec<i64>> = BTreeMap::new();
        m2.insert("B".to_string(), vec![10, 20, 30]);

        let module = HirModule {
            kernels: vec![
                kernel_with_strategy("k1", m1),
                kernel_with_strategy("k2", m2),
            ],
        };

        let union = union_module_holes(&module).expect("union must succeed");
        assert_eq!(union.map.len(), 2);
        assert_eq!(union.map["A"], vec![1, 2]);
        assert_eq!(union.map["B"], vec![10, 20, 30]);
    }

    /// A module with no @strategy anywhere produces an empty union (not an error).
    #[test]
    fn union_no_strategy_anywhere_is_empty_not_error() {
        let module = HirModule {
            kernels: vec![kernel_with_strategy("k", BTreeMap::new())],
        };
        let union = union_module_holes(&module).expect("empty union must be Ok, not Err");
        assert!(union.map.is_empty());
    }
}
