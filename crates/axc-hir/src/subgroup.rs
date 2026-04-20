//! HIR data types for subgroup operations and barriers (M1.4).
//!
//! `SubgroupOp` enumerates every subgroup builtin call that lowers to a
//! typed `HirExprKind::SubgroupBuiltin` node. `workgroup_barrier()` is a
//! STATEMENT (lowered to `HirStmt::Barrier`) and therefore NOT in `SubgroupOp`.
//!
//! `SubgroupReduceKind` distinguishes Add/Min/Max reductions.
//! `BarrierKind` distinguishes barrier scopes (Workgroup is the only variant in M1.4;
//! Subgroup is deferred to M1.5).

/// Subgroup operation variant for `HirExprKind::SubgroupBuiltin`.
///
/// `workgroup_barrier` is a STATEMENT (`HirStmt::Barrier`), not an expression op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubgroupOp {
    /// `subgroup_invocation_id() -> u32` — OpLoad of BuiltIn SubgroupLocalInvocationId.
    InvocationId,
    /// `subgroup_size() -> u32` — OpLoad of BuiltIn SubgroupSize.
    Size,
    /// `subgroup_elect() -> bool` — OpGroupNonUniformElect.
    Elect,
    /// `subgroup_reduce_add/min/max(v: T) -> T` — OpGroupNonUniform{I,F}{Add,SMin,UMin,...}.
    Reduce(SubgroupReduceKind),
    /// `subgroup_broadcast_first(v: T) -> T` — OpGroupNonUniformBroadcastFirst.
    BroadcastFirst,
    /// `subgroup_all(pred: bool) -> bool` — OpGroupNonUniformAll.
    All,
    /// `subgroup_any(pred: bool) -> bool` — OpGroupNonUniformAny.
    Any,
}

/// The reduce operation kind for `SubgroupOp::Reduce`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubgroupReduceKind {
    Add,
    Min,
    Max,
}

/// Barrier scope for `HirStmt::Barrier`.
///
/// `Subgroup` scope is deferred to M1.5.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierKind {
    /// `workgroup_barrier()` — OpControlBarrier with Workgroup execution/memory scope.
    Workgroup,
}

impl SubgroupOp {
    /// Return the canonical source-code name for this operation.
    pub fn source_name(self) -> &'static str {
        match self {
            SubgroupOp::InvocationId => "subgroup_invocation_id",
            SubgroupOp::Size => "subgroup_size",
            SubgroupOp::Elect => "subgroup_elect",
            SubgroupOp::Reduce(SubgroupReduceKind::Add) => "subgroup_reduce_add",
            SubgroupOp::Reduce(SubgroupReduceKind::Min) => "subgroup_reduce_min",
            SubgroupOp::Reduce(SubgroupReduceKind::Max) => "subgroup_reduce_max",
            SubgroupOp::BroadcastFirst => "subgroup_broadcast_first",
            SubgroupOp::All => "subgroup_all",
            SubgroupOp::Any => "subgroup_any",
        }
    }

    /// The number of arguments this op accepts at the call site.
    pub fn arity(self) -> usize {
        match self {
            SubgroupOp::InvocationId | SubgroupOp::Size | SubgroupOp::Elect => 0,
            SubgroupOp::Reduce(_)
            | SubgroupOp::BroadcastFirst
            | SubgroupOp::All
            | SubgroupOp::Any => 1,
        }
    }

    /// Returns `true` for collective ops that trigger `SubgroupOpInDivergentContext` warnings.
    ///
    /// Non-collective reads (`InvocationId`, `Size`) do NOT warn.
    /// `workgroup_barrier` is a STATEMENT and not consulted here.
    pub fn is_collective(self) -> bool {
        match self {
            SubgroupOp::Elect
            | SubgroupOp::Reduce(_)
            | SubgroupOp::BroadcastFirst
            | SubgroupOp::All
            | SubgroupOp::Any => true,
            SubgroupOp::InvocationId | SubgroupOp::Size => false,
        }
    }

    /// Parse a source-level call name into a `SubgroupOp`, or return `None` if
    /// the name is not a recognized subgroup builtin.
    pub fn from_source_name(name: &str) -> Option<SubgroupOp> {
        match name {
            "subgroup_invocation_id" => Some(SubgroupOp::InvocationId),
            "subgroup_size" => Some(SubgroupOp::Size),
            "subgroup_elect" => Some(SubgroupOp::Elect),
            "subgroup_reduce_add" => Some(SubgroupOp::Reduce(SubgroupReduceKind::Add)),
            "subgroup_reduce_min" => Some(SubgroupOp::Reduce(SubgroupReduceKind::Min)),
            "subgroup_reduce_max" => Some(SubgroupOp::Reduce(SubgroupReduceKind::Max)),
            "subgroup_broadcast_first" => Some(SubgroupOp::BroadcastFirst),
            "subgroup_all" => Some(SubgroupOp::All),
            "subgroup_any" => Some(SubgroupOp::Any),
            _ => None,
        }
    }
}
