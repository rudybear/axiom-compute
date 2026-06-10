"""M4.3 — a ``torch.compile`` custom backend (``backend="axiom"``) that AUTO-LOWERS a
matching ``scaled_dot_product_attention`` (SDPA) to the merged ``torch.ops.axiom.flash_attention``
op (M4.1 Phase 4, :mod:`axiom_compute.attention_ops`).

A user writes a normal PyTorch model with ``F.scaled_dot_product_attention``, opts in via
``torch.compile(model, backend="axiom")``, and the matching attention is auto-routed to the
AXIOM coopmat kernel — auto-routing, not an explicit op call. Everything that does NOT match
(and every non-SDPA op) is delegated to the DEFAULT Inductor backend, UNCHANGED.

HONEST FRAMING (load-bearing, repeated here, in DESIGN §3.1.33, the demo, and every docstring):
  - **CAPABILITY demo, NOT a perf win.** torch's native SDPA dispatches to FlashAttention /
    memory-efficient / fused-math — all faster than AXIOM's coopmat-QKᵀ + scalar-softmax +
    coopmat-PV kernel. Auto-routing SDPA→AXIOM makes the model SLOWER than native. The value is:
    AXIOM CAN register as a torch.compile backend, auto-lower a matching op, and stay CORRECT in
    a real Dynamo/Inductor flow. NO beat claim anywhere; NO latency gate.
  - **OPT-IN ONLY.** ``backend="axiom"`` is never a default.
  - **q4km is OUT of auto-lowering scope.** ``torch.ops.axiom.q4km_matmul`` needs a pre-packed
    Q4_K_M int ``weights_id`` from a prior ``q4km_register_weights`` — an int registry handle, NOT
    a natural FX graph value. There is NO FX pattern that produces a Q4_K_M-packed weight, so
    q4km stays an EXPLICIT-op kernel; M4.3 does not auto-lower it.

THE MATCH PREDICATE is PROVABLY FAIL-CLOSED (the r2 design blocker fix). Every SDPA flag is read
through the single :func:`_get_sdpa_arg` helper (positional-or-keyword reads ``args[idx]`` if
present else the kwarg else the default; keyword-only reads the kwarg only), so a positionally
causal/masked SDPA (``F.sdpa(q,k,v,None,0.0,True)``) is READ from ``args[5]`` / ``args[3]`` and
REJECTED — never silently lowered to FULL non-causal attention (silent wrong math). A node
matches iff EVERY constraint provably holds (non-causal AND no mask AND dropout==0 AND
scale==1/√D AND f32 AND D==64 AND S%16==0 AND q.shape==k.shape==v.shape static AND not
requires_grad AND B*H<=AXIOM_SDPA_MAX_HEADS); any uncertainty (missing meta, symbolic shape,
unknown kwarg/form) ⇒ NO MATCH (fall back to Inductor). FROZEN 1e-3 vs eager (never loosens).

In-env torch (the Coder STEP-0 spike, 2026-06-10): torch 2.9.1+cu128; the SDPA node target is
``<built-in function scaled_dot_product_attention>`` == ``F.scaled_dot_product_attention`` ==
``torch._C._nn.scaled_dot_product_attention``; the FakeTensor meta lives in
``node.meta['example_value']`` (``val`` / ``tensor_meta`` ABSENT at the Dynamo backend boundary);
a positional-causal call arrives as ``args=(q,k,v,None,0.0,True)`` with empty kwargs (is_causal at
``args[5]``). The pins match the 2.9.1 design spike exactly.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Optional

AXIOM_BACKEND_NAME: str = "axiom"

# Scope guard: the per-head loop unrolls B*H op-call subgraphs. Over this bound ⇒ NO MATCH
# (fall back to Inductor) rather than unrolling an enormous graph. Documented, tunable.
AXIOM_SDPA_MAX_HEADS: int = 64

# The coopmat-leader head_dim pin (flash_attention.py:56). The op requires D==64.
_AXIOM_HEAD_DIM: int = 64
# The 16-key block (flash_attention.py:57). seq must be a positive multiple of 16.
_AXIOM_SEQ_MULTIPLE: int = 16

# The recognized SDPA kwarg keys (the schema's positional-or-keyword + keyword-only names). Any
# OTHER kwarg key ⇒ an unrecognized form ⇒ NO MATCH (P0 unknown-form fallback).
_KNOWN_SDPA_KWARGS = frozenset(
    {"attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa"}
)

# The maximum number of positional-or-keyword slots in the SDPA schema:
# (query, key, value, attn_mask, dropout_p, is_causal). scale/enable_gqa are KEYWORD-ONLY and can
# NEVER be positional ⇒ len(args) > 6 is an unknown/future form ⇒ NO MATCH (P0).
_MAX_SDPA_POSITIONAL = 6


# ── The two process-global fire counters (test hooks, NOT a correctness mechanism) ──────
_COUNTER_LOCK = threading.Lock()
# Compile-time: incremented once per SDPA node actually rewritten to the AXIOM op.
_AXIOM_SDPA_DISPATCH_COUNT = 0
# Runtime: incremented each time a rewritten subgraph actually CALLS the AXIOM op at runtime
# (proves the op FIRED, not merely that a node was swapped at compile).
_AXIOM_SDPA_RUNTIME_FIRE_COUNT = 0


def axiom_sdpa_dispatch_count() -> int:
    """Process-global count of SDPA nodes rewritten to ``torch.ops.axiom.flash_attention`` at
    compile time (the node-swap counter). Distinct from the RUNTIME fire count."""
    with _COUNTER_LOCK:
        return _AXIOM_SDPA_DISPATCH_COUNT


def axiom_sdpa_runtime_fire_count() -> int:
    """Process-global count of times a rewritten subgraph actually CALLED the AXIOM op at
    RUNTIME (proves the op FIRED end-to-end, not merely that a node was swapped)."""
    with _COUNTER_LOCK:
        return _AXIOM_SDPA_RUNTIME_FIRE_COUNT


def reset_axiom_sdpa_counters() -> None:
    """Test hygiene: zero both the compile-time dispatch counter and the runtime fire counter."""
    global _AXIOM_SDPA_DISPATCH_COUNT, _AXIOM_SDPA_RUNTIME_FIRE_COUNT
    with _COUNTER_LOCK:
        _AXIOM_SDPA_DISPATCH_COUNT = 0
        _AXIOM_SDPA_RUNTIME_FIRE_COUNT = 0


def _bump_dispatch_count() -> None:
    global _AXIOM_SDPA_DISPATCH_COUNT
    with _COUNTER_LOCK:
        _AXIOM_SDPA_DISPATCH_COUNT += 1


def _bump_runtime_fire_count() -> None:
    global _AXIOM_SDPA_RUNTIME_FIRE_COUNT
    with _COUNTER_LOCK:
        _AXIOM_SDPA_RUNTIME_FIRE_COUNT += 1


# ── The runtime-fire counting shim (a registered custom op so Inductor treats it as an opaque
# extern call and never inlines/elides it — the clean place to count a RUNTIME fire). It bumps
# the runtime counter then forwards to torch.ops.axiom.flash_attention VERBATIM. ──────────────
_SHIM_OP_NAME = "flash_attention_counted_shim"
_SHIM_REGISTERED = False


def _register_shim_op() -> None:
    """Idempotent: define ``torch.ops.axiom.flash_attention_counted_shim`` — a 1:1 wrapper of the
    flash op that ALSO bumps the runtime-fire counter. Registered as a custom op (NOT a plain
    Python function) so Dynamo/Inductor treat it as an opaque extern call that survives compile
    and is never re-traced or optimized away. Guarded for the no-torch surface."""
    global _SHIM_REGISTERED
    if _SHIM_REGISTERED:
        return
    try:
        import torch
        from torch.library import custom_op, register_fake
    except Exception:
        return
    if not hasattr(torch.library, "custom_op"):
        return

    @custom_op(
        f"axiom::{_SHIM_OP_NAME}",
        mutates_args=(),
        schema="(Tensor q, Tensor k, Tensor v, int seq_len, int head_dim) -> Tensor",
    )
    def flash_attention_counted_shim(q, k, v, seq_len, head_dim):
        # The runtime-fire proof: this body runs ONLY when the rewritten subgraph actually
        # executes the AXIOM attention at runtime.
        _bump_runtime_fire_count()
        return torch.ops.axiom.flash_attention(q, k, v, int(seq_len), int(head_dim))

    @register_fake(f"axiom::{_SHIM_OP_NAME}")
    def _flash_attention_counted_shim_fake(q, k, v, seq_len, head_dim):
        # Shape-only meta for torch.compile tracing — identical to the flash op's meta so the
        # surrounding graph traces. PURE (never bumps the counter under FakeTensorMode).
        return q.new_empty((int(seq_len), int(head_dim)), dtype=torch.float32)

    _SHIM_REGISTERED = True


# ── The match descriptor ──────────────────────────────────────────────────────────────
@dataclass
class _SdpaMatch:
    """A matched SDPA node descriptor: the q/k/v argument nodes + the static [B,H,S,D] dims."""

    q: object  # torch.fx.Node
    k: object  # torch.fx.Node
    v: object  # torch.fx.Node
    batch: int
    n_heads: int
    seq: int
    head_dim: int


# ── The single fail-closed flag extractor (THE r2 blocker fix) ──────────────────────────
def _get_sdpa_arg(node, idx: Optional[int], name: str, default):
    """Read ONE SDPA flag fail-closed.

    POSITIONAL-OR-KEYWORD param (attn_mask idx=3, dropout_p idx=4, is_causal idx=5): return
    ``node.args[idx]`` if it is present positionally, ELSE the kwarg, ELSE the ``default``.
    KEYWORD-ONLY param (scale, enable_gqa): pass ``idx=None`` ⇒ return ``node.kwargs.get(name,
    default)`` ONLY (never positional).

    This DEFEATS the silent-wrong-math hole: ``F.scaled_dot_product_attention(q,k,v,None,0.0,True)``
    ⇒ ``node.args=(q,k,v,None,0.0,True)``, ``node.kwargs={}`` ⇒ ``is_causal`` is READ from
    ``args[5]==True`` (NO MATCH), not the kwargs-only default ``False`` (which would silently lower
    a CAUSAL SDPA to FULL non-causal attention).
    """
    if idx is not None and len(node.args) > idx:
        return node.args[idx]
    return node.kwargs.get(name, default)


# ── The meta read — pinned to 'example_value' FIRST (the r2 meta-key fix) ────────────────
def _read_meta_tensor(node_or_arg):
    """Return the FakeTensor-like meta (with ``.shape``/``.dtype``/``.requires_grad``) for an FX
    node/arg, or ``None`` ⇒ caller falls back (NO MATCH).

    Tries ``arg.meta[k]`` for ``k in ['example_value','val','tensor_meta']`` IN ORDER —
    ``example_value`` FIRST: at the Dynamo BACKEND boundary the FakeTensor lives there
    (``val``/``tensor_meta`` ABSENT, verified live by the Coder STEP-0 spike + both design
    reviewers). The ``val``/``tensor_meta`` fallbacks cover post-AOT paths / torch-version drift.

    FAIL-CLOSED: if ``node_or_arg`` has no ``.meta``, all keys are absent, or the value is not a
    single tensor-like with ``.shape``/``.dtype``/``.requires_grad`` (e.g. a tuple/list/None/int),
    return ``None`` — the predicate NEVER reads a None-shape as a match.
    """
    meta = getattr(node_or_arg, "meta", None)
    if not isinstance(meta, dict):
        return None
    for key in ("example_value", "val", "tensor_meta"):
        if key not in meta:
            continue
        val = meta[key]
        # Must be a single tensor-like exposing shape/dtype/requires_grad. A tuple/list (multi-
        # output), None, or a bare int/SymInt ⇒ not a usable tensor meta ⇒ keep looking, then fail.
        if (
            hasattr(val, "shape")
            and hasattr(val, "dtype")
            and hasattr(val, "requires_grad")
        ):
            return val
    return None


def _static_shape(meta_tensor):
    """Return ``tuple(meta.shape)`` as plain Python ints, or ``None`` if ANY dim is symbolic /
    non-int (a dynamic shape ⇒ cannot prove ``S%16==0`` / ``D==64`` ⇒ NO MATCH)."""
    try:
        shape = meta_tensor.shape
    except Exception:
        return None
    dims = []
    for d in shape:
        # A SymInt / non-int dim is dynamic — int(SymInt) would specialize; instead reject it.
        if isinstance(d, bool) or not isinstance(d, int):
            return None
        dims.append(int(d))
    return tuple(dims)


# ── The EXACT match predicate (fire ONLY when provably correct) ─────────────────────────
def _known_sdpa_targets():
    """The pinned SDPA target set for the in-env torch. ``F.scaled_dot_product_attention`` IS
    ``torch._C._nn.scaled_dot_product_attention`` (same object — the Coder STEP-0 spike + both
    design reviewers); the aten overload covers an already-decomposed path. Returns a set of the
    available target objects (guarded so a torch-less import does not crash)."""
    targets = set()
    try:
        import torch
        import torch.nn.functional as F

        targets.add(F.scaled_dot_product_attention)
        targets.add(torch._C._nn.scaled_dot_product_attention)
        targets.add(torch.ops.aten.scaled_dot_product_attention.default)
    except Exception:
        pass
    return targets


def _sdpa_match(node) -> Optional[_SdpaMatch]:
    """The EXACT SDPA match predicate. Returns a :class:`_SdpaMatch` iff EVERY constraint provably
    holds, else ``None`` (the node is LEFT UNTOUCHED — Inductor handles it, no graph break, no
    error). Every flag is read via :func:`_get_sdpa_arg`; shapes via :func:`_read_meta_tensor`.

    Order: P0 unknown-form ⇒ P1 target ⇒ P2 flags ⇒ P3 dtype ⇒ P4 shape/GQA ⇒ P5 batch/head ⇒
    P-grad. ANY failure ⇒ ``None``.
    """
    # (P0) UNKNOWN-FORM FALLBACK — checked FIRST, before reading any flag.
    if getattr(node, "op", None) != "call_function":
        return None
    if node.target not in _known_sdpa_targets():
        return None
    # More positionals than the schema's 6 positional-or-keyword slots ⇒ a future/unknown form.
    if len(node.args) > _MAX_SDPA_POSITIONAL:
        return None
    # Fewer than the 3 required positional tensors (query, key, value) ⇒ unknown form.
    if len(node.args) < 3:
        return None
    # An unrecognized kwarg key ⇒ a form we do not understand ⇒ do NOT match.
    for kw_key in node.kwargs:
        if kw_key not in _KNOWN_SDPA_KWARGS:
            return None

    # (P2) FLAG EXTRACTION — the single fail-closed helper. POSITIONAL-OR-KEYWORD:
    attn_mask = _get_sdpa_arg(node, idx=3, name="attn_mask", default=None)
    dropout_p = _get_sdpa_arg(node, idx=4, name="dropout_p", default=0.0)
    is_causal = _get_sdpa_arg(node, idx=5, name="is_causal", default=False)
    # KEYWORD-ONLY (idx=None ⇒ kwargs only, NEVER positional):
    scale = _get_sdpa_arg(node, idx=None, name="scale", default=None)
    enable_gqa = _get_sdpa_arg(node, idx=None, name="enable_gqa", default=False)

    # MATCH requires (read via the helper, NOT by omission): no mask, no dropout, NON-causal,
    # GQA off. A non-None attn_mask (a literal OR an FX Node — a positional mask arrives as a
    # tensor node) ⇒ NO MATCH. ``is not False`` rejects True AND any truthy/Node value.
    if attn_mask is not None:
        return None
    if dropout_p != 0.0:
        return None
    if is_causal is not False:
        return None
    if enable_gqa is not False:
        return None

    # (P3/P4) META — read q/k/v shape/dtype/requires_grad (example_value first). The first 3
    # positional args are query, key, value.
    q_node, k_node, v_node = node.args[0], node.args[1], node.args[2]
    q_meta = _read_meta_tensor(q_node)
    k_meta = _read_meta_tensor(k_node)
    v_meta = _read_meta_tensor(v_node)
    if q_meta is None or k_meta is None or v_meta is None:
        return None  # missing/non-Tensor meta ⇒ fail-closed no match.

    try:
        import torch

        f32 = torch.float32
    except Exception:
        return None
    # (P3) DTYPE f32 for all three.
    if q_meta.dtype != f32 or k_meta.dtype != f32 or v_meta.dtype != f32:
        return None

    # (P-grad) inference-only — the op's register_fake has no backward.
    if q_meta.requires_grad or k_meta.requires_grad or v_meta.requires_grad:
        return None

    # (P4) SHAPE + the FOREGROUNDED GQA GUARD (the PRIMARY GQA/MQA defense).
    q_shape = _static_shape(q_meta)
    k_shape = _static_shape(k_meta)
    v_shape = _static_shape(v_meta)
    if q_shape is None or k_shape is None or v_shape is None:
        return None  # symbolic/dynamic dim ⇒ cannot prove S%16==0 / D==64 ⇒ no match.
    if len(q_shape) != 4:
        return None  # only the canonical 4-D [B,H,S,D] SDPA layout (3-D/other ranks out of scope).
    # PRIMARY GQA guard: full 4-tuple equality (B,H,S,D ALL equal). GQA/MQA (Hkv<Hq) fails on H.
    if not (q_shape == k_shape == v_shape):
        return None

    batch, n_heads, seq, head_dim = q_shape
    if head_dim != _AXIOM_HEAD_DIM:
        return None  # the coopmat-leader CORE pin.
    if seq <= 0 or seq % _AXIOM_SEQ_MULTIPLE != 0:
        return None  # the 16-key block.

    # (P2b) SCALE — the kernel applies inv_sqrt_d = 1/sqrt(head_dim) INTERNALLY. A custom scale
    # that differs would change the math ⇒ NO MATCH. (head_dim known now, after P4.)
    if scale is not None:
        try:
            expected = 1.0 / math.sqrt(head_dim)
        except Exception:
            return None
        if not isinstance(scale, (int, float)) or isinstance(scale, bool):
            return None  # a non-numeric / Node scale ⇒ unknown ⇒ no match.
        if not math.isclose(float(scale), expected, rel_tol=1e-6, abs_tol=1e-9):
            return None

    # (P5) BATCH/HEAD expressible — the op is single-head [S,D]; we unroll B*H op calls.
    if batch <= 0 or n_heads <= 0:
        return None
    if batch * n_heads > AXIOM_SDPA_MAX_HEADS:
        return None  # avoid unrolling an enormous loop ⇒ fall back.

    return _SdpaMatch(
        q=q_node,
        k=k_node,
        v=v_node,
        batch=batch,
        n_heads=n_heads,
        seq=seq,
        head_dim=head_dim,
    )


# ── The node rewrite (the [B,H,S,D] -> per-head [S,D] handling, math-faithful) ──────────
def _rewrite_one_sdpa(graph, node, match: _SdpaMatch) -> None:
    """Replace ONE matched SDPA ``node`` in ``graph`` with a per-head loop calling the AXIOM
    counting-shim op, stacked back to ``[B,H,S,D]``.

    For each of the B*H heads: reshape q/k/v ``[B,H,S,D]`` → ``[B*H,S,D]`` (a no-op view on
    contiguous input), ``select(0, idx)`` the ``[S,D]`` head, ``.contiguous()`` it (the op
    requires contiguous ``[S,D]``), call ``torch.ops.axiom.flash_attention_counted_shim`` →
    ``[S,D]``; stack the B*H outputs → ``[B*H,S,D]`` → reshape → ``[B,H,S,D]``.
    """
    import torch

    B, H, S, D = match.batch, match.n_heads, match.seq, match.head_dim
    bh = B * H

    with graph.inserting_before(node):
        # reshape each of q/k/v from [B,H,S,D] to [B*H,S,D] once (no-op on contiguous input).
        q_flat = graph.call_method("reshape", (match.q, bh, S, D))
        k_flat = graph.call_method("reshape", (match.k, bh, S, D))
        v_flat = graph.call_method("reshape", (match.v, bh, S, D))

        out_heads = []
        for i in range(bh):
            qi = graph.call_function(torch.select, (q_flat, 0, i))
            ki = graph.call_function(torch.select, (k_flat, 0, i))
            vi = graph.call_function(torch.select, (v_flat, 0, i))
            qi_c = graph.call_method("contiguous", (qi,))
            ki_c = graph.call_method("contiguous", (ki,))
            vi_c = graph.call_method("contiguous", (vi,))
            oi = graph.call_function(
                torch.ops.axiom.flash_attention_counted_shim,
                (qi_c, ki_c, vi_c, S, D),
            )
            out_heads.append(oi)

        stacked = graph.call_function(torch.stack, (out_heads,))  # [B*H, S, D]
        result = graph.call_method("reshape", (stacked, B, H, S, D))

    node.replace_all_uses_with(result)
    graph.erase_node(node)
    _bump_dispatch_count()


def _rewrite_sdpa_to_axiom(gm, example_inputs) -> int:
    """IN-PLACE mutate ``gm.graph``: rewrite every MATCHING SDPA node to the per-head AXIOM op
    loop. Returns the count of nodes lowered (0 ⇒ pure passthrough, NO behavior change). Each node
    is matched independently; non-matching nodes are left untouched. After the rewrites,
    ``gm.graph.lint()`` + ``gm.recompile()`` so the mutated graph is valid for Inductor."""
    _register_shim_op()  # ensure the counting-shim op exists before inserting calls to it.
    graph = gm.graph
    # Snapshot the node list — we mutate the graph (erase/insert) while iterating.
    matched = []
    for node in list(graph.nodes):
        match = _sdpa_match(node)
        if match is not None:
            matched.append((node, match))

    for node, match in matched:
        _rewrite_one_sdpa(graph, node, match)

    if matched:
        graph.lint()
        gm.recompile()
    return len(matched)


# ── The backend entry ───────────────────────────────────────────────────────────────────
def axiom_backend(gm, example_inputs):
    """The ``backend="axiom"`` entry. Dynamo calls this with ``(gm: torch.fx.GraphModule,
    example_inputs)``. We rewrite matching SDPA nodes to the AXIOM op IN-PLACE, then DELEGATE the
    rewritten graph to the DEFAULT Inductor backend so the user keeps Inductor's optimization on
    the rest of the model. HARD FALLBACK to ``gm.forward`` (eager of the rewritten graph) if
    Inductor is unavailable / raises — still correct, disclosed.

    HONEST: auto-routing matching SDPA to the AXIOM kernel is a CAPABILITY demo, SLOWER than
    native SDPA, OPT-IN only. NO beat claim.
    """
    _rewrite_sdpa_to_axiom(gm, example_inputs)
    try:
        import torch._inductor

        return torch._inductor.compile(gm, example_inputs)
    except Exception:
        # Inductor unavailable / raised on the (mutated) graph ⇒ eager of the rewritten graph.
        return gm.forward


# ── Registration (idempotent, guarded for the no-torch surface) ─────────────────────────
_BACKEND_REGISTERED = False


def register_axiom_backend() -> None:
    """Idempotent: ``torch._dynamo.register_backend(name="axiom", compiler_fn=axiom_backend)`` so
    ``torch.compile(model, backend="axiom")`` resolves after ``import axiom_compute``. Also
    registers the runtime-fire counting-shim op. Guarded so a torch-less import still succeeds
    (mirrors :func:`axiom_compute.attention_ops.register_flash_attention_op`)."""
    global _BACKEND_REGISTERED
    if _BACKEND_REGISTERED:
        return
    try:
        import torch._dynamo
    except Exception:
        return  # no torch / no dynamo — the non-torch surface still imports.

    _register_shim_op()
    try:
        torch._dynamo.register_backend(name=AXIOM_BACKEND_NAME, compiler_fn=axiom_backend)
    except Exception:
        # A duplicate registration (e.g. a prior partial import) is benign — the backend is
        # already resolvable. Any other failure leaves _BACKEND_REGISTERED False (retry next call).
        try:
            import torch._dynamo as _d

            if AXIOM_BACKEND_NAME not in _d.list_backends():
                return
        except Exception:
            return

    _BACKEND_REGISTERED = True


__all__ = [
    "AXIOM_BACKEND_NAME",
    "AXIOM_SDPA_MAX_HEADS",
    "register_axiom_backend",
    "axiom_backend",
    "axiom_sdpa_dispatch_count",
    "axiom_sdpa_runtime_fire_count",
    "reset_axiom_sdpa_counters",
]
