"""M4.1 Phase 4 — FlashAttention as a FIRST-CLASS PyTorch operator.

Registers ``torch.ops.axiom.flash_attention(Tensor q, Tensor k, Tensor v, int seq_len,
int head_dim) -> Tensor`` via :func:`torch.library.custom_op` (torch 2.12) so the merged
coopmat FlashAttention kernel (:class:`axiom_compute.flash_attention.FlashAttention`) (a) is a
registered op with a schema, (b) composes under ``torch.compile`` via :func:`register_fake`,
and (c) runs the existing 4-buffer zero-copy dispatch on CUDA tensors. This is the EXACT
structural twin of :mod:`axiom_compute.ops` (the Q4_K_M op) — it REUSES the R1/R2/R3 + PCR-1
machinery wholesale (no reinvention).

The op caches a :class:`FlashAttention`/``SharedSession`` keyed by
``(device_index, seq_len, head_dim, kernel)`` so torch's stateless-op view does NOT
re-export/re-import the 4 shared buffers per call (that would destroy the zero-copy
amortization). UNLIKE q4km there is NO upload-once weights step — Q/K/V are ALL per-call data.

R1 — the real CUDA impl FAIL-CLOSED-VALIDATES q/k/v are 2-D CUDA f32 ``[seq,head_dim]``,
     ``head_dim==64`` (coopmat), ``seq%16==0``, and the three shapes agree, BEFORE any dispatch,
     so register_fake's ``q.new_empty((seq,head_dim), f32)`` is PROVABLY identical (shape, dtype,
     device) to the real clone — a mismatch RAISES (no torch.compile miscompile).
R2 — on a wait_completion timeout / dispatch error the op RELEASES the dangling CUDA wait,
     POISONS + evicts the session (rebuilt next call), then raises (``_recover_faulted_session``
     REUSED VERBATIM from :mod:`axiom_compute.ops`).
R3 — the op EXECUTES its handshake + (device) copy + ``o.clone()`` inside
     ``with torch.cuda.stream(session.stream)`` with cross-stream events (ev_in caller→S covering
     ALL THREE inputs q/k/v, ev_out S→caller). It returns the CLONE (never the persistent shared
     O view). SAME-CALLER-STREAM PRECONDITION: a single ev_in recorded on the current caller
     stream orders S after all three input writes ONLY IF q, k, AND v were produced on the SAME
     caller stream before the op call (the standard eager-torch case). A k written on a DIFFERENT
     stream than the op's calling stream is out of scope (the q4km single-x assumption,
     generalized to three inputs).

Honest framing: NO beat-SDPA / beat-cuBLAS. The value is REAL torch integration of a SECOND
kernel + ``torch.compile`` composition. The zero-copy fast path is EAGER-scoped (under fullgraph
compile Inductor functionalization may break the data_ptr identity → the op correctly falls to
the device_copy path).
"""

from __future__ import annotations

import atexit
import threading
from typing import Optional

from . import AxiomError
from .flash_attention import FlashAttention
# Reuse the q4km recovery path VERBATIM (R2 + PCR-1) — no reinvention.
from .ops import _recover_faulted_session

AXIOM_OP_NAMESPACE: str = "axiom"
FLASH_ATTENTION_OP_NAME: str = "flash_attention"

# ── The (device, seq, head_dim, kernel)-keyed process-local registry ──────────────────
# `_FA_REGISTRY_LOCK` is held across the WHOLE op body (lookup → R1-validate → ensure_session
# → dispatch → wait_completion → clone → cross-stream hand-back) so a concurrent
# release_attention_session cannot close a session mid-dispatch, AND so no two op calls on the
# same key are in-flight concurrently (the mutates_args=() soundness invariant).
_FA_REGISTRY_LOCK = threading.Lock()
_FA_REGISTRY: dict[tuple, FlashAttention] = {}


def _fa_key(device_index: int, seq_len: int, head_dim: int, kernel: str) -> tuple:
    return (int(device_index), int(seq_len), int(head_dim), str(kernel))


def _device_index_of(t) -> int:
    idx = t.device.index
    if idx is None:
        import torch

        idx = torch.cuda.current_device()
    return int(idx)


def _get_or_build(key: tuple) -> FlashAttention:
    """Reuse the (cached, non-poisoned) FlashAttention for the key, or build a fresh one.

    R2: a POISONED session (a prior fault) is torn down + rebuilt so a faulted session + its
    stream S are never reused for a new dispatch. Caller MUST hold ``_FA_REGISTRY_LOCK``.
    """
    device_index, seq_len, head_dim, kernel = key
    op = _FA_REGISTRY.get(key)
    if op is not None and op.session is not None and op.session.is_poisoned:
        try:
            op.close()
        except Exception:
            pass
        op = None
        del _FA_REGISTRY[key]
    if op is None:
        op = FlashAttention(device=device_index, kernel=kernel)
        # Materialize the session now (allocate the 4 buffers + import the fds) so the
        # *_view() entries are available and subsequent calls reuse it.
        op.q_view(seq_len, head_dim)
        _FA_REGISTRY[key] = op
    return op


def flash_attention_session(
    device, seq_len: int, head_dim: int, kernel: str = "coopmat"
) -> FlashAttention:
    """Get-or-build the cached :class:`FlashAttention` for ``(device, seq_len, head_dim,
    kernel)``. ``device`` may be an int index, a torch device, or a CUDA tensor."""
    device_index = _resolve_device_index(device)
    key = _fa_key(device_index, seq_len, head_dim, kernel)
    with _FA_REGISTRY_LOCK:
        return _get_or_build(key)


def _resolve_device_index(device) -> int:
    import torch

    if device is None:
        return int(torch.cuda.current_device())
    if isinstance(device, int):
        return int(device)
    if isinstance(device, torch.Tensor):
        return _device_index_of(device)
    idx = torch.device(device).index
    return int(torch.cuda.current_device()) if idx is None else int(idx)


def flash_attention_views(device, seq_len: int, head_dim: int, kernel: str = "coopmat"):
    """The 4 zero-copy shared views ``(q_view, k_view, v_view, o_view)`` for the EAGER zero-copy
    entry. Write q AND k AND v directly into their views, then call the op with THOSE tensors for
    the staging-free 0/0 path. Under torch.compile the zero-copy identity may be lost to
    functionalization → the op falls to device_copy (disclosed)."""
    op = flash_attention_session(device, seq_len, head_dim, kernel)
    return (
        op.q_view(seq_len, head_dim),
        op.k_view(seq_len, head_dim),
        op.v_view(seq_len, head_dim),
        op.o_view(seq_len, head_dim),
    )


def last_attn_dispatch_path(device, seq_len: int, head_dim: int, kernel: str = "coopmat") -> str:
    """``'zero_copy'`` | ``'device_copy'`` — which path the most recent op call for this session
    took (honest path reporting, AT-2125)."""
    device_index = _resolve_device_index(device)
    key = _fa_key(device_index, seq_len, head_dim, kernel)
    with _FA_REGISTRY_LOCK:
        op = _FA_REGISTRY.get(key)
        if op is None or op.report_path is None:
            raise AxiomError(
                f"flash_attention op: no dispatch has run yet for {key}"
            )
        return op.report_path


def last_attn_dispatch_staging(device, seq_len: int, head_dim: int, kernel: str = "coopmat"):
    """``(staging_buffers_allocated, copy_buffer_records)`` of the most recent dispatch —
    asserts 0/0 on the zero-copy path (AT-2125)."""
    device_index = _resolve_device_index(device)
    key = _fa_key(device_index, seq_len, head_dim, kernel)
    with _FA_REGISTRY_LOCK:
        op = _FA_REGISTRY.get(key)
        if op is None or op.session is None:
            raise AxiomError(
                f"flash_attention op: no session has been built yet for {key}"
            )
        sess = op.session
        return (sess.staging_buffers_allocated(), sess.copy_buffer_records())


def release_attention_session(
    device, seq_len: int, head_dim: int, kernel: str = "coopmat"
) -> None:
    """G-4 teardown: drain → cudaDestroy → vkFree the session for the key + drop it. Idempotent.
    Takes the registry lock (so it cannot close a session mid-dispatch — it blocks until any
    in-flight op call completes)."""
    device_index = _resolve_device_index(device)
    key = _fa_key(device_index, seq_len, head_dim, kernel)
    with _FA_REGISTRY_LOCK:
        op = _FA_REGISTRY.pop(key, None)
    if op is not None:
        try:
            op.close()
        except Exception:
            pass


# ── The registered op (defined once at import, guarded for the no-torch surface) ──────
_REGISTERED = False


def register_flash_attention_op() -> None:
    """Idempotent: define ``torch.ops.axiom.flash_attention`` + register_fake + the CUDA impl.

    Guarded so importing the package WITHOUT torch (or with a torch lacking
    ``torch.library.custom_op``) still succeeds for the non-torch surface.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    try:
        import torch
        from torch.library import custom_op, register_fake
    except Exception:
        return  # no torch / no custom_op — the non-torch surface still imports.
    if not hasattr(torch.library, "custom_op"):
        return

    # An EXPLICIT schema string (not inferred): `from __future__ import annotations` turns the
    # function annotations into strings, which torch's infer_schema rejects — so we pin the
    # schema directly. mutates_args=() declares the op pure (it reads q/k/v, returns a fresh
    # clone; the shared Q/K/V/O device state is hidden behind the registry key, not aliased).
    @custom_op(
        f"{AXIOM_OP_NAMESPACE}::{FLASH_ATTENTION_OP_NAME}",
        mutates_args=(),
        schema="(Tensor q, Tensor k, Tensor v, int seq_len, int head_dim) -> Tensor",
    )
    def flash_attention(q, k, v, seq_len, head_dim):
        return _flash_attention_impl(q, k, v, int(seq_len), int(head_dim))

    @register_fake(f"{AXIOM_OP_NAMESPACE}::{FLASH_ATTENTION_OP_NAME}")
    def _flash_attention_fake(q, k, v, seq_len, head_dim):
        # Shape-only meta for torch.compile tracing. PURE — never touches the registry
        # (FakeTensorMode has no real device memory). Provably == the real clone after R1.
        return q.new_empty((int(seq_len), int(head_dim)), dtype=torch.float32)

    _REGISTERED = True


def _flash_attention_impl(q, k, v, seq_len: int, head_dim: int):
    """The real CUDA impl (R1 validate + R3 on-S execute + R2 bounded barrier/recover).

    The default kernel is the coopmat leader (head_dim==64). A non-default kernel is selectable
    only via :func:`flash_attention_session` pre-building; the registered op uses 'coopmat'.
    """
    import torch

    kernel = "coopmat"
    caller_stream = torch.cuda.current_stream()
    with _FA_REGISTRY_LOCK:  # held across the WHOLE op body (refinement)
        # R1 — FAIL-CLOSED validate q/k/v vs (seq_len, head_dim) BEFORE dispatch.
        for name, t in (("q", q), ("k", k), ("v", v)):
            if not isinstance(t, torch.Tensor):
                raise AxiomError(f"flash_attention op: {name} must be a torch.Tensor")
            if not t.is_cuda:
                raise AxiomError(
                    f"flash_attention op: {name} must be a CUDA tensor (no host round-trip)"
                )
            if t.dtype != torch.float32:
                raise AxiomError(
                    f"flash_attention op: {name} must be float32; got {t.dtype}"
                )
            if t.dim() != 2 or tuple(t.shape) != (seq_len, head_dim):
                raise AxiomError(
                    f"flash_attention op: {name} must be 2-D [seq_len, head_dim] = "
                    f"[{seq_len}, {head_dim}]; got shape {tuple(t.shape)}"
                )
        if head_dim != 64:
            raise AxiomError(
                f"flash_attention op: head_dim must be 64 (the coopmat-leader CORE pin); got "
                f"{head_dim}"
            )
        if seq_len <= 0 or seq_len % 16 != 0:
            raise AxiomError(
                f"flash_attention op: seq_len must be a positive multiple of 16 (the 16-key "
                f"block); got {seq_len}"
            )

        device_index = _device_index_of(q)
        key = _fa_key(device_index, seq_len, head_dim, kernel)
        op = _get_or_build(key)  # may rebuild a poisoned session (R2)
        session = op.session
        s = session.stream

        # R3 — make S see the caller's prior writes to q, k AND v (single ev_in covers all three
        # ONLY under the same-caller-stream precondition documented in the module docstring).
        cross_stream = caller_stream.cuda_stream != s.cuda_stream
        if cross_stream:
            ev_in = torch.cuda.Event()
            ev_in.record(caller_stream)
            s.wait_event(ev_in)

        # R3 — execute the handshake + (device) copy + clone on S; bounded barrier (R2);
        # clone strictly AFTER wait_completion so it captures the kernel's O write.
        try:
            with torch.cuda.stream(s):
                ov = op.attention(q, k, v)        # signal V1 → dispatch → wait V2 on S
                session.wait_completion()          # bounded owned-output barrier (R2)
                out = ov.clone()                   # clone on S, ordered after wait_completion
        except Exception as run_exc:
            # R2 — release the dangling CUDA wait, poison + evict so the faulted session/stream
            # is never reused, then re-raise. _recover_faulted_session is REUSED VERBATIM; it
            # pops the session from `entry.sessions[n]` — we adapt via a tiny shim entry.
            _recover_fa_session(key, op, session)
            raise AxiomError(
                f"flash_attention op: dispatch failed and the session was poisoned (it will "
                f"be rebuilt on the next call). Underlying: {run_exc}"
            ) from run_exc

        # R3 — make the caller's stream see `out` only after the clone.
        if cross_stream:
            ev_out = torch.cuda.Event()
            ev_out.record(s)
            caller_stream.wait_event(ev_out)

    return out


class _FaRecoveryEntry:
    """A tiny adapter so the VERBATIM-reused :func:`axiom_compute.ops._recover_faulted_session`
    (which evicts ``entry.sessions.pop(n, None)``) tears the FA session out of ``_FA_REGISTRY``.
    Caller holds ``_FA_REGISTRY_LOCK``."""

    def __init__(self, key: tuple):
        self._key = key
        self.sessions = self  # `entry.sessions.pop(n, None)` → this.pop(key, None)

    def pop(self, _n, _default=None):
        return _FA_REGISTRY.pop(self._key, _default)


def _recover_fa_session(key: tuple, op, session) -> None:
    """R2 + PCR-1: reuse the q4km ``_recover_faulted_session`` (device_wait_idle drain → only-if-
    needed host-signal → poison + evict) VERBATIM, adapting its ``entry.sessions.pop(n)`` eviction
    to ``_FA_REGISTRY.pop(key)`` via :class:`_FaRecoveryEntry`. Caller holds ``_FA_REGISTRY_LOCK``."""
    _recover_faulted_session(_FaRecoveryEntry(key), key, op, session)


@atexit.register
def _release_all_attention_sessions() -> None:
    """Best-effort teardown of every cached FA session at interpreter shutdown (swallows errors —
    torch's CUDA context may already be torn down). The PRIMARY teardown is explicit
    :func:`release_attention_session`."""
    try:
        with _FA_REGISTRY_LOCK:
            ops = list(_FA_REGISTRY.values())
            _FA_REGISTRY.clear()
    except Exception:
        return
    for op in ops:
        try:
            op.close()
        except Exception:
            pass


__all__ = [
    "AXIOM_OP_NAMESPACE",
    "FLASH_ATTENTION_OP_NAME",
    "register_flash_attention_op",
    "flash_attention_session",
    "flash_attention_views",
    "last_attn_dispatch_path",
    "last_attn_dispatch_staging",
    "release_attention_session",
]
