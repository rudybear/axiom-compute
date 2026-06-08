"""M4.1 Phase 3 — the Q4_K_M zero-copy matmul as a FIRST-CLASS PyTorch operator.

Registers ``torch.ops.axiom.q4km_matmul(Tensor x, int weights_id, int M, int N, int K)
-> Tensor`` via :func:`torch.library.custom_op` (torch 2.12) so the merged Phase-2
zero-copy Q4_K_M kernel (:class:`axiom_compute.q4km.Q4KMMatmul`) (a) is a registered op
with a schema, (b) composes under ``torch.compile`` via :func:`register_fake`, and (c)
runs the existing zero-copy dispatch on CUDA tensors with NO change to the codegen /
handshake / FROZEN 1e-3 oracle.

The op caches a :class:`Q4KMMatmul`/``SharedSession`` keyed by ``(weights_id, N)`` so
torch's stateless-op view does NOT re-export/re-import the shared buffers per call (that
would destroy the zero-copy amortization). The static quantized weights are uploaded ONCE
into the shared q buffer and referenced by an integer ``weights_id`` handle — the
inference-time analogue of a bound nn.Parameter referenced by id, and the only design that
keeps the op a clean compiled-graph LEAF (an int traces as a constant).

r2 blocker fixes (see ``.pipeline/runs/M4.1p3-architect.json`` §0):
  R1 — the real CUDA impl FAIL-CLOSED-VALIDATES caller (M,N,K) vs the resolved weights
       record + x.shape (M==entry.m, K==entry.k==x.shape[0], N==x.shape[1]) BEFORE any
       dispatch, so register_fake's ``x.new_empty((M,N), f32)`` is PROVABLY identical
       (shape, dtype, device) to the real clone — a mismatch RAISES (no torch.compile
       miscompile).
  R2 — on a wait_completion timeout / dispatch error AFTER V1 signaled the op RELEASES the
       dangling CUDA wait(V2) (host-signal the Vulkan timeline to V2) so stream S unblocks,
       then POISONS + evicts the session so it is rebuilt next call (faulted stream never
       reused), then raises. The timeout bounds BOTH the host AND the CUDA side.
  R3 — the op EXECUTES its handshake + device copy + ``cv.clone()`` inside
       ``with torch.cuda.stream(session.stream)`` with cross-stream events (ev_in caller→S,
       ev_out S→caller) so it is correct regardless of torch's active stream under
       torch.compile — the clone runs on S strictly AFTER wait_completion. It returns the
       CLONE (never the persistent shared C view).

Honest framing (R-6): NO beat-cuBLAS (~53% of f32 cuBLAS, Phase 2 measured). The value is
REAL torch integration + ``torch.compile`` composition. The zero-copy fast path is
EAGER-scoped (under fullgraph compile Inductor functionalization may break x.data_ptr()
identity at the op boundary → the op correctly falls to the device_copy path).
"""

from __future__ import annotations

import atexit
import itertools
import threading
from typing import Optional

from . import AxiomError
from .q4km import Q4KMMatmul

AXIOM_OP_NAMESPACE: str = "axiom"
Q4KM_OP_NAME: str = "q4km_matmul"

# ── The (weights_id, N)-keyed process-local registry ─────────────────────────────────
# `_REGISTRY_LOCK` is held across the WHOLE op body (lookup → R1-validate → ensure_session
# → dispatch → wait_completion → clone → cross-stream hand-back) so a concurrent
# q4km_release_weights cannot close a session mid-dispatch, AND so no two op calls on the
# same (weights_id, N) session are in-flight concurrently (the mutates_args=() soundness
# invariant). q4km_release_weights takes the same lock.
_REGISTRY_LOCK = threading.Lock()
# Never-reused monotone u63 weights_id (UAF-safe): a released id fails closed.
_WEIGHTS_ID_COUNTER = itertools.count(1)


class _WeightsEntry:
    """The per-weights_id registry record: the static (M,K) + a {N -> Q4KMMatmul} map.

    The FIRST Q4KMMatmul owns the compiled kernel + the q bytes; a new N constructs an
    additional Q4KMMatmul re-using the SAME q bytes (a second upload-once into a new
    session) — keeping q4km.py UNCHANGED (single-N-per-instance) and honest (a new N is a
    disclosed amortized new-session + q upload, NOT a per-call cost).
    """

    def __init__(self, q_bytes: bytes, m: int, k: int, device, strategy):
        self.q_bytes = q_bytes
        self.m = int(m)
        self.k = int(k)
        self.device = device
        self.strategy = strategy
        self.sessions: dict[int, Q4KMMatmul] = {}  # N -> Q4KMMatmul
        self.last_path: Optional[str] = None  # the most recent dispatch path

    def ensure_session(self, n: int) -> Q4KMMatmul:
        """Reuse the (cached, non-poisoned) Q4KMMatmul for N, or build a fresh one.

        R2: a POISONED session (a prior fault) is torn down + rebuilt so a faulted
        session + its stream S are never reused for a new dispatch.
        """
        op = self.sessions.get(n)
        if op is not None and op.session is not None and op.session.is_poisoned:
            # Faulted session — evict + tear down (best-effort) before rebuilding.
            try:
                op.close()
            except Exception:
                pass
            op = None
            del self.sessions[n]
        if op is None:
            op = Q4KMMatmul(device=self.device, strategy=self.strategy)
            op.upload_weights(self.q_bytes, self.m, self.k)
            # Materialize the (weights_id, N) session now (upload-once) so subsequent calls
            # reuse it (AT-2113b amortization) and so x_view()/c_view() are available.
            op.x_view(n)
            self.sessions[n] = op
        return op

    def close(self) -> None:
        for op in list(self.sessions.values()):
            try:
                op.close()
            except Exception:
                pass
        self.sessions.clear()


_WEIGHTS_REGISTRY: dict[int, _WeightsEntry] = {}


def _registry_lookup(weights_id: int) -> _WeightsEntry:
    entry = _WEIGHTS_REGISTRY.get(int(weights_id))
    if entry is None:
        raise AxiomError(
            f"q4km op: weights_id {weights_id} is not registered (or was released). "
            "weights_id is a never-reused monotone handle — register weights with "
            "q4km_register_weights(...) and keep the returned id."
        )
    return entry


def q4km_register_weights(
    packed_weights, m: int, k: int, *, device=None, strategy: Optional[dict] = None
) -> int:
    """Upload the static Q4_K_M weight bytes ONCE and return a stable u63 ``weights_id``.

    ``packed_weights`` is the ggml Q4_K_M bytes — a 1-D ``uint8`` CUDA-or-CPU torch tensor,
    a numpy array, or raw ``bytes``. The entry RECORDS ``(m, k)`` so the op can fail-closed
    -validate caller ``M==entry.m`` and ``K==entry.k`` (R1). The returned id is the torch-op
    handle for those weights (the inference analogue of a bound nn.Parameter referenced by
    id); it is NEVER reused (a released id fails closed).
    """
    q_bytes = _as_q_bytes(packed_weights)
    entry = _WeightsEntry(q_bytes, m, k, device, strategy)
    with _REGISTRY_LOCK:
        wid = next(_WEIGHTS_ID_COUNTER)
        _WEIGHTS_REGISTRY[wid] = entry
    return wid


def q4km_release_weights(weights_id: int) -> None:
    """G-4 teardown: drain → cudaDestroy → vkFree every session for ``weights_id`` + drop
    the entry. Idempotent. Takes the registry lock (so it cannot close a session mid-
    dispatch — it blocks until any in-flight op call completes). ``weights_id`` is never
    reused, so a released id fails closed on a later op call (never freed-then-reused)."""
    with _REGISTRY_LOCK:
        entry = _WEIGHTS_REGISTRY.pop(int(weights_id), None)
    if entry is not None:
        entry.close()


def q4km_activation_view(weights_id: int, n: int):
    """The zero-copy x buffer (a thin pass-through to ``Q4KMMatmul.x_view``).

    Write activations directly into this ``[K, N] f16`` view then call the op with THAT
    tensor as ``x`` for the EAGER zero-copy fast path (0/0 staging). Under torch.compile
    the zero-copy identity may be lost to functionalization → the op falls to device_copy
    (disclosed)."""
    with _REGISTRY_LOCK:
        entry = _registry_lookup(weights_id)
        op = entry.ensure_session(n)
        return op.x_view(n)


def last_dispatch_path(weights_id: int) -> str:
    """``'zero_copy'`` | ``'device_copy'`` — which path the most recent op call for
    ``weights_id`` took (honest path reporting, AT-2113)."""
    with _REGISTRY_LOCK:
        entry = _registry_lookup(weights_id)
        if entry.last_path is None:
            raise AxiomError(
                f"q4km op: no dispatch has run yet for weights_id {weights_id}"
            )
        return entry.last_path


def last_dispatch_staging(weights_id: int):
    """``(staging_buffers_allocated, copy_buffer_records)`` of the most recent dispatch —
    asserts 0/0 on the zero-copy path (AT-2112)."""
    with _REGISTRY_LOCK:
        entry = _registry_lookup(weights_id)
        # Any session shares the same staging counters (they are 0/0 on the zero-copy path
        # for every session); report the most recently-used N's session if available.
        for op in entry.sessions.values():
            sess = op.session
            if sess is not None:
                return (sess.staging_buffers_allocated(), sess.copy_buffer_records())
        raise AxiomError(
            f"q4km op: no session has been built yet for weights_id {weights_id}"
        )


def _as_q_bytes(packed_weights) -> bytes:
    """Accept a torch uint8 tensor, a numpy array, or raw bytes → bytes (upload-once)."""
    if isinstance(packed_weights, (bytes, bytearray, memoryview)):
        return bytes(packed_weights)
    # torch tensor?
    try:
        import torch

        if isinstance(packed_weights, torch.Tensor):
            t = packed_weights
            if t.dtype != torch.uint8:
                raise AxiomError(
                    f"q4km_register_weights: packed_weights must be uint8; got {t.dtype}"
                )
            return bytes(t.detach().cpu().contiguous().view(-1).numpy().tobytes())
    except ImportError:
        pass
    # numpy array (or anything with tobytes)?
    tobytes = getattr(packed_weights, "tobytes", None)
    if tobytes is not None:
        return bytes(tobytes())
    raise AxiomError(
        "q4km_register_weights: packed_weights must be a uint8 torch tensor, a numpy "
        f"array, or bytes; got {type(packed_weights)!r}"
    )


# ── The registered op (defined once at import, guarded for the no-torch surface) ──────
_REGISTERED = False


def register_q4km_op() -> None:
    """Idempotent: define ``torch.ops.axiom.q4km_matmul`` + register_fake + the CUDA impl.

    Guarded so importing the package WITHOUT torch (or with a torch lacking
    ``torch.library.custom_op``) still succeeds for the non-torch surface — the op only
    materializes when torch is present.
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

    # An EXPLICIT schema string (not inferred): `from __future__ import annotations` turns
    # the function annotations into strings, which torch's infer_schema rejects — so we pin
    # the schema directly. mutates_args=() declares the op pure (it reads x, returns a fresh
    # clone; the shared q/x/C device state is hidden behind weights_id, not aliased).
    @custom_op(
        f"{AXIOM_OP_NAMESPACE}::{Q4KM_OP_NAME}",
        mutates_args=(),
        schema="(Tensor x, int weights_id, int M, int N, int K) -> Tensor",
    )
    def q4km_matmul(x, weights_id, M, N, K):
        return _q4km_matmul_impl(x, int(weights_id), int(M), int(N), int(K))

    @register_fake(f"{AXIOM_OP_NAMESPACE}::{Q4KM_OP_NAME}")
    def _q4km_matmul_fake(x, weights_id, M, N, K):
        # Shape-only meta for torch.compile tracing. PURE — never touches the registry
        # (FakeTensorMode has no real device memory). Provably == the real clone after R1.
        return x.new_empty((int(M), int(N)), dtype=torch.float32)

    _REGISTERED = True


def _q4km_matmul_impl(x, weights_id: int, M: int, N: int, K: int):
    """The real CUDA impl (R1 validate + R3 on-S execute + R2 bounded barrier/recover)."""
    import torch

    caller_stream = torch.cuda.current_stream()
    with _REGISTRY_LOCK:  # held across the WHOLE op body (refinement)
        entry = _registry_lookup(weights_id)

        # R1 — FAIL-CLOSED validate (M,N,K) vs the weights record + x.shape BEFORE dispatch.
        if not (isinstance(x, torch.Tensor)):
            raise AxiomError("q4km op: x must be a torch.Tensor")
        if not x.is_cuda:
            raise AxiomError("q4km op: x must be a CUDA tensor (no host round-trip)")
        if x.dtype != torch.float16:
            raise AxiomError(f"q4km op: x must be float16; got {x.dtype}")
        if x.dim() != 2:
            raise AxiomError(
                f"q4km op: x must be 2-D [K, N] row-major; got shape {tuple(x.shape)}"
            )
        x_k, x_n = int(x.shape[0]), int(x.shape[1])
        if K != entry.k or K != x_k:
            raise AxiomError(
                f"q4km op: K mismatch — op K={K}, weights K={entry.k}, x.shape[0]={x_k}"
            )
        if M != entry.m:
            raise AxiomError(
                f"q4km op: M mismatch — op M={M}, weights M={entry.m}"
            )
        if N != x_n:
            raise AxiomError(
                f"q4km op: N mismatch — op N={N}, x.shape[1]={x_n}"
            )

        op = entry.ensure_session(N)  # may rebuild a poisoned session (R2)
        session = op.session
        s = session.stream

        # R3 — make S see the caller's prior writes to x.
        cross_stream = caller_stream.cuda_stream != s.cuda_stream
        if cross_stream:
            ev_in = torch.cuda.Event()
            ev_in.record(caller_stream)
            s.wait_event(ev_in)

        # R3 — execute the handshake + (device) copy + clone on S; bounded barrier (R2);
        # clone strictly AFTER wait_completion so it captures the kernel's C write.
        try:
            with torch.cuda.stream(s):
                cv = op.matmul(x)                 # signal V1 → dispatch → wait V2 on S
                session.wait_completion()         # bounded owned-output barrier (R2)
                out = cv.clone()                  # clone on S, ordered after wait_completion
        except Exception as run_exc:
            # R2 — release the dangling CUDA wait(V2) so stream S unblocks, then poison +
            # evict so the faulted session/stream is never reused, then re-raise.
            _recover_faulted_session(entry, N, op, session)
            raise AxiomError(
                f"q4km op: dispatch failed and the session was poisoned (it will be "
                f"rebuilt on the next call). Underlying: {run_exc}"
            ) from run_exc

        entry.last_path = op.report_path

        # R3 — make the caller's stream see `out` only after the clone.
        if cross_stream:
            ev_out = torch.cuda.Event()
            ev_out.record(s)
            caller_stream.wait_event(ev_out)

    return out


def _recover_faulted_session(entry, n, op, session) -> None:
    """R2 + PCR-1: drain the queue, conditionally release the dangling CUDA wait(V2), then
    poison + evict.

    PCR-1 (recovery-path double-signal / UAF fix). A ``wait_completion`` timeout cannot tell
    a REAL GPU fault (the V2-signaling submit faulted → V2 never signals) from a merely-SLOW
    dispatch (the submit is STILL ENQUEUED and WILL signal V2 shortly). Host-signaling V2
    unconditionally would, on a FALSE-POSITIVE timeout, double-signal the same timeline value
    (the GPU's still-queued submit ALSO signals it) — a non-monotone timeline double-signal =
    Vulkan UB — and would advance the payload so the later close()/drain returns immediately,
    BYPASSING the device_wait_idle UAF fallback while the submit still references the SSBOs.

    The fix (this path is the EXCEPTIONAL/error path; the extra barrier is acceptable here —
    the G-5 happy path is untouched):
      1. device_wait_idle FIRST — drains the queue. A merely-SLOW submit RETIRES and signals
         its OWN V2 (the GPU stays the SOLE signaler); a genuine fault returns with V2 still
         unsignaled.
      2. AFTER the drain, read the timeline payload. If it already reached V2 (false positive,
         or a real-but-now-retired completion) → do NOT host-signal: the GPU already released
         the CUDA wait(V2), and re-signaling the same value is the UB we must avoid. Only if
         payload < V2 (genuine fault — no enqueued submit will ever signal it, proven by the
         drain) do we host-signal V2 as the SOLE signaler to release the parked CUDA wait.
      3. THEN poison + evict. Because the queue was drained, the subsequent close()/teardown
         cannot race an unretired submit → no UAF.

    All steps are best-effort/guarded (this runs while already handling a fault). The host
    signal stays at-most-once / monotone (guarded inside SharedSession.release_cuda_wait)."""
    target = session._last_signal

    # (1) Drain the WHOLE queue FIRST so a merely-SLOW submit retires + self-signals V2, and a
    # genuine fault is left with V2 unsignaled. This also removes the UAF window before close.
    drained = False
    try:
        session.device_wait_idle()
        drained = True
    except Exception:
        # Could not drain (wedged device / binary fallback). Fall back to the prior R2
        # behavior: host-signal to release the parked wait (still at-most-once/monotone).
        pass

    # (2) Only host-signal V2 if the GPU did NOT already signal it (avoid the double-signal).
    # If we could not drain, we cannot prove the GPU won't signal — but an undrained device is
    # already the UAF-fallback's job at close(); here we still release the parked wait so the
    # stream unblocks (the monotone guard prevents re-signaling an already-signaled value).
    need_host_signal = True
    if drained:
        try:
            payload = session.external_timeline_value()
            # GPU already reached V2 (false positive / retired completion): SKIP host-signal.
            need_host_signal = payload < target
        except Exception:
            # Could not read the counter (e.g. binary fallback): fall back to host-signal,
            # guarded at-most-once/monotone inside release_cuda_wait.
            need_host_signal = True

    if need_host_signal:
        try:
            session.release_cuda_wait(target)
        except Exception:
            pass

    try:
        session.poison()
    except Exception:
        pass
    try:
        op.close()
    except Exception:
        pass
    entry.sessions.pop(n, None)


@atexit.register
def _release_all_weights() -> None:
    """Best-effort teardown of every registered weights entry at interpreter shutdown
    (swallows errors — torch's CUDA context may already be torn down). The PRIMARY teardown
    is explicit :func:`q4km_release_weights`."""
    try:
        with _REGISTRY_LOCK:
            entries = list(_WEIGHTS_REGISTRY.values())
            _WEIGHTS_REGISTRY.clear()
    except Exception:
        return
    for entry in entries:
        try:
            entry.close()
        except Exception:
            pass


__all__ = [
    "AXIOM_OP_NAMESPACE",
    "Q4KM_OP_NAME",
    "register_q4km_op",
    "q4km_register_weights",
    "q4km_release_weights",
    "q4km_activation_view",
    "last_dispatch_path",
    "last_dispatch_staging",
]
