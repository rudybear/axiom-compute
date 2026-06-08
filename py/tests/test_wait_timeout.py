"""M4.1 Phase 3 — the wait-timeout residual fix (R2), bounded on BOTH sides.

AT-2114 (the key test): in an ISOLATED session, ENQUEUE the real CUDA wait(V2) on stream S
        but never GPU-signal V2 (simulate a faulted dispatch). Assert:
          (a) wait_completion(timeout_ms=small) RAISES within the bound (not a hang),
          (b) the host-signal-to-V2 (release_cuda_wait) releases the dangling CUDA wait so a
              SUBSEQUENT CUDA op on stream S does NOT hang (S recovered, not just the host bound),
          (c) the rebuilt (fresh) session works.
        A hard wall-clock watchdog (<~5 s) so a TRUE hang FAILS the suite (not hangs it).
        WATCH-ITEM 2: also confirms host vkSignalSemaphore on a faulted-GPU timeline works on
        the 580.x driver — if it doesn't, this fails LOUDLY.

AT-2115: the bare SharedSession.run / op happy path has no new host sync and stays bit-exact
         over >=100 calls; wait_completion on a completed session returns immediately; no poison
         fires on a healthy session.

Gated by AXC_ENABLE_GPU_TESTS=1 + torch.cuda.is_available() + an nvidia ICD.
"""

import os
import threading

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import axiom_compute as axc
from axiom_compute import (
    AxiomError,
    q4km_register_weights,
    q4km_release_weights,
    q4km_activation_view,
)

from test_zero_copy_q4km import (
    make_q4km_weights,
    make_x_f16_bits,
    _x_f16_tensor_from_bits,
)


def _gpu_enabled():
    return os.environ.get("AXC_ENABLE_GPU_TESTS") == "1" and torch.cuda.is_available()


def _run_with_watchdog(fn, seconds: float):
    """Run ``fn()`` on a daemon thread; FAIL if it does not return within ``seconds``.

    A TRUE deadlock (the CUDA wait never released) would otherwise hang the whole suite —
    the watchdog turns it into a loud assertion failure instead.
    """
    result = {}

    def target():
        try:
            result["value"] = fn()
        except BaseException as e:  # noqa: BLE001 — capture to re-raise on the main thread
            result["error"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        pytest.fail(
            f"watchdog: operation did not complete within {seconds:.1f}s — a CUDA wait was "
            "left enqueued (the both-sides deadlock fix did NOT release stream S). "
            "If this is the host vkSignalSemaphore-on-a-faulted-timeline path, it failed on "
            "the 580.x driver (WATCH-ITEM 2)."
        )
    if "error" in result:
        raise result["error"]
    return result.get("value")


@pytest.mark.skipif(not _gpu_enabled(), reason="GPU tests gated (AXC_ENABLE_GPU_TESTS=1)")
def test_wait_completion_releases_cuda_wait_and_poisons():
    """AT-2114 (R2): the injected one-sided stall is bounded AND stream S is recovered."""
    m, n, k = 256, 256, 256
    n_bpr = k // 256
    q = make_q4km_weights(m, n_bpr, 0xC0FFEE ^ m)

    # ISOLATED session (its own Q4KMMatmul/SharedSession/stream) so the wedge cannot affect
    # sibling tests.
    op = axc.Q4KMMatmul()
    op.upload_weights(q, m, k)
    op.x_view(n)  # materialize the session
    session = op.session
    s = session.stream

    def inject_and_recover():
        # Reserve a fresh (V1, V2) and ENQUEUE the REAL CUDA wait(V2) on stream S, but never
        # GPU-signal V2 — exactly the GPU-fault-mid-dispatch case (V1 may signal, V2 never).
        v1, v2 = op.kernel._next_values()
        # Park the CUDA stream on a timeline value the GPU will NEVER reach (the faulted case).
        session._last_signal = v2
        session._sem.wait(v2, s)  # cudaWaitExternalSemaphoresAsync(V2) enqueued on S

        # (a) wait_completion must RAISE within the small bound (not a hang).
        raised = False
        try:
            session.wait_completion(timeout_ms=1500)
        except AxiomError as e:
            raised = True
            print(f"[AT-2114] wait_completion bounded-raised: {e}")
        assert raised, "wait_completion did NOT raise on the never-signaled V2 (it hung/returned)"

        # (b) RELEASE the dangling CUDA wait (host-signal the Vulkan timeline to V2) so S
        # unblocks; then a subsequent CUDA op on S must NOT hang.
        session.release_cuda_wait(v2)
        session.poison()
        # A subsequent CUDA op on the SAME stream S — would deadlock if S were still parked.
        with torch.cuda.stream(s):
            probe = torch.ones(16, device="cuda") * 3.0
            probe = probe + 1.0
        s.synchronize()  # this is the actual "does S unblock?" assertion
        assert float(probe[0].item()) == 4.0
        print("[AT-2114] release_cuda_wait(V2) unblocked stream S; subsequent op completed")
        return True

    try:
        # Hard wall-clock watchdog: a true hang fails fast instead of wedging CI.
        _run_with_watchdog(inject_and_recover, seconds=5.0)
        assert session.is_poisoned
    finally:
        try:
            op.close()
        except Exception:
            pass

    # (c) the rebuilt session works (a fresh weights_id → a fresh session).
    wid = q4km_register_weights(q, m, k)
    try:
        x = _x_f16_tensor_from_bits(make_x_f16_bits(k, n, 0xBADF00D ^ n), k, n)
        c = torch.ops.axiom.q4km_matmul(x, wid, m, n, k)
        assert tuple(c.shape) == (m, n)
        print("[AT-2114] rebuilt session works after the fault")
    finally:
        q4km_release_weights(wid)


@pytest.mark.skipif(not _gpu_enabled(), reason="GPU tests gated (AXC_ENABLE_GPU_TESTS=1)")
def test_wait_timeout_happy_path_unchanged():
    """AT-2115: the happy path is unchanged — bare run() stays bit-exact over >=100 calls;
    wait_completion on a completed session returns immediately; no poison fires."""
    m, n, k = 256, 256, 256
    n_bpr = k // 256
    q = make_q4km_weights(m, n_bpr, 0xC0FFEE ^ m)
    op = axc.Q4KMMatmul()
    op.upload_weights(q, m, k)
    try:
        s = op.kernel.stream
        xv = op.x_view(n)
        x_bits = make_x_f16_bits(k, n, 0xBADF00D ^ n)
        with torch.cuda.stream(s):
            xv.copy_(_x_f16_tensor_from_bits(x_bits, k, n))
        s.synchronize()

        op.matmul(xv)
        s.synchronize()
        first = op.c_view(n).detach().clone()
        sess = op.session

        # wait_completion on a completed session returns immediately (timeline already >= V2).
        sess.wait_completion(timeout_ms=2000)
        assert not sess.is_poisoned

        for i in range(100):
            op.matmul(xv)
            s.synchronize()
            cur = op.c_view(n)
            assert (cur - first).abs().max().item() == 0.0, f"call {i} not bit-identical"
        # wait_completion still returns immediately, no poison on the healthy session.
        sess.wait_completion()
        assert not sess.is_poisoned
        print("[AT-2115] happy path unchanged: 100 calls bit-identical, no poison, fast drain")
    finally:
        op.close()
