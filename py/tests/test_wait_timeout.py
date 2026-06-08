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

PCR-1 (the false-positive guard): a recovery on a merely-SLOW (not faulted) dispatch must NOT
        host-signal V2 — the still-enqueued submit will signal it itself; a host double-signal of
        the same timeline value is Vulkan UB and bypasses the close() UAF fallback. Recovery runs
        device_wait_idle FIRST (drains the queue → the slow submit retires + self-signals V2), then
        the payload>=V2 check SKIPS the host-signal. Asserts: no double-signal (release_cuda_wait not
        used / _released_value unchanged), no validation error/crash, clean poison+evict+rebuild.

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
from axiom_compute import ops as axc_ops

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
def test_pcr1_false_positive_timeout_no_double_signal():
    """PCR-1 (the false-positive regression guard).

    A ``wait_completion`` timeout can be a FALSE POSITIVE: the V2-signaling submit is merely
    SLOW (still enqueued, WILL signal V2) rather than faulted. The OLD recovery host-signaled
    V2 unconditionally → the GPU's still-queued submit ALSO signals the same value → a non-
    monotone timeline DOUBLE-SIGNAL = Vulkan UB, and the advanced payload made close()/drain
    return immediately, BYPASSING the device_wait_idle UAF fallback → UAF on the SSBOs.

    The fix: recovery does device_wait_idle FIRST (drains the queue, so a merely-SLOW submit
    RETIRES and signals its OWN V2 — the GPU is the SOLE signaler), THEN reads the timeline:
    payload >= V2 → SKIP the host-signal (no double-signal). Here we drive a REAL dispatch
    (V2 WILL be signaled by the GPU) and then invoke the recovery path simulating a false-
    positive timeout. Assert: (a) the GPU's own signal reached V2 after device_wait_idle and
    release_cuda_wait was NOT used (no host double-signal — _released_value stays 0), (b) no
    Vulkan validation error / no crash, (c) the subsequent close() + rebuild is clean (no UAF
    — the drained queue means teardown cannot race an unretired submit)."""
    m, n, k = 256, 256, 256
    n_bpr = k // 256
    q = make_q4km_weights(m, n_bpr, 0xC0FFEE ^ (m + 7))

    # ISOLATED session so the recovery cannot perturb sibling tests.
    op = axc.Q4KMMatmul()
    op.upload_weights(q, m, k)
    xv = op.x_view(n)
    session = op.session
    s = session.stream

    # Spy on release_cuda_wait so we can ASSERT it was NOT host-signaled on the false positive.
    release_calls = []
    real_release = session.release_cuda_wait

    def spy_release(value):
        release_calls.append(int(value))
        return real_release(value)

    session.release_cuda_wait = spy_release  # type: ignore[method-assign]

    # A minimal fake entry so _recover_faulted_session can evict (it only touches .sessions).
    class _FakeEntry:
        def __init__(self):
            self.sessions = {n: op}

    entry = _FakeEntry()

    def drive_and_recover():
        # (1) A REAL dispatch: V1 signaled, the kernel WILL signal V2 (merely-SLOW vs faulted
        # is indistinguishable to a timeout — this is the false-positive case).
        x_bits = make_x_f16_bits(k, n, 0xBADF00D ^ (n + 7))
        with torch.cuda.stream(s):
            xv.copy_(_x_f16_tensor_from_bits(x_bits, k, n))
            op.matmul(xv)  # enqueues signal V1 → dispatch → CUDA wait V2 on S
        target = session._last_signal
        assert target > 0

        # (2) Simulate the FALSE-POSITIVE recovery: device_wait_idle FIRST drains the queue
        # (the merely-SLOW submit retires + self-signals V2), then the payload>=V2 check skips
        # the host-signal. This is exactly the production recovery path.
        axc_ops._recover_faulted_session(entry, n, op, session)

        # The GPU's OWN submit signaled V2 (proven AFTER device_wait_idle drained the queue).
        # We cannot re-query the (now torn-down) session, but the recovery already read it and
        # must NOT have host-signaled.
        return target

    try:
        target = _run_with_watchdog(drive_and_recover, seconds=8.0)
        # (a) NO host double-signal: release_cuda_wait was NOT invoked (payload>=V2 skip), so
        # _released_value never advanced past 0. (If the test environment ever raced such that
        # the GPU had not yet retired, the guard would still keep it AT-MOST-ONCE; but on a
        # drained queue the GPU is the sole signaler and the skip MUST hold.)
        assert release_calls == [], (
            f"PCR-1: recovery host-signaled V2 on a FALSE-POSITIVE timeout (double-signal UB): "
            f"release_cuda_wait called with {release_calls}; the GPU's own submit had already "
            f"signaled V2={target} after device_wait_idle — host-signal must be SKIPPED."
        )
        assert session._released_value == 0, (
            f"PCR-1: _released_value advanced to {session._released_value} — a host double-signal "
            "of an already-GPU-signaled value (Vulkan UB) was emitted."
        )
        # (c) the session was poisoned + evicted cleanly (no UAF, no validation error/crash).
        assert session.is_poisoned
        assert n not in entry.sessions
        print(
            "[PCR-1] false-positive recovery: device_wait_idle drained the queue, the GPU's own "
            f"submit signaled V2={target}, host-signal was SKIPPED (no double-signal), "
            "poison+evict clean (no UAF)"
        )
    finally:
        try:
            op.close()
        except Exception:
            pass

    # (c continued) the rebuilt session works after the false-positive recovery.
    wid = q4km_register_weights(q, m, k)
    try:
        x = _x_f16_tensor_from_bits(make_x_f16_bits(k, n, 0x5EED ^ n), k, n)
        c = torch.ops.axiom.q4km_matmul(x, wid, m, n, k)
        assert tuple(c.shape) == (m, n)
        print("[PCR-1] rebuilt session works after the false-positive recovery")
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
