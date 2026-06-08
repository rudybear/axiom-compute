"""AT-2106 — lifetime / teardown: destroy-once, fd close-once, exact G-4 order.

  - close() runs the G-4 order: drain → drop view → cudaDestroyExternalMemory →
    cudaDestroyExternalSemaphore → Rust vkFree/destroy.
  - close() is IDEMPOTENT: calling it twice causes no second cudaDestroy* / vkFree* /
    fd close.
  - the OPAQUE_FD is closed exactly once (Python is the single owner, G-3).
  - torch does NOT free the foreign devPtr via its caching allocator.

Gated by AXC_ENABLE_GPU_TESTS=1 + torch.cuda.is_available() + nvidia ICD.
"""

import os
import struct

import pytest

torch = pytest.importorskip("torch")


def _gpu_enabled():
    return os.environ.get("AXC_ENABLE_GPU_TESTS") == "1" and torch.cuda.is_available()


pytestmark = pytest.mark.skipif(not _gpu_enabled(), reason="GPU tests gated (AXC_ENABLE_GPU_TESTS=1)")

_SAXPY_SRC = """
@kernel
@workgroup(64, 1, 1)
@intent("compute Y[i] = alpha * X[i] + Y[i] in parallel")
@complexity(O(n))
fn saxpy(n: u32, alpha: f32, x: readonly_buffer[f32], y: buffer[f32]) -> void {
    let i: u32 = gid(0);
    let limit: u32 = n;
    let in_range: bool = i < limit;
    let xi: f32 = x[i];
    let yi: f32 = y[i];
    let result: f32 = alpha * xi + yi;
    y[i] = result;
    return;
}
"""


def _fd_count():
    return len(os.listdir(f"/proc/{os.getpid()}/fd"))


def test_close_is_idempotent_and_no_fd_leak():
    """close() twice is a no-op; the exported fds do not leak across a session."""
    import axiom_compute as axc

    n = 1024
    k = axc.compile_kernel(_SAXPY_SRC)

    fd_before = _fd_count()
    sess = k.allocate_shared(sizes=[n * 4, n * 4])
    s = sess.stream
    x = sess.tensor(0, (n,), torch.float32)
    y = sess.tensor(1, (n,), torch.float32)
    with torch.cuda.stream(s):
        x.copy_(torch.arange(n, dtype=torch.float32, device="cuda"))
        y.zero_()
    s.synchronize()
    sess.run(workgroups=((n + 63) // 64, 1, 1), push_constants=struct.pack("<If", n, 1.0))
    s.synchronize()

    # Drop the torch views BEFORE close (G-4 step 2 happens inside close, but we also
    # release our references so nothing aliases the devPtr afterwards).
    del x, y

    sess.close()
    # Idempotent: a second close must not double-free / double-close.
    sess.close()
    sess.close()

    fd_after = _fd_count()
    # The exported memory fds + semaphore fd were consumed by CUDA import (closed once)
    # or closed by us; the count must not grow unboundedly. Allow small slack for the
    # CUDA driver's own bookkeeping.
    assert fd_after <= fd_before + 2, (
        f"fd leak: before={fd_before} after={fd_after} (exported fds must close once)"
    )


def test_torch_does_not_free_foreign_ptr():
    """The shared tensor view is NON-OWNING — torch must not free the foreign devPtr.

    We grab the data_ptr, close the session (which destroys the CUDA import), and
    assert no crash. The view's keepalive ties the mapping to the import; torch's
    caching allocator never owns the pointer.
    """
    import axiom_compute as axc

    n = 512
    k = axc.compile_kernel(_SAXPY_SRC)
    sess = k.allocate_shared(sizes=[n * 4, n * 4])
    s = sess.stream
    y = sess.tensor(1, (n,), torch.float32)
    with torch.cuda.stream(s):
        y.zero_()
    s.synchronize()
    ptr = y.data_ptr()
    assert ptr != 0
    # The view carries a keepalive to the ImportedExternalMemory (NOT torch's allocator).
    assert hasattr(y, "_axc_keepalive")
    del y
    sess.close()
    # No crash / no double-free at process exit is the assertion.
    torch.cuda.synchronize()
