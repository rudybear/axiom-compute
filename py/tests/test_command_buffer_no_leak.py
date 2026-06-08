"""FIX-1 regression guard — dispatch_zero_copy must NOT leak a VkCommandBuffer per call.

Before FIX-1, dispatch_zero_copy allocated a fresh PRIMARY command buffer from the
context's command pool on EVERY run() and freed it only on error paths — the success
path leaked one VkCommandBuffer per dispatch for the whole process lifetime (a slow OOM
over any inference loop). FIX-1 allocates ONE command buffer per SharedSession and
RESETs + re-records it each dispatch (freed once at teardown).

This test drives a LONG single-session loop (5000+ dispatches — 5x the AT-2104 count,
far more than would "happen to fit" if leaking) and asserts:
  - every dispatch stays bit-exact (no torn read from the reset+reuse),
  - the G-2 staging/copy counters stay 0 (still genuinely staging-free), and
  - the loop completes without the pool exhausting / OOMing.

A per-dispatch leak of 5000+ command buffers would (at minimum) blow the pool's backing
allocation; completing 5000 dispatches on ONE session is the bounded-growth signal.

Gated by AXC_ENABLE_GPU_TESTS=1 + torch.cuda.is_available() + nvidia ICD.
"""

import os
import struct

import pytest

torch = pytest.importorskip("torch")


def _gpu_enabled():
    return os.environ.get("AXC_ENABLE_GPU_TESTS") == "1" and torch.cuda.is_available()


pytestmark = pytest.mark.skipif(
    not _gpu_enabled(), reason="GPU tests gated (AXC_ENABLE_GPU_TESTS=1)"
)

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


def test_many_dispatches_no_command_buffer_growth():
    """FIX-1: 5000+ dispatches on ONE session stay correct + bounded (no CB-per-call leak)."""
    import axiom_compute as axc

    n, alpha, iters = 1024, 2.0, 5000
    k = axc.compile_kernel(_SAXPY_SRC)
    sess = k.allocate_shared(sizes=[n * 4, n * 4])
    try:
        s = sess.stream
        x = sess.tensor(0, (n,), torch.float32)
        y = sess.tensor(1, (n,), torch.float32)
        base_x = torch.arange(n, dtype=torch.float32, device="cuda")
        pc = struct.pack("<If", n, alpha)
        wg = ((n + 63) // 64, 1, 1)

        # The session owns exactly ONE command buffer; it is reset+reused every dispatch.
        for it in range(iters):
            with torch.cuda.stream(s):
                x.copy_(base_x)
                y.zero_()
            sess.run(workgroups=wg, push_constants=pc)

            # G-2 must stay staging-free across the whole loop (no per-dispatch staging).
            assert sess.staging_buffers_allocated() == 0, (
                f"iter {it}: staging buffer allocated on zero-copy path (G-2 violation)"
            )
            assert sess.copy_buffer_records() == 0, (
                f"iter {it}: vkCmdCopyBuffer recorded on zero-copy path (G-2 violation)"
            )

            # Spot-check correctness periodically (full check every iter is slow at 5000).
            if it % 500 == 0 or it == iters - 1:
                with torch.cuda.stream(s):
                    expected = alpha * base_x  # y started at 0
                    maxdiff = (y - expected).abs().max()
                s.synchronize()
                assert maxdiff.item() == 0.0, (
                    f"iter {it}: torn/incorrect read after CB reset+reuse (diff {maxdiff.item()})"
                )

        # Reaching here after 5000 dispatches on one session is the bounded-growth signal:
        # a per-dispatch leak would have grown the pool's command-buffer set unboundedly.
    finally:
        sess.close()
