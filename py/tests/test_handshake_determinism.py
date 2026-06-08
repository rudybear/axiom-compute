"""AT-2104 — cross-API timeline handshake is race-free over 1000+ iterations.

Back-to-back torch-write → CUDA-signal(V1) → Vulkan-compute(wait V1 / signal V2) →
CUDA-wait(V2) → torch-read, with a FRESH monotone (V1, V2) per iteration (G-7), all
on the captured stream S (G-5). EVERY iteration must be bit-exact; a torn/raced read
would show non-deterministic diffs. This is the cross-API analogue of the M3.x
barrier-race anchors and the #1 correctness risk (R-1).

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


def test_timeline_handshake_no_race():
    """AT-2104: 1000 back-to-back zero-copy iterations are ALL bit-exact + distinct triples."""
    import axiom_compute as axc

    n, alpha, iters = 2048, 1.5, 1000
    k = axc.compile_kernel(_SAXPY_SRC)
    sess = k.allocate_shared(sizes=[n * 4, n * 4])
    seen_values = set()
    try:
        s = sess.stream
        x = sess.tensor(0, (n,), torch.float32)
        y = sess.tensor(1, (n,), torch.float32)
        base_x = torch.arange(n, dtype=torch.float32, device="cuda")
        pc = struct.pack("<If", n, alpha)
        wg = ((n + 63) // 64, 1, 1)

        for it in range(iters):
            with torch.cuda.stream(s):
                x.copy_(base_x + float(it))
                y.zero_()
                xr = x.clone()
            v2 = sess.run(workgroups=wg, push_constants=pc)
            # G-7: every (V1,V2) triple is distinct.
            assert v2 not in seen_values, f"timeline value {v2} reused (G-7 violation)"
            seen_values.add(v2)
            with torch.cuda.stream(s):
                expected = alpha * xr  # y was 0
                maxdiff = (y - expected).abs().max()
            s.synchronize()
            assert maxdiff.item() == 0.0, f"iter {it}: torn/raced read (diff {maxdiff.item()})"

        assert len(seen_values) == iters, "G-7: 1000 iterations → 1000 distinct triples"
    finally:
        sess.close()
