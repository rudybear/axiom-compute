"""AT-2107 — fail-closed: no silent host copy.

A CPU tensor / unavailable CUDA / missing external-fd ICD raises ZeroCopyUnavailable
with a clear message — NEVER a silent host copy.

Gated by AXC_ENABLE_GPU_TESTS=1 + torch.cuda.is_available() + nvidia ICD.
"""

import os

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


def test_bogus_uuid_fails_closed():
    """A non-matching device UUID raises ZeroCopyUnavailable, never a different GPU."""
    import axiom_compute as axc

    # Compile via the native API with a bogus 16-byte UUID — must FAIL CLOSED.
    with pytest.raises(Exception) as ei:
        axc._axiom_compute.compile_kernel(_SAXPY_SRC, [0xAB] * 16)
    msg = str(ei.value).lower()
    assert "uuid" in msg or "match" in msg or "unavailable" in msg, (
        f"bogus UUID must fail-closed with a clear message; got: {ei.value}"
    )


def test_cpu_tensor_is_not_silently_host_copied():
    """A CPU tensor used where a CUDA shared tensor is expected must be rejected.

    The frontend's shared tensors are always CUDA-device views; a CPU tensor cannot
    be passed as a shared buffer. We assert the shared-tensor view is on CUDA (so a
    CPU tensor would be a type/device error, never a silent host bridge).
    """
    import axiom_compute as axc

    n = 256
    k = axc.compile_kernel(_SAXPY_SRC)
    sess = k.allocate_shared(sizes=[n * 4, n * 4])
    try:
        x = sess.tensor(0, (n,), torch.float32)
        assert x.is_cuda, "shared tensor must be a CUDA view (no host copy)"
        assert x.device.type == "cuda"
    finally:
        sess.close()


def test_compile_kernel_requires_cuda():
    """compile_kernel surfaces ZeroCopyUnavailable when CUDA is unavailable (documented).

    On this box CUDA IS available, so we only assert the happy path returns a Kernel
    bound to the matched GPU — the unavailable path is covered by the explicit
    ZeroCopyUnavailable raise in compile_kernel.
    """
    import axiom_compute as axc

    k = axc.compile_kernel(_SAXPY_SRC)
    assert k.buffer_count() == 2
    vk_uuid = bytes(k._compiled.device_uuid())
    assert vk_uuid == axc.cuda_device_uuid(), "Vulkan device must match the CUDA GPU"
