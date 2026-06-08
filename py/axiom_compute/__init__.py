"""AXIOM-Compute PyTorch frontend — CUDA↔Vulkan zero-copy kernels (M4.1, Phase 1).

A torch CUDA tensor feeds an AXIOM-Compute Vulkan kernel on the SAME physical GPU
with ZERO host copies. Vulkan allocates + exports the shared buffers (OPAQUE_FD);
CUDA imports them; a shared Vulkan TIMELINE external semaphore synchronizes the
handshake (torch writes → AXIOM dispatches → torch reads), all on ONE CUDA stream.

Honest framing (R-6): zero-copy ELIMINATES the host transfer, but the Vulkan kernel
is ~34% of cuBLAS — this is NOT a cuBLAS replacement. The win is no-host-copy + real
torch interop. Inputs are truly zero-copy when allocated DIRECTLY in the shared
buffers (``Kernel.shared_tensor``); an arbitrary torch tensor uses a device-to-device
copy into the shared buffer ("no host round-trip, but not input-zero-copy").

Public API:
    axiom_compute.compile_kernel(source: str, device=None) -> Kernel
    Kernel(*tensors, out_shape, out_dtype, workgroups, push_constants=b'') -> torch.Tensor
    AxiomError, ZeroCopyUnavailable
"""

from __future__ import annotations

import itertools
import threading
from typing import Optional, Sequence

from . import _cuda_interop
from ._cuda_interop import CudaInteropError

try:
    from . import _axiom_compute  # native module built by maturin
except Exception as _e:  # pragma: no cover
    _axiom_compute = None
    _IMPORT_ERROR = _e
else:
    _IMPORT_ERROR = None


class AxiomError(RuntimeError):
    """Base error for the axiom_compute frontend."""


class ZeroCopyUnavailable(AxiomError):
    """The zero-copy path is unavailable (e.g. CPU tensor, wrong GPU, missing ICD).

    Raised instead of silently copying through the host (AT-2107) — fail-closed.
    """


def _require_native():
    if _axiom_compute is None:
        raise AxiomError(
            "the native _axiom_compute module is not built; run "
            "`pip install maturin && (cd py && maturin develop --release)`. "
            f"Original import error: {_IMPORT_ERROR}"
        )


def cuda_device_uuid(device=None) -> bytes:
    """Return the CUDA device's UUID as 16 raw bytes (matches Vulkan deviceUUID)."""
    import torch

    if device is None:
        idx = torch.cuda.current_device()
    elif isinstance(device, int):
        idx = device
    else:
        idx = torch.device(device).index
        if idx is None:
            idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return bytes(props.uuid.bytes)


def _check_cuda_available():
    import torch

    if not torch.cuda.is_available():
        raise ZeroCopyUnavailable("torch.cuda is not available — no CUDA device")


def compile_kernel(source: str, device=None) -> "Kernel":
    """Compile a `.axc` source into a zero-copy :class:`Kernel` bound to a CUDA GPU.

    FAIL-CLOSED (AT-2107): if torch.cuda is unavailable, the ICD lacks the external-fd
    extensions, or no Vulkan device matches the CUDA device UUID, a
    :class:`ZeroCopyUnavailable` is raised — never a silent host copy.
    """
    _require_native()
    _check_cuda_available()
    uuid = cuda_device_uuid(device)
    try:
        compiled = _axiom_compute.compile_kernel(source, list(uuid))
    except Exception as e:
        # UUID-mismatch / missing extensions surface here.
        raise ZeroCopyUnavailable(f"zero-copy unavailable: {e}") from e
    return Kernel(compiled, device)


class Kernel:
    """A compiled AXIOM kernel callable on torch CUDA tensors (zero host copy).

    G-5: captures ONE CUDA stream S at construction (``torch.cuda.current_stream()``);
    all shared-tensor ops + both cudaSignal/Wait run on S.
    G-7: a single monotone counter reserves a fresh (V1, V2) per call.
    G-4: :meth:`close` runs the exact teardown order (drain → drop view →
    cudaDestroyExternalMemory → cudaDestroyExternalSemaphore → Vulkan free), idempotent.
    """

    def __init__(self, compiled, device=None):
        import torch

        self._compiled = compiled
        self._device = torch.device("cuda" if device is None else device)
        # G-5: capture ONE stream for all shared-tensor ops + signal/wait.
        self._stream = torch.cuda.current_stream(self._device)
        # G-7: monotone (V1, V2) counter — itertools.count under a lock. Each call
        # reserves V1 = 2k+1, V2 = 2k+2 from a strictly increasing source.
        self._counter = itertools.count(1)
        self._counter_lock = threading.Lock()
        self._closed = False
        # Cross-check the Vulkan device UUID == our CUDA UUID (defense in depth).
        vk_uuid = bytes(compiled.device_uuid())
        cu_uuid = cuda_device_uuid(self._device)
        if vk_uuid and vk_uuid != cu_uuid:
            raise ZeroCopyUnavailable(
                f"Vulkan device UUID {vk_uuid.hex()} != CUDA device UUID {cu_uuid.hex()}"
            )

    @property
    def stream(self):
        """The captured CUDA stream S — ALL shared-tensor ops MUST use it (G-5)."""
        return self._stream

    def buffer_count(self) -> int:
        return self._compiled.buffer_count()

    def workgroup_size(self):
        return self._compiled.workgroup_size()

    def _next_values(self):
        with self._counter_lock:
            k = next(self._counter)
        # Two distinct, strictly increasing values per call (G-7).
        return 2 * k - 1, 2 * k

    def allocate_shared(self, sizes: Sequence[int], use_binary_fallback: bool = False):
        """Allocate the exportable DEDICATED shared buffers + import them into CUDA.

        Returns a :class:`SharedSession` holding the torch CUDA tensor views (one per
        binding) and the imported semaphore. The user writes inputs into these views,
        then calls :meth:`SharedSession.run`.
        """
        if self._closed:
            raise AxiomError("kernel is closed")
        bufs = self._compiled.allocate_shared_buffers(list(int(s) for s in sizes), use_binary_fallback)
        return SharedSession(self, bufs)

    def close(self):
        """Idempotent; sessions own their own teardown. Kept for symmetry."""
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class SharedSession:
    """One set of shared buffers + imported CUDA handles for a kernel (G-4 lifetime)."""

    def __init__(self, kernel: Kernel, bufs):
        self._kernel = kernel
        self._bufs = bufs
        self._closed = False
        self._last_signal = 0

        sizes = bufs.sizes()
        dedicated = bufs.dedicated()
        fds = bufs.memory_fds()  # G-3: ownership transferred to Python here.

        # Import each exported buffer fd into CUDA (G-1 dedicated-flag match).
        self._imports = []
        for fd, size, ded in zip(fds, sizes, dedicated):
            self._imports.append(_cuda_interop.import_memory_fd(fd, size, ded))

        # Import the shared semaphore (timeline, or the binary-pair fallback, G-6).
        self._timeline = bufs.timeline_supported()
        if self._timeline:
            sem_fd = bufs.semaphore_fd()
            self._sem = _cuda_interop.import_timeline_semaphore_fd(sem_fd)
            self._binpair = None
        else:
            fd_a, fd_b = bufs.binary_semaphore_fds()
            self._binpair = _cuda_interop.import_binary_semaphore_fds(fd_a, fd_b)
            self._sem = None

    @property
    def stream(self):
        return self._kernel.stream

    def tensor(self, index: int, shape, dtype):
        """A NON-OWNING torch CUDA tensor view of shared buffer ``index`` (G-4)."""
        return self._imports[index].as_torch_tensor(shape, dtype, self.stream)

    def run(self, workgroups, push_constants: bytes = b""):
        """Drive the zero-copy handshake: CUDA signal V1 → Vulkan compute → CUDA wait V2.

        All on the captured stream S (G-5). Returns after the wait is ENQUEUED; the
        caller reads the output tensor on the SAME stream (it will be ordered after the
        kernel write). For binary fallback, signal_a/wait_b replace the timeline values.
        """
        if self._closed:
            raise AxiomError("session is closed")
        v1, v2 = self._kernel._next_values()

        if self._timeline:
            # 2. CUDA signals V1 after its writes (device-side, stream-ordered).
            self._sem.signal(v1, self.stream)
            # 3-4. Vulkan waits V1 @COMPUTE_SHADER, signals V2 after the kernel writes.
            self._kernel._compiled.dispatch_zero_copy(
                self._bufs, tuple(workgroups), bytes(push_constants), v1, v2
            )
            # 5. CUDA waits V2 before any torch op reads the output (stream-ordered).
            self._sem.wait(v2, self.stream)
        else:
            # G-6 binary fallback: signal A (torch→Vulkan), Vulkan waits A / signals B,
            # CUDA waits B (Vulkan→torch). Values are ignored by binary semaphores.
            self._binpair.signal_a(self.stream)
            self._kernel._compiled.dispatch_zero_copy(
                self._bufs, tuple(workgroups), bytes(push_constants), v1, v2
            )
            self._binpair.wait_b(self.stream)

        self._last_signal = v2
        return v2

    def staging_buffers_allocated(self) -> int:
        """AT-2109 (G-2): MUST be 0 — no staging buffer on the zero-copy path."""
        return self._bufs.staging_buffers_allocated()

    def copy_buffer_records(self) -> int:
        """AT-2109 (G-2): MUST be 0 — no vkCmdCopyBuffer on the zero-copy path."""
        return self._bufs.copy_buffer_records()

    def close(self):
        """G-4 EXACT teardown order, idempotent.

        (1) drain timeline >= last V2 (or device_wait_idle on binary fallback)
        (2) drop torch views (release Python references to the imports)
        (3) cudaDestroyExternalMemory
        (4) cudaDestroyExternalSemaphore
        (5) Rust vkFreeMemory + vkDestroyBuffer + vkDestroySemaphore
        """
        if self._closed:
            return
        self._closed = True

        # (1) drain in-flight GPU work BEFORE any free (no host fence on the data path).
        try:
            self._kernel._compiled.drain(self._bufs, self._last_signal)
        except Exception:
            # Best-effort: even if drain fails, proceed so we do not leak handles.
            pass

        # (2) drop torch views: nothing torch-owned references the foreign ptr beyond
        # the user's tensors; we release our import references next which invalidates them.
        # (3) cudaDestroyExternalMemory (per import).
        for imp in self._imports:
            imp.close()
        self._imports = []

        # (4) cudaDestroyExternalSemaphore.
        if self._sem is not None:
            self._sem.close()
            self._sem = None
        if self._binpair is not None:
            self._binpair.close()
            self._binpair = None

        # (5) Rust frees the Vulkan side LAST.
        self._kernel._compiled.teardown_shared_buffers(self._bufs)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "compile_kernel",
    "cuda_device_uuid",
    "Kernel",
    "SharedSession",
    "AxiomError",
    "ZeroCopyUnavailable",
]
