"""CUDA-side import for AXIOM-Compute zero-copy interop — M4.1 (Phase 1b/1c).

This module wraps the CUDA runtime calls that import the Vulkan-exported OPAQUE_FD
handles into CUDA and drive the cross-API timeline handshake:

    cudaImportExternalMemory / cudaExternalMemoryGetMappedBuffer
    cudaImportExternalSemaphore
    cudaSignalExternalSemaphoresAsync / cudaWaitExternalSemaphoresAsync
    cudaDestroyExternalMemory / cudaDestroyExternalSemaphore

cuda-python (``from cuda.bindings import runtime``) is REQUIRED for this path. There
is NO ctypes-to-libcudart fallback and NO host-copy degradation: if cuda-python is
absent, every import entry point FAILS CLOSED by raising :class:`CudaInteropError`
(see ``have_cuda_python``). The CUDA-ABI calls are isolated here so the rest of the
package never touches the CUDA ABI directly.

G-items honored:
  - G-1: ``import_memory_fd`` sets ``flags = cudaExternalMemoryDedicated`` when
    ``dedicated`` (always, for now) to MATCH the Vulkan ``VkMemoryDedicatedAllocateInfo``.
  - G-3: ``import_memory_fd`` takes RAW fd ownership transferred from Rust and is the
    SINGLE owner that closes it after ``cudaImportExternalMemory`` consumes it.
  - G-4: ``ImportedExternalMemory`` / ``ImportedExternalSemaphore`` track ownership and
    destroy on ``close`` in the exact order; ``close`` is idempotent.
  - G-5: ``signal`` / ``wait`` take an explicit CUDA stream; ALL callers MUST pass the
    SAME stream all torch ops on the shared tensors use.
"""

from __future__ import annotations

import os
from typing import Optional

try:
    from cuda.bindings import runtime as _rt  # type: ignore
    _HAVE_CUDA_PYTHON = True
except Exception:  # pragma: no cover - exercised only without cuda-python
    _rt = None
    _HAVE_CUDA_PYTHON = False


class CudaInteropError(RuntimeError):
    """A CUDA runtime call in the interop path failed."""


def have_cuda_python() -> bool:
    """Whether the cuda-python bindings are importable (preferred path)."""
    return _HAVE_CUDA_PYTHON


def _err_msg(err) -> str:
    """Human-readable message for a cudaError_t (cudaGetErrorString returns (err, bytes))."""
    res = _rt.cudaGetErrorString(err)
    name = res[1] if isinstance(res, tuple) else res
    return name.decode() if isinstance(name, (bytes, bytearray)) else str(name)


def _check(err, *rest):
    """Raise on a non-success cudaError_t; return the trailing result(s).

    cuda-python returns ``(err, *results)`` from every call. Callers may pass either
    the unpacked ``*ret`` (so ``err`` is the bare code) OR the whole tuple as the first
    arg (so ``err`` is a 1-tuple). Both are handled.
    """
    if isinstance(err, tuple):
        # A whole (err, *results) tuple was passed as the single argument.
        rest = err[1:] + rest
        err = err[0]
    success = _rt.cudaError_t.cudaSuccess
    if err != success:
        raise CudaInteropError(f"{err}: {_err_msg(err)}")
    if len(rest) == 0:
        return None
    if len(rest) == 1:
        return rest[0]
    return rest


class ImportedExternalMemory:
    """A CUDA external-memory import + its mapped device pointer (NON-OWNING torch view).

    Lifetime: owns the ``cudaExternalMemory_t`` and the OPAQUE_FD (G-3). ``close``
    destroys the import exactly once (idempotent, G-4). The mapped ``ptr`` is invalid
    after ``close``; any torch view tied to this object must be dropped first.
    """

    def __init__(self, ext_mem, ptr: int, size: int, dedicated: bool, fd: int):
        self._ext_mem = ext_mem
        self.ptr = int(ptr)
        self.size = int(size)
        self.dedicated = bool(dedicated)
        self._fd = fd  # already closed after import; kept for diagnostics only
        self._closed = False

    def as_torch_tensor(self, shape, dtype, stream):
        """Wrap the mapped CUDA pointer as a NON-OWNING torch CUDA tensor (G-4).

        Uses a ``__cuda_array_interface__`` proxy + ``torch.as_tensor`` so torch does
        NOT take the pointer into its caching allocator and never frees the foreign
        devPtr. The returned tensor's data lifetime is tied to THIS object: keep a
        reference to this ``ImportedExternalMemory`` for as long as the tensor is used.
        """
        import torch

        nbytes = 1
        for s in shape:
            nbytes *= int(s)
        itemsize = torch.empty(0, dtype=dtype).element_size()
        nbytes *= itemsize
        if nbytes > self.size:
            raise CudaInteropError(
                f"requested view {nbytes} bytes exceeds mapped buffer {self.size} bytes"
            )

        typestr = _numpy_typestr(dtype)
        ptr = self.ptr

        class _CAIProxy:
            # A minimal CUDA Array Interface proxy. read_only=False; the pointer is a
            # plain device allocation (not owned by torch). torch.as_tensor on a CAI
            # object with a matching device performs a ZERO-COPY wrap.
            __cuda_array_interface__ = {
                "shape": tuple(int(s) for s in shape),
                "typestr": typestr,
                "data": (ptr, False),
                "version": 3,
                "strides": None,
            }

        proxy = _CAIProxy()
        with torch.cuda.stream(stream):
            t = torch.as_tensor(proxy, device="cuda")
        # Tie THIS object (the owner of the cudaExternalMemory_t + mapped ptr) to the
        # tensor's Python lifetime so the mapping outlives the tensor. `self` is the
        # load-bearing keepalive; the proxy instance is kept only for clarity (its
        # __cuda_array_interface__ describes the wrap).
        t._axc_keepalive = (self, proxy)  # type: ignore[attr-defined]
        return t

    def close(self):
        """cudaDestroyExternalMemory exactly once (idempotent, G-4 step 3)."""
        if self._closed:
            return
        self._closed = True
        if _HAVE_CUDA_PYTHON and self._ext_mem is not None:
            _check(_rt.cudaDestroyExternalMemory(self._ext_mem))
        self._ext_mem = None


class ImportedExternalSemaphore:
    """A CUDA external TIMELINE semaphore import driving the cross-API handshake."""

    def __init__(self, ext_sem):
        self._ext_sem = ext_sem
        self._closed = False

    def signal(self, value: int, stream):
        """cudaSignalExternalSemaphoresAsync(fence.value=value) on ``stream`` (G-5)."""
        params = _rt.cudaExternalSemaphoreSignalParams()
        params.params.fence.value = int(value)
        params.flags = 0
        _check(
            _rt.cudaSignalExternalSemaphoresAsync(
                [self._ext_sem], [params], 1, _stream_handle(stream)
            )
        )

    def wait(self, value: int, stream):
        """cudaWaitExternalSemaphoresAsync(fence.value=value) on ``stream`` (G-5)."""
        params = _rt.cudaExternalSemaphoreWaitParams()
        params.params.fence.value = int(value)
        params.flags = 0
        _check(
            _rt.cudaWaitExternalSemaphoresAsync(
                [self._ext_sem], [params], 1, _stream_handle(stream)
            )
        )

    def close(self):
        """cudaDestroyExternalSemaphore exactly once (idempotent, G-4 step 4)."""
        if self._closed:
            return
        self._closed = True
        if _HAVE_CUDA_PYTHON and self._ext_sem is not None:
            _check(_rt.cudaDestroyExternalSemaphore(self._ext_sem))
        self._ext_sem = None


class ImportedBinarySemaphorePair:
    """G-6 fallback: a pair of CUDA external BINARY semaphores (one per direction)."""

    def __init__(self, ext_a, ext_b):
        self._a = ext_a
        self._b = ext_b
        self._closed = False

    def signal_a(self, stream):
        self._signal(self._a, stream)

    def wait_a(self, stream):
        self._wait(self._a, stream)

    def signal_b(self, stream):
        self._signal(self._b, stream)

    def wait_b(self, stream):
        self._wait(self._b, stream)

    def _signal(self, sem, stream):
        params = _rt.cudaExternalSemaphoreSignalParams()
        params.flags = 0
        _check(_rt.cudaSignalExternalSemaphoresAsync([sem], [params], 1, _stream_handle(stream)))

    def _wait(self, sem, stream):
        params = _rt.cudaExternalSemaphoreWaitParams()
        params.flags = 0
        _check(_rt.cudaWaitExternalSemaphoresAsync([sem], [params], 1, _stream_handle(stream)))

    def close(self):
        if self._closed:
            return
        self._closed = True
        if _HAVE_CUDA_PYTHON:
            if self._a is not None:
                _check(_rt.cudaDestroyExternalSemaphore(self._a))
            if self._b is not None:
                _check(_rt.cudaDestroyExternalSemaphore(self._b))
        self._a = None
        self._b = None


def import_memory_fd(fd: int, size: int, dedicated: bool = True) -> ImportedExternalMemory:
    """Import a Vulkan-exported OPAQUE_FD into CUDA + map it to a device pointer.

    G-1: sets ``flags = cudaExternalMemoryDedicated`` when ``dedicated`` to MATCH the
    Vulkan dedicated allocation. G-3: this is the SINGLE owner of ``fd`` and closes it
    after ``cudaImportExternalMemory`` consumes it (CUDA does not take ownership of an
    OpaqueFd on import).
    """
    if not _HAVE_CUDA_PYTHON:
        raise CudaInteropError("cuda-python is required for the CUDA import path")

    handle_desc = _rt.cudaExternalMemoryHandleDesc()
    handle_desc.type = _rt.cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeOpaqueFd
    handle_desc.handle.fd = int(fd)
    handle_desc.size = int(size)
    # G-1: MATCH the Vulkan VkMemoryDedicatedAllocateInfo (always dedicated for now).
    handle_desc.flags = int(_rt.cudaExternalMemoryDedicated) if dedicated else 0

    # FIX-3: an OPAQUE_FD memory fd is NOT consumed by cudaImportExternalMemory — Python
    # is the single owner and MUST close it exactly once, on BOTH the success and the
    # failure path (the import RAISES on error, which previously skipped the close →
    # fd leak per failed allocate). Close in a finally so the fd never leaks.
    # (Distinct from the SEMAPHORE fd, which IS taken by CUDA on import — see
    # import_timeline_semaphore_fd / import_binary_semaphore_fds, which do NOT close.)
    try:
        ext_mem = _check(*_rt.cudaImportExternalMemory(handle_desc))
    finally:
        # G-3: CUDA dup'd/consumed the fd by now (or failed); we are the single owner.
        try:
            os.close(int(fd))
        except OSError:
            pass

    buf_desc = _rt.cudaExternalMemoryBufferDesc()
    buf_desc.offset = 0
    buf_desc.size = int(size)
    buf_desc.flags = 0
    ptr = _check(*_rt.cudaExternalMemoryGetMappedBuffer(ext_mem, buf_desc))

    return ImportedExternalMemory(ext_mem, int(ptr), int(size), dedicated, int(fd))


def import_timeline_semaphore_fd(fd: int) -> ImportedExternalSemaphore:
    """Import a Vulkan TIMELINE semaphore fd (type=9 TimelineSemaphoreFd)."""
    if not _HAVE_CUDA_PYTHON:
        raise CudaInteropError("cuda-python is required for the CUDA import path")
    desc = _rt.cudaExternalSemaphoreHandleDesc()
    desc.type = (
        _rt.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd
    )
    desc.handle.fd = int(fd)
    desc.flags = 0
    ext_sem = _check(*_rt.cudaImportExternalSemaphore(desc))
    # CUDA takes ownership of the semaphore fd on import; do NOT close it here.
    return ImportedExternalSemaphore(ext_sem)


def import_binary_semaphore_fds(fd_a: int, fd_b: int) -> ImportedBinarySemaphorePair:
    """G-6 fallback: import two binary external semaphore fds (OpaqueFd)."""
    if not _HAVE_CUDA_PYTHON:
        raise CudaInteropError("cuda-python is required for the CUDA import path")

    def _imp(fd):
        desc = _rt.cudaExternalSemaphoreHandleDesc()
        desc.type = _rt.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeOpaqueFd
        desc.handle.fd = int(fd)
        desc.flags = 0
        return _check(*_rt.cudaImportExternalSemaphore(desc))

    return ImportedBinarySemaphorePair(_imp(fd_a), _imp(fd_b))


def _stream_handle(stream) -> int:
    """Extract the raw cudaStream_t integer handle from a torch stream / int."""
    if isinstance(stream, int):
        return stream
    # torch.cuda.Stream exposes .cuda_stream as an int.
    cs = getattr(stream, "cuda_stream", None)
    if cs is not None:
        return int(cs)
    raise CudaInteropError(f"cannot extract a CUDA stream handle from {stream!r}")


def _numpy_typestr(dtype) -> str:
    """Map a torch dtype to a CUDA-Array-Interface typestr (little-endian)."""
    import torch

    table = {
        torch.float32: "<f4",
        torch.float16: "<f2",
        torch.float64: "<f8",
        torch.int32: "<i4",
        torch.int64: "<i8",
        torch.int16: "<i2",
        torch.int8: "|i1",
        torch.uint8: "|u1",
    }
    ts = table.get(dtype)
    if ts is None:
        raise CudaInteropError(f"unsupported dtype for CAI wrap: {dtype}")
    return ts
