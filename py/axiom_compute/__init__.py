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

Phase 1 ASSUMPTIONS (revisit for M4.2 cross-vendor):
  - SINGLE CUDA STREAM (G-5): every shared-tensor op AND the signal/wait handshake MUST
    run on the ONE stream captured at ``Kernel`` construction. A cross-stream op without
    an explicit event races the cross-API handshake (silent torn read). The frontend
    DETECTS divergence and warns (``SharedSession.run``/``tensor``), but cannot force the
    caller's ops onto the captured stream.
  - NVIDIA-ONLY sync: the exportable buffers are ``VK_SHARING_MODE_EXCLUSIVE`` and the
    dispatch emits NO ``VK_QUEUE_FAMILY_EXTERNAL`` ownership-transfer barrier — it relies
    SOLELY on the external timeline semaphore for execution AND memory ordering. This is
    correct on the M4.1 target (discrete NVIDIA, single Vulkan queue, 580.x driver) and
    is proven bit-exact there, but is NOT spec-portable: AMD/Intel/MoltenVK generally
    require CONCURRENT sharing over {compute, VK_QUEUE_FAMILY_EXTERNAL} or explicit
    ownership-transfer barriers. This MUST be revisited for cross-vendor support.

Public API:
    axiom_compute.compile_kernel(source: str, device=None) -> Kernel
    Kernel(*tensors, out_shape, out_dtype, workgroups, push_constants=b'') -> torch.Tensor
    AxiomError, ZeroCopyUnavailable
"""

from __future__ import annotations

import itertools
import struct
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


def compile_kernel(source: str, device=None, strategy: Optional[dict] = None) -> "Kernel":
    """Compile a `.axc` source into a zero-copy :class:`Kernel` bound to a CUDA GPU.

    FAIL-CLOSED (AT-2107): if torch.cuda is unavailable, the ICD lacks the external-fd
    extensions, or no Vulkan device matches the CUDA device UUID, a
    :class:`ZeroCopyUnavailable` is raised — never a silent host copy.

    M4.1p2: ``strategy`` is an optional ``dict[str, int]`` of strategy-hole assignments. A
    hole-carrying source (e.g. the M3.6 Q4_K_M kernel) MUST pass all its holes; ``strategy=None``
    on such a source FAILS at compile (UnresolvedStrategyHole) — the kwarg is load-bearing
    (AT-2103d). The Phase-1 saxpy path (no holes) passes ``strategy=None`` unchanged.

    A compile error from an unresolved strategy hole is a COMPILE error (raised here as
    :class:`AxiomError`), NOT a zero-copy-availability problem — so it is surfaced distinctly
    from the UUID-mismatch / missing-extension :class:`ZeroCopyUnavailable` class.
    """
    _require_native()
    _check_cuda_available()
    uuid = cuda_device_uuid(device)
    strat = None
    if strategy is not None:
        # Normalize to a plain {str: int} dict for the PyO3 boundary.
        strat = {str(k): int(v) for k, v in strategy.items()}
    try:
        compiled = _axiom_compute.compile_kernel(source, list(uuid), strat)
    except Exception as e:
        msg = str(e)
        # A strategy-hole / source compile failure is a COMPILE error, not an interop
        # availability problem — surface it as AxiomError so callers (AT-2103d) can tell
        # "the kernel did not compile" apart from "zero-copy is unavailable on this box".
        if "compile error" in msg or "UnresolvedStrategyHole" in msg or "strategy" in msg.lower():
            raise AxiomError(f"compile failed: {e}") from e
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

    def push_constant_bytes(self) -> int:
        return self._compiled.push_constant_bytes()

    def scalar_layout(self):
        """The scalar push-constant slots as ``[(name, offset, size_bytes, ty_str), ...]``.

        Python uses this to assemble push constants by OFFSET+NAME (not positionally) — the
        analogue of the bench's ``assemble_pc`` — so a wrong scalar ORDER is impossible (each
        value lands at its own std430 offset).

        M4.1p4: the 4th element ``ty_str`` is the scalar's source type keyword
        (``'f32'``/``'u32'``/...), surfaced from the HIR slot's ``ScalarTy`` so the packer can
        tell an f32 push constant (e.g. ``inv_sqrt_d``) from a u32 (both report ``size_bytes==4``)
        and IEEE-754-pack it instead of ``int(0.125)==0``-ing it to zero bytes.
        """
        return self._compiled.scalar_layout()

    def assemble_push_constants(self, values: dict) -> bytes:
        """Pack a ``{name: scalar}`` dict into the std430 push-constant blob.

        Each named scalar is written at its own ``offset`` from :meth:`scalar_layout` (little-
        endian, ``size_bytes`` wide). Every scalar slot the kernel declares MUST be supplied;
        an unknown name or a missing slot raises (no silent zero-fill of a required scalar).

        M4.1p4 — the f32 push-constant fix: the packer BRANCHES on the slot ``ty_str``. A FLOAT
        slot (``ty_str == 'f32'``) is IEEE-754-packed via ``struct.pack('<f', float(value))`` so
        e.g. ``inv_sqrt_d=0.125`` packs to ``b'\\x00\\x00\\x00\\x3e'`` (0x3E000000), NOT
        ``int(0.125)==0`` (four zero bytes → uniform softmax, silently wrong). An INTEGER slot
        keeps the existing ``int(value).to_bytes(...)`` path (the q4km M/N/K/n_blocks_per_row
        path is UNCHANGED). f16/f64 floats are rejected (not needed by any current kernel — the
        only float push constant is the f32 ``inv_sqrt_d``; fail-loud rather than silently
        narrow).
        """
        layout = self.scalar_layout()
        total = self._compiled.push_constant_bytes()
        blob = bytearray(total)
        names_in_layout = {name for (name, _off, _sz, _ty) in layout}
        unknown = set(values) - names_in_layout
        if unknown:
            raise AxiomError(
                f"assemble_push_constants: value(s) {sorted(unknown)} are not scalar push "
                f"constants of this kernel (declared: {sorted(names_in_layout)})"
            )
        for (name, offset, size, ty_str) in layout:
            if name not in values:
                raise AxiomError(
                    f"assemble_push_constants: missing required scalar push constant '{name}' "
                    f"(kernel declares {sorted(names_in_layout)})"
                )
            value = values[name]
            if ty_str == "f32":
                # IEEE-754 f32 (e.g. inv_sqrt_d) — NOT int(0.125)==0.
                blob[offset:offset + 4] = struct.pack("<f", float(value))
            elif ty_str in ("f16", "f64"):
                raise AxiomError(
                    f"assemble_push_constants: float scalar '{name}' has type {ty_str!r}, "
                    "which this packer does not support yet (only f32 floats and integer "
                    "scalars are packable); add a packing branch if a kernel needs it."
                )
            else:
                # Integer slot (u8/u16/u32/u64/i*): the existing int path, UNCHANGED.
                val = int(value)
                if size == 4:
                    blob[offset:offset + 4] = val.to_bytes(4, "little", signed=False)
                elif size == 8:
                    blob[offset:offset + 8] = val.to_bytes(8, "little", signed=False)
                elif size == 2:
                    blob[offset:offset + 2] = val.to_bytes(2, "little", signed=False)
                elif size == 1:
                    blob[offset:offset + 1] = val.to_bytes(1, "little", signed=False)
                else:
                    raise AxiomError(
                        f"assemble_push_constants: unsupported scalar size {size} for '{name}'"
                    )
        return bytes(blob)

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
        # M4.1p3 (R2): poison + at-most-once host-signal-release bookkeeping.
        self._poisoned = False
        self._released_value = 0  # the timeline value already host-signaled (monotone guard)

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

    def _assert_captured_stream(self, where: str):
        """FIX-4 (G-5): warn LOUDLY if the active CUDA stream != the captured stream S.

        Phase 1 assumes ALL shared-tensor ops + the signal/wait handshake run on the ONE
        stream captured at :class:`Kernel` construction. A cross-stream op without an
        explicit event would race the cross-API handshake (a silent torn read). We cannot
        force the caller's ops onto S, but we DETECT divergence here and warn so the
        footgun is not silent.
        """
        import torch
        import warnings

        active = torch.cuda.current_stream(self._kernel._device)
        captured = self.stream
        # torch.cuda.Stream compares by underlying cudaStream_t handle.
        if active.cuda_stream != captured.cuda_stream:
            warnings.warn(
                f"axiom_compute: {where} called while the active CUDA stream "
                f"(cuda_stream={active.cuda_stream:#x}) differs from the stream captured "
                f"at Kernel construction (cuda_stream={captured.cuda_stream:#x}). Phase 1 "
                "REQUIRES all shared-tensor ops AND the signal/wait handshake to run on "
                "the ONE captured stream (G-5); a cross-stream op without an explicit "
                "event races the CUDA<->Vulkan handshake and can torn-read the shared "
                "buffer. Wrap your ops in `with torch.cuda.stream(kernel.stream):` or "
                "construct the Kernel on the stream you intend to use.",
                stacklevel=3,
            )

    def tensor(self, index: int, shape, dtype):
        """A NON-OWNING torch CUDA tensor view of shared buffer ``index`` (G-4).

        G-5: the view is created on the captured stream S; ALL ops on it MUST use S.
        """
        self._assert_captured_stream("SharedSession.tensor()")
        return self._imports[index].as_torch_tensor(shape, dtype, self.stream)

    def run(self, workgroups, push_constants: bytes = b""):
        """Drive the zero-copy handshake: CUDA signal V1 → Vulkan compute → CUDA wait V2.

        All on the captured stream S (G-5). Returns after the wait is ENQUEUED; the
        caller reads the output tensor on the SAME stream (it will be ordered after the
        kernel write). For binary fallback, signal_a/wait_b replace the timeline values.
        """
        if self._closed:
            raise AxiomError("session is closed")
        # FIX-4 (G-5): the whole handshake (signal V1 / dispatch / wait V2) runs on the
        # captured stream S; warn if the active stream diverged.
        self._assert_captured_stream("SharedSession.run()")
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

    # ── M4.1p3 (R2): bounded completion barrier + CUDA-wait release + poison ──────────
    def wait_completion(self, timeout_ms: Optional[int] = None) -> None:
        """BOUNDED host wait until the timeline reaches the last signaled V2.

        Used by the registered op as its owned-output completion barrier (before cloning
        C) and by error recovery — NOT on the zero-overhead :meth:`run` data path (G-5
        intact). It calls the Rust ``drain_with_timeout`` (already FenceTimeout-returning)
        with an explicit per-call bound (default: the context's ``AXC_FENCE_TIMEOUT_MS``).

        On the happy path the timeline is already >= V2 so this returns immediately. On a
        one-sided GPU fault (V1 signaled, V2 never reached) it raises :class:`AxiomError`
        within the bound (the op's error path then releases + poisons + rebuilds). On the
        binary fallback there is no host-waitable counter; we fall back to the bounded
        ``device_wait_idle`` (the Rust drain does so internally).
        """
        if self._closed:
            raise AxiomError("session is closed")
        try:
            if self._timeline:
                self._kernel._compiled.drain_with_timeout(
                    self._bufs, self._last_signal, timeout_ms
                )
            else:
                # No host-waitable counter on the binary pair; bound via device_wait_idle.
                self._kernel._compiled.device_wait_idle()
        except Exception as e:
            raise AxiomError(
                f"wait_completion: the cross-API handshake did not complete within "
                f"{'AXC_FENCE_TIMEOUT_MS' if timeout_ms is None else f'{timeout_ms} ms'} "
                f"(a one-sided GPU fault may have stalled the wait); raise "
                f"AXC_FENCE_TIMEOUT_MS for a legitimately long kernel. Underlying: {e}"
            ) from e

    def release_cuda_wait(self, value: int) -> None:
        """R2: HOST-signal the shared Vulkan timeline to ``value`` (== V2) so a dangling
        ``cudaWaitExternalSemaphoresAsync(value)`` enqueued on stream S is SATISFIED and S
        unblocks (the both-sides release after a GPU-fault-mid-dispatch).

        AT-MOST-ONCE per (session, value) and strictly MONOTONE (WATCH-ITEM 1): a host
        timeline signal to a value not greater than the current payload is illegal. We
        guard with ``self._released_value`` so the poison/teardown path NEVER re-signals a
        value already released (or any value <= one already released). A no-op on the
        binary fallback (no single-value host-signal release exists there).
        """
        value = int(value)
        if not self._timeline:
            return  # no single-value host-signal release for the binary pair (M4.2)
        if value <= self._released_value:
            return  # already released at >= this value; re-signalling would be non-monotone
        self._kernel._compiled.host_signal_timeline(self._bufs, value)
        self._released_value = value

    def device_wait_idle(self) -> None:
        """PCR-1: drain the WHOLE Vulkan queue (``vkDeviceWaitIdle``). On the recovery path
        this runs BEFORE any host-signal: it RETIRES every still-enqueued submit, so a
        merely-SLOW (false-positive-timeout) dispatch signals its own V2 (the GPU stays the
        SOLE signaler), and a genuine fault leaves V2 unsignaled — letting the caller decide
        host-signal vs. skip via :meth:`external_timeline_value`. Bounded only by the driver;
        this is the exceptional/error path, never the happy path (G-5 fast path untouched)."""
        if self._closed:
            raise AxiomError("session is closed")
        self._kernel._compiled.device_wait_idle()

    def external_timeline_value(self) -> int:
        """PCR-1: current payload of the shared external timeline (``vkGetSemaphoreCounterValue``).
        Queried AFTER :meth:`device_wait_idle` to learn whether the GPU's own submit already
        signaled V2 (payload >= V2 → skip host-signal, avoid a double-signal UB) or the
        dispatch genuinely faulted (payload < V2 → host-signal is the SOLE signaler).
        Raises on the binary fallback (no host-queryable counter)."""
        if self._closed:
            raise AxiomError("session is closed")
        return int(self._kernel._compiled.get_external_timeline_value(self._bufs))

    def poison(self) -> None:
        """R2: mark the session UNUSABLE so the op cache evicts + rebuilds it on the next
        call (a faulted session + its stream S are NEVER reused for a new dispatch)."""
        self._poisoned = True

    @property
    def is_poisoned(self) -> bool:
        return self._poisoned

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
        # FIX-2: NEVER free buffers / the CUDA import while work may still be in-flight —
        # that is a use-after-free. If the timeline drain fails or times out, HARD-FALL
        # BACK to a blocking device_wait_idle (a queue-wide barrier) so the device is
        # PROVABLY idle before any free. Only if EVEN device_wait_idle fails is the device
        # in an unrecoverable state — we then log loudly and re-raise WITHOUT freeing
        # (freeing into an in-flight GPU is worse than leaking handles).
        try:
            self._kernel._compiled.drain(self._bufs, self._last_signal)
        except Exception as drain_exc:
            try:
                self._kernel._compiled.device_wait_idle()
            except Exception as idle_exc:
                # Unrecoverable: the device never went idle. Do NOT free (UAF risk).
                # Re-arm _closed=False is pointless (the device is wedged); surface loudly.
                import warnings

                warnings.warn(
                    "axiom_compute: SharedSession.close() could not drain the timeline "
                    f"({drain_exc!r}) AND device_wait_idle failed ({idle_exc!r}); the GPU "
                    "is in an unrecoverable state. Refusing to free shared buffers / CUDA "
                    "imports while work may be in-flight (would be a use-after-free). "
                    "Handles are intentionally LEAKED.",
                    stacklevel=2,
                )
                raise AxiomError(
                    "SharedSession.close(): drain and device_wait_idle both failed; "
                    "refusing to free while GPU work may be in-flight (UAF). "
                    f"drain={drain_exc!r} device_wait_idle={idle_exc!r}"
                ) from idle_exc
            # device_wait_idle succeeded → the device is provably idle; safe to free below.

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


# M4.1p2: the Q4_K_M headline op + the (Rust-backed) reference helpers. Imported at the bottom
# so the module-level names (compile_kernel/Kernel/AxiomError/...) the op depends on already exist.
from .q4km import (  # noqa: E402
    Q4KMMatmul,
    q4km_matmul,
    pack_q4km,
    q4km_f32_reference,
    q4km_combined_metric,
    q4km_raw_rel_diff,
    Q4KM_SUPERBLOCK_BYTES,
)

# M4.1p3: the torch.library custom-op surface. Importing ops registers helpers; calling
# register_q4km_op() defines torch.ops.axiom.q4km_matmul. Both are guarded so the package
# still imports WITHOUT torch present (the op only materializes when torch is available).
from .ops import (  # noqa: E402
    AXIOM_OP_NAMESPACE,
    Q4KM_OP_NAME,
    register_q4km_op,
    q4km_register_weights,
    q4km_release_weights,
    q4km_activation_view,
    last_dispatch_path,
    last_dispatch_staging,
)

# M4.1p4: the merged FlashAttention zero-copy wrapper (mirrors the q4km import block).
from .flash_attention import (  # noqa: E402
    FlashAttention,
    flash_attention,
)

# M4.1p4: the FlashAttention torch.library custom-op surface (mirrors the q4km ops block).
from .attention_ops import (  # noqa: E402
    FLASH_ATTENTION_OP_NAME,
    register_flash_attention_op,
    flash_attention_session,
    flash_attention_views,
    last_attn_dispatch_path,
    last_attn_dispatch_staging,
    release_attention_session,
)

# M4.3: the torch.compile custom backend (backend="axiom") that auto-lowers a matching SDPA to
# torch.ops.axiom.flash_attention. Guarded like the op registrations (no-op without torch).
from .compile_backend import (  # noqa: E402
    AXIOM_BACKEND_NAME,
    register_axiom_backend,
    axiom_backend,
    axiom_sdpa_dispatch_count,
    axiom_sdpa_runtime_fire_count,
    reset_axiom_sdpa_counters,
)

# Define torch.ops.axiom.q4km_matmul + torch.ops.axiom.flash_attention at import
# (no-op if torch is absent).
register_q4km_op()
register_flash_attention_op()
# Register backend="axiom" at import so torch.compile(model, backend="axiom") resolves
# (idempotent, no-op without torch).
register_axiom_backend()

__all__ = [
    "compile_kernel",
    "cuda_device_uuid",
    "Kernel",
    "SharedSession",
    "AxiomError",
    "ZeroCopyUnavailable",
    # M4.1p2 Q4_K_M headline op + Rust-backed reference.
    "Q4KMMatmul",
    "q4km_matmul",
    "pack_q4km",
    "q4km_f32_reference",
    "q4km_combined_metric",
    "q4km_raw_rel_diff",
    "Q4KM_SUPERBLOCK_BYTES",
    # M4.1p3 torch.library custom-op surface.
    "AXIOM_OP_NAMESPACE",
    "Q4KM_OP_NAME",
    "register_q4km_op",
    "q4km_register_weights",
    "q4km_release_weights",
    "q4km_activation_view",
    "last_dispatch_path",
    "last_dispatch_staging",
    # M4.1p4 FlashAttention zero-copy wrapper + torch.library custom-op surface.
    "FlashAttention",
    "flash_attention",
    "FLASH_ATTENTION_OP_NAME",
    "register_flash_attention_op",
    "flash_attention_session",
    "flash_attention_views",
    "last_attn_dispatch_path",
    "last_attn_dispatch_staging",
    "release_attention_session",
    # M4.3 torch.compile custom backend (backend="axiom") auto-lowering SDPA -> flash_attention.
    "AXIOM_BACKEND_NAME",
    "register_axiom_backend",
    "axiom_backend",
    "axiom_sdpa_dispatch_count",
    "axiom_sdpa_runtime_fire_count",
    "reset_axiom_sdpa_counters",
]
