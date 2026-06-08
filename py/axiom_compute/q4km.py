"""M4.1 Phase 2 — the Q4_K_M dequant+coopmat matmul as the HEADLINE op through the
CUDA↔Vulkan zero-copy PyTorch frontend.

Runs the M3.6 leader kernel
(``examples/q4km_matmul_rb_coopmat_f32acc_cached.axc``, 42.86 TFLOPS) — Q4_K_M dequant
fused into a 2×2 register-blocked f32-accumulator coopmat matmul — directly on torch CUDA
tensors with NO host copies on the per-call data path.

BUFFER PLAN (the honest zero-copy split):
  - q (Q4_K_M weight superblocks, 144 bytes each)  = UPLOAD-ONCE shared buffer. The static
    quantized weights are written ONCE into an exportable shared view (``upload_weights``) and
    reused across every ``matmul`` call. This is the legitimate static-weights carve-out — it is
    a one-time device-side copy into the shared view, NOT a per-call host copy and NOT a
    dispatch-path ``vkCmdCopyBuffer``.
  - x (f16 activations)  = ZERO-COPY per-call torch CUDA tensor, layout **[K, N] row-major**
    (``x[k*N + col]``). The kernel reads this f16 buffer DIRECTLY.
  - C (f32 output)       = ZERO-COPY per-call torch CUDA tensor, layout **[M, N]** (``C[m*N+col]``).

  Per-call x/C are GENUINELY staging-free: 0 staging buffers, 0 ``vkCmdCopyBuffer`` (AT-2103c).

ZERO-COPY LAYOUT DISCLOSURE (honest — carry the reviewer's note):
  The gated zero-copy path requires the activation already in **[K, N] f16** layout, written
  DIRECTLY into the shared x view (use :meth:`Q4KMMatmul.x_view` / write into the returned view).
  A natural [N, K] inference activation (tokens × features under the ``W[M,K] @ x[K,N]``
  convention) would need a per-call transpose/repack — a device-to-device copy that is NOT the
  zero-copy gated path. :meth:`Q4KMMatmul.matmul` offers a CONVENIENCE overload that copies an
  arbitrary already-[K,N] f16 CUDA tensor into the shared view (no host round-trip, but the copy
  is disclosed); a [N,K] tensor is rejected (we never silently transpose-and-pretend-zero-copy).

CORRECTNESS ORACLE (single source of truth):
  The reference is the Rust ``axc_driver::q4km_oracle`` (the f32-accumulator CPU reference every
  M3.x Q4_K_M test gates on), exposed via the PyO3 bridge ``_axiom_compute.q4km_f32_reference`` /
  ``q4km_combined_metric``. There is NO torch dequant re-implementation. The FROZEN 1e-3
  combined condition-aware gate is computed in RUST and cannot be loosened in Python (AT-2103).

HONEST FRAMING (R-6): zero-copy ELIMINATES the host transfer, but the Vulkan kernel is below
  cuBLAS at the A/B shape (MEASURED ~53% of a same-machine f32 ``torch.matmul`` cuBLAS GEMM) —
  this is NOT a cuBLAS replacement and we make NO beat-cuBLAS claim.
"""

from __future__ import annotations

from typing import Optional

from . import _axiom_compute
from . import compile_kernel as _compile_kernel
from . import AxiomError, ZeroCopyUnavailable

# ── Q4_K_M constants (ggml block_q4_K) ──────────────────────────────────────────────
Q4KM_SUPERBLOCK_BYTES: int = 144  # 144-byte superblock (256 weights)
_Q4KM_SUPERBLOCK_ELEMS: int = 256

# The M3.6 leader kernel source (compiled with the full 5-hole strategy assignment).
import os as _os  # noqa: E402

_KERNEL_FILENAME = "q4km_matmul_rb_coopmat_f32acc_cached.axc"
# Candidate locations: (1) a copy bundled INSIDE the installed package (kernels/...), (2) the
# repo's examples/ dir (3 levels up — the editable/source-tree layout).
_KERNEL_CANDIDATES = (
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "kernels", _KERNEL_FILENAME),
    _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
        "examples",
        _KERNEL_FILENAME,
    ),
)


def _load_kernel_src() -> str:
    for path in _KERNEL_CANDIDATES:
        if _os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    raise AxiomError(
        f"Q4_K_M kernel source '{_KERNEL_FILENAME}' not found; looked in {list(_KERNEL_CANDIDATES)}"
    )


Q4KM_KERNEL_SRC: str = _load_kernel_src()

# The M3.6 LEADER strategy assignment (== the bench's rb2x2_assignments). rb_m/rb_n are inert
# declared holes (2×2 RB is structurally hardcoded as 4 accumulators) but are passed anyway to
# avoid the UnresolvedStrategyHole codegen backstop. a_block_size is PINNED at 512 (a HARD cache
# invariant: row_local*8+is must stay < 256).
Q4KM_LEADER_STRATEGY = {
    "rb_m": 2,
    "rb_n": 2,
    "tile_k": 16,
    "a_block_size": 512,
    "b_block_size": 512,
}


# ── Rust-backed reference helpers (re-exported; single source of truth) ─────────────
def q4km_f32_reference(q_bytes, x_f16_bits, m: int, n: int, n_blocks_per_row: int):
    """Call the Rust f32-accumulator Q4_K_M reference: returns ``(y_ref, abs_scale)``.

    ``x_f16_bits`` is the activation as f16 BIT PATTERNS (u16), row-major ``x[k*n+col]`` ([K,N]).
    Both returned lists are row-major (m × n) f32. Used by AT-2103 with
    :func:`q4km_combined_metric` to compute the FROZEN gate in Rust.
    """
    return _axiom_compute.q4km_f32_reference(
        bytes(q_bytes), list(x_f16_bits), int(m), int(n), int(n_blocks_per_row)
    )


def q4km_combined_metric(gpu, reference, abs_scale) -> float:
    """The FROZEN condition-aware combined gate (computed in Rust). ``<= 1e-3`` ⇒ pass."""
    return _axiom_compute.q4km_combined_metric(list(gpu), list(reference), list(abs_scale))


def q4km_raw_rel_diff(gpu, reference) -> float:
    """The RAW relative error (REPORTING ONLY — never the gate)."""
    return _axiom_compute.q4km_raw_rel_diff(list(gpu), list(reference))


# ── A faithful Q4_K quantizer (ergonomics; NOT on the AT-2103 critical path) ────────
def _get_scale_min_k4_pack(j: int, sc: int, m: int, scales: bytearray) -> None:
    """Inverse of ggml ``get_scale_min_k4`` — pack 6-bit (sc, m) for sub-block ``j`` (0..7)."""
    sc &= 0x3F
    m &= 0x3F
    if j < 4:
        scales[j] = (scales[j] & 0xC0) | sc
        scales[j + 4] = (scales[j + 4] & 0xC0) | m
    else:
        # low 4 bits of sc/m -> bytes[j+4]; high 2 bits -> packed into bytes[j-4]/bytes[j].
        scales[j + 4] = (scales[j + 4] & 0xF0) | (sc & 0x0F)
        scales[j + 4] = (scales[j + 4] & 0x0F) | ((m & 0x0F) << 4)
        scales[j - 4] = (scales[j - 4] & 0x3F) | (((sc >> 4) & 0x03) << 6)
        scales[j] = (scales[j] & 0x3F) | (((m >> 4) & 0x03) << 6)


def pack_q4km(weights, n_blocks_per_row: int) -> bytes:
    """Quantize an ``[M, K]`` f32 weight matrix to the ggml Q4_K_M 144-byte superblock layout.

    LOSSY (a faithful Q4_K quantizer): each 256-weight superblock gets a super-scale ``d`` and
    super-min ``dmin`` (f16) plus 8 sub-block 6-bit (sc, m) and 256 4-bit quants. This is provided
    for real-weight ergonomics; the AT-2103 CORRECTNESS gate does NOT use it (it uses raw RNG
    fixture bytes so q is byte-identical on the GPU and the oracle). The bytes ``pack_q4km``
    produces are read back EXACTLY by the Rust oracle's ``dequant_q4km_row_f32``.
    """
    import numpy as np

    w = np.asarray(weights, dtype=np.float32)
    if w.ndim != 2:
        raise AxiomError(f"pack_q4km: weights must be 2-D [M, K], got shape {w.shape}")
    m, k = w.shape
    if k != n_blocks_per_row * _Q4KM_SUPERBLOCK_ELEMS:
        raise AxiomError(
            f"pack_q4km: K={k} != n_blocks_per_row*256 = {n_blocks_per_row * 256}"
        )

    out = bytearray(m * n_blocks_per_row * 144)
    for row in range(m):
        for sb in range(n_blocks_per_row):
            base = row * n_blocks_per_row * 144 + sb * 144
            block = w[row, sb * 256:(sb + 1) * 256]
            # Per-sub-block (32 weights) min/max -> 6-bit scale + 6-bit min, then a super d/dmin.
            sub_sc = np.zeros(8, dtype=np.float32)
            sub_min = np.zeros(8, dtype=np.float32)
            for s in range(8):
                seg = block[s * 32:(s + 1) * 32]
                lo = float(seg.min())
                hi = float(seg.max())
                sub_min[s] = -lo if lo < 0 else 0.0  # store positive min offset
                rng = hi - lo
                sub_sc[s] = rng / 15.0 if rng > 0 else 0.0
            d = float(sub_sc.max()) / 63.0 if sub_sc.max() > 0 else 1.0
            dmin = float(sub_min.max()) / 63.0 if sub_min.max() > 0 else 1.0
            d = d if d != 0 else 1e-8
            dmin = dmin if dmin != 0 else 1e-8
            d16 = np.float16(d)
            dmin16 = np.float16(dmin)
            out[base:base + 2] = int(d16.view(np.uint16)).to_bytes(2, "little")
            out[base + 2:base + 4] = int(dmin16.view(np.uint16)).to_bytes(2, "little")
            scales = bytearray(12)
            sc6 = np.clip(np.round(sub_sc / float(d16)), 0, 63).astype(np.int32)
            m6 = np.clip(np.round(sub_min / float(dmin16)), 0, 63).astype(np.int32)
            for s in range(8):
                _get_scale_min_k4_pack(s, int(sc6[s]), int(m6[s]), scales)
            out[base + 4:base + 16] = bytes(scales)
            # Quantize the 256 weights to 4-bit nibbles using the per-sub-block dsc/dmm.
            qs = bytearray(128)
            for s in range(8):
                dsc = float(d16) * float(sc6[s])
                dmm = float(dmin16) * float(m6[s])
                seg = block[s * 32:(s + 1) * 32]
                if dsc <= 0:
                    nibs = np.zeros(32, dtype=np.int32)
                else:
                    nibs = np.clip(np.round((seg + dmm) / dsc), 0, 15).astype(np.int32)
                # ggml nibble layout: chunk = s//2, is_hi = s%2, byte qs[chunk*32 + l].
                chunk = s // 2
                is_hi = s % 2
                for l in range(32):
                    bi = chunk * 32 + l
                    if is_hi == 0:
                        qs[bi] = (qs[bi] & 0xF0) | (int(nibs[l]) & 0x0F)
                    else:
                        qs[bi] = (qs[bi] & 0x0F) | ((int(nibs[l]) & 0x0F) << 4)
            out[base + 16:base + 144] = bytes(qs)
    return bytes(out)


class Q4KMMatmul:
    """Q4_K_M dequant+coopmat matmul as a zero-copy torch op.

    Lifetime: compile ONCE (constructor); :meth:`upload_weights` registers the static q bytes;
    the shared session (q/x/C exportable buffers) is allocated on the FIRST :meth:`matmul` (when
    N is known from x) and REUSED across calls. A different (M, N, K) requires a NEW
    :class:`Q4KMMatmul` (the x/C view sizes are fixed per session).

    Zero-copy (per-call): x in / C out are direct exportable-buffer SSBOs — 0 staging, 0 copy
    (AT-2103c). q is upload-once (the static-weights carve-out).
    """

    def __init__(self, device=None, strategy: Optional[dict] = None):
        self._device = device
        self._strategy = dict(Q4KM_LEADER_STRATEGY if strategy is None else strategy)
        # Compile the M3.6 kernel WITH the full strategy assignment (load-bearing; strategy=None
        # would trip UnresolvedStrategyHole — AT-2103d).
        self._kernel = _compile_kernel(Q4KM_KERNEL_SRC, device=device, strategy=self._strategy)
        if self._kernel.buffer_count() != 3:
            raise AxiomError(
                f"Q4_K_M kernel must bind 3 SSBOs (q, x, C); got {self._kernel.buffer_count()}"
            )
        self._q_bytes: Optional[bytes] = None
        self._m: Optional[int] = None
        self._k: Optional[int] = None
        self._n_bpr: Optional[int] = None
        self._n: Optional[int] = None
        self._sess = None  # allocated on first matmul (N-dependent sizes)
        # M4.1p3: which path the most recent matmul() took ('zero_copy' | 'device_copy').
        self.report_path: Optional[str] = None

    # ── shape / scalar helpers ──────────────────────────────────────────────────────
    @property
    def kernel(self):
        return self._kernel

    @property
    def session(self):
        return self._sess

    @property
    def m(self) -> Optional[int]:
        """The registered M (weight rows) — fixed at :meth:`upload_weights` (R1)."""
        return self._m

    @property
    def k(self) -> Optional[int]:
        """The registered K (contraction dim) — fixed at :meth:`upload_weights` (R1)."""
        return self._k

    @property
    def n(self) -> Optional[int]:
        """The session N (activation cols) — fixed on the first matmul/x_view (R1)."""
        return self._n

    def scalar_layout(self):
        return self._kernel.scalar_layout()

    def upload_weights(self, q_bytes: bytes, m: int, k: int) -> None:
        """Register the static Q4_K_M weight bytes + (M, K). The shared q buffer is written ONCE.

        The actual session allocation + the single device-side q write happen on the first
        :meth:`matmul` (when N is known). q is NEVER re-uploaded across calls.
        """
        if k % 256 != 0:
            raise AxiomError(f"K must be a multiple of 256 (Q4_K_M superblock); got K={k}")
        if m % 32 != 0:
            raise AxiomError(f"M must be a multiple of 32 (coopmat 2x2 RB tile); got M={m}")
        n_bpr = k // 256
        expected = m * n_bpr * 144
        q = bytes(q_bytes)
        if len(q) != expected:
            raise AxiomError(
                f"upload_weights: q has {len(q)} bytes, expected m*n_bpr*144 = {expected}"
            )
        self._q_bytes = q
        self._m = int(m)
        self._k = int(k)
        self._n_bpr = int(n_bpr)

    def _ensure_session(self, n: int):
        import torch

        if self._q_bytes is None:
            raise AxiomError("matmul() called before upload_weights() — no weights uploaded")
        if n % 32 != 0:
            raise AxiomError(f"N must be a multiple of 32 (coopmat tile); got N={n}")
        if self._sess is None:
            self._n = int(n)
            q_size = len(self._q_bytes)
            x_size = 2 * self._k * n           # f16 activations [K, N]
            c_size = 4 * self._m * n           # f32 output [M, N]
            # 50 GB cap (CLAUDE.md hard rule 1).
            total = q_size + x_size + c_size
            if total >= 50 * 1024 * 1024 * 1024:
                raise AxiomError(
                    f"Q4_K_M buffers total {total} bytes exceeds the 50 GB cap "
                    f"(M={self._m} N={n} K={self._k})"
                )
            self._sess = self._kernel.allocate_shared(sizes=[q_size, x_size, c_size])
            # UPLOAD-ONCE: write the static q bytes into the shared q view (buffer 0) on stream S.
            # This is a one-time device-side copy into the shared view (NOT a per-call host copy,
            # NOT a dispatch-path vkCmdCopyBuffer).
            s = self._sess.stream
            q_view = self._sess.tensor(0, (q_size,), torch.uint8)
            q_host = torch.frombuffer(bytearray(self._q_bytes), dtype=torch.uint8)
            with torch.cuda.stream(s):
                q_view.copy_(q_host.to(device="cuda", non_blocking=True))
            s.synchronize()
        elif self._n != n:
            raise AxiomError(
                f"Q4KMMatmul session is fixed at N={self._n}; a different N={n} needs a NEW "
                "Q4KMMatmul (x/C view sizes are fixed per session)"
            )

    def x_view(self, n: int):
        """The shared x activation view [K, N] f16 — write activations DIRECTLY into this for the
        genuinely-zero-copy gated path (no per-call copy)."""
        import torch

        self._ensure_session(n)
        return self._sess.tensor(1, (self._k, n), torch.float16)

    def c_view(self, n: int):
        """The shared C output view [M, N] f32 (zero-copy)."""
        import torch

        self._ensure_session(n)
        return self._sess.tensor(2, (self._m, n), torch.float32)

    def matmul(self, x_f16, *, out=None):
        """Run C = dequant(q) @ x for an f16 activation ``x_f16`` of shape ``[K, N]``.

        GENUINELY zero-copy when ``x_f16`` IS the shared x view (:meth:`x_view`). Otherwise this
        is a CONVENIENCE overload: an already-[K,N] f16 CUDA tensor is device-to-device copied into
        the shared view (no host round-trip, but the copy is DISCLOSED — not the zero-copy gated
        path). A [N,K] activation is REJECTED (we never silently transpose-and-pretend-zero-copy).

        Returns the C output as a torch CUDA f32 tensor [M, N] (the shared view, or ``out`` if
        given). The result is valid on the session stream after the call returns.
        """
        import torch

        if x_f16.dtype != torch.float16:
            raise ZeroCopyUnavailable(
                f"x must be torch.float16 (the kernel reads readonly_buffer[f16]); got {x_f16.dtype}"
            )
        if not x_f16.is_cuda:
            raise ZeroCopyUnavailable("x must be a CUDA tensor (no host round-trip on the data path)")
        if x_f16.dim() != 2:
            raise AxiomError(f"x must be 2-D [K, N]; got shape {tuple(x_f16.shape)}")
        k_in, n = int(x_f16.shape[0]), int(x_f16.shape[1])
        if k_in != self._k:
            raise AxiomError(
                f"x has K={k_in} rows; expected [K, N] with K={self._k} "
                f"(a [N, K] activation needs a transpose/repack — NOT the zero-copy path)"
            )

        self._ensure_session(n)
        s = self._sess.stream
        xv = self._sess.tensor(1, (self._k, n), torch.float16)
        # If the caller handed us the shared view itself, no copy. Otherwise device-to-device
        # copy into the shared view (DISCLOSED convenience path — still no host round-trip).
        if x_f16.data_ptr() == xv.data_ptr():
            # GENUINE zero-copy (no-copy) path. data_ptr equality alone is NOT enough: a
            # same-storage NON-CONTIGUOUS view (e.g. x_view(n).t() at a square shape) shares the
            # data_ptr and can pass shape[0]==K, yet its logical [K,N] layout does NOT match the
            # underlying contiguous storage the kernel reads as x[k*N+col] — that would SILENTLY
            # reinterpret memory (the silent-reinterpret class the project forbids, W1). A shared
            # view used for zero-copy MUST be the contiguous [K, N] buffer; reject otherwise rather
            # than fall back to a copy (the caller explicitly asked for the zero-copy gated path).
            if not x_f16.is_contiguous() or tuple(x_f16.shape) != (self._k, n):
                raise ZeroCopyUnavailable(
                    "zero-copy x must be the CONTIGUOUS shared [K, N] view "
                    f"(K={self._k}, N={n}); got shape {tuple(x_f16.shape)} "
                    f"contiguous={x_f16.is_contiguous()}. A non-contiguous/transposed same-storage "
                    "view would silently reinterpret the underlying buffer — write activations "
                    "directly into x_view(n) (do not pass a transposed/strided view)."
                )
            self.report_path = "zero_copy"
        else:
            with torch.cuda.stream(s):
                xv.copy_(x_f16.contiguous())
            self.report_path = "device_copy"

        n_bpr = self._k // 256
        pc = self._kernel.assemble_push_constants(
            {"M": self._m, "N": n, "K": self._k, "n_blocks_per_row": n_bpr}
        )
        workgroups = (n // 32, self._m // 32, 1)
        self._sess.run(workgroups=workgroups, push_constants=pc)

        cv = self._sess.tensor(2, (self._m, n), torch.float32)
        if out is not None:
            with torch.cuda.stream(s):
                out.copy_(cv)
            return out
        return cv

    def close(self):
        if self._sess is not None:
            self._sess.close()
            self._sess = None
        if self._kernel is not None:
            self._kernel.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def q4km_matmul(q_bytes: bytes, x_f16, m: int, k: int, *, device=None, strategy=None):
    """One-shot convenience: compile, upload q, run one matmul, and tear down.

    For repeated calls, construct a :class:`Q4KMMatmul` and reuse it (the weights stay resident).
    Returns a torch CUDA f32 tensor [M, N] (cloned off the shared view so it survives teardown).
    """
    op = Q4KMMatmul(device=device, strategy=strategy)
    try:
        op.upload_weights(q_bytes, m, k)
        c = op.matmul(x_f16)
        result = c.clone()
        op.session.stream.synchronize()
        return result
    finally:
        op.close()
