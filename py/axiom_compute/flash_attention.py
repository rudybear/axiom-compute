"""M4.1 Phase 4 — the merged FlashAttention kernel as a zero-copy torch op.

Runs the M3.2c-perf coopmat-accelerated FlashAttention-2 leader kernel
(``examples/flash_attention_coopmat.axc`` — coopmat S=QKᵀ + scalar online-softmax with the
output accumulator held in shared) directly on torch CUDA tensors with NO host copies on the
per-call data path. This is the SECOND application of the M4.1 Phase-3 zero-copy SharedSession
frontend (after the Q4_K_M matmul), completing the dual-kernel PyTorch adoption story.

BUFFER PLAN (the honest 4-buffer zero-copy split — SIMPLER than q4km: NO upload-once weights):
  - Q  [seq, head_dim] f32  = ZERO-COPY per-call exportable shared buffer (readonly_buffer[f32]).
  - K  [seq, head_dim] f32  = ZERO-COPY per-call exportable shared buffer (readonly_buffer[f32]).
  - V  [seq, head_dim] f32  = ZERO-COPY per-call exportable shared buffer (readonly_buffer[f32]).
  - O  [seq, head_dim] f32  = ZERO-COPY per-call exportable shared buffer (buffer[f32]).
  All four are per-call data (FlashAttention has no static weights); per-call Q/K/V/O are
  GENUINELY staging-free: 0 staging buffers, 0 ``vkCmdCopyBuffer`` (AT-2125).

dtype DECISION (q/k/v f32 — NOT f16): the coopmat kernel's Q/K/V params are
  ``readonly_buffer[f32]`` (flash_attention_coopmat.axc:64-66); the kernel itself does the
  f32→f16 cast INTERNALLY via ``f32_to_f16`` when staging Q/K into the coopmat f16 tiles
  (:97,:127), and keeps V f32 for the scalar P·V. So the zero-copy buffers are f32 and the op
  requires ``q/k/v.dtype == torch.float32`` (UNLIKE q4km, which required f16 x). O is f32.

LAYOUT MATCH (load-bearing): the kernel reads Q/K/V/O as CONTIGUOUS ROW-MAJOR ``[seq, head_dim]``
  f32 (``Q[g_row*head_dim + d]`` :97, ``K[g_krow*head_dim + d_full]`` :127, V :137, O :242). A
  torch contiguous ``[seq, head_dim]`` f32 tensor is the EXACT zero-copy layout — no
  transpose/repack. A non-contiguous/transposed same-storage view is REJECTED on the zero-copy
  path (the W1 silent-reinterpret guard), applied to ALL THREE inputs q/k/v.

PUSH CONSTANTS: ``{seq_len, head_dim, inv_sqrt_d}``. ``inv_sqrt_d = 1.0/sqrt(head_dim)`` is an
  f32 scalar computed IN PYTHON (the wrapper does NOT trust the caller and does NOT pre-scale Q;
  the kernel applies the 1/sqrt(d) scale internally at :180,:199). The f32 packing rides the
  M4.1p4 ``assemble_push_constants`` fix (``scalar_layout()`` reports ``ty_str=='f32'`` for the
  inv_sqrt_d slot and the packer ``struct.pack('<f', ·)``s it — NOT ``int(0.125)==0``).

CORRECTNESS ORACLE: torch SDPA f32 with ``SDPBackend.MATH`` forced (real f32 softmax,
  apples-to-apples with the kernel's GLSL.std.450 ``exp()``), FROZEN combined abs/rel 1e-3
  (see ops in ``attention_ops`` / the test suite). Attention outputs are CONVEX COMBINATIONS of
  V (softmax weights in [0,1] summing to 1) — not cancellation-prone — so a combined gate holds.

HONEST FRAMING: NO beat-SDPA / beat-cuBLAS claim. The attention kernel is NOT competitive yet
  (coopmat-PV is deferred); the value is a SECOND kernel callable from PyTorch as
  ``torch.ops.axiom.flash_attention`` that composes under ``torch.compile``, zero-copy on the
  eager path, correct within FROZEN 1e-3.
"""

from __future__ import annotations

import math
import os as _os
from typing import Optional

from . import compile_kernel as _compile_kernel
from . import AxiomError, ZeroCopyUnavailable

# The coopmat-leader CORE pin (flash_attention_coopmat.axc:24) and the 16-key block.
FA_COOPMAT_HEAD_DIM: int = 64
FA_SEQ_MULTIPLE: int = 16

_KERNEL_FILENAMES = {
    "coopmat": "flash_attention_coopmat.axc",
    "scalar": "flash_attention_exp.axc",
}


def _kernel_candidates(filename: str):
    """Candidate locations: (1) a copy bundled INSIDE the installed package (kernels/...),
    (2) the repo's examples/ dir (3 levels up — the editable/source-tree layout)."""
    here = _os.path.dirname(_os.path.abspath(__file__))  # .../py/axiom_compute
    repo_root = _os.path.dirname(_os.path.dirname(here))  # .../py -> repo root
    return (
        _os.path.join(here, "kernels", filename),
        _os.path.join(repo_root, "examples", filename),
    )


def _load_kernel_src(kernel: str) -> str:
    filename = _KERNEL_FILENAMES.get(kernel)
    if filename is None:
        raise AxiomError(
            f"flash_attention: unknown kernel {kernel!r}; choose 'coopmat' (head_dim==64) "
            "or 'scalar' (head_dim<=64 fallback)"
        )
    for path in _kernel_candidates(filename):
        if _os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    raise AxiomError(
        f"FlashAttention kernel source '{filename}' not found; looked in "
        f"{list(_kernel_candidates(filename))}"
    )


class FlashAttention:
    """The merged FlashAttention kernel as a 4-buffer zero-copy torch op.

    Lifetime: compile ONCE (constructor); the shared session (Q/K/V/O exportable buffers) is
    allocated on the FIRST :meth:`attention` / view access (when seq is known) and REUSED across
    calls. A different (seq, head_dim) requires a NEW :class:`FlashAttention` (the view sizes are
    fixed per session). There is NO upload-once step — Q/K/V are ALL per-call data.

    Constraints (coopmat leader, validated host-side in R1): ``head_dim == 64`` EXACTLY,
    ``seq % 16 == 0``, ``@workgroup(32,1,1)`` (one subgroup), dispatch ``(seq//16, 1, 1)``. The
    ``kernel='scalar'`` fallback relaxes head_dim to ``<= 64`` (``@workgroup(1,1,1)``, dispatch
    ``(seq, 1, 1)``) but is ~28× slower.

    Zero-copy (per-call): Q/K/V in + O out are direct exportable-buffer SSBOs — 0 staging, 0
    copy (AT-2125) when the inputs ARE the shared views.
    """

    def __init__(self, device=None, kernel: str = "coopmat"):
        self._device = device
        self._kernel_name = str(kernel)
        # No strategy holes in either FA kernel → strategy=None.
        self._kernel = _compile_kernel(_load_kernel_src(self._kernel_name), device=device)
        if self._kernel.buffer_count() != 4:
            raise AxiomError(
                f"FlashAttention kernel must bind 4 SSBOs (Q, K, V, O); got "
                f"{self._kernel.buffer_count()}"
            )
        self._seq: Optional[int] = None
        self._head_dim: Optional[int] = None
        self._sess = None  # allocated on the first attention/view (seq-dependent sizes)
        # Which path the most recent attention() took ('zero_copy' | 'device_copy').
        self.report_path: Optional[str] = None

    # ── properties ───────────────────────────────────────────────────────────────────
    @property
    def kernel(self):
        return self._kernel

    @property
    def kernel_name(self) -> str:
        return self._kernel_name

    @property
    def session(self):
        return self._sess

    @property
    def seq(self) -> Optional[int]:
        return self._seq

    @property
    def head_dim(self) -> Optional[int]:
        return self._head_dim

    def scalar_layout(self):
        return self._kernel.scalar_layout()

    # ── host preflight (the coopmat fa2_coopmat_preflight analogue) ────────────────────
    def _preflight(self, seq_len: int, head_dim: int) -> None:
        if self._kernel_name == "coopmat":
            if head_dim != FA_COOPMAT_HEAD_DIM:
                raise AxiomError(
                    f"FlashAttention(coopmat): head_dim must be {FA_COOPMAT_HEAD_DIM} exactly "
                    f"(the coopmat-leader CORE pin); got head_dim={head_dim}. Use "
                    "kernel='scalar' for head_dim<=64."
                )
        else:  # scalar fallback: head_dim <= 64.
            if not (0 < head_dim <= FA_COOPMAT_HEAD_DIM):
                raise AxiomError(
                    f"FlashAttention(scalar): head_dim must be in 1..{FA_COOPMAT_HEAD_DIM}; "
                    f"got head_dim={head_dim}"
                )
        if seq_len <= 0 or seq_len % FA_SEQ_MULTIPLE != 0:
            raise AxiomError(
                f"FlashAttention: seq_len must be a positive multiple of {FA_SEQ_MULTIPLE} "
                f"(the 16-key block); got seq_len={seq_len}"
            )

    def _ensure_session(self, seq_len: int, head_dim: int):
        import torch

        self._preflight(seq_len, head_dim)
        if self._sess is None:
            self._seq = int(seq_len)
            self._head_dim = int(head_dim)
            buf = 4 * seq_len * head_dim  # f32 [seq, head_dim] per buffer
            # 50 GB cap (CLAUDE.md hard rule 1): sum ALL FOUR buffers.
            total = 4 * buf
            if total >= 50 * 1024 * 1024 * 1024:
                raise AxiomError(
                    f"FlashAttention buffers total {total} bytes exceeds the 50 GB cap "
                    f"(seq_len={seq_len} head_dim={head_dim})"
                )
            # Four per-call zero-copy buffers {Q, K, V, O} (NO upload-once).
            self._sess = self._kernel.allocate_shared(sizes=[buf, buf, buf, buf])
        elif self._seq != seq_len or self._head_dim != head_dim:
            raise AxiomError(
                f"FlashAttention session is fixed at (seq={self._seq}, head_dim="
                f"{self._head_dim}); a different (seq={seq_len}, head_dim={head_dim}) needs a "
                "NEW FlashAttention (the Q/K/V/O view sizes are fixed per session)"
            )

    # ── the shared zero-copy views (write activations DIRECTLY into these) ─────────────
    def q_view(self, seq_len: int, head_dim: int):
        import torch

        self._ensure_session(seq_len, head_dim)
        return self._sess.tensor(0, (seq_len, head_dim), torch.float32)

    def k_view(self, seq_len: int, head_dim: int):
        import torch

        self._ensure_session(seq_len, head_dim)
        return self._sess.tensor(1, (seq_len, head_dim), torch.float32)

    def v_view(self, seq_len: int, head_dim: int):
        import torch

        self._ensure_session(seq_len, head_dim)
        return self._sess.tensor(2, (seq_len, head_dim), torch.float32)

    def o_view(self, seq_len: int, head_dim: int):
        import torch

        self._ensure_session(seq_len, head_dim)
        return self._sess.tensor(3, (seq_len, head_dim), torch.float32)

    # ── the W1 per-input zero-copy contiguity/dtype/device/shape guard ─────────────────
    def _validate_input(self, name: str, t, seq_len: int, head_dim: int) -> None:
        import torch

        if not isinstance(t, torch.Tensor):
            raise AxiomError(f"flash_attention: {name} must be a torch.Tensor")
        if not t.is_cuda:
            raise ZeroCopyUnavailable(
                f"flash_attention: {name} must be a CUDA tensor (no host round-trip on the "
                "data path)"
            )
        if t.dtype != torch.float32:
            raise AxiomError(
                f"flash_attention: {name} must be float32 (the kernel reads "
                f"readonly_buffer[f32]); got {t.dtype}"
            )
        if t.dim() != 2 or tuple(t.shape) != (seq_len, head_dim):
            raise AxiomError(
                f"flash_attention: {name} must be 2-D [seq, head_dim] = "
                f"[{seq_len}, {head_dim}]; got shape {tuple(t.shape)}"
            )

    def _copy_into_view(self, name: str, t, view, seq_len: int, head_dim: int):
        """W1: if the caller handed us the shared view itself, no copy (genuine zero-copy);
        else device-to-device copy into the shared view (disclosed convenience path — no host
        round-trip). A non-contiguous/transposed same-storage view on the zero-copy path is
        REJECTED (never silently reinterpret the buffer)."""
        import torch

        if t.data_ptr() == view.data_ptr():
            # data_ptr equality alone is NOT enough: a same-storage NON-CONTIGUOUS view shares
            # the data_ptr yet its logical [seq,d] layout need not match the contiguous storage
            # the kernel reads — that would SILENTLY reinterpret memory (W1). Require the
            # contiguous [seq, head_dim] view; reject otherwise (the caller asked for zero-copy).
            if not t.is_contiguous() or tuple(t.shape) != (seq_len, head_dim):
                raise ZeroCopyUnavailable(
                    f"flash_attention: zero-copy {name} must be the CONTIGUOUS shared "
                    f"[seq, head_dim]=[{seq_len}, {head_dim}] view; got shape "
                    f"{tuple(t.shape)} contiguous={t.is_contiguous()}. A non-contiguous/"
                    "transposed same-storage view would silently reinterpret the underlying "
                    "buffer — write activations directly into the *_view(seq, head_dim)."
                )
            return True  # genuine zero-copy (no copy)
        # Disclosed device-to-device copy into the shared view (still no host round-trip).
        with torch.cuda.stream(self._sess.stream):
            view.copy_(t.contiguous())
        return False

    def attention(self, q, k, v, *, out=None):
        """Run O = softmax(Q·Kᵀ / sqrt(d)) · V for f32 ``[seq, head_dim]`` Q/K/V.

        GENUINELY zero-copy when q/k/v ARE the shared views (:meth:`q_view`/:meth:`k_view`/
        :meth:`v_view`). Otherwise each input is device-to-device copied into its shared view
        (no host round-trip, but the copy is DISCLOSED — not the zero-copy gated path). A
        non-contiguous/transposed same-storage view is REJECTED for any input.

        Returns the O output as a torch CUDA f32 tensor ``[seq, head_dim]`` (the shared view, or
        ``out`` if given). Valid on the session stream after the call returns.
        """
        import torch

        if q.dim() != 2:
            raise AxiomError(
                f"flash_attention: q must be 2-D [seq, head_dim]; got {tuple(q.shape)}"
            )
        seq_len, head_dim = int(q.shape[0]), int(q.shape[1])
        self._ensure_session(seq_len, head_dim)

        qv = self._sess.tensor(0, (seq_len, head_dim), torch.float32)
        kv = self._sess.tensor(1, (seq_len, head_dim), torch.float32)
        vv = self._sess.tensor(2, (seq_len, head_dim), torch.float32)

        # W1: validate + (no-)copy ALL THREE inputs into their views.
        self._validate_input("q", q, seq_len, head_dim)
        self._validate_input("k", k, seq_len, head_dim)
        self._validate_input("v", v, seq_len, head_dim)
        zq = self._copy_into_view("q", q, qv, seq_len, head_dim)
        zk = self._copy_into_view("k", k, kv, seq_len, head_dim)
        zv = self._copy_into_view("v", v, vv, seq_len, head_dim)
        self.report_path = "zero_copy" if (zq and zk and zv) else "device_copy"

        inv_sqrt_d = 1.0 / math.sqrt(head_dim)
        pc = self._kernel.assemble_push_constants(
            {"seq_len": seq_len, "head_dim": head_dim, "inv_sqrt_d": inv_sqrt_d}
        )
        if self._kernel_name == "coopmat":
            workgroups = (seq_len // 16, 1, 1)
        else:  # scalar: one workgroup per query row.
            workgroups = (seq_len, 1, 1)
        self._sess.run(workgroups=workgroups, push_constants=pc)

        ov = self._sess.tensor(3, (seq_len, head_dim), torch.float32)
        if out is not None:
            with torch.cuda.stream(self._sess.stream):
                out.copy_(ov)
            return out
        return ov

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


def flash_attention(q, k, v, *, device=None, kernel: str = "coopmat"):
    """One-shot convenience: compile, run one attention, and tear down.

    For repeated calls, construct a :class:`FlashAttention` (or use the
    ``torch.ops.axiom.flash_attention`` registered op, which caches the session). Returns a
    torch CUDA f32 tensor ``[seq, head_dim]`` (cloned off the shared view so it survives
    teardown).
    """
    op = FlashAttention(device=device, kernel=kernel)
    try:
        o = op.attention(q, k, v)
        result = o.clone()
        op.session.stream.synchronize()
        return result
    finally:
        op.close()
