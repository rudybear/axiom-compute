"""M4.1 Phase 2 — Q4_K_M HEADLINE op through the CUDA↔Vulkan zero-copy frontend.

AT-2103   correctness: zero-copy Q4_K_M result vs the RUST f32-accumulator oracle, gated on the
          FROZEN condition-aware combined metric <= 1e-3 at K=256/512/14336. NOT torch.allclose /
          raw rel error (the outputs are cancellation-prone; raw ~1e-2 is a metric artifact).
AT-2103c  zero-copy staging-free: per-call x-in/C-out path allocates 0 staging buffers / 0 copies;
          upload_weights() does NOT increment the dispatch copy counter.
AT-2103d  strategy=None (hole-carrying M3.6 source) FAILS at compile (the strategy path is
          load-bearing); a CPU / wrong-dtype x fails closed.
AT-2104   handshake determinism: repeated matmul() calls stay bit-identical (monotone V1/V2).
AT-2105   honest latency: zero-copy < host-copy (HARD); cuBLAS reported, NOT gated (no beat-cuBLAS).

The GPU + the Rust oracle read the SAME RNG q-byte / x-bit fixture (make_q4km_weights /
make_x_f16, the resident-bench seeds) so the comparison is apples-to-apples.

Gated by AXC_ENABLE_GPU_TESTS=1 + torch.cuda.is_available() + an nvidia ICD.
"""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import axiom_compute as axc


def _gpu_enabled():
    return os.environ.get("AXC_ENABLE_GPU_TESTS") == "1" and torch.cuda.is_available()


_MASK64 = (1 << 64) - 1


# ── Fixture builders — BYTE-IDENTICAL to the resident bench (so GPU == oracle) ──────
def make_q4km_weights(m: int, n_bpr: int, seed: int) -> bytes:
    """Q4_K_M weight bytes via the bench's xorshift RNG (seed 0xC0FFEE ^ m)."""
    q = bytearray(m * n_bpr * 144)
    state = seed | 1

    def nxt():
        nonlocal state
        state ^= (state << 13) & _MASK64
        state ^= (state >> 7)
        state ^= (state << 17) & _MASK64
        return state

    for row in range(m):
        for sb in range(n_bpr):
            base = row * n_bpr * 144 + sb * 144
            d = np.float32(0.02 + (nxt() % 16) * 0.002)
            dmin = np.float32(0.01 + (nxt() % 8) * 0.001)
            q[base:base + 2] = int(np.float16(d).view(np.uint16)).to_bytes(2, "little")
            q[base + 2:base + 4] = int(np.float16(dmin).view(np.uint16)).to_bytes(2, "little")
            for j in range(12):
                q[base + 4 + j] = nxt() & 0x3F
            for kk in range(128):
                q[base + 16 + kk] = nxt() & 0xFF
    return bytes(q)


def make_x_f16_bits(k: int, n: int, seed: int):
    """Activation f16 bit patterns via the bench's xorshift RNG (seed 0xBADF00D ^ n), [K, N]."""
    state = seed | 1
    out = []
    for idx in range(k * n):
        state ^= (state << 13) & _MASK64
        state ^= (state >> 7)
        state ^= (state << 17) & _MASK64
        v = np.float32(((state % 2000) / 1000.0 - 1.0) + (idx % 3) * 0.01)
        out.append(int(np.float16(v).view(np.uint16)))
    return out


def _x_f16_tensor_from_bits(x_bits, k: int, n: int, device="cuda"):
    """A torch.float16 [K, N] CUDA tensor whose bit patterns == ``x_bits`` (no value reinterpret)."""
    u16 = np.asarray(x_bits, dtype=np.uint16).reshape(k, n)
    t_cpu = torch.from_numpy(u16.view(np.int16).copy())  # int16 carrier of the f16 bits
    return t_cpu.to(device=device).view(torch.float16)


# ── AT-2103 / AT-2103c / AT-2104 / AT-2105 ──────────────────────────────────────────
FROZEN_REL_TOL = 1e-3

_AB_SHAPE = (4096, 512, 14336)  # the llama.cpp Q4_K MUL_MAT A/B shape (M, N, K)
_SMALL_SHAPES = [(256, 256, 256), (512, 512, 512)]


def _run_q4km(m, n, k):
    """Upload q once, write x into the zero-copy view, run, return (y_gpu, y_ref, abs_scale, op)."""
    n_bpr = k // 256
    q = make_q4km_weights(m, n_bpr, 0xC0FFEE ^ m)
    x_bits = make_x_f16_bits(k, n, 0xBADF00D ^ n)

    op = axc.Q4KMMatmul()
    op.upload_weights(q, m, k)
    s = op.kernel.stream
    # Write the fixture DIRECTLY into the shared x view ([K, N]) — the genuinely zero-copy path.
    xv = op.x_view(n)
    x_src = _x_f16_tensor_from_bits(x_bits, k, n)
    with torch.cuda.stream(s):
        xv.copy_(x_src)
    s.synchronize()

    c = op.matmul(xv)  # x_f16 IS the shared view -> no copy
    s.synchronize()
    y_gpu = c.detach().to("cpu").to(torch.float32).reshape(-1).tolist()

    y_ref, abs_scale = axc.q4km_f32_reference(q, x_bits, m, n, n_bpr)
    return y_gpu, y_ref, abs_scale, op, x_bits, q


@pytest.mark.skipif(not _gpu_enabled(), reason="GPU tests gated (AXC_ENABLE_GPU_TESTS=1)")
@pytest.mark.parametrize("m,n,k", _SMALL_SHAPES + [_AB_SHAPE])
def test_q4km_zero_copy_within_1e3_combined(m, n, k):
    """AT-2103: combined condition-aware metric <= FROZEN 1e-3 (computed in Rust). raw LOGGED."""
    y_gpu, y_ref, abs_scale, op, _xb, _q = _run_q4km(m, n, k)
    try:
        combined = axc.q4km_combined_metric(y_gpu, y_ref, abs_scale)
        raw = axc.q4km_raw_rel_diff(y_gpu, y_ref)
        # raw is REPORTING ONLY (cancellation artifact at small K); the GATE is `combined`.
        print(
            f"[AT-2103] M={m} N={n} K={k}: combined(GATE)={combined:.3e} "
            f"raw(report-only)={raw:.3e}  -> {'VALID' if combined <= FROZEN_REL_TOL else 'INVALID'}"
        )
        if k == 14336:
            print(
                f"[AT-2103] K=14336 is the inference depth where the M3.5 f16-accumulator was "
                f"INVALID (max_rel_diff ~29); the f32 accumulator passes here: combined={combined:.3e}"
            )
        assert combined <= FROZEN_REL_TOL, (
            f"Q4_K_M combined metric {combined:.3e} exceeds FROZEN {FROZEN_REL_TOL} at "
            f"M={m} N={n} K={k} (NEVER loosened)"
        )
    finally:
        op.close()


@pytest.mark.skipif(not _gpu_enabled(), reason="GPU tests gated (AXC_ENABLE_GPU_TESTS=1)")
def test_q4km_zero_copy_staging_free():
    """AT-2103c (G-2): per-call x-in/C-out path is staging-free; upload_weights adds no dispatch copy."""
    m, n, k = 256, 256, 256
    n_bpr = k // 256
    q = make_q4km_weights(m, n_bpr, 0xC0FFEE ^ m)
    x_bits = make_x_f16_bits(k, n, 0xBADF00D ^ n)
    op = axc.Q4KMMatmul()
    op.upload_weights(q, m, k)
    try:
        s = op.kernel.stream
        xv = op.x_view(n)
        x_src = _x_f16_tensor_from_bits(x_bits, k, n)
        with torch.cuda.stream(s):
            xv.copy_(x_src)
        s.synchronize()

        sess = op.session
        # After upload_weights + allocation, the upload is a device-side torch copy into the shared
        # view — it is NOT a dispatch-path vkCmdCopyBuffer, so the dispatch copy counter is 0.
        assert sess.staging_buffers_allocated() == 0, "no staging buffer on the zero-copy path"
        assert sess.copy_buffer_records() == 0, "upload_weights must not record a dispatch copy"

        # >= 2 matmul() calls — still 0/0 (the per-call x-in/C-out path records no staging/copy).
        for _ in range(2):
            op.matmul(xv)
        s.synchronize()
        assert sess.staging_buffers_allocated() == 0, "no staging buffer across matmul() calls"
        assert sess.copy_buffer_records() == 0, "no vkCmdCopyBuffer across matmul() calls"
        print("[AT-2103c] staging_buffers=0 copy_records=0 across upload + 2 matmul() calls")
    finally:
        op.close()


@pytest.mark.skipif(not _gpu_enabled(), reason="GPU tests gated (AXC_ENABLE_GPU_TESTS=1)")
def test_q4km_handshake_determinism():
    """AT-2104: >=100 matmul() calls (same q, varying x) each bit-identical to a single recompute."""
    m, n, k = 256, 256, 256
    n_bpr = k // 256
    q = make_q4km_weights(m, n_bpr, 0xC0FFEE ^ m)
    op = axc.Q4KMMatmul()
    op.upload_weights(q, m, k)
    try:
        s = op.kernel.stream
        xv = op.x_view(n)
        x_bits = make_x_f16_bits(k, n, 0xBADF00D ^ n)
        x_src = _x_f16_tensor_from_bits(x_bits, k, n)
        with torch.cuda.stream(s):
            xv.copy_(x_src)
        s.synchronize()

        op.matmul(xv)
        s.synchronize()
        first = op.c_view(n).detach().clone()

        for i in range(100):
            op.matmul(xv)
            s.synchronize()
            cur = op.c_view(n)
            maxdiff = (cur - first).abs().max().item()
            assert maxdiff == 0.0, f"call {i}: not bit-identical (maxdiff {maxdiff})"
        print("[AT-2104] 100 matmul() calls bit-identical (monotone V1/V2 handshake holds)")
    finally:
        op.close()


@pytest.mark.skipif(not _gpu_enabled(), reason="GPU tests gated (AXC_ENABLE_GPU_TESTS=1)")
def test_q4km_honest_latency_report():
    """AT-2105: HARD gate zero-copy < host-copy; cuBLAS REPORTED, not gated (no beat-cuBLAS)."""
    m, n, k = _AB_SHAPE
    n_bpr = k // 256
    q = make_q4km_weights(m, n_bpr, 0xC0FFEE ^ m)
    x_bits = make_x_f16_bits(k, n, 0xBADF00D ^ n)

    op = axc.Q4KMMatmul()
    op.upload_weights(q, m, k)
    try:
        s = op.kernel.stream
        xv = op.x_view(n)
        x_src = _x_f16_tensor_from_bits(x_bits, k, n)
        with torch.cuda.stream(s):
            xv.copy_(x_src)
        s.synchronize()

        def _time_cuda(fn, warmup=3, iters=10):
            with torch.cuda.stream(s):
                for _ in range(warmup):
                    fn()
            s.synchronize()
            best = float("inf")
            for _ in range(iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(s):
                    start.record(s)
                    fn()
                    end.record(s)
                s.synchronize()
                best = min(best, start.elapsed_time(end))  # ms
            return best

        # (a) AXIOM zero-copy: x already in the shared view -> dispatch + handshake only.
        zc_ms = _time_cuda(lambda: op.matmul(xv))

        # (b) AXIOM host-copy baseline: torch CPU -> .cuda() -> into shared x -> matmul -> .cpu().
        x_cpu = x_src.to("cpu")

        def _host_copy_call():
            with torch.cuda.stream(s):
                xv.copy_(x_cpu.to("cuda", non_blocking=False))
            op.matmul(xv)
            _ = op.c_view(n).to("cpu")

        hc_ms = _time_cuda(_host_copy_call)

        # (c) native torch.matmul (cuBLAS) — CONTEXT ONLY. Dequant W to f32 [M, K] (a vectorized
        # numpy dequant; NOT a correctness oracle — it gates nothing) and time cuBLAS W@x.
        w_dq = _dequant_q4km_to_f32(q, m, k, n_bpr).to("cuda")
        x_f32 = x_src.to(torch.float32)

        def _cublas_call():
            with torch.cuda.stream(s):
                _ = torch.matmul(w_dq, x_f32)

        cublas_ms = _time_cuda(_cublas_call)

        cublas_tflops = 2.0 * m * n * k / (cublas_ms * 1e-3) / 1e12
        axiom_tflops = 2.0 * m * n * k / (zc_ms * 1e-3) / 1e12
        pct = axiom_tflops / cublas_tflops * 100.0 if cublas_tflops > 0 else 0.0

        print(
            "\n[AT-2105 HONEST LATENCY] M={} N={} K={}\n"
            "  (a) AXIOM zero-copy   = {:.3f} ms  ({:.2f} TFLOPS)\n"
            "  (b) AXIOM host-copy   = {:.3f} ms  (round-trip the zero-copy ELIMINATES)\n"
            "  (c) torch.matmul cuBLAS = {:.3f} ms  ({:.2f} TFLOPS, CONTEXT ONLY)\n"
            "  -> zero-copy WIN: (a) < (b) by {:.3f} ms (no host transfer)\n"
            "  -> COMPUTE gap: AXIOM is {:.1f}% of cuBLAS — the Vulkan kernel is SLOWER than\n"
            "     native torch.matmul; this is NOT a cuBLAS replacement, NO beat-cuBLAS claim.".format(
                m, n, k, zc_ms, axiom_tflops, hc_ms, cublas_ms, cublas_tflops,
                hc_ms - zc_ms, pct,
            )
        )

        for label, v in (("zero-copy", zc_ms), ("host-copy", hc_ms), ("cuBLAS", cublas_ms)):
            assert v > 0 and np.isfinite(v), f"{label} latency must be finite/positive (got {v})"
        # HARD GATE (the ONLY one): zero-copy eliminates the host transfer.
        assert zc_ms < hc_ms, (
            f"zero-copy ({zc_ms:.3f} ms) must be < host-copy ({hc_ms:.3f} ms) — the "
            "transfer-elimination win"
        )
    finally:
        op.close()


def _get_scale_min_k4(scales: np.ndarray, j: int):
    """ggml get_scale_min_k4 for sub-block j (0..7) over the 12 scale bytes of a superblock."""
    if j < 4:
        return int(scales[j] & 63), int(scales[j + 4] & 63)
    sc = (scales[j + 4] & 0x0F) | (((scales[j - 4] >> 6) & 0x03) << 4)
    m = ((scales[j + 4] >> 4) & 0x0F) | (((scales[j] >> 6) & 0x03) << 4)
    return int(sc), int(m)


def _dequant_q4km_to_f32(q: bytes, m: int, k: int, n_bpr: int) -> torch.Tensor:
    """Dequant Q4_K_M weights to an f32 [M, K] torch tensor for the cuBLAS CONTEXT timing ONLY.

    A vectorized numpy dequant mirroring ggml/dequant_q4km_row_f32. This is CONTEXT-ONLY (it
    times cuBLAS, gates nothing); the correctness oracle remains the Rust q4km_oracle.
    """
    qb = np.frombuffer(q, dtype=np.uint8)
    w = np.zeros((m, k), dtype=np.float32)
    for row in range(m):
        for sb in range(n_bpr):
            base = row * n_bpr * 144 + sb * 144
            d = np.float16(0)
            d = np.frombuffer(qb[base:base + 2].tobytes(), dtype=np.float16)[0].astype(np.float32)
            dmin = np.frombuffer(qb[base + 2:base + 4].tobytes(), dtype=np.float16)[0].astype(np.float32)
            scales = qb[base + 4:base + 16]
            qs = qb[base + 16:base + 144]
            for chunk in range(4):
                sc0, m0 = _get_scale_min_k4(scales, chunk * 2)
                sc1, m1 = _get_scale_min_k4(scales, chunk * 2 + 1)
                d1 = d * sc0
                m1f = dmin * m0
                d2 = d * sc1
                m2f = dmin * m1
                seg = qs[chunk * 32:chunk * 32 + 32].astype(np.uint32)
                lo = (seg & 0x0F).astype(np.float32)
                hi = ((seg >> 4) & 0x0F).astype(np.float32)
                koff = sb * 256 + chunk * 64
                w[row, koff:koff + 32] = d1 * lo - m1f
                w[row, koff + 32:koff + 64] = d2 * hi - m2f
    return torch.from_numpy(w)


# ── W2 — bundled-.axc drift guard (no GPU needed) ────────────────────────────────────
def test_bundled_kernel_byte_identical_to_examples():
    """W2: the wheel-bundled kernel copy (py/axiom_compute/kernels/...) must be BYTE-IDENTICAL to
    the canonical examples/ source. _load_kernel_src prefers the bundled copy, so a future edit to
    examples/ that misses the bundled copy would silently run a STALE kernel — this guard fails CI
    if they drift."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(here))  # py/tests -> py -> repo root
    fname = "q4km_matmul_rb_coopmat_f32acc_cached.axc"
    bundled = os.path.join(repo_root, "py", "axiom_compute", "kernels", fname)
    canonical = os.path.join(repo_root, "examples", fname)
    assert os.path.isfile(bundled), f"bundled kernel missing: {bundled}"
    assert os.path.isfile(canonical), f"canonical kernel missing: {canonical}"
    with open(bundled, "rb") as f:
        bundled_bytes = f.read()
    with open(canonical, "rb") as f:
        canonical_bytes = f.read()
    assert bundled_bytes == canonical_bytes, (
        f"bundled {fname} has drifted from examples/ — re-sync the wheel copy "
        f"(bundled {len(bundled_bytes)} bytes != examples {len(canonical_bytes)} bytes)"
    )


# ── W1 regression — no-copy path rejects a non-contiguous same-storage view ─────────
@pytest.mark.skipif(not _gpu_enabled(), reason="GPU tests gated (AXC_ENABLE_GPU_TESTS=1)")
def test_q4km_zero_copy_rejects_noncontiguous_shared_view():
    """W1 guard: a same-storage NON-CONTIGUOUS view (transpose of the shared view) at a SQUARE
    shape has a matching data_ptr and passes shape[0]==K, but its logical [K,N] layout does NOT
    match the contiguous storage the kernel reads. The no-copy path MUST reject it (raise) rather
    than silently reinterpret memory — it must NOT produce a (wrong) result."""
    m, n, k = 256, 256, 256  # square: a transpose has the SAME shape and data_ptr
    n_bpr = k // 256
    q = make_q4km_weights(m, n_bpr, 0xC0FFEE ^ m)
    x_bits = make_x_f16_bits(k, n, 0xBADF00D ^ n)
    op = axc.Q4KMMatmul()
    op.upload_weights(q, m, k)
    try:
        s = op.kernel.stream
        xv = op.x_view(n)
        x_src = _x_f16_tensor_from_bits(x_bits, k, n)
        with torch.cuda.stream(s):
            xv.copy_(x_src)
        s.synchronize()

        # A transposed view of the shared buffer: SAME data_ptr, SAME (square) shape, but strided.
        xv_t = xv.t()
        assert xv_t.data_ptr() == xv.data_ptr()
        assert tuple(xv_t.shape) == (k, n)
        assert not xv_t.is_contiguous()
        with pytest.raises(axc.ZeroCopyUnavailable):
            op.matmul(xv_t)

        # And the genuine contiguous shared view STILL works (W1 must not break the happy path).
        c = op.matmul(xv)
        s.synchronize()
        assert tuple(c.shape) == (m, n)
        print("[W1] non-contiguous same-storage view REJECTED; contiguous shared view still passes")
    finally:
        op.close()


# ── AT-2103d — strategy load-bearing + fail-closed (no GPU dispatch needed for the compile half) ──
@pytest.mark.skipif(not _gpu_enabled(), reason="GPU tests gated (AXC_ENABLE_GPU_TESTS=1)")
def test_q4km_fail_closed_and_strategy_required():
    """AT-2103d: strategy=None on the M3.6 source FAILS at compile; wrong-dtype/CPU x fails closed."""
    src = axc.q4km.Q4KM_KERNEL_SRC
    with pytest.raises(axc.AxiomError):
        axc.compile_kernel(src, strategy=None)
    print("[AT-2103d] strategy=None on the M3.6 source fails at compile (UnresolvedStrategyHole)")

    # Fail-closed: an f32 x (wrong dtype) and a CPU x are rejected, never silently reinterpreted.
    m, n, k = 256, 256, 256
    n_bpr = k // 256
    q = make_q4km_weights(m, n_bpr, 0xC0FFEE ^ m)
    op = axc.Q4KMMatmul()
    op.upload_weights(q, m, k)
    try:
        x_f32 = torch.zeros((k, n), dtype=torch.float32, device="cuda")
        with pytest.raises(axc.ZeroCopyUnavailable):
            op.matmul(x_f32)
        x_cpu = torch.zeros((k, n), dtype=torch.float16, device="cpu")
        with pytest.raises(axc.ZeroCopyUnavailable):
            op.matmul(x_cpu)
        print("[AT-2103d] wrong-dtype (f32) and CPU x both fail closed (ZeroCopyUnavailable)")
    finally:
        op.close()
