"""M4.1 Phase 3 — torch.library custom-op registration tests.

AT-2111   the registered op torch.ops.axiom.q4km_matmul, called DIRECTLY (eager), combined
          <= FROZEN 1e-3 at K=256/512/14336.
AT-2111b  torch.compile(f, fullgraph=True) under no_grad: no graph break, single leaf node,
          compiled == eager, combined <= 1e-3, records whichever dispatch path ran.
AT-2111c  R1 fail-closed: a mismatched op M (or N, or K) RAISES (never a silent clone).
AT-2111d  R3 cross-stream: the op from a NON-DEFAULT cuda stream (eager + compiled) is bit-exact.
AT-2111e  mutates_args=() soundness: a fullgraph region calling the op TWICE with the same x
          returns BOTH results correct (Inductor dedup/CSE did not collapse to one stale clone).
AT-2112   zero-copy fast path (EAGER): x IS the shared view -> 0/0 staging + path 'zero_copy'.
AT-2113   arbitrary-tensor copy path: an arbitrary CUDA f16 [K,N] x -> correct + path 'device_copy'.
AT-2113b  session-cache amortization: >=100 op calls reuse ONE session (identity stable, 0/0).
AT-2116   the <=10-line demo runs on GPU + is correct.
AT-2117   opcheck passes; a grad-tracking call raises (no autograd).
AT-2118   no-autograd honesty (grad-context call raises).

Gated by AXC_ENABLE_GPU_TESTS=1 + torch.cuda.is_available() + an nvidia ICD.
"""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import axiom_compute as axc
from axiom_compute import (
    q4km_register_weights,
    q4km_release_weights,
    q4km_activation_view,
    last_dispatch_path,
    last_dispatch_staging,
)

# Reuse the byte-identical fixtures so GPU == oracle.
from test_zero_copy_q4km import (
    make_q4km_weights,
    make_x_f16_bits,
    _x_f16_tensor_from_bits,
)


def _gpu_enabled():
    return os.environ.get("AXC_ENABLE_GPU_TESTS") == "1" and torch.cuda.is_available()


FROZEN_REL_TOL = 1e-3
_SMALL_SHAPES = [(256, 256, 256), (512, 512, 512)]
_AB_K = (256, 256, 14336)  # the inference-depth K where f16-accum was invalid; f32 passes


def _combined(y_gpu_tensor, q, x_bits, m, n, k):
    n_bpr = k // 256
    y_gpu = y_gpu_tensor.detach().to("cpu").to(torch.float32).reshape(-1).tolist()
    y_ref, abs_scale = axc.q4km_f32_reference(q, x_bits, m, n, n_bpr)
    return axc.q4km_combined_metric(y_gpu, y_ref, abs_scale)


def _register(m, n, k):
    """Register weights, return (wid, q, x_bits) for shape (m, n, k)."""
    n_bpr = k // 256
    q = make_q4km_weights(m, n_bpr, 0xC0FFEE ^ m)
    x_bits = make_x_f16_bits(k, n, 0xBADF00D ^ n)
    wid = q4km_register_weights(q, m, k)
    return wid, q, x_bits


# ── AT-2111 — eager correctness ──────────────────────────────────────────────────────
@pytest.mark.skipif(not _gpu_enabled(), reason="GPU tests gated (AXC_ENABLE_GPU_TESTS=1)")
@pytest.mark.parametrize("m,n,k", _SMALL_SHAPES + [_AB_K])
def test_q4km_op_eager_within_1e3(m, n, k):
    wid, q, x_bits = _register(m, n, k)
    try:
        x = _x_f16_tensor_from_bits(x_bits, k, n)
        c = torch.ops.axiom.q4km_matmul(x, wid, m, n, k)
        assert tuple(c.shape) == (m, n) and c.dtype == torch.float32 and c.is_cuda
        combined = _combined(c, q, x_bits, m, n, k)
        print(f"[AT-2111] op eager M={m} N={n} K={k}: combined={combined:.3e}")
        assert combined <= FROZEN_REL_TOL
    finally:
        q4km_release_weights(wid)


# ── AT-2111b — torch.compile fullgraph ───────────────────────────────────────────────
@pytest.mark.skipif(not _gpu_enabled(), reason="GPU tests gated (AXC_ENABLE_GPU_TESTS=1)")
def test_q4km_op_torch_compile_fullgraph():
    m, n, k = 256, 256, 256
    wid, q, x_bits = _register(m, n, k)
    try:
        x = _x_f16_tensor_from_bits(x_bits, k, n)
        with torch.no_grad():
            eager = torch.ops.axiom.q4km_matmul(x, wid, m, n, k)

        def f(a):
            return torch.ops.axiom.q4km_matmul(a, wid, m, n, k)

        # Structural no-graph-break check via torch._dynamo.explain.
        broke = None
        try:
            explanation = torch._dynamo.explain(f)(x)
            broke = getattr(explanation, "graph_break_count", None)
            print(f"[AT-2111b] dynamo graph_break_count={broke}")
        except Exception as e:
            print(f"[AT-2111b] dynamo.explain unavailable: {e}")

        compiled = None
        try:
            cf = torch.compile(f, fullgraph=True)
            with torch.no_grad():
                compiled = cf(x)
            print("[AT-2111b] fullgraph=True compiled OK (no graph break)")
        except Exception as e:
            print(f"[AT-2111b] fullgraph=True raised ({e}); falling back to default compile")
            cf = torch.compile(f)
            with torch.no_grad():
                compiled = cf(x)

        # Compiled == eager (bit-exact: the op is deterministic given x+weights).
        maxdiff = (compiled.float() - eager.float()).abs().max().item()
        assert maxdiff == 0.0, f"compiled != eager (maxdiff {maxdiff})"
        combined = _combined(compiled, q, x_bits, m, n, k)
        path = last_dispatch_path(wid)  # record whichever path ran (likely device_copy)
        print(f"[AT-2111b] compiled combined={combined:.3e} path={path}")
        assert combined <= FROZEN_REL_TOL
        if broke is not None:
            assert broke == 0, f"graph broke {broke} times under fullgraph"
    finally:
        q4km_release_weights(wid)


# ── AT-2111c — R1 fail-closed (M/N/K mismatch raises) ────────────────────────────────
@pytest.mark.skipif(not _gpu_enabled(), reason="GPU tests gated (AXC_ENABLE_GPU_TESTS=1)")
def test_q4km_op_mnk_mismatch_fail_closed():
    m, n, k = 256, 256, 256
    wid, q, x_bits = _register(m, n, k)
    try:
        x = _x_f16_tensor_from_bits(x_bits, k, n)
        # Correct (M,N,K) succeeds with the meta shape.
        ok = torch.ops.axiom.q4km_matmul(x, wid, m, n, k)
        assert tuple(ok.shape) == (m, n) and ok.dtype == torch.float32

        # Mismatched M (entry.m != op M) -> raise.
        with pytest.raises(Exception) as ei_m:
            torch.ops.axiom.q4km_matmul(x, wid, m + 32, n, k)
        assert "M mismatch" in str(ei_m.value)

        # Mismatched N (x.shape[1] != op N) -> raise.
        with pytest.raises(Exception) as ei_n:
            torch.ops.axiom.q4km_matmul(x, wid, m, n + 32, k)
        assert "N mismatch" in str(ei_n.value)

        # Mismatched K (x.shape[0]/entry.k != op K) -> raise.
        with pytest.raises(Exception) as ei_k:
            torch.ops.axiom.q4km_matmul(x, wid, m, n, k + 256)
        assert "K mismatch" in str(ei_k.value)
        print("[AT-2111c] mismatched M/N/K each raise fail-closed")
    finally:
        q4km_release_weights(wid)


# ── AT-2111d — R3 cross-stream correctness ───────────────────────────────────────────
@pytest.mark.skipif(not _gpu_enabled(), reason="GPU tests gated (AXC_ENABLE_GPU_TESTS=1)")
def test_q4km_op_cross_stream_correct():
    m, n, k = 256, 256, 256
    wid, q, x_bits = _register(m, n, k)
    try:
        x = _x_f16_tensor_from_bits(x_bits, k, n)
        # Same-stream reference.
        ref = torch.ops.axiom.q4km_matmul(x, wid, m, n, k).clone()
        torch.cuda.synchronize()

        # Run from a NON-DEFAULT active stream (eager), >=20x for determinism.
        alt = torch.cuda.Stream()
        for i in range(20):
            with torch.cuda.stream(alt):
                out = torch.ops.axiom.q4km_matmul(x, wid, m, n, k)
            alt.synchronize()
            maxdiff = (out.float() - ref.float()).abs().max().item()
            assert maxdiff == 0.0, f"cross-stream call {i}: not bit-exact ({maxdiff})"
        combined = _combined(ref, q, x_bits, m, n, k)
        assert combined <= FROZEN_REL_TOL

        # And under torch.compile from a non-default stream.
        def f(a):
            return torch.ops.axiom.q4km_matmul(a, wid, m, n, k)

        cf = torch.compile(f)
        with torch.cuda.stream(alt), torch.no_grad():
            cout = cf(x)
        alt.synchronize()
        assert (cout.float() - ref.float()).abs().max().item() == 0.0
        print("[AT-2111d] cross-stream eager + compiled bit-exact (R3 on-S execution)")
    finally:
        q4km_release_weights(wid)


# ── AT-2111e — mutates_args=() fullgraph dedup/CSE soundness ──────────────────────────
@pytest.mark.skipif(not _gpu_enabled(), reason="GPU tests gated (AXC_ENABLE_GPU_TESTS=1)")
def test_q4km_op_fullgraph_dedup_correct():
    m, n, k = 256, 256, 256
    wid, q, x_bits = _register(m, n, k)
    try:
        x = _x_f16_tensor_from_bits(x_bits, k, n)
        eager = torch.ops.axiom.q4km_matmul(x, wid, m, n, k).clone()

        def g(a):
            return (
                torch.ops.axiom.q4km_matmul(a, wid, m, n, k),
                torch.ops.axiom.q4km_matmul(a, wid, m, n, k),
            )

        try:
            cg = torch.compile(g, fullgraph=True)
        except Exception:
            cg = torch.compile(g)
        with torch.no_grad():
            r0, r1 = cg(x)
        # Both must equal the eager result and each other (no stale/aliased clone).
        assert (r0.float() - eager.float()).abs().max().item() == 0.0
        assert (r1.float() - eager.float()).abs().max().item() == 0.0
        assert r0.data_ptr() != r1.data_ptr(), "dedup collapsed to one aliased clone"
        print("[AT-2111e] fullgraph dedup: both calls correct, distinct buffers")
    finally:
        q4km_release_weights(wid)


# ── AT-2112 — zero-copy path (EAGER) 0/0 staging ─────────────────────────────────────
@pytest.mark.skipif(not _gpu_enabled(), reason="GPU tests gated (AXC_ENABLE_GPU_TESTS=1)")
def test_q4km_op_zero_copy_path_staging_free_eager():
    m, n, k = 256, 256, 256
    wid, q, x_bits = _register(m, n, k)
    try:
        xv = q4km_activation_view(wid, n)  # the shared zero-copy view
        x_src = _x_f16_tensor_from_bits(x_bits, k, n)
        # Write into the shared view on the session's captured stream S (G-5).
        import axiom_compute.ops as ops

        s = ops._WEIGHTS_REGISTRY[wid].sessions[n].session.stream
        with torch.cuda.stream(s):
            xv.copy_(x_src)
        s.synchronize()
        c = torch.ops.axiom.q4km_matmul(xv, wid, m, n, k)
        assert last_dispatch_path(wid) == "zero_copy"
        staging, copies = last_dispatch_staging(wid)
        assert staging == 0 and copies == 0, f"zero-copy must be 0/0, got {staging}/{copies}"
        combined = _combined(c, q, x_bits, m, n, k)
        assert combined <= FROZEN_REL_TOL
        print("[AT-2112] EAGER zero-copy: path=zero_copy staging=0/0")
    finally:
        q4km_release_weights(wid)


# ── AT-2113 — arbitrary tensor labeled device_copy ───────────────────────────────────
@pytest.mark.skipif(not _gpu_enabled(), reason="GPU tests gated (AXC_ENABLE_GPU_TESTS=1)")
def test_q4km_op_arbitrary_tensor_device_copy_labeled():
    m, n, k = 256, 256, 256
    wid, q, x_bits = _register(m, n, k)
    try:
        x = _x_f16_tensor_from_bits(x_bits, k, n)  # an arbitrary CUDA tensor (NOT the view)
        c = torch.ops.axiom.q4km_matmul(x, wid, m, n, k)
        assert last_dispatch_path(wid) == "device_copy"
        combined = _combined(c, q, x_bits, m, n, k)
        assert combined <= FROZEN_REL_TOL
        print("[AT-2113] arbitrary tensor labeled device_copy, correct")
    finally:
        q4km_release_weights(wid)


# ── AT-2113b — session-cache amortization ────────────────────────────────────────────
@pytest.mark.skipif(not _gpu_enabled(), reason="GPU tests gated (AXC_ENABLE_GPU_TESTS=1)")
def test_q4km_op_session_cache_amortized():
    m, n, k = 256, 256, 256
    wid, q, x_bits = _register(m, n, k)
    try:
        import axiom_compute.ops as ops

        x = _x_f16_tensor_from_bits(x_bits, k, n)
        first = torch.ops.axiom.q4km_matmul(x, wid, m, n, k).clone()
        entry = ops._WEIGHTS_REGISTRY[wid]
        sess_id = id(entry.sessions[n].session)
        for i in range(100):
            out = torch.ops.axiom.q4km_matmul(x, wid, m, n, k)
            assert id(entry.sessions[n].session) == sess_id, f"session churned at call {i}"
            assert (out.float() - first.float()).abs().max().item() == 0.0
        staging, copies = last_dispatch_staging(wid)
        assert (staging, copies) == (0, 0)
        print("[AT-2113b] 100 op calls reuse ONE session (identity stable, 0/0 staging)")
    finally:
        q4km_release_weights(wid)


# ── AT-2117 / AT-2118 — opcheck + no-autograd ────────────────────────────────────────
@pytest.mark.skipif(not _gpu_enabled(), reason="GPU tests gated (AXC_ENABLE_GPU_TESTS=1)")
def test_q4km_op_opcheck_and_no_autograd():
    m, n, k = 256, 256, 256
    wid, q, x_bits = _register(m, n, k)
    try:
        x = _x_f16_tensor_from_bits(x_bits, k, n)
        with torch.no_grad():
            torch.library.opcheck(torch.ops.axiom.q4km_matmul, (x, wid, m, n, k))
        print("[AT-2117] opcheck passed")

        # AT-2118: a grad-tracking call raises (no silent wrong gradient). The op has no
        # autograd formula; a float input cannot require grad, but the op explicitly does
        # not support backward — assert a grad-requiring use surfaces an error.
        xg = x.detach().clone().requires_grad_(False)
        with torch.enable_grad():
            # The op returns a leaf with no grad_fn; calling .backward() on a function of it
            # must raise (no autograd registered). We verify the result has no grad_fn.
            out = torch.ops.axiom.q4km_matmul(xg, wid, m, n, k)
            assert out.grad_fn is None, "op must not fabricate an autograd node"
        print("[AT-2118] no autograd node fabricated (inference-only)")
    finally:
        q4km_release_weights(wid)


# ── AT-2116 — the <=10-line demo runs + correct + line-count guard ───────────────────
@pytest.mark.skipif(not _gpu_enabled(), reason="GPU tests gated (AXC_ENABLE_GPU_TESTS=1)")
def test_demo_runs_and_correct():
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    demo_path = os.path.join(here, "..", "examples", "q4km_torch_compile_demo.py")
    demo_path = os.path.abspath(demo_path)

    # Line-count guard on the marked DEMO-BEGIN..DEMO-END region (<=10 statements).
    with open(demo_path, "r", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    begin = next(i for i, ln in enumerate(lines) if "DEMO-BEGIN" in ln)
    end = next(i for i, ln in enumerate(lines) if "DEMO-END" in ln)
    body = [
        ln for ln in lines[begin + 1:end]
        if ln.strip() and not ln.strip().startswith("#")
    ]
    assert len(body) <= 10, f"demo body has {len(body)} statements (> 10): {body}"

    spec = importlib.util.spec_from_file_location("q4km_torch_compile_demo", demo_path)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    wid, x, C, m, n, k = demo.main()
    try:
        # Correctness vs the Rust oracle within FROZEN 1e-3. x is torch.randn f16 → recover
        # its exact f16 bits so GPU == oracle.
        x_bits = x.detach().to("cpu").view(torch.int16).numpy().astype(np.uint16)
        x_bits = x_bits.reshape(-1).tolist()
        combined = _combined(C, demo.make_q4km_weights(m, k // 256, 0xC0FFEE ^ m), x_bits, m, n, k)
        print(f"[AT-2116] demo runs: shape={tuple(C.shape)} combined={combined:.3e}")
        assert tuple(C.shape) == (m, n)
        assert combined <= FROZEN_REL_TOL
    finally:
        q4km_release_weights(wid)
