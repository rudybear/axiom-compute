"""M4.3 — AT-2140..AT-2149 for the ``backend="axiom"`` torch.compile backend.

AT-2149 (CPU, no GPU) is the LOAD-BEARING falsifier of the fail-closed ``_get_sdpa_arg`` extractor
+ ``_sdpa_match`` predicate over the full positional/keyword cross product (catches the
positional-form silent-wrong-math hole WITHOUT a GPU). AT-2140 (registration) is CPU-ok. The rest
(AT-2141..2147) need the coopmat kernel ⇒ NVIDIA; they are gated on ``AXC_ENABLE_GPU_TESTS=1`` +
CUDA and SKIP otherwise (Lavapipe is a typed coopmat-skip for those legs).

HONEST: this milestone is a CAPABILITY demo, NOT a perf win; FROZEN 1e-3 vs eager never loosens.
"""

from __future__ import annotations

import os

import pytest

import axiom_compute as axc
from axiom_compute.compile_backend import (
    AXIOM_SDPA_MAX_HEADS,
    _get_sdpa_arg,
    _sdpa_match,
    _SdpaMatch,
)

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402


_GPU_ENABLED = os.environ.get("AXC_ENABLE_GPU_TESTS") == "1" and torch.cuda.is_available()
_gpu = pytest.mark.skipif(
    not _GPU_ENABLED,
    reason="GPU leg: needs AXC_ENABLE_GPU_TESTS=1 + CUDA + the NVIDIA coopmat kernel",
)

_ATOL = 1e-3  # FROZEN — never loosens.
_RTOL = 1e-3


@pytest.fixture(autouse=True)
def _reset_dynamo_and_counters():
    """Test hygiene (risk R-H): the fire counters are process-global advisory hooks. Dynamo caches
    a compiled artifact per ``forward`` CODE OBJECT — without a reset, a second ``torch.compile`` of
    the SAME module class (different shape) reuses the cached guards and the backend is NOT
    re-invoked, so the counters do not increment. Reset Dynamo + zero the counters before EACH test
    so each backend invocation is observed independently."""
    try:
        import torch._dynamo

        torch._dynamo.reset()
    except Exception:
        pass
    axc.reset_axiom_sdpa_counters()
    yield


# ════════════════════════════════════════════════════════════════════════════════════════
#  AT-2149 — CPU falsifier of the extractor + predicate (no GPU, no compile, pure logic)
# ════════════════════════════════════════════════════════════════════════════════════════
class _FakeMeta(dict):
    """A dict that also doubles as an FX-arg-like with a ``.meta`` of itself — minimal stub."""


class _FakeTensorMeta:
    """A FakeTensor-like exposing shape/dtype/requires_grad."""

    def __init__(self, shape, dtype=torch.float32, requires_grad=False):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.requires_grad = requires_grad


class _FakeArg:
    """An FX-node-like q/k/v ARG: exposes ``.meta`` with an 'example_value' FakeTensor."""

    def __init__(self, shape, dtype=torch.float32, requires_grad=False, meta_keys=("example_value",)):
        tm = _FakeTensorMeta(shape, dtype, requires_grad)
        self.meta = {k: tm for k in meta_keys}
        self.name = "arg"


class _FakeNode:
    """An FX-node-like SDPA node: ``.op``, ``.target``, ``.args``, ``.kwargs``."""

    def __init__(self, target, args, kwargs=None, op="call_function"):
        self.op = op
        self.target = target
        self.args = tuple(args)
        self.kwargs = dict(kwargs or {})


_SDPA = F.scaled_dot_product_attention


def _qkv(shape):
    return _FakeArg(shape), _FakeArg(shape), _FakeArg(shape)


def test_AT2149_get_sdpa_arg_positional_or_keyword():
    """_get_sdpa_arg: positional-or-keyword reads args[idx] when present, else kwarg, else default."""
    # positional present ⇒ read from args.
    n = _FakeNode(_SDPA, args=("q", "k", "v", "MASK", 0.3, True))
    assert _get_sdpa_arg(n, 3, "attn_mask", None) == "MASK"
    assert _get_sdpa_arg(n, 4, "dropout_p", 0.0) == 0.3
    assert _get_sdpa_arg(n, 5, "is_causal", False) is True
    # positional absent ⇒ read from kwarg.
    n2 = _FakeNode(_SDPA, args=("q", "k", "v"), kwargs={"is_causal": True, "attn_mask": "M2"})
    assert _get_sdpa_arg(n2, 5, "is_causal", False) is True
    assert _get_sdpa_arg(n2, 3, "attn_mask", None) == "M2"
    # absent everywhere ⇒ default.
    assert _get_sdpa_arg(n2, 4, "dropout_p", 0.0) == 0.0


def test_AT2149_get_sdpa_arg_keyword_only():
    """_get_sdpa_arg: keyword-only (idx=None) reads kwargs ONLY, never positional."""
    # Even if a 7th positional existed, scale (idx=None) must NOT read it.
    n = _FakeNode(_SDPA, args=("q", "k", "v", None, 0.0, False, "NOT_SCALE"))
    assert _get_sdpa_arg(n, None, "scale", None) is None  # ignores the positional tail.
    n2 = _FakeNode(_SDPA, args=("q", "k", "v"), kwargs={"scale": 0.125, "enable_gqa": True})
    assert _get_sdpa_arg(n2, None, "scale", None) == 0.125
    assert _get_sdpa_arg(n2, None, "enable_gqa", False) is True


def test_AT2149_positional_causal_no_match():
    """THE hole fixture: F.sdpa(q,k,v,None,0.0,True) — is_causal at args[5] ⇒ NO MATCH.

    A kwargs-only reader would read is_causal=False (default) and MATCH ⇒ silent wrong math. This
    asserts the fail-closed extractor catches it WITHOUT a GPU."""
    q, k, v = _qkv([1, 1, 512, 64])
    n = _FakeNode(_SDPA, args=(q, k, v, None, 0.0, True))
    assert _sdpa_match(n) is None


def test_AT2149_positional_mask_no_match():
    """F.sdpa(q,k,v,mask) — attn_mask at args[3] (a non-None node) ⇒ NO MATCH."""
    q, k, v = _qkv([1, 1, 512, 64])
    mask = _FakeArg([1, 1, 512, 512])
    n = _FakeNode(_SDPA, args=(q, k, v, mask))
    assert _sdpa_match(n) is None


def test_AT2149_kwarg_causal_no_match():
    q, k, v = _qkv([1, 1, 512, 64])
    n = _FakeNode(_SDPA, args=(q, k, v), kwargs={"is_causal": True})
    assert _sdpa_match(n) is None


def test_AT2149_kwarg_dropout_no_match():
    q, k, v = _qkv([1, 1, 512, 64])
    n = _FakeNode(_SDPA, args=(q, k, v), kwargs={"dropout_p": 0.1})
    assert _sdpa_match(n) is None


def test_AT2149_custom_scale_no_match_but_default_scale_matches():
    """scale=0.5 (≠ 1/8 at D=64) ⇒ NO MATCH; scale=0.125 (== 1/sqrt(64)) ⇒ MATCH."""
    q, k, v = _qkv([1, 1, 512, 64])
    n_bad = _FakeNode(_SDPA, args=(q, k, v), kwargs={"scale": 0.5})
    assert _sdpa_match(n_bad) is None
    q2, k2, v2 = _qkv([1, 1, 512, 64])
    n_ok = _FakeNode(_SDPA, args=(q2, k2, v2), kwargs={"scale": 0.125})
    m = _sdpa_match(n_ok)
    assert isinstance(m, _SdpaMatch) and m.head_dim == 64


def test_AT2149_gqa_shapes_disagree_no_match():
    """GQA: q [2,8,512,64], k/v [2,2,512,64] — shapes disagree on H ⇒ NO MATCH (PRIMARY guard)."""
    q = _FakeArg([2, 8, 512, 64])
    k = _FakeArg([2, 2, 512, 64])
    v = _FakeArg([2, 2, 512, 64])
    n = _FakeNode(_SDPA, args=(q, k, v))
    assert _sdpa_match(n) is None


def test_AT2149_enable_gqa_flag_no_match():
    q, k, v = _qkv([1, 1, 512, 64])
    n = _FakeNode(_SDPA, args=(q, k, v), kwargs={"enable_gqa": True})
    assert _sdpa_match(n) is None


def test_AT2149_unknown_form_no_match():
    """len(args)>6, an unknown kwarg key, and a non-SDPA target ⇒ NO MATCH (P0 unknown-form)."""
    q, k, v = _qkv([1, 1, 512, 64])
    # len(args) > 6.
    n_long = _FakeNode(_SDPA, args=(q, k, v, None, 0.0, False, "extra"))
    assert _sdpa_match(n_long) is None
    # unknown kwarg key.
    q2, k2, v2 = _qkv([1, 1, 512, 64])
    n_kw = _FakeNode(_SDPA, args=(q2, k2, v2), kwargs={"foo": 1})
    assert _sdpa_match(n_kw) is None
    # target not in the SDPA set.
    q3, k3, v3 = _qkv([1, 1, 512, 64])
    n_tgt = _FakeNode(torch.add, args=(q3, k3, v3))
    assert _sdpa_match(n_tgt) is None
    # not a call_function.
    q4, k4, v4 = _qkv([1, 1, 512, 64])
    n_op = _FakeNode(_SDPA, args=(q4, k4, v4), op="call_method")
    assert _sdpa_match(n_op) is None


def test_AT2149_head_dim_and_seq_constraints_no_match():
    """head_dim != 64 ⇒ NO MATCH; seq % 16 != 0 ⇒ NO MATCH; f16 ⇒ NO MATCH; requires_grad ⇒ NO."""
    # head_dim 32.
    assert _sdpa_match(_FakeNode(_SDPA, args=_qkv([1, 1, 512, 32]))) is None
    # seq 500 (not %16).
    assert _sdpa_match(_FakeNode(_SDPA, args=_qkv([1, 1, 500, 64]))) is None
    # f16.
    q = _FakeArg([1, 1, 512, 64], dtype=torch.float16)
    assert _sdpa_match(_FakeNode(_SDPA, args=(q, _FakeArg([1, 1, 512, 64], dtype=torch.float16),
                                             _FakeArg([1, 1, 512, 64], dtype=torch.float16)))) is None
    # requires_grad.
    qg = _FakeArg([1, 1, 512, 64], requires_grad=True)
    assert _sdpa_match(_FakeNode(_SDPA, args=(qg, _FakeArg([1, 1, 512, 64]),
                                             _FakeArg([1, 1, 512, 64])))) is None


def test_AT2149_over_max_heads_no_match():
    """B*H > AXIOM_SDPA_MAX_HEADS ⇒ NO MATCH (avoid unrolling an enormous loop)."""
    big = AXIOM_SDPA_MAX_HEADS + 1
    assert _sdpa_match(_FakeNode(_SDPA, args=_qkv([1, big, 512, 64]))) is None


def test_AT2149_matching_case_returns_descriptor():
    """The matching case: f32, q==k==v [1,1,512,64], no mask/causal/dropout/scale, no grad ⇒ MATCH."""
    q, k, v = _qkv([1, 1, 512, 64])
    m = _sdpa_match(_FakeNode(_SDPA, args=(q, k, v)))
    assert isinstance(m, _SdpaMatch)
    assert (m.batch, m.n_heads, m.seq, m.head_dim) == (1, 1, 512, 64)


def test_AT2149_matching_multihead_descriptor():
    q, k, v = _qkv([2, 4, 512, 64])
    m = _sdpa_match(_FakeNode(_SDPA, args=(q, k, v)))
    assert isinstance(m, _SdpaMatch)
    assert (m.batch, m.n_heads, m.seq, m.head_dim) == (2, 4, 512, 64)


def test_AT2149_meta_key_example_value_first_and_no_meta_no_match():
    """example_value-only meta ⇒ MATCH (tried first); NO meta keys ⇒ NO MATCH (fail-closed)."""
    # Only 'example_value' present (no 'val'/'tensor_meta') ⇒ still reads the shape ⇒ MATCH.
    q, k, v = _qkv([1, 1, 512, 64])
    assert isinstance(_sdpa_match(_FakeNode(_SDPA, args=(q, k, v))), _SdpaMatch)
    # No meta keys at all ⇒ fail-closed NO MATCH.
    q0 = _FakeArg([1, 1, 512, 64], meta_keys=())
    k0 = _FakeArg([1, 1, 512, 64], meta_keys=())
    v0 = _FakeArg([1, 1, 512, 64], meta_keys=())
    assert _sdpa_match(_FakeNode(_SDPA, args=(q0, k0, v0))) is None


# ════════════════════════════════════════════════════════════════════════════════════════
#  AT-2140 — registration (CPU-ok)
# ════════════════════════════════════════════════════════════════════════════════════════
def test_AT2140_backend_registered_and_idempotent():
    import torch._dynamo as d

    assert axc.AXIOM_BACKEND_NAME in d.list_backends()
    # idempotent: a second call is a no-op (no raise, still registered).
    axc.register_axiom_backend()
    assert axc.AXIOM_BACKEND_NAME in d.list_backends()
    # torch.compile constructs with backend="axiom" without error.
    m = nn.Linear(8, 8)
    _ = torch.compile(m, backend="axiom")


# ════════════════════════════════════════════════════════════════════════════════════════
#  Test models
# ════════════════════════════════════════════════════════════════════════════════════════
class _SdpaModel(nn.Module):
    def __init__(self, is_causal=False, scale=None, use_mask=False, mask_positional=False,
                 causal_positional=False):
        super().__init__()
        self.is_causal = is_causal
        self.scale = scale
        self.use_mask = use_mask
        self.mask_positional = mask_positional
        self.causal_positional = causal_positional

    def forward(self, q, k, v, mask=None):
        if self.causal_positional:
            return F.scaled_dot_product_attention(q, k, v, None, 0.0, True)
        if self.mask_positional:
            return F.scaled_dot_product_attention(q, k, v, mask)
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=(mask if self.use_mask else None),
            is_causal=self.is_causal, scale=self.scale,
        )


class _MixedModel(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.lin1 = nn.Linear(d, d)
        self.lin2 = nn.Linear(d, d)

    def forward(self, q, k, v):
        q2 = torch.relu(self.lin1(q))
        o = F.scaled_dot_product_attention(q2, k, v)
        return self.lin2(o)


def _randqkv(shape, dtype=torch.float32):
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(*shape, device="cuda", dtype=dtype, generator=g)
    k = torch.randn(*shape, device="cuda", dtype=dtype, generator=g)
    v = torch.randn(*shape, device="cuda", dtype=dtype, generator=g)
    return q, k, v


# ════════════════════════════════════════════════════════════════════════════════════════
#  AT-2141/2142/2143 — the op FIRES on a matching SDPA, output within FROZEN 1e-3
# ════════════════════════════════════════════════════════════════════════════════════════
@_gpu
def test_AT2141_2142_matching_sdpa_fires_and_correct():
    model = _SdpaModel().cuda().eval()
    q, k, v = _randqkv([1, 1, 512, 64])
    axc.reset_axiom_sdpa_counters()
    with torch.no_grad():
        out_eager = model(q, k, v)
        compiled = torch.compile(model, backend="axiom")
        out_axiom = compiled(q, k, v)
    # AT-2141: the op FIRED — node swapped AND called at runtime.
    assert axc.axiom_sdpa_dispatch_count() > 0, "no SDPA node was rewritten"
    assert axc.axiom_sdpa_runtime_fire_count() > 0, "the AXIOM op did not fire at runtime"
    # AT-2142: within FROZEN 1e-3 of eager.
    max_diff = (out_axiom - out_eager).abs().max().item()
    assert torch.allclose(out_axiom, out_eager, atol=_ATOL, rtol=_RTOL), (
        f"max diff {max_diff:.2e} exceeds FROZEN 1e-3 (f16-coopmat QKᵀ/PV limitation)"
    )


@_gpu
def test_AT2143_matching_sdpa_matches_inductor():
    model = _SdpaModel().cuda().eval()
    q, k, v = _randqkv([1, 1, 512, 64])
    with torch.no_grad():
        out_inductor = torch.compile(model, backend="inductor")(q, k, v)
        axc.reset_axiom_sdpa_counters()
        out_axiom = torch.compile(model, backend="axiom")(q, k, v)
    assert axc.axiom_sdpa_runtime_fire_count() > 0
    assert torch.allclose(out_axiom, out_inductor, atol=_ATOL, rtol=_RTOL)


# ════════════════════════════════════════════════════════════════════════════════════════
#  AT-2144 — non-matching SDPA falls back CLEANLY (incl positional causal/mask)
# ════════════════════════════════════════════════════════════════════════════════════════
@_gpu
@pytest.mark.parametrize(
    "label,model_kwargs,shape,dtype,extra_mask",
    [
        ("head_dim32", {}, [1, 1, 512, 32], torch.float32, False),
        ("seq500", {}, [1, 1, 496 + 4, 64], torch.float32, False),  # 500 not %16
        ("causal_kwarg", {"is_causal": True}, [1, 1, 512, 64], torch.float32, False),
        ("causal_positional", {"causal_positional": True}, [1, 1, 512, 64], torch.float32, False),
        ("mask_kwarg", {"use_mask": True}, [1, 1, 512, 64], torch.float32, True),
        ("mask_positional", {"mask_positional": True}, [1, 1, 512, 64], torch.float32, True),
        ("f16", {}, [1, 1, 512, 64], torch.float16, False),
        ("custom_scale", {"scale": 0.5}, [1, 1, 512, 64], torch.float32, False),
    ],
)
def test_AT2144_nonmatching_falls_back(label, model_kwargs, shape, dtype, extra_mask):
    model = _SdpaModel(**model_kwargs).cuda().eval()
    q, k, v = _randqkv(shape, dtype=dtype)
    mask = None
    if extra_mask:
        mask = torch.zeros(shape[0], shape[1], shape[2], shape[2], device="cuda", dtype=dtype)
    axc.reset_axiom_sdpa_counters()
    with torch.no_grad():
        if extra_mask:
            out_eager = model(q, k, v, mask)
            out_axiom = torch.compile(model, backend="axiom")(q, k, v, mask)
        else:
            out_eager = model(q, k, v)
            out_axiom = torch.compile(model, backend="axiom")(q, k, v)
    # The AXIOM op did NOT fire (non-matching).
    assert axc.axiom_sdpa_runtime_fire_count() == 0, f"{label}: op fired but should NOT have"
    # Correct via Inductor fallback (torch's own SDPA tolerance for f16/non-default math).
    tol = 2e-2 if dtype == torch.float16 else 1e-3
    assert torch.allclose(out_axiom, out_eager, atol=tol, rtol=tol), f"{label}: wrong output"


@_gpu
def test_AT2144_gqa_falls_back():
    """GQA q[2,8,512,64] / kv[2,2,512,64] ⇒ shapes disagree ⇒ NO MATCH (enable_gqa=True for math)."""
    class _GQA(nn.Module):
        def forward(self, q, k, v):
            return F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

    model = _GQA().cuda().eval()
    g = torch.Generator(device="cuda").manual_seed(1)
    q = torch.randn(2, 8, 512, 64, device="cuda", generator=g)
    k = torch.randn(2, 2, 512, 64, device="cuda", generator=g)
    v = torch.randn(2, 2, 512, 64, device="cuda", generator=g)
    axc.reset_axiom_sdpa_counters()
    with torch.no_grad():
        out_eager = model(q, k, v)
        out_axiom = torch.compile(model, backend="axiom")(q, k, v)
    assert axc.axiom_sdpa_runtime_fire_count() == 0
    assert torch.allclose(out_axiom, out_eager, atol=1e-3, rtol=1e-3)


@_gpu
def test_AT2144_non_sdpa_op_no_fire():
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU()).cuda().eval()
    x = torch.randn(4, 64, device="cuda")
    axc.reset_axiom_sdpa_counters()
    with torch.no_grad():
        out_eager = model(x)
        out_axiom = torch.compile(model, backend="axiom")(x)
    assert axc.axiom_sdpa_runtime_fire_count() == 0
    assert torch.allclose(out_axiom, out_eager, atol=1e-3, rtol=1e-3)


# ════════════════════════════════════════════════════════════════════════════════════════
#  AT-2145 — mixed model lowers ONLY the SDPA
# ════════════════════════════════════════════════════════════════════════════════════════
@_gpu
def test_AT2145_mixed_model_only_sdpa_lowered():
    model = _MixedModel().cuda().eval()
    q, k, v = _randqkv([1, 1, 512, 64])
    axc.reset_axiom_sdpa_counters()
    with torch.no_grad():
        out_eager = model(q, k, v)
        out_axiom = torch.compile(model, backend="axiom")(q, k, v)
    assert axc.axiom_sdpa_runtime_fire_count() == 1, "exactly one SDPA should fire"
    assert torch.allclose(out_axiom, out_eager, atol=_ATOL, rtol=_RTOL)


# ════════════════════════════════════════════════════════════════════════════════════════
#  AT-2146 — [B,H,S,D] multi-head per-head loop, output correct (no head permutation)
# ════════════════════════════════════════════════════════════════════════════════════════
@_gpu
def test_AT2146_multihead_lowered_correct():
    model = _SdpaModel().cuda().eval()
    # DISTINCT per-head data so a head permutation would blow 1e-3.
    g = torch.Generator(device="cuda").manual_seed(7)
    q = torch.randn(2, 4, 512, 64, device="cuda", generator=g)
    k = torch.randn(2, 4, 512, 64, device="cuda", generator=g)
    v = torch.randn(2, 4, 512, 64, device="cuda", generator=g)
    axc.reset_axiom_sdpa_counters()
    with torch.no_grad():
        out_eager = model(q, k, v)
        out_axiom = torch.compile(model, backend="axiom")(q, k, v)
    assert axc.axiom_sdpa_runtime_fire_count() == 2 * 4, "B*H==8 per-head fires expected"
    max_diff = (out_axiom - out_eager).abs().max().item()
    assert torch.allclose(out_axiom, out_eager, atol=_ATOL, rtol=_RTOL), (
        f"max diff {max_diff:.2e} — a head permutation or wrong stack would exceed 1e-3"
    )


# ════════════════════════════════════════════════════════════════════════════════════════
#  AT-2147 — the demo runs on GPU, op fires, within 1e-3
# ════════════════════════════════════════════════════════════════════════════════════════
@_gpu
def test_AT2147_demo_runs():
    import importlib.util
    import pathlib

    demo_path = pathlib.Path(__file__).resolve().parents[1] / "examples" / "torch_compile_axiom_backend_demo.py"
    spec = importlib.util.spec_from_file_location("axiom_backend_demo", demo_path)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    fired, max_diff = demo.main()
    assert fired > 0
    assert max_diff <= _ATOL


# ════════════════════════════════════════════════════════════════════════════════════════
#  AT-2148 — NO regression to the explicit ops + no-SDPA passthrough
# ════════════════════════════════════════════════════════════════════════════════════════
@_gpu
def test_AT2148_explicit_flash_op_still_works():
    q, k, v = _randqkv([1, 1, 512, 64])
    q2, k2, v2 = q[0, 0].contiguous(), k[0, 0].contiguous(), v[0, 0].contiguous()
    with torch.no_grad():
        out = torch.ops.axiom.flash_attention(q2, k2, v2, 512, 64)
    assert out.shape == (512, 64) and out.dtype == torch.float32


@_gpu
def test_AT2148_no_sdpa_passthrough_matches_inductor():
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU()).cuda().eval()
    x = torch.randn(4, 64, device="cuda")
    axc.reset_axiom_sdpa_counters()
    with torch.no_grad():
        out_inductor = torch.compile(model, backend="inductor")(x)
        out_axiom = torch.compile(model, backend="axiom")(x)
    assert axc.axiom_sdpa_runtime_fire_count() == 0
    assert torch.equal(out_axiom, out_inductor)
