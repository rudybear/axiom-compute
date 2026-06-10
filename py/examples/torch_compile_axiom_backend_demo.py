"""M4.3 — the <=15-line capability demo for backend="axiom".

A tiny non-causal attention nn.Module using F.scaled_dot_product_attention is compiled with
torch.compile(model, backend="axiom"); the matching SDPA is AUTO-routed to the AXIOM coopmat
flash_attention kernel. We assert the AXIOM op FIRED at runtime and the output matches eager
within the FROZEN 1e-3 tolerance.

HONEST: this is a CAPABILITY demonstration — AXIOM registers as a torch.compile backend and
auto-lowers a matching op while staying correct. It is NOT a perf win: auto-routing SDPA to the
AXIOM kernel is SLOWER than native SDPA (FlashAttention / mem-efficient / fused-math). The backend
is OPT-IN only (backend="axiom"), never a default, and makes NO beat claim. q4km is OUT of
auto-lowering scope (it needs pre-packed Q4_K_M weights, not a graph value).

Run:  VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json AXC_ENABLE_GPU_TESTS=1 \
      python3 py/examples/torch_compile_axiom_backend_demo.py
"""


def main():
    # >>> DEMO-BEGIN (<=15 statements) >>>
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import axiom_compute as axc                                       # registers backend="axiom"
    class Attn(nn.Module):
        def forward(self, q, k, v):
            return F.scaled_dot_product_attention(q, k, v)            # non-causal, no mask, default scale
    model = Attn().cuda().eval()
    q = torch.randn(1, 1, 512, 64, device="cuda", dtype=torch.float32)
    k = torch.randn(1, 1, 512, 64, device="cuda", dtype=torch.float32)
    v = torch.randn(1, 1, 512, 64, device="cuda", dtype=torch.float32)
    axc.reset_axiom_sdpa_counters()
    with torch.no_grad():
        out_axiom = torch.compile(model, backend="axiom")(q, k, v)   # SDPA auto-routed to AXIOM
        out_eager = model(q, k, v)
    fired = axc.axiom_sdpa_runtime_fire_count()
    max_diff = (out_axiom - out_eager).abs().max().item()
    assert fired > 0, "the AXIOM op did not fire"
    assert torch.allclose(out_axiom, out_eager, atol=1e-3, rtol=1e-3)
    print(f"AXIOM flash_attention fired {fired} time(s); max diff vs eager = {max_diff:.2e} "
          "(capability demo — SLOWER than native SDPA, opt-in)")
    # <<< DEMO-END <<<
    return fired, max_diff


if __name__ == "__main__":
    main()
