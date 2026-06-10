"""M4.1 Phase 4 — the <=10-line PyTorch user demo for the FlashAttention op.

Builds q/k/v CUDA f32 tensors, calls the registered op torch.ops.axiom.flash_attention under
torch.compile, and prints the output shape + which dispatch path ran. The op is a first-class
torch operator that composes under torch.compile (register_fake supplies the [seq, head_dim] f32
meta) — the value of Phase 4 is REAL torch integration of a SECOND kernel (Q4_K_M matmul +
FlashAttention BOTH callable from PyTorch), NOT speed (no beat-SDPA / beat-cuBLAS claim).

The printed path is honestly 'device_copy' under torch.compile: Inductor functionalization may
realize/clone the inputs before the op, so the EAGER zero-copy fast path (write directly into
axc.flash_attention_views(...) and pass THOSE tensors) may be unreachable under compile. The op
is correct either way. Constraints (coopmat leader): head_dim == 64, seq % 16 == 0.

Run:  VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json AXC_ENABLE_GPU_TESTS=1 \
      python py/examples/flash_attention_torch_compile_demo.py
"""


def main():
    # >>> DEMO-BEGIN (<=10 statements) >>>
    import torch
    import axiom_compute as axc
    SEQ, D = 512, 64
    q = torch.randn(SEQ, D, device="cuda", dtype=torch.float32)                          # [seq, head_dim] f32
    k = torch.randn(SEQ, D, device="cuda", dtype=torch.float32)
    v = torch.randn(SEQ, D, device="cuda", dtype=torch.float32)
    f = torch.compile(lambda a, b, c: torch.ops.axiom.flash_attention(a, b, c, SEQ, D))  # composes under torch.compile
    with torch.no_grad():
        O = f(q, k, v)                                                                    # compiled call -> [seq, D] f32
    print(O.shape, axc.last_attn_dispatch_path(q, SEQ, D))                                # torch.Size([512, 64]) <path>
    # <<< DEMO-END <<<
    return q, k, v, O, SEQ, D


if __name__ == "__main__":
    main()
