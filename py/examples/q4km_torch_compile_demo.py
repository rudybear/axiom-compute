"""M4.1 Phase 3 — the <=10-line PyTorch user demo (ROADMAP 'a 10-line PyTorch user code').

Registers Q4_K_M weights ONCE, builds an f16 [K, N] activation, calls the registered op
torch.ops.axiom.q4km_matmul under torch.compile, and prints the output shape + which
dispatch path ran. The op is a first-class torch operator that composes under torch.compile
(register_fake supplies the [M, N] f32 meta) — the value of Phase 3 is REAL torch integration,
NOT speed (no beat-cuBLAS; ~53% of f32 cuBLAS).

The printed path is honestly 'device_copy' under torch.compile: Inductor functionalization may
realize/clone the input before the op, so the EAGER zero-copy fast path (write directly into
axc.q4km_activation_view(wid, N) and pass THAT tensor) may be unreachable under compile. The op
is correct either way.

Run:  VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json AXC_ENABLE_GPU_TESTS=1 \
      python py/examples/q4km_torch_compile_demo.py
"""

import numpy as np

# The Q4_K_M weight bytes here use the same byte-identical RNG fixture as the test suite so
# the result is correct against the Rust oracle; real users pass their own ggml Q4_K_M bytes.
_MASK64 = (1 << 64) - 1


def make_q4km_weights(m: int, n_bpr: int, seed: int) -> bytes:
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


def main():
    # >>> DEMO-BEGIN (<=10 statements) >>>
    import torch
    import axiom_compute as axc
    M, N, K = 256, 256, 256
    wid = axc.q4km_register_weights(make_q4km_weights(M, K // 256, 0xC0FFEE ^ M), M, K)  # upload-once; records (M,K)
    x = torch.randn(K, N, device="cuda", dtype=torch.float16)                            # arbitrary f16 [K,N] activation
    f = torch.compile(lambda a: torch.ops.axiom.q4km_matmul(a, wid, M, N, K))            # composes under torch.compile
    with torch.no_grad():
        C = f(x)                                                                          # compiled call -> [M, N] f32
    print(C.shape, axc.last_dispatch_path(wid))                                           # torch.Size([256, 256]) device_copy
    # <<< DEMO-END <<<
    return wid, x, C, M, N, K


if __name__ == "__main__":
    main()
