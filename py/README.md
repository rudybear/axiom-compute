# axiom-compute (PyTorch frontend) — M4.1

A `pip`-installable PyTorch frontend for AXIOM-Compute: a torch **CUDA** tensor feeds
an AXIOM-Compute **Vulkan** kernel on the **same physical GPU** with **zero host
copies**, via `VK_KHR_external_memory_fd` export → `cudaImportExternalMemory` import,
synchronized by a shared Vulkan **timeline** external semaphore.

## Honest framing

Zero-copy ELIMINATES the host transfer, but the Vulkan kernel is ~34% of cuBLAS — this
is **not** a cuBLAS replacement. The win is **no host round-trip** + **real torch
interop** from a single annotated `.axc` source.

## Phase 1 correctness preconditions (hard requirements)

- **Single CUDA stream (G-5).** ALL ops on the shared tensors **and** the internal
  signal/wait handshake MUST run on the **one** stream captured at `compile_kernel` /
  `Kernel` construction (`kernel.stream`). Running a shared-tensor op on a *different*
  stream without an explicit event **races the CUDA↔Vulkan handshake** and can torn-read
  the shared buffer. The frontend **warns loudly** when it detects a divergent active
  stream in `run()` / `tensor()`, but it cannot move your ops onto the captured stream —
  always wrap them in `with torch.cuda.stream(kernel.stream): ...` (see the example).
- **NVIDIA-only sync model.** The exportable buffers are `VK_SHARING_MODE_EXCLUSIVE` and
  the dispatch emits **no** `VK_QUEUE_FAMILY_EXTERNAL` ownership-transfer barrier; it
  relies solely on the external **timeline semaphore** for both execution and memory
  ordering. This is correct and bit-exact on the M4.1 target (discrete NVIDIA, single
  Vulkan queue, 580.x driver) but is **not spec-portable** — AMD/Intel/MoltenVK generally
  require a CONCURRENT sharing mode over `{compute, VK_QUEUE_FAMILY_EXTERNAL}` or explicit
  ownership-transfer barriers. **Cross-vendor support (M4.2) must revisit this.**

## Prerequisites (runtime)

- CUDA 12.x + a torch build for it (`torch.cuda.is_available() == True`)
- An NVIDIA Vulkan ICD exposing `VK_KHR_external_memory_fd` + `VK_KHR_external_semaphore_fd`
- `cuda-python` (preferred) — `pip install cuda-python`
- `maturin` for the build — `pip install "maturin>=1.7,<2"`

## Build + install

```bash
pip install maturin
cd py
maturin develop --release      # or: maturin build --release && pip install ../target/wheels/axiom_compute-*.whl
```

Run with the NVIDIA ICD selected:

```bash
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json python your_script.py
```

## Minimal example (zero-copy saxpy)

```python
import struct, torch, axiom_compute as axc

src = open("examples/saxpy.axc").read()
k = axc.compile_kernel(src)                    # UUID-matched to the CUDA GPU (fail-closed)

n, alpha = 4096, 2.5
sess = k.allocate_shared(sizes=[n*4, n*4])     # exportable DEDICATED shared buffers
x = sess.tensor(0, (n,), torch.float32)        # NON-OWNING torch CUDA views
y = sess.tensor(1, (n,), torch.float32)
with torch.cuda.stream(sess.stream):           # G-5: one stream for all shared ops
    x.copy_(torch.arange(n, device="cuda"))
    y.zero_()
sess.run(workgroups=((n+63)//64, 1, 1),        # CUDA signal V1 -> Vulkan -> CUDA wait V2
         push_constants=struct.pack("<If", n, alpha))
# y == alpha*x + y, zero host copies
sess.close()                                   # G-4 exact teardown order
```

## Tests (self-hosted NVIDIA, gated)

```bash
AXC_ENABLE_GPU_TESTS=1 VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    python -m pytest tests/ -v
```

Covers the timeline-over-fd interop spike (AT-2110), zero-copy-VERIFIED same-memory
(AT-2101), saxpy bit-exactness (AT-2102), 1000-iter handshake determinism (AT-2104),
fail-closed (AT-2107), and lifetime/teardown (AT-2106).
