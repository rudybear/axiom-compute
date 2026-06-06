# AXIOM-Compute -- Design Document (v0.1, Living)

> Working document. Updated at every design-review approval. Last revision: M1.2 (this architect run): buffers, array indexing, gid.

---

## 1. Problem statement

### 1.1 The gap

GPU kernel programming is splintered across CUDA (NVIDIA-locked), ROCm/HIP (AMD), Metal (Apple), Vulkan compute + SPIR-V (cross-vendor), OpenCL (deprecated but reusable), and WebGPU (browser). Every existing shader/compute language was designed for humans:

| Project | Surface | LLM-first annotation layer? |
|---|---|---|
| CUDA C++ | C++ with `<<<>>>` | No |
| HLSL / GLSL | C-like shading | No |
| Slang (Khronos, 2024) | HLSL-superset + autodiff | No (autodiff, not LLM-intent) |
| Rust-GPU | Rust MIR → SPIR-V | No |
| Triton | Python tile DSL | No (tile abstraction, not intent) |
| TileLang / Gluon / ThunderKittens | CUDA-embedded / tile DSLs | No |
| IREE | MLIR → SPIR-V | No (graph-level) |
| MLIR SPIR-V dialect | Compiler plumbing | No |

No project was designed around *LLMs as primary authors* with an explicit semantic-intent annotation vocabulary over SPIR-V.

### 1.2 The opportunity

LLMs iteratively optimize CUDA kernels at the *source-text* level today (Sakana AI CUDA Engineer, Kevin, CUDA-L1, EvoEngineer, STARK, CudaForge, GPU Kernel Scientist, Simon Guo's Metamorph). Every one works on raw CUDA, with optimization intent implicit. The LLM infers `@coalesced` from the access pattern, `@cooperative_matrix` from the tile shape, `@divergence_free` from the branch structure -- but none of that inference is *verified*, which is why Sakana had reward-hacking retractions in 2025.

**AXIOM-Compute's wedge:** make the inference explicit. If the source says `@coalesced`, the compiler *verifies* coalescing or rejects the code. If the source says `@equiv_fp_tol(1e-3)`, rewrite proposals are machine-checked against the baseline. This is exactly the hole Sakana left open.

### 1.3 What AXIOM (CPU) contributes and does not

**Transfers from AXIOM:** Rust-based multi-crate compiler skeleton, lexer/parser/HIR/codegen structure, `Spanned<T>` pattern, `@annotations` as first-class data, MCP server for agent integration, `@strategy` holes for LLM-driven tuning, `@strict` with pre/postconditions, self-optimize loop, 7-agent development pipeline.

**Does NOT transfer:** `noalias` / `nsw` / `fast-math` / arena allocators / `@lifetime` heap-to-stack. These are CPU-LLVM-specific advantages. The GPU optimization vocabulary is different and ~3× larger (see §3.2).

---

## 2. Architecture

```
.axc source (annotated for LLM consumption)     <- LLM agents author here
       |
       v
Lexer (reuse AXIOM recursive descent)
       |
       v
Parser (Pratt expressions, AST)
       |
       v
HIR (GPU annotation validation, type checking, @strict enforcement)
       |
       v
MIR (lowered, auto-tune @strategy holes resolved here)
       |
       v
Codegen: dual-path
    Path A: LLVM IR -> LLVM SPIR-V backend (official Jan 2025)
            -> OpenCL-flavor SPIR-V for oneAPI / OpenCL compute
    Path B: rspirv direct -> Vulkan-flavor SPIR-V with decorations
       |
       v
spirv-val  (validator -- reject malformed IR)
spirv-opt  (peephole, constant-propagation, DCE)
       |
       v
SPIR-V binary
       |
       v
Loader/Dispatcher (ash for Vulkan, OpenCL ICD for OpenCL)
       |
       v
Benchmark + Correctness check (tolerance-aware)
       |
       v
Agent self-optimize loop (LLM proposes new @strategy values -> re-run)
```

### 2.1 Crate layout

```
axiom-compute/
├── crates/
│   ├── axc-lexer/       # Tokenizer (port AXIOM's lexer + GPU-keyword additions)
│   ├── axc-parser/      # Recursive descent + Pratt expressions
│   ├── axc-hir/         # Annotation validation, type checking, @strict enforcement
│   ├── axc-codegen/     # SPIR-V emission (rspirv direct + LLVM SPIR-V backend)
│   ├── axc-optimize/    # @strategy hole resolution, autotuner, LLM bridge
│   ├── axc-driver/      # CLI, MCP server, dispatcher, benchmark harness
│   └── axc-runtime/     # Vulkan/OpenCL loader, kernel launch wrappers
├── spec/                # Formal language spec (grammar, types, annotations)
├── examples/            # First target: llama.cpp Vulkan equivalents
├── benchmarks/          # KernelBench-Vulkan submission + llama.cpp comparisons
└── .pipeline/           # 7-agent development pipeline (adapted from AXIOM)
```

---

## 3. Language

### 3.1 M1.2 parameter binding model

#### 3.1.1 Buffer types (M1.2)

Buffer parameters are SSBO-backed arrays exposed as `buffer[T]`, `readonly_buffer[T]`,
or `writeonly_buffer[T]` in kernel parameter lists. The allowed element types in M1.2
are `i32`, `u32`, `i64`, `u64`, `f32`, and `f64`.

**Binding assignment — worked saxpy example:**

```
fn saxpy(a: f32, x: readonly_buffer[f32], y: buffer[f32]) -> void { ... }
```

Buffer params are assigned consecutive descriptor bindings in left-to-right order among
buffer parameters only (scalar params skip the binding counter):

- `x` -> descriptor binding 0  (first buffer param)  ← x -> descriptor binding 0
- `y` -> descriptor binding 1  (second buffer param)  ← y -> descriptor binding 1
- `a` -> push-constant member 0  (scalar params go to push-constant block, not descriptors)  ← a -> push-constant member 0

All buffer bindings are in **DescriptorSet 0**. The binding index equals the buffer's
0-based position among buffer parameters, not its position among all parameters.

**SPIR-V layout (one SSBO per buffer param):**

```
OpTypeRuntimeArray  %arr_T    %T
OpTypeStruct        %block    %arr_T         ; { T[] data; }
OpTypePointer       %ptr      StorageBuffer  %block
OpVariable          %var      StorageBuffer
OpDecorate          %arr_T    ArrayStride <elem_bytes>  ; 4 for f32, 8 for f64
OpDecorate          %block    Block
OpDecorate          %var      DescriptorSet 0
OpDecorate          %var      Binding <slot>
OpDecorate          %var      NonWritable    ; readonly_buffer only
OpDecorate          %var      NonReadable    ; writeonly_buffer only
```

Note: `Block` decoration (SPIR-V 1.3+) is used — NOT `BufferBlock` (deprecated).

**Interface list (SPIR-V 1.3 §2.17):** StorageBuffer and PushConstant variables
are NOT included in the OpEntryPoint interface list. Only Input/Output variables
(such as `gl_GlobalInvocationID`) must be listed. This is enforced via the
`CURRENT_SPIRV_VERSION` compile-time constant guard (AT-228).

#### 3.1.2 Scalar kernel parameters (M1.2)

Scalar parameters (`i32`, `u32`, `i64`, `u64`, `f32`, `f64`) are passed via a
single push-constant struct block. Member layout follows `std430`:

- Members are ordered by their position in the push-constant member list (i.e., the order
  of scalar params left-to-right, ignoring buffer params).
- Alignment: each member is aligned to `max(4, sizeof(T))` bytes.
  - `i32`, `u32`, `f32` → 4-byte aligned (no padding from prior 4-byte-aligned member)
  - `i64`, `u64`, `f64` → 8-byte aligned (4-byte padding after any 4-byte member)
- Member index is independent of global param position — it is only counted among scalar
  params. A `buffer[f32]` param at position 0 does not consume a member index.
- Total push-constant block size must not exceed 128 bytes (Vulkan `minPushConstantsSize`).
  Exceeding this limit produces `BindingPlanError::PushConstantTooLarge`, with the
  `overflowing_param_name` field pointing at the FIRST param that causes overflow
  (not param[0]).

#### 3.1.3 Global invocation ID (M1.2)

The builtin `gid(axis)` returns the global invocation ID component for the given axis:

```
let i: u32 = gid(0);   // X axis (typical for 1-D dispatch)
let j: u32 = gid(1);   // Y axis
let k: u32 = gid(2);   // Z axis
```

Rules:
- `axis` must be an integer literal (0, 1, or 2). A variable or out-of-range constant
  produces a `GidAxisMustBeConstant` or `GidAxisOutOfRange` typecheck error.
- Each call lowers to an `OpLoad` of the `gl_GlobalInvocationID` uvec3, followed by
  an `OpCompositeExtract` with the axis index.
- The `gl_GlobalInvocationID` `Input` variable is emitted ONCE per module regardless
  of how many times `gid()` is called. The variable's ID appears in the OpEntryPoint
  interface list (required for Input variables in SPIR-V 1.3).

### 3.1.4 Control flow (M1.3)

AXIOM-Compute lowers all control flow to SPIR-V §2.11 structured CFG:

- `if cond { ... } else { ... }` → OpSelectionMerge + OpBranchConditional
- `for i in range(start, end[, step]) { body }` → OpLoopMerge with induction-
  variable OpVariable in Function storage, header-body-continue_target-merge
  4-block shape
- `while cond { body }` → OpLoopMerge with dedicated continue_target
- `break;` → OpBranch to innermost loop's merge block
- `continue;` → OpBranch to innermost loop's continue_target

`and`/`or` short-circuit expressions are not allowed in if/while condition
position (use a temp bool). `return` inside a loop is rejected (deferred to M1.4).

### 3.1.5 Subgroup operations and workgroup barrier (M1.4)

AXIOM-Compute exposes portable subgroup/wave-level primitives via SPV_KHR_shader_subgroup_*
extensions. Ten builtin call names:

- Basic (GroupNonUniform cap, SPV_KHR_shader_subgroup_basic):
  - `subgroup_invocation_id() -> u32` → OpLoad of SubgroupLocalInvocationId
  - `subgroup_size() -> u32` → OpLoad of SubgroupSize
  - `subgroup_elect() -> bool` → OpGroupNonUniformElect
- Arithmetic (GroupNonUniformArithmetic cap, SPV_KHR_shader_subgroup_arithmetic):
  - `subgroup_reduce_add/min/max(T) -> T` → OpGroupNonUniformIAdd/FAdd with Reduce op
- Ballot (GroupNonUniformBallot cap, SPV_KHR_shader_subgroup_ballot):
  - `subgroup_broadcast_first(T) -> T` → OpGroupNonUniformBroadcastFirst
- Vote (GroupNonUniformVote cap, SPV_KHR_shader_subgroup_vote):
  - `subgroup_all(bool) -> bool` → OpGroupNonUniformAll
  - `subgroup_any(bool) -> bool` → OpGroupNonUniformAny
- Synchronization:
  - `workgroup_barrier()` → OpControlBarrier with exec=Workgroup, mem=Workgroup,
    semantics=AcquireRelease|WorkgroupMemory (0x108)

**Parent capability chain.** Every child capability implicitly requires GroupNonUniform (basic).
AXIOM-Compute mechanically forces this in the capability aggregation step to avoid spirv-val
rejection (SPIR-V §3.31).

### 3.1.6 Runtime dispatch (M1.5)

#### VulkanContext lifecycle

`VulkanContext::new()` initializes Vulkan 1.1: loads `ash::Entry`, creates an `Instance`,
selects the first physical device with a compute queue family (or the index in
`AXC_PHYSICAL_DEVICE_INDEX`), creates a logical `Device` + `Queue`, and a
`CommandPool` with `RESET_COMMAND_BUFFER`. Cached fields:
- `max_compute_work_group_count: [u32; 3]` — for dispatch pre-validation
- `memory_properties: VkPhysicalDeviceMemoryProperties` — for buffer allocation

`Drop` calls `vkDeviceWaitIdle` then destroys: CommandPool → Device → Instance.
This order is critical on Lavapipe to prevent `VK_ERROR_DEVICE_LOST` shutdown races.

#### DispatchRequest API

```rust
pub struct DispatchRequest<'a> {
    pub spirv: &'a [u32],
    pub binding_plan: &'a ParamBindingPlan,
    pub workgroups: [u32; 3],
    pub inputs: &'a [&'a [u8]],
    pub output_sizes: &'a [usize],
    pub push_constants: &'a [u8],
    pub entry_point: &'a str,
}
```

`VulkanContext::dispatch(req)` returns `Vec<Vec<u8>>` — one output per buffer binding.
All Vulkan resources (shader module, pipeline, buffers, descriptors, command buffer, fence)
are freed via `DispatchResources` RAII on both success and error paths.

#### Metadata sidecar schema v1

Written by `axc_driver::compile_file` as `<output>.axc.meta.json`. Fields:
- `schema_version: 1`
- `kernel_name: String`
- `workgroup_size: [u32; 3]`
- `binding_plan: ParamBindingPlan` (serde-enabled; Span fields skipped)
- `push_constant_total_bytes: u32`
- `entry_point: String` (always `"main"` in M1.5)

#### Host-visible memory + M2 staging-buffer plan

M1.5 allocates all buffers in `HOST_VISIBLE | HOST_COHERENT` memory. This avoids
explicit `vkFlushMappedMemoryRanges` / `vkInvalidateMappedMemoryRanges`. Mobile GPUs
that lack coherent host-visible memory will hit `DispatchError::NoCompatibleMemoryType`
until M2 adds a staging-buffer fallback path.

#### Fence timeout

Default: 10,000 ms. Override via `AXC_FENCE_TIMEOUT_MS` environment variable.

#### Push-constant byte-assembly discipline

Callers MUST iterate `binding_plan.scalars` in stored order, dispatch on `scalar.ty`,
and write `scalar.offset` bytes. Never hardcode layout. This ensures correctness if
future milestones add alignment padding or reorder scalars.

#### Workgroup-count device-limit pre-validation

Before any resource allocation, `dispatch()` checks that all three workgroup dimensions
do not exceed `VkPhysicalDeviceLimits::max_compute_work_group_count` (cached at
`VulkanContext::new()`). Returns `DispatchError::WorkgroupCountExceedsDeviceLimit`
if any dimension exceeds the limit.

#### Vulkan 1.1 subgroup capability notes

Vulkan 1.1 core REQUIRES `GroupNonUniform` + `GroupNonUniformVote` (BASIC + VOTE).
`GroupNonUniformArithmetic`, `GroupNonUniformBallot`, `GroupNonUniformShuffle`,
`GroupNonUniformClustered`, `GroupNonUniformQuad` are device-OPTIONAL. Lavapipe (Mesa 23+)
supports all. M2 adds `VulkanContext::preflight()` for real-GPU capability checks.

**Divergent-context warning.** Subgroup collective operations inside divergent control flow
(if/while bodies, but not for-range bodies since induction is uniform) emit a non-fatal
HirWarning::SubgroupOpInDivergentContext. The canonical pattern `if subgroup_elect() { ... }`
does NOT trigger at the condition position (cond runs at parent depth). Strict enforcement
deferred to M1.5.

**Subgroup ballot (`subgroup_ballot(bool) -> uvec4`) deferred to M1.5** pending uvec4 primitive type.

### 3.1.7 Benchmark harness (M2.2)

The first performance measurement layer for AXIOM-Compute is implemented in
`crates/axc-driver/benches/` using the Criterion microbenchmark framework.
Three bench groups are provided:

- `compile_pipeline` (`cargo bench --bench compile -p axc-driver`): measures
  source → SPIR-V wall time for saxpy and vector_add.
- `cpu_reference` (`cargo bench --bench cpu_reference -p axc-driver`): measures
  equivalent Rust loops at N ∈ {1024, 1M}; GPU-independent.
- `dispatch_gpu` (`cargo bench --bench dispatch -p axc-driver`): measures
  end-to-end `VulkanContext::dispatch` latency; gated on `AXC_ENABLE_GPU_BENCHES=1`.

A regression gate (`crates/axc-driver/tests/bench_regression.rs`) compares
11-sample medians against `.pipeline/benchmarks/baselines.json` with a 15%
threshold.  See `BENCHMARKS.md` for the blessed command, blessing workflow,
regression gate invocation, and CI matrix.

### 3.1.8 Q4_0 dequantization builtins (M2.5)

AXIOM-Compute adds four intrinsic builtins for efficient Q4_0 (4-bit GGUF)
weight dequantization from `buffer[u8]` SSOBs:

#### Q4_0 block layout

Each Q4_0 block is 18 bytes encoding 32 f32 elements:

```
byte 0..1:  f16 scale (little-endian IEEE 754 half-precision)
bytes 2..17: 16 packed nibble pairs
             byte k → lo nibble = weight at index k
                      hi nibble = weight at index k+16
```

Dequantization: `weight_i = (nibble_i - 8) * scale`

The bias of 8 centers the unsigned nibble range [0,15] at zero (effective signed
range [-8, 7]).

#### Four Q4_0 builtins

| Builtin | SPIR-V emission | Capabilities set |
|---|---|---|
| `ptr_read_u8_zext(buf, offset)` | OpAccessChain + OpLoad(u8) + OpUConvert(u32) | `Int8`, `StorageBuffer8BitAccess` |
| `ptr_read_u16_zext(buf, offset)` | Two u8 loads + shift + BitwiseOr into u32 | `Int8`, `StorageBuffer8BitAccess` |
| `f16_bits_to_f32(bits_u32)` | OpUConvert(u32→u16) + OpBitcast(u16→f16) + OpFConvert(f16→f32) | `Int16`, `Float16` |
| `f32_from_u32(n_u32)` | OpConvertUToF(u32→f32) | (none new) |

All four builtins are only valid for `buffer[u8]` SSBO arguments and are
lowered by `crates/axc-codegen/src/q4_0.rs`.

#### Capability side-effects

Capabilities are lazily accumulated via `CapabilitiesRequired` (the same
pattern as M2.1 cooperative-matrix caps):

- `ptr_read_u8_zext` / `ptr_read_u16_zext`: set `caps.int8` + `caps.storage_8bit`
  → emit `OpCapability Int8` + `OpCapability StorageBuffer8BitAccess` + `OpExtension "SPV_KHR_8bit_storage"`
- `f16_bits_to_f32`: set `caps.int16` + `caps.float16`
  → emit `OpCapability Int16` + `OpCapability Float16`

Additionally, if the kernel binding plan contains a `buffer[u8]` SSBO,
`StorageBuffer8BitAccess` and `Int8` are pre-enabled from the binding plan
before body emission begins (same pattern as `StorageBuffer16BitAccess`).

**SPIR-V capability numeric values** (spirv-0.3.0+sdk-1.3.268.0):
- `Int8 = 39` (NOT 40 — common off-by-one from older spec drafts)
- `Int16 = 22`
- `Float16 = 9`
- `StorageBuffer8BitAccess = 4448`
- `StorageBuffer16BitAccess = 4433`

#### Integration tests

`crates/axc-driver/tests/compile_q4_0_dequant_matvec.rs` provides AT-901 through
AT-918 (17 compile-time + 1 GPU dispatch test).  AT-918 is `#[ignore]`-gated and
requires `AXC_ENABLE_GPU_TESTS=1`.

### 3.1.10 MCP server (M2.4)

`crates/axc-driver/src/mcp/` implements a JSON-RPC 2.0 stdio bridge exposing 6 tools to LLM agents via `axc mcp [--log stderr|null]`.

#### Protocol

- **Framing:** NDJSON (one JSON object per line).
- **Inbound cap:** 8 MiB per line; oversize lines return `PARSE_ERROR (-32700)` and the server continues.
- **Notifications (§4.1):** requests with no `id` key produce no stdout output. Requests with `"id": null` respond with `"id": null`.
- **B-5 validation:** `jsonrpc != "2.0"` → `INVALID_REQUEST (-32600)`; empty `method` → `METHOD_NOT_FOUND (-32601)`.

#### Tools

| Tool | Code | Description |
|---|---|---|
| `initialize` | — | Returns `server`, `version`, `tools` list |
| `load_source` | — | Parse + HIR-lower; return kernel metadata, strategy_holes |
| `enumerate_variants` | — | Cartesian-product enumeration of `@strategy` holes |
| `compile_variant` | -32001 | Compile one variant to SPIR-V; return base64 + capabilities |
| `bench_variant` | -32004 | GPU dispatch + correctness check; return median_ns + samples |
| `grid_search` | -32003 | Run all variants, rank by median_ns, persist winner to history |
| `optimization_history` | — | Read JSONL history for a source file |

#### History file format

Each `grid_search` call appends one JSONL entry to `.pipeline/history/<xxh3_hex16(source)>.jsonl`. Concurrent writers are serialized via `flock(LOCK_EX)` (POSIX advisory lock, B-1 fix). History directory is overridden by `AXC_MCP_HISTORY_DIR` env var.

#### Base64 encoding (B-2 fix)

RFC 4648 §4 STANDARD alphabet (`A-Za-z0-9+/=`). Index 62 = `+`, index 63 = `/`. NOT URL-safe.

#### Timestamps (N-1 fix)

RFC 3339 UTC with millisecond resolution: `YYYY-MM-DDTHH:MM:SS.NNNZ` (always 24 chars). Uses Howard-Hinnant civil-time algorithm.

#### Error codes

| Constant | Code | Meaning |
|---|---|---|
| `PARSE_ERROR` | -32700 | Invalid JSON or line too long |
| `INVALID_REQUEST` | -32600 | `jsonrpc != "2.0"` |
| `METHOD_NOT_FOUND` | -32601 | Unknown or empty method |
| `INVALID_PARAMS` | -32602 | Wrong types / missing required fields |
| `COMPILE_ERROR` | -32001 | Lex/parse/HIR/codegen failure |
| `ENUMERATE_ERROR` | -32002 | No `@strategy` block |
| `GRID_SEARCH_ERROR` | -32003 | Grid search setup failure |
| `VULKAN_UNAVAILABLE` | -32004 | No Vulkan ICD / device creation failed |
| `IO_ERROR` | -32005 | History file or source file I/O |
| `SPIRV_VAL_FAILED` | -32006 | In-process spirv-tools rejected SPIR-V |

#### Acceptance tests

AT-1101 through AT-1132 in `crates/axc-driver/tests/mcp_roundtrip.rs`. GPU-execution tests (AT-1114 through AT-1117) are gated behind `AXC_ENABLE_GPU_TESTS=1`.

### 3.1.11 Q4_K_M superblock layout and dequant kernel (M2.6)

Q4_K_M is the production quantization scheme for mainstream LLM deployments (Meta Llama-3-70B,
Mistral-7B-v0.3, Qwen2-72B, Phi-4, and most GGUF checkpoints distributed at "Q4" precision).
M2.6 ships the first AXIOM-Compute kernel matching this format.

#### Block byte layout (144 bytes per 256 output elements)

```c
struct block_q4_K {
    ggml_half d;           // super-scale,  bytes [0..2]  (IEEE-754 binary16 LE)
    ggml_half dmin;        // super-min,    bytes [2..4]  (IEEE-754 binary16 LE)
    uint8_t   scales[12];  // packed scales, bytes [4..16]  (96 bits = 8 × 12-bit (sc,m) pairs)
    uint8_t   qs[128];     // 4-bit weights, bytes [16..144] (256 weights, 2 nibbles per byte)
};  // total: 2 + 2 + 12 + 128 = 144 bytes per 256 elements
```

#### Bit-spread unpacking of the 12-byte scales region

The 12 bytes encode 8 × (6-bit scale + 6-bit min) = 96 bits.  The canonical ggml
`get_scale_min_k4(j, scales, &sc, &m)` decoding for j ∈ [0, 8):

```c
if (j < 4) {
    sc = scales[j]     & 63;   // low 6 bits of scales[j]
    m  = scales[j + 4] & 63;   // low 6 bits of scales[j+4]
} else {
    sc = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4);  // 4-bit lo from scales[j+4], 2-bit hi from scales[j-4]
    m  = (scales[j+4] >> 4)   | ((scales[j]   >> 6) << 4);  // 4-bit lo from scales[j+4], 2-bit hi from scales[j]
    //                            ^^^^^^^^^^^^^^^^
    //   CRITICAL: m-high uses scales[j] (q[j-0]), NOT scales[j-4] (q[j-4]).
    //   q[4..7]'s UPPER 2 bits alias to m's high-2 for j ∈ {4,5,6,7}.
    //   This is canonical ggml bit-spread packing, not a typo.
}
```

The AXIOM-Compute kernel inlines this as an imperative `if j < 4u32 { ... } else { ... }` block
using `band` / `lshr` / `bor` / `shl` builtins from M2.5.

#### Dequantization formula (four-chunk iteration)

```c
// Four chunks of 64 elements each (4 × 64 = 256 per superblock).
// Each chunk processes sub-blocks [is, is+1] where is = chunk * 2.
for chunk in 0..4 {
    get_scale_min_k4(is,   scales, &sc0, &m0); d1 = d * sc0; m1f = dmin * m0;
    get_scale_min_k4(is+1, scales, &sc1, &m1); d2 = d * sc1; m2f = dmin * m1;
    for l in 0..32 {
        byte = qs[chunk*32 + l];
        lo_nibble = byte & 0x0F;        hi_nibble = byte >> 4;
        // NO -8 offset (unlike Q4_0): Q4_K_M uses unsigned 4-bit nibbles.
        y[chunk*64 + l]      = d1 * lo_nibble - m1f;
        y[chunk*64 + 32 + l] = d2 * hi_nibble - m2f;
    }
}
```

x-vector indexing: `x[sb*256 + chunk*64 + l]` for lo-nibble outputs (offset 0..32),
`x[sb*256 + chunk*64 + 32 + l]` for hi-nibble outputs (offset 32..64).

#### SPIR-V capabilities (same as M2.5 Q4_0, zero additions)

| Capability | Value | Reason |
|---|---|---|
| Int8 | 39 | u8 SSBO access for scales/qs bytes |
| Int16 | 22 | intermediate u16 in f16_bits_to_f32 |
| Float16 | 9 | OpBitcast u16→f16 + OpFConvert f16→f32 |
| StorageBuffer8BitAccess | 4448 | u8 SSBO loads (SPV_KHR_8bit_storage) |

StorageBuffer16BitAccess is **NOT** required: d and dmin are loaded as two u8 loads
via ptr_read_u16_zext, never as native f16 SSBO loads.

#### Tolerance

`@equiv_fp_tol(1e-3)` relative tolerance, matching two f16 roundings (d and dmin,
each ~2^-11 relative error) plus FMA divergence across vendors.  If Lavapipe runs
prove marginal, widen to 2e-3 with documented rationale (advisory, not blocking per
M2.6-design-review.json §advisory_notes).

#### Integration tests (M2.6)

AT-1301..AT-1331 in `crates/axc-driver/tests/compile_q4km_dequant_matvec.rs`.
GPU dispatch tests (AT-1324 Lavapipe, AT-1331 NVIDIA) are gated on `AXC_ENABLE_GPU_TESTS=1`.
Bench group `dispatch_gpu_q4km` in `crates/axc-driver/benches/dispatch_q4km.rs`
(dispatch_q4km_128 + dispatch_q4km_512).

---

## 3.1.12 M3.0 — Dispatch bandwidth rework (runtime, no codegen change) [revision r1]

M3.0 eliminates the staging-bound per-dispatch overhead that made `dispatch_saxpy_1m` cost 23 ms and `dispatch_q4km_512` cost 8.84 ms on the NVIDIA RTX PRO 6000 (~100x off PCIe peak). It is a pure **data-movement** change: SPIR-V codegen, the `.axc` language, and `spirv-val` are untouched, so the correctness oracle is **bit-exact preservation** against the existing CPU references AND **byte-equality across four execution paths** (single-queue coherent / dedicated-queue / forced-non-coherent / forced-binary-semaphore).

### Four layered levers (each falls back cleanly to prior behavior)

1. **Persistent-mapped staging.** Each HOST_VISIBLE staging allocation is `vkMapMemory`'d exactly once at allocation time; the raw host pointer is PRIVATE inside `PersistentMapping`, accessed only via guarded `copy_in`/`copy_out` under the per-handle Mutex. `PersistentMapping` is `Send`-only (never `Sync`). **Unmap-before-free** is mandatory: on buffer-pool growth and on `KernelHandleInner::drop`, each staging memory is `vkUnmapMemory`'d BEFORE `vkDestroyBuffer`/`vkFreeMemory`; `unmap` consumes `self` so a forgotten unmap is a compile error.

2. **HOST_CACHED staging + flush/invalidate.** Staging prefers `HOST_VISIBLE|HOST_CACHED|HOST_COHERENT`, then `HOST_VISIBLE|HOST_CACHED` (non-coherent), then `HOST_VISIBLE|HOST_COHERENT` (prior behavior; Lavapipe/iGPU). Cached host **reads** are ~100x faster than write-combined coherent reads — the single largest contributor to the 23 ms readback. On non-coherent memory, `vkFlushMappedMemoryRanges` fires after EACH upload slot (before submit) and `vkInvalidateMappedMemoryRanges` before EACH readback slot (after the fence). Ranges use `offset=0` and `size = align_up(len, nonCoherentAtomSize)` or `VK_WHOLE_SIZE` when that meets/exceeds the allocation (no clamp-to-alloc bug). On coherent memory both are no-ops (zero Lavapipe regression). **Host-visibility on readback is path-identical:** the device->staging copy is ALWAYS followed by an in-stream TRANSFER_WRITE->HOST_READ barrier (dstStage=HOST) on every path, and on non-coherent memory the host invalidate is ALSO unconditional — neither relies on the fence-signal host-domain guarantee as a substitute.

3. **Dedicated transfer queue.** When the device exposes a queue family with TRANSFER but **not** COMPUTE/GRAPHICS (the discrete DMA copy engine), the context acquires it plus its own command pool. In this mode the **device-local buffer is created `VK_SHARING_MODE_CONCURRENT`** over `{compute_family, transfer_family}`, so there are **no queue-family ownership-transfer barriers anywhere** — the canonical NVIDIA-passes/AMD-corrupts release/acquire-range-mismatch failure class is structurally eliminated. (EXCLUSIVE+ownership-transfer is a possible M3.1 optimization once real AMD/Intel CI can validate the barrier pairs.) Cross-queue visibility uses plain `QUEUE_FAMILY_IGNORED` memory barriers + semaphore execution dependencies. When no distinct family exists (Lavapipe), the context uses `QueueMode::SingleQueue` with the **identical** command stream to M2.3a (EXCLUSIVE buffers, one CB, one queue).

4. **Transfer/compute overlap via timeline semaphores.** The dedicated path splits each dispatch into three submits — transfer upload -> compute -> transfer readback — synchronized by a **timeline semaphore** (Vulkan 1.2 / `VK_KHR_timeline_semaphore`, **primary**; monotonic per-dispatch values `B+1`,`B+2` reserved atomically, so there is NO reuse hazard across dispatches/errors/skipped stages) with a **binary-semaphore fallback** for 1.1-only ICDs (safe via the per-handle Mutex + mandatory host fence wait; binaries recreated on mid-chain error). The host still blocks on a single `readback_fence`. A context-level `queue_submit_lock` serializes the **entire three-submit group of one dispatch** (atomic group), released before any host wait, which makes cross-handle FIFO-queue interleaving impossible and the wait-for graph acyclic (deadlock-free); per-pool `command_pool_lock`s guard shared-pool alloc/free.

### Error recovery
The three-submit dedicated chain has a fully specified partial-failure matrix: on any mid-chain submit failure the host calls `device_wait_idle()`, frees every allocated CB from its OWN pool (transfer/compute) under the pool locks, recreates binary semaphores (Binary mode only), and returns the typed error WITHOUT waiting a never-submitted fence — no CB leak, no hang.

### Acceptance honesty — coherent-hardware coverage + bounded gap
The dedicated-queue and non-coherent paths are only naturally exercised on coherent NVIDIA + coherent single-queue Lavapipe. To make their risk visible, the three force-options (`force_single_queue`, `force_binary_semaphores`, `force_noncoherent_staging`) are FIRST-CLASS so CI EXECUTES the binary-semaphore and flush/invalidate code paths even on coherent hardware (semantic no-ops, but the code runs and the byte-exact oracle still holds). The central test **AT-1418** runs the same kernel through all four configurations and asserts byte-identical output to the CPU reference, so a missing invalidate/flush or a sync bug surfaces as divergence on the dev box. **Bounded, acknowledged gap:** TRUE non-coherent-hardware validation (where a dropped invalidate genuinely corrupts) requires **EB.1 cross-vendor CI (AMD/Intel)**, which is **OUT of M3.0 scope**. M3.0 proves code-path execution + byte-exact equality on coherent hardware; EB.1 proves non-coherent semantics on real hardware.

### VK_KHR_buffer_device_address — deferred to M3.1; VK_EXT_host_memory_alloc_placement — out of scope
BDA would change the **shader** (descriptor-bound SSBO -> 64-bit `PhysicalStorageBuffer` pointer, requiring `PhysicalStorageBufferAddresses`, codegen, and a `@target` declaration), violating M3.0's no-codegen invariant, and it does not address the M3.0 bottleneck. Deferred to M3.1 (tiled matmul + cooperative_matrix), which already touches codegen. **VK_EXT_host_memory_alloc_placement** (import-host-pointer) is consciously scoped OUT in favor of the HOST_CACHED ladder, which reaches the gate; revisit only if a target vendor lacks HOST_CACHED.

### Acceptance gates
- `dispatch_saxpy_1m` < **1 ms** and `dispatch_q4km_512` < **2 ms** on NVIDIA RTX PRO 6000 (bench IDs unchanged).
- All existing dispatch + Q4_K_M bit-exact tests pass; the four force-option paths (AT-1418) produce byte-identical output.
- Lavapipe single-queue fallback shows no regression; `cargo test --workspace` (+`--ignored`) green; `clippy -D warnings` clean; `spirv-val` clean.

### New/changed error variants
`DispatchError` grows to 28 variants: `SemaphoreCreationFailed`, `TransferQueueSubmitFailed`, and `MappedRangeOpFailed { op: MappedRangeOp }` (`MappedRangeOp in {Flush, Invalidate}`).

### 3.1.12 (r2 postmortem) — Readback profiling + the ReBAR negative result

r1 cut `dispatch_saxpy_1m` from 23 ms to 3.08 ms but MISSED the <1 ms gate. A profiling pass on the residual 3.08 ms isolated the bottleneck precisely, an r2 fix (ReBAR zero-copy readback) was attempted, and it was **empirically reverted** — the gate was re-scoped. The profiling table and the lessons are kept here as the record:

| phase | time | % |
|---|---|---|
| copy_in (host→staging, 8 MB cached writes) | 252 µs (~31 GB/s) | 9% |
| GPU timeline (upload DMA + compute + readback DMA) | 750 µs | 27% |
| **copy_out (staging→host, 8 MB reads)** | **1730 µs (~4.5 GB/s)** | **62% — bottleneck** |
| fixed (CB alloc/free, locks) | ~31 µs | 1% |

**Root cause of the slow copy_out:** with HOST_CACHED staging the host *write* side (copy_in) is fast, but the host *read* side is still slow because the GPU DMAs results into system-RAM staging pages, so the CPU read is cold-cache and PCIe-snoop-limited at ~4.5 GB/s. Two compounding problems: (1) read-only INPUT bindings were being read back needlessly (saxpy's `x` is `readonly` yet 4 MB of it was copied device→host — half of copy_out); (2) even the legitimate 4 MB output is slow because it lives in system-RAM staging.

**Lever A — skip readback of read-only bindings (sound; folded into M3.1).** Readback should be driven by the binding plan's `BufferAccess`: a binding whose access is `readonly` need never be read back, since it carries the SPIR-V `NonWritable` decoration and the shader provably cannot write it. The r1 saxpy bench echoed back its 4 MB read-only `x` input needlessly. This is a correct, vendor-independent win, but it was implemented as part of the r2 patch that was reverted (see below), so it is re-scheduled cleanly for M3.1 rather than shipped half-tested.

**Lever B — zero-copy readback via ReBAR: TRIED AND REVERTED (empirical negative result).** The hypothesis was that allocating the output storage buffer in `DEVICE_LOCAL | HOST_VISIBLE` (ReBAR) memory and reading it directly on the host would eliminate the slow staging readback. **Measured on the RTX PRO 6000, this made `dispatch_saxpy_1m` ~60–70× SLOWER (3.08 ms → ~200 ms).** Root cause: **CPU reads from the ReBAR/BAR aperture are pathologically slow (~20 MB/s here).** BAR-mapped device memory is write-combined — optimized for CPU *writes* into VRAM, not CPU *reads* out of it. The profiling projection of "~60 GB/s ReBAR read" was wrong; real hardware falsified it. The 7-agent pipeline caught the regression at the real-GPU measurement gate, **before merge**, and r2 was reverted in full. **Lesson recorded:** ReBAR is a tool for *upload* (host→VRAM), never for *readback* (VRAM→host).

**The <1 ms gate is the wrong metric for a host-round-trip workload.** `dispatch_saxpy_1m` round-trips ~12 MB host→device→host on every call; its time is dominated by PCIe transfer + readback, not kernel quality. r1's irreducible cost is ~252 µs copy_in + ~750 µs GPU timeline + readback — even *free* readback lands near ~1 ms, and readback itself is PCIe/snoop-bound on either staging or BAR. Real LLM inference keeps weights *resident* in VRAM and never round-trips per dispatch. **The <1 ms target is therefore re-scoped to a GPU-resident benchmark (upload once, dispatch N times, measure kernel time) in M3.1/M3.4**, which is the metric that actually reflects the llama.cpp thesis. The `q4km_512 < 2 ms` gate is likewise re-scoped to M3.1's multi-row matmul (its 5.76 ms is 2 MB x-vector upload + single-row matvec compute, not readback).

**M3.0 shipped outcome (r1).** The merged M3.0 is the r1 implementation: persistent-mapped HOST_CACHED staging + optional dedicated transfer queue + timeline/binary-semaphore overlap, single-queue fallback byte-identical to M2.3a. Measured on the RTX PRO 6000: `dispatch_saxpy_1m` 23 ms → **3.08 ms (7.5×)**, `dispatch_saxpy_1024` 1.22 ms → **31 µs (39×)**, `dispatch_q4km_512` 8.84 ms → **5.76 ms (1.5×)**. All paths byte-exact vs CPU reference (AT-1418 four-config oracle). This is a substantial, correct bandwidth reduction; the remaining gap to "peak" is host-round-trip PCIe cost that only a resident-buffer methodology removes.

---

## 3.1.13 M3.1 — Cooperative-matrix dispatch + multi-row matmul + GPU-resident benchmark

M3.1 makes the M2.1 cooperative-matrix codegen EXECUTE on real tensor cores for the first time, adds a multi-row Q4_K_M matmul, and builds the GPU-resident benchmark methodology that M3.0's `<1 ms` gate was re-scoped onto (§3.1.12 r2). The emitted coopmat SPIR-V is UNCHANGED from M2.1; M3.1 is a RUNTIME + metadata + kernel + benchmark milestone. The only codegen touch is a mechanical internal-cache-key extension (CoopMatKey gains K + result_type) that leaves the emitted module byte-identical.

### Coopmat shape metadata (HIR → sidecar → runtime)

Previously the runtime had no way to know a kernel used cooperative matrices or what shape it required: `@cooperative_matrix` was a bare bool, `CoopMatKey` carried no K and no result type, and the metadata sidecar carried nothing coopmat. M3.1 plumbs the shape end-to-end. The HIR derives a kernel-level `CoopMatShape{m,n,k,a/b/c/result element types, scope}` from the body's `matrix[T,M,N,use]` types of the coopmat ops (reusing the M2.1 type-check guarantees, so K = a.n = b.m). The metadata sidecar gains a `coopmat: Option<CoopMatShapeMeta>` field; `CURRENT_SCHEMA_VERSION` is bumped 1→2 and `load_kernel_metadata` accepts both versions (a v1 sidecar with no field deserializes to `None` via `#[serde(default)]`, which is dispatch-identical to a non-coopmat kernel). The runtime builds the required shape FROM the metadata — nothing is hardcoded.

### Device-feature enablement: the capability→feature pass

spirv-val validates a module but does NOT check that the device features its capabilities require are enabled — passing spirv-val does NOT imply `vkCreateComputePipelines` will succeed. M3.1 adds a binding-plan-driven `required_device_features` pass mapping each EMITTED SPIR-V capability to the Vulkan device feature the runtime must enable:

| SPIR-V capability | trigger | Vulkan device feature |
|---|---|---|
| VulkanMemoryModel | coopmat op used | VkPhysicalDeviceVulkanMemoryModelFeatures.vulkanMemoryModel (+DeviceScope) |
| CooperativeMatrixKHR | coopmat op used | VK_KHR_cooperative_matrix ext + VkPhysicalDeviceCooperativeMatrixFeaturesKHR.cooperativeMatrix |
| StorageBuffer16BitAccess | any f16 SSBO | VkPhysicalDevice16BitStorageFeatures.storageBuffer16BitAccess |
| StorageBuffer8BitAccess | any u8/i8 SSBO | VkPhysicalDevice8BitStorageFeatures.storageBuffer8BitAccess |
| Int8 | u8/i8 SSBO / ptr_read_u8 | VkPhysicalDeviceShaderFloat16Int8Features.shaderInt8 |
| Int16 | f16_bits_to_f32 | VkPhysicalDeviceFeatures.shaderInt16 |

Matmul_tile is the FIRST f16-SSBO kernel ever dispatched (it needs `storageBuffer16BitAccess` on top of the memory-model + coopmat features); the multi-row q4km kernel is the first 8-bit-storage + Int8/Int16 dispatch. The runtime enables the device-SUPPORTED subset of this superset at context creation (all feature-struct locals at function scope so they outlive `create_device`, chain assembled unconditionally), and a kernel that needs a feature the device lacks FAILS CLOSED with the typed `DeviceFeatureUnsupported` skip — never enable-and-hope.

### Preflight, subgroup guard, graceful skip

A preflight queries `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` for the device's supported `(M,N,K,AType,BType,CType,ResultType,scope,saturatingAccumulation)` tuples and the `cooperativeMatrix` feature. A pure matcher checks the metadata-derived required shape (16×16×16 f16, Subgroup, non-saturating) against that set. Because matmul_tile is `@workgroup(32,1,1)` and assumes one 32-lane subgroup == one tile, the dispatch path additionally REQUIRES `subgroupSize == 32`; a wave64/SIMD16 device that advertises the shape is SKIPPED (not miscomputed). When unsupported — Lavapipe, software, wrong subgroup size, or the `force_no_coopmat` CI option — the dispatch returns `DispatchError::CoopMatUnsupported` and tests/benches SKIP cleanly. The same single `matmul_tile.axc` SKIPS on Lavapipe and DISPATCHES on NVIDIA Blackwell — the preflight, not the source, decides. The NVIDIA runner is a MANDATORY sign-off gate: an all-device skip of the first-dispatch proof is a milestone FAIL.

### Multi-row Q4_K_M matmul (real workload) — staged honestly

`q4km_dequant_matmul.axc` extends M2.6's single-row matvec to an N-row × M-col output, each invocation computing one output via the proven Q4_K_M dequant dot-product, bit-exact vs a CPU reference for a 256×256 fixture, on Lavapipe AND NVIDIA (a regular, non-ignored GPU test). The TRUE single-kernel `dequant → shared-f16-tile → coopmat` fusion needs `shared[T,N]` (FG.6, deferred to M3.2), so M3.1 delivers two honest pieces: the plain multi-row matmul (the bit-exact deliverable, runs everywhere) plus a pre-dequantized-input coopmat bridge (`q4km_dequant_matmul_coopmat.axc`, NVIDIA-only proof that coopmat runs on a Q4_K_M-derived workload).

### GPU-resident benchmark methodology

The thesis-relevant metric is **upload weights ONCE to resident VRAM, dispatch N times, measure KERNEL time** — exactly how llama.cpp inference works. M3.1 adds `upload_resident` / `dispatch_resident` / `readback_resident`: inputs are staged once, the descriptor set bound to resident device-local buffers, and each iteration records ONLY the compute submit, timed by `vkCmdWriteTimestamp` × 2 (`elapsed = (end−begin) · timestampPeriod`, each endpoint masked to `timestampValidBits` before subtraction), with a CPU fence-wall fallback recorded in `ResidentDispatchTiming.timing_source`. The N-loop reuses one command buffer, fence, and query pool. The 4096×4096 f32 resident matmul reports **effective TFLOPS** (the honest, reproducible in-tree number); the `≥50%-of-cuBLAS` comparison is an external estimate, labeled as such (the project is CUDA-free by thesis) — only effective TFLOPS is asserted.

### Per-vendor @strategy tile holes and Lever A re-land

Tile dimensions are ordinary M2.3 `@strategy` holes; for coopmat variants each candidate is VALIDATED against the preflight supported set (preflight DRIVES selection). Lever A is re-landed cleanly (without the reverted ReBAR Lever B): a binding is read back iff `output_size > 0 && access != ReadOnly`, applied uniformly across all four readback loops; read-only inputs skip the device→staging copy, the HOST_READ barrier, and the invalidate.

### 3.1.13 (M3.1.5) — resident benchmark COMPLETED + honest reframe

M3.1 shipped the resident benchmark as a typed skeleton (struct/Drop/timestamp-masking) with the three `*_resident` methods stubbed. **M3.1.5 implements them** and corrects an honesty overreach in the paragraph above. Measured on the NVIDIA RTX PRO 6000 (all `GpuTimestamp`, zero `CpuFenceWall`):

| kernel | resident kernel-only span | host-round-trip (dispatch_handle) |
|---|---|---|
| saxpy 1M | **~2.0 µs** | 3.08 ms |
| q4km matvec | **~131 µs** | 5.47 ms (q4km_512) |
| q4km matmul (256-out) | **~1.9 µs** | — |

This is the milestone's real result: the saxpy_1m "3.08 ms" from M3.0 is **~99.9 % PCIe transfer, ~2 µs actual kernel** — fully validating the M3.0 decision to re-scope the `<1 ms` host-round-trip gate to this resident metric. Timing semantics are labeled honestly: the `TOP→BOTTOM` span is *GPU dispatch duration* (excludes host upload/readback, **includes** dispatch launch + pipeline latency), NOT isolated ALU time; warmup-discard is mandatory (2 discarded, min-of-N); `CpuFenceWall` (Lavapipe / `timestampValidBits==0`) is flagged and never quoted as a GPU kernel time (no query pool or `vkCmdWriteTimestamp` is recorded there).

**Correction to the paragraph above:** there is **no honest competitive 4096² matmul TFLOPS in M3.1/M3.1.5.** `matmul_f32_tiled.axc` had inert `@strategy{tile_m,tile_n}` holes (never referenced in the body — removed in M3.1.5), so it is a *naive un-tiled scalar GEMM*. Its number is reported only via the `naive_gemm_harness_validation` bench, explicitly labeled "NAIVE … harness validation only … NOT an optimized matmul … real tiling is M3.2" — measured **4.5 TFLOPS ≈ 3.6 % of cuBLAS f32 (datasheet ESTIMATE, not a same-machine A/B)**. A *competitive* tiled/coopmat matmul TFLOPS needs `shared[T,N]` workgroup memory (FG.6) and is **deferred to M3.2**. The dispatch-time coopmat/feature preflight is wired via an additive `prepare_kernel_checked`; raw `prepare_kernel` remains an opt-in gap (M3.2 carry-forward).
---

## 3.1.14 M3.2 — shared[T,N] workgroup memory + competitive tiled matmul + tiled attention (NOT FlashAttention-2)

M3.2 adds the first NEW LANGUAGE FEATURE since M2.1: `shared[T,N]` workgroup-local memory (FG.6), threaded lexer->parser->HIR->typecheck->codegen->spirv-val->GPU. On top of it: a competitive shared-staged + cooperative-matrix tiled matmul (the dequant->shared-f16-tile->coopmat fusion carried forward from M3.1.5), and a STAGED attention whose correctness-first NON-streaming half (C1, `tiled_attention`) lands here while the FlashAttention-2 streaming-online-softmax half (C2) defers to M3.2b.

### shared[T,N] — syntax, lowering, barrier semantics

Source: a body-level declaration `shared tile: shared[f32, 256];`. The leading `shared` keyword (lexed since M1.2) introduces the declaration; the type `shared[elem, N]` carries a scalar element type and a compile-time **unsuffixed positive integer literal** N (anti-pattern #1: no inference, no const-expr fold in M3.2). A shared array is KERNEL-WIDE workgroup storage shared by all invocations — NOT a parameter, consumes NO descriptor; a shared type in parameter position is a hard error.

Lowering: parser -> `Stmt::SharedDecl` + `TypeRef::Shared(elem,N)`; lower.rs collects each into `KernelBodyTyped.shared` with a monotonic `SharedId`, validating N>0, N<=65536, allowed element type (f32/f16/f64/i8/u8/i32/u32/i64/u64 — Bool rejected), unique name, no param/binding collision, aggregate-bytes limits (warn >16384 portable min, error >65536). typecheck resolves `name[i]`->`SharedRead` (index must be U32, no coercion) and `name[i]=v`->`SharedWrite` (value elem type exact).

SPIR-V codegen (emit_shared_globals, before begin_block, like SSBOs):
```
%elem    = <scalar type>
%n_const = OpConstant %u32 256          ; length is an OpConstant id (SPIR-V 3.32.6), NOT a literal
%arr     = OpTypeArray %elem %n_const   ; SIZED array (not RuntimeArray)
%ptr_wg  = OpTypePointer Workgroup %arr
%var     = OpVariable %ptr_wg Workgroup ; global, no initializer
%ptr_e   = OpTypePointer Workgroup %elem ; shared across arrays of the same elem type
```
Indexed access uses a SINGLE-index access chain `OpAccessChain %ptr_e %var %idx` (contrast the SSBO two-index `%var %0 %idx` — the shared array has no wrapping struct). The Workgroup StorageClass requires **no new capability or extension** (core Shader); f16/i8/i16/i64/f64 elements pull in only the corresponding Float16/Int8/Int16/Int64/Float64 type capability (set explicitly in emit_shared_globals — observe_type does NOT cover F16) — explicitly NOT the StorageBuffer8/16BitAccess KHR storage extensions (those gate the StorageBuffer class only). Workgroup vars are excluded from the SPIR-V 1.3 OpEntryPoint interface list (forward-compat: 1.4+ lists all globals; repo pins 1.3). Two distinct shared arrays of the same elem type get distinct OpTypeArray+OpVariable ids but share the elem-ptr type (dedup must not collapse them). The Workgroup OpVariable is uninitialized (Vulkan forbids initializers without VK_KHR_zero_initialize_workgroup_memory) — reading a never-written cell reads garbage; correctness rests on write-before-read + barrier discipline.

### REQUIRED barrier + memory semantics (the correctness hazard)

A cross-invocation read-after-write through shared memory is correct ONLY with a workgroup control barrier carrying WorkgroupMemory semantics between the write and read phases. AXIOM-Compute reuses `workgroup_barrier()` verbatim (verified in codegen/src/subgroup.rs):
```
OpControlBarrier %Workgroup(2) %Workgroup(2) %(AcquireRelease|WorkgroupMemory = 0x108)
```
Execution scope Workgroup(2) => all invocations rendezvous; WorkgroupMemory(0x100) => prior shared writes visible to subsequent shared reads across invocations. A single barrier suffices for the straight-line stage->barrier->read pattern; the LOOPED matmul needs TWO barriers per K-iteration (write-visibility + WAR before the next iteration overwrites the staging arrays).

**Missing-barrier policy (sound, SET-based: provably-other slot -> hard error; provable self-read -> no diagnostic; undecidable -> warning).** A barrier is required only for a CROSS-invocation read — one invocation reading a slot a DIFFERENT invocation wrote. An invocation always observes its OWN prior writes in program order, so a same-invocation self read-after-write (`tile[i]=v; ...=tile[i];`, AND the multi-slot `tile[0]=a; tile[1]=b; let w=tile[0];`) needs NO barrier and must never be flagged. The compiler tracks the SET W_X of ALL prior SharedWrite indices to id X in the SAME basic block since the last barrier (a barrier clears the set). For a later SharedRead `S[r]` of X with no intervening barrier and no branch/back-edge: if `r` is PROVABLY EQUAL to ANY prior write index in W_X it emits NO diagnostic (correct self-RAW — the invocation reads a slot it itself wrote); else if `r` is PROVABLY DISEQUAL to EVERY prior write index in W_X (a provably-other slot — distinct constants, or e.g. `S[local_id]` vs `S[bxor(local_id,1)]`, or a read of a never-written slot) it emits a HARD ERROR (`SharedMissingBarrierBeforeCrossInvocationRead`) — the canonical NVIDIA-passes/AMD-races trap, decidable with zero false positives; else (the relation is UNDECIDABLE against at least one prior write — dynamic/arithmetic index, a reassigned binding, or the write and read are in different blocks / loop-carried across a back-edge / behind a divergent branch) it emits only an advisory warning (`SharedWriteWithoutBarrierBeforeRead`). Provable index equality/disequality is decided structurally (same LocalRead binding with no intervening reassignment, equal IntLit, same/distinct GidBuiltin axis). Pairing against the SET of prior writes (not just the most-recent) is what avoids false-positiving the multi-slot self-write. The hard error fires on NO correct kernel; the one residual case it rejects — a single-invocation kernel reading a slot NOBODY wrote — is an uninitialized read, buggy regardless of any barrier (only the diagnostic name is slightly off). The compiler does NOT auto-insert barriers.

**Divergent-barrier policy (hard error).** `workgroup_barrier()` inside an `if`/`else` body that not all invocations provably enter is UB in Vulkan and is a HARD ERROR (`BarrierInDivergentContext`). A barrier inside a `for-range` or workgroup-uniform `while` loop body (entered by all invocations) is accepted. The distinction is made by a dedicated `conditional_depth` counter that increments ONLY at if-then/else bodies — NOT at while or for-range — so a barrier in a uniform loop body (the matmul/attention K-loop) is correctly permitted while a conditional-branch barrier is rejected. (The pre-existing `divergent_context_depth`, which while DOES increment, continues to gate the separate subgroup-collective-op warning and is left unchanged.)

**Cross-vendor race honesty (bounded gap).** A missing/insufficient WorkgroupMemory barrier passes on NVIDIA (subgroup-32 lockstep) AND on Lavapipe (serial CPU emulation) but RACES on AMD wave64 / Intel where a workgroup spans independent SIMD groups. The barrier-visibility oracle (examples/shared_reduce.axc, AT-1606) therefore proves only (i) bit-exactness WITH the barrier and (ii) that the diagnostic fires for the barrier-absent variant — it CANNOT prove the barrier is load-bearing on any available hardware. The barrier semantics are provably correct on paper and verified in subgroup.rs; the empirical cross-vendor race-detection that would catch a Coder's missing-barrier bug requires AMD/Intel hardware the project does not yet have (EB.1). The REAL in-CI protection is the decidable-case HARD ERROR above, NOT the test. This gap is untested, NOT implied-covered.

### Competitive tiled matmul (shared-staged + coopmat)

Each workgroup computes a TILE_M×TILE_N output tile. Across K in TILE_K steps: cooperatively stage A/B (or DEQUANTIZED Q4_K_M weights) into `shared[f16,...]` tiles -> workgroup_barrier() -> coopmat_load FROM shared -> coopmat_mul_add accumulate -> barrier (WAR) -> next K-tile. After K, coopmat_store the accumulator. The coopmat-from-shared load uses a SEPARATE emit path: a Workgroup shared array is NOT struct-wrapped, so its access chain is SINGLE-index (`%var %offset`) against a Workgroup-class pointer, in contrast to the SSBO TWO-index chain (`%var %0 %offset`). This is a distinct function (`emit_coopmat_load_shared_inline`), leaving the M2.1 Buffer-source coopmat emission byte-identical (`CoopMatLoadSource::Buffer` is the default discriminator). The coopmat op itself is storage-class-agnostic; only the access-chain pointer type and index arity differ. The shared-tile row stride is in ELEMENTS of the shared row (differs from the SSBO row stride). The `@strategy { tile_m, tile_n, tile_k }` holes are GENUINELY used: resolved by M2.3 source-text substitution BEFORE the parser, so different tile_k yields different literal shared-array sizes AND different K-loop bounds AND bit-exact GPU results for both values (AT-1622 — the explicit fix for the M3.1.5 inert-holes mistake; size-diff alone is necessary but not sufficient). Effective-TFLOPS for 4096² is measured via the resident benchmark and compared to a LABELED cuBLAS datasheet ESTIMATE (not a same-machine A/B); un-padded shared tiles may bank-conflict on the 32-bank model, so sub-peak numbers are expected (the @shared_tile(pad=K) fix is deferred). A pure-f32 shared-staged variant runs + is bit-exact on Lavapipe (the coopmat variant graceful-skips there); the f32 variant does NOT exercise the shared-source coopmat path, which is NVIDIA-only-tested.

### Tiled attention — staged (C1 lands as `tiled_attention`; FlashAttention-2 defers to C2)

C1 (lands in M3.2, named `tiled_attention`): a correctness-first NON-streaming attention (examples/tiled_attention.axc) with K/V head-dim tiles staged into shared[f16,...], FULL two-pass softmax over the K dimension (with max-subtraction for numerical stability — that is just correct softmax), bit-exact within @equiv_fp_tol(1e-3) vs a CPU reference attention, on Lavapipe + NVIDIA. **This is the bit-exact baseline; it is NOT FlashAttention-2.** FlashAttention-2's defining contribution is the block-streaming ONLINE softmax (running max m_i, running denominator l_i, output rescaling) that AVOIDS materializing the SxS score block — C1 materializes the score row (fine at seq_len=64, does not scale to the FA2 regime). C2 (DEFERRED to M3.2b, earns the flash_attention_v2 name): the streaming online softmax and the >=80%-of-FA3 TFLOPS aspiration — deferred because its online-rescale arithmetic is a separate high-risk bit-exactness surface that would overrun this milestone stacked on shared[T,N]+matmul, and because C1 first establishes the bit-exact baseline against which C2's streaming optimization is @equiv_fp_tol-verified (the project's anti-reward-hacking discipline).

### Runtime preflight

Metadata schema bumps v2->v3 with an additive `shared_memory_bytes` field; the version guard becomes an explicit allowed-set {1,2,3} so v1 (pre-coopmat) and v2 (coopmat) sidecars still load (shared_memory_bytes defaults to 0). The max_compute_shared_memory_size device limit is cached on VulkanContext at init; prepare_kernel_checked threads the kernel's shared_memory_bytes into preflight_kernel_support, which fail-closes with DispatchError::SharedMemoryExceedsDeviceLimit (graceful skip) when a kernel requests more shared memory than the device offers — mirroring the M3.1 coopmat/feature preflight. Raw prepare_kernel stays unchecked (documented opt-in gap).

*(Changelog: 2026-06-02 — M3.2 (r1): shared[T,N] workgroup memory (FG.6) lex->parse->HIR->typecheck->codegen->spirv-val->GPU; barrier-visibility correctness oracle reusing the workgroup barrier (scope=Workgroup, semantics=AcquireRelease|WorkgroupMemory=0x108); decidable missing-barrier HARD error + advisory warning for undecidable cases; if/else divergent-barrier HARD error reusing divergent_context_depth; documented cross-vendor (AMD/Intel) race gap (EB.1, untested in current CI); competitive shared-staged+coopmat tiled matmul with a SEPARATE single-index shared-source coopmat emit (Buffer-source byte-identical) and genuinely-parameterizing @strategy tile holes; tiled attention staged (C1 `tiled_attention`, NON-streaming, bit-exact 1e-3 — explicitly NOT FlashAttention-2; the FA2 streaming online softmax = C2 -> M3.2b reserves the flash_attention_v2 name); metadata schema v3 with an allowed-set {1,2,3} version guard + maxComputeSharedMemorySize preflight cached on VulkanContext; AT-1600..AT-1635.)*

*(Changelog: 2026-06-02 — M3.2 (r2): OQ1 missing-barrier analysis made SOUND — the HARD error now fires ONLY for a provable CROSS-SLOT same-block access (index_relation ProvablyDisequal); a provable same-index self read-after-write (`tile[i]=v; ...=tile[i];`) emits NO diagnostic, and an undecidable index relation is demoted to the advisory warning — zero false positives on correct kernels (new AT-1636). OQ2 divergent-barrier check clarified to key on a dedicated conditional_depth counter (if/else bodies only; NOT while/for-range) so uniform while-body barriers are correctly permitted (AT-1635 extended). No other change vs r1.)*

*(Changelog: 2026-06-02 — M3.2 (r3, final design revision): TWO tiny fixes. (1) OQ1 missing-barrier made fully SOUND by tracking the SET of ALL prior same-block write indices to an id since the last barrier (not just the most-recent): the HARD error fires only when the read is ProvablyDisequal to EVERY prior write, NO diagnostic when ProvablyEqual to ANY prior write — so the multi-slot self-write `tile[0]=a; tile[1]=b; let w=tile[0];` is now correctly silent (the r2 residual false-positive), restoring zero false positives (AT-1636 extended); the single-invocation read-of-an-unwritten-slot still hard-errors and is documented acceptable (uninitialized read). (2) The new OQ2 BarrierInDivergentContext hard error inverts the existing test typecheck.rs:4393 (AT-429, `if p { workgroup_barrier(); }`): the assertion flips from errors-empty to expecting the new error, and the test is renamed — the only barrier-in-conditional site in the workspace, mandated so the suite stays green. No other change vs r2.)*

*(Changelog: 2026-06-02 — M3.2 CODER IMPLEMENTATION LANDED: shared[T,N] full pipeline implemented (crates/axc-hir/src/shared.rs, crates/axc-codegen/src/shared.rs) including the SET-based OQ1 missing-barrier analysis, OQ2 conditional_depth divergent-barrier check, AT-429 test inverted, metadata v2→v3 bump with SUPPORTED_SCHEMA_VERSIONS=[1,2,3] allowed-set (CRITICAL-1), all hardcoded schema version assertions updated (CRITICAL-2), CoopMatLoadSource discriminator + emit_coopmat_load_shared_inline/store_shared_inline separate single-index emit functions (CRITICAL-3 infrastructure), maxComputeSharedMemorySize preflight cached on VulkanContext with SharedMemoryExceedsDeviceLimit (CRITICAL-4). Tests AT-1600..AT-1636 passing. Note: AT-1614/AT-1620..AT-1622 (coopmat-from-shared GPU bit-exact) require `#[ignore]`-gated GPU tests on NVIDIA that are not yet fully wired to the HIR coopmat_load dispatch. AT-1630/AT-1631/AT-1632 (tiled attention GPU bit-exact + preflight) require `#[ignore]`-gated GPU tests. AT-1606 barrier-visibility oracle (shared_reduce.axc) compiles + emits correct SPIR-V; race-validation gap documented as EB.1.)*

---

## 3.1.15 M3.3 — OpPhi loop-carried SSA (GPU-proven) + working f32 matmul & attention; full coopmat matmul deferred

M3.2 shipped shared[T,N] (AT-1606 bit-exact on real GPU) but the multi-tile coopmat matmul and tiled attention computed ZEROS. Root cause (coopmat path): emit_for_range emitted the structured loop (OpLoopMerge + CFG) but NEVER an OpPhi for loop-carried SSA values. Coopmat bindings use pure SSA (no OpVariable — opaque tensor types can't live in Function memory), so a coopmat accumulator updated by coopmat_mul_add inside a K-loop re-read its pre-header OpConstantNull every trip; the accumulation was lost.

**PART A — OpPhi loop-carried SSA.** For each coopmat (SSA) binding that is loop-carried — defined before the loop AND reassigned inside the body at this loop level — emit_for_range now emits an OpPhi at the TOP of the loop header block (the first non-label instructions, before the induction load and OpLoopMerge, per SPIR-V 2.4). The phi's operands are (pre_header_value, pre_header_label) and (latch_value, latch_label) — exactly the header's two CFG predecessors. var_ids[binding] is set to the phi id before the body is emitted, so body reads resolve to the phi; the in-loop reassignment produces the latch value; on loop exit var_ids[binding] is left = phi id so a post-loop coopmat_store reads the merged value (correct for zero-trip loops too). SCALARS are UNCHANGED: they keep Function-storage load/store (reduction.axc/AT-1606 prove this works), so detection returns ONLY coopmat bindings and no scalar accumulator gets a phi — minimal and non-breaking. break/continue over a loop-carried coopmat value is conservatively rejected (hard error, never a silent malformed phi). The phi result id is pre-allocated and the phi is constructed once both operands are known via a dr-level insert at the header head (InsertPoint::Begin); the selected block is restored to the merge block afterwards so post-loop emission is correct.

**PART A result — OpPhi GPU-PROVEN.** **AT-1707** dispatches a loop-carried coopmat accumulator (`acc = coopmat_mul_add(A, B, acc)` over K=4 iterations) on the NVIDIA RTX PRO 6000 and asserts the result is **bit-exact = K·(A·B)** — proving the OpPhi loop-carried SSA accumulation is numerically correct on real hardware, not just spirv-val-valid (AT-1701). This resolves the M3.2 blocker. AT-1700 confirms scalar loop-carried bindings still emit ZERO phis (Function-storage path unchanged; reduction.axc/AT-1606 unaffected).

**PART B — multi-tile coopmat matmul (PARTIAL; full numerics deferred).** matmul_shared_coopmat.axc became a real K-loop: `let mut acc = coopmat_zero(); for k_block in range(0,K/tile_k){ stage A,B into shared[f16]; barrier; acc = coopmat_mul_add(coopmat_load(a_tile), coopmat_load(b_tile), acc); barrier; } coopmat_store(acc,C,...)`. The @strategy holes (tile_k, tile_a_size, tile_b_size) genuinely parameterize shared sizes + load strides, and the OpPhi K-accumulation works (AT-1707). **However, the full matmul does NOT yet compute correct numerics for outputs wider than one coopmat tile**: the test fixture is M=16, N=24, K=32, but a single coopmat output tile is 16×16, so N=24 spans more than one output N-tile — the single-output-tile kernel covers only N=0..15. **AT-1620/1622 are therefore compile + spirv-val only (WIP stubs for the bit-exact GPU assertion)**, and the competitive effective-TFLOPS bench was **removed** (a TFLOPS number from a wrong-computing kernel would be misleading). The remaining work — a multi-N-tile output loop (and register/multi-warp blocking for real throughput) — is a kernel-tiling follow-up, NOT a compiler gap. matmul_shared_f32.axc (scalar acc on the working Function-storage path) had a separate dispatch/index-math bug, debugged to **bit-exact on NVIDIA (AT-1621)**.

**PART C — working tiled attention (C1, NOT FA2).** tiled_attention.axc accumulators are scalar (already work); the zeros were dispatch-geometry/index bugs. Fixed to dispatch (seq_len,1,1) workgroups (one per query row); exp uses a Taylor approximation (1+x+x²/2) and the CPU reference uses the IDENTICAL formula for bit-accurate comparison. NON-streaming full softmax (streaming FA2 online-softmax = C2, still M3.2b). **AT-1630: within-1e-3 bit-exact vs the CPU reference on NVIDIA.**

**Honesty.** The OpPhi gap was the coopmat-matmul root cause and is now FIXED + GPU-proven (AT-1707). The f32-matmul and attention zeros were separate scalar-path kernel-logic bugs (not OpPhi), debugged to bit-exact (AT-1621/1630). The FULL competitive coopmat matmul numerics + its TFLOPS are honestly DEFERRED — the kernel needs multi-N-tile output coverage; no wrong-computing test ships as passing and no TFLOPS is reported until the kernel is correct. M3.3 ships: OpPhi loop-carried SSA (GPU-proven) + a working f32 tiled matmul + a working tiled attention.

---

### 3.1.16 M3.3b — multi-tile coopmat matmul (bit-exact full matmul + honest effective-TFLOPS)

M3.3 (§3.1.15) shipped OpPhi loop-carried SSA (GPU-proven, AT-1707) but DEFERRED the full coopmat matmul: matmul_shared_coopmat.axc computed correct numerics only for one 16x16 output tile. The deferral root cause was misdiagnosed as 'needs a multi-N-tile output loop' — the actual cause was (a) the test dispatched a single (1,1,1) workgroup against an N=24 fixture (so only output tile (0,0) ran), and (b) the kernel recovered tile_col = gid(0) directly, which is the GLOBAL invocation id, not the output-tile index. The idiomatic GPU tiling is ONE workgroup == ONE output tile, dispatched as a GRID of workgroups (the grid IS the output-tile loop); no per-kernel output loop is needed.

**The fix (kernel: one line; the rest is dispatch).** With @workgroup(32,1,1) and dispatch (N/16, M/16, 1) workgroups: GlobalInvocationId = workgroup_id*local_size + local_id. AXIOM exposes only gid(axis)=GlobalInvocationId and subgroup_invocation_id()=SubgroupLocalInvocationId (no WorkgroupId/NumWorkgroups builtin), so the output-tile index is recovered by integer division:
- tile_col = gid(0) / 32  (local_size.x=32; all 32 lanes of a workgroup collapse to one tile_col).
- tile_row = gid(1)        (local_size.y=1, so gid(1)==workgroup_id.y; NO division).
- lane = subgroup_invocation_id() == gid(0)%32 (canonical coopmat lane source).
The per-tile body (shared staging of A[tile_row,k_block] + B[k_block,tile_col], barrier, coopmat_load from shared, acc = coopmat_mul_add(a,b,acc) over K/tile_k blocks with the OpPhi accumulator, coopmat_store at c_base=(tile_row*16)*N+(tile_col*16) stride N) is UNCHANGED from M3.3 and per-tile-correct. NO codegen change, NO new builtin/capability — pure kernel-index + dispatch.

**tile_k constraint (M3.3b fix).** tile_k MUST equal the cooperative-matrix K dimension = 16 for `matrix[f16,16,16,*]`. A single `coopmat_mul_add` covers exactly K=16 K-elements. tile_k=32 with one coopmat call per K-block is semantically invalid: the kernel reads K=0..15 of each K-block and produces exactly half the correct output. The @strategy candidate set is therefore `tile_k: ?[16]` (one valid value). The meaningful variation is the **K-block COUNT** (K / tile_k), not tile_k itself. Sub-K-loop support (2 coopmat_mul_add calls per K-block to handle tile_k > 16) is a follow-up (M3.3c+).

**Bit-exact (AT-1620/1622 fixed).** Non-symmetric multiple-of-16 fixture M=32, N=48 (3×2 workgroup grid covers all of C). tile_k=16 FIXED. K is VARIED to exercise the K-block count — the genuinely load-bearing loop variation:
- K=32 (AT-1620): 2 K-blocks. Integer-valued f16 inputs (A∈{1..4}, B∈{1..3}); per-element sum ≤ 384 ≤ 2048 (f16 integer-exact). max_diff == 0.0. Proven on NVIDIA RTX PRO 6000.
- K=48 (AT-1622): 3 K-blocks. Per-element sum ≤ 48×4×3 = 576 ≤ 2048 (f16 integer-exact). max_diff == 0.0. Proves OpPhi accumulation is correct over 3 trip counts.
Typed-skip on Lavapipe (CoopMatUnsupported); matmul_shared_f32.axc remains the Lavapipe-runnable scalar correctness path (AT-1621, unaffected).

**Honest effective-TFLOPS (AT-1710, measured).** resident_matmul_competitive.rs dispatches M=N=K=256 (full 16×16 tile grid) via the resident upload-once/dispatch-N path (N_WARMUP=2 discarded, MIN-of-10, GpuTimestamp). **Measured on NVIDIA RTX PRO 6000 Blackwell: 5.04 TFLOPS = 4.0% of the 125-TFLOPS f32 datasheet estimate.** This is a real bit-exact tensor-core matmul at 5 TFLOPS. Throughput is single-subgroup-per-tile (one 32-lane subgroup per 16×16 output tile) and a small fraction of cuBLAS; it is NOT claimed "competitive" (4% does not warrant it). Register/multi-warp blocking for real throughput is future work. NO ratio is asserted in the test (only tflops>0 && finite). The GpuTimestamp label carries honest provenance; the CpuFenceWall fallback path omits the % and appends the scheduling-inclusive qualifier.

**Out of scope (follow-up M3.3c).** Partial edge tiles (M or N not a multiple of 16) require masked/predicated coopmat load/store; M3.3b restricts to multiples of 16. K not a multiple of tile_k drops the partial final K-block (fixtures use K multiple of tile_k).

---

### 3.1.17 M3.3c — register-blocked coopmat matmul (bit-exact + measured TFLOPS uplift)

M3.3b (§3.1.16) shipped a bit-exact multi-tile coopmat matmul at 5.04 TFLOPS = 4.0% of the 125-TFLOPS f32 datasheet estimate (NVIDIA RTX PRO 6000 Blackwell). The bottleneck is arithmetic intensity: ONE workgroup = ONE 16x16 output tile = ONE subgroup = ONE coopmat accumulator, so the tensor cores idle on shared staging + barriers (issue/bandwidth-bound, not FLOP-bound).

**Register blocking (matmul_rb_coopmat.axc).** Keep ONE workgroup = ONE subgroup (32 lanes), but compute an RB_M x RB_N BLOCK of 16x16 output tiles, holding RB_M*RB_N loop-carried coopmat accumulators in ONE K-loop. The win: each K-block loads RB_M A row-tiles + RB_N B col-tiles from shared ONCE and REUSES them across the RB_M*RB_N MMAs (acc[i][j] += A[i]*B[j]), raising arithmetic intensity ~RB-fold. Dispatch grid shrinks to (N/(RB_N*16), M/(RB_M*16), 1); index recovery is the SAME gid + integer-division idiom as M3.3b (block_col=gid(0)/32, block_row=gid(1), lane=subgroup_invocation_id()) — NO new builtin, NO new capability.

**N loop-carried coopmat accumulators — no codegen change.** detect_loop_carried_coopmat (body.rs) collects the WHOLE SET of carried coopmat bindings and emits N OpPhis in ONE loop header via parallel pre_header_values/latch_values/carried Vecs (reverse-order step-10 insert keeps all N phis first per SPIR-V 2.4). AT-1702 proved the multi-phi insert mechanism (2 phis across 2 nested loops); AT-1733 (new) LOCKS N=4 phis in ONE loop. Register blocking therefore needs NO codegen change. Each accumulator must be coopmat-typed, zero-initialized before the loop, and reassigned UNCONDITIONALLY at top level (AT-1708), with no break/continue (AT-1704) — the unrolled body satisfies all four.

**Hand-unrolled, not a runtime RB loop.** AXIOM has no compile-time unroll over a @strategy hole and no coopmat array type (coopmats are SSA, not addressable), so the shipped example HARD-CODES RB_M=RB_N=2 (4 named accumulators, fully unrolled). The @strategy holes (rb_m, rb_n, a_block_size, b_block_size) genuinely drive the shared-tile sizes + the bench dispatch grid; sweeping RB dims selects among PRE-WRITTEN unrolled variant files. This is the honest expressibility boundary; a strategy-unroll / coopmat-array language feature is future work.

**Bit-exact (AT-1731/1732).** Non-symmetric integer-f16 fixture (A in {1..4}, B in {1..3}, K-sum <= 2048, f16-integer-exact). RB tiling changes ONLY loop structure, not the numeric op sequence: each acc_ij accumulates the SAME ordered coopmat_mul_add over the SAME K-blocks as a single-tile kernel for tile (i,j) -> bit-identical. AT-1731 (2x2, K=32, max_diff==0) + AT-1732 (2x2, K=32 vs K=48, max_diff==0). The single-tile AT-1620/1622 are RETAINED unchanged. Typed-skip on Lavapipe (CoopMatUnsupported); matmul_shared_f32.axc remains the Lavapipe scalar path.

**Measured TFLOPS (AT-1730) — measured at 256³ AND larger sizes.** resident_matmul_rb.rs, same resident upload-once/N_WARMUP=2/MIN-of-10/GpuTimestamp methodology. Reports the BARE effective_tflops + % of the 125-TFLOPS estimate HONESTLY at MULTIPLE sizes; NO ratio asserted (only tflops>0 && finite); 'competitive' label ONLY if pct>=25.0. **MEASURED on NVIDIA RTX PRO 6000 Blackwell (QA run, 2026-06-04):** 256³ = **3.1 TFLOPS (2.5% of datasheet) — REGRESSES vs M3.3b's 5.04 TFLOPS** (under-occupied: 64 WGs < 188 SMs); 512³ = **14.6 TFLOPS (11.7%)** (honest); 768³ = **31.2 TFLOPS (24.96% of the datasheet estimate — a 6.2× improvement over M3.3b's 5.04 TFLOPS)**. Note: 31.2/125 = 24.96% is *just under* the project's 25% "competitive" threshold, so the bench does NOT apply the "competitive" label (it requires pct ≥ 25.0); it is honestly reported as ~25%, NOT claimed competitive. The 256³ regression is reported honestly (not hidden). Register pressure is the cap: 2x2 (4 acc + ~4 transient coopmats) is safe; 4x4 (16 acc) may spill the warp register file (collapsing throughput) — the bench measures and the HONEST best wins. Shared is NOT the cap (2x2 = 2048 B, 4x4 = 4096 B, both << maxComputeSharedMemorySize).

**OCCUPANCY vs ARITHMETIC INTENSITY TRADEOFF (non-blocking, must be stated honestly).** The RB 2×2 dispatch grid at M=N=256 is (N/32, M/32, 1) = (8, 8, 1) = **64 workgroups**, compared to M3.3b's (N/16, M/16, 1) = (16, 16, 1) = **256 workgroups**. The RTX PRO 6000 Blackwell has ~188 SMs; 64 single-subgroup workgroups is FEWER than the SM count and may cause under-occupancy — the RB kernel could be SLOWER than M3.3b's baseline at 256³, despite higher arithmetic intensity per workgroup. The bench therefore measures at **both 256³ and larger sizes (512³, 768³)**: at larger sizes the grid is (16, 16, 1) = 256 workgroups (512³) or (24, 24, 1) = 576 workgroups (768³), restoring occupancy while preserving the register-blocking arithmetic-intensity gain. If 256³ regresses vs 5.04 TFLOPS but 512³ or 768³ improves, **both are reported honestly** — the smaller grid at 256³ understates RB's win for larger matmuls. The acceptance gate (AT-1730) is honest improvement at ANY measured size, not a specific size.

**Deferred (follow-up M3.3d / M3.4).** (1) Multi-subgroup blocking (multiple warps per workgroup sharing the staged A/B) needs a LocalInvocationId / SubgroupId-within-workgroup builtin — AXIOM exposes only GlobalInvocationId + SubgroupLocalInvocationId + SubgroupSize (grep-confirmed). Adding local_invocation_id() (SPIR-V BuiltIn LocalInvocationId, lowering identical to the existing GlobalInvocationId Input var) is a bounded but separate language addition, deferred to keep M3.3c low-risk. (2) Double-buffered shared staging (software pipelining the next K-block) — higher complexity, dominant win is register blocking, deferred. (3) Partial RB-blocks / edge tiles (M,N not multiples of RB*16) — masked/predicated coopmat load/store, carried from M3.3b.

---

## 3.1.18 M3.3d — local_invocation_id() builtin + multi-subgroup (multi-warp) coopmat matmul

M3.3c (§3.1.17) reached 31.2 TFLOPS = 24.96% of the 125-TFLOPS datasheet estimate at 768 via 2x2 register blocking, SINGLE-subgroup-per-workgroup. The next gain is MULTI-subgroup blocking: a workgroup with N_SG subgroups, each computing distinct output tiles while SHARING the staged A/B in workgroup memory -> higher occupancy AND amortized staging. This needs a builtin AXIOM lacked: local_invocation_id().

**PART A — local_invocation_id(axis: u32) -> u32.** A new HIR builtin lowering to SPIR-V BuiltIn LocalInvocationId (a vec3 u32 Input OpVariable + OpLoad + OpCompositeExtract by literal axis) — IDENTICAL lowering shape to gid() (GlobalInvocationId). LocalInvocationId is a CORE Shader builtin: NO new SPIR-V capability, NO extension (anti-pattern #7: no silent capability). Threaded through lexer (plain Ident, like gid), parser (Expr::Call, no grammar change), HIR (HirExprKind::LocalInvocationIdBuiltin{axis}, typecheck: arity 1, axis literal 0..=2, result u32), codegen (emit_local_invocation_id_variable in buffers.rs + emit_local_invocation_id_component in body.rs + emit.rs wiring; emitted ONLY when used). AT-1740 (lex/parse/HIR/typecheck), AT-1741 (codegen + spirv-val + no-new-capability), AT-1742 (real-GPU bit-exact: out[g*64+l]==l).

**PART B — matmul_msg_coopmat.axc (multi-subgroup, SHIPPED N_SG=2).** @workgroup(64,1,1) = 2 subgroups of 32 lanes. subgroup_id_in_wg = local_invocation_id(0u32) / subgroup_size(). HARD PRECONDITION: the kernel REQUIRES subgroup_size==32 (one workgroup = exactly 2 32-lane subgroups). Its output coverage is hard-wired to 2 subgroups by the constants 64 (b_tile width / grid divisor N/64) and sg_id*32 (B/C offset), so it MISCOMPUTES on other wave widths and is NOT a graceful degrade: wave64 (sg_size=64) -> sg_id=0 for all threads -> only ONE subgroup runs -> half of C (cols +32..+64 per WG) NEVER written; SIMD16 (sg_size=16) -> sg_id in {0,1,2,3} -> sg_id*32 OOB of the 64-wide b_tile + cross-workgroup C overwrite. Therefore ALL its GPU tests/bench TYPED-SKIP when subgroup_size()!=32 (mirror dispatch_matmul_tile.rs:231 AT-1510); on NVIDIA RTX PRO 6000 (wave32) they run. Both subgroups SHARE the same staged 32-row A block (max A-reuse); they STACK ALONG N, each computing a 2x2 RB block (32x32 C sub-tile) on its own B columns (subgroup s reads shared b_tile cols [s*32..+32); writes C cols [block_col*64 + s*32 .. +32)). All 64 threads cooperatively stage a_tile[512]/b_tile[1024] (3072 B shared; a/b_block_size % wg_threads == 0 so every element is covered exactly once); ONE workgroup_barrier. **Cross-subgroup bit-exactness is GUARANTEED BY CONSTRUCTION** by the emitted workgroup-scope OpControlBarrier (execution scope Workgroup=2, memory semantics 0x108 AcquireRelease|WorkgroupMemory; emit_workgroup_barrier subgroup.rs:285-301, unit-tested subgroup.rs:476) — there is NO subgroup-scope barrier path to weaken it, so this does not rely on a test passing via NVIDIA lockstep luck. Each subgroup then independently runs its coopmat block (coopmat scope = Subgroup -> per-32-lane-subgroup execution). 4 loop-carried coopmat accumulators per subgroup via the M3.3c N-phi path (AT-1733) — NO codegen change for the matmul. Grid = (N/64, M/32, 1). @strategy holes wg_threads/n_sg/rb_m/rb_n/tile_k/a_block_size/b_block_size genuinely drive shared sizes + bench grid (hand-unrolled per OQ-1).

**Bit-exact.** AT-1743 (M=32,N=64,K=32, grid (1,1,1) = 1 WG = 2 subgroups, 2 K-blocks, max_diff==0) + AT-1744 (M=64,N=128,K=48, grid (2,2,1), 3 K-blocks => 2 WAR barriers, max_diff==0) on the non-symmetric integer-f16 fixture (catches a swapped sg_id offset on the 64-wide stride-64 b_tile). The workgroup-scope barrier guarantees cross-subgroup correctness by construction; AT-1743/1744 are the end-to-end confirmation and BOTH carry the subgroup_size()==32 typed-skip guard. Single-subgroup AT-1731/1732 RETAINED. Typed-skip on Lavapipe (CoopMatUnsupported) AND on subgroup_size()!=32.

**Measured TFLOPS (AT-1750) — HONEST, no asserted ratio.** resident_matmul_msg.rs (M3.3c methodology) at 256/512/768 (+1024 if <50GB). Reports BARE tflops + pct of 125-TFLOPS; competitive label only if pct>=25.0. Single-subgroup RB bench RETAINED for A/B. **MEASURED on NVIDIA RTX PRO 6000 Blackwell (QA run, 2026-06-05):** 256³ = **2.5 TFLOPS (2.0%)** (under-occupied: 32 WGs); 512³ = not the peak; 768³ = **24.0 TFLOPS (19.2% of the 125-TFLOPS datasheet estimate)**. **REGRESSION vs M3.3c single-subgroup RB** (31.2 TFLOPS = 24.96% at 768³): the halved workgroup count at 768³ (288 MSG workgroups vs 576 single-subgroup RB workgroups) + cross-subgroup barrier overhead outweigh the staging amortization. 19.2% < 25.0% threshold — the bench does NOT apply the "competitive" label; multi-subgroup is NOT faster than single-subgroup RB at any measured size. **The local_invocation_id() builtin (PART A) is the durable deliverable; the multi-subgroup kernel is an honest negative performance result.** Single-subgroup RB (M3.3c) remains the best matmul.

**Deferred to M3.4+.** Double-buffered shared staging (software pipelining); partial/edge tiles (M,N not multiples of 64/32 — masked coopmat); N_SG=4 / true strategy-unroll over N_SG and RB (needs coopmat-array / strategy-unroll language feature, OQ-1).

---

### 3.1.19 M3.4 — llama.cpp Vulkan Q4_K_M A/B (the pre-registered kill-criterion run)

DESIGN.md §5 kill criterion: *"cannot match llama.cpp Vulkan Q4_K_M within 15% on any vendor."* M3.4 runs the head-to-head on NVIDIA RTX PRO 6000 Blackwell, same `nvidia_icd.json`, fence-synchronized, kernel-only-vs-kernel-only.

**llama.cpp side.** ggml-org/llama.cpp pinned at the LITERAL 40-char commit SHA `6b80c74f285390368b3c99c5e750f19e9b096e98` (resolved from release tag `b9542`; both recorded in `ab_results.json` — a tag is re-pointable, only the SHA is reproducible), built `-DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON`, target `test-backend-ops`. The comparable op is `test-backend-ops perf -o MUL_MAT` filtered to `type_a=q4_K`, the GEMV (n==1) case — per-op throughput, EXCLUDING host transfer / model load. Chosen over `llama-bench` because test-backend-ops measures a SINGLE Q4_K matmul op resident on the Vulkan backend (closest to AXIOM's kernel-only span), whereas llama-bench reports whole-model prompt-processing tok/s (folds in attention/norms/sampling — modeling error). The selected line is recorded verbatim in `.pipeline/benchmarks/m34/ab_results.json`:

```
MUL_MAT(type_a=q4_K,type_b=f32,m=4096,n=1,k=14336,...): 63048 runs - 15.89 us/run - 117.44 MFLOP/run - 7.39 TFLOPS
```

A Q4_K MUL_MAT n==1 (GEMV) case DOES exist at this SHA (generated by the `for (int bs : {1,2,3,4,5,8,512})` × `all_types` loop in `make_test_cases_perf`), so the kill-criterion is computed from a true GEMV — never a GEMM substitution (CRITICAL-2 satisfied; status is FAIL, not INCOMPLETE).

**Measurement-boundary comparability (the central fairness control).** test-backend-ops `perf` was READ at the pinned SHA and recorded verbatim in `ab_results.json` (`llamacpp_timer_boundary`). It brackets `ggml_backend_graph_compute` with **CPU wall-clock** (`ggml_time_us()`) over a graph that **duplicates the op `n_runs` times** (FLOP-targeted ~100 GFLOP), loops until `total_time_us ≥ 1 s`, and reports `avg_time_us = total_time_us / total_runs` — i.e. **sustained, batched-amortized MEAN, one warmup discarded, no overhead subtracted**. To be comparable, the AXIOM side reports its on-device GpuTimestamp MIN/MEAN/MEDIAN AND a **sustained CPU-wall** number (200 back-to-back dispatches / 200). The headline ratio uses the matched (sustained CPU-wall) boundary; the GpuTimestamp framing is reported alongside for disclosure. FLOP convention is asserted identical (2·M·N·K matmul MACs only, dequant excluded both sides) and verified by recomputing GFLOPS from llama.cpp's reported µs + m/n/k (recomputed 7.39 vs reported 7.39 TFLOPS, 0.01% disagreement — PASS).

**INCOMPLETE is a first-class outcome.** If no n==1 Q4_K case existed, or the FLOP-consistency check failed (>2%), or the device strings/ICDs were not byte-identical, the run would be recorded as `kill_criterion_status=INCOMPLETE` — NEVER a GEMM-substituted or boundary-mismatched FAIL ratio. None of those fired here.

**AXIOM side.** `examples/q4km_dequant_matvec.axc` (the M2.6 single-row dequant+matvec, FROZEN — no codegen change), measured via the resident path (upload-once / N_WARMUP=2 / MIN-of-10 / GpuTimestamp), REAL 144-byte Q4_K_M weights, bit-exact-pre-flight vs the ggml CPU reference (AT-1761: rel_err ≈ 1.2e-7). K = 56·256 = 14336, matched to the llama.cpp case.

**The fairness contract.** Kernel-only-vs-kernel-only (both exclude host transfer), identical K=14336 contraction. Both kernels produce CORRECT Q4_K_M output (AXIOM bit-exact vs ggml; llama.cpp IS ggml) — the A/B is purely PERF. The apples-to-oranges CAVEAT, stated plainly: AXIOM's `q4km_dequant_matvec` produces ONE output row per dispatch using ONE workgroup (the `if i>=1 return` guard), so it occupies ~1 of ~188 SMs and does only 14336 MACs per 316 µs dispatch (dispatch-latency-dominated); llama.cpp's MUL_MAT computes **m=4096 output rows** over the same K, tiled across all SMs with vendor-tuned shared-memory staging + dequant fusion. Because the two compute different amounts of work per call, the FAIR metric is **work-normalized throughput (TFLOPS)**, not raw µs.

**MEASURED RESULT (same machine, same ICD, NVIDIA RTX PRO 6000 Blackwell, 2026-06-06):**

| metric | AXIOM (M2.6 single-row matvec) | llama.cpp (Q4_K MUL_MAT n=1) |
|---|---|---|
| output rows | 1 | 4096 |
| µs (GpuTimestamp MIN) | 315.5 | (CPU-wall) |
| µs/op (sustained CPU-wall) | 338.7 | 15.89 |
| TFLOPS (GpuTimestamp MIN) | 0.000091 | — |
| TFLOPS (sustained CPU-wall) | 0.000085 | 7.39 |

**Headline ratio (AXIOM/llama, work-normalized TFLOPS, matched sustained boundary): ≈ 1.0e-5 → llama.cpp is ≈ 87,000× higher throughput.** **Kill-criterion verdict (DESIGN §5): within 15% on NVIDIA? — NO (FAIL).** This is the HONEST documented baseline that quantifies the gap. It does NOT fire the project kill-criterion: DESIGN §5 reads "within 15% on ANY vendor", so a single-vendor (NVIDIA) FAIL with the current single-row kernel is a data point, not the firing of the criterion. The AMD APU / Intel Arc halves (where AXIOM's portability thesis is strongest) are deferred-not-dropped, pending cross-vendor hardware (EB.1). No prose here should be read as "kill criterion FAILED" without this "NVIDIA only; criterion is any-vendor; AMD/Intel pending" qualifier.

The ~87,000× gap is brutal but expected and decomposes into two stacked deficits: (1) AXIOM computes **1/4096th** of the output rows per dispatch, and (2) that single dispatch is **dispatch-latency-dominated** (316 µs for 14336 MACs ≈ 0.09 GFLOPS) because one workgroup uses ~1 of ~188 SMs. The gap-closing path: fuse the Q4_K_M dequant front-end onto the M3.3c register-blocked coopmat matmul (dequant → shared f16 tile → coopmat mul_add; the plain-f16 version reached 31.2 TFLOPS = 24.96% of datasheet) — a follow-up milestone. M3.4 establishes the baseline that motivates that fusion.

**Reproducibility.** `scripts/m34_llamacpp_ab.sh` rebuilds llama.cpp from the pinned SHA into the gitignored `vendor/llama.cpp`, runs both sides under `VK_DRIVER_FILES=nvidia_icd.json`, and emits the A/B table + `ab_results.json` (device, ICD, both tag+SHA, dims, AXIOM min/mean/median/sustained, llama.cpp number, ratio, verdict, timer-boundary excerpt). llama.cpp is NEVER committed. NOTE: on a box where `libvulkan-dev` does not ship the SPIRV-Headers CMake config, the script points cmake at a locally-installed copy (`vendor/spirv-headers/install`) — ggml-vulkan's CMakeLists does `find_package(SPIRV-Headers)` but does not link the target, so its include dir is injected via `CMAKE_CXX_FLAGS`.

---

## 3.1.20 M3.5 — Q4_K_M dequant fused into the register-blocked coopmat matmul (closing the M3.4 gap, SAME-SHAPE)

M3.4 (§3.1.19) measured the FROZEN M2.6 single-row Q4_K_M matvec at ~0.000085 TFLOPS = ~87,000x slower than llama.cpp Vulkan Q4_K_M — but that was an HONEST same-shape GEMV-vs-GEMV loss (both sides n==1, 7.39 TFLOPS on the llama side). M3.3c (§3.1.17) reached 31.2 TFLOPS (24.96% of the 125-TFLOPS datasheet estimate) with a GENERIC f16 register-blocked coopmat matmul. M3.5 FUSES them: a Q4_K_M matmul that dequantizes a tile of weights into a shared[f16] tile then runs the M3.3c RB coopmat K-loop.

**The f32->f16 builtin (the single language gap).** The dequant produces f32 (f16_bits_to_f32 widens scales/mins; all dequant arithmetic is f32 — matching ggml's f32 dequant). The coopmat tiles are shared[f16]. Writing f32 into shared[f16] is a HARD ERROR (SharedWriteTypeMismatch; no coercion, anti-pattern #1) and AXIOM had no f32->f16. M3.5 adds f32_to_f16(x: f32) -> f16 as a 5th Q4_0Builtin variant lowering to a single OpFConvert f32->f16 (IEEE RNE, the SPIR-V default; no FPRoundingMode decoration). The builtin needs ONLY Float16 (already declared). The fused kernel's readonly_buffer[u8] weight buffer additionally pulls in Int8/StorageBuffer8BitAccess/SPV_KHR_8bit_storage — these are the FROZEN M2.6 Q4_K_M dequant caps, NOT new; so AT-1773's no-new-capability baseline is the UNION of the M3.3c coopmat caps and the M2.6 Q4_K_M dequant caps, and f32_to_f16 adds nothing beyond that union.

**The fused kernel (q4km_matmul_rb_coopmat.axc).** @workgroup(32,1,1), grid (N/32, M/32, 1), one workgroup = a 2x2 block of 16x16 output tiles. Per K-block: cooperatively dequantize the 32x16 A-block from Q4_K_M (the byte-identical M2.6 inline get_scale_min_k4 + nibble path, q4km_dequant_matmul.axc:41-104), convert `let w: f16 = f32_to_f16(...); a_tile[ei]=w;` (each a_tile element written exactly once, AT-1620-class coverage asserted); stage the f16 x B-tile (no conversion); barrier; M3.3c 4-accumulator A/B-reuse coopmat mul_adds (matrix[f16,16,16,accumulator], N-phi loop-carried); WAR barrier; after K, coopmat_store.

**Numerics + oracle (f16-accumulator-MATCHED).** NOT bit-exact vs the f32 path. The GPU coopmat accumulator is f16 (matmul_rb_coopmat.axc:122-125); the CPU oracle MATCHES that — dequant in f32 exactly as ggml, round weights+x f32->f16->f32 (RNE), then accumulate in f16 (round the running accumulator to f16 after each depth-16 tile add). Validated within the FROZEN @equiv_fp_tol(1e-3) (AT-1520/AT-1521 value, NOT loosened). The within-tol gate is asserted at the MEASURED provably-1e-3 K (AT-1770 K=256, AT-1771 K=512); larger-K max-rel-diff is reported SEPARATELY and the measured max-rel-diff is reported at EVERY size. The f16-accumulator within-tol result is NVIDIA-coopmat-accumulator-specific.

**Measured TFLOPS (AT-1772) — HONEST, no asserted ratio.** resident_q4km_matmul_rb.rs (M3.3c methodology) at 256/512/768(/1024) + the pinned A/B shape. Reports bare tflops + % of 125-TFLOPS + measured max-rel-diff; 'competitive' label only if pct>=25.0. Expected BELOW M3.3c's 31.2 TFLOPS (Q4_K_M dequant is ALU-heavy). [Coder fills the measured table.]

**The M3.4 A/B re-run (AT-1774) — SAME-SHAPE, the fair fight.** scripts/m34_llamacpp_ab.sh --fused: same machine/ICD/SHA (b9542/6b80c74), device-match byte-identical, FLOP-consistency (2*m*n*k, dequant excluded both sides), kernel-only. The HEADLINE / kill-criterion-of-record is SAME-SHAPE: AXIOM fused GEMM at (m=4096,n=512,k=14336) vs llama.cpp Q4_K MUL_MAT at the IDENTICAL shape = 101.00 TFLOPS. Honest expected result: AXIOM ~10-25 TFLOPS vs llama 101 → AXIOM ~4-10x BEHIND same-shape — a MASSIVE improvement from M3.4's ~87,000x but STILL behind, reported as that (NOT a win). The n==1 GEMV (llama 7.39 TFLOPS) appears ONLY as labeled cross-shape context — the SAME llama kernel runs 13.7x faster at n=512 than n=1, so headlining AXIOM-GEMM vs llama-n1-GEMV would flatter AXIOM ~13x (the inverse of M3.4's honest GEMV-vs-GEMV loss, and exactly the intent-laundered claim the thesis forbids). Kill-criterion verdict (DESIGN §5, within 15%): does NOT fire (AXIOM behind same-shape); reported honestly. Recorded in ab_results_fused.json. HONESTY: if dequant dominates and AXIOM is further behind 101, that is the reported result.

*(Changelog: M3.5 — f32_to_f16 builtin (5th Q4_0Builtin variant, OpFConvert f32->f16 RNE, Float16-only, no capability beyond the M3.3c+M2.6 union) + the fused Q4_K_M register-blocked coopmat matmul (dequant->shared[f16]->M3.3c RB coopmat); within the FROZEN 1e-3 vs an f16-ACCUMULATOR-matched ggml Q4_K_M reference (AT-1770/1771, measured max-rel-diff reported, gate-K capped at the provably-1e-3 K); honest multi-size TFLOPS + max-rel-diff (AT-1772); CI no-new-capability-beyond-union anchor (AT-1773); the M3.4 A/B RE-RUN with the fused kernel SAME-SHAPE (AT-1774, ab_results_fused.json) — AXIOM ~4-10x behind llama's same-shape 101 TFLOPS, down from M3.4's ~87,000x cross-shape; f32_to_f16 layer + codegen unit coverage (AT-1775/1776). FROZEN M2.6 matvec + M3.3c RB matmul + M3.4 matvec A/B + 1e-3 tolerance unchanged. AT-1770..AT-1776.)*

---

### 3.1 Types

```
Primitives:     i8 i16 i32 i64 u8 u16 u32 u64 f16 bf16 f32 f64 bool
Vectors:        vec2..vec4 (f32), dvec2..dvec4 (f64), ivec2..ivec4 (i32), uvec2..uvec4 (u32)
Buffers:        buffer[T]              // SSBO on Vulkan, cl_mem on OpenCL
                readonly_buffer[T]     // readonly decoration
                writeonly_buffer[T]    // writeonly decoration
Images:         image2d[T] image3d[T]  // opaque image handles
Shared:         shared[T, N]           // workgroup-local memory (SLM)
Matrices:       matrix[T, M, N]        // cooperative_matrix
Subgroup:       subgroup[T]            // subgroup-uniform values
```

### 3.2 GPU-specific annotations (the core value add)

| Annotation | Meaning | Lowering |
|---|---|---|
| `@kernel` | Entry point (compute shader) | `OpEntryPoint GLCompute` + `OpExecutionMode` |
| `@workgroup(X, Y, Z)` | Workgroup dimensions | `LocalSize` execution mode |
| `@subgroup_uniform` | Value invariant across subgroup | `SubgroupUniform` decoration |
| `@uniform_branch` | Control flow uniform across subgroup | Guides structurizer |
| `@divergence_free` | No divergence in this region | Verified + unlocks shuffles |
| `@coalesced(stride=1)` | Adjacent threads access adjacent elements | Verified by access-pattern analysis |
| `@shared_tile(M, N, pad=K)` | SLM tile with bank-conflict padding | Emits `shared[T, M*(N+K)]` |
| `@no_bank_conflict` | Asserted; verified by swizzle/stride analysis | Compile-time check |
| `@cooperative_matrix(M, N, K, A_type, B_type, C_type)` | Tensor-core-style tile op | `OpCooperativeMatrixLoadKHR` / `MulAddKHR` |
| `@occupancy(min=0.5)` | Minimum target occupancy | Register-pressure estimate; rejects if violated |
| `@max_registers(N)` | Register budget | Informs codegen + spill heuristics |
| `@async_copy` | Overlap memory + compute | `cp.async` / equivalent where available |
| `@reduction(op)` | Parallel reduction over subgroup/workgroup | `SubgroupReduce` / tree reduction |
| `@barrier(scope)` | Explicit barrier with scope | `OpControlBarrier` |
| `@equiv_fp_tol(eps)` | FP tolerance for rewrite verification | Correctness oracle |
| `@strategy { tile_m: ?, tile_n: ?, stages: ? }` | LLM autotune hole | Resolved by `axc optimize` |
| `@strict` | Require `@intent` + `@complexity` + pre/post on every kernel | Compile error if missing |
| `@precondition(expr)` / `@postcondition(expr)` | Runtime-checkable contract | Debug-mode assertion |
| `@target(vendor, caps)` | Target capability set | Guards codegen paths |

### 3.3 SPIR-V extensions in scope

**Portable baseline (must work on NVIDIA/AMD/Intel):**
- `SPV_KHR_shader_subgroup*` (ballot, vote, shuffle, reduce)
- `SPV_KHR_cooperative_matrix` (tensor-core portable access, promoted in Vulkan 1.3.300)
- `SPV_KHR_float_controls`
- `SPV_KHR_16bit_storage` / `SPV_KHR_8bit_storage`
- `SPV_KHR_integer_dot_product`
- `SPV_EXT_shader_atomic_float_add`

**Vendor fast paths (path-split, not required):**
- `SPV_NV_cooperative_matrix2` (workgroup-scope matrices, per-element ops) -- NVIDIA only
- `SPV_INTEL_*` -- Intel DPC++ compute

---

## 4. Phased plan

### Phase M0 -- Bootstrap (week 0-2)

- Cargo workspace with 7 empty crates
- Lexer port from AXIOM with GPU keyword additions
- `axc lex` CLI emitting tokens
- CI on GitHub Actions (test + build)
- `.pipeline/` copied and adapted (this document produced)
- First kernel compiles (empty kernel with `@workgroup(64,1,1)`) → SPIR-V → spirv-val clean

**Exit gate M0:** `axc compile empty_kernel.axc -o empty.spv && spirv-val empty.spv` succeeds on Linux CI.

**M0 addendum:** Correctness for M0 is syntactic (spirv-val only). GPU dispatch + Lavapipe fallback + equivalence checks all begin at M1 per exit gate. Anti-pattern #9 (no feature without a GPU test) is formally relaxed for M0; re-armed from M1 onward.

### Phase M1 -- Minimum viable SPIR-V backend (month 1-3)

- Parser + HIR + SPIR-V codegen path for: scalar ops, buffers, workgroup dispatch, barriers, subgroup ops
- `axc run` dispatcher on Vulkan via `ash`
- Validator integration (`spirv-val`, `spirv-opt`)
- Saxpy + vector add + parallel reduction benchmarks running on NVIDIA + AMD + Intel + Lavapipe

**Exit gate M1:** 3-vendor execution of 5 elementwise kernels, correctness verified vs CPU reference.

### Phase M2 -- Cooperative matrix + llama.cpp Q4_K_M matmul (month 3-9)

- `@cooperative_matrix` lowering to `SPV_KHR_cooperative_matrix`
- `@strategy` hole infrastructure + LLM autotuner
- MCP server port from AXIOM
- llama.cpp Q4_K_M dequant + matmul reference kernel
- Bench harness producing `tok/s` comparison vs llama.cpp current Vulkan baseline

**Exit gate M2:** On RTX 4090, within 5% of hand-tuned llama.cpp Vulkan Q4_K_M. On AMD APU or Intel Arc, beat current Vulkan by ≥25%. One annotated source, three targets.

### Phase M3 -- FlashAttention + KernelBench-Vulkan submission (month 9-15)

- Attention kernel (FA2 shape, then FA3 where extensions allow)
- First public SPIR-V KernelBench submission
- `@equiv_fp_tol` correctness oracle (Sakana-proof verification)

**Exit gate M3:** ≥80% of cuBLAS+FA3 on H100 via Vulkan compute; ≥90% of rocBLAS on MI300X. Public KernelBench leaderboard entry.

### Phase M4 -- PyTorch custom-op frontend + adoption (month 15-24)

- `torch.compile` backend or `torch.library` custom-op integration
- Upstream PR to llama.cpp (or candle/MLX) replacing at least one hot kernel with AXIOM-Compute output

**Exit gate M4:** External adoption by ≥1 production inference framework.

---

## 5. Kill criteria (pre-registered)

Stop the project if, at the stated gate:

- **M1 slip (>6 mo):** cannot produce a correctness-verified SPIR-V kernel on 3 vendors. Means the validator/lowering path is structurally broken.
- **M2 slip (>12 mo):** cannot match llama.cpp Vulkan Q4_K_M within 15% on *any* vendor. Means annotation-to-SPIR-V lowering adds no value above handwritten GLSL.
- **M3 slip (>18 mo):** FA variant cannot clear 50% of cuBLAS+FA3 on any GPU. Means LLMs + annotations cannot close the handwritten-vendor-kernel gap -- project thesis is refuted.
- **Ecosystem preempt:** if Slang grows an equivalent annotation sublanguage, or Triton-Vulkan ships and hits M2 targets first, pivot to contribute there instead of competing.

---

## 6. Development workflow

7-agent pipeline inherited from AXIOM (see `.pipeline/PIPELINE.md`):

1. **Architect** (Opus) -- designs spec + plan
2. **Optimistic Design Reviewer** (Sonnet) -- validates feasibility + completeness
3. **Pessimistic Design Reviewer** (Opus) -- adversarial review
4. **Coder** (Sonnet) -- implements agreed plan
5. **QA** (Sonnet) -- verifies test conformance, not-trust-always-verify
6. **Optimistic Code Reviewer** (Sonnet) -- spec compliance + quality
7. **Pessimistic Code Reviewer** (Opus) -- UB, races, correctness

**Verdict enforcement:** every reviewer agent emits `{"verdict": "APPROVE|REJECT|NEEDS_REVISION"}` as the FIRST key in a JSON object. Supervisor parses this and gates the next pipeline phase. Memory-tracked rule: agents lie about their own work, so independent adversarial verification is mandatory.

**Memory limits:** test executables must not exceed 50 GB (from prior incident with a 336 GB crash).

---

## 7. Open questions (M0 -> architect resolves)

1. OpenCL SPIR-V flavor vs Vulkan SPIR-V flavor -- emit both, or OpenCL first?
2. Direct `rspirv` emission vs going through LLVM SPIR-V backend -- which for M1?
3. Naming: `.axc` for source files (for AXIOM-Compute) or `.axm` reused? Current doc assumes `.axc`.
4. Runtime dispatch: start with Kompute (Vulkan-only) or `ash` (more control)? Current doc assumes `ash`.
5. How much should we inherit from `axiom` as a git subtree vs re-implement? Prefer crate-level reuse where APIs are stable.
6. Where is the line between `axc-optimize` (autotuner) and `axc-driver` (MCP server)? Keep them separate, match AXIOM layout.

---

### Integer division undefined behavior

SPIR-V `OpSDiv` and `OpSRem` with `INT_MIN / -1` (e.g., `i32::MIN / -1`) are UNDEFINED
BEHAVIOR per SPIR-V unified spec §3.32.14. AXIOM-Compute does NOT emit runtime checks
for this case. Programs that rely on well-defined behavior for this specific input
must guard it manually at the source level, e.g.:

    let x: i32 = if a == -2147483648 && b == -1 { -2147483648 } else { a / b };

The same UB applies to unsigned integer division by zero and signed remainder by zero:
both are undefined. Compile-time constant-folded cases (both operands literals that
trigger UB) may be rejected at HIR typecheck in a future milestone.

---

## Revision log

- **2026-04-18:** Initial draft (v0.1), pre-architect review. To be revised through dual design review.
- **2026-04-18:** M1.1 revision — added §3 integer division UB note (CRITICAL-2 fix from pessimistic review).
- **2026-04-18:** M1.2 revision — added §3.1 M1.2 parameter binding model (buffer types, scalar params, gid builtin), saxpy binding assignment walkthrough, and interface-list SPIR-V 1.3 rule.
- **2026-04-18:** M1.3 revision — added §3.1.4 Control flow (M1.3): OpLoopMerge, continue_target, structured CFG for if/for/while/break/continue.
- **2026-04-18:** M1.5 revision — added §3.1.6 Runtime dispatch (M1.5): VulkanContext lifecycle + Drop ordering, DispatchRequest API + ownership model, metadata sidecar schema v1, host-visible memory simplification + M2 staging-buffer plan, fence timeout default, push-constant byte-assembly discipline, workgroup-count device-limit pre-validation, Vulkan 1.1 subgroup BASIC+VOTE guaranteed / ARITHMETIC+BALLOT+SHUFFLE+CLUSTERED+QUAD device-optional note.
- **2026-04-18:** M2.2 revision — added §3.1.7 Benchmark harness: Criterion bench groups (compile_pipeline, cpu_reference, dispatch_gpu), regression gate (11-sample median, 15% threshold), baselines.json schema v1, BENCHMARKS.md forward reference.
- **2026-04-18:** M2.5 revision — added §3.1.8 Q4_0 dequantization builtins: Q4_0 block layout (18 bytes/block, 32 f32 elements), four new builtins (ptr_read_u8_zext, ptr_read_u16_zext, f16_bits_to_f32, f32_from_u32), capability side-effects (Int8=39, Int16=22, Float16=9, StorageBuffer8BitAccess=4448), integration tests AT-901..AT-918, dispatch_gpu_q4_0 bench group (n_blocks=128 and 1024).
- **2026-04-18:** M2.4 revision — added §3.1.10 MCP server: JSON-RPC 2.0 stdio bridge exposing 6 tools (initialize, load_source, enumerate_variants, compile_variant, bench_variant, grid_search, optimization_history); NDJSON framing; 8 MiB inbound cap; RFC 4648 §4 STANDARD base64; RFC 3339 UTC millisecond timestamps; POSIX flock(LOCK_EX) history append; lazy Vulkan init (OnceVulkan); tri-state CorrectnessStatus; seeded deterministic inputs; AXC_MCP_HISTORY_DIR env override; 10 error codes (-32700 through -32006); acceptance tests AT-1101 through AT-1132.
- **2026-04-18:** M2.6 revision — added §3.1.11 Q4_K_M superblock layout and dequant kernel: 144-byte block format (2-byte f16 d + 2-byte f16 dmin + 12-byte packed scales + 128-byte packed weights), bit-spread unpacking via inlined get_scale_min_k4 with canonical q[j] (NOT q[j-4]) m-high idiom, four-chunk two-sub-block-per-chunk iteration, dequant formula y=d*sc*nibble-dmin*m (NO -8 offset unlike Q4_0), @equiv_fp_tol(1e-3) tolerance, same M2.5 capability set (zero additions), integration tests AT-1301..AT-1331, bench group dispatch_gpu_q4km (dispatch_q4km_128 + dispatch_q4km_512).
- **2026-06-01:** M3.1 revision — added §3.1.13 Cooperative-matrix dispatch + multi-row matmul + GPU-resident benchmark: coopmat shape metadata plumbing (HIR CoopMatShape, KernelMetadata schema v1→v2 back-compat), device-feature enablement pass (capability→feature table, CRITICAL-2/-3), preflight + subgroupSize==32 guard + typed graceful skips, multi-row Q4_K_M matmul (bit-exact 256×256), staged pre-dequant coopmat bridge, GPU-resident benchmark methodology (upload-once/dispatch-N/kernel-only timing), Lever A readback skip. See §3.1.13 for full details.

---

## §3.1.13 M3.1 — Cooperative-matrix dispatch + multi-row matmul + GPU-resident benchmark

M3.1 makes the M2.1 cooperative-matrix codegen EXECUTE on real tensor cores for the first time, adds a multi-row Q4_K_M matmul, and builds the GPU-resident benchmark methodology. The emitted coopmat SPIR-V is UNCHANGED from M2.1; M3.1 is a RUNTIME + metadata + kernel + benchmark milestone. The only codegen touch is a mechanical internal-cache-key extension (CoopMatKey gains K + result_type) that leaves the emitted module byte-identical.

### Coopmat shape metadata (HIR → sidecar → runtime)

Previously the runtime had no way to know a kernel used cooperative matrices or what shape it required. M3.1 plumbs the shape end-to-end. The HIR derives a kernel-level `CoopMatShape{m,n,k,a/b/c/result element types, scope}` from the body's `matrix[T,M,N,use]` types of the coopmat ops (reusing M2.1 typecheck guarantees, so K = a.n = b.m). The metadata sidecar gains a `coopmat: Option<CoopMatShapeMeta>` field; `CURRENT_SCHEMA_VERSION` is bumped 1→2 and `load_kernel_metadata` accepts both versions (v1 sidecar with no field deserializes to `None` via `#[serde(default)]`, dispatch-identical to non-coopmat). The runtime builds the required shape FROM the metadata — nothing hardcoded.

### Device-feature enablement: the capability→feature pass

spirv-val validates a module but does NOT check that the device features its capabilities require are enabled. M3.1 adds a binding-plan-driven `required_device_features` pass:

| SPIR-V capability | trigger | Vulkan device feature |
|---|---|---|
| VulkanMemoryModel | coopmat op | VkPhysicalDeviceVulkanMemoryModelFeatures.vulkanMemoryModel (+DeviceScope) |
| CooperativeMatrixKHR | coopmat op | VK_KHR_cooperative_matrix ext + cooperativeMatrix feature |
| StorageBuffer16BitAccess | f16 SSBO | storageBuffer16BitAccess |
| StorageBuffer8BitAccess | u8/i8 SSBO | storageBuffer8BitAccess |
| Int8 | u8/i8 SSBO | shaderInt8 |
| Int16 | f16_bits_to_f32 | shaderInt16 |

The runtime enables the device-SUPPORTED subset at context creation (all feature-struct locals at function scope so they outlive `create_device`, chain assembled unconditionally), and a kernel that needs a feature the device lacks FAILS CLOSED with `DeviceFeatureUnsupported` — never enable-and-hope.

### Preflight, subgroup guard, graceful skip

A preflight queries `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` for the device's supported `(M,N,K,AType,BType,CType,ResultType,scope,saturatingAccumulation)` tuples. A pure matcher checks the metadata-derived required shape against that set. matmul_tile requires `subgroupSize == 32`; wave64/SIMD16 devices are SKIPPED (not miscomputed). When unsupported, dispatch returns `DispatchError::CoopMatUnsupported`. The NVIDIA runner is a mandatory sign-off gate.

### Multi-row Q4_K_M matmul and staged coopmat bridge

`q4km_dequant_matmul.axc` extends M2.6's single-row matvec to N-row × M-col, bit-exact vs CPU reference for 256×256 on Lavapipe AND NVIDIA. A pre-dequantized coopmat bridge (`q4km_dequant_matmul_coopmat.axc`) is provided for NVIDIA-only proof without `shared[T,N]` (deferred to M3.2).

### GPU-resident benchmark methodology

upload_resident / dispatch_resident / readback_resident: inputs staged once, descriptor set bound to resident device-local buffers, each iteration records ONLY the compute submit timed by `vkCmdWriteTimestamp` × 2 (elapsed = (end−begin) · timestampPeriod, each endpoint masked to `timestampValidBits` before subtraction, CPU fallback). Effective TFLOPS = 2*M*N*K / kernel_seconds (the honest, reproducible in-tree number). The cuBLAS comparison is an external estimate, labeled as such.

### Lever A re-land

`binding_is_readback(access, output_size) = output_size > 0 && access != ReadOnly` applied to all four readback loops (HN-9 — all loops must agree). ReadOnly bindings skip the device→staging copy, HOST_READ barrier, and invalidate. Re-landed without the reverted ReBAR Lever B from M3.0.
