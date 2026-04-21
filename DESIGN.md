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
