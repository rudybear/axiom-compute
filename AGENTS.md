# AXIOM-Compute for Agents

This guide is for **LLM agents** (Claude, GPT, custom) driving AXIOM-Compute to optimize GPU kernels. The compiler exposes its optimization loop over a Model Context Protocol (MCP) stdio bridge; an agent picks values for `@strategy` holes, receives measured GPU timings, and iterates.

---

## The optimization loop

```
.axc source with @strategy { x: ?[32, 64, 128] }  ← agent writes this
       │
       ▼
agent → `load_source` → {strategy_holes, binding_plan, workgroup, complexity}
       │
       ▼
agent reasons → picks candidate assignments
       │
       ▼
agent → `bench_variant({assignments})` → {median_ns, samples, correctness}
       │
       ▼
agent compares timings, proposes next strategy
       │
       ▼
agent → `grid_search` to enumerate ALL candidates at once (when cheap)
       │
       ▼
winner SPIR-V returned; agent can `compile_variant` for reproducibility
```

---

## Starting the server

```bash
# From workspace root
./target/release/axc mcp
```

Reads newline-delimited JSON-RPC 2.0 requests on stdin, writes responses on stdout. One request per line.

### Handshake

```json
→ {"jsonrpc":"2.0","id":1,"method":"initialize"}
← {"jsonrpc":"2.0","id":1,"result":{
     "server":"axc-mcp",
     "version":"0.1.0",
     "tools":["load_source","enumerate_variants","compile_variant",
              "bench_variant","grid_search","optimization_history"]
   }}
```

---

## Tools

### `load_source`

Parse a kernel source (or file path) and return its structure.

```json
→ {"jsonrpc":"2.0","id":2,"method":"load_source",
   "params":{"path":"examples/saxpy.axc"}}

← {"jsonrpc":"2.0","id":2,"result":{
     "kernel_name":"saxpy",
     "workgroup_size":[64,1,1],
     "binding_plan_summary":{
       "buffers":[
         {"name":"x","elem":"F32","access":"ReadOnly","binding":0},
         {"name":"y","elem":"F32","access":"ReadWrite","binding":1}
       ],
       "scalars":[{"name":"alpha","ty":"F32","offset":0}],
       "push_constant_total_bytes":4
     },
     "strategy_holes":{"workgroup_x":[32,64,128,256]},
     "complexity":"O(n)",
     "intent":"scaled vector add-accumulate"
   }}
```

Accepts **either** `source` (string) or `path` (file path) — exactly one. Path variant logs a warning if absolute/relative-dot path.

### `enumerate_variants`

Given source with `@strategy` holes, produce Cartesian-product assignments.

```json
→ {"method":"enumerate_variants","params":{"source":"..."}}

← {"variants":[
     {"variant_id":14863910742301845263,"assignments":{"workgroup_x":32}},
     {"variant_id":10938451025378249104,"assignments":{"workgroup_x":64}},
     ...
   ]}
```

`variant_id` is `xxh3_64` of the canonical assignments encoding — stable across machines.

### `compile_variant`

Substitute holes, compile, return SPIR-V.

```json
→ {"method":"compile_variant",
   "params":{"source":"...","assignments":{"workgroup_x":64}}}

← {"spirv_base64":"BwAjBwEAAA...",
   "metadata":{"kernel_name":"saxpy","workgroup_size":[64,1,1],...},
   "capabilities":["Shader"],
   "extensions":[],
   "size_bytes":488}
```

`spirv_base64` uses RFC 4648 standard alphabet with `=` padding. Decode to get the actual SPIR-V binary. spirv-tools validation passes in-process before return.

### `bench_variant`

Dispatch on the local GPU and measure. Requires Vulkan + `AXC_ENABLE_GPU_TESTS=1` server environment.

```json
→ {"method":"bench_variant",
   "params":{"source":"...",
             "assignments":{"workgroup_x":64},
             "input_sizes":[4096,4096],
             "sample_count":11}}

← {"median_ns":38104,
   "samples":[37891,38102,38221,...],
   "correctness":{"status":"ok"},
   "machine":{
     "os":"linux","rustc":"1.80.0",
     "vulkan_device":"NVIDIA RTX PRO 6000 Blackwell...",
     ...
   }}
```

`correctness` is tri-state:
- `{"status":"ok"}` — variant produced expected output (if oracle registered)
- `{"status":"failed","reason":"..."}` — numeric divergence beyond tolerance
- `{"status":"not_checked","reason":"..."}` — no CPU oracle for this kernel; valid variant, no verification

**Agents should treat `not_checked` as valid but unverified** — proceed with caution; do NOT discard.

### `grid_search`

End-to-end: enumerate + compile + bench all variants + pick winner. Best for small hole spaces (< 100 variants).

```json
→ {"method":"grid_search",
   "params":{"source":"...","sample_count":11}}

← {"winner":{"variant_id":10938...,"assignments":{"workgroup_x":64}},
   "winner_median_ns":38104,
   "ranked":[
     {"variant":..., "median_ns":38104,"correctness":{"status":"ok"}},
     {"variant":..., "median_ns":42330,"correctness":{"status":"ok"}},
     ...
   ]}
```

The result is also appended to `.pipeline/history/<source_xxh3>.jsonl` via POSIX `flock` (safe under concurrent grid searches).

Optional `holes_override` parameter **REPLACES** (does not merge) candidate lists:

```json
{"source":"...", "holes_override":{"workgroup_x":[128,256]}}
```

### `optimization_history`

```json
→ {"method":"optimization_history",
   "params":{"source_path":"examples/saxpy.axc"}}

← {"entries":[
     {"timestamp":"2026-04-21T14:30:15.123Z",
      "git_sha":"d476b5e",
      "source_xxh3":"abc123...",
      "grid_search":{winner:..., ranked:..., machine:...}},
     ...
   ]}
```

Append-only JSONL keyed by source content hash. Lets agents see prior optimization runs across sessions.

---

## Error codes

Standard JSON-RPC 2.0:

| Code | Meaning |
|---|---|
| -32700 | Parse error (invalid JSON) |
| -32600 | Invalid request (missing `jsonrpc: "2.0"`) |
| -32601 | Method not found |
| -32602 | Invalid params |
| -32603 | Internal error |

AXIOM-specific (-32001 to -32099):

| Code | Meaning |
|---|---|
| -32001 | Compile error (syntax / type / lowering) |
| -32002 | Enumerate error (bad `@strategy` shape) |
| -32003 | Grid search error |
| -32004 | Vulkan unavailable (no ICD / disabled via env) |
| -32005 | I/O error (file not found, permission) |
| -32006 | SPIR-V validator rejected emitted binary |

---

## Notifications (per JSON-RPC 2.0 §4.1)

A request **without** an `id` field is a notification — the server executes the handler for side effects and does NOT emit a response line. Use notifications for fire-and-forget operations you don't need to wait on.

Distinct from `"id": null`, which is still a normal request (server responds with `id: null`).

---

## Example: end-to-end LLM session

```python
import json, subprocess

p = subprocess.Popen(["./target/release/axc", "mcp"],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

def call(method, params=None):
    req = {"jsonrpc":"2.0","id":next_id(),"method":method}
    if params: req["params"] = params
    p.stdin.write(json.dumps(req) + "\n")
    p.stdin.flush()
    return json.loads(p.stdout.readline())

# 1. Handshake
print(call("initialize"))

# 2. Load kernel
info = call("load_source", {"path":"examples/saxpy.axc"})["result"]
print("holes:", info["strategy_holes"])

# 3. Let the LLM reason about which candidates to try
#    (either use the full grid or pick a promising subset)
result = call("grid_search", {
    "source": open("examples/saxpy.axc").read(),
    "sample_count": 11,
})["result"]

print(f"Winner: {result['winner']['assignments']} at {result['winner_median_ns']:,} ns")
```

---

## Tips for LLMs writing `.axc` kernels

### Favor explicit types

Every `let` must have `: type_annotation`. The typechecker won't infer.

```axc
let sum: f32 = 0.0f32;
let i: u32 = gid(0u32);  // gid takes u32 literal for compile-time axis constant
```

### Use `@strategy` liberally

Any tuning knob should be a hole. Workgroup size, tile dimensions, stage counts, vector widths, unroll factors — all candidates.

```axc
@strategy {
    workgroup_x: ?[32, 64, 128, 256],
    tile_m: ?[16, 32, 64],
    stages: ?[1, 2, 3],
}
@workgroup(?workgroup_x, 1, 1)
fn kernel(...) -> void { ... }
```

Cartesian product here = 4 × 3 × 3 = 36 variants. Reasonable for grid search.

### Use `@equiv_fp_tol` for FP kernels

```axc
@equiv_fp_tol(1e-3)  // LLM rewrites are verified within this tolerance
```

Prevents Sakana-style reward-hacking where a "faster" variant silently returns wrong output.

### `band` / `lshr` not `>>`

AXIOM-Compute has no `>>` operator by design (signed/unsigned ambiguity). Use explicit builtins:

```axc
let hi: u32 = band(lshr(byte, 4u32), 0x0Fu32);  // extract high nibble
```

### No `as` casts

Instead of `x as f32`, use typed builtin conversions:

```axc
let f: f32 = f32_from_u32(nibble);
```

### Use the M2.5 builtins for quantization

```axc
ptr_read_u8_zext(buf, offset)    // u8 load, zero-extend to u32
ptr_read_u16_zext(buf, offset)   // u16 LE load, zero-extend
f16_bits_to_f32(u32_bits)        // u16 → f16 → f32 via OpBitcast + OpFConvert
f32_from_u32(u)                  // OpConvertUToF
```

Q4_0 and Q4_K_M kernels in `examples/` demonstrate the full set.

---

## Architectural invariants (agents: do not violate)

1. **Never emit raw opcode/capability/decoration u32 values in tests.** Use typed `spirv::Op::X` enum API. The pipeline has caught 5+ silent false positives from this.
2. **Always `git status` + `git stash` check "No local changes to save" before claiming done.** Agents have left implementations uncommitted 3 times.
3. **`spirv_tools::val(TargetEnv::Vulkan_1_1)` is mandatory** on every integration test emitting SPIR-V. No `which spirv-val` shell-out — the crate is in-process.
4. **BTreeMap for anything driving emission order**, never HashMap.
5. **Cross-check against parent documentation.** This project's `CLAUDE.md` has the full design philosophy.

---

## When to NOT use AXIOM-Compute

- If you have access to NVIDIA-only tensor-core primitives and don't need portability → use CUDA C++ + CUTLASS
- If you just need to call existing cuBLAS/cuDNN → use those directly
- If your kernel is simple and hand-written CUDA works → keep the handwritten code
- If you need Metal/MSL specifically → use Apple's Shader Language

AXIOM-Compute's niche is: **portable SPIR-V across vendor GPUs + LLM-driven optimization loop + clean agent integration via MCP**. That's the thesis.

---

## Feedback

Open an issue at https://github.com/rudybear/axiom-compute/issues with the `agent-ux` label. Include the MCP request/response transcript.
