# AXIOM-Compute Agentic Development Pipeline v2.0

Inherited from [AXIOM](https://github.com/rudybear/axiom/blob/master/.pipeline/PIPELINE.md) with GPU-specific adaptations. See `templates/` for agent prompts.

## Phase Flow

```
Phase 1: DESIGN
═══════════════════════════════════════════════════════════
                    ┌──────────────┐
                    │  ARCHITECT   │ Designs spec + plan + test requirements
                    └──────┬───────┘
                           │
              ┌────────────▼────────────────┐
              │     DESIGN REVIEW           │ (parallel)
              │  ┌──────────┐ ┌──────────┐  │
              │  │OPTIMISTIC│ │PESSIMISTIC│  │
              │  │ JSON ✓   │ │ JSON ✓    │  │
              │  └────┬─────┘ └────┬──────┘  │
              └───────┼────────────┼─────────┘
                      │ if any NEEDS_REVISION/REJECT
                      ▼
                  ┌────────────────┐
                  │ARCHITECT revise│  (max 3 cycles before human escalation)
                  └────────┬───────┘
                           │ both APPROVE
                           ▼
                  DESIGN.md + milestone.json updated

Phase 2: IMPLEMENTATION
═══════════════════════════════════════════════════════════
                   ┌─────────────────┐
                   │     CODER       │ Implements agreed plan
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │    QA AGENT     │ Independent diff verification
                   │  (never trust,  │ (memory rule: agents lie)
                   │   always verify)│
                   └────────┬────────┘
                            │

Phase 3: CODE REVIEW
═══════════════════════════════════════════════════════════
              ┌────────────▼────────────────┐
              │     CODE REVIEW             │ (parallel)
              │  ┌──────────┐ ┌──────────┐  │
              │  │OPTIMISTIC│ │PESSIMISTIC│  │
              │  │ JSON ✓   │ │ JSON ✓    │  │
              │  └────┬─────┘ └────┬──────┘  │
              └───────┼────────────┼─────────┘
                      └─────┬──────┘
                            │
              ACCEPT ───────┤────── CHANGES REQUESTED
                  │                     │
                  ▼                     ▼
              MERGE            Minor: Coder fixes
                              Major: back to Architect
```

## Verdict enforcement (MANDATORY)

Every reviewer output MUST begin with:

```json
{
  "verdict": "APPROVE" | "REJECT" | "NEEDS_REVISION" | "PASS" | "FAIL" | "REQUEST_CHANGES",
  ...
}
```

Supervisor:
1. Parses JSON.
2. Gates next phase on verdict.
3. On FAIL/REJECT, loops back to prior phase (max retry policy in `config.json`).

Empirical rule (verified multiple times): **agents lie about their own work.** Every implementation claim from the Coder is re-verified by an independent QA agent that reads `git diff` directly rather than trusting the Coder's self-report.

## GPU-specific differences from parent AXIOM pipeline

- Code reviewers must read emitted SPIR-V (`spirv-dis`) in addition to Rust source
- QA must run `spirv-val` on every emitted kernel
- Correctness tests require at least one real GPU execution (Lavapipe software fallback allowed in CI)
- Performance claims require measurements on at least 2 vendor GPUs (NVIDIA + AMD minimum for M2+)
- Memory cap: test GPU buffers must not exceed 50 GB total (rule inherited from AXIOM incident history)

## Retry policy

| Situation | Action | Max cycles |
|---|---|---|
| Design reviewer NEEDS_REVISION | Architect revises | 3 |
| Design reviewer REJECT | Architect fundamentally redesigns | 3 |
| QA FAIL (missing tests or missing diff) | Coder adds missing items | 3 |
| Code reviewer REQUEST_CHANGES | Coder fixes (minor) or Architect revises (design) | 2 |
| Code reviewer REJECT | Back to Architect | 2 |

## Directory layout

```
.pipeline/
├── PIPELINE.md          # This file
├── config.json          # Pipeline configuration
├── templates/           # Agent system prompts (7 files)
├── scripts/             # Orchestration helpers (ported from AXIOM)
├── milestones/          # Milestone JSON definitions
├── benchmarks/          # Performance baselines + KernelBench-Vulkan targets
└── runs/                # Per-execution state (gitignored)
```
