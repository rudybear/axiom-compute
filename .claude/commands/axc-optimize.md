---
description: Run AXIOM-Compute's grid-search autotuner on a .axc file and show the winning strategy assignments
---

I'll run the AXIOM-Compute optimizer on `$ARGUMENTS`.

Steps:
1. Verify the file exists and contains `@strategy` holes
2. Run `axc optimize $ARGUMENTS --output /tmp/winner.spv --bench-on-gpu`
3. Read the resulting `/tmp/winner.spv.axc.strategy.json` sidecar
4. Report the winning variant's assignments, median_ns, and the ranking of all variants
5. Suggest refined candidate lists for follow-up runs based on the ranking

This exercises the grid-search path. For MCP-driven exploration (fewer variants per iteration with LLM reasoning between calls), use the `axc` MCP server directly.
