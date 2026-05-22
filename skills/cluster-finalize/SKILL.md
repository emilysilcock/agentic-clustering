---
name: cluster-finalize
description: Finalize clusters and produce output artifacts
allowed-tools: Task, Read, Bash, Write
---

The workspace defaults to `.claude/clustering/` — override with
`CLUSTERING_WORKSPACE` env var if a custom workspace was configured.

1. Read `<workspace>/summary.md` — check readiness
2. If there are open questions or low-confidence clusters, warn the user
3. Spawn the **critic** for a final review
4. Report critic findings — ask user to proceed or address issues
5. If proceeding:
   ```bash
   uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/state.py \
     finalize --output <workspace>/final_taxonomy.json --max-examples 5
   ```
6. Display `<workspace>/taxonomy.md` to the user — this is the primary
   human-readable artifact. The JSON (`final_taxonomy.json`) is for programmatic
   use.
7. Confirm the workspace is clean: after finalization, the workspace root should
   contain only `taxonomy.md`, `final_taxonomy.json`, `state.json`, and an
   `archive/` directory with all intermediate files.
