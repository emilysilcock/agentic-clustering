---
name: cluster-finalize
description: >
  Wrap up a clustering session: dispatch a final auditor + critic, then export
  the finalized taxonomy (taxonomy.md + final_taxonomy.json) and archive
  intermediate proposals/audits/investigations/critiques. Use when the user
  is done iterating and wants the deliverable artifacts.
allowed-tools: Task, Read, Bash, Write
---

Scripts live at `$CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/`. The
workspace defaults to `.claude/clustering/`; override with
`$CLUSTERING_WORKSPACE`. Resolve both before any script call:

```bash
if [ -z "$CLAUDE_PLUGIN_ROOT" ]; then
  export CLAUDE_PLUGIN_ROOT=$(cat .claude/clustering/.plugin_root 2>/dev/null)
fi
if [ -z "$CLUSTERING_WORKSPACE" ]; then
  export CLUSTERING_WORKSPACE=$(cat .claude/clustering/.active_workspace 2>/dev/null || echo .claude/clustering)
fi
```

1. Read `$CLUSTERING_WORKSPACE/summary.md` — check readiness
2. If there are open questions or low-confidence clusters, warn the user
3. **Dispatch a final auditor** to refresh coverage and per-cluster confidence
   against the *current* cluster set. Without this, `taxonomy.md`'s header
   numbers reflect whatever the last audit said, which may have been for an
   older cluster version. Use a comfortable sample (60–150 fresh texts,
   depending on text length).
4. Spawn the **critic** for a final adversarial review of the (now freshly-
   audited) cluster set.
5. Report critic findings — ask user to proceed or address issues
6. If proceeding:
   ```bash
   uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/state.py \
     finalize --output $CLUSTERING_WORKSPACE/final_taxonomy.json --max-examples 5
   ```
7. Display `$CLUSTERING_WORKSPACE/taxonomy.md` to the user — this is the
   primary human-readable artifact. The JSON (`final_taxonomy.json`) is for
   programmatic use.
8. Confirm the workspace is clean: after finalization, the workspace root
   contains the output artifacts (`taxonomy.md`, `final_taxonomy.json`),
   `state.json`, `corpus.json` (kept so `/cluster-label` can still sample the
   original corpus), the internal `.state.lock` / `.plugin_root` /
   `.active_workspace` files, and an `archive/` directory holding all
   intermediate files (proposals, audits, investigations, critiques, metrics,
   logs, and the final `summary.md`).
