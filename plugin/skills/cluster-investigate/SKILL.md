---
name: cluster-investigate
description: >
  User-directed deep-dive into a specific cluster or question.
allowed-tools: Task, Read, Bash
---

The user wants to investigate something specific. Scripts live at
`$CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/`. The workspace defaults to
`.claude/clustering/`; override with `$CLUSTERING_WORKSPACE`. Resolve both
before any script call:

```bash
if [ -z "$CLAUDE_PLUGIN_ROOT" ]; then
  export CLAUDE_PLUGIN_ROOT=$(cat .claude/clustering/.plugin_root 2>/dev/null)
fi
if [ -z "$CLUSTERING_WORKSPACE" ]; then
  export CLUSTERING_WORKSPACE=$(cat .claude/clustering/.active_workspace 2>/dev/null || echo .claude/clustering)
fi
```

1. Read `$CLUSTERING_WORKSPACE/summary.md` for context
2. Clarify the question if needed
3. Spawn the **investigator** agent with the specific question
4. Report findings to the user
5. If the investigator recommends an action (merge/split/etc.), ask the user
   if they want to apply it
6. If yes — find the file the investigator just wrote (it's the newest
   `inv_*.json` in `$CLUSTERING_WORKSPACE/investigations/`) and run:
   ```bash
   uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/state.py \
     apply-recommendation $CLUSTERING_WORKSPACE/investigations/<latest_inv_file>.json
   ```
