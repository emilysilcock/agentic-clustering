---
name: cluster-investigate
description: >
  User-directed deep-dive into a specific cluster or question.
allowed-tools: Task, Read, Bash
---

The user wants to investigate something specific.

1. Read `.claude/clustering/summary.md` for context
2. Clarify the question if needed
3. Spawn the **investigator** agent with the specific question
4. Report findings to the user
5. If the investigator recommends an action (merge/split/etc.), ask the user
   if they want to apply it
6. If yes:
   ```bash
   uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/state.py \
     apply-recommendation .claude/clustering/investigations/<latest>.json
   ```
