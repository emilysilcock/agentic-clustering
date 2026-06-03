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
   older cluster version. Use the same chars-per-text bracket as cluster-run's
   audit guidance (200-400 short, 50-150 medium, 20-50 long). One headline-
   stability override: for long-text corpora (500+ chars avg), push toward
   the high end of that range and don't drop below ~50 — the published
   coverage/confidence numbers shouldn't ride on a 20-text sample.
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
   original corpus), `seen_ids.json` (kept so `/cluster-label` skips
   discovery-audited texts), `log.jsonl` (kept so the chronological trace
   keeps appending across phases), `plan.md` (kept as the orchestrator's
   forward-looking notes so a re-finalize or follow-up session has the
   context), the internal `.state.lock` / `.plugin_root` /
   `.active_workspace` files, and an `archive/` directory holding all
   intermediate files (proposals, audits, investigations, critiques,
   metrics, `run_log.md`, and the final `summary.md`).

## When something goes wrong

If `state.py finalize` fails, the final auditor or critic returns
malformed output, or the user is unhappy with the taxonomy that came out,
ask once whether to file a GitHub issue with the workspace context
attached. On yes, invoke `/cluster-report-issue` (or call
`$CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/report_issue.py`
directly). Reserve this for things you can't fix locally — don't offer for
a critic that flags an issue the user can simply address by iterating.
