---
name: cluster-status
description: Show current cluster discovery status and progress
allowed-tools: Read, Bash
---

The workspace defaults to `.claude/clustering/`; override with
`$CLUSTERING_WORKSPACE`. Before any file read, resolve it:

```bash
if [ -z "$CLUSTERING_WORKSPACE" ]; then
  export CLUSTERING_WORKSPACE=$(cat .claude/clustering/.active_workspace 2>/dev/null || echo .claude/clustering)
fi
```

Read and display `$CLUSTERING_WORKSPACE/summary.md`. If it doesn't exist, tell
the user the workspace hasn't been initialized and suggest `/cluster-run`.

Also show a brief action history from the last 5 entries in
`$CLUSTERING_WORKSPACE/log.jsonl`.

If `$CLUSTERING_WORKSPACE/run_log.md` exists, show the last few entries from it
as well — this gives the user a chronological trace of agent dispatches and
orchestrator decisions from the most recent run session.
