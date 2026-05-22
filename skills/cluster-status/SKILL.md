---
name: cluster-status
description: Show current cluster discovery status and progress
allowed-tools: Read, Bash
---

Read and display `.claude/clustering/summary.md`. If it doesn't exist, tell the
user the workspace hasn't been initialized and suggest `/cluster-run`.

Also show a brief action history from the last 5 entries in
`.claude/clustering/log.jsonl`.

If `.claude/clustering/run_log.md` exists, show the last few entries from it
as well — this gives the user a chronological trace of agent dispatches and
orchestrator decisions from the most recent run session.
