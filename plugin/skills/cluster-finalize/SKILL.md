---
name: cluster-finalize
description: >
  Wrap up a clustering session: dispatch a final auditor + critic, then export
  the finalized taxonomy (taxonomy.md + final_taxonomy.json + categories.json)
  and archive intermediate proposals/audits/investigations/critiques. Use
  when the user is done iterating and wants the deliverable artifacts.
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
   older cluster version. Two draws, sized for two different questions:

   - **Coverage draw** — `--strategy random`, sized by cluster-run's
     chars-per-text bracket (200-400 short, 50-150 medium, 20-50 long). One
     headline-stability override: for long-text corpora (500+ chars avg),
     push toward the high end of that range and don't drop below ~50 — the
     published coverage/confidence numbers shouldn't ride on a 20-text
     sample. Written with `"sample_basis": "random"`.
   - **Per-cluster draw** — `--strategy stratified`, so
     every cluster's confidence label rests on a defensible n. A uniform draw
     allocates sample by corpus prevalence, so on a large taxonomy the rare
     clusters arrive at finalize with 0-2 texts each; `finalize` publishes a
     confidence label on every cluster, and it should not be publishing
     `[high]` off two texts. Only short clusters are served, so this is
     cheaper than it sounds. Written with `"sample_basis": "stratified"` plus
     the `strata_file` manifest path — the basis field stops this draw from
     contaminating coverage, and the manifest is what distinguishes a cluster
     that was tested and rejected from one that was never sampled.

4. **Clear the per-cluster audit floor before going further.** `finalize`
   refuses to export while any cluster's confidence label rests on fewer than
   5 audit assignments, and it refuses *before* writing or archiving anything,
   so a refusal costs nothing and the run simply continues. Read
   `summary.md`'s **Per-Cluster Audit Sample** section and work through it —
   this is a loop, not a checklist item, and it may take several rounds:

   - **under-sampled** — the corpus hasn't been searched for these. Draw
     another stratified pass and audit it. Repeat until the section is empty
     or stops shrinking. Each pass is smaller than the last, because only
     short clusters are served.
   - **unsupported** — candidates *were* aimed at these and the auditor
     assigned them elsewhere, so auditing again returns the same answer.
     Dispatch an **investigator** to find where the candidates went, then
     apply what it recommends: a neighbour absorbing them argues for a merge
     or a sharper boundary in both descriptions, nothing absorbing them argues
     for removal. A structural change resets coverage, so re-audit afterwards.

   Expect this to be the most expensive part of finalizing on a taxonomy with
   many clusters, and do it anyway — it is the difference between a taxonomy
   whose labels mean something and one that has to explain itself. Only bring
   it to the user if it stalls: a cluster that stays `unsupported` after an
   investigator pass is a real decision (drop it, or ship it unvalidated), and
   theirs to make.
5. Spawn the **critic** for a final adversarial review of the (now freshly-
   audited) cluster set.
6. Report critic findings — ask user to proceed or address issues
7. If proceeding:
   ```bash
   uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/state.py \
     finalize --output $CLUSTERING_WORKSPACE/final_taxonomy.json --max-examples 5
   ```
   If this refuses, go back to step 4 — don't reach for `--allow-unvalidated`
   to get past it. That flag ships the affected clusters labelled
   `unvalidated` with their n, named once at the top of `taxonomy.md`, and is
   only for a case the user has explicitly decided to ship unresolved.
   `--min-audit-n N` moves the floor; `--min-audit-n 0` removes it and
   publishes every label regardless of sample size.

   This also writes `$CLUSTERING_WORKSPACE/categories.json` — the canonical
   input for the **text-classification** plugin's `/classify-run`,
   `/classify-tune`, and `/classify-label` commands. By default it includes
   a `"none"` (out-of-scope) entry; pass `--no-none-category` to
   `state.py finalize` if every text in the downstream corpus must be
   assigned to a real cluster.
8. Display `$CLUSTERING_WORKSPACE/taxonomy.md` to the user — this is the
   primary human-readable artifact. `final_taxonomy.json` is for
   programmatic use; `categories.json` is for the classify commands.
9. Tell the user the phase-2 commands are available: `/classify-label`,
   `/classify-tune`, `/classify-run` — they pick up `categories.json`
   automatically because text-classification is installed as a hard
   dependency of this plugin and its workspace auto-detection follows the
   `.claude/clustering/.active_workspace` pointer to this workspace's
   `categories.json`.
10. Confirm the workspace is clean: after finalization, the workspace root
   contains the output artifacts (`taxonomy.md`, `final_taxonomy.json`,
   `categories.json`), `state.json`, `corpus.json` (kept as a reference for
   the original corpus and for re-finalize sessions), `seen_ids.json`
   (kept as a record of what was discovery-audited so a re-finalize can
   draw fresh unseen samples), `log.jsonl` (kept so the chronological trace
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
