---
name: cluster-label
description: >
  Interactively label sample texts with cluster IDs to create a validation
  set. The user reviews each text in chat and assigns a cluster (or "none").
  Saves labels for use by cluster-tune.
allowed-tools: Bash, Read, Write
---

# Interactive Labelling

Build a labelled validation set by walking the user through sample texts one
at a time. Output is `labels.json` in the workspace, consumed by
`cluster-tune` and `evaluate_prompt.py`.

## Environment

Scripts at `$CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/`. Workspace
defaults to `.claude/clustering/` or `$CLUSTERING_WORKSPACE`. Verify
`$CLAUDE_PLUGIN_ROOT` before any script call (same boilerplate as
cluster-run).

## Workflow

### 1. Verify prerequisites

`taxonomy.md` must exist in the workspace. If not, tell the user to run
`/cluster-finalize` first.

### 2. Decide the sample

Ask the user how many texts to label. Defaults:
- Minimum useful: 30
- Recommended: 50–100
- For tight tuning of header variants: 100

Ask whether to draw the sample from:
- The **clustering corpus** (the one passed to `/cluster-run`) — most common,
  uses `sample.py` directly
- A **separate file** the user provides (e.g., a held-out set) — read texts
  from that path

For the clustering-corpus case, sample N fresh (unseen) texts:
```bash
uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/sample.py \
  --n 50 --strategy random > /tmp/labelling_sample.json
```

For the separate-file case, ask the user for path + text column + (optional)
id column, and read directly with `cat`/`Read`.

### 3. Read the cluster definitions

Read `taxonomy.md` so you can show definitions to the user as they label.
Strip the `**Examples:**` blocks before showing — examples can prime the
labeller and bias the validation set.

### 4. Walk through each text

Initialize an empty `labels = []`.

For each sampled text, present in this format:

> **Text 3 of 50** (id: `abc-123`)
>
> _[the text, possibly truncated to ~500 chars with a note if longer]_
>
> Available clusters:
> - `c1` — Name: short description
> - `c2` — ...
> - `none` — does not fit any cluster
>
> **Which cluster?** (reply with the ID, or `skip`, or `quit` to save and stop)

Wait for the user's reply. Validate it's one of the cluster IDs, `none`,
`skip`, or `quit`. Re-prompt if invalid.

- On a valid label: append `{"id": ..., "cluster": ..., "text": ...}` to
  `labels`. Save to `<workspace>/classification/labels.json` after each
  response (incremental save — don't lose progress if the session drops).
- On `skip`: do not record, move to the next text.
- On `quit`: save and break.

### 5. Save and report

Final write to `<workspace>/classification/labels.json` (it should already be
there from incremental saves). Report:
- How many labelled, how many skipped
- Distribution across clusters (catch under-represented ones)
- Path to labels.json

If any cluster has 0 labels, warn the user — tuning quality suffers when
clusters are missing from the validation set.

## Notes

- **Save incrementally.** Long labelling sessions get interrupted; the user
  shouldn't have to start over.
- **Don't cap text length silently.** If you truncate a text for display,
  say so explicitly: *"(truncated to first 500 chars)"*. The user might
  want to see more before deciding — accept a `more` reply to show the full
  text.
- **Don't suggest a label.** This is a validation set; suggestions bias it.
- **`labels.json` schema** — the format consumed downstream is either
  `{id: cluster_id, ...}` or `[{"id": ..., "cluster": ...}, ...]`. Use the
  list form so we can attach the text for later review.
