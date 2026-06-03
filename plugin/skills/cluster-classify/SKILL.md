---
name: cluster-classify
description: >
  Apply a finalized cluster taxonomy to a corpus by classifying each text into
  one of the clusters via Claude or GPT, with prompt caching and structured
  output enforcement. Optionally use a tuned prompt from cluster-tune.
allowed-tools: Bash, Read, Write
---

# Cluster Classification

Apply a finalized taxonomy to a new corpus. The skill builds a system prompt
from `taxonomy.md` (with example texts stripped), then runs each text
through the chosen provider/model. Output is a CSV with cluster, confidence,
and reasoning per text.

## Environment

Scripts at `$CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/`. Workspace
defaults to `.claude/clustering/` but can be overridden via
`$CLUSTERING_WORKSPACE`. Verify both resolve before any script call:

```bash
if [ -z "$CLAUDE_PLUGIN_ROOT" ]; then
  export CLAUDE_PLUGIN_ROOT=$(cat .claude/clustering/.plugin_root 2>/dev/null)
fi
if [ -z "$CLUSTERING_WORKSPACE" ]; then
  export CLUSTERING_WORKSPACE=$(cat .claude/clustering/.active_workspace 2>/dev/null || echo .claude/clustering)
fi
```

## API keys

Require one of these environment variables, depending on `--provider`:
- `OPENAI_API_KEY` (default; uses GPT-5-mini — cheap, fast, automatic prompt cache works at any prompt size ≥1024 tokens)
- `ANTHROPIC_API_KEY` (uses Claude Haiku 4.5 — comparable cost on large prompts but cache requires ~4096-token minimum, so small-`k` taxonomies don't cache)

If the relevant key is not set, stop and tell the user to set it.

## Workflow

### 1. Verify prerequisites

Check that the workspace contains a finalized taxonomy:
```bash
test -f $CLUSTERING_WORKSPACE/taxonomy.md || echo "no taxonomy"
```

If missing, tell the user to run `/cluster-finalize` first.

### 2. Ensure a classification prompt exists

Default location: `<workspace>/classification/prompt.md`.

If the user has run `/cluster-tune`, there will also be
`<workspace>/classification/tuned_prompt.md` — prefer this when available.

If no prompt exists, build the default:
```bash
uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/build_classification_prompt.py \
  --taxonomy $CLUSTERING_WORKSPACE/taxonomy.md \
  --output $CLUSTERING_WORKSPACE/classification/prompt.md
```

### 3. Get input corpus from user

Ask: **what corpus to classify?** Need a CSV or JSON path with at least a text
column (and ideally an ID column). Confirm:
- File path
- Text column name
- ID column name. If the corpus has no ID column, the user must say so
  explicitly — pass `--no-id` to `classify.py` for a row-index fallback. A
  typoed `--id-col` value is a fatal error, not a silent index fallback.
- **Should every text be assigned?** If the user's downstream pipeline can't
  cope with a `none` label (e.g. classifying into a fixed taxonomy with no
  out-of-scope class), set `--force-assign` on both `build_classification_prompt.py`
  and `classify.py` in the next steps. Defaults to allowing `none`.

### 4. Choose execution mode

Ask the user (or pick a default):
- **`async`** — real-time with concurrency cap (~20 in parallel). Use for
  small corpora (< 1000 texts) or when you want fast turnaround.
- **`batch`** — provider Batch API. **50% cheaper** but takes minutes to hours.
  Use for full-corpus runs.

For corpora over ~1000 texts, default to `batch` and tell the user why (cost
saving). Confirm before submitting.

### 5. Run classification

Small-corpus / quick-turnaround run (async with concurrency cap):

```bash
RUN_NAME=run_$(date -u +%Y%m%dT%H%M%SZ)
uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/classify.py \
  --input <corpus_path> \
  --text-col <text_col> --id-col <id_col> \
  --prompt $CLUSTERING_WORKSPACE/classification/prompt.md \
  --output $CLUSTERING_WORKSPACE/classification/classifications/${RUN_NAME}.csv \
  --provider openai --model gpt-5-mini \
  --mode async --concurrency 20
```

For large-corpus runs (the default above ~1000 texts), swap
`--mode async --concurrency 20` for `--mode batch` — same script, ~50%
cheaper, ≤24h SLA. Substitute the prompt path with `tuned_prompt.md` when
present.

### 6. Report

Read the script's stderr summary (totals, errors, token usage). Surface to the
user:
- Number classified, errors
- Cache hit rate (large = caching is working; near 0% means the corpus is too
  small to cross the per-model cache threshold, or the prompt is changing
  between runs)
- Output path

Show a small sample (first 10 rows) so the user can sanity-check labels.

## Notes

- **Prompt caching is on by default** — the system prompt is sent with
  `cache_control: ephemeral` on every call, which cuts cost ~10× for the
  cached portion after the first request. Verify it's working via the
  `cache_read_tokens` column.
- **Structured outputs are enforced** — the JSON schema constrains the
  `cluster` field to the IDs defined in the taxonomy, so the output is always
  parseable and assigns a valid cluster.
- **`--force-assign`** — drops `"none"` from the schema enum and swaps in a
  stricter prompt header so every text must be assigned to a real cluster.
  Pass it on both `build_classification_prompt.py` *and* `classify.py` (the
  prompt and the schema must agree). Use when the user's downstream pipeline
  has no out-of-scope handling.
- **Re-runs are non-destructive** — each run gets its own timestamped output
  file under `classifications/`. Don't overwrite prior runs.

## When something goes wrong

If `classify.py` exits with errors you can't diagnose (auth failures aside —
those are the user's to fix), the provider returns malformed structured
output repeatedly, or batch retrieval hangs / fails, ask the user once
whether to file a GitHub issue with the workspace context attached. On yes,
invoke `/cluster-report-issue` (or call
`$CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/report_issue.py` directly).
Skip the offer for missing API keys, unsupported file formats, or anything
the error message itself tells the user how to fix.
