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
defaults to `.claude/clustering/` or `$CLUSTERING_WORKSPACE`. Verify
`$CLAUDE_PLUGIN_ROOT` resolves before any script call:

```bash
if [ -z "$CLAUDE_PLUGIN_ROOT" ]; then
  export CLAUDE_PLUGIN_ROOT=$(cat .claude/clustering/.plugin_root 2>/dev/null)
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
- ID column name (or use row index if absent)

### 4. Choose execution mode

Ask the user (or pick a default):
- **`async`** — real-time with concurrency cap (~20 in parallel). Use for
  small corpora (< 1000 texts) or when you want fast turnaround.
- **`batch`** — provider Batch API. **50% cheaper** but takes minutes to hours.
  Use for full-corpus runs.

For corpora over ~1000 texts, default to `batch` and tell the user why (cost
saving). Confirm before submitting.

### 5. Run classification

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

Substitute the prompt path with `tuned_prompt.md` when present.

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
- **Re-runs are non-destructive** — each run gets its own timestamped output
  file under `classifications/`. Don't overwrite prior runs.
