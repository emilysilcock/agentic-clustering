---
name: corpus-tools
description: >
  Data access scripts for working with the corpus and workspace state.
  Preloaded into agents that need corpus access. Scripts handle ONLY pure
  data operations — no LLM calls.
allowed-tools: Bash
---

# Corpus Tools

All scripts are at `$CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/`. The
`$CLAUDE_PLUGIN_ROOT` environment variable is set automatically by Claude Code
and expands when you run Bash commands — use it as-is in your shell commands.

## Verifying Environment

`$CLAUDE_PLUGIN_ROOT` can be empty in some subagent contexts. Before running
any script, verify it resolves:
```bash
if [ -z "$CLAUDE_PLUGIN_ROOT" ]; then
  export CLAUDE_PLUGIN_ROOT=$(cat .claude/clustering/.plugin_root 2>/dev/null)
fi
```

The workspace defaults to `.claude/clustering/` in the project root. Override
with the `CLUSTERING_WORKSPACE` environment variable (all scripts read it
automatically).

## Initialization
```bash
# Set up workspace from a corpus file
uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/init.py \
  --corpus <path> --text-col <col> --k-range <min> <max> \
  --model-tier <quality|balanced|economy> --instructions "..." \
  --workspace <dir>  # optional, defaults to .claude/clustering
# Output: corpus stats (size, text length distribution), workspace created
```

## Sampling

If `config.max_texts_per_sample` is set in state.json, `sample.py` automatically
caps `--n` to that value (with a stderr note).

```bash
# Random sample (default: excludes previously seen texts)
uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/sample.py --n 50

# Targeted (texts similar to a query, via TF-IDF)
uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/sample.py \
  --n 30 --strategy targeted --query "billing overcharge"

# Targeted (texts assigned to a specific cluster in recent audits)
uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/sample.py \
  --n 30 --strategy cluster --cluster-id c3

# Fetch specific texts by ID (e.g., to verify merge candidates)
uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/sample.py \
  --ids id1 id2 id3

# Include previously seen texts (opt-in; default excludes them)
uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/sample.py \
  --n 50 --include-seen

# Output: JSON list of {id, text} objects to stdout
```

## Search
```bash
# TF-IDF similarity search (builds/caches TF-IDF matrix on first call)
uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/search.py \
  --query "login password reset" --n 10
# Output: JSON list of {id, text, similarity} objects
```

## State Management
```bash
# Regenerate summary.md from current state.json
uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/state.py summarize

# Increment proposal counter (called by proposer after writing output)
uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/state.py count-proposal

# Increment investigation counter (called by investigator/critic after writing output)
uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/state.py count-investigation

# Set clusters from synthesizer output (see Input Schemas section for format)
uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/state.py \
  set-clusters <clusters_json_file>

# Update state with audit results (coverage, confidence metrics)
uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/state.py \
  update-from-audit .claude/clustering/audits/<file>.json

# Apply an investigation recommendation (see Investigator output format)
uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/state.py \
  apply-recommendation .claude/clustering/investigations/<file>.json

# Mark texts as seen (called automatically by sample.py)
uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/state.py \
  mark-seen <ids...>

# Export final taxonomy (with example texts enrichment)
uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/state.py \
  finalize --output .claude/clustering/final_taxonomy.json --max-examples 5
```

## Input Schemas

### `set-clusters` format
The input JSON file must have a top-level `"clusters"` array. Each cluster
object requires `name` and `description`. For example text IDs, the script
accepts any of these key names (checked in order):
- `evidence_text_ids` (canonical)
- `example_ids`
- `text_ids` (recommended for proposer/synthesizer output)
- `example_text_ids`

```json
{
  "clusters": [
    {
      "name": "Billing disputes",
      "description": "Complaints about incorrect charges...",
      "text_ids": ["id1", "id2", "id3"],
      "confidence": "medium",
      "source_proposals": ["prop_001.json"]
    }
  ]
}
```

## Metrics
```bash
# Compute algorithmic metrics from audit data
uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/metrics.py
# Output: coverage, confidence distribution, cluster size distribution, etc.
```

## Cross-Proposal Agreement
```bash
# Compare all proposal pairs (ARI, NMI, entropy, element similarity)
uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/confusion.py cross-proposal
# Output: JSON report to .claude/clustering/metrics/, human-readable summary to stdout

# Per-text consistency analysis (standalone)
uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/confusion.py \
  element-similarity --source proposals
# Output: per-text similarity scores, most inconsistent texts

# Store metrics in state (for summary.md)
uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/state.py \
  update-cross-proposal-metrics .claude/clustering/metrics/<file>.json
```

## Classification

After `cluster-finalize` produces `taxonomy.md`, the classification stage
applies the taxonomy to a corpus.

```bash
# Build the classification system prompt from taxonomy.md
# (strips example texts; wraps cluster definitions with classifier instructions)
uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/build_classification_prompt.py \
  --taxonomy $CLUSTERING_WORKSPACE/taxonomy.md \
  --output   $CLUSTERING_WORKSPACE/classification/prompt.md
# Optional: --header <path> to override the default header
# Optional: --keep-examples to keep the **Examples:** blocks

# Classify a corpus. Supports openai (default, gpt-5-mini) and anthropic.
# Mode `batch` is ~50% cheaper but takes ≤24h (Anthropic-only today; OpenAI batch
# support coming).
uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/classify.py \
  --input <corpus.csv|json|jsonl> --text-col <text_col> --id-col <id_col> \
  --prompt $CLUSTERING_WORKSPACE/classification/prompt.md \
  --output $CLUSTERING_WORKSPACE/classification/classifications/<run>.csv \
  --provider openai --model gpt-5-mini \
  --mode async --concurrency 20
# Requires OPENAI_API_KEY (or ANTHROPIC_API_KEY for --provider anthropic).
# Prompt caching is on by default — verify via the cache_read_tokens column.
# OpenAI caches automatically once the cacheable prefix is ≥1024 tokens.
# Anthropic Haiku 4.5 needs ≥~4096 tokens to cache; small-k taxonomies don't qualify.
# Structured outputs guarantee the cluster field comes from the taxonomy's IDs.

# Evaluate predictions against human labels
uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/evaluate_prompt.py \
  --predictions <classifications.csv> \
  --labels      $CLUSTERING_WORKSPACE/classification/labels.json \
  --output      <eval.json>
# Output: accuracy, per-cluster precision/recall/F1, disagreement list.
# Labels JSON accepts {id: cluster} or [{"id": ..., "cluster": ...}, ...].
```
