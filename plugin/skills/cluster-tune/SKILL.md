---
name: cluster-tune
description: >
  Tune the classification prompt by generating header variants, scoring each
  against human labels, and recommending the best. Requires labels.json from
  cluster-label.
allowed-tools: Bash, Read, Write
---

# Prompt Tuning

Find the best classification prompt header by generating variants tailored to
the taxonomy and observed errors, scoring each against human labels, and
recommending the winner.

## Environment

Scripts at `$CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/`. Workspace
defaults to `.claude/clustering/` or `$CLUSTERING_WORKSPACE`. Resolve both
before any script call:

```bash
if [ -z "$CLAUDE_PLUGIN_ROOT" ]; then export CLAUDE_PLUGIN_ROOT=$(cat .claude/clustering/.plugin_root 2>/dev/null); fi
if [ -z "$CLUSTERING_WORKSPACE" ]; then export CLUSTERING_WORKSPACE=$(cat .claude/clustering/.active_workspace 2>/dev/null || echo .claude/clustering); fi
```

## Workflow

### 1. Verify prerequisites

Check that the workspace contains:
- `taxonomy.md`
- `classification/labels.json` (from `/cluster-label`)

If `labels.json` is missing, tell the user to run `/cluster-label` first.

### 2. Run a baseline classification (default header)

This gives you (a) accuracy of the unmodified prompt and (b) a list of
disagreements that inform the variants.

```bash
uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/build_classification_prompt.py \
  --taxonomy $CLUSTERING_WORKSPACE/taxonomy.md \
  --output $CLUSTERING_WORKSPACE/classification/tuning/baseline_prompt.md

# Build a {id, text} corpus subset from labels.json (drops the cluster column
# so classify.py can consume it). The --corpus fallback recovers text bodies
# if labels.json was written in the dict shape.
uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/labels_to_corpus.py \
  --labels $CLUSTERING_WORKSPACE/classification/labels.json \
  --corpus $CLUSTERING_WORKSPACE/corpus.json \
  --output $CLUSTERING_WORKSPACE/classification/tuning/labelled_corpus.json

uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/classify.py \
  --input $CLUSTERING_WORKSPACE/classification/tuning/labelled_corpus.json \
  --text-col text --id-col id \
  --prompt $CLUSTERING_WORKSPACE/classification/tuning/baseline_prompt.md \
  --output $CLUSTERING_WORKSPACE/classification/tuning/baseline.csv \
  --provider openai --model gpt-5-mini --mode async

uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/evaluate_prompt.py \
  --predictions $CLUSTERING_WORKSPACE/classification/tuning/baseline.csv \
  --labels $CLUSTERING_WORKSPACE/classification/labels.json \
  --output $CLUSTERING_WORKSPACE/classification/tuning/baseline.eval.json
```

Read the eval output. Note the accuracy and inspect the `disagreements` list —
patterns in disagreements (e.g., "the model keeps confusing c3 with c5",
"unclear cases get assigned to a cluster instead of `none`") are what the
variants need to address.

### 3. Generate header variants

You (the orchestrator) generate **3-4 candidate headers**, conditioned on:
- The cluster taxonomy (read `taxonomy.md`)
- The baseline disagreements (read `baseline.eval.json`)

Aim for variants that target distinct hypothetical failure modes. Examples of
useful axes:
- **Strictness** — when in doubt, prefer `none` (reduces false positives on
  weak fits)
- **Focus** — classify by core narrative, not incidental details (reduces
  spurious cluster assignments based on a single keyword)
- **Boundary handling** — explicit instructions for distinguishing the
  cluster pairs that actually got confused in the baseline

Write each variant to a file:
```
<workspace>/classification/tuning/header_<name>.txt
```

Keep variants focused — a single, additional paragraph of guidance, not a
rewrite. The body of the prompt (cluster definitions) is generated from
`taxonomy.md` and stays the same across variants.

### 4. Score each variant

For each variant header:

```bash
NAME=strict
uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/build_classification_prompt.py \
  --taxonomy $CLUSTERING_WORKSPACE/taxonomy.md \
  --header  $CLUSTERING_WORKSPACE/classification/tuning/header_${NAME}.txt \
  --output  $CLUSTERING_WORKSPACE/classification/tuning/prompt_${NAME}.md

uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/classify.py \
  --input $CLUSTERING_WORKSPACE/classification/tuning/labelled_corpus.json \
  --text-col text --id-col id \
  --prompt $CLUSTERING_WORKSPACE/classification/tuning/prompt_${NAME}.md \
  --output $CLUSTERING_WORKSPACE/classification/tuning/run_${NAME}.csv \
  --provider openai --model gpt-5-mini --mode async

uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/evaluate_prompt.py \
  --predictions $CLUSTERING_WORKSPACE/classification/tuning/run_${NAME}.csv \
  --labels     $CLUSTERING_WORKSPACE/classification/labels.json \
  --output     $CLUSTERING_WORKSPACE/classification/tuning/eval_${NAME}.json
```

Run variants in parallel where possible (each `classify.py` call is
independent). Use the same provider and model for all variants for fair
comparison — provider changes prompt caching, schema enforcement, and
tokenization, so swapping it mid-experiment confounds the accuracy diff.

### 5. Compare and recommend

Read all eval JSON files, compare accuracy. Show the user a table:

```
header              accuracy   disagreements   notes
baseline            72%        14/50           keeps assigning weak fits to c3
strict              82%        9/50            ↓ false positives, ↑ none-rate
focus               78%        11/50           helps c3/c5 boundary
strict_focus        85%        7.5/50          best on this validation set
```

Recommend the best, but explicitly flag:
- Whether the difference is within noise (sample size matters — 50 labels
  means a single label's worth is 2%)
- Whether one variant is much better on a specific cluster the user cares
  about, even if not best overall
- That the user can override the recommendation

### 6. Save the chosen prompt

After the user confirms (or accepts the default recommendation), copy the
winning prompt into the location `/cluster-classify` looks for. Substitute
`<chosen>` below with the variant name you recommended (e.g. `strict_focus`).
Use a Python one-liner so it works on both Git Bash and vanilla PowerShell —
`cp` is not available in stock PowerShell. Paths are passed as argv (not
interpolated into the source string) so a space or quote in
`$CLUSTERING_WORKSPACE` can't break the call:

```bash
uv run python -c "import shutil, sys; shutil.copy(sys.argv[1], sys.argv[2])" \
  "$CLUSTERING_WORKSPACE/classification/tuning/prompt_<chosen>.md" \
  "$CLUSTERING_WORKSPACE/classification/tuned_prompt.md"
```

`/cluster-classify` will pick up `tuned_prompt.md` automatically when present.

## Notes

- **No majority-vote eval.** Single run per variant. Sample size is the
  trust signal — bigger labels.json = more reliable comparison.
- **Be honest about noise.** With 50 labels, accuracy differences under 5%
  are often not meaningful. Show absolute counts alongside percentages.
- **Variants don't need to win on overall accuracy** to be useful. If a
  variant resolves a specific confusion the user cares about (e.g., the
  cluster that downstream analysis hinges on), that may matter more than
  aggregate accuracy.
- **Cost** — N variants × M labels = N×M classifier calls per tuning run.
  With caching this is cheap, but warn the user if running > 4 variants on
  > 200 labels.
