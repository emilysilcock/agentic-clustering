# Huang & He baseline — harness-side substitutions

This baseline does **not patch** the vendored upstream. The files in
`_vendored/` are byte-identical to commit
`6d02bb0d0d8b08902aeeb3761be966f203d2af7f`. Everything below is a
description of what the *harness* does differently from what `run.sh`
would do if executed directly.

## Imports actually used from `_vendored/`

| Symbol | Source file | Why we use it |
|---|---|---|
| `prompt_construct_generate_label` | `label_generation.py` | Stage-1 prompt builder (preserves the upstream `"classicifation"` typo) |
| `prompt_construct_merge_label` | `label_generation.py` | Stage-1.5 merge prompt |
| `prompt_construct` | `given_label_classification.py` | Stage-2 classification prompt |

Everything else — `chat()`, `ini_client()`, `eval()`-based parsing, `main()`,
`evaluate.py`, `select_part_labels.py`, `run.sh` — is not imported and not
executed.

## What changes vs. upstream's `run.sh`

### Seed selection (`select_part_labels.py`)

- Upstream: per-dataset random sample of `0.2 * |gold labels|` names
  (with replacement, no `random.seed`) → seed list.
- Harness (SPEC §5.6.2 — 0%-seed configuration only): **empty seed list**.
  We never invoke `select_part_labels.py`; the per-dataset seed list passed
  into `prompt_construct_generate_label` as `given_labels=[]`.

### Label generation (`label_generation.py`)

- Upstream: sync OpenAI `gpt-3.5-turbo-0125`, `random.shuffle(data_list)`
  (unseeded), parse via `eval(response)`, silently drop batches whose
  response doesn't parse.
- Harness: **OpenAI Batch API**, `gpt-5-mini`, `random.seed(0)` before
  the shuffle, parse via `json.loads`, drop on parse failure (no
  re-prompt). Temperature is **not passed** — gpt-5-mini rejects the
  parameter as a reasoning model; the harness uses the model's internal
  fixed sampling strategy. Includes `is_none` documents end-to-end
  (SPEC §5.5).

### Label merging

- Upstream: same sync `gpt-3.5-turbo-0125` call as Stage 1.
- Harness: **single sync call to `claude-opus-4-7`** via the Claude Code
  Max subscription (`benchmarking.llm_clients.claude_code.call_claude`).
  One call per dataset (= 7 calls per sweep) → frontier tier per SPEC §5.6.

### Classification (`given_label_classification.py`)

- Upstream: sync sequential per-doc calls. `ini_client(args.api_key)` is
  called with an argument but the function takes none — script crashes
  on import-by-CLI.
- Harness: **OpenAI Batch API**, `gpt-5-mini`, auto-cache on the stable
  prefix (label list at the start of the user message, per-doc sentence
  at the end). Malformed JSON or out-of-list responses are mapped to
  `predicted_cluster_id = NONE_LABEL_ID (-1)` in the result parser so
  they show up in the partition metrics rather than being silently
  dropped from the denominator.

### Evaluation (`evaluate.py`)

- Upstream: O(N·K) string-membership scan over the prediction dict;
  truncates ground truth to `len(predict_labels)` so unparseable rows
  are dropped from both numerator and denominator → inflated scores.
- Harness: `benchmarking.evaluation.metrics.compute_partition_metrics`,
  which scores the full corpus (Hungarian ACC + NMI + ARI) with no
  silent drops.

## Cost accounting

Phases 1 + 3 are metered (OpenAI Batch). Phase 2 (~7 sync Opus calls per
sweep) is reported as `subscription_usd = $14.29` per dataset
(= $100/7), symmetric with `agentic_clustering` and `clusterllm` per
SPEC §5.6.3.
