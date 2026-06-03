# agentic-clustering

A Claude Code plugin for **iterative agentic discovery of natural clusters in text corpora**. Multiple specialised subagents — proposer, synthesizer, auditor, investigator, critic — collaborate under an orchestrator to converge on a stable, well-supported cluster taxonomy, which you can then **apply to your whole corpus** by classifying every text into it.

The workflow has two phases:

1. **Discover** a cluster taxonomy from a sample of your corpus.
2. **Classify** the full corpus into that taxonomy (optionally tuning the classifier against hand labels first).

## Prerequisites

**For discovery** (`/cluster-run`, `/cluster-status`, `/cluster-investigate`, `/cluster-finalize`):

- **[`uv`](https://docs.astral.sh/uv/)** — the skill scripts run via `uv run` and resolve their own dependencies (PEP 723), so no manual `pip install` is needed.

**For classification, labelling, and tuning** (`/classify-run`, `/classify-tune`, `/classify-label`): everything above, **plus** an API key —

- `OPENAI_API_KEY` (default — uses GPT-5-mini, cheap and fast), **or**
- `ANTHROPIC_API_KEY` (uses Claude Haiku 4.5).

The classify commands come from the **[`text-classification`](https://github.com/emilysilcock/text-classification)** plugin, which is installed automatically as a hard dependency of this plugin — you don't need to install it separately.

## Installation

Through Claude Code's marketplace mechanism. From a directory you trust:

```
/plugin marketplace add emilysilcock/econ-nlp-plugins
/plugin install agentic-clustering@econ-nlp-plugins
```

This auto-installs `text-classification` alongside it.

## Quick start

### Phase 1 — Discover a taxonomy

1. **`/cluster-run`** — point it at your corpus (a CSV/JSON file + the text column). It asks for a target cluster-count range, optional clustering instructions (e.g. *"cluster by issue type"*, *"group by sentiment"*), and a model tier. It then runs an iterative loop of specialised agents (proposer → synthesizer → auditor → investigator → critic), writing all working state to `.claude/clustering/`.
2. **`/cluster-status`** — check progress at any time (cluster count, coverage, confidence, cross-proposal agreement).
3. **`/cluster-investigate`** — dig into a specific cluster or question (the orchestrator also does this automatically; use this to steer it).
4. **`/cluster-finalize`** — runs a final auditor + critic review and exports:
   - `taxonomy.md` — the human-readable taxonomy
   - `final_taxonomy.json` — the same, for programmatic use
   - `categories.json` — the structured category definitions consumed by the classify commands below

### Phase 2 — Classify your corpus

5. **`/classify-run`** — applies the finalized `categories.json` to a corpus, classifying **every** text into a category (with a confidence score and reasoning) and writing a timestamped CSV under `.claude/clustering/classification/classifications/`. Pick an execution mode:
   - **`async`** — real-time, for small corpora (< ~1000 texts).
   - **`batch`** — the provider's Batch API: **~50% cheaper**, takes minutes–hours, best for full-corpus runs.

   Prompt caching is on by default, so cost drops sharply after the first call.

   **Optional — tune accuracy first.** Before a full classification run, you can validate and improve the classifier against hand labels:
   - **`/classify-label`** — walks you through a sample of texts one at a time; you assign each a category (or `none`). Produces a `labels.json` validation set.
   - **`/classify-tune`** — generates several prompt-header variants, scores each against your labels, and recommends the best one (saved as `classification/header.md`). `/classify-run` picks it up automatically on the next run.

   The classify commands auto-detect this workspace via `.claude/clustering/categories.json`, so no extra configuration is needed when you've just run `/cluster-finalize`.

## Commands at a glance

| Command | Phase | What it does | From |
|---|---|---|---|
| `/cluster-run` | discover | Iterative agentic discovery of clusters | this plugin |
| `/cluster-status` | discover | Show progress on the current workspace | this plugin |
| `/cluster-investigate` | discover | Investigate a specific cluster or question | this plugin |
| `/cluster-finalize` | discover | Final review + export `taxonomy.md` / `final_taxonomy.json` / `categories.json` | this plugin |
| `/cluster-report-issue` | any | File a GitHub issue against agentic-clustering with workspace context | this plugin |
| `/classify-label` | tune | Hand-label a validation sample → `labels.json` | text-classification |
| `/classify-tune` | tune | Tune the classification prompt against labels | text-classification |
| `/classify-run` | classify | Classify a corpus into the taxonomy → CSV | text-classification |
| `/classify-report-issue` | any | File a GitHub issue against text-classification | text-classification |

Commands may appear namespaced in the `/` menu as `/agentic-clustering:cluster-run` or `/text-classification:classify-run`, etc.

## Where things live

All state is written to `.claude/clustering/` in the project you're analysing (override by setting `CLUSTERING_WORKSPACE` before launching Claude Code, or by answering `/cluster-run`'s "where should the workspace live?" prompt with a custom path).

Whether or not you use a custom workspace, two tiny pointer files (`.plugin_root` and `.active_workspace`) always live at `.claude/clustering/` — they're how Claude Code hooks and subagent contexts find the real workspace location. Don't delete the `.claude/clustering/` directory to "clean up" after a custom-workspace run; the pointer files there are still load-bearing.

**During discovery** (between `/cluster-run` and `/cluster-finalize`):

- `summary.md`, `run_log.md`, `plan.md` — live status, a session diary, and resume notes
- `state.json` — workspace state (clusters, evidence, metrics)
- `proposals/`, `audits/`, `investigations/`, `critiques/`, `metrics/` — intermediate per-agent outputs

**After `/cluster-finalize`** — the workspace root is cleaned up:

- `taxonomy.md` / `final_taxonomy.json` / `categories.json` — the finalized taxonomy and the category definitions consumed by `/classify-*`
- `state.json`, `corpus.json` — kept as a reference for the original corpus and for re-finalize sessions
- `seen_ids.json`, `log.jsonl` — kept as a record of discovery-audited texts (so a re-finalize can draw fresh unseen samples) and so the action trace keeps appending across phases
- `plan.md` — kept as the orchestrator's forward-looking notes, available to a re-finalize or follow-up session
- `archive/` — every mid-discovery artifact above (proposals, audits, investigations, critiques, metrics, `summary.md`, `run_log.md`) moved here so the root stays clean

**Phase-2 outputs** (created by the text-classification plugin's `/classify-label`, `/classify-tune`, `/classify-run`; preserved across `/cluster-finalize`):

- `classification/labels.json` — hand labels from `/classify-label`
- `classification/header.md` — tuned classifier prompt header from `/classify-tune` (auto-picked up by `/classify-run`)
- `classification/classifications/run_*.csv` — classification outputs (one timestamped file per run)
- `classification/tuning/` — per-variant intermediate outputs from `/classify-tune`

## Reporting issues

If something goes wrong during a `/cluster-*` run, the orchestrator will
offer to file a GitHub issue for you. You can also invoke
**`/cluster-report-issue`** any time — it asks for a one-line title and a
short description, then attaches the workspace context (plugin commit,
`summary.md` tail, `log.jsonl` tail, and a `state.json` metrics snapshot)
and files the issue on `emilysilcock/agentic-clustering`. Raw corpus text
is never auto-attached.

If you have the [GitHub CLI](https://cli.github.com/) (`gh`) installed and
authenticated, the issue is filed directly and you get the issue URL back.
Otherwise the command prints a pre-filled
`github.com/.../issues/new?title=...&body=...` URL — open it in a browser
and click submit.

## How it works

Discovery is an orchestrated loop of specialised subagents — proposer, synthesizer, auditor, investigator, critic — that converges on a stable, well-supported taxonomy with measured coverage and cross-proposal agreement. Classification then applies that taxonomy at scale through a cheap external model, with prompt caching and schema-enforced outputs so every text lands in a valid cluster.

## Authors

- Emily Silcock <emilysilcock@fas.harvard.edu>
- Simon Löwe <loewe.sim@gmail.com>

## License

MIT — see [LICENSE](LICENSE).
