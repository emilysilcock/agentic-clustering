# agentic-clustering

A Claude Code plugin for **iterative agentic discovery of natural clusters in text corpora**. Multiple specialised subagents — proposer, synthesizer, auditor, investigator, critic — collaborate under an orchestrator to converge on a stable, well-supported cluster taxonomy, which you can then **apply to your whole corpus** by classifying every text into it.

The workflow has two phases:

1. **Discover** a cluster taxonomy from a sample of your corpus.
2. **Classify** the full corpus into that taxonomy (optionally tuning the classifier against hand labels first).

## Prerequisites

- **Claude Code** with this plugin installed (see below).
- **[`uv`](https://docs.astral.sh/uv/)** — the skill scripts run via `uv run` and resolve their own dependencies (PEP 723), so no manual `pip install` is needed.
- **An API key — only for the classification phase** (`/cluster-classify` and `/cluster-tune`):
  - `OPENAI_API_KEY` (default — uses GPT-5-mini, cheap and fast), **or**
  - `ANTHROPIC_API_KEY` (uses Claude Haiku 4.5).
  - Discovery (`/cluster-run`) needs neither — it runs on Claude Code's own subagents.

## Installation

```text
/plugin marketplace add emilysilcock/econ-nlp-plugins
/plugin install agentic-clustering@econ-nlp-plugins
```

For local development from a clone of this repo:

```text
claude --plugin-dir /path/to/agentic-clustering/plugin
```

## Quick start

### Phase 1 — Discover a taxonomy

1. **`/cluster-run`** — point it at your corpus (a CSV/JSON file + the text column). It asks for a target cluster-count range, optional clustering instructions (e.g. *"cluster by issue type"*, *"group by sentiment"*), and a model tier. It then runs an iterative loop of specialised agents (proposer → synthesizer → auditor → investigator → critic), writing all working state to `.claude/clustering/`.
2. **`/cluster-status`** — check progress at any time (cluster count, coverage, confidence, cross-proposal agreement).
3. **`/cluster-investigate`** — dig into a specific cluster or question (the orchestrator also does this automatically; use this to steer it).
4. **`/cluster-finalize`** — runs a final critic review and exports the taxonomy:
   - `taxonomy.md` — the human-readable taxonomy
   - `final_taxonomy.json` — the same, for programmatic use

### Phase 2 — Classify your corpus

5. **`/cluster-classify`** — applies the finalized `taxonomy.md` to a corpus, classifying **every** text into a cluster (with a confidence score and reasoning) and writing a timestamped CSV under `.claude/clustering/classification/`. Pick an execution mode:
   - **`async`** — real-time, for small corpora (< ~1000 texts).
   - **`batch`** — the provider's Batch API: **~50% cheaper**, takes minutes–hours, best for full-corpus runs.

   Prompt caching is on by default, so cost drops sharply after the first call.

### Optional — Tune classification accuracy first

If you want to validate and improve the classifier before a full run:

6. **`/cluster-label`** — walks you through a sample of texts one at a time; you assign each a cluster (or `none`). Produces a `labels.json` validation set.
7. **`/cluster-tune`** — generates several prompt variants, scores each against your labels, and recommends the best one (`tuned_prompt.md`). `/cluster-classify` picks it up automatically on the next run.

## Commands at a glance

| Command | Phase | What it does |
|---|---|---|
| `/cluster-run` | discover | Iterative agentic discovery of clusters |
| `/cluster-status` | discover | Show progress on the current workspace |
| `/cluster-investigate` | discover | Investigate a specific cluster or question |
| `/cluster-finalize` | discover | Final review + export `taxonomy.md` / `final_taxonomy.json` |
| `/cluster-label` | tune | Hand-label a validation sample → `labels.json` |
| `/cluster-tune` | tune | Tune the classification prompt against labels |
| `/cluster-classify` | classify | Classify a corpus into the taxonomy → CSV |

Commands may appear namespaced in the `/` menu as `/agentic-clustering:cluster-run`, etc.

## Where things live

All state is written to `.claude/clustering/` in the project you're analysing (override with the `CLUSTERING_WORKSPACE` env var or `--workspace` on `/cluster-run`):

- `taxonomy.md` / `final_taxonomy.json` — the finalized taxonomy
- `classification/classifications/run_*.csv` — classification outputs (one timestamped file per run)
- `classification/labels.json`, `classification/tuned_prompt.md` — tuning artifacts
- `summary.md`, `run_log.md`, `plan.md` — live status, a session diary, and resume notes

## How it works

Discovery is an orchestrated loop of specialised subagents — proposer, synthesizer, auditor, investigator, critic — that converges on a stable, well-supported taxonomy with measured coverage and cross-proposal agreement. Classification then applies that taxonomy at scale through a cheap external model, with prompt caching and schema-enforced outputs so every text lands in a valid cluster.

## Authors

- Emily Silcock <emilysilcock@fas.harvard.edu>
- Simon Löwe <loewe.sim@gmail.com>

## License

MIT — see [LICENSE](LICENSE).
