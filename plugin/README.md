# agentic-clustering

A Claude Code plugin for **iterative agentic discovery of natural clusters in text corpora**. Multiple specialised subagents — proposer, synthesizer, auditor, investigator, critic — collaborate under an orchestrator to converge on a stable, well-supported cluster taxonomy, which you can then **apply to your whole corpus** by classifying every text into it.

The workflow has two phases:

1. **Discover** a cluster taxonomy from a sample of your corpus.
2. **Classify** the full corpus into that taxonomy (optionally tuning the classifier against hand labels first).

## Prerequisites

**For discovery and labelling** (`/cluster-run`, `/cluster-status`, `/cluster-investigate`, `/cluster-finalize`, `/cluster-label`):

- **[`uv`](https://docs.astral.sh/uv/)** — the skill scripts run via `uv run` and resolve their own dependencies (PEP 723), so no manual `pip install` is needed.

**For classification and tuning** (`/cluster-classify`, `/cluster-tune`): everything above, **plus** an API key —

- `OPENAI_API_KEY` (default — uses GPT-5-mini, cheap and fast), **or**
- `ANTHROPIC_API_KEY` (uses Claude Haiku 4.5).


## Installation

Clone this repo and point Claude Code at the `plugin/` directory:

```text
claude --plugin-dir /path/to/agentic-clustering/plugin
```

The plugin loads for the current session; repeat the flag on subsequent launches or symlink it into your Claude Code plugin directory.

## Quick start

### Phase 1 — Discover a taxonomy

1. **`/cluster-run`** — point it at your corpus (a CSV/JSON file + the text column). It asks for a target cluster-count range, optional clustering instructions (e.g. *"cluster by issue type"*, *"group by sentiment"*), and a model tier. It then runs an iterative loop of specialised agents (proposer → synthesizer → auditor → investigator → critic), writing all working state to `.claude/clustering/`.
2. **`/cluster-status`** — check progress at any time (cluster count, coverage, confidence, cross-proposal agreement).
3. **`/cluster-investigate`** — dig into a specific cluster or question (the orchestrator also does this automatically; use this to steer it).
4. **`/cluster-finalize`** — runs a final auditor + critic review and exports the taxonomy:
   - `taxonomy.md` — the human-readable taxonomy
   - `final_taxonomy.json` — the same, for programmatic use

### Phase 2 — Classify your corpus

5. **`/cluster-classify`** — applies the finalized `taxonomy.md` to a corpus, classifying **every** text into a cluster (with a confidence score and reasoning) and writing a timestamped CSV under `.claude/clustering/classification/classifications/`. Pick an execution mode:
   - **`async`** — real-time, for small corpora (< ~1000 texts).
   - **`batch`** — the provider's Batch API: **~50% cheaper**, takes minutes–hours, best for full-corpus runs.

   Prompt caching is on by default, so cost drops sharply after the first call.

   **Optional — tune accuracy first.** Before a full classification run, you can validate and improve the classifier against hand labels:
   - **`/cluster-label`** — walks you through a sample of texts one at a time; you assign each a cluster (or `none`). Produces a `labels.json` validation set.
   - **`/cluster-tune`** — generates several prompt variants, scores each against your labels, and recommends the best one (`tuned_prompt.md`). `/cluster-classify` picks it up automatically on the next run.

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

All state is written to `.claude/clustering/` in the project you're analysing (override by setting `CLUSTERING_WORKSPACE` before launching Claude Code, or by answering `/cluster-run`'s "where should the workspace live?" prompt with a custom path).

Whether or not you use a custom workspace, two tiny pointer files (`.plugin_root` and `.active_workspace`) always live at `.claude/clustering/` — they're how Claude Code hooks and subagent contexts find the real workspace location. Don't delete the `.claude/clustering/` directory to "clean up" after a custom-workspace run; the pointer files there are still load-bearing.

**During discovery** (between `/cluster-run` and `/cluster-finalize`):

- `summary.md`, `run_log.md`, `plan.md` — live status, a session diary, and resume notes
- `state.json` — workspace state (clusters, evidence, metrics)
- `proposals/`, `audits/`, `investigations/`, `critiques/`, `metrics/` — intermediate per-agent outputs

**After `/cluster-finalize`** — the workspace root is cleaned up:

- `taxonomy.md` / `final_taxonomy.json` — the finalized taxonomy
- `state.json`, `corpus.json` — kept for `/cluster-label` and future re-finalize
- `seen_ids.json`, `log.jsonl` — kept so `/cluster-label` doesn't re-sample discovery-audited texts, and the action trace keeps appending across phases
- `plan.md` — kept as the orchestrator's forward-looking notes, available to a re-finalize or follow-up session
- `archive/` — every mid-discovery artifact above (proposals, audits, investigations, critiques, metrics, `summary.md`, `run_log.md`) moved here so the root stays clean

**Phase-2 outputs** (created by `/cluster-label`, `/cluster-tune`, `/cluster-classify`; preserved across `/cluster-finalize`):

- `classification/labels.json`, `classification/tuned_prompt.md` — labelling and tuning artifacts
- `classification/classifications/run_*.csv` — classification outputs (one timestamped file per run)

## How it works

Discovery is an orchestrated loop of specialised subagents — proposer, synthesizer, auditor, investigator, critic — that converges on a stable, well-supported taxonomy with measured coverage and cross-proposal agreement. Classification then applies that taxonomy at scale through a cheap external model, with prompt caching and schema-enforced outputs so every text lands in a valid cluster.

## Authors

- Emily Silcock <emilysilcock@fas.harvard.edu>
- Simon Löwe <loewe.sim@gmail.com>

## License

MIT — see [LICENSE](LICENSE).
