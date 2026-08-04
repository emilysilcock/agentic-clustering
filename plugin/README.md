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

For local development from a clone of [the repo](https://github.com/emilysilcock/agentic-clustering):

```
claude --plugin-dir /path/to/agentic-clustering/plugin
```

## Quick start

Three commands. Everything else in this README is optional detail.

1. **`/cluster-run`** — answer the seven setup questions (where the workspace goes, which file and column your texts are in, what the texts are, what to group them by, how many clusters you want, which model tier), then **step away**. The loop runs unattended: the orchestrator samples texts, dispatches proposers, synthesizes, audits, investigates, and critiques on its own, and converges without you — nudging it between iterations is usually counterproductive. Expect on the order of 15–20 minutes for a small corpus.
2. **`/cluster-finalize`** — once the run reports convergence. Runs a final review, then exports the deliverables: `taxonomy.md` (human-readable), `final_taxonomy.json` (structured), and `categories.json` (the input for step 3).
3. **`/classify-run`** — needs an API key (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`). Applies the finalized taxonomy to **every** text in your corpus and writes a CSV with a cluster, a confidence score, and reasoning per text.

Of the seven questions, two do most of the work: the **cluster-count range** (the granularity budget — you give a min and max, the loop settles on the count the data supports) and the **clustering lens** (one sentence saying what to group by, e.g. *"cluster by issue type"* — the same corpus clusters completely differently depending on the lens). Both offer presets; pick Other to write your own. See [Shaping the result](#shaping-the-result-the-two-answers-that-matter) for how to choose them well.

## Try it out — a worked example

A small corpus you can run end-to-end in about half an hour to see the workflow in action. The corpus is 18 open-ended survey responses to *"What is the most important problem facing the country today?"* — the canonical Gallup MIP item — and falls into three clear themes (economy, healthcare, climate). Discovery should converge on `k = 3` with high cross-proposal agreement.

### 1. Save the corpus

Save the following as `mip_responses.csv` in any directory you trust:

```csv
id,text
1,"Wages haven't kept up with the cost of groceries and rent for years."
2,"I can barely afford gas and basic bills, prices keep climbing."
3,"Good jobs are disappearing from my town and nothing is replacing them."
4,"Inflation is eating away at every paycheck I bring home."
5,"Young people can't afford houses anymore, the economy is broken."
6,"Stagnant wages and rising prices, that's what's killing the middle class."
7,"Insurance premiums keep going up and the coverage keeps getting worse."
8,"I had to skip my medication last month because I couldn't afford it."
9,"Hospital bills are bankrupting families even with so-called good insurance."
10,"Prescription drug prices in this country are completely out of control."
11,"My doctor's appointment took six months to schedule, the system is overwhelmed."
12,"Mental health care is impossible to access unless you're wealthy."
13,"The wildfires near my home get worse every single year now."
14,"We're not doing anything serious about climate change before it's too late."
15,"Water quality where I live has been deteriorating for a decade."
16,"Extreme weather keeps destroying communities and we just rebuild and wait for the next one."
17,"Air pollution near the highways is making my kids sick."
18,"Future generations will inherit a planet we made unlivable, and we know it."
```

### 2. Discover the taxonomy

In a Claude Code session opened in the directory containing `mip_responses.csv`:

```
/cluster-run
```

The orchestrator will ask seven setup questions. For this run, use:

| Question | Example user answer |
|---|---|
| Where should the workspace live? | `./clustering/` (default) |
| Where is your corpus? | `mip_responses.csv` |
| Which column holds the text? | `text` |
| What are these texts? | `survey responses on the most important problem facing the country` |
| What should the texts be grouped by? | `cluster by the type of problem the respondent describes` |
| Cluster count range | `2–8` (the "Broad" preset) |
| Model tier | `quality` (default) |

The run takes on the order of 15–20 minutes on this corpus (it dispatches six to seven proposers plus audit and critique passes). You should see the orchestrator dispatch proposers, then a synthesizer, then an auditor and critic, iterating until coverage and cross-proposal agreement both look stable. Expect roughly three clusters, ~100% coverage, and high mean confidence on this corpus.

Use **`/cluster-status`** at any time to peek at the live numbers.

### 3. Finalize the taxonomy

```
/cluster-finalize
```

This dispatches a final auditor + critic, then exports three artifacts in the workspace (`./clustering/` by default):

- `taxonomy.md` — the human-readable taxonomy with a short description and example texts per cluster
- `final_taxonomy.json` — the same content, structured
- `categories.json` — the cluster definitions consumed by the classify commands

`taxonomy.md` will look something like:

```markdown
# Cluster Taxonomy

## c1 — Economic hardship
Concerns about wages, prices, housing affordability, and job availability …

## c2 — Healthcare access and cost
Difficulty affording insurance, prescriptions, and timely medical care …

## c3 — Climate and environmental decline
Worsening wildfires, extreme weather, pollution, and long-term climate risk …
```

### 4. Classify the corpus into the taxonomy

With an API key set (`OPENAI_API_KEY` for GPT-5-mini, or `ANTHROPIC_API_KEY` for Haiku 4.5):

```
/classify-run
```

It auto-detects `clustering/categories.json` from the step above, so you only need to confirm the corpus path (`mip_responses.csv`) and text column (`text`). Pick **`async`** mode — the corpus is tiny. The output is a timestamped CSV under `clustering/classification/classifications/run_<timestamp>.csv` with one row per text: the assigned cluster id, the cluster name, a confidence score, and the model's reasoning. On this corpus you should see all 18 texts land in `c1`/`c2`/`c3` matching the obvious theme.

That's the full discover → finalize → classify loop. Swap in your own corpus and instructions to use it for real.

## Power users

A normal run needs none of what follows — most runs go straight from `/cluster-run` to `/cluster-finalize` to `/classify-run`. This section is for shaping, steering, and scaling beyond the defaults.

### Shaping the result: the two answers that matter

The setup questions are where you actually shape the result, and two of them do most of the work. `/cluster-run` **will** ask you both — and while each offers preset options, those are just starting points: pick Other and anything you can express in a sentence is a valid answer.

**1. The cluster-count range.** You give a min and a max, not a fixed `k`; the loop searches within it. The range is a budget for how coarse or fine the taxonomy should be. The question offers three ballpark presets:

| Range | What you get |
|---|---|
| `Broad: 2–8` | a handful of broad themes |
| `Mid: 10–30` | a working taxonomy for a coding scheme |
| `Fine: 60–150` | fine-grained, closer to a labelled category list |

The presets are just ballparks — pick Other to type any exact range you like (e.g. `4–6`, `25–40`, `80–150`).

Pick it from what you'll *do* with the clusters — a summary report needs far fewer than a ticket-routing system. If you genuinely don't know, give a wide range and narrow it on a second run once you've seen the first taxonomy.

**2. The clustering lens.** What should the texts be grouped by — a free-text instruction telling every agent what to pay attention to. Optional, but the single highest-leverage input: the same corpus clusters completely differently depending on what you ask for. The run drafts a few candidate lenses from a peek at your corpus; pick one or write your own.

```text
cluster by the type of problem the respondent describes
group by sentiment, not topic
focus on actionable categories a support team could route tickets to
distinguish by policy area; ignore which politician is mentioned
split on the mechanism of harm described, not the industry
```

Your lens — combined with the one-sentence dataset description asked just before it — is carried into every proposer, synthesizer, auditor, investigator, and critic dispatch and acts as the **primary constraint** on cluster formation — say "cluster by issue type" and the agents won't cluster by sentiment. Skip it and they'll discover whatever structure is most salient in the data, which is a reasonable starting point but rarely the one you actually wanted. One sentence here is usually worth more than a longer run.

Both answers are stored in the workspace's `state.json`, so resuming a session picks them back up. To try a different lens or a different granularity, start a fresh run — they're cheap to compare.

### Watching and steering a run

- **`/cluster-status`** — check progress at any time (cluster count, coverage, confidence, cross-proposal agreement).
- **`/cluster-investigate`** — dig into a specific cluster or question (the orchestrator also investigates automatically; use this to steer it, e.g. when you disagree with a call it made).

The orchestrator decides on its own when to pull a fresh sample, when to propose new clusters, when to merge or split them, and when to send an auditor, investigator, or critic after a weak spot. These commands are there if you want them, not because a normal run needs them.

### Choosing a model tier

The model tier is the cost knob for discovery: `quality` (the default) runs every agent on the same model as your session, `balanced` moves the high-volume reading agents (proposers, auditor) to Haiku, and `economy` runs all agents on Haiku.

### Classification: modes, cost, and tuning

`/classify-run` offers two execution modes:

- **`async`** — real-time, for small corpora (< ~1000 texts).
- **`batch`** — the provider's Batch API: **~50% cheaper**, takes minutes to hours (≤ 24h SLA), best for full-corpus runs.

Prompt caching is on by default, so cost drops sharply after the first call.

**Tune accuracy first (optional).** Before a full classification run, you can validate and improve the classifier against hand labels:

- **`/classify-label`** — walks you through a sample of texts one at a time; you assign each a category (or `none`). Produces a `labels.json` validation set.
- **`/classify-tune`** — generates several prompt-header variants, scores each against your labels, and recommends the best one (saved as `classification/header.md`). `/classify-run` picks it up automatically on the next run.

The classify commands auto-detect the clustering workspace via the `.claude/clustering/.active_workspace` pointer, so no extra configuration is needed when you've just run `/cluster-finalize`.

### Commands at a glance

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

### Where things live

All state is written to the workspace you chose at setup — `./clustering/` at the root of the project you're analysing by default, or `.claude/clustering/` if you picked the hidden option. For any other location, pick Other on the workspace question, or set `CLUSTERING_WORKSPACE` before launching Claude Code.

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

## This repository

This README is the single source of documentation for the repo. The plugin itself lives in `plugin/` and is the only thing shipped to plugin users; the rest of the repository is the experimental evaluation for the paper introducing the method.

```
plugin/              the Claude Code plugin (this directory)
  .claude-plugin/      plugin manifest (plugin.json)
  skills/              plugin skills (cluster-run, cluster-investigate, etc.)
  agents/              subagent definitions (proposer, synthesizer, auditor, investigator, critic)
  hooks/               post-subagent validation + summary hooks
benchmarking/        paper experiments — Python package for evaluating the plugin against baselines
  data_processing/     HuggingFace download + preprocessing (see its README for the dataset loaders)
  baselines/           prior clustering methods
  evaluation/          shared metrics
  experiments/         runner scripts (benchmark x method)
slurm/               FASRC/SLURM harness for the GPU-bound baseline phases (see its README)
data/                benchmark data (gitignored — downloaded from HuggingFace)
results/             figures, tables, logs, predictions (gitignored)
paper/               manuscript
```

### Paper experiments

```bash
uv sync
uv run python -m benchmarking.experiments.<name>
```

Data-processing entry points should call `ensure_data_dirs()` from `benchmarking.paths` so `data/raw/` and `data/derived/` exist on a fresh clone:

```python
from benchmarking.paths import ensure_data_dirs, DATA_RAW
ensure_data_dirs()
```

Two internal READMEs document harness details that don't belong here: [`benchmarking/data_processing/README.md`](../benchmarking/data_processing/README.md) (the 7 benchmark dataset loaders and their unified schema) and [`slurm/README.md`](../slurm/README.md) (running the ClusterLLM baseline's GPU phases on a SLURM cluster).

## Authors

- Emily Silcock <emilysilcock@fas.harvard.edu>
- Simon Löwe <loewe.sim@gmail.com>

## License

MIT — see [LICENSE](LICENSE).
