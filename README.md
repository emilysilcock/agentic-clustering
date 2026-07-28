# agentic-clustering

A Claude Code plugin for **iterative agentic discovery of natural clusters in text corpora**. Multiple specialised subagents — proposer, synthesizer, auditor, investigator, critic — collaborate under an orchestrator to converge on a stable, well-supported cluster taxonomy.

## Installation

The plugin lives in [`plugin/`](plugin/) and is published through the
[`econ-nlp-plugins`](https://github.com/emilysilcock/econ-nlp-plugins) marketplace:

```text
/plugin marketplace add emilysilcock/econ-nlp-plugins
/plugin install agentic-clustering@econ-nlp-plugins
```

For local development from a clone of this repo:

```text
claude --plugin-dir /path/to/agentic-clustering/plugin
```

## Usage

The plugin works in two phases — **discover** a cluster taxonomy, then **classify** your corpus into it.

Point it at a CSV or JSON file and run:

```text
/cluster-run
```

It asks a handful of setup questions — where to put the workspace, the corpus path, the
text column, **how many clusters you want**, **what to cluster on**, and a model tier —
then runs an iterative loop of subagents until the taxonomy stabilises. When you're happy,
`/cluster-finalize` exports `taxonomy.md`, `final_taxonomy.json`, and `categories.json`,
and `/classify-run` applies that taxonomy to every text in the corpus.

**Once it starts, let it run.** `/cluster-run` is designed to work unattended: the
orchestrator decides on its own when to sample more texts, when to propose new clusters,
when to merge or split them, and when to send an auditor, investigator, or critic after a
weak spot. It converges without you. Answer the setup questions and step away — you don't
need to babysit the loop or nudge it between iterations, and doing so is usually
counterproductive.

The other commands are there if you want them, not because a normal run needs them:
`/cluster-status` shows live progress (cluster count, coverage, confidence, cross-proposal
agreement), and `/cluster-investigate` lets you point the loop at a specific cluster or
question if you disagree with a call it made. Most runs go straight from `/cluster-run` to
`/cluster-finalize`.

### The two answers that matter

Two of the setup questions do most of the work. The run *will* ask you both — the point is
that **you decide them**, and you are not picking from a menu. Anything you can express in
a sentence is a valid answer.

**1. The cluster-count range.** You give a min and a max, not a fixed `k`. The range is a
budget for how coarse or fine the taxonomy should be, and the loop searches within it:

```text
2 6        a handful of broad themes
10 20      a working taxonomy for a coding scheme
30 60      fine-grained, closer to a labelled category list
```

Pick the range from what you'll *do* with the clusters — a report needs fewer than a
routing system. If you genuinely don't know, give a wide range and narrow it on a second
run once you've seen the first taxonomy.

**2. The clustering instructions.** This is a free-text lens telling every agent what to
pay attention to. It's optional, but it's the single highest-leverage input: the same
corpus clusters completely differently depending on what you ask for.

```text
cluster by the type of problem the respondent describes
group by sentiment, not topic
focus on actionable categories a support team could route tickets to
distinguish by policy area; ignore which politician is mentioned
split on the mechanism of harm described, not the industry
```

The instructions propagate into every proposer, auditor, investigator, and critic dispatch
and act as the primary constraint on cluster formation — if you say "cluster by issue
type", agents won't cluster by sentiment. Leave it blank and the agents discover whatever
structure is most salient in the data, which is a fine starting point but rarely the one
you actually wanted. Writing one sentence here is usually worth more than a longer run.

Both answers are recorded in the workspace's `state.json`, so a follow-up `/cluster-run`
in the same workspace resumes with them. To try a different lens or a different granularity,
start a fresh run — they're cheap to compare.

See **[`plugin/README.md`](plugin/README.md)** for the full quick-start: prerequisites, an
end-to-end worked example, the complete `/cluster-*` and `/classify-*` command list, and
where outputs land.

## Repository layout

```
plugin/              the Claude Code plugin (the only thing shipped to plugin users)
  .claude-plugin/      plugin manifest (plugin.json)
  skills/              plugin skills (cluster-run, cluster-investigate, etc.)
  agents/              subagent definitions (proposer, synthesizer, auditor, investigator, critic)
  hooks/               post-subagent validation + summary hooks
benchmarking/        paper experiments — Python package for evaluating the plugin against baselines
  data_processing/     HuggingFace download + preprocessing
  baselines/           prior clustering methods
  evaluation/          shared metrics
  experiments/         runner scripts (benchmark x method)
data/                benchmark data (gitignored — downloaded from HuggingFace)
results/             figures, tables, logs, predictions (gitignored)
paper/               manuscript
```

## Paper experiments

The `benchmarking/` package and the `paper/` directory hold the experimental evaluation that accompanies the paper introducing this method — they are secondary to the plugin.

```bash
uv sync
uv run python -m benchmarking.experiments.<name>
```

Data-processing entry points should call `ensure_data_dirs()` from `benchmarking.paths` so `data/raw/` and `data/derived/` exist on a fresh clone:

```python
from benchmarking.paths import ensure_data_dirs, DATA_RAW
ensure_data_dirs()
```

## Authors

- Emily Silcock <emilysilcock@gmail.com>
- Simon Löwe <loewe.sim@gmail.com>

## License

MIT — see [LICENSE](LICENSE).
