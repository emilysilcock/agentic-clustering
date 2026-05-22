# agentic-clustering

A Claude Code plugin for **iterative agentic discovery of natural clusters in text corpora**. Multiple specialised subagents — proposer, synthesizer, auditor, investigator, critic — collaborate under an orchestrator to converge on a stable, well-supported cluster taxonomy.

## Installation

Once published, install via the Claude Code marketplace:

```text
/plugin marketplace add <github-org>/agentic-clustering
/plugin install agentic-clustering@agentic-clustering-marketplace
```

For local development (cloned repo):

```text
/plugin marketplace add /path/to/agentic-clustering
/plugin install agentic-clustering@agentic-clustering-marketplace
```

## Usage

Inside any Claude Code session, after installing:

- `/cluster-run` — run the discovery workflow on a corpus (asks for path, text column, k range, etc.)
- `/cluster-status` — show progress on the current workspace
- `/cluster-investigate` — run an investigation on a specific question or cluster
- `/cluster-finalize` — export a final taxonomy

The workflow writes its state to `.claude/clustering/` in the project being analysed.

## Repository layout

```
.claude-plugin/      plugin manifest + marketplace listing
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
