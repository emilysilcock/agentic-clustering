# agentic-clustering

A Claude Code plugin for **iterative agentic discovery of natural clusters in text corpora**. Multiple specialised subagents — proposer, synthesizer, auditor, investigator, critic — collaborate under an orchestrator to converge on a stable, well-supported cluster taxonomy, which you can then apply to your whole corpus by classifying every text into it.

```text
/plugin marketplace add emilysilcock/econ-nlp-plugins
/plugin install agentic-clustering@econ-nlp-plugins
```

**User documentation lives in [`plugin/README.md`](plugin/README.md)** — prerequisites, installation, quick start, an end-to-end worked example, the full `/cluster-*` and `/classify-*` command list, and where outputs land. The rest of this page covers the repository, which is the experimental evaluation for the paper introducing the method and is not part of what plugin users install.

## Repository layout

```
plugin/              the Claude Code plugin — the only thing shipped to plugin users
  .claude-plugin/      plugin manifest (plugin.json)
  skills/              plugin skills (cluster-run, cluster-investigate, etc.)
  agents/              subagent definitions (proposer, synthesizer, auditor, investigator, critic)
  hooks/               post-subagent validation + summary hooks
  examples/            the bundled example corpus
benchmarking/        paper experiments — Python package for evaluating the plugin against baselines
  data_processing/     HuggingFace download + preprocessing (see its README for the dataset loaders)
  baselines/           prior clustering methods
  evaluation/          shared metrics
  experiments/         runner scripts (benchmark x method)
data/                benchmark data (gitignored — downloaded from HuggingFace)
results/             figures, tables, logs, predictions (gitignored)
```

## Paper experiments

```bash
uv sync
uv run python -m benchmarking.experiments.<name>
```

Data-processing entry points should call `ensure_data_dirs()` from `benchmarking.paths` so `data/raw/` and `data/derived/` exist on a fresh clone:

```python
from benchmarking.paths import ensure_data_dirs, DATA_RAW
ensure_data_dirs()
```

The ClusterLLM baseline is the only step that needs a GPU: its fine-tuning and clustering phases need CUDA (we used a single A100, roughly 15–25 minutes per dataset to fine-tune and 5–15 to cluster). `run_clusterllm --phase all` runs the whole pipeline; phases cache their output, and `--only <dataset>` lets datasets run in parallel on separate machines.

An internal README, [`benchmarking/data_processing/README.md`](benchmarking/data_processing/README.md), covers the seven benchmark dataset loaders and their unified schema.

## Authors

- Emily Silcock <emilysilcock@fas.harvard.edu>
- Simon Löwe <loewe.sim@gmail.com>

## License

MIT — see [LICENSE](LICENSE).
