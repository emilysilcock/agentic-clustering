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

## Reproducing the paper

### Requirements

- [uv](https://docs.astral.sh/uv/) and Python 3.11 (`uv sync` installs everything else).
- An OpenAI API key (`OPENAI_API_KEY`) for the cheap-tier calls (gpt-5-mini via the Batch API) and the `text-embedding-3-large` baseline. Set it in the environment or in a `secrets.json` at the repo root (`{"OPENAI_API_KEY": "..."}`, gitignored).
- [Claude Code](https://claude.com/claude-code), logged in to a Claude subscription, for the frontier-tier calls (our method's agent loop, TopicGPT's generation and refinement, Huang & He's label merging). These run through `claude -p`.
- The [text-classification](https://github.com/emilysilcock/text-classification) plugin cloned next to this repo (`../text-classification/`); our method's final classification step loads it.
- A CUDA GPU for ClusterLLM's fine-tuning and clustering phases only (we used a single A100: roughly 15–25 minutes per dataset to fine-tune, 5–15 to cluster).

Per-method spend is reported in the paper's results table.

### Steps

```bash
uv sync
uv run python -m benchmarking.data_processing.process_all     # download + preprocess the 7 datasets
uv run python -m benchmarking.experiments.<runner>             # one per method, see below
uv run python -m benchmarking.data_processing.<builder>        # writes tables to results/tables/
```

| Paper table | Runners (`benchmarking.experiments.*`) | Builder (`benchmarking.data_processing.*`) |
|---|---|---|
| Datasets | — | `build_summary_table` |
| Main results | `run_lda`, `run_sbert_kmeans`, `run_bertopic`, `run_openai_embedding_kmeans`, `run_clusterllm --phase all`, `run_topicgpt`, `run_huang_he`, `run_agentic_clustering --all` (given k) and `--all --discover-k`, `run_ablations --nok --all` (no k) | `build_results_table` |
| Seed variance | `run_agentic_clustering --all --discover-k --seed {1,2}` | `build_seed_table` |
| Ablations | `run_ablations --synthonly --all`, `run_ablations --notask --all` | `build_ablation_table` |
| Design ablations | `run_proposer_sweep --variant all --all`, `run_sample_size_sweep --variant all --all` | `compare_proposer_sweep`, `compare_sample_size_sweep` (the table is assembled from their summaries in `results/`) |

Each runner's `--help` lists its options; `--only <dataset>` restricts a run to some datasets. Every step caches its outputs under `data/` and `results/`, so runs can be split up or resumed. `--paper-config` on `run_agentic_clustering` and `run_ablations` reproduces the exact configuration of the paper's runs.

[`benchmarking/data_processing/README.md`](benchmarking/data_processing/README.md) covers the seven dataset loaders and their unified schema.

## Authors

- Emily Silcock <emilysilcock@fas.harvard.edu>
- Simon Löwe <loewe.sim@gmail.com>

## License

MIT — see [LICENSE](LICENSE).
