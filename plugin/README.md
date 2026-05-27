# agentic-clustering

A Claude Code plugin for **iterative agentic discovery of natural clusters in text corpora**. Multiple specialised subagents — proposer, synthesizer, auditor, investigator, critic — collaborate under an orchestrator to converge on a stable, well-supported cluster taxonomy.

## Installation

```text
/plugin marketplace add emilysilcock/econ-nlp-plugins
/plugin install agentic-clustering@econ-nlp-plugins
```

For local development from a clone of this repo:

```text
claude --plugin-dir /path/to/agentic-clustering/plugin
```

## Usage

Inside any Claude Code session, after installing:

- `/cluster-run` — run the discovery workflow on a corpus (asks for path, text column, k range, etc.)
- `/cluster-status` — show progress on the current workspace
- `/cluster-investigate` — run an investigation on a specific question or cluster
- `/cluster-finalize` — export a final taxonomy

The workflow writes its state to `.claude/clustering/` in the project being analysed.

## Layout

```
.claude-plugin/plugin.json   plugin manifest
skills/                      plugin skills (cluster-run, cluster-investigate, …)
agents/                      subagent definitions (proposer, synthesizer, auditor, investigator, critic)
hooks/hooks.json             post-subagent validation + summary hooks
```

Skill scripts declare their own dependencies via [PEP 723](https://peps.python.org/pep-0723/) inline metadata and run under `uv run`, so the plugin is self-contained.

## Authors

- Emily Silcock <emilysilcock@fas.harvard.edu>
- Simon Löwe <loewe.sim@gmail.com>

## License

MIT — see [LICENSE](LICENSE).
