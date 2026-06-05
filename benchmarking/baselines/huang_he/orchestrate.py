"""Phase orchestrator for the Huang & He baseline.

Thin dispatch layer matching ``topicgpt/orchestrate.py`` shape. Each
phase function is a one-liner over the module that actually implements
it, kept here so callers can import a single ``orchestrate.*`` surface.

| Phase | Output                                       | Runner                         |
|-------|----------------------------------------------|--------------------------------|
| 0     | data/huang_he/<ds>/input.jsonl               | ``dataset_adapter.adapt``      |
| 1     | data/huang_he/<ds>/labels_pre_merge.json     | ``batch_generate.generate``    |
|       |                                              |   (OpenAI Batch, gpt-5-mini)   |
| 2     | data/huang_he/<ds>/labels_merged.json        | ``merge.merge``                |
|       |                                              |   (Claude Code, Opus 4.7)      |
| 3     | data/huang_he/<ds>/classifications.jsonl     | ``batch_classify.classify``    |
|       |                                              |   (OpenAI Batch, gpt-5-mini)   |
| 4     | results/predictions/huang_he/<ds>/seed=<n>.* | ``result_parser.write``        |

Discover-$k$ only (SPEC §5.5) --- the method has no ``k`` input.
Huang & He appears only in the discover-$k$ panel of the results table.
"""

from __future__ import annotations

from benchmarking.baselines.huang_he import batch_classify, batch_generate, merge as merge_mod
from benchmarking.baselines.huang_he.dataset_adapter import adapt


def generate(dataset_name: str, *, overwrite: bool = False):
    return batch_generate.generate(dataset_name, overwrite=overwrite)


def merge(dataset_name: str, *, overwrite: bool = False):
    return merge_mod.merge(dataset_name, overwrite=overwrite)


def classify(dataset_name: str, *, overwrite: bool = False):
    return batch_classify.classify(dataset_name, overwrite=overwrite)


__all__ = ["adapt", "generate", "merge", "classify"]
