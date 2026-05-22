# Vendored ClusterLLM source — edits from upstream

Source: `zhang-yu-wei/ClusterLLM` @ commit on `main` as of 2026-05-21
(repo's last push: 2023-10-18). No LICENSE file in upstream; redistributed
inside this research repo for reproducibility, not as a standalone artifact.

## Files vendored

Only the subset of the `perspective/` track that we actually drive from
`benchmarking/baselines/clusterllm/orchestrate.py`:

- `perspective/1_predict_triplet/triplet_sampling.py` — Phase 1
- `perspective/2_finetune/get_embedding.py` — Phase 0 / Phase 4
- `perspective/2_finetune/convert_triplet.py` — Phase 2.5 format conversion
- `perspective/2_finetune/finetune.py` — Phase 3
- `perspective/2_finetune/prompts.json` — Instructor instruction prefixes
- `perspective/2_finetune/clustering_utils/{__init__,evaluator}.py`
- `perspective/2_finetune/InstructorEmbedding/{__init__,instructor}.py`

**Not vendored** (explicitly omitted):

- `1_predict_triplet/predict_triplet.py`, `tools.py`, `prompts.json` —
  replaced by `../triplet_judge.py` + `../prompts.json` (Claude Code path).
- `1_predict_triplet/calculate_accuracy.py` — offline analysis we don't need.
- `1_predict_triplet/random_triplet_sampling.py` — random-sampling baseline,
  not part of the canonical ClusterLLM result.
- `2_finetune/convert_triplet_self.py` — no-LLM ablation we won't run.
- `2_finetune/finetune_e5.py`, `get_embedding_e5.py`, `e5_utils.py` — the E5
  encoder variant; we use Instructor-large per the paper's headline result.
- `granularity/` — separate experimental track ("how many clusters?") we
  don't use; we take `k` from `k_in_scope` per SPEC §5.5.
- `scripts/*.sh` — bash drivers; `orchestrate.py` drives subprocesses directly.

## Edits

### `convert_triplet.py`

Replace `assert not os.path.exists(output_path)` (line 58) with an unlink
+ rewrite so orchestrate.py can re-run idempotently. The original aborts on
any second call; we want to be able to re-derive train triplets from an
updated judgments file without manually deleting outputs.

### `clustering_utils/evaluator.py`

Extended the ``DEFINITIONS['hkunlp/instructor-large']`` dict with four
aliases so the Instructor prompts work with our canonical dataset names
(``clinc150``, ``massive_domain``, ``goemotions``, ``twenty_newsgroups``).
``clinc150`` / ``goemotions`` / ``twenty_newsgroups`` reuse the exact
instruction strings the authors already provide for ``clinc`` /
``go_emotion`` / ``TwentyNewsgroupsClustering`` respectively;
``massive_domain`` is a new entry parallel to the upstream
``massive_intent`` entry. Flag in paper appendix as a configuration detail.

## What was NOT touched

- All scientific logic (sampler, evaluator, trainer, model class) is
  byte-identical to upstream. Any difference between our reported numbers
  and the paper is attributable to the LLM swap (Claude Code Opus 4.7 in
  place of OpenAI gpt-3.5-turbo/gpt-4) and to our dataset adapter, not to
  changes here.
