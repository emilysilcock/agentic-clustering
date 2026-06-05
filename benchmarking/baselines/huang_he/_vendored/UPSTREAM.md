# Vendored upstream — Huang & He (2024)

| | |
|---|---|
| Paper | Chen Huang and Guoxiu He. *Text Clustering as Classification with LLMs.* arXiv:2410.00927, 2024. Later: SIGIR-AP 2025. |
| Source repo | https://github.com/ECNU-Text-Computing/Text-Clustering-via-LLM |
| Pinned commit | `6d02bb0d0d8b08902aeeb3761be966f203d2af7f` (2025-10-03) |
| License | **None published** — the upstream repo contains no `LICENSE` file and the GitHub API reports `"license": null`. Used here as a research-baseline reproduction of the method described in arXiv:2410.00927; the SIGIR-AP 2025 venue policy on baseline reproduction applies. We have not modified any vendored source. |

## Files copied verbatim

| File | Bytes | Used? |
|---|---|---|
| `label_generation.py` | 7831 | yes — we import `prompt_construct_generate_label`, `prompt_construct_merge_label` |
| `given_label_classification.py` | 6245 | yes — we import `prompt_construct` |
| `select_part_labels.py` | 1731 | no — 0%-seed config means seed selection is bypassed |
| `evaluate.py` | 3734 | no — superseded by `benchmarking.evaluation.metrics` (Hungarian, NMI, ARI), which evaluates the full corpus rather than upstream's silent-drop-unsuccessful behaviour |
| `run.sh` | 866 | no — superseded by `benchmarking.experiments.run_huang_he` |
| `README.md` | 1545 | no |

## What we use, what we don't

We re-export the three prompt builders (`prompt_construct_generate_label`,
`prompt_construct_merge_label`, `prompt_construct`) verbatim — including the
upstream typo `"classicifation"` — so the prompts seen by the LLM are
byte-identical to the paper. Everything else (LLM dispatch, JSON parsing,
checkpointing, evaluation, output schema) is replaced by our harness wrappers
because:

- Upstream parses LLM JSON with `eval(response)` (security + correctness hazard).
- Upstream's `given_label_classification.main()` has a signature mismatch
  (`ini_client(args.api_key)` vs `def ini_client()`).
- Upstream uses no random seed and `random.choices(..., k=...)` for the seed
  list (sampling with replacement, not a 20% partition).
- Upstream's `evaluate.py` silently drops `"Unsuccessful"` rows from the
  denominator, inflating metrics.
- Upstream uses sync `chat.completions` with no batching; we route to the
  OpenAI Batch API (50% discount) per SPEC §5.6 and use Opus 4.7 via Claude
  Code Max subscription for the single per-dataset merge call.

See `../CHANGES.md` for the full list of harness-side substitutions.
