"""CLI runner for the Huang & He baseline.

Drives all phases via ``benchmarking.baselines.huang_he``:

  Phase 0 ``adapt``    --- documents.jsonl -> data/huang_he/<ds>/input.jsonl
  Phase 1 ``generate`` --- gpt-5-mini (OpenAI Batch) -> labels_pre_merge.json
  Phase 2 ``merge``    --- Opus 4.7 (Claude Code) -> labels_merged.json
  Phase 3 ``classify`` --- gpt-5-mini (OpenAI Batch) -> classifications.jsonl
  Phase 4 ``write``    --- parse + score -> results/predictions/huang_he/<ds>/seed=0.{jsonl,meta.json}

Discover-$k$ only (SPEC §5.5) --- no ``--k`` flag; Huang & He decides $k$
itself via the merge step. The method appears only in the discover-$k$
panel of the results table, symmetric with TopicGPT.

Routing per SPEC §5.6.2 and the §5.6 >1,000-text rule:

* Phase 1: ~6k chunked-15 calls across the sweep -> cheap (gpt-5-mini Batch).
* Phase 2: 7 calls total (one per dataset) -> frontier (Opus 4.7 via subscription).
* Phase 3: ~94k per-doc calls across the sweep -> cheap (gpt-5-mini Batch
  with auto-cache on the merged-label-list prefix).
* Phase 4: local; no model calls.

Examples:

    # Full pipeline on one dataset
    uv run --native-tls python -m benchmarking.experiments.run_huang_he \
        --phase all --only banking77

    # Re-score (no model calls) after editing a metric
    uv run --native-tls python -m benchmarking.experiments.run_huang_he \
        --phase write --only banking77

    # Force a re-run of one phase
    uv run --native-tls python -m benchmarking.experiments.run_huang_he \
        --phase generate --only banking77 --overwrite
"""

from __future__ import annotations

import argparse

import benchmarking  # noqa: F401 — truststore.inject_into_ssl()

from benchmarking.baselines.huang_he import orchestrate, result_parser

METHOD = "huang_he"

DATASETS = [
    "banking77",
    "clinc150",
    "massive_intent",
    "massive_domain",
    "goemotions",
    "twenty_newsgroups",
    "stackexchange",
]

PHASE_CHOICES = (
    "adapt",
    "generate",
    "merge",
    "classify",
    "write",
    "all",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASE_CHOICES, default="all")
    parser.add_argument("--only", nargs="+", choices=DATASETS)
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Persistence seed label (predictions go to seed=<n>.jsonl).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-run cached phases.",
    )
    args = parser.parse_args()

    names = args.only or DATASETS

    do_adapt    = args.phase in ("adapt", "all")
    do_generate = args.phase in ("generate", "all")
    do_merge    = args.phase in ("merge", "all")
    do_classify = args.phase in ("classify", "all")
    do_write    = args.phase in ("write", "all")

    for name in names:
        print(f"\n========== huang_he / {name} ==========", flush=True)
        if do_adapt:
            res = orchestrate.adapt(name, force=args.overwrite)
            tail = (
                f"{res.n_truncated} truncated"
                if res.n_truncated >= 0
                else "cache hit"
            )
            print(
                f"[huang_he/{name}/phase=adapt] {res.n_docs} docs | {tail}",
                flush=True,
            )
        if do_generate:
            orchestrate.generate(name, overwrite=args.overwrite)
        if do_merge:
            orchestrate.merge(name, overwrite=args.overwrite)
        if do_classify:
            orchestrate.classify(name, overwrite=args.overwrite)
        if do_write:
            jsonl, meta = result_parser.write(name, seed=args.seed)
            print(f"[huang_he/{name}/phase=write] -> {jsonl} | {meta}", flush=True)


if __name__ == "__main__":
    main()
