"""CLI runner for the ClusterLLM baseline.

Drives all phases via ``benchmarking.baselines.clusterllm.orchestrate``.
``finetune`` and ``cluster`` need a CUDA GPU (we used a single A100: roughly
15-25 min per dataset to fine-tune, 5-15 min to cluster). ``embed`` runs on
CPU but is slow on the larger corpora; a GPU helps there too. Every phase
caches its output, so a run can be split or resumed, and ``--only`` lets
datasets run in parallel on separate machines.

Examples:
    # Everything up to fine-tuning (embed, sample, judge) for one dataset
    uv run python -m benchmarking.experiments.run_clusterllm \\
        --phase pre-finetune --only banking77

    # The full pipeline for one dataset, on a GPU machine
    uv run python -m benchmarking.experiments.run_clusterllm \\
        --phase all --only banking77
"""

from __future__ import annotations

import argparse

from benchmarking.baselines.clusterllm.orchestrate import (
    cluster,
    convert_triplets,
    embed_base,
    finetune,
    judge,
    sample_triplets,
)
from benchmarking.llm_clients.claude_code import DEFAULT_MODEL

METHOD = "clusterllm"

DATASETS = [
    "banking77",
    "clinc150",
    "massive_intent",
    "massive_domain",
    "goemotions",
    "twenty_newsgroups",
    "stackexchange",
]

# ``pre-finetune`` runs embed, sample and judge; ``all`` chains through
# convert, finetune and cluster as well.
PHASE_CHOICES = (
    "embed",
    "sample",
    "judge",
    "pre-finetune",
    "convert",
    "finetune",
    "cluster",
    "all",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASE_CHOICES, default="pre-finetune")
    parser.add_argument("--only", nargs="+", choices=DATASETS)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--judge-backend",
        choices=("openai_batch", "claude"),
        default="openai_batch",
        help="Phase-2 judging backend.",
    )
    parser.add_argument("--model", default=None,
                        help="Model id; defaults: gpt-5-mini for openai_batch, "
                             f"{DEFAULT_MODEL} for claude.")
    parser.add_argument("--sample-seed", type=int, default=100,
                        help="Seed for triplet sampling (paper default: 100).")
    parser.add_argument("--max-query", type=int, default=1024,
                        help="Triplets per dataset (paper default: 1024).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run cached phases (embed/sample only).")
    args = parser.parse_args()

    names = args.only or DATASETS

    do_embed = args.phase in ("embed", "pre-finetune", "all")
    do_sample = args.phase in ("sample", "pre-finetune", "all")
    do_judge = args.phase in ("judge", "pre-finetune", "all")
    do_convert = args.phase in ("convert", "finetune", "cluster", "all")
    do_finetune = args.phase in ("finetune", "cluster", "all")
    do_cluster = args.phase in ("cluster", "all")

    for name in names:
        print(f"\n========== clusterllm / {name} ==========", flush=True)
        if do_embed:
            embed_base(name, overwrite=args.overwrite)
        if do_sample:
            sample_triplets(
                name,
                seed=args.sample_seed,
                max_query=args.max_query,
                overwrite=args.overwrite,
            )
        if do_judge:
            judge(
                name,
                concurrency=args.concurrency,
                model=args.model,
                backend=args.judge_backend,
            )
        if do_convert:
            convert_triplets(name, overwrite=args.overwrite)
        if do_finetune:
            finetune(name, overwrite=args.overwrite)
        if do_cluster:
            cluster(name, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
