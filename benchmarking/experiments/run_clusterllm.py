"""CLI runner for the ClusterLLM baseline.

Drives all phases via ``benchmarking.baselines.clusterllm.orchestrate``.
Tonight's slice is ``--phase {embed,sample,judge}`` (no GPU needed); phases
``finetune`` and ``cluster`` run tomorrow on FASRC GPU.

Examples:
    # Full tonight slice for one dataset
    uv run --native-tls python -m benchmarking.experiments.run_clusterllm \\
        --phase all-tonight --only banking77

    # Just the overnight LLM-judging step across all 7 datasets
    uv run --native-tls python -m benchmarking.experiments.run_clusterllm \\
        --phase judge --concurrency 4
"""

from __future__ import annotations

import argparse

from benchmarking.baselines.clusterllm.orchestrate import (
    embed_base,
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

# Tomorrow's GPU phases. ``all-tonight`` is the CPU+LLM subset we run from
# this laptop; ``all`` would chain through finetune+cluster but those aren't
# implemented yet (TODO once FASRC scaffolding lands).
PHASE_CHOICES = (
    "embed",
    "sample",
    "judge",
    "all-tonight",
    "finetune",
    "cluster",
    "all",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASE_CHOICES, default="all-tonight")
    parser.add_argument("--only", nargs="+", choices=DATASETS)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sample-seed", type=int, default=100,
                        help="Seed for triplet sampling (paper default: 100).")
    parser.add_argument("--max-query", type=int, default=1024,
                        help="Triplets per dataset (paper default: 1024).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run cached phases (embed/sample only).")
    args = parser.parse_args()

    names = args.only or DATASETS

    do_embed = args.phase in ("embed", "all-tonight", "all")
    do_sample = args.phase in ("sample", "all-tonight", "all")
    do_judge = args.phase in ("judge", "all-tonight", "all")
    do_finetune = args.phase in ("finetune", "all")
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
            judge(name, concurrency=args.concurrency, model=args.model)
        if do_finetune or do_cluster:
            print(f"[{METHOD}/{name}] phase 3/4 not implemented yet; "
                  f"see TODO in orchestrate.py.", flush=True)


if __name__ == "__main__":
    main()
