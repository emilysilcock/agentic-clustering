"""SLURM array dispatcher for ClusterLLM on FASRC.

Maps ``SLURM_ARRAY_TASK_ID`` to one of the 7 datasets and runs the requested
phase. The sbatch script picks the phase via ``--phase``; the dispatcher
calls the matching orchestrate function:

    prep      → embed_base + sample_triplets        (phase 0+1, GPU)
    finetune  → finetune                            (phase 3, GPU, heavy)
    cluster   → cluster                             (phase 4, GPU encode + CPU k-means)

Phase 2 (LLM triplet judging) runs locally — OpenAI Batch API has no
FASRC dependency. Phase 2.5 (convert_triplets) is a tiny CPU step that also
runs locally; its output ``train_triplets.json`` is scp'd over before phase 3.

Local test (no SLURM):
    python -m slurm.array_task --phase prep --task-id 0
"""

from __future__ import annotations

import argparse
import os
import sys

import benchmarking  # noqa: F401 — Windows TLS injection; no-op on Linux.

DATASETS = [
    "banking77",
    "clinc150",
    "massive_intent",
    "massive_domain",
    "twenty_newsgroups",
    "goemotions",
    "stackexchange",
]

PHASES = ("prep", "finetune", "cluster")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase",
        choices=PHASES,
        default=os.environ.get("CLUSTERLLM_PHASE", "prep"),
        help="Which phase to run. May also be set via CLUSTERLLM_PHASE env var.",
    )
    p.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Override SLURM_ARRAY_TASK_ID (for local testing).",
    )
    args = p.parse_args()

    task_id = args.task_id
    if task_id is None:
        env_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_id is None:
            sys.exit("ERROR: SLURM_ARRAY_TASK_ID not set and --task-id not given")
        task_id = int(env_id)
    if not (0 <= task_id < len(DATASETS)):
        sys.exit(f"ERROR: task_id {task_id} out of range 0..{len(DATASETS) - 1}")

    dataset = DATASETS[task_id]
    print(f"[array_task] phase={args.phase} task_id={task_id} -> dataset={dataset}", flush=True)

    # Lazy imports keep failures fast and avoid pulling torch into the help text.
    from benchmarking.baselines.clusterllm.orchestrate import (
        cluster,
        embed_base,
        finetune,
        sample_triplets,
    )

    if args.phase == "prep":
        embed_base(dataset)
        sample_triplets(dataset)
    elif args.phase == "finetune":
        finetune(dataset)
    elif args.phase == "cluster":
        cluster(dataset)
    else:
        sys.exit(f"unknown phase: {args.phase}")


if __name__ == "__main__":
    main()
