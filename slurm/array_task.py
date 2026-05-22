"""SLURM array dispatcher for ClusterLLM phase 0+1 on FASRC.

Maps ``SLURM_ARRAY_TASK_ID`` to one of the 7 datasets and runs Instructor-
large embedding (phase 0) + entropy-rank triplet sampling (phase 1).
Phase 2 (Claude judging) runs locally on the laptop with Claude Code, not
here, so the array stops after phase 1.

Outputs per dataset land under
``data/clusterllm/<dataset>/{base_embeds.hdf5,triplets.json}``. After the
array completes, rsync (or scp+tar) those back to the local machine.

Local test (no SLURM):
    python -m slurm.array_task --task-id 0
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
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
    print(f"[array_task] task_id={task_id} -> dataset={dataset}", flush=True)

    # Imports here (after sanity checks) keep failures fast and avoid pulling
    # in torch/transformers when we just want to print the resolved dataset.
    from benchmarking.baselines.clusterllm.orchestrate import (
        embed_base,
        sample_triplets,
    )

    embed_base(dataset)
    sample_triplets(dataset)


if __name__ == "__main__":
    main()
