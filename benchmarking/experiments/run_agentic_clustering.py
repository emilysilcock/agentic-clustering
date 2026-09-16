"""CLI runner for our method (the agentic-clustering plugin).

Defaults to a Banking77-only run — it's the smallest dataset and the
designated pilot per SPEC §8. Use ``--all`` for the full overnight sweep
(strictly sequential, smallest-up).

Examples:
    # Smoke-test the agent loop on Banking77 without burning Haiku quota
    uv run --native-tls python -m benchmarking.experiments.run_agentic_clustering --skip-classify

    # Full pilot on Banking77 (agent loop + classify)
    uv run --native-tls python -m benchmarking.experiments.run_agentic_clustering

    # Overnight sweep, all 7 datasets in size order
    uv run --native-tls python -m benchmarking.experiments.run_agentic_clustering --all
"""

from __future__ import annotations

import argparse
from typing import Iterable

from benchmarking.baselines.agentic_clustering import (
    METHOD,
    run_agentic_clustering,
)

# Sweep order is smallest-up, with Banking77 forced to the front per the
# pilot decision (smallest plus the SPEC §8 go/no-go dataset). MASSIVE-Intent
# and MASSIVE-Domain are technically marginally smaller but Banking77 leads.
SWEEP_ORDER = [
    "banking77",
    "massive_intent",
    "massive_domain",
    "stackexchange",
    "clinc150",
    "twenty_newsgroups",
    "goemotions",
]


def _print_rows(rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    skipped = any(r.get("skipped_classify") for r in rows)
    if skipped:
        header = f"{'method':<22}{'dataset':<22}{'n':>7}{'k':>5}{'orch_s':>10}"
        print()
        print(header)
        print("-" * len(header))
        for r in rows:
            orch_s = r.get("orchestrator_wall_clock_s", 0.0)
            print(
                f"{r['method']:<22}{r['dataset']:<22}{r['n_docs']:>7}"
                f"{r['k_in_scope']:>5}{orch_s:>10.1f}"
            )
        return

    header = (
        f"{'method':<22}{'dataset':<22}{'n':>7}{'k':>5}{'k_act':>7}"
        f"{'api_usd':>10}{'usd':>10}{'time_s':>10}"
    )
    print()
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['method']:<22}{r['dataset']:<22}{r['n_docs']:>7}"
            f"{r['k_in_scope']:>5}{r['k_actual']:>7}"
            f"{r['api_usd']:>10.4f}{r['usd']:>10.4f}{r['wall_clock_s']:>10.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--only",
        nargs="+",
        choices=SWEEP_ORDER,
        help="Run only the named datasets (in the order given).",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run all 7 datasets in the standard sweep order (smallest-up, Banking77 first).",
    )
    parser.add_argument("--seed", type=int, default=0)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--skip-classify",
        action="store_true",
        help="Stop after the agent loop produces final_taxonomy.json. No Haiku spend.",
    )
    mode_group.add_argument(
        "--resume-classify",
        action="store_true",
        help=(
            "Skip corpus-build / init / orchestrator and start from the existing "
            "results/clustering/<ds>/seed=<n>[_discoverk]/ workspace. Use to turn "
            "a prior --skip-classify run into a real predictions artifact."
        ),
    )
    parser.add_argument(
        "--discover-k",
        action="store_true",
        help=(
            "Run the discover-k variant: k_range = gold_k ± 20%%, orchestrator "
            "picks k within the range. Writes to a separate method "
            "(agentic_clustering_discoverk) and workspace "
            "(seed=<n>_discoverk) so the given-k artifacts are untouched."
        ),
    )
    args = parser.parse_args()

    if args.all:
        datasets = SWEEP_ORDER
    elif args.only:
        datasets = list(args.only)
    else:
        datasets = ["banking77"]  # default: pilot

    method_label = "agentic_clustering_discoverk" if args.discover_k else METHOD
    rows: list[dict] = []
    for name in datasets:
        print(f"\n========== {method_label} / {name} ==========")
        row = run_agentic_clustering(
            name,
            seed=args.seed,
            skip_classify=args.skip_classify,
            resume_classify=args.resume_classify,
            discover_k=args.discover_k,
        )
        rows.append(row)

    _print_rows(rows)


if __name__ == "__main__":
    main()
