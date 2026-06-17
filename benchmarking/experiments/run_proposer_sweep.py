"""CLI driver for the proposer-count sweep ablation.

Four variants × seven datasets = 28 runs, sequential. Each run is taxonomy-stage
only — no classification, no predictions, no metered API spend (the Opus 4.7
orchestrator + subagents run on the Claude Code Max subscription).

Examples:
    # Single cell: v1 (4-5 proposers) on Banking77
    uv run --native-tls python -m benchmarking.experiments.run_proposer_sweep \\
        --variant v1 --only banking77

    # One variant across all 7 datasets
    uv run --native-tls python -m benchmarking.experiments.run_proposer_sweep \\
        --variant v2 --all

    # Full 28-run sweep (all variants, all datasets), smallest-up
    uv run --native-tls python -m benchmarking.experiments.run_proposer_sweep \\
        --variant all --all
"""

from __future__ import annotations

import argparse
from typing import Iterable

from benchmarking.baselines.agentic_proposer_sweep import (
    SWEEP_ORDER,
    VARIANTS,
    run_proposer_sweep,
)


def _print_rows(rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    header = (
        f"{'variant':<8}{'dataset':<22}{'n':>7}{'k_gold':>8}"
        f"{'k_range':>10}{'k_act':>7}{'time_s':>10}"
    )
    print()
    print(header)
    print("-" * len(header))
    for r in rows:
        krange = f"[{r['k_range'][0]},{r['k_range'][1]}]"
        print(
            f"{r['variant']:<8}{r['dataset']:<22}{r['n_docs']:>7}"
            f"{r['k_in_scope']:>8}{krange:>10}{r['k_actual']:>7}"
            f"{r['wall_clock_s']:>10.1f}"
        )


def _variant_run_order(variant_arg: str) -> list[str]:
    """Order variants smallest-cost-first (v0 = 2-3 proposers, v3 = 8-9).

    Cheaper variants first protects the rest of the sweep from a subscription-
    cap surprise: if we discover v3 burns the daily quota faster than expected,
    we still have v0/v1/v2 results banked.
    """
    if variant_arg == "all":
        return ["v0", "v1", "v2", "v3"]
    return [variant_arg]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--variant",
        choices=[*VARIANTS.keys(), "all"],
        required=True,
        help=(
            "Which proposer-count variant to run. v0=2-3 (baseline rerun), "
            "v1=4-5, v2=6-7, v3=8-9. 'all' runs all four in order v0→v3."
        ),
    )
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
        help="Run all 7 datasets in the standard sweep order (smallest-up).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.all:
        datasets = SWEEP_ORDER
    elif args.only:
        datasets = list(args.only)
    else:
        # Default to Banking77 only — matches run_agentic_clustering's pilot
        # convention and lets `--variant vN` smoke-test a single cell.
        datasets = ["banking77"]

    variants = _variant_run_order(args.variant)

    rows: list[dict] = []
    # Outer loop = variant, inner loop = dataset. This way v0 finishes across
    # all datasets before v1 begins, so a mid-sweep failure leaves a complete
    # variant rather than seven half-finished ones.
    for vname in variants:
        for ds in datasets:
            v = VARIANTS[vname]
            print(
                f"\n========== proposers-{v.name} ({v.count_phrase}) / {ds} =========="
            )
            row = run_proposer_sweep(ds, variant_name=vname, seed=args.seed)
            rows.append(row)

    _print_rows(rows)


if __name__ == "__main__":
    main()
