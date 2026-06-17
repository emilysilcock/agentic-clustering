"""CLI driver for the proposer-sample-overlap sweep.

Four variants × seven datasets = 28 runs, sequential. Each run is taxonomy-stage
only — no classification, no predictions, no metered API spend (the Opus
orchestrator + subagents run on the Claude Code Max subscription).

Variants: v0 = disjoint (production default, baseline rerun), v1 = 25% shared,
v2 = 50% shared, v3 = 100% shared (every proposer the same sample).

Examples:
    # Single cell: v2 (50% overlap) on Banking77 — the re-validation cell
    uv run --native-tls python -m benchmarking.experiments.run_overlap_sweep \\
        --variant v2 --only banking77

    # One variant across all 7 datasets
    uv run --native-tls python -m benchmarking.experiments.run_overlap_sweep \\
        --variant v1 --all

    # Full 28-run sweep (all variants, all datasets), smallest-up
    uv run --native-tls python -m benchmarking.experiments.run_overlap_sweep \\
        --variant all --all
"""

from __future__ import annotations

import argparse
from typing import Iterable

from benchmarking.baselines.agentic_overlap_sweep import (
    SWEEP_ORDER,
    VARIANTS,
    run_overlap_sweep,
)


def _print_rows(rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    header = (
        f"{'variant':<8}{'overlap':>8}{'dataset':<22}{'n':>7}{'k_gold':>8}"
        f"{'k_range':>10}{'k_act':>7}{'time_s':>10}"
    )
    print()
    print(header)
    print("-" * len(header))
    for r in rows:
        krange = f"[{r['k_range'][0]},{r['k_range'][1]}]"
        print(
            f"{r['variant']:<8}{r['overlap_fraction']:>8.0%}{r['dataset']:<22}"
            f"{r['n_docs']:>7}{r['k_in_scope']:>8}{krange:>10}{r['k_actual']:>7}"
            f"{r['wall_clock_s']:>10.1f}"
        )


def _variant_run_order(variant_arg: str) -> list[str]:
    """Order variants disjoint-first (v0 → v3).

    v0/v1 carry no risk of the shared core shrinking effective coverage, so
    running them first banks the cheapest, least-surprising cells before the
    higher-overlap variants where behaviour is less certain.
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
            "Which overlap variant to run. v0=disjoint (baseline rerun), "
            "v1=25%, v2=50%, v3=100%. 'all' runs all four in order v0→v3."
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
        # Default to Banking77 only — matches the pilot convention and lets
        # `--variant vN` smoke-test a single cell.
        datasets = ["banking77"]

    variants = _variant_run_order(args.variant)

    rows: list[dict] = []
    failures: list[tuple[str, str, str]] = []
    # Outer loop = variant, inner loop = dataset. This way v0 finishes across
    # all datasets before v1 begins, so a mid-sweep failure leaves a complete
    # variant rather than seven half-finished ones.
    #
    # Each cell is wrapped so one failure (e.g. a session-limit truncation that
    # leaves the orchestrator without final_taxonomy.json) is logged and skipped
    # rather than aborting the remaining cells. Failed cells are listed at the
    # end for targeted re-runs.
    for vname in variants:
        for ds in datasets:
            v = VARIANTS[vname]
            print(
                f"\n========== overlap-{v.name} ({v.fraction:.0%}) / {ds} =========="
            )
            try:
                row = run_overlap_sweep(ds, variant_name=vname, seed=args.seed)
                rows.append(row)
            except Exception as exc:  # noqa: BLE001 — one bad cell must not kill the sweep
                import traceback

                traceback.print_exc()
                print(
                    f"[overlap-{v.name}/{ds}] FAILED: {exc!r} — skipping to next cell"
                )
                failures.append((vname, ds, repr(exc)))

    _print_rows(rows)
    if failures:
        print(f"\n{len(failures)} cell(s) FAILED — re-run individually:")
        for vname, ds, err in failures:
            print(f"  --variant {vname} --only {ds}    # {err}")


if __name__ == "__main__":
    main()
