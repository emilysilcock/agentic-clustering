"""CLI driver for the auditor-diversity-sampling sweep.

One arm (diverse-auditor) × seven datasets = 7 runs, sequential. Each run is
taxonomy-stage only — no classification, no predictions, no metered API spend
(the Opus orchestrator + subagents run on the Claude Code Max subscription). The
random-auditor baseline is the existing seed=0_discoverk run; it is NOT re-run
here — the comparison script pairs the two.

Examples:
    # Single cell — smoke-test on Banking77
    uv run --native-tls python -m benchmarking.experiments.run_audit_diversity_sweep \\
        --only banking77

    # Full 7-dataset sweep, smallest-up
    uv run --native-tls python -m benchmarking.experiments.run_audit_diversity_sweep \\
        --all
"""

from __future__ import annotations

import argparse
from typing import Iterable

from benchmarking.baselines.agentic_audit_diversity_sweep import (
    SWEEP_ORDER,
    run_audit_diversity_sweep,
)


def _print_rows(rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    header = (
        f"{'dataset':<22}{'n':>7}{'k_gold':>8}{'k_range':>10}{'k_act':>7}"
        f"{'time_s':>10}"
    )
    print()
    print(header)
    print("-" * len(header))
    for r in rows:
        krange = f"[{r['k_range'][0]},{r['k_range'][1]}]"
        print(
            f"{r['dataset']:<22}{r['n_docs']:>7}{r['k_in_scope']:>8}{krange:>10}"
            f"{r['k_actual']:>7}{r['wall_clock_s']:>10.1f}"
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
        help="Run all 7 datasets in the standard sweep order (smallest-up).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.all:
        datasets = SWEEP_ORDER
    elif args.only:
        datasets = list(args.only)
    else:
        # Default to Banking77 only — matches the pilot convention.
        datasets = ["banking77"]

    rows: list[dict] = []
    failures: list[tuple[str, str]] = []
    # Each cell is wrapped so one failure (e.g. a session-limit truncation that
    # leaves the orchestrator without final_taxonomy.json) is logged and skipped
    # rather than aborting the remaining cells. Failed cells are listed at the
    # end for targeted re-runs.
    for ds in datasets:
        print(f"\n========== auditdiv / {ds} ==========")
        try:
            row = run_audit_diversity_sweep(ds, seed=args.seed)
            rows.append(row)
        except Exception as exc:  # noqa: BLE001 — one bad cell must not kill the sweep
            import traceback

            traceback.print_exc()
            print(f"[auditdiv/{ds}] FAILED: {exc!r} — skipping to next cell")
            failures.append((ds, repr(exc)))

    _print_rows(rows)
    if failures:
        print(f"\n{len(failures)} cell(s) FAILED — re-run individually:")
        for ds, err in failures:
            print(f"  --only {ds}    # {err}")


if __name__ == "__main__":
    main()
