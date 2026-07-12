"""CLI runner for the three ablations on our method.

All three are isolated in benchmarking.baselines.agentic_ablations and write to
new method names / workspace dirs — the completed main runs are never touched.
Ablations 1-2 run from the discover-k config; ablation 3 (no-k) removes the k
anchor entirely.

Examples:
    # Ablation 1 (synth-only) — cheap, classification-only, runs on OpenAI.
    uv run --native-tls python -m benchmarking.experiments.run_ablations --synthonly --all

    # Ablation 2 (no-task) — full frontier runs, Opus subscription.
    uv run --native-tls python -m benchmarking.experiments.run_ablations --notask --all

    # Ablation 3 (no-k) — full frontier runs, keeps the lens, drops k. Opus subscription.
    uv run --native-tls python -m benchmarking.experiments.run_ablations --nok --only massive_domain

    # Single dataset
    uv run --native-tls python -m benchmarking.experiments.run_ablations --synthonly --only banking77

The synth-only sweep hits OpenAI while no-task and no-k drive the Claude Code
subscription, so synth-only can run concurrently with either — but no-task and
no-k must not run at the same time (they'd contend for the same subscription).
"""

from __future__ import annotations

import argparse

from benchmarking.baselines.agentic_ablations import (
    SWEEP_ORDER,
    run_nok,
    run_notask,
    run_synthonly,
)


def _print_rows(rows: list[dict]) -> None:
    rows = [r for r in rows if r]
    if not rows:
        return
    header = (
        f"{'method':<38}{'dataset':<18}{'n':>7}{'k_act':>7}"
        f"{'ARI':>8}{'NMI':>8}{'ACC':>8}{'api_usd':>10}{'time_s':>9}"
    )
    print()
    print(header)
    print("-" * len(header))
    for r in rows:
        if r.get("skipped"):
            print(f"{r['method']:<38}{r['dataset']:<18}{'SKIPPED — ' + r.get('reason', ''):>0}")
            continue
        print(
            f"{r['method']:<38}{r['dataset']:<18}{r['n_docs']:>7}{r['k_actual']:>7}"
            f"{r.get('ari', 0):>8.3f}{r.get('nmi', 0):>8.3f}{r.get('acc', 0):>8.3f}"
            f"{r.get('api_usd', 0):>10.4f}{r.get('wall_clock_s', 0):>9.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    ab = parser.add_mutually_exclusive_group(required=True)
    ab.add_argument("--synthonly", action="store_true", help="Ablation 1: re-classify against first synth taxonomy.")
    ab.add_argument("--notask", action="store_true", help="Ablation 2: full discover-k run with blank instructions.")
    ab.add_argument("--nok", action="store_true", help="Ablation 3: full run with lens kept but k information removed.")

    sel = parser.add_mutually_exclusive_group()
    sel.add_argument("--all", action="store_true", help="Run all 7 datasets in sweep order (smallest-up).")
    sel.add_argument("--only", nargs="+", choices=SWEEP_ORDER, help="Run only the named datasets.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume-classify",
        action="store_true",
        help=(
            "no-task / no-k only: skip init + orchestrator and re-run just the "
            "classify pass from the existing seed=<n>_discoverk_notask / "
            "seed=<n>_nok workspace. Use after a classify-step failure to avoid "
            "re-running the agent loop."
        ),
    )
    parser.add_argument(
        "--reuse-existing-classify",
        action="store_true",
        help=(
            "synth-only only: skip the classify call and assemble the artifact from "
            "the seed_0.csv already in the workspace (e.g. after repairing a "
            "partially-failed classify with scripts/retry_classify_errors.py)."
        ),
    )
    args = parser.parse_args()
    if args.resume_classify and args.synthonly:
        parser.error("--resume-classify applies to --notask / --nok only")
    if args.reuse_existing_classify and not args.synthonly:
        parser.error("--reuse-existing-classify applies to --synthonly only")

    if args.all:
        datasets = SWEEP_ORDER
    elif args.only:
        datasets = list(args.only)
    else:
        datasets = ["banking77"]  # default: smoke test on the smallest

    label = "synthonly" if args.synthonly else ("nok" if args.nok else "notask")

    rows: list[dict] = []
    for name in datasets:
        print(f"\n========== ablation={label} / {name} ==========")
        if args.synthonly:
            rows.append(run_synthonly(name, seed=args.seed, reuse_existing_classify=args.reuse_existing_classify))
        elif args.nok:
            rows.append(run_nok(name, seed=args.seed, resume_classify=args.resume_classify))
        else:
            rows.append(run_notask(name, seed=args.seed, resume_classify=args.resume_classify))

    _print_rows(rows)


if __name__ == "__main__":
    main()
