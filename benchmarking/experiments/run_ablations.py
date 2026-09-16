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
import sys
import traceback

from benchmarking.baselines.agentic_clustering import PAPER_CONFIG
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
    parser.add_argument(
        "--paper-config",
        action="store_true",
        help=(
            "Reproduce the configuration the paper's run of this ablation used, "
            "rather than the shipped plugin's current defaults: pins the initial "
            "proposer count and restores the cumulative agent-dispatch cap "
            "(no-task: 3 proposers / 8 dispatches, 2026-05-25; no-k: 3 / 20, "
            "2026-07-12). The plugin itself is untouched either way. Ignored for "
            "--synthonly, which runs no orchestrator."
        ),
    )
    parser.add_argument(
        "--initial-proposers",
        type=int,
        default=None,
        help=(
            "Pin the initial-round proposer count. Unset = whatever the shipped "
            "cluster-run skill says (currently 6-7). Overrides --paper-config."
        ),
    )
    parser.add_argument(
        "--max-agent-dispatches",
        type=int,
        default=None,
        help=(
            "Stop the loop after this many cumulative agent dispatches. Unset = "
            "no cap, i.e. the skill's own state-grounded stop criteria (the "
            "hard checkpoint was removed in issue #2). Overrides --paper-config."
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

    # Explicit flags win over --paper-config, which wins over "leave it to the
    # skill". synthonly has no orchestrator, so both stay None there.
    paper = PAPER_CONFIG[label] if args.paper_config else {}
    overrides = {
        "initial_proposers": (
            args.initial_proposers
            if args.initial_proposers is not None
            else paper.get("initial_proposers")
        ),
        "max_agent_dispatches": (
            args.max_agent_dispatches
            if args.max_agent_dispatches is not None
            else paper.get("max_agent_dispatches")
        ),
    }
    if args.synthonly and any(v is not None for v in overrides.values()):
        parser.error(
            "--initial-proposers / --max-agent-dispatches apply to --notask / "
            "--nok only; --synthonly re-classifies an archived taxonomy and "
            "dispatches no agents"
        )
    if any(v is not None for v in overrides.values()):
        print(
            f"[{label}] harness overrides: "
            f"initial_proposers={overrides['initial_proposers']} "
            f"max_agent_dispatches={overrides['max_agent_dispatches']}"
        )

    rows: list[dict] = []
    failures: list[tuple[str, str]] = []
    for name in datasets:
        print(f"\n========== ablation={label} / {name} ==========")
        # Sweep resilience: a single dataset's failure (e.g. an orchestrator that
        # returns without finalizing, or a transient API/subprocess error) must
        # NOT abort the remaining datasets. Log it, record it, and continue; the
        # failed dataset simply has no prediction and can be re-run later with
        # --only. Mirrors run_overlap_sweep's skip-and-continue behaviour.
        try:
            if args.synthonly:
                rows.append(run_synthonly(name, seed=args.seed, reuse_existing_classify=args.reuse_existing_classify))
            elif args.nok:
                rows.append(run_nok(name, seed=args.seed, resume_classify=args.resume_classify, **overrides))
            else:
                rows.append(run_notask(name, seed=args.seed, resume_classify=args.resume_classify, **overrides))
        except Exception as exc:  # noqa: BLE001 — intentional: isolate per-dataset failures
            print(
                f"\n[{label}/{name}] FAILED — skipping, sweep continues:\n"
                f"{traceback.format_exc()}",
                file=sys.stderr,
                flush=True,
            )
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            continue

    _print_rows(rows)
    if failures:
        print(
            f"\n{'='*60}\n{len(failures)}/{len(datasets)} dataset(s) FAILED and were skipped:",
            file=sys.stderr,
        )
        for name, err in failures:
            print(f"  - {name}: {err}", file=sys.stderr)
        print(
            f"Re-run just these with: --{label} --only {' '.join(n for n, _ in failures)}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
