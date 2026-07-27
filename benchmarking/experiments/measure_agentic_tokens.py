"""Measure the big-model (Opus) token usage of our agent loop, in isolation.

Why this exists: the main-results workspaces at
``results/clustering/<ds>/seed=0[_discoverk]/`` and their prediction meta.json
files were produced *before* the orchestrator captured token usage (the Claude
Code subscription meters none, and the runs used ``--output-format`` text). The
Opus token counts cannot be recovered after the fact. This module re-runs *only
the agent loop* (no classification) into **separate** ``seed=0_tokrun`` /
``seed=0_tokrun_discoverk`` workspaces and a **separate** collection file, so:

  * the published workspaces, predictions, and Table-2 metric cells are never
    touched (isolation --- see the project's "never overwrite completed runs"
    rule), and the ``-refinement loop`` ablation that reuses seed=0 workspaces
    is likewise untouched;
  * every run goes through the instrumented ``_run_orchestrator``, so the
    session-wide ``modelUsage`` / ``total_cost_usd`` (which include all Task
    sub-agents) are captured to ``<ws>/orchestrator_result.json`` and distilled
    by ``_summarize_orchestrator_usage``.

The token counts are a compute-cost property of running the method under
identical settings; they are reported alongside the published metrics (which
come from the original runs). This is the only obtainable big-model figure.

Idempotent / resumable: a dataset/config whose tokrun workspace already has an
``orchestrator_result.json`` is skipped, so a session-limit interruption can be
re-launched and it picks up where it left off.

    uv run --native-tls python -m benchmarking.experiments.measure_agentic_tokens --all
    uv run --native-tls python -m benchmarking.experiments.measure_agentic_tokens --only banking77
    # limit to one config:
    uv run --native-tls python -m benchmarking.experiments.measure_agentic_tokens --all --config given
"""

from __future__ import annotations

import argparse
import json
import time
import traceback

from benchmarking.baselines.agentic_clustering import (
    DATASET_LENS,
    DISCOVER_K_FRACTION,
    _ensure_orchestrator_outputs,
    _init_workspace,
    _materialize_capped_corpus,
    _run_orchestrator,
    _summarize_orchestrator_usage,
)
from benchmarking.data_processing.load import load_processed
from benchmarking.paths import RESULTS

# Banking77 first (smallest + pilot), then smallest-up, mirroring the main
# runner's SWEEP_ORDER so the quick datasets land first for verification.
SWEEP_ORDER = [
    "banking77",
    "massive_intent",
    "massive_domain",
    "stackexchange",
    "clinc150",
    "twenty_newsgroups",
    "goemotions",
]

# Isolated collection dir --- NOT under results/predictions/ (which the table
# builder globs), so it can never pollute the averaged metric cells.
OUT_DIR = RESULTS / "agentic_token_measurement"

# The published main-results runs (May 22-23) each dispatched 3 proposers in
# the initial round (the "2-3" plugin era, pre-commit 2720ff0). The current
# plugin defaults to 6-7, so we pin the count to reproduce the configuration
# that produced the Table-2 metric cells --- otherwise the big-model token
# figure would describe a more expensive pipeline than the reported accuracy.
PUBLISHED_INITIAL_PROPOSERS = 3


def _workspace_for(dataset: str, discover_k: bool):
    suffix = "seed=0_tokrun_discoverk" if discover_k else "seed=0_tokrun"
    return RESULTS / "clustering" / dataset / suffix


def _measure_one(dataset: str, *, discover_k: bool) -> dict | None:
    lens = DATASET_LENS[dataset]
    ds = load_processed(dataset)
    k_in_scope = int(ds.meta["k_in_scope"])
    if discover_k:
        k_min = round(k_in_scope * (1 - DISCOVER_K_FRACTION))
        k_max = round(k_in_scope * (1 + DISCOVER_K_FRACTION))
    else:
        k_min = k_max = k_in_scope

    workspace_dir = _workspace_for(dataset, discover_k)
    result_path = workspace_dir / "orchestrator_result.json"
    config = "discoverk" if discover_k else "given"
    tag = f"[tokrun/{config}/{dataset}]"

    if result_path.exists():
        # Already measured --- distil the saved envelope and move on.
        usage = _summarize_orchestrator_usage(
            json.loads(result_path.read_text(encoding="utf-8"))
        )
        print(f"{tag} already done; skipping. big_in={usage and usage['big_input_tokens']:,}")
        return usage

    documents_path = _materialize_capped_corpus(dataset, ds)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    print(f"{tag} init (k_range=[{k_min},{k_max}], allow_none={lens.allow_none}, n={len(ds.documents)})", flush=True)
    _init_workspace(
        workspace_dir=workspace_dir,
        documents_path=documents_path,
        k_min=k_min,
        k_max=k_max,
        lens_text=lens.text,
    )
    print(f"{tag} dispatching orchestrator (skip-classify; tokens only)", flush=True)
    t0 = time.perf_counter()
    orch = _run_orchestrator(
        workspace_dir=workspace_dir,
        dataset=dataset,
        k_min=k_min,
        k_max=k_max,
        allow_none=lens.allow_none,
        initial_proposers=PUBLISHED_INITIAL_PROPOSERS,
    )
    # Sanity: the finalize outputs should exist (a well-formed run), but a
    # missing one shouldn't discard the token capture we came for.
    try:
        _ensure_orchestrator_outputs(workspace_dir)
    except RuntimeError as exc:
        print(f"{tag} WARN: {exc}", flush=True)
    usage = orch.get("usage")
    print(
        f"{tag} done in {orch['wall_clock_s']:.0f}s | "
        + (
            f"big_in={usage['big_input_tokens']:,} big_out={usage['big_output_tokens']:,} "
            f"cost_usd=${usage.get('total_cost_usd') or 0:.2f}"
            if usage
            else "NO USAGE CAPTURED"
        ),
        flush=True,
    )
    return usage


def _collect_and_write() -> None:
    """Aggregate whatever tokrun workspaces exist into a single summary JSON."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {"given": {}, "discoverk": {}}
    for discover_k in (False, True):
        key = "discoverk" if discover_k else "given"
        for dataset in SWEEP_ORDER:
            rp = _workspace_for(dataset, discover_k) / "orchestrator_result.json"
            if not rp.exists():
                continue
            u = _summarize_orchestrator_usage(json.loads(rp.read_text(encoding="utf-8")))
            if u:
                out[key][dataset] = {
                    "big_input_tokens": u["big_input_tokens"],
                    "big_input_tokens_no_cache": u["big_input_tokens_no_cache"],
                    "big_output_tokens": u["big_output_tokens"],
                    "total_cost_usd": u["total_cost_usd"],
                    "num_turns": u["num_turns"],
                }
    for key in ("given", "discoverk"):
        got = out[key]
        if got:
            bi = sum(v["big_input_tokens"] for v in got.values())
            bo = sum(v["big_output_tokens"] for v in got.values())
            print(f"[collect] {key}: {len(got)}/7 datasets | big_in={bi/1e6:.2f}M big_out={bo/1e6:.2f}M")
    (OUT_DIR / "tokens.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[collect] wrote {OUT_DIR / 'tokens.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--only", nargs="+", choices=SWEEP_ORDER)
    group.add_argument("--all", action="store_true")
    parser.add_argument(
        "--config", choices=("given", "discoverk", "both"), default="both",
        help="Which k-configuration(s) to measure. Default: both.",
    )
    args = parser.parse_args()

    datasets = SWEEP_ORDER if args.all else (args.only or ["banking77"])
    configs: list[bool] = {
        "given": [False], "discoverk": [True], "both": [False, True],
    }[args.config]

    for dataset in datasets:
        for discover_k in configs:
            try:
                _measure_one(dataset, discover_k=discover_k)
            except Exception:
                cfg = "discoverk" if discover_k else "given"
                print(f"[tokrun/{cfg}/{dataset}] FAILED:\n{traceback.format_exc()}", flush=True)
            _collect_and_write()  # persist after every run so progress survives


if __name__ == "__main__":
    main()
