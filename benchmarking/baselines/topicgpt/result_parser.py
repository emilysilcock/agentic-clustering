"""Phase 5: parse TopicGPT outputs into the SPEC §5.11 prediction layout.

Reads ``data/topicgpt/<ds>/corrected.jsonl`` (or ``assignment.jsonl`` if
correction wasn't run), maps LLM-produced topic names to contiguous cluster
ids derived from ``topics_refined.md``, computes ARI/NMI/ACC against the
canonical gold labels, and calls
``benchmarking.evaluation.persistence.write_run_artifacts``.

## Fallback policy for unparseable / hallucinated rows

After phase 4 (correction), rows whose ``responses`` field still parses to
zero valid topic names are force-assigned to the **largest predicted
cluster**. This matches the SPEC §5.5 framing that TopicGPT "joins the
non-'none'-aware baselines" --- the method has no native unassigned path,
so we don't invent one. The fallback rate is recorded in ``meta.json``
(``n_unparseable``, ``n_hallucinated_post_correction``) so reviewers can
see how often it fires.

## None-class documents

In datasets with a ``none`` gold class (CLINC OOS, GoEmotions neutral),
``is_none`` documents are **passed through** to TopicGPT at adapt time
(no privileged information leak --- see ``dataset_adapter.py`` module
docstring). The method then assigns its own choice of topic to each
is_none doc. Here we use those predictions as-is; metrics are computed
against the gold labels (which include ``__none__`` for these rows),
naturally penalising TopicGPT for not having a native "none" output
--- the penalty SPEC §5.5.3 anticipates.

The fallback-to-largest-cluster policy described in
"Fallback policy" above still applies to rows that fail parsing after
correction (unparseable + hallucinated), regardless of whether they are
gold-none. None of the fallback choices know about gold labels.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import regex

import benchmarking  # noqa: F401 — truststore.inject_into_ssl()
import benchmarking.baselines.topicgpt  # noqa: F401 — _vendored on sys.path

from benchmarking.baselines.topicgpt.dataset_adapter import TOPICGPT_ROOT
from benchmarking.data_processing.load import load_processed
from benchmarking.evaluation.cost import CostAccumulator
from benchmarking.evaluation.metrics import compute_partition_metrics
from benchmarking.evaluation.persistence import (
    DocPrediction,
    TaxonomyEntry,
    write_run_artifacts,
)

METHOD = "topicgpt"
FRONTIER_MODEL = "claude-opus-4-7"   # phases 1+2 (generate, refine) via Claude Code Max
BULK_MODEL = "gpt-5-mini"            # phases 3+4 (assign, correct) via OpenAI Batch

# Flat split of the $100/mo Claude Code Max subscription across the 7 datasets
# in the sweep (symmetric with huang_he / agentic_clustering per SPEC §5.6.3).
SUBSCRIPTION_USD_PER_DATASET = 100.0 / 7

# OpenAI gpt-5-mini Batch API pricing (50% off sync rates). Pinned as a paper
# artefact — the rates we paid at run time, not whatever is current at re-run.
GPT5_MINI_USD_PER_1M_INPUT = 0.125
GPT5_MINI_USD_PER_1M_CACHE_READ = 0.0125
GPT5_MINI_USD_PER_1M_OUTPUT = 1.00
PRICING_BASIS = "openai_gpt_5_mini_batch_api_50pct_discount_2026_05"

# Per-phase usage_<phase>.json sidecars summed into the cost. The optional
# ``correct_initial_estimated`` carries a *peer-rate estimate* of an initial
# correction pass whose measured usage was overwritten by a later
# --retry-correct run (only twenty_newsgroups has one — see its ``basis``
# field). It is priced and counted exactly like the measured OpenAI phases.
_USAGE_PHASES = ("generate", "refine", "assign", "correct", "correct_initial_estimated")

# Phases billed to OpenAI (gpt-5-mini Batch); the rest (generate, refine) run
# on the Opus subscription covered by SUBSCRIPTION_USD_PER_DATASET.
_OPENAI_PHASES = ("assign", "correct", "correct_initial_estimated")

# Same regex correction.py uses to extract topic names from responses; see
# vendored correction.py:31. Keeping the patterns identical so what survives
# correction here parses identically here.
_TOPIC_PATTERN = regex.compile(r"^\[\d+\] ([\w\s\-'\&]+):", regex.MULTILINE)


def _load_topic_taxonomy(topic_file: Path) -> list[str]:
    """Read topics_refined.md and return topic names in file order."""
    from topicgpt_python.utils import TopicTree

    tree = TopicTree().from_topic_list(str(topic_file), from_file=True)
    return tree.get_root_descendants_name()


def _load_responses(out_dir: Path) -> dict[str, str]:
    """Prefer corrected.jsonl over assignment.jsonl. Key by ``id``."""
    src = out_dir / "corrected.jsonl"
    if not src.exists() or src.stat().st_size == 0:
        src = out_dir / "assignment.jsonl"
    if not src.exists() or src.stat().st_size == 0:
        raise FileNotFoundError(
            f"phase 5 needs phase 3 (and ideally phase 4) outputs in {out_dir}"
        )
    by_id: dict[str, str] = {}
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        by_id[row["id"]] = row.get("responses", "")
    return by_id


def _sum_usage(out_dir: Path) -> dict:
    """Aggregate the per-phase usage sidecars (_USAGE_PHASES) into one dict."""
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    per_phase: dict[str, dict] = {}
    for phase in _USAGE_PHASES:
        p = out_dir / f"usage_{phase}.json"
        if not p.exists():
            continue
        u = json.loads(p.read_text(encoding="utf-8"))
        per_phase[phase] = u
        for k in totals:
            totals[k] += int(u.get(k, 0) or 0)
    return {"total": totals, "per_phase": per_phase}


def _phase_api_usd(usage: dict) -> float:
    """gpt-5-mini Batch-API dollar cost from one phase's token totals."""
    in_t = int(usage.get("input_tokens", 0) or 0)
    cache_t = int(usage.get("cache_read_input_tokens", 0) or 0)
    out_t = int(usage.get("output_tokens", 0) or 0)
    return (
        in_t * GPT5_MINI_USD_PER_1M_INPUT
        + cache_t * GPT5_MINI_USD_PER_1M_CACHE_READ
        + out_t * GPT5_MINI_USD_PER_1M_OUTPUT
    ) / 1_000_000.0


def write(
    dataset_name: str,
    *,
    seed: int = 0,
    hyperparameters: dict | None = None,
) -> tuple[Path, Path]:
    """Phase 5 entry point. Returns (jsonl_path, meta_path)."""
    ds = load_processed(dataset_name)
    out_dir = TOPICGPT_ROOT / dataset_name

    topic_names = _load_topic_taxonomy(out_dir / "topics_refined.md")
    if not topic_names:
        raise RuntimeError(
            f"{out_dir / 'topics_refined.md'} contains zero topics; "
            f"phase 1/2 likely produced no usable taxonomy."
        )
    name_to_cid = {name: i for i, name in enumerate(topic_names)}
    responses_by_id = _load_responses(out_dir)

    # First pass: parse responses, count predictions, accumulate stats.
    # ``is_none`` docs are NOT skipped --- they were fed to TopicGPT (no
    # privileged-info leak, see dataset_adapter.py docstring) and have
    # their own LLM-assigned topic, which we use as-is.
    pred_for_id: dict[str, tuple[int, str]] = {}
    cluster_counts: Counter[int] = Counter()
    n_unparseable = 0
    n_hallucinated = 0
    n_missing_response = 0
    for doc in ds.documents:
        resp = responses_by_id.get(doc["doc_id"])
        if resp is None:
            n_missing_response += 1
            continue
        topics = [t.strip() for t in _TOPIC_PATTERN.findall(resp)]
        if not topics:
            n_unparseable += 1
            continue
        valid = [t for t in topics if t in name_to_cid]
        if not valid:
            n_hallucinated += 1
            continue
        # First valid extracted topic wins (matches upstream parse order).
        first_topic = valid[0]
        cid = name_to_cid[first_topic]
        pred_for_id[doc["doc_id"]] = (cid, first_topic)
        cluster_counts[cid] += 1

    if cluster_counts:
        fallback_cid = cluster_counts.most_common(1)[0][0]
    else:
        fallback_cid = 0
    fallback_label = topic_names[fallback_cid]

    # Second pass: emit one DocPrediction per source document (including is_none).
    predictions: list[DocPrediction] = []
    for doc in ds.documents:
        cid, label = pred_for_id.get(doc["doc_id"], (fallback_cid, fallback_label))
        predictions.append(
            DocPrediction(
                doc_id=doc["doc_id"],
                text=doc["text"],
                gold_label=doc["gold_label_name"],
                gold_label_id=int(doc["gold_label_id"]),
                is_none=bool(doc["is_none"]),
                predicted_cluster_id=int(cid),
                predicted_cluster_label=label,
                confidence=None,
                iteration=0,
            )
        )

    taxonomy = [
        TaxonomyEntry(cluster_id=cid, label=name, description="")
        for cid, name in enumerate(topic_names)
    ]

    pred_ids = [p.predicted_cluster_id for p in predictions]
    gold_ids = [int(d["gold_label_id"]) for d in ds.documents]
    metrics = compute_partition_metrics(pred_ids=pred_ids, gold_ids=gold_ids)

    # Cost (SPEC §5.6.3, symmetric with huang_he / agentic_clustering):
    #   * subscription_usd — phases 1+2 (generate, refine) run on Opus 4.7 via
    #     the Claude Code Max subscription, which has no per-call USD; charged
    #     as a flat $100/7 per dataset (sums to $100 across the 7-dataset sweep).
    #   * api_usd — phases 3+4 (assign, correct) run on gpt-5-mini via the
    #     OpenAI Batch API; metered from the per-phase token totals. Includes
    #     the optional ``correct_initial_estimated`` sidecar (peer-rate estimate
    #     of a correction pass overwritten by --retry-correct; 20NG only).
    #   * usd = subscription_usd + api_usd  (the figure the results table sums).
    usage = _sum_usage(out_dir)
    api_usd = sum(
        _phase_api_usd(usage["per_phase"].get(phase, {})) for phase in _OPENAI_PHASES
    )
    est = usage["per_phase"].get("correct_initial_estimated")
    cost = CostAccumulator(
        input_tokens=int(usage["total"]["input_tokens"]),
        output_tokens=int(usage["total"]["output_tokens"]),
        subscription_usd=SUBSCRIPTION_USD_PER_DATASET,
        api_usd=api_usd,
        usd=SUBSCRIPTION_USD_PER_DATASET + api_usd,
    )

    extra_meta = {
        "k_discovered": len(topic_names),
        "n_unparseable": n_unparseable,
        "n_hallucinated_post_correction": n_hallucinated,
        "n_missing_response": n_missing_response,
        "n_force_assigned_to_fallback": (
            n_unparseable + n_hallucinated + n_missing_response
        ),
        "fallback_cluster_id": int(fallback_cid),
        "fallback_cluster_label": fallback_label,
        "usage_per_phase": usage["per_phase"],
        "n_docs": len(ds.documents),
        "n_gold_none": sum(1 for d in ds.documents if d["is_none"]),
        "subscription_usd_basis": "claude_code_max_100usd_div_7_datasets",
        "pricing_basis": PRICING_BASIS,
        "correction_initial_pass_estimated_usd": (
            round(_phase_api_usd(est), 4) if est else None
        ),
    }

    return write_run_artifacts(
        method=METHOD,
        dataset=dataset_name,
        seed=seed,
        predictions=predictions,
        taxonomy=taxonomy,
        cost=cost,
        model_versions={"frontier": FRONTIER_MODEL, "bulk": BULK_MODEL},
        iterations=0,
        metrics=metrics.to_dict(),
        hyperparameters=hyperparameters or {},
        extra_meta=extra_meta,
    )
