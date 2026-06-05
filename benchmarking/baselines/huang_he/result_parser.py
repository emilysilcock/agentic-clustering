"""Phase 4: parse Huang & He outputs into the SPEC §5.11 prediction layout.

Reads ``classifications.jsonl`` + ``labels_merged.json``, maps LLM-produced
label names to contiguous cluster ids (in the order they appear in the
merged label list), computes ARI/NMI/ACC against canonical gold labels,
and calls ``benchmarking.evaluation.persistence.write_run_artifacts``.

## Fallback policy for unparseable / hallucinated rows

Phase 3 emits two sentinel labels for rows the LLM couldn't classify:

* ``<unparseable>``: JSON parse failure or batch-level error.
* ``<hallucinated>``: response parsed but label name not in the merged
  list.

Both are force-assigned to the **largest predicted cluster** here, same
policy as ``topicgpt/result_parser.py``. The method has no native
unassigned path; ``is_none`` documents go through the same force-assign
path and end up wherever the LLM puts them, naturally penalising
Huang & He for not having a native "none" output (the SPEC §5.5.3 hit).

## Cost

* Phases 1 and 3 (gpt-5-mini Batch) are metered; per-phase usage files
  (``usage_generate.json``, ``usage_classify.json``) hold the token totals.
* Phase 2 (Opus 4.7 via Claude Code Max subscription) is not metered;
  reported as ``subscription_usd = $100/7 = $14.29`` per dataset
  (symmetric with ``agentic_clustering`` per SPEC §5.6.3).
* ``api_usd`` is the dollar cost of phases 1+3 priced against the
  ``gpt-5-mini`` Batch API rates pinned by the harness (see
  ``GPT5_MINI_*`` constants).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import benchmarking  # noqa: F401 — truststore.inject_into_ssl()
import benchmarking.baselines.huang_he  # noqa: F401 — _vendored on sys.path

from benchmarking.baselines.huang_he.batch_classify import HALLUCINATED, UNPARSEABLE
from benchmarking.baselines.huang_he.dataset_adapter import HUANG_HE_ROOT
from benchmarking.data_processing.load import load_processed
from benchmarking.evaluation.cost import CostAccumulator
from benchmarking.evaluation.metrics import compute_partition_metrics
from benchmarking.evaluation.persistence import (
    DocPrediction,
    TaxonomyEntry,
    write_run_artifacts,
)

METHOD = "huang_he"
FRONTIER_MODEL = "claude-opus-4-7"
BULK_MODEL = "gpt-5-mini"

# Flat split of the $100/mo Claude Code Max subscription across the 7
# datasets in the sweep (symmetric with agentic_clustering per SPEC §5.6.3).
SUBSCRIPTION_USD_PER_DATASET = 100.0 / 7

# OpenAI gpt-5-mini Batch API pricing (50% off sync rates) — same numbers
# pinned by ``agentic_clustering.py`` as the paper artefact.
GPT5_MINI_USD_PER_1M_INPUT = 0.125
GPT5_MINI_USD_PER_1M_CACHE_READ = 0.0125
GPT5_MINI_USD_PER_1M_OUTPUT = 1.00
PRICING_BASIS = "openai_gpt_5_mini_batch_api_50pct_discount_2026_05"


def _load_classifications(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_merged_labels(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        labels = json.load(f)
    if not labels:
        raise RuntimeError(f"{path} is empty — phase 2 produced no taxonomy.")
    return [str(x) for x in labels]


def _load_phase_usage(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _phase_api_usd(usage: dict) -> float:
    """gpt-5-mini Batch-API dollar cost from per-phase token totals."""
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
    """Phase 4 entry point. Returns (jsonl_path, meta_path)."""
    ds = load_processed(dataset_name)
    out_dir = HUANG_HE_ROOT / dataset_name

    merged_labels = _load_merged_labels(out_dir / "labels_merged.json")
    name_to_cid: dict[str, int] = {name: i for i, name in enumerate(merged_labels)}

    classifications = _load_classifications(out_dir / "classifications.jsonl")
    pred_label_by_id: dict[str, str] = {c["doc_id"]: c["predicted_label_name"] for c in classifications}

    # First pass: count predictions, identify sentinel rows for fallback.
    cluster_counts: Counter[int] = Counter()
    n_unparseable = 0
    n_hallucinated = 0
    n_missing_classification = 0
    cid_for_id: dict[str, int] = {}
    for doc in ds.documents:
        pred_name = pred_label_by_id.get(doc["doc_id"])
        if pred_name is None:
            n_missing_classification += 1
            continue
        if pred_name == UNPARSEABLE:
            n_unparseable += 1
            continue
        if pred_name == HALLUCINATED:
            n_hallucinated += 1
            continue
        cid = name_to_cid.get(pred_name)
        if cid is None:
            # Defensive: a real-label-string that doesn't appear in
            # merged_labels (shouldn't happen — phase 3 sentinels them as
            # HALLUCINATED — but a manual edit to classifications.jsonl
            # could put us here).
            n_hallucinated += 1
            continue
        cid_for_id[doc["doc_id"]] = cid
        cluster_counts[cid] += 1

    if cluster_counts:
        fallback_cid = cluster_counts.most_common(1)[0][0]
    else:
        fallback_cid = 0
    fallback_label = merged_labels[fallback_cid]

    # Second pass: emit one DocPrediction per source document.
    predictions: list[DocPrediction] = []
    for doc in ds.documents:
        cid = cid_for_id.get(doc["doc_id"], fallback_cid)
        label = merged_labels[cid] if cid_for_id.get(doc["doc_id"]) is not None else fallback_label
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
        for cid, name in enumerate(merged_labels)
    ]

    pred_ids = [p.predicted_cluster_id for p in predictions]
    gold_ids = [int(d["gold_label_id"]) for d in ds.documents]
    metrics = compute_partition_metrics(pred_ids=pred_ids, gold_ids=gold_ids)

    # Cost: sum phase 1 + phase 3 metered usage; add subscription share for phase 2.
    usage_generate = _load_phase_usage(out_dir / "usage_generate.json")
    usage_classify = _load_phase_usage(out_dir / "usage_classify.json")
    usage_merge = _load_phase_usage(out_dir / "usage_merge.json")

    api_usd = _phase_api_usd(usage_generate) + _phase_api_usd(usage_classify)
    in_tokens = (
        int(usage_generate.get("input_tokens", 0) or 0)
        + int(usage_generate.get("cache_read_input_tokens", 0) or 0)
        + int(usage_classify.get("input_tokens", 0) or 0)
        + int(usage_classify.get("cache_read_input_tokens", 0) or 0)
    )
    out_tokens = (
        int(usage_generate.get("output_tokens", 0) or 0)
        + int(usage_classify.get("output_tokens", 0) or 0)
    )
    wall_clock_s = float(usage_merge.get("wall_clock_s", 0.0) or 0.0)

    cost = CostAccumulator(
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        subscription_usd=SUBSCRIPTION_USD_PER_DATASET,
        api_usd=api_usd,
        usd=SUBSCRIPTION_USD_PER_DATASET + api_usd,
        wall_clock_s=wall_clock_s,
    )

    extra_meta = {
        "k_discovered": len(merged_labels),
        "n_docs": len(ds.documents),
        "n_gold_none": sum(1 for d in ds.documents if d["is_none"]),
        "n_unparseable": n_unparseable,
        "n_hallucinated": n_hallucinated,
        "n_missing_classification": n_missing_classification,
        "n_force_assigned_to_fallback": (
            n_unparseable + n_hallucinated + n_missing_classification
        ),
        "fallback_cluster_id": int(fallback_cid),
        "fallback_cluster_label": fallback_label,
        "subscription_usd_basis": "claude_code_max_100usd_div_7_datasets",
        "pricing_basis": PRICING_BASIS,
        "usage_per_phase": {
            "generate": usage_generate,
            "merge": usage_merge,
            "classify": usage_classify,
        },
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
