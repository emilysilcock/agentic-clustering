"""Persist a single run to results/predictions/<method>/<dataset>/seed=<n>.{jsonl,meta.json}.

SPEC §5.11 defines this layout. Every runner in benchmarking/experiments/ calls
write_run_artifacts() exactly once per (method, dataset, seed) cell. Metric
computation reads from these files, so re-scoring with a new metric requires
no model calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from benchmarking.data_processing.base import Document
from benchmarking.evaluation.cost import CostAccumulator
from benchmarking.paths import RESULTS


@dataclass
class TaxonomyEntry:
    cluster_id: int
    label: str
    description: str = ""

    def to_dict(self) -> dict:
        return {"cluster_id": self.cluster_id, "label": self.label, "description": self.description}


@dataclass
class DocPrediction:
    doc_id: str
    text: str
    gold_label: str
    gold_label_id: int
    is_none: bool
    predicted_cluster_id: int
    predicted_cluster_label: str
    confidence: float | None = None
    iteration: int = 0

    def to_dict(self) -> dict:
        d = {
            "doc_id": self.doc_id,
            "text": self.text,
            "gold_label": self.gold_label,
            "gold_label_id": self.gold_label_id,
            "is_none": self.is_none,
            "iteration": self.iteration,
            "predicted_cluster_id": self.predicted_cluster_id,
            "predicted_cluster_label": self.predicted_cluster_label,
        }
        if self.confidence is not None:
            d["confidence"] = self.confidence
        return d


def predictions_dir(method: str, dataset: str) -> Path:
    return RESULTS / "predictions" / method / dataset


def write_run_artifacts(
    *,
    method: str,
    dataset: str,
    seed: int,
    predictions: Iterable[DocPrediction],
    taxonomy: Sequence[TaxonomyEntry],
    cost: CostAccumulator,
    model_versions: dict | None = None,
    iterations: int = 0,
    metrics: dict | None = None,
    hyperparameters: dict | None = None,
    extra_meta: dict | None = None,
) -> tuple[Path, Path]:
    """Write seed=<n>.jsonl and seed=<n>.meta.json. Returns the two paths."""
    out_dir = predictions_dir(method, dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / f"seed={seed}.jsonl"
    meta_path = out_dir / f"seed={seed}.meta.json"

    jsonl_path.write_text(
        "\n".join(json.dumps(p.to_dict(), ensure_ascii=False) for p in predictions) + "\n",
        encoding="utf-8",
    )

    meta: dict = {
        "method": method,
        "dataset": dataset,
        "seed": seed,
        "taxonomy": [t.to_dict() for t in taxonomy],
        "cost": cost.to_dict(),
        "model_versions": model_versions or {},
        "iterations": iterations,
        "hyperparameters": hyperparameters or {},
    }
    if metrics is not None:
        meta["metrics"] = metrics
    if extra_meta:
        meta.update(extra_meta)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return jsonl_path, meta_path


def predictions_to_doc_predictions(
    documents: list[Document],
    pred_ids: list[int],
    *,
    predicted_label_for: callable = lambda cid: f"cluster_{cid}",
    confidences: list[float] | None = None,
) -> list[DocPrediction]:
    """Adapter: zip (document, predicted_cluster_id) into DocPrediction records."""
    assert len(documents) == len(pred_ids), (
        f"docs ({len(documents)}) vs preds ({len(pred_ids)}) length mismatch"
    )
    if confidences is not None:
        assert len(confidences) == len(documents)
    out: list[DocPrediction] = []
    for i, (doc, cid) in enumerate(zip(documents, pred_ids)):
        out.append(
            DocPrediction(
                doc_id=doc["doc_id"],
                text=doc["text"],
                gold_label=doc["gold_label_name"],
                gold_label_id=int(doc["gold_label_id"]),
                is_none=bool(doc["is_none"]),
                predicted_cluster_id=int(cid),
                predicted_cluster_label=predicted_label_for(int(cid)),
                confidence=(float(confidences[i]) if confidences is not None else None),
                iteration=0,
            )
        )
    return out
