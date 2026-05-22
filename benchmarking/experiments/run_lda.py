"""LDA runner — Cat 1 baseline over all 7 processed datasets.

Deterministic; single seed (default 0) per SPEC §5.7.

Usage:
    uv run --native-tls python -m benchmarking.experiments.run_lda
    uv run --native-tls python -m benchmarking.experiments.run_lda --only banking77 clinc150
    uv run --native-tls python -m benchmarking.experiments.run_lda --seed 1
"""

from __future__ import annotations

import argparse
from typing import Iterable

from benchmarking.baselines.lda import run_lda
from benchmarking.data_processing.load import load_processed
from benchmarking.evaluation.cost import CostAccumulator, WallClock
from benchmarking.evaluation.metrics import compute_partition_metrics
from benchmarking.evaluation.persistence import (
    DocPrediction,
    TaxonomyEntry,
    write_run_artifacts,
)

METHOD = "lda"

DATASETS = [
    "banking77",
    "clinc150",
    "massive_intent",
    "massive_domain",
    "goemotions",
    "twenty_newsgroups",
    "stackexchange",
]


def run_one(dataset_name: str, seed: int) -> dict:
    ds = load_processed(dataset_name)
    k = int(ds.meta["k_in_scope"])
    texts = [d["text"] for d in ds.documents]
    gold_ids = [int(d["gold_label_id"]) for d in ds.documents]

    cost = CostAccumulator()
    with WallClock(cost):
        result = run_lda(texts=texts, k=k, seed=seed)

    metrics = compute_partition_metrics(pred_ids=result.pred_ids, gold_ids=gold_ids)

    taxonomy = [
        TaxonomyEntry(
            cluster_id=i,
            label=f"cluster_{i}",
            description=", ".join(words),
        )
        for i, words in enumerate(result.topic_top_words)
    ]

    predictions = [
        DocPrediction(
            doc_id=doc["doc_id"],
            text=doc["text"],
            gold_label=doc["gold_label_name"],
            gold_label_id=int(doc["gold_label_id"]),
            is_none=bool(doc["is_none"]),
            predicted_cluster_id=cid,
            predicted_cluster_label=f"cluster_{cid}",
            confidence=conf,
            iteration=0,
        )
        for doc, cid, conf in zip(ds.documents, result.pred_ids, result.confidences)
    ]

    write_run_artifacts(
        method=METHOD,
        dataset=dataset_name,
        seed=seed,
        predictions=predictions,
        taxonomy=taxonomy,
        cost=cost,
        model_versions={},
        iterations=0,
        metrics=metrics.to_dict(),
        hyperparameters=result.hyperparameters,
        extra_meta={"k_used": k, "n_docs": len(ds.documents)},
    )

    return {
        "dataset": dataset_name,
        "n_docs": len(ds.documents),
        "k": k,
        **metrics.to_dict(),
        "wall_clock_s": cost.wall_clock_s,
    }


def _print_table(rows: Iterable[dict]) -> None:
    rows = list(rows)
    header = f"{'dataset':<22}{'n':>8}{'k':>5}{'ARI':>10}{'NMI':>10}{'ACC':>10}{'time_s':>10}"
    print()
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['dataset']:<22}{r['n_docs']:>8}{r['k']:>5}"
            f"{r['ari']:>10.4f}{r['nmi']:>10.4f}{r['acc']:>10.4f}{r['wall_clock_s']:>10.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=DATASETS, help="Run only the named datasets.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    names = args.only if args.only else DATASETS
    rows: list[dict] = []
    for name in names:
        print(f"[{METHOD}/{name}] running…")
        row = run_one(name, args.seed)
        print(
            f"[{METHOD}/{name}] n={row['n_docs']} k={row['k']} "
            f"ARI={row['ari']:.4f} NMI={row['nmi']:.4f} ACC={row['acc']:.4f} "
            f"time={row['wall_clock_s']:.2f}s"
        )
        rows.append(row)

    _print_table(rows)


if __name__ == "__main__":
    main()
