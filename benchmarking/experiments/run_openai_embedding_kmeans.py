"""LLM-embedding + k-means runner over all 7 processed datasets.

Uses OpenAI `text-embedding-3-large` via the Batch API (per SPEC §5.6.3) and
sklearn k-means with `k = k_in_scope` per SPEC §5.5.

Deterministic given the seed; single seed per SPEC §5.7.

Usage:
    uv run --native-tls python -m benchmarking.experiments.run_openai_embedding_kmeans
    uv run --native-tls python -m benchmarking.experiments.run_openai_embedding_kmeans --only banking77
"""

from __future__ import annotations

import argparse
from typing import Iterable

from benchmarking.baselines.kmeans import run_kmeans
from benchmarking.data_processing.load import load_processed
from benchmarking.embeddings.openai_embeddings import (
    cost_from_meta,
    embed_dataset,
    embed_datasets,
)
from benchmarking.evaluation.cost import CostAccumulator, WallClock
from benchmarking.evaluation.metrics import compute_partition_metrics
from benchmarking.evaluation.persistence import (
    DocPrediction,
    TaxonomyEntry,
    write_run_artifacts,
)

METHOD = "openai_embedding_kmeans"
DEFAULT_MODEL = "text-embedding-3-large"

DATASETS = [
    "banking77",
    "clinc150",
    "massive_intent",
    "massive_domain",
    "goemotions",
    "twenty_newsgroups",
    "stackexchange",
]


def run_one(dataset_name: str, seed: int, model_name: str) -> dict:
    ds = load_processed(dataset_name)
    k = int(ds.meta["k_in_scope"])
    gold_ids = [int(d["gold_label_id"]) for d in ds.documents]

    cost = CostAccumulator()
    with WallClock(cost):
        emb = embed_dataset(dataset_name, model_name=model_name)
        result = run_kmeans(embeddings=emb.embeddings, k=k, seed=seed)

    # Persisted cost from the embedding sidecar is the actual amount paid to
    # OpenAI for these vectors. Cache hits and cache misses both report this
    # — for cache hits it's the historical real cost, not zero.
    persisted_cost = cost_from_meta(emb.meta_path)
    cost.input_tokens = int(persisted_cost.get("input_tokens", 0))
    cost.usd = float(persisted_cost.get("usd", 0.0))

    metrics = compute_partition_metrics(pred_ids=result.pred_ids, gold_ids=gold_ids)

    taxonomy = [
        TaxonomyEntry(cluster_id=i, label=f"cluster_{i}", description="")
        for i in range(k)
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
            confidence=None,
            iteration=0,
        )
        for doc, cid in zip(ds.documents, result.pred_ids)
    ]

    hyperparameters = {
        **result.hyperparameters,
        "embedding_model": model_name,
        "embedding_cache_hit": emb.cache_hit,
        "embedding_api_mode": "batch",
    }

    write_run_artifacts(
        method=METHOD,
        dataset=dataset_name,
        seed=seed,
        predictions=predictions,
        taxonomy=taxonomy,
        cost=cost,
        model_versions={"embedding": model_name},
        iterations=0,
        metrics=metrics.to_dict(),
        hyperparameters=hyperparameters,
        extra_meta={"k_used": k, "n_docs": len(ds.documents)},
    )

    return {
        "dataset": dataset_name,
        "n_docs": len(ds.documents),
        "k": k,
        **metrics.to_dict(),
        "wall_clock_s": cost.wall_clock_s,
        "input_tokens": cost.input_tokens,
        "usd": cost.usd,
        "cache_hit": emb.cache_hit,
    }


def _print_table(rows: Iterable[dict]) -> None:
    rows = list(rows)
    header = (
        f"{'dataset':<22}{'n':>8}{'k':>5}{'ARI':>10}{'NMI':>10}{'ACC':>10}"
        f"{'tokens':>12}{'USD':>9}{'time_s':>10}  cache"
    )
    print()
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['dataset']:<22}{r['n_docs']:>8}{r['k']:>5}"
            f"{r['ari']:>10.4f}{r['nmi']:>10.4f}{r['acc']:>10.4f}"
            f"{r['input_tokens']:>12d}{r['usd']:>9.4f}{r['wall_clock_s']:>10.2f}  "
            f"{'hit' if r['cache_hit'] else 'miss'}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=DATASETS, help="Run only the named datasets.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()

    names = args.only if args.only else DATASETS

    # Phase 1: kick off all embedding batches in parallel and wait for them.
    # Subsequent embed_dataset() calls inside run_one will all be cache hits.
    print(f"[{METHOD}] phase 1: embedding {len(names)} dataset(s) in parallel...")
    embed_datasets(names, model_name=args.model)
    print(f"[{METHOD}] phase 1 complete; running k-means + persistence per dataset.")

    # Phase 2: per-dataset k-means + metrics + persistence.
    rows: list[dict] = []
    for name in names:
        print(f"[{METHOD}/{name}] running…")
        row = run_one(name, args.seed, args.model)
        print(
            f"[{METHOD}/{name}] n={row['n_docs']} k={row['k']} "
            f"ARI={row['ari']:.4f} NMI={row['nmi']:.4f} ACC={row['acc']:.4f} "
            f"tokens={row['input_tokens']} USD={row['usd']:.4f} "
            f"time={row['wall_clock_s']:.2f}s cache={'hit' if row['cache_hit'] else 'miss'}"
        )
        rows.append(row)

    _print_table(rows)


if __name__ == "__main__":
    main()
