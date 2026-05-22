"""BERTopic runner — Cat 2 baseline over all 7 processed datasets.

Runs *both* given-k (for the main table) and discover-k (for the SPEC §5.5
secondary table) per dataset, writing to two separate artifact paths:
    results/predictions/bertopic/<dataset>/seed=0.{jsonl,meta.json}            # given-k
    results/predictions/bertopic_discoverk/<dataset>/seed=0.{jsonl,meta.json}  # discover-k

Both variants reuse the same SBERT embedding cache as SBERT+kmeans
(`all-mpnet-base-v2`), so embedding is free on a fresh run.

Usage:
    uv run --native-tls python -m benchmarking.experiments.run_bertopic
    uv run --native-tls python -m benchmarking.experiments.run_bertopic --only banking77 clinc150
    uv run --native-tls python -m benchmarking.experiments.run_bertopic --variants given_k
"""

from __future__ import annotations

import argparse
import time
from typing import Iterable

from benchmarking.baselines.bertopic import run_bertopic
from benchmarking.data_processing.load import load_processed
from benchmarking.embeddings.sbert import embed_dataset
from benchmarking.evaluation.cost import CostAccumulator
from benchmarking.evaluation.metrics import compute_partition_metrics
from benchmarking.evaluation.persistence import (
    DocPrediction,
    TaxonomyEntry,
    write_run_artifacts,
)

METHOD_GIVEN_K = "bertopic"
METHOD_DISCOVER_K = "bertopic_discoverk"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

DATASETS = [
    "banking77",
    "clinc150",
    "massive_intent",
    "massive_domain",
    "goemotions",
    "twenty_newsgroups",
    "stackexchange",
]

VARIANTS = ("given_k", "discover_k")


def _label_for(cid: int) -> str:
    return "noise" if cid == -1 else f"cluster_{cid}"


def _run_one_variant(
    *,
    method: str,
    dataset_name: str,
    seed: int,
    embedding_model: str,
    nr_topics: int | None,
    k_in_scope: int,
) -> dict:
    ds = load_processed(dataset_name)
    texts = [d["text"] for d in ds.documents]
    gold_ids = [int(d["gold_label_id"]) for d in ds.documents]
    has_none = bool(ds.meta["has_none_class"])
    force_assign = not has_none

    t0 = time.perf_counter()
    emb = embed_dataset(dataset_name, model_name=embedding_model)
    t1 = time.perf_counter()
    result = run_bertopic(
        texts=texts,
        embeddings=emb.embeddings,
        seed=seed,
        nr_topics=nr_topics,
        force_assign_outliers=force_assign,
    )
    t2 = time.perf_counter()

    embedding_s = t1 - t0
    clustering_s = t2 - t1
    cost = CostAccumulator(wall_clock_s=t2 - t0)

    metrics = compute_partition_metrics(pred_ids=result.pred_ids, gold_ids=gold_ids)

    taxonomy = [
        TaxonomyEntry(
            cluster_id=cid,
            label=_label_for(cid),
            description=", ".join(words),
        )
        for cid, words in sorted(result.topic_top_words.items())
    ]

    predictions = [
        DocPrediction(
            doc_id=doc["doc_id"],
            text=doc["text"],
            gold_label=doc["gold_label_name"],
            gold_label_id=int(doc["gold_label_id"]),
            is_none=bool(doc["is_none"]),
            predicted_cluster_id=cid,
            predicted_cluster_label=_label_for(cid),
            confidence=None,
            iteration=0,
        )
        for doc, cid in zip(ds.documents, result.pred_ids)
    ]

    hyperparameters = {
        **result.hyperparameters,
        "embedding_model": embedding_model,
        "embedding_cache_hit": emb.cache_hit,
        "k_in_scope": k_in_scope,
        "variant": "given_k" if nr_topics is not None else "discover_k",
    }

    write_run_artifacts(
        method=method,
        dataset=dataset_name,
        seed=seed,
        predictions=predictions,
        taxonomy=taxonomy,
        cost=cost,
        model_versions={"embedding": embedding_model},
        iterations=0,
        metrics=metrics.to_dict(),
        hyperparameters=hyperparameters,
        extra_meta={
            "k_used": k_in_scope,
            "n_docs": len(ds.documents),
            "n_topics_actual": result.n_topics_actual,
            "n_noise": result.n_noise,
            "cost_breakdown": {
                "embedding_s": embedding_s,
                "clustering_s": clustering_s,
                "cache_hit_embedding": emb.cache_hit,
            },
        },
    )

    return {
        "method": method,
        "dataset": dataset_name,
        "n_docs": len(ds.documents),
        "k": k_in_scope,
        "n_topics_actual": result.n_topics_actual,
        "n_noise": result.n_noise,
        **metrics.to_dict(),
        "wall_clock_s": cost.wall_clock_s,
    }


def run_one(dataset_name: str, seed: int, embedding_model: str, variants: tuple[str, ...]) -> list[dict]:
    ds_meta = load_processed(dataset_name).meta
    k_in_scope = int(ds_meta["k_in_scope"])
    rows: list[dict] = []
    if "given_k" in variants:
        rows.append(
            _run_one_variant(
                method=METHOD_GIVEN_K,
                dataset_name=dataset_name,
                seed=seed,
                embedding_model=embedding_model,
                nr_topics=k_in_scope,
                k_in_scope=k_in_scope,
            )
        )
    if "discover_k" in variants:
        rows.append(
            _run_one_variant(
                method=METHOD_DISCOVER_K,
                dataset_name=dataset_name,
                seed=seed,
                embedding_model=embedding_model,
                nr_topics=None,
                k_in_scope=k_in_scope,
            )
        )
    return rows


def _print_table(rows: Iterable[dict]) -> None:
    rows = list(rows)
    header = (
        f"{'method':<22}{'dataset':<22}{'n':>7}{'k':>5}{'k_act':>7}"
        f"{'noise':>7}{'ARI':>9}{'NMI':>9}{'ACC':>9}{'time_s':>9}"
    )
    print()
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['method']:<22}{r['dataset']:<22}{r['n_docs']:>7}{r['k']:>5}"
            f"{r['n_topics_actual']:>7}{r['n_noise']:>7}"
            f"{r['ari']:>9.4f}{r['nmi']:>9.4f}{r['acc']:>9.4f}{r['wall_clock_s']:>9.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=DATASETS, help="Run only the named datasets.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
        help="Which BERTopic variants to run (default: both).",
    )
    args = parser.parse_args()

    names = args.only if args.only else DATASETS
    variants = tuple(args.variants)
    rows: list[dict] = []
    for name in names:
        print(f"[bertopic/{name}] running variants={variants}…")
        for row in run_one(name, args.seed, args.model, variants):
            print(
                f"[{row['method']}/{name}] n={row['n_docs']} k={row['k']} k_act={row['n_topics_actual']} "
                f"noise={row['n_noise']} ARI={row['ari']:.4f} NMI={row['nmi']:.4f} "
                f"ACC={row['acc']:.4f} time={row['wall_clock_s']:.2f}s"
            )
            rows.append(row)

    _print_table(rows)


if __name__ == "__main__":
    main()
