"""BERTopic baseline (SPEC §5.2, Cat 2).

Pure function: takes pre-computed SBERT embeddings (same cache as SBERT+kmeans) and
returns cluster assignments + intrinsic per-cluster top words from c-TF-IDF.

Two call modes per the SPEC §5.5 k-handling rule:
- given-k:     `nr_topics=k_in_scope` (BERTopic post-hoc-merges discovered topics)
- discover-k:  `nr_topics=None`       (HDBSCAN's natural discovery is kept)

We deliberately do NOT call `reduce_outliers()` — HDBSCAN's noise label (-1) is
BERTopic's native "none" output, which aligns with our gold -1 / __none__ class
on CLINC OOS and GoEmotions neutral (SPEC §5.3 advantage).

Deterministic given the seed (via UMAP's random_state); HDBSCAN is deterministic
given the same inputs. Single seed per SPEC §5.7.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BERTopicResult:
    pred_ids: list[int]                 # may contain -1 (HDBSCAN noise / native "none")
    topic_top_words: dict[int, list[str]]  # cluster_id -> top-N words from c-TF-IDF
    n_topics_actual: int                # excludes the -1 noise cluster
    n_noise: int                        # number of docs assigned to cluster -1
    hyperparameters: dict


def run_bertopic(
    texts: list[str],
    embeddings: np.ndarray,
    seed: int,
    *,
    nr_topics: int | None = None,
    force_assign_outliers: bool = False,
    top_n_words: int = 10,
    umap_n_neighbors: int = 15,
    umap_n_components: int = 5,
    umap_min_dist: float = 0.0,
    umap_metric: str = "cosine",
    hdbscan_min_cluster_size: int = 10,
    hdbscan_min_samples: int | None = None,
) -> BERTopicResult:
    """Fit BERTopic on pre-computed SBERT embeddings.

    `nr_topics`:
      - `None`        — keep HDBSCAN's natural topics (discover-k).
      - int           — post-hoc hierarchical-merge to this many topics (given-k).

    `force_assign_outliers`:
      - `False`       — keep HDBSCAN's noise cluster (-1) as the method's native "none" output.
      - `True`        — post-hoc reassign every -1 doc to its nearest topic via
                        BERTopic's `reduce_outliers(strategy='embeddings')`, leaving the
                        partition with no noise. Used on datasets without a gold "none" class
                        per our dataset-specific noise-handling rule.
    """
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    umap_model = UMAP(
        n_neighbors=umap_n_neighbors,
        n_components=umap_n_components,
        min_dist=umap_min_dist,
        metric=umap_metric,
        random_state=seed,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=False,
    )
    vectoriser_model = CountVectorizer(stop_words="english")

    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectoriser_model,
        nr_topics=nr_topics,
        top_n_words=top_n_words,
        calculate_probabilities=False,
        verbose=False,
    )

    topics, _ = topic_model.fit_transform(documents=texts, embeddings=embeddings)

    n_noise_pre_reassign = sum(1 for t in topics if t == -1)
    if force_assign_outliers and n_noise_pre_reassign > 0:
        topics = topic_model.reduce_outliers(
            documents=texts,
            topics=topics,
            strategy="embeddings",
            embeddings=embeddings,
        )

    pred_ids = [int(t) for t in topics]

    # Build {cluster_id: top_words} from BERTopic's c-TF-IDF representation.
    topic_top_words: dict[int, list[str]] = {}
    for cid in set(pred_ids):
        words_and_scores = topic_model.get_topic(cid) or []
        topic_top_words[cid] = [w for (w, _score) in words_and_scores]

    unique_topics = set(pred_ids)
    n_noise = sum(1 for t in pred_ids if t == -1)
    n_topics_actual = len(unique_topics - {-1})

    hyperparameters = {
        "nr_topics_requested": nr_topics,
        "n_topics_actual": n_topics_actual,
        "force_assign_outliers": force_assign_outliers,
        "n_noise_pre_reassign": n_noise_pre_reassign,
        "top_n_words": top_n_words,
        "umap": {
            "n_neighbors": umap_n_neighbors,
            "n_components": umap_n_components,
            "min_dist": umap_min_dist,
            "metric": umap_metric,
            "random_state": seed,
        },
        "hdbscan": {
            "min_cluster_size": hdbscan_min_cluster_size,
            "min_samples": hdbscan_min_samples,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
        },
        "vectorizer_stop_words": "english",
        "embedding_dim": int(embeddings.shape[1]),
    }

    return BERTopicResult(
        pred_ids=pred_ids,
        topic_top_words=topic_top_words,
        n_topics_actual=n_topics_actual,
        n_noise=n_noise,
        hyperparameters=hyperparameters,
    )
