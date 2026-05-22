"""k-means clustering on pre-computed embeddings.

Pure function: takes any (n_docs, dim) embedding array and returns cluster
assignments. Embedding production lives in `benchmarking.embeddings.*` so
multiple methods (SBERT+kmeans, LLM-embedding+kmeans, ...) share both this
clusterer and the per-model embedding cache.

Deterministic given the seed; single seed per SPEC §5.7.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


@dataclass
class KMeansResult:
    pred_ids: list[int]
    hyperparameters: dict


def run_kmeans(
    embeddings: np.ndarray,
    k: int,
    seed: int,
    *,
    n_init: int = 10,
    max_iter: int = 300,
    l2_normalize: bool = True,
) -> KMeansResult:
    """Fit k-means on (optionally L2-normalized) embeddings; return per-doc cluster ids.

    L2-normalizing turns Euclidean k-means into cosine k-means on the unit sphere,
    which is the standard convention for sentence embeddings.
    """
    X = embeddings.astype(np.float32, copy=False)
    if l2_normalize:
        X = normalize(X, norm="l2", axis=1, copy=True)

    km = KMeans(
        n_clusters=k,
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
    )
    pred_ids = km.fit_predict(X).astype(int).tolist()

    hyperparameters = {
        "k": k,
        "n_init": n_init,
        "max_iter": max_iter,
        "l2_normalize": l2_normalize,
        "kmeans_algorithm": "lloyd",
        "init": "k-means++",
        "embedding_dim": int(X.shape[1]),
    }
    return KMeansResult(pred_ids=pred_ids, hyperparameters=hyperparameters)
