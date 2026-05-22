"""LDA baseline (SPEC §5.2, Cat 1).

sklearn's CountVectorizer + LatentDirichletAllocation. Deterministic given a
seed; single seed per SPEC §5.7. Returns per-doc cluster ids (argmax over the
doc-topic distribution), per-doc confidences (max of that distribution), and
per-topic top words so the taxonomy.json sidecar carries something readable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


@dataclass
class LDAResult:
    pred_ids: list[int]
    confidences: list[float]
    topic_top_words: list[list[str]]  # length k; each is a list of top-N words
    hyperparameters: dict  # the actual values used (after any min_df fallback)


def run_lda(
    texts: list[str],
    k: int,
    seed: int,
    *,
    max_iter: int = 20,
    top_words: int = 10,
    min_df: int = 2,
    max_df: float = 0.95,
) -> LDAResult:
    """Fit LDA with k topics and return per-doc assignments + topic top words.

    - `min_df=2` drops hapaxes; falls back to 1 if the corpus is too small for that to leave any vocab.
    - `max_df=0.95` drops words present in >95% of docs (LDA's natural stop-word filter on short corpora).
    """
    effective_min_df = min_df
    vectoriser_kwargs = dict(stop_words="english", lowercase=True, min_df=min_df, max_df=max_df)
    try:
        vectoriser = CountVectorizer(**vectoriser_kwargs)
        X = vectoriser.fit_transform(texts)
    except ValueError:
        effective_min_df = 1
        vectoriser_kwargs["min_df"] = 1
        vectoriser = CountVectorizer(**vectoriser_kwargs)
        X = vectoriser.fit_transform(texts)

    lda = LatentDirichletAllocation(
        n_components=k,
        random_state=seed,
        learning_method="batch",
        max_iter=max_iter,
    )
    doc_topic = lda.fit_transform(X)  # (n_docs, k)

    pred_ids = doc_topic.argmax(axis=1).astype(int).tolist()
    confidences = doc_topic.max(axis=1).astype(float).tolist()

    vocab = np.array(vectoriser.get_feature_names_out())
    topic_top_words: list[list[str]] = []
    for topic_row in lda.components_:
        order = np.argsort(topic_row)[::-1][:top_words]
        topic_top_words.append(vocab[order].tolist())

    hyperparameters = {
        "k": k,
        "max_iter": max_iter,
        "min_df": effective_min_df,
        "max_df": max_df,
        "top_words": top_words,
        "stop_words": "english",
        "lowercase": True,
        "learning_method": "batch",
        "vocab_size": int(X.shape[1]),
    }

    return LDAResult(
        pred_ids=pred_ids,
        confidences=confidences,
        topic_top_words=topic_top_words,
        hyperparameters=hyperparameters,
    )
