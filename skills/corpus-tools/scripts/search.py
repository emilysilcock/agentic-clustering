#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["scikit-learn"]
# ///
"""TF-IDF similarity search over the corpus.

Builds and caches a TF-IDF matrix on first call. Returns top-N matches
for a query string.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path


def _get_workspace() -> Path:
    env_ws = os.environ.get("CLUSTERING_WORKSPACE")
    if env_ws:
        return Path(env_ws)
    return Path(".claude/clustering")


WORKSPACE = _get_workspace()
CACHE_DIR = WORKSPACE / "tfidf_cache"


def load_corpus() -> list[dict]:
    corpus_path = WORKSPACE / "corpus.json"
    if not corpus_path.exists():
        print("Error: workspace not initialized. Run init.py first.", file=sys.stderr)
        sys.exit(1)
    with open(corpus_path, encoding="utf-8") as f:
        return json.load(f)


def get_or_build_tfidf(corpus: list[dict]):
    """Load cached TF-IDF matrix or build and cache it."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        print("Error: scikit-learn required. Install with: uv add scikit-learn", file=sys.stderr)
        sys.exit(1)

    vectorizer_path = CACHE_DIR / "vectorizer.pkl"
    matrix_path = CACHE_DIR / "matrix.pkl"
    ids_path = CACHE_DIR / "ids.json"

    corpus_path = WORKSPACE / "corpus.json"
    cache_valid = (
        vectorizer_path.exists()
        and matrix_path.exists()
        and ids_path.exists()
        and corpus_path.exists()
        and vectorizer_path.stat().st_mtime >= corpus_path.stat().st_mtime
    )

    if cache_valid:
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        with open(matrix_path, "rb") as f:
            matrix = pickle.load(f)
        with open(ids_path, encoding="utf-8") as f:
            ids = json.load(f)
        return vectorizer, matrix, ids

    # Build from scratch
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    texts = [r["text"] for r in corpus]
    ids = [r["id"] for r in corpus]

    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    matrix = vectorizer.fit_transform(texts)

    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(matrix_path, "wb") as f:
        pickle.dump(matrix, f)
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(ids, f)

    print("TF-IDF index built and cached.", file=sys.stderr)
    return vectorizer, matrix, ids


def search(query: str, n: int) -> list[dict]:
    """Search corpus for texts similar to query."""
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    corpus = load_corpus()
    vectorizer, matrix, ids = get_or_build_tfidf(corpus)

    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, matrix).flatten()

    top_indices = np.argsort(similarities)[::-1][:n]

    id_to_text = {r["id"]: r["text"] for r in corpus}
    results = []
    for idx in top_indices:
        sim = float(similarities[idx])
        if sim > 0:
            text_id = ids[idx]
            results.append({
                "id": text_id,
                "text": id_to_text.get(text_id, ""),
                "similarity": round(sim, 4),
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="TF-IDF similarity search")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--n", type=int, default=10, help="Number of results")
    args = parser.parse_args()

    results = search(args.query, args.n)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
