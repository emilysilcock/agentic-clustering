#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["scikit-learn", "filelock"]
# ///
"""TF-IDF similarity search over the corpus.

Builds and caches a TF-IDF matrix on first call. Returns top-N matches
for a query string. The cache build is wrapped in a dedicated FileLock so
concurrent agents (e.g. parallel investigators) can't race on the pickle
writes; the lock is separate from sample.py's `.state.lock` so a long
rebuild doesn't block sampling.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

# Force UTF-8 on stdout/stderr — Windows defaults to cp1252 and crashes on
# non-ASCII cluster names / corpus content. Idempotent; no-op on streams that
# aren't TextIOWrapper (e.g. captured in tests).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from filelock import FileLock

from _workspace import get_workspace

WORKSPACE = get_workspace()
CACHE_DIR = WORKSPACE / "tfidf_cache"
CACHE_LOCK_PATH = WORKSPACE / ".tfidf_cache.lock"


def load_corpus() -> list[dict]:
    corpus_path = WORKSPACE / "corpus.json"
    if not corpus_path.exists():
        print("Error: workspace not initialized. Run init.py first.", file=sys.stderr)
        sys.exit(1)
    with open(corpus_path, encoding="utf-8") as f:
        return json.load(f)


def get_or_build_tfidf(corpus: list[dict]):
    """Load cached TF-IDF matrix or build and cache it.

    The build path is wrapped in a dedicated FileLock. A double-checked
    validity test inside the lock means a contender that loses the race
    picks up the freshly-built artifacts instead of rebuilding.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        print("Error: scikit-learn required. Install with: uv add scikit-learn", file=sys.stderr)
        sys.exit(1)

    vectorizer_path = CACHE_DIR / "vectorizer.pkl"
    matrix_path = CACHE_DIR / "matrix.pkl"
    ids_path = CACHE_DIR / "ids.json"
    corpus_path = WORKSPACE / "corpus.json"

    def cache_valid() -> bool:
        return (
            vectorizer_path.exists()
            and matrix_path.exists()
            and ids_path.exists()
            and corpus_path.exists()
            and vectorizer_path.stat().st_mtime >= corpus_path.stat().st_mtime
        )

    def load_cache():
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        with open(matrix_path, "rb") as f:
            matrix = pickle.load(f)
        with open(ids_path, encoding="utf-8") as f:
            ids = json.load(f)
        return vectorizer, matrix, ids

    # Fast path — no lock needed; the cache files were written atomically by
    # whoever built them, under the lock below.
    if cache_valid():
        return load_cache()

    # Slow path — acquire the lock, re-check, then build.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(CACHE_LOCK_PATH))
    with lock:
        if cache_valid():
            return load_cache()

        texts = [r["text"] for r in corpus]
        ids = [r["id"] for r in corpus]
        vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
        matrix = vectorizer.fit_transform(texts)

        with open(vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)
        with open(matrix_path, "wb") as f:
            pickle.dump(matrix, f)
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump(ids, f, ensure_ascii=False)

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


def main() -> int:
    parser = argparse.ArgumentParser(description="TF-IDF similarity search")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--n", type=int, default=10, help="Number of results")
    args = parser.parse_args()

    results = search(args.query, args.n)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
