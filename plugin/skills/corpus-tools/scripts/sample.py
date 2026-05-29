#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["scikit-learn", "filelock"]
# ///
"""Sample texts from the corpus.

Supports random, targeted (TF-IDF), cluster-based, and ID-based sampling.
Uses file locking for atomic seen-ID tracking under concurrent access.
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 on stdout/stderr — Windows defaults to cp1252 and crashes on
# non-ASCII cluster names / corpus content. Idempotent; no-op on streams that
# aren't TextIOWrapper (e.g. captured in tests).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from filelock import FileLock


def _get_workspace() -> Path:
    env_ws = os.environ.get("CLUSTERING_WORKSPACE")
    if env_ws:
        return Path(env_ws)
    # CLUSTERING_WORKSPACE does not survive across Bash tool calls or reach hook
    # subprocesses, so fall back to the pointer init.py writes at a fixed,
    # project-root-relative location (hooks and tool calls share that cwd).
    pointer = Path(".claude/clustering/.active_workspace")
    if pointer.exists():
        ws = pointer.read_text(encoding="utf-8").strip()
        if ws:
            return Path(ws)
    return Path(".claude/clustering")


WORKSPACE = _get_workspace()
LOCK_PATH = WORKSPACE / ".state.lock"


def load_corpus() -> list[dict]:
    corpus_path = WORKSPACE / "corpus.json"
    if not corpus_path.exists():
        print("Error: workspace not initialized. Run init.py first.", file=sys.stderr)
        sys.exit(1)
    with open(corpus_path, encoding="utf-8") as f:
        return json.load(f)


def load_seen_ids() -> set:
    seen_path = WORKSPACE / "seen_ids.json"
    if not seen_path.exists():
        return set()
    with open(seen_path, encoding="utf-8") as f:
        return set(json.load(f))


def save_seen_ids(seen: set):
    seen_path = WORKSPACE / "seen_ids.json"
    with open(seen_path, "w", encoding="utf-8") as f:
        json.dump(sorted(seen), f)


def update_sampled_count(n: int):
    """Update total_texts_sampled in state.json."""
    state_path = WORKSPACE / "state.json"
    with open(state_path, encoding="utf-8") as f:
        state = json.load(f)
    state["meta"]["total_texts_sampled"] = state["meta"].get("total_texts_sampled", 0) + n
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def log_sample(detail: str):
    """Append a sample event to log.jsonl (same shape as init.py / state.py)."""
    log_path = WORKSPACE / "log.jsonl"
    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "action": "sample",
        "detail": detail,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def sample_random(corpus: list[dict], n: int, include_seen: bool) -> list[dict]:
    if include_seen:
        candidates = corpus
    else:
        # Default: exclude seen texts to maximize corpus coverage
        seen = load_seen_ids()
        candidates = [r for r in corpus if r["id"] not in seen]

    if not candidates:
        print("Warning: no unseen texts remaining", file=sys.stderr)
        return []

    n = min(n, len(candidates))
    return random.sample(candidates, n)


def sample_targeted(corpus: list[dict], n: int, query: str, include_seen: bool) -> list[dict]:
    """Sample texts similar to a query using TF-IDF."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("Error: scikit-learn required for targeted sampling. Install with: uv add scikit-learn", file=sys.stderr)
        sys.exit(1)

    import numpy as np

    if include_seen:
        candidates = corpus
    else:
        seen = load_seen_ids()
        candidates = [r for r in corpus if r["id"] not in seen]

    if not candidates:
        print("Warning: no unseen texts remaining", file=sys.stderr)
        return []

    texts = [r["text"] for r in candidates]
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts + [query])

    query_vec = tfidf_matrix[-1]
    corpus_matrix = tfidf_matrix[:-1]
    similarities = cosine_similarity(query_vec, corpus_matrix).flatten()

    top_indices = np.argsort(similarities)[::-1][:n]
    return [candidates[i] for i in top_indices if similarities[i] > 0]


def sample_cluster(corpus: list[dict], n: int, cluster_id: str) -> list[dict]:
    """Sample texts that were assigned to a specific cluster in recent audits."""
    audit_dir = WORKSPACE / "audits"
    if not audit_dir.exists():
        print("Error: no audits found", file=sys.stderr)
        sys.exit(1)

    # Read current cluster version to filter stale audits
    state_path = WORKSPACE / "state.json"
    current_version = None
    if state_path.exists():
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
        current_version = state.get("meta", {}).get("cluster_version")

    # Collect text IDs assigned to this cluster, only from current-version audits
    cluster_text_ids = set()
    for audit_file in sorted(audit_dir.glob("*.json")):
        with open(audit_file, encoding="utf-8") as f:
            audit = json.load(f)
        if current_version is not None and audit.get("cluster_definitions_version") != current_version:
            continue
        for assignment in audit.get("assignments", []):
            if assignment.get("cluster_id") == cluster_id:
                cluster_text_ids.add(assignment["text_id"])

    if not cluster_text_ids:
        print(f"Warning: no texts found for cluster {cluster_id} in audits", file=sys.stderr)
        return []

    # Find the actual texts
    id_to_text = {r["id"]: r for r in corpus}
    results = [id_to_text[tid] for tid in cluster_text_ids if tid in id_to_text]

    n = min(n, len(results))
    return random.sample(results, n) if len(results) > n else results


def sample_by_ids(corpus: list[dict], ids: list[str]) -> list[dict]:
    """Fetch specific texts by ID."""
    id_to_text = {r["id"]: r for r in corpus}
    results = []
    for tid in ids:
        if tid in id_to_text:
            results.append(id_to_text[tid])
        else:
            print(f"Warning: text ID '{tid}' not found in corpus", file=sys.stderr)
    return results


def main():
    parser = argparse.ArgumentParser(description="Sample texts from corpus")
    parser.add_argument("--n", type=int, default=50, help="Number of texts to sample")
    parser.add_argument("--strategy", default="random", choices=["random", "targeted", "cluster"],
                        help="Sampling strategy")
    parser.add_argument("--query", help="Query string for targeted sampling")
    parser.add_argument("--cluster-id", help="Cluster ID for cluster-based sampling")
    parser.add_argument("--ids", nargs="+", help="Specific text IDs to fetch")
    parser.add_argument("--include-seen", action="store_true",
                        help="Include previously sampled texts (default: exclude seen)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for random sampling. If omitted, an auto-seed is generated "
                             "and recorded in log.jsonl so the sample is reproducible after the fact.")
    args = parser.parse_args()

    # Pick a seed (user-provided or auto-generated) and apply it. Auto-generating
    # rather than leaving system entropy means every sample is traceable via
    # log.jsonl even when the caller forgot --seed.
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    random.seed(seed)

    # Enforce max_texts_per_sample cap from config if set
    state_path = WORKSPACE / "state.json"
    if state_path.exists():
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
        cap = state.get("config", {}).get("max_texts_per_sample")
        if cap is not None and args.n > cap:
            print(f"Note: capping --n from {args.n} to {cap} (max_texts_per_sample config)", file=sys.stderr)
            args.n = cap

    lock = FileLock(str(LOCK_PATH))

    with lock:
        corpus = load_corpus()

        if args.ids:
            results = sample_by_ids(corpus, args.ids)
        elif args.strategy == "targeted":
            if not args.query:
                print("Error: --query required for targeted strategy", file=sys.stderr)
                sys.exit(1)
            results = sample_targeted(corpus, args.n, args.query, args.include_seen)
        elif args.strategy == "cluster":
            if not args.cluster_id:
                print("Error: --cluster-id required for cluster strategy", file=sys.stderr)
                sys.exit(1)
            results = sample_cluster(corpus, args.n, args.cluster_id)
        else:
            results = sample_random(corpus, args.n, args.include_seen)

        # Mark sampled IDs as seen (unless fetching by specific IDs)
        if results and not args.ids:
            seen = load_seen_ids()
            new_ids = {r["id"] for r in results}
            seen.update(new_ids)
            save_seen_ids(seen)
            update_sampled_count(len(results))

        # Record the sample for reproducibility. ID lookups and targeted (TF-IDF
        # argsort) are deterministic, but log them anyway so the trail is uniform.
        log_detail_parts = [
            f"strategy={'ids' if args.ids else args.strategy}",
            f"n_requested={args.n if not args.ids else len(args.ids)}",
            f"n_returned={len(results)}",
            f"seed={seed}",
            f"include_seen={args.include_seen}",
        ]
        if args.strategy == "targeted" and args.query:
            log_detail_parts.append(f"query={args.query!r}")
        if args.strategy == "cluster" and args.cluster_id:
            log_detail_parts.append(f"cluster_id={args.cluster_id}")
        log_sample(" ".join(log_detail_parts))

    # Output results as JSON to stdout
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
