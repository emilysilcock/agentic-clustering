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
import random
import sys
import uuid
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

from _audit_metrics import MIN_AUDIT_N_PER_CLUSTER
from _log import append_log
from _workspace import get_workspace

WORKSPACE = get_workspace()
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
    """Append a sample event to log.jsonl. Thin wrapper around _log.append_log."""
    append_log(WORKSPACE / "log.jsonl", "sample", detail)


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


def sample_diverse(corpus: list[dict], n: int, include_seen: bool) -> list[dict]:
    """Sample a maximally diverse set via TF-IDF farthest-point (max-min) sampling.

    Greedy farthest-first traversal (k-center) over TF-IDF vectors: seed one
    text, then repeatedly add the candidate whose minimum cosine distance to the
    already-selected set is largest. This spreads the draw across the corpus
    instead of over-representing dense regions the way a uniform random draw
    does, so a downstream consumer (e.g. an auditor stress-testing a taxonomy)
    sees the corpus's breadth — including sparse/edge regions a random sample
    tends to miss.

    Same dependency footprint as the ``targeted`` strategy (scikit-learn).
    Reproducible via ``--seed``: the seed only picks the starting text, which
    then fully determines the deterministic traversal.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("Error: scikit-learn required for diverse sampling. Install with: uv add scikit-learn", file=sys.stderr)
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

    n = min(n, len(candidates))
    if n >= len(candidates):
        # Asking for everything — the diversity ordering is moot, return all.
        return list(candidates)

    texts = [r["text"] for r in candidates]
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    tfidf = vectorizer.fit_transform(texts)  # L2-normalized sparse rows
    n_cand = tfidf.shape[0]

    # Farthest-first traversal. ``min_dist[i]`` = cosine distance from candidate
    # i to the nearest already-selected point; start at +inf so the first pick
    # is the seeded random start, then every subsequent pick maximizes the
    # minimum distance to the selected set.
    min_dist = np.full(n_cand, np.inf)
    start = random.randrange(n_cand)
    selected = [start]
    while len(selected) < n:
        sims = cosine_similarity(tfidf[selected[-1]], tfidf).ravel()
        np.minimum(min_dist, 1.0 - sims, out=min_dist)
        min_dist[selected] = -np.inf  # never reselect an already-picked point
        selected.append(int(np.argmax(min_dist)))

    return [candidates[i] for i in selected]


def write_strata_manifest(
    strata: dict[str, list[str]],
    *,
    per_cluster: int,
    oversample: int,
    seed_from: str,
    seed: int,
) -> Path:
    """Persist which text ids were aimed at which cluster.

    Without this the draw's intent is lost the moment the texts reach the
    auditor, and a cluster that was searched for and not found reads exactly
    like one that was never searched for. `state.py update-from-audit` reads
    the manifest to record ``evidence.audit_targeted``.

    Written under ``strata/`` so `finalize` sweeps it into ``archive/`` with
    the other intermediate artifacts.
    """
    strata_dir = WORKSPACE / "strata"
    strata_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = strata_dir / f"strata_{stamp}_{uuid.uuid4().hex[:8]}.json"

    # Pin the manifest to the cluster set it was drawn against, the same way
    # audits are pinned. A draw aimed at c37-as-it-was-then says nothing about
    # c37 after a merge or split, and `metrics.py` filters on this.
    cluster_version = None
    state_path = WORKSPACE / "state.json"
    if state_path.exists():
        with open(state_path, encoding="utf-8") as f:
            cluster_version = json.load(f).get("meta", {}).get("cluster_version")

    path.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "cluster_definitions_version": cluster_version,
                "per_cluster": per_cluster,
                "oversample": oversample,
                "seed_from": seed_from,
                "seed": seed,
                "strata": strata,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def assigned_text_ids_by_cluster() -> dict[str, list[str]]:
    """Text IDs the auditor assigned to each cluster, from current-version audits.

    Stale audits (written against an older ``cluster_definitions_version``) are
    skipped, so this reflects the current cluster definitions only.
    """
    audit_dir = WORKSPACE / "audits"
    if not audit_dir.exists():
        return {}
    current_version = None
    state_path = WORKSPACE / "state.json"
    if state_path.exists():
        with open(state_path, encoding="utf-8") as f:
            current_version = json.load(f).get("meta", {}).get("cluster_version")

    out: dict[str, list[str]] = {}
    for audit_file in sorted(audit_dir.glob("*.json")):
        with open(audit_file, encoding="utf-8") as f:
            audit = json.load(f)
        if current_version is not None and audit.get("cluster_definitions_version") != current_version:
            continue
        for a in audit.get("assignments", []):
            cid, tid = a.get("cluster_id"), a.get("text_id")
            if cid and tid and tid not in out.setdefault(cid, []):
                out[cid].append(tid)
    return out


# Multiplier on a cluster's shortfall when deciding how many texts to aim at
# it. A stratified draw is a hypothesis, and the auditor reassigns a share of
# it elsewhere — measured at 62-74% retention on a corpus of deliberately
# confusable role-pairs ("imposes sanctions" vs "bears sanctions"), and as low
# as 19% for the worst single cluster there. Aiming exactly the shortfall would
# therefore undershoot and cost an extra pass, so we aim at roughly double it,
# capped by --per-cluster.
RETENTION_MARGIN = 2


def sample_stratified(
    corpus: list[dict],
    per_cluster: int,
    include_seen: bool,
    oversample: int,
    total_cap: int | None,
    target_n: int,
    all_clusters: bool,
    seed_from: str,
) -> tuple[list[dict], int, dict[str, list[str]]]:
    """Draw a per-cluster floor of texts so thin clusters get a defensible n.

    Returns ``(texts, effective_per_cluster, strata)`` where ``strata`` maps
    each cluster id to the text ids aimed at it. The caller persists that
    mapping so ``update-from-audit`` can tell "0 assigned of 8 aimed here"
    (a finding about the cluster) from "0 assigned of 0 aimed here" (a fact
    about the sample).

    A uniform random draw allocates sample in proportion to corpus prevalence,
    which is the opposite of what per-cluster measurement needs: the rare
    clusters are exactly the ones whose fit is least certain and they get the
    fewest texts. On a 65-cluster taxonomy a 300-text uniform audit averages
    4.6 assignments per cluster, with the tail drawing 0-2.

    This strategy instead serves only the clusters that are short, thinnest
    first, and for each pulls unseen texts that plausibly belong to it.

    Design points, each measured rather than assumed:

    * **Skip clusters already at ``target_n``.** Drawing for every cluster
      every pass spends most of the budget on clusters that need nothing — on
      the role-pair corpus, the third pass put 64 of 80 texts into clusters
      already sitting above n=39. Serving only the deficit makes the cost
      scale with the thin tail, which shrinks each pass, rather than with k.
      ``--all-clusters`` restores the draw-for-everything behaviour, which is
      what you want after a definition rewrite has made old evidence stale.
    * **Aim ~2x the shortfall** (RETENTION_MARGIN), capped at
      ``--per-cluster``, because a share of every stratum is reassigned
      elsewhere by the auditor.
    * **Oversample, then draw at random inside the pool.** Taking the literal
      top-k by similarity would hand the auditor the easiest possible cases —
      texts that echo the description's own vocabulary — and inflate
      per-cluster confidence. We take the top ``k * oversample`` and
      random-sample within it. Reproducible via ``--seed``.
    * **This is a hypothesis, not an assignment.** A text aimed at c12 is one
      the corpus vocabulary suggests *might* be c12; the auditor decides
      independently and may put it anywhere, or nowhere. So the strategy
      raises the expected per-cluster n substantially but cannot guarantee a
      floor.

    ``seed_from`` chooses what the similarity query is built from:

    * ``description`` (default) — the cluster's name + description.
    * ``assigned`` — the centroid of the texts the auditor has already assigned
      to this cluster, falling back to the description for clusters with none.
      Uses the auditor's own past judgments instead of the description's
      wording. Needs a prior audit, and inherits that audit's mistakes.

    ``assigned`` was added on the theory that it would retrieve confusable
    rare clusters better than their descriptions do. On the one synthetic
    corpus it has been measured against — five pairs sharing all their
    vocabulary and differing only by role, at 240-vs-14 prevalence — it did
    not: both reached the per-cluster floor on every cluster, but
    ``description`` got there in 80 drawn texts over 4 passes and ``assigned``
    took 148 over 8. The likely reason is that a description names the
    distinguishing rare term once and TF-IDF weights it heavily, while
    averaging several texts washes that term out against the shared
    vocabulary. On a separable 20-cluster corpus the two were
    indistinguishable (110 vs 112 texts). So ``assigned`` is kept as an option
    worth trying on a corpus where descriptions are vague or boilerplate, not
    as a recommended default — and one synthetic is not enough to conclude it
    never helps.

    The resulting draw is NOT a prevalence sample. Audits built on it must
    declare ``"sample_basis": "stratified"`` so coverage stays sourced from
    uniform audits (see _audit_metrics.SAMPLE_BASES).
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("Error: scikit-learn required for stratified sampling. Install with: uv add scikit-learn", file=sys.stderr)
        sys.exit(1)

    import numpy as np

    state_path = WORKSPACE / "state.json"
    if not state_path.exists():
        print("Error: workspace not initialized. Run init.py first.", file=sys.stderr)
        sys.exit(1)
    with open(state_path, encoding="utf-8") as f:
        state = json.load(f)
    clusters = state.get("clusters", [])
    if not clusters:
        print(
            "Error: stratified sampling needs an existing cluster set — there "
            "is nothing to stratify by. Use --strategy random until the "
            "synthesizer has produced clusters.",
            file=sys.stderr,
        )
        sys.exit(1)

    if include_seen:
        candidates = corpus
    else:
        seen = load_seen_ids()
        candidates = [r for r in corpus if r["id"] not in seen]

    if not candidates:
        print("Warning: no unseen texts remaining", file=sys.stderr)
        return [], per_cluster, {}

    def _current_n(c: dict) -> int:
        return (c.get("evidence", {}) or {}).get("audit_assignments", 0) or 0

    # Only serve clusters that are actually short. This is the difference
    # between a cost that scales with k and one that scales with the thin tail.
    if all_clusters:
        wanted = [(c, per_cluster) for c in clusters]
    else:
        wanted = []
        for c in clusters:
            deficit = target_n - _current_n(c)
            if deficit > 0:
                wanted.append((c, min(per_cluster, deficit * RETENTION_MARGIN)))
        if not wanted:
            print(
                f"Nothing to do: every cluster already has at least {target_n} "
                f"audit assignments. Use --all-clusters to draw anyway (e.g. "
                f"after rewriting descriptions), or --target N to raise the bar.",
                file=sys.stderr,
            )
            return [], per_cluster, {}
        skipped = len(clusters) - len(wanted)
        if skipped:
            print(
                f"Serving {len(wanted)} of {len(clusters)} clusters "
                f"({skipped} already at n>={target_n}; --all-clusters to "
                f"include them)",
                file=sys.stderr,
            )

    # A total budget scales each cluster's ask down rather than truncating the
    # last clusters to nothing — a shallow draw across every short cluster is
    # more useful than a deep one across the first few.
    if total_cap is not None:
        affordable = max(1, total_cap // len(wanted))
        if affordable < per_cluster:
            print(
                f"Note: capping --per-cluster from {per_cluster} to {affordable} "
                f"({total_cap} total budget / {len(wanted)} clusters served)",
                file=sys.stderr,
            )
            per_cluster = affordable
            wanted = [(c, min(k, affordable)) for c, k in wanted]

    texts = [r["text"] for r in candidates]
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    tfidf = vectorizer.fit_transform(texts)

    # Thinnest cluster first, so it gets first refusal on the texts that look
    # like it before a better-served neighbour can claim them.
    wanted.sort(key=lambda t: (_current_n(t[0]), t[0].get("id", "")))

    prior = assigned_text_ids_by_cluster() if seed_from == "assigned" else {}
    corpus_text_by_id = {r["id"]: r["text"] for r in corpus}

    def _query_vector(cluster: dict):
        """One L2-normalized row to rank candidates against."""
        if seed_from == "assigned":
            prior_texts = [
                corpus_text_by_id[t]
                for t in prior.get(cluster["id"], [])
                if t in corpus_text_by_id
            ]
            if prior_texts:
                rows = vectorizer.transform(prior_texts)
                centroid = np.asarray(rows.mean(axis=0))
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    return centroid / norm
                # All-zero centroid (no in-vocabulary terms) — fall through.
        return vectorizer.transform(
            [f"{cluster.get('name', '')}. {cluster.get('description', '')}"]
        )

    taken: set[int] = set()
    results: list[dict] = []
    strata: dict[str, list[str]] = {}
    fell_back = 0

    for cluster, ask in wanted:
        cid = cluster.get("id", "?")
        if seed_from == "assigned" and not prior.get(cid):
            fell_back += 1
        qvec = _query_vector(cluster)
        sims = cosine_similarity(qvec, tfidf).ravel()
        # Descending similarity, dropping zero-similarity candidates (no shared
        # vocabulary at all) and anything another cluster already claimed.
        ranked = [i for i in np.argsort(sims)[::-1] if sims[i] > 0 and i not in taken]
        pool = ranked[: max(ask, ask * max(1, oversample))]
        if not pool:
            strata[cid] = []
            continue
        k = min(ask, len(pool))
        picked = random.sample(pool, k) if len(pool) > k else pool
        for i in picked:
            taken.add(i)
            results.append(candidates[i])
        strata[cid] = [candidates[i]["id"] for i in picked]

    asked = {c.get("id", "?"): ask for c, ask in wanted}
    short = [
        f"{cid}:{len(tids)}/{asked[cid]}"
        for cid, tids in strata.items()
        if len(tids) < asked[cid]
    ]
    print(
        f"Stratified draw ({seed_from}-seeded): {len(results)} texts over "
        f"{len(wanted)} clusters (max {per_cluster} each, oversample "
        f"x{max(1, oversample)})",
        file=sys.stderr,
    )
    if fell_back:
        print(
            f"  {fell_back} cluster(s) had no prior assignments; used their "
            f"description instead",
            file=sys.stderr,
        )
    if short:
        print(
            f"  Under ask (corpus had too few unclaimed similar texts): "
            f"{', '.join(short)}",
            file=sys.stderr,
        )
    return results, per_cluster, strata


def sample_cluster(corpus: list[dict], n: int, cluster_id: str) -> list[dict]:
    """Sample texts that were assigned to a specific cluster in recent audits.

    Note: ``--include-seen`` is intentionally not honored here. Cluster-strategy
    pulls from audit assignments, and audited texts are always already in
    ``seen_ids`` (audits mark-seen). Honoring the flag would always return zero
    when unset; the use case is "show me examples assigned to cluster X", which
    fundamentally requires seen texts. The CLI help notes this asymmetry.
    """
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
    parser.add_argument("--n", type=int, default=None,
                        help="Number of texts to sample (default 50). With --strategy stratified "
                             "this is a total budget instead: --per-cluster is scaled down to "
                             "fit it (pass no --n to let the per-cluster floor decide).")
    parser.add_argument("--strategy", default="random",
                        choices=["random", "targeted", "diverse", "stratified", "cluster"],
                        help="Sampling strategy. random=uniform draw; targeted=TF-IDF "
                             "similarity to --query; diverse=TF-IDF farthest-point (max-min) "
                             "spread across the corpus; stratified=a --per-cluster floor of "
                             "plausible texts for every current cluster, thinnest cluster "
                             "first (for per-cluster audit power; NOT a prevalence sample); "
                             "cluster=texts assigned to --cluster-id in recent audits")
    parser.add_argument("--per-cluster", type=int, default=MIN_AUDIT_N_PER_CLUSTER,
                        help=f"Stratified strategy: maximum texts to aim at any one "
                             f"cluster (default {MIN_AUDIT_N_PER_CLUSTER}, matching the "
                             f"per-cluster audit floor that state.py finalize enforces). "
                             f"A cluster that only needs a couple more gets a couple more, "
                             f"not the full amount.")
    parser.add_argument("--target", type=int, default=MIN_AUDIT_N_PER_CLUSTER,
                        help=f"Stratified strategy: clusters with at least this many audit "
                             f"assignments already are skipped entirely (default "
                             f"{MIN_AUDIT_N_PER_CLUSTER}). Keeps the budget on the thin "
                             f"tail instead of re-serving clusters that need nothing.")
    parser.add_argument("--all-clusters", action="store_true",
                        help="Stratified strategy: draw for every cluster, including ones "
                             "already above --target. Use after rewriting descriptions, "
                             "when existing per-cluster evidence no longer reflects the "
                             "definitions it was collected under.")
    parser.add_argument("--seed-from", default="description",
                        choices=["description", "assigned"],
                        help="Stratified strategy: build each cluster's similarity query "
                             "from its name+description (default), or from the centroid of "
                             "texts the auditor already assigned to it ('assigned' — needs "
                             "a prior audit, falls back to the description per cluster). "
                             "'assigned' did NOT beat the default on the corpora it has "
                             "been measured against (it needed ~2x the draws to converge "
                             "on a confusable one); it is worth trying where descriptions "
                             "are vague or boilerplate, not as a default.")
    parser.add_argument("--oversample", type=int, default=4,
                        help="Stratified strategy: draw each cluster's texts at random from "
                             "its top (per-cluster x oversample) most similar candidates "
                             "rather than taking the literal top-n, so the auditor sees "
                             "merely-plausible cases and not just the obvious ones "
                             "(default 4).")
    parser.add_argument("--query", help="Query string for targeted sampling")
    parser.add_argument("--cluster-id", help="Cluster ID for cluster-based sampling")
    parser.add_argument("--ids", nargs="+", help="Specific text IDs to fetch")
    parser.add_argument("--include-seen", action="store_true",
                        help="Include previously sampled texts (default: exclude seen). "
                             "Only applies to --strategy random/targeted; the cluster strategy "
                             "intentionally ignores this flag because audited texts are seen by "
                             "definition.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for random sampling. If omitted, an auto-seed is generated "
                             "and recorded in log.jsonl so the sample is reproducible after the fact.")
    args = parser.parse_args()

    # `--n` defaults to None so the stratified strategy can tell "no budget
    # given, let --per-cluster decide" apart from an explicit total. Every
    # other strategy keeps its long-standing default of 50.
    explicit_n = args.n
    if args.n is None:
        args.n = 50

    # Pick a seed (user-provided or auto-generated) and apply it. Auto-generating
    # rather than leaving system entropy means every sample is traceable via
    # log.jsonl even when the caller forgot --seed.
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    random.seed(seed)

    # Enforce max_texts_per_sample cap from config if set
    cap = None
    state_path = WORKSPACE / "state.json"
    if state_path.exists():
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
        cap = state.get("config", {}).get("max_texts_per_sample")
        # Stratified ignores args.n (it draws per_cluster x clusters), so
        # capping it there would only print a note about a number nothing uses.
        if cap is not None and args.n > cap and args.strategy != "stratified":
            print(f"Note: capping --n from {args.n} to {cap} (max_texts_per_sample config)", file=sys.stderr)
            args.n = cap

    # Stratified reads --n as a total budget, not a draw size, so the config
    # cap and an explicit --n combine as the tighter of the two.
    budgets = [b for b in (explicit_n, cap) if b is not None]
    total_cap = min(budgets) if budgets else None

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
        elif args.strategy == "diverse":
            results = sample_diverse(corpus, args.n, args.include_seen)
        elif args.strategy == "stratified":
            # Reassign args.per_cluster to the effective maximum so the
            # log.jsonl entry records what was drawn, not what was asked for.
            results, args.per_cluster, strata = sample_stratified(
                corpus,
                args.per_cluster,
                args.include_seen,
                args.oversample,
                total_cap,
                args.target,
                args.all_clusters,
                args.seed_from,
            )
            if strata:
                strata_path = write_strata_manifest(
                    strata,
                    per_cluster=args.per_cluster,
                    oversample=args.oversample,
                    seed_from=args.seed_from,
                    seed=seed,
                )
                # The auditor copies this path into its audit file's
                # `strata_file` field; update-from-audit uses it to separate
                # "nothing was aimed here" from "things were aimed here and
                # the auditor put them elsewhere".
                print(f"Strata manifest: {strata_path}", file=sys.stderr)
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
        # Stratified has no single requested draw size — it's per_cluster x
        # clusters, logged as its own fields below.
        if args.ids:
            n_requested = len(args.ids)
        elif args.strategy == "stratified":
            n_requested = "derived"
        else:
            n_requested = args.n
        log_detail_parts = [
            f"strategy={'ids' if args.ids else args.strategy}",
            f"n_requested={n_requested}",
            f"n_returned={len(results)}",
            f"seed={seed}",
            f"include_seen={args.include_seen}",
        ]
        if args.strategy == "targeted" and args.query:
            log_detail_parts.append(f"query={args.query!r}")
        if args.strategy == "stratified":
            log_detail_parts.append(f"per_cluster={args.per_cluster}")
            log_detail_parts.append(f"target={args.target}")
            log_detail_parts.append(f"all_clusters={args.all_clusters}")
            log_detail_parts.append(f"seed_from={args.seed_from}")
            log_detail_parts.append(f"oversample={args.oversample}")
            log_detail_parts.append(f"total_budget={total_cap}")
        if args.strategy == "cluster" and args.cluster_id:
            log_detail_parts.append(f"cluster_id={args.cluster_id}")
        log_sample(" ".join(log_detail_parts))

    # Output results as JSON to stdout
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
