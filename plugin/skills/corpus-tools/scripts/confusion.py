#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["scikit-learn"]
# ///
"""Cross-proposal agreement analysis.

Compares clustering proposals pairwise using ARI, NMI, per-cluster entropy,
and element-centric similarity. Writes a JSON report and prints a
human-readable summary for the orchestrator.
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path


def _get_workspace() -> Path:
    env_ws = os.environ.get("CLUSTERING_WORKSPACE")
    if env_ws:
        return Path(env_ws)
    return Path(".claude/clustering")


WORKSPACE = _get_workspace()
PROPOSALS_DIR = WORKSPACE / "proposals"
METRICS_DIR = WORKSPACE / "metrics"


# ---------------------------------------------------------------------------
# Core algorithms (ported from llm_clustering.modules.refine.confusion)
# ---------------------------------------------------------------------------

def _shannon_entropy(counts: dict[str, int]) -> float:
    """Compute Shannon entropy of a distribution given as counts."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def _build_text_to_cluster(proposal: dict) -> dict[str, str]:
    """Build text_id -> cluster_name mapping from a proposal."""
    mapping = {}
    for cluster in proposal.get("clusters", []):
        name = cluster["name"]
        for text_id in cluster.get("text_ids", []):
            mapping[text_id] = name
    return mapping


def _compute_agreement_rate(
    contingency: dict[str, dict[str, int]],
    n_common: int,
) -> float:
    """Compute agreement rate using majority mapping from contingency table."""
    if n_common == 0:
        return 0.0
    agreed = 0
    for dist in contingency.values():
        if dist:
            agreed += max(dist.values())
    return agreed / n_common


def _compute_clustering_metrics(
    anchor_map: dict[str, str],
    other_map: dict[str, str],
    common_texts: set[str],
) -> tuple[float, float]:
    """Compute ARI and NMI from two label assignments over common texts."""
    if not common_texts:
        return 0.0, 0.0

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    sorted_texts = sorted(common_texts)
    labels_anchor = [anchor_map[t] for t in sorted_texts]
    labels_other = [other_map[t] for t in sorted_texts]

    ari = adjusted_rand_score(labels_anchor, labels_other)
    nmi = normalized_mutual_info_score(labels_anchor, labels_other)
    return float(ari), float(nmi)


def _compute_pairwise(
    anchor_map: dict[str, str],
    other_map: dict[str, str],
    top_n: int = 5,
) -> dict:
    """Compute pairwise confusion metrics between two proposals."""
    common_texts = set(anchor_map.keys()) & set(other_map.keys())

    if not common_texts:
        return {
            "common_texts": 0,
            "ari": 0.0,
            "nmi": 0.0,
            "agreement_rate": 0.0,
            "high_entropy_clusters": [],
        }

    # Build contingency table: anchor_cluster -> {other_cluster -> count}
    contingency: dict[str, dict[str, int]] = {}
    for text_id in common_texts:
        a = anchor_map[text_id]
        o = other_map[text_id]
        contingency.setdefault(a, {})
        contingency[a][o] = contingency[a].get(o, 0) + 1

    # Row entropy (anchor side)
    anchor_entropies = []
    for a_cluster, dist in contingency.items():
        entropy = _shannon_entropy(dist)
        if entropy > 0:
            anchor_entropies.append({
                "name": a_cluster,
                "side": "anchor",
                "entropy": round(entropy, 2),
                "n_texts": sum(dist.values()),
                "distribution": dict(sorted(dist.items(), key=lambda x: x[1], reverse=True)),
            })

    # Column entropy (other side) — reverse contingency
    reverse: dict[str, dict[str, int]] = {}
    for a_cluster, dist in contingency.items():
        for o_cluster, count in dist.items():
            reverse.setdefault(o_cluster, {})
            reverse[o_cluster][a_cluster] = reverse[o_cluster].get(a_cluster, 0) + count

    other_entropies = []
    for o_cluster, dist in reverse.items():
        entropy = _shannon_entropy(dist)
        if entropy > 0:
            other_entropies.append({
                "name": o_cluster,
                "side": "other",
                "entropy": round(entropy, 2),
                "n_texts": sum(dist.values()),
                "distribution": dict(sorted(dist.items(), key=lambda x: x[1], reverse=True)),
            })

    # Combine and take top N by entropy
    all_entropies = anchor_entropies + other_entropies
    all_entropies.sort(key=lambda x: x["entropy"], reverse=True)
    high_entropy = all_entropies[:top_n]

    ari, nmi = _compute_clustering_metrics(anchor_map, other_map, common_texts)
    agreement = _compute_agreement_rate(contingency, len(common_texts))

    return {
        "common_texts": len(common_texts),
        "ari": round(ari, 3),
        "nmi": round(nmi, 3),
        "agreement_rate": round(agreement, 3),
        "high_entropy_clusters": high_entropy,
    }


def _compute_element_similarity(
    proposal_maps: dict[str, dict[str, str]],
    corpus_lookup: dict[str, str] | None = None,
    max_inconsistent: int = 10,
) -> dict:
    """Compute element-centric similarity using co-membership agreement.

    For each text, check how consistently it is co-clustered with other texts
    across all proposal pairs.
    """
    prop_ids = list(proposal_maps.keys())
    if len(prop_ids) < 2:
        return {"overall": 1.0, "n_texts_compared": 0, "inconsistent_texts": []}

    # Find texts appearing in 2+ proposals
    text_proposals: dict[str, list[str]] = {}
    for pid, mapping in proposal_maps.items():
        for text_id in mapping:
            text_proposals.setdefault(text_id, []).append(pid)

    # Only consider texts in 2+ proposals
    eligible = {t: pids for t, pids in text_proposals.items() if len(pids) >= 2}

    if not eligible:
        return {"overall": 1.0, "n_texts_compared": 0, "inconsistent_texts": []}

    eligible_ids = sorted(eligible.keys())

    # Build per-proposal cluster lists for eligible texts
    # For each (text, proposal) pair, store the cluster name
    text_cluster: dict[str, dict[str, str]] = {}
    for text_id in eligible_ids:
        text_cluster[text_id] = {}
        for pid in eligible[text_id]:
            text_cluster[text_id][pid] = proposal_maps[pid][text_id]

    # Co-membership consistency per text
    per_text_sim: dict[str, float] = {}

    for t in eligible_ids:
        t_pids = eligible[t]
        agreements = 0
        total = 0

        for other_t in eligible_ids:
            if other_t == t:
                continue
            # Only compare proposal pairs where both texts appear
            other_pids = eligible[other_t]
            shared_pids = [p for p in t_pids if p in other_pids]
            if len(shared_pids) < 2:
                continue

            for i in range(len(shared_pids)):
                for j in range(i + 1, len(shared_pids)):
                    pi, pj = shared_pids[i], shared_pids[j]
                    same_i = text_cluster[t][pi] == text_cluster[other_t][pi]
                    same_j = text_cluster[t][pj] == text_cluster[other_t][pj]
                    if same_i == same_j:
                        agreements += 1
                    total += 1

        per_text_sim[t] = agreements / total if total > 0 else 1.0

    overall = sum(per_text_sim.values()) / len(per_text_sim) if per_text_sim else 1.0

    # Find most inconsistent texts
    sorted_texts = sorted(per_text_sim.items(), key=lambda x: x[1])
    inconsistent = []
    for text_id, sim_score in sorted_texts[:max_inconsistent]:
        entry: dict = {
            "text_id": text_id,
            "similarity": round(sim_score, 3),
            "assignments": text_cluster[text_id],
        }
        if corpus_lookup and text_id in corpus_lookup:
            preview = corpus_lookup[text_id]
            if len(preview) > 300:
                preview = preview[:300] + "..."
            entry["text_preview"] = preview
        inconsistent.append(entry)

    return {
        "overall": round(overall, 3),
        "n_texts_compared": len(eligible_ids),
        "inconsistent_texts": inconsistent,
    }


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _load_proposals() -> dict[str, dict]:
    """Load all proposal JSON files from the proposals directory."""
    if not PROPOSALS_DIR.exists():
        return {}
    proposals = {}
    for path in sorted(PROPOSALS_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            proposals[path.name] = json.load(f)
    return proposals


def _load_corpus_lookup() -> dict[str, str] | None:
    """Load text_id -> text mapping from corpus.json if available."""
    corpus_path = WORKSPACE / "corpus.json"
    if not corpus_path.exists():
        return None
    with open(corpus_path, encoding="utf-8") as f:
        records = json.load(f)
    return {r["id"]: r["text"] for r in records}


def _ari_label(ari: float) -> str:
    if ari > 0.65:
        return "strong"
    elif ari > 0.25:
        return "moderate"
    else:
        return "weak"


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_cross_proposal(args):
    """Compare all proposal pairs and compute agreement metrics."""
    proposals = _load_proposals()

    if len(proposals) < 2:
        print(f"Need 2+ proposals to compare (found {len(proposals)}). Skipping.")
        sys.exit(0)

    corpus_lookup = _load_corpus_lookup()

    # Build text->cluster maps for each proposal
    prop_maps: dict[str, dict[str, str]] = {}
    for name, prop in proposals.items():
        prop_maps[name] = _build_text_to_cluster(prop)

    prop_names = sorted(prop_maps.keys())

    # Pairwise metrics
    pairwise = {}
    for a_name, b_name in combinations(prop_names, 2):
        key = f"{a_name}:{b_name}"
        pairwise[key] = _compute_pairwise(
            prop_maps[a_name], prop_maps[b_name], top_n=args.top_n,
        )

    # Element similarity across all proposals
    element_sim = _compute_element_similarity(
        prop_maps, corpus_lookup, max_inconsistent=args.top_n,
    )

    # Build report
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "proposals_compared": prop_names,
        "pairwise": pairwise,
        "element_similarity": element_sim,
    }

    # Write JSON report
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = METRICS_DIR / f"cross_proposal_{timestamp}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print human-readable summary
    print("=== Cross-Proposal Agreement ===")
    print(f"Proposals: {', '.join(prop_names)}")
    print()

    print("Pairwise Metrics:")
    ari_values = []
    for key, pw in pairwise.items():
        a, b = key.split(":")
        label = _ari_label(pw["ari"])
        ari_values.append(pw["ari"])
        print(f"  {a} vs {b}: ARI={pw['ari']:.3f} ({label}), NMI={pw['nmi']:.3f}, "
              f"agreement={pw['agreement_rate']:.0%} ({pw['common_texts']} common texts)")
    print()

    # High-entropy clusters
    all_high = []
    for key, pw in pairwise.items():
        for he in pw.get("high_entropy_clusters", []):
            he["pair"] = key
            all_high.append(he)
    all_high.sort(key=lambda x: x["entropy"], reverse=True)

    if all_high:
        print("High-Entropy Clusters (fuzzy boundaries):")
        for he in all_high[:args.top_n]:
            top_dist = list(he["distribution"].items())[:3]
            dist_str = ", ".join(f'{c}→"{n}"' for n, c in top_dist)
            print(f'  "{he["name"]}" ({he["side"]}, entropy={he["entropy"]}, '
                  f'{he["n_texts"]} texts): {dist_str}')
        print()

    print(f"Element Similarity: {element_sim['overall']:.3f} overall "
          f"({element_sim['n_texts_compared']} texts compared)")
    if element_sim["inconsistent_texts"]:
        print("Most Inconsistent Texts:")
        for i, it in enumerate(element_sim["inconsistent_texts"][:5], 1):
            preview = it.get("text_preview", "")
            if preview:
                preview = f': "{preview[:80]}..."' if len(preview) > 80 else f': "{preview}"'
            assign_str = ", ".join(f'{p}→"{c}"' for p, c in it["assignments"].items())
            print(f"  {i}. [{it['text_id']}] (similarity={it['similarity']}){preview}")
            print(f"     {assign_str}")
    print()
    print(f"Full report: {report_path}")

    # Print path for state.py integration
    print(f"\n__REPORT_PATH__:{report_path}")


def cmd_element_similarity(args):
    """Standalone element-centric similarity analysis."""
    if args.source == "proposals":
        proposals = _load_proposals()
        if len(proposals) < 2:
            print(f"Need 2+ proposals (found {len(proposals)}). Skipping.")
            sys.exit(0)
        prop_maps = {name: _build_text_to_cluster(prop) for name, prop in proposals.items()}
    else:
        print("Audit-based element similarity not yet implemented.", file=sys.stderr)
        sys.exit(1)

    corpus_lookup = _load_corpus_lookup()
    result = _compute_element_similarity(prop_maps, corpus_lookup, max_inconsistent=args.top_n)

    print(f"Element Similarity: {result['overall']:.3f} overall "
          f"({result['n_texts_compared']} texts compared)")
    if result["inconsistent_texts"]:
        print("Most Inconsistent Texts:")
        for i, it in enumerate(result["inconsistent_texts"], 1):
            preview = it.get("text_preview", "")
            if preview:
                preview = f': "{preview[:80]}..."' if len(preview) > 80 else f': "{preview}"'
            assign_str = ", ".join(f'{p}→"{c}"' for p, c in it["assignments"].items())
            print(f"  {i}. [{it['text_id']}] (similarity={it['similarity']}){preview}")
            print(f"     {assign_str}")

    print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Cross-proposal agreement analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # cross-proposal
    cp = subparsers.add_parser("cross-proposal", help="Compare all proposal pairs")
    cp.add_argument("--top-n", type=int, default=5, help="Number of top items to show")

    # element-similarity
    es = subparsers.add_parser("element-similarity", help="Per-text consistency analysis")
    es.add_argument("--source", choices=["proposals", "audits"], default="proposals",
                    help="Data source (default: proposals)")
    es.add_argument("--top-n", type=int, default=10, help="Number of top items to show")

    args = parser.parse_args()

    commands = {
        "cross-proposal": cmd_cross_proposal,
        "element-similarity": cmd_element_similarity,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
