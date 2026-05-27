#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Compute algorithmic metrics from audit data.

Reads audit files and state.json to compute coverage, confidence distributions,
cluster size distributions, and other metrics.
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path


def _get_workspace() -> Path:
    env_ws = os.environ.get("CLUSTERING_WORKSPACE")
    if env_ws:
        return Path(env_ws)
    return Path(".claude/clustering")


WORKSPACE = _get_workspace()


def load_state() -> dict:
    state_path = WORKSPACE / "state.json"
    if not state_path.exists():
        print("Error: workspace not initialized. Run init.py first.", file=sys.stderr)
        sys.exit(1)
    with open(state_path, encoding="utf-8") as f:
        return json.load(f)


def load_audits() -> list[dict]:
    """Load all audit files."""
    audit_dir = WORKSPACE / "audits"
    audits = []
    if audit_dir.exists():
        for f in sorted(audit_dir.glob("*.json")):
            with open(f, encoding="utf-8") as fh:
                audits.append(json.load(fh))
    return audits


def compute_metrics():
    state = load_state()
    audits = load_audits()

    if not audits:
        print("No audit data available yet.")
        return

    # Aggregate all assignments from audits matching current cluster version
    current_version = state["meta"]["cluster_version"]
    all_assignments = []
    for audit in audits:
        if audit.get("cluster_definitions_version") == current_version:
            all_assignments.extend(audit.get("assignments", []))

    if not all_assignments:
        print("No audit assignments for current cluster version.")
        print(f"Current cluster version: {current_version}")
        print(f"Total audit files: {len(audits)}")
        return

    # Auto-detect 0-1 float scale and normalize to 1-5 integer scale
    all_conf_values = [a["confidence"] for a in all_assignments if a.get("confidence") is not None]
    if all_conf_values and max(all_conf_values) <= 1.0:
        print(
            "WARNING: Detected 0-1 float confidence scale in audit data. "
            "Normalizing to 1-5 integer scale (multiply by 5).",
            file=sys.stderr,
        )
        for a in all_assignments:
            if a.get("confidence") is not None:
                a["confidence"] = round(a["confidence"] * 5, 1)

    # Coverage
    total = len(all_assignments)
    assigned = sum(1 for a in all_assignments if a.get("cluster_id") is not None)
    unclustered = total - assigned
    coverage = assigned / total if total > 0 else 0

    # Confidence distribution
    confidences = [a["confidence"] for a in all_assignments if a.get("confidence") is not None]
    conf_dist = Counter(confidences)
    mean_conf = sum(confidences) / len(confidences) if confidences else 0

    # Per-cluster stats
    cluster_stats = {}
    for a in all_assignments:
        cid = a.get("cluster_id")
        if cid:
            cluster_stats.setdefault(cid, {"count": 0, "confidences": []})
            cluster_stats[cid]["count"] += 1
            if a.get("confidence") is not None:
                cluster_stats[cid]["confidences"].append(a["confidence"])

    # Cluster size distribution
    cluster_sizes = {cid: s["count"] for cid, s in cluster_stats.items()}

    # Build output
    output = {
        "cluster_version": current_version,
        "total_assignments": total,
        "coverage": {
            "assigned": assigned,
            "unclustered": unclustered,
            "coverage_pct": round(coverage * 100, 1),
        },
        "confidence": {
            "mean": round(mean_conf, 2),
            "distribution": {str(k): v for k, v in sorted(conf_dist.items())},
            "total_scored": len(confidences),
        },
        "per_cluster": {},
        "cluster_size_distribution": {
            "min": min(cluster_sizes.values()) if cluster_sizes else 0,
            "max": max(cluster_sizes.values()) if cluster_sizes else 0,
            "mean": round(sum(cluster_sizes.values()) / len(cluster_sizes), 1) if cluster_sizes else 0,
        },
    }

    # Per-cluster details
    cluster_id_to_name = {c["id"]: c["name"] for c in state.get("clusters", [])}
    for cid, stats in sorted(cluster_stats.items()):
        confs = stats["confidences"]
        mean_c = sum(confs) / len(confs) if confs else 0
        label = "high" if mean_c >= 4.0 else "medium" if mean_c >= 3.0 else "low"
        output["per_cluster"][cid] = {
            "name": cluster_id_to_name.get(cid, "unknown"),
            "count": stats["count"],
            "mean_confidence": round(mean_c, 2),
            "confidence_label": label,
        }

    # Print formatted output
    print(json.dumps(output, indent=2))

    # Also print human-readable summary
    print()
    print(f"=== Metrics Summary (cluster version {current_version}) ===")
    print(f"Coverage: {coverage:.0%} ({assigned}/{total} assigned, {unclustered} unclustered)")
    print(f"Mean confidence: {mean_conf:.2f}")
    print(f"Confidence distribution: {dict(sorted(conf_dist.items()))}")
    print()
    print("Per-cluster breakdown:")
    for cid, info in sorted(output["per_cluster"].items()):
        print(f"  {cid} ({info['name']}): N={info['count']}, mean_conf={info['mean_confidence']}, [{info['confidence_label']}]")

    # Identify weak clusters
    weak = [cid for cid, info in output["per_cluster"].items() if info["confidence_label"] == "low"]
    if weak:
        print(f"\nWeak clusters (low confidence): {', '.join(weak)}")


if __name__ == "__main__":
    compute_metrics()
