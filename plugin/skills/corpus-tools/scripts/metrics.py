#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Compute algorithmic metrics from audit data.

Reads audit files and state.json to report coverage, confidence distribution,
per-cluster stats, and cluster-size distribution. Arithmetic is shared with
state.py update-from-audit via the _audit_metrics helper, so this standalone
reporter cannot disagree with the live workspace state.
"""

import json
import sys
from collections import Counter
from pathlib import Path

# Force UTF-8 on stdout/stderr — Windows defaults to cp1252 and crashes on
# non-ASCII cluster names / corpus content. Idempotent; no-op on streams that
# aren't TextIOWrapper (e.g. captured in tests).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from _audit_metrics import (
    cluster_size_distribution,
    compute_assignment_stats,
    confidence_label,
    normalize_confidence_scale,
)
from _workspace import get_workspace

WORKSPACE = get_workspace()


def load_state() -> dict:
    state_path = WORKSPACE / "state.json"
    if not state_path.exists():
        print("Error: workspace not initialized. Run init.py first.", file=sys.stderr)
        sys.exit(1)
    with open(state_path, encoding="utf-8") as f:
        return json.load(f)


def load_audits() -> list[dict]:
    audit_dir = WORKSPACE / "audits"
    audits = []
    if audit_dir.exists():
        for f in sorted(audit_dir.glob("*.json")):
            with open(f, encoding="utf-8") as fh:
                audits.append(json.load(fh))
    return audits


def main() -> int:
    state = load_state()
    audits = load_audits()

    if not audits:
        print("No audit data available yet.")
        return 0

    # Aggregate assignments from audits matching the current cluster version.
    current_version = state["meta"]["cluster_version"]
    all_assignments = []
    for audit in audits:
        if audit.get("cluster_definitions_version") == current_version:
            all_assignments.extend(audit.get("assignments", []))

    if not all_assignments:
        print("No audit assignments for current cluster version.")
        print(f"Current cluster version: {current_version}")
        print(f"Total audit files: {len(audits)}")
        return 0

    normalize_confidence_scale(all_assignments)
    stats = compute_assignment_stats(all_assignments)
    size_dist = cluster_size_distribution(stats["per_cluster"])

    # Whole-corpus confidence value distribution on the 1-5 scale.
    confidences = [a["confidence"] for a in all_assignments if a.get("confidence") is not None]
    conf_dist = Counter(confidences)

    cluster_id_to_name = {c["id"]: c["name"] for c in state.get("clusters", [])}
    per_cluster_out = {}
    for cid, slot in sorted(stats["per_cluster"].items()):
        mc = slot["mean_confidence"]
        per_cluster_out[cid] = {
            "name": cluster_id_to_name.get(cid, "unknown"),
            "count": slot["count"],
            "mean_confidence": round(mc, 2) if mc is not None else None,
            "confidence_label": confidence_label(mc),
        }

    # Human-readable summary only. The structured aggregates are already in
    # state.json (live workspace metrics) and in the per-audit files; emitting
    # JSON here is just noise in the orchestrator's context.
    print(f"=== Metrics Summary (cluster version {current_version}) ===")
    print(
        f"Coverage: {stats['coverage']:.0%} "
        f"({stats['assigned']}/{stats['total']} assigned, {stats['unclustered']} unclustered)"
    )
    mc = stats["mean_confidence"]
    print(f"Mean confidence: {mc:.2f}" if mc is not None else "Mean confidence: n/a")
    print(f"Confidence distribution: {dict(sorted(conf_dist.items()))}")
    print()
    print("Per-cluster breakdown:")
    for cid, info in sorted(per_cluster_out.items()):
        mc_str = info["mean_confidence"] if info["mean_confidence"] is not None else "n/a"
        print(f"  {cid} ({info['name']}): N={info['count']}, mean_conf={mc_str}, [{info['confidence_label']}]")

    weak = [cid for cid, info in per_cluster_out.items() if info["confidence_label"] == "low"]
    if weak:
        print(f"\nWeak clusters (low confidence): {', '.join(weak)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
