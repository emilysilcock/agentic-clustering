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
    COVERAGE_SAMPLE_BASIS,
    MIN_AUDIT_N_PER_CLUSTER,
    cluster_size_distribution,
    compute_assignment_stats,
    confidence_label,
    normalize_confidence_scale,
    sample_basis,
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


def load_strata(current_version: int | None) -> dict[str, set[str]]:
    """Text ids aimed at each cluster, from manifests for the current version.

    Read from ``strata/`` rather than from ``state.json``'s cumulative
    ``audit_targeted`` counter so it is filtered the same way this script's
    assignments are. Mixing a version-filtered numerator with a cumulative
    denominator would report clusters as `unsupported` on the strength of a
    draw aimed at a since-merged or since-split definition.
    """
    strata_dir = WORKSPACE / "strata"
    if not strata_dir.exists():
        return {}
    out: dict[str, set[str]] = {}
    for f in sorted(strata_dir.glob("strata_*.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        version = data.get("cluster_definitions_version")
        # Manifests written before the version field existed can't be placed,
        # so they're skipped rather than credited to the current cluster set.
        if current_version is not None and version != current_version:
            continue
        for cid, tids in (data.get("strata") or {}).items():
            out.setdefault(cid, set()).update(tids)
    return out


def main() -> int:
    state = load_state()
    audits = load_audits()

    if not audits:
        print("No audit data available yet.")
        return 0

    # Aggregate assignments from audits matching the current cluster version,
    # split by sampling basis. Per-cluster fit uses every assignment; coverage
    # and the headline mean use the uniform (random) draws only, because a
    # stratified audit deliberately over-draws each cluster's own region and
    # so is not a prevalence estimate.
    current_version = state["meta"]["cluster_version"]
    all_assignments = []
    coverage_assignments = []
    basis_counts: Counter = Counter()
    for audit in audits:
        if audit.get("cluster_definitions_version") != current_version:
            continue
        rows = audit.get("assignments", [])
        basis = sample_basis(audit)
        basis_counts[basis] += len(rows)
        all_assignments.extend(rows)
        if basis == COVERAGE_SAMPLE_BASIS:
            coverage_assignments.extend(rows)

    if not all_assignments:
        print("No audit assignments for current cluster version.")
        print(f"Current cluster version: {current_version}")
        print(f"Total audit files: {len(audits)}")
        return 0

    normalize_confidence_scale(all_assignments)
    stats = compute_assignment_stats(all_assignments)
    size_dist = cluster_size_distribution(stats["per_cluster"])
    cov_stats = compute_assignment_stats(coverage_assignments) if coverage_assignments else None

    # Whole-corpus confidence value distribution on the 1-5 scale.
    confidences = [a["confidence"] for a in all_assignments if a.get("confidence") is not None]
    conf_dist = Counter(confidences)

    cluster_id_to_name = {c["id"]: c["name"] for c in state.get("clusters", [])}
    # Targets, version-filtered to match the assignments above, and counted
    # only where the aimed text actually got audited — that intersection is
    # the honest denominator for "aimed here and placed elsewhere".
    audited_ids = {a.get("text_id") for a in all_assignments if a.get("text_id")}
    targeted_by_cid = {
        cid: len(tids & audited_ids)
        for cid, tids in load_strata(current_version).items()
    }
    per_cluster_out = {}
    # Include clusters that drew nothing — an absent row reads as "no problem
    # here" when it actually means the opposite.
    for cid in cluster_id_to_name:
        stats["per_cluster"].setdefault(cid, {"count": 0, "confidences": [], "mean_confidence": None})
    for cid, slot in sorted(stats["per_cluster"].items()):
        mc = slot["mean_confidence"]
        targeted = targeted_by_cid.get(cid, 0)
        per_cluster_out[cid] = {
            "name": cluster_id_to_name.get(cid, "unknown"),
            "count": slot["count"],
            "targeted": targeted,
            "mean_confidence": round(mc, 2) if mc is not None else None,
            "confidence_label": confidence_label(mc, slot["count"], targeted),
        }

    # Human-readable summary only. The structured aggregates are already in
    # state.json (live workspace metrics) and in the per-audit files; emitting
    # JSON here is just noise in the orchestrator's context.
    print(f"=== Metrics Summary (cluster version {current_version}) ===")
    if cov_stats is None:
        print(
            "Coverage: n/a — every audit for this cluster version was a "
            "stratified draw, which cannot estimate corpus coverage. Run a "
            "random audit."
        )
    else:
        print(
            f"Coverage: {cov_stats['coverage']:.0%} "
            f"({cov_stats['assigned']}/{cov_stats['total']} assigned, "
            f"{cov_stats['unclustered']} unclustered; random draws only)"
        )
        mc = cov_stats["mean_confidence"]
        print(f"Mean confidence: {mc:.2f}" if mc is not None else "Mean confidence: n/a")
    if basis_counts.get("stratified"):
        print(
            f"Assignment basis: {basis_counts.get('random', 0)} random + "
            f"{basis_counts['stratified']} stratified "
            f"(per-cluster figures below pool both; coverage uses random only)"
        )
    print(f"Confidence distribution: {dict(sorted(conf_dist.items()))}")
    print(
        f"Assignments per cluster: min {size_dist['min']}, max {size_dist['max']}, "
        f"mean {size_dist['mean']:.1f} (over clusters that drew at least one)"
    )
    print()
    print(f"Per-cluster breakdown (label floor: n>={MIN_AUDIT_N_PER_CLUSTER}):")
    for cid, info in sorted(per_cluster_out.items()):
        mc_str = info["mean_confidence"] if info["mean_confidence"] is not None else "n/a"
        aimed = f", aimed={info['targeted']}" if info["targeted"] else ""
        print(
            f"  {cid} ({info['name']}): N={info['count']}{aimed}, "
            f"mean_conf={mc_str}, [{info['confidence_label']}]"
        )

    weak = [cid for cid, info in per_cluster_out.items() if info["confidence_label"] == "low"]
    if weak:
        print(f"\nWeak clusters (low confidence, on a sample that met the floor): "
              f"{', '.join(weak)}")

    under = [
        (cid, info["count"])
        for cid, info in sorted(per_cluster_out.items())
        if info["confidence_label"] in ("insufficient-sample", "unaudited")
    ]
    if under:
        needed = sum(MIN_AUDIT_N_PER_CLUSTER - n for _, n in under)
        print(
            f"\nUnder-sampled — audit these ({len(under)}/{len(per_cluster_out)} "
            f"clusters under n={MIN_AUDIT_N_PER_CLUSTER}, never searched for): "
            f"{', '.join(f'{cid}(n={n})' for cid, n in under)}"
        )
        print(
            f"  Sample-size problems, not quality findings. Top up with "
            f"`sample.py --strategy stratified --per-cluster "
            f"{MIN_AUDIT_N_PER_CLUSTER}` (~{needed} more assignments) and audit "
            f"that draw with \"sample_basis\": \"stratified\"."
        )

    unsupported = [
        (cid, info["count"], info["targeted"])
        for cid, info in sorted(per_cluster_out.items())
        if info["confidence_label"] == "unsupported"
    ]
    if unsupported:
        print(
            f"\nUnsupported — investigate these, don't re-audit ({len(unsupported)} "
            f"clusters): "
            + ", ".join(f"{cid}(n={n} of {t} aimed)" for cid, n, t in unsupported)
        )
        print(
            "  Candidates were drawn for these and the auditor assigned them "
            "elsewhere, so more of the same auditing won't move them. Find where "
            "the candidates went: a neighbour absorbing them argues for a merge "
            "or a sharper boundary, nothing absorbing them argues the corpus "
            "does not support the cluster."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
