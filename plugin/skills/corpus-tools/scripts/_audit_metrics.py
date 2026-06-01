"""Shared audit-assignment aggregation.

`state.py update-from-audit` and `metrics.py` both summarise the same audit
data — coverage = assigned/total, global mean confidence, per-cluster counts
and mean confidence. Centralising the arithmetic here makes the two paths
agree by construction and removes the previous reliance on the auditor LLM's
own `summary` block for the headline coverage and mean-confidence numbers.

Stdlib only (no third-party deps); safe to import from any PEP 723 script in
this directory. Note: import has one side effect — it reconfigures
`sys.stdout`/`sys.stderr` to UTF-8 to match the rest of the codebase (idempotent,
no-op on already-reconfigured or non-TextIOWrapper streams).
"""

from __future__ import annotations

import sys

# Force UTF-8 on stdout/stderr — Windows defaults to cp1252 and crashes on
# non-ASCII cluster names / corpus content. Idempotent; no-op on streams that
# aren't TextIOWrapper (e.g. captured in tests).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# (threshold, label) pairs in descending order. Matches the long-standing
# state.py update-from-audit semantics so the per-cluster confidence label
# doesn't shift under callers.
CONFIDENCE_LABEL_THRESHOLDS = ((4.0, "high"), (3.0, "medium"))


def normalize_confidence_scale(assignments: list[dict], *, warn: bool = True) -> bool:
    """If the assignments look like 0-1 floats, rescale to 1-5 in place.
    Returns True if normalisation was applied.

    The auditor agent is instructed to emit integer 1-5, but models sometimes
    drift to 0-1 floats. The trigger is "at least one non-integer value AND
    every non-null value ≤ 1.0" — the non-integer requirement is what
    distinguishes a genuine 0-1 audit (which will contain values like 0.85)
    from a legitimate all-1 integer audit (where every text was a forced guess).
    The old max-only heuristic inflated the all-1 case to all-5 and made a
    terrible audit look perfect.
    """
    confs = [a["confidence"] for a in assignments if a.get("confidence") is not None]
    if not confs or max(confs) > 1.0:
        return False
    # Require at least one fractional value before treating as 0-1 scale.
    # A genuine 0-1 audit will have values like 0.4, 0.85, etc.; an all-1
    # integer audit (all forced guesses) will have only 1s and must be left
    # alone.
    has_fraction = any(float(c) != int(float(c)) for c in confs)
    if not has_fraction:
        return False
    if warn:
        print(
            "WARNING: detected 0-1 float confidence scale in audit assignments. "
            "Normalising to 1-5 integer scale (multiply by 5).",
            file=sys.stderr,
        )
    for a in assignments:
        if a.get("confidence") is not None:
            a["confidence"] = int(round(a["confidence"] * 5))
    return True


def compute_assignment_stats(assignments: list[dict]) -> dict:
    """Aggregate a flat list of audit assignments. Pure; does not mutate input.

    Returns the canonical metric names used by state.py's meta.coverage /
    meta.mean_confidence and by metrics.py:

        total            – number of assignments
        assigned         – assignments with a non-null cluster_id
        unclustered      – total - assigned
        coverage         – assigned / total (or 0.0 when total == 0)
        mean_confidence  – mean of non-null confidences (or None)
        per_cluster      – {cid: {count, confidences: [...], mean_confidence}}

    Pass concatenated assignments from multiple audits to get a cross-audit
    aggregate (this is how metrics.py uses it).
    """
    total = len(assignments)
    assigned_list = [a for a in assignments if a.get("cluster_id")]
    assigned = len(assigned_list)
    unclustered = total - assigned
    coverage = (assigned / total) if total else 0.0

    confs = [a["confidence"] for a in assignments if a.get("confidence") is not None]
    mean_confidence = (sum(confs) / len(confs)) if confs else None

    per_cluster: dict[str, dict] = {}
    for a in assigned_list:
        cid = a["cluster_id"]
        slot = per_cluster.setdefault(cid, {"count": 0, "confidences": []})
        slot["count"] += 1
        if a.get("confidence") is not None:
            slot["confidences"].append(a["confidence"])
    for slot in per_cluster.values():
        cs = slot["confidences"]
        slot["mean_confidence"] = (sum(cs) / len(cs)) if cs else None

    return {
        "total": total,
        "assigned": assigned,
        "unclustered": unclustered,
        "coverage": coverage,
        "mean_confidence": mean_confidence,
        "per_cluster": per_cluster,
    }


def cluster_size_distribution(per_cluster: dict[str, dict]) -> dict:
    """Min/max/mean of cluster sizes. Empty input → all zeros."""
    sizes = [s["count"] for s in per_cluster.values()]
    if not sizes:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {
        "min": min(sizes),
        "max": max(sizes),
        "mean": sum(sizes) / len(sizes),
    }


def confidence_label(mean: float | None) -> str:
    """Per-cluster confidence label on the 1-5 mean scale.

    Returns ``"unaudited"`` when ``mean`` is None.
    Thresholds: ≥4.0 → high, ≥3.0 → medium, else low.
    """
    if mean is None:
        return "unaudited"
    for threshold, label in CONFIDENCE_LABEL_THRESHOLDS:
        if mean >= threshold:
            return label
    return "low"
