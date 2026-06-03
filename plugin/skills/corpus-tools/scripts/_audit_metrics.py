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
    """Normalise an audit's confidences to the integer 1-5 contract, in place.

    The auditor prompt requires INTEGER 1-5. Models occasionally drift; this
    helper detects each realistic drift mode and rescues it, clamping the rest
    so downstream code can rely on the contract. Returns True iff any value
    was changed.

    Scale-detection rules (evaluated in this order):

    1. **All-1 integers** — left alone. The audit's signal IS that every
       assignment is a forced guess on the 1-5 scale; rescaling ×5 here
       would silently invert the audit's meaning. (Round 2 #7 fix; the
       reason this whole function exists in its current shape.)
    2. **0-1 floats with at least one fractional value** (e.g., 0.85, 0.4) —
       rescale ×5, round, clamp to [1, 5]. The classic model-drift case.
    3. **Binary 0/1 integers** (mix of 0 and 1, no fractions, NOT all-1) —
       rescale ×5 and clamp, so 0 → 1 (the minimum valid 1-5) and 1 → 5.
       Distinguished from rule 1 by the presence of any non-1 value.
    4. **All-zero audit** — degenerate; clamp each to 1 with a warning. A 0
       on the 1-5 scale is invalid, so passing the values through unchanged
       would leak invalid confidences into downstream metrics.
    5. **Integer 1-5 in range** — left alone (the happy path).
    6. **Out-of-range or non-integer values >1** (e.g., 7, 3.5, -0.5) —
       round and clamp to [1, 5] with a warning. Defensive; the auditor
       shouldn't emit these but we'd rather repair than crash.
    7. **Non-numeric confidences** — warn and leave the audit untouched
       (we can't repair what we can't read; downstream will catch it).
    """
    # Coerce + collect; abort if anything is non-numeric.
    try:
        values = [
            float(a["confidence"])
            for a in assignments
            if a.get("confidence") is not None
        ]
    except (TypeError, ValueError):
        if warn:
            # Re-iterate to find which assignments couldn't be coerced so the
            # operator has a starting point for debugging. Only runs on the
            # error path, so the happy-path cost is unchanged.
            bad_ids: list[str] = []
            for a in assignments:
                c = a.get("confidence")
                if c is None:
                    continue
                try:
                    float(c)
                except (TypeError, ValueError):
                    bad_ids.append(str(a.get("text_id", "<no text_id>")))
            preview = ", ".join(bad_ids[:5])
            if len(bad_ids) > 5:
                preview += f" (+ {len(bad_ids) - 5} more)"
            print(
                f"WARNING: non-numeric confidence in audit ({len(bad_ids)} "
                f"assignment(s): {preview}); leaving as-is.",
                file=sys.stderr,
            )
        return False
    if not values:
        return False
    max_val = max(values)
    min_val = min(values)

    def _warn(msg: str) -> None:
        if warn:
            print(f"WARNING: {msg}", file=sys.stderr)

    def _rescale_and_clamp(scale: float) -> bool:
        for a in assignments:
            c = a.get("confidence")
            if c is not None:
                a["confidence"] = max(1, min(5, int(round(c * scale))))
        return True

    if max_val <= 1.0:
        has_fraction = any(v != int(v) for v in values)
        if has_fraction:
            _warn("detected 0-1 float confidence scale; rescaling to integer 1-5")
            return _rescale_and_clamp(5.0)
        # From here down, all values are integers with max <= 1. Match the
        # specific shapes explicitly — falling through "anything else" would
        # silently accept pathological inputs like [-1, -1, -1] or [-1, 0, 1].
        if min_val == 1 and max_val == 1:
            # All-1 integers — Round 2 preserves this case.
            return False
        if min_val == 0 and max_val == 1:
            _warn("detected binary 0/1 confidence scale; rescaling to integer 1-5")
            return _rescale_and_clamp(5.0)
        if min_val == 0 and max_val == 0:
            _warn("all-zero confidence audit; clamping each value to 1")
            for a in assignments:
                if a.get("confidence") is not None:
                    a["confidence"] = 1
            return True
        # Anything left has negative values (max <= 1, integer, and not one
        # of the well-formed shapes above). Clamp into [1, 5] defensively.
        _warn(
            "non-positive confidence values detected; clamping each into [1, 5]"
        )
        return _rescale_and_clamp(1.0)

    # max_val > 1.0 — expected integer 1-5 scale. Clamp anything out of range
    # or non-integer.
    if any(v < 1 or v > 5 or v != int(v) for v in values):
        _warn(
            "confidence values outside integer 1-5 range; rounding and "
            "clamping to [1, 5]"
        )
        return _rescale_and_clamp(1.0)
    return False


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
