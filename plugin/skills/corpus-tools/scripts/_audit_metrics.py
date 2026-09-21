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

# Minimum per-cluster audit assignments before a confidence label is
# publishable. Below this, `confidence_label` returns "insufficient-sample"
# instead of high/medium/low.
#
# Why this exists: the audit sample size guidance is calibrated for an agent's
# context window ("200-400 short texts"), which says nothing about how many
# clusters the draw has to cover. On a 65-cluster taxonomy a 300-text audit
# averages 4.6 assignments per cluster, and a uniform draw allocates them in
# proportion to corpus prevalence — so the rare clusters, whose fit is least
# certain, get the fewest texts. A mean of two 5s is not evidence that a
# cluster is "high confidence", and publishing it as such in taxonomy.md
# misleads the human reading it and the downstream classify-tune /
# classify-label steps that consume it.
#
# 5 is a judgment call, not a derivation. The standard error of a 1-5 mean
# (σ≈1 in practice) is ~0.45 at n=5, which still sits inside the 1.0 gap
# between adjacent labels, so the label is not pure sampling noise — but with
# less margin than n=8 (~0.35) or n=10 (~0.32) would give.
#
# The floor is chosen at the low end deliberately. Clearing it is real work:
# the per-cluster draw aims at roughly `floor x (number of short clusters)`,
# every clustering run pays that cost, and `finalize` refuses to export until
# it is paid. Going from 5 to 8 tightens the standard error by ~0.1 and raises
# the added audit budget by ~60%, which is a poor trade at this margin.
#
# Tunable per run via `state.py finalize --min-audit-n`; 0 disables the guard
# entirely and publishes every label regardless of sample size.
MIN_AUDIT_N_PER_CLUSTER = 5

# An audit's sampling basis, declared by the auditor in the audit file's
# `sample_basis` field (absent → "random", which is what every audit written
# before this field existed was).
#
# "random" is a uniform draw from unseen texts, so its assignments estimate
# corpus-wide coverage and mean confidence. "stratified" deliberately
# over-draws texts that plausibly belong to each cluster (sample.py
# --strategy stratified) to give thin clusters a defensible per-cluster n.
# That is the right basis for per-cluster fit and the wrong basis for
# prevalence: folding it into coverage would count the same corpus region
# many times over and overstate how much of the corpus is covered. So
# stratified audits update per-cluster evidence only, and the headline
# coverage / mean-confidence figures stay sourced from random audits.
SAMPLE_BASES = ("random", "stratified")
COVERAGE_SAMPLE_BASIS = "random"


def sample_basis(audit: dict) -> str:
    """Read an audit's declared sampling basis, defaulting to "random".

    Unknown values fall back to "stratified" — the conservative choice, since
    it keeps an audit we can't vouch for out of the coverage figure rather
    than silently letting it in.
    """
    basis = audit.get("sample_basis") or COVERAGE_SAMPLE_BASIS
    if basis not in SAMPLE_BASES:
        print(
            f"WARNING: unrecognised sample_basis {basis!r}; treating as "
            f"'stratified' so it cannot contaminate the coverage estimate.",
            file=sys.stderr,
        )
        return "stratified"
    return basis


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


def confidence_label(
    mean: float | None,
    n: int | None = None,
    targeted: int | None = None,
    *,
    min_n: int = MIN_AUDIT_N_PER_CLUSTER,
) -> str:
    """Per-cluster label from the audit evidence. One of:

        high / medium / low  – a real verdict: mean ≥4.0 / ≥3.0 / below, on a
                               sample that met the floor
        insufficient-sample  – too few assignments to say, AND the corpus has
                               not been searched for this cluster either. The
                               fix is another audit pass
        unsupported          – too few assignments, but ``min_n`` or more texts
                               were deliberately aimed at this cluster and the
                               auditor put them elsewhere. Repeating the same
                               draw will not help. Three readings, in the order
                               worth checking: the sampler could not retrieve
                               the cluster's texts (try
                               ``--seed-from assigned``); a neighbouring
                               cluster is absorbing them; or the corpus really
                               does not support the cluster
        unaudited            – nothing assigned and nothing targeted

    ``n`` is the number of audit assignments the mean rests on; ``targeted``
    is how many texts a stratified draw aimed at this cluster
    (``evidence.audit_targeted``). Separating them is the whole point: a cluster
    that drew 0 of 8 targeted texts has been tested and failed, which is a
    finding, whereas one that drew 0 of 0 has simply not been looked at, which
    is not. Before this distinction existed both read as thin.

    Both are optional. With neither, the label is the unguarded
    mean-threshold verdict, which is what every pre-floor caller expects.
    Pass ``min_n=0`` to disable the floor entirely.
    """
    tested = bool(min_n) and targeted is not None and targeted >= min_n
    if mean is None:
        # Nothing landed here. Whether that is a finding depends entirely on
        # whether anything was aimed here.
        return "unsupported" if tested else "unaudited"
    if n is not None and min_n and n < min_n:
        return "unsupported" if tested else "insufficient-sample"
    for threshold, label in CONFIDENCE_LABEL_THRESHOLDS:
        if mean >= threshold:
            return label
    return "low"


# Labels that mean "the sample is too thin to publish a verdict", as opposed to
# a verdict of low quality. Callers use this to route: thin → audit more,
# low → investigate.
THIN_LABELS = ("unaudited", "insufficient-sample", "unsupported")

# What a withheld label is called in the published taxonomy.
PUBLISHED_WITHHELD_LABEL = "unvalidated"


def display_label(label: str) -> str:
    """Collapse the thin labels for the published taxonomy.

    The three-way split (unaudited / insufficient-sample / unsupported) exists
    to route the discovery loop: the first two mean "audit more", the third
    means "investigate". That is working state, and it belongs in `summary.md`,
    `metrics.py` and `update-from-audit` output, where the orchestrator reads
    it and acts on it.

    `taxonomy.md` is the deliverable, and a reader of the deliverable doesn't
    need the sampler's reasoning — only that this cluster's label isn't backed
    by enough evidence to state. A finalize that has been gated properly
    produces none of these at all; when one survives, it survives because
    somebody chose to ship it. The precise label stays in
    `final_taxonomy.json` alongside `audit_n` / `audit_targeted`.
    """
    return PUBLISHED_WITHHELD_LABEL if label in THIN_LABELS else label


def audit_power(
    clusters: list[dict],
    *,
    min_n: int = MIN_AUDIT_N_PER_CLUSTER,
) -> dict:
    """Summarise whether per-cluster audit samples are thick enough to publish.

    Takes state.json-shaped cluster dicts (reading
    ``evidence.audit_assignments`` and ``evidence.audit_targeted``) and returns:

        min_audit_n   – the floor applied
        n_clusters    – clusters considered
        below_floor   – [{id, name, n, targeted, tested}] for clusters under
                        the floor, thinnest first, including n == 0
        needs_audit   – the subset of below_floor that has NOT been searched
                        for (targeted < min_n). More audit fixes these
        unsupported   – the subset that HAS been searched for and still came up
                        short (targeted >= min_n). More audit will not fix
                        these; they are a taxonomy finding
        unaudited     – count with n == 0
        median_n      – median per-cluster n (0 when there are no clusters)
        total_needed  – extra assignments required to bring every cluster in
                        `needs_audit` to the floor. Excludes `unsupported`,
                        since drawing more for those is money after a question
                        that has already been answered

    Pure; does not mutate the input.
    """
    rows: list[dict] = []
    for c in clusters:
        ev = c.get("evidence", {}) or {}
        n = int(ev.get("audit_assignments", 0) or 0)
        targeted = int(ev.get("audit_targeted", 0) or 0)
        rows.append({
            "id": c.get("id", "?"),
            "name": c.get("name", ""),
            "n": n,
            "targeted": targeted,
            "tested": bool(min_n) and targeted >= min_n,
        })

    ns = sorted(r["n"] for r in rows)
    if ns:
        mid = len(ns) // 2
        median_n = ns[mid] if len(ns) % 2 else (ns[mid - 1] + ns[mid]) / 2
    else:
        median_n = 0

    below = sorted(
        (r for r in rows if r["n"] < min_n),
        key=lambda r: (r["n"], r["id"]),
    )
    needs_audit = [r for r in below if not r["tested"]]
    unsupported = [r for r in below if r["tested"]]
    return {
        "min_audit_n": min_n,
        "n_clusters": len(rows),
        "below_floor": below,
        "needs_audit": needs_audit,
        "unsupported": unsupported,
        "unaudited": sum(1 for r in rows if r["n"] == 0),
        "median_n": median_n,
        "total_needed": sum(min_n - r["n"] for r in needs_audit),
    }
