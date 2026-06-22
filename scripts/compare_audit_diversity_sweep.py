"""Compare the diverse-auditor arm against the random-auditor baseline.

The treatment (``seed=0_auditdiv``, produced by
``benchmarking.experiments.run_audit_diversity_sweep``) has the auditor draw its
fresh audit texts with TF-IDF farthest-point diversity sampling. The baseline is
the existing production ``seed=0_discoverk`` run (random auditor sampling, same
discover-k config, taxonomy stage only) — NOT re-run here.

Emits, at ``results/audit_diversity_sweep/``:
  - ``comparison.json`` — machine-readable, one row per dataset (baseline + diverse)
  - ``comparison.md``   — a Δk table (baseline vs diverse) + auditor metrics +
                          the audit-region diversity diagnostic + per-dataset detail.

Signals per arm (all valid without classification):
  * ``k_actual`` and Δk vs gold ``k_in_scope``
  * Auditor coverage and mean_confidence (state.meta) — NOTE these are measured
    on *different* audit samples (diverse vs random), so they are not strictly
    apples-to-apples; reported for context, not as the headline.
  * **Audit-region diversity diagnostic** — mean pairwise cosine distance of the
    texts the auditor actually drew, under a TF-IDF model fit on the full corpus.
    This is the treatment check: the diverse arm should score meaningfully higher
    (a more spread audit set) or the directive didn't take.
  * Total agent dispatches and (diverse arm only) orchestrator wall clock.

After finalize, audits/ etc. move to archive/, so readers try archive/ first.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from benchmarking.baselines.agentic_audit_diversity_sweep import (
    SWEEP_ORDER,
    workspace_for,
)
from benchmarking.data_processing.load import load_processed
from benchmarking.paths import RESULTS

OUT_DIR = RESULTS / "audit_diversity_sweep"


def _baseline_workspace(dataset: str, *, seed: int) -> Path:
    return RESULTS / "clustering" / dataset / f"seed={seed}_discoverk"


# --------------------------------------------------------------------------- #
# Helpers (archive-first, mirror compare_overlap_sweep.py)
# --------------------------------------------------------------------------- #


def _read_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _count_glob(directory: Path, pattern: str) -> int:
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.glob(pattern))


def _resolve_dir(ws: Path, name: str) -> Path:
    """Prefer archive/<name> (post-finalize), else live ws/<name>."""
    archived = ws / "archive" / name
    if archived.exists():
        return archived
    return ws / name


def _audited_text_ids(ws: Path) -> set[str]:
    """Union of every text_id the auditor assigned, across all audit files."""
    audit_dir = _resolve_dir(ws, "audits")
    ids: set[str] = set()
    if not audit_dir.exists():
        return ids
    for f in sorted(audit_dir.glob("*.json")):
        data = _read_json(f) or {}
        for a in data.get("assignments", []) or []:
            tid = a.get("text_id")
            if tid is not None:
                ids.add(tid)
    return ids


def _audit_region_diversity(ws: Path) -> float | None:
    """Mean pairwise cosine distance of the audited texts (TF-IDF, fit on corpus).

    The treatment check: a diverse auditor draw should spread the audit set
    across the corpus, raising mean pairwise distance vs a random draw. Returns
    None if there's nothing to measure (no corpus / <2 audited texts).
    """
    corpus = _read_json(ws / "corpus.json")
    if not isinstance(corpus, list) or len(corpus) < 2:
        return None
    audited = _audited_text_ids(ws)
    if len(audited) < 2:
        return None

    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return None

    id_to_text = {r["id"]: r.get("text", "") for r in corpus}
    # Fit on the FULL corpus so both arms share a vector space; transform only
    # the audited subset.
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    vectorizer.fit([r.get("text", "") for r in corpus])
    sub_ids = [tid for tid in audited if tid in id_to_text]
    if len(sub_ids) < 2:
        return None
    mat = vectorizer.transform([id_to_text[t] for t in sub_ids])
    sims = cosine_similarity(mat)
    n = sims.shape[0]
    iu = np.triu_indices(n, k=1)
    dists = 1.0 - sims[iu]
    return float(np.mean(dists))


# --------------------------------------------------------------------------- #
# Per-arm gathering
# --------------------------------------------------------------------------- #


@dataclass
class ArmResult:
    dataset: str
    arm: str  # "baseline" | "diverse"
    workspace: str
    completed: bool = False
    k_in_scope: int | None = None
    k_actual: int | None = None
    delta_k: int | None = None
    audit_coverage: float | None = None
    audit_mean_confidence: float | None = None
    n_audited_texts: int | None = None
    audit_region_diversity: float | None = None
    n_proposers: int | None = None
    n_audits: int | None = None
    n_investigations: int | None = None
    n_synthesizer_outputs: int | None = None
    n_critiques: int | None = None
    total_dispatches: int | None = None
    orchestrator_wall_clock_s: float | None = None
    missing_outputs: list[str] = field(default_factory=list)


def _gather_arm(dataset: str, arm: str, ws: Path, *, gold_k: int) -> ArmResult:
    r = ArmResult(dataset=dataset, arm=arm, workspace=str(ws), k_in_scope=gold_k)
    if not ws.exists():
        r.missing_outputs.append("workspace dir")
        return r

    required = ["final_taxonomy.json", "taxonomy.md"]
    r.missing_outputs.extend([f for f in required if not (ws / f).exists()])

    final = _read_json(ws / "final_taxonomy.json")
    if isinstance(final, dict) and isinstance(final.get("clusters"), list):
        r.k_actual = len(final["clusters"])

    state = _read_json(ws / "state.json")
    if isinstance(state, dict):
        meta = state.get("meta", {}) or {}
        r.n_proposers = meta.get("total_proposals")
        r.n_audits = meta.get("total_audits")
        r.n_investigations = meta.get("total_investigations")
        cov = meta.get("coverage") or {}
        mc = meta.get("mean_confidence") or {}
        if isinstance(cov, dict):
            r.audit_coverage = cov.get("value")
        if isinstance(mc, dict):
            r.audit_mean_confidence = mc.get("value")
        # Fallback only when final_taxonomy is absent AND clusters are populated;
        # an empty list is the freshly-init'd (mid-run) state, not k=0.
        if r.k_actual is None and state.get("clusters"):
            r.k_actual = len(state["clusters"])
    else:
        r.missing_outputs.append("state.json")

    # Wall clock only exists for the diverse arm (our summary writes it).
    summ = _read_json(ws / "_audit_diversity_sweep_summary.json")
    if isinstance(summ, dict):
        r.orchestrator_wall_clock_s = summ.get("orchestrator_wall_clock_s")

    syn_dir = _resolve_dir(ws, "investigations")
    n_synth = _count_glob(syn_dir, "synthesis_*.json") - _count_glob(
        syn_dir, "synthesis_*_clusters.json"
    )
    r.n_synthesizer_outputs = max(n_synth, 0)
    r.n_critiques = _count_glob(_resolve_dir(ws, "critiques"), "critique_*.json")
    r.total_dispatches = sum(
        x or 0
        for x in (
            r.n_proposers,
            r.n_audits,
            r.n_investigations,
            r.n_synthesizer_outputs,
            r.n_critiques,
        )
    )

    audited = _audited_text_ids(ws)
    r.n_audited_texts = len(audited)
    r.audit_region_diversity = _audit_region_diversity(ws)

    if r.k_actual is not None:
        r.delta_k = r.k_actual - gold_k
    if not r.missing_outputs:
        r.completed = True
    return r


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _fmt(x, spec: str = "") -> str:
    if x is None:
        return "—"
    if spec:
        return format(x, spec)
    return str(x)


def _sd(x) -> str:
    """Signed delta-k."""
    if x is None:
        return "—"
    return f"+{x}" if x >= 0 else str(x)


def _render_md(pairs: list[tuple[ArmResult, ArmResult]]) -> str:
    lines: list[str] = []
    lines.append("# Auditor diversity-sampling sweep — comparison\n")
    lines.append(
        "_Auto-generated by `scripts/compare_audit_diversity_sweep.py`. "
        "Treatment = diverse auditor sampling (`seed=0_auditdiv`); baseline = "
        "random auditor sampling (existing `seed=0_discoverk`). Durable writeup "
        "lives in `notes.md` and is not overwritten._\n"
    )

    # Headline Δk table
    lines.append("## Headline — Δk (k_actual − gold)\n")
    lines.append("| dataset | gold | baseline (random) | diverse | Δ(diverse−baseline) |")
    lines.append("|---|---:|---:|---:|---:|")
    sum_abs_base = sum_abs_div = 0
    n_both = 0
    for base, div in pairs:
        gold = base.k_in_scope or div.k_in_scope
        ddiff = (
            (div.delta_k - base.delta_k)
            if (div.delta_k is not None and base.delta_k is not None)
            else None
        )
        if base.delta_k is not None and div.delta_k is not None:
            sum_abs_base += abs(base.delta_k)
            sum_abs_div += abs(div.delta_k)
            n_both += 1
        lines.append(
            f"| {base.dataset} | {_fmt(gold)} | {_sd(base.delta_k)} "
            f"({_fmt(base.k_actual)}) | {_sd(div.delta_k)} ({_fmt(div.k_actual)}) "
            f"| {_sd(ddiff)} |"
        )
    lines.append(
        f"| **Σ\\|Δk\\| (n={n_both})** | | **{sum_abs_base}** | **{sum_abs_div}** | "
        f"**{_sd(sum_abs_div - sum_abs_base)}** |"
    )
    lines.append("")

    # Audit-region diversity diagnostic (treatment check)
    lines.append("## Audit-region diversity (treatment check)\n")
    lines.append(
        "Mean pairwise cosine distance of the texts the auditor actually drew "
        "(TF-IDF fit on the full corpus). Diverse should sit clearly above "
        "baseline, else the directive didn't take.\n"
    )
    lines.append("| dataset | baseline | diverse | Δ | n_audited (base/div) |")
    lines.append("|---|---:|---:|---:|---:|")
    for base, div in pairs:
        d = (
            (div.audit_region_diversity - base.audit_region_diversity)
            if (
                div.audit_region_diversity is not None
                and base.audit_region_diversity is not None
            )
            else None
        )
        lines.append(
            f"| {base.dataset} | {_fmt(base.audit_region_diversity, '.3f')} | "
            f"{_fmt(div.audit_region_diversity, '.3f')} | "
            f"{('+'+format(d,'.3f')) if (d is not None and d>=0) else (format(d,'.3f') if d is not None else '—')} | "
            f"{_fmt(base.n_audited_texts)}/{_fmt(div.n_audited_texts)} |"
        )
    lines.append("")

    # Auditor metrics
    lines.append("## Auditor metrics (context — measured on different samples)\n")
    lines.append(
        "| dataset | coverage base→div | mean-conf base→div | dispatches base→div |"
    )
    lines.append("|---|---:|---:|---:|")
    for base, div in pairs:
        lines.append(
            f"| {base.dataset} | {_fmt(base.audit_coverage, '.2f')}→"
            f"{_fmt(div.audit_coverage, '.2f')} | "
            f"{_fmt(base.audit_mean_confidence, '.2f')}→"
            f"{_fmt(div.audit_mean_confidence, '.2f')} | "
            f"{_fmt(base.total_dispatches)}→{_fmt(div.total_dispatches)} |"
        )
    lines.append("")

    # Diverse-arm wall clock
    lines.append("## Diverse-arm wall clock\n")
    lines.append("| dataset | diverse wall clock (min) |")
    lines.append("|---|---:|")
    for _base, div in pairs:
        wc = div.orchestrator_wall_clock_s
        lines.append(f"| {div.dataset} | {_fmt(wc/60.0 if wc else None, '.1f')} |")
    lines.append("")

    # Incomplete / missing
    incomplete = [
        (r.dataset, r.arm, r.missing_outputs)
        for pair in pairs
        for r in pair
        if r.missing_outputs
    ]
    if incomplete:
        lines.append("## Incomplete cells\n")
        for ds, arm, miss in incomplete:
            lines.append(f"- **{ds}** ({arm}): missing {', '.join(miss)}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=SWEEP_ORDER,
        help="Restrict to the named datasets (default: all in SWEEP_ORDER).",
    )
    args = parser.parse_args()

    datasets = list(args.only) if args.only else SWEEP_ORDER

    pairs: list[tuple[ArmResult, ArmResult]] = []
    for ds in datasets:
        try:
            gold_k = int(load_processed(ds).meta["k_in_scope"])
        except Exception:  # noqa: BLE001 — best-effort; fall back to None
            gold_k = None
        base = _gather_arm(
            ds, "baseline", _baseline_workspace(ds, seed=args.seed), gold_k=gold_k
        )
        div = _gather_arm(
            ds, "diverse", workspace_for(ds, seed=args.seed), gold_k=gold_k
        )
        pairs.append((base, div))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = [
        {"baseline": asdict(base), "diverse": asdict(div)} for base, div in pairs
    ]
    (OUT_DIR / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "comparison.md").write_text(_render_md(pairs), encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'comparison.json'} and {OUT_DIR / 'comparison.md'}")
    # Console roll-up
    for base, div in pairs:
        print(
            f"  {base.dataset:<20} Δk base={_sd(base.delta_k):>4} "
            f"div={_sd(div.delta_k):>4}  "
            f"audit-div base={_fmt(base.audit_region_diversity,'.3f')} "
            f"div={_fmt(div.audit_region_diversity,'.3f')}"
        )


if __name__ == "__main__":
    main()
