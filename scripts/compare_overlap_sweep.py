"""Tabulate the four proposer-overlap variants (v0=disjoint, v1=25%, v2=50%,
v3=100%) and judge taxonomy alignment.

Reads every (dataset, variant) workspace produced by
``benchmarking.experiments.run_overlap_sweep`` and emits, at
``results/overlap_sweep/``:

  - ``comparison.json`` — machine-readable, one row per (dataset, variant) cell
  - ``comparison.md``   — a Δk pivot across the four overlap arms + a per-cell
                          roll-up + per-dataset detail.

Signals per cell (everything valid without classification):
  * ``k_actual`` and Δk vs ``k_in_scope``
  * Realized proposer overlap (mean pairwise, from the cell's proposals +
    shared core) and a proposer-health flag (empty proposals → DEGRADED). This
    is the data-quality gate: a cell whose proposers didn't actually share the
    intended core, or where proposers came back empty, is flagged so it doesn't
    masquerade as a clean data point.
  * Cross-proposal ARI / NMI — the agreement signal overlap *unlocks* (disjoint
    v0 has no shared texts, so it stays None). Best-effort: read from
    state.meta.cross_proposal_metrics, else the latest metrics/*.json.
  * Final auditor coverage and mean_confidence (state.meta)
  * Critic finalize_recommendation (latest critique)
  * Total agent dispatches and orchestrator wall clock
  * Optional LLM-as-judge alignment vs gold taxonomy (--skip-judge to omit)

After finalize, proposals/ and shared_samples/ move to archive/, so all
workspace readers try archive/ first, then the live dir.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path

from benchmarking.baselines.agentic_overlap_sweep import (
    SWEEP_ORDER,
    VARIANTS,
    method_for,
    workspace_for,
)
from benchmarking.llm_clients.claude_code import call_claude
from benchmarking.paths import DATA_DERIVED, RESULTS

OUT_DIR = RESULTS / "overlap_sweep"
JUDGE_MODEL = "claude-opus-4-7"


# --------------------------------------------------------------------------- #
# Data gathering
# --------------------------------------------------------------------------- #


@dataclass
class CellResult:
    dataset: str
    variant: str
    overlap_fraction: float
    method: str
    workspace: str
    completed: bool
    k_in_scope: int | None = None
    k_actual: int | None = None
    k_range: list[int] = field(default_factory=list)
    # Realized overlap / proposer health
    n_proposals_files: int | None = None
    n_empty_proposals: int | None = None
    core_size: int | None = None
    realized_overlap_mean: float | None = None
    degraded: bool = False
    degraded_reason: str | None = None
    # Cross-proposal agreement (unlocked by overlap)
    cross_proposal_ari: float | None = None
    cross_proposal_nmi: float | None = None
    # Auditor / critic / cost
    audit_coverage: float | None = None
    audit_mean_confidence: float | None = None
    n_proposers: int | None = None
    n_audits: int | None = None
    n_investigations: int | None = None
    n_synthesizer_outputs: int | None = None
    n_critiques: int | None = None
    total_dispatches: int | None = None
    orchestrator_wall_clock_s: float | None = None
    critic_finalize_recommendation: str | None = None
    judge_score: int | None = None
    judge_rationale: str | None = None
    missing_outputs: list[str] = field(default_factory=list)


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _latest_by_mtime(directory: Path, pattern: str) -> Path | None:
    if not directory.exists():
        return None
    candidates = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


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


def _proposal_ids(path: Path) -> set[str]:
    data = _read_json(path) or {}
    ids: set[str] = set()
    for c in data.get("clusters", []) or []:
        ids.update(c.get("text_ids", []) or [])
    ids.update(data.get("unclustered_ids", []) or [])
    return ids


def _gather_overlap_health(result: CellResult, ws: Path) -> None:
    """Compute realized mean pairwise overlap + flag degraded proposers.

    v0 (disjoint, fraction 0) has no shared core — realized overlap is expected
    ~0 and we leave core_size None. For v1/v2/v3 we read the persisted core and
    each proposal's full id set; pairwise overlap is |A∩B|/min(|A|,|B|).
    """
    prop_dir = _resolve_dir(ws, "proposals")
    props = sorted(prop_dir.glob("prop_*.json")) if prop_dir.exists() else []
    samples = {p.name: _proposal_ids(p) for p in props}
    result.n_proposals_files = len(samples)
    n_empty = sum(1 for ids in samples.values() if not ids)
    result.n_empty_proposals = n_empty

    core_path = _resolve_dir(ws, "shared_samples") / "proposer_core.json"
    core_ids = set(_read_json(core_path) or []) if core_path.exists() else set()
    if result.overlap_fraction > 0 and core_ids:
        result.core_size = len(core_ids)

    nonempty = {n: ids for n, ids in samples.items() if ids}
    if len(nonempty) >= 2:
        fracs = []
        for (_, a), (_, b) in combinations(nonempty.items(), 2):
            denom = min(len(a), len(b)) or 1
            fracs.append(len(a & b) / denom)
        result.realized_overlap_mean = sum(fracs) / len(fracs)

    # Degraded if any proposer came back empty, or (for an overlap arm) fewer
    # than half the proposers actually carry the shared core.
    reasons = []
    if n_empty > 0:
        reasons.append(f"{n_empty} empty proposal(s)")
    if result.overlap_fraction > 0 and core_ids and nonempty:
        carrying = sum(1 for ids in nonempty.values() if core_ids <= ids)
        if carrying < max(2, len(nonempty) // 2 + 1):
            reasons.append(
                f"only {carrying}/{len(nonempty)} proposers carry the full core"
            )
    if reasons:
        result.degraded = True
        result.degraded_reason = "; ".join(reasons)


def _gather_cross_proposal(result: CellResult, ws: Path, state: dict | None) -> None:
    """Best-effort cross-proposal ARI/NMI. v0 has no shared texts → leave None.

    confusion.py writes to metrics/<file>.json and state.py copies a summary
    into state.meta.cross_proposal_metrics. We try state.meta first, then the
    latest metrics file. We look for mean/aggregate ARI fields under a few
    plausible keys to stay robust to the exact schema.
    """
    if result.overlap_fraction <= 0:
        return

    def _pluck(d: dict) -> None:
        if not isinstance(d, dict):
            return
        for ari_key in ("mean_ari", "ari", "aggregate_ari", "ari_mean"):
            if isinstance(d.get(ari_key), (int, float)):
                result.cross_proposal_ari = float(d[ari_key])
                break
        for nmi_key in ("mean_nmi", "nmi", "aggregate_nmi", "nmi_mean"):
            if isinstance(d.get(nmi_key), (int, float)):
                result.cross_proposal_nmi = float(d[nmi_key])
                break

    if state:
        _pluck((state.get("meta", {}) or {}).get("cross_proposal_metrics", {}) or {})
    if result.cross_proposal_ari is None:
        metrics_file = _latest_by_mtime(_resolve_dir(ws, "metrics"), "*.json")
        data = _read_json(metrics_file) if metrics_file else None
        if isinstance(data, dict):
            _pluck(data)
            # Some schemas nest per-pair under "pairs" with an aggregate at top.
            if result.cross_proposal_ari is None and isinstance(data.get("pairs"), list):
                aris = [
                    p["ari"]
                    for p in data["pairs"]
                    if isinstance(p, dict) and isinstance(p.get("ari"), (int, float))
                ]
                if aris:
                    result.cross_proposal_ari = sum(aris) / len(aris)


def _gather_cell(dataset: str, variant_name: str, *, seed: int) -> CellResult:
    variant = VARIANTS[variant_name]
    ws = workspace_for(dataset, seed=seed, variant=variant)
    result = CellResult(
        dataset=dataset,
        variant=variant_name,
        overlap_fraction=variant.fraction,
        method=method_for(variant),
        workspace=str(ws),
        completed=False,
    )
    if not ws.exists():
        result.missing_outputs.append("workspace dir")
        return result

    required = ["final_taxonomy.json", "taxonomy.md", "categories.json"]
    missing = [r for r in required if not (ws / r).exists()]
    if missing:
        result.missing_outputs.extend(missing)

    summary = _read_json(ws / "_overlap_sweep_summary.json")
    if summary:
        result.k_in_scope = summary.get("k_in_scope")
        result.k_actual = summary.get("k_actual")
        result.k_range = summary.get("k_range", [])
        result.orchestrator_wall_clock_s = summary.get("orchestrator_wall_clock_s")
    else:
        result.missing_outputs.append("_overlap_sweep_summary.json")

    state = _read_json(ws / "state.json")
    if state:
        meta = state.get("meta", {}) or {}
        result.n_proposers = meta.get("total_proposals")
        result.n_audits = meta.get("total_audits")
        result.n_investigations = meta.get("total_investigations")
        cov = meta.get("coverage") or {}
        mc = meta.get("mean_confidence") or {}
        if isinstance(cov, dict):
            result.audit_coverage = cov.get("value")
        if isinstance(mc, dict):
            result.audit_mean_confidence = mc.get("value")
        if result.k_actual is None and isinstance(state.get("clusters"), list):
            result.k_actual = len(state["clusters"])
    else:
        result.missing_outputs.append("state.json")

    syn_dir = _resolve_dir(ws, "investigations")
    n_synth = _count_glob(syn_dir, "synthesis_*.json") - _count_glob(
        syn_dir, "synthesis_*_clusters.json"
    )
    result.n_synthesizer_outputs = max(n_synth, 0)

    crit_dir = _resolve_dir(ws, "critiques")
    result.n_critiques = _count_glob(crit_dir, "critique_*.json")

    result.total_dispatches = sum(
        x or 0
        for x in (
            result.n_proposers,
            result.n_audits,
            result.n_investigations,
            result.n_synthesizer_outputs,
            result.n_critiques,
        )
    )

    crit_path = _latest_by_mtime(crit_dir, "critique_*.json")
    crit = _read_json(crit_path) if crit_path else None
    if crit:
        result.critic_finalize_recommendation = crit.get("finalize_recommendation")

    _gather_overlap_health(result, ws)
    _gather_cross_proposal(result, ws, state)

    if not result.missing_outputs:
        result.completed = True
    return result


# --------------------------------------------------------------------------- #
# LLM-as-judge taxonomy alignment (same rubric as the proposer sweep)
# --------------------------------------------------------------------------- #


def _load_gold_taxonomy(dataset: str) -> list[str]:
    path = DATA_DERIVED / dataset / "taxonomy.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [raw[k] for k in sorted(raw.keys(), key=lambda s: int(s))]


def _judge_prompt(*, dataset: str, gold_names: list[str], taxonomy_md: str) -> str:
    gold_block = "\n".join(f"- {name}" for name in gold_names)
    return f"""\
You are scoring how well a discovered cluster taxonomy aligns with a gold
taxonomy for the {dataset} dataset.

GOLD TAXONOMY ({len(gold_names)} categories):
{gold_block}

DISCOVERED TAXONOMY (name + description per cluster):
---
{taxonomy_md.strip()}
---

Score the alignment on this 1-5 rubric:
  5 — Excellent: every gold category maps cleanly to one distinct cluster.
  4 — Good: most gold categories covered, mapping mostly 1:1, a few merges/splits.
  3 — Mixed: notable merges/splits/missing categories.
  2 — Weak: organization differs substantially; many gold categories missing or fragmented.
  1 — Poor: little resemblance.

Be strict — wrong granularity (far more/fewer clusters than gold) should not
score 5 even if broad themes are right.

Reply with valid JSON only: {{"score": <1-5>, "rationale": "<1-3 sentences>"}}
"""


_JUDGE_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _judge_cell(cell: CellResult, *, dataset: str, gold_names: list[str]) -> None:
    if not cell.completed:
        return
    tax_path = Path(cell.workspace) / "taxonomy.md"
    if not tax_path.exists():
        return
    prompt = _judge_prompt(
        dataset=dataset, gold_names=gold_names, taxonomy_md=tax_path.read_text(encoding="utf-8")
    )
    log_prefix = f"[judge/{dataset}/{cell.variant}]"
    try:
        from benchmarking.llm_clients.claude_code import ClaudeCodeError

        stdout = call_claude(
            prompt,
            model=JUDGE_MODEL,
            timeout_s=60 * 10,
            log_prefix=log_prefix,
            extra_args=["--permission-mode", "bypassPermissions"],
        )
    except ClaudeCodeError as e:
        print(f"{log_prefix} claude-p failed ({e}); skipping judge for this cell")
        return
    match = _JUDGE_JSON_RE.search(stdout or "")
    if not match:
        print(f"{log_prefix} could not parse judge reply; raw: {stdout!r}")
        return
    try:
        parsed = json.loads(match.group(0))
        score = int(parsed["score"])
        if not 1 <= score <= 5:
            raise ValueError(f"score {score} out of range")
        cell.judge_score = score
        cell.judge_rationale = str(parsed.get("rationale", "")).strip()
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"{log_prefix} judge JSON malformed ({e}); raw: {match.group(0)!r}")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _fmt(v, fmt: str = "{:.3f}") -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return fmt.format(v)
    return str(v)


def _dk(r: CellResult) -> int | None:
    if r.k_actual is None or r.k_in_scope is None:
        return None
    return r.k_actual - r.k_in_scope


def _write_markdown(rows: list[CellResult], *, datasets: list[str], variant_order: list[str]) -> Path:
    out = OUT_DIR / "comparison.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    by: dict[tuple[str, str], CellResult] = {(r.dataset, r.variant): r for r in rows}
    lines: list[str] = []
    lines.append("# Proposer-overlap sweep — comparison\n")
    lines.append(
        "Four variants differing only in how much of each proposer's sample is "
        "shared with the others: v0 disjoint (production default), v1 25%, v2 "
        "50%, v3 100%. Discover-k (k_range = k_in_scope ± 20%), taxonomy stage "
        "only. Sample size stays an agentic decision; only the shared fraction "
        "is fixed.\n"
    )

    # Δk pivot — the headline.
    lines.append("## Δk pivot (k_actual − gold), by dataset × overlap\n")
    header = "| dataset | gold | " + " | ".join(
        f"{v} ({VARIANTS[v].fraction:.0%})" for v in variant_order
    ) + " |"
    lines.append(header)
    lines.append("|---" * (2 + len(variant_order)) + "|")
    sums = {v: 0 for v in variant_order}
    counts = {v: 0 for v in variant_order}
    for ds in datasets:
        cells = [by.get((ds, v)) for v in variant_order]
        gold = next((c.k_in_scope for c in cells if c and c.k_in_scope is not None), None)
        row = [ds, _fmt(gold)]
        for v, c in zip(variant_order, cells):
            dk = _dk(c) if c else None
            tag = ""
            if c and c.degraded:
                tag = "⚠"
            elif c and not c.completed:
                tag = "✗"
            row.append((f"{dk:+d}{tag}" if dk is not None else ("✗" if c else "—")))
            if dk is not None and c and c.completed and not c.degraded:
                sums[v] += abs(dk)
                counts[v] += 1
        lines.append("| " + " | ".join(row) + " |")
    # Σ|Δk| over datasets clean in ALL arms.
    clean_ds = [
        ds
        for ds in datasets
        if all(
            (by.get((ds, v)) and by[(ds, v)].completed and not by[(ds, v)].degraded)
            for v in variant_order
        )
    ]
    sig = {v: 0 for v in variant_order}
    for ds in clean_ds:
        for v in variant_order:
            dk = _dk(by[(ds, v)])
            if dk is not None:
                sig[v] += abs(dk)
    lines.append(
        f"| **Σ\\|Δk\\| (clean in all arms, n={len(clean_ds)})** | — | "
        + " | ".join(f"**{sig[v]}**" for v in variant_order)
        + " |"
    )
    lines.append("")
    lines.append(
        "⚠ = degraded cell (empty/short proposers — see detail); ✗ = did not "
        "finalize. Σ|Δk| excludes any dataset not clean in all four arms "
        f"(clean: {', '.join(clean_ds) or 'none'}).\n"
    )

    # Roll-up — one row per cell.
    lines.append("## Per-cell roll-up\n")
    lines.append(
        "| dataset | variant | Δk | realized ovlp | core | empty | x-prop ARI | "
        "cov | conf | disp | critic | judge | time(s) |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|")
    for ds in datasets:
        for v in variant_order:
            r = by.get((ds, v))
            if not r:
                continue
            dk = _dk(r)
            lines.append(
                "| "
                + " | ".join(
                    [
                        r.dataset,
                        f"{r.variant} ({r.overlap_fraction:.0%})",
                        (f"{dk:+d}" if dk is not None else "—"),
                        _fmt(r.realized_overlap_mean, "{:.0%}"),
                        _fmt(r.core_size),
                        _fmt(r.n_empty_proposals),
                        _fmt(r.cross_proposal_ari, "{:.3f}"),
                        _fmt(r.audit_coverage, "{:.2f}"),
                        _fmt(r.audit_mean_confidence, "{:.2f}"),
                        _fmt(r.total_dispatches),
                        r.critic_finalize_recommendation or "—",
                        _fmt(r.judge_score),
                        _fmt(r.orchestrator_wall_clock_s, "{:.0f}"),
                    ]
                )
                + " |"
            )
    lines.append("")

    # Flag degraded / missing cells explicitly.
    problems = [r for r in rows if r.degraded or not r.completed]
    if problems:
        lines.append("## Cells needing attention\n")
        for r in problems:
            why = r.degraded_reason or ("missing: " + ", ".join(r.missing_outputs))
            lines.append(f"- **{r.dataset}/{r.variant}** — {why}")
        lines.append("")

    lines.append("## Verdict (to be written after review)\n")
    lines.append("_(populated by the human reviewer based on the rows above)_\n")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _write_json(rows: list[CellResult]) -> Path:
    out = OUT_DIR / "comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps([asdict(r) for r in rows], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", nargs="+", choices=SWEEP_ORDER)
    parser.add_argument(
        "--variants", nargs="+", choices=list(VARIANTS.keys()), default=list(VARIANTS.keys())
    )
    parser.add_argument("--skip-judge", action="store_true")
    args = parser.parse_args()

    datasets = args.only or SWEEP_ORDER
    rows: list[CellResult] = []
    for ds in datasets:
        gold = _load_gold_taxonomy(ds) if not args.skip_judge else []
        for v in args.variants:
            print(f"[gather] {ds} / {v}")
            cell = _gather_cell(ds, v, seed=args.seed)
            if not args.skip_judge:
                _judge_cell(cell, dataset=ds, gold_names=gold)
            rows.append(cell)

    json_path = _write_json(rows)
    md_path = _write_markdown(rows, datasets=datasets, variant_order=args.variants)
    print(f"\nWrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
