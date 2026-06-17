"""Tabulate the sample-size sweep cells alongside the 1.0x baseline.

Reads:
  - v1 (0.5x), v2 (2.0x) workspaces produced by
    ``benchmarking.experiments.run_sample_size_sweep``
  - the 1.0x baseline = proposer-count sweep's ``seed=0_proposers_v2``
    workspaces (the SKILL.md "6-7 proposers" default these runs inherited).

Emits two artefacts at ``results/sample_size_sweep/``:
  - ``comparison.json`` — machine-readable, one row per (dataset, variant) cell
  - ``comparison.md``   — per-dataset section + roll-up table + verdict hook,
                          matching the layout of the proposer-count sweep.

Same signal set as ``compare_proposer_sweep.py`` (Δk, audit coverage,
mean_confidence, dispatch breakdown, wall clock); LLM-as-judge is skipped by
default since the structural comparison is the deliverable.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from benchmarking.baselines.agentic_sample_size_sweep import (
    SWEEP_ORDER,
    VARIANTS as SAMPLE_SIZE_VARIANTS,
    method_for as sample_size_method_for,
    workspace_for as sample_size_workspace_for,
)
from benchmarking.baselines.agentic_proposer_sweep import (
    VARIANTS as PROPOSER_VARIANTS,
    workspace_for as proposer_workspace_for,
)
from benchmarking.paths import RESULTS

OUT_DIR = RESULTS / "sample_size_sweep"

# Display order: 0.5x → 1.0x baseline → 2.0x, so the U-shape is visually obvious.
VARIANT_DISPLAY_ORDER = ["v1", "baseline", "v2"]


@dataclass
class CellResult:
    dataset: str
    variant: str  # one of v1 / baseline / v2
    label: str    # e.g. "0.5x", "1.0x (baseline)", "2.0x"
    method: str
    workspace: str
    completed: bool
    k_in_scope: int | None = None
    k_actual: int | None = None
    k_range: list[int] = field(default_factory=list)
    audit_coverage: float | None = None
    audit_mean_confidence: float | None = None
    audit_sample_size: int | None = None
    critic_finalize_recommendation: str | None = None
    critic_overall_assessment: str | None = None
    n_proposers: int | None = None
    n_audits: int | None = None
    n_investigations: int | None = None
    n_synthesizer_outputs: int | None = None
    n_critiques: int | None = None
    total_dispatches: int | None = None
    orchestrator_wall_clock_s: float | None = None
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


def _resolve_cell(dataset: str, variant: str, *, seed: int) -> tuple[Path, str, str, str]:
    """Returns (workspace_dir, summary_filename, method_name, display_label)."""
    if variant == "baseline":
        ws = proposer_workspace_for(dataset, seed=seed, variant=PROPOSER_VARIANTS["v2"])
        return ws, "_proposer_sweep_summary.json", "agentic_clustering (1.0x baseline)", "1.0x"
    v = SAMPLE_SIZE_VARIANTS[variant]
    ws = sample_size_workspace_for(dataset, seed=seed, variant=v)
    label = f"{v.factor}x"
    return ws, "_sample_size_sweep_summary.json", sample_size_method_for(v), label


def _gather_cell(dataset: str, variant: str, *, seed: int) -> CellResult:
    ws, summary_name, method_name, label = _resolve_cell(dataset, variant, seed=seed)
    result = CellResult(
        dataset=dataset, variant=variant, label=label, method=method_name,
        workspace=str(ws), completed=False,
    )
    if not ws.exists():
        result.missing_outputs.append("workspace dir")
        return result

    required = ["final_taxonomy.json", "taxonomy.md", "categories.json"]
    missing = [r for r in required if not (ws / r).exists()]
    if missing:
        result.missing_outputs.extend(missing)

    summary = _read_json(ws / summary_name)
    if summary:
        result.k_in_scope = summary.get("k_in_scope")
        result.k_actual = summary.get("k_actual")
        result.k_range = summary.get("k_range", [])
        result.orchestrator_wall_clock_s = summary.get("orchestrator_wall_clock_s")
    else:
        result.missing_outputs.append(summary_name)

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
            result.audit_sample_size = cov.get("sample_size")
        if isinstance(mc, dict):
            result.audit_mean_confidence = mc.get("value")
        if result.k_actual is None and isinstance(state.get("clusters"), list):
            result.k_actual = len(state["clusters"])
    else:
        result.missing_outputs.append("state.json")

    n_synth = _count_glob(ws / "investigations", "synthesis_*.json") - _count_glob(
        ws / "investigations", "synthesis_*_clusters.json"
    )
    if n_synth <= 0:
        n_synth = _count_glob(
            ws / "archive" / "investigations", "synthesis_*.json"
        ) - _count_glob(ws / "archive" / "investigations", "synthesis_*_clusters.json")
    result.n_synthesizer_outputs = max(n_synth, 0)

    n_crit = _count_glob(ws / "critiques", "critique_*.json")
    if n_crit == 0:
        n_crit = _count_glob(ws / "archive" / "critiques", "critique_*.json")
    if n_crit == 0:
        n_crit = _count_glob(ws / "investigations", "critique_*.json") + _count_glob(
            ws / "archive" / "investigations", "critique_*.json"
        )
    result.n_critiques = n_crit

    result.total_dispatches = sum(
        [
            result.n_proposers or 0,
            result.n_audits or 0,
            result.n_investigations or 0,
            result.n_synthesizer_outputs or 0,
            result.n_critiques or 0,
        ]
    )

    crit_path = (
        _latest_by_mtime(ws / "critiques", "critique_*.json")
        or _latest_by_mtime(ws / "archive" / "critiques", "critique_*.json")
        or _latest_by_mtime(ws / "investigations", "critique_*.json")
        or _latest_by_mtime(ws / "archive" / "investigations", "critique_*.json")
    )
    crit = _read_json(crit_path) if crit_path else None
    if crit:
        result.critic_finalize_recommendation = crit.get("finalize_recommendation")
        result.critic_overall_assessment = crit.get("overall_assessment")

    if not result.missing_outputs:
        result.completed = True
    return result


def _fmt(v, fmt: str = "{:.3f}") -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return fmt.format(v)
    return str(v)


def _write_markdown(rows: list[CellResult]) -> Path:
    out = OUT_DIR / "comparison.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Sample-size sweep — comparison\n")
    lines.append(
        "Two variants of the SKILL.md 'Sample sizes' rule, identical in every "
        "word except the three numeric bands (short/medium/long text). The "
        "1.0x baseline cell reuses the proposer-count sweep's "
        "``seed=0_proposers_v2`` workspaces (same SKILL.md defaults, including "
        "the now-default 6-7 proposers). All cells are discover-k "
        "(k_range = k_in_scope ± 20%), taxonomy stage only (no classification "
        "step).\n"
    )
    lines.append("| variant | factor | short (<100) | medium (100-500) | long (500+) |")
    lines.append("|---|---|---|---|---|")
    v1 = SAMPLE_SIZE_VARIANTS["v1"]
    v2 = SAMPLE_SIZE_VARIANTS["v2"]
    lines.append(
        f"| v1 | {v1.factor}x | {v1.short_band[0]}-{v1.short_band[1]} | "
        f"{v1.medium_band[0]}-{v1.medium_band[1]} | "
        f"{v1.long_band[0]}-{v1.long_band[1]} |"
    )
    lines.append(
        "| baseline | 1.0x | 200-400 | 50-150 | 20-50 |"
    )
    lines.append(
        f"| v2 | {v2.factor}x | {v2.short_band[0]}-{v2.short_band[1]} | "
        f"{v2.medium_band[0]}-{v2.medium_band[1]} | "
        f"{v2.long_band[0]}-{v2.long_band[1]} |"
    )
    lines.append("")

    lines.append("## Summary table\n")
    lines.append(
        "| dataset | variant | k_gold | k_act | Δk | cov | mean_conf | "
        "dispatches | critic | time (s) |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|---:|")
    for r in rows:
        dk = (
            r.k_actual - r.k_in_scope
            if (r.k_actual is not None and r.k_in_scope is not None)
            else None
        )
        critic = r.critic_finalize_recommendation or "—"
        lines.append(
            "| "
            + " | ".join(
                [
                    r.dataset,
                    f"{r.variant} ({r.label})",
                    _fmt(r.k_in_scope),
                    _fmt(r.k_actual),
                    _fmt(dk, "{:+d}") if dk is not None else "—",
                    _fmt(r.audit_coverage),
                    _fmt(r.audit_mean_confidence, "{:.2f}"),
                    _fmt(r.total_dispatches),
                    str(critic),
                    _fmt(r.orchestrator_wall_clock_s, "{:.0f}"),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Per-dataset detail\n")
    by_ds: dict[str, list[CellResult]] = {}
    for r in rows:
        by_ds.setdefault(r.dataset, []).append(r)
    for ds, cells in by_ds.items():
        lines.append(f"### {ds}\n")
        for c in cells:
            lines.append(
                f"**{c.variant} ({c.label})** — "
                f"k={_fmt(c.k_actual)} (gold {_fmt(c.k_in_scope)}), "
                f"cov={_fmt(c.audit_coverage)}, "
                f"mean_conf={_fmt(c.audit_mean_confidence, '{:.2f}')}, "
                f"dispatches={_fmt(c.total_dispatches)} "
                f"(P={_fmt(c.n_proposers)}, S={_fmt(c.n_synthesizer_outputs)}, "
                f"A={_fmt(c.n_audits)}, C={_fmt(c.n_critiques)}, "
                f"I={_fmt(c.n_investigations)}), "
                f"critic={c.critic_finalize_recommendation or '—'}"
            )
            if c.missing_outputs:
                lines.append(f"> ⚠ missing: {', '.join(c.missing_outputs)}")
            lines.append("")
    lines.append("")

    lines.append("## Aggregates across all 21 cells (7 datasets × 3 variants)\n")
    agg_lines = ["| variant | Σ|Δk| | undershoots | mean cov | mean conf | mean dispatches | mean proposers | mean investigations | mean wall clock |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for v in VARIANT_DISPLAY_ORDER:
        v_rows = [r for r in rows if r.variant == v and r.completed]
        if not v_rows:
            continue
        sum_abs_dk = sum(
            abs((r.k_actual or 0) - (r.k_in_scope or 0)) for r in v_rows
        )
        undershoots = sum(
            1 for r in v_rows
            if r.k_actual is not None and r.k_in_scope is not None
            and r.k_actual < r.k_in_scope
        )
        def _mean(values):
            vals = [v for v in values if v is not None]
            return sum(vals) / len(vals) if vals else None
        mean_cov = _mean([r.audit_coverage for r in v_rows])
        mean_conf = _mean([r.audit_mean_confidence for r in v_rows])
        mean_disp = _mean([r.total_dispatches for r in v_rows])
        mean_props = _mean([r.n_proposers for r in v_rows])
        mean_invs = _mean([r.n_investigations for r in v_rows])
        mean_wall = _mean([r.orchestrator_wall_clock_s for r in v_rows])
        label = SAMPLE_SIZE_VARIANTS[v].factor if v != "baseline" else 1.0
        agg_lines.append(
            f"| {v} ({label}x) | {sum_abs_dk} | {undershoots} | "
            f"{_fmt(mean_cov)} | {_fmt(mean_conf, '{:.2f}')} | "
            f"{_fmt(mean_disp, '{:.1f}')} | {_fmt(mean_props, '{:.1f}')} | "
            f"{_fmt(mean_invs, '{:.1f}')} | "
            f"{_fmt(mean_wall/60 if mean_wall else None, '{:.0f}')} min |"
        )
    lines.extend(agg_lines)
    lines.append("")

    lines.append("Per-dataset Δk and per-row winner:\n")
    win_lines = ["| dataset | gold k | v1 Δk (0.5x) | baseline Δk (1.0x) | v2 Δk (2.0x) | best |",
                 "|---|---:|---:|---:|---:|---|"]
    for ds in SWEEP_ORDER:
        ds_rows = {r.variant: r for r in rows if r.dataset == ds}
        if not all(v in ds_rows for v in VARIANT_DISPLAY_ORDER):
            continue
        gold = ds_rows["v1"].k_in_scope
        dks = {v: (ds_rows[v].k_actual or 0) - (gold or 0) for v in VARIANT_DISPLAY_ORDER}
        best_v = min(dks, key=lambda v: abs(dks[v]))
        best_label = {"v1": "v1", "baseline": "baseline", "v2": "v2"}[best_v]
        win_lines.append(
            f"| {ds} | {gold} | {dks['v1']:+d} | {dks['baseline']:+d} | "
            f"{dks['v2']:+d} | {best_label} |"
        )
    lines.extend(win_lines)
    lines.append("")

    lines.append("## Verdict (to be written after reviewing the table)\n")
    lines.append(
        "_(populated by the human reviewer based on the per-dataset rows above)_\n"
    )

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
    parser.add_argument(
        "--only",
        nargs="+",
        choices=SWEEP_ORDER,
        help="Restrict to these datasets (default: all 7).",
    )
    args = parser.parse_args()

    datasets = args.only or SWEEP_ORDER
    rows: list[CellResult] = []
    for ds in datasets:
        for v in VARIANT_DISPLAY_ORDER:
            print(f"[gather] {ds} / {v}")
            rows.append(_gather_cell(ds, v, seed=args.seed))

    json_path = _write_json(rows)
    md_path = _write_markdown(rows)
    print(f"\nWrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
