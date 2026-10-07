"""Tabulate the four proposer-count variants and judge taxonomy alignment.

Reads every (dataset, variant) workspace produced by
``benchmarking.experiments.run_proposer_sweep`` and emits two artefacts at
``results/proposer_sweep/``:

  - ``comparison.json`` — machine-readable, one row per (dataset, variant) cell
  - ``comparison.md``   — per-dataset section + a roll-up table + a written
                          verdict on the optimal variant.

Signals collected per cell (everything that is *valid* without classification —
we skip cross-proposal ARI because our proposers don't share samples by
default, so any inter-proposal disagreement conflates "different opinions"
with "different inputs"):

  * ``k_actual`` and its delta vs ``k_in_scope``
  * Final auditor coverage and mean_confidence (from state.meta, which the
    finalize step copies from the most recent audit)
  * Critic ``finalize_recommendation`` from the most recent critique JSON
  * Total agent dispatches = proposals + audits + investigations + synthesizer
    files + critique files (the cap-relevant counter)
  * Orchestrator wall clock (from _proposer_sweep_summary.json)
  * LLM-as-judge alignment score against each dataset's gold taxonomy

The LLM judge is dispatched via ``claude -p`` (Max subscription, no metered
spend) on the same Opus 4.7 we use for the agent loop. Each call is one
(dataset, variant) cell — 28 calls total when the full sweep is in.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from benchmarking.baselines.agentic_proposer_sweep import (
    SWEEP_ORDER,
    VARIANTS,
    method_for,
    workspace_for,
)
from benchmarking.llm_clients.claude_code import call_claude
from benchmarking.paths import DATA_DERIVED, RESULTS

OUT_DIR = RESULTS / "proposer_sweep"
JUDGE_MODEL = "claude-opus-4-7"


# --------------------------------------------------------------------------- #
# Data gathering
# --------------------------------------------------------------------------- #


@dataclass
class CellResult:
    dataset: str
    variant: str
    count_phrase: str
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


def _gather_cell(dataset: str, variant_name: str, *, seed: int) -> CellResult:
    variant = VARIANTS[variant_name]
    ws = workspace_for(dataset, seed=seed, variant=variant)
    result = CellResult(
        dataset=dataset,
        variant=variant_name,
        count_phrase=variant.count_phrase,
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

    summary = _read_json(ws / "_proposer_sweep_summary.json")
    if summary:
        result.k_in_scope = summary.get("k_in_scope")
        result.k_actual = summary.get("k_actual")
        result.k_range = summary.get("k_range", [])
        result.orchestrator_wall_clock_s = summary.get("orchestrator_wall_clock_s")
    else:
        result.missing_outputs.append("_proposer_sweep_summary.json")

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

    # Synthesizer / critique counts — neither is tracked in state.meta. Count
    # files instead. Synth outputs live under investigations/ as
    # synthesis_*.json (paired _clusters.json belongs to set-clusters input,
    # not a separate dispatch).
    n_synth = _count_glob(ws / "investigations", "synthesis_*.json") - _count_glob(
        ws / "investigations", "synthesis_*_clusters.json"
    )
    # Older runs stored under archive/investigations/ — try there if the live
    # dir is empty.
    if n_synth <= 0:
        n_synth = _count_glob(
            ws / "archive" / "investigations", "synthesis_*.json"
        ) - _count_glob(ws / "archive" / "investigations", "synthesis_*_clusters.json")
    result.n_synthesizer_outputs = max(n_synth, 0)

    n_crit = _count_glob(ws / "critiques", "critique_*.json")
    if n_crit == 0:
        n_crit = _count_glob(ws / "archive" / "critiques", "critique_*.json")
    if n_crit == 0:
        # Earlier runs sometimes wrote critiques alongside investigations.
        n_crit = _count_glob(ws / "investigations", "critique_*.json") + _count_glob(
            ws / "archive" / "investigations", "critique_*.json"
        )
    result.n_critiques = n_crit

    parts = [
        result.n_proposers or 0,
        result.n_audits or 0,
        result.n_investigations or 0,
        result.n_synthesizer_outputs or 0,
        result.n_critiques or 0,
    ]
    result.total_dispatches = sum(parts)

    # Most recent critique → finalize recommendation. Look in critiques/, then
    # archive/critiques/, then investigations/.
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


# --------------------------------------------------------------------------- #
# LLM-as-judge taxonomy alignment
# --------------------------------------------------------------------------- #


def _load_gold_taxonomy(dataset: str) -> list[str]:
    """Gold category names for a dataset, as a flat list."""
    path = DATA_DERIVED / dataset / "taxonomy.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    # The file is a dict {id_str: name_str}. Preserve gold-id order.
    return [raw[k] for k in sorted(raw.keys(), key=lambda s: int(s))]


def _judge_prompt(*, dataset: str, gold_names: list[str], taxonomy_md: str) -> str:
    gold_block = "\n".join(f"- {name}" for name in gold_names)
    return f"""\
You are scoring how well a discovered cluster taxonomy aligns with a gold
taxonomy for the {dataset} dataset.

GOLD TAXONOMY ({len(gold_names)} categories):
{gold_block}

DISCOVERED TAXONOMY (from a clustering algorithm — name + description per cluster):
---
{taxonomy_md.strip()}
---

Score the alignment on this 1-5 rubric:
  5 — Excellent: every gold category corresponds to a distinct cluster; almost
      every cluster maps cleanly to exactly one gold category; no significant
      gold gaps and no spurious extra clusters.
  4 — Good: most gold categories are covered by a distinct cluster, mapping is
      mostly 1:1, with a handful of merged/split/missing items.
  3 — Mixed: significant coverage of gold categories but with notable
      merges/splits/missing categories; many-to-one or one-to-many mappings
      common.
  2 — Weak: organization differs substantially from gold; many gold categories
      missing or fragmented; clusters often span multiple gold categories.
  1 — Poor: little resemblance to the gold organization.

Be strict — a discovered taxonomy with the wrong granularity (e.g. far more
or far fewer clusters than gold) should not score 5 even if the broad themes
are right.

Reply with valid JSON only, of the form:
{{"score": <1|2|3|4|5>, "rationale": "<1-3 sentence justification — name
specific gold categories that are missing, merged, or split; or specific
clusters that are spurious>"}}
"""


_JUDGE_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _judge_cell(cell: CellResult, *, dataset: str, gold_names: list[str]) -> None:
    """Mutates ``cell`` in place with judge_score / judge_rationale."""
    if not cell.completed:
        return
    tax_path = Path(cell.workspace) / "taxonomy.md"
    if not tax_path.exists():
        return
    taxonomy_md = tax_path.read_text(encoding="utf-8")
    prompt = _judge_prompt(
        dataset=dataset, gold_names=gold_names, taxonomy_md=taxonomy_md
    )
    log_prefix = f"[judge/{dataset}/{cell.variant}]"
    # Catch ClaudeCodeError (spend cap, transient claude-p failure) so a mid-
    # run cap-hit doesn't lose the structural comparison data for cells we
    # already processed. Skip the judge for this cell, leave judge_score=None,
    # carry on.
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
        print(f"{log_prefix} could not parse judge reply; raw stdout: {stdout!r}")
        return
    try:
        parsed = json.loads(match.group(0))
        score = int(parsed["score"])
        if score < 1 or score > 5:
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


def _write_markdown(rows: list[CellResult], *, seed: int) -> Path:
    out = OUT_DIR / "comparison.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Proposer-count sweep — comparison\n")
    lines.append(
        "Four variants of the SKILL.md 'Number of proposals' rule, identical "
        "in every word except the count phrase. All four runs are "
        "discover-k (k_range = k_in_scope ± 20%), taxonomy stage only "
        "(no classification step).\n"
    )
    lines.append("| variant | count phrase | description |")
    lines.append("|---|---|---|")
    for v in VARIANTS.values():
        lines.append(f"| {v.name} | {v.count_phrase} | {v.description} |")
    lines.append("")

    # Roll-up table — one row per cell.
    lines.append("## Summary table\n")
    lines.append(
        "| dataset | variant | k_gold | k_act | Δk | cov | mean_conf | "
        "dispatches | critic | judge | time (s) |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|"
    )
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
                    f"{r.variant} ({r.count_phrase})",
                    _fmt(r.k_in_scope),
                    _fmt(r.k_actual),
                    _fmt(dk, "{:+d}") if dk is not None else "—",
                    _fmt(r.audit_coverage),
                    _fmt(r.audit_mean_confidence, "{:.2f}"),
                    _fmt(r.total_dispatches),
                    str(critic),
                    _fmt(r.judge_score),
                    _fmt(r.orchestrator_wall_clock_s, "{:.0f}"),
                ]
            )
            + " |"
        )
    lines.append("")

    # Per-dataset detail sections.
    lines.append("## Per-dataset detail\n")
    by_ds: dict[str, list[CellResult]] = {}
    for r in rows:
        by_ds.setdefault(r.dataset, []).append(r)
    for ds, cells in by_ds.items():
        lines.append(f"### {ds}\n")
        for c in cells:
            lines.append(
                f"**{c.variant} ({c.count_phrase})** — "
                f"k={_fmt(c.k_actual)} (gold {_fmt(c.k_in_scope)}), "
                f"cov={_fmt(c.audit_coverage)}, "
                f"mean_conf={_fmt(c.audit_mean_confidence, '{:.2f}')}, "
                f"dispatches={_fmt(c.total_dispatches)} "
                f"(P={_fmt(c.n_proposers)}, S={_fmt(c.n_synthesizer_outputs)}, "
                f"A={_fmt(c.n_audits)}, C={_fmt(c.n_critiques)}, "
                f"I={_fmt(c.n_investigations)}), "
                f"critic={c.critic_finalize_recommendation or '—'}, "
                f"judge={_fmt(c.judge_score)}/5"
            )
            if c.judge_rationale:
                lines.append(f"> {c.judge_rationale}")
            if c.missing_outputs:
                lines.append(f"> ⚠ missing: {', '.join(c.missing_outputs)}")
            lines.append("")
    lines.append("")

    # Verdict — leave a hook the human/LLM filler can populate. We don't
    # auto-pick the winner here because the right answer depends on whether
    # judge_score, Δk, or coverage matters most for the paper.
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
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS.keys()),
        default=list(VARIANTS.keys()),
        help="Which variants to include (default: all four).",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip the LLM-as-judge step (still writes table + JSON).",
    )
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
    md_path = _write_markdown(rows, seed=args.seed)
    print(f"\nWrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
