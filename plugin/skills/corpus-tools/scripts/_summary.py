"""Shared summary.md generator.

Both `init.py` (one-shot at workspace creation) and `state.py` (called by the
SubagentStop hook to refresh after every agent stop) write `summary.md` from
the current state.json. Centralising the markdown layout here keeps the two
paths from drifting — they used to maintain near-identical copies that did
drift (state.py grew cross-proposal-metrics and per-cluster audit_info; init.py
didn't).

Stdlib only (no third-party deps); safe to import from any PEP 723 script in
this directory. Note: import has one side effect — it reconfigures
`sys.stdout`/`sys.stderr` to UTF-8 to match the rest of the codebase
(idempotent, no-op on already-reconfigured or non-TextIOWrapper streams).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Force UTF-8 on stdout/stderr — Windows defaults to cp1252 and crashes on
# non-ASCII cluster names / corpus content. Idempotent; no-op on streams that
# aren't TextIOWrapper (e.g. captured in tests).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from _audit_metrics import MIN_AUDIT_N_PER_CLUSTER, audit_power


def render_summary(
    state: dict,
    log_path: Path | None = None,
    recent_n: int = 5,
) -> str:
    """Build summary.md content from state. Returns the markdown as a string.

    If ``log_path`` is provided and exists, the last ``recent_n`` log entries
    are appended as a "Recent Actions" section.
    """
    lines = ["# Clustering Workspace Summary", ""]

    corpus = state["corpus"]
    lines.append("## Corpus")
    lines.append(f"- **Path**: {corpus['path']}")
    lines.append(f"- **Size**: {corpus['size']} texts")
    lines.append(f"- **Avg length**: {corpus['stats']['avg_length']} chars")
    lines.append(f"- **Median length**: {corpus['stats']['median_length']} chars")
    lines.append(f"- **P95 length**: {corpus['stats']['p95_length']} chars")
    lines.append("")

    config = state["config"]
    lines.append("## Config")
    lines.append(f"- **k_range**: {config['k_range'][0]}-{config['k_range'][1]}")
    lines.append(f"- **Model tier**: {config['model_tier']}")
    if config.get("instructions"):
        lines.append(f"- **Instructions**: {config['instructions']}")
    lines.append("")

    meta = state["meta"]
    lines.append("## Progress")
    lines.append(f"- **Cluster version**: {meta['cluster_version']}")
    lines.append(f"- **Texts sampled**: {meta['total_texts_sampled']}")
    lines.append(f"- **Proposals**: {meta['total_proposals']}")
    lines.append(f"- **Audits**: {meta['total_audits']}")
    lines.append(f"- **Investigations**: {meta['total_investigations']}")
    lines.append(f"- **Critiques**: {meta.get('total_critiques', 0)}")
    lines.append("")

    if state["clusters"]:
        lines.append(f"## Clusters ({len(state['clusters'])})")
        for c in state["clusters"]:
            conf = c.get("confidence", "unaudited")
            evidence = c.get("evidence", {})
            audit_info = ""
            if evidence.get("audit_assignments"):
                audit_info = (
                    f" (N={evidence['audit_assignments']}, "
                    f"mean_conf={evidence.get('audit_mean_confidence', '?')})"
                )
            lines.append(f"- **{c['id']}**: {c['name']} [{conf}]{audit_info}")
            lines.append(f"  {c['description']}")
        lines.append("")

        # Per-cluster audit power. The orchestrator needs this to tell a
        # genuinely weak cluster apart from one that simply hasn't been
        # sampled enough — a uniform draw gives the rarest clusters the
        # fewest texts, which is backwards for measuring their fit.
        #
        # Gated on at least one audit having run: before that every cluster is
        # trivially below the floor, and saying so would push the orchestrator
        # toward a stratified draw before it has taken a coverage draw.
        power = audit_power(state["clusters"])
        if meta.get("total_audits", 0) and power["below_floor"]:
            def _fmt(rows: list[dict], limit: int = 12) -> str:
                out = ", ".join(f"{d['id']}(n={d['n']})" for d in rows[:limit])
                if len(rows) > limit:
                    out += f" +{len(rows) - limit} more"
                return out

            lines.append("## Per-Cluster Audit Sample")
            lines.append(
                f"- **Median n**: {power['median_n']} "
                f"(floor for a publishable confidence label: "
                f"n>={MIN_AUDIT_N_PER_CLUSTER})"
            )
            if power["needs_audit"]:
                lines.append(
                    f"- **Under-sampled** ({len(power['needs_audit'])}/"
                    f"{power['n_clusters']}): {_fmt(power['needs_audit'])}"
                )
                lines.append(
                    f"  These have not been searched for. Draw for them with "
                    f"`sample.py --strategy stratified --per-cluster "
                    f"{MIN_AUDIT_N_PER_CLUSTER}` (~{power['total_needed']} more "
                    f"assignments), audited with `\"sample_basis\": \"stratified\"` "
                    f"and its strata manifest in `strata_file`. **Audit these.**"
                )
            if power["unsupported"]:
                lines.append(
                    f"- **Unsupported** ({len(power['unsupported'])}/"
                    f"{power['n_clusters']}): "
                    + ", ".join(
                        f"{d['id']}(n={d['n']} of {d['targeted']} aimed)"
                        for d in power["unsupported"][:12]
                    )
                )
                lines.append(
                    "  Candidates were drawn for these and the auditor assigned "
                    "them elsewhere, so repeating the same draw won't move them. "
                    "**Investigate these, don't re-audit.** Find where the "
                    "candidates went: a neighbour absorbing them argues for a "
                    "merge or a sharper boundary; nothing absorbing them argues "
                    "the corpus doesn't support the cluster."
                )
            lines.append("")

    if (
        meta.get("coverage")
        and isinstance(meta["coverage"], dict)
        and meta["coverage"].get("value") is not None
    ):
        cov = meta["coverage"]
        lines.append("## Metrics")
        pct = f"{cov['value']:.0%}" if isinstance(cov["value"], float) else str(cov["value"])
        lines.append(
            f"- **Coverage**: ~{pct} (computed from N={cov['sample_size']} "
            f"{cov.get('sample_method', 'random')} sample)"
        )
        if (
            meta.get("mean_confidence")
            and isinstance(meta["mean_confidence"], dict)
            and meta["mean_confidence"].get("value") is not None
        ):
            mc = meta["mean_confidence"]
            lines.append(f"- **Mean confidence**: {mc['value']:.1f} (N={mc['sample_size']})")
        lines.append("")

    if meta.get("cross_proposal_metrics") and isinstance(meta["cross_proposal_metrics"], dict):
        cp = meta["cross_proposal_metrics"]
        lines.append("## Cross-Proposal Agreement")
        if cp.get("mean_ari") is not None:
            lines.append(f"- **Mean ARI**: {cp['mean_ari']:.3f}")
        if cp.get("overall_element_similarity") is not None:
            lines.append(f"- **Element similarity**: {cp['overall_element_similarity']:.3f}")
        if cp.get("n_inconsistent_texts") is not None:
            lines.append(f"- **Inconsistent texts**: {cp['n_inconsistent_texts']} identified")
        if cp.get("file"):
            lines.append(f"- Full report: {cp['file']}")
        lines.append("")

    if meta.get("rejected_hypotheses"):
        lines.append("## Rejected Hypotheses")
        for rh in meta["rejected_hypotheses"]:
            lines.append(f"- {rh['hypothesis']} -> {rh['finding']}")
        lines.append("")

    if meta.get("open_questions"):
        lines.append("## Open Questions")
        for q in meta["open_questions"]:
            lines.append(f"- {q}")
        lines.append("")

    if log_path is not None and log_path.exists():
        log_lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        log_lines = [l for l in log_lines if l.strip()]
        recent = log_lines[-recent_n:] if len(log_lines) > recent_n else log_lines
        if recent:
            lines.append("## Recent Actions")
            for entry_str in recent:
                try:
                    entry = json.loads(entry_str)
                    lines.append(
                        f"- [{entry.get('timestamp', '?')}] "
                        f"{entry.get('action', '?')}: {entry.get('detail', '')}"
                    )
                except json.JSONDecodeError:
                    pass
            lines.append("")

    return "\n".join(lines)
