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
