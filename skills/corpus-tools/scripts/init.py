#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Initialize workspace from a corpus file.

Creates .claude/clustering/ directory structure and state.json with corpus stats.
"""

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path


def load_corpus(corpus_path: str, text_col: str) -> list[dict]:
    """Load corpus from CSV or JSON file."""
    path = Path(corpus_path)
    if not path.exists():
        print(f"Error: corpus file not found: {corpus_path}", file=sys.stderr)
        sys.exit(1)

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_csv(path, text_col)
    elif suffix == ".json":
        return _load_json(path, text_col)
    else:
        print(f"Error: unsupported file format: {suffix} (use .csv or .json)", file=sys.stderr)
        sys.exit(1)


def _load_csv(path: Path, text_col: str) -> list[dict]:
    records = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if text_col not in (reader.fieldnames or []):
            print(f"Error: column '{text_col}' not found. Available: {reader.fieldnames}", file=sys.stderr)
            sys.exit(1)
        for i, row in enumerate(reader):
            text = row[text_col]
            if text and text.strip():
                records.append({"id": str(i), "text": text.strip()})
    return records


def _load_json(path: Path, text_col: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        records = []
        for i, item in enumerate(data):
            if isinstance(item, dict) and text_col in item:
                text = str(item[text_col]).strip()
                if text:
                    item_id = str(item.get("id", i))
                    records.append({"id": item_id, "text": text})
            elif isinstance(item, str):
                records.append({"id": str(i), "text": item.strip()})
        return records
    else:
        print("Error: JSON file must contain a list of objects or strings", file=sys.stderr)
        sys.exit(1)



def compute_stats(records: list[dict]) -> dict:
    """Compute text length statistics."""
    lengths = [len(r["text"]) for r in records]
    lengths.sort()
    n = len(lengths)
    p95_idx = int(n * 0.95)
    return {
        "avg_length": round(statistics.mean(lengths)),
        "median_length": round(statistics.median(lengths)),
        "p95_length": lengths[min(p95_idx, n - 1)],
        "min_length": lengths[0],
        "max_length": lengths[-1],
    }


def main():
    parser = argparse.ArgumentParser(description="Initialize clustering workspace")
    parser.add_argument("--corpus", required=True, help="Path to corpus file (CSV or JSON)")
    parser.add_argument("--text-col", required=True, help="Column/field name containing text")
    parser.add_argument("--k-range", nargs=2, type=int, required=True, metavar=("MIN", "MAX"),
                        help="Target cluster count range")
    parser.add_argument("--model-tier", default="quality", choices=["quality", "balanced", "economy"],
                        help="Model tier for agent dispatch (default: quality)")
    parser.add_argument("--instructions", default="", help="Domain-specific instructions")
    parser.add_argument("--workspace", default=".claude/clustering",
                        help="Workspace directory (default: .claude/clustering)")
    parser.add_argument("--max-texts-per-sample", type=int, default=None,
                        help="Hard cap on texts per sample (optional)")
    args = parser.parse_args()

    corpus_path = os.path.abspath(args.corpus)
    records = load_corpus(corpus_path, args.text_col)

    if not records:
        print("Error: no valid texts found in corpus", file=sys.stderr)
        sys.exit(1)

    stats = compute_stats(records)

    # Create workspace directory structure
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "proposals").mkdir(exist_ok=True)
    (workspace / "audits").mkdir(exist_ok=True)
    (workspace / "investigations").mkdir(exist_ok=True)
    (workspace / "tfidf_cache").mkdir(exist_ok=True)
    (workspace / "metrics").mkdir(exist_ok=True)

    # Save corpus data for sampling/searching
    corpus_store = workspace / "corpus.json"
    with open(corpus_store, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    # Initialize state.json
    state = {
        "corpus": {
            "path": corpus_path,
            "text_column": args.text_col,
            "size": len(records),
            "stats": stats,
        },
        "config": {
            "k_range": args.k_range,
            "instructions": args.instructions,
            "model_tier": args.model_tier,
            "workspace_path": str(workspace),
            "max_texts_per_sample": args.max_texts_per_sample,
        },
        "clusters": [],
        "meta": {
            "cluster_version": 0,
            "next_cluster_id": 1,
            "total_texts_sampled": 0,
            "total_proposals": 0,
            "total_audits": 0,
            "total_investigations": 0,
            "coverage": None,
            "mean_confidence": None,
            "rejected_hypotheses": [],
            "open_questions": [],
            "last_action": "initialized workspace",
        },
    }

    state_path = workspace / "state.json"
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    # Initialize seen_ids
    seen_path = workspace / "seen_ids.json"
    with open(seen_path, "w", encoding="utf-8") as f:
        json.dump([], f)

    # Initialize empty log
    log_path = workspace / "log.jsonl"
    log_path.touch()

    # Write .plugin_root for fallback resolution of $CLAUDE_PLUGIN_ROOT
    # Walk up from this script's location: scripts/ -> corpus-tools/ -> skills/ -> plugin root
    plugin_root = Path(__file__).resolve().parent.parent.parent.parent
    plugin_root_file = workspace / ".plugin_root"
    plugin_root_file.write_text(str(plugin_root), encoding="utf-8")

    # Generate initial summary
    _generate_summary(state, workspace)

    # Log the init action
    _log_action(workspace, "init", f"Initialized workspace with {len(records)} texts")

    # Print results
    print(f"Workspace initialized at {workspace}/")
    print(f"")
    print(f"Corpus: {corpus_path}")
    print(f"  Texts: {len(records)}")
    print(f"  Avg length: {stats['avg_length']} chars")
    print(f"  Median length: {stats['median_length']} chars")
    print(f"  P95 length: {stats['p95_length']} chars")
    print(f"  Min/Max: {stats['min_length']}/{stats['max_length']} chars")
    print(f"")
    print(f"Config:")
    print(f"  k_range: {args.k_range[0]}-{args.k_range[1]}")
    print(f"  model_tier: {args.model_tier}")
    if args.instructions:
        print(f"  instructions: {args.instructions}")


def _generate_summary(state: dict, workspace: Path):
    """Generate summary.md from state."""
    lines = ["# Clustering Workspace Summary", ""]
    corpus = state["corpus"]
    lines.append(f"## Corpus")
    lines.append(f"- **Path**: {corpus['path']}")
    lines.append(f"- **Size**: {corpus['size']} texts")
    lines.append(f"- **Avg length**: {corpus['stats']['avg_length']} chars")
    lines.append(f"- **Median length**: {corpus['stats']['median_length']} chars")
    lines.append(f"- **P95 length**: {corpus['stats']['p95_length']} chars")
    lines.append("")

    config = state["config"]
    lines.append(f"## Config")
    lines.append(f"- **k_range**: {config['k_range'][0]}-{config['k_range'][1]}")
    lines.append(f"- **Model tier**: {config['model_tier']}")
    if config.get("instructions"):
        lines.append(f"- **Instructions**: {config['instructions']}")
    lines.append("")

    meta = state["meta"]
    lines.append(f"## Progress")
    lines.append(f"- **Cluster version**: {meta['cluster_version']}")
    lines.append(f"- **Texts sampled**: {meta['total_texts_sampled']}")
    lines.append(f"- **Proposals**: {meta['total_proposals']}")
    lines.append(f"- **Audits**: {meta['total_audits']}")
    lines.append(f"- **Investigations**: {meta['total_investigations']}")
    lines.append("")

    if state["clusters"]:
        lines.append(f"## Clusters ({len(state['clusters'])})")
        for c in state["clusters"]:
            conf = c.get("confidence", "unaudited")
            lines.append(f"- **{c['id']}**: {c['name']} [{conf}]")
            lines.append(f"  {c['description']}")
        lines.append("")

    if meta.get("coverage") and meta["coverage"].get("value") is not None:
        cov = meta["coverage"]
        lines.append(f"## Metrics")
        lines.append(f"- **Coverage**: ~{cov['value']:.0%} (estimated from N={cov['sample_size']} {cov.get('sample_method', 'random')} sample)")
        if meta.get("mean_confidence") and meta["mean_confidence"].get("value") is not None:
            mc = meta["mean_confidence"]
            lines.append(f"- **Mean confidence**: {mc['value']:.1f} (N={mc['sample_size']})")
        lines.append("")

    if meta.get("rejected_hypotheses"):
        lines.append("## Rejected Hypotheses")
        for rh in meta["rejected_hypotheses"]:
            lines.append(f"- {rh['hypothesis']} → {rh['finding']}")
        lines.append("")

    if meta.get("open_questions"):
        lines.append("## Open Questions")
        for q in meta["open_questions"]:
            lines.append(f"- {q}")
        lines.append("")

    # Recent log entries
    log_path = workspace / "log.jsonl"
    if log_path.exists():
        log_lines = log_path.read_text().strip().split("\n")
        log_lines = [l for l in log_lines if l.strip()]
        recent = log_lines[-5:] if len(log_lines) > 5 else log_lines
        if recent:
            lines.append("## Recent Actions")
            for entry_str in recent:
                entry = json.loads(entry_str)
                lines.append(f"- [{entry.get('timestamp', '?')}] {entry.get('action', '?')}: {entry.get('detail', '')}")
            lines.append("")

    summary_path = workspace / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def _log_action(workspace: Path, action: str, detail: str):
    """Append an action to log.jsonl."""
    from datetime import datetime, timezone
    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "action": action,
        "detail": detail,
    }
    log_path = workspace / "log.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
