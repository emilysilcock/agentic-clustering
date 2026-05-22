#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["filelock"]
# ///
"""State management for the clustering workspace.

Handles CRUD on state.json, summary.md generation, audit integration,
and recommendation application. All writes use file locking.
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock


def _get_workspace() -> Path:
    env_ws = os.environ.get("CLUSTERING_WORKSPACE")
    if env_ws:
        return Path(env_ws)
    return Path(".claude/clustering")


WORKSPACE = _get_workspace()
LOCK_PATH = WORKSPACE / ".state.lock"
STATE_PATH = WORKSPACE / "state.json"
LOG_PATH = WORKSPACE / "log.jsonl"


def load_state() -> dict:
    if not STATE_PATH.exists():
        print("Error: workspace not initialized. Run init.py first.", file=sys.stderr)
        sys.exit(1)
    with open(STATE_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def log_action(action: str, detail: str, metadata: dict | None = None):
    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "action": action,
        "detail": detail,
    }
    if metadata:
        entry["metadata"] = metadata
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def generate_summary(state: dict):
    """Generate summary.md from current state."""
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
    lines.append("")

    if state["clusters"]:
        lines.append(f"## Clusters ({len(state['clusters'])})")
        for c in state["clusters"]:
            conf = c.get("confidence", "unaudited")
            evidence = c.get("evidence", {})
            audit_info = ""
            if evidence.get("audit_assignments"):
                audit_info = f" (N={evidence['audit_assignments']}, mean_conf={evidence.get('audit_mean_confidence', '?')})"
            lines.append(f"- **{c['id']}**: {c['name']} [{conf}]{audit_info}")
            lines.append(f"  {c['description']}")
        lines.append("")

    if meta.get("coverage") and isinstance(meta["coverage"], dict) and meta["coverage"].get("value") is not None:
        cov = meta["coverage"]
        lines.append("## Metrics")
        pct = f"{cov['value']:.0%}" if isinstance(cov['value'], float) else str(cov['value'])
        lines.append(f"- **Coverage**: ~{pct} (estimated from N={cov['sample_size']} {cov.get('sample_method', 'random')} sample)")
        if meta.get("mean_confidence") and isinstance(meta["mean_confidence"], dict) and meta["mean_confidence"].get("value") is not None:
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

    # Recent log entries
    if LOG_PATH.exists():
        log_lines = LOG_PATH.read_text().strip().split("\n")
        log_lines = [l for l in log_lines if l.strip()]
        recent = log_lines[-5:] if len(log_lines) > 5 else log_lines
        if recent:
            lines.append("## Recent Actions")
            for entry_str in recent:
                try:
                    entry = json.loads(entry_str)
                    lines.append(f"- [{entry.get('timestamp', '?')}] {entry.get('action', '?')}: {entry.get('detail', '')}")
                except json.JSONDecodeError:
                    pass
            lines.append("")

    summary_path = WORKSPACE / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"summary.md updated ({len(state.get('clusters', []))} clusters)")


def cmd_summarize(_args):
    """Regenerate summary.md from state.json."""
    lock = FileLock(str(LOCK_PATH))
    with lock:
        state = load_state()
        generate_summary(state)


def cmd_set_clusters(args):
    """Set clusters from a synthesizer output file."""
    clusters_file = Path(args.file)
    if not clusters_file.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    with open(clusters_file, encoding="utf-8") as f:
        data = json.load(f)

    clusters_input = data.get("clusters", [])
    if not clusters_input:
        print("Error: no clusters found in input file", file=sys.stderr)
        sys.exit(1)

    lock = FileLock(str(LOCK_PATH))
    with lock:
        state = load_state()
        next_id = state["meta"]["next_cluster_id"]

        # Build lookup of existing clusters by normalized name for evidence preservation
        def _normalize(name: str) -> str:
            return name.strip().lower()

        existing_by_name = {_normalize(c["name"]): c for c in state.get("clusters", [])}

        new_clusters = []
        for c in clusters_input:
            text_id_list = c.get("evidence_text_ids",
                                 c.get("example_ids",
                                       c.get("text_ids",
                                             c.get("example_text_ids", []))))

            # When input has no text_ids, preserve evidence from existing cluster
            existing = existing_by_name.get(_normalize(c["name"]))
            if not text_id_list and existing:
                old_evidence = existing.get("evidence", {})
                cluster = {
                    "id": f"c{next_id}",
                    "name": c["name"],
                    "description": c["description"],
                    "confidence": existing.get("confidence", c.get("confidence", "unaudited")),
                    "evidence": {
                        "proposed_in": old_evidence.get("proposed_in", c.get("source_proposals", [])),
                        "evidence_text_ids": old_evidence.get("evidence_text_ids", []),
                        "audit_assignments": old_evidence.get("audit_assignments", 0),
                        "audit_mean_confidence": old_evidence.get("audit_mean_confidence"),
                        "total_texts_seen": old_evidence.get("total_texts_seen", 0),
                    },
                    "status": existing.get("status", "new"),
                }
            else:
                cluster = {
                    "id": f"c{next_id}",
                    "name": c["name"],
                    "description": c["description"],
                    "confidence": c.get("confidence", "unaudited"),
                    "evidence": {
                        "proposed_in": c.get("source_proposals", []),
                        "evidence_text_ids": text_id_list,
                        "audit_assignments": 0,
                        "audit_mean_confidence": None,
                        "total_texts_seen": len(text_id_list),
                    },
                    "status": "new",
                }
            new_clusters.append(cluster)
            next_id += 1

        state["clusters"] = new_clusters
        state["meta"]["next_cluster_id"] = next_id
        state["meta"]["cluster_version"] += 1
        # Reset metrics since cluster set changed
        state["meta"]["coverage"] = None
        state["meta"]["mean_confidence"] = None
        state["meta"]["last_action"] = f"set-clusters: {len(new_clusters)} clusters (v{state['meta']['cluster_version']})"

        save_state(state)
        generate_summary(state)
        log_action("set-clusters", f"{len(new_clusters)} clusters set (version {state['meta']['cluster_version']})")

    print(f"Set {len(new_clusters)} clusters (version {state['meta']['cluster_version']})")


def cmd_count_proposal(_args):
    """Increment the total_proposals counter."""
    lock = FileLock(str(LOCK_PATH))
    with lock:
        state = load_state()
        state["meta"]["total_proposals"] = state["meta"].get("total_proposals", 0) + 1
        save_state(state)
    print(f"Proposals: {state['meta']['total_proposals']}")


def cmd_count_investigation(_args):
    """Increment the total_investigations counter."""
    lock = FileLock(str(LOCK_PATH))
    with lock:
        state = load_state()
        state["meta"]["total_investigations"] = state["meta"].get("total_investigations", 0) + 1
        save_state(state)
    print(f"Investigations: {state['meta']['total_investigations']}")


def cmd_update_from_audit(args):
    """Update state with audit results."""
    audit_file = Path(args.file)
    if not audit_file.exists():
        print(f"Error: audit file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    with open(audit_file, encoding="utf-8") as f:
        audit = json.load(f)

    lock = FileLock(str(LOCK_PATH))
    with lock:
        state = load_state()

        # Reject audits from a different cluster version
        audit_version = audit.get("cluster_definitions_version")
        current_version = state["meta"]["cluster_version"]
        if audit_version is not None and audit_version != current_version:
            print(
                f"Error: audit was for cluster version {audit_version}, "
                f"but current version is {current_version}. "
                f"This audit is stale and cannot be applied.",
                file=sys.stderr,
            )
            sys.exit(1)

        assignments = audit.get("assignments", [])
        summary = audit.get("summary", {})

        # Auto-detect 0-1 float scale and normalize to 1-5 integer scale
        all_conf_values = [a["confidence"] for a in assignments if a.get("confidence") is not None]
        if all_conf_values and max(all_conf_values) <= 1.0:
            print(
                "WARNING: Detected 0-1 float confidence scale. "
                "Normalizing to 1-5 integer scale (multiply by 5).",
                file=sys.stderr,
            )
            for a in assignments:
                if a.get("confidence") is not None:
                    a["confidence"] = round(a["confidence"] * 5, 1)
            # Also fix summary mean_confidence if present
            if summary.get("mean_confidence") is not None and summary["mean_confidence"] <= 1.0:
                summary["mean_confidence"] = round(summary["mean_confidence"] * 5, 1)

        # Update per-cluster confidence from audit
        cluster_assignments = {}
        for a in assignments:
            cid = a.get("cluster_id")
            if cid and a.get("confidence") is not None:
                cluster_assignments.setdefault(cid, []).append(a["confidence"])

        for cluster in state["clusters"]:
            cid = cluster["id"]
            if cid in cluster_assignments:
                confs = cluster_assignments[cid]
                mean_conf = sum(confs) / len(confs)

                # Accumulate with existing audit data
                existing_count = cluster.get("evidence", {}).get("audit_assignments", 0)
                existing_mean = cluster.get("evidence", {}).get("audit_mean_confidence")

                if existing_mean is not None and existing_count > 0:
                    total_count = existing_count + len(confs)
                    combined_mean = (existing_mean * existing_count + mean_conf * len(confs)) / total_count
                else:
                    total_count = len(confs)
                    combined_mean = mean_conf

                cluster.setdefault("evidence", {})
                cluster["evidence"]["audit_assignments"] = total_count
                cluster["evidence"]["audit_mean_confidence"] = round(combined_mean, 2)
                cluster["evidence"]["total_texts_seen"] = cluster["evidence"].get("total_texts_seen", 0) + len(confs)

                # Set confidence label
                if combined_mean >= 4.0:
                    cluster["confidence"] = "high"
                elif combined_mean >= 3.0:
                    cluster["confidence"] = "medium"
                else:
                    cluster["confidence"] = "low"

                cluster["status"] = "audited"

        # Update global metrics
        if summary:
            state["meta"]["coverage"] = {
                "value": summary.get("coverage_estimate"),
                "sample_size": summary.get("total", len(assignments)),
                "sample_method": audit.get("sample_method", "random, exclude-seen"),
                "cluster_version": state["meta"]["cluster_version"],
                "note": "Estimated from audit sample -- not a corpus-wide measurement",
            }
            state["meta"]["mean_confidence"] = {
                "value": summary.get("mean_confidence"),
                "sample_size": summary.get("total", len(assignments)),
                "cluster_version": state["meta"]["cluster_version"],
            }

        state["meta"]["total_audits"] += 1
        state["meta"]["last_action"] = f"audit: coverage ~{summary.get('coverage_estimate', '?')}, confidence {summary.get('mean_confidence', '?')}"

        save_state(state)
        generate_summary(state)
        log_action("update-from-audit", f"Processed {len(assignments)} assignments from {audit_file.name}")

    print(f"Updated state from audit ({len(assignments)} assignments)")


def cmd_apply_recommendation(args):
    """Apply an investigation recommendation to the cluster set."""
    rec_file = Path(args.file)
    if not rec_file.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    with open(rec_file, encoding="utf-8") as f:
        data = json.load(f)

    rec = data.get("recommendation", {})
    rec_type = rec.get("type")
    if not rec_type:
        print("Error: no recommendation type found in file", file=sys.stderr)
        sys.exit(1)

    lock = FileLock(str(LOCK_PATH))
    with lock:
        state = load_state()

        # Track whether this is a structural change that invalidates metrics
        structural_change = rec_type in ("merge", "split", "add", "remove")

        if rec_type == "merge":
            _apply_merge(state, rec)
        elif rec_type == "split":
            _apply_split(state, rec)
        elif rec_type == "rename":
            _apply_rename(state, rec)
        elif rec_type == "add":
            _apply_add(state, rec)
        elif rec_type == "remove":
            _apply_remove(state, rec)
        elif rec_type == "no_change":
            _apply_no_change(state, rec, data, rec_file.name)
        else:
            print(f"Error: unknown recommendation type: {rec_type}", file=sys.stderr)
            sys.exit(1)

        # Structural changes invalidate existing metrics
        if structural_change:
            state["meta"]["cluster_version"] += 1
            state["meta"]["coverage"] = None
            state["meta"]["mean_confidence"] = None

        save_state(state)
        generate_summary(state)
        log_action("apply-recommendation", f"{rec_type}: {rec.get('reasoning', '')[:100]}")

    print(f"Applied recommendation: {rec_type}")


def _apply_merge(state: dict, rec: dict):
    merge_info = rec.get("merge_into", {})
    surviving_id = merge_info.get("surviving_id")
    targets = rec.get("targets", [])

    if not surviving_id or not targets:
        print("Error: merge requires surviving_id and targets", file=sys.stderr)
        sys.exit(1)

    # Update surviving cluster
    for cluster in state["clusters"]:
        if cluster["id"] == surviving_id:
            if merge_info.get("name"):
                cluster["name"] = merge_info["name"]
            if merge_info.get("description"):
                cluster["description"] = merge_info["description"]
            cluster["status"] = "modified"
            break

    # Remove merged-away clusters
    ids_to_remove = [t for t in targets if t != surviving_id]
    state["clusters"] = [c for c in state["clusters"] if c["id"] not in ids_to_remove]
    state["meta"]["last_action"] = f"merge: {', '.join(ids_to_remove)} into {surviving_id}"


def _apply_split(state: dict, rec: dict):
    targets = rec.get("targets", [])
    split_into = rec.get("split_into", [])

    if not targets or not split_into:
        print("Error: split requires targets and split_into", file=sys.stderr)
        sys.exit(1)

    original_id = targets[0]

    # Verify the target cluster exists before mutating state
    original_cluster = None
    for cluster in state["clusters"]:
        if cluster["id"] == original_id:
            original_cluster = cluster
            break

    if original_cluster is None:
        print(f"Error: split target cluster '{original_id}' not found in current clusters", file=sys.stderr)
        sys.exit(1)

    # First new cluster keeps the original ID
    first_split = split_into[0]
    original_cluster["name"] = first_split["name"]
    original_cluster["description"] = first_split["description"]
    original_cluster["confidence"] = "unaudited"
    original_cluster["status"] = "modified"

    # Remaining splits get new IDs
    next_id = state["meta"]["next_cluster_id"]
    for split_def in split_into[1:]:
        new_cluster = {
            "id": f"c{next_id}",
            "name": split_def["name"],
            "description": split_def["description"],
            "confidence": "unaudited",
            "evidence": {
                "proposed_in": [],
                "evidence_text_ids": split_def.get("example_text_ids", []),
                "audit_assignments": 0,
                "audit_mean_confidence": None,
                "total_texts_seen": 0,
            },
            "status": "new",
        }
        state["clusters"].append(new_cluster)
        next_id += 1

    state["meta"]["next_cluster_id"] = next_id
    state["meta"]["last_action"] = f"split: {original_id} into {len(split_into)} clusters"


def _apply_rename(state: dict, rec: dict):
    targets = rec.get("targets", [])
    rename_to = rec.get("rename_to", {})

    if not targets or not rename_to:
        print("Error: rename requires targets and rename_to", file=sys.stderr)
        sys.exit(1)

    target_id = targets[0]
    for cluster in state["clusters"]:
        if cluster["id"] == target_id:
            if rename_to.get("name"):
                cluster["name"] = rename_to["name"]
            if rename_to.get("description"):
                cluster["description"] = rename_to["description"]
            cluster["status"] = "modified"
            break

    state["meta"]["last_action"] = f"rename: {target_id}"


def _apply_add(state: dict, rec: dict):
    new_cluster_def = rec.get("new_cluster", {})
    if not new_cluster_def.get("name"):
        print("Error: add requires new_cluster with name", file=sys.stderr)
        sys.exit(1)

    next_id = state["meta"]["next_cluster_id"]
    new_cluster = {
        "id": f"c{next_id}",
        "name": new_cluster_def["name"],
        "description": new_cluster_def.get("description", ""),
        "confidence": "unaudited",
        "evidence": {
            "proposed_in": [],
            "evidence_text_ids": new_cluster_def.get("example_text_ids", []),
            "audit_assignments": 0,
            "audit_mean_confidence": None,
            "total_texts_seen": 0,
        },
        "status": "new",
    }
    state["clusters"].append(new_cluster)
    state["meta"]["next_cluster_id"] = next_id + 1
    state["meta"]["last_action"] = f"add: {new_cluster_def['name']} (c{next_id})"


def _apply_remove(state: dict, rec: dict):
    targets = rec.get("targets", [])
    if not targets:
        print("Error: remove requires targets", file=sys.stderr)
        sys.exit(1)

    state["clusters"] = [c for c in state["clusters"] if c["id"] not in targets]
    state["meta"]["last_action"] = f"remove: {', '.join(targets)}"


def _apply_no_change(state: dict, rec: dict, data: dict, source_filename: str):
    rejected = {
        "hypothesis": data.get("question", rec.get("rejected_hypothesis", "unknown")),
        "investigated_in": source_filename,
        "finding": rec.get("reasoning", rec.get("rejected_hypothesis", "")),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    state["meta"]["rejected_hypotheses"].append(rejected)
    state["meta"]["last_action"] = f"no_change: {data.get('question', 'investigation')[:80]}"


def cmd_update_descriptions(args):
    """Update cluster names/descriptions without resetting evidence or IDs.

    Input JSON format: {"clusters": [{"id": "c1", "name": "...", "description": "..."}, ...]}
    Matches by id first, then by normalized name.
    """
    desc_file = Path(args.file)
    if not desc_file.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    with open(desc_file, encoding="utf-8") as f:
        data = json.load(f)

    updates = data.get("clusters", [])
    if not updates:
        print("Error: no clusters found in input file", file=sys.stderr)
        sys.exit(1)

    lock = FileLock(str(LOCK_PATH))
    with lock:
        state = load_state()

        # Build lookup for existing clusters
        by_id = {c["id"]: c for c in state["clusters"]}
        by_name = {c["name"].strip().lower(): c for c in state["clusters"]}

        updated = 0
        for u in updates:
            # Match by id first, then by normalized name
            target = None
            if u.get("id") and u["id"] in by_id:
                target = by_id[u["id"]]
            elif u.get("name"):
                target = by_name.get(u["name"].strip().lower())

            if target is None:
                print(f"  Warning: no match for {u.get('id', '?')} / {u.get('name', '?')}", file=sys.stderr)
                continue

            if u.get("name"):
                target["name"] = u["name"]
            if u.get("description"):
                target["description"] = u["description"]
            updated += 1

        state["meta"]["last_action"] = f"update-descriptions: {updated} clusters updated"
        save_state(state)
        generate_summary(state)
        log_action("update-descriptions", f"{updated}/{len(updates)} cluster descriptions updated")

    print(f"Updated {updated}/{len(updates)} cluster descriptions (evidence preserved)")


def cmd_update_cross_proposal_metrics(args):
    """Store cross-proposal metrics summary in state."""
    metrics_file = Path(args.file)
    if not metrics_file.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    with open(metrics_file, encoding="utf-8") as f:
        report = json.load(f)

    # Compute summary stats from the full report
    pairwise = report.get("pairwise", {})
    ari_values = [pw["ari"] for pw in pairwise.values() if "ari" in pw]
    mean_ari = sum(ari_values) / len(ari_values) if ari_values else None

    element_sim = report.get("element_similarity", {})

    lock = FileLock(str(LOCK_PATH))
    with lock:
        state = load_state()
        state["meta"]["cross_proposal_metrics"] = {
            "file": str(metrics_file),
            "proposals_compared": len(report.get("proposals_compared", [])),
            "mean_ari": round(mean_ari, 3) if mean_ari is not None else None,
            "overall_element_similarity": element_sim.get("overall"),
            "n_inconsistent_texts": len(element_sim.get("inconsistent_texts", [])),
            "computed_at": report.get("timestamp", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
        }
        save_state(state)
        generate_summary(state)
        log_action("update-cross-proposal-metrics", f"ARI={mean_ari:.3f}, elem_sim={element_sim.get('overall', '?')}" if mean_ari is not None else "metrics stored")

    cp = state["meta"]["cross_proposal_metrics"]
    print(f"Cross-proposal metrics stored (mean ARI={cp['mean_ari']}, element_similarity={cp['overall_element_similarity']})")


def cmd_mark_seen(args):
    """Mark text IDs as seen."""
    lock = FileLock(str(LOCK_PATH))
    with lock:
        seen_path = WORKSPACE / "seen_ids.json"
        if seen_path.exists():
            with open(seen_path, encoding="utf-8") as f:
                seen = set(json.load(f))
        else:
            seen = set()

        seen.update(args.ids)
        with open(seen_path, "w", encoding="utf-8") as f:
            json.dump(sorted(seen), f)

    print(f"Marked {len(args.ids)} IDs as seen (total: {len(seen)})")


def cmd_finalize(args):
    """Export final taxonomy."""
    lock = FileLock(str(LOCK_PATH))
    with lock:
        state = load_state()

        # Load corpus for example text enrichment
        corpus_path = WORKSPACE / "corpus.json"
        corpus_lookup = {}
        if corpus_path.exists():
            with open(corpus_path, encoding="utf-8") as f:
                corpus_records = json.load(f)
            corpus_lookup = {r["id"]: r["text"] for r in corpus_records}

        # Fallback: collect example text IDs from audit files for clusters with empty evidence
        audit_examples_by_cluster = {}
        audit_dir = WORKSPACE / "audits"
        if audit_dir.exists():
            for af in sorted(audit_dir.glob("*.json")):
                with open(af, encoding="utf-8") as f:
                    audit_data = json.load(f)
                for assignment in audit_data.get("assignments", []):
                    cid = assignment.get("cluster_id")
                    conf = assignment.get("confidence")
                    tid = assignment.get("text_id")
                    if cid and tid and conf is not None:
                        # Normalize confidence if on 0-1 scale
                        if conf <= 1.0:
                            conf = conf * 5
                        audit_examples_by_cluster.setdefault(cid, []).append((conf, tid))
            # Sort by confidence descending, keep unique text IDs
            for cid in audit_examples_by_cluster:
                seen_tids = set()
                unique = []
                for conf, tid in sorted(audit_examples_by_cluster[cid], key=lambda x: -x[0]):
                    if tid not in seen_tids:
                        unique.append(tid)
                        seen_tids.add(tid)
                audit_examples_by_cluster[cid] = unique

        output = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "corpus": {
                "path": state["corpus"]["path"],
                "size": state["corpus"]["size"],
            },
            "config": state["config"],
            "cluster_version": state["meta"]["cluster_version"],
            "clusters": [],
            "metrics": {
                "coverage": state["meta"].get("coverage"),
                "mean_confidence": state["meta"].get("mean_confidence"),
            },
        }

        max_examples = args.max_examples
        for c in state["clusters"]:
            evidence = c.get("evidence", {})
            text_ids = evidence.get("evidence_text_ids", [])

            # Fallback: if no evidence text IDs, pull from audit assignments
            if not text_ids and c["id"] in audit_examples_by_cluster:
                text_ids = audit_examples_by_cluster[c["id"]]
                print(f"  {c['id']}: using {len(text_ids)} examples from audit data (evidence was empty)", file=sys.stderr)

            # Enrich with actual example texts (id + content), capped
            example_texts = []
            for tid in text_ids[:max_examples]:
                entry = {"id": tid}
                if tid in corpus_lookup:
                    entry["text"] = corpus_lookup[tid]
                example_texts.append(entry)

            output["clusters"].append({
                "id": c["id"],
                "name": c["name"],
                "description": c["description"],
                "confidence": c.get("confidence", "unaudited"),
                "evidence": evidence,
                "example_texts": example_texts,
            })

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        # Generate taxonomy.md — human-readable version
        taxonomy_lines = ["# Cluster Taxonomy", ""]

        # Corpus info
        taxonomy_lines.append(f"**Corpus**: {state['corpus']['path']} ({state['corpus']['size']} texts)")
        taxonomy_lines.append(f"**Clusters**: {len(output['clusters'])}")
        cov = state["meta"].get("coverage")
        if cov and isinstance(cov, dict) and cov.get("value") is not None:
            pct = f"{cov['value']:.0%}" if isinstance(cov['value'], float) else str(cov['value'])
            taxonomy_lines.append(f"**Coverage**: ~{pct}")
        mc = state["meta"].get("mean_confidence")
        if mc and isinstance(mc, dict) and mc.get("value") is not None:
            taxonomy_lines.append(f"**Mean confidence**: {mc['value']:.1f}")
        taxonomy_lines.append(f"**Cluster version**: {state['meta']['cluster_version']}")
        taxonomy_lines.append("")
        taxonomy_lines.append("---")
        taxonomy_lines.append("")

        for cluster_out in output["clusters"]:
            conf = cluster_out["confidence"]
            taxonomy_lines.append(f"## {cluster_out['name']} (`{cluster_out['id']}`) [{conf}]")
            taxonomy_lines.append("")
            taxonomy_lines.append(cluster_out["description"])
            taxonomy_lines.append("")

            if cluster_out.get("example_texts"):
                taxonomy_lines.append("**Examples:**")
                taxonomy_lines.append("")
                for ex in cluster_out["example_texts"]:
                    text = ex.get("text", "(text not available)")
                    # Truncate long texts for readability
                    if len(text) > 300:
                        text = text[:297] + "..."
                    taxonomy_lines.append(f"> {text}")
                    taxonomy_lines.append("")
            taxonomy_lines.append("---")
            taxonomy_lines.append("")

        taxonomy_path = WORKSPACE / "taxonomy.md"
        taxonomy_path.write_text("\n".join(taxonomy_lines), encoding="utf-8")

        # Archive intermediate files to clean up workspace root
        archive_dir = WORKSPACE / "archive"
        archive_dir.mkdir(exist_ok=True)

        # Directories to archive
        for subdir_name in ["proposals", "audits", "investigations", "metrics", "tfidf_cache"]:
            subdir = WORKSPACE / subdir_name
            if subdir.exists():
                dest = archive_dir / subdir_name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.move(str(subdir), str(dest))

        # Loose files to archive (keep state.json, taxonomy.md, final_taxonomy.json, corpus.json)
        keep_files = {"state.json", "taxonomy.md", "final_taxonomy.json", "corpus.json",
                      ".state.lock", ".plugin_root"}
        keep_dirs = {"archive"}
        for item in WORKSPACE.iterdir():
            if item.is_file() and item.name not in keep_files:
                shutil.move(str(item), str(archive_dir / item.name))
            elif item.is_dir() and item.name not in keep_dirs:
                dest = archive_dir / item.name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.move(str(item), str(dest))

        state["meta"]["last_action"] = f"finalized: {len(output['clusters'])} clusters exported"
        save_state(state)
        log_action("finalize", f"Exported {len(output['clusters'])} clusters to {args.output}")

    print(f"Final taxonomy exported to {args.output}")
    print(f"Human-readable taxonomy: {taxonomy_path}")
    print(f"Intermediate files archived to {archive_dir}/")
    print(json.dumps(output, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Clustering workspace state management")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # summarize
    subparsers.add_parser("summarize", help="Regenerate summary.md")

    # count-proposal
    subparsers.add_parser("count-proposal", help="Increment proposal counter")

    # count-investigation
    subparsers.add_parser("count-investigation", help="Increment investigation counter")

    # set-clusters
    sc = subparsers.add_parser("set-clusters", help="Set clusters from file")
    sc.add_argument("file", help="JSON file with clusters")

    # update-from-audit
    ua = subparsers.add_parser("update-from-audit", help="Update state from audit")
    ua.add_argument("file", help="Audit JSON file")

    # update-descriptions
    ud = subparsers.add_parser("update-descriptions", help="Update cluster names/descriptions without resetting evidence")
    ud.add_argument("file", help="JSON file with cluster id/name/description updates")

    # apply-recommendation
    ar = subparsers.add_parser("apply-recommendation", help="Apply investigation recommendation")
    ar.add_argument("file", help="Investigation JSON file")

    # update-cross-proposal-metrics
    ucpm = subparsers.add_parser("update-cross-proposal-metrics", help="Store cross-proposal metrics in state")
    ucpm.add_argument("file", help="Cross-proposal metrics JSON file")

    # mark-seen
    ms = subparsers.add_parser("mark-seen", help="Mark text IDs as seen")
    ms.add_argument("ids", nargs="+", help="Text IDs to mark as seen")

    # finalize
    fin = subparsers.add_parser("finalize", help="Export final taxonomy")
    fin.add_argument("--output", required=True, help="Output file path")
    fin.add_argument("--max-examples", type=int, default=5,
                     help="Max example texts per cluster in output (default: 5)")

    args = parser.parse_args()

    commands = {
        "summarize": cmd_summarize,
        "count-proposal": cmd_count_proposal,
        "count-investigation": cmd_count_investigation,
        "set-clusters": cmd_set_clusters,
        "update-from-audit": cmd_update_from_audit,
        "update-descriptions": cmd_update_descriptions,
        "apply-recommendation": cmd_apply_recommendation,
        "update-cross-proposal-metrics": cmd_update_cross_proposal_metrics,
        "mark-seen": cmd_mark_seen,
        "finalize": cmd_finalize,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
