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
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 on stdout/stderr — Windows defaults to cp1252 and crashes on
# non-ASCII cluster names / corpus content. Idempotent; no-op on streams that
# aren't TextIOWrapper (e.g. captured in tests).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from filelock import FileLock

from _audit_metrics import (
    compute_assignment_stats,
    confidence_label,
    normalize_confidence_scale,
)
from _log import append_log
from _summary import render_summary
from _workspace import get_workspace

WORKSPACE = get_workspace()
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
    append_log(LOG_PATH, action, detail, metadata)


def generate_summary(state: dict):
    """Generate summary.md from current state. Layout lives in _summary.render_summary."""
    content = render_summary(state, log_path=LOG_PATH)
    summary_path = WORKSPACE / "summary.md"
    summary_path.write_text(content, encoding="utf-8")
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

    # Pre-validate before taking the lock: every cluster must carry a non-empty
    # `name` and `description`. Without this, `_normalize(c["name"])` below
    # KeyErrors (or AttributeError, for a JSON null) mid-loop on a malformed
    # synthesizer payload, after some `new_clusters` have already been built —
    # the user sees a Python traceback instead of a surfaceable error and
    # synthesizer.md step 11's "stop and surface the error" branch never fires.
    # Mirrors Round 6's fail-loud guards on _apply_rename / _apply_remove.
    for i, c in enumerate(clusters_input):
        if not isinstance(c, dict):
            print(f"Error: cluster {i} is not a JSON object", file=sys.stderr)
            sys.exit(1)
        name = c.get("name")
        if not isinstance(name, str) or not name.strip():
            print(f"Error: cluster {i} missing or empty 'name'", file=sys.stderr)
            sys.exit(1)
        desc = c.get("description")
        if not isinstance(desc, str) or not desc.strip():
            print(
                f"Error: cluster {i} (name={name!r}) missing or empty "
                "'description'",
                file=sys.stderr,
            )
            sys.exit(1)

    lock = FileLock(str(LOCK_PATH))
    with lock:
        state = load_state()
        next_id = state["meta"]["next_cluster_id"]

        # Build lookup of existing clusters by normalized name. Used both for
        # evidence preservation (when the new input has no text_ids) and for
        # ID reuse — re-synthesizing a cluster with the same name keeps its
        # existing ID instead of churning to c{next}, so external references
        # like `taxonomy.md` IDs and "investigate c3" remain stable.
        def _normalize(name: str) -> str:
            return name.strip().lower()

        existing_by_name = {_normalize(c["name"]): c for c in state.get("clusters", [])}

        new_clusters = []
        for c in clusters_input:
            text_id_list = c.get("evidence_text_ids",
                                 c.get("example_ids",
                                       c.get("text_ids",
                                             c.get("example_text_ids", []))))

            existing = existing_by_name.get(_normalize(c["name"]))
            # Reuse existing cluster's ID when name matches; otherwise mint a
            # fresh one and advance next_cluster_id.
            if existing:
                cluster_id = existing["id"]
            else:
                cluster_id = f"c{next_id}"
                next_id += 1

            # When input has no text_ids, preserve evidence from existing cluster
            if not text_id_list and existing:
                old_evidence = existing.get("evidence", {})
                cluster = {
                    "id": cluster_id,
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
                    "id": cluster_id,
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


def cmd_count_critique(_args):
    """Increment the total_critiques counter. Critiques are tracked separately
    from investigations because they're structural reviews, not actionable
    recommendations — apply-recommendation operates on investigations only."""
    lock = FileLock(str(LOCK_PATH))
    with lock:
        state = load_state()
        state["meta"]["total_critiques"] = state["meta"].get("total_critiques", 0) + 1
        save_state(state)
    print(f"Critiques: {state['meta']['total_critiques']}")


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

        # Reject audits from a different (or unknown) cluster version. Missing
        # field is also a hard reject: validate.py requires it, but this is
        # defence-in-depth for audits that bypassed the hook (e.g. dropped
        # directly into audits/ outside an agent turn).
        audit_version = audit.get("cluster_definitions_version")
        current_version = state["meta"]["cluster_version"]
        if audit_version is None:
            print(
                f"Error: audit is missing cluster_definitions_version. "
                f"Cannot verify it targets the current cluster set (version "
                f"{current_version}); refusing to apply.",
                file=sys.stderr,
            )
            sys.exit(1)
        if audit_version != current_version:
            print(
                f"Error: audit was for cluster version {audit_version}, "
                f"but current version is {current_version}. "
                f"This audit is stale and cannot be applied.",
                file=sys.stderr,
            )
            sys.exit(1)

        assignments = audit.get("assignments", [])

        # Defensive: if the auditor emitted 0-1 floats, rescale before counting.
        normalize_confidence_scale(assignments)

        # Single source of truth for coverage / mean_confidence / per-cluster
        # arithmetic. Replaces the LLM-authored summary.coverage_estimate and
        # summary.mean_confidence the auditor used to provide — those numbers
        # are now derived directly from `assignments` here, so they cannot
        # disagree with metrics.py or with the per-cluster counts below.
        stats = compute_assignment_stats(assignments)

        for cluster in state["clusters"]:
            cid = cluster["id"]
            slot = stats["per_cluster"].get(cid)
            if slot is None or not slot["confidences"]:
                continue
            confs = slot["confidences"]
            mean_conf = slot["mean_confidence"]

            # Accumulate with existing audit data (running average across audits).
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

            cluster["confidence"] = confidence_label(combined_mean)
            cluster["status"] = "audited"

        # Global metrics — computed from `assignments`, not from the auditor's
        # LLM-authored summary block.
        current_version = state["meta"]["cluster_version"]
        state["meta"]["coverage"] = {
            "value": stats["coverage"],
            "sample_size": stats["total"],
            "sample_method": audit.get("sample_method", "random, exclude-seen"),
            "cluster_version": current_version,
            "note": "Computed from audit assignments -- not a corpus-wide measurement",
        }
        state["meta"]["mean_confidence"] = {
            "value": stats["mean_confidence"],
            "sample_size": stats["total"],
            "sample_method": audit.get("sample_method", "random, exclude-seen"),
            "cluster_version": current_version,
        }

        state["meta"]["total_audits"] += 1
        cov_str = f"~{stats['coverage']:.0%}" if stats["total"] else "n/a"
        mc_str = f"{stats['mean_confidence']:.2f}" if stats["mean_confidence"] is not None else "n/a"
        state["meta"]["last_action"] = f"audit: coverage {cov_str}, confidence {mc_str}"

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

    # Build a lookup so we can read evidence from the about-to-be-removed
    # clusters before they go.
    clusters_by_id = {c["id"]: c for c in state["clusters"]}
    ids_to_remove = [t for t in targets if t != surviving_id]

    survivor = clusters_by_id.get(surviving_id)
    if survivor is None:
        print(f"Error: merge survivor cluster '{surviving_id}' not found", file=sys.stderr)
        sys.exit(1)

    # Union the evidence_text_ids (dedup preserving order) and accumulate the
    # audit counters across all sources. "Merge" implies the survivor now
    # represents the union of inputs — its evidence should reflect that, not
    # silently keep only its pre-merge subset.
    survivor_evidence = survivor.setdefault("evidence", {})
    merged_text_ids: list[str] = list(survivor_evidence.get("evidence_text_ids", []))
    seen_text_ids = set(merged_text_ids)

    total_audit_count = survivor_evidence.get("audit_assignments", 0) or 0
    total_audit_conf_sum = (
        (survivor_evidence.get("audit_mean_confidence") or 0.0) * total_audit_count
    )
    total_texts_seen = survivor_evidence.get("total_texts_seen", 0) or 0

    for rid in ids_to_remove:
        removed = clusters_by_id.get(rid)
        if removed is None:
            continue
        r_ev = removed.get("evidence", {})
        for tid in r_ev.get("evidence_text_ids", []):
            if tid not in seen_text_ids:
                merged_text_ids.append(tid)
                seen_text_ids.add(tid)
        r_count = r_ev.get("audit_assignments", 0) or 0
        r_mean = r_ev.get("audit_mean_confidence")
        if r_mean is not None and r_count > 0:
            total_audit_conf_sum += r_mean * r_count
            total_audit_count += r_count
        total_texts_seen += r_ev.get("total_texts_seen", 0) or 0

    survivor_evidence["evidence_text_ids"] = merged_text_ids
    survivor_evidence["audit_assignments"] = total_audit_count
    survivor_evidence["audit_mean_confidence"] = (
        round(total_audit_conf_sum / total_audit_count, 2) if total_audit_count > 0 else None
    )
    survivor_evidence["total_texts_seen"] = total_texts_seen

    # Update name/description on the survivor; refresh its confidence label
    # from the newly-merged audit mean.
    if merge_info.get("name"):
        survivor["name"] = merge_info["name"]
    if merge_info.get("description"):
        survivor["description"] = merge_info["description"]
    survivor["confidence"] = confidence_label(survivor_evidence["audit_mean_confidence"])
    survivor["status"] = "modified"

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

    # Enforce coverage: the investigator must assign EVERY evidence_text_id
    # from the parent cluster to one of the split buckets. Without this, the
    # first bucket silently inherits the parent's full evidence list — over-
    # claiming texts that conceptually moved to a sibling.
    original_text_ids = list(
        original_cluster.get("evidence", {}).get("evidence_text_ids", [])
    )

    # Each parent text_id must appear in EXACTLY ONE bucket. The downstream
    # missing / extras checks both use a set so an id claimed by two buckets
    # would silently absorb to one. A duplicate-claimed text could even mask
    # a genuinely missing one and pass the coverage check spuriously. Check
    # overlap first so the error names the offending ids and buckets.
    seen_buckets: dict[str, list[str]] = {}
    for split_def in split_into:
        bucket_name = split_def.get("name", "<unnamed>")
        for tid in split_def.get("example_text_ids", []):
            seen_buckets.setdefault(tid, []).append(bucket_name)
    overlaps = {tid: buckets for tid, buckets in seen_buckets.items() if len(buckets) > 1}
    if overlaps:
        sample = ", ".join(
            f"{tid!r}→[{', '.join(buckets)}]"
            for tid, buckets in list(overlaps.items())[:10]
        )
        more = f" ... and {len(overlaps) - 10} more" if len(overlaps) > 10 else ""
        print(
            f"Error: split for '{original_id}' assigns {len(overlaps)} "
            f"text_id(s) to multiple buckets — each parent text must land in "
            f"exactly one bucket. Overlaps: {sample}{more}",
            file=sys.stderr,
        )
        sys.exit(1)

    assigned_ids: set[str] = set(seen_buckets.keys())
    missing = [tid for tid in original_text_ids if tid not in assigned_ids]
    if missing:
        print(
            f"Error: split for '{original_id}' must assign every parent "
            f"evidence_text_id to a split bucket. Missing {len(missing)}: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}",
            file=sys.stderr,
        )
        sys.exit(1)

    # And the inverse: a bucket cannot smuggle in text_ids that weren't on the
    # parent. An investigator that hallucinates ids would otherwise land them
    # as evidence on the new clusters; downstream example-text enrichment then
    # either silently drops them ("text not available") or pulls an unrelated
    # text in by id collision.
    extra = sorted(assigned_ids - set(original_text_ids))
    if extra:
        print(
            f"Error: split for '{original_id}' has bucket text_ids not in the "
            f"parent's evidence ({len(extra)} extras: "
            f"{extra[:10]}{'...' if len(extra) > 10 else ''}). Buckets must "
            f"redistribute parent texts, not add new ones.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Provenance carried by ALL split buckets, not just the first. The split
    # is one source cluster becoming N — every resulting bucket traces back
    # to the same proposal(s). Fresh shallow copy per bucket so a later edit
    # to one bucket's provenance list doesn't leak across siblings.
    parent_proposed_in = list(
        original_cluster.get("evidence", {}).get("proposed_in", [])
    )

    # First new cluster keeps the original ID. Replace its evidence_text_ids
    # with the first split's explicit bucket (no longer inherits the parent's
    # full list).
    first_split = split_into[0]
    original_cluster["name"] = first_split["name"]
    original_cluster["description"] = first_split["description"]
    original_cluster["confidence"] = "unaudited"
    first_bucket = list(first_split.get("example_text_ids", []))
    original_cluster["evidence"] = {
        "proposed_in": list(parent_proposed_in),
        "evidence_text_ids": first_bucket,
        "audit_assignments": 0,
        "audit_mean_confidence": None,
        "total_texts_seen": len(first_bucket),
    }
    original_cluster["status"] = "modified"

    # Remaining splits get new IDs
    next_id = state["meta"]["next_cluster_id"]
    for split_def in split_into[1:]:
        bucket = list(split_def.get("example_text_ids", []))
        new_cluster = {
            "id": f"c{next_id}",
            "name": split_def["name"],
            "description": split_def["description"],
            "confidence": "unaudited",
            "evidence": {
                "proposed_in": list(parent_proposed_in),
                "evidence_text_ids": bucket,
                "audit_assignments": 0,
                "audit_mean_confidence": None,
                "total_texts_seen": len(bucket),
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
    target = next((c for c in state["clusters"] if c["id"] == target_id), None)
    if target is None:
        print(
            f"Error: rename target cluster '{target_id}' not found in current clusters",
            file=sys.stderr,
        )
        sys.exit(1)
    if rename_to.get("name"):
        target["name"] = rename_to["name"]
    if rename_to.get("description"):
        target["description"] = rename_to["description"]
    target["status"] = "modified"

    state["meta"]["last_action"] = f"rename: {target_id}"


def _apply_add(state: dict, rec: dict):
    new_cluster_def = rec.get("new_cluster", {})
    if not new_cluster_def.get("name"):
        print("Error: add requires new_cluster with name", file=sys.stderr)
        sys.exit(1)

    next_id = state["meta"]["next_cluster_id"]
    example_ids = list(new_cluster_def.get("example_text_ids", []))
    new_cluster = {
        "id": f"c{next_id}",
        "name": new_cluster_def["name"],
        "description": new_cluster_def.get("description", ""),
        "confidence": "unaudited",
        "evidence": {
            "proposed_in": [],
            "evidence_text_ids": example_ids,
            "audit_assignments": 0,
            "audit_mean_confidence": None,
            # Match _apply_split's accounting: when an add carries explicit
            # example_text_ids, those texts have been "seen" in the sense the
            # counter tracks. Hardcoding 0 here disagreed with the populated
            # evidence_text_ids on the same dict.
            "total_texts_seen": len(example_ids),
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

    existing_ids = {c["id"] for c in state["clusters"]}
    missing = [t for t in targets if t not in existing_ids]
    if missing:
        print(
            f"Error: remove target cluster(s) not found in current clusters: {missing}",
            file=sys.stderr,
        )
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

        # Name lookup: detect collisions explicitly so we don't silently let one
        # cluster update its duplicate-named sibling (dict-comprehension would
        # be last-write-wins). Colliding names are dropped from the lookup so
        # the only path to update them is by id.
        by_name: dict[str, dict] = {}
        duplicate_names: set[str] = set()
        for c in state["clusters"]:
            key = c["name"].strip().lower()
            if key in by_name:
                duplicate_names.add(key)
            else:
                by_name[key] = c
        for k in duplicate_names:
            by_name.pop(k, None)
        if duplicate_names:
            print(
                f"  Warning: duplicate cluster names detected ({sorted(duplicate_names)}); "
                f"name-based matching disabled for these — use cluster id instead.",
                file=sys.stderr,
            )

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

    # Store the path relative to the workspace so a moved or zipped workspace
    # still resolves it. Fall back to absolute if the file lives outside the
    # workspace (unusual but possible if the orchestrator passed a path from
    # elsewhere).
    try:
        rel_metrics_path = str(metrics_file.resolve().relative_to(WORKSPACE.resolve()))
    except ValueError:
        rel_metrics_path = str(metrics_file)

    lock = FileLock(str(LOCK_PATH))
    with lock:
        state = load_state()
        state["meta"]["cross_proposal_metrics"] = {
            "file": rel_metrics_path,
            "proposals_compared": len(report.get("proposals_compared", [])),
            "mean_ari": round(mean_ari, 3) if mean_ari is not None else None,
            "overall_element_similarity": element_sim.get("overall"),
            "n_inconsistent_texts": len(element_sim.get("inconsistent_texts", [])),
            "computed_at": report.get("timestamp", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
        }
        save_state(state)
        generate_summary(state)
        ari_str = f"{mean_ari:.3f}" if mean_ari is not None else "n/a"
        log_action(
            "update-cross-proposal-metrics",
            f"ARI={ari_str}, elem_sim={element_sim.get('overall', '?')}",
        )

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

        sample_ids = list(args.ids)[:5]
        more = "..." if len(args.ids) > 5 else ""
        log_action(
            "mark-seen",
            f"{len(args.ids)} ids: {sample_ids}{more} (total seen: {len(seen)})",
        )

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
                audit_assignments = audit_data.get("assignments", [])
                # Whole-audit scale rescue. The previous per-value `if conf <=
                # 1.0: conf *= 5` misfired on mixed-scale audits (e.g. a 1-5
                # audit that happened to contain a legitimate `1` got that
                # single value inflated to 5). Use the shared helper so the
                # rescale decision is made once per audit, matching how
                # update-from-audit and metrics.py treat the same data.
                normalize_confidence_scale(audit_assignments, warn=False)
                for assignment in audit_assignments:
                    cid = assignment.get("cluster_id")
                    conf = assignment.get("confidence")
                    tid = assignment.get("text_id")
                    if cid and tid and conf is not None:
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

        # Regenerate summary.md from the freshly-saved state BEFORE archiving,
        # so the archived summary matches the state.json snapshot we ship
        # alongside the taxonomy. (cmd_finalize doesn't otherwise call
        # generate_summary, and summary.md is then moved into archive/.)
        generate_summary(state)

        # Archive intermediate files to clean up workspace root
        archive_dir = WORKSPACE / "archive"
        archive_dir.mkdir(exist_ok=True)

        # Directories to archive. `critiques/` is included so critic outputs
        # ride along with the other intermediate artifacts.
        for subdir_name in ["proposals", "audits", "investigations", "critiques", "metrics"]:
            subdir = WORKSPACE / subdir_name
            if subdir.exists():
                dest = archive_dir / subdir_name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.move(str(subdir), str(dest))

        # tfidf_cache holds regeneratable pickled vectorizers — delete rather
        # than archive. search.py rebuilds lazily on next use; bytes saved
        # outweigh the small rebuild cost (which is a discovery-time tool,
        # rarely used post-finalize anyway).
        tfidf_dir = WORKSPACE / "tfidf_cache"
        if tfidf_dir.exists():
            shutil.rmtree(tfidf_dir)

        # Loose files to archive. `seen_ids.json` stays so /cluster-label's
        # subsequent sample.py calls don't re-pull discovery-audited texts.
        # `log.jsonl` stays so its chronological trace keeps appending across
        # phases (labelling, tuning, classification) instead of resetting at
        # finalize. `plan.md` stays as the orchestrator's forward-looking
        # notes so a re-finalize or follow-up session has the context (the
        # backward-looking `run_log.md` and `summary.md` move into archive).
        keep_files = {"state.json", "taxonomy.md", "final_taxonomy.json", "corpus.json",
                      "seen_ids.json", "log.jsonl", "plan.md",
                      ".state.lock", ".plugin_root", ".active_workspace"}
        # Keep `classification/` so a re-finalize after labelling/tuning/
        # classifying doesn't sweep labels.json, tuned_prompt.md, or the
        # timestamped run_*.csv outputs into the archive.
        keep_dirs = {"archive", "classification"}
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

    total_examples = sum(len(c.get("example_texts", [])) for c in output["clusters"])
    print(f"Final taxonomy exported to {args.output}")
    print(f"Human-readable taxonomy: {taxonomy_path}")
    print(f"Intermediate files archived to {archive_dir}/ (tfidf_cache/ deleted — regenerates on demand)")
    print(f"  {len(output['clusters'])} clusters, {total_examples} example texts")


def main():
    parser = argparse.ArgumentParser(description="Clustering workspace state management")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # summarize
    subparsers.add_parser("summarize", help="Regenerate summary.md")

    # count-proposal
    subparsers.add_parser("count-proposal", help="Increment proposal counter")

    # count-investigation
    subparsers.add_parser("count-investigation", help="Increment investigation counter")

    # count-critique
    subparsers.add_parser("count-critique", help="Increment critique counter")

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
        "count-critique": cmd_count_critique,
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
