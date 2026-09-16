"""One-off: re-classify only the failed rows of an existing classify CSV and merge.

Recovers a classify output that was partially corrupted (e.g. an OpenAI billing
hard-limit hit mid-batch) without re-running the whole corpus. Reads the existing
seed_0.csv, finds rows with an error set or a blank cluster, classifies just those
doc_ids against the workspace's existing classification/prompt.md, and merges the
new results back into seed_0.csv (snapshotting the original to *.contaminated.csv
first).

After running this, re-assemble the artifact with the runner's reuse flag, e.g.:
    uv run --native-tls python -m benchmarking.experiments.run_ablations \
        --synthonly --only twenty_newsgroups --reuse-existing-classify

Usage:
    uv run --native-tls python scripts/retry_classify_errors.py \
        --csv    results/clustering/<ds>/<ws>/classification/classifications/seed_0.csv \
        --prompt results/clustering/<ds>/<ws>/classification/prompt.md \
        --corpus data/agentic_clustering/<ds>/documents.jsonl \
        --force-assign
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

# Make `benchmarking` importable regardless of how this script is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

csv.field_size_limit(2**31 - 1)

from benchmarking.baselines.agentic_clustering import (  # noqa: E402
    CLASSIFY_MODEL,
    CLASSIFY_PROVIDER,
    CLASSIFY_SCRIPT,
    _run_uv_script,
)
from benchmarking.secrets import load_secrets_into_env  # noqa: E402


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _is_failed(row: dict) -> bool:
    return bool((row.get("error") or "").strip()) or not (row.get("cluster") or "").strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="Existing classify seed_0.csv to repair (merged in place).")
    ap.add_argument("--prompt", required=True, help="classification prompt.md the CSV was produced against.")
    ap.add_argument("--corpus", required=True, help="Full (capped) documents.jsonl to pull failed docs from.")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--id-col", default="doc_id")
    ap.add_argument("--force-assign", action="store_true", help="Forbid 'none' (datasets with no OOS class).")
    ap.add_argument("--mode", default="async", choices=["async", "batch"],
                    help="async is faster for a small retry set (default); batch is 50%% cheaper.")
    ap.add_argument("--concurrency", type=int, default=20, help="async mode only.")
    args = ap.parse_args()

    # classify.py runs as a subprocess and inherits our env; inject the API key
    # the same way the production _run_classify does (mirrors agentic_clustering).
    load_secrets_into_env()

    csv_path = Path(args.csv)
    rows = _read_csv(csv_path)
    if not rows:
        print(f"[retry] {csv_path} is empty — nothing to do", file=sys.stderr)
        return 1
    by_id = {r["id"]: r for r in rows}
    failed_ids = [r["id"] for r in rows if _is_failed(r)]
    print(f"[retry] {len(failed_ids)} / {len(rows)} rows need retry")
    if not failed_ids:
        print("[retry] no failed rows — CSV is already clean")
        return 0

    # Pull just the failed docs out of the full corpus, preserving schema.
    failed_set = set(failed_ids)
    subset_lines: list[str] = []
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            if str(doc.get(args.id_col)) in failed_set:
                subset_lines.append(line)
    found = {str(json.loads(l)[args.id_col]) for l in subset_lines}
    missing = failed_set - found
    if missing:
        raise SystemExit(f"[retry] {len(missing)} failed ids not found in {args.corpus}, e.g. {list(missing)[:5]}")

    tmpdir = Path(tempfile.mkdtemp(prefix="retry_classify_"))
    subset_corpus = tmpdir / "subset.jsonl"
    subset_corpus.write_text("\n".join(subset_lines) + "\n", encoding="utf-8")
    retry_out = tmpdir / "retry.csv"

    classify_args = [
        "--input", str(subset_corpus),
        "--text-col", args.text_col,
        "--id-col", args.id_col,
        "--prompt", str(args.prompt),
        "--output", str(retry_out),
        "--provider", CLASSIFY_PROVIDER,
        "--model", CLASSIFY_MODEL,
        "--mode", args.mode,
    ]
    if args.mode == "async":
        classify_args += ["--concurrency", str(args.concurrency)]
    if args.force_assign:
        classify_args.append("--force-assign")

    print(f"[retry] classifying {len(subset_lines)} docs ({args.mode}) -> {retry_out}")
    _run_uv_script(CLASSIFY_SCRIPT, classify_args)

    retry_by_id = {r["id"]: r for r in _read_csv(retry_out)}
    still_bad = [i for i in failed_ids if i not in retry_by_id or _is_failed(retry_by_id[i])]
    print(f"[retry] retried {len(retry_by_id)} docs; still failing after retry: {len(still_bad)}")
    if still_bad:
        print(f"[retry]   e.g. {still_bad[:5]}", file=sys.stderr)

    # Snapshot the contaminated CSV once, then merge the retried rows in place.
    snap = csv_path.with_name(csv_path.stem + ".contaminated.csv")
    if not snap.exists():
        shutil.copy2(csv_path, snap)
        print(f"[retry] snapshot -> {snap.name}")
    fieldnames = list(rows[0].keys())
    for i in failed_ids:
        if i in retry_by_id:
            by_id[i] = {**by_id[i], **retry_by_id[i]}
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(by_id[r["id"]])
    print(f"[retry] merged {len(failed_ids) - len(still_bad)} repaired rows into {csv_path.name}")
    return 0 if not still_bad else 2


if __name__ == "__main__":
    sys.exit(main())
