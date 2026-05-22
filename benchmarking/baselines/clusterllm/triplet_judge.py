"""Phase-2 of ClusterLLM: judge sampled triplets with Claude Code (Opus 4.7).

Replaces ``perspective/1_predict_triplet/predict_triplet.py`` in the vendored
author source. Reads the JSON-list output of ``triplet_sampling.py`` and
writes a streaming JSONL of judgments that ``orchestrate.collate_predictions``
later folds back into the JSON-list shape ``convert_triplet.py`` expects.

Why a separate file instead of patching ``predict_triplet.py``: the original
uses the legacy ``openai==0.x`` API and has a hardcoded ``breakpoint()`` on
the first failure. Replacing that with the Claude Code subprocess client is
cleaner than monkey-patching their loop. The prompt assembly and
post-processing here mirror ``tools.py:prepare_data`` / ``post_process``
exactly so judgments stay format-compatible with ``convert_triplet.py``.

Resumability:
- Each judgment is appended to the output JSONL and ``fsync``'d immediately
  on success. Only the in-flight call at the time of interrupt is lost.
- On restart, ``(query_idx, choice1_idx, choice2_idx)`` triples already
  present in the JSONL are skipped.

Usage limits:
- Handled inside ``benchmarking.llm_clients.claude_code.call_claude`` — this
  module just propagates that retry behaviour. Multiple workers share one
  Max account, so they typically hit the same window together and each
  waits out the same reset independently.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from benchmarking.llm_clients.claude_code import (
    DEFAULT_MODEL,
    ClaudeCodeError,
    call_claude,
)

PROMPTS_PATH = Path(__file__).resolve().parent / "prompts.json"
POSTFIX = "\n\nPlease respond with 'Choice 1' or 'Choice 2' without explanation."


def _load_prompts() -> dict[str, str]:
    return json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))


def _prepare(task_instruction: str, input_txt: str) -> str:
    if input_txt.endswith("\nChoice"):
        input_txt = input_txt[: -len("\nChoice")]
    return task_instruction + input_txt + POSTFIX


def _parse_response(content: str) -> list[str]:
    matches: list[str] = []
    for opt in (" 1", " 2"):
        if ("Choice" + opt) in content:
            matches.append(opt)
    return matches


def _load_done(out_path: Path) -> set[tuple[int, int, int]]:
    if not out_path.exists():
        return set()
    done: set[tuple[int, int, int]] = set()
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done.add(
                (
                    int(rec["query_idx"]),
                    int(rec["choice1_idx"]),
                    int(rec["choice2_idx"]),
                )
            )
    return done


class _JsonlAppender:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, rec: dict[str, Any]) -> None:
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        with self._lock:
            self._fh.close()


def judge_triplets(
    triplets_path: Path,
    out_path: Path,
    *,
    dataset: str,
    model: str = DEFAULT_MODEL,
    concurrency: int = 4,
    log_prefix: str = "[triplet_judge]",
) -> dict[str, int]:
    """Judge every triplet in ``triplets_path``, streaming to ``out_path``.

    Returns a summary dict with counts.
    """
    prompts = _load_prompts()
    if dataset not in prompts:
        raise KeyError(
            f"No task instruction for dataset {dataset!r} in {PROMPTS_PATH}. "
            f"Available: {sorted(prompts)}"
        )
    task_instruction = prompts[dataset]

    with triplets_path.open(encoding="utf-8") as f:
        triplets = json.load(f)

    done = _load_done(out_path)
    pending = [
        t
        for t in triplets
        if (int(t["query_idx"]), int(t["choice1_idx"]), int(t["choice2_idx"]))
        not in done
    ]

    print(
        f"{log_prefix} {dataset}: {len(triplets)} total | "
        f"{len(done)} already judged | {len(pending)} pending | "
        f"concurrency={concurrency} | model={model}",
        flush=True,
    )

    if not pending:
        return {
            "total": len(triplets),
            "judged_pre": len(done),
            "new": 0,
            "ok": 0,
            "ambiguous": 0,
            "errored": 0,
        }

    appender = _JsonlAppender(out_path)
    counters = {"ok": 0, "ambiguous": 0, "errored": 0}
    counters_lock = threading.Lock()

    def _judge_one(i_rec: tuple[int, dict]) -> None:
        i, rec = i_rec
        prompt = _prepare(task_instruction, rec["input"])
        try:
            content = call_claude(prompt, model=model)
        except ClaudeCodeError as exc:
            with counters_lock:
                counters["errored"] += 1
            print(
                f"{log_prefix} {dataset} triplet#{i} ERR: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return

        content = content.strip()
        matches = _parse_response(content)
        appender.write({**rec, "content": content, "prediction": matches})
        with counters_lock:
            if len(matches) == 1:
                counters["ok"] += 1
            else:
                counters["ambiguous"] += 1

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_judge_one, (i, t)) for i, t in enumerate(pending)]
            for n, fut in enumerate(as_completed(futures), 1):
                fut.result()  # surfaces any uncaught exception inside the worker
                if n % 25 == 0 or n == len(futures):
                    with counters_lock:
                        ok, ambig, err = counters["ok"], counters["ambiguous"], counters["errored"]
                    print(
                        f"{log_prefix} {dataset} {n}/{len(futures)} "
                        f"ok={ok} ambig={ambig} err={err}",
                        flush=True,
                    )
    finally:
        appender.close()

    return {
        "total": len(triplets),
        "judged_pre": len(done),
        "new": len(pending),
        **counters,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triplets", type=Path, required=True, help="Phase-1 triplets.json")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL")
    parser.add_argument("--dataset", required=True, help="Dataset key in prompts.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    summary = judge_triplets(
        triplets_path=args.triplets,
        out_path=args.out,
        dataset=args.dataset,
        model=args.model,
        concurrency=args.concurrency,
    )
    print(f"[triplet_judge] {args.dataset} summary: {summary}", flush=True)


if __name__ == "__main__":
    main()
