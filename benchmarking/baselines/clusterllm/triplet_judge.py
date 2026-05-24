"""Phase-2 of ClusterLLM: judge sampled triplets.

Two judging paths live here:

1. ``judge_triplets`` (legacy) — Claude Code subprocess on Opus 4.7. Retained
   for the 1810 already-judged records archived as ``triplets_judged.opus.jsonl``;
   not used for new runs.
2. ``judge_triplets_openai_batch`` (current) — ``gpt-5-mini`` via OpenAI Batch
   API. Per SPEC §5.6.3 (updated 2026-05-23), any phase running an LLM over
   >1,000 texts is on the cheap tier; ClusterLLM's 1,024-triplet-per-dataset
   shape is just over the threshold and routes here. See the SPEC's "ClusterLLM"
   row in §5.6.2.

The prompt assembly and post-processing mirror ``tools.py:prepare_data`` /
``post_process`` from the vendored upstream exactly so judgments stay
format-compatible with ``convert_triplet.py``.

Resumability:
- Each judgment is appended to the output JSONL and ``fsync``'d immediately
  on write. ``(query_idx, choice1_idx, choice2_idx)`` triples already present
  in the JSONL are skipped on restart.
- The Batch-API path is idempotent at the per-record level too: on retry it
  re-submits only the still-pending records.
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


# ---------------------------------------------------------------------------
# gpt-5-mini Batch-API path (current default per SPEC §5.6.3)
# ---------------------------------------------------------------------------

# Pricing on the OpenAI Batch API (50% off sync). Pinned as paper artefact.
_GPT5_MINI_USD_PER_1M_INPUT = 0.125
_GPT5_MINI_USD_PER_1M_OUTPUT = 1.000
_BATCH_TERMINAL = {"completed", "failed", "expired", "cancelled"}


def _get_openai_client():
    # Local import so the legacy Claude path stays usable without OpenAI installed.
    from benchmarking.embeddings.openai_embeddings import _get_client
    return _get_client()


def _stream_download_file(client, file_id: str, dest: Path, *, max_attempts: int = 6) -> None:
    """Stream an OpenAI file to disk, retrying on Cloudflare 504s.

    Cribbed from scripts/recover_orphan_batches.py — the embedding output
    bug taught us that ``client.files.content(...).text`` blocks the whole
    body and 504s on large files, even though batch judgment outputs here
    are small. Cheap to keep streaming for robustness.
    """
    import time as _time
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            with client.files.with_streaming_response.content(file_id) as resp:
                with dest.open("wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            return
        except Exception as exc:
            last_exc = exc
            wait = min(30 * attempt, 300)
            print(
                f"  download attempt {attempt}/{max_attempts} failed "
                f"({type(exc).__name__}); retry in {wait}s",
                flush=True,
            )
            _time.sleep(wait)
    raise RuntimeError(f"file download failed after {max_attempts} attempts: {last_exc}")


def judge_triplets_openai_batch(
    triplets_path: Path,
    out_path: Path,
    *,
    dataset: str,
    model: str = "gpt-5-mini",
    poll_interval_s: int = 30,
    max_wait_seconds: int = 24 * 60 * 60,
    log_prefix: str = "[triplet_judge_batch]",
) -> dict[str, int]:
    """Judge every pending triplet via OpenAI Batch API on gpt-5-mini.

    Same contract as ``judge_triplets`` (skip-on-resume, JSONL-append, same
    record shape) but submits the entire pending set as a single batch and
    waits up to ``max_wait_seconds`` for it to terminate.
    """
    import time as _time

    prompts = _load_prompts()
    if dataset not in prompts:
        raise KeyError(
            f"No task instruction for {dataset!r} in {PROMPTS_PATH}. "
            f"Have: {sorted(prompts)}"
        )
    task_instruction = prompts[dataset]

    with triplets_path.open(encoding="utf-8") as f:
        triplets = json.load(f)

    done = _load_done(out_path)
    pending: list[tuple[int, dict]] = [
        (i, t)
        for i, t in enumerate(triplets)
        if (int(t["query_idx"]), int(t["choice1_idx"]), int(t["choice2_idx"])) not in done
    ]
    print(
        f"{log_prefix} {dataset}: {len(triplets)} total | "
        f"{len(done)} already judged | {len(pending)} pending | model={model}",
        flush=True,
    )
    if not pending:
        return {"total": len(triplets), "judged_pre": len(done), "new": 0,
                "ok": 0, "ambiguous": 0, "errored": 0,
                "input_tokens": 0, "output_tokens": 0, "usd": 0.0}

    client = _get_openai_client()
    work_dir = out_path.parent / f".batch_judge_{dataset}"
    work_dir.mkdir(parents=True, exist_ok=True)

    requests_path = work_dir / "requests.jsonl"
    with requests_path.open("w", encoding="utf-8") as f:
        for i, rec in pending:
            prompt = _prepare(task_instruction, rec["input"])
            cid = f"trip-{i}-{rec['query_idx']}-{rec['choice1_idx']}-{rec['choice2_idx']}"
            # gpt-5-mini is a reasoning model: ``max_completion_tokens`` counts
            # hidden reasoning + visible tokens. Budget 2048 so the API-default
            # ``reasoning_effort="medium"`` has room to think before emitting
            # the "Choice 1"/"Choice 2" line. Per SPEC convention we leave
            # ``reasoning_effort`` unset so it matches the default used by every
            # other gpt-5-mini call site in the paper (TopicGPT, Huang & He,
            # our method's classify.py); flagging any deviation in the paper.
            req = {
                "custom_id": cid,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": 2048,
                },
            }
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

    print(f"{log_prefix} {dataset}: uploading {len(pending)} requests...", flush=True)
    with requests_path.open("rb") as fh:
        input_file = client.files.create(file=fh, purpose="batch")
    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"dataset": dataset, "phase": "clusterllm_judge", "n_requests": str(len(pending))},
    )
    print(f"{log_prefix} {dataset}: batch_id={batch.id} submitted", flush=True)

    deadline = _time.monotonic() + max_wait_seconds
    last_status = ""
    while True:
        if _time.monotonic() > deadline:
            raise TimeoutError(f"batch {batch.id} did not terminate within {max_wait_seconds}s")
        batch = client.batches.retrieve(batch.id)
        counts = batch.request_counts
        if batch.status != last_status or counts.completed % 100 == 0:
            print(
                f"{log_prefix} {dataset}: status={batch.status} "
                f"done={counts.completed}/{counts.total} failed={counts.failed}",
                flush=True,
            )
            last_status = batch.status
        if batch.status in _BATCH_TERMINAL:
            break
        _time.sleep(poll_interval_s)

    if batch.status != "completed":
        raise RuntimeError(
            f"batch {batch.id} terminated with status={batch.status}; "
            f"counts={batch.request_counts}"
        )

    output_path = work_dir / "output.jsonl"
    _stream_download_file(client, batch.output_file_id, output_path)

    appender = _JsonlAppender(out_path)
    counters = {"ok": 0, "ambiguous": 0, "errored": 0}
    input_tokens = 0
    output_tokens = 0
    cid_to_rec = {
        f"trip-{i}-{rec['query_idx']}-{rec['choice1_idx']}-{rec['choice2_idx']}": rec
        for i, rec in pending
    }
    try:
        with output_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                resp_rec = json.loads(line)
                cid = resp_rec.get("custom_id", "")
                rec = cid_to_rec.get(cid)
                if rec is None:
                    counters["errored"] += 1
                    continue
                err = resp_rec.get("error")
                resp = resp_rec.get("response")
                if err or not resp or resp.get("status_code") != 200:
                    counters["errored"] += 1
                    continue
                body = resp["body"]
                content = body["choices"][0]["message"]["content"]
                content = (content or "").strip()
                matches = _parse_response(content)
                appender.write({**rec, "content": content, "prediction": matches})
                if len(matches) == 1:
                    counters["ok"] += 1
                else:
                    counters["ambiguous"] += 1
                usage = body.get("usage", {}) or {}
                input_tokens += int(usage.get("prompt_tokens", 0))
                output_tokens += int(usage.get("completion_tokens", 0))
    finally:
        appender.close()

    usd = (
        input_tokens * _GPT5_MINI_USD_PER_1M_INPUT / 1_000_000
        + output_tokens * _GPT5_MINI_USD_PER_1M_OUTPUT / 1_000_000
    )

    # Tidy up the work dir on success — the cache layer is the JSONL.
    for f in work_dir.glob("*"):
        f.unlink()
    work_dir.rmdir()

    summary = {
        "total": len(triplets),
        "judged_pre": len(done),
        "new": len(pending),
        **counters,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "usd": round(usd, 4),
        "batch_id": batch.id,
    }
    print(f"{log_prefix} {dataset} summary: {summary}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triplets", type=Path, required=True, help="Phase-1 triplets.json")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL")
    parser.add_argument("--dataset", required=True, help="Dataset key in prompts.json")
    parser.add_argument(
        "--judge",
        choices=("openai_batch", "claude"),
        default="openai_batch",
        help="Which judging backend to use. Default per SPEC §5.6.3.",
    )
    parser.add_argument("--model", default=None,
                        help="Model id; defaults depend on --judge.")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Workers for the Claude path; ignored for openai_batch.")
    args = parser.parse_args()

    if args.judge == "openai_batch":
        model = args.model or "gpt-5-mini"
        summary = judge_triplets_openai_batch(
            triplets_path=args.triplets,
            out_path=args.out,
            dataset=args.dataset,
            model=model,
        )
    else:
        model = args.model or DEFAULT_MODEL
        summary = judge_triplets(
            triplets_path=args.triplets,
            out_path=args.out,
            dataset=args.dataset,
            model=model,
            concurrency=args.concurrency,
        )
    print(f"[triplet_judge] {args.dataset} summary: {summary}", flush=True)


if __name__ == "__main__":
    main()
