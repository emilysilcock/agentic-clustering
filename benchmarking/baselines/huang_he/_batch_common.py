"""Shared OpenAI Batch API helpers for the Huang & He baseline.

Two call sites in this package both use OpenAI Batch:

* ``batch_generate`` (phase 1) — Stage-1 label generation, B=15 sentences
  per request, no shared prefix.
* ``batch_classify`` (phase 3) — Stage-2 per-doc classification, label
  list as a stable cached prefix.

The submission / polling / parsing boilerplate is identical between them,
so it lives here. Same pattern as ``topicgpt/batch_assigner.py`` and
``topicgpt/batch_correct.py``; a future refactor (out of scope for this PR)
should lift this up to ``benchmarking/llm_clients/openai_batch.py`` and
migrate the topicgpt baseline to use it too.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

# OpenAI Batch API hard limits and polling cadence (mirrors topicgpt/).
BATCH_REQUEST_LIMIT = 50_000
# OpenAI rejects batch input files over 209,715,200 bytes (200 MiB) with
# `maximum_input_file_size_exceeded`. We chunk on both request count AND
# serialized byte size, whichever is hit first, and leave ~10% headroom
# below the hard cap so newline framing / encoding slack can't tip us over.
BATCH_FILE_SIZE_LIMIT = 190 * 1024 * 1024
COMPLETION_WINDOW = "24h"
POLL_INITIAL_S = 30
POLL_MAX_S = 120
MAX_WAIT_SECONDS = 24 * 60 * 60  # match COMPLETION_WINDOW
_TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


def get_openai_client():
    """OpenAI client with API key loaded from secrets.json or env."""
    from openai import OpenAI

    from benchmarking.secrets import load_secrets_into_env

    load_secrets_into_env()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Export the env var or add it to "
            "secrets.json at the project root."
        )
    return OpenAI(api_key=api_key)


def build_chat_request(
    *,
    custom_id: str,
    user_text: str,
    model: str,
    system_message: str,
    max_completion_tokens: int,
    response_format: dict | None = None,
) -> dict:
    """One OpenAI Batch chat-completions request body for a reasoning model.

    ``gpt-5-mini`` rejects ``temperature`` / ``top_p`` so we omit them.
    ``response_format={"type": "json_object"}`` enforces JSON output at the
    API level, which is essential for both phases (the upstream code uses
    ``eval()`` to parse — we don't, and we set this header instead).
    """
    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_text},
        ],
        "max_completion_tokens": max_completion_tokens,
    }
    if response_format is not None:
        body["response_format"] = response_format
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def write_requests_jsonl(path: Path, requests: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")


def submit_one_batch(
    client,
    *,
    requests_path: Path,
    log_prefix: str,
    chunk_idx: int,
    method_tag: str,
    dataset_name: str,
) -> str:
    """Upload the request JSONL, create a batch, return its id."""
    tag = f"{log_prefix} chunk={chunk_idx}"
    print(f"{tag} uploading {requests_path.name}", flush=True)
    with requests_path.open("rb") as fh:
        input_file = client.files.create(file=fh, purpose="batch")
    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/chat/completions",
        completion_window=COMPLETION_WINDOW,
        metadata={
            "dataset": dataset_name,
            "chunk_index": str(chunk_idx),
            "method": method_tag,
        },
    )
    print(f"{tag} submitted batch_id={batch.id}", flush=True)
    return batch.id


def poll_until_done(client, batch_ids: list[str], *, log_prefix: str) -> list:
    """Poll every in-flight batch until terminal. Returns batch objects."""
    pending = list(batch_ids)
    deadline = time.monotonic() + MAX_WAIT_SECONDS
    interval = POLL_INITIAL_S
    last_log = 0.0
    final: dict[str, object] = {}

    while pending:
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"polling deadline exceeded; {len(pending)} batch(es) still "
                f"in-flight: {pending[:5]}"
            )

        for batch_id in list(pending):
            batch = client.batches.retrieve(batch_id)
            if batch.status in _TERMINAL_STATUSES:
                pending.remove(batch_id)
                final[batch_id] = batch
                counts = batch.request_counts
                print(
                    f"{log_prefix} batch={batch_id} terminal={batch.status} "
                    f"completed={counts.completed}/{counts.total} "
                    f"failed={counts.failed}",
                    flush=True,
                )

        if not pending:
            break

        now = time.monotonic()
        if now - last_log >= 60:
            for batch_id in pending:
                batch = client.batches.retrieve(batch_id)
                counts = batch.request_counts
                print(
                    f"{log_prefix} batch={batch_id} status={batch.status} "
                    f"completed={counts.completed}/{counts.total} "
                    f"failed={counts.failed}",
                    flush=True,
                )
            last_log = now

        time.sleep(interval)
        if time.monotonic() - (deadline - MAX_WAIT_SECONDS) > 600:
            interval = POLL_MAX_S

    return [final[bid] for bid in batch_ids]


@dataclass(frozen=True)
class BatchParseResult:
    responses: dict[str, str]            # custom_id -> assistant content
    usages: dict[str, dict]              # custom_id -> {input/output/cache tokens}
    errors: list[dict]                   # entries with err or non-200 status


def parse_batch_output(output_jsonl: str) -> BatchParseResult:
    """Pull responses + usage out of one OpenAI Batch output JSONL."""
    responses: dict[str, str] = {}
    usages: dict[str, dict] = {}
    errors: list[dict] = []

    for line in output_jsonl.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        cid = rec.get("custom_id", "")
        err = rec.get("error")
        resp = rec.get("response")
        if err or not resp or resp.get("status_code") != 200:
            errors.append({"custom_id": cid, "error": err, "response": resp})
            continue
        body = resp["body"]
        responses[cid] = body["choices"][0]["message"]["content"] or ""
        u = body.get("usage", {}) or {}
        details = u.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens", 0) or 0
        # OpenAI's prompt_tokens is total (cached + uncached); subtract
        # cached so input_tokens means the non-cached billable portion
        # (matches the convention in topicgpt/batch_assigner.py).
        usages[cid] = {
            "input_tokens": max(0, u.get("prompt_tokens", 0) - cached),
            "output_tokens": u.get("completion_tokens", 0),
            "cache_read_input_tokens": cached,
            "cache_creation_input_tokens": 0,
        }

    return BatchParseResult(responses=responses, usages=usages, errors=errors)


def summarize_usage(usages: dict[str, dict]) -> dict:
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "n_responses": len(usages),
    }
    for u in usages.values():
        for k in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ):
            totals[k] += u[k]
    cr = totals["cache_read_input_tokens"]
    plain = totals["input_tokens"]
    total = cr + plain
    totals["cache_hit_rate"] = (cr / total) if total > 0 else 0.0
    return totals


def _chunk_requests(requests: list[dict]) -> list[list[dict]]:
    """Split ``requests`` into chunks under both the count and byte caps.

    OpenAI enforces a 50k-request limit *and* a 200 MiB input-file limit per
    batch. A chunk is closed when adding the next request would exceed
    either ``BATCH_REQUEST_LIMIT`` or ``BATCH_FILE_SIZE_LIMIT`` (measured on
    the exact bytes ``write_requests_jsonl`` will write: the UTF-8 encoding
    of ``json.dumps(req, ensure_ascii=False) + "\\n"``).

    Raises if a single request alone exceeds the byte cap — chunking can't
    help there; the per-doc prompt (i.e. the merged label list) is simply
    too large for the Batch API and the caller needs a smaller taxonomy.
    """
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    cur_bytes = 0
    for req in requests:
        line_bytes = len(json.dumps(req, ensure_ascii=False).encode("utf-8")) + 1  # +1 for "\n"
        if line_bytes > BATCH_FILE_SIZE_LIMIT:
            raise RuntimeError(
                f"a single batch request is {line_bytes / 1024 / 1024:.1f} MB, "
                f"over the {BATCH_FILE_SIZE_LIMIT / 1024 / 1024:.0f} MB per-file "
                f"cap — the per-doc prompt (merged label list) is too large for "
                f"the Batch API. custom_id={req.get('custom_id')!r}."
            )
        would_overflow = (
            cur and (len(cur) >= BATCH_REQUEST_LIMIT or cur_bytes + line_bytes > BATCH_FILE_SIZE_LIMIT)
        )
        if would_overflow:
            chunks.append(cur)
            cur = []
            cur_bytes = 0
        cur.append(req)
        cur_bytes += line_bytes
    if cur:
        chunks.append(cur)
    return chunks


def submit_and_collect(
    client,
    *,
    requests: list[dict],
    out_dir: Path,
    log_prefix: str,
    method_tag: str,
    dataset_name: str,
    batch_inputs_subdir: str,
) -> tuple[dict[str, str], dict[str, dict], list[dict]]:
    """Chunk-submit ``requests`` under the count + byte caps, poll, collect.

    Returns ``(responses_by_custom_id, usages_by_custom_id, errors)``.
    """
    if not requests:
        return {}, {}, []

    requests_dir = out_dir / batch_inputs_subdir
    requests_dir.mkdir(parents=True, exist_ok=True)

    chunks = _chunk_requests(requests)
    batch_ids: list[str] = []
    chunk_paths: list[Path] = []
    for chunk_idx, chunk in enumerate(chunks):
        req_path = requests_dir / f"requests_chunk{chunk_idx:02d}.jsonl"
        write_requests_jsonl(req_path, chunk)
        chunk_paths.append(req_path)
        batch_id = submit_one_batch(
            client,
            requests_path=req_path,
            log_prefix=log_prefix,
            chunk_idx=chunk_idx,
            method_tag=method_tag,
            dataset_name=dataset_name,
        )
        batch_ids.append(batch_id)

    print(
        f"{log_prefix} {len(requests)} requests -> {len(batch_ids)} batch(es) "
        f"(count cap {BATCH_REQUEST_LIMIT}, size cap "
        f"{BATCH_FILE_SIZE_LIMIT // 1024 // 1024} MB); polling…",
        flush=True,
    )
    finished = poll_until_done(client, batch_ids, log_prefix=log_prefix)

    all_responses: dict[str, str] = {}
    all_usages: dict[str, dict] = {}
    all_errors: list[dict] = []
    for batch in finished:
        if batch.status != "completed":
            raise RuntimeError(
                f"batch {batch.id} did not complete: status={batch.status}"
            )
        output_jsonl = client.files.content(batch.output_file_id).text
        parsed = parse_batch_output(output_jsonl)
        all_responses.update(parsed.responses)
        all_usages.update(parsed.usages)
        all_errors.extend(parsed.errors)

    # Clean up batch input files (kept only for debugging up to this point).
    for p in chunk_paths:
        p.unlink(missing_ok=True)
    if requests_dir.exists() and not any(requests_dir.iterdir()):
        requests_dir.rmdir()

    return all_responses, all_usages, all_errors
