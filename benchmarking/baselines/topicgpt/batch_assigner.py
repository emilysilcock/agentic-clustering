"""Phase 3: per-doc topic assignment via the OpenAI Batch API.

Replaces the vendored ``assign_topics``. Same prompt template
(``prompts/assignment.txt``, byte-identical to upstream); different dispatch.
SPEC §5.6.2 (TopicGPT row, 2026-05-23 revision) pins this phase to
``gpt-5-mini`` via the OpenAI Batch API for the 50% batch discount, with
OpenAI's automatic prompt caching on the stable prefix.

## Cache layout (OpenAI auto-cache)

Unlike Anthropic's per-block ``cache_control``, OpenAI auto-caches the
longest stable prefix across messages whose token count is ≥1,024. We
structure each request so:

* the **system** message is identical across all requests (constant);
* the **user** message starts with the assignment prompt + the full topic
  hierarchy + few-shot examples + the ``[Document]\\n`` marker (the
  "stable prefix" --- identical across all requests in this batch);
* the per-doc document body and the trailing ``Your response:`` portion
  follow the marker as the variable tail.

OpenAI hashes the longest matching prefix automatically, so we don't need
explicit cache-control annotations. Per SPEC §5.6.3, this is reliable
across all our seven datasets because every classification prompt's stable
prefix is well over 1,024 tokens.

No probe-batch guard --- the cache mechanic is documented and reliable.
We do record ``cached_tokens`` per response and persist the aggregate
``cache_hit_rate`` in ``usage_assign.json`` so a low rate (e.g., from a
broken prompt split) surfaces visibly post-hoc.

## Output

Writes ``data/topicgpt/<dataset>/assignment.jsonl`` with the schema the
vendored ``correct_topics`` expects::

    {"id": ..., "text": ..., "prompted_docs": <body>, "responses": <LLM text>}

Plus a ``usage_assign.json`` sidecar with the full cost breakdown.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import benchmarking  # noqa: F401 — truststore.inject_into_ssl()
import benchmarking.baselines.topicgpt  # noqa: F401 — _vendored on sys.path

from benchmarking.baselines.topicgpt.dataset_adapter import TOPICGPT_ROOT

MODEL = "gpt-5-mini"
SYSTEM_MESSAGE = "You are a helpful assistant."

# Upstream uses max_tokens=1000 for assignment. gpt-5-mini takes
# max_completion_tokens. The assignment response is short (single line
# `[1] Topic: reasoning (quote)`), so 500 is plenty and curbs runaway.
MAX_COMPLETION_TOKENS = 500

# OpenAI Batch API hard limits.
# 50,000 = the per-batch request hard cap; we chunk to 25,000 instead because
# the per-batch input-file size cap (200 MB) is the tighter constraint when
# the cached prefix + per-doc tail averages ~5 KB/request. GoEmotions phase 3
# tripped the 200 MB limit at 45,446 requests with a 50k chunk (2026-05-25);
# 25k chunks comfortably stay under for every dataset we have.
BATCH_REQUEST_LIMIT = 25_000
COMPLETION_WINDOW = "24h"

# Polling.
POLL_INITIAL_S = 30
POLL_MAX_S = 120
MAX_WAIT_SECONDS = 24 * 60 * 60  # match the completion window
_TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass
class AssignmentResult:
    out_path: Path
    n_docs: int
    n_errors: int
    usage: dict


def _get_client():
    from openai import OpenAI

    from benchmarking.secrets import load_secrets_into_env

    load_secrets_into_env()
    import os
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Export the env var or add it to secrets.json at the project root."
        )
    return OpenAI(api_key=api_key)


def _load_inputs(dataset_name: str) -> tuple[list[dict], str, str]:
    """Return (docs, assignment_prompt_template, topic_tree_str)."""
    out_dir = TOPICGPT_ROOT / dataset_name
    input_path = out_dir / "input.jsonl"
    topic_path = out_dir / "topics_refined.md"
    if not input_path.exists():
        raise FileNotFoundError(
            f"phase 3 needs phase 0 output: {input_path}. Run "
            f"`python -m benchmarking.baselines.topicgpt.dataset_adapter --only {dataset_name}`."
        )
    if not topic_path.exists():
        raise FileNotFoundError(
            f"phase 3 needs phase 2 output: {topic_path}. Run "
            f"`python -m benchmarking.experiments.run_topicgpt --phase refine --only {dataset_name}`."
        )

    docs = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assignment_prompt = (PROMPTS_DIR / "assignment.txt").read_text(encoding="utf-8")
    topic_tree_str = topic_path.read_text(encoding="utf-8").strip()
    return docs, assignment_prompt, topic_tree_str


def _split_prompt_template(template: str, tree_str: str) -> tuple[str, str]:
    """Substitute the tree, then split the rendered template on ``{Document}``.

    Returns (cached_prefix, per_doc_suffix). The prefix is byte-identical
    across every request in this batch, so OpenAI's auto-cache hashes it
    as a single key.
    """
    with_tree = template.replace("{tree}", tree_str)
    if "{Document}" not in with_tree:
        raise ValueError("assignment prompt template missing {Document} placeholder")
    prefix, suffix = with_tree.split("{Document}", maxsplit=1)
    return prefix, suffix


def _build_request(
    custom_id: str,
    user_text: str,
) -> dict:
    """One OpenAI Batch chat-completion request body.

    gpt-5-mini doesn't accept ``temperature`` / ``top_p`` --- omit them
    (matches the convention in the patched vendored utils.py).
    """
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_text},
            ],
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
        },
    }


def _write_requests_jsonl(
    path: Path,
    requests: list[dict],
) -> None:
    with path.open("w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")


def _submit_one_batch(client, requests_path: Path, dataset_name: str, chunk_idx: int) -> str:
    """Upload the requests JSONL, submit a batch, return batch_id."""
    tag = f"[topicgpt/{dataset_name}/phase=assign chunk={chunk_idx}]"
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
            "method": "topicgpt-assignment",
        },
    )
    print(f"{tag} submitted batch_id={batch.id}", flush=True)
    return batch.id


def _poll_until_done(client, batch_ids: list[str], dataset_name: str) -> list[dict]:
    """Poll every in-flight batch until terminal. Returns list of batch objects."""
    pending = list(batch_ids)
    deadline = time.monotonic() + MAX_WAIT_SECONDS
    interval = POLL_INITIAL_S
    last_log = 0.0
    final: dict[str, object] = {}

    while pending:
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"polling deadline exceeded; {len(pending)} batch(es) still in-flight: {pending[:5]}"
            )

        for batch_id in list(pending):
            batch = client.batches.retrieve(batch_id)
            if batch.status in _TERMINAL_STATUSES:
                pending.remove(batch_id)
                final[batch_id] = batch
                counts = batch.request_counts
                print(
                    f"[topicgpt/{dataset_name}/phase=assign] "
                    f"batch={batch_id} terminal={batch.status} "
                    f"completed={counts.completed}/{counts.total} failed={counts.failed}",
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
                    f"[topicgpt/{dataset_name}/phase=assign] "
                    f"batch={batch_id} status={batch.status} "
                    f"completed={counts.completed}/{counts.total} failed={counts.failed}",
                    flush=True,
                )
            last_log = now

        time.sleep(interval)
        if time.monotonic() - (deadline - MAX_WAIT_SECONDS) > 600:
            interval = POLL_MAX_S

    return [final[bid] for bid in batch_ids]


def _parse_batch_output(output_jsonl: str) -> tuple[dict[str, str], dict[str, dict], list[dict]]:
    """Return (custom_id -> response_text, custom_id -> usage_dict, errors).

    ``errors`` are entries with non-null error or non-200 status; caller
    decides how to record them (we record as response_text="Error" so the
    downstream correction phase picks them up).
    """
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
        # OpenAI's prompt_tokens is the total (cached+uncached). Subtract
        # cached so input_tokens means the non-cached billable portion,
        # matching the convention used in skills/corpus-tools/scripts/
        # classify.py and the patched vendored utils.py.
        usages[cid] = {
            "input_tokens": max(0, u.get("prompt_tokens", 0) - cached),
            "output_tokens": u.get("completion_tokens", 0),
            "cache_read_input_tokens": cached,
            "cache_creation_input_tokens": 0,
        }

    return responses, usages, errors


def _summarize_usage(usages: dict[str, dict]) -> dict:
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "n_responses": len(usages),
    }
    for u in usages.values():
        for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            totals[k] += u[k]
    return totals


def _cache_hit_rate(usage: dict) -> float:
    cr = usage["cache_read_input_tokens"]
    plain = usage["input_tokens"]
    total = cr + plain
    return (cr / total) if total > 0 else 0.0


def assign(
    dataset_name: str,
    *,
    overwrite: bool = False,
) -> AssignmentResult:
    """Phase 3 entry point."""
    out_dir = TOPICGPT_ROOT / dataset_name
    out_path = out_dir / "assignment.jsonl"
    usage_path = out_dir / "usage_assign.json"

    if not overwrite and out_path.exists() and out_path.stat().st_size > 0:
        print(f"[topicgpt/{dataset_name}/phase=assign] cache hit -> {out_path}", flush=True)
        with usage_path.open(encoding="utf-8") as f:
            usage = json.load(f)
        n = sum(1 for _ in out_path.open(encoding="utf-8"))
        return AssignmentResult(out_path=out_path, n_docs=n, n_errors=0, usage=usage)

    docs, prompt_template, tree_str = _load_inputs(dataset_name)
    cached_prefix, suffix_template = _split_prompt_template(prompt_template, tree_str)
    print(
        f"[topicgpt/{dataset_name}/phase=assign] {len(docs)} docs; "
        f"cached_prefix={len(cached_prefix)} chars; suffix_template={len(suffix_template)} chars",
        flush=True,
    )

    client = _get_client()

    # Build all requests; chunk under the OpenAI Batch hard cap.
    requests_dir = out_dir / "_batch_inputs"
    requests_dir.mkdir(parents=True, exist_ok=True)

    batch_ids: list[str] = []
    chunk_paths: list[Path] = []
    for chunk_idx, start in enumerate(range(0, len(docs), BATCH_REQUEST_LIMIT)):
        chunk = docs[start : start + BATCH_REQUEST_LIMIT]
        requests = [
            _build_request(
                custom_id=doc["id"],
                user_text=cached_prefix + doc["text"] + suffix_template,
            )
            for doc in chunk
        ]
        req_path = requests_dir / f"requests_chunk{chunk_idx:02d}.jsonl"
        _write_requests_jsonl(req_path, requests)
        chunk_paths.append(req_path)
        batch_id = _submit_one_batch(client, req_path, dataset_name, chunk_idx)
        batch_ids.append(batch_id)

    print(
        f"[topicgpt/{dataset_name}/phase=assign] {len(batch_ids)} batch(es) submitted; polling…",
        flush=True,
    )

    finished = _poll_until_done(client, batch_ids, dataset_name)

    # Pull output JSONLs and stitch.
    all_responses: dict[str, str] = {}
    all_usages: dict[str, dict] = {}
    all_errors: list[dict] = []
    for batch in finished:
        if batch.status != "completed":
            raise RuntimeError(
                f"batch {batch.id} did not complete: status={batch.status}"
            )
        output_jsonl = client.files.content(batch.output_file_id).text
        responses, usages, errors = _parse_batch_output(output_jsonl)
        all_responses.update(responses)
        all_usages.update(usages)
        all_errors.extend(errors)

    total_usage = _summarize_usage(all_usages)
    total_usage["cache_hit_rate"] = _cache_hit_rate(total_usage)
    total_usage["n_errors"] = len(all_errors)
    total_usage["model"] = MODEL

    # Stitch -> assignment.jsonl
    out_rows: list[str] = []
    for doc in docs:
        response_text = all_responses.get(doc["id"], "Error")
        out_rows.append(
            json.dumps(
                {
                    "id": doc["id"],
                    "text": doc["text"],
                    "gold_label_id": doc["gold_label_id"],
                    "prompted_docs": doc["text"],  # 512-tok cap applied at adapt time
                    "responses": response_text,
                },
                ensure_ascii=False,
            )
        )
    out_path.write_text("\n".join(out_rows) + "\n", encoding="utf-8")
    usage_path.write_text(json.dumps(total_usage, ensure_ascii=False, indent=2), encoding="utf-8")

    # Clean up batch input files (kept only for resubmission if we needed to).
    for p in chunk_paths:
        p.unlink(missing_ok=True)
    if requests_dir.exists() and not any(requests_dir.iterdir()):
        requests_dir.rmdir()

    print(
        f"[topicgpt/{dataset_name}/phase=assign] -> {out_path} | "
        f"n_docs={len(docs)} n_errors={len(all_errors)} | usage={total_usage}",
        flush=True,
    )
    return AssignmentResult(
        out_path=out_path, n_docs=len(docs), n_errors=len(all_errors), usage=total_usage
    )
