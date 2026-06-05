"""Phase 4 alternative: per-row correction via the OpenAI Batch API.

Mirrors ``batch_assigner.py`` but for the correction reprompt loop --- the
vendored ``correct_topics`` only supports sync iterative prompts (or vLLM
batch); we want OpenAI's batch discount + parallelism + auto-cache for the
~hundreds-to-low-thousands of reprompts that surface after phase 3.

## What gets reprompted

Same identification logic as vendored ``correct_topics``: re-uses
``topic_parser`` to find rows whose phase-3 ``responses`` either failed to
parse against the regex (``error``) or matched a topic name not in the
refined taxonomy (``hallucinated``). Both groups get a single LLM call each
using the upstream correction prompt template.

## Cache layout

The upstream prompt has three placeholders: ``{tree}`` (identical across
rows), ``{Document}`` (per-row), ``{Message}`` (per-row, embeds the prior
bad response). After substituting ``{tree}``, we split on ``{Document}`` so
the cached prefix is everything *before* the document body. The per-row
tail = ``{Document}`` content + the remainder of the template with
``{Message}`` substituted for the row's previous bad response.

OpenAI auto-caches the longest stable prefix per request --- same mechanic
as ``batch_assigner.py``. For tiny taxonomies (Banking77: 5 topics) the
prefix may stay under the 1,024-token threshold and not cache; for larger
taxonomies (CLINC150 etc.) it caches comfortably.

## max_completion_tokens

Bumped to ``CORRECT_MAX_COMPLETION_TOKENS = 2000`` (vs the vendored sync
path's 500) because ``gpt-5-mini`` is a reasoning model whose
``max_completion_tokens`` budget covers both reasoning tokens and visible
output. With 500, observed Banking77 sync calls hit the 400 error
``max_tokens or model output limit was reached`` when reasoning consumed
the whole budget.

## Output

Reads ``assignment.jsonl``, overwrites the ``responses`` field on the
reprompted rows, writes ``corrected.jsonl`` with the same schema as
``assignment.jsonl``. Plus a ``usage_correct.json`` sidecar. A
``--retry-correct`` run *accumulates* its tokens into that sidecar (keeping a
per-pass ``passes`` breakdown) rather than overwriting it, so the initial
pass's spend stays in the cost accounting.
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

# gpt-5-mini reasoning + output share this budget. Bumped from 2000 to 4000
# (2026-05-25) after the 20NG correction batch had 2,806 / 12,768 rows
# (22%) hit the 400 error "max_tokens or model output limit was reached" on
# long-doc reprompts where the reasoning tokens alone exceeded 2000.
CORRECT_MAX_COMPLETION_TOKENS = 4000

# OpenAI Batch API hard limits.
BATCH_REQUEST_LIMIT = 50_000
COMPLETION_WINDOW = "24h"

# Polling.
POLL_INITIAL_S = 30
POLL_MAX_S = 120
MAX_WAIT_SECONDS = 24 * 60 * 60  # match the completion window
_TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass
class CorrectionResult:
    out_path: Path
    n_total: int
    n_reprompted: int
    n_post_correction_errors: int
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


def _load_inputs(dataset_name: str, *, source: str = "assignment"):
    """Return (df_input, correction_prompt_template, topic_tree_str, topics_root).

    ``source="assignment"`` reads ``assignment.jsonl`` (default --- normal phase 4).
    ``source="corrected"`` reads ``corrected.jsonl`` instead; used by the
    retry-mode path so only previously-failed rows get reprompted.
    """
    import pandas as pd
    from topicgpt_python.utils import TopicTree

    out_dir = TOPICGPT_ROOT / dataset_name
    if source == "assignment":
        input_path = out_dir / "assignment.jsonl"
    elif source == "corrected":
        input_path = out_dir / "corrected.jsonl"
    else:
        raise ValueError(f"unknown source={source!r}, expected 'assignment' or 'corrected'")
    topic_path = out_dir / "topics_refined.md"
    if not input_path.exists():
        raise FileNotFoundError(
            f"phase 4 needs {source}.jsonl: {input_path}. Run the earlier phase first."
        )
    if not topic_path.exists():
        raise FileNotFoundError(
            f"phase 4 needs phase 2 output: {topic_path}. Run "
            f"`python -m benchmarking.experiments.run_topicgpt --phase refine --only {dataset_name}`."
        )

    df = pd.read_json(input_path, lines=True)
    topics_root = TopicTree().from_topic_list(str(topic_path), from_file=True)
    correction_prompt = (PROMPTS_DIR / "correction.txt").read_text(encoding="utf-8")
    # Match the vendored `correct` function's tree rendering exactly:
    # `"\n".join(topics_root.to_topic_list(desc=True, count=False))`.
    topic_tree_str = "\n".join(topics_root.to_topic_list(desc=True, count=False))
    return df, correction_prompt, topic_tree_str, topics_root


def _identify_targets(df, topics_root) -> tuple[list[int], list[int]]:
    """Run the vendored regex parser to find error + hallucinated row indices."""
    from topicgpt_python.correction import topic_parser

    error, hallucinated = topic_parser(topics_root, df, verbose=False)
    return error, hallucinated


def _split_prompt_template(template: str, tree_str: str) -> tuple[str, str]:
    """Substitute {tree}, then split on {Document}. Returns (cached_prefix, suffix_with_msg_placeholder)."""
    with_tree = template.replace("{tree}", tree_str)
    if "{Document}" not in with_tree:
        raise ValueError("correction prompt template missing {Document} placeholder")
    if "{Message}" not in with_tree:
        raise ValueError("correction prompt template missing {Message} placeholder")
    prefix, suffix_with_msg = with_tree.split("{Document}", maxsplit=1)
    return prefix, suffix_with_msg


def _build_request(custom_id: str, user_text: str) -> dict:
    """One OpenAI Batch chat-completion request body for gpt-5-mini.

    gpt-5-mini rejects temperature/top_p; max_completion_tokens covers both
    reasoning and visible output (see CORRECT_MAX_COMPLETION_TOKENS doc).
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
            "max_completion_tokens": CORRECT_MAX_COMPLETION_TOKENS,
        },
    }


def _write_requests_jsonl(path: Path, requests: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")


def _submit_one_batch(client, requests_path: Path, dataset_name: str, chunk_idx: int) -> str:
    tag = f"[topicgpt/{dataset_name}/phase=correct chunk={chunk_idx}]"
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
            "method": "topicgpt-correction",
        },
    )
    print(f"{tag} submitted batch_id={batch.id}", flush=True)
    return batch.id


def _poll_until_done(client, batch_ids: list[str], dataset_name: str) -> list:
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
                    f"[topicgpt/{dataset_name}/phase=correct] "
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
                    f"[topicgpt/{dataset_name}/phase=correct] "
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
    """Return (custom_id -> response_text, custom_id -> usage_dict, errors)."""
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


def correct(
    dataset_name: str,
    *,
    overwrite: bool = False,
    retry_from_corrected: bool = False,
) -> CorrectionResult:
    """Phase 4 entry point — batch correction via OpenAI Batch API.

    ``retry_from_corrected=True`` re-reads ``corrected.jsonl`` (instead of
    ``assignment.jsonl``), so only rows that *still* failed to parse after
    the previous correction pass get reprompted. Implies overwrite=True for
    the cache check. Used when the prior correction batch hit gpt-5-mini's
    `max_completion_tokens` and we want to retry with a larger budget
    without redoing the rows that succeeded.
    """
    out_dir = TOPICGPT_ROOT / dataset_name
    out_path = out_dir / "corrected.jsonl"
    usage_path = out_dir / "usage_correct.json"

    if retry_from_corrected:
        if not out_path.exists():
            raise FileNotFoundError(
                f"retry_from_corrected=True requires existing {out_path}; "
                f"run a normal correct first."
            )
        overwrite = True
        source = "corrected"
    else:
        source = "assignment"

    if not overwrite and out_path.exists() and out_path.stat().st_size > 0:
        print(f"[topicgpt/{dataset_name}/phase=correct] cache hit -> {out_path}", flush=True)
        with usage_path.open(encoding="utf-8") as f:
            usage = json.load(f)
        return CorrectionResult(
            out_path=out_path,
            n_total=usage.get("n_total", 0),
            n_reprompted=usage.get("n_reprompted", 0),
            n_post_correction_errors=usage.get("n_post_correction_errors", 0),
            usage=usage,
        )

    df, correction_prompt, tree_str, topics_root = _load_inputs(dataset_name, source=source)
    n_total = len(df)
    error_idx, hallucinated_idx = _identify_targets(df, topics_root)
    reprompt_idx = sorted(set(error_idx) | set(hallucinated_idx))
    print(
        f"[topicgpt/{dataset_name}/phase=correct] n_total={n_total} "
        f"n_error={len(error_idx)} n_hallucinated={len(hallucinated_idx)} "
        f"n_to_correct={len(reprompt_idx)}",
        flush=True,
    )

    if not reprompt_idx:
        # Nothing to correct — copy assignment.jsonl forward as corrected.jsonl.
        out_path.write_text(
            "\n".join(json.dumps(rec, ensure_ascii=False) for rec in df.to_dict(orient="records")) + "\n",
            encoding="utf-8",
        )
        usage = {
            "n_total": n_total, "n_reprompted": 0, "n_post_correction_errors": 0,
            "input_tokens": 0, "output_tokens": 0,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
            "cache_hit_rate": 0.0, "model": MODEL,
            "max_completion_tokens": CORRECT_MAX_COMPLETION_TOKENS,
        }
        usage_path.write_text(json.dumps(usage, ensure_ascii=False, indent=2), encoding="utf-8")
        return CorrectionResult(
            out_path=out_path, n_total=n_total, n_reprompted=0,
            n_post_correction_errors=0, usage=usage,
        )

    cached_prefix, suffix_with_msg = _split_prompt_template(correction_prompt, tree_str)
    print(
        f"[topicgpt/{dataset_name}/phase=correct] "
        f"cached_prefix={len(cached_prefix)} chars; suffix_with_msg={len(suffix_with_msg)} chars",
        flush=True,
    )

    client = _get_client()
    requests_dir = out_dir / "_correct_batch_inputs"
    requests_dir.mkdir(parents=True, exist_ok=True)

    # Build requests (one per reprompt_idx).
    all_requests: list[dict] = []
    custom_id_to_row_idx: dict[str, int] = {}
    for i in reprompt_idx:
        row = df.iloc[i]
        doc_text = str(row["prompted_docs"])
        prev_response = str(row["responses"])
        msg = (
            f"Previously, this document was assigned to: {prev_response}. "
            f"Please reassign it to an existing topic in the hierarchy."
        )
        suffix = suffix_with_msg.replace("{Message}", msg)
        user_text = cached_prefix + doc_text + suffix
        cid = f"correct-{i:06d}"
        all_requests.append(_build_request(cid, user_text))
        custom_id_to_row_idx[cid] = int(i)

    # Chunk under the OpenAI Batch hard cap and submit.
    batch_ids: list[str] = []
    chunk_paths: list[Path] = []
    for chunk_idx, start in enumerate(range(0, len(all_requests), BATCH_REQUEST_LIMIT)):
        chunk = all_requests[start : start + BATCH_REQUEST_LIMIT]
        req_path = requests_dir / f"requests_chunk{chunk_idx:02d}.jsonl"
        _write_requests_jsonl(req_path, chunk)
        chunk_paths.append(req_path)
        batch_id = _submit_one_batch(client, req_path, dataset_name, chunk_idx)
        batch_ids.append(batch_id)

    print(
        f"[topicgpt/{dataset_name}/phase=correct] {len(batch_ids)} batch(es) submitted; polling…",
        flush=True,
    )
    finished = _poll_until_done(client, batch_ids, dataset_name)

    all_responses: dict[str, str] = {}
    all_usages: dict[str, dict] = {}
    all_errors: list[dict] = []
    for batch in finished:
        if batch.status != "completed":
            raise RuntimeError(f"batch {batch.id} did not complete: status={batch.status}")
        output_jsonl = client.files.content(batch.output_file_id).text
        responses, usages, errors = _parse_batch_output(output_jsonl)
        all_responses.update(responses)
        all_usages.update(usages)
        all_errors.extend(errors)

    # Overwrite the dataframe rows for successfully reprompted indices.
    # For failed reprompts (custom_id missing from all_responses or in
    # all_errors), leave the original bad response in place --- result_parser
    # will then count them under n_unparseable / n_hallucinated and force-
    # assign to the largest cluster per its fallback policy.
    n_overwritten = 0
    for cid, new_response in all_responses.items():
        row_idx = custom_id_to_row_idx[cid]
        df.at[row_idx, "responses"] = new_response
        n_overwritten += 1

    # Count residual errors after correction: rerun topic_parser to see how
    # many rows STILL fail to parse.
    post_error_idx, post_hallucinated_idx = _identify_targets(df, topics_root)
    n_post_correction_errors = len(set(post_error_idx) | set(post_hallucinated_idx))

    # Write corrected.jsonl in the same row order as assignment.jsonl.
    rows_out = df.to_dict(orient="records")
    out_path.write_text(
        "\n".join(json.dumps(rec, ensure_ascii=False) for rec in rows_out) + "\n",
        encoding="utf-8",
    )

    total_usage = _summarize_usage(all_usages)
    total_usage.update(
        {
            "cache_hit_rate": _cache_hit_rate(total_usage),
            "n_total": n_total,
            "n_reprompted": len(reprompt_idx),
            "n_overwritten": n_overwritten,
            "n_batch_failures": len(all_errors),
            "n_post_correction_errors": n_post_correction_errors,
            "model": MODEL,
            "max_completion_tokens": CORRECT_MAX_COMPLETION_TOKENS,
        }
    )
    # On --retry-correct, ACCUMULATE token totals into the prior pass instead of
    # overwriting it: the retry reprompts a *subset* of rows the initial pass
    # already paid for, so the initial pass's (larger) spend must stay in the
    # cost. Overwriting silently undercounts downstream — exactly what happened
    # to 20NG before this fix (its initial 2000-budget pass had to be recovered
    # as a peer-rate estimate; see result_parser's correct_initial_estimated).
    if retry_from_corrected and usage_path.exists():
        prior = json.loads(usage_path.read_text(encoding="utf-8"))
        token_fields = (
            "input_tokens", "output_tokens",
            "cache_read_input_tokens", "cache_creation_input_tokens",
        )
        pass_keys = (*token_fields, "n_reprompted", "max_completion_tokens")
        passes = prior.get("passes") or [{k: prior.get(k, 0) for k in pass_keys}]
        passes.append({k: total_usage.get(k, 0) for k in pass_keys})
        for k in token_fields:
            total_usage[k] = sum(int(p.get(k, 0) or 0) for p in passes)
        total_usage["n_reprompted"] = sum(int(p.get("n_reprompted", 0) or 0) for p in passes)
        total_usage["n_passes"] = len(passes)
        total_usage["passes"] = passes
        total_usage["cache_hit_rate"] = _cache_hit_rate(total_usage)
    usage_path.write_text(json.dumps(total_usage, ensure_ascii=False, indent=2), encoding="utf-8")

    # Clean up batch input files.
    for p in chunk_paths:
        p.unlink(missing_ok=True)
    if requests_dir.exists() and not any(requests_dir.iterdir()):
        requests_dir.rmdir()

    print(
        f"[topicgpt/{dataset_name}/phase=correct] -> {out_path} | "
        f"n_reprompted={len(reprompt_idx)} "
        f"n_batch_failures={len(all_errors)} "
        f"n_post_correction_errors={n_post_correction_errors} | "
        f"usage={total_usage}",
        flush=True,
    )
    return CorrectionResult(
        out_path=out_path,
        n_total=n_total,
        n_reprompted=len(reprompt_idx),
        n_post_correction_errors=n_post_correction_errors,
        usage=total_usage,
    )
