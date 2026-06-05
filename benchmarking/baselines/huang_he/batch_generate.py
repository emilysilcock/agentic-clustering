"""Phase 1: label generation via OpenAI Batch (gpt-5-mini).

Replaces upstream's ``label_generation.label_generation`` sync loop. Same
prompt template (``prompt_construct_generate_label``, byte-identical to
upstream — including the ``"classicifation"`` typo); different dispatch.

## Steps

1. Load ``data/huang_he/<ds>/input.jsonl`` (built by ``dataset_adapter``).
2. ``random.seed(0)`` + ``random.shuffle(docs)`` --- mirrors upstream's
   pre-chunk shuffle, but deterministic (upstream is unseeded).
3. Chunk into batches of B=15 documents (upstream default).
4. Build one prompt per chunk via the vendored
   ``prompt_construct_generate_label`` with ``given_labels=[]`` (the
   0%-seed configuration --- SPEC §5.6.2).
5. Submit all chunks to OpenAI Batch as a single submission group;
   ``response_format={"type":"json_object"}`` so output is valid JSON.
6. Parse each response with ``json.loads``; extract the first list-valued
   field (mirrors upstream's ``response[list(response.keys())[0]]``).
   Skip chunks whose response fails to parse or doesn't contain a list.
7. Filter out the upstream's "meaningless" placeholder labels
   (``"unknown_topic"`` / ``"new_label"`` substring tests), preserve
   insertion order, dedupe globally.
8. Write the pre-merge label list to ``labels_pre_merge.json`` and the
   per-phase usage breakdown to ``usage_generate.json``.

## Cost / cache

Each chunk has a distinct B=15 sentences, so OpenAI's auto-cache does not
hit between chunks. The prompt prefix (``"Given the labels, under a text
classicifation scenario, ... Labels: []"``) is short anyway. We record
``cached_tokens`` for completeness; expect near-zero cache hit rate.

Phase 1 is the cheap-tier route per SPEC §5.6.2 (>1,000-text rule does
not apply --- 6k total calls is per-sweep, but each request is one of
6,308 across the 7 datasets; routed to ``gpt-5-mini`` for cost not
volume). See ``CHANGES.md``.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import benchmarking  # noqa: F401 — truststore.inject_into_ssl()
import benchmarking.baselines.huang_he  # noqa: F401 — _vendored on sys.path

from benchmarking.baselines.huang_he import _batch_common as bc
from benchmarking.baselines.huang_he.dataset_adapter import HUANG_HE_ROOT
from benchmarking.baselines.huang_he.prompts import prompt_construct_generate_label

MODEL = "gpt-5-mini"
SYSTEM_MESSAGE = "You are a helpful assistant designed to output JSON."
RESPONSE_FORMAT = {"type": "json_object"}

# Upstream `prompt_construct_generate_label` returns a short list (15
# elements ~ a few hundred tokens of visible JSON). gpt-5-mini's
# ``max_completion_tokens`` budget covers reasoning + visible output;
# at 1500 the Banking77 pilot saw 24/206 chunks (11.7%) exhaust the
# budget on reasoning and emit empty content (finish_reason=length,
# avg observed reasoning+output = 1149 tokens, tail to 1500+).
# 4000 gives comfortable headroom — incremental cost is small because
# only the tail-end chunks consume above the original 1500.
MAX_COMPLETION_TOKENS = 4000

CHUNK_SIZE = 15            # upstream default (`label_generation.py:163`)
SHUFFLE_SEED = 0           # deterministic — upstream is unseeded

# Upstream filter substrings from `label_generation.py:94`. We preserve
# them so a label like "new_label_3" is dropped in the same way upstream
# would drop it.
_BAD_LABEL_SUBSTRINGS = ("unknown_topic", "new_label")


@dataclass
class GenerationResult:
    out_path: Path
    n_chunks_submitted: int
    n_chunks_parsed: int
    n_labels_pre_filter: int
    n_labels: int
    usage: dict


def _load_docs(dataset_name: str) -> list[dict]:
    in_path = HUANG_HE_ROOT / dataset_name / "input.jsonl"
    if not in_path.exists():
        raise FileNotFoundError(
            f"phase 1 needs phase 0 output: {in_path}. Run "
            f"`python -m benchmarking.baselines.huang_he.dataset_adapter "
            f"--only {dataset_name}`."
        )
    return [
        json.loads(line)
        for line in in_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _build_requests(docs: list[dict]) -> tuple[list[dict], dict[str, list[str]]]:
    """Build one chat-completions request per B=15 chunk.

    Returns ``(requests, custom_id -> sentences)`` so we can debug a
    chunk's response after the fact.
    """
    requests: list[dict] = []
    chunk_sentences: dict[str, list[str]] = {}

    for chunk_idx, start in enumerate(range(0, len(docs), CHUNK_SIZE)):
        chunk = docs[start : start + CHUNK_SIZE]
        sentences = [doc["input"] for doc in chunk]
        prompt = prompt_construct_generate_label(sentences, [])
        cid = f"gen-{chunk_idx:06d}"
        requests.append(
            bc.build_chat_request(
                custom_id=cid,
                user_text=prompt,
                model=MODEL,
                system_message=SYSTEM_MESSAGE,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                response_format=RESPONSE_FORMAT,
            )
        )
        chunk_sentences[cid] = sentences

    return requests, chunk_sentences


def _parse_one_response(response_text: str) -> list[str] | None:
    """Return the first list-valued field of the JSON response, or None.

    Upstream uses ``response[list(response.keys())[0]]`` which assumes the
    LLM returns a JSON object whose first value is the list of labels (the
    prompt examples it with ``{"labels": [...]}``). We do the same, but
    safely via ``json.loads`` rather than ``eval``.
    """
    try:
        obj = json.loads(response_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not obj:
        return None
    first_value = obj[next(iter(obj))]
    if not isinstance(first_value, list):
        return None
    return [str(x) for x in first_value]


def _aggregate_labels(per_chunk_labels: list[list[str]]) -> tuple[list[str], int]:
    """Union per-chunk label outputs into the pre-merge list.

    Mirrors upstream's accumulation order (first occurrence wins, dedupe
    by exact string equality, filter out meaningless placeholders).
    Returns ``(labels, n_pre_filter_total)``.
    """
    all_labels: list[str] = []
    seen: set[str] = set()
    n_pre_filter = 0
    for labels in per_chunk_labels:
        for label in labels:
            n_pre_filter += 1
            if any(bad in label for bad in _BAD_LABEL_SUBSTRINGS):
                continue
            if label in seen:
                continue
            seen.add(label)
            all_labels.append(label)
    return all_labels, n_pre_filter


def generate(dataset_name: str, *, overwrite: bool = False) -> GenerationResult:
    """Phase 1 entry point."""
    out_dir = HUANG_HE_ROOT / dataset_name
    out_path = out_dir / "labels_pre_merge.json"
    usage_path = out_dir / "usage_generate.json"

    if not overwrite and out_path.exists() and out_path.stat().st_size > 0:
        with usage_path.open(encoding="utf-8") as f:
            usage = json.load(f)
        with out_path.open(encoding="utf-8") as f:
            labels = json.load(f)
        print(
            f"[huang_he/{dataset_name}/phase=generate] cache hit -> {out_path} "
            f"(n_labels={len(labels)})",
            flush=True,
        )
        return GenerationResult(
            out_path=out_path,
            n_chunks_submitted=int(usage.get("n_chunks_submitted", 0)),
            n_chunks_parsed=int(usage.get("n_chunks_parsed", 0)),
            n_labels_pre_filter=int(usage.get("n_labels_pre_filter", 0)),
            n_labels=len(labels),
            usage=usage,
        )

    docs = _load_docs(dataset_name)
    random.seed(SHUFFLE_SEED)
    random.shuffle(docs)

    requests, _ = _build_requests(docs)
    log_prefix = f"[huang_he/{dataset_name}/phase=generate]"
    print(
        f"{log_prefix} {len(docs)} docs -> {len(requests)} chunks "
        f"(chunk_size={CHUNK_SIZE}, model={MODEL})",
        flush=True,
    )

    client = bc.get_openai_client()
    responses, usages, errors = bc.submit_and_collect(
        client,
        requests=requests,
        out_dir=out_dir,
        log_prefix=log_prefix,
        method_tag="huang_he-generate",
        dataset_name=dataset_name,
        batch_inputs_subdir="_generate_batch_inputs",
    )

    # Parse responses in original request order so per_chunk_labels is
    # ordered by chunk_idx (matters for deterministic aggregation).
    per_chunk: list[list[str]] = []
    n_parsed = 0
    for req in requests:
        cid = req["custom_id"]
        text = responses.get(cid)
        if text is None:
            continue
        labels = _parse_one_response(text)
        if labels is None:
            continue
        per_chunk.append(labels)
        n_parsed += 1

    all_labels, n_pre_filter = _aggregate_labels(per_chunk)

    usage = bc.summarize_usage(usages)
    usage.update(
        {
            "model": MODEL,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "n_chunks_submitted": len(requests),
            "n_chunks_parsed": n_parsed,
            "n_chunk_failures": len(errors) + (len(requests) - len(responses)),
            "n_chunks_response_unparseable": len(responses) - n_parsed,
            "n_labels_pre_filter": n_pre_filter,
            "n_labels": len(all_labels),
            "chunk_size": CHUNK_SIZE,
            "shuffle_seed": SHUFFLE_SEED,
        }
    )

    out_path.write_text(
        json.dumps(all_labels, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    usage_path.write_text(
        json.dumps(usage, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"{log_prefix} -> {out_path} | "
        f"n_chunks_submitted={len(requests)} n_chunks_parsed={n_parsed} "
        f"n_labels_pre_filter={n_pre_filter} n_labels={len(all_labels)} | "
        f"usage={usage}",
        flush=True,
    )
    return GenerationResult(
        out_path=out_path,
        n_chunks_submitted=len(requests),
        n_chunks_parsed=n_parsed,
        n_labels_pre_filter=n_pre_filter,
        n_labels=len(all_labels),
        usage=usage,
    )
