"""Cached OpenAI embeddings via the Batch API for processed datasets.

Mirrors the interface of `benchmarking.embeddings.sbert.embed_dataset` so that
downstream baselines (LLM-embedding+kmeans, plus any future consumer) can swap
encoders without touching their own code.

Cache layout (same shape as the SBERT cache):
    data/embeddings/<model_shortname>/<dataset>.npy            # (n_docs, dim) float32, raw (NOT L2-normalized)
    data/embeddings/<model_shortname>/<dataset>.meta.json      # provenance + sha256 + actual token usage and USD

All bulk embedding calls go through the OpenAI Batch API (50% discount, ≤24h
SLA) per SPEC §5.6.3. We persist actual tokens and USD into the sidecar so
cache-hit runs report the real amount paid, not a re-estimate.

Two public entry points:
- `embed_dataset(name)` — single dataset, used by smoke tests.
- `embed_datasets(names)` — submits batches for *all* listed datasets up
  front, then polls every in-flight batch in one combined loop. Wall-clock
  is bounded by the slowest dataset rather than the sum.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from benchmarking.data_processing.load import load_processed
from benchmarking.embeddings import EmbeddingCache, model_shortname, texts_sha256
from benchmarking.paths import DATA

EMBEDDINGS_ROOT = DATA / "embeddings"

# text-embedding-3-large per-token pricing on the OpenAI Batch API (50% off
# the sync $0.13 / 1M-token rate). Pinned here, not loaded from anywhere,
# because this is a paper artefact: it must report what we actually paid,
# not what current pricing would be at re-run time.
BATCH_PRICE_USD_PER_1M_TOKENS = 0.065
BATCH_PRICE_BASIS = "openai_batch_api_50pct_discount_2026_05"

# OpenAI Batch API hard limits.
MAX_REQUESTS_PER_BATCH = 50_000
MAX_TOKENS_PER_INPUT = 8192  # text-embedding-3-large input ceiling

# Terminal statuses for a batch (no further polling needed).
_TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


def _get_client():
    from openai import OpenAI

    from benchmarking.secrets import load_secrets_into_env

    load_secrets_into_env()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Export the env var or add it to secrets.json at the project root."
        )
    return OpenAI(api_key=api_key)


def _get_tokenizer():
    """Return the cl100k_base tokenizer used by text-embedding-3-large."""
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def _truncate_to_token_limit(text: str, tokenizer, limit: int = MAX_TOKENS_PER_INPUT) -> str:
    """Truncate `text` so it fits within `limit` tokens. No-op if already under."""
    tokens = tokenizer.encode(text)
    if len(tokens) <= limit:
        return text
    return tokenizer.decode(tokens[:limit])


def _chunked(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _write_requests_jsonl(
    path: Path,
    *,
    texts: list[str],
    custom_id_offset: int,
    model_name: str,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        for local_idx, text in enumerate(texts):
            global_idx = custom_id_offset + local_idx
            req = {
                "custom_id": f"doc-{global_idx}",
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {"model": model_name, "input": text},
            }
            f.write(json.dumps(req, ensure_ascii=False) + "\n")


def _parse_batch_output(
    output_jsonl: str,
    *,
    expected_dim: int | None,
) -> tuple[dict[int, np.ndarray], int, list[dict]]:
    """Return (idx -> embedding, total_input_tokens, error_records).

    `error_records` are entries with non-null error or non-200 status — they need
    to be retried via the sync API.
    """
    embeddings: dict[int, np.ndarray] = {}
    total_tokens = 0
    errors: list[dict] = []

    for line in output_jsonl.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        custom_id = rec.get("custom_id", "")
        if not custom_id.startswith("doc-"):
            raise ValueError(f"unexpected custom_id: {custom_id!r}")
        idx = int(custom_id.split("-", 1)[1])

        err = rec.get("error")
        resp = rec.get("response")
        if err or not resp or resp.get("status_code") != 200:
            errors.append({"idx": idx, "error": err, "response": resp})
            continue

        body = resp["body"]
        vec = np.asarray(body["data"][0]["embedding"], dtype=np.float32)
        if expected_dim is not None and vec.shape[0] != expected_dim:
            raise ValueError(
                f"dim mismatch at idx={idx}: got {vec.shape[0]}, expected {expected_dim}"
            )
        embeddings[idx] = vec
        total_tokens += int(body["usage"]["total_tokens"])

    return embeddings, total_tokens, errors


def _sync_retry_one(client, *, text: str, model_name: str) -> tuple[np.ndarray, int]:
    """Fallback retry for individual requests that errored inside the batch."""
    resp = client.embeddings.create(model=model_name, input=text)
    vec = np.asarray(resp.data[0].embedding, dtype=np.float32)
    return vec, int(resp.usage.total_tokens)


@dataclass
class _ChunkInflight:
    """One Batch API job, possibly one of several for a single dataset."""

    dataset_name: str
    chunk_idx: int
    n_chunks_in_dataset: int
    batch_id: str
    input_file_id: str
    chunk_offset: int  # global doc index where this chunk starts
    chunk_texts: list[str]  # kept for sync-API retry on per-request errors
    n_requests: int
    submitted_at: str
    # Terminal state, set when the batch reaches a terminal status:
    status: str = "submitted"
    output_file_id: str | None = None
    completed_at: str | None = None


@dataclass
class _PendingDataset:
    """Per-dataset bundle of in-flight chunks plus the metadata needed to finalise."""

    dataset_name: str
    model_name: str
    short: str
    npy_path: Path
    meta_path: Path
    requests_dir: Path
    texts: list[str]            # post-truncation
    original_texts_sha256: str  # of the *original* (pre-truncation) texts
    chunks: list[_ChunkInflight] = field(default_factory=list)


def _paths_for(dataset_name: str, short: str) -> tuple[Path, Path, Path]:
    out_dir = EMBEDDINGS_ROOT / short
    out_dir.mkdir(parents=True, exist_ok=True)
    return (
        out_dir / f"{dataset_name}.npy",
        out_dir / f"{dataset_name}.meta.json",
        out_dir / f".batch_inputs_{dataset_name}",
    )


def _try_load_cache(
    *,
    dataset_name: str,
    model_name: str,
    texts: list[str],
    expected_sha: str,
    short: str,
    npy_path: Path,
    meta_path: Path,
) -> EmbeddingCache | None:
    if not (npy_path.exists() and meta_path.exists()):
        return None
    sidecar = json.loads(meta_path.read_text(encoding="utf-8"))
    if sidecar.get("texts_sha256") != expected_sha or sidecar.get("n_docs") != len(texts):
        return None
    arr = np.load(npy_path)
    assert arr.shape[0] == len(texts), (
        f"cache shape mismatch: {arr.shape[0]} rows vs {len(texts)} texts"
    )
    return EmbeddingCache(
        embeddings=arr,
        model=model_name,
        short=short,
        npy_path=npy_path,
        meta_path=meta_path,
        n_docs=arr.shape[0],
        dim=int(arr.shape[1]),
        cache_hit=True,
    )


def _submit_one_dataset(
    *,
    client,
    tokenizer,
    dataset_name: str,
    model_name: str,
) -> EmbeddingCache | _PendingDataset:
    """Return a cache hit if available, otherwise submit batches and return a pending bundle."""
    ds = load_processed(dataset_name)
    texts = [d["text"] for d in ds.documents]
    short = model_shortname(model_name)
    npy_path, meta_path, requests_dir = _paths_for(dataset_name, short)
    expected_sha = texts_sha256(texts)

    cached = _try_load_cache(
        dataset_name=dataset_name,
        model_name=model_name,
        texts=texts,
        expected_sha=expected_sha,
        short=short,
        npy_path=npy_path,
        meta_path=meta_path,
    )
    if cached is not None:
        return cached

    print(
        f"[openai_embeddings/{dataset_name}] truncating long inputs to {MAX_TOKENS_PER_INPUT} tokens..."
    )
    truncated = [_truncate_to_token_limit(t, tokenizer) for t in texts]
    requests_dir.mkdir(parents=True, exist_ok=True)

    chunks_iter = list(_chunked(truncated, MAX_REQUESTS_PER_BATCH))
    print(
        f"[openai_embeddings/{dataset_name}] {len(texts)} docs -> {len(chunks_iter)} batch(es) "
        f"of up to {MAX_REQUESTS_PER_BATCH}."
    )

    pending = _PendingDataset(
        dataset_name=dataset_name,
        model_name=model_name,
        short=short,
        npy_path=npy_path,
        meta_path=meta_path,
        requests_dir=requests_dir,
        texts=truncated,
        original_texts_sha256=expected_sha,
    )

    for chunk_idx, chunk_texts in enumerate(chunks_iter):
        offset = chunk_idx * MAX_REQUESTS_PER_BATCH
        tag = f"[openai_embeddings/{dataset_name} batch {chunk_idx + 1}/{len(chunks_iter)}]"

        req_path = requests_dir / f"requests_chunk{chunk_idx:02d}.jsonl"
        _write_requests_jsonl(
            req_path,
            texts=chunk_texts,
            custom_id_offset=offset,
            model_name=model_name,
        )

        print(f"{tag} uploading {len(chunk_texts)} requests...")
        with req_path.open("rb") as fh:
            input_file = client.files.create(file=fh, purpose="batch")

        submitted_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        batch = client.batches.create(
            input_file_id=input_file.id,
            endpoint="/v1/embeddings",
            completion_window="24h",
            metadata={
                "dataset": dataset_name,
                "model": model_name,
                "chunk_index": str(chunk_idx),
                "n_chunks": str(len(chunks_iter)),
            },
        )
        print(f"{tag} submitted batch_id={batch.id}")

        pending.chunks.append(
            _ChunkInflight(
                dataset_name=dataset_name,
                chunk_idx=chunk_idx,
                n_chunks_in_dataset=len(chunks_iter),
                batch_id=batch.id,
                input_file_id=input_file.id,
                chunk_offset=offset,
                chunk_texts=chunk_texts,
                n_requests=len(chunk_texts),
                submitted_at=submitted_at,
            )
        )

    return pending


def _poll_all_until_done(
    client,
    pendings: list[_PendingDataset],
    *,
    max_wait_seconds: int,
    poll_initial_s: int,
    poll_max_s: int,
) -> None:
    """Mutate `pendings` in place: poll every in-flight chunk across all datasets in a
    single combined loop. Exits when no chunk is still in-progress."""
    inflight: list[_ChunkInflight] = [c for p in pendings for c in p.chunks if c.status not in _TERMINAL_STATUSES]
    if not inflight:
        return

    deadline = time.monotonic() + max_wait_seconds
    interval = poll_initial_s
    last_log = 0.0

    while inflight:
        if time.monotonic() > deadline:
            still = [(c.dataset_name, c.batch_id, c.status) for c in inflight]
            raise TimeoutError(f"polling deadline exceeded; {len(inflight)} chunk(s) still in-flight: {still[:5]}")

        for chunk in list(inflight):
            batch = client.batches.retrieve(chunk.batch_id)
            chunk.status = batch.status
            if batch.status in _TERMINAL_STATUSES:
                chunk.output_file_id = getattr(batch, "output_file_id", None)
                chunk.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                inflight.remove(chunk)
                counts = batch.request_counts
                print(
                    f"[openai_embeddings/{chunk.dataset_name} batch {chunk.chunk_idx + 1}"
                    f"/{chunk.n_chunks_in_dataset}] terminal status={batch.status} "
                    f"completed={counts.completed}/{counts.total} failed={counts.failed}"
                )

        if not inflight:
            break

        # Periodic compact status snapshot of everything still in flight.
        now = time.monotonic()
        if now - last_log >= 60:
            for chunk in inflight:
                batch = client.batches.retrieve(chunk.batch_id)
                counts = batch.request_counts
                print(
                    f"[openai_embeddings/{chunk.dataset_name} batch {chunk.chunk_idx + 1}"
                    f"/{chunk.n_chunks_in_dataset}] polling... status={batch.status} "
                    f"completed={counts.completed}/{counts.total} failed={counts.failed}"
                )
            last_log = now

        time.sleep(interval)
        if time.monotonic() - (deadline - max_wait_seconds) > 600:
            interval = poll_max_s


def _download_and_save(client, pending: _PendingDataset) -> EmbeddingCache:
    """Download outputs for a fully-terminal pending dataset, write cache, return EmbeddingCache."""
    failures = [c for c in pending.chunks if c.status != "completed"]
    if failures:
        details = ", ".join(f"chunk{c.chunk_idx}={c.status}({c.batch_id})" for c in failures)
        raise RuntimeError(
            f"[openai_embeddings/{pending.dataset_name}] {len(failures)} chunk(s) did not complete: {details}"
        )

    all_embeddings: dict[int, np.ndarray] = {}
    total_tokens = 0
    batch_provenance: list[dict] = []

    for chunk in pending.chunks:
        tag = f"[openai_embeddings/{pending.dataset_name} batch {chunk.chunk_idx + 1}/{chunk.n_chunks_in_dataset}]"

        output_jsonl = client.files.content(chunk.output_file_id).text
        embeddings, chunk_tokens, errors = _parse_batch_output(output_jsonl, expected_dim=None)

        if errors:
            print(f"{tag} {len(errors)} request(s) errored in batch; retrying via sync API...")
            for err_rec in errors:
                idx = err_rec["idx"]
                local_idx = idx - chunk.chunk_offset
                vec, used = _sync_retry_one(
                    client,
                    text=chunk.chunk_texts[local_idx],
                    model_name=pending.model_name,
                )
                embeddings[idx] = vec
                chunk_tokens += used

        all_embeddings.update(embeddings)
        total_tokens += chunk_tokens
        batch_provenance.append(
            {
                "batch_id": chunk.batch_id,
                "input_file_id": chunk.input_file_id,
                "output_file_id": chunk.output_file_id,
                "n_requests": chunk.n_requests,
                "submitted_at": chunk.submitted_at,
                "completed_at": chunk.completed_at,
                "status": chunk.status,
                "n_errors": len(errors),
            }
        )

    if len(all_embeddings) != len(pending.texts):
        missing = sorted(set(range(len(pending.texts))) - set(all_embeddings.keys()))[:10]
        raise RuntimeError(
            f"[openai_embeddings/{pending.dataset_name}] embeddings count mismatch: "
            f"got {len(all_embeddings)} for {len(pending.texts)} docs; first missing: {missing}"
        )

    dim = next(iter(all_embeddings.values())).shape[0]
    arr = np.empty((len(pending.texts), dim), dtype=np.float32)
    for idx, vec in all_embeddings.items():
        arr[idx] = vec

    usd = total_tokens * BATCH_PRICE_USD_PER_1M_TOKENS / 1_000_000

    import openai as _openai_pkg

    np.save(pending.npy_path, arr)
    pending.meta_path.write_text(
        json.dumps(
            {
                "model_name": pending.model_name,
                "model_shortname": pending.short,
                "openai_version": getattr(_openai_pkg, "__version__", "unknown"),
                "dataset": pending.dataset_name,
                "n_docs": len(pending.texts),
                "dim": int(arr.shape[1]),
                "dtype": str(arr.dtype),
                "normalized": False,
                "texts_sha256": pending.original_texts_sha256,
                "api_mode": "batch",
                "cost": {
                    "input_tokens": int(total_tokens),
                    "usd": float(usd),
                    "rate_per_1m_input_tokens_usd": BATCH_PRICE_USD_PER_1M_TOKENS,
                    "rate_basis": BATCH_PRICE_BASIS,
                },
                "batches": batch_provenance,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for f in pending.requests_dir.glob("*.jsonl"):
        f.unlink()
    if pending.requests_dir.exists():
        pending.requests_dir.rmdir()

    return EmbeddingCache(
        embeddings=arr,
        model=pending.model_name,
        short=pending.short,
        npy_path=pending.npy_path,
        meta_path=pending.meta_path,
        n_docs=arr.shape[0],
        dim=int(arr.shape[1]),
        cache_hit=False,
    )


def embed_datasets(
    dataset_names: list[str],
    model_name: str = "text-embedding-3-large",
    *,
    max_wait_seconds: int = 12 * 60 * 60,
    poll_initial_s: int = 30,
    poll_max_s: int = 120,
) -> dict[str, EmbeddingCache]:
    """Embed multiple datasets in parallel via the OpenAI Batch API.

    Submits all needed batches up front (skipping cache hits), then polls
    every in-flight batch in one combined loop. Wall-clock is bounded by
    the slowest dataset rather than the sum.
    """
    client = _get_client()
    tokenizer = _get_tokenizer()

    submissions: dict[str, EmbeddingCache | _PendingDataset] = {}
    for name in dataset_names:
        submissions[name] = _submit_one_dataset(
            client=client,
            tokenizer=tokenizer,
            dataset_name=name,
            model_name=model_name,
        )

    pendings = [s for s in submissions.values() if isinstance(s, _PendingDataset)]
    n_hits = sum(1 for s in submissions.values() if isinstance(s, EmbeddingCache))
    print(
        f"[openai_embeddings] {n_hits} cache hit(s), {len(pendings)} dataset(s) running "
        f"({sum(len(p.chunks) for p in pendings)} chunk(s) total). Polling..."
    )

    _poll_all_until_done(
        client,
        pendings,
        max_wait_seconds=max_wait_seconds,
        poll_initial_s=poll_initial_s,
        poll_max_s=poll_max_s,
    )

    out: dict[str, EmbeddingCache] = {}
    for name, s in submissions.items():
        if isinstance(s, EmbeddingCache):
            out[name] = s
        else:
            out[name] = _download_and_save(client, s)
    return out


def embed_dataset(
    dataset_name: str,
    model_name: str = "text-embedding-3-large",
    *,
    max_wait_seconds: int = 12 * 60 * 60,
    poll_initial_s: int = 30,
    poll_max_s: int = 120,
) -> EmbeddingCache:
    """Single-dataset convenience wrapper around `embed_datasets`."""
    return embed_datasets(
        [dataset_name],
        model_name,
        max_wait_seconds=max_wait_seconds,
        poll_initial_s=poll_initial_s,
        poll_max_s=poll_max_s,
    )[dataset_name]


def cost_from_meta(meta_path: Path) -> dict:
    """Read the persisted cost block from a sidecar. Used by runners to report actual paid cost."""
    sidecar = json.loads(meta_path.read_text(encoding="utf-8"))
    return sidecar.get("cost", {})
