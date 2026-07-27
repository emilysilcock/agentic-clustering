"""One-shot recovery for the OpenAI Batch API run that died on the Cloudflare 504.

What happened: 7 batches all completed on OpenAI's side, but the download step
errored out after only 4 were saved (banking77, clinc150, massive_intent,
massive_domain). The remaining 3 (goemotions, twenty_newsgroups, stackexchange)
are "completed" on OpenAI's side, with downloadable output files. We never
fetched those, so two subsequent attempts re-submitted duplicates, leaving us
with 6 in-flight duplicate batches plus the 3 paid-for orphans.

This script:
  1. Cancels every active duplicate batch for the 3 missing datasets.
  2. Downloads the orphan completed batches and writes the cache files
     (`<dataset>.npy` + `<dataset>.meta.json`) that `embed_dataset` expects.

After running, `run_openai_embedding_kmeans` will see all 7 as cache hits.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from benchmarking.data_processing.load import load_processed
from benchmarking.embeddings import model_shortname, texts_sha256
from benchmarking.embeddings.openai_embeddings import (
    BATCH_PRICE_BASIS,
    BATCH_PRICE_USD_PER_1M_TOKENS,
    EMBEDDINGS_ROOT,
    _get_client,
    _parse_batch_output,
)


def _download_file_streaming(client, file_id: str, dest_path: Path, *, max_attempts: int = 6) -> None:
    """Stream an OpenAI file to disk with retry on Cloudflare 504s.

    The default `client.files.content(id).text` loads the entire body in
    memory and waits for the full transfer; for ~600 MB outputs that
    routinely 504s through Cloudflare. Streaming via `with_streaming_response`
    + `iter_bytes` keeps the TCP connection active and lets us retry on
    transient gateway errors.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            with client.files.with_streaming_response.content(file_id) as resp:
                with dest_path.open("wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            return
        except Exception as exc:
            last_exc = exc
            wait = min(30 * attempt, 300)
            print(
                f"  download attempt {attempt}/{max_attempts} failed ({type(exc).__name__}: "
                f"{str(exc)[:120]!r}); retrying in {wait}s",
                flush=True,
            )
            time.sleep(wait)
    raise RuntimeError(f"file download failed after {max_attempts} attempts: {last_exc}")

MISSING = ["goemotions", "twenty_newsgroups", "stackexchange"]
MODEL = "text-embedding-3-large"
SHORT = model_shortname(MODEL)
ACTIVE_STATUSES = {"validating", "in_progress", "finalizing"}


def main() -> None:
    client = _get_client()

    # 1. Walk all recent batches, partition into orphan-completed and active-duplicate.
    print("[recover] listing recent batches...")
    batches = list(client.batches.list(limit=100).data)
    by_dataset_completed: dict[str, list] = {ds: [] for ds in MISSING}
    by_dataset_active: dict[str, list] = {ds: [] for ds in MISSING}

    for b in batches:
        ds = (b.metadata or {}).get("dataset")
        if ds not in MISSING:
            continue
        if b.status == "completed":
            by_dataset_completed[ds].append(b)
        elif b.status in ACTIVE_STATUSES:
            by_dataset_active[ds].append(b)

    # 2. Cancel every active duplicate (we'll use the orphan completed output instead).
    print("[recover] cancelling active duplicates...")
    for ds in MISSING:
        for b in by_dataset_active[ds]:
            try:
                client.batches.cancel(b.id)
                print(f"  cancelled {b.id}  ds={ds}  status_was={b.status}")
            except Exception as exc:
                print(f"  cancel failed for {b.id}  ds={ds}: {exc}")

    # 3. For each missing dataset, pick the most-recent completed orphan and download.
    import openai as _openai_pkg

    for ds in MISSING:
        completed = by_dataset_completed[ds]
        if not completed:
            raise RuntimeError(f"no completed orphan batch found for {ds}")
        completed.sort(key=lambda b: b.created_at, reverse=True)
        b = completed[0]
        print(f"[recover/{ds}] downloading batch {b.id} (created={b.created_at})")

        # Load processed docs to validate dim + compute texts_sha256.
        loaded = load_processed(ds)
        texts = [d["text"] for d in loaded.documents]
        expected_sha = texts_sha256(texts)

        # Stream the output file to disk first; large files (goemotions ~600 MB)
        # routinely 504 through Cloudflare with the blocking .text path.
        tmp_path = EMBEDDINGS_ROOT / SHORT / f".batch_output_{ds}.jsonl"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        _download_file_streaming(client, b.output_file_id, tmp_path)
        output_jsonl = tmp_path.read_text(encoding="utf-8")
        embeddings, total_tokens, errors = _parse_batch_output(
            output_jsonl, expected_dim=None
        )
        if errors:
            print(f"  WARN: {len(errors)} per-request errors in this batch — will need sync retry")
            raise RuntimeError(
                f"{ds}: batch has {len(errors)} request-level errors; "
                f"this script doesn't implement sync retry. Re-run the full pipeline instead."
            )

        if len(embeddings) != len(texts):
            missing_idx = sorted(set(range(len(texts))) - set(embeddings.keys()))[:5]
            raise RuntimeError(
                f"{ds}: embeddings count mismatch — got {len(embeddings)} for {len(texts)} docs "
                f"(first missing: {missing_idx})"
            )

        dim = next(iter(embeddings.values())).shape[0]
        arr = np.empty((len(texts), dim), dtype=np.float32)
        for idx, vec in embeddings.items():
            arr[idx] = vec

        out_dir = EMBEDDINGS_ROOT / SHORT
        out_dir.mkdir(parents=True, exist_ok=True)
        npy_path = out_dir / f"{ds}.npy"
        meta_path = out_dir / f"{ds}.meta.json"

        np.save(npy_path, arr)
        usd = total_tokens * BATCH_PRICE_USD_PER_1M_TOKENS / 1_000_000
        meta = {
            "model_name": MODEL,
            "model_shortname": SHORT,
            "openai_version": getattr(_openai_pkg, "__version__", "unknown"),
            "dataset": ds,
            "n_docs": len(texts),
            "dim": int(arr.shape[1]),
            "dtype": str(arr.dtype),
            "normalized": False,
            "texts_sha256": expected_sha,
            "api_mode": "batch",
            "cost": {
                "input_tokens": int(total_tokens),
                "usd": float(usd),
                "rate_per_1m_input_tokens_usd": BATCH_PRICE_USD_PER_1M_TOKENS,
                "rate_basis": BATCH_PRICE_BASIS,
            },
            "batches": [
                {
                    "batch_id": b.id,
                    "input_file_id": b.input_file_id,
                    "output_file_id": b.output_file_id,
                    "n_requests": len(texts),
                    "submitted_at": str(b.created_at),
                    "completed_at": str(b.completed_at),
                    "status": b.status,
                    "n_errors": 0,
                    "recovered_via": "scripts/recover_orphan_batches.py",
                }
            ],
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Clean up the stale .batch_inputs_<ds>/ directory if it exists.
        stale_dir = out_dir / f".batch_inputs_{ds}"
        if stale_dir.exists():
            for f in stale_dir.glob("*"):
                f.unlink()
            stale_dir.rmdir()

        # Clean up the streamed-output tmp file.
        if tmp_path.exists():
            tmp_path.unlink()

        print(
            f"  -> {npy_path.name}  shape={arr.shape}  tokens={total_tokens}  USD={usd:.4f}"
        )

    print("[recover] done. Re-run run_openai_embedding_kmeans — all 7 should be cache hits.")


if __name__ == "__main__":
    main()
