"""Cached SBERT (sentence-transformers) embeddings for a processed dataset.

One file per (model, dataset) at:
    data/embeddings/<model_shortname>/<dataset>.npy            # (n_docs, dim) float32, raw (NOT L2-normalized)
    data/embeddings/<model_shortname>/<dataset>.meta.json      # provenance + sha256 of texts

Embeddings are stored *unnormalized* so downstream methods can choose their
own normalisation (k-means → L2-normalize; UMAP/BERTopic → raw).

Consumers (current + planned): SBERT+kmeans (now), BERTopic (next), our
method's investigator agent (later) — multiple methods will read the same
cache, so we factor this out from the start.
"""

from __future__ import annotations

import json

import numpy as np
import sentence_transformers

from benchmarking.data_processing.load import load_processed
from benchmarking.embeddings import EmbeddingCache, model_shortname, texts_sha256
from benchmarking.paths import DATA

EMBEDDINGS_ROOT = DATA / "embeddings"

# Module-level singleton: avoids reloading the model once per dataset in a multi-dataset runner.
_MODEL_CACHE: dict[str, "sentence_transformers.SentenceTransformer"] = {}


def _get_model(model_name: str) -> "sentence_transformers.SentenceTransformer":
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def embed_dataset(
    dataset_name: str,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    *,
    batch_size: int = 64,
    show_progress: bool = True,
) -> EmbeddingCache:
    """Return cached SBERT embeddings for a processed dataset, computing them once if needed.

    The cache is keyed on (model shortname, dataset name). Cache invalidation: the sidecar
    stores a sha256 of the documents' texts; on mismatch we recompute (catches the
    'I edited a loader' footgun).
    """
    ds = load_processed(dataset_name)
    texts = [d["text"] for d in ds.documents]
    short = model_shortname(model_name)

    out_dir = EMBEDDINGS_ROOT / short
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / f"{dataset_name}.npy"
    meta_path = out_dir / f"{dataset_name}.meta.json"

    expected_sha = texts_sha256(texts)

    if npy_path.exists() and meta_path.exists():
        sidecar = json.loads(meta_path.read_text(encoding="utf-8"))
        if sidecar.get("texts_sha256") == expected_sha and sidecar.get("n_docs") == len(texts):
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
                dim=arr.shape[1],
                cache_hit=True,
            )

    model = _get_model(model_name)
    arr = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32, copy=False)

    np.save(npy_path, arr)
    meta_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "model_shortname": short,
                "sentence_transformers_version": sentence_transformers.__version__,
                "dataset": dataset_name,
                "n_docs": len(texts),
                "dim": int(arr.shape[1]),
                "dtype": str(arr.dtype),
                "normalized": False,
                "texts_sha256": expected_sha,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return EmbeddingCache(
        embeddings=arr,
        model=model_name,
        short=short,
        npy_path=npy_path,
        meta_path=meta_path,
        n_docs=arr.shape[0],
        dim=int(arr.shape[1]),
        cache_hit=False,
    )
