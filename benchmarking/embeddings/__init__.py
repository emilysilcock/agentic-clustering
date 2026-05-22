"""Shared types and helpers for cached dataset embeddings.

`EmbeddingCache` is the return type of every dataset embedder under this
package (currently `sbert.py` and `openai_embeddings.py`). Each embedder
owns its own cache directory under `data/embeddings/<model_shortname>/`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EmbeddingCache:
    embeddings: np.ndarray  # (n_docs, dim) float32, unnormalized
    model: str               # the model_name argument passed in
    short: str               # shortname used in the cache path
    npy_path: Path
    meta_path: Path
    n_docs: int
    dim: int
    cache_hit: bool          # True if loaded from disk, False if freshly computed


def model_shortname(model_name: str) -> str:
    """Return just the model name without any HF org prefix."""
    return model_name.rsplit("/", 1)[-1]


def texts_sha256(texts: list[str]) -> str:
    """Stable content hash for the document list, used as cache invalidation key."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()
