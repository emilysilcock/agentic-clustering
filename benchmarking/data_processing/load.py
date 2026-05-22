"""Read a processed dataset from data/derived/<name>/.

Every method in benchmarking/experiments/ goes through this. No method should
call `datasets.load_dataset` itself — that lives only in the per-dataset
loaders in this package.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from benchmarking.data_processing.base import Document
from benchmarking.paths import DATA_DERIVED


@dataclass(frozen=True)
class LoadedDataset:
    name: str
    documents: list[Document]
    taxonomy: dict[int, str]
    meta: dict


def load_processed(name: str) -> LoadedDataset:
    """Read documents.jsonl + taxonomy.json + meta.json from data/derived/<name>/."""
    base = DATA_DERIVED / name
    docs_path = base / "documents.jsonl"
    tax_path = base / "taxonomy.json"
    meta_path = base / "meta.json"

    if not docs_path.exists():
        raise FileNotFoundError(
            f"{docs_path} not found — run "
            f"`uv run --native-tls python -m benchmarking.data_processing.process_all --only {name}` first."
        )

    documents: list[Document] = [
        json.loads(line) for line in docs_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    raw_taxonomy = json.loads(tax_path.read_text(encoding="utf-8"))
    taxonomy: dict[int, str] = {int(k): v for k, v in raw_taxonomy.items()}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    return LoadedDataset(name=name, documents=documents, taxonomy=taxonomy, meta=meta)
