"""Shared schema, writer, and validator for processed benchmark datasets.

Every per-dataset module in this package exports `load() -> ProcessedDataset`.
`process_all.py` calls `validate()` and `write()` for each dataset.

Schema is documented in `paper/SPEC.md` §5.1.1 and `benchmarking/data_processing/README.md`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, TypedDict

from benchmarking.paths import DATA_DERIVED

NONE_LABEL_ID = -1
NONE_LABEL_NAME = "__none__"


class Document(TypedDict):
    doc_id: str
    text: str
    gold_label_id: int
    gold_label_name: str
    is_none: bool
    source_split: str
    source_row_index: int


@dataclass
class ProcessedDataset:
    name: str
    documents: list[Document]
    taxonomy: dict[int, str]
    meta: dict = field(default_factory=dict)


def format_doc_id(dataset_name: str, source_split: str, row_index: int) -> str:
    return f"{dataset_name}-{source_split}-{row_index:06d}"


def build_meta(
    *,
    dataset_name: str,
    hf_handle: str | None,
    hf_config: str | None,
    splits_used: list[str],
    k_in_scope: int,
    has_none_class: bool,
    documents: list[Document],
    extra: dict | None = None,
) -> dict:
    n_none = sum(1 for d in documents if d["is_none"])
    n_docs = len(documents)
    meta: dict = {
        "dataset_name": dataset_name,
        "hf_handle": hf_handle,
        "hf_config": hf_config,
        "splits_used": splits_used,
        "k_in_scope": k_in_scope,
        "has_none_class": has_none_class,
        "n_docs": n_docs,
        "n_none": n_none,
        "share_none": (n_none / n_docs) if n_docs else 0.0,
        "mean_text_chars": (sum(len(d["text"]) for d in documents) / n_docs) if n_docs else 0.0,
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        meta.update(extra)
    return meta


def validate(ds: ProcessedDataset, *, expected_k_in_scope: int, expects_none: bool) -> None:
    """Hard-fail validation. Raises AssertionError on any mismatch."""
    n = len(ds.documents)
    assert n > 0, f"{ds.name}: zero documents"

    seen_ids: set[str] = set()
    for d in ds.documents:
        assert d["text"] and d["text"].strip(), f"{ds.name}: empty text at doc_id={d['doc_id']}"
        assert d["doc_id"] not in seen_ids, f"{ds.name}: duplicate doc_id={d['doc_id']}"
        seen_ids.add(d["doc_id"])
        if d["is_none"]:
            assert d["gold_label_id"] == NONE_LABEL_ID, (
                f"{ds.name}: is_none=True but gold_label_id={d['gold_label_id']} (expected {NONE_LABEL_ID})"
            )
            assert d["gold_label_name"] == NONE_LABEL_NAME, (
                f"{ds.name}: is_none=True but gold_label_name={d['gold_label_name']!r}"
            )
        else:
            assert 0 <= d["gold_label_id"] < expected_k_in_scope, (
                f"{ds.name}: gold_label_id={d['gold_label_id']} out of range "
                f"[0, {expected_k_in_scope}) at doc_id={d['doc_id']}"
            )

    in_scope_taxonomy = {k: v for k, v in ds.taxonomy.items() if k != NONE_LABEL_ID}
    assert len(in_scope_taxonomy) == expected_k_in_scope, (
        f"{ds.name}: taxonomy has {len(in_scope_taxonomy)} in-scope labels, expected {expected_k_in_scope}"
    )
    assert set(in_scope_taxonomy.keys()) == set(range(expected_k_in_scope)), (
        f"{ds.name}: taxonomy keys are not 0..{expected_k_in_scope - 1}: "
        f"{sorted(in_scope_taxonomy.keys())[:5]}..."
    )

    n_none = sum(1 for d in ds.documents if d["is_none"])
    if expects_none:
        assert n_none > 0, f"{ds.name}: expected a 'none' class but found 0 such documents"
        assert NONE_LABEL_ID in ds.taxonomy, f"{ds.name}: expects_none but taxonomy lacks -1"
        assert ds.taxonomy[NONE_LABEL_ID] == NONE_LABEL_NAME
    else:
        assert n_none == 0, f"{ds.name}: expects_none=False but found {n_none} 'none' docs"
        assert NONE_LABEL_ID not in ds.taxonomy, (
            f"{ds.name}: expects_none=False but taxonomy contains -1"
        )


def write(ds: ProcessedDataset, out_root: Path = DATA_DERIVED) -> Path:
    """Write documents.jsonl, taxonomy.json, meta.json. Returns the output directory."""
    out_dir = out_root / ds.name
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "documents.jsonl").write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in ds.documents) + "\n",
        encoding="utf-8",
    )
    (out_dir / "taxonomy.json").write_text(
        json.dumps({str(k): v for k, v in sorted(ds.taxonomy.items())}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "meta.json").write_text(
        json.dumps(ds.meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_dir


def documents_from_rows(
    rows: Iterable[tuple[str, int, str]],
    *,
    dataset_name: str,
    source_split: str,
    none_label_id_in_source: int | None = None,
) -> list[Document]:
    """Build Document records from (text, label_id, label_name) tuples.

    If `none_label_id_in_source` is given, rows with that source-label-id are mapped
    to is_none=True / gold_label_id=-1 / gold_label_name='__none__'.
    """
    docs: list[Document] = []
    for i, (text, source_label_id, source_label_name) in enumerate(rows):
        is_none = none_label_id_in_source is not None and source_label_id == none_label_id_in_source
        docs.append(
            Document(
                doc_id=format_doc_id(dataset_name, source_split, i),
                text=text,
                gold_label_id=NONE_LABEL_ID if is_none else source_label_id,
                gold_label_name=NONE_LABEL_NAME if is_none else source_label_name,
                is_none=is_none,
                source_split=source_split,
                source_row_index=i,
            )
        )
    return docs


def remap_in_scope_label_ids(documents: list[Document]) -> tuple[list[Document], dict[int, str]]:
    """Compact in-scope label ids to a contiguous 0..k-1 range, preserving '-1' for none.

    Returns (new_documents, new_taxonomy). Label name -> new id is assigned in the order
    in-scope label ids first appear in `documents`.

    Use this when the source dataset has a label set that is not already 0..k-1
    (e.g. after filtering, or when source ids are sparse).
    """
    name_to_new_id: dict[str, int] = {}
    new_docs: list[Document] = []
    for d in documents:
        if d["is_none"]:
            new_docs.append(d)
            continue
        name = d["gold_label_name"]
        if name not in name_to_new_id:
            name_to_new_id[name] = len(name_to_new_id)
        new_docs.append({**d, "gold_label_id": name_to_new_id[name]})

    new_taxonomy: dict[int, str] = {v: k for k, v in name_to_new_id.items()}
    if any(d["is_none"] for d in documents):
        new_taxonomy[NONE_LABEL_ID] = NONE_LABEL_NAME
    return new_docs, new_taxonomy
