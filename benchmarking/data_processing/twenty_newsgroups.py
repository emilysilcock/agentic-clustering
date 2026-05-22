"""20 Newsgroups — long-doc topic classification, 20 classes.

Uses sklearn's `fetch_20newsgroups` with `subset='all'` and
`remove=('headers','footers','quotes')` — the BERTopic preprocessing recipe, which
matches SPEC §5.1's stripping requirement. Rows with empty text after stripping are dropped.
"""

from __future__ import annotations

import sklearn
from sklearn.datasets import fetch_20newsgroups

from benchmarking.data_processing.base import (
    Document,
    ProcessedDataset,
    build_meta,
    format_doc_id,
)

NAME = "twenty_newsgroups"
HF_HANDLE = None  # sklearn-sourced
SPLIT = "all"
K = 20
REMOVE = ("headers", "footers", "quotes")


def load() -> ProcessedDataset:
    raw = fetch_20newsgroups(subset=SPLIT, remove=REMOVE, shuffle=False)
    target_names: list[str] = list(raw.target_names)

    documents: list[Document] = []
    dropped_empty = 0
    for i, (text, label_id) in enumerate(zip(raw.data, raw.target)):
        stripped = text.strip()
        if not stripped:
            dropped_empty += 1
            continue
        documents.append(
            Document(
                doc_id=format_doc_id(NAME, SPLIT, i),
                text=stripped,
                gold_label_id=int(label_id),
                gold_label_name=target_names[int(label_id)],
                is_none=False,
                source_split=SPLIT,
                source_row_index=i,
            )
        )

    taxonomy = {i: name for i, name in enumerate(target_names)}
    meta = build_meta(
        dataset_name=NAME,
        hf_handle=HF_HANDLE,
        hf_config=None,
        splits_used=[SPLIT],
        k_in_scope=K,
        has_none_class=False,
        documents=documents,
        extra={
            "remove": list(REMOVE),
            "sklearn_version": sklearn.__version__,
            "dropped_empty_rows": dropped_empty,
        },
    )
    return ProcessedDataset(name=NAME, documents=documents, taxonomy=taxonomy, meta=meta)


if __name__ == "__main__":
    from benchmarking.data_processing.base import validate, write

    ds = load()
    validate(ds, expected_k_in_scope=K, expects_none=False)
    out = write(ds)
    print(
        f"{NAME}: n_docs={ds.meta['n_docs']} k={ds.meta['k_in_scope']} "
        f"dropped_empty={ds.meta['dropped_empty_rows']} -> {out}"
    )
