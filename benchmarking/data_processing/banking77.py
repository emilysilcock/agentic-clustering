"""Banking77 — fine-grained intent classification, 77 classes."""

from __future__ import annotations

from datasets import load_dataset

from benchmarking.data_processing.base import (
    Document,
    ProcessedDataset,
    build_meta,
    format_doc_id,
)

NAME = "banking77"
HF_HANDLE = "PolyAI/banking77"
SPLIT = "test"
K = 77


def load() -> ProcessedDataset:
    # PolyAI/banking77 still ships as a legacy dataset script, which datasets>=3.0
    # refuses to execute. Pin to the auto-generated parquet conversion branch.
    ds = load_dataset(HF_HANDLE, split=SPLIT, revision="refs/convert/parquet")
    label_names: list[str] = ds.features["label"].names

    documents: list[Document] = []
    for i, row in enumerate(ds):
        label_id = int(row["label"])
        documents.append(
            Document(
                doc_id=format_doc_id(NAME, SPLIT, i),
                text=row["text"],
                gold_label_id=label_id,
                gold_label_name=label_names[label_id],
                is_none=False,
                source_split=SPLIT,
                source_row_index=i,
            )
        )

    taxonomy = {i: name for i, name in enumerate(label_names)}
    meta = build_meta(
        dataset_name=NAME,
        hf_handle=HF_HANDLE,
        hf_config=None,
        splits_used=[SPLIT],
        k_in_scope=K,
        has_none_class=False,
        documents=documents,
    )
    return ProcessedDataset(name=NAME, documents=documents, taxonomy=taxonomy, meta=meta)


if __name__ == "__main__":
    from benchmarking.data_processing.base import validate, write

    ds = load()
    validate(ds, expected_k_in_scope=K, expects_none=False)
    out = write(ds)
    print(f"{NAME}: n_docs={ds.meta['n_docs']} k={ds.meta['k_in_scope']} -> {out}")
