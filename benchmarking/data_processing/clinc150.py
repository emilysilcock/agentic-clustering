"""CLINC150 — intent classification with native out-of-scope (OOS) examples.

Config: `plus` (matches ClusterLLM). Split: test (5,500 rows = 4,500 in-scope + 1,000 OOS).
The OOS class is kept and remapped to the "none" gold class (-1 / '__none__').
"""

from __future__ import annotations

from datasets import load_dataset

from benchmarking.data_processing.base import (
    NONE_LABEL_ID,
    NONE_LABEL_NAME,
    Document,
    ProcessedDataset,
    build_meta,
    format_doc_id,
    remap_in_scope_label_ids,
)

NAME = "clinc150"
HF_HANDLE = "clinc/clinc_oos"
HF_CONFIG = "plus"
SPLIT = "test"
K = 150
OOS_LABEL_NAME_IN_SOURCE = "oos"


def load() -> ProcessedDataset:
    # clinc/clinc_oos ships as a legacy dataset script and its parquet conversion
    # doesn't preserve config names; instead, configs live as subdirectories on the
    # `refs/convert/parquet` branch. Load the parquet file for our config + split directly.
    parquet_url = (
        f"hf://datasets/{HF_HANDLE}@refs/convert/parquet/{HF_CONFIG}/{SPLIT}/0000.parquet"
    )
    ds = load_dataset("parquet", data_files=parquet_url, split="train")
    source_label_names: list[str] = ds.features["intent"].names

    if OOS_LABEL_NAME_IN_SOURCE not in source_label_names:
        raise RuntimeError(
            f"{NAME}: expected an '{OOS_LABEL_NAME_IN_SOURCE}' class in the source label set, "
            f"got {source_label_names}"
        )
    oos_label_id_in_source = source_label_names.index(OOS_LABEL_NAME_IN_SOURCE)

    raw_documents: list[Document] = []
    for i, row in enumerate(ds):
        source_label_id = int(row["intent"])
        source_label_name = source_label_names[source_label_id]
        is_none = source_label_id == oos_label_id_in_source
        raw_documents.append(
            Document(
                doc_id=format_doc_id(NAME, SPLIT, i),
                text=row["text"],
                gold_label_id=NONE_LABEL_ID if is_none else source_label_id,
                gold_label_name=NONE_LABEL_NAME if is_none else source_label_name,
                is_none=is_none,
                source_split=SPLIT,
                source_row_index=i,
            )
        )

    # The source label space has 151 ids (150 in-scope + oos). Compact in-scope ids to 0..149.
    documents, taxonomy = remap_in_scope_label_ids(raw_documents)

    meta = build_meta(
        dataset_name=NAME,
        hf_handle=HF_HANDLE,
        hf_config=HF_CONFIG,
        splits_used=[SPLIT],
        k_in_scope=K,
        has_none_class=True,
        documents=documents,
        extra={"oos_label_name_in_source": OOS_LABEL_NAME_IN_SOURCE},
    )
    return ProcessedDataset(name=NAME, documents=documents, taxonomy=taxonomy, meta=meta)


if __name__ == "__main__":
    from benchmarking.data_processing.base import validate, write

    ds = load()
    validate(ds, expected_k_in_scope=K, expects_none=True)
    out = write(ds)
    print(
        f"{NAME}: n_docs={ds.meta['n_docs']} k={ds.meta['k_in_scope']} "
        f"n_none={ds.meta['n_none']} -> {out}"
    )
