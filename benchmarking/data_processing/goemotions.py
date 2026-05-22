"""GoEmotions — emotion classification (27 emotions + 'neutral' as the 'none' class).

Config: `simplified` (single-label rows only — multi-label rows have already been split
upstream). All three splits are concatenated, matching ClusterLLM's recipe; we depart
from that recipe by *keeping* neutral and treating it as the gold 'none' class.
"""

from __future__ import annotations

from datasets import concatenate_datasets, load_dataset

from benchmarking.data_processing.base import (
    NONE_LABEL_ID,
    NONE_LABEL_NAME,
    Document,
    ProcessedDataset,
    build_meta,
    format_doc_id,
    remap_in_scope_label_ids,
)

NAME = "goemotions"
HF_HANDLE = "google-research-datasets/go_emotions"
HF_CONFIG = "simplified"
SPLITS = ("train", "validation", "test")
K = 27
NEUTRAL_LABEL_NAME = "neutral"


def load() -> ProcessedDataset:
    per_split = [load_dataset(HF_HANDLE, HF_CONFIG, split=s) for s in SPLITS]
    ds = concatenate_datasets(per_split)
    source_label_names: list[str] = ds.features["labels"].feature.names

    if NEUTRAL_LABEL_NAME not in source_label_names:
        raise RuntimeError(
            f"{NAME}: expected a '{NEUTRAL_LABEL_NAME}' class in source labels, got {source_label_names}"
        )
    neutral_label_id = source_label_names.index(NEUTRAL_LABEL_NAME)

    raw_documents: list[Document] = []
    skipped_multilabel = 0
    for i, row in enumerate(ds):
        labels = row["labels"]
        if len(labels) != 1:
            skipped_multilabel += 1
            continue
        source_label_id = int(labels[0])
        source_label_name = source_label_names[source_label_id]
        is_none = source_label_id == neutral_label_id
        raw_documents.append(
            Document(
                doc_id=format_doc_id(NAME, "all", i),
                text=row["text"],
                gold_label_id=NONE_LABEL_ID if is_none else source_label_id,
                gold_label_name=NONE_LABEL_NAME if is_none else source_label_name,
                is_none=is_none,
                source_split="all",
                source_row_index=i,
            )
        )

    documents, taxonomy = remap_in_scope_label_ids(raw_documents)

    meta = build_meta(
        dataset_name=NAME,
        hf_handle=HF_HANDLE,
        hf_config=HF_CONFIG,
        splits_used=list(SPLITS),
        k_in_scope=K,
        has_none_class=True,
        documents=documents,
        extra={"skipped_multilabel_rows": skipped_multilabel},
    )
    return ProcessedDataset(name=NAME, documents=documents, taxonomy=taxonomy, meta=meta)


if __name__ == "__main__":
    from benchmarking.data_processing.base import validate, write

    ds = load()
    validate(ds, expected_k_in_scope=K, expects_none=True)
    out = write(ds)
    print(
        f"{NAME}: n_docs={ds.meta['n_docs']} k={ds.meta['k_in_scope']} "
        f"n_none={ds.meta['n_none']} skipped_multilabel={ds.meta['skipped_multilabel_rows']} "
        f"-> {out}"
    )
