"""MASSIVE-Intent — fine-grained intent classification, EN test split (60 classes)."""

from __future__ import annotations

from datasets import load_dataset

from benchmarking.data_processing.base import (
    Document,
    ProcessedDataset,
    build_meta,
    format_doc_id,
)

NAME = "massive_intent"
HF_HANDLE = "mteb/amazon_massive_intent"
SPLIT = "test"
K = 60
LANG_CONFIG = "en"
EN_LOCALE = "en-US"


def load() -> ProcessedDataset:
    test_ds = _load_en_split(SPLIT)
    text_col = _pick_text_column(test_ds.column_names)
    label_col = _pick_label_column(test_ds.column_names)

    # The full intent label space has 60 classes (per MASSIVE paper). EN test only
    # contains 59 of them — 'cooking_query' is absent. To keep k=60 (matching SPEC and
    # prior clustering papers), derive the canonical label set from train+val+test
    # and emit only test rows.
    label_names: list[str] = sorted(
        {
            *(str(x) for x in test_ds[label_col]),
            *(str(x) for x in _load_en_split("train")[label_col]),
            *(str(x) for x in _load_en_split("validation")[label_col]),
        }
    )
    label_name_to_id = {name: i for i, name in enumerate(label_names)}

    documents: list[Document] = []
    for i, row in enumerate(test_ds):
        label_name = str(row[label_col])
        documents.append(
            Document(
                doc_id=format_doc_id(NAME, SPLIT, i),
                text=row[text_col],
                gold_label_id=label_name_to_id[label_name],
                gold_label_name=label_name,
                is_none=False,
                source_split=SPLIT,
                source_row_index=i,
            )
        )

    taxonomy = {v: k for k, v in label_name_to_id.items()}
    meta = build_meta(
        dataset_name=NAME,
        hf_handle=HF_HANDLE,
        hf_config=LANG_CONFIG,
        splits_used=[SPLIT],
        k_in_scope=K,
        has_none_class=False,
        documents=documents,
    )
    return ProcessedDataset(name=NAME, documents=documents, taxonomy=taxonomy, meta=meta)


def _load_en_split(split: str):
    try:
        return load_dataset(HF_HANDLE, LANG_CONFIG, split=split)
    except (ValueError, KeyError):
        ds = load_dataset(HF_HANDLE, split=split)
        if "locale" in ds.column_names:
            ds = ds.filter(lambda r: r["locale"] == EN_LOCALE)
        return ds


def _pick_text_column(cols: list[str]) -> str:
    for c in ("text", "utt", "utterance"):
        if c in cols:
            return c
    raise RuntimeError(f"{NAME}: no text column among {cols}")


def _pick_label_column(cols: list[str]) -> str:
    # The MTEB MASSIVE datasets store the gold label as a *string* under 'label'
    # (e.g. 'alarm_set'). Prefer that; fall back to other names defensively.
    for c in ("label", "label_text", "intent"):
        if c in cols:
            return c
    raise RuntimeError(f"{NAME}: no label column among {cols}")


if __name__ == "__main__":
    from benchmarking.data_processing.base import validate, write

    ds = load()
    validate(ds, expected_k_in_scope=K, expects_none=False)
    out = write(ds)
    print(f"{NAME}: n_docs={ds.meta['n_docs']} k={ds.meta['k_in_scope']} -> {out}")
