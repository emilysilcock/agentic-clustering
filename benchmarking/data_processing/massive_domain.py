"""MASSIVE-Domain — same EN test corpus as MASSIVE-Intent, projected onto the scenario
(18-class) label space."""

from __future__ import annotations

from datasets import load_dataset

from benchmarking.data_processing import massive_intent
from benchmarking.data_processing.base import (
    Document,
    ProcessedDataset,
    build_meta,
    format_doc_id,
)

NAME = "massive_domain"
HF_HANDLE = "mteb/amazon_massive_scenario"
SPLIT = "test"
K = 18
LANG_CONFIG = "en"
EN_LOCALE = "en-US"


def load() -> ProcessedDataset:
    test_ds = _load_en_split(SPLIT)
    text_col = _pick_text_column(test_ds.column_names)
    label_col = _pick_label_column(test_ds.column_names)

    # Derive canonical label space from train+val+test to keep k=18 stable even if
    # test doesn't cover all scenarios (mirrors massive_intent's approach).
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
    for c in ("label", "label_text", "scenario"):
        if c in cols:
            return c
    raise RuntimeError(f"{NAME}: no label column among {cols}")


def verify_alignment_with_massive_intent(domain_ds: ProcessedDataset) -> None:
    """Cross-check that MASSIVE-Domain rows correspond to the same texts as MASSIVE-Intent.

    Both datasets derive from the same MASSIVE EN test corpus, so row i in each should
    contain the same utterance.
    """
    intent_ds = massive_intent.load()
    n_intent = len(intent_ds.documents)
    n_domain = len(domain_ds.documents)
    assert n_intent == n_domain, (
        f"MASSIVE corpus alignment: massive_intent has {n_intent} docs, "
        f"massive_domain has {n_domain}. Both should be the EN test split."
    )
    for i, (a, b) in enumerate(zip(intent_ds.documents, domain_ds.documents)):
        if a["text"] != b["text"]:
            raise AssertionError(
                f"MASSIVE alignment broken at row {i}: "
                f"intent={a['text']!r} vs domain={b['text']!r}"
            )


if __name__ == "__main__":
    from benchmarking.data_processing.base import validate, write

    ds = load()
    validate(ds, expected_k_in_scope=K, expects_none=False)
    out = write(ds)
    print(f"{NAME}: n_docs={ds.meta['n_docs']} k={ds.meta['k_in_scope']} -> {out}")
