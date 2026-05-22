"""StackExchange-clustering — ClusterLLM-small recipe.

Uses the `data/stackexchange/small.jsonl` file from the ClusterLLM data zip
(`https://drive.google.com/file/d/1TBq3vkfm3OZLi90GVH-PVNKi3fk1Vba7/view`):
4,156 forum-title docs across 121 stackexchange-site labels. Selected over MTEB's
`stackexchange-clustering` (25-subset list-of-clusterings format) because (a) it gives
a single flat partition compatible with our ARI/NMI/ACC reporting; (b) it is the de
facto small-StackExchange benchmark for the LLM-clustering line of work; (c) it sits
inside the size range of our other datasets.

Label names in the source are filenames like `english.stackexchange.com.txt`; we strip
the `.txt` suffix for the taxonomy.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import gdown

from benchmarking.data_processing.base import (
    Document,
    ProcessedDataset,
    build_meta,
    format_doc_id,
)
from benchmarking.paths import DATA_RAW

NAME = "stackexchange"
K = 121  # 121 stackexchange sites in ClusterLLM-small
GDRIVE_FILE_ID = "1TBq3vkfm3OZLi90GVH-PVNKi3fk1Vba7"
ZIP_PATH = DATA_RAW / "clusterllm" / "clusterllm-data.zip"
EXTRACT_DIR = DATA_RAW / "clusterllm"
JSONL_REL_PATH = Path("datasets") / "stackexchange" / "small.jsonl"
LABEL_SUFFIX = ".txt"


def load() -> ProcessedDataset:
    jsonl_path = _ensure_local()

    label_name_to_id: dict[str, int] = {}
    documents: list[Document] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            text = row["input"]
            raw_label = row["label"]
            label_name = (
                raw_label[: -len(LABEL_SUFFIX)] if raw_label.endswith(LABEL_SUFFIX) else raw_label
            )
            if label_name not in label_name_to_id:
                label_name_to_id[label_name] = len(label_name_to_id)
            documents.append(
                Document(
                    doc_id=format_doc_id(NAME, "small", i),
                    text=text,
                    gold_label_id=label_name_to_id[label_name],
                    gold_label_name=label_name,
                    is_none=False,
                    source_split="small",
                    source_row_index=i,
                )
            )

    taxonomy = {v: k for k, v in label_name_to_id.items()}
    meta = build_meta(
        dataset_name=NAME,
        hf_handle=None,
        hf_config=None,
        splits_used=["clusterllm-small"],
        k_in_scope=len(taxonomy),
        has_none_class=False,
        documents=documents,
        extra={
            "source": "ClusterLLM data zip",
            "source_url": f"https://drive.google.com/file/d/{GDRIVE_FILE_ID}/view",
            "source_relpath": str(JSONL_REL_PATH).replace("\\", "/"),
        },
    )
    return ProcessedDataset(name=NAME, documents=documents, taxonomy=taxonomy, meta=meta)


def _ensure_local() -> Path:
    target = EXTRACT_DIR / JSONL_REL_PATH
    if target.exists():
        return target

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        print(f"[{NAME}] downloading ClusterLLM data zip…")
        gdown.download(id=GDRIVE_FILE_ID, output=str(ZIP_PATH), quiet=False)

    print(f"[{NAME}] extracting {ZIP_PATH.name} → {EXTRACT_DIR}")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(EXTRACT_DIR)

    if not target.exists():
        raise FileNotFoundError(
            f"{NAME}: expected {target} after extracting {ZIP_PATH}, but it is missing"
        )
    return target


if __name__ == "__main__":
    from benchmarking.data_processing.base import validate, write

    ds = load()
    validate(ds, expected_k_in_scope=ds.meta["k_in_scope"], expects_none=False)
    out = write(ds)
    print(
        f"{NAME}: n_docs={ds.meta['n_docs']} k={ds.meta['k_in_scope']} -> {out}"
    )
