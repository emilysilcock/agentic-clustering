"""Adapt canonical processed datasets to Huang & He's input schema.

Upstream loads each dataset from ``./dataset/<name>/small.jsonl`` and expects
one JSON object per line with two keys::

    {"input": "<text>", "label": "<gold cluster name>"}

We materialise the same schema under ``data/huang_he/<dataset>/input.jsonl``,
adding a ``doc_id`` field so the result parser can join LLM-produced label
names back to our canonical Document records.

Text policy: per SPEC §5.1.1, all LLM-input document bodies are truncated to
512 tokens (tiktoken ``cl100k_base``). The cap is applied here so every
downstream phase --- generation, merge, classification --- sees identical
text.

Label policy: out-of-scope ("none") documents are **passed through**, not
filtered. The method has no native unassigned path; classification will
force-assign them to some in-list label, which is the penalty SPEC §5.5
anticipates for non-"none"-aware baselines. We do **not** leak the
``__none__`` gold name into the ``label`` column for is_none rows --- we
emit an empty string instead, since the upstream pipeline only reads
``label`` from the corpus during 20%-seeded label generation
(``get_label_list``), which we do not invoke under the 0%-seed
configuration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from benchmarking.data_processing.load import load_processed
from benchmarking.paths import DATA

HUANG_HE_ROOT = DATA / "huang_he"
LLM_TOKEN_CAP = 512


@dataclass(frozen=True)
class AdaptedDataset:
    name: str
    jsonl_path: Path
    n_docs: int
    n_truncated: int  # -1 when returning a cached file (count not recomputed)


def _truncate_to_token_limit(text: str, encoder, limit: int) -> tuple[str, bool]:
    tokens = encoder.encode(text)
    if len(tokens) <= limit:
        return text, False
    return encoder.decode(tokens[:limit]), True


def adapt(dataset_name: str, *, force: bool = False) -> AdaptedDataset:
    """Write ``data/huang_he/<dataset>/input.jsonl`` from our processed dataset.

    Idempotent: returns the existing file if ``force=False`` and the path
    already exists. ``is_none`` rows are passed through end-to-end --- see
    module docstring for the rationale.
    """
    import tiktoken

    out_dir = HUANG_HE_ROOT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "input.jsonl"

    if out_path.exists() and not force:
        n = sum(1 for _ in out_path.open(encoding="utf-8"))
        return AdaptedDataset(name=dataset_name, jsonl_path=out_path, n_docs=n, n_truncated=-1)

    ds = load_processed(dataset_name)
    encoder = tiktoken.get_encoding("cl100k_base")

    n_truncated = 0
    lines: list[str] = []
    for doc in ds.documents:
        text, truncated = _truncate_to_token_limit(doc["text"], encoder, LLM_TOKEN_CAP)
        if truncated:
            n_truncated += 1
        rec = {
            "doc_id": doc["doc_id"],
            "input": text,
            "label": "" if doc["is_none"] else doc["gold_label_name"],
            "gold_label_id": int(doc["gold_label_id"]),
            "is_none": bool(doc["is_none"]),
        }
        lines.append(json.dumps(rec, ensure_ascii=False))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return AdaptedDataset(
        name=dataset_name,
        jsonl_path=out_path,
        n_docs=len(lines),
        n_truncated=n_truncated,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", help="Adapt only these datasets")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    names = args.only or [
        "banking77", "clinc150", "massive_intent", "massive_domain",
        "goemotions", "twenty_newsgroups", "stackexchange",
    ]
    for name in names:
        res = adapt(name, force=args.force)
        if res.n_truncated >= 0:
            print(
                f"[huang_he/adapt] {name}: {res.n_docs} docs, "
                f"{res.n_truncated} truncated at {LLM_TOKEN_CAP} tokens -> {res.jsonl_path}"
            )
        else:
            print(f"[huang_he/adapt] {name}: cache hit ({res.n_docs} docs) -> {res.jsonl_path}")


if __name__ == "__main__":
    main()
