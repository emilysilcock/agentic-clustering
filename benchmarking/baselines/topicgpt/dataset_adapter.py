"""Adapt canonical processed datasets to TopicGPT's expected JSONL format.

TopicGPT's vendored code reads input via ``pandas.read_json(data, lines=True)``
and indexes on ``df["text"]`` (`generation_1.py` and `assignment.py`). Later
phases tack on ``responses`` and ``prompted_docs`` columns to the same
dataframe and write it back. Any other columns we include get passed through.

Text policy: per SPEC §5.1.1, all LLM-input document bodies are truncated to
512 tokens (tiktoken ``cl100k_base``). The cap is applied here so every
downstream phase --- generation, refinement, assignment, correction --- sees
identical text.

Label policy: out-of-scope ("none") documents are PASSED THROUGH, not
filtered. TopicGPT has no native unassigned path (verified upstream —
see SPEC §5.6.2 pre-run verifications); we feed it the full corpus
including is_none docs because at deployment time the method wouldn't
know which docs are out-of-scope. Filtering here would constitute a
privileged-information leak — the same kind we flag for the Huang & He
20%-seed config in SPEC §5.6.3. The method then assigns its own choice
of topic to each is_none doc, and ``result_parser.py`` measures the
predictions against the gold labels (including ``__none__``) without
special handling — this is exactly the penalty SPEC §5.5.3 anticipates
for non-"none"-aware methods on CLINC OOS / GoEmotions neutral.

We pass our canonical ``doc_id`` and ``gold_label_id`` through as extra
columns so the result parser can map LLM-produced topic names back to a
contiguous cluster_id under ``results/predictions/topicgpt/<dataset>/`` per
SPEC §5.11.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from benchmarking.data_processing.load import load_processed
from benchmarking.paths import DATA

TOPICGPT_ROOT = DATA / "topicgpt"
LLM_TOKEN_CAP = 512


@dataclass(frozen=True)
class AdaptedDataset:
    name: str
    jsonl_path: Path
    n_docs: int
    n_truncated: int


def _truncate_to_token_limit(text: str, encoder, limit: int) -> tuple[str, bool]:
    tokens = encoder.encode(text)
    if len(tokens) <= limit:
        return text, False
    return encoder.decode(tokens[:limit]), True


def adapt(dataset_name: str, *, force: bool = False) -> AdaptedDataset:
    """Write ``data/topicgpt/<dataset>/input.jsonl`` from our processed dataset.

    Idempotent: returns the existing file if ``force=False`` and the path
    already exists. **None-class documents are passed through, not filtered**
    --- see module docstring for the privileged-info rationale.
    """
    import tiktoken

    out_dir = TOPICGPT_ROOT / dataset_name
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
            "id": doc["doc_id"],
            "text": text,
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
                f"[topicgpt/adapt] {name}: {res.n_docs} docs, "
                f"{res.n_truncated} truncated at {LLM_TOKEN_CAP} tokens -> {res.jsonl_path}"
            )
        else:
            print(f"[topicgpt/adapt] {name}: cache hit ({res.n_docs} docs) -> {res.jsonl_path}")


if __name__ == "__main__":
    main()
