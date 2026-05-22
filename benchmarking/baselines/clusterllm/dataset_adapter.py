"""Adapt our canonical processed datasets to ClusterLLM's expected layout.

ClusterLLM's vendored code (``triplet_sampling.py``, ``get_embedding.py``,
``finetune.py``, ``convert_triplet.py``) all expect a JSONL with one
``{"input": str, "label": int|str}`` record per line, located on disk at a
path the caller provides via ``--data_path``.

This adapter reads the canonical dataset (the same one every other method
in ``benchmarking/`` consumes, via ``load_processed``) and writes the
ClusterLLM-shaped JSONL into ``data/clusterllm/<dataset>/large.jsonl``.

Text policy: per SPEC §5.1.1, all LLM-input document bodies are truncated
to 512 tokens (tiktoken ``cl100k_base``). The cap is applied here so that
every downstream phase — Instructor encoding, triplet sampling, Claude
judging — sees identical text. Bytes that would have been seen only by a
non-LLM phase are not preserved separately; the SPEC's intent is that the
512-token-capped text is the unit of analysis for this baseline.

Label policy: we pass the integer ``gold_label_id`` through. The author
code only uses the label for diagnostic/oracle "would the gold disagree"
analyses (``output`` field in ``triplets.json``) and doesn't feed it to the
LLM. Out-of-scope (none) documents are excluded — they have no clustering
target.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from benchmarking.data_processing.load import load_processed
from benchmarking.paths import DATA

CLUSTERLLM_ROOT = DATA / "clusterllm"
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
    """Write ``data/clusterllm/<dataset>/large.jsonl`` from our processed dataset.

    Idempotent: returns the existing file if ``force=False`` and the path
    already exists. None-class documents are filtered out (per SPEC §5.5,
    clustering targets exclude the explicit None class).
    """
    import tiktoken

    out_dir = CLUSTERLLM_ROOT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "large.jsonl"

    if out_path.exists() and not force:
        n = sum(1 for _ in out_path.open(encoding="utf-8"))
        return AdaptedDataset(name=dataset_name, jsonl_path=out_path, n_docs=n, n_truncated=-1)

    ds = load_processed(dataset_name)
    encoder = tiktoken.get_encoding("cl100k_base")

    n_truncated = 0
    lines: list[str] = []
    for doc in ds.documents:
        if doc["is_none"]:
            continue
        text, truncated = _truncate_to_token_limit(doc["text"], encoder, LLM_TOKEN_CAP)
        if truncated:
            n_truncated += 1
        rec = {"input": text, "label": int(doc["gold_label_id"])}
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
                f"[clusterllm/adapt] {name}: {res.n_docs} docs, "
                f"{res.n_truncated} truncated at {LLM_TOKEN_CAP} tokens -> {res.jsonl_path}"
            )
        else:
            print(f"[clusterllm/adapt] {name}: cache hit ({res.n_docs} docs) -> {res.jsonl_path}")


if __name__ == "__main__":
    main()
