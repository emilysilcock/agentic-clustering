"""Run every dataset loader, validate, and write to data/derived/<dataset>/.

Usage:
    uv run --native-tls python -m benchmarking.data_processing.process_all
    uv run --native-tls python -m benchmarking.data_processing.process_all --only banking77 clinc150
"""

from __future__ import annotations

import argparse
import time
from typing import Callable

from benchmarking.data_processing import (
    banking77,
    clinc150,
    goemotions,
    massive_domain,
    massive_intent,
    stackexchange,
    twenty_newsgroups,
)
from benchmarking.data_processing.base import ProcessedDataset, validate, write
from benchmarking.paths import ensure_data_dirs

Loader = Callable[[], ProcessedDataset]


REGISTRY: dict[str, tuple[Loader, int, bool]] = {
    # name: (load_fn, expected_k_in_scope, expects_none)
    "banking77": (banking77.load, banking77.K, False),
    "clinc150": (clinc150.load, clinc150.K, True),
    "massive_intent": (massive_intent.load, massive_intent.K, False),
    "massive_domain": (massive_domain.load, massive_domain.K, False),
    "goemotions": (goemotions.load, goemotions.K, True),
    "twenty_newsgroups": (twenty_newsgroups.load, twenty_newsgroups.K, False),
    "stackexchange": (stackexchange.load, stackexchange.K, False),
}


def process(name: str) -> ProcessedDataset:
    load_fn, expected_k, expects_none = REGISTRY[name]
    t0 = time.perf_counter()
    print(f"[{name}] loading…")
    ds = load_fn()

    validate(ds, expected_k_in_scope=expected_k, expects_none=expects_none)

    if name == "massive_domain":
        massive_domain.verify_alignment_with_massive_intent(ds)

    out = write(ds)
    elapsed = time.perf_counter() - t0
    print(
        f"[{name}] n_docs={ds.meta['n_docs']:>6}  k={ds.meta['k_in_scope']:>3}  "
        f"n_none={ds.meta['n_none']:>5}  share_none={ds.meta['share_none']:.3f}  "
        f"mean_text_chars={ds.meta['mean_text_chars']:.1f}  "
        f"elapsed={elapsed:.1f}s  -> {out}"
    )
    return ds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(REGISTRY.keys()),
        help="Process only the named datasets (default: all).",
    )
    args = parser.parse_args()

    ensure_data_dirs()
    names = args.only if args.only else list(REGISTRY.keys())
    for name in names:
        process(name)


if __name__ == "__main__":
    main()
