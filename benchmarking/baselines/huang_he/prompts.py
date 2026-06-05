"""Re-export the three vendored prompt builders verbatim.

Going through this module — rather than importing the vendored helpers
directly at every call site — keeps the surface area of the
``huang_he/_vendored/`` dependency to a single line and makes it obvious to
a reader that we are using upstream's prompts byte-identically.

The upstream typo ``"classicifation"`` in ``prompt_construct_generate_label``
is preserved deliberately. See ``CHANGES.md``.

Helper signatures (from ``_vendored/label_generation.py`` and
``_vendored/given_label_classification.py``):

* ``prompt_construct_generate_label(sentence_list: list[str], given_labels: list[str]) -> str``
* ``prompt_construct_merge_label(label_list: list[str]) -> str``
* ``prompt_construct(label_list: list[str], sentence: str) -> str``

For the 0%-seed configuration (SPEC §5.6.2), pass ``given_labels=[]`` to
``prompt_construct_generate_label``.
"""

from __future__ import annotations

import benchmarking.baselines.huang_he  # noqa: F401 — puts _vendored/ on sys.path

from label_generation import (  # type: ignore[import-not-found]
    prompt_construct_generate_label,
    prompt_construct_merge_label,
)
from given_label_classification import (  # type: ignore[import-not-found]
    prompt_construct as prompt_construct_classify,
)

__all__ = [
    "prompt_construct_generate_label",
    "prompt_construct_merge_label",
    "prompt_construct_classify",
]
