"""Per-dataset clustering lens for LLM-clustering methods that take one.

Only our method (`agentic_clustering`) consumes the free-text ``text`` field
today; TopicGPT and Huang & He don't expose an equivalent input surface in
their original implementations, so their runners import only ``allow_none``
to drive the classify-step force-assign flag.

Texts are deliberately label-name-free and k-free: they describe the corpus
and the grouping criterion (intent / domain / emotion / topic) without
leaking gold-label names or category counts. The OOS / "none" handling is
controlled out-of-band by ``allow_none``, not by the lens text itself.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetLens:
    text: str
    allow_none: bool


DATASET_LENS: dict[str, DatasetLens] = {
    "banking77": DatasetLens(
        text=(
            "Banking customer-service queries. Group texts by the specific "
            "action or information the customer is asking about. Clusters are "
            "fine-grained customer intents."
        ),
        allow_none=False,
    ),
    "clinc150": DatasetLens(
        text=(
            "Multi-domain virtual-assistant queries. Group texts by the "
            "specific intent — the precise action or piece of information the "
            "user is requesting."
        ),
        allow_none=True,
    ),
    "massive_intent": DatasetLens(
        text=(
            "Voice-assistant utterances. Group texts by user intent — the "
            "specific action being requested. Clusters are fine-grained "
            "intents (not broad domains)."
        ),
        allow_none=False,
    ),
    "massive_domain": DatasetLens(
        text=(
            "Voice-assistant utterances. Group texts by the broad domain or "
            "scenario the utterance belongs to. Clusters are coarse domain "
            "categories (not fine intents)."
        ),
        allow_none=False,
    ),
    "goemotions": DatasetLens(
        text=(
            "Reddit comments. Group texts by the primary emotion expressed."
        ),
        allow_none=True,
    ),
    "twenty_newsgroups": DatasetLens(
        text=(
            "Usenet newsgroup posts from the 1990s. Group texts by topic — "
            "the subject the post is about."
        ),
        allow_none=False,
    ),
    "stackexchange": DatasetLens(
        text=(
            "StackExchange posts collected across multiple sites. Group texts "
            "by topic or subject area."
        ),
        allow_none=False,
    ),
}
