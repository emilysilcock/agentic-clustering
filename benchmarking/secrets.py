"""Load API keys from a gitignored `secrets.json` into `os.environ`.

Convention: `secrets.json` sits at the project root with a flat dict of
`env-var-name → value`. Calling `load_secrets_into_env()` copies entries into
`os.environ` only if the variable isn't already set — so real environment
variables (CI, user-level Windows env vars) take precedence over the file.

Convenience for local development. The file is gitignored.

Consumers: `benchmarking.embeddings.openai_embeddings` (OpenAI_API_KEY).
Future: Anthropic batch-API runners for TopicGPT / Viswanathan+ / Huang & He
(ANTHROPIC_API_KEY) will use the same loader.
"""

from __future__ import annotations

import json
import os

from benchmarking.paths import ROOT

SECRETS_PATH = ROOT / "secrets.json"


def load_secrets_into_env() -> None:
    """Populate `os.environ` from `secrets.json`. Idempotent; no-op if file missing."""
    if not SECRETS_PATH.exists():
        return
    data = json.loads(SECRETS_PATH.read_text(encoding="utf-8"))
    for key, value in data.items():
        if key not in os.environ and value:
            os.environ[key] = str(value)
