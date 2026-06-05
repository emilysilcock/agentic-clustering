"""Add the vendored Huang & He upstream to sys.path on import.

Mirror of the ``topicgpt/__init__.py`` pattern. We only re-export the three
prompt builders (see ``prompts.py``); upstream's ``main()`` / ``chat()`` /
``eval()``-based parsing / ``evaluate.py`` are never executed. See
``_vendored/UPSTREAM.md`` and ``CHANGES.md`` for the full list of
harness-side substitutions.
"""

from __future__ import annotations

import sys
from pathlib import Path

_VENDORED = Path(__file__).resolve().parent / "_vendored"
if str(_VENDORED) not in sys.path:
    sys.path.insert(0, str(_VENDORED))
