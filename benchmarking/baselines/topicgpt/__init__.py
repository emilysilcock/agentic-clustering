"""Add the vendored topicgpt_python package to sys.path on import.

Side-effect import: subsequent ``from topicgpt_python.utils import APIClient``
inside this baseline (or in the vendored modules themselves, via their
relative ``from .utils import *``) resolves to our patched copy.
"""

from __future__ import annotations

import sys
from pathlib import Path

_VENDORED = Path(__file__).resolve().parent / "_vendored"
if str(_VENDORED) not in sys.path:
    sys.path.insert(0, str(_VENDORED))
