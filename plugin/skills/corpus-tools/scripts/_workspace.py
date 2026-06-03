"""Shared workspace-path resolution.

Six entry points (state.py, sample.py, search.py, metrics.py, confusion.py,
validate.py) all need to resolve the active workspace via the same three-step
fallback: env var, pointer file at the fixed default location, default. Centralising
that lookup here means a fix to the resolution rule lands once.

The bash-script version of this fallback (in agent .md and SKILL.md files)
cannot share Python code; it stays inlined there.

Stdlib only (no third-party deps); safe to import from any PEP 723 script in
this directory.
"""

from __future__ import annotations

import os
from pathlib import Path


def get_workspace() -> Path:
    """Return the active workspace path.

    Resolution order:
      1. ``$CLUSTERING_WORKSPACE`` if set.
      2. The pointer file ``.claude/clustering/.active_workspace`` that
         ``init.py`` writes at a fixed, project-root-relative location.
         (``$CLUSTERING_WORKSPACE`` does not survive across Bash tool calls
         or reach hook subprocesses, but hooks and tool calls share the
         project-root cwd, so the pointer is reliable.)
      3. The default ``.claude/clustering``.
    """
    env_ws = os.environ.get("CLUSTERING_WORKSPACE")
    if env_ws:
        return Path(env_ws)
    pointer = Path(".claude/clustering/.active_workspace")
    if pointer.exists():
        ws = pointer.read_text(encoding="utf-8").strip()
        if ws:
            return Path(ws)
    return Path(".claude/clustering")
