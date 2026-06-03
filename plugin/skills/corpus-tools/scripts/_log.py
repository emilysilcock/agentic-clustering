"""Shared log.jsonl writer.

`init.py`, `state.py`, and `sample.py` all append the same shape of entry to
the workspace's `log.jsonl`. Keeping the writer here means the shape can't
drift between callers — previously three slightly-different functions did the
same thing and only stayed in sync because someone remembered to update all
three.

Stdlib only (no third-party deps); safe to import from any PEP 723 script in
this directory.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def append_log(
    log_path: Path,
    action: str,
    detail: str,
    metadata: dict | None = None,
) -> None:
    """Append one JSON line to ``log_path``.

    Schema: ``{"timestamp": "<iso8601 Z>", "action": ..., "detail": ...}``,
    plus an optional ``"metadata": {...}`` when ``metadata`` is non-empty.
    Timestamps are UTC, second precision, matching the rest of the codebase.
    """
    entry: dict = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "action": action,
        "detail": detail,
    }
    if metadata:
        entry["metadata"] = metadata
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
