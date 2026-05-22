"""Subprocess wrapper around `claude -p` (Opus 4.7, billed against the Max plan).

Why subprocess instead of the Anthropic SDK: the Max plan covers Opus 4.7
usage under a single monthly subscription. Calls routed through the SDK with
an API key would bill at metered rates instead. See SPEC §5.6.2 for the
per-baseline routing decisions.

Usage-limit handling: the Max plan enforces a rolling 5-hour usage window.
When hit, `claude -p` exits non-zero and prints a reset time. ``call_claude``
catches that case, sleeps until the reset (with a buffer) and retries the
same prompt. Any other non-zero exit raises ``ClaudeCodeError`` immediately.

The first time we encounter a usage-limit error in the wild, the raw stderr
gets logged so we can refine the patterns below if the heuristics miss.

Consumers:
- ``benchmarking.baselines.clusterllm.triplet_judge`` (Phase 2 triplet
  judging — the only consumer at the time of writing).
- Future: Huang & He merge step (Opus on small samples), and any other
  Opus-routed call site that doesn't fit the Anthropic Batch API.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_TIMEOUT_S = 180.0
DEFAULT_MAX_LIMIT_WAITS = 12
FALLBACK_WAIT_S = 30 * 60
WAKE_BUFFER_S = 60

_USAGE_LIMIT_MARKERS = (
    re.compile(r"usage limit", re.IGNORECASE),
    re.compile(r"5-hour limit", re.IGNORECASE),
    re.compile(r"reached your[^.]*limit", re.IGNORECASE),
    re.compile(r"quota exceeded", re.IGNORECASE),
    re.compile(r"rate[\s-]?limit(?:ed)?", re.IGNORECASE),
)

_RESET_ISO = re.compile(
    r"reset[s]?\s+(?:at\s+)?(?P<iso>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?Z?)",
    re.IGNORECASE,
)
_RESET_CLOCK = re.compile(
    r"reset[s]?\s+at\s+(?P<hm>\d{1,2}:\d{2})\s*(?P<ampm>am|pm)?",
    re.IGNORECASE,
)
_RESET_RELATIVE = re.compile(
    r"(?:try\s+again\s+in|reset[s]?\s+in)\s+"
    r"(?:(?P<h>\d+)\s*h)?\s*(?:(?P<m>\d+)\s*m)?",
    re.IGNORECASE,
)


@dataclass
class ClaudeCodeError(RuntimeError):
    """Non-usage-limit failure from `claude -p`."""

    returncode: int
    stderr: str

    def __str__(self) -> str:
        head = self.stderr.strip().splitlines()[:6]
        return f"claude -p exited {self.returncode}: " + " | ".join(head)


def call_claude(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_limit_waits: int = DEFAULT_MAX_LIMIT_WAITS,
    log_prefix: str = "[claude_code]",
) -> str:
    """Run `claude -p` once. Block on usage limits until reset, then retry.

    Returns the assistant text (`proc.stdout`). Raises ``ClaudeCodeError`` on
    any non-usage-limit non-zero exit, or after ``max_limit_waits`` cycles of
    usage-limit hits in a row.
    """
    waits = 0
    cmd = [
        "claude",
        "-p",
        "--model",
        model,
        "--no-session-persistence",
        prompt,
    ]

    while True:
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired as exc:
            print(
                f"{log_prefix} timeout after {timeout_s}s; one retry then bail",
                file=sys.stderr,
                flush=True,
            )
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    check=False,
                    encoding="utf-8",
                    errors="replace",
                )
            except subprocess.TimeoutExpired:
                raise ClaudeCodeError(
                    returncode=-1,
                    stderr=f"timeout twice in a row after {timeout_s}s (orig: {exc})",
                )

        stderr = proc.stderr or ""
        stdout = proc.stdout or ""
        combined = stdout + "\n" + stderr

        if proc.returncode == 0 and not _looks_like_usage_limit(combined):
            return stdout

        if _looks_like_usage_limit(combined):
            waits += 1
            if waits > max_limit_waits:
                raise ClaudeCodeError(
                    returncode=proc.returncode,
                    stderr=(
                        f"usage limit hit {waits} cycles in a row; giving up.\n"
                        f"last stderr:\n{stderr}"
                    ),
                )
            wait_s = _parse_wait_seconds(combined)
            wake_at = datetime.now() + timedelta(seconds=wait_s)
            print(
                f"{log_prefix} usage-limit (cycle {waits}/{max_limit_waits}); "
                f"sleeping {wait_s:.0f}s — wake at "
                f"{wake_at.strftime('%Y-%m-%d %H:%M:%S')}",
                file=sys.stderr,
                flush=True,
            )
            if waits == 1:
                preview = "\n".join(combined.strip().splitlines()[:12])
                print(
                    f"{log_prefix} first usage-limit hit; raw error preview:\n{preview}",
                    file=sys.stderr,
                    flush=True,
                )
            time.sleep(wait_s)
            continue

        raise ClaudeCodeError(returncode=proc.returncode, stderr=combined)


def _looks_like_usage_limit(text: str) -> bool:
    return any(p.search(text) for p in _USAGE_LIMIT_MARKERS)


def _parse_wait_seconds(text: str) -> float:
    """Best-effort parse of when the limit resets. Falls back to FALLBACK_WAIT_S."""
    m = _RESET_ISO.search(text)
    if m:
        try:
            raw = m["iso"].replace(" ", "T").rstrip("Z")
            t = datetime.fromisoformat(raw).replace(tzinfo=timezone.utc)
            delta = (t - datetime.now(timezone.utc)).total_seconds()
            return max(60.0, delta + WAKE_BUFFER_S)
        except ValueError:
            pass

    m = _RESET_CLOCK.search(text)
    if m:
        try:
            h, mm = m["hm"].split(":")
            h, mm = int(h), int(mm)
            if m["ampm"] and m["ampm"].lower() == "pm" and h < 12:
                h += 12
            if m["ampm"] and m["ampm"].lower() == "am" and h == 12:
                h = 0
            now = datetime.now()
            t = now.replace(hour=h, minute=mm, second=0, microsecond=0)
            if t <= now:
                t += timedelta(days=1)
            return max(60.0, (t - now).total_seconds() + WAKE_BUFFER_S)
        except (ValueError, KeyError):
            pass

    m = _RESET_RELATIVE.search(text)
    if m and (m["h"] or m["m"]):
        hours = int(m["h"]) if m["h"] else 0
        mins = int(m["m"]) if m["m"] else 0
        return max(60.0, hours * 3600 + mins * 60 + WAKE_BUFFER_S)

    return FALLBACK_WAIT_S
