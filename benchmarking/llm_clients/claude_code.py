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

import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


# No console window for the `claude` child — matters when the sweep is launched
# as a detached background process (a DETACHED_PROCESS parent has no console, so
# a default-flag child would allocate a *visible* one). Platform-guarded: 0 on
# non-Windows.
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_TIMEOUT_S = 180.0
DEFAULT_MAX_LIMIT_WAITS = 12
FALLBACK_WAIT_S = 30 * 60
WAKE_BUFFER_S = 60

_USAGE_LIMIT_MARKERS = (
    re.compile(r"usage limit", re.IGNORECASE),
    # Claude Code CLI prints "You've hit your session limit ... resets HH:MMam
    # (America/New_York)" on the rolling-window cap. Matched verbatim from
    # observed stderr on 2026-05-22; mismatching this wording previously caused
    # ~6000 ClusterLLM triplets to burn as fatal errors instead of waiting.
    re.compile(r"session limit", re.IGNORECASE),
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
    r"reset[s]?\s+(?:at\s+)?(?P<hm>\d{1,2}:\d{2})\s*(?P<ampm>am|pm)?",
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


# modelUsage token keys we sum across retry attempts; contextWindow /
# maxOutputTokens are constants (carried through, not summed).
_USAGE_SUM_KEYS = (
    "inputTokens", "outputTokens", "cacheReadInputTokens",
    "cacheCreationInputTokens", "costUSD", "webSearchRequests",
)


def _accumulate_model_usage(acc: dict, parsed: dict) -> None:
    """Add one attempt's ``modelUsage`` + ``total_cost_usd`` into ``acc``.

    Each ``claude -p`` invocation is a distinct session; on a usage-limit (429)
    the client re-invokes and the CLI resumes the same workspace, but each
    invocation's result JSON reports only *its own* session's usage. Summing
    across every attempt (429s + final success) recovers the true total
    consumed to produce the output --- otherwise the pre-limit work (often the
    bulk: proposers / synth / audit) is silently dropped and only the final
    session's tokens are seen. Distinct sessions don't share usage, so there is
    no double-counting.
    """
    for model, u in (parsed.get("modelUsage") or {}).items():
        s = acc["model_usage"].setdefault(model, {})
        for k in _USAGE_SUM_KEYS:
            v = u.get(k)
            if isinstance(v, (int, float)):
                s[k] = s.get(k, 0) + v
        for k in ("contextWindow", "maxOutputTokens"):
            if k in u:
                s[k] = u[k]
    c = float(parsed.get("total_cost_usd") or 0.0)
    acc["cost_usd"] += c
    acc["attempts"] += 1
    acc["per_attempt_cost_usd"].append(c)


def call_claude(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_limit_waits: int = DEFAULT_MAX_LIMIT_WAITS,
    log_prefix: str = "[claude_code]",
    extra_args: list[str] | None = None,
    capture: dict | None = None,
) -> str:
    """Run `claude -p` once. Block on usage limits until reset, then retry.

    Returns the assistant text (`proc.stdout`). Raises ``ClaudeCodeError`` on
    any non-usage-limit non-zero exit, or after ``max_limit_waits`` cycles of
    usage-limit hits in a row.

    ``capture``: when a dict is passed, the call runs with
    ``--output-format json`` and the parsed result object is stored under
    ``capture["result_json"]`` (fields include ``modelUsage`` and
    ``total_cost_usd``, which are session-wide and *include Task sub-agents* ---
    verified empirically; the top-level ``usage`` is main-loop only). The return
    value is then the JSON's ``result`` text rather than raw stdout, so the
    contract (return the assistant's text) is unchanged for callers. Consumers
    that don't pass ``capture`` keep the plain-text stdout path untouched.

    ``extra_args`` are inserted between the standard flags and the prompt —
    used by the agentic-clustering benchmark wrapper to pass
    ``--plugin-dir`` and ``--permission-mode bypassPermissions`` so the
    headless session can load the plugin and dispatch Task subagents.
    """
    waits = 0
    # Windows' CreateProcess caps the full command line at ~32 KB (Unicode
    # API); long prompts trip ``FileNotFoundError: [WinError 206] The
    # filename or extension is too long``. Above 8 KB we pipe the prompt
    # through stdin (``claude -p`` reads it from stdin when no positional
    # argument is given) so the cmdline stays short regardless of prompt
    # size. The threshold is a conservative cap below the ~30 KB Windows
    # ceiling — leaves headroom for the model/flag args.
    prompt_via_stdin = len(prompt) > 8000
    cmd = [
        "claude",
        "-p",
        "--model",
        model,
        "--no-session-persistence",
        *(["--output-format", "json"] if capture is not None else []),
        *(extra_args or []),
    ]
    if not prompt_via_stdin:
        cmd.append(prompt)
    stdin_input = prompt if prompt_via_stdin else None

    # Accumulates token usage across every attempt (incl. usage-limit retries)
    # so the captured total reflects all work, not just the final session.
    acc = {"model_usage": {}, "cost_usd": 0.0, "attempts": 0, "per_attempt_cost_usd": []}

    while True:
        try:
            proc = subprocess.run(
                cmd,
                input=stdin_input,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
                encoding="utf-8",
                errors="replace",
                creationflags=_CREATE_NO_WINDOW,
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
                    input=stdin_input,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    check=False,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=_CREATE_NO_WINDOW,
                )
            except subprocess.TimeoutExpired:
                raise ClaudeCodeError(
                    returncode=-1,
                    stderr=f"timeout twice in a row after {timeout_s}s (orig: {exc})",
                )

        stderr = proc.stderr or ""
        stdout = proc.stdout or ""
        combined = stdout + "\n" + stderr

        # Capture mode: a well-formed success envelope is authoritative, so we
        # accept it *before* the text-based usage-limit heuristic --- otherwise
        # a benign "rate limit"/"session limit" substring inside the JSON result
        # (e.g. in a cluster description) would be mistaken for a real cap and
        # trigger a spurious wait. A non-success/unparseable body falls through
        # to the limit/error handling below.
        if capture is not None and proc.returncode == 0:
            try:
                parsed = json.loads(stdout)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict) and parsed.get("type") == "result" and not parsed.get("is_error"):
                _accumulate_model_usage(acc, parsed)
                # Overwrite the final envelope's per-session totals with the
                # cross-attempt sums so downstream (orchestrator_result.json,
                # _summarize_orchestrator_usage) sees the true total consumed,
                # not just this last session's slice.
                parsed["modelUsage"] = acc["model_usage"]
                parsed["total_cost_usd"] = acc["cost_usd"]
                parsed["retry_attempts"] = acc["attempts"]
                parsed["per_attempt_cost_usd"] = acc["per_attempt_cost_usd"]
                capture["result_json"] = parsed
                return parsed.get("result", "") or ""

        if proc.returncode == 0 and not _looks_like_usage_limit(combined):
            if capture is not None:
                raise ClaudeCodeError(
                    returncode=0,
                    stderr=(
                        "--output-format json: exit 0 but result was not a "
                        f"success envelope; head: {stdout[:400]!r}"
                    ),
                )
            return stdout

        if _looks_like_usage_limit(combined):
            # Bank this interrupted attempt's usage before sleeping --- the
            # pre-limit work (proposers/synth/audit) is real and would otherwise
            # be lost when the retry starts a fresh session.
            if capture is not None:
                try:
                    p429 = json.loads(stdout)
                except ValueError:
                    p429 = None
                if isinstance(p429, dict) and p429.get("modelUsage"):
                    _accumulate_model_usage(acc, p429)
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
