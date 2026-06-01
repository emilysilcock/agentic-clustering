#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Validate agent output files.

Reads SubagentStop hook payload from stdin to identify the exact output file
the agent created. Validates JSON structure and required keys.

Exit codes:
  0 = valid (or max retries exceeded)
  2 = invalid (keep agent alive to fix)
"""

import json
import os
import re
import sys
import tempfile
from pathlib import Path

# Force UTF-8 on stdout/stderr — Windows defaults to cp1252 and crashes on
# non-ASCII cluster names / corpus content. Idempotent; no-op on streams that
# aren't TextIOWrapper (e.g. captured in tests).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _get_workspace() -> Path:
    env_ws = os.environ.get("CLUSTERING_WORKSPACE")
    if env_ws:
        return Path(env_ws)
    # CLUSTERING_WORKSPACE does not survive across Bash tool calls or reach hook
    # subprocesses, so fall back to the pointer init.py writes at a fixed,
    # project-root-relative location (hooks and tool calls share that cwd).
    pointer = Path(".claude/clustering/.active_workspace")
    if pointer.exists():
        ws = pointer.read_text(encoding="utf-8").strip()
        if ws:
            return Path(ws)
    return Path(".claude/clustering")


WORKSPACE = _get_workspace()
MAX_RETRIES = 3

# Required keys per agent type (inferred from output directory)
REQUIRED_KEYS = {
    "proposals": ["timestamp", "sample_size", "clusters"],
    "audits": ["timestamp", "n_texts", "assignments", "summary"],
    "investigations": ["timestamp"],  # investigations have varied formats
    "critiques": ["timestamp", "clusters_reviewed", "checklist", "issues", "overall_assessment"],
}

# Additional keys for specific investigation types (synthesizer and
# investigator outputs both live under investigations/).
INVESTIGATION_SUBTYPES = {
    "synthesis_": ["proposals_merged", "clusters_produced"],
    "inv_": ["question", "recommendation"],
}


def parse_stdin() -> tuple[str | None, str | None]:
    """Parse SubagentStop hook payload from stdin.

    Returns (output_file, agent_session_key).
    The agent_session_key is unique per agent invocation and includes the
    output filename (which contains a timestamp+UUID). This ensures parallel
    agents of the same type get independent retry counters. The key stays
    stable across retries for the same invocation because the hook payload
    references the same file path each time.
    """
    try:
        payload = sys.stdin.read()
    except Exception:
        return None, None

    if not payload.strip():
        return None, None

    # Extract output file path. We try two strategies:
    # 1. Look for Write tool file_path arguments (strongest signal)
    # 2. Fall back to last matching path in payload
    # Normalize Windows path separators: JSON encodes each \ as \\, so convert
    # pairs to / in both payload and workspace, then match. Keeps the regex
    # platform-agnostic without parsing the payload as JSON.
    ws_norm = str(WORKSPACE).replace("\\", "/")
    payload_norm = payload.replace("\\\\", "/")
    ws_pattern = re.escape(ws_norm)
    file_pattern = ws_pattern + r'/(proposals|audits|investigations|critiques)/[^\s"\'}\]]+\.json'

    # Strategy 1: Match paths near Write tool indicators
    # The hook payload includes tool call history; Write calls have "file_path"
    write_pattern = r'"file_path"\s*:\s*"([^"]*' + ws_pattern + r'/(proposals|audits|investigations|critiques)/[^"]+\.json)"'
    write_matches = list(re.finditer(write_pattern, payload_norm))

    output_file = None
    output_dir = None
    if write_matches:
        # Use the last Write tool target
        last = write_matches[-1]
        output_file = last.group(1)
        output_dir = last.group(2)
    else:
        # Strategy 2: Fall back to last general match
        all_full = list(re.finditer(file_pattern, payload_norm))
        if all_full:
            last = all_full[-1]
            output_file = last.group(0)
            output_dir = last.group(1)

    # Extract agent name for stable session key.
    # SubagentStop payload field is `agent_type` (NOT `agent_name`), and plugin
    # agents arrive namespaced as `agentic-clustering:<name>` — verified empirically
    # 2026-06-01 by dumping the live payload. The capture group strips the
    # namespace so the session key reads "proposer_<file_stem>" instead of
    # "agentic-clustering:proposer_<file_stem>".
    agent_pattern = r'"agent_type"\s*:\s*"agentic-clustering:(proposer|synthesizer|auditor|investigator|critic)"'
    agent_match = re.search(agent_pattern, payload_norm)
    agent_name = agent_match.group(1) if agent_match else (output_dir or "unknown")

    # Build a session key that is unique per agent *invocation*.
    # Include the output filename (which contains a timestamp+UUID) so that
    # parallel agents of the same type get independent retry counters.
    # The key still stays stable across retries for the same agent because
    # the hook payload references the same file path each time.
    if output_file:
        file_stem = Path(output_file).stem
        session_key = f"{agent_name}_{file_stem}"
    else:
        session_key = f"{agent_name}_{output_dir or 'unknown'}"

    return output_file, session_key


def get_retry_count(session_key: str) -> int:
    """Get retry count keyed by agent session."""
    retry_file = Path(tempfile.gettempdir()) / f"clustering_validate_{session_key}"
    if retry_file.exists():
        try:
            return int(retry_file.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            return 0
    return 0


def increment_retry(session_key: str) -> int:
    """Increment and return retry count."""
    retry_file = Path(tempfile.gettempdir()) / f"clustering_validate_{session_key}"
    count = get_retry_count(session_key) + 1
    retry_file.write_text(str(count))
    return count


def clear_retry(session_key: str):
    """Clear retry tracking."""
    retry_file = Path(tempfile.gettempdir()) / f"clustering_validate_{session_key}"
    if retry_file.exists():
        retry_file.unlink()


def log_error(message: str):
    """Log a validation error to log.jsonl."""
    from datetime import datetime, timezone
    log_path = WORKSPACE / "log.jsonl"
    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "action": "validation_error",
        "detail": message,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def validate_file(file_path: str) -> tuple[bool, str]:
    """Validate an output file. Returns (is_valid, error_message)."""
    path = Path(file_path)

    # Check file exists
    if not path.exists():
        return False, f"Output file not found: {file_path}"

    # Check valid JSON
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON in {file_path}: {e}"

    if not isinstance(data, dict):
        return False, f"Expected JSON object, got {type(data).__name__}"

    # Determine agent type from directory
    parent_dir = path.parent.name
    required = REQUIRED_KEYS.get(parent_dir, [])

    # Check required keys
    missing = [k for k in required if k not in data]
    if missing:
        return False, f"Missing required keys in {parent_dir} output: {missing}"

    # Check investigation subtypes
    if parent_dir == "investigations":
        filename = path.name
        for prefix, keys in INVESTIGATION_SUBTYPES.items():
            if filename.startswith(prefix):
                missing = [k for k in keys if k not in data]
                if missing:
                    return False, f"Missing keys for {prefix} investigation: {missing}"
                break

    return True, ""


def main():
    output_file, session_key = parse_stdin()

    if not output_file:
        # Can't determine output file — pass through silently
        # This can happen if the agent didn't write a file yet
        print("Warning: could not determine output file from hook payload", file=sys.stderr)
        sys.exit(0)

    is_valid, error_msg = validate_file(output_file)

    if is_valid:
        clear_retry(session_key)
        print(f"Validation passed: {output_file}")
        sys.exit(0)

    # Invalid — check retry count (keyed by stable session identity,
    # not output filename, so retries that write a new file still count)
    retry_count = increment_retry(session_key)

    if retry_count >= MAX_RETRIES:
        # Give up after max retries
        clear_retry(session_key)
        log_error(f"Agent failed validation {MAX_RETRIES} times: {error_msg}")
        print(f"Validation failed {MAX_RETRIES} times, giving up: {error_msg}", file=sys.stderr)
        sys.exit(0)  # Exit 0 so the agent stops (we logged the failure)

    # Signal the agent to retry
    print(f"Validation failed (attempt {retry_count}/{MAX_RETRIES}): {error_msg}", file=sys.stderr)
    sys.exit(2)  # Exit 2 keeps the agent alive to fix


if __name__ == "__main__":
    main()
