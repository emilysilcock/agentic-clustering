"""Run a command as a fully DETACHED background process and return at once.

Why this exists: a long-blocking command run directly inside a Claude Code
session's Bash tool is bounded by that tool's per-call timeout (10 minutes),
and any child of the session is reaped when the session exits. A provider
Batch API poll runs for minutes to hours, so it needs to outlive both. This
launcher spawns the command in its own process group / with DETACHED_PROCESS,
so it survives the session, and exits immediately so no tool call blocks.

Generalises ``launch_tokrun_detached.py`` (which hardcodes one command) to an
arbitrary argv. Windowless per the project's no-console rule.

Usage (note the ``--`` separator; everything after it is the command):

    python scripts/launch_detached.py --log logs/classify.log -- \\
        uv run /path/to/classify.py --input ... --output ...

Prints ``pid=<n> log=<path>`` on success. The caller polls the command's real
output (or the log) to know when it finished; this script tells you only that
it started.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DETACHED_PROCESS = 0x00000008
CREATE_NO_WINDOW = 0x08000000


def main() -> int:
    p = argparse.ArgumentParser(
        description="Launch a command detached from this process and exit.",
    )
    p.add_argument("--log", required=True, help="File to append stdout+stderr to")
    p.add_argument(
        "--pid-file",
        help="Optional path to write the child PID to (default: <log>.pid)",
    )
    p.add_argument(
        "--cwd",
        default=str(REPO),
        help="Working directory for the child (default: repo root)",
    )
    p.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="The command to run, after a `--` separator",
    )
    args = p.parse_args()

    cmd = args.command
    # argparse.REMAINDER keeps the leading `--` when it is present.
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print(
            "no command given; put it after a `--` separator, e.g.\n"
            "  python scripts/launch_detached.py --log x.log -- uv run script.py",
            file=sys.stderr,
        )
        return 2

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path = Path(args.pid_file) if args.pid_file else log_path.with_suffix(
        log_path.suffix + ".pid"
    )

    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")

    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = DETACHED_PROCESS | CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True

    # Append rather than truncate, so a retry keeps the earlier attempt's log
    # (a submitted batch id in there is what makes an orphan recoverable).
    with open(log_path, "a", encoding="utf-8") as log_fh:
        log_fh.write(f"\n=== launch_detached: {' '.join(cmd)}\n")
        log_fh.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=args.cwd,
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            close_fds=True,
            **kwargs,
        )

    pid_path.write_text(str(proc.pid), encoding="utf-8")
    print(f"pid={proc.pid} log={log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
