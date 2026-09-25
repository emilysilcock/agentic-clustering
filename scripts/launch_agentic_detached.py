"""Launch a ``run_agentic_clustering`` run as a fully DETACHED background process.

Why: a Bash ``run_in_background`` task is a child of the Claude Code session and
gets reaped when a long session summarizes or transitions --- which cost five
restarts of the token sweep before ``launch_tokrun_detached.py`` existed. A
``DETACHED_PROCESS`` runs independently and survives until the run finishes,
which matters here because the agent loop plus an OpenAI batch is comfortably
longer than a session turn. Windowless per the project's no-console rule
(``pythonw.exe`` + ``CREATE_NO_WINDOW``; the ``claude`` children get
``CREATE_NO_WINDOW`` of their own via claude_code.py).

Everything after ``--`` is passed through to the experiment runner, so this is
the detached form of any invocation of it:

    python scripts/launch_agentic_detached.py --log-name banking77_givenk -- --only banking77

Returns immediately, printing the pid and the log path. Not idempotent by
itself --- the runner is what decides whether re-running a dataset re-does work,
so check the log before relaunching a run that may still be in flight (a
duplicate classify batch is a duplicate bill).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DETACHED_PROCESS = 0x00000008
CREATE_NO_WINDOW = 0x08000000


def _interpreter() -> Path:
    """pythonw (GUI subsystem, no console) is the robust windowless choice.

    A venv ``python.exe`` is a redirector that re-launches the base interpreter
    as a child, and CREATE_NO_WINDOW on the redirector does not stop that
    grandchild taking its own console --- so prefer pythonw and only fall back
    when the venv doesn't ship it.
    """
    scripts = REPO / ".venv" / "Scripts"
    for name in ("pythonw.exe", "python.exe"):
        cand = scripts / name
        if cand.exists():
            return cand
    return Path(sys.executable)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-name",
        default=None,
        help="Basename for the log under logs/ (default: a timestamp).",
    )
    parser.add_argument(
        "runner_args",
        nargs=argparse.REMAINDER,
        help="Args for run_agentic_clustering, after a literal --.",
    )
    args = parser.parse_args()

    runner_args = args.runner_args
    if runner_args and runner_args[0] == "--":
        runner_args = runner_args[1:]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.log_name or f"agentic_{stamp}"
    log_path = REPO / "logs" / f"{name}.log"
    log_path.parent.mkdir(exist_ok=True)

    env = dict(os.environ)
    # Every corpus-tools script reconfigures stdout to UTF-8, but the runner's
    # own prints go through this process; Windows would otherwise pick cp1252
    # and die on a cluster name with a non-ASCII character in it.
    env["PYTHONIOENCODING"] = "utf-8"

    interp = _interpreter()
    cmd = [str(interp), "-m", "benchmarking.experiments.run_agentic_clustering", *runner_args]

    log_fh = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO),
        stdin=subprocess.DEVNULL,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=env,
        creationflags=DETACHED_PROCESS | CREATE_NO_WINDOW,
        close_fds=True,
    )
    print(f"detached run launched: interpreter={interp.name} pid={proc.pid}")
    print(f"  cmd: {' '.join(cmd[1:])}")
    print(f"  log: {log_path}")


if __name__ == "__main__":
    main()
