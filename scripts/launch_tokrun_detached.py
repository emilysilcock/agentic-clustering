"""Launch the token-measurement sweep as a fully DETACHED background process.

Why: the Bash ``run_in_background`` task is a child of the Claude Code session
and got reaped repeatedly (5x) when this long session summarized/transitioned.
A ``DETACHED_PROCESS`` runs independently of the session and survives until the
sweep actually finishes. Windowless per the project's no-console rule
(``pythonw.exe`` + ``CREATE_NO_WINDOW``; the ``claude`` children are also
``CREATE_NO_WINDOW`` via claude_code.py).

Run once (returns immediately):
    python scripts/launch_tokrun_detached.py

Idempotent: the sweep skips datasets whose ``orchestrator_result.json`` exists,
so re-launching after any interruption just continues.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / ".venv" / "Scripts"
# pythonw (GUI subsystem, no console) is the robust windowless choice; fall back
# to python.exe if the venv doesn't ship it.
interp = SCRIPTS / "pythonw.exe"
if not interp.exists():
    interp = SCRIPTS / "python.exe"

log_path = REPO / "logs" / "tokrun_sweep.log"
log_path.parent.mkdir(exist_ok=True)

env = dict(os.environ)
env["PYTHONIOENCODING"] = "utf-8"

DETACHED_PROCESS = 0x00000008
CREATE_NO_WINDOW = 0x08000000

log_fh = open(log_path, "w", encoding="utf-8")
proc = subprocess.Popen(
    [str(interp), "-m", "benchmarking.experiments.measure_agentic_tokens", "--all", "--config", "both"],
    cwd=str(REPO),
    stdin=subprocess.DEVNULL,
    stdout=log_fh,
    stderr=subprocess.STDOUT,
    env=env,
    creationflags=DETACHED_PROCESS | CREATE_NO_WINDOW,
    close_fds=True,
)
print(f"detached sweep launched: interpreter={interp.name} pid={proc.pid} log={log_path}")
