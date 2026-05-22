"""Subprocess entry point for vendored ClusterLLM scripts.

We can't run the vendored scripts as ``python script.py`` directly on Windows
because TLS to HuggingFace fails unless ``truststore`` routes Python's SSL
context through the OS cert store. The injection lives in
``benchmarking/__init__.py`` (see comment there) and only runs when something
imports the ``benchmarking`` package — but the vendored scripts don't.

This wrapper imports the parent package (triggering the injection), then
dispatches to the target script via ``runpy`` with cwd preserved so the
vendored ``from InstructorEmbedding import INSTRUCTOR`` style imports still
resolve.

Driven by ``orchestrate._run``. Usage:

    python -m benchmarking.baselines.clusterllm._run_vendored \\
        <absolute-path-to-vendored-script> [args...]
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import benchmarking  # noqa: F401 — side effect: truststore.inject_into_ssl()


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: _run_vendored.py <script.py> [args...]", file=sys.stderr)
        sys.exit(2)

    script = Path(sys.argv[1])
    if not script.is_file():
        print(f"_run_vendored: not a file: {script}", file=sys.stderr)
        sys.exit(2)

    # Shift sys.argv so the target script sees a normal argv shape.
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
