"""Autonomous finisher for the auditor-diversity sweep.

A ~26-min network outage instant-failed 4 cells (massive_domain, stackexchange,
clinc150, twenty_newsgroups) while the --all driver was running; goemotions then
recovered and is finishing. This script:

  Phase 1 — wait for the current --all driver to exit, watching goemotions for a
            hard stall (no log activity for >30 min while the driver is alive →
            exit with STALL so the operator is pinged).
  Phase 2 — retry loop: for every target cell still missing final_taxonomy.json,
            delete its stale (empty) workspace and re-run it via the driver. A
            fresh network blip can't permanently skip a cell — it's retried up to
            MAX_ROUNDS with a pause between rounds.

Idempotent: a cell that already has final_taxonomy.json is never touched, so the
two clean cells (banking77, massive_intent) and any cell goemotions finishes are
left alone.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "clustering"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0

# Cells the outage killed (re-run after the current driver exits) plus goemotions
# (only re-run if the current driver didn't finalize it).
TARGETS = ["massive_domain", "stackexchange", "clinc150", "twenty_newsgroups", "goemotions"]
MAX_ROUNDS = 5
ROUND_PAUSE_S = 120
STALL_LIMIT_S = 30 * 60


def log(msg: str) -> None:
    print(f"[finisher {datetime.now(timezone.utc):%H:%M:%SZ}] {msg}", flush=True)


def ws(ds: str) -> Path:
    return RESULTS / ds / "seed=0_auditdiv"


def finalized(ds: str) -> bool:
    return (ws(ds) / "final_taxonomy.json").exists()


def driver_running() -> bool:
    out = subprocess.run(
        [
            "powershell", "-NoProfile", "-Command",
            "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
            "Where-Object { $_.CommandLine -like '*run_audit_diversity_sweep*' }) "
            "{'Y'} else {'N'}",
        ],
        capture_output=True, text=True, creationflags=NO_WINDOW,
    )
    return "Y" in (out.stdout or "")


def goe_mtime() -> float:
    p = ws("goemotions") / "log.jsonl"
    return p.stat().st_mtime if p.exists() else 0.0


def run_driver(datasets: list[str]) -> None:
    env = dict(os.environ)
    env.pop("SSL_CERT_FILE", None)
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [
        "uv", "run", "--native-tls", "python", "-m",
        "benchmarking.experiments.run_audit_diversity_sweep", "--only", *datasets,
    ]
    log(f"running driver --only {' '.join(datasets)}")
    subprocess.run(cmd, cwd=str(ROOT), env=env, creationflags=NO_WINDOW)


def main() -> None:
    # Phase 1 — wait for the current --all driver to finish.
    log("phase 1: waiting for current --all driver to exit")
    last_m, last_change = goe_mtime(), time.time()
    while driver_running():
        time.sleep(60)
        m = goe_mtime()
        if m != last_m:
            last_m, last_change = m, time.time()
        elif not finalized("goemotions") and (time.time() - last_change) > STALL_LIMIT_S:
            log("STALL: goemotions log idle >30min while driver alive — exiting for operator review")
            return
    log("current driver has exited")

    # Phase 2 — re-run every target still missing its taxonomy, with retries.
    for rnd in range(1, MAX_ROUNDS + 1):
        missing = [ds for ds in TARGETS if not finalized(ds)]
        if not missing:
            log("all target cells finalized — done")
            break
        log(f"round {rnd}/{MAX_ROUNDS}: {len(missing)} missing -> {missing}")
        for ds in missing:
            d = ws(ds)
            if d.exists():
                import shutil
                log(f"  removing stale workspace {d}")
                shutil.rmtree(d, ignore_errors=True)
        run_driver(missing)
        still = [ds for ds in TARGETS if not finalized(ds)]
        if not still:
            log("all target cells finalized after this round — done")
            break
        if rnd < MAX_ROUNDS:
            log(f"still missing {still}; pausing {ROUND_PAUSE_S}s before retry")
            time.sleep(ROUND_PAUSE_S)
    else:
        log(f"reached MAX_ROUNDS; still missing: {[d for d in TARGETS if not finalized(d)]}")

    done = [ds for ds in ["banking77", "massive_intent", *TARGETS] if finalized(ds)]
    log(f"FINISHER DONE. finalized cells ({len(done)}/7): {done}")


if __name__ == "__main__":
    main()
