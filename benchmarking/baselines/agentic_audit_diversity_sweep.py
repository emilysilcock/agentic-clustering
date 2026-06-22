"""Single-arm sweep: does the AUDITOR sampling diversely improve the taxonomy?

By default the auditor pulls its fresh audit texts with a uniform random draw
(``sample.py --strategy random``), which over-represents dense regions of the
corpus. This experiment instead has the auditor draw with **TF-IDF
farthest-point (max-min) diversity sampling** (``sample.py --strategy diverse``,
the permanent capability added on this branch), so the audit set spreads across
the corpus's breadth — including sparse/edge regions a random draw misses. The
hypothesis: a diverse audit surfaces genuine coverage gaps the proposers' random
samples skipped, so the orchestrator's add/refine decisions land a better final
taxonomy.

Only one arm is run here — ``diverse``. The **random baseline is the existing
production ``seed=0_discoverk`` run** (same plugin lineage, discover-k, taxonomy
stage only); we don't re-run it. The downstream comparison script reads both.

Like the proposer-count / sample-size / overlap sweeps, this runs in
**discover-k** mode (``k_in_scope ± 20%``) and **skips classification entirely**
(no predictions, no metered API spend — the Opus orchestrator + subagents run on
the Claude Code Max subscription). The verdict is on the discovered taxonomy's k
and structure (plus a diversity-of-audited-region diagnostic), not downstream
partition metrics.

Capability vs policy (mirrors agentic_overlap_sweep.py):
- **Capability (permanent, shipped in the plugin):** ``sample.py --strategy
  diverse``. General-purpose, not wired into any default — the auditor still
  samples randomly unless told otherwise.
- **Policy (per-run, lives here, NOT the plugin):** an additive directive
  appended to the orchestrator prompt instructing it to dispatch the Auditor
  with ``--strategy diverse``. Additive — the plugin has no rule about which
  strategy the auditor uses, so nothing is contradicted.

Workspaces:
    method:    agentic_clustering_auditdiv
    workspace: clustering/<ds>/seed=<n>_auditdiv/          (NEW)
    baseline:  clustering/<ds>/seed=<n>_discoverk/          (READ-ONLY, existing)

Isolation rules from agentic_ablations.py / agentic_overlap_sweep.py: every
helper imported from ``agentic_clustering`` is read-only; the production module
is never edited; a new method + new workspace name guarantee existing artifacts
cannot be overwritten.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from benchmarking.baselines.agentic_clustering import (
    DISCOVER_K_FRACTION,
    LLM_TOKEN_CAP,
    MAX_AGENT_DISPATCHES,
    ORCHESTRATOR_MODEL,
    PLUGIN_ROOT,
    _ensure_orchestrator_outputs,
    _init_workspace,
    _materialize_capped_corpus,
    _orchestrator_prompt,
    _read_final_taxonomy,
)
from benchmarking.data_processing.load import load_processed
from benchmarking.dataset_lens import DATASET_LENS
from benchmarking.llm_clients.claude_code import call_claude
from benchmarking.paths import RESULTS

# Smallest-up sweep order — mirrors run_agentic_clustering.SWEEP_ORDER so the
# Max subscription's rolling cap drains predictably from the cheapest run up.
SWEEP_ORDER = [
    "banking77",
    "massive_intent",
    "massive_domain",
    "stackexchange",
    "clinc150",
    "twenty_newsgroups",
    "goemotions",
]

METHOD = "agentic_clustering_auditdiv"


def workspace_for(dataset: str, *, seed: int) -> Path:
    return RESULTS / "clustering" / dataset / f"seed={seed}_auditdiv"


def _diversity_override_block() -> str:
    """Additive directive: dispatch the Auditor with diverse sampling.

    The auditor's default is a uniform random draw (sample.py --strategy random).
    This block tells the orchestrator to instruct the Auditor, every time it is
    dispatched, to draw its audit sample with --strategy diverse instead. The
    plugin has no rule about which strategy the auditor uses, so this is purely
    additive — it does not contradict any SKILL.md / agent text.
    """
    sample = "$CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/sample.py"
    return (
        "\n"
        "AUDITOR DIVERSITY-SAMPLING DIRECTIVE FOR THIS RUN. By default the "
        "Auditor pulls its fresh audit texts with a uniform random draw "
        "(sample.py --strategy random), which over-represents dense regions of "
        "the corpus. For THIS run, every time you dispatch the Auditor you must "
        "instruct it — in its task description — to draw its audit sample with "
        "TF-IDF farthest-point DIVERSITY sampling instead of the default random "
        "draw:\n"
        "\n"
        f"  uv run {sample} --n <S> --strategy diverse\n"
        "\n"
        "where <S> is the audit sample size the Auditor would normally choose. "
        "This is the only change: it spreads the audit set across the breadth of "
        "the corpus (including sparse/edge regions a random draw under-samples) "
        "so the taxonomy is stress-tested against the corpus's full variety. "
        "Everything else is unchanged — the same audit sample size, the "
        "Proposer/Synthesizer/Critic stages, the stop criteria, and finalize. "
        "ONLY the Auditor's sampling strategy changes; the Proposers keep their "
        "normal (random, disjoint) sampling.\n"
    )


def _orchestrator_prompt_with_override(
    *,
    workspace_dir: Path,
    dataset: str,
    k_min: int,
    k_max: int,
    allow_none: bool,
) -> str:
    """Production orchestrator prompt + the auditor-diversity directive block.

    Uses the unmodified _orchestrator_prompt() so the rest of the harness
    (allow_none clause, finalize-only constraint, k-range clause, no-classify
    expectation) stays identical to production; only the directive is appended.
    """
    base = _orchestrator_prompt(
        workspace_dir=workspace_dir,
        dataset=dataset,
        k_min=k_min,
        k_max=k_max,
        allow_none=allow_none,
    )
    return base + _diversity_override_block()


def _run_orchestrator(
    *,
    workspace_dir: Path,
    dataset: str,
    k_min: int,
    k_max: int,
    allow_none: bool,
) -> dict:
    prompt = _orchestrator_prompt_with_override(
        workspace_dir=workspace_dir,
        dataset=dataset,
        k_min=k_min,
        k_max=k_max,
        allow_none=allow_none,
    )
    os.environ["CLUSTERING_WORKSPACE"] = str(workspace_dir)
    (workspace_dir / "orchestrator_prompt.txt").write_text(prompt, encoding="utf-8")

    # No classify step in this sweep, so the text-classification plugin isn't
    # needed in the headless session. Loading only agentic-clustering keeps it
    # minimal.
    extra_args = [
        "--plugin-dir", str(PLUGIN_ROOT),
        "--permission-mode", "bypassPermissions",
    ]
    # See agentic_overlap_sweep.py: strip ANTHROPIC_API_KEY defensively so
    # claude -p stays on the Max subscription rather than metered billing.
    os.environ.pop("ANTHROPIC_API_KEY", None)

    log_prefix = f"[auditdiv/{dataset}]"
    t0 = time.perf_counter()
    stdout = call_claude(
        prompt,
        model=ORCHESTRATOR_MODEL,
        timeout_s=60 * 60 * 4,
        log_prefix=log_prefix,
        extra_args=extra_args,
    )
    t1 = time.perf_counter()
    (workspace_dir / "orchestrator_stdout.txt").write_text(stdout or "", encoding="utf-8")
    return {"wall_clock_s": t1 - t0, "stdout": stdout}


def _write_run_summary(
    *,
    workspace_dir: Path,
    dataset: str,
    k_in_scope: int,
    k_range: tuple[int, int],
    allow_none: bool,
    n_docs: int,
    wall_clock_s: float,
    final_taxonomy: dict,
) -> None:
    """Compact JSON summary at the workspace root.

    The comparison script reads this first (plus log.jsonl and the audits) to
    populate its table. Kept minimal — anything richer is fetched on demand.
    """
    summary = {
        "method": METHOD,
        "variant": "diverse",
        "auditor_sampling": "diverse",
        "dataset": dataset,
        "n_docs": n_docs,
        "k_in_scope": k_in_scope,
        "k_range": list(k_range),
        "k_actual": len(final_taxonomy.get("clusters", [])),
        "cluster_version_at_finalize": int(final_taxonomy.get("cluster_version", 0)),
        "allow_none": allow_none,
        "discover_k": True,
        "llm_input_token_cap": LLM_TOKEN_CAP,
        "max_agent_dispatches": MAX_AGENT_DISPATCHES,
        "orchestrator_model": ORCHESTRATOR_MODEL,
        "orchestrator_wall_clock_s": wall_clock_s,
        "classify_skipped": True,
    }
    (workspace_dir / "_audit_diversity_sweep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_audit_diversity_sweep(dataset_name: str, *, seed: int = 0) -> dict:
    """Run one dataset of the sweep — taxonomy stage only, diverse-auditor arm.

    Stops after cluster-finalize writes final_taxonomy.json + taxonomy.md +
    categories.json. No classify call, no predictions. The downstream comparison
    script consumes the workspace directly and pairs it with the existing
    seed=<n>_discoverk baseline.
    """
    if dataset_name not in DATASET_LENS:
        raise KeyError(f"no DATASET_LENS entry for {dataset_name!r}")
    lens = DATASET_LENS[dataset_name]
    ds = load_processed(dataset_name)
    k_in_scope = int(ds.meta["k_in_scope"])
    n_docs = len(ds.documents)

    # 512-token-capped corpus (SPEC §5.1.1 / §5.6.3), same artefact the
    # production agentic_clustering run uses; the helper is idempotent.
    documents_path = _materialize_capped_corpus(dataset_name, ds)

    # Discover-k window — identical to the production discover-k config.
    k_min = round(k_in_scope * (1 - DISCOVER_K_FRACTION))
    k_max = round(k_in_scope * (1 + DISCOVER_K_FRACTION))

    workspace_dir = workspace_for(dataset_name, seed=seed)
    # Safety: never collide with the production discover-k workspace or any of
    # the existing ablation / sweep workspaces. Catches typos before they can
    # clobber a completed expensive run — especially the seed=<n>_discoverk
    # baseline this experiment compares against.
    forbidden = [
        RESULTS / "clustering" / dataset_name / f"seed={seed}",
        RESULTS / "clustering" / dataset_name / f"seed={seed}_discoverk",
        RESULTS / "clustering" / dataset_name / f"seed={seed}_discoverk_synthonly",
        RESULTS / "clustering" / dataset_name / f"seed={seed}_discoverk_notask",
    ]
    for f in forbidden:
        if workspace_dir.resolve() == f.resolve():
            raise RuntimeError(
                f"refusing to run: audit-diversity workspace {workspace_dir} "
                f"collides with existing workspace {f}"
            )

    workspace_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[auditdiv/{dataset_name}] init "
        f"(k_range=[{k_min},{k_max}], allow_none={lens.allow_none}, n={n_docs})"
    )
    _init_workspace(
        workspace_dir=workspace_dir,
        documents_path=documents_path,
        k_min=k_min,
        k_max=k_max,
        lens_text=lens.text,
    )

    t_start = time.perf_counter()
    print(
        f"[auditdiv/{dataset_name}] dispatching orchestrator on {ORCHESTRATOR_MODEL}"
    )
    orch = _run_orchestrator(
        workspace_dir=workspace_dir,
        dataset=dataset_name,
        k_min=k_min,
        k_max=k_max,
        allow_none=lens.allow_none,
    )
    print(
        f"[auditdiv/{dataset_name}] orchestrator returned in {orch['wall_clock_s']:.1f}s"
    )

    # Same finalize-output contract as the production run.
    _ensure_orchestrator_outputs(workspace_dir)

    final_taxonomy = _read_final_taxonomy(workspace_dir)
    wall_clock_s = time.perf_counter() - t_start

    _write_run_summary(
        workspace_dir=workspace_dir,
        dataset=dataset_name,
        k_in_scope=k_in_scope,
        k_range=(k_min, k_max),
        allow_none=lens.allow_none,
        n_docs=n_docs,
        wall_clock_s=wall_clock_s,
        final_taxonomy=final_taxonomy,
    )

    k_actual = len(final_taxonomy["clusters"])
    print(
        f"[auditdiv/{dataset_name}] done. k_actual={k_actual} "
        f"(k_in_scope={k_in_scope}, range=[{k_min},{k_max}]) "
        f"wall_clock={wall_clock_s:.1f}s"
    )
    return {
        "method": METHOD,
        "variant": "diverse",
        "dataset": dataset_name,
        "n_docs": n_docs,
        "k_in_scope": k_in_scope,
        "k_range": [k_min, k_max],
        "k_actual": k_actual,
        "wall_clock_s": wall_clock_s,
        "workspace": str(workspace_dir),
    }
