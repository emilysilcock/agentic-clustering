"""Sweep over the orchestrator's initial proposer-count guidance.

Four variants, isolated from production: ``v0`` re-runs the current "2-3"
guidance as a fresh baseline (other plugin code has changed since the existing
``seed=0_discoverk`` workspaces were produced, so re-running gives an apples-to-
apples control), then ``v1`` / ``v2`` / ``v3`` substitute the count phrase
with "4-5" / "6-7" / "8-9" respectively. Every other word of SKILL.md's
"Number of proposals" paragraph is preserved verbatim — only the leading count
phrase changes.

All four variants run in **discover-k** mode (``k_in_scope ± 20%``, matching the
production discover-k baseline) and **skip the classification step entirely**.
We don't compute partition metrics here: with classification off there are no
predictions, and cross-proposal ARI isn't a valid signal because our proposers
don't share samples by default (each proposer pulls a fresh sample, so any
disagreement between proposers conflates "the model disagrees" with "they saw
different texts"). The comparison script in ``scripts/compare_proposer_sweep.py``
reads each variant's workspace and tabulates the signals that *are* valid:
final k vs gold k, the final auditor's coverage and mean_confidence, critic
verdict, agent-dispatch budget consumed, and an LLM-as-judge alignment score
against each dataset's gold taxonomy.

Workspaces:
    method:    agentic_clustering_proposers_v{0,1,2,3}
    workspace: clustering/<ds>/seed=<n>_proposers_v{0,1,2,3}/   (NEW)
    source:    clustering/<ds>/seed=<n>_discoverk/             (READ-ONLY,
                                                                untouched)

This module follows the isolation rules from agentic_ablations.py: every helper
imported from ``agentic_clustering`` is read-only; the production module is
never edited; new method + new workspace names guarantee the existing
``agentic_clustering`` / ``agentic_clustering_discoverk`` artifacts cannot be
overwritten.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ProposerVariant:
    name: str  # workspace + method suffix, e.g. "v1"
    count_phrase: str  # the phrase that replaces "2-3" in the SKILL.md rule
    description: str  # human label for logs / comparison table


# v0 is the current SKILL.md text verbatim — a fresh baseline so the comparison
# isn't muddled by other plugin changes since the existing seed=0_discoverk runs
# were produced.
VARIANTS: dict[str, ProposerVariant] = {
    "v0": ProposerVariant("v0", "2-3", "baseline (current SKILL.md guidance)"),
    "v1": ProposerVariant("v1", "4-5", "modest bump"),
    "v2": ProposerVariant("v2", "6-7", "substantial bump"),
    "v3": ProposerVariant("v3", "8-9", "aggressive bump"),
}


def method_for(variant: ProposerVariant) -> str:
    return f"agentic_clustering_proposers_{variant.name}"


def workspace_for(dataset: str, *, seed: int, variant: ProposerVariant) -> Path:
    return RESULTS / "clustering" / dataset / f"seed={seed}_proposers_{variant.name}"


# Verbatim copy of plugin/skills/cluster-run/SKILL.md L152-155 ("Number of
# proposals" paragraph), parameterised on the count phrase. Every other word
# must match the source — if SKILL.md changes, this string must be updated to
# match it, otherwise the override silently injects stale wording. The
# count-phrase token "{count}" is the *only* placeholder.
_PROPOSER_RULE_TEMPLATE = (
    "**Number of proposals** — start with {count} from different angles. "
    "Dispatch them in parallel using concurrent Task calls (send multiple Task "
    "invocations in one message). If they converge, that's signal. If they "
    "diverge, get 1-2 more. Wider k_range warrants more proposals."
)


def _proposer_override_block(variant: ProposerVariant) -> str:
    """The override paragraph appended to the orchestrator prompt.

    Differs from the SKILL.md "Number of proposals" rule by exactly the count
    phrase (every other word preserved). We wrap it in a one-line lead-in so
    the orchestrator unambiguously prefers this over the SKILL.md text.
    """
    rule = _PROPOSER_RULE_TEMPLATE.format(count=variant.count_phrase)
    baseline = _PROPOSER_RULE_TEMPLATE.format(count="2-3")
    return (
        "\n"
        "PROPOSER-COUNT OVERRIDE FOR THIS RUN. The 'Number of proposals' rule "
        "in cluster-run/SKILL.md is replaced for this session by the following "
        "(differs from SKILL.md only by the leading count phrase):\n"
        "\n"
        f"{rule}\n"
        "\n"
        f"(For reference, the SKILL.md baseline phrase this overrides is: "
        f"'{baseline}')\n"
    )


def _orchestrator_prompt_with_override(
    *,
    workspace_dir: Path,
    dataset: str,
    k_min: int,
    k_max: int,
    allow_none: bool,
    variant: ProposerVariant,
) -> str:
    """Production orchestrator prompt + the proposer-count override block.

    Uses the unmodified _orchestrator_prompt() so the rest of the harness
    (allow_none clause, finalize-only constraint, k-range clause, no-classify
    expectation) stays identical to the production run; only the proposer
    count guidance differs.
    """
    base = _orchestrator_prompt(
        workspace_dir=workspace_dir,
        dataset=dataset,
        k_min=k_min,
        k_max=k_max,
        allow_none=allow_none,
    )
    return base + _proposer_override_block(variant)


def _run_orchestrator(
    *,
    workspace_dir: Path,
    dataset: str,
    k_min: int,
    k_max: int,
    allow_none: bool,
    variant: ProposerVariant,
) -> dict:
    prompt = _orchestrator_prompt_with_override(
        workspace_dir=workspace_dir,
        dataset=dataset,
        k_min=k_min,
        k_max=k_max,
        allow_none=allow_none,
        variant=variant,
    )
    os.environ["CLUSTERING_WORKSPACE"] = str(workspace_dir)
    (workspace_dir / "orchestrator_prompt.txt").write_text(prompt, encoding="utf-8")

    # No classify step in this sweep, so the text-classification plugin isn't
    # needed in the headless session. Loading only agentic-clustering keeps
    # the session minimal.
    extra_args = [
        "--plugin-dir", str(PLUGIN_ROOT),
        "--permission-mode", "bypassPermissions",
    ]
    # See agentic_ablations.py: load_secrets_into_env() (used by the classify
    # step) injects ANTHROPIC_API_KEY which then routes claude -p to metered
    # billing. We never call classify here, but strip the var defensively in
    # case something earlier in the process set it.
    os.environ.pop("ANTHROPIC_API_KEY", None)

    log_prefix = f"[proposers-{variant.name}/{dataset}]"
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
    variant: ProposerVariant,
    k_in_scope: int,
    k_range: tuple[int, int],
    allow_none: bool,
    n_docs: int,
    wall_clock_s: float,
    final_taxonomy: dict,
) -> None:
    """Compact JSON summary at the workspace root.

    The comparison script reads this first (plus log.jsonl and the final audit
    / critique files) to populate its table. Kept minimal — anything richer
    lives in state.json / the audit/critique JSONs and is fetched on demand.
    """
    summary = {
        "method": method_for(variant),
        "variant": variant.name,
        "variant_count_phrase": variant.count_phrase,
        "variant_description": variant.description,
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
    (workspace_dir / "_proposer_sweep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_proposer_sweep(
    dataset_name: str,
    *,
    variant_name: str,
    seed: int = 0,
) -> dict:
    """Run one (dataset, variant) cell of the sweep — taxonomy stage only.

    Stops after cluster-finalize writes final_taxonomy.json + taxonomy.md +
    categories.json. No classify call, no predictions, no metrics.to_dict().
    The downstream comparison script consumes the workspace directly.
    """
    if dataset_name not in DATASET_LENS:
        raise KeyError(f"no DATASET_LENS entry for {dataset_name!r}")
    if variant_name not in VARIANTS:
        raise KeyError(
            f"unknown variant {variant_name!r}; expected one of "
            f"{sorted(VARIANTS)}"
        )
    variant = VARIANTS[variant_name]
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

    workspace_dir = workspace_for(dataset_name, seed=seed, variant=variant)
    # Safety: assert we never collide with the production discover-k workspace
    # or any of the existing ablation workspaces. Catches typos in name/seed
    # before they can clobber a completed expensive run.
    forbidden = [
        RESULTS / "clustering" / dataset_name / f"seed={seed}",
        RESULTS / "clustering" / dataset_name / f"seed={seed}_discoverk",
        RESULTS / "clustering" / dataset_name / f"seed={seed}_discoverk_synthonly",
        RESULTS / "clustering" / dataset_name / f"seed={seed}_discoverk_notask",
    ]
    for f in forbidden:
        if workspace_dir.resolve() == f.resolve():
            raise RuntimeError(
                f"refusing to run: proposer-sweep workspace {workspace_dir} "
                f"collides with existing workspace {f}"
            )

    workspace_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[proposers-{variant.name}/{dataset_name}] init "
        f"(k_range=[{k_min},{k_max}], allow_none={lens.allow_none}, n={n_docs}, "
        f"proposer_count='{variant.count_phrase}')"
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
        f"[proposers-{variant.name}/{dataset_name}] dispatching orchestrator on "
        f"{ORCHESTRATOR_MODEL}"
    )
    orch = _run_orchestrator(
        workspace_dir=workspace_dir,
        dataset=dataset_name,
        k_min=k_min,
        k_max=k_max,
        allow_none=lens.allow_none,
        variant=variant,
    )
    print(
        f"[proposers-{variant.name}/{dataset_name}] orchestrator returned in "
        f"{orch['wall_clock_s']:.1f}s"
    )

    # Same finalize-output contract as the production run: taxonomy.md +
    # final_taxonomy.json + categories.json must all be present.
    _ensure_orchestrator_outputs(workspace_dir)

    final_taxonomy = _read_final_taxonomy(workspace_dir)
    wall_clock_s = time.perf_counter() - t_start

    _write_run_summary(
        workspace_dir=workspace_dir,
        dataset=dataset_name,
        variant=variant,
        k_in_scope=k_in_scope,
        k_range=(k_min, k_max),
        allow_none=lens.allow_none,
        n_docs=n_docs,
        wall_clock_s=wall_clock_s,
        final_taxonomy=final_taxonomy,
    )

    k_actual = len(final_taxonomy["clusters"])
    print(
        f"[proposers-{variant.name}/{dataset_name}] done. k_actual={k_actual} "
        f"(k_in_scope={k_in_scope}, range=[{k_min},{k_max}]) "
        f"wall_clock={wall_clock_s:.1f}s"
    )
    return {
        "method": method_for(variant),
        "variant": variant.name,
        "dataset": dataset_name,
        "n_docs": n_docs,
        "k_in_scope": k_in_scope,
        "k_range": [k_min, k_max],
        "k_actual": k_actual,
        "wall_clock_s": wall_clock_s,
        "workspace": str(workspace_dir),
    }
