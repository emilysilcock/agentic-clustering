"""Sweep over how much text sample the proposers share with each other.

Four variants, isolated from production: ``v0`` is fully disjoint sampling (the
production default — proposers each pull a fresh, non-overlapping sample, so
this is a clean baseline rerun under the current plugin code), then ``v1`` /
``v2`` / ``v3`` make each proposer share 25% / 50% / 100% of its sample with
every other proposer.

How the overlap is produced (additive directive, not a plugin edit): the
permanent ``sample.py --save-as``/``--load`` primitive lets a single shared core
be drawn once and re-fetched by every proposer. This sweep appends a directive
to the orchestrator prompt instructing it to draw a shared core of ``round(p*S)``
texts once and have each proposer ``--load`` that core then top up with a fresh
disjoint draw of ``S - round(p*S)``. The plugin has no opinion on sample overlap
(disjointness is an emergent property of seen-tracking, not a SKILL.md rule), so
this directive is purely additive — it does not contradict any plugin text. The
per-proposer sample size ``S`` stays an agentic decision; only the shared
*fraction* is fixed per variant.

Like the proposer-count sweep, all variants run in **discover-k** mode
(``k_in_scope ± 20%``) and **skip classification entirely**. We don't compute
partition metrics here (no predictions). Unlike that sweep, cross-proposal ARI
*is* a valid signal for v1/v2/v3: with a shared core, ``confusion.py`` scores
agreement over the texts proposers actually have in common. v0 has no shared
texts, so its cross-proposal ARI is undefined (as in every prior disjoint run).

Workspaces:
    method:    agentic_clustering_overlap_v{0,1,2,3}
    workspace: clustering/<ds>/seed=<n>_overlap_v{0,1,2,3}/   (NEW)
    source:    clustering/<ds>/seed=<n>_discoverk/             (READ-ONLY,
                                                                untouched)

This module follows the isolation rules from agentic_ablations.py /
agentic_proposer_sweep.py: every helper imported from ``agentic_clustering`` is
read-only; the production module is never edited; new method + new workspace
names guarantee the existing artifacts cannot be overwritten.
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
class OverlapVariant:
    name: str  # workspace + method suffix, e.g. "v1"
    fraction: float  # shared fraction of each proposer's sample (0.0 .. 1.0)
    description: str  # human label for logs / comparison table


# v0 is fully disjoint — the production default, run fresh under current plugin
# code so the comparison isn't muddled by changes since the existing
# seed=0_discoverk runs were produced.
VARIANTS: dict[str, OverlapVariant] = {
    "v0": OverlapVariant("v0", 0.0, "disjoint (production default, baseline rerun)"),
    "v1": OverlapVariant("v1", 0.25, "25% shared core"),
    "v2": OverlapVariant("v2", 0.50, "50% shared core"),
    "v3": OverlapVariant("v3", 1.0, "100% shared (every proposer the same sample)"),
}


def method_for(variant: OverlapVariant) -> str:
    return f"agentic_clustering_overlap_{variant.name}"


def workspace_for(dataset: str, *, seed: int, variant: OverlapVariant) -> Path:
    return RESULTS / "clustering" / dataset / f"seed={seed}_overlap_{variant.name}"


def _overlap_override_block(variant: OverlapVariant) -> str:
    """Additive directive appended to the orchestrator prompt.

    v0 (fraction 0.0) appends nothing — the plugin's default sampling is already
    fully disjoint, so the baseline rerun uses the unmodified prompt. For p>0 we
    spell out the shared-core protocol using the permanent --save-as/--load
    primitive. The directive is additive: the plugin has no sample-overlap rule
    to contradict.
    """
    p = variant.fraction
    if p <= 0.0:
        return ""

    pct = round(p * 100)
    save = "$CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/sample.py"
    if p >= 1.0:
        proposer_steps = (
            f"  uv run {save} --load proposer_core   # the full shared sample; "
            "no fresh top-up\n"
            "and have it cluster exactly that shared set. Every proposer clusters "
            "the identical sample."
        )
        share_clause = (
            "C = S (the entire per-proposer sample is the shared core; there is "
            "no fresh top-up)"
        )
        remainder_clause = ", i.e. every proposer clusters the identical sample"
    else:
        proposer_steps = (
            f"  uv run {save} --load proposer_core            # the C shared texts\n"
            f"  uv run {save} --strategy random --n <S - C>   # fresh, disjoint top-up\n"
            "and have it cluster the combined set of S texts."
        )
        share_clause = f"C = round({p} * S)"
        remainder_clause = (
            "; the remainder is unique to each proposer (the fresh draws are "
            "disjoint by seen-tracking)"
        )

    return (
        "\n"
        f"PROPOSER SAMPLE OVERLAP DIRECTIVE FOR THIS RUN (target {pct}% overlap). "
        "By default each proposer samples a fresh, disjoint set of texts. For "
        "this run, proposers must SHARE part of their sample, as follows:\n"
        "\n"
        "1. Decide the per-proposer sample size S as you normally would (per the "
        "chars-per-text guidance in the cluster-run skill).\n"
        f"2. Compute the shared-core size {share_clause}.\n"
        "3. BEFORE dispatching any proposers, draw the shared core ONCE and save "
        "it under a name:\n"
        f"  uv run {save} --strategy random --n <C> --save-as proposer_core\n"
        "4. Give EVERY proposer the same sampling instruction in its task "
        "description:\n"
        f"{proposer_steps}\n"
        "\n"
        "Draw the shared core only ONCE — do NOT re-run --save-as proposer_core. "
        f"This makes ~{pct}% of every proposer's sample shared with every other "
        f"proposer's{remainder_clause}. Everything else about the run is "
        "unchanged.\n"
    )


def _orchestrator_prompt_with_override(
    *,
    workspace_dir: Path,
    dataset: str,
    k_min: int,
    k_max: int,
    allow_none: bool,
    variant: OverlapVariant,
) -> str:
    """Production orchestrator prompt + the overlap directive block.

    Uses the unmodified _orchestrator_prompt() so the rest of the harness
    (allow_none clause, finalize-only constraint, k-range clause, no-classify
    expectation) stays identical to the production run; only the sample-overlap
    directive is appended.
    """
    base = _orchestrator_prompt(
        workspace_dir=workspace_dir,
        dataset=dataset,
        k_min=k_min,
        k_max=k_max,
        allow_none=allow_none,
    )
    return base + _overlap_override_block(variant)


def _run_orchestrator(
    *,
    workspace_dir: Path,
    dataset: str,
    k_min: int,
    k_max: int,
    allow_none: bool,
    variant: OverlapVariant,
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
    # needed in the headless session. Loading only agentic-clustering keeps the
    # session minimal.
    extra_args = [
        "--plugin-dir", str(PLUGIN_ROOT),
        "--permission-mode", "bypassPermissions",
    ]
    # See agentic_proposer_sweep.py: strip ANTHROPIC_API_KEY defensively so
    # claude -p stays on the Max subscription rather than metered billing.
    os.environ.pop("ANTHROPIC_API_KEY", None)

    log_prefix = f"[overlap-{variant.name}/{dataset}]"
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
    variant: OverlapVariant,
    k_in_scope: int,
    k_range: tuple[int, int],
    allow_none: bool,
    n_docs: int,
    wall_clock_s: float,
    final_taxonomy: dict,
) -> None:
    """Compact JSON summary at the workspace root.

    The comparison script reads this first (plus log.jsonl, the proposals, and
    the final audit / critique / cross-proposal-metrics files) to populate its
    table. Kept minimal — anything richer is fetched on demand.
    """
    summary = {
        "method": method_for(variant),
        "variant": variant.name,
        "overlap_fraction": variant.fraction,
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
    (workspace_dir / "_overlap_sweep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_overlap_sweep(
    dataset_name: str,
    *,
    variant_name: str,
    seed: int = 0,
) -> dict:
    """Run one (dataset, variant) cell of the sweep — taxonomy stage only.

    Stops after cluster-finalize writes final_taxonomy.json + taxonomy.md +
    categories.json. No classify call, no predictions. The downstream comparison
    script consumes the workspace directly.
    """
    if dataset_name not in DATASET_LENS:
        raise KeyError(f"no DATASET_LENS entry for {dataset_name!r}")
    if variant_name not in VARIANTS:
        raise KeyError(
            f"unknown variant {variant_name!r}; expected one of {sorted(VARIANTS)}"
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
    # Safety: never collide with the production discover-k workspace or any of
    # the existing ablation / sweep workspaces. Catches typos before they can
    # clobber a completed expensive run.
    forbidden = [
        RESULTS / "clustering" / dataset_name / f"seed={seed}",
        RESULTS / "clustering" / dataset_name / f"seed={seed}_discoverk",
        RESULTS / "clustering" / dataset_name / f"seed={seed}_discoverk_synthonly",
        RESULTS / "clustering" / dataset_name / f"seed={seed}_discoverk_notask",
    ]
    for f in forbidden:
        if workspace_dir.resolve() == f.resolve():
            raise RuntimeError(
                f"refusing to run: overlap-sweep workspace {workspace_dir} "
                f"collides with existing workspace {f}"
            )

    workspace_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[overlap-{variant.name}/{dataset_name}] init "
        f"(k_range=[{k_min},{k_max}], allow_none={lens.allow_none}, n={n_docs}, "
        f"overlap={variant.fraction:.0%})"
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
        f"[overlap-{variant.name}/{dataset_name}] dispatching orchestrator on "
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
        f"[overlap-{variant.name}/{dataset_name}] orchestrator returned in "
        f"{orch['wall_clock_s']:.1f}s"
    )

    # Same finalize-output contract as the production run.
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
        f"[overlap-{variant.name}/{dataset_name}] done. k_actual={k_actual} "
        f"(k_in_scope={k_in_scope}, range=[{k_min},{k_max}]) "
        f"wall_clock={wall_clock_s:.1f}s"
    )
    return {
        "method": method_for(variant),
        "variant": variant.name,
        "overlap_fraction": variant.fraction,
        "dataset": dataset_name,
        "n_docs": n_docs,
        "k_in_scope": k_in_scope,
        "k_range": [k_min, k_max],
        "k_actual": k_actual,
        "wall_clock_s": wall_clock_s,
        "workspace": str(workspace_dir),
    }
