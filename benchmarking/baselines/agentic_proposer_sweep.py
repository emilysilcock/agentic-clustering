"""Sweep over the orchestrator's initial proposer-count guidance.

Four variants, isolated from production, substituting the count phrase with
"2-3" / "4-5" / "6-7" / "8-9" for ``v0`` / ``v1`` / ``v2`` / ``v3``. Every other
word of SKILL.md's "Number of proposals" paragraph is preserved verbatim — only
the leading count phrase changes.

WHICH VARIANT IS THE SHIPPED DEFAULT MOVED. When this sweep was first written
SKILL.md said "2-3", so ``v0`` was the control. This sweep's own result changed
the rule to "6-7", so **the shipped default is now ``v2``**, and ``v0`` is a
historical setting. Nothing in the
code keys off "which one is the default" — every variant, ``v2`` included, has
its count phrase injected explicitly and never inherits SKILL.md's wording — but
read the variant table below as four parameterised settings, not as "a baseline
plus three bumps".

All four variants run in **discover-k** mode (``k_in_scope ± 20%``, matching the
production discover-k baseline) and **skip the classification step entirely**.
We don't compute partition metrics here: with classification off there are no
predictions, and cross-proposal ARI isn't a valid signal because our proposers
don't share samples by default (each proposer pulls a fresh sample, so any
disagreement between proposers conflates "the model disagrees" with "they saw
different texts"). The comparison script in ``benchmarking/data_processing/compare_proposer_sweep.py``
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
import re
import time
from dataclasses import dataclass
from pathlib import Path

from benchmarking.baselines.agentic_clustering import (
    DISCOVER_K_FRACTION,
    LLM_TOKEN_CAP,
    ORCHESTRATOR_MODEL,
    PLUGIN_ROOT,
    dispatch_orchestrator_with_retry,
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


# Each variant's count phrase is injected into the orchestrator prompt
# explicitly, including v2's — no variant inherits SKILL.md's own wording, so
# the sweep stays correct however the shipped default moves. (It has moved
# once, from "2-3" to "6-7".)
VARIANTS: dict[str, ProposerVariant] = {
    # Descriptions are relative to the SKILL.md guidance at the time of the
    # first sweep, when "2-3" was the shipped rule. That sweep is what moved
    # the rule to "6-7", so v2 is the shipped default and v0 is the historical
    # baseline, not the current one.
    "v0": ProposerVariant("v0", "2-3", "pre-2026-06-09 SKILL.md guidance"),
    "v1": ProposerVariant("v1", "4-5", "modest bump"),
    "v2": ProposerVariant("v2", "6-7", "shipped default since 2026-06-09"),
    "v3": ProposerVariant("v3", "8-9", "aggressive bump"),
}


def method_for(variant: ProposerVariant) -> str:
    return f"agentic_clustering_proposers_{variant.name}"


def workspace_for(dataset: str, *, seed: int, variant: ProposerVariant) -> Path:
    return RESULTS / "clustering" / dataset / f"seed={seed}_proposers_{variant.name}"


# Verbatim copy of plugin/skills/cluster-run/SKILL.md ("Number of
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


_SKILL_COUNT_RE = re.compile(
    r"\*\*Number of proposals\*\*\s*[—-]\s*start with\s+(?P<count>[0-9]+-[0-9]+)\s+from different angles",
)


def _live_skill_count_phrase() -> str:
    """The count phrase currently shipped in cluster-run/SKILL.md.

    Read at run time so the override block always names the real baseline. A
    missing match is fatal rather than defaulted: silently guessing the wrong
    baseline is what this function exists to prevent, and a SKILL.md reword
    should fail loudly here (mirroring the asserted-present needle guards the
    no-task and no-k ablations use).
    """
    skill = PLUGIN_ROOT / "skills" / "cluster-run" / "SKILL.md"
    m = _SKILL_COUNT_RE.search(skill.read_text(encoding="utf-8"))
    if not m:
        raise RuntimeError(
            f"could not find the 'Number of proposals' count phrase in {skill}. "
            "The rule was reworded; update _SKILL_COUNT_RE and "
            "_PROPOSER_RULE_TEMPLATE to match before running the sweep."
        )
    return m["count"]


def _proposer_override_block(variant: ProposerVariant) -> str:
    """The override paragraph appended to the orchestrator prompt.

    Differs from the SKILL.md "Number of proposals" rule by exactly the count
    phrase (every other word preserved). We wrap it in a one-line lead-in so
    the orchestrator unambiguously prefers this over the SKILL.md text.

    A second paragraph pins the cluster-finalize --output path. It is harness
    plumbing, not an experimental parameter: it says where one artifact is
    written and nothing about how to cluster, and it is identical across all
    four variants, so it cannot bias the comparison between them. Note for the
    write-up that banking77's four cells ran BEFORE this paragraph was added;
    the only difference is that they were not told the
    output path.
    """
    rule = _PROPOSER_RULE_TEMPLATE.format(count=variant.count_phrase)
    # Read the baseline count from the LIVE SKILL.md rather than hardcoding it.
    # It was pinned to "2-3", which went stale when this very sweep moved the
    # shipped rule to "6-7". Naming the wrong baseline
    # defeats the point of naming one at all: the block exists so the
    # orchestrator cannot quietly fall back to the skill's own wording, and it
    # was telling the orchestrator to avoid a phrase the skill no longer uses.
    baseline = _PROPOSER_RULE_TEMPLATE.format(count=_live_skill_count_phrase())
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
        "\n"
        "FINALIZE OUTPUT PATH. When you run cluster-finalize, the `finalize`\n"
        "command's --output must be exactly\n"
        "  $CLUSTERING_WORKSPACE/final_taxonomy.json\n"
        "as cluster-finalize/SKILL.md specifies. --output names the JSON export\n"
        "only; taxonomy.md and categories.json go to fixed paths regardless. On\n"
        "2026-09-30 a run passed --output .../taxonomy.md, so the JSON landed\n"
        "under the wrong name, final_taxonomy.json was never created, and the\n"
        "cell failed after three attempts even though the taxonomy itself was\n"
        "complete (69 clusters) -- the re-dispatched sessions correctly declined\n"
        "to redo a workspace that already looked finalized.\n"
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
        # Was MAX_AGENT_DISPATCHES, a constant agentic_clustering removed when
        # the cumulative dispatch cap was dropped (first 8, then 20,
        # then none) so the harness exercises the shipped plugin's own
        # state-grounded stop criteria. None records "no cap" explicitly rather
        # than omitting the field, so earlier capped cells stay distinguishable
        # from these uncapped ones.
        "max_agent_dispatches": None,
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

    # 512-token-capped corpus, same artefact the
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
    # Retry an orchestrator that returns without finalizing, exactly as the
    # main runner and the no-task/no-k ablations do. In headless `claude -p`
    # the session often ends its turn after dispatching a sub-agent ("waiting
    # for notifications") and exits incomplete; it is retryable because the
    # workspace is left fine to run again. Before this the sweep dispatched
    # once and any such turn failed the cell outright -- stackexchange hit
    # exactly that, and over 28 cells it is near-certain.
    #
    # The wrapper also owns the finalize-output contract (taxonomy.md +
    # final_taxonomy.json + categories.json), checking each attempt against the
    # moment of dispatch. That freshness bound is mandatory, not optional:
    # these workspaces held earlier runs, init.py does not clear them, and
    # existence alone satisfies the check -- so without it a re-run that
    # produced nothing would pass on the old final_taxonomy.json and be written
    # up as a fresh cell.
    orch = dispatch_orchestrator_with_retry(
        lambda: _run_orchestrator(
            workspace_dir=workspace_dir,
            dataset=dataset_name,
            k_min=k_min,
            k_max=k_max,
            allow_none=lens.allow_none,
            variant=variant,
        ),
        workspace_dir=workspace_dir,
        label=f"proposers-{variant.name}/{dataset_name}",
    )
    print(
        f"[proposers-{variant.name}/{dataset_name}] orchestrator returned in "
        f"{orch['wall_clock_s']:.1f}s"
    )

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
