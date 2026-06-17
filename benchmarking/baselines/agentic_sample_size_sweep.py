"""Sweep over the orchestrator's proposer sample-size guidance.

Companion to ``agentic_proposer_sweep.py``: instead of varying the count of
proposers, we hold that fixed (SKILL.md's current "6-7" guidance, inherited as
the unchanged baseline) and vary the *size* of each proposer's text sample. The
sliding scale stays — short / medium / long text buckets — but each band is
multiplied by a factor.

Variants:

==========  ========  ==================  ===============  =============
variant     factor    short (<100 chars)  medium (100-500) long (500+)
==========  ========  ==================  ===============  =============
v1          0.5x      100-200             25-75            10-25
v2          2.0x      400-800             100-300          40-100
==========  ========  ==================  ===============  =============

The 1.0x "baseline" cell isn't re-run here — it's equivalent to the
``seed=0_proposers_v2`` workspaces already produced by ``agentic_proposer_sweep``
(those used current-SKILL.md sample sizes with the now-default 6-7 proposers).
``v0`` (0.25x) and ``v3`` (4.0x) names are reserved for a follow-up sweep but
not implemented here — they only ship if v1/v2 results warrant exploring the
tails.

All cells run in **discover-k** mode (``k_in_scope ± 20%``, identical to the
production discover-k baseline) and **skip classification entirely**, same as
the proposer sweep — partition metrics aren't valid (proposers don't share
samples) and would require a metered API budget we don't need to spend.

Workspaces:
    method:    agentic_clustering_sample_size_v{1,2}
    workspace: clustering/<ds>/seed=<n>_sample_size_v{1,2}/   (NEW)
    source:    clustering/<ds>/seed=<n>_discoverk/            (READ-ONLY,
                                                               untouched)

Isolation rules (per agentic_ablations.py / agentic_proposer_sweep.py): every
helper imported from ``agentic_clustering`` is read-only; the production module
is never edited; new method + new workspace names guarantee none of the eight
existing seed=0_* workspaces (seed=0, seed=0_discoverk,
seed=0_discoverk_notask, seed=0_discoverk_synthonly, seed=0_proposers_v0..v3)
can be overwritten.
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

# Smallest-up sweep order — identical to run_agentic_clustering.SWEEP_ORDER and
# agentic_proposer_sweep.SWEEP_ORDER so the Max subscription's rolling cap
# drains predictably from the cheapest run up.
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
class SampleSizeVariant:
    name: str  # workspace + method suffix, e.g. "v1"
    factor: float  # multiplier vs SKILL.md baseline (1.0x)
    short_band: tuple[int, int]  # (lo, hi) for <100-char texts
    medium_band: tuple[int, int]  # (lo, hi) for 100-500-char texts
    long_band: tuple[int, int]  # (lo, hi) for 500+-char texts
    description: str  # human label for logs / comparison table


# Bands are agreed verbatim with Emily (2026-06-09). If we later add v0 (0.25x)
# and v3 (4.0x), the rounded bands would be (50-100, 12-40, 5-15) and
# (800-1600, 200-600, 80-200) respectively — kept here as a comment so the
# follow-up cells stay consistent with what was sketched at proposal time.
VARIANTS: dict[str, SampleSizeVariant] = {
    "v1": SampleSizeVariant(
        name="v1",
        factor=0.5,
        short_band=(100, 200),
        medium_band=(25, 75),
        long_band=(10, 25),
        description="0.5x (halved sample bands)",
    ),
    "v2": SampleSizeVariant(
        name="v2",
        factor=2.0,
        short_band=(400, 800),
        medium_band=(100, 300),
        long_band=(40, 100),
        description="2.0x (doubled sample bands)",
    ),
}


def method_for(variant: SampleSizeVariant) -> str:
    return f"agentic_clustering_sample_size_{variant.name}"


def workspace_for(dataset: str, *, seed: int, variant: SampleSizeVariant) -> Path:
    return (
        RESULTS
        / "clustering"
        / dataset
        / f"seed={seed}_sample_size_{variant.name}"
    )


# Verbatim snapshot of plugin/skills/cluster-run/SKILL.md L138-146 ("Sample
# sizes" paragraph) parameterised on the three numeric bands. Every other word
# must match the source — if SKILL.md changes, this template must be updated to
# match, otherwise the override silently injects stale wording. The six band
# tokens "{a}".."{f}" are the *only* placeholders.
_SAMPLE_SIZES_RULE_TEMPLATE = (
    "**Sample sizes** — think about what fits in an agent's context window "
    "(~50K tokens of working space). Rough guide:\n"
    "- Texts < 100 chars avg → agents can handle {a}-{b} texts comfortably\n"
    "- Texts 100-500 chars → {c}-{d} texts\n"
    "- Texts 500+ chars → {e}-{f} texts\n"
    "\n"
    "Scale to corpus size — don't sample 200 from a corpus of 300."
)


def _sample_size_override_block(variant: SampleSizeVariant) -> str:
    """The override paragraph appended to the orchestrator prompt.

    Differs from the SKILL.md "Sample sizes" rule by exactly the three numeric
    bands (every other word preserved). We wrap it in a one-line lead-in so the
    orchestrator unambiguously prefers this over the SKILL.md text, and add an
    explicit "audit unchanged" clarification — without that, SKILL.md's
    "**Audit sample size** — same chars-per-text guidance as proposals" would
    silently inherit the multiplier and contaminate the signal.
    """
    rule = _SAMPLE_SIZES_RULE_TEMPLATE.format(
        a=variant.short_band[0], b=variant.short_band[1],
        c=variant.medium_band[0], d=variant.medium_band[1],
        e=variant.long_band[0], f=variant.long_band[1],
    )
    baseline = _SAMPLE_SIZES_RULE_TEMPLATE.format(
        a=200, b=400, c=50, d=150, e=20, f=50,
    )
    return (
        "\n"
        f"SAMPLE-SIZE OVERRIDE FOR THIS RUN. The 'Sample sizes' rule in "
        f"cluster-run/SKILL.md is replaced for this session by the following "
        f"(differs from SKILL.md only by the three numeric bands — multiplier "
        f"{variant.factor}x applied to each):\n"
        "\n"
        f"{rule}\n"
        "\n"
        "This override applies to PROPOSER sampling only. The 'Audit sample "
        "size' rule (200-400 / 50-150 / 20-50 for short / medium / long texts) "
        "is unchanged for this run — keep audits at the SKILL.md baseline.\n"
        "\n"
        f"(For reference, the SKILL.md baseline 'Sample sizes' rule this "
        f"overrides is:\n{baseline})\n"
    )


def _orchestrator_prompt_with_override(
    *,
    workspace_dir: Path,
    dataset: str,
    k_min: int,
    k_max: int,
    allow_none: bool,
    variant: SampleSizeVariant,
) -> str:
    """Production orchestrator prompt + the sample-size override block.

    Uses the unmodified _orchestrator_prompt() so the rest of the harness
    (allow_none clause, finalize-only constraint, k-range clause, no-classify
    expectation) stays identical to the production run; only the proposer
    sample-size guidance differs. The current SKILL.md "Number of proposals"
    default of 6-7 is inherited unchanged — this sweep deliberately doesn't
    re-assert it, relying on SKILL.md as the source of truth.
    """
    base = _orchestrator_prompt(
        workspace_dir=workspace_dir,
        dataset=dataset,
        k_min=k_min,
        k_max=k_max,
        allow_none=allow_none,
    )
    return base + _sample_size_override_block(variant)


def _run_orchestrator(
    *,
    workspace_dir: Path,
    dataset: str,
    k_min: int,
    k_max: int,
    allow_none: bool,
    variant: SampleSizeVariant,
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

    log_prefix = f"[sample_size-{variant.name}/{dataset}]"
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
    variant: SampleSizeVariant,
    k_in_scope: int,
    k_range: tuple[int, int],
    allow_none: bool,
    n_docs: int,
    wall_clock_s: float,
    final_taxonomy: dict,
) -> None:
    """Compact JSON summary at the workspace root.

    Mirrors agentic_proposer_sweep._write_run_summary so a single comparison
    script can consume both sweeps' workspaces.
    """
    summary = {
        "method": method_for(variant),
        "variant": variant.name,
        "variant_factor": variant.factor,
        "variant_short_band": list(variant.short_band),
        "variant_medium_band": list(variant.medium_band),
        "variant_long_band": list(variant.long_band),
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
    (workspace_dir / "_sample_size_sweep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# Every existing seed=0_* workspace flavor — the script refuses to launch if
# its target path resolves into one of these. Cheap insurance against a typo
# in variant name or seed value clobbering a completed expensive run.
def _forbidden_paths(dataset: str, seed: int) -> list[Path]:
    base = RESULTS / "clustering" / dataset
    return [
        base / f"seed={seed}",
        base / f"seed={seed}_discoverk",
        base / f"seed={seed}_discoverk_synthonly",
        base / f"seed={seed}_discoverk_notask",
        base / f"seed={seed}_proposers_v0",
        base / f"seed={seed}_proposers_v1",
        base / f"seed={seed}_proposers_v2",
        base / f"seed={seed}_proposers_v3",
    ]


def run_sample_size_sweep(
    dataset_name: str,
    *,
    variant_name: str,
    seed: int = 0,
) -> dict:
    """Run one (dataset, variant) cell of the sweep — taxonomy stage only.

    Stops after cluster-finalize writes final_taxonomy.json + taxonomy.md +
    categories.json. No classify call, no predictions, no metrics.to_dict().
    The comparison script consumes the workspace directly.
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

    # Refuse to start if the target collides with any existing seed=0_*
    # workspace (catches typos before they can clobber a completed run) OR if
    # the target itself already has finalize artefacts (catches re-runs of an
    # already-completed cell of this sweep).
    for f in _forbidden_paths(dataset_name, seed):
        if workspace_dir.resolve() == f.resolve():
            raise RuntimeError(
                f"refusing to run: sample-size workspace {workspace_dir} "
                f"collides with existing workspace {f}"
            )
    if (workspace_dir / "final_taxonomy.json").exists():
        raise RuntimeError(
            f"refusing to run: {workspace_dir} already contains "
            f"final_taxonomy.json — this cell has already completed. Delete "
            f"or move the workspace if you want to re-run."
        )

    workspace_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[sample_size-{variant.name}/{dataset_name}] init "
        f"(k_range=[{k_min},{k_max}], allow_none={lens.allow_none}, n={n_docs}, "
        f"factor={variant.factor}x, "
        f"bands=short{variant.short_band}/med{variant.medium_band}/long{variant.long_band})"
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
        f"[sample_size-{variant.name}/{dataset_name}] dispatching orchestrator "
        f"on {ORCHESTRATOR_MODEL}"
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
        f"[sample_size-{variant.name}/{dataset_name}] orchestrator returned in "
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
        f"[sample_size-{variant.name}/{dataset_name}] done. k_actual={k_actual} "
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
