"""Ablations on our method, built on top of the discover-k runs.

Both ablations live here, fully isolated: they *import* the production helpers
from ``agentic_clustering`` read-only and never modify that module. Every
output goes to a brand-new method name and a brand-new workspace directory, so
the already-completed main runs (``agentic_clustering`` /
``agentic_clustering_discoverk`` predictions, ``seed=0`` / ``seed=0_discoverk``
workspaces) are never touched or re-executed.

Per the 2026-05-25 ablation decision, both run from the **discover-k** config.

Ablation 1 — synth-only ("no auditor / critic / investigator"):
    Take the *first* taxonomy the synthesizer produced (recovered from the
    existing discover-k workspace's archive), rebuild the classification prompt
    from it, and re-run only the cheap gpt-5-mini classification pass. No
    frontier calls. Isolates the contribution of the whole post-synthesis
    refinement stage.
      method:    agentic_clustering_synthonly_discoverk
      workspace: clustering/<ds>/seed=<n>_discoverk_synthonly/   (NEW)
      source:    clustering/<ds>/seed=<n>_discoverk/             (READ-ONLY)

    The first-synthesis file for each of the 7 datasets is hardcoded in
    FIRST_SYNTH_DISCOVERK below. Each was verified (2026-05-25) against the
    run's archive/log.jsonl: its cluster count equals the first
    `set-clusters (version 1)` commit and it was written just before that
    commit — so it is the synthesizer's first output, not a single proposer's
    and not a later re-synthesis. (goemotions saved it under the non-obvious
    name cluster_set_tmp.json.)

Ablation 2 — no-task ("blank user instructions"):
    A full end-to-end discover-k run with ``config.instructions = ""`` (the
    plugin's documented "discover from data alone" mode, cluster-run/SKILL.md
    L117-118) and the dataset identity stripped from the harness driver prompt.
    Everything else (allow_none, the +-20% k-range) is held identical to the
    main discover-k run, so the only changed variable is the task description.
      method:    agentic_clustering_notask_discoverk
      workspace: clustering/<ds>/seed=<n>_discoverk_notask/      (NEW)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

# Read-only imports of the production helpers. This module never edits
# agentic_clustering.py; it only calls into it.
from benchmarking.baselines.agentic_clustering import (
    BUILD_PROMPT_SCRIPT,
    CLASSIFY_MODEL,
    DISCOVER_K_FRACTION,
    LLM_TOKEN_CAP,
    ORCHESTRATOR_MODEL,
    PLUGIN_ROOT,
    PRICING_BASIS,
    SUBSCRIPTION_USD_PER_DATASET,
    _build_predictions,
    _build_taxonomy_entries,
    _classify_cost_usd,
    _ensure_orchestrator_outputs,
    _init_workspace,
    _materialize_capped_corpus,
    _orchestrator_prompt,
    _read_classify_csv,
    _read_final_taxonomy,
    _run_classify,
    _run_uv_script,
    _taxonomy_str_to_int_id,
)
from benchmarking.data_processing.load import load_processed
from benchmarking.dataset_lens import DATASET_LENS
from benchmarking.evaluation.cost import CostAccumulator
from benchmarking.evaluation.metrics import compute_partition_metrics
from benchmarking.evaluation.persistence import write_run_artifacts
from benchmarking.llm_clients.claude_code import call_claude
from benchmarking.paths import RESULTS

METHOD_SYNTHONLY = "agentic_clustering_synthonly_discoverk"
METHOD_NOTASK = "agentic_clustering_notask_discoverk"

# Smallest-up, Banking77 first (mirrors run_agentic_clustering.SWEEP_ORDER).
SWEEP_ORDER = [
    "banking77",
    "massive_intent",
    "massive_domain",
    "stackexchange",
    "clinc150",
    "twenty_newsgroups",
    "goemotions",
]


# --------------------------------------------------------------------------- #
# Ablation 1: synth-only
# --------------------------------------------------------------------------- #

# Verified first-synthesis taxonomy per discover-k workspace, hardcoded after
# confirming each against archive/log.jsonl (see module docstring). Value is
# (path relative to seed=0_discoverk/, expected version-1 cluster count). The
# count is re-checked at run time in run_synthonly and aborts loudly on any
# mismatch, so a moved/regenerated workspace can't silently yield a wrong file.
FIRST_SYNTH_DISCOVERK: dict[str, tuple[str, int]] = {
    "banking77":         ("archive/investigations/synthesis_clusters_20260523_133400_390c.json", 86),
    "massive_intent":    ("archive/investigations/synthesis_20260523_180956_3caec723.json", 72),
    "massive_domain":    ("archive/investigations/synthesis_clusters_20260523_144047_e7a1.json", 20),
    "stackexchange":     ("archive/investigations/synthesized_20260523_150419.json", 131),
    "clinc150":          ("archive/investigations/synthesized_clusters_20260523_200126.json", 169),
    "twenty_newsgroups": ("archive/investigations/synth_clusters_2b94.json", 19),
    "goemotions":        ("archive/investigations/cluster_set_tmp.json", 27),
}


def _load_synth_clusters(synth_path: Path) -> list[dict]:
    """Normalise a synth cluster-list to [{id, name, description}], assigning
    ids c1..cN in file order (the synth files predate the c<N> id scheme)."""
    raw = json.loads(synth_path.read_text(encoding="utf-8"))["clusters"]
    out: list[dict] = []
    for i, c in enumerate(raw, start=1):
        out.append(
            {
                "id": f"c{i}",
                "name": str(c["name"]),
                "description": " ".join(str(c.get("description", "")).split()),
            }
        )
    return out


def _write_synth_taxonomy_md(clusters: list[dict], out_path: Path, source_synth: Path) -> None:
    """Write a taxonomy.md that build_classification_prompt.py can consume.

    classify.py reads cluster ids from the first backtick token of each ``## ``
    header (``extract_cluster_ids``), so each cluster header is
    ``## <name> (`<id>`)``. The metadata header before the first ``## `` is
    stripped by the prompt builder, so it's purely informational here.
    """
    lines = [
        "# Cluster Taxonomy (synth-only ablation)",
        "",
        f"**Source synth**: {source_synth.name}",
        f"**Clusters**: {len(clusters)}",
        "",
        "---",
        "",
    ]
    for c in clusters:
        lines.append(f"## {c['name']} (`{c['id']}`)")
        lines.append("")
        lines.append(c["description"])
        lines.append("")
        lines.append("---")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_synthonly(dataset_name: str, *, seed: int = 0, reuse_existing_classify: bool = False) -> dict:
    """Ablation 1 on the discover-k run: re-classify against the first synth.

    ``reuse_existing_classify`` skips the classify call and assembles the
    artifact from the ``seed_0.csv`` already in the workspace. Use after
    repairing a partially-failed classify CSV (see
    ``scripts/retry_classify_errors.py``) so the clean artifact is built without
    re-running the whole corpus.
    """
    if dataset_name not in DATASET_LENS:
        raise KeyError(f"no DATASET_LENS entry for {dataset_name!r}")
    lens = DATASET_LENS[dataset_name]
    ds = load_processed(dataset_name)
    k_in_scope = int(ds.meta["k_in_scope"])
    # 512-token-capped corpus (SPEC §5.1.1 / §5.6.3), same as the production
    # runs — so the synth-only classify pass sees the same truncated bodies.
    documents_path = _materialize_capped_corpus(dataset_name, ds)

    src_ws = RESULTS / "clustering" / dataset_name / f"seed={seed}_discoverk"
    if not src_ws.exists():
        raise FileNotFoundError(
            f"discover-k workspace not found at {src_ws}; nothing to recover from."
        )
    if dataset_name not in FIRST_SYNTH_DISCOVERK:
        raise KeyError(f"no verified first-synth file recorded for {dataset_name!r}")
    rel_path, expected_n = FIRST_SYNTH_DISCOVERK[dataset_name]
    synth_path = src_ws / rel_path
    if not synth_path.exists():
        raise FileNotFoundError(f"verified first-synth file missing: {synth_path}")

    out_ws = RESULTS / "clustering" / dataset_name / f"seed={seed}_discoverk_synthonly"
    # Safety: never write into the source (read-only) workspace.
    assert out_ws.resolve() != src_ws.resolve(), "ablation workspace collides with source"
    (out_ws / "classification").mkdir(parents=True, exist_ok=True)

    clusters = _load_synth_clusters(synth_path)
    if len(clusters) != expected_n:
        raise RuntimeError(
            f"[{dataset_name}] recovered {len(clusters)} clusters from {synth_path.name}, "
            f"expected {expected_n} (verified version-1 synth count). Aborting to avoid a "
            f"wrong ablation."
        )
    print(
        f"[synthonly/{dataset_name}] recovered {len(clusters)} synth clusters "
        f"from {synth_path.name}"
    )

    final_taxonomy = {
        "clusters": clusters,
        "cluster_version": 0,
        "source_synth": str(synth_path),
        "note": "synth-only ablation: pre-audit/critic/investigator taxonomy",
    }
    (out_ws / "final_taxonomy.json").write_text(
        json.dumps(final_taxonomy, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    taxonomy_md = out_ws / "taxonomy.md"
    _write_synth_taxonomy_md(clusters, taxonomy_md, synth_path)

    # Build the classification prompt from the synth taxonomy (force-assign
    # matches the main run: on per lens.allow_none).
    prompt_md = out_ws / "classification" / "prompt.md"
    build_args = ["--taxonomy", str(taxonomy_md), "--output", str(prompt_md)]
    if not lens.allow_none:
        build_args.append("--force-assign")
    _run_uv_script(BUILD_PROMPT_SCRIPT, build_args)

    if reuse_existing_classify:
        classify_csv_path = out_ws / "classification" / "classifications" / "seed_0.csv"
        if not classify_csv_path.exists():
            raise FileNotFoundError(
                f"--reuse-existing-classify: no classify CSV at {classify_csv_path}. "
                f"Run without it (or repair the CSV) first."
            )
        print(f"[synthonly/{dataset_name}] reuse: assembling from existing {classify_csv_path.name} (no re-classify)")
        wall_clock_s = 0.0
    else:
        print(
            f"[synthonly/{dataset_name}] classifying {len(ds.documents)} docs on "
            f"{CLASSIFY_MODEL} (force_assign={not lens.allow_none})"
        )
        t0 = time.perf_counter()
        classify_csv_path = _run_classify(
            workspace_dir=out_ws, documents_path=documents_path, allow_none=lens.allow_none
        )
        wall_clock_s = time.perf_counter() - t0

    id_map = {c["id"]: i for i, c in enumerate(clusters)}
    taxonomy_by_str_id = {c["id"]: c for c in clusters}
    classify_rows = _read_classify_csv(classify_csv_path)
    predictions = _build_predictions(
        documents=ds.documents,
        classify_rows=classify_rows,
        id_map=id_map,
        taxonomy_by_str_id=taxonomy_by_str_id,
    )
    taxonomy_entries = _build_taxonomy_entries({"clusters": clusters}, id_map)

    api_usd, in_tokens, out_tokens = _classify_cost_usd(classify_rows)
    cost = CostAccumulator(
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        # Frontier work was proposers + synthesizer only (a subset of the full
        # pipeline). We report the same flat subscription attribution as the
        # main run for comparability; the partial stage is noted in extra_meta.
        subscription_usd=SUBSCRIPTION_USD_PER_DATASET,
        api_usd=api_usd,
        usd=SUBSCRIPTION_USD_PER_DATASET + api_usd,
        wall_clock_s=wall_clock_s,
    )
    metrics = compute_partition_metrics(
        pred_ids=[p.predicted_cluster_id for p in predictions],
        gold_ids=[p.gold_label_id for p in predictions],
    )

    write_run_artifacts(
        method=METHOD_SYNTHONLY,
        dataset=dataset_name,
        seed=seed,
        predictions=predictions,
        taxonomy=taxonomy_entries,
        cost=cost,
        metrics=metrics.to_dict(),
        model_versions={"orchestrator": ORCHESTRATOR_MODEL, "classify": CLASSIFY_MODEL},
        iterations=0,
        hyperparameters={
            "ablation": "synthonly",
            "discover_k": True,
            "k_in_scope": k_in_scope,
            "allow_none": lens.allow_none,
            "classify_force_assign": not lens.allow_none,
            "llm_input_token_cap": LLM_TOKEN_CAP,
            "pricing_basis": PRICING_BASIS,
        },
        extra_meta={
            "ablation": "synthonly",
            "frontier_stage": "proposers+synthesizer only (no audit/critic/investigator)",
            "source_workspace": str(src_ws),
            "source_synth": synth_path.name,
            "n_docs": len(ds.documents),
            "k_actual": len(clusters),
            "classify_csv_path": str(classify_csv_path),
        },
    )
    print(
        f"[synthonly/{dataset_name}] done. k_actual={len(clusters)} "
        f"api_usd=${api_usd:.4f} wall_clock={wall_clock_s:.1f}s "
        f"ARI={metrics.to_dict().get('ari'):.3f}"
    )
    return {
        "method": METHOD_SYNTHONLY,
        "dataset": dataset_name,
        "n_docs": len(ds.documents),
        "k_in_scope": k_in_scope,
        "k_actual": len(clusters),
        "api_usd": api_usd,
        "wall_clock_s": wall_clock_s,
        "source_synth": synth_path.name,
        **metrics.to_dict(),
    }


# --------------------------------------------------------------------------- #
# Ablation 2: no-task
# --------------------------------------------------------------------------- #

def _notask_orchestrator_prompt(*, workspace_dir, dataset, k_min, k_max, allow_none) -> str:
    """Reuse the production orchestrator prompt verbatim, then strip the one
    dataset-identity phrase. We assert the phrase is present so a future change
    to the production prompt fails loudly rather than silently leaking the
    dataset name into a 'no-information' run."""
    prompt = _orchestrator_prompt(
        workspace_dir=workspace_dir,
        dataset=dataset,
        k_min=k_min,
        k_max=k_max,
        allow_none=allow_none,
    )
    needle = f"the {dataset} benchmark dataset"
    if needle not in prompt:
        raise RuntimeError(
            f"no-task neutralization target {needle!r} not found in orchestrator "
            "prompt; refusing to run to avoid silently leaking the dataset name."
        )
    return prompt.replace(needle, "an unlabelled text corpus")


def _run_notask_orchestrator(*, workspace_dir: Path, dataset: str, k_min: int, k_max: int, allow_none: bool) -> dict:
    prompt = _notask_orchestrator_prompt(
        workspace_dir=workspace_dir,
        dataset=dataset,
        k_min=k_min,
        k_max=k_max,
        allow_none=allow_none,
    )
    os.environ["CLUSTERING_WORKSPACE"] = str(workspace_dir)
    (workspace_dir / "orchestrator_prompt.txt").write_text(prompt, encoding="utf-8")
    extra_args = [
        "--plugin-dir", str(PLUGIN_ROOT),
        "--permission-mode", "bypassPermissions",
    ]
    # The orchestrator must run on the Claude Code Max subscription, not metered
    # API (SPEC §5.6.1). _run_classify -> load_secrets_into_env() injects
    # ANTHROPIC_API_KEY into os.environ after the first dataset's classify, and
    # call_claude's `claude -p` subprocess inherits it -> silently routes every
    # subsequent orchestrator to metered billing, which exhausted the API credit
    # balance mid-sweep. Strip it so `claude -p` falls back to the subscription
    # login on every dataset, exactly as the first one already does.
    os.environ.pop("ANTHROPIC_API_KEY", None)
    t0 = time.perf_counter()
    stdout = call_claude(
        prompt,
        model=ORCHESTRATOR_MODEL,
        timeout_s=60 * 60 * 4,
        log_prefix=f"[notask/{dataset}]",
        extra_args=extra_args,
    )
    t1 = time.perf_counter()
    (workspace_dir / "orchestrator_stdout.txt").write_text(stdout or "", encoding="utf-8")
    return {"wall_clock_s": t1 - t0, "stdout": stdout}


def run_notask(dataset_name: str, *, seed: int = 0, resume_classify: bool = False) -> dict:
    """Ablation 2: full discover-k run with blank user instructions.

    ``resume_classify`` skips init + orchestrator and starts from the existing
    ``seed=<n>_discoverk_notask`` workspace (which must already hold
    final_taxonomy.json + classification/prompt.md). Use to re-run only the
    classify pass after a classify-step failure, without re-burning the agent
    loop. Mirrors ``run_agentic_clustering(resume_classify=...)``.
    """
    if dataset_name not in DATASET_LENS:
        raise KeyError(f"no DATASET_LENS entry for {dataset_name!r}")
    lens = DATASET_LENS[dataset_name]
    ds = load_processed(dataset_name)
    k_in_scope = int(ds.meta["k_in_scope"])
    # 512-token-capped corpus (SPEC §5.1.1 / §5.6.3), same as the production
    # runs — feeds both the agent loop (via init) and the classify pass.
    documents_path = _materialize_capped_corpus(dataset_name, ds)

    # Discover-k config, identical to the main discover-k run.
    k_min = round(k_in_scope * (1 - DISCOVER_K_FRACTION))
    k_max = round(k_in_scope * (1 + DISCOVER_K_FRACTION))
    workspace_dir = RESULTS / "clustering" / dataset_name / f"seed={seed}_discoverk_notask"

    # Safety: must not collide with the main discover-k workspace.
    main_ws = RESULTS / "clustering" / dataset_name / f"seed={seed}_discoverk"
    assert workspace_dir.resolve() != main_ws.resolve(), "no-task workspace collides with main run"

    if resume_classify:
        if not workspace_dir.exists():
            raise FileNotFoundError(
                f"--resume-classify: no-task workspace not found at {workspace_dir}. "
                f"Run without --resume-classify first to produce it."
            )
        print(f"[notask/{dataset_name}] resume: skipping init / orchestrator, re-running classify only")
        _ensure_orchestrator_outputs(workspace_dir)
        orch = {"wall_clock_s": None}
        t_start = time.perf_counter()
    else:
        workspace_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[notask/{dataset_name}] init (k_range=[{k_min},{k_max}], "
            f"allow_none={lens.allow_none}, n={len(ds.documents)}, instructions=BLANK)"
        )
        # The ablation: blank user instructions (cluster-run/SKILL.md L117-118 fallback).
        _init_workspace(
            workspace_dir=workspace_dir,
            documents_path=documents_path,
            k_min=k_min,
            k_max=k_max,
            lens_text="",
        )

        t_start = time.perf_counter()
        print(f"[notask/{dataset_name}] dispatching orchestrator on {ORCHESTRATOR_MODEL} (dataset name stripped)")
        orch = _run_notask_orchestrator(
            workspace_dir=workspace_dir,
            dataset=dataset_name,
            k_min=k_min,
            k_max=k_max,
            allow_none=lens.allow_none,
        )
        print(f"[notask/{dataset_name}] orchestrator returned in {orch['wall_clock_s']:.1f}s")
        _ensure_orchestrator_outputs(workspace_dir)

    print(
        f"[notask/{dataset_name}] classifying {len(ds.documents)} docs on "
        f"{CLASSIFY_MODEL} (force_assign={not lens.allow_none})"
    )
    classify_csv_path = _run_classify(
        workspace_dir=workspace_dir, documents_path=documents_path, allow_none=lens.allow_none
    )
    t_end = time.perf_counter()

    final_taxonomy = _read_final_taxonomy(workspace_dir)
    id_map = _taxonomy_str_to_int_id(final_taxonomy)
    taxonomy_by_str_id = {c["id"]: c for c in final_taxonomy["clusters"]}
    classify_rows = _read_classify_csv(classify_csv_path)

    predictions = _build_predictions(
        documents=ds.documents,
        classify_rows=classify_rows,
        id_map=id_map,
        taxonomy_by_str_id=taxonomy_by_str_id,
    )
    taxonomy_entries = _build_taxonomy_entries(final_taxonomy, id_map)

    api_usd, in_tokens, out_tokens = _classify_cost_usd(classify_rows)
    wall_clock_s = t_end - t_start
    cost = CostAccumulator(
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        subscription_usd=SUBSCRIPTION_USD_PER_DATASET,
        api_usd=api_usd,
        usd=SUBSCRIPTION_USD_PER_DATASET + api_usd,
        wall_clock_s=wall_clock_s,
    )
    metrics = compute_partition_metrics(
        pred_ids=[p.predicted_cluster_id for p in predictions],
        gold_ids=[p.gold_label_id for p in predictions],
    )

    write_run_artifacts(
        method=METHOD_NOTASK,
        dataset=dataset_name,
        seed=seed,
        predictions=predictions,
        taxonomy=taxonomy_entries,
        cost=cost,
        metrics=metrics.to_dict(),
        model_versions={"orchestrator": ORCHESTRATOR_MODEL, "classify": CLASSIFY_MODEL},
        iterations=0,
        hyperparameters={
            "ablation": "notask",
            "discover_k": True,
            "k_in_scope": k_in_scope,
            "k_range": [k_min, k_max],
            "allow_none": lens.allow_none,
            "classify_force_assign": not lens.allow_none,
            "lens_text": "",
            "llm_input_token_cap": LLM_TOKEN_CAP,
            "pricing_basis": PRICING_BASIS,
        },
        extra_meta={
            "ablation": "notask",
            "instructions": "",
            "dataset_name_in_driver": "stripped (neutralized to 'an unlabelled text corpus')",
            "n_docs": len(ds.documents),
            "k_actual": len(final_taxonomy["clusters"]),
            "cluster_version_at_finalize": int(final_taxonomy.get("cluster_version", 0)),
            "orchestrator_wall_clock_s": orch["wall_clock_s"],
            "classify_csv_path": str(classify_csv_path),
        },
    )
    print(
        f"[notask/{dataset_name}] done. k_actual={len(final_taxonomy['clusters'])} "
        f"api_usd=${api_usd:.4f} wall_clock={wall_clock_s:.1f}s "
        f"ARI={metrics.to_dict().get('ari'):.3f}"
    )
    return {
        "method": METHOD_NOTASK,
        "dataset": dataset_name,
        "n_docs": len(ds.documents),
        "k_in_scope": k_in_scope,
        "k_actual": len(final_taxonomy["clusters"]),
        "api_usd": api_usd,
        "wall_clock_s": wall_clock_s,
        **metrics.to_dict(),
    }
