"""Run the agentic-clustering plugin as a benchmark method.

Drives the plugin's iterative cluster-discovery workflow headlessly across the
7 processed datasets. For each dataset:

  1. Materialise a 512-token-capped copy of the canonical
     data/derived/<ds>/documents.jsonl (SPEC §5.1.1 / §5.6.3 LLM-input cap —
     the same cl100k_base truncation the ClusterLLM / Huang & He / TopicGPT
     dataset_adapters apply) at data/agentic_clustering/<ds>/documents.jsonl,
     then initialise the plugin workspace via init.py pointing at that capped
     file (no CSV round-trip — init.py reads JSONL natively as of 2026-05-23)
     with the dataset's lens and fixed k=k_in_scope (SPEC §5.5 headline =
     given-k). Both the agent loop and the classify step (4) read the capped
     file, so every LLM that sees a document body sees the same ≤512 tokens
     the baselines do.
  2. Invoke ``claude -p`` on Opus 4.7 (via the Max subscription) with an
     orchestration prompt that runs the cluster-run loop to completion and
     finalises.
  3. Build the classification prompt — with ``--force-assign`` for the 5
     datasets whose gold labels don't include an OOS/none class.
  4. Classify the full corpus via classify.py on gpt-5-mini in async mode
     (concurrency=20) with prompt caching. Switched from Claude Haiku 4.5 on
     2026-05-23 because Haiku 4.5's cache threshold is empirically ~4096
     tokens and our smaller-k taxonomies fell below that, causing 0%
     cache hits on three of seven datasets. OpenAI caches automatically
     at any prompt ≥1024 tokens. See SPEC §5.6.3.
  5. Convert outputs to DocPrediction / TaxonomyEntry records and write
     results/predictions/agentic_clustering/<ds>/seed=<n>.{jsonl,meta.json}.

Cost reporting (SPEC §5.6.3): the agent loop's frontier-tier cost is the
literal Claude Code Max subscription, split flat across the 7 datasets in
the sweep ($14.29 = $100/7 per dataset, recorded as ``subscription_usd``).
Classify spend is metered and recorded separately as ``api_usd``.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Defensive: classify.py output rows include a "reasoning" cell which is
# usually short but unbounded. Raise the field-size cap here too, mirroring
# init.py / classify.py. Capped at 2**31-1 because Windows' C long can't
# hold sys.maxsize.
csv.field_size_limit(2**31 - 1)

from benchmarking.data_processing.base import NONE_LABEL_ID, NONE_LABEL_NAME
from benchmarking.data_processing.load import load_processed
from benchmarking.dataset_lens import DATASET_LENS
from benchmarking.evaluation.cost import CostAccumulator
from benchmarking.evaluation.metrics import compute_partition_metrics
from benchmarking.evaluation.persistence import (
    DocPrediction,
    TaxonomyEntry,
    write_run_artifacts,
)
from benchmarking.llm_clients.claude_code import call_claude
from benchmarking.paths import DATA, DATA_DERIVED, RESULTS
from benchmarking.secrets import load_secrets_into_env

METHOD = "agentic_clustering"

# SPEC §5.6.1: Opus 4.7 for the agent loop, GPT-5-mini for bulk per-doc.
# (Switched the cheap tier from Claude Haiku 4.5 → gpt-5-mini on 2026-05-23:
# Haiku 4.5's empirical cache minimum is ~4096 tokens, so three of our seven
# datasets — 20NG, MASSIVE-Intent, MASSIVE-Domain — had small-k classification
# prompts that didn't cache; gpt-5-mini caches automatically at any prompt
# size ≥1024 tokens, and the A/B test on Banking77 was within noise on
# quality. See SPEC §5.6.3.)
ORCHESTRATOR_MODEL = "claude-opus-4-7"
CLASSIFY_MODEL = "gpt-5-mini"
CLASSIFY_PROVIDER = "openai"
CLASSIFY_MODE = "batch"
CLASSIFY_CONCURRENCY = 20  # only used when CLASSIFY_MODE == "async"

# Flat split of the $100/mo Claude Code Max subscription across the 7
# datasets in the sweep. Reported as ``subscription_usd``; api_usd is
# metered classify spend reported separately.
SUBSCRIPTION_USD_PER_DATASET = 100.0 / 7

# OpenAI gpt-5-mini Batch API pricing (50% off sync rates). Pinned as a paper
# artefact — these are the rates we paid at run time, not whatever pricing is
# current at re-run time.
GPT5_MINI_USD_PER_1M_INPUT = 0.125
GPT5_MINI_USD_PER_1M_CACHE_READ = 0.0125
GPT5_MINI_USD_PER_1M_OUTPUT = 1.00
PRICING_BASIS = "openai_gpt_5_mini_batch_api_50pct_discount_2026_05"

# Plugin lives at <repo>/plugin/ after the 23a2126 marketplace restructure
# (skills/ and agents/ used to be at the repo root). parents[2] is the repo
# root; the /"plugin" segment is what makes the headless claude -p session
# resolve /cluster-run + the subagents and what makes _run_uv_script find
# init.py.
PLUGIN_ROOT = Path(__file__).resolve().parents[2] / "plugin"
SCRIPTS_DIR = PLUGIN_ROOT / "skills" / "corpus-tools" / "scripts"
INIT_SCRIPT = SCRIPTS_DIR / "init.py"

# The classification scripts were split off into a separate plugin in
# commit 36861bb. `build_classification_prompt.py` was dissolved entirely
# (cluster-finalize now writes categories.json directly via state.py:952);
# `classify.py` was moved to the text-classification plugin's classify-tools
# scripts dir. We expect the text-classification repo cloned as a sibling
# of agentic-clustering so the headless session can load both plugins.
TEXT_CLASSIFICATION_ROOT = PLUGIN_ROOT.parents[1] / "text-classification" / "plugin"
CLASSIFY_SCRIPT = TEXT_CLASSIFICATION_ROOT / "skills" / "classify-tools" / "scripts" / "classify.py"

# Agent-dispatch budget for the orchestrator (Proposer + Synthesizer + Auditor
# + Critic + Investigator combined, counted cumulatively across the whole run).
# Matches the cluster-run SKILL.md hard checkpoint — both were raised from 8
# on 2026-06-05 because audits across the 7-dataset sweep showed the Investigator
# was dispatched on only 2 of 7 runs, with the 8-cap leaving no headroom beyond
# the baseline 6 (3 proposer + synth + auditor + critic). At 20, a single
# investigate → re-audit cycle (~2 slots) is cheap and the orchestrator can
# afford ~7 such cycles before hitting the cap.
MAX_AGENT_DISPATCHES = 20

# SPEC §5.1.1 / §5.6.3: every method that feeds a document body to an LLM caps
# it at 512 tiktoken cl100k_base tokens. The ClusterLLM / Huang & He / TopicGPT
# baselines apply this in their dataset_adapters; we mirror it here so the agent
# loop (Proposer/Synthesizer/Auditor/Critic/Investigator) and the gpt-5-mini
# classify step see the same truncated bodies. Without it our method read full
# untruncated docs while the baselines saw only the first 512 tokens — a length
# advantage on long-doc datasets (materially: 20 Newsgroups, ~9% of docs). The
# plugin scripts stay uncapped by default; the cap is a benchmark policy applied
# here, so we keep it out of the shipped tool.
LLM_TOKEN_CAP = 512


def _truncate_to_token_limit(text: str, encoder, limit: int) -> tuple[str, bool]:
    tokens = encoder.encode(text)
    if len(tokens) <= limit:
        return text, False
    return encoder.decode(tokens[:limit]), True


def _materialize_capped_corpus(dataset_name: str, ds, *, force: bool = False) -> Path:
    """Write a 512-token-capped copy of the dataset's documents.jsonl, return its path.

    Schema-identical to data/derived/<ds>/documents.jsonl — every field is
    preserved and only ``text`` is truncated — so init.py / classify.py read it
    with the same --text-col/--id-col. Idempotent: returns the existing file
    unless ``force``. Mirrors the baselines' dataset_adapter cap exactly
    (tiktoken cl100k_base, decode of the first LLM_TOKEN_CAP tokens).
    """
    import tiktoken

    out_dir = DATA / "agentic_clustering" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "documents.jsonl"

    if out_path.exists() and not force:
        return out_path

    encoder = tiktoken.get_encoding("cl100k_base")
    n_truncated = 0
    lines: list[str] = []
    for doc in ds.documents:
        text, truncated = _truncate_to_token_limit(doc["text"], encoder, LLM_TOKEN_CAP)
        if truncated:
            n_truncated += 1
        lines.append(json.dumps({**doc, "text": text}, ensure_ascii=False))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"[agentic/{dataset_name}] capped corpus: {n_truncated} of {len(lines)} "
        f"docs truncated at {LLM_TOKEN_CAP} tokens (cl100k_base) -> {out_path}"
    )
    return out_path


def _uv_env() -> dict[str, str]:
    """uv on Windows fails to TLS-verify unless SSL_CERT_FILE is unset (see
    feedback_uv_tls_workaround). Inherit the parent env, drop that var, and
    let --native-tls do the verification.
    """
    env = os.environ.copy()
    env.pop("SSL_CERT_FILE", None)
    return env


def _run_uv_script(script: Path, args: list[str]) -> None:
    """Invoke a plugin script via `uv run --native-tls` so its inline
    `# /// script` deps (anthropic, openai for classify.py) are honoured."""
    cmd = ["uv", "run", "--native-tls", str(script), *args]
    subprocess.run(cmd, check=True, env=_uv_env())


def _init_workspace(
    *, workspace_dir: Path, documents_path: Path, k_min: int, k_max: int, lens_text: str
) -> None:
    """Point init.py at the canonical documents.jsonl directly — no CSV
    round-trip. Requires init.py to accept --id-col and .jsonl input (added
    on 2026-05-23 alongside dropping _build_corpus_csv)."""
    workspace_dir.mkdir(parents=True, exist_ok=True)
    _run_uv_script(
        INIT_SCRIPT,
        [
            "--corpus", str(documents_path),
            "--text-col", "text",
            "--id-col", "doc_id",
            "--k-range", str(k_min), str(k_max),
            "--model-tier", "quality",
            "--instructions", lens_text,
            "--workspace", str(workspace_dir),
        ],
    )


def _orchestrator_prompt(
    *, workspace_dir: Path, dataset: str, k_min: int, k_max: int, allow_none: bool
) -> str:
    is_fixed_k = k_min == k_max
    k_clause = (
        f"Target k is exactly {k_min}. The Synthesizer must converge on exactly "
        f"{k_min} clusters."
        if is_fixed_k
        else (
            f"Target k is in the range [{k_min}, {k_max}] inclusive. The "
            f"Synthesizer should converge on a number of clusters within this "
            f"range that best fits the natural structure of the corpus — do "
            f"NOT default to either endpoint without reason."
        )
    )
    cluster_count_phrase = (
        f"the {k_min} clusters" if is_fixed_k else "the clusters"
    )
    none_clause = (
        f"Some texts will not fit any of {cluster_count_phrase} — leave them "
        "unclustered (the plugin tracks these via `unclustered_ids` on "
        "proposer / auditor output). Do NOT create an explicit 'none' or "
        "'other' cluster in the taxonomy."
        if allow_none
        else (
            f"Every text in this corpus belongs to one of {cluster_count_phrase}. "
            "Do NOT create a 'none' or 'other' cluster — every document must "
            "end up assigned to one of the real clusters."
        )
    )
    return f"""\
You are orchestrating headless cluster discovery on the {dataset} benchmark dataset.

The clustering workspace is ALREADY initialised at:

    {workspace_dir}

Export this as the workspace for all corpus-tools scripts before any other call:

    export CLUSTERING_WORKSPACE={workspace_dir}
    if [ -z "$CLAUDE_PLUGIN_ROOT" ]; then export CLAUDE_PLUGIN_ROOT=$(cat {workspace_dir}/.plugin_root); fi

Run the iteration loop described in the cluster-run skill, with these
benchmark-mode constraints:

1. Do NOT ask the user any questions. There is no human in this session.
2. The workspace is already initialised — do NOT re-init.
3. Model tier is 'quality' (set in state.json); all subagents inherit Opus 4.7.
4. {k_clause}
5. {none_clause}
6. Iterate (Proposer → Synthesizer → Auditor → Critic, dispatching Investigator
   on demand) until the standard stop criteria fire (Critic 'ready', coverage
   >85%, diminishing returns), OR you have dispatched {MAX_AGENT_DISPATCHES} agents — whichever
   comes first (this matches the cluster-run skill's hard checkpoint; there is
   no user to confirm continuation in benchmark mode). When stop criteria
   fire, run cluster-finalize — it writes taxonomy.md, final_taxonomy.json,
   AND categories.json (the canonical handoff to the text-classification
   plugin's /classify-run). Do NOT run classify.py — the benchmark harness
   handles that.

Finally, print a 5-line summary: number of iterations, final k, coverage,
mean confidence, and any caveats.
"""


def _run_orchestrator(
    *, workspace_dir: Path, dataset: str, k_min: int, k_max: int, allow_none: bool
) -> dict:
    prompt = _orchestrator_prompt(
        workspace_dir=workspace_dir,
        dataset=dataset,
        k_min=k_min,
        k_max=k_max,
        allow_none=allow_none,
    )
    os.environ["CLUSTERING_WORKSPACE"] = str(workspace_dir)
    # Persist the prompt next to the workspace for post-mortems.
    (workspace_dir / "orchestrator_prompt.txt").write_text(prompt, encoding="utf-8")
    extra_args = [
        # Load both plugins so /cluster-run + the proposer/auditor/critic/
        # investigator/synthesizer agents + corpus-tools scripts resolve
        # (agentic-clustering), and so /classify-run + classify.py resolve
        # for the post-finalize classification step (text-classification —
        # split off from this plugin in commit 36861bb).
        "--plugin-dir", str(PLUGIN_ROOT),
        "--plugin-dir", str(TEXT_CLASSIFICATION_ROOT),
        # Headless sessions block tool execution by default. We need Bash +
        # Task + Read + Write + Edit unrestricted to drive the iteration loop.
        "--permission-mode", "bypassPermissions",
    ]
    t0 = time.perf_counter()
    stdout = call_claude(
        prompt,
        model=ORCHESTRATOR_MODEL,
        timeout_s=60 * 60 * 4,
        log_prefix=f"[agentic/{dataset}]",
        extra_args=extra_args,
    )
    t1 = time.perf_counter()
    # Save the orchestrator's textual reply (its 5-line summary + any
    # narration) for post-mortems. Subagent outputs go into the workspace
    # under proposals/, audits/, investigations/ as before.
    (workspace_dir / "orchestrator_stdout.txt").write_text(stdout or "", encoding="utf-8")
    return {"wall_clock_s": t1 - t0, "stdout": stdout}


def _ensure_orchestrator_outputs(workspace_dir: Path) -> None:
    # cluster-finalize writes all three. classification/prompt.md is gone
    # since commit 36861bb dissolved build_classification_prompt.py;
    # categories.json is the new canonical handoff to /classify-run.
    required = [
        workspace_dir / "final_taxonomy.json",
        workspace_dir / "taxonomy.md",
        workspace_dir / "categories.json",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise RuntimeError(
            f"orchestrator finished but expected outputs missing: {missing}. "
            f"Inspect {workspace_dir}/ to diagnose."
        )


def _run_classify(*, workspace_dir: Path, documents_path: Path, allow_none: bool) -> Path:
    load_secrets_into_env()
    required_key = "OPENAI_API_KEY" if CLASSIFY_PROVIDER == "openai" else "ANTHROPIC_API_KEY"
    if not os.environ.get(required_key):
        raise RuntimeError(
            f"{required_key} not set. Add it to secrets.json at the project root "
            f"(flat dict, e.g. {{\"{required_key}\": \"...\"}}) or export it as a "
            f"shell env var. classify.py needs it to call the {CLASSIFY_MODEL} API."
        )
    classify_dir = workspace_dir / "classification" / "classifications"
    classify_dir.mkdir(parents=True, exist_ok=True)
    output_path = classify_dir / "seed_0.csv"
    args = [
        "--input", str(documents_path),
        "--text-col", "text",
        "--id-col", "doc_id",
        "--prompt", str(workspace_dir / "classification" / "prompt.md"),
        "--output", str(output_path),
        "--provider", CLASSIFY_PROVIDER,
        "--model", CLASSIFY_MODEL,
        "--mode", CLASSIFY_MODE,
    ]
    if CLASSIFY_MODE == "async":
        args += ["--concurrency", str(CLASSIFY_CONCURRENCY)]
    if not allow_none:
        args.append("--force-assign")
    _run_uv_script(CLASSIFY_SCRIPT, args)
    return output_path


def _read_final_taxonomy(workspace_dir: Path) -> dict:
    return json.loads((workspace_dir / "final_taxonomy.json").read_text(encoding="utf-8"))


def _taxonomy_str_to_int_id(final_taxonomy: dict) -> dict[str, int]:
    """Plugin cluster IDs ('c1', 'c2', ...) → ints in final_taxonomy order.
    'none' is reserved for -1 and never present in this map."""
    return {c["id"]: i for i, c in enumerate(final_taxonomy["clusters"])}


def _read_classify_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _classify_cost_usd(rows: list[dict]) -> tuple[float, int, int]:
    in_tokens = sum(int(r.get("input_tokens") or 0) for r in rows)
    cache_tokens = sum(int(r.get("cache_read_tokens") or 0) for r in rows)
    out_tokens = sum(int(r.get("output_tokens") or 0) for r in rows)
    usd = (
        in_tokens * GPT5_MINI_USD_PER_1M_INPUT
        + cache_tokens * GPT5_MINI_USD_PER_1M_CACHE_READ
        + out_tokens * GPT5_MINI_USD_PER_1M_OUTPUT
    ) / 1_000_000.0
    return usd, in_tokens + cache_tokens, out_tokens


def _build_predictions(
    *,
    documents: list[dict],
    classify_rows: list[dict],
    id_map: dict[str, int],
    taxonomy_by_str_id: dict[str, dict],
) -> list[DocPrediction]:
    classify_by_doc = {r["id"]: r for r in classify_rows}
    preds: list[DocPrediction] = []
    for doc in documents:
        row = classify_by_doc.get(doc["doc_id"])
        if row is None:
            pred_id, pred_label, confidence = NONE_LABEL_ID, NONE_LABEL_NAME, None
        else:
            cluster_str = (row.get("cluster") or "").strip()
            if cluster_str in ("", "none"):
                pred_id, pred_label = NONE_LABEL_ID, NONE_LABEL_NAME
            else:
                pred_id = id_map[cluster_str]
                pred_label = taxonomy_by_str_id[cluster_str]["name"]
            try:
                confidence = float(row.get("confidence") or "") / 5.0
            except ValueError:
                confidence = None
        preds.append(
            DocPrediction(
                doc_id=doc["doc_id"],
                text=doc["text"],
                gold_label=doc["gold_label_name"],
                gold_label_id=int(doc["gold_label_id"]),
                is_none=bool(doc["is_none"]),
                predicted_cluster_id=pred_id,
                predicted_cluster_label=pred_label,
                confidence=confidence,
                iteration=0,
            )
        )
    return preds


def _build_taxonomy_entries(final_taxonomy: dict, id_map: dict[str, int]) -> list[TaxonomyEntry]:
    return [
        TaxonomyEntry(
            cluster_id=id_map[c["id"]],
            label=c["name"],
            description=c.get("description", ""),
        )
        for c in final_taxonomy["clusters"]
    ]


DISCOVER_K_FRACTION = 0.2  # discover-k variant uses gold_k ± 20%.
METHOD_DISCOVER_K = "agentic_clustering_discoverk"


def run_agentic_clustering(
    dataset_name: str,
    *,
    seed: int = 0,
    skip_classify: bool = False,
    resume_classify: bool = False,
    discover_k: bool = False,
) -> dict:
    """Run our method on one dataset. Returns a small row dict for printing.

    ``resume_classify`` skips the corpus-build / init / orchestrator steps and
    starts from the existing workspace at results/clustering/<ds>/seed=<n>[_discoverk]/.
    Use after a ``--skip-classify`` run that you want to turn into a real
    predictions artifact without re-running the agent loop.

    ``discover_k`` runs the discover-k variant: k_range is gold_k ± 20% (the
    orchestrator picks a k within that range rather than being pinned). Writes
    to a separate predictions dir (``agentic_clustering_discoverk``) and a
    separate workspace (``seed=<n>_discoverk``) so the given-k artifacts are
    never overwritten.
    """
    if skip_classify and resume_classify:
        raise ValueError("skip_classify and resume_classify are mutually exclusive")
    if dataset_name not in DATASET_LENS:
        raise KeyError(f"no DATASET_LENS entry for {dataset_name!r}")
    lens = DATASET_LENS[dataset_name]
    ds = load_processed(dataset_name)
    k_in_scope = int(ds.meta["k_in_scope"])
    # Feed init.py + classify.py the 512-token-capped corpus, not the canonical
    # one, so the agent loop and per-doc classification see the same truncated
    # bodies as the baselines (SPEC §5.1.1 / §5.6.3).
    documents_path = _materialize_capped_corpus(dataset_name, ds)

    if discover_k:
        k_min = round(k_in_scope * (1 - DISCOVER_K_FRACTION))
        k_max = round(k_in_scope * (1 + DISCOVER_K_FRACTION))
        method = METHOD_DISCOVER_K
        workspace_dir = RESULTS / "clustering" / dataset_name / f"seed={seed}_discoverk"
    else:
        k_min = k_max = k_in_scope
        method = METHOD
        workspace_dir = RESULTS / "clustering" / dataset_name / f"seed={seed}"

    k_range_str = f"k={k_min}" if k_min == k_max else f"k_range=[{k_min},{k_max}]"

    if resume_classify:
        if not workspace_dir.exists():
            raise FileNotFoundError(
                f"--resume-classify: workspace not found at {workspace_dir}. "
                f"Run without --resume-classify first to produce it."
            )
        print(f"[agentic/{dataset_name}] resume: skipping corpus build / init / orchestrator")
        _ensure_orchestrator_outputs(workspace_dir)
        orch = {"wall_clock_s": None}
        t_start = time.perf_counter()
    else:
        workspace_dir.mkdir(parents=True, exist_ok=True)

        print(f"[agentic/{dataset_name}] initialising workspace ({k_range_str}, allow_none={lens.allow_none}, n={len(ds.documents)})")
        _init_workspace(
            workspace_dir=workspace_dir,
            documents_path=documents_path,
            k_min=k_min,
            k_max=k_max,
            lens_text=lens.text,
        )

        t_start = time.perf_counter()
        print(f"[agentic/{dataset_name}] dispatching orchestrator on {ORCHESTRATOR_MODEL}")
        orch = _run_orchestrator(
            workspace_dir=workspace_dir,
            dataset=dataset_name,
            k_min=k_min,
            k_max=k_max,
            allow_none=lens.allow_none,
        )
        print(f"[agentic/{dataset_name}] orchestrator returned in {orch['wall_clock_s']:.1f}s")

        _ensure_orchestrator_outputs(workspace_dir)

        if skip_classify:
            print(f"[agentic/{dataset_name}] --skip-classify; stopping before classify step.")
            return {
                "method": method,
                "dataset": dataset_name,
                "n_docs": len(ds.documents),
                "k_in_scope": k_in_scope,
                "k_range": [k_min, k_max],
                "orchestrator_wall_clock_s": orch["wall_clock_s"],
                "skipped_classify": True,
            }

    print(
        f"[agentic/{dataset_name}] classifying {len(ds.documents)} docs on {CLASSIFY_MODEL} "
        f"(force_assign={not lens.allow_none})"
    )
    classify_csv_path = _run_classify(
        workspace_dir=workspace_dir,
        documents_path=documents_path,
        allow_none=lens.allow_none,
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
        method=method,
        dataset=dataset_name,
        seed=seed,
        predictions=predictions,
        taxonomy=taxonomy_entries,
        cost=cost,
        metrics=metrics.to_dict(),
        model_versions={
            "orchestrator": ORCHESTRATOR_MODEL,
            "classify": CLASSIFY_MODEL,
        },
        iterations=0,
        hyperparameters={
            "k_in_scope": k_in_scope,
            "k_range": [k_min, k_max],
            "discover_k": discover_k,
            "model_tier": "quality",
            "allow_none": lens.allow_none,
            "llm_input_token_cap": LLM_TOKEN_CAP,
            "lens_text": lens.text,
            "classify_mode": CLASSIFY_MODE,
            "classify_force_assign": not lens.allow_none,
            "pricing_basis": PRICING_BASIS,
        },
        extra_meta={
            "n_docs": len(ds.documents),
            "k_actual": len(final_taxonomy["clusters"]),
            "cluster_version_at_finalize": int(final_taxonomy.get("cluster_version", 0)),
            "subscription_usd_basis": "claude_code_max_100usd_div_7_datasets",
            "orchestrator_wall_clock_s": orch["wall_clock_s"],
            "resumed_from_existing_workspace": resume_classify,
            "classify_csv_path": str(classify_csv_path),
        },
    )

    print(
        f"[agentic/{dataset_name}] done. k_actual={len(final_taxonomy['clusters'])} "
        f"api_usd=${api_usd:.4f} total_usd=${cost.usd:.4f} wall_clock={wall_clock_s:.1f}s"
    )
    return {
        "method": method,
        "dataset": dataset_name,
        "n_docs": len(ds.documents),
        "k_in_scope": k_in_scope,
        "k_range": [k_min, k_max],
        "k_actual": len(final_taxonomy["clusters"]),
        "api_usd": api_usd,
        "subscription_usd": SUBSCRIPTION_USD_PER_DATASET,
        "usd": cost.usd,
        "wall_clock_s": wall_clock_s,
        **metrics.to_dict(),
    }
