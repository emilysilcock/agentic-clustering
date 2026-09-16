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
  3. Reconcile the finalized categories.json against the dataset's label
     policy: strip its ``none`` entry for the 5 datasets whose gold labels
     don't include an OOS/none class (post-split classify.py derives
     force-assign from the category set, not a flag).
  4. Classify the full corpus by running the /classify-run skill in a second
     headless ``claude -p`` session — the same path a plugin user takes —
     on gpt-5-mini in batch mode with prompt caching. Switched from Claude Haiku 4.5 on
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

# Repo-level helper the /classify-run session uses to start classify.py
# detached, so a multi-hour batch poll outlives both the session's Bash
# per-call timeout and the session itself.
REPO_ROOT = PLUGIN_ROOT.parent
LAUNCH_DETACHED_SCRIPT = REPO_ROOT / "scripts" / "launch_detached.py"


def _venv_python() -> Path:
    """Interpreter for the detached launcher. Prefer the venv's console
    python: the launcher itself is short-lived and its own child gets
    CREATE_NO_WINDOW, so no console is ever shown.
    """
    candidate = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if candidate.exists():
        return candidate
    candidate = REPO_ROOT / ".venv" / "bin" / "python"
    if candidate.exists():
        return candidate
    return Path(sys.executable)

# The classification scripts were split off into a separate plugin in
# commit 36861bb. `build_classification_prompt.py` was dissolved entirely
# (cluster-finalize now writes categories.json directly via state.py:952);
# `classify.py` was moved to the text-classification plugin's classify-tools
# scripts dir. We expect the text-classification repo cloned as a sibling
# of agentic-clustering so the headless session can load both plugins.
TEXT_CLASSIFICATION_ROOT = PLUGIN_ROOT.parents[1] / "text-classification" / "plugin"
CLASSIFY_SCRIPT = TEXT_CLASSIFICATION_ROOT / "skills" / "classify-tools" / "scripts" / "classify.py"

# There is deliberately no agent-dispatch cap here. The harness must exercise
# the shipped plugin's own control flow, so the orchestrator stops on the
# state-grounded criteria in cluster-run/SKILL.md ("When NOT to Continue":
# critic satisfied + coverage >85%, or diminishing returns) and nothing else.
# Earlier revisions imposed a cumulative cap (8 until 2026-06-05, then 20),
# mirroring the SKILL.md hard checkpoint that issue #2 removed; the cap went
# with it, since a benchmark-only stopping rule measures the harness rather
# than the method. Results published before 2026-09-16 were produced at 20.

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
    *, workspace_dir: Path, dataset: str, k_min: int, k_max: int, allow_none: bool,
    initial_proposers: int | None = None,
    max_agent_dispatches: int | None = None,
) -> str:
    """Build the headless orchestration prompt.

    ``initial_proposers`` and ``max_agent_dispatches`` are both unset by
    default, which leaves the proposer count and the stopping rule to the
    shipped cluster-run skill. Setting either pins that decision from the
    harness, which is how a published run gets reproduced after the plugin has
    moved on (see PAPER_CONFIG). Neither touches the plugin.
    """
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
    # Unset => the skill's own stopping rule, verbatim. Set => the pre-issue-#2
    # cumulative cap, restored for reproducing a published run.
    stop_clause = (
        "There is no dispatch cap: keep iterating while\n"
        "   the taxonomy is still improving and stop when it is not, exactly as the\n"
        "   skill describes. A dispatch count is not a stopping condition."
        if max_agent_dispatches is None
        else (
            f"OR you have dispatched {max_agent_dispatches} agents cumulatively —\n"
            "   whichever comes first. The cap is a harness-imposed reproduction\n"
            "   constraint, not the skill's own rule."
        )
    )
    proposer_clause = (
        f"\n7. In the INITIAL proposal round, dispatch exactly {initial_proposers} "
        f"proposers in parallel, overriding the cluster-run skill's default "
        f"proposer count. Follow-up targeted proposer / investigator dispatches "
        f"later in the loop proceed as normal."
        if initial_proposers is not None
        else ""
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

CRITICAL — SINGLE-SHOT HEADLESS SESSION. This is one non-interactive `claude -p`
turn. There is NO user and NO mechanism to resume you once your turn ends. If you
end your turn while ANY sub-agent (proposer / synthesizer / auditor / critic /
investigator) is still pending, the run DIES incomplete and is discarded. The
Task tool is SYNCHRONOUS: each Task call runs the sub-agent and returns its result
WITHIN your current turn. You MUST therefore drive the entire loop to completion
in one continuous flow — dispatch a Task (you may issue several Task calls
together to run proposers concurrently), let it RETURN inline, read the result,
and immediately continue to the next step. NEVER stop to "wait for completion
notifications", NEVER say you are "waiting", and NEVER end your turn before
cluster-finalize has written final_taxonomy.json. Just keep working until finalize
is done.

Run the iteration loop described in the cluster-run skill, with these
benchmark-mode constraints:

1. Do NOT ask the user any questions. There is no human in this session.
2. The workspace is already initialised — do NOT re-init.
3. Model tier is 'quality' (set in state.json); all subagents inherit Opus 4.7.
4. {k_clause}
5. {none_clause}
6. Iterate (Proposer → Synthesizer → Auditor → Critic, dispatching Investigator
   on demand) until the standard stop criteria fire (Critic 'ready', coverage
   >85%, diminishing returns). {stop_clause} When stop
   criteria fire, run cluster-finalize — it writes taxonomy.md, final_taxonomy.json,
   AND categories.json (the canonical handoff to the text-classification
   plugin's /classify-run). Stop there: do NOT classify. The harness runs
   /classify-run itself, in a separate headless session, once you are done.{proposer_clause}

Finally, print a 5-line summary: number of iterations, final k, coverage,
mean confidence, and any caveats.
"""


def _summarize_orchestrator_usage(result_json: dict) -> dict | None:
    """Distil a ``claude -p --output-format json`` result into a big-model
    (Opus) token summary for meta.json.

    ``modelUsage`` and ``total_cost_usd`` are session-wide totals that include
    every Task sub-agent (proposer / synthesizer / auditor / investigator /
    critic), not just the orchestrator's own turns --- verified empirically
    2026-07-12 (a sub-agent-dispatching run reports ~3x the tokens and cost of
    the same task done inline, and ``total_cost_usd`` equals the summed
    ``modelUsage`` cost). The top-level ``usage`` block is main-loop only and is
    deliberately ignored. Returns None when the result carries no ``modelUsage``,
    so the meta field is simply absent and the results-table cell stays ``?``.

    ``big_input_tokens`` counts every token the model processed, including
    cache reads/creations (a multi-turn agent re-reads its cached prefix each
    turn) --- this is the figure comparable to the baselines' per-call tiktoken
    sums. ``big_input_tokens_no_cache`` is the uncached-input portion only, kept
    for transparency. The full per-model breakdown is preserved under
    ``model_usage`` so nothing is lost.
    """
    if not result_json:
        return None
    model_usage = result_json.get("modelUsage") or {}
    if not model_usage:
        return None
    raw_in = cache_read = cache_creation = big_out = 0
    for mu in model_usage.values():
        raw_in += int(mu.get("inputTokens", 0) or 0)
        cache_read += int(mu.get("cacheReadInputTokens", 0) or 0)
        cache_creation += int(mu.get("cacheCreationInputTokens", 0) or 0)
        big_out += int(mu.get("outputTokens", 0) or 0)
    return {
        "model_usage": model_usage,
        "total_cost_usd": result_json.get("total_cost_usd"),
        "num_turns": result_json.get("num_turns"),
        "duration_ms": result_json.get("duration_ms"),
        "session_id": result_json.get("session_id"),
        "big_input_tokens": raw_in + cache_read + cache_creation,
        "big_input_tokens_no_cache": raw_in,
        "big_cache_read_tokens": cache_read,
        "big_cache_creation_tokens": cache_creation,
        "big_output_tokens": big_out,
    }


def _run_orchestrator(
    *, workspace_dir: Path, dataset: str, k_min: int, k_max: int, allow_none: bool,
    initial_proposers: int | None = None,
    max_agent_dispatches: int | None = None,
) -> dict:
    prompt = _orchestrator_prompt(
        workspace_dir=workspace_dir,
        dataset=dataset,
        k_min=k_min,
        k_max=k_max,
        allow_none=allow_none,
        initial_proposers=initial_proposers,
        max_agent_dispatches=max_agent_dispatches,
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
    # capture={} routes the call through `--output-format json` so we recover
    # the session-wide Opus token usage (modelUsage / total_cost_usd, which
    # include every Task sub-agent). The Claude Code Max subscription meters no
    # tokens itself, so this is the only place the big-model usage is observable
    # --- it must be captured live; it cannot be reconstructed after the fact.
    capture: dict = {}
    stdout = call_claude(
        prompt,
        model=ORCHESTRATOR_MODEL,
        timeout_s=60 * 60 * 4,
        log_prefix=f"[agentic/{dataset}]",
        extra_args=extra_args,
        capture=capture,
    )
    t1 = time.perf_counter()
    # Save the orchestrator's textual reply (its 5-line summary + any
    # narration) for post-mortems. Subagent outputs go into the workspace
    # under proposals/, audits/, investigations/ as before.
    (workspace_dir / "orchestrator_stdout.txt").write_text(stdout or "", encoding="utf-8")
    result_json = capture.get("result_json") or {}
    if result_json:
        # Persist the full result envelope (usage, modelUsage, cost, session_id,
        # num_turns) as the raw record behind the meta.json summary.
        (workspace_dir / "orchestrator_result.json").write_text(
            json.dumps(result_json, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    usage = _summarize_orchestrator_usage(result_json)
    if usage is not None:
        print(
            f"[agentic/{dataset}] orchestrator usage: "
            f"big_in={usage['big_input_tokens']:,} (no-cache {usage['big_input_tokens_no_cache']:,}) "
            f"big_out={usage['big_output_tokens']:,} "
            f"cost_usd=${usage.get('total_cost_usd') or 0:.2f} turns={usage.get('num_turns')}",
            flush=True,
        )
    else:
        print(
            f"[agentic/{dataset}] WARNING: no modelUsage in orchestrator result; "
            f"big-model tokens will be absent (results-table cell stays '?').",
            flush=True,
        )
    return {"wall_clock_s": t1 - t0, "stdout": stdout, "usage": usage}


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


def _reconcile_categories_with_allow_none(categories_path: Path, allow_none: bool) -> int:
    """Make categories.json's ``none`` entry match the dataset's allow_none
    policy, reproducing the old ``classify.py --force-assign`` semantics
    deterministically.

    Post-split, classify.py has no --force-assign flag: it derives behaviour
    purely from whether a ``{"id": "none"}`` entry is present in categories.json
    (build_schema / build_system_prompt). Present => ``none`` is a permitted
    label; absent => the schema enum forces every text onto a real cluster.
    cluster-finalize appends a ``none`` entry by default (state.py), so for
    force-assign datasets (allow_none=False) we strip it here rather than relying
    on the orchestrator having passed --no-none-category. The appended entry
    mirrors the one state.py writes, so allow_none=True runs are identical
    whether the orchestrator kept it or we re-add it. Returns the category count.
    """
    cats = json.loads(categories_path.read_text(encoding="utf-8"))
    has_none = any(c.get("id") == "none" for c in cats)
    if allow_none and not has_none:
        cats.append({
            "id": "none",
            "name": "Out of scope",
            "description": (
                "Text does not fit any of the categories above. Use when the "
                "text is genuinely outside the taxonomy, not just a poor fit."
            ),
        })
    elif not allow_none and has_none:
        cats = [c for c in cats if c.get("id") != "none"]
    categories_path.write_text(
        json.dumps(cats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return len(cats)


def _classify_prompt(
    *,
    workspace_dir: Path,
    documents_path: Path,
    output_path: Path,
    n_texts: int | None,
    launcher_prefix: str,
    log_path: Path,
) -> str:
    """Directions for a headless /classify-run session.

    The skill's own workflow asks the user four things — where categories.json
    is, which corpus, which provider, which execution mode. There is no user
    here, so we answer all four up front and let the skill do the rest (prompt
    assembly, caching, structured outputs, the report).
    Provider / model / mode are the SPEC §5.6.1 cheap tier, the same values the
    harness used to pass to classify.py itself.

    The one deviation from the skill's step 4 is *how* the command is started.
    classify.py's batch path submits and then polls in a single blocking
    process (``while True`` / ``asyncio.sleep(60)`` until the batch is
    terminal), which for a full corpus runs minutes to hours. Run directly, it
    would sit inside one Bash tool call and be killed at that tool's per-call
    timeout, and it would in any case be reaped when this session exits. So the
    session launches the skill's own command detached and returns; the harness
    does the waiting.
    """
    mode_clause = (
        f"--mode batch (the corpus has {n_texts} texts; batch is ~50% cheaper "
        f"and the SLA is fine for a benchmark)"
        if CLASSIFY_MODE == "batch"
        else f"--mode async --concurrency {CLASSIFY_CONCURRENCY}"
    )
    return f"""\
Classify a benchmark corpus by running the /classify-run skill.

CRITICAL — HEADLESS SESSION. This is one non-interactive `claude -p` turn.
There is NO user: do not ask any questions, do not offer choices, do not wait
for confirmation before submitting the batch. Every decision the skill would
normally put to a user is fixed below.

The classification workspace is ALREADY set up at:

    {workspace_dir}

`CLASSIFY_WORKSPACE` is already exported in this session's environment and
points there, so the skill's workspace-resolution block is a no-op — do not
re-export it and do not go looking for a pointer file.

These are Windows paths running under Git Bash: put every path in double
quotes in every command, or the backslashes will be eaten as escapes.

`categories.json` is already there, written by cluster-finalize and then
reconciled by the harness to this dataset's label policy. Do NOT edit it, do
NOT add or remove a `none` entry, and do NOT regenerate it — its category set
is the experimental condition. Use it exactly as it is.

Invoke /classify-run with these answers to its setup questions:

1. categories.json  -> {workspace_dir / "categories.json"}
2. corpus           -> {documents_path}
                       --text-col text --id-col doc_id
3. provider / model -> --provider {CLASSIFY_PROVIDER} --model {CLASSIFY_MODEL}
4. execution mode   -> {mode_clause}
5. output           -> --output {output_path}
                       (exact path; the harness reads this file by name, so do
                       not use the skill's timestamped run_<...>.csv default)

Do not pass --overwrite. If the output already exists the run should fail
rather than clobber it — the harness decides when a re-run is allowed.

HOW TO START IT — read this before running anything.

Build the skill's step-4 `classify.py` command exactly as the skill documents
it, with the flags above. Then do NOT run it in the foreground. Prefix it with
this launcher, which starts it detached and returns immediately:

    {launcher_prefix} --log "{log_path}" -- <the skill's uv run classify.py command>

Everything after `--` is the skill's command, unchanged. The launcher prints
`pid=<n> log=<path>` and exits in milliseconds.

Why: classify.py's batch path submits and then polls until the batch is
terminal, in one blocking process that runs for minutes to hours. In the
foreground it would exceed this session's Bash per-call timeout, and it would
be killed when this session ends. Detached, it outlives both, and the harness
waits for the output CSV.

Then, to confirm the submission actually got off the ground:

1. Wait a short while (a `sleep 45` is fine — one short Bash call).
2. Read "{log_path}". A healthy OpenAI batch logs an upload line and then
   `batch id: batch_...`, followed by `status=... completed=N/M`.
3. If the log shows a traceback or an auth error instead, report that as a
   failure — do not retry and do not resubmit, since a duplicate batch costs
   real money.

Do NOT poll the batch to completion yourself, and do NOT wait for the CSV.
Once you have seen the batch id, report and end your turn:

- the launcher's pid and log path
- the batch id (or ids, if the corpus was chunked)
- the command you ran, verbatim
- whether the log looks healthy

The harness takes it from there.

"""


def _run_classify(
    *, workspace_dir: Path, documents_path: Path, allow_none: bool, dataset: str = "?"
) -> Path:
    """Classify the full corpus through the text-classification plugin's
    /classify-run skill, in a headless Claude Code session.

    This deliberately goes through the skill rather than calling classify.py
    directly: the benchmark should exercise the same path a user gets. The
    harness still owns the two things that are experimental conditions rather
    than user choices — the categories.json `none` policy (reconciled below)
    and the cheap-tier provider/model/mode — and it still reads the output CSV
    itself for metrics and cost.
    """
    load_secrets_into_env()
    required_key = "OPENAI_API_KEY" if CLASSIFY_PROVIDER == "openai" else "ANTHROPIC_API_KEY"
    if not os.environ.get(required_key):
        raise RuntimeError(
            f"{required_key} not set. Add it to secrets.json at the project root "
            f"(flat dict, e.g. {{\"{required_key}\": \"...\"}}) or export it as a "
            f"shell env var. classify.py needs it to call the {CLASSIFY_MODEL} API."
        )
    # cluster-finalize writes categories.json to the workspace root; reconcile
    # its `none` entry with allow_none before the skill reads it (see above).
    categories_path = workspace_dir / "categories.json"
    if not categories_path.exists():
        raise FileNotFoundError(
            f"categories.json not found at {categories_path}; cluster-finalize must "
            f"run before classify (was the orchestrator/finalize step skipped?)."
        )
    n_cats = _reconcile_categories_with_allow_none(categories_path, allow_none)
    print(
        f"[classify] categories.json reconciled to allow_none={allow_none} "
        f"({n_cats} categories, none {'included' if allow_none else 'stripped'})"
    )
    classify_dir = workspace_dir / "classification" / "classifications"
    classify_dir.mkdir(parents=True, exist_ok=True)
    output_path = classify_dir / "seed_0.csv"

    n_texts: int | None = None
    try:
        with open(documents_path, encoding="utf-8") as f:
            n_texts = sum(1 for line in f if line.strip())
    except OSError:
        pass

    log_path = workspace_dir / "classification" / "classify.log"
    prompt = _classify_prompt(
        workspace_dir=workspace_dir,
        documents_path=documents_path,
        output_path=output_path,
        n_texts=n_texts,
        launcher_prefix=f'"{_venv_python()}" "{LAUNCH_DETACHED_SCRIPT}"',
        log_path=log_path,
    )
    # The skill resolves CLASSIFY_WORKSPACE from the env first, before any
    # pointer-file lookup, so setting it here keeps the session off the
    # .claude/clustering/.active_workspace path entirely.
    os.environ["CLASSIFY_WORKSPACE"] = str(workspace_dir)
    # The skill's documented command is a plain `uv run`, but uv on this machine
    # can't TLS-verify with SSL_CERT_FILE set (see _uv_env / the uv TLS
    # workaround). The harness's own calls pass --native-tls explicitly; we
    # can't edit the skill's command, so set the environment equivalents here
    # and let the skill's command work unmodified.
    os.environ.pop("SSL_CERT_FILE", None)
    os.environ["UV_NATIVE_TLS"] = "1"
    (workspace_dir / "classify_prompt.txt").write_text(prompt, encoding="utf-8")
    extra_args = [
        "--plugin-dir", str(TEXT_CLASSIFICATION_ROOT),
        "--permission-mode", "bypassPermissions",
    ]
    # The session only submits and reports, so it needs minutes, not hours.
    # capture={} records its Opus usage, which is new overhead the direct
    # classify.py call didn't have — small next to the agent loop, but it
    # should not be invisible.
    capture: dict = {}
    print(
        f"[classify/{dataset}] running /classify-run headless "
        f"({CLASSIFY_PROVIDER}/{CLASSIFY_MODEL}, mode={CLASSIFY_MODE})",
        flush=True,
    )
    stdout = call_claude(
        prompt,
        model=ORCHESTRATOR_MODEL,
        timeout_s=60 * 30,
        log_prefix=f"[classify/{dataset}]",
        extra_args=extra_args,
        capture=capture,
    )
    (workspace_dir / "classify_stdout.txt").write_text(stdout or "", encoding="utf-8")
    result_json = capture.get("result_json") or {}
    if result_json:
        (workspace_dir / "classify_session_result.json").write_text(
            json.dumps(result_json, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    _await_classify_output(
        output_path=output_path,
        log_path=log_path,
        dataset=dataset,
        session_stdout_path=workspace_dir / "classify_stdout.txt",
    )
    return output_path


def _read_pid(pid_path: Path) -> int | None:
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    """Best-effort liveness check for the detached classify process."""
    if sys.platform == "win32":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _await_classify_output(
    *,
    output_path: Path,
    log_path: Path,
    dataset: str,
    session_stdout_path: Path,
    timeout_s: float = 60 * 60 * 24,
    poll_s: float = 60.0,
) -> None:
    """Block until the detached classify process writes its output CSV.

    This is where the long wait lives now. The session that submitted the batch
    has already exited; the work continues in a detached process, so the only
    things to watch are the output file and the process itself. A batch that is
    still running with a dead writer is recoverable — the batch id is in the
    log — so say so rather than silently hanging until the timeout.
    """
    pid = _read_pid(Path(str(log_path) + ".pid"))
    deadline = time.time() + timeout_s
    last_line = ""
    grace_checks = 3  # tolerate a not-yet-visible pid right after launch

    while time.time() < deadline:
        if output_path.exists():
            print(f"[classify/{dataset}] output ready: {output_path}", flush=True)
            return
        # Surface the newest progress line so a long batch isn't a silent wait.
        try:
            lines = [
                ln.strip()
                for ln in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                if ln.strip()
            ]
            if lines and lines[-1] != last_line:
                last_line = lines[-1]
                print(f"[classify/{dataset}] {last_line}", flush=True)
        except OSError:
            pass

        if pid is not None and not _pid_alive(pid):
            if grace_checks > 0:
                grace_checks -= 1
            else:
                raise RuntimeError(
                    f"detached classify process (pid {pid}) exited without writing "
                    f"{output_path}.\nLog: {log_path}\nSession transcript: "
                    f"{session_stdout_path}\nIf the log shows a submitted batch id, "
                    f"the batch is still billable and collectable — "
                    f"scripts/recover_orphan_batches.py can pick it up rather than "
                    f"resubmitting."
                )
        else:
            grace_checks = 3
        time.sleep(poll_s)

    raise TimeoutError(
        f"classify did not finish within {timeout_s / 3600:.1f}h for {dataset}. "
        f"Log: {log_path}. The batch may still be in flight; check it before "
        f"resubmitting (a duplicate batch is a duplicate bill)."
    )


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
            # Post-split classify.py writes the assigned cluster id in the
            # `label` column (was `cluster` under the old prompt-based CLI).
            cluster_str = (row.get("label") or "").strip()
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

# What the paper's runs actually used, recovered from the archived run_logs
# under results/clustering/<ds>/<workspace>/. The shipped plugin has since moved
# on — 6-7 initial proposers as of commit 2720ff0, and no dispatch cap at all
# since issue #2 — so a default harness run no longer matches these. Reproducing
# a published number means asking for that configuration explicitly; nothing
# here is a default, and none of it changes the plugin.
#
#   main      given-k + discover-k seed=0, 2026-05-22/23
#   notask    Ablation 2 (blank instructions), 2026-05-25
#   nok       Ablation 3 (k anchor removed), 2026-07-12
#   synthonly Ablation 1 runs no orchestrator at all — it re-classifies the
#             archived first-synth taxonomy — so neither knob applies.
PAPER_CONFIG: dict[str, dict[str, int]] = {
    "main": {"initial_proposers": 3, "max_agent_dispatches": 8},
    "notask": {"initial_proposers": 3, "max_agent_dispatches": 8},
    "nok": {"initial_proposers": 3, "max_agent_dispatches": 20},
    "synthonly": {},
}


def run_agentic_clustering(
    dataset_name: str,
    *,
    seed: int = 0,
    skip_classify: bool = False,
    resume_classify: bool = False,
    discover_k: bool = False,
    initial_proposers: int | None = None,
    max_agent_dispatches: int | None = None,
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

    ``initial_proposers`` and ``max_agent_dispatches`` default to ``None``,
    leaving both decisions to the shipped skill. Pass ``**PAPER_CONFIG["main"]``
    to reproduce the published seed=0 configuration instead.
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
            initial_proposers=initial_proposers,
            max_agent_dispatches=max_agent_dispatches,
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
        dataset=dataset_name,
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
            # None = left to the shipped skill; a number = harness override,
            # which is what a paper-reproduction run looks like.
            "initial_proposers": initial_proposers,
            "max_agent_dispatches": max_agent_dispatches,
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
            # Big-model (Opus) token usage for the agent loop, captured live
            # from the orchestrator's `--output-format json` result. Absent on
            # resume runs (orchestrator not re-dispatched) => results-table cell
            # stays '?'. Present => build_results_table reads big_input_tokens /
            # big_output_tokens from here.
            **({"orchestrator_usage": orch["usage"]} if orch.get("usage") else {}),
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
