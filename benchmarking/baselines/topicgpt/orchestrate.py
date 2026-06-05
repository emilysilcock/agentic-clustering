"""Phase orchestrator for the TopicGPT baseline.

| Phase | Output                              | Runner                              |
|-------|-------------------------------------|-------------------------------------|
| 0     | data/topicgpt/<ds>/input.jsonl      | ``dataset_adapter.adapt``           |
| 1     | data/topicgpt/<ds>/topics_lvl1.md   | vendored ``generate_topic_lvl1``    |
|       | data/topicgpt/<ds>/generation.jsonl |   (Claude Code, Opus 4.7)           |
| 2     | data/topicgpt/<ds>/topics_refined.md| vendored ``refine_topics``          |
|       | data/topicgpt/<ds>/updated.jsonl    |   (Claude Code, Opus 4.7)           |
| 3     | data/topicgpt/<ds>/assignment.jsonl | ``batch_assigner.assign``           |
|       |                                     |   (OpenAI Batch API, gpt-5-mini)    |
| 4     | data/topicgpt/<ds>/corrected.jsonl  | vendored ``correct_topics``         |
|       |                                     |   (OpenAI sync, gpt-5-mini)         |
| 5     | results/predictions/topicgpt/...    | ``result_parser.write``             |

Routing follows SPEC §5.6.2 (TopicGPT row, 2026-05-24 revision) and the
§5.6 >1,000-text rule:

* Phase 1 runs the LLM over the corpus until ``early_stop=200`` saturation
  (per the paper's recommendation: stop "when no new topics are generated
  for some threshold (e.g., 200 documents)"). Per-dataset call counts are
  ~k_gold + 200 = ~220–350, well under the 1,000-text threshold ->
  frontier (Opus 4.7 via the Claude Code subscription). Sequential by
  construction (each prompt embeds the topic tree grown from prior calls).
* Phase 2 makes one merge pass per dataset (≪1,000 calls) -> frontier
  (Opus 4.7 via the Claude Code subscription).
* Phase 3 is bulk per-doc -> cheap tier (gpt-5-mini) via OpenAI Batch
  (50% discount, ≤24 h SLA), with auto-caching on the stable prompt prefix.
* Phase 4 reprompts the ~700–1,000 rows whose phase-3 response didn't
  match the upstream regex. Routed to the cheap tier (gpt-5-mini via
  OpenAI sync) for two reasons: (a) volume is high enough that an Opus
  subscription burn would be wasteful, and (b) model consistency --- the
  same model produced the assignments, so the correction reprompt sees
  the same output distribution. The runner reports the failure count
  before phase 4 fires so the user can sanity-check the bill regardless.

Discover-$k$ only --- no ``--k`` flag; TopicGPT decides $k$ itself.
TopicGPT appears only in the discover-$k$ panel of the results table
(SPEC §5.5).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import benchmarking  # noqa: F401 — triggers truststore.inject_into_ssl()
import benchmarking.baselines.topicgpt  # noqa: F401 — puts _vendored/ on sys.path

from benchmarking.baselines.topicgpt.dataset_adapter import (
    TOPICGPT_ROOT,
    adapt,
)
from benchmarking.paths import DATA

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# SPEC §5.6.2 locked pinning (TopicGPT row, post-2026-05-23 revision)
FRONTIER_API = "claude_code"
FRONTIER_MODEL = "claude-opus-4-7"   # via Claude Code Max subscription
CHEAP_API = "openai"
CHEAP_MODEL = "gpt-5-mini"            # via OpenAI sync (phase 1) or Batch (phase 3)

# Upstream defaults --- preserved unless the runner overrides.
GENERATION_MAX_TOKENS = 1000
GENERATION_TEMPERATURE = 0.0
GENERATION_TOP_P = 1.0
# 200 matches the TopicGPT paper's recommended early-stop window
# ("stop when no new topics are generated for some threshold (e.g., 200
# documents)"). The vendored code default is 1000, but the paper's number
# is what's actually empirically tuned and (a) finishes much faster, (b)
# keeps per-phase call counts well under the SPEC §5.6 >1,000-text rule so
# phase 1 routes to the frontier (Opus) tier rather than the cheap tier.
GENERATION_EARLY_STOP = 200

# Context windows for the upstream prompt-fit check in prompt_formatting().
# gpt-5-mini is 400k context; we pin explicitly so we don't depend on the
# vendored fallback (which defaults to 128k for unknown models).
GPT5_MINI_CONTEXT_LEN = 400_000
CLAUDE_OPUS_CONTEXT_LEN = 200_000

REFINE_MAX_TOKENS = 1000
REFINE_TEMPERATURE = 0.0
REFINE_TOP_P = 1.0
# ``remove`` drops low-frequency topics during refinement. Upstream demo
# default is True; we leave that ON for discover-$k$ so the merged taxonomy
# matches the upstream method.
REFINE_REMOVE_LOW_FREQ = True

CORRECT_MAX_TOKENS = 500
CORRECT_TEMPERATURE = 0.0
CORRECT_TOP_P = 1.0


@dataclass(frozen=True)
class PhaseOutput:
    """Output paths for one phase. Returned by each phase function."""
    primary: Path
    secondary: Path | None = None
    usage: dict | None = None


def _topicgpt_dir(dataset_name: str) -> Path:
    out = TOPICGPT_ROOT / dataset_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _exists_all(paths: list[Path]) -> bool:
    return all(p.exists() and p.stat().st_size > 0 for p in paths)


def _save_usage(out_dir: Path, phase: str, usage: dict) -> None:
    (out_dir / f"usage_{phase}.json").write_text(
        json.dumps(usage, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def generate(
    dataset_name: str,
    *,
    overwrite: bool = False,
    verbose: bool = True,
    early_stop: int = GENERATION_EARLY_STOP,
    max_gen_docs: int | None = None,
) -> PhaseOutput:
    """Phase 1: discover topics by streaming through the corpus (Opus 4.7).

    Per SPEC §5.6.2 (TopicGPT row, 2026-05-23 revision): at the paper's
    recommended ``early_stop=200``, per-dataset call counts are ~k_gold + 200
    = ~220–350, well under the SPEC §5.6 >1,000-text threshold. Runs on the
    frontier tier (Opus 4.7 via the Claude Code Max subscription).
    Sequential by construction --- each prompt embeds the topic tree grown
    from prior calls --- so batch parallelism does not apply.
    """
    # Defer import so module-load time SBERT init only happens when this
    # phase is actually invoked.
    from topicgpt_python import generate_topic_lvl1

    adapted = adapt(dataset_name)
    out_dir = _topicgpt_dir(dataset_name)
    topic_file = out_dir / "topics_lvl1.md"
    out_file = out_dir / "generation.jsonl"

    if not overwrite and _exists_all([topic_file, out_file]):
        print(f"[topicgpt/{dataset_name}/phase=generate] cache hit -> {topic_file}", flush=True)
        return PhaseOutput(primary=topic_file, secondary=out_file)

    # Empty seed file --- discover-$k$, no prior topic seeding.
    seed_file = out_dir / "_empty_seed.md"
    seed_file.write_text("", encoding="utf-8")

    # Hard cap on docs scanned (defensive backstop above early_stop). For
    # datasets where the model keeps proposing novel topics every few
    # hundred docs (observed on GoEmotions: emotion-bearing utterances are
    # subtle enough that the model rarely fully saturates), early_stop
    # never fires and generation can run indefinitely. The cap writes a
    # truncated input file and points the vendored loop at it.
    gen_data_path = adapted.jsonl_path
    n_scanned = adapted.n_docs
    if max_gen_docs is not None and adapted.n_docs > max_gen_docs:
        capped_path = out_dir / f"input_capped_{max_gen_docs}.jsonl"
        lines = adapted.jsonl_path.read_text(encoding="utf-8").splitlines()
        capped_path.write_text("\n".join(lines[:max_gen_docs]) + "\n", encoding="utf-8")
        gen_data_path = capped_path
        n_scanned = max_gen_docs

    print(
        f"[topicgpt/{dataset_name}/phase=generate] {n_scanned} docs"
        + (f" (capped from {adapted.n_docs})" if max_gen_docs and adapted.n_docs > max_gen_docs else "")
        + f" | model={FRONTIER_MODEL} via {FRONTIER_API} | "
        + f"early_stop={early_stop}",
        flush=True,
    )

    # We can't capture the APIClient out of generate_topic_lvl1 directly (it
    # constructs one internally). Instead, monkey-patch APIClient.__init__ to
    # tee a reference into a list so we can read .usage afterwards.
    from topicgpt_python.utils import APIClient
    constructed: list[APIClient] = []
    orig_init = APIClient.__init__
    def _tee_init(self, *a, **kw):
        orig_init(self, *a, **kw)
        constructed.append(self)
    APIClient.__init__ = _tee_init
    try:
        generate_topic_lvl1(
            api=FRONTIER_API,
            model=FRONTIER_MODEL,
            data=str(gen_data_path),
            prompt_file=str(PROMPTS_DIR / "generation_1.txt"),
            seed_file=str(seed_file),
            out_file=str(out_file),
            topic_file=str(topic_file),
            verbose=verbose,
            max_tokens=GENERATION_MAX_TOKENS,
            temperature=GENERATION_TEMPERATURE,
            top_p=GENERATION_TOP_P,
            early_stop=early_stop,
            context_len=CLAUDE_OPUS_CONTEXT_LEN,
        )
    finally:
        APIClient.__init__ = orig_init

    usage = constructed[0].usage if constructed else {}
    _save_usage(out_dir, "generate", usage)
    print(
        f"[topicgpt/{dataset_name}/phase=generate] -> {topic_file} | "
        f"usage={usage}",
        flush=True,
    )
    return PhaseOutput(primary=topic_file, secondary=out_file, usage=usage)


def refine(
    dataset_name: str,
    *,
    overwrite: bool = False,
    verbose: bool = True,
) -> PhaseOutput:
    """Phase 2: merge near-duplicate topics, drop low-frequency ones."""
    from topicgpt_python import refine_topics

    out_dir = _topicgpt_dir(dataset_name)
    topic_in = out_dir / "topics_lvl1.md"
    generation_in = out_dir / "generation.jsonl"
    if not _exists_all([topic_in, generation_in]):
        raise FileNotFoundError(
            f"phase 2 needs phase 1 outputs: {topic_in}, {generation_in}"
        )

    topic_out = out_dir / "topics_refined.md"
    updated_out = out_dir / "updated.jsonl"
    mapping_out = out_dir / "refinement_mapping.json"

    if not overwrite and _exists_all([topic_out, updated_out]):
        print(f"[topicgpt/{dataset_name}/phase=refine] cache hit -> {topic_out}", flush=True)
        return PhaseOutput(primary=topic_out, secondary=updated_out)

    print(
        f"[topicgpt/{dataset_name}/phase=refine] model={FRONTIER_MODEL} via {FRONTIER_API}",
        flush=True,
    )

    from topicgpt_python.utils import APIClient
    constructed: list[APIClient] = []
    orig_init = APIClient.__init__
    def _tee_init(self, *a, **kw):
        orig_init(self, *a, **kw)
        constructed.append(self)
    APIClient.__init__ = _tee_init
    try:
        refine_topics(
            api=FRONTIER_API,
            model=FRONTIER_MODEL,
            prompt_file=str(PROMPTS_DIR / "refinement.txt"),
            generation_file=str(generation_in),
            topic_file=str(topic_in),
            out_file=str(topic_out),
            updated_file=str(updated_out),
            verbose=verbose,
            remove=REFINE_REMOVE_LOW_FREQ,
            mapping_file=str(mapping_out),
            max_tokens=REFINE_MAX_TOKENS,
            temperature=REFINE_TEMPERATURE,
            top_p=REFINE_TOP_P,
        )
    finally:
        APIClient.__init__ = orig_init

    usage = constructed[0].usage if constructed else {}
    _save_usage(out_dir, "refine", usage)
    print(
        f"[topicgpt/{dataset_name}/phase=refine] -> {topic_out} | usage={usage}",
        flush=True,
    )
    return PhaseOutput(primary=topic_out, secondary=updated_out, usage=usage)


def count_correction_targets(dataset_name: str) -> dict:
    """Count rows in ``assignment.jsonl`` that would need a correction LLM call.

    Re-uses vendored ``identify_errors`` so the counting logic stays in lock-step
    with what ``correct_topics`` will actually reprompt. Returns a dict with
    ``n_total``, ``n_error`` (no topic name extracted), ``n_hallucinated``
    (topic name not in the refined taxonomy), and ``n_to_correct``.
    """
    import pandas as pd
    from topicgpt_python.correction import topic_parser
    from topicgpt_python.utils import TopicTree

    out_dir = _topicgpt_dir(dataset_name)
    topic_file = out_dir / "topics_refined.md"
    assignment_in = out_dir / "assignment.jsonl"
    if not _exists_all([topic_file, assignment_in]):
        raise FileNotFoundError(
            f"need phase 2 + 3 outputs: {topic_file}, {assignment_in}"
        )

    df = pd.read_json(assignment_in, lines=True)
    topics_root = TopicTree().from_topic_list(str(topic_file), from_file=True)
    # topic_parser returns (error_indices, hallucinated_indices); upstream's
    # `correct` reprompts the union of those.
    error, hallucinated = topic_parser(topics_root, df, verbose=False)
    return {
        "n_total": len(df),
        "n_error": len(error),
        "n_hallucinated": len(hallucinated),
        "n_to_correct": len(set(error) | set(hallucinated)),
    }


def correct(
    dataset_name: str,
    *,
    overwrite: bool = False,
    verbose: bool = True,
) -> PhaseOutput:
    """Phase 4: reassign rows whose responses are 'Error' or hallucinated.

    Routes through ``batch_correct.correct()`` (OpenAI Batch API,
    ``gpt-5-mini``) per the SPEC §5.6.2 revision (2026-05-24). The
    vendored ``correct_topics`` only supports sync iterative prompts (or
    vLLM batch); for the ~hundreds-to-low-thousands of reprompts that
    surface after phase 3 we want OpenAI's batch 50% discount + parallelism
    + auto-cache on the stable prompt prefix. See ``batch_correct.py``.
    """
    from benchmarking.baselines.topicgpt import batch_correct

    result = batch_correct.correct(dataset_name, overwrite=overwrite)
    _save_usage(out_dir := _topicgpt_dir(dataset_name), "correct", result.usage)
    return PhaseOutput(primary=result.out_path, usage=result.usage)
