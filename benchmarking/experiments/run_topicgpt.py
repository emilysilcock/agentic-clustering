"""CLI runner for the TopicGPT baseline.

Drives all phases via ``benchmarking.baselines.topicgpt``:

  Phase 0 ``adapt``         --- documents.jsonl -> data/topicgpt/<ds>/input.jsonl
  Phase 1 ``generate``      --- Opus 4.7 (Claude Code) -> topics_lvl1.md
  Phase 2 ``refine``        --- Opus 4.7 (Claude Code) -> topics_refined.md
  Phase 3 ``assign``        --- gpt-5-mini (OpenAI Batch API) -> assignment.jsonl
  Phase 3.5 ``count-correct``--- print the row count needing correction (no LLM)
  Phase 4 ``correct``       --- gpt-5-mini (OpenAI Batch API) -> corrected.jsonl
                                **requires ``--confirm-correct``** (gate keeps an
                                unbounded subscription burn from sneaking in if
                                assignment had an unusually high error rate)
  Phase 5 ``write``         --- parse + score -> results/predictions/topicgpt/<ds>/seed=0.{jsonl,meta.json}

Discover-$k$ only (SPEC §5.5) --- no ``--k`` flag. TopicGPT appears only in
the discover-$k$ panel of the results table.

Routing per SPEC §5.6.2 (post-2026-05-23 revision) and the §5.6 >1,000-text
rule: phases with >1,000 calls go cheap (gpt-5-mini via OpenAI);
phases with ≪1,000 calls stay on Opus 4.7 via the Claude Code subscription.

Examples:
    # End-to-end up to (but not including) the correction step. Prints the
    # count of rows needing correction, then exits so you can decide.
    uv run --native-tls python -m benchmarking.experiments.run_topicgpt \\
        --phase all --only banking77

    # See the correction count without re-running anything earlier:
    uv run --native-tls python -m benchmarking.experiments.run_topicgpt \\
        --phase count-correct --only banking77

    # Once you've seen the count and you're happy, run the correction:
    uv run --native-tls python -m benchmarking.experiments.run_topicgpt \\
        --phase correct --only banking77 --confirm-correct

    # Score directly off assignment.jsonl (skipping correction entirely):
    uv run --native-tls python -m benchmarking.experiments.run_topicgpt \\
        --phase write --only banking77
"""

from __future__ import annotations

import argparse

import benchmarking  # noqa: F401 — truststore.inject_into_ssl()

from benchmarking.baselines.topicgpt import batch_assigner
from benchmarking.baselines.topicgpt import dataset_adapter
from benchmarking.baselines.topicgpt import orchestrate
from benchmarking.baselines.topicgpt import result_parser

METHOD = "topicgpt"

DATASETS = [
    "banking77",
    "clinc150",
    "massive_intent",
    "massive_domain",
    "goemotions",
    "twenty_newsgroups",
    "stackexchange",
]

PHASE_CHOICES = (
    "adapt",
    "generate",
    "refine",
    "assign",
    "count-correct",
    "correct",
    "write",
    "all",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASE_CHOICES, default="all")
    parser.add_argument("--only", nargs="+", choices=DATASETS)
    parser.add_argument("--seed", type=int, default=0,
                        help="Persistence seed label (predictions go to seed=<n>.jsonl).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run cached phases.")
    parser.add_argument(
        "--early-stop", type=int, default=orchestrate.GENERATION_EARLY_STOP,
        help="Phase 1 early-stop threshold (upstream default).",
    )
    parser.add_argument(
        "--max-gen-docs", type=int, default=None,
        help="Hard cap on docs scanned in phase 1. Defensive backstop above "
             "early_stop --- when the model keeps proposing novel topics "
             "every few hundred docs (observed on GoEmotions), early_stop "
             "never fires and generation can run indefinitely. Caps the "
             "input file passed to generate_topic_lvl1.",
    )
    parser.add_argument(
        "--confirm-correct", action="store_true",
        help="Required to actually run phase 4. Without it, the runner prints "
             "the count of rows needing correction and exits before phase 4 "
             "fires --- the gate prevents an unexpectedly-large OpenAI Batch "
             "spend if assignment had an unusually high error rate.",
    )
    parser.add_argument(
        "--retry-correct", action="store_true",
        help="Reprompt only the rows that STILL fail to parse after a "
             "previous correction pass (reads corrected.jsonl instead of "
             "assignment.jsonl). Use when the prior batch hit gpt-5-mini's "
             "`max_completion_tokens` and we want to retry with a larger "
             "budget without redoing the rows that succeeded.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    names = args.only or DATASETS

    do_adapt          = args.phase in ("adapt", "all")
    do_generate       = args.phase in ("generate", "all")
    do_refine         = args.phase in ("refine", "all")
    do_assign         = args.phase in ("assign", "all")
    do_count_correct  = args.phase in ("count-correct", "all", "correct")
    do_correct        = args.phase in ("correct", "all")
    do_write          = args.phase in ("write", "all")

    for name in names:
        print(f"\n========== topicgpt / {name} ==========", flush=True)
        if do_adapt:
            res = dataset_adapter.adapt(name, force=args.overwrite)
            tail = (
                f"{res.n_truncated} truncated"
                if res.n_truncated >= 0
                else "cache hit"
            )
            print(f"[topicgpt/{name}/phase=adapt] {res.n_docs} docs | {tail}", flush=True)
        if do_generate:
            orchestrate.generate(
                name,
                overwrite=args.overwrite,
                verbose=args.verbose,
                early_stop=args.early_stop,
                max_gen_docs=args.max_gen_docs,
            )
        if do_refine:
            orchestrate.refine(name, overwrite=args.overwrite, verbose=args.verbose)
        if do_assign:
            batch_assigner.assign(name, overwrite=args.overwrite)
        if do_count_correct:
            counts = orchestrate.count_correction_targets(name)
            print(
                f"[topicgpt/{name}/phase=count-correct] "
                f"n_total={counts['n_total']} "
                f"n_error={counts['n_error']} "
                f"n_hallucinated={counts['n_hallucinated']} "
                f"n_to_correct={counts['n_to_correct']}",
                flush=True,
            )
            if do_correct and not args.confirm_correct:
                print(
                    f"[topicgpt/{name}/phase=correct] SKIPPED --- pass "
                    f"`--confirm-correct` to run the correction LLM "
                    f"({counts['n_to_correct']} gpt-5-mini OpenAI Batch calls). "
                    f"Or run `--phase write` directly to score off "
                    f"assignment.jsonl without correction.",
                    flush=True,
                )
                do_correct = False
                do_write = False
        if do_correct:
            if args.retry_correct:
                # Bypass orchestrate.correct's wrapper to pass the flag through
                # to batch_correct.
                from benchmarking.baselines.topicgpt import batch_correct
                batch_correct.correct(name, retry_from_corrected=True)
            else:
                orchestrate.correct(name, overwrite=args.overwrite, verbose=args.verbose)
        if do_write:
            jsonl, meta = result_parser.write(name, seed=args.seed)
            print(f"[topicgpt/{name}/phase=write] -> {jsonl} | {meta}", flush=True)


if __name__ == "__main__":
    main()
