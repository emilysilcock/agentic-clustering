# Vendored TopicGPT source — edits from upstream

Source: `chtmp223/topicGPT` @ commit `e6499ef7f3db8bba2bcff30e4d636b11d2acc97d`
(`main` as of 2026-05-15). Upstream is unlicensed; redistributed inside this
research repo for reproducibility, not as a standalone artifact.

## Files vendored

Only the subset of `topicgpt_python/` we actually drive from
`benchmarking/baselines/topicgpt/orchestrate.py`:

- `topicgpt_python/__init__.py`
- `topicgpt_python/utils.py` — `APIClient`, `TopicTree`, metric helpers
- `topicgpt_python/generation_1.py` — Phase 1 (topic discovery, sequential
  document scan)
- `topicgpt_python/refinement.py` — Phase 2 (merge near-duplicate topics, drop
  low-frequency ones)
- `topicgpt_python/assignment.py` — sync per-doc assignment (vendored for
  completeness; the actual Phase 3 driver is our `batch_assigner.py`, which
  goes through the Anthropic Batch API for the per-doc cost reduction
  required by SPEC §5.6.2)
- `topicgpt_python/correction.py` — Phase 4 (reassign hallucinated topic
  names; small N, sync calls)
- `topicgpt_python/data_sample.py`

Plus the prompt templates, copied verbatim into
`benchmarking/baselines/topicgpt/prompts/`:

- `prompt/generation_1.txt`
- `prompt/refinement.txt`
- `prompt/assignment.txt`
- `prompt/correction.txt`

**Not vendored** (explicitly omitted):

- `topicgpt_python/generation_2.py` — second-level hierarchical topic
  generation. SPEC §5.5 / §5.1.1 fix depth=1 (flat taxonomies), so this
  branch is dead code for us.
- `topicgpt_python/metrics.py` — upstream's metrics module. We use
  `benchmarking/evaluation/metrics.py` (ARI / NMI / ACC with Hungarian
  alignment, matching every other baseline in this repo).
- `prompt/generation_2.txt` — paired with the unused `generation_2.py`.
- `prompt/seed_1.md` — seed topic list. Discover-k means we pass an empty
  seed file, so the example one isn't useful.

## Edits

### `__init__.py`

Removed `from .generation_2 import generate_topic_lvl2`, since
`generation_2.py` is not vendored.

### `generation_1.py`, `refinement.py`, `correction.py`, `assignment.py`,
### `data_sample.py`

One-line import fix on each: replaced

```python
from topicgpt_python.utils import *
```

with

```python
from .utils import *
```

so the vendored package is self-contained and doesn't require a
`pip install topicgpt_python` for our top-level absolute import to resolve.
This matches the pattern used in
`benchmarking/baselines/clusterllm/_vendored/`.

### `utils.py`

Three additive edits, all marked in the file with comments beginning
"Vendored addition":

1. **Guarded optional-provider imports.** The original imports
   `vertexai`, `anthropic.AnthropicVertex`, and `google.generativeai`
   unconditionally at module load, so the file fails to import unless
   every backend's SDK is installed. We wrap each in
   `try: ... except ImportError: _HAS_X = False`, then check the flag
   inside the relevant branch of `__init__` and `iterative_prompt`. No
   change to the scientific logic of the existing branches.

2. **New `claude_code` provider.** Added a branch to `APIClient.__init__`
   (no-op; the subprocess wrapper is stateless) and to
   `iterative_prompt` that dispatches to
   `benchmarking.llm_clients.claude_code.call_claude(...)`. Used for
   Phase 1 (`generate_topic_lvl1`) and Phase 2 (`refine_topics`), per
   SPEC §5.6.2 — Opus 4.7 via the Claude Code Max subscription, not
   metered.

3. **New `anthropic` provider.** Added a branch to `APIClient.__init__`
   (constructs an `anthropic.Anthropic()` client from
   `ANTHROPIC_API_KEY`) and to `iterative_prompt` that uses the metered
   sync Messages API. Used for Phase 4 (`correct_topics`) only — Haiku
   4.5 on the small number of `error` / `hallucinated` rows. **Per-doc
   assignment (Phase 3) does NOT route through this branch**; it goes
   through `benchmarking/baselines/topicgpt/batch_assigner.py`, which
   submits to the Anthropic Batch API with prompt caching to halve the
   per-token cost (SPEC §5.6.2, §5.6.3 — "Batch API used for all bulk
   LLM/embedding operations").

4. **Per-call usage tracking.** Added `self.usage` dict (input_tokens,
   output_tokens, cache_read_input_tokens, cache_creation_input_tokens,
   n_calls) to `APIClient.__init__`, populated inside the
   `claude_code`, `anthropic`, and `openai` branches of
   `iterative_prompt`. Read by `orchestrate.py` to populate the SPEC
   §5.11 meta.json `cost` field. For the `openai` branch we follow the
   convention from `skills/corpus-tools/scripts/classify.py`:
   `input_tokens = prompt_tokens - cached_tokens` so the field means
   the same thing across providers (non-cached billable portion).

5. **`openai` branch parameter handling.** `temperature` and `top_p` are
   now passed only when the model is *not* a `gpt-5*` / `o1` / `o3`
   variant — these refuse or silently ignore custom sampling params.
   Matches the convention in `skills/corpus-tools/scripts/classify.py`.

### `utils.py` — `TopicTree.to_file` encoding fix

Upstream opens the topic-output `.md` with the platform default
encoding, which is `cp1252` on Windows. LLM-generated topic descriptions
frequently contain characters outside that codec (observed in pilot:
`‑` non-breaking hyphen, `–` en-dash, smart quotes), causing a hard
`UnicodeEncodeError` on write that the surrounding code does not catch.
We add `encoding="utf-8"` to the `open(...)` call. Pure-platform fix;
no change to what gets written.

### `generation_1.py` — per-doc checkpoint in `generate_topics`

Upstream writes the topic file and the responses jsonl *only* in the
outer `generate_topic_lvl1` after the inner loop returns. A SIGTERM
mid-loop loses everything in memory --- on GoEmotions, that cost ~3.5
hours of Opus subscription compute (~990 of ~29k docs scanned before
the kill). Added optional `df_full`, `out_file`, `topic_file` params to
`generate_topics` and an internal `_checkpoint()` helper that rewrites
both files after every successful iteration, on `early_stop` exit, on
`KeyboardInterrupt`, and in the `except Exception` branch. Cost: O(N)
file rewrites of a small (~5 KB) markdown and a growing jsonl --- well
below the per-call latency. If the new params are not provided, the
function behaves byte-identically to upstream.

## What was NOT touched

All scientific logic (`TopicTree`, `prompt_formatting`, `generate_topics`
loop, `refine_topics` merging logic, `correct_topics` hallucination
handling, the SBERT context-shrink path) is byte-identical to upstream.
Any difference between our reported numbers and the paper is
attributable to (a) the LLM swap to `claude-opus-4-7` / `gpt-5-mini`,
(b) the paper-recommended `early_stop=200` (rather than the vendored
code default 1000), (c) the depth-1 flat-taxonomy re-port to
short-text intent/emotion data, and (c) the 512-token document cap from
the dataset adapter — not to changes here.
