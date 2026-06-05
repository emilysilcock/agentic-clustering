"""Phase 2: merge duplicate / near-duplicate labels via Opus 4.7.

Single LLM call per dataset (seven calls per full sweep --- well under
the SPEC §5.6 >1,000-text threshold, so it routes to the frontier tier
via the Claude Code Max subscription rather than the OpenAI Batch API).

Same prompt template as upstream (``prompt_construct_merge_label``,
byte-identical); different model and dispatch. Upstream uses
``gpt-3.5-turbo-0125`` with ``response_format={"type":"json_object"}``;
we use ``claude-opus-4-7`` and post-validate the JSON by hand, since
``claude -p`` doesn't expose OpenAI's response_format gate.

## Robustness around the LLM response

The merge prompt asks for ``{"merged_labels": [...]}`` but the model may
wrap it in markdown code fences or include a short preamble. We strip
fences, find the first top-level JSON object, parse it, and extract the
first list-valued field (mirroring upstream's
``response[list(response.keys())[0]]``). If parsing fails (extremely
rare on Opus for this prompt), we fall back to the pre-merge list ---
this is upstream's behaviour too (``label_generation.merge_labels``
returns the input list on parse failure).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from benchmarking.baselines.huang_he.dataset_adapter import HUANG_HE_ROOT
from benchmarking.baselines.huang_he.prompts import prompt_construct_merge_label
from benchmarking.llm_clients.claude_code import call_claude

MODEL = "claude-opus-4-7"

# Opus 4.7 via `claude -p` doesn't bill on per-call API tokens (Max
# subscription), so we don't need a max_tokens cap. We do need a wall-clock
# timeout — at Banking77's ~2261 pre-merge labels Opus took 97s; bigger
# datasets (GoEmotions, StackExchange) will likely run 5–10× longer
# pre-merge lists, so 20 min keeps comfortable headroom without disguising
# real hangs.
TIMEOUT_S = 20 * 60


@dataclass
class MergeResult:
    out_path: Path
    n_pre_merge: int
    n_merged: int
    parse_ok: bool
    usage: dict


# Match the first top-level {...} block in the response — tolerates
# preambles and code fences. Non-greedy on { so we don't eat trailing
# narration after a valid JSON object.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_merge_response(text: str) -> list[str] | None:
    """Pull a flat list of label names out of the model's response.

    Mirrors upstream's `merge_labels` extractor:
    ``response[list(response.keys())[0]]`` is the first list-valued
    field. Returns None on parse failure.
    """
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        return None
    blob = match.group(0)
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not obj:
        return None
    first_value = obj[next(iter(obj))]
    # Upstream's merge_labels iterates `for key, sub_label_list in
    # response.items()` --- it can handle a dict-of-list-of-list too.
    # Flatten one level if the first value is a list of lists.
    if isinstance(first_value, list):
        flat: list[str] = []
        for elem in first_value:
            if isinstance(elem, list):
                flat.extend(str(x) for x in elem)
            else:
                flat.append(str(elem))
        # Dedupe preserving order.
        out: list[str] = []
        seen: set[str] = set()
        for label in flat:
            if label and label not in seen:
                seen.add(label)
                out.append(label)
        return out
    return None


def merge(dataset_name: str, *, overwrite: bool = False) -> MergeResult:
    """Phase 2 entry point. Runs a single sync Opus 4.7 call."""
    out_dir = HUANG_HE_ROOT / dataset_name
    pre_merge_path = out_dir / "labels_pre_merge.json"
    out_path = out_dir / "labels_merged.json"
    usage_path = out_dir / "usage_merge.json"
    raw_path = out_dir / "_merge_raw_response.txt"

    if not overwrite and out_path.exists() and out_path.stat().st_size > 0:
        with out_path.open(encoding="utf-8") as f:
            labels = json.load(f)
        with usage_path.open(encoding="utf-8") as f:
            usage = json.load(f)
        print(
            f"[huang_he/{dataset_name}/phase=merge] cache hit -> {out_path} "
            f"(n_merged={len(labels)})",
            flush=True,
        )
        return MergeResult(
            out_path=out_path,
            n_pre_merge=int(usage.get("n_pre_merge", 0)),
            n_merged=len(labels),
            parse_ok=bool(usage.get("parse_ok", True)),
            usage=usage,
        )

    if not pre_merge_path.exists():
        raise FileNotFoundError(
            f"phase 2 needs phase 1 output: {pre_merge_path}. Run "
            f"`python -m benchmarking.experiments.run_huang_he --phase generate "
            f"--only {dataset_name}`."
        )

    with pre_merge_path.open(encoding="utf-8") as f:
        pre_merge_labels = json.load(f)

    prompt = prompt_construct_merge_label(pre_merge_labels)
    log_prefix = f"[huang_he/{dataset_name}/phase=merge]"
    print(
        f"{log_prefix} n_pre_merge={len(pre_merge_labels)} model={MODEL} "
        f"(via Claude Code Max subscription)",
        flush=True,
    )

    t0 = time.perf_counter()
    raw = call_claude(
        prompt,
        model=MODEL,
        timeout_s=TIMEOUT_S,
        log_prefix=log_prefix,
    )
    elapsed = time.perf_counter() - t0

    raw_path.write_text(raw or "", encoding="utf-8")
    merged = _parse_merge_response(raw or "")
    parse_ok = merged is not None
    if not parse_ok:
        # Upstream's fallback: return the pre-merge list verbatim.
        merged = list(pre_merge_labels)
        print(
            f"{log_prefix} WARN: response failed to parse; falling back to "
            f"pre-merge list ({len(merged)} labels). Raw response in {raw_path}",
            flush=True,
        )

    out_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    usage = {
        "model": MODEL,
        "subscription_billed": True,
        "wall_clock_s": elapsed,
        "n_pre_merge": len(pre_merge_labels),
        "n_merged": len(merged),
        "parse_ok": parse_ok,
        # The Max subscription is billed as a flat monthly cost, not metered
        # tokens. We don't have per-call token counts from `claude -p` and
        # don't try to estimate them; subscription_usd accounting lives in
        # result_parser.write (split flat across the 7 datasets per
        # SPEC §5.6.3).
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    usage_path.write_text(json.dumps(usage, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"{log_prefix} -> {out_path} | n_pre_merge={len(pre_merge_labels)} "
        f"n_merged={len(merged)} parse_ok={parse_ok} wall_clock={elapsed:.1f}s",
        flush=True,
    )

    return MergeResult(
        out_path=out_path,
        n_pre_merge=len(pre_merge_labels),
        n_merged=len(merged),
        parse_ok=parse_ok,
        usage=usage,
    )
