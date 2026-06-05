"""Phase 3: per-doc classification via OpenAI Batch (gpt-5-mini).

Replaces upstream's ``given_label_classification.known_label_categorize``
sync loop. Same prompt template (``prompt_construct``, byte-identical to
upstream); different dispatch.

## Cache layout

Empirically, OpenAI's auto-cache anchors on the **system message** — a
long stable prefix inside a single user message does *not* cache even
when byte-identical across requests (observed: 0% cache hit on
Banking77 + topicgpt across MASSIVE-{Intent,Domain} for the same
single-user pattern). The fix, used by ``skills/corpus-tools/scripts/
classify.py`` which sees 49–94% cache hits on the same Batch API, is to
put the stable prefix in the ``system`` role and the per-doc variable
in the ``user`` role.

We render the upstream prompt **once with a placeholder sentence**, then
split on the ``Sentence:<placeholder>.`` line to lift the
``Sentence:`` line out into the user message. The instructions and
label list go into the system message; the ``return in JSON format``
spec follows them (so it moves from *after* ``Sentence:`` in upstream's
single-string render to *before* the user message here). The content
the LLM sees is byte-identical token-wise; only the role boundaries
shift. See ``CHANGES.md``.

For datasets where the merged label list is small enough that the
system message falls under 1024 tokens (e.g. MASSIVE-Domain with very
few merged labels), the auto-cache will not trigger and we pay full
input rate — a method+model limitation we can't avoid.

## Parsing

Upstream's ``answer_process`` uses ``eval()`` then string-membership
scans. We use ``json.loads`` and an exact-match-first lookup into the
merged label list. The first valid label substring match wins to mirror
upstream behaviour. Rows whose response is malformed or whose label name
is not in the merged list keep a sentinel ``"<unparseable>"`` /
``"<hallucinated>"`` token in the output JSONL --- ``result_parser`` maps
those to ``predicted_cluster_id = NONE_LABEL_ID`` so they show up in
partition metrics rather than being silently dropped from the
denominator (which is the upstream evaluate.py bug we don't reproduce).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import benchmarking  # noqa: F401 — truststore.inject_into_ssl()
import benchmarking.baselines.huang_he  # noqa: F401 — _vendored on sys.path

from benchmarking.baselines.huang_he import _batch_common as bc
from benchmarking.baselines.huang_he.dataset_adapter import HUANG_HE_ROOT
from benchmarking.baselines.huang_he.prompts import prompt_construct_classify

MODEL = "gpt-5-mini"

# Wrapping system-role message: short generic helper instruction that
# preserves the upstream ``"You are a helpful assistant designed to output
# JSON."`` voice. The dataset-specific instructions + label list +
# return-format spec are appended to this on a per-batch basis (see
# ``_build_system_message``) so the whole thing becomes the long stable
# prefix that anchors OpenAI's auto-cache.
SYSTEM_PREAMBLE = "You are a helpful assistant designed to output JSON."

RESPONSE_FORMAT = {"type": "json_object"}

# gpt-5-mini's max_completion_tokens covers reasoning + output. Visible
# output is one short ``{"label_name": "..."}`` JSON, but reasoning over
# a long merged label list (up to ~150 candidates on CLINC150) burns
# meaningful tokens. Matches the bump in ``batch_generate.MAX_COMPLETION_TOKENS``
# after the Banking77 pilot showed 11.7% of chunks exhausting the prior
# 1500-token cap on reasoning alone.
MAX_COMPLETION_TOKENS = 4000

# Placeholder we substitute into prompt_construct to get a stable
# cache prefix. Choose something unlikely to appear in any document.
_SENTENCE_SENTINEL = "<<HUANG_HE_SENTENCE_PLACEHOLDER>>"

# Sentinel labels emitted into classifications.jsonl when the LLM
# response can't be mapped to a real cluster. Picked to be guaranteed
# disjoint from any legitimate LLM-generated label.
UNPARSEABLE = "<unparseable>"
HALLUCINATED = "<hallucinated>"


@dataclass
class ClassificationResult:
    out_path: Path
    n_docs: int
    n_unparseable: int
    n_hallucinated: int
    usage: dict


def _load_inputs(dataset_name: str) -> tuple[list[dict], list[str]]:
    """Return (docs, merged_label_list)."""
    out_dir = HUANG_HE_ROOT / dataset_name
    docs_path = out_dir / "input.jsonl"
    labels_path = out_dir / "labels_merged.json"
    if not docs_path.exists():
        raise FileNotFoundError(
            f"phase 3 needs phase 0 output: {docs_path}. Run "
            f"`python -m benchmarking.baselines.huang_he.dataset_adapter "
            f"--only {dataset_name}`."
        )
    if not labels_path.exists():
        raise FileNotFoundError(
            f"phase 3 needs phase 2 output: {labels_path}. Run "
            f"`python -m benchmarking.experiments.run_huang_he --phase merge "
            f"--only {dataset_name}`."
        )
    docs = [
        json.loads(line)
        for line in docs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with labels_path.open(encoding="utf-8") as f:
        labels = json.load(f)
    return docs, labels


def _split_prompt_for_caching(merged_labels: list[str]) -> tuple[str, str]:
    """Render upstream prompt; split off the per-doc ``Sentence:`` line.

    Returns ``(system_message, user_sentence_prefix)`` such that:

    * ``system_message`` is the long stable prefix (instructions + label
      list + return-format spec), prepended with ``SYSTEM_PREAMBLE``.
      Identical across every request in the batch → caches.
    * ``user_sentence_prefix`` is the literal "Sentence:" prefix
      upstream uses immediately before the doc text. The full per-doc
      user message is ``user_sentence_prefix + <doc text> + "."``.

    Upstream's ``prompt_construct`` produces::

        Given the label list and the sentence, ...
        Label list: [...].
        Sentence:{sentence}.
        You should only return the label name, ...

    We split at ``Sentence:<sentinel>.`` and stitch the head + tail (minus
    the sentence line) into the system message. Content the LLM sees is
    byte-identical token-wise; only the role boundaries shift.
    """
    rendered = prompt_construct_classify(merged_labels, _SENTENCE_SENTINEL)
    marker = f"Sentence:{_SENTENCE_SENTINEL}."
    if marker not in rendered:
        raise RuntimeError(
            "prompt_construct unexpectedly did not include the "
            f"{marker!r} marker — upstream prompt format has changed."
        )
    head, tail = rendered.split(marker, maxsplit=1)
    # ``head`` ends with "Label list: [...].\n" (trailing newline from
    # upstream). ``tail`` starts with "\nYou should only return..." (the
    # trailing format-spec). Stitching head + tail's lstrip drops the
    # one-line gap left by the removed Sentence line.
    instructions_block = head + tail.lstrip("\n")
    system_message = f"{SYSTEM_PREAMBLE}\n{instructions_block}"
    user_sentence_prefix = "Sentence:"
    return system_message, user_sentence_prefix


def _parse_label_from_response(response_text: str, label_set: set[str]) -> str | None:
    """Extract a recognised label name from the model's JSON response.

    Returns the matched label string (one of ``label_set``) on success,
    None on failure. Mirrors upstream's ``answer_process``:

    * Try ``json.loads`` first (upstream uses ``eval``).
    * If the result is a dict, scan its values for an exact label match.
    * If the result is a string, scan for substring match.
    * Otherwise fall back to substring match on the raw response text.

    Upstream returned the first matched label in ``label_list`` iteration
    order, irrespective of the actual position of the match in the
    response. We preserve that ordering by iterating ``label_list``
    (passed in via the caller's ordered list, see ``classify``).
    """
    # Try strict JSON first.
    try:
        obj = json.loads(response_text)
    except json.JSONDecodeError:
        obj = None

    if isinstance(obj, dict):
        values = [str(v) for v in obj.values()]
        # Exact match first.
        for v in values:
            if v in label_set:
                return v
        # Then substring match (upstream's behaviour).
        for v in values:
            for lbl in label_set:
                if lbl in v:
                    return lbl
        return None

    haystack = response_text if obj is None else str(obj)
    for lbl in label_set:
        if lbl in haystack:
            return lbl
    return None


def _build_requests(
    docs: list[dict], system_message: str, user_sentence_prefix: str
) -> list[dict]:
    requests: list[dict] = []
    for doc in docs:
        user_text = f"{user_sentence_prefix}{doc['input']}."
        requests.append(
            bc.build_chat_request(
                custom_id=doc["doc_id"],
                user_text=user_text,
                model=MODEL,
                system_message=system_message,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                response_format=RESPONSE_FORMAT,
            )
        )
    return requests


def classify(dataset_name: str, *, overwrite: bool = False) -> ClassificationResult:
    """Phase 3 entry point."""
    out_dir = HUANG_HE_ROOT / dataset_name
    out_path = out_dir / "classifications.jsonl"
    usage_path = out_dir / "usage_classify.json"

    if not overwrite and out_path.exists() and out_path.stat().st_size > 0:
        with usage_path.open(encoding="utf-8") as f:
            usage = json.load(f)
        n = sum(1 for _ in out_path.open(encoding="utf-8"))
        print(
            f"[huang_he/{dataset_name}/phase=classify] cache hit -> {out_path} "
            f"(n_docs={n})",
            flush=True,
        )
        return ClassificationResult(
            out_path=out_path,
            n_docs=n,
            n_unparseable=int(usage.get("n_unparseable", 0)),
            n_hallucinated=int(usage.get("n_hallucinated", 0)),
            usage=usage,
        )

    docs, merged_labels = _load_inputs(dataset_name)
    if not merged_labels:
        raise RuntimeError(
            f"merged label list at {HUANG_HE_ROOT / dataset_name / 'labels_merged.json'} "
            f"is empty — cannot classify."
        )

    system_message, user_sentence_prefix = _split_prompt_for_caching(merged_labels)
    label_set = set(merged_labels)
    log_prefix = f"[huang_he/{dataset_name}/phase=classify]"
    print(
        f"{log_prefix} {len(docs)} docs | k_merged={len(merged_labels)} | "
        f"system_message={len(system_message)} chars (anchors auto-cache)",
        flush=True,
    )

    requests = _build_requests(docs, system_message, user_sentence_prefix)
    client = bc.get_openai_client()
    responses, usages, errors = bc.submit_and_collect(
        client,
        requests=requests,
        out_dir=out_dir,
        log_prefix=log_prefix,
        method_tag="huang_he-classify",
        dataset_name=dataset_name,
        batch_inputs_subdir="_classify_batch_inputs",
    )

    # Resolve every doc to a final label string (or a sentinel).
    n_unparseable = 0
    n_hallucinated = 0
    out_rows: list[str] = []
    for doc in docs:
        cid = doc["doc_id"]
        text = responses.get(cid)
        if text is None:
            # Batch-level failure (or missing custom_id in response).
            label = UNPARSEABLE
            n_unparseable += 1
        else:
            matched = _parse_label_from_response(text, label_set)
            if matched is None:
                # Distinguish "couldn't parse anything" from "parsed but
                # not in label_set" by sniffing whether json.loads worked.
                try:
                    obj = json.loads(text)
                except json.JSONDecodeError:
                    label = UNPARSEABLE
                    n_unparseable += 1
                else:
                    if obj is None:
                        label = UNPARSEABLE
                        n_unparseable += 1
                    else:
                        label = HALLUCINATED
                        n_hallucinated += 1
            else:
                label = matched
        out_rows.append(
            json.dumps(
                {
                    "doc_id": doc["doc_id"],
                    "input": doc["input"],
                    "gold_label_id": doc["gold_label_id"],
                    "is_none": doc["is_none"],
                    "raw_response": responses.get(cid, ""),
                    "predicted_label_name": label,
                },
                ensure_ascii=False,
            )
        )

    out_path.write_text("\n".join(out_rows) + "\n", encoding="utf-8")

    usage = bc.summarize_usage(usages)
    usage.update(
        {
            "model": MODEL,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "n_docs": len(docs),
            "n_batch_errors": len(errors),
            "n_unparseable": n_unparseable,
            "n_hallucinated": n_hallucinated,
            "k_merged": len(merged_labels),
        }
    )
    usage_path.write_text(json.dumps(usage, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"{log_prefix} -> {out_path} | n_docs={len(docs)} "
        f"n_unparseable={n_unparseable} n_hallucinated={n_hallucinated} | "
        f"usage={usage}",
        flush=True,
    )
    return ClassificationResult(
        out_path=out_path,
        n_docs=len(docs),
        n_unparseable=n_unparseable,
        n_hallucinated=n_hallucinated,
        usage=usage,
    )
