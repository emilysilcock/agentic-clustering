#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "anthropic>=0.42.0",
#   "openai>=1.50.0",
# ]
# ///
"""Classify texts into the cluster taxonomy.

Reads a system prompt (built by build_classification_prompt.py) and a corpus,
runs each text through the chosen provider/model, and writes per-text results
to CSV. Supports two providers (anthropic, openai) and two execution modes
(async with concurrency cap; batch via Anthropic Messages Batches API for 50%
cost reduction). Anthropic prompt caching is enabled by default.

Usage:
    uv run classify.py \\
        --input corpus.csv --text-col text \\
        --prompt classification/prompt.md \\
        --output classifications/run.csv \\
        --provider anthropic --model claude-haiku-4-5 \\
        --mode async --concurrency 20
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-5-mini",
}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_corpus(path: Path, text_col: str, id_col: str) -> list[dict]:
    if not path.exists():
        print(f"error: input not found: {path}", file=sys.stderr)
        sys.exit(1)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_csv(path, text_col, id_col)
    if suffix == ".json":
        return _load_json(path, text_col, id_col)
    print(f"error: unsupported format {suffix} (use .csv or .json)", file=sys.stderr)
    sys.exit(1)


def _load_csv(path: Path, text_col: str, id_col: str) -> list[dict]:
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if text_col not in fields:
            print(f"error: column '{text_col}' not in CSV. Available: {fields}", file=sys.stderr)
            sys.exit(1)
        has_id = id_col in fields
        for i, row in enumerate(reader):
            text = (row[text_col] or "").strip()
            if not text:
                continue
            tid = str(row[id_col]).strip() if has_id else str(i)
            out.append({"id": tid, "text": text})
    return out


def _load_json(path: Path, text_col: str, id_col: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("error: JSON must be a list of objects", file=sys.stderr)
        sys.exit(1)
    out = []
    for i, item in enumerate(data):
        if isinstance(item, dict) and text_col in item:
            text = str(item[text_col]).strip()
            if not text:
                continue
            tid = str(item.get(id_col, i))
            out.append({"id": tid, "text": text})
        elif isinstance(item, str):
            out.append({"id": str(i), "text": item.strip()})
    return out


# ---------------------------------------------------------------------------
# Schema construction
# ---------------------------------------------------------------------------

def extract_cluster_ids(prompt: str) -> list[str]:
    """Pull cluster IDs from `## Name (`<id>`) ...` lines in the prompt."""
    ids = []
    for line in prompt.split("\n"):
        if line.startswith("## "):
            m = re.search(r"`([^`]+)`", line)
            if m:
                ids.append(m.group(1))
    return ids


def build_schema(cluster_ids: list[str]) -> dict:
    """Strict JSON schema enforced at decode time on both providers."""
    return {
        "type": "object",
        "properties": {
            "cluster": {"type": "string", "enum": [*cluster_ids, "none"]},
            "confidence": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
            "reasoning": {"type": "string"},
        },
        "required": ["cluster", "confidence", "reasoning"],
        "additionalProperties": False,
    }


# ---------------------------------------------------------------------------
# Anthropic — async
# ---------------------------------------------------------------------------

async def classify_anthropic_async(
    model: str,
    prompt: str,
    schema: dict,
    records: list[dict],
    concurrency: int,
) -> dict[str, dict]:
    import anthropic

    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(concurrency)
    results: dict[str, dict] = {}

    async def one(rec: dict) -> None:
        async with sem:
            try:
                resp = await client.messages.create(
                    model=model,
                    max_tokens=512,
                    system=[{
                        "type": "text",
                        "text": prompt,
                        "cache_control": {"type": "ephemeral"},
                    }],
                    messages=[{"role": "user", "content": rec["text"]}],
                    output_config={
                        "format": {"type": "json_schema", "schema": schema},
                    },
                )
                text = next(b.text for b in resp.content if b.type == "text")
                parsed = json.loads(text)
                results[rec["id"]] = {
                    **parsed,
                    "error": None,
                    "input_tokens": resp.usage.input_tokens,
                    "cache_read_tokens": resp.usage.cache_read_input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                }
            except Exception as e:
                results[rec["id"]] = {
                    "cluster": None, "confidence": None, "reasoning": None,
                    "error": f"{type(e).__name__}: {e}",
                    "input_tokens": 0, "cache_read_tokens": 0, "output_tokens": 0,
                }

    total = len(records)
    done = 0
    tasks = [asyncio.create_task(one(r)) for r in records]
    for t in asyncio.as_completed(tasks):
        await t
        done += 1
        if done % 50 == 0 or done == total:
            print(f"  classified {done}/{total}", file=sys.stderr)
    return results


# ---------------------------------------------------------------------------
# Anthropic — batch
# ---------------------------------------------------------------------------

async def classify_anthropic_batch(
    model: str,
    prompt: str,
    schema: dict,
    records: list[dict],
) -> dict[str, dict]:
    import anthropic

    client = anthropic.AsyncAnthropic()

    requests = [
        {
            "custom_id": rec["id"],
            "params": {
                "model": model,
                "max_tokens": 512,
                "system": [{
                    "type": "text",
                    "text": prompt,
                    "cache_control": {"type": "ephemeral"},
                }],
                "messages": [{"role": "user", "content": rec["text"]}],
                "output_config": {
                    "format": {"type": "json_schema", "schema": schema},
                },
            },
        }
        for rec in records
    ]

    print(f"  submitting batch of {len(requests)} requests...", file=sys.stderr)
    batch = await client.messages.batches.create(requests=requests)
    print(f"  batch id: {batch.id}", file=sys.stderr)

    while True:
        b = await client.messages.batches.retrieve(batch.id)
        counts = b.request_counts
        print(
            f"  status={b.processing_status} "
            f"processing={counts.processing} succeeded={counts.succeeded} "
            f"errored={counts.errored}",
            file=sys.stderr,
        )
        if b.processing_status == "ended":
            break
        await asyncio.sleep(60)

    results: dict[str, dict] = {}
    async for r in await client.messages.batches.results(batch.id):
        if r.result.type == "succeeded":
            msg = r.result.message
            text = next(blk.text for blk in msg.content if blk.type == "text")
            parsed = json.loads(text)
            results[r.custom_id] = {
                **parsed,
                "error": None,
                "input_tokens": msg.usage.input_tokens,
                "cache_read_tokens": msg.usage.cache_read_input_tokens,
                "output_tokens": msg.usage.output_tokens,
            }
        else:
            err = getattr(r.result, "error", r.result.type)
            results[r.custom_id] = {
                "cluster": None, "confidence": None, "reasoning": None,
                "error": str(err),
                "input_tokens": 0, "cache_read_tokens": 0, "output_tokens": 0,
            }
    return results


# ---------------------------------------------------------------------------
# OpenAI — async
# ---------------------------------------------------------------------------

async def classify_openai_async(
    model: str,
    prompt: str,
    schema: dict,
    records: list[dict],
    concurrency: int,
) -> dict[str, dict]:
    import openai

    client = openai.AsyncOpenAI()
    sem = asyncio.Semaphore(concurrency)
    results: dict[str, dict] = {}

    async def one(rec: dict) -> None:
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": rec["text"]},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "classification",
                            "strict": True,
                            "schema": schema,
                        },
                    },
                )
                text = resp.choices[0].message.content
                parsed = json.loads(text or "{}")
                usage = resp.usage
                cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
                results[rec["id"]] = {
                    **parsed,
                    "error": None,
                    "input_tokens": usage.prompt_tokens,
                    "cache_read_tokens": cached,
                    "output_tokens": usage.completion_tokens,
                }
            except Exception as e:
                results[rec["id"]] = {
                    "cluster": None, "confidence": None, "reasoning": None,
                    "error": f"{type(e).__name__}: {e}",
                    "input_tokens": 0, "cache_read_tokens": 0, "output_tokens": 0,
                }

    total = len(records)
    done = 0
    tasks = [asyncio.create_task(one(r)) for r in records]
    for t in asyncio.as_completed(tasks):
        await t
        done += 1
        if done % 50 == 0 or done == total:
            print(f"  classified {done}/{total}", file=sys.stderr)
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_output(records: list[dict], results: dict[str, dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id", "cluster", "confidence", "reasoning", "error",
        "input_tokens", "cache_read_tokens", "output_tokens",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rec in records:
            r = results.get(rec["id"], {})
            w.writerow({
                "id": rec["id"],
                "cluster": r.get("cluster"),
                "confidence": r.get("confidence"),
                "reasoning": r.get("reasoning"),
                "error": r.get("error"),
                "input_tokens": r.get("input_tokens", 0),
                "cache_read_tokens": r.get("cache_read_tokens", 0),
                "output_tokens": r.get("output_tokens", 0),
            })


def summarize(results: dict[str, dict]) -> None:
    n = len(results)
    n_err = sum(1 for r in results.values() if r.get("error"))
    in_t = sum(r.get("input_tokens", 0) for r in results.values())
    cache_t = sum(r.get("cache_read_tokens", 0) for r in results.values())
    out_t = sum(r.get("output_tokens", 0) for r in results.values())
    print(file=sys.stderr)
    print(f"summary: {n} classified, {n_err} errors", file=sys.stderr)
    print(
        f"tokens:  input={in_t:,} (cached={cache_t:,}, "
        f"{cache_t / max(in_t, 1):.0%}) output={out_t:,}",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Path to corpus CSV or JSON")
    p.add_argument("--text-col", default="text", help="Text column name (default: text)")
    p.add_argument("--id-col", default="id", help="ID column name (default: id)")
    p.add_argument("--prompt", required=True, help="Path to classification prompt (built by build_classification_prompt.py)")
    p.add_argument("--output", required=True, help="Path to write per-text classifications CSV")
    p.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    p.add_argument("--model", help=f"Model ID (default: {DEFAULT_MODELS})")
    p.add_argument("--mode", choices=["async", "batch"], default="async",
                   help="async = real-time with concurrency cap; batch = Anthropic Messages Batches API (50%% cheaper, ≤24h)")
    p.add_argument("--concurrency", type=int, default=20, help="Async mode only (default: 20)")
    args = p.parse_args()

    if args.mode == "batch" and args.provider != "anthropic":
        print("error: --mode batch is only supported for --provider anthropic", file=sys.stderr)
        return 1

    if args.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1
    if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("error: OPENAI_API_KEY not set", file=sys.stderr)
        return 1

    model = args.model or DEFAULT_MODELS[args.provider]

    prompt_path = Path(args.prompt)
    if not prompt_path.exists():
        print(f"error: prompt not found: {prompt_path}", file=sys.stderr)
        return 1
    prompt = prompt_path.read_text(encoding="utf-8")
    cluster_ids = extract_cluster_ids(prompt)
    if not cluster_ids:
        print("error: no cluster IDs found in prompt (expected `## Name (`cN`)` headings)", file=sys.stderr)
        return 1
    schema = build_schema(cluster_ids)

    records = load_corpus(Path(args.input), args.text_col, args.id_col)
    print(
        f"classifying {len(records)} texts: provider={args.provider} "
        f"model={model} mode={args.mode} clusters={len(cluster_ids)}",
        file=sys.stderr,
    )

    t0 = time.time()
    if args.provider == "anthropic" and args.mode == "async":
        results = asyncio.run(classify_anthropic_async(model, prompt, schema, records, args.concurrency))
    elif args.provider == "anthropic" and args.mode == "batch":
        results = asyncio.run(classify_anthropic_batch(model, prompt, schema, records))
    elif args.provider == "openai":
        results = asyncio.run(classify_openai_async(model, prompt, schema, records, args.concurrency))
    else:
        print("error: unsupported provider/mode combination", file=sys.stderr)
        return 1
    elapsed = time.time() - t0

    write_output(records, results, Path(args.output))
    summarize(results)
    print(f"  elapsed: {elapsed:.1f}s", file=sys.stderr)
    print(f"  wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
