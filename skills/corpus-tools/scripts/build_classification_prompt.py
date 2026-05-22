#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Build a classification system prompt from taxonomy.md.

Strips:
  - The metadata header (everything before the first "## " line)
  - The "**Examples:**" block from each cluster section
    (between "**Examples:**" and the next "---" or "## " line)

Wraps the cleaned cluster definitions with a header (instructions) and a
footer (output guidance). The JSON output schema is enforced by the
classifier at call time, not by the prompt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_HEADER = """\
You are a text classifier. Read the input text and assign it to exactly one of \
the clusters defined below, or "none" if no cluster fits.

"""

DEFAULT_FOOTER = """

INSTRUCTIONS:
- Assign exactly ONE cluster ID from those defined above, or "none" if no cluster fits.
- Use any boundary notes in the cluster descriptions to resolve ambiguous cases.
- Confidence is an integer 1-5: 1 = very uncertain, 5 = very confident.
- Reasoning should be one short sentence explaining the assignment.
"""


def strip_metadata_header(lines: list[str]) -> list[str]:
    for i, line in enumerate(lines):
        if line.startswith("## "):
            return lines[i:]
    return lines


def strip_example_blocks(lines: list[str]) -> list[str]:
    """Remove '**Examples:**' blocks. A block runs from the marker line to the
    next '---' or '## ' (cluster boundary), exclusive of the boundary."""
    out = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == "**Examples:**":
            i += 1
            while i < len(lines) and not (
                lines[i].startswith("---") or lines[i].startswith("## ")
            ):
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--taxonomy", required=True, help="Path to taxonomy.md")
    p.add_argument("--output", required=True, help="Path to write the prompt to")
    p.add_argument(
        "--header",
        help="Optional path to a file containing a custom header (replaces the default)",
    )
    p.add_argument(
        "--keep-examples",
        action="store_true",
        help="Keep the **Examples:** blocks in the prompt (default: strip them)",
    )
    args = p.parse_args()

    taxonomy_path = Path(args.taxonomy)
    if not taxonomy_path.exists():
        print(f"error: taxonomy not found: {taxonomy_path}", file=sys.stderr)
        return 1

    text = taxonomy_path.read_text(encoding="utf-8")
    lines = text.split("\n")

    lines = strip_metadata_header(lines)
    if not args.keep_examples:
        lines = strip_example_blocks(lines)

    cluster_definitions = "\n".join(lines).strip() + "\n"

    if args.header:
        header = Path(args.header).read_text(encoding="utf-8")
        if not header.endswith("\n\n"):
            header = header.rstrip() + "\n\n"
    else:
        header = DEFAULT_HEADER

    prompt = header + cluster_definitions + DEFAULT_FOOTER

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prompt, encoding="utf-8")

    print(f"wrote {output_path} ({len(prompt)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
