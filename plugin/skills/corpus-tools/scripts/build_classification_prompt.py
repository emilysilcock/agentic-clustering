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

# Force UTF-8 on stdout/stderr — Windows defaults to cp1252 and crashes on
# non-ASCII cluster names / corpus content. Idempotent; no-op on streams that
# aren't TextIOWrapper (e.g. captured in tests).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Shared with classify.py via _taxonomy. Cluster-header detection MUST agree
# across the two scripts — drift means the prompt the model sees names a
# different set of ids than the schema enum it must produce.
from _taxonomy import CLUSTER_HEADER_RE


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

FORCE_ASSIGN_HEADER = """\
You are a text classifier. Read the input text and assign it to exactly one of \
the clusters defined below. Every text must be assigned to a cluster.

"""

FORCE_ASSIGN_FOOTER = """

INSTRUCTIONS:
- Assign exactly ONE cluster ID from those defined above. Every text must be assigned.
- If no cluster is a perfect fit, pick the closest one and lower the confidence.
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
    next cluster header (or end of input), exclusive of the header. The block is
    delimited by the cluster header rather than a '---' divider because example
    texts frequently contain '----' signature lines (and occasionally '## '
    markdown) that would otherwise end the block early and leak example text into
    the prompt. A single '---' divider is re-emitted before the next header to
    preserve the visual separation between cluster definitions."""
    out = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == "**Examples:**":
            i += 1
            while i < len(lines) and not CLUSTER_HEADER_RE.match(lines[i]):
                i += 1
            if i < len(lines):  # stopped on a cluster header, not EOF
                out.append("---")
                out.append("")
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
    p.add_argument(
        "--force-assign",
        action="store_true",
        help=(
            "Forbid 'none' as a classification output. Use for datasets whose gold "
            "labels don't include an OOS/none class — every text must be assigned "
            "to a real cluster. Must be paired with classify.py --force-assign so "
            "the JSON schema also rejects 'none'."
        ),
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
    elif args.force_assign:
        header = FORCE_ASSIGN_HEADER
    else:
        header = DEFAULT_HEADER

    footer = FORCE_ASSIGN_FOOTER if args.force_assign else DEFAULT_FOOTER
    prompt = header + cluster_definitions + footer

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prompt, encoding="utf-8")

    print(f"wrote {output_path} ({len(prompt)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
