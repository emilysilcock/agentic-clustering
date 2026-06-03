"""Shared taxonomy.md regex helpers.

Both `build_classification_prompt.py` (strips Examples blocks bounded by
cluster headers) and `classify.py` (extracts cluster ids to build the JSON
schema enum) need to recognise the same line shape. Keeping the regex here
makes the two paths agree by construction — if it drifts in one place, the
prompt the model sees no longer names the same set of ids the schema
enforces.

Stdlib only (no third-party deps); safe to import from any PEP 723 script in
this directory.
"""

from __future__ import annotations

import re

# Matches taxonomy.md cluster headers, e.g. "## Some Name (`c12`) [high]".
# The capture group is the cluster id. Two anchoring tricks:
# 1. Only `cN` (digits-only after `c`) — a backticked token inside the cluster
#    name (e.g. "## Use of `npm` commands (`c5`) [high]") can't be mistaken
#    for the id.
# 2. Require the trailing confidence label `[...]` that state.py:cmd_finalize
#    always emits — pins the captured id to the *last* `(`cN`)` before `[`,
#    so a hypothetical name containing a literal `(`cN`)` can't shadow the
#    real id via greedy `.*`. If a future change ever drops the [conf]
#    suffix, classify.py's `if not cluster_ids` guard fails loudly rather
#    than silently producing a wrong-enum schema.
CLUSTER_HEADER_RE = re.compile(r"^##\s+.*\(`(c\d+)`\)\s*\[[^\]]+\]\s*$")
