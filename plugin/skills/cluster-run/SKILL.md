---
name: cluster-run
description: >
  Run the agentic cluster discovery workflow. Iteratively discovers natural
  clusters in a text corpus using proposer, synthesizer, auditor, investigator,
  and critic subagents.
allowed-tools: Task, Read, Bash, Write
---

# Cluster Discovery Orchestration

You are orchestrating an iterative cluster discovery process.

## Environment

Scripts are at `$CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/`. This
environment variable is set automatically by Claude Code and expands when
you run Bash commands — pass it through as-is.

**Verify before first script call**: `$CLAUDE_PLUGIN_ROOT` and
`$CLUSTERING_WORKSPACE` can be empty in some subagent contexts. Before running
any script, verify both resolve:
```bash
if [ -z "$CLAUDE_PLUGIN_ROOT" ]; then
  export CLAUDE_PLUGIN_ROOT=$(cat .claude/clustering/.plugin_root 2>/dev/null)
fi
if [ -z "$CLUSTERING_WORKSPACE" ]; then
  export CLUSTERING_WORKSPACE=$(cat .claude/clustering/.active_workspace 2>/dev/null || echo .claude/clustering)
fi
```
Run this check once at the start of orchestration. `init.py` writes
`.claude/clustering/.plugin_root` and `.claude/clustering/.active_workspace` at
fixed project-root-relative locations (so the cats above work for custom
workspaces too). If `$CLAUDE_PLUGIN_ROOT` and its pointer file are both missing,
that's a Claude Code configuration problem, not something this skill should
paper over silently — fail and surface it.

The workspace directory defaults to `.claude/clustering/` but can be overridden
via the `CLUSTERING_WORKSPACE` environment variable or `--workspace` flag on
init.

## Session Start

### 1. Check for existing workspace

Scan for `state.json` in these locations (in order):
- `CLUSTERING_WORKSPACE` env var (if set)
- `$(cat .claude/clustering/.active_workspace)/state.json` (the pointer file
  init.py writes — works for custom workspaces)
- `.claude/clustering/state.json` (default location)
- `./clustering/state.json`

If found, read the state file and show the user a brief status:
> "Found existing clustering workspace at `<path>` — corpus: `<corpus_path>`
> (`<size>` texts), `<n>` clusters, coverage: `<coverage>`."

Then ask: **"Continue this session or start fresh?"**
- **Continue** → set `CLUSTERING_WORKSPACE` to the found path, read `summary.md`
  and `plan.md`, continue from where things left off.
- **Start fresh** → move the existing workspace to `<path>.archive.<timestamp>/`
  and proceed with new run setup below.

### 2. New run setup

For new runs (no existing workspace, or user chose "start fresh"), ask up front:

1. **Output directory** — "Where should the clustering workspace live?" Suggest
   the project root as default (e.g., `./clustering/`). The user might want it
   alongside their data, in a specific output folder, etc. This is asked early
   because it determines where everything goes.
2. **Corpus path** — CSV/JSON file
3. **Text column name**
4. **Target cluster count range** (k_range)
5. **Clustering instructions** (optional but highly recommended — how should
   clusters be defined? e.g., "cluster by issue type", "group by sentiment and
   topic", "focus on actionable categories for a support team"). Ask the user
   if they have specific instructions; if they decline, proceed without them.
6. **Model tier**: "quality" (default), "balanced", or "economy" (optional)
7. **Max texts per sample** (optional — hard cap on how many texts agents pull
   per sample; useful for large corpora or cost control)

Then initialize:
```bash
uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/init.py \
  --corpus <path> --text-col <col> --k-range <min> <max> \
  --model-tier <tier> --instructions "<instructions>" \
  --workspace <dir> \
  --max-texts-per-sample <n>  # optional hard cap
```

Set the env var for the *current* Bash call (note: env vars don't propagate
across separate Bash tool calls and never reach hooks — later contexts resolve
the workspace via the `.claude/clustering/.active_workspace` pointer file that
`init.py` writes at a fixed location):
```bash
export CLUSTERING_WORKSPACE=<dir>
```

Read the output — it will show corpus stats (size, text length distribution).
Use these to calibrate your approach.

## Staying Grounded

**Always re-read `$CLUSTERING_WORKSPACE/summary.md` before making decisions.** This
is your ground truth. Don't rely on your memory of previous state — the summary
is always fresh (updated by hooks after each agent finishes). This is especially
important after context compaction, which may lose earlier details.

Check the **rejected hypotheses** section of the summary before dispatching
investigations. Don't re-investigate questions that have already been answered.

## User Instructions

Check `config.instructions` in state.json. When present (non-empty), these are
the **primary constraint** on how clusters should be formed. Every agent
dispatch MUST include them in the task description. For example:

> "The user's clustering instructions are: '{instructions}'. Keep these in mind
> as you propose/audit/investigate clusters."

If the instructions say "cluster by issue type", agents should not cluster by
sentiment. If they say "focus on actionable categories", agents should avoid
abstract or overly granular clusters. The instructions shape every decision.

When instructions are empty, agents should discover clusters based on the
natural structure of the data without a specific lens.

## Model Tier

Check `config.model_tier` in state.json. When dispatching agents via Task:

- **quality** (default / unset): Don't set model — all agents inherit yours.
- **balanced**: Set `model: haiku` for proposer and auditor. Don't set model
  for synthesizer, investigator, or critic (they inherit).
- **economy**: Set `model: haiku` for all agents.

## Choosing Parameters Intelligently

You have corpus stats from init. Use them:

**Sample sizes** — think about what fits in an agent's context window (~50K
tokens of working space). Rough guide:
- Texts < 100 chars avg → agents can handle 200-400 texts comfortably
- Texts 100-500 chars → 50-150 texts
- Texts 500+ chars → 20-50 texts
Scale to corpus size — don't sample 200 from a corpus of 300.

If `config.max_texts_per_sample` is set in state.json, `sample.py` enforces
the cap automatically. Respect this in your task descriptions too — don't
ask agents to process more texts than the cap allows.

**Number of proposals** — start with 2-3 from different angles. Dispatch them
in parallel using concurrent Task calls (send multiple Task invocations in one
message). If they converge, that's signal. If they diverge, get 1-2 more.
Wider k_range warrants more proposals.

**Audit sample size** — same chars-per-text guidance as proposals
(200-400 for short texts, 50-150 for medium, 20-50 for long). Floor of ~50
texts for the coverage % to be meaningful — a 20-text audit gives a ±10
percentage-point confidence interval on coverage, which is noisier than the
signal you're trying to read. Seen texts are excluded by default, so audits
always get fresh texts.

## Iteration Loop

At each step:

1. Read `$CLUSTERING_WORKSPACE/summary.md`

2. Reason about what would be most valuable right now:
   - No proposals yet → **propose** (start with 2-3 proposals in parallel)
   - Have 2+ proposals, no synthesized cluster set → run **cross-proposal metrics**, then dispatch **synthesizer**
   - Have clusters but no audit → **audit**
   - Audit shows weak clusters → **investigate** the weak ones
   - Audit shows unclustered patterns → **investigate** unclustered region
   - Haven't critiqued after major changes → **critique**
   - Critic flagged structural issues (overlap, gaps, granularity, boundary
     confusion) with concrete evidence → **investigate** the flagged clusters
   - About to suggest finalize → glance back at the most recent audit's
     `weak_clusters` and the most recent critique's open issues; if anything
     concrete remains unaddressed, an Investigator pass is usually cheaper
     than shipping the issue
   - Everything looks solid → suggest **finalizing**

3. **Before dispatching the synthesizer** (when 2+ proposals exist), run
   cross-proposal metrics to quantify agreement:
   ```bash
   uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/confusion.py cross-proposal
   ```
   Read the output. Include key findings (ARI scores, high-entropy clusters,
   most inconsistent texts) in the synthesizer's task description. This gives
   the synthesizer concrete evidence about where proposals agree and disagree.

   Then store the metrics in state so summary.md reflects them:
   ```bash
   uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/state.py \
     update-cross-proposal-metrics $CLUSTERING_WORKSPACE/metrics/<latest_file>.json
   ```

4. Dispatch the appropriate agent with a clear, specific task description.
   Tell it what to focus on, roughly how many texts to work with (but let
   it adjust), and what question to answer. Tell it the workspace path.
   Set the model parameter according to the model tier.

5. Read the agent's return summary. Re-read `$CLUSTERING_WORKSPACE/summary.md`.

6. Decide: continue, report to user, or suggest finalizing.

## Reporting and Checkpoints

After every 2-3 agent dispatches, update the user:
- What you did and found
- Current cluster count, coverage, confidence
- Cross-proposal agreement scores (ARI, element similarity) when available
- What you plan to do next
- Ask if they want to steer in a different direction

**Hard checkpoint**: After 20 agent dispatches without explicit user input,
STOP and report. Don't keep going — the user should confirm direction.
The baseline P→S→A→C cycle is 6 dispatches (3 proposers + synth + auditor +
critic); the remaining budget is for Investigator passes (each typically
costs ~2 slots, since a structural change resets coverage and warrants a
re-audit) and any follow-up proposer / synth refinements.

## Before Stopping

Before ending a session (whether pausing for user input or completing), write
`$CLUSTERING_WORKSPACE/plan.md` with:
- Current assessment of cluster quality
- What you just did
- What you would do next and why
- Any open questions

This enables seamless resume in a new session.

## Run Logging

Maintain a chronological trace at `$CLUSTERING_WORKSPACE/run_log.md`. This file
is your session diary — it lets humans (and future sessions) understand exactly
what happened.

**Write the dispatch entry BEFORE calling the agent.** Append the result entry
AFTER it returns. This way the log is useful even if the run is interrupted.

Each entry should record:
- **Timestamp** and **action type** (e.g., `dispatch-proposer`, `dispatch-auditor`,
  `result-proposer`, `decision`, `checkpoint`, `user-input`)
- **Agent type** and **model** used (if set)
- **Task summary**: What was asked (1-2 sentences)
- **Result summary**: What came back (1-2 sentences, for result entries)
- **Orchestrator reasoning**: What to do next and why (1-2 sentences)

Format example:
```markdown
### 2026-05-28T14:32:00Z — dispatch-proposer
- **Agent**: proposer (model: haiku)
- **Task**: Sample 100 random texts, propose clusters with balanced style
- **Reasoning**: First run, need initial proposals for synthesis

### 2026-05-28T14:33:15Z — result-proposer
- **Result**: Proposed 8 clusters from 100 texts, 4 unclustered
- **Reasoning**: Need a second proposal before synthesizing — will dispatch with different angle
```

## File Hygiene

Agents must write output files to proper subdirectories, never the workspace
root. The expected layout during a run:
- `proposals/` — proposer outputs (`prop_*.json`)
- `audits/` — auditor outputs (`audit_*.json`)
- `investigations/` — investigator outputs (`inv_*.json`) and synthesizer
  outputs (paired `synthesis_*.json` reasoning + `synthesis_*_clusters.json`
  set-clusters input)
- `critiques/` — critic outputs (`critique_*.json`). Critiques live in their
  own directory because `state.py apply-recommendation` only operates on
  `investigations/` — critiques carry findings, not actions.
- `metrics/` — cross-proposal metrics

Do not leave loose files (like `synthesized_clusters.json` or
`REAUDIT_SUMMARY.md`) in the workspace root. If an agent writes to the root,
move it to the appropriate subdirectory.

When refining descriptions (after critic feedback, after investigation), use
`update-descriptions` instead of `set-clusters` to avoid wiping evidence data.

## Description Quality

Cluster descriptions must be generalizable. When refining descriptions (after
critic feedback, after investigation), enforce these rules:
- **No text IDs** — never write "Text 139 belongs in X, not Y"
- **No corpus-specific statistics** — no dollar amounts, dates, or numbers from
  the data itself
- **Concepts, not observations** — describe *what kind* of texts belong, not
  what you observed in specific texts
- **Principles, not corrections** — boundary rules should be general ("monetary
  vs fiscal policy") not instance-specific ("Text 42 is macro, not rates")
- **No routing rules / triage checklists** — descriptions define categories,
  they aren't dispatch instructions

When you need to update descriptions without changing cluster structure, use the
`update-descriptions` command instead of `set-clusters`:
```bash
uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/state.py \
  update-descriptions <file>
```
Input format: `{"clusters": [{"id": "c1", "name": "...", "description": "..."}, ...]}`
This preserves all evidence, audit data, and text IDs.

## When NOT to Continue

- User says stop → stop
- All clusters high confidence, coverage > 85%, critic satisfied → suggest finalize
- Last 2-3 actions improved nothing → suggest finalize (diminishing returns)

## When something goes wrong

If an agent dispatch fails repeatedly, a script exits non-zero in a way you
can't explain, or the user expresses real dissatisfaction with how the run
is going, ask once: *"Want me to file a GitHub issue with the context
attached?"* On yes, invoke `/cluster-report-issue` (or call
`$CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/report_issue.py` directly).

Don't ask for trivial recoverable errors — a single retry, an expected
`validate.py` rejection that the next dispatch will fix, or anything you
can repair yourself. Reserve the offer for situations where you would
otherwise have to tell the user "I'm stuck."
