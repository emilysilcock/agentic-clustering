---
name: cluster-run
description: >
  Run the agentic cluster discovery workflow. Iteratively discovers natural
  clusters in a text corpus using proposer, synthesizer, auditor, investigator,
  and critic subagents.
allowed-tools: Task, Read, Bash, Write, AskUserQuestion
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

For new runs (no existing workspace, or user chose "start fresh"), collect the
seven setup answers below, in this order, before running init:

1. **Workspace directory** (Q1)
2. **Corpus path** (Q2)
3. **Text column** (Q3)
4. **Dataset description** (Q4)
5. **Clustering lens** (Q5)
6. **Cluster count range** (Q6)
7. **Model tier** (Q7)

Every answer is **entirely the user's call** — you are not offering a menu
they must pick from, and the questions say so where it matters.

Rules that apply to every question:

- **One at a time.** One AskUserQuestion call per question, in the order
  above, waiting for the answer before asking the next.
- **Render through the AskUserQuestion tool. Do not reproduce the options as
  a markdown table or a bulleted list in your reply** — the tool call is what
  makes the run stop and wait for the user. Writing the options out as prose
  is the failure mode to avoid, not a lighter-weight alternative to the tool.
  (Headless/scripted invocations that already supply the answers skip the
  asking entirely — AskUserQuestion is unavailable in `-p` mode.)
- **Skip what's already answered.** If the user's invocation or an earlier
  answer already supplies an item (users may volunteer several at once), skip
  that question. If everything is supplied, ask nothing — confirm the
  configuration in one line and proceed to init.
- **Fixed text is verbatim.** Question text, headers, and the fixed options
  below are pinned — do not paraphrase, shorten, or expand them. Where options
  are marked *(drafted)*, draft them fresh from the actual project/corpus as
  described. The single allowed substitution is `<session model>` in Q7:
  replace it with the model named in your environment context (e.g.
  "Opus 4.7").
- **Never mark any option "(Recommended)".** Where there is a default it is
  named in the option label; nothing else is ranked.
- **Do not call `init.py` until all seven are answered.** Opt-out picks
  ("No description", "No instructions") count as answers; silence does not.
- **Follow-ups** get answered from the facts in this section — never invent a
  default that doesn't exist. Q2, Q3, and Q6 are required by init.py and have
  no defaults.

#### Q1 — Workspace

- **question**: "Where should the clustering workspace live? This is where the
  run's working files and outputs (proposals, audits, state, the final
  summary) are written."
- **header**: "Workspace"
- **options**:
  - `./clustering/ (default)` — "A clustering/ directory at the project root."
  - `.claude/clustering/` — "Hidden away inside the project's .claude
    directory."

#### Q2 — Corpus path

Before asking, scan the project for plausible corpus files (`*.csv`,
`*.json`, `*.jsonl`; ignore workspace/config directories like `.claude/` and
`clustering/`). Offer up to 4 as options.

- **question**: "Where is your corpus? Give the path to the CSV, JSON, or
  JSONL file containing your texts."
- **header**: "Corpus"
- **options**: *(drafted)* the candidate files found, one per option, the
  path as the label. If the scan finds no candidates, ask the same question
  as plain text instead (the only sanctioned prose fallback in this flow).

The plugin bundles a demo corpus at
`$CLAUDE_PLUGIN_ROOT/examples/mip_responses.csv` (18 short survey responses;
the README's worked example uses it). When the user's answer refers to it —
`examples/mip_responses.csv`, "the example corpus", "the bundled demo" —
pass the full `$CLAUDE_PLUGIN_ROOT/examples/mip_responses.csv` path to
`--corpus` and to the Q3/Q4 peeks: the relative path does not exist in the
user's project. If the scan finds no candidates, mention in the plain-text
fallback that this demo corpus is available.

#### Q3 — Text column

Read the corpus header (or first record) first. If it has exactly one
column/field, skip this question and use it (note that in your reply).

- **question**: "Which column or field in that file holds the text to
  cluster?"
- **header**: "Text column"
- **options**: *(drafted)* the actual column/field names, most text-like
  first (up to 4; Other covers the rest).

#### Q4 — Dataset description

Read ~15 rows from the corpus file directly (Bash/Read on the path the user
gave — `init.py` hasn't run yet, so `sample.py` isn't available). Reuse this
peek for Q5. From it, draft 2–3 candidate one-sentence descriptions of what
the texts are. If the corpus is unreadable or too opaque to draft from, fall
back to generic candidates rather than blocking on it.

- **question**: "What are these texts? One sentence of context is carried
  into every agent dispatch. The options are drafted from a peek at your
  corpus — pick Other to write your own."
- **header**: "Dataset"
- **options** (first is always the opt-out):
  - `No description` — "Skip — agents will infer context from the texts
    themselves."
  - *(drafted)* 2–3 candidate descriptions, e.g. "Customer support tickets
    for an online retailer." — full sentence as the description, a short
    handle as the label. Never reuse this example verbatim; it illustrates
    the format, not the content.

#### Q5 — Clustering lens

From the same ~15-row peek, draft 3 lenses that are genuinely plausible *for
this corpus*. Never reuse the examples below verbatim; they illustrate the
format, not the content.

- **question**: "What should the texts be grouped by? Entirely your call —
  the same corpus often supports several groupings (academic abstracts could
  be clustered by topic, by methodology, or by dataset used), and your
  instructions pick the lens. The options are illustrations drawn from your
  corpus; pick Other to name any lens you like."
- **header**: "Lens"
- **options** (first is always the opt-out):
  - `No instructions` — "Agents discover whatever structure is most salient
    in the data."
  - *(drafted)* 3 corpus-drawn lenses — a short handle as the label, a
    one-sentence instruction as the description, e.g. `Issue type` — "Group
    texts by the type of issue the customer is raising."

The label is a handle, not the instruction. When the user picks a lens
option, its **description** is what flows into `--instructions` — never the
label. "Issue type" is not a usable instruction; the sentence is.

**Assembling `--instructions`**: concatenate whichever parts exist — the Q4
sentence + the Q5 lens sentence (e.g. "Customer support tickets for an online
retailer. Group texts by the type of issue the customer is raising."). If one
was skipped, use the other alone; if both were skipped, pass the empty
string.

#### Q6 — Cluster count range

Ask with exactly these options. Do not invent your own buckets, do not narrow
them, and never derive or propose a range yourself — not from corpus size,
not from the peeked rows, not even if the user asks you to choose. Corpus
stats set sample sizes, not taxonomy granularity: a 200-text corpus can
warrant 40 fine-grained categories; a 50,000-text one can warrant 5. If the
user asks you to pick, explain that the range is theirs to set and re-ask.
The ordering is ascending, not a preference ranking.

- **question**: "What range should the cluster count land in? This is the
  granularity control — within your range, the agents settle on the count the
  data actually supports. Min and max are entirely your call and the options
  below are just ballparks: pick Other and type any exact range you like
  (e.g. '4–6', '25–40', '80–150'). The range is a target, not an exact
  promise — when unsure, go wide."
- **header**: "Range"
- **options**:
  - `Broad: 2–8` — "A few headline groups"
  - `Mid: 10–30` — "A codebook-sized scheme"
  - `Fine: 60–150` — "A detailed classification"

If the user picks a bucket, record its numbers verbatim as min/max. If they
use Other, accept whatever they give, however wide or narrow — `5–80` is a
legitimate answer and means "search broadly, I'll narrow on a rerun".

#### Q7 — Model tier

- **question**: "Which model tier should the agents run on? quality (the
  default) runs every agent on the same model as this session
  (<session model>); balanced moves the high-volume reading agents
  (proposers, auditor) to Haiku; economy runs all agents on Haiku."
- **header**: "Model tier"
- **options**:
  - `quality (default)` — "Every agent on <session model>."
  - `balanced` — "Proposers and auditor on Haiku; synthesizer, investigator,
    critic on <session model>."
  - `economy` — "All agents on Haiku."

Then initialize:
```bash
uv run $CLAUDE_PLUGIN_ROOT/skills/corpus-tools/scripts/init.py \
  --corpus <path> --text-col <col> --k-range <min> <max> \
  --model-tier <tier> --instructions "<instructions>" \
  --workspace <dir>
```
(`--max-texts-per-sample <n>` remains available as an advanced cost-control
flag for very large corpora; don't ask about it — apply it only if the user
raises it.)

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

**Number of proposals** — start with 6-7 from different angles. Dispatch them
in parallel using concurrent Task calls (send multiple Task invocations in one
message). If they converge, that's signal. If they diverge, get 1-2 more.
Wider k_range warrants more proposals.

**Audit sample size** — an audit answers two different questions and they need
two different draws. Size each separately; don't try to make one sample serve
both.

*Coverage draw* (`--strategy random`, the default) — how much of the corpus
the taxonomy covers. Same chars-per-text guidance as proposals (200-400 for
short texts, 50-150 for medium, 20-50 for long), with a floor of ~50 texts: a
20-text audit gives a ±10 percentage-point interval on coverage, noisier than
the signal you're reading. This draw must stay uniform, so never size it by
cluster count.

*Per-cluster draw* (`--strategy stratified`) — whether
each individual cluster holds up. The chars-per-text bracket above is about
what fits an agent's context; it says nothing about how many clusters the
sample has to cover, and a uniform draw allocates texts in proportion to
corpus prevalence — exactly backwards, since the rare clusters are the ones
whose fit is least certain. On a 65-cluster taxonomy a 300-text uniform audit
averages 4.6 assignments per cluster and leaves the tail on 0-2, which is not
enough to call a cluster high-confidence or to call it weak.

The floor is **n≥5 per cluster**. Below that, `update-from-audit` withholds
the verdict and **`finalize` refuses to export at all** — it checks before
writing or archiving anything, so a refusal costs nothing, but you cannot ship
a taxonomy whose labels the sample can't support without an explicit user
decision. Treat clearing the floor as part of the work, not a formality. The
stratified draw serves only the clusters that are short, so the cost is
proportional to the thin tail rather than to k, and it shrinks every pass —
expect the draw to get smaller each time until `sample.py` says there is
nothing to do. That, plus the **Per-Cluster Audit Sample** section
disappearing from `summary.md`, is the signal you're done auditing.

The draw ranks candidates by TF-IDF similarity to each cluster's description,
so it depends on the descriptions being specific. `--seed-from assigned`
substitutes the cluster's already-assigned texts for its description; it did
not beat the default in testing, so reach for it only if descriptions are
vague or boilerplate and clusters are coming back `unsupported` you believe in.

Dispatch the two draws as separate auditors (or one auditor told to write two
audit files). Each audit file carries `"sample_basis": "random"` or
`"stratified"`, and only random audits move the headline coverage number. A
stratified audit must also carry `strata_file` — the manifest path `sample.py`
prints — or the run loses the ability to tell an untested cluster from a
rejected one. Seen texts are excluded by default, so every draw gets fresh
texts.

**The three thin labels mean different things and want different responses:**

| label | means | do |
|---|---|---|
| `unaudited` / `insufficient-sample` | too few assignments, and nothing was aimed here | audit again, stratified |
| `unsupported` | texts *were* aimed here and the auditor put them elsewhere | investigate where they went — merge, sharpen the boundary, or remove |
| `low` | a real verdict on a sample that met the floor | investigate |

Never send an investigator after `insufficient-sample` — it's chasing a
sampling artefact. Never send another identical audit after `unsupported` —
it will come back the same.

## Iteration Loop

At each step:

1. Read `$CLUSTERING_WORKSPACE/summary.md`

2. Reason about what would be most valuable right now:
   - No proposals yet → **propose** (start with 6-7 proposals in parallel)
   - Have 2+ proposals, no synthesized cluster set → run **cross-proposal metrics**, then dispatch **synthesizer**
   - Have clusters but no audit → **audit** (coverage draw first)
   - Clusters listed as **under-sampled** in `summary.md`'s **Per-Cluster
     Audit Sample** section → **audit** with `--strategy stratified`. These
     are unmeasured, not weak; an investigator sent after one is chasing a
     sampling artefact
   - Clusters listed as **unsupported** there → **investigate**: candidates
     were aimed at them and the auditor assigned them elsewhere, so another
     identical audit returns the same answer
   - Audit shows weak clusters (a `low` label on a cluster that met the
     floor) → **investigate** the weak ones
   - Audit shows unclustered patterns → **investigate** unclustered region
   - Haven't critiqued after major changes → **critique**
   - Critic flagged structural issues (overlap, gaps, granularity, boundary
     confusion) with concrete evidence → **investigate** the flagged clusters
   - About to suggest finalize → glance back at the most recent audit's
     `weak_clusters`, the most recent critique's open issues, and
     `summary.md`'s **Per-Cluster Audit Sample** section; if anything concrete
     remains unaddressed, an Investigator pass is usually cheaper than
     shipping the issue, and a thin per-cluster sample is cheaper to fix now
     than to explain in the finalized taxonomy
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

These are reports, not stops — deliver the update and carry on working. Only
stop when one of the conditions in **When NOT to Continue** fires. There is
deliberately no dispatch cap: a dispatch count tells you nothing about whether
the taxonomy is converging or whether there is a question only the user can
answer, so a long run that is still improving should continue and a short one
that has stalled should not. For orientation, the baseline P→S→A→C cycle is
9-10 dispatches (6-7 proposers + synth + auditor + critic), and each
Investigator pass costs ~2 more, since a structural change resets coverage and
warrants a re-audit.

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
- `strata/` — stratified-draw manifests (`strata_*.json`), written by
  `sample.py`, referenced by an audit's `strata_file`
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
- All clusters high confidence **on a sample that met the n≥5 floor**,
  coverage > 85%, critic satisfied → suggest finalize. "High confidence" off
  two texts is not a reason to stop; check `summary.md`'s **Per-Cluster Audit
  Sample** section before reading the labels as settled
- Last 2-3 actions improved nothing → suggest finalize (diminishing returns),
  *unless* what's still missing is per-cluster sample. Clusters below the
  floor aren't diminishing returns, they're unfinished work, and `finalize`
  will refuse until they're cleared. Keep going: another stratified pass for
  the under-sampled ones, an investigator for the unsupported ones. Bring it
  to the user only if a cluster stays unsupported after an investigator has
  looked at it — that's a genuine decision (drop it, or ship it unvalidated)
  and it's theirs

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
