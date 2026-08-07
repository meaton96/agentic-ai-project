# Agent Reference — agentic-ml-classification

Quick-reference companion to [`PROJECT_OVERVIEW.md`](../PROJECT_OVERVIEW.md)
(full rationale) and [`CLAUDE.md`](../CLAUDE.md) (invariants). Built for
answering "how does this actually work" in conversation — table first, then
one section per agent, then a Q&A cheat sheet for likely follow-ups.

## The one-line pitch

**Agents propose, the harness decides.** Every LLM call in this system
produces a JSON proposal; a separate, deterministic, non-LLM piece of code
independently re-validates it before anything happens. An agent can never
compute its own metrics, choose its own train/test split, or see the locked
test set. If an agent hallucinates or is wrong, the harness rejects the
proposal — it doesn't corrupt the result.

## Quick reference

| # | Agent | Goal | Decision surface | Can never do |
|---|---|---|---|---|
| 1 | **Intake** | Turn a raw CSV + optional goal sentence into a formal problem spec | Propose target column, task type, id/group/time columns | Approve its own guess — harness re-validates class count (2–20) independently |
| 2 | **Feature Engineering** | Decide if the raw columns need structural changes before modeling | Propose columns to drop + derived features from a 5-op vetted catalog (ratio, interaction, log1p, datetime_parts, missing_indicator) | Write transformation code; use fitted statistics (means, scalers) — every op is stateless/row-wise |
| 3 | **Profiler** | Characterize the dataset in plain language + recommend a split strategy | Narrate facts; explain risks | Override the rule-based `recommended_split_strategy` — that's 100% deterministic |
| 4 | **Modeling** | Choose a modeling approach and produce an evaluable pipeline | Pick 1 of 6 vetted templates + fill a config dict (columns, hyperparameters) | Write model code; see the test set; compute its own score |
| 5 | **Verification** | Second opinion on a candidate that already passed every deterministic gate | Verdict: `approved` / `flagged` / `rejected` | Approve or unblock a candidate that failed a gate — it is structurally never shown one |
| 6 | **Deep-Dive** | Explain *why* one already-flagged example was flagged (on-demand, post-hoc) | Synthesize a hedged hypothesis from 3 deterministic measurements (phase segmentation, occlusion attribution, cross-signal localization) | Compute evidence itself; recommend a specific fix/part — it's a lead, not a diagnosis |
| 7 | **Planner** *(dynamic orchestrator only)* | Decide which agent runs next given the goal + run state | Propose `{action, agent_id, args}` — one step at a time | Invent an agent id or skip a precondition — `validate_plan()` re-checks against the real registry/state every time |

Agents 1–5 run as one fixed sequential loop (`run_orchestrator.py`). Agent 6
runs separately, on demand, against a saved model artifact. Agent 7 exists
only in the second entry point (`run_dynamic_orchestrator.py`), which
replaces the fixed sequence with a planning loop over the same underlying
agents — `run_orchestrator.py` is untouched and remains the baseline.

## Per-agent detail

### 1. Intake Agent
- **Problem it solves:** "here's a CSV, here's maybe a sentence about what I
  want" → a formal spec (which column is the label, which are IDs/timestamps
  to exclude).
- **Sees:** column names, dtypes, missingness, cardinality, name-based hints
  — computed *before* any target is known (can't see class balance/
  correlation with a label that doesn't exist yet — that'd be circular).
- **Proposes:** `{target_column, task, id_columns, group_column,
  time_column, positive_label, reasoning}`.
- **Harness re-checks independently:** target must have 2–20 non-null
  unique values. The agent's self-reported `task` field is not trusted.

### 2. Feature Engineering Agent
- **Problem it solves:** should the feature set itself change before
  modeling — drop a column, add a derived one (ratio, interaction, log1p,
  extracted date parts, missing-value flag)?
- **Sees:** the same dataset profile the Profiler sees, plus a catalog of 5
  vetted, stateless operations.
- **Proposes:** `{drop_columns, derived_features, explanation}` — never
  code. Every op computes a value from that row's own columns only, which is
  exactly what makes it safe to run on the *whole* dataset before the
  train/val/test split exists.
- **What stays out of scope:** imputation values, scaling, target encoding
  — those need a fitted statistic, so they stay inside the modeling
  templates, fit only on the training fold.

### 3. Profiler Agent
- **Problem it solves:** produce a human-readable characterization of the
  dataset and a recommendation for how to split it.
- **Sees:** one tool call's worth of facts — types, missingness,
  cardinality, class imbalance, a rule-based `recommended_split_strategy`.
- **Proposes:** a summary + key risks. Instructed not to contradict the
  tool's own recommendation.
- **Why this design:** the split strategy is arguably the single decision
  most likely to silently invalidate every downstream result, so it's
  computed by tested, rule-based code with zero LLM involvement. The
  agent's only job is narration.

### 4. Modeling Agent
- **Problem it solves:** choose a modeling approach and produce a working,
  evaluable pipeline.
- **Sees:** the profiler's facts + a catalog of 6 pre-built, pre-validated
  "recipe templates" with descriptions of when each applies.
- **Proposes:** `{candidate_id, template_id, config, explanation}` — picks a
  template by name, fills in a config dict. **Never writes model code.**
- **What the harness does that the agent controls none of:** re-validates
  every column name, static-checks and sandbox-builds the template, fits on
  train, scores on validation with bootstrapped CIs, and requires **two
  independent leakage checks to both pass** before the candidate is even
  eligible for review.

### 5. Verification Agent
- **Problem it solves:** a second, independent opinion — like a teammate
  reviewing an already-passing pull request before merge.
- **Sees:** a bundle of facts already computed elsewhere: template
  description, candidate config/explanation, validation metrics + CIs, both
  leakage results, imbalance/risk facts. Computes nothing itself.
- **Proposes:** `{verdict, concerns, reasoning}` — `approved`, `flagged`
  (proceeds, but recorded for human review), or `rejected` (blocked).
- **The asymmetric trust boundary:** it is only ever shown candidates that
  already passed both deterministic leakage gates, so it structurally
  cannot approve or unblock a failure — its schema has no override option.
  A malformed response degrades to `flagged`, never `approved`.
- **Orchestrator behavior:** candidates are reviewed best-first by
  validation score; a `rejected` verdict falls back to the next-best
  candidate instead of failing the run.

### 6. Deep-Dive Agent
- **Problem it solves:** a different question than 1–5. They answer "is
  this positive (and can we trust that)?" This one answers "why?" — given
  one example a completed run already flagged, what in the data explains
  it. Runs on demand, per example, after a run has finished.
- **Sees:** one bundled tool that runs three deterministic measurements and
  returns the combined output — the agent cannot compute any of it: phase
  segmentation, occlusion attribution (replace one feature at a time,
  measure the probability drop — model-agnostic, not aviation-specific),
  and cross-signal localization (does one channel deviate from its siblings
  specifically during a relevant phase?).
- **Proposes:** `{hypothesis, agrees_with_localization, confidence}` — a
  hedged 2–4 sentence explanation. Instructed never to invent a cause or
  recommend a specific fix — a lead for inspection, not a diagnosis.
- **Scope limit (stated, not silent):** binary targets only for now —
  occlusion attribution's "positive class probability" framing doesn't
  generalize to multiclass yet.

### 7. Planner Agent (dynamic orchestrator)
- **Problem it solves:** the fixed 5-agent sequence can't express things
  like "skip straight to explaining an already-flagged example" or "try one
  more modeling candidate before verification." The planner replaces the
  fixed order with a real decision at each step.
- **Sees:** the goal + a compact, JSON-safe summary of what's happened so
  far (`RunStateSummary`) — never raw dataframes or fitted pipelines.
- **Proposes:** exactly one next action per turn: run a specific catalog
  agent, or declare the run finished.
- **Independently re-checked by `validate_plan()`:** the proposed agent
  against the *real* registry (not a name the planner invented), and its
  preconditions against the *real* run state (not the planner's claim). A
  rejected proposal never executes — it's fed back as an error and retried,
  bounded.
- **Concrete proof it's doing something a fixed script can't:** given an
  "explain this flight" goal + an existing model artifact, the planner
  routes straight to Deep-Dive — intake, feature engineering, profiler, and
  modeling never run at all.

## The other half: what the deterministic harness owns

Worth having ready, since "what do the agents *not* do" is usually the more
interesting question:

- **Splitting** — 5 strategies (random/stratified/group/time/group+time),
  explicit strategy required, no silent default to random on data that
  isn't i.i.d.
- **Leakage checks**, run on every candidate, independently: duplicate
  rows across splits, group overlap, time ordering, raw feature/target
  correlation, and a **label-permutation test** (fit the actual candidate
  pipeline on shuffled labels, confirm it scores at chance).
- **Sandboxed execution** — candidate model code is AST-checked (reject
  `eval`, `open`, `subprocess`, network imports) then run in an isolated
  subprocess; only an *unfitted* model object crosses back, never data.
- **Metrics with bootstrapped confidence intervals.**
- **An append-only leaderboard** — every candidate ever evaluated is
  logged, not just the winner.
- **The test set is touched exactly once per run**, enforced by a one-shot
  guard in the finalize step.

## Likely questions and short answers

**"Why not just let the LLM write the model code?"**
Free-form generated code is hard to statically verify for safety or
correctness, and a subtly wrong preprocessing step is exactly the kind of
bug an LLM introduces easily and a static checker can't catch. Instead
there's a library of 6 verified `build_pipeline(config)` templates; the
agent's decision surface is *which template + which config*, not arbitrary
code — auditable and sandboxable.

**"What stops an agent from just hallucinating a good-looking result?"**
Every proposal is JSON, and every field is independently re-validated
against facts the harness itself computed — hallucinated column names,
made-up metrics, or an invented template id are all rejected before they
can affect anything. The agent never computes its own score.

**"What happens if an agent's output is garbled or unparseable?"**
It degrades conservatively, never silently succeeds: unparseable
verification → `flagged` (not `approved`); unparseable deep-dive →
falls back to a deterministic template built from the same evidence;
unparseable planner output → bounded retry, then a clean abort. Never a
crash over a formatting glitch, never a silent free pass.

**"Can the verification agent override a failed leakage check?"**
No — structurally, not just by instruction. It is only ever shown
candidates that already passed both leakage gates, so there's no code path
where it sees (let alone overrides) a failure. Its JSON schema has no
"override" field.

**"How many leakage checks does a candidate have to pass, and why two?"**
Two, and they catch different bugs: the label-permutation test catches
leakage in the modeling *process* (e.g. a scaler fit before the split); the
feature-correlation check catches leakage in raw *feature content* (a
column that's just a copy of the target). There's a test proving a
candidate can pass one and fail the other — neither is redundant.

**"What's the actual scope right now?"** Single-machine, CPU, tabular data,
binary or multiclass classification (up to 20 classes). No regression, no
GPU, no distributed training — deliberately narrow so the trust boundary
stays provable.

**"What's not built yet?"** Cross-run priors/evidence reuse (Phase 6) and
parallel candidate search (Phase 7) — both deliberately on hold until the
pipeline has been proven against more datasets and problem types than the
three so far (Titanic, Iris, an aviation predictive-maintenance dataset).

**"Why is there a 7th 'agent' (Planner) separate from the other 6?"**
The first 5 run in `run_orchestrator.py`'s fixed sequence — a script that
*uses* agents. The Planner exists only in the second, independent entry
point (`run_dynamic_orchestrator.py`) that replaces the fixed order with a
real per-step decision. The original static orchestrator is untouched and
is the baseline the dynamic one is evaluated against.
