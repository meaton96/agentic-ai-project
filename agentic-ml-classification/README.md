# agentic-ml

Agentic ML classification pipeline. Constrained MVP scope: CSV/Parquet
input, binary or multiclass classification (2-20 distinct class labels),
tabular data, sklearn/LightGBM/XGBoost, single-machine CPU,
deterministic harness-owned evaluation. Agents never get direct access
to test labels or final scoring logic.

For the *what and why* behind each agent and design decision (written
for presenting the project, not for modifying it), see
[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md). This file is the detailed
file-by-file build log.

## Why there's no OpenClaw here

The original plan used OpenClaw as the agent runtime. During bootstrap
testing we hit a confirmed, currently-unfixed OpenClaw bug: the CLI's
own post-run transcript compaction throws `Already compacted` even on
a fresh, single-turn session with headroom to spare (`shouldCompact:
false` in its own session state, yet it compacted anyway and then
choked finalizing). This reproduced identically on OpenClaw 2026.5.28
and 2026.6.11 (latest as of this writing), independent of model,
prompt, or config.

None of this pipeline's phases actually need OpenClaw's real value-add
(multi-channel messaging, cron/heartbeat automation, skills
marketplace). What's needed is: call a model, get a structured
response, optionally call tools, done. That's `src/agentic_ml/
model_client.py` + `agent_runtime.py` — no sessions, no compaction, no
daemon. If OpenClaw's bug gets fixed upstream and you later want its
multi-channel features (e.g. a Slack front-end onto this pipeline), it
can be added back as an optional caller of the same LiteLLM gateway
without touching anything below.

## Three model endpoints: RIT, gateway, or local

`resolve_model_endpoint()` (`src/agentic_ml/cli_common.py`) supports
three interchangeable OpenAI-compatible endpoints, selected per-script
with `--use-gateway` / `--use-local` (RIT direct is the default if
neither is given; `--use-local` takes precedence if both are given):

- **RIT direct** (default) — `RIT_BASE_URL`/`RIT_API_KEY` in `.env`.
- **LiteLLM gateway** (`--use-gateway`) — routes to RIT or local models
  via `docker compose up -d model-gateway`; useful once you have more
  than one caller.
- **Local** (`--use-local`) — any OpenAI-compatible server on this
  machine (vLLM/Ollama/SGLang), configured via `LOCAL_MODEL_BASE_URL` /
  `LOCAL_MODEL_API_KEY` / `LOCAL_DEFAULT_MODEL` in `.env`. Added because
  RIT's shared endpoint returns real, intermittent 504s under agentic
  tool-calling load (every agent here makes several tool-calling turns
  per run) — a local server sidesteps that for development, at the cost
  of not being the model an eventual deployment would run against.
  Verify it's reachable and that tool-calling actually works (not just
  plain completions — every agent here depends on real `tool_calls`
  coming back) with `python scripts/check_local_connection.py`.

## What's built so far

**Phase 0 (connectivity) — replaces the OpenClaw bootstrap milestone:**
- `src/agentic_ml/model_client.py` — thin `openai` SDK wrapper, works
  against RIT directly or the LiteLLM gateway
- `src/agentic_ml/agent_runtime.py` — minimal tool-calling loop, no
  session state, tested against a mock model client (13/13 unit tests
  pass without needing network access)
- `scripts/check_rit_connection.py` — direct RIT smoke test
- `scripts/check_gateway_connection.py` — gateway-mediated smoke test
- `scripts/check_local_connection.py` — local OpenAI-compatible server
  smoke test (plain completion + tool-calling), see "Three model
  endpoints" below
- `configs/litellm.yaml` + `docker-compose.yml` — LiteLLM gateway,
  routes to the models your own benchmarking proved reliable
  (`qwen3-coder:30b`, `gemma4:26b`/`latest`, `qwen3:8b`, `gpt-oss:120b`);
  `cogito:70b`/`llama3.1:70b` deliberately commented out with the
  reason why (too slow under RIT's shared load for agentic tool-calling
  turns — confirmed via real 504 timeouts in your own testing)

**Phase 1 (deterministic harness, zero LLM involvement) — fully built
and tested:**
- `harness/dataset.py` — CSV/Parquet loading, schema validation,
  deterministic content hashing
- `harness/splits.py` — `random`, `stratified`, `group`, `time`,
  `group_time` split strategies, fully seeded/deterministic, plus CV
  fold generation. Also `resolve_split_columns()` (added after a real
  crash): the profiler's `recommended_split_strategy` is derived purely
  from its own heuristic column detection, independent of what intake
  actually declared as `group_column`/`time_column` — if a run skips
  intake (explicit `--target`) and doesn't pass `--time-column`, the
  profiler can still recommend `"time"` because it *detected* a
  datetime-like column, and `make_split()` would raise a bare
  `ValueError` deep in the call stack. Whenever the recommendation is
  `"time"`/`"group"`/`"group_time"`, the corresponding detected-column
  list is guaranteed non-empty (that's exactly what produced the
  recommendation), so `resolve_split_columns()` auto-adopts that same
  column transparently (prints a note, doesn't fail silently) rather
  than either crashing or guessing from nothing.
- `harness/leakage.py` — five independent checks: duplicate rows
  across splits, group overlap, time ordering (strategy-aware — does
  NOT wrongly flag `group_time`'s expected cross-group calendar
  overlap), label-permutation test (catches pipeline-level leakage,
  e.g. preprocessing fit on the full dataset), suspicious feature
  correlation (catches raw feature-equals-target leakage — a
  deliberately different failure mode than the permutation test; see
  `priors/general/leakage_rules.md` Rule 4)
- `harness/baseline_ladder.py` — Dummy, LogisticRegression,
  RandomForest, HistGradientBoosting, LightGBM, XGBoost with shared
  preprocessing (median/mode imputation, scaling, one-hot encoding)
- `harness/metrics.py` — roc_auc, pr_auc, f1, precision, recall,
  accuracy, brier, each with bootstrap confidence intervals. Binary and
  multiclass share one code path: every caller passes the full
  `predict_proba()` matrix (never a pre-sliced positive-class column),
  and `compute_metrics()` picks binary formulas or macro-averaged
  one-vs-rest multiclass formulas based on how many classes are
  actually in `y_true` — see "Multiclass support" below.
- `harness/sandbox.py` — AST static checks (forbidden imports/calls)
  + subprocess isolation with wall-clock timeout and POSIX
  CPU/memory rlimits for untrusted candidate code; candidates only
  return an unfitted pipeline object, never touch raw data
- `harness/leaderboard.py` — append-only, file-lock-safe JSONL leaderboard
- `scripts/run_baseline_ladder.py` — Phase 1 CLI entry point, runs the
  whole pipeline (load → split → leakage checks → baseline ladder →
  metrics → leaderboard) with **zero agent/LLM involvement**
- `tests/` — 19 passing tests (`test_harness.py` + `test_leaky_fixtures.py`),
  including a regression fixture
  (`tests/leaky_fixtures/obvious_feature_leak.csv`) that must always
  be caught by the leakage checks. 55 pass across the full suite as of
  Phase 4.

**Phase 2 (Profiler agent) — fully built and tested:**
- `harness/profiler.py` — deterministic column typing, missingness,
  cardinality, likely id/group/datetime detection, target imbalance,
  a rule-based recommended split strategy, and leakage risk flags. No
  LLM call can change these facts.
- `tools/profiler_tool.py` — binds a harness-loaded (not file-path)
  dataframe into a single `get_dataset_profile` tool the agent can call
- `scripts/run_profiler_agent.py` — the agent calls the tool once, then
  narrates/recommends in JSON; the deterministic report is always saved
  regardless of whether the LLM's narrative parses cleanly

**Phase 3 (Recipe templates + Modeling agent) — fully built and tested:**
- `src/agentic_ml/templates/` — six verified, static recipe templates
  (`logistic_numeric`, `sklearn_mixed_pipeline`, `lightgbm_mixed`,
  `xgboost_mixed`, `imbalanced_binary_boosted`,
  `high_cardinality_target_encoding`), each exposing the same
  `build_pipeline(config)` contract `harness/sandbox.py` expects.
  `templates/registry.py` is the single source of truth for
  template_id → source file + required/optional config keys.
- `tools/template_tool.py` — a `list_templates` tool exposing the
  registry's metadata (not the source code) to the agent
- `scripts/run_modeling_agent.py` — the agent calls `get_dataset_profile`
  and `list_templates`, then proposes exactly one `{candidate_id,
  template_id, config, explanation}`. The harness never trusts this
  proposal blindly: it re-validates every proposed column against the
  profiler's facts (rejecting unknown/target/id/group/time columns),
  static-checks + sandbox-builds the template source with the agent's
  config, fits/scores on the harness-owned split, and requires TWO
  independent leakage checks to both pass — `label_permutation_test`
  and `check_suspicious_feature_correlation` re-scoped to exactly the
  candidate's selected columns (added in Phase 4, see below; closes a
  gap where only one of the two checks Rule 4 requires was actually
  wired in) — before the candidate is even eligible for promotion. The
  agent picks a recipe; it never gets to grade its own homework.

**Phase 4 (Verification agent) — fully built and tested; built after
Phase 5 (see the note at the end of the Phase 5 entry for why):**
- `harness/verification.py` — `build_review_bundle()`, pure deterministic
  assembly of everything the verification agent is allowed to see: the
  template's metadata, the candidate's config/explanation, validation
  metrics with CIs, both leakage check results, and relevant profiler
  facts. No LLM involvement in this module.
- `tools/verification_tool.py` — a `get_candidate_review_bundle` tool,
  same pattern as the other three tools.
- `steps/verification_step.py` — the agent reviews ONE already-gated
  candidate and returns `{verdict, concerns, reasoning}` where
  `verdict` is `"approved"`, `"flagged"`, or `"rejected"`. Design
  constraint enforced by construction, not just instructed: this agent
  can only make the outcome *more* conservative than "both deterministic
  gates passed" — it is never shown, and therefore can never approve or
  unblock, a candidate that failed one. An unparseable/malformed
  response degrades to `"flagged"` (proceed, but recorded) rather than
  silently `"approved"` or a hard `"rejected"` over a formatting glitch.
- `scripts/run_modeling_agent.py` and `scripts/run_orchestrator.py` both
  call this after a candidate passes the harness's deterministic gates.
  The orchestrator reviews gate-passing candidates **best-first by
  validation metric**; a `"rejected"` verdict falls back to the
  next-best candidate instead of aborting the run, and every reviewed
  candidate (not just the winner) is logged to the leaderboard with its
  verdict for a full audit trail.
- `tests/test_verification.py` — verdict-handling tests (including the
  unparseable→flagged default) plus a regression test proving the new
  feature-correlation gate catches a raw-feature-copies-target leak
  that `label_permutation_test` alone does *not* reliably catch — the
  empirical proof behind `priors/general/leakage_rules.md` Rule 4's
  claim that the two checks are complementary, not redundant.
  `tests/test_orchestrator.py` gained a test proving the reject-and-
  fallback selection actually changes which candidate gets promoted.

**Phase 5 (Orchestrator) — fully built and tested:**
- `harness/intake.py` — `raw_schema_summary()` computes column facts
  with NO target column assumed (dtype, missingness, cardinality,
  name-hint flags for id/group/datetime — deliberately excludes
  anything target-dependent, since intake's whole job is to propose
  the target); `validate_dataset_spec_proposal()` re-validates the
  agent's proposed target/id/group/time columns against those facts
  and enforces the classification-eligibility MVP constraint (target
  must have between 2 and `MAX_CLASSES` (20) non-null unique values —
  binary or multiclass) regardless of what the agent claims.
- `tools/intake_tool.py` — `get_raw_schema` tool, same pattern as the
  other two tools.
- `steps/{intake_step,profiler_step,modeling_step}.py` — the agent-loop
  + validation + sandbox-build + fit/score + leakage-gate logic for
  each phase, extracted out of the standalone scripts into plain
  functions so `run_orchestrator.py` and the standalone scripts drive
  identical logic instead of duplicating it.
- `cli_common.py` — small shared plumbing (model endpoint resolution,
  run-dir/trace-log setup) that all three scripts need.
- `scripts/run_orchestrator.py` — the entry point that runs the full
  loop: **intake** (skipped if `--target` is given) → **profiler** (its
  `recommended_split_strategy` drives the split unless overridden) →
  up to `--max-candidates` **modeling** proposals, nudged toward trying
  a different template each time, each gated by two deterministic
  leakage checks → **select-and-verify**, best-first by validation
  metric, where the **verification agent** can veto a candidate and
  trigger fallback to the next-best one → refit the accepted candidate
  on train+val → one locked test-set evaluation → a short LLM-narrated
  plain-text summary (no tools, no decisions — it only narrates
  already-computed facts, including a caveat if the accepted candidate
  was `"flagged"`). Works from just `--data` (intake guesses a target
  from the schema alone), from `--data --goal "<natural language>"`, or
  from an explicit `--target` that bypasses intake entirely.
- `tests/test_orchestrator.py` — integration tests that drive the
  *entire* loop end to end against a synthetic dataset with a stubbed
  `ModelClient` (dispatch by system-prompt identity + message count, no
  real network), covering intake/no-intake entry modes and the
  reject-and-fallback selection path. This is the automated answer to
  "does prompt (or just a dataset) + dataset → finished classification
  actually work."

Why Phase 4 was built after Phase 5: the deterministic gates that
already existed (sandbox static checks + `label_permutation_test`)
blocked leaky/broken candidates without needing an LLM to audit them,
so closing the full loop end to end first surfaced more real
integration bugs sooner than a dedicated verification agent would
have — including the fact that only one of the two leakage checks Rule
4 calls for was actually wired into the modeling step, which Phase 4's
work then fixed. Also worth knowing: a real dry run against
the Titanic dataset showed `label_permutation_test`'s default
`n_permutations=5` occasionally rejecting a legitimate (non-leaky)
candidate by chance on smaller datasets — the gate is working as
designed, but that default may be worth revisiting. The Phase 4
feature-correlation gate and verification agent are additional,
independent checks, not a fix for that specific permutation-test
sensitivity.

**Observability (added after Phase 4):** `trace.jsonl` only ever
recorded metadata (model, latency, token counts, whether a tool was
called) — not what was actually said. `cli_common.make_transcript_writer`
now writes the full conversation for every agent invocation —
system prompt, tool calls with real (not escaped-string) arguments,
tool results, final response — to `runs/<run_id>/transcripts/
<agent_name>_NN.json` (numbered per agent, so two modeling candidates
in one run produce `modeling_01.json`/`modeling_02.json`, not a
clobber). Every script and the notebook write these; the notebook's
final section demonstrates opening one and reading it. This is
independent of Phases 6/7 but exists now because more candidates
running (Phase 7) means more to inspect after the fact.

**Feature Engineering agent (extensibility addition, runs between
intake and the profiler in the actual pipeline sequence, even though
it's listed here at the end of the build order):**
- `harness/feature_engineering.py` — a vetted, deterministic operation
  catalog (`ratio`, `interaction`, `log1p`, `datetime_parts`,
  `missing_indicator`) plus `validate_feature_proposal()`. Every op is
  stateless and row-wise (depends only on that row's own values, never
  a fitted statistic) — that's what makes it safe to compute on the
  full dataset before the split even exists, the same way the
  profiler's own descriptive facts are. This agent does NOT decide
  imputation values or scaling — those stay inside the modeling
  templates' `ColumnTransformer`, fit only on the training fold,
  unchanged.
- `tools/feature_tool.py` — a `list_feature_ops` tool, same pattern as
  the other four tools.
- `steps/feature_engineering_step.py` — the agent proposes
  `{drop_columns, derived_features, explanation}` using the same
  `get_dataset_profile` facts the profiler sees, plus `list_feature_ops`.
  The harness validates every column reference against the profiler's
  dtype flags (e.g. `datetime_parts` requires `is_likely_datetime`),
  forbids the target column as any op's input, and forbids dropping
  the declared group/time column (though it IS a valid `datetime_parts`
  input — extracting parts from the time column is the expected use).
  Only then are the ops deterministically applied to produce an
  augmented dataframe.
- `scripts/run_feature_engineering_agent.py` — standalone driver
  (mirrors `run_profiler_agent.py`'s pattern), writes a CSV preview of
  the engineered dataframe for quick inspection.
- `scripts/run_orchestrator.py` and the notebook: runs right after
  intake, before the profiler agent, so the profiler's facts and
  everything downstream reflect the augmented column set. The
  resulting `drop_columns` are folded into `id_columns` (the same
  exclusion mechanism intake already uses) rather than mutating the
  dataframe directly — the augmented dataframe still contains dropped
  columns for reference; they're just excluded from `X`. Skippable via
  `--skip-feature-engineering`.
- `tests/test_feature_engineering.py` — correctness of every op
  (division-by-zero → NaN, negative → NaN for log1p, exact datetime
  part values, missing-indicator flags) and every validation rejection
  case. `tests/test_orchestrator.py` gained a test proving a real
  (non-no-op) proposal's derived column actually reaches the modeling
  agent's candidate config, not just that the step runs.

**Multiclass support (extensibility addition, prompted by a real crash
on the Iris dataset — the first non-Titanic dataset actually tried):**
- Original MVP scope hard-required exactly 2 unique target values
  everywhere. Running the orchestrator against Iris (3 species) crashed
  immediately at intake validation. Generalized to binary-or-multiclass
  (2 to `MAX_CLASSES=20` distinct labels — above that, a column is far
  more likely a continuous/ID field than genuine class labels):
  - `harness/metrics.py` — every metric function now has a
    macro-averaged one-vs-rest multiclass variant alongside its binary
    form, dispatched on `len(np.unique(y_true))`. Every caller
    (`modeling_step.py`, `run_orchestrator.py`'s final test-set eval,
    the notebook) was updated to pass the *full* `predict_proba()`
    matrix through instead of the old `proba[:, 1] if proba.shape[1]
    == 2 else proba.max(axis=1)` fallback, which silently produced a
    meaningless number for 3+ classes instead of erroring.
  - `harness/intake.py` — `validate_dataset_spec_proposal()` now checks
    `2 <= n_unique_target <= MAX_CLASSES` instead of `== 2`;
    `steps/intake_step.py`'s prompt updated to match.
  - `templates/sources/xgboost_mixed.py` and `harness/baseline_ladder.py`
    both hardcoded `eval_metric="logloss"`, which is invalid for a
    multiclass fit and would crash XGBoost. Removed entirely (not
    replaced with a conditional) — XGBoost infers the correct objective
    itself from the number of classes it sees in `y` at fit time.
  - Notebook's ROC-curve cell is binary-only by construction
    (`sklearn.metrics.roc_curve` doesn't support multiclass); it now
    checks the test set's class count and shows an informational note
    instead of a curve for 3+ classes, while the confusion-matrix panel
    next to it is unchanged (already generic over class count).
  - A second, unrelated real bug surfaced only once Iris cleared
    intake: `harness/profiler.py`'s `_name_hints()` did a raw substring
    check, so `"id" in "sepalwidthcm"` matched (the "id" inside
    "**Wid**th") and flagged `SepalWidthCm`/`PetalWidthCm` as
    likely-ID/group columns — excluding two of Iris's four real
    features from every candidate and failing the run. Fixed by
    tokenizing the column name (snake_case/camelCase-aware) and
    matching whole tokens instead of substrings; also incidentally
    fixes the same class of false positive for e.g. `GROUP_NAME_HINTS`
    `"session"` matching a column literally named `obsession_score`.
  - `tests/test_harness.py`, `tests/test_intake.py`,
    `tests/test_templates.py` — multiclass unit tests (metric
    correctness against synthetic 3-class data, intake range-check
    acceptance/rejection, every template builds and fits on a 3-class
    target). Verified end to end with a real dry run of
    `run_orchestrator.py` against `datasets/raw/iris.csv` (stubbed
    `ModelClient`, real intake → feature-engineering → profiler →
    modeling → verification → test-eval loop) — the exact scenario
    originally reported as a crash now completes successfully.

**Time-series featurization (extensibility addition, pre-processing —
runs *before* the pipeline, not inside it):** the first non-tabular
dataset tried was a real aviation predictive-maintenance dataset
(NGAFID-MC): 22 sensor channels sampled ~1Hz, ~6363 timesteps per
flight, many flights per plane, concatenated into one long CSV
identified by a flight `id` column. A standard row-per-example split
can't run against data shaped like that — there's no "one row" yet.
- `harness/timeseries_features.py` — dataset-agnostic, deterministic
  (no LLM) rollup: `channel_features()` computes 12 summary stats
  (mean/std/min/max/range/p10/p50/p90/slope/mean_abs_diff/max_abs_diff/
  last) per channel; `build_flight_feature_table_streaming()` streams a
  long-format CSV chunk-by-chunk and emits exactly one row per id, so
  memory use is bounded by chunk size, not file size (the real source
  file is 4.2GB). The streaming reader groups by contiguous *run*
  position, not id value, specifically so that two separate,
  non-contiguous runs of the same id (a real corruption case, not just
  a hypothetical) can't silently merge into one row — an earlier,
  simpler version of this check (matching on the trailing chunk's id
  value) let exactly that slip through undetected; caught by
  `tests/test_timeseries_features.py::test_streaming_non_contiguous_id_raises`
  during development, not after. No dataset-specific column names or
  label vocabulary live in this module — see the adapter below for
  those — so it's reusable for any future long-format grouped
  time-series dataset, not just this one.
- `scripts/featurize_ngafid_flights.py` — the NGAFID-specific adapter:
  the 22 sensor column names, `before_after` label-string resolution
  (`before`/`pre`/`1`/`true` → 1, `after`/`post`/`0`/`false` → 0), and a
  defensive exactly-2-classes check (since passing `--target` straight
  to `run_orchestrator.py` skips intake's own class-count validation).
  Run once, standalone, to produce an ordinary flight-level tabular CSV;
  nothing downstream changes — the existing group-aware split strategy
  (`--group-column plane_id --strategy group`) and `--target` CLI
  plumbing already handle it, because after this step the data is just
  a normal classification table.
- `tests/test_timeseries_features.py` — channel-stat correctness
  (including empty/all-NaN/single-value edge cases), and the
  streaming-vs-naive-groupby equivalence test described above, run with
  a deliberately tiny `chunksize` to force multiple ids to straddle
  chunk boundaries — the exact scenario the carry-over logic exists to
  handle correctly.

**Deep-Dive agent (sixth agent; extensibility addition — a second
analysis problem, "why was this flight flagged," not a step in the
classification loop):** ported from a working prototype in the sibling
`aviation_mas_mvp/` project's `scripts/deep_dive/`, following the exact
same trust-boundary pattern as every other agent here — one bundled
tool exposing deterministic evidence the agent cannot alter, a strict
JSON response contract, and a conservative deterministic fallback on
anything unparseable.
- `harness/attribution.py` — generic (not aviation-specific) occlusion
  attribution: `channel_of()`, `compute_background()`,
  `attribute_prediction()`. Generalized from the original prototype,
  which assumed a raw classifier taking a pre-aligned numpy array; this
  repo's accepted candidates are sklearn `Pipeline`s taking a named
  DataFrame (they do their own imputation/scaling internally), so the
  port occludes columns *by name* in a single-row DataFrame and calls
  `pipeline.predict_proba()` instead — works for any accepted candidate
  from this pipeline, tabular or engineered-time-series alike, not just
  the aviation dataset it was first built against. Binary-classification
  only for now (a scope limit, not a silent multiclass wrong-answer).
- `domain/aviation/flight_phases.py` (`segment_flight`) and
  `domain/aviation/anomaly_localization.py` (`localize_anomaly`) —
  straight ports; these genuinely are aviation-specific (hardcoded
  `AltMSL`/`IAS`/`VSpd`/`E1 EGT<n>`/`E1 CHT<n>` column names), so they
  live under a new `domain/` package rather than `harness/`, keeping the
  same separation Phase 1 established between generic and
  dataset-specific code.
- `harness/timeseries_features.py` gained `extract_single_flight_raw()`
  — streams a raw long-format CSV for just one flight's timesteps
  (reusing the existing chunked reader), so the deep-dive agent doesn't
  need to load a multi-GB file to explain one flagged flight.
- `tools/deep_dive_tool.py` + `steps/deep_dive_step.py` — one bundled
  tool, `get_flight_deep_dive_evidence` (segmentation → attribution →
  localization, composed exactly as the original prototype's
  `gather_evidence` did), then a single LLM call synthesizing a hedged
  2-4-sentence hypothesis as JSON. An unparseable/malformed response
  degrades to a deterministic template built from the same evidence
  (ported from the prototype's own template fallback) rather than
  failing outright — same conservative default this repo uses
  everywhere.
- `scripts/run_deep_dive_agent.py` — standalone CLI, same pattern as
  `run_profiler_agent.py`. Needs a persisted model bundle (see below).
- `run_orchestrator.py` gained one small additive step: after the final
  test-set evaluation, it now saves the accepted, refit pipeline to
  `artifacts/models/<run_id>_model.joblib` (bundled with its feature
  columns and a healthy-cohort background reference) — previously the
  fitted model was never saved anywhere. Skipped for a multiclass target
  (attribution is binary-only for now). This is what
  `run_deep_dive_agent.py` loads to explain a specific later flight.
- `tests/test_deep_dive.py` — phase segmentation and short-run merging,
  a planted single-cylinder fault correctly detected *and* a benign
  constant cross-cylinder offset correctly rejected (the specific
  false-positive claim `localize_anomaly`'s docstring makes), an
  attribution regression test (adapted from the prototype's own
  controlled test) proving the DataFrame-occlusion generalization still
  recovers a planted fault channel when fit through one of this repo's
  own real template pipelines (not a bare classifier), and the
  unparseable-response-degrades-to-template case.

**Not yet built:** priors/evidence reuse (Phase 6), parallelization
(Phase 7), dynamic agent orchestration (the orchestrator is still a
fixed sequence of phases, not an agent that decides which agents/
pipelines to run for a given problem — the deep-dive agent above is
triggered manually, per flight, not routed to automatically). Multiclass
occlusion attribution is also unbuilt (a scope limit noted above, not a
crash). All intentionally on hold until the pipeline has been proven
against more real datasets and problem types.

## Quickstart

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in RIT_API_KEY

# 2. Prove RIT connectivity (replaces the OpenClaw milestone)
set -a; source .env; set +a
python scripts/check_rit_connection.py

# 3. (optional) stand up the gateway once you have multiple callers
docker compose up -d model-gateway
python scripts/check_gateway_connection.py

# 3.25. (optional) or point every script at a local OpenAI-compatible
#       server instead (vLLM/Ollama/SGLang) via --use-local — sidesteps
#       RIT's real, intermittent 504s under agentic tool-calling load
python scripts/check_local_connection.py

# 3.5. (only for long-format, grouped time-series data, e.g. one row per
#      timestep with many timesteps per example) roll it up into one row
#      per example first — no LLM, run once, standalone:
python scripts/featurize_ngafid_flights.py \
    --csv /path/to/raw_ngafid.csv \
    --out datasets/processed/ngafid_flights.csv \
    --max-flights 100   # smoke test a slice first; omit for the full file
#      the resulting CSV is ordinary tabular data — every step below
#      works on it unchanged.

# 4. Run the deterministic harness against a real dataset — no LLM needed
python scripts/run_baseline_ladder.py \
    --data datasets/raw/your_dataset.csv \
    --target your_target_column \
    --group-column customer_id \
    --time-column event_date \
    --strategy group_time

# 5. Run the profiler agent (Phase 2 — one deterministic tool call + LLM narrative)
python scripts/run_profiler_agent.py \
    --data datasets/raw/your_dataset.csv \
    --target your_target_column

# 5.5. Run the feature engineering agent standalone (proposes columns to
#      drop + derived features from a vetted catalog; writes a CSV preview)
python scripts/run_feature_engineering_agent.py \
    --data datasets/raw/your_dataset.csv \
    --target your_target_column

# 6. Run the modeling agent (Phase 3 — agent picks a template + fills config,
#    harness validates/builds/scores/double-leakage-gates it, then Phase 4's
#    VerificationAgent reviews it, before the leaderboard append)
python scripts/run_modeling_agent.py \
    --data datasets/raw/your_dataset.csv \
    --target your_target_column \
    --group-column customer_id \
    --time-column event_date \
    --strategy group_time

# 7. Run the full orchestrator loop (Phase 5 — intake -> profiler ->
#    N gated modeling candidates -> select-and-verify, best-first, with
#    fallback on a VerificationAgent rejection -> one locked test-set
#    eval -> narrated summary). Omit --target to let intake infer it
#    from --goal (or from the schema alone if --goal is also omitted).
python scripts/run_orchestrator.py \
    --data datasets/raw/your_dataset.csv \
    --goal "predict your_target_column" \
    --max-candidates 2

# 8. Or walk through the same loop interactively, one phase per cell,
#    with inline tables/ROC curve instead of print statements:
jupyter notebook notebooks/end_to_end_pipeline.ipynb

# 8.5. (aviation/time-series datasets only, binary target) explain why
#      one already-flagged flight was flagged — needs the model bundle
#      step 7 just saved to artifacts/models/<run_id>_model.joblib:
python scripts/run_deep_dive_agent.py \
    --model artifacts/models/<run_id>_model.joblib \
    --raw-csv /path/to/raw_ngafid.csv \
    --features-csv datasets/processed/ngafid_flights.csv \
    --flight-id 5

# 9. Run the test suite
pytest tests/ -v
```

## Repo structure

```
agentic-ml/
  docker-compose.yml       # LiteLLM gateway only
  .env.example
  requirements.txt

  configs/
    litellm.yaml           # model routing + fallback chains
    schemas/                # JSON schemas for dataset/candidate specs

  priors/
    general/                # baseline ladder, leakage rules (Rule 1-6)
    anti_patterns/           # documented failure modes + how the harness prevents them

  src/agentic_ml/
    model_client.py         # OpenAI-compatible client, stateless
    agent_runtime.py        # tool-calling loop, no session/compaction state
    cli_common.py            # shared script plumbing (model endpoint, run dirs)
    harness/                 # the trust boundary — see Phase 1 above; also
                               # intake.py (Phase 5 pre-target schema facts),
                               # verification.py (Phase 4 review bundle
                               # assembly), feature_engineering.py (vetted
                               # stateless derived-feature/drop-column ops),
                               # and timeseries_features.py (dataset-agnostic
                               # long-format time-series -> flight-level
                               # tabular rollup — pre-processing, run
                               # standalone before the pipeline, not a step
                               # inside it; also attribution.py — generic
                               # occlusion attribution used by the deep-dive
                               # agent, not aviation-specific)
    domain/aviation/          # aviation-specific deep-dive measurements:
                               # flight_phases.py (segment_flight),
                               # anomaly_localization.py (localize_anomaly) —
                               # separate from harness/ because these hardcode
                               # aviation column names, unlike attribution.py
    tools/                    # profiler_tool.py, template_tool.py, intake_tool.py,
                               # verification_tool.py, feature_tool.py,
                               # deep_dive_tool.py — thin bindings exposing
                               # harness facts/registry data to agents as tool
                               # calls
    templates/                # Phase 3 recipe templates + registry.py
      sources/                 # verified build_pipeline(config) .py files
    steps/                    # intake_step.py, profiler_step.py, modeling_step.py,
                               # verification_step.py, feature_engineering_step.py,
                               # deep_dive_step.py — agent-loop + validation
                               # logic, shared by both the standalone scripts
                               # and run_orchestrator.py

  scripts/
    check_rit_connection.py
    check_gateway_connection.py
    check_local_connection.py
    run_baseline_ladder.py
    run_profiler_agent.py     # thin CLI wrapper around steps/profiler_step.py
    run_feature_engineering_agent.py  # thin CLI wrapper around steps/feature_engineering_step.py
    run_modeling_agent.py     # thin CLI wrapper around steps/modeling_step.py
    run_orchestrator.py       # Phase 5 — the full loop, see above; also
                               # persists the accepted model bundle to
                               # artifacts/models/<run_id>_model.joblib
    featurize_ngafid_flights.py  # NGAFID-specific adapter around
                               # harness/timeseries_features.py — run once,
                               # standalone, before the orchestrator
    run_deep_dive_agent.py    # thin CLI wrapper around steps/deep_dive_step.py
                               # — explains one already-flagged flight,
                               # on-demand, using a saved model bundle

  notebooks/
    end_to_end_pipeline.ipynb # same steps/* functions as the scripts, one
                               # phase per cell, inline tables + ROC curve

  datasets/raw/              # put input CSV/Parquet here
  runs/                       # per-run trace.jsonl, split_manifest.json,
                               # transcripts/<agent>_NN.json, etc.
  artifacts/reports/          # leaderboard.jsonl lives here
  artifacts/models/           # <run_id>_model.joblib bundles (model +
                               # feature_columns + background), one per
                               # successful run with a binary target — what
                               # the deep-dive agent loads
  tests/                      # pytest suite + leaky_fixtures/ regression data
```

## Design principles carried over from the original architecture doc

1. **Agents propose, harness evaluates.** No agent ever computes final
   metrics, picks its own split, or sees the locked test set.
2. **Group/time-aware splitting from day one** — no silent default to
   random splitting when a group or time column is declared.
3. **Filesystem-first.** No database until the workflow is stable.
4. **Priors guide agents, they don't replace evaluation.** Templates
   and evidence inform candidate proposals; the harness still scores
   everything independently.
5. **Untrusted code is sandboxed twice**: static AST rejection before
   execution, subprocess isolation with resource limits during
   execution. Candidates receive a `config` dict and return an
   unfitted pipeline — never a file path, never raw data.
6. **A second-opinion agent can only veto, never approve.** The
   verification agent (Phase 4) reviews candidates that already passed
   every deterministic gate; it can flag or reject one, but it is never
   shown — and therefore can never unblock — a candidate that already
   failed a gate.

