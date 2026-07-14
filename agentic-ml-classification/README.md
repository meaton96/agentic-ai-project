# agentic-ml

Agentic ML classification pipeline. Constrained MVP scope: CSV/Parquet
input, binary classification, tabular data, sklearn/LightGBM/XGBoost,
single-machine CPU, deterministic harness-owned evaluation. Agents
never get direct access to test labels or final scoring logic.

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

## What's built so far

**Phase 0 (connectivity) — replaces the OpenClaw bootstrap milestone:**
- `src/agentic_ml/model_client.py` — thin `openai` SDK wrapper, works
  against RIT directly or the LiteLLM gateway
- `src/agentic_ml/agent_runtime.py` — minimal tool-calling loop, no
  session state, tested against a mock model client (13/13 unit tests
  pass without needing network access)
- `scripts/check_rit_connection.py` — direct RIT smoke test
- `scripts/check_gateway_connection.py` — gateway-mediated smoke test
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
  fold generation
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
  accuracy, brier, each with bootstrap confidence intervals
- `harness/sandbox.py` — AST static checks (forbidden imports/calls)
  + subprocess isolation with wall-clock timeout and POSIX
  CPU/memory rlimits for untrusted candidate code; candidates only
  return an unfitted pipeline object, never touch raw data
- `harness/leaderboard.py` — append-only, file-lock-safe JSONL leaderboard
- `scripts/run_baseline_ladder.py` — Phase 1 CLI entry point, runs the
  whole pipeline (load → split → leakage checks → baseline ladder →
  metrics → leaderboard) with **zero agent/LLM involvement**
- `tests/` — 14 passing tests, including a regression fixture
  (`tests/leaky_fixtures/obvious_feature_leak.csv`) that must always
  be caught by the leakage checks

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
  config, fits/scores on the harness-owned split, runs
  `label_permutation_test` as a leakage gate, and only appends to the
  leaderboard if that gate passes. The agent picks a recipe; it never
  gets to grade its own homework.

**Phase 5 (Orchestrator) — fully built and tested, skipping ahead of
Phase 4 by design (see below):**
- `harness/intake.py` — `raw_schema_summary()` computes column facts
  with NO target column assumed (dtype, missingness, cardinality,
  name-hint flags for id/group/datetime — deliberately excludes
  anything target-dependent, since intake's whole job is to propose
  the target); `validate_dataset_spec_proposal()` re-validates the
  agent's proposed target/id/group/time columns against those facts
  and enforces the binary-classification-only MVP constraint (target
  must have exactly 2 non-null unique values) regardless of what the
  agent claims.
- `tools/intake_tool.py` — `get_raw_schema` tool, same pattern as the
  other two tools.
- `steps/{intake_step,profiler_step,modeling_step}.py` — the agent-loop
  + validation + sandbox-build + fit/score + leakage-gate logic for
  each phase, extracted out of the standalone scripts into plain
  functions so `run_orchestrator.py` and the standalone scripts drive
  identical logic instead of duplicating it.
- `cli_common.py` — small shared plumbing (model endpoint resolution,
  run-dir/trace-log setup) that all three scripts need.
- `scripts/run_orchestrator.py` — the first entry point that runs the
  full loop: **intake** (skipped if `--target` is given) → **profiler**
  (its `recommended_split_strategy` drives the split unless overridden)
  → up to `--max-candidates` **modeling** proposals, nudged toward
  trying a different template each time → select the best candidate
  that passes the label-permutation gate → refit it on train+val → one
  locked test-set evaluation → a short LLM-narrated plain-text summary
  (no tools, no decisions — it only narrates already-computed facts).
  Works from just `--data` (intake guesses a target from the schema
  alone), from `--data --goal "<natural language>"`, or from an
  explicit `--target` that bypasses intake entirely.
- `tests/test_orchestrator.py` — integration tests that drive the
  *entire* loop end to end against a synthetic dataset with a stubbed
  `ModelClient` (dispatch by system-prompt identity + message count, no
  real network), covering all three entry modes above. This is the
  automated answer to "does prompt (or just a dataset) + dataset →
  finished classification actually work."

Why Phase 5 before Phase 4: the deterministic gates that already exist
(sandbox static checks + `label_permutation_test`) block leaky/broken
candidates without needing an LLM to audit them, so closing the full
loop end to end surfaces more real integration bugs sooner than adding
a dedicated verification agent would. **Not yet built:** a dedicated
verification/audit agent (Phase 4), priors/evidence reuse (Phase 6),
parallelization (Phase 7). Also worth knowing: a real dry run against
the Titanic dataset surfaced `label_permutation_test`'s default
`n_permutations=5` occasionally rejecting a legitimate (non-leaky)
candidate by chance on smaller datasets — the gate is working as
designed, but that default may be worth revisiting.

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

# 6. Run the modeling agent (Phase 3 — agent picks a template + fills config,
#    harness validates/builds/scores/leakage-gates it before the leaderboard)
python scripts/run_modeling_agent.py \
    --data datasets/raw/your_dataset.csv \
    --target your_target_column \
    --group-column customer_id \
    --time-column event_date \
    --strategy group_time

# 7. Run the full orchestrator loop (Phase 5 — intake -> profiler ->
#    N modeling candidates -> selection -> one locked test-set eval ->
#    narrated summary). Omit --target to let intake infer it from
#    --goal (or from the schema alone if --goal is also omitted).
python scripts/run_orchestrator.py \
    --data datasets/raw/your_dataset.csv \
    --goal "predict your_target_column" \
    --max-candidates 2

# 8. Or walk through the same loop interactively, one phase per cell,
#    with inline tables/ROC curve instead of print statements:
jupyter notebook notebooks/end_to_end_pipeline.ipynb

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
                               # intake.py (Phase 5 pre-target schema facts)
    tools/                    # profiler_tool.py, template_tool.py, intake_tool.py
                               # — thin bindings exposing harness facts/registry
                               # data to agents as tool calls
    templates/                # Phase 3 recipe templates + registry.py
      sources/                 # verified build_pipeline(config) .py files
    steps/                    # intake_step.py, profiler_step.py, modeling_step.py
                               # — agent-loop + validation logic, shared by both
                               # the standalone scripts and run_orchestrator.py

  scripts/
    check_rit_connection.py
    check_gateway_connection.py
    run_baseline_ladder.py
    run_profiler_agent.py     # thin CLI wrapper around steps/profiler_step.py
    run_modeling_agent.py     # thin CLI wrapper around steps/modeling_step.py
    run_orchestrator.py       # Phase 5 — the full loop, see above

  notebooks/
    end_to_end_pipeline.ipynb # same steps/* functions as the scripts, one
                               # phase per cell, inline tables + ROC curve

  datasets/raw/              # put input CSV/Parquet here
  runs/                       # per-run trace.jsonl, split_manifest.json, etc.
  artifacts/reports/          # leaderboard.jsonl lives here
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

