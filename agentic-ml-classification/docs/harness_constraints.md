# Harness Constraints, Step by Step

*A concrete inventory of what the deterministic
harness enforces at each stage of the pipeline, and what it structurally
forbids an LLM agent from doing. This is not a proposal; every constraint
below is implemented and covered by tests as of this writing. File:line
references point at `src/agentic_ml/` unless noted.*

## How to read this document

The pipeline alternates between **agent steps** (an LLM proposes a JSON
object) and **harness steps** (deterministic code validates or rejects that
proposal, or performs a computation with no LLM involved at all). For each
step below: what the agent is allowed to see, what it's allowed to propose,
and — the part relevant to "is there a publishable architecture here" —
exactly what deterministic code stands between that proposal and any effect
on the result.

The overarching claim, stated once so it doesn't need repeating per step:
**every agent output is a proposal, never an action.** No LLM call happens
inside `harness/`, `domain/`, or the plan validator
(`orchestrator/dynamic_loop.py`) — see `CLAUDE.md` invariant #2. If a task
ever seems to require an LLM to compute a metric, decide a split, or approve
a candidate, that's treated as a design violation, not an implementation
detail.

---

## Step 0 — Dataset loading

**Nature:** pure harness, no agent involved.

- Loads CSV/Parquet and computes a SHA-256 content hash of the exact bytes
  used (`harness/dataset.py`), so any downstream result is traceable to
  specific data, not a filename that may have changed since.
- No LLM sees the raw dataframe at this stage — only facts computed *about*
  it later (Step 1).

**Why it's a constraint, not a convenience:** without the hash, "we ran this
on the churn dataset" is not a reproducible claim. With it, it is.

---

## Step 1 — Intake Agent (target selection)

**Agent sees:** column names, dtypes, missingness, cardinality, and
name-based hints — computed *before* any target is known. It never sees
class balance or feature-target correlation, because those would require
already knowing the target, which is circular for the thing it's proposing.

**Agent proposes:** `{target_column, task, id_columns, group_column,
time_column, positive_label, reasoning}`.

**Harness constraints (`harness/intake.py::validate_dataset_spec_proposal`):**
- `MAX_CLASSES = 20` (`harness/intake.py:25`) — the target column must have
  between **2 and 20** non-null unique values. Below 2, there's no
  classification problem; above 20, the harness assumes the agent
  mis-identified a continuous or ID column as a label, and rejects it
  outright (`harness/intake.py:82-85`).
- The harness does not trust the agent's self-reported `task` field
  (binary vs. multiclass) — it derives that itself from the actual unique
  count.
- Proposed `id_columns`, `group_column`, `time_column` must reference real
  columns in the dataframe; a hallucinated column name is rejected, not
  silently ignored.

---

## Step 2 — Feature Engineering Agent

**Agent sees:** the same profiler facts as Step 3, plus a fixed catalog of
five operations (`harness/feature_engineering.py:25`, `FEATURE_OPS`):
`ratio`, `interaction`, `log1p`, `datetime_parts`, `missing_indicator`.

**Agent proposes:** `{drop_columns, derived_features, explanation}` — a list
of columns to drop and a list of `{op_id, params}` selected from the fixed
catalog. **It cannot write transformation code.**

**Harness constraints (`harness/feature_engineering.py::validate_feature_proposal`,
line 140):**
- `op_id` must be a real catalog entry (line 194) — not an invented
  operation.
- Every referenced column must exist and satisfy the op's type requirement:
  numeric ops (`ratio`, `interaction`, `log1p`) require an actual numeric
  *dtype* check (`_is_numeric_dtype`, line 63), not a cardinality heuristic
  borrowed from an unrelated part of the profiler (see failure mode below).
  `datetime_parts` requires a datetime-like column (line 220).
- The **target column is forbidden as input to any op** (line 211) — an
  agent cannot construct a feature that is a disguised copy of the label.
- The declared group/time column cannot be dropped.
- **Statelessness is structural, not policed by convention:** all five ops
  compute a value from that row's own columns only (no fitted mean,
  quantile, or per-group aggregate is available in the catalog at all). This
  is what makes it safe to apply the whole feature-engineering step to the
  full dataset *before* the train/val/test split exists — anything that
  needs a fitted statistic (imputation, scaling, target encoding) is
  confined to the modeling templates' `ColumnTransformer`, fit only on the
  training fold, in Step 5.

**Failure mode this closed:** the validator originally gated numeric ops on
the profiler's `is_likely_numeric` flag — a heuristic built for a *different*
decision (one-hot vs. scaling in Step 5's templates) that is `False` for any
low-cardinality integer column, including genuine counts. This rejected a
correct, well-known engineered feature (`SibSp * Parch` on Titanic). Fixed
by checking actual dtype instead of reusing a heuristic tuned for an
unrelated question. General lesson documented in the code: a validation rule
should encode "would this literally break," not silently inherit a proxy
that happens to be lying around.

---

## Step 3 — Profiler Agent

**Agent sees:** one bundle of deterministic facts — column types,
missingness, cardinality, likely id/group/datetime flags, class-imbalance
ratio, and a rule-based `recommended_split_strategy`.

**Agent proposes:** a plain-language summary and a list of risks.

**Harness constraint — this is the strongest one in the whole system:** the
agent is *instructed* not to override `recommended_split_strategy` or the
leakage flags, but more importantly, **nothing downstream ever reads an
agent-authored split strategy in the first place.** The split strategy used
in Step 4 comes from the rule-based function, full stop — the profiler
agent's output is narration for a human reader, not an input to any
decision. There is no code path from this agent's JSON to `make_split()`'s
`strategy` argument.

**A real reconciliation bug this surfaced
(`harness/splits.py::resolve_split_columns`, line 32):** the profiler's
*recommendation* is computed by its own heuristic column detection —
independent of whatever `id_columns`/`group_column`/`time_column` intake
actually declared. Running with an explicit `--target` (skipping intake) and
no `--time-column` could get a `"time"` recommendation with no declared time
column to satisfy it, crashing deep in `make_split()`. `resolve_split_columns`
reconciles the two using the same evidence that produced the recommendation,
rather than crashing or silently guessing.

---

## Step 4 — Split + leakage checks (pure harness, no agent)

**Constraint: five split strategies, explicit selection required
(`harness/splits.py:29`, `VALID_STRATEGIES`):**
`{random, stratified, group, time, group_time}`. Choosing `group` or
`group_time` without a declared `group_column` raises
(`harness/splits.py:140-141`); same for `time`/`group_time` without a
`time_column` (line 142-143). **There is no silent fallback to a random
split** when a group or time column has been declared — the harness treats
an unsatisfiable strategy as an error to surface, not a default to paper
over.

**Five independent leakage checks run automatically, before any model
fitting:**
1. `check_duplicate_rows_across_splits` — exact-duplicate rows spanning
   train/val/test.
2. `check_group_overlap` — the same group-id present in more than one split.
3. `check_time_ordering` — chronological violation for time-based splits.
4. `check_fold_class_presence` — every class present in every fold.
5. `check_suspicious_feature_correlation` (`harness/leakage.py:131`,
   `threshold=0.98`) — any raw feature with `|correlation| >= 0.98` against
   the target is flagged as a likely leaked/proxy column.

These run on the raw split, independent of anything a modeling candidate
later does — see Step 5 for the sixth check, which is candidate-specific.

---

## Step 5 — Modeling Agent + sandboxed build + two leakage gates

**Agent sees:** profiler facts plus a fixed catalog of six pre-built
"recipe templates" (`templates/registry.py:50`): `logistic_numeric`,
`sklearn_mixed_pipeline`, `lightgbm_mixed`, `xgboost_mixed`,
`imbalanced_binary_boosted`, `high_cardinality_target_encoding`.

**Agent proposes:** `{candidate_id, template_id, config, explanation}` —
picks a template by name, fills in a config dict (column roles, a few
hyperparameters). **It does not write model code.** This is the single
biggest design constraint in the project: free-form generated sklearn code
was explicitly rejected as too hard to statically verify for leakage-safety,
and a single fixed pipeline was rejected as leaving no real decision for the
agent. The template catalog is the deliberate middle ground.

**Harness constraints, applied in order, each a hard stop:**

1. **Shape validation** (`steps/modeling_step.py:49`,
   `_validate_candidate_spec_shape`) — well-formed JSON with required keys.
2. **Column validation** (`steps/modeling_step.py:65`,
   `_validate_candidate_columns`) — every referenced column must exist per
   the profiler's facts, and must not be the target, id, group, or time
   column. A hallucinated column name is rejected here.
3. **Config validation** (`templates/registry.py::validate_config`) — the
   config must satisfy the chosen template's required keys.
4. **Static AST check** (`harness/sandbox.py:75`, `static_check`) — before
   any execution, reject:
   - Forbidden imports (`harness/sandbox.py:61`): `os`, `sys`, `subprocess`,
     `socket`, `shutil`, `requests`, `urllib`, `http`, `ftplib`,
     `telnetlib`, `smtplib`, `ctypes`, `multiprocessing`, `pty`, `pickle`.
   - Forbidden calls (`harness/sandbox.py:66`): `eval`, `exec`,
     `__import__`, `compile`, `open`, `input`.
5. **Sandboxed subprocess execution** (`harness/sandbox.py::run_candidate_build`,
   line 151) — the statically-cleared template code runs in an isolated
   subprocess with:
   - `timeout_seconds=30` wall-clock (default),
   - `cpu_seconds=20` CPU-time `RLIMIT_CPU` cap (default),
   - BLAS/OpenMP thread pools forced to 1 (`OPENBLAS_NUM_THREADS`,
     `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS` all set to
     `"1"`),
   - no network access exposed,
   - **only an unfitted pipeline object crosses the process boundary** —
     never raw data, never a file path. (No `RLIMIT_AS` memory ceiling is
     set — documented as a deliberate exception, since BLAS-backed libraries
     reserve large virtual address ranges regardless of actual memory used;
     CPU-time capping is the real containment for compute abuse.)
   - A structural pickling rule falls out of this boundary: templates may
     only compose pipelines from objects in real, importable libraries — a
     template defining a custom transformer class inline failed here during
     development, because Python can't reliably pickle a class defined
     inside dynamically executed code across a process boundary.
6. **Fit on train fold, score on validation fold** with bootstrapped
   confidence intervals (`harness/metrics.py::compute_metrics`,
   `n_bootstrap=1000` default) — a point estimate alone is not reported;
   every metric ships a CI.
7. **Label-permutation leakage gate**
   (`harness/leakage.py::label_permutation_test`, line 165) — fits the
   *actual candidate pipeline* on shuffled training labels
   (`n_permutations=5`, `chance_tolerance=0.08`, default `seed=42`). Passes
   only if mean shuffled-label score stays within `0.08` of chance
   (`~0.5`). Catches leakage in the *process* — e.g. a scaler or encoder
   fit outside its proper training-fold scope.
8. **Candidate-scoped feature-correlation gate**
   (`harness/leakage.py::check_suspicious_feature_correlation`, same
   `threshold=0.98`, re-run here scoped to exactly the columns *this*
   candidate selected) — catches leakage in raw *feature content*, e.g. a
   near-duplicate-of-target column the candidate happened to select. A test
   in the suite proves these two gates are complementary, not redundant: a
   candidate selecting a near-perfect proxy column passes the permutation
   test (shuffled labels make the proxy equally useless) but is caught by
   the correlation gate.

**Only a candidate that clears all eight of these layers becomes eligible
for Step 6.**

---

## Step 6 — Verification Agent (one-way ratchet)

**Agent sees:** a bundle assembled entirely from facts already computed
elsewhere — the template's own description, the candidate's config and
stated explanation, validation metrics with CIs, both leakage-gate results,
and the profiler's imbalance/risk facts. It computes nothing new itself.

**Agent proposes:** `{verdict, concerns, reasoning}` where `verdict` is one
of `"approved"`, `"flagged"`, or `"rejected"`.

**The constraint that makes this a genuine trust boundary, not a rubber
stamp:** the harness only ever shows this agent candidates that have
*already* passed all eight Step-5 gates. Its output schema has no
"override" option — there is no field it can set that unblocks a
gate-failed candidate. Structurally:

- It can move a passing candidate to `"flagged"` (proceeds, but is logged
  as a caveat) or `"rejected"` (blocked, falls back to the next-best
  gate-passing candidate).
- It **cannot** move a candidate the other direction. There is no code path
  from this agent's output back to Step 5's gates.
- **Malformed/unparseable LLM output degrades to `"flagged"`**, never to
  `"approved"` and never a hard crash — the same fail-conservative pattern
  used everywhere an LLM output fails to parse in this system.

**Orchestrator policy:** candidates are reviewed **best-first, by validation
score** — not "verify everyone, then pick the best." A rejection tries the
next-best gate-passer instead of aborting the run; if every gate-passing
candidate is rejected, the run stops and **the test set is never touched.**

---

## Step 7 — Finalize (test-set evaluation, pure harness)

**Constraint: the test set is touched exactly once per run.** Enforced by a
one-shot guard, `final_test_metrics_present`
(`steps/finalize_step.py:12`), which the dynamic orchestrator's
precondition system also checks before allowing the `finalize` action to
run at all (`orchestrator/agent_registry.py`). `CLAUDE.md` states this as a
non-negotiable invariant (#3): no API endpoint or refactor may add a second
test-evaluation path.

Sequence: refit the accepted candidate on train+validation combined →
evaluate once on the test set locked since Step 4 → (binary target only)
persist the fitted pipeline to `artifacts/models/<run_id>_model.joblib`.

---

## Step 8 — Summarization Agent (narration only)

**Agent sees:** the final metrics and verdict.

**Agent proposes:** a plain-language paragraph.

**Constraint:** no tools, no ability to alter any prior result. If the
accepted candidate was `"flagged"` rather than cleanly `"approved"`, the
prompt requires that caveat be mentioned. This step cannot feed back into
anything upstream — it's a pure sink.

---

## Step 8.5 — Dynamic orchestrator: the planner is bound by the same rules

The dynamic orchestrator (`scripts/run_dynamic_orchestrator.py`) replaces
the fixed step sequence above with a planning agent that picks the next
action from a registry. The constraint pattern repeats one level up:

- **Agent catalog** (`orchestrator/agent_registry.py`) — each of the nine
  capabilities (intake, feature engineering, profiler, split, modeling,
  verification, finalize, deep-dive, summarize) is a fixed, named registry
  entry with `required_state` preconditions (e.g. modeling requires
  `split_leakage_passed=True`; finalize requires
  `final_test_metrics_present=False`, enforcing the same one-touch rule as
  Step 7).
- **Planner agent** (`steps/planner_step.py`) sees a compact, JSON-safe
  summary of run state — never raw dataframes or fitted pipelines — and
  proposes exactly one next action.
- **Plan validator** (`orchestrator/dynamic_loop.py::validate_plan`) —
  re-checks the planner's proposed agent id against the *real* registry and
  the claimed preconditions against the *real* run state. A rejected
  proposal is never executed; it's fed back as an error with a bounded
  retry count. `CLAUDE.md` invariant #8 specifically constrains the one
  normalization helper (`normalize_proposal`) that repairs known formatting
  slips, so it cannot be widened into something that repairs a genuinely
  unknown/hallucinated action — a regression test
  (`test_normalize_proposal_does_not_repair_a_genuinely_unknown_action`)
  guards this directly.

**Why this matters for the "is it publishable" question:** this is the
place the architecture had to answer a new question a fixed pipeline never
faced — a hallucination at the control-flow level doesn't just produce a
wrong number, it could skip a safety gate entirely. The answer applied is
the same propose-then-independently-verify discipline used everywhere else
in the system, just one level higher (agent choice of *action*, not just
content).

---

## Cross-cutting constraints (apply to every step above)

| Constraint | Where enforced |
|---|---|
| No LLM call inside the trust boundary | `harness/`, `domain/`, `orchestrator/dynamic_loop.py` — `CLAUDE.md` invariant #2 |
| Agents never see raw file paths, dataframes, or fitted pipelines | Every tool binding in `tools/*.py` returns harness-computed facts only |
| Unparseable agent output fails conservatively | `"flagged"` (verification), a deterministic template (deep-dive), bounded retry (planner) — never silent approval, never a crash |
| Seeded, reproducible splits and metrics | `harness/splits.py`, `harness/metrics.py` — fixed default seeds throughout |
| Filesystem-first run state | `runs/<run_id>/{events.jsonl, transcripts/, manifests}`, no database |
| Every candidate evaluated is logged, not just the winner | append-only leaderboard (`harness/leaderboard.py`) |

---

## Related work (preliminary literature pass)

*Scope note: this is a first pass over arXiv/alphaXiv (searched
2026-08-10), not a systematic review — it's meant to give faculty
reviewers real anchors to check this project against, not to stand in for
their own literature judgment. Every paper below is a real, citable arXiv
ID; where marked "title/abstract only," content wasn't pulled in full and
the description should be verified before citing in a paper. Four
clusters turned up, in decreasing order of how directly they bear on the
"is there something new here" question.*

### 1. Agent trust-boundary / governance architectures — the closest match

This is the most relevant cluster, and the one worth reading first. A wave
of 2026 papers converges on almost exactly this project's vocabulary and
shape — an LLM as a *proposer* whose outputs pass through a separate,
non-probabilistic authority before they can act — applied to general
agentic systems, not ML pipelines specifically:

- **[Proof-Carrying Certificates for LLM Pipelines: A Trust-Boundary
  Architecture](https://www.alphaxiv.org/abs/2605.16407)** (Koomullil,
  Ascendr, 2026) is the closest single paper found. It formalizes an
  explicit "trust boundary": the LLM sits *below* it as a proposer of
  typed actions/claims, and a Lean 4-verified layer *above* it is "the
  final authority" — language that maps almost directly onto this
  project's "agents propose, harness decides." Its Hoare-style agent
  action certificate (pre/postconditions on tool calls, chained into a
  trajectory certificate) is a formally-proved analog of what this
  project does empirically via column/config/AST/sandbox validation
  (§Step 5, layers 1–5). Concretely, in its own pilot (a sandboxed
  filesystem-agent benchmark), the Lean-checked gate blocked 66.7% of
  injected unsafe actions vs. 27.8% for a regex deny-list and 0% for an
  unaudited agent, with zero false blocks on benign destructive actions.
  **Difference worth naming:** their guarantee is a machine-checked formal
  proof (conditional on named "Tier-4" human/ML oracles they're explicit
  about not being able to discharge); this project's guarantee is
  empirical/statistical (a permutation test, a correlation threshold, a
  sandboxed subprocess) — cheaper to build, weaker in the formal sense,
  but arguably a more realistic match for what an ML pipeline's actual
  failure modes (leakage, not memory-safety) look like.
- **[SAFETY SENTRY: Context-Aware Human Intervention via EXECUTE-ASK-
  REFUSE Routing](https://www.alphaxiv.org/abs/2607.13594)**
  (ShanghaiTech, 2026) reframes agent action review as a three-way
  per-instance routing decision — execute autonomously, ask a human, or
  refuse — instead of a binary safe/unsafe guard. **This is conceptually
  the same three-way shape as the Verification Agent's `approved` /
  `flagged` / `rejected` (§Step 6)**, arrived at independently in a
  completely different domain (enterprise tool-calling agents, not ML).
  Worth citing as convergent evidence that a three-way, per-instance
  verdict is a recurring good answer to "how much should one review step
  trust an agent," not something specific to ML. Difference: SAFETY
  SENTRY's router is itself a trained model deciding *whether* to
  escalate; this project's verification agent is always invoked, and the
  one-way-ratchet property (it cannot promote a gate-failure) is
  structural rather than something the router could get wrong.
- A cluster of adjacent 2026 papers (title/abstract only — not fetched in
  full) sits in the same space and is worth a skim for related-work
  breadth: **[AgentBound: Verifiable Behavioral Governance for Autonomous
  AI Agents](https://www.alphaxiv.org/abs/2606.30970)**, **[Containment
  Verification: AI Safety Guarantees Independent of
  Alignment](https://www.alphaxiv.org/abs/2605.09045)**, **[Governance by
  Construction for Generalist
  Agents](https://www.alphaxiv.org/abs/2605.20874)**, **[Cordon: Semantic
  Transactions for Tool-Using LLM
  Agents](https://www.alphaxiv.org/abs/2606.17573)**, and **[Capability
  Gates Are Not Authorization: Confused-Deputy Failures in LLM Agent
  Frameworks](https://www.alphaxiv.org/abs/2606.28679)** — the last one in
  particular is a useful adversarial check, since it argues that a
  capability gate alone (agent can only call vetted tools) is not the same
  as authorization (the *specific* tool call was actually intended);
  worth checking whether that critique lands against this project's
  column/config validation or is already closed by it.

**This is the strongest candidate for the vision paper's actual
contribution claim**, not the ML pipeline specifics: this project is a
concrete, empirically-gated instance of the "agent proposes, separate
deterministic authority decides" pattern that a same-year literature is
independently converging on for general agentic systems — applied to a
domain (tabular ML, with its own specific failure mode: leakage) that the
governance-architecture papers above don't cover, since they're about tool
misuse/irreversible actions, not statistical validity.

### 2. LLM-agent full-pipeline AutoML — the closest *functional* competitor

- **[AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline
  AutoML](https://www.alphaxiv.org/abs/2410.02958)** (KAIST/DeepAuto,
  2024) — five LLM agents (manager, prompt, data, model, operation) plan
  and then an "Operation Agent" **writes and executes arbitrary Python**
  to produce a deployment-ready model. Verification exists at three
  stages (request/execution/implementation), but it's checking whether
  the plan satisfies the user's stated constraints and scores well — not
  auditing for leakage. No sandboxing or static code-safety check is
  described. Reports strong results (avg. comprehensive score 0.90 vs.
  0.22 for zero-shot GPT-4) on 14 datasets across 5 modalities.
- **[MLE-bench](https://www.alphaxiv.org/abs/2410.07095)** (OpenAI, 2024,
  with DeepMind/Anthropic ties via its stated safety-framework purpose) —
  not a system but the benchmark most of this space is measured against:
  75 real Kaggle competitions, agents (AIDE, MLAB, OpenHands) **write and
  run arbitrary code** in an isolated Docker container purely to maximize
  leaderboard score against a 24h/100h time budget. Best result (o1-preview
  + AIDE) medals in 16.9% of competitions (34.1% at pass@8). Guards against
  cheating (plagiarism detection, rule-violation log scanning) but has no
  analog to a leakage-safety gate — a pipeline that leaks and scores well
  simply scores well.
- **[LightAutoDS-Tab: Multi-AutoML Agentic System for Tabular
  Data](https://www.alphaxiv.org/abs/2507.13413)** (ITMO/Sber AI Lab,
  2025, title/abstract only) — an LLM agent orchestrating *existing*
  classical AutoML tools rather than writing pipeline code itself, aimed
  at efficiency rather than safety; worth a closer read since it's the
  closest existing system to this project's "agent picks, doesn't write"
  stance, but for a different reason (compute cost, not leakage safety).

### 3. Automated ML-pipeline leakage detection — narrower, but a direct technical comparison

- **[LeakageDetector: An Open Source Data Leakage Analysis Tool in
  Machine Learning Pipelines](https://www.alphaxiv.org/abs/2503.14723)**
  (Stevens Institute, 2025) is the most directly comparable *technique*
  found, despite not being agentic at all: a PyCharm plugin doing
  **static code analysis** (built on an earlier command-line tool by Yang
  et al.) that flags three leakage patterns — overlap leakage (train/test
  row overlap), multi-test leakage (reusing a test set across repeated
  evaluations), preprocessing leakage (fitting a transform before the
  split) — and offers *skeletal* quick-fixes a human must complete. In
  their own preliminary user study, "preprocessing leakage" was the most
  common finding (55.6% of cases) — the same failure category this
  project's label-permutation gate (§Step 5, layer 7) targets.
  **The core architectural difference is the mechanism, not the target
  bugs**: LeakageDetector reads *source code* pre-execution and flags a
  pattern for a human to fix; this project's two leakage gates run
  *empirically*, post-fit, against the actual data (shuffled-label
  refit; raw feature-target correlation) and **block promotion
  automatically** rather than surfacing a TODO comment. A pattern-based
  static check can miss leakage that only manifests through actual
  values (e.g. a near-duplicate column with an innocuous name); an
  empirical gate can miss a leakage pattern that happens not to move the
  metric on this particular split. Genuinely complementary detection
  strategies, not competing claims — worth stating that explicitly if
  cited.
- **[Enhancing Automated Machine Learning via Homogeneous Train-Test
  Splitting Methods](https://www.alphaxiv.org/abs/2607.26625)** (2026,
  title/abstract only) — appears to address split-correctness within
  classical AutoML directly; worth a closer read given §Step 4's own
  five-strategy split logic is a similar concern from a different angle.

### 4. Classical (non-LLM) AutoML — background, not a competitor on this axis

- **[Auto-Sklearn 2.0](https://www.alphaxiv.org/abs/2007.04074)**
  (Freiburg et al., 2020/2022) and **[AutoGluon-
  Tabular](https://www.alphaxiv.org/abs/2003.06505)** (2020) are the two
  standard reference points for search-based AutoML: Bayesian
  optimization / meta-learning / stacked ensembling over a large
  hyperparameter space, no LLM, no natural-language interface, and (as
  far as this pass found) no automated leakage detection built in —
  correctness of the split/feature scope is the caller's responsibility,
  same as it is for a hand-written sklearn script. These are not a
  competitor to this project's safety architecture; they're the honest
  baseline for "how much better is the *search*" if that comparison is
  ever run — worth being upfront that this project's 6-template catalog,
  chosen by an LLM across N candidates, is not attempting to compete with
  a Bayesian-optimization search over hundreds of configurations, and a
  reviewer familiar with this space will ask that question directly.


---

## What's *not* claimed here

This document only covers what's implemented and tested (87+ tests,
`tests/`), plus the preliminary literature pass above. It does not
substitute for a real systematic review — the search above covered
roughly a dozen queries against one index (arXiv/alphaXiv) on one day: no
ACM/IEEE digital library pass, no citation-graph traversal, no attempt to
find the *strongest* prior art rather than the first plausible matches.
The scope of the system itself is also deliberately narrow (single-machine,
CPU, tabular, classification only, ≤20 classes), and the two design choices
most likely to be worth a literature comparison are:

1. The **template-catalog middle ground** for the modeling agent (§Step 5)
   — neither free-form code generation nor a single fixed pipeline.
2. The **asymmetric one-way-ratchet verification agent** (§Step 6) — an LLM
   reviewer that is structurally capable of only making an outcome more
   conservative, never less, with a formal guarantee (no code path exists,
   not just a policy) that it cannot be shown or unblock a gate-failure.

Full design rationale and worked failure-mode examples for every step above
are in [`PROJECT_OVERVIEW.md`](../PROJECT_OVERVIEW.md); this document is the
condensed, constraint-focused companion to it.
