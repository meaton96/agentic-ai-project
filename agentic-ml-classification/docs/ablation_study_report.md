# Ablation Study Report

*Leave-one-out testing on harness rules, per
[`harness_pseudocode.md`](harness_pseudocode.md). Code:
[`scripts/run_ablation_study.py`](../scripts/run_ablation_study.py),
[`src/agentic_ml/ablation.py`](../src/agentic_ml/ablation.py). Tests:
[`tests/test_verification.py`](../tests/test_verification.py),
[`tests/test_ablation_scenarios.py`](../tests/test_ablation_scenarios.py),
[`tests/test_feature_engineering.py`](../tests/test_feature_engineering.py),
[`tests/test_intake.py`](../tests/test_intake.py),
[`tests/test_harness.py`](../tests/test_harness.py),
[`tests/test_dynamic_orchestrator.py`](../tests/test_dynamic_orchestrator.py).*

- **Phase 1** — the two original candidate-scoped leakage gates in
  `steps/modeling_step.py` (`label_permutation_test`,
  `check_suspicious_feature_correlation`).
- **Phase 1b** — the five structural checks in
  `harness/feature_engineering.py::validate_feature_proposal`.
- **Phase 1c** — the six checks in
  `harness/intake.py::validate_dataset_spec_proposal`.
- **Phase 1d** — the structural checks in
  `steps/modeling_step.py::run_modeling_step` and `harness/sandbox.py`
  (shape, columns, template/config, the static AST check, and whether a
  sandbox build failure is heeded).
- **Phase 1e** — `harness/splits.py`: `make_split`'s own 3 checks plus
  `resolve_split_columns`' auto-fill reconciliation.
- **Phase 1f** — the four checks
  `harness/leakage.py::run_all_split_leakage_checks` actually calls.
- **Phase 1g** — Dynamic Planner, Verification, and Finalize
  (`orchestrator/dynamic_loop.py`): the registry check, the precondition
  check (which turns out to *be* the Finalize one-shot guard), and a
  real, previously-shipped gap in Verification's candidate-targeting
  this study found and fixed — not a hypothetical ablation.
- **Phase 2** — closing the gap Phase 1 found: a third leakage gate
  (`train_cv_consistency_check`), including a real false-positive it
  introduced and the redesign that fixed it.

**Study complete: 30 of 30 identified rules ablation-tested.** For every
harness rule alongside its pseudocode in one place, see
[`rule_effects.md`](rule_effects.md).

---

## Phase 1 — candidate-scoped leakage gates

`AblationConfig` adds an opt-in flag per gate to `run_modeling_step`,
defaulting to current production behavior — zero effect on any existing
call site. Three synthetic fault scenarios, each run through every
ablation combination:

| Scenario | The fault |
|---|---|
| 1 — content leak | a raw feature column that's a near-duplicate of the target |
| 2 — encoder leak | a target encoder whose `fit()` ignores the labels it's given and always recomputes from the full train+val dataset |
| 3 — bypass leak | a classifier whose `predict_proba()` answers from a closed-over reference to the true labels, regardless of what `fit()` was called with |

**Scenario 1 — content leak**

| ablation | accepted | permutation | correlation |
|---|---|---|---|
| both gates on | **rejected** | pass | **fail** |
| correlation off | **accepted** | pass | pass |

Correlation is necessary and sufficient; permutation contributes nothing.

**Scenario 2 — encoder leak**

| ablation | accepted | permutation | correlation |
|---|---|---|---|
| both gates on | **accepted** | pass | pass |

Neither gate catches this fault, at any setting. Real validation
ROC-AUC: 0.76 (vs. ~0.5 expected). Not a fixture bug: correlation is
structurally blind to non-numeric columns (`pd.to_numeric` coerces the
whole column to `NaN`), and permutation refits the *entire* pipeline —
the downstream classifier legitimately relearns from whatever labels it
receives each shuffle, washing the leak out. Traced directly: mean
shuffled-label AUC = `1 − real_AUC` (0.24 vs 0.76), a coefficient-sign
coin flip, not a reliable signal in either direction. **This is the gap
Phase 2 addresses.**

**Scenario 3 — bypass leak**

| ablation | accepted | permutation | correlation |
|---|---|---|---|
| both gates on | **rejected** | **fail** | pass |
| permutation off | **accepted** | pass | pass |

Permutation is necessary and sufficient; correlation contributes nothing.

**Summary**

| | correlation | permutation |
|---|---|---|
| Scenario 1 (raw duplicate column) | **catches** | misses |
| Scenario 2 (encoder ignores its fold) | misses | misses |
| Scenario 3 (pipeline bypasses fit's y) | misses | **catches** |

The complementary-gates claim holds for scenarios 1 and 3 — what the
project's existing regression test was built to prove. Scenario 2 is
new: a fault that clears both gates at every setting, including
production defaults.

---

## Phase 1b — Feature Engineering rule ablation

Same leave-one-out treatment applied to the five structural checks in
`validate_feature_proposal` (`docs/harness_pseudocode.md`'s Feature
Engineering rules). For each rule: does disabling it let a bad proposal
through, and — the more useful question — what *actually* happens if
that proposal is then applied for real, not just "rejected vs. accepted."

| Rule | Disabled → validation | What actually happens when applied |
|---|---|---|
| `op_id` in catalog | accepts an invented op | **crashes**: `KeyError: "Unknown feature op 'not_a_real_op'"` |
| input ≠ target_column | accepts `ratio(target, amount)` | applies cleanly — new column correlated **0.836** with target |
| numeric dtype for numeric ops | accepts `ratio` on a string column | **crashes**: `TypeError: operation 'truediv' not supported for dtype 'str'` |
| datetime-like for `datetime_parts` | accepts `datetime_parts` on a numeric column | **no crash** — `pd.to_datetime` silently coerces to epoch, producing `1970` for every row |
| group/time column protected from drop | accepts dropping the group column | drops cleanly, then **breaks downstream**: `make_split(strategy="group")` raises `KeyError: 'customer_id'` |

Three distinct failure shapes, not one: a clean crash (2 rules), silent
garbage with no crash at all (1 rule — arguably the worst outcome, since
nothing signals anything went wrong), and a deferred crash one step
later in the pipeline (1 rule). None of these degrade gracefully without
the check; "reject" is not a redundant formality anywhere in this table.

**The target-column check is not backstopped by the modeling-level
correlation gate.** `target / amount` produces correlation 0.836 —
comfortably under the `0.98` threshold gate 8 uses (§Step 5). If a future
modeling candidate selected this derived column, it would pass the
downstream gate too. The two checks are not redundant safety margin for
each other; each is load-bearing on its own.

Reproduce: `python scripts/run_ablation_study.py` (writes
`runs/ablation/feature_engineering_phase1b.json`).

---

## Phase 1c — Intake rule ablation

The rule inventory undercounted this step: `docs/harness_pseudocode.md`
originally condensed intake to 2 checks. Reading
`harness/intake.py::validate_dataset_spec_proposal` closely for this
phase found 6 distinct checks, each independently ablated — and the
results are the most varied of any step tested so far, ranging from a
severe correctness bug to a near-total no-op.

| Rule disabled | What actually happens |
|---|---|
| target existence | **crashes inside the validation function itself** (`KeyError` from `df[target]`) |
| cardinality, lower bound (n=1) | accepted at intake, but `check_fold_class_presence` (a *different*, split-level check) independently catches it |
| cardinality, upper bound (n=30) | accepted, no crash at split time — but downstream multiclass metric computation grows measurably more fragile to class/fold mismatches as cardinality rises relative to fold size |
| group/time existence | **crashes** — `KeyError` inside `make_split` |
| group/time == target_column | **most severe finding in this study**: accepted, then group-based splitting assigns entire classes to entire folds — the tested fixture's **training fold ends up with zero rows**. `check_group_overlap` reports nothing wrong (there is genuinely no group overlap); only `check_fold_class_presence` catches it |
| `id_columns` type (must be a list) | a string gets unpacked character-by-character (Python string iteration); a real feature column silently vanishes if its name matches one of the characters — no error anywhere |
| `id_columns` existence / target-collision | **near-total no-op** — `LoadedDataset.X`'s own defensive `if c in df.columns` filter already absorbs both faults; only the clean error message is lost |

**The group/time-vs-target collision check is the highest-consequence
rule found by this study so far.** Its failure mode isn't a crash — it's
a *quietly empty training set*, the kind of bug a less-attentive caller
could genuinely miss (a run would still "succeed," just against zero
training examples). Verified directly: `make_split(strategy="group",
group_column="target")` on a clean 200-row binary fixture produces a
training fold of length 0.

**Not every rule is equally load-bearing, and that's worth reporting
too.** The `id_columns` existence/target-collision check turned out to
be almost pure redundancy with a defensive filter one layer down in
`LoadedDataset.X`. This study isn't only useful for finding gaps — it
also distinguishes checks doing real work from checks that are mostly
about a clean error message.

Reproduce: `python scripts/run_ablation_study.py` (writes
`runs/ablation/intake_phase1c.json`). Regression tests for the three
most significant findings (empty train fold, single-class-target
backstop, silent feature drop) are in `tests/test_intake.py`.

---

## Phase 1d — Modeling Candidate structural checks

Same undercounting pattern as Phase 1c: the rule inventory listed 6
checks (shape, columns, template_id, config, AST, sandbox); the real
code has 5 distinct `AblationConfig` flags, since `template_id` and
`config` validation share one code path (`validate_config` /
`get_template`). This phase is the first in the study to test
containment of *actively adversarial* candidate code, not just
statistically leaky proposals — and finds the same asymmetric-backstop
pattern Phase 1c found in Intake, in a completely different step.

| Rule disabled | What actually happens |
|---|---|
| candidate shape | **crashes**: `KeyError: 'template_id'` |
| column check — target selected as a feature | **caught downstream** by the feature-correlation gate (corr = 1.0) |
| column check — likely-ID column selected as a feature | **accepted, no errors** — nothing downstream has any signal to catch an uncorrelated ID column |
| template/config — unknown `template_id` | **crashes**: `KeyError` from `get_template` |
| template/config — missing required config key | **backstopped** — the sandbox's own exception handling converts the resulting `KeyError` from inside `build_pipeline()` into a normal rejection |
| static AST check | a template importing `os` **genuinely executes** inside the sandboxed subprocess |
| sandbox build-error check | **crashes**: `AttributeError: 'NoneType' object has no attribute 'fit'` |

**The column-check asymmetry.** A raw, direct leak (the target column
itself, selected as a feature) is backstopped by the correlation gate —
same mechanism as always, correlation exactly 1.0. A likely-ID column
(unique per row, no real relationship to the target) is not backstopped
by anything: it's accepted cleanly, with zero errors, because there's
genuinely no statistical signal for any of the three leakage gates to
find. This is structurally the same shape of finding as Intake's
group/time-vs-target case — a check whose removal produces a
*silent* acceptance, not a crash, is more dangerous than one that
crashes, because nothing downstream will ever flag it.

**The AST-check finding, verified safely.** The fault template used
here does nothing destructive — it imports `os`, then raises an
exception reporting `os.getcwd()`, proving the import executed without
ever reading or writing anything outside its own sandboxed subprocess
tempdir:

```
static check ON:  static check failed: forbidden import: os
static check OFF: candidate build_pipeline() raised: RuntimeError:
                   forbidden import executed; os.getcwd()=/tmp/tmp77c8gt_c
```

This is the first result in the whole study that isn't about leakage at
all — it's evidence the harness's two containment mechanisms (a
pre-execution static check, and post-execution statistical leakage
gates) are solving genuinely different problems, and neither is a
substitute for the other.

Reproduce: `python scripts/run_ablation_study.py` (writes
`runs/ablation/modeling_structural_phase1d.json`). Regression tests for
the three most significant findings (ID-column gap, AST bypass, ignored
build-error crash) are in `tests/test_verification.py`.

---

## Phase 1e — Split Resolution

Another undercount: the inventory listed 2 rules; the real code splits
into 4 across two functions — `resolve_split_columns` (an auto-fill
*recovery* mechanism, not a reject gate) and `make_split`'s own 3 real
checks (strategy validity, group required, time required).

| Rule disabled | What actually happens |
|---|---|
| strategy validity | **crashes**, but with a different, less informative exception — falls through every `elif` branch to the guarded `else: raise AssertionError` |
| group_column required | **crashes** with `KeyError: None` — no hint the real problem is a missing `group_column` |
| time_column required | same — `KeyError: None` |
| split-column reconciliation | **no safety cost** — `make_split`'s own check still catches the unresolved column cleanly; only the auto-fill convenience is lost |

**This phase found something the study hadn't seen before: rules whose
removal costs nothing, and rules whose removal doesn't reduce safety but
does degrade diagnosability.** Every prior phase's findings sorted
cleanly into "backstopped" or "not backstopped." Split Resolution
introduces a third category — still crashes, still safe, but the
*wrong* exception type or a confusing message. Concretely: a caller
written to catch `ValueError` specifically (which is what `make_split`
documents itself as raising) would not catch either ablated failure
mode here, since one becomes `AssertionError` and the other `KeyError`.
That's a real, if narrower, risk — not to data correctness, but to
whatever error-handling code sits above this layer.

Reproduce: `python scripts/run_ablation_study.py` (writes
`runs/ablation/split_resolution_phase1e.json`). Regression tests are in
`tests/test_harness.py`.

---

## Phase 1f — Split-Level Leakage Checks

A third undercount, same pattern as Phase 1c and 1e: the inventory
listed 5 rules including a split-level correlation check that turns out
not to exist — `check_suspicious_feature_correlation` is only ever
called from `modeling_step.py`, candidate-scoped, already covered under
Phase 1/2. The real `run_all_split_leakage_checks` calls exactly 4
checks.

| Rule disabled | What actually happens |
|---|---|
| duplicate rows across splits | **accepted, zero cross-check redundancy** |
| group overlap | **accepted, zero cross-check redundancy** (verified with a fixture free of accidental exact-row duplication, which confounded an earlier attempt) |
| time ordering | **accepted, zero cross-check redundancy** |
| fold class presence | **accepted at the split level — backstopped downstream, but by accident, not by design** |

**The fold-class-presence finding is the most interesting result in this
phase.** A single-class validation fold reaching modeling produces
`score = NaN`. The downstream permutation gate still rejects the
resulting candidate, but only because `NaN <= threshold` evaluates to
`False` in Python/NumPy — not because anything was written to detect a
NaN metric specifically. Verified directly:
`label_permutation_test`'s own detail string reports `mean roc_auc on
shuffled labels = nan` and `passed=False`. Real backstop, fragile
mechanism — it depends on NaN happening to reach a `<=` comparison
rather than being coerced to `0` or raising somewhere earlier in the
metric pipeline.

**The other three checks have no such luck — verified, not assumed.**
For each of duplicate-rows, group-overlap, and time-ordering, the test
explicitly checks whether any of the *other* three checks in the same
`run_all_split_leakage_checks` call also failed on the same fault. None
did, in any of the three cases. Combined with Phase 1e's reconciliation
finding (the one rule that costs nothing to disable), the pattern across
Phases 1e and 1f is now consistent: most rules in this system are the
*only* thing standing between a proposal and a real correctness
failure — defense in depth is the exception here, not the rule.

Reproduce: `python scripts/run_ablation_study.py` (writes
`runs/ablation/split_leakage_phase1f.json`). Regression tests are in
`tests/test_harness.py`.

---

## Phase 1g — Dynamic Planner, Verification, and Finalize (closing phase)

The last three sections of the inventory turned out to be one mechanism
under three names, plus the most significant finding in the whole study.

**Finalize has no check of its own.** `steps/finalize_step.py` will
refit and re-evaluate on the test set if called twice — nothing inside
it stops that. The entire one-shot guarantee is Dynamic Planner's
`required_state={"final_test_metrics_present": False}` precondition,
evaluated against the `"finalize"` registry entry specifically. The
assumed 4 rules (Verification 1, Finalize 1, Dynamic Planner 2) are
really 3: a registry check, a precondition check (which *is* the
Finalize guard), and Verification's gate-status check.

| Rule disabled | What actually happens |
|---|---|
| Dynamic Planner registry check | **crashes inside `validate_plan` itself** — `KeyError` from `get_agent()` |
| Dynamic Planner precondition check | **silently permits `finalize` a second time** after the test set was already touched — this is what disabling the Finalize guard actually looks like |
| Verification gate-status check | **a real, previously-shipped gap** (see below) — not a hypothetical ablation |

**The finding.** `execute_agent_step`'s `"verification"` branch resolves
its target via `args.get("candidate_id") or state.best_unverified_candidate_id()`.
The fallback method correctly filters to gate-passing candidates — but a
planner proposal supplying an explicit `candidate_id` in `args` bypasses
that filter entirely, and `ctx.modeling_results` retains every candidate
ever proposed, gate failures included. Nothing downstream re-checked
gate status before this study looked for it.

Verified directly: a fixture with one passing candidate and one
gate-failed (leaky, `feature_correlation_check.passed=False`) candidate,
a proposal with `args={"candidate_id": "<the failed one>"}`, and a
stubbed client that always returns `"approved"`:

```
execute_agent_step(...) → the failed candidate's verification_verdict = "approved"
```

This is a direct violation of `CLAUDE.md` invariant #4 — the
verification agent "must never be shown... one that failed [the
deterministic gates]." Not a subtle edge case: any planner turn (or
malformed/adversarial one) that names a specific `candidate_id` instead
of leaving candidate selection to the default path could trigger it.

**Fixed, not just documented.** `orchestrator/dynamic_loop.py` now
checks `candidate.ok` immediately after resolving `candidate_id` — before
`build_review_bundle` or any LLM call happens:

```python
if not ablation.skip_verification_gate_status_check and not candidate.ok:
    return False, [f"candidate {candidate_id!r} failed its harness gates and cannot be sent to verification"]
```

`AblationConfig.skip_verification_gate_status_check` reproduces the
original bug on demand, verified against the same fixture: `ok=False`
with a clear rejection message when the fix is active; `verification_verdict
== "approved"` when ablated, exactly reproducing the pre-fix behavior.

**This closes the study at 30 of 30 identified rules** — full coverage
of the inventory as currently understood (which revised its own rule
count five separate times along the way; see `rule_effects.md`'s closing
section for what that fraction does and doesn't claim).

Reproduce: `python scripts/run_ablation_study.py` (writes
`runs/ablation/dynamic_planner_phase1g.json`). Regression tests are in
`tests/test_dynamic_orchestrator.py`.

---

## Phase 2 — closing the Scenario 2 gap

**What Scenario 2 implied.** `label_permutation_test`, as implemented,
only reliably catches a leak when the pipeline's output is independent
of `y_train` at *every* stage. A leak confined to one upstream component
(an encoder) gets "corrected" by a downstream estimator that's honestly
refit on whatever labels it receives — broader than the gate's own
docstring language ("a preprocessing step fit outside its proper fold")
suggests.

**New gate: `train_cv_consistency_check`** (`harness/leakage.py`),
wired into `modeling_step.py` as a third gate alongside the original
two, with its own `AblationConfig.skip_train_cv_consistency_gate` flag.
Compares an in-sample score (fit and score on the same training data)
against a held-out score (a single internal split within the training
fold) — the standard statistical signature of a component that overfits
its own training data, which is exactly where Scenario 2's leak lives
and label-permutation refitting never looks.

**v1 broke real candidates.** The first version compared the
train-vs-holdout gap to a fixed absolute threshold. It correctly caught
the synthetic leak — and incorrectly rejected a legitimate candidate in
`tests/test_orchestrator.py` using the `imbalanced_binary_boosted`
(gradient-boosted) template: in-sample AUC 0.9996 vs. held-out 0.4637,
gap 0.54, purely from ordinary high-capacity-model overfitting on a
small stub dataset, no leak involved. A tree-based model's normal
overfitting gap can be *larger* than a genuine leak's gap, so an
absolute threshold can't tell them apart. This broke 4 tests across
`test_orchestrator.py`, `test_dynamic_orchestrator.py`, and
`test_streaming_monitor.py`.

**v2 fix: compare against a shuffled-label baseline**, the same trick
`label_permutation_test` already uses one level up. Compute the same
train-vs-holdout gap under one round of label permutation; a model's
capacity-driven overfitting shows up similarly whether it's fitting
real or shuffled labels, so that part isn't leakage. What's left after
subtracting — the *excess* gap — is the in-sample bias specifically tied
to the real labels. Default `excess_gap_tolerance=0.15`.

Verified directly against the exact case that broke:

```
real train-vs-holdout gap=0.5628; shuffled-label baseline gap=0.4484
(excess gap=0.1144, tolerance=0.15)  →  passed
```

And against a deliberately extreme high-capacity, non-leaky control (an
unregularized `DecisionTreeClassifier` memorizing pure noise — real gap
~0.50, excess gap ~0.05) — passes. Both are locked in as regression
tests in `tests/test_ablation_scenarios.py`.

**v2 also had to stay cheap.** An earlier full-n-fold-CV version of the
same idea (5 folds × 2 label conditions = 18 refits per candidate) made
the test suite time out. The shipped version uses a single internal
holdout split instead of k-fold (4 refits total — comparable to
`label_permutation_test`'s own 5).

**Does the new gate close Scenario 2?** Only partially, and the reason
why is itself worth reporting. The original Scenario 2 fixture reads a
*closed-over* reference to the true labels — bypassing `fit()`'s actual
`(X, y)` arguments entirely — which defeats every refit-based test
equally (permutation, k-fold CV, this new holdout check), since every
fold or shuffle is "poisoned" identically. That specific bug class turns
out to be **structurally impossible for any real template in this
system**: a template's `build_pipeline(config)` never receives data
(§Step 5, layer 5), so it has no dataset reference to close over in the
first place. A follow-up fixture built a more realistic version —
properly scoped to its actual `fit(X, y)` arguments, just not internally
cross-fitted — and empirically it did **not** meaningfully inflate
validation-fold performance (real val AUC 0.537, barely above chance);
the bias stayed confined to training-set self-referential optimism,
which this project's gates don't score against at all. The residual risk
the new gate guards against is real (a component *could* be scoped
correctly but still exhibit an excess in-sample gap for other reasons)
but narrower than the original finding suggested.

**Test suite status: green.** `pytest tests/ -q` — **227 passed**, 0
failed, 51s (all of Phase 1's `test_ablation_scenarios.py` +
Phase 2's two new regression tests included). An earlier attempt at
this same run took 18 minutes under heavy, unrelated CPU contention
from another process on the shared machine — confirmed not a property
of this change once that contention cleared.

---

## Reproduction

```bash
python scripts/run_ablation_study.py
python -m pytest tests/test_ablation_scenarios.py tests/test_feature_engineering.py tests/test_verification.py tests/test_intake.py tests/test_harness.py tests/test_dynamic_orchestrator.py -v
python -m pytest tests/ -q   # full suite — confirm green before considering this done
```

**Current status: 241/241 tests pass** (`pytest tests/ -q`, 76s). Study
complete — 30/30 identified rules ablation-tested, one real bug found
and fixed.
