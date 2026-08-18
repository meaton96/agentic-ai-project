# Harness Rules as Pseudocode

Companion to [`harness_constraints.md`](harness_constraints.md) — same
rules, condensed. For the rules combined with their measured
ablation-study effects (which ones have been leave-one-out tested, and
what actually happens when each is disabled), see
[`rule_effects.md`](rule_effects.md).

## Constants

```
MAX_CLASSES        = 20
STRATEGIES         = {random, stratified, group, time, group_time}
FEATURE_OPS        = {ratio, interaction, log1p, datetime_parts, missing_indicator}
TEMPLATES          = {logistic_numeric, sklearn_mixed_pipeline, lightgbm_mixed,
                       xgboost_mixed, imbalanced_binary_boosted,
                       high_cardinality_target_encoding}
FORBIDDEN_IMPORTS  = {os, sys, subprocess, socket, shutil, requests, urllib,
                       http, ftplib, telnetlib, smtplib, ctypes,
                       multiprocessing, pty, pickle}
FORBIDDEN_CALLS    = {eval, exec, __import__, compile, open, input}
CORR_THRESHOLD     = 0.98
PERM_N, TOLERANCE  = 5, 0.08          (chance baseline = 0.5)
SANDBOX_TIMEOUT    = 30s wall-clock, 20s CPU
CV_HOLDOUT_FRAC    = 0.3, EXCESS_GAP_TOLERANCE = 0.15   (added after ablation study, see below)
```

## Pipeline

```
spec           ← Intake(D)
D, spec        ← FeatureEngineering(D, spec)
strategy       ← ResolveSplit(profile, spec)
Tr, Va, Te     ← Split(D, spec, strategy)
if SplitLeakageChecks(Tr, Va, Te) fails
    abort
for i in 1..N
    c ← Modeling(profile)
    if ValidateCandidate(c) and LeakageGates(c) pass
        candidates += c
candidates ← sort by val_score, descending
for c in candidates
    verdict ← Verification(c)
    if verdict != rejected
        accepted ← c; break
if no accepted
    abort                              (test set Te never touched)
Finalize(accepted, Tr+Va, Te)          (one and only test-set read)
Summarize(final_metrics)
```

## Intake

```
if target_column not in dataset
    reject
if unique(target_column) < 2 or > MAX_CLASSES
    reject
else
    task ← binary if 2 classes else multiclass     (agent's task field unused)
if group_column or time_column not in dataset
    reject
if group_column or time_column == target_column
    reject
if id_columns is not a list
    reject
if any id_column not in dataset, or == target_column
    reject
```

Six distinct checks, not the two this doc originally condensed them to —
see [`rule_effects.md`](rule_effects.md) for what each one actually does
when disabled, including one severe finding (an empty training fold).

## Feature Engineering

```
if op_id not in FEATURE_OPS
    reject
if column == target_column
    reject
if op is numeric and column dtype is not numeric
    reject
if op == datetime_parts and column is not datetime-like
    reject
if drop_column == group_column or drop_column == time_column
    reject
else
    accept
```

Leave-one-out results for each of these five rules (does disabling it
let a bad proposal through, and what actually happens if it's then
applied — a clean rejection is not the only alternative to a crash) are
in [`ablation_study_report.md`](ablation_study_report.md) Phase 1b.

## Split Resolution

```
if strategy not in STRATEGIES
    error
if strategy needs group_column and none declared
    if a likely group column was detected
        use it
    else
        error
if strategy needs time_column and none declared
    if a likely time column was detected
        use it
    else
        error
```

Split across two functions, not one: the "use it" auto-fill lives in
`resolve_split_columns` (a recovery mechanism), the "error" lives in
`make_split` itself (`VALID_STRATEGIES`, `requires group_column`,
`requires time_column`) — ablating the auto-fill costs nothing (`make_split`
still catches the result cleanly); ablating `make_split`'s own checks
degrades the error, not the safety (an `AssertionError` or a
`KeyError: None` instead of a clear `ValueError`). See
[`rule_effects.md`](rule_effects.md) Phase 1e.

## Split-Level Leakage Checks

```
fail if duplicate rows appear across splits
fail if a group id appears in more than one split
fail if time ordering is violated
fail if a class is missing from any fold
else pass                              (all four run, independently)
```

Four checks, not five — an earlier version of this doc listed a
split-level correlation check that doesn't exist.
`check_suspicious_feature_correlation` is only ever called from
`modeling_step.py`, candidate-scoped (see Candidate Leakage Gates
below); `run_all_split_leakage_checks` never calls it. `group id
appears in more than one split` and `time ordering is violated` each
auto-pass (not fail) when no `group_column`/`time_column` is declared —
that's existing, correct behavior, not something ablated here. See
[`rule_effects.md`](rule_effects.md) Phase 1f for leave-one-out results,
including one check whose removal is only backstopped by an emergent
side effect of NaN comparison semantics, not by design.

## Modeling Candidate

```
if candidate shape is malformed
    reject
if a referenced column is hallucinated, or is target/id/group/time
    reject
if template_id not in TEMPLATES
    reject
if config is missing a required key
    reject
if source imports FORBIDDEN_IMPORTS or calls FORBIDDEN_CALLS
    reject
if sandbox execution times out or raises
    reject
else
    accept unfitted pipeline           (only object allowed to cross back)
```

Leave-one-out results for all six checks — including the AST check, the
first fixture in this whole study to test containment of adversarial
code rather than a statistical leak — are in
[`rule_effects.md`](rule_effects.md) Phase 1d.

## Candidate Leakage Gates

```
fail if mean(score on PERM_N label-shuffles) > 0.5 + TOLERANCE
fail if any |corr(selected_column, target)| >= CORR_THRESHOLD
fail if (train_score - holdout_score) - (shuffled_train_score - shuffled_holdout_score) > EXCESS_GAP_TOLERANCE
else pass                              (all three required, none is redundant)
```

Third rule added by [`ablation_study_report.md`](ablation_study_report.md)
Phase 2: an ablation study found the first two gates both miss a
component that's properly scoped to its `fit(X, y)` arguments but
internally self-referential (not cross-fitted). The excess-over-
shuffled-baseline form exists because a naive absolute train/holdout
gap threshold false-positived on legitimate high-capacity templates
(gradient-boosted trees) — see the report for the exact regression and
fix.

## Verification

```
if LLM output is unparseable
    verdict ← flagged
else
    verdict ← approved | flagged | rejected

verdict can only downgrade a passing candidate, never upgrade a failed one
    (only candidates that passed every gate above ever reach this step)
```

The parenthetical was a claim, not yet an enforced fact, when this line
was first written. The dynamic orchestrator's `execute_agent_step`
selected the default (safe) candidate via `best_unverified_candidate_id()`
— which does filter correctly — but an explicit
`args={"candidate_id": ...}` in a planner proposal bypassed that filter
entirely, and nothing re-checked gate status afterward. Found by this
ablation study, confirmed to let a gate-failed candidate reach the
verification LLM and be approved, and fixed directly — see
[`rule_effects.md`](rule_effects.md) Phase 1g.

## Finalize

```
if test set already evaluated this run
    error
else
    refit on train+val
    evaluate once on test
```

This step has no check of its own. `steps/finalize_step.py` will
happily refit and re-evaluate on the test set if called twice — the
entire one-shot guarantee lives one layer up, in the Dynamic Planner's
precondition check below (`required_state={"final_test_metrics_present":
False}` on the `"finalize"` registry entry). "Finalize" is not an
independently-ablatable rule; it's the same mechanism as Dynamic
Planner's second line, applied to one specific registry entry.

## Dynamic Planner (dynamic orchestrator only)

```
if agent_id not in registry
    reject
if a required_state precondition is not met
    reject
else
    accept
```

The second line **is** the Finalize one-shot guard, the Verification
gate-status filter's registry-level backstop for other agents, and every
other precondition in `orchestrator/agent_registry.py` — one check,
many consequences depending on which registry entry it's evaluated
against. See [`rule_effects.md`](rule_effects.md) Phase 1g for full
results, including the real (not hypothetical) gap this phase found and
fixed.
