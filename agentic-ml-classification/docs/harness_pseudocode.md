# Harness Rules as Pseudocode

Companion to [`harness_constraints.md`](harness_constraints.md) — same
rules, condensed.

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
if unique(target_column) < 2 or > MAX_CLASSES
    reject
else
    task ← binary if 2 classes else multiclass     (agent's task field unused)

if id_column or group_column or time_column not in dataset
    reject
```

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

## Split-Level Leakage Checks

```
fail if duplicate rows appear across splits
fail if a group id appears in more than one split
fail if time ordering is violated
fail if a class is missing from any fold
fail if |corr(feature, target)| >= CORR_THRESHOLD
else pass                              (all five run, independently)
```

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

## Candidate Leakage Gates

```
fail if mean(score on PERM_N label-shuffles) > 0.5 + TOLERANCE
fail if any |corr(selected_column, target)| >= CORR_THRESHOLD
else pass                              (both required, neither is redundant)
```

## Verification

```
if LLM output is unparseable
    verdict ← flagged
else
    verdict ← approved | flagged | rejected

verdict can only downgrade a passing candidate, never upgrade a failed one
    (only candidates that passed every gate above ever reach this step)
```

## Finalize

```
if test set already evaluated this run
    error
else
    refit on train+val
    evaluate once on test
```

## Dynamic Planner (dynamic orchestrator only)

```
if agent_id not in registry
    reject
if a required_state precondition is not met
    reject
else
    accept
```
